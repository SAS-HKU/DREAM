"""5 Hz ROS wrapper for the scaled DRIFT field."""

from __future__ import annotations

import json
from typing import List, Optional

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String

from .core.risk_field import DREAMRiskField, NumericalStabilityError
from .core.types import EgoState, Vehicle
from .limo_scale import default_deployment_config
from .ros_utils import ego_from_odometry, vehicle_from_mapping


class DriftFieldNode(Node):
    def __init__(self) -> None:
        super().__init__("dream_drift_field")
        self.config = default_deployment_config()
        self.field = DREAMRiskField(self.config)
        self.declare_parameter("ego_topic", "/dream/ego_state")
        self.declare_parameter("world_topic", "/dream/world_model")
        self.declare_parameter("mask_topic", "/dream/occlusion_mask")
        self.declare_parameter("update_rate", 5.0)
        self.declare_parameter("input_timeout", 0.5)
        self.declare_parameter("warmup_duration", self.config.pde.warmup_duration)

        self.ego: Optional[EgoState] = None
        self.ego_receipt: Optional[float] = None
        self.vehicles: List[Vehicle] = []
        self.world_receipt: Optional[float] = None
        self.shadow: Optional[np.ndarray] = None
        self.mask_receipt: Optional[float] = None
        self.warmup_done = False

        self.create_subscription(Odometry, str(self.get_parameter("ego_topic").value), self._on_ego, 10)
        self.create_subscription(String, str(self.get_parameter("world_topic").value), self._on_world, 10)
        self.create_subscription(
            OccupancyGrid, str(self.get_parameter("mask_topic").value), self._on_mask, 10
        )
        self.raw_publisher = self.create_publisher(Image, "/dream/risk_field_raw", 2)
        self.grid_publisher = self.create_publisher(OccupancyGrid, "/dream/risk_field", 2)
        self.ready_publisher = self.create_publisher(Bool, "/dream/drift_ready", 10)
        self.status_publisher = self.create_publisher(String, "/dream/drift_status", 10)
        self.create_timer(1.0 / float(self.get_parameter("update_rate").value), self._update)

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1.0e-9

    def _on_ego(self, message: Odometry) -> None:
        self.ego = ego_from_odometry(message)
        self.ego_receipt = self._now()

    def _on_world(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
            if payload.get("frame_id") != self.config.grid.frame_id:
                raise ValueError("world frame does not match risk-grid frame")
            self.vehicles = [vehicle_from_mapping(item) for item in payload.get("vehicles", [])]
            self.world_receipt = self._now()
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            self.get_logger().warning(f"Rejected world model: {exc}")

    def _on_mask(self, message: OccupancyGrid) -> None:
        try:
            if message.header.frame_id != self.config.grid.frame_id:
                raise ValueError("occlusion mask frame mismatch")
            if message.info.width != self.config.grid.nx or message.info.height != self.config.grid.ny:
                raise ValueError("occlusion mask dimensions mismatch")
            mask = np.asarray(message.data, dtype=np.float64).reshape(self.field.shape)
            self.shadow = np.clip(mask / 100.0, 0.0, 1.0)
            self.mask_receipt = self._now()
        except (ValueError, TypeError) as exc:
            self.get_logger().warning(f"Rejected occlusion mask: {exc}")

    def _publish_ready(self, ready: bool) -> None:
        message = Bool()
        message.data = bool(ready)
        self.ready_publisher.publish(message)

    def _publish_field(self) -> None:
        stamp = self.get_clock().now().to_msg()
        raw = Image()
        raw.header.stamp = stamp
        raw.header.frame_id = self.config.grid.frame_id
        raw.height = self.config.grid.ny
        raw.width = self.config.grid.nx
        raw.encoding = "32FC1"
        raw.is_bigendian = False
        raw.step = self.config.grid.nx * 4
        raw.data = np.asarray(self.field.R, dtype=np.float32).tobytes()
        self.raw_publisher.publish(raw)

        grid = OccupancyGrid()
        grid.header.stamp = stamp
        grid.header.frame_id = self.config.grid.frame_id
        grid.info.resolution = self.config.grid.resolution
        grid.info.width = self.config.grid.nx
        grid.info.height = self.config.grid.ny
        grid.info.origin.position.x = self.config.grid.x_min
        grid.info.origin.position.y = self.config.grid.y_min
        grid.info.origin.orientation.w = 1.0
        grid.data = np.rint(
            np.clip(self.field.R / self.config.pde.risk_ceiling, 0.0, 1.0) * 100.0
        ).astype(np.int8).ravel().tolist()
        self.grid_publisher.publish(grid)

    def _update(self) -> None:
        now = self._now()
        timeout = float(self.get_parameter("input_timeout").value)
        fresh = (
            self.ego is not None
            and self.shadow is not None
            and self.ego_receipt is not None
            and self.world_receipt is not None
            and self.mask_receipt is not None
            and now - self.ego_receipt < timeout
            and now - self.world_receipt < timeout
            and now - self.mask_receipt < timeout
        )
        if not fresh:
            self._publish_ready(False)
            status = String()
            status.data = json.dumps({"ready": False, "reason": "STALE_INPUT"}, separators=(",", ":"))
            self.status_publisher.publish(status)
            return
        try:
            if not self.warmup_done:
                duration = float(self.get_parameter("warmup_duration").value)
                self.field.warmup(self.vehicles, self.ego, self.shadow, duration=duration)
                self.warmup_done = True
                now_after_warmup = self._now()
                if any(
                    now_after_warmup - stamp >= timeout
                    for stamp in (self.ego_receipt, self.world_receipt, self.mask_receipt)
                ):
                    self._publish_ready(False)
                    status = String()
                    status.data = json.dumps(
                        {
                            "ready": False,
                            "reason": "STALE_AFTER_WARMUP",
                            "warmup_model_seconds": self.field.elapsed_model_time,
                        },
                        separators=(",", ":"),
                    )
                    self.status_publisher.publish(status)
                    return
            else:
                self.field.step(self.vehicles, self.ego, self.shadow)
        except (NumericalStabilityError, ValueError) as exc:
            self.get_logger().error(f"DRIFT update failed closed: {exc}")
            self._publish_ready(False)
            status = String()
            status.data = json.dumps(
                {"ready": False, "reason": "NUMERICAL_FAILURE", "detail": str(exc)},
                separators=(",", ":"),
            )
            self.status_publisher.publish(status)
            return
        self._publish_field()
        self._publish_ready(True)
        digest = self.field.last_digest
        status = String()
        status.data = json.dumps(
            {
                "ready": True,
                "warmup_model_seconds": self.field.elapsed_model_time,
                "substeps": digest.substeps,
                "compute_seconds": digest.compute_seconds,
                "field_maximum": digest.field_maximum,
                "field_mean": digest.field_mean,
                "raw_minimum": digest.raw_minimum,
                "maximum_diffusion": digest.maximum_diffusion,
            },
            separators=(",", ":"),
        )
        self.status_publisher.publish(status)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DriftFieldNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
