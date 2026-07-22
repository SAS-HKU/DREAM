"""Fail-closed LiDAR collision grid and reference-trajectory monitor.

This node publishes no velocity or arm topic.  Existing SIL/live dry-run
launches do not start it; the dedicated disabled-by-default hardware launch
consumes ``/dream/collision_status`` continuously at the final command gate.
"""

from __future__ import annotations

import json
from math import hypot, isfinite
from typing import Optional

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener

from .core.collision import (
    axis_aligned_road_mask,
    CollisionEnvelope,
    CollisionGridSpec,
    footprint_self_return_mask,
    TrajectoryAssessment,
    transform_points,
)
from .limo_scale import deployment_config_for_arena


class DreamCollisionMonitorNode(Node):
    def __init__(self) -> None:
        super().__init__("dream_collision_monitor")
        self.declare_parameter("arena_file", "")
        self.config = deployment_config_for_arena(
            str(self.get_parameter("arena_file").value)
        )
        self._declare_parameters()
        self.map_frame = self._str_parameter("map_frame")
        self.scan_timeout = self._positive_parameter("scan_timeout")
        self.scan_rejection_grace = self._positive_parameter(
            "scan_rejection_grace"
        )
        if self.scan_rejection_grace > min(self.scan_timeout, 0.20):
            raise ValueError(
                "scan_rejection_grace must not exceed scan_timeout or 0.20 s"
            )
        self.mask_timeout = self._positive_parameter("mask_timeout")
        self.path_timeout = self._positive_parameter("path_timeout")
        self.tf_timeout = self._positive_parameter("tf_timeout")
        self.future_stamp_tolerance = self._nonnegative_parameter(
            "future_stamp_tolerance"
        )
        self.base_frame = self._str_parameter("base_frame")
        self.self_return_filter_enabled = bool(
            self.get_parameter("self_return_filter_enabled").value
        )
        self.self_return_max_range = self._positive_parameter(
            "self_return_max_range"
        )
        self.self_return_footprint_padding = self._nonnegative_parameter(
            "self_return_footprint_padding"
        )
        self.occlusion_shadow_blocks_trajectory = bool(
            self.get_parameter("occlusion_shadow_blocks_trajectory").value
        )

        grid = self.config.grid
        self.spec = CollisionGridSpec(
            width=grid.nx,
            height=grid.ny,
            resolution=grid.resolution,
            origin_x=grid.x_min,
            origin_y=grid.y_min,
            frame_id=self.map_frame,
        )
        traversable = axis_aligned_road_mask(
            self.spec,
            y_min=grid.road_y_min,
            y_max=grid.road_y_max,
        )
        robot_half_diagonal = hypot(
            0.5 * self.config.mpc.robot_length,
            0.5 * self.config.mpc.robot_width,
        )
        inflation_radius = robot_half_diagonal + self._nonnegative_parameter(
            "inflation_margin"
        )
        self.envelope = CollisionEnvelope(
            self.spec,
            surface_retention_seconds=self._positive_parameter(
                "surface_retention_seconds"
            ),
            inflation_radius=inflation_radius,
            minimum_valid_rays=self._positive_int_parameter("minimum_valid_rays"),
            interpolation_spacing=self._positive_parameter(
                "trajectory_interpolation_spacing"
            ),
            traversable_mask=traversable,
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        reliable = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(
            LaserScan,
            self._str_parameter("scan_topic"),
            self._on_scan,
            sensor_qos,
        )
        self.create_subscription(
            OccupancyGrid,
            self._str_parameter("occlusion_mask_topic"),
            self._on_mask,
            reliable,
        )
        self.create_subscription(
            Path,
            self._str_parameter("trajectory_topic"),
            self._on_path,
            reliable,
        )
        self.grid_publisher = self.create_publisher(
            OccupancyGrid, self._str_parameter("collision_grid_topic"), reliable
        )
        self.status_publisher = self.create_publisher(
            String, self._str_parameter("collision_status_topic"), reliable
        )

        self.shadow_unknown: Optional[np.ndarray] = None
        self.path_points: Optional[np.ndarray] = None
        self.last_scan_receipt: Optional[float] = None
        self.last_mask_receipt: Optional[float] = None
        self.last_path_receipt: Optional[float] = None
        self.last_tf_receipt: Optional[float] = None
        self.scan_ok = False
        self.mask_ok = False
        self.path_ok = False
        self.tf_ok = False
        self.scan_error = "WAITING_FOR_SCAN"
        self.mask_error = "WAITING_FOR_OCCLUSION_MASK"
        self.path_error = "WAITING_FOR_REFERENCE_TRAJECTORY"
        self.tf_error = "WAITING_FOR_SCAN_TRANSFORM"
        self.raw_valid_ray_count = 0
        self.valid_ray_count = 0
        self.self_return_rejection_count = 0
        self.transformed_surface_cells = 0
        self.latest_scan_rejection: Optional[str] = None
        self.latest_scan_rejection_receipt: Optional[float] = None
        self.scan_rejection_count = 0
        self.consecutive_scan_rejections = 0
        self.inflation_radius = inflation_radius
        rate = self._positive_parameter("publish_rate")
        self.create_timer(1.0 / rate, self._publish)

    def _declare_parameters(self) -> None:
        grid = self.config.grid
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("occlusion_mask_topic", "/dream/occlusion_mask")
        self.declare_parameter("trajectory_topic", "/dream/reference_trajectory")
        self.declare_parameter("collision_grid_topic", "/dream/collision_grid")
        self.declare_parameter("collision_status_topic", "/dream/collision_status")
        self.declare_parameter("map_frame", grid.frame_id)
        self.declare_parameter("publish_rate", 10.0)
        self.declare_parameter("scan_timeout", 0.40)
        self.declare_parameter("scan_rejection_grace", 0.20)
        self.declare_parameter("mask_timeout", 0.50)
        self.declare_parameter("path_timeout", 0.50)
        self.declare_parameter("tf_timeout", 0.10)
        self.declare_parameter("future_stamp_tolerance", 0.05)
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("self_return_filter_enabled", True)
        self.declare_parameter("self_return_max_range", 0.05)
        self.declare_parameter("self_return_footprint_padding", 0.0)
        self.declare_parameter("surface_retention_seconds", 0.75)
        self.declare_parameter("minimum_valid_rays", 20)
        # Default fail-closed for direct node use.  The reviewed hardware YAML
        # explicitly selects risk-only shadows because DRIFT and its veto own
        # occlusion response; measured LiDAR returns remain hard obstacles.
        self.declare_parameter("occlusion_shadow_blocks_trajectory", True)
        self.declare_parameter(
            "inflation_margin", self.config.safety.collision_inflation_margin
        )
        self.declare_parameter(
            "trajectory_interpolation_spacing", 0.5 * grid.resolution
        )

    def _str_parameter(self, name: str) -> str:
        value = str(self.get_parameter(name).value)
        if not value:
            raise ValueError(f"{name} cannot be empty")
        return value

    def _positive_parameter(self, name: str) -> float:
        value = float(self.get_parameter(name).value)
        if not isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
        return value

    def _nonnegative_parameter(self, name: str) -> float:
        value = float(self.get_parameter(name).value)
        if not isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")
        return value

    def _positive_int_parameter(self, name: str) -> int:
        value = int(self.get_parameter(name).value)
        if value < 1:
            raise ValueError(f"{name} must be positive")
        return value

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1.0e-9

    @staticmethod
    def _stamp_seconds(message_stamp) -> float:
        return float(message_stamp.sec) + 1.0e-9 * float(message_stamp.nanosec)

    def _source_stamp_fresh(self, message_stamp, now: float, timeout: float) -> bool:
        stamp = self._stamp_seconds(message_stamp)
        if stamp <= 0.0:
            return False
        age = now - stamp
        return -self.future_stamp_tolerance <= age < timeout

    def _on_scan(self, message: LaserScan) -> None:
        now = self._now()
        self.raw_valid_ray_count = 0
        self.valid_ray_count = 0
        self.self_return_rejection_count = 0
        self.transformed_surface_cells = 0
        if not message.header.frame_id:
            self._reject_scan(now, "SCAN_FRAME_EMPTY", "SCAN_FRAME_EMPTY")
            return
        if not self._source_stamp_fresh(message.header.stamp, now, self.scan_timeout):
            self._reject_scan(
                now,
                "SCAN_SOURCE_STAMP_STALE",
                "SCAN_SOURCE_STAMP_STALE",
            )
            return
        ranges = np.asarray(message.ranges, dtype=np.float64)
        angles = float(message.angle_min) + np.arange(len(ranges)) * float(
            message.angle_increment
        )
        valid = (
            np.isfinite(ranges)
            & (ranges >= float(message.range_min))
            & (ranges <= float(message.range_max))
        )
        self.raw_valid_ray_count = int(np.count_nonzero(valid))
        self.valid_ray_count = self.raw_valid_ray_count
        if self.raw_valid_ray_count < self.envelope.minimum_valid_rays:
            self.envelope.record_scan(
                np.empty((0, 2)),
                receipt_time=now,
                valid_ray_count=self.valid_ray_count,
            )
            self._reject_scan(
                now,
                "INSUFFICIENT_VALID_RAYS",
                "TRANSFORM_NOT_EVALUATED",
            )
            return
        local_points = np.column_stack(
            (
                ranges[valid] * np.cos(angles[valid]),
                ranges[valid] * np.sin(angles[valid]),
                np.zeros(self.raw_valid_ray_count, dtype=np.float64),
            )
        )
        valid_ranges = ranges[valid]
        try:
            # Exact sensor timestamp is required; Time() / latest-TF lookup is
            # intentionally never used by this physical collision layer.
            stamp = Time.from_msg(message.header.stamp)
            map_transform = self.tf_buffer.lookup_transform(
                self.map_frame,
                message.header.frame_id,
                stamp,
                timeout=Duration(seconds=self.tf_timeout),
            )
            translation = map_transform.transform.translation
            rotation = map_transform.transform.rotation
            points = transform_points(
                local_points,
                translation_xyz=(translation.x, translation.y, translation.z),
                quaternion_xyzw=(rotation.x, rotation.y, rotation.z, rotation.w),
            )
            if self.self_return_filter_enabled:
                base_transform = self.tf_buffer.lookup_transform(
                    self.base_frame,
                    message.header.frame_id,
                    stamp,
                    timeout=Duration(seconds=self.tf_timeout),
                )
                base_translation = base_transform.transform.translation
                base_rotation = base_transform.transform.rotation
                points_in_base = transform_points(
                    local_points,
                    translation_xyz=(
                        base_translation.x,
                        base_translation.y,
                        base_translation.z,
                    ),
                    quaternion_xyzw=(
                        base_rotation.x,
                        base_rotation.y,
                        base_rotation.z,
                        base_rotation.w,
                    ),
                )
                rejected = footprint_self_return_mask(
                    points_in_base,
                    valid_ranges,
                    maximum_self_return_range=self.self_return_max_range,
                    footprint_length=self.config.mpc.robot_length,
                    footprint_width=self.config.mpc.robot_width,
                    footprint_padding=self.self_return_footprint_padding,
                )
                self.self_return_rejection_count = int(
                    np.count_nonzero(rejected)
                )
                points = points[~rejected]
                self.valid_ray_count = int(points.shape[0])
        except (TransformException, ValueError) as exc:
            self._reject_scan(
                now,
                "SCAN_TF_FAILURE",
                f"SCAN_TF_FAILURE:{exc}",
            )
            self.get_logger().warning(
                f"Collision monitor rejected scan transform: {exc}",
                throttle_duration_sec=2.0,
            )
            return
        self.last_tf_receipt = now
        self.tf_ok = True
        self.tf_error = "ok"
        if self.valid_ray_count < self.envelope.minimum_valid_rays:
            self.envelope.record_scan(
                np.empty((0, 2)),
                receipt_time=now,
                valid_ray_count=self.valid_ray_count,
            )
            self._reject_scan(
                now, "INSUFFICIENT_VALID_RAYS_AFTER_SELF_FILTER"
            )
            return
        self.transformed_surface_cells = self.envelope.record_scan(
            points,
            receipt_time=now,
            valid_ray_count=self.valid_ray_count,
        )
        self.scan_ok = self.envelope.last_scan_accepted
        self.scan_error = "ok" if self.scan_ok else "INSUFFICIENT_VALID_RAYS"
        if self.scan_ok:
            self.last_scan_receipt = now
            self.consecutive_scan_rejections = 0

    def _reject_scan(
        self,
        now: float,
        scan_error: str,
        tf_error: Optional[str] = None,
    ) -> None:
        """Retain one recent exact-TF scan through a brief bad sample.

        The final hardware gate independently watches raw ``/scan`` freshness.
        This latch therefore covers only collision-node scheduling or one
        exact-time TF miss; it cannot keep motion authorized after the raw
        sensor itself exceeds that independent timeout.
        """

        self.latest_scan_rejection = str(scan_error)
        self.latest_scan_rejection_receipt = float(now)
        self.scan_rejection_count += 1
        self.consecutive_scan_rejections += 1
        retain_last_good = (
            self.consecutive_scan_rejections == 1
            and self._fresh(
                now,
                self.last_scan_receipt,
                self.scan_rejection_grace,
            )
        )
        if not retain_last_good:
            self.scan_ok = False
            self.scan_error = str(scan_error)
        if tf_error is not None and not retain_last_good:
            self.tf_ok = False
            self.tf_error = str(tf_error)

    def _grid_metadata_valid(self, message: OccupancyGrid) -> bool:
        info = message.info
        origin = info.origin
        return (
            message.header.frame_id == self.map_frame
            and int(info.width) == self.spec.width
            and int(info.height) == self.spec.height
            and abs(float(info.resolution) - self.spec.resolution) <= 1.0e-9
            and abs(float(origin.position.x) - self.spec.origin_x) <= 1.0e-9
            and abs(float(origin.position.y) - self.spec.origin_y) <= 1.0e-9
            and abs(float(origin.orientation.x)) <= 1.0e-6
            and abs(float(origin.orientation.y)) <= 1.0e-6
            and abs(float(origin.orientation.z)) <= 1.0e-6
            and abs(float(origin.orientation.w) - 1.0) <= 1.0e-6
        )

    def _on_mask(self, message: OccupancyGrid) -> None:
        now = self._now()
        self.last_mask_receipt = now
        self.mask_ok = False
        if not self._source_stamp_fresh(message.header.stamp, now, self.mask_timeout):
            self.mask_error = "MASK_SOURCE_STAMP_STALE"
            return
        if not self._grid_metadata_valid(message):
            self.mask_error = "MASK_GRID_MISMATCH"
            return
        values = np.asarray(message.data, dtype=np.int16)
        if values.size != self.spec.width * self.spec.height:
            self.mask_error = "MASK_PAYLOAD_SIZE_MISMATCH"
            return
        # Both explicit OccupancyGrid unknown (-1) and DREAM shadow (>0) become
        # UNKNOWN in the diagnostic grid.  The configured policy determines
        # whether UNKNOWN blocks the trajectory; it never changes surfaces.
        self.shadow_unknown = (values.reshape(self.spec.shape) != 0)
        self.mask_ok = True
        self.mask_error = "ok"

    def _on_path(self, message: Path) -> None:
        now = self._now()
        self.last_path_receipt = now
        self.path_ok = False
        if not self._source_stamp_fresh(message.header.stamp, now, self.path_timeout):
            self.path_error = "PATH_SOURCE_STAMP_STALE"
            return
        if message.header.frame_id != self.map_frame:
            self.path_error = "PATH_FRAME_MISMATCH"
            return
        points = np.asarray(
            [[pose.pose.position.x, pose.pose.position.y] for pose in message.poses],
            dtype=np.float64,
        )
        if len(points) < 2 or points.shape != (len(points), 2):
            self.path_error = "PATH_TOO_SHORT"
            return
        if not np.all(np.isfinite(points)):
            self.path_error = "PATH_NONFINITE"
            return
        self.path_points = points
        self.path_ok = True
        self.path_error = "ok"

    @staticmethod
    def _age(now: float, receipt: Optional[float]) -> Optional[float]:
        return None if receipt is None else max(0.0, now - receipt)

    def _fresh(self, now: float, receipt: Optional[float], timeout: float) -> bool:
        return receipt is not None and 0.0 <= now - receipt < timeout

    def _scan_evidence_timeout(self) -> float:
        """Use the short last-good limit after any rejected scan callback."""

        if self.consecutive_scan_rejections > 0:
            return self.scan_rejection_grace
        return self.scan_timeout

    def _readiness_reason(self, now: float) -> tuple[bool, str]:
        scan_evidence_timeout = self._scan_evidence_timeout()
        if not self.scan_ok:
            return False, self.scan_error
        if not self._fresh(now, self.last_scan_receipt, scan_evidence_timeout):
            return False, "SCAN_STALE"
        if not self.tf_ok:
            return False, self.tf_error
        if not self._fresh(now, self.last_tf_receipt, scan_evidence_timeout):
            return False, "SCAN_TF_STALE"
        if not self.mask_ok or self.shadow_unknown is None:
            return False, self.mask_error
        if not self._fresh(now, self.last_mask_receipt, self.mask_timeout):
            return False, "MASK_STALE"
        if not self.path_ok or self.path_points is None:
            return False, self.path_error
        if not self._fresh(now, self.last_path_receipt, self.path_timeout):
            return False, "PATH_STALE"
        return True, "INPUTS_READY"

    def _publish_grid(self, grid: np.ndarray) -> None:
        message = OccupancyGrid()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self.map_frame
        message.info.resolution = self.spec.resolution
        message.info.width = self.spec.width
        message.info.height = self.spec.height
        message.info.origin.position.x = self.spec.origin_x
        message.info.origin.position.y = self.spec.origin_y
        message.info.origin.orientation.w = 1.0
        message.data = np.asarray(grid, dtype=np.int8).ravel().tolist()
        self.grid_publisher.publish(message)

    def _publish(self) -> None:
        now = self._now()
        ready, reason = self._readiness_reason(now)
        scan_evidence_timeout = self._scan_evidence_timeout()
        # A stale/missing perception input makes the full grid unknown.  The
        # retained surface grid remains useful diagnostically but can never be
        # mistaken for a complete collision map.
        perception_ready = (
            self.scan_ok
            and self.tf_ok
            and self._fresh(now, self.last_scan_receipt, scan_evidence_timeout)
            and self._fresh(now, self.last_tf_receipt, scan_evidence_timeout)
            and self.mask_ok
            and self._fresh(now, self.last_mask_receipt, self.mask_timeout)
            and self.shadow_unknown is not None
        )
        shadow = (
            self.shadow_unknown
            if perception_ready
            else np.ones(self.spec.shape, dtype=bool)
        )
        assessment = TrajectoryAssessment(False, reason, 0)
        trajectory_shadow_overlap_samples = None
        try:
            grid, digest = self.envelope.render(shadow, now=now)
            if ready:
                trajectory_shadow_overlap_samples = (
                    self.envelope.trajectory_mask_overlap_samples(
                        self.path_points, self.shadow_unknown
                    )
                )
                assessment = self.envelope.assess_trajectory(
                    self.path_points,
                    grid,
                    unknown_is_collision=self.occlusion_shadow_blocks_trajectory,
                )
                reason = assessment.reason
        except ValueError as exc:
            ready = False
            reason = f"COLLISION_MONITOR_ERROR:{exc}"
            grid = np.full(
                self.spec.shape, CollisionEnvelope.UNKNOWN, dtype=np.int8
            )
            digest = None
            assessment = TrajectoryAssessment(False, reason, 0)
        trajectory_clear = bool(ready and assessment.clear)
        self._publish_grid(grid)
        payload = {
            "stamp": now,
            "ready": bool(ready),
            "trajectory_clear": trajectory_clear,
            "reason": reason,
            "frame_id": self.map_frame,
            "shadow_policy": (
                "hard_collision"
                if self.occlusion_shadow_blocks_trajectory
                else "risk_only"
            ),
            "occlusion_shadow_blocks_trajectory": (
                self.occlusion_shadow_blocks_trajectory
            ),
            "scan_fresh": self._fresh(
                now, self.last_scan_receipt, scan_evidence_timeout
            ),
            "mask_fresh": self._fresh(now, self.last_mask_receipt, self.mask_timeout),
            "path_fresh": self._fresh(now, self.last_path_receipt, self.path_timeout),
            "tf_fresh": self._fresh(
                now, self.last_tf_receipt, scan_evidence_timeout
            ),
            "scan_error": self.scan_error,
            "mask_error": self.mask_error,
            "path_error": self.path_error,
            "tf_error": self.tf_error,
            "latest_scan_rejection": self.latest_scan_rejection,
            "latest_scan_rejection_age": self._age(
                now, self.latest_scan_rejection_receipt
            ),
            "scan_rejection_count": self.scan_rejection_count,
            "consecutive_scan_rejections": self.consecutive_scan_rejections,
            "scan_rejection_grace": self.scan_rejection_grace,
            "scan_age": self._age(now, self.last_scan_receipt),
            "mask_age": self._age(now, self.last_mask_receipt),
            "path_age": self._age(now, self.last_path_receipt),
            "tf_age": self._age(now, self.last_tf_receipt),
            "raw_valid_rays": self.raw_valid_ray_count,
            "valid_rays": self.valid_ray_count,
            "minimum_valid_rays": self.envelope.minimum_valid_rays,
            "self_return_filter_enabled": self.self_return_filter_enabled,
            "self_return_max_range": self.self_return_max_range,
            "self_return_footprint_padding": (
                self.self_return_footprint_padding
            ),
            "self_return_rejections": self.self_return_rejection_count,
            "surface_cells_from_latest_scan": self.transformed_surface_cells,
            "inflation_radius": self.inflation_radius,
            "retained_surface_cells": (
                None if digest is None else digest.retained_surface_cells
            ),
            "inflated_surface_cells": (
                None if digest is None else digest.inflated_surface_cells
            ),
            "shadow_unknown_cells": (
                None if digest is None else digest.shadow_unknown_cells
            ),
            "outside_road_cells": (
                None if digest is None else digest.outside_road_cells
            ),
            "blocked_cells": None if digest is None else digest.blocked_cells,
            "trajectory_samples": assessment.evaluated_samples,
            "trajectory_shadow_overlap_samples": trajectory_shadow_overlap_samples,
            "first_unsafe": (
                None
                if assessment.first_unsafe_x is None
                else {
                    "x": assessment.first_unsafe_x,
                    "y": assessment.first_unsafe_y,
                    "value": assessment.first_unsafe_value,
                }
            ),
            "note": "collision evidence only; authorization remains in the final hardware gate",
        }
        self.status_publisher.publish(
            String(data=json.dumps(payload, separators=(",", ":")))
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DreamCollisionMonitorNode()
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
