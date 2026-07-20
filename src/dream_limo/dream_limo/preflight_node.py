"""Read-only ROS graph checks required before any DREAM hardware launch."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from .core.hardware_gate import exact_publisher_owner


def evaluate_occlusion_requirement(
    world_status: Dict[str, Any],
    *,
    world_status_fresh: bool,
    required: bool,
    latch: bool,
    previously_observed: bool,
) -> tuple[bool, bool, bool, bool]:
    """Evaluate initial occlusion evidence while preserving live watchdogs.

    Returns ``(ready, current, observed, world_live_and_aligned)``. A latched
    observation permits the intended later reveal, but never masks a stale or
    misaligned world model.
    """

    live = bool(
        world_status_fresh
        and world_status.get("ready", False)
        and world_status.get("alignment_received", False)
        and world_status.get("occlusion_source") == "lidar_first_return"
        and not world_status.get("surveyed_static_geometry_used", True)
    )
    current = bool(
        live
        and int(world_status.get("shadow_cells", 0)) > 0
        and int(world_status.get("shadow_route_samples", 0)) > 0
    )
    observed = bool(previously_observed or current)
    ready = bool(not required or current or (latch and observed and live))
    return ready, current, observed, live


class DreamPreflight(Node):
    def __init__(self) -> None:
        super().__init__("dream_preflight")
        self.declare_parameter("expected_sensor_owner", "")
        self.declare_parameter("expected_cmd_vel_owner", "")
        self.declare_parameter("expected_arm_owner", "")
        self.declare_parameter("allow_arm_publisher", False)
        self.declare_parameter("require_camera_evidence", False)
        self.declare_parameter("require_perceived_occlusion", False)
        self.declare_parameter("latch_perceived_occlusion", False)
        self.publisher = self.create_publisher(String, "/dream/preflight_status", 10)
        self.last_payload = None
        self.camera_status: Dict[str, Any] = {}
        self.camera_status_receipt: Optional[float] = None
        self.world_status: Dict[str, Any] = {}
        self.world_status_receipt: Optional[float] = None
        self.perceived_occlusion_observed = False
        self.create_subscription(
            String,
            "/dream/camera_evidence_status",
            self._on_camera_status,
            10,
        )
        self.create_subscription(String, "/dream/world_status", self._on_world_status, 10)
        self.create_timer(1.0, self._check)

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1.0e-9

    def _on_camera_status(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
            self.camera_status = payload if isinstance(payload, dict) else {}
            self.camera_status_receipt = self._now()
        except (json.JSONDecodeError, TypeError):
            self.camera_status = {}
            self.camera_status_receipt = None

    def _on_world_status(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
            self.world_status = payload if isinstance(payload, dict) else {}
            self.world_status_receipt = self._now()
        except (json.JSONDecodeError, TypeError):
            self.world_status = {}
            self.world_status_receipt = None

    def _check(self) -> None:
        command_publishers = self.get_publishers_info_by_topic("/cmd_vel")
        test_publishers = self.get_publishers_info_by_topic("/cmd_vel_test")
        arm_publishers = self.get_publishers_info_by_topic("/dream/arm")
        topics = dict(self.get_topic_names_and_types())
        required = ("/wheel/odom", "/scan", "/limo_status", "/cmd_vel_test")
        missing = [name for name in required if name not in topics]
        expected_sensor_owner = str(self.get_parameter("expected_sensor_owner").value)
        expected_cmd_vel_owner = str(
            self.get_parameter("expected_cmd_vel_owner").value
        )
        command_publisher_names = [item.node_name for item in command_publishers]
        cmd_vel_mode = "hardware_gate" if expected_cmd_vel_owner else "dry_run"
        cmd_vel_owner_safe = (
            exact_publisher_owner(command_publisher_names, expected_cmd_vel_owner)
            if expected_cmd_vel_owner
            else not command_publisher_names
        )
        sensor_owner_mismatches = {}
        if expected_sensor_owner:
            for topic in ("/wheel/odom", "/scan", "/limo_status"):
                owners = [
                    item.node_name for item in self.get_publishers_info_by_topic(topic)
                ]
                if owners != [expected_sensor_owner]:
                    sensor_owner_mismatches[topic] = owners
        safe_owner = (
            len(test_publishers) == 1
            and test_publishers[0].node_name == "dream_safety_supervisor"
        )
        arm_publisher_names = [item.node_name for item in arm_publishers]
        expected_arm_owner = str(self.get_parameter("expected_arm_owner").value)
        arm_source_safe = (
            exact_publisher_owner(arm_publisher_names, expected_arm_owner)
            if expected_arm_owner
            else bool(self.get_parameter("allow_arm_publisher").value)
            or not arm_publisher_names
        )
        camera_required = bool(self.get_parameter("require_camera_evidence").value)
        camera_status_fresh = (
            self.camera_status_receipt is not None
            and self._now() - self.camera_status_receipt < 0.5
        )
        camera_ready = (
            not camera_required
            or (camera_status_fresh and bool(self.camera_status.get("ready", False)))
        )
        perceived_occlusion_required = bool(
            self.get_parameter("require_perceived_occlusion").value
        )
        world_status_fresh = (
            self.world_status_receipt is not None
            and self._now() - self.world_status_receipt < 0.5
        )
        latch_perceived_occlusion = bool(
            self.get_parameter("latch_perceived_occlusion").value
        )
        (
            perceived_occlusion_ready,
            perceived_occlusion_current,
            self.perceived_occlusion_observed,
            _world_live_and_aligned,
        ) = evaluate_occlusion_requirement(
            self.world_status,
            world_status_fresh=world_status_fresh,
            required=perceived_occlusion_required,
            latch=latch_perceived_occlusion,
            previously_observed=self.perceived_occlusion_observed,
        )
        passed = (
            cmd_vel_owner_safe
            and safe_owner
            and not missing
            and not sensor_owner_mismatches
            and arm_source_safe
            and camera_ready
            and perceived_occlusion_ready
        )
        payload = {
            "passed": passed,
            "cmd_vel_mode": cmd_vel_mode,
            "expected_cmd_vel_owner": expected_cmd_vel_owner or None,
            "cmd_vel_owner_safe": cmd_vel_owner_safe,
            "cmd_vel_publishers": command_publisher_names,
            "cmd_vel_test_publishers": [item.node_name for item in test_publishers],
            "missing_topics": missing,
            "expected_sensor_owner": expected_sensor_owner or None,
            "sensor_owner_mismatches": sensor_owner_mismatches,
            "expected_arm_owner": expected_arm_owner or None,
            "arm_publishers": arm_publisher_names,
            "arm_source_allowed": arm_source_safe,
            "camera_required": camera_required,
            "camera_ready": camera_ready,
            "camera_status_fresh": camera_status_fresh,
            "camera_status": self.camera_status if camera_required else None,
            "perceived_occlusion_required": perceived_occlusion_required,
            "perceived_occlusion_ready": perceived_occlusion_ready,
            "perceived_occlusion_current": perceived_occlusion_current,
            "perceived_occlusion_observed": self.perceived_occlusion_observed,
            "latch_perceived_occlusion": latch_perceived_occlusion,
            "world_status_fresh": world_status_fresh,
            "world_status": self.world_status if perceived_occlusion_required else None,
            "note": (
                "Passing preflight is only one hardware-gate prerequisite; "
                "it does not independently authorize physical motion."
            ),
        }
        message = String()
        message.data = json.dumps(payload, separators=(",", ":"))
        self.publisher.publish(message)
        if message.data != self.last_payload:
            if passed:
                self.get_logger().info(message.data)
            else:
                self.get_logger().warning(message.data)
            self.last_payload = message.data


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DreamPreflight()
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
