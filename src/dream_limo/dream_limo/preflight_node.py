"""Read-only ROS graph checks required before any DREAM hardware launch."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class DreamPreflight(Node):
    def __init__(self) -> None:
        super().__init__("dream_preflight")
        self.declare_parameter("expected_sensor_owner", "")
        self.declare_parameter("allow_arm_publisher", False)
        self.declare_parameter("require_camera_evidence", False)
        self.declare_parameter("require_perceived_occlusion", False)
        self.publisher = self.create_publisher(String, "/dream/preflight_status", 10)
        self.last_payload = None
        self.camera_status: Dict[str, Any] = {}
        self.camera_status_receipt: Optional[float] = None
        self.world_status: Dict[str, Any] = {}
        self.world_status_receipt: Optional[float] = None
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
        arm_source_safe = bool(self.get_parameter("allow_arm_publisher").value) or not arm_publishers
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
        perceived_occlusion_ready = (
            not perceived_occlusion_required
            or (
                world_status_fresh
                and bool(self.world_status.get("ready", False))
                and bool(self.world_status.get("alignment_received", False))
                and self.world_status.get("occlusion_source") == "lidar_first_return"
                and not bool(self.world_status.get("surveyed_static_geometry_used", True))
                and int(self.world_status.get("shadow_cells", 0)) > 0
                and int(self.world_status.get("shadow_route_samples", 0)) > 0
            )
        )
        passed = (
            not command_publishers
            and safe_owner
            and not missing
            and not sensor_owner_mismatches
            and arm_source_safe
            and camera_ready
            and perceived_occlusion_ready
        )
        payload = {
            "passed": passed,
            "cmd_vel_publishers": [item.node_name for item in command_publishers],
            "cmd_vel_test_publishers": [item.node_name for item in test_publishers],
            "missing_topics": missing,
            "expected_sensor_owner": expected_sensor_owner or None,
            "sensor_owner_mismatches": sensor_owner_mismatches,
            "arm_publishers": [item.node_name for item in arm_publishers],
            "arm_source_allowed": arm_source_safe,
            "camera_required": camera_required,
            "camera_ready": camera_ready,
            "camera_status_fresh": camera_status_fresh,
            "camera_status": self.camera_status if camera_required else None,
            "perceived_occlusion_required": perceived_occlusion_required,
            "perceived_occlusion_ready": perceived_occlusion_ready,
            "world_status_fresh": world_status_fresh,
            "world_status": self.world_status if perceived_occlusion_required else None,
            "note": "Passing dry-run preflight does not authorize physical motion.",
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
