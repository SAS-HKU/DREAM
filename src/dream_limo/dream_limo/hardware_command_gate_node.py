"""Reviewed, fail-closed final publisher for physical LIMO motion.

This is intentionally separate from the dry-run safety supervisor. Merely
starting this node publishes zeros: physical output additionally needs two
explicit launch booleans and continuously fresh independent safety evidence.
"""

from __future__ import annotations

import json
from math import isfinite
from typing import Any, Dict, Optional

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from limo_msgs.msg import LimoStatus
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

from .core.command_adapter import VelocityCommand
from .core.hardware_gate import HardwareCommandGateCore, HardwareGateConfig
from .limo_scale import default_deployment_config


class DreamHardwareCommandGateNode(Node):
    """The only DREAM node permitted to publish the physical ``/cmd_vel``."""

    OUTPUT_TOPIC = "/cmd_vel"

    def __init__(self) -> None:
        super().__init__("dream_hardware_command_gate")
        deployment = default_deployment_config()
        defaults = HardwareGateConfig(
            maximum_speed=deployment.safety.initial_hardware_speed_cap,
            maximum_acceleration=deployment.safety.maximum_acceleration,
            maximum_ackermann_angular_command=(
                deployment.safety.maximum_ackermann_angular_command
            ),
            required_motion_mode=deployment.safety.required_motion_mode,
        )

        # Both are false by default. They express separate operator assertions:
        # reviewed hardware output is intended, and the robot is physically at
        # the checked-in mission start pose.
        self.declare_parameter("hardware_output_enabled", False)
        self.declare_parameter("staging_pose_verified", False)
        # The serial-boundary watchdog is installed but is not yet physically
        # commissioned. These remain false until a wheels-off-ground stale-
        # command test and the human's independent stop control have both been
        # demonstrated for this unit.
        self.declare_parameter("platform_watchdog_verified", False)
        self.declare_parameter("operator_kill_verified", False)
        self.declare_parameter("candidate_topic", "/cmd_vel_test")
        self.declare_parameter("odom_topic", "/wheel/odom")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("status_topic", "/limo_status")
        self.declare_parameter("safety_status_topic", "/dream/safety_status")
        self.declare_parameter("preflight_status_topic", "/dream/preflight_status")
        self.declare_parameter("collision_status_topic", "/dream/collision_status")
        self.declare_parameter("deadman_status_topic", "/dream/deadman_status")
        self.declare_parameter("world_status_topic", "/dream/world_status")
        self.declare_parameter("drift_status_topic", "/dream/drift_status")
        self.declare_parameter("planner_status_topic", "/dream/planner_status")
        self.declare_parameter("gate_status_topic", "/dream/hardware_gate_status")
        self.declare_parameter("expected_candidate_owner", "dream_safety_supervisor")
        self.declare_parameter("expected_deadman_owner", "dream_hardware_deadman")
        self.declare_parameter("publish_rate", defaults.publish_rate)
        self.declare_parameter(
            "readiness_countdown_seconds", defaults.readiness_countdown_seconds
        )
        self.declare_parameter("maximum_speed", defaults.maximum_speed)
        self.declare_parameter("maximum_acceleration", defaults.maximum_acceleration)
        self.declare_parameter(
            "maximum_ackermann_angular_command",
            defaults.maximum_ackermann_angular_command,
        )
        self.declare_parameter(
            "maximum_ackermann_angular_slew",
            defaults.maximum_ackermann_angular_slew,
        )
        self.declare_parameter("candidate_timeout", defaults.candidate_timeout)
        self.declare_parameter("odom_timeout", defaults.odom_timeout)
        self.declare_parameter("scan_timeout", defaults.scan_timeout)
        self.declare_parameter("status_timeout", defaults.status_timeout)
        self.declare_parameter("safety_status_timeout", defaults.safety_status_timeout)
        self.declare_parameter("preflight_timeout", defaults.preflight_timeout)
        self.declare_parameter("collision_timeout", defaults.collision_timeout)
        self.declare_parameter("deadman_timeout", defaults.deadman_timeout)
        self.declare_parameter("world_timeout", defaults.world_timeout)
        self.declare_parameter("drift_timeout", defaults.drift_timeout)
        self.declare_parameter("planner_status_timeout", defaults.planner_status_timeout)
        self.declare_parameter("required_motion_mode", defaults.required_motion_mode)

        self.gate_config = HardwareGateConfig(
            maximum_speed=float(self.get_parameter("maximum_speed").value),
            maximum_acceleration=float(
                self.get_parameter("maximum_acceleration").value
            ),
            maximum_ackermann_angular_command=float(
                self.get_parameter("maximum_ackermann_angular_command").value
            ),
            maximum_ackermann_angular_slew=float(
                self.get_parameter("maximum_ackermann_angular_slew").value
            ),
            publish_rate=float(self.get_parameter("publish_rate").value),
            readiness_countdown_seconds=float(
                self.get_parameter("readiness_countdown_seconds").value
            ),
            candidate_timeout=float(self.get_parameter("candidate_timeout").value),
            odom_timeout=float(self.get_parameter("odom_timeout").value),
            scan_timeout=float(self.get_parameter("scan_timeout").value),
            status_timeout=float(self.get_parameter("status_timeout").value),
            safety_status_timeout=float(
                self.get_parameter("safety_status_timeout").value
            ),
            preflight_timeout=float(self.get_parameter("preflight_timeout").value),
            collision_timeout=float(self.get_parameter("collision_timeout").value),
            deadman_timeout=float(self.get_parameter("deadman_timeout").value),
            world_timeout=float(self.get_parameter("world_timeout").value),
            drift_timeout=float(self.get_parameter("drift_timeout").value),
            planner_status_timeout=float(
                self.get_parameter("planner_status_timeout").value
            ),
            required_motion_mode=int(
                self.get_parameter("required_motion_mode").value
            ),
        )
        self.core = HardwareCommandGateCore(self.gate_config)

        reliable = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(
            Twist,
            str(self.get_parameter("candidate_topic").value),
            self._on_candidate,
            reliable,
        )
        self.create_subscription(
            Odometry,
            str(self.get_parameter("odom_topic").value),
            self._on_odom,
            reliable,
        )
        self.create_subscription(
            LaserScan,
            str(self.get_parameter("scan_topic").value),
            self._on_scan,
            sensor_qos,
        )
        self.create_subscription(
            LimoStatus,
            str(self.get_parameter("status_topic").value),
            self._on_status,
            reliable,
        )
        for parameter, callback in (
            ("safety_status_topic", self._on_safety),
            ("preflight_status_topic", self._on_preflight),
            ("collision_status_topic", self._on_collision),
            ("deadman_status_topic", self._on_deadman),
            ("world_status_topic", self._on_world),
            ("drift_status_topic", self._on_drift),
            ("planner_status_topic", self._on_planner),
        ):
            self.create_subscription(
                String,
                str(self.get_parameter(parameter).value),
                callback,
                reliable,
            )

        self.output_publisher = self.create_publisher(
            Twist, self.OUTPUT_TOPIC, reliable
        )
        self.gate_status_publisher = self.create_publisher(
            String, str(self.get_parameter("gate_status_topic").value), reliable
        )
        self.create_timer(1.0 / self.gate_config.publish_rate, self._publish)

        if not bool(self.get_parameter("hardware_output_enabled").value):
            self.get_logger().warning(
                "Hardware output disabled (safe default); /cmd_vel will remain zero"
            )

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1.0e-9

    @staticmethod
    def _payload(message: String) -> Optional[Dict[str, Any]]:
        try:
            value = json.loads(message.data)
        except (json.JSONDecodeError, TypeError):
            return None
        return value if isinstance(value, dict) else None

    @staticmethod
    def _finite_number(value: Any, default: float = float("inf")) -> float:
        if isinstance(value, bool):
            return default
        try:
            result = float(value)
        except (TypeError, ValueError):
            return default
        return result if isfinite(result) else default

    @staticmethod
    def _integer(value: Any, default: int = -1) -> int:
        if isinstance(value, bool):
            return default
        try:
            return int(value)
        except (TypeError, ValueError, OverflowError):
            return default

    def _on_candidate(self, message: Twist) -> None:
        expected_zero = (
            float(message.linear.y),
            float(message.linear.z),
            float(message.angular.x),
            float(message.angular.y),
        )
        values = (float(message.linear.x), float(message.angular.z), *expected_zero)
        valid = all(isfinite(value) for value in values) and all(
            abs(value) <= 1.0e-9 for value in expected_zero
        )
        command = (
            VelocityCommand(values[0], values[1], True, "ok")
            if valid
            else VelocityCommand.zero("MALFORMED_CANDIDATE")
        )
        self.core.update_candidate(command, self._now())

    def _on_odom(self, _message: Odometry) -> None:
        self.core.update_odom(self._now())

    def _on_scan(self, message: LaserScan) -> None:
        values = np.asarray(message.ranges, dtype=np.float64)
        ray_valid = np.any(np.isfinite(values) & (values > 0.0)) or (
            np.any(np.isposinf(values))
            and isfinite(float(message.range_max))
            and float(message.range_max) > 0.0
        )
        if values.size > 0 and ray_valid:
            self.core.update_scan(self._now())

    def _on_status(self, message: LimoStatus) -> None:
        self.core.update_status(int(message.motion_mode), self._now())

    def _on_safety(self, message: String) -> None:
        payload = self._payload(message)
        if payload is None:
            self.core.update_safety(False, "MALFORMED_STATUS", self._now())
            return
        safe = bool(payload.get("safe", False)) and all(
            (
                payload.get("reason") == "ok",
                payload.get("output_topic")
                == str(self.get_parameter("candidate_topic").value),
                self._integer(payload.get("motion_mode"))
                == self.gate_config.required_motion_mode,
                not bool(payload.get("obstacle_latched", True)),
                not bool(payload.get("external_stop_latched", True)),
                bool(payload.get("armed", False)),
            )
        )
        self.core.update_safety(
            safe, str(payload.get("reason", "INVALID_STATUS")), self._now()
        )

    def _on_preflight(self, message: String) -> None:
        payload = self._payload(message)
        passed = bool(
            payload
            and payload.get("passed", False)
            and payload.get("cmd_vel_mode") == "hardware_gate"
            and payload.get("cmd_vel_owner_safe", False)
            and payload.get("expected_cmd_vel_owner") == self.get_name()
        )
        self.core.update_preflight(passed, self._now())

    def _on_collision(self, message: String) -> None:
        payload = self._payload(message) or {}
        self.core.update_collision(
            ready=bool(payload.get("ready", False)),
            trajectory_clear=bool(payload.get("trajectory_clear", False)),
            stamp=self._now(),
        )

    def _on_deadman(self, message: String) -> None:
        payload = self._payload(message) or {}
        self.core.update_deadman(
            ready=bool(payload.get("ready", False))
            and not bool(payload.get("external_stop", True)),
            armed=bool(payload.get("armed", False)),
            stamp=self._now(),
        )

    def _on_world(self, message: String) -> None:
        payload = self._payload(message) or {}
        self.core.update_world(
            ready=bool(payload.get("ready", False)),
            ego_fresh=bool(payload.get("ego_fresh", False)),
            scan_fresh=bool(payload.get("scan_fresh", False)),
            tracks_fresh=bool(payload.get("tracks_fresh", False)),
            alignment_received=bool(payload.get("alignment_received", False)),
            stamp=self._now(),
        )

    def _on_drift(self, message: String) -> None:
        payload = self._payload(message) or {}
        self.core.update_drift(ready=bool(payload.get("ready", False)), stamp=self._now())

    def _on_planner(self, message: String) -> None:
        payload = self._payload(message) or {}
        self.core.update_planner(
            ready=bool(payload.get("ready", False)),
            used_fallback=bool(payload.get("mpc_fallback", True)),
            maximum_cbf_slack=self._finite_number(
                payload.get("maximum_cbf_slack")
            ),
            maximum_allowed_cbf_slack=self._finite_number(
                payload.get("maximum_allowed_cbf_slack"), -1.0
            ),
            map_bounds_enforced=bool(payload.get("map_bounds_enforced", False)),
            stamp=self._now(),
        )

    def _publisher_names(self, topic: str) -> list[str]:
        return [item.node_name for item in self.get_publishers_info_by_topic(topic)]

    def _publish(self) -> None:
        now = self._now()
        candidate_topic = str(self.get_parameter("candidate_topic").value)
        deadman_topic = str(self.get_parameter("deadman_status_topic").value)
        candidate_owners = self._publisher_names(candidate_topic)
        output_owners = self._publisher_names(self.OUTPUT_TOPIC)
        deadman_owners = self._publisher_names(deadman_topic)
        expected_candidate = str(self.get_parameter("expected_candidate_owner").value)
        expected_deadman = str(self.get_parameter("expected_deadman_owner").value)
        candidate_owner_ok = candidate_owners == [expected_candidate]
        output_owner_ok = output_owners == [self.get_name()]
        deadman_owner_ok = deadman_owners == [expected_deadman]
        enabled = bool(self.get_parameter("hardware_output_enabled").value)
        staging_verified = bool(self.get_parameter("staging_pose_verified").value)
        platform_watchdog_verified = bool(
            self.get_parameter("platform_watchdog_verified").value
        )
        operator_kill_verified = bool(
            self.get_parameter("operator_kill_verified").value
        )
        command = self.core.evaluate(
            now,
            hardware_output_enabled=enabled,
            staging_pose_verified=staging_verified,
            platform_watchdog_verified=platform_watchdog_verified,
            operator_kill_verified=operator_kill_verified,
            candidate_owner_ok=candidate_owner_ok,
            output_owner_ok=output_owner_ok,
            deadman_owner_ok=deadman_owner_ok,
        )
        readiness_countdown_remaining = self.core.readiness_countdown_remaining(now)
        readiness_countdown_started = (
            self.core.readiness_countdown_started_at is not None
        )
        output = Twist()
        output.linear.x = command.linear_x
        output.angular.z = command.angular_z
        self.output_publisher.publish(output)

        def age(stamp: Optional[float]) -> Optional[float]:
            return None if stamp is None else max(0.0, now - stamp)

        status = {
            "ready": command.valid,
            "reason": command.reason,
            "hardware_output_enabled": enabled,
            "staging_pose_verified": staging_verified,
            "platform_watchdog_verified": platform_watchdog_verified,
            "operator_kill_verified": operator_kill_verified,
            "readiness_countdown_seconds": (
                self.gate_config.readiness_countdown_seconds
            ),
            "readiness_countdown_remaining": readiness_countdown_remaining,
            "readiness_countdown_active": bool(
                readiness_countdown_started and readiness_countdown_remaining > 0.0
            ),
            "readiness_countdown_complete": bool(
                readiness_countdown_started and readiness_countdown_remaining <= 0.0
            ),
            "output_topic": self.OUTPUT_TOPIC,
            "linear_x": command.linear_x,
            "angular_z": command.angular_z,
            "candidate_owners": candidate_owners,
            "candidate_owner_ok": candidate_owner_ok,
            "cmd_vel_owners": output_owners,
            "cmd_vel_owner_ok": output_owner_ok,
            "deadman_owners": deadman_owners,
            "deadman_owner_ok": deadman_owner_ok,
            "motion_mode": self.core.motion_mode,
            "required_motion_mode": self.gate_config.required_motion_mode,
            "deadman_ready": self.core.deadman_ready,
            "deadman_armed": self.core.deadman_armed,
            "collision_ready": self.core.collision_ready,
            "trajectory_clear": self.core.trajectory_clear,
            "world_ready": self.core.world_ready,
            "drift_ready": self.core.drift_ready,
            "planner_ready": self.core.planner_ready,
            "mpc_fallback": self.core.planner_used_fallback,
            "map_bounds_enforced": self.core.planner_map_bounds_enforced,
            "maximum_cbf_slack": (
                self.core.planner_slack if isfinite(self.core.planner_slack) else None
            ),
            "maximum_allowed_cbf_slack": self.core.planner_allowed_slack,
            "ages": {
                "candidate": age(self.core.candidate_stamp),
                "odom": age(self.core.odom_stamp),
                "scan": age(self.core.scan_stamp),
                "status": age(self.core.status_stamp),
                "safety": age(self.core.safety_stamp),
                "preflight": age(self.core.preflight_stamp),
                "collision": age(self.core.collision_stamp),
                "deadman": age(self.core.deadman_stamp),
                "world": age(self.core.world_stamp),
                "drift": age(self.core.drift_stamp),
                "planner": age(self.core.planner_stamp),
            },
        }
        self.gate_status_publisher.publish(
            String(data=json.dumps(status, separators=(",", ":")))
        )

    def publish_stop(self) -> None:
        self.output_publisher.publish(Twist())


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DreamHardwareCommandGateNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            # Explicit stop on orderly teardown; the platform driver watchdog
            # remains a separately verified hardware prerequisite.
            node.publish_stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
