"""Independent, fail-closed final publisher for dry-run commands."""

from __future__ import annotations

import json
from math import isfinite

import numpy as np
import rclpy
from geometry_msgs.msg import TwistStamped
from limo_msgs.msg import LimoStatus
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, String

from .core.command_adapter import SafetySupervisorCore, VelocityCommand
from .limo_scale import default_deployment_config
from .ros_utils import (
    ControlSourceStamp,
    stamped_twist_from_velocity_command,
    velocity_command_from_stamped_twist,
)


class DreamSafetySupervisorNode(Node):
    """The only DREAM publisher on ``/cmd_vel_test``.

    This node deliberately has no parameter or remapping helper for ``/cmd_vel``.
    A later hardware stage must introduce a separately reviewed launch file.
    """

    OUTPUT_TOPIC = "/cmd_vel_test"

    def __init__(self) -> None:
        super().__init__("dream_safety_supervisor")
        self.config = default_deployment_config()
        self.core = SafetySupervisorCore(self.config.safety)
        self.declare_parameter("candidate_topic", "/dream/cmd_vel_candidate")
        self.declare_parameter("odom_topic", "/wheel/odom")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("status_topic", "/limo_status")
        self.declare_parameter("publish_rate", 20.0)
        self.declare_parameter("external_stop_topic", "/dream/external_stop")
        self.declare_parameter("reset_topic", "/dream/reset_stop")
        self.declare_parameter("arm_topic", "/dream/arm")
        self.candidate_source_stamp: ControlSourceStamp | None = None

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
            TwistStamped,
            str(self.get_parameter("candidate_topic").value),
            self._on_candidate,
            reliable,
        )
        self.create_subscription(
            Odometry, str(self.get_parameter("odom_topic").value), self._on_odom, reliable
        )
        self.create_subscription(
            LaserScan, str(self.get_parameter("scan_topic").value), self._on_scan, sensor_qos
        )
        self.create_subscription(
            LimoStatus, str(self.get_parameter("status_topic").value), self._on_status, reliable
        )
        self.create_subscription(
            Bool,
            str(self.get_parameter("external_stop_topic").value),
            self._on_external_stop,
            reliable,
        )
        self.create_subscription(
            Bool, str(self.get_parameter("reset_topic").value), self._on_reset, reliable
        )
        self.create_subscription(
            Bool, str(self.get_parameter("arm_topic").value), self._on_arm, reliable
        )
        self.output_publisher = self.create_publisher(
            TwistStamped, self.OUTPUT_TOPIC, reliable
        )
        self.status_publisher = self.create_publisher(String, "/dream/safety_status", reliable)
        self.create_timer(1.0 / float(self.get_parameter("publish_rate").value), self._publish)

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1.0e-9

    def _on_candidate(self, message: TwistStamped) -> None:
        command, source_stamp = velocity_command_from_stamped_twist(
            message,
            malformed_reason="MALFORMED_CANDIDATE",
        )
        self.candidate_source_stamp = source_stamp
        self.core.update_candidate(command, self._now())

    def _on_odom(self, _message: Odometry) -> None:
        self.core.update_odom(self._now())

    def _on_status(self, message: LimoStatus) -> None:
        self.core.update_status(int(message.motion_mode), self._now())

    def _on_scan(self, message: LaserScan) -> None:
        ranges = np.asarray(message.ranges, dtype=np.float64)
        angles = message.angle_min + np.arange(len(ranges)) * message.angle_increment
        sector = np.abs(angles) <= self.config.safety.front_sector_half_angle
        self.core.update_scan(
            ranges[sector],
            self._now(),
            range_max=float(message.range_max),
        )

    def _on_external_stop(self, message: Bool) -> None:
        self.core.set_external_stop(bool(message.data))

    def _on_reset(self, message: Bool) -> None:
        if message.data and not self.core.request_reset():
            self.get_logger().warning("Safety reset rejected while candidate command is nonzero")

    def _on_arm(self, message: Bool) -> None:
        self.core.set_armed(bool(message.data), self._now())

    def _publish(self) -> None:
        now = self._now()
        command = self.core.evaluate(now)
        if command.valid and self.candidate_source_stamp is None:
            command = VelocityCommand.zero("MISSING_CONTROL_SOURCE_STAMP")
        message = stamped_twist_from_velocity_command(
            command,
            self.candidate_source_stamp,
        )
        self.output_publisher.publish(message)
        status = String()
        status.data = json.dumps(
            {
                "safe": command.valid,
                "reason": command.reason,
                "output_topic": self.OUTPUT_TOPIC,
                "linear_x": command.linear_x,
                "angular_z": command.angular_z,
                "control_source_stamp": (
                    self.candidate_source_stamp.as_mapping()
                    if command.valid
                    and self.candidate_source_stamp is not None
                    else None
                ),
                "motion_mode": self.core.motion_mode,
                "obstacle_latched": self.core.obstacle_latched,
                "external_stop_latched": self.core.external_stop_latched,
                "front_minimum_range": (
                    self.core.front_minimum_range
                    if isfinite(self.core.front_minimum_range)
                    else None
                ),
                "front_stop_distance": self.config.safety.front_stop_distance,
                "armed": self.core.armed_since is not None,
                "arm_heartbeat_age": (
                    None
                    if self.core.arm_heartbeat_stamp is None
                    else max(0.0, now - self.core.arm_heartbeat_stamp)
                ),
                "arm_heartbeat_timeout": self.config.safety.arm_heartbeat_timeout,
                "maximum_ackermann_angular_command": (
                    self.config.safety.maximum_ackermann_angular_command
                ),
            },
            separators=(",", ":"),
        )
        self.status_publisher.publish(status)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DreamSafetySupervisorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Publish an explicit zero before teardown when the executor permits it.
        if rclpy.ok():
            node.output_publisher.publish(TwistStamped())
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
