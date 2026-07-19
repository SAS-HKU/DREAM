"""20 Hz drive-mode gate and LIMO command conversion."""

from __future__ import annotations

import json
from math import tan
from typing import Optional

import rclpy
from geometry_msgs.msg import Twist, TwistStamped
from limo_msgs.msg import LimoStatus
from rclpy.node import Node
from std_msgs.msg import String

from .core.command_adapter import CommandAdapter, VelocityCommand
from .limo_scale import default_deployment_config


class DreamCommandAdapterNode(Node):
    def __init__(self) -> None:
        super().__init__("dream_command_adapter")
        self.config = default_deployment_config()
        self.declare_parameter("control_topic", "/dream/control")
        self.declare_parameter("status_topic", "/limo_status")
        self.declare_parameter("candidate_topic", "/dream/cmd_vel_candidate")
        self.declare_parameter("allow_differential", False)
        self.declare_parameter("publish_rate", 20.0)
        self.declare_parameter("control_timeout", self.config.safety.planner_timeout)
        self.declare_parameter("status_timeout", self.config.safety.status_timeout)
        self.adapter = CommandAdapter(
            self.config.safety,
            control_dt=1.0 / float(self.get_parameter("publish_rate").value),
        )
        self.control: Optional[TwistStamped] = None
        self.control_receipt: Optional[float] = None
        self.motion_mode: Optional[int] = None
        self.status_receipt: Optional[float] = None
        self.create_subscription(
            TwistStamped,
            str(self.get_parameter("control_topic").value),
            self._on_control,
            10,
        )
        self.create_subscription(
            LimoStatus,
            str(self.get_parameter("status_topic").value),
            self._on_status,
            10,
        )
        self.candidate_publisher = self.create_publisher(
            Twist, str(self.get_parameter("candidate_topic").value), 10
        )
        self.status_publisher = self.create_publisher(String, "/dream/adapter_status", 10)
        self.create_timer(1.0 / float(self.get_parameter("publish_rate").value), self._publish)

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1.0e-9

    def _on_control(self, message: TwistStamped) -> None:
        self.control = message
        self.control_receipt = self._now()

    def _on_status(self, message: LimoStatus) -> None:
        self.motion_mode = int(message.motion_mode)
        self.status_receipt = self._now()

    def _zero(self, reason: str) -> VelocityCommand:
        self.adapter.reset()
        return VelocityCommand.zero(reason)

    def _publish(self) -> None:
        now = self._now()
        if self.control is None or self.control_receipt is None or now - self.control_receipt >= float(
            self.get_parameter("control_timeout").value
        ):
            command = self._zero("STALE_CONTROL")
        elif self.status_receipt is None or now - self.status_receipt >= float(
            self.get_parameter("status_timeout").value
        ):
            command = self._zero("STALE_STATUS")
        else:
            center_steer = float(self.control.twist.angular.z)
            speed = float(self.control.twist.linear.x)
            desired_yaw_rate = speed / self.config.mpc.wheelbase * tan(center_steer)
            command = self.adapter.adapt(
                target_speed=speed,
                center_steer=center_steer,
                motion_mode=self.motion_mode,
                allow_differential=bool(self.get_parameter("allow_differential").value),
                desired_yaw_rate=desired_yaw_rate,
            )
        message = Twist()
        message.linear.x = command.linear_x
        message.angular.z = command.angular_z
        self.candidate_publisher.publish(message)
        status = String()
        status.data = json.dumps(
            {
                "valid": command.valid,
                "reason": command.reason,
                "motion_mode": self.motion_mode,
                "linear_x": command.linear_x,
                "angular_z": command.angular_z,
                "required_mode": self.config.safety.required_motion_mode,
            },
            separators=(",", ":"),
        )
        self.status_publisher.publish(status)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DreamCommandAdapterNode()
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
