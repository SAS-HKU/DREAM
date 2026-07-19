"""Joystick-held arming heartbeat for the separately gated hardware stage."""

from __future__ import annotations

import json

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_msgs.msg import Bool, String

from .core.hardware_deadman import DeadmanDecision, evaluate_deadman_buttons


class DreamHardwareDeadmanNode(Node):
    """Publish `/dream/arm` only while a fresh two-button chord is held."""

    def __init__(self) -> None:
        super().__init__("dream_hardware_deadman")
        self.declare_parameter("enabled", False)
        self.declare_parameter("joy_topic", "/joy")
        self.declare_parameter("expected_joy_owner", "joy_node")
        self.declare_parameter("arm_topic", "/dream/arm")
        self.declare_parameter("external_stop_topic", "/dream/external_stop")
        self.declare_parameter("status_topic", "/dream/deadman_status")
        self.declare_parameter("hold_button", 4)
        self.declare_parameter("confirm_button", 5)
        self.declare_parameter("stop_button", 1)
        self.declare_parameter("joy_timeout", 0.25)
        self.declare_parameter("publish_rate", 20.0)

        self.enabled = bool(self.get_parameter("enabled").value)
        self.hold_button = int(self.get_parameter("hold_button").value)
        self.confirm_button = int(self.get_parameter("confirm_button").value)
        self.stop_button = int(self.get_parameter("stop_button").value)
        self.joy_timeout = float(self.get_parameter("joy_timeout").value)
        publish_rate = float(self.get_parameter("publish_rate").value)
        if self.joy_timeout <= 0.0 or publish_rate <= 0.0:
            raise ValueError("deadman timing parameters must be positive")

        self.last_joy_receipt: float | None = None
        self.last_decision = DeadmanDecision(
            False, False, False, "WAITING_FOR_JOY"
        )
        self.create_subscription(
            Joy,
            str(self.get_parameter("joy_topic").value),
            self._on_joy,
            10,
        )
        self.arm_publisher = self.create_publisher(
            Bool, str(self.get_parameter("arm_topic").value), 10
        )
        self.stop_publisher = self.create_publisher(
            Bool, str(self.get_parameter("external_stop_topic").value), 10
        )
        self.status_publisher = self.create_publisher(
            String, str(self.get_parameter("status_topic").value), 10
        )
        self.create_timer(1.0 / publish_rate, self._publish)
        if not self.enabled:
            self.get_logger().warning(
                "Hardware deadman is disabled; physical arming is impossible"
            )

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1.0e-9

    def _on_joy(self, message: Joy) -> None:
        self.last_joy_receipt = self._now()
        self.last_decision = evaluate_deadman_buttons(
            message.buttons,
            hold_button=self.hold_button,
            confirm_button=self.confirm_button,
            stop_button=self.stop_button,
        )

    def _publish(self) -> None:
        now = self._now()
        joy_topic = str(self.get_parameter("joy_topic").value)
        joy_owners = [
            item.node_name for item in self.get_publishers_info_by_topic(joy_topic)
        ]
        expected_joy_owner = str(self.get_parameter("expected_joy_owner").value)
        joy_owner_ok = bool(expected_joy_owner) and joy_owners == [expected_joy_owner]
        age = (
            None
            if self.last_joy_receipt is None
            else max(0.0, now - self.last_joy_receipt)
        )
        fresh = age is not None and age < self.joy_timeout
        decision = self.last_decision
        armed = bool(
            self.enabled
            and joy_owner_ok
            and fresh
            and decision.valid
            and decision.armed
        )
        external_stop = bool(
            self.enabled and joy_owner_ok and fresh and decision.external_stop
        )
        reason = (
            "DISABLED"
            if not self.enabled
            else "JOY_OWNER_MISMATCH"
            if not joy_owner_ok
            else "STALE_JOY"
            if not fresh
            else decision.reason
        )
        self.arm_publisher.publish(Bool(data=armed))
        self.stop_publisher.publish(Bool(data=external_stop))
        payload = {
            "ready": bool(
                self.enabled and joy_owner_ok and fresh and decision.valid
            ),
            "armed": armed,
            "external_stop": external_stop,
            "reason": reason,
            "joy_fresh": fresh,
            "joy_age": age,
            "joy_timeout": self.joy_timeout,
            "joy_topic": joy_topic,
            "joy_owners": joy_owners,
            "expected_joy_owner": expected_joy_owner,
            "joy_owner_ok": joy_owner_ok,
            "button_map": {
                "hold": self.hold_button,
                "confirm": self.confirm_button,
                "stop": self.stop_button,
            },
        }
        self.status_publisher.publish(
            String(data=json.dumps(payload, separators=(",", ":")))
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DreamHardwareDeadmanNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.arm_publisher.publish(Bool(data=False))
            node.stop_publisher.publish(Bool(data=True))
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
