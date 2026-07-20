"""Align a second LIMO's local odometry with the ego experiment frame."""

from __future__ import annotations

import copy
import json
from math import atan2, cos, isfinite, sin, sqrt
from typing import Optional, Tuple

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from .core.merger_odometry import (
    AlignmentResolver,
    PlanarPose,
    PlanarOdometry,
    align_planar_odometry,
    rotate_pose_covariance,
    validate_finite,
    validate_frame_id,
    validate_source_frames,
    validate_source_time,
)
from .ros_utils import stamp_to_seconds


def _normalized_quaternion(message) -> Tuple[float, float, float, float]:
    values = validate_finite(
        (message.x, message.y, message.z, message.w),
        label="source orientation",
    )
    norm = sqrt(sum(value * value for value in values))
    if norm < 0.9 or norm > 1.1:
        raise ValueError(f"source orientation quaternion norm is invalid ({norm:.6f})")
    return tuple(value / norm for value in values)


def _apply_yaw_to_quaternion(
    quaternion: Tuple[float, float, float, float],
    yaw: float,
) -> Tuple[float, float, float, float]:
    """Left-multiply a quaternion by a parent-frame yaw rotation."""
    x, y, z, w = quaternion
    sh, ch = sin(0.5 * yaw), cos(0.5 * yaw)
    result = (
        ch * x - sh * y,
        sh * x + ch * y,
        ch * z + sh * w,
        ch * w - sh * z,
    )
    norm = sqrt(sum(value * value for value in result))
    return tuple(value / norm for value in result)


def _quaternion_yaw(quaternion: Tuple[float, float, float, float]) -> float:
    x, y, z, w = quaternion
    return atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


class DreamMergerOdometryAdapter(Node):
    """Publish merger odometry only after explicit, validated frame alignment."""

    def __init__(self) -> None:
        super().__init__("dream_merger_odometry_adapter")
        self.declare_parameter("input_topic", "/merger/raw/wheel/odom")
        self.declare_parameter("output_topic", "/merger/wheel/odom")
        self.declare_parameter(
            "status_topic",
            "/dream/merger_odometry_adapter_status",
        )
        self.declare_parameter("expected_source_frame", "merger/odom")
        self.declare_parameter("expected_source_child_frame", "merger/base_link")
        self.declare_parameter("output_frame", "odom")
        self.declare_parameter("output_child_frame", "merger/base_link")
        self.declare_parameter("allow_source_output_frame_alias", False)

        self.declare_parameter("alignment_mode", "measured_correspondence")
        self.declare_parameter("alignment_verified", False)
        self.declare_parameter("source_reference_x", 0.0)
        self.declare_parameter("source_reference_y", 0.0)
        self.declare_parameter("source_reference_yaw", 0.0)
        self.declare_parameter("target_reference_x", 0.0)
        self.declare_parameter("target_reference_y", 0.0)
        self.declare_parameter("target_reference_yaw", 0.0)

        self.declare_parameter("maximum_input_age", 0.50)
        self.declare_parameter("maximum_future_skew", 0.05)
        self.declare_parameter("status_rate", 10.0)

        self.expected_source_frame = validate_frame_id(
            str(self.get_parameter("expected_source_frame").value),
            label="expected source parent frame",
        )
        self.expected_source_child_frame = validate_frame_id(
            str(self.get_parameter("expected_source_child_frame").value),
            label="expected source child frame",
        )
        self.output_frame = validate_frame_id(
            str(self.get_parameter("output_frame").value),
            label="output parent frame",
        )
        self.output_child_frame = validate_frame_id(
            str(self.get_parameter("output_child_frame").value),
            label="output child frame",
        )
        if self.output_frame == self.output_child_frame:
            raise RuntimeError("output parent and child frames must differ")

        self.maximum_input_age = float(
            self.get_parameter("maximum_input_age").value
        )
        self.maximum_future_skew = float(
            self.get_parameter("maximum_future_skew").value
        )
        if (
            not isfinite(self.maximum_input_age)
            or self.maximum_input_age <= 0.0
            or not isfinite(self.maximum_future_skew)
            or self.maximum_future_skew < 0.0
        ):
            raise RuntimeError("odometry timestamp tolerances are invalid")

        target_reference = PlanarPose(
            float(self.get_parameter("target_reference_x").value),
            float(self.get_parameter("target_reference_y").value),
            float(self.get_parameter("target_reference_yaw").value),
        )
        mode = str(self.get_parameter("alignment_mode").value)
        source_reference: Optional[PlanarPose]
        if mode == "measured_correspondence":
            source_reference = PlanarPose(
                float(self.get_parameter("source_reference_x").value),
                float(self.get_parameter("source_reference_y").value),
                float(self.get_parameter("source_reference_yaw").value),
            )
        else:
            source_reference = None
        try:
            self.resolver = AlignmentResolver(
                mode=mode,
                source_reference=source_reference,
                target_reference=target_reference,
            )
        except ValueError as exc:
            raise RuntimeError(f"invalid merger alignment configuration: {exc}") from exc

        self.alignment_verified = bool(
            self.get_parameter("alignment_verified").value
        )
        self.allow_parent_alias = bool(
            self.get_parameter("allow_source_output_frame_alias").value
        )
        self.last_valid_receipt: Optional[float] = None
        self.last_source_stamp: Optional[float] = None
        self.last_event_valid = False
        self.last_reason = (
            "WAITING_FOR_ODOMETRY"
            if self.alignment_verified
            else "ALIGNMENT_NOT_VERIFIED"
        )
        self.last_logged_reason: Optional[str] = None

        odometry_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=20,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        status_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.publisher = self.create_publisher(
            Odometry,
            str(self.get_parameter("output_topic").value),
            odometry_qos,
        )
        self.status_publisher = self.create_publisher(
            String,
            str(self.get_parameter("status_topic").value),
            status_qos,
        )
        self.create_subscription(
            Odometry,
            str(self.get_parameter("input_topic").value),
            self._on_odometry,
            odometry_qos,
        )
        status_rate = float(self.get_parameter("status_rate").value)
        if not isfinite(status_rate) or status_rate <= 0.0:
            raise RuntimeError("status_rate must be finite and positive")
        self.create_timer(1.0 / status_rate, self._publish_status)

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1.0e-9

    def _reject(self, reason: str) -> None:
        self.last_event_valid = False
        self.last_reason = reason
        if reason != self.last_logged_reason:
            self.get_logger().warning(f"Merger odometry rejected: {reason}")
            self.last_logged_reason = reason
        self._publish_status()

    def _on_odometry(self, message: Odometry) -> None:
        if not self.alignment_verified:
            self._reject("ALIGNMENT_NOT_VERIFIED")
            return
        now = self._now()
        try:
            validate_source_frames(
                actual_parent=message.header.frame_id,
                actual_child=message.child_frame_id,
                expected_parent=self.expected_source_frame,
                expected_child=self.expected_source_child_frame,
                output_parent=self.output_frame,
                allow_parent_alias=self.allow_parent_alias,
            )
            source_stamp = stamp_to_seconds(message.header.stamp)
            validate_source_time(
                source_stamp,
                now,
                maximum_age=self.maximum_input_age,
                maximum_future_skew=self.maximum_future_skew,
                previous_stamp=self.last_source_stamp,
            )
            position = message.pose.pose.position
            validate_finite(
                (position.x, position.y, position.z),
                label="source position",
            )
            source_quaternion = _normalized_quaternion(
                message.pose.pose.orientation
            )
            source_yaw = _quaternion_yaw(source_quaternion)
            source_pose = PlanarPose(position.x, position.y, source_yaw)
            twist = message.twist.twist
            child_twist = validate_finite(
                (
                    twist.linear.x,
                    twist.linear.y,
                    twist.linear.z,
                    twist.angular.x,
                    twist.angular.y,
                    twist.angular.z,
                ),
                label="source twist",
            )
            validate_finite(message.twist.covariance, label="twist covariance")
            if len(message.twist.covariance) != 36:
                raise ValueError("twist covariance must contain exactly 36 values")
            # Validate all content before first-message anchoring is allowed to
            # latch a transform.
            validate_finite(message.pose.covariance, label="pose covariance")
            if len(message.pose.covariance) != 36:
                raise ValueError("pose covariance must contain exactly 36 values")

            alignment = self.resolver.resolve(source_pose)
            aligned = align_planar_odometry(
                PlanarOdometry(
                    stamp=source_stamp,
                    pose=source_pose,
                    child_twist=child_twist,
                ),
                alignment,
            )
            target_pose = aligned.pose
            target_quaternion = _apply_yaw_to_quaternion(
                source_quaternion,
                alignment.yaw,
            )
            target_covariance = rotate_pose_covariance(
                message.pose.covariance,
                alignment.yaw,
            )
        except (TypeError, ValueError) as exc:
            self._reject(str(exc))
            return

        output = Odometry()
        # Keep acquisition time exactly.  Re-stamping here would corrupt both
        # the visibility metric and downstream freshness checks.
        output.header.stamp = message.header.stamp
        output.header.frame_id = self.output_frame
        output.child_frame_id = self.output_child_frame
        output.pose.pose.position.x = target_pose.x
        output.pose.pose.position.y = target_pose.y
        output.pose.pose.position.z = message.pose.pose.position.z
        output.pose.pose.orientation.x = target_quaternion[0]
        output.pose.pose.orientation.y = target_quaternion[1]
        output.pose.pose.orientation.z = target_quaternion[2]
        output.pose.pose.orientation.w = target_quaternion[3]
        output.pose.covariance = target_covariance
        # Twist and its covariance are child-frame quantities.  The child axes
        # have only been given a unique name, so copying is the correct
        # standards-compliant operation.
        output.twist = copy.deepcopy(message.twist)
        self.last_source_stamp = source_stamp
        self.last_valid_receipt = now
        self.last_event_valid = True
        self.last_reason = "READY"
        self.last_logged_reason = None
        # Publish readiness before the corresponding odometry sample.  A
        # fail-closed consumer can then require a fresh READY contract without
        # dropping the first valid sample after startup or recovery.
        self._publish_status()
        self.publisher.publish(output)

    def _publish_status(self) -> None:
        now = self._now()
        input_fresh = (
            self.last_valid_receipt is not None
            and now - self.last_valid_receipt <= self.maximum_input_age
        )
        reason = self.last_reason
        if self.last_event_valid and not input_fresh:
            reason = "STALE_INPUT"
        ready = bool(
            self.alignment_verified
            and self.resolver.initialized
            and self.last_event_valid
            and input_fresh
        )
        alignment = self.resolver.alignment
        status = {
            "ready": ready,
            "reason": reason,
            "input_fresh": input_fresh,
            "last_message_valid": self.last_event_valid,
            "alignment_verified": self.alignment_verified,
            "alignment_mode": self.resolver.mode,
            "alignment_initialized": self.resolver.initialized,
            "source_frame": self.expected_source_frame,
            "source_child_frame": self.expected_source_child_frame,
            "output_frame": self.output_frame,
            "output_child_frame": self.output_child_frame,
            "source_stamp": self.last_source_stamp,
            "alignment": (
                None
                if alignment is None
                else {"x": alignment.tx, "y": alignment.ty, "yaw": alignment.yaw}
            ),
        }
        message = String()
        message.data = json.dumps(status, separators=(",", ":"))
        self.status_publisher.publish(message)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DreamMergerOdometryAdapter()
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
