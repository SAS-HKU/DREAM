"""Small ROS conversion helpers shared by DREAM nodes."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, sin
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
from geometry_msgs.msg import TwistStamped

from .core.command_adapter import VelocityCommand
from .core.types import EgoState, Vehicle


def stamp_to_seconds(stamp: Any) -> float:
    return float(stamp.sec) + 1.0e-9 * float(stamp.nanosec)


@dataclass(frozen=True)
class ControlSourceStamp:
    """Exact planner-command identity preserved across internal safety hops."""

    sec: int
    nanosec: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.sec, bool)
            or not isinstance(self.sec, int)
            or isinstance(self.nanosec, bool)
            or not isinstance(self.nanosec, int)
            or self.sec < 0
            or not 0 <= self.nanosec < 1_000_000_000
            or (self.sec == 0 and self.nanosec == 0)
        ):
            raise ValueError("control source stamp must be a nonzero ROS time")

    @classmethod
    def from_ros_stamp(cls, stamp: Any) -> "ControlSourceStamp":
        return cls(sec=stamp.sec, nanosec=stamp.nanosec)

    @classmethod
    def from_mapping(cls, value: Any) -> "ControlSourceStamp":
        if not isinstance(value, Mapping):
            raise ValueError("control source stamp payload must be a mapping")
        try:
            return cls(sec=value["sec"], nanosec=value["nanosec"])
        except KeyError as exc:
            raise ValueError("control source stamp payload is incomplete") from exc

    def apply_to(self, stamp: Any) -> None:
        stamp.sec = self.sec
        stamp.nanosec = self.nanosec

    def as_mapping(self) -> Dict[str, int]:
        return {"sec": self.sec, "nanosec": self.nanosec}


def velocity_command_from_stamped_twist(
    message: TwistStamped,
    *,
    malformed_reason: str,
    expected_frame: str = "base_link",
) -> tuple[VelocityCommand, Optional[ControlSourceStamp]]:
    """Decode one internal command and reject missing identity or extra axes."""

    twist = message.twist
    expected_zero = (
        float(twist.linear.y),
        float(twist.linear.z),
        float(twist.angular.x),
        float(twist.angular.y),
    )
    values = (
        float(twist.linear.x),
        float(twist.angular.z),
        *expected_zero,
    )
    try:
        source_stamp = ControlSourceStamp.from_ros_stamp(message.header.stamp)
    except (AttributeError, TypeError, ValueError):
        source_stamp = None
    valid = bool(
        message.header.frame_id == expected_frame
        and source_stamp is not None
        and all(np.isfinite(value) for value in values)
        and all(abs(value) <= 1.0e-9 for value in expected_zero)
    )
    if not valid:
        return VelocityCommand.zero(malformed_reason), None
    return (
        VelocityCommand(values[0], values[1], True, "ok"),
        source_stamp,
    )


def stamped_twist_from_velocity_command(
    command: VelocityCommand,
    source_stamp: Optional[ControlSourceStamp],
    *,
    frame_id: str = "base_link",
) -> TwistStamped:
    """Encode a valid internal command with its unchanged planner identity."""

    message = TwistStamped()
    if not command.valid:
        return message
    if source_stamp is None or not frame_id:
        raise ValueError("valid internal commands require an exact source stamp")
    source_stamp.apply_to(message.header.stamp)
    message.header.frame_id = frame_id
    message.twist.linear.x = command.linear_x
    message.twist.angular.z = command.angular_z
    return message


def quaternion_to_yaw(quaternion: Any) -> float:
    return atan2(
        2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
        1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z),
    )


def yaw_to_quaternion(yaw: float) -> Tuple[float, float, float, float]:
    return 0.0, 0.0, sin(0.5 * yaw), cos(0.5 * yaw)


def transform_planar(
    x: float,
    y: float,
    vx: float,
    vy: float,
    *,
    tx: float,
    ty: float,
    yaw: float,
) -> Tuple[float, float, float, float]:
    ch, sh = cos(yaw), sin(yaw)
    return (
        tx + ch * x - sh * y,
        ty + sh * x + ch * y,
        ch * vx - sh * vy,
        sh * vx + ch * vy,
    )


def child_velocity_to_parent(
    longitudinal: float,
    lateral: float,
    *,
    child_yaw: float,
) -> Tuple[float, float]:
    """Rotate a child-frame planar velocity into its odometry parent frame.

    ROS ``nav_msgs/Odometry`` expresses pose in ``header.frame_id`` but twist
    in ``child_frame_id``.  Consumers must perform this rotation before applying
    a transform between odometry parent frames.
    """
    ch, sh = cos(child_yaw), sin(child_yaw)
    return (
        ch * float(longitudinal) - sh * float(lateral),
        sh * float(longitudinal) + ch * float(lateral),
    )


def alignment_from_initial_pose(
    source_x: float,
    source_y: float,
    source_yaw: float,
    *,
    target_x: float,
    target_y: float,
    target_yaw: float,
) -> Tuple[float, float, float]:
    """Return the fixed transform that maps a first odom pose to a mission pose."""
    yaw = float(target_yaw) - float(source_yaw)
    ch, sh = cos(yaw), sin(yaw)
    tx = float(target_x) - (ch * float(source_x) - sh * float(source_y))
    ty = float(target_y) - (sh * float(source_x) + ch * float(source_y))
    return tx, ty, yaw


def ego_from_odometry(message: Any, lane_index: int = 0) -> EgoState:
    pose = message.pose.pose
    twist = message.twist.twist
    speed = float(np.hypot(twist.linear.x, twist.linear.y))
    stamp = stamp_to_seconds(message.header.stamp)
    return EgoState(
        x=float(pose.position.x),
        y=float(pose.position.y),
        yaw=quaternion_to_yaw(pose.orientation),
        speed=speed,
        yaw_rate=float(twist.angular.z),
        stamp=stamp,
        lane_index=lane_index,
    )


def vehicle_from_mapping(raw: Mapping[str, Any]) -> Vehicle:
    return Vehicle(
        vehicle_id=str(raw.get("id", raw.get("vehicle_id", "unknown"))),
        x=float(raw["x"]),
        y=float(raw["y"]),
        vx=float(raw.get("vx", 0.0)),
        vy=float(raw.get("vy", 0.0)),
        heading=float(raw.get("heading", 0.0)),
        vehicle_class=str(raw.get("class", raw.get("vehicle_class", "car"))),
        length=float(raw.get("length", 0.22)),
        width=float(raw.get("width", 0.22)),
        acceleration=float(raw.get("a", raw.get("acceleration", 0.0))),
        confidence=float(raw.get("confidence", 1.0)),
        stamp=float(raw.get("stamp", 0.0)),
    )


def vehicle_to_mapping(vehicle: Vehicle) -> Dict[str, Any]:
    result = vehicle.as_drift_dict()
    result["confidence"] = vehicle.confidence
    result["stamp"] = vehicle.stamp
    return result
