"""Small ROS conversion helpers shared by DREAM nodes."""

from __future__ import annotations

from math import atan2, cos, sin
from typing import Any, Dict, Mapping, Tuple

import numpy as np

from .core.types import EgoState, Vehicle


def stamp_to_seconds(stamp: Any) -> float:
    return float(stamp.sec) + 1.0e-9 * float(stamp.nanosec)


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
