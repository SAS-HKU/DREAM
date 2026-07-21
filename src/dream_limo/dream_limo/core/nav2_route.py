"""Pure validation contracts for the planner-only Nav2 route provider."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, hypot, isfinite, sin, sqrt
from typing import Optional, Sequence, Tuple


@dataclass(frozen=True)
class FreshnessResult:
    """Result of validating one ROS source or receipt timestamp."""

    valid: bool
    reason: str
    age: Optional[float] = None


@dataclass(frozen=True)
class PathValidation:
    """Result of validating a geometric path before it crosses the ROS boundary."""

    valid: bool
    reason: str
    pose_count: int = 0
    source_age: Optional[float] = None
    receipt_age: Optional[float] = None


def goal_identity_matches(
    *,
    actual_x: float,
    actual_y: float,
    actual_yaw: float,
    actual_stamp: float,
    expected_x: float,
    expected_y: float,
    expected_yaw: float,
    expected_stamp: float,
    position_tolerance: float,
    identity_tolerance: float,
) -> bool:
    """Match the complete immutable pose goal, including its revision stamp."""

    values = (
        actual_x,
        actual_y,
        actual_yaw,
        actual_stamp,
        expected_x,
        expected_y,
        expected_yaw,
        expected_stamp,
        position_tolerance,
        identity_tolerance,
    )
    if (
        not all(isfinite(float(value)) for value in values)
        or position_tolerance < 0.0
        or identity_tolerance < 0.0
        or actual_stamp <= 0.0
        or expected_stamp <= 0.0
    ):
        return False
    yaw_error = atan2(
        sin(float(actual_yaw) - float(expected_yaw)),
        cos(float(actual_yaw) - float(expected_yaw)),
    )
    return bool(
        hypot(
            float(actual_x) - float(expected_x),
            float(actual_y) - float(expected_y),
        )
        <= position_tolerance
        and abs(yaw_error) <= identity_tolerance
        and abs(float(actual_stamp) - float(expected_stamp))
        <= identity_tolerance
    )


def validate_freshness(
    stamp: float,
    *,
    now: float,
    maximum_age: float,
    future_tolerance: float,
    label: str,
) -> FreshnessResult:
    """Validate a positive timestamp against a bounded clock window."""
    values = (stamp, now, maximum_age, future_tolerance)
    if not all(isfinite(float(value)) for value in values):
        return FreshnessResult(False, f"{label}_TIME_NONFINITE")
    if maximum_age <= 0.0 or future_tolerance < 0.0:
        return FreshnessResult(False, f"{label}_TIMING_INVALID")
    if stamp <= 0.0:
        return FreshnessResult(False, f"{label}_STAMP_INVALID")
    age = float(now) - float(stamp)
    if age < -float(future_tolerance):
        return FreshnessResult(False, f"{label}_STAMP_FUTURE", age)
    if age >= float(maximum_age):
        return FreshnessResult(False, f"{label}_STALE", age)
    return FreshnessResult(True, "ok", max(0.0, age))


def validate_planar_pose(
    *,
    frame_id: str,
    expected_frame: str,
    position_xyz: Sequence[float],
    quaternion_xyzw: Sequence[float],
    label: str,
    quaternion_norm_tolerance: float = 0.02,
) -> str:
    """Return ``ok`` for a finite, normalized pose in the expected frame."""
    if frame_id != expected_frame:
        return f"{label}_FRAME_MISMATCH"
    position = tuple(float(value) for value in position_xyz)
    quaternion = tuple(float(value) for value in quaternion_xyzw)
    if len(position) != 3 or len(quaternion) != 4:
        return f"{label}_SHAPE_INVALID"
    if not all(isfinite(value) for value in (*position, *quaternion)):
        return f"{label}_NONFINITE"
    if not isfinite(quaternion_norm_tolerance) or quaternion_norm_tolerance <= 0.0:
        return f"{label}_QUATERNION_TOLERANCE_INVALID"
    norm = sqrt(sum(value * value for value in quaternion))
    if abs(norm - 1.0) > quaternion_norm_tolerance:
        return f"{label}_QUATERNION_INVALID"
    return "ok"


def validate_transform_sample(
    *,
    parent_frame: str,
    child_frame: str,
    expected_parent: str,
    expected_child: str,
    translation_xyz: Sequence[float],
    quaternion_xyzw: Sequence[float],
    source_stamp: float,
    now: float,
    maximum_age: float,
    future_tolerance: float,
) -> FreshnessResult:
    """Validate the current world-to-robot transform used as the plan start."""
    if child_frame != expected_child:
        return FreshnessResult(False, "TF_CHILD_FRAME_MISMATCH")
    pose_reason = validate_planar_pose(
        frame_id=parent_frame,
        expected_frame=expected_parent,
        position_xyz=translation_xyz,
        quaternion_xyzw=quaternion_xyzw,
        label="TF",
    )
    if pose_reason != "ok":
        return FreshnessResult(False, pose_reason)
    return validate_freshness(
        source_stamp,
        now=now,
        maximum_age=maximum_age,
        future_tolerance=future_tolerance,
        label="TF",
    )


def validate_geometric_path(
    *,
    frame_id: str,
    pose_frames: Sequence[str],
    positions_xyz: Sequence[Sequence[float]],
    quaternions_xyzw: Sequence[Sequence[float]],
    source_stamp: float,
    receipt_stamp: float,
    now: float,
    expected_frame: str,
    source_timeout: float,
    receipt_timeout: float,
    future_tolerance: float,
) -> PathValidation:
    """Validate a successful Nav2 result before latching it for DREAM."""
    count = len(positions_xyz)
    if count == 0:
        return PathValidation(False, "PATH_EMPTY")
    if len(pose_frames) != count or len(quaternions_xyzw) != count:
        return PathValidation(False, "PATH_PAYLOAD_SIZE_MISMATCH", count)
    if frame_id != expected_frame:
        return PathValidation(False, "PATH_FRAME_MISMATCH", count)
    for pose_frame, position, quaternion in zip(
        pose_frames,
        positions_xyz,
        quaternions_xyzw,
    ):
        # Pose headers are commonly empty in nav_msgs/Path. When populated,
        # they must agree with the path's world frame.
        effective_frame = expected_frame if pose_frame == "" else pose_frame
        reason = validate_planar_pose(
            frame_id=effective_frame,
            expected_frame=expected_frame,
            position_xyz=position,
            quaternion_xyzw=quaternion,
            label="PATH_POSE",
        )
        if reason != "ok":
            return PathValidation(False, reason, count)

    source = validate_freshness(
        source_stamp,
        now=now,
        maximum_age=source_timeout,
        future_tolerance=future_tolerance,
        label="PATH_SOURCE",
    )
    if not source.valid:
        return PathValidation(False, source.reason, count, source.age)
    receipt = validate_freshness(
        receipt_stamp,
        now=now,
        maximum_age=receipt_timeout,
        future_tolerance=future_tolerance,
        label="PATH_RECEIPT",
    )
    if not receipt.valid:
        return PathValidation(
            False,
            receipt.reason,
            count,
            source.age,
            receipt.age,
        )
    return PathValidation(True, "PATH_VALID", count, source.age, receipt.age)


def path_message_values(path) -> Tuple[Tuple[str, ...], Tuple[tuple, ...], Tuple[tuple, ...]]:
    """Extract primitive path values without making the pure contract ROS-specific."""
    frames = []
    positions = []
    quaternions = []
    for pose_stamped in path.poses:
        pose = pose_stamped.pose
        frames.append(str(pose_stamped.header.frame_id))
        positions.append(
            (
                float(pose.position.x),
                float(pose.position.y),
                float(pose.position.z),
            )
        )
        quaternions.append(
            (
                float(pose.orientation.x),
                float(pose.orientation.y),
                float(pose.orientation.z),
                float(pose.orientation.w),
            )
        )
    return tuple(frames), tuple(positions), tuple(quaternions)
