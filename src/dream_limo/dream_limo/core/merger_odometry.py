"""Fail-closed planar alignment for a second LIMO's local odometry.

The two robots normally start with unrelated odometry origins.  This module
contains the ROS-independent part of the adapter which maps the merger's
local odometry pose into the ego experiment frame.  Twist is deliberately not
transformed here: ``nav_msgs/Odometry.twist`` is expressed in the child frame,
and changing only the odometry parent frame must not rotate it.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, isfinite, pi, sin
from typing import Iterable, Optional, Tuple

import numpy as np


def wrap_angle(angle: float) -> float:
    """Wrap an angle to ``[-pi, pi)``."""
    return (float(angle) + pi) % (2.0 * pi) - pi


@dataclass(frozen=True)
class PlanarPose:
    x: float
    y: float
    yaw: float

    def validate(self, label: str = "pose") -> None:
        if not all(isfinite(value) for value in (self.x, self.y, self.yaw)):
            raise ValueError(f"{label} contains a non-finite value")


@dataclass(frozen=True)
class PlanarAlignment:
    """A fixed ``target_frame <- source_frame`` SE(2) transform."""

    tx: float
    ty: float
    yaw: float

    def __post_init__(self) -> None:
        if not all(isfinite(value) for value in (self.tx, self.ty, self.yaw)):
            raise ValueError("alignment contains a non-finite value")

    @classmethod
    def from_pose_correspondence(
        cls,
        source: PlanarPose,
        target: PlanarPose,
    ) -> "PlanarAlignment":
        """Build an alignment from one measured source/target pose pair."""
        source.validate("source reference pose")
        target.validate("target reference pose")
        yaw = wrap_angle(target.yaw - source.yaw)
        ch, sh = cos(yaw), sin(yaw)
        tx = target.x - (ch * source.x - sh * source.y)
        ty = target.y - (sh * source.x + ch * source.y)
        return cls(tx=tx, ty=ty, yaw=yaw)

    def transform_pose(self, pose: PlanarPose) -> PlanarPose:
        pose.validate()
        ch, sh = cos(self.yaw), sin(self.yaw)
        return PlanarPose(
            x=self.tx + ch * pose.x - sh * pose.y,
            y=self.ty + sh * pose.x + ch * pose.y,
            yaw=wrap_angle(self.yaw + pose.yaw),
        )


@dataclass(frozen=True)
class PlanarOdometry:
    """ROS-independent odometry subset used to enforce frame semantics."""

    stamp: float
    pose: PlanarPose
    child_twist: Tuple[float, float, float, float, float, float]

    def validate(self) -> None:
        if not isfinite(self.stamp) or self.stamp <= 0.0:
            raise ValueError("odometry timestamp is zero, negative, or non-finite")
        self.pose.validate("odometry pose")
        if len(self.child_twist) != 6:
            raise ValueError("odometry child twist must contain exactly 6 values")
        validate_finite(self.child_twist, label="odometry child twist")


def align_planar_odometry(
    sample: PlanarOdometry,
    alignment: PlanarAlignment,
) -> PlanarOdometry:
    """Align the parent-frame pose while preserving stamp and child twist."""
    sample.validate()
    return PlanarOdometry(
        stamp=sample.stamp,
        pose=alignment.transform_pose(sample.pose),
        child_twist=sample.child_twist,
    )


class AlignmentResolver:
    """Resolve either a surveyed alignment or a first-message anchor.

    ``measured_correspondence`` uses the supplied source and target reference
    poses immediately.  ``first_message_anchor`` latches the first valid source
    pose and maps it onto ``target_reference``.  A resolver never re-anchors;
    recovering from an odometry reset requires restarting the adapter.
    """

    MODES = {"measured_correspondence", "first_message_anchor"}

    def __init__(
        self,
        *,
        mode: str,
        target_reference: PlanarPose,
        source_reference: Optional[PlanarPose] = None,
    ) -> None:
        if mode not in self.MODES:
            choices = ", ".join(sorted(self.MODES))
            raise ValueError(f"alignment mode must be one of: {choices}")
        target_reference.validate("target reference pose")
        self.mode = mode
        self.target_reference = target_reference
        self._alignment: Optional[PlanarAlignment] = None
        if mode == "measured_correspondence":
            if source_reference is None:
                raise ValueError(
                    "measured_correspondence requires a source reference pose"
                )
            self._alignment = PlanarAlignment.from_pose_correspondence(
                source_reference,
                target_reference,
            )
        elif source_reference is not None:
            raise ValueError(
                "first_message_anchor must not have a source reference pose"
            )

    @property
    def alignment(self) -> Optional[PlanarAlignment]:
        return self._alignment

    @property
    def initialized(self) -> bool:
        return self._alignment is not None

    def resolve(self, source_pose: PlanarPose) -> PlanarAlignment:
        source_pose.validate("source pose")
        if self._alignment is None:
            self._alignment = PlanarAlignment.from_pose_correspondence(
                source_pose,
                self.target_reference,
            )
        return self._alignment


def validate_frame_id(frame_id: str, *, label: str) -> str:
    """Validate an unambiguous ROS frame id without silently normalizing it."""
    value = str(frame_id)
    if not value:
        raise ValueError(f"{label} is empty")
    if value != value.strip():
        raise ValueError(f"{label} has leading or trailing whitespace")
    if value.startswith("/"):
        raise ValueError(f"{label} must not start with '/'")
    if any(character.isspace() for character in value):
        raise ValueError(f"{label} contains whitespace")
    return value


def validate_source_frames(
    *,
    actual_parent: str,
    actual_child: str,
    expected_parent: str,
    expected_child: str,
    output_parent: str,
    allow_parent_alias: bool,
) -> None:
    """Reject unexpected or semantically ambiguous odometry frame labels."""
    actual_parent = validate_frame_id(actual_parent, label="source parent frame")
    actual_child = validate_frame_id(actual_child, label="source child frame")
    expected_parent = validate_frame_id(
        expected_parent,
        label="expected source parent frame",
    )
    expected_child = validate_frame_id(
        expected_child,
        label="expected source child frame",
    )
    output_parent = validate_frame_id(output_parent, label="output parent frame")
    if actual_parent != expected_parent:
        raise ValueError(
            f"source parent frame {actual_parent!r} does not match "
            f"expected {expected_parent!r}"
        )
    if actual_child != expected_child:
        raise ValueError(
            f"source child frame {actual_child!r} does not match "
            f"expected {expected_child!r}"
        )
    if actual_parent == actual_child:
        raise ValueError("source parent and child frames are identical")
    if actual_parent == output_parent and not allow_parent_alias:
        raise ValueError(
            "source and output parent frame ids are identical even though their "
            "origins are unrelated; prefix the merger frame or explicitly allow "
            "the alias after verifying topic isolation"
        )


def validate_source_time(
    source_stamp: float,
    now: float,
    *,
    maximum_age: float,
    maximum_future_skew: float,
    previous_stamp: Optional[float] = None,
) -> None:
    """Validate timestamp freshness and monotonicity."""
    values = (source_stamp, now, maximum_age, maximum_future_skew)
    if not all(isfinite(value) for value in values):
        raise ValueError("timestamp validation contains a non-finite value")
    if source_stamp <= 0.0:
        raise ValueError("source timestamp is zero or negative")
    if maximum_age <= 0.0 or maximum_future_skew < 0.0:
        raise ValueError("timestamp tolerances are invalid")
    age = now - source_stamp
    if age > maximum_age:
        raise ValueError(
            f"source odometry is stale ({age:.3f}s > {maximum_age:.3f}s)"
        )
    if age < -maximum_future_skew:
        raise ValueError(
            "source odometry timestamp is too far in the future "
            f"({-age:.3f}s > {maximum_future_skew:.3f}s)"
        )
    if previous_stamp is not None and source_stamp <= previous_stamp:
        raise ValueError("source odometry timestamp is not strictly increasing")


def validate_finite(values: Iterable[float], *, label: str) -> Tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not all(isfinite(value) for value in result):
        raise ValueError(f"{label} contains a non-finite value")
    return result


def rotate_pose_covariance(
    covariance: Iterable[float],
    alignment_yaw: float,
) -> Tuple[float, ...]:
    """Rotate a 6D pose covariance into the target odometry frame.

    Pose covariance is expressed in ``header.frame_id``.  Both the position and
    small-angle orientation blocks therefore rotate with the parent-frame
    alignment.  Twist covariance is not handled here because it remains in the
    unchanged child frame.
    """
    values = validate_finite(covariance, label="pose covariance")
    if len(values) != 36:
        raise ValueError("pose covariance must contain exactly 36 values")
    if not isfinite(alignment_yaw):
        raise ValueError("alignment yaw is non-finite")
    ch, sh = cos(alignment_yaw), sin(alignment_yaw)
    rotation = np.asarray(
        ((ch, -sh, 0.0), (sh, ch, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    jacobian = np.zeros((6, 6), dtype=np.float64)
    jacobian[:3, :3] = rotation
    jacobian[3:, 3:] = rotation
    matrix = np.asarray(values, dtype=np.float64).reshape((6, 6))
    transformed = jacobian @ matrix @ jacobian.T
    return tuple(float(value) for value in transformed.ravel())
