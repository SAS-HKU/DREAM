"""Fail-closed validation and authorization for arbitrary free-space goals.

This module deliberately has no lane, merge-station, or ROS dependency.  A
goal is accepted only when its complete clearance disk lies in cells reported
as observed free by a fresh occupancy grid.  Unknown and every positive cost
are non-free.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import asin, atan2, cos, floor, isfinite, pi, sin
from typing import Optional, Sequence, Tuple


@dataclass(frozen=True)
class FreeGoalConfig:
    frame_id: str = "map"
    footprint_clearance: float = 0.21
    goal_timeout: float = 1.0
    ego_timeout: float = 0.50
    costmap_timeout: float = 0.75
    planner_timeout: float = 0.75
    preflight_timeout: float = 2.0
    future_tolerance: float = 0.10
    quaternion_norm_tolerance: float = 0.02
    maximum_planar_tilt: float = 0.10
    maximum_transform_tilt: float = 1.0e-6
    maximum_absolute_z: float = 0.20
    goal_match_tolerance: float = 1.0e-3

    def __post_init__(self) -> None:
        values = (
            self.footprint_clearance,
            self.goal_timeout,
            self.ego_timeout,
            self.costmap_timeout,
            self.planner_timeout,
            self.preflight_timeout,
            self.future_tolerance,
            self.quaternion_norm_tolerance,
            self.maximum_planar_tilt,
            self.maximum_transform_tilt,
            self.maximum_absolute_z,
            self.goal_match_tolerance,
        )
        if not self.frame_id:
            raise ValueError("free-goal frame cannot be empty")
        if not all(isfinite(float(value)) for value in values):
            raise ValueError("free-goal configuration must be finite")
        if self.footprint_clearance < 0.0 or self.future_tolerance < 0.0:
            raise ValueError("clearance and future tolerance must be non-negative")
        positive = (
            self.goal_timeout,
            self.ego_timeout,
            self.costmap_timeout,
            self.planner_timeout,
            self.preflight_timeout,
            self.quaternion_norm_tolerance,
            self.maximum_planar_tilt,
            self.maximum_transform_tilt,
            self.maximum_absolute_z,
            self.goal_match_tolerance,
        )
        if any(value <= 0.0 for value in positive):
            raise ValueError("free-goal timeouts and tolerances must be positive")


@dataclass(frozen=True)
class FreeGoalRequest:
    frame_id: str
    x: float
    y: float
    z: float
    qx: float
    qy: float
    qz: float
    qw: float
    source_stamp: float
    receipt_stamp: float


@dataclass(frozen=True)
class FreeGoalEgoState:
    frame_id: str
    x: float
    y: float
    source_stamp: float
    receipt_stamp: float


@dataclass(frozen=True)
class CostmapSnapshot:
    frame_id: str
    width: int
    height: int
    resolution: float
    origin_x: float
    origin_y: float
    origin_yaw: float
    data: Tuple[int, ...]
    source_stamp: float
    receipt_stamp: float

    @classmethod
    def from_sequence(
        cls,
        *,
        frame_id: str,
        width: int,
        height: int,
        resolution: float,
        origin_x: float,
        origin_y: float,
        origin_yaw: float,
        data: Sequence[int],
        source_stamp: float,
        receipt_stamp: float,
    ) -> "CostmapSnapshot":
        return cls(
            frame_id=str(frame_id),
            width=int(width),
            height=int(height),
            resolution=float(resolution),
            origin_x=float(origin_x),
            origin_y=float(origin_y),
            origin_yaw=float(origin_yaw),
            data=tuple(int(value) for value in data),
            source_stamp=float(source_stamp),
            receipt_stamp=float(receipt_stamp),
        )


@dataclass(frozen=True)
class FreeGoalValidation:
    accepted: bool
    reason: str
    goal_x: Optional[float] = None
    goal_y: Optional[float] = None
    goal_yaw: Optional[float] = None
    goal_source_age: Optional[float] = None
    goal_receipt_age: Optional[float] = None
    ego_source_age: Optional[float] = None
    ego_receipt_age: Optional[float] = None
    costmap_source_age: Optional[float] = None
    costmap_receipt_age: Optional[float] = None
    blocking_cell_x: Optional[int] = None
    blocking_cell_y: Optional[int] = None
    blocking_value: Optional[int] = None


@dataclass(frozen=True)
class FreeGoalPlannerReadiness:
    ready: bool
    goal_x: Optional[float]
    goal_y: Optional[float]
    receipt_stamp: float


@dataclass(frozen=True)
class FreeGoalPreflightReadiness:
    passed: bool
    receipt_stamp: float


@dataclass(frozen=True)
class FreeGoalAuthorization:
    ready: bool
    armed: bool
    reason: str
    ego_source_age: Optional[float] = None
    ego_receipt_age: Optional[float] = None
    costmap_source_age: Optional[float] = None
    costmap_receipt_age: Optional[float] = None
    planner_age: Optional[float] = None
    preflight_age: Optional[float] = None


def _age(
    now: float, stamp: float, timeout: float, future_tolerance: float
) -> Optional[float]:
    if not isfinite(float(stamp)) or float(stamp) <= 0.0:
        return None
    delta = float(now) - float(stamp)
    if delta < -float(future_tolerance) or delta >= float(timeout):
        return None
    return max(0.0, delta)


def _quaternion_roll_pitch_yaw(
    qx: float, qy: float, qz: float, qw: float
) -> tuple[float, float, float]:
    sin_roll_cos_pitch = 2.0 * (qw * qx + qy * qz)
    cos_roll_cos_pitch = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = atan2(sin_roll_cos_pitch, cos_roll_cos_pitch)
    sin_pitch = 2.0 * (qw * qy - qz * qx)
    pitch = asin(max(-1.0, min(1.0, sin_pitch)))
    sin_yaw_cos_pitch = 2.0 * (qw * qz + qx * qy)
    cos_yaw_cos_pitch = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = atan2(sin_yaw_cos_pitch, cos_yaw_cos_pitch)
    return roll, pitch, yaw


def _normalized_angle(value: float) -> float:
    return (float(value) + pi) % (2.0 * pi) - pi


def transform_planar_goal(
    goal: FreeGoalRequest,
    *,
    target_frame: str,
    translation_x: float,
    translation_y: float,
    translation_z: float,
    transform_yaw: float,
    maximum_transform_tilt: float = 0.10,
    transform_roll: float = 0.0,
    transform_pitch: float = 0.0,
) -> FreeGoalRequest:
    """Apply a verified planar target<-source transform to a goal request."""

    values = (
        translation_x,
        translation_y,
        translation_z,
        transform_yaw,
        transform_roll,
        transform_pitch,
        maximum_transform_tilt,
    )
    goal_values = (
        goal.x,
        goal.y,
        goal.z,
        goal.qx,
        goal.qy,
        goal.qz,
        goal.qw,
        goal.source_stamp,
        goal.receipt_stamp,
    )
    if (
        not target_frame
        or not all(isfinite(float(value)) for value in values)
        or not all(isfinite(float(value)) for value in goal_values)
    ):
        raise ValueError("GOAL_TF_INVALID")
    if maximum_transform_tilt <= 0.0:
        raise ValueError("GOAL_TF_TILT_LIMIT_INVALID")
    if (
        abs(float(transform_roll)) > maximum_transform_tilt
        or abs(float(transform_pitch)) > maximum_transform_tilt
    ):
        raise ValueError("GOAL_TF_NOT_PLANAR")
    quaternion_norm = (
        goal.qx * goal.qx
        + goal.qy * goal.qy
        + goal.qz * goal.qz
        + goal.qw * goal.qw
    ) ** 0.5
    if abs(quaternion_norm - 1.0) > 0.02:
        raise ValueError("GOAL_QUATERNION_NOT_NORMALIZED")
    source_roll, source_pitch, source_yaw = _quaternion_roll_pitch_yaw(
        goal.qx, goal.qy, goal.qz, goal.qw
    )
    if (
        abs(source_roll) > maximum_transform_tilt
        or abs(source_pitch) > maximum_transform_tilt
    ):
        raise ValueError("GOAL_ORIENTATION_NOT_PLANAR")
    ch, sh = cos(transform_yaw), sin(transform_yaw)
    target_yaw = _normalized_angle(transform_yaw + source_yaw)
    return FreeGoalRequest(
        frame_id=str(target_frame),
        x=float(translation_x) + ch * float(goal.x) - sh * float(goal.y),
        y=float(translation_y) + sh * float(goal.x) + ch * float(goal.y),
        z=float(translation_z) + float(goal.z),
        qx=0.0,
        qy=0.0,
        qz=sin(0.5 * target_yaw),
        qw=cos(0.5 * target_yaw),
        source_stamp=float(goal.source_stamp),
        receipt_stamp=float(goal.receipt_stamp),
    )


def _costmap_ages(
    costmap: Optional[CostmapSnapshot], *, now: float, config: FreeGoalConfig
) -> tuple[Optional[float], Optional[float]]:
    if costmap is None:
        return None, None
    return (
        _age(
            now,
            costmap.source_stamp,
            config.costmap_timeout,
            config.future_tolerance,
        ),
        _age(
            now,
            costmap.receipt_stamp,
            config.costmap_timeout,
            config.future_tolerance,
        ),
    )


def _costmap_structure_reason(
    costmap: Optional[CostmapSnapshot], *, config: FreeGoalConfig
) -> Optional[str]:
    if costmap is None:
        return "COSTMAP_UNAVAILABLE"
    values = (
        costmap.resolution,
        costmap.origin_x,
        costmap.origin_y,
        costmap.origin_yaw,
        costmap.source_stamp,
        costmap.receipt_stamp,
    )
    if costmap.frame_id != config.frame_id:
        return "COSTMAP_FRAME_MISMATCH"
    if costmap.width < 1 or costmap.height < 1:
        return "COSTMAP_DIMENSIONS_INVALID"
    if not all(isfinite(float(value)) for value in values):
        return "COSTMAP_NONFINITE"
    if costmap.resolution <= 0.0:
        return "COSTMAP_RESOLUTION_INVALID"
    if abs(costmap.origin_yaw) > 1.0e-6:
        return "COSTMAP_ORIGIN_NOT_AXIS_ALIGNED"
    if len(costmap.data) != costmap.width * costmap.height:
        return "COSTMAP_PAYLOAD_SIZE_MISMATCH"
    return None


def footprint_free_check(
    x: float,
    y: float,
    costmap: CostmapSnapshot,
    *,
    clearance: float,
) -> tuple[bool, str, Optional[int], Optional[int], Optional[int]]:
    """Require every grid cell touched by a clearance disk to equal FREE (0)."""

    values = (x, y, clearance)
    if not all(isfinite(float(value)) for value in values) or clearance < 0.0:
        return False, "GOAL_CLEARANCE_INVALID", None, None, None
    map_x_max = costmap.origin_x + costmap.width * costmap.resolution
    map_y_max = costmap.origin_y + costmap.height * costmap.resolution
    epsilon = 1.0e-12
    if (
        x - clearance < costmap.origin_x - epsilon
        or x + clearance > map_x_max + epsilon
        or y - clearance < costmap.origin_y - epsilon
        or y + clearance > map_y_max + epsilon
    ):
        return False, "GOAL_FOOTPRINT_OUTSIDE_COSTMAP", None, None, None

    resolution = costmap.resolution
    min_ix = max(0, int(floor((x - clearance - costmap.origin_x) / resolution)))
    max_ix = min(
        costmap.width - 1,
        int(floor((x + clearance - costmap.origin_x) / resolution)),
    )
    min_iy = max(0, int(floor((y - clearance - costmap.origin_y) / resolution)))
    max_iy = min(
        costmap.height - 1,
        int(floor((y + clearance - costmap.origin_y) / resolution)),
    )

    # A zero-radius goal still owns its containing cell.
    if clearance == 0.0:
        min_ix = max_ix = min(
            costmap.width - 1,
            max(0, int(floor((x - costmap.origin_x) / resolution))),
        )
        min_iy = max_iy = min(
            costmap.height - 1,
            max(0, int(floor((y - costmap.origin_y) / resolution))),
        )

    radius_squared = clearance * clearance
    for iy in range(min_iy, max_iy + 1):
        cell_y0 = costmap.origin_y + iy * resolution
        cell_y1 = cell_y0 + resolution
        nearest_y = min(max(y, cell_y0), cell_y1)
        for ix in range(min_ix, max_ix + 1):
            cell_x0 = costmap.origin_x + ix * resolution
            cell_x1 = cell_x0 + resolution
            nearest_x = min(max(x, cell_x0), cell_x1)
            intersects = (
                (nearest_x - x) ** 2 + (nearest_y - y) ** 2
                <= radius_squared + epsilon
            )
            if not intersects:
                continue
            value = int(costmap.data[iy * costmap.width + ix])
            if value != 0:
                reason = "GOAL_IN_UNKNOWN" if value < 0 else "GOAL_NOT_FREE"
                return False, reason, ix, iy, value
    return True, "GOAL_FOOTPRINT_FREE", None, None, None


def validate_free_goal_request(
    goal: FreeGoalRequest,
    ego: Optional[FreeGoalEgoState],
    costmap: Optional[CostmapSnapshot],
    *,
    now: float,
    config: FreeGoalConfig,
) -> FreeGoalValidation:
    """Validate an arbitrary map goal without lane or distance assumptions."""

    if not isfinite(float(now)):
        return FreeGoalValidation(False, "INVALID_CURRENT_TIME")
    if goal.frame_id != config.frame_id:
        return FreeGoalValidation(False, "GOAL_FRAME_MISMATCH")
    goal_values = (
        goal.x,
        goal.y,
        goal.z,
        goal.qx,
        goal.qy,
        goal.qz,
        goal.qw,
        goal.source_stamp,
        goal.receipt_stamp,
    )
    if not all(isfinite(float(value)) for value in goal_values):
        return FreeGoalValidation(False, "NONFINITE_GOAL")
    goal_receipt_age = _age(
        now, goal.receipt_stamp, config.goal_timeout, config.future_tolerance
    )
    if goal_receipt_age is None:
        return FreeGoalValidation(False, "STALE_GOAL_RECEIPT")
    goal_source_age = _age(
        now, goal.source_stamp, config.goal_timeout, config.future_tolerance
    )
    if goal_source_age is None:
        return FreeGoalValidation(
            False, "STALE_GOAL_SOURCE", goal_receipt_age=goal_receipt_age
        )

    quaternion_norm = (
        goal.qx * goal.qx
        + goal.qy * goal.qy
        + goal.qz * goal.qz
        + goal.qw * goal.qw
    ) ** 0.5
    if abs(quaternion_norm - 1.0) > config.quaternion_norm_tolerance:
        return FreeGoalValidation(
            False,
            "GOAL_QUATERNION_NOT_NORMALIZED",
            goal_source_age=goal_source_age,
            goal_receipt_age=goal_receipt_age,
        )
    roll, pitch, yaw = _quaternion_roll_pitch_yaw(
        goal.qx, goal.qy, goal.qz, goal.qw
    )
    if abs(roll) > config.maximum_planar_tilt or abs(pitch) > config.maximum_planar_tilt:
        return FreeGoalValidation(
            False,
            "GOAL_ORIENTATION_NOT_PLANAR",
            goal_source_age=goal_source_age,
            goal_receipt_age=goal_receipt_age,
        )
    if abs(goal.z) > config.maximum_absolute_z:
        return FreeGoalValidation(
            False,
            "GOAL_OUTSIDE_MAP_PLANE",
            goal_source_age=goal_source_age,
            goal_receipt_age=goal_receipt_age,
        )

    if ego is None:
        return FreeGoalValidation(
            False,
            "EGO_UNAVAILABLE",
            goal_source_age=goal_source_age,
            goal_receipt_age=goal_receipt_age,
        )
    ego_values = (ego.x, ego.y, ego.source_stamp, ego.receipt_stamp)
    if ego.frame_id != config.frame_id:
        return FreeGoalValidation(False, "EGO_FRAME_MISMATCH")
    if not all(isfinite(float(value)) for value in ego_values):
        return FreeGoalValidation(False, "NONFINITE_EGO")
    ego_source_age = _age(
        now, ego.source_stamp, config.ego_timeout, config.future_tolerance
    )
    ego_receipt_age = _age(
        now, ego.receipt_stamp, config.ego_timeout, config.future_tolerance
    )
    if ego_source_age is None:
        return FreeGoalValidation(False, "STALE_EGO_SOURCE")
    if ego_receipt_age is None:
        return FreeGoalValidation(False, "STALE_EGO_RECEIPT")

    structure_reason = _costmap_structure_reason(costmap, config=config)
    if structure_reason is not None:
        return FreeGoalValidation(False, structure_reason)
    assert costmap is not None
    costmap_source_age, costmap_receipt_age = _costmap_ages(
        costmap, now=now, config=config
    )
    if costmap_source_age is None:
        return FreeGoalValidation(False, "STALE_COSTMAP_SOURCE")
    if costmap_receipt_age is None:
        return FreeGoalValidation(False, "STALE_COSTMAP_RECEIPT")
    free, reason, cell_x, cell_y, value = footprint_free_check(
        goal.x,
        goal.y,
        costmap,
        clearance=config.footprint_clearance,
    )
    if not free:
        return FreeGoalValidation(
            False,
            reason,
            goal_source_age=goal_source_age,
            goal_receipt_age=goal_receipt_age,
            ego_source_age=ego_source_age,
            ego_receipt_age=ego_receipt_age,
            costmap_source_age=costmap_source_age,
            costmap_receipt_age=costmap_receipt_age,
            blocking_cell_x=cell_x,
            blocking_cell_y=cell_y,
            blocking_value=value,
        )
    return FreeGoalValidation(
        True,
        "GOAL_ACCEPTED",
        goal_x=float(goal.x),
        goal_y=float(goal.y),
        goal_yaw=_normalized_angle(yaw),
        goal_source_age=goal_source_age,
        goal_receipt_age=goal_receipt_age,
        ego_source_age=ego_source_age,
        ego_receipt_age=ego_receipt_age,
        costmap_source_age=costmap_source_age,
        costmap_receipt_age=costmap_receipt_age,
    )


class FreeGoalMissionLatch:
    """Replaceable accepted goal plus process-lifetime external-stop latch."""

    def __init__(self) -> None:
        self.accepted_goal: Optional[FreeGoalValidation] = None
        self.last_validation: Optional[FreeGoalValidation] = None
        self.stop_latched = False
        self.mission_complete = False
        self.reason = "WAITING_FOR_GOAL"
        self.revision = 0

    @property
    def active(self) -> bool:
        return bool(
            self.accepted_goal is not None
            and not self.stop_latched
            and not self.mission_complete
        )

    def consider(self, validation: FreeGoalValidation) -> bool:
        self.last_validation = validation
        if self.stop_latched:
            self.reason = "STOP_LATCHED"
            return False
        if not validation.accepted:
            # A newly submitted but invalid goal must cancel the preceding
            # mission.  Continuing toward an older destination after the
            # operator believes they replaced it is unsafe and surprising.
            self.accepted_goal = None
            self.mission_complete = False
            self.reason = validation.reason
            return False
        self.accepted_goal = validation
        self.mission_complete = False
        self.revision += 1
        self.reason = "GOAL_ACCEPTED"
        return True

    def complete(self) -> None:
        self.mission_complete = True
        self.reason = "MISSION_COMPLETE"

    def stop(self) -> None:
        self.stop_latched = True
        self.reason = "STOP_MISSION_REQUESTED"


def evaluate_free_goal_authorization(
    state: FreeGoalMissionLatch,
    ego: Optional[FreeGoalEgoState],
    costmap: Optional[CostmapSnapshot],
    planner: Optional[FreeGoalPlannerReadiness],
    preflight: Optional[FreeGoalPreflightReadiness],
    *,
    now: float,
    config: FreeGoalConfig,
    enabled: bool = True,
) -> FreeGoalAuthorization:
    """Evaluate the held arm heartbeat for the current replaceable goal."""

    if not isfinite(float(now)):
        return FreeGoalAuthorization(False, False, "INVALID_CURRENT_TIME")
    if not enabled:
        return FreeGoalAuthorization(False, False, "DISABLED")
    if state.stop_latched:
        return FreeGoalAuthorization(False, False, "STOP_LATCHED")
    if state.mission_complete:
        return FreeGoalAuthorization(False, False, "MISSION_COMPLETE")
    goal = state.accepted_goal
    if goal is None:
        return FreeGoalAuthorization(False, False, state.reason)
    if ego is None or ego.frame_id != config.frame_id:
        return FreeGoalAuthorization(False, False, "STALE_EGO")
    ego_source_age = _age(
        now, ego.source_stamp, config.ego_timeout, config.future_tolerance
    )
    ego_receipt_age = _age(
        now, ego.receipt_stamp, config.ego_timeout, config.future_tolerance
    )
    if ego_source_age is None or ego_receipt_age is None:
        return FreeGoalAuthorization(False, False, "STALE_EGO")

    structure_reason = _costmap_structure_reason(costmap, config=config)
    if structure_reason is not None:
        return FreeGoalAuthorization(
            False,
            False,
            structure_reason,
            ego_source_age,
            ego_receipt_age,
        )
    assert costmap is not None
    costmap_source_age, costmap_receipt_age = _costmap_ages(
        costmap, now=now, config=config
    )
    if costmap_source_age is None or costmap_receipt_age is None:
        return FreeGoalAuthorization(
            False,
            False,
            "STALE_COSTMAP",
            ego_source_age,
            ego_receipt_age,
            costmap_source_age,
            costmap_receipt_age,
        )
    free, reason, _, _, _ = footprint_free_check(
        float(goal.goal_x),
        float(goal.goal_y),
        costmap,
        clearance=config.footprint_clearance,
    )
    if not free:
        return FreeGoalAuthorization(
            False,
            False,
            reason,
            ego_source_age,
            ego_receipt_age,
            costmap_source_age,
            costmap_receipt_age,
        )

    preflight_age = (
        None
        if preflight is None
        else _age(
            now,
            preflight.receipt_stamp,
            config.preflight_timeout,
            config.future_tolerance,
        )
    )
    if preflight_age is None or not preflight.passed:
        return FreeGoalAuthorization(
            False,
            False,
            "WAITING_FOR_PREFLIGHT",
            ego_source_age,
            ego_receipt_age,
            costmap_source_age,
            costmap_receipt_age,
            preflight_age=preflight_age,
        )

    planner_age = (
        None
        if planner is None
        else _age(
            now,
            planner.receipt_stamp,
            config.planner_timeout,
            config.future_tolerance,
        )
    )
    planner_matches = bool(
        planner is not None
        and planner.ready
        and planner.goal_x is not None
        and planner.goal_y is not None
        and isfinite(float(planner.goal_x))
        and isfinite(float(planner.goal_y))
        and abs(float(planner.goal_x) - float(goal.goal_x))
        <= config.goal_match_tolerance
        and abs(float(planner.goal_y) - float(goal.goal_y))
        <= config.goal_match_tolerance
    )
    if planner_age is None or not planner_matches:
        return FreeGoalAuthorization(
            False,
            False,
            "WAITING_FOR_PLANNER",
            ego_source_age,
            ego_receipt_age,
            costmap_source_age,
            costmap_receipt_age,
            planner_age,
            preflight_age,
        )
    return FreeGoalAuthorization(
        True,
        True,
        "GOAL_ACTIVE",
        ego_source_age,
        ego_receipt_age,
        costmap_source_age,
        costmap_receipt_age,
        planner_age,
        preflight_age,
    )
