"""Pure validation and one-shot activation for an operator-selected mission goal."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import asin, atan2, cos, hypot, isfinite, pi, sin
from operator import index
from typing import Optional, Sequence, Tuple


@dataclass(frozen=True)
class GoalMissionConfig:
    """Geometry and freshness limits used at the autonomous-motion boundary."""

    frame_id: str
    lane_centers: Tuple[float, ...]
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    footprint_length: float
    footprint_width: float
    collision_inflation_margin: float = 0.05
    grid_resolution: float = 0.025
    lane_tolerance: float = 0.18
    minimum_ahead_distance: float = 0.50
    maximum_stopped_speed: float = 0.03
    goal_timeout: float = 1.00
    ego_timeout: float = 0.50
    future_tolerance: float = 0.10
    lane_change_minimum_goal_x: float = 5.30
    lane_heading: float = 0.0
    quaternion_norm_tolerance: float = 0.02
    maximum_planar_tilt: float = 0.10
    maximum_absolute_z: float = 0.20

    def __post_init__(self) -> None:
        numeric = (
            *self.lane_centers,
            self.x_min,
            self.x_max,
            self.y_min,
            self.y_max,
            self.footprint_length,
            self.footprint_width,
            self.collision_inflation_margin,
            self.grid_resolution,
            self.lane_tolerance,
            self.minimum_ahead_distance,
            self.maximum_stopped_speed,
            self.goal_timeout,
            self.ego_timeout,
            self.future_tolerance,
            self.lane_change_minimum_goal_x,
            self.lane_heading,
            self.quaternion_norm_tolerance,
            self.maximum_planar_tilt,
            self.maximum_absolute_z,
        )
        if not self.frame_id or not self.lane_centers:
            raise ValueError("goal mission frame and lane centers are required")
        if not all(isfinite(float(value)) for value in numeric):
            raise ValueError("goal mission configuration must be finite")
        if self.x_min >= self.x_max or self.y_min >= self.y_max:
            raise ValueError("goal mission bounds are reversed")
        if self.footprint_length <= 0.0 or self.footprint_width <= 0.0:
            raise ValueError("vehicle footprint must be positive")
        positive = (
            self.lane_tolerance,
            self.minimum_ahead_distance,
            self.goal_timeout,
            self.ego_timeout,
            self.quaternion_norm_tolerance,
            self.maximum_planar_tilt,
            self.maximum_absolute_z,
            self.grid_resolution,
        )
        if any(value <= 0.0 for value in positive):
            raise ValueError("goal mission tolerances must be positive")
        if (
            self.maximum_stopped_speed < 0.0
            or self.future_tolerance < 0.0
            or self.collision_inflation_margin < 0.0
        ):
            raise ValueError("goal speed and future tolerance must be non-negative")
        if len(set(self.lane_centers)) != len(self.lane_centers):
            raise ValueError("lane centers must be unique")
        lane_steps = tuple(
            right - left
            for left, right in zip(self.lane_centers, self.lane_centers[1:])
        )
        if lane_steps and not (
            all(step > 0.0 for step in lane_steps)
            or all(step < 0.0 for step in lane_steps)
        ):
            raise ValueError("lane centers must be strictly ordered")


def goal_mission_config_from_deployment(deployment, **overrides) -> GoalMissionConfig:
    """Build the shared validation contract from a deployment configuration.

    ``deployment`` is intentionally duck typed so this pure module does not
    import ROS or the package's configuration loader.  Callers may use keyword
    overrides for parameters explicitly exposed by their ROS node.
    """

    config = GoalMissionConfig(
        frame_id=str(deployment.grid.frame_id),
        lane_centers=tuple(float(value) for value in deployment.arena.lane_centers),
        x_min=float(deployment.grid.x_min),
        x_max=float(deployment.grid.x_max),
        y_min=max(float(deployment.grid.y_min), float(deployment.grid.road_y_min)),
        y_max=min(float(deployment.grid.y_max), float(deployment.grid.road_y_max)),
        footprint_length=float(deployment.mpc.robot_length),
        footprint_width=float(deployment.mpc.robot_width),
        collision_inflation_margin=float(
            deployment.safety.collision_inflation_margin
        ),
        grid_resolution=float(deployment.grid.resolution),
        lane_tolerance=min(0.20, 0.45 * float(deployment.arena.lane_width)),
        maximum_stopped_speed=float(deployment.mpc.mission_stop_speed_tolerance),
        lane_change_minimum_goal_x=max(
            float(deployment.arena.merge_path_x_max),
            float(deployment.arena.conflict_zone_x_max),
        ),
    )
    return replace(config, **overrides) if overrides else config


@dataclass(frozen=True)
class GoalRequest:
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
class EgoMissionState:
    x: float
    y: float
    speed: float
    source_stamp: float
    receipt_stamp: float


@dataclass(frozen=True)
class GoalValidation:
    accepted: bool
    reason: str
    target_lane: Optional[int] = None
    goal_x: Optional[float] = None
    goal_y: Optional[float] = None
    goal_yaw: Optional[float] = None
    goal_source_age: Optional[float] = None
    goal_receipt_age: Optional[float] = None
    ego_source_age: Optional[float] = None
    ego_receipt_age: Optional[float] = None


@dataclass(frozen=True)
class PlannerGoalReadiness:
    ready: bool
    mission_goal_x: Optional[float]
    target_lane: Optional[int]
    receipt_stamp: float


@dataclass(frozen=True)
class PreflightReadiness:
    passed: bool
    receipt_stamp: float


@dataclass(frozen=True)
class GoalAuthorization:
    ready: bool
    armed: bool
    reason: str
    ego_source_age: Optional[float] = None
    ego_receipt_age: Optional[float] = None
    planner_age: Optional[float] = None
    preflight_age: Optional[float] = None


def nearest_lane(
    y: float, lane_centers: Sequence[float], tolerance: float
) -> Optional[int]:
    """Return the unique nearest surveyed lane within ``tolerance``."""
    if not isfinite(float(y)) or not isfinite(float(tolerance)) or tolerance < 0.0:
        return None
    if not lane_centers:
        return None
    distances = [abs(float(y) - float(center)) for center in lane_centers]
    index = min(range(len(distances)), key=distances.__getitem__)
    return index if distances[index] <= tolerance else None


def _age(
    now: float, stamp: float, timeout: float, future_tolerance: float
) -> Optional[float]:
    if not isfinite(float(stamp)) or float(stamp) <= 0.0:
        return None
    delta = float(now) - float(stamp)
    if delta < -float(future_tolerance) or delta >= float(timeout):
        return None
    return max(0.0, delta)


def _quaternion_roll_pitch(goal: GoalRequest) -> Tuple[float, float]:
    sin_roll_cos_pitch = 2.0 * (goal.qw * goal.qx + goal.qy * goal.qz)
    cos_roll_cos_pitch = 1.0 - 2.0 * (goal.qx * goal.qx + goal.qy * goal.qy)
    roll = atan2(sin_roll_cos_pitch, cos_roll_cos_pitch)
    sin_pitch = 2.0 * (goal.qw * goal.qy - goal.qz * goal.qx)
    pitch = asin(max(-1.0, min(1.0, sin_pitch)))
    return roll, pitch


def validate_goal_request(
    goal: GoalRequest,
    ego: Optional[EgoMissionState],
    *,
    now: float,
    config: GoalMissionConfig,
) -> GoalValidation:
    """Validate and snap one RViz-style pose goal, failing closed on ambiguity."""
    if not isfinite(float(now)):
        return GoalValidation(False, "INVALID_CURRENT_TIME")
    if goal.frame_id != config.frame_id:
        return GoalValidation(False, "GOAL_FRAME_MISMATCH")

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
        return GoalValidation(False, "NONFINITE_GOAL")

    goal_receipt_age = _age(
        now, goal.receipt_stamp, config.goal_timeout, config.future_tolerance
    )
    if goal_receipt_age is None:
        return GoalValidation(False, "STALE_GOAL_RECEIPT")
    goal_source_age = _age(
        now, goal.source_stamp, config.goal_timeout, config.future_tolerance
    )
    if goal_source_age is None:
        return GoalValidation(
            False,
            "STALE_GOAL_SOURCE",
            goal_receipt_age=goal_receipt_age,
        )

    quaternion_norm = (
        goal.qx * goal.qx
        + goal.qy * goal.qy
        + goal.qz * goal.qz
        + goal.qw * goal.qw
    ) ** 0.5
    if abs(quaternion_norm - 1.0) > config.quaternion_norm_tolerance:
        return GoalValidation(
            False,
            "GOAL_QUATERNION_NOT_NORMALIZED",
            goal_source_age=goal_source_age,
            goal_receipt_age=goal_receipt_age,
        )
    roll, pitch = _quaternion_roll_pitch(goal)
    if abs(roll) > config.maximum_planar_tilt or abs(pitch) > config.maximum_planar_tilt:
        return GoalValidation(
            False,
            "GOAL_ORIENTATION_NOT_PLANAR",
            goal_source_age=goal_source_age,
            goal_receipt_age=goal_receipt_age,
        )
    if abs(goal.z) > config.maximum_absolute_z:
        return GoalValidation(
            False,
            "GOAL_OUTSIDE_MAP_PLANE",
            goal_source_age=goal_source_age,
            goal_receipt_age=goal_receipt_age,
        )

    footprint_radius = hypot(
        0.5 * config.footprint_length, 0.5 * config.footprint_width
    ) + config.collision_inflation_margin
    # Match the map constraints in RiskAwareMPC, including half-cell
    # quantization allowance, so an accepted goal cannot be solver-infeasible.
    footprint_clearance = footprint_radius - 0.5 * config.grid_resolution
    if not (
        config.x_min + footprint_clearance
        <= goal.x
        <= config.x_max - footprint_clearance
        and config.y_min + footprint_clearance
        <= goal.y
        <= config.y_max - footprint_clearance
    ):
        return GoalValidation(
            False,
            "GOAL_FOOTPRINT_OUT_OF_BOUNDS",
            goal_source_age=goal_source_age,
            goal_receipt_age=goal_receipt_age,
        )
    target_lane = nearest_lane(goal.y, config.lane_centers, config.lane_tolerance)
    if target_lane is None:
        return GoalValidation(
            False,
            "GOAL_NOT_NEAR_LANE",
            goal_source_age=goal_source_age,
            goal_receipt_age=goal_receipt_age,
        )

    if ego is None:
        return GoalValidation(
            False,
            "EGO_UNAVAILABLE",
            target_lane=target_lane,
            goal_source_age=goal_source_age,
            goal_receipt_age=goal_receipt_age,
        )
    ego_values = (ego.x, ego.y, ego.speed, ego.source_stamp, ego.receipt_stamp)
    if not all(isfinite(float(value)) for value in ego_values):
        return GoalValidation(False, "NONFINITE_EGO", target_lane=target_lane)
    ego_receipt_age = _age(
        now, ego.receipt_stamp, config.ego_timeout, config.future_tolerance
    )
    if ego_receipt_age is None:
        return GoalValidation(False, "STALE_EGO_RECEIPT", target_lane=target_lane)
    ego_source_age = _age(
        now, ego.source_stamp, config.ego_timeout, config.future_tolerance
    )
    if ego_source_age is None:
        return GoalValidation(
            False,
            "STALE_EGO_SOURCE",
            target_lane=target_lane,
            ego_receipt_age=ego_receipt_age,
        )
    if abs(ego.speed) > config.maximum_stopped_speed:
        return GoalValidation(
            False,
            "EGO_NOT_STOPPED",
            target_lane=target_lane,
            ego_source_age=ego_source_age,
            ego_receipt_age=ego_receipt_age,
        )
    current_lane = nearest_lane(ego.y, config.lane_centers, config.lane_tolerance)
    if current_lane is None:
        return GoalValidation(False, "EGO_NOT_NEAR_LANE", target_lane=target_lane)
    if abs(target_lane - current_lane) > 1:
        return GoalValidation(False, "NONADJACENT_LANE_GOAL", target_lane=target_lane)
    if goal.x - ego.x < config.minimum_ahead_distance:
        return GoalValidation(False, "GOAL_NOT_FAR_ENOUGH_AHEAD", target_lane=target_lane)
    if target_lane != current_lane and goal.x <= config.lane_change_minimum_goal_x:
        return GoalValidation(
            False, "LANE_CHANGE_GOAL_BEFORE_CONFLICT_EXIT", target_lane=target_lane
        )

    snapped_y = float(config.lane_centers[target_lane])
    # Lane centers were checked above; re-check the snapped vehicle footprint
    # so a malformed deployment cannot authorize an out-of-road trajectory.
    if not (
        config.y_min + footprint_clearance
        <= snapped_y
        <= config.y_max - footprint_clearance
    ):
        return GoalValidation(False, "SNAPPED_GOAL_OUT_OF_BOUNDS", target_lane=target_lane)
    snapped_yaw = (float(config.lane_heading) + pi) % (2.0 * pi) - pi
    return GoalValidation(
        True,
        "GOAL_ACCEPTED",
        target_lane=target_lane,
        goal_x=float(goal.x),
        goal_y=snapped_y,
        goal_yaw=snapped_yaw,
        goal_source_age=goal_source_age,
        goal_receipt_age=goal_receipt_age,
        ego_source_age=ego_source_age,
        ego_receipt_age=ego_receipt_age,
    )


def validate_configured_auto_goal(
    ego: Optional[EgoMissionState],
    *,
    now: float,
    config: GoalMissionConfig,
    mission_goal_x: float,
    target_lane: int,
) -> GoalValidation:
    """Synthesize and validate the configured one-shot autonomous goal.

    The arena configuration remains the only source of mission geometry.  The
    synthesized request deliberately passes through :func:`validate_goal_request`
    so auto-start cannot bypass the stopped-ego, freshness, map-boundary, or
    adjacent-lane checks applied to an operator-selected goal.
    """
    if isinstance(target_lane, bool):
        return GoalValidation(False, "AUTO_TARGET_LANE_INVALID")
    try:
        lane_index = index(target_lane)
    except TypeError:
        return GoalValidation(False, "AUTO_TARGET_LANE_INVALID")
    if lane_index < 0 or lane_index >= len(config.lane_centers):
        return GoalValidation(False, "AUTO_TARGET_LANE_INVALID")
    if not isfinite(float(now)) or float(now) <= 0.0:
        return GoalValidation(False, "INVALID_CURRENT_TIME")

    try:
        goal_x = float(mission_goal_x)
    except (TypeError, ValueError, OverflowError):
        return GoalValidation(False, "NONFINITE_GOAL")

    yaw = float(config.lane_heading)
    request = GoalRequest(
        frame_id=config.frame_id,
        x=goal_x,
        y=float(config.lane_centers[lane_index]),
        z=0.0,
        qx=0.0,
        qy=0.0,
        qz=sin(0.5 * yaw),
        qw=cos(0.5 * yaw),
        source_stamp=float(now),
        receipt_stamp=float(now),
    )
    return validate_goal_request(request, ego, now=now, config=config)


def evaluate_goal_authorization(
    state: "GoalMissionLatch",
    ego: Optional[EgoMissionState],
    planner: Optional[PlannerGoalReadiness],
    preflight: Optional[PreflightReadiness],
    *,
    now: float,
    config: GoalMissionConfig,
    planner_timeout: float,
    preflight_timeout: float,
    enabled: bool = True,
) -> GoalAuthorization:
    """Evaluate the continuous arm heartbeat for an already accepted goal."""
    timing = (now, planner_timeout, preflight_timeout)
    if (
        not all(isfinite(float(value)) for value in timing)
        or planner_timeout <= 0.0
        or preflight_timeout <= 0.0
    ):
        return GoalAuthorization(False, False, "INVALID_AUTHORIZATION_TIMING")
    if not enabled:
        return GoalAuthorization(False, False, "DISABLED")
    if state.stop_latched:
        return GoalAuthorization(False, False, "STOP_LATCHED")
    if state.mission_complete:
        return GoalAuthorization(False, False, "MISSION_COMPLETE")
    goal = state.accepted_goal
    if goal is None:
        return GoalAuthorization(False, False, state.reason)
    if ego is None:
        return GoalAuthorization(False, False, "STALE_EGO")
    ego_source_age = _age(
        now, ego.source_stamp, config.ego_timeout, config.future_tolerance
    )
    ego_receipt_age = _age(
        now, ego.receipt_stamp, config.ego_timeout, config.future_tolerance
    )
    if ego_source_age is None or ego_receipt_age is None:
        return GoalAuthorization(False, False, "STALE_EGO")

    preflight_age = (
        None
        if preflight is None
        else _age(
            now,
            preflight.receipt_stamp,
            preflight_timeout,
            config.future_tolerance,
        )
    )
    if preflight_age is None or not preflight.passed:
        return GoalAuthorization(
            False,
            False,
            "WAITING_FOR_PREFLIGHT",
            ego_source_age,
            ego_receipt_age,
            preflight_age=preflight_age,
        )

    planner_age = (
        None
        if planner is None
        else _age(
            now,
            planner.receipt_stamp,
            planner_timeout,
            config.future_tolerance,
        )
    )
    planner_matches = bool(
        planner is not None
        and planner.ready
        and planner.mission_goal_x is not None
        and isfinite(float(planner.mission_goal_x))
        and abs(float(planner.mission_goal_x) - float(goal.goal_x)) <= 1.0e-3
        and planner.target_lane == goal.target_lane
    )
    if planner_age is None or not planner_matches:
        return GoalAuthorization(
            False,
            False,
            "WAITING_FOR_PLANNER",
            ego_source_age,
            ego_receipt_age,
            planner_age,
            preflight_age,
        )
    return GoalAuthorization(
        True,
        True,
        "GOAL_ACTIVE",
        ego_source_age,
        ego_receipt_age,
        planner_age,
        preflight_age,
    )


class GoalMissionLatch:
    """Process-lifetime one-shot mission activation and stop state."""

    def __init__(self) -> None:
        self.goal_received = False
        self.accepted_goal: Optional[GoalValidation] = None
        self.stop_latched = False
        self.mission_complete = False
        self.reason = "WAITING_FOR_GOAL"
        self.last_validation: Optional[GoalValidation] = None

    @property
    def active(self) -> bool:
        return bool(
            self.accepted_goal is not None
            and not self.stop_latched
            and not self.mission_complete
        )

    def consider(self, validation: GoalValidation) -> bool:
        """Accept at most one valid goal; invalid attempts may be corrected."""
        self.goal_received = True
        self.last_validation = validation
        if self.stop_latched:
            self.reason = "STOP_LATCHED"
            return False
        if self.mission_complete:
            self.reason = "MISSION_COMPLETE"
            return False
        if self.accepted_goal is not None:
            self.reason = "GOAL_ALREADY_ACCEPTED"
            return False
        if not validation.accepted:
            self.reason = validation.reason
            return False
        self.accepted_goal = validation
        self.reason = validation.reason
        return True

    def complete(self) -> None:
        self.mission_complete = True
        self.reason = "MISSION_COMPLETE"

    def stop(self) -> None:
        self.stop_latched = True
        self.reason = "STOP_MISSION_REQUESTED"
