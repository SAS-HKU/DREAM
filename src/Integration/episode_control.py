"""Context-injected IDEAM/PRIDEAM episode control for paired experiments.

This module deliberately does not import either of the top-level uncertainty
scenario scripts.  Those scripts own figures, CLI parsing, scenario state, and
mutable module-level configuration; a paired benchmark needs its controller
state to be explicit and fresh for every scenario/variant arm.

The public factories create fresh IDEAM or PRIDEAM primitives.  The caller is
responsible for evolving the DRIFT field and for providing only the traffic
that is observable by the planner.  In particular, hidden ground-truth actors
must not be put into :class:`LaneTraffic` before the scenario's reveal event.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import copy
import math
import time
import traceback
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from Control.MPC import LMPC
from Control.constraint_params import constraint_params
from Control.utils import clac_last_X
from DecisionMaking.decision import decision
from DecisionMaking.decision_params import decision_params
from DecisionMaking.give_desired_path import repropagate
from DecisionMaking.util import LeaderFollower_Uitl
from DecisionMaking.util_params import util_params
from Integration.prideam_controller import create_prideam_controller
from Model.Dynamical_model import Dynamic
from Model.params import params
from Path.path import coordinate_remapping


_LANE_FROM_GROUP_PREFIX = {"L": 0, "C": 1, "R": 2}
_LANE_LABELS = ("L1", "C1", "R1")


def _copy_last_solution(value: Sequence[Any] | None) -> list[Any] | None:
    """Copy an MPC warm start without sharing NumPy arrays across arms."""
    if value is None:
        return None
    copied: list[Any] = []
    for item in value:
        if isinstance(item, np.ndarray):
            copied.append(item.copy())
        elif isinstance(item, (list, tuple)):
            copied.append(np.asarray(item, dtype=float).copy())
        else:
            copied.append(copy.deepcopy(item))
    return copied


def _as_float_list(values: Sequence[float], *, expected: int | None = None) -> list[float]:
    result = [float(value) for value in values]
    if expected is not None and len(result) != expected:
        raise ValueError(f"Expected {expected} state values, received {len(result)}")
    return result


def _normalise_lane_rows(rows: Any) -> np.ndarray:
    """Return an independent, sorted ``(n, 8)`` IDEAM traffic array."""
    if rows is None:
        return np.zeros((0, 8), dtype=float)
    array = np.asarray(rows, dtype=float)
    if array.size == 0:
        return np.zeros((0, 8), dtype=float)
    if array.size % 8:
        raise ValueError("Lane traffic must use rows of [s, ey, epsi, x, y, psi, vx, a]")
    array = array.reshape(-1, 8).copy()
    return array[np.argsort(array[:, 0])]


@dataclass(frozen=True)
class RoadContext:
    """All road-dependent information required by an IDEAM planning step.

    ``lane_lookup`` receives an ego global pose ``[x, y, heading]`` and must
    return one of ``0`` (left), ``1`` (centre), or ``2`` (right).  Supplying it
    explicitly keeps the controller seam independent from a scenario script's
    module-level road-boundary arrays.
    """

    paths: Mapping[int, Any]
    samples: Mapping[int, Sequence[float]]
    x_lists: Mapping[int, Sequence[float]]
    y_lists: Mapping[int, Sequence[float]]
    lane_lookup: Callable[[Sequence[float]], int]
    boundary: float = 1.0
    dt: float = 0.1
    state_dimension: int = 6
    short_target_gap_m: float = 7.5

    def __post_init__(self) -> None:
        if not callable(self.lane_lookup):
            raise TypeError("lane_lookup must be callable")
        if not math.isfinite(float(self.dt)) or self.dt <= 0.0:
            raise ValueError("dt must be a positive finite value")
        if self.state_dimension <= 0:
            raise ValueError("state_dimension must be positive")
        if self.short_target_gap_m < 0.0:
            raise ValueError("short_target_gap_m cannot be negative")

        for lane in range(3):
            if lane not in self.paths:
                raise ValueError(f"Missing path for lane {lane}")
            if lane not in self.samples or lane not in self.x_lists or lane not in self.y_lists:
                raise ValueError(f"Missing path samples for lane {lane}")
            sample_len = len(self.samples[lane])
            if sample_len == 0:
                raise ValueError(f"Lane {lane} has no path samples")
            if len(self.x_lists[lane]) != sample_len or len(self.y_lists[lane]) != sample_len:
                raise ValueError(f"Lane {lane} sample, x, and y arrays must have equal length")

    def lane_of(self, ego_global: Sequence[float]) -> int:
        lane = int(self.lane_lookup(list(ego_global)))
        if lane not in (0, 1, 2):
            raise ValueError(f"lane_lookup returned invalid lane {lane}")
        return lane

    def path(self, lane: int) -> Any:
        return self.paths[int(lane)]

    def path_data(self, lane: int) -> tuple[Sequence[float], Sequence[float], Sequence[float]]:
        lane = int(lane)
        return self.samples[lane], self.x_lists[lane], self.y_lists[lane]


@dataclass(frozen=True)
class CouplingFlags:
    """Enable the three DREAM-to-IDEAM coupling channels independently."""

    enable_decision_veto: bool = True
    enable_mpc_cost: bool = True
    enable_cbf_modulation: bool = True


@dataclass(frozen=True)
class ManeuverRequest:
    """A route-level lane preference that remains subject to all safety gates.

    This is deliberately not a forced target-lane command.  When active, it
    may select the ordinary IDEAM gap group for the requested adjacent lane
    only after IDEAM's own gap-magnitude and first-hop risk screens accept
    that group.  The existing short-gap/probe guard and the DREAM decision
    veto can still turn the request into a keep-lane action.  The benchmark
    uses it only to give every paired arm the same, logged route objective.
    """

    target_lane: int
    start_time_s: float = 0.0
    end_time_s: float | None = None
    label: str = "route_lane_request"

    def __post_init__(self) -> None:
        if (
            isinstance(self.target_lane, bool)
            or not isinstance(self.target_lane, (int, np.integer))
            or int(self.target_lane) not in (0, 1, 2)
        ):
            raise ValueError("ManeuverRequest.target_lane must be 0, 1, or 2")
        if not math.isfinite(float(self.start_time_s)) or self.start_time_s < 0.0:
            raise ValueError("ManeuverRequest.start_time_s must be finite and non-negative")
        if self.end_time_s is not None:
            if not math.isfinite(float(self.end_time_s)) or self.end_time_s < self.start_time_s:
                raise ValueError("ManeuverRequest.end_time_s must follow start_time_s")
        if not self.label:
            raise ValueError("ManeuverRequest.label must be non-empty")

    def active_at(self, time_s: float) -> bool:
        if time_s + 1.0e-9 < self.start_time_s:
            return False
        return self.end_time_s is None or time_s <= self.end_time_s + 1.0e-9


@dataclass(frozen=True)
class LaneTraffic:
    """Visible traffic supplied to the decision and MPC layers only."""

    left: np.ndarray
    centre: np.ndarray
    right: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "left", _normalise_lane_rows(self.left))
        object.__setattr__(self, "centre", _normalise_lane_rows(self.centre))
        object.__setattr__(self, "right", _normalise_lane_rows(self.right))

    @classmethod
    def from_arrays(cls, left: Any, centre: Any, right: Any) -> "LaneTraffic":
        return cls(left=left, centre=centre, right=right)


class _RoadAwareLeaderFollowerUtil(LeaderFollower_Uitl):
    """IDEAM utility with its two path lookups bound to ``RoadContext``.

    ``LeaderFollower_Uitl`` otherwise reads ``Path.path.get_path_info`` at
    runtime.  Overriding these methods keeps the original gap grouping and
    constraint calculations intact while preventing a custom benchmark road
    from accidentally being evaluated against the legacy three-lane loop.
    """

    def __init__(self, road: RoadContext, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._road = road

    def _path_info(self, lane: int) -> tuple[Any, Sequence[float], Sequence[float], Sequence[float]]:
        sample, x_values, y_values = self._road.path_data(lane)
        return self._road.path(lane), x_values, y_values, sample

    def get_alllane_lf(
        self,
        path_ego: Any,
        X0_g: Sequence[float],
        path_now: int,
        vehicle_left: np.ndarray,
        vehicle_centre: np.ndarray,
        vehicle_right: np.ndarray,
    ) -> Any:
        # ``path_ego`` is retained for the parent API, but the context is the
        # single source of truth for its geometry and sampled representation.
        del path_ego
        path, x_values, y_values, sample = self._path_info(int(path_now))
        se, _ = self.coordinate_remapping(path, x_values, y_values, sample, X0_g[0:2])
        theta = path.get_theta_r(se)
        forward_vector = [np.cos(theta), np.sin(theta)]
        xy_ego = [X0_g[0], X0_g[1]]

        if path_now == 0:
            lf_ego, proj_ego, proje_sey = self.get_egolane_lf(vehicle_left, se)
            lf_right, proj_right, projr_sey = self.get_onelane_lf(
                vehicle_centre, xy_ego, forward_vector, path, x_values, y_values, sample, se
            )
            lf_most_right, proj_most_right, projmr_sey = self.get_onelane_lf(
                vehicle_right, xy_ego, forward_vector, path, x_values, y_values, sample, se
            )
            return (
                lf_ego, lf_right, lf_most_right, proj_ego, proj_right,
                proj_most_right, proje_sey, projr_sey, projmr_sey,
            )
        if path_now == 1:
            lf_ego, proj_ego, proje_sey = self.get_egolane_lf(vehicle_centre, se)
            lf_left, proj_left, projl_sey = self.get_onelane_lf(
                vehicle_left, xy_ego, forward_vector, path, x_values, y_values, sample, se
            )
            lf_right, proj_right, projr_sey = self.get_onelane_lf(
                vehicle_right, xy_ego, forward_vector, path, x_values, y_values, sample, se
            )
            return (
                lf_ego, lf_left, lf_right, proj_ego, proj_left, proj_right,
                proje_sey, projl_sey, projr_sey,
            )
        if path_now == 2:
            lf_ego, proj_ego, proje_sey = self.get_egolane_lf(vehicle_right, se)
            lf_left, proj_left, projl_sey = self.get_onelane_lf(
                vehicle_centre, xy_ego, forward_vector, path, x_values, y_values, sample, se
            )
            lf_most_left, proj_most_left, projml_sey = self.get_onelane_lf(
                vehicle_left, xy_ego, forward_vector, path, x_values, y_values, sample, se
            )
            return (
                lf_ego, lf_left, lf_most_left, proj_ego, proj_left,
                proj_most_left, proje_sey, projl_sey, projml_sey,
            )
        raise ValueError(f"Invalid current lane {path_now}")

    def get_remap_vehicles(
        self,
        x0_g_l: Sequence[float | None],
        prediction_vl_ego: Sequence[float] | None,
        path_dindex: int,
        path_d: Any,
    ) -> np.ndarray | None:
        """Use context samples for the base class's constraint remapping."""
        if x0_g_l[0] is None or prediction_vl_ego is None:
            return None
        _, x_values, y_values, sample = self._path_info(int(path_dindex))
        sl_ego_remap, eyl_ego_remap = self.coordinate_remapping(
            path_d, x_values, y_values, sample, x0_g_l
        )
        prediction_ahead = np.zeros((2, self.T + 1))
        prediction_ahead[0, 0] = sl_ego_remap
        prediction_ahead[1, 0] = eyl_ego_remap
        for index in range(1, self.T + 1):
            prediction_ahead[0, index] = (
                prediction_vl_ego[index - 1] * self.dt + prediction_ahead[0, index - 1]
            )
            prediction_ahead[1, index] = eyl_ego_remap
        return prediction_ahead


@dataclass
class PlannerState:
    """Mutable per-arm ego/MPC state; never share an instance across variants."""

    X0: list[float]
    X0_g: list[float]
    oa: Sequence[float] | float | None = 0.0
    od: Sequence[float] | float | None = 0.0
    last_X: list[Any] | None = None
    path_changed: int | None = None

    def __post_init__(self) -> None:
        self.X0 = _as_float_list(self.X0)
        self.X0_g = _as_float_list(self.X0_g, expected=3)
        self.last_X = _copy_last_solution(self.last_X)
        if self.path_changed is not None:
            self.path_changed = int(self.path_changed)

    def copy(self) -> "PlannerState":
        return PlannerState(
            X0=list(self.X0),
            X0_g=list(self.X0_g),
            oa=copy.deepcopy(self.oa),
            od=copy.deepcopy(self.od),
            last_X=_copy_last_solution(self.last_X),
            path_changed=self.path_changed,
        )


@dataclass(frozen=True)
class DecisionDiagnostics:
    """Audit record for one control step."""

    current_lane: int
    candidate_lane: int
    final_lane: int
    desired_group: str | None
    natural_desired_group: str | None
    route_request_active: bool
    route_requested_lane: int | None
    route_request_feasible: bool | None
    route_request_selected: bool
    route_request_reason: str | None
    proposed_constraint_mode: str | None
    executed_constraint_mode: str | None
    target_follower_projection_m: float | None
    raw_label: str
    virtual_label: str
    probe_blocked: bool
    veto_evaluated: bool
    vetoed: bool
    veto_score: float | None
    veto_threshold: float | None
    veto_allowed: bool | None
    solver_success: bool
    fallback_used: bool
    fallback_observable: bool
    fallback_reason: str | None
    control_accel: float
    control_steer: float
    decision_time_s: float
    mpc_time_s: float
    error: str | None = None


@dataclass(frozen=True)
class EpisodeStepResult:
    """State snapshot and diagnostics after one completed or fallback step."""

    state: PlannerState
    diagnostics: DecisionDiagnostics


@dataclass
class EpisodeArm:
    """One independently instantiated IDEAM or PRIDEAM evaluation arm.

    Use :func:`create_prideam_episode_arm` or
    :func:`create_ideam_episode_arm` rather than reusing a controller from a
    previous scenario.  ``step`` intentionally exposes no forced-target-lane
    argument: the benchmark must allow the decision module and veto to act.
    """

    road: RoadContext
    state: PlannerState
    controller: Any
    utils: Any
    decision: Any
    dynamics: Any
    coupling: CouplingFlags = field(default_factory=CouplingFlags)
    maneuver_request: ManeuverRequest | None = None
    name: str = "arm"
    step_index: int = 0
    _base_mpc_cost: float | None = field(init=False, default=None, repr=False)
    _base_cbf_modulation: float | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if len(self.state.X0) != self.road.state_dimension:
            raise ValueError(
                "PlannerState.X0 length must match RoadContext.state_dimension "
                f"({self.road.state_dimension})"
            )
        if self.state.path_changed is None:
            self.state.path_changed = self.road.lane_of(self.state.X0_g)

        _set_util(self.controller, self.utils)
        _set_path_curvature(self.controller, self.road.path(self.state.path_changed))

        weights = getattr(self.controller, "weights", None)
        if weights is not None:
            self._base_mpc_cost = float(weights.mpc_cost)
            self._base_cbf_modulation = float(weights.cbf_modulation)

    @property
    def is_prideam(self) -> bool:
        return callable(getattr(self.controller, "solve_with_risk", None))

    def apply_coupling(self) -> None:
        """Set channel weights for this arm without modifying any shared preset."""
        weights = getattr(self.controller, "weights", None)
        if weights is None:
            return
        if self._base_mpc_cost is not None:
            weights.mpc_cost = (
                self._base_mpc_cost if self.coupling.enable_mpc_cost else 0.0
            )
        if self._base_cbf_modulation is not None:
            weights.cbf_modulation = (
                self._base_cbf_modulation if self.coupling.enable_cbf_modulation else 0.0
            )

    def step(self, traffic: LaneTraffic) -> EpisodeStepResult:
        return step_episode_arm(self, traffic)


def _mpc_of(controller: Any) -> Any:
    return getattr(controller, "mpc", controller)


def _set_util(controller: Any, utils: Any) -> None:
    setter = getattr(controller, "set_util", None)
    if callable(setter):
        setter(utils)
        return
    _mpc_of(controller).set_util(utils)


def _set_path_curvature(controller: Any, path: Any) -> None:
    setter = getattr(controller, "get_path_curvature", None)
    if callable(setter):
        setter(path=path)
        return
    _mpc_of(controller).get_path_curvature(path)


def _repropagate(road: RoadContext, lane: int, x0_g: Sequence[float], x0: Sequence[float]) -> list[float]:
    sample, x_values, y_values = road.path_data(lane)
    return list(
        repropagate(
            road.path(lane), sample, x_values, y_values,
            list(x0_g), list(x0),
        )
    )


def _lane_from_group(desired_group: Mapping[str, Any]) -> int:
    name = str(desired_group.get("name", ""))
    lane = _LANE_FROM_GROUP_PREFIX.get(name[:1])
    if lane is None:
        raise ValueError(f"Unable to map desired group {name!r} to a lane")
    return lane


def _group_for_lane(
    group_dict: Mapping[str, Mapping[str, Any]], lane: int
) -> Mapping[str, Any]:
    """Return the normal IDEAM gap group for a route-requested lane.

    ``L1``/``C1``/``R1`` are the same first-choice groups used by the legacy
    route-command adapter.  Crucially, this helper only chooses a desired gap
    group; it does not alter the later probe guard, target-gap rule, veto, or
    MPC constraints.
    """

    prefix = _LANE_LABELS[int(lane)][0]
    for name in (f"{prefix}1", f"{prefix}2"):
        if name in group_dict:
            return group_dict[name]
    raise ValueError(f"No ordinary gap group is available for lane {lane}")


def _route_request_group(
    decision_obj: Any,
    group_dict: Mapping[str, Mapping[str, Any]],
    ego_group: Mapping[str, Any],
    *,
    current_lane: int,
    requested_lane: int,
) -> tuple[Mapping[str, Any] | None, bool, str]:
    """Return a route-preferred group only if IDEAM accepts its first hop.

    A route request is allowed to affect the tactical objective, but it must
    not resurrect a target group that IDEAM itself rejected as too small or
    unsafe.  The later short-gap/probe and DREAM veto gates still execute in
    ``step_episode_arm`` after this helper returns.
    """

    if requested_lane == current_lane:
        return ego_group, True, "already_in_requested_lane"
    if abs(requested_lane - current_lane) != 1:
        return None, False, "requested_lane_not_adjacent"

    candidate = _group_for_lane(group_dict, requested_lane)
    current_name = str(ego_group.get("name"))
    candidate_name = str(candidate.get("name"))
    excluded = set(decision_obj.gap_mag_judge(group_dict, current_name))
    if candidate_name in excluded:
        return None, False, "ideam_gap_magnitude_rejected"
    if not bool(decision_obj.risk_assessment(group_dict, current_name, candidate_name, 1)):
        return None, False, "ideam_first_hop_risk_rejected"
    return candidate, True, "ideam_first_hop_accepted"


def _decision_info(
    road: RoadContext,
    x0: Sequence[float],
    x0_g: Sequence[float],
    desired_group: Mapping[str, Any],
    current_lane: int,
) -> tuple[Any, int, str, Sequence[float], Sequence[float], Sequence[float], list[float]]:
    """Context-safe equivalent of the legacy ``Decision_info`` helper.

    It preserves the existing short-gap rule while selecting paths from the
    supplied context rather than the global ``Path.path`` module variables.
    """
    target_lane = _lane_from_group(desired_group)
    current_state = _repropagate(road, current_lane, x0_g, x0)

    target_leader = desired_group.get("sl")
    target_s = 10000.0 if target_leader is None else float(target_leader[0])
    target_gap_is_short = abs(target_s - float(current_state[3])) <= road.short_target_gap_m

    if target_lane != current_lane and not target_gap_is_short:
        label = "R" if target_lane > current_lane else "L"
        next_state = _repropagate(road, target_lane, x0_g, x0)
        sample, x_values, y_values = road.path_data(target_lane)
        return road.path(target_lane), target_lane, label, sample, x_values, y_values, next_state

    sample, x_values, y_values = road.path_data(current_lane)
    return road.path(current_lane), current_lane, "K", sample, x_values, y_values, current_state


def _project_last_solution(
    road: RoadContext,
    last_x: list[Any],
    old_lane: int,
    new_lane: int,
) -> list[Any]:
    """Project a warm-start trajectory between context-provided lane paths."""
    if last_x is None or last_x[3] is None or last_x[4] is None:
        return _copy_last_solution(last_x) or []

    old_path = road.path(old_lane)
    new_path = road.path(new_lane)
    sample, x_values, y_values = road.path_data(new_lane)
    source_s = np.asarray(last_x[3], dtype=float).reshape(-1)
    source_ey = np.asarray(last_x[4], dtype=float).reshape(-1)
    if source_s.shape != source_ey.shape:
        raise ValueError("MPC warm-start s and ey trajectories must have equal shape")

    projected_s: list[float] = []
    projected_ey: list[float] = []
    for s_value, ey_value in zip(source_s, source_ey):
        global_position = old_path.get_cartesian_coords(float(s_value), float(ey_value))
        s_new, ey_new = coordinate_remapping(
            new_path, x_values, y_values, sample, global_position
        )
        projected_s.append(float(s_new))
        projected_ey.append(float(ey_new))

    result = _copy_last_solution(last_x) or []
    result[3] = np.asarray(projected_s, dtype=float)
    result[4] = np.asarray(projected_ey, dtype=float)
    return result


def _first_control(values: Sequence[float] | float | None) -> float:
    if values is None:
        return 0.0
    if isinstance(values, (float, int, np.floating, np.integer)):
        return float(values)
    return float(values[0]) if len(values) else 0.0


def _observable_solver_status(controller: Any) -> tuple[bool, bool, str | None]:
    """Read optional future PRIDEAM solve instrumentation without requiring it.

    Current PRIDEAM releases hide an internal zero-control fallback inside
    ``solve_with_risk``.  Until that controller exposes ``last_solve_status``,
    this seam can only observe its own outer exception fallback.
    """
    status = getattr(controller, "last_solve_status", None)
    if not isinstance(status, Mapping):
        return False, False, None
    fallback = bool(status.get("fallback_used", False))
    reason = status.get("fallback_reason")
    return True, fallback, None if reason is None else str(reason)


def _fallback_step(
    arm: EpisodeArm,
    traffic: LaneTraffic,
    current_lane: int,
    error: BaseException,
    *,
    candidate_lane: int | None = None,
    desired_group: str | None = None,
    raw_label: str = "K",
    virtual_label: str = "K",
    probe_blocked: bool = False,
    veto_evaluated: bool = False,
    vetoed: bool = False,
    veto_score: float | None = None,
    veto_threshold: float | None = None,
    veto_allowed: bool | None = None,
    natural_desired_group: str | None = None,
    route_request_active: bool = False,
    route_requested_lane: int | None = None,
    route_request_feasible: bool | None = None,
    route_request_selected: bool = False,
    route_request_reason: str | None = None,
    proposed_constraint_mode: str | None = None,
    executed_constraint_mode: str | None = None,
    target_follower_projection_m: float | None = None,
    decision_time_s: float = 0.0,
    mpc_time_s: float = 0.0,
) -> EpisodeStepResult:
    """Apply an explicit zero-control fallback and make it visible in logs."""
    del traffic  # retained in the signature so callers preserve the same context
    mpc = _mpc_of(arm.controller)
    oa_cmd = [0.0] * int(mpc.T)
    od_cmd = [0.0] * int(mpc.T)
    path = arm.road.path(current_lane)
    sample, x_values, y_values = arm.road.path_data(current_lane)
    try:
        next_x0, next_x0_g, _, _ = arm.dynamics.propagate(
            list(arm.state.X0), [0.0, 0.0], arm.road.dt,
            list(arm.state.X0_g), path, sample, x_values, y_values,
            arm.road.boundary,
        )
        next_state = PlannerState(
            X0=next_x0,
            X0_g=next_x0_g,
            oa=oa_cmd,
            od=od_cmd,
            last_X=arm.state.last_X,
            path_changed=current_lane,
        )
    except Exception:
        # If even propagation fails, return an unchanged but independently
        # copied state so the caller can retain a complete failure record.
        next_state = arm.state.copy()
        next_state.oa = oa_cmd
        next_state.od = od_cmd
        next_state.path_changed = current_lane

    arm.state = next_state
    arm.step_index += 1
    diagnostics = DecisionDiagnostics(
        current_lane=current_lane,
        candidate_lane=current_lane if candidate_lane is None else candidate_lane,
        final_lane=current_lane,
        desired_group=desired_group,
        natural_desired_group=natural_desired_group,
        route_request_active=route_request_active,
        route_requested_lane=route_requested_lane,
        route_request_feasible=route_request_feasible,
        route_request_selected=route_request_selected,
        route_request_reason=route_request_reason,
        proposed_constraint_mode=proposed_constraint_mode,
        executed_constraint_mode=executed_constraint_mode,
        target_follower_projection_m=target_follower_projection_m,
        raw_label=raw_label,
        virtual_label=virtual_label,
        probe_blocked=probe_blocked,
        veto_evaluated=veto_evaluated,
        vetoed=vetoed,
        veto_score=veto_score,
        veto_threshold=veto_threshold,
        veto_allowed=veto_allowed,
        solver_success=False,
        fallback_used=True,
        fallback_observable=True,
        fallback_reason=f"episode_control exception: {error}",
        control_accel=0.0,
        control_steer=0.0,
        decision_time_s=float(decision_time_s),
        mpc_time_s=float(mpc_time_s),
        error=f"{error}\n{traceback.format_exc()}",
    )
    return EpisodeStepResult(state=next_state.copy(), diagnostics=diagnostics)


def step_episode_arm(arm: EpisodeArm, traffic: LaneTraffic) -> EpisodeStepResult:
    """Advance one unconstrained IDEAM or PRIDEAM arm by one simulation step.

    The traffic arrays must contain exactly the actors that are observable by
    the decision/MPC stack at this step.  DRIFT field evolution happens outside
    this function so a paired runner can independently control field inputs
    and record them.
    """
    road = arm.road
    current_lane = road.lane_of(arm.state.X0_g)
    candidate_lane = current_lane
    desired_group_name: str | None = None
    natural_desired_group_name: str | None = None
    route_request_active = False
    route_requested_lane: int | None = None
    route_request_feasible: bool | None = None
    route_request_selected = False
    route_request_reason: str | None = None
    proposed_constraint_mode: str | None = None
    executed_constraint_mode: str | None = None
    target_follower_projection_m: float | None = None
    raw_label = "K"
    virtual_label = "K"
    probe_blocked = False
    veto_evaluated = False
    vetoed = False
    veto_score: float | None = None
    veto_threshold: float | None = None
    veto_allowed: bool | None = None
    decision_duration = 0.0
    mpc_duration = 0.0

    try:
        arm.apply_coupling()
        mpc = _mpc_of(arm.controller)
        x0 = list(arm.state.X0)
        x0_g = list(arm.state.X0_g)
        path_ego = road.path(current_lane)

        last_x = _copy_last_solution(arm.state.last_X)
        if last_x is None:
            last_x = list(
                clac_last_X(
                    arm.state.oa, arm.state.od, int(mpc.T), path_ego,
                    road.dt, road.state_dimension, list(x0), list(x0_g),
                )
            )

        decision_start = time.perf_counter()
        all_info = arm.utils.get_alllane_lf(
            path_ego, x0_g, current_lane,
            traffic.left, traffic.centre, traffic.right,
        )
        group_dict, ego_group = arm.utils.formulate_gap_group(
            current_lane, last_x, all_info,
            traffic.left, traffic.centre, traffic.right,
        )
        natural_desired_group = arm.decision.decision_making(
            group_dict, _LANE_LABELS[current_lane]
        )
        natural_desired_group_name = str(natural_desired_group.get("name"))
        desired_group = natural_desired_group
        request = arm.maneuver_request
        if request is not None and request.active_at(arm.step_index * road.dt):
            route_request_active = True
            route_requested_lane = int(request.target_lane)
            requested_group, route_request_feasible, route_request_reason = _route_request_group(
                arm.decision,
                group_dict,
                ego_group,
                current_lane=current_lane,
                requested_lane=route_requested_lane,
            )
            if requested_group is not None and route_request_feasible:
                desired_group = requested_group
                route_request_selected = True
        # The group passed into the MPC must match the *final* decision.  If a
        # probe guard or risk veto converts a lane-change candidate into a
        # keep-lane action, retaining the rejected target group can still feed
        # a lateral target/follower constraint into the MPC even though the
        # logged decision says "K".  That silently defeats the veto.
        solver_target_group = desired_group
        desired_group_name = str(desired_group.get("name"))
        path_d, candidate_lane, raw_label, sample, x_values, y_values, working_x0 = _decision_info(
            road, x0, x0_g, desired_group, current_lane
        )
        virtual_label = raw_label
        raw_projection = desired_group.get("proj_f")
        if raw_projection is not None:
            candidate_projection = float(raw_projection)
            if math.isfinite(candidate_projection):
                target_follower_projection_m = candidate_projection
        additive_label = arm.utils.inquire_C_state(raw_label, desired_group)
        proposed_constraint_mode = str(additive_label)
        executed_constraint_mode = str(additive_label)

        # There is intentionally no force target and no probe-guard bypass in
        # this seam.  A probe is a hold action before a veto is considered.
        if additive_label == "Probe":
            probe_blocked = True
            path_d = path_ego
            candidate_lane = current_lane
            virtual_label = "K"
            sample, x_values, y_values = road.path_data(current_lane)
            working_x0 = _repropagate(road, current_lane, x0_g, working_x0)
            solver_target_group = ego_group
            # ``LMPC.iMPC_solve_OneStep`` independently derives the
            # constraint-tuple shape from its action label.  Keep its
            # externally supplied branch selector in sync with the final
            # keep-lane action; otherwise it attempts to unpack the
            # seven-item lane-change tuple after internally creating the
            # five-item keep-lane tuple.
            additive_label = arm.utils.inquire_C_state(virtual_label, solver_target_group)
            executed_constraint_mode = str(additive_label)

        if (
            arm.is_prideam
            and arm.coupling.enable_decision_veto
            and raw_label != "K"
            and not probe_blocked
            and candidate_lane != current_lane
        ):
            veto_evaluated = True
            veto_score_raw, veto_allowed_raw, _ = arm.controller.evaluate_decision_risk(
                list(working_x0), current_lane, candidate_lane
            )
            veto_score = float(veto_score_raw)
            veto_allowed = bool(veto_allowed_raw)
            veto_threshold = float(arm.controller.weights.decision_threshold)
            if not veto_allowed:
                vetoed = True
                path_d = path_ego
                candidate_lane = current_lane
                virtual_label = "K"
                sample, x_values, y_values = road.path_data(current_lane)
                working_x0 = _repropagate(road, current_lane, x0_g, working_x0)
                solver_target_group = ego_group
                # See the matching probe-guard case above.  This is more
                # than a logging correction: it makes the actual MPC
                # constraint branch a keep-lane branch, so a veto cannot be
                # bypassed by an inconsistent legacy action tuple.
                additive_label = arm.utils.inquire_C_state(virtual_label, solver_target_group)
                executed_constraint_mode = str(additive_label)

        if arm.state.path_changed != candidate_lane and last_x is not None:
            _set_path_curvature(arm.controller, path_d)
            last_x = _project_last_solution(
                road, last_x, int(arm.state.path_changed), candidate_lane
            )

        decision_duration = time.perf_counter() - decision_start
        mpc_start = time.perf_counter()
        if arm.is_prideam:
            solution = arm.controller.solve_with_risk(
                working_x0, arm.state.oa, arm.state.od, road.dt,
                None, None, virtual_label, x0_g, path_d, last_x,
                current_lane, ego_group, path_ego, solver_target_group,
                traffic.left, traffic.centre, traffic.right,
                candidate_lane, additive_label, virtual_label,
            )
        else:
            solution = mpc.iterative_linear_mpc_control(
                working_x0, arm.state.oa, arm.state.od, road.dt,
                None, None, virtual_label, x0_g, path_d, last_x,
                current_lane, ego_group, path_ego, solver_target_group,
                traffic.left, traffic.centre, traffic.right,
                candidate_lane, additive_label, virtual_label,
            )
        mpc_duration = time.perf_counter() - mpc_start

        if solution is None or len(solution) != 8:
            raise RuntimeError("MPC returned no complete solution")
        oa_cmd, od_cmd, ovx, ovy, owz, o_s, o_ey, o_epsi = solution
        if oa_cmd is None or od_cmd is None or len(oa_cmd) == 0 or len(od_cmd) == 0:
            raise RuntimeError("MPC returned empty control arrays")

        next_x0, next_x0_g, _, _ = arm.dynamics.propagate(
            list(working_x0), [oa_cmd[0], od_cmd[0]], road.dt, x0_g,
            path_d, sample, x_values, y_values, road.boundary,
        )
        next_state = PlannerState(
            X0=next_x0,
            X0_g=next_x0_g,
            oa=copy.deepcopy(oa_cmd),
            od=copy.deepcopy(od_cmd),
            last_X=[ovx, ovy, owz, o_s, o_ey, o_epsi],
            path_changed=candidate_lane,
        )
        fallback_observable, internal_fallback, internal_reason = _observable_solver_status(
            arm.controller
        )
        arm.state = next_state
        arm.step_index += 1
        diagnostics = DecisionDiagnostics(
            current_lane=current_lane,
            candidate_lane=_lane_from_group(desired_group),
            final_lane=candidate_lane,
            desired_group=desired_group_name,
            natural_desired_group=natural_desired_group_name,
            route_request_active=route_request_active,
            route_requested_lane=route_requested_lane,
            route_request_feasible=route_request_feasible,
            route_request_selected=route_request_selected,
            route_request_reason=route_request_reason,
            proposed_constraint_mode=proposed_constraint_mode,
            executed_constraint_mode=executed_constraint_mode,
            target_follower_projection_m=target_follower_projection_m,
            raw_label=raw_label,
            virtual_label=virtual_label,
            probe_blocked=probe_blocked,
            veto_evaluated=veto_evaluated,
            vetoed=vetoed,
            veto_score=veto_score,
            veto_threshold=veto_threshold,
            veto_allowed=veto_allowed,
            solver_success=True,
            fallback_used=internal_fallback,
            fallback_observable=fallback_observable,
            fallback_reason=internal_reason,
            control_accel=_first_control(oa_cmd),
            control_steer=_first_control(od_cmd),
            decision_time_s=float(decision_duration),
            mpc_time_s=float(mpc_duration),
        )
        return EpisodeStepResult(state=next_state.copy(), diagnostics=diagnostics)
    except Exception as error:
        return _fallback_step(
            arm, traffic, current_lane, error,
            candidate_lane=candidate_lane,
            desired_group=desired_group_name,
            raw_label=raw_label,
            virtual_label=virtual_label,
            probe_blocked=probe_blocked,
            veto_evaluated=veto_evaluated,
            vetoed=vetoed,
            veto_score=veto_score,
            veto_threshold=veto_threshold,
            veto_allowed=veto_allowed,
            natural_desired_group=natural_desired_group_name,
            route_request_active=route_request_active,
            route_requested_lane=route_requested_lane,
            route_request_feasible=route_request_feasible,
            route_request_selected=route_request_selected,
            route_request_reason=route_request_reason,
            proposed_constraint_mode=proposed_constraint_mode,
            executed_constraint_mode=executed_constraint_mode,
            target_follower_projection_m=target_follower_projection_m,
            decision_time_s=decision_duration,
            mpc_time_s=mpc_duration,
        )


def create_prideam_episode_arm(
    road: RoadContext,
    state: PlannerState,
    *,
    coupling: CouplingFlags = CouplingFlags(),
    risk_weights: Mapping[str, float] | None = None,
    mpc_overrides: Mapping[str, Any] | None = None,
    maneuver_request: ManeuverRequest | None = None,
    name: str = "prideam",
) -> EpisodeArm:
    """Create fresh PRIDEAM/IDEAM primitives for one scenario × variant arm."""
    overrides = dict(mpc_overrides or {})
    if "dt" in overrides and not math.isclose(float(overrides["dt"]), road.dt):
        raise ValueError("mpc_overrides['dt'] must equal RoadContext.dt")
    overrides["dt"] = road.dt
    controller = create_prideam_controller(
        paths=dict(road.paths),
        risk_weights=dict(risk_weights or {}),
        **overrides,
    )
    if int(controller.mpc.NX) != road.state_dimension:
        raise ValueError("RoadContext.state_dimension must match the MPC NX setting")
    utility_config = util_params()
    utility_config["dt"] = road.dt
    utils = _RoadAwareLeaderFollowerUtil(road, **utility_config)
    decision_obj = decision(**decision_params())
    dynamics = Dynamic(**params())
    return EpisodeArm(
        road=road,
        state=state.copy(),
        controller=controller,
        utils=utils,
        decision=decision_obj,
        dynamics=dynamics,
        coupling=coupling,
        maneuver_request=maneuver_request,
        name=name,
    )


def create_ideam_episode_arm(
    road: RoadContext,
    state: PlannerState,
    *,
    maneuver_request: ManeuverRequest | None = None,
    name: str = "ideam",
) -> EpisodeArm:
    """Create a fresh no-risk IDEAM arm using the same explicit road context."""
    mpc_config = constraint_params()
    mpc_config["dt"] = road.dt
    controller = LMPC(**mpc_config)
    if int(controller.NX) != road.state_dimension:
        raise ValueError("RoadContext.state_dimension must match the MPC NX setting")
    utility_config = util_params()
    utility_config["dt"] = road.dt
    utils = _RoadAwareLeaderFollowerUtil(road, **utility_config)
    decision_obj = decision(**decision_params())
    dynamics = Dynamic(**params())
    return EpisodeArm(
        road=road,
        state=state.copy(),
        controller=controller,
        utils=utils,
        decision=decision_obj,
        dynamics=dynamics,
        coupling=CouplingFlags(
            enable_decision_veto=False,
            enable_mpc_cost=False,
            enable_cbf_modulation=False,
        ),
        maneuver_request=maneuver_request,
        name=name,
    )


__all__ = [
    "CouplingFlags",
    "DecisionDiagnostics",
    "EpisodeArm",
    "EpisodeStepResult",
    "LaneTraffic",
    "ManeuverRequest",
    "PlannerState",
    "RoadContext",
    "create_ideam_episode_arm",
    "create_prideam_episode_arm",
    "step_episode_arm",
]
