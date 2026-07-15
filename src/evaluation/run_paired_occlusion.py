"""Run the revised paired occlusion-ablation benchmark.

This is intentionally a fresh, context-injected evaluation runner.  It does
not import either legacy uncertainty script, does not force a target lane, and
does not bypass the decision veto.  At every step it maintains three separate
views of traffic:

* ground truth used only for physical evaluation;
* traffic visible to the IDEAM decision/MPC stack;
* traffic supplied to the DRIFT PDE.

Before the *runtime map-geometry* reveal, the hidden actor is absent from both
planner inputs.  DREAM must therefore react to trailer-induced uncertainty,
not leaked hidden-vehicle state.  Reveal is evaluated from the actual ego
pose at every decision step and latches only after the target footprint is
observable through the actual trailer geometry.

The default protocol is deliberately suitable for a staged workflow: run a
small development pilot, inspect traces and solver provenance, freeze the
scenario manifest, and only then run the held-out aggregate suite.
"""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

# CasADi and CVXPY load distinct Intel OpenMP runtimes in the current IDEAM
# environment.  The legacy scripts make the same setting before importing the
# control stack.  It does not alter model behavior; it only permits the
# existing binary dependencies to coexist in a single process.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

from DecisionMaking.give_desired_path import judge_current_position
from Integration.drift_interface import DRIFTInterface
from Integration.episode_control import (
    CouplingFlags,
    LaneTraffic,
    ManeuverRequest,
    PlannerState,
    RoadContext,
    create_prideam_episode_arm,
)
from Path.path import (
    path1,
    path1c,
    path2,
    path2c,
    path3c,
    samples1,
    samples1c,
    samples2,
    samples2c,
    samples3c,
    x1,
    x1c,
    x2,
    x2c,
    x3c,
    y1,
    y1c,
    y2,
    y2c,
    y3c,
)
from pde_solver import create_vehicle as create_drift_vehicle

from evaluation.field_variants import FieldVariant, standard_field_variants
from evaluation.occlusion_benchmark_scenarios import (
    ObservationMode,
    OrientedRectangle,
    ScenarioBankConfig,
    ScenarioSpec,
    ScenarioStratum,
    assess_oriented_rectangle_visibility,
    generate_paired_scenario_bank,
)
from evaluation.scenario_qualification import qualification_manifest
from evaluation.physical_safety_metrics import (
    KinematicBoxState,
    SceneSafetySnapshot,
    evaluate_scene_safety,
    evaluate_swept_pair_safety,
    summarize_episode_safety,
)


SCHEMA_VERSION = "paired_occlusion_episode_v3"
PROTOCOL_ID = "r1c1_paired_occlusion_v3"

# These values are copied into every episode record.  The variants are
# component ablations: each channel is either fully active or explicitly off.
RISK_WEIGHTS: dict[str, float] = {
    "mpc_cost": 0.5,
    "cbf_modulation": 0.6,
    "decision_threshold": 1.5,
    "headway_modulation": 0.4,
    "max_cbf_scale": 2.5,
    "max_headway_scale": 2.0,
    "cbf_risk_normalization": 1.5,
}

S_BASE_M = 20.0
# The legacy decision policy naturally chooses its increasing-index transition
# around a centre-lane trailer under this traffic arrangement.  The benchmark
# therefore maps the scenario bank's target corridor to IDEAM lane 2.  The
# physical construction is a mirror image of the abstract bank geometry; no
# behavior is forced and the hidden actor remains on a lane-valid route.
TARGET_LANE = 2
EGO_LANE = 1
REFERENCE_LANE = 0
TTC_HORIZON_S = 10.0
TTC_CRITICAL_S = 1.5
NEAR_CLEARANCE_M = 1.0
POST_REVEAL_WINDOW_S = 3.0
SWEPT_SUBSTEP_S = 0.01
FIELD_TUBE_THRESHOLD = 0.50
FIELD_TUBE_HORIZONS_S = (0.0, 0.5, 1.0)


@dataclass(frozen=True)
class ExperimentVariant:
    """One pre-registered field or coupling ablation arm."""

    key: str
    suite: str
    label: str
    description: str
    field_key: str
    coupling: CouplingFlags


@dataclass(frozen=True)
class RunnerSettings:
    """Numerical settings recorded with every episode result."""

    dt_s: float = 0.1
    field_substeps: int = 3
    warmup_s: float = 1.0
    save_traces: bool = False

    def __post_init__(self) -> None:
        if self.dt_s <= 0.0 or not math.isfinite(self.dt_s):
            raise ValueError("dt_s must be positive and finite")
        if self.field_substeps <= 0:
            raise ValueError("field_substeps must be positive")
        if self.warmup_s < 0.0 or not math.isfinite(self.warmup_s):
            raise ValueError("warmup_s must be finite and non-negative")


# Support-actor definitions live with the frozen scenario specification, not
# in this runner.  That guarantees they are serialized and qualified before
# any variant is executed.


def _road_context(settings: RunnerSettings) -> RoadContext:
    return RoadContext(
        paths={0: path1c, 1: path2c, 2: path3c},
        samples={0: samples1c, 1: samples2c, 2: samples3c},
        x_lists={0: x1c, 1: x2c, 2: x3c},
        y_lists={0: y1c, 1: y2c, 2: y3c},
        lane_lookup=lambda pose: judge_current_position(
            pose[:2], [x1, x2], [y1, y2], [path1, path2], [samples1, samples2]
        ),
        boundary=1.0,
        dt=settings.dt_s,
    )


def _safe_float(value: Any) -> Any:
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.ndarray):
        return [_safe_float(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _safe_float(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_float(item) for item in value]
    if is_dataclass(value):
        return _safe_float(asdict(value))
    if hasattr(value, "value") and value.__class__.__module__ == "enum":
        return value.value
    return value


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_safe_float(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        json.dump(_safe_float(payload), handle, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _row_for_lane(lane: int, s_m: float, speed_mps: float, acceleration_mps2: float = 0.0) -> np.ndarray:
    path = (path1c, path2c, path3c)[int(lane)]
    x_m, y_m = path(float(s_m))
    heading = float(path.get_theta_r(float(s_m)))
    return np.asarray(
        [
            float(s_m),
            0.0,
            0.0,
            float(x_m),
            float(y_m),
            heading,
            float(speed_mps),
            float(acceleration_mps2),
        ],
        dtype=float,
    )


def _trajectory_row(scenario: ScenarioSpec, *, trailer: bool, time_s: float) -> np.ndarray:
    trajectory = (
        scenario.trailer.trajectory
        if trailer
        else scenario.counterfactual_latent_trajectory
    )
    lane = EGO_LANE if trailer else TARGET_LANE
    return _row_for_lane(
        lane,
        S_BASE_M + trajectory.longitudinal_distance_m(time_s) + trajectory.initial_x_m,
        trajectory.speed_mps(time_s),
        trajectory.longitudinal_acceleration_mps2,
    )


def _support_rows(scenario: ScenarioSpec, time_s: float) -> dict[str, np.ndarray]:
    rows: dict[str, np.ndarray] = {}
    for actor in scenario.support_actors:
        rows[actor.label] = _row_for_lane(
            actor.lane,
            S_BASE_M + actor.initial_local_s_m + actor.speed_mps * time_s,
            actor.speed_mps,
        )
    return rows


def _visible_traffic(
    scenario: ScenarioSpec, time_s: float, latent_revealed: bool
) -> tuple[LaneTraffic, dict[str, np.ndarray], bool]:
    support = _support_rows(scenario, time_s)
    trailer_row = _trajectory_row(scenario, trailer=True, time_s=time_s)
    revealed = bool(
        scenario.observation_mode is ObservationMode.FULLY_VISIBLE or latent_revealed
    )
    latent_row: np.ndarray | None = None
    if scenario.latent_present and revealed:
        latent_row = _trajectory_row(scenario, trailer=False, time_s=time_s)

    support_by_label = {actor.label: actor for actor in scenario.support_actors}
    lane_rows: dict[int, list[np.ndarray]] = {
        0: [],
        1: [trailer_row],
        2: [],
    }
    for label, row in support.items():
        lane_rows[int(support_by_label[label].lane)].append(row)
    if latent_row is not None:
        lane_rows[TARGET_LANE].append(latent_row)
    traffic = LaneTraffic.from_arrays(
        lane_rows[0],
        lane_rows[1],
        lane_rows[2],
    )
    all_rows = {"trailer": trailer_row, **support}
    if latent_row is not None:
        all_rows["latent_target_lane_vehicle"] = latent_row
    return traffic, all_rows, revealed


def _drift_vehicle_from_row(
    row: np.ndarray, *, vid: int, vehicle_class: str, length_m: float | None = None, width_m: float | None = None
) -> dict[str, Any]:
    heading = float(row[5])
    speed = float(row[6])
    vehicle = create_drift_vehicle(
        vid=int(vid),
        x=float(row[3]),
        y=float(row[4]),
        vx=speed * math.cos(heading),
        vy=speed * math.sin(heading),
        vclass=vehicle_class,
    )
    vehicle["heading"] = heading
    vehicle["a"] = float(row[7])
    if length_m is not None:
        vehicle["length"] = float(length_m)
    if width_m is not None:
        vehicle["width"] = float(width_m)
    return vehicle


def _pde_vehicles(rows: Mapping[str, np.ndarray], scenario: ScenarioSpec) -> list[dict[str, Any]]:
    vehicles: list[dict[str, Any]] = []
    vehicle_meta: list[tuple[str, int, str, float | None, float | None]] = [
        (
            "trailer", 1, "truck",
            scenario.trailer.trajectory.length_m,
            scenario.trailer.trajectory.width_m,
        )
    ]
    for index, actor in enumerate(scenario.support_actors, start=2):
        vehicle_meta.append((actor.label, index, "car", actor.length_m, actor.width_m))
    vehicle_meta.append((
        "latent_target_lane_vehicle", len(scenario.support_actors) + 2, "car", None, None,
    ))
    for label, vid, vehicle_class, length, width in vehicle_meta:
        if label in rows:
            vehicles.append(
                _drift_vehicle_from_row(
                    rows[label], vid=vid, vehicle_class=vehicle_class,
                    length_m=length, width_m=width,
                )
            )
    return vehicles


def _ego_drift_vehicle(state: PlannerState) -> dict[str, Any]:
    heading = float(state.X0_g[2])
    longitudinal = float(state.X0[0])
    lateral = float(state.X0[1])
    vehicle = create_drift_vehicle(
        vid=0,
        x=float(state.X0_g[0]),
        y=float(state.X0_g[1]),
        vx=longitudinal * math.cos(heading) - lateral * math.sin(heading),
        vy=longitudinal * math.sin(heading) + lateral * math.cos(heading),
        vclass="car",
    )
    vehicle["heading"] = heading
    return vehicle


def _box_from_row(row: np.ndarray, *, length_m: float, width_m: float, label: str) -> KinematicBoxState:
    heading = float(row[5])
    speed = float(row[6])
    return KinematicBoxState(
        x=float(row[3]),
        y=float(row[4]),
        heading=heading,
        vx=speed * math.cos(heading),
        vy=speed * math.sin(heading),
        length=float(length_m),
        width=float(width_m),
        label=label,
    )


def _oriented_rectangle(box: KinematicBoxState) -> OrientedRectangle:
    """Convert a map-coordinate physical-safety box to LOS geometry."""

    return OrientedRectangle(
        center_x_m=float(box.x),
        center_y_m=float(box.y),
        heading_rad=float(box.heading),
        length_m=float(box.length),
        width_m=float(box.width),
        label=box.label,
    )


def _runtime_visibility(
    scenario: ScenarioSpec,
    ego_state: PlannerState,
    time_s: float,
) -> dict[str, Any]:
    """Evaluate current map-coordinate LOS and planner exposure separately.

    The latent box is constructed even for the empty-shadow stratum so that
    all three members of a pair share the same counterfactual visibility
    crossing.  It is *not* inserted into traffic/PDE inputs unless the
    scenario has a real latent actor.
    """

    if scenario.static_occluders:
        raise ValueError(
            "Runtime map LOS does not support abstract static_occluders; "
            "transform them to map coordinates before using this runner"
        )

    ego_box = _ego_box(ego_state)
    trailer_box = _box_from_row(
        _trajectory_row(scenario, trailer=True, time_s=time_s),
        length_m=scenario.trailer.trajectory.length_m,
        width_m=scenario.trailer.trajectory.width_m,
        label="trailer",
    )
    latent_box = _box_from_row(
        _trajectory_row(scenario, trailer=False, time_s=time_s),
        length_m=scenario.counterfactual_latent_trajectory.length_m,
        width_m=scenario.counterfactual_latent_trajectory.width_m,
        label="counterfactual_latent_target_lane_vehicle",
    )
    forward = (math.cos(ego_box.heading), math.sin(ego_box.heading))
    left = (-forward[1], forward[0])
    sensor = (
        ego_box.x
        + scenario.ego.sensor_forward_offset_m * forward[0]
        + scenario.ego.sensor_lateral_offset_m * left[0],
        ego_box.y
        + scenario.ego.sensor_forward_offset_m * forward[1]
        + scenario.ego.sensor_lateral_offset_m * left[1],
    )
    relative = (latent_box.x - sensor[0], latent_box.y - sensor[1])
    target_in_forward_hemisphere = (
        relative[0] * forward[0] + relative[1] * forward[1] > 0.0
    )
    assessment = assess_oriented_rectangle_visibility(
        observer=sensor,
        target=_oriented_rectangle(latent_box),
        occluders=(_oriented_rectangle(trailer_box),),
        sensor_range_m=scenario.ego.sensor_range_m,
        minimum_visible_fraction=scenario.visibility.minimum_visible_fraction,
        conservative_boundary_occlusion=scenario.visibility.conservative_boundary_occlusion,
    )
    geometric_visible = bool(target_in_forward_hemisphere and assessment.visible)
    planner_visible = bool(
        scenario.observation_mode is ObservationMode.FULLY_VISIBLE
        or geometric_visible
    )
    return {
        "observation_mode": scenario.observation_mode.value,
        "time_s": float(time_s),
        "sensor_origin_x_m": float(sensor[0]),
        "sensor_origin_y_m": float(sensor[1]),
        "target_in_forward_hemisphere": bool(target_in_forward_hemisphere),
        "geometric_visible": geometric_visible,
        "visible_to_planner": planner_visible,
        "fully_visible_control_override": bool(
            scenario.observation_mode is ObservationMode.FULLY_VISIBLE
        ),
        "assessment": asdict(assessment),
    }


def _ego_box(state: PlannerState) -> KinematicBoxState:
    heading = float(state.X0_g[2])
    longitudinal = float(state.X0[0])
    lateral = float(state.X0[1])
    return KinematicBoxState(
        x=float(state.X0_g[0]),
        y=float(state.X0_g[1]),
        heading=heading,
        vx=longitudinal * math.cos(heading) - lateral * math.sin(heading),
        vy=longitudinal * math.sin(heading) + lateral * math.cos(heading),
        length=4.8,
        width=2.0,
        label="ego",
    )


def _ground_truth_boxes(scenario: ScenarioSpec, time_s: float) -> dict[str, KinematicBoxState]:
    support = _support_rows(scenario, time_s)
    boxes = {
        "trailer": _box_from_row(
            _trajectory_row(scenario, trailer=True, time_s=time_s),
            length_m=scenario.trailer.trajectory.length_m,
            width_m=scenario.trailer.trajectory.width_m,
            label="trailer",
        ),
    }
    for actor in scenario.support_actors:
        boxes[actor.label] = _box_from_row(
            support[actor.label], length_m=actor.length_m, width_m=actor.width_m,
            label=actor.label,
        )
    if scenario.latent_present:
        boxes["latent_target_lane_vehicle"] = _box_from_row(
            _trajectory_row(scenario, trailer=False, time_s=time_s),
            length_m=scenario.counterfactual_latent_trajectory.length_m,
            width_m=scenario.counterfactual_latent_trajectory.width_m,
            label="latent_target_lane_vehicle",
        )
    return boxes


def _critical_snapshot(snapshots: Iterable[SceneSafetySnapshot]) -> SceneSafetySnapshot:
    snapshots = tuple(snapshots)
    if not snapshots:
        raise ValueError("At least one safety snapshot is required")
    clearance = min(snapshots, key=lambda value: value.min_clearance_m)
    ttc = min(snapshots, key=lambda value: value.min_ttc_s)
    return SceneSafetySnapshot(
        min_clearance_m=float(clearance.min_clearance_m),
        min_ttc_s=float(ttc.min_ttc_s),
        clearance_vehicle=clearance.clearance_vehicle,
        ttc_vehicle=ttc.ttc_vehicle,
    )


def _step_safety_snapshots(
    ego_previous: KinematicBoxState,
    ego_current: KinematicBoxState,
    obstacles_previous: Mapping[str, KinematicBoxState],
    obstacles_current: Mapping[str, KinematicBoxState],
    settings: RunnerSettings,
) -> tuple[SceneSafetySnapshot, Mapping[str, SceneSafetySnapshot]]:
    """Return global and actor-specific safety over one control interval.

    The global record preserves the primary all-traffic endpoint.  The
    per-actor records make it possible to establish that the latent actor,
    rather than an incidental support vehicle, generated a reported margin.
    """

    per_actor: dict[str, SceneSafetySnapshot] = {}
    for label, current in obstacles_current.items():
        previous = obstacles_previous[label]
        instantaneous = evaluate_scene_safety(
            ego_current, [current], ttc_horizon_s=TTC_HORIZON_S
        )
        swept = (
            instantaneous,
            evaluate_swept_pair_safety(
                ego_previous,
                ego_current,
                previous,
                current,
                interval_s=settings.dt_s,
                max_substep_s=SWEPT_SUBSTEP_S,
                ttc_horizon_s=TTC_HORIZON_S,
            ),
        )
        per_actor[label] = _critical_snapshot(swept)
    return _critical_snapshot(per_actor.values()), per_actor


def _safety_provenance(
    snapshots: Sequence[SceneSafetySnapshot],
    interval_end_times_s: Sequence[float],
) -> dict[str, Any]:
    """Attach actor and interval provenance to episode-level safety extrema.

    A swept snapshot represents a 0.1-s control interval, rather than a
    falsely precise instantaneous collision timestamp.  Reporting the
    interval and the responsible oriented-box label makes a failed pilot
    diagnosable without changing the primary safety definitions.
    """

    if len(snapshots) != len(interval_end_times_s):
        raise ValueError("Safety snapshots and interval end times must align")

    def interval(index: int) -> tuple[float, float]:
        return (
            0.0 if index == 0 else float(interval_end_times_s[index - 1]),
            float(interval_end_times_s[index]),
        )

    clearance_index, clearance = min(
        enumerate(snapshots), key=lambda item: item[1].min_clearance_m
    )
    ttc_index, ttc = min(
        enumerate(snapshots), key=lambda item: item[1].min_ttc_s
    )
    clearance_start, clearance_end = interval(clearance_index)
    ttc_start, ttc_end = interval(ttc_index)
    result: dict[str, Any] = {
        "min_clearance_vehicle": clearance.clearance_vehicle,
        "min_clearance_interval_start_s": clearance_start,
        "min_clearance_interval_end_s": clearance_end,
        "min_ttc_vehicle": ttc.ttc_vehicle,
        "min_ttc_interval_start_s": ttc_start,
        "min_ttc_interval_end_s": ttc_end,
    }
    collision_index = next(
        (index for index, item in enumerate(snapshots) if item.min_clearance_m <= 0.0),
        None,
    )
    if collision_index is not None:
        collision = snapshots[collision_index]
        start, end = interval(collision_index)
        result.update({
            "first_collision_vehicle": collision.clearance_vehicle,
            "first_collision_interval_start_s": start,
            "first_collision_interval_end_s": end,
        })
    else:
        result.update({
            "first_collision_vehicle": None,
            "first_collision_interval_start_s": None,
            "first_collision_interval_end_s": None,
        })
    return result


def _summarize_safety_with_provenance(
    snapshots: Sequence[SceneSafetySnapshot],
    interval_end_times_s: Sequence[float],
    *,
    reveal_step: int | None,
    post_reveal_steps: int,
) -> dict[str, Any]:
    """Produce one episode- or actor-level safety record without TTC caps."""

    safety = summarize_episode_safety(
        snapshots,
        reveal_step=reveal_step,
        post_reveal_steps=post_reveal_steps,
        near_clearance_m=NEAR_CLEARANCE_M,
        ttc_horizon_s=TTC_HORIZON_S,
    )
    safety.update(_safety_provenance(snapshots, interval_end_times_s))
    min_ttc = float(safety["min_ttc_s"])
    safety["ttc_critical_threshold_s"] = TTC_CRITICAL_S
    safety["ttc_critical_incident"] = bool(
        math.isfinite(min_ttc) and min_ttc < TTC_CRITICAL_S
    )
    # ``None`` plus the censoring flag is preferable to a made-up finite TTC
    # value when no collision is predicted within the declared horizon.
    if not math.isfinite(min_ttc):
        safety["min_ttc_s"] = None
    post_ttc = safety.get("post_reveal_min_ttc_s")
    if isinstance(post_ttc, (float, np.floating)) and not math.isfinite(float(post_ttc)):
        safety["post_reveal_min_ttc_s"] = None
    return safety


def _tube_sample(
    drift: Any,
    row: np.ndarray,
    *,
    length_m: float,
    width_m: float,
) -> float:
    """Average field value over a compact five-point occupancy footprint."""

    heading = float(row[5])
    longitudinal = np.asarray([math.cos(heading), math.sin(heading)])
    lateral = np.asarray([-math.sin(heading), math.cos(heading)])
    center = np.asarray([float(row[3]), float(row[4])])
    offsets = (
        np.zeros(2),
        0.25 * length_m * longitudinal,
        -0.25 * length_m * longitudinal,
        0.25 * width_m * lateral,
        -0.25 * width_m * lateral,
    )
    values = [
        float(drift.get_risk_cartesian(*(center + offset)))
        for offset in offsets
    ]
    return float(np.mean(values))


def _field_tube_metrics(
    drift: Any,
    scenario: ScenarioSpec,
    time_s: float,
) -> tuple[float, float]:
    """Risk in the hidden route versus a matched empty right-lane corridor."""

    target_values: list[float] = []
    reference_values: list[float] = []
    for horizon in FIELD_TUBE_HORIZONS_S:
        target_row = _trajectory_row(scenario, trailer=False, time_s=time_s + horizon)
        reference_row = _row_for_lane(
            REFERENCE_LANE,
            float(target_row[0]),
            float(target_row[6]),
        )
        target_values.append(
            _tube_sample(
                drift, target_row,
                length_m=scenario.counterfactual_latent_trajectory.length_m,
                width_m=scenario.counterfactual_latent_trajectory.width_m,
            )
        )
        reference_values.append(
            _tube_sample(
                drift, reference_row,
                length_m=scenario.counterfactual_latent_trajectory.length_m,
                width_m=scenario.counterfactual_latent_trajectory.width_m,
            )
        )
    return float(np.mean(target_values)), float(np.mean(reference_values))


def _reference_ego_state(scenario: ScenarioSpec, time_s: float) -> PlannerState:
    """Return the fixed centre-lane ego reference used for field diagnostics.

    This state is not a controller rollout.  It exists so component ablations
    can be compared on the same pre-reveal traffic history rather than on
    controller trajectories that diverge because the ablation itself changed
    a decision.
    """

    trajectory = scenario.ego.reference_trajectory
    station = S_BASE_M + trajectory.longitudinal_distance_m(time_s)
    x_m, y_m = path2c(station)
    return PlannerState(
        X0=[trajectory.speed_mps(time_s), 0.0, 0.0, station, 0.0, 0.0],
        X0_g=[float(x_m), float(y_m), float(path2c.get_theta_r(station))],
        oa=0.0,
        od=0.0,
        last_X=None,
        path_changed=EGO_LANE,
    )


def _reference_field_playback(
    scenario: ScenarioSpec,
    field_variant: FieldVariant,
    settings: RunnerSettings,
    qualification: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate a field variant on one common, hidden-actor-free history.

    The construction-qualified nominal reveal is the fixed cutoff.  Until
    then the counterfactual latent vehicle is absent from the PDE for every
    variant.  Consequently, between-variant differences in these values are
    attributable to the specified field component rather than a different
    controller trajectory or an oracle actor injection.
    """

    cutoff = qualification.get("nominal_reveal_time_s")
    if not isinstance(cutoff, (int, float)) or not math.isfinite(float(cutoff)):
        return {
            "available": False,
            "reason": "qualification_has_no_nominal_reveal",
        }
    cutoff_s = float(cutoff)
    if cutoff_s <= 0.0:
        return {
            "available": False,
            "reason": "nonpositive_nominal_reveal_cutoff",
        }

    drift = DRIFTInterface()
    drift.reset()
    initial_traffic, initial_rows, _ = _visible_traffic(scenario, 0.0, False)
    del initial_traffic
    warmup_ego = _ego_drift_vehicle(_reference_ego_state(scenario, 0.0))
    for _ in range(int(round(settings.warmup_s / settings.dt_s))):
        drift.step(
            _pde_vehicles(initial_rows, scenario),
            warmup_ego,
            dt=settings.dt_s,
            substeps=settings.field_substeps,
            source_fn=field_variant.source_fn,
            velocity_fn=field_variant.velocity_fn,
            diffusion_fn=field_variant.diffusion_fn,
        )

    target_values: list[float] = []
    control_values: list[float] = []
    anticipation_time_s: float | None = None
    step_count = int(math.ceil(cutoff_s / settings.dt_s))
    for step_index in range(step_count):
        time_s = min(step_index * settings.dt_s, cutoff_s)
        _, rows, _ = _visible_traffic(scenario, time_s, False)
        ego = _ego_drift_vehicle(_reference_ego_state(scenario, time_s))
        drift.step(
            _pde_vehicles(rows, scenario),
            ego,
            dt=settings.dt_s,
            substeps=settings.field_substeps,
            source_fn=field_variant.source_fn,
            velocity_fn=field_variant.velocity_fn,
            diffusion_fn=field_variant.diffusion_fn,
        )
        target_value, control_value = _field_tube_metrics(drift, scenario, time_s)
        target_values.append(target_value)
        control_values.append(control_value)
        if anticipation_time_s is None and target_value >= FIELD_TUBE_THRESHOLD:
            anticipation_time_s = time_s

    target_mean = float(np.mean(target_values)) if target_values else math.nan
    control_mean = float(np.mean(control_values)) if control_values else math.nan
    return {
        "available": True,
        "protocol": (
            "fixed centre-lane reference playback; hidden/counterfactual latent "
            "actor absent from PDE; samples stop at the construction-qualified "
            "nominal map-visibility reveal"
        ),
        "nominal_reveal_cutoff_s": cutoff_s,
        "n_pre_reveal_samples": len(target_values),
        "risk_mass_target_maneuver_tube": target_mean,
        "risk_mass_opposite_lane_control_tube": control_mean,
        "risk_contrast_target_vs_control": target_mean - control_mean,
        "risk_threshold": FIELD_TUBE_THRESHOLD,
        "anticipation_detected": anticipation_time_s is not None,
        "anticipation_time_s": anticipation_time_s,
        "anticipation_lead_time_s": (
            cutoff_s - anticipation_time_s
            if anticipation_time_s is not None
            else math.nan
        ),
    }


def _field_variant_for(
    field_key: str, initial_trailer: Mapping[str, Any]
) -> FieldVariant:
    variants = {item.key: item for item in standard_field_variants(initial_trailer)}
    try:
        return variants[field_key]
    except KeyError as error:
        raise ValueError(f"Unknown field variant {field_key!r}") from error


def _variants_for_suite(suite: str) -> tuple[ExperimentVariant, ...]:
    field_full = CouplingFlags(True, True, True)
    field_variants = (
        ("field_full", "Full field", "All field components and all coupling channels are active."),
        ("field_no_advection", "No advection", "Ablates field transport only."),
        ("field_no_occ_source", "No occlusion source", "Ablates Q_occ only."),
        ("field_no_occ_diffusion", "No occlusion diffusion", "Ablates D_occ only."),
        ("field_static_trailer_occ", "Static trailer coupling", "Freezes the trailer-linked occlusion source."),
    )
    result: list[ExperimentVariant] = []
    if suite in {"field", "all"}:
        result.extend(
            ExperimentVariant(
                key=key,
                suite="field",
                label=label,
                description=description,
                field_key=key,
                coupling=field_full,
            )
            for key, label, description in field_variants
        )
    if suite in {"channels", "all"}:
        channel_variants = (
            ("coupling_full", "Full coupling", CouplingFlags(True, True, True)),
            ("coupling_no_veto", "No decision veto", CouplingFlags(False, True, True)),
            ("coupling_no_mpc_cost", "No MPC risk cost", CouplingFlags(True, False, True)),
            ("coupling_no_cbf", "No CBF modulation", CouplingFlags(True, True, False)),
            ("coupling_none", "No coupling channels", CouplingFlags(False, False, False)),
        )
        result.extend(
            ExperimentVariant(
                key=key,
                suite="channels",
                label=label,
                description=(
                    "Full field fixed; only the named decision/MPC/CBF channel "
                    "configuration differs."
                ),
                field_key="field_full",
                coupling=coupling,
            )
            for key, label, coupling in channel_variants
        )
    return tuple(result)


def _initial_state() -> PlannerState:
    x_m, y_m = path2c(S_BASE_M)
    return PlannerState(
        X0=[18.0, 0.0, 0.0, S_BASE_M, 0.0, 0.0],
        X0_g=[float(x_m), float(y_m), float(path2c.get_theta_r(S_BASE_M))],
        oa=0.0,
        od=0.0,
        last_X=None,
        path_changed=EGO_LANE,
    )


def _run_episode(
    scenario: ScenarioSpec,
    variant: ExperimentVariant,
    settings: RunnerSettings,
    output_dir: Path,
    qualification: Mapping[str, Any],
) -> dict[str, Any]:
    """Execute one fresh scenario × variant arm and return an auditable record."""

    started = time.perf_counter()
    road = _road_context(settings)
    n_steps = int(math.ceil(float(scenario.duration_s) / settings.dt_s))
    route_request = ManeuverRequest(
        target_lane=TARGET_LANE,
        start_time_s=float(scenario.nominal_merge.route_request_start_s),
        end_time_s=scenario.nominal_merge.route_request_end_s,
        label="paired_occlusion_route_preference",
    )
    arm = create_prideam_episode_arm(
        road,
        _initial_state(),
        coupling=variant.coupling,
        risk_weights=RISK_WEIGHTS,
        maneuver_request=route_request,
        name=variant.key,
    )
    initial_visibility = _runtime_visibility(scenario, arm.state, 0.0)
    planner_visible_latched = bool(initial_visibility["visible_to_planner"])
    first_planner_exposure_step: int | None = 0 if planner_visible_latched else None
    first_planner_exposure_time_s: float | None = 0.0 if planner_visible_latched else None
    first_planner_exposure_visibility: Mapping[str, Any] | None = (
        initial_visibility if planner_visible_latched else None
    )
    first_geometric_visibility_step: int | None = (
        0 if initial_visibility["geometric_visible"] else None
    )
    first_geometric_visibility_time_s: float | None = (
        0.0 if initial_visibility["geometric_visible"] else None
    )
    first_geometric_visibility: Mapping[str, Any] | None = (
        initial_visibility if initial_visibility["geometric_visible"] else None
    )
    initial_traffic, initial_rows, _ = _visible_traffic(
        scenario, 0.0, planner_visible_latched
    )
    initial_pde_vehicles = _pde_vehicles(initial_rows, scenario)
    initial_trailer = next(vehicle for vehicle in initial_pde_vehicles if vehicle["id"] == 1)
    field_variant = _field_variant_for(variant.field_key, initial_trailer)
    reference_field_playback = _reference_field_playback(
        scenario, field_variant, settings, qualification
    )

    # Reset and precondition the PDE on *visible* traffic only.  Direct step
    # calls avoid warmup's console output while retaining exactly the same PDE
    # integration path as a normal simulation step.
    arm.controller.drift.reset()
    warmup_steps = int(round(settings.warmup_s / settings.dt_s))
    warmup_ego = _ego_drift_vehicle(arm.state)
    for _ in range(warmup_steps):
        arm.controller.drift.step(
            initial_pde_vehicles,
            warmup_ego,
            dt=settings.dt_s,
            substeps=settings.field_substeps,
            source_fn=field_variant.source_fn,
            velocity_fn=field_variant.velocity_fn,
            diffusion_fn=field_variant.diffusion_fn,
        )

    snapshots: list[SceneSafetySnapshot] = []
    # Snapshot zero is instantaneous at t=0; all later entries summarize the
    # preceding swept control interval and are labelled by its end time.
    snapshot_end_times_s: list[float] = [0.0]
    ego_previous = _ego_box(arm.state)
    obstacles_previous = _ground_truth_boxes(scenario, 0.0)
    actor_snapshots: dict[str, list[SceneSafetySnapshot]] = {
        label: [
            evaluate_scene_safety(
                ego_previous, [box], ttc_horizon_s=TTC_HORIZON_S
            )
        ]
        for label, box in obstacles_previous.items()
    }
    snapshots.append(_critical_snapshot(
        values[0] for values in actor_snapshots.values()
    ))
    accel_trace: list[float] = []
    speed_trace: list[float] = [float(arm.state.X0[0])]
    field_tube_values: list[float] = []
    field_reference_values: list[float] = []
    anticipation_time_s: float | None = None
    diagnostics_trace: list[dict[str, Any]] = []
    fallback_steps = 0
    solver_failure_steps = 0
    veto_count = 0
    attempted_lane_changes = 0
    revealed_steps = 0

    for step_index in range(n_steps):
        time_s = step_index * settings.dt_s
        visibility = _runtime_visibility(scenario, arm.state, time_s)
        if (
            first_geometric_visibility_step is None
            and bool(visibility["geometric_visible"])
        ):
            first_geometric_visibility_step = step_index
            first_geometric_visibility_time_s = float(time_s)
            first_geometric_visibility = visibility
        if (
            not planner_visible_latched
            and bool(visibility["visible_to_planner"])
        ):
            planner_visible_latched = True
            first_planner_exposure_step = step_index
            first_planner_exposure_time_s = float(time_s)
            first_planner_exposure_visibility = visibility
        traffic, visible_rows, revealed = _visible_traffic(
            scenario, time_s, planner_visible_latched
        )
        if revealed:
            revealed_steps += 1
        ego_for_field = _ego_drift_vehicle(arm.state)
        arm.controller.drift.step(
            _pde_vehicles(visible_rows, scenario),
            ego_for_field,
            dt=settings.dt_s,
            substeps=settings.field_substeps,
            source_fn=field_variant.source_fn,
            velocity_fn=field_variant.velocity_fn,
            diffusion_fn=field_variant.diffusion_fn,
        )
        if not revealed:
            tube_value, reference_value = _field_tube_metrics(
                arm.controller.drift, scenario, time_s
            )
            field_tube_values.append(tube_value)
            field_reference_values.append(reference_value)
            if anticipation_time_s is None and tube_value >= FIELD_TUBE_THRESHOLD:
                anticipation_time_s = time_s

        # The legacy IDEAM utility prints large constraint arrays on ordinary
        # successful steps.  Keep those implementation diagnostics out of the
        # benchmark console/log while retaining a bounded provenance excerpt
        # in the raw trace when requested.
        solver_stdout = io.StringIO()
        with contextlib.redirect_stdout(solver_stdout):
            result = arm.step(traffic)
        diagnostic = result.diagnostics
        accel_trace.append(float(diagnostic.control_accel))
        speed_trace.append(float(result.state.X0[0]))
        if diagnostic.raw_label != "K":
            attempted_lane_changes += 1
        if diagnostic.vetoed:
            veto_count += 1
        if diagnostic.fallback_used:
            fallback_steps += 1
        if not diagnostic.solver_success:
            solver_failure_steps += 1
        diagnostics_trace.append({
            "step": step_index,
            "time_s": time_s,
            **asdict(diagnostic),
            "revealed_to_planner": revealed,
            "runtime_visibility": visibility,
            "ego_state": list(result.state.X0),
            "ego_global": list(result.state.X0_g),
            "legacy_solver_stdout": solver_stdout.getvalue()[-2000:] or None,
        })

        next_time_s = min((step_index + 1) * settings.dt_s, scenario.duration_s)
        ego_current = _ego_box(result.state)
        obstacles_current = _ground_truth_boxes(scenario, next_time_s)
        step_snapshot, actor_step_snapshots = _step_safety_snapshots(
            ego_previous,
            ego_current,
            obstacles_previous,
            obstacles_current,
            settings,
        )
        snapshots.append(step_snapshot)
        for label, actor_snapshot in actor_step_snapshots.items():
            actor_snapshots[label].append(actor_snapshot)
        snapshot_end_times_s.append(float(next_time_s))
        ego_previous = ego_current
        obstacles_previous = obstacles_current

    post_reveal_steps = max(1, int(round(POST_REVEAL_WINDOW_S / settings.dt_s)))
    safety = _summarize_safety_with_provenance(
        snapshots,
        snapshot_end_times_s,
        reveal_step=first_planner_exposure_step,
        post_reveal_steps=post_reveal_steps,
    )
    safety_by_actor = {
        label: _summarize_safety_with_provenance(
            actor_values,
            snapshot_end_times_s,
            reveal_step=first_planner_exposure_step,
            post_reveal_steps=post_reveal_steps,
        )
        for label, actor_values in actor_snapshots.items()
    }

    acceleration = np.asarray(accel_trace, dtype=float)
    jerk = np.diff(acceleration) / settings.dt_s if acceleration.size > 1 else np.asarray([])
    braking_duration = float(np.count_nonzero(acceleration < -0.3) * settings.dt_s)
    peak_deceleration = float(max(0.0, -float(np.min(acceleration)))) if acceleration.size else 0.0
    final_progress = float(arm.state.X0[3] - S_BASE_M)
    nominal_progress = float(18.0 * scenario.duration_s)
    tube_mean = float(np.mean(field_tube_values)) if field_tube_values else math.nan
    reference_mean = float(np.mean(field_reference_values)) if field_reference_values else math.nan
    anticipation_lead = (
        float(first_planner_exposure_time_s - anticipation_time_s)
        if anticipation_time_s is not None and first_planner_exposure_time_s is not None
        else math.nan
    )
    scenario_qualified = bool(qualification.get("passed", False))
    reveal_metadata: dict[str, Any] = {
        "observation_mode": scenario.observation_mode.value,
        "runtime_geometry": True,
        "visibility_latch_policy": "first planner-visible map-coordinate footprint observation",
        "initial_visibility": initial_visibility,
        "initial_visible_to_planner": bool(initial_visibility["visible_to_planner"]),
        "initial_geometric_visible": bool(initial_visibility["geometric_visible"]),
        "planner_exposure_occurred": first_planner_exposure_step is not None,
        "planner_exposure_step": first_planner_exposure_step,
        "planner_exposure_time_s": first_planner_exposure_time_s,
        "planner_exposure_visibility": first_planner_exposure_visibility,
        "geometric_reveal_occurred": first_geometric_visibility_step is not None,
        "geometric_reveal_step": first_geometric_visibility_step,
        "geometric_reveal_time_s": first_geometric_visibility_time_s,
        "geometric_reveal_visibility": first_geometric_visibility,
        "qualification_nominal_reveal_time_s": qualification.get("nominal_reveal_time_s"),
        "qualification_nominal_reveal_lead_time_s": qualification.get("nominal_reveal_lead_time_s"),
        "target_reveal_lead_time_s": scenario.conflict.target_reveal_lead_time_s,
    }
    trace_path: str | None = None
    if settings.save_traces:
        trace_file = output_dir / "traces" / f"{scenario.scenario_id}__{variant.key}.json"
        _json_dump(trace_file, {
            "schema_version": SCHEMA_VERSION,
            "scenario_id": scenario.scenario_id,
            "variant": variant.key,
            "settings": asdict(settings),
            "route_request": asdict(route_request),
            "reveal": reveal_metadata,
            "qualification": dict(qualification),
            "diagnostics": diagnostics_trace,
            "physical_safety": [
                {
                    "interval_end_time_s": end_time_s,
                    **asdict(snapshot),
                }
                for end_time_s, snapshot in zip(snapshot_end_times_s, snapshots)
            ],
            "physical_safety_by_actor": {
                label: [
                    {
                        "interval_end_time_s": end_time_s,
                        **asdict(snapshot),
                    }
                    for end_time_s, snapshot in zip(snapshot_end_times_s, actor_values)
                ]
                for label, actor_values in actor_snapshots.items()
            },
        })
        trace_path = str(trace_file)

    support_labels = {actor.label for actor in scenario.support_actors}
    support_nonbinding_floor_m = max(
        NEAR_CLEARANCE_M,
        float(qualification.get("nominal_go_latent_clearance_m", 0.0)) + 2.0,
    )
    runtime_support_binding = any(
        float(safety_by_actor[label]["min_clearance_m"]) < support_nonbinding_floor_m
        or (
            safety_by_actor[label].get("min_ttc_s") is not None
            and float(safety_by_actor[label]["min_ttc_s"]) < TTC_CRITICAL_S
        )
        for label in support_labels
    )
    execution_valid = bool(solver_failure_steps == 0 and fallback_steps == 0)
    run_valid = bool(
        scenario_qualified and execution_valid and not runtime_support_binding
    )
    pre_reveal_diagnostics = [
        item for item in diagnostics_trace if not bool(item["revealed_to_planner"])
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "scenario_id": scenario.scenario_id,
        "pair_id": scenario.pair_id,
        "stratum": scenario.stratum.value,
        "variant": variant.key,
        "variant_label": variant.label,
        "suite": variant.suite,
        "variant_config": {
            "field": field_variant.manifest(),
            "coupling": asdict(variant.coupling),
        },
        "settings": asdict(settings),
        "scenario_design": {
            "family_id": scenario.family_id,
            "seed": scenario.seed,
            "severity": scenario.conflict.label,
            "nominal_time_to_conflict_s": scenario.conflict.nominal_time_to_conflict_s,
            "nominal_unmitigated_clearance_m": scenario.conflict.nominal_unmitigated_clearance_m,
            "defensive_action_window_s": scenario.conflict.defensive_action_window_s,
            "nominal_merge": asdict(scenario.nominal_merge),
            "route_request": asdict(route_request),
            "route_request_policy": (
                "The request may select an adjacent target gap only after "
                "IDEAM gap-magnitude and first-hop risk acceptance; the "
                "short-gap/probe guard and DREAM veto remain active."
            ),
            "support_traffic": [asdict(actor) for actor in scenario.support_actors],
        },
        "qualification": dict(qualification),
        "reveal": reveal_metadata,
        "validity": {
            "sim_completed": True,
            "run_valid": run_valid,
            "scenario_qualified": scenario_qualified,
            "execution_valid": execution_valid,
            "actual_reveal_occurred": first_planner_exposure_step is not None,
            "post_reveal_metrics_available": bool(safety["post_reveal_available"]),
            "valid_reveal": first_planner_exposure_step is not None,
            "fallback_used": bool(fallback_steps > 0),
            "support_actor_nonbinding_runtime": not runtime_support_binding,
            "support_nonbinding_floor_m": support_nonbinding_floor_m,
            "route_request_started_while_hidden": (
                None
                if scenario.observation_mode is ObservationMode.FULLY_VISIBLE
                else bool(
                    first_planner_exposure_time_s is None
                    or route_request.start_time_s < first_planner_exposure_time_s
                )
            ),
            "pde_hidden_actor_pre_reveal": False,
            "forced_target_lane": False,
            "probe_guard_bypassed": False,
            "veto_bypassed": False,
        },
        "solver": {
            "n_steps": n_steps,
            "n_solver_failure_steps": solver_failure_steps,
            "n_fallback_steps": fallback_steps,
            "fallback_rate": fallback_steps / max(1, n_steps),
            "mean_decision_time_s": float(np.mean([
                row["decision_time_s"] for row in diagnostics_trace
            ])) if diagnostics_trace else 0.0,
            "mean_mpc_time_s": float(np.mean([
                row["mpc_time_s"] for row in diagnostics_trace
            ])) if diagnostics_trace else 0.0,
        },
        "safety": safety,
        "safety_by_actor": safety_by_actor,
        "field": {
            "tube_sampling_definition": (
                "five footprint probes over the target-maneuver counterfactual "
                "tube and an opposite-lane control tube at 0, 0.5, and 1.0 s "
                "horizons; before planner exposure only"
            ),
            "risk_mass_target_maneuver_tube": tube_mean,
            "risk_mass_opposite_lane_control_tube": reference_mean,
            "risk_contrast_target_vs_control": tube_mean - reference_mean,
            "risk_threshold": FIELD_TUBE_THRESHOLD,
            "anticipation_detected": anticipation_time_s is not None,
            "anticipation_time_s": anticipation_time_s,
            "anticipation_lead_time_s": anticipation_lead,
            "n_pre_reveal_samples": len(field_tube_values),
            "reference_playback": reference_field_playback,
        },
        "tradeoff": {
            "progress_m": final_progress,
            "nominal_progress_m": nominal_progress,
            "time_loss_s": max(0.0, (nominal_progress - final_progress) / 18.0),
            "mean_speed_mps": float(np.mean(speed_trace)),
            "braking_duration_s": braking_duration,
            "peak_deceleration_mps2": peak_deceleration,
            "mean_abs_jerk_mps3": float(np.mean(np.abs(jerk))) if jerk.size else 0.0,
            "decision_attempt_count": attempted_lane_changes,
            "executed_lane_change_count": int(sum(
                item["virtual_label"] != "K" for item in diagnostics_trace
            )),
            "route_request_selected_count": int(sum(
                bool(item["route_request_selected"]) for item in diagnostics_trace
            )),
            "pre_reveal_route_request_selected_count": int(sum(
                bool(item["route_request_selected"]) for item in pre_reveal_diagnostics
            )),
            "pre_reveal_executed_lane_change_count": int(sum(
                item["virtual_label"] != "K" for item in pre_reveal_diagnostics
            )),
            "pre_reveal_veto_count": int(sum(
                bool(item["vetoed"]) for item in pre_reveal_diagnostics
            )),
            "pre_reveal_constraint_count": int(sum(
                item["executed_constraint_mode"] == "constraint"
                for item in pre_reveal_diagnostics
            )),
            "veto_count": veto_count,
            "false_veto_incident": bool(
                scenario.stratum is ScenarioStratum.EMPTY_SHADOW and veto_count > 0
            ),
            "unnecessary_braking_incident": bool(
                scenario.stratum is ScenarioStratum.EMPTY_SHADOW and braking_duration > 0.5
            ),
            "revealed_steps": revealed_steps,
        },
        "trace_path": trace_path,
        "wall_time_s": time.perf_counter() - started,
    }


def _failed_record(
    scenario: ScenarioSpec,
    variant: ExperimentVariant,
    settings: RunnerSettings,
    error: BaseException,
    qualification: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return an explicit invalid record instead of silently dropping an arm."""

    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "scenario_id": scenario.scenario_id,
        "pair_id": scenario.pair_id,
        "stratum": scenario.stratum.value,
        "variant": variant.key,
        "variant_label": variant.label,
        "suite": variant.suite,
        "settings": asdict(settings),
        "qualification": None if qualification is None else dict(qualification),
        "validity": {
            "sim_completed": False,
            "run_valid": False,
            "scenario_qualified": bool(qualification and qualification.get("passed", False)),
            "execution_valid": False,
            "actual_reveal_occurred": False,
            "post_reveal_metrics_available": False,
            "valid_reveal": False,
            "fallback_used": False,
            "support_actor_nonbinding_runtime": False,
            "route_request_started_while_hidden": None,
            "pde_hidden_actor_pre_reveal": False,
            "forced_target_lane": False,
            "probe_guard_bypassed": False,
            "veto_bypassed": False,
        },
        "solver": {"n_steps": 0, "n_solver_failure_steps": 0, "n_fallback_steps": 0},
        "error": f"{type(error).__name__}: {error}",
    }


def _select_scenarios(
    scenarios: Sequence[ScenarioSpec],
    *,
    strata: Sequence[str],
    severity: Sequence[str] | None,
    max_pairs: int | None,
) -> tuple[ScenarioSpec, ...]:
    wanted_strata = {ScenarioStratum(item) for item in strata}
    wanted_severity = None if not severity else set(severity)
    grouped: dict[str, list[ScenarioSpec]] = {}
    for scenario in scenarios:
        if scenario.stratum not in wanted_strata:
            continue
        if wanted_severity is not None and scenario.conflict.label not in wanted_severity:
            continue
        grouped.setdefault(scenario.pair_id, []).append(scenario)
    pair_ids = sorted(grouped)
    if max_pairs is not None:
        if max_pairs <= 0:
            raise ValueError("max_pairs must be positive")
        pair_ids = pair_ids[:max_pairs]
    selected = [scenario for pair_id in pair_ids for scenario in grouped[pair_id]]
    return tuple(sorted(selected, key=lambda item: (item.pair_id, item.stratum.value)))


def _scenario_manifest(
    bank_config: ScenarioBankConfig,
    scenarios: Sequence[ScenarioSpec],
    qualification: Mapping[str, Any],
) -> dict[str, Any]:
    frozen_design = {
        "protocol_id": PROTOCOL_ID,
        "bank_config": _safe_float(asdict(bank_config)),
        "selected_scenario_ids": [item.scenario_id for item in scenarios],
    }
    frozen_design_sha256 = hashlib.sha256(
        json.dumps(frozen_design, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "protocol_id": PROTOCOL_ID,
        "schema_version": SCHEMA_VERSION,
        "frozen_design_sha256": frozen_design_sha256,
        "bank_config": asdict(bank_config),
        "selected_scenarios": [asdict(item) for item in scenarios],
        "support_traffic": [asdict(actor) for actor in bank_config.support_actors],
        "qualification": dict(qualification),
        "field_protocol": {
            "hidden_actor_input_policy": (
                "absent from PDE and decision/MPC until the runtime "
                "map-coordinate visibility latch"
            ),
            "tube_threshold": FIELD_TUBE_THRESHOLD,
            "tube_horizons_s": FIELD_TUBE_HORIZONS_S,
        },
        "route_request_protocol": {
            "target_lane": TARGET_LANE,
            "forced_target_lane": False,
            "preference_gate": "IDEAM gap-magnitude and first-hop risk acceptance",
            "later_gates": ("short-gap/probe", "DREAM decision veto"),
        },
    }


def _completed_arm_keys(path: Path) -> set[tuple[str, str]]:
    """Read completed arm identifiers from an append-only JSONL run file."""

    completed: set[tuple[str, str]] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                key = (str(record["scenario_id"]), str(record["variant"]))
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise ValueError(
                    f"Cannot resume: malformed episode record at {path}:{line_number}"
                ) from error
            if key in completed:
                raise ValueError(f"Cannot resume: duplicate completed arm {key!r}")
            completed.add(key)
    return completed


class BenchmarkRunLockedError(RuntimeError):
    """Raised when another runner owns an output directory's append-only log."""


def _acquire_output_lock(output_path: Path) -> tuple[Path, str]:
    """Atomically reserve one benchmark output directory for one process.

    A timed-out client can leave its child runner alive.  Without an exclusive
    lock, a second ``--resume`` process can read the same completed-key set and
    append the same arm concurrently.  The token lets the releasing process
    prove it is removing only its own lock file.
    """

    lock_path = output_path / ".paired_occlusion_run.lock"
    token = hashlib.sha256(
        f"{os.getpid()}|{time.time_ns()}|{output_path}".encode("utf-8")
    ).hexdigest()
    payload = {
        "protocol_id": PROTOCOL_ID,
        "pid": os.getpid(),
        "created_unix_s": time.time(),
        "output_dir": str(output_path),
        "token": token,
    }
    try:
        descriptor = os.open(
            str(lock_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL
        )
    except FileExistsError as error:
        try:
            existing = lock_path.read_text(encoding="utf-8").strip()
        except OSError:
            existing = "<unreadable lock metadata>"
        raise BenchmarkRunLockedError(
            "Benchmark output is already locked by a running or interrupted "
            f"process: {lock_path}. Do not launch a second --resume command. "
            f"Lock metadata: {existing}"
        ) from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, sort_keys=True)
            handle.write("\n")
    except BaseException:
        try:
            lock_path.unlink()
        except OSError:
            pass
        raise
    return lock_path, token


def _release_output_lock(lock_path: Path, token: str) -> None:
    """Remove a lock only if it still belongs to this runner instance."""

    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if payload.get("token") != token:
        return
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass


def run_benchmark(
    *,
    output_dir: str | Path,
    suite: str = "field",
    replicates_per_cell: int = 1,
    base_seed: int = 20260713,
    strata: Sequence[str] = tuple(item.value for item in ScenarioStratum),
    severity: Sequence[str] | None = None,
    max_pairs: int | None = None,
    variant_keys: Sequence[str] | None = None,
    settings: RunnerSettings = RunnerSettings(),
    overwrite: bool = False,
    resume: bool = False,
) -> Path:
    """Generate, freeze, and run a deterministic paired benchmark subset."""

    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    lock_path, lock_token = _acquire_output_lock(output_path)
    try:
        episodes_path = output_path / "episodes.jsonl"
        if resume and overwrite:
            raise ValueError("resume and overwrite cannot be used together")
        if episodes_path.exists() and not overwrite and not resume:
            raise FileExistsError(
                f"{episodes_path} exists; use resume=True, overwrite=True, or choose a new output directory"
            )
        if overwrite and episodes_path.exists():
            episodes_path.unlink()

        bank_config = ScenarioBankConfig(
            replicates_per_cell=int(replicates_per_cell), base_seed=int(base_seed)
        )
        bank = generate_paired_scenario_bank(bank_config)
        all_qualification = qualification_manifest(bank)
        if not bool(all_qualification.get("all_passed", False)):
            failures = {
                pair_id: item.get("failure_codes", [])
                for pair_id, item in all_qualification.get("pairs", {}).items()
                if not bool(item.get("passed", False))
            }
            raise ValueError(
                "Scenario qualification failed; controller evaluation is blocked: "
                + json.dumps(failures, sort_keys=True)
            )
        selected = _select_scenarios(
            bank.scenarios,
            strata=strata,
            severity=severity,
            max_pairs=max_pairs,
        )
        if not selected:
            raise ValueError("Scenario selection is empty")
        variants = _variants_for_suite(suite)
        if variant_keys:
            wanted_variants = set(variant_keys)
            variants = tuple(item for item in variants if item.key in wanted_variants)
            unknown = wanted_variants - {item.key for item in _variants_for_suite(suite)}
            if unknown:
                raise ValueError(
                    "Requested variants are not available for the selected suite: "
                    + ", ".join(sorted(unknown))
                )
        if not variants:
            raise ValueError("Variant selection is empty")
        selected_pair_ids = {item.pair_id for item in selected}
        selected_qualification = {
            "qualification_config": all_qualification["qualification_config"],
            "pairs": {
                pair_id: value
                for pair_id, value in all_qualification["pairs"].items()
                if pair_id in selected_pair_ids
            },
            "all_passed": True,
        }
        scenario_manifest = _scenario_manifest(bank_config, selected, selected_qualification)
        completed_keys: set[tuple[str, str]] = set()
        if resume and episodes_path.exists():
            manifest_path = output_path / "scenario_manifest.json"
            if not manifest_path.is_file():
                raise ValueError("Cannot resume without the original scenario_manifest.json")
            with manifest_path.open("r", encoding="utf-8") as handle:
                existing_manifest = json.load(handle)
            if existing_manifest.get("frozen_design_sha256") != scenario_manifest.get("frozen_design_sha256"):
                raise ValueError("Cannot resume: frozen scenario design hash does not match")
            completed_keys = _completed_arm_keys(episodes_path)
        else:
            _json_dump(output_path / "qualification_manifest.json", selected_qualification)
            _json_dump(output_path / "scenario_manifest.json", scenario_manifest)
            _json_dump(output_path / "variant_manifest.json", {
                "protocol_id": PROTOCOL_ID,
                "variants": [asdict(item) for item in variants],
                "risk_weights": RISK_WEIGHTS,
            })

        total = len(selected) * len(variants)
        completed = len(completed_keys)
        for scenario in selected:
            for variant in variants:
                arm_key = (scenario.scenario_id, variant.key)
                if arm_key in completed_keys:
                    continue
                completed += 1
                print(
                    f"[{completed:03d}/{total:03d}] {scenario.scenario_id} | {variant.key}",
                    flush=True,
                )
                try:
                    record = _run_episode(
                        scenario,
                        variant,
                        settings,
                        output_path,
                        selected_qualification["pairs"][scenario.pair_id],
                    )
                except Exception as error:  # write a claim-blocking provenance record
                    record = _failed_record(
                        scenario,
                        variant,
                        settings,
                        error,
                        selected_qualification["pairs"].get(scenario.pair_id),
                    )
                _append_jsonl(episodes_path, record)
                status = "valid" if record.get("validity", {}).get("run_valid") else "INVALID"
                wall_time = record.get("wall_time_s")
                timing = f" {float(wall_time):.1f}s" if isinstance(wall_time, (int, float)) else ""
                print(f"  -> {status}{timing}", flush=True)
        return episodes_path
    finally:
        _release_output_lock(lock_path, lock_token)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="Directory for JSONL records and manifests.")
    parser.add_argument("--suite", choices=("field", "channels", "all"), default="field")
    parser.add_argument("--replicates-per-cell", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=20260713)
    parser.add_argument(
        "--strata", nargs="+", choices=[item.value for item in ScenarioStratum],
        default=[item.value for item in ScenarioStratum],
    )
    parser.add_argument("--severity", nargs="+", choices=("mild", "moderate", "critical"))
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument(
        "--variants", nargs="+",
        help="Optional pre-registered variant keys for a development smoke run.",
    )
    parser.add_argument("--warmup-s", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--field-substeps", type=int, default=3)
    parser.add_argument("--save-traces", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume only missing arms after verifying the frozen scenario-design hash.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    settings = RunnerSettings(
        dt_s=args.dt,
        field_substeps=args.field_substeps,
        warmup_s=args.warmup_s,
        save_traces=args.save_traces,
    )
    path = run_benchmark(
        output_dir=args.out,
        suite=args.suite,
        replicates_per_cell=args.replicates_per_cell,
        base_seed=args.base_seed,
        strata=args.strata,
        severity=args.severity,
        max_pairs=args.max_pairs,
        variant_keys=args.variants,
        settings=settings,
        overwrite=args.overwrite,
        resume=args.resume,
    )
    print(f"Episode records written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
