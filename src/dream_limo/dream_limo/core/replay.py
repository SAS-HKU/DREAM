"""Headless Stage 1 DREAM/pure-MPC occluded-merge replay."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from math import cos, hypot, inf
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional, Tuple

import numpy as np

from dream_limo.limo_scale import DeploymentConfig, default_deployment_config, get_preset

from .decision import IDEAMDREAMDecision
from .mpc import RiskAwareMPC
from .occlusion import (
    LidarShadowBuilder,
    line_of_sight_visible,
    rectangle_polygon,
    simulate_polygon_scan,
)
from .risk_field import DREAMRiskField
from .types import EgoState, Vehicle


@dataclass(frozen=True)
class ReplaySample:
    time: float
    ego_x: float
    ego_y: float
    ego_speed: float
    acceleration: float
    steering: float
    ego_lane: int
    selected_lane: int
    merger_x: float
    merger_y: float
    merger_visible: bool
    vetoed: bool
    decision_risk: float
    risk_at_ego: float
    clearance: float
    ttc: float
    drift_seconds: float
    decision_seconds: float
    mpc_seconds: float
    mpc_fallback: bool


@dataclass(frozen=True)
class ReplayMetrics:
    preset: str
    field_ready: bool
    veto_activations: int
    reveal_time: Optional[float]
    ttc_at_reveal: float
    predicted_conflict_arrival_margin_at_reveal: float
    ego_conflict_entry_time: Optional[float]
    merger_conflict_entry_time: Optional[float]
    conflict_zone_overlap_samples: int
    minimum_clearance: float
    minimum_post_reveal_clearance: float
    minimum_ttc: float
    minimum_speed: float
    maximum_abs_acceleration: float
    maximum_abs_jerk: float
    maximum_drift_seconds: float
    maximum_decision_seconds: float
    maximum_mpc_seconds: float
    mpc_fallbacks: int
    hidden_track_leaks: int
    final_x: float
    final_y: float


@dataclass(frozen=True)
class ReplayResult:
    metrics: ReplayMetrics
    samples: Tuple[ReplaySample, ...]


def _merger_state(time: float) -> Vehicle:
    """Identical scripted merger trajectory for every A/B arm."""
    x = 3.10 + 0.18 * time
    if time < 3.0:
        y, vy = -0.45, 0.0
    elif time < 6.0:
        phase = (time - 3.0) / 3.0
        smooth = phase * phase * (3.0 - 2.0 * phase)
        y = -0.45 + 0.45 * smooth
        vy = 0.45 / 3.0 * 6.0 * phase * (1.0 - phase)
    else:
        y, vy = 0.0, 0.0
    heading = float(np.arctan2(vy, 0.18))
    return Vehicle(
        vehicle_id="merger",
        x=x,
        y=y,
        vx=0.18,
        vy=vy,
        heading=heading,
        vehicle_class="car",
        length=0.22,
        width=0.22,
        stamp=time,
    )


def _clearance(ego: EgoState, vehicle: Vehicle) -> float:
    center_distance = hypot(ego.x - vehicle.x, ego.y - vehicle.y)
    return center_distance - 0.5 * (0.22 + max(vehicle.length, vehicle.width))


def _ttc(ego: EgoState, vehicle: Vehicle) -> float:
    # Project relative position/velocity on the lane tangent. TTC is meaningful
    # only while both actors occupy or enter the shared middle-lane corridor.
    if abs(ego.y - vehicle.y) > 0.55:
        return inf
    gap = vehicle.x - ego.x - 0.22
    closing = ego.speed * cos(ego.yaw) - vehicle.vx
    if gap <= 0.0 or closing <= 1.0e-6:
        return inf
    return gap / closing


def _inside_conflict_zone(
    x: float,
    y: float,
    config: DeploymentConfig,
) -> bool:
    """Whether an actor's centre occupies the shared middle-lane conflict zone."""
    target_y = config.arena.lane_centers[config.arena.target_lane]
    return (
        config.arena.conflict_zone_x_min <= x <= config.arena.conflict_zone_x_max
        and abs(y - target_y) <= 0.5 * config.arena.lane_width
    )


def _merger_conflict_entry_after(
    reveal_time: float,
    config: DeploymentConfig,
    *,
    search_duration: float = 12.0,
    resolution: float = 0.01,
) -> float:
    """Find scripted ground-truth conflict entry for evaluation, never planning."""
    for time in np.arange(reveal_time, reveal_time + search_duration + resolution, resolution):
        merger = _merger_state(float(time))
        if _inside_conflict_zone(merger.x, merger.y, config):
            return float(time)
    return inf


def _predicted_conflict_arrival_margin(
    ego: EgoState,
    reveal_time: float,
    config: DeploymentConfig,
) -> float:
    """Route-aware reveal margin: ego arrival time minus merger arrival time.

    A positive value means the revealed merger is predicted to enter the shared
    zone first.  Larger values therefore represent more time for it to clear.
    This replaces the old lane-tangent TTC-at-reveal gate, which is undefined
    while the two robots are still in different lanes.
    """
    forward_speed = ego.speed * cos(ego.yaw)
    if forward_speed <= 1.0e-6:
        return inf
    ego_entry = reveal_time + max(
        0.0, config.arena.conflict_zone_x_min - ego.x
    ) / forward_speed
    merger_entry = _merger_conflict_entry_after(reveal_time, config)
    return ego_entry - merger_entry


def _static_world() -> Tuple[List[Vehicle], object]:
    truck_polygon = rectangle_polygon("truck", 2.40, 0.0, 1.20, 0.24)
    truck = Vehicle(
        "truck",
        2.40,
        0.0,
        heading=0.0,
        vehicle_class="truck",
        length=1.20,
        width=0.24,
    )
    return [truck], truck_polygon


def run_replay_arm(
    preset_name: str,
    *,
    config: Optional[DeploymentConfig] = None,
    duration: float = 12.0,
) -> ReplayResult:
    config = default_deployment_config() if config is None else config
    preset = get_preset(preset_name)
    field = DREAMRiskField(config)
    decision = IDEAMDREAMDecision(config)
    mpc = RiskAwareMPC(config)
    shadow_builder = LidarShadowBuilder(maximum_shadow_range=config.pde.occlusion_range)
    static_vehicles, truck_polygon = _static_world()
    ego = EgoState(0.35, config.arena.lane_centers[0], 0.0, 0.50, lane_index=0)

    initial_scan = simulate_polygon_scan(
        (ego.x + 0.10, ego.y, ego.yaw), [truck_polygon], stamp=0.0
    )
    initial_shadow = shadow_builder.build(
        field.X, field.Y, field.road_mask, initial_scan, [truck_polygon]
    )
    field.warmup(static_vehicles, ego, initial_shadow)
    if not field.ready:
        raise RuntimeError("DRIFT warmup did not reach READY")

    samples: List[ReplaySample] = []
    reveal_time: Optional[float] = None
    ttc_at_reveal = inf
    predicted_conflict_margin = inf
    hidden_track_leaks = 0
    time = 0.0
    step_count = int(round(duration / config.pde.control_dt))
    for _ in range(step_count):
        merger = _merger_state(time)
        visible = line_of_sight_visible(
            (ego.x + 0.10, ego.y), (merger.x, merger.y), [truck_polygon]
        )
        visible = visible and hypot(merger.x - ego.x, merger.y - ego.y) <= 6.0
        visible_vehicles = [*static_vehicles]
        if visible:
            visible_vehicles.append(merger)
            if reveal_time is None:
                reveal_time = time
                ttc_at_reveal = _ttc(ego, merger)
                predicted_conflict_margin = _predicted_conflict_arrival_margin(
                    ego, time, config
                )
        # This invariant is intentionally explicit: hidden truth is used below
        # only for metrics and never enters Q_veh, decision groups or MPC CBFs.
        if not visible and any(item.vehicle_id == "merger" for item in visible_vehicles):
            hidden_track_leaks += 1

        scan = simulate_polygon_scan(
            (ego.x + 0.10, ego.y, ego.yaw), [truck_polygon], stamp=time
        )
        shadow = shadow_builder.build(
            field.X, field.Y, field.road_mask, scan, [truck_polygon]
        )
        drift_started = perf_counter()
        field.step(visible_vehicles, ego, shadow)
        drift_seconds = perf_counter() - drift_started
        outcome = decision.decide(
            ego,
            visible_vehicles,
            field,
            preset,
            requested_lane=config.arena.target_lane,
        )
        solution = mpc.solve(ego, outcome.selected_lane, visible_vehicles, field, preset)
        command = solution.command
        state = mpc.model.step(
            np.asarray([ego.x, ego.y, ego.speed, ego.yaw]),
            np.asarray([command.acceleration, command.steering]),
        )
        state[2] = np.clip(state[2], config.mpc.minimum_speed, config.mpc.maximum_speed)
        lane_index = ego.lane_index
        if abs(state[1] - config.arena.lane_centers[outcome.selected_lane]) < 0.12:
            lane_index = outcome.selected_lane
        clearance = _clearance(ego, merger)
        ttc = _ttc(ego, merger)
        samples.append(
            ReplaySample(
                time=time,
                ego_x=ego.x,
                ego_y=ego.y,
                ego_speed=ego.speed,
                acceleration=command.acceleration,
                steering=command.steering,
                ego_lane=ego.lane_index,
                selected_lane=outcome.selected_lane,
                merger_x=merger.x,
                merger_y=merger.y,
                merger_visible=visible,
                vetoed=outcome.vetoed,
                decision_risk=outcome.risk_score,
                risk_at_ego=field.risk_at(ego.x, ego.y),
                clearance=clearance,
                ttc=ttc,
                drift_seconds=drift_seconds,
                decision_seconds=outcome.compute_seconds,
                mpc_seconds=solution.solve_seconds,
                mpc_fallback=solution.used_fallback,
            )
        )
        time += config.pde.control_dt
        ego = EgoState(
            x=float(state[0]),
            y=float(state[1]),
            yaw=float(state[3]),
            speed=float(state[2]),
            stamp=time,
            lane_index=lane_index,
        )

    accelerations = np.asarray([item.acceleration for item in samples])
    jerks = np.diff(accelerations) / config.pde.control_dt
    finite_ttc = [item.ttc for item in samples if np.isfinite(item.ttc)]
    post_reveal = (
        [
            item.clearance
            for item in samples
            if reveal_time is not None and reveal_time <= item.time <= reveal_time + 3.0
        ]
        or [inf]
    )
    ego_conflict_samples = [
        item
        for item in samples
        if _inside_conflict_zone(item.ego_x, item.ego_y, config)
    ]
    merger_conflict_samples = [
        item
        for item in samples
        if _inside_conflict_zone(item.merger_x, item.merger_y, config)
    ]
    overlap_samples = sum(
        _inside_conflict_zone(item.ego_x, item.ego_y, config)
        and _inside_conflict_zone(item.merger_x, item.merger_y, config)
        for item in samples
    )
    metrics = ReplayMetrics(
        preset=preset_name,
        field_ready=field.ready,
        veto_activations=sum(item.vetoed for item in samples),
        reveal_time=reveal_time,
        ttc_at_reveal=ttc_at_reveal,
        predicted_conflict_arrival_margin_at_reveal=predicted_conflict_margin,
        ego_conflict_entry_time=(
            ego_conflict_samples[0].time if ego_conflict_samples else None
        ),
        merger_conflict_entry_time=(
            merger_conflict_samples[0].time if merger_conflict_samples else None
        ),
        conflict_zone_overlap_samples=overlap_samples,
        minimum_clearance=min(item.clearance for item in samples),
        minimum_post_reveal_clearance=min(post_reveal),
        minimum_ttc=min(finite_ttc, default=inf),
        minimum_speed=min(item.ego_speed for item in samples),
        maximum_abs_acceleration=float(np.max(np.abs(accelerations))),
        maximum_abs_jerk=float(np.max(np.abs(jerks))) if len(jerks) else 0.0,
        maximum_drift_seconds=max(item.drift_seconds for item in samples),
        maximum_decision_seconds=max(item.decision_seconds for item in samples),
        maximum_mpc_seconds=max(item.mpc_seconds for item in samples),
        mpc_fallbacks=sum(item.mpc_fallback for item in samples),
        hidden_track_leaks=hidden_track_leaks,
        final_x=ego.x,
        final_y=ego.y,
    )
    return ReplayResult(metrics=metrics, samples=tuple(samples))


def validate_stage1(results: Dict[str, ReplayResult]) -> None:
    baseline = results["pure_mpc"].metrics
    dream = results["balanced"].metrics
    errors = []
    if not baseline.field_ready or not dream.field_ready:
        errors.append("field did not warm up")
    if dream.veto_activations < 1:
        errors.append("balanced DREAM never activated the decision veto")
    if dream.hidden_track_leaks or baseline.hidden_track_leaks:
        errors.append("a hidden merger leaked into planner inputs")
    if dream.minimum_speed >= 0.49:
        errors.append("balanced DREAM did not slow/yield")
    if dream.maximum_abs_acceleration <= 0.01:
        errors.append("balanced DREAM did not command braking")
    if dream.reveal_time is None or baseline.reveal_time is None:
        errors.append("merger was never revealed")
    if not np.isfinite(baseline.predicted_conflict_arrival_margin_at_reveal):
        errors.append("pure-MPC conflict-arrival margin at reveal is not finite")
    elif (
        dream.predicted_conflict_arrival_margin_at_reveal
        <= baseline.predicted_conflict_arrival_margin_at_reveal
    ):
        errors.append("DREAM conflict-arrival margin at reveal is not larger than pure MPC")
    if baseline.conflict_zone_overlap_samples < 1:
        errors.append("pure MPC never shared the intended middle-lane conflict zone")
    if dream.conflict_zone_overlap_samples >= baseline.conflict_zone_overlap_samples:
        errors.append("DREAM did not reduce simultaneous conflict-zone occupancy")
    if dream.minimum_clearance <= baseline.minimum_clearance:
        errors.append("DREAM did not improve minimum clearance over pure MPC")
    if dream.minimum_clearance <= 0.0:
        errors.append("balanced DREAM replay contacted the merger")
    if errors:
        raise AssertionError("Stage 1 replay failed: " + "; ".join(errors))


def run_stage1(output_path: Optional[Path] = None) -> Dict[str, ReplayResult]:
    results = {
        name: run_replay_arm(name)
        for name in ("pure_mpc", "balanced")
    }
    validate_stage1(results)
    if output_path is not None:
        payload = {
            "metrics": {name: asdict(result.metrics) for name, result in results.items()},
            "samples": {
                name: [asdict(sample) for sample in result.samples]
                for name, result in results.items()
            },
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)

        def json_safe(value):
            if isinstance(value, dict):
                return {key: json_safe(item) for key, item in value.items()}
            if isinstance(value, list):
                return [json_safe(item) for item in value]
            if isinstance(value, float) and not np.isfinite(value):
                return None
            return value

        output_path.write_text(json.dumps(json_safe(payload), indent=2, allow_nan=False) + "\n")
    return results
