"""Map-faithful, controller-independent qualification for occlusion scenarios.

The original scenario generator only checked an abstract reveal time.  This
module rejects a cell before controller evaluation unless its intended
occlusion, latent conflict, and defensive braking alternative coexist in the
actual three-lane IDEAM map.  It is deliberately a construction check, not a
source of safety results.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence

from Path.path import path2c

from evaluation.occlusion_benchmark_scenarios import (
    OrientedRectangle,
    ScenarioBank,
    ScenarioSpec,
    ScenarioStratum,
    assess_oriented_rectangle_visibility,
)
from evaluation.physical_safety_metrics import (
    KinematicBoxState,
    signed_oriented_box_clearance,
)


S_BASE_M = 20.0
EGO_SPEED_MPS = 18.0
EGO_LENGTH_M = 4.8
EGO_WIDTH_M = 2.0


@dataclass(frozen=True)
class QualificationConfig:
    """Fixed preflight definitions, serialized alongside a benchmark run."""

    sweep_dt_s: float = 0.01
    lane_width_m: float = 3.5
    minimum_clearance_m: float = 1.0
    max_brake_mps2: float = 3.0
    reaction_delay_s: float = 0.1
    reveal_lead_tolerance_s: float = 0.20
    min_hidden_duration_s: float = 0.50
    sensor_forward_offset_m: float = 1.5
    sensor_range_m: float = 120.0
    min_straight_remaining_m: float = 1.0
    support_nonbinding_margin_m: float = 2.0
    require_route_request_during_occlusion: bool = True
    min_route_request_hidden_s: float = 0.10

    def __post_init__(self) -> None:
        values = (
            self.sweep_dt_s, self.lane_width_m, self.minimum_clearance_m,
            self.max_brake_mps2, self.reaction_delay_s,
            self.reveal_lead_tolerance_s, self.min_hidden_duration_s,
            self.sensor_forward_offset_m, self.sensor_range_m,
            self.min_straight_remaining_m, self.support_nonbinding_margin_m,
            self.min_route_request_hidden_s,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("QualificationConfig values must be finite")
        if self.sweep_dt_s <= 0.0 or self.lane_width_m <= 0.0:
            raise ValueError("Qualification sweep and lane width must be positive")
        if self.minimum_clearance_m <= 0.0 or self.max_brake_mps2 <= 0.0:
            raise ValueError("Qualification clearance and braking bounds must be positive")
        if self.reaction_delay_s < 0.0 or self.min_hidden_duration_s < 0.0:
            raise ValueError("Qualification durations must be non-negative")
        if self.support_nonbinding_margin_m < 0.0 or self.min_route_request_hidden_s < 0.0:
            raise ValueError("Qualification safety margins and durations must be non-negative")


@dataclass(frozen=True)
class ScenarioQualification:
    """Auditable qualification result for one paired physical construction."""

    scenario_id: str
    pair_id: str
    passed: bool
    failure_codes: tuple[str, ...]
    initial_visible_fraction: float
    nominal_reveal_time_s: float | None
    nominal_reveal_lead_time_s: float | None
    nominal_go_latent_clearance_m: float
    nominal_go_trailer_clearance_m: float
    nominal_support_clearance_m: Mapping[str, float]
    immediate_brake_trailer_clearance_m: float
    reveal_brake_trailer_clearance_m: float | None
    nominal_contact_actor: str | None
    route_request_hidden_duration_s: float | None
    map_straight_until_s: float

    def manifest(self) -> dict[str, Any]:
        return asdict(self)


def _quintic_progress(time_s: float, start_s: float, duration_s: float) -> tuple[float, float]:
    """Return smooth lane-change progress and time derivative."""

    q = (time_s - start_s) / duration_s
    if q <= 0.0:
        return 0.0, 0.0
    if q >= 1.0:
        return 1.0, 0.0
    progress = 10.0 * q**3 - 15.0 * q**4 + 6.0 * q**5
    derivative = (30.0 * q**2 - 60.0 * q**3 + 30.0 * q**4) / duration_s
    return float(progress), float(derivative)


def _box_from_master_station(
    *,
    station_m: float,
    lateral_offset_m: float,
    tangent_speed_mps: float,
    lateral_speed_mps: float,
    length_m: float,
    width_m: float,
    label: str,
) -> KinematicBoxState:
    """Create a map-coordinate footprint from the centre-lane station."""

    x_m, y_m = path2c.get_cartesian_coords(station_m, lateral_offset_m)
    tangent = float(path2c.get_theta_r(station_m))
    vx = tangent_speed_mps * math.cos(tangent) - lateral_speed_mps * math.sin(tangent)
    vy = tangent_speed_mps * math.sin(tangent) + lateral_speed_mps * math.cos(tangent)
    heading = math.atan2(vy, vx) if abs(vx) + abs(vy) > 1.0e-12 else tangent
    return KinematicBoxState(
        x=float(x_m), y=float(y_m), heading=float(heading), vx=float(vx), vy=float(vy),
        length=float(length_m), width=float(width_m), label=label,
    )


def _nominal_ego_box(spec: ScenarioSpec, time_s: float, cfg: QualificationConfig) -> KinematicBoxState:
    merge = spec.nominal_merge
    progress, progress_rate = _quintic_progress(
        time_s, merge.lane_change_start_s, merge.lane_change_duration_s
    )
    # Lane 2 is right of the centre-lane travel direction in the active map.
    lateral_offset = -merge.lateral_shift_m * progress
    lateral_speed = -merge.lateral_shift_m * progress_rate
    station = S_BASE_M + spec.ego.reference_trajectory.longitudinal_distance_m(time_s)
    return _box_from_master_station(
        station_m=station,
        lateral_offset_m=lateral_offset,
        tangent_speed_mps=spec.ego.reference_trajectory.speed_mps(time_s),
        lateral_speed_mps=lateral_speed,
        length_m=spec.ego.reference_trajectory.length_m,
        width_m=spec.ego.reference_trajectory.width_m,
        label="ego_nominal_merge",
    )


def _braking_ego_box(
    spec: ScenarioSpec,
    time_s: float,
    brake_start_s: float,
    cfg: QualificationConfig,
) -> KinematicBoxState:
    """Centre-lane reaction-plus-maximum-braking reference."""

    v0 = spec.ego.reference_trajectory.initial_speed_mps
    if time_s <= brake_start_s:
        distance = v0 * time_s
        speed = v0
    else:
        before = v0 * brake_start_s
        elapsed = time_s - brake_start_s
        coast = min(elapsed, cfg.reaction_delay_s)
        after_reaction = max(0.0, elapsed - cfg.reaction_delay_s)
        stopping_time = v0 / cfg.max_brake_mps2
        braking_time = min(after_reaction, stopping_time)
        distance = before + v0 * coast + v0 * braking_time - 0.5 * cfg.max_brake_mps2 * braking_time**2
        speed = max(0.0, v0 - cfg.max_brake_mps2 * braking_time)
    return _box_from_master_station(
        station_m=S_BASE_M + distance,
        lateral_offset_m=0.0,
        tangent_speed_mps=speed,
        lateral_speed_mps=0.0,
        length_m=spec.ego.reference_trajectory.length_m,
        width_m=spec.ego.reference_trajectory.width_m,
        label="ego_brake_reference",
    )


def _trailer_box(spec: ScenarioSpec, time_s: float) -> KinematicBoxState:
    trajectory = spec.trailer.trajectory
    return _box_from_master_station(
        station_m=S_BASE_M + trajectory.longitudinal_distance_m(time_s) + trajectory.initial_x_m,
        lateral_offset_m=0.0,
        tangent_speed_mps=trajectory.speed_mps(time_s),
        lateral_speed_mps=0.0,
        length_m=trajectory.length_m,
        width_m=trajectory.width_m,
        label="trailer",
    )


def _latent_box(spec: ScenarioSpec, time_s: float, cfg: QualificationConfig) -> KinematicBoxState:
    trajectory = spec.counterfactual_latent_trajectory
    return _box_from_master_station(
        station_m=S_BASE_M + trajectory.longitudinal_distance_m(time_s) + trajectory.initial_x_m,
        lateral_offset_m=-cfg.lane_width_m,
        tangent_speed_mps=trajectory.speed_mps(time_s),
        lateral_speed_mps=0.0,
        length_m=trajectory.length_m,
        width_m=trajectory.width_m,
        label="latent_target_lane_vehicle",
    )


def _support_box(
    spec: ScenarioSpec,
    label: str,
    lane: int,
    initial_local_s_m: float,
    speed_mps: float,
    length_m: float,
    width_m: float,
    time_s: float,
    cfg: QualificationConfig,
) -> KinematicBoxState:
    """Create a map-coordinate support vehicle from frozen scenario metadata."""

    del spec  # retained to make this helper symmetric with the other boxes
    lateral_offset = (1 - int(lane)) * cfg.lane_width_m
    return _box_from_master_station(
        station_m=S_BASE_M + initial_local_s_m + speed_mps * time_s,
        lateral_offset_m=lateral_offset,
        tangent_speed_mps=speed_mps,
        lateral_speed_mps=0.0,
        length_m=length_m,
        width_m=width_m,
        label=label,
    )


def _rectangle(box: KinematicBoxState) -> OrientedRectangle:
    return OrientedRectangle(
        center_x_m=box.x, center_y_m=box.y, heading_rad=box.heading,
        length_m=box.length, width_m=box.width, label=box.label,
    )


def _visibility_fraction(
    ego: KinematicBoxState,
    trailer: KinematicBoxState,
    latent: KinematicBoxState,
    cfg: QualificationConfig,
    *,
    minimum_visible_fraction: float,
    conservative_boundary_occlusion: bool,
) -> tuple[bool, float]:
    forward = (math.cos(ego.heading), math.sin(ego.heading))
    sensor = (
        ego.x + cfg.sensor_forward_offset_m * forward[0],
        ego.y + cfg.sensor_forward_offset_m * forward[1],
    )
    relative = (latent.x - sensor[0], latent.y - sensor[1])
    # A rearward object cannot become a forward sensor reveal even if the
    # geometric ray is unobstructed.
    if relative[0] * forward[0] + relative[1] * forward[1] <= 0.0:
        return False, 0.0
    assessment = assess_oriented_rectangle_visibility(
        observer=sensor,
        target=_rectangle(latent),
        occluders=(_rectangle(trailer),),
        sensor_range_m=cfg.sensor_range_m,
        minimum_visible_fraction=minimum_visible_fraction,
        conservative_boundary_occlusion=conservative_boundary_occlusion,
    )
    return bool(assessment.visible), float(assessment.visible_fraction)


def _minimum_clearance(
    first: Sequence[KinematicBoxState], second: Sequence[KinematicBoxState]
) -> tuple[float, float]:
    best = math.inf
    time_at_best = 0.0
    for index, (left, right) in enumerate(zip(first, second)):
        clearance = signed_oriented_box_clearance(left, right)
        if clearance < best:
            best = float(clearance)
            time_at_best = float(index)
    return best, time_at_best


def _time_grid(stop_s: float, dt_s: float) -> tuple[float, ...]:
    steps = int(math.ceil(stop_s / dt_s))
    return tuple(min(stop_s, index * dt_s) for index in range(steps + 1))


def qualify_scenario(
    spec: ScenarioSpec,
    config: QualificationConfig = QualificationConfig(),
) -> ScenarioQualification:
    """Return a deterministic pass/fail qualification for one scenario spec."""

    failures: list[str] = []
    if spec.nominal_merge.target_lane != 2:
        failures.append("UNSUPPORTED_TARGET_LANE")
    if not math.isclose(spec.nominal_merge.lateral_shift_m, config.lane_width_m, abs_tol=0.05):
        failures.append("RUNNER_GEOMETRY_MISMATCH")
    if not math.isclose(abs(spec.counterfactual_latent_trajectory.initial_y_m), config.lane_width_m, abs_tol=0.05):
        failures.append("ABSTRACT_LANE_WIDTH_MISMATCH")

    max_station = S_BASE_M + EGO_SPEED_MPS * spec.duration_s
    if max_station >= 200.0 - config.min_straight_remaining_m:
        failures.append("MAP_STRAIGHT_EXCEEDED")

    grid = _time_grid(spec.duration_s, config.sweep_dt_s)
    ego_go = tuple(_nominal_ego_box(spec, time_s, config) for time_s in grid)
    trailer = tuple(_trailer_box(spec, time_s) for time_s in grid)
    latent = tuple(_latent_box(spec, time_s, config) for time_s in grid)
    supports = {
        actor.label: tuple(
            _support_box(
                spec,
                actor.label,
                actor.lane,
                actor.initial_local_s_m,
                actor.speed_mps,
                actor.length_m,
                actor.width_m,
                time_s,
                config,
            )
            for time_s in grid
        )
        for actor in spec.support_actors
    }

    visible_flags: list[bool] = []
    visible_fractions: list[float] = []
    for ego, truck, target in zip(ego_go, trailer, latent):
        visible, fraction = _visibility_fraction(
            ego,
            truck,
            target,
            config,
            minimum_visible_fraction=spec.visibility.minimum_visible_fraction,
            conservative_boundary_occlusion=spec.visibility.conservative_boundary_occlusion,
        )
        visible_flags.append(visible)
        visible_fractions.append(fraction)
    initial_fraction = visible_fractions[0]
    if visible_flags[0]:
        failures.append("VISIBLE_AT_START")
    reveal_index = next((index for index, flag in enumerate(visible_flags) if flag), None)
    reveal_time = None if reveal_index is None else grid[reveal_index]
    if reveal_time is None:
        failures.append("NO_NOMINAL_REVEAL")
    elif reveal_time < config.min_hidden_duration_s:
        failures.append("OCCLUSION_TOO_SHORT")

    conflict_time = float(spec.conflict.nominal_time_to_conflict_s)
    reveal_lead = None if reveal_time is None else conflict_time - reveal_time
    if reveal_lead is not None:
        if reveal_lead < spec.conflict.defensive_action_window_s - config.reveal_lead_tolerance_s:
            failures.append("ACTION_WINDOW_INFEASIBLE")
        if abs(reveal_lead - spec.conflict.target_reveal_lead_time_s) > config.reveal_lead_tolerance_s:
            failures.append("REVEAL_LEAD_MISMATCH")

    route_request_hidden_duration: float | None = None
    if reveal_time is not None:
        route_request_hidden_duration = float(
            reveal_time - spec.nominal_merge.route_request_start_s
        )
        if (
            config.require_route_request_during_occlusion
            and route_request_hidden_duration < config.min_route_request_hidden_s
        ):
            failures.append("ROUTE_REQUEST_NOT_DURING_OCCLUSION")

    latent_clearances = [signed_oriented_box_clearance(left, right) for left, right in zip(ego_go, latent)]
    trailer_clearances = [signed_oriented_box_clearance(left, right) for left, right in zip(ego_go, trailer)]
    conflict_index = min(range(len(grid)), key=lambda index: abs(grid[index] - conflict_time))
    # The nominal construction is evaluated up to its declared conflict.  It
    # is not a controller policy and is intentionally not continued through
    # the latent actor after the point where a real controller must react.
    go_latent_clearance = float(min(latent_clearances[:conflict_index + 1]))
    go_trailer_clearance = float(min(trailer_clearances[:conflict_index + 1]))
    conflict_clearance = float(latent_clearances[conflict_index])
    if abs(conflict_clearance - spec.conflict.nominal_unmitigated_clearance_m) > 0.12:
        failures.append("NOMINAL_CLEARANCE_MISMATCH")
    if go_trailer_clearance < config.minimum_clearance_m - 0.05:
        failures.append("NOMINAL_GO_TRAILER_UNSAFE")
    if reveal_index is not None and min(trailer_clearances[:reveal_index + 1]) < config.minimum_clearance_m - 0.05:
        failures.append("TRAILER_CONTACT_BEFORE_REVEAL")

    support_clearances: dict[str, float] = {}
    for label, support in supports.items():
        stations = [box.x for box in support]
        # The map check uses centre positions only; the runner's oriented-box
        # safety calculation separately accounts for vehicle footprint.
        if not all(math.isfinite(station) for station in stations):
            failures.append(f"SUPPORT_NONFINITE_{label}")
        clearances = [
            signed_oriented_box_clearance(ego_box, support_box)
            for ego_box, support_box in zip(ego_go, support)
        ]
        minimum = float(min(clearances))
        support_clearances[label] = minimum
        nonbinding_floor = max(
            config.minimum_clearance_m,
            go_latent_clearance + config.support_nonbinding_margin_m,
        )
        if minimum < nonbinding_floor - 0.05:
            failures.append(f"SUPPORT_ACTOR_BINDING_{label}")

    immediate_brake = tuple(_braking_ego_box(spec, time_s, 0.0, config) for time_s in grid)
    immediate_brake_clearance = float(min(
        signed_oriented_box_clearance(left, right) for left, right in zip(immediate_brake, trailer)
    ))
    if immediate_brake_clearance < config.minimum_clearance_m - 0.05:
        failures.append("IMMEDIATE_BRAKE_INFEASIBLE")

    reveal_brake_clearance: float | None = None
    if reveal_time is not None:
        reveal_brake = tuple(_braking_ego_box(spec, time_s, reveal_time, config) for time_s in grid)
        reveal_brake_clearance = float(min(
            signed_oriented_box_clearance(left, right) for left, right in zip(reveal_brake, trailer)
        ))
        if reveal_brake_clearance < config.minimum_clearance_m - 0.05:
            failures.append("REVEAL_BRAKE_INFEASIBLE")

    contact_actor = "latent_target_lane_vehicle" if go_latent_clearance <= go_trailer_clearance else "trailer"
    return ScenarioQualification(
        scenario_id=spec.scenario_id,
        pair_id=spec.pair_id,
        passed=not failures,
        failure_codes=tuple(sorted(set(failures))),
        initial_visible_fraction=float(initial_fraction),
        nominal_reveal_time_s=reveal_time,
        nominal_reveal_lead_time_s=reveal_lead,
        nominal_go_latent_clearance_m=go_latent_clearance,
        nominal_go_trailer_clearance_m=go_trailer_clearance,
        nominal_support_clearance_m=support_clearances,
        immediate_brake_trailer_clearance_m=immediate_brake_clearance,
        reveal_brake_trailer_clearance_m=reveal_brake_clearance,
        nominal_contact_actor=contact_actor,
        route_request_hidden_duration_s=route_request_hidden_duration,
        map_straight_until_s=200.0,
    )


def qualify_bank(
    bank: ScenarioBank,
    config: QualificationConfig = QualificationConfig(),
) -> Mapping[str, ScenarioQualification]:
    """Qualify one true-threat construction per pair for a paired bank."""

    results: dict[str, ScenarioQualification] = {}
    for pair_id, group in bank.paired_cases().items():
        true_threat = next(item for item in group if item.stratum is ScenarioStratum.TRUE_OCCLUDED_THREAT)
        results[pair_id] = qualify_scenario(true_threat, config)
    return results


def qualification_manifest(
    bank: ScenarioBank,
    config: QualificationConfig = QualificationConfig(),
) -> dict[str, Any]:
    results = qualify_bank(bank, config)
    return {
        "qualification_config": asdict(config),
        "pairs": {pair_id: result.manifest() for pair_id, result in results.items()},
        "all_passed": all(result.passed for result in results.values()),
    }


if __name__ == "__main__":
    from evaluation.occlusion_benchmark_scenarios import generate_paired_scenario_bank

    manifest = qualification_manifest(generate_paired_scenario_bank())
    print(manifest)
