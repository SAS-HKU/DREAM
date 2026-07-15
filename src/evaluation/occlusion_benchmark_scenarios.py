"""Deterministic, simulator-neutral occlusion benchmark specifications.

This module defines the *scenario contract* for the revised DREAM ablation.
It deliberately does not import a simulator, controller, NumPy, or the risk
field implementation.  An experiment adapter can convert :class:`ScenarioSpec`
objects into its own simulator state while retaining an auditable description of
the true hidden agent, the trailer-induced visibility geometry, and the paired
counterfactuals.

The default family models a trailer ahead of the ego that masks a vehicle in
the lane the ego would like to enter.  The ego reference trajectory is used
only to compute nominal conflict and visibility timing; it is *not* a forced
control trajectory.  A closed-loop controller remains free to hold, brake,
probe, or merge.

The three strata have intentionally different meanings:

``true_occluded_threat``
    A continuously existing latent vehicle is present in ground truth and is
    geometrically hidden until the trailer no longer blocks line of sight.

``empty_shadow``
    The road, trailer, ego, and counterfactual latent route are identical, but
    no latent vehicle is spawned.  This measures false-positive conservatism.

``visible_control``
    The true latent vehicle and all kinematics are identical to the threat
    case, but the observation stream is declared fully visible from time zero.
    It is an information-control condition, not a deployable baseline.

The geometry functions use conservative closed-set intersection semantics:
a sight line touching an occluder boundary is treated as occluded.  This avoids
declaring a barely grazing line of sight visible because of floating-point
round-off.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import math
from typing import Iterable, Mapping, Sequence


_EPS = 1.0e-9
Point2D = tuple[float, float]


class ScenarioStratum(str, Enum):
    """Paired scenario strata used in the revised safety benchmark."""

    TRUE_OCCLUDED_THREAT = "true_occluded_threat"
    EMPTY_SHADOW = "empty_shadow"
    VISIBLE_CONTROL = "visible_control"


class ObservationMode(str, Enum):
    """How the adapter should expose the latent vehicle to the controller."""

    GEOMETRIC_OCCLUSION = "geometric_occlusion"
    FULLY_VISIBLE = "fully_visible"


def _require_finite(name: str, *values: float) -> None:
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError(f"{name} values must be finite")


def _add(first: Point2D, second: Point2D) -> Point2D:
    return first[0] + second[0], first[1] + second[1]


def _subtract(first: Point2D, second: Point2D) -> Point2D:
    return first[0] - second[0], first[1] - second[1]


def _scale(point: Point2D, factor: float) -> Point2D:
    return point[0] * factor, point[1] * factor


def _dot(first: Point2D, second: Point2D) -> float:
    return first[0] * second[0] + first[1] * second[1]


def _cross(first: Point2D, second: Point2D) -> float:
    return first[0] * second[1] - first[1] * second[0]


def _distance(first: Point2D, second: Point2D) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


@dataclass(frozen=True)
class OrientedRectangle:
    """A convex, oriented rectangular footprint in a global 2-D frame."""

    center_x_m: float
    center_y_m: float
    heading_rad: float
    length_m: float
    width_m: float
    label: str = "occluder"

    def __post_init__(self) -> None:
        _require_finite(
            "OrientedRectangle",
            self.center_x_m,
            self.center_y_m,
            self.heading_rad,
            self.length_m,
            self.width_m,
        )
        if self.length_m <= 0.0 or self.width_m <= 0.0:
            raise ValueError("Rectangle length and width must be positive")
        if not self.label:
            raise ValueError("Rectangle label must be non-empty")

    @property
    def center(self) -> Point2D:
        return self.center_x_m, self.center_y_m

    @property
    def longitudinal_axis(self) -> Point2D:
        return math.cos(self.heading_rad), math.sin(self.heading_rad)

    @property
    def lateral_axis(self) -> Point2D:
        longitudinal = self.longitudinal_axis
        return -longitudinal[1], longitudinal[0]

    def corners(self) -> tuple[Point2D, Point2D, Point2D, Point2D]:
        """Return corners in counter-clockwise order."""

        longitudinal = _scale(self.longitudinal_axis, 0.5 * self.length_m)
        lateral = _scale(self.lateral_axis, 0.5 * self.width_m)
        center = self.center
        return (
            _subtract(_subtract(center, longitudinal), lateral),
            _add(_subtract(center, lateral), longitudinal),
            _add(_add(center, longitudinal), lateral),
            _add(_subtract(center, longitudinal), lateral),
        )


@dataclass(frozen=True)
class LaneAlignedTrajectory:
    """Continuous constant-acceleration motion along a lane centreline.

    ``initial_*`` values describe the vehicle centre at ``start_time_s``.
    The class is intentionally simple: it is a ground-truth/reference route
    for a benchmark, not a replacement for the simulation dynamics model.
    """

    initial_x_m: float
    initial_y_m: float
    heading_rad: float
    initial_speed_mps: float
    longitudinal_acceleration_mps2: float = 0.0
    start_time_s: float = 0.0
    length_m: float = 4.8
    width_m: float = 2.0
    label: str = "vehicle"

    def __post_init__(self) -> None:
        _require_finite(
            "LaneAlignedTrajectory",
            self.initial_x_m,
            self.initial_y_m,
            self.heading_rad,
            self.initial_speed_mps,
            self.longitudinal_acceleration_mps2,
            self.start_time_s,
            self.length_m,
            self.width_m,
        )
        if self.initial_speed_mps < 0.0:
            raise ValueError("Initial speed must be non-negative")
        if self.length_m <= 0.0 or self.width_m <= 0.0:
            raise ValueError("Vehicle length and width must be positive")
        if not self.label:
            raise ValueError("Trajectory label must be non-empty")

    def elapsed_s(self, time_s: float) -> float:
        """Elapsed trajectory time, clamped before the declared start time."""

        _require_finite("time_s", time_s)
        return max(0.0, time_s - self.start_time_s)

    def longitudinal_distance_m(self, time_s: float) -> float:
        elapsed = self.elapsed_s(time_s)
        return (
            self.initial_speed_mps * elapsed
            + 0.5 * self.longitudinal_acceleration_mps2 * elapsed * elapsed
        )

    def speed_mps(self, time_s: float) -> float:
        elapsed = self.elapsed_s(time_s)
        return max(0.0, self.initial_speed_mps + self.longitudinal_acceleration_mps2 * elapsed)

    def position_at(self, time_s: float) -> Point2D:
        distance = self.longitudinal_distance_m(time_s)
        direction = math.cos(self.heading_rad), math.sin(self.heading_rad)
        return (
            self.initial_x_m + distance * direction[0],
            self.initial_y_m + distance * direction[1],
        )

    def footprint_at(self, time_s: float) -> OrientedRectangle:
        position = self.position_at(time_s)
        return OrientedRectangle(
            center_x_m=position[0],
            center_y_m=position[1],
            heading_rad=self.heading_rad,
            length_m=self.length_m,
            width_m=self.width_m,
            label=self.label,
        )


@dataclass(frozen=True)
class EgoParameters:
    """Initial/reference ego state plus observation geometry."""

    reference_trajectory: LaneAlignedTrajectory
    desired_maneuver: str = "merge_left_to_pass_trailer"
    sensor_forward_offset_m: float = 1.5
    sensor_lateral_offset_m: float = 0.0
    sensor_range_m: float = 120.0

    def __post_init__(self) -> None:
        _require_finite(
            "EgoParameters",
            self.sensor_forward_offset_m,
            self.sensor_lateral_offset_m,
            self.sensor_range_m,
        )
        if self.sensor_range_m <= 0.0:
            raise ValueError("Sensor range must be positive")
        if not self.desired_maneuver:
            raise ValueError("Desired maneuver must be non-empty")

    def sensor_origin_at(self, time_s: float) -> Point2D:
        """Return the fixed body-frame sensor origin in global coordinates."""

        center = self.reference_trajectory.position_at(time_s)
        heading = self.reference_trajectory.heading_rad
        forward = math.cos(heading), math.sin(heading)
        left = -forward[1], forward[0]
        return _add(
            center,
            _add(
                _scale(forward, self.sensor_forward_offset_m),
                _scale(left, self.sensor_lateral_offset_m),
            ),
        )


@dataclass(frozen=True)
class TrailerGeometry:
    """Dynamic trailer footprint that creates the geometric visibility mask."""

    trajectory: LaneAlignedTrajectory
    role: str = "dynamic_occluding_trailer"

    def __post_init__(self) -> None:
        if not self.role:
            raise ValueError("Trailer role must be non-empty")

    def footprint_at(self, time_s: float) -> OrientedRectangle:
        footprint = self.trajectory.footprint_at(time_s)
        return replace(footprint, label=self.role)


@dataclass(frozen=True)
class ConflictSeverity:
    """Predeclared counterfactual severity for a scenario cell.

    The values are set before running any controller variant.  They describe
    the nominal conflict if the ego continues the designated maneuver, and
    prevent post-hoc selection of only dramatic episodes.
    """

    label: str
    nominal_time_to_conflict_s: float
    nominal_unmitigated_clearance_m: float
    target_reveal_lead_time_s: float
    defensive_action_window_s: float

    def __post_init__(self) -> None:
        _require_finite(
            "ConflictSeverity",
            self.nominal_time_to_conflict_s,
            self.nominal_unmitigated_clearance_m,
            self.target_reveal_lead_time_s,
            self.defensive_action_window_s,
        )
        if not self.label:
            raise ValueError("Severity label must be non-empty")
        if self.nominal_time_to_conflict_s <= 0.0:
            raise ValueError("Nominal conflict time must be positive")
        if self.target_reveal_lead_time_s < 0.0:
            raise ValueError("Reveal lead time cannot be negative")
        if self.target_reveal_lead_time_s >= self.nominal_time_to_conflict_s:
            raise ValueError("Reveal lead time must precede the nominal conflict")
        if self.defensive_action_window_s <= 0.0:
            raise ValueError("Defensive action window must be positive")


@dataclass(frozen=True)
class NominalMergeReference:
    """Pre-registered construction path for qualifying a target-lane conflict.

    This reference is used only to construct and accept a scenario.  It is
    never injected into the evaluated controller.  At runtime, a route-level
    request may be accepted, delayed, or rejected by the ordinary IDEAM probe
    guard and DREAM veto.
    """

    target_lane: int = 2
    lane_change_start_s: float = 0.0
    lane_change_duration_s: float = 2.5
    route_request_start_s: float = 0.0
    route_request_end_s: float | None = None
    lateral_shift_m: float = 3.5

    def __post_init__(self) -> None:
        if int(self.target_lane) not in (0, 1, 2):
            raise ValueError("NominalMergeReference.target_lane must be 0, 1, or 2")
        _require_finite(
            "NominalMergeReference",
            self.lane_change_start_s,
            self.lane_change_duration_s,
            self.route_request_start_s,
            self.lateral_shift_m,
        )
        if self.lane_change_start_s < 0.0 or self.route_request_start_s < 0.0:
            raise ValueError("Nominal merge times must be non-negative")
        if self.route_request_end_s is not None:
            _require_finite("NominalMergeReference.route_request_end_s", self.route_request_end_s)
            if self.route_request_end_s < self.route_request_start_s:
                raise ValueError("Route-request end time must follow its start time")
        if self.lane_change_duration_s <= 0.0:
            raise ValueError("Nominal lane-change duration must be positive")
        if self.lateral_shift_m <= 0.0:
            raise ValueError("Nominal lateral shift must be positive")


@dataclass(frozen=True)
class OpenLoopSupportActor:
    """Visible background vehicle shared by every member of a paired case.

    Support actors exist to give the legacy IDEAM interface an ordinary
    leader/follower relation.  They are *not* latent hazards and must be
    checked as non-binding by the controller-independent qualification.
    Their station is expressed relative to the benchmark ego's initial
    centre-lane station, rather than as an unlogged simulator-side global.
    """

    label: str
    lane: int
    initial_local_s_m: float
    speed_mps: float
    length_m: float = 4.8
    width_m: float = 2.0

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("OpenLoopSupportActor.label must be non-empty")
        if (
            isinstance(self.lane, bool)
            or not isinstance(self.lane, int)
            or self.lane not in (0, 1, 2)
        ):
            raise ValueError("OpenLoopSupportActor.lane must be 0, 1, or 2")
        _require_finite(
            "OpenLoopSupportActor",
            self.initial_local_s_m,
            self.speed_mps,
            self.length_m,
            self.width_m,
        )
        if self.speed_mps < 0.0 or self.length_m <= 0.0 or self.width_m <= 0.0:
            raise ValueError("Support-actor speed must be non-negative and dimensions positive")


@dataclass(frozen=True)
class ScenarioConstruction:
    """Physical parameters used to create one pre-registered severity cell."""

    label: str
    nominal_time_to_conflict_s: float
    trailer_speed_mps: float
    latent_speed_mps: float
    trailer_bumper_gap_m: float
    nominal_clearance_m: float
    lane_change_start_s: float
    lane_change_duration_s: float
    route_request_start_s: float
    target_reveal_lead_time_s: float
    defensive_action_window_s: float
    duration_s: float
    construction_id: str = ""
    route_request_end_s: float | None = None

    def __post_init__(self) -> None:
        _require_finite(
            "ScenarioConstruction",
            self.nominal_time_to_conflict_s,
            self.trailer_speed_mps,
            self.latent_speed_mps,
            self.trailer_bumper_gap_m,
            self.nominal_clearance_m,
            self.lane_change_start_s,
            self.lane_change_duration_s,
            self.route_request_start_s,
            self.target_reveal_lead_time_s,
            self.defensive_action_window_s,
            self.duration_s,
        )
        if not self.label:
            raise ValueError("ScenarioConstruction.label must be non-empty")
        if self.construction_id and not self.construction_id.replace("_", "").isalnum():
            raise ValueError("ScenarioConstruction.construction_id must be alphanumeric/underscore")
        if min(
            self.nominal_time_to_conflict_s,
            self.trailer_speed_mps,
            self.latent_speed_mps,
            self.trailer_bumper_gap_m,
            self.nominal_clearance_m,
            self.lane_change_duration_s,
            self.defensive_action_window_s,
            self.duration_s,
        ) <= 0.0:
            raise ValueError("ScenarioConstruction values must be positive")
        if self.lane_change_start_s < 0.0 or self.route_request_start_s < 0.0:
            raise ValueError("ScenarioConstruction times must be non-negative")
        if self.route_request_end_s is not None:
            _require_finite("ScenarioConstruction.route_request_end_s", self.route_request_end_s)
            if self.route_request_end_s < self.route_request_start_s:
                raise ValueError("Route-request end time must follow its start time")
        if self.duration_s <= self.nominal_time_to_conflict_s:
            raise ValueError("Scenario duration must extend beyond conflict time")


@dataclass(frozen=True)
class VisibilityParameters:
    """Stable rules for declaring a latent vehicle geometrically visible."""

    # A majority of the deterministic footprint probes must be visible.  A
    # single exposed corner is too fragile to constitute a robust perception
    # reveal for this benchmark.
    minimum_visible_fraction: float = 5.0 / 9.0
    reveal_search_step_s: float = 0.05
    conservative_boundary_occlusion: bool = True

    def __post_init__(self) -> None:
        _require_finite(
            "VisibilityParameters",
            self.minimum_visible_fraction,
            self.reveal_search_step_s,
        )
        if not 0.0 < self.minimum_visible_fraction <= 1.0:
            raise ValueError("Minimum visible fraction must be in (0, 1]")
        if self.reveal_search_step_s <= 0.0:
            raise ValueError("Reveal search step must be positive")


@dataclass(frozen=True)
class VisibilityAssessment:
    """Visibility of a target footprint from one ego sensor location."""

    visible: bool
    visible_sample_count: int
    sample_count: int
    visible_fraction: float
    blocked_sample_count: int
    out_of_range_sample_count: int


@dataclass(frozen=True)
class RevealEvent:
    """First geometric observation event for a scenario's latent route."""

    reveal_time_s: float | None
    visible_fraction: float
    actual_reveal_lead_time_s: float | None
    initial_visible: bool
    search_end_time_s: float


@dataclass(frozen=True)
class ScenarioSpec:
    """Complete, immutable specification of one benchmark episode.

    ``counterfactual_latent_trajectory`` is present in every stratum.  It is a
    real ground-truth actor only when ``latent_present`` is true.  Retaining it
    in empty-shadow cases lets the field study compare risk in the same
    plausible-but-empty occupancy tube without introducing a physical agent.
    """

    scenario_id: str
    pair_id: str
    family_id: str
    seed: int
    stratum: ScenarioStratum
    ego: EgoParameters
    trailer: TrailerGeometry
    counterfactual_latent_trajectory: LaneAlignedTrajectory
    latent_present: bool
    conflict: ConflictSeverity
    visibility: VisibilityParameters
    duration_s: float
    observation_mode: ObservationMode
    nominal_merge: NominalMergeReference = NominalMergeReference()
    support_actors: tuple[OpenLoopSupportActor, ...] = ()
    static_occluders: tuple[OrientedRectangle, ...] = ()
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_finite("ScenarioSpec", self.duration_s)
        if not self.scenario_id or not self.pair_id or not self.family_id:
            raise ValueError("Scenario, pair, and family identifiers must be non-empty")
        if self.duration_s <= self.conflict.nominal_time_to_conflict_s:
            raise ValueError("Duration must extend beyond the nominal conflict time")
        if self.nominal_merge.target_lane == 1:
            raise ValueError("Nominal merge target must differ from the ego lane")
        support_labels = [actor.label for actor in self.support_actors]
        if len(set(support_labels)) != len(support_labels):
            raise ValueError("Support-actor labels must be unique within a scenario")
        reserved = {
            "trailer",
            "latent_target_lane_vehicle",
            "counterfactual_latent_target_lane_vehicle",
            "ego",
        }
        if reserved.intersection(support_labels):
            raise ValueError("Support-actor labels collide with reserved benchmark actors")
        if self.stratum is ScenarioStratum.TRUE_OCCLUDED_THREAT:
            if not self.latent_present:
                raise ValueError("A true threat must contain a latent vehicle")
            if self.observation_mode is not ObservationMode.GEOMETRIC_OCCLUSION:
                raise ValueError("A true threat must use geometric occlusion")
        elif self.stratum is ScenarioStratum.EMPTY_SHADOW:
            if self.latent_present:
                raise ValueError("An empty shadow must not contain a latent vehicle")
            if self.observation_mode is not ObservationMode.GEOMETRIC_OCCLUSION:
                raise ValueError("An empty shadow must retain geometric occlusion")
        elif self.stratum is ScenarioStratum.VISIBLE_CONTROL:
            if not self.latent_present:
                raise ValueError("A visible control must contain a latent vehicle")
            if self.observation_mode is not ObservationMode.FULLY_VISIBLE:
                raise ValueError("A visible control must expose the latent vehicle")

    def ego_footprint_at(self, time_s: float) -> OrientedRectangle:
        return self.ego.reference_trajectory.footprint_at(time_s)

    def latent_footprint_at(self, time_s: float) -> OrientedRectangle | None:
        """Return the real latent vehicle only when it exists in ground truth."""

        if not self.latent_present:
            return None
        return self.counterfactual_latent_trajectory.footprint_at(time_s)

    def counterfactual_latent_footprint_at(self, time_s: float) -> OrientedRectangle:
        """Return the fixed hypothetical route in all three strata."""

        return self.counterfactual_latent_trajectory.footprint_at(time_s)

    def physical_occluders_at(self, time_s: float) -> tuple[OrientedRectangle, ...]:
        """Trailer/static geometry, independent of the observation policy."""

        return (self.trailer.footprint_at(time_s), *self.static_occluders)

    def observation_occluders_at(self, time_s: float) -> tuple[OrientedRectangle, ...]:
        """Obstacles that hide the latent agent from the controller."""

        if self.observation_mode is ObservationMode.FULLY_VISIBLE:
            return ()
        return self.physical_occluders_at(time_s)

    def visibility_at(self, time_s: float) -> VisibilityAssessment:
        """Evaluate the counterfactual route's observation state at ``time_s``."""

        target = self.counterfactual_latent_footprint_at(time_s)
        if self.observation_mode is ObservationMode.FULLY_VISIBLE:
            sample_count = len(oriented_rectangle_visibility_samples(target))
            return VisibilityAssessment(
                visible=True,
                visible_sample_count=sample_count,
                sample_count=sample_count,
                visible_fraction=1.0,
                blocked_sample_count=0,
                out_of_range_sample_count=0,
            )
        return assess_oriented_rectangle_visibility(
            observer=self.ego.sensor_origin_at(time_s),
            target=target,
            occluders=self.observation_occluders_at(time_s),
            sensor_range_m=self.ego.sensor_range_m,
            minimum_visible_fraction=self.visibility.minimum_visible_fraction,
            conservative_boundary_occlusion=self.visibility.conservative_boundary_occlusion,
        )

    def geometric_reveal(self, *, search_end_time_s: float | None = None) -> RevealEvent:
        """Find the first time at which the counterfactual target is visible."""

        return compute_geometric_reveal(self, search_end_time_s=search_end_time_s)


@dataclass(frozen=True)
class ScenarioFamily:
    """Reusable physical template for a deterministic scenario bank."""

    family_id: str = "route_conditioned_trailer_merge_v2"
    # Match the active IDEAM map (`Path.path`), whose lane centres are 3.5 m
    # apart on the benchmark's first straight.
    lane_width_m: float = 3.5
    ego_speed_mps: float = 18.0
    latent_speed_mps: float = 10.0
    trailer_length_m: float = 18.0
    trailer_width_m: float = 3.0
    ego_length_m: float = 4.8
    ego_width_m: float = 2.0
    latent_length_m: float = 4.8
    latent_width_m: float = 2.0
    sensor_forward_offset_m: float = 1.5
    sensor_range_m: float = 120.0
    episode_tail_s: float = 3.0

    def __post_init__(self) -> None:
        _require_finite(
            "ScenarioFamily",
            self.lane_width_m,
            self.ego_speed_mps,
            self.latent_speed_mps,
            self.trailer_length_m,
            self.trailer_width_m,
            self.ego_length_m,
            self.ego_width_m,
            self.latent_length_m,
            self.latent_width_m,
            self.sensor_forward_offset_m,
            self.sensor_range_m,
            self.episode_tail_s,
        )
        if not self.family_id:
            raise ValueError("Family identifier must be non-empty")
        if self.lane_width_m <= 0.0:
            raise ValueError("Lane width must be positive")
        if self.ego_speed_mps <= self.latent_speed_mps:
            raise ValueError("Ego reference speed must exceed latent speed for this family")
        if min(
            self.trailer_length_m,
            self.trailer_width_m,
            self.ego_length_m,
            self.ego_width_m,
            self.latent_length_m,
            self.latent_width_m,
            self.sensor_range_m,
            self.episode_tail_s,
        ) <= 0.0:
            raise ValueError("Family dimensions and horizon quantities must be positive")


@dataclass(frozen=True)
class ScenarioBankConfig:
    """Frozen pre-run design for the paired scenario generator."""

    family: ScenarioFamily = ScenarioFamily()
    constructions: tuple[ScenarioConstruction, ...] = (
        # Held-out v2 bank: these cells were frozen after development-only
        # calibration and before their controller outcomes were inspected.
        # Five controller-independent constructions per predeclared severity.
        # They were retained by map-geometry, footprint, and braking
        # qualification only—not by a controller outcome.  Each expands into
        # matched true-threat, empty-shadow, and visible-control episodes.
        ScenarioConstruction("critical", 2.90, 18.0, 4.10, 3.8, 0.25, 0.45, 2.5, 0.45, 1.55, 0.90, 5.0, "critical_01"),
        ScenarioConstruction("critical", 2.95, 17.9, 4.20, 3.9, 0.25, 0.50, 2.5, 0.50, 1.55, 0.90, 5.0, "critical_02"),
        ScenarioConstruction("critical", 3.05, 18.1, 4.30, 4.0, 0.25, 0.50, 2.5, 0.50, 1.55, 0.90, 5.1, "critical_03"),
        ScenarioConstruction("critical", 3.10, 18.0, 4.40, 4.1, 0.25, 0.55, 2.5, 0.55, 1.55, 0.90, 5.1, "critical_04"),
        ScenarioConstruction("critical", 3.00, 18.2, 4.60, 4.2, 0.25, 0.45, 2.5, 0.45, 1.55, 0.90, 5.0, "critical_05"),
        ScenarioConstruction("moderate", 3.05, 17.9, 4.70, 3.9, 0.25, 0.45, 2.5, 0.45, 1.55, 0.90, 5.1, "moderate_01"),
        ScenarioConstruction("moderate", 3.10, 18.0, 4.80, 4.0, 0.25, 0.50, 2.5, 0.50, 1.55, 0.90, 5.1, "moderate_02"),
        ScenarioConstruction("moderate", 3.15, 18.1, 4.90, 4.1, 0.25, 0.55, 2.5, 0.55, 1.55, 0.90, 5.2, "moderate_03"),
        ScenarioConstruction("moderate", 3.00, 18.2, 5.05, 4.2, 0.25, 0.50, 2.5, 0.50, 1.55, 0.90, 5.0, "moderate_04"),
        ScenarioConstruction("moderate", 3.20, 18.0, 5.15, 4.3, 0.25, 0.60, 2.5, 0.60, 1.55, 0.90, 5.2, "moderate_05"),
        ScenarioConstruction("mild", 3.15, 17.9, 5.30, 4.0, 0.25, 0.45, 2.5, 0.45, 1.55, 0.90, 5.2, "mild_01"),
        ScenarioConstruction("mild", 3.20, 18.0, 5.45, 4.1, 0.25, 0.50, 2.5, 0.50, 1.55, 0.90, 5.2, "mild_02"),
        ScenarioConstruction("mild", 3.25, 18.1, 5.60, 4.2, 0.25, 0.55, 2.5, 0.55, 1.55, 0.90, 5.3, "mild_03"),
        ScenarioConstruction("mild", 3.10, 18.2, 5.75, 4.3, 0.25, 0.50, 2.5, 0.50, 1.55, 0.90, 5.2, "mild_04"),
        ScenarioConstruction("mild", 3.30, 18.0, 5.90, 4.4, 0.25, 0.60, 2.5, 0.60, 1.55, 0.90, 5.4, "mild_05"),
    )
    # Do not mistake a deterministic repeat for a new statistical scenario.
    # Each construction above is unique and explicitly parameterized.
    replicates_per_cell: int = 1
    base_seed: int = 20260713
    reveal_lead_tolerance_s: float = 0.20
    visibility: VisibilityParameters = VisibilityParameters(
        minimum_visible_fraction=7.0 / 9.0,
    )
    support_actors: tuple[OpenLoopSupportActor, ...] = (
        # This rear follower activates the legacy target-follower constraint
        # but is explicitly qualified as non-binding.  There is deliberately
        # no target-lane leader/gate.
        OpenLoopSupportActor("target_lane_rear_follower", 2, -18.0, 9.0),
    )

    def __post_init__(self) -> None:
        _require_finite("ScenarioBankConfig", self.reveal_lead_tolerance_s)
        if self.replicates_per_cell <= 0:
            raise ValueError("Replicates per cell must be positive")
        if self.reveal_lead_tolerance_s < 0.0:
            raise ValueError("Reveal-lead tolerance cannot be negative")
        if not self.constructions:
            raise ValueError("At least one scenario construction is required")
        construction_ids = [
            item.construction_id or f"{item.label}_{index:02d}"
            for index, item in enumerate(self.constructions)
        ]
        if len(set(construction_ids)) != len(construction_ids):
            raise ValueError("Scenario construction IDs must be unique")
        support_labels = [actor.label for actor in self.support_actors]
        if len(set(support_labels)) != len(support_labels):
            raise ValueError("Scenario-bank support-actor labels must be unique")


@dataclass(frozen=True)
class ScenarioBank:
    """Immutable collection of paired specifications and design diagnostics."""

    config: ScenarioBankConfig
    scenarios: tuple[ScenarioSpec, ...]

    def __post_init__(self) -> None:
        if not self.scenarios:
            raise ValueError("Scenario bank must not be empty")
        validate_paired_scenario_bank(self.scenarios)

    def by_stratum(self, stratum: ScenarioStratum) -> tuple[ScenarioSpec, ...]:
        return tuple(item for item in self.scenarios if item.stratum is stratum)

    def paired_cases(self) -> Mapping[str, tuple[ScenarioSpec, ...]]:
        grouped: dict[str, list[ScenarioSpec]] = {}
        for scenario in self.scenarios:
            grouped.setdefault(scenario.pair_id, []).append(scenario)
        return {
            pair_id: tuple(sorted(items, key=lambda item: item.stratum.value))
            for pair_id, items in grouped.items()
        }

    def reveal_diagnostics(self) -> Mapping[str, RevealEvent]:
        """Return reveal events for true-threat cells only."""

        return {
            scenario.scenario_id: scenario.geometric_reveal()
            for scenario in self.by_stratum(ScenarioStratum.TRUE_OCCLUDED_THREAT)
        }

    def validate_geometric_reveals(self) -> Mapping[str, RevealEvent]:
        """Verify that every true-threat cell has the frozen reveal design.

        This preflight check belongs before controller experiments.  It catches
        a scenario that is visible from the start, never reveals, or drifts too
        far from the predeclared target reveal lead time.
        """

        diagnostics = self.reveal_diagnostics()
        errors: list[str] = []
        for scenario in self.by_stratum(ScenarioStratum.TRUE_OCCLUDED_THREAT):
            event = diagnostics[scenario.scenario_id]
            if event.initial_visible:
                errors.append(f"{scenario.scenario_id}: latent route is visible at t=0")
                continue
            if event.reveal_time_s is None or event.actual_reveal_lead_time_s is None:
                errors.append(f"{scenario.scenario_id}: no geometric reveal")
                continue
            lead_error = abs(
                event.actual_reveal_lead_time_s
                - scenario.conflict.target_reveal_lead_time_s
            )
            if lead_error > self.config.reveal_lead_tolerance_s:
                errors.append(
                    f"{scenario.scenario_id}: reveal-lead error {lead_error:.3f}s "
                    f"exceeds {self.config.reveal_lead_tolerance_s:.3f}s"
                )
        if errors:
            raise ValueError("Invalid geometric reveal design: " + "; ".join(errors))
        return diagnostics

    def summary(self) -> Mapping[str, int]:
        return {
            "paired_cases": len(self.paired_cases()),
            "total_scenarios": len(self.scenarios),
            "true_occluded_threat": len(self.by_stratum(ScenarioStratum.TRUE_OCCLUDED_THREAT)),
            "empty_shadow": len(self.by_stratum(ScenarioStratum.EMPTY_SHADOW)),
            "visible_control": len(self.by_stratum(ScenarioStratum.VISIBLE_CONTROL)),
        }


def _orientation(first: Point2D, second: Point2D, third: Point2D) -> float:
    return _cross(_subtract(second, first), _subtract(third, first))


def _point_on_segment(point: Point2D, start: Point2D, end: Point2D, *, epsilon: float) -> bool:
    if abs(_orientation(start, end, point)) > epsilon:
        return False
    return (
        min(start[0], end[0]) - epsilon <= point[0] <= max(start[0], end[0]) + epsilon
        and min(start[1], end[1]) - epsilon <= point[1] <= max(start[1], end[1]) + epsilon
    )


def line_segments_intersect(
    first_start: Point2D,
    first_end: Point2D,
    second_start: Point2D,
    second_end: Point2D,
    *,
    epsilon: float = _EPS,
) -> bool:
    """Return whether two closed 2-D segments intersect.

    Collinear overlap and endpoint contact count as intersection.  The
    implementation avoids slope division, so vertical and nearly parallel
    sight lines are handled without special cases.
    """

    for point in (first_start, first_end, second_start, second_end):
        _require_finite("line segment", *point)
    if epsilon < 0.0:
        raise ValueError("epsilon must be non-negative")

    first_second_start = _orientation(first_start, first_end, second_start)
    first_second_end = _orientation(first_start, first_end, second_end)
    second_first_start = _orientation(second_start, second_end, first_start)
    second_first_end = _orientation(second_start, second_end, first_end)

    if (
        (first_second_start > epsilon and first_second_end < -epsilon
         or first_second_start < -epsilon and first_second_end > epsilon)
        and (second_first_start > epsilon and second_first_end < -epsilon
             or second_first_start < -epsilon and second_first_end > epsilon)
    ):
        return True

    return (
        _point_on_segment(second_start, first_start, first_end, epsilon=epsilon)
        or _point_on_segment(second_end, first_start, first_end, epsilon=epsilon)
        or _point_on_segment(first_start, second_start, second_end, epsilon=epsilon)
        or _point_on_segment(first_end, second_start, second_end, epsilon=epsilon)
    )


def point_in_oriented_rectangle(
    point: Point2D,
    rectangle: OrientedRectangle,
    *,
    include_boundary: bool = True,
    epsilon: float = _EPS,
) -> bool:
    """Return whether a point lies in an oriented rectangle."""

    _require_finite("point", *point)
    relative = _subtract(point, rectangle.center)
    longitudinal = abs(_dot(relative, rectangle.longitudinal_axis))
    lateral = abs(_dot(relative, rectangle.lateral_axis))
    if include_boundary:
        return (
            longitudinal <= 0.5 * rectangle.length_m + epsilon
            and lateral <= 0.5 * rectangle.width_m + epsilon
        )
    return (
        longitudinal < 0.5 * rectangle.length_m - epsilon
        and lateral < 0.5 * rectangle.width_m - epsilon
    )


def line_segment_intersects_oriented_rectangle(
    start: Point2D,
    end: Point2D,
    rectangle: OrientedRectangle,
    *,
    conservative_boundary_occlusion: bool = True,
    epsilon: float = _EPS,
) -> bool:
    """Return whether a closed sight-line segment is blocked by a rectangle."""

    if point_in_oriented_rectangle(
        start,
        rectangle,
        include_boundary=conservative_boundary_occlusion,
        epsilon=epsilon,
    ):
        return True
    if point_in_oriented_rectangle(
        end,
        rectangle,
        include_boundary=conservative_boundary_occlusion,
        epsilon=epsilon,
    ):
        return True

    corners = rectangle.corners()
    for index, edge_start in enumerate(corners):
        edge_end = corners[(index + 1) % len(corners)]
        if line_segments_intersect(start, end, edge_start, edge_end, epsilon=epsilon):
            return True
    return False


def line_of_sight_clear(
    observer: Point2D,
    target_point: Point2D,
    occluders: Iterable[OrientedRectangle],
    *,
    sensor_range_m: float,
    conservative_boundary_occlusion: bool = True,
) -> bool:
    """Return whether a target point is within range and not ray-blocked."""

    _require_finite("observer", *observer)
    _require_finite("target point", *target_point)
    _require_finite("sensor range", sensor_range_m)
    if sensor_range_m <= 0.0:
        raise ValueError("Sensor range must be positive")
    if _distance(observer, target_point) > sensor_range_m + _EPS:
        return False
    return not any(
        line_segment_intersects_oriented_rectangle(
            observer,
            target_point,
            occluder,
            conservative_boundary_occlusion=conservative_boundary_occlusion,
        )
        for occluder in occluders
    )


def oriented_rectangle_visibility_samples(rectangle: OrientedRectangle) -> tuple[Point2D, ...]:
    """Return centre, corners, and edge midpoints for partial-visibility tests."""

    corners = rectangle.corners()
    edge_midpoints = tuple(
        _scale(_add(corners[index], corners[(index + 1) % len(corners)]), 0.5)
        for index in range(len(corners))
    )
    return rectangle.center, *corners, *edge_midpoints


def assess_oriented_rectangle_visibility(
    *,
    observer: Point2D,
    target: OrientedRectangle,
    occluders: Iterable[OrientedRectangle],
    sensor_range_m: float,
    minimum_visible_fraction: float = 5.0 / 9.0,
    conservative_boundary_occlusion: bool = True,
) -> VisibilityAssessment:
    """Assess partial target-footprint visibility from a sensor point.

    A target becomes visible once at least ``minimum_visible_fraction`` of the
    nine deterministic footprint samples have a clear line of sight.  This is
    more stable than using only its centre, while remaining light enough for a
    scenario-validation prepass.
    """

    _require_finite("minimum visible fraction", minimum_visible_fraction)
    if not 0.0 < minimum_visible_fraction <= 1.0:
        raise ValueError("Minimum visible fraction must be in (0, 1]")
    occluder_tuple = tuple(occluders)
    samples = oriented_rectangle_visibility_samples(target)
    visible = 0
    blocked = 0
    out_of_range = 0
    for sample in samples:
        if _distance(observer, sample) > sensor_range_m + _EPS:
            out_of_range += 1
        elif line_of_sight_clear(
            observer,
            sample,
            occluder_tuple,
            sensor_range_m=sensor_range_m,
            conservative_boundary_occlusion=conservative_boundary_occlusion,
        ):
            visible += 1
        else:
            blocked += 1
    fraction = visible / len(samples)
    return VisibilityAssessment(
        visible=fraction + _EPS >= minimum_visible_fraction,
        visible_sample_count=visible,
        sample_count=len(samples),
        visible_fraction=fraction,
        blocked_sample_count=blocked,
        out_of_range_sample_count=out_of_range,
    )


def compute_geometric_reveal(
    scenario: ScenarioSpec,
    *,
    search_end_time_s: float | None = None,
) -> RevealEvent:
    """Search the scenario clock for the first valid latent-agent reveal."""

    end_time = scenario.duration_s if search_end_time_s is None else search_end_time_s
    _require_finite("search end time", end_time)
    if end_time < 0.0:
        raise ValueError("Search end time cannot be negative")
    initial = scenario.visibility_at(0.0)
    step = scenario.visibility.reveal_search_step_s
    step_count = int(math.floor(end_time / step + _EPS))
    for index in range(step_count + 1):
        time_s = min(end_time, index * step)
        assessment = initial if index == 0 else scenario.visibility_at(time_s)
        if assessment.visible:
            lead = scenario.conflict.nominal_time_to_conflict_s - time_s
            return RevealEvent(
                reveal_time_s=time_s,
                visible_fraction=assessment.visible_fraction,
                actual_reveal_lead_time_s=lead,
                initial_visible=initial.visible,
                search_end_time_s=end_time,
            )
    return RevealEvent(
        reveal_time_s=None,
        visible_fraction=0.0,
        actual_reveal_lead_time_s=None,
        initial_visible=initial.visible,
        search_end_time_s=end_time,
    )


def _stable_jitter(seed: int, amplitude: float) -> float:
    """Small deterministic offset without relying on Python's randomized hash."""

    value = (int(seed) & 0xFFFFFFFFFFFFFFFF) + 0x9E3779B97F4A7C15
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
    value = (value ^ (value >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
    value ^= value >> 31
    unit = (value & 0xFFFFFFFF) / float(0xFFFFFFFF)
    return (2.0 * unit - 1.0) * amplitude


def _make_paired_specs(
    *,
    family: ScenarioFamily,
    construction: ScenarioConstruction,
    visibility: VisibilityParameters,
    support_actors: tuple[OpenLoopSupportActor, ...],
    pair_id: str,
    seed: int,
) -> tuple[ScenarioSpec, ScenarioSpec, ScenarioSpec]:
    """Create matched true-threat, empty-shadow, and visible-control specs."""

    relative_speed = family.ego_speed_mps - construction.latent_speed_mps
    if relative_speed <= 0.0:
        raise ValueError("Construction latent speed must be below ego speed")
    severity = ConflictSeverity(
        label=construction.label,
        nominal_time_to_conflict_s=construction.nominal_time_to_conflict_s,
        nominal_unmitigated_clearance_m=construction.nominal_clearance_m,
        target_reveal_lead_time_s=construction.target_reveal_lead_time_s,
        defensive_action_window_s=construction.defensive_action_window_s,
    )
    nominal_merge = NominalMergeReference(
        target_lane=2,
        lane_change_start_s=construction.lane_change_start_s,
        lane_change_duration_s=construction.lane_change_duration_s,
        route_request_start_s=construction.route_request_start_s,
        route_request_end_s=construction.route_request_end_s,
        lateral_shift_m=family.lane_width_m,
    )
    ego_reference = LaneAlignedTrajectory(
        initial_x_m=0.0,
        initial_y_m=0.0,
        heading_rad=0.0,
        initial_speed_mps=family.ego_speed_mps,
        length_m=family.ego_length_m,
        width_m=family.ego_width_m,
        label="ego_reference",
    )
    ego = EgoParameters(
        reference_trajectory=ego_reference,
        desired_maneuver="route-requested pass around an occluding trailer",
        sensor_forward_offset_m=family.sensor_forward_offset_m,
        sensor_range_m=family.sensor_range_m,
    )
    # Include both vehicle half-lengths when encoding the declared target-lane
    # clearance at the nominal conflict time.
    latent_x = (
        0.5 * (family.ego_length_m + family.latent_length_m)
        + construction.nominal_clearance_m
        + relative_speed * construction.nominal_time_to_conflict_s
    )
    latent = LaneAlignedTrajectory(
        initial_x_m=latent_x,
        initial_y_m=family.lane_width_m,
        heading_rad=0.0,
        initial_speed_mps=construction.latent_speed_mps,
        length_m=family.latent_length_m,
        width_m=family.latent_width_m,
        label="latent_target_lane_vehicle",
    )

    # Truck placement is determined by a bumper-gap braking envelope, never
    # by driving a reference sensor through the truck at the reveal time.
    trailer_x = (
        0.5 * family.ego_length_m
        + construction.trailer_bumper_gap_m
        + 0.5 * family.trailer_length_m
    )
    trailer_trajectory = LaneAlignedTrajectory(
        initial_x_m=trailer_x,
        initial_y_m=0.0,
        heading_rad=0.0,
        initial_speed_mps=construction.trailer_speed_mps,
        length_m=family.trailer_length_m,
        width_m=family.trailer_width_m,
        label="trailer",
    )
    trailer = TrailerGeometry(trajectory=trailer_trajectory)
    common = dict(
        pair_id=pair_id,
        family_id=family.family_id,
        seed=seed,
        ego=ego,
        trailer=trailer,
        counterfactual_latent_trajectory=latent,
        conflict=severity,
        visibility=visibility,
        duration_s=construction.duration_s,
        nominal_merge=nominal_merge,
        support_actors=support_actors,
        notes=(
            "Paired counterpart uses identical ego, trailer, and counterfactual latent route.",
            "A map-coordinate nominal merge is used only for scenario qualification.",
            "Runtime visibility is evaluated from the actual ego pose; no hidden actor is timer-spawned.",
        ),
    )
    return (
        ScenarioSpec(
            scenario_id=f"{pair_id}__{ScenarioStratum.TRUE_OCCLUDED_THREAT.value}",
            stratum=ScenarioStratum.TRUE_OCCLUDED_THREAT,
            latent_present=True,
            observation_mode=ObservationMode.GEOMETRIC_OCCLUSION,
            **common,
        ),
        ScenarioSpec(
            scenario_id=f"{pair_id}__{ScenarioStratum.EMPTY_SHADOW.value}",
            stratum=ScenarioStratum.EMPTY_SHADOW,
            latent_present=False,
            observation_mode=ObservationMode.GEOMETRIC_OCCLUSION,
            **common,
        ),
        ScenarioSpec(
            scenario_id=f"{pair_id}__{ScenarioStratum.VISIBLE_CONTROL.value}",
            stratum=ScenarioStratum.VISIBLE_CONTROL,
            latent_present=True,
            observation_mode=ObservationMode.FULLY_VISIBLE,
            **common,
        ),
    )


def generate_paired_scenario_bank(config: ScenarioBankConfig | None = None) -> ScenarioBank:
    """Generate the deterministic, three-stratum occlusion benchmark bank.

    The frozen construction list spans severity and physical-geometry
    variation.  Every base case expands into exactly three same-geometry
    strata.  The config should be serialized/frozen before the held-out
    aggregate run.
    """

    config = ScenarioBankConfig() if config is None else config
    if config.replicates_per_cell != 1:
        raise ValueError(
            "Repeated deterministic cells are not valid independent scenarios; "
            "add distinct ScenarioConstruction entries instead."
        )
    scenarios: list[ScenarioSpec] = []
    for case_index, construction in enumerate(config.constructions):
        seed = config.base_seed + 1009 * case_index
        construction_id = construction.construction_id or f"{construction.label}_{case_index:02d}"
        pair_id = f"{config.family.family_id}__{construction_id}__case_{case_index:02d}"
        scenarios.extend(
            _make_paired_specs(
                family=config.family,
                construction=construction,
                visibility=config.visibility,
                support_actors=config.support_actors,
                pair_id=pair_id,
                seed=seed,
            )
        )
    return ScenarioBank(config=config, scenarios=tuple(scenarios))


def validate_paired_scenario_bank(scenarios: Sequence[ScenarioSpec]) -> None:
    """Raise when a bank is not composed of matched three-stratum episodes."""

    groups: dict[str, list[ScenarioSpec]] = {}
    for scenario in scenarios:
        groups.setdefault(scenario.pair_id, []).append(scenario)
    required = set(ScenarioStratum)
    for pair_id, group in groups.items():
        observed = {item.stratum for item in group}
        if observed != required or len(group) != len(required):
            raise ValueError(f"Paired case {pair_id!r} does not contain exactly three strata")
        anchor = group[0]
        for item in group[1:]:
            if (
                item.family_id != anchor.family_id
                or item.seed != anchor.seed
                or item.ego != anchor.ego
                or item.trailer != anchor.trailer
                or item.counterfactual_latent_trajectory != anchor.counterfactual_latent_trajectory
                or item.conflict != anchor.conflict
                or item.visibility != anchor.visibility
                or item.duration_s != anchor.duration_s
                or item.nominal_merge != anchor.nominal_merge
                or item.support_actors != anchor.support_actors
                or item.static_occluders != anchor.static_occluders
            ):
                raise ValueError(f"Paired case {pair_id!r} has mismatched kinematics or geometry")


def _smoke() -> None:
    """Run a lightweight internal integrity check without a simulator."""

    bank = generate_paired_scenario_bank(
        ScenarioBankConfig(replicates_per_cell=1)
    )
    summary = bank.summary()
    assert summary["paired_cases"] == 15
    assert summary["total_scenarios"] == 45
    true_threat = bank.by_stratum(ScenarioStratum.TRUE_OCCLUDED_THREAT)[0]
    empty_shadow = bank.by_stratum(ScenarioStratum.EMPTY_SHADOW)[0]
    visible_control = bank.by_stratum(ScenarioStratum.VISIBLE_CONTROL)[0]
    assert not true_threat.visibility_at(0.0).visible
    assert not empty_shadow.latent_present
    assert visible_control.visibility_at(0.0).visible
    assert true_threat.nominal_merge.target_lane == 2
    print("occlusion benchmark scenario smoke check passed", dict(summary))


if __name__ == "__main__":
    _smoke()
