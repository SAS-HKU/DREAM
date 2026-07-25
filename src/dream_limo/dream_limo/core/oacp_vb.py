"""ROS-independent risk primitives for the OACP-VB comparison arm.

``OACP-VB`` means *velocity-bound adaptation of Zheng et al.*  It retains the
paper's simplified reachability quantification (SRQ) and dynamic velocity
boundary, but it is not the published Bézier/consensus-ADMM planner.

Two interpretation choices are deliberately explicit here:

* Equation (12) typesets a normal distribution with a variance that appears to
  depend on the evaluation point ``d``.  A literal reading is asymmetric and is
  not maximal at the lane centre.  The authors' public review snapshot instead
  uses the constant standard deviation ``lane_width / (2 * confidence_z)``.
  We follow that intent and normalize the Gaussian so ``r_lat(0) == 1``.
* Reducing risk to the maximum over an MPC horizon and percentile-based
  threshold calibration are integration choices for this shared-LMPC baseline,
  not claims about the published planner.
* The merge connector and finite conflict-distance test are scenario mappings
  used to apply Remark 2 to the LIMO occluded merge; they are not paper planner
  geometry.

The module intentionally contains no ROS dependencies or file I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import exp, floor, isfinite
from typing import Iterable, Sequence

import numpy as np


class PVSLengthPolicy(str, Enum):
    """Policy for a PVS longer than one prediction-horizon reach."""

    CLIP = "clip"
    REJECT = "reject"


class ContingencyBranch(str, Enum):
    """Names retained from the paper without resolving their semantic ambiguity."""

    EXPLORATION = "exploration"
    FALLBACK = "fallback"


class VelocityRegion(str, Enum):
    """Region of the three-part dynamic velocity-bound mapping."""

    MAXIMUM = "maximum"
    INTERPOLATED = "interpolated"
    MINIMUM = "minimum"


@dataclass(frozen=True)
class OACPVBConfig:
    """Immutable configuration for SRQ and both velocity bounds.

    Speeds and thresholds are required rather than silently importing the
    paper's values: this comparison must use the deployed DREAM arm's shared
    nominal speed and empirically calibrated risk scale.
    """

    v_pv_max: float
    prediction_horizon: float
    lane_width: float
    confidence_z: float
    c_th_min: float
    c_th_max_exploration: float
    c_th_max_fallback: float
    v_occ_min: float
    v_occ_max: float
    pvs_length_policy: PVSLengthPolicy = PVSLengthPolicy.CLIP

    def __post_init__(self) -> None:
        finite_fields = (
            "v_pv_max",
            "prediction_horizon",
            "lane_width",
            "confidence_z",
            "c_th_min",
            "c_th_max_exploration",
            "c_th_max_fallback",
            "v_occ_min",
            "v_occ_max",
        )
        for name in finite_fields:
            _require_finite(name, getattr(self, name))
        if self.v_pv_max <= 0.0:
            raise ValueError("v_pv_max must be positive")
        if self.prediction_horizon <= 0.0:
            raise ValueError("prediction_horizon must be positive")
        if self.lane_width <= 0.0:
            raise ValueError("lane_width must be positive")
        if self.confidence_z <= 0.0:
            raise ValueError("confidence_z must be positive")
        if self.c_th_min < 0.0:
            raise ValueError("c_th_min must be nonnegative")
        if self.c_th_max_exploration <= self.c_th_min:
            raise ValueError("c_th_max_exploration must exceed c_th_min")
        if self.c_th_max_fallback <= self.c_th_min:
            raise ValueError("c_th_max_fallback must exceed c_th_min")
        if self.v_occ_min < 0.0:
            raise ValueError("v_occ_min must be nonnegative")
        if self.v_occ_max < self.v_occ_min:
            raise ValueError("v_occ_max must be at least v_occ_min")
        try:
            policy = PVSLengthPolicy(self.pvs_length_policy)
        except ValueError as exc:
            raise ValueError(
                f"unsupported pvs_length_policy: {self.pvs_length_policy!r}"
            ) from exc
        object.__setattr__(self, "pvs_length_policy", policy)

    @property
    def maximum_pvs_length(self) -> float:
        """Maximum valid PVS length in Eq. (10), ``v_pv_max * T``."""

        return self.v_pv_max * self.prediction_horizon

    @property
    def lateral_sigma(self) -> float:
        """Constant lateral standard deviation used by the adaptation."""

        return self.lane_width / (2.0 * self.confidence_z)

    def maximum_risk_threshold(self, branch: ContingencyBranch | str) -> float:
        branch_value = ContingencyBranch(branch)
        if branch_value is ContingencyBranch.EXPLORATION:
            return self.c_th_max_exploration
        return self.c_th_max_fallback


@dataclass(frozen=True)
class PVSInterval:
    """Validated PVS interval and any explicit Eq. (10) length clipping."""

    start: float
    requested_end: float
    end: float
    maximum_length: float
    was_clipped: bool

    @property
    def length(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class HorizonRiskEvaluation:
    """Result of reducing point risks over a planned horizon."""

    risk_total: float
    raw_maximum: float
    maximum_index: int
    sample_count: int
    frs_intersects_trajectory: bool
    ignored_by_remark_2: bool


@dataclass(frozen=True)
class VelocityBoundEvaluation:
    """One branch's dynamic velocity-bound result."""

    branch: ContingencyBranch
    risk_total: float
    maximum_risk_threshold: float
    velocity_bound: float
    region: VelocityRegion


@dataclass(frozen=True)
class ThresholdCalibration:
    """Deterministic percentile calibration summary for an occluded phase."""

    percentile: float
    sample_count: int
    observed_minimum: float
    observed_maximum: float
    exploration_threshold: float
    fallback_threshold: float
    fallback_ratio: float


@dataclass(frozen=True)
class MergeConnector:
    """Sampled phantom-lane connector in the planned route's local frame.

    ``reference_points`` are samples of the ego route.  ``points`` start one
    lane to the right of those samples and smoothly merge onto them.  Both
    arrays, and their corresponding arc coordinates, are immutable copies.
    """

    points: np.ndarray
    cumulative_s: np.ndarray
    reference_points: np.ndarray
    reference_s: np.ndarray
    ego_route_s: float
    requested_range: float
    effective_range: float
    merge_length: float
    route_end_clipped: bool

    def __post_init__(self) -> None:
        points = _immutable_polyline("points", self.points)
        reference_points = _immutable_polyline(
            "reference_points",
            self.reference_points,
        )
        if points.shape != reference_points.shape:
            raise ValueError("points and reference_points must have the same shape")
        cumulative_s = _immutable_coordinate(
            "cumulative_s",
            self.cumulative_s,
            points.shape[0],
            require_zero_start=True,
        )
        reference_s = _immutable_coordinate(
            "reference_s",
            self.reference_s,
            points.shape[0],
            require_zero_start=True,
        )
        ego_route_s = _require_finite("ego_route_s", self.ego_route_s)
        requested_range = _require_positive(
            "requested_range",
            self.requested_range,
        )
        effective_range = _require_positive(
            "effective_range",
            self.effective_range,
        )
        merge_length = _require_positive("merge_length", self.merge_length)
        if effective_range > requested_range + 1.0e-9:
            raise ValueError("effective_range cannot exceed requested_range")
        if not isinstance(self.route_end_clipped, bool):
            raise TypeError("route_end_clipped must be bool")
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "cumulative_s", cumulative_s)
        object.__setattr__(self, "reference_points", reference_points)
        object.__setattr__(self, "reference_s", reference_s)
        object.__setattr__(self, "ego_route_s", ego_route_s)
        object.__setattr__(self, "requested_range", requested_range)
        object.__setattr__(self, "effective_range", effective_range)
        object.__setattr__(self, "merge_length", merge_length)


@dataclass(frozen=True)
class PVSComponent:
    """One contiguous occluded segment retained as an independent PVS."""

    interval: PVSInterval
    first_sample_index: int
    last_sample_index: int
    range_clipped: bool

    def __post_init__(self) -> None:
        if self.interval.length <= 0.0:
            raise ValueError("PVS components must have nonzero length")
        if self.first_sample_index < 0:
            raise ValueError("first_sample_index must be nonnegative")
        if self.last_sample_index < self.first_sample_index:
            raise ValueError(
                "last_sample_index must not precede first_sample_index"
            )
        if not isinstance(self.range_clipped, bool):
            raise TypeError("range_clipped must be bool")

    @property
    def was_clipped(self) -> bool:
        """Whether range capping or the Eq. (10) length cap changed the PVS."""

        return self.range_clipped or self.interval.was_clipped


@dataclass(frozen=True)
class PVSExtraction:
    """Result of sampling a world-axis shadow mask along a connector."""

    components: tuple[PVSComponent, ...]
    route_sample_count: int
    shadow_sample_count: int
    in_range_sample_count: int
    range_was_clipped: bool

    def __post_init__(self) -> None:
        components = tuple(self.components)
        if not all(isinstance(component, PVSComponent) for component in components):
            raise TypeError("components must contain only PVSComponent values")
        if self.route_sample_count < 2:
            raise ValueError("route_sample_count must be at least two")
        if not 0 <= self.shadow_sample_count <= self.route_sample_count:
            raise ValueError("shadow_sample_count is inconsistent")
        if not 0 <= self.in_range_sample_count <= self.route_sample_count:
            raise ValueError("in_range_sample_count is inconsistent")
        if not isinstance(self.range_was_clipped, bool):
            raise TypeError("range_was_clipped must be bool")
        object.__setattr__(self, "components", components)


@dataclass(frozen=True)
class GeometryRiskEvaluation:
    """Multi-component horizon risk with an explicit finite Remark-2 gate."""

    risk_total: float
    raw_maximum: float
    active_component_index: int | None
    active_horizon_index: int | None
    raw_component_index: int | None
    raw_horizon_index: int | None
    selected_route_s: float | None
    selected_lateral_offset: float | None
    selected_conflict_distance: float | None
    component_minimum_distances: tuple[float, ...]
    component_intersections: tuple[bool, ...]
    horizon_sample_count: int
    ignored_by_remark_2: bool

    def __post_init__(self) -> None:
        _require_nonnegative_risk(self.risk_total)
        _require_nonnegative_risk(self.raw_maximum)
        if self.risk_total > self.raw_maximum + 1.0e-12:
            raise ValueError("risk_total cannot exceed raw_maximum")
        if self.horizon_sample_count < 1:
            raise ValueError("horizon_sample_count must be positive")
        distances = tuple(
            _require_nonnegative_finite_or_infinite(
                "component minimum distance",
                value,
            )
            for value in self.component_minimum_distances
        )
        intersections = tuple(self.component_intersections)
        if len(distances) != len(intersections):
            raise ValueError(
                "component distances and intersections must have equal length"
            )
        if not all(isinstance(value, bool) for value in intersections):
            raise TypeError("component_intersections must contain bool values")
        for name, index in (
            ("active_component_index", self.active_component_index),
            ("active_horizon_index", self.active_horizon_index),
            ("raw_component_index", self.raw_component_index),
            ("raw_horizon_index", self.raw_horizon_index),
        ):
            if index is not None and (not isinstance(index, int) or index < 0):
                raise ValueError(f"{name} must be a nonnegative integer or None")
        for name, value in (
            ("selected_route_s", self.selected_route_s),
            ("selected_lateral_offset", self.selected_lateral_offset),
            ("selected_conflict_distance", self.selected_conflict_distance),
        ):
            if value is not None:
                _require_finite(name, value)
        if not isinstance(self.ignored_by_remark_2, bool):
            raise TypeError("ignored_by_remark_2 must be bool")
        object.__setattr__(self, "component_minimum_distances", distances)
        object.__setattr__(self, "component_intersections", intersections)


def build_phantom_merge_connector(
    route: Sequence[Sequence[float]] | np.ndarray,
    ego_xy: Sequence[float] | np.ndarray,
    *,
    lane_width: float,
    perception_range: float,
    sampling_spacing: float,
    merge_length: float,
) -> MergeConnector:
    """Build a right-lane phantom route that smoothly joins the ego route.

    The input route is first parameterized by arc length.  ``s=0`` is the
    closest projection of ``ego_xy`` on that route, not the first input vertex.
    The right normal is ``(t_y, -t_x)``, so a route pointing along positive x
    starts at negative y.  A cubic smoothstep tapers the full lane offset to
    zero over ``merge_length``.
    """

    route_points = _validated_polyline("route", route)
    ego = _validated_xy("ego_xy", ego_xy)
    width = _require_positive("lane_width", lane_width)
    requested_range = _require_positive("perception_range", perception_range)
    spacing = _require_positive("sampling_spacing", sampling_spacing)
    taper_length = _require_positive("merge_length", merge_length)

    route_s = _polyline_cumulative_s(route_points)
    ego_route_s = float(
        _project_queries_to_polyline(ego[None, :], route_points, route_s)[0][0]
    )
    available_range = float(route_s[-1] - ego_route_s)
    if available_range <= 1.0e-9:
        raise ValueError("ego projection leaves no forward route to sample")
    effective_range = min(requested_range, available_range)

    reference_s = np.arange(
        0.0,
        effective_range + 0.5 * spacing,
        spacing,
        dtype=float,
    )
    reference_s = reference_s[reference_s < effective_range - 1.0e-12]
    reference_s = np.concatenate((reference_s, np.array([effective_range])))
    if reference_s[0] != 0.0:
        reference_s = np.concatenate((np.array([0.0]), reference_s))
    absolute_s = ego_route_s + reference_s
    reference_points, tangents = _interpolate_polyline(
        route_points,
        route_s,
        absolute_s,
    )

    right_normals = np.column_stack((tangents[:, 1], -tangents[:, 0]))
    fraction = np.clip(reference_s / taper_length, 0.0, 1.0)
    taper = 1.0 - 3.0 * fraction**2 + 2.0 * fraction**3
    connector_points = reference_points + width * taper[:, None] * right_normals
    connector_s = _polyline_cumulative_s(connector_points)

    return MergeConnector(
        points=connector_points,
        cumulative_s=connector_s,
        reference_points=reference_points,
        reference_s=reference_s,
        ego_route_s=ego_route_s,
        requested_range=requested_range,
        effective_range=effective_range,
        merge_length=taper_length,
        route_end_clipped=available_range < requested_range - 1.0e-9,
    )


def extract_pvs_components(
    shadow_mask: Sequence[Sequence[float]] | np.ndarray,
    connector: MergeConnector,
    ego_xy: Sequence[float] | np.ndarray,
    *,
    grid_origin_xy: Sequence[float] | np.ndarray,
    grid_resolution: float,
    perception_range: float,
    config: OACPVBConfig,
) -> PVSExtraction:
    """Extract disconnected PVS intervals from a world-axis shadow mask.

    The mask follows ``OccupancyGrid`` array convention: row is world y, column
    is world x, and ``grid_origin_xy`` is the lower-left cell corner.  Positive
    values mean occluded; zero, negative (including an OccupancyGrid unknown
    sentinel), and out-of-grid samples are not treated as occlusion.  Visible
    samples split components rather than being bridged.
    """

    if not isinstance(connector, MergeConnector):
        raise TypeError("connector must be a MergeConnector")
    if not isinstance(config, OACPVBConfig):
        raise TypeError("config must be an OACPVBConfig")
    mask = _validated_shadow_mask(shadow_mask)
    ego = _validated_xy("ego_xy", ego_xy)
    origin = _validated_xy("grid_origin_xy", grid_origin_xy)
    resolution = _require_positive("grid_resolution", grid_resolution)
    range_limit = _require_positive("perception_range", perception_range)

    columns = np.floor((connector.points[:, 0] - origin[0]) / resolution).astype(
        np.int64
    )
    rows = np.floor((connector.points[:, 1] - origin[1]) / resolution).astype(
        np.int64
    )
    in_bounds = (
        (rows >= 0)
        & (rows < mask.shape[0])
        & (columns >= 0)
        & (columns < mask.shape[1])
    )
    shadowed = np.zeros(connector.points.shape[0], dtype=bool)
    shadowed[in_bounds] = mask[rows[in_bounds], columns[in_bounds]] > 0.0

    ego_distances = np.linalg.norm(connector.points - ego[None, :], axis=1)
    in_range = ego_distances <= range_limit + 1.0e-12
    retained = shadowed & in_range
    components: list[PVSComponent] = []

    for first_index, last_index in _contiguous_true_runs(retained):
        start, start_range_clipped = _component_start_boundary(
            first_index,
            shadowed,
            in_range,
            connector,
            ego,
            range_limit,
        )
        end, end_range_clipped = _component_end_boundary(
            last_index,
            shadowed,
            in_range,
            connector,
            ego,
            range_limit,
        )
        if end - start <= 1.0e-9:
            continue
        interval = make_pvs_interval(start, end, config)
        if interval.length <= 1.0e-9:
            continue
        components.append(
            PVSComponent(
                interval=interval,
                first_sample_index=first_index,
                last_sample_index=last_index,
                range_clipped=start_range_clipped or end_range_clipped,
            )
        )

    return PVSExtraction(
        components=tuple(components),
        route_sample_count=connector.points.shape[0],
        shadow_sample_count=int(np.count_nonzero(shadowed)),
        in_range_sample_count=int(np.count_nonzero(in_range)),
        range_was_clipped=bool(np.any(shadowed & ~in_range)),
    )


def evaluate_geometry_risk(
    planned_horizon: Sequence[Sequence[float]] | np.ndarray,
    connector: MergeConnector,
    components: Sequence[PVSComponent] | PVSExtraction,
    config: OACPVBConfig,
    *,
    conflict_distance: float,
) -> GeometryRiskEvaluation:
    """Evaluate all horizon samples against all disconnected PVS components.

    Remark 2 is implemented as a finite geometric conflict test.  A component
    is active only when at least one horizon point projects into that phantom
    component's forward-reachable longitudinal support *and* lies within
    ``conflict_distance`` of the phantom route.  Once active, its maximum risk
    over the complete horizon participates in the deterministic global maximum.
    """

    horizon = _validated_point_array("planned_horizon", planned_horizon)
    if not isinstance(connector, MergeConnector):
        raise TypeError("connector must be a MergeConnector")
    if not isinstance(config, OACPVBConfig):
        raise TypeError("config must be an OACPVBConfig")
    distance_limit = _require_positive("conflict_distance", conflict_distance)
    if isinstance(components, PVSExtraction):
        component_values = components.components
    else:
        component_values = tuple(components)
    if not all(
        isinstance(component, PVSComponent) for component in component_values
    ):
        raise TypeError("components must contain only PVSComponent values")

    if not component_values:
        return GeometryRiskEvaluation(
            risk_total=0.0,
            raw_maximum=0.0,
            active_component_index=None,
            active_horizon_index=None,
            raw_component_index=None,
            raw_horizon_index=None,
            selected_route_s=None,
            selected_lateral_offset=None,
            selected_conflict_distance=None,
            component_minimum_distances=(),
            component_intersections=(),
            horizon_sample_count=horizon.shape[0],
            ignored_by_remark_2=False,
        )

    route_s, lateral_offsets, route_distances = _project_queries_to_polyline(
        horizon,
        connector.points,
        connector.cumulative_s,
    )
    component_intersections: list[bool] = []
    component_minimum_distances: list[float] = []
    component_risks: list[np.ndarray] = []
    raw_candidates: list[tuple[float, int, int]] = []
    active_candidates: list[tuple[float, int, int]] = []

    for component_index, component in enumerate(component_values):
        interval = component.interval
        risks = np.asarray(
            [
                point_risk(position, offset, interval, config)
                for position, offset in zip(route_s, lateral_offsets)
            ],
            dtype=float,
        )
        component_risks.append(risks)
        maximum_index = int(np.argmax(risks))
        candidate = (float(risks[maximum_index]), component_index, maximum_index)
        raw_candidates.append(candidate)

        support = (
            (route_s >= interval.start - 1.0e-12)
            & (
                route_s
                <= interval.end + config.maximum_pvs_length + 1.0e-12
            )
        )
        if np.any(support):
            minimum_distance = float(np.min(route_distances[support]))
        else:
            minimum_distance = float("inf")
        intersects = bool(
            np.any(support & (route_distances <= distance_limit + 1.0e-12))
        )
        component_minimum_distances.append(minimum_distance)
        component_intersections.append(intersects)
        if intersects:
            active_candidates.append(candidate)

    raw_value, raw_component_index, raw_horizon_index = max(
        raw_candidates,
        key=lambda value: value[0],
    )
    if active_candidates:
        active_value, active_component_index, active_horizon_index = max(
            active_candidates,
            key=lambda value: value[0],
        )
        selected_index = active_horizon_index
        ignored = False
    else:
        active_value = 0.0
        active_component_index = None
        active_horizon_index = None
        selected_index = raw_horizon_index
        ignored = True

    return GeometryRiskEvaluation(
        risk_total=active_value,
        raw_maximum=raw_value,
        active_component_index=active_component_index,
        active_horizon_index=active_horizon_index,
        raw_component_index=raw_component_index,
        raw_horizon_index=raw_horizon_index,
        selected_route_s=float(route_s[selected_index]),
        selected_lateral_offset=float(lateral_offsets[selected_index]),
        selected_conflict_distance=float(route_distances[selected_index]),
        component_minimum_distances=tuple(component_minimum_distances),
        component_intersections=tuple(component_intersections),
        horizon_sample_count=horizon.shape[0],
        ignored_by_remark_2=ignored,
    )


def make_pvs_interval(
    s_s: float,
    s_e: float,
    config: OACPVBConfig,
) -> PVSInterval:
    """Validate a PVS and enforce ``s_e - s_s <= v_pv_max * T``.

    The interval precondition is stated by the original SRQ derivation but is
    implicit in the OACP paper's piecewise equation.  ``CLIP`` retains the near
    boundary and clips the far boundary; ``REJECT`` refuses the interval.
    """

    start = _require_finite("s_s", s_s)
    requested_end = _require_finite("s_e", s_e)
    if requested_end < start:
        raise ValueError("s_e must be greater than or equal to s_s")
    maximum_length = config.maximum_pvs_length
    requested_length = requested_end - start
    if requested_length <= maximum_length:
        return PVSInterval(
            start=start,
            requested_end=requested_end,
            end=requested_end,
            maximum_length=maximum_length,
            was_clipped=False,
        )
    if config.pvs_length_policy is PVSLengthPolicy.REJECT:
        raise ValueError(
            "PVS length exceeds v_pv_max * prediction_horizon "
            f"({requested_length} > {maximum_length})"
        )
    return PVSInterval(
        start=start,
        requested_end=requested_end,
        end=start + maximum_length,
        maximum_length=maximum_length,
        was_clipped=True,
    )


def potential_pv_count(
    s: float,
    pvs: PVSInterval,
    config: OACPVBConfig,
) -> float:
    """Evaluate the paper's piecewise potential-PV quantity ``g(s)`` (Eq. 10)."""

    position = _require_finite("s", s)
    length = pvs.length
    if length <= 0.0:
        return 0.0
    reach = config.maximum_pvs_length
    if length > reach:
        raise ValueError("PVSInterval violates Eq. (10) maximum-length precondition")

    s_s = pvs.start
    s_e = pvs.end
    if position < s_s or position > s_e + reach:
        return 0.0

    if position <= s_e:  # I1 = [s_s, s_e]
        value = 0.5 * (
            2.0 * config.v_pv_max
            - (position - s_s) / config.prediction_horizon
        ) * (position - s_s)
    elif position <= s_s + reach:  # I2 = [s_e, s_s + v_pv_max*T]
        value = 0.5 * (
            2.0 * config.v_pv_max
            - (position - s_s) / config.prediction_horizon
            - (position - s_e) / config.prediction_horizon
        ) * length
    else:  # I3 = [s_s + v_pv_max*T, s_e + v_pv_max*T]
        value = 0.5 * (
            config.v_pv_max
            - (position - s_e) / config.prediction_horizon
        ) * (s_e - (position - reach))

    # The valid piecewise geometry is nonnegative; remove only round-off noise.
    return max(0.0, float(value))


def longitudinal_risk(
    s: float,
    pvs: PVSInterval,
    config: OACPVBConfig,
) -> float:
    """Evaluate longitudinal risk ``r_lon(s) = (s_e - s_s) * g(s)`` (Eq. 11)."""

    return pvs.length * potential_pv_count(s, pvs, config)


def lateral_risk(d: float, config: OACPVBConfig) -> float:
    """Evaluate the normalized constant-sigma lateral Gaussian.

    This follows the centre-peaked interpretation in the authors' review
    snapshot and guarantees symmetry and monotonic decrease in ``abs(d)``.
    """

    offset = _require_finite("d", d)
    sigma = config.lateral_sigma
    return exp(-0.5 * (offset / sigma) ** 2)


def point_risk(
    s: float,
    d: float,
    pvs: PVSInterval,
    config: OACPVBConfig,
) -> float:
    """Evaluate adapted point risk ``r_lon(s) * r_lat(d)`` (Eq. 13)."""

    return longitudinal_risk(s, pvs, config) * lateral_risk(d, config)


def reduce_horizon_risk(
    point_risks: Iterable[float],
    *,
    frs_intersects_trajectory: bool,
) -> HorizonRiskEvaluation:
    """Reduce point risks by maximum, applying the paper's Remark 2 gate.

    ``frs_intersects_trajectory`` is mandatory and explicit so an unrelated
    occluded lane cannot silently slow the ego.  The raw maximum is retained
    for diagnostics even when Remark 2 sets the consumed risk to zero.
    """

    if not isinstance(frs_intersects_trajectory, bool):
        raise TypeError("frs_intersects_trajectory must be bool")
    values = tuple(_require_nonnegative_risk(value) for value in point_risks)
    if not values:
        raise ValueError("point_risks must contain at least one horizon sample")
    maximum_index = max(range(len(values)), key=values.__getitem__)
    raw_maximum = values[maximum_index]
    ignored = not frs_intersects_trajectory
    return HorizonRiskEvaluation(
        risk_total=0.0 if ignored else raw_maximum,
        raw_maximum=raw_maximum,
        maximum_index=maximum_index,
        sample_count=len(values),
        frs_intersects_trajectory=frs_intersects_trajectory,
        ignored_by_remark_2=ignored,
    )


def dynamic_velocity_bound(
    risk_total: float,
    config: OACPVBConfig,
    branch: ContingencyBranch | str,
) -> VelocityBoundEvaluation:
    """Map aggregate risk to one branch's clamped velocity bound (Eqs. 14-15)."""

    risk = _require_nonnegative_risk(risk_total)
    branch_value = ContingencyBranch(branch)
    c_th_max = config.maximum_risk_threshold(branch_value)

    if risk <= config.c_th_min:
        velocity = config.v_occ_max
        region = VelocityRegion.MAXIMUM
    elif risk >= c_th_max:
        velocity = config.v_occ_min
        region = VelocityRegion.MINIMUM
    else:
        slope = (
            (config.v_occ_min - config.v_occ_max)
            / (c_th_max - config.c_th_min)
        )
        velocity = slope * (risk - config.c_th_min) + config.v_occ_max
        velocity = min(config.v_occ_max, max(config.v_occ_min, velocity))
        region = VelocityRegion.INTERPOLATED

    return VelocityBoundEvaluation(
        branch=branch_value,
        risk_total=risk,
        maximum_risk_threshold=c_th_max,
        velocity_bound=float(velocity),
        region=region,
    )


def calibrate_thresholds(
    occluded_phase_risks: Iterable[float],
    *,
    percentile: float = 0.70,
    fallback_ratio: float = 4.0 / 3.0,
) -> ThresholdCalibration:
    """Calibrate exploration at a linear percentile and fallback by a ratio.

    This helper intentionally consumes only caller-selected occluded-phase
    samples.  It does not infer phase boundaries or read logs.
    """

    quantile = _require_finite("percentile", percentile)
    ratio = _require_finite("fallback_ratio", fallback_ratio)
    if not 0.0 < quantile <= 1.0:
        raise ValueError("percentile must be in (0, 1]")
    if ratio <= 1.0:
        raise ValueError("fallback_ratio must be greater than 1")

    values = sorted(
        _require_nonnegative_risk(value) for value in occluded_phase_risks
    )
    if not values:
        raise ValueError("occluded_phase_risks must not be empty")
    rank = (len(values) - 1) * quantile
    lower = floor(rank)
    upper = min(lower + 1, len(values) - 1)
    fraction = rank - lower
    exploration = values[lower] + fraction * (values[upper] - values[lower])
    if exploration <= 0.0:
        raise ValueError(
            "calibration produced a nonpositive threshold; "
            "the occluded phase contains insufficient positive risk"
        )
    return ThresholdCalibration(
        percentile=quantile,
        sample_count=len(values),
        observed_minimum=values[0],
        observed_maximum=values[-1],
        exploration_threshold=float(exploration),
        fallback_threshold=float(exploration * ratio),
        fallback_ratio=ratio,
    )


def _validated_point_array(
    name: str,
    values: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    try:
        points = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric Nx2 array") from exc
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 1:
        raise ValueError(f"{name} must be a nonempty Nx2 array")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{name} must contain only finite coordinates")
    return np.array(points, dtype=float, copy=True)


def _validated_polyline(
    name: str,
    values: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    points = _validated_point_array(name, values)
    if points.shape[0] < 2:
        raise ValueError(f"{name} must contain at least two points")
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    if np.any(lengths <= 1.0e-12):
        raise ValueError(f"{name} must not contain duplicate consecutive points")
    return points


def _validated_xy(
    name: str,
    value: Sequence[float] | np.ndarray,
) -> np.ndarray:
    try:
        point = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite length-two coordinate") from exc
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        raise ValueError(f"{name} must be a finite length-two coordinate")
    return np.array(point, dtype=float, copy=True)


def _immutable_polyline(
    name: str,
    values: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    points = _validated_polyline(name, values)
    points.setflags(write=False)
    return points


def _immutable_coordinate(
    name: str,
    values: Sequence[float] | np.ndarray,
    expected_count: int,
    *,
    require_zero_start: bool,
) -> np.ndarray:
    try:
        coordinate = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric one-dimensional array") from exc
    if coordinate.shape != (expected_count,):
        raise ValueError(f"{name} must have one value per route point")
    if not np.all(np.isfinite(coordinate)):
        raise ValueError(f"{name} must contain only finite values")
    if require_zero_start and abs(float(coordinate[0])) > 1.0e-9:
        raise ValueError(f"{name} must start at zero")
    if np.any(np.diff(coordinate) <= 1.0e-12):
        raise ValueError(f"{name} must be strictly increasing")
    result = np.array(coordinate, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _validated_shadow_mask(
    values: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError("shadow_mask must be a nonempty two-dimensional array")
    if not (
        np.issubdtype(array.dtype, np.number)
        or np.issubdtype(array.dtype, np.bool_)
    ):
        raise ValueError("shadow_mask must contain numeric or boolean values")
    result = np.asarray(array, dtype=float)
    if not np.all(np.isfinite(result)):
        raise ValueError("shadow_mask must contain only finite values")
    return result


def _polyline_cumulative_s(points: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    if np.any(lengths <= 1.0e-12):
        raise ValueError("polyline must not contain duplicate consecutive points")
    return np.concatenate((np.array([0.0]), np.cumsum(lengths)))


def _project_queries_to_polyline(
    queries: np.ndarray,
    points: np.ndarray,
    cumulative_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    segments = points[1:] - points[:-1]
    length_squared = np.sum(segments * segments, axis=1)
    lengths = np.sqrt(length_squared)
    route_s = np.empty(queries.shape[0], dtype=float)
    lateral_offsets = np.empty(queries.shape[0], dtype=float)
    distances = np.empty(queries.shape[0], dtype=float)

    for query_index, query in enumerate(queries):
        relative = query[None, :] - points[:-1]
        fractions = np.clip(
            np.sum(relative * segments, axis=1) / length_squared,
            0.0,
            1.0,
        )
        projections = points[:-1] + fractions[:, None] * segments
        displacement = query[None, :] - projections
        distance_squared = np.sum(displacement * displacement, axis=1)
        segment_index = int(np.argmin(distance_squared))
        tangent = segments[segment_index] / lengths[segment_index]
        selected_displacement = displacement[segment_index]
        route_s[query_index] = (
            cumulative_s[segment_index]
            + fractions[segment_index] * lengths[segment_index]
        )
        lateral_offsets[query_index] = (
            tangent[0] * selected_displacement[1]
            - tangent[1] * selected_displacement[0]
        )
        distances[query_index] = np.sqrt(distance_squared[segment_index])

    return route_s, lateral_offsets, distances


def _interpolate_polyline(
    points: np.ndarray,
    cumulative_s: np.ndarray,
    query_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.searchsorted(cumulative_s, query_s, side="right") - 1
    indices = np.clip(indices, 0, points.shape[0] - 2)
    segments = points[indices + 1] - points[indices]
    lengths = np.linalg.norm(segments, axis=1)
    fractions = (query_s - cumulative_s[indices]) / lengths
    fractions = np.clip(fractions, 0.0, 1.0)
    interpolated = points[indices] + fractions[:, None] * segments
    tangents = segments / lengths[:, None]
    return interpolated, tangents


def _contiguous_true_runs(values: np.ndarray) -> tuple[tuple[int, int], ...]:
    indices = np.flatnonzero(values)
    if indices.size == 0:
        return ()
    breaks = np.flatnonzero(np.diff(indices) > 1)
    starts = np.concatenate((np.array([0]), breaks + 1))
    ends = np.concatenate((breaks, np.array([indices.size - 1])))
    return tuple(
        (int(indices[start]), int(indices[end]))
        for start, end in zip(starts, ends)
    )


def _component_start_boundary(
    index: int,
    shadowed: np.ndarray,
    in_range: np.ndarray,
    connector: MergeConnector,
    ego: np.ndarray,
    perception_range: float,
) -> tuple[float, bool]:
    if index == 0:
        return float(connector.cumulative_s[0]), False
    previous = index - 1
    if shadowed[previous] and not in_range[previous]:
        return (
            _circle_boundary_s(
                connector.points[previous],
                connector.points[index],
                float(connector.cumulative_s[previous]),
                float(connector.cumulative_s[index]),
                ego,
                perception_range,
            ),
            True,
        )
    return (
        0.5
        * float(connector.cumulative_s[previous] + connector.cumulative_s[index]),
        False,
    )


def _component_end_boundary(
    index: int,
    shadowed: np.ndarray,
    in_range: np.ndarray,
    connector: MergeConnector,
    ego: np.ndarray,
    perception_range: float,
) -> tuple[float, bool]:
    if index == connector.points.shape[0] - 1:
        return float(connector.cumulative_s[-1]), False
    following = index + 1
    if shadowed[following] and not in_range[following]:
        return (
            _circle_boundary_s(
                connector.points[index],
                connector.points[following],
                float(connector.cumulative_s[index]),
                float(connector.cumulative_s[following]),
                ego,
                perception_range,
            ),
            True,
        )
    return (
        0.5
        * float(connector.cumulative_s[index] + connector.cumulative_s[following]),
        False,
    )


def _circle_boundary_s(
    first: np.ndarray,
    second: np.ndarray,
    first_s: float,
    second_s: float,
    centre: np.ndarray,
    radius: float,
) -> float:
    segment = second - first
    relative = first - centre
    a = float(np.dot(segment, segment))
    b = 2.0 * float(np.dot(relative, segment))
    c = float(np.dot(relative, relative) - radius**2)
    discriminant = max(0.0, b**2 - 4.0 * a * c)
    root_scale = np.sqrt(discriminant)
    roots = (
        (-b - root_scale) / (2.0 * a),
        (-b + root_scale) / (2.0 * a),
    )
    valid_roots = [
        min(1.0, max(0.0, root))
        for root in roots
        if -1e-9 <= root <= 1.0 + 1e-9
    ]
    if not valid_roots:
        fraction = 0.5
    else:
        first_inside = np.linalg.norm(first - centre) <= radius
        fraction = max(valid_roots) if first_inside else min(valid_roots)
    return first_s + fraction * (second_s - first_s)


def _require_finite(name: str, value: float) -> float:
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _require_positive(name: str, value: float) -> float:
    result = _require_finite(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _require_nonnegative_finite_or_infinite(name: str, value: float) -> float:
    result = float(value)
    if result != result or result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _require_nonnegative_risk(value: float) -> float:
    result = _require_finite("risk", value)
    if result < 0.0:
        raise ValueError("risk values must be nonnegative")
    return result


__all__ = [
    "ContingencyBranch",
    "GeometryRiskEvaluation",
    "HorizonRiskEvaluation",
    "MergeConnector",
    "OACPVBConfig",
    "PVSComponent",
    "PVSExtraction",
    "PVSInterval",
    "PVSLengthPolicy",
    "ThresholdCalibration",
    "VelocityBoundEvaluation",
    "VelocityRegion",
    "build_phantom_merge_connector",
    "calibrate_thresholds",
    "dynamic_velocity_bound",
    "evaluate_geometry_risk",
    "extract_pvs_components",
    "lateral_risk",
    "longitudinal_risk",
    "make_pvs_interval",
    "point_risk",
    "potential_pv_count",
    "reduce_horizon_risk",
]
