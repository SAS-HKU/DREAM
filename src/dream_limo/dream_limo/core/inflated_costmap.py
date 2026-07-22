"""Swept-footprint checks against DREAM's live, inflated Nav2 costmap."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, ceil, cos, hypot, isfinite, sin
from typing import Sequence

import numpy as np

from .free_goal import CostmapSnapshot


@dataclass(frozen=True)
class InflatedCostmapCheck:
    """Result of checking a predicted trajectory against the live costmap."""

    safe: bool
    reason: str
    sample_index: int | None = None
    cell_x: int | None = None
    cell_y: int | None = None
    cell_value: int | None = None


def _cell(
    costmap: CostmapSnapshot, x: float, y: float
) -> tuple[int | None, int | None, int | None]:
    ix = int(np.floor((float(x) - costmap.origin_x) / costmap.resolution))
    iy = int(np.floor((float(y) - costmap.origin_y) / costmap.resolution))
    if ix < 0 or iy < 0 or ix >= costmap.width or iy >= costmap.height:
        return None, None, None
    return ix, iy, int(costmap.data[iy * costmap.width + ix])


def _metadata_valid(costmap: CostmapSnapshot, expected_frame: str) -> bool:
    values = (
        costmap.resolution,
        costmap.origin_x,
        costmap.origin_y,
        costmap.origin_yaw,
    )
    return bool(
        costmap.frame_id == expected_frame
        and costmap.width > 0
        and costmap.height > 0
        and all(isfinite(float(value)) for value in values)
        and costmap.resolution > 0.0
        and abs(costmap.origin_yaw) <= 1.0e-6
        and len(costmap.data) == costmap.width * costmap.height
    )


def validate_swept_trajectory(
    states: Sequence[Sequence[float]] | np.ndarray,
    costmap: CostmapSnapshot,
    *,
    expected_frame: str,
    robot_length: float,
    robot_width: float,
    footprint_padding: float,
    inflation_radius: float,
    interpolation_spacing: float | None = None,
    allow_initial_inflated_center_prefix: bool = False,
    allow_known_soft_center: bool = False,
    verified_start_clearance_center: Sequence[float] | None = None,
    verified_start_clearance_radius: float | None = None,
) -> InflatedCostmapCheck:
    """Require a solved trajectory to remain in known, footprint-safe space.

    The Nav2 layer is configured so every physical obstacle is inflated beyond
    the padded LIMO circumscribed radius.  Consequently the robot centre must
    remain in cost exactly zero.  Dense samples over the rotated footprint are
    additionally required to stay in *known* cells, preventing an offset from
    putting a corner into unobserved/occluded space.  Poses are interpolated
    between MPC knots so a corner cannot pass through an obstacle between two
    otherwise valid discrete states.

    ``allow_initial_inflated_center_prefix`` is a narrowly scoped recovery
    policy for a robot whose initial centre is already in soft Nav2 inflation.
    When explicitly enabled *and the first centre sample is soft* (cost 1
    through 98), later positive samples may hold or decrease but never exceed
    the preceding positive sample.  Zero-cost cells between those samples do
    not end recovery: a discretized inflation band can contain zero-valued
    gaps.  A subsequent control horizon beginning at zero cannot activate the
    exception, so entry into soft inflation from known-free space still fails
    closed.  Cost 99 (Nav2's inscribed value), unknown or occupied centre
    cells, and unknown or occupied padded-footprint samples are never
    permitted.  The default remains the strict zero-centre policy.

    ``allow_known_soft_center`` aligns the check with Nav2's footprint-aware
    planner: known soft-inflation centre costs 1 through 98 may be traversed,
    but the complete padded footprint is still densely checked below and must
    remain known and free of lethal occupancy.  Cost 99 (inscribed), cost 100
    (lethal), unknown cells, and cells outside the costmap remain hard stops.
    The option is explicit and disabled by default for callers that rely on a
    zero-centre clearance certificate.

    A front-limited lidar cannot observe a rear padded-footprint corner during
    a small initial turn, even when the complete footprint at rest is known
    free.  ``verified_start_clearance_center`` and
    ``verified_start_clearance_radius`` provide a separate, opt-in bootstrap
    contract for that exact case.  The *first* footprint must still be fully
    known and free.  Later unknown footprint samples are permitted only inside
    the fixed, operator-verified start-clearance disc, must recover to a fully
    known footprint before the trajectory ends, and may not re-enter unknown
    space.  Unknown centre cells, occupied cells, and unknown footprint cells
    anywhere outside that disc always fail closed.
    """

    values = (
        robot_length,
        robot_width,
        footprint_padding,
        inflation_radius,
    )
    if (
        not expected_frame
        or not all(isfinite(float(value)) for value in values)
        or robot_length <= 0.0
        or robot_width <= 0.0
        or footprint_padding < 0.0
        or inflation_radius <= 0.0
    ):
        return InflatedCostmapCheck(False, "TRAJECTORY_FOOTPRINT_CONFIG_INVALID")
    if not _metadata_valid(costmap, expected_frame):
        return InflatedCostmapCheck(False, "TRAJECTORY_COSTMAP_INVALID")

    start_clearance: np.ndarray | None = None
    start_clearance_radius: float | None = None
    if (
        verified_start_clearance_center is None
        and verified_start_clearance_radius is not None
    ) or (
        verified_start_clearance_center is not None
        and verified_start_clearance_radius is None
    ):
        return InflatedCostmapCheck(
            False, "TRAJECTORY_START_CLEARANCE_CONFIG_INVALID"
        )
    if verified_start_clearance_center is not None:
        try:
            start_clearance = np.asarray(
                verified_start_clearance_center, dtype=np.float64
            )
            start_clearance_radius = float(verified_start_clearance_radius)
        except (TypeError, ValueError):
            return InflatedCostmapCheck(
                False, "TRAJECTORY_START_CLEARANCE_CONFIG_INVALID"
            )
        if (
            start_clearance.shape != (2,)
            or not np.all(np.isfinite(start_clearance))
            or not isfinite(start_clearance_radius)
            or start_clearance_radius <= 0.0
        ):
            return InflatedCostmapCheck(
                False, "TRAJECTORY_START_CLEARANCE_CONFIG_INVALID"
            )

    spacing = (
        0.5 * costmap.resolution
        if interpolation_spacing is None
        else float(interpolation_spacing)
    )
    if not isfinite(spacing) or spacing <= 0.0 or spacing > costmap.resolution:
        return InflatedCostmapCheck(False, "TRAJECTORY_SPACING_INVALID")
    try:
        trajectory = np.asarray(states, dtype=np.float64)
    except (TypeError, ValueError):
        return InflatedCostmapCheck(False, "TRAJECTORY_STATES_INVALID")
    if trajectory.ndim != 2 or trajectory.shape[0] < 4 or trajectory.shape[1] < 1:
        return InflatedCostmapCheck(False, "TRAJECTORY_STATES_INVALID")
    poses = trajectory[[0, 1, 3], :].T
    if not np.all(np.isfinite(poses)):
        return InflatedCostmapCheck(False, "TRAJECTORY_STATES_NONFINITE")

    half_length = 0.5 * robot_length + footprint_padding
    half_width = 0.5 * robot_width + footprint_padding
    footprint_spacing = min(spacing, 0.5 * costmap.resolution)
    local_x = np.linspace(
        -half_length,
        half_length,
        int(ceil(2.0 * half_length / footprint_spacing)) + 1,
    )
    local_y = np.linspace(
        -half_width,
        half_width,
        int(ceil(2.0 * half_width / footprint_spacing)) + 1,
    )
    footprint_samples = np.asarray(
        [(x, y) for x in local_x for y in local_y], dtype=np.float64
    )
    radius = hypot(half_length, half_width)
    # A zero-valued centre cell is a footprint-clear certificate only if the
    # configured obstacle inflation covers the complete padded footprint and
    # the worst centre-cell quantization error.
    required_inflation = radius + hypot(
        0.5 * costmap.resolution, 0.5 * costmap.resolution
    )
    if inflation_radius + 1.0e-12 < required_inflation:
        return InflatedCostmapCheck(
            False, "TRAJECTORY_INFLATION_CONTRACT_INVALID"
        )
    if (
        start_clearance_radius is not None
        and start_clearance_radius + 1.0e-12 < required_inflation
    ):
        return InflatedCostmapCheck(
            False, "TRAJECTORY_START_CLEARANCE_CONFIG_INVALID"
        )

    start_clearance_active = bool(
        start_clearance is not None
        and start_clearance_radius is not None
        and hypot(
            float(poses[0, 0] - start_clearance[0]),
            float(poses[0, 1] - start_clearance[1]),
        )
        <= start_clearance_radius + 1.0e-12
    )

    swept: list[np.ndarray] = [poses[0]]
    for start, end in zip(poses[:-1], poses[1:]):
        translation = hypot(end[0] - start[0], end[1] - start[1])
        yaw_delta = atan2(sin(end[2] - start[2]), cos(end[2] - start[2]))
        rotation = abs(yaw_delta) * radius
        count = max(1, int(ceil((translation + rotation) / spacing)))
        for fraction in np.linspace(0.0, 1.0, count + 1)[1:]:
            pose = start + fraction * (end - start)
            pose[2] = start[2] + fraction * yaw_delta
            swept.append(pose)

    initial_soft_recovery = False
    previous_positive_center_cost: int | None = None
    start_unknown_seen = False
    start_unknown_recovered = False
    for sample_index, pose in enumerate(swept):
        x, y, yaw = (float(value) for value in pose)
        cell_x, cell_y, value = _cell(costmap, x, y)
        if value is None:
            return InflatedCostmapCheck(
                False,
                "TRAJECTORY_CENTER_OUTSIDE_COSTMAP",
                sample_index,
                cell_x,
                cell_y,
                value,
            )
        if value < 0:
            return InflatedCostmapCheck(
                False,
                "TRAJECTORY_CENTER_UNKNOWN",
                sample_index,
                cell_x,
                cell_y,
                value,
            )
        # nav_msgs/OccupancyGrid represents Nav2's inscribed-inflated cost as
        # 99 and lethal occupancy as 100.  Neither belongs to the narrowly
        # permitted soft-inflation recovery range.
        if value >= 99:
            return InflatedCostmapCheck(
                False,
                "TRAJECTORY_CENTER_NOT_FREE",
                sample_index,
                cell_x,
                cell_y,
                value,
            )
        if value > 0:
            if not allow_known_soft_center:
                if sample_index == 0 and allow_initial_inflated_center_prefix:
                    initial_soft_recovery = True
                if not initial_soft_recovery:
                    return InflatedCostmapCheck(
                        False,
                        "TRAJECTORY_CENTER_NOT_FREE",
                        sample_index,
                        cell_x,
                        cell_y,
                        value,
                    )
                if (
                    previous_positive_center_cost is not None
                    and value > previous_positive_center_cost
                ):
                    return InflatedCostmapCheck(
                        False,
                        "TRAJECTORY_CENTER_INFLATION_INCREASE",
                        sample_index,
                        cell_x,
                        cell_y,
                        value,
                    )
                previous_positive_center_cost = value

        rotation = np.asarray(
            [[cos(yaw), -sin(yaw)], [sin(yaw), cos(yaw)]],
            dtype=np.float64,
        )
        world_samples = footprint_samples @ rotation.T + np.asarray([x, y])
        sample_has_allowed_unknown = False
        for point in world_samples:
            foot_x, foot_y, foot_value = _cell(costmap, point[0], point[1])
            if foot_value is None:
                return InflatedCostmapCheck(
                    False,
                    "TRAJECTORY_FOOTPRINT_OUTSIDE_COSTMAP",
                    sample_index,
                    foot_x,
                    foot_y,
                    foot_value,
                )
            if foot_value < 0:
                point_in_verified_start = bool(
                    start_clearance_active
                    and sample_index > 0
                    and not start_unknown_recovered
                    and start_clearance is not None
                    and start_clearance_radius is not None
                    and hypot(
                        float(point[0] - start_clearance[0]),
                        float(point[1] - start_clearance[1]),
                    )
                    <= start_clearance_radius + 1.0e-12
                )
                if not point_in_verified_start:
                    return InflatedCostmapCheck(
                        False,
                        "TRAJECTORY_FOOTPRINT_UNKNOWN",
                        sample_index,
                        foot_x,
                        foot_y,
                        foot_value,
                    )
                sample_has_allowed_unknown = True
            if foot_value >= 100:
                return InflatedCostmapCheck(
                    False,
                    "TRAJECTORY_FOOTPRINT_OCCUPIED",
                    sample_index,
                    foot_x,
                    foot_y,
                    foot_value,
                )
        if sample_has_allowed_unknown:
            start_unknown_seen = True
        elif start_unknown_seen:
            start_unknown_recovered = True
    if start_unknown_seen and not start_unknown_recovered:
        return InflatedCostmapCheck(
            False, "TRAJECTORY_START_CLEARANCE_NOT_RECOVERED"
        )
    return InflatedCostmapCheck(True, "TRAJECTORY_COSTMAP_CLEAR")
