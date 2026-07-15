"""Physical safety metrics for ego-versus-traffic trajectory evaluation.

The submitted experiments used centre-distance and longitudinal TTC proxies.
Those proxies are unsuitable for lateral merge conflicts because they ignore
vehicle footprints and two-dimensional relative motion.  This module provides
an evaluation-only replacement based on oriented rectangular footprints.

The constant-velocity TTC is exact for translating rectangles while headings
remain fixed over the prediction interval.  ``math.inf`` means that no contact
is predicted within the declared TTC horizon; it is not converted to an
arbitrary numerical cap.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable, Mapping, Sequence

import numpy as np


_EPS = 1e-9


@dataclass(frozen=True)
class KinematicBoxState:
    """Planar vehicle state with an oriented rectangular footprint."""

    x: float
    y: float
    heading: float
    vx: float
    vy: float
    length: float
    width: float
    label: str = "vehicle"

    def __post_init__(self) -> None:
        values = (self.x, self.y, self.heading, self.vx, self.vy,
                  self.length, self.width)
        if not all(math.isfinite(float(v)) for v in values):
            raise ValueError("KinematicBoxState values must be finite")
        if self.length <= 0.0 or self.width <= 0.0:
            raise ValueError("Vehicle length and width must be positive")


@dataclass(frozen=True)
class SceneSafetySnapshot:
    """Minimum ego safety quantities at one simulation step."""

    min_clearance_m: float
    min_ttc_s: float
    clearance_vehicle: str | None
    ttc_vehicle: str | None


def _body_axes(state: KinematicBoxState) -> tuple[np.ndarray, np.ndarray]:
    longitudinal = np.array(
        [math.cos(state.heading), math.sin(state.heading)], dtype=float
    )
    lateral = np.array([-longitudinal[1], longitudinal[0]], dtype=float)
    return longitudinal, lateral


def oriented_box_corners(state: KinematicBoxState) -> np.ndarray:
    """Return the four rectangle corners in counter-clockwise order."""

    longitudinal, lateral = _body_axes(state)
    center = np.array([state.x, state.y], dtype=float)
    half_l = 0.5 * state.length
    half_w = 0.5 * state.width
    return np.array(
        [
            center - half_l * longitudinal - half_w * lateral,
            center + half_l * longitudinal - half_w * lateral,
            center + half_l * longitudinal + half_w * lateral,
            center - half_l * longitudinal + half_w * lateral,
        ],
        dtype=float,
    )


def _projection_interval(
    state: KinematicBoxState, axis: np.ndarray
) -> tuple[float, float]:
    longitudinal, lateral = _body_axes(state)
    center_projection = state.x * axis[0] + state.y * axis[1]
    radius = (
        0.5 * state.length * abs(float(np.dot(longitudinal, axis)))
        + 0.5 * state.width * abs(float(np.dot(lateral, axis)))
    )
    return center_projection - radius, center_projection + radius


def _point_segment_distance(
    point: np.ndarray, start: np.ndarray, end: np.ndarray
) -> float:
    segment = end - start
    denom = float(np.dot(segment, segment))
    if denom <= _EPS:
        return float(np.linalg.norm(point - start))
    fraction = float(np.dot(point - start, segment) / denom)
    fraction = min(1.0, max(0.0, fraction))
    closest = start + fraction * segment
    return float(np.linalg.norm(point - closest))


def _polygon_distance(first: np.ndarray, second: np.ndarray) -> float:
    distances: list[float] = []
    for polygon_a, polygon_b in ((first, second), (second, first)):
        for point in polygon_a:
            for index in range(len(polygon_b)):
                distances.append(
                    _point_segment_distance(
                        point,
                        polygon_b[index],
                        polygon_b[(index + 1) % len(polygon_b)],
                    )
                )
    return min(distances)


def signed_oriented_box_clearance(
    first: KinematicBoxState, second: KinematicBoxState
) -> float:
    """Return signed surface clearance between two oriented rectangles.

    Positive values are Euclidean surface-to-surface gaps.  Zero denotes
    contact.  Negative values are the minimum separating-axis penetration
    required to resolve an overlap.
    """

    first_axes = _body_axes(first)
    second_axes = _body_axes(second)
    overlaps: list[float] = []
    separated = False

    for axis in (*first_axes, *second_axes):
        first_min, first_max = _projection_interval(first, axis)
        second_min, second_max = _projection_interval(second, axis)
        gap = max(second_min - first_max, first_min - second_max)
        if gap > _EPS:
            separated = True
        else:
            # Minimum translation needed to separate the intervals.  This
            # form also handles full containment, where the intersection
            # length alone would underestimate penetration.
            overlaps.append(max(0.0, min(
                first_max - second_min,
                second_max - first_min,
            )))

    if separated:
        return _polygon_distance(
            oriented_box_corners(first), oriented_box_corners(second)
        )
    if not overlaps:
        return 0.0
    penetration = min(overlaps)
    return -float(penetration) if penetration > _EPS else 0.0


def constant_velocity_ttc(
    first: KinematicBoxState,
    second: KinematicBoxState,
    horizon_s: float = 10.0,
) -> float:
    """Return time to first footprint contact under constant translation.

    The calculation uses dynamic separating-axis intervals.  Vehicle headings
    are held fixed, while both centres translate using their global velocity
    vectors.  ``math.inf`` indicates no predicted contact within ``horizon_s``.
    """

    if horizon_s <= 0.0 or not math.isfinite(horizon_s):
        raise ValueError("horizon_s must be a positive finite value")

    relative_velocity = np.array(
        [second.vx - first.vx, second.vy - first.vy], dtype=float
    )
    enter_time = -math.inf
    exit_time = math.inf

    for axis in (*_body_axes(first), *_body_axes(second)):
        first_min, first_max = _projection_interval(first, axis)
        second_min, second_max = _projection_interval(second, axis)
        relative_axis_speed = float(np.dot(relative_velocity, axis))

        if abs(relative_axis_speed) <= _EPS:
            if second_min > first_max + _EPS or first_min > second_max + _EPS:
                return math.inf
            continue

        crossing_1 = (first_min - second_max) / relative_axis_speed
        crossing_2 = (first_max - second_min) / relative_axis_speed
        axis_enter = min(crossing_1, crossing_2)
        axis_exit = max(crossing_1, crossing_2)
        enter_time = max(enter_time, axis_enter)
        exit_time = min(exit_time, axis_exit)

        if enter_time > exit_time + _EPS:
            return math.inf

    if exit_time < -_EPS or enter_time > horizon_s + _EPS:
        return math.inf
    return float(max(0.0, enter_time))


def evaluate_scene_safety(
    ego: KinematicBoxState,
    obstacles: Iterable[KinematicBoxState],
    ttc_horizon_s: float = 10.0,
) -> SceneSafetySnapshot:
    """Evaluate the minimum ego clearance and TTC against all obstacles."""

    min_clearance = math.inf
    min_ttc = math.inf
    clearance_vehicle: str | None = None
    ttc_vehicle: str | None = None

    for obstacle in obstacles:
        clearance = signed_oriented_box_clearance(ego, obstacle)
        if clearance < min_clearance:
            min_clearance = clearance
            clearance_vehicle = obstacle.label

        ttc = constant_velocity_ttc(ego, obstacle, horizon_s=ttc_horizon_s)
        if ttc < min_ttc:
            min_ttc = ttc
            ttc_vehicle = obstacle.label

    return SceneSafetySnapshot(
        min_clearance_m=float(min_clearance),
        min_ttc_s=float(min_ttc),
        clearance_vehicle=clearance_vehicle,
        ttc_vehicle=ttc_vehicle,
    )


def _interpolate_heading(start: float, end: float, fraction: float) -> float:
    """Interpolate a heading through its shortest signed angular difference."""

    delta = math.atan2(math.sin(end - start), math.cos(end - start))
    return float(start + float(fraction) * delta)


def interpolate_box_state(
    start: KinematicBoxState,
    end: KinematicBoxState,
    fraction: float,
) -> KinematicBoxState:
    """Linearly interpolate a footprint state between two logged frames.

    The utility is intentionally evaluation-only.  It does not claim that the
    planner used this interpolation; it avoids missing a physical overlap
    simply because two recorded control frames are 0.1 s apart.
    """

    if start.label != end.label:
        raise ValueError("Interpolated states must describe the same vehicle")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0, 1]")
    if not math.isclose(start.length, end.length) or not math.isclose(
        start.width, end.width
    ):
        raise ValueError("Vehicle footprint dimensions must remain fixed")

    def blend(first: float, second: float) -> float:
        return float(first + fraction * (second - first))

    return KinematicBoxState(
        x=blend(start.x, end.x),
        y=blend(start.y, end.y),
        heading=_interpolate_heading(start.heading, end.heading, fraction),
        vx=blend(start.vx, end.vx),
        vy=blend(start.vy, end.vy),
        length=start.length,
        width=start.width,
        label=start.label,
    )


def evaluate_swept_pair_safety(
    ego_start: KinematicBoxState,
    ego_end: KinematicBoxState,
    obstacle_start: KinematicBoxState,
    obstacle_end: KinematicBoxState,
    *,
    interval_s: float,
    max_substep_s: float = 0.01,
    ttc_horizon_s: float = 10.0,
) -> SceneSafetySnapshot:
    """Evaluate a footprint pair throughout a recorded simulation interval.

    The original implementation evaluated only logged control frames.  At
    urban-road speeds, an overlap can begin and end between 0.1-s frames.  We
    therefore linearly interpolate the logged poses at a declared maximum
    substep (default 0.01 s) and return the most critical clearance and TTC.
    This is a conservative discretized sweep, not a replacement for the
    continuous-time model used by the controller.
    """

    if interval_s <= 0.0 or not math.isfinite(interval_s):
        raise ValueError("interval_s must be positive and finite")
    if max_substep_s <= 0.0 or not math.isfinite(max_substep_s):
        raise ValueError("max_substep_s must be positive and finite")

    n_substeps = max(1, int(math.ceil(interval_s / max_substep_s)))
    minimum = SceneSafetySnapshot(
        min_clearance_m=math.inf,
        min_ttc_s=math.inf,
        clearance_vehicle=None,
        ttc_vehicle=None,
    )
    for index in range(n_substeps + 1):
        fraction = index / n_substeps
        snapshot = evaluate_scene_safety(
            interpolate_box_state(ego_start, ego_end, fraction),
            [interpolate_box_state(obstacle_start, obstacle_end, fraction)],
            ttc_horizon_s=ttc_horizon_s,
        )
        if snapshot.min_clearance_m < minimum.min_clearance_m:
            minimum = SceneSafetySnapshot(
                min_clearance_m=snapshot.min_clearance_m,
                min_ttc_s=minimum.min_ttc_s,
                clearance_vehicle=snapshot.clearance_vehicle,
                ttc_vehicle=minimum.ttc_vehicle,
            )
        if snapshot.min_ttc_s < minimum.min_ttc_s:
            minimum = SceneSafetySnapshot(
                min_clearance_m=minimum.min_clearance_m,
                min_ttc_s=snapshot.min_ttc_s,
                clearance_vehicle=minimum.clearance_vehicle,
                ttc_vehicle=snapshot.ttc_vehicle,
            )
    return minimum


def _event_count(mask: np.ndarray) -> int:
    if mask.size == 0:
        return 0
    previous = np.concatenate(([False], mask[:-1]))
    return int(np.count_nonzero(mask & ~previous))


def _finite_min(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.min(finite)) if finite.size else math.inf


def summarize_episode_safety(
    snapshots: Sequence[SceneSafetySnapshot],
    *,
    reveal_step: int | None,
    post_reveal_steps: int,
    near_clearance_m: float = 1.0,
    ttc_horizon_s: float = 10.0,
) -> dict[str, object]:
    """Reduce per-step scene metrics to one auditable episode record."""

    if near_clearance_m <= 0.0:
        raise ValueError("near_clearance_m must be positive")
    if post_reveal_steps <= 0:
        raise ValueError("post_reveal_steps must be positive")
    if not snapshots:
        raise ValueError("At least one safety snapshot is required")

    clearances = np.asarray([s.min_clearance_m for s in snapshots], dtype=float)
    ttc_values = np.asarray([s.min_ttc_s for s in snapshots], dtype=float)
    collision_mask = clearances <= 0.0
    near_mask = (clearances > 0.0) & (clearances < near_clearance_m)
    critical_mask = clearances < near_clearance_m

    post_start: int | None = None
    post_stop: int | None = None
    post_clearance = math.nan
    post_ttc = math.nan
    if reveal_step is not None and 0 <= reveal_step < len(snapshots):
        post_start = int(reveal_step)
        post_stop = min(len(snapshots), post_start + int(post_reveal_steps))
        post_clearance = float(np.min(clearances[post_start:post_stop]))
        post_ttc = _finite_min(ttc_values[post_start:post_stop])

    min_ttc = _finite_min(ttc_values)
    return {
        "n_steps": int(len(snapshots)),
        "reveal_step": None if reveal_step is None else int(reveal_step),
        "post_reveal_available": bool(post_start is not None),
        "post_reveal_start_step": post_start,
        "post_reveal_stop_step": post_stop,
        "near_clearance_m": float(near_clearance_m),
        "ttc_horizon_s": float(ttc_horizon_s),
        "collision_incident": bool(np.any(collision_mask)),
        "near_collision_incident": bool(np.any(near_mask)),
        "collision_or_near_incident": bool(np.any(critical_mask)),
        "collision_event_count": _event_count(collision_mask),
        "near_collision_event_count": _event_count(near_mask),
        "min_clearance_m": float(np.min(clearances)),
        "min_ttc_s": min_ttc,
        "min_ttc_censored": bool(math.isinf(min_ttc)),
        "post_reveal_min_clearance_m": post_clearance,
        "post_reveal_min_ttc_s": post_ttc,
        "post_reveal_min_ttc_censored": bool(math.isinf(post_ttc)),
    }


def aggregate_episode_safety(
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Aggregate episode-level physical metrics without inferential claims."""

    if not records:
        raise ValueError("At least one episode record is required")

    def rate(key: str) -> float:
        return float(np.mean([bool(record[key]) for record in records]))

    def conditional_rate(key: str, availability_key: str) -> float:
        available = [record for record in records if bool(record[availability_key])]
        if not available:
            return math.nan
        return float(np.mean([bool(record[key]) for record in available]))

    def distribution(key: str) -> dict[str, float | int]:
        values = np.asarray([float(record[key]) for record in records], dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return {"n_finite": 0, "median": math.nan,
                    "q25": math.nan, "q75": math.nan}
        return {
            "n_finite": int(finite.size),
            "median": float(np.median(finite)),
            "q25": float(np.quantile(finite, 0.25)),
            "q75": float(np.quantile(finite, 0.75)),
        }

    return {
        "n_episodes": int(len(records)),
        "collision_rate": rate("collision_incident"),
        "near_collision_rate": rate("near_collision_incident"),
        "collision_or_near_rate": rate("collision_or_near_incident"),
        "min_clearance_m": distribution("min_clearance_m"),
        "min_ttc_s": distribution("min_ttc_s"),
        "min_ttc_censor_rate": rate("min_ttc_censored"),
        "post_reveal_n_episodes": int(sum(
            bool(record["post_reveal_available"]) for record in records
        )),
        "post_reveal_min_clearance_m": distribution(
            "post_reveal_min_clearance_m"
        ),
        "post_reveal_min_ttc_s": distribution("post_reveal_min_ttc_s"),
        "post_reveal_min_ttc_censor_rate": conditional_rate(
            "post_reveal_min_ttc_censored", "post_reveal_available"
        ),
    }


def snapshot_to_dict(snapshot: SceneSafetySnapshot) -> dict[str, object]:
    """Convert a snapshot dataclass to a serialization-friendly dictionary."""

    return asdict(snapshot)
