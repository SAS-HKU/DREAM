"""Route-relative DREAM maneuver gating for arbitrary free-space paths.

Upstream DREAM applies its risk veto only to a lane-change maneuver; ordinary
lane following remains available so the ego can approach/past an occluder and
gain visibility.  Free navigation has no lane indices, so the equivalent
contract is expressed relative to the robot's current heading: a path segment
with material lateral displacement or heading change is a maneuver, while a
near-straight segment is route following.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, isfinite, pi, sin
from typing import Callable, Sequence

import numpy as np

from dream_limo.limo_scale import IntegrationPreset

from .path_tracking import validate_path_points


Array = np.ndarray


@dataclass(frozen=True)
class RouteRiskDecision:
    maneuver: bool
    vetoed: bool
    score: float
    maximum: float
    mean: float
    sampled_points: Array
    sampled_risk: Array


def _wrap_angle(value: float) -> float:
    return (float(value) + pi) % (2.0 * pi) - pi


def _closest_route_from_ego(
    points: Array, ego: Array, maximum_cross_track_error: float
) -> Array:
    best_distance = float("inf")
    best_index = 0
    best_projection = points[0]
    for index, (start, end) in enumerate(zip(points[:-1], points[1:])):
        delta = end - start
        denominator = float(delta @ delta)
        if denominator <= 1.0e-18:
            continue
        fraction = float(np.clip(((ego - start) @ delta) / denominator, 0.0, 1.0))
        projection = start + fraction * delta
        distance = float((projection - ego) @ (projection - ego))
        if distance < best_distance:
            best_distance = distance
            best_index = index
            best_projection = projection
    if best_distance**0.5 > maximum_cross_track_error:
        raise ValueError("ego is too far from the current geometric path")
    route = np.vstack((ego, best_projection, points[best_index + 1:]))
    keep = np.concatenate(
        ([True], np.linalg.norm(np.diff(route, axis=0), axis=1) > 1.0e-8)
    )
    return route[keep]


def sample_upcoming_route(
    path_points: Sequence[Sequence[float]] | Array,
    *,
    ego_xy: Sequence[float] | Array,
    lookahead: float,
    samples: int,
    maximum_cross_track_error: float = 0.10,
) -> Array:
    """Sample a path ahead of the closest ego projection by arc length."""

    if not isfinite(float(lookahead)) or lookahead <= 0.0:
        raise ValueError("route-risk lookahead must be finite and positive")
    if not isinstance(samples, (int, np.integer)) or int(samples) < 2:
        raise ValueError("route-risk sampling needs at least two points")
    if (
        not isfinite(float(maximum_cross_track_error))
        or maximum_cross_track_error <= 0.0
    ):
        raise ValueError("maximum route cross-track error must be positive")
    ego = np.asarray(ego_xy, dtype=np.float64)
    if ego.shape != (2,) or not np.all(np.isfinite(ego)):
        raise ValueError("ego position must be a finite two-vector")
    points = validate_path_points(path_points)
    route = _closest_route_from_ego(
        points, ego, float(maximum_cross_track_error)
    )
    if len(route) < 2:
        return np.repeat(ego[None, :], int(samples), axis=0)
    lengths = np.linalg.norm(np.diff(route, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    distances = np.linspace(0.0, min(float(lookahead), cumulative[-1]), int(samples))
    sampled = np.empty((len(distances), 2), dtype=np.float64)
    for output_index, distance in enumerate(distances):
        if distance >= cumulative[-1]:
            sampled[output_index] = route[-1]
            continue
        segment = int(np.searchsorted(cumulative, distance, side="right") - 1)
        segment = min(max(segment, 0), len(route) - 2)
        span = cumulative[segment + 1] - cumulative[segment]
        fraction = 0.0 if span <= 1.0e-12 else (distance - cumulative[segment]) / span
        sampled[output_index] = route[segment] + fraction * (
            route[segment + 1] - route[segment]
        )
    return sampled


def evaluate_route_maneuver_risk(
    path_points: Sequence[Sequence[float]] | Array,
    *,
    ego_xy: Sequence[float] | Array,
    ego_yaw: float,
    risk_at: Callable[[float, float], float],
    preset: IntegrationPreset,
    lookahead: float,
    samples: int,
    minimum_lateral_displacement: float = 0.15,
    minimum_heading_change: float = np.deg2rad(15.0),
    maximum_cross_track_error: float = 0.10,
) -> RouteRiskDecision:
    """Apply DREAM's ``0.6*max + 0.4*mean`` veto to a route maneuver."""

    values = (
        ego_yaw,
        minimum_lateral_displacement,
        minimum_heading_change,
    )
    if not all(isfinite(float(value)) for value in values):
        raise ValueError("route-maneuver parameters must be finite")
    if minimum_lateral_displacement < 0.0 or minimum_heading_change < 0.0:
        raise ValueError("route-maneuver thresholds must be non-negative")
    sampled = sample_upcoming_route(
        path_points,
        ego_xy=ego_xy,
        lookahead=lookahead,
        samples=samples,
        maximum_cross_track_error=maximum_cross_track_error,
    )
    ego = np.asarray(ego_xy, dtype=np.float64)
    displacement = sampled[-1] - ego
    lateral = -sin(float(ego_yaw)) * displacement[0] + cos(float(ego_yaw)) * displacement[1]
    deltas = np.diff(sampled, axis=0)
    nonzero = np.flatnonzero(np.linalg.norm(deltas, axis=1) > 1.0e-8)
    terminal_heading = (
        float(ego_yaw)
        if nonzero.size == 0
        else float(np.arctan2(deltas[nonzero[-1], 1], deltas[nonzero[-1], 0]))
    )
    heading_change = abs(_wrap_angle(terminal_heading - float(ego_yaw)))
    maneuver = bool(
        abs(float(lateral)) >= minimum_lateral_displacement
        or heading_change >= minimum_heading_change
    )
    risk = np.asarray(
        [float(risk_at(float(point[0]), float(point[1]))) for point in sampled],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(risk)) or np.any(risk < 0.0):
        raise ValueError("route risk queries must be finite and non-negative")
    maximum = float(np.max(risk))
    mean = float(np.mean(risk))
    score = 0.6 * maximum + 0.4 * mean
    vetoed = bool(
        preset.decision_veto
        and maneuver
        and score > float(preset.decision_threshold)
    )
    return RouteRiskDecision(
        maneuver=maneuver,
        vetoed=vetoed,
        score=score,
        maximum=maximum,
        mean=mean,
        sampled_points=sampled,
        sampled_risk=risk,
    )


def heading_hold_path(
    *,
    ego_x: float,
    ego_y: float,
    ego_yaw: float,
    distance: float,
    samples: int = 12,
) -> Array:
    """Return the free-space equivalent of upstream DREAM lane keeping."""

    values = (ego_x, ego_y, ego_yaw, distance)
    if not all(isfinite(float(value)) for value in values) or distance <= 0.0:
        raise ValueError("heading-hold pose and distance must be finite and positive")
    if not isinstance(samples, (int, np.integer)) or int(samples) < 2:
        raise ValueError("heading-hold path requires at least two samples")
    progress = np.linspace(0.0, float(distance), int(samples))
    return np.column_stack(
        (
            float(ego_x) + progress * cos(float(ego_yaw)),
            float(ego_y) + progress * sin(float(ego_yaw)),
        )
    )
