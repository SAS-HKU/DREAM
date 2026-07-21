"""Path validation and horizon reference generation for free-space tracking.

The ROS boundary is deliberately kept out of this module.  Callers provide a
polyline in the same Cartesian frame as the ego state and receive the
``[x, y, speed, yaw]`` reference consumed by :class:`RiskAwareMPC`.
"""

from __future__ import annotations

from math import cos, isfinite, pi
from typing import Sequence, Tuple

import numpy as np

from .mission import stopping_speed_limit


Array = np.ndarray


class PathValidationError(ValueError):
    """Raised when a path cannot define a finite, non-degenerate polyline."""


def validate_path_points(
    path_points: Sequence[Sequence[float]] | Array,
    *,
    duplicate_tolerance: float = 1.0e-6,
) -> Array:
    """Return a finite ``N x 2`` polyline with adjacent duplicates removed.

    Both the conventional ``N x 2`` representation and a ``2 x N`` array are
    accepted.  A path must retain at least two distinct points.  Silently
    dropping consecutive duplicates is useful for grid planners, which often
    repeat the start or goal while stitching path segments.
    """

    if not isfinite(float(duplicate_tolerance)) or duplicate_tolerance < 0.0:
        raise PathValidationError("duplicate tolerance must be finite and non-negative")
    try:
        points = np.asarray(path_points, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise PathValidationError("path points must be numeric") from exc
    if points.ndim != 2:
        raise PathValidationError("path must be a two-dimensional array")
    if points.shape[1] == 2:
        points = points.copy()
    elif points.shape[0] == 2 and points.shape[1] >= 2:
        points = points.T.copy()
    else:
        raise PathValidationError("path shape must be N x 2 or 2 x N")
    if points.shape[0] < 2:
        raise PathValidationError("path must contain at least two points")
    if not np.all(np.isfinite(points)):
        raise PathValidationError("path points must be finite")

    kept = [points[0]]
    for point in points[1:]:
        if float(np.linalg.norm(point - kept[-1])) > duplicate_tolerance:
            kept.append(point)
    clean = np.asarray(kept, dtype=np.float64)
    if clean.shape[0] < 2:
        raise PathValidationError("path has fewer than two distinct points")
    return clean


def validate_forward_pose_alignment(
    path_points: Sequence[Sequence[float]] | Array,
    pose_yaws: Sequence[float] | Array,
    *,
    minimum_cosine: float = 0.0,
) -> None:
    """Reject a geometric path containing reverse-oriented pose segments."""
    points = np.asarray(path_points, dtype=np.float64)
    yaws = np.asarray(pose_yaws, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (2,) or points.shape[0] < 2:
        raise PathValidationError("forward-path validation needs an N x 2 path")
    if yaws.shape != (points.shape[0],) or not np.all(np.isfinite(yaws)):
        raise PathValidationError("path pose yaws must match the path length")
    if not isfinite(float(minimum_cosine)) or not -1.0 < minimum_cosine < 1.0:
        raise PathValidationError("forward alignment cosine must lie in (-1, 1)")
    checked = 0
    for index, delta in enumerate(np.diff(points, axis=0)):
        length = float(np.linalg.norm(delta))
        if length <= 1.0e-8:
            continue
        heading = np.asarray([cos(yaws[index]), np.sin(yaws[index])])
        if float(heading @ (delta / length)) <= minimum_cosine:
            raise PathValidationError("path contains a reverse-oriented segment")
        checked += 1
    if checked == 0:
        raise PathValidationError("path has no forward-oriented segment")


def _closest_projection(points: Array, ego_xy: Array) -> Tuple[int, Array, float]:
    """Return the closest segment index and the projection onto that segment."""

    best_index = 0
    best_point = points[0].copy()
    best_distance_squared = float("inf")
    for index, (start, end) in enumerate(zip(points[:-1], points[1:])):
        delta = end - start
        length_squared = float(delta @ delta)
        if length_squared <= 1.0e-18:
            continue
        fraction = float(np.clip(((ego_xy - start) @ delta) / length_squared, 0.0, 1.0))
        projection = start + fraction * delta
        distance_squared = float((ego_xy - projection) @ (ego_xy - projection))
        if distance_squared < best_distance_squared:
            best_index = index
            best_point = projection
            best_distance_squared = distance_squared
    return best_index, best_point, best_distance_squared**0.5


def _route_from_ego(
    points: Array,
    ego_xy: Array,
    tolerance: float,
    maximum_cross_track_error: float,
) -> Array:
    """Trim a stale path prefix and anchor the remaining polyline at the ego."""
    segment_index, projection, cross_track_error = _closest_projection(
        points, ego_xy
    )
    if cross_track_error > maximum_cross_track_error:
        raise PathValidationError(
            "ego cross-track error exceeds the replanning limit"
        )
    candidates = [ego_xy]
    if float(np.linalg.norm(projection - candidates[-1])) > tolerance:
        candidates.append(projection)
    for point in points[segment_index + 1:]:
        if float(np.linalg.norm(point - candidates[-1])) > tolerance:
            candidates.append(point)
    return np.asarray(candidates, dtype=np.float64)


def _interpolate_polyline(route: Array, cumulative: Array, distances: Array) -> Array:
    result = np.empty((distances.size, 2), dtype=np.float64)
    for index, distance in enumerate(distances):
        if distance >= cumulative[-1]:
            result[index] = route[-1]
            continue
        segment = int(np.searchsorted(cumulative, distance, side="right") - 1)
        segment = min(max(segment, 0), route.shape[0] - 2)
        span = cumulative[segment + 1] - cumulative[segment]
        fraction = 0.0 if span <= 1.0e-12 else (distance - cumulative[segment]) / span
        result[index] = route[segment] + fraction * (route[segment + 1] - route[segment])
    return result


def _sample_yaw(route: Array, cumulative: Array, distances: Array, ego_yaw: float) -> Array:
    headings = np.empty(distances.size, dtype=np.float64)
    for index, distance in enumerate(distances):
        if distance >= cumulative[-1] - 1.0e-12:
            segment = route.shape[0] - 2
        else:
            segment = int(np.searchsorted(cumulative, distance, side="right") - 1)
            segment = min(max(segment, 0), route.shape[0] - 2)
        tangent = route[segment + 1] - route[segment]
        headings[index] = np.arctan2(tangent[1], tangent[0])
    headings = np.unwrap(headings)
    # Keep the continuous branch nearest the measured yaw.  This prevents a
    # path crossing the -pi/pi boundary from producing a spurious full turn in
    # the quadratic heading objective.
    headings += 2.0 * pi * round((float(ego_yaw) - float(headings[0])) / (2.0 * pi))
    return headings


def build_path_reference(
    path_points: Sequence[Sequence[float]] | Array,
    *,
    ego_xy: Sequence[float] | Array,
    ego_yaw: float,
    horizon: int,
    dt: float,
    cruise_speed: float,
    braking_deceleration: float,
    duplicate_tolerance: float = 1.0e-6,
    maximum_cross_track_error: float = 0.10,
    terminal_yaw: float | None = None,
) -> Array:
    """Build a ``4 x (horizon + 1)`` arc-length path reference.

    The closest point on the supplied path removes obsolete points behind the
    ego.  The first reference position is always the measured ego position.
    Subsequent positions advance by ``speed * dt`` along arc length.  Reference
    speed uses *Euclidean* distance to the final goal in the standard stopping
    relation ``v^2 = 2*a*d``; this is conservative on curved paths and prevents
    a bend from hiding a nearby destination behind a long remaining arc.
    """

    scalar_values = (
        ego_yaw,
        dt,
        cruise_speed,
        braking_deceleration,
        maximum_cross_track_error,
    )
    if not all(isfinite(float(value)) for value in scalar_values):
        raise PathValidationError("reference parameters must be finite")
    if not isinstance(horizon, (int, np.integer)) or int(horizon) < 1:
        raise PathValidationError("horizon must be a positive integer")
    if dt <= 0.0:
        raise PathValidationError("dt must be positive")
    if cruise_speed < 0.0:
        raise PathValidationError("cruise speed must be non-negative")
    if braking_deceleration <= 0.0:
        raise PathValidationError("braking deceleration must be positive")
    if maximum_cross_track_error <= 0.0:
        raise PathValidationError("maximum cross-track error must be positive")
    if terminal_yaw is not None and not isfinite(float(terminal_yaw)):
        raise PathValidationError("terminal yaw must be finite when supplied")
    ego = np.asarray(ego_xy, dtype=np.float64)
    if ego.shape != (2,) or not np.all(np.isfinite(ego)):
        raise PathValidationError("ego position must be a finite two-vector")

    points = validate_path_points(
        path_points, duplicate_tolerance=duplicate_tolerance
    )
    route = _route_from_ego(
        points,
        ego,
        duplicate_tolerance,
        float(maximum_cross_track_error),
    )
    count = int(horizon) + 1
    goal = points[-1]
    if route.shape[0] < 2:
        reference = np.zeros((4, count), dtype=np.float64)
        reference[0, :] = ego[0]
        reference[1, :] = ego[1]
        reference[3, :] = float(ego_yaw)
        return reference

    lengths = np.linalg.norm(np.diff(route, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    total_length = float(cumulative[-1])
    if total_length <= duplicate_tolerance:
        reference = np.zeros((4, count), dtype=np.float64)
        reference[0, :] = ego[0]
        reference[1, :] = ego[1]
        reference[3, :] = float(ego_yaw)
        return reference

    sample_distance = np.zeros(count, dtype=np.float64)
    positions = np.empty((count, 2), dtype=np.float64)
    speeds = np.empty(count, dtype=np.float64)
    for index in range(count):
        positions[index] = _interpolate_polyline(
            route, cumulative, sample_distance[index:index + 1]
        )[0]
        euclidean_remaining = float(np.linalg.norm(goal - positions[index]))
        speeds[index] = stopping_speed_limit(
            euclidean_remaining,
            cruise_speed=float(cruise_speed),
            braking_deceleration=float(braking_deceleration),
        )
        if index + 1 < count:
            sample_distance[index + 1] = min(
                total_length,
                sample_distance[index] + float(dt) * speeds[index],
            )

    reference = np.empty((4, count), dtype=np.float64)
    reference[0:2, :] = positions.T
    reference[2, :] = speeds
    reference[3, :] = _sample_yaw(route, cumulative, sample_distance, float(ego_yaw))
    # The clicked goal orientation belongs only to the actual route endpoint.
    # Applying it at the end of every receding horizon makes a distant goal
    # pull the controller away from the current path tangent prematurely.
    reaches_endpoint = sample_distance[-1] >= total_length - duplicate_tolerance
    if terminal_yaw is not None and reaches_endpoint:
        goal_yaw = float(terminal_yaw)
        goal_yaw += 2.0 * pi * round(
            (float(reference[3, -1]) - goal_yaw) / (2.0 * pi)
        )
        reference[3, -1] = goal_yaw
    if not np.all(np.isfinite(reference)):
        raise PathValidationError("path resampling produced non-finite values")
    return reference
