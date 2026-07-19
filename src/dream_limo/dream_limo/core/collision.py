"""ROS-independent collision envelope and trajectory gate.

DRIFT risk is intentionally not a collision map.  This module keeps measured
LiDAR first-return surfaces, inflates them by the complete circular robot
footprint, and treats the externally supplied occlusion/shadow mask as unknown
space.  It is conservative by design: unknown, occupied, and outside-grid
trajectory samples are all unsafe.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, isfinite
from typing import Sequence

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion


Array = np.ndarray


@dataclass(frozen=True)
class CollisionGridSpec:
    """Nodal grid geometry matching the existing DREAM risk/mask grids."""

    width: int
    height: int
    resolution: float
    origin_x: float
    origin_y: float
    frame_id: str = "map"

    def __post_init__(self) -> None:
        if self.width < 2 or self.height < 2:
            raise ValueError("collision grid needs at least two cells per axis")
        if not isfinite(self.resolution) or self.resolution <= 0.0:
            raise ValueError("collision-grid resolution must be positive")
        if not all(isfinite(value) for value in (self.origin_x, self.origin_y)):
            raise ValueError("collision-grid origin must be finite")
        if not self.frame_id:
            raise ValueError("collision-grid frame cannot be empty")

    @property
    def x_max(self) -> float:
        return self.origin_x + (self.width - 1) * self.resolution

    @property
    def y_max(self) -> float:
        return self.origin_y + (self.height - 1) * self.resolution

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    def indices(self, points: Array) -> tuple[Array, Array, Array]:
        values = _points_array(points)
        ix = np.rint((values[:, 0] - self.origin_x) / self.resolution).astype(int)
        iy = np.rint((values[:, 1] - self.origin_y) / self.resolution).astype(int)
        valid = (
            (values[:, 0] >= self.origin_x)
            & (values[:, 0] <= self.x_max)
            & (values[:, 1] >= self.origin_y)
            & (values[:, 1] <= self.y_max)
            & (ix >= 0)
            & (ix < self.width)
            & (iy >= 0)
            & (iy < self.height)
        )
        return ix, iy, valid


@dataclass(frozen=True)
class CollisionGridDigest:
    retained_surface_cells: int
    inflated_surface_cells: int
    shadow_unknown_cells: int
    outside_road_cells: int
    blocked_cells: int


@dataclass(frozen=True)
class TrajectoryAssessment:
    clear: bool
    reason: str
    evaluated_samples: int
    first_unsafe_x: float | None = None
    first_unsafe_y: float | None = None
    first_unsafe_value: int | None = None


class CollisionEnvelope:
    """Temporally retained surface map plus fail-closed trajectory checking."""

    FREE = np.int8(0)
    UNKNOWN = np.int8(-1)
    OCCUPIED = np.int8(100)

    def __init__(
        self,
        spec: CollisionGridSpec,
        *,
        surface_retention_seconds: float,
        inflation_radius: float,
        minimum_valid_rays: int,
        interpolation_spacing: float | None = None,
        traversable_mask: Array | None = None,
    ) -> None:
        if surface_retention_seconds <= 0.0 or not isfinite(surface_retention_seconds):
            raise ValueError("surface retention must be finite and positive")
        if inflation_radius < 0.0 or not isfinite(inflation_radius):
            raise ValueError("inflation radius must be finite and nonnegative")
        if minimum_valid_rays < 1:
            raise ValueError("minimum valid rays must be positive")
        spacing = (
            0.5 * spec.resolution
            if interpolation_spacing is None
            else float(interpolation_spacing)
        )
        if spacing <= 0.0 or spacing > spec.resolution:
            raise ValueError("trajectory interpolation spacing must lie in (0,resolution]")
        self.spec = spec
        self.surface_retention_seconds = float(surface_retention_seconds)
        self.inflation_radius = float(inflation_radius)
        self.minimum_valid_rays = int(minimum_valid_rays)
        self.interpolation_spacing = spacing
        self._surface_expiry = np.full(spec.shape, -np.inf, dtype=np.float64)
        self.last_valid_ray_count = 0
        self.last_scan_accepted = False
        self._inflation_structure = self._make_inflation_structure()
        if traversable_mask is None:
            self.traversable_mask = np.ones(spec.shape, dtype=bool)
        else:
            traversable = np.asarray(traversable_mask)
            if traversable.shape != spec.shape:
                raise ValueError("traversable-mask shape mismatch")
            self.traversable_mask = traversable.astype(bool)
        # A valid center must keep the full circular robot footprint on-road.
        self.center_traversable_mask = binary_erosion(
            self.traversable_mask,
            structure=self._inflation_structure,
            border_value=0,
        )

    def _make_inflation_structure(self) -> Array:
        radius_cells = int(ceil(self.inflation_radius / self.spec.resolution))
        offsets = np.arange(-radius_cells, radius_cells + 1, dtype=np.float64)
        xx, yy = np.meshgrid(offsets, offsets)
        # Include half a grid cell to avoid under-inflating a discretized circle.
        limit = self.inflation_radius + 0.5 * self.spec.resolution
        return (np.hypot(xx, yy) * self.spec.resolution <= limit)

    def record_scan(
        self,
        transformed_surface_points: Array,
        *,
        receipt_time: float,
        valid_ray_count: int,
    ) -> int:
        """Retain surface cells when the complete scan passes its ray-count gate."""

        now = float(receipt_time)
        if not isfinite(now):
            raise ValueError("scan receipt time must be finite")
        self.last_valid_ray_count = int(valid_ray_count)
        self.last_scan_accepted = self.last_valid_ray_count >= self.minimum_valid_rays
        if not self.last_scan_accepted:
            return 0
        points = _points_array(transformed_surface_points)
        ix, iy, valid = self.spec.indices(points)
        if not np.any(valid):
            return 0
        expiry = now + self.surface_retention_seconds
        np.maximum.at(self._surface_expiry, (iy[valid], ix[valid]), expiry)
        return len(set(zip(ix[valid].tolist(), iy[valid].tolist())))

    def render(self, shadow_unknown: Array, *, now: float) -> tuple[Array, CollisionGridDigest]:
        timestamp = float(now)
        if not isfinite(timestamp):
            raise ValueError("render time must be finite")
        shadow = np.asarray(shadow_unknown)
        if shadow.shape != self.spec.shape:
            raise ValueError(
                f"shadow shape {shadow.shape} does not match {self.spec.shape}"
            )
        if not np.all(np.isfinite(shadow)):
            raise ValueError("shadow mask must be finite")
        retained = self._surface_expiry >= timestamp
        inflated = binary_dilation(retained, structure=self._inflation_structure)
        unknown = shadow > 0
        outside_road = ~self.traversable_mask
        grid = np.full(self.spec.shape, self.FREE, dtype=np.int8)
        grid[unknown] = self.UNKNOWN
        grid[outside_road] = self.OCCUPIED
        grid[inflated] = self.OCCUPIED
        digest = CollisionGridDigest(
            retained_surface_cells=int(np.count_nonzero(retained)),
            inflated_surface_cells=int(np.count_nonzero(inflated)),
            shadow_unknown_cells=int(np.count_nonzero(unknown)),
            outside_road_cells=int(np.count_nonzero(outside_road)),
            blocked_cells=int(np.count_nonzero(grid != self.FREE)),
        )
        return grid, digest

    def assess_trajectory(self, path_points: Array, collision_grid: Array) -> TrajectoryAssessment:
        grid = np.asarray(collision_grid)
        if grid.shape != self.spec.shape:
            raise ValueError("collision-grid shape mismatch")
        points = interpolate_polyline(path_points, self.interpolation_spacing)
        if len(points) == 0:
            return TrajectoryAssessment(False, "EMPTY_TRAJECTORY", 0)
        ix, iy, inside_grid = self.spec.indices(points)
        # Keeping the center inside the numerical grid is insufficient: the
        # complete circular footprint must also remain inside it.
        footprint_inside = (
            (points[:, 0] >= self.spec.origin_x + self.inflation_radius)
            & (points[:, 0] <= self.spec.x_max - self.inflation_radius)
            & (points[:, 1] >= self.spec.origin_y + self.inflation_radius)
            & (points[:, 1] <= self.spec.y_max - self.inflation_radius)
        )
        inside = inside_grid & footprint_inside
        for index, point in enumerate(points):
            if not inside[index]:
                return TrajectoryAssessment(
                    False,
                    "OUTSIDE_GRID",
                    index + 1,
                    float(point[0]),
                    float(point[1]),
                    None,
                )
            if not self.center_traversable_mask[iy[index], ix[index]]:
                return TrajectoryAssessment(
                    False,
                    "OUTSIDE_ROAD",
                    index + 1,
                    float(point[0]),
                    float(point[1]),
                    int(grid[iy[index], ix[index]]),
                )
            value = int(grid[iy[index], ix[index]])
            if value != int(self.FREE):
                return TrajectoryAssessment(
                    False,
                    "UNKNOWN_SHADOW" if value < 0 else "OCCUPIED_SURFACE",
                    index + 1,
                    float(point[0]),
                    float(point[1]),
                    value,
                )
        return TrajectoryAssessment(True, "CLEAR", len(points))


def axis_aligned_road_mask(
    spec: CollisionGridSpec, *, y_min: float, y_max: float
) -> Array:
    """Return an inclusive straight-road mask without floating boundary loss."""

    lower = float(y_min)
    upper = float(y_max)
    if not isfinite(lower) or not isfinite(upper) or lower >= upper:
        raise ValueError("road bounds must be finite and ordered")
    y_coordinates = spec.origin_y + np.arange(spec.height) * spec.resolution
    epsilon = 1.0e-9 * max(1.0, abs(spec.origin_y), abs(spec.y_max))
    rows = (y_coordinates >= lower - epsilon) & (y_coordinates <= upper + epsilon)
    return np.broadcast_to(rows[:, None], spec.shape).copy()


def interpolate_polyline(points: Array, spacing: float) -> Array:
    """Sample every segment densely enough that a grid cell cannot be skipped."""

    values = _points_array(points)
    if spacing <= 0.0 or not isfinite(spacing):
        raise ValueError("interpolation spacing must be finite and positive")
    if len(values) == 0:
        return np.empty((0, 2), dtype=np.float64)
    if len(values) == 1:
        return values.copy()
    sampled = [values[0]]
    for start, end in zip(values[:-1], values[1:]):
        distance = float(np.linalg.norm(end - start))
        count = max(1, int(ceil(distance / spacing)))
        for index in range(1, count + 1):
            sampled.append(start + (end - start) * (index / count))
    return np.asarray(sampled, dtype=np.float64)


def transform_points(
    points_xyz: Array,
    *,
    translation_xyz: Sequence[float],
    quaternion_xyzw: Sequence[float],
) -> Array:
    """Apply one rigid transform and return planar coordinates.

    The ROS node obtains this transform for the exact LaserScan timestamp.  The
    math lives here so it can be tested without ROS or a TF graph.
    """

    points = np.asarray(points_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("3-D points must have shape (N,3)")
    translation = np.asarray(tuple(translation_xyz), dtype=np.float64)
    quaternion = np.asarray(tuple(quaternion_xyzw), dtype=np.float64)
    if translation.shape != (3,) or quaternion.shape != (4,):
        raise ValueError("transform translation/quaternion have invalid dimensions")
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(translation)):
        raise ValueError("transform inputs must be finite")
    if not np.all(np.isfinite(quaternion)):
        raise ValueError("transform quaternion must be finite")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1.0e-9:
        raise ValueError("transform quaternion has zero norm")
    x, y, z, w = quaternion / norm
    rotation = np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    transformed = points @ rotation.T + translation
    return transformed[:, :2]


def _points_array(points: Array) -> Array:
    values = np.asarray(points, dtype=np.float64)
    if values.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("planar points must have shape (N,2)")
    if not np.all(np.isfinite(values)):
        raise ValueError("planar points must be finite")
    return values
