"""LiDAR visibility and hidden-track gating for the DREAM world model."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin
from typing import Sequence, Tuple

import numpy as np
from scipy.ndimage import minimum_filter1d


Array = np.ndarray
Point = Tuple[float, float]


def wrap_angle(value: Array | float) -> Array | float:
    return (value + pi) % (2.0 * pi) - pi


@dataclass(frozen=True)
class PolygonObstacle:
    name: str
    vertices: Tuple[Point, ...]
    vehicle_class: str = "truck"
    heading: float = 0.0

    def __post_init__(self) -> None:
        if len(self.vertices) < 3:
            raise ValueError("an obstacle polygon requires at least three vertices")
        values = np.asarray(self.vertices, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("obstacle vertices must be finite")
        if abs(polygon_area(self.vertices)) < 1.0e-8:
            raise ValueError("obstacle polygon has zero area")

    @property
    def centroid(self) -> Point:
        values = np.asarray(self.vertices, dtype=np.float64)
        return float(np.mean(values[:, 0])), float(np.mean(values[:, 1]))

    @property
    def length(self) -> float:
        values = np.asarray(self.vertices, dtype=np.float64)
        ch, sh = cos(self.heading), sin(self.heading)
        projection = ch * values[:, 0] + sh * values[:, 1]
        return float(np.max(projection) - np.min(projection))

    @property
    def width(self) -> float:
        values = np.asarray(self.vertices, dtype=np.float64)
        ch, sh = cos(self.heading), sin(self.heading)
        projection = -sh * values[:, 0] + ch * values[:, 1]
        return float(np.max(projection) - np.min(projection))


def polygon_area(vertices: Sequence[Point]) -> float:
    points = np.asarray(vertices, dtype=np.float64)
    shifted = np.roll(points, -1, axis=0)
    return 0.5 * float(np.sum(points[:, 0] * shifted[:, 1] - shifted[:, 0] * points[:, 1]))


def rectangle_polygon(
    name: str,
    center_x: float,
    center_y: float,
    length: float,
    width: float,
    heading: float = 0.0,
    vehicle_class: str = "truck",
) -> PolygonObstacle:
    if length <= 0.0 or width <= 0.0:
        raise ValueError("rectangle dimensions must be positive")
    local = np.asarray(
        [
            [-length / 2.0, -width / 2.0],
            [length / 2.0, -width / 2.0],
            [length / 2.0, width / 2.0],
            [-length / 2.0, width / 2.0],
        ]
    )
    ch, sh = cos(heading), sin(heading)
    rotation = np.asarray([[ch, -sh], [sh, ch]])
    points = local @ rotation.T + np.asarray([center_x, center_y])
    return PolygonObstacle(
        name=name,
        vertices=tuple((float(x), float(y)) for x, y in points),
        vehicle_class=vehicle_class,
        heading=heading,
    )


def points_in_polygon(x: Array, y: Array, polygon: PolygonObstacle) -> Array:
    """Vectorized even/odd point-in-polygon test."""
    inside = np.zeros(np.broadcast(x, y).shape, dtype=bool)
    vertices = polygon.vertices
    j = len(vertices) - 1
    for i in range(len(vertices)):
        xi, yi = vertices[i]
        xj, yj = vertices[j]
        crossing = ((yi > y) != (yj > y)) & (
            x < (xj - xi) * (y - yi) / (yj - yi + 1.0e-15) + xi
        )
        inside ^= crossing
        j = i
    return inside


def point_to_polygon_distance(x: Array, y: Array, polygon: PolygonObstacle) -> Array:
    distance_sq = np.full(np.broadcast(x, y).shape, np.inf, dtype=np.float64)
    vertices = polygon.vertices
    for start, end in zip(vertices, (*vertices[1:], vertices[0])):
        ax, ay = start
        bx, by = end
        dx, dy = bx - ax, by - ay
        denom = max(dx * dx + dy * dy, 1.0e-15)
        t = np.clip(((x - ax) * dx + (y - ay) * dy) / denom, 0.0, 1.0)
        px, py = ax + t * dx, ay + t * dy
        distance_sq = np.minimum(distance_sq, (x - px) ** 2 + (y - py) ** 2)
    distance = np.sqrt(distance_sq)
    return np.where(points_in_polygon(x, y, polygon), 0.0, distance)


def segment_intersects_polygon(start: Point, end: Point, polygon: PolygonObstacle) -> bool:
    """Return true when the open line of sight crosses the polygon."""
    px, py = start
    qx, qy = end
    if bool(points_in_polygon(np.asarray(qx), np.asarray(qy), polygon)):
        return True
    rx, ry = qx - px, qy - py
    vertices = polygon.vertices
    for a, b in zip(vertices, (*vertices[1:], vertices[0])):
        ax, ay = a
        sx, sy = b[0] - ax, b[1] - ay
        denominator = rx * sy - ry * sx
        if abs(denominator) < 1.0e-12:
            continue
        apx, apy = ax - px, ay - py
        t = (apx * sy - apy * sx) / denominator
        u = (apx * ry - apy * rx) / denominator
        if 1.0e-9 < t < 1.0 - 1.0e-9 and -1.0e-9 <= u <= 1.0 + 1.0e-9:
            return True
    return False


def line_of_sight_visible(
    observer: Point,
    target: Point,
    occluders: Sequence[PolygonObstacle],
) -> bool:
    return not any(segment_intersects_polygon(observer, target, item) for item in occluders)


@dataclass(frozen=True)
class PlanarScan:
    ranges: Array
    angle_min: float
    angle_increment: float
    range_min: float
    range_max: float
    sensor_x: float
    sensor_y: float
    sensor_yaw: float
    stamp: float = 0.0

    def __post_init__(self) -> None:
        ranges = np.asarray(self.ranges, dtype=np.float64)
        object.__setattr__(self, "ranges", ranges)
        if ranges.ndim != 1 or len(ranges) < 2:
            raise ValueError("scan ranges must be a one-dimensional array")
        if self.angle_increment <= 0.0:
            raise ValueError("angle_increment must be positive")
        if self.range_min < 0.0 or self.range_max <= self.range_min:
            raise ValueError("invalid scan range limits")

    @property
    def angles(self) -> Array:
        return self.angle_min + np.arange(len(self.ranges)) * self.angle_increment


class LidarShadowBuilder:
    """Build a DRIFT occlusion mask from first-return LiDAR visibility."""

    def __init__(
        self,
        *,
        maximum_shadow_range: float = 6.0,
        behind_surface_margin: float = 0.05,
        obstacle_match_margin: float = 0.08,
        angular_fill_bins: int = 2,
        require_known_occluder: bool = True,
    ) -> None:
        self.maximum_shadow_range = float(maximum_shadow_range)
        self.behind_surface_margin = float(behind_surface_margin)
        self.obstacle_match_margin = float(obstacle_match_margin)
        self.angular_fill_bins = int(angular_fill_bins)
        self.require_known_occluder = bool(require_known_occluder)
        if self.maximum_shadow_range <= 0.0:
            raise ValueError("maximum_shadow_range must be positive")

    def build(
        self,
        X: Array,
        Y: Array,
        road_mask: Array,
        scan: PlanarScan,
        occluders: Sequence[PolygonObstacle],
    ) -> Array:
        if X.shape != Y.shape or X.shape != road_mask.shape:
            raise ValueError("grid and road-mask shapes must match")
        ranges = np.asarray(scan.ranges, dtype=np.float64)
        valid = (
            np.isfinite(ranges)
            & (ranges >= scan.range_min)
            & (ranges <= min(scan.range_max, self.maximum_shadow_range))
        )
        angles_world = scan.sensor_yaw + scan.angles
        hit_x = scan.sensor_x + ranges * np.cos(angles_world)
        hit_y = scan.sensor_y + ranges * np.sin(angles_world)
        safe_hit_x = np.where(valid, hit_x, scan.sensor_x)
        safe_hit_y = np.where(valid, hit_y, scan.sensor_y)
        if self.require_known_occluder:
            confirmed = np.zeros_like(valid)
            for polygon in occluders:
                confirmed |= valid & (
                    point_to_polygon_distance(safe_hit_x, safe_hit_y, polygon)
                    <= self.obstacle_match_margin
                )
        else:
            # Deployed mode: the first return itself is the visibility boundary.
            # A surveyed polygon must not be required to discover unseen space.
            # The road mask below prevents returns outside the experiment's
            # drivable corridor from injecting Q_occ into irrelevant cells.
            confirmed = valid.copy()
        if self.angular_fill_bins > 0 and np.any(confirmed):
            window = 2 * self.angular_fill_bins + 1
            confirmed = (
                np.convolve(confirmed.astype(np.int8), np.ones(window, dtype=np.int8), mode="same")
                > 0
            )
            filtered_ranges = minimum_filter1d(
                np.where(valid, ranges, np.inf), size=window, mode="nearest"
            )
        else:
            filtered_ranges = ranges

        dx = X - scan.sensor_x
        dy = Y - scan.sensor_y
        cell_range = np.hypot(dx, dy)
        cell_angle = wrap_angle(np.arctan2(dy, dx) - scan.sensor_yaw)
        indices = np.rint((cell_angle - scan.angle_min) / scan.angle_increment).astype(int)
        in_fov = (indices >= 0) & (indices < len(ranges))
        safe_indices = np.clip(indices, 0, len(ranges) - 1)
        first_return = filtered_ranges[safe_indices]
        ray_confirmed = confirmed[safe_indices]
        shadow = (
            in_fov
            & ray_confirmed
            & np.isfinite(first_return)
            & (cell_range > first_return + self.behind_surface_margin)
            & (cell_range <= self.maximum_shadow_range)
            & (road_mask > 0.0)
        )
        return shadow.astype(np.float64)


def scan_line_of_sight_visible(
    scan: PlanarScan,
    target: Point,
    *,
    target_radius: float = 0.11,
    range_margin: float = 0.05,
) -> bool:
    """Gate a known target using the measured first-return visibility.

    This is used only when a second robot supplies ground-truth odometry.  Its
    state is withheld whenever a closer LiDAR return blocks every beam covering
    the target.  Targets outside the scan FOV/range fail closed.
    """
    dx = float(target[0]) - scan.sensor_x
    dy = float(target[1]) - scan.sensor_y
    target_range = float(np.hypot(dx, dy))
    if target_range <= 1.0e-9 or target_range > scan.range_max:
        return False
    relative_angle = float(wrap_angle(np.arctan2(dy, dx) - scan.sensor_yaw))
    minimum_angle = scan.angle_min
    maximum_angle = scan.angle_min + (len(scan.ranges) - 1) * scan.angle_increment
    if relative_angle < minimum_angle or relative_angle > maximum_angle:
        return False

    center = int(round((relative_angle - scan.angle_min) / scan.angle_increment))
    angular_radius = np.arctan2(max(0.0, float(target_radius)), target_range)
    half_bins = max(1, int(np.ceil(angular_radius / scan.angle_increment)))
    lo = max(0, center - half_bins)
    hi = min(len(scan.ranges), center + half_bins + 1)
    ranges = np.asarray(scan.ranges[lo:hi], dtype=np.float64)
    valid = np.isfinite(ranges) & (ranges >= scan.range_min) & (ranges <= scan.range_max)
    # At least one unobstructed target-covering ray is sufficient. No return is
    # free line of sight up to the sensor's declared range; a nearer return is
    # an occlusion boundary.
    visible_surface_range = target_range - max(0.0, float(target_radius)) - range_margin
    return bool(np.any(~valid) or np.any(ranges[valid] >= visible_surface_range))


def _ray_segment_distance(origin: Point, direction: Point, a: Point, b: Point) -> float:
    px, py = origin
    rx, ry = direction
    ax, ay = a
    sx, sy = b[0] - ax, b[1] - ay
    denominator = rx * sy - ry * sx
    if abs(denominator) < 1.0e-12:
        return np.inf
    apx, apy = ax - px, ay - py
    t = (apx * sy - apy * sx) / denominator
    u = (apx * ry - apy * rx) / denominator
    if t >= 0.0 and 0.0 <= u <= 1.0:
        return float(t)
    return np.inf


def simulate_polygon_scan(
    sensor_pose: Tuple[float, float, float],
    occluders: Sequence[PolygonObstacle],
    *,
    angle_min: float = -1.9198622,
    angle_max: float = 1.9198622,
    sample_count: int = 401,
    range_min: float = 0.01,
    range_max: float = 6.0,
    stamp: float = 0.0,
) -> PlanarScan:
    """Deterministic first-return scan used by Stage 1 and SIL tests."""
    if sample_count < 2:
        raise ValueError("sample_count must be at least two")
    sx, sy, yaw = sensor_pose
    angles = np.linspace(angle_min, angle_max, sample_count)
    ranges = np.full(sample_count, np.inf, dtype=np.float64)
    for index, angle in enumerate(angles):
        world_angle = yaw + float(angle)
        direction = (cos(world_angle), sin(world_angle))
        for polygon in occluders:
            vertices = polygon.vertices
            for a, b in zip(vertices, (*vertices[1:], vertices[0])):
                ranges[index] = min(
                    ranges[index], _ray_segment_distance((sx, sy), direction, a, b)
                )
        if not range_min <= ranges[index] <= range_max:
            ranges[index] = np.inf
    return PlanarScan(
        ranges=ranges,
        angle_min=angle_min,
        angle_increment=(angle_max - angle_min) / (sample_count - 1),
        range_min=range_min,
        range_max=range_max,
        sensor_x=sx,
        sensor_y=sy,
        sensor_yaw=yaw,
        stamp=stamp,
    )
