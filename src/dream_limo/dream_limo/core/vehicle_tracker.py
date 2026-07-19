"""ROS-independent tracking of a LIMO-sized merger from neutral LiDAR clusters.

The SFG LiDAR front end deliberately publishes geometry-only cluster JSON.
This module consumes that public contract without importing :mod:`sfg_nav`.
It rejects clusters outside the configured vehicle envelope and only exposes
tracks after motion has been observed over a time window. Consequently, a static
occluder does not become a perceived merger merely because it is visible.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ClusterMeasurement:
    """One geometry-only LiDAR cluster in a fixed world frame."""

    cluster_id: str
    x: float
    y: float
    width: float
    points: int
    range_m: float


@dataclass(frozen=True)
class ClusterFrame:
    """Validated cluster payload and filtering diagnostics."""

    stamp: float
    frame_id: str
    clusters: tuple[ClusterMeasurement, ...]
    raw_count: int
    rejected_count: int


@dataclass
class VehicleTrack:
    """Internal constant-velocity track state."""

    track_id: int
    x: float
    y: float
    vx: float
    vy: float
    width: float
    first_update_sec: float
    last_update_sec: float
    motion_anchor_x: float
    motion_anchor_y: float
    motion_anchor_sec: float
    hits: int = 1
    dynamic_confirmed: bool = False
    last_dynamic_sec: float = -math.inf
    motion_window_speed: float = 0.0

    def age(self, now_sec: float) -> float:
        return max(0.0, float(now_sec) - self.last_update_sec)

    def predicted_position(self, now_sec: float) -> tuple[float, float]:
        dt = self.age(now_sec)
        return self.x + self.vx * dt, self.y + self.vy * dt

    @property
    def speed(self) -> float:
        return math.hypot(self.vx, self.vy)


def parse_cluster_payload(
    payload: Any,
    *,
    expected_frame: str = "odom",
    minimum_width_m: float = 0.08,
    maximum_width_m: float = 0.50,
    minimum_points: int = 3,
    minimum_range_m: float = 0.25,
    maximum_range_m: float = 6.0,
) -> ClusterFrame:
    """Validate and size-filter the public ``/sfg/lidar_clusters`` payload.

    Invalid individual clusters are rejected instead of invalidating a complete
    scan.  A malformed top-level contract or unexpected coordinate frame is an
    error because silently mixing frames would create unsafe phantom motion.
    """

    if not isinstance(payload, Mapping):
        raise ValueError("LiDAR cluster payload must be an object")
    raw_clusters = payload.get("clusters")
    if not isinstance(raw_clusters, list):
        raise ValueError("LiDAR cluster payload must contain a clusters list")

    stamp = _finite_float(payload.get("stamp", 0.0), "stamp")
    frame_id = str(payload.get("frame_id", ""))
    if not frame_id:
        raise ValueError("LiDAR cluster payload has no frame_id")
    if expected_frame and frame_id != expected_frame:
        raise ValueError(
            f"LiDAR clusters are in {frame_id!r}, expected {expected_frame!r}"
        )
    if minimum_width_m <= 0.0 or maximum_width_m < minimum_width_m:
        raise ValueError("invalid cluster width limits")
    if minimum_range_m < 0.0 or maximum_range_m <= minimum_range_m:
        raise ValueError("invalid cluster range limits")

    accepted: list[ClusterMeasurement] = []
    rejected = 0
    for index, raw in enumerate(raw_clusters):
        try:
            if not isinstance(raw, Mapping):
                raise ValueError("cluster must be an object")
            x = _finite_float(raw.get("x"), "x")
            y = _finite_float(raw.get("y"), "y")
            width = _finite_float(raw.get("width"), "width")
            range_m = _finite_float(raw.get("range"), "range")
            points = int(raw.get("points", 0))
            if width < minimum_width_m or width > maximum_width_m:
                raise ValueError("cluster width outside vehicle envelope")
            if points < minimum_points:
                raise ValueError("too few cluster points")
            if range_m < minimum_range_m or range_m > maximum_range_m:
                raise ValueError("cluster range outside tracking envelope")
            accepted.append(
                ClusterMeasurement(
                    cluster_id=str(raw.get("id", f"cluster_{index}")),
                    x=x,
                    y=y,
                    width=width,
                    points=points,
                    range_m=range_m,
                )
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            rejected += 1

    return ClusterFrame(
        stamp=stamp,
        frame_id=frame_id,
        clusters=tuple(accepted),
        raw_count=len(raw_clusters),
        rejected_count=rejected,
    )


class MergerVehicleTracker:
    """Track clusters and require displacement-based motion confirmation."""

    def __init__(
        self,
        *,
        association_distance_m: float = 0.45,
        velocity_alpha: float = 0.45,
        position_alpha: float = 0.70,
        coast_timeout_sec: float = 0.50,
        stale_remove_sec: float = 1.00,
        motion_window_sec: float = 0.50,
        motion_enter_speed_mps: float = 0.10,
        motion_exit_speed_mps: float = 0.04,
        motion_min_displacement_m: float = 0.08,
        motion_hold_sec: float = 0.80,
        minimum_track_hits: int = 3,
    ) -> None:
        if association_distance_m <= 0.0:
            raise ValueError("association distance must be positive")
        if not 0.0 < velocity_alpha <= 1.0:
            raise ValueError("velocity alpha must lie in (0, 1]")
        if not 0.0 < position_alpha <= 1.0:
            raise ValueError("position alpha must lie in (0, 1]")
        if coast_timeout_sec <= 0.0 or stale_remove_sec < coast_timeout_sec:
            raise ValueError(
                "stale timeout must be at least the coast timeout"
            )
        if motion_window_sec <= 0.0 or motion_hold_sec < 0.0:
            raise ValueError("motion timing parameters are invalid")
        if motion_enter_speed_mps < 0.0 or motion_exit_speed_mps < 0.0:
            raise ValueError("motion speed thresholds cannot be negative")
        if motion_min_displacement_m < 0.0:
            raise ValueError("motion displacement cannot be negative")
        if minimum_track_hits < 2:
            raise ValueError("at least two track hits are required")

        self.association_distance_m = float(association_distance_m)
        self.velocity_alpha = float(velocity_alpha)
        self.position_alpha = float(position_alpha)
        self.coast_timeout_sec = float(coast_timeout_sec)
        self.stale_remove_sec = float(stale_remove_sec)
        self.motion_window_sec = float(motion_window_sec)
        self.motion_enter_speed_mps = float(motion_enter_speed_mps)
        self.motion_exit_speed_mps = float(motion_exit_speed_mps)
        self.motion_min_displacement_m = float(
            motion_min_displacement_m
        )
        self.motion_hold_sec = float(motion_hold_sec)
        self.minimum_track_hits = int(minimum_track_hits)
        self.tracks: list[VehicleTrack] = []
        self.next_track_id = 1

    def update(
        self, measurements: Sequence[ClusterMeasurement], now_sec: float
    ) -> None:
        """Associate one cluster frame and update all track states."""

        now_sec = _finite_float(now_sec, "now_sec")
        self._remove_stale(now_sec)

        # Global greedy association is deterministic and avoids assigning two
        # measurements to one track when several static surfaces are nearby.
        candidate_pairs: list[tuple[float, int, int]] = []
        for track_index, track in enumerate(self.tracks):
            pred_x, pred_y = track.predicted_position(now_sec)
            for measurement_index, measurement in enumerate(measurements):
                distance = math.hypot(
                    measurement.x - pred_x, measurement.y - pred_y
                )
                if distance <= self.association_distance_m:
                    candidate_pairs.append(
                        (distance, track_index, measurement_index)
                    )
        candidate_pairs.sort(key=lambda item: (item[0], item[1], item[2]))

        used_tracks: set[int] = set()
        used_measurements: set[int] = set()
        for _, track_index, measurement_index in candidate_pairs:
            if (
                track_index in used_tracks
                or measurement_index in used_measurements
            ):
                continue
            self._update_track(
                self.tracks[track_index],
                measurements[measurement_index],
                now_sec,
            )
            used_tracks.add(track_index)
            used_measurements.add(measurement_index)

        for measurement_index, measurement in enumerate(measurements):
            if measurement_index not in used_measurements:
                self._start_track(measurement, now_sec)
        self._remove_stale(now_sec)

    def publishable_tracks(self, now_sec: float) -> list[VehicleTrack]:
        """Return fresh tracks with enough hits and confirmed motion."""

        now_sec = _finite_float(now_sec, "now_sec")
        return [
            track
            for track in self.tracks
            if track.age(now_sec) <= self.coast_timeout_sec
            and track.hits >= self.minimum_track_hits
            and track.dynamic_confirmed
        ]

    def fresh_candidate_count(self, now_sec: float) -> int:
        return sum(
            track.age(now_sec) <= self.coast_timeout_sec
            for track in self.tracks
        )

    def _start_track(
        self, measurement: ClusterMeasurement, now_sec: float
    ) -> None:
        self.tracks.append(
            VehicleTrack(
                track_id=self.next_track_id,
                x=measurement.x,
                y=measurement.y,
                vx=0.0,
                vy=0.0,
                width=measurement.width,
                first_update_sec=now_sec,
                last_update_sec=now_sec,
                motion_anchor_x=measurement.x,
                motion_anchor_y=measurement.y,
                motion_anchor_sec=now_sec,
            )
        )
        self.next_track_id += 1

    def _update_track(
        self,
        track: VehicleTrack,
        measurement: ClusterMeasurement,
        now_sec: float,
    ) -> None:
        dt = max(1.0e-3, now_sec - track.last_update_sec)
        old_x, old_y = track.x, track.y
        alpha_p = self.position_alpha
        new_x = (1.0 - alpha_p) * old_x + alpha_p * measurement.x
        new_y = (1.0 - alpha_p) * old_y + alpha_p * measurement.y
        measured_vx = (new_x - old_x) / dt
        measured_vy = (new_y - old_y) / dt
        alpha_v = self.velocity_alpha
        track.vx = (1.0 - alpha_v) * track.vx + alpha_v * measured_vx
        track.vy = (1.0 - alpha_v) * track.vy + alpha_v * measured_vy
        track.x = new_x
        track.y = new_y
        track.width = (
            (1.0 - alpha_p) * track.width + alpha_p * measurement.width
        )
        track.last_update_sec = now_sec
        track.hits += 1
        self._update_motion_state(track, now_sec)

    def _update_motion_state(
        self, track: VehicleTrack, now_sec: float
    ) -> None:
        elapsed = now_sec - track.motion_anchor_sec
        if elapsed < self.motion_window_sec:
            return
        dx = track.x - track.motion_anchor_x
        dy = track.y - track.motion_anchor_y
        displacement = math.hypot(dx, dy)
        track.motion_window_speed = displacement / max(elapsed, 1.0e-6)
        speed_threshold = (
            self.motion_exit_speed_mps
            if track.dynamic_confirmed
            else self.motion_enter_speed_mps
        )
        displacement_threshold = (
            0.5 * self.motion_min_displacement_m
            if track.dynamic_confirmed
            else self.motion_min_displacement_m
        )
        moving = (
            track.motion_window_speed >= speed_threshold
            and displacement >= displacement_threshold
        )
        if moving:
            track.dynamic_confirmed = True
            track.last_dynamic_sec = now_sec
            # Window velocity is much less sensitive to cluster-edge jitter
            # than scan-to-scan differentiation.
            track.vx = dx / max(elapsed, 1.0e-6)
            track.vy = dy / max(elapsed, 1.0e-6)
        elif (
            track.dynamic_confirmed
            and now_sec - track.last_dynamic_sec > self.motion_hold_sec
        ):
            track.dynamic_confirmed = False
            track.vx = 0.0
            track.vy = 0.0

        track.motion_anchor_x = track.x
        track.motion_anchor_y = track.y
        track.motion_anchor_sec = now_sec

    def _remove_stale(self, now_sec: float) -> None:
        self.tracks = [
            track
            for track in self.tracks
            if track.age(now_sec) <= self.stale_remove_sec
        ]


def track_to_agent_payload(
    track: VehicleTrack,
    now_sec: float,
    *,
    class_label: str = "car",
    nominal_radius_m: float = 0.18,
    radius_padding_m: float = 0.04,
) -> dict[str, Any]:
    """Convert a confirmed track to DREAM's existing tracked-agent schema."""

    if not class_label:
        raise ValueError("class label cannot be empty")
    if nominal_radius_m <= 0.0 or radius_padding_m < 0.0:
        raise ValueError("invalid vehicle radius parameters")
    x, y = track.predicted_position(now_sec)
    radius = max(nominal_radius_m, 0.5 * track.width + radius_padding_m)
    confidence = min(0.95, 0.55 + 0.08 * track.hits)
    return {
        "id": f"dream_vehicle_{track.track_id}",
        "class_label": class_label,
        "position": {"x": float(x), "y": float(y)},
        "velocity": {"x": float(track.vx), "y": float(track.vy)},
        "radius": float(radius),
        "confidence": float(confidence),
        "stamp": float(now_sec),
        "age": float(track.age(now_sec)),
        "source": "dream_lidar_vehicle_tracker",
        "motion_state": "dynamic",
        "motion_window_speed": float(track.motion_window_speed),
    }


def _finite_float(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result
