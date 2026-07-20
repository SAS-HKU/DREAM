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
    consistent_motion_windows: int = 0
    last_motion_dx: float = 0.0
    last_motion_dy: float = 0.0

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
        association_noise_margin_m: float = 0.06,
        maximum_vehicle_speed_mps: float = 0.60,
        maximum_width_change_m: float = 0.12,
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
        minimum_consistent_motion_windows: int = 2,
        minimum_direction_cosine: float = 0.50,
    ) -> None:
        if association_distance_m <= 0.0:
            raise ValueError("association distance must be positive")
        if (
            association_noise_margin_m < 0.0
            or association_noise_margin_m > association_distance_m
        ):
            raise ValueError(
                "association noise margin must lie in [0, association distance]"
            )
        if maximum_vehicle_speed_mps <= 0.0:
            raise ValueError("maximum vehicle speed must be positive")
        if maximum_width_change_m < 0.0:
            raise ValueError("maximum width change cannot be negative")
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
        if minimum_consistent_motion_windows < 2:
            raise ValueError(
                "at least two direction-consistent motion windows are required"
            )
        if not -1.0 <= minimum_direction_cosine <= 1.0:
            raise ValueError("minimum direction cosine must lie in [-1, 1]")

        self.association_distance_m = float(association_distance_m)
        self.association_noise_margin_m = float(association_noise_margin_m)
        self.maximum_vehicle_speed_mps = float(maximum_vehicle_speed_mps)
        self.maximum_width_change_m = float(maximum_width_change_m)
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
        self.minimum_consistent_motion_windows = int(
            minimum_consistent_motion_windows
        )
        self.minimum_direction_cosine = float(minimum_direction_cosine)
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
            update_age = max(0.0, now_sec - track.last_update_sec)
            plausible_innovation = min(
                self.association_distance_m,
                self.association_noise_margin_m
                + self.maximum_vehicle_speed_mps * update_age,
            )
            for measurement_index, measurement in enumerate(measurements):
                if (
                    abs(measurement.width - track.width)
                    > self.maximum_width_change_m
                ):
                    continue
                distance = math.hypot(
                    measurement.x - pred_x, measurement.y - pred_y
                )
                if distance <= plausible_innovation:
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
        track.vx, track.vy = self._bounded_velocity(track.vx, track.vy)
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
            and track.motion_window_speed <= self.maximum_vehicle_speed_mps
            and displacement >= displacement_threshold
        )
        if moving:
            previous_norm = math.hypot(
                track.last_motion_dx, track.last_motion_dy
            )
            direction_cosine = None
            if previous_norm > 1.0e-9 and displacement > 1.0e-9:
                direction_cosine = (
                    dx * track.last_motion_dx + dy * track.last_motion_dy
                ) / (displacement * previous_norm)

            if direction_cosine is None:
                track.consistent_motion_windows = 1
            elif direction_cosine >= self.minimum_direction_cosine:
                track.consistent_motion_windows += 1
            else:
                # A centroid that walks back and forth along a static wall must
                # not remain a dynamic vehicle. A real reversing object can be
                # confirmed again after two windows in its new direction.
                if direction_cosine < 0.0:
                    track.dynamic_confirmed = False
                    track.last_dynamic_sec = -math.inf
                track.consistent_motion_windows = 1

            track.last_motion_dx = dx
            track.last_motion_dy = dy
            window_vx, window_vy = self._bounded_velocity(
                dx / max(elapsed, 1.0e-6),
                dy / max(elapsed, 1.0e-6),
            )
            track.vx = window_vx
            track.vy = window_vy
            if (
                track.consistent_motion_windows
                >= self.minimum_consistent_motion_windows
            ):
                track.dynamic_confirmed = True
                track.last_dynamic_sec = now_sec
        elif (
            track.dynamic_confirmed
            and now_sec - track.last_dynamic_sec > self.motion_hold_sec
        ):
            track.dynamic_confirmed = False
            track.vx = 0.0
            track.vy = 0.0
            track.consistent_motion_windows = 0
            track.last_motion_dx = 0.0
            track.last_motion_dy = 0.0
        elif not track.dynamic_confirmed:
            track.consistent_motion_windows = 0
            track.last_motion_dx = 0.0
            track.last_motion_dy = 0.0

        track.motion_anchor_x = track.x
        track.motion_anchor_y = track.y
        track.motion_anchor_sec = now_sec

    def _bounded_velocity(self, vx: float, vy: float) -> tuple[float, float]:
        """Limit an internal estimate to the configured physical speed."""

        speed = math.hypot(vx, vy)
        if speed <= self.maximum_vehicle_speed_mps or speed <= 1.0e-12:
            return float(vx), float(vy)
        scale = self.maximum_vehicle_speed_mps / speed
        return float(vx * scale), float(vy * scale)

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


def validate_cluster_source_stamp(
    source_stamp: float,
    *,
    receipt_stamp: float,
    previous_source_stamp: float | None,
    maximum_age: float,
    future_tolerance: float,
) -> float:
    """Validate one cluster frame's source time and return its receipt age.

    Tracking uses sensor time, not callback receipt time. Strict monotonicity
    prevents duplicate or reordered scan frames from manufacturing velocity.
    """

    source = _finite_float(source_stamp, "source_stamp")
    receipt = _finite_float(receipt_stamp, "receipt_stamp")
    maximum = _finite_float(maximum_age, "maximum_age")
    future = _finite_float(future_tolerance, "future_tolerance")
    if source <= 0.0:
        raise ValueError("cluster source stamp must be positive")
    if maximum <= 0.0 or future < 0.0:
        raise ValueError("cluster source timing limits are invalid")
    age = receipt - source
    if age < -future:
        raise ValueError("cluster source stamp is in the future")
    if age > maximum:
        raise ValueError("cluster source stamp is stale")
    if previous_source_stamp is not None:
        previous = _finite_float(previous_source_stamp, "previous_source_stamp")
        if source <= previous:
            raise ValueError("cluster source stamp is not strictly monotonic")
    return max(0.0, age)
