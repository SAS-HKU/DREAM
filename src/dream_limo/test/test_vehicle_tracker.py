import pytest

from dream_limo.core.types import parse_tracked_agents
from dream_limo.core.vehicle_tracker import (
    ClusterMeasurement,
    MergerVehicleTracker,
    parse_cluster_payload,
    track_to_agent_payload,
    validate_cluster_source_stamp,
)


def cluster(x, y=0.0, width=0.24, cluster_id="scan_0"):
    return ClusterMeasurement(
        cluster_id=cluster_id,
        x=float(x),
        y=float(y),
        width=float(width),
        points=8,
        range_m=1.5,
    )


def payload():
    return {
        "stamp": 12.5,
        "frame_id": "odom",
        "clusters": [
            {
                "id": "vehicle",
                "x": 1.2,
                "y": -0.4,
                "range": 1.4,
                "width": 0.24,
                "points": 9,
            },
            {
                "id": "long_occluder",
                "x": 2.0,
                "y": 0.0,
                "range": 2.0,
                "width": 1.1,
                "points": 45,
            },
            {
                "id": "noise",
                "x": 0.4,
                "y": 0.1,
                "range": 0.4,
                "width": 0.03,
                "points": 2,
            },
        ],
    }


def test_neutral_cluster_parser_keeps_limo_sized_geometry_only():
    frame = parse_cluster_payload(payload())
    assert frame.frame_id == "odom"
    assert frame.raw_count == 3
    assert frame.rejected_count == 2
    assert [item.cluster_id for item in frame.clusters] == ["vehicle"]


def test_neutral_cluster_parser_rejects_frame_mismatch_and_bad_contract():
    bad_frame = payload()
    bad_frame["frame_id"] = "laser_link"
    with pytest.raises(ValueError, match="expected"):
        parse_cluster_payload(bad_frame)
    with pytest.raises(ValueError, match="clusters list"):
        parse_cluster_payload({"stamp": 0.0, "frame_id": "odom"})


def test_static_occluder_jitter_never_becomes_merger():
    tracker = MergerVehicleTracker(
        motion_window_sec=0.4,
        motion_enter_speed_mps=0.10,
        motion_min_displacement_m=0.08,
    )
    for step, x in enumerate((1.00, 1.01, 0.99, 1.01, 1.00, 1.01)):
        now = 0.2 * step
        tracker.update([cluster(x, width=0.30)], now)
        assert tracker.publishable_tracks(now) == []
    assert tracker.fresh_candidate_count(1.0) == 1


def test_motion_confirmed_vehicle_publishes_existing_agent_schema():
    tracker = MergerVehicleTracker(
        motion_window_sec=0.4,
        motion_enter_speed_mps=0.10,
        motion_min_displacement_m=0.08,
        minimum_track_hits=3,
    )
    for now, x in (
        (0.0, 0.0),
        (0.2, 0.06),
        (0.4, 0.13),
        (0.6, 0.19),
        (0.8, 0.25),
    ):
        tracker.update([cluster(x)], now)

    tracks = tracker.publishable_tracks(0.8)
    assert len(tracks) == 1
    assert tracks[0].dynamic_confirmed
    assert tracks[0].vx > 0.10

    agent = track_to_agent_payload(tracks[0], 0.8)
    assert agent["class_label"] == "car"
    assert agent["source"] == "dream_lidar_vehicle_tracker"
    assert agent["motion_state"] == "dynamic"
    assert "pedestrian" not in str(agent)
    parsed = parse_tracked_agents([agent], now=0.8)
    assert len(parsed) == 1
    assert parsed[0].class_label == "car"


def test_confirmed_track_coasts_briefly_then_expires():
    tracker = MergerVehicleTracker(
        coast_timeout_sec=0.5,
        stale_remove_sec=1.0,
        motion_window_sec=0.4,
        motion_min_displacement_m=0.08,
    )
    for now, x in (
        (0.0, 0.0),
        (0.2, 0.06),
        (0.4, 0.13),
        (0.6, 0.19),
        (0.8, 0.25),
    ):
        tracker.update([cluster(x)], now)
    assert len(tracker.publishable_tracks(1.2)) == 1
    assert tracker.publishable_tracks(1.31) == []
    tracker.update([], 1.81)
    assert tracker.tracks == []


def test_vehicle_radius_covers_nominal_body_and_observed_width():
    tracker = MergerVehicleTracker(motion_window_sec=0.2)
    for now, x in (
        (0.0, 0.0),
        (0.1, 0.05),
        (0.2, 0.11),
        (0.3, 0.17),
        (0.4, 0.23),
    ):
        tracker.update([cluster(x, width=0.40)], now)
    track = tracker.publishable_tracks(0.4)[0]
    payload_out = track_to_agent_payload(track, 0.4)
    assert payload_out["radius"] >= 0.24


def test_observed_wall_centroid_oscillation_never_confirms_motion():
    """Regression for the stationary wall sequence observed on the LIMO."""

    tracker = MergerVehicleTracker(
        motion_window_sec=0.50,
        motion_enter_speed_mps=0.10,
        motion_min_displacement_m=0.08,
    )
    wall_sequence = (
        (0.0, -0.55),
        (0.2, -0.48),
        (0.4, -0.35),
        (0.6, -0.35),
        (0.8, -0.45),
        (1.0, -0.55),
        (1.2, -0.55),
        (1.4, -0.45),
        (1.6, -0.35),
        (1.8, -0.35),
    )
    for now, y in wall_sequence:
        tracker.update([cluster(1.0, y=y, width=0.30)], now)
        assert tracker.publishable_tracks(now) == []


def test_monotonic_merger_needs_two_consistent_windows_then_confirms():
    tracker = MergerVehicleTracker(
        motion_window_sec=0.50,
        motion_enter_speed_mps=0.10,
        motion_min_displacement_m=0.08,
    )
    merger_sequence = (
        (0.0, 0.00),
        (0.2, 0.05),
        (0.4, 0.10),
        (0.6, 0.15),
        (0.8, 0.20),
        (1.0, 0.25),
        (1.2, 0.30),
    )
    for now, y in merger_sequence[:-1]:
        tracker.update([cluster(1.0, y=y)], now)
        assert tracker.publishable_tracks(now) == []

    now, y = merger_sequence[-1]
    tracker.update([cluster(1.0, y=y)], now)
    tracks = tracker.publishable_tracks(now)
    assert len(tracks) == 1
    assert tracks[0].consistent_motion_windows == 2
    assert 0.10 < tracks[0].speed <= 0.60


def test_direction_reversal_immediately_deconfirms_track():
    tracker = MergerVehicleTracker(motion_window_sec=0.50)
    for now, y in (
        (0.0, 0.00),
        (0.2, 0.05),
        (0.4, 0.10),
        (0.6, 0.15),
        (0.8, 0.20),
        (1.0, 0.25),
        (1.2, 0.30),
    ):
        tracker.update([cluster(1.0, y=y)], now)
    assert len(tracker.publishable_tracks(1.2)) == 1

    for now, y in ((1.4, 0.25), (1.6, 0.20), (1.8, 0.15)):
        tracker.update([cluster(1.0, y=y)], now)
    assert tracker.publishable_tracks(1.8) == []


def test_association_rejects_implausible_innovation_and_width_change():
    innovation_tracker = MergerVehicleTracker()
    innovation_tracker.update([cluster(0.0)], 0.0)
    innovation_tracker.update([cluster(0.25)], 0.1)
    assert len(innovation_tracker.tracks) == 2

    width_tracker = MergerVehicleTracker()
    width_tracker.update([cluster(0.0, width=0.20)], 0.0)
    width_tracker.update([cluster(0.01, width=0.40)], 0.1)
    assert len(width_tracker.tracks) == 2


def test_cluster_source_stamp_must_be_fresh_and_strictly_monotonic():
    assert validate_cluster_source_stamp(
        9.8,
        receipt_stamp=10.0,
        previous_source_stamp=9.7,
        maximum_age=0.5,
        future_tolerance=0.05,
    ) == pytest.approx(0.2)

    for source_stamp, previous, error in (
        (9.8, 9.8, "strictly monotonic"),
        (9.7, 9.8, "strictly monotonic"),
        (9.4, None, "stale"),
        (10.06, None, "future"),
    ):
        with pytest.raises(ValueError, match=error):
            validate_cluster_source_stamp(
                source_stamp,
                receipt_stamp=10.0,
                previous_source_stamp=previous,
                maximum_age=0.5,
                future_tolerance=0.05,
            )
