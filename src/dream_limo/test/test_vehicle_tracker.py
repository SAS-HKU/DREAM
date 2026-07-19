import pytest

from dream_limo.core.types import parse_tracked_agents
from dream_limo.core.vehicle_tracker import (
    ClusterMeasurement,
    MergerVehicleTracker,
    parse_cluster_payload,
    track_to_agent_payload,
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
    for now, x in ((0.0, 0.0), (0.2, 0.06), (0.4, 0.13), (0.6, 0.19)):
        tracker.update([cluster(x)], now)

    tracks = tracker.publishable_tracks(0.6)
    assert len(tracks) == 1
    assert tracks[0].dynamic_confirmed
    assert tracks[0].vx > 0.10

    agent = track_to_agent_payload(tracks[0], 0.6)
    assert agent["class_label"] == "car"
    assert agent["source"] == "dream_lidar_vehicle_tracker"
    assert agent["motion_state"] == "dynamic"
    assert "pedestrian" not in str(agent)
    parsed = parse_tracked_agents([agent], now=0.6)
    assert len(parsed) == 1
    assert parsed[0].class_label == "car"


def test_confirmed_track_coasts_briefly_then_expires():
    tracker = MergerVehicleTracker(
        coast_timeout_sec=0.5,
        stale_remove_sec=1.0,
        motion_window_sec=0.4,
        motion_min_displacement_m=0.08,
    )
    for now, x in ((0.0, 0.0), (0.2, 0.06), (0.4, 0.13)):
        tracker.update([cluster(x)], now)
    assert len(tracker.publishable_tracks(0.8)) == 1
    assert tracker.publishable_tracks(0.91) == []
    tracker.update([], 1.41)
    assert tracker.tracks == []


def test_vehicle_radius_covers_nominal_body_and_observed_width():
    tracker = MergerVehicleTracker(motion_window_sec=0.2)
    for now, x in ((0.0, 0.0), (0.1, 0.08), (0.2, 0.17)):
        tracker.update([cluster(x, width=0.40)], now)
    track = tracker.publishable_tracks(0.2)[0]
    payload_out = track_to_agent_payload(track, 0.2)
    assert payload_out["radius"] >= 0.24
