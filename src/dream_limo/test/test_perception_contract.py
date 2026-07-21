import math

import pytest

from dream_limo.core.types import Vehicle, parse_tracked_agents
from dream_limo.world_model_node import (
    evaluate_dynamic_source_fresh,
    evaluate_merger_adapter_status,
    filter_live_track_envelope,
    select_perception_tracks,
)


def payload(stamp=10.0):
    return {
        "id": "7",
        "class_label": "car",
        "position": {"x": 1.2, "y": 0.3},
        "velocity": {"x": 0.1, "y": 0.0},
        "radius": 0.2,
        "confidence": 0.9,
        "stamp": stamp,
        "age": 0.05,
        "source": "lidar",
        "motion_state": "dynamic",
    }


def test_bare_and_wrapped_sfg_payloads_and_empty_heartbeat():
    assert len(parse_tracked_agents([payload()], now=10.1)) == 1
    assert len(parse_tracked_agents({"agents": [payload()]}, now=10.1)) == 1
    assert parse_tracked_agents([], now=10.1) == []


def test_stale_and_nonfinite_tracks_are_not_accepted():
    assert parse_tracked_agents([payload(1.0)], now=10.0) == []
    bad = payload()
    bad["position"]["x"] = math.nan
    with pytest.raises(ValueError):
        parse_tracked_agents([bad], now=10.1)


def adapter_status(**updates):
    payload = {
        "ready": True,
        "input_fresh": True,
        "last_message_valid": True,
        "alignment_verified": True,
        "alignment_initialized": True,
        "output_frame": "odom",
        "output_child_frame": "merger/base_link",
        "reason": "READY",
    }
    payload.update(updates)
    return payload


def test_merger_adapter_contract_requires_ready_fresh_verified_alignment():
    ready, reason = evaluate_merger_adapter_status(
        adapter_status(),
        expected_output_frame="odom",
        expected_output_child_frame="merger/base_link",
    )
    assert ready
    assert reason == "READY"

    ready, _ = evaluate_merger_adapter_status(
        adapter_status(input_fresh=False, reason="STALE_INPUT"),
        expected_output_frame="odom",
        expected_output_child_frame="merger/base_link",
    )
    assert not ready


def test_merger_adapter_contract_rejects_truthy_strings_and_frame_mismatch():
    ready, _ = evaluate_merger_adapter_status(
        adapter_status(ready="true"),
        expected_output_frame="odom",
        expected_output_child_frame="merger/base_link",
    )
    assert not ready

    ready, reason = evaluate_merger_adapter_status(
        adapter_status(output_frame="map"),
        expected_output_frame="odom",
        expected_output_child_frame="merger/base_link",
    )
    assert not ready
    assert reason == "OUTPUT_FRAME_MISMATCH"


def test_aligned_merger_odom_can_replace_sfg_track_heartbeat():
    assert evaluate_dynamic_source_fresh(
        perception_tracks_fresh=False,
        merger_odom_required=True,
        merger_inputs_ready=True,
    )


def test_fresh_sfg_tracks_cannot_mask_required_adapter_failure():
    assert not evaluate_dynamic_source_fresh(
        perception_tracks_fresh=True,
        merger_odom_required=True,
        merger_inputs_ready=False,
    )
    assert evaluate_dynamic_source_fresh(
        perception_tracks_fresh=True,
        merger_odom_required=False,
        merger_inputs_ready=False,
    )


def test_aligned_merger_mode_ignores_all_perception_tracks():
    agents = parse_tracked_agents([payload()], now=10.1)
    assert len(agents) == 1
    assert select_perception_tracks(
        agents,
        perception_tracks_fresh=True,
        merger_odom_required=True,
    ) == []
    assert select_perception_tracks(
        agents,
        perception_tracks_fresh=True,
        merger_odom_required=False,
    ) == agents


def test_live_track_envelope_keeps_three_lanes_and_rejects_wall_fragments():
    lanes = [
        Vehicle(f"lane_{index}", 2.0, y, vx=0.4, length=0.22, width=0.22)
        for index, y in enumerate((0.45, 0.0, -0.45))
    ]
    wall_fragments = [
        Vehicle("above_road", 2.5, 1.16, vx=0.2, length=0.22, width=0.22),
        Vehicle("below_road", 2.0, -1.2, vx=0.2, length=0.22, width=0.22),
        Vehicle("implausibly_fast", 2.0, 0.0, vx=6.0, length=0.22, width=0.22),
    ]

    accepted, rejected = filter_live_track_envelope(
        [*lanes, *wall_fragments],
        grid_x_min=0.0,
        grid_x_max=6.0,
        road_y_min=-0.70,
        road_y_max=0.70,
        maximum_speed=1.0,
    )

    assert [vehicle.vehicle_id for vehicle in accepted] == [
        "lane_0",
        "lane_1",
        "lane_2",
    ]
    assert rejected == 3


def test_live_track_envelope_requires_complete_rotated_footprint_inside():
    rotated_edge = Vehicle(
        "rotated_edge",
        2.0,
        0.55,
        vx=0.1,
        heading=math.pi / 4.0,
        length=0.40,
        width=0.20,
    )
    accepted, rejected = filter_live_track_envelope(
        [rotated_edge],
        grid_x_min=0.0,
        grid_x_max=6.0,
        road_y_min=-0.70,
        road_y_max=0.70,
        maximum_speed=1.0,
    )
    assert accepted == []
    assert rejected == 1
