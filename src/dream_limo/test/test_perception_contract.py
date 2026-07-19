import math

import pytest

from dream_limo.core.types import parse_tracked_agents


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
