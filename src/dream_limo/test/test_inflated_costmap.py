import numpy as np

from dream_limo.core.free_goal import CostmapSnapshot
from dream_limo.core.inflated_costmap import validate_swept_trajectory


def _map(data=None):
    width = height = 40
    return CostmapSnapshot.from_sequence(
        frame_id="map",
        width=width,
        height=height,
        resolution=0.05,
        origin_x=-1.0,
        origin_y=-1.0,
        origin_yaw=0.0,
        data=[0] * (width * height) if data is None else data,
        source_stamp=9.9,
        receipt_stamp=10.0,
    )


def _check(states, costmap=None):
    return validate_swept_trajectory(
        states,
        _map() if costmap is None else costmap,
        expected_frame="map",
        robot_length=0.32,
        robot_width=0.22,
        footprint_padding=0.05,
        inflation_radius=0.30,
    )


def test_known_zero_cost_swept_trajectory_is_clear():
    states = np.asarray(
        [[0.0, 0.20], [0.0, 0.0], [0.1, 0.1], [0.0, 0.0]]
    )
    result = _check(states)
    assert result.safe
    assert result.reason == "TRAJECTORY_COSTMAP_CLEAR"


def test_positive_center_cost_and_unknown_footprint_fail_closed():
    states = np.asarray([[0.0], [0.0], [0.0], [0.0]])
    positive = [0] * 1600
    positive[20 * 40 + 20] = 50
    result = _check(states, _map(positive))
    assert not result.safe
    assert result.reason == "TRAJECTORY_CENTER_NOT_FREE"

    unknown = [0] * 1600
    # Inside the padded front of the footprint, but outside the centre cell.
    unknown[20 * 40 + 24] = -1
    result = _check(states, _map(unknown))
    assert not result.safe
    assert result.reason == "TRAJECTORY_FOOTPRINT_UNKNOWN"

    occupied = [0] * 1600
    occupied[20 * 40 + 24] = 100
    result = _check(states, _map(occupied))
    assert not result.safe
    assert result.reason == "TRAJECTORY_FOOTPRINT_OCCUPIED"


def test_interpolation_catches_obstacle_between_mpc_knots():
    states = np.asarray(
        [[-0.40, 0.40], [0.0, 0.0], [0.1, 0.1], [0.0, 0.0]]
    )
    data = [0] * 1600
    data[20 * 40 + 20] = 100
    result = _check(states, _map(data))
    assert not result.safe
    assert result.reason == "TRAJECTORY_FOOTPRINT_OCCUPIED"


def test_inflation_must_cover_padded_radius_and_grid_quantization():
    states = np.asarray([[0.0], [0.0], [0.0], [0.0]])
    result = validate_swept_trajectory(
        states,
        _map(),
        expected_frame="map",
        robot_length=0.32,
        robot_width=0.22,
        footprint_padding=0.05,
        inflation_radius=0.29,
    )
    assert not result.safe
    assert result.reason == "TRAJECTORY_INFLATION_CONTRACT_INVALID"


def test_combined_translation_and_rotation_use_corner_travel_bound():
    radius = np.hypot(0.16 + 0.05, 0.11 + 0.05)
    states = np.asarray(
        [[-0.02, 0.02], [0.0, 0.0], [0.1, 0.1], [0.0, 0.04 / radius]]
    )
    data = [0] * 1600
    # This cell is touched near one-quarter of the simultaneous translation
    # and turn, but not at either knot or the old max-motion midpoint sample.
    data[16 * 40 + 24] = 100
    result = _check(states, _map(data))
    assert not result.safe
    assert result.reason == "TRAJECTORY_FOOTPRINT_OCCUPIED"
