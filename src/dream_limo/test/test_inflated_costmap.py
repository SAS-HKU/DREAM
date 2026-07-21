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


def _check(
    states,
    costmap=None,
    *,
    allow_initial_inflated_center_prefix=False,
):
    return validate_swept_trajectory(
        states,
        _map() if costmap is None else costmap,
        expected_frame="map",
        robot_length=0.32,
        robot_width=0.22,
        footprint_padding=0.05,
        inflation_radius=0.30,
        allow_initial_inflated_center_prefix=(
            allow_initial_inflated_center_prefix
        ),
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


def test_initial_soft_inflation_prefix_can_be_explicitly_allowed():
    states = np.asarray(
        [[-0.30, 0.30], [0.0, 0.0], [0.1, 0.1], [0.0, 0.0]]
    )
    data = [0] * 1600
    # Interpolated centres begin in soft inflation, then enter zero cost.
    for cell_x in range(13, 19):
        data[20 * 40 + cell_x] = 50

    strict = _check(states, _map(data))
    assert not strict.safe
    assert strict.reason == "TRAJECTORY_CENTER_NOT_FREE"

    escape = _check(
        states,
        _map(data),
        allow_initial_inflated_center_prefix=True,
    )
    assert escape.safe
    assert escape.reason == "TRAJECTORY_COSTMAP_CLEAR"


def test_constant_or_decreasing_soft_prefix_need_not_reach_zero():
    states = np.asarray(
        [[-0.30, 0.30], [0.0, 0.0], [0.1, 0.1], [0.0, 0.0]]
    )
    data = [50] * 1600
    # The centre crosses into a lower-cost region but remains in soft
    # inflation through the end of this finite horizon.
    for cell_y in range(40):
        for cell_x in range(20, 40):
            data[cell_y * 40 + cell_x] = 20

    result = _check(
        states,
        _map(data),
        allow_initial_inflated_center_prefix=True,
    )
    assert result.safe
    assert result.reason == "TRAJECTORY_COSTMAP_CLEAR"


def test_soft_inflation_increase_before_zero_fails_closed():
    states = np.asarray(
        [[-0.30, 0.30], [0.0, 0.0], [0.1, 0.1], [0.0, 0.0]]
    )
    data = [0] * 1600
    for cell_x in range(13, 16):
        data[20 * 40 + cell_x] = 20
    for cell_x in range(16, 19):
        data[20 * 40 + cell_x] = 80

    result = _check(
        states,
        _map(data),
        allow_initial_inflated_center_prefix=True,
    )
    assert not result.safe
    assert result.reason == "TRAJECTORY_CENTER_INFLATION_INCREASE"
    assert result.cell_value == 80


def test_inscribed_cost_99_is_never_a_soft_recovery_cell():
    states = np.asarray([[0.0], [0.0], [0.0], [0.0]])
    data = [0] * 1600
    data[20 * 40 + 20] = 99

    result = _check(
        states,
        _map(data),
        allow_initial_inflated_center_prefix=True,
    )
    assert not result.safe
    assert result.reason == "TRAJECTORY_CENTER_NOT_FREE"
    assert result.cell_value == 99


def test_zero_grid_gap_does_not_end_bounded_initial_recovery():
    states = np.asarray(
        [[-0.30, 0.30], [0.0, 0.0], [0.1, 0.1], [0.0, 0.0]]
    )
    data = [0] * 1600
    for cell_x in range(13, 16):
        data[20 * 40 + cell_x] = 50
    for cell_x in range(24, 27):
        data[20 * 40 + cell_x] = 20

    result = _check(
        states,
        _map(data),
        allow_initial_inflated_center_prefix=True,
    )
    assert result.safe
    assert result.reason == "TRAJECTORY_COSTMAP_CLEAR"


def test_horizon_starting_at_zero_cannot_enter_soft_inflation():
    states = np.asarray(
        [[-0.30, 0.30], [0.0, 0.0], [0.1, 0.1], [0.0, 0.0]]
    )
    data = [0] * 1600
    for cell_x in range(24, 27):
        data[20 * 40 + cell_x] = 20

    result = _check(
        states,
        _map(data),
        allow_initial_inflated_center_prefix=True,
    )
    assert not result.safe
    assert result.reason == "TRAJECTORY_CENTER_NOT_FREE"
    assert result.cell_value == 20


def test_initial_prefix_option_never_allows_unknown_or_occupied_cells():
    states = np.asarray([[0.0], [0.0], [0.0], [0.0]])
    for value, expected_reason in (
        (-1, "TRAJECTORY_CENTER_UNKNOWN"),
        (100, "TRAJECTORY_CENTER_NOT_FREE"),
    ):
        data = [0] * 1600
        data[20 * 40 + 20] = value
        result = _check(
            states,
            _map(data),
            allow_initial_inflated_center_prefix=True,
        )
        assert not result.safe
        assert result.reason == expected_reason
        assert result.cell_value == value


def test_initial_prefix_option_keeps_padded_footprint_fail_closed():
    states = np.asarray([[0.0], [0.0], [0.0], [0.0]])
    for value, expected_reason in (
        (-1, "TRAJECTORY_FOOTPRINT_UNKNOWN"),
        (100, "TRAJECTORY_FOOTPRINT_OCCUPIED"),
    ):
        data = [0] * 1600
        data[20 * 40 + 20] = 50
        # Inside the padded front of the footprint, away from the centre.
        data[20 * 40 + 24] = value
        result = _check(
            states,
            _map(data),
            allow_initial_inflated_center_prefix=True,
        )
        assert not result.safe
        assert result.reason == expected_reason
        assert result.cell_value == value


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
