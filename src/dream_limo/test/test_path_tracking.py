import numpy as np
import pytest

from dream_limo.core.path_tracking import (
    PathValidationError,
    build_path_reference,
    validate_forward_pose_alignment,
    validate_path_points,
)


def test_validate_path_accepts_transpose_and_removes_adjacent_duplicates():
    path_2_by_n = np.array(
        [
            [0.0, 0.0, 0.5, 1.0],
            [0.0, 0.0, 0.2, 0.4],
        ]
    )
    clean = validate_path_points(path_2_by_n)
    np.testing.assert_allclose(clean, [[0.0, 0.0], [0.5, 0.2], [1.0, 0.4]])


@pytest.mark.parametrize(
    "path",
    (
        [[0.0, 0.0]],
        [[0.0, 0.0], [0.0, 0.0]],
        [[0.0, 0.0], [np.nan, 1.0]],
        np.zeros((2, 2, 1)),
        np.zeros((3, 3)),
    ),
)
def test_validate_path_rejects_nonfinite_or_degenerate_input(path):
    with pytest.raises(PathValidationError):
        validate_path_points(path)


def test_reference_is_ego_anchored_arc_length_sampled_and_stops_euclideanly():
    path = np.array(
        [
            [0.0, 0.0],
            [0.5, 0.0],
            [0.5, 0.5],
        ]
    )
    reference = build_path_reference(
        path,
        ego_xy=[0.1, 0.0],
        ego_yaw=0.0,
        horizon=8,
        dt=0.2,
        cruise_speed=0.5,
        braking_deceleration=0.1,
    )

    assert reference.shape == (4, 9)
    np.testing.assert_allclose(reference[0:2, 0], [0.1, 0.0])
    assert np.all(np.isfinite(reference))
    chord_lengths = np.linalg.norm(np.diff(reference[0:2], axis=1), axis=0)
    assert np.all(chord_lengths <= 0.2 * reference[2, :-1] + 1.0e-12)
    goal = path[-1]
    euclidean_remaining = np.linalg.norm(goal[:, None] - reference[0:2], axis=0)
    expected_speed = np.minimum(0.5, np.sqrt(2.0 * 0.1 * euclidean_remaining))
    np.testing.assert_allclose(reference[2], expected_speed, atol=1e-12)


def test_reference_unwraps_yaw_across_minus_pi_boundary():
    path = np.array(
        [
            [0.0, 0.0],
            [-0.10, 0.001],
            [-0.20, -0.001],
        ]
    )
    reference = build_path_reference(
        path,
        ego_xy=path[0],
        ego_yaw=np.pi - 0.01,
        horizon=8,
        dt=0.2,
        cruise_speed=0.5,
        braking_deceleration=1.0,
    )

    assert np.max(np.abs(np.diff(reference[3]))) < 0.1
    assert abs(reference[3, 0] - (np.pi - 0.01)) < 0.05
    assert np.max(reference[3]) > np.pi


def test_reference_at_existing_goal_is_a_stationary_horizon():
    reference = build_path_reference(
        [[0.0, 0.0], [1.0, 0.4]],
        ego_xy=[1.0, 0.4],
        ego_yaw=0.7,
        horizon=6,
        dt=0.2,
        cruise_speed=0.5,
        braking_deceleration=0.1,
    )
    np.testing.assert_allclose(reference[0], 1.0)
    np.testing.assert_allclose(reference[1], 0.4)
    np.testing.assert_allclose(reference[2], 0.0)
    np.testing.assert_allclose(reference[3], 0.7)


def test_reverse_pose_alignment_and_large_cross_track_error_fail_closed():
    with pytest.raises(PathValidationError, match="reverse-oriented"):
        validate_forward_pose_alignment(
            [[0.0, 0.0], [1.0, 0.0]],
            [np.pi, np.pi],
        )
    with pytest.raises(PathValidationError, match="cross-track"):
        build_path_reference(
            [[0.0, 0.0], [1.0, 0.0]],
            ego_xy=[0.0, 0.2],
            ego_yaw=0.0,
            horizon=6,
            dt=0.2,
            cruise_speed=0.15,
            braking_deceleration=0.1,
            maximum_cross_track_error=0.10,
        )


def test_reference_honors_terminal_goal_yaw_only_when_horizon_reaches_goal():
    reference = build_path_reference(
        [[0.0, 0.0], [0.10, 0.0]],
        ego_xy=[0.0, 0.0],
        ego_yaw=0.0,
        horizon=6,
        dt=0.2,
        cruise_speed=0.15,
        braking_deceleration=0.1,
        terminal_yaw=0.25,
    )
    assert reference[3, -1] == pytest.approx(0.25)

    distant = build_path_reference(
        [[0.0, 0.0], [1.0, 0.0]],
        ego_xy=[0.0, 0.0],
        ego_yaw=0.0,
        horizon=6,
        dt=0.2,
        cruise_speed=0.15,
        braking_deceleration=0.1,
        terminal_yaw=0.25,
    )
    assert distant[3, -1] == pytest.approx(0.0)
