from dataclasses import replace

import numpy as np
import pytest

from dream_limo.core.mission import MissionEndGuard, stopping_speed_limit
from dream_limo.core.mpc import RiskAwareMPC
from dream_limo.core.risk_field import DREAMRiskField
from dream_limo.core.types import EgoState
from dream_limo.limo_scale import default_deployment_config, get_preset


def test_square_root_stop_profile_is_bounded_and_reaches_zero():
    assert stopping_speed_limit(
        10.0, cruise_speed=0.15, braking_deceleration=0.10
    ) == pytest.approx(0.15)
    assert stopping_speed_limit(
        0.05, cruise_speed=0.15, braking_deceleration=0.10
    ) == pytest.approx(0.10)
    assert stopping_speed_limit(
        0.0, cruise_speed=0.15, braking_deceleration=0.10
    ) == 0.0
    assert stopping_speed_limit(
        -0.2, cruise_speed=0.15, braking_deceleration=0.10
    ) == 0.0
    with pytest.raises(ValueError):
        stopping_speed_limit(1.0, cruise_speed=0.15, braking_deceleration=0.0)


def test_mission_completion_is_a_one_way_latch():
    guard = MissionEndGuard(
        goal_x=5.55,
        position_tolerance=0.04,
        stop_speed_tolerance=0.03,
    )
    assert not guard.update(5.52, 0.10)
    assert guard.update(5.52, 0.02)
    # Neither a pose reset nor renewed speed can clear completion.
    assert guard.update(0.35, 0.50)
    assert guard.complete


def test_crossing_goal_latches_immediately_even_if_odometry_reports_speed():
    guard = MissionEndGuard(5.55, 0.04, 0.03)
    assert guard.update(5.551, 0.15)
    assert guard.remaining_distance(5.551) == 0.0


def test_mpc_reference_brakes_to_the_goal_without_crossing_it():
    base = default_deployment_config()
    config = replace(base, mpc=replace(base.mpc, target_speed=0.15))
    mpc = RiskAwareMPC(config, enforce_map_bounds=True)
    ego = EgoState(
        config.arena.mission_goal_x - 0.06,
        config.arena.lane_centers[config.arena.target_lane],
        0.0,
        0.11,
        lane_index=config.arena.target_lane,
    )
    reference = mpc._reference(ego, config.arena.target_lane)

    assert np.all(np.diff(reference[0]) >= -1.0e-12)
    assert np.max(reference[0]) <= config.arena.mission_goal_x + 1.0e-12
    assert np.all(np.diff(reference[2]) <= 1.0e-12)
    assert reference[2, 0] < config.mpc.target_speed
    assert reference[2, -1] < 1.0e-8


def test_balanced_and_pure_mpc_share_target_speed_and_mission_braking():
    base = default_deployment_config()
    config = replace(base, mpc=replace(base.mpc, target_speed=0.15))
    field = DREAMRiskField(config)
    ego = EgoState(
        config.arena.mission_goal_x - 0.06,
        config.arena.lane_centers[config.arena.target_lane],
        0.0,
        0.11,
        lane_index=config.arena.target_lane,
    )
    balanced = RiskAwareMPC(config, enforce_map_bounds=True).solve(
        ego, config.arena.target_lane, [], field, get_preset("balanced")
    )
    pure = RiskAwareMPC(config, enforce_map_bounds=True).solve(
        ego, config.arena.target_lane, [], field, get_preset("pure_mpc")
    )

    assert not balanced.used_fallback
    assert not pure.used_fallback
    assert balanced.command.acceleration < 0.0
    assert balanced.command.target_speed < ego.speed
    np.testing.assert_allclose(balanced.states, pure.states, rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(balanced.controls, pure.controls, rtol=1.0e-6, atol=1.0e-6)
