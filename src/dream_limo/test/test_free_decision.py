import numpy as np

from dream_limo.core.free_decision import (
    evaluate_route_maneuver_risk,
    heading_hold_path,
    sample_upcoming_route,
)
from dream_limo.limo_scale import get_preset


def test_upcoming_route_is_trimmed_to_ego_projection():
    sampled = sample_upcoming_route(
        [[-2.0, 0.0], [0.0, 0.0], [2.0, 0.0]],
        ego_xy=[0.4, 0.1],
        lookahead=1.0,
        samples=3,
    )
    np.testing.assert_allclose(sampled[0], [0.4, 0.1])
    assert sampled[-1, 0] >= 1.3 - 1.0e-12


def test_balanced_vetoes_risky_lateral_maneuver_but_not_straight_following():
    def risk_at(_x, _y):
        return 2.0

    lateral = evaluate_route_maneuver_risk(
        [[0.0, 0.0], [1.0, 0.4], [2.0, 0.5]],
        ego_xy=[0.0, 0.0],
        ego_yaw=0.0,
        risk_at=risk_at,
        preset=get_preset("balanced"),
        lookahead=2.0,
        samples=10,
    )
    straight = evaluate_route_maneuver_risk(
        [[0.0, 0.0], [2.0, 0.0]],
        ego_xy=[0.0, 0.0],
        ego_yaw=0.0,
        risk_at=risk_at,
        preset=get_preset("balanced"),
        lookahead=2.0,
        samples=10,
    )
    assert lateral.maneuver and lateral.vetoed
    assert not straight.maneuver and not straight.vetoed
    assert lateral.score == 2.0


def test_pure_mpc_never_applies_route_veto():
    decision = evaluate_route_maneuver_risk(
        [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]],
        ego_xy=[0.0, 0.0],
        ego_yaw=0.0,
        risk_at=lambda _x, _y: 10.0,
        preset=get_preset("pure_mpc"),
        lookahead=2.0,
        samples=10,
    )
    assert decision.maneuver
    assert not decision.vetoed


def test_heading_hold_path_preserves_current_direction():
    path = heading_hold_path(
        ego_x=1.0,
        ego_y=-2.0,
        ego_yaw=np.pi / 2.0,
        distance=1.0,
        samples=5,
    )
    np.testing.assert_allclose(path[0], [1.0, -2.0], atol=1.0e-12)
    np.testing.assert_allclose(path[-1], [1.0, -1.0], atol=1.0e-12)
