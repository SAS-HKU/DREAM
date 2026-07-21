import numpy as np

from dream_limo.core.mpc import RiskAwareMPC
from dream_limo.core.path_tracking import build_path_reference
from dream_limo.core.risk_field import DREAMRiskField
from dream_limo.core.types import EgoState, Vehicle
from dream_limo.limo_scale import default_deployment_config, get_preset


def test_mpc_is_finite_at_standstill_and_enforces_nonnegative_speed():
    config = default_deployment_config()
    field = DREAMRiskField(config)
    mpc = RiskAwareMPC(config, enforce_map_bounds=True)
    ego = EgoState(0.3, 0.45, 0.0, 0.0, lane_index=0)
    result = mpc.solve(ego, 0, [], field, get_preset("balanced"))
    assert not result.used_fallback
    assert np.all(np.isfinite(result.states))
    assert np.min(result.states[2]) >= -1e-4
    assert np.max(result.states[2]) <= config.mpc.maximum_speed + 1e-4
    radius = np.hypot(
        0.5 * config.mpc.robot_length,
        0.5 * config.mpc.robot_width,
    ) + config.safety.collision_inflation_margin
    quantization = 0.5 * config.grid.resolution
    assert np.min(result.states[0, 1:]) >= (
        config.grid.x_min + radius - quantization - 1e-4
    )
    assert np.max(result.states[0, 1:]) <= (
        config.grid.x_max - radius + quantization + 1e-4
    )
    assert np.min(result.states[1, 2:]) >= (
        config.grid.road_y_min + radius - quantization - 1e-4
    )
    assert np.max(result.states[1, 2:]) <= (
        config.grid.road_y_max - radius + quantization + 1e-4
    )


def test_risk_expands_cbf_and_headway_and_enters_cost():
    config = default_deployment_config()
    field = DREAMRiskField(config)
    field.R.fill(3.0)
    balanced = get_preset("balanced")
    baseline = get_preset("baseline")
    assert field.cbf_scale(1.0, 0.45, balanced) > 1.0
    assert field.headway_scale(1.0, 0.45, balanced) > 1.0
    assert field.cbf_scale(1.0, 0.45, baseline) == 1.0
    ego = EgoState(0.3, 0.45, 0.0, 0.3, lane_index=0)
    leader = Vehicle("leader", 1.5, 0.45, vx=0.1, length=0.22, width=0.22)
    dream_result = RiskAwareMPC(config).solve(ego, 0, [leader], field, balanced)
    base_result = RiskAwareMPC(config).solve(ego, 0, [leader], field, baseline)
    assert not dream_result.used_fallback
    assert not base_result.used_fallback
    assert dream_result.risk_profile[0] == 3.0
    assert dream_result.objective != base_result.objective


def test_pure_mpc_control_is_invariant_to_dream_risk_field():
    """The baseline retains nominal CBFs but has no DREAM behavioral channel."""
    config = default_deployment_config()
    zero_field = DREAMRiskField(config)
    high_field = DREAMRiskField(config)
    high_field.R.fill(config.pde.risk_ceiling)
    ego = EgoState(3.1, 0.40, 0.0, 0.35, lane_index=0)
    merger = Vehicle("merger", 4.0, -0.05, vx=0.18, length=0.22, width=0.22)
    preset = get_preset("pure_mpc")

    zero_result = RiskAwareMPC(config).solve(ego, 1, [merger], zero_field, preset)
    high_result = RiskAwareMPC(config).solve(ego, 1, [merger], high_field, preset)

    assert not zero_result.used_fallback
    assert not high_result.used_fallback
    np.testing.assert_allclose(
        high_result.controls,
        zero_result.controls,
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    assert high_result.maximum_slack == zero_result.maximum_slack


def test_reference_mpc_tracks_a_curved_path_and_enforces_full_footprint_bounds():
    config = default_deployment_config()
    field = DREAMRiskField(config)
    ego = EgoState(0.3, 0.0, 0.0, 0.1)
    path = np.array(
        [
            [0.3, 0.0],
            [0.6, 0.03],
            [1.0, 0.15],
            [2.0, 0.30],
        ]
    )

    result = RiskAwareMPC(config).solve_reference(
        ego, path, [], field, get_preset("balanced")
    )

    assert not result.used_fallback
    assert result.command.steering > 0.01
    assert result.states[1, -1] > result.states[1, 0]
    reference = build_path_reference(
        path,
        ego_xy=[ego.x, ego.y],
        ego_yaw=ego.yaw,
        horizon=config.mpc.horizon,
        dt=config.mpc.dt,
        cruise_speed=config.mpc.target_speed,
        braking_deceleration=config.mpc.mission_braking_deceleration,
        maximum_cross_track_error=config.mpc.maximum_path_cross_track_error,
    )
    for step in range(config.mpc.horizon + 1):
        tangent = np.array(
            [np.cos(reference[3, step]), np.sin(reference[3, step])]
        )
        normal = np.array(
            [-np.sin(reference[3, step]), np.cos(reference[3, step])]
        )
        position_error = (
            result.states[0:2, step] - reference[0:2, step]
        )
        along_track = tangent @ position_error
        cross_track = normal @ position_error
        assert abs(along_track) <= (
            config.mpc.path_longitudinal_half_width + 2.0e-3
        )
        assert abs(cross_track) <= config.mpc.path_corridor_half_width + 2.0e-3
    radius = np.hypot(
        0.5 * config.mpc.robot_length,
        0.5 * config.mpc.robot_width,
    ) + config.safety.collision_inflation_margin
    quantization = 0.5 * config.grid.resolution
    assert np.min(result.states[0]) >= config.grid.x_min + radius - quantization - 1e-4
    assert np.max(result.states[0]) <= config.grid.x_max - radius + quantization + 1e-4
    assert np.min(result.states[1]) >= config.grid.road_y_min + radius - quantization - 1e-4
    assert np.max(result.states[1]) <= config.grid.road_y_max - radius + quantization + 1e-4


def test_reference_mpc_headway_uses_route_tangent_for_vertical_path():
    config = default_deployment_config()
    field = DREAMRiskField(config)
    preset = get_preset("pure_mpc")
    ego = EgoState(1.0, -0.30, np.pi / 2.0, 0.1)
    path = np.array([[1.0, -0.30], [1.0, 0.0], [1.0, 0.30], [1.0, 0.45]])
    leader = Vehicle("leader", 1.0, 0.20, vx=0.0, vy=0.05)
    lateral_vehicle = Vehicle("lateral", 1.40, 0.20, vx=0.0, vy=0.05)

    open_result = RiskAwareMPC(config).solve_reference(
        ego, path, [], field, preset
    )
    leader_result = RiskAwareMPC(config).solve_reference(
        ego, path, [leader], field, preset
    )
    lateral_result = RiskAwareMPC(config).solve_reference(
        ego, path, [lateral_vehicle], field, preset
    )

    assert not open_result.used_fallback
    assert not leader_result.used_fallback
    assert not lateral_result.used_fallback
    assert leader_result.command.target_speed < open_result.command.target_speed - 0.01
    assert lateral_result.command.target_speed > leader_result.command.target_speed + 0.01


def test_reference_pure_mpc_is_invariant_to_dream_risk_field():
    config = default_deployment_config()
    zero_field = DREAMRiskField(config)
    high_field = DREAMRiskField(config)
    high_field.R.fill(config.pde.risk_ceiling)
    preset = get_preset("pure_mpc")
    ego = EgoState(0.3, 0.0, 0.0, 0.1)
    path = np.array([[0.3, 0.0], [0.6, 0.03], [1.0, 0.15], [2.0, 0.30]])
    leader = Vehicle("leader", 1.1, 0.15, vx=0.1)

    zero_result = RiskAwareMPC(config).solve_reference(
        ego, path, [leader], zero_field, preset
    )
    high_result = RiskAwareMPC(config).solve_reference(
        ego, path, [leader], high_field, preset
    )

    assert not zero_result.used_fallback
    assert not high_result.used_fallback
    np.testing.assert_allclose(
        high_result.controls,
        zero_result.controls,
        rtol=1.0e-7,
        atol=1.0e-7,
    )
    assert high_result.maximum_slack == zero_result.maximum_slack


def test_reference_balanced_preset_retains_dream_risk_speed_cost():
    config = default_deployment_config()
    zero_field = DREAMRiskField(config)
    high_field = DREAMRiskField(config)
    high_field.R.fill(3.0)
    preset = get_preset("balanced")
    ego = EgoState(0.3, 0.0, 0.0, 0.3)
    path = np.array([[0.3, 0.0], [1.0, 0.0], [2.0, 0.0]])

    zero_result = RiskAwareMPC(config).solve_reference(
        ego, path, [], zero_field, preset
    )
    high_result = RiskAwareMPC(config).solve_reference(
        ego, path, [], high_field, preset
    )

    assert not zero_result.used_fallback
    assert not high_result.used_fallback
    assert high_result.risk_profile[0] == 3.0
    assert high_result.objective != zero_result.objective
    assert not np.allclose(high_result.controls, zero_result.controls)
