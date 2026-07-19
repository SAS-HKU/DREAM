import numpy as np

from dream_limo.core.mpc import RiskAwareMPC
from dream_limo.core.risk_field import DREAMRiskField
from dream_limo.core.types import EgoState, Vehicle
from dream_limo.limo_scale import default_deployment_config, get_preset


def test_mpc_is_finite_at_standstill_and_enforces_nonnegative_speed():
    config = default_deployment_config()
    field = DREAMRiskField(config)
    mpc = RiskAwareMPC(config)
    ego = EgoState(0.3, 0.45, 0.0, 0.0, lane_index=0)
    result = mpc.solve(ego, 0, [], field, get_preset("balanced"))
    assert not result.used_fallback
    assert np.all(np.isfinite(result.states))
    assert np.min(result.states[2]) >= -1e-4
    assert np.max(result.states[2]) <= config.mpc.maximum_speed + 1e-4


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
