import numpy as np
import pytest

from dream_limo.core.mpc import RiskAwareMPC
from dream_limo.core.risk_field import DREAMRiskField
from dream_limo.core.types import EgoState
from dream_limo.limo_scale import default_deployment_config, get_preset


def _straight_problem(speed: float = 0.10):
    config = default_deployment_config()
    field = DREAMRiskField(config)
    ego = EgoState(0.4, 0.0, 0.0, speed)
    path = np.asarray([[0.4, 0.0], [2.5, 0.0]], dtype=np.float64)
    return config, field, ego, path, get_preset("pure_mpc")


def test_reference_mpc_default_path_is_unchanged_when_hooks_are_absent():
    config, field, ego, path, preset = _straight_problem()

    legacy_result = RiskAwareMPC(config).solve_reference(
        ego, path, [], field, preset
    )
    explicit_result = RiskAwareMPC(config).solve_reference(
        ego,
        path,
        [],
        field,
        preset,
        velocity_upper_bound=None,
        fixed_control_prefix=None,
        commit_solution=True,
    )

    assert not legacy_result.used_fallback
    assert not explicit_result.used_fallback
    np.testing.assert_allclose(
        explicit_result.states, legacy_result.states, rtol=0.0, atol=1.0e-10
    )
    np.testing.assert_allclose(
        explicit_result.controls, legacy_result.controls, rtol=0.0, atol=1.0e-10
    )
    assert explicit_result.objective == pytest.approx(
        legacy_result.objective, rel=0.0, abs=1.0e-10
    )
    assert legacy_result.maximum_velocity_slack == 0.0
    assert explicit_result.maximum_velocity_slack == 0.0


def test_reference_mpc_applies_soft_velocity_bound_to_every_state():
    config, field, ego, path, preset = _straight_problem()
    bound = 0.12

    nominal_result = RiskAwareMPC(config).solve_reference(
        ego, path, [], field, preset
    )
    bounded_result = RiskAwareMPC(config).solve_reference(
        ego,
        path,
        [],
        field,
        preset,
        velocity_upper_bound=bound,
        velocity_slack_weight=1.0e6,
    )

    assert not nominal_result.used_fallback
    assert not bounded_result.used_fallback
    assert bounded_result.maximum_velocity_slack < 2.0e-3
    assert np.max(bounded_result.states[2, :]) <= (
        bound + bounded_result.maximum_velocity_slack + 2.0e-3
    )
    assert np.max(nominal_result.states[2, :]) > (
        np.max(bounded_result.states[2, :]) + 0.05
    )


def test_velocity_bound_uses_slack_when_current_speed_exceeds_new_bound():
    config, field, ego, path, preset = _straight_problem(speed=0.20)

    result = RiskAwareMPC(config).solve_reference(
        ego,
        path,
        [],
        field,
        preset,
        velocity_upper_bound=0.05,
        velocity_slack_weight=1.0e4,
    )

    assert not result.used_fallback
    assert result.states[2, 0] == pytest.approx(ego.speed, abs=2.0e-3)
    assert result.maximum_velocity_slack >= 0.145
    assert np.isfinite(result.maximum_velocity_slack)
    assert result.maximum_future_velocity_slack < result.maximum_velocity_slack
    assert result.controls[0, 0] < 0.0


def test_contingency_prefix_can_be_verified_without_committing_warm_start():
    config, field, ego, path, preset = _straight_problem(speed=0.10)
    mpc = RiskAwareMPC(config)
    exploration = mpc.solve_reference(
        ego,
        path,
        [],
        field,
        preset,
        velocity_upper_bound=0.14,
    )
    assert not exploration.used_fallback
    committed_states = mpc.last_states.copy()
    committed_controls = mpc.last_controls.copy()
    committed_applied_control = mpc.last_applied_control.copy()
    prefix = exploration.controls[:, :2].copy()

    fallback = mpc.solve_reference(
        ego,
        path,
        [],
        field,
        preset,
        velocity_upper_bound=0.20,
        fixed_control_prefix=prefix,
        commit_solution=False,
    )

    assert not fallback.used_fallback
    np.testing.assert_allclose(
        fallback.controls[:, :2], prefix, rtol=0.0, atol=2.0e-3
    )
    np.testing.assert_array_equal(mpc.last_states, committed_states)
    np.testing.assert_array_equal(mpc.last_controls, committed_controls)
    np.testing.assert_array_equal(
        mpc.last_applied_control, committed_applied_control
    )


def test_noncommitting_branch_can_be_selected_explicitly_after_verification():
    config, field, ego, path, preset = _straight_problem(speed=0.10)
    mpc = RiskAwareMPC(config)
    selected = mpc.solve_reference(
        ego,
        path,
        [],
        field,
        preset,
        velocity_upper_bound=0.14,
        commit_solution=False,
    )

    assert not selected.used_fallback
    assert mpc.last_states is None
    assert mpc.last_controls is None
    np.testing.assert_array_equal(mpc.last_applied_control, np.zeros(2))

    mpc.commit_result(selected)

    np.testing.assert_array_equal(mpc.last_states, selected.states)
    np.testing.assert_array_equal(mpc.last_controls, selected.controls)
    np.testing.assert_array_equal(
        mpc.last_applied_control,
        np.asarray(
            [
                selected.command.acceleration,
                selected.command.steering,
            ]
        ),
    )


@pytest.mark.parametrize("bound", [-0.01, 0.61, np.nan, np.inf])
def test_velocity_bound_rejects_values_outside_configured_speed_range(bound):
    config, field, ego, path, preset = _straight_problem()

    with pytest.raises(ValueError, match="velocity_upper_bound"):
        RiskAwareMPC(config).solve_reference(
            ego,
            path,
            [],
            field,
            preset,
            velocity_upper_bound=bound,
        )


def test_control_prefix_rejects_wrong_shape():
    config, field, ego, path, preset = _straight_problem()

    with pytest.raises(ValueError, match="fixed_control_prefix"):
        RiskAwareMPC(config).solve_reference(
            ego,
            path,
            [],
            field,
            preset,
            fixed_control_prefix=np.zeros((3, 2)),
        )
