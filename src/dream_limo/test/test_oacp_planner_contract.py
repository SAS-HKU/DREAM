from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from dream_limo.core.mpc import MPCResult
from dream_limo.core.types import ControlCommand, EgoState, Vehicle
from dream_limo.free_planner_node import (
    DreamFreePlannerNode,
    _ZeroRiskField,
    shared_controller_parameter_fingerprint,
)
from dream_limo.limo_scale import default_deployment_config
from dream_limo.ros_utils import ControlSourceStamp


def test_shared_controller_fingerprint_excludes_the_selected_risk_arm():
    config = default_deployment_config()
    fingerprint = shared_controller_parameter_fingerprint(config.mpc)
    assert len(fingerprint) == 64
    assert fingerprint == shared_controller_parameter_fingerprint(config.mpc)
    changed = replace(config.mpc, target_speed=0.15)
    assert shared_controller_parameter_fingerprint(changed) != fingerprint


def test_oacp_uses_an_explicit_zero_risk_adapter():
    field = _ZeroRiskField()
    preset = SimpleNamespace()
    assert field.risk_at(0.0, 0.0) == 0.0
    assert field.risk_at(10.0, -2.0) == 0.0
    assert field.cbf_scale(0.0, 0.0, preset) == 1.0
    assert field.headway_scale(0.0, 0.0, preset) == 1.0


def _planner_with_oacp_bounds():
    planner = DreamFreePlannerNode.__new__(DreamFreePlannerNode)
    config = default_deployment_config()
    planner.config = replace(
        config,
        mpc=replace(config.mpc, target_speed=0.15),
    )
    planner.oacp_status = {
        "exploration_velocity_bound": 0.12,
        "fallback_velocity_bound": 0.15,
        "v_occ_min": 0.08,
        "v_occ_max": 0.15,
        "risk_total": 1.0,
    }
    return planner


@pytest.mark.parametrize(
    ("key", "expected"),
    (
        ("exploration_velocity_bound", 0.12),
        ("fallback_velocity_bound", 0.15),
        ("v_occ_min", 0.08),
    ),
)
def test_oacp_bound_contract_accepts_only_shared_speed_limits(key, expected):
    assert _planner_with_oacp_bounds()._oacp_bound(key) == expected


@pytest.mark.parametrize("value", [-0.01, 0.61, float("inf"), "invalid"])
def test_oacp_bound_contract_rejects_invalid_provider_values(value):
    planner = _planner_with_oacp_bounds()
    planner.oacp_status["exploration_velocity_bound"] = value
    with pytest.raises(ValueError, match="OACP bound"):
        planner._oacp_bound("exploration_velocity_bound")


@pytest.mark.parametrize(
    "updates",
    (
        {"v_occ_min": 0.13},
        {"fallback_velocity_bound": 0.11},
        {"v_occ_max": 0.20},
    ),
)
def test_complete_oacp_bound_relation_must_match_shared_target(updates):
    planner = _planner_with_oacp_bounds()
    planner.oacp_status.update(updates)
    with pytest.raises(ValueError, match="OACP bounds violate"):
        planner._validated_oacp_bounds()


def test_arm_status_discloses_adaptation_and_shared_controller():
    planner = _planner_with_oacp_bounds()
    planner.planner_mode = "oacp_vb"
    planner.oacp_mode = True
    planner.preset = SimpleNamespace(name="pure_mpc")
    planner.controller_parameter_hash = "a" * 64
    planner.get_parameter = lambda name: SimpleNamespace(
        value={
            "update_rate": 5.0,
            "oacp_calibration_logging_only": False,
        }[name]
    )
    status = planner._arm_status()
    assert status["arm"] == "oacp_vb"
    assert status["risk_channel"] == "phantom_reachability_velocity_bound"
    assert status["shared_controller_parameter_hash"] == "a" * 64
    assert status["shared_mpc_horizon_steps"] == 6
    assert status["shared_mpc_dt"] == 0.2


def test_oacp_path_and_bound_are_activated_as_one_matching_pair():
    planner = DreamFreePlannerNode.__new__(DreamFreePlannerNode)
    planner.oacp_mode = True
    planner.path_points = np.asarray([[0.0, 0.0], [1.0, 0.0]])
    planner.path_receipt = 9.0
    planner.path_source_stamp = 9.0
    planner.oacp_status = {"path_source_stamp": 9.0}
    planner.oacp_status_receipt = 9.1
    pending = np.asarray([[0.1, 0.0], [1.0, 0.0]])
    planner.pending_path_points = pending
    planner.pending_path_receipt = 10.0
    planner.pending_path_source_stamp = 10.0
    planner.pending_oacp_status = {
        "provider": "oacp_vb",
        "ready": True,
        "exact_bound_valid": True,
        "path_source_stamp": 10.0,
    }
    planner.pending_oacp_status_receipt = 10.1
    planner.oacp_contingency_cached_valid = True
    planner.get_parameter = lambda name: SimpleNamespace(
        value={"path_stamp_tolerance": 1.0e-6}[name]
    )

    assert planner._try_activate_oacp_pair()
    assert planner.path_points is pending
    assert planner.path_source_stamp == 10.0
    assert planner.oacp_status["path_source_stamp"] == 10.0
    assert planner.pending_path_points is None
    assert planner.pending_oacp_status == {}
    assert planner.oacp_contingency_cached_valid is None


def test_mismatched_pending_oacp_pair_preserves_previous_active_pair():
    planner = DreamFreePlannerNode.__new__(DreamFreePlannerNode)
    planner.oacp_mode = True
    active = np.asarray([[0.0, 0.0], [1.0, 0.0]])
    planner.path_points = active
    planner.path_receipt = 9.0
    planner.path_source_stamp = 9.0
    planner.oacp_status = {"path_source_stamp": 9.0}
    planner.oacp_status_receipt = 9.1
    planner.pending_path_points = np.asarray([[0.1, 0.0], [1.0, 0.0]])
    planner.pending_path_receipt = 10.0
    planner.pending_path_source_stamp = 10.0
    planner.pending_oacp_status = {
        "ready": True,
        "exact_bound_valid": True,
        "path_source_stamp": 10.2,
    }
    planner.pending_oacp_status_receipt = 10.1
    planner.get_parameter = lambda name: SimpleNamespace(
        value={"path_stamp_tolerance": 1.0e-6}[name]
    )

    assert not planner._try_activate_oacp_pair()
    assert planner.path_points is active
    assert planner.path_source_stamp == 9.0
    assert planner.oacp_status["path_source_stamp"] == 9.0


def _mpc_result(*, future_slack=0.0, cbf_slack=0.0, fallback=False):
    states = np.zeros((4, 7), dtype=np.float64)
    states[0, :] = np.linspace(0.0, 0.12, 7)
    states[2, :] = 0.1
    return MPCResult(
        command=ControlCommand(0.1, 0.0, 0.0),
        states=states,
        controls=np.vstack(
            (
                np.linspace(0.01, 0.06, 6),
                np.linspace(0.001, 0.006, 6),
            )
        ),
        status="fallback" if fallback else "optimal",
        solve_seconds=0.01,
        objective=1.0,
        maximum_slack=cbf_slack,
        risk_profile=np.zeros(7),
        used_fallback=fallback,
        maximum_velocity_slack=future_slack,
        maximum_future_velocity_slack=future_slack,
    )


class _RecordingMPC:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []
        self.committed = []

    def solve_reference(self, *args, **kwargs):
        self.calls.append(kwargs)
        return self.results.pop(0)

    def commit_result(self, result):
        self.committed.append(result)


def _orchestration_planner(results, *, calibration=False):
    planner = _planner_with_oacp_bounds()
    planner.path_points = np.asarray([[0.0, 0.0], [2.0, 0.0]])
    planner.path_source_stamp = 1.0
    planner.vehicles = []
    planner.field = _ZeroRiskField()
    planner.preset = SimpleNamespace(name="pure_mpc")
    planner.mpc = _RecordingMPC(results)
    values = {
        "oacp_velocity_slack_weight": 1.0e4,
        "oacp_calibration_logging_only": calibration,
        "oacp_enable_contingency": True,
        "oacp_contingency_check_rate": 1.0,
        "oacp_shared_prefix_steps": 2,
        "oacp_contingency_slack_tolerance": 1.0e-4,
        "oacp_maximum_future_velocity_slack": 0.01,
        "oacp_gate_status_timeout": 0.30,
        "oacp_prefix_position_tracking_tolerance": 0.01,
        "oacp_prefix_speed_tracking_tolerance": 0.03,
        "oacp_prefix_yaw_tracking_tolerance": 0.05,
        "oacp_prefix_advance_minimum_progress": 0.95,
        "maximum_allowed_cbf_slack": 0.05,
    }
    planner.get_parameter = lambda name: SimpleNamespace(value=values[name])
    planner.oacp_contingency_last_check_stamp = None
    planner.oacp_contingency_cached_valid = None
    planner.oacp_contingency_cached_context = None
    planner.oacp_contingency_cached_prefix = None
    planner.oacp_contingency_cached_states = None
    planner.oacp_contingency_cached_prefix_cursor = 0
    planner.oacp_prefix_pending_control_stamp = None
    planner.oacp_prefix_pending_cursor = None
    planner.hardware_gate_status = {}
    planner.hardware_gate_status_receipt = None
    return planner


def _token(value):
    sec = int(value)
    nanosec = int(round((value - sec) * 1.0e9))
    return ControlSourceStamp(sec=sec, nanosec=nanosec)


def _confirm_prefix_forwarding(planner, *, cursor, publish=10.01, receipt=10.15):
    source_stamp = _token(publish)
    planner.oacp_prefix_pending_control_stamp = source_stamp
    planner.oacp_prefix_pending_cursor = cursor
    planner.hardware_gate_status = {
        "ready": True,
        "hardware_output_enabled": True,
        "candidate_receipt_stamp": publish + 0.01,
        "forwarded_control_source_stamp": source_stamp.as_mapping(),
    }
    planner.hardware_gate_status_receipt = receipt


def test_oacp_contingency_verifies_a_noncommitting_shared_prefix():
    exploration = _mpc_result()
    alternative = _mpc_result()
    planner = _orchestration_planner([exploration, alternative])
    result, details = planner._solve_oacp_reference(
        EgoState(0.0, 0.0, 0.0, 0.1),
        0.0,
        now=10.0,
    )
    assert result is exploration
    assert len(planner.mpc.calls) == 2
    assert planner.mpc.calls[0]["velocity_upper_bound"] == 0.12
    assert planner.mpc.calls[1]["velocity_upper_bound"] == 0.15
    assert planner.mpc.calls[0]["commit_solution"] is False
    assert planner.mpc.calls[1]["commit_solution"] is False
    np.testing.assert_array_equal(
        planner.mpc.calls[1]["fixed_control_prefix"],
        exploration.controls[:, :2],
    )
    assert details["oacp_contingency_valid"] is True
    assert details["oacp_contingency_clamp_event"] is False
    assert details["oacp_contingency_check_performed"] is True
    assert len(planner.mpc.committed) == 1
    assert planner.mpc.committed[0] is exploration


@pytest.mark.parametrize(
    "invalid_alternative",
    (
        _mpc_result(future_slack=0.001),
        _mpc_result(cbf_slack=0.001),
    ),
)
def test_active_contingency_slack_forces_a_new_minimum_bound_solve(
    invalid_alternative,
):
    exploration = _mpc_result()
    clamped = _mpc_result()
    planner = _orchestration_planner(
        [exploration, invalid_alternative, clamped]
    )
    result, details = planner._solve_oacp_reference(
        EgoState(0.0, 0.0, 0.0, 0.1),
        0.0,
        now=10.0,
    )
    assert result is clamped
    assert len(planner.mpc.calls) == 3
    assert planner.mpc.calls[2]["velocity_upper_bound"] == 0.08
    assert details["oacp_contingency_valid"] is False
    assert details["oacp_contingency_clamp_event"] is True
    assert planner.mpc.committed == []


def test_calibration_logging_computes_but_does_not_apply_a_bound():
    nominal = _mpc_result()
    planner = _orchestration_planner([nominal], calibration=True)
    result, details = planner._solve_oacp_reference(
        EgoState(0.0, 0.0, 0.0, 0.1),
        0.0,
        now=10.0,
    )
    assert result is nominal
    assert len(planner.mpc.calls) == 1
    assert "velocity_upper_bound" not in planner.mpc.calls[0]
    assert details["oacp_calibration_logging_only"] is True
    assert details["oacp_bound_applied"] is False


def test_zero_phantom_risk_releases_bound_without_contingency_solve():
    released = _mpc_result()
    planner = _orchestration_planner([released])
    planner.oacp_status.update(
        {
            "risk_total": 0.0,
            "exploration_velocity_bound": 0.15,
            "fallback_velocity_bound": 0.15,
        }
    )

    result, details = planner._solve_oacp_reference(
        EgoState(0.0, 0.0, 0.0, 0.1),
        0.0,
        now=10.0,
    )

    assert result is released
    assert len(planner.mpc.calls) == 1
    assert planner.mpc.calls[0]["velocity_upper_bound"] == 0.15
    assert details["oacp_contingency_applicable"] is False
    assert (
        details["oacp_contingency_not_applicable_reason"]
        == "NO_ACTIVE_PHANTOM_RISK"
    )


def test_valid_contingency_reuse_enforces_the_certified_remaining_prefix():
    exploration = _mpc_result()
    alternative = _mpc_result()
    cached_exploration = _mpc_result()
    planner = _orchestration_planner(
        [exploration, alternative, cached_exploration]
    )
    planner._solve_oacp_reference(
        EgoState(0.0, 0.0, 0.0, 0.1),
        0.0,
        now=10.0,
    )
    _confirm_prefix_forwarding(planner, cursor=0)
    result, details = planner._solve_oacp_reference(
        EgoState(0.02, 0.0, 0.0, 0.1),
        0.0,
        now=10.2,
    )
    assert result is cached_exploration
    assert len(planner.mpc.calls) == 3
    assert planner.mpc.calls[2]["commit_solution"] is True
    np.testing.assert_array_equal(
        planner.mpc.calls[2]["fixed_control_prefix"],
        exploration.controls[:, 1:2],
    )
    assert details["oacp_contingency_check_performed"] is False
    assert details["oacp_contingency_cached_valid"] is True
    assert details["oacp_cached_prefix_enforced"] is True
    assert details["oacp_cached_prefix_remaining_steps"] == 1


@pytest.mark.parametrize("ego_x", [0.0, 0.005])
def test_missing_hardware_ack_revokes_and_clamps_without_state_inference(
    ego_x,
):
    exploration = _mpc_result()
    alternative = _mpc_result()
    clamped = _mpc_result()
    planner = _orchestration_planner([exploration, alternative, clamped])
    ego = EgoState(0.0, 0.0, 0.0, 0.1)
    planner._solve_oacp_reference(ego, 0.0, now=10.0)
    planner.oacp_prefix_pending_control_stamp = _token(10.01)
    planner.oacp_prefix_pending_cursor = 0
    planner.hardware_gate_status = {
        "ready": False,
        "hardware_output_enabled": True,
        "candidate_receipt_stamp": 10.02,
        "forwarded_control_source_stamp": None,
    }
    planner.hardware_gate_status_receipt = 10.15

    result, details = planner._solve_oacp_reference(
        EgoState(ego_x, 0.0, 0.0, 0.1),
        0.0,
        now=10.2,
    )

    assert result is clamped
    assert planner.mpc.calls[2]["velocity_upper_bound"] == 0.08
    assert planner.oacp_contingency_cached_prefix_cursor == 0
    assert (
        details["oacp_prefix_execution_state"]
        == "PREFIX_EXECUTION_UNCONFIRMED_REVOKED"
    )
    assert details["oacp_contingency_clamp_event"] is True
    assert planner.oacp_contingency_cached_valid is False


def test_stale_matching_hardware_ack_revokes_and_clamps():
    exploration = _mpc_result()
    alternative = _mpc_result()
    clamped = _mpc_result()
    planner = _orchestration_planner([exploration, alternative, clamped])
    planner._solve_oacp_reference(
        EgoState(0.0, 0.0, 0.0, 0.1),
        0.0,
        now=10.0,
    )
    _confirm_prefix_forwarding(
        planner,
        cursor=0,
        publish=10.01,
        receipt=9.0,
    )

    result, details = planner._solve_oacp_reference(
        EgoState(0.005, 0.0, 0.0, 0.1),
        0.0,
        now=10.2,
    )

    assert result is clamped
    assert planner.mpc.calls[2]["velocity_upper_bound"] == 0.08
    assert (
        details["oacp_prefix_execution_state"]
        == "PREFIX_EXECUTION_UNCONFIRMED_REVOKED"
    )
    assert details["oacp_contingency_clamp_event"] is True
    assert planner.oacp_contingency_cached_valid is False


@pytest.mark.parametrize("ego_x", [0.0, 0.005])
def test_old_source_token_revokes_and_clamps_without_state_inference(
    ego_x,
):
    exploration = _mpc_result()
    alternative = _mpc_result()
    clamped = _mpc_result()
    planner = _orchestration_planner([exploration, alternative, clamped])
    ego = EgoState(0.0, 0.0, 0.0, 0.1)
    planner._solve_oacp_reference(ego, 0.0, now=10.0)
    planner.oacp_prefix_pending_control_stamp = _token(10.20)
    planner.oacp_prefix_pending_cursor = 0
    planner.hardware_gate_status = {
        "ready": True,
        "hardware_output_enabled": True,
        "candidate_receipt_stamp": 10.25,
        "forwarded_control_source_stamp": _token(10.01).as_mapping(),
    }
    planner.hardware_gate_status_receipt = 10.25

    result, details = planner._solve_oacp_reference(
        EgoState(ego_x, 0.0, 0.0, 0.1),
        0.0,
        now=10.3,
    )

    assert result is clamped
    assert planner.mpc.calls[2]["velocity_upper_bound"] == 0.08
    assert planner.oacp_contingency_cached_prefix_cursor == 0
    assert (
        details["oacp_prefix_execution_state"]
        == "PREFIX_EXECUTION_UNCONFIRMED_REVOKED"
    )
    assert details["oacp_contingency_clamp_event"] is True
    assert planner.oacp_contingency_cached_valid is False


@pytest.mark.parametrize(
    "partial_x",
    [0.005, 0.010, 0.015, 0.016, 0.018],
)
def test_exact_ack_with_partial_segment_progress_revokes_and_clamps(
    partial_x,
):
    exploration = _mpc_result()
    alternative = _mpc_result()
    clamped = _mpc_result()
    planner = _orchestration_planner([exploration, alternative, clamped])
    planner._solve_oacp_reference(
        EgoState(0.0, 0.0, 0.0, 0.1),
        0.0,
        now=10.0,
    )
    _confirm_prefix_forwarding(planner, cursor=0)

    result, details = planner._solve_oacp_reference(
        EgoState(partial_x, 0.0, 0.0, 0.1),
        0.0,
        now=10.2,
    )

    assert result is clamped
    assert planner.oacp_contingency_cached_prefix_cursor == 0
    assert (
        details["oacp_prefix_execution_state"]
        == "FORWARDED_PARTIAL_PREFIX_REVOKED"
    )
    assert planner.mpc.calls[2]["velocity_upper_bound"] == 0.08
    assert details["oacp_contingency_clamp_event"] is True
    assert details["oacp_contingency_valid"] is False


def test_position_progress_prevents_premature_advance_from_speed_and_yaw():
    exploration = _mpc_result()
    exploration.states[2, 1] = 0.13
    exploration.states[3, 1] = 0.04
    alternative = _mpc_result()
    clamped = _mpc_result()
    planner = _orchestration_planner([exploration, alternative, clamped])
    planner._solve_oacp_reference(
        EgoState(0.0, 0.0, 0.0, 0.1),
        0.0,
        now=10.0,
    )
    _confirm_prefix_forwarding(planner, cursor=0)

    _, details = planner._solve_oacp_reference(
        EgoState(0.01, 0.0, 0.04, 0.13),
        0.0,
        now=10.2,
    )

    assert (
        details["oacp_prefix_execution_state"]
        == "FORWARDED_PARTIAL_PREFIX_REVOKED"
    )
    assert details["oacp_contingency_clamp_event"] is True
    assert planner.oacp_contingency_cached_valid is False


def test_duplicate_forwarded_token_revokes_next_prefix_certificate():
    exploration = _mpc_result()
    alternative = _mpc_result()
    cached = _mpc_result()
    planner = _orchestration_planner([exploration, alternative, cached])
    planner._solve_oacp_reference(
        EgoState(0.0, 0.0, 0.0, 0.1),
        0.0,
        now=10.0,
    )
    first_token = _token(10.01)
    _confirm_prefix_forwarding(planner, cursor=0, publish=10.01)
    planner._solve_oacp_reference(
        EgoState(0.02, 0.0, 0.0, 0.1),
        0.0,
        now=10.2,
    )
    assert planner.oacp_contingency_cached_prefix_cursor == 1

    planner.oacp_prefix_pending_control_stamp = _token(10.21)
    planner.oacp_prefix_pending_cursor = 1
    planner.hardware_gate_status = {
        "ready": True,
        "hardware_output_enabled": True,
        "candidate_receipt_stamp": 10.30,
        "forwarded_control_source_stamp": first_token.as_mapping(),
    }
    planner.hardware_gate_status_receipt = 10.30

    state = planner._reconcile_oacp_prefix_execution(
        EgoState(0.02, 0.0, 0.0, 0.1),
        10.35,
    )
    assert state == "PREFIX_EXECUTION_UNCONFIRMED_REVOKED"
    assert planner.oacp_contingency_cached_valid is False
    assert planner.oacp_contingency_cached_prefix is None


def test_forwarded_command_off_certified_segment_revokes_and_clamps():
    exploration = _mpc_result()
    alternative = _mpc_result()
    clamped = _mpc_result()
    planner = _orchestration_planner([exploration, alternative, clamped])
    ego = EgoState(0.0, 0.0, 0.0, 0.1)
    planner._solve_oacp_reference(ego, 0.0, now=10.0)
    _confirm_prefix_forwarding(planner, cursor=0)

    result, details = planner._solve_oacp_reference(
        EgoState(0.0, 0.05, 0.0, 0.1),
        0.0,
        now=10.2,
    )

    assert result is clamped
    assert planner.mpc.calls[2]["velocity_upper_bound"] == 0.08
    assert (
        details["oacp_prefix_execution_state"]
        == "FORWARDED_EXECUTION_STATE_MISMATCH"
    )
    assert details["oacp_contingency_valid"] is False
    assert planner.oacp_contingency_cached_prefix is None


def test_consumed_certified_prefix_clamps_until_scheduled_branch_check():
    first_exploration = _mpc_result()
    first_alternative = _mpc_result()
    cached_exploration = _mpc_result()
    clamped = _mpc_result()
    refreshed_exploration = _mpc_result()
    refreshed_alternative = _mpc_result()
    planner = _orchestration_planner(
        [
            first_exploration,
            first_alternative,
            cached_exploration,
            clamped,
            refreshed_exploration,
            refreshed_alternative,
        ]
    )
    ego = EgoState(0.0, 0.0, 0.0, 0.1)
    planner._solve_oacp_reference(ego, 0.0, now=10.0)
    _confirm_prefix_forwarding(planner, cursor=0)
    planner._solve_oacp_reference(
        EgoState(0.02, 0.0, 0.0, 0.1),
        0.0,
        now=10.2,
    )
    _confirm_prefix_forwarding(
        planner,
        cursor=1,
        publish=10.21,
        receipt=10.35,
    )
    clamped_result, clamped_details = planner._solve_oacp_reference(
        EgoState(0.04, 0.0, 0.0, 0.1),
        0.0,
        now=10.4,
    )

    assert clamped_result is clamped
    assert len(planner.mpc.calls) == 4
    assert planner.mpc.calls[3]["velocity_upper_bound"] == 0.08
    assert clamped_details["oacp_contingency_check_performed"] is False
    assert clamped_details["oacp_contingency_clamp_event"] is True

    result, details = planner._solve_oacp_reference(
        EgoState(0.04, 0.0, 0.0, 0.1),
        0.0,
        now=11.0,
    )
    assert result is refreshed_exploration
    assert len(planner.mpc.calls) == 6
    assert planner.mpc.calls[4]["commit_solution"] is False
    np.testing.assert_array_equal(
        planner.mpc.calls[5]["fixed_control_prefix"],
        refreshed_exploration.controls[:, :2],
    )
    assert details["oacp_contingency_check_performed"] is True


def test_changed_visible_vehicle_invalidates_cached_contingency_and_clamps():
    exploration = _mpc_result()
    alternative = _mpc_result()
    clamped = _mpc_result()
    planner = _orchestration_planner([exploration, alternative, clamped])
    planner._solve_oacp_reference(
        EgoState(0.0, 0.0, 0.0, 0.1),
        0.0,
        now=10.0,
    )
    planner.vehicles = [
        Vehicle("merger", 1.0, 0.2, vx=0.1, vy=0.0)
    ]

    result, details = planner._solve_oacp_reference(
        EgoState(0.0, 0.0, 0.0, 0.1),
        0.0,
        now=10.2,
    )

    assert result is clamped
    assert len(planner.mpc.calls) == 3
    assert planner.mpc.calls[2]["velocity_upper_bound"] == 0.08
    assert details["oacp_contingency_check_performed"] is False
    assert details["oacp_contingency_cached_valid"] is False
    assert details["oacp_contingency_cache_context_match"] is False


def test_tightened_exploration_bound_invalidates_cache_and_clamps():
    exploration = _mpc_result()
    alternative = _mpc_result()
    clamped = _mpc_result()
    planner = _orchestration_planner([exploration, alternative, clamped])
    planner._solve_oacp_reference(
        EgoState(0.0, 0.0, 0.0, 0.1),
        0.0,
        now=10.0,
    )
    planner.oacp_status["exploration_velocity_bound"] = 0.10

    result, details = planner._solve_oacp_reference(
        EgoState(0.01, 0.0, 0.0, 0.11),
        0.0,
        now=10.2,
    )

    assert result is clamped
    assert len(planner.mpc.calls) == 3
    assert planner.mpc.calls[2]["velocity_upper_bound"] == 0.08
    assert details["oacp_contingency_check_performed"] is False
    assert details["oacp_contingency_cached_valid"] is False
    assert details["oacp_contingency_cache_context_match"] is False


def test_failed_cached_prefix_solve_revokes_certificate_and_clamps():
    exploration = _mpc_result()
    alternative = _mpc_result()
    failed_execution = _mpc_result(fallback=True)
    clamped = _mpc_result()
    planner = _orchestration_planner(
        [exploration, alternative, failed_execution, clamped]
    )
    ego = EgoState(0.0, 0.0, 0.0, 0.1)
    planner._solve_oacp_reference(ego, 0.0, now=10.0)
    _confirm_prefix_forwarding(planner, cursor=0)

    result, details = planner._solve_oacp_reference(
        EgoState(0.02, 0.0, 0.0, 0.1),
        0.0,
        now=10.2,
    )

    assert result is clamped
    assert planner.mpc.calls[2]["fixed_control_prefix"] is not None
    assert planner.mpc.calls[3]["velocity_upper_bound"] == 0.08
    assert details["oacp_contingency_valid"] is False
    assert details["oacp_contingency_clamp_event"] is True
    assert planner.oacp_contingency_cached_prefix is None
