import json

import pytest

from dream_limo.metrics_node import (
    AcceptedGoalRevision,
    ExperimentRunAccumulator,
    accepted_goal_revision,
    parse_json_object,
)


def _goal(revision=1, receipt=10.0, publication=10.1, x=3.0, y=0.0):
    return AcceptedGoalRevision(revision, receipt, publication, x, y)


def test_authorizer_goal_receipt_is_common_t0_and_parser_is_fail_safe():
    payload = {
        "goal_accepted": True,
        "goal_revision": 7,
        "goal_receipt_stamp": 12.5,
        "goal_publication_stamp": 12.6,
        "goal_x": 4.0,
        "goal_y": -0.2,
    }
    parsed = accepted_goal_revision(payload)
    assert parsed == AcceptedGoalRevision(7, 12.5, 12.6, 4.0, -0.2)
    assert parsed.identity == (7, 12.5)
    assert accepted_goal_revision({**payload, "goal_receipt_stamp": "nan"}) is None
    assert parse_json_object("{broken") is None
    assert parse_json_object("[]") is None
    assert parse_json_object(json.dumps(payload)) == payload


def test_ego_accumulator_reports_distance_mean_low_speed_and_fixed_time():
    run = ExperimentRunAccumulator(_goal(), fixed_time_seconds=2.0)
    run.configured_target_speed = 1.0
    assert not run.add_ego(9.9, 0.0, 0.0, 0.0)
    assert run.add_ego(10.0, 0.0, 0.0, 0.0)
    assert run.add_ego(11.0, 1.0, 0.0, 1.0)
    assert run.add_ego(12.0, 2.0, 0.0, 1.0)
    assert run.add_ego(13.0, 3.0, 0.0, 0.0)
    snapshot = run.snapshot(13.0)
    assert snapshot["run_elapsed_seconds"] == 3.0
    assert snapshot["traveled_distance"] == 3.0
    assert snapshot["time_weighted_mean_speed"] == pytest.approx(2.0 / 3.0)
    assert snapshot["time_below_half_target_speed"] == pytest.approx(1.0)
    assert snapshot["distance_to_goal_at_fixed_time"] == pytest.approx(1.0)


def test_completion_freezes_traversal_and_motion_acceptance_stamps():
    run = ExperimentRunAccumulator(_goal())
    run.accept_motion(10.4)
    run.accept_motion(11.0)
    run.complete(14.0)
    snapshot = run.snapshot(20.0)
    assert snapshot["goal_receipt_stamp"] == 10.0
    assert snapshot["goal_publication_stamp"] == 10.1
    assert snapshot["goal_motion_acceptance_stamp"] == 10.4
    assert snapshot["run_elapsed_seconds"] == 4.0
    assert snapshot["traversal_time_seconds"] == 4.0


def test_planner_accumulator_tracks_arm_oacp_solves_slack_and_contingency():
    run = ExperimentRunAccumulator(_goal(), slack_threshold=0.01)
    run.update_planner(
        {
            "arm": "oacp_vb",
            "shared_controller_parameter_hash": "a" * 64,
            "configured_target_speed": 0.2,
            "navigation_goal_remaining": 2.5,
            "oacp_risk_total": 1.2,
            "oacp_exploration_velocity_bound": 0.12,
            "oacp_fallback_velocity_bound": 0.16,
            "oacp_pvs_component_count": 2,
            "oacp_frs_intersects_trajectory": True,
            "oacp_exploration_solve_seconds": 0.02,
            "oacp_fallback_solve_seconds": 0.03,
            "t_mpc_total": 0.05,
            "maximum_velocity_slack": 0.04,
            "maximum_future_velocity_slack": 0.02,
            "oacp_contingency_valid": False,
            "oacp_contingency_clamp_event": True,
        }
    )
    run.update_planner(
        {
            "oacp_exploration_solve_seconds": 0.01,
            "t_mpc_total": 0.01,
            "maximum_velocity_slack": 0.0,
            "maximum_future_velocity_slack": 0.0,
        }
    )
    snapshot = run.snapshot(11.0)
    assert snapshot["planner_arm"] == "oacp_vb"
    assert snapshot["shared_controller_parameter_hash"] == "a" * 64
    assert snapshot["configured_target_speed"] == 0.2
    assert snapshot["current_goal_remaining"] == 2.5
    assert snapshot["oacp_risk_total"] == 1.2
    assert snapshot["oacp_pvs_component_count"] == 2.0
    assert snapshot["oacp_frs_intersects_trajectory"] is True
    assert snapshot["mpc_solve_count"] == 3
    assert snapshot["mpc_solve_mean_seconds"] == pytest.approx(0.02)
    assert snapshot["mpc_solve_max_seconds"] == 0.03
    assert snapshot["mpc_exploration_solve_count"] == 2
    assert snapshot["mpc_fallback_solve_count"] == 1
    assert snapshot["mpc_cycle_total_count"] == 2
    assert snapshot["velocity_slack_activation_samples"] == 1
    assert snapshot["velocity_slack_activation_events"] == 1
    assert snapshot["velocity_slack_maximum"] == 0.04
    assert snapshot["future_velocity_slack_activation_samples"] == 1
    assert snapshot["future_velocity_slack_maximum"] == 0.02
    assert snapshot["contingency_failure_count"] == 1
    assert snapshot["contingency_clamp_event_count"] == 1


def test_safety_false_transitions_are_edges_not_false_samples():
    run = ExperimentRunAccumulator(_goal())
    previous = None
    for current in (True, False, False, True, False):
        run.record_safety_transition(previous, current)
        previous = current
    assert run.snapshot(11.0)["safety_false_transition_count"] == 2


def test_new_revision_uses_a_fresh_accumulator():
    first = ExperimentRunAccumulator(_goal(revision=1))
    first.update_planner({"t_mpc": 0.02, "maximum_velocity_slack": 0.1})
    second = ExperimentRunAccumulator(
        _goal(revision=2, receipt=20.0, publication=20.1)
    )
    snapshot = second.snapshot(20.0)
    assert snapshot["goal_revision"] == 2
    assert snapshot["mpc_solve_count"] == 0
    assert snapshot["velocity_slack_activation_samples"] == 0


def test_malformed_planner_fields_are_ignored_without_poisoning_metrics():
    run = ExperimentRunAccumulator(_goal())
    run.update_planner(
        {
            "configured_target_speed": "nan",
            "navigation_goal_remaining": {},
            "t_mpc": "broken",
            "maximum_velocity_slack": float("inf"),
            "oacp_risk_total": -1.0,
        }
    )
    snapshot = run.snapshot(11.0)
    assert snapshot["configured_target_speed"] is None
    assert snapshot["current_goal_remaining"] is None
    assert snapshot["mpc_solve_count"] == 0
    assert snapshot["velocity_slack_activation_samples"] == 0
    assert "oacp_risk_total" not in snapshot
