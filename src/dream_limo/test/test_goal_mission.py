from dataclasses import replace
from math import cos, sin

import pytest

from dream_limo.core.goal_mission import (
    EgoMissionState,
    GoalMissionLatch,
    GoalRequest,
    PlannerGoalReadiness,
    PreflightReadiness,
    evaluate_goal_authorization,
    goal_mission_config_from_deployment,
    nearest_lane,
    validate_goal_request,
)
from dream_limo.limo_scale import default_deployment_config


NOW = 100.0


def _config(**overrides):
    base = goal_mission_config_from_deployment(default_deployment_config())
    return replace(base, **overrides)


def _ego(**overrides):
    values = {
        "x": 0.35,
        "y": 0.45,
        "speed": 0.0,
        "source_stamp": 99.95,
        "receipt_stamp": 99.98,
    }
    values.update(overrides)
    return EgoMissionState(**values)


def _goal(**overrides):
    values = {
        "frame_id": "map",
        "x": 5.55,
        "y": 0.03,
        "z": 0.0,
        "qx": 0.0,
        "qy": 0.0,
        "qz": sin(0.25),
        "qw": cos(0.25),
        "source_stamp": 99.95,
        "receipt_stamp": 99.99,
    }
    values.update(overrides)
    return GoalRequest(**values)


def _validate(goal=None, ego=None, config=None):
    return validate_goal_request(
        goal or _goal(),
        _ego() if ego is None else ego,
        now=NOW,
        config=config or _config(),
    )


def test_valid_goal_is_snapped_to_adjacent_lane_and_lane_heading():
    result = _validate()
    assert result.accepted
    assert result.reason == "GOAL_ACCEPTED"
    assert result.target_lane == 1
    assert result.goal_x == pytest.approx(5.55)
    assert result.goal_y == pytest.approx(0.0)
    assert result.goal_yaw == pytest.approx(0.0)
    assert result.goal_source_age == pytest.approx(0.05)


@pytest.mark.parametrize(
    ("goal", "reason"),
    (
        (_goal(frame_id="odom"), "GOAL_FRAME_MISMATCH"),
        (_goal(x=float("nan")), "NONFINITE_GOAL"),
        (_goal(qw=0.5, qz=0.0), "GOAL_QUATERNION_NOT_NORMALIZED"),
        (_goal(qx=sin(0.1), qw=cos(0.1), qz=0.0), "GOAL_ORIENTATION_NOT_PLANAR"),
        (_goal(z=0.3), "GOAL_OUTSIDE_MAP_PLANE"),
        (_goal(x=5.85), "GOAL_FOOTPRINT_OUT_OF_BOUNDS"),
        (_goal(y=0.225), "GOAL_NOT_NEAR_LANE"),
        (_goal(source_stamp=98.0), "STALE_GOAL_SOURCE"),
        (_goal(receipt_stamp=98.0), "STALE_GOAL_RECEIPT"),
        (_goal(source_stamp=100.2), "STALE_GOAL_SOURCE"),
    ),
)
def test_invalid_or_stale_goal_fails_closed(goal, reason):
    result = _validate(goal=goal)
    assert not result.accepted
    assert result.reason == reason


@pytest.mark.parametrize(
    ("ego", "reason"),
    (
        (None, "EGO_UNAVAILABLE"),
        (_ego(speed=0.031), "EGO_NOT_STOPPED"),
        (_ego(source_stamp=98.0), "STALE_EGO_SOURCE"),
        (_ego(receipt_stamp=98.0), "STALE_EGO_RECEIPT"),
        (_ego(y=0.225), "EGO_NOT_NEAR_LANE"),
    ),
)
def test_unavailable_stale_or_moving_ego_fails_closed(ego, reason):
    # ``_validate`` uses its default ego when passed None, so call directly for
    # the deliberately unavailable case.
    result = validate_goal_request(
        _goal(), ego, now=NOW, config=_config()
    )
    assert not result.accepted
    assert result.reason == reason


def test_goal_must_be_ahead_and_lane_change_must_end_after_conflict_zone():
    too_close = _validate(goal=_goal(x=0.84, y=0.45))
    assert not too_close.accepted
    assert too_close.reason == "GOAL_NOT_FAR_ENOUGH_AHEAD"

    before_conflict_exit = _validate(goal=_goal(x=5.20, y=0.0))
    assert not before_conflict_exit.accepted
    assert before_conflict_exit.reason == "LANE_CHANGE_GOAL_BEFORE_CONFLICT_EXIT"

    # Same-lane goals need not pass through the merge/conflict region.
    same_lane = _validate(goal=_goal(x=1.20, y=0.45))
    assert same_lane.accepted


def test_nonadjacent_lane_goal_is_rejected():
    result = _validate(goal=_goal(y=-0.45))
    assert not result.accepted
    assert result.reason == "NONADJACENT_LANE_GOAL"


def test_nearest_lane_requires_configured_tolerance():
    assert nearest_lane(0.02, (0.45, 0.0, -0.45), 0.18) == 1
    assert nearest_lane(0.24, (0.45, 0.0, -0.45), 0.18) is None


def test_one_shot_latch_allows_correction_but_never_replaces_accepted_goal():
    latch = GoalMissionLatch()
    invalid = _validate(goal=_goal(frame_id="odom"))
    assert not latch.consider(invalid)
    assert latch.goal_received and not latch.active

    accepted = _validate()
    assert latch.consider(accepted)
    assert latch.active
    assert not latch.consider(_validate(goal=_goal(x=5.70)))
    assert latch.accepted_goal == accepted
    assert latch.reason == "GOAL_ALREADY_ACCEPTED"

    latch.complete()
    assert not latch.active
    assert latch.reason == "MISSION_COMPLETE"


def test_stop_latches_and_prevents_future_activation():
    latch = GoalMissionLatch()
    latch.stop()
    assert latch.stop_latched and not latch.active
    assert not latch.consider(_validate())
    assert latch.reason == "STOP_LATCHED"


def test_authorization_waits_for_fresh_preflight_and_matching_planner():
    latch = GoalMissionLatch()
    assert latch.consider(_validate())
    ego = _ego(source_stamp=100.0, receipt_stamp=100.0)
    preflight = PreflightReadiness(True, NOW)
    planner = PlannerGoalReadiness(True, 5.55, 1, NOW)

    waiting_preflight = evaluate_goal_authorization(
        latch,
        ego,
        planner,
        None,
        now=NOW,
        config=_config(),
        planner_timeout=0.75,
        preflight_timeout=2.0,
    )
    assert not waiting_preflight.armed
    assert waiting_preflight.reason == "WAITING_FOR_PREFLIGHT"

    waiting_planner = evaluate_goal_authorization(
        latch,
        ego,
        PlannerGoalReadiness(True, 5.55, 0, NOW),
        preflight,
        now=NOW,
        config=_config(),
        planner_timeout=0.75,
        preflight_timeout=2.0,
    )
    assert not waiting_planner.armed
    assert waiting_planner.reason == "WAITING_FOR_PLANNER"

    active = evaluate_goal_authorization(
        latch,
        ego,
        planner,
        preflight,
        now=NOW,
        config=_config(),
        planner_timeout=0.75,
        preflight_timeout=2.0,
    )
    assert active.ready and active.armed
    assert active.reason == "GOAL_ACTIVE"


def test_authorization_disarms_for_stale_ego_complete_or_stop():
    latch = GoalMissionLatch()
    assert latch.consider(_validate())
    planner = PlannerGoalReadiness(True, 5.55, 1, NOW)
    preflight = PreflightReadiness(True, NOW)

    stale = evaluate_goal_authorization(
        latch,
        _ego(source_stamp=99.0, receipt_stamp=NOW),
        planner,
        preflight,
        now=NOW,
        config=_config(),
        planner_timeout=0.75,
        preflight_timeout=2.0,
    )
    assert not stale.armed and stale.reason == "STALE_EGO"

    latch.complete()
    complete = evaluate_goal_authorization(
        latch,
        _ego(source_stamp=NOW, receipt_stamp=NOW),
        planner,
        preflight,
        now=NOW,
        config=_config(),
        planner_timeout=0.75,
        preflight_timeout=2.0,
    )
    assert not complete.armed and complete.reason == "MISSION_COMPLETE"


def test_authorization_disarms_for_stale_readiness_heartbeats():
    latch = GoalMissionLatch()
    assert latch.consider(_validate())
    ego = _ego(source_stamp=NOW, receipt_stamp=NOW)

    stale_preflight = evaluate_goal_authorization(
        latch,
        ego,
        PlannerGoalReadiness(True, 5.55, 1, NOW),
        PreflightReadiness(True, NOW - 2.0),
        now=NOW,
        config=_config(),
        planner_timeout=0.75,
        preflight_timeout=2.0,
    )
    assert not stale_preflight.armed
    assert stale_preflight.reason == "WAITING_FOR_PREFLIGHT"

    stale_planner = evaluate_goal_authorization(
        latch,
        ego,
        PlannerGoalReadiness(True, 5.55, 1, NOW - 0.75),
        PreflightReadiness(True, NOW),
        now=NOW,
        config=_config(),
        planner_timeout=0.75,
        preflight_timeout=2.0,
    )
    assert not stale_planner.armed
    assert stale_planner.reason == "WAITING_FOR_PLANNER"
