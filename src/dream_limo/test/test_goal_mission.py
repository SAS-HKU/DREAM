from dataclasses import replace
from math import cos, sin

import pytest
from builtin_interfaces.msg import Time
from std_msgs.msg import String

from dream_limo.core.goal_mission import (
    EgoMissionState,
    GoalMissionLatch,
    GoalRequest,
    PlannerGoalReadiness,
    PreflightReadiness,
    evaluate_goal_authorization,
    goal_mission_config_from_deployment,
    nearest_lane,
    validate_configured_auto_goal,
    validate_goal_request,
)
from dream_limo.limo_scale import default_deployment_config
from dream_limo.goal_authorizer_node import DreamGoalAuthorizerNode


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


def test_configured_auto_goal_uses_arena_destination_and_same_validator():
    result = validate_configured_auto_goal(
        _ego(),
        now=NOW,
        config=_config(),
        mission_goal_x=5.55,
        target_lane=1,
    )
    assert result.accepted
    assert result.reason == "GOAL_ACCEPTED"
    assert result.goal_x == pytest.approx(5.55)
    assert result.goal_y == pytest.approx(0.0)
    assert result.target_lane == 1


@pytest.mark.parametrize("target_lane", (-1, 3, 1.0, True, "1"))
def test_configured_auto_goal_rejects_invalid_target_lane(target_lane):
    result = validate_configured_auto_goal(
        _ego(),
        now=NOW,
        config=_config(),
        mission_goal_x=5.55,
        target_lane=target_lane,
    )
    assert not result.accepted
    assert result.reason == "AUTO_TARGET_LANE_INVALID"


@pytest.mark.parametrize("goal_x", (None, "bad", float("inf"), float("nan")))
def test_configured_auto_goal_rejects_malformed_destination(goal_x):
    result = validate_configured_auto_goal(
        _ego(),
        now=NOW,
        config=_config(),
        mission_goal_x=goal_x,
        target_lane=1,
    )
    assert not result.accepted
    assert result.reason == "NONFINITE_GOAL"


def test_configured_auto_goal_cannot_bypass_stop_or_adjacent_lane_validation():
    moving = validate_configured_auto_goal(
        _ego(speed=0.04),
        now=NOW,
        config=_config(),
        mission_goal_x=5.55,
        target_lane=1,
    )
    assert not moving.accepted
    assert moving.reason == "EGO_NOT_STOPPED"

    nonadjacent = validate_configured_auto_goal(
        _ego(),
        now=NOW,
        config=_config(),
        mission_goal_x=5.55,
        target_lane=2,
    )
    assert not nonadjacent.accepted
    assert nonadjacent.reason == "NONADJACENT_LANE_GOAL"

    latch = GoalMissionLatch()
    latch.stop()
    assert not latch.consider(
        validate_configured_auto_goal(
            _ego(),
            now=NOW,
            config=_config(),
            mission_goal_x=5.55,
            target_lane=1,
        )
    )
    assert latch.stop_latched
    assert latch.reason == "STOP_LATCHED"


class _RecordingPublisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class _RecordingLogger:
    def __init__(self):
        self.info_messages = []
        self.warning_messages = []

    def info(self, message):
        self.info_messages.append(message)

    def warning(self, message):
        self.warning_messages.append(message)


class _FixedClock:
    class _Instant:
        @staticmethod
        def to_msg():
            return Time(sec=int(NOW), nanosec=0)

    @staticmethod
    def now():
        return _FixedClock._Instant()


class _AutoStartPlannerHandshakeHarness:
    def __init__(self):
        self.enabled = True
        self.auto_start = True
        self.planner_waiting_for_goal = False
        self.planner = None
        self.state = GoalMissionLatch()
        self.ego = _ego()
        self.contract = _config()
        self.configured_goal_x = 5.55
        self.configured_target_lane = 1
        self.accepted_goal_source_stamp = None
        self.accepted_goal_receipt_stamp = None
        self.accepted_goal_source = "waiting"
        self.mission_goal_publisher = _RecordingPublisher()
        self.logger = _RecordingLogger()
        self.heartbeat_count = 0

    def _now(self):
        return NOW

    def _try_auto_start(self):
        DreamGoalAuthorizerNode._try_auto_start(self)

    def _consider_and_publish_goal(self, *args, **kwargs):
        return DreamGoalAuthorizerNode._consider_and_publish_goal(
            self, *args, **kwargs
        )

    def _publish_heartbeat(self):
        self.heartbeat_count += 1

    @staticmethod
    def get_clock():
        return _FixedClock()

    def get_logger(self):
        return self.logger


def test_auto_start_waits_for_planner_waiting_heartbeat_before_goal_publication():
    harness = _AutoStartPlannerHandshakeHarness()
    unrelated = String(
        data=(
            '{"ready":false,"reason":"STALE_INPUT",'
            '"mission_goal_required":true,"mission_goal_received":false,'
            '"mission_goal_x":5.55,"mission_goal_target_lane":1}'
        )
    )
    DreamGoalAuthorizerNode._on_planner_status(harness, unrelated)
    assert not harness.planner_waiting_for_goal
    assert not harness.state.active
    assert harness.mission_goal_publisher.messages == []

    waiting = String(
        data=(
            '{"ready":false,"reason":"WAITING_FOR_MISSION_GOAL",'
            '"mission_goal_required":true,"mission_goal_received":false,'
            '"mission_goal_x":5.55,"mission_goal_target_lane":1}'
        )
    )
    DreamGoalAuthorizerNode._on_planner_status(harness, waiting)
    assert harness.planner_waiting_for_goal
    assert harness.state.active
    assert harness.accepted_goal_source == "auto_forward"
    assert len(harness.mission_goal_publisher.messages) == 1
    goal = harness.mission_goal_publisher.messages[0]
    assert goal.header.frame_id == "map"
    assert goal.pose.position.x == pytest.approx(5.55)
    assert goal.pose.position.y == pytest.approx(0.0)

    # Repeated planner heartbeats cannot republish or replace the one-shot goal.
    DreamGoalAuthorizerNode._on_planner_status(harness, waiting)
    assert len(harness.mission_goal_publisher.messages) == 1


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
