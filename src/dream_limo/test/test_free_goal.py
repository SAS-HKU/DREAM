from dataclasses import replace
from math import cos, sin

import pytest

from dream_limo.core.free_goal import (
    CostmapSnapshot,
    FreeGoalConfig,
    FreeGoalEgoState,
    FreeGoalMissionLatch,
    FreeGoalPlannerReadiness,
    FreeGoalPreflightReadiness,
    FreeGoalRequest,
    FreeGoalValidation,
    evaluate_free_goal_authorization,
    transform_planar_goal,
    validate_free_goal_request,
)
from dream_limo.free_goal_authorizer_node import DreamFreeGoalAuthorizerNode


NOW = 100.0


def _config(**overrides):
    return replace(
        FreeGoalConfig(frame_id="map", footprint_clearance=0.11),
        **overrides,
    )


def _costmap(*, data=None, frame_id="map", source_stamp=99.95, receipt_stamp=99.98):
    width = height = 20
    return CostmapSnapshot.from_sequence(
        frame_id=frame_id,
        width=width,
        height=height,
        resolution=0.10,
        origin_x=-1.0,
        origin_y=-1.0,
        origin_yaw=0.0,
        data=[0] * (width * height) if data is None else data,
        source_stamp=source_stamp,
        receipt_stamp=receipt_stamp,
    )


def _ego(**overrides):
    values = {
        "frame_id": "map",
        "x": 0.0,
        "y": 0.0,
        "source_stamp": 99.95,
        "receipt_stamp": 99.98,
    }
    values.update(overrides)
    return FreeGoalEgoState(**values)


def _goal(**overrides):
    values = {
        "frame_id": "map",
        "x": -0.42,
        "y": 0.31,
        "z": 0.0,
        "qx": 0.0,
        "qy": 0.0,
        "qz": sin(0.35),
        "qw": cos(0.35),
        "source_stamp": 99.95,
        "receipt_stamp": 99.99,
    }
    values.update(overrides)
    return FreeGoalRequest(**values)


def _validate(*, goal=None, ego=None, costmap=None, config=None):
    return validate_free_goal_request(
        _goal() if goal is None else goal,
        _ego() if ego is None else ego,
        _costmap() if costmap is None else costmap,
        now=NOW,
        config=_config() if config is None else config,
    )


def test_arbitrary_observed_free_goal_is_preserved_without_lane_or_x_snap():
    result = _validate()

    assert result.accepted
    assert result.reason == "GOAL_ACCEPTED"
    assert result.goal_x == pytest.approx(-0.42)
    assert result.goal_y == pytest.approx(0.31)
    assert result.goal_yaw == pytest.approx(0.70)


@pytest.mark.parametrize("value", (-1, 1, 50, 100))
def test_unknown_or_any_positive_cost_under_clearance_rejects_goal(value):
    data = [0] * 400
    # Goal (-0.42, 0.31) lies in cell (5,13); its 0.11 m disk also touches this
    # adjacent cell, proving the validator checks the footprint, not one center.
    data[13 * 20 + 6] = value
    result = _validate(costmap=_costmap(data=data))

    assert not result.accepted
    assert result.reason == ("GOAL_IN_UNKNOWN" if value < 0 else "GOAL_NOT_FREE")
    assert result.blocking_value == value


def test_complete_clearance_disk_must_remain_inside_costmap():
    result = _validate(goal=_goal(x=0.95, y=0.0))
    assert not result.accepted
    assert result.reason == "GOAL_FOOTPRINT_OUTSIDE_COSTMAP"


@pytest.mark.parametrize(
    ("goal", "ego", "costmap", "reason"),
    (
        (_goal(frame_id="base_link"), _ego(), _costmap(), "GOAL_FRAME_MISMATCH"),
        (_goal(x=float("nan")), _ego(), _costmap(), "NONFINITE_GOAL"),
        (_goal(source_stamp=98.0), _ego(), _costmap(), "STALE_GOAL_SOURCE"),
        (_goal(), None, _costmap(), "EGO_UNAVAILABLE"),
        (_goal(), _ego(frame_id="odom"), _costmap(), "EGO_FRAME_MISMATCH"),
        (_goal(), _ego(source_stamp=98.0), _costmap(), "STALE_EGO_SOURCE"),
        (_goal(), _ego(), _costmap(frame_id="odom"), "COSTMAP_FRAME_MISMATCH"),
        (
            _goal(),
            _ego(),
            _costmap(source_stamp=98.0),
            "STALE_COSTMAP_SOURCE",
        ),
    ),
)
def test_goal_ego_and_costmap_contracts_fail_closed(goal, ego, costmap, reason):
    result = validate_free_goal_request(
        goal, ego, costmap, now=NOW, config=_config()
    )
    assert not result.accepted
    assert result.reason == reason


def test_planar_odom_goal_transform_preserves_arbitrary_pose():
    transformed = transform_planar_goal(
        _goal(frame_id="odom", x=1.0, y=0.0, qz=0.0, qw=1.0),
        target_frame="map",
        translation_x=2.0,
        translation_y=-0.5,
        translation_z=0.0,
        transform_yaw=0.5,
    )

    assert transformed.frame_id == "map"
    assert transformed.x == pytest.approx(2.0 + cos(0.5))
    assert transformed.y == pytest.approx(-0.5 + sin(0.5))
    assert transformed.qz == pytest.approx(sin(0.25))
    assert transformed.qw == pytest.approx(cos(0.25))


def test_nonplanar_transform_and_nonplanar_source_goal_are_rejected():
    with pytest.raises(ValueError, match="GOAL_TF_NOT_PLANAR"):
        transform_planar_goal(
            _goal(frame_id="odom"),
            target_frame="map",
            translation_x=0.0,
            translation_y=0.0,
            translation_z=0.0,
            transform_yaw=0.0,
            transform_roll=0.2,
        )
    with pytest.raises(ValueError, match="GOAL_ORIENTATION_NOT_PLANAR"):
        transform_planar_goal(
            _goal(frame_id="odom", qx=sin(0.1), qz=0.0, qw=cos(0.1)),
            target_frame="map",
            translation_x=0.0,
            translation_y=0.0,
            translation_z=0.0,
            transform_yaw=0.0,
        )
    with pytest.raises(ValueError, match="GOAL_TF_NOT_PLANAR"):
        transform_planar_goal(
            _goal(frame_id="odom"),
            target_frame="map",
            translation_x=0.0,
            translation_y=0.0,
            translation_z=0.0,
            transform_yaw=0.0,
            transform_roll=0.01,
            maximum_transform_tilt=1.0e-6,
        )


def test_invalid_replacement_cancels_old_goal_and_valid_goal_increments_revision():
    first = _validate()
    second = _validate(goal=_goal(x=0.45, y=-0.25))
    latch = FreeGoalMissionLatch()

    assert latch.consider(first)
    assert latch.revision == 1
    assert not latch.consider(FreeGoalValidation(False, "GOAL_NOT_FREE"))
    assert latch.accepted_goal is None
    assert latch.revision == 1
    assert latch.consider(second)
    assert latch.accepted_goal == second
    assert latch.revision == 2


def test_authorization_requires_fresh_matching_planner_after_replacement():
    first = _validate()
    second = _validate(goal=_goal(x=0.45, y=-0.25))
    latch = FreeGoalMissionLatch()
    assert latch.consider(first)
    ego = _ego(source_stamp=NOW, receipt_stamp=NOW)
    costmap = _costmap(source_stamp=NOW, receipt_stamp=NOW)
    preflight = FreeGoalPreflightReadiness(True, NOW)
    first_planner = FreeGoalPlannerReadiness(
        True, first.goal_x, first.goal_y, NOW
    )

    active = evaluate_free_goal_authorization(
        latch,
        ego,
        costmap,
        first_planner,
        preflight,
        now=NOW,
        config=_config(),
    )
    assert active.armed and active.reason == "GOAL_ACTIVE"

    assert latch.consider(second)
    stale_match = evaluate_free_goal_authorization(
        latch,
        ego,
        costmap,
        first_planner,
        preflight,
        now=NOW,
        config=_config(),
    )
    assert not stale_match.armed
    assert stale_match.reason == "WAITING_FOR_PLANNER"

    second_planner = FreeGoalPlannerReadiness(
        True, second.goal_x, second.goal_y, NOW
    )
    replaced_active = evaluate_free_goal_authorization(
        latch,
        ego,
        costmap,
        second_planner,
        preflight,
        now=NOW,
        config=_config(),
    )
    assert replaced_active.armed


def test_active_goal_dearms_if_latest_costmap_no_longer_reports_free():
    validation = _validate()
    latch = FreeGoalMissionLatch()
    assert latch.consider(validation)
    data = [0] * 400
    data[13 * 20 + 6] = 100
    authorization = evaluate_free_goal_authorization(
        latch,
        _ego(source_stamp=NOW, receipt_stamp=NOW),
        _costmap(data=data, source_stamp=NOW, receipt_stamp=NOW),
        FreeGoalPlannerReadiness(
            True, validation.goal_x, validation.goal_y, NOW
        ),
        FreeGoalPreflightReadiness(True, NOW),
        now=NOW,
        config=_config(),
    )

    assert not authorization.armed
    assert authorization.reason == "GOAL_NOT_FREE"


def test_planner_status_parser_requires_both_goal_coordinates():
    assert DreamFreeGoalAuthorizerNode._planner_goal(
        {"navigation_goal_x": 0.4, "navigation_goal_y": -0.2}
    ) == (0.4, -0.2)
    assert DreamFreeGoalAuthorizerNode._planner_goal(
        {"mission_goal_x": 0.4}
    ) == (None, None)
