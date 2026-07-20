from math import isclose
from types import SimpleNamespace

import pytest
from geometry_msgs.msg import PoseStamped

from dream_limo.core.goal_mission import goal_mission_config_from_deployment
from dream_limo.core.mpc import RiskAwareMPC
from dream_limo.core.risk_field import DREAMRiskField
from dream_limo.core.types import EgoState
from dream_limo.limo_scale import default_deployment_config
from dream_limo.planner_node import DreamPlannerNode, deployment_for_mission_goal


NOW = 100.0


class _Logger:
    def info(self, _message):
        pass

    def warning(self, _message):
        pass

    def error(self, _message):
        pass


class _CallbackHarness:
    def __init__(self, *, ego_speed=0.0):
        self.config = default_deployment_config()
        self.goal_contract = goal_mission_config_from_deployment(self.config)
        self.require_mission_goal = True
        self.mission_goal_received = False
        self.pending_mission_goal = None
        self.mission_goal_source = "waiting"
        self.mission_goal_last_rejection = ""
        self.route_target_lane = self.config.arena.target_lane
        self.ego = EgoState(
            x=0.35,
            y=self.config.arena.lane_centers[self.config.arena.ego_lane],
            yaw=0.0,
            speed=ego_speed,
            stamp=NOW - 0.02,
            lane_index=self.config.arena.ego_lane,
        )
        self.ego_receipt = NOW - 0.01
        self.activation_calls = []
        self.stop_reasons = []
        self.mpc = SimpleNamespace(reset=lambda: None)
        self._logger = _Logger()

    def _now(self):
        return NOW

    def get_parameter(self, name):
        assert name == "mission_goal_topic"
        return SimpleNamespace(value="/dream/mission_goal")

    def get_logger(self):
        return self._logger

    def _publish_stop(self, reason, _details=None):
        self.stop_reasons.append(reason)

    def _consider_pending_mission_goal(self):
        DreamPlannerNode._consider_pending_mission_goal(self)

    def _activate_mission_goal(self, *, goal_x, target_lane):
        self.activation_calls.append((goal_x, target_lane))
        self.config = deployment_for_mission_goal(
            self.config, goal_x=goal_x, target_lane=target_lane
        )
        self.route_target_lane = target_lane


def _goal(*, x=5.55, y=0.0, frame_id="map"):
    message = PoseStamped()
    message.header.frame_id = frame_id
    message.header.stamp.sec = int(NOW)
    message.pose.position.x = x
    message.pose.position.y = y
    message.pose.orientation.w = 1.0
    return message


def test_dynamic_goal_rebuilds_only_route_geometry():
    config = default_deployment_config()
    updated = deployment_for_mission_goal(config, goal_x=5.65, target_lane=1)

    assert updated.arena.mission_goal_x == pytest.approx(5.65)
    assert updated.arena.target_lane == 1
    assert updated.grid is config.grid
    assert updated.pde is config.pde
    assert updated.mpc is config.mpc


def test_same_lane_forward_goal_is_valid_but_early_lane_change_is_not():
    config = default_deployment_config()
    same_lane = deployment_for_mission_goal(config, goal_x=1.20, target_lane=0)
    assert same_lane.arena.mission_goal_x == pytest.approx(1.20)
    assert same_lane.arena.target_lane == same_lane.arena.ego_lane

    with pytest.raises(ValueError, match="lane-change mission goal"):
        deployment_for_mission_goal(config, goal_x=5.20, target_lane=1)


def test_planner_activation_preserves_live_risk_and_rebuilds_controllers():
    config = default_deployment_config()
    holder = SimpleNamespace(
        config=config,
        field=DREAMRiskField(config),
        decision=None,
        mpc=RiskAwareMPC(config, enforce_map_bounds=True),
        mission=None,
        route_target_lane=config.arena.target_lane,
        enforce_map_bounds=True,
        get_parameter=lambda name: SimpleNamespace(value=2.5),
    )
    holder.field.R[20, 30] = 1.75
    field_identity = id(holder.field)

    DreamPlannerNode._activate_mission_goal(
        holder, goal_x=5.65, target_lane=1
    )

    assert id(holder.field) == field_identity
    assert isclose(holder.field.R[20, 30], 1.75)
    assert holder.field.config is holder.config
    assert holder.decision.config is holder.config
    assert holder.mpc.deployment is holder.config
    assert holder.mpc.enforce_map_bounds
    assert holder.mission.goal_x == pytest.approx(5.65)
    assert holder.route_target_lane == 1


def test_planner_independently_accepts_one_fresh_stopped_goal_only_once():
    planner = _CallbackHarness()
    DreamPlannerNode._on_mission_goal(planner, _goal())

    assert planner.mission_goal_received
    assert planner.mission_goal_source == "/dream/mission_goal"
    assert planner.activation_calls == [(5.55, 1)]
    assert planner.stop_reasons == []

    DreamPlannerNode._on_mission_goal(planner, _goal(x=5.70))
    assert planner.activation_calls == [(5.55, 1)]
    assert planner.config.arena.mission_goal_x == pytest.approx(5.55)


def test_planner_rejects_goal_while_ego_is_moving_and_remains_waiting():
    planner = _CallbackHarness(ego_speed=0.04)
    DreamPlannerNode._on_mission_goal(planner, _goal())

    assert not planner.mission_goal_received
    assert planner.activation_calls == []
    assert planner.mission_goal_last_rejection == "EGO_NOT_STOPPED"
    assert planner.stop_reasons == ["EGO_NOT_STOPPED"]


def test_planner_rejects_non_map_goal_independently():
    planner = _CallbackHarness()
    DreamPlannerNode._on_mission_goal(planner, _goal(frame_id="odom"))

    assert not planner.mission_goal_received
    assert planner.activation_calls == []
    assert planner.mission_goal_last_rejection == "GOAL_FRAME_MISMATCH"
    assert planner.stop_reasons == ["GOAL_FRAME_MISMATCH"]


def test_required_planner_publishes_waiting_stop_before_any_goal():
    planner = _CallbackHarness()
    DreamPlannerNode._plan(planner)
    assert planner.stop_reasons == ["WAITING_FOR_MISSION_GOAL"]


def test_fresh_goal_is_retried_after_cross_topic_ego_delivery_race():
    planner = _CallbackHarness()
    planner.ego = None
    planner.ego_receipt = None
    DreamPlannerNode._on_mission_goal(planner, _goal())

    assert not planner.mission_goal_received
    assert planner.pending_mission_goal is not None
    assert planner.stop_reasons == ["EGO_UNAVAILABLE"]

    planner.ego = EgoState(
        x=0.35,
        y=0.45,
        yaw=0.0,
        speed=0.0,
        stamp=NOW - 0.02,
        lane_index=0,
    )
    planner.ego_receipt = NOW - 0.01
    planner._consider_pending_mission_goal()

    assert planner.mission_goal_received
    assert planner.pending_mission_goal is None
    assert planner.activation_calls == [(5.55, 1)]


def test_ready_and_stop_status_share_dynamic_goal_fields():
    planner = _CallbackHarness()
    DreamPlannerNode._on_mission_goal(planner, _goal())

    fields = DreamPlannerNode._mission_status_fields(planner)
    assert fields == {
        "mission_goal_required": True,
        "mission_goal_received": True,
        "mission_goal_source": "/dream/mission_goal",
        "mission_goal_target_lane": 1,
        "mission_goal_x": pytest.approx(5.55),
        "mission_goal_last_rejection": "",
    }
