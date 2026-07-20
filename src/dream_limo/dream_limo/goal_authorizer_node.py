"""One-shot, goal-triggered autonomous mission authorization."""

from __future__ import annotations

import json
from math import cos, isfinite, sin
from typing import Optional

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger

from .core.goal_mission import (
    EgoMissionState,
    GoalAuthorization,
    GoalMissionLatch,
    GoalRequest,
    GoalValidation,
    PlannerGoalReadiness,
    PreflightReadiness,
    evaluate_goal_authorization,
    goal_mission_config_from_deployment,
    validate_configured_auto_goal,
    validate_goal_request,
)
from .limo_scale import default_deployment_config, deployment_config_for_arena
from .ros_utils import stamp_to_seconds


class DreamGoalAuthorizerNode(Node):
    """Authorize one sanitized map goal while keeping motion fail closed."""

    def __init__(self) -> None:
        super().__init__("dream_goal_authorizer")
        defaults = default_deployment_config()
        default_contract = goal_mission_config_from_deployment(defaults)
        self.declare_parameter("arena_file", "")
        self.declare_parameter("enabled", False)
        self.declare_parameter("auto_start", False)
        self.declare_parameter("goal_topic", "/goal_pose")
        self.declare_parameter("ego_topic", "/dream/ego_state")
        self.declare_parameter("accepted_goal_topic", "/dream/mission_goal")
        self.declare_parameter("planner_status_topic", "/dream/planner_status")
        self.declare_parameter("preflight_status_topic", "/dream/preflight_status")
        self.declare_parameter("arm_topic", "/dream/arm")
        self.declare_parameter("external_stop_topic", "/dream/external_stop")
        self.declare_parameter("status_topic", "/dream/deadman_status")
        self.declare_parameter("stop_service", "/dream/stop_mission")
        self.declare_parameter("publish_rate", 20.0)
        self.declare_parameter("goal_frame", default_contract.frame_id)
        self.declare_parameter("lane_tolerance", default_contract.lane_tolerance)
        self.declare_parameter(
            "minimum_ahead_distance", default_contract.minimum_ahead_distance
        )
        self.declare_parameter(
            "maximum_stopped_speed", default_contract.maximum_stopped_speed
        )
        self.declare_parameter("goal_timeout", default_contract.goal_timeout)
        self.declare_parameter("ego_timeout", default_contract.ego_timeout)
        self.declare_parameter("planner_timeout", 0.75)
        self.declare_parameter("preflight_timeout", 2.0)
        self.declare_parameter(
            "future_tolerance", default_contract.future_tolerance
        )

        deployment = deployment_config_for_arena(
            str(self.get_parameter("arena_file").value)
        )
        self.contract = goal_mission_config_from_deployment(
            deployment,
            frame_id=str(self.get_parameter("goal_frame").value),
            lane_tolerance=float(self.get_parameter("lane_tolerance").value),
            minimum_ahead_distance=float(
                self.get_parameter("minimum_ahead_distance").value
            ),
            maximum_stopped_speed=float(
                self.get_parameter("maximum_stopped_speed").value
            ),
            goal_timeout=float(self.get_parameter("goal_timeout").value),
            ego_timeout=float(self.get_parameter("ego_timeout").value),
            future_tolerance=float(self.get_parameter("future_tolerance").value),
        )
        publish_rate = float(self.get_parameter("publish_rate").value)
        self.enabled = bool(self.get_parameter("enabled").value)
        self.auto_start = bool(self.get_parameter("auto_start").value)
        self.configured_goal_x = float(deployment.arena.mission_goal_x)
        self.configured_target_lane = deployment.arena.target_lane
        self.planner_timeout = float(self.get_parameter("planner_timeout").value)
        self.preflight_timeout = float(
            self.get_parameter("preflight_timeout").value
        )
        if (
            publish_rate <= 0.0
            or self.planner_timeout <= 0.0
            or self.preflight_timeout <= 0.0
        ):
            raise ValueError("authorizer timing parameters must be positive")
        if self.contract.frame_id != deployment.grid.frame_id:
            raise ValueError("goal_frame must match the deployed risk-grid frame")

        reliable = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        latched = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.state = GoalMissionLatch()
        self.ego: Optional[EgoMissionState] = None
        self.planner: Optional[PlannerGoalReadiness] = None
        self.preflight: Optional[PreflightReadiness] = None
        # Auto-start waits until the planner has advertised its fail-closed
        # WAITING_FOR_MISSION_GOAL state.  This prevents the one-shot, latched
        # goal from aging out if the authorizer receives odometry before the
        # planner has finished constructing its subscriptions.
        self.planner_waiting_for_goal = False
        self.accepted_goal_source_stamp: Optional[float] = None
        self.accepted_goal_receipt_stamp: Optional[float] = None
        self.accepted_goal_source = "waiting"

        self.mission_goal_publisher = self.create_publisher(
            PoseStamped,
            str(self.get_parameter("accepted_goal_topic").value),
            latched,
        )
        self.arm_publisher = self.create_publisher(
            Bool, str(self.get_parameter("arm_topic").value), reliable
        )
        self.stop_publisher = self.create_publisher(
            Bool, str(self.get_parameter("external_stop_topic").value), reliable
        )
        self.status_publisher = self.create_publisher(
            String, str(self.get_parameter("status_topic").value), reliable
        )
        self.create_subscription(
            PoseStamped,
            str(self.get_parameter("goal_topic").value),
            self._on_goal,
            reliable,
        )
        self.create_subscription(
            Odometry,
            str(self.get_parameter("ego_topic").value),
            self._on_ego,
            reliable,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("planner_status_topic").value),
            self._on_planner_status,
            reliable,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("preflight_status_topic").value),
            self._on_preflight_status,
            reliable,
        )
        self.create_service(
            Trigger,
            str(self.get_parameter("stop_service").value),
            self._on_stop_mission,
        )
        self.create_timer(1.0 / publish_rate, self._publish_heartbeat)

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1.0e-9

    def _on_ego(self, message: Odometry) -> None:
        twist = message.twist.twist.linear
        self.ego = EgoMissionState(
            x=float(message.pose.pose.position.x),
            y=float(message.pose.pose.position.y),
            speed=(float(twist.x) ** 2 + float(twist.y) ** 2) ** 0.5,
            source_stamp=stamp_to_seconds(message.header.stamp),
            receipt_stamp=self._now(),
        )
        if self.auto_start:
            self._try_auto_start()

    def _try_auto_start(self) -> None:
        """Accept the configured mission once a fresh, stopped ego is available."""
        if (
            not self.enabled
            or not self.auto_start
            or not self.planner_waiting_for_goal
            or self.state.accepted_goal is not None
            or self.state.stop_latched
            or self.state.mission_complete
        ):
            return
        now = self._now()
        validation = validate_configured_auto_goal(
            self.ego,
            now=now,
            config=self.contract,
            mission_goal_x=self.configured_goal_x,
            target_lane=self.configured_target_lane,
        )
        self._consider_and_publish_goal(
            validation,
            source_stamp=now,
            receipt_stamp=now,
            source="auto_forward",
            warn_on_rejection=False,
        )

    def _on_goal(self, message: PoseStamped) -> None:
        now = self._now()
        if not self.enabled:
            self.state.consider(GoalValidation(False, "DISABLED"))
            self._publish_heartbeat()
            return
        if self.auto_start:
            self.get_logger().warning(
                "Ignoring operator goal because configured auto-start is active"
            )
            self._publish_heartbeat()
            return
        pose = message.pose
        validation = validate_goal_request(
            GoalRequest(
                frame_id=str(message.header.frame_id),
                x=float(pose.position.x),
                y=float(pose.position.y),
                z=float(pose.position.z),
                qx=float(pose.orientation.x),
                qy=float(pose.orientation.y),
                qz=float(pose.orientation.z),
                qw=float(pose.orientation.w),
                source_stamp=stamp_to_seconds(message.header.stamp),
                receipt_stamp=now,
            ),
            self.ego,
            now=now,
            config=self.contract,
        )
        self._consider_and_publish_goal(
            validation,
            source_stamp=stamp_to_seconds(message.header.stamp),
            receipt_stamp=now,
            source="operator_goal",
            warn_on_rejection=True,
        )

    def _consider_and_publish_goal(
        self,
        validation: GoalValidation,
        *,
        source_stamp: float,
        receipt_stamp: float,
        source: str,
        warn_on_rejection: bool,
    ) -> bool:
        previous_reason = self.state.reason
        if not self.state.consider(validation):
            if warn_on_rejection or self.state.reason != previous_reason:
                self.get_logger().warning(f"Goal rejected: {self.state.reason}")
            self._publish_heartbeat()
            return False

        goal = PoseStamped()
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = self.contract.frame_id
        goal.pose.position.x = float(validation.goal_x)
        goal.pose.position.y = float(validation.goal_y)
        goal.pose.orientation.z = sin(0.5 * float(validation.goal_yaw))
        goal.pose.orientation.w = cos(0.5 * float(validation.goal_yaw))
        self.mission_goal_publisher.publish(goal)
        self.accepted_goal_source_stamp = float(source_stamp)
        self.accepted_goal_receipt_stamp = float(receipt_stamp)
        self.accepted_goal_source = str(source)
        self.get_logger().info(
            "Accepted one-shot mission goal: "
            f"x={validation.goal_x:.3f}, y={validation.goal_y:.3f}, "
            f"lane={validation.target_lane}, source={source}"
        )
        self._publish_heartbeat()
        return True

    def _on_planner_status(self, message: String) -> None:
        now = self._now()
        try:
            payload = json.loads(message.data)
        except (json.JSONDecodeError, TypeError):
            self.planner = PlannerGoalReadiness(False, None, None, now)
            return
        if isinstance(payload, dict) and (
            payload.get("mission_complete") is True
            or payload.get("reason") == "MISSION_COMPLETE"
        ):
            self.state.complete()
            self._publish_heartbeat()
            return
        if not isinstance(payload, dict):
            self.planner = PlannerGoalReadiness(False, None, None, now)
            return
        if (
            self.auto_start
            and payload.get("mission_goal_required") is True
            and payload.get("mission_goal_received") is False
            and payload.get("reason") == "WAITING_FOR_MISSION_GOAL"
        ):
            self.planner_waiting_for_goal = True
        goal_x = payload.get("mission_goal_x")
        lane = payload.get(
            "mission_goal_target_lane", payload.get("route_target_lane")
        )
        try:
            goal_x = float(goal_x)
            lane = int(lane)
        except (TypeError, ValueError, OverflowError):
            goal_x = None
            lane = None
        if goal_x is not None and not isfinite(goal_x):
            goal_x = None
        self.planner = PlannerGoalReadiness(
            ready=payload.get("ready") is True,
            mission_goal_x=goal_x,
            target_lane=lane,
            receipt_stamp=now,
        )
        if self.auto_start:
            self._try_auto_start()

    def _on_preflight_status(self, message: String) -> None:
        now = self._now()
        try:
            payload = json.loads(message.data)
        except (json.JSONDecodeError, TypeError):
            payload = None
        self.preflight = PreflightReadiness(
            passed=bool(isinstance(payload, dict) and payload.get("passed") is True),
            receipt_stamp=now,
        )

    def _on_stop_mission(self, _request: Trigger.Request, response: Trigger.Response):
        already_stopped = self.state.stop_latched
        self.state.stop()
        self._publish_heartbeat()
        response.success = True
        response.message = (
            "mission stop was already latched"
            if already_stopped
            else "mission stop latched"
        )
        return response

    def _status_payload(self) -> dict:
        now = self._now()
        goal = self.state.accepted_goal
        validation = self.state.last_validation
        authorization = self._authorization(now)
        goal_source_age = (
            None
            if self.accepted_goal_source_stamp is None
            else max(0.0, now - self.accepted_goal_source_stamp)
        )
        goal_receipt_age = (
            None
            if self.accepted_goal_receipt_stamp is None
            else max(0.0, now - self.accepted_goal_receipt_stamp)
        )
        return {
            "owner": self.get_name(),
            "owner_node": self.get_name(),
            "ready": authorization.ready,
            "armed": authorization.armed,
            "external_stop": self.state.stop_latched,
            "reason": authorization.reason,
            "goal_active": self.state.active,
            "goal_source": self.accepted_goal_source,
            "goal_received": self.state.goal_received,
            "goal_accepted": goal is not None,
            "target_lane": None if goal is None else goal.target_lane,
            "goal_x": None if goal is None else goal.goal_x,
            "goal_y": None if goal is None else goal.goal_y,
            "goal_yaw": None if goal is None else goal.goal_yaw,
            "goal_source_age": (
                goal_source_age
                if goal is not None
                else None if validation is None else validation.goal_source_age
            ),
            "goal_receipt_age": (
                goal_receipt_age
                if goal is not None
                else None if validation is None else validation.goal_receipt_age
            ),
            "ego_source_age": authorization.ego_source_age,
            "ego_receipt_age": authorization.ego_receipt_age,
            "planner_status_age": authorization.planner_age,
            "preflight_status_age": authorization.preflight_age,
            "goal_timeout": self.contract.goal_timeout,
            "ego_timeout": self.contract.ego_timeout,
            "maximum_stopped_speed": self.contract.maximum_stopped_speed,
            "minimum_ahead_distance": self.contract.minimum_ahead_distance,
            "one_shot": True,
            "mission_complete": self.state.mission_complete,
            "stop_latched": self.state.stop_latched,
            "enabled": self.enabled,
            "auto_start": self.auto_start,
            "auto_start_planner_waiting_seen": self.planner_waiting_for_goal,
        }

    def _authorization(self, now: Optional[float] = None) -> GoalAuthorization:
        return evaluate_goal_authorization(
            self.state,
            self.ego,
            self.planner,
            self.preflight,
            now=self._now() if now is None else now,
            config=self.contract,
            planner_timeout=self.planner_timeout,
            preflight_timeout=self.preflight_timeout,
            enabled=self.enabled,
        )

    def _publish_heartbeat(self) -> None:
        authorization = self._authorization()
        self.arm_publisher.publish(Bool(data=authorization.armed))
        self.stop_publisher.publish(Bool(data=self.state.stop_latched))
        self.status_publisher.publish(
            String(data=json.dumps(self._status_payload(), separators=(",", ":")))
        )

    def publish_shutdown_stop(self) -> None:
        """Publish the explicit teardown state while the ROS context is valid."""
        self.arm_publisher.publish(Bool(data=False))
        self.stop_publisher.publish(Bool(data=True))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DreamGoalAuthorizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.publish_shutdown_stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
