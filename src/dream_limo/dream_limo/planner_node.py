"""IDEAM decision, DREAM veto and LIMO MPC-CBF ROS node."""

from __future__ import annotations

import json
from dataclasses import replace
from math import isfinite
from typing import List, Optional

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String

from .core.decision import IDEAMDREAMDecision
from .core.command_adapter import gate_mpc_output
from .core.goal_mission import (
    EgoMissionState,
    GoalRequest,
    goal_mission_config_from_deployment,
    validate_goal_request,
)
from .core.mission import MissionEndGuard
from .core.mpc import RiskAwareMPC
from .core.risk_field import DREAMRiskField
from .core.types import EgoState, Vehicle
from .limo_scale import (
    DeploymentConfig,
    default_deployment_config,
    deployment_config_for_arena,
    get_preset,
)
from .ros_utils import (
    ego_from_odometry,
    stamp_to_seconds,
    vehicle_from_mapping,
    yaw_to_quaternion,
)


def deployment_for_mission_goal(
    config: DeploymentConfig, *, goal_x: float, target_lane: int
) -> DeploymentConfig:
    """Return a deployment whose route endpoint is an independently validated goal."""
    return replace(
        config,
        arena=replace(
            config.arena,
            mission_goal_x=float(goal_x),
            target_lane=int(target_lane),
        ),
    )


class DreamPlannerNode(Node):
    def __init__(self) -> None:
        super().__init__("dream_planner")
        self.config = default_deployment_config()
        self.declare_parameter("preset", "balanced")
        self.declare_parameter("ego_topic", "/dream/ego_state")
        self.declare_parameter("world_topic", "/dream/world_model")
        self.declare_parameter("risk_topic", "/dream/risk_field_raw")
        self.declare_parameter("ready_topic", "/dream/drift_ready")
        self.declare_parameter("input_timeout", 0.50)
        self.declare_parameter("update_rate", 5.0)
        self.declare_parameter("blocker_trigger_distance", 2.5)
        self.declare_parameter(
            "maximum_allowed_cbf_slack", self.config.mpc.maximum_allowed_cbf_slack
        )
        self.declare_parameter("enforce_map_bounds", False)
        self.declare_parameter("require_mission_goal", False)
        self.declare_parameter("mission_goal_topic", "/dream/mission_goal")
        self.declare_parameter("arena_file", "")
        self.config = deployment_config_for_arena(
            str(self.get_parameter("arena_file").value)
        )
        self.declare_parameter("target_speed", self.config.mpc.target_speed)
        self.config = replace(
            self.config,
            mpc=replace(
                self.config.mpc,
                target_speed=float(self.get_parameter("target_speed").value),
            ),
        )
        self.declare_parameter("route_intent_enabled", True)
        self.declare_parameter("route_target_lane", self.config.arena.target_lane)
        self.declare_parameter("route_merge_start_x", self.config.arena.merge_request_x)
        self.require_mission_goal = bool(
            self.get_parameter("require_mission_goal").value
        )
        self.mission_goal_received = False
        self.pending_mission_goal: Optional[PoseStamped] = None
        self.mission_goal_source = (
            "waiting" if self.require_mission_goal else "configured_arena"
        )
        self.mission_goal_last_rejection = ""
        self.route_target_lane = int(self.get_parameter("route_target_lane").value)
        self.enforce_map_bounds = bool(
            self.get_parameter("enforce_map_bounds").value
        )
        self.preset = get_preset(str(self.get_parameter("preset").value))
        self.field = DREAMRiskField(self.config)
        self.decision = IDEAMDREAMDecision(
            self.config,
            blocker_trigger_distance=float(self.get_parameter("blocker_trigger_distance").value),
        )
        self.mpc = RiskAwareMPC(
            self.config,
            enforce_map_bounds=self.enforce_map_bounds,
        )
        self.mission = MissionEndGuard(
            goal_x=self.config.arena.mission_goal_x,
            position_tolerance=self.config.mpc.mission_position_tolerance,
            stop_speed_tolerance=self.config.mpc.mission_stop_speed_tolerance,
        )
        self.ego: Optional[EgoState] = None
        self.ego_receipt: Optional[float] = None
        self.vehicles: List[Vehicle] = []
        self.world_receipt: Optional[float] = None
        self.risk_receipt: Optional[float] = None
        self.ready_receipt: Optional[float] = None
        self.drift_ready = False
        self.goal_contract = goal_mission_config_from_deployment(self.config)

        self.create_subscription(
            Odometry, str(self.get_parameter("ego_topic").value), self._on_ego, 10
        )
        self.create_subscription(
            String, str(self.get_parameter("world_topic").value), self._on_world, 10
        )
        self.create_subscription(
            Image, str(self.get_parameter("risk_topic").value), self._on_risk, 2
        )
        self.create_subscription(
            Bool, str(self.get_parameter("ready_topic").value), self._on_ready, 10
        )
        mission_goal_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            PoseStamped,
            str(self.get_parameter("mission_goal_topic").value),
            self._on_mission_goal,
            mission_goal_qos,
        )
        self.control_publisher = self.create_publisher(TwistStamped, "/dream/control", 10)
        self.path_publisher = self.create_publisher(Path, "/dream/reference_trajectory", 2)
        self.status_publisher = self.create_publisher(String, "/dream/planner_status", 10)
        self.create_timer(1.0 / float(self.get_parameter("update_rate").value), self._plan)

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1.0e-9

    def _on_ego(self, message: Odometry) -> None:
        self.ego = ego_from_odometry(message)
        self.ego_receipt = self._now()
        if (
            self.require_mission_goal
            and not self.mission_goal_received
            and self.pending_mission_goal is not None
        ):
            self._consider_pending_mission_goal()

    def _on_world(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
            if payload.get("frame_id") != self.config.grid.frame_id:
                raise ValueError("world frame mismatch")
            self.vehicles = [vehicle_from_mapping(item) for item in payload.get("vehicles", [])]
            self.world_receipt = self._now()
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            self.get_logger().warning(f"Rejected world model: {exc}")

    def _on_risk(self, message: Image) -> None:
        try:
            if message.header.frame_id != self.config.grid.frame_id:
                raise ValueError("risk-field frame mismatch")
            if message.encoding != "32FC1":
                raise ValueError("risk-field encoding must be 32FC1")
            if message.width != self.config.grid.nx or message.height != self.config.grid.ny:
                raise ValueError("risk-field dimensions mismatch")
            values = np.frombuffer(bytes(message.data), dtype=np.float32).astype(np.float64)
            if values.size != self.config.grid.nx * self.config.grid.ny:
                raise ValueError("risk-field payload length mismatch")
            values = values.reshape(self.field.shape)
            if not np.all(np.isfinite(values)):
                raise ValueError("risk-field payload is non-finite")
            self.field.R = np.clip(values, 0.0, self.config.pde.risk_ceiling)
            self.field.elapsed_model_time = self.config.pde.warmup_duration
            self.risk_receipt = self._now()
        except (ValueError, TypeError) as exc:
            self.get_logger().warning(f"Rejected risk field: {exc}")

    def _on_ready(self, message: Bool) -> None:
        self.drift_ready = bool(message.data)
        self.ready_receipt = self._now()

    def _on_mission_goal(self, message: PoseStamped) -> None:
        if not self.require_mission_goal:
            return
        if self.mission_goal_received:
            self.get_logger().warning("Ignored later mission goal after one-shot acceptance")
            return
        self.pending_mission_goal = message
        self._consider_pending_mission_goal()

    def _consider_pending_mission_goal(self) -> None:
        message = self.pending_mission_goal
        if message is None or self.mission_goal_received:
            return
        now = self._now()
        ego = None
        if self.ego is not None:
            ego = EgoMissionState(
                x=self.ego.x,
                y=self.ego.y,
                speed=self.ego.speed,
                source_stamp=self.ego.stamp,
                receipt_stamp=(
                    float(self.ego_receipt) if self.ego_receipt is not None else 0.0
                ),
            )
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
            ego,
            now=now,
            config=self.goal_contract,
        )
        if not validation.accepted:
            self.mission_goal_last_rejection = validation.reason
            if validation.reason not in {
                "EGO_UNAVAILABLE",
                "STALE_EGO_RECEIPT",
                "STALE_EGO_SOURCE",
            }:
                self.pending_mission_goal = None
            self.mpc.reset()
            self._publish_stop(validation.reason, {"stamp": now})
            self.get_logger().warning(
                f"Rejected sanitized mission goal independently: {validation.reason}"
            )
            return
        try:
            self._activate_mission_goal(
                goal_x=float(validation.goal_x),
                target_lane=int(validation.target_lane),
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            self.mission_goal_last_rejection = "GOAL_CONFIGURATION_REJECTED"
            self.pending_mission_goal = None
            self.mpc.reset()
            self._publish_stop(
                "GOAL_CONFIGURATION_REJECTED",
                {"stamp": now, "goal_error": str(exc)},
            )
            self.get_logger().error(f"Rejected mission goal configuration: {exc}")
            return
        self.mission_goal_source = str(self.get_parameter("mission_goal_topic").value)
        self.mission_goal_last_rejection = ""
        self.mission_goal_received = True
        self.pending_mission_goal = None
        self.get_logger().info(
            "Activated one-shot planner mission: "
            f"x={self.config.arena.mission_goal_x:.3f}, lane={self.route_target_lane}"
        )

    def _activate_mission_goal(self, *, goal_x: float, target_lane: int) -> None:
        config = deployment_for_mission_goal(
            self.config,
            goal_x=goal_x,
            target_lane=target_lane,
        )
        decision = IDEAMDREAMDecision(
            config,
            blocker_trigger_distance=float(
                self.get_parameter("blocker_trigger_distance").value
            ),
        )
        mpc = RiskAwareMPC(config, enforce_map_bounds=self.enforce_map_bounds)
        mission = MissionEndGuard(
            goal_x=config.arena.mission_goal_x,
            position_tolerance=config.mpc.mission_position_tolerance,
            stop_speed_tolerance=config.mpc.mission_stop_speed_tolerance,
        )
        self.mpc.reset()
        self.config = config
        # The validated goal changes only the route endpoint/lane. Keep the
        # received DRIFT field and its warm-up state; update its immutable
        # geometry reference so risk queries use the same mission definition.
        self.field.config = config
        self.decision = decision
        self.mpc = mpc
        self.mission = mission
        self.route_target_lane = target_lane

    def _publish_stop(self, reason: str, details: Optional[dict] = None) -> None:
        control = TwistStamped()
        control.header.stamp = self.get_clock().now().to_msg()
        control.header.frame_id = "base_link"
        self.control_publisher.publish(control)
        status = String()
        payload = {
            "ready": False,
            "preset": self.preset.name,
            "reason": reason,
            **self._mission_status_fields(),
        }
        if details:
            payload.update(details)
        status.data = json.dumps(payload, separators=(",", ":"))
        self.status_publisher.publish(status)

    def _mission_status_fields(self) -> dict:
        return {
            "mission_goal_required": self.require_mission_goal,
            "mission_goal_received": self.mission_goal_received,
            "mission_goal_source": self.mission_goal_source,
            "mission_goal_target_lane": self.route_target_lane,
            "mission_goal_x": self.config.arena.mission_goal_x,
            "mission_goal_last_rejection": self.mission_goal_last_rejection,
        }

    def _publish_trajectory(self, states: np.ndarray) -> None:
        if states.size == 0:
            return
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = self.config.grid.frame_id
        for index in range(states.shape[1]):
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = float(states[0, index])
            pose.pose.position.y = float(states[1, index])
            qx, qy, qz, qw = yaw_to_quaternion(float(states[3, index]))
            pose.pose.orientation.x = qx
            pose.pose.orientation.y = qy
            pose.pose.orientation.z = qz
            pose.pose.orientation.w = qw
            path.poses.append(pose)
        self.path_publisher.publish(path)

    def _publish_mission_complete(self, now: float, remaining_distance: float) -> None:
        self.mpc.reset()
        self._publish_stop(
            "MISSION_COMPLETE",
            {
                "stamp": now,
                "mission_complete": True,
                "mission_goal_x": self.config.arena.mission_goal_x,
                "mission_remaining_distance": remaining_distance,
                "configured_target_speed": self.config.mpc.target_speed,
                "map_bounds_enforced": self.mpc.enforce_map_bounds,
            },
        )

    def _plan(self) -> None:
        now = self._now()
        if self.require_mission_goal and not self.mission_goal_received:
            self.mpc.reset()
            self._publish_stop("WAITING_FOR_MISSION_GOAL", {"stamp": now})
            return
        # Completion is a process-lifetime latch. Keep publishing zero with an
        # unambiguous status even if perception inputs later become stale.
        if self.mission.complete:
            remaining_distance = (
                self.mission.remaining_distance(self.ego.x)
                if self.ego is not None
                else 0.0
            )
            self._publish_mission_complete(now, remaining_distance)
            return
        timeout = float(self.get_parameter("input_timeout").value)
        timestamps = (self.ego_receipt, self.world_receipt, self.risk_receipt, self.ready_receipt)
        if any(value is None or now - value >= timeout for value in timestamps):
            self.mpc.reset()
            self._publish_stop("STALE_INPUT")
            return
        if not self.drift_ready:
            self.mpc.reset()
            self._publish_stop("DRIFT_NOT_READY")
            return
        try:
            lane = self.decision.lane_for_y(self.ego.y)
            ego = EgoState(
                x=self.ego.x,
                y=self.ego.y,
                yaw=self.ego.yaw,
                speed=self.ego.speed,
                yaw_rate=self.ego.yaw_rate,
                stamp=self.ego.stamp,
                lane_index=lane,
            )
            remaining_distance = self.mission.remaining_distance(ego.x)
            if self.mission.update(ego.x, ego.speed):
                self._publish_mission_complete(now, remaining_distance)
                return
            route_active = (
                bool(self.get_parameter("route_intent_enabled").value)
                and ego.x >= float(self.get_parameter("route_merge_start_x").value)
            )
            requested_lane = (
                self.route_target_lane
                if route_active
                else None
            )
            decision = self.decision.decide(
                ego,
                self.vehicles,
                self.field,
                self.preset,
                requested_lane=requested_lane,
            )
            result = self.mpc.solve(
                ego, decision.selected_lane, self.vehicles, self.field, self.preset
            )
        except (ValueError, RuntimeError) as exc:
            self.get_logger().error(f"Planner failed closed: {exc}")
            self.mpc.reset()
            self._publish_stop(f"PLANNER_ERROR:{exc}")
            return
        maximum_allowed_slack = float(
            self.get_parameter("maximum_allowed_cbf_slack").value
        )
        try:
            gated = gate_mpc_output(
                target_speed=result.command.target_speed,
                acceleration=result.command.acceleration,
                steering=result.command.steering,
                command_valid=result.command.valid,
                used_fallback=result.used_fallback,
                maximum_cbf_slack=result.maximum_slack,
                maximum_allowed_cbf_slack=maximum_allowed_slack,
            )
        except ValueError as exc:
            self.get_logger().error(f"Planner safety gate failed closed: {exc}")
            self.mpc.reset()
            self._publish_stop("PLANNER_SAFETY_CONFIG_ERROR")
            return
        if not gated.valid:
            self.get_logger().error(
                f"Planner output rejected fail-closed: {gated.reason} "
                f"(status={result.status}, slack={result.maximum_slack})"
            )
            self.mpc.reset()
            self._publish_stop(
                gated.reason,
                {
                    "stamp": now,
                    "mpc_status": result.status,
                    "mpc_fallback": result.used_fallback,
                    "maximum_cbf_slack": (
                        result.maximum_slack if isfinite(result.maximum_slack) else None
                    ),
                    "maximum_allowed_cbf_slack": maximum_allowed_slack,
                },
            )
            return
        control = TwistStamped()
        control.header.stamp = self.get_clock().now().to_msg()
        control.header.frame_id = "base_link"
        # Internal DREAM contract: x=target speed, y=acceleration, angular.z=center steer.
        control.twist.linear.x = gated.target_speed
        control.twist.linear.y = gated.acceleration
        control.twist.angular.z = gated.steering
        self.control_publisher.publish(control)
        self._publish_trajectory(result.states)
        status = String()
        status.data = json.dumps(
            {
                "stamp": now,
                "control_dt": self.config.pde.control_dt,
                "ready": result.command.valid,
                "preset": self.preset.name,
                "control_stack": (
                    "dream"
                    if (
                        self.preset.decision_veto
                        or self.preset.mpc_risk_cost
                        or self.preset.cbf_risk_expansion
                    )
                    else "pure_mpc"
                ),
                "route_intent_active": route_active,
                "route_target_lane": self.route_target_lane,
                "maneuver": decision.maneuver,
                "current_lane": decision.current_lane,
                "requested_lane": decision.requested_lane,
                "selected_lane": decision.selected_lane,
                "vetoed": decision.vetoed,
                "decision_risk": decision.risk_score,
                "risk_at_ego": self.field.risk_at(ego.x, ego.y),
                "mission_complete": False,
                "mission_goal_x": self.config.arena.mission_goal_x,
                "mission_remaining_distance": remaining_distance,
                "configured_target_speed": self.config.mpc.target_speed,
                "target_speed": result.command.target_speed,
                "acceleration": result.command.acceleration,
                "center_steer": result.command.steering,
                "t_decision": decision.compute_seconds,
                "t_mpc": result.solve_seconds,
                "mpc_status": result.status,
                "mpc_fallback": result.used_fallback,
                "maximum_cbf_slack": result.maximum_slack,
                "maximum_allowed_cbf_slack": maximum_allowed_slack,
                "map_bounds_enforced": self.mpc.enforce_map_bounds,
                **self._mission_status_fields(),
            },
            separators=(",", ":"),
        )
        self.status_publisher.publish(status)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DreamPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
