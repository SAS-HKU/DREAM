"""IDEAM decision, DREAM veto and LIMO MPC-CBF ROS node."""

from __future__ import annotations

import json
from math import isfinite
from typing import List, Optional

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String

from .core.decision import IDEAMDREAMDecision
from .core.command_adapter import gate_mpc_output
from .core.mpc import RiskAwareMPC
from .core.risk_field import DREAMRiskField
from .core.types import EgoState, Vehicle
from .limo_scale import default_deployment_config, deployment_config_for_arena, get_preset
from .ros_utils import ego_from_odometry, vehicle_from_mapping, yaw_to_quaternion


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
        self.declare_parameter("arena_file", "")
        self.config = deployment_config_for_arena(
            str(self.get_parameter("arena_file").value)
        )
        self.declare_parameter("route_intent_enabled", True)
        self.declare_parameter("route_target_lane", self.config.arena.target_lane)
        self.declare_parameter("route_merge_start_x", self.config.arena.merge_request_x)
        self.preset = get_preset(str(self.get_parameter("preset").value))
        self.field = DREAMRiskField(self.config)
        self.decision = IDEAMDREAMDecision(
            self.config,
            blocker_trigger_distance=float(self.get_parameter("blocker_trigger_distance").value),
        )
        self.mpc = RiskAwareMPC(
            self.config,
            enforce_map_bounds=bool(
                self.get_parameter("enforce_map_bounds").value
            ),
        )
        self.ego: Optional[EgoState] = None
        self.ego_receipt: Optional[float] = None
        self.vehicles: List[Vehicle] = []
        self.world_receipt: Optional[float] = None
        self.risk_receipt: Optional[float] = None
        self.ready_receipt: Optional[float] = None
        self.drift_ready = False

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
        self.control_publisher = self.create_publisher(TwistStamped, "/dream/control", 10)
        self.path_publisher = self.create_publisher(Path, "/dream/reference_trajectory", 2)
        self.status_publisher = self.create_publisher(String, "/dream/planner_status", 10)
        self.create_timer(1.0 / float(self.get_parameter("update_rate").value), self._plan)

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1.0e-9

    def _on_ego(self, message: Odometry) -> None:
        self.ego = ego_from_odometry(message)
        self.ego_receipt = self._now()

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

    def _publish_stop(self, reason: str, details: Optional[dict] = None) -> None:
        control = TwistStamped()
        control.header.stamp = self.get_clock().now().to_msg()
        control.header.frame_id = "base_link"
        self.control_publisher.publish(control)
        status = String()
        payload = {"ready": False, "preset": self.preset.name, "reason": reason}
        if details:
            payload.update(details)
        status.data = json.dumps(payload, separators=(",", ":"))
        self.status_publisher.publish(status)

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

    def _plan(self) -> None:
        now = self._now()
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
            route_active = (
                bool(self.get_parameter("route_intent_enabled").value)
                and ego.x >= float(self.get_parameter("route_merge_start_x").value)
            )
            requested_lane = (
                int(self.get_parameter("route_target_lane").value)
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
                "route_target_lane": int(self.get_parameter("route_target_lane").value),
                "maneuver": decision.maneuver,
                "current_lane": decision.current_lane,
                "requested_lane": decision.requested_lane,
                "selected_lane": decision.selected_lane,
                "vetoed": decision.vetoed,
                "decision_risk": decision.risk_score,
                "risk_at_ego": self.field.risk_at(ego.x, ego.y),
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
