"""Online experiment metrics published for rosbag capture."""

from __future__ import annotations

import json
from math import cos, hypot, inf, isfinite
from typing import Optional

import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, String

from .core.types import EgoState, Vehicle
from .limo_scale import deployment_config_for_arena
from .ros_utils import ego_from_odometry, quaternion_to_yaw, transform_planar


class DreamMetricsNode(Node):
    def __init__(self) -> None:
        super().__init__("dream_metrics")
        self.declare_parameter("ego_topic", "/dream/ego_state")
        self.declare_parameter("merger_odom_topic", "/merger/wheel/odom")
        self.declare_parameter("merger_visible_topic", "/dream/merger_visible")
        self.declare_parameter("planner_status_topic", "/dream/planner_status")
        self.declare_parameter("arena_file", "")
        self.declare_parameter("map_to_odom_x", 0.0)
        self.declare_parameter("map_to_odom_y", 0.0)
        self.declare_parameter("map_to_odom_yaw", 0.0)
        self.declare_parameter("alignment_topic", "/dream/map_alignment")
        self.config = deployment_config_for_arena(
            str(self.get_parameter("arena_file").value)
        )
        self.ego: Optional[EgoState] = None
        self.merger: Optional[Vehicle] = None
        self.visible = False
        self.reveal_time: Optional[float] = None
        self.ttc_at_reveal = inf
        self.ttc_at_reveal_reason = "NOT_REVEALED"
        self.closing_speed_at_reveal: Optional[float] = None
        self.lateral_separation_at_reveal: Optional[float] = None
        self.projected_conflict_arrival_margin_at_reveal = inf
        self.projected_conflict_arrival_margin_reason = "NOT_REVEALED"
        self.ego_conflict_entry_time: Optional[float] = None
        self.merger_conflict_entry_time: Optional[float] = None
        self.conflict_zone_overlap_samples = 0
        self.minimum_conflict_zone_clearance = inf
        self.minimum_clearance = inf
        self.minimum_post_reveal_clearance = inf
        self.minimum_ttc = inf
        self.maximum_abs_acceleration = 0.0
        self.maximum_abs_jerk = 0.0
        self.previous_acceleration: Optional[float] = None
        self.previous_acceleration_time: Optional[float] = None
        self.veto_activations = 0
        self.veto_events = 0
        self.previous_veto = False
        self.risk_at_ego = 0.0
        self.maximum_drift_seconds = 0.0
        self.maximum_decision_seconds = 0.0
        self.maximum_mpc_seconds = 0.0
        self.map_alignment = (
            float(self.get_parameter("map_to_odom_x").value),
            float(self.get_parameter("map_to_odom_y").value),
            float(self.get_parameter("map_to_odom_yaw").value),
        )

        alignment_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            TransformStamped,
            str(self.get_parameter("alignment_topic").value),
            self._on_alignment,
            alignment_qos,
        )

        self.create_subscription(
            Odometry, str(self.get_parameter("ego_topic").value), self._on_ego, 10
        )
        self.create_subscription(
            Odometry,
            str(self.get_parameter("merger_odom_topic").value),
            self._on_merger,
            10,
        )
        self.create_subscription(
            Bool,
            str(self.get_parameter("merger_visible_topic").value),
            self._on_visibility,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("planner_status_topic").value),
            self._on_planner,
            10,
        )
        self.create_subscription(String, "/dream/drift_status", self._on_drift, 10)
        self.publisher = self.create_publisher(String, "/dream/metrics", 10)
        self.create_timer(0.2, self._publish)

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1.0e-9

    def _on_ego(self, message: Odometry) -> None:
        self.ego = ego_from_odometry(message)
        self._update_pair_metrics()

    def _on_alignment(self, message: TransformStamped) -> None:
        self.map_alignment = (
            float(message.transform.translation.x),
            float(message.transform.translation.y),
            quaternion_to_yaw(message.transform.rotation),
        )

    def _on_merger(self, message: Odometry) -> None:
        position = message.pose.pose.position
        velocity = message.twist.twist.linear
        tx, ty, yaw = self.map_alignment
        x, y, vx, vy = transform_planar(
            position.x,
            position.y,
            velocity.x,
            velocity.y,
            tx=tx,
            ty=ty,
            yaw=yaw,
        )
        self.merger = Vehicle(
            "metrics_merger",
            x,
            y,
            vx=vx,
            vy=vy,
            heading=quaternion_to_yaw(message.pose.pose.orientation) + yaw,
            length=0.22,
            width=0.22,
            stamp=self._now(),
        )
        self._update_pair_metrics()

    def _ttc_diagnostic(self):
        if self.ego is None or self.merger is None:
            return inf, "MISSING_ACTOR"
        lateral_separation = abs(self.ego.y - self.merger.y)
        if lateral_separation > 0.55:
            return inf, "OUTSIDE_SHARED_CORRIDOR"
        gap = self.merger.x - self.ego.x - 0.22
        closing = self.ego.speed - self.merger.vx
        if gap <= 0.0:
            return inf, "NONPOSITIVE_GAP"
        if closing <= 1.0e-6:
            return inf, "NOT_CLOSING"
        return gap / closing, "VALID"

    def _ttc(self) -> float:
        return self._ttc_diagnostic()[0]

    def _inside_conflict_zone(self, x: float, y: float) -> bool:
        arena = self.config.arena
        target_y = arena.lane_centers[arena.target_lane]
        return (
            arena.conflict_zone_x_min <= x <= arena.conflict_zone_x_max
            and abs(y - target_y) <= 0.5 * arena.lane_width
        )

    def _projected_conflict_arrival_margin(self):
        """Constant-velocity reveal diagnostic; it is never a planner input."""
        if self.ego is None or self.merger is None:
            return inf, "MISSING_ACTOR"
        arena = self.config.arena
        ego_forward_speed = self.ego.speed * cos(self.ego.yaw)
        if ego_forward_speed <= 1.0e-6:
            return inf, "EGO_NOT_ADVANCING"
        ego_eta = max(0.0, arena.conflict_zone_x_min - self.ego.x) / ego_forward_speed

        if self.merger.x > arena.conflict_zone_x_max:
            return inf, "MERGER_PAST_CONFLICT_ZONE"
        if self.merger.x < arena.conflict_zone_x_min:
            if self.merger.vx <= 1.0e-6:
                return inf, "MERGER_NOT_ADVANCING"
            merger_x_eta = (arena.conflict_zone_x_min - self.merger.x) / self.merger.vx
        else:
            merger_x_eta = 0.0

        target_y = arena.lane_centers[arena.target_lane]
        half_width = 0.5 * arena.lane_width
        lateral_offset = self.merger.y - target_y
        if abs(lateral_offset) <= half_width:
            merger_y_eta = 0.0
        elif lateral_offset < -half_width and self.merger.vy > 1.0e-6:
            merger_y_eta = (-half_width - lateral_offset) / self.merger.vy
        elif lateral_offset > half_width and self.merger.vy < -1.0e-6:
            merger_y_eta = (half_width - lateral_offset) / self.merger.vy
        else:
            return inf, "MERGER_NOT_APPROACHING_MIDDLE_LANE"

        merger_eta = max(merger_x_eta, merger_y_eta)
        projected_x = self.merger.x + self.merger.vx * merger_eta
        if projected_x > arena.conflict_zone_x_max:
            return inf, "MERGER_MISSES_CONFLICT_ZONE"
        return ego_eta - merger_eta, "VALID"

    def _update_pair_metrics(self) -> None:
        if self.ego is None or self.merger is None:
            return
        clearance = hypot(self.ego.x - self.merger.x, self.ego.y - self.merger.y) - 0.22
        self.minimum_clearance = min(self.minimum_clearance, clearance)
        now = self._now()
        ego_in_conflict = self._inside_conflict_zone(self.ego.x, self.ego.y)
        merger_in_conflict = self._inside_conflict_zone(self.merger.x, self.merger.y)
        if ego_in_conflict and self.ego_conflict_entry_time is None:
            self.ego_conflict_entry_time = now
        if merger_in_conflict and self.merger_conflict_entry_time is None:
            self.merger_conflict_entry_time = now
        if ego_in_conflict and merger_in_conflict:
            self.minimum_conflict_zone_clearance = min(
                self.minimum_conflict_zone_clearance, clearance
            )
        ttc = self._ttc()
        if isfinite(ttc):
            self.minimum_ttc = min(self.minimum_ttc, ttc)
        if self.reveal_time is not None and self._now() <= self.reveal_time + 3.0:
            self.minimum_post_reveal_clearance = min(
                self.minimum_post_reveal_clearance, clearance
            )

    def _on_visibility(self, message: Bool) -> None:
        was_visible = self.visible
        self.visible = bool(message.data)
        if self.visible and not was_visible and self.reveal_time is None:
            self.reveal_time = self._now()
            self.ttc_at_reveal, self.ttc_at_reveal_reason = self._ttc_diagnostic()
            if self.ego is not None and self.merger is not None:
                self.closing_speed_at_reveal = self.ego.speed - self.merger.vx
                self.lateral_separation_at_reveal = abs(self.ego.y - self.merger.y)
                (
                    self.projected_conflict_arrival_margin_at_reveal,
                    self.projected_conflict_arrival_margin_reason,
                ) = self._projected_conflict_arrival_margin()

    def _on_planner(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
        except json.JSONDecodeError:
            return
        vetoed = bool(payload.get("vetoed", False))
        if vetoed and not self.previous_veto:
            self.veto_events += 1
        if vetoed:
            self.veto_activations += 1
        self.previous_veto = vetoed
        self.risk_at_ego = float(payload.get("risk_at_ego", self.risk_at_ego))
        self.maximum_decision_seconds = max(
            self.maximum_decision_seconds, float(payload.get("t_decision", 0.0))
        )
        self.maximum_mpc_seconds = max(
            self.maximum_mpc_seconds, float(payload.get("t_mpc", 0.0))
        )
        acceleration = float(payload.get("acceleration", 0.0))
        now = float(payload.get("stamp", self._now()))
        self.maximum_abs_acceleration = max(self.maximum_abs_acceleration, abs(acceleration))
        if self.previous_acceleration is not None and now > self.previous_acceleration_time:
            control_dt = float(payload.get("control_dt", now - self.previous_acceleration_time))
            jerk = (acceleration - self.previous_acceleration) / max(control_dt, 1.0e-6)
            self.maximum_abs_jerk = max(self.maximum_abs_jerk, abs(jerk))
        self.previous_acceleration = acceleration
        self.previous_acceleration_time = now

    def _on_drift(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
            self.maximum_drift_seconds = max(
                self.maximum_drift_seconds, float(payload.get("compute_seconds", 0.0))
            )
        except (json.JSONDecodeError, ValueError, TypeError):
            return

    @staticmethod
    def _finite_or_none(value: float):
        return value if isfinite(value) else None

    def _publish(self) -> None:
        if (
            self.ego is not None
            and self.merger is not None
            and self._inside_conflict_zone(self.ego.x, self.ego.y)
            and self._inside_conflict_zone(self.merger.x, self.merger.y)
        ):
            self.conflict_zone_overlap_samples += 1
        message = String()
        message.data = json.dumps(
            {
                "stamp": self._now(),
                "merger_visible": self.visible,
                "t_reveal": self.reveal_time,
                "ttc_at_reveal": self._finite_or_none(self.ttc_at_reveal),
                "ttc_at_reveal_valid": self.ttc_at_reveal_reason == "VALID",
                "ttc_at_reveal_reason": self.ttc_at_reveal_reason,
                "closing_speed_at_reveal": self.closing_speed_at_reveal,
                "lateral_separation_at_reveal": self.lateral_separation_at_reveal,
                "projected_conflict_arrival_margin_at_reveal": self._finite_or_none(
                    self.projected_conflict_arrival_margin_at_reveal
                ),
                "projected_conflict_arrival_margin_valid": (
                    self.projected_conflict_arrival_margin_reason == "VALID"
                ),
                "projected_conflict_arrival_margin_reason": (
                    self.projected_conflict_arrival_margin_reason
                ),
                "ego_conflict_entry_time": self.ego_conflict_entry_time,
                "merger_conflict_entry_time": self.merger_conflict_entry_time,
                "conflict_zone_overlap_samples": self.conflict_zone_overlap_samples,
                "conflict_zone_sample_period": 0.2,
                "minimum_conflict_zone_clearance": self._finite_or_none(
                    self.minimum_conflict_zone_clearance
                ),
                "minimum_clearance": self._finite_or_none(self.minimum_clearance),
                "minimum_post_reveal_clearance": self._finite_or_none(
                    self.minimum_post_reveal_clearance
                ),
                "minimum_ttc": self._finite_or_none(self.minimum_ttc),
                "veto_activations": self.veto_activations,
                "veto_events": self.veto_events,
                "risk_at_ego": self.risk_at_ego,
                "maximum_abs_acceleration": self.maximum_abs_acceleration,
                "maximum_abs_jerk": self.maximum_abs_jerk,
                "maximum_drift_seconds": self.maximum_drift_seconds,
                "maximum_decision_seconds": self.maximum_decision_seconds,
                "maximum_mpc_seconds": self.maximum_mpc_seconds,
            },
            separators=(",", ":"),
            allow_nan=False,
        )
        self.publisher.publish(message)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DreamMetricsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except RuntimeError:
        # ROS Humble can surface a take_message conversion error when launch
        # tears down several publishers and subscribers concurrently.
        if rclpy.ok():
            raise
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
