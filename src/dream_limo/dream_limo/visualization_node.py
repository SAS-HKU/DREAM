"""RViz markers that make the occluded-merger control flow explicit."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import numpy as np
import rclpy
from geometry_msgs.msg import Point, TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, String
from visualization_msgs.msg import Marker, MarkerArray

from .core.types import EgoState, Vehicle
from .core.route import anchored_lane_change_y
from .limo_scale import default_deployment_config, deployment_config_for_arena
from .ros_utils import (
    ego_from_odometry,
    quaternion_to_yaw,
    transform_planar,
    vehicle_from_mapping,
    yaw_to_quaternion,
)


def world_visibility_label(
    *, merger_visible: bool, visibility_source: str, dynamic_track_count: int
) -> str:
    """Describe only what the selected perception source actually establishes."""
    if visibility_source == "perception_only_no_merger_ground_truth":
        return (
            f"TRACK OBSERVED ({dynamic_track_count})"
            if dynamic_track_count > 0
            else "SHADOW / NO TRACK"
        )
    return "VISIBLE" if merger_visible else "OCCLUDED"


class DreamVisualizationNode(Node):
    """Render ground truth separately from the planner-visible world."""

    def __init__(self) -> None:
        super().__init__("dream_visualization")
        self.config = default_deployment_config()
        self.declare_parameter("ego_topic", "/dream/ego_state")
        self.declare_parameter("world_topic", "/dream/world_model")
        self.declare_parameter("merger_odom_topic", "/merger/wheel/odom")
        self.declare_parameter("merger_visible_topic", "/dream/merger_visible")
        self.declare_parameter("planner_status_topic", "/dream/planner_status")
        self.declare_parameter("safety_status_topic", "/dream/safety_status")
        self.declare_parameter("metrics_topic", "/dream/metrics")
        self.declare_parameter("scenario_status_topic", "/dream/scenario_status")
        self.declare_parameter("smoke_status_topic", "/dream/smoke_status")
        self.declare_parameter("publish_rate", 10.0)
        self.declare_parameter("map_to_odom_x", 0.0)
        self.declare_parameter("map_to_odom_y", 0.0)
        self.declare_parameter("map_to_odom_yaw", 0.0)
        self.declare_parameter("alignment_topic", "/dream/map_alignment")
        self.declare_parameter("arena_file", "")
        self.declare_parameter("mode_label", "STARTING")
        self.config = deployment_config_for_arena(
            str(self.get_parameter("arena_file").value)
        )

        self.ego: Optional[EgoState] = None
        self.world_vehicles: List[Vehicle] = []
        self.world_meta: Dict[str, Any] = {}
        self.static_ids = set()
        self.merger: Optional[Vehicle] = None
        self.merger_visible = False
        self.planner: Dict[str, Any] = {}
        self.safety: Dict[str, Any] = {}
        self.metrics: Dict[str, Any] = {}
        self.scenario: Dict[str, Any] = {}
        self.smoke: Dict[str, Any] = {}
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
            String, str(self.get_parameter("world_topic").value), self._on_world, 10
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
            self._on_merger_visible,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("planner_status_topic").value),
            lambda message: self._load_json(message, "planner"),
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("safety_status_topic").value),
            lambda message: self._load_json(message, "safety"),
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("metrics_topic").value),
            lambda message: self._load_json(message, "metrics"),
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("scenario_status_topic").value),
            lambda message: self._load_json(message, "scenario"),
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("smoke_status_topic").value),
            lambda message: self._load_json(message, "smoke"),
            10,
        )
        self.publisher = self.create_publisher(MarkerArray, "/dream/scenario_markers", 10)
        self.create_timer(1.0 / float(self.get_parameter("publish_rate").value), self._publish)

    def _on_ego(self, message: Odometry) -> None:
        self.ego = ego_from_odometry(message)

    def _on_alignment(self, message: TransformStamped) -> None:
        self.map_alignment = (
            float(message.transform.translation.x),
            float(message.transform.translation.y),
            quaternion_to_yaw(message.transform.rotation),
        )

    def _on_world(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
            self.world_vehicles = [
                vehicle_from_mapping(item) for item in payload.get("vehicles", [])
            ]
            self.static_ids = set(payload.get("static_vehicle_ids", []))
            self.world_meta = {
                "visibility_source": payload.get("visibility_source", "unknown"),
                "dynamic_track_count": int(payload.get("dynamic_track_count", 0)),
                "shadow_cells": int(payload.get("shadow_cells", 0)),
            }
        except (json.JSONDecodeError, ValueError, TypeError, KeyError):
            return

    def _on_merger(self, message: Odometry) -> None:
        pose = message.pose.pose
        twist = message.twist.twist
        tx, ty, yaw = self.map_alignment
        x, y, vx, vy = transform_planar(
            float(pose.position.x),
            float(pose.position.y),
            float(twist.linear.x),
            float(twist.linear.y),
            tx=tx,
            ty=ty,
            yaw=yaw,
        )
        self.merger = Vehicle(
            "merger_ground_truth",
            x,
            y,
            vx=vx,
            vy=vy,
            heading=(
                quaternion_to_yaw(pose.orientation) + yaw
            ),
            length=0.22,
            width=0.22,
        )

    def _on_merger_visible(self, message: Bool) -> None:
        self.merger_visible = bool(message.data)

    def _load_json(self, message: String, target: str) -> None:
        try:
            setattr(self, target, json.loads(message.data))
        except (json.JSONDecodeError, TypeError):
            return

    def _base_marker(self, marker_id: int, marker_type: int, namespace: str) -> Marker:
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = self.config.grid.frame_id
        marker.ns = namespace
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.lifetime.sec = 0
        marker.lifetime.nanosec = 350_000_000
        return marker

    @staticmethod
    def _set_color(marker: Marker, red: float, green: float, blue: float, alpha: float) -> None:
        marker.color.r = red
        marker.color.g = green
        marker.color.b = blue
        marker.color.a = alpha

    def _line(
        self,
        marker_id: int,
        namespace: str,
        points,
        color,
        width: float = 0.02,
    ) -> Marker:
        marker = self._base_marker(marker_id, Marker.LINE_STRIP, namespace)
        marker.scale.x = width
        self._set_color(marker, *color)
        marker.points = [Point(x=float(x), y=float(y), z=float(z)) for x, y, z in points]
        return marker

    def _text(
        self,
        marker_id: int,
        namespace: str,
        x: float,
        y: float,
        z: float,
        text: str,
        color,
        size: float = 0.13,
    ) -> Marker:
        marker = self._base_marker(marker_id, Marker.TEXT_VIEW_FACING, namespace)
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = z
        marker.scale.z = size
        marker.text = text
        self._set_color(marker, *color)
        return marker

    def _box(self, marker_id: int, namespace: str, vehicle: Vehicle, color, height: float) -> Marker:
        marker = self._base_marker(marker_id, Marker.CUBE, namespace)
        marker.pose.position.x = vehicle.x
        marker.pose.position.y = vehicle.y
        marker.pose.position.z = 0.5 * height
        qx, qy, qz, qw = yaw_to_quaternion(vehicle.heading)
        marker.pose.orientation.x = qx
        marker.pose.orientation.y = qy
        marker.pose.orientation.z = qz
        marker.pose.orientation.w = qw
        marker.scale.x = vehicle.length
        marker.scale.y = vehicle.width
        marker.scale.z = height
        self._set_color(marker, *color)
        return marker

    def _arena_markers(self) -> List[Marker]:
        markers: List[Marker] = []
        for index, center in enumerate(self.config.arena.lane_centers):
            markers.append(
                self._line(
                    10 + index,
                    "lanes",
                    ((self.config.grid.x_min, center, 0.012), (self.config.grid.x_max, center, 0.012)),
                    (0.82, 0.82, 0.82, 0.65),
                    0.012,
                )
            )
            markers.append(
                self._text(
                    20 + index,
                    "lane_labels",
                    0.15,
                    center,
                    0.08,
                    ("LEFT", "MIDDLE", "RIGHT")[index],
                    (0.9, 0.9, 0.9, 0.9),
                    0.09,
                )
            )
        for index, boundary in enumerate(
            (self.config.grid.road_y_min, self.config.grid.road_y_max)
        ):
            markers.append(
                self._line(
                    30 + index,
                    "road_bounds",
                    ((self.config.grid.x_min, boundary, 0.01), (self.config.grid.x_max, boundary, 0.01)),
                    (1.0, 1.0, 1.0, 0.45),
                    0.02,
                )
            )
        conflict = Vehicle(
            "conflict_zone",
            0.5
            * (
                self.config.arena.conflict_zone_x_min
                + self.config.arena.conflict_zone_x_max
            ),
            self.config.arena.lane_centers[self.config.arena.target_lane],
            length=(
                self.config.arena.conflict_zone_x_max
                - self.config.arena.conflict_zone_x_min
            ),
            width=0.82 * self.config.arena.lane_width,
        )
        markers.append(
            self._box(40, "conflict_zone", conflict, (1.0, 0.05, 0.02, 0.16), 0.018)
        )
        markers.append(
            self._text(
                41,
                "conflict_zone_label",
                conflict.x,
                conflict.y,
                0.10,
                "SHARED MIDDLE-LANE CONFLICT ZONE",
                (1.0, 0.25, 0.05, 1.0),
                0.085,
            )
        )
        route_x = np.linspace(
            self.config.arena.merge_request_x,
            self.config.arena.conflict_zone_x_max,
            80,
        )
        route_y = anchored_lane_change_y(
            route_x,
            source_y=self.config.arena.lane_centers[self.config.arena.ego_lane],
            target_y=self.config.arena.lane_centers[self.config.arena.target_lane],
            start_x=self.config.arena.merge_path_x_min,
            end_x=self.config.arena.merge_path_x_max,
        )
        markers.append(
            self._line(
                42,
                "route_intent",
                tuple((x, y, 0.035) for x, y in zip(route_x, route_y)),
                (0.95, 0.55, 0.05, 0.55),
                0.018,
            )
        )
        return markers

    def _world_markers(self) -> List[Marker]:
        markers: List[Marker] = []
        next_id = 100
        for vehicle in self.world_vehicles:
            if vehicle.vehicle_id == "merger_odom":
                continue
            if vehicle.vehicle_id == "truck":
                color, height, label = (0.95, 0.48, 0.08, 0.95), 0.34, "OCCLUDING TRUCK"
            else:
                color, height, label = (0.8, 0.8, 0.2, 0.9), 0.20, vehicle.vehicle_id
            markers.append(self._box(next_id, "planner_world", vehicle, color, height))
            markers.append(
                self._text(
                    next_id + 1,
                    "planner_world_labels",
                    vehicle.x,
                    vehicle.y,
                    height + 0.08,
                    label,
                    (color[0], color[1], color[2], 1.0),
                    0.10,
                )
            )
            next_id += 2
        return markers

    def _ego_markers(self) -> List[Marker]:
        if self.ego is None:
            return []
        ego_vehicle = self.ego.as_vehicle()
        ego_vehicle = Vehicle(
            ego_vehicle.vehicle_id,
            ego_vehicle.x,
            ego_vehicle.y,
            vx=ego_vehicle.vx,
            vy=ego_vehicle.vy,
            heading=self.ego.yaw,
            length=0.22,
            width=0.22,
        )
        box = self._box(200, "ego", ego_vehicle, (0.05, 0.35, 1.0, 1.0), 0.18)
        arrow = self._base_marker(201, Marker.ARROW, "ego_heading")
        arrow.pose.position.x = self.ego.x
        arrow.pose.position.y = self.ego.y
        arrow.pose.position.z = 0.20
        qx, qy, qz, qw = yaw_to_quaternion(self.ego.yaw)
        arrow.pose.orientation.x = qx
        arrow.pose.orientation.y = qy
        arrow.pose.orientation.z = qz
        arrow.pose.orientation.w = qw
        arrow.scale.x = 0.35
        arrow.scale.y = 0.07
        arrow.scale.z = 0.07
        self._set_color(arrow, 0.05, 0.75, 1.0, 1.0)
        return [box, arrow]

    def _merger_markers(self) -> List[Marker]:
        if self.merger is None:
            return []
        if self.merger_visible:
            color = (1.0, 0.08, 0.08, 1.0)
            label = "MERGER REVEALED -> PLANNER TRACK"
            line_color = (0.1, 1.0, 0.1, 0.9)
        else:
            color = (0.72, 0.15, 0.95, 0.35)
            label = "HIDDEN MERGER (GROUND TRUTH ONLY)"
            line_color = (1.0, 0.1, 0.1, 0.8)
        markers = [
            self._box(300, "merger_truth", self.merger, color, 0.20),
            self._text(
                301,
                "merger_truth_label",
                self.merger.x,
                self.merger.y,
                0.33,
                label,
                (color[0], color[1], color[2], 1.0),
                0.10,
            ),
        ]
        if self.ego is not None:
            markers.append(
                self._line(
                    302,
                    "line_of_sight",
                    ((self.ego.x, self.ego.y, 0.23), (self.merger.x, self.merger.y, 0.23)),
                    line_color,
                    0.018,
                )
            )
        return markers

    def _decision_markers(self) -> List[Marker]:
        if self.ego is None or not self.planner:
            return []
        centers = self.config.arena.lane_centers
        current = int(self.planner.get("current_lane", 0))
        requested = int(self.planner.get("requested_lane", current))
        selected = int(self.planner.get("selected_lane", current))
        requested = max(0, min(requested, len(centers) - 1))
        selected = max(0, min(selected, len(centers) - 1))
        x_end = min(self.config.grid.x_max, self.ego.x + 1.2)
        markers = [
            self._line(
                400,
                "requested_maneuver",
                ((self.ego.x, self.ego.y, 0.27), (x_end, centers[requested], 0.27)),
                (1.0, 0.58, 0.05, 0.95),
                0.025,
            ),
            self._line(
                401,
                "selected_maneuver",
                ((self.ego.x, self.ego.y, 0.30), (x_end, centers[selected], 0.30)),
                (0.05, 1.0, 0.20, 0.95),
                0.035,
            ),
        ]
        vetoed = bool(self.planner.get("vetoed", False))
        control_stack = str(self.planner.get("control_stack", "dream"))
        if vetoed:
            headline = "DREAM VETO ACTIVE - KEEP / YIELD"
            color = (1.0, 0.12, 0.05, 1.0)
        elif control_stack == "pure_mpc":
            headline = "PURE MPC BASELINE - ROUTE MERGE"
            color = (0.95, 0.58, 0.05, 1.0)
        else:
            headline = f"DREAM {self.planner.get('maneuver', 'K')} - NO VETO"
            color = (0.1, 1.0, 0.25, 1.0)
        detail = (
            f"{headline}\n"
            f"risk={float(self.planner.get('decision_risk', 0.0)):.2f}  "
            f"v={float(self.planner.get('target_speed', 0.0)):.2f} m/s"
        )
        markers.append(
            self._text(402, "decision_text", 1.45, -0.78, 0.55, detail, color, 0.11)
        )
        return markers

    def _status_markers(self) -> List[Marker]:
        phase = str(
            self.scenario.get("phase", str(self.get_parameter("mode_label").value))
        )
        safety_reason = str(self.safety.get("reason", "WAITING"))
        visibility_source = str(self.world_meta.get("visibility_source", "unknown"))
        track_count = int(self.world_meta.get("dynamic_track_count", 0))
        visible = world_visibility_label(
            merger_visible=self.merger_visible,
            visibility_source=visibility_source,
            dynamic_track_count=track_count,
        )
        scenario_text = (
            f"SCENARIO: {phase} | MERGER: {visible}\n"
            f"SAFETY: {safety_reason}"
        )
        metric_text = (
            f"veto steps: {int(self.metrics.get('veto_activations', 0))}  "
            f"risk@ego: {float(self.metrics.get('risk_at_ego', 0.0)):.2f}\n"
            f"min clearance: {self.metrics.get('minimum_clearance', 'n/a')}  "
            f"max MPC: {1000.0 * float(self.metrics.get('maximum_mpc_seconds', 0.0)):.0f} ms"
        )
        markers = [
            self._text(500, "scenario_text", 3.0, 0.82, 0.35, scenario_text, (1.0, 1.0, 1.0, 1.0), 0.105),
            self._text(501, "metrics_text", 4.65, -0.78, 0.35, metric_text, (0.2, 0.95, 1.0, 1.0), 0.09),
        ]
        if self.smoke:
            passed = bool(self.smoke.get("passed", False))
            arm = str(self.smoke.get("experiment_arm", "SIL")).upper()
            markers.append(
                self._text(
                    502,
                    "smoke_result",
                    5.20,
                    0.58,
                    0.48,
                    f"{arm} SIL: {'PASS' if passed else 'FAIL'}",
                    (0.1, 1.0, 0.2, 1.0) if passed else (1.0, 0.1, 0.05, 1.0),
                    0.16,
                )
            )
        return markers

    def _publish(self) -> None:
        delete = self._base_marker(0, Marker.CUBE, "clear")
        delete.action = Marker.DELETEALL
        markers = [delete]
        markers.extend(self._arena_markers())
        markers.extend(self._world_markers())
        markers.extend(self._ego_markers())
        markers.extend(self._merger_markers())
        markers.extend(self._decision_markers())
        markers.extend(self._status_markers())
        message = MarkerArray()
        message.markers = markers
        self.publisher.publish(message)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DreamVisualizationNode()
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
