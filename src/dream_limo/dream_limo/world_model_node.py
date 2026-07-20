"""Perception-driven world model with a surveyed SIL fallback."""

from __future__ import annotations

import json
from math import atan2, cos, hypot, sin
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, String

from .core.occlusion import (
    LidarShadowBuilder,
    PlanarScan,
    PolygonObstacle,
    line_of_sight_visible,
    rectangle_polygon,
    scan_line_of_sight_visible,
)
from .core.risk_field import DREAMRiskField
from .core.route import anchored_lane_change_y
from .core.types import EgoState, Vehicle, parse_tracked_agents
from .limo_scale import default_deployment_config, deployment_config_for_arena
from .ros_utils import (
    child_velocity_to_parent,
    ego_from_odometry,
    quaternion_to_yaw,
    stamp_to_seconds,
    transform_planar,
    vehicle_to_mapping,
)


def evaluate_merger_adapter_status(
    payload,
    *,
    expected_output_frame: str,
    expected_output_child_frame: str,
) -> Tuple[bool, str]:
    """Validate the adapter contract without accepting truthy substitutes."""
    if not isinstance(payload, dict):
        return False, "INVALID_STATUS_PAYLOAD"
    required_true_fields = (
        "ready",
        "input_fresh",
        "last_message_valid",
        "alignment_verified",
        "alignment_initialized",
    )
    for field in required_true_fields:
        if payload.get(field) is not True:
            return False, str(payload.get("reason", f"{field.upper()}_FALSE"))
    if payload.get("output_frame") != expected_output_frame:
        return False, "OUTPUT_FRAME_MISMATCH"
    if payload.get("output_child_frame") != expected_output_child_frame:
        return False, "OUTPUT_CHILD_FRAME_MISMATCH"
    return True, "READY"


def evaluate_dynamic_source_fresh(
    *,
    perception_tracks_fresh: bool,
    merger_odom_required: bool,
    merger_inputs_ready: bool,
) -> bool:
    """Select the freshness contract used by the hardware command gate.

    In measured second-LIMO mode the aligned odometry adapter is the required
    dynamic source, so an intentionally disabled SFG tracker must not prevent
    motion.  Conversely, a fresh SFG heartbeat must not hide an adapter fault.
    """
    if merger_odom_required:
        return bool(merger_inputs_ready)
    return bool(perception_tracks_fresh)


def select_perception_tracks(
    agents: List[Vehicle],
    *,
    perception_tracks_fresh: bool,
    merger_odom_required: bool,
) -> List[Vehicle]:
    """Use exactly one dynamic-object provider.

    An aligned second-LIMO stream is scan-gated later in the world-model
    update.  When that mode is selected, ignore every ``/tracked_agents``
    message, including one from an unexpectedly running external publisher,
    so it cannot duplicate or bypass the gated merger track.
    """
    if merger_odom_required or not perception_tracks_fresh:
        return []
    return list(agents)


class DreamWorldModel(Node):
    def __init__(self) -> None:
        super().__init__("dream_world_model")
        self.config = default_deployment_config()
        self.declare_parameter("ego_topic", "/dream/ego_state")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("tracked_agents_topic", "/tracked_agents")
        self.declare_parameter("tracked_agents_frame", "odom")
        self.declare_parameter("arena_frame", "map")
        self.declare_parameter("arena_file", "")
        self.declare_parameter("occlusion_source", "lidar_first_return")
        self.declare_parameter("map_to_odom_x", 0.0)
        self.declare_parameter("map_to_odom_y", 0.0)
        self.declare_parameter("map_to_odom_yaw", 0.0)
        self.declare_parameter("alignment_topic", "/dream/map_alignment")
        self.declare_parameter("laser_x", 0.10)
        self.declare_parameter("laser_y", 0.0)
        self.declare_parameter("laser_yaw", 0.0)
        self.declare_parameter("track_timeout", 0.8)
        self.declare_parameter("ego_timeout", 0.30)
        self.declare_parameter("scan_timeout", 0.40)
        self.declare_parameter("use_merger_odom", False)
        self.declare_parameter("merger_odom_topic", "/merger/wheel/odom")
        self.declare_parameter("merger_odom_frame", "odom")
        self.declare_parameter("merger_odom_child_frame", "merger/base_link")
        self.declare_parameter(
            "merger_adapter_status_topic",
            "/dream/merger_odometry_adapter_status",
        )
        self.declare_parameter("merger_adapter_status_timeout", 0.30)
        self.declare_parameter("publish_rate", 10.0)

        self.config = deployment_config_for_arena(
            str(self.get_parameter("arena_file").value)
        )
        self.use_merger_odom = bool(self.get_parameter("use_merger_odom").value)
        self.merger_adapter_status_timeout = float(
            self.get_parameter("merger_adapter_status_timeout").value
        )
        if (
            not np.isfinite(self.merger_adapter_status_timeout)
            or self.merger_adapter_status_timeout <= 0.0
        ):
            raise RuntimeError("merger_adapter_status_timeout must be positive")
        self.grid_helper = DREAMRiskField(self.config)

        surveyed_vehicles, surveyed_obstacles = self._load_arena(
            str(self.get_parameter("arena_file").value)
        )
        self.occlusion_source = str(self.get_parameter("occlusion_source").value)
        if self.occlusion_source not in {"lidar_first_return", "surveyed_polygon"}:
            raise RuntimeError(
                "occlusion_source must be 'lidar_first_return' or 'surveyed_polygon'"
            )
        if self.occlusion_source == "surveyed_polygon":
            if not any(item.vehicle_class == "truck" for item in surveyed_vehicles):
                raise RuntimeError(
                    "surveyed_polygon mode requires at least one truck-class occluder"
                )
            self.static_vehicles = surveyed_vehicles
            self.obstacles = surveyed_obstacles
        else:
            # Live mode never inserts unchecked YAML geometry into DRIFT/MPC.
            # Static scene visibility comes directly from /scan.
            self.static_vehicles = []
            self.obstacles = []
        self.occluders = [item for item in self.obstacles if item.vehicle_class == "truck"]
        self.shadow_builder = LidarShadowBuilder(
            maximum_shadow_range=self.config.pde.occlusion_range,
            require_known_occluder=self.occlusion_source == "surveyed_polygon",
        )
        self.ego: Optional[EgoState] = None
        self.ego_receipt: Optional[float] = None
        self.scan: Optional[PlanarScan] = None
        self.scan_receipt: Optional[float] = None
        self.agents: List[Vehicle] = []
        self.agents_receipt: Optional[float] = None
        self.raw_merger: Optional[Vehicle] = None
        self.raw_merger_receipt: Optional[float] = None
        self.merger_odom_valid = False
        self.merger_odom_reason = "WAITING_FOR_ODOMETRY"
        self.merger_adapter_ready = False
        self.merger_adapter_reason = "WAITING_FOR_ADAPTER_STATUS"
        self.merger_adapter_status_receipt: Optional[float] = None
        self.map_alignment = (
            float(self.get_parameter("map_to_odom_x").value),
            float(self.get_parameter("map_to_odom_y").value),
            float(self.get_parameter("map_to_odom_yaw").value),
        )
        self.alignment_received = False

        reliable = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
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
            Odometry, str(self.get_parameter("ego_topic").value), self._on_ego, reliable
        )
        self.create_subscription(
            LaserScan, str(self.get_parameter("scan_topic").value), self._on_scan, sensor_qos
        )
        self.create_subscription(
            String,
            str(self.get_parameter("tracked_agents_topic").value),
            self._on_tracked_agents,
            reliable,
        )
        if self.use_merger_odom:
            self.create_subscription(
                String,
                str(self.get_parameter("merger_adapter_status_topic").value),
                self._on_merger_adapter_status,
                alignment_qos,
            )
            self.create_subscription(
                Odometry,
                str(self.get_parameter("merger_odom_topic").value),
                self._on_merger_odom,
                reliable,
            )
        self.world_publisher = self.create_publisher(String, "/dream/world_model", reliable)
        self.mask_publisher = self.create_publisher(
            OccupancyGrid, "/dream/occlusion_mask", reliable
        )
        self.visibility_publisher = self.create_publisher(Bool, "/dream/merger_visible", reliable)
        self.status_publisher = self.create_publisher(String, "/dream/world_status", reliable)
        period = 1.0 / float(self.get_parameter("publish_rate").value)
        self.create_timer(period, self._publish)

    def _load_arena(self, path_text: str) -> Tuple[List[Vehicle], List[PolygonObstacle]]:
        if not path_text:
            return [], []
        path = Path(path_text).expanduser()
        with path.open("r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream) or {}
        entries = data.get("obstacles", [])
        vehicles: List[Vehicle] = []
        polygons: List[PolygonObstacle] = []
        for entry in entries:
            center_x, center_y = (float(value) for value in entry["center"])
            length, width = (float(value) for value in entry["size"])
            heading = float(entry.get("heading", 0.0))
            vehicle_class = str(entry.get("class", "car"))
            name = str(entry["id"])
            polygons.append(
                rectangle_polygon(
                    name,
                    center_x,
                    center_y,
                    length,
                    width,
                    heading,
                    vehicle_class,
                )
            )
            vehicles.append(
                Vehicle(
                    name,
                    center_x,
                    center_y,
                    heading=heading,
                    vehicle_class=vehicle_class,
                    length=length,
                    width=width,
                )
            )
        return vehicles, polygons

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1.0e-9

    def _on_ego(self, message: Odometry) -> None:
        self.ego = ego_from_odometry(message)
        self.ego_receipt = self._now()

    def _on_scan(self, message: LaserScan) -> None:
        if self.ego is None:
            return
        ego = self.ego
        base_yaw = ego.yaw
        laser_x = float(self.get_parameter("laser_x").value)
        laser_y = float(self.get_parameter("laser_y").value)
        sensor_x = ego.x + cos(base_yaw) * laser_x - sin(base_yaw) * laser_y
        sensor_y = ego.y + sin(base_yaw) * laser_x + cos(base_yaw) * laser_y
        self.scan = PlanarScan(
            ranges=np.asarray(message.ranges, dtype=np.float64),
            angle_min=float(message.angle_min),
            angle_increment=float(message.angle_increment),
            range_min=float(message.range_min),
            range_max=float(message.range_max),
            sensor_x=sensor_x,
            sensor_y=sensor_y,
            sensor_yaw=base_yaw + float(self.get_parameter("laser_yaw").value),
            stamp=self._now(),
        )
        self.scan_receipt = self._now()

    def _on_alignment(self, message: TransformStamped) -> None:
        self.map_alignment = (
            float(message.transform.translation.x),
            float(message.transform.translation.y),
            quaternion_to_yaw(message.transform.rotation),
        )
        self.alignment_received = True

    def _transform_from_odom(self, x: float, y: float, vx: float, vy: float):
        tx, ty, yaw = self.map_alignment
        return transform_planar(
            x,
            y,
            vx,
            vy,
            tx=tx,
            ty=ty,
            yaw=yaw,
        )

    def _on_tracked_agents(self, message: String) -> None:
        now = self._now()
        try:
            raw = json.loads(message.data)
            parsed = parse_tracked_agents(
                raw,
                now=now,
                maximum_observation_age=float(self.get_parameter("track_timeout").value),
            )
            vehicles = []
            for agent in parsed:
                x, y, vx, vy = self._transform_from_odom(agent.x, agent.y, agent.vx, agent.vy)
                vehicles.append(
                    Vehicle(
                        vehicle_id=agent.agent_id,
                        x=x,
                        y=y,
                        vx=vx,
                        vy=vy,
                        heading=atan2(vy, vx) if hypot(vx, vy) > 1.0e-6 else 0.0,
                        vehicle_class=agent.class_label,
                        length=max(0.22, 2.0 * agent.radius),
                        width=max(0.10, 2.0 * agent.radius),
                        confidence=agent.confidence,
                        stamp=agent.stamp,
                    )
                )
            self.agents = vehicles
            self.agents_receipt = now
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            self.get_logger().warning(f"Rejected tracked_agents payload: {exc}")

    def _on_merger_adapter_status(self, message: String) -> None:
        now = self._now()
        try:
            payload = json.loads(message.data)
            ready, reason = evaluate_merger_adapter_status(
                payload,
                expected_output_frame=str(
                    self.get_parameter("merger_odom_frame").value
                ),
                expected_output_child_frame=str(
                    self.get_parameter("merger_odom_child_frame").value
                ),
            )
        except (json.JSONDecodeError, TypeError, ValueError):
            ready, reason = False, "INVALID_STATUS_PAYLOAD"
        self.merger_adapter_ready = ready
        self.merger_adapter_reason = reason
        self.merger_adapter_status_receipt = now
        if not ready:
            self._clear_merger_odom(reason)

    def _merger_adapter_is_ready(self, now: float) -> Tuple[bool, bool]:
        fresh = (
            self.merger_adapter_status_receipt is not None
            and now - self.merger_adapter_status_receipt
            < self.merger_adapter_status_timeout
        )
        return self.merger_adapter_ready and fresh, fresh

    def _clear_merger_odom(self, reason: str) -> None:
        self.raw_merger = None
        self.raw_merger_receipt = None
        self.merger_odom_valid = False
        self.merger_odom_reason = reason

    def _on_merger_odom(self, message: Odometry) -> None:
        now = self._now()
        adapter_ready, _ = self._merger_adapter_is_ready(now)
        if not adapter_ready:
            self._clear_merger_odom("ADAPTER_NOT_READY")
            return
        expected_frame = str(self.get_parameter("merger_odom_frame").value)
        expected_child_frame = str(
            self.get_parameter("merger_odom_child_frame").value
        )
        if message.header.frame_id != expected_frame:
            self._clear_merger_odom("ODOMETRY_FRAME_MISMATCH")
            return
        if message.child_frame_id != expected_child_frame:
            self._clear_merger_odom("ODOMETRY_CHILD_FRAME_MISMATCH")
            return
        position = message.pose.pose.position
        velocity = message.twist.twist.linear
        source_yaw = quaternion_to_yaw(message.pose.pose.orientation)
        source_stamp = stamp_to_seconds(message.header.stamp)
        values = (
            position.x,
            position.y,
            source_yaw,
            velocity.x,
            velocity.y,
            source_stamp,
        )
        if not all(np.isfinite(value) for value in values) or source_stamp <= 0.0:
            self._clear_merger_odom("INVALID_ODOMETRY")
            return
        parent_vx, parent_vy = child_velocity_to_parent(
            velocity.x,
            velocity.y,
            child_yaw=source_yaw,
        )
        arena_frame = str(self.get_parameter("arena_frame").value)
        if expected_frame == arena_frame:
            x, y = float(position.x), float(position.y)
            vx, vy = parent_vx, parent_vy
            map_yaw = source_yaw
        else:
            x, y, vx, vy = self._transform_from_odom(
                position.x,
                position.y,
                parent_vx,
                parent_vy,
            )
            map_yaw = self.map_alignment[2] + source_yaw
        self.raw_merger = Vehicle(
            "merger_odom",
            x,
            y,
            vx=vx,
            vy=vy,
            heading=map_yaw,
            vehicle_class="car",
            length=0.22,
            width=0.22,
            stamp=source_stamp,
        )
        self.raw_merger_receipt = now
        self.merger_odom_valid = True
        self.merger_odom_reason = "READY"

    def _publish_mask(self, mask: np.ndarray, stamp) -> None:
        message = OccupancyGrid()
        message.header.stamp = stamp
        message.header.frame_id = self.config.grid.frame_id
        message.info.resolution = self.config.grid.resolution
        message.info.width = self.config.grid.nx
        message.info.height = self.config.grid.ny
        message.info.origin.position.x = self.config.grid.x_min
        message.info.origin.position.y = self.config.grid.y_min
        message.info.origin.orientation.w = 1.0
        message.data = np.rint(np.clip(mask, 0.0, 1.0) * 100.0).astype(np.int8).ravel().tolist()
        self.mask_publisher.publish(message)

    def _route_shadow_samples(self, mask: np.ndarray) -> Tuple[int, int]:
        if self.ego is None:
            return 0, 0
        arena = self.config.arena
        count = max(2, arena.veto_samples)
        x_values = np.linspace(
            self.ego.x,
            min(self.config.grid.x_max, self.ego.x + arena.veto_lookahead),
            count,
        )
        y_values = anchored_lane_change_y(
            x_values,
            source_y=arena.lane_centers[arena.ego_lane],
            target_y=arena.lane_centers[arena.target_lane],
            start_x=arena.merge_path_x_min,
            end_x=arena.merge_path_x_max,
        )
        ix = np.rint(
            (x_values - self.config.grid.x_min) / self.config.grid.resolution
        ).astype(int)
        iy = np.rint(
            (y_values - self.config.grid.y_min) / self.config.grid.resolution
        ).astype(int)
        valid = (
            (ix >= 0)
            & (ix < self.config.grid.nx)
            & (iy >= 0)
            & (iy < self.config.grid.ny)
        )
        if not np.any(valid):
            return 0, 0
        radius = max(
            1,
            int(np.ceil(0.5 * arena.lane_width / self.config.grid.resolution)),
        )
        overlap_count = 0
        for x_index, y_index in zip(ix[valid], iy[valid]):
            x0 = max(0, x_index - radius)
            x1 = min(self.config.grid.nx, x_index + radius + 1)
            y0 = max(0, y_index - radius)
            y1 = min(self.config.grid.ny, y_index + radius + 1)
            if np.any(mask[y0:y1, x0:x1] > 0.0):
                overlap_count += 1
        return overlap_count, int(np.count_nonzero(valid))

    def _publish(self) -> None:
        now = self._now()
        ego_fresh = self.ego is not None and now - self.ego_receipt < float(
            self.get_parameter("ego_timeout").value
        )
        scan_fresh = self.scan is not None and now - self.scan_receipt < float(
            self.get_parameter("scan_timeout").value
        )
        perception_tracks_fresh = (
            self.agents_receipt is not None
            and now - self.agents_receipt
            < float(self.get_parameter("track_timeout").value)
        )
        if self.use_merger_odom:
            merger_adapter_ready, merger_adapter_fresh = (
                self._merger_adapter_is_ready(now)
            )
            merger_odom_fresh = (
                self.merger_odom_valid
                and self.raw_merger is not None
                and self.raw_merger_receipt is not None
                and now - self.raw_merger_receipt
                < float(self.get_parameter("track_timeout").value)
            )
            if not merger_adapter_fresh:
                merger_adapter_reason = "STALE_ADAPTER_STATUS"
                self._clear_merger_odom(merger_adapter_reason)
                merger_odom_fresh = False
            else:
                merger_adapter_reason = self.merger_adapter_reason
            merger_inputs_ready = merger_adapter_ready and merger_odom_fresh
        else:
            merger_adapter_ready = True
            merger_adapter_fresh = True
            merger_adapter_reason = "NOT_REQUIRED"
            merger_odom_fresh = True
            merger_inputs_ready = True
        tracks_fresh = evaluate_dynamic_source_fresh(
            perception_tracks_fresh=perception_tracks_fresh,
            merger_odom_required=self.use_merger_odom,
            merger_inputs_ready=merger_inputs_ready,
        )
        merger_contract = {
            "merger_adapter_required": self.use_merger_odom,
            "merger_adapter_ready": merger_adapter_ready,
            "merger_adapter_status_fresh": merger_adapter_fresh,
            "merger_adapter_reason": merger_adapter_reason,
            "merger_odom_valid": (
                self.merger_odom_valid if self.use_merger_odom else True
            ),
            "merger_odom_fresh": merger_odom_fresh,
            "merger_odom_reason": (
                self.merger_odom_reason if self.use_merger_odom else "NOT_REQUIRED"
            ),
            "perception_tracks_fresh": perception_tracks_fresh,
            "dynamic_source_fresh": tracks_fresh,
        }
        visible_merger = False
        dynamic = select_perception_tracks(
            self.agents,
            perception_tracks_fresh=perception_tracks_fresh,
            merger_odom_required=self.use_merger_odom,
        )
        if self.occlusion_source == "lidar_first_return" and scan_fresh:
            # SFG publishes only perceived tracks, but its tracker may briefly
            # coast a track after it moves behind a surface. Never leak such a
            # state into Q_veh: the same measured visibility gate applies.
            dynamic = [
                item
                for item in dynamic
                if scan_line_of_sight_visible(
                    self.scan,
                    (item.x, item.y),
                    target_radius=0.5 * max(item.length, item.width),
                )
            ]
        if (
            self.use_merger_odom
            and merger_inputs_ready
            and ego_fresh
            and self.raw_merger is not None
        ):
            observer = (
                (self.scan.sensor_x, self.scan.sensor_y)
                if scan_fresh
                else (self.ego.x, self.ego.y)
            )
            if self.occlusion_source == "lidar_first_return" and scan_fresh:
                visible_merger = scan_line_of_sight_visible(
                    self.scan,
                    (self.raw_merger.x, self.raw_merger.y),
                    target_radius=0.5 * max(self.raw_merger.length, self.raw_merger.width),
                )
            else:
                visible_merger = line_of_sight_visible(
                    observer,
                    (self.raw_merger.x, self.raw_merger.y),
                    self.occluders,
                )
            if visible_merger:
                dynamic.append(self.raw_merger)
        visible_message = Bool()
        visible_message.data = visible_merger
        self.visibility_publisher.publish(visible_message)

        if (
            not ego_fresh
            or not scan_fresh
            or not self.alignment_received
            or not merger_inputs_ready
        ):
            status = String()
            status.data = json.dumps(
                {
                    "ready": False,
                    "ego_fresh": ego_fresh,
                    "scan_fresh": scan_fresh,
                    "tracks_fresh": tracks_fresh,
                    "occlusion_source": self.occlusion_source,
                    "alignment_received": self.alignment_received,
                    **merger_contract,
                },
                separators=(",", ":"),
            )
            self.status_publisher.publish(status)
            return
        mask = self.shadow_builder.build(
            self.grid_helper.X,
            self.grid_helper.Y,
            self.grid_helper.road_mask,
            self.scan,
            self.occluders,
        )
        shadow_route_samples, route_samples = self._route_shadow_samples(mask)
        stamp = self.get_clock().now().to_msg()
        self._publish_mask(mask, stamp)
        world = String()
        world.data = json.dumps(
            {
                "stamp": now,
                "frame_id": self.config.grid.frame_id,
                "vehicles": [
                    vehicle_to_mapping(item)
                    for item in [*self.static_vehicles, *dynamic]
                ],
                "static_vehicle_ids": [item.vehicle_id for item in self.static_vehicles],
                "merger_visible": visible_merger,
                "visibility_source": (
                    "merger_odom_gate"
                    if self.use_merger_odom
                    else "perception_only_no_merger_ground_truth"
                ),
                "dynamic_track_count": len(dynamic),
                "shadow_cells": int(np.count_nonzero(mask)),
                "shadow_route_samples": shadow_route_samples,
                "route_samples": route_samples,
                "tracks_fresh": tracks_fresh,
                "occlusion_source": self.occlusion_source,
                "surveyed_static_geometry_used": self.occlusion_source
                == "surveyed_polygon",
                "alignment_received": self.alignment_received,
                **merger_contract,
            },
            separators=(",", ":"),
        )
        self.world_publisher.publish(world)
        status = String()
        status.data = json.dumps(
            {
                "ready": self.alignment_received,
                "ego_fresh": True,
                "scan_fresh": True,
                "tracks_fresh": tracks_fresh,
                "merger_visible": visible_merger,
                "visibility_source": (
                    "merger_odom_gate"
                    if self.use_merger_odom
                    else "perception_only_no_merger_ground_truth"
                ),
                "dynamic_track_count": len(dynamic),
                "shadow_cells": int(np.count_nonzero(mask)),
                "shadow_route_samples": shadow_route_samples,
                "route_samples": route_samples,
                "occlusion_source": self.occlusion_source,
                "surveyed_static_geometry_used": self.occlusion_source
                == "surveyed_polygon",
                "alignment_received": self.alignment_received,
                **merger_contract,
            },
            separators=(",", ":"),
        )
        self.status_publisher.publish(status)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DreamWorldModel()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except RuntimeError:
        if rclpy.ok():
            raise
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
