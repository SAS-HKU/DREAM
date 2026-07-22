"""DREAM risk gating and MPC tracking for an arbitrary Nav2 geometric path."""

from __future__ import annotations

import json
from dataclasses import replace
from math import asin, atan2, hypot, isfinite, pi
from typing import List, Optional

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String

from .core.command_adapter import gate_mpc_output
from .core.free_goal import CostmapSnapshot
from .core.free_decision import (
    evaluate_route_maneuver_risk,
)
from .core.inflated_costmap import validate_swept_trajectory
from .core.mpc import RiskAwareMPC
from .core.nav2_route import goal_identity_matches
from .core.path_tracking import (
    PathValidationError,
    anchor_local_path_start,
    validate_forward_pose_alignment,
    validate_path_points,
)
from .core.risk_field import DREAMRiskField
from .core.types import EgoState, Vehicle
from .limo_scale import deployment_config_for_arena, get_preset
from .ros_utils import (
    ego_from_odometry,
    quaternion_to_yaw,
    stamp_to_seconds,
    vehicle_from_mapping,
    yaw_to_quaternion,
)


class DreamFreePlannerNode(Node):
    """Keep Nav2 geometric planning separate from DREAM's sole controller."""

    def __init__(self) -> None:
        super().__init__("dream_free_planner")
        self.declare_parameter("arena_file", "")
        self.declare_parameter("preset", "balanced")
        self.declare_parameter("target_speed", 0.15)
        self.declare_parameter("ego_topic", "/dream/ego_state")
        self.declare_parameter("world_topic", "/dream/world_model")
        self.declare_parameter("risk_topic", "/dream/risk_field_raw")
        self.declare_parameter("drift_ready_topic", "/dream/drift_ready")
        self.declare_parameter("goal_topic", "/dream/navigation_goal")
        self.declare_parameter("path_topic", "/dream/geometric_path")
        self.declare_parameter("route_status_topic", "/dream/route_status")
        self.declare_parameter("costmap_topic", "/global_costmap/costmap")
        self.declare_parameter("update_rate", 5.0)
        self.declare_parameter("input_timeout", 0.50)
        self.declare_parameter("costmap_timeout", 0.75)
        self.declare_parameter("path_timeout", 1.50)
        self.declare_parameter("route_status_timeout", 1.50)
        self.declare_parameter("path_goal_tolerance", 0.10)
        self.declare_parameter("path_goal_yaw_tolerance", 0.30)
        # SMAC can omit the exact planning start and begin at its first
        # reachable lattice pose.  This separate local bound lets us certify
        # that short start segment without weakening the ordinary 0.10 m
        # off-route/cross-track rejection used by DREAM and the MPC.
        self.declare_parameter("path_start_anchor_tolerance", 0.20)
        self.declare_parameter("goal_match_tolerance", 1.0e-3)
        self.declare_parameter("path_stamp_tolerance", 1.0e-6)
        self.declare_parameter("goal_position_tolerance", 0.12)
        self.declare_parameter("goal_stop_speed_tolerance", 0.03)
        self.declare_parameter("goal_yaw_tolerance", 0.30)
        self.declare_parameter("source_future_tolerance", 0.10)
        self.declare_parameter("risk_lookahead", 3.0)
        self.declare_parameter("risk_samples", 10)
        self.declare_parameter("maximum_allowed_cbf_slack", 0.05)
        self.declare_parameter("enforce_map_bounds", True)
        self.declare_parameter("verified_start_clearance_enabled", False)
        self.declare_parameter("verified_start_clearance_radius", 0.30)

        self.config = deployment_config_for_arena(
            str(self.get_parameter("arena_file").value)
        )
        target_speed = float(self.get_parameter("target_speed").value)
        self.config = replace(
            self.config,
            mpc=replace(self.config.mpc, target_speed=target_speed),
        )
        self.preset = get_preset(str(self.get_parameter("preset").value))
        if self.preset.name not in {"balanced", "pure_mpc"}:
            raise ValueError("free navigation supports balanced or pure_mpc")
        self.field = DREAMRiskField(self.config)
        self.mpc = RiskAwareMPC(
            self.config,
            enforce_map_bounds=bool(
                self.get_parameter("enforce_map_bounds").value
            ),
        )
        self.verified_start_clearance_enabled = bool(
            self.get_parameter("verified_start_clearance_enabled").value
        )
        self.verified_start_clearance_radius = float(
            self.get_parameter("verified_start_clearance_radius").value
        )
        padded_radius = hypot(
            0.5 * self.config.mpc.robot_length
            + self.config.mpc.navigation_footprint_padding,
            0.5 * self.config.mpc.robot_width
            + self.config.mpc.navigation_footprint_padding,
        )
        if (
            not isfinite(self.verified_start_clearance_radius)
            or self.verified_start_clearance_radius < padded_radius
        ):
            raise ValueError(
                "verified start-clearance radius must cover the complete "
                "padded robot footprint"
            )

        self.ego: Optional[EgoState] = None
        self.ego_receipt: Optional[float] = None
        self.ego_source_stamp: Optional[float] = None
        self.vehicles: List[Vehicle] = []
        self.world_receipt: Optional[float] = None
        self.world_source_stamp: Optional[float] = None
        self.risk_receipt: Optional[float] = None
        self.risk_source_stamp: Optional[float] = None
        self.ready_receipt: Optional[float] = None
        self.drift_ready = False
        self.goal: Optional[PoseStamped] = None
        self.goal_receipt: Optional[float] = None
        self.goal_yaw: Optional[float] = None
        self.path_points: Optional[np.ndarray] = None
        self.path_receipt: Optional[float] = None
        self.path_source_stamp: Optional[float] = None
        self.path_rejection_reason: Optional[str] = None
        self.route_status: dict = {}
        self.route_status_receipt: Optional[float] = None
        self.costmap: Optional[CostmapSnapshot] = None
        self.costmap_receipt: Optional[float] = None
        self.costmap_source_stamp: Optional[float] = None
        self.goal_complete = False
        self.last_goal_key: Optional[tuple[float, float, float, float]] = None
        self.reference_active = False
        self.verified_start_clearance_center: Optional[
            tuple[float, float]
        ] = None
        self.verified_start_clearance_available = (
            self.verified_start_clearance_enabled
        )

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
        self.create_subscription(
            Odometry,
            str(self.get_parameter("ego_topic").value),
            self._on_ego,
            reliable,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("world_topic").value),
            self._on_world,
            reliable,
        )
        self.create_subscription(
            Image,
            str(self.get_parameter("risk_topic").value),
            self._on_risk,
            2,
        )
        self.create_subscription(
            Bool,
            str(self.get_parameter("drift_ready_topic").value),
            self._on_ready,
            reliable,
        )
        self.create_subscription(
            PoseStamped,
            str(self.get_parameter("goal_topic").value),
            self._on_goal,
            latched,
        )
        self.create_subscription(
            Path,
            str(self.get_parameter("path_topic").value),
            self._on_path,
            latched,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("route_status_topic").value),
            self._on_route_status,
            reliable,
        )
        self.create_subscription(
            OccupancyGrid,
            str(self.get_parameter("costmap_topic").value),
            self._on_costmap,
            latched,
        )
        self.control_publisher = self.create_publisher(
            TwistStamped, "/dream/control", reliable
        )
        self.path_publisher = self.create_publisher(
            Path, "/dream/reference_trajectory", reliable
        )
        self.status_publisher = self.create_publisher(
            String, "/dream/planner_status", reliable
        )
        update_rate = float(self.get_parameter("update_rate").value)
        if not isfinite(update_rate) or update_rate <= 0.0:
            raise ValueError("free-planner update rate must be positive")
        self.create_timer(1.0 / update_rate, self._plan)

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1.0e-9

    @staticmethod
    def _goal_key(message: PoseStamped) -> tuple[float, float, float, float]:
        return (
            float(message.pose.position.x),
            float(message.pose.position.y),
            quaternion_to_yaw(message.pose.orientation),
            stamp_to_seconds(message.header.stamp),
        )

    def _on_ego(self, message: Odometry) -> None:
        self.ego = ego_from_odometry(message)
        self.ego_receipt = self._now()
        self.ego_source_stamp = stamp_to_seconds(message.header.stamp)
        if not self.verified_start_clearance_enabled:
            return
        if self.verified_start_clearance_center is None:
            self.verified_start_clearance_center = (self.ego.x, self.ego.y)
            return
        if self.verified_start_clearance_available:
            displacement = hypot(
                self.ego.x - self.verified_start_clearance_center[0],
                self.ego.y - self.verified_start_clearance_center[1],
            )
            if displacement > self.verified_start_clearance_radius:
                # This is a one-way transition.  Returning to the launch pose
                # later cannot reuse an old operator clearance attestation.
                self.verified_start_clearance_available = False
                self.get_logger().info(
                    "Exited the verified start-clearance disc; all future "
                    "footprint samples now require live costmap observation"
                )

    def _start_clearance_contract(
        self,
    ) -> tuple[Optional[tuple[float, float]], Optional[float]]:
        if (
            not self.verified_start_clearance_available
            or self.verified_start_clearance_center is None
        ):
            return None, None
        return (
            self.verified_start_clearance_center,
            self.verified_start_clearance_radius,
        )

    def _on_world(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
            if payload.get("frame_id") != self.config.grid.frame_id:
                raise ValueError("world frame mismatch")
            source_stamp = float(payload["stamp"])
            if not isfinite(source_stamp) or source_stamp <= 0.0:
                raise ValueError("world source stamp is invalid")
            self.vehicles = [
                vehicle_from_mapping(item) for item in payload.get("vehicles", [])
            ]
            self.world_receipt = self._now()
            self.world_source_stamp = source_stamp
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            self.world_receipt = None
            self.world_source_stamp = None
            self.get_logger().warning(f"Rejected world model: {exc}")

    def _on_risk(self, message: Image) -> None:
        try:
            grid = self.config.grid
            if message.header.frame_id != grid.frame_id:
                raise ValueError("risk-field frame mismatch")
            if message.encoding != "32FC1":
                raise ValueError("risk-field encoding must be 32FC1")
            if int(message.width) != grid.nx or int(message.height) != grid.ny:
                raise ValueError("risk-field dimensions mismatch")
            values = np.frombuffer(bytes(message.data), dtype=np.float32).astype(
                np.float64
            )
            if values.size != grid.nx * grid.ny:
                raise ValueError("risk-field payload length mismatch")
            values = values.reshape(self.field.shape)
            if not np.all(np.isfinite(values)):
                raise ValueError("risk-field payload is non-finite")
            self.field.R = np.clip(values, 0.0, self.config.pde.risk_ceiling)
            self.field.elapsed_model_time = self.config.pde.warmup_duration
            self.risk_receipt = self._now()
            self.risk_source_stamp = stamp_to_seconds(message.header.stamp)
            if self.risk_source_stamp <= 0.0:
                raise ValueError("risk-field source stamp is invalid")
        except (ValueError, TypeError) as exc:
            self.risk_receipt = None
            self.risk_source_stamp = None
            self.get_logger().warning(f"Rejected risk field: {exc}")

    def _on_ready(self, message: Bool) -> None:
        self.drift_ready = bool(message.data)
        self.ready_receipt = self._now()

    def _on_costmap(self, message: OccupancyGrid) -> None:
        """Retain the same fresh inflated grid used to produce the Nav2 path."""
        now = self._now()
        orientation = message.info.origin.orientation
        values = (
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w,
        )
        norm = sum(float(value) ** 2 for value in values) ** 0.5
        sin_roll_cos_pitch = 2.0 * (
            orientation.w * orientation.x + orientation.y * orientation.z
        )
        cos_roll_cos_pitch = 1.0 - 2.0 * (
            orientation.x**2 + orientation.y**2
        )
        roll = atan2(sin_roll_cos_pitch, cos_roll_cos_pitch)
        sin_pitch = 2.0 * (
            orientation.w * orientation.y - orientation.z * orientation.x
        )
        pitch = asin(max(-1.0, min(1.0, sin_pitch)))
        yaw = quaternion_to_yaw(orientation)
        try:
            if (
                not isfinite(norm)
                or abs(norm - 1.0) > 1.0e-6
                or abs(roll) > 1.0e-6
                or abs(pitch) > 1.0e-6
            ):
                raise ValueError("costmap origin is not a normalized planar pose")
            source_stamp = stamp_to_seconds(message.header.stamp)
            if source_stamp <= 0.0 or not isfinite(source_stamp):
                raise ValueError("costmap source stamp is invalid")
            self.costmap = CostmapSnapshot.from_sequence(
                frame_id=str(message.header.frame_id),
                width=int(message.info.width),
                height=int(message.info.height),
                resolution=float(message.info.resolution),
                origin_x=float(message.info.origin.position.x),
                origin_y=float(message.info.origin.position.y),
                origin_yaw=float(yaw),
                data=message.data,
                source_stamp=source_stamp,
                receipt_stamp=now,
            )
            self.costmap_receipt = now
            self.costmap_source_stamp = source_stamp
        except (TypeError, ValueError, OverflowError) as exc:
            self.costmap = None
            self.costmap_receipt = None
            self.costmap_source_stamp = None
            self.get_logger().warning(f"Rejected live navigation costmap: {exc}")

    def _on_goal(self, message: PoseStamped) -> None:
        key = self._goal_key(message)
        if message.header.frame_id == self.config.grid.frame_id and key == self.last_goal_key:
            return
        # Every replacement, including an explicit invalidation from the goal
        # authorizer, first cancels the preceding route and controller state.
        self.goal = None
        self.goal_receipt = None
        self.goal_yaw = None
        self.path_points = None
        self.path_receipt = None
        self.path_source_stamp = None
        self.path_rejection_reason = None
        self.route_status = {}
        self.route_status_receipt = None
        self.last_goal_key = None
        self.goal_complete = False
        self.mpc.reset()
        if message.header.frame_id != self.config.grid.frame_id:
            self.get_logger().warning("Ignored navigation goal with wrong frame")
            return
        values = (
            message.pose.position.x,
            message.pose.position.y,
            key[2],
            key[3],
        )
        if not all(isfinite(float(value)) for value in values):
            self.get_logger().warning("Ignored non-finite navigation goal")
            return
        self.goal = message
        self.goal_receipt = self._now()
        self.goal_yaw = quaternion_to_yaw(message.pose.orientation)
        self.last_goal_key = key

    def _on_path(self, message: Path) -> None:
        # The provider publishes an empty latched path whenever its watchdog
        # invalidates a route.  Clear the old route before validation so one
        # callback-order window can never keep stale geometry active.
        self.path_points = None
        self.path_receipt = None
        self.path_source_stamp = None
        self.path_rejection_reason = None
        if self.goal is None or message.header.frame_id != self.config.grid.frame_id:
            return
        if not message.poses:
            # An empty path is the provider's intentional fail-closed
            # invalidation signal, not malformed route data.
            return
        try:
            raw_points = np.asarray(
                [
                    (pose.pose.position.x, pose.pose.position.y)
                    for pose in message.poses
                ],
                dtype=np.float64,
            )
            path_yaws = [
                quaternion_to_yaw(pose.pose.orientation)
                for pose in message.poses
            ]
            validate_forward_pose_alignment(raw_points, path_yaws)
            points = validate_path_points(
                raw_points
            )
            inserted_start = False
            if self.ego is not None:
                points, inserted_start = anchor_local_path_start(
                    points,
                    ego_xy=(self.ego.x, self.ego.y),
                    ego_yaw=self.ego.yaw,
                    maximum_start_gap=float(
                        self.get_parameter("path_start_anchor_tolerance").value
                    ),
                )
        except PathValidationError as exc:
            self.path_rejection_reason = f"PATH_REJECTED:{exc}"
            self.get_logger().warning(f"Rejected geometric path: {exc}")
            return
        goal = self.goal.pose.position
        endpoint_error = hypot(points[-1, 0] - goal.x, points[-1, 1] - goal.y)
        if endpoint_error > float(self.get_parameter("path_goal_tolerance").value):
            self.path_rejection_reason = "PATH_GOAL_MISMATCH"
            self.get_logger().warning(
                f"Rejected path for a different goal (error={endpoint_error:.3f} m)"
            )
            return
        assert self.goal_yaw is not None
        final_yaw_error = abs(self._angle_error(path_yaws[-1], self.goal_yaw))
        if final_yaw_error > float(
            self.get_parameter("path_goal_yaw_tolerance").value
        ):
            self.path_rejection_reason = "PATH_GOAL_YAW_MISMATCH"
            self.get_logger().warning(
                "Rejected path with a different terminal orientation "
                f"(error={final_yaw_error:.3f} rad)"
            )
            return
        if inserted_start:
            # The added segment was not explicitly present in Nav2's message.
            # Require its complete padded footprint to be known and unoccupied
            # in the same live inflated costmap used for trajectory gating.
            if self.costmap is None or self.ego is None:
                self.path_rejection_reason = "PATH_START_CONTEXT_UNAVAILABLE"
                self.get_logger().warning(
                    "Rejected path-start anchor without a live costmap and ego"
                )
                return
            # Include the next Nav2 pose as well as the omitted first pose.
            # A verified blind-corner bootstrap is accepted only if the
            # complete padded footprint becomes live-costmap-known again by
            # the end of this short prefix.
            anchor_pose_count = min(3, points.shape[0])
            anchor_yaws = [self.ego.yaw]
            anchor_yaws.extend(path_yaws[: anchor_pose_count - 1])
            anchor_states = np.asarray(
                [
                    points[:anchor_pose_count, 0],
                    points[:anchor_pose_count, 1],
                    [self.ego.speed] * anchor_pose_count,
                    anchor_yaws,
                ],
                dtype=np.float64,
            )
            start_center, start_radius = self._start_clearance_contract()
            anchor_check = validate_swept_trajectory(
                anchor_states,
                self.costmap,
                expected_frame=self.config.grid.frame_id,
                robot_length=self.config.mpc.robot_length,
                robot_width=self.config.mpc.robot_width,
                footprint_padding=self.config.mpc.navigation_footprint_padding,
                inflation_radius=self.config.mpc.navigation_inflation_radius,
                interpolation_spacing=0.5 * self.costmap.resolution,
                allow_initial_inflated_center_prefix=True,
                verified_start_clearance_center=start_center,
                verified_start_clearance_radius=start_radius,
            )
            if not anchor_check.safe:
                self.path_rejection_reason = (
                    f"PATH_START_{anchor_check.reason}"
                )
                self.get_logger().warning(
                    "Rejected unsafe path-start anchor: "
                    f"{anchor_check.reason}"
                )
                return
        self.path_points = points
        self.path_receipt = self._now()
        self.path_source_stamp = stamp_to_seconds(message.header.stamp)

    def _on_route_status(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
        except (json.JSONDecodeError, TypeError):
            return
        if isinstance(payload, dict):
            self.route_status = payload
            self.route_status_receipt = self._now()

    def _route_matches_goal(self) -> bool:
        if self.goal is None or not self.route_status.get("ready", False):
            return False
        try:
            goal_x = float(self.route_status["goal_x"])
            goal_y = float(self.route_status["goal_y"])
            route_goal_yaw = float(self.route_status["goal_yaw"])
            route_goal_stamp = float(self.route_status["goal_stamp"])
        except (KeyError, TypeError, ValueError):
            return False
        position = self.goal.pose.position
        goal_stamp = stamp_to_seconds(self.goal.header.stamp)
        goal_yaw = quaternion_to_yaw(self.goal.pose.orientation)
        identity_tolerance = float(
            self.get_parameter("path_stamp_tolerance").value
        )
        if not goal_identity_matches(
            actual_x=goal_x,
            actual_y=goal_y,
            actual_yaw=route_goal_yaw,
            actual_stamp=route_goal_stamp,
            expected_x=float(position.x),
            expected_y=float(position.y),
            expected_yaw=goal_yaw,
            expected_stamp=goal_stamp,
            position_tolerance=float(
                self.get_parameter("goal_match_tolerance").value
            ),
            identity_tolerance=identity_tolerance,
        ):
            return False
        try:
            route_path_stamp = float(self.route_status["path_source_stamp"])
        except (KeyError, TypeError, ValueError):
            return False
        # The Path and route-status messages use separate DDS topics.  During
        # a normal replan the provider publishes the new Path before its next
        # status update, so demanding exact source-stamp equality creates a
        # false stop at every handoff and prevents the hardware readiness
        # countdown from ever completing.
        # Identity remains fail-closed through the provider's goal generation,
        # the status goal revision above, endpoint/yaw checks in _on_path,
        # independent freshness checks, and the provider's explicit empty-Path
        # invalidation on every failure.  Both stamps must still be valid, and
        # a status claiming a path newer than the locally received Path is
        # rejected until that newer Path callback arrives.
        stamp_tolerance = float(
            self.get_parameter("path_stamp_tolerance").value
        )
        return bool(
            self.path_source_stamp is not None
            and isfinite(float(self.path_source_stamp))
            and float(self.path_source_stamp) > 0.0
            and isfinite(route_path_stamp)
            and route_path_stamp > 0.0
            and route_path_stamp
            <= float(self.path_source_stamp) + stamp_tolerance
        )

    def _goal_status(self) -> dict:
        if self.goal is None:
            return {
                "navigation_goal_received": False,
                "navigation_goal_x": None,
                "navigation_goal_y": None,
                "navigation_goal_yaw": None,
                "navigation_goal_stamp": None,
            }
        position = self.goal.pose.position
        return {
            "navigation_goal_received": True,
            "navigation_goal_x": float(position.x),
            "navigation_goal_y": float(position.y),
            "navigation_goal_yaw": quaternion_to_yaw(
                self.goal.pose.orientation
            ),
            "navigation_goal_stamp": stamp_to_seconds(self.goal.header.stamp),
        }

    def _publish_stop(self, reason: str, details: Optional[dict] = None) -> None:
        if self.reference_active:
            # Do not leave a previously accepted MPC path visible or usable
            # after the current planning cycle has failed closed.  This also
            # invalidates collision-monitor trajectory evidence immediately.
            empty_reference = Path()
            empty_reference.header.stamp = self.get_clock().now().to_msg()
            empty_reference.header.frame_id = self.config.grid.frame_id
            self.path_publisher.publish(empty_reference)
            self.reference_active = False
        control = TwistStamped()
        control.header.stamp = self.get_clock().now().to_msg()
        control.header.frame_id = "base_link"
        self.control_publisher.publish(control)
        payload = {
            "stamp": self._now(),
            "ready": False,
            "reason": reason,
            "preset": self.preset.name,
            "control_stack": (
                "dream" if self.preset.name != "pure_mpc" else "pure_mpc"
            ),
            "navigation_mode": "free_space",
            "mission_complete": self.goal_complete,
            "mpc_fallback": False,
            "maximum_cbf_slack": 0.0,
            "maximum_allowed_cbf_slack": float(
                self.get_parameter("maximum_allowed_cbf_slack").value
            ),
            "map_bounds_enforced": self.mpc.enforce_map_bounds,
            "verified_start_clearance_active": (
                self.verified_start_clearance_available
            ),
            "verified_start_clearance_radius": (
                self.verified_start_clearance_radius
                if self.verified_start_clearance_enabled
                else None
            ),
            **self._goal_status(),
        }
        if details:
            payload.update(details)
        self.status_publisher.publish(
            String(data=json.dumps(payload, separators=(",", ":")))
        )

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
        self.reference_active = True

    @staticmethod
    def _angle_error(first: float, second: float) -> float:
        return (float(first) - float(second) + pi) % (2.0 * pi) - pi

    def _source_is_fresh(self, source_stamp: Optional[float], now: float) -> bool:
        if source_stamp is None or not isfinite(float(source_stamp)):
            return False
        age = now - float(source_stamp)
        return bool(
            age
            >= -float(self.get_parameter("source_future_tolerance").value)
            and age < float(self.get_parameter("input_timeout").value)
        )

    def _inputs_ready(self, now: float) -> tuple[bool, str]:
        if self.goal is None:
            return False, "WAITING_FOR_NAVIGATION_GOAL"
        timeout = float(self.get_parameter("input_timeout").value)
        for name, receipt in (
            ("EGO", self.ego_receipt),
            ("WORLD", self.world_receipt),
            ("RISK", self.risk_receipt),
            ("DRIFT_READY", self.ready_receipt),
        ):
            if receipt is None or now - receipt >= timeout:
                return False, f"STALE_{name}"
        for name, source_stamp in (
            ("EGO_SOURCE", self.ego_source_stamp),
            ("WORLD_SOURCE", self.world_source_stamp),
            ("RISK_SOURCE", self.risk_source_stamp),
        ):
            if not self._source_is_fresh(source_stamp, now):
                return False, f"STALE_{name}"
        if not self.drift_ready:
            return False, "DRIFT_NOT_READY"
        costmap_timeout = float(self.get_parameter("costmap_timeout").value)
        if (
            self.costmap is None
            or self.costmap_receipt is None
            or now - self.costmap_receipt >= costmap_timeout
        ):
            return False, "STALE_COSTMAP"
        if self.costmap_source_stamp is None:
            return False, "STALE_COSTMAP_SOURCE"
        costmap_source_age = now - self.costmap_source_stamp
        future_tolerance = float(
            self.get_parameter("source_future_tolerance").value
        )
        if not -future_tolerance <= costmap_source_age < costmap_timeout:
            return False, "STALE_COSTMAP_SOURCE"
        path_timeout = float(self.get_parameter("path_timeout").value)
        if self.path_points is None or self.path_receipt is None:
            return False, (
                self.path_rejection_reason or "WAITING_FOR_GEOMETRIC_PATH"
            )
        if now - self.path_receipt >= path_timeout:
            return False, "STALE_GEOMETRIC_PATH"
        route_timeout = float(self.get_parameter("route_status_timeout").value)
        if self.route_status_receipt is None or now - self.route_status_receipt >= route_timeout:
            return False, "STALE_ROUTE_STATUS"
        if not self._route_matches_goal():
            return False, str(self.route_status.get("reason", "ROUTE_NOT_READY"))
        return True, "INPUTS_READY"

    def _plan(self) -> None:
        now = self._now()
        if self.goal_complete:
            self.mpc.reset()
            self._publish_stop("MISSION_COMPLETE")
            return
        ready, reason = self._inputs_ready(now)
        if not ready:
            self.mpc.reset()
            self._publish_stop(reason)
            return
        assert self.ego is not None
        assert self.goal is not None
        assert self.path_points is not None
        goal = self.goal.pose.position
        remaining = hypot(self.ego.x - goal.x, self.ego.y - goal.y)
        assert self.goal_yaw is not None
        yaw_error = abs(self._angle_error(self.ego.yaw, self.goal_yaw))
        if (
            remaining
            <= float(self.get_parameter("goal_position_tolerance").value)
            and self.ego.speed
            <= float(self.get_parameter("goal_stop_speed_tolerance").value)
            and yaw_error
            <= float(self.get_parameter("goal_yaw_tolerance").value)
        ):
            self.goal_complete = True
            self.mpc.reset()
            self._publish_stop(
                "MISSION_COMPLETE", {"navigation_goal_remaining": remaining}
            )
            return

        ego = EgoState(
            x=self.ego.x,
            y=self.ego.y,
            yaw=self.ego.yaw,
            speed=self.ego.speed,
            yaw_rate=self.ego.yaw_rate,
            stamp=self.ego.stamp,
            lane_index=0,
        )
        try:
            decision = evaluate_route_maneuver_risk(
                self.path_points,
                ego_xy=(ego.x, ego.y),
                ego_yaw=ego.yaw,
                risk_at=self.field.risk_at,
                preset=self.preset,
                lookahead=float(self.get_parameter("risk_lookahead").value),
                samples=int(self.get_parameter("risk_samples").value),
                maximum_cross_track_error=(
                    self.config.mpc.maximum_path_cross_track_error
                ),
            )
            if decision.vetoed:
                # A lane-keep fallback is well-defined upstream because a
                # surveyed lane corridor exists.  In arbitrary free space no
                # substitute geometry is certified, so the veto is a stop.
                self.mpc.reset()
                self._publish_stop(
                    "DECISION_RISK_VETO",
                    {
                        "maneuver": "STOP_FOR_RISK_VETO",
                        "route_maneuver": decision.maneuver,
                        "vetoed": True,
                        "decision_risk": decision.score,
                        "risk_maximum": decision.maximum,
                        "risk_mean": decision.mean,
                        "risk_at_ego": self.field.risk_at(ego.x, ego.y),
                        "navigation_goal_remaining": remaining,
                        "navigation_goal_yaw_error": yaw_error,
                    },
                )
                return
            result = self.mpc.solve_reference(
                ego,
                self.path_points,
                self.vehicles,
                self.field,
                self.preset,
                terminal_yaw=self.goal_yaw,
            )
        except (PathValidationError, ValueError, RuntimeError) as exc:
            self.get_logger().error(f"Free planner failed closed: {exc}")
            self.mpc.reset()
            self._publish_stop(f"PLANNER_ERROR:{exc}")
            return

        assert self.costmap is not None
        start_center, start_radius = self._start_clearance_contract()
        trajectory_check = validate_swept_trajectory(
            result.states,
            self.costmap,
            expected_frame=self.config.grid.frame_id,
            robot_length=self.config.mpc.robot_length,
            robot_width=self.config.mpc.robot_width,
            footprint_padding=self.config.mpc.navigation_footprint_padding,
            inflation_radius=self.config.mpc.navigation_inflation_radius,
            interpolation_spacing=0.5 * self.costmap.resolution,
            allow_initial_inflated_center_prefix=True,
            verified_start_clearance_center=start_center,
            verified_start_clearance_radius=start_radius,
        )
        if not trajectory_check.safe:
            self.mpc.reset()
            self._publish_stop(
                trajectory_check.reason,
                {
                    "trajectory_costmap_sample": trajectory_check.sample_index,
                    "trajectory_costmap_cell_x": trajectory_check.cell_x,
                    "trajectory_costmap_cell_y": trajectory_check.cell_y,
                    "trajectory_costmap_value": trajectory_check.cell_value,
                },
            )
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
            self.mpc.reset()
            self._publish_stop(f"PLANNER_SAFETY_CONFIG_ERROR:{exc}")
            return
        if not gated.valid:
            self.mpc.reset()
            self._publish_stop(
                gated.reason,
                {
                    "mpc_status": result.status,
                    "mpc_fallback": result.used_fallback,
                    "maximum_cbf_slack": (
                        result.maximum_slack
                        if isfinite(result.maximum_slack)
                        else None
                    ),
                },
            )
            return

        control = TwistStamped()
        control.header.stamp = self.get_clock().now().to_msg()
        control.header.frame_id = "base_link"
        control.twist.linear.x = gated.target_speed
        control.twist.linear.y = gated.acceleration
        control.twist.angular.z = gated.steering
        self.control_publisher.publish(control)
        self._publish_trajectory(result.states)
        self.status_publisher.publish(
            String(
                data=json.dumps(
                    {
                        "stamp": now,
                        "ready": True,
                        "reason": "ok",
                        "preset": self.preset.name,
                        "control_stack": (
                            "dream"
                            if self.preset.name != "pure_mpc"
                            else "pure_mpc"
                        ),
                        "navigation_mode": "free_space",
                        "mission_complete": False,
                        "maneuver": (
                            "HOLD_HEADING"
                            if decision.vetoed
                            else "FOLLOW_ROUTE"
                        ),
                        "route_maneuver": decision.maneuver,
                        "vetoed": decision.vetoed,
                        "decision_risk": decision.score,
                        "risk_maximum": decision.maximum,
                        "risk_mean": decision.mean,
                        "risk_at_ego": self.field.risk_at(ego.x, ego.y),
                        "navigation_goal_remaining": remaining,
                        "navigation_goal_yaw_error": yaw_error,
                        "configured_target_speed": self.config.mpc.target_speed,
                        "target_speed": result.command.target_speed,
                        "acceleration": result.command.acceleration,
                        "center_steer": result.command.steering,
                        "t_mpc": result.solve_seconds,
                        "mpc_status": result.status,
                        "mpc_fallback": result.used_fallback,
                        "maximum_cbf_slack": result.maximum_slack,
                        "maximum_allowed_cbf_slack": maximum_allowed_slack,
                        "map_bounds_enforced": self.mpc.enforce_map_bounds,
                        "verified_start_clearance_active": (
                            self.verified_start_clearance_available
                        ),
                        "verified_start_clearance_radius": (
                            self.verified_start_clearance_radius
                            if self.verified_start_clearance_enabled
                            else None
                        ),
                        **self._goal_status(),
                    },
                    separators=(",", ":"),
                )
            )
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DreamFreePlannerNode()
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
