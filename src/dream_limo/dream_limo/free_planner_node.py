"""DREAM risk gating and MPC tracking for an arbitrary Nav2 geometric path."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from hashlib import sha256
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
from .limo_scale import (
    IntegrationPreset,
    deployment_config_for_arena,
    get_preset,
)
from .ros_utils import (
    ControlSourceStamp,
    ego_from_odometry,
    quaternion_to_yaw,
    stamp_to_seconds,
    vehicle_from_mapping,
    yaw_to_quaternion,
)


class _ZeroRiskField:
    """Duck-typed risk source for arms that do not use the DRIFT PDE."""

    @staticmethod
    def risk_at(_x: float, _y: float) -> float:
        return 0.0

    @staticmethod
    def cbf_scale(
        _x: float, _y: float, _preset: IntegrationPreset
    ) -> float:
        """Keep the shared base CBF ellipse axes fixed in non-DREAM arms."""

        return 1.0

    @staticmethod
    def headway_scale(
        _x: float, _y: float, _preset: IntegrationPreset
    ) -> float:
        """Keep the shared base headway fixed in non-DREAM arms."""

        return 1.0


def shared_controller_parameter_fingerprint(mpc_config) -> str:
    """Hash only controller parameters, excluding every arm's risk channel."""

    payload = json.dumps(
        asdict(mpc_config), sort_keys=True, separators=(",", ":")
    )
    return sha256(payload.encode("utf-8")).hexdigest()


class DreamFreePlannerNode(Node):
    """Keep Nav2 geometric planning separate from DREAM's sole controller."""

    def __init__(self) -> None:
        super().__init__("dream_free_planner")
        self.declare_parameter("arena_file", "")
        self.declare_parameter("preset", "balanced")
        self.declare_parameter("planner_mode", "")
        self.declare_parameter("target_speed", 0.15)
        self.declare_parameter("ego_topic", "/dream/ego_state")
        self.declare_parameter("world_topic", "/dream/world_model")
        self.declare_parameter("risk_topic", "/dream/risk_field_raw")
        self.declare_parameter("drift_ready_topic", "/dream/drift_ready")
        self.declare_parameter("oacp_status_topic", "/dream/oacp_vb_status")
        self.declare_parameter(
            "hardware_gate_status_topic", "/dream/hardware_gate_status"
        )
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
        self.declare_parameter("oacp_velocity_slack_weight", 1.0e4)
        self.declare_parameter("oacp_maximum_future_velocity_slack", 0.01)
        self.declare_parameter("oacp_enable_contingency", True)
        self.declare_parameter("oacp_calibration_logging_only", False)
        self.declare_parameter("oacp_shared_prefix_steps", 2)
        self.declare_parameter("oacp_contingency_check_rate", 1.0)
        # Numerical zero used only to decide whether the contingency
        # alternative exists.  The executed branch has its separate future
        # violation limit below; all nonzero values remain logged as slack.
        self.declare_parameter("oacp_contingency_slack_tolerance", 1.0e-4)
        self.declare_parameter("oacp_status_timeout", 0.50)
        self.declare_parameter("oacp_gate_status_timeout", 0.30)
        self.declare_parameter(
            "oacp_prefix_position_tracking_tolerance", 0.01
        )
        self.declare_parameter(
            "oacp_prefix_speed_tracking_tolerance", 0.03
        )
        self.declare_parameter(
            "oacp_prefix_yaw_tracking_tolerance", 0.05
        )
        self.declare_parameter(
            "oacp_prefix_advance_minimum_progress", 0.95
        )
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
        configured_mode = str(self.get_parameter("planner_mode").value).strip()
        if not configured_mode:
            configured_mode = str(self.get_parameter("preset").value).strip()
        if configured_mode not in {
            "balanced",
            "pure_mpc",
            "nominal",
            "oacp_vb",
        }:
            raise ValueError(
                "free navigation supports balanced, nominal/pure_mpc, or oacp_vb"
            )
        self.planner_mode = configured_mode
        self.oacp_mode = configured_mode == "oacp_vb"
        self.nominal_mode = configured_mode == "nominal"
        self.preset = get_preset(
            "balanced" if configured_mode == "balanced" else "pure_mpc"
        )
        self.field = (
            _ZeroRiskField()
            if self.oacp_mode or self.nominal_mode
            else DREAMRiskField(self.config)
        )
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
        self.oacp_status: dict = {}
        self.oacp_status_receipt: Optional[float] = None
        self.pending_oacp_status: dict = {}
        self.pending_oacp_status_receipt: Optional[float] = None
        self.oacp_contingency_last_check_stamp: Optional[float] = None
        self.oacp_contingency_cached_valid: Optional[bool] = None
        self.oacp_contingency_cached_context: Optional[tuple] = None
        self.oacp_contingency_cached_prefix: Optional[np.ndarray] = None
        self.oacp_contingency_cached_states: Optional[np.ndarray] = None
        self.oacp_contingency_cached_prefix_cursor = 0
        self.oacp_prefix_pending_control_stamp: Optional[
            ControlSourceStamp
        ] = None
        self.oacp_prefix_pending_cursor: Optional[int] = None
        self.hardware_gate_status: dict = {}
        self.hardware_gate_status_receipt: Optional[float] = None
        self.goal: Optional[PoseStamped] = None
        self.goal_receipt: Optional[float] = None
        self.goal_yaw: Optional[float] = None
        self.path_points: Optional[np.ndarray] = None
        self.path_receipt: Optional[float] = None
        self.path_source_stamp: Optional[float] = None
        self.pending_path_points: Optional[np.ndarray] = None
        self.pending_path_receipt: Optional[float] = None
        self.pending_path_source_stamp: Optional[float] = None
        self.path_rejection_reason: Optional[str] = None
        self.path_rejection_details: dict = {}
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
        if self.oacp_mode:
            self.create_subscription(
                String,
                str(self.get_parameter("oacp_status_topic").value),
                self._on_oacp_status,
                reliable,
            )
            self.create_subscription(
                String,
                str(
                    self.get_parameter(
                        "hardware_gate_status_topic"
                    ).value
                ),
                self._on_hardware_gate_status,
                reliable,
            )
        elif not self.nominal_mode:
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
        velocity_slack_weight = float(
            self.get_parameter("oacp_velocity_slack_weight").value
        )
        maximum_future_velocity_slack = float(
            self.get_parameter("oacp_maximum_future_velocity_slack").value
        )
        shared_prefix_steps = int(
            self.get_parameter("oacp_shared_prefix_steps").value
        )
        contingency_slack_tolerance = float(
            self.get_parameter(
                "oacp_contingency_slack_tolerance"
            ).value
        )
        contingency_check_rate = float(
            self.get_parameter("oacp_contingency_check_rate").value
        )
        oacp_status_timeout = float(
            self.get_parameter("oacp_status_timeout").value
        )
        gate_status_timeout = float(
            self.get_parameter("oacp_gate_status_timeout").value
        )
        prefix_tracking_tolerances = (
            float(
                self.get_parameter(
                    "oacp_prefix_position_tracking_tolerance"
                ).value
            ),
            float(
                self.get_parameter(
                    "oacp_prefix_speed_tracking_tolerance"
                ).value
            ),
            float(
                self.get_parameter(
                    "oacp_prefix_yaw_tracking_tolerance"
                ).value
            ),
        )
        prefix_advance_minimum_progress = float(
            self.get_parameter(
                "oacp_prefix_advance_minimum_progress"
            ).value
        )
        if self.oacp_mode and (
            not isfinite(velocity_slack_weight)
            or velocity_slack_weight <= 0.0
            or not isfinite(maximum_future_velocity_slack)
            or maximum_future_velocity_slack < 0.0
            or not 1 <= shared_prefix_steps < self.config.mpc.horizon
            or not isfinite(contingency_slack_tolerance)
            or contingency_slack_tolerance < 0.0
            or not isfinite(contingency_check_rate)
            or contingency_check_rate <= 0.0
            or contingency_check_rate > update_rate
            or not isfinite(oacp_status_timeout)
            or oacp_status_timeout <= 0.0
            or not isfinite(gate_status_timeout)
            or gate_status_timeout <= 0.0
            or any(
                not isfinite(value) or value <= 0.0
                for value in prefix_tracking_tolerances
            )
            or not isfinite(prefix_advance_minimum_progress)
            or not 0.0 < prefix_advance_minimum_progress <= 1.0
        ):
            raise ValueError("invalid OACP-VB planner integration parameters")
        self.controller_parameter_hash = (
            shared_controller_parameter_fingerprint(self.config.mpc)
        )
        self.get_logger().info(
            json.dumps(
                {
                    "event": "shared_controller_parameter_record",
                    "planner_mode": self.planner_mode,
                    "shared_controller_parameter_hash": (
                        self.controller_parameter_hash
                    ),
                    "shared_controller_parameters": asdict(self.config.mpc),
                    "permitted_arm_difference": (
                        "occlusion_risk_assessment_and_evaluation_channel_only"
                    ),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
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

    def _clear_geometric_paths(self) -> None:
        self.path_points = None
        self.path_receipt = None
        self.path_source_stamp = None
        self.pending_path_points = None
        self.pending_path_receipt = None
        self.pending_path_source_stamp = None
        if self.oacp_mode:
            self.oacp_status = {}
            self.oacp_status_receipt = None
            self.pending_oacp_status = {}
            self.pending_oacp_status_receipt = None
            self._clear_oacp_contingency_certificate()

    def _clear_oacp_contingency_certificate(
        self, *, validity: Optional[bool] = None
    ) -> None:
        """Discard every artifact tied to one verified branch pair."""

        self.oacp_contingency_cached_valid = validity
        self.oacp_contingency_cached_context = None
        self.oacp_contingency_cached_prefix = None
        self.oacp_contingency_cached_states = None
        self.oacp_contingency_cached_prefix_cursor = 0
        self.oacp_prefix_pending_control_stamp = None
        self.oacp_prefix_pending_cursor = None

    def _try_activate_oacp_pair(self) -> bool:
        """Atomically activate one path and its exact matching OACP bound."""

        if (
            not self.oacp_mode
            or self.pending_path_points is None
            or self.pending_path_receipt is None
            or self.pending_path_source_stamp is None
            or not self.pending_oacp_status
            or self.pending_oacp_status_receipt is None
            or self.pending_oacp_status.get("ready") is not True
            or self.pending_oacp_status.get("exact_bound_valid") is not True
        ):
            return False
        try:
            assessment_stamp = float(
                self.pending_oacp_status["path_source_stamp"]
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            return False
        if (
            not isfinite(assessment_stamp)
            or assessment_stamp <= 0.0
            or abs(assessment_stamp - self.pending_path_source_stamp)
            > float(self.get_parameter("path_stamp_tolerance").value)
        ):
            return False
        self.path_points = self.pending_path_points
        self.path_receipt = self.pending_path_receipt
        self.path_source_stamp = self.pending_path_source_stamp
        self.oacp_status = self.pending_oacp_status
        self.oacp_status_receipt = self.pending_oacp_status_receipt
        self.pending_path_points = None
        self.pending_path_receipt = None
        self.pending_path_source_stamp = None
        self.pending_oacp_status = {}
        self.pending_oacp_status_receipt = None
        # A contingency result for the previous geometry cannot certify this
        # newly activated path/bound pair.
        self.oacp_contingency_last_check_stamp = None
        self._clear_oacp_contingency_certificate()
        return True

    def _on_oacp_status(self, message: String) -> None:
        now = self._now()
        try:
            payload = json.loads(message.data)
        except (json.JSONDecodeError, TypeError):
            payload = None
        if not isinstance(payload, dict) or payload.get("provider") != "oacp_vb":
            self.oacp_status = {}
            self.oacp_status_receipt = None
            self.pending_oacp_status = {}
            self.pending_oacp_status_receipt = None
            return
        if (
            payload.get("ready") is not True
            or payload.get("exact_bound_valid") is not True
        ):
            # Do not let a pre-goal or invalid assessment replace an active
            # exact pair.  Its existing receipt will age out if no valid
            # replacement arrives.
            if self.path_points is None:
                self.oacp_status = payload
                self.oacp_status_receipt = now
            return
        try:
            assessment_stamp = float(payload["path_source_stamp"])
        except (KeyError, TypeError, ValueError, OverflowError):
            return
        tolerance = float(
            self.get_parameter("path_stamp_tolerance").value
        )
        if (
            self.path_source_stamp is not None
            and isfinite(assessment_stamp)
            and abs(assessment_stamp - self.path_source_stamp) <= tolerance
        ):
            self.oacp_status = payload
            self.oacp_status_receipt = now
            return
        self.pending_oacp_status = payload
        self.pending_oacp_status_receipt = now
        self._try_activate_oacp_pair()

    def _on_hardware_gate_status(self, message: String) -> None:
        """Retain the final gate's acknowledgement of physical forwarding."""

        try:
            payload = json.loads(message.data)
        except (json.JSONDecodeError, TypeError):
            payload = None
        if not isinstance(payload, dict):
            self.hardware_gate_status = {}
            self.hardware_gate_status_receipt = None
            return
        self.hardware_gate_status = payload
        self.hardware_gate_status_receipt = self._now()

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
        self._clear_geometric_paths()
        self.path_rejection_reason = None
        self.path_rejection_details = {}
        self.route_status = {}
        self.route_status_receipt = None
        self.last_goal_key = None
        self.goal_complete = False
        self.mpc.reset()
        self.oacp_contingency_last_check_stamp = None
        self._clear_oacp_contingency_certificate()
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
        # invalidates a route.  Valid OACP replans are staged until the exact
        # matching bound arrives; invalid/empty paths clear both active and
        # pending pairs immediately.
        self.path_rejection_reason = None
        self.path_rejection_details = {}
        if self.goal is None or message.header.frame_id != self.config.grid.frame_id:
            self._clear_geometric_paths()
            return
        if not message.poses:
            # An empty path is the provider's intentional fail-closed
            # invalidation signal, not malformed route data.
            self._clear_geometric_paths()
            return
        try:
            source_stamp = stamp_to_seconds(message.header.stamp)
            if not isfinite(source_stamp) or source_stamp <= 0.0:
                raise PathValidationError("path source stamp is invalid")
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
            self._clear_geometric_paths()
            self.path_rejection_reason = f"PATH_REJECTED:{exc}"
            self.get_logger().warning(f"Rejected geometric path: {exc}")
            return
        goal = self.goal.pose.position
        endpoint_error = hypot(points[-1, 0] - goal.x, points[-1, 1] - goal.y)
        if endpoint_error > float(self.get_parameter("path_goal_tolerance").value):
            self._clear_geometric_paths()
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
            self._clear_geometric_paths()
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
                self._clear_geometric_paths()
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
                allow_known_soft_center=True,
                verified_start_clearance_center=start_center,
                verified_start_clearance_radius=start_radius,
            )
            if not anchor_check.safe:
                self._clear_geometric_paths()
                self.path_rejection_reason = (
                    f"PATH_START_{anchor_check.reason}"
                )
                self.path_rejection_details = {
                    "path_start_costmap_sample": anchor_check.sample_index,
                    "path_start_costmap_cell_x": anchor_check.cell_x,
                    "path_start_costmap_cell_y": anchor_check.cell_y,
                    "path_start_costmap_value": anchor_check.cell_value,
                }
                self.get_logger().warning(
                    "Rejected unsafe path-start anchor: "
                    f"{anchor_check.reason} "
                    f"sample={anchor_check.sample_index} "
                    f"cell=({anchor_check.cell_x},{anchor_check.cell_y}) "
                    f"value={anchor_check.cell_value}"
                )
                return
        receipt = self._now()
        if self.oacp_mode:
            self.pending_path_points = points
            self.pending_path_receipt = receipt
            self.pending_path_source_stamp = source_stamp
            self._try_activate_oacp_pair()
        else:
            self.path_points = points
            self.path_receipt = receipt
            self.path_source_stamp = source_stamp

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
        received_path_stamps = [
            float(stamp)
            for stamp in (
                self.path_source_stamp,
                getattr(self, "pending_path_source_stamp", None),
            )
            if stamp is not None and isfinite(float(stamp))
        ]
        newest_received_path_stamp = (
            None if not received_path_stamps else max(received_path_stamps)
        )
        return bool(
            newest_received_path_stamp is not None
            and newest_received_path_stamp > 0.0
            and isfinite(route_path_stamp)
            and route_path_stamp > 0.0
            and route_path_stamp
            <= newest_received_path_stamp + stamp_tolerance
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

    def _arm_status(self) -> dict:
        if self.planner_mode == "balanced":
            arm_name = "dream"
            risk_channel = "drift_pde_veto_cost_cbf"
            risk_channel_settings = {
                "decision_veto": True,
                "mpc_risk_cost": True,
                "cbf_risk_expansion": True,
                "oacp_velocity_bound": False,
            }
        elif self.oacp_mode:
            arm_name = (
                "oacp_vb_calibration"
                if bool(
                    self.get_parameter(
                        "oacp_calibration_logging_only"
                    ).value
                )
                else "oacp_vb"
            )
            risk_channel = "phantom_reachability_velocity_bound"
            risk_channel_settings = {
                "decision_veto": False,
                "mpc_risk_cost": False,
                "cbf_risk_expansion": False,
                "oacp_velocity_bound": not bool(
                    self.get_parameter(
                        "oacp_calibration_logging_only"
                    ).value
                ),
            }
        else:
            arm_name = "nominal"
            risk_channel = "none"
            risk_channel_settings = {
                "decision_veto": False,
                "mpc_risk_cost": False,
                "cbf_risk_expansion": False,
                "oacp_velocity_bound": False,
            }
        return {
            "arm": arm_name,
            "planner_mode": self.planner_mode,
            "preset": self.preset.name,
            "control_stack": arm_name,
            "controller_stack": "shared_lmpc_cbf",
            "risk_channel": risk_channel,
            "risk_channel_settings": risk_channel_settings,
            "shared_controller_parameter_hash": self.controller_parameter_hash,
            "shared_controller_rate_hz": float(
                self.get_parameter("update_rate").value
            ),
            "shared_mpc_horizon_steps": self.config.mpc.horizon,
            "shared_mpc_dt": self.config.mpc.dt,
        }

    def _oacp_status_details(self) -> dict:
        if not self.oacp_mode:
            return {}
        allowed = (
            "assessment_ready",
            "pre_goal_bound_valid",
            "exact_bound_valid",
            "thresholds_calibrated",
            "path_source_stamp",
            "pvs_component_count",
            "pvs_start",
            "pvs_end",
            "pvs_length",
            "frs_intersects_trajectory",
            "risk_total",
            "raw_risk_maximum",
            "risk_reducer",
            "exploration_velocity_bound",
            "fallback_velocity_bound",
            "v_occ_min",
            "v_occ_max",
            "c_th_max_exploration",
            "c_th_max_fallback",
            "calibration_sample_count",
            "calibration_logging_only",
            "calibration_run_active",
            "calibration_goal_revision",
            "calibration_goal_receipt_stamp",
            "calibration_sample_scope",
            "suggested_c_th_max_exploration",
            "suggested_c_th_max_fallback",
            "geometry_assumption",
        )
        return {
            f"oacp_{key}": self.oacp_status.get(key)
            for key in allowed
            if key in self.oacp_status
        }

    def _publish_stop(self, reason: str, details: Optional[dict] = None) -> None:
        if self.oacp_mode:
            # A command that failed any planner-side gate was not executed.
            # Never carry its branch certificate across the stop.
            self.oacp_contingency_last_check_stamp = None
            self._clear_oacp_contingency_certificate(validity=False)
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
            **self._arm_status(),
            **self._oacp_status_details(),
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
        ):
            if receipt is None or now - receipt >= timeout:
                return False, f"STALE_{name}"
        for name, source_stamp in (
            ("EGO_SOURCE", self.ego_source_stamp),
            ("WORLD_SOURCE", self.world_source_stamp),
        ):
            if not self._source_is_fresh(source_stamp, now):
                return False, f"STALE_{name}"
        if self.oacp_mode:
            oacp_timeout = float(
                self.get_parameter("oacp_status_timeout").value
            )
            if (
                self.oacp_status_receipt is None
                or now - self.oacp_status_receipt >= oacp_timeout
            ):
                return False, "STALE_OACP_ASSESSMENT"
            if self.oacp_status.get("assessment_ready") is not True:
                return False, "OACP_ASSESSMENT_NOT_READY"
            if self.oacp_status.get("exact_bound_valid") is not True:
                return False, "OACP_EXACT_BOUND_NOT_READY"
            if self.oacp_status.get("ready") is not True:
                return False, str(
                    self.oacp_status.get("reason", "OACP_BOUND_NOT_READY")
                )
        elif not self.nominal_mode:
            for name, receipt in (
                ("RISK", self.risk_receipt),
                ("DRIFT_READY", self.ready_receipt),
            ):
                if receipt is None or now - receipt >= timeout:
                    return False, f"STALE_{name}"
            if not self._source_is_fresh(self.risk_source_stamp, now):
                return False, "STALE_RISK_SOURCE"
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
        if self.oacp_mode:
            try:
                oacp_path_stamp = float(
                    self.oacp_status["path_source_stamp"]
                )
            except (KeyError, TypeError, ValueError, OverflowError):
                return False, "OACP_PATH_IDENTITY_MISSING"
            assert self.path_source_stamp is not None
            if (
                not isfinite(oacp_path_stamp)
                or oacp_path_stamp <= 0.0
                or abs(oacp_path_stamp - self.path_source_stamp)
                > float(self.get_parameter("path_stamp_tolerance").value)
            ):
                return False, "OACP_PATH_IDENTITY_MISMATCH"
        return True, "INPUTS_READY"

    def _oacp_bound(self, key: str) -> float:
        try:
            value = float(self.oacp_status[key])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"missing or invalid OACP bound {key}") from exc
        if (
            not isfinite(value)
            or value < self.config.mpc.minimum_speed
            or value > self.config.mpc.maximum_speed
        ):
            raise ValueError(f"OACP bound {key} is outside shared speed limits")
        return value

    def _validated_oacp_bounds(self) -> tuple[float, float, float, float]:
        """Return bounds only when the complete provider relation is valid."""

        minimum_bound = self._oacp_bound("v_occ_min")
        maximum_bound = self._oacp_bound("v_occ_max")
        exploration_bound = self._oacp_bound(
            "exploration_velocity_bound"
        )
        fallback_bound = self._oacp_bound("fallback_velocity_bound")
        tolerance = 1.0e-9
        if (
            abs(maximum_bound - self.config.mpc.target_speed) > tolerance
            or minimum_bound > exploration_bound + tolerance
            or exploration_bound > fallback_bound + tolerance
            or fallback_bound > maximum_bound + tolerance
        ):
            raise ValueError(
                "OACP bounds violate v_min <= exploration <= fallback "
                "<= v_max == shared target speed"
            )
        return (
            minimum_bound,
            maximum_bound,
            exploration_bound,
            fallback_bound,
        )

    def _oacp_contingency_context(
        self,
        *,
        minimum_bound: float,
        maximum_bound: float,
        exploration_bound: float,
        fallback_bound: float,
    ) -> tuple:
        """Describe every material input covered by a fallback verification."""

        if self.path_source_stamp is None:
            raise ValueError("OACP contingency has no active path identity")
        vehicles = tuple(
            sorted(
                (
                    vehicle.vehicle_id,
                    float(vehicle.x),
                    float(vehicle.y),
                    float(vehicle.vx),
                    float(vehicle.vy),
                    float(vehicle.length),
                    float(vehicle.width),
                )
                for vehicle in self.vehicles
            )
        )
        return (
            float(self.path_source_stamp),
            vehicles,
            float(minimum_bound),
            float(maximum_bound),
            float(exploration_bound),
            float(fallback_bound),
        )

    @staticmethod
    def _oacp_cached_context_covers(
        cached: Optional[tuple],
        current: tuple,
    ) -> bool:
        """Allow reuse only when geometry is identical and the cap did not tighten."""

        if cached is None or len(cached) != 6 or len(current) != 6:
            return False
        return bool(
            cached[0] == current[0]
            and cached[1] == current[1]
            and cached[2] == current[2]
            and cached[3] == current[3]
            # A looser cap preserves the previously certified prefix.  Any
            # tightened executed or fallback cap requires a fresh pair.
            and float(current[4]) + 1.0e-9 >= float(cached[4])
            and float(current[5]) + 1.0e-9 >= float(cached[5])
        )

    def _oacp_prefix_state_matches(
        self, ego: EgoState, state_index: int
    ) -> bool:
        """Check that physical state still follows the certified common segment."""

        states = self.oacp_contingency_cached_states
        if (
            not isinstance(states, np.ndarray)
            or states.ndim != 2
            or states.shape[0] != 4
            or not 0 <= state_index < states.shape[1]
            or not np.all(np.isfinite(states))
        ):
            return False
        expected = states[:, state_index]
        position_error = hypot(
            ego.x - float(expected[0]),
            ego.y - float(expected[1]),
        )
        speed_error = abs(ego.speed - float(expected[2]))
        yaw_error = abs(self._angle_error(ego.yaw, float(expected[3])))
        return bool(
            position_error
            <= float(
                self.get_parameter(
                    "oacp_prefix_position_tracking_tolerance"
                ).value
            )
            and speed_error
            <= float(
                self.get_parameter(
                    "oacp_prefix_speed_tracking_tolerance"
                ).value
            )
            and yaw_error
            <= float(
                self.get_parameter(
                    "oacp_prefix_yaw_tracking_tolerance"
                ).value
            )
        )

    def _oacp_prefix_segment_progress(
        self, ego: EgoState, state_index: int
    ) -> Optional[float]:
        """Return bounded progress when ego lies in the certified state tube."""

        states = self.oacp_contingency_cached_states
        if (
            not isinstance(states, np.ndarray)
            or states.ndim != 2
            or states.shape[0] != 4
            or not 0 <= state_index < states.shape[1] - 1
            or not np.all(np.isfinite(states))
        ):
            return None
        start = states[:, state_index]
        end = states[:, state_index + 1]
        position_delta = end[0:2] - start[0:2]
        position_norm_squared = float(
            np.dot(position_delta, position_delta)
        )
        if position_norm_squared > 1.0e-12:
            raw_progress = float(
                np.dot(
                    np.asarray([ego.x, ego.y]) - start[0:2],
                    position_delta,
                )
                / position_norm_squared
            )
        else:
            speed_delta = float(end[2] - start[2])
            yaw_delta = self._angle_error(
                float(end[3]), float(start[3])
            )
            if abs(speed_delta) > 1.0e-9:
                raw_progress = float(
                    (ego.speed - start[2]) / speed_delta
                )
            elif abs(yaw_delta) > 1.0e-9:
                raw_progress = (
                    self._angle_error(ego.yaw, float(start[3]))
                    / yaw_delta
                )
            else:
                # A stationary certified step is complete once its exact
                # command is acknowledged; there is no state change to observe.
                raw_progress = 1.0
        progress = float(np.clip(raw_progress, 0.0, 1.0))
        speed_delta = float(end[2] - start[2])
        yaw_delta = self._angle_error(float(end[3]), float(start[3]))
        expected_position = (
            start[0:2] + progress * position_delta
        )
        expected_speed = float(start[2] + progress * speed_delta)
        expected_yaw = float(start[3] + progress * yaw_delta)
        position_error = hypot(
            ego.x - float(expected_position[0]),
            ego.y - float(expected_position[1]),
        )
        speed_error = abs(ego.speed - expected_speed)
        yaw_error = abs(self._angle_error(ego.yaw, expected_yaw))
        if (
            position_error
            > float(
                self.get_parameter(
                    "oacp_prefix_position_tracking_tolerance"
                ).value
            )
            or speed_error
            > float(
                self.get_parameter(
                    "oacp_prefix_speed_tracking_tolerance"
                ).value
            )
            or yaw_error
            > float(
                self.get_parameter(
                    "oacp_prefix_yaw_tracking_tolerance"
                ).value
            )
        ):
            return None
        return progress

    def _reconcile_oacp_prefix_execution(
        self, ego: EgoState, now: float
    ) -> str:
        """Advance a prefix cursor only after the final hardware gate forwarded it."""

        if self.oacp_contingency_cached_valid is not True:
            self.oacp_prefix_pending_control_stamp = None
            self.oacp_prefix_pending_cursor = None
            return "NO_VALID_CERTIFICATE"
        prefix = self.oacp_contingency_cached_prefix
        states = self.oacp_contingency_cached_states
        cursor = self.oacp_contingency_cached_prefix_cursor
        if (
            not isinstance(prefix, np.ndarray)
            or prefix.ndim != 2
            or prefix.shape[0] != 2
            or not np.all(np.isfinite(prefix))
            or not isinstance(states, np.ndarray)
            or states.shape != (4, prefix.shape[1] + 1)
            or not np.all(np.isfinite(states))
            or not 0 <= cursor < prefix.shape[1]
        ):
            self._clear_oacp_contingency_certificate(validity=False)
            return "CERTIFICATE_MALFORMED"
        pending_stamp = self.oacp_prefix_pending_control_stamp
        pending_cursor = self.oacp_prefix_pending_cursor
        if pending_stamp is None or pending_cursor is None:
            self._clear_oacp_contingency_certificate(validity=False)
            return "PREFIX_COMMAND_NOT_PUBLISHED"
        if pending_cursor != cursor:
            self._clear_oacp_contingency_certificate(validity=False)
            return "PREFIX_CURSOR_MISMATCH"

        gate_fresh = bool(
            self.hardware_gate_status_receipt is not None
            and now >= self.hardware_gate_status_receipt
            and now - self.hardware_gate_status_receipt
            < float(
                self.get_parameter("oacp_gate_status_timeout").value
            )
        )
        try:
            forwarded_stamp = ControlSourceStamp.from_mapping(
                self.hardware_gate_status["forwarded_control_source_stamp"]
            )
        except (KeyError, TypeError, ValueError):
            forwarded_stamp = None
        forwarded = bool(
            gate_fresh
            and self.hardware_gate_status.get("ready") is True
            and self.hardware_gate_status.get("hardware_output_enabled") is True
            and forwarded_stamp == pending_stamp
        )
        self.oacp_prefix_pending_control_stamp = None
        self.oacp_prefix_pending_cursor = None
        if not forwarded:
            self._clear_oacp_contingency_certificate(validity=False)
            return "PREFIX_EXECUTION_UNCONFIRMED_REVOKED"

        next_cursor = cursor + 1
        segment_progress = self._oacp_prefix_segment_progress(ego, cursor)
        if segment_progress is None:
            self._clear_oacp_contingency_certificate(validity=False)
            return "FORWARDED_EXECUTION_STATE_MISMATCH"
        if (
            segment_progress
            < float(
                self.get_parameter(
                    "oacp_prefix_advance_minimum_progress"
                ).value
            )
            or not self._oacp_prefix_state_matches(ego, next_cursor)
        ):
            # Reapplying this control for a new full dt from a partially
            # progressed state would extend it beyond the trajectory covered
            # by the cached fallback solve.  Revoke and let the caller execute
            # the fail-closed minimum-bound solve until the next scheduled
            # two-branch verification.
            self._clear_oacp_contingency_certificate(validity=False)
            return "FORWARDED_PARTIAL_PREFIX_REVOKED"
        self.oacp_contingency_cached_prefix_cursor = next_cursor
        return "FORWARDED_PREFIX_ADVANCED"

    def _solve_oacp_reference(
        self,
        ego: EgoState,
        terminal_yaw: float,
        *,
        now: Optional[float] = None,
    ):
        """Execute the tighter branch and periodically verify an alternative.

        Rebuilding two CVXPY problems every 200 ms missed the onboard 5 Hz
        deadline in the reviewed benchmark.  The executed bound is still
        solved every cycle; only the non-executed contingency check runs at
        its explicit reduced rate, as permitted by the baseline protocol.
        """

        assert self.path_points is not None
        cycle_stamp = self._now() if now is None else float(now)
        if not isfinite(cycle_stamp):
            raise ValueError("OACP solve stamp must be finite")
        (
            minimum_bound,
            maximum_bound,
            exploration_bound,
            fallback_bound,
        ) = self._validated_oacp_bounds()
        contingency_context = self._oacp_contingency_context(
            minimum_bound=minimum_bound,
            maximum_bound=maximum_bound,
            exploration_bound=exploration_bound,
            fallback_bound=fallback_bound,
        )
        cache_context_matches = self._oacp_cached_context_covers(
            self.oacp_contingency_cached_context,
            contingency_context,
        )
        if (
            self.oacp_contingency_cached_valid is True
            and not cache_context_matches
        ):
            # Never reuse a verified alternative after a new path, a changed
            # visible vehicle, or a tighter executed/fallback velocity bound.
            self._clear_oacp_contingency_certificate(validity=False)
        slack_weight = float(
            self.get_parameter("oacp_velocity_slack_weight").value
        )
        if bool(
            self.get_parameter("oacp_calibration_logging_only").value
        ):
            calibration = self.mpc.solve_reference(
                ego,
                self.path_points,
                self.vehicles,
                self.field,
                self.preset,
                terminal_yaw=terminal_yaw,
            )
            return calibration, {
                "oacp_calibration_logging_only": True,
                "oacp_bound_applied": False,
                "oacp_executed_velocity_bound": None,
                "oacp_computed_exploration_velocity_bound": (
                    exploration_bound
                ),
                "oacp_computed_fallback_velocity_bound": fallback_bound,
                "oacp_calibration_mpc_status": calibration.status,
                "oacp_calibration_solve_seconds": (
                    calibration.solve_seconds
                ),
                "oacp_contingency_enabled": False,
                "oacp_contingency_valid": None,
                "oacp_contingency_clamp_event": False,
            }
        try:
            risk_total = float(self.oacp_status["risk_total"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError("missing or invalid OACP risk_total") from exc
        if not isfinite(risk_total) or risk_total < 0.0:
            raise ValueError("OACP risk_total must be finite and nonnegative")
        if risk_total <= 1.0e-12:
            # Remark 2 (or a collapsed PVS at reveal) removes the phantom
            # hazard.  The shared executed MPC still enforces all visible-
            # vehicle CBF constraints, but no phantom contingency is needed.
            self._clear_oacp_contingency_certificate()
            exploration = self.mpc.solve_reference(
                ego,
                self.path_points,
                self.vehicles,
                self.field,
                self.preset,
                terminal_yaw=terminal_yaw,
                velocity_upper_bound=exploration_bound,
                velocity_slack_weight=slack_weight,
            )
            return exploration, {
                "oacp_velocity_slack_weight": slack_weight,
                "oacp_calibration_logging_only": False,
                "oacp_bound_applied": True,
                "oacp_executed_velocity_bound": exploration_bound,
                "oacp_exploration_mpc_status": exploration.status,
                "oacp_exploration_solve_seconds": exploration.solve_seconds,
                "oacp_exploration_cbf_slack": exploration.maximum_slack,
                "oacp_exploration_velocity_slack": (
                    exploration.maximum_velocity_slack
                ),
                "oacp_exploration_future_velocity_slack": (
                    exploration.maximum_future_velocity_slack
                ),
                "oacp_contingency_enabled": bool(
                    self.get_parameter("oacp_enable_contingency").value
                ),
                "oacp_contingency_applicable": False,
                "oacp_contingency_not_applicable_reason": (
                    "NO_ACTIVE_PHANTOM_RISK"
                ),
                "oacp_contingency_valid": None,
                "oacp_contingency_clamp_event": False,
            }
        contingency_enabled = bool(
            self.get_parameter("oacp_enable_contingency").value
        )
        prefix_execution_state = (
            self._reconcile_oacp_prefix_execution(ego, cycle_stamp)
            if contingency_enabled
            else "CONTINGENCY_DISABLED"
        )
        check_rate = float(
            self.get_parameter("oacp_contingency_check_rate").value
        )
        check_period = 1.0 / check_rate
        check_age = (
            None
            if self.oacp_contingency_last_check_stamp is None
            else max(
                0.0,
                cycle_stamp - self.oacp_contingency_last_check_stamp,
            )
        )
        check_due = bool(
            contingency_enabled
            and (
                self.oacp_contingency_last_check_stamp is None
                or check_age is None
                or check_age >= check_period
            )
        )
        cached_prefix = self.oacp_contingency_cached_prefix
        cached_prefix_cursor = self.oacp_contingency_cached_prefix_cursor
        cached_prefix_well_formed = bool(
            isinstance(cached_prefix, np.ndarray)
            and cached_prefix.ndim == 2
            and cached_prefix.shape[0] == 2
            and np.all(np.isfinite(cached_prefix))
            and 0 <= cached_prefix_cursor <= cached_prefix.shape[1]
        )
        if (
            contingency_enabled
            and self.oacp_contingency_cached_valid is True
            and not check_due
        ):
            if not cached_prefix_well_formed:
                self._clear_oacp_contingency_certificate(validity=False)
            elif cached_prefix_cursor >= cached_prefix.shape[1]:
                # The verified common segment has been consumed.  Hold the
                # fail-closed minimum cap until the scheduled reduced-rate
                # check instead of silently increasing solver load.
                self._clear_oacp_contingency_certificate(validity=False)

        # A previously invalid alternative stays fail-closed between its
        # reduced-rate rechecks.  Avoid spending an exploration solve whose
        # command cannot be executed while that cached result remains invalid.
        if (
            contingency_enabled
            and not check_due
            and self.oacp_contingency_cached_valid is not True
        ):
            clamped = self.mpc.solve_reference(
                ego,
                self.path_points,
                self.vehicles,
                self.field,
                self.preset,
                terminal_yaw=terminal_yaw,
                velocity_upper_bound=minimum_bound,
                velocity_slack_weight=slack_weight,
            )
            return clamped, {
                "oacp_velocity_slack_weight": slack_weight,
                "oacp_calibration_logging_only": False,
                "oacp_bound_applied": True,
                "oacp_executed_velocity_bound": minimum_bound,
                "oacp_contingency_enabled": True,
                "oacp_contingency_applicable": True,
                "oacp_contingency_check_rate_hz": check_rate,
                "oacp_contingency_check_performed": False,
                "oacp_contingency_check_age": check_age,
                "oacp_contingency_cached_valid": False,
                "oacp_contingency_cache_context_match": (
                    cache_context_matches
                ),
                "oacp_prefix_execution_state": prefix_execution_state,
                "oacp_contingency_valid": False,
                "oacp_contingency_clamp_event": True,
                "oacp_clamped_mpc_status": clamped.status,
                "oacp_clamped_solve_seconds": clamped.solve_seconds,
                "oacp_clamped_velocity_slack": (
                    clamped.maximum_velocity_slack
                ),
                "oacp_clamped_future_velocity_slack": (
                    clamped.maximum_future_velocity_slack
                ),
            }

        cached_execution_prefix = None
        cached_execution_cursor = None
        if (
            contingency_enabled
            and not check_due
            and self.oacp_contingency_cached_valid is True
        ):
            assert isinstance(self.oacp_contingency_cached_prefix, np.ndarray)
            cached_execution_cursor = (
                self.oacp_contingency_cached_prefix_cursor
            )
            cached_execution_prefix = self.oacp_contingency_cached_prefix[
                :, cached_execution_cursor:
            ]

        exploration = self.mpc.solve_reference(
            ego,
            self.path_points,
            self.vehicles,
            self.field,
            self.preset,
            terminal_yaw=terminal_yaw,
            velocity_upper_bound=exploration_bound,
            velocity_slack_weight=slack_weight,
            fixed_control_prefix=cached_execution_prefix,
            # On a contingency-check cycle the selected branch is not known
            # yet.  Keep steer-rate history tied to the last physical command
            # until the alternative has been verified.
            commit_solution=not check_due,
        )
        details = {
            "oacp_velocity_slack_weight": slack_weight,
            "oacp_calibration_logging_only": False,
            "oacp_bound_applied": True,
            "oacp_executed_velocity_bound": exploration_bound,
            "oacp_exploration_mpc_status": exploration.status,
            "oacp_exploration_solve_seconds": exploration.solve_seconds,
            "oacp_exploration_cbf_slack": exploration.maximum_slack,
            "oacp_exploration_velocity_slack": (
                exploration.maximum_velocity_slack
            ),
            "oacp_exploration_future_velocity_slack": (
                exploration.maximum_future_velocity_slack
            ),
            "oacp_contingency_enabled": bool(
                self.get_parameter("oacp_enable_contingency").value
            ),
            "oacp_contingency_applicable": True,
            "oacp_contingency_check_rate_hz": check_rate,
            "oacp_contingency_check_performed": False,
            "oacp_contingency_check_age": check_age,
            "oacp_contingency_cached_valid": (
                self.oacp_contingency_cached_valid
            ),
            "oacp_contingency_cache_context_match": (
                cache_context_matches
            ),
            "oacp_prefix_execution_state": prefix_execution_state,
            "oacp_contingency_valid": (
                self.oacp_contingency_cached_valid
            ),
            "oacp_contingency_clamp_event": False,
            "oacp_cached_prefix_enforced": (
                cached_execution_prefix is not None
            ),
            "oacp_cached_prefix_cursor": cached_execution_cursor,
        }
        if exploration.used_fallback and contingency_enabled:
            # A failed executed-branch solve cannot inherit the previous
            # fallback certificate.  Re-solve at the fail-closed minimum cap.
            self._clear_oacp_contingency_certificate(validity=False)
            clamped = self.mpc.solve_reference(
                ego,
                self.path_points,
                self.vehicles,
                self.field,
                self.preset,
                terminal_yaw=terminal_yaw,
                velocity_upper_bound=minimum_bound,
                velocity_slack_weight=slack_weight,
            )
            details.update(
                {
                    "oacp_contingency_cached_valid": False,
                    "oacp_contingency_valid": False,
                    "oacp_contingency_clamp_event": True,
                    "oacp_executed_velocity_bound": minimum_bound,
                    "oacp_clamped_mpc_status": clamped.status,
                    "oacp_clamped_solve_seconds": clamped.solve_seconds,
                    "oacp_clamped_velocity_slack": (
                        clamped.maximum_velocity_slack
                    ),
                    "oacp_clamped_future_velocity_slack": (
                        clamped.maximum_future_velocity_slack
                    ),
                }
            )
            return clamped, details
        if not contingency_enabled:
            return exploration, details
        if not check_due:
            assert cached_execution_cursor is not None
            assert self.oacp_contingency_cached_prefix is not None
            details["oacp_prefix_command_cursor"] = (
                cached_execution_cursor
            )
            details["oacp_cached_prefix_remaining_steps"] = max(
                0,
                self.oacp_contingency_cached_prefix.shape[1]
                - cached_execution_cursor,
            )
            return exploration, details

        prefix_steps = int(
            self.get_parameter("oacp_shared_prefix_steps").value
        )
        fallback = self.mpc.solve_reference(
            ego,
            self.path_points,
            self.vehicles,
            self.field,
            self.preset,
            terminal_yaw=terminal_yaw,
            velocity_upper_bound=fallback_bound,
            velocity_slack_weight=slack_weight,
            fixed_control_prefix=exploration.controls[:, :prefix_steps],
            commit_solution=False,
        )
        contingency_slack_tolerance = float(
            self.get_parameter(
                "oacp_contingency_slack_tolerance"
            ).value
        )
        fallback_valid = bool(
            not fallback.used_fallback
            and fallback.maximum_velocity_slack
            <= contingency_slack_tolerance
            and fallback.maximum_slack <= contingency_slack_tolerance
        )
        self.oacp_contingency_last_check_stamp = cycle_stamp
        self.oacp_contingency_cached_valid = fallback_valid
        self.oacp_contingency_cached_context = contingency_context
        if fallback_valid:
            self.oacp_contingency_cached_prefix = np.array(
                exploration.controls[:, :prefix_steps],
                dtype=np.float64,
                copy=True,
            )
            self.oacp_contingency_cached_states = np.array(
                exploration.states[:, : prefix_steps + 1],
                dtype=np.float64,
                copy=True,
            )
            # The first command is only pending here.  The cursor advances on
            # the next cycle after a fresh final-gate acknowledgement and a
            # bounded physical-state tracking check.
            self.oacp_contingency_cached_prefix_cursor = 0
            self.oacp_prefix_pending_control_stamp = None
            self.oacp_prefix_pending_cursor = None
        else:
            self.oacp_contingency_cached_prefix = None
            self.oacp_contingency_cached_states = None
            self.oacp_contingency_cached_prefix_cursor = 0
        details.update(
            {
                "oacp_contingency_check_performed": True,
                "oacp_contingency_check_age": 0.0,
                "oacp_contingency_cached_valid": fallback_valid,
                "oacp_shared_prefix_steps": prefix_steps,
                "oacp_shared_prefix_seconds": (
                    prefix_steps * self.config.mpc.dt
                ),
                "oacp_fallback_mpc_status": fallback.status,
                "oacp_fallback_solve_seconds": fallback.solve_seconds,
                "oacp_fallback_cbf_slack": fallback.maximum_slack,
                "oacp_fallback_velocity_slack": (
                    fallback.maximum_velocity_slack
                ),
                "oacp_fallback_future_velocity_slack": (
                    fallback.maximum_future_velocity_slack
                ),
                "oacp_contingency_slack_tolerance": (
                    contingency_slack_tolerance
                ),
                "oacp_contingency_valid": fallback_valid,
                "oacp_cached_prefix_enforced": False,
                "oacp_cached_prefix_cursor": (
                    self.oacp_contingency_cached_prefix_cursor
                    if fallback_valid
                    else None
                ),
                "oacp_prefix_command_cursor": (
                    0 if fallback_valid else None
                ),
            }
        )
        if fallback_valid:
            self.mpc.commit_result(exploration)
            return exploration, details

        # This is a third solve only on a contingency failure.  Executing the
        # earlier exploration solution under a newly declared lower bound
        # would be internally inconsistent, so solve the clamped branch now.
        clamped = self.mpc.solve_reference(
            ego,
            self.path_points,
            self.vehicles,
            self.field,
            self.preset,
            terminal_yaw=terminal_yaw,
            velocity_upper_bound=minimum_bound,
            velocity_slack_weight=slack_weight,
        )
        details.update(
            {
                "oacp_contingency_clamp_event": True,
                "oacp_executed_velocity_bound": minimum_bound,
                "oacp_clamped_mpc_status": clamped.status,
                "oacp_clamped_solve_seconds": clamped.solve_seconds,
                "oacp_clamped_velocity_slack": (
                    clamped.maximum_velocity_slack
                ),
                "oacp_clamped_future_velocity_slack": (
                    clamped.maximum_future_velocity_slack
                ),
            }
        )
        return clamped, details

    def _plan(self) -> None:
        now = self._now()
        if self.goal_complete:
            self.mpc.reset()
            self._publish_stop("MISSION_COMPLETE")
            return
        ready, reason = self._inputs_ready(now)
        if not ready:
            self.mpc.reset()
            self._publish_stop(reason, self.path_rejection_details)
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
            oacp_solver_details = {}
            if self.oacp_mode:
                result, oacp_solver_details = self._solve_oacp_reference(
                    ego, self.goal_yaw, now=now
                )
            else:
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

        if self.oacp_mode:
            future_slack_limit = float(
                self.get_parameter(
                    "oacp_maximum_future_velocity_slack"
                ).value
            )
            if result.maximum_future_velocity_slack > future_slack_limit:
                self.mpc.reset()
                self._publish_stop(
                    "OACP_VELOCITY_BOUND_VIOLATION",
                    {
                        **oacp_solver_details,
                        "maximum_velocity_slack": (
                            result.maximum_velocity_slack
                        ),
                        "maximum_future_velocity_slack": (
                            result.maximum_future_velocity_slack
                        ),
                        "maximum_allowed_future_velocity_slack": (
                            future_slack_limit
                        ),
                    },
                )
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
            allow_known_soft_center=True,
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
        if self.oacp_mode:
            prefix_cursor = oacp_solver_details.get(
                "oacp_prefix_command_cursor"
            )
            if (
                oacp_solver_details.get("oacp_contingency_valid") is True
                and isinstance(prefix_cursor, int)
                and not isinstance(prefix_cursor, bool)
                and prefix_cursor
                == self.oacp_contingency_cached_prefix_cursor
            ):
                self.oacp_prefix_pending_control_stamp = (
                    ControlSourceStamp.from_ros_stamp(control.header.stamp)
                )
                self.oacp_prefix_pending_cursor = prefix_cursor
            else:
                self.oacp_prefix_pending_control_stamp = None
                self.oacp_prefix_pending_cursor = None
        self.control_publisher.publish(control)
        self._publish_trajectory(result.states)
        total_mpc_seconds = result.solve_seconds
        if self.oacp_mode:
            total_mpc_seconds = sum(
                float(value)
                for key, value in oacp_solver_details.items()
                if key.endswith("_solve_seconds")
                and value is not None
                and isfinite(float(value))
            )
        self.status_publisher.publish(
            String(
                data=json.dumps(
                    {
                        "stamp": now,
                        "ready": True,
                        "reason": "ok",
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
                        "t_mpc_total": total_mpc_seconds,
                        "mpc_status": result.status,
                        "mpc_fallback": result.used_fallback,
                        "maximum_cbf_slack": result.maximum_slack,
                        "maximum_velocity_slack": (
                            result.maximum_velocity_slack
                        ),
                        "maximum_future_velocity_slack": (
                            result.maximum_future_velocity_slack
                        ),
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
                        **self._arm_status(),
                        **self._oacp_status_details(),
                        **oacp_solver_details,
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
