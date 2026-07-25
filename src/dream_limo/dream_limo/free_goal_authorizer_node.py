"""Replaceable, costmap-validated free-space goal authorization.

This node is intentionally separate from ``dream_goal_authorizer``.  It owns
the same held arm/stop/status interface, but it does not know about lanes or a
surveyed merge endpoint.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import asin, atan2, cos, hypot, isfinite, sin
from typing import Any, Mapping, Optional

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformException, TransformListener

from .core.free_goal import (
    CostmapSnapshot,
    FreeGoalAuthorization,
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
from .core.nav2_route import goal_identity_matches
from .limo_scale import default_deployment_config
from .ros_utils import stamp_to_seconds


def _normalize_frame(frame: str) -> str:
    value = str(frame).strip()
    return value[1:] if value.startswith("/") else value


def _quaternion_rpy(quaternion) -> tuple[float, float, float]:
    qx = float(quaternion.x)
    qy = float(quaternion.y)
    qz = float(quaternion.z)
    qw = float(quaternion.w)
    sin_roll_cos_pitch = 2.0 * (qw * qx + qy * qz)
    cos_roll_cos_pitch = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = atan2(sin_roll_cos_pitch, cos_roll_cos_pitch)
    sin_pitch = 2.0 * (qw * qy - qz * qx)
    pitch = asin(max(-1.0, min(1.0, sin_pitch)))
    sin_yaw_cos_pitch = 2.0 * (qw * qz + qx * qy)
    cos_yaw_cos_pitch = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = atan2(sin_yaw_cos_pitch, cos_yaw_cos_pitch)
    return roll, pitch, yaw


@dataclass(frozen=True)
class RiskAssessmentReadiness:
    """Pre-goal readiness of an optional arm-specific risk provider."""

    required: bool
    ready: bool
    reason: str
    provider: Optional[str]
    age: Optional[float]


def _risk_reason(required_provider: str, suffix: str) -> str:
    """Return stable, explicit reasons while keeping future providers generic."""

    prefix = "OACP" if required_provider == "oacp_vb" else "RISK"
    return f"{prefix}_{suffix}"


def evaluate_required_risk_assessment(
    required_provider: str,
    payload: Optional[Mapping[str, Any]],
    receipt_stamp: Optional[float],
    *,
    now: float,
    timeout: float,
    shared_minimum_speed: float,
    shared_target_speed: float,
) -> RiskAssessmentReadiness:
    """Validate the pre-goal half of the two-phase risk contract.

    With an empty ``required_provider`` this is deliberately a no-op, preserving
    the existing DREAM and pure-MPC goal path.  A configured provider must have
    a fresh status heartbeat that names that exact provider and explicitly
    asserts both assessment readiness and a valid conservative pre-goal bound.
    Truthy substitutes are rejected at this authorization boundary.
    """

    required = str(required_provider).strip()
    if not required:
        return RiskAssessmentReadiness(
            required=False,
            ready=True,
            reason="RISK_ASSESSMENT_NOT_REQUIRED",
            provider=None,
            age=None,
        )
    values = (
        now,
        timeout,
        shared_minimum_speed,
        shared_target_speed,
    )
    if (
        not all(isfinite(float(value)) for value in values)
        or timeout <= 0.0
        or shared_minimum_speed < 0.0
        or shared_target_speed < shared_minimum_speed
    ):
        return RiskAssessmentReadiness(
            required=True,
            ready=False,
            reason=_risk_reason(required, "ASSESSMENT_CONFIG_INVALID"),
            provider=None,
            age=None,
        )
    if not isinstance(payload, Mapping) or receipt_stamp is None:
        return RiskAssessmentReadiness(
            required=True,
            ready=False,
            reason=_risk_reason(required, "ASSESSMENT_UNAVAILABLE"),
            provider=None,
            age=None,
        )
    try:
        receipt = float(receipt_stamp)
    except (TypeError, ValueError, OverflowError):
        receipt = float("nan")
    age = float(now) - receipt
    if not isfinite(receipt) or age < 0.0 or age >= float(timeout):
        return RiskAssessmentReadiness(
            required=True,
            ready=False,
            reason=_risk_reason(required, "ASSESSMENT_STALE"),
            provider=(
                str(payload.get("provider"))
                if payload.get("provider") is not None
                else None
            ),
            age=age if isfinite(age) and age >= 0.0 else None,
        )
    provider_value = payload.get("provider")
    provider = None if provider_value is None else str(provider_value)
    if provider != required:
        return RiskAssessmentReadiness(
            required=True,
            ready=False,
            reason=_risk_reason(required, "ASSESSMENT_PROVIDER_MISMATCH"),
            provider=provider,
            age=age,
        )
    if payload.get("assessment_ready") is not True:
        return RiskAssessmentReadiness(
            required=True,
            ready=False,
            reason=_risk_reason(required, "ASSESSMENT_NOT_READY"),
            provider=provider,
            age=age,
        )
    if payload.get("pre_goal_bound_valid") is not True:
        return RiskAssessmentReadiness(
            required=True,
            ready=False,
            reason=_risk_reason(required, "PRE_GOAL_BOUND_INVALID"),
            provider=provider,
            age=age,
        )
    try:
        minimum_bound = float(payload["v_occ_min"])
        maximum_bound = float(payload["v_occ_max"])
        pre_goal_bound = float(payload["pre_goal_velocity_bound"])
    except (KeyError, TypeError, ValueError, OverflowError):
        minimum_bound = maximum_bound = pre_goal_bound = float("nan")
    if (
        not all(
            isfinite(value)
            for value in (
                minimum_bound,
                maximum_bound,
                pre_goal_bound,
            )
        )
        or minimum_bound < shared_minimum_speed
        or maximum_bound < minimum_bound
        or abs(maximum_bound - shared_target_speed) > 1.0e-9
        or not minimum_bound <= pre_goal_bound <= maximum_bound
    ):
        return RiskAssessmentReadiness(
            required=True,
            ready=False,
            reason=_risk_reason(required, "PRE_GOAL_BOUND_INVALID"),
            provider=provider,
            age=age,
        )
    return RiskAssessmentReadiness(
        required=True,
        ready=True,
        reason=_risk_reason(required, "ASSESSMENT_READY"),
        provider=provider,
        age=age,
    )


class DreamFreeGoalAuthorizerNode(Node):
    """Authorize arbitrary observed-free goals while continuously failing closed."""

    def __init__(self) -> None:
        super().__init__("dream_free_goal_authorizer")
        deployment = default_deployment_config()
        default_clearance = hypot(
            0.5 * deployment.mpc.robot_length,
            0.5 * deployment.mpc.robot_width,
        ) + deployment.safety.collision_inflation_margin

        self.declare_parameter("enabled", False)
        self.declare_parameter("goal_topic", "/goal_pose")
        self.declare_parameter("ego_topic", "/dream/ego_state")
        self.declare_parameter("costmap_topic", "/global_costmap/costmap")
        self.declare_parameter("accepted_goal_topic", "/dream/navigation_goal")
        self.declare_parameter("planner_status_topic", "/dream/planner_status")
        self.declare_parameter("preflight_status_topic", "/dream/preflight_status")
        self.declare_parameter("required_risk_provider", "")
        self.declare_parameter(
            "risk_assessment_status_topic", "/dream/oacp_vb_status"
        )
        self.declare_parameter("risk_assessment_timeout", 0.50)
        self.declare_parameter(
            "shared_minimum_speed", deployment.mpc.minimum_speed
        )
        self.declare_parameter(
            "shared_target_speed", deployment.mpc.target_speed
        )
        self.declare_parameter("arm_topic", "/dream/arm")
        self.declare_parameter("external_stop_topic", "/dream/external_stop")
        self.declare_parameter("status_topic", "/dream/deadman_status")
        self.declare_parameter("stop_service", "/dream/stop_mission")
        self.declare_parameter("goal_frame", "map")
        self.declare_parameter("odom_goal_frame", "odom")
        self.declare_parameter("footprint_clearance", default_clearance)
        self.declare_parameter("goal_timeout", 1.0)
        self.declare_parameter("ego_timeout", 0.50)
        self.declare_parameter("costmap_timeout", 0.75)
        self.declare_parameter("planner_timeout", 0.75)
        self.declare_parameter("preflight_timeout", 2.0)
        self.declare_parameter("future_tolerance", 0.10)
        self.declare_parameter("goal_match_tolerance", 1.0e-3)
        self.declare_parameter("tf_timeout", 0.10)
        self.declare_parameter("publish_rate", 20.0)

        self.config = FreeGoalConfig(
            frame_id=_normalize_frame(str(self.get_parameter("goal_frame").value)),
            footprint_clearance=float(
                self.get_parameter("footprint_clearance").value
            ),
            goal_timeout=float(self.get_parameter("goal_timeout").value),
            ego_timeout=float(self.get_parameter("ego_timeout").value),
            costmap_timeout=float(self.get_parameter("costmap_timeout").value),
            planner_timeout=float(self.get_parameter("planner_timeout").value),
            preflight_timeout=float(
                self.get_parameter("preflight_timeout").value
            ),
            future_tolerance=float(self.get_parameter("future_tolerance").value),
            goal_match_tolerance=float(
                self.get_parameter("goal_match_tolerance").value
            ),
        )
        self.enabled = bool(self.get_parameter("enabled").value)
        self.odom_goal_frame = _normalize_frame(
            str(self.get_parameter("odom_goal_frame").value)
        )
        self.tf_timeout = float(self.get_parameter("tf_timeout").value)
        self.required_risk_provider = str(
            self.get_parameter("required_risk_provider").value
        ).strip()
        self.risk_assessment_timeout = float(
            self.get_parameter("risk_assessment_timeout").value
        )
        self.shared_minimum_speed = float(
            self.get_parameter("shared_minimum_speed").value
        )
        self.shared_target_speed = float(
            self.get_parameter("shared_target_speed").value
        )
        publish_rate = float(self.get_parameter("publish_rate").value)
        if (
            not self.odom_goal_frame
            or not isfinite(self.tf_timeout)
            or self.tf_timeout <= 0.0
            or not isfinite(self.risk_assessment_timeout)
            or self.risk_assessment_timeout <= 0.0
            or not isfinite(self.shared_minimum_speed)
            or self.shared_minimum_speed < 0.0
            or not isfinite(self.shared_target_speed)
            or self.shared_target_speed < self.shared_minimum_speed
            or not isfinite(publish_rate)
            or publish_rate <= 0.0
        ):
            raise ValueError("free-goal node timing and odom frame must be valid")

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
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.state = FreeGoalMissionLatch()
        self.ego: Optional[FreeGoalEgoState] = None
        self.costmap: Optional[CostmapSnapshot] = None
        self.planner: Optional[FreeGoalPlannerReadiness] = None
        self.preflight: Optional[FreeGoalPreflightReadiness] = None
        self.risk_assessment_payload: Optional[dict] = None
        self.risk_assessment_receipt_stamp: Optional[float] = None
        self.accepted_goal_source_stamp: Optional[float] = None
        self.accepted_goal_receipt_stamp: Optional[float] = None
        self.accepted_goal_publication_stamp: Optional[float] = None
        self.candidate_goal_receipt_stamp: Optional[float] = None
        self.last_goal_source_frame: Optional[str] = None
        self.last_tf_error: Optional[str] = None

        self.goal_publisher = self.create_publisher(
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
            OccupancyGrid,
            str(self.get_parameter("costmap_topic").value),
            self._on_costmap,
            latched,
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
        self.create_subscription(
            String,
            str(self.get_parameter("risk_assessment_status_topic").value),
            self._on_risk_assessment_status,
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
        self.ego = FreeGoalEgoState(
            frame_id=_normalize_frame(message.header.frame_id),
            x=float(message.pose.pose.position.x),
            y=float(message.pose.pose.position.y),
            source_stamp=stamp_to_seconds(message.header.stamp),
            receipt_stamp=self._now(),
        )

    def _on_costmap(self, message: OccupancyGrid) -> None:
        now = self._now()
        try:
            origin_q = message.info.origin.orientation
            quaternion_norm = (
                float(origin_q.x) ** 2
                + float(origin_q.y) ** 2
                + float(origin_q.z) ** 2
                + float(origin_q.w) ** 2
            ) ** 0.5
            roll, pitch, yaw = _quaternion_rpy(message.info.origin.orientation)
            if (
                not isfinite(quaternion_norm)
                or abs(quaternion_norm - 1.0) > 1.0e-6
                or abs(roll) > 1.0e-6
                or abs(pitch) > 1.0e-6
            ):
                yaw = float("nan")
            self.costmap = CostmapSnapshot.from_sequence(
                frame_id=_normalize_frame(message.header.frame_id),
                width=int(message.info.width),
                height=int(message.info.height),
                resolution=float(message.info.resolution),
                origin_x=float(message.info.origin.position.x),
                origin_y=float(message.info.origin.position.y),
                origin_yaw=yaw,
                data=message.data,
                source_stamp=stamp_to_seconds(message.header.stamp),
                receipt_stamp=now,
            )
        except (TypeError, ValueError, OverflowError) as exc:
            self.costmap = None
            self.get_logger().warning(
                f"Rejected malformed navigation costmap: {exc}",
                throttle_duration_sec=2.0,
            )

    def _goal_request(self, message: PoseStamped, now: float) -> FreeGoalRequest:
        pose = message.pose
        frame = _normalize_frame(message.header.frame_id)
        request = FreeGoalRequest(
            frame_id=frame,
            x=float(pose.position.x),
            y=float(pose.position.y),
            z=float(pose.position.z),
            qx=float(pose.orientation.x),
            qy=float(pose.orientation.y),
            qz=float(pose.orientation.z),
            qw=float(pose.orientation.w),
            source_stamp=stamp_to_seconds(message.header.stamp),
            receipt_stamp=now,
        )
        if frame == self.config.frame_id:
            return request
        if frame != self.odom_goal_frame:
            raise ValueError(
                f"GOAL_FRAME_MISMATCH:{frame or '<empty>'}"
            )
        transform = self.tf_buffer.lookup_transform(
            self.config.frame_id,
            frame,
            Time.from_msg(message.header.stamp),
            timeout=Duration(seconds=self.tf_timeout),
        )
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        transform_quaternion_norm = (
            float(rotation.x) ** 2
            + float(rotation.y) ** 2
            + float(rotation.z) ** 2
            + float(rotation.w) ** 2
        ) ** 0.5
        if (
            not isfinite(transform_quaternion_norm)
            or abs(transform_quaternion_norm - 1.0) > 1.0e-3
        ):
            raise ValueError("GOAL_TF_INVALID_QUATERNION")
        roll, pitch, yaw = _quaternion_rpy(rotation)
        return transform_planar_goal(
            request,
            target_frame=self.config.frame_id,
            translation_x=float(translation.x),
            translation_y=float(translation.y),
            translation_z=float(translation.z),
            transform_yaw=yaw,
            transform_roll=roll,
            transform_pitch=pitch,
            maximum_transform_tilt=self.config.maximum_transform_tilt,
        )

    def _on_goal(self, message: PoseStamped) -> None:
        now = self._now()
        self.candidate_goal_receipt_stamp = now
        source_frame = _normalize_frame(message.header.frame_id)
        self.last_goal_source_frame = source_frame
        self.last_tf_error = None
        if not self.enabled:
            self.state.consider(FreeGoalValidation(False, "DISABLED"))
            self._publish_goal_invalidation()
            self._publish_heartbeat()
            return
        try:
            request = self._goal_request(message, now)
        except TransformException as exc:
            self.last_tf_error = str(exc)
            self.state.consider(FreeGoalValidation(False, "GOAL_TF_UNAVAILABLE"))
            self._publish_goal_invalidation()
            self.get_logger().warning(f"Goal transform unavailable: {exc}")
            self._publish_heartbeat()
            return
        except ValueError as exc:
            reason = str(exc).split(":", 1)[0]
            self.last_tf_error = str(exc)
            self.state.consider(FreeGoalValidation(False, reason))
            self._publish_goal_invalidation()
            self.get_logger().warning(f"Goal rejected: {exc}")
            self._publish_heartbeat()
            return

        validation = validate_free_goal_request(
            request,
            self.ego,
            self.costmap,
            now=now,
            config=self.config,
        )
        if not validation.accepted:
            self.state.consider(validation)
            self._publish_goal_invalidation()
            self.get_logger().warning(f"Goal rejected: {validation.reason}")
            self._publish_heartbeat()
            return
        risk_readiness = self._risk_assessment_readiness(now)
        if not risk_readiness.ready:
            self.state.consider(
                FreeGoalValidation(False, risk_readiness.reason)
            )
            self._publish_goal_invalidation()
            self.get_logger().warning(
                f"Goal rejected: {risk_readiness.reason}"
            )
            self._publish_heartbeat()
            return
        if not self.state.consider(validation):
            self._publish_goal_invalidation()
            self.get_logger().warning(f"Goal rejected: {validation.reason}")
            self._publish_heartbeat()
            return

        # A valid replacement is published immediately, but old planner
        # readiness can never keep the arm asserted for the new destination.
        self.planner = None
        self.accepted_goal_source_stamp = request.source_stamp
        self.accepted_goal_receipt_stamp = now
        goal = PoseStamped()
        goal.header.stamp = self.get_clock().now().to_msg()
        self.accepted_goal_publication_stamp = stamp_to_seconds(goal.header.stamp)
        goal.header.frame_id = self.config.frame_id
        goal.pose.position.x = float(validation.goal_x)
        goal.pose.position.y = float(validation.goal_y)
        goal.pose.orientation.z = sin(0.5 * float(validation.goal_yaw))
        goal.pose.orientation.w = cos(0.5 * float(validation.goal_yaw))
        self.goal_publisher.publish(goal)
        self.get_logger().info(
            "Accepted free-space navigation goal: "
            f"x={validation.goal_x:.3f}, y={validation.goal_y:.3f}, "
            f"revision={self.state.revision}"
        )
        self._publish_heartbeat()

    def _publish_goal_invalidation(self) -> None:
        """Invalidate the latched downstream route without inventing a pose."""
        self.accepted_goal_source_stamp = None
        self.accepted_goal_receipt_stamp = None
        self.accepted_goal_publication_stamp = None
        message = PoseStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        # Both downstream consumers reject this frame after first clearing
        # their previous goal/path state.
        message.header.frame_id = ""
        self.goal_publisher.publish(message)

    @staticmethod
    def _planner_goal(payload: dict) -> tuple[Optional[float], Optional[float]]:
        candidates = (
            ("navigation_goal_x", "navigation_goal_y"),
            ("mission_goal_x", "mission_goal_y"),
            ("goal_x", "goal_y"),
        )
        for x_key, y_key in candidates:
            if x_key not in payload or y_key not in payload:
                continue
            try:
                x = float(payload[x_key])
                y = float(payload[y_key])
            except (TypeError, ValueError, OverflowError):
                return None, None
            if isfinite(x) and isfinite(y):
                return x, y
            return None, None
        return None, None

    def _on_planner_status(self, message: String) -> None:
        now = self._now()
        try:
            payload = json.loads(message.data)
        except (json.JSONDecodeError, TypeError):
            payload = None
        if not isinstance(payload, dict):
            self.planner = FreeGoalPlannerReadiness(False, None, None, now)
            return
        goal_x, goal_y = self._planner_goal(payload)
        try:
            planner_goal_stamp = float(payload["navigation_goal_stamp"])
            planner_goal_yaw = float(payload["navigation_goal_yaw"])
        except (KeyError, TypeError, ValueError, OverflowError):
            planner_goal_stamp = None
            planner_goal_yaw = None
        accepted = self.state.accepted_goal
        complete_identity_match = bool(
            accepted is not None
            and goal_x is not None
            and goal_y is not None
            and planner_goal_stamp is not None
            and planner_goal_yaw is not None
            and self.accepted_goal_publication_stamp is not None
            and goal_identity_matches(
                actual_x=goal_x,
                actual_y=goal_y,
                actual_yaw=planner_goal_yaw,
                actual_stamp=planner_goal_stamp,
                expected_x=float(accepted.goal_x),
                expected_y=float(accepted.goal_y),
                expected_yaw=float(accepted.goal_yaw),
                expected_stamp=self.accepted_goal_publication_stamp,
                position_tolerance=self.config.goal_match_tolerance,
                identity_tolerance=1.0e-6,
            )
        )
        self.planner = FreeGoalPlannerReadiness(
            ready=payload.get("ready") is True and complete_identity_match,
            goal_x=goal_x,
            goal_y=goal_y,
            receipt_stamp=now,
        )
        matching_completion = bool(
            complete_identity_match
            and (
                payload.get("mission_complete") is True
                or payload.get("reason") == "MISSION_COMPLETE"
            )
        )
        if matching_completion:
            self.state.complete()

    def _on_preflight_status(self, message: String) -> None:
        now = self._now()
        try:
            payload = json.loads(message.data)
        except (json.JSONDecodeError, TypeError):
            payload = None
        self.preflight = FreeGoalPreflightReadiness(
            passed=bool(isinstance(payload, dict) and payload.get("passed") is True),
            receipt_stamp=now,
        )

    def _on_risk_assessment_status(self, message: String) -> None:
        now = self._now()
        try:
            payload = json.loads(message.data)
        except (json.JSONDecodeError, TypeError):
            payload = None
        self.risk_assessment_payload = payload if isinstance(payload, dict) else None
        self.risk_assessment_receipt_stamp = now

    def _risk_assessment_readiness(
        self, now: Optional[float] = None
    ) -> RiskAssessmentReadiness:
        return evaluate_required_risk_assessment(
            self.required_risk_provider,
            self.risk_assessment_payload,
            self.risk_assessment_receipt_stamp,
            now=self._now() if now is None else now,
            timeout=self.risk_assessment_timeout,
            shared_minimum_speed=self.shared_minimum_speed,
            shared_target_speed=self.shared_target_speed,
        )

    def _on_stop_mission(self, _request: Trigger.Request, response: Trigger.Response):
        already_stopped = self.state.stop_latched
        self.state.stop()
        self._publish_goal_invalidation()
        self._publish_heartbeat()
        response.success = True
        response.message = (
            "mission stop was already latched"
            if already_stopped
            else "mission stop latched"
        )
        return response

    def _authorization(self, now: Optional[float] = None) -> FreeGoalAuthorization:
        return evaluate_free_goal_authorization(
            self.state,
            self.ego,
            self.costmap,
            self.planner,
            self.preflight,
            now=self._now() if now is None else now,
            config=self.config,
            enabled=self.enabled,
        )

    def _status_payload(self) -> dict:
        now = self._now()
        authorization = self._authorization(now)
        risk_readiness = self._risk_assessment_readiness(now)
        goal = self.state.accepted_goal
        validation = self.state.last_validation
        return {
            "owner": self.get_name(),
            "owner_node": self.get_name(),
            "ready": authorization.ready,
            "armed": authorization.armed,
            "external_stop": self.state.stop_latched,
            "reason": authorization.reason,
            "goal_active": self.state.active,
            "goal_accepted": goal is not None,
            "candidate_goal_received": self.candidate_goal_receipt_stamp is not None,
            "candidate_goal_receipt_stamp": self.candidate_goal_receipt_stamp,
            "goal_accepted_for_planning": goal is not None,
            "accepted_for_motion": authorization.armed,
            "goal_revision": self.state.revision,
            "goal_replaceable": True,
            "one_shot": False,
            "goal_frame": self.config.frame_id,
            "goal_source_frame": self.last_goal_source_frame,
            "goal_x": None if goal is None else goal.goal_x,
            "goal_y": None if goal is None else goal.goal_y,
            "goal_yaw": None if goal is None else goal.goal_yaw,
            "goal_source_stamp": self.accepted_goal_source_stamp,
            "goal_receipt_stamp": self.accepted_goal_receipt_stamp,
            "goal_publication_stamp": self.accepted_goal_publication_stamp,
            "required_risk_provider": self.required_risk_provider or None,
            "risk_assessment_required": risk_readiness.required,
            "risk_assessment_ready": risk_readiness.ready,
            "risk_assessment_reason": risk_readiness.reason,
            "risk_assessment_provider": risk_readiness.provider,
            "risk_assessment_age": risk_readiness.age,
            "pre_goal_bound_valid": bool(
                self.risk_assessment_payload is not None
                and self.risk_assessment_payload.get("pre_goal_bound_valid") is True
            ),
            "last_goal_validation": (
                None if validation is None else validation.reason
            ),
            "last_goal_blocking_cell": (
                None
                if validation is None or validation.blocking_cell_x is None
                else {
                    "x": validation.blocking_cell_x,
                    "y": validation.blocking_cell_y,
                    "value": validation.blocking_value,
                }
            ),
            "last_tf_error": self.last_tf_error,
            "footprint_clearance": self.config.footprint_clearance,
            "ego_source_age": authorization.ego_source_age,
            "ego_receipt_age": authorization.ego_receipt_age,
            "costmap_source_age": authorization.costmap_source_age,
            "costmap_receipt_age": authorization.costmap_receipt_age,
            "planner_status_age": authorization.planner_age,
            "preflight_status_age": authorization.preflight_age,
            "planner_goal_x": None if self.planner is None else self.planner.goal_x,
            "planner_goal_y": None if self.planner is None else self.planner.goal_y,
            "planner_ready": bool(self.planner is not None and self.planner.ready),
            "costmap_frame": None if self.costmap is None else self.costmap.frame_id,
            "costmap_width": None if self.costmap is None else self.costmap.width,
            "costmap_height": None if self.costmap is None else self.costmap.height,
            "mission_complete": self.state.mission_complete,
            "stop_latched": self.state.stop_latched,
            "enabled": self.enabled,
            "note": (
                "Only an observed zero-cost destination can enter planning; "
                "Nav2 route proof and swept-footprint validation remain required."
            ),
        }

    def _publish_heartbeat(self) -> None:
        authorization = self._authorization()
        self.arm_publisher.publish(Bool(data=authorization.armed))
        self.stop_publisher.publish(Bool(data=self.state.stop_latched))
        self.status_publisher.publish(
            String(data=json.dumps(self._status_payload(), separators=(",", ":")))
        )

    def publish_shutdown_stop(self) -> None:
        self.arm_publisher.publish(Bool(data=False))
        self.stop_publisher.publish(Bool(data=True))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DreamFreeGoalAuthorizerNode()
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
