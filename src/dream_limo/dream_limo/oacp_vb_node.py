"""ROS 2 assessor for the OACP-VB comparison arm.

This node implements only the phantom-vehicle reachability assessment and its
two dynamic velocity bounds.  The shared free-navigation node remains the sole
LMPC controller.  ``OACP-VB`` is therefore a velocity-bound adaptation of
Zheng et al. (2025), not the paper's Bézier/consensus-ADMM planner.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import atan2, ceil, cos, isfinite, sin
from typing import Optional, Sequence

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

from .core.oacp_vb import (
    ContingencyBranch,
    GeometryRiskEvaluation,
    MergeConnector,
    OACPVBConfig,
    PVSExtraction,
    VelocityBoundEvaluation,
    build_phantom_merge_connector,
    calibrate_thresholds,
    dynamic_velocity_bound,
    evaluate_geometry_risk,
    extract_pvs_components,
)
from .core.path_tracking import (
    PathValidationError,
    anchor_local_path_start,
    build_path_reference,
    validate_forward_pose_alignment,
    validate_path_points,
)
from .core.types import EgoState
from .limo_scale import DeploymentConfig, deployment_config_for_arena
from .ros_utils import ego_from_odometry, stamp_to_seconds


GEOMETRY_ASSUMPTION = (
    "path_relative_right_lane_merge_connector_nominal_risk_horizon"
)


@dataclass(frozen=True)
class GridContract:
    """ROS-independent world-grid metadata required by the assessor."""

    frame_id: str
    width: int
    height: int
    resolution: float
    origin_x: float
    origin_y: float


@dataclass(frozen=True)
class AssessmentResult:
    """Pure geometry result used by the ROS wrapper and unit tests."""

    connector: MergeConnector
    extraction: PVSExtraction
    pre_goal_velocity_bound: float
    planned_horizon: Optional[np.ndarray]
    risk: Optional[GeometryRiskEvaluation]
    exploration: Optional[VelocityBoundEvaluation]
    fallback: Optional[VelocityBoundEvaluation]

    @property
    def exact_bound_valid(self) -> bool:
        return (
            self.planned_horizon is not None
            and self.risk is not None
            and self.exploration is not None
            and self.fallback is not None
        )


def validate_grid_payload(
    *,
    contract: GridContract,
    frame_id: str,
    width: int,
    height: int,
    resolution: float,
    origin_xyz: Sequence[float],
    origin_quaternion: Sequence[float],
    data: Sequence[float],
) -> np.ndarray:
    """Validate an exact, identity-oriented ``OccupancyGrid`` contract."""

    if frame_id != contract.frame_id:
        raise ValueError(
            "occlusion mask frame does not match the fixed map frame"
        )
    if int(width) != contract.width or int(height) != contract.height:
        raise ValueError(
            "occlusion mask dimensions do not match the fixed grid"
        )
    if (
        not isfinite(float(resolution))
        or abs(float(resolution) - contract.resolution) > 1.0e-9
    ):
        raise ValueError(
            "occlusion mask resolution does not match the fixed grid"
        )
    origin = np.asarray(origin_xyz, dtype=np.float64)
    quaternion = np.asarray(origin_quaternion, dtype=np.float64)
    expected_origin = np.asarray(
        [contract.origin_x, contract.origin_y, 0.0], dtype=np.float64
    )
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        raise ValueError("occlusion mask origin must be finite")
    if not np.allclose(origin, expected_origin, atol=1.0e-9, rtol=0.0):
        raise ValueError("occlusion mask origin does not match the fixed grid")
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError("occlusion mask orientation must be finite")
    if not np.allclose(
        quaternion,
        np.asarray([0.0, 0.0, 0.0, 1.0]),
        atol=1.0e-9,
        rtol=0.0,
    ):
        raise ValueError("occlusion mask grid orientation must be identity")
    try:
        values = np.asarray(data, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("occlusion mask data must be numeric") from exc
    if values.shape != (contract.width * contract.height,):
        raise ValueError("occlusion mask data length is inconsistent")
    if not np.all(np.isfinite(values)):
        raise ValueError("occlusion mask data must be finite")
    if np.any(values < 0.0) or np.any(values > 100.0):
        raise ValueError("occlusion mask values must lie in [0, 100]")
    return values.reshape((contract.height, contract.width)).copy()


def validate_planar_quaternion(values: Sequence[float]) -> float:
    """Return yaw after strict finite, normalized, planar validation."""

    quaternion = np.asarray(values, dtype=np.float64)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError("pose quaternion must contain four finite values")
    norm = float(np.linalg.norm(quaternion))
    if abs(norm - 1.0) > 1.0e-3:
        raise ValueError("pose quaternion must be normalized")
    x, y, z, w = (float(value) for value in quaternion)
    if abs(x) > 1.0e-3 or abs(y) > 1.0e-3:
        raise ValueError("pose quaternion must be planar")
    return atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def provisional_straight_route(
    ego_xy: Sequence[float],
    ego_yaw: float,
    *,
    route_length: float,
    sampling_spacing: float,
) -> np.ndarray:
    """Create the conservative pre-goal route used only for readiness."""

    ego = np.asarray(ego_xy, dtype=np.float64)
    scalar_values = (ego_yaw, route_length, sampling_spacing)
    if (
        ego.shape != (2,)
        or not np.all(np.isfinite(ego))
        or not all(isfinite(float(value)) for value in scalar_values)
        or route_length <= 0.0
        or sampling_spacing <= 0.0
    ):
        raise ValueError("invalid provisional-route inputs")
    sample_count = max(2, int(ceil(route_length / sampling_spacing)) + 1)
    distances = np.linspace(0.0, float(route_length), sample_count)
    direction = np.asarray([cos(float(ego_yaw)), sin(float(ego_yaw))])
    return ego[None, :] + distances[:, None] * direction[None, :]


def canonical_assessment_path(
    raw_points: Sequence[Sequence[float]] | np.ndarray,
    pose_yaws: Sequence[float] | np.ndarray,
    ego: Optional[EgoState],
    *,
    maximum_start_gap: float,
) -> np.ndarray:
    """Apply the same forward/anchor contract as the shared free planner."""

    points = np.asarray(raw_points, dtype=np.float64)
    yaws = np.asarray(pose_yaws, dtype=np.float64)
    validate_forward_pose_alignment(points, yaws)
    validated = validate_path_points(points)
    if ego is None:
        return validated
    anchored, _inserted = anchor_local_path_start(
        validated,
        ego_xy=(ego.x, ego.y),
        ego_yaw=ego.yaw,
        maximum_start_gap=maximum_start_gap,
    )
    return anchored


def compute_assessment(
    *,
    ego: EgoState,
    shadow_mask: np.ndarray,
    path_points: Optional[np.ndarray],
    deployment: DeploymentConfig,
    oacp_config: OACPVBConfig,
    perception_range: float,
    sampling_spacing: float,
    merge_length: float,
    conflict_distance: float,
    risk_evaluation_steps: int,
) -> AssessmentResult:
    """Compute a pre-goal bound or exact path-relative OACP-VB bounds."""

    ego_xy = np.asarray([ego.x, ego.y], dtype=np.float64)
    if path_points is None:
        route = provisional_straight_route(
            ego_xy,
            ego.yaw,
            route_length=perception_range,
            sampling_spacing=sampling_spacing,
        )
    else:
        route = validate_path_points(path_points)
    connector = build_phantom_merge_connector(
        route,
        ego_xy,
        lane_width=oacp_config.lane_width,
        perception_range=perception_range,
        sampling_spacing=sampling_spacing,
        merge_length=merge_length,
    )
    extraction = extract_pvs_components(
        shadow_mask,
        connector,
        ego_xy,
        grid_origin_xy=(deployment.grid.x_min, deployment.grid.y_min),
        grid_resolution=deployment.grid.resolution,
        perception_range=perception_range,
        config=oacp_config,
    )
    pre_goal_bound = (
        oacp_config.v_occ_min
        if extraction.components
        else oacp_config.v_occ_max
    )
    if path_points is None:
        return AssessmentResult(
            connector=connector,
            extraction=extraction,
            pre_goal_velocity_bound=pre_goal_bound,
            planned_horizon=None,
            risk=None,
            exploration=None,
            fallback=None,
        )

    reference = build_path_reference(
        route,
        ego_xy=ego_xy,
        ego_yaw=ego.yaw,
        horizon=risk_evaluation_steps,
        dt=deployment.mpc.dt,
        cruise_speed=oacp_config.v_occ_max,
        braking_deceleration=deployment.mpc.mission_braking_deceleration,
        maximum_cross_track_error=(
            deployment.mpc.maximum_path_cross_track_error
        ),
    )
    planned_horizon = np.asarray(reference[0:2, :].T, dtype=np.float64)
    risk = evaluate_geometry_risk(
        planned_horizon,
        connector,
        extraction,
        oacp_config,
        conflict_distance=conflict_distance,
    )
    exploration = dynamic_velocity_bound(
        risk.risk_total,
        oacp_config,
        ContingencyBranch.EXPLORATION,
    )
    fallback = dynamic_velocity_bound(
        risk.risk_total,
        oacp_config,
        ContingencyBranch.FALLBACK,
    )
    return AssessmentResult(
        connector=connector,
        extraction=extraction,
        pre_goal_velocity_bound=pre_goal_bound,
        planned_horizon=planned_horizon,
        risk=risk,
        exploration=exploration,
        fallback=fallback,
    )


def _polyline_interval(
    connector: MergeConnector,
    start: float,
    end: float,
) -> np.ndarray:
    """Interpolate a visible marker segment on connector arc coordinates."""

    lower = max(0.0, float(start))
    upper = min(float(end), float(connector.cumulative_s[-1]))
    if upper <= lower:
        return np.empty((0, 2), dtype=np.float64)
    inside = (
        (connector.cumulative_s > lower)
        & (connector.cumulative_s < upper)
    )
    coordinates = np.concatenate(
        (
            np.asarray([lower]),
            connector.cumulative_s[inside],
            np.asarray([upper]),
        )
    )
    return np.column_stack(
        (
            np.interp(
                coordinates, connector.cumulative_s, connector.points[:, 0]
            ),
            np.interp(
                coordinates, connector.cumulative_s, connector.points[:, 1]
            ),
        )
    )


class OACPVBNode(Node):
    """Publish path-relative phantom reachability and velocity bounds."""

    def __init__(self) -> None:
        super().__init__("dream_oacp_vb_assessor")
        self.declare_parameter("arena_file", "")
        self.declare_parameter("target_speed", 0.15)
        self.declare_parameter("c_th_max_exploration", 4.5)
        self.declare_parameter("c_th_max_fallback", 6.0)
        self.declare_parameter("thresholds_calibrated", False)
        self.declare_parameter("calibration_logging_only", False)
        self.declare_parameter("input_timeout", 0.50)
        self.declare_parameter("path_timeout", 1.50)
        self.declare_parameter("path_start_anchor_tolerance", 0.20)
        self.declare_parameter("source_future_tolerance", 0.10)
        self.declare_parameter("update_rate", 10.0)
        self.declare_parameter("prediction_horizon", 4.0)
        self.declare_parameter("v_pv_max", 1.0)
        self.declare_parameter("confidence_z", 1.645)
        self.declare_parameter("perception_range", 3.0)
        self.declare_parameter("sampling_spacing", 0.05)
        # Zero selects the reviewed automatic connector length.  An explicit
        # positive override remains available for documented geometry studies.
        self.declare_parameter("merge_length", 0.0)
        self.declare_parameter("v_occ_min_ratio", 0.55)

        self.deployment = deployment_config_for_arena(
            str(self.get_parameter("arena_file").value)
        )
        default_conflict_distance = (
            self.deployment.mpc.robot_width
            + self.deployment.safety.collision_inflation_margin
        )
        self.declare_parameter("conflict_distance", default_conflict_distance)

        target_speed = self._finite_parameter("target_speed")
        input_timeout = self._positive_parameter("input_timeout")
        path_timeout = self._positive_parameter("path_timeout")
        path_start_anchor_tolerance = self._positive_parameter(
            "path_start_anchor_tolerance"
        )
        source_future_tolerance = self._finite_parameter(
            "source_future_tolerance"
        )
        update_rate = self._positive_parameter("update_rate")
        prediction_horizon = self._positive_parameter("prediction_horizon")
        v_pv_max = self._positive_parameter("v_pv_max")
        confidence_z = self._positive_parameter("confidence_z")
        perception_range = self._positive_parameter("perception_range")
        sampling_spacing = self._positive_parameter("sampling_spacing")
        v_occ_min_ratio = self._finite_parameter("v_occ_min_ratio")
        conflict_distance = self._positive_parameter("conflict_distance")
        c_th_max_exploration = self._positive_parameter(
            "c_th_max_exploration"
        )
        c_th_max_fallback = self._positive_parameter("c_th_max_fallback")
        if not (
            self.deployment.mpc.minimum_speed
            <= target_speed
            <= self.deployment.mpc.maximum_speed
        ):
            raise ValueError(
                "target_speed is outside the shared MPC speed limits"
            )
        if not 0.0 < v_occ_min_ratio <= 1.0:
            raise ValueError("v_occ_min_ratio must lie in (0, 1]")
        if source_future_tolerance < 0.0:
            raise ValueError("source_future_tolerance must be nonnegative")
        if c_th_max_fallback < c_th_max_exploration:
            raise ValueError(
                "fallback threshold must not be below exploration threshold"
            )
        if sampling_spacing >= perception_range:
            raise ValueError("sampling_spacing must be below perception_range")

        configured_merge_length = self._finite_parameter("merge_length")
        nominal_risk_distance = target_speed * prediction_horizon
        if configured_merge_length < 0.0:
            raise ValueError("merge_length must be zero (auto) or positive")
        self.merge_length_is_automatic = configured_merge_length == 0.0
        if self.merge_length_is_automatic:
            merge_length = min(perception_range, nominal_risk_distance)
            if (
                merge_length <= sampling_spacing
                or merge_length > nominal_risk_distance + 1.0e-12
            ):
                raise ValueError(
                    "automatic phantom connector cannot converge within the "
                    "nominal OACP risk horizon"
                )
        else:
            merge_length = configured_merge_length
        if merge_length > perception_range:
            raise ValueError("merge_length must not exceed perception_range")

        self.input_timeout = input_timeout
        self.path_timeout = path_timeout
        self.path_start_anchor_tolerance = path_start_anchor_tolerance
        self.source_future_tolerance = source_future_tolerance
        self.target_speed = target_speed
        self.prediction_horizon = prediction_horizon
        self.perception_range = perception_range
        self.sampling_spacing = sampling_spacing
        self.merge_length = merge_length
        self.conflict_distance = conflict_distance
        self.risk_evaluation_steps = int(
            ceil(prediction_horizon / self.deployment.mpc.dt)
        )
        if self.risk_evaluation_steps < self.deployment.mpc.horizon:
            raise ValueError(
                "risk-evaluation horizon must cover the shared MPC horizon"
            )
        self.thresholds_calibrated = bool(
            self.get_parameter("thresholds_calibrated").value
        )
        self.calibration_logging_only = bool(
            self.get_parameter("calibration_logging_only").value
        )
        self.oacp_config = OACPVBConfig(
            v_pv_max=v_pv_max,
            prediction_horizon=prediction_horizon,
            lane_width=self.deployment.arena.lane_width,
            confidence_z=confidence_z,
            c_th_min=0.0,
            c_th_max_exploration=c_th_max_exploration,
            c_th_max_fallback=c_th_max_fallback,
            v_occ_min=target_speed * v_occ_min_ratio,
            v_occ_max=target_speed,
        )
        self.grid_contract = GridContract(
            frame_id=self.deployment.grid.frame_id,
            width=self.deployment.grid.nx,
            height=self.deployment.grid.ny,
            resolution=self.deployment.grid.resolution,
            origin_x=self.deployment.grid.x_min,
            origin_y=self.deployment.grid.y_min,
        )
        self._validate_default_connector_convergence()

        self.ego: Optional[EgoState] = None
        self.ego_receipt: Optional[float] = None
        self.ego_source_stamp: Optional[float] = None
        self.ego_rejection_reason = "WAITING_FOR_EGO_STATE"
        self.shadow_mask: Optional[np.ndarray] = None
        self.mask_receipt: Optional[float] = None
        self.mask_source_stamp: Optional[float] = None
        self.mask_rejection_reason = "WAITING_FOR_OCCLUSION_MASK"
        self.path_points: Optional[np.ndarray] = None
        self.path_receipt: Optional[float] = None
        self.path_source_stamp: Optional[float] = None
        self.path_rejection_reason = "WAITING_FOR_GEOMETRIC_PATH"
        self.risk_samples: list[float] = []
        self.last_risk_sample_key: Optional[tuple[float, float, float]] = None
        self.calibration_goal_identity: Optional[tuple[int, float]] = None
        self.calibration_run_active = False

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
        self.status_publisher = self.create_publisher(
            String, "/dream/oacp_vb_status", reliable
        )
        self.marker_publisher = self.create_publisher(
            MarkerArray, "/dream/oacp_vb_markers", reliable
        )
        self.create_subscription(
            Odometry, "/dream/ego_state", self._on_ego, reliable
        )
        self.create_subscription(
            OccupancyGrid,
            "/dream/occlusion_mask",
            self._on_mask,
            reliable,
        )
        self.create_subscription(
            Path, "/dream/geometric_path", self._on_path, latched
        )
        if self.calibration_logging_only:
            self.create_subscription(
                String,
                "/dream/deadman_status",
                self._on_deadman_status,
                reliable,
            )
        self.create_timer(1.0 / update_rate, self._publish_assessment)
        self.get_logger().info(
            "OACP-VB assessor ready: risk horizon "
            f"{self.risk_evaluation_steps}x{self.deployment.mpc.dt:.3f}s, "
            f"shared MPC {self.deployment.mpc.horizon}x"
            f"{self.deployment.mpc.dt:.3f}s, merge_length="
            f"{self.merge_length:.3f}m"
        )

    def _finite_parameter(self, name: str) -> float:
        value = float(self.get_parameter(name).value)
        if not isfinite(value):
            raise ValueError(f"{name} must be finite")
        return value

    def _positive_parameter(self, name: str) -> float:
        value = self._finite_parameter(name)
        if value <= 0.0:
            raise ValueError(f"{name} must be positive")
        return value

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1.0e-9

    def _validate_default_connector_convergence(self) -> None:
        route = provisional_straight_route(
            (0.0, 0.0),
            0.0,
            route_length=self.perception_range,
            sampling_spacing=self.sampling_spacing,
        )
        connector = build_phantom_merge_connector(
            route,
            (0.0, 0.0),
            lane_width=self.oacp_config.lane_width,
            perception_range=self.perception_range,
            sampling_spacing=self.sampling_spacing,
            merge_length=self.merge_length,
        )
        final_offset = float(
            np.linalg.norm(
                connector.points[-1] - connector.reference_points[-1]
            )
        )
        if self.merge_length_is_automatic and final_offset > 1.0e-6:
            raise ValueError(
                "automatic phantom connector does not converge inside "
                "its range"
            )

    def _on_ego(self, message: Odometry) -> None:
        try:
            source_stamp = stamp_to_seconds(message.header.stamp)
            if (
                message.header.frame_id != self.grid_contract.frame_id
                or not isfinite(source_stamp)
                or source_stamp <= 0.0
            ):
                raise ValueError("ego frame or source stamp is invalid")
            pose = message.pose.pose
            twist = message.twist.twist
            values = (
                pose.position.x,
                pose.position.y,
                pose.position.z,
                twist.linear.x,
                twist.linear.y,
                twist.linear.z,
                twist.angular.x,
                twist.angular.y,
                twist.angular.z,
            )
            if not all(isfinite(float(value)) for value in values):
                raise ValueError("ego pose or twist is non-finite")
            validate_planar_quaternion(
                (
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                    pose.orientation.w,
                )
            )
            ego = ego_from_odometry(message)
            if not all(
                isfinite(value)
                for value in (
                    ego.x,
                    ego.y,
                    ego.yaw,
                    ego.speed,
                    ego.yaw_rate,
                    ego.stamp,
                )
            ):
                raise ValueError("derived ego state is non-finite")
        except (TypeError, ValueError, OverflowError) as exc:
            self.ego = None
            self.ego_receipt = None
            self.ego_source_stamp = None
            self.ego_rejection_reason = f"INVALID_EGO_STATE:{exc}"
            return
        self.ego = ego
        self.ego_receipt = self._now()
        self.ego_source_stamp = source_stamp
        self.ego_rejection_reason = "READY"

    def _on_mask(self, message: OccupancyGrid) -> None:
        try:
            source_stamp = stamp_to_seconds(message.header.stamp)
            if not isfinite(source_stamp) or source_stamp <= 0.0:
                raise ValueError("occlusion mask source stamp is invalid")
            origin = message.info.origin
            mask = validate_grid_payload(
                contract=self.grid_contract,
                frame_id=message.header.frame_id,
                width=message.info.width,
                height=message.info.height,
                resolution=message.info.resolution,
                origin_xyz=(
                    origin.position.x,
                    origin.position.y,
                    origin.position.z,
                ),
                origin_quaternion=(
                    origin.orientation.x,
                    origin.orientation.y,
                    origin.orientation.z,
                    origin.orientation.w,
                ),
                data=message.data,
            )
        except (TypeError, ValueError, OverflowError) as exc:
            self.shadow_mask = None
            self.mask_receipt = None
            self.mask_source_stamp = None
            self.mask_rejection_reason = f"INVALID_OCCLUSION_MASK:{exc}"
            return
        self.shadow_mask = mask
        self.mask_receipt = self._now()
        self.mask_source_stamp = source_stamp
        self.mask_rejection_reason = "READY"

    def _on_path(self, message: Path) -> None:
        self.path_points = None
        self.path_receipt = None
        self.path_source_stamp = None
        self.path_rejection_reason = "WAITING_FOR_GEOMETRIC_PATH"
        if not message.poses:
            return
        try:
            source_stamp = stamp_to_seconds(message.header.stamp)
            if (
                message.header.frame_id != self.grid_contract.frame_id
                or not isfinite(source_stamp)
                or source_stamp <= 0.0
            ):
                raise ValueError("path frame or source stamp is invalid")
            points = []
            pose_yaws = []
            for pose_stamped in message.poses:
                pose = pose_stamped.pose
                values = (
                    pose.position.x,
                    pose.position.y,
                    pose.position.z,
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                    pose.orientation.w,
                )
                if not all(isfinite(float(value)) for value in values):
                    raise ValueError("path contains a non-finite pose")
                points.append((pose.position.x, pose.position.y))
                pose_yaws.append(
                    validate_planar_quaternion(
                        (
                            pose.orientation.x,
                            pose.orientation.y,
                            pose.orientation.z,
                            pose.orientation.w,
                        )
                    )
                )
            validated = canonical_assessment_path(
                points,
                pose_yaws,
                self.ego,
                maximum_start_gap=self.path_start_anchor_tolerance,
            )
        except (
            PathValidationError,
            TypeError,
            ValueError,
            OverflowError,
        ) as exc:
            self.path_rejection_reason = f"INVALID_GEOMETRIC_PATH:{exc}"
            return
        self.path_points = validated
        self.path_receipt = self._now()
        self.path_source_stamp = source_stamp
        self.path_rejection_reason = "READY"

    def _on_deadman_status(self, message: String) -> None:
        """Scope percentile samples to one accepted, actually armed run."""

        if not self.calibration_logging_only:
            return
        try:
            payload = json.loads(message.data)
        except (json.JSONDecodeError, TypeError):
            return
        if not isinstance(payload, dict):
            return
        if payload.get("goal_accepted") is True:
            try:
                revision = int(payload["goal_revision"])
                receipt_stamp = float(payload["goal_receipt_stamp"])
            except (KeyError, TypeError, ValueError, OverflowError):
                return
            if (
                isinstance(payload.get("goal_revision"), bool)
                or revision < 0
                or not isfinite(receipt_stamp)
                or receipt_stamp < 0.0
            ):
                return
            identity = (revision, receipt_stamp)
            if identity != self.calibration_goal_identity:
                self.calibration_goal_identity = identity
                self.calibration_run_active = False
                self.risk_samples.clear()
                self.last_risk_sample_key = None
            if payload.get("accepted_for_motion") is True:
                # Latch once the readiness countdown and complete safety chain
                # have allowed motion; transient later stops remain part of
                # this same identified calibration approach.
                self.calibration_run_active = True
        if (
            payload.get("mission_complete") is True
            or payload.get("stop_latched") is True
        ):
            self.calibration_run_active = False

    def _fresh(
        self,
        receipt: Optional[float],
        source_stamp: Optional[float],
        now: float,
        *,
        timeout: Optional[float] = None,
    ) -> bool:
        if receipt is None or source_stamp is None:
            return False
        maximum_age = self.input_timeout if timeout is None else float(timeout)
        receipt_age = now - receipt
        source_age = now - source_stamp
        return bool(
            isfinite(receipt_age)
            and isfinite(source_age)
            and 0.0 <= receipt_age < maximum_age
            and -self.source_future_tolerance
            <= source_age
            < maximum_age
        )

    @staticmethod
    def _age(stamp: Optional[float], now: float) -> Optional[float]:
        if stamp is None:
            return None
        age = now - stamp
        return float(age) if isfinite(age) else None

    def _calibration(self):
        if not any(value > 0.0 for value in self.risk_samples):
            return None
        try:
            return calibrate_thresholds(self.risk_samples)
        except ValueError:
            # The 70th percentile can still be zero when most of the correctly
            # retained occluded-phase samples are zero.  Such a run cannot yet
            # provide a positive velocity-bound threshold.
            return None

    def _base_status(self, now: float) -> dict:
        calibration = self._calibration()
        return {
            "stamp": now,
            "provider": "oacp_vb",
            "method_name": (
                "OACP-VB (velocity-bound adaptation of Zheng et al., 2025)"
            ),
            "assessment_ready": False,
            "pre_goal_bound_valid": False,
            "exact_bound_valid": False,
            "ready": False,
            "reason": "ASSESSMENT_NOT_READY",
            "frame_id": self.grid_contract.frame_id,
            "geometry_assumption": GEOMETRY_ASSUMPTION,
            "thresholds_calibrated": self.thresholds_calibrated,
            "calibration_logging_only": self.calibration_logging_only,
            "calibration_run_active": self.calibration_run_active,
            "calibration_goal_revision": (
                None
                if self.calibration_goal_identity is None
                else self.calibration_goal_identity[0]
            ),
            "calibration_goal_receipt_stamp": (
                None
                if self.calibration_goal_identity is None
                else self.calibration_goal_identity[1]
            ),
            "calibration_sample_scope": (
                "single_goal_after_first_motion_authorization"
                if self.calibration_logging_only
                else "disabled"
            ),
            "path_source_stamp": self.path_source_stamp,
            "risk_reducer": "maximum_over_nominal_risk_horizon",
            "risk_evaluation_horizon_seconds": (
                self.risk_evaluation_steps * self.deployment.mpc.dt
            ),
            "risk_evaluation_steps": self.risk_evaluation_steps,
            "shared_mpc_horizon_seconds": (
                self.deployment.mpc.horizon * self.deployment.mpc.dt
            ),
            "shared_mpc_horizon_steps": self.deployment.mpc.horizon,
            "shared_mpc_dt": self.deployment.mpc.dt,
            "prediction_horizon": self.prediction_horizon,
            "perception_range": self.perception_range,
            "merge_length": self.merge_length,
            "merge_length_automatic": self.merge_length_is_automatic,
            "conflict_distance": self.conflict_distance,
            "v_pv_max": self.oacp_config.v_pv_max,
            "v_occ_min": self.oacp_config.v_occ_min,
            "v_occ_max": self.oacp_config.v_occ_max,
            "c_th_min": self.oacp_config.c_th_min,
            "c_th_max_exploration": (
                self.oacp_config.c_th_max_exploration
            ),
            "c_th_max_fallback": self.oacp_config.c_th_max_fallback,
            "calibration_sample_count": len(self.risk_samples),
            "suggested_c_th_max_exploration": (
                None
                if calibration is None
                else calibration.exploration_threshold
            ),
            "suggested_c_th_max_fallback": (
                None if calibration is None else calibration.fallback_threshold
            ),
            "ego_receipt_age": self._age(self.ego_receipt, now),
            "ego_source_age": self._age(self.ego_source_stamp, now),
            "mask_receipt_age": self._age(self.mask_receipt, now),
            "mask_source_age": self._age(self.mask_source_stamp, now),
            "path_receipt_age": self._age(self.path_receipt, now),
            "path_source_age": self._age(self.path_source_stamp, now),
        }

    @staticmethod
    def _component_status(extraction: PVSExtraction) -> dict:
        components = extraction.components
        return {
            "pvs_component_count": len(components),
            "pvs_start": (
                None
                if not components
                else min(component.interval.start for component in components)
            ),
            "pvs_end": (
                None
                if not components
                else max(component.interval.end for component in components)
            ),
            "pvs_length": sum(
                component.interval.length for component in components
            ),
            "pvs_components": [
                {
                    "start": component.interval.start,
                    "end": component.interval.end,
                    "length": component.interval.length,
                    "was_clipped": component.was_clipped,
                }
                for component in components
            ],
            "pvs_route_sample_count": extraction.route_sample_count,
            "pvs_shadow_sample_count": extraction.shadow_sample_count,
            "pvs_in_range_sample_count": extraction.in_range_sample_count,
        }

    def _publish_assessment(self) -> None:
        now = self._now()
        status = self._base_status(now)
        result: Optional[AssessmentResult] = None
        if not self._fresh(self.ego_receipt, self.ego_source_stamp, now):
            status["reason"] = (
                self.ego_rejection_reason
                if self.ego is None
                else "STALE_EGO_STATE"
            )
        elif not self._fresh(
            self.mask_receipt, self.mask_source_stamp, now
        ):
            status["reason"] = (
                self.mask_rejection_reason
                if self.shadow_mask is None
                else "STALE_OCCLUSION_MASK"
            )
        else:
            assert self.ego is not None
            assert self.shadow_mask is not None
            try:
                result = compute_assessment(
                    ego=self.ego,
                    shadow_mask=self.shadow_mask,
                    path_points=None,
                    deployment=self.deployment,
                    oacp_config=self.oacp_config,
                    perception_range=self.perception_range,
                    sampling_spacing=self.sampling_spacing,
                    merge_length=self.merge_length,
                    conflict_distance=self.conflict_distance,
                    risk_evaluation_steps=self.risk_evaluation_steps,
                )
                status.update(self._component_status(result.extraction))
                status.update(
                    {
                        "assessment_ready": True,
                        "pre_goal_bound_valid": True,
                        "pre_goal_velocity_bound": (
                            result.pre_goal_velocity_bound
                        ),
                        "pre_goal_shadow_pvs_present": bool(
                            result.extraction.components
                        ),
                        "exploration_velocity_bound": (
                            result.pre_goal_velocity_bound
                        ),
                        "fallback_velocity_bound": (
                            result.pre_goal_velocity_bound
                        ),
                        "reason": "WAITING_FOR_GEOMETRIC_PATH",
                    }
                )
                if self._fresh(
                    self.path_receipt,
                    self.path_source_stamp,
                    now,
                    timeout=self.path_timeout,
                ):
                    assert self.path_points is not None
                    result = compute_assessment(
                        ego=self.ego,
                        shadow_mask=self.shadow_mask,
                        path_points=self.path_points,
                        deployment=self.deployment,
                        oacp_config=self.oacp_config,
                        perception_range=self.perception_range,
                        sampling_spacing=self.sampling_spacing,
                        merge_length=self.merge_length,
                        conflict_distance=self.conflict_distance,
                        risk_evaluation_steps=self.risk_evaluation_steps,
                    )
                    assert result.risk is not None
                    assert result.exploration is not None
                    assert result.fallback is not None
                    status.update(self._component_status(result.extraction))
                    status.update(
                        {
                            "ready": True,
                            "exact_bound_valid": True,
                            "reason": "READY",
                            "path_source_stamp": self.path_source_stamp,
                            "risk_total": result.risk.risk_total,
                            "raw_risk_maximum": result.risk.raw_maximum,
                            "frs_intersects_trajectory": any(
                                result.risk.component_intersections
                            ),
                            "ignored_by_remark_2": (
                                result.risk.ignored_by_remark_2
                            ),
                            "exploration_velocity_bound": (
                                result.exploration.velocity_bound
                            ),
                            "fallback_velocity_bound": (
                                result.fallback.velocity_bound
                            ),
                            "exploration_velocity_region": (
                                result.exploration.region.value
                            ),
                            "fallback_velocity_region": (
                                result.fallback.region.value
                            ),
                        }
                    )
                    self._record_calibration_sample(result)
                    # Refresh suggestions after the newly accepted sample.
                    status["calibration_sample_count"] = len(
                        self.risk_samples
                    )
                    calibration = self._calibration()
                    if calibration is not None:
                        status.update(
                            {
                                "suggested_c_th_max_exploration": (
                                    calibration.exploration_threshold
                                ),
                                "suggested_c_th_max_fallback": (
                                    calibration.fallback_threshold
                                ),
                            }
                        )
                elif self.path_points is not None:
                    status["reason"] = "STALE_GEOMETRIC_PATH"
                else:
                    status["reason"] = self.path_rejection_reason
            except (
                PathValidationError,
                TypeError,
                ValueError,
                RuntimeError,
            ) as exc:
                status["reason"] = f"ASSESSMENT_ERROR:{exc}"
                status["ready"] = False
                status["exact_bound_valid"] = False
                self.get_logger().warning(
                    f"OACP-VB assessment failed closed: {exc}",
                    throttle_duration_sec=2.0,
                )

        try:
            encoded = json.dumps(
                status, separators=(",", ":"), allow_nan=False
            )
        except (TypeError, ValueError) as exc:
            encoded = json.dumps(
                {
                    "stamp": now,
                    "provider": "oacp_vb",
                    "assessment_ready": False,
                    "pre_goal_bound_valid": False,
                    "exact_bound_valid": False,
                    "ready": False,
                    "reason": f"STATUS_ENCODING_ERROR:{exc}",
                    "geometry_assumption": GEOMETRY_ASSUMPTION,
                },
                separators=(",", ":"),
            )
            result = None
        self.status_publisher.publish(String(data=encoded))
        self._publish_markers(result, now)

    def _record_calibration_sample(self, result: AssessmentResult) -> None:
        assert result.risk is not None
        if (
            not self.calibration_logging_only
            or not self.calibration_run_active
            or not result.extraction.components
            or self.ego_source_stamp is None
            or self.mask_source_stamp is None
            or self.path_source_stamp is None
        ):
            return
        key = (
            self.ego_source_stamp,
            self.mask_source_stamp,
            self.path_source_stamp,
        )
        if key == self.last_risk_sample_key:
            return
        self.last_risk_sample_key = key
        self.risk_samples.append(float(result.risk.risk_total))
        if len(self.risk_samples) > 10000:
            del self.risk_samples[: len(self.risk_samples) - 10000]

    @staticmethod
    def _set_points(marker: Marker, points: np.ndarray) -> None:
        marker.points = [
            Point(x=float(point[0]), y=float(point[1]), z=0.04)
            for point in points
        ]

    def _line_marker(
        self,
        *,
        stamp,
        marker_id: int,
        namespace: str,
        points: np.ndarray,
        rgba: tuple[float, float, float, float],
        width: float,
    ) -> Marker:
        marker = Marker()
        marker.header.frame_id = self.grid_contract.frame_id
        marker.header.stamp = stamp
        marker.ns = namespace
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = width
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = rgba
        self._set_points(marker, points)
        return marker

    def _publish_markers(
        self,
        result: Optional[AssessmentResult],
        _now: float,
    ) -> None:
        stamp = self.get_clock().now().to_msg()
        clear = Marker()
        clear.action = Marker.DELETEALL
        markers = [clear]
        if result is not None:
            markers.append(
                self._line_marker(
                    stamp=stamp,
                    marker_id=0,
                    namespace="oacp_connector",
                    points=result.connector.points,
                    rgba=(0.15, 0.65, 1.0, 0.85),
                    width=0.025,
                )
            )
            for index, component in enumerate(result.extraction.components):
                pvs_points = _polyline_interval(
                    result.connector,
                    component.interval.start,
                    component.interval.end,
                )
                reachable_points = _polyline_interval(
                    result.connector,
                    component.interval.start,
                    component.interval.end
                    + self.oacp_config.maximum_pvs_length,
                )
                if pvs_points.shape[0] >= 2:
                    markers.append(
                        self._line_marker(
                            stamp=stamp,
                            marker_id=index,
                            namespace="oacp_pvs",
                            points=pvs_points,
                            rgba=(0.95, 0.10, 0.15, 1.0),
                            width=0.075,
                        )
                    )
                if reachable_points.shape[0] >= 2:
                    markers.append(
                        self._line_marker(
                            stamp=stamp,
                            marker_id=index,
                            namespace="oacp_reachable_extent",
                            points=reachable_points,
                            rgba=(1.0, 0.55, 0.05, 0.55),
                            width=0.035,
                        )
                    )
            text = Marker()
            text.header.frame_id = self.grid_contract.frame_id
            text.header.stamp = stamp
            text.ns = "oacp_bounds"
            text.id = 0
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.orientation.w = 1.0
            if self.ego is not None:
                text.pose.position.x = self.ego.x
                text.pose.position.y = self.ego.y
            text.pose.position.z = 0.55
            text.scale.z = 0.13
            text.color.r = 1.0
            text.color.g = 1.0
            text.color.b = 1.0
            text.color.a = 1.0
            if result.risk is None:
                text.text = (
                    "OACP-VB pre-goal "
                    f"v<={result.pre_goal_velocity_bound:.3f} m/s"
                )
            else:
                assert result.exploration is not None
                assert result.fallback is not None
                text.text = (
                    f"OACP-VB r={result.risk.risk_total:.4g} "
                    f"vE={result.exploration.velocity_bound:.3f} "
                    f"vF={result.fallback.velocity_bound:.3f} m/s"
                )
            markers.append(text)
        self.marker_publisher.publish(MarkerArray(markers=markers))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = OACPVBNode()
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
