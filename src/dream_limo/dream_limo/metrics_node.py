"""Online experiment metrics published for rosbag capture."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from math import cos, hypot, inf, isfinite
from typing import Any, Mapping, Optional

import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, String

from .core.types import EgoState, Vehicle
from .limo_scale import deployment_config_for_arena
from .ros_utils import (
    child_velocity_to_parent,
    ego_from_odometry,
    quaternion_to_yaw,
    stamp_to_seconds,
    transform_planar,
)


def parse_json_object(data: object) -> Optional[dict]:
    """Return a JSON object, or ``None`` for malformed/non-object input."""

    if not isinstance(data, str):
        return None
    try:
        payload = json.loads(data)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _finite_number(value: object, *, nonnegative: bool = False) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not isfinite(result) or (nonnegative and result < 0.0):
        return None
    return result


@dataclass(frozen=True)
class AcceptedGoalRevision:
    revision: int
    receipt_stamp: float
    publication_stamp: Optional[float]
    goal_x: Optional[float]
    goal_y: Optional[float]

    @property
    def identity(self) -> tuple[int, float]:
        return self.revision, self.receipt_stamp


def accepted_goal_revision(payload: Mapping[str, Any]) -> Optional[AcceptedGoalRevision]:
    """Parse the free-goal authorizer's accepted revision and common time zero."""

    if payload.get("goal_accepted") is not True:
        return None
    revision_value = payload.get("goal_revision")
    if isinstance(revision_value, bool):
        return None
    try:
        revision = int(revision_value)
    except (TypeError, ValueError, OverflowError):
        return None
    if revision < 0 or revision_value != revision:
        return None
    receipt = _finite_number(payload.get("goal_receipt_stamp"), nonnegative=True)
    if receipt is None:
        return None
    publication = _finite_number(
        payload.get("goal_publication_stamp"),
        nonnegative=True,
    )
    goal_x = _finite_number(payload.get("goal_x"))
    goal_y = _finite_number(payload.get("goal_y"))
    if (goal_x is None) != (goal_y is None):
        goal_x = goal_y = None
    return AcceptedGoalRevision(
        revision=revision,
        receipt_stamp=receipt,
        publication_stamp=publication,
        goal_x=goal_x,
        goal_y=goal_y,
    )


@dataclass
class RunningStatistic:
    count: int = 0
    total: float = 0.0
    maximum: float = 0.0

    def add(self, value: object) -> bool:
        sample = _finite_number(value, nonnegative=True)
        if sample is None:
            return False
        self.count += 1
        self.total += sample
        self.maximum = max(self.maximum, sample)
        return True

    @property
    def mean(self) -> Optional[float]:
        return None if self.count == 0 else self.total / self.count


@dataclass
class SlackActivation:
    threshold: float
    activation_samples: int = 0
    activation_events: int = 0
    maximum: float = 0.0
    previous_active: bool = False

    def add(self, value: object) -> bool:
        sample = _finite_number(value, nonnegative=True)
        if sample is None:
            return False
        active = sample > self.threshold
        if active:
            self.activation_samples += 1
            if not self.previous_active:
                self.activation_events += 1
        self.maximum = max(self.maximum, sample)
        self.previous_active = active
        return True


def _duration_below_linear(
    first_speed: float,
    second_speed: float,
    threshold: float,
    duration: float,
) -> float:
    if first_speed < threshold and second_speed < threshold:
        return duration
    if first_speed >= threshold and second_speed >= threshold:
        return 0.0
    fraction = (threshold - first_speed) / (second_speed - first_speed)
    crossing = min(1.0, max(0.0, fraction)) * duration
    return crossing if first_speed < threshold else duration - crossing


@dataclass
class ExperimentRunAccumulator:
    """Pure accumulator for one accepted navigation-goal revision."""

    goal: AcceptedGoalRevision
    fixed_time_seconds: float = 5.0
    slack_threshold: float = 1.0e-6
    motion_acceptance_stamp: Optional[float] = None
    completion_stamp: Optional[float] = None
    configured_target_speed: Optional[float] = None
    current_goal_remaining: Optional[float] = None
    planner_arm: Optional[str] = None
    shared_controller_parameter_hash: Optional[str] = None
    ego_sample_count: int = 0
    traveled_distance: float = 0.0
    speed_time_integral: float = 0.0
    integrated_duration: float = 0.0
    distance_to_goal_at_fixed_time: Optional[float] = None
    last_ego_stamp: Optional[float] = None
    last_ego_x: Optional[float] = None
    last_ego_y: Optional[float] = None
    last_ego_speed: Optional[float] = None
    speed_segments: list[tuple[float, float, float]] = field(default_factory=list)
    cycle_solve_time: RunningStatistic = field(default_factory=RunningStatistic)
    solve_time: RunningStatistic = field(default_factory=RunningStatistic)
    branch_solve_time: dict[str, RunningStatistic] = field(
        default_factory=lambda: {
            "shared": RunningStatistic(),
            "exploration": RunningStatistic(),
            "fallback": RunningStatistic(),
            "clamped": RunningStatistic(),
        }
    )
    velocity_slack: SlackActivation = field(init=False)
    future_velocity_slack: SlackActivation = field(init=False)
    contingency_failures: int = 0
    contingency_clamp_events: int = 0
    safety_false_transitions: int = 0
    oacp: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        fixed = _finite_number(self.fixed_time_seconds, nonnegative=True)
        threshold = _finite_number(self.slack_threshold, nonnegative=True)
        if fixed is None or fixed <= 0.0:
            raise ValueError("fixed_time_seconds must be finite and positive")
        if threshold is None:
            raise ValueError("slack_threshold must be finite and nonnegative")
        self.fixed_time_seconds = fixed
        self.slack_threshold = threshold
        self.velocity_slack = SlackActivation(threshold)
        self.future_velocity_slack = SlackActivation(threshold)

    def accept_motion(self, stamp: object) -> None:
        value = _finite_number(stamp, nonnegative=True)
        if (
            value is not None
            and value >= self.goal.receipt_stamp
            and self.motion_acceptance_stamp is None
        ):
            self.motion_acceptance_stamp = value

    def complete(self, stamp: object) -> None:
        value = _finite_number(stamp, nonnegative=True)
        if (
            value is not None
            and value >= self.goal.receipt_stamp
            and self.completion_stamp is None
        ):
            self.completion_stamp = value

    def add_ego(self, stamp: object, x: object, y: object, speed: object) -> bool:
        sample_stamp = _finite_number(stamp, nonnegative=True)
        sample_x = _finite_number(x)
        sample_y = _finite_number(y)
        sample_speed = _finite_number(speed, nonnegative=True)
        if (
            sample_stamp is None
            or sample_x is None
            or sample_y is None
            or sample_speed is None
            or sample_stamp < self.goal.receipt_stamp
            or (
                self.completion_stamp is not None
                and sample_stamp > self.completion_stamp
            )
            or (
                self.last_ego_stamp is not None
                and sample_stamp <= self.last_ego_stamp
            )
        ):
            return False
        self.ego_sample_count += 1
        if self.last_ego_stamp is not None:
            duration = sample_stamp - self.last_ego_stamp
            self.traveled_distance += hypot(
                sample_x - self.last_ego_x,
                sample_y - self.last_ego_y,
            )
            self.speed_time_integral += (
                0.5 * (self.last_ego_speed + sample_speed) * duration
            )
            self.integrated_duration += duration
            self.speed_segments.append(
                (duration, self.last_ego_speed, sample_speed)
            )
            fixed_stamp = self.goal.receipt_stamp + self.fixed_time_seconds
            if (
                self.distance_to_goal_at_fixed_time is None
                and self.goal.goal_x is not None
                and self.last_ego_stamp <= fixed_stamp <= sample_stamp
            ):
                fraction = (fixed_stamp - self.last_ego_stamp) / duration
                fixed_x = self.last_ego_x + fraction * (sample_x - self.last_ego_x)
                fixed_y = self.last_ego_y + fraction * (sample_y - self.last_ego_y)
                self.distance_to_goal_at_fixed_time = hypot(
                    self.goal.goal_x - fixed_x,
                    self.goal.goal_y - fixed_y,
                )
        elif (
            self.goal.goal_x is not None
            and abs(
                sample_stamp
                - (self.goal.receipt_stamp + self.fixed_time_seconds)
            )
            <= 1.0e-9
        ):
            self.distance_to_goal_at_fixed_time = hypot(
                self.goal.goal_x - sample_x,
                self.goal.goal_y - sample_y,
            )
        self.last_ego_stamp = sample_stamp
        self.last_ego_x = sample_x
        self.last_ego_y = sample_y
        self.last_ego_speed = sample_speed
        return True

    def update_planner(self, payload: Mapping[str, Any]) -> None:
        arm = payload.get("arm")
        if isinstance(arm, str) and arm:
            self.planner_arm = arm
        fingerprint = payload.get("shared_controller_parameter_hash")
        if isinstance(fingerprint, str) and fingerprint:
            self.shared_controller_parameter_hash = fingerprint
        speed = _finite_number(
            payload.get("configured_target_speed"),
            nonnegative=True,
        )
        if speed is not None and speed > 0.0:
            self.configured_target_speed = speed
        for key in ("navigation_goal_remaining", "mission_remaining_distance"):
            remaining = _finite_number(payload.get(key), nonnegative=True)
            if remaining is not None:
                self.current_goal_remaining = remaining
                break

        oacp_keys = (
            "oacp_risk_total",
            "oacp_raw_risk_maximum",
            "oacp_exploration_velocity_bound",
            "oacp_fallback_velocity_bound",
            "oacp_executed_velocity_bound",
            "oacp_v_occ_min",
            "oacp_v_occ_max",
            "oacp_pvs_component_count",
            "oacp_pvs_start",
            "oacp_pvs_end",
            "oacp_pvs_length",
            "oacp_frs_intersects_trajectory",
        )
        for key in oacp_keys:
            value = payload.get(key)
            if isinstance(value, bool):
                self.oacp[key] = value
            else:
                parsed = _finite_number(value, nonnegative=True)
                if parsed is not None:
                    self.oacp[key] = parsed

        branch_keys = {
            "exploration": "oacp_exploration_solve_seconds",
            "fallback": "oacp_fallback_solve_seconds",
            "clamped": "oacp_clamped_solve_seconds",
        }
        branch_samples: list[float] = []
        for branch, key in branch_keys.items():
            sample = _finite_number(payload.get(key), nonnegative=True)
            if sample is not None:
                self.branch_solve_time[branch].add(sample)
                self.solve_time.add(sample)
                branch_samples.append(sample)
        if not branch_samples:
            sample = _finite_number(payload.get("t_mpc"), nonnegative=True)
            if sample is not None:
                self.branch_solve_time["shared"].add(sample)
                self.solve_time.add(sample)
                branch_samples.append(sample)
        cycle = _finite_number(payload.get("t_mpc_total"), nonnegative=True)
        if cycle is None and branch_samples:
            cycle = sum(branch_samples)
        if cycle is not None:
            self.cycle_solve_time.add(cycle)

        self.velocity_slack.add(payload.get("maximum_velocity_slack"))
        self.future_velocity_slack.add(
            payload.get("maximum_future_velocity_slack")
        )
        if payload.get("oacp_contingency_valid") is False:
            self.contingency_failures += 1
        if payload.get("oacp_contingency_clamp_event") is True:
            self.contingency_clamp_events += 1

    def record_safety_transition(self, previous: Optional[bool], current: bool) -> None:
        if previous is True and current is False:
            self.safety_false_transitions += 1

    @property
    def time_weighted_mean_speed(self) -> Optional[float]:
        if self.integrated_duration <= 0.0:
            return None
        return self.speed_time_integral / self.integrated_duration

    @property
    def time_below_half_target_speed(self) -> Optional[float]:
        if self.configured_target_speed is None:
            return None
        threshold = 0.5 * self.configured_target_speed
        return sum(
            _duration_below_linear(first, second, threshold, duration)
            for duration, first, second in self.speed_segments
        )

    def snapshot(self, now: object) -> dict:
        current = _finite_number(now, nonnegative=True)
        end = self.completion_stamp if self.completion_stamp is not None else current
        elapsed = (
            None
            if end is None
            else max(0.0, end - self.goal.receipt_stamp)
        )
        result = {
            "run_metrics_active": True,
            "goal_revision": self.goal.revision,
            "goal_receipt_stamp": self.goal.receipt_stamp,
            "goal_publication_stamp": self.goal.publication_stamp,
            "goal_motion_acceptance_stamp": self.motion_acceptance_stamp,
            "goal_motion_acceptance_stamp_source": (
                None
                if self.motion_acceptance_stamp is None
                else "metrics_receipt_of_authorizer_transition"
            ),
            "run_elapsed_seconds": elapsed,
            "traversal_time_seconds": (
                None
                if self.completion_stamp is None
                else self.completion_stamp - self.goal.receipt_stamp
            ),
            "ego_sample_count": self.ego_sample_count,
            "traveled_distance": self.traveled_distance,
            "time_weighted_mean_speed": self.time_weighted_mean_speed,
            "time_below_half_target_speed": self.time_below_half_target_speed,
            "distance_to_goal_fixed_time_seconds": self.fixed_time_seconds,
            "distance_to_goal_at_fixed_time": self.distance_to_goal_at_fixed_time,
            "planner_arm": self.planner_arm,
            "shared_controller_parameter_hash": (
                self.shared_controller_parameter_hash
            ),
            "configured_target_speed": self.configured_target_speed,
            "current_goal_remaining": self.current_goal_remaining,
            "mpc_solve_count": self.solve_time.count,
            "mpc_solve_mean_seconds": self.solve_time.mean,
            "mpc_solve_max_seconds": (
                None if self.solve_time.count == 0 else self.solve_time.maximum
            ),
            "mpc_total_solve_count": self.solve_time.count,
            "mpc_total_solve_mean_seconds": self.solve_time.mean,
            "mpc_total_solve_max_seconds": (
                None if self.solve_time.count == 0 else self.solve_time.maximum
            ),
            "mpc_cycle_total_count": self.cycle_solve_time.count,
            "mpc_cycle_total_mean_seconds": self.cycle_solve_time.mean,
            "mpc_cycle_total_max_seconds": (
                None
                if self.cycle_solve_time.count == 0
                else self.cycle_solve_time.maximum
            ),
            "velocity_slack_activation_samples": (
                self.velocity_slack.activation_samples
            ),
            "velocity_slack_activation_events": (
                self.velocity_slack.activation_events
            ),
            "velocity_slack_maximum": self.velocity_slack.maximum,
            "future_velocity_slack_activation_samples": (
                self.future_velocity_slack.activation_samples
            ),
            "future_velocity_slack_activation_events": (
                self.future_velocity_slack.activation_events
            ),
            "future_velocity_slack_maximum": self.future_velocity_slack.maximum,
            "velocity_slack_activation_threshold": self.slack_threshold,
            "contingency_failure_count": self.contingency_failures,
            "contingency_clamp_event_count": self.contingency_clamp_events,
            "safety_false_transition_count": self.safety_false_transitions,
        }
        for branch, statistic in self.branch_solve_time.items():
            result[f"mpc_{branch}_solve_count"] = statistic.count
            result[f"mpc_{branch}_solve_mean_seconds"] = statistic.mean
            result[f"mpc_{branch}_solve_max_seconds"] = (
                None if statistic.count == 0 else statistic.maximum
            )
        result.update(self.oacp)
        return result


class DreamMetricsNode(Node):
    def __init__(self) -> None:
        super().__init__("dream_metrics")
        self.declare_parameter("ego_topic", "/dream/ego_state")
        self.declare_parameter("merger_odom_topic", "/merger/wheel/odom")
        self.declare_parameter("merger_visible_topic", "/dream/merger_visible")
        self.declare_parameter("planner_status_topic", "/dream/planner_status")
        self.declare_parameter("deadman_status_topic", "/dream/deadman_status")
        self.declare_parameter("safety_status_topic", "/dream/safety_status")
        self.declare_parameter("distance_to_goal_fixed_time", 5.0)
        self.declare_parameter("velocity_slack_activation_threshold", 1.0e-6)
        self.declare_parameter("arena_file", "")
        self.declare_parameter("map_to_odom_x", 0.0)
        self.declare_parameter("map_to_odom_y", 0.0)
        self.declare_parameter("map_to_odom_yaw", 0.0)
        self.declare_parameter("alignment_topic", "/dream/map_alignment")
        self.config = deployment_config_for_arena(
            str(self.get_parameter("arena_file").value)
        )
        self.fixed_time_seconds = _finite_number(
            self.get_parameter("distance_to_goal_fixed_time").value,
            nonnegative=True,
        )
        self.slack_activation_threshold = _finite_number(
            self.get_parameter("velocity_slack_activation_threshold").value,
            nonnegative=True,
        )
        if self.fixed_time_seconds is None or self.fixed_time_seconds <= 0.0:
            raise ValueError("distance_to_goal_fixed_time must be positive")
        if self.slack_activation_threshold is None:
            raise ValueError(
                "velocity_slack_activation_threshold must be nonnegative"
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
        self.run_metrics: Optional[ExperimentRunAccumulator] = None
        self.accepted_goal_identity: Optional[tuple[int, float]] = None
        self.latest_safety_safe: Optional[bool] = None
        self.merger_sample_received_for_run = False
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
        deadman_topic = str(self.get_parameter("deadman_status_topic").value)
        if deadman_topic:
            self.create_subscription(
                String,
                deadman_topic,
                self._on_deadman_status,
                10,
            )
        safety_topic = str(self.get_parameter("safety_status_topic").value)
        if safety_topic:
            self.create_subscription(
                String,
                safety_topic,
                self._on_safety_status,
                10,
            )
        self.create_subscription(String, "/dream/drift_status", self._on_drift, 10)
        self.publisher = self.create_publisher(String, "/dream/metrics", 10)
        self.create_timer(0.2, self._publish)

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1.0e-9

    def _reset_legacy_run_metrics(self) -> None:
        self.reveal_time = None
        self.ttc_at_reveal = inf
        self.ttc_at_reveal_reason = "NOT_REVEALED"
        self.closing_speed_at_reveal = None
        self.lateral_separation_at_reveal = None
        self.projected_conflict_arrival_margin_at_reveal = inf
        self.projected_conflict_arrival_margin_reason = "NOT_REVEALED"
        self.ego_conflict_entry_time = None
        self.merger_conflict_entry_time = None
        self.conflict_zone_overlap_samples = 0
        self.minimum_conflict_zone_clearance = inf
        self.minimum_clearance = inf
        self.minimum_post_reveal_clearance = inf
        self.minimum_ttc = inf
        self.maximum_abs_acceleration = 0.0
        self.maximum_abs_jerk = 0.0
        self.previous_acceleration = None
        self.previous_acceleration_time = None
        self.veto_activations = 0
        self.veto_events = 0
        self.previous_veto = False
        self.risk_at_ego = 0.0
        self.maximum_drift_seconds = 0.0
        self.maximum_decision_seconds = 0.0
        self.maximum_mpc_seconds = 0.0

    def _on_deadman_status(self, message: String) -> None:
        payload = parse_json_object(message.data)
        if payload is None:
            return
        now = self._now()
        accepted = accepted_goal_revision(payload)
        if accepted is not None and accepted.identity != self.accepted_goal_identity:
            self._reset_legacy_run_metrics()
            self.accepted_goal_identity = accepted.identity
            self.merger_sample_received_for_run = False
            self.run_metrics = ExperimentRunAccumulator(
                accepted,
                fixed_time_seconds=self.fixed_time_seconds,
                slack_threshold=self.slack_activation_threshold,
            )
            if self.ego is not None:
                self.run_metrics.add_ego(
                    accepted.receipt_stamp,
                    self.ego.x,
                    self.ego.y,
                    self.ego.speed,
                )
        if (
            self.run_metrics is not None
            and accepted is not None
            and accepted.identity == self.accepted_goal_identity
        ):
            if payload.get("accepted_for_motion") is True:
                self.run_metrics.accept_motion(now)
            if payload.get("mission_complete") is True:
                self.run_metrics.complete(now)

    def _on_safety_status(self, message: String) -> None:
        payload = parse_json_object(message.data)
        if payload is None or not isinstance(payload.get("safe"), bool):
            return
        current = bool(payload["safe"])
        run_metrics = getattr(self, "run_metrics", None)
        if run_metrics is not None:
            run_metrics.record_safety_transition(
                getattr(self, "latest_safety_safe", None),
                current,
            )
        self.latest_safety_safe = current

    def _on_ego(self, message: Odometry) -> None:
        self.ego = ego_from_odometry(message)
        run_metrics = getattr(self, "run_metrics", None)
        if run_metrics is not None:
            run_metrics.add_ego(
                self._now(),
                self.ego.x,
                self.ego.y,
                self.ego.speed,
            )
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
        source_yaw = quaternion_to_yaw(message.pose.pose.orientation)
        odom_vx, odom_vy = child_velocity_to_parent(
            velocity.x,
            velocity.y,
            child_yaw=source_yaw,
        )
        tx, ty, yaw = self.map_alignment
        x, y, vx, vy = transform_planar(
            position.x,
            position.y,
            odom_vx,
            odom_vy,
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
            heading=source_yaw + yaw,
            length=0.22,
            width=0.22,
            stamp=stamp_to_seconds(message.header.stamp),
        )
        run_metrics = getattr(self, "run_metrics", None)
        if run_metrics is not None and self._now() >= run_metrics.goal.receipt_stamp:
            self.merger_sample_received_for_run = True
        self._update_pair_metrics()

    def _ttc_diagnostic(self):
        if (
            getattr(self, "run_metrics", None) is not None
            and not getattr(self, "merger_sample_received_for_run", False)
        ):
            return inf, "MISSING_MERGER_ODOM_FOR_RUN"
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
        if (
            getattr(self, "run_metrics", None) is not None
            and not getattr(self, "merger_sample_received_for_run", False)
        ):
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
            if (
                self.ego is not None
                and self.merger is not None
                and (
                    getattr(self, "run_metrics", None) is None
                    or getattr(self, "merger_sample_received_for_run", False)
                )
            ):
                self.closing_speed_at_reveal = self.ego.speed - self.merger.vx
                self.lateral_separation_at_reveal = abs(self.ego.y - self.merger.y)
                (
                    self.projected_conflict_arrival_margin_at_reveal,
                    self.projected_conflict_arrival_margin_reason,
                ) = self._projected_conflict_arrival_margin()

    def _on_planner(self, message: String) -> None:
        payload = parse_json_object(message.data)
        if payload is None:
            return
        receipt_stamp = self._now()
        planner_stamp = _finite_number(payload.get("stamp"), nonnegative=True)
        sample_stamp = receipt_stamp if planner_stamp is None else planner_stamp
        run_metrics = getattr(self, "run_metrics", None)
        if run_metrics is not None and sample_stamp >= run_metrics.goal.receipt_stamp:
            run_metrics.update_planner(payload)
            if (
                payload.get("mission_complete") is True
                or payload.get("reason") == "MISSION_COMPLETE"
            ):
                run_metrics.complete(sample_stamp)

        vetoed = payload.get("vetoed") is True
        if vetoed and not self.previous_veto:
            self.veto_events += 1
        if vetoed:
            self.veto_activations += 1
        self.previous_veto = vetoed
        risk = _finite_number(payload.get("risk_at_ego"), nonnegative=True)
        if risk is not None:
            self.risk_at_ego = risk
        decision_seconds = _finite_number(
            payload.get("t_decision"),
            nonnegative=True,
        )
        if decision_seconds is not None:
            self.maximum_decision_seconds = max(
                self.maximum_decision_seconds,
                decision_seconds,
            )
        mpc_seconds = _finite_number(payload.get("t_mpc"), nonnegative=True)
        if mpc_seconds is not None:
            self.maximum_mpc_seconds = max(
                self.maximum_mpc_seconds,
                mpc_seconds,
            )
        acceleration = _finite_number(payload.get("acceleration"))
        if acceleration is None:
            return
        now = sample_stamp
        self.maximum_abs_acceleration = max(self.maximum_abs_acceleration, abs(acceleration))
        if (
            self.previous_acceleration is not None
            and self.previous_acceleration_time is not None
            and now > self.previous_acceleration_time
        ):
            control_dt = _finite_number(
                payload.get("control_dt"),
                nonnegative=True,
            )
            if control_dt is None or control_dt <= 0.0:
                control_dt = now - self.previous_acceleration_time
            jerk = (acceleration - self.previous_acceleration) / max(control_dt, 1.0e-6)
            self.maximum_abs_jerk = max(self.maximum_abs_jerk, abs(jerk))
        self.previous_acceleration = acceleration
        self.previous_acceleration_time = now

    def _on_drift(self, message: String) -> None:
        payload = parse_json_object(message.data)
        if payload is None:
            return
        seconds = _finite_number(payload.get("compute_seconds"), nonnegative=True)
        if seconds is not None:
            self.maximum_drift_seconds = max(
                self.maximum_drift_seconds,
                seconds,
            )

    @staticmethod
    def _finite_or_none(value: float):
        return value if isfinite(value) else None

    def _publish(self) -> None:
        run_metrics = getattr(self, "run_metrics", None)
        if (
            self.ego is not None
            and self.merger is not None
            and (
                run_metrics is None
                or getattr(self, "merger_sample_received_for_run", False)
            )
            and self._inside_conflict_zone(self.ego.x, self.ego.y)
            and self._inside_conflict_zone(self.merger.x, self.merger.y)
        ):
            self.conflict_zone_overlap_samples += 1
        message = String()
        payload = {
            "stamp": self._now(),
            "merger_visible": self.visible,
            "merger_odom_available_for_run": bool(
                self.merger is not None
                and (
                    run_metrics is None
                    or getattr(
                        self,
                        "merger_sample_received_for_run",
                        False,
                    )
                )
            ),
            "t_reveal": self.reveal_time,
            "ttc_at_reveal": self._finite_or_none(self.ttc_at_reveal),
            "ttc_at_reveal_valid": self.ttc_at_reveal_reason == "VALID",
            "ttc_at_reveal_reason": self.ttc_at_reveal_reason,
            "closing_speed_at_reveal": self.closing_speed_at_reveal,
            "lateral_separation_at_reveal": (
                self.lateral_separation_at_reveal
            ),
            "projected_conflict_arrival_margin_at_reveal": (
                self._finite_or_none(
                    self.projected_conflict_arrival_margin_at_reveal
                )
            ),
            "projected_conflict_arrival_margin_valid": (
                self.projected_conflict_arrival_margin_reason == "VALID"
            ),
            "projected_conflict_arrival_margin_reason": (
                self.projected_conflict_arrival_margin_reason
            ),
            "ego_conflict_entry_time": self.ego_conflict_entry_time,
            "merger_conflict_entry_time": self.merger_conflict_entry_time,
            "conflict_zone_overlap_samples": (
                self.conflict_zone_overlap_samples
            ),
            "conflict_zone_sample_period": 0.2,
            "minimum_conflict_zone_clearance": self._finite_or_none(
                self.minimum_conflict_zone_clearance
            ),
            "minimum_clearance": self._finite_or_none(
                self.minimum_clearance
            ),
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
            "run_metrics_active": False,
        }
        if run_metrics is not None:
            payload.update(run_metrics.snapshot(payload["stamp"]))
        message.data = json.dumps(
            payload,
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
