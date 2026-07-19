"""Pure command conversion and independent watchdog/safety logic."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, isfinite, sin
from typing import Optional

import numpy as np

from dream_limo.limo_scale import SafetyConfig


FOUR_DIFF = 0
ACKERMANN = 1
MECANUM = 2


@dataclass(frozen=True)
class VelocityCommand:
    linear_x: float = 0.0
    angular_z: float = 0.0
    valid: bool = True
    reason: str = "ok"

    @classmethod
    def zero(cls, reason: str) -> "VelocityCommand":
        return cls(0.0, 0.0, False, reason)


@dataclass(frozen=True)
class PlannerControlCommand:
    """MPC output after the planner's fail-closed acceptance gate."""

    target_speed: float = 0.0
    acceleration: float = 0.0
    steering: float = 0.0
    valid: bool = True
    reason: str = "ok"

    @classmethod
    def zero(cls, reason: str) -> "PlannerControlCommand":
        return cls(0.0, 0.0, 0.0, False, reason)


def gate_mpc_output(
    *,
    target_speed: float,
    acceleration: float,
    steering: float,
    command_valid: bool,
    used_fallback: bool,
    maximum_cbf_slack: float,
    maximum_allowed_cbf_slack: float,
) -> PlannerControlCommand:
    """Reject any MPC result that is not safe to forward to the adapter.

    The solver fallback deliberately contains a braking suggestion for offline
    replay/diagnostics.  It is not a certified solution and therefore must
    never cross the ROS planner boundary as a nonzero command.
    """
    limit = float(maximum_allowed_cbf_slack)
    if not isfinite(limit) or limit < 0.0:
        raise ValueError("maximum allowed CBF slack must be finite and nonnegative")
    if used_fallback:
        return PlannerControlCommand.zero("MPC_FALLBACK")
    if not command_valid:
        return PlannerControlCommand.zero("MPC_INVALID")
    values = (float(target_speed), float(acceleration), float(steering))
    if not all(isfinite(value) for value in values):
        return PlannerControlCommand.zero("MPC_NONFINITE_CONTROL")
    slack = float(maximum_cbf_slack)
    if not isfinite(slack):
        return PlannerControlCommand.zero("MPC_NONFINITE_CBF_SLACK")
    if slack > limit:
        return PlannerControlCommand.zero("MPC_CBF_SLACK_EXCEEDED")
    return PlannerControlCommand(*values, True, "ok")


def center_steer_to_limo_firmware(
    center_steer: float,
    *,
    wheelbase: float = 0.20,
    track: float = 0.172,
    steering_scale: float = 2.47,
    maximum_inner_angle: float = 0.48869,
) -> float:
    """Convert bicycle center steer to the raw Ackermann driver field.

    The deployed LIMO driver calculates this conversion but accidentally sends
    the unconverted ``Twist.angular.z``.  DREAM therefore supplies the protocol
    value itself and clamps it before publishing.
    """
    if not isfinite(center_steer):
        raise ValueError("steering command must be finite")
    if abs(center_steer) < 1.0e-12:
        return 0.0
    magnitude = abs(center_steer)
    numerator = 2.0 * wheelbase * sin(magnitude)
    denominator = 2.0 * wheelbase * cos(magnitude) - track * sin(magnitude)
    inner_angle = atan2(numerator, denominator)
    inner_angle = min(inner_angle, maximum_inner_angle)
    return float(np.copysign(inner_angle / steering_scale, center_steer))


class CommandAdapter:
    """Mode-gated, slew-limited conversion from DREAM control to LIMO Twist."""

    def __init__(self, safety: SafetyConfig, *, control_dt: float = 0.05) -> None:
        self.safety = safety
        self.control_dt = float(control_dt)
        self._last_speed = 0.0
        self._last_mode: Optional[int] = None
        self._zero_cycle_required = False

    def reset(self) -> None:
        self._last_speed = 0.0
        self._last_mode = None
        self._zero_cycle_required = False

    def adapt(
        self,
        *,
        target_speed: float,
        center_steer: float,
        motion_mode: Optional[int],
        allow_differential: bool = False,
        desired_yaw_rate: float = 0.0,
    ) -> VelocityCommand:
        if not all(isfinite(value) for value in (target_speed, center_steer, desired_yaw_rate)):
            self._last_speed = 0.0
            return VelocityCommand.zero("NONFINITE_CONTROL")
        if (
            abs(target_speed) <= 1.0e-12
            and abs(center_steer) <= 1.0e-12
            and abs(desired_yaw_rate) <= 1.0e-12
        ):
            # The planner's all-zero control is a fail-closed stop sentinel.
            # It must bypass the normal acceleration slew limiter immediately.
            self._last_speed = 0.0
            return VelocityCommand.zero("PLANNER_STOP")
        if motion_mode is None:
            self._last_speed = 0.0
            return VelocityCommand.zero("NO_MOTION_MODE")
        if self._last_mode is not None and motion_mode != self._last_mode:
            self._zero_cycle_required = True
            self._last_speed = 0.0
        self._last_mode = motion_mode
        if self._zero_cycle_required:
            self._zero_cycle_required = False
            return VelocityCommand.zero("MODE_CHANGED")

        speed = float(np.clip(target_speed, 0.0, self.safety.maximum_speed))
        maximum_delta = self.safety.maximum_acceleration * self.control_dt
        speed = float(
            np.clip(
                speed,
                self._last_speed - maximum_delta,
                self._last_speed + maximum_delta,
            )
        )

        if motion_mode == ACKERMANN:
            angular = float(
                np.clip(
                    center_steer_to_limo_firmware(center_steer),
                    -self.safety.maximum_ackermann_angular_command,
                    self.safety.maximum_ackermann_angular_command,
                )
            )
        elif motion_mode == FOUR_DIFF and allow_differential:
            angular = float(
                np.clip(
                    desired_yaw_rate,
                    -self.safety.maximum_yaw_rate,
                    self.safety.maximum_yaw_rate,
                )
            )
        else:
            self._last_speed = 0.0
            return VelocityCommand.zero("MODE_MISMATCH")
        self._last_speed = speed
        return VelocityCommand(speed, angular, True, "ok")


class SafetySupervisorCore:
    """Fail-closed state machine used by the ROS safety supervisor."""

    def __init__(self, config: SafetyConfig) -> None:
        self.config = config
        self.candidate = VelocityCommand.zero("NO_CANDIDATE")
        self.candidate_stamp: Optional[float] = None
        self.odom_stamp: Optional[float] = None
        self.scan_stamp: Optional[float] = None
        self.status_stamp: Optional[float] = None
        self.motion_mode: Optional[int] = None
        self.front_minimum_range = float("inf")
        self.obstacle_latched = False
        self.external_stop_latched = False
        self.armed_since: Optional[float] = None
        self.arm_heartbeat_stamp: Optional[float] = None

    def update_candidate(self, command: VelocityCommand, stamp: float) -> None:
        self.candidate = command
        self.candidate_stamp = float(stamp)

    def update_odom(self, stamp: float) -> None:
        self.odom_stamp = float(stamp)

    def update_status(self, motion_mode: int, stamp: float) -> None:
        self.motion_mode = int(motion_mode)
        self.status_stamp = float(stamp)

    def update_scan(
        self,
        front_ranges: np.ndarray,
        stamp: float,
        *,
        range_max: Optional[float] = None,
    ) -> None:
        values = np.asarray(front_ranges, dtype=np.float64)
        # LaserScan uses +inf for a valid ray with no return inside range_max.
        # Preserve fail-closed behavior for NaN/empty/invalid data, but do not
        # mistake known free space for a missing observation.
        if range_max is not None and isfinite(range_max) and range_max > 0.0:
            values = values.copy()
            values[np.isposinf(values)] = float(range_max)
        finite = values[np.isfinite(values) & (values > 0.0)]
        self.front_minimum_range = (
            float(np.min(finite)) if finite.size else float("inf")
        )
        # No valid ray in the safety sector is unsafe, not free space.
        if finite.size == 0 or self.front_minimum_range < self.config.front_stop_distance:
            self.obstacle_latched = True
        self.scan_stamp = float(stamp)

    def set_external_stop(self, active: bool) -> None:
        if active:
            self.external_stop_latched = True

    def set_armed(self, active: bool, now: float) -> None:
        if not active:
            self.armed_since = None
            self.arm_heartbeat_stamp = None
        elif self.armed_since is None:
            self.armed_since = float(now)
            self.arm_heartbeat_stamp = float(now)
        else:
            # A held-to-run source must repeat True; one latched message is not
            # sufficient to keep the vehicle armed indefinitely.
            self.arm_heartbeat_stamp = float(now)

    def request_reset(self) -> bool:
        if abs(self.candidate.linear_x) > 1.0e-6 or abs(self.candidate.angular_z) > 1.0e-6:
            return False
        self.obstacle_latched = False
        self.external_stop_latched = False
        return True

    @staticmethod
    def _stale(now: float, stamp: Optional[float], timeout: float) -> bool:
        return stamp is None or now - stamp >= timeout or now < stamp

    def evaluate(self, now: float) -> VelocityCommand:
        now = float(now)
        if self.external_stop_latched:
            return VelocityCommand.zero("EXTERNAL_STOP_LATCHED")
        if self.obstacle_latched:
            return VelocityCommand.zero("OBSTACLE_STOP_LATCHED")
        if self.armed_since is None:
            if self.arm_heartbeat_stamp is not None:
                return VelocityCommand.zero("STALE_ARM_HEARTBEAT")
            return VelocityCommand.zero("NOT_ARMED")
        if self._stale(
            now,
            self.arm_heartbeat_stamp,
            self.config.arm_heartbeat_timeout,
        ):
            # A lapse requires explicit re-arming and a new full countdown.
            self.armed_since = None
            return VelocityCommand.zero("STALE_ARM_HEARTBEAT")
        if now - self.armed_since < self.config.countdown_seconds:
            return VelocityCommand.zero("COUNTDOWN")
        if self._stale(now, self.candidate_stamp, self.config.planner_timeout):
            return VelocityCommand.zero("STALE_PLANNER")
        if self._stale(now, self.odom_stamp, self.config.odom_timeout):
            return VelocityCommand.zero("STALE_ODOM")
        if self._stale(now, self.scan_stamp, self.config.scan_timeout):
            return VelocityCommand.zero("STALE_SCAN")
        if self._stale(now, self.status_stamp, self.config.status_timeout):
            return VelocityCommand.zero("STALE_STATUS")
        if self.motion_mode != self.config.required_motion_mode:
            return VelocityCommand.zero("MODE_MISMATCH")
        if not self.candidate.valid:
            return VelocityCommand.zero(self.candidate.reason)
        if not all(
            isfinite(value)
            for value in (self.candidate.linear_x, self.candidate.angular_z)
        ):
            return VelocityCommand.zero("NONFINITE_CANDIDATE")
        angular_limit = (
            self.config.maximum_ackermann_angular_command
            if self.motion_mode == ACKERMANN
            else self.config.maximum_yaw_rate
        )
        return VelocityCommand(
            linear_x=float(np.clip(self.candidate.linear_x, 0.0, self.config.maximum_speed)),
            angular_z=float(
                np.clip(
                    self.candidate.angular_z,
                    -angular_limit,
                    angular_limit,
                )
            ),
            valid=True,
            reason="ok",
        )
