"""Pure, independently testable final gate for physical LIMO commands.

The upstream safety supervisor publishes a *candidate* on ``/cmd_vel_test``.
This gate deliberately repeats the critical checks at the final ROS boundary;
no single stale status message or launch parameter is sufficient to move the
robot.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Optional

import numpy as np

from .command_adapter import ACKERMANN, VelocityCommand


def exact_publisher_owner(node_names: list[str], expected_owner: str) -> bool:
    """Return true only for one publisher with the reviewed node name."""

    return bool(expected_owner) and node_names == [expected_owner]


@dataclass(frozen=True)
class HardwareGateConfig:
    """Conservative limits for the first reviewed physical-motion stage."""

    maximum_speed: float = 0.15
    maximum_acceleration: float = 0.35
    maximum_ackermann_angular_command: float = 0.198
    maximum_ackermann_angular_slew: float = 0.40
    publish_rate: float = 20.0
    candidate_timeout: float = 0.20
    odom_timeout: float = 0.25
    scan_timeout: float = 0.40
    status_timeout: float = 1.25
    safety_status_timeout: float = 0.20
    preflight_timeout: float = 2.0
    collision_timeout: float = 0.30
    deadman_timeout: float = 0.30
    world_timeout: float = 0.50
    drift_timeout: float = 0.50
    planner_status_timeout: float = 0.50
    required_motion_mode: int = ACKERMANN

    def __post_init__(self) -> None:
        positive = (
            self.maximum_speed,
            self.maximum_acceleration,
            self.maximum_ackermann_angular_command,
            self.maximum_ackermann_angular_slew,
            self.publish_rate,
            self.candidate_timeout,
            self.odom_timeout,
            self.scan_timeout,
            self.status_timeout,
            self.safety_status_timeout,
            self.preflight_timeout,
            self.collision_timeout,
            self.deadman_timeout,
            self.world_timeout,
            self.drift_timeout,
            self.planner_status_timeout,
        )
        if not all(isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("hardware-gate limits and timeouts must be positive and finite")
        if self.maximum_speed > 0.15:
            raise ValueError("initial hardware speed cap must not exceed 0.15 m/s")
        if self.maximum_acceleration > 0.35:
            raise ValueError("hardware acceleration cap must not exceed 0.35 m/s^2")
        if self.maximum_ackermann_angular_command > 0.198:
            raise ValueError("raw Ackermann command cap must not exceed 0.198")
        if self.maximum_ackermann_angular_slew > 0.40:
            raise ValueError("raw Ackermann slew cap must not exceed 0.40/s")
        if self.publish_rate < 20.0:
            raise ValueError("hardware command gate must publish at 20 Hz or faster")
        timeout_limits = (
            (self.candidate_timeout, 0.50),
            (self.odom_timeout, 0.50),
            (self.scan_timeout, 0.75),
            (self.status_timeout, 1.50),
            (self.safety_status_timeout, 0.50),
            (self.preflight_timeout, 2.50),
            (self.collision_timeout, 0.50),
            (self.deadman_timeout, 0.50),
            (self.world_timeout, 1.00),
            (self.drift_timeout, 1.00),
            (self.planner_status_timeout, 1.00),
        )
        if any(value > maximum for value, maximum in timeout_limits):
            raise ValueError("hardware-gate timeout exceeds its fail-closed ceiling")
        if self.required_motion_mode != ACKERMANN:
            raise ValueError("physical DREAM deployment requires Ackermann mode")


class HardwareCommandGateCore:
    """Fail-closed state machine used by the sole physical command publisher."""

    def __init__(self, config: HardwareGateConfig) -> None:
        self.config = config
        self.candidate = VelocityCommand.zero("NO_CANDIDATE")
        self.candidate_stamp: Optional[float] = None
        self.odom_stamp: Optional[float] = None
        self.scan_stamp: Optional[float] = None
        self.motion_mode: Optional[int] = None
        self.status_stamp: Optional[float] = None

        self.safety_ok = False
        self.safety_reason = "NO_SAFETY_STATUS"
        self.safety_stamp: Optional[float] = None
        self.preflight_ok = False
        self.preflight_stamp: Optional[float] = None
        self.collision_ready = False
        self.trajectory_clear = False
        self.collision_stamp: Optional[float] = None
        self.deadman_ready = False
        self.deadman_armed = False
        self.deadman_stamp: Optional[float] = None
        self.world_ready = False
        self.world_ego_fresh = False
        self.world_scan_fresh = False
        self.world_tracks_fresh = False
        self.world_alignment_received = False
        self.world_stamp: Optional[float] = None
        self.drift_ready = False
        self.drift_stamp: Optional[float] = None
        self.planner_ready = False
        self.planner_used_fallback = True
        self.planner_map_bounds_enforced = False
        self.planner_slack = float("inf")
        self.planner_allowed_slack = 0.0
        self.planner_stamp: Optional[float] = None

        self._last_speed = 0.0
        self._last_angular = 0.0
        self._last_evaluation_stamp: Optional[float] = None

    def update_candidate(self, command: VelocityCommand, stamp: float) -> None:
        self.candidate = command
        self.candidate_stamp = float(stamp)

    def update_odom(self, stamp: float) -> None:
        self.odom_stamp = float(stamp)

    def update_scan(self, stamp: float) -> None:
        self.scan_stamp = float(stamp)

    def update_status(self, motion_mode: int, stamp: float) -> None:
        self.motion_mode = int(motion_mode)
        self.status_stamp = float(stamp)

    def update_safety(self, safe: bool, reason: str, stamp: float) -> None:
        self.safety_ok = bool(safe)
        self.safety_reason = str(reason)
        self.safety_stamp = float(stamp)

    def update_preflight(self, passed: bool, stamp: float) -> None:
        self.preflight_ok = bool(passed)
        self.preflight_stamp = float(stamp)

    def update_collision(
        self, *, ready: bool, trajectory_clear: bool, stamp: float
    ) -> None:
        self.collision_ready = bool(ready)
        self.trajectory_clear = bool(trajectory_clear)
        self.collision_stamp = float(stamp)

    def update_deadman(self, *, ready: bool, armed: bool, stamp: float) -> None:
        self.deadman_ready = bool(ready)
        self.deadman_armed = bool(armed)
        self.deadman_stamp = float(stamp)

    def update_world(
        self,
        *,
        ready: bool,
        ego_fresh: bool,
        scan_fresh: bool,
        tracks_fresh: bool,
        alignment_received: bool,
        stamp: float,
    ) -> None:
        self.world_ready = bool(ready)
        self.world_ego_fresh = bool(ego_fresh)
        self.world_scan_fresh = bool(scan_fresh)
        self.world_tracks_fresh = bool(tracks_fresh)
        self.world_alignment_received = bool(alignment_received)
        self.world_stamp = float(stamp)

    def update_drift(self, *, ready: bool, stamp: float) -> None:
        self.drift_ready = bool(ready)
        self.drift_stamp = float(stamp)

    def update_planner(
        self,
        *,
        ready: bool,
        used_fallback: bool,
        maximum_cbf_slack: float,
        maximum_allowed_cbf_slack: float,
        map_bounds_enforced: bool,
        stamp: float,
    ) -> None:
        self.planner_ready = bool(ready)
        self.planner_used_fallback = bool(used_fallback)
        self.planner_map_bounds_enforced = bool(map_bounds_enforced)
        self.planner_slack = float(maximum_cbf_slack)
        self.planner_allowed_slack = float(maximum_allowed_cbf_slack)
        self.planner_stamp = float(stamp)

    @staticmethod
    def _stale(now: float, stamp: Optional[float], timeout: float) -> bool:
        return stamp is None or now < stamp or now - stamp >= timeout

    def _stop(self, now: float, reason: str) -> VelocityCommand:
        # Every rejected cycle resets the ramp. Re-enabling can never resume at
        # a previously commanded speed or steering value.
        self._last_speed = 0.0
        self._last_angular = 0.0
        self._last_evaluation_stamp = float(now)
        return VelocityCommand.zero(reason)

    def evaluate(
        self,
        now: float,
        *,
        hardware_output_enabled: bool,
        staging_pose_verified: bool,
        platform_watchdog_verified: bool,
        operator_kill_verified: bool,
        candidate_owner_ok: bool,
        output_owner_ok: bool,
        deadman_owner_ok: bool,
    ) -> VelocityCommand:
        """Return a capped command only if every independent condition passes."""

        now = float(now)
        if not isfinite(now):
            return self._stop(0.0, "NONFINITE_TIME")
        if not hardware_output_enabled:
            return self._stop(now, "HARDWARE_OUTPUT_DISABLED")
        if not staging_pose_verified:
            return self._stop(now, "STAGING_POSE_NOT_VERIFIED")
        if not platform_watchdog_verified:
            return self._stop(now, "PLATFORM_WATCHDOG_NOT_VERIFIED")
        if not operator_kill_verified:
            return self._stop(now, "OPERATOR_KILL_NOT_VERIFIED")
        if not output_owner_ok:
            return self._stop(now, "CMD_VEL_OWNER_MISMATCH")
        if not candidate_owner_ok:
            return self._stop(now, "CANDIDATE_OWNER_MISMATCH")
        if not deadman_owner_ok:
            return self._stop(now, "DEADMAN_OWNER_MISMATCH")

        if self._stale(now, self.preflight_stamp, self.config.preflight_timeout):
            return self._stop(now, "STALE_PREFLIGHT")
        if not self.preflight_ok:
            return self._stop(now, "PREFLIGHT_FAILED")
        if self._stale(now, self.safety_stamp, self.config.safety_status_timeout):
            return self._stop(now, "STALE_SAFETY_STATUS")
        if not self.safety_ok:
            reason = self.safety_reason.strip().upper() or "UNSAFE"
            return self._stop(now, f"SAFETY_{reason}")
        if self._stale(now, self.collision_stamp, self.config.collision_timeout):
            return self._stop(now, "STALE_COLLISION_STATUS")
        if not self.collision_ready:
            return self._stop(now, "COLLISION_MONITOR_NOT_READY")
        if not self.trajectory_clear:
            return self._stop(now, "TRAJECTORY_BLOCKED")
        if self._stale(now, self.deadman_stamp, self.config.deadman_timeout):
            return self._stop(now, "STALE_DEADMAN")
        if not self.deadman_ready:
            return self._stop(now, "DEADMAN_NOT_READY")
        if not self.deadman_armed:
            return self._stop(now, "DEADMAN_RELEASED")

        if self._stale(now, self.world_stamp, self.config.world_timeout):
            return self._stop(now, "STALE_WORLD_STATUS")
        if not self.world_ready:
            return self._stop(now, "WORLD_NOT_READY")
        if not self.world_ego_fresh:
            return self._stop(now, "WORLD_EGO_STALE")
        if not self.world_scan_fresh:
            return self._stop(now, "WORLD_SCAN_STALE")
        if not self.world_tracks_fresh:
            return self._stop(now, "WORLD_TRACKS_STALE")
        if not self.world_alignment_received:
            return self._stop(now, "WORLD_ALIGNMENT_MISSING")
        if self._stale(now, self.drift_stamp, self.config.drift_timeout):
            return self._stop(now, "STALE_DRIFT_STATUS")
        if not self.drift_ready:
            return self._stop(now, "DRIFT_NOT_READY")
        if self._stale(now, self.planner_stamp, self.config.planner_status_timeout):
            return self._stop(now, "STALE_PLANNER_STATUS")
        if not self.planner_ready:
            return self._stop(now, "PLANNER_NOT_READY")
        if self.planner_used_fallback:
            return self._stop(now, "MPC_FALLBACK")
        if not self.planner_map_bounds_enforced:
            return self._stop(now, "MPC_MAP_BOUNDS_DISABLED")
        if not all(
            isfinite(value)
            for value in (self.planner_slack, self.planner_allowed_slack)
        ):
            return self._stop(now, "NONFINITE_CBF_SLACK")
        if self.planner_allowed_slack < 0.0 or self.planner_slack < 0.0:
            return self._stop(now, "INVALID_CBF_SLACK")
        if self.planner_slack > self.planner_allowed_slack:
            return self._stop(now, "CBF_SLACK_EXCEEDED")

        for stamp, timeout, reason in (
            (self.candidate_stamp, self.config.candidate_timeout, "STALE_CANDIDATE"),
            (self.odom_stamp, self.config.odom_timeout, "STALE_ODOM"),
            (self.scan_stamp, self.config.scan_timeout, "STALE_SCAN"),
            (self.status_stamp, self.config.status_timeout, "STALE_STATUS"),
        ):
            if self._stale(now, stamp, timeout):
                return self._stop(now, reason)
        if self.motion_mode != self.config.required_motion_mode:
            return self._stop(now, "MODE_MISMATCH")
        if not self.candidate.valid:
            return self._stop(now, self.candidate.reason)
        if not all(
            isfinite(value)
            for value in (self.candidate.linear_x, self.candidate.angular_z)
        ):
            return self._stop(now, "NONFINITE_CANDIDATE")

        target_speed = float(
            np.clip(self.candidate.linear_x, 0.0, self.config.maximum_speed)
        )
        target_angular = float(
            np.clip(
                self.candidate.angular_z,
                -self.config.maximum_ackermann_angular_command,
                self.config.maximum_ackermann_angular_command,
            )
        )
        if target_speed <= 1.0e-9:
            return self._stop(now, "ZERO_SPEED")

        if self._last_evaluation_stamp is None or now < self._last_evaluation_stamp:
            return self._stop(now, "CLOCK_ROLLBACK")
        nominal_dt = 1.0 / self.config.publish_rate
        dt = min(max(now - self._last_evaluation_stamp, 0.0), 2.0 * nominal_dt)
        speed = float(
            np.clip(
                target_speed,
                max(0.0, self._last_speed - self.config.maximum_acceleration * dt),
                self._last_speed + self.config.maximum_acceleration * dt,
            )
        )
        angular = float(
            np.clip(
                target_angular,
                self._last_angular - self.config.maximum_ackermann_angular_slew * dt,
                self._last_angular + self.config.maximum_ackermann_angular_slew * dt,
            )
        )
        self._last_speed = speed
        self._last_angular = angular
        self._last_evaluation_stamp = now
        return VelocityCommand(speed, angular, True, "ok")
