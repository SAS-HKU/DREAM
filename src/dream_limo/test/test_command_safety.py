import math

import numpy as np

from dream_limo.core.command_adapter import (
    ACKERMANN,
    FOUR_DIFF,
    CommandAdapter,
    gate_mpc_output,
    SafetySupervisorCore,
    VelocityCommand,
    center_steer_to_limo_firmware,
)
from dream_limo.limo_scale import default_deployment_config


def test_ackermann_conversion_matches_current_driver_geometry():
    assert center_steer_to_limo_firmware(0.0) == 0.0
    assert math.isclose(center_steer_to_limo_firmware(0.1), 0.0422983329, rel_tol=1e-8)
    assert math.isclose(center_steer_to_limo_firmware(0.2), 0.0884698706, rel_tol=1e-8)
    assert math.isclose(center_steer_to_limo_firmware(0.3), 0.1387512524, rel_tol=1e-8)
    assert abs(center_steer_to_limo_firmware(1.0)) <= 0.197850203
    assert center_steer_to_limo_firmware(-0.2) == -center_steer_to_limo_firmware(0.2)


def test_adapter_fails_closed_on_mode_and_nonfinite_input():
    config = default_deployment_config()
    adapter = CommandAdapter(config.safety)
    assert not adapter.adapt(target_speed=0.2, center_steer=0.1, motion_mode=None).valid
    assert not adapter.adapt(target_speed=0.2, center_steer=0.1, motion_mode=FOUR_DIFF).valid
    transition = adapter.adapt(target_speed=0.2, center_steer=0.1, motion_mode=ACKERMANN)
    assert not transition.valid and transition.reason == "MODE_CHANGED"
    command = adapter.adapt(target_speed=0.2, center_steer=0.1, motion_mode=ACKERMANN)
    assert command.valid
    assert command.linear_x <= config.safety.maximum_acceleration * 0.05 + 1e-12
    stop = adapter.adapt(target_speed=0.0, center_steer=0.0, motion_mode=ACKERMANN)
    assert not stop.valid and stop.reason == "PLANNER_STOP"
    assert (stop.linear_x, stop.angular_z) == (0.0, 0.0)
    assert not adapter.adapt(
        target_speed=float("nan"), center_steer=0.0, motion_mode=ACKERMANN
    ).valid


def test_planner_gate_zeroes_fallback_invalid_and_excessive_slack():
    common = {
        "target_speed": 0.4,
        "acceleration": -0.2,
        "steering": 0.1,
        "command_valid": True,
        "used_fallback": False,
        "maximum_cbf_slack": 0.01,
        "maximum_allowed_cbf_slack": 0.05,
    }
    accepted = gate_mpc_output(**common)
    assert accepted.valid and accepted.target_speed == 0.4

    for override, reason in (
        ({"used_fallback": True}, "MPC_FALLBACK"),
        ({"command_valid": False}, "MPC_INVALID"),
        ({"maximum_cbf_slack": float("inf")}, "MPC_NONFINITE_CBF_SLACK"),
        ({"maximum_cbf_slack": 0.051}, "MPC_CBF_SLACK_EXCEEDED"),
        ({"target_speed": float("nan")}, "MPC_NONFINITE_CONTROL"),
    ):
        rejected = gate_mpc_output(**(common | override))
        assert not rejected.valid and rejected.reason == reason
        assert (rejected.target_speed, rejected.acceleration, rejected.steering) == (
            0.0,
            0.0,
            0.0,
        )


def test_supervisor_watchdogs_latches_and_countdown():
    config = default_deployment_config().safety
    core = SafetySupervisorCore(config)
    core.update_candidate(VelocityCommand(0.1, 0.0), 0.0)
    core.update_odom(0.0)
    core.update_scan(np.asarray([1.0]), 0.0)
    assert core.front_minimum_range == 1.0
    core.update_status(ACKERMANN, 0.0)
    core.set_armed(True, 0.0)
    assert core.evaluate(0.5).reason == "COUNTDOWN"
    core.set_armed(True, 0.7)
    assert core.evaluate(0.7).reason == "COUNTDOWN"
    # Refresh sources at the end of the countdown.
    core.update_candidate(VelocityCommand(0.1, 0.0), 3.0)
    core.update_odom(3.0)
    core.update_scan(np.asarray([1.0]), 3.0)
    core.update_status(ACKERMANN, 3.0)
    core.set_armed(True, 3.0)
    assert core.evaluate(3.0).valid
    assert core.evaluate(3.5).reason == "STALE_PLANNER"
    core.update_candidate(VelocityCommand.zero("RESET"), 3.6)
    core.update_odom(3.6)
    core.update_status(ACKERMANN, 3.6)
    core.update_scan(np.asarray([0.2]), 3.6)
    assert core.evaluate(3.6).reason == "OBSTACLE_STOP_LATCHED"
    assert core.request_reset()
    core.set_external_stop(True)
    assert core.evaluate(3.6).reason == "EXTERNAL_STOP_LATCHED"


def test_supervisor_requires_arm_heartbeat_and_restarts_countdown_after_lapse():
    config = default_deployment_config().safety
    core = SafetySupervisorCore(config)
    core.update_candidate(VelocityCommand(0.1, 0.0), 0.0)
    core.update_odom(0.0)
    core.update_scan(np.asarray([1.0]), 0.0)
    core.update_status(ACKERMANN, 0.0)
    core.set_armed(True, 0.0)
    assert core.evaluate(config.arm_heartbeat_timeout).reason == "STALE_ARM_HEARTBEAT"
    assert core.armed_since is None
    assert core.evaluate(config.arm_heartbeat_timeout + 0.01).reason == (
        "STALE_ARM_HEARTBEAT"
    )

    rearm_time = config.arm_heartbeat_timeout + 0.01
    core.set_armed(True, rearm_time)
    core.update_candidate(VelocityCommand(0.1, 0.0), rearm_time)
    core.update_odom(rearm_time)
    core.update_scan(np.asarray([1.0]), rearm_time)
    core.update_status(ACKERMANN, rearm_time)
    assert core.evaluate(rearm_time).reason == "COUNTDOWN"


def test_supervisor_applies_independent_raw_ackermann_cap():
    config = default_deployment_config().safety
    core = SafetySupervisorCore(config)
    now = config.countdown_seconds
    core.update_candidate(VelocityCommand(0.1, 0.9), now)
    core.update_odom(now)
    core.update_scan(np.asarray([1.0]), now)
    core.update_status(ACKERMANN, now)
    core.set_armed(True, 0.0)
    core.set_armed(True, now)
    command = core.evaluate(now)
    assert command.valid
    assert command.angular_z == config.maximum_ackermann_angular_command
    assert command.angular_z < config.maximum_yaw_rate


def test_laserscan_positive_infinity_is_clear_when_range_max_is_known():
    config = default_deployment_config().safety
    core = SafetySupervisorCore(config)
    core.update_scan(np.asarray([np.inf, np.inf]), 1.0, range_max=6.0)
    assert core.front_minimum_range == 6.0
    assert not core.obstacle_latched

    missing = SafetySupervisorCore(config)
    missing.update_scan(np.asarray([np.nan]), 1.0, range_max=6.0)
    assert missing.obstacle_latched
