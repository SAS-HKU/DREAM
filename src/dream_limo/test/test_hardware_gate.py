import math

import pytest

from dream_limo.core.command_adapter import ACKERMANN, FOUR_DIFF, VelocityCommand
from dream_limo.core.hardware_gate import (
    HardwareCommandGateCore,
    HardwareGateConfig,
    exact_publisher_owner,
)


def _prime(core: HardwareCommandGateCore, now: float, *, speed: float = 0.50) -> None:
    core.update_candidate(VelocityCommand(speed, 0.50, True, "ok"), now)
    core.update_odom(now)
    core.update_scan(now)
    core.update_status(ACKERMANN, now)
    core.update_safety(True, "ok", now)
    core.update_preflight(True, now)
    core.update_collision(ready=True, trajectory_clear=True, stamp=now)
    core.update_deadman(ready=True, armed=True, stamp=now)
    core.update_world(
        ready=True,
        ego_fresh=True,
        scan_fresh=True,
        tracks_fresh=True,
        alignment_received=True,
        stamp=now,
    )
    core.update_drift(ready=True, stamp=now)
    core.update_planner(
        ready=True,
        used_fallback=False,
        maximum_cbf_slack=0.01,
        maximum_allowed_cbf_slack=0.05,
        map_bounds_enforced=True,
        stamp=now,
    )


def _evaluate(
    core: HardwareCommandGateCore,
    now: float,
    *,
    enabled: bool = True,
    staging: bool = True,
    platform_watchdog: bool = True,
    operator_kill: bool = True,
    candidate_owner: bool = True,
    output_owner: bool = True,
    deadman_owner: bool = True,
):
    return core.evaluate(
        now,
        hardware_output_enabled=enabled,
        staging_pose_verified=staging,
        platform_watchdog_verified=platform_watchdog,
        operator_kill_verified=operator_kill,
        candidate_owner_ok=candidate_owner,
        output_owner_ok=output_owner,
        deadman_owner_ok=deadman_owner,
    )


def _complete_readiness_countdown(
    core: HardwareCommandGateCore,
    start: float,
    *,
    speed: float = 0.50,
):
    """Refresh all evidence at the gate rate until the final countdown expires."""
    first = None
    command = None
    steps = math.ceil(
        core.config.readiness_countdown_seconds * core.config.publish_rate
    )
    for step in range(steps + 1):
        now = start + step / core.config.publish_rate
        _prime(core, now, speed=speed)
        command = _evaluate(core, now)
        if first is None:
            first = command
    assert first is not None and first.reason == "READINESS_COUNTDOWN"
    assert command is not None and command.valid
    return now, command


def test_checked_in_defaults_can_never_move_without_two_explicit_assertions():
    core = HardwareCommandGateCore(HardwareGateConfig())
    _prime(core, 1.0)
    disabled = _evaluate(core, 1.0, enabled=False)
    assert not disabled.valid
    assert disabled.reason == "HARDWARE_OUTPUT_DISABLED"
    assert (disabled.linear_x, disabled.angular_z) == (0.0, 0.0)

    unstaged = _evaluate(core, 1.05, staging=False)
    assert not unstaged.valid
    assert unstaged.reason == "STAGING_POSE_NOT_VERIFIED"
    assert (unstaged.linear_x, unstaged.angular_z) == (0.0, 0.0)

    no_platform_watchdog = _evaluate(core, 1.10, platform_watchdog=False)
    assert no_platform_watchdog.reason == "PLATFORM_WATCHDOG_NOT_VERIFIED"
    no_operator_kill = _evaluate(core, 1.15, operator_kill=False)
    assert no_operator_kill.reason == "OPERATOR_KILL_NOT_VERIFIED"


@pytest.mark.parametrize(
    ("owner_kwargs", "reason"),
    (
        ({"candidate_owner": False}, "CANDIDATE_OWNER_MISMATCH"),
        ({"output_owner": False}, "CMD_VEL_OWNER_MISMATCH"),
        ({"deadman_owner": False}, "DEADMAN_OWNER_MISMATCH"),
    ),
)
def test_gate_requires_exact_single_reviewed_publishers(owner_kwargs, reason):
    core = HardwareCommandGateCore(HardwareGateConfig())
    _prime(core, 1.0)
    command = _evaluate(core, 1.0, **owner_kwargs)
    assert not command.valid and command.reason == reason


@pytest.mark.parametrize(
    ("break_condition", "reason"),
    (
        (lambda core, now: core.update_preflight(False, now), "PREFLIGHT_FAILED"),
        (
            lambda core, now: core.update_safety(False, "OBSTACLE", now),
            "SAFETY_OBSTACLE",
        ),
        (
            lambda core, now: core.update_collision(
                ready=False, trajectory_clear=True, stamp=now
            ),
            "COLLISION_MONITOR_NOT_READY",
        ),
        (
            lambda core, now: core.update_collision(
                ready=True, trajectory_clear=False, stamp=now
            ),
            "TRAJECTORY_BLOCKED",
        ),
        (
            lambda core, now: core.update_deadman(
                ready=False, armed=False, stamp=now
            ),
            "DEADMAN_NOT_READY",
        ),
        (
            lambda core, now: core.update_deadman(
                ready=True, armed=False, stamp=now
            ),
            "DEADMAN_RELEASED",
        ),
        (
            lambda core, now: core.update_world(
                ready=True,
                ego_fresh=True,
                scan_fresh=True,
                tracks_fresh=False,
                alignment_received=True,
                stamp=now,
            ),
            "WORLD_TRACKS_STALE",
        ),
        (lambda core, now: core.update_drift(ready=False, stamp=now), "DRIFT_NOT_READY"),
        (
            lambda core, now: core.update_planner(
                ready=True,
                used_fallback=True,
                maximum_cbf_slack=0.0,
                maximum_allowed_cbf_slack=0.05,
                map_bounds_enforced=True,
                stamp=now,
            ),
            "MPC_FALLBACK",
        ),
        (
            lambda core, now: core.update_planner(
                ready=True,
                used_fallback=False,
                maximum_cbf_slack=0.051,
                maximum_allowed_cbf_slack=0.05,
                map_bounds_enforced=True,
                stamp=now,
            ),
            "CBF_SLACK_EXCEEDED",
        ),
        (
            lambda core, now: core.update_planner(
                ready=True,
                used_fallback=False,
                maximum_cbf_slack=0.0,
                maximum_allowed_cbf_slack=0.05,
                map_bounds_enforced=False,
                stamp=now,
            ),
            "MPC_MAP_BOUNDS_DISABLED",
        ),
        (lambda core, now: core.update_status(FOUR_DIFF, now), "MODE_MISMATCH"),
    ),
)
def test_every_continuous_readiness_condition_fails_closed(break_condition, reason):
    core = HardwareCommandGateCore(HardwareGateConfig())
    _prime(core, 0.50)
    assert _evaluate(core, 0.50).reason == "READINESS_COUNTDOWN"
    assert core.readiness_countdown_started_at == pytest.approx(0.50)
    _prime(core, 1.0)
    break_condition(core, 1.0)
    command = _evaluate(core, 1.0)
    assert not command.valid and command.reason == reason
    assert command.linear_x == 0.0 and command.angular_z == 0.0
    assert core.readiness_countdown_started_at is None


def test_fresh_valid_command_is_independently_capped_and_slew_limited():
    config = HardwareGateConfig()
    core = HardwareCommandGateCore(config)
    move_time, command = _complete_readiness_countdown(core, 1.0, speed=9.0)
    assert math.isclose(command.linear_x, config.maximum_acceleration * 0.05)
    assert math.isclose(command.angular_z, config.maximum_ackermann_angular_slew * 0.05)
    assert command.linear_x < config.maximum_speed
    assert command.angular_z < config.maximum_ackermann_angular_command

    # Refresh all upstream evidence and confirm another bounded step.
    second_time = move_time + 0.05
    _prime(core, second_time, speed=9.0)
    second = _evaluate(core, second_time)
    assert second.valid
    assert math.isclose(
        second.linear_x - command.linear_x,
        config.maximum_acceleration * 0.05,
    )
    assert math.isclose(
        second.angular_z - command.angular_z,
        config.maximum_ackermann_angular_slew * 0.05,
    )


def test_watchdog_or_deadman_release_immediately_zeroes_and_resets_ramp():
    config = HardwareGateConfig()
    core = HardwareCommandGateCore(config)
    move_time, moving = _complete_readiness_countdown(core, 1.0)
    assert moving.valid and moving.linear_x > 0.0

    stop_time = move_time + 0.01
    core.update_deadman(ready=True, armed=False, stamp=stop_time)
    stopped = _evaluate(core, stop_time)
    assert not stopped.valid and stopped.reason == "DEADMAN_RELEASED"
    assert (stopped.linear_x, stopped.angular_z) == (0.0, 0.0)
    assert core.readiness_countdown_started_at is None

    restart_time = stop_time + 0.05
    _prime(core, restart_time)
    waiting = _evaluate(core, restart_time)
    assert not waiting.valid and waiting.reason == "READINESS_COUNTDOWN"
    _, resumed = _complete_readiness_countdown(core, restart_time)
    assert resumed.valid
    assert resumed.linear_x <= config.maximum_acceleration * 0.05 + 1.0e-12


def test_stale_collision_status_stops_even_when_all_other_inputs_refresh():
    config = HardwareGateConfig()
    core = HardwareCommandGateCore(config)
    move_time, moving = _complete_readiness_countdown(core, 1.0)
    assert moving.valid

    stale_time = move_time + config.collision_timeout + 1.0e-6
    _prime(core, stale_time)
    # Restore only the old collision receipt after refreshing everything else.
    core.collision_stamp = move_time
    command = _evaluate(core, stale_time)
    assert not command.valid and command.reason == "STALE_COLLISION_STATUS"
    assert core.readiness_countdown_started_at is None


def test_nonfinite_and_zero_candidates_are_rejected():
    core = HardwareCommandGateCore(HardwareGateConfig())
    _prime(core, 1.0)
    core.update_candidate(VelocityCommand(float("nan"), 0.0, True, "ok"), 1.0)
    assert _evaluate(core, 1.0).reason == "NONFINITE_CANDIDATE"
    _prime(core, 1.05, speed=0.0)
    assert _evaluate(core, 1.05).reason == "ZERO_SPEED"
    assert core.readiness_countdown_started_at is None


@pytest.mark.parametrize(
    ("break_condition", "reason"),
    (
        (
            lambda core, now: core.update_collision(
                ready=True, trajectory_clear=False, stamp=now
            ),
            "TRAJECTORY_BLOCKED",
        ),
        (
            lambda core, now: core.update_candidate(
                VelocityCommand(0.0, 0.0, True, "ok"), now
            ),
            "ZERO_SPEED",
        ),
    ),
)
def test_collision_or_zero_candidate_restarts_full_readiness_countdown(
    break_condition, reason
):
    config = HardwareGateConfig()
    core = HardwareCommandGateCore(config)
    _prime(core, 1.0)
    assert _evaluate(core, 1.0).reason == "READINESS_COUNTDOWN"

    interrupted_at = 2.0
    _prime(core, interrupted_at)
    break_condition(core, interrupted_at)
    interrupted = _evaluate(core, interrupted_at)
    assert interrupted.reason == reason
    assert core.readiness_countdown_started_at is None
    assert core.readiness_countdown_remaining(interrupted_at) == pytest.approx(3.0)

    restarted_at = 2.05
    _prime(core, restarted_at)
    assert _evaluate(core, restarted_at).reason == "READINESS_COUNTDOWN"
    assert core.readiness_countdown_started_at == pytest.approx(restarted_at)

    almost_ready = restarted_at + config.readiness_countdown_seconds - 0.01
    _prime(core, almost_ready)
    assert _evaluate(core, almost_ready).reason == "READINESS_COUNTDOWN"
    assert core.readiness_countdown_started_at == pytest.approx(restarted_at)

    ready_at = restarted_at + config.readiness_countdown_seconds
    _prime(core, ready_at)
    assert _evaluate(core, ready_at).valid


def test_repeated_countdown_cycles_preserve_start_and_report_remaining_time():
    core = HardwareCommandGateCore(HardwareGateConfig())
    for now, remaining in ((1.0, 3.0), (1.05, 2.95), (2.0, 2.0), (3.95, 0.05)):
        _prime(core, now)
        command = _evaluate(core, now)
        assert command.reason == "READINESS_COUNTDOWN"
        assert core.readiness_countdown_started_at == pytest.approx(1.0)
        assert core.readiness_countdown_remaining(now) == pytest.approx(remaining)


def test_hardware_config_rejects_weakened_caps_and_timeouts():
    with pytest.raises(ValueError):
        HardwareGateConfig(maximum_speed=0.151)
    with pytest.raises(ValueError):
        HardwareGateConfig(maximum_acceleration=0.351)
    with pytest.raises(ValueError):
        HardwareGateConfig(candidate_timeout=0.51)
    with pytest.raises(ValueError):
        HardwareGateConfig(readiness_countdown_seconds=2.999)
    with pytest.raises(ValueError):
        HardwareGateConfig(required_motion_mode=FOUR_DIFF)


def test_preflight_owner_helper_requires_exact_single_owner():
    assert exact_publisher_owner(["dream_hardware_command_gate"], "dream_hardware_command_gate")
    assert not exact_publisher_owner([], "dream_hardware_command_gate")
    assert not exact_publisher_owner(
        ["dream_hardware_command_gate", "teleop_twist"],
        "dream_hardware_command_gate",
    )
    assert not exact_publisher_owner(["dream_hardware_command_gate"], "")
