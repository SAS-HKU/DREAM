import pytest

from dream_limo.core.command_adapter import VelocityCommand
from dream_limo.ros_utils import (
    ControlSourceStamp,
    stamped_twist_from_velocity_command,
    velocity_command_from_stamped_twist,
)


def test_internal_command_identity_survives_adapter_and_supervisor_hops():
    planner_stamp = ControlSourceStamp(sec=42, nanosec=123456789)
    planner_command = VelocityCommand(0.12, 0.03, True, "ok")

    adapter_output = stamped_twist_from_velocity_command(
        planner_command,
        planner_stamp,
    )
    adapter_command, adapter_stamp = velocity_command_from_stamped_twist(
        adapter_output,
        malformed_reason="MALFORMED_ADAPTER_OUTPUT",
    )
    supervisor_output = stamped_twist_from_velocity_command(
        adapter_command,
        adapter_stamp,
    )
    gate_command, gate_stamp = velocity_command_from_stamped_twist(
        supervisor_output,
        malformed_reason="MALFORMED_SUPERVISOR_OUTPUT",
    )

    assert adapter_stamp == planner_stamp
    assert gate_stamp == planner_stamp
    assert gate_command.linear_x == pytest.approx(planner_command.linear_x)
    assert gate_command.angular_z == pytest.approx(planner_command.angular_z)


def test_identical_numeric_commands_with_different_tokens_remain_distinct():
    command = VelocityCommand(0.12, 0.03, True, "ok")
    first = ControlSourceStamp(sec=42, nanosec=1)
    second = ControlSourceStamp(sec=42, nanosec=2)

    first_message = stamped_twist_from_velocity_command(command, first)
    second_message = stamped_twist_from_velocity_command(command, second)
    first_command, first_token = velocity_command_from_stamped_twist(
        first_message,
        malformed_reason="MALFORMED",
    )
    second_command, second_token = velocity_command_from_stamped_twist(
        second_message,
        malformed_reason="MALFORMED",
    )

    assert first_command == second_command
    assert first_token != second_token


@pytest.mark.parametrize(
    "payload",
    (
        None,
        {},
        {"sec": 0, "nanosec": 0},
        {"sec": 1.0, "nanosec": 0},
        {"sec": 1, "nanosec": 1_000_000_000},
        {"sec": True, "nanosec": 1},
    ),
)
def test_malformed_control_identity_payloads_are_rejected(payload):
    with pytest.raises(ValueError):
        ControlSourceStamp.from_mapping(payload)
