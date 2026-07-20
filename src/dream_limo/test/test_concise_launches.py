import importlib.util
from pathlib import Path

import pytest
from launch import LaunchContext
from launch.actions import IncludeLaunchDescription
from launch.utilities import (
    normalize_to_list_of_substitutions,
    perform_substitutions,
)
from launch_ros.actions import Node


ROOT = Path(__file__).resolve().parents[1]


def _hardware_launch_module():
    path = ROOT / "launch" / "dream_hardware_motion.launch.py"
    spec = importlib.util.spec_from_file_location("dream_hardware_motion", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _perform(context, value):
    return perform_substitutions(
        context, normalize_to_list_of_substitutions(value)
    )


def test_live_demo_has_one_class_neutral_track_pipeline():
    launch = (ROOT / "launch" / "dream_live_demo.launch.py").read_text()
    assert 'executable="lidar_cluster_buffer"' in launch
    assert 'executable="dream_vehicle_tracker"' in launch
    assert "pedestrian_lidar_detector" not in launch
    assert "multi_object_tracker" not in launch
    assert "sfg_perception.launch.py" not in launch
    assert "sfg_full_stack.launch.py" not in launch
    assert 'DeclareLaunchArgument("use_merger_odom", default_value="false")' in launch
    assert '"use_merger_odom": use_merger_odom' in launch
    assert "UnlessCondition(use_merger_odom)" in launch
    assert "/cmd_vel" not in launch

    compatibility = (
        ROOT / "launch" / "dream_with_sfg_perception.launch.py"
    ).read_text()
    assert "dream_live_demo.launch.py" in compatibility
    assert 'executable="pedestrian_lidar_detector"' not in compatibility


def test_motion_demo_routes_model_to_existing_closed_loop_smoke():
    launch = (ROOT / "launch" / "dream_motion_demo.launch.py").read_text()
    assert "dream_rviz_smoke.launch.py" in launch
    assert '"preset": model' in launch
    assert "/cmd_vel" not in launch


def test_hardware_launch_is_explicit_and_disabled_by_default():
    launch = (ROOT / "launch" / "dream_hardware_motion.launch.py").read_text()
    config = (ROOT / "config" / "dream_limo.yaml").read_text()
    assert "dream_live_demo.launch.py" in launch
    assert 'executable="dream_collision_monitor"' in launch
    assert 'executable="dream_goal_authorizer"' in launch
    assert 'executable="dream_hardware_deadman"' in launch
    assert 'executable="dream_hardware_command_gate"' in launch
    assert 'DeclareLaunchArgument("enable_physical_motion", default_value="false")' in launch
    assert 'DeclareLaunchArgument("staging_pose_verified", default_value="false")' in launch
    assert 'DeclareLaunchArgument("platform_watchdog_verified", default_value="false")' in launch
    assert 'DeclareLaunchArgument("operator_kill_verified", default_value="false")' in launch
    assert 'DeclareLaunchArgument("deadman_device_verified", default_value="false")' in launch
    assert 'DeclareLaunchArgument("start_joy", default_value="false")' in launch
    assert '"activation_mode",\n                default_value="goal"' in launch
    assert 'choices=["goal", "auto_forward", "joystick"]' in launch
    assert '"expected_cmd_vel_owner": "dream_hardware_command_gate"' in launch
    assert '"expected_arm_owner": expected_arm_owner' in launch
    assert '"expected_deadman_owner": expected_arm_owner' in launch
    assert '"require_mission_goal": authorizer_mode' in launch
    assert '"enforce_map_bounds": "true"' in launch
    assert '"latch_perceived_occlusion": "true"' in launch
    assert '"target_speed",\n                default_value="0.05"' in launch
    assert '"target_speed": target_speed' in launch
    assert '"maximum_speed": ParameterValue(' in launch
    assert "target_speed, value_type=float" in launch
    assert 'executable="dream_merger_odometry_adapter"' in launch
    assert 'condition=IfCondition(use_merger_odom)' in launch
    assert 'DeclareLaunchArgument("merger_alignment_verified", default_value="false")' in launch
    assert "condition=IfCondition(authorizer_mode)" in launch
    assert '"auto_start": ParameterValue(' in launch
    assert "auto_forward_mode, value_type=bool" in launch
    assert "condition=IfCondition(joystick_mode)" in launch
    assert "condition=IfCondition(start_joy_in_joystick_mode)" in launch
    assert "OpaqueFunction(function=_validate_target_speed)" in launch
    assert "auto_start: false" in config


@pytest.mark.parametrize(
    "value",
    ["0.030000", "0.0", "-0.05", "0.150001", "0.35", "nan", "inf", "fast"],
)
def test_hardware_launch_rejects_unsupported_target_speed(value):
    with pytest.raises(RuntimeError, match="target_speed"):
        _hardware_launch_module()._validated_target_speed(value)


@pytest.mark.parametrize("value", ["0.030001", "0.05", "0.10", "0.15"])
def test_hardware_launch_accepts_commissioned_target_speed(value):
    assert _hardware_launch_module()._validated_target_speed(value) == float(value)


@pytest.mark.parametrize(
    (
        "mode",
        "authorizer_enabled",
        "deadman_enabled",
        "expected_owner",
        "mission_goal_required",
    ),
    (
        ("goal", True, False, "dream_goal_authorizer", "True"),
        ("auto_forward", True, False, "dream_goal_authorizer", "True"),
        ("joystick", False, True, "dream_hardware_deadman", "False"),
    ),
)
def test_hardware_activation_modes_resolve_to_one_owner_and_goal_contract(
    mode,
    authorizer_enabled,
    deadman_enabled,
    expected_owner,
    mission_goal_required,
):
    description = _hardware_launch_module().generate_launch_description()
    nodes = {
        action.node_executable: action
        for action in description.entities
        if isinstance(action, Node)
    }
    include = next(
        action
        for action in description.entities
        if isinstance(action, IncludeLaunchDescription)
    )
    arguments = dict(include.launch_arguments)
    context = LaunchContext()
    context.launch_configurations["activation_mode"] = mode
    context.launch_configurations["start_joy"] = "false"

    assert (
        nodes["dream_goal_authorizer"].condition.evaluate(context)
        is authorizer_enabled
    )
    assert (
        nodes["dream_hardware_deadman"].condition.evaluate(context)
        is deadman_enabled
    )
    assert _perform(context, arguments["expected_arm_owner"]) == expected_owner
    assert (
        _perform(context, arguments["require_mission_goal"])
        == mission_goal_required
    )


def test_hardware_preflight_owner_arguments_reach_the_preflight_node():
    dry_run = (ROOT / "launch" / "dream_limo_dry_run.launch.py").read_text()
    sensor = (ROOT / "launch" / "dream_sensor_smoke.launch.py").read_text()
    live = (ROOT / "launch" / "dream_live_demo.launch.py").read_text()
    for content in (dry_run, sensor, live):
        assert "expected_cmd_vel_owner" in content
        assert "expected_arm_owner" in content
        assert "target_speed" in content
        assert "require_mission_goal" in content


def test_hardware_gate_reports_stop_reason_transitions():
    source = (ROOT / "dream_limo" / "hardware_command_gate_node.py").read_text()
    assert "_last_reported_gate_reason" in source
    assert "Physical command gate holding zero:" in source
    assert "Physical command gate READY" in source


def test_camera_evidence_gate_is_threaded_and_disabled_only_for_motion():
    sensor = (ROOT / "launch" / "dream_sensor_smoke.launch.py").read_text()
    live = (ROOT / "launch" / "dream_live_demo.launch.py").read_text()
    hardware = (ROOT / "launch" / "dream_hardware_motion.launch.py").read_text()

    for content in (sensor, live):
        assert (
            'DeclareLaunchArgument("require_camera_evidence", default_value="true")'
            in content
        )
        assert '"require_camera_evidence": require_camera_evidence' in content

    assert 'executable="dream_camera_evidence"' in sensor
    assert '"rviz": rviz' in live
    assert '"require_camera_evidence": "false"' in hardware


def test_live_rviz_exposes_map_goal_tool():
    rviz = (ROOT / "rviz" / "dream_sensor.rviz").read_text()
    assert "rviz_default_plugins/SetGoal" in rviz
    assert "Topic: /goal_pose" in rviz


def test_aligned_merger_live_launch_is_stationary_and_fail_closed():
    launch = (ROOT / "launch" / "dream_live_merger_odom.launch.py").read_text()
    assert "dream_live_demo.launch.py" in launch
    assert '"use_merger_odom": "true"' in launch
    assert 'executable="dream_merger_odometry_adapter"' in launch
    assert 'DeclareLaunchArgument("merger_alignment_verified", default_value="false")' in launch
    assert "/cmd_vel" not in launch


def test_planner_contains_latched_mission_complete_stop_contract():
    planner = (ROOT / "dream_limo" / "planner_node.py").read_text()
    assert "MissionEndGuard" in planner
    assert 'self._publish_stop(\n            "MISSION_COMPLETE"' in planner
    assert '"mission_complete": True' in planner
    assert "if self.mission.complete:" in planner
