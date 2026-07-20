from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


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
    assert 'choices=["goal", "joystick"]' in launch
    assert '"expected_cmd_vel_owner": "dream_hardware_command_gate"' in launch
    assert '"expected_arm_owner": expected_arm_owner' in launch
    assert '"expected_deadman_owner": expected_arm_owner' in launch
    assert '"require_mission_goal": goal_mode' in launch
    assert '"enforce_map_bounds": "true"' in launch
    assert '"latch_perceived_occlusion": "true"' in launch
    assert '"target_speed",\n                default_value="0.05"' in launch
    assert '"target_speed": target_speed' in launch
    assert '"maximum_speed": ParameterValue(' in launch
    assert "target_speed, value_type=float" in launch
    assert 'executable="dream_merger_odometry_adapter"' in launch
    assert 'condition=IfCondition(use_merger_odom)' in launch
    assert 'DeclareLaunchArgument("merger_alignment_verified", default_value="false")' in launch
    assert "condition=IfCondition(goal_mode)" in launch
    assert "condition=IfCondition(joystick_mode)" in launch
    assert "condition=IfCondition(start_joy_in_joystick_mode)" in launch


def test_hardware_preflight_owner_arguments_reach_the_preflight_node():
    dry_run = (ROOT / "launch" / "dream_limo_dry_run.launch.py").read_text()
    sensor = (ROOT / "launch" / "dream_sensor_smoke.launch.py").read_text()
    live = (ROOT / "launch" / "dream_live_demo.launch.py").read_text()
    for content in (dry_run, sensor, live):
        assert "expected_cmd_vel_owner" in content
        assert "expected_arm_owner" in content
        assert "target_speed" in content
        assert "require_mission_goal" in content


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
