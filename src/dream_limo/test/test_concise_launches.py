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
    assert '"use_merger_odom": "false"' in launch
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
    assert 'executable="dream_hardware_deadman"' in launch
    assert 'executable="dream_hardware_command_gate"' in launch
    assert 'DeclareLaunchArgument("enable_physical_motion", default_value="false")' in launch
    assert 'DeclareLaunchArgument("staging_pose_verified", default_value="false")' in launch
    assert 'DeclareLaunchArgument("platform_watchdog_verified", default_value="false")' in launch
    assert 'DeclareLaunchArgument("operator_kill_verified", default_value="false")' in launch
    assert 'DeclareLaunchArgument("deadman_device_verified", default_value="false")' in launch
    assert 'DeclareLaunchArgument("start_joy", default_value="false")' in launch
    assert '"expected_cmd_vel_owner": "dream_hardware_command_gate"' in launch
    assert '"expected_arm_owner": "dream_hardware_deadman"' in launch
    assert '"enforce_map_bounds": "true"' in launch
    assert '"latch_perceived_occlusion": "true"' in launch


def test_hardware_preflight_owner_arguments_reach_the_preflight_node():
    dry_run = (ROOT / "launch" / "dream_limo_dry_run.launch.py").read_text()
    sensor = (ROOT / "launch" / "dream_sensor_smoke.launch.py").read_text()
    live = (ROOT / "launch" / "dream_live_demo.launch.py").read_text()
    for content in (dry_run, sensor, live):
        assert "expected_cmd_vel_owner" in content
        assert "expected_arm_owner" in content
