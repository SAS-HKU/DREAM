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
