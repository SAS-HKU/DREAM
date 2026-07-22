import importlib.util
from pathlib import Path

import pytest
import yaml


def _package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _free_navigation_launch_module():
    path = _package_root() / "launch" / "dream_free_navigation.launch.py"
    spec = importlib.util.spec_from_file_location("dream_free_navigation", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_free_navigation_launch_has_one_controller_chain_and_safe_default():
    source = (
        _package_root() / "launch" / "dream_free_navigation.launch.py"
    ).read_text(encoding="utf-8")
    assert 'default_value="false"' in source
    assert 'executable="dream_free_planner"' in source
    assert 'executable="dream_hardware_command_gate"' in source
    assert 'executable="planner_server"' in source
    assert 'executable="lifecycle_manager"' in source
    assert 'executable="controller_server"' not in source
    assert 'executable="bt_navigator"' not in source
    assert 'executable="velocity_smoother"' not in source
    assert 'executable="sfg_planner"' not in source
    assert 'executable="lidar_pedestrian_detector"' not in source
    assert "0.03 < speed <= 0.20" in source
    assert '"use_latest_tf": True' in source
    assert '"footprint_clearance": 0.0' in source
    assert '"verified_start_clearance_enabled": ParameterValue(' in source
    assert 'staging_pose_verified, value_type=bool' in source
    assert '"verified_start_clearance_radius": 0.30' in source


@pytest.mark.parametrize("value", ["0.030001", "0.15", "0.20"])
def test_free_navigation_accepts_reviewed_speed_range(value):
    assert _free_navigation_launch_module()._validated_target_speed(value) == float(
        value
    )


@pytest.mark.parametrize(
    "value", ["0.03", "0", "-0.1", "0.200001", "nan", "inf", "fast"]
)
def test_free_navigation_rejects_speed_outside_reviewed_range(value):
    with pytest.raises(RuntimeError, match="target_speed"):
        _free_navigation_launch_module()._validated_target_speed(value)


def test_nav2_costmap_and_drift_use_the_same_fixed_world_bounds():
    root = _package_root()
    deployment = yaml.safe_load(
        (root / "config" / "free_navigation.yaml").read_text(encoding="utf-8")
    )
    nav2 = yaml.safe_load(
        (root / "config" / "nav2_dream_planner.yaml").read_text(
            encoding="utf-8"
        )
    )
    grid = deployment["grid"]
    costmap = nav2["global_costmap"]["global_costmap"]["ros__parameters"]
    assert costmap["global_frame"] == deployment["frame_id"] == "map"
    assert costmap["rolling_window"] is False
    assert costmap["origin_x"] == grid["x_min"]
    assert costmap["origin_y"] == grid["y_min"]
    assert costmap["width"] == grid["x_max"] - grid["x_min"]
    assert costmap["height"] == grid["y_max"] - grid["y_min"]
    assert costmap["resolution"] == grid["resolution"]
    assert costmap["track_unknown_space"] is True
    assert costmap["footprint"] == (
        "[[0.16, 0.11], [0.16, -0.11], [-0.16, -0.11], "
        "[-0.16, 0.11]]"
    )
    assert costmap["footprint_padding"] == 0.05
    obstacle = costmap["obstacle_layer"]
    assert obstacle["combination_method"] == 0
    assert obstacle["scan"]["max_obstacle_height"] > 0.18
    planner = nav2["planner_server"]["ros__parameters"]["GridBased"]
    assert planner["plugin"] == "nav2_smac_planner/SmacPlannerHybrid"
    assert planner["motion_model_for_search"] == "DUBIN"
    assert planner["minimum_turning_radius"] == 0.40
    assert planner["allow_unknown"] is False
    assert costmap["inflation_layer"]["inflation_radius"] >= 0.30


def test_free_navigation_entry_points_and_rviz_goal_contract_are_installed():
    root = _package_root()
    setup = (root / "setup.py").read_text(encoding="utf-8")
    for executable in (
        "dream_free_goal_authorizer",
        "dream_nav2_path_provider",
        "dream_free_planner",
    ):
        assert executable in setup
    rviz = (
        root / "rviz" / "dream_free_navigation.rviz"
    ).read_text(encoding="utf-8")
    assert "Topic: /goal_pose" in rviz
    assert "Value: /global_costmap/costmap" in rviz
    assert "Value: /dream/geometric_path" in rviz
    assert "Value: /dream/reference_trajectory" in rviz
    assert "Value: /global_costmap/published_footprint" in rviz


def test_free_navigation_never_imports_or_edits_sfg_planning_code():
    root = _package_root() / "dream_limo"
    sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            root / "free_goal_authorizer_node.py",
            root / "nav2_path_provider_node.py",
            root / "free_planner_node.py",
        )
    )
    assert "import sfg_nav" not in sources
    assert "from sfg_nav" not in sources
    assert "/cmd_vel" not in (
        root / "nav2_path_provider_node.py"
    ).read_text(encoding="utf-8")


def test_free_planner_allows_only_known_soft_centers_with_footprint_checks():
    source = (
        _package_root() / "dream_limo" / "free_planner_node.py"
    ).read_text(encoding="utf-8")
    assert source.count("allow_known_soft_center=True") == 2
    assert source.count("validate_swept_trajectory(") == 2
