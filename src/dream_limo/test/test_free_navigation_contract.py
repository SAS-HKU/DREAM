import importlib.util
from pathlib import Path

import pytest
import yaml
from launch import LaunchContext
from launch_ros.actions import Node
from launch_ros.utilities import evaluate_parameters


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
    assert source.count('executable="dream_free_planner"') == 1
    assert source.count('executable="dream_metrics"') == 1
    assert '"replan_period": 1.0' in source


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


@pytest.mark.parametrize(
    ("model", "physical", "calibrated"),
    (
        ("oacp_vb", "false", "false"),
        ("oacp_vb", "true", "true"),
        ("balanced", "true", "false"),
        ("pure_mpc", "true", "false"),
        ("nominal", "true", "false"),
    ),
)
def test_free_navigation_allows_safe_oacp_and_existing_arm_combinations(
    model, physical, calibrated
):
    _free_navigation_launch_module()._validate_oacp_physical_readiness(
        model=model,
        enable_physical_motion=physical,
        thresholds_calibrated=calibrated,
        calibration_logging_only="false",
        enable_contingency="true",
        target_speed=0.15,
    )


def test_free_navigation_rejects_uncalibrated_physical_oacp_vb():
    with pytest.raises(RuntimeError, match="oacp_thresholds_calibrated"):
        _free_navigation_launch_module()._validate_oacp_physical_readiness(
            model="oacp_vb",
            enable_physical_motion="true",
            thresholds_calibrated="false",
            calibration_logging_only="false",
            enable_contingency="true",
            target_speed=0.15,
        )


def test_free_navigation_allows_explicit_noncomparison_calibration_logging():
    _free_navigation_launch_module()._validate_oacp_physical_readiness(
        model="oacp_vb",
        enable_physical_motion="true",
        thresholds_calibrated="false",
        calibration_logging_only="true",
        enable_contingency="true",
        target_speed=0.15,
    )


def test_free_navigation_caps_uncalibrated_physical_logging_speed():
    with pytest.raises(RuntimeError, match="0.15"):
        _free_navigation_launch_module()._validate_oacp_physical_readiness(
            model="oacp_vb",
            enable_physical_motion="true",
            thresholds_calibrated="false",
            calibration_logging_only="true",
            enable_contingency="true",
            target_speed=0.20,
        )


def test_free_navigation_rejects_physical_oacp_without_contingency():
    with pytest.raises(RuntimeError, match="enable_contingency"):
        _free_navigation_launch_module()._validate_oacp_physical_readiness(
            model="oacp_vb",
            enable_physical_motion="true",
            thresholds_calibrated="true",
            calibration_logging_only="false",
            enable_contingency="false",
            target_speed=0.15,
        )


def test_free_navigation_launch_validator_enforces_oacp_calibration_gate():
    module = _free_navigation_launch_module()
    context = LaunchContext()
    context.launch_configurations.update(
        {
            "model": "oacp_vb",
            "target_speed": "0.15",
            "enable_physical_motion": "true",
            "oacp_thresholds_calibrated": "false",
            "oacp_calibration_logging_only": "false",
            "enable_contingency": "true",
        }
    )
    with pytest.raises(RuntimeError, match="oacp_thresholds_calibrated"):
        module._validate_arguments(context)


@pytest.mark.parametrize(
    ("model", "drift_enabled", "oacp_enabled"),
    (
        ("balanced", True, False),
        ("pure_mpc", True, False),
        ("nominal", False, False),
        ("oacp_vb", False, True),
    ),
)
def test_free_navigation_starts_only_the_selected_external_risk_provider(
    model, drift_enabled, oacp_enabled
):
    description = _free_navigation_launch_module().generate_launch_description()
    nodes = {
        action.node_executable: action
        for action in description.entities
        if isinstance(action, Node)
    }
    context = LaunchContext()
    context.launch_configurations["model"] = model

    assert (
        nodes["dream_drift_field"].condition.evaluate(context) is drift_enabled
    )
    assert (
        nodes["dream_oacp_vb_assessor"].condition.evaluate(context)
        is oacp_enabled
    )


@pytest.mark.parametrize(
    (
        "model",
        "expected_goal_provider",
        "expected_gate_provider",
        "expected_status_topic",
    ),
    (
        ("balanced", "", "drift", "/dream/drift_status"),
        ("pure_mpc", "", "drift", "/dream/drift_status"),
        ("nominal", "", "nominal", "/dream/planner_status"),
        ("oacp_vb", "oacp_vb", "oacp_vb", "/dream/oacp_vb_status"),
    ),
)
def test_free_navigation_risk_readiness_routes_with_selected_arm(
    model,
    expected_goal_provider,
    expected_gate_provider,
    expected_status_topic,
):
    description = _free_navigation_launch_module().generate_launch_description()
    nodes = {
        action.node_executable: action
        for action in description.entities
        if isinstance(action, Node)
    }
    context = LaunchContext()
    context.launch_configurations.update(
        {
            "model": model,
            "target_speed": "0.15",
            "enable_physical_motion": "false",
            "staging_pose_verified": "false",
            "platform_watchdog_verified": "false",
            "operator_kill_verified": "false",
            "oacp_calibration_logging_only": "false",
        }
    )
    authorizer_parameters = evaluate_parameters(
        context, nodes["dream_free_goal_authorizer"]._Node__parameters
    )[-1]
    gate_parameters = evaluate_parameters(
        context, nodes["dream_hardware_command_gate"]._Node__parameters
    )[-1]

    assert (
        authorizer_parameters["required_risk_provider"]
        == expected_goal_provider
    )
    assert authorizer_parameters["shared_minimum_speed"] == 0.0
    assert authorizer_parameters["shared_target_speed"] == 0.15
    assert (
        gate_parameters["risk_readiness_provider"]
        == expected_gate_provider
    )
    assert gate_parameters["drift_status_topic"] == expected_status_topic
    assert gate_parameters["allow_uncalibrated_oacp_logging"] is False


def test_free_navigation_oacp_vb_parameters_share_the_existing_controller():
    source = (
        _package_root() / "launch" / "dream_free_navigation.launch.py"
    ).read_text(encoding="utf-8")
    assert 'choices=["balanced", "pure_mpc", "nominal", "oacp_vb"]' in source
    assert '"oacp_thresholds_calibrated"' in source
    assert '"oacp_calibration_logging_only"' in source
    assert '"c_th_max_exploration"' in source
    assert '"c_th_max_fallback"' in source
    assert '"velocity_slack_weight"' in source
    assert '"enable_contingency"' in source
    assert '"contingency_check_rate"' in source
    assert '"planner_mode": model' in source
    assert '"oacp_status_topic": "/dream/oacp_vb_status"' in source
    assert '"oacp_enable_contingency": ParameterValue(' in source
    assert '"oacp_contingency_check_rate": ParameterValue(' in source
    assert '"oacp_velocity_slack_weight": ParameterValue(' in source
    assert '"oacp_calibration_logging_only": ParameterValue(' in source
    assert '"calibration_logging_only": ParameterValue(' in source
    assert '"required_risk_provider": ParameterValue(' in source
    assert "required_risk_provider, value_type=str" in source
    assert '"drift_status_topic": ParameterValue(' in source
    assert "risk_status_topic, value_type=str" in source
    assert '"risk_readiness_provider": ParameterValue(' in source
    assert "gate_risk_provider, value_type=str" in source
    assert '"allow_uncalibrated_oacp_logging": ParameterValue(' in source
    assert source.count('"path_start_anchor_tolerance": (') == 2


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
        "dream_oacp_vb_assessor",
        "dream_oacp_calibration",
        "dream_merger_cue",
    ):
        assert executable in setup
    assert "dream_limo.OACP.assessor_node:main" in setup
    rviz = (
        root / "rviz" / "dream_free_navigation.rviz"
    ).read_text(encoding="utf-8")
    assert "Topic: /goal_pose" in rviz
    assert "Value: /global_costmap/costmap" in rviz
    assert "Value: /dream/geometric_path" in rviz
    assert "Value: /dream/reference_trajectory" in rviz
    assert "Value: /global_costmap/published_footprint" in rviz
    assert "Value: /dream/oacp_vb_markers" in rviz


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
