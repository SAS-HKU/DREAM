"""Arbitrary RViz-goal DREAM navigation with a planner-only Nav2 route layer.

The launch is physically disabled by default.  Nav2 supplies only an
Ackermann-feasible geometric path; ``dream_free_planner`` is the sole planner
that produces control, and the existing independent safety/collision/hardware
gates remain the only route to ``/cmd_vel``.
"""

import os
from math import isfinite

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _validated_target_speed(value):
    try:
        speed = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("target_speed must be a finite number") from exc
    if not isfinite(speed) or not 0.03 < speed <= 0.20:
        raise RuntimeError(
            "target_speed must lie in (0.03, 0.20] m/s for the reviewed "
            "physical hardware gate"
        )
    return speed


def _validate_arguments(context):
    target_speed = _validated_target_speed(
        LaunchConfiguration("target_speed").perform(context)
    )
    _validate_oacp_physical_readiness(
        model=LaunchConfiguration("model").perform(context),
        enable_physical_motion=LaunchConfiguration(
            "enable_physical_motion"
        ).perform(context),
        thresholds_calibrated=LaunchConfiguration(
            "oacp_thresholds_calibrated"
        ).perform(context),
        calibration_logging_only=LaunchConfiguration(
            "oacp_calibration_logging_only"
        ).perform(context),
        enable_contingency=LaunchConfiguration(
            "enable_contingency"
        ).perform(context),
        target_speed=target_speed,
    )
    return []


def _launch_boolean(name, value):
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    raise RuntimeError(f"{name} must be a boolean; received {value!r}")


def _validate_oacp_physical_readiness(
    *,
    model,
    enable_physical_motion,
    thresholds_calibrated,
    calibration_logging_only,
    enable_contingency,
    target_speed,
):
    """Refuse physical OACP-VB motion until its geometry thresholds are calibrated."""

    physical_enabled = _launch_boolean(
        "enable_physical_motion", enable_physical_motion
    )
    thresholds_ready = _launch_boolean(
        "oacp_thresholds_calibrated", thresholds_calibrated
    )
    logging_only = _launch_boolean(
        "oacp_calibration_logging_only", calibration_logging_only
    )
    contingency_enabled = _launch_boolean(
        "enable_contingency", enable_contingency
    )
    speed = float(target_speed)
    if (
        str(model).strip() == "oacp_vb"
        and physical_enabled
        and not thresholds_ready
        and not logging_only
    ):
        raise RuntimeError(
            "model=oacp_vb with physical motion requires "
            "oacp_thresholds_calibrated:=true after the required logging run"
        )
    if (
        str(model).strip() == "oacp_vb"
        and physical_enabled
        and not logging_only
        and not contingency_enabled
    ):
        raise RuntimeError(
            "physical OACP-VB comparison runs require "
            "enable_contingency:=true"
        )
    if (
        str(model).strip() == "oacp_vb"
        and physical_enabled
        and logging_only
        and speed > 0.15
    ):
        raise RuntimeError(
            "OACP-VB physical calibration logging is capped at "
            "target_speed:=0.15 m/s"
        )


def generate_launch_description():
    dream_share = get_package_share_directory("dream_limo")
    sfg_share = get_package_share_directory("sfg_nav")
    params = os.path.join(dream_share, "config", "dream_limo.yaml")
    geometry = os.path.join(dream_share, "config", "free_navigation.yaml")
    nav2_params = os.path.join(
        dream_share, "config", "nav2_dream_planner.yaml"
    )
    perception_params = os.path.join(sfg_share, "config", "limo_perception.yaml")
    rviz_config = os.path.join(
        dream_share, "rviz", "dream_free_navigation.rviz"
    )

    model = LaunchConfiguration("model")
    target_speed = LaunchConfiguration("target_speed")
    rviz = LaunchConfiguration("rviz")
    start_lidar_clusters = LaunchConfiguration("start_lidar_clusters")
    enable_physical_motion = LaunchConfiguration("enable_physical_motion")
    staging_pose_verified = LaunchConfiguration("staging_pose_verified")
    platform_watchdog_verified = LaunchConfiguration(
        "platform_watchdog_verified"
    )
    operator_kill_verified = LaunchConfiguration("operator_kill_verified")
    oacp_thresholds_calibrated = LaunchConfiguration(
        "oacp_thresholds_calibrated"
    )
    oacp_calibration_logging_only = LaunchConfiguration(
        "oacp_calibration_logging_only"
    )
    c_th_max_exploration = LaunchConfiguration("c_th_max_exploration")
    c_th_max_fallback = LaunchConfiguration("c_th_max_fallback")
    velocity_slack_weight = LaunchConfiguration("velocity_slack_weight")
    enable_contingency = LaunchConfiguration("enable_contingency")
    contingency_check_rate = LaunchConfiguration("contingency_check_rate")
    path_start_anchor_tolerance = 0.20
    oacp_mode = PythonExpression(["'", model, "' == 'oacp_vb'"])
    drift_mode = PythonExpression(
        ["'", model, "' not in ('oacp_vb', 'nominal')"]
    )
    required_risk_provider = PythonExpression(
        ["'oacp_vb' if '", model, "' == 'oacp_vb' else ''"]
    )
    gate_risk_provider = PythonExpression(
        [
            "'oacp_vb' if '",
            model,
            "' == 'oacp_vb' else ('nominal' if '",
            model,
            "' == 'nominal' else 'drift')",
        ]
    )
    risk_status_topic = PythonExpression(
        [
            "'/dream/oacp_vb_status' if '",
            model,
            "' == 'oacp_vb' else ('/dream/planner_status' if '",
            model,
            "' == 'nominal' else '/dream/drift_status')",
        ]
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "model",
                default_value="balanced",
                choices=["balanced", "pure_mpc", "nominal", "oacp_vb"],
                description=(
                    "Shared-controller experiment arm: DREAM, legacy pure-MPC, "
                    "Nominal, or OACP-VB."
                ),
            ),
            DeclareLaunchArgument(
                "target_speed",
                default_value="0.15",
                description="Cruise speed and final physical output cap in m/s.",
            ),
            DeclareLaunchArgument("rviz", default_value="true"),
            DeclareLaunchArgument("start_lidar_clusters", default_value="true"),
            DeclareLaunchArgument("enable_physical_motion", default_value="false"),
            DeclareLaunchArgument("staging_pose_verified", default_value="false"),
            DeclareLaunchArgument(
                "platform_watchdog_verified", default_value="false"
            ),
            DeclareLaunchArgument("operator_kill_verified", default_value="false"),
            DeclareLaunchArgument(
                "oacp_thresholds_calibrated",
                default_value="false",
                description=(
                    "Explicit acknowledgement that OACP-VB thresholds came from "
                    "the required arena logging run."
                ),
            ),
            DeclareLaunchArgument(
                "oacp_calibration_logging_only",
                default_value="false",
                description=(
                    "Compute and log OACP-VB bounds without applying them; "
                    "calibration runs are excluded from arm comparisons."
                ),
            ),
            DeclareLaunchArgument(
                "c_th_max_exploration",
                default_value="4.5",
                description="OACP-VB exploration-branch risk threshold.",
            ),
            DeclareLaunchArgument(
                "c_th_max_fallback",
                default_value="6.0",
                description="OACP-VB fallback-branch risk threshold.",
            ),
            DeclareLaunchArgument(
                "velocity_slack_weight",
                default_value="1e4",
                description="Heavy MPC penalty for OACP-VB speed-bound slack.",
            ),
            DeclareLaunchArgument(
                "enable_contingency",
                default_value="true",
                description="Verify the OACP-VB fallback branch with a second solve.",
            ),
            DeclareLaunchArgument(
                "contingency_check_rate",
                default_value="1.0",
                description=(
                    "Reduced OACP-VB fallback-verification rate in Hz; "
                    "the executed MPC remains at the shared 5 Hz rate."
                ),
            ),
            OpaqueFunction(function=_validate_arguments),
            # Reuse only SFG's public class-neutral scan clustering.  Its
            # pedestrian detector, tracker, planner, and cmd_vel output are not
            # started or imported by this launch.
            Node(
                package="sfg_nav",
                executable="lidar_cluster_buffer",
                name="lidar_cluster_buffer_node",
                output="screen",
                # SFG's public cluster buffer is perception-only here.  The
                # LIMO base TF can arrive just after the LaserScan timestamp;
                # latest-TF avoids dropping a newly revealed agent while the
                # robot is moving slowly.  DREAM still performs its own
                # world modelling, planning, and control.
                parameters=[perception_params, {"use_latest_tf": True}],
                condition=IfCondition(start_lidar_clusters),
            ),
            Node(
                package="dream_limo",
                executable="dream_vehicle_tracker",
                name="dream_vehicle_tracker",
                output="screen",
                parameters=[params],
            ),
            Node(
                package="dream_limo",
                executable="dream_state_estimator",
                name="dream_state_estimator",
                output="screen",
                parameters=[
                    params,
                    {
                        "initialize_from_first_odom": True,
                        "initial_map_x": 0.0,
                        "initial_map_y": 0.0,
                        "initial_map_yaw": 0.0,
                    },
                ],
            ),
            Node(
                package="dream_limo",
                executable="dream_world_model",
                name="dream_world_model",
                output="screen",
                parameters=[
                    params,
                    {
                        "arena_file": geometry,
                        "use_merger_odom": False,
                        "occlusion_source": "lidar_first_return",
                    },
                ],
            ),
            Node(
                package="dream_limo",
                executable="dream_drift_field",
                name="dream_drift_field",
                output="screen",
                parameters=[params, {"arena_file": geometry}],
                condition=IfCondition(drift_mode),
            ),
            Node(
                package="dream_limo",
                executable="dream_oacp_vb_assessor",
                name="dream_oacp_vb_assessor",
                output="screen",
                parameters=[
                    params,
                    {
                        "arena_file": geometry,
                        "target_speed": ParameterValue(
                            target_speed, value_type=float
                        ),
                        "c_th_max_exploration": ParameterValue(
                            c_th_max_exploration, value_type=float
                        ),
                        "c_th_max_fallback": ParameterValue(
                            c_th_max_fallback, value_type=float
                        ),
                        "thresholds_calibrated": ParameterValue(
                            oacp_thresholds_calibrated, value_type=bool
                        ),
                        "calibration_logging_only": ParameterValue(
                            oacp_calibration_logging_only, value_type=bool
                        ),
                        "path_start_anchor_tolerance": (
                            path_start_anchor_tolerance
                        ),
                    },
                ],
                condition=IfCondition(oacp_mode),
            ),
            # Planner-server only: no Nav2 controller_server, navigator,
            # behavior server, velocity smoother, or cmd_vel publisher exists.
            Node(
                package="nav2_planner",
                executable="planner_server",
                name="planner_server",
                output="screen",
                parameters=[nav2_params],
            ),
            Node(
                package="nav2_lifecycle_manager",
                executable="lifecycle_manager",
                name="dream_planner_lifecycle_manager",
                output="screen",
                parameters=[
                    {
                        "use_sim_time": False,
                        "autostart": True,
                        "node_names": ["planner_server"],
                        "bond_timeout": 4.0,
                    }
                ],
            ),
            Node(
                package="dream_limo",
                executable="dream_nav2_path_provider",
                name="dream_nav2_path_provider",
                output="screen",
                # Shared by all arms.  This matches the reviewed 1 Hz OACP-VB
                # contingency verification rate while the executed MPC and
                # live swept-costmap checks remain at 5 Hz.
                parameters=[{"replan_period": 1.0}],
            ),
            Node(
                package="dream_limo",
                executable="dream_free_planner",
                name="dream_free_planner",
                output="screen",
                parameters=[
                    params,
                    {
                        "arena_file": geometry,
                        "preset": model,
                        "planner_mode": model,
                        "oacp_status_topic": "/dream/oacp_vb_status",
                        "oacp_enable_contingency": ParameterValue(
                            enable_contingency, value_type=bool
                        ),
                        "oacp_contingency_check_rate": ParameterValue(
                            contingency_check_rate, value_type=float
                        ),
                        "oacp_velocity_slack_weight": ParameterValue(
                            velocity_slack_weight, value_type=float
                        ),
                        "oacp_calibration_logging_only": ParameterValue(
                            oacp_calibration_logging_only, value_type=bool
                        ),
                        "target_speed": ParameterValue(
                            target_speed, value_type=float
                        ),
                        "path_start_anchor_tolerance": (
                            path_start_anchor_tolerance
                        ),
                        "enforce_map_bounds": True,
                        # The lidar is currently cropped to a forward 220 deg
                        # field of view.  When the operator has verified the
                        # staging clearance, let only the fixed launch-area
                        # blind corner bootstrap into fully observed space.
                        "verified_start_clearance_enabled": ParameterValue(
                            staging_pose_verified, value_type=bool
                        ),
                        "verified_start_clearance_radius": 0.30,
                    },
                ],
            ),
            Node(
                package="dream_limo",
                executable="dream_command_adapter",
                name="dream_command_adapter",
                output="screen",
                parameters=[params],
            ),
            Node(
                package="dream_limo",
                executable="dream_safety_supervisor",
                name="dream_safety_supervisor",
                output="screen",
                parameters=[params],
            ),
            Node(
                package="dream_limo",
                executable="dream_camera_evidence",
                name="dream_camera_evidence",
                output="screen",
                parameters=[params],
            ),
            Node(
                package="dream_limo",
                executable="dream_metrics",
                name="dream_metrics",
                output="screen",
                parameters=[params, {"arena_file": geometry}],
            ),
            Node(
                package="dream_limo",
                executable="dream_preflight",
                name="dream_preflight",
                output="screen",
                parameters=[
                    params,
                    {
                        "expected_cmd_vel_owner": "dream_hardware_command_gate",
                        "expected_arm_owner": "dream_free_goal_authorizer",
                        # The camera is visual experiment evidence only.
                        "require_camera_evidence": False,
                        # Occlusion remains a live DRIFT input, not a launch-time
                        # condition tied to a surveyed route coordinate.
                        "require_perceived_occlusion": False,
                    },
                ],
            ),
            Node(
                package="dream_limo",
                executable="dream_free_goal_authorizer",
                name="dream_free_goal_authorizer",
                output="screen",
                # The matching Nav2 grid already inflates measured returns by
                # more than the padded LIMO circumscribed radius.  Validate the
                # clicked centre cell here; Nav2 must still prove a complete
                # known-space, footprint-valid route before the arm can assert.
                parameters=[
                    params,
                    {
                        "enabled": True,
                        "footprint_clearance": 0.0,
                        "required_risk_provider": ParameterValue(
                            required_risk_provider, value_type=str
                        ),
                        "shared_minimum_speed": 0.0,
                        "shared_target_speed": ParameterValue(
                            target_speed, value_type=float
                        ),
                    },
                ],
            ),
            Node(
                package="dream_limo",
                executable="dream_collision_monitor",
                name="dream_collision_monitor",
                output="screen",
                parameters=[params, {"arena_file": geometry}],
            ),
            Node(
                package="dream_limo",
                executable="dream_hardware_command_gate",
                name="dream_hardware_command_gate",
                output="screen",
                parameters=[
                    params,
                    {
                        "hardware_output_enabled": ParameterValue(
                            enable_physical_motion, value_type=bool
                        ),
                        "staging_pose_verified": ParameterValue(
                            staging_pose_verified, value_type=bool
                        ),
                        "platform_watchdog_verified": ParameterValue(
                            platform_watchdog_verified, value_type=bool
                        ),
                        "operator_kill_verified": ParameterValue(
                            operator_kill_verified, value_type=bool
                        ),
                        "expected_deadman_owner": "dream_free_goal_authorizer",
                        "maximum_speed": ParameterValue(
                            target_speed, value_type=float
                        ),
                        "drift_status_topic": ParameterValue(
                            risk_status_topic, value_type=str
                        ),
                        "risk_readiness_provider": ParameterValue(
                            gate_risk_provider, value_type=str
                        ),
                        "allow_uncalibrated_oacp_logging": ParameterValue(
                            oacp_calibration_logging_only, value_type=bool
                        ),
                    },
                ],
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="dream_free_navigation_rviz",
                output="screen",
                arguments=["-d", rviz_config],
                condition=IfCondition(rviz),
            ),
        ]
    )
