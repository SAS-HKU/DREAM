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
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _validate_arguments(context):
    try:
        speed = float(LaunchConfiguration("target_speed").perform(context))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("target_speed must be a finite number") from exc
    if not isfinite(speed) or not 0.03 < speed <= 0.15:
        raise RuntimeError(
            "target_speed must lie in (0.03, 0.15] m/s for the reviewed "
            "physical hardware gate"
        )
    return []


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

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "model",
                default_value="balanced",
                choices=["balanced", "pure_mpc"],
                description="DREAM risk-aware arm or matched pure-MPC baseline.",
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
                        "target_speed": ParameterValue(
                            target_speed, value_type=float
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
                    {"enabled": True, "footprint_clearance": 0.0},
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
