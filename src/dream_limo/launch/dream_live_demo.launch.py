"""Concise live camera/LiDAR DREAM dry run with physical output disabled."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    dream_share = get_package_share_directory("dream_limo")
    sfg_share = get_package_share_directory("sfg_nav")
    sensor_launch = os.path.join(
        dream_share, "launch", "dream_sensor_smoke.launch.py"
    )
    perception_params = os.path.join(sfg_share, "config", "limo_perception.yaml")
    dream_params = os.path.join(dream_share, "config", "dream_limo.yaml")
    model = LaunchConfiguration("model")
    rviz = LaunchConfiguration("rviz")
    expected_cmd_vel_owner = LaunchConfiguration("expected_cmd_vel_owner")
    expected_arm_owner = LaunchConfiguration("expected_arm_owner")
    enforce_map_bounds = LaunchConfiguration("enforce_map_bounds")
    latch_perceived_occlusion = LaunchConfiguration("latch_perceived_occlusion")
    target_speed = LaunchConfiguration("target_speed")
    require_mission_goal = LaunchConfiguration("require_mission_goal")
    use_merger_odom = LaunchConfiguration("use_merger_odom")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "model",
                default_value="balanced",
                description="Planner arm: balanced (DREAM) or pure_mpc (baseline).",
            ),
            DeclareLaunchArgument("rviz", default_value="true"),
            DeclareLaunchArgument("expected_cmd_vel_owner", default_value=""),
            DeclareLaunchArgument("expected_arm_owner", default_value=""),
            DeclareLaunchArgument("enforce_map_bounds", default_value="false"),
            DeclareLaunchArgument("latch_perceived_occlusion", default_value="false"),
            DeclareLaunchArgument("target_speed", default_value="0.50"),
            DeclareLaunchArgument("require_mission_goal", default_value="false"),
            DeclareLaunchArgument("use_merger_odom", default_value="false"),
            # Reuse only SFG's neutral scan-to-cluster public front end.  Do not
            # start its pedestrian detector, generic tracker, or planner here.
            Node(
                package="sfg_nav",
                executable="lidar_cluster_buffer",
                name="lidar_cluster_buffer_node",
                output="screen",
                parameters=[perception_params],
                condition=UnlessCondition(use_merger_odom),
            ),
            Node(
                package="dream_limo",
                executable="dream_vehicle_tracker",
                name="dream_vehicle_tracker",
                output="screen",
                parameters=[dream_params],
                condition=UnlessCondition(use_merger_odom),
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(sensor_launch),
                launch_arguments={
                    "preset": model,
                    "use_merger_odom": use_merger_odom,
                    "rviz": rviz,
                    "expected_cmd_vel_owner": expected_cmd_vel_owner,
                    "expected_arm_owner": expected_arm_owner,
                    "enforce_map_bounds": enforce_map_bounds,
                    "latch_perceived_occlusion": latch_perceived_occlusion,
                    "target_speed": target_speed,
                    "require_mission_goal": require_mission_goal,
                }.items(),
            ),
        ]
    )
