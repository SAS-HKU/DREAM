"""Concise live camera/LiDAR DREAM dry run with physical output disabled."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
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

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "model",
                default_value="balanced",
                description="Planner arm: balanced (DREAM) or pure_mpc (baseline).",
            ),
            DeclareLaunchArgument("rviz", default_value="true"),
            # Reuse only SFG's neutral scan-to-cluster public front end.  Do not
            # start its pedestrian detector, generic tracker, or planner here.
            Node(
                package="sfg_nav",
                executable="lidar_cluster_buffer",
                name="lidar_cluster_buffer_node",
                output="screen",
                parameters=[perception_params],
            ),
            Node(
                package="dream_limo",
                executable="dream_vehicle_tracker",
                name="dream_vehicle_tracker",
                output="screen",
                parameters=[dream_params],
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(sensor_launch),
                launch_arguments={
                    "preset": model,
                    "use_merger_odom": "false",
                    "rviz": rviz,
                }.items(),
            ),
        ]
    )
