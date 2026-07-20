"""Deprecated compatibility alias for the neutral DREAM vehicle demo."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    share = get_package_share_directory("dream_limo")
    live_demo = os.path.join(share, "launch", "dream_live_demo.launch.py")
    preset = LaunchConfiguration("preset")
    rviz = LaunchConfiguration("rviz")
    use_merger_odom = LaunchConfiguration("use_merger_odom")

    return LaunchDescription(
        [
            DeclareLaunchArgument("preset", default_value="balanced"),
            DeclareLaunchArgument("use_merger_odom", default_value="false"),
            DeclareLaunchArgument(
                "start_lidar_pedestrian_detector",
                default_value="false",
                description="Deprecated and ignored; no pedestrian detector is started.",
            ),
            DeclareLaunchArgument(
                "start_bev_projector",
                default_value="false",
                description="Deprecated and ignored; no BEV projector is started.",
            ),
            DeclareLaunchArgument("rviz", default_value="true"),
            LogInfo(
                msg=(
                    "dream_with_sfg_perception.launch.py is deprecated; using "
                    "neutral LiDAR clustering + DREAM vehicle tracking. Prefer "
                    "dream_live_demo.launch.py model:=balanced (or pure_mpc)."
                )
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(live_demo),
                launch_arguments={
                    "model": preset,
                    "rviz": rviz,
                    "use_merger_odom": use_merger_odom,
                }.items(),
            ),
        ]
    )
