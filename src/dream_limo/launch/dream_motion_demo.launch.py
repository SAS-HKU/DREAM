"""One-argument closed-loop moving SIL demo with RViz; no physical base."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    share = get_package_share_directory("dream_limo")
    smoke_launch = os.path.join(share, "launch", "dream_rviz_smoke.launch.py")
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
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(smoke_launch),
                launch_arguments={
                    "preset": model,
                    "rviz": rviz,
                }.items(),
            ),
        ]
    )
