"""Isolated Stage 2 RViz smoke test; no real base driver and no /cmd_vel."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _require_isolated_domain(_context):
    domain = os.environ.get("ROS_DOMAIN_ID", "")
    if domain in {"", "0"}:
        raise RuntimeError(
            "The fake SIL publishes sensor-shaped topics. Set a non-default "
            "ROS_DOMAIN_ID (for example 42) before starting this launch."
        )
    return []


def generate_launch_description():
    share = get_package_share_directory("dream_limo")
    replay = os.path.join(share, "launch", "dream_replay.launch.py")
    params = os.path.join(share, "config", "dream_limo.yaml")
    arena = os.path.join(share, "config", "arena.yaml")
    rviz_config = os.path.join(share, "rviz", "dream_limo.rviz")
    preset = LaunchConfiguration("preset")
    scenario_delay = LaunchConfiguration("scenario_delay")
    scenario_duration = LaunchConfiguration("scenario_duration")
    rviz = LaunchConfiguration("rviz")
    report_path = LaunchConfiguration("report_path")
    return LaunchDescription(
        [
            OpaqueFunction(function=_require_isolated_domain),
            DeclareLaunchArgument("preset", default_value="balanced"),
            DeclareLaunchArgument("scenario_delay", default_value="2.0"),
            DeclareLaunchArgument("scenario_duration", default_value="12.0"),
            DeclareLaunchArgument("rviz", default_value="true"),
            DeclareLaunchArgument(
                "report_path",
                default_value=os.path.expanduser(
                    "~/limo_lvv_ws/dream_rviz_smoke_report.json"
                ),
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(replay),
                launch_arguments={
                    "preset": preset,
                    "scenario_delay": scenario_delay,
                    "scenario_duration": scenario_duration,
                }.items(),
            ),
            Node(
                package="dream_limo",
                executable="dream_visualization",
                name="dream_visualization",
                output="screen",
                parameters=[params, {"arena_file": arena}],
            ),
            Node(
                package="dream_limo",
                executable="dream_smoke_monitor",
                name="dream_smoke_monitor",
                output="screen",
                parameters=[
                    {
                        "report_path": report_path,
                        "scenario_duration": scenario_duration,
                        "experiment_arm": preset,
                    }
                ],
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="dream_rviz",
                output="screen",
                arguments=["-d", rviz_config],
                condition=IfCondition(rviz),
            ),
        ]
    )
