"""Stationary real-sensor DREAM dry run with RViz; it never publishes /cmd_vel."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory("dream_limo")
    params = os.path.join(share, "config", "dream_limo.yaml")
    mission = os.path.join(share, "config", "merge_mission.yaml")
    dry_run = os.path.join(share, "launch", "dream_limo_dry_run.launch.py")
    rviz_config = os.path.join(share, "rviz", "dream_sensor.rviz")
    preset = LaunchConfiguration("preset")
    use_merger_odom = LaunchConfiguration("use_merger_odom")
    rviz = LaunchConfiguration("rviz")
    expected_cmd_vel_owner = LaunchConfiguration("expected_cmd_vel_owner")
    expected_arm_owner = LaunchConfiguration("expected_arm_owner")
    enforce_map_bounds = LaunchConfiguration("enforce_map_bounds")
    latch_perceived_occlusion = LaunchConfiguration("latch_perceived_occlusion")
    target_speed = LaunchConfiguration("target_speed")
    require_mission_goal = LaunchConfiguration("require_mission_goal")
    return LaunchDescription(
        [
            DeclareLaunchArgument("preset", default_value="balanced"),
            DeclareLaunchArgument("use_merger_odom", default_value="false"),
            DeclareLaunchArgument("rviz", default_value="true"),
            DeclareLaunchArgument("expected_cmd_vel_owner", default_value=""),
            DeclareLaunchArgument("expected_arm_owner", default_value=""),
            DeclareLaunchArgument("enforce_map_bounds", default_value="false"),
            DeclareLaunchArgument("latch_perceived_occlusion", default_value="false"),
            DeclareLaunchArgument("target_speed", default_value="0.50"),
            DeclareLaunchArgument("require_mission_goal", default_value="false"),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(dry_run),
                launch_arguments={
                    "preset": preset,
                    "use_merger_odom": use_merger_odom,
                    "require_camera_evidence": "true",
                    "require_perceived_occlusion": "true",
                    "expected_cmd_vel_owner": expected_cmd_vel_owner,
                    "expected_arm_owner": expected_arm_owner,
                    "enforce_map_bounds": enforce_map_bounds,
                    "latch_perceived_occlusion": latch_perceived_occlusion,
                    "target_speed": target_speed,
                    "require_mission_goal": require_mission_goal,
                }.items(),
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
                executable="dream_visualization",
                name="dream_visualization",
                output="screen",
                parameters=[
                    params,
                    {"arena_file": mission, "mode_label": "LIVE SENSOR DRY RUN"},
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
