"""Reviewed real-sensor chain with a disabled-by-default physical boundary.

Starting this launch with its checked-in defaults continuously publishes zero
on /cmd_vel.  Nonzero output needs explicit commissioning assertions plus all
fresh runtime safety gates and a held two-button physical joystick chord.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = get_package_share_directory("dream_limo")
    params = os.path.join(share, "config", "dream_limo.yaml")
    live_launch = os.path.join(share, "launch", "dream_live_demo.launch.py")

    model = LaunchConfiguration("model")
    rviz = LaunchConfiguration("rviz")
    enable_physical_motion = LaunchConfiguration("enable_physical_motion")
    staging_pose_verified = LaunchConfiguration("staging_pose_verified")
    platform_watchdog_verified = LaunchConfiguration("platform_watchdog_verified")
    operator_kill_verified = LaunchConfiguration("operator_kill_verified")
    deadman_device_verified = LaunchConfiguration("deadman_device_verified")
    start_joy = LaunchConfiguration("start_joy")
    joy_device_id = LaunchConfiguration("joy_device_id")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "model",
                default_value="balanced",
                choices=["balanced", "pure_mpc"],
                description="DREAM or matched pure-MPC experiment arm.",
            ),
            DeclareLaunchArgument("rviz", default_value="true"),
            DeclareLaunchArgument("enable_physical_motion", default_value="false"),
            DeclareLaunchArgument("staging_pose_verified", default_value="false"),
            DeclareLaunchArgument("platform_watchdog_verified", default_value="false"),
            DeclareLaunchArgument("operator_kill_verified", default_value="false"),
            DeclareLaunchArgument("deadman_device_verified", default_value="false"),
            DeclareLaunchArgument("start_joy", default_value="false"),
            DeclareLaunchArgument("joy_device_id", default_value="0"),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(live_launch),
                launch_arguments={
                    "model": model,
                    "rviz": rviz,
                    "expected_cmd_vel_owner": "dream_hardware_command_gate",
                    "expected_arm_owner": "dream_hardware_deadman",
                    "enforce_map_bounds": "true",
                    "latch_perceived_occlusion": "true",
                }.items(),
            ),
            Node(
                package="dream_limo",
                executable="dream_collision_monitor",
                name="dream_collision_monitor",
                output="screen",
                parameters=[params],
            ),
            Node(
                package="joy",
                executable="joy_node",
                name="joy_node",
                output="screen",
                condition=IfCondition(start_joy),
                parameters=[
                    {
                        "device_id": ParameterValue(joy_device_id, value_type=int),
                        "autorepeat_rate": 20.0,
                        "coalesce_interval_ms": 1,
                    }
                ],
            ),
            Node(
                package="dream_limo",
                executable="dream_hardware_deadman",
                name="dream_hardware_deadman",
                output="screen",
                parameters=[params, {"enabled": deadman_device_verified}],
            ),
            Node(
                package="dream_limo",
                executable="dream_hardware_command_gate",
                name="dream_hardware_command_gate",
                output="screen",
                parameters=[
                    params,
                    {
                        "hardware_output_enabled": enable_physical_motion,
                        "staging_pose_verified": staging_pose_verified,
                        "platform_watchdog_verified": platform_watchdog_verified,
                        "operator_kill_verified": operator_kill_verified,
                    },
                ],
            ),
        ]
    )
