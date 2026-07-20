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
    target_speed = LaunchConfiguration("target_speed")
    use_merger_odom = LaunchConfiguration("use_merger_odom")
    merger_odom_input_topic = LaunchConfiguration("merger_odom_input_topic")
    merger_alignment_mode = LaunchConfiguration("merger_alignment_mode")
    merger_alignment_verified = LaunchConfiguration("merger_alignment_verified")
    merger_source_frame = LaunchConfiguration("merger_source_frame")
    merger_source_child_frame = LaunchConfiguration("merger_source_child_frame")
    source_reference_x = LaunchConfiguration("merger_source_reference_x")
    source_reference_y = LaunchConfiguration("merger_source_reference_y")
    source_reference_yaw = LaunchConfiguration("merger_source_reference_yaw")
    target_reference_x = LaunchConfiguration("merger_target_reference_x")
    target_reference_y = LaunchConfiguration("merger_target_reference_y")
    target_reference_yaw = LaunchConfiguration("merger_target_reference_yaw")

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
            DeclareLaunchArgument(
                "target_speed",
                default_value="0.15",
                description="MPC cruise speed; matched to the first-motion hardware cap.",
            ),
            DeclareLaunchArgument(
                "use_merger_odom",
                default_value="false",
                description=(
                    "Use a separately namespaced second-LIMO odometry stream; "
                    "otherwise use DREAM's LiDAR tracker."
                ),
            ),
            DeclareLaunchArgument(
                "merger_odom_input_topic", default_value="/merger/raw/wheel/odom"
            ),
            DeclareLaunchArgument(
                "merger_alignment_mode", default_value="measured_correspondence"
            ),
            DeclareLaunchArgument("merger_alignment_verified", default_value="false"),
            DeclareLaunchArgument("merger_source_frame", default_value="merger/odom"),
            DeclareLaunchArgument(
                "merger_source_child_frame", default_value="merger/base_link"
            ),
            DeclareLaunchArgument("merger_source_reference_x", default_value="0.0"),
            DeclareLaunchArgument("merger_source_reference_y", default_value="0.0"),
            DeclareLaunchArgument("merger_source_reference_yaw", default_value="0.0"),
            DeclareLaunchArgument("merger_target_reference_x", default_value="0.0"),
            DeclareLaunchArgument("merger_target_reference_y", default_value="0.0"),
            DeclareLaunchArgument("merger_target_reference_yaw", default_value="0.0"),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(live_launch),
                launch_arguments={
                    "model": model,
                    "rviz": rviz,
                    "expected_cmd_vel_owner": "dream_hardware_command_gate",
                    "expected_arm_owner": "dream_hardware_deadman",
                    "enforce_map_bounds": "true",
                    "latch_perceived_occlusion": "true",
                    "target_speed": target_speed,
                    "use_merger_odom": use_merger_odom,
                }.items(),
            ),
            Node(
                package="dream_limo",
                executable="dream_merger_odometry_adapter",
                name="dream_merger_odometry_adapter",
                output="screen",
                condition=IfCondition(use_merger_odom),
                parameters=[
                    params,
                    {
                        "input_topic": merger_odom_input_topic,
                        "alignment_mode": merger_alignment_mode,
                        "alignment_verified": ParameterValue(
                            merger_alignment_verified, value_type=bool
                        ),
                        "expected_source_frame": merger_source_frame,
                        "expected_source_child_frame": merger_source_child_frame,
                        "source_reference_x": ParameterValue(
                            source_reference_x, value_type=float
                        ),
                        "source_reference_y": ParameterValue(
                            source_reference_y, value_type=float
                        ),
                        "source_reference_yaw": ParameterValue(
                            source_reference_yaw, value_type=float
                        ),
                        "target_reference_x": ParameterValue(
                            target_reference_x, value_type=float
                        ),
                        "target_reference_y": ParameterValue(
                            target_reference_y, value_type=float
                        ),
                        "target_reference_yaw": ParameterValue(
                            target_reference_yaw, value_type=float
                        ),
                    },
                ],
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
                        "maximum_speed": ParameterValue(
                            target_speed, value_type=float
                        ),
                    },
                ],
            ),
        ]
    )
