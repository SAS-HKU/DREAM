"""Stationary live DREAM view with one explicitly aligned second-LIMO track."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
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
    target_speed = LaunchConfiguration("target_speed")
    input_topic = LaunchConfiguration("merger_odom_input_topic")
    alignment_mode = LaunchConfiguration("merger_alignment_mode")
    alignment_verified = LaunchConfiguration("merger_alignment_verified")
    source_frame = LaunchConfiguration("merger_source_frame")
    source_child_frame = LaunchConfiguration("merger_source_child_frame")
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
            ),
            DeclareLaunchArgument("rviz", default_value="true"),
            DeclareLaunchArgument("target_speed", default_value="0.50"),
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
                    "target_speed": target_speed,
                    "use_merger_odom": "true",
                }.items(),
            ),
            Node(
                package="dream_limo",
                executable="dream_merger_odometry_adapter",
                name="dream_merger_odometry_adapter",
                output="screen",
                parameters=[
                    params,
                    {
                        "input_topic": input_topic,
                        "alignment_mode": alignment_mode,
                        "alignment_verified": ParameterValue(
                            alignment_verified, value_type=bool
                        ),
                        "expected_source_frame": source_frame,
                        "expected_source_child_frame": source_child_frame,
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
        ]
    )
