"""Reviewed real-sensor chain with a disabled-by-default physical boundary.

Starting this launch with its checked-in defaults continuously publishes zero
on /cmd_vel.  Nonzero output needs explicit commissioning assertions plus all
fresh runtime safety gates.  A validated RViz/ROS goal is the default mission
activation method.  ``auto_forward`` starts the same surveyed merge mission
without an RViz click; the legacy held-to-run joystick remains opt-in.
"""

import os
from math import isfinite

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _validated_target_speed(value):
    """Return a commissioned speed or raise a launch-facing error."""
    try:
        speed = float(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            "target_speed must be a number in (0.03, 0.15] m/s; "
            f"received {value!r}"
        ) from error
    if not isfinite(speed) or not 0.03 < speed <= 0.15:
        raise RuntimeError(
            "target_speed must be finite and in (0.03, 0.15] m/s; "
            f"received {value!r}. Start commissioning at 0.05 m/s."
        )
    return speed


def _validate_target_speed(context):
    _validated_target_speed(LaunchConfiguration("target_speed").perform(context))
    return []


def generate_launch_description():
    share = get_package_share_directory("dream_limo")
    params = os.path.join(share, "config", "dream_limo.yaml")
    mission = os.path.join(share, "config", "merge_mission.yaml")
    live_launch = os.path.join(share, "launch", "dream_live_demo.launch.py")

    model = LaunchConfiguration("model")
    rviz = LaunchConfiguration("rviz")
    activation_mode = LaunchConfiguration("activation_mode")
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
    auto_forward_mode = PythonExpression(
        ["'", activation_mode, "' == 'auto_forward'"]
    )
    authorizer_mode = PythonExpression(
        [
            "'",
            activation_mode,
            "' in ('goal', 'auto_forward')",
        ]
    )
    joystick_mode = PythonExpression(["'", activation_mode, "' == 'joystick'"])
    start_joy_in_joystick_mode = PythonExpression(
        [
            "'",
            activation_mode,
            "' == 'joystick' and '",
            start_joy,
            "'.lower() == 'true'",
        ]
    )
    expected_arm_owner = PythonExpression(
        [
            "'dream_goal_authorizer' if '",
            activation_mode,
            "' in ('goal', 'auto_forward') else 'dream_hardware_deadman'",
        ]
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "model",
                default_value="balanced",
                choices=["balanced", "pure_mpc"],
                description="DREAM or matched pure-MPC experiment arm.",
            ),
            DeclareLaunchArgument(
                "activation_mode",
                default_value="goal",
                choices=["goal", "auto_forward", "joystick"],
                description=(
                    "Mission activation: validated /goal_pose (default), the "
                    "configured forward merge mission, or legacy held-to-run "
                    "joystick."
                ),
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
                default_value="0.05",
                description=(
                    "MPC cruise speed and final hardware cap; start at 0.05 m/s "
                    "and do not exceed 0.15 m/s during commissioning."
                ),
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
            OpaqueFunction(function=_validate_target_speed),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(live_launch),
                launch_arguments={
                    "model": model,
                    "rviz": rviz,
                    # The camera remains visible and recorded as experiment
                    # evidence, but it is deliberately not a planner input or
                    # a physical-motion readiness gate.
                    "require_camera_evidence": "false",
                    "expected_cmd_vel_owner": "dream_hardware_command_gate",
                    "expected_arm_owner": expected_arm_owner,
                    "enforce_map_bounds": "true",
                    "latch_perceived_occlusion": "true",
                    "require_mission_goal": authorizer_mode,
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
                executable="dream_goal_authorizer",
                name="dream_goal_authorizer",
                output="screen",
                condition=IfCondition(authorizer_mode),
                parameters=[
                    params,
                    {
                        "arena_file": mission,
                        "enabled": True,
                        "auto_start": ParameterValue(
                            auto_forward_mode, value_type=bool
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
                condition=IfCondition(start_joy_in_joystick_mode),
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
                condition=IfCondition(joystick_mode),
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
                        "expected_deadman_owner": expected_arm_owner,
                        "maximum_speed": ParameterValue(
                            target_speed, value_type=float
                        ),
                    },
                ],
            ),
        ]
    )
