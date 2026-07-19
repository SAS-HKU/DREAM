"""Stage 2 SIL launch. It never starts limo_base and only uses /cmd_vel_test."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    share = get_package_share_directory("dream_limo")
    params = os.path.join(share, "config", "dream_limo.yaml")
    arena = os.path.join(share, "config", "arena.yaml")
    preset = LaunchConfiguration("preset")
    scenario_delay = LaunchConfiguration("scenario_delay")
    scenario_duration = LaunchConfiguration("scenario_duration")
    common = [params]
    return LaunchDescription(
        [
            DeclareLaunchArgument("preset", default_value="balanced"),
            DeclareLaunchArgument("scenario_delay", default_value="2.0"),
            DeclareLaunchArgument("scenario_duration", default_value="12.0"),
            Node(
                package="dream_limo",
                executable="dream_fake_world",
                name="dream_fake_world",
                output="screen",
                parameters=[
                    {
                        "auto_arm": True,
                        "arena_file": arena,
                        "scenario_delay": scenario_delay,
                        "scenario_duration": scenario_duration,
                    }
                ],
            ),
            Node(
                package="dream_limo",
                executable="dream_state_estimator",
                name="dream_state_estimator",
                output="screen",
                parameters=[params, {"initialize_from_first_odom": False}],
            ),
            Node(
                package="dream_limo",
                executable="dream_world_model",
                name="dream_world_model",
                output="screen",
                parameters=[
                    params,
                    {
                        "arena_file": arena,
                        "use_merger_odom": True,
                        "occlusion_source": "surveyed_polygon",
                    },
                ],
            ),
            Node(
                package="dream_limo",
                executable="dream_drift_field",
                name="dream_drift_field",
                output="screen",
                parameters=common,
            ),
            Node(
                package="dream_limo",
                executable="dream_planner",
                name="dream_planner",
                output="screen",
                parameters=[
                    params,
                    {
                        "preset": preset,
                        "arena_file": arena,
                    },
                ],
            ),
            Node(
                package="dream_limo",
                executable="dream_command_adapter",
                name="dream_command_adapter",
                output="screen",
                parameters=common,
            ),
            Node(
                package="dream_limo",
                executable="dream_safety_supervisor",
                name="dream_safety_supervisor",
                output="screen",
                parameters=common,
            ),
            Node(
                package="dream_limo",
                executable="dream_metrics",
                name="dream_metrics",
                output="screen",
                parameters=[params, {"arena_file": arena}],
            ),
            Node(
                package="dream_limo",
                executable="dream_preflight",
                name="dream_preflight",
                output="screen",
                parameters=[
                    {
                        "expected_sensor_owner": "dream_fake_world",
                        "allow_arm_publisher": True,
                    }
                ],
            ),
        ]
    )
