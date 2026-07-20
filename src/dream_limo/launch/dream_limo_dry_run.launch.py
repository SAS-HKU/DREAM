"""Live read-only inputs with final output hard-limited to /cmd_vel_test."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    share = get_package_share_directory("dream_limo")
    params = os.path.join(share, "config", "dream_limo.yaml")
    mission = os.path.join(share, "config", "merge_mission.yaml")
    preset = LaunchConfiguration("preset")
    use_merger_odom = LaunchConfiguration("use_merger_odom")
    require_camera_evidence = LaunchConfiguration("require_camera_evidence")
    require_perceived_occlusion = LaunchConfiguration("require_perceived_occlusion")
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
            DeclareLaunchArgument("require_camera_evidence", default_value="false"),
            DeclareLaunchArgument("require_perceived_occlusion", default_value="false"),
            DeclareLaunchArgument("expected_cmd_vel_owner", default_value=""),
            DeclareLaunchArgument("expected_arm_owner", default_value=""),
            DeclareLaunchArgument("enforce_map_bounds", default_value="false"),
            DeclareLaunchArgument("latch_perceived_occlusion", default_value="false"),
            DeclareLaunchArgument("target_speed", default_value="0.50"),
            DeclareLaunchArgument("require_mission_goal", default_value="false"),
            Node(
                package="dream_limo",
                executable="dream_state_estimator",
                name="dream_state_estimator",
                output="screen",
                parameters=[params],
            ),
            Node(
                package="dream_limo",
                executable="dream_world_model",
                name="dream_world_model",
                output="screen",
                parameters=[
                    params,
                    {"arena_file": mission, "use_merger_odom": use_merger_odom},
                ],
            ),
            Node(
                package="dream_limo",
                executable="dream_drift_field",
                name="dream_drift_field",
                output="screen",
                parameters=[params],
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
                        "arena_file": mission,
                        "enforce_map_bounds": enforce_map_bounds,
                        "target_speed": ParameterValue(target_speed, value_type=float),
                        "require_mission_goal": ParameterValue(
                            require_mission_goal, value_type=bool
                        ),
                    },
                ],
            ),
            Node(
                package="dream_limo",
                executable="dream_command_adapter",
                name="dream_command_adapter",
                output="screen",
                parameters=[params],
            ),
            Node(
                package="dream_limo",
                executable="dream_safety_supervisor",
                name="dream_safety_supervisor",
                output="screen",
                parameters=[params],
            ),
            Node(
                package="dream_limo",
                executable="dream_metrics",
                name="dream_metrics",
                output="screen",
                parameters=[params, {"arena_file": mission}],
            ),
            Node(
                package="dream_limo",
                executable="dream_preflight",
                name="dream_preflight",
                output="screen",
                parameters=[
                    {
                        "require_camera_evidence": require_camera_evidence,
                        "require_perceived_occlusion": require_perceived_occlusion,
                        "expected_cmd_vel_owner": expected_cmd_vel_owner,
                        "expected_arm_owner": expected_arm_owner,
                        "latch_perceived_occlusion": latch_perceived_occlusion,
                    }
                ],
            ),
        ]
    )
