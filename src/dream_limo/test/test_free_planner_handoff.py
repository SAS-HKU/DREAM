from types import SimpleNamespace

from geometry_msgs.msg import PoseStamped

from dream_limo.free_planner_node import DreamFreePlannerNode
from dream_limo.ros_utils import yaw_to_quaternion


def _planner_for_handoff() -> DreamFreePlannerNode:
    # Exercise the pure route-identity method without creating ROS entities.
    planner = DreamFreePlannerNode.__new__(DreamFreePlannerNode)
    goal = PoseStamped()
    goal.header.frame_id = "map"
    goal.header.stamp.sec = 10
    goal.pose.position.x = 1.0
    goal.pose.position.y = 0.2
    qx, qy, qz, qw = yaw_to_quaternion(0.3)
    goal.pose.orientation.x = qx
    goal.pose.orientation.y = qy
    goal.pose.orientation.z = qz
    goal.pose.orientation.w = qw
    planner.goal = goal
    planner.path_source_stamp = 12.0
    planner.route_status = {
        "ready": True,
        "goal_x": 1.0,
        "goal_y": 0.2,
        "goal_yaw": 0.3,
        "goal_stamp": 10.0,
        "path_source_stamp": 11.0,
    }
    planner.get_parameter = lambda name: SimpleNamespace(
        value={
            "goal_match_tolerance": 1.0e-3,
            "path_stamp_tolerance": 1.0e-6,
        }[name]
    )
    return planner


def test_replan_handoff_keeps_current_goal_ready_across_dds_stamp_ordering():
    planner = _planner_for_handoff()
    # The new Path can arrive before the matching status message.  A different
    # positive path stamp is acceptable because goal revision, endpoint/yaw,
    # path freshness, and costmap validation are enforced independently.
    assert planner.path_source_stamp != planner.route_status["path_source_stamp"]
    assert planner._route_matches_goal()


def test_replan_handoff_still_requires_ready_matching_goal_revision():
    planner = _planner_for_handoff()
    planner.route_status["ready"] = False
    assert not planner._route_matches_goal()

    planner = _planner_for_handoff()
    planner.route_status["goal_stamp"] = 9.9
    assert not planner._route_matches_goal()


def test_replan_handoff_rejects_invalid_path_stamps():
    planner = _planner_for_handoff()
    planner.path_source_stamp = 0.0
    assert not planner._route_matches_goal()

    planner = _planner_for_handoff()
    planner.route_status["path_source_stamp"] = float("nan")
    assert not planner._route_matches_goal()


def test_replan_handoff_waits_if_ready_status_names_a_newer_unreceived_path():
    planner = _planner_for_handoff()
    planner.route_status["path_source_stamp"] = 13.0
    assert not planner._route_matches_goal()


def test_oacp_handoff_accepts_route_status_for_received_pending_path():
    planner = _planner_for_handoff()
    planner.pending_path_source_stamp = 13.0
    planner.route_status["path_source_stamp"] = 13.0
    assert planner._route_matches_goal()
