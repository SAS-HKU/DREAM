from math import pi

import pytest

from dream_limo.ros_utils import (
    alignment_from_initial_pose,
    child_velocity_to_parent,
    transform_planar,
)


def test_first_odom_alignment_maps_pose_to_mission_start():
    source = (2.0, -1.0, pi / 2.0)
    target = (0.35, 0.45, 0.0)
    tx, ty, yaw = alignment_from_initial_pose(
        *source,
        target_x=target[0],
        target_y=target[1],
        target_yaw=target[2],
    )
    x, y, _, _ = transform_planar(
        source[0], source[1], 0.0, 0.0, tx=tx, ty=ty, yaw=yaw
    )
    assert x == pytest.approx(target[0])
    assert y == pytest.approx(target[1])
    assert source[2] + yaw == pytest.approx(target[2])


def test_odometry_child_twist_is_rotated_by_child_pose_before_map_alignment():
    odom_vx, odom_vy = child_velocity_to_parent(
        0.4,
        0.0,
        child_yaw=pi / 2.0,
    )
    assert odom_vx == pytest.approx(0.0, abs=1.0e-12)
    assert odom_vy == pytest.approx(0.4)

    _, _, map_vx, map_vy = transform_planar(
        0.0,
        0.0,
        odom_vx,
        odom_vy,
        tx=1.0,
        ty=2.0,
        yaw=-pi / 2.0,
    )
    assert map_vx == pytest.approx(0.4)
    assert map_vy == pytest.approx(0.0, abs=1.0e-12)
