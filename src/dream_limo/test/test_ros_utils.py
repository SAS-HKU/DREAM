from math import pi

import pytest

from dream_limo.ros_utils import alignment_from_initial_pose, transform_planar


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
