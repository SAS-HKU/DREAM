from math import pi

import numpy as np
import pytest

from dream_limo.core.merger_odometry import (
    AlignmentResolver,
    PlanarAlignment,
    PlanarOdometry,
    PlanarPose,
    align_planar_odometry,
    rotate_pose_covariance,
    validate_source_frames,
    validate_source_time,
)


def test_measured_pose_correspondence_maps_reference_and_other_pose():
    source = PlanarPose(2.0, -1.0, pi / 2.0)
    target = PlanarPose(0.4, 1.2, 0.0)
    alignment = PlanarAlignment.from_pose_correspondence(source, target)

    mapped_reference = alignment.transform_pose(source)
    assert mapped_reference.x == pytest.approx(target.x)
    assert mapped_reference.y == pytest.approx(target.y)
    assert mapped_reference.yaw == pytest.approx(target.yaw)

    # One metre along the source-frame +x axis becomes one metre along the
    # target-frame -y axis for this -90-degree alignment.
    mapped = alignment.transform_pose(PlanarPose(3.0, -1.0, pi / 2.0))
    assert mapped.x == pytest.approx(0.4)
    assert mapped.y == pytest.approx(0.2)
    assert mapped.yaw == pytest.approx(0.0)


def test_first_message_anchor_latches_once_and_does_not_reanchor():
    resolver = AlignmentResolver(
        mode="first_message_anchor",
        target_reference=PlanarPose(1.0, 2.0, pi / 2.0),
    )
    assert not resolver.initialized

    first = PlanarPose(10.0, -3.0, 0.0)
    alignment = resolver.resolve(first)
    assert resolver.initialized
    assert alignment.transform_pose(first).x == pytest.approx(1.0)
    assert alignment.transform_pose(first).y == pytest.approx(2.0)

    second_alignment = resolver.resolve(PlanarPose(11.0, -3.0, 0.0))
    assert second_alignment is alignment
    mapped = second_alignment.transform_pose(PlanarPose(11.0, -3.0, 0.0))
    assert mapped.x == pytest.approx(1.0)
    assert mapped.y == pytest.approx(3.0)


def test_alignment_preserves_source_stamp_and_child_frame_twist_exactly():
    twist = (0.23, -0.04, 0.0, 0.0, 0.0, -0.31)
    sample = PlanarOdometry(
        stamp=1234.56789,
        pose=PlanarPose(1.0, 2.0, 0.2),
        child_twist=twist,
    )
    aligned = align_planar_odometry(
        sample,
        PlanarAlignment(tx=5.0, ty=-2.0, yaw=pi / 2.0),
    )

    assert aligned.stamp == sample.stamp
    assert aligned.child_twist == twist
    assert aligned.pose.x == pytest.approx(3.0)
    assert aligned.pose.y == pytest.approx(-1.0)
    assert aligned.pose.yaw == pytest.approx(0.2 + pi / 2.0)


def test_pose_covariance_rotates_but_child_twist_covariance_need_not():
    covariance = np.zeros((6, 6), dtype=float)
    covariance[0, 0] = 4.0
    covariance[1, 1] = 1.0
    covariance[3, 3] = 9.0
    covariance[4, 4] = 2.0
    transformed = np.asarray(
        rotate_pose_covariance(covariance.ravel(), pi / 2.0)
    ).reshape((6, 6))

    assert transformed[0, 0] == pytest.approx(1.0)
    assert transformed[1, 1] == pytest.approx(4.0)
    assert transformed[3, 3] == pytest.approx(2.0)
    assert transformed[4, 4] == pytest.approx(9.0)
    assert transformed[2, 2] == pytest.approx(0.0)
    assert transformed[5, 5] == pytest.approx(0.0)


def test_frame_validation_requires_exact_namespaced_source_frames():
    validate_source_frames(
        actual_parent="merger/odom",
        actual_child="merger/base_link",
        expected_parent="merger/odom",
        expected_child="merger/base_link",
        output_parent="odom",
        allow_parent_alias=False,
    )

    with pytest.raises(ValueError, match="does not match"):
        validate_source_frames(
            actual_parent="other/odom",
            actual_child="merger/base_link",
            expected_parent="merger/odom",
            expected_child="merger/base_link",
            output_parent="odom",
            allow_parent_alias=False,
        )


def test_generic_source_odom_alias_fails_closed_without_explicit_override():
    arguments = {
        "actual_parent": "odom",
        "actual_child": "merger/base_link",
        "expected_parent": "odom",
        "expected_child": "merger/base_link",
        "output_parent": "odom",
    }
    with pytest.raises(ValueError, match="origins are unrelated"):
        validate_source_frames(**arguments, allow_parent_alias=False)
    validate_source_frames(**arguments, allow_parent_alias=True)


@pytest.mark.parametrize(
    ("source_stamp", "now", "previous_stamp", "match"),
    [
        (9.0, 10.0, None, "stale"),
        (10.2, 10.0, None, "future"),
        (10.0, 10.0, 10.0, "strictly increasing"),
        (0.0, 10.0, None, "zero or negative"),
    ],
)
def test_timestamp_validation_fails_closed(
    source_stamp,
    now,
    previous_stamp,
    match,
):
    with pytest.raises(ValueError, match=match):
        validate_source_time(
            source_stamp,
            now,
            maximum_age=0.5,
            maximum_future_skew=0.05,
            previous_stamp=previous_stamp,
        )


def test_timestamp_at_freshness_bound_is_accepted():
    validate_source_time(
        9.5,
        10.0,
        maximum_age=0.5,
        maximum_future_skew=0.05,
        previous_stamp=9.4,
    )


def test_invalid_alignment_and_nonfinite_inputs_are_rejected_before_latching():
    with pytest.raises(ValueError, match="mode"):
        AlignmentResolver(
            mode="guess",
            target_reference=PlanarPose(0.0, 0.0, 0.0),
        )
    resolver = AlignmentResolver(
        mode="first_message_anchor",
        target_reference=PlanarPose(0.0, 0.0, 0.0),
    )
    with pytest.raises(ValueError, match="non-finite"):
        resolver.resolve(PlanarPose(float("nan"), 0.0, 0.0))
    assert not resolver.initialized


def test_covariance_shape_and_finiteness_are_validated():
    with pytest.raises(ValueError, match="36"):
        rotate_pose_covariance([0.0] * 35, 0.0)
    covariance = [0.0] * 36
    covariance[7] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        rotate_pose_covariance(covariance, 0.0)
