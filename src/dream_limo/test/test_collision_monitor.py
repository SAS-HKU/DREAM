from math import pi
from pathlib import Path

import numpy as np
import pytest

from dream_limo.core.collision import (
    axis_aligned_road_mask,
    CollisionEnvelope,
    CollisionGridSpec,
    interpolate_polyline,
    transform_points,
)


def make_spec():
    return CollisionGridSpec(
        width=21,
        height=21,
        resolution=0.1,
        origin_x=0.0,
        origin_y=0.0,
        frame_id="map",
    )


def make_envelope(**overrides):
    values = {
        "surface_retention_seconds": 0.5,
        "inflation_radius": 0.15,
        "minimum_valid_rays": 3,
        "interpolation_spacing": 0.05,
    }
    values.update(overrides)
    return CollisionEnvelope(make_spec(), **values)


def test_surface_cells_are_inflated_retained_then_expire():
    envelope = make_envelope()
    accepted = envelope.record_scan(
        np.asarray([[1.0, 1.0]]), receipt_time=10.0, valid_ray_count=3
    )
    assert accepted == 1
    grid, digest = envelope.render(np.zeros(make_spec().shape), now=10.4)
    assert grid[10, 10] == CollisionEnvelope.OCCUPIED
    assert grid[10, 11] == CollisionEnvelope.OCCUPIED
    assert digest.retained_surface_cells == 1
    assert digest.inflated_surface_cells > digest.retained_surface_cells

    expired, digest = envelope.render(np.zeros(make_spec().shape), now=10.51)
    assert digest.retained_surface_cells == 0
    assert expired[10, 10] == CollisionEnvelope.FREE


def test_insufficient_valid_rays_cannot_refresh_surface_map():
    envelope = make_envelope()
    accepted = envelope.record_scan(
        np.asarray([[1.0, 1.0]]), receipt_time=2.0, valid_ray_count=2
    )
    assert accepted == 0
    assert not envelope.last_scan_accepted
    grid, digest = envelope.render(np.zeros(make_spec().shape), now=2.0)
    assert digest.retained_surface_cells == 0
    assert grid[10, 10] == CollisionEnvelope.FREE


def test_shadow_unknown_is_nontraversable_and_sparse_path_is_interpolated():
    envelope = make_envelope(inflation_radius=0.0)
    shadow = np.zeros(make_spec().shape)
    shadow[10, 10] = 100
    grid, _ = envelope.render(shadow, now=1.0)
    assert grid[10, 10] == CollisionEnvelope.UNKNOWN

    assessment = envelope.assess_trajectory(
        np.asarray([[0.5, 1.0], [1.5, 1.0]]), grid
    )
    assert not assessment.clear
    assert assessment.reason == "UNKNOWN_SHADOW"
    assert assessment.evaluated_samples > 2


def test_surface_between_sparse_path_poses_is_not_skipped():
    envelope = make_envelope(inflation_radius=0.0)
    envelope.record_scan(
        np.asarray([[1.0, 1.0]]), receipt_time=1.0, valid_ray_count=3
    )
    grid, _ = envelope.render(np.zeros(make_spec().shape), now=1.1)
    result = envelope.assess_trajectory(
        np.asarray([[0.5, 1.0], [1.5, 1.0]]), grid
    )
    assert not result.clear
    assert result.reason == "OCCUPIED_SURFACE"


def test_outside_grid_and_outside_road_are_distinct_fail_closed_reasons():
    spec = make_spec()
    y = spec.origin_y + np.arange(spec.height) * spec.resolution
    road = np.broadcast_to(((y >= 0.5) & (y <= 1.5))[:, None], spec.shape)
    envelope = CollisionEnvelope(
        spec,
        surface_retention_seconds=0.5,
        inflation_radius=0.1,
        minimum_valid_rays=3,
        interpolation_spacing=0.05,
        traversable_mask=road,
    )
    grid, digest = envelope.render(np.zeros(spec.shape), now=1.0)
    assert digest.outside_road_cells > 0
    assert grid[2, 10] == CollisionEnvelope.OCCUPIED

    off_road = envelope.assess_trajectory(
        np.asarray([[0.5, 0.4], [1.0, 0.4]]), grid
    )
    assert not off_road.clear
    assert off_road.reason == "OUTSIDE_ROAD"

    outside = envelope.assess_trajectory(
        np.asarray([[0.5, 1.0], [2.1, 1.0]]), grid
    )
    assert not outside.clear
    assert outside.reason == "OUTSIDE_GRID"


def test_rigid_transform_math_used_after_timestamped_tf_lookup():
    half_angle = 0.5 * pi / 2.0
    transformed = transform_points(
        np.asarray([[1.0, 0.0, 0.0]]),
        translation_xyz=(2.0, 3.0, 0.0),
        quaternion_xyzw=(0.0, 0.0, np.sin(half_angle), np.cos(half_angle)),
    )
    assert transformed[0, 0] == pytest.approx(2.0)
    assert transformed[0, 1] == pytest.approx(4.0)


def test_polyline_interpolation_spacing_is_bounded():
    points = interpolate_polyline(np.asarray([[0.0, 0.0], [0.31, 0.0]]), 0.1)
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    assert np.max(distances) <= 0.1 + 1.0e-12
    assert points[-1, 0] == pytest.approx(0.31)


def test_ros_node_uses_scan_stamp_and_has_no_command_interface():
    root = Path(__file__).resolve().parents[1]
    source = (root / "dream_limo" / "collision_monitor_node.py").read_text()
    assert "Time.from_msg(message.header.stamp)" in source
    assert "lookup_transform(" in source
    assert "create_publisher(Twist" not in source
    assert '"/cmd_vel"' not in source
    assert '"/dream/arm"' not in source

    # Only the dedicated, disabled-by-default hardware boundary may include it;
    # existing SIL/live dry-run behavior remains unchanged.
    launch_sources = {
        path.name: path.read_text() for path in (root / "launch").glob("*.launch.py")
    }
    assert "dream_collision_monitor" in launch_sources[
        "dream_hardware_motion.launch.py"
    ]
    for name, content in launch_sources.items():
        if name != "dream_hardware_motion.launch.py":
            assert "dream_collision_monitor" not in content


def test_collision_node_keeps_configured_road_boundaries_inclusive():
    spec = CollisionGridSpec(
        width=241,
        height=81,
        resolution=0.025,
        origin_x=0.0,
        origin_y=-1.0,
    )
    mask = axis_aligned_road_mask(spec, y_min=-0.7, y_max=0.7)
    # -1 + 68*0.025 evaluates to 0.7000000000000002 on this platform.
    assert mask[68, 0]
    assert not mask[69, 0]

    envelope = CollisionEnvelope(
        spec,
        surface_retention_seconds=0.5,
        inflation_radius=0.20556349186104045,
        minimum_valid_rays=3,
        interpolation_spacing=0.0125,
        traversable_mask=mask,
    )
    grid, _ = envelope.render(np.zeros(spec.shape), now=1.0)
    result = envelope.assess_trajectory(
        np.asarray([[0.25, 0.5069], [0.50, 0.5069]]), grid
    )
    assert result.clear
