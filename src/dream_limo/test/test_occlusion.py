import numpy as np

from dream_limo.core.occlusion import (
    LidarShadowBuilder,
    line_of_sight_visible,
    rectangle_polygon,
    scan_line_of_sight_visible,
    simulate_polygon_scan,
)
from dream_limo.core.risk_field import DREAMRiskField
from dream_limo.limo_scale import default_deployment_config


def test_line_of_sight_gate_hides_track_behind_truck():
    truck = rectangle_polygon("truck", 2.4, 0.0, 1.2, 0.24)
    assert not line_of_sight_visible((0.35, 0.45), (3.1, -0.45), [truck])
    assert line_of_sight_visible((3.3, 0.45), (3.8, -0.45), [truck])


def test_lidar_shadow_requires_confirmed_truck_return():
    config = default_deployment_config()
    field = DREAMRiskField(config)
    truck = rectangle_polygon("truck", 2.4, 0.0, 1.2, 0.24)
    scan = simulate_polygon_scan((0.45, 0.45, 0.0), [truck])
    builder = LidarShadowBuilder(maximum_shadow_range=6.0)
    mask = builder.build(field.X, field.Y, field.road_mask, scan, [truck])
    assert np.count_nonzero(mask) > 100
    assert mask[field.y.searchsorted(0.0), field.x.searchsorted(4.0)] == 1.0
    no_confirmation = builder.build(field.X, field.Y, field.road_mask, scan, [])
    assert np.count_nonzero(no_confirmation) == 0


def test_live_lidar_shadow_needs_no_surveyed_polygon():
    config = default_deployment_config()
    field = DREAMRiskField(config)
    truck = rectangle_polygon("unknown_occluder", 2.4, 0.0, 1.2, 0.24)
    scan = simulate_polygon_scan((0.45, 0.45, 0.0), [truck])
    builder = LidarShadowBuilder(
        maximum_shadow_range=6.0,
        require_known_occluder=False,
    )
    mask = builder.build(field.X, field.Y, field.road_mask, scan, [])
    assert np.count_nonzero(mask) > 100
    assert mask[field.y.searchsorted(0.0), field.x.searchsorted(4.0)] == 1.0


def test_scan_visibility_gate_uses_first_return_and_fails_outside_fov():
    truck = rectangle_polygon("unknown_occluder", 2.4, 0.0, 1.2, 0.24)
    scan = simulate_polygon_scan((0.45, 0.45, 0.0), [truck])
    assert not scan_line_of_sight_visible(scan, (3.1, -0.10), target_radius=0.01)
    assert scan_line_of_sight_visible(scan, (1.0, 0.45), target_radius=0.01)
    assert not scan_line_of_sight_visible(scan, (-1.0, 0.45), target_radius=0.01)
