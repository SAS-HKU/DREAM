from dataclasses import FrozenInstanceError
from math import isclose

import numpy as np
import pytest

from dream_limo.OACP.oacp_vb import (
    ContingencyBranch,
    OACPVBConfig,
    PVSComponent,
    PVSLengthPolicy,
    VelocityRegion,
    build_phantom_merge_connector,
    calibrate_thresholds,
    dynamic_velocity_bound,
    evaluate_geometry_risk,
    extract_pvs_components,
    lateral_risk,
    longitudinal_risk,
    make_pvs_interval,
    point_risk,
    potential_pv_count,
    reduce_horizon_risk,
)


def _config(**overrides):
    values = {
        "v_pv_max": 1.0,
        "prediction_horizon": 4.0,
        "lane_width": 0.375,
        "confidence_z": 1.645,
        "c_th_min": 0.0,
        "c_th_max_exploration": 4.5,
        "c_th_max_fallback": 6.0,
        "v_occ_min": 0.11,
        "v_occ_max": 0.20,
        "pvs_length_policy": PVSLengthPolicy.CLIP,
    }
    values.update(overrides)
    return OACPVBConfig(**values)


def _connector(
    *,
    route=None,
    ego=(0.0, 0.0),
    lane_width=0.375,
    perception_range=4.0,
    sampling_spacing=0.2,
    merge_length=1.0,
):
    if route is None:
        route = np.array([[0.0, 0.0], [5.0, 0.0]])
    return build_phantom_merge_connector(
        route,
        ego,
        lane_width=lane_width,
        perception_range=perception_range,
        sampling_spacing=sampling_spacing,
        merge_length=merge_length,
    )


def _mask_for_connector_samples(
    connector,
    sample_indices,
    *,
    origin=(-1.0, -1.0),
    resolution=0.05,
    shape=(60, 140),
):
    mask = np.zeros(shape, dtype=np.uint8)
    for index in sample_indices:
        point = connector.points[index]
        column = int(np.floor((point[0] - origin[0]) / resolution))
        row = int(np.floor((point[1] - origin[1]) / resolution))
        mask[row, column] = 1
    return mask


def test_pvs_clip_policy_makes_eq10_intervals_valid():
    config = _config()
    pvs = make_pvs_interval(2.0, 8.5, config)
    assert pvs.start == 2.0
    assert pvs.requested_end == 8.5
    assert pvs.end == 6.0
    assert pvs.length == config.v_pv_max * config.prediction_horizon
    assert pvs.was_clipped


def test_pvs_reject_policy_refuses_too_long_interval():
    config = _config(pvs_length_policy=PVSLengthPolicy.REJECT)
    with pytest.raises(ValueError, match="exceeds"):
        make_pvs_interval(2.0, 8.5, config)


def test_eq10_is_continuous_at_interval_boundaries():
    config = _config()
    pvs = make_pvs_interval(2.0, 4.0, config)
    epsilon = 1.0e-8
    for boundary in (pvs.end, pvs.start + config.maximum_pvs_length):
        left = potential_pv_count(boundary - epsilon, pvs, config)
        exact = potential_pv_count(boundary, pvs, config)
        right = potential_pv_count(boundary + epsilon, pvs, config)
        assert isclose(left, exact, rel_tol=0.0, abs_tol=1.0e-7)
        assert isclose(right, exact, rel_tol=0.0, abs_tol=1.0e-7)


def test_eq10_is_nonnegative_and_zero_outside_full_reach():
    config = _config()
    pvs = make_pvs_interval(2.0, 4.0, config)
    samples = [1.9 + index * 0.01 for index in range(621)]
    assert all(potential_pv_count(value, pvs, config) >= 0.0 for value in samples)
    assert potential_pv_count(pvs.start - 1.0e-6, pvs, config) == 0.0
    assert potential_pv_count(
        pvs.end + config.maximum_pvs_length + 1.0e-6,
        pvs,
        config,
    ) == 0.0
    assert potential_pv_count(pvs.start, pvs, config) == 0.0
    assert potential_pv_count(
        pvs.end + config.maximum_pvs_length,
        pvs,
        config,
    ) == 0.0


def test_longitudinal_and_point_risk_follow_eq11_and_eq13():
    config = _config()
    pvs = make_pvs_interval(2.0, 4.0, config)
    position = 3.5
    g_value = potential_pv_count(position, pvs, config)
    assert longitudinal_risk(position, pvs, config) == pvs.length * g_value
    assert point_risk(position, 0.0, pvs, config) == pvs.length * g_value


def test_normalized_lateral_risk_is_symmetric_and_monotone_in_abs_offset():
    config = _config()
    assert lateral_risk(0.0, config) == 1.0
    offsets = [0.0, 0.02, 0.05, 0.10, 0.20, 0.40]
    values = [lateral_risk(offset, config) for offset in offsets]
    assert all(left >= right for left, right in zip(values, values[1:]))
    for offset in offsets:
        assert lateral_risk(offset, config) == lateral_risk(-offset, config)


def test_horizon_reducer_uses_max_when_frs_intersects():
    result = reduce_horizon_risk(
        [0.1, 0.9, 0.3],
        frs_intersects_trajectory=True,
    )
    assert result.risk_total == 0.9
    assert result.raw_maximum == 0.9
    assert result.maximum_index == 1
    assert result.sample_count == 3
    assert not result.ignored_by_remark_2


def test_remark2_gate_zeros_risk_when_frs_does_not_intersect():
    result = reduce_horizon_risk(
        [0.1, 0.9, 0.3],
        frs_intersects_trajectory=False,
    )
    assert result.risk_total == 0.0
    assert result.raw_maximum == 0.9
    assert result.maximum_index == 1
    assert result.ignored_by_remark_2


@pytest.mark.parametrize(
    ("risk", "expected", "region"),
    [
        (0.0, 0.20, VelocityRegion.MAXIMUM),
        (2.25, 0.155, VelocityRegion.INTERPOLATED),
        (4.5, 0.11, VelocityRegion.MINIMUM),
        (100.0, 0.11, VelocityRegion.MINIMUM),
    ],
)
def test_dynamic_velocity_bound_is_clamped_in_all_three_regions(
    risk,
    expected,
    region,
):
    result = dynamic_velocity_bound(
        risk,
        _config(),
        ContingencyBranch.EXPLORATION,
    )
    assert isclose(result.velocity_bound, expected, rel_tol=0.0, abs_tol=1.0e-12)
    assert result.region is region


def test_velocity_bound_honours_nonzero_lower_threshold():
    config = _config(
        c_th_min=1.0,
        c_th_max_exploration=5.0,
        c_th_max_fallback=7.0,
    )
    below = dynamic_velocity_bound(0.25, config, "exploration")
    middle = dynamic_velocity_bound(3.0, config, "exploration")
    assert below.velocity_bound == config.v_occ_max
    assert below.region is VelocityRegion.MAXIMUM
    assert isclose(middle.velocity_bound, 0.155, rel_tol=0.0, abs_tol=1.0e-12)


def test_branch_threshold_order_is_logged_not_semantically_relabelled():
    config = _config()
    exploration = dynamic_velocity_bound(3.0, config, "exploration")
    fallback = dynamic_velocity_bound(3.0, config, "fallback")
    assert exploration.maximum_risk_threshold == 4.5
    assert fallback.maximum_risk_threshold == 6.0
    assert exploration.velocity_bound < fallback.velocity_bound


def test_threshold_calibration_uses_linear_p70_and_fallback_ratio():
    result = calibrate_thresholds([4.0, 0.0, 3.0, 1.0, 2.0])
    assert result.percentile == 0.70
    assert result.sample_count == 5
    assert result.observed_minimum == 0.0
    assert result.observed_maximum == 4.0
    assert isclose(result.exploration_threshold, 2.8)
    assert isclose(result.fallback_threshold, 2.8 * 4.0 / 3.0)


def test_threshold_calibration_rejects_riskless_occluded_phase():
    with pytest.raises(ValueError, match="insufficient positive risk"):
        calibrate_thresholds([0.0, 0.0, 0.0])


def test_configuration_and_results_are_immutable():
    config = _config()
    pvs = make_pvs_interval(0.0, 1.0, config)
    result = reduce_horizon_risk(
        [point_risk(0.5, 0.0, pvs, config)],
        frs_intersects_trajectory=True,
    )
    with pytest.raises(FrozenInstanceError):
        config.v_pv_max = 2.0
    with pytest.raises(FrozenInstanceError):
        result.risk_total = 0.0


@pytest.mark.parametrize(
    "values",
    [
        [],
        [-0.1],
        [float("nan")],
        [float("inf")],
    ],
)
def test_horizon_reducer_rejects_invalid_samples(values):
    with pytest.raises(ValueError):
        reduce_horizon_risk(values, frs_intersects_trajectory=True)


def test_straight_connector_starts_one_lane_right_and_tapers_to_route():
    connector = _connector(
        ego=(1.0, 0.0),
        lane_width=0.4,
        perception_range=3.0,
        sampling_spacing=0.25,
        merge_length=1.0,
    )
    np.testing.assert_allclose(connector.reference_points[0], [1.0, 0.0])
    np.testing.assert_allclose(connector.points[0], [1.0, -0.4])
    np.testing.assert_allclose(connector.points[-1], connector.reference_points[-1])
    assert connector.reference_s[-1] == 3.0
    assert np.all(np.diff(connector.cumulative_s) > 0.0)
    with pytest.raises(ValueError):
        connector.points[0, 0] = 99.0


def test_curved_connector_keeps_right_side_sign_and_arc_length_sampling():
    route = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.5],
            [3.0, 1.5],
        ]
    )
    connector = _connector(
        route=route,
        lane_width=0.3,
        perception_range=2.5,
        sampling_spacing=0.2,
        merge_length=1.5,
    )
    first_tangent = route[1] - route[0]
    first_offset = connector.points[0] - connector.reference_points[0]
    cross = (
        first_tangent[0] * first_offset[1]
        - first_tangent[1] * first_offset[0]
    )
    assert cross < 0.0
    assert isclose(np.linalg.norm(first_offset), 0.3)
    assert np.allclose(np.diff(connector.reference_s)[:-1], 0.2)
    np.testing.assert_allclose(connector.points[-1], connector.reference_points[-1])


def test_connector_caps_sampling_at_route_end_and_reports_it():
    connector = _connector(
        route=np.array([[0.0, 0.0], [1.1, 0.0]]),
        perception_range=3.0,
        sampling_spacing=0.2,
        merge_length=0.5,
    )
    assert connector.route_end_clipped
    assert isclose(connector.effective_range, 1.1)
    assert isclose(connector.reference_s[-1], 1.1)


def test_mask_extraction_preserves_visible_gaps_as_separate_components():
    connector = _connector(merge_length=0.5)
    mask = _mask_for_connector_samples(
        connector,
        [2, 3, 4, 8, 9, 10],
    )
    result = extract_pvs_components(
        mask,
        connector,
        (0.0, 0.0),
        grid_origin_xy=(-1.0, -1.0),
        grid_resolution=0.05,
        perception_range=4.0,
        config=_config(),
    )
    assert len(result.components) == 2
    assert (
        result.components[0].interval.end
        < result.components[1].interval.start
    )
    assert (
        result.components[0].first_sample_index,
        result.components[0].last_sample_index,
    ) == (2, 4)
    assert (
        result.components[1].first_sample_index,
        result.components[1].last_sample_index,
    ) == (8, 10)


def test_mask_extraction_range_caps_a_shadow_component_explicitly():
    connector = _connector(
        lane_width=0.2,
        perception_range=4.0,
        sampling_spacing=0.2,
        merge_length=0.5,
    )
    mask = _mask_for_connector_samples(
        connector,
        range(connector.points.shape[0]),
    )
    result = extract_pvs_components(
        mask,
        connector,
        (0.0, 0.0),
        grid_origin_xy=(-1.0, -1.0),
        grid_resolution=0.05,
        perception_range=1.1,
        config=_config(),
    )
    assert result.range_was_clipped
    assert len(result.components) == 1
    assert result.components[0].range_clipped
    assert result.components[0].was_clipped
    assert result.components[0].interval.end < connector.cumulative_s[-1]


def test_visible_mask_collapses_pvs_and_empty_geometry_risk_is_valid():
    connector = _connector()
    extraction = extract_pvs_components(
        np.zeros((60, 140), dtype=np.uint8),
        connector,
        (0.0, 0.0),
        grid_origin_xy=(-1.0, -1.0),
        grid_resolution=0.05,
        perception_range=4.0,
        config=_config(),
    )
    assert extraction.components == ()
    evaluation = evaluate_geometry_risk(
        [[0.5, 0.0], [1.0, 0.0]],
        connector,
        extraction,
        _config(),
        conflict_distance=0.2,
    )
    assert evaluation.risk_total == 0.0
    assert evaluation.raw_maximum == 0.0
    assert evaluation.active_component_index is None
    assert not evaluation.ignored_by_remark_2


def test_remark2_finite_conflict_gate_ignores_nonintersecting_horizon():
    config = _config()
    connector = _connector(merge_length=4.0)
    component = PVSComponent(
        interval=make_pvs_interval(0.2, 0.9, config),
        first_sample_index=1,
        last_sample_index=4,
        range_clipped=False,
    )
    evaluation = evaluate_geometry_risk(
        [[0.5, 0.0], [0.8, 0.0], [1.1, 0.0]],
        connector,
        [component],
        config,
        conflict_distance=0.05,
    )
    assert evaluation.raw_maximum > 0.0
    assert evaluation.risk_total == 0.0
    assert evaluation.component_intersections == (False,)
    assert evaluation.component_minimum_distances[0] > 0.05
    assert evaluation.ignored_by_remark_2


def test_merge_intersection_produces_positive_multi_component_maximum():
    config = _config()
    connector = _connector(merge_length=1.0)
    components = [
        PVSComponent(
            interval=make_pvs_interval(0.1, 0.5, config),
            first_sample_index=1,
            last_sample_index=2,
            range_clipped=False,
        ),
        PVSComponent(
            interval=make_pvs_interval(0.7, 1.1, config),
            first_sample_index=4,
            last_sample_index=5,
            range_clipped=False,
        ),
    ]
    horizon = np.array([[0.65, 0.0], [0.9, 0.0], [1.2, 0.0], [1.5, 0.0]])
    evaluation = evaluate_geometry_risk(
        horizon,
        connector,
        components,
        config,
        conflict_distance=0.12,
    )
    assert evaluation.risk_total > 0.0
    assert evaluation.risk_total == evaluation.raw_maximum
    assert any(evaluation.component_intersections)
    assert evaluation.active_component_index in (0, 1)
    assert evaluation.active_horizon_index in range(len(horizon))
    assert evaluation.selected_conflict_distance <= 0.12


@pytest.mark.parametrize(
    ("mask", "origin", "resolution", "message"),
    [
        (np.zeros(10), (-1.0, -1.0), 0.05, "two-dimensional"),
        (
            np.array([[0.0, np.nan]]),
            (-1.0, -1.0),
            0.05,
            "finite",
        ),
        (np.zeros((2, 2)), (float("nan"), 0.0), 0.05, "finite"),
        (np.zeros((2, 2)), (-1.0, -1.0), 0.0, "positive"),
    ],
)
def test_mask_extraction_rejects_malformed_grid_metadata(
    mask,
    origin,
    resolution,
    message,
):
    with pytest.raises(ValueError, match=message):
        extract_pvs_components(
            mask,
            _connector(),
            (0.0, 0.0),
            grid_origin_xy=origin,
            grid_resolution=resolution,
            perception_range=4.0,
            config=_config(),
        )
