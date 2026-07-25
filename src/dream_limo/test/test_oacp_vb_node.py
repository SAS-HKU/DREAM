"""Focused tests for the OACP-VB ROS assessor's pure boundary helpers."""

import json

import numpy as np
import pytest
from std_msgs.msg import String

from dream_limo.core.oacp_vb import OACPVBConfig
from dream_limo.core.types import EgoState
from dream_limo.limo_scale import deployment_config_for_arena
from dream_limo.oacp_vb_node import (
    GEOMETRY_ASSUMPTION,
    GridContract,
    OACPVBNode,
    canonical_assessment_path,
    compute_assessment,
    provisional_straight_route,
    validate_grid_payload,
    validate_planar_quaternion,
)


def _config(target_speed=0.15):
    return OACPVBConfig(
        v_pv_max=1.0,
        prediction_horizon=4.0,
        lane_width=0.45,
        confidence_z=1.645,
        c_th_min=0.0,
        c_th_max_exploration=4.5,
        c_th_max_fallback=6.0,
        v_occ_min=0.55 * target_speed,
        v_occ_max=target_speed,
    )


def test_grid_contract_rejects_nonidentity_and_nonfinite_data():
    contract = GridContract("map", 3, 2, 0.05, -1.0, -2.0)
    mask = validate_grid_payload(
        contract=contract,
        frame_id="map",
        width=3,
        height=2,
        resolution=0.05,
        origin_xyz=(-1.0, -2.0, 0.0),
        origin_quaternion=(0.0, 0.0, 0.0, 1.0),
        data=(0, 100, 0, 100, 0, 100),
    )
    assert mask.shape == (2, 3)
    with pytest.raises(ValueError, match="identity"):
        validate_grid_payload(
            contract=contract,
            frame_id="map",
            width=3,
            height=2,
            resolution=0.05,
            origin_xyz=(-1.0, -2.0, 0.0),
            origin_quaternion=(0.0, 0.0, 0.1, 0.995),
            data=(0, 100, 0, 100, 0, 100),
        )
    with pytest.raises(ValueError, match="finite"):
        validate_grid_payload(
            contract=contract,
            frame_id="map",
            width=3,
            height=2,
            resolution=0.05,
            origin_xyz=(-1.0, -2.0, 0.0),
            origin_quaternion=(0.0, 0.0, 0.0, 1.0),
            data=(0, 100, 0, float("nan"), 0, 100),
        )


def test_planar_quaternion_and_provisional_route_are_strict():
    assert validate_planar_quaternion((0.0, 0.0, 0.0, 1.0)) == 0.0
    with pytest.raises(ValueError, match="normalized"):
        validate_planar_quaternion((0.0, 0.0, 0.0, 0.5))
    route = provisional_straight_route(
        (1.0, 2.0), np.pi / 2.0, route_length=3.0, sampling_spacing=0.05
    )
    assert np.allclose(route[0], (1.0, 2.0))
    assert np.allclose(route[-1], (1.0, 5.0))


def test_assessor_anchors_the_same_valid_nav2_start_gap_as_the_planner():
    ego = EgoState(0.0, 0.0, 0.0, 0.0)
    raw = np.asarray([[0.15, 0.0], [1.0, 0.0], [2.0, 0.0]])
    canonical = canonical_assessment_path(
        raw,
        np.zeros(3),
        ego,
        maximum_start_gap=0.20,
    )
    np.testing.assert_allclose(canonical[0], (0.0, 0.0))
    np.testing.assert_allclose(canonical[1:], raw)


def test_default_scale_occluded_connector_has_nonzero_conflict_risk(tmp_path):
    arena_file = tmp_path / "arena.yaml"
    arena_file.write_text(
        "\n".join(
            (
                "frame_id: map",
                "grid:",
                "  x_min: -1.0",
                "  x_max: 4.0",
                "  y_min: -2.0",
                "  y_max: 2.0",
                "  resolution: 0.05",
                "  road_y_min: -2.0",
                "  road_y_max: 2.0",
                "lanes:",
                "  width: 0.45",
                "  centers: [0.45, 0.0, -0.45]",
                "route:",
                "  target_lane: 1",
                "  merge_request_x: 0.35",
                "  merge_path_x: [1.0, 2.0]",
                "  conflict_zone_x: [1.5, 3.0]",
                "  mission_goal_x: 3.5",
            )
        ),
        encoding="utf-8",
    )
    deployment = deployment_config_for_arena(str(arena_file))
    config = _config()
    mask = np.full(
        (deployment.grid.ny, deployment.grid.nx), 100.0, dtype=np.float64
    )
    route = np.column_stack(
        (
            np.linspace(0.0, 3.0, 61),
            np.zeros(61, dtype=np.float64),
        )
    )
    result = compute_assessment(
        ego=EgoState(
            x=0.0,
            y=0.0,
            yaw=0.0,
            speed=0.0,
            yaw_rate=0.0,
            stamp=1.0,
            lane_index=0,
        ),
        shadow_mask=mask,
        path_points=route,
        deployment=deployment,
        oacp_config=config,
        perception_range=3.0,
        sampling_spacing=0.05,
        merge_length=config.v_occ_max * config.prediction_horizon,
        conflict_distance=deployment.mpc.robot_width
        + deployment.safety.collision_inflation_margin,
        risk_evaluation_steps=20,
    )
    assert result.exact_bound_valid
    assert result.risk is not None
    assert result.risk.risk_total > 0.0
    assert not result.risk.ignored_by_remark_2
    assert result.exploration is not None
    assert result.exploration.velocity_bound < config.v_occ_max
    assert (
        GEOMETRY_ASSUMPTION
        == "path_relative_right_lane_merge_connector_nominal_risk_horizon"
    )


def test_calibration_samples_are_reset_and_armed_per_accepted_goal():
    node = OACPVBNode.__new__(OACPVBNode)
    node.calibration_logging_only = True
    node.calibration_goal_identity = None
    node.calibration_run_active = False
    node.risk_samples = [99.0]
    node.last_risk_sample_key = (1.0, 1.0, 1.0)

    OACPVBNode._on_deadman_status(
        node,
        String(
            data=json.dumps(
                {
                    "goal_accepted": True,
                    "goal_revision": 4,
                    "goal_receipt_stamp": 10.0,
                    "accepted_for_motion": False,
                }
            )
        ),
    )

    assert node.calibration_goal_identity == (4, 10.0)
    assert node.risk_samples == []
    assert node.last_risk_sample_key is None
    assert node.calibration_run_active is False

    OACPVBNode._on_deadman_status(
        node,
        String(
            data=json.dumps(
                {
                    "goal_accepted": True,
                    "goal_revision": 4,
                    "goal_receipt_stamp": 10.0,
                    "accepted_for_motion": True,
                }
            )
        ),
    )
    assert node.calibration_run_active is True

    OACPVBNode._on_deadman_status(
        node,
        String(data=json.dumps({"stop_latched": True})),
    )
    assert node.calibration_run_active is False
