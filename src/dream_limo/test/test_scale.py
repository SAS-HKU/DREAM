from dataclasses import replace
from math import isclose

import numpy as np

from dream_limo.core.risk_field import DREAMRiskField
from dream_limo.limo_scale import (
    GridConfig,
    MPCConfig,
    default_deployment_config,
    deployment_config_for_arena,
)


def test_formula_correct_similarity_values():
    config = default_deployment_config()
    assert isclose(config.scale.diffusion(0.3), config.pde.d0, rel_tol=0.0, abs_tol=1e-15)
    assert isclose(config.scale.diffusion(6.0), config.pde.d_occ, rel_tol=0.0, abs_tol=1e-15)
    assert config.scale.source(1.0) == config.pde.source_scale
    assert config.scale.speed(10.0) == 0.5
    assert config.scale.acceleration(1.0) == 0.025
    assert config.mpc.minimum_speed == 0.0
    assert config.mpc.maximum_steer < np.deg2rad(23.4)
    assert config.safety.collision_inflation_margin == 0.05
    assert config.mpc.cbf_slack_weight == 2.0e4
    assert config.arena.mission_goal_x == 5.55


def test_mpc_rejects_cruise_speed_that_conflicts_with_limits():
    for value in (0.02, 0.61):
        try:
            MPCConfig(target_speed=value)
        except ValueError as exc:
            assert "target_speed" in str(exc)
        else:
            raise AssertionError("unsafe target speed was accepted")


def test_pde_operator_is_similarity_invariant():
    """Scaled grid/operator has identical field shape and beta-scaled RHS."""
    base = default_deployment_config()
    highway_grid = GridConfig(
        x_min=0.0,
        x_max=6.0,
        y_min=-1.0,
        y_max=1.0,
        resolution=0.25,
        road_y_min=-0.7,
        road_y_max=0.7,
        road_taper=0.25,
    )
    scaled_grid = replace(
        highway_grid,
        x_max=0.6,
        y_min=-0.1,
        y_max=0.1,
        resolution=0.025,
        road_y_min=-0.07,
        road_y_max=0.07,
        road_taper=0.025,
    )
    highway = DREAMRiskField(replace(base, grid=highway_grid))
    scaled = DREAMRiskField(replace(base, grid=scaled_grid))
    assert highway.shape == scaled.shape
    rng = np.random.default_rng(7)
    field = rng.uniform(0.1, 0.8, highway.shape)
    source = rng.uniform(0.0, 1.0, highway.shape)
    diffusion = np.full(highway.shape, 0.3)
    vx = np.full(highway.shape, 10.0)
    vy = np.full(highway.shape, 0.5)
    decay = np.full(highway.shape, 0.15)
    rhs_highway = highway._rhs(field, source, diffusion, vx, vy, decay)
    rhs_scaled = scaled._rhs(
        field,
        base.scale.source(1.0) * source,
        base.scale.diffusion(0.3) * np.ones_like(diffusion),
        base.scale.speed(10.0) * np.ones_like(vx),
        base.scale.speed(0.5) * np.ones_like(vy),
        base.scale.decay(0.15) * np.ones_like(decay),
    )
    np.testing.assert_allclose(rhs_scaled, rhs_highway / base.scale.beta, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(
        field + base.scale.time(0.01) * rhs_scaled,
        field + 0.01 * rhs_highway,
        rtol=1e-11,
        atol=1e-11,
    )


def test_surveyed_arena_geometry_is_loaded_once(tmp_path):
    arena = tmp_path / "arena.yaml"
    arena.write_text(
        "frame_id: map\n"
        "lanes:\n"
        "  width: 0.48\n"
        "  centers: [0.48, 0.0, -0.48]\n"
        "route:\n"
        "  mission_goal_x: 5.56\n",
        encoding="utf-8",
    )
    config = deployment_config_for_arena(str(arena))
    assert config.arena.lane_width == 0.48
    assert config.arena.lane_centers == (0.48, 0.0, -0.48)
    assert config.arena.mission_goal_x == 5.56
    assert config.pde == default_deployment_config().pde


def test_surveyed_arena_rejects_wrong_frame(tmp_path):
    arena = tmp_path / "arena.yaml"
    arena.write_text("frame_id: odom\n", encoding="utf-8")
    try:
        deployment_config_for_arena(str(arena))
    except ValueError as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("wrong-frame arena was accepted")
