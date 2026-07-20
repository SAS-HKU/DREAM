from dream_limo.preflight_node import evaluate_occlusion_requirement


def perceived_world(*, shadow_cells=100, route_samples=10):
    return {
        "ready": True,
        "alignment_received": True,
        "occlusion_source": "lidar_first_return",
        "surveyed_static_geometry_used": False,
        "shadow_cells": shadow_cells,
        "shadow_route_samples": route_samples,
    }


def test_hardware_occlusion_evidence_latches_through_reveal_only_while_live():
    ready, current, observed, live = evaluate_occlusion_requirement(
        perceived_world(),
        world_status_fresh=True,
        required=True,
        latch=True,
        previously_observed=False,
    )
    assert ready and current and observed and live

    revealed = perceived_world(shadow_cells=0, route_samples=0)
    ready, current, observed, live = evaluate_occlusion_requirement(
        revealed,
        world_status_fresh=True,
        required=True,
        latch=True,
        previously_observed=observed,
    )
    assert ready and not current and observed and live

    ready, current, observed, live = evaluate_occlusion_requirement(
        revealed,
        world_status_fresh=False,
        required=True,
        latch=True,
        previously_observed=observed,
    )
    assert not ready and not current and observed and not live


def test_occlusion_cannot_latch_from_surveyed_or_unaligned_geometry():
    world = perceived_world()
    world["surveyed_static_geometry_used"] = True
    result = evaluate_occlusion_requirement(
        world,
        world_status_fresh=True,
        required=True,
        latch=True,
        previously_observed=False,
    )
    assert result == (False, False, False, False)
