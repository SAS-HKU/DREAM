import pytest

from dream_limo.core.decision import IDEAMDREAMDecision
from dream_limo.core.occlusion import LidarShadowBuilder, rectangle_polygon, simulate_polygon_scan
from dream_limo.core.risk_field import DREAMRiskField, NumericalStabilityError
from dream_limo.core.types import EgoState, Vehicle
from dream_limo.limo_scale import default_deployment_config, get_preset


def warmed_field():
    config = default_deployment_config()
    field = DREAMRiskField(config)
    ego = EgoState(0.35, 0.45, 0.0, 0.5, lane_index=0)
    truck_poly = rectangle_polygon("truck", 2.4, 0.0, 1.2, 0.24)
    scan = simulate_polygon_scan((0.45, 0.45, 0.0), [truck_poly])
    mask = LidarShadowBuilder(maximum_shadow_range=6.0).build(
        field.X, field.Y, field.road_mask, scan, [truck_poly]
    )
    vehicles = [
        Vehicle("truck", 2.4, 0.0, vehicle_class="truck", length=1.2, width=0.24),
    ]
    field.warmup(vehicles, ego, mask)
    return config, field, ego, vehicles, mask


def test_dynamic_cfl_replaces_unsafe_three_substeps():
    _, field, ego, vehicles, mask = warmed_field()
    assert field.last_digest.substeps > 3
    with pytest.raises(NumericalStabilityError):
        field.step(vehicles, ego, mask, substeps=3)
    assert field.last_digest.raw_minimum >= -1e-6


def test_live_veto_uses_explicit_lane_center_sign():
    config, field, ego, vehicles, _ = warmed_field()
    decision = IDEAMDREAMDecision(config, blocker_trigger_distance=4.0)
    result = decision.decide(ego, vehicles, field, get_preset("balanced"), requested_lane=1)
    assert result.vetoed
    assert result.selected_lane == 0
    assert result.maneuver == "K"
    assert result.risk_score > get_preset("balanced").decision_threshold
    baseline = decision.decide(ego, vehicles, field, get_preset("baseline"), requested_lane=1)
    assert not baseline.vetoed
    assert baseline.selected_lane == 1
    assert baseline.maneuver == "R"
