import math
from types import SimpleNamespace

import numpy as np
import pytest

import carla_external_planner as planner_module


def _actor(**overrides):
    actor = {
        "actor_id": 17,
        "role": "latent_vehicle",
        "station_m": 120.0,
        "lateral_error_m": 0.65,
        "heading_error_rad": math.radians(30.0),
        "local_x_m": -80.0,
        "local_y_m": -208.1,
        "local_yaw_rad": math.radians(30.0),
        "lane_index": 2,
        "speed_mps": 99.0,
        "body_vx_mps": 10.0,
        "body_vy_mps": 2.0,
        "longitudinal_accel_mps2": -1.5,
        "length_m": 5.1,
        "width_m": 1.9,
    }
    actor.update(overrides)
    return actor


def _observation(*, simulation_time_s=0.0, ego_overrides=None):
    ego = _actor(
        actor_id=1,
        role="ego",
        station_m=100.0,
        local_x_m=-100.0,
        local_y_m=-201.75,
        lateral_error_m=0.0,
        heading_error_rad=0.0,
        local_yaw_rad=0.0,
        lane_index=0,
        speed_mps=30.0,
        body_vx_mps=30.0,
        body_vy_mps=0.0,
        length_m=4.892,
        width_m=1.837,
    )
    ego.update(ego_overrides or {})
    return {
        "simulation_time_s": float(simulation_time_s),
        "ego": ego,
        "visible_actors": [],
    }


def test_lane_row_uses_road_longitudinal_velocity_and_retains_cut_in_pose():
    actor = _actor()
    row = planner_module._actor_row(actor)
    expected_vx = 10.0 * math.cos(math.radians(30.0)) - 2.0 * math.sin(
        math.radians(30.0)
    )

    assert row.shape == (8,)
    assert row[1] == pytest.approx(actor["lateral_error_m"])
    assert row[2] == pytest.approx(actor["heading_error_rad"])
    assert row[5] == pytest.approx(actor["local_yaw_rad"])
    assert row[6] == pytest.approx(expected_vx)
    assert row[6] != pytest.approx(actor["speed_mps"])
    assert row[7] == pytest.approx(-1.5 * math.cos(math.radians(30.0)))


def test_lane_traffic_inserts_overlapping_actor_into_every_occupied_lane():
    actor = _actor(lane_index=2, occupied_lane_indices=[1, 2])

    traffic = planner_module._lane_traffic([actor])

    assert traffic.left.shape == (0, 8)
    assert traffic.centre.shape == (1, 8)
    assert traffic.right.shape == (1, 8)
    assert traffic.centre[0] == pytest.approx(planner_module._actor_row(actor))
    assert traffic.right[0] == pytest.approx(planner_module._actor_row(actor))


def test_lane_traffic_deduplicates_repeated_occupied_lane_indices():
    actor = _actor(occupied_lane_indices=[1, 1, 2, 2, 1])

    traffic = planner_module._lane_traffic([actor])

    assert traffic.centre.shape == (1, 8)
    assert traffic.right.shape == (1, 8)


@pytest.mark.parametrize("occupied_lanes", [None, []])
def test_lane_traffic_falls_back_to_legacy_lane_index(occupied_lanes):
    actor = _actor()
    if occupied_lanes is not None:
        actor["occupied_lane_indices"] = occupied_lanes

    traffic = planner_module._lane_traffic([actor])

    assert traffic.left.shape == (0, 8)
    assert traffic.centre.shape == (0, 8)
    assert traffic.right.shape == (1, 8)


def test_drift_actor_uses_measured_lateral_velocity_and_physical_geometry():
    actor = _actor()
    vehicle = planner_module._drift_vehicle(actor)
    heading = math.radians(30.0)

    assert vehicle["vx"] == pytest.approx(10.0 * math.cos(heading) - 2.0 * math.sin(heading))
    assert vehicle["vy"] == pytest.approx(10.0 * math.sin(heading) + 2.0 * math.cos(heading))
    assert vehicle["heading"] == pytest.approx(heading)
    assert vehicle["length"] == pytest.approx(5.1)
    assert vehicle["width"] == pytest.approx(1.9)


def test_carla_ego_geometry_is_applied_to_drift_mpc_and_gap_utility():
    observation = _observation()
    ego_vehicle = planner_module._ego_drift_vehicle(observation)
    assert ego_vehicle["length"] == pytest.approx(4.892)
    assert ego_vehicle["width"] == pytest.approx(1.837)

    mpc = SimpleNamespace(vehicle_length=3.5, vehicle_width=1.2)
    utility = SimpleNamespace(vehicle_width=1.2, l=3.5, l_diag=math.hypot(3.5, 1.2))
    arm = SimpleNamespace(controller=SimpleNamespace(mpc=mpc), utils=utility)
    planner_module._configure_arm_ego_geometry(arm, length_m=4.892, width_m=1.837)

    assert mpc.vehicle_length == pytest.approx(4.892)
    assert mpc.vehicle_width == pytest.approx(1.837)
    assert utility.vehicle_width == pytest.approx(1.837)
    assert utility.l == pytest.approx(4.892)
    assert utility.l_diag == pytest.approx(math.hypot(4.892, 1.837))


def test_dream_factory_receives_carla_mpc_geometry(monkeypatch):
    captured = {}

    class _Drift:
        def reset(self):
            captured["reset"] = True

    fake_arm = SimpleNamespace(
        controller=SimpleNamespace(
            mpc=SimpleNamespace(vehicle_length=3.5, vehicle_width=1.2),
            drift=_Drift(),
        ),
        utils=SimpleNamespace(vehicle_width=1.2, l=3.5, l_diag=math.hypot(3.5, 1.2)),
    )

    def fake_factory(*args, **kwargs):
        captured.update(kwargs)
        return fake_arm

    monkeypatch.setattr(planner_module, "create_prideam_episode_arm", fake_factory)
    service = planner_module.ExternalPhysicsPlanner("DREAM", {})
    service._ensure_arm(_observation())

    assert captured["mpc_overrides"] == {
        "vehicle_length": pytest.approx(4.892),
        "vehicle_width": pytest.approx(1.837),
    }
    assert captured["reset"] is True
    assert service.ego_geometry_m == pytest.approx((4.892, 1.837))


@pytest.mark.parametrize(
    ("controller", "expected_source"),
    [
        ("DREAM", None),
        ("ADA", planner_module.compute_Q_ADA),
        ("APF", planner_module.compute_Q_APF),
    ],
)
def test_field_controller_selects_declared_source(controller, expected_source):
    service = planner_module.ExternalPhysicsPlanner(controller, {})
    assert service._source_function() is expected_source


def test_oa_cmpc_is_not_an_accepted_carla_controller():
    with pytest.raises(ValueError, match="controller must be one of"):
        planner_module.ExternalPhysicsPlanner("OA-CMPC", {})


def test_zero_previous_field_timestamp_advances_full_elapsed_interval():
    class _Drift:
        def __init__(self):
            self.steps = []

        def step(self, vehicles, ego, *, dt, substeps):
            self.steps.append((vehicles, ego, dt, substeps))

    drift = _Drift()
    service = planner_module.ExternalPhysicsPlanner("DREAM", {})
    service.arm = SimpleNamespace(controller=SimpleNamespace(drift=drift))
    service.field_warmed = True
    service.last_field_time_s = 0.0

    service._update_field(_observation(simulation_time_s=0.35))

    assert len(drift.steps) == 4
    assert sum(step[2] for step in drift.steps) == pytest.approx(0.35)
    assert all(step[2] <= planner_module.PLANNER_DT_S for step in drift.steps)
    assert service.last_field_time_s == pytest.approx(0.35)


@pytest.mark.parametrize(
    "ego_overrides",
    [
        {"length_m": 0.0},
        {"width_m": float("nan")},
        {"length_m": True},
    ],
)
def test_invalid_carla_ego_geometry_is_rejected(ego_overrides):
    with pytest.raises(ValueError, match="positive finite number"):
        planner_module._ego_geometry(_observation(ego_overrides=ego_overrides))


def test_velocity_magnitude_fallback_remains_available_for_old_fixtures():
    actor = _actor()
    actor.pop("body_vx_mps")
    actor.pop("body_vy_mps")
    actor["speed_mps"] = 12.0
    local_vx, local_vy = planner_module._local_velocity_components(actor)

    assert np.asarray([local_vx, local_vy]) == pytest.approx(
        [12.0 * math.cos(actor["local_yaw_rad"]), 12.0 * math.sin(actor["local_yaw_rad"])]
    )
