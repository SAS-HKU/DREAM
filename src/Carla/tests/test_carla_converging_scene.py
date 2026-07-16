import copy
from pathlib import Path

from carla_converging_scene import (
    construction_hash,
    load_template,
    resolve_scene,
)


TEMPLATE = (
    Path(__file__).resolve().parents[1]
    / "carla_converging_overtake_manifest.json"
)


def test_resolution_is_deterministic_and_seed_changes_physics():
    template = load_template(TEMPLATE)
    first = resolve_scene(template, 101)
    repeated = resolve_scene(template, 101)
    other = resolve_scene(template, 102)

    assert first == repeated
    assert construction_hash(first) == first["scene_construction"][
        "construction_hash_sha256"
    ]
    assert construction_hash(first) != construction_hash(other)
    assert first["actors"]["ego"]["s_m"] != other["actors"]["ego"]["s_m"]
    assert first["actors"]["ego"]["speed_mps"] != other["actors"]["ego"]["speed_mps"]
    assert first["actors"]["latent_vehicle"]["cut_in_start_s"] != other["actors"][
        "latent_vehicle"
    ]["cut_in_start_s"]
    first_idm = first["actors"]["followers"][0]["idm"]
    other_idm = other["actors"]["followers"][0]["idm"]
    assert first_idm != other_idm


def test_resolved_scene_expresses_converging_overtake_and_idm_traffic():
    manifest = resolve_scene(load_template(TEMPLATE), 999)
    actors = manifest["actors"]
    truck = actors["occluder"]
    ego = actors["ego"]
    hidden = actors["latent_vehicle"]
    factors = manifest["scene_construction"]["realized_factors"]

    assert truck["blueprint"] == "vehicle.carlamotors.firetruck"
    assert truck["longitudinal_model"] == "IDM"
    assert truck["idm"]["exponent"] == 4.0
    assert ego["speed_mps"] > truck["speed_mps"]
    assert hidden["speed_mps"] > truck["speed_mps"]
    assert manifest["route_request"]["target_lane"] == manifest["planner_lane_map"][
        "centre"
    ]
    assert hidden["source_planner_lane"] == manifest["planner_lane_map"]["right"]
    assert hidden["target_planner_lane"] == manifest["planner_lane_map"]["centre"]
    assert hidden["target_lane_id"] == manifest["lane_map"]["centre"]
    assert factors["ego_clearance_ahead_at_lane_change_start_m"] >= 10.0
    assert factors["hidden_clearance_ahead_at_lane_change_start_m"] >= 10.0

    assert [item["label"] for item in actors["followers"]] == [
        "follower_1",
        "follower_2",
        "follower_3",
        "follower_4",
    ]
    idm_labels = {item["label"] for item in actors["idm_npcs"]}
    assert idm_labels == {"lead_left", "lead_centre", "lead_right", "rear_centre"}
    assert "rear_right" not in idm_labels
    assert {
        item["lane_name"]
        for item in actors["idm_npcs"]
        if item["traffic_role"] == "leader"
    } == {"left", "centre", "right"}
    assert all(item["longitudinal_model"] == "IDM" for item in actors["idm_npcs"])
    assert all("idm" in item for item in actors["followers"])


def test_construction_hash_ignores_labels_but_detects_physical_changes():
    manifest = resolve_scene(load_template(TEMPLATE), 17)
    digest = construction_hash(manifest)
    relabelled = copy.deepcopy(manifest)
    relabelled["scenario_id"] = "human_readable_alias"
    relabelled["scene_construction"]["seed"] = 999999
    relabelled["scene_construction"]["realized_factors"]["resolution_attempt"] = 999
    assert construction_hash(relabelled) == digest

    changed = copy.deepcopy(manifest)
    changed["actors"]["ego"]["speed_mps"] += 0.01
    assert construction_hash(changed) != digest

    changed_limit = copy.deepcopy(manifest)
    changed_limit["road_station_limit_m"] -= 1.0
    assert construction_hash(changed_limit) != digest


def test_many_seeds_resolve_without_pseudoreplication():
    template = load_template(TEMPLATE)
    manifests = [resolve_scene(template, seed) for seed in range(30, 50)]
    hashes = {
        manifest["scene_construction"]["construction_hash_sha256"]
        for manifest in manifests
    }
    positions = {
        (
            manifest["actors"]["ego"]["s_m"],
            manifest["actors"]["occluder"]["s_m"],
            manifest["actors"]["latent_vehicle"]["s_m"],
        )
        for manifest in manifests
    }
    assert len(hashes) == len(manifests)
    assert len(positions) == len(manifests)
