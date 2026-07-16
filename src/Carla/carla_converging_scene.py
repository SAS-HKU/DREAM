#!/usr/bin/env python3
"""Resolve deterministic CARLA converging-overtake scene constructions.

The input file is a *template*, not a runnable CARLA trial manifest.  A seed is
resolved once into a fully explicit ``carla_overtaking_manifest_v1`` document.
That frozen document is then shared, byte for byte, by every controller and
occlusion-condition arm in the matched four-arm block.

Only standard-library features available in Python 3.7 are used so the module
can be imported from the CARLA environment as well as from the analysis
environment.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from pathlib import Path


TEMPLATE_SCHEMA = "carla_converging_scene_template_v1"
RESOLVED_SCHEMA = "carla_overtaking_manifest_v1"
GENERATOR_VERSION = "carla_converging_scene_resolver_v19"
CONSTRUCTION_HASH_ALGORITHM = "sha256"


class SceneResolutionError(ValueError):
    """Raised when a template or resolved construction violates the design."""


def _json_bytes(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value):
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _rounded(value):
    return round(float(value), 6)


def _segment_intersects_axis_aligned_box(start_xy, end_xy, half_length,
                                         half_width):
    """Liang--Barsky segment test for a box centred at the origin."""
    direction = (
        float(end_xy[0]) - float(start_xy[0]),
        float(end_xy[1]) - float(start_xy[1]),
    )
    t_enter = 0.0
    t_exit = 1.0
    for position, velocity, half_extent in (
        (float(start_xy[0]), direction[0], float(half_length)),
        (float(start_xy[1]), direction[1], float(half_width)),
    ):
        if abs(velocity) <= 1e-12:
            if abs(position) > half_extent:
                return False
            continue
        lower = (-half_extent - position) / velocity
        upper = (half_extent - position) / velocity
        if lower > upper:
            lower, upper = upper, lower
        t_enter = max(t_enter, lower)
        t_exit = min(t_exit, upper)
        if t_enter > t_exit:
            return False
    return True


def _nominal_full_footprint_occluded(template, ego_relative_station,
                                     hidden_relative_station, ego_speed,
                                     hidden_speed, truck_speed):
    """Conservative pre-CARLA projection gate for the complete hidden footprint."""
    prototypes = template["actor_prototypes"]
    truck_dims = prototypes["occluder"]["nominal_dimensions_m"]
    hidden_dims = prototypes["latent_vehicle"]["nominal_dimensions_m"]
    lane_width = float(template.get("nominal_lane_width_m", 3.5))
    camera_forward = float(template["rgb_camera"].get("x_m", 0.0))
    qualification = template["qualification"]
    horizon = float(qualification["minimum_initial_occlusion_s"])
    inset = float(qualification.get("nominal_occluder_projection_inset_m", 0.10))
    truck_half_length = 0.5 * float(truck_dims["length"]) - inset
    truck_half_width = 0.5 * float(truck_dims["width"]) - inset
    hidden_half_length = 0.5 * float(hidden_dims["length"])
    hidden_half_width = 0.5 * float(hidden_dims["width"])
    sample_coefficients = (
        (0.0, 0.0),
        (-1.0, -1.0), (-1.0, 1.0),
        (1.0, -1.0), (1.0, 1.0),
        (-1.0, 0.0), (1.0, 0.0),
        (0.0, -1.0), (0.0, 1.0),
    )
    evaluation_start = float(
        template["semantic_lidar"].get("reveal_arming_delay_s", 0.0)
    )
    times = [evaluation_start, 0.50, horizon]
    for time_s in times:
        ego_relative = (
            float(ego_relative_station)
            + (float(ego_speed) - float(truck_speed)) * time_s
        )
        hidden_relative = (
            float(hidden_relative_station)
            + (float(hidden_speed) - float(truck_speed)) * time_s
        )
        camera_xy = (ego_relative + camera_forward, lane_width)
        for longitudinal, lateral in sample_coefficients:
            hidden_point = (
                hidden_relative + longitudinal * hidden_half_length,
                -lane_width + lateral * hidden_half_width,
            )
            if not _segment_intersects_axis_aligned_box(
                camera_xy,
                hidden_point,
                truck_half_length,
                truck_half_width,
            ):
                return False
    return True


def _require_range(sampling, key):
    value = sampling.get(key)
    if not isinstance(value, list) or len(value) != 2:
        raise SceneResolutionError("sampling.{} must be a two-element list".format(key))
    lower = float(value[0])
    upper = float(value[1])
    if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
        raise SceneResolutionError("sampling.{} is not a finite ordered range".format(key))
    return lower, upper


def _uniform(rng, sampling, key):
    lower, upper = _require_range(sampling, key)
    return rng.uniform(lower, upper)


def _clip(value, lower, upper):
    return max(float(lower), min(float(upper), float(value)))


def _actor_spec(prototype, label, lane_id, station, speed, color):
    result = {
        "label": str(label),
        "blueprint": str(prototype["blueprint"]),
        "lane_id": int(lane_id),
        "s_m": _rounded(station),
        "speed_mps": _rounded(speed),
        "color": str(color),
    }
    if prototype.get("nominal_dimensions_m") is not None:
        result["nominal_dimensions_m"] = copy.deepcopy(
            prototype["nominal_dimensions_m"]
        )
    return result


def _sample_idm(rng, sampling, desired_speed):
    return {
        "desired_speed_mps": _rounded(desired_speed),
        "time_headway_s": _rounded(_uniform(rng, sampling, "idm_time_headway_s")),
        "minimum_gap_m": _rounded(_uniform(rng, sampling, "idm_minimum_gap_m")),
        "maximum_acceleration_mps2": _rounded(
            _uniform(rng, sampling, "idm_maximum_acceleration_mps2")
        ),
        "comfortable_deceleration_mps2": _rounded(
            _uniform(rng, sampling, "idm_comfortable_deceleration_mps2")
        ),
        "exponent": 4.0,
    }


def _traffic_actor(
    rng,
    template,
    label,
    traffic_role,
    lane_name,
    station,
    speed,
    desired_speed,
    color,
):
    prototypes = template["actor_prototypes"]
    pool = prototypes["traffic_blueprints"]
    if not pool:
        raise SceneResolutionError("actor_prototypes.traffic_blueprints cannot be empty")
    result = {
        "label": str(label),
        "traffic_role": str(traffic_role),
        "longitudinal_model": "IDM",
        "blueprint": str(rng.choice(pool)),
        "lane_id": int(template["lane_map"][lane_name]),
        "lane_name": str(lane_name),
        "s_m": _rounded(station),
        "speed_mps": _rounded(speed),
        "color": str(color),
        "idm": _sample_idm(rng, template["sampling"], desired_speed),
    }
    return result


def _follower_chain(
    rng,
    template,
    lane_name,
    leader_label,
    leader_station,
    leader_speed,
    count,
    label_prefix,
    colors,
):
    sampling = template["sampling"]
    station = float(leader_station)
    leader_length = 4.9
    chain = []
    for index in range(1, int(count) + 1):
        desired_speed = _clip(
            float(leader_speed) + _uniform(rng, sampling, "follower_desired_speed_offset_mps"),
            20.0,
            32.0,
        )
        profile = _sample_idm(rng, sampling, desired_speed)
        speed = _clip(
            float(leader_speed) + _uniform(rng, sampling, "follower_initial_speed_offset_mps"),
            15.0,
            desired_speed,
        )
        follower_length = 4.8
        equilibrium_spacing = (
            0.5 * (leader_length + follower_length)
            + float(profile["minimum_gap_m"])
            + speed * float(profile["time_headway_s"])
        )
        spacing_scale = _uniform(rng, sampling, "follower_spacing_scale")
        station -= equilibrium_spacing * spacing_scale
        result = {
            "label": "{}_{}".format(label_prefix, index),
            "traffic_role": "follower",
            "leader_at_initialization": str(leader_label if index == 1 else chain[-1]["label"]),
            "longitudinal_model": "IDM",
            "blueprint": str(rng.choice(template["actor_prototypes"]["traffic_blueprints"])),
            "lane_id": int(template["lane_map"][lane_name]),
            "lane_name": str(lane_name),
            "s_m": _rounded(station),
            "speed_mps": _rounded(speed),
            "color": str(colors[(index - 1) % len(colors)]),
            "idm": profile,
        }
        chain.append(result)
        leader_length = follower_length
        leader_speed = speed
    return chain


def construction_payload(manifest):
    """Return only fields that define the physical/protocol construction.

    Human-readable identifiers and generator bookkeeping are intentionally
    excluded.  Thus two files with identical realized physics receive the same
    hash even if they have different filenames or scene labels.
    """

    keys = (
        "map",
        "road_id",
        "road_station_limit_m",
        "duration_s",
        "physics_dt_s",
        "measurement_preroll_s",
        "weather",
        "lane_map",
        "planner_lane_map",
        "route_request",
        "actors",
        "traffic_control",
        "semantic_lidar",
        "rgb_camera",
        "evaluation_metrics",
        "qualification",
    )
    missing = [key for key in keys if key not in manifest]
    if missing:
        raise SceneResolutionError(
            "resolved manifest is missing construction fields: {}".format(
                ", ".join(missing)
            )
        )
    return {key: copy.deepcopy(manifest[key]) for key in keys}


def construction_hash(manifest):
    """Return the SHA-256 digest of the realized construction."""

    return _sha256(construction_payload(manifest))


def _validate_template(template):
    if template.get("schema_version") != TEMPLATE_SCHEMA:
        raise SceneResolutionError(
            "unsupported scene template schema: {!r}".format(template.get("schema_version"))
        )
    required = (
        "scenario_family",
        "map",
        "road_id",
        "duration_s",
        "physics_dt_s",
        "measurement_preroll_s",
        "weather",
        "lane_map",
        "planner_lane_map",
        "actor_prototypes",
        "traffic_layout",
        "sampling",
        "semantic_lidar",
        "rgb_camera",
        "evaluation_metrics",
        "qualification",
    )
    missing = [key for key in required if key not in template]
    if missing:
        raise SceneResolutionError(
            "scene template is missing fields: {}".format(", ".join(missing))
        )
    for name in ("left", "centre", "right"):
        if name not in template["lane_map"]:
            raise SceneResolutionError("lane_map is missing {!r}".format(name))
    if template["actor_prototypes"]["occluder"].get("blueprint") != "vehicle.carlamotors.firetruck":
        raise SceneResolutionError(
            "the converging-overtake template must use vehicle.carlamotors.firetruck"
        )


def _validate_resolved(manifest):
    actors = manifest["actors"]
    ego = actors["ego"]
    truck = actors["occluder"]
    hidden = actors["latent_vehicle"]
    centre_lane = int(manifest["lane_map"]["centre"])
    if float(ego["speed_mps"]) <= float(truck["speed_mps"]):
        raise SceneResolutionError("ego is not overtaking the occluder")
    if float(hidden["speed_mps"]) <= float(truck["speed_mps"]):
        raise SceneResolutionError("latent vehicle is not overtaking the occluder")
    if int(hidden["target_lane_id"]) != centre_lane:
        raise SceneResolutionError("latent vehicle does not converge into the centre lane")
    if int(manifest["route_request"]["target_lane"]) != int(
        manifest["planner_lane_map"]["centre"]
    ):
        raise SceneResolutionError("ego route request does not target the centre lane")

    factors = manifest["scene_construction"]["realized_factors"]
    minimum_pass = float(manifest["qualification"]["minimum_pass_clearance_m"])
    if float(factors["ego_clearance_ahead_at_lane_change_start_m"]) < minimum_pass:
        raise SceneResolutionError("ego starts its lane change before clearing the firetruck")
    if float(factors["hidden_clearance_ahead_at_lane_change_start_m"]) < minimum_pass:
        raise SceneResolutionError("latent vehicle starts its lane change before clearing the firetruck")
    if float(factors["conflict_station_m"]) <= (
        float(truck["s_m"])
        + float(truck["speed_mps"]) * float(factors["nominal_conflict_time_s"])
    ):
        raise SceneResolutionError("nominal conflict region is not ahead of the firetruck")

    labels = ["ego", "occluder", "latent_vehicle"]
    labels.extend(item["label"] for item in actors["followers"])
    labels.extend(item["label"] for item in actors["idm_npcs"])
    if len(labels) != len(set(labels)):
        raise SceneResolutionError("actor labels must be unique")
    leaders = {
        item["lane_name"]
        for item in actors["idm_npcs"]
        if item.get("traffic_role") == "leader"
    }
    if leaders != {"left", "centre", "right"}:
        raise SceneResolutionError("one front IDM leader is required in every lane")

    preroll = float(manifest["measurement_preroll_s"])
    all_specs = (
        [ego, truck, hidden]
        + list(actors["followers"])
        + list(actors["idm_npcs"])
    )
    for spec in all_specs:
        if float(spec["s_m"]) - preroll * float(spec["speed_mps"]) <= 2.0:
            raise SceneResolutionError(
                "measurement preroll places {} before the road origin".format(
                    spec.get("label", spec.get("blueprint", "actor"))
                )
            )

    for lane_id in manifest["lane_map"].values():
        lane_specs = [spec for spec in all_specs if int(spec["lane_id"]) == int(lane_id)]
        lane_specs.sort(key=lambda item: float(item["s_m"]))
        for rear, front in zip(lane_specs, lane_specs[1:]):
            if float(front["s_m"]) - float(rear["s_m"]) < 7.5:
                raise SceneResolutionError(
                    "initial same-lane spacing is too small between {} and {}".format(
                        rear.get("label", rear.get("blueprint")),
                        front.get("label", front.get("blueprint")),
                    )
                )


def _resolve_attempt(template, seed, rng, attempt):
    sampling = template["sampling"]
    lanes = template["lane_map"]
    prototypes = template["actor_prototypes"]
    duration = float(template["duration_s"])

    truck_station = _uniform(rng, sampling, "occluder_station_m")
    truck_speed = _uniform(rng, sampling, "occluder_speed_mps")
    ego_speed = truck_speed + _uniform(rng, sampling, "ego_speed_advantage_mps")
    hidden_speed = truck_speed + _uniform(rng, sampling, "hidden_speed_advantage_mps")
    conflict_time = _uniform(rng, sampling, "nominal_conflict_time_s")
    arrival_offset = _uniform(rng, sampling, "arrival_time_offset_hidden_minus_ego_s")
    conflict_lead = _uniform(rng, sampling, "conflict_clearance_ahead_occluder_m")
    ego_lc_duration = _uniform(rng, sampling, "ego_lane_change_duration_s")
    hidden_lc_duration = _uniform(rng, sampling, "hidden_lane_change_duration_s")

    ego_arrival_time = conflict_time - 0.5 * arrival_offset
    hidden_arrival_time = conflict_time + 0.5 * arrival_offset
    conflict_station = truck_station + truck_speed * conflict_time + conflict_lead
    ego_station = conflict_station - ego_speed * ego_arrival_time
    hidden_station = conflict_station - hidden_speed * hidden_arrival_time
    ego_lc_start = ego_arrival_time - 0.5 * ego_lc_duration
    hidden_lc_start = hidden_arrival_time - 0.5 * hidden_lc_duration
    ego_start_clearance = (
        ego_station
        - truck_station
        + (ego_speed - truck_speed) * ego_lc_start
    )
    hidden_start_clearance = (
        hidden_station
        - truck_station
        + (hidden_speed - truck_speed) * hidden_lc_start
    )

    # Place the latent vehicle just ahead of the firetruck after it has nearly
    # completed the right-side pass.  From the ego vehicle, which is still
    # behind the truck in the left lane, this makes the firetruck lie between
    # the camera/LiDAR origin and the latent vehicle instead of leaving the
    # latter's rear quarter exposed beside the truck.  CARLA subsequently
    # applies the authoritative full-footprint sensor-occlusion qualification.
    ego_relative_station = ego_station - truck_station
    hidden_relative_station = hidden_station - truck_station
    if not (-25.0 <= ego_relative_station <= -13.5):
        raise SceneResolutionError("sampled ego is not initially behind the firetruck")
    if not (2.0 <= hidden_relative_station <= 6.0):
        raise SceneResolutionError("sampled hidden car is outside the occlusion neighbourhood")
    if hidden_relative_station - ego_relative_station < 8.0:
        raise SceneResolutionError(
            "sampled hidden car is not ahead of the ego within the truck shadow"
        )
    if not _nominal_full_footprint_occluded(
        template,
        ego_relative_station,
        hidden_relative_station,
        ego_speed,
        hidden_speed,
        truck_speed,
    ):
        raise SceneResolutionError(
            "sampled geometry does not occlude the full hidden footprint through "
            "the initial qualification horizon"
        )
    if min(ego_lc_start, hidden_lc_start) < 1.5:
        raise SceneResolutionError("sampled lane change starts too early")
    if max(
        ego_lc_start + ego_lc_duration,
        hidden_lc_start + hidden_lc_duration,
    ) > duration - 0.4:
        raise SceneResolutionError("sampled lane change does not finish inside the episode")

    ego = _actor_spec(
        prototypes["ego"], "ego", lanes["left"], ego_station, ego_speed, "0,80,255"
    )
    truck = _actor_spec(
        prototypes["occluder"],
        "occluder",
        lanes["centre"],
        truck_station,
        truck_speed,
        "220,45,25",
    )
    truck["longitudinal_model"] = "IDM"
    truck["idm"] = _sample_idm(rng, sampling, truck_speed)
    hidden = _actor_spec(
        prototypes["latent_vehicle"],
        "latent_vehicle",
        lanes["right"],
        hidden_station,
        hidden_speed,
        "220,20,60",
    )
    hidden.update(
        {
            "cut_in_start_s": _rounded(hidden_lc_start),
            "cut_in_duration_s": _rounded(hidden_lc_duration),
            "target_lane_id": int(lanes["centre"]),
            "source_planner_lane": int(template["planner_lane_map"]["right"]),
            "target_planner_lane": int(template["planner_lane_map"]["centre"]),
            "nominal_conflict_station_m": _rounded(conflict_station),
        }
    )

    front_offset_range = _require_range(sampling, "front_npc_offset_after_conflict_m")
    front_specs = []
    front_config = (
        ("lead_left", "left", ego_speed, "25,180,95"),
        ("lead_centre", "centre", truck_speed + 2.0, "180,180,40"),
        ("lead_right", "right", hidden_speed, "155,95,210"),
    )
    for label, lane_name, lane_reference_speed, color in front_config:
        station = conflict_station + rng.uniform(*front_offset_range)
        maximum_road_speed = (float(template["road_station_limit_m"]) - station) / duration
        desired_speed = _clip(
            lane_reference_speed
            + _uniform(rng, sampling, "front_npc_desired_speed_offset_mps"),
            18.0,
            min(32.0, maximum_road_speed),
        )
        if maximum_road_speed < 18.0:
            raise SceneResolutionError("front NPC cannot remain on the declared road segment")
        speed = _clip(
            desired_speed + _uniform(rng, sampling, "front_npc_initial_speed_offset_mps"),
            16.0,
            maximum_road_speed,
        )
        front_specs.append(
            _traffic_actor(
                rng,
                template,
                label,
                "leader",
                lane_name,
                station,
                speed,
                desired_speed,
                color,
            )
        )

    counts = template["traffic_layout"]["trailing_counts"]
    followers = _follower_chain(
        rng,
        template,
        "left",
        "ego",
        ego_station,
        ego_speed,
        int(counts["left"]),
        "follower",
        ["85,85,85", "105,105,105", "125,125,125", "145,145,145"],
    )
    rear_centre = _follower_chain(
        rng,
        template,
        "centre",
        "occluder",
        truck_station,
        truck_speed,
        int(counts["centre"]),
        "rear_centre",
        ["75,115,145", "90,130,160"],
    )
    if len(rear_centre) == 1:
        rear_centre[0]["label"] = "rear_centre"
    rear_right = _follower_chain(
        rng,
        template,
        "right",
        "latent_vehicle",
        hidden_station,
        hidden_speed,
        int(counts["right"]),
        "rear_right",
        ["135,90,90", "155,105,105"],
    )
    if len(rear_right) == 1:
        rear_right[0]["label"] = "rear_right"
    idm_npcs = list(front_specs) + rear_centre + rear_right

    realized_factors = {
        "resolution_attempt": int(attempt),
        "occluder_station_m": _rounded(truck_station),
        "occluder_speed_mps": _rounded(truck_speed),
        "ego_station_m": _rounded(ego_station),
        "ego_speed_mps": _rounded(ego_speed),
        "hidden_station_m": _rounded(hidden_station),
        "hidden_speed_mps": _rounded(hidden_speed),
        "nominal_conflict_time_s": _rounded(conflict_time),
        "arrival_time_offset_hidden_minus_ego_s": _rounded(arrival_offset),
        "conflict_station_m": _rounded(conflict_station),
        "conflict_clearance_ahead_occluder_m": _rounded(conflict_lead),
        "ego_lane_change_start_s": _rounded(ego_lc_start),
        "ego_lane_change_duration_s": _rounded(ego_lc_duration),
        "hidden_lane_change_start_s": _rounded(hidden_lc_start),
        "hidden_lane_change_duration_s": _rounded(hidden_lc_duration),
        "ego_initial_station_relative_to_occluder_m": _rounded(ego_relative_station),
        "hidden_initial_station_relative_to_occluder_m": _rounded(hidden_relative_station),
        "ego_clearance_ahead_at_lane_change_start_m": _rounded(ego_start_clearance),
        "hidden_clearance_ahead_at_lane_change_start_m": _rounded(hidden_start_clearance),
        "idm_npc_count": 1 + len(followers) + len(idm_npcs),
        "legacy_left_follower_count": len(followers),
    }

    manifest = {
        "schema_version": RESOLVED_SCHEMA,
        "scenario_id": "pending",
        "scenario_family": str(template["scenario_family"]),
        "description": str(template["description"]),
        "map": str(template["map"]),
        "road_id": int(template["road_id"]),
        "road_station_limit_m": float(template["road_station_limit_m"]),
        "duration_s": float(template["duration_s"]),
        "physics_dt_s": float(template["physics_dt_s"]),
        "measurement_preroll_s": float(template["measurement_preroll_s"]),
        "weather": str(template["weather"]),
        "lane_map": copy.deepcopy(template["lane_map"]),
        "planner_lane_map": copy.deepcopy(template["planner_lane_map"]),
        "route_request": {
            "target_lane": int(template["planner_lane_map"]["centre"]),
            "start_time_s": _rounded(ego_lc_start),
            "end_time_s": _rounded(duration - 0.2),
            "nominal_lane_change_duration_s": _rounded(ego_lc_duration),
            "nominal_conflict_time_s": _rounded(ego_arrival_time),
            "conflict_station_m": _rounded(conflict_station),
        },
        "actors": {
            "ego": ego,
            "occluder": truck,
            "latent_vehicle": hidden,
            "blocker": {
                "enabled": False,
                "blueprint": "vehicle.audi.a2",
                "lane_id": int(lanes["left"]),
                "s_m": _rounded(conflict_station + 80.0),
                "speed_mps": _rounded(ego_speed),
                "color": "250,210,0",
            },
            "followers": followers,
            "idm_npcs": idm_npcs,
        },
        "traffic_control": copy.deepcopy(template["traffic_control"]),
        "semantic_lidar": copy.deepcopy(template["semantic_lidar"]),
        "rgb_camera": copy.deepcopy(template["rgb_camera"]),
        "evaluation_metrics": copy.deepcopy(template["evaluation_metrics"]),
        "qualification": copy.deepcopy(template["qualification"]),
        "scene_construction": {
            "generator_version": GENERATOR_VERSION,
            "seed": int(seed),
            "hash_algorithm": CONSTRUCTION_HASH_ALGORITHM,
            "construction_hash_sha256": "pending",
            "realized_factors": realized_factors,
        },
        "notes": copy.deepcopy(template.get("notes", [])),
    }
    digest = construction_hash(manifest)
    manifest["scenario_id"] = "{}_seed{:06d}_{}".format(
        template["scenario_family"], int(seed), digest[:12]
    )
    manifest["scene_construction"]["construction_hash_sha256"] = digest
    _validate_resolved(manifest)
    if construction_hash(manifest) != digest:
        raise SceneResolutionError("construction hash is not stable")
    return manifest


def resolve_scene(template, seed, max_attempts=1000):
    """Resolve ``seed`` to one explicit, validated physical construction."""

    _validate_template(template)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise SceneResolutionError("seed must be an integer")
    if max_attempts < 1:
        raise SceneResolutionError("max_attempts must be positive")
    rng = random.Random(int(seed))
    failures = []
    for attempt in range(1, int(max_attempts) + 1):
        try:
            return _resolve_attempt(template, seed, rng, attempt)
        except SceneResolutionError as error:
            failures.append(str(error))
    tail = "; ".join(failures[-3:])
    raise SceneResolutionError(
        "seed {} did not yield a valid construction in {} attempts ({})".format(
            seed, max_attempts, tail
        )
    )


def load_template(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        template = json.load(handle)
    _validate_template(template)
    return template


def resolve_file(template_path, seed):
    """Load a template path and resolve a seed (convenience API)."""

    return resolve_scene(load_template(template_path), int(seed))


__all__ = [
    "CONSTRUCTION_HASH_ALGORITHM",
    "GENERATOR_VERSION",
    "RESOLVED_SCHEMA",
    "TEMPLATE_SCHEMA",
    "SceneResolutionError",
    "construction_hash",
    "construction_payload",
    "load_template",
    "resolve_file",
    "resolve_scene",
]
