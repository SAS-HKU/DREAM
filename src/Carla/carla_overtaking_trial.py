"""Run the paired CARLA box-truck overtaking/occluded-merge pilot.

Run this file with the Python 3.7 interpreter that owns the local CARLA 0.9.14
extension.  The DREAM/IDEAM optimizer is launched in the repository's modern
Python environment as a separate process and communicates through the small
versioned protocol in ``Integration/carla_protocol.py``.

The scientific comparison has two matched conditions:

``true_threat``
    A right-lane vehicle physically exists from frame zero, is initially
    blocked by the centre-lane box truck, and executes a fixed cut-in.  It is
    withheld from both MPC and DRIFT until semantic-LiDAR reveal.

``empty_shadow``
    The latent vehicle is absent; every other actor, command, route request,
    sensor, and seed is unchanged.  DREAM receives no oracle empty flag.

The supported planners are DREAM, IDEAM, ADA-field MPC-CBF, and APF-MPC-CBF.
OA-CMPC is intentionally excluded because the repository's OA adapter does
not reproduce the published dual-branch contingency optimizer.

The script records ground truth, sensor visibility, asynchronous plan age,
low-level commands, follower response, collisions, synchronized composite
frames, and a JSON summary.  It is a pilot runner, not a statistical result by
itself; aggregate inference belongs in a separate paired analysis step.
"""

from __future__ import print_function

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import queue
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback

import numpy as np


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
INTEGRATION_DIR = os.path.join(REPO_ROOT, "Integration")
if INTEGRATION_DIR not in sys.path:
    # Import the protocol as a standalone Python-3.7 module.  Importing the
    # Integration package itself would load SciPy-dependent DREAM modules.
    sys.path.insert(0, INTEGRATION_DIR)

from carla_protocol import (  # noqa: E402
    ConnectionClosed,
    ProtocolError,
    make_message,
    recv_message,
    send_message,
)

try:  # Script execution and package import use different module roots.
    from .physical_safety_metrics import (  # type: ignore
        KinematicBoxState,
        constant_velocity_ttc,
        signed_oriented_box_clearance,
    )
except (ImportError, ValueError):  # pragma: no cover - script execution path
    from physical_safety_metrics import (  # noqa: E402
        KinematicBoxState,
        constant_velocity_ttc,
        signed_oriented_box_clearance,
    )

try:
    from .carla_converging_scene import (  # type: ignore
        RESOLVED_SCHEMA,
        TEMPLATE_SCHEMA,
        construction_hash,
        resolve_scene,
    )
except (ImportError, ValueError):  # pragma: no cover - script execution path
    from carla_converging_scene import (  # noqa: E402
        RESOLVED_SCHEMA,
        TEMPLATE_SCHEMA,
        construction_hash,
        resolve_scene,
    )

try:
    import carla  # noqa: E402
except ImportError as error:  # pragma: no cover - depends on local CARLA install
    raise SystemExit(
        "CARLA Python API is unavailable. Run with the installed Python 3.7 "
        "interpreter containing carla-0.9.14. Original error: {}".format(error)
    )

try:
    import pygame  # noqa: E402
except ImportError as error:  # pragma: no cover
    raise SystemExit("pygame is required for composite rendering: {}".format(error))


CONDITIONS = ("true_threat", "empty_shadow")
CONTROLLERS = ("DREAM", "IDEAM", "ADA", "APF")
PLANNER_LANE_CENTRES_Y = (-201.75, -205.25, -208.75)
PLANNER_PATH_TRANSLATION_X = -200.0
COMPOSITE_SIZE = (1920, 1440)
RGB_PANEL_SIZE = (1920, 1080)
DRIVER_PANEL_RECT = (0, 0, 1920, 1080)
BEV_PANEL_RECT = (0, 1080, 1920, 360)
LOW_LEVEL_MAX_LATERAL_ACCEL_MPS2 = 2.5
LOW_LEVEL_WHEELBASE_M = 2.85
LOW_LEVEL_MAX_PLAN_AGE_S = 1.25


def _clip(value, lower, upper):
    return max(lower, min(upper, value))


def _wrap_angle(angle):
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def _norm_xy(vector):
    return math.hypot(float(vector.x), float(vector.y))


def _percentile(values, percentile):
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), percentile))


def _finite_min(values):
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return min(values) if values else None


def _finite_max(values):
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return max(values) if values else None


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def _write_json(path, payload):
    with open(path, "w") as handle:
        json.dump(_json_safe(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _load_manifest(path, seed):
    with open(path, "r") as handle:
        manifest = json.load(handle)
    schema = manifest.get("schema_version")
    if schema == TEMPLATE_SCHEMA:
        manifest = resolve_scene(manifest, int(seed))
    elif schema == RESOLVED_SCHEMA:
        construction = manifest.get("scene_construction")
        if construction is not None:
            if int(construction.get("seed")) != int(seed):
                raise ValueError(
                    "resolved CARLA manifest seed {} does not match --seed {}".format(
                        construction.get("seed"), seed
                    )
                )
            declared = str(construction.get("construction_hash_sha256", ""))
            calculated = construction_hash(manifest)
            if not declared or declared != calculated:
                raise ValueError(
                    "resolved CARLA manifest construction hash is missing or invalid"
                )
    else:
        raise ValueError("unsupported CARLA overtaking manifest version: {}".format(schema))
    return manifest


def _validate_execution_contract(manifest):
    traffic = manifest.get("traffic_control", {})
    expected = {
        "model": "IDM",
        "equation": "complete_idm",
        "leader_assignment": "nearest_forward_actor_in_current_lane",
        "per_tick_random_noise": False,
    }
    mismatches = [
        key for key, value in expected.items() if traffic.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            "unsupported traffic_control contract fields: {}".format(
                ", ".join(mismatches)
            )
        )
    if abs(
        float(traffic.get("update_interval_s", -1.0))
        - float(manifest["physics_dt_s"])
    ) > 1e-9:
        raise ValueError(
            "traffic_control.update_interval_s must equal physics_dt_s"
        )
    if manifest.get("scene_construction"):
        actor_specs = manifest["actors"]
        realized_roles = {
            str(spec.get("label"))
            for spec in (
                list(actor_specs.get("followers", []))
                + list(actor_specs.get("idm_npcs", []))
            )
        }
        expected_roles = {
            "follower_1", "follower_2", "follower_3", "follower_4",
            "lead_left", "lead_centre", "lead_right", "rear_centre",
        }
        if realized_roles != expected_roles:
            raise ValueError(
                "generated converging bank requires exact traffic roles {}; got {}"
                .format(sorted(expected_roles), sorted(realized_roles))
            )


class RouteFrame(object):
    """Rigid CARLA-to-DREAM frame for straight Town06 road 40."""

    def __init__(self, anchor_transform, anchor_station_m, centre_local_y):
        self.anchor_x = float(anchor_transform.location.x)
        self.anchor_y = float(anchor_transform.location.y)
        self.anchor_yaw_deg = float(anchor_transform.rotation.yaw)
        yaw = math.radians(self.anchor_yaw_deg)
        self.forward = np.asarray([math.cos(yaw), math.sin(yaw)], dtype=float)
        # CARLA is left-handed: at yaw zero, vehicle-left is negative world y.
        self.left = np.asarray([math.sin(yaw), -math.cos(yaw)], dtype=float)
        self.anchor_local_x = float(anchor_station_m) + PLANNER_PATH_TRANSLATION_X
        self.anchor_local_y = float(centre_local_y)

    def world_to_local(self, location, yaw_deg):
        displacement = np.asarray(
            [float(location.x) - self.anchor_x, float(location.y) - self.anchor_y],
            dtype=float,
        )
        local_x = self.anchor_local_x + float(np.dot(displacement, self.forward))
        local_y = self.anchor_local_y + float(np.dot(displacement, self.left))
        local_yaw = _wrap_angle(-math.radians(float(yaw_deg) - self.anchor_yaw_deg))
        return local_x, local_y, local_yaw

    def local_to_world(self, local_x, local_y, local_yaw=0.0, z=0.3):
        displacement = (
            (float(local_x) - self.anchor_local_x) * self.forward
            + (float(local_y) - self.anchor_local_y) * self.left
        )
        return carla.Transform(
            carla.Location(
                x=self.anchor_x + float(displacement[0]),
                y=self.anchor_y + float(displacement[1]),
                z=float(z),
            ),
            carla.Rotation(yaw=self.anchor_yaw_deg - math.degrees(float(local_yaw))),
        )

    def lane_index(self, local_y):
        return int(np.argmin(np.abs(np.asarray(PLANNER_LANE_CENTRES_Y) - float(local_y))))

    def actor_packet(self, actor, role):
        transform = actor.get_transform()
        velocity = actor.get_velocity()
        acceleration = actor.get_acceleration()
        yaw = math.radians(float(transform.rotation.yaw))
        forward = np.asarray([math.cos(yaw), math.sin(yaw)], dtype=float)
        right = np.asarray([-math.sin(yaw), math.cos(yaw)], dtype=float)
        bbox = actor.bounding_box
        bbox_offset = np.asarray(
            [float(bbox.location.x), float(bbox.location.y)], dtype=float
        )
        bbox_centre_xy = (
            np.asarray([float(transform.location.x), float(transform.location.y)], dtype=float)
            + bbox_offset[0] * forward
            + bbox_offset[1] * right
        )
        bbox_centre = carla.Location(
            x=float(bbox_centre_xy[0]),
            y=float(bbox_centre_xy[1]),
            z=float(transform.location.z + bbox.location.z),
        )
        local_x, local_y, local_yaw = self.world_to_local(
            bbox_centre, transform.rotation.yaw
        )
        lane_index = self.lane_index(local_y)
        lane_y = PLANNER_LANE_CENTRES_Y[lane_index]
        station = local_x - PLANNER_PATH_TRANSLATION_X
        left = np.asarray([math.sin(yaw), -math.cos(yaw)], dtype=float)
        velocity_xy = np.asarray([float(velocity.x), float(velocity.y)], dtype=float)
        acceleration_xy = np.asarray([float(acceleration.x), float(acceleration.y)], dtype=float)
        extent = bbox.extent
        length_m = 2.0 * float(extent.x)
        width_m = 2.0 * float(extent.y)
        projected_half_width = (
            0.5 * length_m * abs(math.sin(local_yaw))
            + 0.5 * width_m * abs(math.cos(local_yaw))
        )
        occupied_lanes = [
            index
            for index, centre_y in enumerate(PLANNER_LANE_CENTRES_Y)
            if abs(float(local_y) - float(centre_y)) <= 1.75 + projected_half_width
        ]
        if lane_index not in occupied_lanes:
            occupied_lanes.append(lane_index)
        occupied_lanes = sorted(set(occupied_lanes))
        return {
            "actor_id": int(actor.id),
            "role": str(role),
            "class": "truck" if role == "occluder" else "car",
            "local_x_m": local_x,
            "local_y_m": local_y,
            "local_yaw_rad": local_yaw,
            "station_m": station,
            "lane_index": lane_index,
            "lateral_error_m": local_y - lane_y,
            "heading_error_rad": local_yaw,
            "speed_mps": float(np.linalg.norm(velocity_xy)),
            "body_vx_mps": float(np.dot(velocity_xy, forward)),
            "body_vy_mps": float(np.dot(velocity_xy, left)),
            "longitudinal_accel_mps2": float(np.dot(acceleration_xy, forward)),
            "length_m": length_m,
            "width_m": width_m,
            "occupied_lane_indices": occupied_lanes,
            "bbox_offset_body_x_m": float(bbox.location.x),
            "bbox_offset_body_y_m": float(bbox.location.y),
            "actor_origin_world_x_m": float(transform.location.x),
            "actor_origin_world_y_m": float(transform.location.y),
            "world_x_m": float(bbox_centre.x),
            "world_y_m": float(bbox_centre.y),
            "world_yaw_deg": float(transform.rotation.yaw),
        }

    def ego_packet(self, actor):
        packet = self.actor_packet(actor, "ego")
        angular = actor.get_angular_velocity()
        packet["yaw_rate_rps"] = -math.radians(float(angular.z))
        return packet


class LatestSensorBuffer(object):
    def __init__(self):
        self._lock = threading.Lock()
        self.rgb = None
        self.lidar = None
        self.collisions = []

    def on_rgb(self, measurement):
        with self._lock:
            self.rgb = measurement

    def on_lidar(self, measurement):
        with self._lock:
            self.lidar = measurement

    def on_collision(self, event):
        with self._lock:
            self.collisions.append(
                {
                    "frame": int(event.frame),
                    "other_actor_id": int(event.other_actor.id),
                    "other_actor_type": str(event.other_actor.type_id),
                    "impulse": [
                        float(event.normal_impulse.x),
                        float(event.normal_impulse.y),
                        float(event.normal_impulse.z),
                    ],
                }
            )

    def snapshot(self):
        with self._lock:
            return self.rgb, self.lidar, list(self.collisions)

    def clear_measurement_boundary(self):
        """Discard unrecorded warm-up frames before the evaluated episode."""
        with self._lock:
            self.rgb = None
            self.lidar = None
            self.collisions = []


_SEMANTIC_LIDAR_DTYPE = np.dtype(
    [
        ("x", np.float32),
        ("y", np.float32),
        ("z", np.float32),
        ("cos_angle", np.float32),
        ("object_idx", np.uint32),
        ("object_tag", np.uint32),
    ]
)


def _semantic_lidar_array(measurement):
    if measurement is None:
        return np.zeros(0, dtype=_SEMANTIC_LIDAR_DTYPE)
    return np.frombuffer(measurement.raw_data, dtype=_SEMANTIC_LIDAR_DTYPE)


class PlannerClient(threading.Thread):
    """One in-flight solve and one replaceable pending observation."""

    def __init__(self, host, port, init_payload, timeout_s=120.0):
        threading.Thread.__init__(self, name="dream-planner-client")
        self.daemon = True
        self.host = host
        self.port = int(port)
        self.init_payload = dict(init_payload)
        self.timeout_s = float(timeout_s)
        self.pending = queue.Queue(maxsize=1)
        self.ready = threading.Event()
        self.stopped = threading.Event()
        self._lock = threading.Lock()
        self.latest_plan = None
        self.latest_error = None
        self.dropped_requests = 0
        self.completed_requests = 0
        self.socket = None

    def submit(self, observation):
        try:
            self.pending.put_nowait(observation)
        except queue.Full:
            try:
                self.pending.get_nowait()
            except queue.Empty:
                pass
            self.pending.put_nowait(observation)
            self.dropped_requests += 1

    def get_latest(self):
        with self._lock:
            return self.latest_plan, self.latest_error

    def stop(self):
        self.stopped.set()
        try:
            self.pending.put_nowait(None)
        except queue.Full:
            try:
                self.pending.get_nowait()
            except queue.Empty:
                pass
            self.pending.put_nowait(None)

    def _connect(self):
        deadline = time.time() + self.timeout_s
        last_error = None
        while time.time() < deadline and not self.stopped.is_set():
            try:
                sock = socket.create_connection((self.host, self.port), timeout=2.0)
                sock.settimeout(self.timeout_s)
                return sock
            except socket.error as error:
                last_error = error
                time.sleep(0.25)
        raise RuntimeError("planner service connection failed: {}".format(last_error))

    def run(self):
        try:
            self.socket = self._connect()
            send_message(self.socket, make_message("hello", self.init_payload))
            ready = recv_message(self.socket, expected_type="hello")
            if ready["payload"].get("status") != "ready":
                raise ProtocolError("planner did not acknowledge readiness")
            self.ready.set()
            while not self.stopped.is_set():
                try:
                    observation = self.pending.get(timeout=0.25)
                except queue.Empty:
                    continue
                if observation is None:
                    break
                send_message(self.socket, make_message("observation", observation))
                envelope = recv_message(self.socket)
                if envelope["type"] == "error":
                    with self._lock:
                        self.latest_error = envelope["payload"]
                    continue
                if envelope["type"] != "plan":
                    raise ProtocolError(
                        "expected plan response, received {!r}".format(envelope["type"])
                    )
                with self._lock:
                    self.latest_plan = envelope["payload"]
                    self.latest_error = None
                    self.completed_requests += 1
        except Exception as error:
            with self._lock:
                self.latest_error = {
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
            self.ready.set()
        finally:
            if self.socket is not None:
                try:
                    send_message(self.socket, make_message("shutdown", {}))
                    recv_message(self.socket, expected_type="shutdown")
                except Exception:
                    pass
                try:
                    self.socket.close()
                except Exception:
                    pass


def _spawn_actor(world, blueprint_library, map_obj, spec):
    blueprint = blueprint_library.find(spec["blueprint"])
    if blueprint.has_attribute("color") and spec.get("color"):
        blueprint.set_attribute("color", str(spec["color"]))
    transform = map_obj.get_waypoint_xodr(
        int(spec.get("road_id", 40)), int(spec["lane_id"]), float(spec["s_m"])
    )
    if transform is None:
        raise RuntimeError("no OpenDRIVE waypoint for actor spec {!r}".format(spec))
    transform = transform.transform
    transform.location.z += 0.45
    actor = world.try_spawn_actor(blueprint, transform)
    if actor is None:
        raise RuntimeError("failed to spawn {} at lane {}, s={}".format(
            spec["blueprint"], spec["lane_id"], spec["s_m"]
        ))
    forward = transform.get_forward_vector()
    speed = float(spec["speed_mps"])
    actor.set_target_velocity(carla.Vector3D(forward.x * speed, forward.y * speed, 0.0))
    initial_control = carla.VehicleAckermannControl()
    initial_control.speed = speed
    initial_control.steer = 0.0
    initial_control.steer_speed = 1.5
    actor.apply_ackermann_control(initial_control)
    return actor


def _reset_actor_state(actor, map_obj, spec):
    """Restore a warmed-up vehicle to the frozen pre-episode manifest state."""
    # Keep the chassis height reached after the unrecorded suspension/powertrain
    # warm-up.  Reapplying the spawn clearance (+0.45 m) here would drop the
    # vehicle onto the road at t=0 and create a shared but non-experimental
    # braking transient in both paired conditions.
    settled_z = float(actor.get_transform().location.z)
    waypoint = map_obj.get_waypoint_xodr(
        int(spec.get("road_id", 40)), int(spec["lane_id"]), float(spec["s_m"])
    )
    if waypoint is None:
        raise RuntimeError("cannot reset actor outside the configured OpenDRIVE road")
    transform = waypoint.transform
    transform.location.z = settled_z
    actor.set_transform(transform)
    forward = transform.get_forward_vector()
    speed = float(spec["speed_mps"])
    actor.set_target_velocity(carla.Vector3D(forward.x * speed, forward.y * speed, 0.0))
    actor.set_target_angular_velocity(carla.Vector3D())
    control = carla.VehicleAckermannControl()
    control.speed = speed
    control.steer = 0.0
    control.steer_speed = 1.5
    actor.apply_ackermann_control(control)


def _speed_command(actor, target_speed_mps, steer, feedforward_accel=0.0):
    del feedforward_accel
    speed = max(_norm_xy(actor.get_velocity()), 3.0)
    dynamic_steer_limit = math.atan(
        LOW_LEVEL_MAX_LATERAL_ACCEL_MPS2 * LOW_LEVEL_WHEELBASE_M / (speed * speed)
    )
    dynamic_steer_limit = _clip(
        dynamic_steer_limit, math.radians(0.35), math.radians(10.0)
    )
    control = carla.VehicleAckermannControl()
    control.speed = max(0.0, float(target_speed_mps))
    control.steer = float(_clip(steer, -dynamic_steer_limit, dynamic_steer_limit))
    control.steer_speed = 0.20
    return control


def _apply_command(actor, control):
    if isinstance(control, carla.VehicleAckermannControl):
        actor.apply_ackermann_control(control)
    else:
        actor.apply_control(control)


def _steer_to_location(actor, target_location):
    transform = actor.get_transform()
    dx = float(target_location.x - transform.location.x)
    dy = float(target_location.y - transform.location.y)
    target_yaw = math.atan2(dy, dx)
    current_yaw = math.radians(float(transform.rotation.yaw))
    error = _wrap_angle(target_yaw - current_yaw)
    lookahead = max(1.0, math.hypot(dx, dy))
    return math.atan2(
        2.0 * LOW_LEVEL_WHEELBASE_M * math.sin(error),
        lookahead,
    )


def _smoothstep5(value):
    tau = _clip(float(value), 0.0, 1.0)
    return 10.0 * tau ** 3 - 15.0 * tau ** 4 + 6.0 * tau ** 5


def _smoothstep5_derivatives(value):
    tau = _clip(float(value), 0.0, 1.0)
    if tau <= 0.0 or tau >= 1.0:
        return 0.0, 0.0
    first = 30.0 * tau ** 2 - 60.0 * tau ** 3 + 30.0 * tau ** 4
    second = 60.0 * tau - 180.0 * tau ** 2 + 120.0 * tau ** 3
    return first, second


def _apply_hidden_cut_in_control(actor, route_frame, hidden_spec, sim_time_s,
                                 target_speed_mps=None):
    """Track the exogenous quintic cut-in with a bounded lateral-acceleration law.

    The hidden actor's route is evaluator-defined and independent of the ego
    controller.  Feedforward comes from the differentiable quintic reference;
    proportional/derivative feedback removes CARLA tracking error without pose
    injection or teleportation.
    """
    start_s = float(hidden_spec["cut_in_start_s"])
    duration_s = float(hidden_spec["cut_in_duration_s"])
    tau = (float(sim_time_s) - start_s) / duration_s
    blend = _smoothstep5(tau)
    first, second = _smoothstep5_derivatives(tau)
    source_lane = int(hidden_spec.get("source_planner_lane", 2))
    target_lane = int(hidden_spec.get("target_planner_lane", 1))
    source_y = PLANNER_LANE_CENTRES_Y[source_lane]
    target_y = PLANNER_LANE_CENTRES_Y[target_lane]
    delta_y = target_y - source_y
    desired_y = source_y + delta_y * blend
    desired_vy = delta_y * first / duration_s
    desired_ay = delta_y * second / (duration_s * duration_s)

    packet = route_frame.actor_packet(actor, "latent_vehicle")
    lateral_error = desired_y - float(packet["local_y_m"])
    velocity = actor.get_velocity()
    road_lateral_velocity = (
        float(velocity.x) * float(route_frame.left[0])
        + float(velocity.y) * float(route_frame.left[1])
    )
    lateral_velocity_error = desired_vy - road_lateral_velocity
    commanded_ay = desired_ay + 0.80 * lateral_error + 1.80 * lateral_velocity_error
    commanded_ay = _clip(
        commanded_ay,
        -LOW_LEVEL_MAX_LATERAL_ACCEL_MPS2,
        LOW_LEVEL_MAX_LATERAL_ACCEL_MPS2,
    )
    speed = max(float(packet["speed_mps"]), 3.0)
    local_steer = math.atan(
        LOW_LEVEL_WHEELBASE_M * commanded_ay / (speed * speed)
    )
    # Local +y is CARLA vehicle-left, whereas the world yaw convention used by
    # VehicleAckermannControl has the opposite sign in this RouteFrame.
    control = _speed_command(
        actor,
        float(
            hidden_spec["speed_mps"]
            if target_speed_mps is None
            else target_speed_mps
        ),
        -local_steer,
        0.0,
    )
    _apply_command(actor, control)
    return {
        "reference_y_m": desired_y,
        "reference_vy_mps": desired_vy,
        "reference_ay_mps2": desired_ay,
        "road_lateral_velocity_mps": road_lateral_velocity,
        "commanded_ay_mps2": commanded_ay,
        "lateral_error_m": lateral_error,
    }


def _lane_target_location(map_obj, road_id, source_lane, target_lane, station, blend):
    source = map_obj.get_waypoint_xodr(int(road_id), int(source_lane), float(station))
    target = map_obj.get_waypoint_xodr(int(road_id), int(target_lane), float(station))
    if source is None or target is None:
        raise RuntimeError("lane target left the valid OpenDRIVE road segment")
    source_loc = source.transform.location
    target_loc = target.transform.location
    blend = _clip(blend, 0.0, 1.0)
    return carla.Location(
        x=(1.0 - blend) * source_loc.x + blend * target_loc.x,
        y=(1.0 - blend) * source_loc.y + blend * target_loc.y,
        z=max(source_loc.z, target_loc.z),
    )


def _road_station(map_obj, actor, road_id):
    waypoint = map_obj.get_waypoint(
        actor.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving
    )
    if waypoint is None or int(waypoint.road_id) != int(road_id):
        return None
    return float(waypoint.s)


def _apply_lane_speed_control(actor, map_obj, road_id, source_lane, target_lane,
                              lane_blend, target_speed_mps, feedforward_accel=0.0,
                              road_station_limit_m=465.0):
    station = _road_station(map_obj, actor, road_id)
    if station is None:
        actor.apply_control(carla.VehicleControl(brake=1.0))
        return
    lookahead = max(8.0, 0.45 * max(_norm_xy(actor.get_velocity()), 1.0))
    target_station = min(float(road_station_limit_m), station + lookahead)
    location = _lane_target_location(
        map_obj, road_id, source_lane, target_lane, target_station, lane_blend
    )
    steer = _steer_to_location(actor, location)
    _apply_command(actor, _speed_command(actor, target_speed_mps, steer, feedforward_accel))


def _idm_acceleration(follower, leader, desired_speed_mps=None, parameters=None,
                      bumper_gap_m=None):
    parameters = dict(parameters or {})
    if desired_speed_mps is None:
        desired_speed_mps = parameters.get(
            "desired_speed_mps", max(_norm_xy(follower.get_velocity()), 1.0)
        )
    velocity = _norm_xy(follower.get_velocity())
    minimum_gap = float(parameters.get("minimum_gap_m", 2.0))
    time_headway = float(parameters.get("time_headway_s", 1.2))
    a_max = float(parameters.get("maximum_acceleration_mps2", 1.4))
    b_comfort = float(parameters.get("comfortable_deceleration_mps2", 2.2))
    exponent = float(parameters.get("exponent", 4.0))
    gap = None
    closing = 0.0
    if leader is not None:
        leader_velocity = _norm_xy(leader.get_velocity())
        if bumper_gap_m is None:
            raise ValueError(
                "bbox-centred bumper_gap_m is required when an IDM leader is present"
            )
        gap = max(0.5, float(bumper_gap_m))
        closing = velocity - leader_velocity
    desired_gap = minimum_gap + max(
        0.0,
        velocity * time_headway
        + velocity * closing / (2.0 * math.sqrt(a_max * b_comfort)),
    )
    interaction = 0.0 if gap is None else (desired_gap / gap) ** 2
    acceleration = a_max * (
        1.0
        - (velocity / max(1.0, float(desired_speed_mps))) ** exponent
        - interaction
    )
    return _clip(acceleration, -6.0, a_max), gap


def _lane_overlap(packet_a, packet_b):
    lanes_a = set(packet_a.get("occupied_lane_indices", [packet_a["lane_index"]]))
    lanes_b = set(packet_b.get("occupied_lane_indices", [packet_b["lane_index"]]))
    return bool(lanes_a.intersection(lanes_b))


def _sensor_origin_local(ego_packet, sensor_config):
    heading = float(ego_packet["local_yaw_rad"])
    forward = np.asarray([math.cos(heading), math.sin(heading)], dtype=float)
    left = np.asarray([-math.sin(heading), math.cos(heading)], dtype=float)
    sensor_forward_from_bbox_m = (
        float(sensor_config.get("x_m", 0.0))
        - float(ego_packet.get("bbox_offset_body_x_m", 0.0))
    )
    # CARLA body +y is right; RouteFrame local +y is left.
    sensor_left_from_bbox_m = (
        float(ego_packet.get("bbox_offset_body_y_m", 0.0))
        - float(sensor_config.get("y_m", 0.0))
    )
    centre = np.asarray(
        [float(ego_packet["local_x_m"]), float(ego_packet["local_y_m"])],
        dtype=float,
    )
    return (
        centre
        + sensor_forward_from_bbox_m * forward
        + sensor_left_from_bbox_m * left
    )


def _segment_intersects_oriented_box(start_xy, end_xy, box_packet,
                                      footprint_margin_m=0.05):
    start = np.asarray(start_xy, dtype=float)
    direction = np.asarray(end_xy, dtype=float) - start
    heading = float(box_packet["local_yaw_rad"])
    axes = (
        np.asarray([math.cos(heading), math.sin(heading)], dtype=float),
        np.asarray([-math.sin(heading), math.cos(heading)], dtype=float),
    )
    centre = np.asarray(
        [float(box_packet["local_x_m"]), float(box_packet["local_y_m"])],
        dtype=float,
    )
    half_extents = (
        0.5 * float(box_packet["length_m"]) + float(footprint_margin_m),
        0.5 * float(box_packet["width_m"]) + float(footprint_margin_m),
    )
    t_enter = 0.0
    t_exit = 1.0
    relative_start = start - centre
    for axis, half_extent in zip(axes, half_extents):
        position = float(np.dot(relative_start, axis))
        velocity = float(np.dot(direction, axis))
        if abs(velocity) <= 1e-9:
            if abs(position) > half_extent + 1e-9:
                return False
            continue
        lower = (-half_extent - position) / velocity
        upper = (half_extent - position) / velocity
        if lower > upper:
            lower, upper = upper, lower
        t_enter = max(t_enter, lower)
        t_exit = min(t_exit, upper)
        if t_enter > t_exit + 1e-9:
            return False
    return True


def _oriented_footprint_sample_points(box_packet):
    """Return centre, corner, and edge-midpoint samples of an actor footprint."""
    heading = float(box_packet["local_yaw_rad"])
    forward = np.asarray([math.cos(heading), math.sin(heading)], dtype=float)
    left = np.asarray([-math.sin(heading), math.cos(heading)], dtype=float)
    centre = np.asarray(
        [float(box_packet["local_x_m"]), float(box_packet["local_y_m"])],
        dtype=float,
    )
    half_length = 0.5 * float(box_packet["length_m"])
    half_width = 0.5 * float(box_packet["width_m"])
    coefficients = (
        (0.0, 0.0),
        (-1.0, -1.0), (-1.0, 1.0),
        (1.0, -1.0), (1.0, 1.0),
        (-1.0, 0.0), (1.0, 0.0),
        (0.0, -1.0), (0.0, 1.0),
    )
    return [
        centre + longitudinal * half_length * forward + lateral * half_width * left
        for longitudinal, lateral in coefficients
    ]


def _hidden_centre_geometric_visibility(ego_packet, hidden_packet,
                                        occluder_packet, sensor_config):
    sensor_origin = _sensor_origin_local(ego_packet, sensor_config)
    footprint_points = _oriented_footprint_sample_points(hidden_packet)
    hidden_centre = footprint_points[0]
    ray = hidden_centre - sensor_origin
    heading = float(ego_packet["local_yaw_rad"])
    forward = np.asarray([math.cos(heading), math.sin(heading)], dtype=float)
    left = np.asarray([-math.sin(heading), math.cos(heading)], dtype=float)
    forward_component = float(np.dot(ray, forward))
    left_component = float(np.dot(ray, left))
    range_m = float(np.linalg.norm(ray))
    bearing_rad = math.atan2(left_component, forward_component)
    blocked_samples = [
        _segment_intersects_oriented_box(
            sensor_origin, point, occluder_packet, footprint_margin_m=0.0
        )
        for point in footprint_points
    ]
    blocked = bool(blocked_samples[0])
    in_range = range_m <= float(sensor_config["range_m"])
    horizontal_fov_rad = math.radians(
        float(sensor_config.get("horizontal_fov_deg", 360.0))
    )
    in_fov = (
        forward_component > 0.0
        and abs(bearing_rad) <= 0.5 * horizontal_fov_rad + 1e-9
    ) if horizontal_fov_rad < 2.0 * math.pi - 1e-9 else True
    return {
        # A partial footprint reveal is a geometric reveal.  This is stricter
        # than the previous centre-ray test and matches what a reader can see
        # in the synchronized driver-view frame.
        "visible_now": bool(
            in_range and in_fov and not all(blocked_samples)
        ),
        "los_blocked": bool(blocked),
        "footprint_fully_blocked": bool(all(blocked_samples)),
        "blocked_sample_count": int(sum(bool(value) for value in blocked_samples)),
        "footprint_sample_count": int(len(blocked_samples)),
        "blocked_sample_fraction": float(
            sum(bool(value) for value in blocked_samples)
        ) / float(len(blocked_samples)),
        "in_range": bool(in_range),
        "in_horizontal_fov": bool(in_fov),
        "range_m": range_m,
        "bearing_rad": bearing_rad,
        "footprint_margin_m": 0.0,
    }


def _select_forward_leader(actor, role, candidates, route_frame):
    follower_packet = route_frame.actor_packet(actor, role)
    best = None
    best_gap = None
    for candidate_role, candidate in candidates:
        if candidate.id == actor.id:
            continue
        candidate_packet = route_frame.actor_packet(candidate, candidate_role)
        if not _lane_overlap(follower_packet, candidate_packet):
            continue
        centre_delta = float(candidate_packet["local_x_m"]) - float(
            follower_packet["local_x_m"]
        )
        bumper_gap = (
            centre_delta
            - 0.5 * float(candidate_packet["length_m"])
            - 0.5 * float(follower_packet["length_m"])
        )
        if bumper_gap <= 0.0:
            continue
        if best_gap is None or bumper_gap < best_gap:
            best = candidate
            best_gap = bumper_gap
    return best, best_gap


def _axis_aligned_clearance(packet_a, packet_b):
    dx = max(
        0.0,
        abs(float(packet_a["local_x_m"]) - float(packet_b["local_x_m"]))
        - 0.5 * (float(packet_a["length_m"]) + float(packet_b["length_m"])),
    )
    dy = max(
        0.0,
        abs(float(packet_a["local_y_m"]) - float(packet_b["local_y_m"]))
        - 0.5 * (float(packet_a["width_m"]) + float(packet_b["width_m"])),
    )
    return math.hypot(dx, dy)


def _local_velocity(packet):
    yaw = float(packet["local_yaw_rad"])
    body_vx = float(packet["body_vx_mps"])
    body_vy = float(packet["body_vy_mps"])
    return np.asarray(
        [
            body_vx * math.cos(yaw) - body_vy * math.sin(yaw),
            body_vx * math.sin(yaw) + body_vy * math.cos(yaw),
        ],
        dtype=float,
    )


def _packet_box_state(packet):
    velocity = _local_velocity(packet)
    return KinematicBoxState(
        x=float(packet["local_x_m"]),
        y=float(packet["local_y_m"]),
        heading=float(packet["local_yaw_rad"]),
        vx=float(velocity[0]),
        vy=float(velocity[1]),
        length=float(packet["length_m"]),
        width=float(packet["width_m"]),
        label=str(packet["role"]),
    )


def _oriented_box_clearance(packet_a, packet_b):
    return float(
        signed_oriented_box_clearance(
            _packet_box_state(packet_a), _packet_box_state(packet_b)
        )
    )


def _two_dimensional_ttc(packet_a, packet_b, horizon_s=5.0):
    value = float(
        constant_velocity_ttc(
            _packet_box_state(packet_a),
            _packet_box_state(packet_b),
            horizon_s=float(horizon_s),
        )
    )
    return None if not math.isfinite(value) else value


def _longitudinal_ttc(ego_packet, actor_packet):
    longitudinal_gap = (
        float(actor_packet["local_x_m"]) - float(ego_packet["local_x_m"])
        - 0.5 * (float(actor_packet["length_m"]) + float(ego_packet["length_m"]))
    )
    lateral_overlap = abs(
        float(actor_packet["local_y_m"]) - float(ego_packet["local_y_m"])
    ) <= 0.5 * (float(actor_packet["width_m"]) + float(ego_packet["width_m"])) + 0.4
    closing = float(ego_packet["body_vx_mps"]) - float(actor_packet["body_vx_mps"])
    if not lateral_overlap or longitudinal_gap <= 0.0 or closing <= 0.05:
        return None
    return longitudinal_gap / closing


def _select_plan(plan, sim_time_s, ego_packet):
    if not plan:
        return None, "no_valid_plan"
    if plan.get("status") == "fallback":
        return None, "planner_fallback"
    if plan.get("status") != "ok":
        return None, "no_valid_plan"
    if not plan.get("states"):
        return None, "empty_plan"
    plan_age_s = float(sim_time_s) - float(plan["source_simulation_time_s"])
    if plan_age_s > LOW_LEVEL_MAX_PLAN_AGE_S:
        return None, "stale_plan"
    if float(plan.get("validity_end_time_s", -1.0)) < float(sim_time_s) + 0.2:
        return None, "expired_plan"
    states = plan["states"]
    current_index = min(
        range(len(states)),
        key=lambda index: abs(float(states[index]["time_s"]) - float(sim_time_s)),
    )
    current_reference = states[current_index]
    deviation = math.hypot(
        float(current_reference["local_x_m"]) - float(ego_packet["local_x_m"]),
        float(current_reference["local_y_m"]) - float(ego_packet["local_y_m"]),
    )
    if deviation > 15.0:
        return None, "state_deviation"
    target_index = min(len(states) - 1, current_index + 3)
    controls = plan.get("controls") or []
    control_index = min(len(controls) - 1, current_index) if controls else None
    return {
        "current_index": current_index,
        "target_state": states[target_index],
        "current_state": current_reference,
        "control": None if control_index is None else controls[control_index],
        "deviation_m": deviation,
        "plan_age_s": plan_age_s,
    }, "accepted"


def _safety_supervisor(ego_packet, visible_packets):
    most_critical = None
    for packet in visible_packets:
        if packet["role"].startswith("follower"):
            continue
        oriented_clearance = _oriented_box_clearance(ego_packet, packet)
        ttc_2d = _two_dimensional_ttc(ego_packet, packet)
        longitudinal_ttc = _longitudinal_ttc(ego_packet, packet)
        triggers = []
        if oriented_clearance < 0.50:
            triggers.append("clearance")
        if ttc_2d is not None and ttc_2d < 1.00:
            triggers.append("ttc_2d")
        if longitudinal_ttc is not None and longitudinal_ttc < 1.35:
            triggers.append("longitudinal_ttc")
        score = min(
            oriented_clearance / 0.50,
            float("inf") if ttc_2d is None else ttc_2d / 1.00,
            float("inf") if longitudinal_ttc is None else longitudinal_ttc / 1.35,
        )
        detail = {
            "actor": packet,
            "oriented_clearance_m": oriented_clearance,
            "ttc_2d_s": ttc_2d,
            "longitudinal_ttc_s": longitudinal_ttc,
            "triggers": triggers,
            "criticality_score": score,
        }
        if most_critical is None or score < most_critical["criticality_score"]:
            most_critical = detail
    if most_critical is None:
        return False, None
    return bool(most_critical["triggers"]), most_critical


def _observation_payload(planner_episode_id, scenario_id, frame, sim_time_s,
                         route_frame, ego, visible_actor_items):
    ego_packet = route_frame.ego_packet(ego)
    visible_packets = [
        route_frame.actor_packet(actor, role) for role, actor in visible_actor_items
    ]
    return {
        "run_id": planner_episode_id,
        "scenario_id": scenario_id,
        "frame_id": int(frame),
        "simulation_time_s": float(sim_time_s),
        "capture_monotonic_ns": int(time.perf_counter() * 1e9),
        "ego": ego_packet,
        "visible_actors": visible_packets,
    }


def _rgb_surface(image):
    if image is None:
        return pygame.Surface(RGB_PANEL_SIZE)
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))[:, :, :3][:, :, ::-1]
    surface = pygame.surfarray.make_surface(np.swapaxes(array, 0, 1))
    if surface.get_size() != RGB_PANEL_SIZE:
        surface = pygame.transform.smoothscale(surface, RGB_PANEL_SIZE)
    return surface


def _risk_colour(value, maximum):
    normalized = _clip(float(value) / max(0.25, float(maximum)), 0.0, 1.0)
    if normalized < 0.33:
        ratio = normalized / 0.33
        return (0, int(220 * ratio), 255)
    if normalized < 0.66:
        ratio = (normalized - 0.33) / 0.33
        return (int(255 * ratio), 230, int(255 * (1.0 - ratio)))
    ratio = (normalized - 0.66) / 0.34
    return (255, int(230 * (1.0 - ratio)), 0)


def _draw_text(surface, font, text, position, colour=(240, 240, 240)):
    surface.blit(font.render(str(text), True, colour), position)


def _oriented_box_screen_points(packet, map_point):
    heading = float(packet["local_yaw_rad"])
    forward = np.asarray([math.cos(heading), math.sin(heading)], dtype=float)
    left = np.asarray([-math.sin(heading), math.cos(heading)], dtype=float)
    centre = np.asarray(
        [float(packet["local_x_m"]), float(packet["local_y_m"])], dtype=float
    )
    half_length = 0.5 * float(packet["length_m"])
    half_width = 0.5 * float(packet["width_m"])
    corners = [
        centre + longitudinal * half_length * forward + lateral * half_width * left
        for longitudinal, lateral in ((1, 1), (1, -1), (-1, -1), (-1, 1))
    ]
    return [map_point(point[0], point[1]) for point in corners]


def _render_bev_surface(lidar_array, route_frame, actor_items, ego, hidden_visible,
                        latest_plan, telemetry, manifest):
    del hidden_visible
    _, _, width, height = BEV_PANEL_RECT
    surface = pygame.Surface((width, height))
    surface.fill((20, 24, 31))
    font = pygame.font.Font(None, 25)
    small = pygame.font.Font(None, 20)
    ego_packet = route_frame.ego_packet(ego)

    x_min = float(ego_packet["local_x_m"]) - 45.0
    x_max = float(ego_packet["local_x_m"]) + 105.0
    y_min = float(PLANNER_LANE_CENTRES_Y[-1]) - 6.0
    y_max = float(PLANNER_LANE_CENTRES_Y[0]) + 6.0
    header_height = 56
    footer_height = 28
    pixels_per_metre = min(
        float(width - 80) / (x_max - x_min),
        float(height - header_height - footer_height - 12) / (y_max - y_min),
    )
    road_width_px = (x_max - x_min) * pixels_per_metre
    road_height_px = (y_max - y_min) * pixels_per_metre
    road_left = 0.5 * (float(width) - road_width_px)
    road_top = float(header_height) + 0.5 * (
        float(height - header_height - footer_height) - road_height_px
    )

    def bev_point(local_x, local_y):
        return (
            int(round(road_left + (float(local_x) - x_min) * pixels_per_metre)),
            int(round(road_top + (y_max - float(local_y)) * pixels_per_metre)),
        )

    field = None if latest_plan is None else latest_plan.get("field")
    if field and field.get("values"):
        values = field["values"]
        ny = int(field["ny"])
        nx = int(field["nx"])
        maximum = max(1.0, float(field.get("max", 1.0)))
        for iy in range(ny):
            local_y = float(field["y_min"]) + iy * (
                float(field["y_max"]) - float(field["y_min"])
            ) / max(1, ny - 1)
            if local_y < y_min or local_y > y_max:
                continue
            for ix in range(nx):
                value = float(values[iy][ix])
                if value < 0.04:
                    continue
                local_x = float(field["x_min"]) + ix * (
                    float(field["x_max"]) - float(field["x_min"])
                ) / max(1, nx - 1)
                if x_min <= local_x <= x_max:
                    pygame.draw.circle(
                        surface,
                        _risk_colour(value, maximum),
                        bev_point(local_x, local_y),
                        3,
                    )

    lane_boundaries = [
        PLANNER_LANE_CENTRES_Y[0] + 1.75,
        0.5 * (PLANNER_LANE_CENTRES_Y[0] + PLANNER_LANE_CENTRES_Y[1]),
        0.5 * (PLANNER_LANE_CENTRES_Y[1] + PLANNER_LANE_CENTRES_Y[2]),
        PLANNER_LANE_CENTRES_Y[2] - 1.75,
    ]
    for boundary_index, boundary_y in enumerate(lane_boundaries):
        colour = (215, 215, 215) if boundary_index in (0, 3) else (125, 132, 142)
        line_width = 3 if boundary_index in (0, 3) else 2
        pygame.draw.line(
            surface,
            colour,
            bev_point(x_min, boundary_y),
            bev_point(x_max, boundary_y),
            line_width,
        )

    if lidar_array.size:
        stride = max(1, int(lidar_array.size / 3500))
        ego_yaw = float(ego_packet["local_yaw_rad"])
        c_value = math.cos(ego_yaw)
        s_value = math.sin(ego_yaw)
        lidar_cfg = manifest["semantic_lidar"]
        # Sensor points are expressed at the semantic-LiDAR mount, whereas the
        # actor packet is expressed at the true CARLA bounding-box centre.
        sensor_origin = _sensor_origin_local(ego_packet, lidar_cfg)
        sensor_local_x = float(sensor_origin[0])
        sensor_local_y = float(sensor_origin[1])
        for point in lidar_array[::stride]:
            forward_m = float(point["x"])
            left_m = -float(point["y"])
            local_x = sensor_local_x + c_value * forward_m - s_value * left_m
            local_y = sensor_local_y + s_value * forward_m + c_value * left_m
            if x_min <= local_x <= x_max and y_min <= local_y <= y_max:
                pygame.draw.circle(surface, (145, 215, 255), bev_point(local_x, local_y), 1)

    if latest_plan and latest_plan.get("states"):
        plan_points = [
            bev_point(state["local_x_m"], state["local_y_m"])
            for state in latest_plan["states"]
            if x_min <= state["local_x_m"] <= x_max
            and y_min <= state["local_y_m"] <= y_max
        ]
        if len(plan_points) >= 2:
            pygame.draw.lines(surface, (0, 255, 255), False, plan_points, 4)

    colours = {
        "ego": (25, 115, 245),
        "occluder": (235, 95, 25),
        "latent_vehicle": (220, 45, 80),
        "blocker": (235, 195, 35),
    }
    all_items = [("ego", ego)] + list(actor_items)
    for role, actor in all_items:
        packet = route_frame.actor_packet(actor, role)
        half_diagonal = 0.5 * math.hypot(packet["length_m"], packet["width_m"])
        if (
            packet["local_x_m"] + half_diagonal < x_min
            or packet["local_x_m"] - half_diagonal > x_max
            or packet["local_y_m"] + half_diagonal < y_min
            or packet["local_y_m"] - half_diagonal > y_max
        ):
            continue
        colour = colours.get(role, (145, 150, 158))
        polygon = _oriented_box_screen_points(packet, bev_point)
        if role == "latent_vehicle" and not bool(telemetry.get("hidden_visible")):
            pygame.draw.polygon(surface, colour, polygon, 2)
            label = "latent (evaluator only)"
        else:
            pygame.draw.polygon(surface, colour, polygon)
            pygame.draw.polygon(surface, (235, 238, 242), polygon, 1)
            label = role
        if role in ("ego", "occluder", "latent_vehicle") or role.startswith("lead"):
            label_point = bev_point(packet["local_x_m"], packet["local_y_m"])
            _draw_text(surface, small, label, (label_point[0] + 7, label_point[1] - 18))

    field_controller = str(telemetry.get("controller", "")).upper()
    if field and field_controller in ("DREAM", "ADA", "APF"):
        title_text = "Metric-scale BEV | {} planning field + semantic LiDAR".format(
            field_controller
        )
    elif field:
        title_text = "Metric-scale BEV | planning field + semantic LiDAR"
    else:
        title_text = "Metric-scale BEV | planned path + semantic LiDAR"
    _draw_text(surface, font, title_text, (18, 8))
    if telemetry.get("condition") == "empty_shadow":
        visibility_text = "Empty-shadow control: no latent vehicle"
        visibility_colour = (145, 205, 255)
    else:
        visibility_text = "Latent vehicle: {} | LiDAR hits: {}".format(
            "REVEALED" if telemetry.get("hidden_visible") else "OCCLUDED",
            telemetry.get("hidden_lidar_hits", 0),
        )
        visibility_colour = (
            (120, 255, 155) if telemetry.get("hidden_visible") else (255, 210, 80)
        )
    _draw_text(surface, small, visibility_text, (18, 34), visibility_colour)
    telemetry_text = (
        "{} | {} | t={:.2f} s | ego={:.1f} m/s | plan age={} | shield={}".format(
            telemetry["controller"],
            telemetry["condition"],
            telemetry["time_s"],
            telemetry["ego_speed_mps"],
            "N/A" if telemetry.get("plan_age_s") is None else "{:.2f} s".format(
                telemetry["plan_age_s"]
            ),
            "ACTIVE" if telemetry.get("shield_active") else "inactive",
        )
    )
    _draw_text(surface, small, telemetry_text, (18, height - 23), (230, 233, 238))
    _draw_text(
        surface,
        small,
        "cyan: planned path | blue: ego | orange: rigid truck | red: latent car",
        (1110, height - 23),
        (190, 195, 205),
    )
    return surface


def _compose_frame(rgb, lidar_array, route_frame, actor_items, ego, hidden,
                   hidden_visible, latest_plan, telemetry, history, manifest):
    del hidden, history
    canvas = pygame.Surface(COMPOSITE_SIZE)
    canvas.fill((12, 15, 20))
    driver_surface = _rgb_surface(rgb)
    canvas.blit(driver_surface, DRIVER_PANEL_RECT[:2])
    font = pygame.font.Font(None, 25)
    title = pygame.Surface((DRIVER_PANEL_RECT[2], 42), pygame.SRCALPHA)
    title.fill((0, 0, 0, 150))
    canvas.blit(title, (0, 0))
    _draw_text(
        canvas,
        font,
        "Qualitative driver view | camera is not planner input",
        (18, 10),
        (255, 255, 255),
    )
    telemetry["hidden_visible"] = bool(hidden_visible)
    bev_surface = _render_bev_surface(
        lidar_array,
        route_frame,
        actor_items,
        ego,
        hidden_visible,
        latest_plan,
        telemetry,
        manifest,
    )
    canvas.blit(bev_surface, BEV_PANEL_RECT[:2])
    return canvas, driver_surface, bev_surface


def _planner_process(args, output_dir):
    log_path = os.path.join(output_dir, "planner_service.log")
    log_handle = open(log_path, "w")
    packaged_planner = os.path.join(
        os.path.dirname(__file__), "carla_external_planner.py"
    )
    planner_script = (
        packaged_planner
        if os.path.exists(packaged_planner)
        else os.path.join(REPO_ROOT, "Integration", "carla_external_planner.py")
    )
    command = [
        args.planner_python,
        planner_script,
        "--host", args.planner_host,
        "--port", str(args.planner_port),
        "--timeout-s", str(args.planner_timeout_s),
    ]
    planner_environment = os.environ.copy()
    # The submitted stack already relies on Intel-OpenMP components from more
    # than one numerical package.  Isolate its existing compatibility setting
    # to the planner process and record it in provenance rather than applying
    # it to the CARLA control process.
    planner_environment.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        env=planner_environment,
    )
    return process, log_handle, command


def _launch_carla_server(args):
    if not args.launch_server:
        return None, None
    executable = args.carla_executable
    command = [
        executable,
        "CarlaUE4",
        "-carla-port={}".format(args.carla_port),
        "-RenderOffScreen",
        "-nosound",
        "-quality-level={}".format(args.quality_level),
        "-ResX=1280",
        "-ResY=720",
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return process, command


def _connect_carla(host, port, timeout_s):
    deadline = time.time() + float(timeout_s)
    last_error = None
    while time.time() < deadline:
        try:
            client = carla.Client(host, int(port))
            client.set_timeout(10.0)
            client.get_server_version()
            return client
        except Exception as error:
            last_error = error
            time.sleep(0.5)
    raise RuntimeError("CARLA connection timed out: {}".format(last_error))


def _require_unbound_tcp_port(host, port, purpose):
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        probe.bind((socket.gethostbyname(str(host)), int(port)))
    except OSError as error:
        raise RuntimeError(
            "{} port {}:{} is already occupied: {}".format(
                purpose, host, port, error
            )
        )
    finally:
        probe.close()


def _spawn_sensors(world, blueprint_library, ego, manifest, buffer_obj,
                   include_rgb=True):
    sensors = []
    if include_rgb:
        rgb_cfg = manifest["rgb_camera"]
        rgb_bp = blueprint_library.find("sensor.camera.rgb")
        rgb_bp.set_attribute("image_size_x", str(rgb_cfg["width_px"]))
        rgb_bp.set_attribute("image_size_y", str(rgb_cfg["height_px"]))
        rgb_bp.set_attribute("fov", str(rgb_cfg["fov_deg"]))
        rgb_bp.set_attribute("sensor_tick", str(rgb_cfg["sensor_tick_s"]))
        rgb = world.spawn_actor(
            rgb_bp,
            carla.Transform(
                carla.Location(x=rgb_cfg["x_m"], z=rgb_cfg["z_m"]),
                carla.Rotation(pitch=rgb_cfg["pitch_deg"]),
            ),
            attach_to=ego,
        )
        rgb.listen(buffer_obj.on_rgb)
        sensors.append(rgb)

    lidar_cfg = manifest["semantic_lidar"]
    lidar_bp = blueprint_library.find("sensor.lidar.ray_cast_semantic")
    attributes = {
        "channels": lidar_cfg["channels"],
        "range": lidar_cfg["range_m"],
        "points_per_second": lidar_cfg["points_per_second"],
        "rotation_frequency": lidar_cfg["rotation_frequency_hz"],
        "upper_fov": lidar_cfg["upper_fov_deg"],
        "lower_fov": lidar_cfg["lower_fov_deg"],
        "horizontal_fov": lidar_cfg["horizontal_fov_deg"],
        "sensor_tick": lidar_cfg["sensor_tick_s"],
    }
    for key, value in attributes.items():
        lidar_bp.set_attribute(str(key), str(value))
    lidar = world.spawn_actor(
        lidar_bp,
        carla.Transform(carla.Location(x=lidar_cfg["x_m"], z=lidar_cfg["z_m"])),
        attach_to=ego,
    )
    lidar.listen(buffer_obj.on_lidar)
    sensors.append(lidar)

    collision_bp = blueprint_library.find("sensor.other.collision")
    collision = world.spawn_actor(
        collision_bp, carla.Transform(), attach_to=ego
    )
    collision.listen(buffer_obj.on_collision)
    sensors.append(collision)
    return sensors


def _summary(rows, manifest, condition, controller, reveal_time_s, collisions,
             planner_client, runtime, qualification, actor_geometry, initial_state):
    ego_speeds = [row["ego_speed_mps"] for row in rows]
    ego_accels = [row["ego_accel_mps2"] for row in rows]
    # A completed asynchronous plan remains the latest plan for several physics
    # ticks.  Count each plan once rather than treating those repeated samples as
    # independent planner executions.
    planner_times = [
        row["planner_total_s"]
        for row in rows
        if row.get("new_plan") and row.get("planner_total_s") is not None
    ]
    planner_field_times = [
        row["planner_field_s"]
        for row in rows
        if row.get("new_plan") and row.get("planner_field_s") is not None
    ]
    planner_decision_times = [
        row["planner_decision_s"]
        for row in rows
        if row.get("new_plan") and row.get("planner_decision_s") is not None
    ]
    planner_mpc_times = [
        row["planner_mpc_s"]
        for row in rows
        if row.get("new_plan") and row.get("planner_mpc_s") is not None
    ]
    plan_ages = [
        row["plan_age_s"] for row in rows if row.get("plan_age_s") is not None
    ]
    low_level_times = [
        row["low_level_time_s"] for row in rows if row.get("low_level_updated")
    ]
    control_cycle_times = [
        row["physics_control_cycle_time_s"]
        for row in rows
        if row.get("physics_control_cycle_time_s") is not None
    ]
    metric_cfg = manifest.get("evaluation_metrics", {})
    near_clearance_m = float(metric_cfg.get("near_collision_clearance_m", 0.50))
    near_ttc_2d_s = float(metric_cfg.get("near_collision_ttc_2d_s", 1.00))
    near_collision_rows = [
        row
        for row in rows
        if (
            row.get("minimum_oriented_clearance_m") is not None
            and 0.0 < float(row["minimum_oriented_clearance_m"]) < near_clearance_m
        )
        or (
            row.get("minimum_ttc_2d_s") is not None
            and float(row["minimum_ttc_2d_s"]) < near_ttc_2d_s
        )
    ]
    post_reveal_rows = []
    if reveal_time_s is not None:
        post_reveal_rows = [
            row
            for row in rows
            if float(reveal_time_s) <= float(row["time_s"]) <= float(reveal_time_s) + 3.0
        ]
    clearance_rows = [
        row for row in rows if row.get("minimum_oriented_clearance_m") is not None
    ]
    minimum_clearance_row = (
        None
        if not clearance_rows
        else min(clearance_rows, key=lambda row: float(row["minimum_oriented_clearance_m"]))
    )
    ttc_2d_rows = [row for row in rows if row.get("minimum_ttc_2d_s") is not None]
    minimum_ttc_2d_row = (
        None
        if not ttc_2d_rows
        else min(ttc_2d_rows, key=lambda row: float(row["minimum_ttc_2d_s"]))
    )
    hidden_observation_rows = [
        row for row in rows if row.get("hidden_observation_submitted")
    ]
    hidden_aware_plan_rows = [row for row in rows if row.get("hidden_aware_plan")]
    first_hidden_observation_time_s = (
        None if not hidden_observation_rows else float(hidden_observation_rows[0]["time_s"])
    )
    first_hidden_aware_plan_time_s = (
        None if not hidden_aware_plan_rows else float(hidden_aware_plan_rows[0]["time_s"])
    )
    ego_overtake_rows = [
        row for row in rows
        if row.get("ego_pass_clearance_ahead_occluder_m") is not None
        and float(row["ego_pass_clearance_ahead_occluder_m"]) >= 0.0
    ]
    ego_centre_overlap_rows = [
        row for row in rows if row.get("ego_centre_lane_overlap")
    ]
    ego_centre_complete_rows = [
        row for row in rows
        if row.get("ego_target_lane_error_m") is not None
        and float(row["ego_target_lane_error_m"])
        <= float(manifest["qualification"]["maximum_cut_in_lateral_error_m"])
    ]
    hidden_overtake_rows = [
        row for row in rows
        if row.get("hidden_pass_clearance_ahead_occluder_m") is not None
        and float(row["hidden_pass_clearance_ahead_occluder_m"]) >= 0.0
    ]
    hidden_centre_overlap_rows = [
        row for row in rows if row.get("hidden_centre_lane_overlap")
    ]
    concurrent_overlap_rows = [
        row for row in rows if row.get("concurrent_centre_lane_overlap")
    ]
    geometric_visible_rows = [
        row for row in rows if row.get("hidden_center_geometric_visible_now")
    ]
    geometric_confirmed_rows = [
        row for row in rows
        if row.get("hidden_center_geometric_reveal_confirmed")
    ]
    geometric_first_clear_time_s = (
        None if not geometric_visible_rows
        else float(geometric_visible_rows[0]["time_s"])
    )
    geometric_confirmed_time_s = (
        None if not geometric_confirmed_rows
        else float(geometric_confirmed_rows[0]["time_s"])
    )
    construction = manifest.get("scene_construction", {})
    result = {
        "schema_version": "carla_overtaking_trial_result_v1",
        "scenario_id": manifest["scenario_id"],
        "scenario_family": manifest.get("scenario_family", manifest["scenario_id"]),
        "construction_hash": construction.get("construction_hash_sha256"),
        "scene_seed": construction.get("seed"),
        "generator_version": construction.get("generator_version"),
        "resolved_manifest_path": "resolved_manifest.json",
        "condition": condition,
        "controller": controller,
        "duration_s": manifest["duration_s"],
        "reveal_time_s": reveal_time_s,
        "collision_count": len(collisions),
        "collision_incidence": int(bool(collisions)),
        "near_collision_incidence": int(bool(near_collision_rows)),
        "collision_or_near_incidence": int(bool(collisions) or bool(near_collision_rows)),
        "collisions": collisions,
        "ego": {
            "mean_speed_mps": float(np.mean(ego_speeds)),
            "minimum_speed_mps": float(np.min(ego_speeds)),
            "maximum_speed_loss_mps": float(ego_speeds[0] - np.min(ego_speeds)),
            "peak_deceleration_mps2": float(np.min(ego_accels)),
            "minimum_axis_aligned_clearance_m": _finite_min(
                [row.get("minimum_clearance_m") for row in rows]
            ),
            "minimum_longitudinal_ttc_s": _finite_min(
                [row.get("minimum_ttc_s") for row in rows]
            ),
            "minimum_oriented_box_clearance_m": _finite_min(
                [row.get("minimum_oriented_clearance_m") for row in rows]
            ),
            "minimum_oriented_box_clearance_actor_role": (
                None
                if minimum_clearance_row is None
                else minimum_clearance_row.get("minimum_oriented_clearance_actor_role")
            ),
            "minimum_oriented_box_clearance_time_s": (
                None if minimum_clearance_row is None else minimum_clearance_row.get("time_s")
            ),
            "minimum_ttc_2d_s": _finite_min(
                [row.get("minimum_ttc_2d_s") for row in rows]
            ),
            "minimum_ttc_2d_actor_role": (
                None
                if minimum_ttc_2d_row is None
                else minimum_ttc_2d_row.get("minimum_ttc_2d_actor_role")
            ),
            "minimum_ttc_2d_time_s": (
                None if minimum_ttc_2d_row is None else minimum_ttc_2d_row.get("time_s")
            ),
            "minimum_oriented_box_clearance_first_3s_after_reveal_m": _finite_min(
                [row.get("minimum_oriented_clearance_m") for row in post_reveal_rows]
            ),
            "minimum_hidden_oriented_box_clearance_m": _finite_min(
                [row.get("hidden_oriented_clearance_m") for row in rows]
            ),
            "minimum_hidden_ttc_2d_s": _finite_min(
                [row.get("hidden_ttc_2d_s") for row in rows]
            ),
            "shield_active_time_s": sum(
                manifest["physics_dt_s"] for row in rows if row.get("shield_active")
            ),
            "overtake_completion_time_s": (
                None if not ego_overtake_rows
                else float(ego_overtake_rows[0]["time_s"])
            ),
            "centre_lane_overlap_time_s": (
                None if not ego_centre_overlap_rows
                else float(ego_centre_overlap_rows[0]["time_s"])
            ),
            "centre_lane_completion_time_s": (
                None if not ego_centre_complete_rows
                else float(ego_centre_complete_rows[0]["time_s"])
            ),
            "final_target_lane_error_m": float(rows[-1]["ego_target_lane_error_m"]),
            "route_completed": int(
                bool(ego_overtake_rows) and bool(ego_centre_complete_rows)
            ),
        },
        "latent_vehicle": {
            "present": int(condition == "true_threat"),
            "overtake_completion_time_s": (
                None if not hidden_overtake_rows
                else float(hidden_overtake_rows[0]["time_s"])
            ),
            "centre_lane_overlap_time_s": (
                None if not hidden_centre_overlap_rows
                else float(hidden_centre_overlap_rows[0]["time_s"])
            ),
            "final_target_lane_error_m": (
                None if rows[-1].get("hidden_local_y_m") is None
                else abs(
                    float(rows[-1]["hidden_local_y_m"])
                    - PLANNER_LANE_CENTRES_Y[1]
                )
            ),
        },
        "scene_realization": {
            "concurrent_centre_lane_overlap_incidence": int(
                bool(concurrent_overlap_rows)
            ),
            "first_concurrent_centre_lane_overlap_time_s": (
                None if not concurrent_overlap_rows
                else float(concurrent_overlap_rows[0]["time_s"])
            ),
            "minimum_hidden_longitudinal_separation_during_overlap_m": (
                None if not concurrent_overlap_rows
                else min(
                    abs(
                        float(row["ego_station_m"])
                        - float(row["hidden_station_m"])
                    )
                    for row in concurrent_overlap_rows
                    if row.get("hidden_station_m") is not None
                )
            ),
            "nominal_factors": construction.get("realized_factors"),
        },
        "visibility": {
            "geometric_definition": (
                "semantic-LiDAR origin to nine latent oriented-footprint samples; "
                "the latent actor is geometrically hidden only when every ray is "
                "blocked by the measured firetruck footprint"
            ),
            "geometric_footprint_margin_m": 0.0,
            "geometric_footprint_sample_count": 9,
            "initial_occlusion_evaluation_start_s": qualification.get(
                "initial_occlusion_evaluation_start_s"
            ),
            "initial_full_footprint_occlusion_pass": qualification.get(
                "initial_hidden_footprint_geometric_occlusion_pass"
            ),
            "initial_minimum_blocked_sample_fraction": qualification.get(
                "initial_hidden_footprint_minimum_blocked_sample_fraction"
            ),
            "geometric_confirmation_frames": 2,
            "hidden_center_first_geometric_clear_time_s": geometric_first_clear_time_s,
            "hidden_center_confirmed_geometric_reveal_time_s": geometric_confirmed_time_s,
            "semantic_lidar_confirmed_reveal_time_s": reveal_time_s,
            "geometric_to_semantic_lidar_reveal_s": (
                None
                if geometric_confirmed_time_s is None or reveal_time_s is None
                else float(reveal_time_s) - geometric_confirmed_time_s
            ),
            "geometric_to_planner_observation_s": (
                None
                if geometric_confirmed_time_s is None
                or first_hidden_observation_time_s is None
                else first_hidden_observation_time_s - geometric_confirmed_time_s
            ),
            "geometric_to_hidden_aware_plan_s": (
                None
                if geometric_confirmed_time_s is None
                or first_hidden_aware_plan_time_s is None
                else first_hidden_aware_plan_time_s - geometric_confirmed_time_s
            ),
        },
        "followers": {},
        "traffic_stream": {},
        "planner": {
            "completed_requests": int(planner_client.completed_requests),
            "dropped_requests": int(planner_client.dropped_requests),
            "dropped_request_fraction": (
                float(planner_client.dropped_requests)
                / max(
                    1.0,
                    float(
                        planner_client.completed_requests
                        + planner_client.dropped_requests
                    ),
                )
            ),
            "effective_completed_update_rate_hz": (
                float(planner_client.completed_requests)
                / float(manifest["duration_s"])
            ),
            "mean_total_s": None if not planner_times else float(np.mean(planner_times)),
            "p95_total_s": _percentile(planner_times, 95),
            "maximum_total_s": _finite_max(planner_times),
            "mean_decision_s": (
                None if not planner_decision_times
                else float(np.mean(planner_decision_times))
            ),
            "mean_mpc_s": (
                None if not planner_mpc_times else float(np.mean(planner_mpc_times))
            ),
            "mean_field_s": (
                None if not planner_field_times
                else float(np.mean(planner_field_times))
            ),
            "mean_applied_plan_age_s": (
                None if not plan_ages else float(np.mean(plan_ages))
            ),
            "p95_applied_plan_age_s": _percentile(plan_ages, 95),
            "maximum_applied_plan_age_s": _finite_max(plan_ages),
            "fallback_responses": sum(
                1
                for row in rows
                if row.get("new_plan") and row.get("latest_plan_status") == "fallback"
            ),
            "first_hidden_observation_submitted_time_s": first_hidden_observation_time_s,
            "first_hidden_aware_plan_applied_time_s": first_hidden_aware_plan_time_s,
            "reveal_to_hidden_observation_submitted_s": (
                None
                if reveal_time_s is None or first_hidden_observation_time_s is None
                else first_hidden_observation_time_s - float(reveal_time_s)
            ),
            "reveal_to_hidden_aware_plan_applied_s": (
                None
                if reveal_time_s is None or first_hidden_aware_plan_time_s is None
                else first_hidden_aware_plan_time_s - float(reveal_time_s)
            ),
        },
        "low_level": {
            "mean_time_s": float(np.mean(low_level_times)),
            "p95_time_s": _percentile(low_level_times, 95),
            "maximum_time_s": _finite_max(low_level_times),
            "deadline_miss_fraction": float(np.mean(np.asarray(low_level_times) > 0.1)),
            "control_executions": len(low_level_times),
            "observed_control_rate_hz": len(low_level_times) / float(manifest["duration_s"]),
            "wall_clock_effective_control_rate_hz": (
                len(low_level_times) / max(float(runtime["wall_duration_s"]), 1e-9)
            ),
            "maximum_plan_age_s": LOW_LEVEL_MAX_PLAN_AGE_S,
            "lateral_acceleration_limit_mps2": LOW_LEVEL_MAX_LATERAL_ACCEL_MPS2,
            "stale_plan_fallback_time_s": sum(
                manifest["physics_dt_s"]
                for row in rows
                if row.get("plan_rejection_reason") == "stale_plan"
            ),
            "stale_plan_exposure_fraction": float(
                np.mean(
                    [
                        row.get("plan_rejection_reason") == "stale_plan"
                        for row in rows
                    ]
                )
            ),
            "planner_fallback_time_s": sum(
                manifest["physics_dt_s"]
                for row in rows
                if row.get("plan_rejection_reason") == "planner_fallback"
            ),
            "steering_envelope_active_time_s": sum(
                manifest["physics_dt_s"]
                for row in rows
                if row.get("steering_envelope_active")
            ),
        },
        "physics_control_loop": {
            "target_rate_hz": 1.0 / float(manifest["physics_dt_s"]),
            "mean_cycle_time_s": float(np.mean(control_cycle_times)),
            "p95_cycle_time_s": _percentile(control_cycle_times, 95),
            "maximum_cycle_time_s": _finite_max(control_cycle_times),
            "deadline_miss_fraction": float(
                np.mean(
                    np.asarray(control_cycle_times, dtype=float)
                    > float(manifest["physics_dt_s"])
                )
            ),
        },
        "runtime": runtime,
        "qualification": qualification,
        "actor_geometry": actor_geometry,
        "measurement_initial_state": initial_state,
        "metric_definitions": {
            "near_collision_clearance_m": near_clearance_m,
            "near_collision_ttc_2d_s": near_ttc_2d_s,
            "ttc_2d_model": "constant velocity, fixed yaw, oriented rectangles, 5 s horizon",
            "integrated_speed_deficit_m": (
                "Per-actor integral of desired-speed deficit over time; units are metres."
            ),
            "total_integrated_speed_deficit_vehicle_m": (
                "Sum of per-actor integrated speed deficits; units are vehicle-metres."
            ),
            "geometric_reveal_is_evaluator_only": True,
        },
    }
    traffic_specs = [dict(manifest["actors"]["occluder"])]
    traffic_specs[0]["label"] = "occluder"
    traffic_specs.extend(manifest["actors"].get("followers", []))
    traffic_specs.extend(manifest["actors"].get("idm_npcs", []))
    seen_traffic_roles = set()
    for spec in traffic_specs:
        role = str(spec.get("label", ""))
        if not role or role in seen_traffic_roles:
            continue
        seen_traffic_roles.add(role)
        speed_key = "{}_speed_mps".format(role)
        accel_key = "{}_accel_mps2".format(role)
        speeds = [
            float(row[speed_key]) for row in rows
            if row.get(speed_key) is not None
        ]
        accelerations = [
            float(row[accel_key]) for row in rows
            if row.get(accel_key) is not None
        ]
        if not speeds or not accelerations:
            continue
        desired_speed = float(
            spec.get("idm", {}).get("desired_speed_mps", spec.get("speed_mps", speeds[0]))
        )
        speed_deficits = [max(0.0, desired_speed - speed) for speed in speeds]
        peak_deficit_index = int(np.argmax(np.asarray(speed_deficits, dtype=float)))
        recovery_time_s = None
        for index in range(peak_deficit_index, len(speeds)):
            if abs(speeds[index] - desired_speed) <= 0.5:
                recovery_time_s = float(rows[index]["time_s"])
                break
        record = {
            "traffic_role": spec.get("traffic_role", "occluder"),
            "lane_name": spec.get("lane_name", "centre" if role == "occluder" else None),
            "desired_speed_mps": desired_speed,
            "mean_speed_mps": float(np.mean(speeds)),
            "minimum_speed_mps": float(np.min(speeds)),
            "maximum_speed_loss_mps": max(0.0, float(speeds[0] - np.min(speeds))),
            "maximum_desired_speed_deficit_mps": float(max(speed_deficits)),
            "integrated_speed_deficit_m": float(
                sum(speed_deficits) * float(manifest["physics_dt_s"])
            ),
            "peak_deceleration_mps2": float(np.min(accelerations)),
            "hard_brake_time_s": sum(
                manifest["physics_dt_s"] for value in accelerations if value < -3.0
            ),
            "recovery_time_s": recovery_time_s,
            "recovered_within_episode": int(recovery_time_s is not None),
        }
        result["traffic_stream"][role] = record
        if role.startswith("follower_"):
            result["followers"][role] = dict(record)

    stream_records = list(result["traffic_stream"].values())
    follower_roles = sorted(
        result["followers"],
        key=lambda item: int(item.rsplit("_", 1)[-1]),
    )
    amplification = []
    for leader_role, follower_role in zip(follower_roles, follower_roles[1:]):
        upstream = float(
            result["followers"][leader_role]["maximum_speed_loss_mps"]
        )
        downstream = float(
            result["followers"][follower_role]["maximum_speed_loss_mps"]
        )
        if upstream > 0.10:
            amplification.append(downstream / upstream)
    result["traffic_disturbance"] = {
        "maximum_speed_loss_mps": max(
            record["maximum_speed_loss_mps"] for record in stream_records
        ),
        "total_integrated_speed_deficit_vehicle_m": sum(
            record["integrated_speed_deficit_m"] for record in stream_records
        ),
        "peak_deceleration_mps2": min(
            record["peak_deceleration_mps2"] for record in stream_records
        ),
        "hard_braking_actor_count": sum(
            1 for record in stream_records if record["hard_brake_time_s"] > 0.0
        ),
        "maximum_follower_disturbance_amplification": (
            None if not amplification else max(amplification)
        ),
        "episode_horizon_limits_full_wave_recovery_assessment": True,
    }
    return result


def run_trial(args):
    with open(args.manifest, "rb") as manifest_source_handle:
        manifest_source_bytes = manifest_source_handle.read()
    manifest_source_sha256 = hashlib.sha256(manifest_source_bytes).hexdigest()
    manifest_source_schema = json.loads(
        manifest_source_bytes.decode("utf-8")
    ).get("schema_version")
    manifest = _load_manifest(args.manifest, args.seed)
    _validate_execution_contract(manifest)
    frozen_manifest_input_pass = manifest_source_schema == RESOLVED_SCHEMA
    if args.duration_s is not None:
        if manifest.get("scene_construction") is not None:
            raise ValueError(
                "--duration-s cannot override a frozen randomized construction; "
                "change the scene template before freezing the bank"
            )
        manifest["duration_s"] = float(args.duration_s)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_id = "{}_{}_seed{}_{}".format(
        args.controller.lower(), args.condition, args.seed, timestamp
    )
    # This identifier crosses the evaluator/planner boundary and therefore
    # deliberately excludes both paired condition and latent-actor truth.
    planner_episode_id = "{}_seed{}".format(manifest["scenario_id"], args.seed)
    output_dir = os.path.abspath(os.path.join(args.output_root, run_id))
    frames_dir = os.path.join(output_dir, "frames")
    driver_frames_dir = os.path.join(output_dir, "driver_frames")
    bev_frames_dir = os.path.join(output_dir, "bev_frames")
    os.makedirs(frames_dir)
    if args.record_frames:
        os.makedirs(driver_frames_dir)
        os.makedirs(bev_frames_dir)
    _write_json(os.path.join(output_dir, "resolved_manifest.json"), manifest)

    pygame.init()
    carla_process = None
    carla_command = None
    planner_process = None
    planner_log = None
    planner_client = None
    world = None
    original_settings = None
    actors = []
    sensors = []
    tick_rows = []
    csv_handle = None
    actor_state_handle = None
    npc_trace_handle = None
    npc_trace_row_count = 0
    started_wall = time.perf_counter()
    experiment_command = " ".join(sys.argv)

    try:
        carla_process, carla_command = _launch_carla_server(args)
        client = _connect_carla(args.carla_host, args.carla_port, args.server_timeout_s)
        client.set_timeout(60.0)
        world = client.load_world(manifest["map"])
        original_settings = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = float(manifest["physics_dt_s"])
        settings.substepping = True
        settings.max_substep_delta_time = 0.01
        settings.max_substeps = 5
        settings.no_rendering_mode = False
        world.apply_settings(settings)
        weather_name = str(manifest["weather"])
        if not hasattr(carla.WeatherParameters, weather_name):
            raise ValueError("unsupported CARLA weather preset: {}".format(weather_name))
        world.set_weather(getattr(carla.WeatherParameters, weather_name))
        map_obj = world.get_map()
        blueprints = world.get_blueprint_library()

        actor_specs = manifest["actors"]
        ego_spawn_spec = copy.deepcopy(actor_specs["ego"])
        ego_spawn_spec["road_id"] = manifest["road_id"]
        occluder_spawn_spec = copy.deepcopy(actor_specs["occluder"])
        occluder_spawn_spec["road_id"] = manifest["road_id"]
        ego = _spawn_actor(world, blueprints, map_obj, ego_spawn_spec)
        truck = _spawn_actor(world, blueprints, map_obj, occluder_spawn_spec)
        blocker = None
        if bool(actor_specs.get("blocker", {}).get("enabled", True)):
            blocker_spawn_spec = copy.deepcopy(actor_specs["blocker"])
            blocker_spawn_spec["road_id"] = manifest["road_id"]
            blocker = _spawn_actor(world, blueprints, map_obj, blocker_spawn_spec)
        actors.extend([ego, truck])
        if blocker is not None:
            actors.append(blocker)
        hidden = None
        if args.condition == "true_threat":
            hidden_spawn_spec = copy.deepcopy(actor_specs["latent_vehicle"])
            hidden_spawn_spec["road_id"] = manifest["road_id"]
            hidden = _spawn_actor(world, blueprints, map_obj, hidden_spawn_spec)
            actors.append(hidden)
        idm_specs = copy.deepcopy(list(actor_specs.get("idm_npcs", [])))
        existing_idm_labels = {str(spec.get("label")) for spec in idm_specs}
        for legacy_spec in actor_specs.get("followers", []):
            if str(legacy_spec.get("label")) in existing_idm_labels:
                continue
            spec = copy.deepcopy(legacy_spec)
            spec.setdefault("traffic_role", "follower")
            spec.setdefault(
                "idm",
                {
                    "desired_speed_mps": float(actor_specs["ego"]["speed_mps"]),
                    "time_headway_s": 1.2,
                    "minimum_gap_m": 2.0,
                    "maximum_acceleration_mps2": 1.4,
                    "comfortable_deceleration_mps2": 2.2,
                    "exponent": 4.0,
                },
            )
            idm_specs.append(spec)
        idm_npcs = []
        for spec in idm_specs:
            spec["road_id"] = manifest["road_id"]
            npc = _spawn_actor(world, blueprints, map_obj, spec)
            idm_npcs.append((spec["label"], npc, spec))
            actors.append(npc)
        followers = [
            (label, npc)
            for label, npc, spec in idm_npcs
            if str(spec.get("traffic_role", "")).lower() == "follower"
            and str(label).startswith("follower_")
        ]
        idm_items = [(label, npc) for label, npc, _ in idm_npcs]
        controlled_npc_records = [
            ("occluder", truck, actor_specs["occluder"])
        ] + list(idm_npcs)

        anchor = map_obj.get_waypoint_xodr(
            int(manifest["road_id"]), int(manifest["lane_map"]["centre"]),
            float(actor_specs["ego"]["s_m"])
        ).transform
        route_frame = RouteFrame(
            anchor,
            anchor_station_m=float(actor_specs["ego"]["s_m"]),
            centre_local_y=PLANNER_LANE_CENTRES_Y[1],
        )

        def apply_scene_lane_speed_control(
            actor, source_lane, target_lane, lane_blend, target_speed_mps,
            feedforward_accel=0.0
        ):
            _apply_lane_speed_control(
                actor,
                map_obj,
                manifest["road_id"],
                source_lane,
                target_lane,
                lane_blend,
                target_speed_mps,
                feedforward_accel,
                road_station_limit_m=manifest["road_station_limit_m"],
            )

        def lane_holding_items(include_hidden=True):
            items = [("ego", ego), ("occluder", truck)]
            if blocker is not None:
                items.append(("blocker", blocker))
            if include_hidden and hidden is not None:
                items.append(("latent_vehicle", hidden))
            items.extend((label, npc) for label, npc, _ in idm_npcs)
            return items

        def apply_idm_population(dt_s, include_hidden=True):
            candidate_items = lane_holding_items(include_hidden=include_hidden)
            decisions = {}
            for role, actor, spec in controlled_npc_records:
                parameters = dict(spec.get("idm", {}))
                if not parameters and role == "occluder":
                    apply_scene_lane_speed_control(
                        actor,
                        spec["lane_id"],
                        spec["lane_id"],
                        0.0,
                        spec["speed_mps"],
                    )
                    decisions[role] = {
                        "leader_role": None,
                        "gap_m": None,
                        "acceleration_mps2": 0.0,
                        "target_speed_mps": float(spec["speed_mps"]),
                    }
                    continue
                leader, geometric_gap = _select_forward_leader(
                    actor, role, candidate_items, route_frame
                )
                acceleration, idm_gap = _idm_acceleration(
                    actor,
                    leader,
                    parameters.get("desired_speed_mps", spec["speed_mps"]),
                    parameters,
                    bumper_gap_m=geometric_gap,
                )
                target_speed = max(
                    0.0, _norm_xy(actor.get_velocity()) + acceleration * float(dt_s)
                )
                apply_scene_lane_speed_control(
                    actor,
                    spec["lane_id"],
                    spec["lane_id"],
                    0.0,
                    target_speed,
                    acceleration,
                )
                leader_role = None
                if leader is not None:
                    leader_role = next(
                        (
                            candidate_role
                            for candidate_role, candidate in candidate_items
                            if candidate.id == leader.id
                        ),
                        None,
                    )
                decisions[role] = {
                    "leader_role": leader_role,
                    "gap_m": idm_gap if idm_gap is not None else geometric_gap,
                    "acceleration_mps2": acceleration,
                    "target_speed_mps": target_speed,
                }
            return decisions

        sensor_buffer = LatestSensorBuffer()
        sensors = _spawn_sensors(
            world, blueprints, ego, manifest, sensor_buffer,
            include_rgb=bool(args.record_frames),
        )
        actors.extend(sensors)

        _require_unbound_tcp_port(
            args.planner_host, args.planner_port, "planner service"
        )
        planner_process, planner_log, planner_command = _planner_process(args, output_dir)
        planner_client = PlannerClient(
            args.planner_host,
            args.planner_port,
            {
                "controller": args.controller,
                "run_id": planner_episode_id,
                "scenario_id": manifest["scenario_id"],
                "route_request": manifest["route_request"],
            },
            timeout_s=args.planner_timeout_s,
        )
        planner_client.start()
        planner_client.ready.wait(args.planner_timeout_s)
        _, planner_error = planner_client.get_latest()
        if planner_error is not None:
            raise RuntimeError("planner service failed: {}".format(planner_error))
        if not planner_client.ready.is_set():
            raise RuntimeError("planner service did not become ready")

        fixed_dt = float(manifest["physics_dt_s"])
        # Warm the CARLA powertrains/controllers before measurement.  A raw
        # velocity injection into a newly spawned vehicle otherwise overlaps
        # with automatic gear selection and creates a non-experimental speed
        # transient.  A second, short post-reset preroll below lets suspension
        # and longitudinal control settle at the measurement geometry.
        for _ in range(40):
            apply_scene_lane_speed_control(
                ego, actor_specs["ego"]["lane_id"],
                actor_specs["ego"]["lane_id"], 0.0, actor_specs["ego"]["speed_mps"]
            )
            apply_idm_population(fixed_dt, include_hidden=False)
            if blocker is not None:
                apply_scene_lane_speed_control(
                    blocker, actor_specs["blocker"]["lane_id"],
                    actor_specs["blocker"]["lane_id"], 0.0, actor_specs["blocker"]["speed_mps"]
                )
            if hidden is not None:
                hidden_spec = actor_specs["latent_vehicle"]
                apply_scene_lane_speed_control(
                    hidden, hidden_spec["lane_id"],
                    hidden_spec["lane_id"], 0.0, hidden_spec["speed_mps"]
                )
            world.tick()

        measurement_preroll_s = float(manifest.get("measurement_preroll_s", 0.0))

        def preroll_spec(spec):
            shifted = dict(spec)
            shifted["s_m"] = (
                float(spec["s_m"])
                - float(spec["speed_mps"]) * measurement_preroll_s
            )
            if shifted["s_m"] <= 0.5:
                raise RuntimeError(
                    "measurement preroll places {} outside road 40".format(
                        spec.get("label", spec.get("blueprint", "actor"))
                    )
                )
            return shifted

        _reset_actor_state(ego, map_obj, preroll_spec(actor_specs["ego"]))
        _reset_actor_state(truck, map_obj, preroll_spec(actor_specs["occluder"]))
        if blocker is not None:
            _reset_actor_state(blocker, map_obj, preroll_spec(actor_specs["blocker"]))
        if hidden is not None:
            _reset_actor_state(hidden, map_obj, preroll_spec(actor_specs["latent_vehicle"]))
        for _, npc, spec in idm_npcs:
            _reset_actor_state(npc, map_obj, preroll_spec(spec))

        for _ in range(int(round(measurement_preroll_s / fixed_dt))):
            apply_scene_lane_speed_control(
                ego, actor_specs["ego"]["lane_id"],
                actor_specs["ego"]["lane_id"], 0.0, actor_specs["ego"]["speed_mps"]
            )
            apply_idm_population(fixed_dt, include_hidden=False)
            if blocker is not None:
                apply_scene_lane_speed_control(
                    blocker, actor_specs["blocker"]["lane_id"],
                    actor_specs["blocker"]["lane_id"], 0.0, actor_specs["blocker"]["speed_mps"]
                )
            if hidden is not None:
                hidden_spec = actor_specs["latent_vehicle"]
                apply_scene_lane_speed_control(
                    hidden, hidden_spec["lane_id"],
                    hidden_spec["lane_id"], 0.0, hidden_spec["speed_mps"]
                )
            world.tick()

        initial_state = {}
        initial_actor_records = [
            ("ego", ego, actor_specs["ego"]),
            ("occluder", truck, actor_specs["occluder"]),
        ]
        if blocker is not None:
            initial_actor_records.append(("blocker", blocker, actor_specs["blocker"]))
        if hidden is not None:
            initial_actor_records.append(
                ("latent_vehicle", hidden, actor_specs["latent_vehicle"])
            )
        initial_actor_records.extend(
            (label, npc, spec) for label, npc, spec in idm_npcs
        )
        for role, actor, spec in initial_actor_records:
            packet = route_frame.actor_packet(actor, role)
            initial_state[role] = {
                "station_m": packet["station_m"],
                "expected_station_m": float(spec["s_m"]),
                "station_error_m": packet["station_m"] - float(spec["s_m"]),
                "speed_mps": packet["speed_mps"],
                "expected_speed_mps": float(spec["speed_mps"]),
                "speed_error_mps": packet["speed_mps"] - float(spec["speed_mps"]),
                "lateral_error_m": packet["lateral_error_m"],
            }
        sensor_buffer.clear_measurement_boundary()
        measurement_boundary_frame = int(world.get_snapshot().frame)

        total_ticks = int(round(float(manifest["duration_s"]) / fixed_dt))
        hidden_visible = False
        consecutive_visible_frames = 0
        reveal_time_s = None
        initial_hidden_hits = []
        initial_geometric_blocked = []
        initial_geometric_footprint_blocked = []
        initial_geometric_blocked_sample_fractions = []
        geometric_clear_streak = 0
        geometric_first_clear_time_s = None
        geometric_confirmed_reveal_time_s = None
        last_processed_lidar_frame = measurement_boundary_frame
        pre_reveal_hidden_leak_count = 0
        planner_condition_token_count = 0
        planner_oracle_metadata_count = 0
        seen_plan_metric_ids = set()
        last_plan_identity = None
        accepted_plan = None
        plan_rejection_reason = "no_plan"
        last_control = _speed_command(ego, actor_specs["ego"]["speed_mps"], 0.0)
        last_target_speed = float(actor_specs["ego"]["speed_mps"])
        shield_active = False
        low_level_time_s = 0.0
        last_speeds = {}
        history = []
        frame_count = 0
        hidden_control_detail = None
        first_hidden_observation_time_s = None
        first_hidden_aware_plan_applied_time_s = None
        bootstrap_visible_items = [("occluder", truck)] + list(idm_items)
        if blocker is not None:
            bootstrap_visible_items.append(("blocker", blocker))
        bootstrap_observation = _observation_payload(
            planner_episode_id,
            manifest["scenario_id"],
            measurement_boundary_frame,
            0.0,
            route_frame,
            ego,
            bootstrap_visible_items,
        )
        serialized_bootstrap = json.dumps(bootstrap_observation, sort_keys=True).lower()
        if str(args.condition).lower() in serialized_bootstrap:
            planner_condition_token_count += 1
        if "reveal_state" in bootstrap_observation or "latent_present" in serialized_bootstrap:
            planner_oracle_metadata_count += 1
        bootstrap_visible_ids = {
            int(packet["actor_id"])
            for packet in bootstrap_observation["visible_actors"]
        }
        if hidden is not None and int(hidden.id) in bootstrap_visible_ids:
            pre_reveal_hidden_leak_count += 1
        planner_client.submit(bootstrap_observation)
        bootstrap_deadline = time.perf_counter() + float(args.planner_timeout_s)
        bootstrap_plan = None
        while time.perf_counter() < bootstrap_deadline:
            bootstrap_plan, bootstrap_error = planner_client.get_latest()
            if bootstrap_error is not None:
                raise RuntimeError("planner bootstrap failed: {}".format(bootstrap_error))
            if bootstrap_plan is not None:
                break
            time.sleep(0.01)
        if bootstrap_plan is None:
            raise RuntimeError("planner bootstrap did not complete before timeout")
        bootstrap_plan_status = str(bootstrap_plan.get("status", "unknown"))
        wall_loop_start = time.perf_counter()

        csv_path = os.path.join(output_dir, "tick_trace.csv")
        csv_handle = open(csv_path, "w", newline="")
        actor_state_path = os.path.join(output_dir, "evaluator_actor_states.jsonl")
        actor_state_handle = open(actor_state_path, "w")
        npc_trace_path = os.path.join(output_dir, "npc_trace.csv")
        npc_trace_handle = open(npc_trace_path, "w", newline="")
        csv_fieldnames = [
            "frame", "time_s", "condition", "controller", "hidden_visible",
            "hidden_lidar_hits", "lidar_frame", "lidar_time_s", "lidar_is_new",
            "semantic_lidar_reveal_armed",
            "ego_station_m", "ego_local_y_m", "hidden_station_m", "hidden_local_y_m",
            "hidden_reference_y_m", "hidden_lateral_error_m", "hidden_commanded_ay_mps2",
            "hidden_road_lateral_velocity_mps",
            "hidden_observation_submitted", "hidden_aware_plan",
            "ego_speed_mps", "ego_accel_mps2",
            "ego_throttle", "ego_brake", "ego_steer", "commanded_ackermann_steer_rad",
            "target_speed_mps",
            "steer_limit_rad", "steering_envelope_active", "shield_active",
            "minimum_clearance_m", "minimum_ttc_s", "minimum_oriented_clearance_m",
            "minimum_oriented_clearance_actor_role", "minimum_ttc_2d_s",
            "minimum_ttc_2d_actor_role", "hidden_oriented_clearance_m", "hidden_ttc_2d_s",
            "plan_age_s", "plan_source_frame", "plan_status", "plan_rejection_reason",
            "new_plan", "latest_plan_status", "planner_total_s", "planner_decision_s",
            "planner_mpc_s", "planner_field_s", "low_level_updated", "low_level_time_s",
            "physics_control_cycle_time_s", "physics_control_deadline_miss",
            "occluder_station_m", "ego_pass_clearance_ahead_occluder_m",
            "hidden_pass_clearance_ahead_occluder_m", "ego_target_lane_error_m",
            "ego_centre_lane_overlap", "hidden_centre_lane_overlap",
            "concurrent_centre_lane_overlap",
            "hidden_center_geometric_visible_now", "hidden_center_los_blocked",
            "hidden_footprint_fully_blocked", "hidden_footprint_blocked_sample_count",
            "hidden_footprint_sample_count", "hidden_footprint_blocked_sample_fraction",
            "hidden_center_in_range", "hidden_center_in_horizontal_fov",
            "hidden_center_geometric_range_m", "hidden_center_geometric_bearing_rad",
            "hidden_center_geometric_clear_streak",
            "hidden_center_geometric_reveal_confirmed",
        ]
        for role, _, _ in controlled_npc_records:
            csv_fieldnames.extend(
                [
                    "{}_speed_mps".format(role),
                    "{}_accel_mps2".format(role),
                ]
            )
        writer = csv.DictWriter(csv_handle, fieldnames=csv_fieldnames)
        writer.writeheader()
        npc_trace_fieldnames = [
            "frame", "time_s", "condition", "controller", "role", "actor_id",
            "blueprint", "traffic_role", "initial_lane_id", "lane_name",
            "lane_index", "occupied_lane_indices", "station_m",
            "local_y_m", "speed_mps", "longitudinal_accel_mps2", "leader_role",
            "idm_gap_m", "target_speed_mps", "desired_speed_mps", "time_headway_s",
        ]
        npc_trace_writer = csv.DictWriter(
            npc_trace_handle, fieldnames=npc_trace_fieldnames
        )
        npc_trace_writer.writeheader()

        for tick_index in range(total_ticks):
            tick_wall_started = time.perf_counter()
            sim_time_s = tick_index * fixed_dt

            traffic_control_details = apply_idm_population(
                fixed_dt, include_hidden=bool(hidden_visible)
            )
            if blocker is not None:
                apply_scene_lane_speed_control(
                    blocker, actor_specs["blocker"]["lane_id"],
                    actor_specs["blocker"]["lane_id"], 0.0, actor_specs["blocker"]["speed_mps"]
                )
            if hidden is not None:
                hidden_spec = actor_specs["latent_vehicle"]
                hidden_control_detail = _apply_hidden_cut_in_control(
                    hidden, route_frame, hidden_spec, sim_time_s
                )

            if tick_index % 2 == 0:
                low_started = time.perf_counter()
                ego_packet = route_frame.ego_packet(ego)
                latest_plan, planner_error = planner_client.get_latest()
                if planner_error is not None:
                    plan_rejection_reason = "planner_error"
                if latest_plan is not None:
                    identity = (
                        latest_plan.get("source_frame_id"),
                        latest_plan.get("planning_end_ns"),
                    )
                    if identity != last_plan_identity:
                        selected, reason = _select_plan(latest_plan, sim_time_s, ego_packet)
                        if selected is not None:
                            accepted_plan = latest_plan
                        plan_rejection_reason = reason
                        last_plan_identity = identity
                selected, reason = _select_plan(accepted_plan, sim_time_s, ego_packet)
                if selected is None:
                    plan_rejection_reason = reason
                    fallback_station = min(
                        float(manifest["road_station_limit_m"]),
                        ego_packet["station_m"] + 20.0,
                    )
                    fallback = map_obj.get_waypoint_xodr(
                        int(manifest["road_id"]), int(manifest["lane_map"]["left"]),
                        fallback_station
                    ).transform.location
                    steer = _steer_to_location(ego, fallback)
                    last_target_speed = float(actor_specs["ego"]["speed_mps"])
                    plan_age_s = None
                    plan_source_frame = None
                    plan_status = "fallback_lane_hold"
                    feedforward_accel = 0.0
                else:
                    target_state = selected["target_state"]
                    target_transform = route_frame.local_to_world(
                        target_state["local_x_m"], target_state["local_y_m"],
                        target_state["local_yaw_rad"]
                    )
                    steer = _steer_to_location(ego, target_transform.location)
                    last_target_speed = max(0.0, float(selected["current_state"]["body_vx_mps"]))
                    feedforward_accel = 0.0
                    if selected["control"] is not None:
                        feedforward_accel = float(selected["control"]["acceleration_mps2"])
                    plan_age_s = selected["plan_age_s"]
                    plan_source_frame = int(accepted_plan["source_frame_id"])
                    plan_status = str(accepted_plan.get("status", "unknown"))
                    if (
                        hidden is not None
                        and "latent_vehicle" in accepted_plan.get("visible_actor_roles", [])
                        and first_hidden_aware_plan_applied_time_s is None
                    ):
                        first_hidden_aware_plan_applied_time_s = sim_time_s

                visible_items = [("occluder", truck)] + list(idm_items)
                if blocker is not None:
                    visible_items.append(("blocker", blocker))
                if hidden is not None and hidden_visible:
                    visible_items.append(("latent_vehicle", hidden))
                visible_packets = [route_frame.actor_packet(actor, role) for role, actor in visible_items]
                shield_active, shield_detail = _safety_supervisor(ego_packet, visible_packets)
                last_control = _speed_command(
                    ego, last_target_speed, steer, feedforward_accel
                )
                if shield_active:
                    last_control = carla.VehicleControl(throttle=0.0, brake=0.75)
                    last_target_speed = min(
                        last_target_speed,
                        float(shield_detail["actor"]["speed_mps"]),
                    )
                _apply_command(ego, last_control)
                low_level_time_s = time.perf_counter() - low_started

                observation = _observation_payload(
                    planner_episode_id,
                    manifest["scenario_id"],
                    world.get_snapshot().frame,
                    sim_time_s,
                    route_frame,
                    ego,
                    visible_items,
                )
                serialized_observation = json.dumps(observation, sort_keys=True).lower()
                if str(args.condition).lower() in serialized_observation:
                    planner_condition_token_count += 1
                if "reveal_state" in observation or "latent_present" in serialized_observation:
                    planner_oracle_metadata_count += 1
                visible_actor_ids = {
                    int(actor_packet["actor_id"])
                    for actor_packet in observation["visible_actors"]
                }
                if (
                    hidden is not None
                    and not hidden_visible
                    and int(hidden.id) in visible_actor_ids
                ):
                    pre_reveal_hidden_leak_count += 1
                planner_client.submit(observation)
                hidden_observation_submitted = bool(
                    hidden is not None
                    and any(
                        packet["role"] == "latent_vehicle"
                        for packet in observation["visible_actors"]
                    )
                )
                if (
                    hidden_observation_submitted
                    and first_hidden_observation_time_s is None
                ):
                    first_hidden_observation_time_s = sim_time_s
            else:
                _apply_command(ego, last_control)
                hidden_observation_submitted = False

            frame = int(world.tick())
            rgb, lidar_measurement, collisions = sensor_buffer.snapshot()
            lidar_frame = None if lidar_measurement is None else int(lidar_measurement.frame)
            lidar_is_new = bool(
                lidar_frame is not None
                and lidar_frame > measurement_boundary_frame
                and lidar_frame > last_processed_lidar_frame
            )
            lidar_time_s = None
            if lidar_is_new:
                last_processed_lidar_frame = lidar_frame
                lidar_time_s = (lidar_frame - measurement_boundary_frame) * fixed_dt
                lidar_array = _semantic_lidar_array(lidar_measurement)
            else:
                # A callback may still deliver the last unrecorded warm-up
                # measurement after the reset, or may lag the current world
                # tick.  Such a frame is neither used for reveal qualification
                # nor repeated as an independent sensor observation.
                lidar_array = np.empty(0, dtype=_SEMANTIC_LIDAR_DTYPE)
            hidden_hits = 0
            if hidden is not None and lidar_array.size:
                hidden_hits = int(np.count_nonzero(lidar_array["object_idx"] == int(hidden.id)))
            lidar_cfg = manifest["semantic_lidar"]
            semantic_lidar_reveal_armed = bool(
                lidar_time_s is not None
                and float(lidar_time_s)
                >= float(lidar_cfg.get("reveal_arming_delay_s", 0.0))
            )
            if lidar_is_new:
                if (
                    semantic_lidar_reveal_armed
                    and hidden is not None
                    and hidden_hits >= int(lidar_cfg["minimum_actor_hits"])
                ):
                    consecutive_visible_frames += 1
                else:
                    consecutive_visible_frames = 0
            if (
                hidden is not None
                and not hidden_visible
                and consecutive_visible_frames >= int(lidar_cfg["consecutive_frames"])
            ):
                hidden_visible = True
                reveal_time_s = lidar_time_s
            if (
                lidar_is_new
                and lidar_time_s <= float(manifest["qualification"]["minimum_initial_occlusion_s"])
            ):
                initial_hidden_hits.append(hidden_hits)

            ego_packet = route_frame.ego_packet(ego)
            actor_items = [("occluder", truck)] + list(idm_items)
            if blocker is not None:
                actor_items.append(("blocker", blocker))
            if hidden is not None:
                actor_items.append(("latent_vehicle", hidden))
            actor_packets = [route_frame.actor_packet(actor, role) for role, actor in actor_items]
            packet_by_role = {
                str(packet["role"]): packet for packet in actor_packets
            }
            hidden_packet = next(
                (packet for packet in actor_packets if packet["role"] == "latent_vehicle"),
                None,
            )
            occluder_packet = packet_by_role["occluder"]
            geometric_visibility = None
            if hidden_packet is not None:
                geometric_visibility = _hidden_centre_geometric_visibility(
                    ego_packet,
                    hidden_packet,
                    occluder_packet,
                    manifest["semantic_lidar"],
                )
                current_time_s = sim_time_s + fixed_dt
                if geometric_visibility["visible_now"]:
                    if geometric_first_clear_time_s is None:
                        geometric_first_clear_time_s = current_time_s
                    geometric_clear_streak += 1
                    if (
                        geometric_confirmed_reveal_time_s is None
                        and geometric_clear_streak >= 2
                    ):
                        geometric_confirmed_reveal_time_s = current_time_s
                else:
                    geometric_clear_streak = 0
                occlusion_evaluation_start_s = max(
                    float(manifest["semantic_lidar"].get("reveal_arming_delay_s", 0.0)),
                    float(manifest.get("rgb_camera", {}).get("recording_start_s", 0.0)),
                )
                if (
                    occlusion_evaluation_start_s <= current_time_s
                    <= float(manifest["qualification"]["minimum_initial_occlusion_s"])
                ):
                    initial_geometric_blocked.append(
                        bool(geometric_visibility["los_blocked"])
                    )
                    initial_geometric_footprint_blocked.append(
                        bool(geometric_visibility["footprint_fully_blocked"])
                    )
                    initial_geometric_blocked_sample_fractions.append(
                        float(geometric_visibility["blocked_sample_fraction"])
                    )
            ego_pass_clearance = (
                float(ego_packet["station_m"])
                - 0.5 * float(ego_packet["length_m"])
                - float(occluder_packet["station_m"])
                - 0.5 * float(occluder_packet["length_m"])
            )
            hidden_pass_clearance = None
            if hidden_packet is not None:
                hidden_pass_clearance = (
                    float(hidden_packet["station_m"])
                    - 0.5 * float(hidden_packet["length_m"])
                    - float(occluder_packet["station_m"])
                    - 0.5 * float(occluder_packet["length_m"])
                )
            ego_target_lane_error = abs(
                float(ego_packet["local_y_m"]) - PLANNER_LANE_CENTRES_Y[1]
            )
            ego_centre_lane_overlap = 1 in ego_packet.get(
                "occupied_lane_indices", []
            )
            hidden_centre_lane_overlap = bool(
                hidden_packet is not None
                and 1 in hidden_packet.get("occupied_lane_indices", [])
            )
            clearances = [_axis_aligned_clearance(ego_packet, packet) for packet in actor_packets]
            ttcs = [_longitudinal_ttc(ego_packet, packet) for packet in actor_packets]
            minimum_clearance = min(clearances) if clearances else None
            minimum_ttc = _finite_min(ttcs)
            oriented_clearances = [
                _oriented_box_clearance(ego_packet, packet) for packet in actor_packets
            ]
            ttcs_2d = [
                _two_dimensional_ttc(ego_packet, packet) for packet in actor_packets
            ]
            minimum_oriented_clearance = (
                min(oriented_clearances) if oriented_clearances else None
            )
            minimum_oriented_role = None
            if oriented_clearances:
                minimum_oriented_role = actor_packets[
                    int(np.argmin(np.asarray(oriented_clearances, dtype=float)))
                ]["role"]
            finite_ttc_2d_indices = [
                index for index, value in enumerate(ttcs_2d) if value is not None
            ]
            minimum_ttc_2d = None
            minimum_ttc_2d_role = None
            if finite_ttc_2d_indices:
                minimum_ttc_2d_index = min(
                    finite_ttc_2d_indices, key=lambda index: float(ttcs_2d[index])
                )
                minimum_ttc_2d = float(ttcs_2d[minimum_ttc_2d_index])
                minimum_ttc_2d_role = actor_packets[minimum_ttc_2d_index]["role"]
            hidden_oriented_clearance = (
                None
                if hidden_packet is None
                else _oriented_box_clearance(ego_packet, hidden_packet)
            )
            hidden_ttc_2d = (
                None
                if hidden_packet is None
                else _two_dimensional_ttc(ego_packet, hidden_packet)
            )
            ego_speed = float(ego_packet["speed_mps"])
            ego_accel = (ego_speed - last_speeds.get("ego", ego_speed)) / fixed_dt
            last_speeds["ego"] = ego_speed
            actual_ego_control = ego.get_control()
            commanded_ackermann_steer = (
                float(last_control.steer)
                if isinstance(last_control, carla.VehicleAckermannControl)
                else None
            )
            dynamic_steer_limit = math.atan(
                LOW_LEVEL_MAX_LATERAL_ACCEL_MPS2
                * LOW_LEVEL_WHEELBASE_M
                / max(ego_speed * ego_speed, 9.0)
            )
            dynamic_steer_limit = _clip(
                dynamic_steer_limit, math.radians(0.35), math.radians(10.0)
            )
            latest_plan, planner_error = planner_client.get_latest()
            latest_plan_identity = None
            is_new_plan = False
            if latest_plan is not None:
                latest_plan_identity = (
                    latest_plan.get("source_frame_id"),
                    latest_plan.get("planning_end_ns"),
                )
                if latest_plan_identity not in seen_plan_metric_ids:
                    seen_plan_metric_ids.add(latest_plan_identity)
                    is_new_plan = True
            plan_age_value = None
            if accepted_plan is not None:
                plan_age_value = sim_time_s + fixed_dt - float(accepted_plan["source_simulation_time_s"])
            physics_control_cycle_time_s = time.perf_counter() - tick_wall_started
            row = {
                "frame": frame,
                "time_s": sim_time_s + fixed_dt,
                "condition": args.condition,
                "controller": args.controller,
                "hidden_visible": int(hidden_visible),
                "hidden_lidar_hits": hidden_hits,
                "lidar_frame": lidar_frame,
                "lidar_time_s": lidar_time_s,
                "lidar_is_new": int(lidar_is_new),
                "semantic_lidar_reveal_armed": int(
                    semantic_lidar_reveal_armed
                ),
                "ego_station_m": ego_packet["station_m"],
                "ego_local_y_m": ego_packet["local_y_m"],
                "hidden_station_m": None if hidden_packet is None else hidden_packet["station_m"],
                "hidden_local_y_m": None if hidden_packet is None else hidden_packet["local_y_m"],
                "hidden_reference_y_m": (
                    None if hidden_control_detail is None
                    else hidden_control_detail["reference_y_m"]
                ),
                "hidden_lateral_error_m": (
                    None if hidden_control_detail is None
                    else hidden_control_detail["lateral_error_m"]
                ),
                "hidden_commanded_ay_mps2": (
                    None if hidden_control_detail is None
                    else hidden_control_detail["commanded_ay_mps2"]
                ),
                "hidden_road_lateral_velocity_mps": (
                    None if hidden_control_detail is None
                    else hidden_control_detail["road_lateral_velocity_mps"]
                ),
                "hidden_observation_submitted": int(hidden_observation_submitted),
                "hidden_aware_plan": int(
                    accepted_plan is not None
                    and "latent_vehicle" in accepted_plan.get("visible_actor_roles", [])
                ),
                "ego_speed_mps": ego_speed,
                "ego_accel_mps2": ego_accel,
                "ego_throttle": float(actual_ego_control.throttle),
                "ego_brake": float(actual_ego_control.brake),
                "ego_steer": float(actual_ego_control.steer),
                "commanded_ackermann_steer_rad": commanded_ackermann_steer,
                "target_speed_mps": last_target_speed,
                "steer_limit_rad": dynamic_steer_limit,
                "steering_envelope_active": int(
                    commanded_ackermann_steer is not None
                    and abs(commanded_ackermann_steer) >= 0.95 * dynamic_steer_limit
                ),
                "shield_active": int(shield_active),
                "minimum_clearance_m": minimum_clearance,
                "minimum_ttc_s": minimum_ttc,
                "minimum_oriented_clearance_m": minimum_oriented_clearance,
                "minimum_oriented_clearance_actor_role": minimum_oriented_role,
                "minimum_ttc_2d_s": minimum_ttc_2d,
                "minimum_ttc_2d_actor_role": minimum_ttc_2d_role,
                "hidden_oriented_clearance_m": hidden_oriented_clearance,
                "hidden_ttc_2d_s": hidden_ttc_2d,
                "plan_age_s": plan_age_value,
                "plan_source_frame": None if accepted_plan is None else accepted_plan.get("source_frame_id"),
                "plan_status": "none" if accepted_plan is None else accepted_plan.get("status"),
                "plan_rejection_reason": plan_rejection_reason,
                "new_plan": int(is_new_plan),
                "latest_plan_status": None if latest_plan is None else latest_plan.get("status"),
                "planner_total_s": None if latest_plan is None else latest_plan.get("planning_total_s"),
                "planner_decision_s": None if latest_plan is None else latest_plan.get("decision_time_s"),
                "planner_mpc_s": None if latest_plan is None else latest_plan.get("mpc_time_s"),
                "planner_field_s": None if latest_plan is None else latest_plan.get("field_time_s"),
                "low_level_updated": int(tick_index % 2 == 0),
                "low_level_time_s": low_level_time_s,
                "physics_control_cycle_time_s": physics_control_cycle_time_s,
                "physics_control_deadline_miss": int(
                    physics_control_cycle_time_s > fixed_dt
                ),
                "occluder_station_m": occluder_packet["station_m"],
                "ego_pass_clearance_ahead_occluder_m": ego_pass_clearance,
                "hidden_pass_clearance_ahead_occluder_m": hidden_pass_clearance,
                "ego_target_lane_error_m": ego_target_lane_error,
                "ego_centre_lane_overlap": int(ego_centre_lane_overlap),
                "hidden_centre_lane_overlap": int(hidden_centre_lane_overlap),
                "concurrent_centre_lane_overlap": int(
                    ego_centre_lane_overlap and hidden_centre_lane_overlap
                ),
                "hidden_center_geometric_visible_now": (
                    None if geometric_visibility is None
                    else int(geometric_visibility["visible_now"])
                ),
                "hidden_center_los_blocked": (
                    None if geometric_visibility is None
                    else int(geometric_visibility["los_blocked"])
                ),
                "hidden_footprint_fully_blocked": (
                    None if geometric_visibility is None
                    else int(geometric_visibility["footprint_fully_blocked"])
                ),
                "hidden_footprint_blocked_sample_count": (
                    None if geometric_visibility is None
                    else geometric_visibility["blocked_sample_count"]
                ),
                "hidden_footprint_sample_count": (
                    None if geometric_visibility is None
                    else geometric_visibility["footprint_sample_count"]
                ),
                "hidden_footprint_blocked_sample_fraction": (
                    None if geometric_visibility is None
                    else geometric_visibility["blocked_sample_fraction"]
                ),
                "hidden_center_in_range": (
                    None if geometric_visibility is None
                    else int(geometric_visibility["in_range"])
                ),
                "hidden_center_in_horizontal_fov": (
                    None if geometric_visibility is None
                    else int(geometric_visibility["in_horizontal_fov"])
                ),
                "hidden_center_geometric_range_m": (
                    None if geometric_visibility is None
                    else geometric_visibility["range_m"]
                ),
                "hidden_center_geometric_bearing_rad": (
                    None if geometric_visibility is None
                    else geometric_visibility["bearing_rad"]
                ),
                "hidden_center_geometric_clear_streak": (
                    None if geometric_visibility is None else geometric_clear_streak
                ),
                "hidden_center_geometric_reveal_confirmed": int(
                    geometric_confirmed_reveal_time_s is not None
                ),
            }
            for role, _, _ in controlled_npc_records:
                packet = packet_by_role.get(str(role))
                row["{}_speed_mps".format(role)] = (
                    None if packet is None else packet.get("speed_mps")
                )
                row["{}_accel_mps2".format(role)] = (
                    None if packet is None
                    else packet.get("longitudinal_accel_mps2")
                )
            writer.writerow(row)
            for role, actor, spec in controlled_npc_records:
                packet = packet_by_role.get(str(role))
                if packet is None:
                    continue
                detail = traffic_control_details.get(str(role), {})
                idm = spec.get("idm", {})
                npc_trace_writer.writerow(
                    {
                        "frame": frame,
                        "time_s": sim_time_s + fixed_dt,
                        "condition": args.condition,
                        "controller": args.controller,
                        "role": role,
                        "actor_id": int(actor.id),
                        "blueprint": str(actor.type_id),
                        "traffic_role": spec.get("traffic_role", "occluder"),
                        "initial_lane_id": spec.get("lane_id"),
                        "lane_name": spec.get("lane_name", "centre" if role == "occluder" else None),
                        "lane_index": packet.get("lane_index"),
                        "occupied_lane_indices": json.dumps(
                            packet.get("occupied_lane_indices", []),
                            separators=(",", ":"),
                        ),
                        "station_m": packet.get("station_m"),
                        "local_y_m": packet.get("local_y_m"),
                        "speed_mps": packet.get("speed_mps"),
                        "longitudinal_accel_mps2": packet.get(
                            "longitudinal_accel_mps2"
                        ),
                        "leader_role": detail.get("leader_role"),
                        "idm_gap_m": detail.get("gap_m"),
                        "target_speed_mps": detail.get("target_speed_mps"),
                        "desired_speed_mps": idm.get(
                            "desired_speed_mps", spec.get("speed_mps")
                        ),
                        "time_headway_s": idm.get("time_headway_s"),
                    }
                )
                npc_trace_row_count += 1
            actor_state_handle.write(
                json.dumps(
                    {
                        "frame": frame,
                        "time_s": sim_time_s + fixed_dt,
                        "ego": ego_packet,
                        "actors": actor_packets,
                        "planner_hidden_visible": bool(hidden_visible),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            tick_rows.append(row)
            history.append(row)

            if (
                args.record_frames
                and sim_time_s + fixed_dt
                >= float(manifest["rgb_camera"].get("recording_start_s", 0.0))
                and tick_index % int(args.frame_stride) == 0
            ):
                telemetry = dict(row)
                telemetry["hidden_lidar_hits"] = hidden_hits
                visual_items = [("occluder", truck)] + list(idm_items)
                if blocker is not None:
                    visual_items.append(("blocker", blocker))
                if hidden is not None:
                    visual_items.append(("latent_vehicle", hidden))
                composite, driver_frame, bev_frame = _compose_frame(
                    rgb,
                    lidar_array,
                    route_frame,
                    visual_items,
                    ego,
                    hidden,
                    hidden_visible,
                    latest_plan,
                    telemetry,
                    history,
                    manifest,
                )
                pygame.image.save(
                    composite,
                    os.path.join(frames_dir, "frame_{:05d}.png".format(frame_count)),
                )
                pygame.image.save(
                    driver_frame,
                    os.path.join(
                        driver_frames_dir, "frame_{:05d}.png".format(frame_count)
                    ),
                )
                pygame.image.save(
                    bev_frame,
                    os.path.join(bev_frames_dir, "frame_{:05d}.png".format(frame_count)),
                )
                frame_count += 1

            if args.pace_realtime:
                target_wall = wall_loop_start + (tick_index + 1) * fixed_dt
                remaining = target_wall - time.perf_counter()
                if remaining > 0.0:
                    time.sleep(remaining)

        csv_handle.flush()
        actor_state_handle.flush()
        npc_trace_handle.flush()
        elapsed_wall = time.perf_counter() - wall_loop_start
        simulated_duration = total_ticks * fixed_dt
        real_time_factor = simulated_duration / max(elapsed_wall, 1e-9)
        low_level_execution_count = sum(
            1 for row in tick_rows if row.get("low_level_updated")
        )
        actor_geometry = {}
        geometry_actors = [("ego", ego), ("occluder", truck)] + list(idm_items)
        if blocker is not None:
            geometry_actors.append(("blocker", blocker))
        for role, actor in geometry_actors:
            extent = actor.bounding_box.extent
            actor_geometry[role] = {
                "type_id": actor.type_id,
                "length_m": 2.0 * float(extent.x),
                "width_m": 2.0 * float(extent.y),
                "height_m": 2.0 * float(extent.z),
            }
        if hidden is not None:
            extent = hidden.bounding_box.extent
            actor_geometry["latent_vehicle"] = {
                "type_id": hidden.type_id,
                "length_m": 2.0 * float(extent.x),
                "width_m": 2.0 * float(extent.y),
                "height_m": 2.0 * float(extent.z),
            }
        scene_construction = manifest.get("scene_construction", {})
        declared_construction_hash = scene_construction.get(
            "construction_hash_sha256"
        )
        calculated_construction_hash = None
        if scene_construction:
            calculated_construction_hash = construction_hash(manifest)
        seeded_construction_provenance_pass = bool(
            scene_construction
            and declared_construction_hash == calculated_construction_hash
            and int(scene_construction.get("seed")) == int(args.seed)
            and str(scene_construction.get("generator_version", "")).strip()
        )
        occluder_geometry = actor_geometry["occluder"]
        occluder_asset_pass = bool(
            occluder_geometry["type_id"] == "vehicle.carlamotors.firetruck"
            and float(occluder_geometry["length_m"])
            >= float(manifest["qualification"]["minimum_occluder_length_m"])
            and float(occluder_geometry["height_m"])
            >= float(manifest["qualification"]["minimum_occluder_height_m"])
        )
        centre_carla_lane = int(manifest["lane_map"]["centre"])
        centre_planner_lane = int(manifest["planner_lane_map"]["centre"])
        converging_route_definition_pass = bool(
            int(actor_specs["ego"]["lane_id"]) == int(manifest["lane_map"]["left"])
            and int(actor_specs["latent_vehicle"]["lane_id"])
            == int(manifest["lane_map"]["right"])
            and int(actor_specs["latent_vehicle"]["target_lane_id"])
            == centre_carla_lane
            and int(manifest["route_request"]["target_lane"])
            == centre_planner_lane
        )
        expected_controlled_npc_count = scene_construction.get(
            "realized_factors", {}
        ).get("idm_npc_count")
        actor_population_pass = bool(
            expected_controlled_npc_count is not None
            and int(expected_controlled_npc_count) == len(controlled_npc_records)
        )
        planner_metric_completeness_pass = any(
            row.get("new_plan") and row.get("planner_total_s") is not None
            for row in tick_rows
        )
        measurement_protocol_pass = bool(
            args.pace_realtime and not args.record_frames
        )
        expected_npc_trace_rows = total_ticks * len(controlled_npc_records)
        primary_initial_roles = ["ego", "occluder"]
        if hidden is not None:
            primary_initial_roles.append("latent_vehicle")
        maximum_primary_station_error_m = max(
            abs(initial_state[role]["station_error_m"])
            for role in primary_initial_roles
        )
        maximum_primary_speed_error_mps = max(
            abs(initial_state[role]["speed_error_mps"])
            for role in primary_initial_roles
        )
        maximum_background_station_error_m = max(
            [
                abs(record["station_error_m"])
                for role, record in initial_state.items()
                if role not in primary_initial_roles
            ]
            or [0.0]
        )
        maximum_background_speed_error_mps = max(
            [
                abs(record["speed_error_mps"])
                for role, record in initial_state.items()
                if role not in primary_initial_roles
            ]
            or [0.0]
        )
        final_hidden_packet = (
            None
            if hidden is None
            else route_frame.actor_packet(hidden, "latent_vehicle")
        )
        final_hidden_lateral_error_m = (
            None
            if final_hidden_packet is None
            else abs(
                float(final_hidden_packet["local_y_m"])
                - PLANNER_LANE_CENTRES_Y[
                    int(actor_specs["latent_vehicle"].get("target_planner_lane", 1))
                ]
            )
        )
        final_hidden_lane_containment_margin_m = (
            None
            if final_hidden_packet is None
            else 1.75
            - final_hidden_lateral_error_m
            - 0.5 * float(final_hidden_packet["width_m"])
        )
        qualification = {
            "initial_occlusion_pass": (
                hidden is None
                or reveal_time_s is None
                or float(reveal_time_s)
                > float(manifest["qualification"]["minimum_initial_occlusion_s"])
            ),
            "initial_hidden_center_geometric_occlusion_pass": (
                hidden is None
                or (
                    bool(initial_geometric_blocked)
                    and all(initial_geometric_blocked)
                )
            ),
            "initial_hidden_footprint_geometric_occlusion_pass": (
                hidden is None
                or (
                    bool(initial_geometric_footprint_blocked)
                    and all(initial_geometric_footprint_blocked)
                )
            ),
            "initial_hidden_footprint_minimum_blocked_sample_fraction": (
                None
                if hidden is None or not initial_geometric_blocked_sample_fractions
                else min(initial_geometric_blocked_sample_fractions)
            ),
            "initial_occlusion_evaluation_start_s": max(
                float(manifest["semantic_lidar"].get("reveal_arming_delay_s", 0.0)),
                float(manifest.get("rgb_camera", {}).get("recording_start_s", 0.0)),
            ),
            "geometric_reveal_observed": (
                hidden is None or geometric_confirmed_reveal_time_s is not None
            ),
            "geometric_reveal_first_clear_time_s": geometric_first_clear_time_s,
            "geometric_reveal_confirmed_time_s": geometric_confirmed_reveal_time_s,
            "valid_reveal_pass": (
                hidden is None or (
                    reveal_time_s is not None
                    and float(manifest["qualification"]["earliest_reveal_s"]) <= reveal_time_s
                    <= float(manifest["qualification"]["latest_reveal_s"])
                )
            ),
            "hidden_actor_never_in_planner_before_reveal": pre_reveal_hidden_leak_count == 0,
            "pre_reveal_hidden_leak_count": pre_reveal_hidden_leak_count,
            "condition_blind_planner_interface": planner_condition_token_count == 0,
            "planner_condition_token_count": planner_condition_token_count,
            "oracle_metadata_absent_from_planner": planner_oracle_metadata_count == 0,
            "planner_oracle_metadata_count": planner_oracle_metadata_count,
            "strict_no_oracle_pass": (
                pre_reveal_hidden_leak_count == 0
                and planner_condition_token_count == 0
                and planner_oracle_metadata_count == 0
            ),
            "paired_manifest_condition_only": bool(
                seeded_construction_provenance_pass and frozen_manifest_input_pass
            ),
            "npc_hidden_coupling_policy": (
                "withheld_from_global_idm_until_ego_semantic_lidar_reveal"
            ),
            "maximum_initial_hidden_lidar_hits": max(initial_hidden_hits or [0]),
            "semantic_lidar_reveal_arming_delay_s": float(
                manifest["semantic_lidar"].get("reveal_arming_delay_s", 0.0)
            ),
            "initial_station_pass": maximum_primary_station_error_m
            <= float(manifest["qualification"]["maximum_initial_station_error_m"]),
            "maximum_primary_initial_station_error_m": maximum_primary_station_error_m,
            "maximum_background_initial_station_error_m": maximum_background_station_error_m,
            "initial_speed_pass": maximum_primary_speed_error_mps
            <= float(manifest["qualification"]["maximum_actor_speed_error_mps"]),
            "maximum_primary_initial_speed_error_mps": maximum_primary_speed_error_mps,
            "maximum_background_initial_speed_error_mps": maximum_background_speed_error_mps,
            "cut_in_complete_pass": (
                hidden is None
                or final_hidden_lane_containment_margin_m >= 0.0
            ),
            "cut_in_completion_is_outcome_not_eligibility_gate": True,
            "cut_in_reference_tracking_pass": (
                hidden is None
                or final_hidden_lateral_error_m
                <= float(manifest["qualification"]["maximum_cut_in_lateral_error_m"])
            ),
            "final_cut_in_lateral_error_m": final_hidden_lateral_error_m,
            "final_cut_in_lane_containment_margin_m": (
                final_hidden_lane_containment_margin_m
            ),
            "real_time_pacing_enabled": bool(args.pace_realtime),
            "measurement_run_without_frame_encoding": not bool(args.record_frames),
            "measurement_protocol_pass": measurement_protocol_pass,
            "real_time_factor_pass": (
                bool(args.pace_realtime)
                and not bool(args.record_frames)
                and 0.95 <= real_time_factor <= 1.05
            ),
            "real_time_claim_eligible": bool(
                measurement_protocol_pass and 0.95 <= real_time_factor <= 1.05
            ),
            "low_level_10hz_pass": (
                low_level_execution_count == int(round(simulated_duration * 10.0))
            ),
            "metric_completeness_pass": (
                bool(tick_rows)
                and all(row.get("minimum_oriented_clearance_m") is not None for row in tick_rows)
            ),
            "planner_metric_completeness_pass": planner_metric_completeness_pass,
            "bootstrap_plan_available": bootstrap_plan is not None,
            "bootstrap_plan_status": bootstrap_plan_status,
            "bootstrap_nominal_plan_pass": bootstrap_plan_status == "ok",
            "seeded_construction_provenance_pass": seeded_construction_provenance_pass,
            "construction_hash_pass": bool(
                declared_construction_hash == calculated_construction_hash
            ),
            "recomputed_construction_hash_sha256": calculated_construction_hash,
            "frozen_manifest_input_pass": frozen_manifest_input_pass,
            "occluder_asset_and_geometry_pass": occluder_asset_pass,
            "converging_route_definition_pass": converging_route_definition_pass,
            "actor_population_pass": actor_population_pass,
            "npc_trace_expected_rows": expected_npc_trace_rows,
            "npc_trace_observed_rows": npc_trace_row_count,
            "npc_trace_complete_pass": (
                npc_trace_row_count == expected_npc_trace_rows
            ),
            "seed_varies_physical_construction": seeded_construction_provenance_pass,
            "statistical_bank_ready": False,
        }
        qualification["statistical_bank_ready"] = all(
            qualification[key]
            for key in (
                "seeded_construction_provenance_pass",
                "construction_hash_pass",
                "frozen_manifest_input_pass",
                "paired_manifest_condition_only",
                "occluder_asset_and_geometry_pass",
                "converging_route_definition_pass",
                "actor_population_pass",
                "npc_trace_complete_pass",
                "metric_completeness_pass",
                "planner_metric_completeness_pass",
            )
        )
        qualification["valid_for_analysis"] = all(
            qualification[key]
            for key in (
                "initial_occlusion_pass",
                "initial_hidden_center_geometric_occlusion_pass",
                "initial_hidden_footprint_geometric_occlusion_pass",
                "strict_no_oracle_pass",
                "initial_station_pass",
                "initial_speed_pass",
                "measurement_protocol_pass",
                "low_level_10hz_pass",
                "metric_completeness_pass",
                "planner_metric_completeness_pass",
                "bootstrap_nominal_plan_pass",
                "statistical_bank_ready",
            )
        )
        runtime = {
            "simulated_duration_s": simulated_duration,
            "unrecorded_measurement_preroll_s": measurement_preroll_s,
            "wall_duration_s": elapsed_wall,
            "real_time_factor": real_time_factor,
            "recorded_frames": frame_count,
            "rgb_sensor_enabled": bool(args.record_frames),
            "rendering_and_measurement_separated": True,
            "command": experiment_command,
            "planner_command": planner_command,
            "carla_command": carla_command,
            "python_version": sys.version,
            "carla_client_version": client.get_client_version(),
            "carla_server_version": client.get_server_version(),
        }
        summary = _summary(
            tick_rows,
            manifest,
            args.condition,
            args.controller,
            reveal_time_s,
            collisions,
            planner_client,
            runtime,
            qualification,
            actor_geometry,
            initial_state,
        )
        _write_json(os.path.join(output_dir, "summary.json"), summary)
        _write_json(
            os.path.join(output_dir, "provenance.json"),
            {
                "run_id": run_id,
                "planner_episode_id": planner_episode_id,
                "experiment_command": experiment_command,
                "planner_command": planner_command,
                "carla_command": carla_command,
                "manifest_source": os.path.abspath(args.manifest),
                "manifest_source_schema": manifest_source_schema,
                "manifest_source_sha256": manifest_source_sha256,
                "resolved_manifest": os.path.join(
                    output_dir, "resolved_manifest.json"
                ),
                "construction_hash": declared_construction_hash,
                "scene_seed": scene_construction.get("seed"),
                "generator_version": scene_construction.get("generator_version"),
                "scenario_family": manifest.get("scenario_family"),
                "output_dir": output_dir,
                "evaluator_actor_states": actor_state_path,
                "npc_trace": npc_trace_path,
                "tick_trace": csv_path,
                "pre_reveal_npc_hidden_coupling_policy": (
                    "cue_suppressed_until_ego_semantic_lidar_reveal"
                ),
                "evaluator_actor_states_are_planner_inaccessible": True,
            },
        )
        return output_dir, summary
    finally:
        if csv_handle is not None:
            csv_handle.close()
        if actor_state_handle is not None:
            actor_state_handle.close()
        if npc_trace_handle is not None:
            npc_trace_handle.close()
        if planner_client is not None:
            planner_client.stop()
            planner_client.join(timeout=5.0)
        if planner_process is not None and planner_process.poll() is None:
            planner_process.terminate()
            try:
                planner_process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                planner_process.kill()
        if planner_log is not None:
            planner_log.close()
        for sensor in sensors:
            try:
                sensor.stop()
            except Exception:
                pass
        for actor in reversed(actors):
            try:
                actor.destroy()
            except Exception:
                pass
        if world is not None and original_settings is not None:
            try:
                world.apply_settings(original_settings)
            except Exception:
                pass
        if carla_process is not None and carla_process.poll() is None:
            carla_process.terminate()
            try:
                carla_process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                carla_process.kill()
        pygame.quit()


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--controller", choices=CONTROLLERS, default="DREAM")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument(
        "--manifest",
        default=os.path.join(
            os.path.dirname(__file__), "carla_converging_overtake_manifest.json"
        ),
    )
    parser.add_argument(
        "--output-root",
        default=os.path.join(REPO_ROOT, "outputs", "carla_overtaking_pilot"),
    )
    parser.add_argument("--duration-s", type=float, default=None)
    parser.add_argument("--record-frames", action="store_true")
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--pace-realtime", action="store_true")
    parser.add_argument(
        "--allow-invalid",
        action="store_true",
        help="return success for calibration/visualization runs that fail evidence gates",
    )
    parser.add_argument("--carla-host", default="127.0.0.1")
    parser.add_argument("--carla-port", type=int, default=2057)
    parser.add_argument("--server-timeout-s", type=float, default=120.0)
    parser.add_argument("--launch-server", action="store_true")
    parser.add_argument(
        "--carla-executable",
        default=(
            "C:\\CARLA_0.9.14\\WindowsNoEditor\\CarlaUE4\\Binaries\\Win64\\"
            "CarlaUE4-Win64-Shipping.exe"
        ),
    )
    parser.add_argument("--quality-level", choices=("Low", "Epic"), default="Low")
    parser.add_argument("--planner-host", default="127.0.0.1")
    parser.add_argument("--planner-port", type=int, default=8765)
    parser.add_argument("--planner-timeout-s", type=float, default=120.0)
    parser.add_argument(
        "--planner-python",
        default="C:\\Users\\ymshu\\anaconda3\\python.exe",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    np.random.seed(args.seed)
    output_dir, summary = run_trial(args)
    print(json.dumps({"output_dir": output_dir, "summary": summary}, indent=2))
    if not summary["qualification"].get("valid_for_analysis", False):
        print(
            "Run retained for diagnostics but failed evidence qualification: {}".format(
                output_dir
            ),
            file=sys.stderr,
        )
        return 0 if args.allow_invalid else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
