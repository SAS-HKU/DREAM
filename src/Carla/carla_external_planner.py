"""External-physics DREAM/IDEAM planner service for CARLA experiments.

This process runs in the repository's scientific Python environment.  CARLA
owns physical propagation in a separate Python 3.7 process; this module only
receives timestamped, planner-frame observations and returns timestamped MPC
trajectories.  Hidden actors must be absent from ``visible_actors`` until the
CARLA bridge's sensor visibility latch fires.

The first supported road is the straight three-lane Town06 overtaking trial.
Its local coordinates are deliberately embedded inside the existing DRIFT
world grid so the submitted risk-field implementation is exercised unchanged.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import math
import socket
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path as FilePath
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


REPO_ROOT = FilePath(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from Integration.carla_protocol import (  # noqa: E402
        ConnectionClosed,
        ProtocolError,
        make_message,
        recv_message,
        send_message,
        validate_message,
    )
except ImportError:  # Packaged ``src/Carla`` reproducibility layout.
    from carla_protocol import (  # type: ignore  # noqa: E402
        ConnectionClosed,
        ProtocolError,
        make_message,
        recv_message,
        send_message,
        validate_message,
    )
from Integration.episode_control import (  # noqa: E402
    CouplingFlags,
    LaneTraffic,
    ManeuverRequest,
    PlannerState,
    RoadContext,
    create_ideam_episode_arm,
    create_prideam_episode_arm,
)
from Path.path import Path as DreamPath  # noqa: E402
from pde_solver import create_vehicle as create_drift_vehicle  # noqa: E402
from Aggressiveness_Modeling.ADA_drift_source import compute_Q_ADA  # noqa: E402
from APF_Modeling.APF_drift_source import compute_Q_APF  # noqa: E402


PLANNER_DT_S = 0.1
SUPPORTED_CONTROLLERS = ("DREAM", "IDEAM", "ADA", "APF")
LANE_CENTRES_Y = (-201.75, -205.25, -208.75)
PATH_TRANSLATION_X = -200.0
RISK_WEIGHTS = {
    "mpc_cost": 0.5,
    "cbf_modulation": 0.6,
    "decision_threshold": 1.5,
    "headway_modulation": 0.4,
    "max_cbf_scale": 2.5,
    "max_headway_scale": 2.0,
    "cbf_risk_normalization": 1.5,
}
TRAFFIC_ADAPTER_PROVENANCE = {
    "lane_row_schema": "[s, ey, epsi, x, y, psi, road_longitudinal_v, road_longitudinal_a]",
    "velocity_transform": "CARLA body-frame velocity transformed into the planner-local frame",
    "drift_velocity_model": "measured planner-local longitudinal and lateral velocity",
    "lane_prediction_model": "legacy constant-lateral-offset horizon with measured current ey and epsi",
    "lane_prediction_limitation": (
        "LaneTraffic has no lateral-velocity column; lateral cut-in motion is refreshed from each "
        "observation but is not extrapolated across the legacy MPC traffic horizon."
    ),
}


def _wrap_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def _positive_dimension(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a positive finite number")
    dimension = float(value)
    if not math.isfinite(dimension) or dimension <= 0.0:
        raise ValueError(f"{label} must be a positive finite number")
    return dimension


def _ego_geometry(observation: Mapping[str, Any]) -> tuple[float, float]:
    ego = observation["ego"]
    return (
        _positive_dimension(ego["length_m"], label="ego.length_m"),
        _positive_dimension(ego["width_m"], label="ego.width_m"),
    )


def _local_velocity_components(actor: Mapping[str, Any]) -> tuple[float, float]:
    """Return the measured actor velocity in the straight-road planner frame.

    CARLA observations expose velocity in the actor body frame.  Rotating both
    components by the measured local heading retains the lateral motion of a
    cut-in.  The magnitude-only fallback supports older observation fixtures.
    """
    heading = float(actor["local_yaw_rad"])
    if "body_vx_mps" in actor and "body_vy_mps" in actor:
        body_vx = float(actor["body_vx_mps"])
        body_vy = float(actor["body_vy_mps"])
        local_vx = body_vx * math.cos(heading) - body_vy * math.sin(heading)
        local_vy = body_vx * math.sin(heading) + body_vy * math.cos(heading)
    else:
        speed = float(actor["speed_mps"])
        local_vx = speed * math.cos(heading)
        local_vy = speed * math.sin(heading)
    if not math.isfinite(local_vx) or not math.isfinite(local_vy):
        raise ValueError("actor velocity components must be finite")
    return local_vx, local_vy


def _road_longitudinal_acceleration(actor: Mapping[str, Any]) -> float:
    acceleration = float(actor.get("longitudinal_accel_mps2", 0.0))
    projected = acceleration * math.cos(float(actor["local_yaw_rad"]))
    if not math.isfinite(projected):
        raise ValueError("actor longitudinal acceleration must be finite")
    return projected


def _straight_road() -> RoadContext:
    paths = {
        lane: DreamPath(
            l1=1000.0,
            l2=1.0,
            r=100.0,
            traslx=PATH_TRANSLATION_X,
            trasly=centre_y,
        )
        for lane, centre_y in enumerate(LANE_CENTRES_Y)
    }
    samples = np.arange(0.0, 500.1, 0.1)
    x_lists: dict[int, np.ndarray] = {}
    y_lists: dict[int, np.ndarray] = {}
    sample_map: dict[int, np.ndarray] = {}
    for lane, path in paths.items():
        coords = np.asarray([path(float(station)) for station in samples], dtype=float)
        sample_map[lane] = samples.copy()
        x_lists[lane] = coords[:, 0]
        y_lists[lane] = coords[:, 1]

    def lane_lookup(pose: Sequence[float]) -> int:
        y_value = float(pose[1])
        return int(np.argmin(np.abs(np.asarray(LANE_CENTRES_Y) - y_value)))

    return RoadContext(
        paths=paths,
        samples=sample_map,
        x_lists=x_lists,
        y_lists=y_lists,
        lane_lookup=lane_lookup,
        boundary=1.0,
        dt=PLANNER_DT_S,
    )


def _shift_sequence(values: Any, count: int) -> Any:
    if count <= 0 or values is None:
        return values
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0:
        return array.copy()
    shift = min(int(count), int(array.size - 1))
    shifted = np.empty_like(array)
    shifted[:-shift] = array[shift:]
    shifted[-shift:] = array[-1]
    return shifted


def _age_warm_start(state: PlannerState, elapsed_s: float) -> PlannerState:
    count = int(max(0.0, math.floor(float(elapsed_s) / PLANNER_DT_S + 1e-9)))
    if count <= 0:
        return state
    state.oa = _shift_sequence(state.oa, count)
    state.od = _shift_sequence(state.od, count)
    if state.last_X is not None:
        state.last_X = [_shift_sequence(values, count) for values in state.last_X]
    return state


def _planner_state(observation: Mapping[str, Any], previous: PlannerState | None) -> PlannerState:
    ego = observation["ego"]
    x0 = [
        float(ego["body_vx_mps"]),
        float(ego.get("body_vy_mps", 0.0)),
        float(ego.get("yaw_rate_rps", 0.0)),
        float(ego["station_m"]),
        float(ego["lateral_error_m"]),
        float(ego["heading_error_rad"]),
    ]
    x0_g = [float(ego["local_x_m"]), float(ego["local_y_m"]), float(ego["local_yaw_rad"])]
    if previous is None:
        lane = int(ego["lane_index"])
        return PlannerState(
            X0=x0,
            X0_g=x0_g,
            oa=0.0,
            od=0.0,
            last_X=None,
            path_changed=lane,
        )
    return PlannerState(
        X0=x0,
        X0_g=x0_g,
        oa=previous.oa,
        od=previous.od,
        last_X=previous.last_X,
        path_changed=previous.path_changed,
    )


def _actor_row(actor: Mapping[str, Any]) -> np.ndarray:
    road_vx, _ = _local_velocity_components(actor)
    return np.asarray(
        [
            actor["station_m"],
            actor["lateral_error_m"],
            actor["heading_error_rad"],
            actor["local_x_m"],
            actor["local_y_m"],
            actor["local_yaw_rad"],
            road_vx,
            _road_longitudinal_acceleration(actor),
        ],
        dtype=float,
    )


def _lane_traffic(actors: Iterable[Mapping[str, Any]]) -> LaneTraffic:
    rows: dict[int, list[np.ndarray]] = {0: [], 1: [], 2: []}
    for actor in actors:
        occupied_lanes = actor.get("occupied_lane_indices")
        if occupied_lanes is None:
            occupied_lanes = (actor["lane_index"],)

        # A lane-changing vehicle can overlap two lane polygons.  Supplying it
        # to both lane arrays is conservative for the legacy MPC, whose traffic
        # rows otherwise carry no footprint-overlap or lateral-velocity field.
        # Preserve first-seen order while preventing repeated lane labels from
        # duplicating the same physical actor within a lane.
        lanes = list(dict.fromkeys(int(lane) for lane in occupied_lanes))
        if not lanes:
            lanes = [int(actor["lane_index"])]
        actor_row = _actor_row(actor)
        for lane in lanes:
            if lane in rows:
                rows[lane].append(actor_row)
    return LaneTraffic.from_arrays(rows[0], rows[1], rows[2])


def _drift_vehicle(actor: Mapping[str, Any]) -> dict[str, Any]:
    heading = float(actor["local_yaw_rad"])
    local_vx, local_vy = _local_velocity_components(actor)
    vehicle = create_drift_vehicle(
        vid=int(actor["actor_id"]),
        x=float(actor["local_x_m"]),
        y=float(actor["local_y_m"]),
        vx=local_vx,
        vy=local_vy,
        vclass="truck" if actor.get("role") == "occluder" else "car",
    )
    vehicle["heading"] = heading
    vehicle["a"] = _road_longitudinal_acceleration(actor)
    vehicle["length"] = float(actor.get("length_m", 5.0))
    vehicle["width"] = float(actor.get("width_m", 2.0))
    return vehicle


def _ego_drift_vehicle(observation: Mapping[str, Any]) -> dict[str, Any]:
    ego = observation["ego"]
    heading = float(ego["local_yaw_rad"])
    local_vx, local_vy = _local_velocity_components(ego)
    length_m, width_m = _ego_geometry(observation)
    vehicle = create_drift_vehicle(
        vid=0,
        x=float(ego["local_x_m"]),
        y=float(ego["local_y_m"]),
        vx=local_vx,
        vy=local_vy,
        vclass="car",
    )
    vehicle["heading"] = heading
    vehicle["length"] = length_m
    vehicle["width"] = width_m
    return vehicle


def _configure_arm_ego_geometry(arm: Any, *, length_m: float, width_m: float) -> None:
    """Apply one CARLA ego footprint to every geometry-aware arm component."""
    mpc = getattr(arm.controller, "mpc", arm.controller)
    mpc.vehicle_length = float(length_m)
    mpc.vehicle_width = float(width_m)

    # IDEAM's gap utility duplicates the submitted MPC footprint defaults.
    # Keep its first-hop screens consistent with the physical CARLA vehicle.
    arm.utils.vehicle_width = float(width_m)
    arm.utils.l = float(length_m)
    arm.utils.l_diag = math.hypot(float(length_m), float(width_m))


def _safe_float(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_safe_float(item) for item in value.tolist()]
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, Mapping):
        return {str(key): _safe_float(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_float(item) for item in value]
    return value


class ExternalPhysicsPlanner:
    """Stateful planner for the four fidelity-defensible CARLA arms.

    DREAM, ADA, and APF share the submitted decision/MPC/CBF coupling stack.
    They differ only in the risk-source model used to update the field.  IDEAM
    is the no-field reference arm.  OA-CMPC is intentionally absent because
    the repository adapter does not reproduce its published dual-branch
    contingency optimizer.
    """

    def __init__(self, controller_name: str, route_request: Mapping[str, Any]) -> None:
        self.controller_name = str(controller_name).upper()
        if self.controller_name not in SUPPORTED_CONTROLLERS:
            raise ValueError(
                "controller must be one of {}".format(
                    ", ".join(SUPPORTED_CONTROLLERS)
                )
            )
        self.road = _straight_road()
        self.route_request = ManeuverRequest(
            target_lane=int(route_request.get("target_lane", 1)),
            start_time_s=float(route_request.get("start_time_s", 2.5)),
            end_time_s=float(route_request.get("end_time_s", 7.5)),
            label="carla_overtaking_route_preference",
        )
        self.arm = None
        self.last_source_time_s: float | None = None
        self.last_field_time_s: float | None = None
        self.field_warmed = False
        self.ego_geometry_m: tuple[float, float] | None = None

    def _ensure_arm(self, observation: Mapping[str, Any]) -> None:
        length_m, width_m = _ego_geometry(observation)
        if self.arm is not None:
            if self.ego_geometry_m is None or not (
                math.isclose(length_m, self.ego_geometry_m[0], rel_tol=0.0, abs_tol=1.0e-6)
                and math.isclose(width_m, self.ego_geometry_m[1], rel_tol=0.0, abs_tol=1.0e-6)
            ):
                raise ValueError("ego geometry changed after planner initialisation")
            return
        initial = _planner_state(observation, None)
        if self.controller_name != "IDEAM":
            self.arm = create_prideam_episode_arm(
                self.road,
                initial,
                coupling=CouplingFlags(True, True, True),
                risk_weights=RISK_WEIGHTS,
                mpc_overrides={
                    "vehicle_length": length_m,
                    "vehicle_width": width_m,
                },
                maneuver_request=self.route_request,
                name="carla_{}".format(self.controller_name.lower()),
            )
            self.arm.controller.drift.reset()
        else:
            self.arm = create_ideam_episode_arm(
                self.road,
                initial,
                maneuver_request=self.route_request,
                name="carla_ideam",
            )
        _configure_arm_ego_geometry(self.arm, length_m=length_m, width_m=width_m)
        self.ego_geometry_m = (length_m, width_m)

    def _source_function(self) -> Any:
        if self.controller_name == "ADA":
            return compute_Q_ADA
        if self.controller_name == "APF":
            return compute_Q_APF
        return None

    def _update_field(self, observation: Mapping[str, Any]) -> float:
        if self.controller_name == "IDEAM":
            return 0.0
        assert self.arm is not None
        started = time.perf_counter()
        source_time = float(observation["simulation_time_s"])
        vehicles = [_drift_vehicle(actor) for actor in observation["visible_actors"]]
        ego = _ego_drift_vehicle(observation)
        source_function = self._source_function()

        def advance(dt: float) -> None:
            kwargs = {"dt": dt, "substeps": 3}
            if source_function is not None:
                kwargs["source_fn"] = source_function
            self.arm.controller.drift.step(vehicles, ego, **kwargs)

        if not self.field_warmed:
            for _ in range(5):
                advance(PLANNER_DT_S)
            self.field_warmed = True
            self.last_field_time_s = source_time
        else:
            previous_field_time = self.last_field_time_s
            if previous_field_time is None:
                raise RuntimeError("field_warmed is true without a previous field timestamp")
            elapsed = max(PLANNER_DT_S, source_time - float(previous_field_time))
            n_steps = max(1, int(math.ceil(elapsed / PLANNER_DT_S)))
            dt_step = min(PLANNER_DT_S, elapsed / n_steps)
            for _ in range(n_steps):
                advance(dt_step)
            self.last_field_time_s = source_time
        return time.perf_counter() - started

    def _synchronise_state(self, observation: Mapping[str, Any]) -> None:
        assert self.arm is not None
        source_time = float(observation["simulation_time_s"])
        previous = self.arm.state.copy()
        if self.last_source_time_s is not None:
            previous = _age_warm_start(previous, source_time - self.last_source_time_s)
        self.arm.state = _planner_state(observation, previous)
        self.arm.step_index = int(max(0, round(source_time / PLANNER_DT_S)))

    def _trajectory(self, result: Any, source_time_s: float) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
        state = result.state
        if state.last_X is None:
            return [], []
        ovx, ovy, owz, stations, lateral_errors, heading_errors = [
            np.asarray(values, dtype=float).reshape(-1) for values in state.last_X
        ]
        path = self.road.path(int(state.path_changed))
        count = min(len(ovx), len(stations), len(lateral_errors), len(heading_errors))
        states: list[dict[str, float]] = []
        for index in range(count):
            x_m, y_m = path.get_cartesian_coords(float(stations[index]), float(lateral_errors[index]))
            yaw = _wrap_angle(path.get_theta_r(float(stations[index])) + float(heading_errors[index]))
            states.append(
                {
                    "time_s": float(source_time_s + index * PLANNER_DT_S),
                    "local_x_m": float(x_m),
                    "local_y_m": float(y_m),
                    "local_yaw_rad": yaw,
                    "body_vx_mps": float(ovx[index]),
                    "body_vy_mps": float(ovy[index]) if index < len(ovy) else 0.0,
                    "yaw_rate_rps": float(owz[index]) if index < len(owz) else 0.0,
                    "station_m": float(stations[index]),
                    "lateral_error_m": float(lateral_errors[index]),
                    "heading_error_rad": float(heading_errors[index]),
                }
            )
        accelerations = np.asarray(state.oa, dtype=float).reshape(-1)
        steering = np.asarray(state.od, dtype=float).reshape(-1)
        controls = [
            {
                "time_s": float(source_time_s + index * PLANNER_DT_S),
                "acceleration_mps2": float(accelerations[index]),
                "steering_angle_rad": float(steering[index]),
            }
            for index in range(min(len(accelerations), len(steering)))
        ]
        return states, controls

    def _field_payload(self) -> Mapping[str, Any] | None:
        if self.controller_name == "IDEAM" or self.arm is None:
            return None
        field = np.asarray(self.arm.controller.drift.risk_field, dtype=float)
        stride_x = 3
        stride_y = 2
        sampled = field[::stride_y, ::stride_x]
        bounds = self.arm.controller.drift.grid_bounds
        return {
            "x_min": float(bounds["x_min"]),
            "x_max": float(bounds["x_max"]),
            "y_min": float(bounds["y_min"]),
            "y_max": float(bounds["y_max"]),
            "nx": int(sampled.shape[1]),
            "ny": int(sampled.shape[0]),
            "values": sampled.tolist(),
            "max": float(np.max(field)),
            "mean": float(np.mean(field)),
        }

    def plan(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        self._ensure_arm(observation)
        assert self.arm is not None
        started_ns = time.perf_counter_ns()
        field_time_s = self._update_field(observation)
        self._synchronise_state(observation)
        traffic = _lane_traffic(observation["visible_actors"])
        solver_stdout = io.StringIO()
        with contextlib.redirect_stdout(solver_stdout):
            result = self.arm.step(traffic)
        ended_ns = time.perf_counter_ns()
        source_time_s = float(observation["simulation_time_s"])
        self.last_source_time_s = source_time_s
        states, controls = self._trajectory(result, source_time_s)
        diagnostics = asdict(result.diagnostics)
        return _safe_float(
            {
                "run_id": observation["run_id"],
                "scenario_id": observation["scenario_id"],
                "controller": self.controller_name,
                "source_frame_id": int(observation["frame_id"]),
                "source_simulation_time_s": source_time_s,
                "visible_actor_ids": [
                    int(actor["actor_id"]) for actor in observation["visible_actors"]
                ],
                "visible_actor_roles": [
                    str(actor["role"]) for actor in observation["visible_actors"]
                ],
                "planning_start_ns": int(started_ns),
                "planning_end_ns": int(ended_ns),
                "planning_total_s": (ended_ns - started_ns) / 1e9,
                "field_time_s": float(field_time_s),
                "decision_time_s": diagnostics["decision_time_s"],
                "mpc_time_s": diagnostics["mpc_time_s"],
                "status": "fallback" if diagnostics["fallback_used"] else "ok",
                "fallback_used": diagnostics["fallback_used"],
                "solver_success": diagnostics["solver_success"],
                "fallback_reason": diagnostics["fallback_reason"],
                "target_lane": int(result.state.path_changed),
                "dt_s": PLANNER_DT_S,
                "validity_end_time_s": source_time_s + max(0, len(states) - 1) * PLANNER_DT_S,
                "states": states,
                "controls": controls,
                "diagnostics": diagnostics,
                "field": self._field_payload(),
                "ego_geometry_m": {
                    "length": self.ego_geometry_m[0],
                    "width": self.ego_geometry_m[1],
                },
                "traffic_adapter_provenance": TRAFFIC_ADAPTER_PROVENANCE,
                "stdout_excerpt": solver_stdout.getvalue()[-2000:],
            }
        )


def _error_response(message: Mapping[str, Any] | None, error: BaseException) -> dict[str, Any]:
    return make_message("error", {
        "run_id": None if message is None else message.get("run_id"),
        "scenario_id": None if message is None else message.get("scenario_id"),
        "error": str(error),
        "traceback": traceback.format_exc(),
    })


def serve(host: str, port: int, timeout_s: float) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, int(port)))
        server.listen(1)
        print(f"CARLA external planner listening on {host}:{port}", flush=True)
        connection, address = server.accept()
        print(f"CARLA bridge connected from {address[0]}:{address[1]}", flush=True)
        with connection:
            connection.settimeout(float(timeout_s))
            init_envelope = recv_message(connection, expected_type="hello")
            validate_message(init_envelope, expected_type="hello")
            init_message = init_envelope["payload"]
            planner = ExternalPhysicsPlanner(
                str(init_message["controller"]),
                init_message.get("route_request", {}),
            )
            send_message(
                connection,
                make_message("hello", {
                    "status": "ready",
                    "controller": planner.controller_name,
                }),
            )
            while True:
                message: Mapping[str, Any] | None = None
                try:
                    envelope = recv_message(connection)
                    validate_message(envelope)
                    message_type = envelope["type"]
                    message = envelope["payload"]
                    if message_type == "shutdown":
                        send_message(connection, make_message("shutdown", {"status": "ack"}))
                        return
                    if message_type != "observation":
                        raise ProtocolError(f"unexpected message type {message_type!r}")
                    send_message(connection, make_message("plan", planner.plan(message)))
                except ConnectionClosed:
                    return
                except Exception as error:  # keep the bridge alive for auditable failure handling
                    send_message(connection, _error_response(message, error))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    serve(args.host, args.port, args.timeout_s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
