"""LIMO-scaled DREAM MPC with risk cost and risk-expanded CBF margins.

The upstream dynamic highway model is singular at zero speed and rebuilds a
large six-state program on every call.  This deployment uses the same
successive-linearization structure on a low-speed kinematic bicycle model.  It
preserves the designed DREAM risk term ``0.1 * weight * R * v^2`` and linearizes
ellipse barriers into separating tangent half-spaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from time import perf_counter
from typing import Optional, Sequence, Tuple

import casadi as ca
import cvxpy as cp
import numpy as np

from dream_limo.limo_scale import DeploymentConfig, IntegrationPreset

from .mission import stopping_speed_limit
from .path_tracking import build_path_reference
from .risk_field import DREAMRiskField
from .route import anchored_lane_change_y
from .types import ControlCommand, EgoState, Vehicle


Array = np.ndarray


@dataclass(frozen=True)
class MPCResult:
    command: ControlCommand
    states: Array
    controls: Array
    status: str
    solve_seconds: float
    objective: float
    maximum_slack: float
    risk_profile: Array
    used_fallback: bool


class KinematicBicycleModel:
    """CasADi-generated, standstill-safe discrete model and Jacobians."""

    def __init__(self, dt: float, wheelbase: float) -> None:
        state = ca.SX.sym("state", 4)  # x, y, v, yaw
        control = ca.SX.sym("control", 2)  # acceleration, center steer
        x, y, speed, yaw = (state[index] for index in range(4))
        acceleration, steering = control[0], control[1]
        next_state = ca.vertcat(
            x + dt * speed * ca.cos(yaw),
            y + dt * speed * ca.sin(yaw),
            speed + dt * acceleration,
            yaw + dt * speed / wheelbase * ca.tan(steering),
        )
        self._step = ca.Function("limo_bicycle_step", [state, control], [next_state])
        self._jacobian = ca.Function(
            "limo_bicycle_jacobian",
            [state, control],
            [ca.jacobian(next_state, state), ca.jacobian(next_state, control)],
        )

    def step(self, state: Array, control: Array) -> Array:
        return np.asarray(self._step(state, control), dtype=np.float64).reshape(4)

    def linearize(self, state: Array, control: Array) -> Tuple[Array, Array, Array]:
        A_raw, B_raw = self._jacobian(state, control)
        A = np.asarray(A_raw, dtype=np.float64)
        B = np.asarray(B_raw, dtype=np.float64)
        nonlinear = self.step(state, control)
        offset = nonlinear - A @ state - B @ control
        if not all(np.all(np.isfinite(value)) for value in (A, B, offset)):
            raise RuntimeError("non-finite kinematic model linearization")
        return A, B, offset


class RiskAwareMPC:
    def __init__(
        self, config: DeploymentConfig, *, enforce_map_bounds: bool = False
    ) -> None:
        self.deployment = config
        self.config = config.mpc
        self.model = KinematicBicycleModel(self.config.dt, self.config.wheelbase)
        self.last_states: Optional[Array] = None
        self.last_controls: Optional[Array] = None
        self.last_applied_control = np.zeros(2, dtype=np.float64)
        self.enforce_map_bounds = bool(enforce_map_bounds)

    def reset(self) -> None:
        self.last_states = None
        self.last_controls = None
        self.last_applied_control = np.zeros(2, dtype=np.float64)

    def _initial_state(self, ego: EgoState) -> Array:
        return np.asarray([ego.x, ego.y, ego.speed, ego.yaw], dtype=np.float64)

    def _reference(self, ego: EgoState, target_lane: int) -> Array:
        count = self.config.horizon + 1
        target_y = self.deployment.arena.lane_centers[target_lane]
        progress = np.linspace(0.0, 1.0, count)
        blend = progress * progress * (3.0 - 2.0 * progress)
        reference = np.zeros((4, count), dtype=np.float64)
        goal_x = self.deployment.arena.mission_goal_x
        reference[0, 0] = ego.x
        for index in range(count):
            reference[2, index] = stopping_speed_limit(
                goal_x - reference[0, index],
                cruise_speed=self.config.target_speed,
                braking_deceleration=self.config.mission_braking_deceleration,
            )
            if index + 1 < count:
                reference[0, index + 1] = min(
                    goal_x,
                    reference[0, index] + self.config.dt * reference[2, index],
                )
        if target_lane == self.deployment.arena.target_lane:
            arena = self.deployment.arena
            reference[1] = anchored_lane_change_y(
                reference[0],
                source_y=arena.lane_centers[arena.ego_lane],
                target_y=target_y,
                start_x=arena.merge_path_x_min,
                end_x=arena.merge_path_x_max,
            )
        else:
            reference[1] = ego.y + blend * (target_y - ego.y)
        dy = np.gradient(reference[1], self.config.dt)
        dx = np.gradient(reference[0], self.config.dt)
        reference[3] = np.arctan2(dy, np.maximum(dx, 1.0e-3))
        return reference

    def _warm_start(self, initial: Array, reference: Array) -> Tuple[Array, Array]:
        horizon = self.config.horizon
        if (
            self.last_states is not None
            and self.last_controls is not None
            and self.last_states.shape == (4, horizon + 1)
            and self.last_controls.shape == (2, horizon)
        ):
            states = np.column_stack((self.last_states[:, 1:], self.last_states[:, -1]))
            controls = np.column_stack((self.last_controls[:, 1:], self.last_controls[:, -1]))
            states[:, 0] = initial
            return states, controls
        states = reference.copy()
        states[:, 0] = initial
        controls = np.zeros((2, horizon), dtype=np.float64)
        for step in range(horizon):
            desired_acceleration = np.clip(
                (reference[2, step + 1] - states[2, step]) / self.config.dt,
                self.config.minimum_acceleration,
                self.config.maximum_acceleration,
            )
            controls[:, step] = (desired_acceleration, 0.0)
            states[:, step + 1] = self.model.step(states[:, step], controls[:, step])
        return states, controls

    def _reference_warm_start(
        self, initial: Array, reference: Array
    ) -> Tuple[Array, Array]:
        """Warm-start an arbitrary path, including its nominal curvature."""

        horizon = self.config.horizon
        if (
            self.last_states is not None
            and self.last_controls is not None
            and self.last_states.shape == (4, horizon + 1)
            and self.last_controls.shape == (2, horizon)
            and np.max(
                np.linalg.norm(
                    self.last_states[0:2, 1:] - reference[0:2, :-1], axis=0
                )
            )
            <= 0.75
        ):
            states = np.column_stack((self.last_states[:, 1:], self.last_states[:, -1]))
            controls = np.column_stack(
                (self.last_controls[:, 1:], self.last_controls[:, -1])
            )
            states[:, 0] = initial
            return states, controls

        states = reference.copy()
        states[:, 0] = initial
        controls = np.zeros((2, horizon), dtype=np.float64)
        for step in range(horizon):
            acceleration = np.clip(
                (reference[2, step + 1] - states[2, step]) / self.config.dt,
                self.config.minimum_acceleration,
                self.config.maximum_acceleration,
            )
            yaw_delta = float(reference[3, step + 1] - reference[3, step])
            nominal_speed = max(float(reference[2, step]), 1.0e-2)
            steering = np.arctan(
                self.config.wheelbase
                * yaw_delta
                / (self.config.dt * nominal_speed)
            )
            steering = np.clip(
                steering, -self.config.maximum_steer, self.config.maximum_steer
            )
            controls[:, step] = (acceleration, steering)
            states[:, step + 1] = self.model.step(
                states[:, step], controls[:, step]
            )
        return states, controls

    def _footprint_center_bounds(self) -> Tuple[float, float, float, float]:
        """Center bounds that keep the complete safety footprint on-road."""

        footprint_radius = hypot(
            0.5 * self.config.robot_length,
            0.5 * self.config.robot_width,
        ) + self.deployment.safety.collision_inflation_margin
        grid = self.deployment.grid
        quantization = 0.5 * grid.resolution
        center_x_min = grid.x_min + footprint_radius - quantization
        center_x_max = grid.x_max - footprint_radius + quantization
        center_y_min = max(grid.y_min, grid.road_y_min) + footprint_radius - quantization
        center_y_max = min(grid.y_max, grid.road_y_max) - footprint_radius + quantization
        if center_x_min >= center_x_max or center_y_min >= center_y_max:
            raise ValueError("collision footprint leaves no drivable MPC corridor")
        return center_x_min, center_x_max, center_y_min, center_y_max

    @staticmethod
    def _predicted_vehicle(vehicle: Vehicle, step: int, dt: float) -> Tuple[float, float]:
        time = step * dt
        return vehicle.x + vehicle.vx * time, vehicle.y + vehicle.vy * time

    @staticmethod
    def _ellipse_tangent(
        center: Tuple[float, float],
        axes: Tuple[float, float],
        reference: Tuple[float, float],
    ) -> Tuple[float, float, float]:
        """Closed-form ray intersection and outward tangent half-space."""
        cx, cy = center
        axis_x, axis_y = axes
        rx, ry = reference[0] - cx, reference[1] - cy
        if hypot(rx, ry) < 1.0e-9:
            rx, ry = -axis_x, 0.0
        denominator = np.sqrt((rx / axis_x) ** 2 + (ry / axis_y) ** 2)
        boundary_x = cx + rx / denominator
        boundary_y = cy + ry / denominator
        normal_x = (boundary_x - cx) / axis_x**2
        normal_y = (boundary_y - cy) / axis_y**2
        # normal . ([x,y] - center) >= 1 is the safe side containing reference.
        return float(normal_x), float(normal_y), float(1.0 + normal_x * cx + normal_y * cy)

    def _fallback(self, ego: EgoState, reason: str, started: float) -> MPCResult:
        acceleration = max(
            self.config.minimum_acceleration,
            -ego.speed / max(self.config.dt, 1.0e-9),
        )
        speed = max(0.0, ego.speed + acceleration * self.config.dt)
        command = ControlCommand(
            target_speed=speed,
            acceleration=acceleration,
            steering=0.0,
            stamp=ego.stamp,
            valid=False,
            reason=reason,
        )
        return MPCResult(
            command=command,
            states=np.empty((4, 0)),
            controls=np.empty((2, 0)),
            status=reason,
            solve_seconds=perf_counter() - started,
            objective=float("inf"),
            maximum_slack=float("inf"),
            risk_profile=np.empty(0),
            used_fallback=True,
        )

    def solve(
        self,
        ego: EgoState,
        target_lane: int,
        vehicles: Sequence[Vehicle],
        risk_field: DREAMRiskField,
        preset: IntegrationPreset,
    ) -> MPCResult:
        started = perf_counter()
        if not 0 <= target_lane < len(self.deployment.arena.lane_centers):
            raise ValueError("target lane is out of range")
        initial = self._initial_state(ego)
        reference = self._reference(ego, target_lane)
        linearization_states, linearization_controls = self._warm_start(initial, reference)
        horizon = self.config.horizon
        state = cp.Variable((4, horizon + 1), name="state")
        control = cp.Variable((2, horizon), name="control")
        obstacle_slack = (
            cp.Variable((len(vehicles), horizon + 1), nonneg=True, name="cbf_slack")
            if vehicles
            else None
        )

        constraints = [state[:, 0] == initial]
        constraints.extend(
            (
                state[2, :] >= self.config.minimum_speed,
                state[2, :] <= self.config.maximum_speed,
                control[0, :] >= self.config.minimum_acceleration,
                control[0, :] <= self.config.maximum_acceleration,
                control[1, :] >= -self.config.maximum_steer,
                control[1, :] <= self.config.maximum_steer,
            )
        )
        # Never plan beyond the declared mission endpoint. A small tolerance
        # matches the completion latch and avoids infeasibility from odometry
        # quantization immediately before the goal.
        constraints.append(
            state[0, 1:]
            <= self.deployment.arena.mission_goal_x
            + self.config.mission_position_tolerance
        )
        if self.enforce_map_bounds:
            # Physical deployment keeps the complete circular safety footprint
            # inside the same grid/road corridor as the final collision gate.
            # Deterministic legacy SIL leaves this off because its surveyed
            # truck ellipse intentionally overlaps that conservative corridor.
            footprint_radius = hypot(
                0.5 * self.config.robot_length,
                0.5 * self.config.robot_width,
            ) + self.deployment.safety.collision_inflation_margin
            grid = self.deployment.grid
            quantization = 0.5 * grid.resolution
            center_x_min = grid.x_min + footprint_radius - quantization
            center_x_max = grid.x_max - footprint_radius + quantization
            center_y_min = grid.road_y_min + footprint_radius - quantization
            center_y_max = grid.road_y_max - footprint_radius + quantization
            if center_x_min >= center_x_max or center_y_min >= center_y_max:
                raise ValueError("collision footprint leaves no drivable MPC corridor")
            constraints.extend(
                (
                    state[0, 1:] >= center_x_min,
                    state[0, 1:] <= center_x_max,
                    # Steering affects lateral position one model step later;
                    # allow two steps to recover a small measured deviation.
                    state[1, 2:] >= center_y_min,
                    state[1, 2:] <= center_y_max,
                )
            )
        steer_delta_limit = self.config.maximum_steer_rate * self.config.dt
        constraints.extend(
            (
                control[1, 0] - self.last_applied_control[1] <= steer_delta_limit,
                control[1, 0] - self.last_applied_control[1] >= -steer_delta_limit,
            )
        )
        if horizon > 1:
            constraints.extend(
                (
                    control[1, 1:] - control[1, :-1] <= steer_delta_limit,
                    control[1, 1:] - control[1, :-1] >= -steer_delta_limit,
                )
            )

        objective = 0.0
        risk_profile = np.asarray(
            [
                risk_field.risk_at(
                    float(linearization_states[0, index]),
                    float(linearization_states[1, index]),
                )
                for index in range(horizon + 1)
            ],
            dtype=np.float64,
        )

        for step in range(horizon):
            A, B, offset = self.model.linearize(
                linearization_states[:, step], linearization_controls[:, step]
            )
            constraints.append(
                state[:, step + 1]
                == A @ state[:, step] + B @ control[:, step] + offset
            )
            objective += self.config.position_weight * cp.square(
                state[1, step] - reference[1, step]
            )
            objective += 0.05 * self.config.position_weight * cp.square(
                state[0, step] - reference[0, step]
            )
            objective += self.config.heading_weight * cp.square(
                state[3, step] - reference[3, step]
            )
            objective += self.config.speed_weight * cp.square(
                state[2, step] - reference[2, step]
            )
            objective += self.config.control_weight_acceleration * cp.square(control[0, step])
            objective += self.config.control_weight_steer * cp.square(control[1, step])
            if step == 0:
                objective += self.config.delta_control_weight * cp.sum_squares(
                    control[:, step] - self.last_applied_control
                )
            else:
                objective += self.config.delta_control_weight * cp.sum_squares(
                    control[:, step] - control[:, step - 1]
                )
            if preset.mpc_risk_cost and risk_profile[step] > 0.05:
                objective += (
                    preset.risk_weight
                    * 0.1
                    * float(risk_profile[step])
                    * cp.square(state[2, step])
                )

        terminal = self.config.terminal_multiplier
        objective += terminal * self.config.position_weight * cp.square(
            state[1, horizon] - reference[1, horizon]
        )
        objective += 0.05 * terminal * self.config.position_weight * cp.square(
            state[0, horizon] - reference[0, horizon]
        )
        objective += terminal * self.config.heading_weight * cp.square(
            state[3, horizon] - reference[3, horizon]
        )
        objective += terminal * self.config.speed_weight * cp.square(
            state[2, horizon] - reference[2, horizon]
        )
        if preset.mpc_risk_cost and risk_profile[-1] > 0.05:
            objective += (
                preset.risk_weight
                * 0.1
                * float(risk_profile[-1])
                * cp.square(state[2, horizon])
            )

        for vehicle_index, vehicle in enumerate(vehicles):
            for step in range(horizon + 1):
                center = self._predicted_vehicle(vehicle, step, self.config.dt)
                ref_point = (
                    float(linearization_states[0, step]),
                    float(linearization_states[1, step]),
                )
                scale = risk_field.cbf_scale(ref_point[0], ref_point[1], preset)
                axes = (
                    scale * (self.config.base_cbf_longitudinal + 0.5 * vehicle.length),
                    scale * (self.config.base_cbf_lateral + 0.5 * vehicle.width),
                )
                normal_x, normal_y, right_hand = self._ellipse_tangent(center, axes, ref_point)
                slack = obstacle_slack[vehicle_index, step]
                constraints.append(
                    normal_x * state[0, step] + normal_y * state[1, step] + slack
                    >= right_hand
                )

                route_y = float(reference[1, step])
                is_leader = center[0] > ref_point[0] and abs(
                    center[1] - route_y
                ) < 0.55 * self.deployment.arena.lane_width
                if is_leader:
                    headway_scale = risk_field.headway_scale(ref_point[0], ref_point[1], preset)
                    constraints.append(
                        state[0, step]
                        + headway_scale * self.config.base_minimum_distance
                        + headway_scale * self.config.base_headway * state[2, step]
                        <= center[0] + slack
                    )
            objective += self.config.cbf_slack_weight * cp.sum_squares(
                obstacle_slack[vehicle_index, :]
            )

        problem = cp.Problem(cp.Minimize(objective), constraints)
        state.value = linearization_states
        control.value = linearization_controls
        try:
            problem.solve(
                solver=cp.OSQP,
                warm_start=True,
                verbose=False,
                eps_abs=1.0e-3,
                eps_rel=1.0e-3,
                max_iter=20_000,
                time_limit=self.config.solver_timeout,
                polishing=False,
            )
        except Exception as exc:  # CVXPY normalizes many backend failures as Exception.
            return self._fallback(ego, f"MPC exception: {exc}", started)
        if problem.status != cp.OPTIMAL:
            return self._fallback(ego, f"MPC status {problem.status}", started)
        states = np.asarray(state.value, dtype=np.float64)
        controls = np.asarray(control.value, dtype=np.float64)
        if not np.all(np.isfinite(states)) or not np.all(np.isfinite(controls)):
            return self._fallback(ego, "MPC returned non-finite values", started)
        acceleration = float(
            np.clip(
                controls[0, 0],
                self.config.minimum_acceleration,
                self.config.maximum_acceleration,
            )
        )
        steering = float(
            np.clip(controls[1, 0], -self.config.maximum_steer, self.config.maximum_steer)
        )
        speed = float(
            np.clip(
                ego.speed + self.config.dt * acceleration,
                self.config.minimum_speed,
                self.config.maximum_speed,
            )
        )
        self.last_states = states
        self.last_controls = controls
        self.last_applied_control = np.asarray([acceleration, steering])
        maximum_slack = (
            0.0
            if obstacle_slack is None or obstacle_slack.value is None
            else float(np.max(obstacle_slack.value))
        )
        command = ControlCommand(
            target_speed=speed,
            acceleration=acceleration,
            steering=steering,
            stamp=ego.stamp,
            valid=True,
            reason=str(problem.status),
        )
        return MPCResult(
            command=command,
            states=states,
            controls=controls,
            status=str(problem.status),
            solve_seconds=perf_counter() - started,
            objective=float(problem.value),
            maximum_slack=maximum_slack,
            risk_profile=risk_profile,
            used_fallback=False,
        )

    def solve_reference(
        self,
        ego: EgoState,
        path_points: Sequence[Sequence[float]] | Array,
        vehicles: Sequence[Vehicle],
        risk_field: DREAMRiskField,
        preset: IntegrationPreset,
        *,
        terminal_yaw: Optional[float] = None,
    ) -> MPCResult:
        """Track an arbitrary Cartesian path with DREAM risk and CBF behavior.

        Unlike :meth:`solve`, this entry point has no lane or one-way mission
        assumption.  It constructs an arc-length horizon from the current ego
        pose, tracks ``x`` and ``y`` isotropically, and keeps the complete robot
        footprint inside the risk grid's road corridor.
        """

        started = perf_counter()
        initial = self._initial_state(ego)
        reference = build_path_reference(
            path_points,
            ego_xy=initial[0:2],
            ego_yaw=float(initial[3]),
            horizon=self.config.horizon,
            dt=self.config.dt,
            cruise_speed=self.config.target_speed,
            braking_deceleration=self.config.mission_braking_deceleration,
            maximum_cross_track_error=(
                self.config.maximum_path_cross_track_error
            ),
            terminal_yaw=terminal_yaw,
        )
        linearization_states, linearization_controls = self._reference_warm_start(
            initial, reference
        )
        horizon = self.config.horizon
        state = cp.Variable((4, horizon + 1), name="reference_state")
        control = cp.Variable((2, horizon), name="reference_control")
        obstacle_slack = (
            cp.Variable(
                (len(vehicles), horizon + 1),
                nonneg=True,
                name="reference_cbf_slack",
            )
            if vehicles
            else None
        )

        constraints = [state[:, 0] == initial]
        constraints.extend(
            (
                state[2, :] >= self.config.minimum_speed,
                state[2, :] <= self.config.maximum_speed,
                control[0, :] >= self.config.minimum_acceleration,
                control[0, :] <= self.config.maximum_acceleration,
                control[1, :] >= -self.config.maximum_steer,
                control[1, :] <= self.config.maximum_steer,
            )
        )
        # Preserve the footprint-checked Nav2 geometry as a hard local tube.
        # Both normal and tangent displacement are bounded at every prediction
        # step; a later live-costmap swept-footprint check verifies the solved
        # trajectory continuously before it may be reported ready.
        corridor = self.config.path_corridor_half_width
        longitudinal_corridor = self.config.path_longitudinal_half_width
        for step in range(horizon + 1):
            yaw = float(reference[3, step])
            tangent = np.asarray([np.cos(yaw), np.sin(yaw)])
            normal = np.asarray([-np.sin(yaw), np.cos(yaw)])
            position_error = (
                state[0:2, step] - reference[0:2, step]
            )
            along_track = tangent @ position_error
            cross_track = normal @ position_error
            constraints.extend(
                (
                    along_track >= -longitudinal_corridor,
                    along_track <= longitudinal_corridor,
                    cross_track >= -corridor,
                    cross_track <= corridor,
                )
            )
        center_x_min, center_x_max, center_y_min, center_y_max = (
            self._footprint_center_bounds()
        )
        constraints.extend(
            (
                state[0, :] >= center_x_min,
                state[0, :] <= center_x_max,
                state[1, :] >= center_y_min,
                state[1, :] <= center_y_max,
            )
        )

        steer_delta_limit = self.config.maximum_steer_rate * self.config.dt
        constraints.extend(
            (
                control[1, 0] - self.last_applied_control[1] <= steer_delta_limit,
                control[1, 0] - self.last_applied_control[1] >= -steer_delta_limit,
            )
        )
        if horizon > 1:
            constraints.extend(
                (
                    control[1, 1:] - control[1, :-1] <= steer_delta_limit,
                    control[1, 1:] - control[1, :-1] >= -steer_delta_limit,
                )
            )

        objective = 0.0
        risk_profile = np.asarray(
            [
                risk_field.risk_at(
                    float(linearization_states[0, index]),
                    float(linearization_states[1, index]),
                )
                for index in range(horizon + 1)
            ],
            dtype=np.float64,
        )

        for step in range(horizon):
            A, B, offset = self.model.linearize(
                linearization_states[:, step], linearization_controls[:, step]
            )
            constraints.append(
                state[:, step + 1]
                == A @ state[:, step] + B @ control[:, step] + offset
            )
            objective += self.config.position_weight * cp.sum_squares(
                state[0:2, step] - reference[0:2, step]
            )
            objective += self.config.heading_weight * cp.square(
                state[3, step] - reference[3, step]
            )
            objective += self.config.speed_weight * cp.square(
                state[2, step] - reference[2, step]
            )
            objective += self.config.control_weight_acceleration * cp.square(
                control[0, step]
            )
            objective += self.config.control_weight_steer * cp.square(control[1, step])
            if step == 0:
                objective += self.config.delta_control_weight * cp.sum_squares(
                    control[:, step] - self.last_applied_control
                )
            else:
                objective += self.config.delta_control_weight * cp.sum_squares(
                    control[:, step] - control[:, step - 1]
                )
            if preset.mpc_risk_cost and risk_profile[step] > 0.05:
                objective += (
                    preset.risk_weight
                    * 0.1
                    * float(risk_profile[step])
                    * cp.square(state[2, step])
                )

        terminal = self.config.terminal_multiplier
        objective += terminal * self.config.position_weight * cp.sum_squares(
            state[0:2, horizon] - reference[0:2, horizon]
        )
        objective += terminal * self.config.heading_weight * cp.square(
            state[3, horizon] - reference[3, horizon]
        )
        objective += terminal * self.config.speed_weight * cp.square(
            state[2, horizon] - reference[2, horizon]
        )
        if preset.mpc_risk_cost and risk_profile[-1] > 0.05:
            objective += (
                preset.risk_weight
                * 0.1
                * float(risk_profile[-1])
                * cp.square(state[2, horizon])
            )

        for vehicle_index, vehicle in enumerate(vehicles):
            for step in range(horizon + 1):
                center = self._predicted_vehicle(vehicle, step, self.config.dt)
                ref_point = (
                    float(linearization_states[0, step]),
                    float(linearization_states[1, step]),
                )
                scale = risk_field.cbf_scale(ref_point[0], ref_point[1], preset)
                axes = (
                    scale
                    * (self.config.base_cbf_longitudinal + 0.5 * vehicle.length),
                    scale * (self.config.base_cbf_lateral + 0.5 * vehicle.width),
                )
                normal_x, normal_y, right_hand = self._ellipse_tangent(
                    center, axes, ref_point
                )
                slack = obstacle_slack[vehicle_index, step]
                constraints.append(
                    normal_x * state[0, step]
                    + normal_y * state[1, step]
                    + slack
                    >= right_hand
                )

                tangent = np.asarray(
                    [
                        np.cos(reference[3, step]),
                        np.sin(reference[3, step]),
                    ],
                    dtype=np.float64,
                )
                normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float64)
                relative = np.asarray(center, dtype=np.float64) - np.asarray(
                    ref_point, dtype=np.float64
                )
                longitudinal = float(tangent @ relative)
                lateral = float(normal @ relative)
                leader_half_width = (
                    0.5 * (self.config.robot_width + vehicle.width)
                    + self.config.path_corridor_half_width
                )
                if longitudinal > 0.0 and abs(lateral) < leader_half_width:
                    headway_scale = risk_field.headway_scale(
                        ref_point[0], ref_point[1], preset
                    )
                    constraints.append(
                        tangent[0] * state[0, step]
                        + tangent[1] * state[1, step]
                        + headway_scale * self.config.base_minimum_distance
                        + headway_scale * self.config.base_headway * state[2, step]
                        <= tangent[0] * center[0]
                        + tangent[1] * center[1]
                        + slack
                    )
            objective += self.config.cbf_slack_weight * cp.sum_squares(
                obstacle_slack[vehicle_index, :]
            )

        problem = cp.Problem(cp.Minimize(objective), constraints)
        state.value = linearization_states
        control.value = linearization_controls
        try:
            problem.solve(
                solver=cp.OSQP,
                warm_start=True,
                verbose=False,
                eps_abs=1.0e-3,
                eps_rel=1.0e-3,
                max_iter=20_000,
                time_limit=self.config.solver_timeout,
                polishing=False,
            )
        except Exception as exc:  # CVXPY normalizes many backend failures as Exception.
            return self._fallback(ego, f"MPC exception: {exc}", started)
        if problem.status != cp.OPTIMAL:
            return self._fallback(ego, f"MPC status {problem.status}", started)
        states = np.asarray(state.value, dtype=np.float64)
        controls = np.asarray(control.value, dtype=np.float64)
        if not np.all(np.isfinite(states)) or not np.all(np.isfinite(controls)):
            return self._fallback(ego, "MPC returned non-finite values", started)
        acceleration = float(
            np.clip(
                controls[0, 0],
                self.config.minimum_acceleration,
                self.config.maximum_acceleration,
            )
        )
        steering = float(
            np.clip(
                controls[1, 0],
                -self.config.maximum_steer,
                self.config.maximum_steer,
            )
        )
        speed = float(
            np.clip(
                ego.speed + self.config.dt * acceleration,
                self.config.minimum_speed,
                self.config.maximum_speed,
            )
        )
        self.last_states = states
        self.last_controls = controls
        self.last_applied_control = np.asarray([acceleration, steering])
        maximum_slack = (
            0.0
            if obstacle_slack is None or obstacle_slack.value is None
            else float(np.max(obstacle_slack.value))
        )
        command = ControlCommand(
            target_speed=speed,
            acceleration=acceleration,
            steering=steering,
            stamp=ego.stamp,
            valid=True,
            reason=str(problem.status),
        )
        return MPCResult(
            command=command,
            states=states,
            controls=controls,
            status=str(problem.status),
            solve_seconds=perf_counter() - started,
            objective=float(problem.value),
            maximum_slack=maximum_slack,
            risk_profile=risk_profile,
            used_fallback=False,
        )
