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
    def __init__(self, config: DeploymentConfig) -> None:
        self.deployment = config
        self.config = config.mpc
        self.model = KinematicBicycleModel(self.config.dt, self.config.wheelbase)
        self.last_states: Optional[Array] = None
        self.last_controls: Optional[Array] = None
        self.last_applied_control = np.zeros(2, dtype=np.float64)

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
        reference[0] = ego.x + self.config.target_speed * self.config.dt * np.arange(count)
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
        reference[3] = np.arctan2(dy, np.maximum(self.config.target_speed, 1.0e-3))
        reference[2] = self.config.target_speed
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
            constraints.append(state[:, step + 1] == A @ state[:, step] + B @ control[:, step] + offset)
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
        if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
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
