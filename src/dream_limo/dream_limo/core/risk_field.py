"""Scaled, headless DRIFT PDE implementation.

This is the single importable LIMO adaptation of upstream ``config.py``,
``pde_solver.py`` and ``Integration/drift_interface.py``.  It retains the
vehicle/occlusion sources, traffic advection, variable diffusion, decay,
telegrapher fallback and risk queries while making every dimensional constant
explicit and injectable.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, cos, hypot, inf, sin
from time import perf_counter
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

from dream_limo.limo_scale import DeploymentConfig, IntegrationPreset

from .types import EgoState, Vehicle
from .route import anchored_lane_change_y


Array = np.ndarray


@dataclass(frozen=True)
class PDEDigest:
    compute_seconds: float
    requested_dt: float
    substeps: int
    stable_substep_dt: float
    raw_minimum: float
    raw_maximum: float
    field_minimum: float
    field_maximum: float
    field_mean: float
    maximum_diffusion: float
    maximum_flow_speed: float


@dataclass(frozen=True)
class SourceBreakdown:
    total: Array
    vehicle: Array
    occlusion: Array
    merge: Array
    occlusion_mask: Array


class NumericalStabilityError(RuntimeError):
    """Raised rather than silently clipping an unstable PDE update."""


class DREAMRiskField:
    """World-fixed DRIFT field for the surveyed LIMO arena."""

    def __init__(self, config: DeploymentConfig):
        self.config = config
        grid = config.grid
        self.x = np.linspace(grid.x_min, grid.x_max, grid.nx, dtype=np.float64)
        self.y = np.linspace(grid.y_min, grid.y_max, grid.ny, dtype=np.float64)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.dx = float(self.x[1] - self.x[0])
        self.dy = float(self.y[1] - self.y[0])
        self.R = np.zeros_like(self.X)
        self.R_t = np.zeros_like(self.X)
        self.road_mask = self._make_road_mask()
        self.last_sources = SourceBreakdown(
            *(np.zeros_like(self.X) for _ in range(4)),
            np.zeros_like(self.X, dtype=bool),
        )
        self.last_diffusion = np.full_like(self.X, config.pde.d0)
        self.last_vx = np.zeros_like(self.X)
        self.last_vy = np.zeros_like(self.X)
        self.last_digest: Optional[PDEDigest] = None
        self.elapsed_model_time = 0.0

    @property
    def shape(self) -> Tuple[int, int]:
        return self.R.shape

    @property
    def ready(self) -> bool:
        return self.elapsed_model_time + 1.0e-9 >= self.config.pde.warmup_duration

    def reset(self) -> None:
        self.R.fill(0.0)
        self.R_t.fill(0.0)
        self.elapsed_model_time = 0.0
        self.last_digest = None

    def _make_road_mask(self) -> Array:
        grid = self.config.grid
        taper = max(grid.road_taper, 1.0e-9)
        lower = np.clip((self.Y - grid.road_y_min) / taper, 0.0, 1.0)
        upper = np.clip((grid.road_y_max - self.Y) / taper, 0.0, 1.0)
        return np.minimum(lower, upper)

    def _validate_mask(self, mask: Array) -> Array:
        result = np.asarray(mask, dtype=np.float64)
        if result.shape != self.shape:
            raise ValueError(f"shadow mask shape {result.shape} does not match {self.shape}")
        if not np.all(np.isfinite(result)):
            raise ValueError("shadow mask must be finite")
        return np.clip(result, 0.0, 1.0) * (self.road_mask > 0.0)

    def truck_heading_shadow(self, vehicles: Sequence[Vehicle], ego: EgoState) -> Array:
        """Upstream truck-heading cone retained only as an explicit A/B fallback."""
        pde = self.config.pde
        mask = np.zeros_like(self.X, dtype=bool)
        for vehicle in vehicles:
            if vehicle.vehicle_class != "truck":
                continue
            if hypot(vehicle.x - ego.x, vehicle.y - ego.y) < 0.1:
                continue
            dx = self.X - vehicle.x
            dy = self.Y - vehicle.y
            ch, sh = cos(vehicle.heading), sin(vehicle.heading)
            along = ch * dx + sh * dy
            lateral = np.abs(-sh * dx + ch * dy)
            distance = np.hypot(dx, dy)
            width = 0.5 * vehicle.width + 0.3 * distance
            mask |= (
                (along > 0.0)
                & (lateral < width)
                & (distance < pde.occlusion_range)
            )
        return mask.astype(np.float64) * (self.road_mask > 0.0)

    def compute_vehicle_source(self, vehicles: Sequence[Vehicle], ego: EgoState) -> Array:
        pde = self.config.pde
        source = np.zeros_like(self.X)
        ego_vx = ego.speed * cos(ego.yaw)
        ego_vy = ego.speed * sin(ego.yaw)
        for vehicle in vehicles:
            rel_vx = vehicle.vx - ego_vx
            rel_vy = vehicle.vy - ego_vy
            rel_speed = hypot(rel_vx, rel_vy)
            to_x = vehicle.x - ego.x
            to_y = vehicle.y - ego.y
            distance = hypot(to_x, to_y)
            forward = cos(ego.yaw) * to_x + sin(ego.yaw) * to_y
            is_ahead = forward > 0.0

            class_weight = 2.5 if vehicle.vehicle_class == "truck" else 2.0
            distance_weight = np.exp(-distance / pde.vehicle_distance_decay)
            relative_weight = 1.5 + rel_speed / pde.relative_speed_scale
            brake_weight = 1.0
            sigma_boost = 1.0
            if is_ahead and vehicle.acceleration < pde.braking_accel_threshold:
                # Compare braking severity in scaled acceleration units.
                severity = abs(vehicle.acceleration) / max(
                    abs(pde.braking_accel_threshold), 1.0e-9
                )
                if severity > 6.5:
                    brake_weight, sigma_boost = 6.0, 2.5
                elif severity > 3.3:
                    brake_weight, sigma_boost = 4.0, 1.8
                else:
                    brake_weight, sigma_boost = 2.5, 1.3
            weight = (
                pde.source_scale
                * class_weight
                * distance_weight
                * relative_weight
                * brake_weight
            )

            dx = self.X - vehicle.x
            dy = self.Y - vehicle.y
            ch, sh = cos(vehicle.heading), sin(vehicle.heading)
            longitudinal = ch * dx + sh * dy
            lateral = -sh * dx + ch * dy
            sigma_parallel = pde.sigma_x * (
                1.0 + 0.05 * abs(rel_vx) / max(self.config.scale.speed(1.0), 1.0e-9)
            ) * sigma_boost
            kernel = np.exp(
                -0.5
                * (
                    longitudinal**2 / sigma_parallel**2
                    + lateral**2 / pde.sigma_y**2
                )
            )
            source += weight * kernel

            closing_speed = -(
                rel_vx * cos(ego.yaw) + rel_vy * sin(ego.yaw)
            )
            if is_ahead and closing_speed > pde.closing_speed_threshold:
                ttc = max(0.1, distance / max(closing_speed, 1.0e-6))
                approach_weight = (
                    pde.source_scale
                    * class_weight
                    * 3.0
                    * np.exp(-ttc / pde.approach_ttc_scale)
                )
                behind = -(ch * dx + sh * dy)
                cross = np.abs(-sh * dx + ch * dy)
                corridor = (
                    (behind > 0.0)
                    & (behind < max(0.1, 1.5 * distance))
                    & (cross < pde.approach_corridor_half_width)
                )
                longitudinal_scale = max(distance / 2.0, self.dx)
                approach_kernel = np.exp(
                    -0.5
                    * (
                        behind**2 / longitudinal_scale**2
                        + cross**2 / max(pde.sigma_y**2, 1.0e-9)
                    )
                )
                source += approach_weight * approach_kernel * corridor
        return source

    def compute_occlusion_source(
        self,
        shadow_mask: Array,
        occluders: Sequence[Vehicle],
    ) -> Array:
        pde = self.config.pde
        mask = self._validate_mask(shadow_mask)
        if not np.any(mask):
            return np.zeros_like(self.X)
        truck_like = [v for v in occluders if v.vehicle_class == "truck"]
        if not truck_like:
            return pde.occlusion_source_amplitude * mask
        distance = np.full_like(self.X, np.inf)
        for vehicle in truck_like:
            distance = np.minimum(distance, np.hypot(self.X - vehicle.x, self.Y - vehicle.y))
        emergence = np.exp(-distance / pde.occlusion_decay)
        return pde.occlusion_source_amplitude * emergence * mask

    def compute_merge_source(self, vehicles: Sequence[Vehicle]) -> Array:
        """Optional topology prior; intentionally zero for the tabletop v1."""
        if not self.config.pde.merge_source_enabled:
            return np.zeros_like(self.X)
        # A small generic prior centered at the physical middle-lane conflict.
        middle = self.config.arena.lane_centers[1]
        longitudinal = np.exp(-0.5 * ((self.X - 3.5) / 0.8) ** 2)
        lateral = np.exp(-0.5 * ((self.Y - middle) / 0.25) ** 2)
        density = np.zeros_like(self.X)
        for vehicle in vehicles:
            density += np.exp(
                -0.5
                * (
                    ((self.X - vehicle.x) / 0.8) ** 2
                    + ((self.Y - vehicle.y) / 0.25) ** 2
                )
            )
        return 0.5 * self.config.pde.source_scale * longitudinal * lateral * np.clip(
            density, 0.0, 1.0
        )

    def compute_sources(
        self,
        vehicles: Sequence[Vehicle],
        ego: EgoState,
        shadow_mask: Array,
    ) -> SourceBreakdown:
        q_vehicle = self.compute_vehicle_source(vehicles, ego)
        q_occlusion = self.compute_occlusion_source(shadow_mask, vehicles)
        q_merge = self.compute_merge_source(vehicles)
        return SourceBreakdown(
            total=q_vehicle + q_occlusion + q_merge,
            vehicle=q_vehicle,
            occlusion=q_occlusion,
            merge=q_merge,
            occlusion_mask=np.asarray(shadow_mask, dtype=np.float64) > 0.5,
        )

    def compute_velocity_field(self, vehicles: Sequence[Vehicle]) -> Tuple[Array, Array]:
        pde = self.config.pde
        vx_sum = np.zeros_like(self.X)
        vy_sum = np.zeros_like(self.X)
        weights = np.full_like(self.X, 1.0e-9)
        for vehicle in vehicles:
            dx = self.X - vehicle.x
            dy = self.Y - vehicle.y
            ch, sh = cos(vehicle.heading), sin(vehicle.heading)
            longitudinal = ch * dx + sh * dy
            lateral = -sh * dx + ch * dy
            kernel = np.exp(
                -0.5
                * (
                    (longitudinal / pde.velocity_kernel_longitudinal) ** 2
                    + (lateral / pde.velocity_kernel_lateral) ** 2
                )
            )
            vx_sum += kernel * vehicle.vx
            vy_sum += kernel * vehicle.vy
            weights += kernel
        return vx_sum / weights, vy_sum / weights

    def compute_diffusion(self, shadow_mask: Array, vehicles: Sequence[Vehicle]) -> Array:
        pde = self.config.pde
        mask = self._validate_mask(shadow_mask)
        diffusion = pde.d0 + pde.d_occ * mask
        for vehicle in vehicles:
            if vehicle.acceleration >= pde.braking_accel_threshold:
                continue
            distance_sq = (self.X - vehicle.x) ** 2 + (self.Y - vehicle.y) ** 2
            diffusion += pde.d_brake_peak * np.exp(
                -distance_sq / max(pde.braking_diffusion_radius**2, 1.0e-9)
            )
        # Smooth coefficient discontinuities as in upstream DREAM.
        return gaussian_filter(diffusion, sigma=1.0, mode="nearest")

    def stable_substep_dt(self, diffusion: Array, vx: Array, vy: Array) -> float:
        pde = self.config.pde
        d_max = float(np.max(diffusion))
        diffusion_limit = inf
        if d_max > 0.0:
            diffusion_limit = 1.0 / (
                2.0 * d_max * (1.0 / self.dx**2 + 1.0 / self.dy**2)
            )
        advective_rate = float(np.max(np.abs(vx)) / self.dx + np.max(np.abs(vy)) / self.dy)
        advection_limit = inf if advective_rate <= 1.0e-12 else 1.0 / advective_rate
        decay_max = (
            pde.lambda_decay
            + float(np.max(np.hypot(vx, vy))) / pde.l_decay
            + pde.lambda_sponge
        )
        decay_limit = inf if decay_max <= 1.0e-12 else 1.0 / decay_max
        telegraph_limit = inf if pde.tau <= 0.0 else 0.5 * pde.tau
        return pde.cfl_safety * min(
            diffusion_limit, advection_limit, decay_limit, telegraph_limit
        )

    def required_substeps(self, dt: float, diffusion: Array, vx: Array, vy: Array) -> int:
        stable_dt = self.stable_substep_dt(diffusion, vx, vy)
        if not np.isfinite(stable_dt) or stable_dt <= 0.0:
            raise NumericalStabilityError("could not derive a finite positive CFL limit")
        return max(self.config.pde.minimum_substeps, int(ceil(dt / stable_dt)))

    def _rhs(
        self,
        field: Array,
        source: Array,
        diffusion: Array,
        vx: Array,
        vy: Array,
        decay: Array,
    ) -> Array:
        ny, nx = field.shape

        # Conservative variable diffusion: div(D grad(R)), with zero flux at
        # the outer numerical boundary. The road mask supplies the physical
        # Dirichlet boundary after each substep.
        diff_flux_x = np.zeros((ny, nx + 1), dtype=np.float64)
        d_face_x = 0.5 * (diffusion[:, :-1] + diffusion[:, 1:])
        diff_flux_x[:, 1:nx] = d_face_x * (field[:, 1:] - field[:, :-1]) / self.dx
        diff_flux_y = np.zeros((ny + 1, nx), dtype=np.float64)
        d_face_y = 0.5 * (diffusion[:-1, :] + diffusion[1:, :])
        diff_flux_y[1:ny, :] = d_face_y * (field[1:, :] - field[:-1, :]) / self.dy
        diffusion_term = (
            (diff_flux_x[:, 1:] - diff_flux_x[:, :-1]) / self.dx
            + (diff_flux_y[1:, :] - diff_flux_y[:-1, :]) / self.dy
        )

        # Conservative first-order upwind advection.
        adv_flux_x = np.zeros((ny, nx + 1), dtype=np.float64)
        vx_face = 0.5 * (vx[:, :-1] + vx[:, 1:])
        upwind_x = np.where(vx_face >= 0.0, field[:, :-1], field[:, 1:])
        adv_flux_x[:, 1:nx] = vx_face * upwind_x
        adv_flux_y = np.zeros((ny + 1, nx), dtype=np.float64)
        vy_face = 0.5 * (vy[:-1, :] + vy[1:, :])
        upwind_y = np.where(vy_face >= 0.0, field[:-1, :], field[1:, :])
        adv_flux_y[1:ny, :] = vy_face * upwind_y
        advection_divergence = (
            (adv_flux_x[:, 1:] - adv_flux_x[:, :-1]) / self.dx
            + (adv_flux_y[1:, :] - adv_flux_y[:-1, :]) / self.dy
        )
        return source - decay * field + diffusion_term - advection_divergence

    def step(
        self,
        vehicles: Sequence[Vehicle],
        ego: EgoState,
        shadow_mask: Optional[Array] = None,
        *,
        dt: Optional[float] = None,
        substeps: Optional[int] = None,
    ) -> Array:
        """Advance DRIFT using only vehicles that are visible to the planner."""
        started = perf_counter()
        requested_dt = self.config.pde.control_dt if dt is None else float(dt)
        if requested_dt <= 0.0:
            raise ValueError("dt must be positive")
        if shadow_mask is None:
            if self.config.arena.lidar_shadow_mode != "truck_heading_cone":
                raise ValueError("the deployed lidar_polygon mode requires an explicit shadow mask")
            shadow_mask = self.truck_heading_shadow(vehicles, ego)
        shadow = self._validate_mask(shadow_mask)
        sources = self.compute_sources(vehicles, ego, shadow)
        vx, vy = self.compute_velocity_field(vehicles)
        diffusion = self.compute_diffusion(shadow, vehicles)
        required = self.required_substeps(requested_dt, diffusion, vx, vy)
        if substeps is None:
            count = required
        else:
            count = int(substeps)
            if count < required:
                raise NumericalStabilityError(
                    f"substeps={count} is below CFL requirement {required}"
                )
        sub_dt = requested_dt / count
        speed = np.hypot(vx, vy)
        decay = self.config.pde.lambda_decay + speed / self.config.pde.l_decay
        sponge_start = self.config.grid.x_max - self.config.pde.sponge_length
        sponge = np.clip(
            (self.X - sponge_start) / max(self.config.pde.sponge_length, self.dx),
            0.0,
            1.0,
        )
        decay = decay + self.config.pde.lambda_sponge * sponge**2

        raw_minimum, raw_maximum = inf, -inf
        for _ in range(count):
            rhs = self._rhs(self.R, sources.total, diffusion, vx, vy, decay)
            if self.config.pde.tau > 0.0:
                raw_rt = self.R_t + sub_dt / self.config.pde.tau * (rhs - self.R_t)
                raw = self.R + sub_dt * raw_rt
            else:
                raw_rt = np.zeros_like(self.R_t)
                raw = self.R + sub_dt * rhs
            if not np.all(np.isfinite(raw)):
                raise NumericalStabilityError("non-finite risk-field update")
            raw_minimum = min(raw_minimum, float(np.min(raw)))
            raw_maximum = max(raw_maximum, float(np.max(raw)))
            if max(abs(raw_minimum), abs(raw_maximum)) > self.config.pde.instability_ceiling:
                raise NumericalStabilityError("risk update exceeded instability ceiling")
            # Small negative values can occur at a sharp Dirichlet edge. Record
            # them before enforcing the physical non-negativity constraint.
            self.R = np.clip(raw, 0.0, self.config.pde.risk_ceiling) * self.road_mask
            self.R_t = raw_rt * self.road_mask

        self.last_sources = sources
        self.last_diffusion = diffusion
        self.last_vx, self.last_vy = vx, vy
        self.elapsed_model_time += requested_dt
        self.last_digest = PDEDigest(
            compute_seconds=perf_counter() - started,
            requested_dt=requested_dt,
            substeps=count,
            stable_substep_dt=self.stable_substep_dt(diffusion, vx, vy),
            raw_minimum=raw_minimum,
            raw_maximum=raw_maximum,
            field_minimum=float(np.min(self.R)),
            field_maximum=float(np.max(self.R)),
            field_mean=float(np.mean(self.R)),
            maximum_diffusion=float(np.max(diffusion)),
            maximum_flow_speed=float(np.max(speed)),
        )
        return self.R.copy()

    def warmup(
        self,
        vehicles: Sequence[Vehicle],
        ego: EgoState,
        shadow_mask: Array,
        *,
        duration: Optional[float] = None,
    ) -> Array:
        target = self.config.pde.warmup_duration if duration is None else float(duration)
        if target < 0.0:
            raise ValueError("warmup duration cannot be negative")
        while self.elapsed_model_time + 1.0e-9 < target:
            step_dt = min(
                self.config.pde.control_dt,
                target - self.elapsed_model_time,
            )
            self.step(vehicles, ego, shadow_mask, dt=step_dt)
        return self.R.copy()

    def risk_at(self, x: float, y: float) -> float:
        """Bilinear risk query; points outside the grid have zero deployed risk."""
        if x < self.x[0] or x > self.x[-1] or y < self.y[0] or y > self.y[-1]:
            return 0.0
        ix = min(len(self.x) - 2, max(0, int(np.searchsorted(self.x, x) - 1)))
        iy = min(len(self.y) - 2, max(0, int(np.searchsorted(self.y, y) - 1)))
        tx = (x - self.x[ix]) / (self.x[ix + 1] - self.x[ix])
        ty = (y - self.y[iy]) / (self.y[iy + 1] - self.y[iy])
        return float(
            (1.0 - tx) * (1.0 - ty) * self.R[iy, ix]
            + tx * (1.0 - ty) * self.R[iy, ix + 1]
            + (1.0 - tx) * ty * self.R[iy + 1, ix]
            + tx * ty * self.R[iy + 1, ix + 1]
        )

    def risk_gradient_at(self, x: float, y: float) -> Tuple[float, float]:
        eps_x, eps_y = self.dx, self.dy
        gx = (self.risk_at(x + eps_x, y) - self.risk_at(x - eps_x, y)) / (2.0 * eps_x)
        gy = (self.risk_at(x, y + eps_y) - self.risk_at(x, y - eps_y)) / (2.0 * eps_y)
        return gx, gy

    def lane_change_risk(
        self,
        ego: EgoState,
        target_lane: int,
        *,
        lookahead: Optional[float] = None,
        samples: Optional[int] = None,
    ) -> Tuple[float, float, float, Array]:
        arena = self.config.arena
        if not 0 <= target_lane < len(arena.lane_centers):
            raise ValueError("target lane index is out of range")
        distance = arena.veto_lookahead if lookahead is None else float(lookahead)
        count = arena.veto_samples if samples is None else int(samples)
        if distance <= 0.0 or count < 2:
            raise ValueError("lane-change sampling requires positive distance and >=2 points")
        target_y = arena.lane_centers[target_lane]
        x_values = np.linspace(ego.x, ego.x + distance, count)
        if target_lane == arena.target_lane:
            y_values = anchored_lane_change_y(
                x_values,
                source_y=arena.lane_centers[arena.ego_lane],
                target_y=target_y,
                start_x=arena.merge_path_x_min,
                end_x=arena.merge_path_x_max,
            )
        else:
            y_values = np.linspace(ego.y, target_y, count)
        risks = np.asarray(
            [self.risk_at(float(x), float(y)) for x, y in zip(x_values, y_values)],
            dtype=np.float64,
        )
        maximum = float(np.max(risks))
        mean = float(np.mean(risks))
        return 0.6 * maximum + 0.4 * mean, maximum, mean, risks

    def cbf_scale(self, x: float, y: float, preset: IntegrationPreset) -> float:
        if not preset.cbf_risk_expansion:
            return 1.0
        risk = self.risk_at(x, y)
        normalized = min(risk / max(preset.risk_normalization, 1.0e-9), 1.0)
        return float(
            np.clip(1.0 + preset.cbf_alpha * normalized, 1.0, preset.cbf_max_scale)
        )

    def headway_scale(self, x: float, y: float, preset: IntegrationPreset) -> float:
        if not preset.cbf_risk_expansion:
            return 1.0
        risk = self.risk_at(x, y)
        normalized = min(risk / max(preset.risk_normalization, 1.0e-9), 1.0)
        return float(np.clip(1.0 + preset.headway_beta * normalized, 1.0, 2.0))
