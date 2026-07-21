"""Single source of truth for highway-to-LIMO dimensional scaling.

The values are derived from the upstream DREAM model at commit
0d298cd6de11c268224173a4d75770e934fd0861.  All quantities that were hard-coded
in the upstream PDE implementation are named here so that the deployment never
mixes highway and tabletop units.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from math import hypot, isfinite, radians
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml


UPSTREAM_DREAM_COMMIT = "0d298cd6de11c268224173a4d75770e934fd0861"


@dataclass(frozen=True)
class SimilarityScale:
    """Dimensional map from an upstream value to a LIMO value."""

    alpha: float = 0.1
    beta: float = 2.0

    def __post_init__(self) -> None:
        if self.alpha <= 0.0 or self.beta <= 0.0:
            raise ValueError("alpha and beta must be positive")

    def length(self, value: float) -> float:
        return self.alpha * value

    def area(self, value: float) -> float:
        return self.alpha**2 * value

    def time(self, value: float) -> float:
        return self.beta * value

    def speed(self, value: float) -> float:
        return self.alpha / self.beta * value

    def acceleration(self, value: float) -> float:
        return self.alpha / self.beta**2 * value

    def diffusion(self, value: float) -> float:
        return self.alpha**2 / self.beta * value

    def decay(self, value: float) -> float:
        return value / self.beta

    def source(self, value: float) -> float:
        """Scale Q when the risk-field magnitude itself is invariant."""
        return value / self.beta


@dataclass(frozen=True)
class GridConfig:
    x_min: float = 0.0
    x_max: float = 6.0
    y_min: float = -1.0
    y_max: float = 1.0
    resolution: float = 0.025
    frame_id: str = "map"
    road_y_min: float = -0.70
    road_y_max: float = 0.70
    road_taper: float = 0.025

    def __post_init__(self) -> None:
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError("grid maximums must exceed minimums")
        if self.resolution <= 0.0:
            raise ValueError("grid resolution must be positive")
        if self.road_y_min >= self.road_y_max:
            raise ValueError("road_y_min must be below road_y_max")

    @property
    def nx(self) -> int:
        return int(round((self.x_max - self.x_min) / self.resolution)) + 1

    @property
    def ny(self) -> int:
        return int(round((self.y_max - self.y_min) / self.resolution)) + 1


@dataclass(frozen=True)
class PDEConfig:
    # Scaled upstream physical parameters.
    d0: float = 0.0015
    d_occ: float = 0.030
    d_brake_peak: float = 0.075
    lambda_decay: float = 0.075
    l_decay: float = 2.5
    sponge_length: float = 1.5
    lambda_sponge: float = 0.75
    tau: float = 0.0

    # Scaled source geometry and thresholds.
    source_scale: float = 0.5
    sigma_x: float = 0.8
    sigma_y: float = 0.25
    vehicle_distance_decay: float = 7.0
    relative_speed_scale: float = 0.175
    closing_speed_threshold: float = 0.05
    approach_corridor_half_width: float = 0.40
    approach_ttc_scale: float = 6.0
    occlusion_range: float = 6.0
    occlusion_decay: float = 3.0
    occlusion_source_amplitude: float = 2.5
    velocity_kernel_longitudinal: float = 2.0
    velocity_kernel_lateral: float = 0.30
    braking_accel_threshold: float = -0.0075
    braking_diffusion_radius: float = 2.0

    # Numerics. The solver derives the required count instead of assuming 3.
    control_dt: float = 0.2
    cfl_safety: float = 0.75
    minimum_substeps: int = 3
    warmup_duration: float = 5.0
    risk_ceiling: float = 10.0
    instability_ceiling: float = 1.0e4
    merge_source_enabled: bool = False

    def __post_init__(self) -> None:
        if self.d0 <= 0.0 or self.d_occ < 0.0:
            raise ValueError("diffusion coefficients must be non-negative")
        if self.lambda_decay < 0.0 or self.l_decay <= 0.0:
            raise ValueError("invalid decay configuration")
        if not 0.0 < self.cfl_safety < 1.0:
            raise ValueError("cfl_safety must lie in (0, 1)")
        if self.control_dt <= 0.0 or self.minimum_substeps < 1:
            raise ValueError("invalid time-step configuration")


@dataclass(frozen=True)
class ArenaConfig:
    lane_width: float = 0.45
    # Left, middle, right when viewed along +x. Positive y is left.
    lane_centers: Tuple[float, float, float] = (0.45, 0.0, -0.45)
    ego_lane: int = 0
    target_lane: int = 1
    merge_request_x: float = 0.35
    merge_path_x_min: float = 2.80
    merge_path_x_max: float = 3.80
    conflict_zone_x_min: float = 3.30
    conflict_zone_x_max: float = 5.30
    # Longitudinal center-position goal after the shared conflict zone.  Both
    # experiment arms use the same goal and mission-end speed profile.
    mission_goal_x: float = 5.55
    lidar_shadow_mode: str = "lidar_polygon"
    veto_lookahead: float = 3.0
    veto_samples: int = 10

    def __post_init__(self) -> None:
        if self.lane_width <= 0.0:
            raise ValueError("lane width must be positive")
        if len(self.lane_centers) != 3:
            raise ValueError("the v1 arena must define exactly three lanes")
        if self.lidar_shadow_mode not in {"lidar_polygon", "truck_heading_cone"}:
            raise ValueError("unsupported shadow mode")
        if not 0 <= self.target_lane < len(self.lane_centers):
            raise ValueError("target lane is outside the surveyed lane set")
        if self.conflict_zone_x_min >= self.conflict_zone_x_max:
            raise ValueError("conflict-zone limits are reversed")
        if self.merge_path_x_min >= self.merge_path_x_max:
            raise ValueError("merge-path limits are reversed")
        if not isfinite(self.mission_goal_x):
            raise ValueError("mission goal must be finite")
        if (
            self.target_lane != self.ego_lane
            and self.mission_goal_x
            <= max(self.merge_path_x_max, self.conflict_zone_x_max)
        ):
            raise ValueError(
                "lane-change mission goal must lie after the merge and conflict zone"
            )


@dataclass(frozen=True)
class MPCConfig:
    # Profiled on the onboard NUC: T=6 preserved the blocker-free A/B outcome
    # while reducing median solve time from about 104 ms to about 72 ms.
    horizon: int = 6
    dt: float = 0.2
    target_speed: float = 0.50
    maximum_speed: float = 0.60
    minimum_speed: float = 0.0
    maximum_acceleration: float = 0.35
    minimum_acceleration: float = -0.50
    # The installed driver forwards Twist.angular.z as the raw firmware
    # steering field. Its inner-wheel limit corresponds to about 23.4 degrees
    # at the bicycle center, so keep the MPC model below that saturation.
    maximum_steer: float = radians(23.0)
    maximum_steer_rate: float = radians(60.0)
    wheelbase: float = 0.20
    # Installed LIMO costmaps use the measured base_link footprint
    # x=+/-0.16 m, y=+/-0.11 m (0.32 x 0.22 m overall).
    robot_length: float = 0.32
    robot_width: float = 0.22
    # Must match global_costmap.footprint_padding in nav2_dream_planner.yaml.
    navigation_footprint_padding: float = 0.05
    # Must match global_costmap.inflation_layer.inflation_radius. The value
    # covers the padded circumscribed radius plus half a grid-cell diagonal.
    navigation_inflation_radius: float = 0.30
    base_cbf_longitudinal: float = 0.34
    base_cbf_lateral: float = 0.24
    base_headway: float = 0.60
    base_minimum_distance: float = 0.35
    cbf_slack_weight: float = 2.0e4
    # The tangent-ellipse constraint is dimensionless; tolerate at most a 5%
    # numerical/soft-constraint relaxation before the ROS boundary stops.
    maximum_allowed_cbf_slack: float = 0.05
    control_weight_acceleration: float = 0.5
    control_weight_steer: float = 2.0
    delta_control_weight: float = 1.0
    position_weight: float = 8.0
    heading_weight: float = 3.0
    speed_weight: float = 2.0
    terminal_multiplier: float = 4.0
    solver_timeout: float = 0.15
    # Free-space MPC is constrained to a narrow tube around the footprint-
    # checked Nav2 route.  This prevents the quadratic tracker from cutting a
    # collision-free corner merely to reduce control effort.
    path_corridor_half_width: float = 0.04
    # Timing mismatch (for example braking from an already higher measured
    # speed) may move a prediction ahead of its same-index reference, but the
    # deviation is still finite and the swept costmap check remains decisive.
    path_longitudinal_half_width: float = 0.30
    maximum_path_cross_track_error: float = 0.10
    # A kinematic square-root speed profile begins braking early enough to
    # reach the route goal at zero speed.  Completion then latches in the ROS
    # planner until that process is deliberately restarted.
    mission_braking_deceleration: float = 0.10
    mission_position_tolerance: float = 0.04
    mission_stop_speed_tolerance: float = 0.03

    def __post_init__(self) -> None:
        if self.horizon < 2 or self.dt <= 0.0:
            raise ValueError("invalid MPC horizon")
        if self.minimum_speed != 0.0:
            raise ValueError("physical DREAM deployment must permit a full stop")
        if self.maximum_speed > 0.60:
            raise ValueError("maximum_speed exceeds the Stage 3 hard cap")
        mission_values = (
            self.target_speed,
            self.mission_braking_deceleration,
            self.mission_position_tolerance,
            self.mission_stop_speed_tolerance,
            self.navigation_footprint_padding,
            self.navigation_inflation_radius,
        )
        if not all(isfinite(value) for value in mission_values):
            raise ValueError("mission MPC parameters must be finite")
        if not 0.0 <= self.mission_stop_speed_tolerance < self.target_speed <= self.maximum_speed:
            raise ValueError(
                "target_speed must exceed the stop tolerance and not exceed maximum_speed"
            )
        if not 0.0 < self.mission_braking_deceleration <= abs(
            self.minimum_acceleration
        ):
            raise ValueError("mission braking must be achievable by the MPC")
        if self.mission_position_tolerance < 0.0:
            raise ValueError("mission position tolerance must be non-negative")
        if self.wheelbase <= 0.0:
            raise ValueError("wheelbase must be positive")
        if self.navigation_footprint_padding < 0.0:
            raise ValueError("navigation footprint padding must be non-negative")
        padded_radius = hypot(
            0.5 * self.robot_length + self.navigation_footprint_padding,
            0.5 * self.robot_width + self.navigation_footprint_padding,
        )
        if self.navigation_inflation_radius <= padded_radius:
            raise ValueError("navigation inflation does not cover the footprint")
        if not 0.0 < self.path_corridor_half_width <= 0.05:
            raise ValueError("path corridor must preserve the Nav2 safety padding")
        if not 0.0 < self.path_longitudinal_half_width <= 0.30:
            raise ValueError("path longitudinal corridor is invalid")
        if self.maximum_path_cross_track_error < self.path_corridor_half_width:
            raise ValueError("maximum path error must contain the MPC corridor")


@dataclass(frozen=True)
class SafetyConfig:
    output_topic: str = "/cmd_vel_test"
    allow_hardware_output: bool = False
    maximum_speed: float = 0.60
    initial_hardware_speed_cap: float = 0.15
    maximum_yaw_rate: float = 1.2
    # Raw ``Twist.angular.z`` accepted by the current LIMO Ackermann driver.
    # This is a protocol-field limit, independent of the differential-drive
    # yaw-rate limit above.
    maximum_ackermann_angular_command: float = 0.198
    maximum_acceleration: float = 0.35
    front_stop_distance: float = 0.25
    front_sector_half_angle: float = radians(28.0)
    collision_inflation_margin: float = 0.05
    planner_timeout: float = 0.50
    odom_timeout: float = 0.25
    scan_timeout: float = 0.40
    status_timeout: float = 1.25
    arm_heartbeat_timeout: float = 0.75
    countdown_seconds: float = 3.0
    required_motion_mode: int = 1


@dataclass(frozen=True)
class IntegrationPreset:
    name: str
    decision_veto: bool
    mpc_risk_cost: bool
    cbf_risk_expansion: bool
    risk_weight: float
    decision_threshold: float
    cbf_alpha: float
    cbf_max_scale: float = 2.5
    risk_normalization: float = 1.5
    headway_beta: float = 0.4


PRESETS: Dict[str, IntegrationPreset] = {
    "baseline": IntegrationPreset(
        "baseline", False, False, False, 0.0, float("inf"), 0.0, 1.0
    ),
    "pure_mpc": IntegrationPreset(
        "pure_mpc", False, False, False, 0.0, float("inf"), 0.0, 1.0
    ),
    "conservative": IntegrationPreset(
        "conservative", True, True, True, 1.0, 1.0, 0.8
    ),
    "balanced": IntegrationPreset(
        "balanced", True, True, True, 0.5, 1.5, 0.6
    ),
    "permissive": IntegrationPreset(
        "permissive", True, True, True, 0.2, 2.0, 0.3, 2.0
    ),
}


@dataclass(frozen=True)
class DeploymentConfig:
    scale: SimilarityScale = field(default_factory=SimilarityScale)
    grid: GridConfig = field(default_factory=GridConfig)
    pde: PDEConfig = field(default_factory=PDEConfig)
    arena: ArenaConfig = field(default_factory=ArenaConfig)
    mpc: MPCConfig = field(default_factory=MPCConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def default_deployment_config() -> DeploymentConfig:
    config = DeploymentConfig()
    # Fail early if the checked-in values drift away from the stated scaling.
    expected_d0 = config.scale.diffusion(0.3)
    expected_d_occ = config.scale.diffusion(6.0)
    if abs(config.pde.d0 - expected_d0) > 1.0e-12:
        raise AssertionError("D0 is inconsistent with alpha^2/beta scaling")
    if abs(config.pde.d_occ - expected_d_occ) > 1.0e-12:
        raise AssertionError("D_occ is inconsistent with alpha^2/beta scaling")
    if abs(config.pde.source_scale - 1.0 / config.scale.beta) > 1.0e-12:
        raise AssertionError("Q scaling is inconsistent with beta")
    footprint_radius = (
        (0.5 * config.mpc.robot_length) ** 2
        + (0.5 * config.mpc.robot_width) ** 2
    ) ** 0.5 + config.safety.collision_inflation_margin
    if (
        config.arena.mission_goal_x + config.mpc.mission_position_tolerance
        > config.grid.x_max - footprint_radius + 0.5 * config.grid.resolution
    ):
        raise AssertionError("mission goal does not leave room for the safety footprint")
    return config


def deployment_config_for_arena(path_text: str) -> DeploymentConfig:
    """Apply deployment geometry while preserving scaled DREAM dynamics.

    The original merge experiment uses only the ``lanes`` and ``route`` keys.
    A free-navigation deployment may additionally declare a world-fixed
    ``grid``.  This keeps all ROS nodes on one immutable PDE/collision-grid
    contract without turning DRIFT into an ego-centred rolling field.

    ``limo_scale.py`` remains authoritative for every dimensional model and
    controller parameter; YAML is authoritative only for deployment geometry.
    """
    config = default_deployment_config()
    if not path_text:
        return config
    path = Path(path_text).expanduser()
    with path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream) or {}
    frame_id = str(payload.get("frame_id", config.grid.frame_id))
    if frame_id != config.grid.frame_id:
        raise ValueError(
            f"arena frame {frame_id!r} does not match risk-grid frame {config.grid.frame_id!r}"
        )
    grid_payload = payload.get("grid", {})
    if not isinstance(grid_payload, dict):
        raise ValueError("grid must be a mapping when provided")
    grid = replace(
        config.grid,
        x_min=float(grid_payload.get("x_min", config.grid.x_min)),
        x_max=float(grid_payload.get("x_max", config.grid.x_max)),
        y_min=float(grid_payload.get("y_min", config.grid.y_min)),
        y_max=float(grid_payload.get("y_max", config.grid.y_max)),
        resolution=float(
            grid_payload.get("resolution", config.grid.resolution)
        ),
        frame_id=frame_id,
        road_y_min=float(
            grid_payload.get("road_y_min", config.grid.road_y_min)
        ),
        road_y_max=float(
            grid_payload.get("road_y_max", config.grid.road_y_max)
        ),
        road_taper=float(
            grid_payload.get("road_taper", config.grid.road_taper)
        ),
    )
    if grid.road_y_min < grid.y_min or grid.road_y_max > grid.y_max:
        raise ValueError("road bounds must lie inside the numerical grid")
    config = replace(config, grid=grid)
    lanes = payload.get("lanes", {})
    centers = tuple(float(value) for value in lanes.get("centers", config.arena.lane_centers))
    width = float(lanes.get("width", config.arena.lane_width))
    route = payload.get("route", {})
    conflict_limits = tuple(
        float(value)
        for value in route.get(
            "conflict_zone_x",
            (config.arena.conflict_zone_x_min, config.arena.conflict_zone_x_max),
        )
    )
    if len(conflict_limits) != 2:
        raise ValueError("route.conflict_zone_x must contain [minimum, maximum]")
    merge_limits = tuple(
        float(value)
        for value in route.get(
            "merge_path_x",
            (config.arena.merge_path_x_min, config.arena.merge_path_x_max),
        )
    )
    if len(merge_limits) != 2:
        raise ValueError("route.merge_path_x must contain [minimum, maximum]")
    arena = replace(
        config.arena,
        lane_width=width,
        lane_centers=centers,
        target_lane=int(route.get("target_lane", config.arena.target_lane)),
        merge_request_x=float(route.get("merge_request_x", config.arena.merge_request_x)),
        merge_path_x_min=merge_limits[0],
        merge_path_x_max=merge_limits[1],
        conflict_zone_x_min=conflict_limits[0],
        conflict_zone_x_max=conflict_limits[1],
        mission_goal_x=float(
            route.get("mission_goal_x", config.arena.mission_goal_x)
        ),
    )
    if any(
        center <= config.grid.road_y_min or center >= config.grid.road_y_max
        for center in arena.lane_centers
    ):
        raise ValueError("surveyed lane centers must lie inside the configured road mask")
    if any(
        left <= right
        for left, right in zip(arena.lane_centers, arena.lane_centers[1:])
    ):
        raise ValueError("lane centers must be ordered left-to-right with decreasing y")
    footprint_radius = (
        (0.5 * config.mpc.robot_length) ** 2
        + (0.5 * config.mpc.robot_width) ** 2
    ) ** 0.5 + config.safety.collision_inflation_margin
    if (
        arena.mission_goal_x + config.mpc.mission_position_tolerance
        > config.grid.x_max - footprint_radius + 0.5 * config.grid.resolution
    ):
        raise ValueError("surveyed mission goal does not leave room for the safety footprint")
    return replace(config, arena=arena)


def get_preset(name: str) -> IntegrationPreset:
    try:
        return PRESETS[name]
    except KeyError as exc:
        raise ValueError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}") from exc
