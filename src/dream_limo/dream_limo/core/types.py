"""Typed, ROS-independent state exchanged by DREAM components."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import atan2, hypot, isfinite
from typing import Any, Dict, List, Mapping, Optional


@dataclass(frozen=True)
class Vehicle:
    vehicle_id: str
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    heading: float = 0.0
    vehicle_class: str = "car"
    length: float = 0.22
    width: float = 0.22
    acceleration: float = 0.0
    confidence: float = 1.0
    stamp: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.x,
            self.y,
            self.vx,
            self.vy,
            self.heading,
            self.length,
            self.width,
            self.acceleration,
            self.confidence,
            self.stamp,
        )
        if not all(isfinite(float(value)) for value in values):
            raise ValueError("vehicle fields must be finite")
        if self.length <= 0.0 or self.width <= 0.0:
            raise ValueError("vehicle dimensions must be positive")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must lie in [0, 1]")

    @property
    def speed(self) -> float:
        return hypot(self.vx, self.vy)

    def as_drift_dict(self) -> Dict[str, Any]:
        """Return the explicit dictionary expected by the original DRIFT API."""
        return {
            "id": self.vehicle_id,
            "x": self.x,
            "y": self.y,
            "vx": self.vx,
            "vy": self.vy,
            "heading": self.heading,
            "class": self.vehicle_class,
            "length": self.length,
            "width": self.width,
            "a": self.acceleration,
        }


@dataclass(frozen=True)
class EgoState:
    x: float
    y: float
    yaw: float
    speed: float
    yaw_rate: float = 0.0
    stamp: float = 0.0
    lane_index: int = 0

    def __post_init__(self) -> None:
        if not all(
            isfinite(float(value))
            for value in (self.x, self.y, self.yaw, self.speed, self.yaw_rate, self.stamp)
        ):
            raise ValueError("ego state must be finite")
        if self.speed < -1.0e-9:
            raise ValueError("reverse motion is not supported by the v1 planner")

    def as_vehicle(self) -> Vehicle:
        from math import cos, sin

        return Vehicle(
            vehicle_id="ego",
            x=self.x,
            y=self.y,
            vx=self.speed * cos(self.yaw),
            vy=self.speed * sin(self.yaw),
            heading=self.yaw,
            vehicle_class="car",
            stamp=self.stamp,
        )


@dataclass(frozen=True)
class TrackedAgent:
    agent_id: str
    class_label: str
    x: float
    y: float
    vx: float
    vy: float
    radius: float
    confidence: float
    stamp: float
    age: float
    source: str = "unknown"
    motion_state: str = "unknown"

    def to_vehicle(self, *, default_length: float = 0.22) -> Vehicle:
        diameter = max(0.05, 2.0 * self.radius)
        heading = atan2(self.vy, self.vx) if hypot(self.vx, self.vy) > 1.0e-6 else 0.0
        return Vehicle(
            vehicle_id=self.agent_id,
            x=self.x,
            y=self.y,
            vx=self.vx,
            vy=self.vy,
            heading=heading,
            vehicle_class=self.class_label,
            length=max(default_length, diameter),
            width=diameter,
            confidence=self.confidence,
            stamp=self.stamp,
        )


def parse_tracked_agents(
    payload: Any,
    *,
    now: Optional[float] = None,
    maximum_observation_age: float = 0.8,
    minimum_confidence: float = 0.0,
) -> List[TrackedAgent]:
    """Parse the public ``sfg_nav`` JSON schema without importing that package.

    Both the bare list emitted by the tracker and ``{"agents": [...]}`` are
    accepted. An empty list is a valid heartbeat.
    """
    if isinstance(payload, Mapping):
        payload = payload.get("agents")
    if not isinstance(payload, list):
        raise ValueError("tracked-agent payload must be a list or {'agents': list}")

    agents: List[TrackedAgent] = []
    for raw in payload:
        if not isinstance(raw, Mapping):
            raise ValueError("each tracked agent must be an object")
        position = raw.get("position", {})
        velocity = raw.get("velocity", {})
        agent = TrackedAgent(
            agent_id=str(raw.get("id", "")),
            class_label=str(raw.get("class_label", "unknown")),
            x=float(position["x"]),
            y=float(position["y"]),
            vx=float(velocity.get("x", 0.0)),
            vy=float(velocity.get("y", 0.0)),
            radius=float(raw.get("radius", 0.11)),
            confidence=float(raw.get("confidence", 0.0)),
            stamp=float(raw.get("stamp", 0.0)),
            age=float(raw.get("age", 0.0)),
            source=str(raw.get("source", "unknown")),
            motion_state=str(raw.get("motion_state", "unknown")),
        )
        if not all(
            isfinite(value)
            for value in (
                agent.x,
                agent.y,
                agent.vx,
                agent.vy,
                agent.radius,
                agent.confidence,
                agent.stamp,
                agent.age,
            )
        ):
            raise ValueError("tracked-agent fields must be finite")
        if agent.radius <= 0.0:
            raise ValueError("tracked-agent radius must be positive")
        if not 0.0 <= agent.confidence <= 1.0:
            raise ValueError("tracked-agent confidence must lie in [0, 1]")
        if agent.age < 0.0:
            raise ValueError("tracked-agent age cannot be negative")
        receive_age = 0.0 if now is None else max(0.0, float(now) - agent.stamp)
        if (
            agent.age <= maximum_observation_age
            and receive_age <= maximum_observation_age
            and agent.confidence >= minimum_confidence
        ):
            agents.append(agent)
    return agents


@dataclass(frozen=True)
class ControlCommand:
    target_speed: float
    acceleration: float
    steering: float
    stamp: float = 0.0
    valid: bool = True
    reason: str = "ok"


@dataclass
class WorldSnapshot:
    stamp: float
    ego: EgoState
    visible_vehicles: List[Vehicle] = field(default_factory=list)
    static_obstacles: List[Vehicle] = field(default_factory=list)
    hidden_merger_present_for_metrics_only: bool = False
    source_frame: str = "map"

    @property
    def drift_vehicles(self) -> List[Vehicle]:
        return [*self.static_obstacles, *self.visible_vehicles]
