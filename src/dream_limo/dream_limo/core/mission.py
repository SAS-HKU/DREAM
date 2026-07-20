"""Longitudinal mission-end profile and one-way completion latch."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt


def stopping_speed_limit(
    remaining_distance: float,
    *,
    cruise_speed: float,
    braking_deceleration: float,
) -> float:
    """Return the largest speed that can stop at the goal under constant braking.

    The square-root profile is the kinematic relation ``v^2 = 2*a*d``.  It is
    deliberately independent of the DREAM preset so the balanced and pure-MPC
    experiment arms receive the same mission-end behavior.
    """
    values = (remaining_distance, cruise_speed, braking_deceleration)
    if not all(isfinite(float(value)) for value in values):
        raise ValueError("mission speed-profile inputs must be finite")
    if cruise_speed < 0.0:
        raise ValueError("cruise_speed must be non-negative")
    if braking_deceleration <= 0.0:
        raise ValueError("braking_deceleration must be positive")
    distance = max(0.0, float(remaining_distance))
    return min(float(cruise_speed), sqrt(2.0 * float(braking_deceleration) * distance))


@dataclass
class MissionEndGuard:
    """Latch completion once the robot stops near, or crosses, the route goal."""

    goal_x: float
    position_tolerance: float
    stop_speed_tolerance: float
    _complete: bool = False

    def __post_init__(self) -> None:
        values = (self.goal_x, self.position_tolerance, self.stop_speed_tolerance)
        if not all(isfinite(float(value)) for value in values):
            raise ValueError("mission-end guard parameters must be finite")
        if self.position_tolerance < 0.0 or self.stop_speed_tolerance < 0.0:
            raise ValueError("mission-end tolerances must be non-negative")

    @property
    def complete(self) -> bool:
        return self._complete

    def remaining_distance(self, x: float) -> float:
        if not isfinite(float(x)):
            raise ValueError("mission position must be finite")
        return max(0.0, self.goal_x - float(x))

    def update(self, x: float, speed: float) -> bool:
        """Update and return the latch; completion cannot clear without restart."""
        if self._complete:
            return True
        if not isfinite(float(x)) or not isfinite(float(speed)):
            raise ValueError("mission state must be finite")
        position = float(x)
        absolute_speed = abs(float(speed))
        stopped_near_goal = (
            position >= self.goal_x - self.position_tolerance
            and absolute_speed <= self.stop_speed_tolerance
        )
        crossed_goal = position >= self.goal_x
        self._complete = stopped_near_goal or crossed_goal
        return self._complete
