"""Straight-arena IDEAM gap decision with the DREAM risk veto."""

from __future__ import annotations

from dataclasses import dataclass
from math import inf
from time import perf_counter
from typing import List, Optional, Sequence, Tuple

from dream_limo.limo_scale import DeploymentConfig, IntegrationPreset

from .risk_field import DREAMRiskField
from .types import EgoState, Vehicle


@dataclass(frozen=True)
class GapGroup:
    name: str
    lane_index: int
    leader: Optional[Vehicle]
    follower: Optional[Vehicle]
    front_clearance: float
    rear_clearance: float
    feasible: bool
    efficiency: float


@dataclass(frozen=True)
class DecisionResult:
    maneuver: str
    current_lane: int
    requested_lane: int
    selected_lane: int
    vetoed: bool
    risk_score: float
    risk_maximum: float
    risk_mean: float
    reason: str
    compute_seconds: float
    groups: Tuple[GapGroup, ...]


class IDEAMDREAMDecision:
    """Reduced six-gap IDEAM graph for three surveyed straight lanes.

    Each lane contributes a leader/follower gap pair (L1/L2, C1/C2, R1/R2).
    The adjacency graph permits only an adjacent-lane transition.  DREAM then
    vetoes the selected transition using the live DRIFT field.
    """

    LABELS = ("L", "C", "R")

    def __init__(
        self,
        config: DeploymentConfig,
        *,
        blocker_trigger_distance: float = 2.5,
        lane_capture_tolerance: Optional[float] = None,
    ) -> None:
        self.config = config
        self.blocker_trigger_distance = float(blocker_trigger_distance)
        self.lane_capture_tolerance = (
            0.55 * config.arena.lane_width
            if lane_capture_tolerance is None
            else float(lane_capture_tolerance)
        )

    def lane_for_y(self, y: float) -> int:
        centers = self.config.arena.lane_centers
        index = min(range(len(centers)), key=lambda item: abs(centers[item] - y))
        if abs(centers[index] - y) > self.config.arena.lane_width:
            raise ValueError(f"y={y:.3f} lies outside the configured lane corridor")
        return index

    @staticmethod
    def _clearance(ego: EgoState, vehicle: Vehicle, *, leader: bool) -> float:
        half_extent = 0.5 * (0.22 + vehicle.length)
        return (
            vehicle.x - ego.x - half_extent
            if leader
            else ego.x - vehicle.x - half_extent
        )

    def _lane_neighbors(self, lane_index: int) -> Tuple[int, ...]:
        result = []
        if lane_index > 0:
            result.append(lane_index - 1)
        if lane_index + 1 < len(self.config.arena.lane_centers):
            result.append(lane_index + 1)
        return tuple(result)

    def build_groups(
        self,
        ego: EgoState,
        vehicles: Sequence[Vehicle],
        risk_field: DREAMRiskField,
        preset: IntegrationPreset,
    ) -> Tuple[GapGroup, ...]:
        groups: List[GapGroup] = []
        mpc = self.config.mpc
        headway_scale = risk_field.headway_scale(ego.x, ego.y, preset)
        front_required = headway_scale * (
            mpc.base_minimum_distance + mpc.base_headway * ego.speed
        )
        for lane_index, lane_y in enumerate(self.config.arena.lane_centers):
            lane_vehicles = [
                vehicle
                for vehicle in vehicles
                if abs(vehicle.y - lane_y) <= self.lane_capture_tolerance
            ]
            ahead = [vehicle for vehicle in lane_vehicles if vehicle.x >= ego.x]
            behind = [vehicle for vehicle in lane_vehicles if vehicle.x < ego.x]
            leader = min(ahead, key=lambda item: item.x, default=None)
            follower = max(behind, key=lambda item: item.x, default=None)
            front = inf if leader is None else self._clearance(ego, leader, leader=True)
            rear = inf if follower is None else self._clearance(ego, follower, leader=False)
            follower_speed = 0.0 if follower is None else follower.speed
            rear_required = headway_scale * (
                0.8 * mpc.base_minimum_distance + 0.4 * follower_speed
            )
            feasible = front >= front_required and rear >= rear_required
            lane_speed = mpc.target_speed if leader is None else min(
                mpc.target_speed, max(0.0, leader.speed)
            )
            # Long-term efficiency dominates; a small transition cost prevents
            # gratuitous lane changes in equal traffic.
            transition_cost = 0.03 * abs(lane_index - ego.lane_index)
            efficiency = lane_speed - transition_cost
            prefix = self.LABELS[lane_index]
            groups.extend(
                (
                    GapGroup(
                        f"{prefix}1",
                        lane_index,
                        leader,
                        follower,
                        front,
                        rear,
                        feasible,
                        efficiency,
                    ),
                    GapGroup(
                        f"{prefix}2",
                        lane_index,
                        leader,
                        follower,
                        front,
                        rear,
                        feasible,
                        efficiency,
                    ),
                )
            )
        return tuple(groups)

    def _request_lane(self, ego: EgoState, groups: Tuple[GapGroup, ...]) -> Tuple[int, str]:
        representatives = {group.lane_index: group for group in groups[::2]}
        current = representatives[ego.lane_index]
        if current.leader is None or current.front_clearance > self.blocker_trigger_distance:
            return ego.lane_index, "current lane remains efficient"
        candidates = [
            representatives[index]
            for index in self._lane_neighbors(ego.lane_index)
            if representatives[index].feasible
        ]
        if not candidates:
            return ego.lane_index, "blocker present but no adjacent feasible gap"
        best = max(candidates, key=lambda item: (item.efficiency, -abs(item.lane_index - ego.lane_index)))
        return best.lane_index, f"blocker triggers transition toward {best.name}"

    def decide(
        self,
        ego: EgoState,
        vehicles: Sequence[Vehicle],
        risk_field: DREAMRiskField,
        preset: IntegrationPreset,
        *,
        requested_lane: Optional[int] = None,
    ) -> DecisionResult:
        started = perf_counter()
        groups = self.build_groups(ego, vehicles, risk_field, preset)
        if requested_lane is None:
            request, reason = self._request_lane(ego, groups)
        else:
            request = int(requested_lane)
            reason = "external route requests adjacent lane"
        if not 0 <= request < len(self.config.arena.lane_centers):
            raise ValueError("requested lane is out of range")
        if abs(request - ego.lane_index) > 1:
            raise ValueError("IDEAM graph forbids non-adjacent lane changes")

        selected = request
        vetoed = False
        score = maximum = mean = 0.0
        if request != ego.lane_index:
            score, maximum, mean, _ = risk_field.lane_change_risk(ego, request)
            if preset.decision_veto and score > preset.decision_threshold:
                selected = ego.lane_index
                vetoed = True
                reason = (
                    f"DREAM veto {score:.3f} exceeds threshold "
                    f"{preset.decision_threshold:.3f}"
                )
        delta = selected - ego.lane_index
        maneuver = "K" if delta == 0 else ("R" if delta > 0 else "L")
        return DecisionResult(
            maneuver=maneuver,
            current_lane=ego.lane_index,
            requested_lane=request,
            selected_lane=selected,
            vetoed=vetoed,
            risk_score=score,
            risk_maximum=maximum,
            risk_mean=mean,
            reason=reason,
            compute_seconds=perf_counter() - started,
            groups=groups,
        )
