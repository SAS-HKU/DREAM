"""Automated acceptance monitor for the isolated Stage 2 RViz smoke test."""

from __future__ import annotations

import json
from math import isfinite
from pathlib import Path as FilePath
from typing import Any, Dict, Optional

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, String
from visualization_msgs.msg import MarkerArray


class DreamSmokeMonitor(Node):
    """Collect one deterministic run and publish/write its acceptance report."""

    def __init__(self) -> None:
        super().__init__("dream_smoke_monitor")
        self.declare_parameter("report_path", "/tmp/dream_rviz_smoke_report.json")
        self.declare_parameter("minimum_shadow_cells", 100)
        self.declare_parameter("minimum_clearance", 0.22)
        self.declare_parameter("scenario_duration", 12.0)
        self.declare_parameter("experiment_arm", "balanced")

        self.receipts: Dict[str, Dict[str, float]] = {}
        self.preflight: Dict[str, Any] = {}
        self.world: Dict[str, Any] = {}
        self.drift: Dict[str, Any] = {}
        self.planner: Dict[str, Any] = {}
        self.safety: Dict[str, Any] = {}
        self.metrics: Dict[str, Any] = {}
        self.scenario: Dict[str, Any] = {}
        self.visible = False
        self.visibility_transitions = 0
        self.raw_merger_hidden_samples = 0
        self.hidden_track_leaks = 0
        self.reveal_track_samples = 0
        self.maximum_shadow_cells = 0
        self.maximum_consecutive_veto = 0
        self.current_consecutive_veto = 0
        self.veto_samples = 0
        self.route_merge_samples = 0
        self.mpc_fallbacks = 0
        self.planner_nonfinite = 0
        self.planner_rejections: Dict[str, int] = {}
        self.supervisor_triggers: Dict[str, int] = {}
        self.command_nonfinite = 0
        self.maximum_command_speed = 0.0
        self.countdown_nonzero = 0
        self.post_done_commands = 0
        self.post_done_nonzero = 0
        self.trailing_post_done_zero = 0
        self.reveal_wall_time: Optional[float] = None
        self.reveal_scenario_time: Optional[float] = None
        self.speed_at_reveal: Optional[float] = None
        self.minimum_post_reveal_speed = float("inf")
        self.latest_ego_speed = 0.0
        self.latest_ego_x = 0.0
        self.latest_ego_y = 0.0
        self.first_supervisor_trigger: Optional[Dict[str, Any]] = None
        self.done_wall_time: Optional[float] = None
        self.report: Optional[Dict[str, Any]] = None

        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        latched = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.create_subscription(String, "/dream/preflight_status", self._on_preflight, 10)
        self.create_subscription(String, "/dream/world_model", self._on_world, 10)
        self.create_subscription(String, "/dream/drift_status", self._on_drift, 10)
        self.create_subscription(String, "/dream/planner_status", self._on_planner, 10)
        self.create_subscription(String, "/dream/safety_status", self._on_safety, 10)
        self.create_subscription(String, "/dream/metrics", self._on_metrics, 10)
        self.create_subscription(String, "/dream/scenario_status", self._on_scenario, 10)
        self.create_subscription(Bool, "/dream/merger_visible", self._on_visibility, 10)
        self.create_subscription(Odometry, "/merger/wheel/odom", self._on_raw_merger, 10)
        self.create_subscription(Odometry, "/wheel/odom", lambda msg: self._on_odom(msg, "odom"), 20)
        self.create_subscription(
            Odometry, "/dream/ego_state", lambda msg: self._on_odom(msg, "ego"), 20
        )
        self.create_subscription(LaserScan, "/scan", lambda msg: self._record("scan"), sensor_qos)
        self.create_subscription(
            OccupancyGrid, "/dream/risk_field", lambda msg: self._record("risk_field"), 5
        )
        self.create_subscription(
            Path, "/dream/reference_trajectory", lambda msg: self._record("trajectory"), 5
        )
        self.create_subscription(
            MarkerArray, "/dream/scenario_markers", lambda msg: self._record("markers"), 10
        )
        self.create_subscription(Twist, "/dream/cmd_vel_candidate", self._on_candidate, 20)
        self.create_subscription(Twist, "/cmd_vel_test", self._on_command, 20)
        self.publisher = self.create_publisher(String, "/dream/smoke_status", latched)
        self.create_timer(0.2, self._tick)

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1.0e-9

    def _record(self, name: str) -> None:
        now = self._now()
        record = self.receipts.setdefault(name, {"count": 0.0, "first": now, "last": now})
        record["count"] += 1.0
        record["last"] = now

    @staticmethod
    def _decode(message: String) -> Dict[str, Any]:
        try:
            value = json.loads(message.data)
            return value if isinstance(value, dict) else {}
        except (json.JSONDecodeError, TypeError):
            return {}

    def _on_preflight(self, message: String) -> None:
        self._record("preflight")
        self.preflight = self._decode(message)

    def _on_world(self, message: String) -> None:
        self._record("world")
        self.world = self._decode(message)
        ids = {
            str(item.get("vehicle_id", item.get("id", "")))
            for item in self.world.get("vehicles", [])
            if isinstance(item, dict)
        }
        world_visible = bool(self.world.get("merger_visible", False))
        if not world_visible and "merger_odom" in ids:
            self.hidden_track_leaks += 1
        if world_visible and "merger_odom" in ids:
            self.reveal_track_samples += 1
        self.maximum_shadow_cells = max(
            self.maximum_shadow_cells, int(self.world.get("shadow_cells", 0))
        )

    def _on_drift(self, message: String) -> None:
        self._record("drift")
        self.drift = self._decode(message)

    def _on_planner(self, message: String) -> None:
        self._record("planner")
        self.planner = self._decode(message)
        if not self.planner.get("ready", False):
            reason = str(self.planner.get("reason", "NOT_READY"))
            if (
                self.scenario.get("scenario_time") is not None
                and not bool(self.scenario.get("done", False))
                and reason not in {"STALE_INPUT", "DRIFT_NOT_READY"}
            ):
                self.planner_rejections[reason] = self.planner_rejections.get(reason, 0) + 1
            self.current_consecutive_veto = 0
            return
        finite_fields = ("decision_risk", "risk_at_ego", "target_speed", "acceleration", "center_steer")
        try:
            if not all(isfinite(float(self.planner[name])) for name in finite_fields):
                self.planner_nonfinite += 1
        except (KeyError, TypeError, ValueError):
            self.planner_nonfinite += 1
        if bool(self.planner.get("mpc_fallback", False)):
            self.mpc_fallbacks += 1
        desired_veto = (
            not self.visible
            and int(self.planner.get("requested_lane", -1)) == 1
            and int(self.planner.get("selected_lane", -1)) == 0
            and bool(self.planner.get("vetoed", False))
        )
        if desired_veto:
            self.veto_samples += 1
            self.current_consecutive_veto += 1
            self.maximum_consecutive_veto = max(
                self.maximum_consecutive_veto, self.current_consecutive_veto
            )
        else:
            self.current_consecutive_veto = 0
        if (
            bool(self.planner.get("route_intent_active", False))
            and int(self.planner.get("requested_lane", -1)) == 1
            and int(self.planner.get("selected_lane", -1)) == 1
            and not bool(self.planner.get("vetoed", False))
        ):
            self.route_merge_samples += 1

    def _on_safety(self, message: String) -> None:
        self._record("safety")
        self.safety = self._decode(message)
        reason = str(self.safety.get("reason", "UNKNOWN"))
        if (
            self.scenario.get("scenario_time") is not None
            and not bool(self.scenario.get("done", False))
            and reason not in {"ok", "COUNTDOWN", "NOT_ARMED"}
        ):
            self.supervisor_triggers[reason] = self.supervisor_triggers.get(reason, 0) + 1
            if self.first_supervisor_trigger is None:
                self.first_supervisor_trigger = {
                    "reason": reason,
                    "scenario_time": self.scenario.get("scenario_time"),
                    "ego_x": self.latest_ego_x,
                    "ego_y": self.latest_ego_y,
                    "ego_speed": self.latest_ego_speed,
                    "front_minimum_range": self.safety.get("front_minimum_range"),
                    "front_stop_distance": self.safety.get("front_stop_distance"),
                    "merger_x": self.scenario.get("merger_x"),
                    "merger_y": self.scenario.get("merger_y"),
                }
        if self.safety.get("reason") == "COUNTDOWN" and (
            abs(float(self.safety.get("linear_x", 0.0))) > 1.0e-6
            or abs(float(self.safety.get("angular_z", 0.0))) > 1.0e-6
        ):
            self.countdown_nonzero += 1

    def _on_metrics(self, message: String) -> None:
        self._record("metrics")
        self.metrics = self._decode(message)

    def _on_scenario(self, message: String) -> None:
        self._record("scenario")
        self.scenario = self._decode(message)
        if bool(self.scenario.get("done", False)) and self.done_wall_time is None:
            self.done_wall_time = self._now()

    def _on_visibility(self, message: Bool) -> None:
        self._record("visibility")
        current = bool(message.data)
        if current and not self.visible:
            self.visibility_transitions += 1
            if self.reveal_wall_time is None:
                self.reveal_wall_time = self._now()
                value = self.scenario.get("scenario_time")
                self.reveal_scenario_time = float(value) if value is not None else None
                self.speed_at_reveal = self.latest_ego_speed
        self.visible = current

    def _on_raw_merger(self, _message: Odometry) -> None:
        self._record("raw_merger")
        if not self.visible:
            self.raw_merger_hidden_samples += 1

    def _on_odom(self, message: Odometry, name: str) -> None:
        self._record(name)
        if name != "ego":
            return
        self.latest_ego_x = float(message.pose.pose.position.x)
        self.latest_ego_y = float(message.pose.pose.position.y)
        self.latest_ego_speed = abs(float(message.twist.twist.linear.x))
        if self.reveal_wall_time is not None and self._now() <= self.reveal_wall_time + 4.0:
            self.minimum_post_reveal_speed = min(
                self.minimum_post_reveal_speed, self.latest_ego_speed
            )

    def _on_candidate(self, message: Twist) -> None:
        self._record("candidate")
        if not all(isfinite(value) for value in (message.linear.x, message.angular.z)):
            self.command_nonfinite += 1

    def _on_command(self, message: Twist) -> None:
        self._record("safe_command")
        values = (float(message.linear.x), float(message.angular.z))
        if not all(isfinite(value) for value in values):
            self.command_nonfinite += 1
            return
        self.maximum_command_speed = max(self.maximum_command_speed, abs(values[0]))
        nonzero = abs(values[0]) > 1.0e-6 or abs(values[1]) > 1.0e-6
        if self.done_wall_time is not None:
            self.post_done_commands += 1
            if nonzero:
                self.post_done_nonzero += 1
                self.trailing_post_done_zero = 0
            else:
                self.trailing_post_done_zero += 1

    def _topic_rates(self) -> Dict[str, Optional[float]]:
        rates: Dict[str, Optional[float]] = {}
        for name, record in self.receipts.items():
            duration = record["last"] - record["first"]
            rates[name] = (
                (record["count"] - 1.0) / duration
                if record["count"] > 1.0 and duration > 0.0
                else None
            )
        return rates

    def _evaluate(self) -> Dict[str, Any]:
        command_publishers = self.get_publishers_info_by_topic("/cmd_vel")
        test_publishers = self.get_publishers_info_by_topic("/cmd_vel_test")
        clearance = self.metrics.get("minimum_clearance")
        minimum_clearance = float(clearance) if clearance is not None else None
        arm = str(self.get_parameter("experiment_arm").value)
        dream_arm = arm not in {"baseline", "pure_mpc"}
        yield_observed = False
        if self.speed_at_reveal is not None and isfinite(self.minimum_post_reveal_speed):
            speed_drop = self.speed_at_reveal - self.minimum_post_reveal_speed
            # DREAM is expected to carry margin into reveal. The baseline is
            # expected to react later, but a meaningful 0.10 m/s post-reveal
            # reduction is sufficient for this smoke check; it must not be
            # misreported as the stronger DREAM yield behavior.
            yield_observed = (
                self.speed_at_reveal <= 0.47
                or self.minimum_post_reveal_speed <= 0.18
                or speed_drop >= (0.15 if dream_arm else 0.10)
            )
        decision_behavior_matches_arm = (
            self.maximum_consecutive_veto >= 2
            if dream_arm
            else self.route_merge_samples >= 2 and self.veto_samples == 0
        )
        overlap_samples = int(self.metrics.get("conflict_zone_overlap_samples", 0))
        conflict_behavior_matches_arm = (
            overlap_samples == 0 if dream_arm else overlap_samples > 0
        )
        checks = {
            "preflight_passed": bool(self.preflight.get("passed", False)),
            "no_cmd_vel_publisher": len(command_publishers) == 0,
            "single_safe_output_owner": (
                len(test_publishers) == 1
                and test_publishers[0].node_name == "dream_safety_supervisor"
            ),
            "drift_ready_after_warmup": (
                bool(self.drift.get("ready", False))
                and float(self.drift.get("warmup_model_seconds", 0.0)) >= 5.0
            ),
            "risk_field_received": self.receipts.get("risk_field", {}).get("count", 0) > 0,
            "lidar_shadow_present": self.maximum_shadow_cells
            >= int(self.get_parameter("minimum_shadow_cells").value),
            "hidden_truth_observed": self.raw_merger_hidden_samples > 0,
            "hidden_track_gate_clean": self.hidden_track_leaks == 0,
            "single_reveal_transition": self.visibility_transitions == 1,
            "revealed_track_reaches_world": self.reveal_track_samples > 0,
            "decision_behavior_matches_arm": decision_behavior_matches_arm,
            "conflict_zone_behavior_matches_arm": conflict_behavior_matches_arm,
            "no_mpc_fallback": self.mpc_fallbacks == 0,
            "no_planner_safety_rejection": not self.planner_rejections,
            "no_supervisor_trigger": not self.supervisor_triggers,
            "planner_values_finite": self.planner_nonfinite == 0,
            "commands_finite_and_capped": (
                self.command_nonfinite == 0 and self.maximum_command_speed <= 0.600001
            ),
            "countdown_output_zero": self.countdown_nonzero == 0,
            "trajectory_received": self.receipts.get("trajectory", {}).get("count", 0) > 0,
            "rviz_markers_received": self.receipts.get("markers", {}).get("count", 0) > 0,
            "slow_or_yield_after_reveal": yield_observed,
            "clearance_at_least_one_robot_width": (
                minimum_clearance is not None
                and minimum_clearance
                >= float(self.get_parameter("minimum_clearance").value)
            ),
            "scenario_completed": bool(self.scenario.get("done", False)),
            "final_output_zero": self.trailing_post_done_zero >= 5,
        }
        rates = self._topic_rates()
        recommended_rates = {
            "odom": 18.0,
            "ego": 18.0,
            "scan": 5.5,
            "world": 8.0,
            "raw_merger": 8.0,
            "risk_field": 4.0,
            "planner": 4.0,
            "trajectory": 4.0,
            "safe_command": 18.0,
            "markers": 8.0,
        }
        rate_warnings = {
            name: {"observed": rates.get(name), "recommended": minimum}
            for name, minimum in recommended_rates.items()
            if rates.get(name) is None or rates[name] < minimum
        }
        warnings = []
        maximum_mpc = float(self.metrics.get("maximum_mpc_seconds", 0.0))
        if maximum_mpc > 0.10:
            warnings.append(
                f"maximum MPC solve {1000.0 * maximum_mpc:.1f} ms exceeds the 100 ms profile target"
            )
        if self.metrics.get("ttc_at_reveal") is None:
            warnings.append(
                "TTC at reveal is not scoreable "
                f"({self.metrics.get('ttc_at_reveal_reason', 'UNKNOWN')}); "
                "null is not interpreted as infinity"
            )
        if rate_warnings:
            warnings.append("one or more observed topic rates were below the smoke-test recommendation")
        return {
            "experiment_arm": arm,
            "passed": all(checks.values()),
            "checks": checks,
            "warnings": warnings,
            "observations": {
                "phase": self.scenario.get("phase"),
                "reveal_scenario_time": self.reveal_scenario_time,
                "speed_at_reveal": self.speed_at_reveal,
                "minimum_post_reveal_speed": (
                    self.minimum_post_reveal_speed
                    if isfinite(self.minimum_post_reveal_speed)
                    else None
                ),
                "minimum_clearance": minimum_clearance,
                "minimum_ttc": self.metrics.get("minimum_ttc"),
                "projected_conflict_arrival_margin_at_reveal": self.metrics.get(
                    "projected_conflict_arrival_margin_at_reveal"
                ),
                "ego_conflict_entry_time": self.metrics.get("ego_conflict_entry_time"),
                "merger_conflict_entry_time": self.metrics.get(
                    "merger_conflict_entry_time"
                ),
                "conflict_zone_overlap_samples": overlap_samples,
                "minimum_conflict_zone_clearance": self.metrics.get(
                    "minimum_conflict_zone_clearance"
                ),
                "maximum_shadow_cells": self.maximum_shadow_cells,
                "maximum_consecutive_veto": self.maximum_consecutive_veto,
                "veto_samples": self.veto_samples,
                "route_merge_samples": self.route_merge_samples,
                "hidden_track_leaks": self.hidden_track_leaks,
                "visibility_transitions": self.visibility_transitions,
                "mpc_fallbacks": self.mpc_fallbacks,
                "planner_rejections": self.planner_rejections,
                "supervisor_triggers": self.supervisor_triggers,
                "first_supervisor_trigger": self.first_supervisor_trigger,
                "maximum_mpc_seconds": maximum_mpc,
                "maximum_drift_seconds": self.metrics.get("maximum_drift_seconds"),
                "maximum_command_speed": self.maximum_command_speed,
                "countdown_nonzero_samples": self.countdown_nonzero,
                "post_done_nonzero_samples": self.post_done_nonzero,
                "trailing_post_done_zero_samples": self.trailing_post_done_zero,
                "topic_rates_hz": rates,
                "rate_warnings": rate_warnings,
            },
            "preflight": self.preflight,
            "metrics": self.metrics,
            "note": "This SIL result does not authorize physical motion.",
        }

    def _publish_report(self) -> None:
        message = String()
        message.data = json.dumps(self.report, separators=(",", ":"), allow_nan=False)
        self.publisher.publish(message)

    def _tick(self) -> None:
        if self.report is not None:
            self._publish_report()
            return
        if self.done_wall_time is None or self._now() - self.done_wall_time < 1.0:
            return
        self.report = self._evaluate()
        path = FilePath(str(self.get_parameter("report_path").value)).expanduser()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self.report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            self.get_logger().info(
                f"RViz smoke {'PASS' if self.report['passed'] else 'FAIL'}: {path}"
            )
        except OSError as exc:
            self.get_logger().error(f"Could not write smoke report: {exc}")
        self._publish_report()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DreamSmokeMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
