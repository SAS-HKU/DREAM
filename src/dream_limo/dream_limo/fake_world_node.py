"""Sensor-level scripted world for Stage 2 SIL; never talks to the base driver."""

from __future__ import annotations

from dataclasses import replace
import json
from math import cos, sin, tan
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import TransformStamped, Twist, TwistStamped
from limo_msgs.msg import LimoStatus
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, String
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from tf2_ros.transform_broadcaster import TransformBroadcaster

from .core.occlusion import PolygonObstacle, rectangle_polygon, simulate_polygon_scan
from .core.replay import _merger_state
from .core.types import Vehicle
from .limo_scale import deployment_config_for_arena
from .ros_utils import yaw_to_quaternion


class DreamFakeWorldNode(Node):
    def __init__(self) -> None:
        super().__init__("dream_fake_world")
        self.declare_parameter("auto_arm", True)
        self.declare_parameter("arena_file", "")
        self.config = deployment_config_for_arena(
            str(self.get_parameter("arena_file").value)
        )
        self.declare_parameter("initial_x", 0.35)
        self.declare_parameter("initial_y", self.config.arena.lane_centers[0])
        self.declare_parameter("scenario_delay", 2.0)
        self.declare_parameter("scenario_duration", 12.0)
        self.x = float(self.get_parameter("initial_x").value)
        self.y = float(self.get_parameter("initial_y").value)
        self.yaw = 0.0
        self.speed = 0.0
        self.center_steer = 0.0
        self.process_started = self._now()
        self.last_update = self.process_started
        self.obstacles = self._load_arena(str(self.get_parameter("arena_file").value))
        self.truck = self.obstacles["truck"]
        self.drift_ready = False
        self.preflight_passed = False
        self.safety_reason = "WAITING"
        self.prerequisites_since: Optional[float] = None
        self.scenario_started: Optional[float] = None
        self.scenario_finished = False
        self.latest_odom_stamp = None

        self.odom_publisher = self.create_publisher(Odometry, "/wheel/odom", 20)
        self.scan_publisher = self.create_publisher(LaserScan, "/scan", 5)
        self.track_publisher = self.create_publisher(String, "/tracked_agents", 10)
        self.status_publisher = self.create_publisher(LimoStatus, "/limo_status", 10)
        self.merger_publisher = self.create_publisher(Odometry, "/merger/wheel/odom", 10)
        self.arm_publisher = self.create_publisher(Bool, "/dream/arm", 10)
        self.scenario_publisher = self.create_publisher(String, "/dream/scenario_status", 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)
        self._publish_laser_transform()
        self.create_subscription(Twist, "/cmd_vel_test", self._on_safe_command, 10)
        self.create_subscription(TwistStamped, "/dream/control", self._on_control, 10)
        self.create_subscription(Bool, "/dream/drift_ready", self._on_drift_ready, 10)
        self.create_subscription(String, "/dream/preflight_status", self._on_preflight, 10)
        self.create_subscription(String, "/dream/safety_status", self._on_safety, 10)
        self.create_timer(0.05, self._update_and_publish_odom)
        self.create_timer(1.0 / 6.1, self._publish_scan)
        self.create_timer(0.10, self._publish_tracks_and_merger)
        self.create_timer(0.50, self._publish_status)
        self.create_timer(0.50, self._publish_arm)

    @staticmethod
    def _load_arena(path_text: str) -> Dict[str, PolygonObstacle]:
        if not path_text:
            raise RuntimeError("arena_file must be supplied to the SIL world")
        with Path(path_text).expanduser().open("r", encoding="utf-8") as stream:
            payload = yaml.safe_load(stream) or {}
        obstacles: Dict[str, PolygonObstacle] = {}
        for entry in payload.get("obstacles", []):
            center_x, center_y = (float(value) for value in entry["center"])
            length, width = (float(value) for value in entry["size"])
            name = str(entry["id"])
            obstacles[name] = rectangle_polygon(
                name,
                center_x,
                center_y,
                length,
                width,
                float(entry.get("heading", 0.0)),
                str(entry.get("class", "car")),
            )
        missing = {"truck"} - set(obstacles)
        if missing:
            raise RuntimeError(f"SIL arena is missing required obstacles: {sorted(missing)}")
        return obstacles

    def _publish_laser_transform(self) -> None:
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = "base_link"
        transform.child_frame_id = "laser_link"
        transform.transform.translation.x = 0.10
        transform.transform.rotation.w = 1.0
        self.static_tf_broadcaster.sendTransform(transform)

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1.0e-9

    def _on_safe_command(self, message: Twist) -> None:
        self.speed = float(np.clip(message.linear.x, 0.0, self.config.mpc.maximum_speed))

    def _on_drift_ready(self, message: Bool) -> None:
        self.drift_ready = bool(message.data)

    def _on_preflight(self, message: String) -> None:
        try:
            self.preflight_passed = bool(json.loads(message.data).get("passed", False))
        except (json.JSONDecodeError, TypeError):
            self.preflight_passed = False

    def _on_safety(self, message: String) -> None:
        try:
            self.safety_reason = str(json.loads(message.data).get("reason", "UNKNOWN"))
        except (json.JSONDecodeError, TypeError):
            self.safety_reason = "INVALID_STATUS"

    def _arm_active(self, now: float) -> bool:
        prerequisites = self.drift_ready and self.preflight_passed
        if not prerequisites:
            self.prerequisites_since = None
            return False
        if self.prerequisites_since is None:
            self.prerequisites_since = now
        delay = float(self.get_parameter("scenario_delay").value)
        return (
            bool(self.get_parameter("auto_arm").value)
            and now - self.prerequisites_since >= delay
            and not self.scenario_finished
        )

    def _scenario_time(self, now: Optional[float] = None) -> Optional[float]:
        now = self._now() if now is None else now
        armed = self._arm_active(now)
        if self.scenario_started is None and armed and self.safety_reason == "ok":
            self.scenario_started = now
        if self.scenario_started is None:
            return None
        duration = float(self.get_parameter("scenario_duration").value)
        elapsed = max(0.0, now - self.scenario_started)
        if elapsed >= duration:
            self.scenario_finished = True
            return duration
        return elapsed

    def _merger_state(self, now: Optional[float] = None) -> Vehicle:
        scenario_time = self._scenario_time(now)
        if scenario_time is None:
            return replace(_merger_state(0.0), vx=0.0, vy=0.0, heading=0.0, stamp=0.0)
        return _merger_state(scenario_time)

    def _on_control(self, message: TwistStamped) -> None:
        self.center_steer = float(
            np.clip(
                message.twist.angular.z,
                -self.config.mpc.maximum_steer,
                self.config.mpc.maximum_steer,
            )
        )

    def _update_and_publish_odom(self) -> None:
        now = self._now()
        dt = min(0.10, max(0.0, now - self.last_update))
        self.last_update = now
        self.x += dt * self.speed * cos(self.yaw)
        self.y += dt * self.speed * sin(self.yaw)
        self.yaw += dt * self.speed / self.config.mpc.wheelbase * tan(self.center_steer)
        message = Odometry()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "odom"
        message.child_frame_id = "base_link"
        message.pose.pose.position.x = self.x
        message.pose.pose.position.y = self.y
        qx, qy, qz, qw = yaw_to_quaternion(self.yaw)
        message.pose.pose.orientation.x = qx
        message.pose.pose.orientation.y = qy
        message.pose.pose.orientation.z = qz
        message.pose.pose.orientation.w = qw
        message.twist.twist.linear.x = self.speed
        message.twist.twist.angular.z = self.speed / self.config.mpc.wheelbase * tan(self.center_steer)
        self.odom_publisher.publish(message)
        self.latest_odom_stamp = message.header.stamp
        transform = TransformStamped()
        transform.header = message.header
        transform.child_frame_id = message.child_frame_id
        transform.transform.translation.x = self.x
        transform.transform.translation.y = self.y
        transform.transform.rotation = message.pose.pose.orientation
        self.tf_broadcaster.sendTransform(transform)

    def _publish_scan(self) -> None:
        merger = self._merger_state()
        merger_polygon = rectangle_polygon(
            "merger",
            merger.x,
            merger.y,
            merger.length,
            merger.width,
            merger.heading,
            "car",
        )
        scan = simulate_polygon_scan(
            (self.x + 0.10 * cos(self.yaw), self.y + 0.10 * sin(self.yaw), self.yaw),
            [*self.obstacles.values(), merger_polygon],
            range_max=6.0,
            stamp=self._now(),
        )
        message = LaserScan()
        message.header.stamp = (
            self.latest_odom_stamp
            if self.latest_odom_stamp is not None
            else self.get_clock().now().to_msg()
        )
        message.header.frame_id = "laser_link"
        message.angle_min = scan.angle_min
        message.angle_max = scan.angle_min + (len(scan.ranges) - 1) * scan.angle_increment
        message.angle_increment = scan.angle_increment
        message.scan_time = 1.0 / 6.1
        message.time_increment = message.scan_time / len(scan.ranges)
        message.range_min = scan.range_min
        message.range_max = scan.range_max
        message.ranges = scan.ranges.astype(np.float32).tolist()
        self.scan_publisher.publish(message)

    def _publish_tracks_and_merger(self) -> None:
        empty = String()
        empty.data = "[]"
        self.track_publisher.publish(empty)
        now = self._now()
        wall_elapsed = now - self.process_started
        scenario_time = self._scenario_time(now)
        merger = self._merger_state(now)
        message = Odometry()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "odom"
        message.child_frame_id = "merger/base_link"
        message.pose.pose.position.x = merger.x
        message.pose.pose.position.y = merger.y
        qx, qy, qz, qw = yaw_to_quaternion(merger.heading)
        message.pose.pose.orientation.x = qx
        message.pose.pose.orientation.y = qy
        message.pose.pose.orientation.z = qz
        message.pose.pose.orientation.w = qw
        message.twist.twist.linear.x = merger.vx
        message.twist.twist.linear.y = merger.vy
        self.merger_publisher.publish(message)
        armed = self._arm_active(now)
        if not self.drift_ready:
            phase = "WAITING_DRIFT"
        elif not self.preflight_passed:
            phase = "WAITING_PREFLIGHT"
        elif self.scenario_finished:
            phase = "DONE"
        elif not armed:
            phase = "RVIZ_PREROLL"
        elif scenario_time is None:
            phase = "COUNTDOWN"
        elif scenario_time < 3.0:
            phase = "HIDDEN_APPROACH"
        elif scenario_time < 6.0:
            phase = "OCCLUDED_MERGE"
        else:
            phase = "REVEAL_WINDOW"
        status = String()
        status.data = json.dumps(
            {
                "phase": phase,
                "wall_elapsed": wall_elapsed,
                "scenario_time": scenario_time,
                "merger_x": merger.x,
                "merger_y": merger.y,
                "drift_ready": self.drift_ready,
                "preflight_passed": self.preflight_passed,
                "armed": armed,
                "safety_reason": self.safety_reason,
                "done": self.scenario_finished,
            },
            separators=(",", ":"),
        )
        self.scenario_publisher.publish(status)
        if self.scenario_finished:
            self.arm_publisher.publish(Bool(data=False))

    def _publish_status(self) -> None:
        message = LimoStatus()
        message.header.stamp = self.get_clock().now().to_msg()
        message.motion_mode = 1
        message.control_mode = 1
        message.vehicle_state = 0
        message.error_code = 0
        message.battery_voltage = 12.4
        self.status_publisher.publish(message)

    def _publish_arm(self) -> None:
        message = Bool()
        message.data = self._arm_active(self._now())
        self.arm_publisher.publish(message)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DreamFakeWorldNode()
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
