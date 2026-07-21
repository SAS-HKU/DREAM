"""DREAM-owned merger-vehicle adapter for SFG's neutral LiDAR clusters."""

from __future__ import annotations

import json
import math
import signal

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

from .core.vehicle_tracker import (
    MergerVehicleTracker,
    VehicleTrack,
    parse_cluster_payload,
    track_to_agent_payload,
    validate_cluster_source_stamp,
)


class DreamVehicleTrackerNode(Node):
    """Publish only motion-confirmed, vehicle-sized cluster tracks."""

    def __init__(self) -> None:
        super().__init__("dream_vehicle_tracker")
        self._declare_parameters()
        self._load_parameters()
        self.tracker = MergerVehicleTracker(
            association_distance_m=self.association_distance_m,
            association_noise_margin_m=self.association_noise_margin_m,
            maximum_vehicle_speed_mps=self.maximum_vehicle_speed_mps,
            maximum_width_change_m=self.maximum_width_change_m,
            velocity_alpha=self.velocity_alpha,
            position_alpha=self.position_alpha,
            coast_timeout_sec=self.coast_timeout_sec,
            stale_remove_sec=self.stale_remove_sec,
            motion_window_sec=self.motion_window_sec,
            motion_enter_speed_mps=self.motion_enter_speed_mps,
            motion_exit_speed_mps=self.motion_exit_speed_mps,
            motion_min_displacement_m=self.motion_min_displacement_m,
            motion_hold_sec=self.motion_hold_sec,
            minimum_track_hits=self.minimum_track_hits,
            minimum_consistent_motion_windows=(
                self.minimum_consistent_motion_windows
            ),
            minimum_direction_cosine=self.minimum_direction_cosine,
        )
        self.last_valid_input_sec: float | None = None
        self.last_source_stamp: float | None = None
        self.last_error = "waiting_for_clusters"
        self.raw_cluster_count = 0
        self.size_candidate_count = 0
        self.rejected_cluster_count = 0

        self.create_subscription(
            String, self.clusters_topic, self._clusters_callback, 10
        )
        self.tracks_publisher = self.create_publisher(
            String, self.tracked_agents_topic, 10
        )
        self.status_publisher = self.create_publisher(
            String, self.status_topic, 10
        )
        self.marker_publisher = self.create_publisher(
            MarkerArray, self.markers_topic, 10
        )
        self.create_timer(1.0 / self.publish_rate_hz, self._publish)
        self.get_logger().info(
            "DREAM vehicle tracker started: clusters=%s tracks=%s "
            "class=%s motion_confirmed_only=true"
            % (
                self.clusters_topic,
                self.tracked_agents_topic,
                self.class_label,
            )
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter("clusters_topic", "/sfg/lidar_clusters")
        self.declare_parameter("tracked_agents_topic", "/tracked_agents")
        self.declare_parameter(
            "markers_topic", "/dream/vehicle_track_markers"
        )
        self.declare_parameter(
            "status_topic", "/dream/vehicle_tracker_status"
        )
        self.declare_parameter("fixed_frame", "odom")
        self.declare_parameter("class_label", "car")
        self.declare_parameter("publish_rate_hz", 10.0)
        self.declare_parameter("input_timeout_sec", 0.50)
        self.declare_parameter("future_stamp_tolerance_sec", 0.05)
        self.declare_parameter("minimum_cluster_width_m", 0.08)
        self.declare_parameter("maximum_cluster_width_m", 0.50)
        self.declare_parameter("minimum_cluster_points", 3)
        self.declare_parameter("minimum_cluster_range_m", 0.25)
        self.declare_parameter("maximum_cluster_range_m", 6.0)
        self.declare_parameter("association_distance_m", 0.45)
        self.declare_parameter("association_noise_margin_m", 0.06)
        self.declare_parameter("maximum_vehicle_speed_mps", 0.60)
        self.declare_parameter("maximum_width_change_m", 0.12)
        self.declare_parameter("velocity_alpha", 0.45)
        self.declare_parameter("position_alpha", 0.70)
        self.declare_parameter("coast_timeout_sec", 0.50)
        self.declare_parameter("stale_remove_sec", 1.00)
        self.declare_parameter("minimum_track_hits", 3)
        self.declare_parameter("motion_window_sec", 0.50)
        self.declare_parameter("motion_enter_speed_mps", 0.10)
        self.declare_parameter("motion_exit_speed_mps", 0.04)
        self.declare_parameter("motion_min_displacement_m", 0.08)
        self.declare_parameter("motion_hold_sec", 0.80)
        self.declare_parameter("minimum_consistent_motion_windows", 2)
        self.declare_parameter("minimum_direction_cosine", 0.50)
        self.declare_parameter("nominal_radius_m", 0.18)
        self.declare_parameter("radius_padding_m", 0.04)

    def _load_parameters(self) -> None:
        self.clusters_topic = self._str_parameter("clusters_topic")
        self.tracked_agents_topic = self._str_parameter("tracked_agents_topic")
        self.markers_topic = self._str_parameter("markers_topic")
        self.status_topic = self._str_parameter("status_topic")
        self.fixed_frame = self._str_parameter("fixed_frame")
        self.class_label = self._str_parameter("class_label")
        self.publish_rate_hz = max(
            1.0, self._float_parameter("publish_rate_hz")
        )
        self.input_timeout_sec = self._float_parameter("input_timeout_sec")
        self.future_stamp_tolerance_sec = self._float_parameter(
            "future_stamp_tolerance_sec"
        )
        self.minimum_cluster_width_m = self._float_parameter(
            "minimum_cluster_width_m"
        )
        self.maximum_cluster_width_m = self._float_parameter(
            "maximum_cluster_width_m"
        )
        self.minimum_cluster_points = self._int_parameter(
            "minimum_cluster_points"
        )
        self.minimum_cluster_range_m = self._float_parameter(
            "minimum_cluster_range_m"
        )
        self.maximum_cluster_range_m = self._float_parameter(
            "maximum_cluster_range_m"
        )
        self.association_distance_m = self._float_parameter(
            "association_distance_m"
        )
        self.association_noise_margin_m = self._float_parameter(
            "association_noise_margin_m"
        )
        self.maximum_vehicle_speed_mps = self._float_parameter(
            "maximum_vehicle_speed_mps"
        )
        self.maximum_width_change_m = self._float_parameter(
            "maximum_width_change_m"
        )
        self.velocity_alpha = self._float_parameter("velocity_alpha")
        self.position_alpha = self._float_parameter("position_alpha")
        self.coast_timeout_sec = self._float_parameter("coast_timeout_sec")
        self.stale_remove_sec = self._float_parameter("stale_remove_sec")
        self.minimum_track_hits = self._int_parameter("minimum_track_hits")
        self.motion_window_sec = self._float_parameter("motion_window_sec")
        self.motion_enter_speed_mps = self._float_parameter(
            "motion_enter_speed_mps"
        )
        self.motion_exit_speed_mps = self._float_parameter(
            "motion_exit_speed_mps"
        )
        self.motion_min_displacement_m = self._float_parameter(
            "motion_min_displacement_m"
        )
        self.motion_hold_sec = self._float_parameter("motion_hold_sec")
        self.minimum_consistent_motion_windows = self._int_parameter(
            "minimum_consistent_motion_windows"
        )
        self.minimum_direction_cosine = self._float_parameter(
            "minimum_direction_cosine"
        )
        self.nominal_radius_m = self._float_parameter("nominal_radius_m")
        self.radius_padding_m = self._float_parameter("radius_padding_m")
        if self.input_timeout_sec <= 0.0:
            raise ValueError("input_timeout_sec must be positive")
        if self.future_stamp_tolerance_sec < 0.0:
            raise ValueError("future_stamp_tolerance_sec cannot be negative")
        if not self.class_label:
            raise ValueError("class_label cannot be empty")

    def _str_parameter(self, name: str) -> str:
        return str(self.get_parameter(name).value)

    def _float_parameter(self, name: str) -> float:
        return float(self.get_parameter(name).value)

    def _int_parameter(self, name: str) -> int:
        return int(self.get_parameter(name).value)

    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1.0e-9

    def _clusters_callback(self, message: String) -> None:
        now = self._now_sec()
        try:
            payload = json.loads(message.data)
            frame = parse_cluster_payload(
                payload,
                expected_frame=self.fixed_frame,
                minimum_width_m=self.minimum_cluster_width_m,
                maximum_width_m=self.maximum_cluster_width_m,
                minimum_points=self.minimum_cluster_points,
                minimum_range_m=self.minimum_cluster_range_m,
                maximum_range_m=self.maximum_cluster_range_m,
            )
            validate_cluster_source_stamp(
                frame.stamp,
                receipt_stamp=now,
                previous_source_stamp=self.last_source_stamp,
                maximum_age=self.input_timeout_sec,
                future_tolerance=self.future_stamp_tolerance_sec,
            )
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            self.last_error = f"invalid_cluster_payload:{exc}"
            self.get_logger().warning(
                f"Rejected neutral LiDAR cluster payload: {exc}",
                throttle_duration_sec=2.0,
            )
            return

        self.tracker.update(frame.clusters, frame.stamp)
        self.last_valid_input_sec = now
        self.last_source_stamp = frame.stamp
        self.last_error = "ok"
        self.raw_cluster_count = frame.raw_count
        self.size_candidate_count = len(frame.clusters)
        self.rejected_cluster_count = frame.rejected_count

    def _publish(self) -> None:
        now = self._now_sec()
        input_age = (
            math.inf
            if self.last_valid_input_sec is None
            else max(0.0, now - self.last_valid_input_sec)
        )
        input_fresh = input_age <= self.input_timeout_sec
        tracks = self.tracker.publishable_tracks(now) if input_fresh else []
        if input_fresh:
            agents = [
                track_to_agent_payload(
                    track,
                    now,
                    class_label=self.class_label,
                    nominal_radius_m=self.nominal_radius_m,
                    radius_padding_m=self.radius_padding_m,
                )
                for track in tracks
            ]
            self.tracks_publisher.publish(
                String(data=json.dumps(agents, separators=(",", ":")))
            )
        self.marker_publisher.publish(
            vehicle_track_markers(
                tracks, self.fixed_frame, self.get_clock().now().to_msg(), now
            )
        )
        status = {
            "ready": input_fresh,
            "input_fresh": input_fresh,
            "input_age": None if math.isinf(input_age) else input_age,
            "source_stamp": self.last_source_stamp,
            "source_age": (
                None
                if self.last_source_stamp is None
                else max(0.0, now - self.last_source_stamp)
            ),
            "input_topic": self.clusters_topic,
            "output_topic": self.tracked_agents_topic,
            "class_label": self.class_label,
            "classification_basis": "controlled_scenario_geometry_and_motion",
            "motion_confirmed_only": True,
            "raw_clusters": self.raw_cluster_count,
            "size_candidates": self.size_candidate_count,
            "rejected_clusters": self.rejected_cluster_count,
            "candidate_tracks": self.tracker.fresh_candidate_count(now),
            "published_tracks": len(tracks),
            "state": (
                self.last_error
                if input_fresh
                else "clusters_missing_or_stale"
            ),
        }
        self.status_publisher.publish(
            String(data=json.dumps(status, separators=(",", ":")))
        )


def vehicle_track_markers(
    tracks: list[VehicleTrack], frame_id: str, stamp, now_sec: float
) -> MarkerArray:
    """Render confirmed merger candidates without pedestrian symbology."""

    markers = MarkerArray()
    clear = Marker()
    clear.action = Marker.DELETEALL
    markers.markers.append(clear)
    for index, track in enumerate(tracks):
        x, y = track.predicted_position(now_sec)
        heading = math.atan2(track.vy, track.vx)
        body = Marker()
        body.header.frame_id = frame_id
        body.header.stamp = stamp
        body.ns = "dream_vehicle_tracks"
        body.id = index
        body.type = Marker.CUBE
        body.action = Marker.ADD
        body.pose.position.x = float(x)
        body.pose.position.y = float(y)
        body.pose.position.z = 0.08
        body.pose.orientation.z = math.sin(0.5 * heading)
        body.pose.orientation.w = math.cos(0.5 * heading)
        body.scale.x = max(0.32, track.width)
        body.scale.y = max(0.22, track.width)
        body.scale.z = 0.16
        body.color.r = 0.05
        body.color.g = 0.85
        body.color.b = 1.0
        body.color.a = 0.80
        body.lifetime.sec = 1
        markers.markers.append(body)

        label = Marker()
        label.header.frame_id = frame_id
        label.header.stamp = stamp
        label.ns = "dream_vehicle_track_labels"
        label.id = 10000 + index
        label.type = Marker.TEXT_VIEW_FACING
        label.action = Marker.ADD
        label.pose.position.x = float(x)
        label.pose.position.y = float(y)
        label.pose.position.z = 0.32
        label.pose.orientation.w = 1.0
        label.scale.z = 0.13
        label.color.r = 0.05
        label.color.g = 0.85
        label.color.b = 1.0
        label.color.a = 1.0
        label.text = f"MOVING VEHICLE CANDIDATE {track.speed:.2f} m/s"
        label.lifetime.sec = 1
        markers.markers.append(label)
    return markers


def main(args=None) -> None:
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = None
    executor = SingleThreadedExecutor()
    stop_requested = False

    def request_stop(signum, frame):
        del signum, frame
        nonlocal stop_requested
        stop_requested = True

    old_sigint = signal.getsignal(signal.SIGINT)
    old_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        node = DreamVehicleTrackerNode()
        executor.add_node(node)
        while rclpy.ok() and not stop_requested:
            executor.spin_once(timeout_sec=0.1)
    finally:
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)
        if node is not None:
            try:
                executor.remove_node(node)
                node.destroy_node()
            except KeyboardInterrupt:
                pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
