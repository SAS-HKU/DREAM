"""
Fail-closed Nav2 geometric path provider for the DREAM controller.

The node deliberately starts no controller and publishes no velocity. It turns
an already-authorized DREAM navigation goal into a periodically refreshed
``nav_msgs/Path`` by calling Nav2's planner-server action.
"""

from __future__ import annotations

import copy
import json
from functools import partial
from math import isfinite
from typing import Optional

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import ComputePathToPose
from nav_msgs.msg import Path
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener

from .core.nav2_route import (
    path_message_values,
    validate_freshness,
    validate_geometric_path,
    validate_planar_pose,
    validate_transform_sample,
)
from .ros_utils import quaternion_to_yaw, stamp_to_seconds


class DreamNav2PathProvider(Node):
    """Request and latch fresh, obstacle-aware geometric paths from Nav2."""

    def __init__(self) -> None:
        super().__init__("dream_nav2_path_provider")
        self._declare_parameters()
        self.goal_topic = self._string_parameter("goal_topic")
        self.path_topic = self._string_parameter("path_topic")
        self.status_topic = self._string_parameter("status_topic")
        self.action_name = self._string_parameter("action_name")
        self.map_frame = self._string_parameter("map_frame")
        self.base_frame = self._string_parameter("base_frame")
        self.planner_id = self._string_parameter("planner_id")
        self.replan_period = self._positive_parameter("replan_period")
        self.failure_retry_period = self._positive_parameter(
            "failure_retry_period"
        )
        self.request_timeout = self._positive_parameter("request_timeout")
        self.path_timeout = self._positive_parameter("path_timeout")
        self.tf_maximum_age = self._positive_parameter("tf_maximum_age")
        self.tf_lookup_timeout = self._positive_parameter("tf_lookup_timeout")
        self.goal_timeout = self._positive_parameter("goal_timeout")
        self.future_tolerance = self._nonnegative_parameter("future_tolerance")
        status_rate = self._positive_parameter("status_rate")

        latched = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.path_publisher = self.create_publisher(Path, self.path_topic, latched)
        self.status_publisher = self.create_publisher(
            String,
            self.status_topic,
            latched,
        )
        self.create_subscription(
            PoseStamped,
            self.goal_topic,
            self._on_goal,
            latched,
        )
        self.action_client = ActionClient(
            self,
            ComputePathToPose,
            self.action_name,
        )
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.goal: Optional[PoseStamped] = None
        self.goal_generation = 0
        self.request_serial = 0
        self.active_request: Optional[int] = None
        self.goal_handle = None
        self.request_started: Optional[float] = None
        self.last_request_started: Optional[float] = None
        self.last_path_receipt: Optional[float] = None
        self.last_path_source_stamp: Optional[float] = None
        self.path_pose_count = 0
        self.path_valid = False
        self.last_tf_age: Optional[float] = None
        self.server_ready = False
        self.state = "WAITING_FOR_GOAL"
        self.reason = "WAITING_FOR_GOAL"
        self._empty_path_published = False

        self._invalidate_path("WAITING_FOR_GOAL", force_publish=True)
        self.create_timer(1.0 / status_rate, self._tick)

    def _declare_parameters(self) -> None:
        self.declare_parameter("goal_topic", "/dream/navigation_goal")
        self.declare_parameter("path_topic", "/dream/geometric_path")
        self.declare_parameter("status_topic", "/dream/route_status")
        self.declare_parameter("action_name", "/compute_path_to_pose")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("planner_id", "GridBased")
        self.declare_parameter("replan_period", 0.50)
        self.declare_parameter("failure_retry_period", 2.00)
        self.declare_parameter("request_timeout", 1.00)
        self.declare_parameter("path_timeout", 1.50)
        self.declare_parameter("tf_maximum_age", 0.50)
        self.declare_parameter("tf_lookup_timeout", 0.10)
        self.declare_parameter("goal_timeout", 1.00)
        self.declare_parameter("future_tolerance", 0.05)
        self.declare_parameter("status_rate", 10.0)

    def _string_parameter(self, name: str) -> str:
        value = str(self.get_parameter(name).value)
        if not value:
            raise RuntimeError(f"{name} cannot be empty")
        return value

    def _positive_parameter(self, name: str) -> float:
        value = float(self.get_parameter(name).value)
        if not isfinite(value) or value <= 0.0:
            raise RuntimeError(f"{name} must be finite and positive")
        return value

    def _nonnegative_parameter(self, name: str) -> float:
        value = float(self.get_parameter(name).value)
        if not isfinite(value) or value < 0.0:
            raise RuntimeError(f"{name} must be finite and nonnegative")
        return value

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1.0e-9

    def _cancel_request(self) -> None:
        handle = self.goal_handle
        self.goal_handle = None
        self.active_request = None
        self.request_started = None
        if handle is not None:
            try:
                handle.cancel_goal_async()
            except Exception as exc:  # rclpy action failures share no narrow base.
                self.get_logger().warning(f"Nav2 cancellation failed: {exc}")

    def _empty_path(self) -> Path:
        message = Path()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self.map_frame
        return message

    def _invalidate_path(self, reason: str, *, force_publish: bool = False) -> None:
        if self.path_valid or force_publish or not self._empty_path_published:
            self.path_publisher.publish(self._empty_path())
            self._empty_path_published = True
        self.path_valid = False
        self.path_pose_count = 0
        self.last_path_receipt = None
        self.last_path_source_stamp = None
        self.reason = reason

    def _on_goal(self, message: PoseStamped) -> None:
        now = self._now()
        # Every new message replaces the preceding mission, even if malformed.
        # An invalid replacement therefore cannot leave an old route active.
        self.goal_generation += 1
        self._cancel_request()
        self.goal = None
        self.last_request_started = None
        self._invalidate_path("VALIDATING_GOAL", force_publish=True)

        pose = message.pose
        reason = validate_planar_pose(
            frame_id=str(message.header.frame_id),
            expected_frame=self.map_frame,
            position_xyz=(
                pose.position.x,
                pose.position.y,
                pose.position.z,
            ),
            quaternion_xyzw=(
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ),
            label="GOAL",
        )
        if reason == "ok":
            freshness = validate_freshness(
                stamp_to_seconds(message.header.stamp),
                now=now,
                maximum_age=self.goal_timeout,
                future_tolerance=self.future_tolerance,
                label="GOAL_SOURCE",
            )
            reason = "ok" if freshness.valid else freshness.reason
        if reason != "ok":
            self.state = "GOAL_REJECTED"
            self._invalidate_path(reason, force_publish=True)
            self.get_logger().warning(f"DREAM navigation goal rejected: {reason}")
            self._publish_status(now)
            return

        self.goal = copy.deepcopy(message)
        self.state = "WAITING_FOR_PLANNER"
        self.reason = "GOAL_ACCEPTED"
        self._attempt_plan(now)
        self._publish_status(now)

    def _lookup_start(self, now: float) -> PoseStamped:
        transform = self.tf_buffer.lookup_transform(
            self.map_frame,
            self.base_frame,
            Time(),
            timeout=Duration(seconds=self.tf_lookup_timeout),
        )
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        source_stamp = stamp_to_seconds(transform.header.stamp)
        validation = validate_transform_sample(
            parent_frame=str(transform.header.frame_id),
            child_frame=str(transform.child_frame_id),
            expected_parent=self.map_frame,
            expected_child=self.base_frame,
            translation_xyz=(translation.x, translation.y, translation.z),
            quaternion_xyzw=(rotation.x, rotation.y, rotation.z, rotation.w),
            source_stamp=source_stamp,
            now=now,
            maximum_age=self.tf_maximum_age,
            future_tolerance=self.future_tolerance,
        )
        self.last_tf_age = validation.age
        if not validation.valid:
            raise ValueError(validation.reason)
        start = PoseStamped()
        start.header = transform.header
        start.pose.position.x = float(translation.x)
        start.pose.position.y = float(translation.y)
        start.pose.position.z = float(translation.z)
        start.pose.orientation = rotation
        return start

    def _attempt_plan(self, now: float) -> None:
        if self.goal is None or self.active_request is not None:
            return
        self.server_ready = bool(self.action_client.server_is_ready())
        if not self.server_ready:
            self.state = "WAITING_FOR_PLANNER"
            self._invalidate_path("PLANNER_SERVER_UNAVAILABLE")
            return
        try:
            start = self._lookup_start(now)
        except (TransformException, ValueError) as exc:
            self.state = "WAITING_FOR_TF"
            self._invalidate_path(f"TF_UNAVAILABLE:{exc}")
            return

        request = ComputePathToPose.Goal()
        request.goal = copy.deepcopy(self.goal)
        request.start = start
        request.planner_id = self.planner_id
        request.use_start = True
        self.request_serial += 1
        request_id = self.request_serial
        generation = self.goal_generation
        self.active_request = request_id
        self.request_started = now
        self.last_request_started = now
        self.state = "REQUESTING_PATH" if not self.path_valid else "REPLANNING"
        self.reason = "PATH_REQUEST_ACTIVE"
        try:
            future = self.action_client.send_goal_async(request)
            future.add_done_callback(
                partial(
                    self._on_goal_response,
                    generation=generation,
                    request_id=request_id,
                )
            )
        except Exception as exc:
            self.active_request = None
            self.request_started = None
            self.state = "PLANNER_ERROR"
            self._invalidate_path(f"PATH_REQUEST_ERROR:{exc}")

    def _request_is_current(self, generation: int, request_id: int) -> bool:
        return bool(
            generation == self.goal_generation
            and request_id == self.active_request
        )

    def _on_goal_response(self, future, *, generation: int, request_id: int) -> None:
        try:
            handle = future.result()
        except Exception as exc:
            if self._request_is_current(generation, request_id):
                self.active_request = None
                self.request_started = None
                self.state = "PLANNER_ERROR"
                self._invalidate_path(f"PATH_GOAL_RESPONSE_ERROR:{exc}")
            return
        if not self._request_is_current(generation, request_id):
            if handle is not None and handle.accepted:
                handle.cancel_goal_async()
            return
        if handle is None or not handle.accepted:
            self.active_request = None
            self.request_started = None
            self.state = "PLANNER_ERROR"
            self._invalidate_path("PATH_REQUEST_REJECTED")
            return
        self.goal_handle = handle
        self.state = "PLANNING" if not self.path_valid else "REPLANNING"
        result_future = handle.get_result_async()
        result_future.add_done_callback(
            partial(
                self._on_path_result,
                generation=generation,
                request_id=request_id,
            )
        )

    def _on_path_result(self, future, *, generation: int, request_id: int) -> None:
        if not self._request_is_current(generation, request_id):
            return
        self.active_request = None
        self.goal_handle = None
        self.request_started = None
        now = self._now()
        try:
            response = future.result()
        except Exception as exc:
            self.state = "PLANNER_ERROR"
            self._invalidate_path(f"PATH_RESULT_ERROR:{exc}")
            return
        if response is None or response.status != GoalStatus.STATUS_SUCCEEDED:
            status = None if response is None else int(response.status)
            self.state = "PLANNER_ERROR"
            self._invalidate_path(f"PATH_RESULT_STATUS:{status}")
            return
        path = response.result.path
        receipt_stamp = now
        frames, positions, quaternions = path_message_values(path)
        validation = validate_geometric_path(
            frame_id=str(path.header.frame_id),
            pose_frames=frames,
            positions_xyz=positions,
            quaternions_xyzw=quaternions,
            source_stamp=stamp_to_seconds(path.header.stamp),
            receipt_stamp=receipt_stamp,
            now=now,
            expected_frame=self.map_frame,
            source_timeout=self.path_timeout,
            receipt_timeout=self.path_timeout,
            future_tolerance=self.future_tolerance,
        )
        if not validation.valid:
            self.state = "PLANNER_ERROR"
            self._invalidate_path(validation.reason)
            return
        self.path_publisher.publish(path)
        self._empty_path_published = False
        self.path_valid = True
        self.path_pose_count = validation.pose_count
        self.last_path_receipt = receipt_stamp
        self.last_path_source_stamp = stamp_to_seconds(path.header.stamp)
        self.state = "READY"
        self.reason = "PATH_READY"

    def _watchdog(self, now: float) -> None:
        if self.goal is None:
            return
        self.server_ready = bool(self.action_client.server_is_ready())
        if not self.server_ready:
            self._cancel_request()
            self.state = "WAITING_FOR_PLANNER"
            self._invalidate_path("PLANNER_SERVER_UNAVAILABLE")
            return
        try:
            self._lookup_start(now)
        except (TransformException, ValueError) as exc:
            self._cancel_request()
            self.state = "WAITING_FOR_TF"
            self._invalidate_path(f"TF_UNAVAILABLE:{exc}")
            return
        if (
            self.active_request is not None
            and self.request_started is not None
            and now - self.request_started >= self.request_timeout
        ):
            self._cancel_request()
            self.last_request_started = now
            self.state = "PLANNER_ERROR"
            self._invalidate_path("PATH_REQUEST_TIMEOUT")
            return
        if self.path_valid:
            source = validate_freshness(
                float(self.last_path_source_stamp or 0.0),
                now=now,
                maximum_age=self.path_timeout,
                future_tolerance=self.future_tolerance,
                label="PATH_SOURCE",
            )
            receipt = validate_freshness(
                float(self.last_path_receipt or 0.0),
                now=now,
                maximum_age=self.path_timeout,
                future_tolerance=self.future_tolerance,
                label="PATH_RECEIPT",
            )
            if not source.valid or not receipt.valid:
                self.state = "PATH_STALE"
                self._invalidate_path(
                    source.reason if not source.valid else receipt.reason
                )

    def _tick(self) -> None:
        now = self._now()
        self._watchdog(now)
        if self.goal is not None and self.active_request is None:
            retry_period = (
                self.replan_period if self.path_valid else self.failure_retry_period
            )
            due = (
                self.last_request_started is None
                or now - self.last_request_started >= retry_period
            )
            if due:
                self._attempt_plan(now)
        self._publish_status(now)

    @staticmethod
    def _age(now: float, stamp: Optional[float]) -> Optional[float]:
        return None if stamp is None else max(0.0, now - stamp)

    def _publish_status(self, now: float) -> None:
        request_age = self._age(now, self.request_started)
        path_source_age = self._age(now, self.last_path_source_stamp)
        path_receipt_age = self._age(now, self.last_path_receipt)
        goal_payload = None
        if self.goal is not None:
            goal_payload = {
                "frame_id": self.goal.header.frame_id,
                "x": float(self.goal.pose.position.x),
                "y": float(self.goal.pose.position.y),
                "yaw": quaternion_to_yaw(self.goal.pose.orientation),
                "stamp": stamp_to_seconds(self.goal.header.stamp),
            }
        payload = {
            "stamp": now,
            "ready": bool(self.path_valid),
            "state": self.state,
            "reason": self.reason,
            "goal_received": self.goal is not None,
            "goal_generation": self.goal_generation,
            "goal": goal_payload,
            "goal_x": None if goal_payload is None else goal_payload["x"],
            "goal_y": None if goal_payload is None else goal_payload["y"],
            "goal_yaw": None if goal_payload is None else goal_payload["yaw"],
            "goal_stamp": None if goal_payload is None else goal_payload["stamp"],
            "map_frame": self.map_frame,
            "base_frame": self.base_frame,
            "action_name": self.action_name,
            "planner_id": self.planner_id,
            "planner_server_ready": self.server_ready,
            "request_active": self.active_request is not None,
            "request_age": request_age,
            "request_timeout": self.request_timeout,
            "replan_period": self.replan_period,
            "failure_retry_period": self.failure_retry_period,
            "path_pose_count": self.path_pose_count,
            "path_source_stamp": self.last_path_source_stamp,
            "path_source_age": path_source_age,
            "path_receipt_age": path_receipt_age,
            "path_timeout": self.path_timeout,
            "tf_age": self.last_tf_age,
            "note": "geometric path only; DREAM and the final hardware gate own motion",
        }
        message = String()
        message.data = json.dumps(payload, separators=(",", ":"), allow_nan=False)
        self.status_publisher.publish(message)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DreamNav2PathProvider()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._cancel_request()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
