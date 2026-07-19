"""Front-camera evidence relay; deliberately excluded from planner inputs."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

import cv2
from cv_bridge import CvBridge, CvBridgeError
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String


class DreamCameraEvidenceNode(Node):
    """Publish an annotated operator view and an unmodified evidence stream."""

    def __init__(self) -> None:
        super().__init__("dream_camera_evidence")
        self.declare_parameter("source_topic", "/camera/color/image_raw")
        self.declare_parameter("publish_rate", 5.0)
        self.declare_parameter("source_timeout", 0.35)
        self.declare_parameter("minimum_width", 320)
        self.declare_parameter("minimum_height", 200)

        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=3,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.bridge = CvBridge()
        self.latest_image: Optional[Image] = None
        self.latest_receipt: Optional[float] = None
        self.frame_count = 0
        self.first_receipt: Optional[float] = None
        self.planner: Dict[str, Any] = {}
        self.safety: Dict[str, Any] = {}
        self.world: Dict[str, Any] = {}
        self.merger_visible = False

        self.create_subscription(
            Image,
            str(self.get_parameter("source_topic").value),
            self._on_image,
            sensor_qos,
        )
        self.create_subscription(String, "/dream/planner_status", self._on_planner, 10)
        self.create_subscription(String, "/dream/safety_status", self._on_safety, 10)
        self.create_subscription(String, "/dream/world_status", self._on_world, 10)
        self.create_subscription(Bool, "/dream/merger_visible", self._on_visibility, 10)
        self.annotated_publisher = self.create_publisher(
            Image, "/dream/driver_view", sensor_qos
        )
        self.raw_publisher = self.create_publisher(
            Image, "/dream/camera_evidence_raw", sensor_qos
        )
        self.status_publisher = self.create_publisher(
            String, "/dream/camera_evidence_status", 10
        )
        self.create_timer(
            1.0 / float(self.get_parameter("publish_rate").value), self._publish
        )

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1.0e-9

    @staticmethod
    def _decode(message: String) -> Dict[str, Any]:
        try:
            payload = json.loads(message.data)
            return payload if isinstance(payload, dict) else {}
        except (json.JSONDecodeError, TypeError):
            return {}

    def _on_image(self, message: Image) -> None:
        now = self._now()
        self.latest_image = message
        self.latest_receipt = now
        self.first_receipt = now if self.first_receipt is None else self.first_receipt
        self.frame_count += 1

    def _on_planner(self, message: String) -> None:
        self.planner = self._decode(message)

    def _on_safety(self, message: String) -> None:
        self.safety = self._decode(message)

    def _on_world(self, message: String) -> None:
        self.world = self._decode(message)

    def _on_visibility(self, message: Bool) -> None:
        self.merger_visible = bool(message.data)

    def _status(self, ready: bool, reason: str, age: Optional[float]) -> Dict[str, Any]:
        now = self._now()
        duration = None if self.first_receipt is None else now - self.first_receipt
        measured_rate = (
            None
            if duration is None or duration <= 0.0
            else max(0, self.frame_count - 1) / duration
        )
        image = self.latest_image
        return {
            "ready": ready,
            "reason": reason,
            "source_topic": str(self.get_parameter("source_topic").value),
            "source_age": age,
            "source_rate_hz": measured_rate,
            "width": None if image is None else int(image.width),
            "height": None if image is None else int(image.height),
            "encoding": None if image is None else image.encoding,
            "source_frame": None if image is None else image.header.frame_id,
            "control_stack": self.planner.get("control_stack"),
            "evidence_only": True,
            "planner_input": False,
        }

    def _publish_status(self, payload: Dict[str, Any]) -> None:
        message = String()
        message.data = json.dumps(payload, separators=(",", ":"), allow_nan=False)
        self.status_publisher.publish(message)

    @staticmethod
    def _visibility_label(
        *, merger_visible: bool, track_count: int, shadow_cells: int
    ) -> str:
        if merger_visible:
            return "REVEALED / ODOM GATE"
        if track_count > 0:
            return f"TRACK OBSERVED ({track_count})"
        if shadow_cells > 0:
            return "SHADOW PRESENT / NO TRACK"
        return "NO TRACK / NO SHADOW"

    def _annotate(self, message: Image) -> Image:
        frame = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8").copy()
        height, width = frame.shape[:2]
        strip_height = min(92, max(62, height // 5))
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, strip_height), (0, 0, 0), thickness=-1)
        frame = cv2.addWeighted(overlay, 0.72, frame, 0.28, 0.0)
        stack = str(self.planner.get("control_stack", "WAITING")).upper()
        track_count = int(self.world.get("dynamic_track_count", 0))
        shadow_cells = int(self.world.get("shadow_cells", 0))
        visibility = self._visibility_label(
            merger_visible=self.merger_visible,
            track_count=track_count,
            shadow_cells=shadow_cells,
        )
        veto = "VETO" if bool(self.planner.get("vetoed", False)) else "NO VETO"
        risk = float(self.planner.get("decision_risk", 0.0))
        speed = float(self.planner.get("target_speed", 0.0))
        safety = str(self.safety.get("reason", "WAITING"))
        lines = (
            "EVIDENCE ONLY - NOT A PLANNER INPUT",
            f"{stack} | {visibility} | {veto} | risk={risk:.2f} | v={speed:.2f} m/s",
            f"safety={safety} | lidar shadow cells={shadow_cells}",
        )
        scale = max(0.42, min(0.62, width / 1100.0))
        for index, text in enumerate(lines):
            color = (0, 255, 255) if index == 0 else (255, 255, 255)
            cv2.putText(
                frame,
                text,
                (10, 22 + 27 * index),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                color,
                1,
                cv2.LINE_AA,
            )
        output = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        output.header = message.header
        return output

    def _publish(self) -> None:
        now = self._now()
        timeout = float(self.get_parameter("source_timeout").value)
        if self.latest_image is None or self.latest_receipt is None:
            self._publish_status(self._status(False, "NO_CAMERA_FRAME", None))
            return
        age = now - self.latest_receipt
        if age < 0.0 or age >= timeout:
            self._publish_status(self._status(False, "STALE_CAMERA_FRAME", age))
            return
        image = self.latest_image
        if (
            image.width < int(self.get_parameter("minimum_width").value)
            or image.height < int(self.get_parameter("minimum_height").value)
        ):
            self._publish_status(self._status(False, "CAMERA_RESOLUTION_TOO_SMALL", age))
            return
        try:
            annotated = self._annotate(image)
        except (CvBridgeError, ValueError, TypeError) as exc:
            self._publish_status(self._status(False, f"IMAGE_DECODE_ERROR:{exc}", age))
            return
        # The raw publication is byte-for-byte the received image message.
        self.raw_publisher.publish(image)
        self.annotated_publisher.publish(annotated)
        self._publish_status(self._status(True, "OK", age))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DreamCameraEvidenceNode()
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
