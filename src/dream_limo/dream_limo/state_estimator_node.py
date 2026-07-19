"""Arena-frame state adapter for LIMO wheel odometry and IMU."""

from __future__ import annotations

import json
from math import cos, sin

import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu
from std_msgs.msg import String
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

from .ros_utils import alignment_from_initial_pose, quaternion_to_yaw, yaw_to_quaternion


class DreamStateEstimator(Node):
    def __init__(self) -> None:
        super().__init__("dream_state_estimator")
        self.declare_parameter("odom_topic", "/wheel/odom")
        self.declare_parameter("imu_topic", "/imu")
        self.declare_parameter("output_topic", "/dream/ego_state")
        self.declare_parameter("arena_frame", "map")
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("map_to_odom_x", 0.0)
        self.declare_parameter("map_to_odom_y", 0.0)
        self.declare_parameter("map_to_odom_yaw", 0.0)
        self.declare_parameter("initialize_from_first_odom", False)
        self.declare_parameter("initial_map_x", 0.35)
        self.declare_parameter("initial_map_y", 0.45)
        self.declare_parameter("initial_map_yaw", 0.0)
        self.declare_parameter("alignment_topic", "/dream/map_alignment")
        self.declare_parameter("publish_map_to_odom_tf", True)

        reliable = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=20,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.publisher = self.create_publisher(
            Odometry, str(self.get_parameter("output_topic").value), reliable
        )
        alignment_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.alignment_publisher = self.create_publisher(
            TransformStamped,
            str(self.get_parameter("alignment_topic").value),
            alignment_qos,
        )
        self.status_publisher = self.create_publisher(String, "/dream/state_status", 10)
        self.create_subscription(
            Odometry, str(self.get_parameter("odom_topic").value), self._on_odom, reliable
        )
        self.create_subscription(
            Imu, str(self.get_parameter("imu_topic").value), self._on_imu, reliable
        )
        self.latest_imu_yaw_rate = None
        self.latest_imu_stamp = None
        self.tf_broadcaster = StaticTransformBroadcaster(self)
        self.alignment = None
        if not bool(self.get_parameter("initialize_from_first_odom").value):
            self.alignment = (
                float(self.get_parameter("map_to_odom_x").value),
                float(self.get_parameter("map_to_odom_y").value),
                float(self.get_parameter("map_to_odom_yaw").value),
            )
            self._publish_alignment()

    def _publish_alignment(self) -> None:
        if self.alignment is None:
            return
        tx, ty, yaw = self.alignment
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = str(self.get_parameter("arena_frame").value)
        transform.child_frame_id = str(self.get_parameter("odom_frame").value)
        transform.transform.translation.x = tx
        transform.transform.translation.y = ty
        qx, qy, qz, qw = yaw_to_quaternion(yaw)
        transform.transform.rotation.x = qx
        transform.transform.rotation.y = qy
        transform.transform.rotation.z = qz
        transform.transform.rotation.w = qw
        self.alignment_publisher.publish(transform)
        if bool(self.get_parameter("publish_map_to_odom_tf").value):
            self.tf_broadcaster.sendTransform(transform)

    def _on_imu(self, message: Imu) -> None:
        self.latest_imu_yaw_rate = float(message.angular_velocity.z)
        self.latest_imu_stamp = self.get_clock().now().nanoseconds * 1.0e-9

    def _on_odom(self, message: Odometry) -> None:
        source = message.pose.pose.position
        odom_yaw = quaternion_to_yaw(message.pose.pose.orientation)
        if self.alignment is None:
            self.alignment = alignment_from_initial_pose(
                source.x,
                source.y,
                odom_yaw,
                target_x=float(self.get_parameter("initial_map_x").value),
                target_y=float(self.get_parameter("initial_map_y").value),
                target_yaw=float(self.get_parameter("initial_map_yaw").value),
            )
            tx, ty, map_yaw = self.alignment
            self._publish_alignment()
            self.get_logger().info(
                "Initialized local experiment frame from first odometry pose: "
                f"tx={tx:.3f} ty={ty:.3f} yaw={map_yaw:.3f}"
            )
        tx, ty, map_yaw = self.alignment
        ch, sh = cos(map_yaw), sin(map_yaw)
        output = Odometry()
        output.header.stamp = message.header.stamp
        output.header.frame_id = str(self.get_parameter("arena_frame").value)
        output.child_frame_id = str(self.get_parameter("base_frame").value)
        output.pose.pose.position.x = tx + ch * source.x - sh * source.y
        output.pose.pose.position.y = ty + sh * source.x + ch * source.y
        output.pose.pose.position.z = source.z
        yaw = map_yaw + odom_yaw
        qx, qy, qz, qw = yaw_to_quaternion(yaw)
        output.pose.pose.orientation.x = qx
        output.pose.pose.orientation.y = qy
        output.pose.pose.orientation.z = qz
        output.pose.pose.orientation.w = qw
        output.pose.covariance = message.pose.covariance
        output.twist = message.twist
        now = self.get_clock().now().nanoseconds * 1.0e-9
        if self.latest_imu_yaw_rate is not None and now - self.latest_imu_stamp < 0.25:
            output.twist.twist.angular.z = self.latest_imu_yaw_rate
        self.publisher.publish(output)
        status = String()
        status.data = json.dumps(
            {
                "ready": True,
                "source_frame": message.header.frame_id,
                "output_frame": output.header.frame_id,
                "imu_fresh": self.latest_imu_stamp is not None and now - self.latest_imu_stamp < 0.25,
                "alignment_source": (
                    "first_odom"
                    if bool(self.get_parameter("initialize_from_first_odom").value)
                    else "configured"
                ),
                "map_to_odom": {"x": tx, "y": ty, "yaw": map_yaw},
            },
            separators=(",", ":"),
        )
        self.status_publisher.publish(status)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DreamStateEstimator()
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
