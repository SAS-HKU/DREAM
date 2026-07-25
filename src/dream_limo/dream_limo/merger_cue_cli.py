"""Publish a reproducible merger-release cue after a newly accepted goal.

This utility does not command either robot.  It emits one ``std_msgs/Empty``
event for a separately safety-reviewed merger runner or a human smoke-test cue.
Empty goal invalidations and transient-local goals retained from before this
process started are ignored.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from math import isfinite
from typing import Optional, Sequence

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from std_msgs.msg import Empty

from .ros_utils import stamp_to_seconds


def is_new_accepted_goal(
    message: PoseStamped,
    *,
    process_start_stamp: float,
    expected_frame: str = "map",
) -> bool:
    """Return true only for a valid accepted-goal publication after startup."""

    source_stamp = stamp_to_seconds(message.header.stamp)
    return bool(
        message.header.frame_id == expected_frame
        and isfinite(source_stamp)
        and source_stamp > float(process_start_stamp)
    )


@dataclass(frozen=True)
class CueScheduleUpdate:
    accepted_goal_stamp: Optional[float]
    release_at: Optional[float]
    changed: bool
    reason: str


def update_cue_schedule(
    message: PoseStamped,
    *,
    process_start_stamp: float,
    expected_frame: str,
    delay: float,
    current_accepted_goal_stamp: Optional[float],
    current_release_at: Optional[float],
) -> CueScheduleUpdate:
    """Apply one accepted-goal or invalidation event to a cue schedule."""

    if not isfinite(delay) or delay < 0.0:
        raise ValueError("cue delay must be finite and nonnegative")
    if not message.header.frame_id:
        return CueScheduleUpdate(
            accepted_goal_stamp=None,
            release_at=None,
            changed=current_release_at is not None,
            reason="INVALIDATED",
        )
    if not is_new_accepted_goal(
        message,
        process_start_stamp=process_start_stamp,
        expected_frame=expected_frame,
    ):
        return CueScheduleUpdate(
            accepted_goal_stamp=current_accepted_goal_stamp,
            release_at=current_release_at,
            changed=False,
            reason="IGNORED",
        )
    accepted_stamp = stamp_to_seconds(message.header.stamp)
    if (
        current_accepted_goal_stamp is not None
        and accepted_stamp <= current_accepted_goal_stamp
    ):
        return CueScheduleUpdate(
            accepted_goal_stamp=current_accepted_goal_stamp,
            release_at=current_release_at,
            changed=False,
            reason="NOT_NEWER",
        )
    return CueScheduleUpdate(
        accepted_goal_stamp=accepted_stamp,
        release_at=accepted_stamp + delay,
        changed=True,
        reason=(
            "REPLACED"
            if current_accepted_goal_stamp is not None
            else "ACCEPTED"
        ),
    )


class MergerCueNode(Node):
    def __init__(
        self,
        *,
        delay: float,
        goal_topic: str,
        cue_topic: str,
        expected_frame: str,
    ) -> None:
        super().__init__("dream_merger_cue")
        self.delay = float(delay)
        self.expected_frame = str(expected_frame)
        self.process_start_stamp = self._now()
        self.accepted_goal_stamp: Optional[float] = None
        self.release_at: Optional[float] = None
        self.released = False
        latched = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        reliable = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.publisher = self.create_publisher(Empty, cue_topic, reliable)
        self.create_subscription(
            PoseStamped,
            goal_topic,
            self._on_goal,
            latched,
        )
        self.create_timer(0.01, self._tick)

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1.0e-9

    def _on_goal(self, message: PoseStamped) -> None:
        if self.released:
            return
        update = update_cue_schedule(
            message,
            process_start_stamp=self.process_start_stamp,
            expected_frame=self.expected_frame,
            delay=self.delay,
            current_accepted_goal_stamp=self.accepted_goal_stamp,
            current_release_at=self.release_at,
        )
        if not update.changed:
            return
        if update.reason == "INVALIDATED":
            if self.release_at is not None:
                self.get_logger().warning(
                    "Accepted goal was invalidated; merger cue cancelled"
                )
        self.accepted_goal_stamp = update.accepted_goal_stamp
        self.release_at = update.release_at
        if update.reason == "INVALIDATED":
            return
        action = "replaced" if update.reason == "REPLACED" else "scheduled"
        self.get_logger().info(
            f"Accepted goal; merger cue {action} for "
            f"{self.delay:.3f} s after its publication stamp"
        )

    def _tick(self) -> None:
        if (
            self.released
            or self.release_at is None
            or self._now() < self.release_at
        ):
            return
        self.publisher.publish(Empty())
        self.released = True
        print(
            json.dumps(
                {
                    "event": "dream_merger_release",
                    "accepted_goal_publication_stamp": (
                        self.accepted_goal_stamp
                    ),
                    "release_stamp": self._now(),
                    "configured_delay_seconds": self.delay,
                    "note": (
                        "cue only; external merger actuation is not part of "
                        "dream_limo"
                    ),
                },
                separators=(",", ":"),
            ),
            flush=True,
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delay", type=float, default=2.0)
    parser.add_argument(
        "--goal-topic", default="/dream/navigation_goal"
    )
    parser.add_argument(
        "--cue-topic", default="/dream/merger_release"
    )
    parser.add_argument("--expected-frame", default="map")
    arguments, ros_arguments = parser.parse_known_args(argv)
    if not isfinite(arguments.delay) or arguments.delay < 0.0:
        parser.error("--delay must be finite and nonnegative")
    if not arguments.goal_topic or not arguments.cue_topic:
        parser.error("goal and cue topics must be nonempty")
    if not arguments.expected_frame:
        parser.error("--expected-frame must be nonempty")

    rclpy.init(args=ros_arguments)
    node = MergerCueNode(
        delay=arguments.delay,
        goal_topic=arguments.goal_topic,
        cue_topic=arguments.cue_topic,
        expected_frame=arguments.expected_frame,
    )
    try:
        while rclpy.ok() and not node.released:
            rclpy.spin_once(node, timeout_sec=0.1)
        # Give the reliable publisher one executor opportunity before teardown.
        if rclpy.ok() and node.released:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        return 130
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
