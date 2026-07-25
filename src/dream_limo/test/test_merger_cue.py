from geometry_msgs.msg import PoseStamped

import pytest

from dream_limo.merger_cue_cli import (
    is_new_accepted_goal,
    update_cue_schedule,
)


def _goal(*, frame="map", stamp=10.0):
    message = PoseStamped()
    message.header.frame_id = frame
    message.header.stamp.sec = int(stamp)
    message.header.stamp.nanosec = int((stamp - int(stamp)) * 1.0e9)
    return message


def test_merger_cue_accepts_only_a_new_nonempty_accepted_goal():
    assert is_new_accepted_goal(
        _goal(stamp=10.1),
        process_start_stamp=10.0,
    )
    assert not is_new_accepted_goal(
        _goal(frame="", stamp=10.1),
        process_start_stamp=10.0,
    )
    assert not is_new_accepted_goal(
        _goal(stamp=9.9),
        process_start_stamp=10.0,
    )
    assert not is_new_accepted_goal(
        _goal(stamp=10.0),
        process_start_stamp=10.0,
    )


def test_merger_cue_schedule_uses_publication_stamp_and_replaces_newer_goal():
    first = update_cue_schedule(
        _goal(stamp=10.1),
        process_start_stamp=10.0,
        expected_frame="map",
        delay=2.0,
        current_accepted_goal_stamp=None,
        current_release_at=None,
    )
    assert first.changed
    assert first.reason == "ACCEPTED"
    assert first.accepted_goal_stamp == pytest.approx(10.1)
    assert first.release_at == pytest.approx(12.1)

    replacement = update_cue_schedule(
        _goal(stamp=10.5),
        process_start_stamp=10.0,
        expected_frame="map",
        delay=2.0,
        current_accepted_goal_stamp=first.accepted_goal_stamp,
        current_release_at=first.release_at,
    )
    assert replacement.changed
    assert replacement.reason == "REPLACED"
    assert replacement.accepted_goal_stamp == pytest.approx(10.5)
    assert replacement.release_at == pytest.approx(12.5)


def test_merger_cue_schedule_cancels_on_goal_invalidation():
    cancelled = update_cue_schedule(
        _goal(frame="", stamp=10.6),
        process_start_stamp=10.0,
        expected_frame="map",
        delay=2.0,
        current_accepted_goal_stamp=10.5,
        current_release_at=12.5,
    )
    assert cancelled.changed
    assert cancelled.reason == "INVALIDATED"
    assert cancelled.accepted_goal_stamp is None
    assert cancelled.release_at is None


def test_merger_cue_schedule_ignores_duplicate_or_older_goal():
    duplicate = update_cue_schedule(
        _goal(stamp=10.5),
        process_start_stamp=10.0,
        expected_frame="map",
        delay=2.0,
        current_accepted_goal_stamp=10.5,
        current_release_at=12.5,
    )
    assert not duplicate.changed
    assert duplicate.accepted_goal_stamp == pytest.approx(10.5)
    assert duplicate.release_at == pytest.approx(12.5)
