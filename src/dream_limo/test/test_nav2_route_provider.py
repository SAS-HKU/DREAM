from pathlib import Path

import pytest
import yaml

from dream_limo.core.nav2_route import (
    goal_identity_matches,
    validate_freshness,
    validate_geometric_path,
    validate_planar_pose,
    validate_transform_sample,
)


def test_goal_identity_includes_yaw_and_revision_stamp():
    values = {
        "actual_x": 1.0,
        "actual_y": -0.2,
        "actual_yaw": 0.4,
        "actual_stamp": 10.0,
        "expected_x": 1.0,
        "expected_y": -0.2,
        "expected_yaw": 0.4,
        "expected_stamp": 10.0,
        "position_tolerance": 1.0e-3,
        "identity_tolerance": 1.0e-6,
    }
    assert goal_identity_matches(**values)
    assert not goal_identity_matches(**{**values, "actual_yaw": 0.5})
    assert not goal_identity_matches(**{**values, "actual_stamp": 10.1})


def _valid_path(**overrides):
    values = {
        "frame_id": "map",
        "pose_frames": ("", "map"),
        "positions_xyz": ((0.3, 0.4, 0.0), (1.0, 0.2, 0.0)),
        "quaternions_xyzw": ((0.0, 0.0, 0.0, 1.0),) * 2,
        "source_stamp": 9.9,
        "receipt_stamp": 10.0,
        "now": 10.0,
        "expected_frame": "map",
        "source_timeout": 1.5,
        "receipt_timeout": 1.5,
        "future_tolerance": 0.05,
    }
    values.update(overrides)
    return validate_geometric_path(**values)


def test_freshness_contract_rejects_zero_future_and_stale_samples():
    assert validate_freshness(
        10.0,
        now=10.0,
        maximum_age=0.5,
        future_tolerance=0.05,
        label="TF",
    ).valid
    assert validate_freshness(
        0.0,
        now=10.0,
        maximum_age=0.5,
        future_tolerance=0.05,
        label="TF",
    ).reason == "TF_STAMP_INVALID"
    assert validate_freshness(
        10.1,
        now=10.0,
        maximum_age=0.5,
        future_tolerance=0.05,
        label="TF",
    ).reason == "TF_STAMP_FUTURE"
    assert validate_freshness(
        9.5,
        now=10.0,
        maximum_age=0.5,
        future_tolerance=0.05,
        label="TF",
    ).reason == "TF_STALE"


def test_planar_pose_requires_exact_frame_finite_values_and_unit_quaternion():
    arguments = {
        "frame_id": "map",
        "expected_frame": "map",
        "position_xyz": (1.0, 2.0, 0.0),
        "quaternion_xyzw": (0.0, 0.0, 0.0, 1.0),
        "label": "GOAL",
    }
    assert validate_planar_pose(**arguments) == "ok"
    assert validate_planar_pose(**{**arguments, "frame_id": "odom"}) == (
        "GOAL_FRAME_MISMATCH"
    )
    assert validate_planar_pose(
        **{**arguments, "position_xyz": (float("nan"), 2.0, 0.0)}
    ) == "GOAL_NONFINITE"
    assert validate_planar_pose(
        **{**arguments, "quaternion_xyzw": (0.0, 0.0, 0.0, 0.5)}
    ) == "GOAL_QUATERNION_INVALID"


def test_transform_contract_checks_direction_and_source_freshness():
    valid = validate_transform_sample(
        parent_frame="map",
        child_frame="base_link",
        expected_parent="map",
        expected_child="base_link",
        translation_xyz=(0.5, 0.2, 0.0),
        quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
        source_stamp=9.9,
        now=10.0,
        maximum_age=0.5,
        future_tolerance=0.05,
    )
    assert valid.valid
    wrong_child = validate_transform_sample(
        parent_frame="map",
        child_frame="base_footprint",
        expected_parent="map",
        expected_child="base_link",
        translation_xyz=(0.5, 0.2, 0.0),
        quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
        source_stamp=9.9,
        now=10.0,
        maximum_age=0.5,
        future_tolerance=0.05,
    )
    assert wrong_child.reason == "TF_CHILD_FRAME_MISMATCH"


def test_path_contract_accepts_nonempty_world_path_and_empty_pose_headers():
    validation = _valid_path()
    assert validation.valid
    assert validation.reason == "PATH_VALID"
    assert validation.pose_count == 2
    assert validation.source_age == pytest.approx(0.1)


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        (
            {
                "pose_frames": (),
                "positions_xyz": (),
                "quaternions_xyzw": (),
            },
            "PATH_EMPTY",
        ),
        ({"frame_id": "odom"}, "PATH_FRAME_MISMATCH"),
        ({"pose_frames": ("odom", "map")}, "PATH_POSE_FRAME_MISMATCH"),
        ({"source_stamp": 8.0}, "PATH_SOURCE_STALE"),
        ({"receipt_stamp": 8.0}, "PATH_RECEIPT_STALE"),
        (
            {
                "positions_xyz": (
                    (0.3, 0.4, 0.0),
                    (float("inf"), 0.2, 0.0),
                )
            },
            "PATH_POSE_NONFINITE",
        ),
    ],
)
def test_path_contract_fails_closed(overrides, reason):
    validation = _valid_path(**overrides)
    assert not validation.valid
    assert validation.reason == reason


def test_nav2_config_is_planner_only_known_space_ackermann_contract():
    package = Path(__file__).resolve().parents[1]
    payload = yaml.safe_load(
        (package / "config" / "nav2_dream_planner.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert set(payload) == {"planner_server", "global_costmap"}
    planner = payload["planner_server"]["ros__parameters"]
    plugin = planner["GridBased"]
    assert planner["planner_plugins"] == ["GridBased"]
    assert plugin["plugin"] == "nav2_smac_planner/SmacPlannerHybrid"
    assert plugin["motion_model_for_search"] == "DUBIN"
    assert plugin["minimum_turning_radius"] == pytest.approx(0.4)
    assert plugin["allow_unknown"] is False

    costmap = payload["global_costmap"]["global_costmap"]["ros__parameters"]
    assert costmap["global_frame"] == "map"
    assert costmap["robot_base_frame"] == "base_link"
    assert costmap["rolling_window"] is False
    assert costmap["track_unknown_space"] is True
    assert costmap["plugins"] == ["obstacle_layer", "inflation_layer"]
    assert costmap["obstacle_layer"]["scan"]["topic"] == "/scan"
    assert "static_layer" not in costmap["plugins"]
