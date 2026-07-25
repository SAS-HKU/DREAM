"""Extract reviewed OACP-VB threshold candidates from a ROS 2 calibration bag.

The command is analysis-only: it reads ``/dream/oacp_vb_status`` and never
creates a ROS node, publisher, service client, or hardware command.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence

import numpy as np


STATUS_TOPIC = "/dream/oacp_vb_status"


@dataclass(frozen=True)
class CalibrationSample:
    status_stamp: float
    relative_stamp: float
    goal_revision: int
    goal_receipt_stamp: float
    sample_count: int
    risk_total: float
    pvs_length: float
    exploration_velocity_bound: float
    fallback_velocity_bound: float


def extract_calibration_samples(
    records: Iterable[tuple[float, dict]],
    *,
    goal_revision: Optional[int] = None,
    start_offset: Optional[float] = None,
    end_offset: Optional[float] = None,
) -> list[CalibrationSample]:
    """Select unique, motion-authorized occluded-phase samples.

    ``records`` contains ``(bag_stamp_seconds, decoded_status_payload)`` pairs.
    The assessor's monotonically increasing ``calibration_sample_count`` is
    used to avoid treating repeated status publications as new measurements.
    """

    if start_offset is not None and (
        not isfinite(float(start_offset)) or float(start_offset) < 0.0
    ):
        raise ValueError("start_offset must be finite and nonnegative")
    if end_offset is not None and (
        not isfinite(float(end_offset)) or float(end_offset) < 0.0
    ):
        raise ValueError("end_offset must be finite and nonnegative")
    if (
        start_offset is not None
        and end_offset is not None
        and float(end_offset) < float(start_offset)
    ):
        raise ValueError("end_offset must be at least start_offset")

    candidates: list[CalibrationSample] = []
    last_count_by_run: dict[tuple[int, float], int] = {}
    for bag_stamp, payload in records:
        if not isinstance(payload, dict):
            continue
        if (
            payload.get("provider") != "oacp_vb"
            or payload.get("calibration_logging_only") is not True
            or payload.get("calibration_run_active") is not True
            or payload.get("ready") is not True
            or payload.get("exact_bound_valid") is not True
        ):
            continue
        try:
            revision = int(payload["calibration_goal_revision"])
            goal_stamp = float(payload["calibration_goal_receipt_stamp"])
            count = int(payload["calibration_sample_count"])
            risk = float(payload["risk_total"])
            pvs_length = float(payload["pvs_length"])
            exploration_bound = float(
                payload["exploration_velocity_bound"]
            )
            fallback_bound = float(payload["fallback_velocity_bound"])
            status_stamp = float(payload.get("stamp", bag_stamp))
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        values = (
            float(bag_stamp),
            goal_stamp,
            risk,
            pvs_length,
            exploration_bound,
            fallback_bound,
            status_stamp,
        )
        if (
            isinstance(payload.get("calibration_goal_revision"), bool)
            or isinstance(payload.get("calibration_sample_count"), bool)
            or revision < 0
            or count <= 0
            or not all(isfinite(value) for value in values)
            or risk < 0.0
            or pvs_length <= 0.0
        ):
            continue
        if goal_revision is not None and revision != int(goal_revision):
            continue
        run_key = (revision, goal_stamp)
        previous_count = last_count_by_run.get(run_key, 0)
        if count <= previous_count:
            continue
        last_count_by_run[run_key] = count
        relative = status_stamp - goal_stamp
        if relative < 0.0:
            continue
        if start_offset is not None and relative < float(start_offset):
            continue
        if end_offset is not None and relative > float(end_offset):
            continue
        candidates.append(
            CalibrationSample(
                status_stamp=status_stamp,
                relative_stamp=relative,
                goal_revision=revision,
                goal_receipt_stamp=goal_stamp,
                sample_count=count,
                risk_total=risk,
                pvs_length=pvs_length,
                exploration_velocity_bound=exploration_bound,
                fallback_velocity_bound=fallback_bound,
            )
        )

    runs = {
        (sample.goal_revision, sample.goal_receipt_stamp)
        for sample in candidates
    }
    if goal_revision is None and len(runs) > 1:
        raise ValueError(
            "bag contains multiple calibration goals; rerun with "
            "--goal-revision after inspecting the listed status records"
        )
    return candidates


def summarize_calibration(
    samples: Sequence[CalibrationSample],
    *,
    fallback_ratio: float = 4.0 / 3.0,
) -> dict:
    """Return the required linear p70 threshold candidates."""

    ratio = float(fallback_ratio)
    if not isfinite(ratio) or ratio <= 1.0:
        raise ValueError("fallback_ratio must be finite and greater than one")
    if not samples:
        raise ValueError("no valid occluded-phase calibration samples found")
    risks = np.asarray(
        [sample.risk_total for sample in samples], dtype=np.float64
    )
    if float(np.max(risks)) <= 0.0:
        raise ValueError(
            "occluded-phase calibration samples contain no positive risk"
        )
    exploration = float(np.quantile(risks, 0.70, method="linear"))
    fallback = ratio * exploration
    first = samples[0]
    return {
        "method": "OACP-VB calibration candidate; human review still required",
        "sample_count": len(samples),
        "goal_revision": first.goal_revision,
        "goal_receipt_stamp": first.goal_receipt_stamp,
        "relative_time_start_seconds": min(
            sample.relative_stamp for sample in samples
        ),
        "relative_time_end_seconds": max(
            sample.relative_stamp for sample in samples
        ),
        "risk_minimum": float(np.min(risks)),
        "risk_median": float(np.median(risks)),
        "risk_p70_linear": exploration,
        "risk_maximum": float(np.max(risks)),
        "suggested_c_th_max_exploration": exploration,
        "fallback_ratio": ratio,
        "suggested_c_th_max_fallback": fallback,
        "approval": (
            "not calibrated until the exported curve, occluded interval, "
            "coverage, and saturation are reviewed"
        ),
    }


def _storage_identifier(bag_path: Path) -> str:
    metadata_path = bag_path / "metadata.yaml"
    if not metadata_path.is_file():
        return "sqlite3"
    import yaml

    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    try:
        identifier = str(
            metadata["rosbag2_bagfile_information"]["storage_identifier"]
        )
    except (KeyError, TypeError):
        return "sqlite3"
    return identifier or "sqlite3"


def read_status_records(bag_path: Path) -> Iterator[tuple[float, dict]]:
    """Yield decoded OACP-VB status records from a ROS 2 bag."""

    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except ImportError as exc:
        raise RuntimeError(
            "ROS 2 bag Python bindings are unavailable; source the Humble "
            "environment before running this command"
        ) from exc

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(
            uri=str(bag_path),
            storage_id=_storage_identifier(bag_path),
        ),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        ),
    )
    topic_types = {
        item.name: item.type for item in reader.get_all_topics_and_types()
    }
    message_type_name = topic_types.get(STATUS_TOPIC)
    if message_type_name != "std_msgs/msg/String":
        raise ValueError(
            f"{STATUS_TOPIC} is absent or has unexpected type "
            f"{message_type_name!r}"
        )
    message_type = get_message(message_type_name)
    while reader.has_next():
        topic, serialized, timestamp_ns = reader.read_next()
        if topic != STATUS_TOPIC:
            continue
        message = deserialize_message(serialized, message_type)
        try:
            payload = json.loads(message.data)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(payload, dict):
            yield float(timestamp_ns) * 1.0e-9, payload


def _write_csv(path: Path, samples: Sequence[CalibrationSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(CalibrationSample.__dataclass_fields__),
        )
        writer.writeheader()
        for sample in samples:
            writer.writerow(sample.__dict__)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path)
    parser.add_argument("--goal-revision", type=int, default=None)
    parser.add_argument("--start-offset", type=float, default=None)
    parser.add_argument("--end-offset", type=float, default=None)
    parser.add_argument("--fallback-ratio", type=float, default=4.0 / 3.0)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args(argv)

    records = list(read_status_records(arguments.bag))
    samples = extract_calibration_samples(
        records,
        goal_revision=arguments.goal_revision,
        start_offset=arguments.start_offset,
        end_offset=arguments.end_offset,
    )
    summary = summarize_calibration(
        samples, fallback_ratio=arguments.fallback_ratio
    )
    summary["bag"] = str(arguments.bag)
    summary["start_offset_filter"] = arguments.start_offset
    summary["end_offset_filter"] = arguments.end_offset
    rendered = json.dumps(summary, indent=2, allow_nan=False) + "\n"
    print(rendered, end="")
    if arguments.csv is not None:
        _write_csv(arguments.csv, samples)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
