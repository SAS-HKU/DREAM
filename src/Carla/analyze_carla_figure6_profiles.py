#!/usr/bin/env python3
"""Build the CARLA speed--clearance evidence for revised Figure 6.

The script analyzes a complete, matched CARLA scene bank containing the four
shared-backbone controllers DREAM, IDEAM, ADA, and APF under two conditions:
an empty occlusion shadow and a true hidden-vehicle threat.  It uses the
physical scene seed as the independent analysis unit.  Within each scene, the
empty-shadow trace is aligned to the semantic-LiDAR reveal time measured in
the matching true-threat arm for the same controller.  No reveal timestamp is
invented for the empty-shadow condition.

Inputs are discovered recursively below ``--results-root``.  Every scene must
contain exactly one valid run for every controller--condition arm, and all
eight arms must share one condition-blind construction hash.  The principal
outputs are a two-by-two SciencePlots figure, aligned profile data, per-episode
metrics, a machine-readable aggregate summary, and a compact LaTeX table.

Example
-------
::

    python evaluation/analyze_carla_figure6_profiles.py \
        --results-root outputs/carla_field_baselines_n5_v20 \
        --output-dir outputs/carla_field_baselines_n5_v20/figure6_analysis
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


SCHEMA = "carla_figure6_reveal_aligned_profiles_v1"
CONTROLLERS = ("DREAM", "IDEAM", "ADA", "APF")
CONDITIONS = ("empty_shadow", "true_threat")
ARMS = tuple((controller, condition) for controller in CONTROLLERS for condition in CONDITIONS)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SEED_RE = re.compile(r"(?:^|[_-])seed[_-]?(\d+)(?:[_-]|$)", re.IGNORECASE)

PALETTE = {
    "DREAM": "#2878B5",
    "IDEAM": "#E07A1F",
    "ADA": "#2A9D6F",
    "APF": "#B84A62",
}
LINESTYLES = {
    "DREAM": "-",
    "IDEAM": "--",
    "ADA": "-.",
    "APF": ":",
}
DISPLAY_LABELS = {
    "DREAM": "DREAM",
    "IDEAM": "IDEAM",
    "ADA": "ADA-sourced",
    "APF": "APF-sourced",
}

# These fields retain the units used by the CARLA run summary.  They are kept
# outside the compact Figure-6 table so that the Section 4.4 safety, traffic,
# and asynchronous-runtime statements can be traced to the same scene bank.
SECTION_4_4_METRICS = {
    "ego_maximum_speed_loss_mps": "Maximum ego speed loss within the episode (m/s).",
    "ego_peak_deceleration_mps2": "Most negative ego longitudinal acceleration (m/s^2).",
    "minimum_hidden_oriented_clearance_m": (
        "Episode minimum signed oriented-box clearance to the hidden vehicle (m)."
    ),
    "minimum_hidden_ttc_2d_s": (
        "Episode minimum constant-velocity, oriented-box two-dimensional TTC to the "
        "hidden vehicle (s)."
    ),
    "minimum_clearance_first_3s_after_reference_m": (
        "Minimum signed oriented-box clearance to any neighbor in the first 3 s after "
        "the matched semantic-LiDAR reveal reference (m)."
    ),
    "traffic_total_integrated_speed_deficit_vehicle_m": (
        "Sum of surrounding-traffic integrated desired-speed deficits (vehicle-m)."
    ),
    "maximum_follower_speed_loss_mps": (
        "Largest speed loss among the designated following vehicles in the episode "
        "(m/s)."
    ),
    "traffic_hard_braking_actor_count": (
        "Number of surrounding traffic actors meeting the hard-braking criterion."
    ),
    "planner_mean_total_s": "Mean completed high-level planning time (s).",
    "planner_p95_total_s": "95th-percentile completed high-level planning time (s).",
    "planner_maximum_total_s": "Maximum completed high-level planning time (s).",
    "planner_effective_completed_update_rate_hz": (
        "Effective completed high-level update rate (Hz)."
    ),
    "planner_dropped_request_fraction": "Fraction of high-level requests dropped.",
    "planner_reveal_to_hidden_aware_plan_applied_s": (
        "Delay from semantic-LiDAR reveal to the first applied hidden-aware plan (s)."
    ),
    "low_level_mean_time_s": "Mean low-level control execution time (s).",
    "low_level_p95_time_s": "95th-percentile low-level control execution time (s).",
    "low_level_maximum_time_s": "Maximum low-level control execution time (s).",
    "low_level_deadline_miss_count": "Observed low-level deadline-miss count.",
    "low_level_deadline_miss_fraction": "Observed low-level deadline-miss fraction.",
    "physics_loop_mean_cycle_time_s": "Mean physics/control-loop cycle time (s).",
    "physics_loop_p95_cycle_time_s": (
        "95th-percentile physics/control-loop cycle time (s)."
    ),
    "physics_loop_maximum_cycle_time_s": "Maximum physics/control-loop cycle time (s).",
    "physics_loop_deadline_miss_count": "Observed physics-loop deadline-miss count.",
    "physics_loop_deadline_miss_fraction": "Observed physics-loop deadline-miss fraction.",
}


class ProfileAnalysisError(ValueError):
    """Raised when a result bank cannot support the requested comparison."""


@dataclass(frozen=True)
class RunRecord:
    """One validated controller--condition run."""

    directory: Path
    summary_path: Path
    trace_path: Path
    summary: Dict[str, Any]
    scene_seed: str
    construction_hash: str
    scenario_family: str
    scenario_id: str
    controller: str
    condition: str
    reveal_time_s: Optional[float]
    time_s: np.ndarray
    ego_speed_mps: np.ndarray
    clearance_m: np.ndarray

    @property
    def arm(self) -> Tuple[str, str]:
        return self.controller, self.condition


@dataclass(frozen=True)
class SceneBlock:
    """Eight matched arms sharing one physical scene construction."""

    scene_seed: str
    construction_hash: str
    scenario_family: str
    scenario_id: str
    arms: Mapping[Tuple[str, str], RunRecord]


def _number(value: Any) -> Optional[float]:
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _nested_number(mapping: Mapping[str, Any], *keys: str) -> Optional[float]:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return _number(value)


def _deadline_miss_count(fraction: Optional[float], executions: Optional[float]) -> Optional[int]:
    if fraction is None or executions is None:
        return None
    return int(round(fraction * executions))


def _maximum_follower_speed_loss(summary: Mapping[str, Any]) -> Optional[float]:
    followers = summary.get("followers")
    if not isinstance(followers, Mapping):
        return None
    values = []
    for record in followers.values():
        if not isinstance(record, Mapping):
            continue
        value = _number(record.get("maximum_speed_loss_mps"))
        if value is not None:
            values.append(value)
    return max(values) if values else None


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ProfileAnalysisError("cannot read JSON {}: {}".format(path, exc)) from exc
    if not isinstance(payload, dict):
        raise ProfileAnalysisError("JSON root must be an object: {}".format(path))
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(_json_safe(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _json_safe(row.get(key)) for key in fieldnames})


def _discover_summary_paths(results_root: Path) -> List[Path]:
    if not results_root.exists() or not results_root.is_dir():
        raise ProfileAnalysisError("results root is not a directory: {}".format(results_root))
    paths = sorted(path.resolve() for path in results_root.rglob("summary.json"))
    if not paths:
        raise ProfileAnalysisError("no summary.json files found under {}".format(results_root))
    return paths


def _resolved_manifest(summary: Mapping[str, Any], summary_path: Path) -> Dict[str, Any]:
    inline = summary.get("resolved_manifest")
    if isinstance(inline, Mapping):
        return dict(inline)
    explicit = summary.get("resolved_manifest_path")
    candidates: List[Path] = []
    if explicit:
        candidate = Path(str(explicit)).expanduser()
        candidates.append(candidate if candidate.is_absolute() else summary_path.parent / candidate)
    candidates.append(summary_path.parent / "resolved_manifest.json")
    for candidate in candidates:
        if candidate.is_file():
            return _read_json(candidate)
    raise ProfileAnalysisError(
        "run does not provide a readable resolved_manifest.json: {}".format(summary_path.parent)
    )


def _manifest_hash(manifest: Mapping[str, Any]) -> Optional[str]:
    provenance = manifest.get("provenance")
    if isinstance(provenance, Mapping):
        value = provenance.get("construction_hash_sha256")
        if value is not None:
            return str(value).lower()
    value = manifest.get("construction_hash")
    return str(value).lower() if value is not None else None


def _scene_seed(summary: Mapping[str, Any], manifest: Mapping[str, Any], path: Path) -> str:
    for source in (summary, manifest):
        for key in ("scene_seed", "scenario_seed", "random_seed", "seed"):
            if source.get(key) is not None:
                return str(source[key])
    for text in (str(summary.get("scenario_id", "")), path.parent.name):
        match = SEED_RE.search(text)
        if match:
            return match.group(1)
    raise ProfileAnalysisError("cannot determine scene seed for {}".format(path))


def _read_trace(path: Path, controller: str, condition: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    required = ("time_s", "ego_speed_mps", "minimum_oriented_clearance_m")
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ProfileAnalysisError("trace has no header: {}".format(path))
            missing = [name for name in required if name not in reader.fieldnames]
            if missing:
                raise ProfileAnalysisError(
                    "trace {} is missing columns: {}".format(path, ", ".join(missing))
                )
            times: List[float] = []
            speeds: List[float] = []
            clearances: List[float] = []
            for row_number, row in enumerate(reader, start=2):
                trace_controller = str(row.get("controller", controller)).upper()
                trace_condition = str(row.get("condition", condition)).lower()
                if trace_controller and trace_controller != controller:
                    raise ProfileAnalysisError(
                        "controller mismatch in {} row {}".format(path, row_number)
                    )
                if trace_condition and trace_condition != condition:
                    raise ProfileAnalysisError(
                        "condition mismatch in {} row {}".format(path, row_number)
                    )
                values = [_number(row.get(name)) for name in required]
                if any(value is None for value in values):
                    raise ProfileAnalysisError(
                        "non-finite required value in {} row {}".format(path, row_number)
                    )
                times.append(float(values[0]))
                speeds.append(float(values[1]))
                clearances.append(float(values[2]))
    except OSError as exc:
        raise ProfileAnalysisError("cannot read trace {}: {}".format(path, exc)) from exc
    if len(times) < 2:
        raise ProfileAnalysisError("trace must contain at least two ticks: {}".format(path))
    time_array = np.asarray(times, dtype=float)
    if not np.all(np.diff(time_array) > 0.0):
        raise ProfileAnalysisError("trace times must be strictly increasing: {}".format(path))
    return time_array, np.asarray(speeds, dtype=float), np.asarray(clearances, dtype=float)


def _load_run(summary_path: Path) -> RunRecord:
    summary = _read_json(summary_path)
    controller = str(summary.get("controller", "")).upper()
    condition = str(summary.get("condition", "")).lower()
    if controller not in CONTROLLERS:
        raise ProfileAnalysisError(
            "unexpected controller {!r} in {}; expected {}".format(
                controller, summary_path, ", ".join(CONTROLLERS)
            )
        )
    if condition not in CONDITIONS:
        raise ProfileAnalysisError(
            "unexpected condition {!r} in {}; expected {}".format(
                condition, summary_path, ", ".join(CONDITIONS)
            )
        )
    qualification = summary.get("qualification")
    if not isinstance(qualification, Mapping) or qualification.get("valid_for_analysis") is not True:
        raise ProfileAnalysisError("run is not valid_for_analysis: {}".format(summary_path.parent))
    manifest = _resolved_manifest(summary, summary_path)
    construction_hash = str(summary.get("construction_hash", "")).lower()
    if not SHA256_RE.fullmatch(construction_hash):
        raise ProfileAnalysisError("invalid construction hash in {}".format(summary_path))
    manifest_hash = _manifest_hash(manifest)
    if manifest_hash is not None and manifest_hash != construction_hash:
        raise ProfileAnalysisError(
            "summary/manifest construction-hash mismatch in {}".format(summary_path.parent)
        )
    reveal_time = _number(summary.get("reveal_time_s"))
    if condition == "true_threat":
        if reveal_time is None:
            raise ProfileAnalysisError(
                "true-threat run lacks a semantic-LiDAR reveal_time_s: {}".format(summary_path.parent)
            )
        if qualification.get("valid_reveal_pass") is not True:
            raise ProfileAnalysisError(
                "true-threat run does not pass valid_reveal_pass: {}".format(summary_path.parent)
            )
    trace_path = summary_path.parent / "tick_trace.csv"
    if not trace_path.is_file():
        raise ProfileAnalysisError("missing tick_trace.csv: {}".format(summary_path.parent))
    time_s, speed, clearance = _read_trace(trace_path, controller, condition)
    reported_minimum = _number(
        summary.get("ego", {}).get("minimum_oriented_box_clearance_m")
        if isinstance(summary.get("ego"), Mapping)
        else None
    )
    if reported_minimum is not None and not math.isclose(
        reported_minimum, float(np.min(clearance)), rel_tol=1e-7, abs_tol=1e-6
    ):
        raise ProfileAnalysisError(
            "summary/trace minimum-clearance mismatch in {}".format(summary_path.parent)
        )
    return RunRecord(
        directory=summary_path.parent.resolve(),
        summary_path=summary_path,
        trace_path=trace_path.resolve(),
        summary=summary,
        scene_seed=_scene_seed(summary, manifest, summary_path),
        construction_hash=construction_hash,
        scenario_family=str(summary.get("scenario_family", manifest.get("scenario_family", ""))),
        scenario_id=str(summary.get("scenario_id", "")),
        controller=controller,
        condition=condition,
        reveal_time_s=reveal_time,
        time_s=time_s,
        ego_speed_mps=speed,
        clearance_m=clearance,
    )


def _seed_sort_key(seed: str) -> Tuple[int, Any]:
    try:
        return 0, int(seed)
    except ValueError:
        return 1, seed


def load_scene_blocks(results_root: Path) -> List[SceneBlock]:
    """Discover runs and validate complete eight-arm physical-scene blocks."""

    by_seed: Dict[str, List[RunRecord]] = {}
    for summary_path in _discover_summary_paths(results_root):
        run = _load_run(summary_path)
        by_seed.setdefault(run.scene_seed, []).append(run)
    blocks: List[SceneBlock] = []
    for seed in sorted(by_seed, key=_seed_sort_key):
        runs = by_seed[seed]
        hashes = {run.construction_hash for run in runs}
        if len(hashes) != 1:
            raise ProfileAnalysisError(
                "scene seed {} has {} construction hashes; arms are not matched".format(
                    seed, len(hashes)
                )
            )
        families = {run.scenario_family for run in runs}
        if len(families) != 1:
            raise ProfileAnalysisError("scene seed {} spans multiple scenario families".format(seed))
        arms: Dict[Tuple[str, str], RunRecord] = {}
        for run in runs:
            if run.arm in arms:
                raise ProfileAnalysisError(
                    "scene seed {} has duplicate {} / {} runs".format(
                        seed, run.controller, run.condition
                    )
                )
            arms[run.arm] = run
        missing = ["{} / {}".format(*arm) for arm in ARMS if arm not in arms]
        extras = ["{} / {}".format(*arm) for arm in arms if arm not in ARMS]
        if missing or extras or len(runs) != len(ARMS):
            details = []
            if missing:
                details.append("missing: {}".format(", ".join(missing)))
            if extras:
                details.append("unexpected: {}".format(", ".join(extras)))
            details.append("observed {} runs, expected {}".format(len(runs), len(ARMS)))
            raise ProfileAnalysisError("incomplete scene seed {} ({})".format(seed, "; ".join(details)))
        scenario_ids = {run.scenario_id for run in runs}
        if len(scenario_ids) != 1:
            raise ProfileAnalysisError("scene seed {} has inconsistent scenario_id values".format(seed))
        blocks.append(
            SceneBlock(
                scene_seed=seed,
                construction_hash=next(iter(hashes)),
                scenario_family=next(iter(families)),
                scenario_id=next(iter(scenario_ids)),
                arms=arms,
            )
        )
    if not blocks:
        raise ProfileAnalysisError("no complete scene blocks found")
    return blocks


def _common_grid(
    blocks: Sequence[SceneBlock], requested_start: float, requested_end: float, step: float
) -> np.ndarray:
    if not math.isfinite(step) or step <= 0.0:
        raise ProfileAnalysisError("grid step must be positive and finite")
    if requested_end <= requested_start:
        raise ProfileAnalysisError("grid end must be later than grid start")
    supported_start = requested_start
    supported_end = requested_end
    for block in blocks:
        for controller in CONTROLLERS:
            reveal = block.arms[(controller, "true_threat")].reveal_time_s
            if reveal is None:
                raise ProfileAnalysisError("internal error: missing validated reveal time")
            for condition in CONDITIONS:
                run = block.arms[(controller, condition)]
                supported_start = max(supported_start, float(run.time_s[0] - reveal))
                supported_end = min(supported_end, float(run.time_s[-1] - reveal))
    start = math.ceil((supported_start - 1e-10) / step) * step
    end = math.floor((supported_end + 1e-10) / step) * step
    if end <= start:
        raise ProfileAnalysisError(
            "traces do not share a usable reveal-aligned interval within [{}, {}] s".format(
                requested_start, requested_end
            )
        )
    count = int(round((end - start) / step)) + 1
    grid = start + np.arange(count, dtype=float) * step
    return np.round(grid, 10)


def _aligned_arrays(
    blocks: Sequence[SceneBlock], grid: np.ndarray
) -> Dict[Tuple[str, str], Dict[str, np.ndarray]]:
    profiles: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}
    for controller in CONTROLLERS:
        for condition in CONDITIONS:
            speeds: List[np.ndarray] = []
            clearances: List[np.ndarray] = []
            for block in blocks:
                reveal = block.arms[(controller, "true_threat")].reveal_time_s
                if reveal is None:
                    raise ProfileAnalysisError("internal error: missing validated reveal time")
                run = block.arms[(controller, condition)]
                query_time = grid + reveal
                speeds.append(np.interp(query_time, run.time_s, run.ego_speed_mps))
                clearances.append(np.interp(query_time, run.time_s, run.clearance_m))
            profiles[(controller, condition)] = {
                "ego_speed_mps": np.vstack(speeds),
                "minimum_oriented_clearance_m": np.vstack(clearances),
            }
    return profiles


def _bootstrap_profile(
    values: np.ndarray, bootstrap_indices: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(values, axis=0)
    boot_means = np.mean(values[bootstrap_indices, :], axis=1)
    lower, upper = np.percentile(boot_means, [2.5, 97.5], axis=0)
    return mean, lower, upper


def _bootstrap_scalar(
    values: Sequence[float], bootstrap_indices: np.ndarray
) -> Dict[str, float]:
    array = np.asarray(values, dtype=float)
    boot_means = np.mean(array[bootstrap_indices], axis=1)
    lower, upper = np.percentile(boot_means, [2.5, 97.5])
    return {
        "mean": float(np.mean(array)),
        "bootstrap_95_lower": float(lower),
        "bootstrap_95_upper": float(upper),
    }


def _optional_descriptive_summary(
    values: Sequence[Any], bootstrap_indices: np.ndarray
) -> Dict[str, Any]:
    """Summarize optional scene-level values without hiding missing outcomes.

    A bootstrap interval is emitted only when every scene contributes a finite
    value.  This is important for reveal-to-plan latency: failure to apply a
    hidden-aware plan is not silently converted into a selectively observed
    timing sample.
    """

    numeric = [_number(value) for value in values]
    observed = [value for value in numeric if value is not None]
    result: Dict[str, Any] = {
        "number_of_scenes": len(numeric),
        "number_observed": len(observed),
        "mean": float(np.mean(observed)) if observed else None,
        "minimum_observed": min(observed) if observed else None,
        "maximum_observed": max(observed) if observed else None,
        "bootstrap_95_lower": None,
        "bootstrap_95_upper": None,
        "interval_status": "not_available_no_observations",
    }
    if len(observed) == len(numeric) and observed:
        result.update(_bootstrap_scalar([float(value) for value in observed], bootstrap_indices))
        result["interval_status"] = "complete_scene_block_descriptive_interval"
    elif observed:
        result["interval_status"] = (
            "not_reported_incomplete_observation; mean_is_descriptive_for_observed_scenes_only"
        )
    return result


def _episode_rows(blocks: Sequence[SceneBlock]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for block in blocks:
        for controller in CONTROLLERS:
            reveal = block.arms[(controller, "true_threat")].reveal_time_s
            if reveal is None:
                raise ProfileAnalysisError("internal error: missing validated reveal time")
            for condition in CONDITIONS:
                run = block.arms[(controller, condition)]
                post_mask = (run.time_s >= reveal) & (run.time_s <= reveal + 3.0)
                summary = run.summary
                low_level_deadline_fraction = _nested_number(
                    summary, "low_level", "deadline_miss_fraction"
                )
                low_level_executions = _nested_number(
                    summary, "low_level", "control_executions"
                )
                physics_deadline_fraction = _nested_number(
                    summary, "physics_control_loop", "deadline_miss_fraction"
                )
                rows.append(
                    {
                        "scene_seed": block.scene_seed,
                        "construction_hash": block.construction_hash,
                        "scenario_id": block.scenario_id,
                        "controller": controller,
                        "condition": condition,
                        "run_directory": str(run.directory),
                        "summary_path": str(run.summary_path),
                        "trace_path": str(run.trace_path),
                        "reveal_reference_time_s": reveal,
                        "observed_semantic_lidar_reveal_time_s": run.reveal_time_s,
                        "number_of_ticks": int(run.time_s.size),
                        "trace_start_time_s": float(run.time_s[0]),
                        "trace_end_time_s": float(run.time_s[-1]),
                        "mean_ego_speed_mps": float(np.mean(run.ego_speed_mps)),
                        "minimum_ego_speed_mps": float(np.min(run.ego_speed_mps)),
                        "minimum_oriented_clearance_m": float(np.min(run.clearance_m)),
                        "minimum_clearance_first_3s_after_reference_m": (
                            float(np.min(run.clearance_m[post_mask])) if np.any(post_mask) else None
                        ),
                        "minimum_hidden_oriented_clearance_m": _nested_number(
                            summary, "ego", "minimum_hidden_oriented_box_clearance_m"
                        ),
                        "minimum_hidden_ttc_2d_s": _nested_number(
                            summary, "ego", "minimum_hidden_ttc_2d_s"
                        ),
                        "ego_maximum_speed_loss_mps": _nested_number(
                            summary, "ego", "maximum_speed_loss_mps"
                        ),
                        "ego_peak_deceleration_mps2": _nested_number(
                            summary, "ego", "peak_deceleration_mps2"
                        ),
                        "collision_incidence": int(summary.get("collision_incidence", 0)),
                        "near_collision_incidence": int(summary.get("near_collision_incidence", 0)),
                        "collision_or_near_incidence": int(
                            summary.get("collision_or_near_incidence", 0)
                        ),
                        "traffic_total_integrated_speed_deficit_vehicle_m": _nested_number(
                            summary,
                            "traffic_disturbance",
                            "total_integrated_speed_deficit_vehicle_m",
                        ),
                        "maximum_follower_speed_loss_mps": (
                            _maximum_follower_speed_loss(summary)
                        ),
                        "traffic_hard_braking_actor_count": _nested_number(
                            summary, "traffic_disturbance", "hard_braking_actor_count"
                        ),
                        "planner_mean_total_s": _nested_number(
                            summary, "planner", "mean_total_s"
                        ),
                        "planner_p95_total_s": _nested_number(
                            summary, "planner", "p95_total_s"
                        ),
                        "planner_maximum_total_s": _nested_number(
                            summary, "planner", "maximum_total_s"
                        ),
                        "planner_effective_completed_update_rate_hz": _nested_number(
                            summary, "planner", "effective_completed_update_rate_hz"
                        ),
                        "planner_dropped_request_fraction": _nested_number(
                            summary, "planner", "dropped_request_fraction"
                        ),
                        "planner_reveal_to_hidden_aware_plan_applied_s": _nested_number(
                            summary, "planner", "reveal_to_hidden_aware_plan_applied_s"
                        ),
                        "low_level_mean_time_s": _nested_number(
                            summary, "low_level", "mean_time_s"
                        ),
                        "low_level_p95_time_s": _nested_number(
                            summary, "low_level", "p95_time_s"
                        ),
                        "low_level_maximum_time_s": _nested_number(
                            summary, "low_level", "maximum_time_s"
                        ),
                        "low_level_deadline_miss_count": _deadline_miss_count(
                            low_level_deadline_fraction, low_level_executions
                        ),
                        "low_level_deadline_miss_fraction": low_level_deadline_fraction,
                        "physics_loop_mean_cycle_time_s": _nested_number(
                            summary, "physics_control_loop", "mean_cycle_time_s"
                        ),
                        "physics_loop_p95_cycle_time_s": _nested_number(
                            summary, "physics_control_loop", "p95_cycle_time_s"
                        ),
                        "physics_loop_maximum_cycle_time_s": _nested_number(
                            summary, "physics_control_loop", "maximum_cycle_time_s"
                        ),
                        "physics_loop_deadline_miss_count": _deadline_miss_count(
                            physics_deadline_fraction, float(run.time_s.size)
                        ),
                        "physics_loop_deadline_miss_fraction": physics_deadline_fraction,
                    }
                )
    return rows


def _aggregate_summary(
    blocks: Sequence[SceneBlock],
    episode_rows: Sequence[Mapping[str, Any]],
    grid: np.ndarray,
    bootstrap_indices: np.ndarray,
    bootstrap_seed: int,
    bootstrap_resamples: int,
    results_root: Path,
) -> Dict[str, Any]:
    by_arm: Dict[str, Any] = {}
    for condition in CONDITIONS:
        by_arm[condition] = {}
        for controller in CONTROLLERS:
            selected = [
                row
                for row in episode_rows
                if row["condition"] == condition and row["controller"] == controller
            ]
            speed_stats = _bootstrap_scalar(
                [float(row["mean_ego_speed_mps"]) for row in selected], bootstrap_indices
            )
            clearance_stats = _bootstrap_scalar(
                [float(row["minimum_oriented_clearance_m"]) for row in selected],
                bootstrap_indices,
            )
            post_stats = _bootstrap_scalar(
                [float(row["minimum_clearance_first_3s_after_reference_m"]) for row in selected],
                bootstrap_indices,
            )
            section_metrics = {
                metric: _optional_descriptive_summary(
                    [row.get(metric) for row in selected], bootstrap_indices
                )
                for metric in SECTION_4_4_METRICS
            }
            by_arm[condition][controller] = {
                "display_label": DISPLAY_LABELS[controller],
                "number_of_independent_scenes": len(selected),
                "mean_episode_mean_ego_speed_mps": speed_stats,
                "mean_episode_minimum_oriented_clearance_m": clearance_stats,
                "mean_episode_minimum_clearance_first_3s_after_reference_m": post_stats,
                "collision_count": sum(int(row["collision_incidence"]) for row in selected),
                "near_collision_count": sum(
                    int(row["near_collision_incidence"]) for row in selected
                ),
                "collision_or_near_count": sum(
                    int(row["collision_or_near_incidence"]) for row in selected
                ),
                "section_4_4_metrics": section_metrics,
            }
    row_index = {
        (str(row["scene_seed"]), str(row["controller"]), str(row["condition"])): row
        for row in episode_rows
    }
    dream_contrasts: Dict[str, Any] = {}
    for condition in CONDITIONS:
        dream_contrasts[condition] = {}
        for comparator in ("IDEAM", "ADA", "APF"):
            speed_differences = []
            clearance_differences = []
            collision_differences = []
            near_collision_differences = []
            for block in blocks:
                dream = row_index[(block.scene_seed, "DREAM", condition)]
                other = row_index[(block.scene_seed, comparator, condition)]
                speed_differences.append(
                    float(dream["mean_ego_speed_mps"]) - float(other["mean_ego_speed_mps"])
                )
                clearance_differences.append(
                    float(dream["minimum_oriented_clearance_m"])
                    - float(other["minimum_oriented_clearance_m"])
                )
                collision_differences.append(
                    float(dream["collision_incidence"])
                    - float(other["collision_incidence"])
                )
                near_collision_differences.append(
                    float(dream["near_collision_incidence"])
                    - float(other["near_collision_incidence"])
                )
            dream_contrasts[condition][comparator] = {
                "comparison": "DREAM minus {}".format(DISPLAY_LABELS[comparator]),
                "paired_difference_episode_mean_ego_speed_mps": _bootstrap_scalar(
                    speed_differences, bootstrap_indices
                ),
                "paired_difference_episode_minimum_oriented_clearance_m": _bootstrap_scalar(
                    clearance_differences, bootstrap_indices
                ),
                "paired_difference_collision_incidence": _bootstrap_scalar(
                    collision_differences, bootstrap_indices
                ),
                "paired_difference_near_collision_incidence": _bootstrap_scalar(
                    near_collision_differences, bootstrap_indices
                ),
                "section_4_4_paired_metric_differences": {
                    metric: _optional_descriptive_summary(
                        [
                            (
                                float(row_index[(block.scene_seed, "DREAM", condition)][metric])
                                - float(
                                    row_index[(block.scene_seed, comparator, condition)][metric]
                                )
                            )
                            if _number(
                                row_index[(block.scene_seed, "DREAM", condition)].get(metric)
                            )
                            is not None
                            and _number(
                                row_index[(block.scene_seed, comparator, condition)].get(metric)
                            )
                            is not None
                            else None
                            for block in blocks
                        ],
                        bootstrap_indices,
                    )
                    for metric in SECTION_4_4_METRICS
                },
            }
    condition_contrasts: Dict[str, Any] = {}
    for controller in CONTROLLERS:
        speed_differences = []
        clearance_differences = []
        for block in blocks:
            threat = row_index[(block.scene_seed, controller, "true_threat")]
            empty = row_index[(block.scene_seed, controller, "empty_shadow")]
            speed_differences.append(
                float(threat["mean_ego_speed_mps"]) - float(empty["mean_ego_speed_mps"])
            )
            clearance_differences.append(
                float(threat["minimum_oriented_clearance_m"])
                - float(empty["minimum_oriented_clearance_m"])
            )
        condition_contrasts[controller] = {
            "display_label": DISPLAY_LABELS[controller],
            "comparison": "true threat minus empty shadow",
            "paired_difference_episode_mean_ego_speed_mps": _bootstrap_scalar(
                speed_differences, bootstrap_indices
            ),
            "paired_difference_episode_minimum_oriented_clearance_m": _bootstrap_scalar(
                clearance_differences, bootstrap_indices
            ),
        }
    empty_shadow_conservatism_tax: Dict[str, Any] = {}
    for controller in CONTROLLERS:
        values = []
        for block in blocks:
            reference = row_index[(block.scene_seed, "IDEAM", "empty_shadow")]
            candidate = row_index[(block.scene_seed, controller, "empty_shadow")]
            values.append(
                float(reference["mean_ego_speed_mps"])
                - float(candidate["mean_ego_speed_mps"])
            )
        empty_shadow_conservatism_tax[controller] = {
            "display_label": DISPLAY_LABELS[controller],
            "definition": "IDEAM minus candidate episode-mean ego speed",
            "conservatism_tax_mps": _bootstrap_scalar(values, bootstrap_indices),
        }
    return {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "results_root": str(results_root),
        "analysis_unit": "condition-blind physical scene construction",
        "number_of_independent_scenes": len(blocks),
        "number_of_valid_runs": len(blocks) * len(ARMS),
        "controllers": list(CONTROLLERS),
        "controller_display_labels": dict(DISPLAY_LABELS),
        "conditions": list(CONDITIONS),
        "scene_blocks": [
            {
                "scene_seed": block.scene_seed,
                "construction_hash": block.construction_hash,
                "scenario_family": block.scenario_family,
                "scenario_id": block.scenario_id,
            }
            for block in blocks
        ],
        "alignment": {
            "event": "semantic-LiDAR reveal in the matched true-threat arm",
            "empty_shadow_policy": (
                "aligned to the true-threat reveal from the same scene and controller"
            ),
            "grid_start_s": float(grid[0]),
            "grid_end_s": float(grid[-1]),
            "grid_step_s": float(grid[1] - grid[0]),
            "number_of_grid_points": int(grid.size),
        },
        "bootstrap": {
            "purpose": "descriptive uncertainty interval; no p-values are produced",
            "resampling_unit": "complete physical scene block",
            "confidence_level": 0.95,
            "resamples": bootstrap_resamples,
            "random_seed": bootstrap_seed,
        },
        "clearance_definition": (
            "signed minimum separation between ego and neighboring oriented boxes; "
            "zero denotes contact and negative values denote overlap"
        ),
        "section_4_4_metric_definitions": dict(SECTION_4_4_METRICS),
        "by_condition_and_controller": by_arm,
        "paired_dream_minus_comparator": dream_contrasts,
        "paired_true_threat_minus_empty_shadow": condition_contrasts,
        "empty_shadow_conservatism_tax_vs_ideam": empty_shadow_conservatism_tax,
    }


def _aligned_rows(
    blocks: Sequence[SceneBlock],
    profiles: Mapping[Tuple[str, str], Mapping[str, np.ndarray]],
    grid: np.ndarray,
    bootstrap_indices: np.ndarray,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for controller in CONTROLLERS:
        for condition in CONDITIONS:
            values = profiles[(controller, condition)]
            speed_mean, speed_lower, speed_upper = _bootstrap_profile(
                values["ego_speed_mps"], bootstrap_indices
            )
            clearance_mean, clearance_lower, clearance_upper = _bootstrap_profile(
                values["minimum_oriented_clearance_m"], bootstrap_indices
            )
            for scene_index, block in enumerate(blocks):
                reveal = block.arms[(controller, "true_threat")].reveal_time_s
                for time_index, aligned_time in enumerate(grid):
                    rows.append(
                        {
                            "profile_type": "scene",
                            "scene_seed": block.scene_seed,
                            "construction_hash": block.construction_hash,
                            "controller": controller,
                            "condition": condition,
                            "time_relative_to_reveal_s": float(aligned_time),
                            "reveal_reference_time_s": reveal,
                            "absolute_query_time_s": float(aligned_time + float(reveal)),
                            "ego_speed_mps": float(values["ego_speed_mps"][scene_index, time_index]),
                            "minimum_oriented_clearance_m": float(
                                values["minimum_oriented_clearance_m"][scene_index, time_index]
                            ),
                        }
                    )
            for label, speed, clearance in (
                ("mean", speed_mean, clearance_mean),
                ("bootstrap_95_lower", speed_lower, clearance_lower),
                ("bootstrap_95_upper", speed_upper, clearance_upper),
            ):
                for time_index, aligned_time in enumerate(grid):
                    rows.append(
                        {
                            "profile_type": label,
                            "scene_seed": "",
                            "construction_hash": "",
                            "controller": controller,
                            "condition": condition,
                            "time_relative_to_reveal_s": float(aligned_time),
                            "reveal_reference_time_s": "",
                            "absolute_query_time_s": "",
                            "ego_speed_mps": float(speed[time_index]),
                            "minimum_oriented_clearance_m": float(clearance[time_index]),
                        }
                    )
    return rows


def _plot_profiles(
    profiles: Mapping[Tuple[str, str], Mapping[str, np.ndarray]],
    grid: np.ndarray,
    bootstrap_indices: np.ndarray,
    output_stem: Path,
) -> Tuple[Path, Path]:
    try:
        import matplotlib.pyplot as plt
        import scienceplots  # noqa: F401  # registers SciencePlots styles
    except ImportError as exc:
        raise ProfileAnalysisError(
            "figure generation requires matplotlib and the SciencePlots package"
        ) from exc

    panel_titles = {
        ("empty_shadow", "ego_speed_mps"): "(a) Empty shadow: ego speed",
        ("empty_shadow", "minimum_oriented_clearance_m"): (
            "(b) Empty shadow: nearest-neighbor clearance"
        ),
        ("true_threat", "ego_speed_mps"): "(c) True threat: ego speed",
        ("true_threat", "minimum_oriented_clearance_m"): (
            "(d) True threat: nearest-neighbor clearance"
        ),
    }
    with plt.style.context(["science", "no-latex"]):
        plt.rcParams.update(
            {
                "font.size": 8.5,
                "axes.titlesize": 9.3,
                "axes.labelsize": 8.8,
                "xtick.labelsize": 8.0,
                "ytick.labelsize": 8.0,
                "legend.fontsize": 8.0,
                "lines.solid_capstyle": "round",
            }
        )
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(7.25, 5.25),
            sharex=True,
            sharey=False,
            constrained_layout=True,
        )
        speed_values = np.concatenate(
            [
                profiles[(controller, condition)]["ego_speed_mps"].ravel()
                for controller in CONTROLLERS
                for condition in CONDITIONS
            ]
        )
        speed_margin = max(0.25, 0.04 * float(np.ptp(speed_values)))
        speed_limits = (
            float(np.min(speed_values) - speed_margin),
            float(np.max(speed_values) + speed_margin),
        )
        metrics = ("ego_speed_mps", "minimum_oriented_clearance_m")
        for row_index, condition in enumerate(CONDITIONS):
            for column_index, metric in enumerate(metrics):
                ax = axes[row_index, column_index]
                for controller in CONTROLLERS:
                    values = profiles[(controller, condition)][metric]
                    mean, lower, upper = _bootstrap_profile(values, bootstrap_indices)
                    color = PALETTE[controller]
                    linestyle = LINESTYLES[controller]
                    for scene_values in values:
                        ax.plot(
                            grid,
                            scene_values,
                            color=color,
                            linestyle=linestyle,
                            linewidth=0.52,
                            alpha=0.16,
                            zorder=1,
                        )
                    ax.fill_between(
                        grid,
                        lower,
                        upper,
                        color=color,
                        alpha=0.10,
                        linewidth=0.0,
                        zorder=2,
                    )
                    ax.plot(
                        grid,
                        mean,
                        color=color,
                        linestyle=linestyle,
                        linewidth=1.75,
                        label=DISPLAY_LABELS[controller],
                        zorder=3,
                    )
                ax.axvline(0.0, color="#444444", linestyle=(0, (2, 2)), linewidth=0.85)
                if metric == "minimum_oriented_clearance_m":
                    ax.axhline(0.0, color="#666666", linestyle="--", linewidth=0.8)
                else:
                    ax.set_ylim(*speed_limits)
                ax.set_title(panel_titles[(condition, metric)], loc="left", pad=4.0)
                ax.set_xlim(float(grid[0]), float(grid[-1]))
                ax.grid(True, color="#D7DCE2", linewidth=0.45, alpha=0.65)
                if row_index == 1:
                    ax.set_xlabel("Time relative to matched reveal reference (s)")
                if column_index == 0:
                    ax.set_ylabel("Ego speed (m s$^{-1}$)")
                else:
                    ax.set_ylabel("Signed oriented-box clearance (m)")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="outside upper center",
            ncol=len(CONTROLLERS),
            frameon=False,
            handlelength=2.7,
            columnspacing=1.5,
        )
        pdf_path = output_stem.with_suffix(".pdf")
        png_path = output_stem.with_suffix(".png")
        fig.savefig(pdf_path, bbox_inches="tight")
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return pdf_path, png_path


def _latex_value(statistic: Mapping[str, Any]) -> str:
    return "{:.2f} [{:.2f}, {:.2f}]".format(
        float(statistic["mean"]),
        float(statistic["bootstrap_95_lower"]),
        float(statistic["bootstrap_95_upper"]),
    )


def _write_latex_table(path: Path, aggregate: Mapping[str, Any]) -> None:
    by_arm = aggregate["by_condition_and_controller"]
    number_scenes = int(aggregate["number_of_independent_scenes"])
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{CARLA evaluation in matched empty-shadow and true-threat scenes. "
        r"Speed entries are means of episode-mean speeds, and clearance entries are "
        r"means of episode minima across scenes; brackets give descriptive 95\% "
        r"scene-block bootstrap intervals, not inferential significance tests. Clearance is "
        r"the episode minimum "
        r"signed oriented-box separation; zero denotes contact and negative values denote "
        r"overlap. Event columns report the observed number of episodes out of $n="
        + str(number_scenes)
        + r"$ independent scene constructions.}",
        r"\label{tab:carla-shadow-profiles}",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Condition & Controller & Mean ego speed (m s$^{-1}$) & "
        r"Episode-minimum clearance (m) & Collision & Near collision \\",
        r"\midrule",
    ]
    condition_labels = {"empty_shadow": "Empty shadow", "true_threat": "True threat"}
    for condition_index, condition in enumerate(CONDITIONS):
        for controller in CONTROLLERS:
            stats = by_arm[condition][controller]
            lines.append(
                "{} & {} & {} & {} & {}/{} & {}/{} \\\\".format(
                    condition_labels[condition],
                    DISPLAY_LABELS[controller],
                    _latex_value(stats["mean_episode_mean_ego_speed_mps"]),
                    _latex_value(stats["mean_episode_minimum_oriented_clearance_m"]),
                    int(stats["collision_count"]),
                    number_scenes,
                    int(stats["near_collision_count"]),
                    number_scenes,
                )
            )
        if condition_index == 0:
            lines.append(r"\midrule")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
        ]
    )
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines))


def analyze(
    results_root: Path,
    output_dir: Path,
    grid_start_s: float = -0.8,
    grid_end_s: float = 4.8,
    grid_step_s: float = 0.05,
    bootstrap_resamples: int = 10000,
    bootstrap_seed: int = 20260717,
    figure_stem: str = "figure6_carla_speed_clearance",
) -> Dict[str, Path]:
    """Validate the bank and write all profile-analysis artifacts."""

    if bootstrap_resamples < 10000:
        raise ProfileAnalysisError("at least 10,000 scene-block bootstrap resamples are required")
    root = results_root.expanduser().resolve()
    output = output_dir.expanduser().resolve()
    blocks = load_scene_blocks(root)
    if len(blocks) < 5:
        raise ProfileAnalysisError(
            "at least five complete physical-scene blocks are required for the descriptive "
            "bootstrap analysis; found {}".format(len(blocks))
        )
    grid = _common_grid(blocks, grid_start_s, grid_end_s, grid_step_s)
    output.mkdir(parents=True, exist_ok=True)
    profiles = _aligned_arrays(blocks, grid)
    rng = np.random.default_rng(bootstrap_seed)
    bootstrap_indices = rng.integers(
        0, len(blocks), size=(bootstrap_resamples, len(blocks)), endpoint=False
    )
    episode_rows = _episode_rows(blocks)
    aggregate = _aggregate_summary(
        blocks,
        episode_rows,
        grid,
        bootstrap_indices,
        bootstrap_seed,
        bootstrap_resamples,
        root,
    )
    aligned_rows = _aligned_rows(blocks, profiles, grid, bootstrap_indices)
    episode_path = output / "episode_metrics.csv"
    aligned_path = output / "aligned_profiles.csv"
    summary_path = output / "aggregate_summary.json"
    latex_path = output / "carla_figure6_results_table.tex"
    pdf_path = (output / figure_stem).with_suffix(".pdf")
    png_path = (output / figure_stem).with_suffix(".png")
    aggregate["artifacts"] = {
        "figure_pdf": str(pdf_path),
        "figure_png_300dpi": str(png_path),
        "aligned_profiles_csv": str(aligned_path),
        "episode_metrics_csv": str(episode_path),
        "latex_results_table": str(latex_path),
    }
    generated_pdf, generated_png = _plot_profiles(
        profiles, grid, bootstrap_indices, output / figure_stem
    )
    if generated_pdf != pdf_path or generated_png != png_path:
        raise ProfileAnalysisError("internal error: unexpected figure output paths")
    _write_csv(episode_path, episode_rows)
    _write_csv(aligned_path, aligned_rows)
    _write_json(summary_path, aggregate)
    _write_latex_table(latex_path, aggregate)
    return {
        "figure_pdf": pdf_path,
        "figure_png": png_path,
        "aligned_profiles_csv": aligned_path,
        "episode_metrics_csv": episode_path,
        "aggregate_summary_json": summary_path,
        "latex_table": latex_path,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        required=True,
        type=Path,
        help="root directory recursively containing run summary.json files",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="directory for the figure and analysis artifacts",
    )
    parser.add_argument("--grid-start-s", type=float, default=-0.8)
    parser.add_argument("--grid-end-s", type=float, default=4.8)
    parser.add_argument("--grid-step-s", type=float, default=0.05)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260717)
    parser.add_argument("--figure-stem", default="figure6_carla_speed_clearance")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        outputs = analyze(
            results_root=args.results_root,
            output_dir=args.output_dir,
            grid_start_s=args.grid_start_s,
            grid_end_s=args.grid_end_s,
            grid_step_s=args.grid_step_s,
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_seed=args.bootstrap_seed,
            figure_stem=args.figure_stem,
        )
    except ProfileAnalysisError as exc:
        print("ERROR: {}".format(exc))
        return 2
    print("Validated CARLA Figure 6 analysis outputs:")
    for label, path in outputs.items():
        print("  {}: {}".format(label, path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
