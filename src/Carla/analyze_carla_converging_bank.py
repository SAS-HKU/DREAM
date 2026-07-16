#!/usr/bin/env python3
"""Analyze complete four-arm CARLA converging-merge scene banks.

The analysis unit is a *physical scene construction*, identified by the
condition-blind SHA-256 hash emitted by the scenario generator.  Every scene
must contain exactly four valid runs::

    DREAM x {empty_shadow, true_threat}
    IDEAM x {empty_shadow, true_threat}

This program never treats repeated controller/condition runs as independent
samples.  It computes raw scene-block summaries, within-scene controller and
condition contrasts, and their difference-in-differences interaction.  A
non-parametric confidence interval is generated only when at least five
complete scene blocks are available, and resampling is performed over whole
scene blocks.  Inferential p-values are intentionally not produced.

Run directories, ``summary.json`` files, or a parent directory containing
multiple run directories may be supplied.  Publication-candidate figures use
the SciencePlots ``science`` style and show individual scene observations;
they are not mean-only bar charts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


RESULT_SCHEMA = "carla_overtaking_trial_result_v1"
BANK_SCHEMA = "carla_converging_scene_bank_analysis_v1"
CONTROLLERS = ("DREAM", "IDEAM")
CONDITIONS = ("empty_shadow", "true_threat")
ARMS = tuple((controller, condition) for controller in CONTROLLERS for condition in CONDITIONS)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class BankAnalysisError(ValueError):
    """Raised when inputs cannot support a valid scene-block analysis."""


@dataclass(frozen=True)
class RunSummary:
    summary_path: Path
    directory: Path
    summary: Dict[str, Any]
    resolved_manifest: Dict[str, Any]
    construction_hash: str
    construction_hash_recomputed: bool
    scene_seed: str
    generator_version: str
    scenario_family: str
    scenario_id: str
    controller: str
    condition: str
    duration_s: float
    metrics: Dict[str, Optional[float]]

    @property
    def arm(self) -> Tuple[str, str]:
        return self.controller, self.condition


@dataclass(frozen=True)
class SceneBlock:
    construction_hash: str
    scene_seed: str
    generator_version: str
    scenario_family: str
    scenario_id: str
    duration_s: float
    arms: Mapping[Tuple[str, str], RunSummary]


def _number(value: Any) -> Optional[float]:
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


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


def _resolve_summary_paths(inputs: Sequence[str]) -> List[Path]:
    resolved: Dict[Path, Path] = {}
    for raw in inputs:
        path = Path(raw).expanduser().resolve()
        if not path.exists():
            raise BankAnalysisError("input does not exist: {}".format(path))
        candidates: Iterable[Path]
        if path.is_file():
            if path.name != "summary.json":
                raise BankAnalysisError("input file must be summary.json: {}".format(path))
            candidates = (path,)
        elif (path / "summary.json").is_file():
            candidates = (path / "summary.json",)
        else:
            candidates = path.rglob("summary.json")
        found = False
        for candidate in candidates:
            found = True
            absolute = candidate.resolve()
            resolved[absolute] = absolute
        if not found:
            raise BankAnalysisError("no summary.json files found under {}".format(path))
    if not resolved:
        raise BankAnalysisError("no run summaries were supplied")
    return sorted(resolved)


def _read_json_object(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise BankAnalysisError("JSON root must be an object: {}".format(path))
    return value


def _resolved_manifest(summary: Mapping[str, Any], summary_path: Path) -> Dict[str, Any]:
    inline = summary.get("resolved_manifest")
    if isinstance(inline, Mapping):
        return dict(inline)
    candidates: List[Path] = []
    explicit = summary.get("resolved_manifest_path")
    if explicit:
        explicit_path = Path(str(explicit)).expanduser()
        candidates.append(
            explicit_path if explicit_path.is_absolute() else summary_path.parent / explicit_path
        )
    candidates.append(summary_path.parent / "resolved_manifest.json")
    for path in candidates:
        if path.is_file():
            return _read_json_object(path.resolve())
    return {}


def _identity_value(
    summary: Mapping[str, Any],
    manifest: Mapping[str, Any],
    top_key: str,
    construction_key: Optional[str] = None,
) -> Any:
    top = summary.get(top_key)
    manifest_value = manifest.get(top_key)
    if manifest_value is None and construction_key:
        manifest_value = _nested(manifest, "construction", construction_key)
    if top is not None and manifest_value is not None and str(top) != str(manifest_value):
        raise BankAnalysisError(
            "summary/manifest {} mismatch: {!r} != {!r}".format(
                top_key, top, manifest_value
            )
        )
    return top if top is not None else manifest_value


def _construction_hash(
    summary: Mapping[str, Any], manifest: Mapping[str, Any], summary_path: Path
) -> Tuple[str, bool]:
    candidates = {
        "summary.construction_hash": summary.get("construction_hash"),
        "resolved_manifest.construction.sha256": _nested(
            manifest, "construction", "sha256"
        ),
        "resolved_manifest.scene_construction.construction_hash_sha256": _nested(
            manifest, "scene_construction", "construction_hash_sha256"
        ),
    }
    declared = {
        label: str(value).lower()
        for label, value in candidates.items()
        if value is not None
    }
    if not declared:
        raise BankAnalysisError(
            "missing construction hash in summary and resolved manifest: {}".format(
                summary_path
            )
        )
    if len(set(declared.values())) != 1:
        raise BankAnalysisError(
            "summary construction_hash disagrees with resolved-manifest declaration: {}"
            .format(summary_path)
        )
    normalized = next(iter(declared.values()))
    if not SHA256_RE.fullmatch(normalized):
        raise BankAnalysisError("invalid SHA-256 construction hash in {}".format(summary_path))
    recomputed = False
    if _nested(manifest, "scene_construction", "construction_hash_sha256") is not None:
        # Use the generator's canonical payload definition instead of silently
        # inventing a second hash contract in the analyzer.
        try:
            from evaluation.carla_converging_scene import construction_hash
        except ImportError as error:
            try:
                from carla_converging_scene import construction_hash  # type: ignore
            except ImportError:
                raise BankAnalysisError(
                    "cannot import the converging-scene hash verifier"
                ) from error
        try:
            recalculated = str(construction_hash(manifest)).lower()
        except (KeyError, TypeError, ValueError) as error:
            raise BankAnalysisError(
                "cannot recompute construction hash from {}: {}".format(
                    summary_path.parent / "resolved_manifest.json", error
                )
            ) from error
        if recalculated != normalized:
            raise BankAnalysisError(
                "resolved-manifest content does not match its construction hash: {}"
                .format(summary_path)
            )
        recomputed = True
    return normalized, recomputed


def _follower_metrics(summary: Mapping[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    followers = summary.get("followers", {})
    if not isinstance(followers, Mapping):
        return None, None, None
    losses: List[float] = []
    decelerations: List[float] = []
    hard_brake_times: List[float] = []
    for follower in followers.values():
        if not isinstance(follower, Mapping):
            continue
        loss = _number(follower.get("maximum_speed_loss_mps"))
        deceleration = _number(follower.get("peak_deceleration_mps2"))
        hard_time = _number(follower.get("hard_brake_time_s"))
        if loss is not None:
            losses.append(loss)
        if deceleration is not None:
            decelerations.append(deceleration)
        if hard_time is not None:
            hard_brake_times.append(hard_time)
    return (
        max(losses) if losses else None,
        min(decelerations) if decelerations else None,
        max(hard_brake_times) if hard_brake_times else None,
    )


def _extract_metrics(summary: Mapping[str, Any]) -> Dict[str, Optional[float]]:
    follower_loss, follower_peak_decel, follower_hard_time = _follower_metrics(summary)
    planner_completed = _number(_nested(summary, "planner", "completed_requests"))
    planner_dropped = _number(_nested(summary, "planner", "dropped_requests"))
    planner_request_total = (
        None
        if planner_completed is None or planner_dropped is None
        else planner_completed + planner_dropped
    )
    planner_dropped_fraction = (
        None
        if planner_request_total is None or planner_request_total <= 0.0
        else planner_dropped / planner_request_total
    )
    wall_duration_s = _number(_nested(summary, "runtime", "wall_duration_s"))
    simulated_duration_s = _number(summary.get("duration_s"))
    if simulated_duration_s is None:
        simulated_duration_s = _number(
            _nested(summary, "runtime", "simulated_duration_s")
        )
    planner_wall_rate = (
        None
        if planner_completed is None or wall_duration_s is None or wall_duration_s <= 0.0
        else planner_completed / wall_duration_s
    )
    planner_simulated_rate = (
        None
        if (
            planner_completed is None
            or simulated_duration_s is None
            or simulated_duration_s <= 0.0
        )
        else planner_completed / simulated_duration_s
    )
    traffic_total_integrated_deficit = _number(
        _nested(
            summary,
            "traffic_disturbance",
            "total_integrated_speed_deficit_vehicle_m",
        )
    )
    if traffic_total_integrated_deficit is None:
        # Compatibility with pilot summaries produced before the dimensional
        # correction.  The old ``_m_s`` value was calculated as speed deficit
        # times seconds and is therefore numerically a distance aggregate.
        traffic_total_integrated_deficit = _number(
            _nested(
                summary,
                "traffic_disturbance",
                "total_integrated_speed_deficit_m_s",
            )
        )
    return {
        "ego_mean_speed_mps": _number(_nested(summary, "ego", "mean_speed_mps")),
        "ego_minimum_speed_mps": _number(_nested(summary, "ego", "minimum_speed_mps")),
        "ego_maximum_speed_loss_mps": _number(
            _nested(summary, "ego", "maximum_speed_loss_mps")
        ),
        "ego_peak_deceleration_mps2": _number(
            _nested(summary, "ego", "peak_deceleration_mps2")
        ),
        "maximum_follower_speed_loss_mps": follower_loss,
        "minimum_follower_peak_deceleration_mps2": follower_peak_decel,
        "maximum_follower_hard_brake_time_s": follower_hard_time,
        "collision_incidence": _number(summary.get("collision_incidence")),
        "near_collision_incidence": _number(summary.get("near_collision_incidence")),
        "minimum_oriented_box_clearance_m": _number(
            _nested(summary, "ego", "minimum_oriented_box_clearance_m")
        ),
        "minimum_hidden_oriented_box_clearance_m": _number(
            _nested(summary, "ego", "minimum_hidden_oriented_box_clearance_m")
        ),
        "minimum_ttc_2d_s": _number(_nested(summary, "ego", "minimum_ttc_2d_s")),
        "minimum_hidden_ttc_2d_s": _number(
            _nested(summary, "ego", "minimum_hidden_ttc_2d_s")
        ),
        "planner_mean_total_s": _number(_nested(summary, "planner", "mean_total_s")),
        "planner_p95_total_s": _number(_nested(summary, "planner", "p95_total_s")),
        "planner_maximum_total_s": _number(
            _nested(summary, "planner", "maximum_total_s")
        ),
        "reveal_to_hidden_aware_plan_applied_s": _number(
            _nested(summary, "planner", "reveal_to_hidden_aware_plan_applied_s")
        ),
        "planner_completed_requests": planner_completed,
        "planner_dropped_requests": planner_dropped,
        "planner_dropped_request_fraction": planner_dropped_fraction,
        "planner_wall_clock_effective_rate_hz": planner_wall_rate,
        "planner_simulated_time_effective_rate_hz": planner_simulated_rate,
        "low_level_mean_time_s": _number(_nested(summary, "low_level", "mean_time_s")),
        "low_level_p95_time_s": _number(_nested(summary, "low_level", "p95_time_s")),
        "low_level_maximum_time_s": _number(
            _nested(summary, "low_level", "maximum_time_s")
        ),
        "low_level_deadline_miss_fraction": _number(
            _nested(summary, "low_level", "deadline_miss_fraction")
        ),
        "low_level_stale_plan_fallback_time_s": _number(
            _nested(summary, "low_level", "stale_plan_fallback_time_s")
        ),
        "low_level_wall_clock_effective_control_rate_hz": _number(
            _nested(summary, "low_level", "wall_clock_effective_control_rate_hz")
        ),
        "physics_control_loop_mean_cycle_time_s": _number(
            _nested(summary, "physics_control_loop", "mean_cycle_time_s")
        ),
        "physics_control_loop_p95_cycle_time_s": _number(
            _nested(summary, "physics_control_loop", "p95_cycle_time_s")
        ),
        "physics_control_loop_maximum_cycle_time_s": _number(
            _nested(summary, "physics_control_loop", "maximum_cycle_time_s")
        ),
        "physics_control_loop_deadline_miss_fraction": _number(
            _nested(summary, "physics_control_loop", "deadline_miss_fraction")
        ),
        "traffic_maximum_speed_loss_mps": _number(
            _nested(summary, "traffic_disturbance", "maximum_speed_loss_mps")
        ),
        "traffic_total_integrated_speed_deficit_vehicle_m": (
            traffic_total_integrated_deficit
        ),
        "traffic_peak_deceleration_mps2": _number(
            _nested(summary, "traffic_disturbance", "peak_deceleration_mps2")
        ),
        "traffic_hard_braking_actor_count": _number(
            _nested(summary, "traffic_disturbance", "hard_braking_actor_count")
        ),
        "traffic_maximum_follower_disturbance_amplification": _number(
            _nested(
                summary,
                "traffic_disturbance",
                "maximum_follower_disturbance_amplification",
            )
        ),
        "real_time_factor": _number(_nested(summary, "runtime", "real_time_factor")),
    }


def _load_run(summary_path: Path) -> RunSummary:
    summary = _read_json_object(summary_path)
    if summary.get("schema_version") != RESULT_SCHEMA:
        raise BankAnalysisError("unsupported result schema in {}".format(summary_path))
    qualification = summary.get("qualification", {})
    required_flags = (
        "valid_for_analysis",
        "seed_varies_physical_construction",
        "statistical_bank_ready",
    )
    failures = [key for key in required_flags if qualification.get(key) is not True]
    if failures:
        raise BankAnalysisError(
            "run is not eligible for statistical-bank analysis ({}): {}".format(
                ", ".join(failures), summary_path
            )
        )
    controller = str(summary.get("controller", "")).upper()
    condition = str(summary.get("condition", "")).lower()
    if controller not in CONTROLLERS or condition not in CONDITIONS:
        raise BankAnalysisError(
            "unsupported controller/condition arm {!r}/{!r}: {}".format(
                controller, condition, summary_path
            )
        )
    scenario_id = str(summary.get("scenario_id", ""))
    if not scenario_id:
        raise BankAnalysisError("missing scenario_id: {}".format(summary_path))
    duration_s = _number(summary.get("duration_s"))
    if duration_s is None:
        duration_s = _number(_nested(summary, "runtime", "simulated_duration_s"))
    if duration_s is None or duration_s <= 0.0:
        raise BankAnalysisError("missing/invalid duration_s: {}".format(summary_path))
    manifest = _resolved_manifest(summary, summary_path)
    manifest_scenario_id = manifest.get("scenario_id")
    if manifest_scenario_id is not None and str(manifest_scenario_id) != scenario_id:
        raise BankAnalysisError(
            "summary/resolved-manifest scenario_id mismatch: {}".format(summary_path)
        )
    manifest_duration = _number(manifest.get("duration_s"))
    if manifest_duration is not None and abs(manifest_duration - duration_s) > 1e-6:
        raise BankAnalysisError(
            "summary/resolved-manifest duration_s mismatch: {}".format(summary_path)
        )
    scenario_family = summary.get("scenario_family")
    manifest_scenario_family = manifest.get("scenario_family")
    if (
        scenario_family is not None
        and manifest_scenario_family is not None
        and str(scenario_family) != str(manifest_scenario_family)
    ):
        raise BankAnalysisError(
            "summary/resolved-manifest scenario_family mismatch: {}".format(summary_path)
        )
    if scenario_family is None:
        scenario_family = manifest_scenario_family
    if scenario_family is None:
        # Legacy fixtures/manifests used a bank-wide scenario_id.  The current
        # generator emits an explicit scenario_family and a seed-specific id.
        scenario_family = scenario_id
    construction_hash, construction_hash_recomputed = _construction_hash(
        summary, manifest, summary_path
    )
    scene_seed = _identity_value(summary, manifest, "scene_seed", "scene_seed")
    generator_version = _identity_value(
        summary, manifest, "generator_version", "generator_version"
    )
    scene_seed_nested = _nested(manifest, "scene_construction", "seed")
    generator_nested = _nested(manifest, "scene_construction", "generator_version")
    if scene_seed is not None and scene_seed_nested is not None and str(scene_seed) != str(scene_seed_nested):
        raise BankAnalysisError("summary/manifest scene_seed mismatch: {}".format(summary_path))
    if generator_version is not None and generator_nested is not None and str(generator_version) != str(generator_nested):
        raise BankAnalysisError(
            "summary/manifest generator_version mismatch: {}".format(summary_path)
        )
    if scene_seed is None:
        scene_seed = scene_seed_nested
    if generator_version is None:
        generator_version = generator_nested
    if scene_seed is None:
        raise BankAnalysisError("missing scene_seed: {}".format(summary_path))
    if generator_version is None or not str(generator_version).strip():
        raise BankAnalysisError("missing generator_version: {}".format(summary_path))
    metrics = _extract_metrics(summary)
    essential = (
        "ego_mean_speed_mps",
        "maximum_follower_speed_loss_mps",
        "collision_incidence",
        "near_collision_incidence",
        "minimum_oriented_box_clearance_m",
        "planner_mean_total_s",
        "low_level_mean_time_s",
        "low_level_deadline_miss_fraction",
        "real_time_factor",
    )
    missing = [key for key in essential if metrics[key] is None]
    if missing:
        raise BankAnalysisError(
            "summary lacks essential bank metrics {}: {}".format(
                ", ".join(missing), summary_path
            )
        )
    return RunSummary(
        summary_path=summary_path,
        directory=summary_path.parent,
        summary=summary,
        resolved_manifest=manifest,
        construction_hash=construction_hash,
        construction_hash_recomputed=construction_hash_recomputed,
        scene_seed=str(scene_seed),
        generator_version=str(generator_version),
        scenario_family=str(scenario_family),
        scenario_id=scenario_id,
        controller=controller,
        condition=condition,
        duration_s=duration_s,
        metrics=metrics,
    )


def _assemble_blocks(runs: Sequence[RunSummary]) -> List[SceneBlock]:
    grouped: Dict[str, List[RunSummary]] = {}
    for run in runs:
        grouped.setdefault(run.construction_hash, []).append(run)
    blocks: List[SceneBlock] = []
    seed_to_hash: Dict[str, str] = {}
    for construction_hash, group in sorted(grouped.items()):
        arms: Dict[Tuple[str, str], RunSummary] = {}
        for run in group:
            if run.arm in arms:
                raise BankAnalysisError(
                    "duplicate {} / {} arm for construction {}".format(
                        run.controller, run.condition, construction_hash
                    )
                )
            arms[run.arm] = run
        missing = ["{}/{}".format(*arm) for arm in ARMS if arm not in arms]
        extra = ["{}/{}".format(*arm) for arm in arms if arm not in ARMS]
        if len(group) != 4 or missing or extra:
            raise BankAnalysisError(
                "construction {} is not a complete four-arm block; missing=[{}], extra=[{}]"
                .format(construction_hash, ", ".join(missing), ", ".join(extra))
            )
        seeds = {run.scene_seed for run in group}
        generators = {run.generator_version for run in group}
        families = {run.scenario_family for run in group}
        scenarios = {run.scenario_id for run in group}
        durations = {round(run.duration_s, 6) for run in group}
        if (
            len(seeds) != 1
            or len(generators) != 1
            or len(families) != 1
            or len(scenarios) != 1
            or len(durations) != 1
        ):
            raise BankAnalysisError(
                "construction {} has inconsistent seed/generator/scenario/duration metadata"
                .format(construction_hash)
            )
        scene_seed = next(iter(seeds))
        previous_hash = seed_to_hash.get(scene_seed)
        if previous_hash is not None and previous_hash != construction_hash:
            raise BankAnalysisError(
                "scene_seed {} maps to multiple physical construction hashes".format(scene_seed)
            )
        seed_to_hash[scene_seed] = construction_hash
        blocks.append(
            SceneBlock(
                construction_hash=construction_hash,
                scene_seed=scene_seed,
                generator_version=next(iter(generators)),
                scenario_family=next(iter(families)),
                scenario_id=next(iter(scenarios)),
                duration_s=next(iter(durations)),
                arms=arms,
            )
        )
    if not blocks:
        raise BankAnalysisError("no complete scene blocks were found")
    if len({block.generator_version for block in blocks}) != 1:
        raise BankAnalysisError("generator_version differs across scene blocks")
    if len({block.scenario_family for block in blocks}) != 1:
        raise BankAnalysisError("scenario_family differs across scene blocks")
    if len({block.construction_hash for block in blocks}) != len(blocks):
        raise BankAnalysisError("construction hashes are not unique")
    return sorted(blocks, key=lambda item: (item.scene_seed, item.construction_hash))


def _difference(left: Optional[float], right: Optional[float]) -> Optional[float]:
    return None if left is None or right is None else float(left) - float(right)


def _block_row(block: SceneBlock) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "construction_hash": block.construction_hash,
        "scene_seed": block.scene_seed,
        "generator_version": block.generator_version,
        "scenario_family": block.scenario_family,
        "scenario_id": block.scenario_id,
        "duration_s": block.duration_s,
    }
    metric_names = sorted(
        {key for run in block.arms.values() for key in run.metrics}
    )
    for controller, condition in ARMS:
        run = block.arms[(controller, condition)]
        prefix = "{}__{}".format(controller.lower(), condition)
        row[prefix + "__run_directory"] = str(run.directory)
        for metric in metric_names:
            row[prefix + "__" + metric] = run.metrics.get(metric)
    for metric in metric_names:
        dream_empty = block.arms[("DREAM", "empty_shadow")].metrics.get(metric)
        dream_true = block.arms[("DREAM", "true_threat")].metrics.get(metric)
        ideam_empty = block.arms[("IDEAM", "empty_shadow")].metrics.get(metric)
        ideam_true = block.arms[("IDEAM", "true_threat")].metrics.get(metric)
        empty_controller = _difference(dream_empty, ideam_empty)
        true_controller = _difference(dream_true, ideam_true)
        dream_condition = _difference(dream_true, dream_empty)
        ideam_condition = _difference(ideam_true, ideam_empty)
        interaction = _difference(dream_condition, ideam_condition)
        row["empty_shadow__dream_minus_ideam__" + metric] = empty_controller
        row["true_threat__dream_minus_ideam__" + metric] = true_controller
        row["dream__true_minus_empty__" + metric] = dream_condition
        row["ideam__true_minus_empty__" + metric] = ideam_condition
        row["interaction__" + metric] = interaction
    speed_metric = "ego_mean_speed_mps"
    row["empty_shadow_ct_v_mps"] = -float(
        row["empty_shadow__dream_minus_ideam__" + speed_metric]
    )
    row["true_threat_ct_v_mps"] = -float(
        row["true_threat__dream_minus_ideam__" + speed_metric]
    )
    row["ct_v_condition_interaction_mps"] = (
        row["true_threat_ct_v_mps"] - row["empty_shadow_ct_v_mps"]
    )
    return row


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("percentile requires at least one value")
    position = (len(sorted_values) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction)


def _bootstrap_mean_ci(
    values: Sequence[float], replicates: int, rng: random.Random
) -> Tuple[Optional[float], Optional[float]]:
    if len(values) < 5:
        return None, None
    n = len(values)
    means = sorted(
        statistics.fmean(values[rng.randrange(n)] for _ in range(n))
        for _ in range(replicates)
    )
    return _percentile(means, 0.025), _percentile(means, 0.975)


def _statistic_row(
    *,
    scope: str,
    metric: str,
    values: Sequence[float],
    n_blocks: int,
    replicates: int,
    rng: random.Random,
    controller: Optional[str] = None,
    condition: Optional[str] = None,
    contrast: Optional[str] = None,
) -> Dict[str, Any]:
    # Preserve the declared resampling unit.  A partially observed metric is
    # summarized descriptively rather than bootstrapping a selected subset of
    # scene blocks (notably relevant for horizon-censored TTC values).
    if len(values) == n_blocks:
        lower, upper = _bootstrap_mean_ci(values, replicates, rng)
    else:
        lower, upper = None, None
    if len(values) != n_blocks:
        inference_label = "incomplete_or_censored_descriptive_only"
    elif lower is not None:
        inference_label = "scene_block_bootstrap_ci"
    else:
        inference_label = "descriptive_only"
    return {
        "scope": scope,
        "controller": controller,
        "condition": condition,
        "contrast": contrast,
        "metric": metric,
        "n_scene_blocks": n_blocks,
        "n_available": len(values),
        "n_missing_or_censored": n_blocks - len(values),
        "mean": statistics.fmean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
        "bootstrap_mean_ci95_lower": lower,
        "bootstrap_mean_ci95_upper": upper,
        "inference_label": inference_label,
    }


def _raw_statistics(
    blocks: Sequence[SceneBlock], replicates: int, seed: int
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    rng = random.Random(seed)
    metric_names = sorted({key for block in blocks for run in block.arms.values() for key in run.metrics})
    for controller, condition in ARMS:
        for metric in metric_names:
            values = [
                float(block.arms[(controller, condition)].metrics[metric])
                for block in blocks
                if block.arms[(controller, condition)].metrics.get(metric) is not None
            ]
            if values:
                rows.append(
                    _statistic_row(
                        scope="raw_arm",
                        controller=controller,
                        condition=condition,
                        contrast=None,
                        metric=metric,
                        values=values,
                        n_blocks=len(blocks),
                        replicates=replicates,
                        rng=rng,
                    )
                )
    return rows


def _effect_statistics(
    block_rows: Sequence[Mapping[str, Any]], replicates: int, seed: int
) -> List[Dict[str, Any]]:
    identity = {
        "construction_hash",
        "scene_seed",
        "generator_version",
        "scenario_family",
        "scenario_id",
        "duration_s",
    }
    keys = sorted(
        key
        for key in block_rows[0]
        if key not in identity
        and "__run_directory" not in key
        and (
            "dream_minus_ideam" in key
            or "true_minus_empty" in key
            or key.startswith("interaction__")
            or key.endswith("ct_v_mps")
            or key == "ct_v_condition_interaction_mps"
        )
    )
    rng = random.Random(seed + 1)
    rows: List[Dict[str, Any]] = []
    for key in keys:
        values = [
            float(row[key])
            for row in block_rows
            if isinstance(row.get(key), (int, float))
            and not isinstance(row.get(key), bool)
            and math.isfinite(float(row[key]))
        ]
        if values:
            rows.append(
                _statistic_row(
                    scope="paired_effect",
                    metric=key,
                    values=values,
                    n_blocks=len(block_rows),
                    replicates=replicates,
                    rng=rng,
                    contrast=key,
                )
            )
    return rows


def _strip_offsets(count: int, width: float = 0.10) -> List[float]:
    if count <= 1:
        return [0.0] * count
    return [(-width + 2.0 * width * index / (count - 1)) for index in range(count)]


def _mean_ci_for_plot(
    values: Sequence[float], replicates: int, seed: int
) -> Tuple[float, Optional[float], Optional[float]]:
    mean = statistics.fmean(values)
    lower, upper = _bootstrap_mean_ci(values, replicates, random.Random(seed))
    return mean, lower, upper


def _paired_effect_panel(
    ax: Any,
    series: Sequence[Sequence[float]],
    labels: Sequence[str],
    colors: Sequence[str],
    markers: Sequence[str],
    ylabel: str,
    title: str,
    replicates: int,
    seed: int,
) -> None:
    count = min((len(values) for values in series), default=0)
    if count == 0:
        ax.text(0.5, 0.5, "Metric unavailable", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, loc="left")
        return
    offsets = _strip_offsets(count)
    for index in range(count):
        ax.plot(
            [position + offsets[index] for position in range(len(series))],
            [values[index] for values in series],
            color="#A7A9AC",
            linewidth=0.6,
            alpha=0.45,
            zorder=1,
        )
    for position, (values, color, marker) in enumerate(zip(series, colors, markers)):
        ax.scatter(
            [position + offset for offset in offsets],
            list(values)[:count],
            s=22,
            color=color,
            marker=marker,
            edgecolor="#202124",
            linewidth=0.35,
            alpha=0.82,
            zorder=2,
        )
        mean, lower, upper = _mean_ci_for_plot(list(values)[:count], replicates, seed + position)
        if lower is not None and upper is not None:
            ax.errorbar(
                position,
                mean,
                yerr=[[mean - lower], [upper - mean]],
                fmt="D",
                color="#111111",
                markersize=4.0,
                capsize=2.5,
                linewidth=1.0,
                zorder=4,
            )
        else:
            ax.plot(position, statistics.median(list(values)[:count]), marker="_", color="#111111", markersize=9, zorder=4)
    ax.axhline(0.0, color="#4B4F52", linewidth=0.7, linestyle="--", zorder=0)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")


def _raw_distribution_panel(
    ax: Any,
    groups: Sequence[Sequence[float]],
    labels: Sequence[str],
    colors: Sequence[str],
    ylabel: str,
    title: str,
) -> None:
    positions = list(range(len(groups)))
    if all(len(values) >= 5 for values in groups):
        artists = ax.boxplot(
            groups,
            positions=positions,
            widths=0.48,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#111111", "linewidth": 1.2},
            whiskerprops={"color": "#55585A", "linewidth": 0.8},
            capprops={"color": "#55585A", "linewidth": 0.8},
            boxprops={"edgecolor": "#55585A", "linewidth": 0.8},
        )
        for patch, color in zip(artists["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.16)
    for position, (values, color) in enumerate(zip(groups, colors)):
        offsets = _strip_offsets(len(values), 0.12)
        marker = "o" if position % 2 == 0 else "s"
        ax.scatter(
            [position + value for value in offsets],
            values,
            s=21,
            color=color,
            marker=marker,
            edgecolor="#202124",
            linewidth=0.35,
            alpha=0.80,
            zorder=3,
        )
        if len(values) < 5:
            ax.plot(position, statistics.median(values), marker="_", color="#111111", markersize=9, zorder=4)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")


def _render_figures(
    blocks: Sequence[SceneBlock],
    block_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> Dict[str, str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import scienceplots  # noqa: F401  # registers SciencePlots styles
    except ImportError as error:
        raise BankAnalysisError(
            "figure generation requires matplotlib and the SciencePlots package"
        ) from error

    blue = "#0C5DA5"
    gold = "#FFB000"
    orange = "#FF9500"
    charcoal = "#3B3B3B"
    n = len(block_rows)

    def values(key: str) -> List[float]:
        return [float(row[key]) for row in block_rows if _number(row.get(key)) is not None]

    with plt.style.context(["science", "no-latex"]):
        plt.rcParams.update(
            {
                "font.size": 8.0,
                "axes.titlesize": 8.5,
                "axes.labelsize": 8.0,
                "xtick.labelsize": 7.2,
                "ytick.labelsize": 7.2,
                "legend.fontsize": 7.0,
                "figure.dpi": 150,
                "savefig.dpi": 400,
                "axes.spines.top": False,
                "axes.spines.right": False,
            }
        )
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.25))
        _paired_effect_panel(
            axes[0, 0],
            [values("empty_shadow_ct_v_mps"), values("true_threat_ct_v_mps")],
            ["Empty shadow", "True threat"],
            [blue, orange],
            ["o", "s"],
            r"$CT_v$ (m s$^{-1}$; IDEAM $-$ DREAM)",
            "A  Ego-speed conservatism tax",
            bootstrap_replicates,
            bootstrap_seed,
        )
        global_clearance = values(
            "true_threat__dream_minus_ideam__minimum_oriented_box_clearance_m"
        )
        hidden_clearance = values(
            "true_threat__dream_minus_ideam__minimum_hidden_oriented_box_clearance_m"
        )
        clearance_series = [global_clearance]
        clearance_labels = ["All actors"]
        clearance_colors = [blue]
        clearance_markers = ["o"]
        if len(hidden_clearance) == n:
            clearance_series.append(hidden_clearance)
            clearance_labels.append("Hidden actor")
            clearance_colors.append(gold)
            clearance_markers.append("s")
        _paired_effect_panel(
            axes[0, 1],
            clearance_series,
            clearance_labels,
            clearance_colors,
            clearance_markers,
            r"Clearance difference (m; DREAM $-$ IDEAM)",
            "B  True-threat safety margins",
            bootstrap_replicates,
            bootstrap_seed + 20,
        )
        _paired_effect_panel(
            axes[1, 0],
            [
                values("empty_shadow__dream_minus_ideam__maximum_follower_speed_loss_mps"),
                values("true_threat__dream_minus_ideam__maximum_follower_speed_loss_mps"),
            ],
            ["Empty shadow", "True threat"],
            [blue, orange],
            ["o", "s"],
            r"Follower speed-loss difference (m s$^{-1}$)",
            "C  Downstream disturbance",
            bootstrap_replicates,
            bootstrap_seed + 40,
        )
        _paired_effect_panel(
            axes[1, 1],
            [
                values("empty_shadow__dream_minus_ideam__planner_mean_total_s"),
                values("true_threat__dream_minus_ideam__planner_mean_total_s"),
            ],
            ["Empty shadow", "True threat"],
            [blue, orange],
            ["o", "s"],
            r"Planning-time difference (s; DREAM $-$ IDEAM)",
            "D  High-level planning time",
            bootstrap_replicates,
            bootstrap_seed + 60,
        )
        fig.suptitle(
            "Paired safety, efficiency, and runtime effects",
            y=0.995,
            fontsize=9.2,
        )
        fig.tight_layout(rect=(0.02, 0.02, 1.0, 0.965), h_pad=1.35, w_pad=1.2)
        effect_png = output_dir / "fig_carla_bank_paired_effects.png"
        effect_pdf = output_dir / "fig_carla_bank_paired_effects.pdf"
        fig.savefig(effect_png, bbox_inches="tight", facecolor="white")
        fig.savefig(effect_pdf, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        arm_order = (
            ("DREAM", "empty_shadow"),
            ("IDEAM", "empty_shadow"),
            ("DREAM", "true_threat"),
            ("IDEAM", "true_threat"),
        )
        arm_labels = ["D-E", "I-E", "D-T", "I-T"]
        arm_colors = [blue, gold, blue, gold]

        def arm_values(metric: str, selected: Sequence[Tuple[str, str]] = arm_order) -> List[List[float]]:
            return [
                [
                    float(block.arms[arm].metrics[metric])
                    for block in blocks
                    if block.arms[arm].metrics.get(metric) is not None
                ]
                for arm in selected
            ]

        fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.25))
        _raw_distribution_panel(
            axes[0, 0],
            arm_values("ego_mean_speed_mps"),
            arm_labels,
            arm_colors,
            r"Time-mean ego speed (m s$^{-1}$)",
            "A  Ego speed",
        )
        true_arms = (("DREAM", "true_threat"), ("IDEAM", "true_threat"))
        _raw_distribution_panel(
            axes[0, 1],
            arm_values("minimum_oriented_box_clearance_m", true_arms),
            ["DREAM", "IDEAM"],
            [blue, gold],
            "Minimum oriented-box clearance (m)",
            "B  True-threat clearance",
        )
        _raw_distribution_panel(
            axes[1, 0],
            arm_values("maximum_follower_speed_loss_mps"),
            arm_labels,
            arm_colors,
            r"Maximum follower speed loss (m s$^{-1}$)",
            "C  Follower disturbance",
        )
        _raw_distribution_panel(
            axes[1, 1],
            arm_values("planner_mean_total_s"),
            arm_labels,
            arm_colors,
            "Mean high-level planning time (s)",
            "D  Planning latency",
        )
        fig.suptitle("Matched-block outcome distributions", y=0.995, fontsize=9.2)
        fig.tight_layout(rect=(0.02, 0.02, 1.0, 0.965), h_pad=1.35, w_pad=1.2)
        raw_png = output_dir / "fig_carla_bank_outcome_distributions.png"
        raw_pdf = output_dir / "fig_carla_bank_outcome_distributions.pdf"
        fig.savefig(raw_png, bbox_inches="tight", facecolor="white")
        fig.savefig(raw_pdf, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    return {
        "paired_effects_png": str(effect_png.resolve()),
        "paired_effects_pdf": str(effect_pdf.resolve()),
        "outcome_distributions_png": str(raw_png.resolve()),
        "outcome_distributions_pdf": str(raw_pdf.resolve()),
        "style": "SciencePlots: science + no-latex",
    }


def analyze_bank(
    inputs: Sequence[str],
    output_dir: Path,
    *,
    bootstrap_replicates: int = 5000,
    bootstrap_seed: int = 20260716,
    make_figures: bool = True,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Validate, summarize, and optionally plot a four-arm CARLA scene bank."""
    if bootstrap_replicates < 100:
        raise BankAnalysisError("bootstrap_replicates must be at least 100")
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise BankAnalysisError(
            "output directory is non-empty; choose a new directory or pass --overwrite: {}"
            .format(output_dir)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = [_load_run(path) for path in _resolve_summary_paths(inputs)]
    blocks = _assemble_blocks(runs)
    block_rows = [_block_row(block) for block in blocks]
    run_rows: List[Dict[str, Any]] = []
    for block in blocks:
        for controller, condition in ARMS:
            run = block.arms[(controller, condition)]
            row: Dict[str, Any] = {
                "construction_hash": block.construction_hash,
                "scene_seed": block.scene_seed,
                "generator_version": block.generator_version,
                "scenario_family": block.scenario_family,
                "scenario_id": block.scenario_id,
                "controller": controller,
                "condition": condition,
                "duration_s": block.duration_s,
                "run_directory": str(run.directory),
            }
            row.update(run.metrics)
            run_rows.append(row)
    raw_statistics = _raw_statistics(blocks, bootstrap_replicates, bootstrap_seed)
    effect_statistics = _effect_statistics(
        block_rows, bootstrap_replicates, bootstrap_seed
    )
    figure_outputs: Dict[str, str] = {}
    if make_figures:
        figure_outputs = _render_figures(
            blocks,
            block_rows,
            output_dir,
            bootstrap_replicates,
            bootstrap_seed,
        )
    bootstrap_performed = len(blocks) >= 5
    payload: Dict[str, Any] = {
        "schema_version": BANK_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scenario_family": blocks[0].scenario_family,
        "scenario_ids": [block.scenario_id for block in blocks],
        "generator_version": blocks[0].generator_version,
        "n_scene_blocks": len(blocks),
        "n_runs": len(runs),
        "complete_four_arm_blocks": True,
        "construction_hash_verification": {
            "algorithm": "sha256",
            "all_well_formed": True,
            "all_unique_across_scene_blocks": True,
            "summary_matches_resolved_manifest_when_both_present": True,
            "n_runs_recomputed_from_canonical_manifest_payload": sum(
                1 for run in runs if run.construction_hash_recomputed
            ),
            "all_runs_recomputed_from_canonical_manifest_payload": all(
                run.construction_hash_recomputed for run in runs
            ),
        },
        "analysis_unit": "independently generated physical scene construction",
        "bootstrap": {
            "performed": bootstrap_performed,
            "minimum_scene_blocks": 5,
            "resampling_unit": "complete four-arm scene block",
            "replicates": bootstrap_replicates if bootstrap_performed else 0,
            "random_seed": bootstrap_seed if bootstrap_performed else None,
            "interval": "percentile 95% interval for the scene-block mean"
            if bootstrap_performed
            else None,
        },
        "inferential_p_values_performed": False,
        "evidence_label": "scene_block_bootstrap_descriptive"
        if bootstrap_performed
        else "small_bank_descriptive_only",
        "definitions": {
            "controller_effect": "DREAM minus IDEAM within the same scene and condition.",
            "condition_effect": "True-threat minus empty-shadow within the same scene and controller.",
            "interaction": (
                "(DREAM_true - DREAM_empty) - (IDEAM_true - IDEAM_empty) within each scene."
            ),
            "ct_v_mps": (
                "IDEAM time-mean ego speed minus DREAM time-mean ego speed in the same scene "
                "and condition; positive values indicate a DREAM speed tax."
            ),
            "ttc_missingness": (
                "Null TTC denotes no finite predicted contact within the evaluator horizon. "
                "Finite-TTC summaries report n_available and n_missing_or_censored."
            ),
            "traffic_disturbance": (
                "Scene-level traffic-stream metrics emitted by the runner: maximum speed "
                "loss, the sum of actor-wise integrated desired-speed deficits in "
                "vehicle-metres, most "
                "negative acceleration, number of hard-braking actors, and maximum "
                "successive-follower speed-loss amplification."
            ),
            "physics_control_loop": (
                "Wall time for one complete synchronous physics-loop iteration; deadline "
                "misses use the manifest physics step as the deadline."
            ),
            "planner_dropped_request_fraction": (
                "dropped_requests / (completed_requests + dropped_requests); null when the "
                "denominator is zero. This is a latest-only scheduler diagnostic."
            ),
            "planner_effective_rate": (
                "Completed planner responses divided by recorded wall-clock or simulated "
                "episode duration. The count follows the runner's completed_requests "
                "definition."
            ),
        },
        "scene_blocks": block_rows,
        "raw_arm_statistics": raw_statistics,
        "paired_effect_statistics": effect_statistics,
        "outputs": {
            "run_metrics_csv": str((output_dir / "run_metrics.csv").resolve()),
            "scene_block_metrics_csv": str(
                (output_dir / "scene_block_metrics.csv").resolve()
            ),
            "raw_arm_statistics_csv": str(
                (output_dir / "raw_arm_statistics.csv").resolve()
            ),
            "paired_effect_statistics_csv": str(
                (output_dir / "paired_effect_statistics.csv").resolve()
            ),
            **figure_outputs,
        },
    }
    _write_csv(output_dir / "run_metrics.csv", run_rows)
    _write_csv(output_dir / "scene_block_metrics.csv", block_rows)
    _write_csv(output_dir / "raw_arm_statistics.csv", raw_statistics)
    _write_csv(output_dir / "paired_effect_statistics.csv", effect_statistics)
    _write_json(output_dir / "bank_analysis.json", payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Run directories, summary.json files, or parent directories containing runs.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260716)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = analyze_bank(
            args.inputs,
            args.output_dir,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
            make_figures=not args.no_figures,
            overwrite=args.overwrite,
        )
    except (BankAnalysisError, OSError, json.JSONDecodeError) as error:
        print("bank analysis failed: {}".format(error), file=sys.stderr)
        return 2
    print(
        "wrote {} complete scene block(s) to {} [{}; p-values: none]".format(
            result["n_scene_blocks"],
            args.output_dir.resolve(),
            result["evidence_label"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
