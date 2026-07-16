import hashlib
import json
import csv
from pathlib import Path

import pytest

from analyze_carla_converging_bank import BankAnalysisError, analyze_bank
from carla_converging_scene import resolve_file


def _scene_hash(seed: int) -> str:
    return hashlib.sha256("fixture-scene-{}".format(seed).encode("utf-8")).hexdigest()


def _write_arm(
    root: Path,
    seed: int,
    controller: str,
    condition: str,
    *,
    manifest_hash: str = "",
    top_hash: str = "",
) -> Path:
    construction_hash = manifest_hash or _scene_hash(seed)
    top_hash = top_hash or construction_hash
    directory = root / "seed_{}".format(seed) / "{}_{}".format(
        controller.lower(), condition
    )
    directory.mkdir(parents=True)
    controller_offset = 0.0 if controller == "IDEAM" else -2.0
    threat_offset = -0.5 if condition == "true_threat" else 0.0
    if controller == "DREAM" and condition == "true_threat":
        threat_offset -= 0.5
    ego_speed = 30.0 + 0.01 * seed + controller_offset + threat_offset
    follower_loss = (
        0.5
        + 0.05 * seed
        + (0.5 if controller == "DREAM" else 0.0)
        + (0.2 if condition == "true_threat" else 0.0)
    )
    clearance = (
        1.2
        + 0.01 * seed
        + (0.3 if controller == "DREAM" else 0.0)
        + (0.1 if condition == "empty_shadow" else 0.0)
    )
    planner_time = 0.30 + (0.20 if controller == "DREAM" else 0.0) + 0.002 * seed
    planner_completed = 12 if controller == "DREAM" else 24
    planner_dropped = 108 if controller == "DREAM" else 96
    followers = {
        "follower_1": {
            "maximum_speed_loss_mps": follower_loss,
            "peak_deceleration_mps2": -0.8 - 0.01 * seed,
            "hard_brake_time_s": 0.0,
        },
        "follower_2": {
            "maximum_speed_loss_mps": follower_loss - 0.1,
            "peak_deceleration_mps2": -0.7,
            "hard_brake_time_s": 0.0,
        },
    }
    summary = {
        "schema_version": "carla_overtaking_trial_result_v1",
        "scenario_id": "fixture_converging_merge_v1",
        "construction_hash": top_hash,
        "scene_seed": seed,
        "generator_version": "fixture-generator-v1",
        "condition": condition,
        "controller": controller,
        "duration_s": 12.0,
        "collision_incidence": 0,
        "near_collision_incidence": int(
            controller == "IDEAM" and condition == "true_threat" and seed % 2 == 0
        ),
        "qualification": {
            "valid_for_analysis": True,
            "seed_varies_physical_construction": True,
            "statistical_bank_ready": True,
        },
        "ego": {
            "mean_speed_mps": ego_speed,
            "minimum_speed_mps": ego_speed - 1.0,
            "maximum_speed_loss_mps": 31.0 - ego_speed,
            "peak_deceleration_mps2": -1.5,
            "minimum_oriented_box_clearance_m": clearance,
            "minimum_hidden_oriented_box_clearance_m": (
                clearance + 0.2 if condition == "true_threat" else None
            ),
            "minimum_ttc_2d_s": 2.5,
            "minimum_hidden_ttc_2d_s": 2.8 if condition == "true_threat" else None,
        },
        "followers": followers,
        "planner": {
            "completed_requests": planner_completed,
            "dropped_requests": planner_dropped,
            "mean_total_s": planner_time,
            "p95_total_s": planner_time + 0.1,
            "maximum_total_s": planner_time + 0.2,
            "reveal_to_hidden_aware_plan_applied_s": (
                0.8 if condition == "true_threat" else None
            ),
        },
        "low_level": {
            "mean_time_s": 0.001,
            "p95_time_s": 0.002,
            "maximum_time_s": 0.003,
            "deadline_miss_fraction": 0.0,
            "stale_plan_fallback_time_s": 0.0,
            "wall_clock_effective_control_rate_hz": 10.0,
        },
        "physics_control_loop": {
            "mean_cycle_time_s": 0.020 if controller == "DREAM" else 0.015,
            "p95_cycle_time_s": 0.030 if controller == "DREAM" else 0.025,
            "maximum_cycle_time_s": 0.045 if controller == "DREAM" else 0.040,
            "deadline_miss_fraction": 0.0,
        },
        "traffic_disturbance": {
            "maximum_speed_loss_mps": follower_loss + 0.4,
            "total_integrated_speed_deficit_vehicle_m": 5.0 + follower_loss,
            "peak_deceleration_mps2": -2.0 - 0.01 * seed,
            "hard_braking_actor_count": 1 if controller == "DREAM" else 0,
            "maximum_follower_disturbance_amplification": (
                1.1 if controller == "DREAM" else 0.8
            ),
        },
        "runtime": {
            "real_time_factor": 1.0,
            "simulated_duration_s": 12.0,
            "wall_duration_s": 12.0,
        },
    }
    (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    resolved_manifest = {
        "scene_seed": seed,
        "generator_version": "fixture-generator-v1",
        "construction": {
            "scene_seed": seed,
            "generator_version": "fixture-generator-v1",
            "sha256": construction_hash,
        },
    }
    (directory / "resolved_manifest.json").write_text(
        json.dumps(resolved_manifest), encoding="utf-8"
    )
    return directory


def _write_bank(root: Path, n_scenes: int) -> None:
    for seed in range(1, n_scenes + 1):
        for controller in ("DREAM", "IDEAM"):
            for condition in ("empty_shadow", "true_threat"):
                _write_arm(root, seed, controller, condition)


def _find_stat(rows, metric):
    return next(row for row in rows if row["metric"] == metric)


def test_small_bank_reports_descriptive_scene_block_effects(tmp_path):
    bank = tmp_path / "runs"
    _write_bank(bank, 2)

    result = analyze_bank(
        [str(bank)],
        tmp_path / "analysis",
        bootstrap_replicates=200,
        make_figures=False,
    )

    assert result["n_scene_blocks"] == 2
    assert result["n_runs"] == 8
    assert result["bootstrap"]["performed"] is False
    assert result["inferential_p_values_performed"] is False
    ctv_empty = _find_stat(result["paired_effect_statistics"], "empty_shadow_ct_v_mps")
    ctv_true = _find_stat(result["paired_effect_statistics"], "true_threat_ct_v_mps")
    interaction = _find_stat(
        result["paired_effect_statistics"], "ct_v_condition_interaction_mps"
    )
    assert ctv_empty["mean"] == pytest.approx(2.0)
    assert ctv_true["mean"] == pytest.approx(2.5)
    assert interaction["mean"] == pytest.approx(0.5)
    assert ctv_empty["sample_sd"] == pytest.approx(0.0)
    assert ctv_empty["median"] == pytest.approx(2.0)
    assert ctv_empty["bootstrap_mean_ci95_lower"] is None
    traffic_amplification = _find_stat(
        result["paired_effect_statistics"],
        "empty_shadow__dream_minus_ideam__traffic_maximum_follower_disturbance_amplification",
    )
    planner_drop_fraction = _find_stat(
        result["paired_effect_statistics"],
        "empty_shadow__dream_minus_ideam__planner_dropped_request_fraction",
    )
    planner_rate = _find_stat(
        result["paired_effect_statistics"],
        "empty_shadow__dream_minus_ideam__planner_wall_clock_effective_rate_hz",
    )
    integrated_deficit = _find_stat(
        result["paired_effect_statistics"],
        "empty_shadow__dream_minus_ideam__traffic_total_integrated_speed_deficit_vehicle_m",
    )
    assert traffic_amplification["mean"] == pytest.approx(0.3)
    assert planner_drop_fraction["mean"] == pytest.approx(0.1)
    assert planner_rate["mean"] == pytest.approx(-1.0)
    assert integrated_deficit["mean"] == pytest.approx(0.5)
    dream_empty_cycle = next(
        row
        for row in result["raw_arm_statistics"]
        if row["controller"] == "DREAM"
        and row["condition"] == "empty_shadow"
        and row["metric"] == "physics_control_loop_mean_cycle_time_s"
    )
    assert dream_empty_cycle["mean"] == pytest.approx(0.020)
    assert (tmp_path / "analysis" / "scene_block_metrics.csv").is_file()
    assert (tmp_path / "analysis" / "bank_analysis.json").is_file()
    with (tmp_path / "analysis" / "run_metrics.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        header = next(csv.reader(handle))
    assert "traffic_total_integrated_speed_deficit_vehicle_m" in header
    assert "traffic_total_integrated_speed_deficit_m_s" not in header
    assert "physics_control_loop_deadline_miss_fraction" in header
    assert "low_level_wall_clock_effective_control_rate_hz" in header


def test_five_scene_bank_bootstraps_whole_blocks_and_renders_scienceplots(tmp_path):
    pytest.importorskip("scienceplots")
    bank = tmp_path / "runs"
    _write_bank(bank, 5)
    censored_path = bank / "seed_5" / "dream_empty_shadow" / "summary.json"
    censored = json.loads(censored_path.read_text(encoding="utf-8"))
    censored["ego"]["minimum_ttc_2d_s"] = None
    censored_path.write_text(json.dumps(censored), encoding="utf-8")

    result = analyze_bank(
        [str(bank)],
        tmp_path / "analysis",
        bootstrap_replicates=200,
        bootstrap_seed=7,
        make_figures=True,
    )

    assert result["bootstrap"]["performed"] is True
    assert result["bootstrap"]["resampling_unit"] == "complete four-arm scene block"
    ctv_empty = _find_stat(result["paired_effect_statistics"], "empty_shadow_ct_v_mps")
    assert ctv_empty["bootstrap_mean_ci95_lower"] is not None
    assert ctv_empty["bootstrap_mean_ci95_upper"] is not None
    finite_ttc = next(
        row
        for row in result["raw_arm_statistics"]
        if row["controller"] == "DREAM"
        and row["condition"] == "empty_shadow"
        and row["metric"] == "minimum_ttc_2d_s"
    )
    assert finite_ttc["n_available"] == 4
    assert finite_ttc["n_missing_or_censored"] == 1
    assert finite_ttc["bootstrap_mean_ci95_lower"] is None
    assert finite_ttc["inference_label"] == "incomplete_or_censored_descriptive_only"
    for name in (
        "fig_carla_bank_paired_effects.png",
        "fig_carla_bank_paired_effects.pdf",
        "fig_carla_bank_outcome_distributions.png",
        "fig_carla_bank_outcome_distributions.pdf",
    ):
        artifact = tmp_path / "analysis" / name
        assert artifact.is_file()
        assert artifact.stat().st_size > 1000


def test_incomplete_four_arm_scene_is_rejected(tmp_path):
    bank = tmp_path / "runs"
    _write_bank(bank, 1)
    missing = bank / "seed_1" / "ideam_true_threat" / "summary.json"
    missing.unlink()

    with pytest.raises(BankAnalysisError, match="complete four-arm block"):
        analyze_bank(
            [str(bank)],
            tmp_path / "analysis",
            bootstrap_replicates=100,
            make_figures=False,
        )


def test_new_runner_metrics_are_optional_for_legacy_valid_summaries(tmp_path):
    bank = tmp_path / "runs"
    _write_bank(bank, 1)
    for summary_path in bank.rglob("summary.json"):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary.pop("traffic_disturbance")
        summary.pop("physics_control_loop")
        summary["planner"].pop("completed_requests")
        summary["planner"].pop("dropped_requests")
        summary["low_level"].pop("wall_clock_effective_control_rate_hz")
        summary["runtime"].pop("wall_duration_s")
        summary_path.write_text(json.dumps(summary), encoding="utf-8")

    result = analyze_bank(
        [str(bank)],
        tmp_path / "analysis",
        bootstrap_replicates=100,
        make_figures=False,
    )

    assert result["n_scene_blocks"] == 1
    raw_metric_names = {row["metric"] for row in result["raw_arm_statistics"]}
    assert "traffic_maximum_speed_loss_mps" not in raw_metric_names
    assert "physics_control_loop_mean_cycle_time_s" not in raw_metric_names
    assert "planner_dropped_request_fraction" not in raw_metric_names


def test_legacy_integrated_deficit_key_is_normalized_to_vehicle_metres(tmp_path):
    bank = tmp_path / "runs"
    _write_bank(bank, 1)
    for summary_path in bank.rglob("summary.json"):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        traffic = summary["traffic_disturbance"]
        traffic["total_integrated_speed_deficit_m_s"] = traffic.pop(
            "total_integrated_speed_deficit_vehicle_m"
        )
        summary_path.write_text(json.dumps(summary), encoding="utf-8")

    result = analyze_bank(
        [str(bank)],
        tmp_path / "analysis",
        bootstrap_replicates=100,
        make_figures=False,
    )

    raw_metric_names = {row["metric"] for row in result["raw_arm_statistics"]}
    assert "traffic_total_integrated_speed_deficit_vehicle_m" in raw_metric_names
    assert "traffic_total_integrated_speed_deficit_m_s" not in raw_metric_names
    paired_metric_names = {
        row["metric"] for row in result["paired_effect_statistics"]
    }
    assert (
        "empty_shadow__dream_minus_ideam__traffic_total_integrated_speed_deficit_vehicle_m"
        in paired_metric_names
    )


def test_summary_and_resolved_manifest_hash_mismatch_is_rejected(tmp_path):
    run = _write_arm(
        tmp_path / "runs",
        1,
        "DREAM",
        "empty_shadow",
        manifest_hash=_scene_hash(1),
        top_hash=_scene_hash(999),
    )

    with pytest.raises(BankAnalysisError, match="disagrees"):
        analyze_bank(
            [str(run)],
            tmp_path / "analysis",
            bootstrap_replicates=100,
            make_figures=False,
        )


def test_calibration_run_without_statistical_bank_flags_is_rejected(tmp_path):
    run = _write_arm(tmp_path / "runs", 1, "DREAM", "empty_shadow")
    summary_path = run / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["qualification"]["statistical_bank_ready"] = False
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(BankAnalysisError, match="statistical_bank_ready"):
        analyze_bank(
            [str(run)],
            tmp_path / "analysis",
            bootstrap_replicates=100,
            make_figures=False,
        )


def test_current_resolved_scene_hash_is_recomputed_from_canonical_payload(tmp_path):
    template = (
        Path(__file__).resolve().parents[1]
        / "carla_converging_overtake_manifest.json"
    )
    bank = tmp_path / "runs"
    for scene_seed in (17, 18):
        manifest = resolve_file(template, scene_seed)
        for controller in ("DREAM", "IDEAM"):
            for condition in ("empty_shadow", "true_threat"):
                run = _write_arm(bank, scene_seed, controller, condition)
                summary_path = run / "summary.json"
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                summary.pop("construction_hash")
                summary.pop("scene_seed")
                summary.pop("generator_version")
                summary["scenario_id"] = manifest["scenario_id"]
                summary["duration_s"] = manifest["duration_s"]
                summary["runtime"]["simulated_duration_s"] = manifest["duration_s"]
                summary_path.write_text(json.dumps(summary), encoding="utf-8")
                (run / "resolved_manifest.json").write_text(
                    json.dumps(manifest), encoding="utf-8"
                )

    result = analyze_bank(
        [str(bank)],
        tmp_path / "analysis",
        bootstrap_replicates=100,
        make_figures=False,
    )

    verification = result["construction_hash_verification"]
    assert result["n_scene_blocks"] == 2
    assert len(set(result["scenario_ids"])) == 2
    assert verification["n_runs_recomputed_from_canonical_manifest_payload"] == 8
    assert verification["all_runs_recomputed_from_canonical_manifest_payload"] is True
