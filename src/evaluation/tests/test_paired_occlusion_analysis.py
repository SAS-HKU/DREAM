"""Synthetic contract tests for the paired occlusion analysis layer."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from evaluation.paired_occlusion_analysis import (
    analyze_episode_records,
    analyze_jsonl,
    write_analysis_outputs,
)


def episode(
    scenario_id: str,
    variant: str,
    *,
    collision: bool,
    clearance: float,
    ttc: float,
    censored: bool = False,
    fallback: bool = False,
    severity: str = "critical",
) -> dict:
    return {
        "scenario_id": scenario_id,
        "stratum": "true_occluded_threat",
        "variant": variant,
        "scenario_design": {"severity": severity},
        "validity": {
            "sim_completed": True,
            "valid_reveal": True,
            "fallback_used": fallback,
        },
        "safety": {
            "collision_incident": collision,
            "min_clearance_m": clearance,
            "min_ttc_s": ttc,
            "min_ttc_censored": censored,
            "ttc_horizon_s": 10.0,
        },
    }


class PairedOcclusionAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.records = []
        for index, (reference_collision, comparator_collision) in enumerate(
            [(False, True), (True, False), (False, False), (False, False)],
            start=1,
        ):
            self.records.append(
                episode(
                    f"s{index}",
                    "full_dream",
                    collision=reference_collision,
                    clearance=float(index),
                    ttc=math.inf if index == 1 else 4.0,
                    censored=index == 1,
                )
            )
            self.records.append(
                episode(
                    f"s{index}",
                    "no_advection",
                    collision=comparator_collision,
                    clearance=float(index * 2),
                    ttc=5.0,
                    fallback=index == 4,
                )
            )
        # This scenario must not enter the paired estimand.
        self.records.append(
            episode(
                "incomplete",
                "full_dream",
                collision=False,
                clearance=1.0,
                ttc=5.0,
            )
        )

    def _analyze(self, *, exclude_fallback=False):
        return analyze_episode_records(
            self.records,
            reference_variant="full_dream",
            comparator_variants=["no_advection"],
            event_metrics=["safety.collision_incident"],
            continuous_metrics=[
                "safety.min_clearance_m",
                "safety.restricted_min_ttc_s",
            ],
            bootstrap_samples=1_000,
            seed=1234,
            exclude_fallback=exclude_fallback,
        )

    def test_complete_pairs_mcnemar_and_deterministic_bootstrap(self):
        first = self._analyze()
        second = self._analyze()

        self.assertEqual(first["population"]["n_complete_scenario_pairs"], 4)
        self.assertEqual(
            first["population"]["excluded_scenario_strata"],
            {"missing_required_variant": 1},
        )
        self.assertEqual(
            first["population"]["fallback_runs_retained_by_variant"],
            {"no_advection": 1},
        )

        event = first["event_results"][0]
        self.assertEqual(event["n_pairs"], 4)
        self.assertEqual(event["reference_only_events"], 1)
        self.assertEqual(event["comparator_only_events"], 1)
        self.assertEqual(event["p_value"], 1.0)

        clearance = next(
            item for item in first["continuous_results"]
            if item["metric"] == "safety.min_clearance_m"
        )
        self.assertAlmostEqual(
            clearance["paired_median_difference_comparator_minus_reference"], 2.5)
        self.assertEqual(
            clearance["bootstrap_percentile_ci_95"],
            next(
                item for item in second["continuous_results"]
                if item["metric"] == "safety.min_clearance_m"
            )["bootstrap_percentile_ci_95"],
        )

        restricted_ttc = next(
            item for item in first["continuous_results"]
            if item["metric"] == "safety.restricted_min_ttc_s"
        )
        self.assertEqual(restricted_ttc["reference_censored_count"], 1)
        self.assertEqual(restricted_ttc["restriction_horizons_s"], [10.0])

    def test_exclude_fallback_is_explicit_and_outputs_are_strict_json(self):
        result = self._analyze(exclude_fallback=True)
        self.assertEqual(result["population"]["n_complete_scenario_pairs"], 3)
        self.assertEqual(
            result["population"]["excluded_scenario_strata"],
            {"fallback_used": 1, "missing_required_variant": 1},
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            json_path, markdown_path = write_analysis_outputs(
                result,
                json_path=root / "analysis.json",
                markdown_path=root / "analysis.md",
            )
            with json_path.open("r", encoding="utf-8") as handle:
                saved = json.load(handle)
            self.assertEqual(saved["schema_version"], result["schema_version"])
            self.assertIn("Paired occlusion ablation analysis", markdown_path.read_text(encoding="utf-8"))

    def test_post_reveal_outcome_requires_explicit_valid_reveal(self):
        for record in self.records:
            record["safety"]["post_reveal_min_clearance_m"] = record["safety"]["min_clearance_m"]
        # The first pair is complete as a simulation run, but must not enter a
        # post-reveal outcome because its comparator's reveal is unverified.
        self.records[1]["validity"]["valid_reveal"] = False
        result = analyze_episode_records(
            self.records,
            reference_variant="full_dream",
            comparator_variants=["no_advection"],
            event_metrics=["safety.collision_incident"],
            continuous_metrics=["safety.post_reveal_min_clearance_m"],
            bootstrap_samples=100,
            seed=3,
        )
        post_reveal = result["continuous_results"][0]
        self.assertEqual(post_reveal["n_pairs"], 3)
        self.assertEqual(post_reveal["metric_exclusions"], {"invalid_or_missing_reveal": 1})

    def test_predeclared_severity_rows_keep_the_pooled_estimand(self):
        for record in self.records:
            record["scenario_design"]["severity"] = (
                "critical" if record["scenario_id"] in {"s1", "s2"} else "mild"
            )
        result = analyze_episode_records(
            self.records,
            reference_variant="full_dream",
            comparator_variants=["no_advection"],
            event_metrics=["safety.collision_incident"],
            continuous_metrics=[],
            bootstrap_samples=100,
            seed=3,
            stratify_by_severity=True,
        )
        rows = {
            entry["severity"]: entry
            for entry in result["event_results"]
        }
        self.assertEqual(set(rows), {"all", "critical", "mild"})
        self.assertEqual(rows["all"]["n_pairs"], 4)
        self.assertEqual(rows["critical"]["n_pairs"], 2)
        self.assertEqual(rows["mild"]["n_pairs"], 2)
        self.assertTrue(result["stratification"]["severity_requested"])
        self.assertEqual(
            result["population"]["complete_pairs_by_severity"],
            {"critical": 2, "mild": 2},
        )

    def test_analysis_can_auditably_merge_separate_episode_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first_path = root / "first.jsonl"
            second_path = root / "second.jsonl"
            first_path.write_text(
                "\n".join(json.dumps(record) for record in self.records[:4]) + "\n",
                encoding="utf-8",
            )
            second_path.write_text(
                "\n".join(json.dumps(record) for record in self.records[4:8]) + "\n",
                encoding="utf-8",
            )
            result = analyze_jsonl(
                [first_path, second_path],
                reference_variant="full_dream",
                comparator_variants=["no_advection"],
                event_metrics=["safety.collision_incident"],
                continuous_metrics=[],
                bootstrap_samples=100,
                seed=3,
            )
        self.assertEqual(result["population"]["n_complete_scenario_pairs"], 4)
        self.assertEqual(len(result["input"]["episode_jsonls"]), 2)


if __name__ == "__main__":
    unittest.main()
