"""Regression tests for the pre-registered paired occlusion bank."""

from __future__ import annotations

import unittest

from evaluation.occlusion_benchmark_scenarios import (
    OpenLoopSupportActor,
    ScenarioBankConfig,
    ScenarioConstruction,
    ScenarioStratum,
    generate_paired_scenario_bank,
)
from evaluation.scenario_qualification import qualification_manifest


class OcclusionBenchmarkScenarioTests(unittest.TestCase):
    def test_default_bank_has_independent_constructions_per_stratum(self) -> None:
        bank = generate_paired_scenario_bank()

        self.assertEqual(bank.summary()["paired_cases"], 15)
        self.assertEqual(bank.summary()["total_scenarios"], 45)
        self.assertEqual(len(bank.by_stratum(ScenarioStratum.TRUE_OCCLUDED_THREAT)), 15)
        self.assertEqual(len(bank.by_stratum(ScenarioStratum.EMPTY_SHADOW)), 15)
        self.assertEqual(len(bank.by_stratum(ScenarioStratum.VISIBLE_CONTROL)), 15)

        severity_counts: dict[str, int] = {}
        for scenario in bank.by_stratum(ScenarioStratum.TRUE_OCCLUDED_THREAT):
            severity_counts[scenario.conflict.label] = (
                severity_counts.get(scenario.conflict.label, 0) + 1
            )
        self.assertEqual(severity_counts, {"critical": 5, "moderate": 5, "mild": 5})

    def test_default_bank_passes_controller_independent_qualification(self) -> None:
        manifest = qualification_manifest(generate_paired_scenario_bank())

        self.assertTrue(manifest["all_passed"])
        self.assertEqual(len(manifest["pairs"]), 15)
        self.assertTrue(all(item["passed"] for item in manifest["pairs"].values()))

    def test_duplicate_construction_ids_are_rejected(self) -> None:
        construction = ScenarioConstruction(
            "critical", 4.5, 15.0, 10.0, 13.0, 0.25,
            2.0, 2.5, 2.0, 1.88, 1.5, 7.0, "duplicate",
        )
        with self.assertRaisesRegex(ValueError, "construction IDs"):
            ScenarioBankConfig(constructions=(construction, construction))

    def test_support_traffic_is_frozen_in_each_paired_specification(self) -> None:
        construction = ScenarioConstruction(
            "critical", 2.90, 18.0, 4.10, 3.8, 0.25,
            0.45, 2.5, 0.45, 1.55, 0.9, 5.0, "support_case",
        )
        support = OpenLoopSupportActor("rear_support", 2, -18.0, 9.0)
        bank = generate_paired_scenario_bank(
            ScenarioBankConfig(
                constructions=(construction,),
                support_actors=(support,),
            )
        )

        self.assertTrue(all(item.support_actors == (support,) for item in bank.scenarios))
        qualification = qualification_manifest(bank)
        pair = next(iter(qualification["pairs"].values()))
        self.assertTrue(pair["passed"])
        self.assertGreater(pair["nominal_support_clearance_m"]["rear_support"], 2.0)

    def test_duplicate_support_labels_are_rejected(self) -> None:
        actor = OpenLoopSupportActor("duplicate_support", 2, -18.0, 9.0)
        with self.assertRaisesRegex(ValueError, "support-actor labels"):
            ScenarioBankConfig(support_actors=(actor, actor))


if __name__ == "__main__":
    unittest.main()
