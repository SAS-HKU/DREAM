import math
import unittest

from evaluation.physical_safety_metrics import (
    KinematicBoxState,
    SceneSafetySnapshot,
    aggregate_episode_safety,
    constant_velocity_ttc,
    evaluate_scene_safety,
    evaluate_swept_pair_safety,
    signed_oriented_box_clearance,
    summarize_episode_safety,
)


def box(x, y=0.0, heading=0.0, vx=0.0, vy=0.0, label="vehicle"):
    return KinematicBoxState(
        x=x,
        y=y,
        heading=heading,
        vx=vx,
        vy=vy,
        length=4.0,
        width=2.0,
        label=label,
    )


class PhysicalSafetyMetricTests(unittest.TestCase):
    def test_axis_aligned_surface_clearance(self):
        self.assertAlmostEqual(
            signed_oriented_box_clearance(box(0.0), box(7.0)), 3.0, places=9
        )

    def test_overlap_has_negative_penetration(self):
        self.assertAlmostEqual(
            signed_oriented_box_clearance(box(0.0), box(3.0)), -1.0, places=9
        )

    def test_footprint_contact_has_zero_clearance(self):
        self.assertAlmostEqual(
            signed_oriented_box_clearance(box(0.0), box(4.0)), 0.0, places=9
        )

    def test_penetration_handles_contained_footprint(self):
        car = box(0.0)
        enclosing = KinematicBoxState(
            x=0.0,
            y=0.0,
            heading=0.0,
            vx=0.0,
            vy=0.0,
            length=10.0,
            width=4.0,
            label="enclosing",
        )
        self.assertAlmostEqual(
            signed_oriented_box_clearance(car, enclosing), -3.0, places=9
        )

    def test_constant_velocity_ttc_uses_vehicle_footprints(self):
        first = box(0.0, vx=1.0)
        second = box(10.0, vx=-1.0)
        self.assertAlmostEqual(
            constant_velocity_ttc(first, second, horizon_s=10.0), 3.0, places=9
        )

    def test_no_predicted_contact_is_not_replaced_by_arbitrary_cap(self):
        first = box(0.0, vx=0.0)
        second = box(10.0, vx=1.0)
        self.assertTrue(math.isinf(constant_velocity_ttc(first, second)))

    def test_ttc_detects_lateral_crossing(self):
        first = box(0.0, y=0.0)
        second = box(
            0.0, y=6.0, heading=math.pi / 2.0, vx=0.0, vy=-1.0
        )
        self.assertAlmostEqual(
            constant_velocity_ttc(first, second, horizon_s=10.0), 3.0, places=9
        )

    def test_scene_uses_most_critical_obstacle(self):
        scene = evaluate_scene_safety(
            box(0.0, vx=1.0, label="ego"),
            [box(8.0, label="far"), box(5.0, label="near")],
        )
        self.assertAlmostEqual(scene.min_clearance_m, 1.0)
        self.assertEqual(scene.clearance_vehicle, "near")

    def test_swept_evaluation_detects_between_frame_overlap(self):
        swept = evaluate_swept_pair_safety(
            box(-5.0, vx=0.0, label="ego"),
            box(5.0, vx=0.0, label="ego"),
            box(0.0, y=-5.0, heading=math.pi / 2.0, label="cross"),
            box(0.0, y=5.0, heading=math.pi / 2.0, label="cross"),
            interval_s=0.1,
            max_substep_s=0.01,
        )
        self.assertLess(swept.min_clearance_m, 0.0)
        self.assertEqual(swept.clearance_vehicle, "cross")

    def test_episode_summary_separates_near_contact_and_collision(self):
        snapshots = [
            SceneSafetySnapshot(2.0, math.inf, "a", None),
            SceneSafetySnapshot(0.5, 0.8, "a", "a"),
            SceneSafetySnapshot(-0.1, 0.0, "a", "a"),
            SceneSafetySnapshot(2.0, math.inf, "a", None),
        ]
        result = summarize_episode_safety(
            snapshots,
            reveal_step=1,
            post_reveal_steps=2,
            near_clearance_m=1.0,
        )
        self.assertTrue(result["near_collision_incident"])
        self.assertTrue(result["collision_incident"])
        self.assertEqual(result["near_collision_event_count"], 1)
        self.assertEqual(result["collision_event_count"], 1)
        self.assertAlmostEqual(result["post_reveal_min_clearance_m"], -0.1)
        self.assertAlmostEqual(result["post_reveal_min_ttc_s"], 0.0)

    def test_episode_aggregation_reports_rates_and_censoring(self):
        first = summarize_episode_safety(
            [SceneSafetySnapshot(2.0, math.inf, "a", None)],
            reveal_step=0,
            post_reveal_steps=1,
        )
        second = summarize_episode_safety(
            [SceneSafetySnapshot(-0.2, 0.0, "a", "a")],
            reveal_step=0,
            post_reveal_steps=1,
        )
        aggregate = aggregate_episode_safety([first, second])
        self.assertEqual(aggregate["n_episodes"], 2)
        self.assertAlmostEqual(aggregate["collision_rate"], 0.5)
        self.assertAlmostEqual(aggregate["min_ttc_censor_rate"], 0.5)
        self.assertEqual(aggregate["post_reveal_n_episodes"], 2)


if __name__ == "__main__":
    unittest.main()
