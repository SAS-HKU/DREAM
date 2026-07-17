# CARLA v20 matched two-condition benchmark

## Run ledger

- Design: 5 scene seeds x 4 controllers x 2 shadow conditions = 40 arms.
- Seeds: 1001--1005.
- Controllers: DREAM, IDEAM, ADA-sourced shared-backbone control, and
  APF-sourced shared-backbone control.
- Conditions: `empty_shadow` and `true_threat`.
- Bank ID: `converging_bank_2efd0ce2a79a425b`.
- Within-scene randomization seed: 20260717.
- Completion: 40/40 succeeded; no failed, skipped, or excluded arm.
- Inference: descriptive 95% whole-scene bootstrap intervals from 10,000
  resamples. The scene seed is the sampling unit; no population-level
  significance claim is made for n=5.

## Principal findings

In true-threat scenes, DREAM recorded 0/5 collisions and 2/5 near collisions,
compared with 4/5 and 5/5 for IDEAM, 0/5 and 5/5 for ADA-sourced, and 0/5 and
4/5 for APF-sourced. The paired DREAM-minus-IDEAM minimum global-clearance
difference was +1.188 m [0.894, 1.561]. The paired hidden-vehicle clearance
differences were +1.491 m [1.061, 1.989] versus IDEAM, +0.970 m [0.767, 1.190]
versus ADA-sourced, and +0.486 m [0.185, 0.841] versus APF-sourced. These data
support a selective physical-margin and near-conflict claim; they do not show
that DREAM was the only collision-free controller.

In empty-shadow scenes, all arms recorded 0/5 collisions and 0/5 near
collisions. DREAM's mean ego speed was 30.12 m/s versus 31.34 m/s for IDEAM,
giving an IDEAM-referenced conservatism tax of 1.224 m/s [1.131, 1.317]. The
paired additional maximum follower speed loss was 5.74e-6 m/s
[-0.001233, 0.001250], and the hard-braking traffic-actor count was unchanged.
The result shows a measurable ego-level efficiency cost but no observed
increase in either of these two follower-braking metrics in the five
six-second scenes. The integrated traffic speed deficit nevertheless increased
by 0.203 vehicle-m [0.139, 0.267], and the experiment cannot exclude longer
phantom traffic waves.

For DREAM true-threat arms, high-level optimization required 442.1 ms on
average, 521.0 ms at the mean per-run 95th percentile, and 536.1 ms at the
observed maximum. The effective rate was 2.33 Hz, with 76.3% of superseded
requests coalesced. The 10 Hz tracker averaged 1.493 ms with 0/300 deadline
misses; the 20 Hz physics/control loop averaged 12.557 ms with 0/600 misses.
The reveal-to-hidden-aware-plan application delay averaged 0.85 s [0.80, 0.95]
and reached 1.05 s, corresponding to approximately 25.4 and 31.4 m of travel
at the reported mean speed. All arms shared the same immediate visible-actor
safety supervisor, lane-hold fallback, and 10 Hz tracker. The evidence
therefore supports an implemented asynchronous proof of concept, not 10 Hz
execution of the high-level optimizer or a formal guarantee during an
arbitrary sudden reveal.

## Artifact locations

- Raw run directories: `outputs/carla_field_baselines_n5_v20/`.
- Frozen manifests and execution ledger:
  `outputs/carla_field_baselines_bank_n5_v20/`.
- Figure/analysis bundle:
  `outputs/carla_field_baselines_n5_v20/figure6_analysis/`.
- Figure 6 PDF: `figure6_analysis/figure6_carla_speed_clearance.pdf`.
- Figure 6 PNG: `figure6_analysis/figure6_carla_speed_clearance.png`.
- Per-episode metrics: `figure6_analysis/episode_metrics.csv`.
- Reveal-aligned profiles: `figure6_analysis/aligned_profiles.csv`.
- Machine-readable estimates: `figure6_analysis/aggregate_summary.json`.
- LaTeX results table: `figure6_analysis/carla_figure6_results_table.tex`.
- Revised manuscript subsection: `revision/section_4_4_carla_two_condition.tex`.
- Revised Figure 7 caption: `revision/figure_7_revised_caption.tex`.
- Baseline rebuttal: `revision/baseline_fidelity_rebuttal.tex`.
- Fidelity table and code map:
  `revision/baseline_implementation_fidelity_table.tex` and
  `revision/baseline_code_map.md`.
- Local release mirror ready for publication:
  `tmp/DREAM_publish/src/Carla/`.

The raw timing runs are camera-free. Existing files under
`tmp/DREAM_publish/src/Carla/videos/` are separate illustrative replays, not
the source of the statistical or runtime results.
