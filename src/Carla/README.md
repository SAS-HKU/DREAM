# CARLA closed-loop occlusion validation

This directory contains the reproducible CARLA 0.9.14 experiment used for the
revised Section 4.4 and Figure 6. It evaluates the converging-overtake scenario
under two matched conditions:

- `empty_shadow`: the occluded volume contains no hidden vehicle;
- `true_threat`: a hidden vehicle overtakes on the opposite side of the
  occluder and requests the same centre-lane merge region as the ego vehicle.

The frozen bank contains five physical scene seeds and eight matched arms per
seed: four controllers by two conditions, for 40/40 valid runs. The bank ID is
`converging_bank_2efd0ce2a79a425b`; the within-scene randomization seed is
`20260717`. All uncertainty intervals are descriptive 95% whole-scene
bootstrap intervals (10,000 resamples). With only five scene blocks, they are
not population-level significance tests.

The label `v20` identifies this eight-arm evidence run. The deterministic
scene resolver remains `carla_converging_scene_resolver_v19`; no scene-geometry
change was introduced between the earlier four-arm pilot and this bank.
The files `analyze_carla_converging_bank.py`,
`plot_carla_paper_validation.py`, `frozen_bank/n5_v19`, and
`results/pilot_n5_v19` are retained only as an explicitly archived development
record; they do not supply the revised estimates.

## Controller identities and baseline fidelity

| Label | Implementation used here | Permitted interpretation |
|---|---|---|
| DREAM | DREAM source, PDE propagation, and all three downstream coupling channels | Proposed method |
| IDEAM | Native no-field decision/LMPC--CBF reference | Shared-backbone no-field reference |
| ADA-sourced | ADA-derived asymmetric source substituted into DREAM's propagation and controller stack | Source-shape control, not a complete ADA planner |
| APF-sourced | Repulsive APF source substituted into DREAM's propagation and controller stack | Source-shape control, not a complete APF planner |

`OA_CMPC/oa_cmpc_source.py` is retained only as a clearly marked archival
single-branch risk-source surrogate. It does **not** reproduce the published
OA-CMPC dual-/multi-branch contingency optimizer and is excluded from every
revised figure, table, and comparative claim. See
`manuscript/baseline_code_map.md` and
`manuscript/baseline_implementation_fidelity_table.tex` for the complete
fidelity audit.

## Scenario and sensing protocol

The ego passes a large CARLA fire-engine asset from the left while the latent
vehicle passes it from the right. Both target a conflict region ahead of the
occluder; nine surrounding vehicles use IDM control. The stock fire engine is
a large rigid occluder, not an articulated tractor--trailer. The paired
empty-shadow arm removes only the latent vehicle.

The hidden vehicle is unavailable to the planner until its first causal
semantic-LiDAR detection. A true-threat episode is admitted only when the
entire projected hidden-vehicle footprint passes the pre-reveal occlusion gate
and the reveal is valid. Timing runs are camera-free, use Low graphics quality,
launch a fresh CARLA server for each arm, and are paced in real time. Driver
and BEV recordings are generated separately and are illustrative rather than
timing evidence.

## Code map

- `carla_converging_scene.py`: deterministic scene resolution and full-footprint
  occlusion gate.
- `carla_overtaking_trial.py`: sensors, actor control, asynchronous execution,
  metrics, logging, and optional separated driver/BEV rendering.
- `carla_external_planner.py`: authoritative DREAM/IDEAM/ADA/APF dispatch.
- `run_carla_converging_bank.py`: bank freezing, randomized eight-arm execution,
  server lifecycle, and ledger generation.
- `analyze_carla_figure6_profiles.py`: complete-block validation, reveal-aligned
  profiles, scene bootstrap, tables, and SciencePlots Figure 6.
- `physical_safety_metrics.py`: signed oriented-box clearance and 2-D TTC.
- `frozen_bank/n5_v20`: five byte-identical manifests and a path-sanitized
  bank index that records their byte-level hashes.
- `results/field_baselines_n5_v20/run_records`: path-sanitized records for all
  40 runs, including summaries, resolved manifests, 20-Hz tick traces, NPC
  traces, and evaluator-only actor-state logs used to compute physical and
  traffic metrics.
- `results/field_baselines_n5_v20/run_ledger.jsonl` and `logs/`: randomized
  attempt order, commands, return codes, and captured standard output/error.
- `results/field_baselines_n5_v20/analysis`: aggregate JSON/CSV, LaTeX table,
  and the supplied PDF/PNG figure.
- `manuscript/section_4_4_carla_two_condition.tex`: manuscript-ready subsection.
- `manuscript/baseline_fidelity_rebuttal.tex`: reviewer response on baseline
  fidelity and the revised comparison.
- `release_manifest.json`: SHA-256 inventory of the active v20 code, evidence,
  figures, tests, and manuscript files.

The ADA and APF source implementations are in
`../Aggressiveness_Modeling/ADA_drift_source.py` and
`../APF_Modeling/APF_drift_source.py`. Shared controller infrastructure is in
`../Integration/episode_control.py`, `../Integration/prideam_controller.py`,
`../Control/MPC.py`, and `../DecisionMaking/decision.py`.

## Tested environments

| Process | Tested environment |
|---|---|
| CARLA bridge | Python 3.7.9, CARLA 0.9.14, NumPy 1.21.6, Pygame 2.6.1 |
| Planner/analysis | Python 3.13.5, NumPy 1.26.4, SciPy 1.15.3, CasADi 3.7.2, CVXPY 1.7.5 |
| Figure generation | NumPy 1.26.4, Matplotlib 3.10.0, SciencePlots commit `8d281eabcf5f8159730e6df82e69c7ecd5437cb6` (`2.2.0-3-g8d281ea`) |

Install the CARLA wheel shipped under `PythonAPI/carla/dist` in the Python 3.7
environment. Install the main repository requirements and
`requirements-analysis.txt` in the planner/analysis environment.

## Reproduce Figure 6 without rerunning CARLA

From `src/Carla`, run:

```powershell
python .\analyze_carla_figure6_profiles.py `
  --results-root .\results\field_baselines_n5_v20\run_records `
  --output-dir .\reproduced_figure6 `
  --bootstrap-resamples 10000 `
  --bootstrap-seed 20260717
```

The analysis rejects an incomplete eight-arm block, invalid reveal, duplicate
arm, construction-hash mismatch, or mismatch between the trace and reported
minimum clearance. The scene seed, rather than simulation tick, is the
sampling unit.

## Freeze and run a new bank

```powershell
$carlaPy = "C:\Path\to\Python37\python.exe"
$plannerPy = "C:\Path\to\DREAM-env\python.exe"
$carlaExe = "C:\CARLA_0.9.14\WindowsNoEditor\CarlaUE4\Binaries\Win64\CarlaUE4-Win64-Shipping.exe"

& $carlaPy .\run_carla_converging_bank.py `
  --seeds 1001,1002,1003,1004,1005 `
  --controllers DREAM,IDEAM,ADA,APF `
  --randomization-seed 20260717 `
  --bank-dir .\new_frozen_bank `
  --output-root .\new_runs `
  --execute --launch-server --pace-realtime `
  --quality-level Low `
  --planner-python $plannerPy `
  --carla-executable $carlaExe `
  --stop-on-failure
```

The runner defaults to freeze-only; `--execute` is required for the expensive
CARLA phase. Do not enable frame recording in runs used for timing claims.

An individual arm can be replayed with:

```powershell
& $carlaPy .\carla_overtaking_trial.py `
  --condition true_threat --controller DREAM --seed 1001 `
  --manifest .\frozen_bank\n5_v20\manifests\scene_0001_seed1001.json `
  --output-root .\single_run `
  --launch-server --pace-realtime --quality-level Low `
  --planner-python $plannerPy --carla-executable $carlaExe
```

For an illustrative stacked driver/BEV replay, use Epic quality and
`--record-frames --allow-invalid`; the emitted `driver_frames`, `bev_frames`,
and non-overlapping vertical `frames` remain separate from timing evidence.

## Supplied v20 results

In the true-threat scenes, DREAM recorded 0/5 collisions and 2/5 near
collisions. IDEAM recorded 4/5 and 5/5, ADA-sourced 0/5 and 5/5, and
APF-sourced 0/5 and 4/5. Relative to IDEAM, DREAM increased minimum global
signed clearance by 1.188 m [0.894, 1.561] and hidden-vehicle clearance by
1.491 m [1.061, 1.989]. Relative hidden-vehicle clearance differences were
0.970 m [0.767, 1.190] versus ADA-sourced and 0.486 m [0.185, 0.841] versus
APF-sourced. The result supports a selective margin/near-conflict claim, not a
claim that DREAM alone avoided collision.

In empty-shadow scenes, DREAM's mean speed was 30.12 m/s versus 31.34 m/s for
IDEAM, giving a conservatism tax of 1.224 m/s [1.131, 1.317]. All four arms
had 0/5 collisions and 0/5 near collisions. DREAM's paired additional maximum
follower speed loss relative to IDEAM was effectively zero
(`5.74e-6` m/s [`-0.001233`, `0.001250`]), and the hard-braking traffic-actor
count showed no observed increase. The integrated traffic speed deficit was
nevertheless 0.203 vehicle-m [0.139, 0.267] larger. The six-second horizon
does not rule out longer traffic waves.

For DREAM true-threat runs, high-level optimization required 442.1 ms on
average, 521.0 ms at the mean per-run 95th percentile, and 536.1 ms at the
observed maximum. The effective high-level rate was 2.33 Hz and 76.3% of
superseded requests were coalesced. The 10 Hz tracker averaged 1.493 ms and
missed 0/300 deadlines; the 20 Hz physics/control loop averaged 12.557 ms and
missed 0/600 deadlines. The mean reveal-to-application delay for a hidden-aware
plan was 0.85 s [0.80, 0.95], and the largest observed delay was 1.05 s
(approximately 25.4 and 31.4 m of travel at the reported mean speed). All arms
shared the same immediate visible-actor safety supervisor, lane-hold fallback,
and 10 Hz tracker. This is an implemented asynchronous proof of concept, not a
10 Hz high-level real-time or formal sudden-reveal safety guarantee.

