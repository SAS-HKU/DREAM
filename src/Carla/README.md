# CARLA closed-loop occlusion validation

This directory contains the reproducible CARLA 0.9.14 implementation used to
complement the mechanism-level Python experiments in DREAM.  It implements a
matched converging-overtake scenario, causal semantic-LiDAR reveal, an
asynchronous DREAM/IDEAM planner service, 10 Hz low-level control, eligibility
checks, scene-block analysis, and SciencePlots figures.

The frozen pilot is intentionally small: five physical scene seeds and four
matched arms per seed (`DREAM/IDEAM` by `true_threat/empty_shadow`), for 20
eligible runs.  Its bank identifier is
`converging_bank_e378a392a5a45a10`, and its generator version is
`carla_converging_scene_resolver_v19`.  Treat the intervals as descriptive
pilot evidence, not as a population-level significance test.

## Scenario and experimental role

The ego vehicle passes a large CARLA fire-engine asset from the left lane while
a hidden vehicle passes it from the right.  Both request the centre-lane merge
region ahead of the occluder.  Nine surrounding vehicles use an IDM controller.
In the `empty_shadow` arm the hidden vehicle is removed but all other scene
construction and traffic inputs remain matched.

This validation does not replace Figs. 6 and 8 of the manuscript.  Those
figures expose the risk-field mechanism; CARLA tests whether the behavior
survives causal sensing, closed-loop vehicle dynamics, background traffic, and
the measured asynchronous execution path.

The stock fire engine is a large rigid occluder, not an articulated
tractor-trailer.  The hidden actor is withheld from the global IDM interaction
until ego semantic-LiDAR reveal so that the two conditions do not leak through
indirect traffic motion.  A later cue-enabled robustness experiment should be
used before making a broad claim about naturalistic traffic anticipation.

## Contents

- `carla_overtaking_trial.py`: CARLA bridge, sensors, actor control, timing,
  qualification, logging, and stacked driver/BEV rendering.
- `carla_external_planner.py`: asynchronous DREAM/IDEAM planner process.
- `carla_protocol.py`: versioned length-prefixed message protocol.
- `carla_converging_scene.py`: deterministic v19 scene resolver and geometric
  full-footprint occlusion gate.
- `run_carla_converging_bank.py`: bank freezing and randomized four-arm runner.
- `analyze_carla_converging_bank.py`: complete-block analysis and block
  bootstrap; it intentionally emits no p-values.
- `physical_safety_metrics.py`: oriented-box clearance and two-dimensional TTC.
- `plot_carla_paper_validation.py`: manuscript figures using SciencePlots.
- `frozen_bank/n5_v19`: exact five scene manifests and bank index.
- `results/pilot_n5_v19/run_summaries`: the 20 eligible summary/provenance
  records from which the supplied CSV results can be regenerated.
- `figures` and `videos`: publication candidates and outcome-blind seed-1001
  DREAM replays.  The videos are illustrative and are not timing evidence.
- `manuscript`: manuscript-ready LaTeX and the joint reviewer response.

## Tested environments

The experiment uses two Python processes because the Windows CARLA 0.9.14
extension is tied to Python 3.7, whereas the submitted DREAM optimization stack
uses the repository's modern scientific environment.

| Process | Tested environment |
|---|---|
| CARLA bridge | Python 3.7.9, CARLA 0.9.14, NumPy 1.21.6, Pygame 2.6.1 |
| Planner/analysis | Python 3.13.5 (Anaconda), NumPy 1.26.4, SciPy 1.15.3, CasADi 3.7.2, CVXPY 1.7.5 |
| Figures | Matplotlib 3.10.0, SciencePlots 2.2.1 development snapshot |

Install the CARLA wheel shipped under
`PythonAPI/carla/dist` into the Python 3.7 environment.  Install the main DREAM
repository dependencies into the planner environment, followed by
`requirements-analysis.txt`.  The short requirement files here document the
CARLA-side and plotting additions; they do not replace the root DREAM
environment.

## Reproduce the supplied analysis without CARLA

From `src/Carla`, use the modern environment:

```powershell
python .\analyze_carla_converging_bank.py `
  .\results\pilot_n5_v19\run_summaries `
  --output-dir .\reproduced_analysis `
  --bootstrap-replicates 10000 `
  --bootstrap-seed 260716
```

This reconstructs the raw-arm and paired-effect tables from the 20 retained
`summary.json` files.  A run is rejected from evidence if the four-arm block is
incomplete, manifest/trace hashes disagree, the projected hidden-vehicle
footprint fails the occlusion gate, or semantic visibility violates the causal
reveal rule.

## Freeze and execute a new matched bank

The following PowerShell example launches a separate CARLA server per arm and
paces the simulation in real time.  Change the executable and interpreter paths
for the local installation.

```powershell
$carlaPy = "C:\Path\to\Python37\python.exe"
$plannerPy = "C:\Path\to\DREAM-env\python.exe"
$carlaExe = "C:\CARLA_0.9.14\WindowsNoEditor\CarlaUE4\Binaries\Win64\CarlaUE4-Win64-Shipping.exe"

& $carlaPy .\run_carla_converging_bank.py `
  --seeds 1001,1002,1003,1004,1005 `
  --randomization-seed 20260716 `
  --bank-dir .\new_frozen_bank `
  --output-root .\new_runs `
  --execute --launch-server --pace-realtime `
  --quality-level Low `
  --planner-python $plannerPy `
  --carla-executable $carlaExe `
  --stop-on-failure
```

The default action of `run_carla_converging_bank.py` is freeze-only; the
explicit `--execute` flag prevents accidental expensive runs.  Low-quality,
camera-free runs are the timing protocol.  Do not add `--record-frames` to runs
used for computational-time claims.

An individual frozen arm can be reproduced with:

```powershell
& $carlaPy .\carla_overtaking_trial.py `
  --condition true_threat --controller DREAM --seed 1001 `
  --manifest .\frozen_bank\n5_v19\manifests\scene_0001_seed1001.json `
  --output-root .\single_run `
  --launch-server --pace-realtime --quality-level Low `
  --planner-python $plannerPy --carla-executable $carlaExe
```

## Create a high-fidelity illustrative replay

Rendering is deliberately separated from measurement.  Use Epic quality and
stacked output for an illustrative replay, and retain `--allow-invalid` because
recording makes it ineligible for the timing analysis:

```powershell
& $carlaPy .\carla_overtaking_trial.py `
  --condition true_threat --controller DREAM --seed 1001 `
  --manifest .\frozen_bank\n5_v19\manifests\scene_0001_seed1001.json `
  --output-root .\visual_replays `
  --launch-server --quality-level Epic `
  --record-frames --frame-stride 2 --allow-invalid `
  --planner-python $plannerPy --carla-executable $carlaExe
```

The `frames` directory contains the non-overlapping vertical composite, while
`driver_frames` and `bev_frames` contain the separated views.  The supplied
seed-1001 scene was selected from manifest parameters before inspecting
controller outcomes.

After regenerating the aggregate analysis and a representative visual run:

```powershell
python .\plot_carla_paper_validation.py `
  --analysis-dir .\reproduced_analysis `
  --visual-run-dir .\visual_replays\<run-directory> `
  --output-dir .\paper_figures `
  --representative-seed 1001
```

## Supplied pilot results

In the five true-threat blocks, DREAM had 0/5 collisions and 0/5 near
collisions, whereas IDEAM had 2/5 and 5/5.  The paired DREAM-minus-IDEAM effects
were +1.50 m [0.92, 2.03] for minimum hidden-vehicle oriented-box clearance and
+0.96 s [0.54, 1.29] for minimum hidden-vehicle two-dimensional TTC.

In the empty-shadow blocks, the ego-speed conservatism tax was 1.30 m/s [1.22,
1.39].  The paired additional maximum speed loss of the nearest follower was
1.9e-6 m/s [0.4e-6, 3.4e-6], hard-braking actor count was unchanged, and the
additional total integrated traffic speed deficit was 0.219 vehicle-m [0.175,
0.268].  These six-second trials do not rule out longer-range traffic-wave
effects.

For DREAM true-threat runs, high-level planning required 451.4 ms on average,
533.1 ms at the mean per-run 95th percentile, and 579.3 ms at the observed
maximum.  The effective high-level rate was 2.30 Hz and 76.8% of superseded
requests were coalesced.  The 10 Hz low-level controller required 1.456 ms on
average (1.826 ms P95, 3.556 ms maximum), and the 20 Hz physics/control loop
required 12.296 ms on average (15.253 ms P95, 30.977 ms maximum); neither loop
missed its deadline.  The current high-level optimizer is therefore not a
10 Hz real-time implementation.  The evidence supports an implemented
asynchronous proof of concept only.

## Recommended manuscript placement

Use `figures/fig_carla_closed_loop_validation.pdf` as the main CARLA result:
panel A establishes the causal occlusion/reveal sequence, panel B reports the
matched safety margins, and panel C reports the empty-shadow ego/follower
profiles.  Use `figures/fig_carla_async_runtime.pdf` with the revised timing
table.  Place the distribution and paired-effect figures in supplementary
material.  Keep the Python Figs. 6 and 8 because they explain the field
mechanism that the CARLA experiment does not isolate.
