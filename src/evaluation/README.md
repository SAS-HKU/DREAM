# Paired trailer-occlusion ablation

This directory contains the reproducibility code for the paired ablation used
in the revised DREAM evaluation. The benchmark separates three questions:

1. whether the field formulation generates a pre-reveal occlusion-risk signal;
2. whether the decision, MPC-cost, and CBF channels affect physical outcomes;
3. what progress and braking cost the anticipatory response incurs when the
   occluded region is empty.

The unit of analysis is a matched scenario construction, not a simulation
step. Every variant in a pair receives the same traffic state, trailer
geometry, latent-actor trajectory, request timing, and reveal rule.

## Files

- `occlusion_benchmark_scenarios.py`: seeded matched true-threat,
  empty-shadow, and visible-control scenario generation.
- `scenario_qualification.py`: geometric and behavioral qualification checks.
- `field_variants.py`: pre-registered DRIFT field-component ablations.
- `run_paired_occlusion.py`: benchmark runner and coupling-variant registry.
- `physical_safety_metrics.py`: oriented-footprint collision, clearance, and
  two-dimensional TTC metrics.
- `paired_occlusion_analysis.py`: exact paired event tests, paired bootstrap
  intervals, paired randomization tests, and Holm adjustment.
- `build_ablation_summary_figure.py`: SciencePlots figure and long-form source
  data export.
- `tests/`: deterministic unit tests for scenario generation, safety metrics,
  and statistical analysis.

The runner also uses the context-injected controller adapter in
`Integration/episode_control.py` and the accompanying updates to
`Integration/drift_interface.py` and `Integration/prideam_controller.py`.
Correctly named compatibility copies of `Control/constraint_params.py` and
`Model/Dynamical_model.py` match imports already used by the controller. The
small `carfigs` assets are included because the existing `Control.utils`
module loads them during controller initialization.

## Environment

Install the repository requirements and run commands from `src` so the DREAM
packages are importable:

```powershell
python -m pip install -r requirements.txt
```

The plotting dependency is pinned as `SciencePlots==2.1.1` in
`requirements.txt`. The figure script also uses a repository-level
`SciencePlots/src` checkout when one is present.

## Smoke test

Run one critical true-threat pair for the full and no-veto variants:

```powershell
python -m evaluation.run_paired_occlusion `
  --out ../outputs/paired_ablation/smoke `
  --suite channels `
  --strata true_occluded_threat `
  --severity critical `
  --max-pairs 1 `
  --variants coupling_full coupling_no_veto `
  --save-traces
```

The output directory includes a frozen scenario manifest, qualification
records, one JSON object per episode arm, and optional step-level traces.

## Full paired suites

Use a fresh output directory for each suite. Do not change seeds, construction
filters, risk weights, or endpoints after a held-out suite begins.

```powershell
python -m evaluation.run_paired_occlusion `
  --out ../outputs/paired_ablation/channels_true `
  --suite channels --strata true_occluded_threat --save-traces

python -m evaluation.run_paired_occlusion `
  --out ../outputs/paired_ablation/channels_empty `
  --suite channels --strata empty_shadow --save-traces

python -m evaluation.run_paired_occlusion `
  --out ../outputs/paired_ablation/channels_visible `
  --suite channels --strata visible_control --save-traces

python -m evaluation.run_paired_occlusion `
  --out ../outputs/paired_ablation/field_true `
  --suite field --strata true_occluded_threat --save-traces
```

If a run is interrupted, repeat the command with `--resume`. The runner first
checks the frozen scenario-design hash and executes only missing arms.

## Paired analysis

The following example compares every channel ablation with full coupling and
emits pooled and predeclared severity-stratified results:

```powershell
python -m evaluation.paired_occlusion_analysis `
  --episodes ../outputs/paired_ablation/channels_true/episodes.jsonl `
  --reference-variant coupling_full `
  --stratify-by-severity `
  --output-json ../outputs/paired_ablation/channels_true/paired_analysis.json `
  --output-markdown ../outputs/paired_ablation/channels_true/paired_analysis.md
```

Repeat the analysis with `field_full` as the reference for the field suite.
The JSON output records input hashes, exclusions, pairing completeness,
endpoint definitions, effect estimates, confidence intervals, raw p-values,
and adjusted p-values.

## Figure

Pass the three episode sources explicitly. Repeat
`--channels-true-episodes` only if an interrupted channel suite was completed
in a second audited output directory:

```powershell
python -m evaluation.build_ablation_summary_figure `
  --channels-true-episodes ../outputs/paired_ablation/channels_true/episodes.jsonl `
  --field-true-episodes ../outputs/paired_ablation/field_true/episodes.jsonl `
  --channels-empty-episodes ../outputs/paired_ablation/channels_empty/episodes.jsonl `
  --figure-root ../figures/revision_r1c1
```

The script writes PDF, SVG, and 600-dpi PNG versions together with
`fig_ablation_summary_source_data.csv`, which contains every plotted value.

## Tests

```powershell
python -m unittest discover -s evaluation/tests -p "test_*.py"
```

The tests do not replace an end-to-end solver smoke run; they verify the
deterministic construction logic, physical metric edge cases, paired
inference, and output bookkeeping.
