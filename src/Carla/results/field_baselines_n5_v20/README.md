# v20 evidence bundle

This directory contains the compact evidence needed to reproduce the revised
CARLA Figure 6 without rerunning CARLA.

- `run_records/`: 40 valid arms organized by scene seed and controller/
  condition. Each arm contains the analysis summary, resolved manifest,
  provenance, 20 Hz tick trace, NPC trace, and evaluator-only actor states.
- `analysis/`: the supplied aggregate JSON/CSV, LaTeX table, and SciencePlots
  PDF/PNG generated from those records.
- `run_ledger.jsonl` and `logs/`: the randomized execution order, commands,
  return codes, and captured process output for all 40 attempts.

Absolute workstation paths in the copied JSON and CSV records were replaced
with `<DREAM_ROOT>`, `<PLANNER_PYTHON>`, `<CARLA_PYTHON>`, and
`<CARLA_EXECUTABLE>`. Scientific values, controller/condition identities,
qualification flags, construction hashes, and traces were not changed. The
byte-identical frozen scenario manifests and a path-sanitized bank index are
stored in `../../frozen_bank/n5_v20/`.

Reproduce the analysis from `src/Carla` with:

```powershell
python .\analyze_carla_figure6_profiles.py `
  --results-root .\results\field_baselines_n5_v20\run_records `
  --output-dir .\reproduced_figure6 `
  --bootstrap-resamples 10000 `
  --bootstrap-seed 20260717
```

The expected design is five complete scene blocks, four controllers, two
conditions, and 40 arms. The intervals are descriptive whole-scene bootstrap
intervals; they are not population-level significance tests.
