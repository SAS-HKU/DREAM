# Baseline implementation fidelity and code map

This map defines the implementation identity and permitted interpretation of every controller arm used in the revised CARLA comparison. Paths are relative to the public repository root. Reproducible environment setup and commands are intentionally maintained in `src/Carla/README.md` so that there is one executable protocol.

## Comparator identities

| Display name in the manuscript | Status | What it is |
|---|---|---|
| DREAM | Proposed method | Occlusion-aware source, field propagation, and all three decision/MPC/CBF coupling channels. |
| IDEAM | Native no-field reference | The repository's IDEAM decision and LMPC--CBF stack with no field and no DREAM coupling channel. |
| ADA-sourced shared-backbone control | Source-substitution control | The ADA-derived asymmetric source substituted into the otherwise unchanged DREAM propagation and controller stack. It is not an end-to-end ADA planner. |
| APF-sourced shared-backbone control | Source-substitution control | The APF repulsive source substituted into the otherwise unchanged DREAM propagation and controller stack. It is not an end-to-end APF planner. |
| OA-inspired single-branch risk-source surrogate | Withdrawn archival adapter | A tangent/reachable-set source adapter that does not implement the published OA-CMPC dual-/multi-branch optimizer. It is excluded from revised results. |

## Shared controller and field infrastructure

| Repository path | Relevant entry point | Responsibility |
|---|---|---|
| `src/Integration/episode_control.py` | `CouplingFlags`, `create_prideam_episode_arm`, `create_ideam_episode_arm`, `step_episode_arm` | Creates independent controller arms, enables or disables the three field-to-controller channels, and advances the decision/MPC stack. |
| `src/Integration/prideam_controller.py` | `PRIDEAMController`, `create_prideam_controller` | Wraps the IDEAM controller with decision veto, MPC field cost, and risk-dependent CBF modulation. |
| `src/Control/MPC.py` | `iterative_linear_mpc_control` | Implements the common IDEAM LMPC--CBF optimization used by DREAM and the retained controls. |
| `src/DecisionMaking/decision.py` | `decision_making` | Implements the common IDEAM gap-based maneuver decision. |
| `src/Integration/drift_interface.py` | `DRIFTInterface.warmup`, `DRIFTInterface.step` | Maps traffic states to the world-frame field grid and advances the field with an optional source-function substitution. |
| `src/pde_solver.py` | `compute_total_Q`, `PDESolver.step` | Implements DREAM's default visible-vehicle, occlusion, and merge sources and the shared field propagation equation. |

## Controller-specific implementation paths

### DREAM

- `src/pde_solver.py::compute_total_Q` constructs the default DREAM source, including explicit occlusion-shadow injection.
- `src/Integration/episode_control.py::create_prideam_episode_arm` creates the controller with all three coupling channels enabled for the CARLA arm.
- `src/Integration/prideam_controller.py` applies field information at the decision, MPC-objective, and CBF-parameter levels.

### IDEAM

- `src/Integration/episode_control.py::create_ideam_episode_arm` is the authoritative no-field factory.
- `src/DecisionMaking/decision.py::decision_making` and `src/Control/MPC.py::iterative_linear_mpc_control` are the retained native decision and controller implementations.
- The CARLA dispatch in `src/Carla/carla_external_planner.py::ExternalPhysicsPlanner._ensure_arm` selects this factory and does not instantiate or update a field.

IDEAM is therefore a shared-backbone no-field reference. It is not described as a newly reproduced external method.

### ADA-sourced shared-backbone control

- `src/Aggressiveness_Modeling/ADA.py::compute_risk_single` contains the underlying asymmetric momentum-wave risk expression.
- `src/Aggressiveness_Modeling/ADA_drift_source.py::compute_Q_ADA` evaluates that expression on the DREAM grid and applies per-class source-integral matching.
- `src/Carla/carla_external_planner.py::ExternalPhysicsPlanner._source_function` selects `compute_Q_ADA`; the planner otherwise uses `create_prideam_episode_arm` with every coupling channel enabled.

The ADA-derived source has no explicit occlusion-shadow term. Once injected, however, it is propagated by the same field equation as DREAM. The arm may be used to interpret source-shape sensitivity only; it must not be labelled simply "ADA" or treated as a faithful end-to-end ADA planner.

### APF-sourced shared-backbone control

- `src/APF_Modeling/APF.py::_repulsive_single_potential` contains the underlying repulsive potential.
- `src/APF_Modeling/APF_drift_source.py::compute_Q_APF` evaluates the repulsive potential on the DREAM grid and applies per-class source-integral matching.
- `src/Carla/carla_external_planner.py::ExternalPhysicsPlanner._source_function` selects `compute_Q_APF`; the planner otherwise uses `create_prideam_episode_arm` with every coupling channel enabled.

The source excludes an attractive goal term because the route request and trajectory tracking are held fixed across arms. It also has no explicit occlusion-shadow term, although its injected risk is propagated by the common field equation. The arm may be used to interpret source-shape sensitivity only; it must not be labelled simply "APF" or treated as a faithful end-to-end APF planner.

### Withdrawn OA-inspired adapter

- The authoritative working-tree module is `src/OA_CMPC/oa_cmpc_source.py::compute_Q_OACMPC`.
- It retains tangent-line occlusion geometry and TTC-based reachable-set circles, combines them with the shared visible-vehicle and merge sources, and maps the result to `Q(x,t)` for the DREAM propagation stack.
- It replaces the published branch structure with one worst-case branch. It does not implement distinct nominal/exploration and contingency trajectories, branch-specific decision variables, shared-horizon consensus constraints, ADMM coordination, or native OA-CMPC risk-boundary constraints.

This module is archival and is not imported by the revised CARLA benchmark. It must be called an **OA-inspired single-branch risk-source surrogate**, never OA-CMPC. No quantitative or qualitative method-level conclusion about OA-CMPC may be drawn from it.

Release check: ensure that the archived filename is consistently `oa_cmpc_source.py`; an earlier release mirror used `oc_cmpc_source.py`. The spelling mismatch must not be carried into a tagged reproducibility release.

## CARLA protocol and traceability

| Repository path | Responsibility |
|---|---|
| `src/Carla/carla_external_planner.py` | Runtime controller dispatch and the authoritative mapping from arm name to field source. Its supported-arm list excludes OA-CMPC. |
| `src/Carla/carla_overtaking_trial.py` | One closed-loop CARLA arm, observation logging, sensor visibility, oriented-box clearance, collision/near-collision outcomes, and optional video rendering. |
| `src/Carla/carla_converging_scene.py` | Deterministic scene and condition construction for the empty-shadow and true-threat cases. |
| `src/Carla/run_carla_converging_bank.py` | Frozen manifest generation, randomized within-scene arm ordering, CARLA process control, and bank execution. |
| `src/Carla/analyze_carla_converging_bank.py` | Original DREAM--IDEAM pilot analysis retained for provenance. |
| `src/Carla/analyze_carla_figure6_profiles.py` | Complete eight-arm validation, reveal-aligned speed/clearance profiles, whole-scene bootstrap summaries, Section 4.4 metrics, and SciencePlots Figure 6. |
| `src/Carla/physical_safety_metrics.py` | Oriented-box clearance and two-dimensional TTC definitions shared by the runner and analysis. |
| `src/Carla/README.md` | Authoritative dependency versions, commands, output schema, expected arm count, validation checks, and artifact inventory. |

All CARLA arms share the same 10-Hz trajectory tracker, lane-hold fallback,
and visible-actor safety supervisor implemented in
`src/Carla/carla_overtaking_trial.py`. These layers can affect collision
outcomes and are therefore part of the declared shared execution backbone;
the comparison changes the high-level plan delivered to that common layer.

The frozen manifest, not a controller-specific scenario script, defines actor geometry, nominal initial states, background-traffic control parameters, and the latent-vehicle reference trajectory. Semantic-LiDAR reveal time is an observed outcome, not a prescribed controller-specific input. All controller arms for a scene must record the same manifest hash and pass common realized-state tolerances. Controller order should be randomized within scene, and summaries should use scene seed as the sampling unit rather than treating simulation ticks as independent observations.

## Terminology rules for the manuscript and rebuttal

1. Use **IDEAM no-field reference**, not "IDEAM field baseline."
2. Use **ADA-sourced shared-backbone control** and **APF-sourced shared-backbone control** at first mention. Short labels `ADA-sourced` and `APF-sourced` are acceptable in compact legends after definition.
3. Describe ADA/APF comparisons as **source-substitution** or **component-control** experiments. Do not call them faithful, native, or end-to-end reproductions.
4. State explicitly that DREAM, ADA-sourced, and APF-sourced share field propagation and all three downstream couplings. Do not state that the substituted sources lack temporal propagation.
5. Use **OA-inspired single-branch risk-source surrogate** only when explaining the withdrawn adapter. Do not show it in revised figures or tables and do not abbreviate it as OA-CMPC.
6. Distinguish a source's lack of an explicit occlusion term from the propagated field's temporal behavior.
7. Report the immutable repository commit in the rebuttal and table or figure reproducibility note after the release has been pushed and verified.

## Remaining limitations to disclose

- Per-class integral matching changes the native amplitude scale of the ADA-derived and APF-derived sources; it is a controlled normalization for source-shape comparison.
- The APF-sourced arm tests only the repulsive source. It does not test a complete potential-field navigation policy.
- The ADA-sourced and APF-sourced arms retain DREAM's PDE and controller couplings, so their results cannot establish superiority to or infer performance of the cited full methods.
- Withdrawing the OA-inspired adapter removes an unfair comparison but does not supply evidence against or in favor of a native OA-CMPC implementation.
- Reproducibility claims should cite a verified commit and frozen bank only after both are present in the public tree.
