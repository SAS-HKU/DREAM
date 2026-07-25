# OACP-VB scientific implementation note


## Primary reference and provenance

The scientific source is:

> Lei Zheng, Rui Yang, Minzhe Zheng, Zengqi Peng, Michael Yu Wang, and Jun Ma,
> “Occlusion-Aware Contingency Safety-Critical Planning for Autonomous
> Driving,” *IEEE Transactions on Cybernetics*, 2026.

- [arXiv:2502.06359, version 2 (22 November 2025)](https://arxiv.org/abs/2502.06359)
- [DOI 10.1109/TCYB.2025.3632366](https://doi.org/10.1109/TCYB.2025.3632366)
- [Author project page](https://zack4417.github.io/oacp-website/)

An [author-origin review snapshot at commit
`06760501d24af6093994f4d6d6e95cf9e26f45e1`](https://github.com/mengxingshifen1218/OACP/commit/06760501d24af6093994f4d6d6e95cf9e26f45e1)
was inspected only to help resolve the interpretation of Equation (12). No code
from that repository is copied or vendored here. The snapshot has no
`LICENSE`/`COPYING` file, is not linked by the paper or project page, and
predates arXiv v2. It is therefore treated as an unlicensed review artifact,
not as an official code release or an implementation authority. Its README
license badge does not provide this project with a license grant.

The implementation in `dream_limo` is an independent implementation of the
paper’s simplified reachability quantification and dynamic velocity boundary.
The paper and arXiv v2 remain the normative scientific references.

## Experimental scope

All three comparison arms use the same received ego state, tracked visible
objects, live costmap, Nav2 geometric path, kinematic LMPC-CBF implementation,
model limits, cost weights, base collision ellipses, goal authorizer, command
gate, and safety supervisor. The startup status includes a hash of the shared
controller parameter dictionary so an arm-dependent controller change can be
detected.

| Runtime arm | Occlusion-risk channel | Effect on the shared controller |
| --- | --- | --- |
| `model:=nominal` | None | Nav2 path tracking and fixed-base LMPC-CBF only. `pure_mpc` remains a legacy alias, not a separate scientific arm. |
| `model:=oacp_vb` | Phantom-vehicle reachability on a path-relative occluded connector | Dynamic speed-reference cap plus a softened velocity upper bound; a second shared-prefix solve checks a contingency branch. |
| `model:=balanced` | DREAM DRIFT PDE field | Route-risk veto, MPC risk cost, and risk-scaled CBF/headway terms. |

In the current free-navigation experiment, Nav2 supplies geometry for every arm.
IDEAM is not run. DREAM’s free-navigation veto may stop on a risky route, but it
does not select an IDEAM gap or synthesize a replacement lane path. Likewise,
any lane change followed by OACP-VB comes from the shared Nav2 route, not from
Zheng et al.’s method.

## Equation-to-code traceability

The ROS-independent implementation is
[`dream_limo/core/oacp_vb.py`](dream_limo/core/oacp_vb.py). Its numerical tests
are in [`test/test_oacp_vb.py`](test/test_oacp_vb.py).

| Paper element | Core symbol | Test evidence |
| --- | --- | --- |
| PVS interval and Equation (10), \(g(s)\) | `make_pvs_interval`, `PVSInterval`, `potential_pv_count` | `test_pvs_clip_policy_makes_eq10_intervals_valid`, `test_pvs_reject_policy_refuses_too_long_interval`, `test_eq10_is_continuous_at_interval_boundaries`, `test_eq10_is_nonnegative_and_zero_outside_full_reach` |
| Equation (11), \(r_{\mathrm{lon}}(s)\) | `longitudinal_risk` | `test_longitudinal_and_point_risk_follow_eq11_and_eq13` |
| Adapted Equation (12), \(r_{\mathrm{lat}}(d)\) | `OACPVBConfig.lateral_sigma`, `lateral_risk` | `test_normalized_lateral_risk_is_symmetric_and_monotone_in_abs_offset` |
| Equation (13), \(r(s,d)\) | `point_risk` | `test_longitudinal_and_point_risk_follow_eq11_and_eq13` |
| Remark 2 trajectory-intersection gate | `reduce_horizon_risk`, `evaluate_geometry_risk` | `test_remark2_gate_zeros_risk_when_frs_does_not_intersect`, `test_remark2_finite_conflict_gate_ignores_nonintersecting_horizon` |
| Equations (14)–(15), dynamic velocity boundary | `dynamic_velocity_bound`, `VelocityBoundEvaluation` | `test_dynamic_velocity_bound_is_clamped_in_all_three_regions`, `test_velocity_bound_honours_nonzero_lower_threshold`, `test_branch_threshold_order_is_logged_not_semantically_relabelled` |
| Integration-specific threshold calibration | `calibrate_thresholds` | `test_threshold_calibration_uses_linear_p70_and_fallback_ratio`, `test_threshold_calibration_rejects_riskless_occluded_phase` |

The scenario geometry is implemented by `build_phantom_merge_connector`,
`extract_pvs_components`, and `evaluate_geometry_risk`. Tests cover straight and
curved routes, disconnected shadow components, PVS collapse after reveal,
finite conflict gating, and a positive merge-conflict case.

The MPC hook is in [`dream_limo/core/mpc.py`](dream_limo/core/mpc.py) and is
tested by [`test/test_mpc_velocity_bound.py`](test/test_mpc_velocity_bound.py).
`RiskAwareMPC.solve_reference`:

1. caps the path-speed reference at the supplied bound;
2. applies \(v_k \leq v_{\mathrm{occ}} + \epsilon_{v,k}\) at every predicted
   state;
3. constrains \(\epsilon_v \geq 0\) and penalizes its squared norm heavily; and
4. reports both maximum total and future velocity-bound slack.

Slack is intentional because the measured initial speed cannot instantly
satisfy a newly lower bound. A nonzero slack is a reported violation, not a
claim that the bound was strictly satisfied.

## Deliberate, method-changing deviations

This arm reproduces neither the complete published optimizer nor its safety
proof. The following deviations are part of the scientific definition of this
baseline:

1. **Shared Nav2 geometry, not IDEAM or OACP path optimization.** The free
   navigation graph receives a Nav2 path and all arms track that same geometry.
   The OACP-VB risk channel is speed-only.
2. **Shared kinematic LMPC-CBF.** The paper’s tenth-order Bézier
   parameterization, biconvex program, consensus ADMM solver, and published
   spatiotemporal barrier constraints are not implemented. The existing LIMO
   kinematic LMPC and tangent-ellipse CBF constraints replace them.
3. **Path-relative merge connector.** The phantom centerline is constructed one
   lane to the right of the current Nav2 route and smoothly joins that route.
   This right-to-middle connector is a mapping for the occluded-merge
   experiment; its lateral transition violates the paper’s assumption that a
   phantom vehicle remains on one lane centerline.
4. **Equation (12) interpretation.** The ambiguous point-dependent variance is
   replaced by the constant
   \(\sigma=l_w/(2Z)\), and
   \(r_{\mathrm{lat}}(d)=\exp[-\tfrac12(d/\sigma)^2]\). The Gaussian is
   normalized so \(r_{\mathrm{lat}}(0)=1\). The author-origin review snapshot
   supports the constant-sigma interpretation, but none of its code is used.
5. **Horizon reduction.** After the finite FRS/trajectory intersection test,
   `risk_total` is the maximum point risk over the nominal risk horizon and all
   retained PVS components. It is not a sum and is independent of sample count.
6. **Soft velocity-bound enforcement.** A heavy-penalty slack makes the new
   bound recoverable when it drops below measured speed. Every activation is
   exposed in planner status and experiment metrics.
7. **Sequential contingency approximation.** The exploration solve is executed.
   A second, non-committing fallback solve is constrained to share the first
   control inputs. These are not jointly optimized branches. If the fallback
   solve is invalid or either its velocity-bound or CBF slack exceeds the
   documented numerical-zero tolerance, a third solve clamps the executed
   bound to `v_occ_min`. The executed branch is solved at the shared 5 Hz rate.
   The fallback verification runs at 1 Hz. Between checks, every executed solve
   is constrained to consume the remaining controls from the certified common
   prefix in order. The adapter and supervisor preserve the exact structured
   ROS timestamp of each planner command on internal `TwistStamped` messages;
   physical `/cmd_vel` remains an ordinary `Twist`. The cursor advances only
   after the final hardware gate reports that exact source timestamp as
   forwarded and the next odometry remains within 0.01 m position, 0.03 m/s
   speed, and 0.05 rad yaw tolerances of the certified state tube. A partially
   executed control is not reissued for a new full \(dt\), because the cached
   fallback did not certify that extension. Its certificate is revoked and a
   fresh `v_occ_min` solve executes until the next scheduled branch check.
   Advancement requires at least 95% spatial progress plus next-knot agreement.
   A withheld, repeated, or differently stamped command is not consumed.
   Because a missing acknowledgement cannot prove that no physical output
   occurred, every missing, stale, or mismatched token revokes the certificate
   and clamps in the same way.
   Once the prefix is exhausted, or if tracking differs, any planner-side gate
   stops, the path or a visible vehicle changes, or either velocity cap
   tightens, the certificate is revoked and the executed bound clamps to
   `v_occ_min` until the scheduled check. Status reports the check age, prefix
   cursor/execution state, certificate state, and clamp event. The same launch
   sets the Nav2 geometric replan period to 1 s for all three arms. A new path
   activates only as an atomic path/bound pair and forces a fresh fallback
   check. When Remark 2 sets phantom risk to zero, or the PVS collapses at
   reveal, the phantom contingency is not applicable: the speed bound releases
   and the shared visible-vehicle CBF remains active.
8. **Different time horizons.** The shared controller currently runs at 5 Hz
   with \(N=6\) and \(dt=0.2\) s, a 1.2 s MPC horizon. OACP-VB separately
   evaluates phantom risk over 4 s (20 steps at the same `dt`); its assessor
   publishes at 10 Hz. The default shared control prefix is two controls
   (0.4 s), not the paper’s \(N_s=5\).
9. **Existing CBF geometry is preserved.** The LIMO configuration keeps base
   longitudinal/lateral axes of 0.34 m and 0.24 m, respectively, plus half the
   tracked object dimensions. OACP-VB uses these fixed, unmodulated axes; it
   does not substitute the paper’s 0.06 m hardware values.
10. **Branch naming remains ambiguous.** With the configured lower
    exploration threshold, the “exploration” branch has the tighter bound at a
    given nonzero risk and is the executed branch. The code retains and logs the
    paper/prompt labels without silently relabeling them; no safety meaning
    should be inferred from the words alone.

These are method-changing adaptations, not performance-neutral implementation
details.

The reduced contingency rate is evidence-driven. A 30-cycle, motion-free
onboard NUC12 benchmark with one visible-vehicle CBF constraint measured a
two-solve median of 185.0 ms, p95 of 247.3 ms, and maximum of 284.0 ms, so
checking both branches every 200 ms did not sustain the shared 5 Hz deadline.
The executed exploration solve’s median was 90.8 ms. The 1 Hz fallback check
keeps the executed solve at 5 Hz but does not reproduce the paper’s joint
optimization; occasional verification cycles can still exceed 200 ms. The
independent 0.5 s command watchdog remains unchanged. The aggregate record is
[`benchmark_results/oacp_vb_contingency_nuc12_2026-07-25.json`](benchmark_results/oacp_vb_contingency_nuc12_2026-07-25.json).
It records the dependency versions and the fact that raw per-cycle timings were
not retained; it is timing evidence, not a fully replayable benchmark trace.

The repository’s archival
[`OA_CMPC/oa_cmpc_source.py`](../OA_CMPC/oa_cmpc_source.py) concerns the
different work arXiv:2503.04563. Its own docstring records why that
single-branch surrogate was withdrawn from benchmarking: it was likewise a
method-changing adaptation. OACP-VB neither imports nor reuses it; that
precedent is the reason this arm’s reduced scope is named and disclosed
explicitly.

## ROS contract and readiness

[`dream_limo/oacp_vb_node.py`](dream_limo/oacp_vb_node.py) consumes:

- `/dream/ego_state` (`nav_msgs/msg/Odometry`);
- `/dream/occlusion_mask` (`nav_msgs/msg/OccupancyGrid`); and
- `/dream/geometric_path` (`nav_msgs/msg/Path`).

It publishes:

- `/dream/oacp_vb_status` (`std_msgs/msg/String`, JSON); and
- `/dream/oacp_vb_markers` (`visualization_msgs/msg/MarkerArray`).

Before a path exists, the assessor must publish
`provider="oacp_vb"`, `assessment_ready=true`, and
`pre_goal_bound_valid=true`. The shared goal authorizer refuses `/goal_pose`
unless this fresh pre-goal assessment exists. The accepted goal is republished
on `/dream/navigation_goal`; `/dream/deadman_status` records candidate receipt,
source, receipt, and publication timestamps. The goal publication timestamp is
recorded separately, while the accepted-goal receipt timestamp is the
experiment’s common \(t=0\).

Once a matching fresh path exists, `/dream/oacp_vb_status` must additionally
report `ready=true`, `exact_bound_valid=true`, and the matching
`path_source_stamp`. The status exposes PVS components, `risk_total`, the
Remark-2 intersection decision, both branch bounds and thresholds, calibration
sample count/suggestions, and the two horizon definitions. RViz displays the
PVS and bound markers on `/dream/oacp_vb_markers`.

`/dream/planner_status` identifies `arm`, `risk_channel`, the shared controller
hash/rate/horizon, applied and computed bounds, per-branch solve times, CBF and
velocity slack, contingency validity, and clamp events. The final hardware gate
must see the expected risk provider and a fresh exact bound; loss or mismatch
fails closed. Its status reports both the candidate receipt time and
`forwarded_control_source_stamp`; only the latter is an execution
acknowledgement.

## Threshold calibration gate

The published hardware thresholds `4.5/6.0` are initial reference values only.
They are not accepted as calibrated values for this arena.

1. Run `model:=oacp_vb` with `oacp_calibration_logging_only:=true`. Bounds are
   computed and logged but are not applied; this run is excluded from arm
   comparisons.
2. The assessor resets its samples for each accepted goal and starts its
   live percentile window only after the first motion authorization. It
   records the goal revision and accepted-goal receipt timestamp in status.
   Select the actual occluded phase from that single run’s rosbag and verify
   the window; the live suggestion is diagnostic, not an automatic approval.
3. The assessor reports the linear 70th percentile as the suggested
   exploration threshold and \(4/3\) of that value as the suggested fallback
   threshold.
4. Inspect the risk/bound curve for nonzero coverage and absence of endpoint
   saturation, then supply the reviewed thresholds for the comparison.
5. Set `oacp_thresholds_calibrated:=true` only after recording that review.

The installed extractor reads the recorded status topic, deduplicates repeated
status publications by the assessor's run-scoped sample count, and exports the
curve plus the linear p70 calculation:

```bash
ros2 run dream_limo dream_oacp_calibration /path/to/calibration_bag \
  --start-offset 0.0 \
  --end-offset 8.0 \
  --csv /tmp/oacp_calibration_curve.csv \
  --output /tmp/oacp_calibration_thresholds.json
```

The offsets are relative to the accepted goal timestamp and must be replaced
with the reviewed occluded interval. The output remains a candidate requiring
human review, not an automatic calibration approval.

Physical OACP-VB motion is launch-gated when the calibration acknowledgement is
false. Uncalibrated physical logging is explicitly non-comparison work and is
capped at 0.15 m/s. `v_occ_max` is the shared commanded cruise speed and
`v_occ_min` defaults to 55% of that speed.

## Claims boundary

This repository may claim only that it evaluates an independently implemented
phantom-reachability risk and dynamic velocity boundary inside the shared LIMO
controller. It may not claim reproduction of “OACP,” equivalence to the
published planner, transfer of its proof, or attribution of measured LIMO
performance to Zheng et al.

Published traversal-time, safety, comfort, and solver-time results are context,
not expected results for this arm. Any local result must be reported as an
OACP-VB result with its controller fingerprint, calibration record, slack
activations, solve timings for both branches, and the deviations above.
