# LIMO ROS2 Agent Prompt: OACP Baseline (occlusion-aware contingency planning)

> **IMPLEMENTATION STATUS:** The selectable baseline delivered in this
> repository is documented in
> [`dream_limo/OACP_VB.md`](dream_limo/OACP_VB.md). It is explicitly named
> **OACP-VB (velocity-bound adaptation of Zheng et al., 2025)** and does not
> claim to reproduce the paper's Bezier/consensus-ADMM planner. This prompt is
> retained as the design input; the implementation document records the
> realized architecture, tests, timing, and deviations.

Companion to `DREAM_LIMO_ROS_AGENT_PROMPT.md`. This is a **baseline arm**, not a new
system. The LIMO already runs DREAM (DRIFT PDE risk field → decision veto + risk cost +
CBF modulation) on top of the IDEAM LMPC. Here you keep that MPC stack and swap **only
the occlusion-risk mechanism** for the one published in:

> L. Zheng, R. Yang, M. Zheng, Z. Peng, M. Y. Wang, J. Ma,
> "Occlusion-Aware Contingency Safety-Critical Planning for Autonomous Driving,"
> arXiv:2502.06359 (v2, Nov 2025). Project page: https://zack4417.github.io/oacp-website/

```text
You are the agent on the AgileX LIMO (ROS2 Humble) that has already deployed the DREAM
occlusion-aware planner. Your task now is to add a SECOND planner arm implementing the
occlusion-aware risk methodology of Zheng et al. (arXiv:2502.06359), reusing the MPC
framework you already built. Do NOT create a new package and do NOT rebuild perception,
estimation, tracking, or the safety supervisor. Add a selectable planner mode to the
existing package.

========================================================================
0. OBJECTIVE
========================================================================
Produce a fair, publishable baseline comparison on ONE physical scenario (the occluded
merge already built for DREAM).

THE CONTROLLER IS THE SAME. We are running the SAME MPC controller in every arm — the
same LMPC code path, the same weights, horizon, kinematics, limits, and the same CBF
collision constraints. What differs between arms is only:

    (i)  RISK ASSESSMENT  — how a hidden hazard is represented, and
    (ii) RISK EVALUATION  — where that representation is queried, how it is reduced to
                            the number(s) the controller consumes, and which controller
                            channel those numbers enter.

  DREAM arm  : assessment = PDE risk field R(x,y,t) on a world grid.
               evaluation = sampled along a candidate lane-change path and along the MPC
               horizon -> enters as a decision veto, a risk term in the MPC cost, and
               risk-scaled CBF ellipses.
  OACP arm   : assessment = phantom-vehicle reachability risk r(s,d) on the occluded
               lane centerline.
               evaluation = reduced to a scalar r_total over the planned horizon -> enters
               as a hard DYNAMIC VELOCITY UPPER BOUND, plus a two-branch
               (exploration/fallback) contingency structure.

Everything else — arena, lane geometry, obstacle and blocker placement, merger script,
run trigger, control rate, MPC horizon and weights, base CBF ellipse axes, vehicle
limits, IDEAM decision layer, state estimation — is IDENTICAL across arms. The arm must
be selected by a runtime parameter on one shared node graph, never by a separate launch
path or a second controller instance. If you change something for one arm, change it for
both and say so in the report.

RUN TRIGGER — reuse what you already built, do not invent a new one. Each run is started
by publishing a navigation goal (the same goal topic, message type, and arming sequence
already implemented for the DREAM arm). The planner arms on goal receipt, holds at
standstill until the countdown completes, then executes. Requirements:
  - identical goal pose and identical trigger path for all arms;
  - the OACP risk module must be running and publishing r_total BEFORE the goal is
    accepted, so the first commanded velocity is already risk-bounded (there is no PDE
    warm-up in this arm, so arming is immediate — but still verify a valid bound exists
    at the moment of goal acceptance, and refuse the goal if not);
  - log the goal-acceptance timestamp. It is t = 0 for every time-aligned plot and the
    common reference for cross-arm comparison.

========================================================================
1. THE SWAP, PRECISELY
========================================================================
REMOVE from the OACP arm (do not delete code; gate it behind the planner-mode flag):
  - the DRIFT PDE solver, its grid, warm-up, and the risk OccupancyGrid publisher
  - the decision-level risk veto (evaluate_decision_risk / gate_lane_change)
  - the risk term in the MPC cost (risk_weight * 0.1 * R * vx^2)
  - risk-based CBF ellipse and headway modulation (apply_risk_modulation)

KEEP unchanged and shared (the controller itself is NOT part of the swap):
  - the LMPC instance and every one of its parameters: horizon T, dt, cost weights
    (R, Rd, Q, Qt, ...), acceleration/steer/jerk limits, slack penalties, and the CBF
    collision constraints with FIXED (unmodulated) ellipse axes. The OACP arm adds ONE
    new constraint (Section 4) and overrides the speed reference; it changes nothing
    else inside the controller. Diff the effective parameter dict between arms at
    startup and log it — if anything other than the risk channel differs, the
    comparison is invalid.
  - the IDEAM decision layer (gap groups + DFS). OACP as published is a SPEED-ONLY
    method ("expert human drivers typically adjust their speed rather than alter their
    path"); it contains no lane-change decision logic. The merge decision in our
    scenario therefore comes from the shared IDEAM base in BOTH arms. This must be
    disclosed in the paper — it is an addition to OACP, not part of it.
  - vehicle kinematics, Frenet path objects, state estimation, tracking, the nav-goal
    trigger, and the safety supervisor.

ADD for the OACP arm:
  - occlusion risk assessment via phantom-vehicle reachability (Section 2)
  - the dynamic velocity boundary as a hard MPC constraint (Section 4)
  - the two-branch contingency structure (Section 5)

========================================================================
2. METHOD TO IMPLEMENT (verbatim from the paper, Sec. III-A)
========================================================================
Assumptions: phantom vehicles (PVs) travel along the lane centerline; with no prior
information a PV's initial position is uniformly distributed over the Phantom Vehicle
Set (PVS), the occluded segment of that centerline, bounded by s_s (near) and s_e (far).
PV speed is constant, uniform on [0, v_pv_max].

Let T be the risk prediction horizon. Define the three intervals
    I1 = [s_s, s_e]
    I2 = [s_e, s_s + v_pv_max*T]
    I3 = [s_s + v_pv_max*T, s_e + v_pv_max*T]

Number of potential PVs able to reach longitudinal position s (paper Eq. 10):
    s in I1:  g(s) = 0.5 * (2*v_pv_max - (s-s_s)/T) * (s - s_s)
    s in I2:  g(s) = 0.5 * (2*v_pv_max - (s-s_s)/T - (s-s_e)/T) * (s_e - s_s)
    s in I3:  g(s) = 0.5 * (v_pv_max - (s-s_e)/T) * (s_e - (s - v_pv_max*T))
    otherwise g(s) = 0

Longitudinal risk, scaled by PVS length so larger occluded zones carry more risk
(Eq. 11):
    r_lon(s) = (s_e - s_s) * g(s)

Lateral risk from PV lateral-position uncertainty (Eq. 12), l_w = lane width,
Z = confidence factor (Z = 1.645 for 90%):
    r_lat(d) = N( 0, ( l_w / (2*Z*(1 - 0.5*(1-d))) )^2 )
Risk is highest at d = 0 (PV on the lane centerline).

Total risk (Eq. 13):
    r(s,d) = r_lon(s) * r_lat(d)

Dynamic maximum velocity boundary (Eqs. 14-15):
    dv     = (v_occ_min - v_occ_max) / (c_th_max - c_th_min)          # negative
    v_occ  = v_occ_min                              if r_total >  c_th_max
    v_occ  = dv*(r_total - c_th_min) + v_occ_max    otherwise
Check: r_total = c_th_min gives v_occ_max; r_total = c_th_max gives v_occ_min.

Two thresholds give two bounds — c_th_max^exploration < c_th_max^fallback — so the
exploration branch clamps to v_occ_min at lower risk (it prioritises situational
awareness) while the fallback branch keeps a looser speed cap and takes its safety from
the barrier constraints.

Remark 2 (implement it — it prevents permanent over-conservatism): occlusion risk is
IGNORED when the PVs' forward reachable set does not intersect the ego's planned
trajectory within the prediction horizon.

IMPLEMENTATION NOTE on Eq. 12: the paper writes r_lat as a normal distribution whose
variance itself depends on the evaluation point d, which is ambiguous. Implement it as
the pdf of N(0, sigma(d)^2) evaluated at d, verify numerically that it decreases
monotonically in |d|, and then NORMALISE so r_lat(0) = 1. Normalisation makes r_total's
magnitude governed by r_lon and makes threshold calibration (Section 7) interpretable.
Log this as a documented deviation.

========================================================================
3. SCENARIO MAPPING: intersection -> occluded merge
========================================================================
The paper's scenario is an occluded intersection; ours is the occluded merge already
built for DREAM (ego left lane, static box in middle lane, blocker forcing a merge,
second robot hidden in the right lane). The SRQ formulation is one-dimensional along a
lane centerline, so it transfers directly:

  - PV lane        = the RIGHT lane (the occluded one).
  - PVS            = the set of points on the right-lane centerline that are inside the
                     ego's perception range r_l AND not visible (inside the geometric or
                     lidar shadow of the box). s_s = nearest occluded point,
                     s_e = farthest occluded point, capped by r_l.
  - d              = lateral offset between the evaluated point and the right-lane
                     centerline.
  - Evaluation set = the ego's planned trajectory over the MPC horizon, mapped into the
                     PV lane's longitudinal frame. Apply Remark 2 first: if the PV FRS
                     over [0,T] does not intersect that trajectory, set r_total = 0.
  - r_total        = max over the horizon of r(s_k, d_k). (Use max, not sum: it is
                     deterministic and horizon-length independent. Expose sum behind a
                     flag; if you switch, recalibrate thresholds.)

Behavioural consequence to verify on hardware: as the ego advances, the shadow sweeps,
so s_s and s_e move and r_total evolves. When the ego clears the box the right lane
becomes visible, the PVS collapses (s_e - s_s -> 0), r_lon -> 0, and the velocity bound
releases to v_occ_max — while the now-visible merger becomes a REAL obstacle handled by
the existing CBF constraints. That handover (phantom risk -> real constraint) is the
core behaviour of this baseline and is what the reveal-time metrics measure.

========================================================================
4. INTEGRATION POINT IN THE EXISTING MPC — read this before coding
========================================================================
AUDIT FINDING (verified in the development repo, Control/MPC.py): `MAX_SPEED` and
`MIN_SPEED` are assigned in `LMPC.__init__` but NEVER appear in any constraint. Speed is
currently shaped only through the reference (`_effective_target_speed()`) and the cost.
There is therefore NO existing speed constraint to retarget — you must add one. This is
the single most important line of this document, because OACP's entire mechanism is a
hard velocity upper bound (paper constraint 18h).

Add, in the MPC constraint assembly, for each horizon step t (state x[0,t] = vx):

    constraints += [ x[0, t] <= v_occ_bound + slack_v[t] ]
    constraints += [ slack_v >= 0 ]
    cost        += W_v * sum_squares(slack_v)          # large W_v

Use a slack with a heavy penalty rather than a bare hard bound: the existing stack relies
on slacks throughout (slack_a, slack_d, slack_cbf_f, lateral d[...]), and an unslacked
state bound will make the QP infeasible the moment the bound drops below the current
speed. Report any step where slack_v is active — that is a genuine constraint violation
and must appear in the results, not be hidden.

Also override the tracking reference each step so the MPC does not fight its own bound:

    controller.mpc.set_target_speed_override(min(v_ref_nominal, v_occ_bound))

(`set_target_speed_override` already exists on LMPC.) The hard bound is the safety
mechanism; the reference override is what makes the behaviour smooth. Implement both.

Also set MIN_SPEED to 0 in the LIMO parameter set (it is currently 2.0 m/s, unscaled and
unenforced) so that if you or a later change ever DO enforce it, the robot can still stop.

========================================================================
5. CONTINGENCY STRUCTURE (honest approximation — read Section 8)
========================================================================
The paper optimises exploration (j=0) and fallback (j=1) trajectories jointly, as a
biconvex NLP over 10th-order Bezier control points, solved by consensus ADMM, with the
branches sharing a common initial segment of N_s steps and diverging after N_s + N_d.

You will NOT reproduce that solver. Approximate the contingency structure with two solves
of the existing MPC:

  1. Solve with v_occ_bound = v_occ^exploration  -> this is the EXECUTED trajectory.
  2. Solve with v_occ_bound = v_occ^fallback, with the first N_s control inputs
     constrained equal to those from solve 1 (the shared-segment consistency
     constraint 18k-18l).
  3. If solve 2 is infeasible or its slack is active, the fallback does not exist:
     clamp the executed bound to v_occ_min for this step and log the event.

This preserves what the contingency structure is FOR (a verified safe alternative sharing
the immediate control action) at roughly 2x MPC cost. Measure that cost; if it breaks
real time, run solve 2 at a reduced rate and say so.

========================================================================
6. ROBOT-SCALE PARAMETERS — use the paper's own 1:10 hardware numbers
========================================================================
Fortunately this paper validated on hardware at our exact scale: a 1:10 Ackermann robot
(TianRacer, 380 x 210 mm) in a 1:10 intersection with 37.5 cm lanes, with AgileX LIMO
Rovers serving as the phantom vehicles, on a Jetson Xavier NX. Use these published
values, not a rescaling of the simulation values:

    lane width l_w              0.375 m   (USE YOUR ARENA'S ACTUAL LANE WIDTH — it must
                                           match the DREAM arm's arena exactly)
    v_pv_max                    1.0 m/s
    ego cruise speed            ~0.5 m/s, dropping to ~0.28-0.30 m/s near the occlusion
    safety ellipse l_x = l_y    0.06 m
    c_th_min                    0
    c_th_max (exploration)      4.5        } recalibrate — see Section 7
    c_th_max (fallback)         6.0        }
    risk horizon T              4 s
    perception range r_l        3 m        (30 m full scale at 1:10)
    Z                           1.645
    planning horizon N          40 steps at dt = 0.1 s
    N_s / N_d                   5 / 5 steps
    control rate                10 Hz
    reported solve time         27.33 ms mean, 44.65 ms max (Jetson Xavier NX)

NOTE ON TIME SCALING — this changes the DREAM arm too. The paper scales LENGTH by 1/10
and leaves TIME unchanged (dt = 0.1 s, 10 Hz, T = 4 s; v_pv_max 10 -> 1 m/s is pure
length scaling). The DREAM prompt recommended alpha = 1/10 with beta = 2 (dt = 0.2 s,
5 Hz). Running the arms at different control rates would confound the comparison.
Harmonise: pick ONE control rate both arms can sustain on your hardware, rerun the DREAM
arm at that rate if it has already been tuned at another, and record the choice. If the
DREAM MPC cannot hold 10 Hz, run both at 5 Hz and note that OACP is being run at half
its published rate.

========================================================================
7. THRESHOLD CALIBRATION (required — do not transplant the numbers blindly)
========================================================================
r_total is geometry-dependent and NOT scale-invariant. Under length scaling by alpha with
time unchanged: r_lon ~ alpha^3, normalised r_lat ~ 1, so r_total ~ alpha^3 (and ~alpha^2
if you do not normalise r_lat). The paper's own numbers confirm the thresholds are tuned
empirically rather than derived: simulation used c_th_max = 40/60 while the 1:10 hardware
run used 4.5/6 — not the ratio any pure scaling law predicts. Therefore:

  1. Run the approach ONCE with the velocity bound computed but NOT applied
     (open-loop logging, ego at constant speed, merger scripted).
  2. Log r_total(t) over the whole approach.
  3. Set c_th_max^exploration ~= the 70th percentile of r_total over the occluded phase,
     so the bound spans [v_occ_min, v_occ_max] across the approach instead of saturating
     at one end. Set c_th_max^fallback ~= 1.33 * c_th_max^exploration (the paper's 6/4.5
     ratio).
  4. Set v_occ_max = the shared nominal cruise speed and v_occ_min ~= 0.55 * v_occ_max
     (the paper's 0.28/0.5 hardware ratio).
  5. Report the calibration curve. A baseline tuned to saturate is a strawman and will be
     challenged in review.

========================================================================
8. SCIENTIFIC INTEGRITY — what you may and may not claim
========================================================================
You are implementing the paper's RISK ASSESSMENT inside our MPC. You are NOT implementing
the paper's PLANNER. Not reproduced: the biconvex NLP formulation, 10th-order Bezier
trajectory parameterisation, consensus-ADMM decomposition, the published spatiotemporal
barrier (Eqs. 4-6) — we substitute IDEAM's existing CBF, which is structurally similar
(elliptical, h = xi - 1) but not identical — and joint optimisation of the two branches.

Consequences you must respect:
  - Name the arm "OACP-VB (velocity-bound adaptation of Zheng et al., 2025)" everywhere:
    code, plots, tables, text. Never "OACP".
  - State explicitly that the lane-change decision comes from the shared IDEAM layer.
  - Do not attribute the arm's performance to the published method. Efficiency claims in
    particular are not transferable: the paper's headline result (30.17% faster traversal
    than ST-RHC) comes from the contingency NLP, which you are approximating.
  - PRECEDENT: the development repo already contains
    `OA_CMPC/oa_cmpc_source.py`, an adapter for a DIFFERENT paper (arXiv:2503.04563)
    that was WITHDRAWN from the benchmark for exactly this class of adaptation. Read its
    docstring before you write any claim — it states the standard this project holds
    itself to: "a method-changing adaptation, not a performance-neutral implementation
    detail."
  - If the paper's authors' own code becomes available, prefer porting it over this
    adaptation, and say which was used.

========================================================================
9. A/B PROTOCOL
========================================================================
Three arms — same controller, same trigger, different risk assessment/evaluation:
  A. Nominal    — shared LMPC-CBF + IDEAM decision, no occlusion risk at all (the floor).
  B. OACP-VB    — this arm: reachability risk -> velocity bound.
  C. DREAM      — already deployed: PDE field -> veto + cost + CBF scaling.

Every run in every arm is started by publishing the same nav goal to the same topic with
the same pose (Section 0). Arm selection is a parameter on the shared node graph. Before
each run, log the startup parameter diff between arms and confirm it contains nothing but
the risk-channel settings.

Runs: >= 5 per arm with the identical scripted merger timing, plus >= 3 per arm with a
randomised merger release time (to test that neither arm is tuned to one script).

Report per arm, and do not omit the efficiency columns — OACP's contribution is
efficiency WITH safety, and reporting only safety margins strawmans it:
  Safety     : min clearance, min TTC, TTC at reveal, post-reveal min clearance (3 s
               window), contacts, safety-supervisor triggers, slack_v activations.
  Efficiency : traversal time over a fixed course, mean speed, distance-to-goal at
               fixed time, time spent below 50% of nominal cruise.
  Comfort    : max |a|, mean |jerk|.
  Compute    : mean and max solve time per arm (both solves for OACP-VB).

Expected qualitative signatures — if you see something else, investigate before writing:
  - OACP-VB slows smoothly and monotonically as the PVS grows, then releases sharply on
    reveal; it does not alter its path because of occlusion.
  - DREAM builds spatial risk and can veto/defer the merge itself, so it may differ in
    WHERE it goes, not only how fast.
  - The Nominal arm shows the smallest TTC at reveal.

========================================================================
10. STAGED PLAN
========================================================================
  S0  Audit: confirm the DREAM arm's current control rate, arena lane width, MPC horizon,
      base ellipse axes, where the constraint list is assembled, and the existing nav-goal
      trigger (topic, message type, arming/countdown sequence). Report these before
      coding — they become the shared configuration that the OACP arm must not alter.
  S1  Implement risk assessment (Section 2) as a standalone module + unit tests:
      g(s) continuous across I1/I2/I3 boundaries; g >= 0; g -> 0 outside I3;
      r_lat monotone decreasing in |d| and equal to 1 at d = 0;
      v_occ = v_occ_max at r_total = 0 and = v_occ_min at r_total = c_th_max.
  S2  PVS geometry from the shadow (Section 3) + RViz markers for PVS extent and the
      current bound. Verify visually that the PVS collapses at reveal.
  S3  Wire the velocity bound into the MPC (Section 4) behind the arm-selection
      parameter, triggered by the existing nav goal. Software-in-the-loop against the
      existing scripted-merger fake world. Verify the bound is actually binding, that the
      goal is refused when no valid bound exists, and that switching to the Nominal or
      DREAM arm restores their previous behaviour bit-for-bit.
  S4  Threshold calibration run (Section 7).
  S5  Contingency second solve (Section 5) + timing check.
  S6  Hardware: static world first (no merger), then the full A/B (Section 9).
  S7  Report: metric table, calibration curve, v_occ(t) and r_total(t) traces aligned to
      the reveal instant, and an explicit deviations list (Section 8).

Safety rules from the DREAM deployment apply unchanged: supervisor node running, human
with kill switch, speed cap, standstill start. Adding a hard state constraint to a
previously unconstrained MPC is a real infeasibility risk — bench-test S3 with the robot
on blocks before it drives.

If any required input is missing or an audit finding contradicts this document, stop and
report rather than improvising.
```
