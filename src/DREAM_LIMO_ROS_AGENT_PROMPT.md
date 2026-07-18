# LIMO ROS2 Agent Prompt: DREAM Occlusion-Aware Planner Deployment

Use this prompt for the agent running on the LIMO ROS2 platform. It was written after a
proof-read of the DREAM development repo (`C:\DREAM_final` on the development machine)
and reflects the actual module inventory, known code caveats, and the target validation
scenario.

```text
You are working on a ROS2 Humble LIMO robot platform (AgileX LIMO). Your task is to plan
and then implement the deployment of the DREAM occlusion-aware planning framework
(PDE risk-field transmission + maneuver-level decision gating + MPC-CBF control) onto
this platform, and to validate it in a physical occluded-merge scenario. You must first
inspect the onboard ROS structure and report what actually exists. Do not run autonomous
motion until the audit, topic/frame checks, and safety plan are complete.

========================================================================
1. WHAT DREAM IS (algorithm summary — treat as ground truth for design)
========================================================================
DREAM couples three layers, all validated in simulation in the development repo:

(A) DRIFT risk field — a 2D PDE on a world-fixed grid:
      tau*d2R/dt2 + dR/dt + div(v R) = div(D grad R) + Q - lambda*R
    - Source Q = Q_veh + Q_occ + Q_merge:
      * Q_veh: anisotropic Gaussian kernels per vehicle, weighted by class
        (truck > car), distance, relative speed; amplified strongly when a lead
        vehicle brakes; plus an "approach corridor" term between ego and a slower
        leader when closing speed is high.
      * Q_occ: latent-hazard source injected inside an occlusion shadow region.
        IMPORTANT CAVEAT: in the sim code the shadow cone extends AHEAD of the
        occluding truck along the TRUCK's heading (not the ego's line of sight).
        A separate ego-viewpoint shadow polygon (compute_truck_shadow) exists only
        for visualization/reveal gating. On the robot you have a real lidar, so
        the DEPLOYED Q_occ should be driven by actual sensed visibility: cells of
        the grid that fall in lidar shadow (blocked line-of-sight) near/inside the
        drivable corridor get the occlusion source. This is more principled than
        the sim heuristic and is the preferred design; keep the sim cone available
        as a fallback flag for A/B comparison.
      * Q_merge: static merge-topology prior gated by local vehicle density —
        OPTIONAL for the LIMO scenario; can be disabled (it models a highway ramp).
    - Advection velocity v: Gaussian-weighted average of surrounding vehicles'
      absolute velocities (risk moves with traffic).
    - Diffusion D: base D0, boosted inside occluded regions (D_occ) and around
      braking vehicles. Decay lambda: base + |v|/L_decay, plus a sponge layer at
      the downstream grid edge. A road mask enforces R=0 off-road (Dirichlet).
    - Solved explicitly with operator splitting + upwind flux advection,
      substeps=3 per 0.1 s control step. Telegrapher term tau=0.2 (tau=0 is a
      supported, more robust fallback).
    - Cold start: warm up the field ~5 s before releasing the planner.

(B) Decision layer (IDEAM graph decision + DREAM risk veto):
    - IDEAM formulates 6 gap groups (L1,L2,C1,C2,R1,R2) from per-lane
      leader/follower arrays and runs a DFS over a fixed adjacency graph with
      kinematic risk/gap feasibility checks, picking a target group by
      long/short-term efficiency.
    - DREAM adds a veto: before executing a lane change, sample the risk field
      along a straight-line interpolated lane-change path (default 30 m lookahead,
      10 samples in sim; scale for LIMO) and compute
      score = 0.6*max + 0.4*mean. If score > decision_risk_threshold, force
      lane-keep ("K"). Presets: conservative=1.0, balanced=1.5, permissive=2.0.

(C) MPC-CBF control (LMPC, horizon T=30, dt=0.1 in sim; cvxpy + casadi):
    - Risk enters the MPC cost as risk_weight * 0.1 * R(x_t) * vx_t^2 — i.e. the
      field slows the ego in risky regions (positional avoidance comes from the
      veto and CBF, not this term — do not "fix" this, it is by design).
    - CBF safety ellipses around leader/follower vehicles are EXPANDED by local
      risk: scale = clip(1 + alpha * min(R/risk_norm, 1), 1, max_scale). Time
      headway and min-distance are modulated the same way.
    - Ellipse tangent linearization uses scipy fsolve per constraint (HOCBF.py) —
      this is a known hotspot; profile it on the robot and cache/replace with the
      closed-form nearest-intersection initial guess if needed.

========================================================================
2. SOURCE INVENTORY (on the development machine — verify what was transferred)
========================================================================
Do NOT assume C:\DREAM_final exists on the robot. First discover what has been
transferred (search ~/ros2_ws/src, ~, /opt for "dream", "drift", "prideam",
"ideam"). If missing, STOP and report the exact list to request. Required:

  config.py                      — grid + PDE params (WORLD-FIXED, highway-scale)
  pde_solver.py                  — PDE solver + Q_veh/Q_occ/Q_merge (640 lines)
  Integration/drift_interface.py — DRIFTInterface: step(), warmup(), risk queries
                                   (Cartesian + Frenet), gradient, CBF/headway
                                   modulation, evaluate_lane_change_risk()
  Integration/prideam_controller.py — PRIDEAMController: solve_with_risk(),
                                   gate_lane_change(), risk modulation wrappers
  Integration/integration_config.py — presets baseline/conservative/balanced/
                                   permissive/dense
  Control/MPC.py, Control/constraint_params.py, Control/HOCBF.py, Control/utils.py
  DecisionMaking/{decision.py, decision_params.py, give_desired_path.py, util.py,
                  util_params.py}
  Model/{Dynamical_model.py, params.py, Surrounding_model.py, surrounding_params.py}
  Prediction/surrounding_prediction.py
  Path/  (path objects; sim geometry — you will REPLACE these with LIMO arena paths)
  uncertainty_merger_DREAM.py    — reference scenario + agent-step loop pattern
                                   (read dream_agent_step() and the main loop lane
                                   array construction; this is the control-flow
                                   template for the ROS node)

Python deps: numpy, scipy, cvxpy (+ a QP solver that works on ARM, e.g. OSQP),
casadi, matplotlib (viz only). NO torch needed for the deployed planner (RL arms
are not being deployed). If cvxpy/casadi wheels are unavailable for the onboard
architecture, report before improvising.

KNOWN CODE CAVEATS FROM PROOF-READING (respect these when wrapping):
  1. Do NOT use PRIDEAMController.update_risk_field() — its ego-state parsing is
     unreliable (it can silently default ego speed to 15 m/s). Call
     controller.drift.step(vehicles, ego, dt, substeps=3) directly with an
     explicit DRIFT ego dict, exactly as uncertainty_merger_DREAM.py does.
  2. create_vehicle() derives heading from velocity — a STATIC obstacle gets
     heading 0. Always set v['heading'] explicitly (road tangent), and tag the
     large obstacle vclass='truck' or the sim-fallback occlusion source will
     ignore it.
  3. Many length constants are HARD-CODED inside pde_solver.py, not in config.py:
     Q_veh distance range exp(-d/70), corridor half-width 4.0 m, occlusion range
     60 m and decay exp(-d/30), velocity-kernel scales (400, 9), braking diffusion
     radius, merge-zone geometry, get_decision_risk_score lane_width default 3.5
     (vs cfg.lane_width 4.0 — a latent inconsistency). Scaling the framework to
     LIMO size REQUIRES touching these; make a single scaled copy of
     pde_solver.py/config.py with every length parameterized by the scale factor,
     and unit-test that the scaled field reproduces the sim field shape.
  4. constraint_params(): MIN_SPEED = 2.0 m/s — the MPC cannot command a stop.
     For a physical merge-conflict test the ego MUST be able to brake to zero.
     Set MIN_SPEED = 0 in the LIMO parameter set and re-verify solver behavior.
  5. The decision veto samples the lane-change path with a fixed lane_width and a
     sign convention (target_ey = ego_ey + (target-current)*lane_width). Verify
     the ey sign convention against your LIMO path definitions before trusting
     the veto direction (left vs right).
  6. PDE stability: explicit scheme; effective CFL for diffusion requires
     sub_dt < dx^2/(4*D_max). The sim runs at the margin. After rescaling, verify
     numerically (inject a point source, confirm no oscillation/blow-up).
  7. Duplicate modules exist (top-level pde_solver.py/config.py vs DRIFT/ stubs).
     Vendor ONE canonical copy inside your ROS package to avoid import ambiguity.

========================================================================
3. SCALING TO LIMO (highway -> tabletop; do this before any ROS code)
========================================================================
The framework is dimensional. Choose a length scale alpha and time scale beta
(sim quantity -> LIMO quantity):
    position x -> alpha*x        speed v -> (alpha/beta)*v
    accel a -> (alpha/beta^2)*a  diffusion D -> (alpha^2/beta)*D
    decay lambda -> lambda/beta  tau -> beta*tau      dt -> beta*dt
Under this scaling the PDE and CFL number are invariant, so the field dynamics
are preserved exactly.

Recommended starting point: alpha = 1/10, beta = 2 (verify against your arena
and LIMO limits, then commit one set):
    lane width 4.0 m -> 0.40 m (widen to 0.45-0.50 m if the arena allows;
                                LIMO is ~0.22 m wide)
    ego speed 10.2 -> ~0.5 m/s; truck/obstacle: static or ~0.25 m/s
    obstacle box: 12 x 2 m -> 1.2 x 0.2 m (a cardboard box tall enough to
                                block the lidar plane — CRITICAL: occlusion must
                                be real at sensor height)
    dt: 0.1 -> 0.2 s control step (planner 5 Hz; low-level tracking at >= 20 Hz)
    sigma_x 8 -> 0.8 m, sigma_y 2.5 -> 0.25 m, L_decay 25 -> 2.5 m
    D0 0.3 -> 0.015, D_occ 6.0 -> 0.30  (alpha^2/beta = 0.005)
    lambda_decay 0.15 -> 0.075, tau 0.2 -> 0.4 (or 0 for robustness)
    occlusion range 60 -> 6 m, occlusion decay 30 -> 3 m
    veto lookahead 30 -> 3 m, headway d0 5 -> 0.5 m
    grid: cover the arena (e.g. 6 x 2 m) at dx ~ 2.5 cm -> ~240 x 80 cells,
          comparable compute to the sim grid
    MPC: T=30 at dt=0.2 is a 6 s horizon — likely too long/slow; start with
         T=10-15. Rescale MAX_SPEED/MAX_ACCEL/steer-rate limits to LIMO
         capability (max ~1 m/s; Ackermann steer limit and min turn radius
         ~0.4 m — measure, don't assume). Wheelbase lf/lr: LIMO wheelbase is
         0.2 m -> lf=lr=0.1. MIN_SPEED=0 (see caveat 4).
Deliverable of this phase: a `limo_scale.py` (single source of truth for alpha,
beta and every derived parameter) + a headless Python replay of the merger
scenario at LIMO scale, run OFF-robot, demonstrating: field warms up, veto fires
while the right lane is occluded, ego brakes/yields when the merger appears.
Do not write ROS code until this replay passes.

========================================================================
4. TARGET VALIDATION SCENARIO (physical)
========================================================================
Three parallel straight lanes on the floor. Frame: map/odom origin at arena
corner, x along lanes.

  - EGO (LIMO, DREAM stack): starts in LEFT lane, cruising ~0.5 m/s.
  - OBSTACLE: large static box in the MIDDLE lane (the "truck"). It blocks
    lidar line-of-sight to the RIGHT lane region beside/ahead of it.
  - BLOCKER (static box or very slow second object) in the LEFT lane ~2-3 m
    beyond the obstacle: forces the ego to plan a merge into the MIDDLE lane
    after passing the obstacle (this reproduces the sim's blocker-forced merge
    conflict — without it the ego would just stay in the left lane).
  - MERGER (second LIMO or manually pushed robot) in the RIGHT lane, initially
    hidden behind the obstacle, which merges into the MIDDLE lane as/after the
    ego passes — contesting the same gap ahead of the obstacle.

Expected DREAM behavior (this is the claim being validated):
  - While the right lane is occluded, Q_occ builds risk in the shadow zone; the
    decision veto delays/blocks the ego's merge into the middle lane and the
    risk-in-cost + CBF expansion slow the ego near the occlusion boundary.
  - When visibility changes (ego passes the obstacle / merger emerges), the
    ego already carries margin: larger TTC at reveal, no hard brake, no
    collision. Baseline (veto/cost/CBF disabled = "baseline" preset) should
    show later reaction and smaller post-reveal margin. Run BOTH presets with
    identical scripted merger timing; that A/B is the experiment.

Perception for the merger (choose by what exists on the robot — audit first):
  Option A (preferred if two ROS2 robots share a network): subscribe to the
    merger robot's odom over a shared ROS_DOMAIN_ID, but pass it through a
    VISIBILITY GATE node: the track is forwarded to the planner ONLY when the
    line segment ego->merger does not intersect the obstacle footprint polygon
    (and optionally only when lidar returns actually confirm it). This gives
    honest occlusion without a full detection pipeline. The DRIFT field gets
    the gated track too — the hidden merger must NOT leak into Q_veh; only
    Q_occ (from sensed shadow) represents it while hidden.
  Option B: onboard lidar clustering (scan -> euclidean clusters -> tracked
    dynamic obstacle with constant-velocity filter). More work; occlusion is
    physically automatic. Acceptable if Option A infrastructure is absent.
  The static obstacle and blocker positions may be given as a small YAML map
  (surveyed once) — do not burn time on SLAM for v1; odom drift over a short
  run is acceptable, but measure it first.

Metrics to log (mirror the sim, rosbag everything):
  - t_reveal (first step merger is visible), TTC to merger at reveal,
    min clearance in the 3 s post-reveal window, min TTC overall,
    ego speed/accel profile (comfort: max |a|, jerk), veto activations,
    risk at ego R(x_ego,t), per-step compute times (t_drift, t_decision, t_mpc).
  - Success criteria per run: no contact; min clearance >= 1 robot width;
    DREAM TTC-at-reveal > baseline TTC-at-reveal across >= 5 repeated runs.

========================================================================
5. ROS2 ARCHITECTURE (build a new package, e.g. `dream_limo`)
========================================================================
Nodes (start simple; one Python package, composable later):
  1. state_estimator: consumes /odom (NOTE: limo_base launch remaps odom to
     /wheel/odom — VERIFY actual runtime topic) + /imu; publishes ego pose/twist
     in the arena frame (static TF from a measured start pose is fine for v1).
  2. world_model: static obstacle/blocker YAML + merger track (Option A gate or
     Option B tracker) -> publishes a vehicle list (custom msg or
     visualization-friendly array) + occlusion shadow polygon from the ego pose
     and obstacle footprint (or directly from lidar free-space).
  3. drift_field_node (5 Hz): wraps DRIFTInterface.step() with the scaled
     config; inputs: vehicle list + ego dict + shadow mask; outputs: risk field
     as nav_msgs/OccupancyGrid (scaled 0-100) for RViz + a latched service or
     shared-memory handle for the planner. Warm up ~5 s before signaling READY.
  4. dream_planner_node (5 Hz): ports dream_agent_step() from
     uncertainty_merger_DREAM.py: build per-lane leader/follower arrays from the
     world model, run IDEAM decision + risk veto (skip the sim's force_* lane
     scaffolding — that exists only for reproducible figures; the veto must be
     LIVE, i.e. enable_decision_veto=True and no force_ignore_veto), then
     PRIDEAMController.solve_with_risk(); output a short reference trajectory
     (or directly v, steer for step 5).
  5. tracker_node (>= 20 Hz): converts the MPC's (accel, steer) or reference
     trajectory into geometry_msgs/Twist on /cmd_vel. Put the LIMO in Ackermann
     mode (the sim model is a bicycle model; verify how limo_base maps Twist to
     steering in ackermann mode — likely angular.z interpreted as steer via
     inner conversion; check limo_driver source). Enforce hard caps here:
     |v| <= 0.6 m/s, accel slew limit, and a watchdog that zeroes cmd_vel if
     planner messages stop for > 0.5 s.
  6. safety_supervisor: independent of the planner — lidar bubble stop
     (min range in front sector < 0.25 m -> zero velocity latch), external
     kill switch (teleop key or joystick deadman). This node must be running
     before ANY autonomous test.

Frames: map (arena) -> odom -> base_link; risk grid in map frame. Keep the PDE
grid axis-aligned with the lanes so lane arrays and Frenet paths stay trivial
(straight-line path objects: implement get_cartesian_coords(s, ey),
get_theta_r(s), __call__(s) — ~30 lines; replaces the sim Path/ package).

Compute budget: profile FIRST on the actual onboard computer (likely Jetson
Nano — verify). Sim timing shows MPC solve dominates. If solve time > 100 ms:
reduce T, warm-start from previous solution (already supported via last_X),
replace fsolve in HOCBF with the closed-form initial guess, or move planner
off-board (WiFi laptop) publishing /cmd_vel — acceptable for validation, note
latency in the report.

========================================================================
6. STAGED PLAN (each stage gates the next; report results at each gate)
========================================================================
  Stage 0 — Audit: onboard packages, topics (ros2 topic list with base
    running), lidar model/height vs obstacle height, compute platform, Python
    env, which DREAM sources are present. Report findings before coding.
  Stage 1 — Scaled headless replay (Section 3 deliverable), off-robot.
  Stage 2 — SIL: run the full node graph against a fake world_model that
    scripts the merger trajectory (no Gazebo needed; the limo_car Gazebo pkg
    is optional and its world files may be broken — do not block on it).
    Verify veto fires, field renders in RViz, cmd_vel stream is sane.
  Stage 3 — Hardware, static world only: ego drives the left lane past the
    obstacle, no merger. Validate estimation drift, tracking, timing, safety
    supervisor, occlusion shadow correctness in RViz.
  Stage 4 — Full scenario: add merger (Option A gate). >= 5 runs DREAM
    ("balanced" preset scaled) + >= 5 runs baseline preset, identical merger
    script. Collect rosbags.
  Stage 5 — Report: metric table (Section 4), plots (TTC around reveal, risk
    at ego, speed profile), and a list of parameter deviations from the scaled
    sim values with justification.

SAFETY RULES (non-negotiable): no autonomous motion before Stage 3 gate;
supervisor node + human with kill switch present for every run; speed hard cap
0.6 m/s; runs start from a standstill with a 3 s countdown; any contact or
supervisor trigger -> stop, root-cause, and report before re-running.

If any required artifact, dependency, or hardware capability is missing, stop
and report the exact gap rather than substituting from memory.
```
