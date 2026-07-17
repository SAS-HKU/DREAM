# Implementation Prompt: Occlusion-Aware Contingency Planning on LIMO ROS 2 Humble

Use the following prompt as the complete task specification for the LIMO implementation agent.

---

You are a senior ROS 2 Humble autonomy engineer and motion-planning researcher working in the existing LIMO workspace. Implement and validate an occlusion-aware contingency planner based on:

- Paper PDF: `C:\Users\IMSE\Downloads\Occlusion-Aware_Contingency_Safety-Critical_Planning_for_Autonomous_Driving.pdf`
- Project page: <https://zack4417.github.io/oacp-website/>
- Open-access paper record: <https://arxiv.org/abs/2502.06359>
- Paper DOI: <https://doi.org/10.1109/TCYB.2025.3632366>
- Current workspace: `D:\limo_ros2-humble`

The goal is a reproducible ROS 2 simulation and a hardware-ready control path for an AgileX LIMO in this scenario: the ego travels in the left lane and passes a tall, large static obstacle in the middle lane. That obstacle hides the right lane. The ego must slow or maintain a cautious fallback while a potentially conflicting right-lane vehicle is hidden, then either brake/continue cautiously if a conflict appears or accelerate/progress when the occluded region is revealed to be clear.

## Operating rules

1. Inspect the repository and the cited paper before editing. Preserve existing user work and avoid unrelated rewrites.
2. Work in small, runnable phases. Build and test after each phase; report commands, results, remaining failures, and unexecuted checks.
3. Do not claim a paper-faithful reproduction unless every paper equation and assumption used by the implementation is traceable to code and tests.
4. Do not allow the planner to consume simulator ground truth for a hidden actor. Ground truth may be used only by the simulated-perception adapter and evaluator. The planner must receive the same visibility-limited interface intended for hardware.
5. Do not use Nav2/DWB, a hand-written speed heuristic, or a generic MPC as a substitute for the requested final planner. A simple controller may be used only for early bring-up or as an explicitly labeled emergency fallback.
6. Keep an independent command watchdog and emergency-stop path. If planning is stale, infeasible, nonfinite, or misses its deadline, command a bounded controlled stop.
7. Treat this as experimental safety research, not a certified safety guarantee. Document all unverified assumptions.
8. Do not begin physical-robot testing until simulation acceptance criteria pass and a human explicitly authorizes hardware use.

## Repository facts that must shape the implementation

The current workspace is a minimal platform repository, not a navigation stack:

- `limo_base` is the physical serial driver. It subscribes to absolute `/cmd_vel` and publishes `/odom`, `/imu`, and `/limo_status`.
- In Ackermann mode, `limo_base/src/limo_driver.cpp` interprets `Twist.linear.x = v` and `Twist.angular.z = yaw_rate`; it derives steering through `r = v / yaw_rate` and a wheelbase model. Do not publish a steering angle in `angular.z`.
- The physical driver constants are wheelbase `0.20 m`, track `0.172 m`, and maximum inner steering angle about `0.48869 rad`.
- The main `limo_car` Gazebo model uses wheelbase `0.24 m`, track `0.168 m`, a `100 Hz` Ackermann plugin, `/cmd_vel`, and `/odom`. This model/driver wheelbase mismatch must be made explicit and configurable; do not silently tune around it.
- The simulated 2-D lidar publishes `scan`, covers about `240 deg`, ranges from `0.2 m` to `8.0 m`, and updates at only `8 Hz`. The depth camera updates at `10 Hz`; the simulated IMU at `100 Hz`.
- `limo_car/launch/ackermann_gazebo.launch.py` points to `worlds/empty_world.model`, but `limo_car/worlds` is absent. `limo_car/CMakeLists.txt` also installs absent `log`, `src`, and `worlds` directories. Repair or supersede this launch path as part of Phase 0.
- `limo_bringup` is named in documentation and referenced by `open_ydlidar_launch.py`, but is absent from this folder.
- The repository snapshot is not a Git working tree. Do not rely on Git for rollback or status reporting.
- Topic names differ between simulation and physical launch conventions. Make state, scan, and command topics parameters; verify the actual ROS graph rather than assuming the `/wheel/odom` remap works with the driver's absolute `/odom` publisher.

Do not modify `limo_base` unless a verified interface defect makes it necessary. Prefer adapter nodes and parameters in new packages.

## Scientific fidelity contract

Create `PAPER_TRACEABILITY.md` and maintain a table with columns:

`Paper item | Meaning | Code symbol/file | Parameter source | Unit test/integration test | Fidelity status | Notes/deviation`

At minimum trace:

- Eq. (1): discrete Dubins-car state `[p_x, p_y, theta, v]` and controls `[yaw_rate, acceleration]`.
- Eqs. (4)-(6): elliptical spatiotemporal barrier and relative obstacle angle.
- Eqs. (7)-(9): order-10 Bezier/Bernstein trajectory parameterization.
- Eqs. (10)-(13): simplified reachability quantification (SRQ) for longitudinal and lateral occlusion risk.
- Eqs. (14)-(15): dynamic maximum velocity boundary.
- Eqs. (16)-(18): costs and biconvex constraints.
- Eqs. (19)-(31): augmented Lagrangian, primal/slack/consensus/dual updates.
- Algorithm 1: receding-horizon loop, residual threshold, and command application.
- Paper assumptions: free space is detected within the field of view; phantom vehicles stay in their lane, have constant velocity, and use a uniform speed distribution.

Use the paper's final IEEE version in the attached PDF as the primary mathematical source. The public project page is secondary evidence for timing and hardware setup. No official source-code release is provided by those sources, so do not invent undocumented implementation details.

The paper does not fully specify several integration choices. Mark the following as `implementation-specific`, not paper-derived:

- extracting occluded lane intervals from sensor/map geometry;
- associating and tracking visible objects;
- selecting the executed branch after new observations;
- mapping planned states to LIMO `/cmd_vel` commands;
- emergency behavior and planner-deadline handling;
- the requested three-lane scenario geometry.

Also document these paper ambiguities instead of silently resolving them:

- The article says the branch selector uses goal tracking, lateral deviation, safety, comfort, and consistency costs from prior work, but does not fully specify that selector here.
- The text/figures use exploration/fallback color labels inconsistently in places.
- With the same risk input and otherwise equal limits, Eq. (14) plus the reported maximum risk thresholds can produce unintuitive branch ordering. Log each branch's risk and velocity limit; add a semantic-ordering test; never swap thresholds merely to make a plot look right.

## Scenario correction required for a meaningful experiment

Three parallel lanes do not by themselves create a collision contingency if the ego stays in the left lane and the hidden vehicle stays in the right lane. The implementation must include both:

1. A **conflict scenario** in which the hidden right-lane actor's admissible route or forward reachable set merges into or crosses the ego corridor within the prediction horizon.
2. A **nonconflict control scenario** in which the hidden actor remains in a disjoint right-lane corridor and the FRS-intersection gate correctly suppresses irrelevant occlusion risk.

This distinction is mandatory. A demo that brakes simply because something is hidden, without a reachable conflict, does not validate the method.

Use a configurable, approximately 1:10-scale test geometry as the initial default, then tune only through YAML:

- world frame: `map`; ego travels in `+x`;
- three parallel lane centerlines near `y = +0.55, 0.0, -0.55 m` with configurable width around `0.45-0.50 m`;
- ego start near `x = -2.5 m` in the left lane; goal beyond `x = +2.5 m`;
- a static middle-lane occluder approximately `0.7-1.0 m` long, `0.30-0.40 m` wide, and tall enough to block the selected sensor rays;
- right-lane phantom maximum speed initially `1.0 m/s`, matching the paper's real-robot setting;
- nominal ego speed initially `0.5 m/s`, cautious speed around `0.3 m/s`, subject to measured LIMO limits;
- safe ellipse initial axes around `0.06 m` only as a paper-derived starting point; inflate them by robot footprint, localization uncertainty, and tracking uncertainty before using them as a separation requirement.

Provide at least these deterministic scenario variants with fixed seeds:

- `clear_no_phantom`: occluded region is empty; ego cautiously approaches then progresses after visibility clears.
- `hidden_nonconflicting`: an actor exists in the right lane but its FRS is disjoint from the ego corridor; risk gating avoids unnecessary braking.
- `hidden_conflicting_merge`: the hidden actor can merge/cross into the ego corridor; ego slows before exposure and chooses/maintains the fallback.
- `late_fast_emergence`: a worst-case actor appears near the configured phantom-speed bound; safety supervisor remains collision-free or reaches a stopped safe state.
- `visible_conflicting`: same conflict without occlusion, used to separate tracking/control behavior from occlusion reasoning.
- `occlusion_ignorant_ablation`: same planner with occlusion risk disabled, for comparison only.

## Package and component architecture

Create modular ROS 2 Humble packages rather than one monolithic node. A recommended layout is:

### `limo_oacp_msgs`

Use a new interface package or carefully extend `limo_msgs`. Define only the messages needed to avoid a heavy `vision_msgs` dependency. Suggested interfaces:

- `TrackedObject.msg`: ID, timestamp, pose, twist, footprint dimensions/polygon, classification, covariance or conservative uncertainty bounds.
- `TrackedObjectArray.msg`.
- `OccludedLaneSegment.msg`: lane ID, `s_start`, `s_end`, lane width, maximum phantom speed, confidence, source frame, and active flag.
- `OacpPlannerStatus.msg`: risk values, exploration/fallback velocity bounds, branch, solver status, solve time, iteration count, residuals, command age, and fail-safe reason.

### `limo_oacp_perception`

Implement two replaceable adapters behind the same output contract:

- A simulation adapter may subscribe to Gazebo model states but must ray-cast against configured occluder polygons and publish only visible objects to the planner-facing topic.
- A hardware adapter consumes real tracked objects/free-space data when available. For the initial milestone it may be a documented stub, but planner interfaces must not require Gazebo types.
- An occlusion extractor combines ego pose, lane centerlines, sensor field of view/range, and static occluder polygons to publish lane-coordinate occluded intervals.

Add an explicit information-flow test: no planner/controller executable may subscribe to raw Gazebo model states or the evaluator's ground-truth topics.

### `limo_oacp_planner`

Implement separable, testable libraries/classes:

- `BernsteinBasis` / `BezierTrajectory`;
- `OcclusionRiskAssessor` implementing SRQ and FRS-intersection gating;
- `SpatiotemporalBarrier`;
- `ConsensusAdmmSolver`;
- `ContingencyPlanner` producing exploration and fallback trajectories with a shared initial segment;
- `TrajectorySelector` with a transparent, configurable implementation-specific policy;
- `SafetySupervisor` for validation, stale-plan handling, and controlled stop.

The planner node should publish both branches, the shared segment, selected path, RViz markers, and full status. Keep the numerical core independent of ROS messages so it can be unit-tested.

### `limo_oacp_control`

Implement trajectory tracking and a watchdog:

- subscribe to the selected trajectory and odometry;
- use feedforward curvature/yaw-rate plus bounded feedback or another clearly documented LIMO-appropriate tracker;
- publish `geometry_msgs/msg/Twist` to a configurable command topic, default `/cmd_vel`;
- set `linear.x = commanded longitudinal speed` and `angular.z = commanded yaw rate = v * curvature`;
- never put steering angle directly in `angular.z`;
- guard zero/near-zero speed, saturate yaw rate/curvature/acceleration, and continuously publish zero during a stop;
- stop on stale odometry, stale trajectory, solver failure, nonfinite values, excessive tracking error, emergency stop, or node shutdown.

### `limo_oacp_sim`

Provide:

- a valid Gazebo Classic world and launch file for the requested three-lane scenario;
- models for the large static occluder and scripted visible/phantom actors;
- scenario YAML files, deterministic actor routes, and reset support;
- RViz configuration showing visible objects, occluded intervals, phantom reachable sets/risk field, both trajectories, shared segment, selected branch, safety ellipses, and text status;
- an evaluator that may consume ground truth but never republishes it on planner-facing topics;
- rosbag/CSV logging and repeatable batch trials.

### `limo_oacp_bringup`

Provide separate launch files for:

- simulation;
- planner/controller with recorded inputs;
- hardware dry-run with commands disabled;
- hardware operation with an explicit `enable_motion:=true` gate.

If fewer packages are chosen, preserve these dependency boundaries as components/libraries and explain the reason in the README.

## ROS interface contract

Use parameters/remappings rather than absolute names inside new code. Recommended defaults:

| Direction | Topic | Type | Notes |
|---|---|---|---|
| input | `/odom` | `nav_msgs/msg/Odometry` | configurable; verify simulation/hardware graph |
| input | `/scan` | `sensor_msgs/msg/LaserScan` | optional validation/input to occlusion adapter |
| input | `/oacp/visible_objects` | custom array | never contains hidden ground truth |
| input | `/oacp/occluded_segments` | custom array/message | lane-coordinate PVS input |
| input | `/oacp/emergency_stop` | `std_msgs/msg/Bool` | latched/safety QoS as appropriate |
| output | `/oacp/exploration_path` | `nav_msgs/msg/Path` | full horizon |
| output | `/oacp/fallback_path` | `nav_msgs/msg/Path` | full horizon |
| output | `/oacp/shared_path` | `nav_msgs/msg/Path` | first `N_s` points |
| output | `/oacp/selected_path` | trajectory/path contract | must include time/speed, not positions alone |
| output | `/oacp/risk_markers` | `visualization_msgs/msg/MarkerArray` | risk/PVS/FRS visualization |
| output | `/oacp/status` | custom status | solver and safety diagnostics |
| output | `/cmd_vel` | `geometry_msgs/msg/Twist` | only controller/supervisor publishes in normal operation |

Avoid two normal-operation publishers racing on `/cmd_vel`. Use a mux or make the supervisor the sole final publisher.

## Algorithm requirements

### 1. State, timing, and scaling

Final planner requirements:

- order-10 Bezier trajectories;
- two branches: exploration `j=0` and fallback `j=1`;
- planning horizon `N = 40`, `dt = 0.1 s`, replanning at `10 Hz` as the initial paper-derived configuration;
- shared horizon `N_s = 5` and free steps `N_d = 5` initially;
- maximum ADMM iterations `200` and relative primal residual threshold `0.1` initially;
- barrier coefficient schedule initially linear from `0.4` to `1.0` across the horizon;
- warm-start from the shifted previous solution;
- use Eigen's stable decompositions, including `HouseholderQR` or an equivalent documented stable factorization, for the paper's linear systems;
- all units in SI and parameters in YAML with declared ranges and validation.

The paper's simulation-scale weights are starting points, not automatically valid LIMO tuning: `Q_theta=150`, `Q_x=100`, `Q_y=100`, `Q_1=50`, `Q_2=100`, with ADMM penalties initially `5`. Keep them configurable and log the exact set used per trial.

### 2. Occlusion risk and reachable-set gating

- Represent each occluded lane segment as the PVS interval `[s_s, s_e]`.
- Implement the piecewise longitudinal risk from Eq. (10), its PVS-length scaling in Eq. (11), the lateral normal model in Eq. (12), and the product risk in Eq. (13).
- Use configurable prediction horizon, lane width, confidence factor, phantom maximum speed, and aggregated-risk rule.
- Implement Eq. (14)-(15) with explicit saturation, continuity checks, and dimensional tests.
- Compute conservative phantom reachable occupancy over the horizon.
- Ignore an occluded actor/segment only if its FRS provably does not intersect the ego candidate corridor/FRS within the horizon. Test this on the nonconflict scenario.
- Publish the risk field, PVS, and active FRS constraints for inspection.

### 3. Dual-trajectory optimization

- Enforce common initial state and terminal lane/heading conditions.
- Enforce the discrete vehicle dynamics, bounded velocity, acceleration, jerk, curvature/yaw rate, and workspace/lane constraints.
- Enforce obstacle separation using the paper's elliptical spatiotemporal barrier, inflated for the LIMO footprint and uncertainty.
- Enforce exploration/fallback consistency over the first `N_s` steps in position, velocity, acceleration, and heading.
- Update orientation, longitudinal, lateral, barrier variables, consensus variables, slack variables, and dual variables in the order described by Algorithm 1 and Eqs. (21)-(31).
- Track both primal and dual residuals, time budget, convergence reason, constraint violations, and objective terms.
- Verify every candidate after solving. Never publish a trajectory whose hard safety, dynamics, bound, or finite-value checks fail.

Build a simple fixed-lane, speed-only dual-branch vertical slice first if useful, but the final default must use the full dual Bezier/ADMM planner. Label the vertical slice as bring-up code and do not use it for final paper-fidelity claims.

### 4. Branch selection and execution

Because the paper's complete selector is under-specified, implement and document a transparent policy:

- before the visibility event, only execute commands belonging to the numerically verified common segment;
- after an actor becomes visible, select fallback when its predicted occupancy conflicts with the ego corridor or safety margin;
- select exploration/progress only when the newly visible free space and predicted objects satisfy clearance for the required horizon;
- retain hysteresis/minimum hold time to prevent branch chattering;
- if evidence is ambiguous, stale, or contradictory, choose the verified fallback or controlled stop;
- log every branch decision and the exact evidence/cost terms that caused it.

Do not describe this selector as paper-derived. If you implement the prior-work selector, cite and trace that source separately.

## Tests required before integration

Use `ament_cmake_gtest` or an equivalent ROS 2 test setup. At minimum add:

### Mathematical unit tests

- Bernstein basis partition of unity, endpoint interpolation, and derivative matrices.
- SRQ Eq. (10) branch boundaries and continuity at `s_s` and `s_e`.
- lateral risk symmetry and maximum at lane center.
- velocity boundary monotonicity, saturation, continuity, and semantic-ordering diagnostics for both branches.
- barrier sign and safety-ellipse geometry.
- FRS-intersection gate: intersecting, tangent, and disjoint cases.
- common-segment equality in position, velocity, acceleration, and heading.
- ADMM residual calculation, stopping rules, maximum-iteration exit, infeasibility, and warm start.
- solver outputs contain no NaN/Inf and satisfy constraints within configured tolerances.
- LIMO command conversion uses yaw rate, respects bounds, and behaves safely at zero speed.

### ROS integration tests

- launch graph and topic/type contract;
- only the final command component publishes `/cmd_vel`;
- planner has no raw Gazebo-ground-truth subscription;
- sensor/odometry/trajectory timeout produces a controlled stop;
- visibility transition changes planner inputs only when line of sight actually clears;
- deterministic reset produces repeatable initial conditions and metrics.

## Evaluation design and acceptance criteria

Create an experiment runner that executes every scenario and ablation for at least 30 deterministic trials or explains why a smaller smoke-test count was used during development. Write raw per-timestep CSV plus a per-run summary containing:

- collision and minimum footprint separation;
- task completion and duration;
- ego speed/acceleration/jerk and maximum tracking error;
- visibility state, occluded interval, risk, and both dynamic velocity limits;
- selected branch and branch-switch count;
- solver time, iterations, residuals, infeasible count, deadline misses;
- emergency-stop and watchdog activations;
- seed and full parameter-file hash/content identifier.

Initial acceptance gates:

1. OACP has zero collisions in all deterministic conflict, late-emergence, and visible-conflict trials.
2. Minimum separation never violates the configured inflated safety boundary.
3. The ego begins risk-responsive slowing before the occluded conflict becomes fully visible; it does not wait for ground-truth revelation.
4. In clear trials, the ego returns toward nominal speed after the occluded interval clears, without oscillatory branch switching.
5. In nonconflicting hidden trials, FRS gating avoids braking attributable solely to a disjoint hidden actor.
6. The first `N_s` samples of both branches agree within declared numerical tolerance for position, velocity, acceleration, and heading.
7. The 10 Hz planning loop has no stale commands. Development gate: `p95 < 100 ms`; target: mean below `50 ms` and no hard deadline miss, with paper results reported only as context rather than a guaranteed LIMO benchmark.
8. No invalid trajectory is published; every solver failure produces a controlled stop.
9. The occlusion-ignorant ablation demonstrates worse risk behavior in the conflict scenario, but do not deliberately allow a damaging physical collision; simulation only.
10. Results report mean, standard deviation, maximum, minimum, and trial count. Do not cherry-pick a successful video.

Also compare at least:

- OACP full method;
- occlusion-ignorant ablation;
- conservative stop-before-occlusion baseline;
- fixed-speed or ordinary tracking baseline, if safe in simulation.

## Implementation phases and mandatory checkpoints

### Phase 0 - Baseline audit and repair

- Inventory ROS packages, topics, frames, model dimensions, plugins, missing paths, and build dependencies.
- Make the existing Ackermann simulation launchable or create a clean superseding launch without breaking original files.
- Resolve/configure the `0.20 m` driver versus `0.24 m` simulation wheelbase mismatch.
- Verify manual bounded `/cmd_vel` motion and odometry.
- Deliver `BASELINE_AUDIT.md` and exact build/launch commands.

Checkpoint: do not implement the optimizer until the vehicle, frames, time source, and command semantics are verified.

### Phase 1 - Scenario, visibility, and logging

- Add the three-lane world, occluder, scripted actors, scenario variants, visibility-limited object output, occluded-lane extraction, RViz, reset, and evaluator.
- Prove with a topic audit that hidden ground truth does not reach the planner-facing interface.

Checkpoint: show that the actor is absent from visible-object output while geometrically occluded and appears after line of sight clears.

### Phase 2 - SRQ and safe vertical slice

- Implement/test Eqs. (10)-(15), phantom occupancy, and FRS gate.
- Add the fixed-lane dual speed vertical slice and safety supervisor to validate brake/progress behavior.

Checkpoint: clear, disjoint, and conflicting cases pass their behavioral tests.

### Phase 3 - Full Bezier consensus-ADMM planner

- Implement/test Eqs. (4)-(31), common-segment constraints, warm start, verifier, and diagnostics.
- Compare numerical outputs against small offline reference problems.

Checkpoint: full method passes mathematical tests and runs at the development timing gate.

### Phase 4 - Controller and closed-loop evaluation

- Integrate tracking, command arbitration, watchdogs, batch trials, baselines, and reports.
- Tune only through versioned YAML; record every final parameter.

Checkpoint: all simulation acceptance gates pass, or provide a precise failure report with logs and next fixes.

### Phase 5 - Hardware-ready dry run

- Add hardware topic/frame adapter, command-disable launch, calibration procedure, low-speed limit, emergency stop, and operator checklist.
- Replay recorded data before enabling motion.

Checkpoint: stop and request explicit human authorization before sending motion commands to a physical LIMO.

## Required deliverables

- Buildable ROS 2 Humble source packages and manifests.
- Scenario world/model/config/launch files.
- Unit, integration, and launch tests.
- `README.md` with dependency installation, build, launch, scenario, replay, and troubleshooting commands.
- `BASELINE_AUDIT.md`.
- `PAPER_TRACEABILITY.md`.
- `SAFETY_CASE.md` covering assumptions, hazards, watchdogs, stop behavior, and hardware gate.
- `EXPERIMENT_PLAN.md` and `RESULTS.md` with raw-data locations and statistical summaries.
- RViz configuration and a short demonstration recording or rosbag if the environment supports it.
- A final change summary listing files changed, commands run, test results, timing results, known limitations, and every item not actually verified.

## Build and verification expectations

Run commands appropriate to the actual Ubuntu/ROS 2 workspace. Prefer:

```bash
rosdep install --from-paths . --ignore-src -r -y
colcon build --symlink-install --event-handlers console_direct+
source install/setup.bash
colcon test --event-handlers console_direct+
colcon test-result --verbose
```

If the packages are nested under a workspace `src/`, adjust paths accordingly. Do not report a build or test as passing unless it was executed. If the current host lacks ROS 2/Gazebo, still implement the code and static/unit-testable core, then clearly mark ROS/Gazebo tests `NOT RUN` and give exact commands for the LIMO Ubuntu environment.

## Start now

Begin with Phase 0. First present a concise baseline audit and proposed file/package tree, then make the smallest changes needed to obtain a verified Ackermann simulation and command path. Continue phase by phase without skipping tests. Stop only for a genuinely blocking choice, a safety authorization boundary, or an environment limitation that cannot be worked around. Do not jump directly to a large untested optimizer implementation.

---
