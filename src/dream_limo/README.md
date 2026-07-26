# DREAM LIMO ROS 2

`dream_limo` deploys DREAM on a ROS 2 Humble AgileX LIMO. The primary mode is
free-space navigation: start the stack, use RViz **2D Goal Pose** to select any
currently observed-free, reachable point, and DREAM follows an obstacle-free
path while accounting for LiDAR occlusion and a suddenly revealed moving
vehicle.

There is no mission distance, lane coordinate, occluder pose, or terminal goal
to configure. The clicked pose is validated against the live LiDAR costmap.
Unknown/occluded cells and occupied/inflated cells are rejected.

The package is a sibling of `sfg_nav`. It does not modify or run the SFG
planner. It reuses only SFG's public, class-neutral LiDAR clustering executable
and message convention; DREAM owns tracking, risk, decision-making, MPC, and
all command output.

## What runs

The primary launch is `dream_free_navigation.launch.py`:

1. `/scan` is converted to a world-fixed observed-free/occupied costmap. A
   planner-only Nav2 SMAC Hybrid node computes Ackermann-feasible geometry.
   No Nav2 controller, behavior tree, navigator, or velocity publisher runs.
   The shared launch replans geometry at 1 Hz for every comparison arm; local
   MPC and swept-costmap validation continue at 5 Hz.
2. SFG's neutral `/sfg/lidar_clusters` are classified over time by
   `dream_vehicle_tracker`. Only motion-confirmed, vehicle-sized tracks are
   forwarded on `/tracked_agents`; the SFG pedestrian detector and SFG planner
   are not started.
3. `dream_world_model` derives visibility and an occlusion mask directly from
   LiDAR first returns. A hidden merger never leaks into the visible-vehicle
   track list.
4. The selected arm changes only the occlusion-risk channel. `balanced` runs
   the warmed DRIFT PDE; `oacp_vb` runs phantom-vehicle reachability and a
   dynamic velocity bound; `nominal` runs no occlusion-risk assessor.
5. `dream_free_planner` tracks the arbitrary geometric path with the shared
   bicycle MPC inside a hard route tube. Every solved footprint and the swept
   motion between MPC knots are rechecked against the fresh inflated costmap.
   In `balanced`, route-level risk veto, risk cost, and risk-expanded
   CBF/headway are active. In `oacp_vb`, DREAM risk is zero, the base
   CBF/headway stays fixed, and phantom risk caps speed through the MPC. In
   `nominal`, every occlusion-risk channel is disabled. `pure_mpc` remains a
   legacy compatibility mode, not a separate scientific arm.
6. Independent collision, front-bubble, watchdog, drive-mode, ownership, and
   hardware gates are the only route to `/cmd_vel`.
7. The front camera is shown and recordable as occlusion evidence. Camera
   pixels do not enter the planner.

## Verified platform and solver

- Ubuntu 22.04, ROS 2 Humble, Python 3.10
- AgileX LIMO in Ackermann mode (`/limo_status.motion_mode == 1`)
- YDLidar on `/scan`; Orbbec Dabai DC1 color stream
- onboard x86_64 NUC12
- CasADi 3.7.2, CVXPY 1.7.5, OSQP 1.1.1

The collision model and Nav2 configuration use the installed LIMO footprint
of 0.32 m × 0.22 m in `base_link`, plus 0.05 m safety padding.

ROS `mpc_local_planner` is not installed and is not the controller used here.
DREAM includes its own CasADi/CVXPY/OSQP MPC. Two live motion-disabled
arbitrary-path samples solved `balanced` in about 70 ms and `pure_mpc` in about
89 ms for the six-step hardware horizon. The longer benchmark in
[`benchmark_results/`](benchmark_results/) records p99 below the 150 ms
acceptance threshold and every solve inside the 200 ms planning period; it
does not claim that every solve is below 100 ms.

For OACP-VB, a motion-free onboard benchmark measured the two sequential
branches at 185.0 ms median and 247.3 ms p95, which did not sustain a second
solve every 200 ms. The executed MPC remains at 5 Hz; the non-executed fallback
is therefore checked at 1 Hz. Between checks, the executed solve is constrained
to consume the remaining control prefix that the verified fallback shares. The
prefix cursor advances only after the final hardware gate returns the exact
planner command's structured ROS timestamp and odometry remains within explicit
position/speed/yaw tolerances. The adapter and supervisor preserve that identity
on internal `TwistStamped` topics; the physical `/cmd_vel` remains `Twist`. A
partially executed command is not reissued for a new full MPC step because that
would exceed the trajectory covered by the cached fallback solve. Instead its
certificate is revoked and a fresh `v_occ_min` solve executes until the next
scheduled two-branch check. Advancement requires at least 95% spatial progress
plus next-knot agreement. A withheld or repeated older command is not consumed;
because absence of an exact acknowledgement cannot prove non-execution, any
missing, stale, or mismatched token also revokes and clamps. An off-segment
mismatch, planner stop, changed path,
visible vehicle, or tightened bound revokes the certificate and clamps to
`v_occ_min` until a new check. This is a disclosed approximation, not the
paper's joint contingency optimizer. The OACP-VB source, scientific note,
calibration utility, and timing record are indexed together in
[`dream_limo/OACP/`](dream_limo/OACP/README.md).

The implementation is based on upstream DREAM commit
`0d298cd6de11c268224173a4d75770e934fd0861`; see
[`UPSTREAM_DREAM.md`](UPSTREAM_DREAM.md) and
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Build

Expected sibling layout:

```text
~/limo_lvv_ws/src/
├── dream_limo/
└── sfg_nav/
```

Keep one canonical `dream_limo` source tree. For a fresh checkout, link the
package from this repository into the ROS workspace instead of maintaining a
second copied version that can silently overwrite newer installed code:

```bash
mkdir -p "$HOME/limo_lvv_ws/src"
ln -s "$HOME/DREAM/src/dream_limo" "$HOME/limo_lvv_ws/src/dream_limo"
```

If `~/limo_lvv_ws/src/dream_limo` already exists, first verify whether it is
the current repository package; do not overwrite an existing directory with
the command above.

```bash
cd "$HOME/limo_lvv_ws"
source /opt/ros/humble/setup.bash
source "$HOME/agilex_ws/install/setup.bash"

rosdep install --from-paths src --ignore-src -r -y
python3 -m pip install --user -r src/dream_limo/requirements.txt
colcon build --symlink-install --packages-select dream_limo
source install/setup.bash
ros2 pkg prefix sfg_nav >/dev/null
```

If this is the first ROS installation on the machine and `rosdep` reports that
it is uninitialized, run `sudo rosdep init` once and then `rosdep update`.
If the final `ros2 pkg prefix` check fails, install/build the sibling `sfg_nav`
once using its own README, source that overlay, and rerun the check. DREAM does
not rebuild or launch the SFG planner.

Verify the numerical stack:

```bash
python3 - <<'PY'
import casadi, cvxpy, osqp
print("casadi", casadi.__version__)
print("cvxpy", cvxpy.__version__)
print("osqp", osqp.__version__)
PY
```

## Complete command order

Use `ROS_DOMAIN_ID=0` and the same overlay order in every terminal. Every
launch below is persistent; leave it running and open the next terminal.

### Terminal 1 — base, odometry, IMU, and LiDAR

```bash
export ROS_DOMAIN_ID=0
source /opt/ros/humble/setup.bash
source "$HOME/agilex_ws/install/setup.bash"
source "$HOME/limo_lvv_ws/install/setup.bash"

ros2 launch limo_bringup limo_start.launch.py \
  start_rf2o:=false \
  base_port_name:=ttylimo
```

This bringup already starts the LiDAR. Do not start a second YDLidar node.

### Terminal 2 — front camera evidence

```bash
export ROS_DOMAIN_ID=0
source /opt/ros/humble/setup.bash
source "$HOME/agilex_ws/install/setup.bash"
source "$HOME/limo_lvv_ws/install/setup.bash"

ros2 launch orbbec_camera dabai.launch.py \
  enable_point_cloud:=false \
  enable_colored_point_cloud:=false \
  enable_depth:=false \
  enable_ir:=false \
  enable_color:=true
```

Only RGB is required. It is evidence of the visual occlusion, not planner
input.

### Terminal 3 — required dry run

Start DREAM with physical output disabled (the default):

```bash
export ROS_DOMAIN_ID=0
source /opt/ros/humble/setup.bash
source "$HOME/agilex_ws/install/setup.bash"
source "$HOME/limo_lvv_ws/install/setup.bash"

ros2 launch dream_limo dream_free_navigation.launch.py \
  model:=balanced \
  target_speed:=0.15 \
  rviz:=true
```

Wait until the costmap, scan, risk field, and camera appear. In RViz:

1. Select **2D Goal Pose**.
2. Click an observed-free, reachable costmap area and drag the arrow toward
   the desired arrival heading.
3. Do not click gray unknown/occluded space or colored obstacle inflation.

No coordinate or distance is entered. A new valid click replaces the previous
goal; an invalid replacement cancels the old mission and keeps the robot
stopped. The dry-run graph computes a route and DREAM command candidate, but
the sole `/cmd_vel` owner continuously publishes zero.

Confirm that behavior:

```bash
ros2 topic echo /dream/deadman_status --once --full-length
ros2 topic echo /dream/route_status --once --full-length
ros2 topic echo /dream/planner_status --once --full-length
ros2 topic echo /dream/collision_status --once --full-length
ros2 topic echo /dream/hardware_gate_status --once --full-length
ros2 topic echo /cmd_vel --once
```

Expected after a goal-valid and route-reachable click: goal `ready=true`,
route `PATH_READY`, planner `ready=true`, MPC `optimal`, collision `CLEAR`, hardware reason
`HARDWARE_OUTPUT_DISABLED`, and a zero `/cmd_vel`.

Stop this dry-run launch with `Ctrl-C` and wait for all its children to exit
before starting physical mode.

### Terminal 3 — physical mode after commissioning

Physical output requires explicit per-run safety attestations. They are not
environment geometry parameters; they assert that the chassis watchdog,
staging, and independent operator stop were actually checked.

For this launch, `staging_pose_verified:=true` also attests that the fixed
0.30 m-radius disc around the robot's launch pose has been physically inspected
and is clear, and that people and movable objects will be kept out of it until
the robot exits that disc. This is needed because the installed LiDAR is
currently cropped to a forward 220-degree field of view: a padded rear corner
can become unobservable during the first small steering motion. The exception
is limited to that fixed start disc, requires the footprint at rest to be fully
known/free, requires the trajectory to recover into fully observed space, and
never permits an occupied cell. It cannot be reused after the robot leaves the
disc.

For the first motion use 0.10–0.15 m/s:

```bash
export ROS_DOMAIN_ID=0
source /opt/ros/humble/setup.bash
source "$HOME/agilex_ws/install/setup.bash"
source "$HOME/limo_lvv_ws/install/setup.bash"

ros2 launch dream_limo dream_free_navigation.launch.py \
  model:=balanced \
  target_speed:=0.15 \
  rviz:=true \
  staging_pose_verified:=true \
  platform_watchdog_verified:=true \
  operator_kill_verified:=true \
  enable_physical_motion:=true
```

Again, use **2D Goal Pose** and click any currently observed-free, reachable
point. The robot remains stopped until the goal, route, DRIFT warm-up, collision result,
motion mode, ownership checks, and readiness countdown all pass. No joystick
is required. `target_speed` is a cap/nominal cruise value, not a forced speed;
DREAM may slow or stop for risk and obstacles. The reviewed physical gate
accepts `0.03 < target_speed <= 0.20` m/s. Use at most 0.15 m/s for the first
straight-line commissioning run, then 0.20 m/s for the next step. Values above
0.20 m/s are rejected rather than silently clipped. The 0.20 m/s step retains
the 0.35 m/s² acceleration cap; 0.25–0.30 m/s has not passed the current
onboard solver/stopping-margin review.

Stop the mission immediately through ROS with:

```bash
ros2 service call /dream/stop_mission std_srvs/srv/Trigger "{}"
```

The human independent hardware/power stop remains mandatory; the ROS service
is not a substitute.

## Three-arm comparison and OACP-VB calibration

The publishable arms are `nominal`, `oacp_vb`, and `balanced`. They use one
launch graph, one `dream_free_planner`, one shared LMPC-CBF configuration, and
the same `/goal_pose` trigger. `pure_mpc` is retained only for regression
compatibility. The OACP-VB arm is the **velocity-bound adaptation of Zheng et
al. (2025)** documented with its reproducible source in
[`dream_limo/OACP/`](dream_limo/OACP/README.md); it is not the published
Bezier/ADMM planner.

First verify OACP-VB with physical output disabled:

```bash
ros2 launch dream_limo dream_free_navigation.launch.py \
  model:=oacp_vb \
  target_speed:=0.15 \
  rviz:=true
```

RViz adds the phantom connector, PVS extent, reachable extent, `r_total`, and
both velocity bounds. `/dream/oacp_vb_status` must show a valid pre-goal bound
before a click and an exact bound with the matching path stamp afterward.

The paper's `4.5/6.0` thresholds are references, not arena calibration. Run one
identified logging approach before an OACP-VB comparison:

Set `ARM=oacp_calibration` in the bag-recording command below and start that
recorder before publishing the calibration goal.

```bash
ros2 launch dream_limo dream_free_navigation.launch.py \
  model:=oacp_vb \
  target_speed:=0.15 \
  oacp_calibration_logging_only:=true \
  rviz:=true \
  staging_pose_verified:=true \
  platform_watchdog_verified:=true \
  operator_kill_verified:=true \
  enable_physical_motion:=true
```

This excluded run computes risk and suggested thresholds but deliberately does
not apply the velocity bound. Samples reset on the accepted goal and begin
after motion authorization. Record the run with the bag command below, then
extract the actual occluded interval directly from the bag:

```bash
BAG="$HOME/limo_lvv_ws/bags/oacp_calibration_YYYYMMDD_HHMMSS"

ros2 run dream_limo dream_oacp_calibration "$BAG" \
  --start-offset 0.0 \
  --end-offset 8.0 \
  --csv "$HOME/oacp_calibration_curve.csv" \
  --output "$HOME/oacp_calibration_thresholds.json"
```

The offsets are seconds from the accepted-goal receipt timestamp (`t=0`);
replace `0.0/8.0` with the visually reviewed interval during which the PVS is
occluded. Plot or inspect the exported CSV, then record the reported linear
70th-percentile exploration threshold and the `4/3` fallback value. The command
rejects bags containing multiple goal revisions unless `--goal-revision` is
selected. Do not set the calibration acknowledgement from the live suggestion
alone.

Run the calibrated OACP-VB arm with the reviewed values:

```bash
CEXP="<reviewed exploration threshold>"
CFALL="<reviewed fallback threshold>"

ros2 launch dream_limo dream_free_navigation.launch.py \
  model:=oacp_vb \
  target_speed:=0.15 \
  c_th_max_exploration:="$CEXP" \
  c_th_max_fallback:="$CFALL" \
  oacp_thresholds_calibrated:=true \
  rviz:=true \
  staging_pose_verified:=true \
  platform_watchdog_verified:=true \
  operator_kill_verified:=true \
  enable_physical_motion:=true
```

Run the no-occlusion-risk floor by changing only the arm:

```bash
ros2 launch dream_limo dream_free_navigation.launch.py \
  model:=nominal \
  target_speed:=0.15 \
  rviz:=true \
  staging_pose_verified:=true \
  platform_watchdog_verified:=true \
  operator_kill_verified:=true \
  enable_physical_motion:=true
```

Run the DREAM comparison arm through the same launch:

```bash
ros2 launch dream_limo dream_free_navigation.launch.py \
  model:=balanced \
  target_speed:=0.15 \
  rviz:=true \
  staging_pose_verified:=true \
  platform_watchdog_verified:=true \
  operator_kill_verified:=true \
  enable_physical_motion:=true
```

Capture the first RViz click without measuring or typing coordinates. Start
this before clicking once:

```bash
GOAL_FILE="$HOME/limo_lvv_ws/bags/comparison_goal.yaml"
mkdir -p "$(dirname "$GOAL_FILE")"
ros2 topic echo /goal_pose --once > "$GOAL_FILE"
```

For every run, first start the merger cue in a separate terminal **before**
clicking or replaying the goal:

```bash
MERGER_DELAY=2.0
ros2 run dream_limo dream_merger_cue --delay "$MERGER_DELAY"
```

Then, for every later arm, replay the exact saved pose after its risk provider
is ready:

```bash
GOAL_MESSAGE="$(
  python3 - "$GOAL_FILE" <<'PY'
import sys
import yaml

with open(sys.argv[1], encoding="utf-8") as stream:
    goal = next(
        document
        for document in yaml.safe_load_all(stream)
        if document is not None
    )
goal["header"]["stamp"] = "now"
print(yaml.safe_dump(goal, default_flow_style=True))
PY
)"

ros2 topic pub --once /goal_pose geometry_msgs/msg/PoseStamped \
  "$GOAL_MESSAGE"
```

ROS 2 expands `stamp: now` at publication, so the authorizer receives a fresh
source timestamp while every pose and frame value remains identical. The
package does not actuate the independent merger robot. For a publishable
fixed-timing run, connect its separately safety-reviewed runner to
`/dream/merger_release`. `dream_merger_cue` ignores retained empty goal
invalidations, cancels a pending cue if the accepted goal is invalidated, and
reschedules for any newer accepted goal. It also ignores transient-local goals
retained from before the cue process started. Its delay starts from publication
of the currently accepted `map` goal, which occurs just after
the authorizer's recorded acceptance-receipt `t=0`; use the recorded
`/dream/merger_release` timestamp as the authoritative release time.

Record `/dream/merger_release`. A human reacting to this cue is acceptable for
a smoke demonstration but is not an identical scripted merger trajectory.
Use the same saved goal, starting geometry, delay, and external merger runner
for `nominal`, `oacp_vb`, and `balanced`, and restart the launch between runs.
The planner status logs the shared controller hash; a mismatch invalidates the
comparison. Physical OACP-VB comparison runs reject
`enable_contingency:=false`.

## Occluder and sudden-merger experiment

- Place a tall, long static occluder so it intersects the LiDAR plane and
  visibly blocks the front camera's view of the merger.
- Keep the ego's initial drivable corridor physically clear. A static object
  directly in front of the ego is correctly treated as an obstacle and will
  prevent motion.
- Initially hide the second vehicle behind the occluder. While hidden, it must
  not appear in `/tracked_agents`. `balanced` represents the shadow through
  DRIFT `Q_occ`, `oacp_vb` through the right-lane PVS, and `nominal` through no
  occlusion-risk channel.
- Move the merger into view and toward the ego's intended path. LiDAR first
  returns provide immediate collision evidence, while temporal clustering
  confirms a moving vehicle for DREAM's dynamic CBF.
- Camera imagery proves visual occlusion only. It is not fused into control.

Useful evidence topics:

```bash
ros2 topic echo /sfg/lidar_cluster_state --once
ros2 topic echo /dream/vehicle_tracker_status --once --full-length
ros2 topic echo /tracked_agents --once --full-length
ros2 topic echo /dream/world_status --once --full-length
ros2 topic echo /dream/planner_status --once --full-length
```

The launch starts only SFG's neutral `lidar_cluster_buffer`; do not also launch
`sfg_perception.launch.py`, `sfg_full_stack.launch.py`, the SFG pedestrian
detector, or the SFG planner. Duplicate `/tracked_agents` or `/cmd_vel`
publishers invalidate the experiment.

## Safety gate before every enabled run

1. A human is assigned to an independent hardware/power stop.
2. The arena is clear enough for the selected goal and the robot starts at a
   standstill.
3. `/limo_status.motion_mode` is `1` (Ackermann).
4. The base serial watchdog has passed a wheels-raised loss-of-command test.
5. No competing navigation, teleop, or DREAM launch is running.
6. The dry-run route and footprint agree with the live scan in RViz.

Read-only checks:

```bash
ros2 topic echo /limo_status --field motion_mode --once
ros2 topic info /cmd_vel --verbose
ros2 topic hz /wheel/odom
ros2 topic hz /scan
ros2 run tf2_ros tf2_echo odom base_link
ros2 run tf2_ros tf2_echo base_link laser_link
```

Before DREAM starts, `/cmd_vel` should have zero publishers and the base
subscriber. During one DREAM launch it must have exactly one publisher:
`dream_hardware_command_gate`.

The reviewed base watchdog patch is
[`patches/limo_base_cmd_vel_watchdog.patch`](patches/limo_base_cmd_vel_watchdog.patch).
Do not apply it a second time to an already patched driver.

## RViz interpretation

- raw rays: current `/scan`;
- global costmap: observed free, occupied/inflated, and unknown regions;
- occlusion mask: LiDAR line-of-sight shadow used by DRIFT or OACP-VB;
- risk grid: DREAM PDE field;
- geometric path: planner-only Nav2 output;
- MPC path: DREAM's short controlled reference;
- vehicle markers: DREAM motion-confirmed tracks;
- driver view: camera evidence.

The clicked centre must be an observed zero-cost cell. That is only the first
gate: Nav2 must then find a known-space route using the padded LIMO footprint,
and DREAM rechecks the MPC's complete swept footprint against the latest
inflated costmap. Unknown cells fail closed at those route/control gates. A
goal may therefore be visible in the camera but remain unreachable when LiDAR
has not observed a complete drivable corridor around the occluder.

The LIMO is Ackermann-steered and this deployment uses a forward-only Dubins
route model. A goal behind the robot can require a large forward loop; it is
not equivalent to a holonomic pivot or reverse maneuver. For the first motion
check, click in the direction shown by the robot arrow in RViz. This still
requires no coordinate or mission-distance entry.

The core swept-trajectory validator is strict by default: every centre sample
must have zero cost. It also exposes an explicit startup-recovery option for a
robot already inside soft inflation. That option permits only an initial
recovery beginning at cost 1 through 98, and each successive positive centre
cost must hold or decrease. Zero-valued grid gaps may occur inside a discretized
inflation band and do not reset that bound. Once a later control horizon starts
at zero cost, it cannot enter positive cost. Cost 99 (Nav2's inscribed value),
unknown or occupied centre cells, and any unknown or occupied padded-footprint
sample remain hard failures in both modes.

## Record a comparison run

```bash
mkdir -p "$HOME/limo_lvv_ws/bags"
ARM=balanced  # nominal, oacp_vb, or balanced

ros2 bag record \
  -o "$HOME/limo_lvv_ws/bags/${ARM}_$(date +%Y%m%d_%H%M%S)" \
  /tf /tf_static /wheel/odom /imu /scan /limo_status \
  /camera/color/image_raw /dream/driver_view \
  /global_costmap/costmap /goal_pose /dream/navigation_goal \
  /dream/geometric_path /dream/reference_trajectory /dream/control \
  /sfg/lidar_clusters /tracked_agents /dream/world_model \
  /dream/occlusion_mask /dream/risk_field /dream/drift_status \
  /dream/oacp_vb_status /dream/oacp_vb_markers \
  /dream/route_status /dream/planner_status \
  /dream/collision_status /dream/safety_status \
  /dream/deadman_status /dream/hardware_gate_status \
  /dream/metrics /dream/merger_release /cmd_vel
```

Record reveal time, clearance, TTC/conflict-arrival margin, speed,
acceleration/jerk, veto state, risk/bounds, velocity-slack activations, and
per-branch MPC timing. Repeat at least five fixed-timing runs per arm and at
least three randomized-release runs per arm with matched starting geometry.

## Tests and offline checks

```bash
cd "$HOME/limo_lvv_ws"
source /opt/ros/humble/setup.bash
source "$HOME/agilex_ws/install/setup.bash"
source install/setup.bash

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 colcon test --packages-select dream_limo
colcon test-result --verbose

ros2 run dream_limo dream_mpc_benchmark \
  --iterations 50 \
  --output "$HOME/dream_mpc_benchmark.json"
```

Legacy fixed-lane replay and SIL remain available for regression testing:

```bash
ros2 run dream_limo dream_stage1_replay --output /tmp/dream_replay.json
ROS_DOMAIN_ID=42 ros2 launch dream_limo dream_motion_demo.launch.py model:=balanced
```

They are not the physical free-navigation workflow.

## Troubleshooting

- **A click does nothing:** inspect `/dream/deadman_status`. `GOAL_NOT_FREE`
  means the clicked cell itself is inflated or occupied; `GOAL_IN_UNKNOWN`
  means it is not LiDAR-observed. Select another visible free point. No
  coordinate edit is needed.
- **Route fails:** inspect `/dream/route_status`. The goal may be free but not
  connected through currently observed free space with the LIMO footprint and
  turn radius. This is a physical-clearance result, not a missing distance
  parameter.
- **Planner is ready but no motion:** inspect
  `/dream/hardware_gate_status.reason`; a non-ready safety condition always
  overrides target speed.
- **`PATH_START_TRAJECTORY_FOOTPRINT_UNKNOWN`:** the path-start footprint has
  not met the bounded launch-clearance contract. Confirm that the complete
  0.30 m start disc is physically clear, that `staging_pose_verified:=true`
  reached the planner, and that the proposed short prefix becomes fully known.
  Unknown footprint cells outside the fixed start disc remain rejected.
- **`PATH_START_TRAJECTORY_CENTER_NOT_FREE` or
  `TRAJECTORY_CENTER_NOT_FREE`:** the centre reached Nav2's inscribed/lethal
  range (cost 99–100), not ordinary soft inflation. Choose a route with more
  clearance; do not raise the hard-cell threshold.
- **The robot moves briefly and stops:** do not keep clicking replacement
  goals; every new goal intentionally disarms the old mission and restarts the
  readiness countdown. Inspect the planner reason. Known Nav2 soft-inflation
  costs 1–98 are accepted only when the complete swept padded footprint is
  known and contains no lethal cell. Unknown, inscribed cost 99, lethal cost
  100, and the independent LiDAR collision envelope remain hard stops.
- **`DECISION_RISK_VETO`:** the balanced DREAM arm is intentionally yielding
  to route risk. This is controller behavior, not a lost RViz goal. Record it
  as a veto activation; the matched scientific floor is `nominal`, which
  disables every occlusion-risk channel but retains all collision and footprint
  gates. `pure_mpc` is legacy regression compatibility only.
- **No LiDAR free space:** check `/scan`, TF, and
  `/global_costmap/costmap`. The scan observation source must cover the LiDAR's
  physical height, and ray clearing must be active.
- **No merger after reveal:** check `/sfg/lidar_cluster_state`, then
  `/dream/vehicle_tracker_status`. The merger must be a vehicle-sized cluster
  with observable displacement; static clusters are intentionally withheld.
- **Camera absent:** verify `/camera/color/image_raw` with Best Effort QoS. The
  planner can remain safe without it, but the visual-occlusion evidence is
  incomplete.
- **MPC fallback or excessive CBF slack:** stop the physical experiment and
  inspect `/dream/planner_status`; the hardware gate rejects it.
- **Repeated countdown:** observe `/dream/hardware_gate_status` continuously.
  Any stale or changing prerequisite restarts the three-second countdown.
  The collision monitor may retain one last-good exact-TF scan through one
  rejected callback, but only until that accepted scan is 0.20 s old. A second
  consecutive rejection fails closed, while the final hardware gate
  independently enforces its 0.40 s raw-scan watchdog.
- **Duplicate owners:** stop all DREAM/SFG launches, wait for child processes,
  then start exactly one primary launch.

## License

`dream_limo` is MIT-licensed, including upstream DREAM attribution; see
[`LICENSE`](LICENSE).
