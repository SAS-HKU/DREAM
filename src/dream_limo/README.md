# dream_limo

ROS 2 Humble deployment of the DREAM occlusion-aware planning framework on an
AgileX LIMO. This package is a sibling of `sfg_nav`: it does not import, copy,
or modify SFG source code. Integration is through SFG's public neutral
`/sfg/lidar_clusters` output, DREAM's `/tracked_agents` convention, and standard
ROS messages.

The implementation is pinned to DREAM commit
`0d298cd6de11c268224173a4d75770e934fd0861`. See
[`UPSTREAM_DREAM.md`](UPSTREAM_DREAM.md) and
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Safety and current status

> **Physical output is implemented, but this audited robot is not commissioned
> for motion yet. Keep it stopped until every physical gate below is verified.**

- SIL and ordinary live-sensor launches still end at `/cmd_vel_test`.
- Only `dream_hardware_motion.launch.py` starts the reviewed `/cmd_vel` gate.
  Its checked-in defaults publish a continuous zero command and cannot arm.
- Never remap `/cmd_vel_test` to `/cmd_vel`.
- The July 2026 audit found `motion_mode=0` (differential) and `/dev/input/js0`
  owned by NoMachine rather than a physical joystick. Both are motion blockers.
- Keep a human at an independent hardware/power stop, and stop every competing
  navigation, teleop, or command node before hardware commissioning.
- Passing DREAM preflight is necessary but never sufficient to move.

The safety path fails closed on stale inputs, invalid/fallback MPC output,
non-finite values, CBF slack above `0.05`, loss of the 0.75 s held-to-run
heartbeat, a front LiDAR stop, or the wrong drive mode. Raw Ackermann
`Twist.angular.z` is capped independently at `0.198`. The physical boundary
adds exact publisher ownership, a retained/inflated LiDAR trajectory gate,
fresh world/DRIFT/MPC checks, a second software watchdog, a 0.15 m/s
commissioning cap, and independent speed/steering slew limits. Any failed
condition outputs zero. A reviewed 0.25 s steady-clock watchdog is now applied
at the installed base driver's serial boundary, but its wheels-raised behavior
must still be verified on this chassis before motion is enabled.

## Implemented experiment

The checked-in blocker-free SIL scenario contains:

- ego in the left lane;
- one long truck-class occluder in the middle lane;
- no static blocker in front of the ego;
- one hidden agent merging from the right lane into the middle lane;
- an identical route objective for both arms: stay left past the truck, merge
  over `x=[2.8, 3.8]`, then enter the middle-lane conflict zone
  `x=[3.3, 5.3]`, brake toward `x=5.55`, and latch a zero command at mission
  completion.

`pure_mpc` is the baseline. It uses the same route, visible tracks, kinematic
model, nominal geometric CBF constraints, tracker, and safety supervisor as
DREAM. Only the DREAM veto, risk cost, and risk-expanded CBF/headway terms are
disabled. `balanced` enables those three DREAM channels.

The local decision layer is a reduced six-labelled-gap, straight-lane IDEAM
adaptation; it is not a claim of reproducing every upstream highway DFS mode.

## Architecture

1. `dream_state_estimator`: wheel odometry plus optional IMU yaw rate into a
   local experiment frame anchored automatically at the first odometry pose.
2. `dream_world_model`: first-return LiDAR visibility/shadow, DREAM vehicle
   tracks derived from SFG's neutral clusters, or one optional scan-gated
   second-robot odometry track. Live mode does not load a YAML occluder into
   DRIFT or MPC.
3. `dream_merger_odometry_adapter` (optional): converts a namespaced second
   LIMO's unrelated local odometry into the ego odom frame using an explicitly
   verified SE(2) correspondence; bad frames, stale time, or bad data suppress
   the track immediately.
4. `dream_drift_field`: CFL-checked DRIFT PDE, five-second warm-up, raw field,
   RViz risk grid, and occlusion mask.
5. `dream_planner`: reduced IDEAM decision, DREAM veto, local MPC-CBF, matched
   cruise-speed profile, and a one-way mission-complete stop latch.
   The hardware launch additionally requires footprint-safe grid/road bounds;
   their presence is reported in every accepted planner status.
6. `dream_command_adapter`: drive-mode gate and LIMO Ackermann conversion.
7. `dream_safety_supervisor`: independent watchdog and non-physical candidate
   output on `/cmd_vel_test`; only the hardware gate can publish physically.
8. `dream_camera_evidence`: untouched and annotated front-camera evidence.
   Camera freshness is an experiment-evidence preflight gate, but pixels and
   annotations never enter DRIFT, decision-making, or MPC.
9. `dream_collision_monitor`: transforms
   each scan at its sensor timestamp, retains/inflates first-return surfaces,
   treats LiDAR shadow and off-road space as non-traversable, and checks the
   densely interpolated reference trajectory. It publishes no command topic.
10. `dream_hardware_deadman`: accepts only one reviewed `joy_node` owner and
   publishes a 20 Hz arm heartbeat only while two configured buttons are held;
   a third button requests a latched stop.
11. `dream_hardware_command_gate`: the sole physical `/cmd_vel` publisher. It
    continuously rechecks all safety/status owners and heartbeats and otherwise
    publishes zero at 20 Hz.

The MPC is included in `dream_limo`; it uses CasADi for bicycle-model
dynamics/Jacobians and CVXPY+OSQP for the QP. ROS `mpc_local_planner` is not
installed or required. It is retained only as an architectural reference.

## Tested platform

- Ubuntu 22.04 and ROS 2 Humble
- Python 3.10
- AgileX LIMO with `/dev/ttylimo`
- YDLidar on `/dev/ttyUSB0`, frame `laser_link`
- Orbbec Dabai DC1 through `astra_camera`
- onboard x86_64 computer

Numerical versions tested on this robot:

- NumPy 1.26.4
- SciPy 1.13.0
- CVXPY 1.7.5
- OSQP 1.1.1
- CasADi 3.7.2

The pins in [`requirements.txt`](requirements.txt) are the audited x86_64
environment. Check wheel availability before installing on ARM.

## Workspace layout

The expected overlays are:

```text
~/agilex_ws/install/          # LIMO, YDLidar and astra_camera drivers
~/limo_lvv_ws/src/
├── dream_limo/               # this package
└── sfg_nav/                  # optional public perception provider
```

Source overlays in this order in every terminal:

```bash
source /opt/ros/humble/setup.bash
source "$HOME/agilex_ws/install/setup.bash"
source "$HOME/limo_lvv_ws/install/setup.bash"
```

All terminals participating in a real sensor run must use the same
`ROS_DOMAIN_ID`. The examples use domain `0`; do not accidentally retain the
isolated SIL domains `42` or `43`.

## Install and build

Place or clone this package at `~/limo_lvv_ws/src/dream_limo`, then run:

```bash
cd "$HOME/limo_lvv_ws"
source /opt/ros/humble/setup.bash
source "$HOME/agilex_ws/install/setup.bash"

rosdep install --from-paths src/dream_limo --ignore-src -r -y
python3 -m pip install --user -r src/dream_limo/requirements.txt

colcon build --symlink-install --packages-select dream_limo
source install/setup.bash
```

If `sfg_nav` is not in the workspace, omit the SFG wrapper described below;
DREAM-only SIL and sensor smoke do not import SFG Python code.

Verify the numerical stack:

```bash
python3 - <<'PY'
import casadi, cvxpy, osqp
print("casadi", casadi.__version__)
print("cvxpy", cvxpy.__version__)
print("osqp", osqp.__version__)
PY
```

Run tests:

```bash
cd "$HOME/limo_lvv_ws"
source /opt/ros/humble/setup.bash
source "$HOME/agilex_ws/install/setup.bash"
source install/setup.bash

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 colcon test --packages-select dream_limo
colcon test-result --verbose
```

The plugin-autoload setting avoids incompatible user-installed pytest plugins
on some LIMO images.

## Short demo commands

Select the planner directly on the launch command. Use `balanced` for DREAM or
`pure_mpc` for the matched MPC-only baseline.

Closed-loop moving ego and sudden merger in the isolated RViz SIL scene:

```bash
ROS_DOMAIN_ID=42 ros2 launch dream_limo dream_motion_demo.launch.py model:=balanced
```

After the real LIMO/LiDAR and camera drivers are already running, start the
stationary live camera + LiDAR-shadow + vehicle-reveal planner view with:

```bash
ROS_DOMAIN_ID=0 ros2 launch dream_limo dream_live_demo.launch.py model:=balanced
```

Change only `model:=pure_mpc` to run the baseline. The live launch starts
SFG's neutral LiDAR cluster buffer and DREAM's car-labelled motion tracker; it
does not start SFG's pedestrian detector, SFG tracker, or SFG planner. The SIL
command moves a simulated ego in closed loop. The live command deliberately
ends at `/cmd_vel_test` and does not move the physical LIMO.

To inspect the complete hardware graph without allowing motion, use the same
model selector. This owns `/cmd_vel` but publishes only zeros:

```bash
ROS_DOMAIN_ID=0 ros2 launch dream_limo dream_hardware_motion.launch.py \
  model:=balanced
```

Do not add the enabling arguments until the commissioning checklist in
“Physical motion commissioning” has passed on that specific robot.

## Perceived scene versus mission intent

The stationary real-sensor launch does **not** require the occluder's center,
size, heading, or polygon. It uses `occlusion_source=lidar_first_return`: every
trustworthy first return is a measured visibility boundary, and road cells
behind it become the DRIFT occlusion mask. A configured truck is neither
inserted into the planner nor used to approve those rays. The optional merger
odometry track is gated by the same measured scan. SFG `/tracked_agents` are
also withheld when their position lies behind a closer return.

The local `map` frame is initialized from the first `/wheel/odom` sample so the
robot begins at the mission's left-lane pose. Start the base only after the
robot is stationary and aligned along the intended lane direction.

[`config/merge_mission.yaml`](config/merge_mission.yaml) contains only semantic
mission intent: three lane centerlines, the requested middle lane, and the
nominal merge/conflict interval plus the terminal stop station. These are
analogous to a route supplied to an autonomous-driving planner; the installed
SFG perception stack does not detect lane markings or infer which lane the
user wants. Adjust this file only if the physical lane spacing or requested
maneuver differs from the checked-in 0.45 m blocker-free experiment.
[`config/arena.yaml`](config/arena.yaml) is the deterministic SIL scene and is
not used as live obstacle truth.

The occluder must still physically intersect the measured LiDAR plane and the
camera line of sight. That is sensor installation/experiment validation, not a
manually entered scene polygon.

One 2-D scan cannot always recover the far face or complete footprint of a long
occluder. The hardware collision monitor therefore treats the measured shadow
as unknown/non-traversable, retains first-return surfaces briefly, inflates
them by the robot half-diagonal plus margin, and rejects any reference path
touching unknown, occupied, off-road, or outside-grid space. This is a final
trajectory gate; it does not fabricate unseen geometry or leak the hidden
merger into the vehicle list.

## Run 1: headless replay

This is the first gate and does not start ROS drivers:

```bash
cd "$HOME/limo_lvv_ws"
source /opt/ros/humble/setup.bash
source "$HOME/agilex_ws/install/setup.bash"
source install/setup.bash

ros2 run dream_limo dream_stage1_replay \
  --output "$HOME/limo_lvv_ws/dream_stage1_replay.json"
```

Expected: field ready, no hidden-track leak or fallback, pure MPC enters the
middle-lane conflict zone, and balanced DREAM vetoes/delays that merge.

## Run 2: blocker-free SIL A/B in RViz

SIL publishes fake sensor-shaped topics, so use isolated non-default domains.
Run the arms sequentially, not simultaneously.

Pure-MPC baseline:

```bash
cd "$HOME/limo_lvv_ws"
source /opt/ros/humble/setup.bash
source "$HOME/agilex_ws/install/setup.bash"
source install/setup.bash

ROS_DOMAIN_ID=42 ros2 launch dream_limo dream_rviz_smoke.launch.py \
  preset:=pure_mpc \
  report_path:="$HOME/limo_lvv_ws/pure_mpc_smoke.json"
```

Wait for `RViz smoke PASS`, then press `Ctrl-C` and wait for every process to
finish. Only then run DREAM:

```bash
ROS_DOMAIN_ID=43 ros2 launch dream_limo dream_rviz_smoke.launch.py \
  preset:=balanced \
  report_path:="$HOME/limo_lvv_ws/dream_balanced_smoke.json"
```

Again wait for `RViz smoke PASS`, then press `Ctrl-C`.

The current horizon is 6 steps at 0.2 s. The A/B behavior is validated, but
observed ROS worst cases can still exceed the 100 ms profiling target. Timing
therefore remains a physical-motion gate.

### Motion-free onboard MPC acceptance check

The benchmark below never imports `rclpy`, creates a ROS publisher, or reaches
`/cmd_vel`. It runs the hardware MPC configuration (`enforce_map_bounds=true`)
for standstill, occluded-truck, and revealed-merger cases in both experiment
arms:

```bash
source /opt/ros/humble/setup.bash
source "$HOME/limo_lvv_ws/install/setup.bash"
ros2 run dream_limo dream_mpc_benchmark \
  --iterations 50 \
  --output "$HOME/dream_mpc_benchmark.json"
```

The command exits nonzero for a fallback, non-finite solution, excessive CBF
slack, p99 above 150 ms, or any solve that misses the 200 ms planning period.
On the audited NUC12 (i7-1260P), 300 hardware-style solves at the 0.15 m/s
commissioning target produced zero fallbacks, finite `optimal` results,
worst-case p99 124.7 ms, absolute maximum 134.1 ms, and maximum CBF slack
`3.33e-8` versus the 0.05 rejection limit. The complete captured report is
[`benchmark_results/nuc12_hardware_015_2026-07-20.json`](benchmark_results/nuc12_hardware_015_2026-07-20.json).
CasADi 3.7.2, CVXPY 1.7.5, and OSQP 1.1.1 were used. This establishes onboard
solver capability; it does not replace a timing run with all sensor nodes and
recording active.

### Select one experiment arm

The same model selector is accepted by SIL and the live stationary launch:

```bash
export MODEL=balanced       # DREAM: veto + risk cost + risk-expanded CBF
# export MODEL=pure_mpc     # baseline: those three DREAM channels disabled

case "$MODEL" in
  balanced|pure_mpc) ;;
  *) echo "MODEL must be balanced or pure_mpc"; exit 2 ;;
esac
```

To reproduce the previous moving RViz result, run the isolated SIL launch
first using `preset:="$MODEL"`. Stop it completely before returning to domain
0 for the live-sensor procedure below. SIL is the only current closed-loop ego
motion demonstration. The live procedure shows the real camera/scan, perceived
shadow, merger track, decision, target control and MPC trajectory, but the
physical ego remains stationary.

## Run 3: stationary real-sensor occluder smoke

This section opens the base serial connection but still publishes no real
motion command. Keep the robot mechanically prevented from moving. Use the same
real-run domain in every terminal:

```bash
export ROS_DOMAIN_ID=0
source /opt/ros/humble/setup.bash
source "$HOME/agilex_ws/install/setup.bash"
source "$HOME/limo_lvv_ws/install/setup.bash"
```

### Terminal 1 — LIMO base and LiDAR

First verify the installed devices:

```bash
ls -l /dev/ttylimo /dev/ttyUSB0
```

Then start the verified Humble bringup:

```bash
ros2 launch limo_bringup limo_start.launch.py \
  start_rf2o:=false \
  base_port_name:=ttylimo
```

Do not separately start YDLidar. Do not use
`limo_base/start_limo.launch.py`; that installed file is incomplete.

### Terminal 2 — front camera

```bash
export ROS_DOMAIN_ID=0
source /opt/ros/humble/setup.bash
source "$HOME/agilex_ws/install/setup.bash"
source "$HOME/limo_lvv_ws/install/setup.bash"

ros2 launch astra_camera dabai.launch.py
```

### Terminal 3 — verify live inputs before DREAM

```bash
export ROS_DOMAIN_ID=0
source /opt/ros/humble/setup.bash
source "$HOME/agilex_ws/install/setup.bash"
source "$HOME/limo_lvv_ws/install/setup.bash"

ros2 topic hz /wheel/odom
ros2 topic hz /scan
ros2 topic echo /limo_status --field motion_mode --once
ros2 topic echo /scan --field header.frame_id --once
ros2 topic echo /camera/color/image_raw \
  --qos-reliability best_effort --once
ros2 run tf2_ros tf2_echo odom base_link
ros2 run tf2_ros tf2_echo base_link laser_link
ros2 topic info /cmd_vel --verbose
```

Stop each continuous diagnostic with `Ctrl-C`. `/cmd_vel` must report zero
publishers. `motion_mode=1` is required before any later Ackermann motion;
stationary dry-run inspection may continue in mode 0, but the adapter will
remain fail-closed.

### Terminal 4 — choose exactly one merger provider

Option A, advanced controlled two-robot input: namespace the second robot so it
publishes `/merger/raw/wheel/odom` with the exact frames `merger/odom` and
`merger/base_link`. Synchronize the two computers' clocks, measure one pose
correspondence between the merger's local odom frame and the ego odom frame,
then export the six measured values. The shell guards deliberately refuse to
launch if any value is missing:

```bash
: "${MERGER_SOURCE_X:?set measured merger-odom x}"
: "${MERGER_SOURCE_Y:?set measured merger-odom y}"
: "${MERGER_SOURCE_YAW:?set measured merger-odom yaw in radians}"
: "${MERGER_TARGET_X:?set corresponding ego-odom x}"
: "${MERGER_TARGET_Y:?set corresponding ego-odom y}"
: "${MERGER_TARGET_YAW:?set corresponding ego-odom yaw in radians}"

ros2 launch dream_limo dream_live_merger_odom.launch.py \
  model:="$MODEL" \
  merger_alignment_verified:=true \
  merger_source_reference_x:="$MERGER_SOURCE_X" \
  merger_source_reference_y:="$MERGER_SOURCE_Y" \
  merger_source_reference_yaw:="$MERGER_SOURCE_YAW" \
  merger_target_reference_x:="$MERGER_TARGET_X" \
  merger_target_reference_y:="$MERGER_TARGET_Y" \
  merger_target_reference_yaw:="$MERGER_TARGET_YAW"
```

The launch starts DREAM's fail-closed common-frame adapter and disables the SFG
vehicle tracker to prevent a duplicate agent. It will not accept generic
unnamespaced `odom` frames or an unverified alignment. The aligned
`/merger/wheel/odom` enters the planner only when measured LiDAR first-return
rays show an unobstructed line of sight; hidden ground truth never enters
`Q_veh`, decision groups, or MPC constraints. Require
`/dream/merger_odometry_adapter_status.ready=true` and world status fields
`merger_adapter_ready=true`, `merger_odom_fresh=true` before continuing.

Option B, recommended self-contained stationary smoke: neutral SFG LiDAR
clustering followed by DREAM's vehicle tracker, with no merger odometry:

```bash
ros2 launch dream_limo dream_live_demo.launch.py model:="$MODEL"
```

This wrapper launches only SFG's class-neutral `lidar_cluster_buffer`, DREAM's
vehicle tracker, and the DREAM sensor/planner/RViz stack. It does not launch
SFG's pedestrian detector, generic tracker, or planner. Never launch
`sfg_perception.launch.py` or `sfg_full_stack.launch.py` alongside it because
that would create a second `/tracked_agents` owner. In perception-only mode
there is no ground-truth merger identity, so the driver view reports `TRACK
OBSERVED` rather than claiming a ground-truth reveal.

Keep the ego chocked. Start the merger fully hidden, wait for DRIFT/preflight,
then move it from behind the occluder into the middle-lane conflict region. The
DREAM vehicle tracker needs observable motion (roughly 0.08 m displacement and
0.10 m/s) before `/tracked_agents` reports it. Monitor that topic directly;
an empty pre-reveal list followed by a moving track is the perception evidence.

The live RViz configuration displays the annotated driver camera, raw LiDAR,
LiDAR cluster and DREAM vehicle-track markers, first-return occlusion mask, DRIFT risk
field, decision state and MPC reference trajectory. An optional raw-camera
display is present but disabled by default. In SFG mode, reveal evidence is
`/tracked_agents` plus the `TRACK OBSERVED (N)` label; `/dream/merger_visible`
is reserved for the separately aligned odometry-gate option.

Do not combine Options A and B for the same agent; that can duplicate the
visible object in the planner world.

### Terminal 5 — acceptance checks

```bash
ros2 topic info /cmd_vel --verbose
ros2 topic info /cmd_vel_test --verbose
ros2 topic echo /dream/preflight_status --once
ros2 topic echo /dream/camera_evidence_status --once
ros2 topic echo /dream/world_status --once
ros2 topic echo /dream/drift_ready --once
ros2 topic echo /dream/planner_status --once
ros2 topic echo /dream/safety_status --once
```

Require all of the following:

- `/cmd_vel`: zero publishers;
- `/cmd_vel_test`: exactly one publisher, `dream_safety_supervisor`;
- preflight: `passed=true`;
- camera: `ready=true`;
- world: `ready=true`, `alignment_received=true`,
  `occlusion_source=lidar_first_return`,
  `surveyed_static_geometry_used=false`, fresh ego/scan, and
  `shadow_cells > 0` with `shadow_route_samples > 0`;
- DRIFT ready after approximately five model seconds;
- safety output remains zero and unarmed;
- RViz shows the raw LiDAR shadow behind the perceived occluder surface;
- the raw front image visibly confirms the line-of-sight obstruction.

If `shadow_cells` or `shadow_route_samples` is zero, stop. Recheck the raw
scan, automatic frame alignment, laser pose, obstacle height, road-mask/route
overlap, and LiDAR returns. Reposition the stationary experiment if the object
does not actually block the requested merge corridor; do not insert a guessed
obstacle polygon to force the test to pass.

For a stationary planner-only A/B, record the balanced run, stop Terminal 4,
then relaunch the same provider and geometry with `MODEL=pure_mpc`. Never run
both planner arms at once. This compares risk, vetoes, proposed trajectories,
and timing; it does not validate physical ego response.

## Record evidence

Start this before the stationary scenario or later physical A/B run:

```bash
mkdir -p "$HOME/limo_lvv_ws/bags"
ARM="$MODEL"

ros2 bag record \
  -o "$HOME/limo_lvv_ws/bags/${ARM}_$(date +%Y%m%d_%H%M%S)" \
  /tf /tf_static /wheel/odom /imu /scan /limo_status \
  /camera/color/image_raw \
  /dream/camera_evidence_raw /dream/driver_view \
  /dream/camera_evidence_status \
  /sfg/lidar_clusters /sfg/lidar_cluster_markers \
  /dream/vehicle_tracker_status /dream/vehicle_track_markers \
  /tracked_agents /merger/raw/wheel/odom /merger/wheel/odom \
  /dream/merger_odometry_adapter_status \
  /dream/ego_state /dream/world_model /dream/world_status \
  /dream/occlusion_mask /dream/merger_visible \
  /dream/risk_field /dream/risk_field_raw \
  /dream/drift_status /dream/drift_ready \
  /dream/planner_status /dream/reference_trajectory /dream/control \
  /dream/cmd_vel_candidate /cmd_vel_test \
  /dream/adapter_status /dream/safety_status \
  /dream/collision_grid /dream/collision_status \
  /dream/deadman_status /dream/hardware_gate_status \
  /dream/preflight_status /dream/metrics /joy /cmd_vel
```

After stopping the bag, run `ros2 bag info <bag-directory>` and verify nonzero
counts for the raw and annotated camera topics. Camera topics use Best Effort
QoS.

## Shutdown order

1. Stop DREAM/SFG with `Ctrl-C` and wait for all child processes.
2. Stop rosbag recording.
3. Stop the camera driver.
4. Stop `limo_bringup` last.
5. Confirm no command publishers or leftover nodes remain.

## Physical motion commissioning

The code path exists, but do not enable it from the currently audited state.
First satisfy and record every item below:

1. Lift/chock the robot, clear the arena, assign one human to the independent
   hardware/power stop, and confirm `/cmd_vel` has no publisher before launch.
2. Mechanically configure both steering latches for Ackermann and verify
   `/limo_status.motion_mode == 1`; this cannot be changed by a ROS parameter.
3. Connect and positively identify a physical joystick. `ros2 run joy
   joy_enumerate_devices` and `/dev/input/by-id` must identify it; the NoMachine
   virtual `js0` is not acceptable. Verify the chosen hold/confirm/stop button
   indices while lifted.
4. Verify the reviewed serial-boundary watchdog, rebuild `limo_base`, then
   characterize it with wheels lifted and the human stop present. The patch is
   [`patches/limo_base_cmd_vel_watchdog.patch`](patches/limo_base_cmd_vel_watchdog.patch):

   On a fresh, unpatched `limo_ros2` checkout, apply it once:

   ```bash
   cd "$HOME/agilex_ws/src/limo_ros2"
   git apply --check "$HOME/limo_lvv_ws/src/dream_limo/patches/limo_base_cmd_vel_watchdog.patch"
   git apply "$HOME/limo_lvv_ws/src/dream_limo/patches/limo_base_cmd_vel_watchdog.patch"
   cd "$HOME/agilex_ws"
   colcon build --packages-select limo_base
   ```

   On this audited LIMO the patch is already applied and `limo_base` has been
   rebuilt, so do not apply it a second time. It uses a steady-clock 0.25 s
   timeout at the serial driver, repeatedly sends
   zero after stale input, rejects non-finite `Twist` immediately, and sends
   zero during orderly destruction. A `SIGKILL`, driver crash, USB failure, or power loss can
   prevent the process from transmitting its final zero; only a verified base
   firmware timeout or independent hardware stop covers that residual. Keep
   `platform_watchdog_verified=false` until the wheels-lifted loss tests confirm
   the required stop behavior on this chassis.
5. Survey the 0.45 m lane spacing, place the robot at the checked-in left-lane
   start pose and heading, and verify the RViz `map`, road, scan, collision grid,
   shadow, and reference trajectory agree. This is what
   `staging_pose_verified` asserts; obstacle geometry is still perceived.
6. Run the zero-output commissioning launch above. Require exact owners plus
   fresh `preflight`, `world`, `drift`, `planner`, `safety`, `collision`, and
   `deadman` status. Collision must be ready and trajectory-clear. Preflight
   latches the initial measured route-shadow evidence so the later physical
   reveal is permitted, but it never latches stale world/scan readiness.

Useful read-only checks while that launch is running:

```bash
ros2 topic info /cmd_vel --verbose
ros2 topic echo /dream/hardware_gate_status --full-length --once
ros2 topic echo /dream/collision_status --full-length --once
ros2 topic echo /dream/preflight_status --full-length --once
ros2 topic echo /limo_status --field motion_mode --once
```

Only after those physical checks have actually passed, set the positively
identified joystick device number and run the reviewed first-motion command:

```bash
: "${PHYSICAL_JOY_ID:?set the verified physical joystick device id}"

ROS_DOMAIN_ID=0 ros2 launch dream_limo dream_hardware_motion.launch.py \
  model:=balanced \
  target_speed:=0.15 \
  start_joy:=true joy_device_id:="$PHYSICAL_JOY_ID" \
  deadman_device_verified:=true \
  staging_pose_verified:=true \
  platform_watchdog_verified:=true \
  operator_kill_verified:=true \
  enable_physical_motion:=true
```

The robot remains stopped until both deadman buttons are held continuously and
the three-second countdown completes. Releasing either button or losing any
input/status heartbeat commands zero immediately. The initial gate caps speed
at the same 0.15 m/s used by the MPC. Near the checked-in `x=5.55` endpoint,
the common mission profile brakes both experiment arms toward zero and latches
`MISSION_COMPLETE`. A stop-button, collision-bubble, or supervisor latch
requires shutdown, root-cause review, and a clean relaunch; do not reset it
in-place to continue a run.

For the controlled second-LIMO provider, append `use_merger_odom:=true`,
`merger_alignment_verified:=true`, and the same six verified merger-frame
arguments used by `dream_live_merger_odom.launch.py`. Without all of them, keep
the default `use_merger_odom:=false` and use LiDAR tracking.

For the physical A/B, run balanced DREAM and pure MPC sequentially with
identical geometry, initial pose, merger timing, perception provider, safety
supervisor, and rosbag topic set. Change only `model:=pure_mpc` for baseline.
Required metrics include reveal time, route-aware projected conflict-arrival
margin, minimum clearance, conflict-zone overlap, ego speed/acceleration/jerk,
veto activations, risk at ego, solver timing, planner rejection, and supervisor
trigger counts.

The current machine still needs the physical checks above, live static-world
tracking validation, and timing acceptance with every sensor and recorder
active. Camera extrinsics are intentionally out of scope because the camera is
evidence-only and never enters planning.

## Troubleshooting

- **Wrong or missing topics:** source Humble, `agilex_ws`, then `limo_lvv_ws`
  in that order and check `ROS_DOMAIN_ID` in every terminal.
- **Preflight missing sensors:** start `limo_bringup` before DREAM.
- **Camera stale/no image:** start `astra_camera dabai.launch.py`; inspect the
  source image with Best Effort QoS.
- **Zero shadow cells:** inspect `/scan`, local-frame alignment, the road-mask
  overlap, and whether the object actually intersects the laser plane. Live
  mode does not consult the SIL truck polygon.
- **No vehicle track after reveal:** inspect `/sfg/lidar_clusters`,
  `/dream/vehicle_tracker_status`, and `/tracked_agents`; verify that the
  merger fits the vehicle-cluster width gate and moves enough to pass the
  temporal motion gate.
- **Merger adapter not ready:** verify clock synchronization, exact namespaced
  frame IDs, strictly increasing fresh timestamps, and the measured pose
  correspondence; never bypass `merger_alignment_verified`.
- **Duplicate agent:** use either the SFG tracker or aligned merger odometry.
  The supplied live/hardware launch files disable SFG tracking automatically
  when `use_merger_odom:=true`.
- **TF conflict:** ensure only one owner publishes each of `map -> odom`,
  `odom -> base_link`, and `base_link -> laser_link`.
- **Mode mismatch:** inspect `/limo_status`; do not bypass the adapter.
- **MPC fallback/slack rejection:** inspect `/dream/planner_status` and stop the
  experiment.
- **MPC over 150 ms p99 or any solve over 200 ms:** reduce load or keep the
  experiment stationary; do not infer full-stack hardware readiness from the
  motion-free benchmark alone.
- **Pytest import errors:** rerun with
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`.

## License

`dream_limo` is distributed under the MIT license in [`LICENSE`](LICENSE),
including the upstream DREAM copyright attribution. SFG source is not copied.
No GPLv3 source from `mpc_local_planner` is incorporated.
