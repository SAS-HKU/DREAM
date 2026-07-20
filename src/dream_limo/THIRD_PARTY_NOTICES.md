# Third-party notices

The DRIFT, IDEAM integration and MPC-CBF design are adapted from DREAM:

> MIT License, Copyright (c) 2026 PeterWANGHK

The complete upstream license text is reproduced in this package's `LICENSE`.
The pinned source is recorded in `UPSTREAM_DREAM.md`.

`sfg_nav` is not copied or imported. The supplied wrapper consumes its public,
class-neutral `/sfg/lidar_clusters` topic through standard ROS messages, then
DREAM publishes the neutral `/tracked_agents` interface used by its planner.

`mpc_local_planner` was consulted as a ROS 1 architectural reference only. It
is GPLv3 code and is not incorporated into this package.

`patches/limo_base_cmd_vel_watchdog.patch` targets the AgileX `limo_ros2`
`limo_base` driver, whose source headers identify it as BSD-3-Clause,
Copyright (c) 2021 Agilex Robotics. The patch does not import that driver into
`dream_limo`; when applied, the resulting driver remains subject to its
upstream BSD license and notices.
