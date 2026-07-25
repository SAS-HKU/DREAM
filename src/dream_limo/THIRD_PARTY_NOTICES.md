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

The OACP-VB experimental arm is an independent velocity-bound adaptation of:

> Lei Zheng, Rui Yang, Minzhe Zheng, Zengqi Peng, Michael Yu Wang, and Jun Ma,
> “Occlusion-Aware Contingency Safety-Critical Planning for Autonomous
> Driving,” *IEEE Transactions on Cybernetics*, 2026,
> [DOI 10.1109/TCYB.2025.3632366](https://doi.org/10.1109/TCYB.2025.3632366);
> [arXiv:2502.06359v2](https://arxiv.org/abs/2502.06359);
> [project page](https://zack4417.github.io/oacp-website/).

The [author-origin review snapshot at commit
`06760501d24af6093994f4d6d6e95cf9e26f45e1`](https://github.com/mengxingshifen1218/OACP/commit/06760501d24af6093994f4d6d6e95cf9e26f45e1)
was consulted as interpretive provenance only. No source from it is copied,
vendored, imported, or linked as a build/runtime dependency. That snapshot
contains no `LICENSE`/`COPYING` file, is not linked from the paper or project
page, and predates arXiv v2; this project therefore treats it as an unlicensed
review artifact rather than an official code release. See `OACP_VB.md` for the
adaptation and claims boundary.
