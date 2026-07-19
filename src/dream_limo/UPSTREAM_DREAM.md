# Upstream DREAM provenance

- Repository: https://github.com/SAS-HKU/DREAM.git
- Pinned commit: `0d298cd6de11c268224173a4d75770e934fd0861`
- License: MIT

`dream_limo` contains one importable, headless LIMO adaptation. It does not keep
a second ambiguous copy of upstream modules. The relevant upstream sources were:

- `src/config.py`, `src/pde_solver.py`
- `src/Integration/drift_interface.py`, `prideam_controller.py`,
  `integration_config.py`, and `episode_control.py`
- `src/Control/MPC.py`, `HOCBF.py`, `constraint_params.py`, and
  `contraint_params.py`
- `src/DecisionMaking/*`, `src/Model/*`, `src/Prediction/*`, and `src/Path/path.py`
- `src/uncertainty_merger_DREAM.py` as scenario/control-flow reference only

Local deployment changes include dimensional scaling, injection of sensed LiDAR
shadow masks, explicit configuration of all former length constants, a
standstill-safe kinematic bicycle model, enforced speed and both steering-slew
bounds, actual risk-expanded headway constraints, closed-form ellipse tangents,
and package-relative imports.

No source from the GPLv3 `mpc_local_planner` repository is copied here.
`mpc_local_planner` is not installed in the audited ROS graph and is not the
solver used by this package. `dream_limo` contains a local DREAM-specific MPC:
CasADi supplies the bicycle dynamics/Jacobians and CVXPY+OSQP solves each
successive-linearization QP. The ROS planner remains a useful implementation
reference only: https://github.com/rst-tu-dortmund/mpc_local_planner.git
