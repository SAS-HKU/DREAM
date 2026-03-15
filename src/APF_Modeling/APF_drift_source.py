"""
APF_drift_source.py
===================
Adapter: Artificial Potential Field (APF) repulsive field as a
DRIFT-compatible source term Q(x, t).

The improved APF (goal-distance-weighted repulsion, Khatib 1986 + GNRON fix)
is evaluated on the DRIFT world-frame grid so it can be fed as the source Q
into the advection-diffusion-telegrapher PDE, replacing the default GVF-style
Gaussian kernels in compute_total_Q().

Only the **repulsive** term is used.  The attractive term encodes driving
efficiency toward a goal — it is a planning artefact, not a hazard source.
A driver-goal point is still required internally by the improved formulation
(for the ρ_goal^n weighting term); it is set 100 m ahead of the ego along
the ego's heading so that the goal-distance weighting is approximately
constant across the domain of interest and does not distort relative risk
magnitudes between nearby vehicles.

The PDE transmission law (advection + diffusion + telegrapher dynamics)
is unchanged — only the source Q differs.  This lets dream_dataset_benchmark
run three DRIFTInterface instances in parallel:

    drift_gvf  ->  compute_total_Q    (DREAM / GVF formulation)
    drift_ada  ->  compute_Q_ADA      (ADA-sourced DRIFT, integral-matched)
    drift_apf  ->  compute_Q_APF      (APF-sourced DRIFT, integral-matched)

Amplitude normalisation — integral matching
-------------------------------------------
The APF repulsive potential has no mass or velocity dependence; the raw
spatial integral depends only on the obstacle-influence radius ρ₀ and the
goal-distance weighting exponent n.  We match the integral to the GVF
reference for fairness.

Reference APF raw integral (2-D, over influence disk of radius ρ₀=15 m,
obstacle at 20 m from ego, goal at 100 m ahead of ego along heading,
vehicle class ignored since APF is class-agnostic):

    ∫∫ U_rep dA  ≈  π · k_rep · ρ_goal² · I_r

where I_r = ∫_{r_min}^{ρ₀} (1/r - 1/ρ₀)² r dr
         = [ln r - 2r/ρ₀ + r²/(2ρ₀²)]_{0.3}^{ρ₀}
         ≈ 2.452   (with r_min = 0.3 m, ρ₀ = 15 m)

    ρ_goal ≈ 80 m  (distance from vehicle at 20 m to goal at 100 m)
    ∫∫ U_rep dA ≈ π · 800 · 6400 · 2.452 ≈ 3.94 × 10⁷

Reference GVF integrals (from ADA_drift_source.py):
    car   → ≈ 1 292
    truck → ≈ 1 615

Per-class scale factors k = GVF_integral / APF_raw_integral:
    k_car   ≈ 3.28e-5   (1292  / 3.94e7)
    k_truck ≈ 4.10e-5   (1615  / 3.94e7)
    k_van   ≈ 3.28e-5   (same as car — APF has no class-dependent geometry)

After this normalisation, APF and GVF inject the same total risk energy per
vehicle.  Remaining visible differences in the propagated field R reflect
genuine model properties:
  • Shape: APF uses Euclidean distance with a hard influence cutoff at ρ₀=15 m;
           GVF uses heading-aligned Gaussians with σ_x=8 m / σ_y=2.5 m.
  • Isotropy: APF repulsion is isotropic (circular influence disk);
              GVF is anisotropic (elongated along vehicle heading).
  • Occlusion: GVF injects Q_occ in the shadow cone behind trucks; APF has
               NO occlusion source.  Any risk in the shadow zone under
               DRIFT (GVF) is attributable solely to the occlusion-aware
               source — not to amplitude scaling.
  • Velocity: APF risk is purely geometric (position-only); GVF risk grows
              with relative speed and braking intensity.
"""

import os
import sys
import numpy as np

# ── path: allow sibling imports (APF.py, APF_config.py) ─────────────────────
_pkg_dir = os.path.dirname(os.path.abspath(__file__))
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

from APF import _repulsive_single_potential   # noqa: E402


# ── per-class integral-matching scale factors ────────────────────────────────
# Derived from:  k = GVF_integral / APF_raw_integral
# Reference condition: obstacle at 20 m from ego, goal at 100 m ahead (ego heading)
# APF raw integral ≈ 3.94e7  (see module docstring for derivation)
# GVF reference integrals:  car ≈ 1292,  truck ≈ 1615  (from ADA_drift_source)
# APF has no class-dependent geometry so only the GVF target differs.
_APF_SCALE = {
    'car':   3.28e-5,   # 1292  / 3.94e7
    'truck': 4.10e-5,   # 1615  / 3.94e7
    'van':   3.28e-5,   # same as car — APF is class-agnostic
}
_APF_SCALE_DEFAULT = _APF_SCALE['car']

# Goal distance used for the ρ_goal^n weighting term.
# Placed 100 m ahead of ego so the weighting is approximately constant.
_GOAL_AHEAD_M = 100.0


# ── public API ───────────────────────────────────────────────────────────────

def compute_Q_APF(vehicles, ego, X, Y):
    """
    APF-based risk source Q(x, t) on the DRIFT world-frame grid,
    with per-class integral-matching normalisation.

    Drop-in replacement for pde_solver.compute_total_Q().
    Signature and return shape are identical so DRIFTInterface.step()
    can call it transparently via the source_fn keyword.

    Only the repulsive potential is used — the attractive term is a planning
    artefact, not a hazard source.  A driver goal is placed _GOAL_AHEAD_M
    metres ahead of the ego along its heading direction.

    Each vehicle's contribution is scaled by _APF_SCALE[class] so that
    its spatial integral equals the GVF integral for the same vehicle type
    under reference traffic conditions.

    Parameters
    ----------
    vehicles : list[dict]  – DRIFT vehicle dicts (id, x, y, vx, vy, class, ...)
    ego      : dict         – DRIFT ego vehicle dict
    X, Y     : ndarray      – world-frame meshgrids (cfg.X, cfg.Y)

    Returns
    -------
    Q_APF     : 2-D ndarray  – integral-matched APF repulsive field
    Q_zero    : 2-D ndarray  – zeros (placeholder for Q_veh slot)
    Q_zero    : 2-D ndarray  – zeros (placeholder for Q_occ slot)
    occ_mask  : bool ndarray – all-False  (APF has no occlusion model)
    """
    # Driver goal: 100 m ahead of ego along ego heading
    heading = float(ego.get('heading', 0.0))
    goal = {
        'x': float(ego['x']) + _GOAL_AHEAD_M * np.cos(heading),
        'y': float(ego['y']) + _GOAL_AHEAD_M * np.sin(heading),
    }

    Q = np.zeros_like(X, dtype=float)
    for v in vehicles:
        if v is None or v.get('id') == ego.get('id'):
            continue
        obs = {
            'x': float(v['x']),
            'y': float(v['y']),
        }
        vclass = v.get('class', 'car')
        scale  = _APF_SCALE.get(vclass, _APF_SCALE_DEFAULT)
        Q += _repulsive_single_potential(X, Y, obs, goal) * scale

    zero = np.zeros_like(X, dtype=float)
    return Q, zero, zero, np.zeros_like(X, dtype=bool)
