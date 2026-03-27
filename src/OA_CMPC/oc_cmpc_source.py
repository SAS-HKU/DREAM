"""
oa_cmpc_source.py
=================
Adapter: Occlusion-Aware Consistent MPC (OA-CMPC) risk source Q(x, t).

Based on:
    Zheng, M., Zheng, L., Zhu, L., & Ma, J. (2026).
    "Occlusion-Aware Consistent Model Predictive Control for Robot Navigation
    in Occluded Obstacle-Dense Environments."
    arXiv:2503.04563

Highway adaptation
------------------
The original paper uses a unicycle robot model with an ADMM-based multi-branch
solver.  This adapter extracts the occlusion-modelling contribution —
**tangent-line geometry + TTC-based reachable-set circles** — and maps it onto
a DRIFT-compatible source term Q(x, t) so it can be used as a drop-in
replacement for pde_solver.compute_total_Q() inside the PRIDEAM pipeline.

The ADMM multi-branch consistency part is not reproduced: instead the
worst-case (maximum v_obs_max) single branch is used, which corresponds to the
most conservative of the Nz branches and is the appropriate choice for a
highway safety benchmark.

Occlusion circle geometry (Paper §III-B, Eq. (5)–(8))
------------------------------------------------------
For each truck at position p_T with half-width w/2:

1.  Tangent angle from ego:
        α = arcsin( (w/2 + margin) / d_ego_truck )

2.  Two tangent directions from ego:
        θ_L = φ + α,   θ_R = φ - α
    where φ = atan2(T_y - E_y, T_x - E_x).

3.  Risk circle centres along each tangent at distances d_risk:
        c_i = p_T + d_risk * [cos θ_i, sin θ_i]
    Circles are kept only if c_i is AHEAD of the truck along its heading
    (i.e. in the shadow zone, not behind).

4.  TTC-based reachable-set radius (Paper Eq. (8)):
        r_i = (‖c_i − p_ego‖ / (v_ego + σ)) · v_obs_max + r_obs

    Interpretation: a hidden agent at c_i can travel at most
    (TTC × v_obs_max) before the ego arrives, so the uncertainty footprint
    at c_i has radius r_i.

5.  Each circle is rendered as a steep Gaussian bump on the Q grid so that
    the DRIFT PDE propagates and diffuses the risk naturally.

Amplitude normalisation
-----------------------
The GVF reference spatial integral for a truck is ≈ 1615 (see
ADA_drift_source.py for derivation).  For OA-CMPC, each truck generates
N_circles = len(D_RISK_LIST) × 2 circles (left + right tangent).  At
r_avg ≈ 5 m each Gaussian contributes ≈ A × π × r_avg² to the integral.

Matching condition:
    N_circles × A × π × r_avg²  =  GVF_truck_integral
    6          × A × π × 25     ≈  1615
    ⟹  A ≈ 3.43

This ensures the total risk energy injected per truck is comparable across
all four source models (GVF, ADA, APF, OA-CMPC), so differences in the
propagated field R reflect genuine structural properties of each model.

Visible-vehicle risk is inherited from GVF (compute_Q_vehicle) so that
non-occluded collision risk is modelled consistently with the other arms.

Drop-in interface
-----------------
compute_Q_OACMPC(vehicles, ego, X, Y)
    → (Q_total, Q_veh, Q_occ, occ_mask)

Identical signature to pde_solver.compute_total_Q().  Pass as `source_fn`
to DRIFTInterface.step() or DRIFTInterface.warmup().
"""

import os
import sys
import numpy as np

# Allow sibling imports from the project root
_pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

from pde_solver import compute_Q_vehicle, compute_Q_merge   # noqa: E402


# ── OA-CMPC parameters (Paper §III-B) ────────────────────────────────────────

# Maximum speed of a hidden agent [m/s].  Paper uses vobs_max as a design
# parameter controlling conservativeness.  10 m/s ≈ 36 km/h covers most
# merging / cut-in agents in a highway scenario.
_V_OBS_MAX = 10.0

# Velocity damping (prevents r_i → ∞ at v_ego → 0), [m/s]
_SIGMA_V = 1.0

# Physical radius of a hidden obstacle [m]
_R_OBS = 1.0

# Distances along the tangent line at which risk circles are placed [m].
# Three circles per tangent direction (= 6 per truck) mimic the Nz=3 risk
# configurations mentioned in the paper.
_D_RISK_LIST = [12.0, 22.0, 35.0]

# Lateral half-width margin added to truck half-width when computing tangent angle.
# Accounts for lane markings and driver comfort margin.
_TRUCK_MARGIN = 0.5  # [m]

# Gaussian amplitude for each risk circle (see amplitude normalisation above).
# A = GVF_truck_integral / (N_circles × π × r_avg²)
#   ≈ 1615 / (6 × π × 25) ≈ 3.43
_CIRCLE_AMP = 3.43

# Gaussian fall-off factor: width = r_circle / sqrt(2*_SIGMA_SCALE).
# Smaller → steeper (more like a hard constraint); larger → smoother.
_SIGMA_SCALE = 2.0   # results in σ_gauss = r_circle / sqrt(4) = r_circle/2

# Clamp on computed circle radius to prevent degenerate (too small / too large)
# values when ego is very close to or very far from the truck.
_R_MIN = 1.5   # [m]
_R_MAX = 9.0   # [m]


# ── public API ────────────────────────────────────────────────────────────────

def compute_Q_OACMPC(vehicles, ego, X, Y):
    """
    OA-CMPC occlusion risk source Q(x, t) on the DRIFT world-frame grid.

    Drop-in replacement for pde_solver.compute_total_Q().
    Signature and return shape are identical so DRIFTInterface.step()
    can call it transparently via the source_fn keyword.

    Risk circles are generated only from trucks (vclass == 'truck').
    Visible-vehicle risk (compute_Q_vehicle) and merge-zone risk
    (compute_Q_merge) are inherited from the GVF formulation for fairness.

    Parameters
    ----------
    vehicles : list[dict]
        DRIFT vehicle dicts — each must have keys:
        id, x, y, vx, vy, heading, class (str), width.
    ego : dict
        DRIFT ego vehicle dict (same schema).
    X, Y : ndarray
        World-frame meshgrids (cfg.X, cfg.Y), shape (ny, nx).

    Returns
    -------
    Q_total  : 2-D ndarray  – combined risk source (visible + occlusion + merge)
    Q_veh    : 2-D ndarray  – visible-vehicle component (GVF Gaussians)
    Q_occ    : 2-D ndarray  – OA-CMPC occlusion circles component
    occ_mask : bool ndarray – True inside any risk circle (used by
                              compute_diffusion_field to boost D_occ)
    """
    Q_occ    = np.zeros_like(X, dtype=float)
    occ_mask = np.zeros_like(X, dtype=bool)

    # Ego position and speed
    ex   = float(ego['x'])
    ey_e = float(ego['y'])
    v_ego = max(1.0, float(np.hypot(ego.get('vx', 0.0), ego.get('vy', 0.0))))

    for v in vehicles:
        if v is None or v.get('class') != 'truck':
            continue
        if v.get('id') == ego.get('id'):
            continue

        tx = float(v['x'])
        ty = float(v['y'])
        truck_hw  = float(v.get('width', 2.5)) * 0.5 + _TRUCK_MARGIN
        truck_hdg = float(v.get('heading', 0.0))

        # Distance ego → truck centre
        d_et = float(np.hypot(tx - ex, ty - ey_e))
        if d_et < truck_hw + 2.0:
            # Ego is too close (inside / touching truck) — skip
            continue

        # Direction from ego to truck centre
        phi = float(np.arctan2(ty - ey_e, tx - ex))

        # Half-angle subtended by truck width from ego (Paper §III-B, Eq. (5))
        sin_alpha = np.clip(truck_hw / d_et, 0.0, 0.99)
        alpha = float(np.arcsin(sin_alpha))

        # Two tangent directions: left edge (φ+α) and right edge (φ−α)
        tangent_dirs = [phi + alpha, phi - alpha]

        for d_risk in _D_RISK_LIST:
            for theta_t in tangent_dirs:

                # Candidate circle centre: along tangent from truck (Paper Eq. (7))
                cx = tx + d_risk * np.cos(theta_t)
                cy = ty + d_risk * np.sin(theta_t)

                # Gate: circle must lie in the shadow zone, i.e. AHEAD of the truck
                # along the truck's own heading (shadow extends forward)
                along_hdg = (np.cos(truck_hdg) * (cx - tx) +
                             np.sin(truck_hdg) * (cy - ty))
                if along_hdg <= 0.0:
                    continue

                # Distance from ego to circle centre
                d_to_circle = float(np.hypot(cx - ex, cy - ey_e))

                # TTC-based reachable-set radius (Paper Eq. (8))
                #   r_i = (‖c_i − p_ego‖ / (v_ego + σ)) · v_obs_max + r_obs
                ttc = d_to_circle / (v_ego + _SIGMA_V)
                r_circle = float(np.clip(ttc * _V_OBS_MAX + _R_OBS, _R_MIN, _R_MAX))

                # Gaussian half-width: σ_g = r_circle / sqrt(_SIGMA_SCALE * 2)
                sigma_g2 = (r_circle ** 2) / _SIGMA_SCALE

                # Squared distance from every grid point to this circle centre
                dist2 = (X - cx) ** 2 + (Y - cy) ** 2

                # Occlusion mask: everything inside the circle
                occ_mask |= (dist2 < r_circle ** 2)

                # Steep Gaussian bump centred on circle
                Q_occ += _CIRCLE_AMP * np.exp(-dist2 / sigma_g2)

    # Visible-vehicle risk: reuse GVF Gaussians (same as DREAM baseline)
    Q_veh  = compute_Q_vehicle(vehicles, ego, X, Y)

    # Merge-zone topology risk (same as GVF — scenario-invariant)
    Q_merge = compute_Q_merge(vehicles, ego, X, Y)

    Q_total = Q_veh + Q_occ + Q_merge
    return Q_total, Q_veh, Q_occ, occ_mask
