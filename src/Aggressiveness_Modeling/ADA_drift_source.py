"""
ADA_drift_source.py
===================
Adapter: Asymmetric Driving Aggressiveness (ADA) risk field as a
DRIFT-compatible source term Q(x, t).

The ADA momentum-wave field (Hu et al. 2025) is evaluated directly on the
DRIFT world-frame grid so it can be fed as the source Q into the
advection-diffusion-telegrapher PDE, replacing the default GVF-style
Gaussian kernels in compute_total_Q().

The PDE transmission law (advection + diffusion + telegrapher dynamics)
is unchanged — only the source Q differs.  This lets drift_dataset_visualization
run two DRIFTInterface instances in parallel:

    drift_gvf  ->  compute_total_Q    (current DREAM / GVF formulation)
    drift_ada  ->  compute_Q_ADA      (ADA-sourced DRIFT, integral-matched)

Amplitude normalisation — integral matching
-------------------------------------------
A single global scale factor cannot fairly align the two models because ADA's
spatial decay exponent is mass-dependent:

    L_char = m_agent · τ₁ / α       (longitudinal characteristic length)

With α=600, τ₁=0.2:
  • car   (m=1 800):  L_char ≈ 0.6 m
  • truck (m=15 000): L_char ≈ 5.0 m

GVF uses σ_x = 8 m for BOTH classes, so the car/truck spatial-integral
ratio is:
  • GVF  : truck/car ≈ 1.25×   (only amplitude differs)
  • ADA  : truck/car ≈ 578×    (m³ scaling: both amplitude AND range differ)

Per-class scale factors are therefore derived so that:

    ∫∫ Q_ADA_scaled(vehicle) dA  =  ∫∫ Q_GVF(vehicle) dA

for a reference traffic condition (v = 10 m/s, rel_speed = 2 m/s,
D = 20 m to ego, no braking, m_ego = 1 800 kg).

Reference GVF integral (2-D Gaussian, σ_par = 8.8 m, σ_perp = 2.5 m):
  ∫∫ Q_GVF dA = ω_class · exp(-20/70) · 2.07 · 3.0 · 2π · 8.8 · 2.5
  car   → ≈ 1 292
  truck → ≈ 1 615

Reference ADA raw integral (2-D exponential over elliptical pseudo-distance):
  ∫∫ ADA_raw dA = (m³ · v · τ₁τ₂ · 2π) / (2δ · m_ego · α²)
  car   (m=1 800) → ≈  11 307
  truck (m=15000) → ≈ 6 544 125
  van   (m=2 500) → ≈  30 297

Resulting per-class scale factors k = GVF_integral / ADA_raw_integral:
  k_car   ≈ 0.1143      (1292  / 11307)
  k_truck ≈ 2.468e-4    (1615  / 6544125)
  k_van   ≈ 0.0426      (1292  / 30297)

After this normalisation the comparison is structurally fair: both models
inject the same total risk energy per vehicle.  Remaining visible differences
in the propagated field R therefore reflect genuine model properties:
  • Shape: ADA car risk is concentrated (0.6 m range) vs GVF's 8 m Gaussian;
           ADA truck risk has Doppler front-loading; GVF is isotropic.
  • Occlusion: GVF injects Q_occ in the shadow cone behind trucks; ADA has
               NO occlusion source term at all.  Any risk that appears in the
               shadow zone under DRIFT (GVF) is therefore attributable to the
               occlusion-aware source — not to amplitude scaling — which
               supports DRIFT's uncertainty-handling narrative directly.
"""

import os
import sys
import numpy as np

# ── path: allow sibling imports (ADA.py, ADA_config.py) ─────────────────────
_pkg_dir = os.path.dirname(os.path.abspath(__file__))
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

from ADA import compute_risk_single   # noqa: E402


# ── vehicle mass defaults (DRIFT dicts carry no mass field) ──────────────────
_MASS = {
    'car':     1800.0,
    'truck':  15000.0,
    'van':     2500.0,
}
_MASS_DEFAULT = 1800.0

# ── per-class integral-matching scale factors ────────────────────────────────
# Derived from:  k = GVF_integral / ADA_raw_integral
# Reference condition: v=10 m/s, rel_speed=2 m/s, D=20 m, no braking,
#                      m_ego=1800 kg  (car ego, standard dataset setup)
# See module docstring for full derivation.
_ADA_SCALE = {
    'car':   0.1143,
    'truck': 2.468e-4,
    'van':   0.0426,
}
_ADA_SCALE_DEFAULT = _ADA_SCALE['car']


# ── vehicle dict conversion ──────────────────────────────────────────────────

def _to_ada(v):
    """Convert a DRIFT vehicle dict to the format ADA.compute_risk_single expects."""
    vx = float(v.get('vx', 0.0))
    vy = float(v.get('vy', 0.0))
    speed = float(np.hypot(vx, vy))
    return dict(
        x=float(v['x']),
        y=float(v['y']),
        vx=vx,
        vy=vy,
        speed=speed,
        mass=_MASS.get(v.get('class', 'car'), _MASS_DEFAULT),
        heading=float(v.get('heading', 0.0)),
        accel=float(v.get('a', 0.0)),
        length=float(v.get('length', 5.0)),
        width=float(v.get('width', 2.2)),
        label=str(v.get('class', 'car')),
    )


# ── public API ───────────────────────────────────────────────────────────────

def compute_Q_ADA(vehicles, ego, X, Y):
    """
    ADA-based risk source Q(x, t) on the DRIFT world-frame grid,
    with per-class integral-matching normalisation.

    Drop-in replacement for pde_solver.compute_total_Q().
    Signature and return shape are identical so DRIFTInterface.step()
    can call it transparently via the source_fn keyword.

    Each vehicle's contribution is scaled by _ADA_SCALE[class] so that
    its spatial integral equals the GVF integral for the same vehicle type
    under reference traffic conditions.  After this normalisation the two
    sources (GVF and ADA) are amplitude-fair: any remaining difference in
    the propagated risk field R reflects structural model properties only —
    most critically, GVF's occlusion source Q_occ which ADA entirely lacks.

    Parameters
    ----------
    vehicles : list[dict]  – DRIFT vehicle dicts (id, x, y, vx, vy, class, ...)
    ego      : dict         – DRIFT ego vehicle dict
    X, Y     : ndarray      – world-frame meshgrids (cfg.X, cfg.Y)

    Returns
    -------
    Q_ADA     : 2-D ndarray  – integral-matched ADA field
    Q_zero    : 2-D ndarray  – zeros (placeholder for Q_veh slot)
    Q_zero    : 2-D ndarray  – zeros (placeholder for Q_occ slot)
    occ_mask  : bool ndarray – all-False  (ADA has no occlusion model)
    """
    ada_ego = _to_ada(ego)

    Q = np.zeros_like(X, dtype=float)
    for v in vehicles:
        if v is None or v.get('id') == ego.get('id'):
            continue
        vclass = v.get('class', 'car')
        scale  = _ADA_SCALE.get(vclass, _ADA_SCALE_DEFAULT)
        Q += compute_risk_single(X, Y, _to_ada(v), ada_ego) * scale

    zero = np.zeros_like(X, dtype=float)
    return Q, zero, zero, np.zeros_like(X, dtype=bool)
