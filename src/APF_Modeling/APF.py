"""
Artificial Potential Field (APF) – driving risk field computation.

The APF constructs a scalar potential landscape around the ego vehicle:
  - An **attractive potential** pulls the ego toward a goal (target lane position
    or desired waypoint), encoding driving efficiency.
  - **Repulsive potentials** push the ego away from surrounding vehicles (obstacles),
    encoding driving risk.

The improved formulation (goal-distance-weighted repulsion) is used so that
repulsive influence gracefully diminishes near the goal, avoiding the classic
goal-unreachable-with-obstacle-nearby (GNRON) problem.

The total field and its negative gradient (virtual force) are provided for
downstream maneuver planning.
"""

import numpy as np
from APF_config import APFConfig as Cfg


# ---------------------------------------------------------------------------
#  Vehicle / obstacle data helpers
# ---------------------------------------------------------------------------
def make_ego(x, y, vx=0.0, vy=0.0, length=4.8, width=2.5, label='Ego'):
    return dict(x=x, y=y, vx=vx, vy=vy, length=length, width=width, label=label)


def make_obstacle(x, y, vx=0.0, vy=0.0, length=5.0, width=2.2, label='Obs'):
    return dict(x=x, y=y, vx=vx, vy=vy, length=length, width=width, label=label)


def make_goal(x, y, label='Goal'):
    return dict(x=x, y=y, label=label)


# ---------------------------------------------------------------------------
#  Attractive potential and force
# ---------------------------------------------------------------------------
def attractive_potential(Xg, Yg, goal):
    """
    Improved attractive potential with conic-quadratic transition.

        U_att = 0.5 k ρ²            if ρ ≤ d
        U_att = d k ρ - 0.5 k d²    if ρ > d

    This caps the attractive gradient at large distances, preventing the
    attraction from overwhelming repulsion far from the goal.
    """
    dx = Xg - goal['x']
    dy = Yg - goal['y']
    rho = np.sqrt(dx ** 2 + dy ** 2)

    U = np.where(
        rho <= Cfg.d_att,
        0.5 * Cfg.k_att * rho ** 2,
        Cfg.d_att * Cfg.k_att * rho - 0.5 * Cfg.k_att * Cfg.d_att ** 2
    )
    return U


def attractive_force(Xg, Yg, goal):
    """Negative gradient of the attractive potential (force toward goal)."""
    dx = Xg - goal['x']
    dy = Yg - goal['y']
    rho = np.sqrt(dx ** 2 + dy ** 2) + 1e-12

    scale = np.where(
        rho <= Cfg.d_att,
        Cfg.k_att,
        Cfg.d_att * Cfg.k_att / rho
    )
    Fx = -scale * dx
    Fy = -scale * dy
    return Fx, Fy


# ---------------------------------------------------------------------------
#  Repulsive potential and force  (improved: goal-distance weighted)
# ---------------------------------------------------------------------------
def _repulsive_single_potential(Xg, Yg, obs, goal):
    """
    Improved repulsive potential from one obstacle, weighted by goal distance.

        U_rep = 0.5 η (1/ρ_obs - 1/ρ₀)² ρ_goal^n     if ρ_obs ≤ ρ₀
        U_rep = 0                                       otherwise

    The ρ_goal^n term ensures repulsion vanishes as the ego approaches the goal,
    resolving the GNRON problem.
    """
    dx_obs = Xg - obs['x']
    dy_obs = Yg - obs['y']
    rho_obs = np.sqrt(dx_obs ** 2 + dy_obs ** 2)

    dx_goal = Xg - goal['x']
    dy_goal = Yg - goal['y']
    rho_goal = np.sqrt(dx_goal ** 2 + dy_goal ** 2)

    in_range = rho_obs <= Cfg.rho_0
    rho_obs_safe = np.where(in_range, np.maximum(rho_obs, 0.3), 1.0)

    U = np.where(
        in_range,
        0.5 * Cfg.k_rep * (1.0 / rho_obs_safe - 1.0 / Cfg.rho_0) ** 2
        * rho_goal ** Cfg.n_rep,
        0.0
    )
    return U


def _repulsive_single_force(Xg, Yg, obs, goal):
    """
    Negative gradient of the improved repulsive potential.

    Two components:
      F_rep1 – pushes away from obstacle (classical term × ρ_goal^n)
      F_rep2 – pulls toward goal (new term, proportional to obstacle proximity)
    """
    dx_obs = Xg - obs['x']
    dy_obs = Yg - obs['y']
    rho_obs = np.sqrt(dx_obs ** 2 + dy_obs ** 2)

    dx_goal = Xg - goal['x']
    dy_goal = Yg - goal['y']
    rho_goal = np.sqrt(dx_goal ** 2 + dy_goal ** 2) + 1e-12

    in_range = rho_obs <= Cfg.rho_0
    rho_obs_safe = np.where(in_range, np.maximum(rho_obs, 0.3), 1.0)

    n = Cfg.n_rep
    inv_diff = 1.0 / rho_obs_safe - 1.0 / Cfg.rho_0

    # Component 1: repulsion from obstacle, scaled by goal distance
    coeff1 = Cfg.k_rep * inv_diff * rho_goal ** n / rho_obs_safe ** 2
    Fx1 = np.where(in_range, coeff1 * dx_obs / rho_obs_safe, 0.0)
    Fy1 = np.where(in_range, coeff1 * dy_obs / rho_obs_safe, 0.0)

    # Component 2: attraction toward goal, proportional to repulsive energy
    coeff2 = 0.5 * n * Cfg.k_rep * inv_diff ** 2 * rho_goal ** (n - 1)
    Fx2 = np.where(in_range, -coeff2 * dx_goal / rho_goal, 0.0)
    Fy2 = np.where(in_range, -coeff2 * dy_goal / rho_goal, 0.0)

    return Fx1 + Fx2, Fy1 + Fy2


# ---------------------------------------------------------------------------
#  Total APF: potential and force from all obstacles
# ---------------------------------------------------------------------------
def construct_APF(obstacles, goal, Xg=None, Yg=None):
    """
    Construct the full Artificial Potential Field.

    Parameters
    ----------
    obstacles : list[dict] – surrounding vehicles / static obstacles
    goal      : dict       – goal position
    Xg, Yg   : optional mesh grids; defaults to APFConfig grids

    Returns
    -------
    U_total     : 2-D ndarray – total potential (attractive + repulsive)
    U_att       : 2-D ndarray – attractive potential only
    U_rep_list  : list[2-D]   – per-obstacle repulsive potentials
    Fx, Fy      : 2-D ndarrays – total virtual force components
    """
    if Xg is None or Yg is None:
        Xg, Yg = Cfg.X_mesh, Cfg.Y_mesh

    # Attractive
    U_att = attractive_potential(Xg, Yg, goal)
    Fx_att, Fy_att = attractive_force(Xg, Yg, goal)

    # Repulsive (superposition)
    U_rep_total = np.zeros_like(Xg, dtype=float)
    Fx_rep_total = np.zeros_like(Xg, dtype=float)
    Fy_rep_total = np.zeros_like(Xg, dtype=float)
    U_rep_list = []

    for obs in obstacles:
        U_k = _repulsive_single_potential(Xg, Yg, obs, goal)
        Fx_k, Fy_k = _repulsive_single_force(Xg, Yg, obs, goal)
        U_rep_list.append(U_k)
        U_rep_total += U_k
        Fx_rep_total += Fx_k
        Fy_rep_total += Fy_k

    U_total = U_att + U_rep_total
    Fx = Fx_att + Fx_rep_total
    Fy = Fy_att + Fy_rep_total

    return U_total, U_att, U_rep_list, Fx, Fy
