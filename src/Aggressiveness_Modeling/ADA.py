"""
Asymmetric Driving Aggressiveness (ADA) – driving risk field computation.

The ADA models the spatially-varying risk perceived by an **ego vehicle**
from **surrounding agents** using a momentum-driven mechanical-wave analogy.
Each surrounding agent acts as a kinetic wave source whose intensity,
directional reach, and Doppler-modulated frequency combine to produce an
asymmetric risk contribution at every point in the ego's neighbourhood.
The total ADA is the linear superposition of all individual contributions.

Key equations follow Hu et al. (2025), IEEE Trans. Vehicular Technology,
Eq. (8)–(12).
"""

import numpy as np
from ADA_config import ADAConfig as Cfg


# ---------------------------------------------------------------------------
#  Vehicle data helpers
# ---------------------------------------------------------------------------
def make_vehicle(x, y, vx, vy, mass, heading=0.0, accel=0.0,
                 length=5.0, width=2.2, label=''):
    """
    Create a vehicle state dict.

    Parameters
    ----------
    x, y         : position (m) in world frame
    vx, vy       : velocity components (m/s)
    mass         : vehicle mass (kg)
    heading      : heading angle φ (rad), CCW from +x
    accel        : longitudinal acceleration (m/s²)
    length, width: vehicle footprint (m)
    label        : descriptive tag, e.g. 'Ego', 'Agent 1'
    """
    speed = np.sqrt(vx ** 2 + vy ** 2)
    return dict(x=x, y=y, vx=vx, vy=vy, speed=speed,
                mass=mass, heading=heading, accel=accel,
                length=length, width=width, label=label)


def make_ego(x, y, vx, vy, mass, heading=0.0, accel=0.0,
             length=5.0, width=2.5, label='Ego'):
    """Convenience wrapper – identical to make_vehicle but defaults to 'Ego'."""
    return make_vehicle(x, y, vx, vy, mass, heading, accel, length, width, label)


def make_agent(x, y, vx, vy, mass, heading=0.0, accel=0.0,
               length=5.0, width=2.2, label='Agent'):
    """Convenience wrapper for surrounding agents."""
    return make_vehicle(x, y, vx, vy, mass, heading, accel, length, width, label)


# ---------------------------------------------------------------------------
#  Elliptical pseudo-distance  (Eq. 11)
# ---------------------------------------------------------------------------
def _elliptical_distance(Xg, Yg, agent):
    """
    Compute the acceleration-modulated elliptical pseudo-distance from
    each grid point to a surrounding agent.  The major axis is aligned
    with the agent's heading; acceleration stretches the front lobe.
    """
    phi = agent['heading']
    dx = Xg - agent['x']
    dy = Yg - agent['y']

    cos_phi = np.cos(-phi)
    sin_phi = np.sin(-phi)

    # Rotate into agent-heading frame
    x_rot = dx * cos_phi - dy * sin_phi
    y_rot = dx * sin_phi + dy * cos_phi

    speed = agent['speed']
    a_i = agent['accel']

    # Ahead / behind discrimination (perpendicular to heading through agent)
    perp_angle = -phi + np.pi / 2
    indicator = dy - np.tan(perp_angle) * dx
    is_ahead = indicator >= 0 if phi >= 0 else indicator < 0

    tau1_ahead = Cfg.tau1 * np.exp(Cfg.beta * (speed - a_i * Cfg.t0))
    tau1_behind = Cfg.tau1 * np.exp(Cfg.beta * (speed + a_i * Cfg.t0))
    tau1_eff = np.where(is_ahead, tau1_ahead, tau1_behind)

    r = np.sqrt(x_rot ** 2 / tau1_eff ** 2 + y_rot ** 2 / Cfg.tau2 ** 2)
    return r


# ---------------------------------------------------------------------------
#  Doppler-like frequency modulation  (Eq. 10)
# ---------------------------------------------------------------------------
def _doppler_modulation(Xg, Yg, agent, ego):
    """
    Compute the Doppler frequency-shift factor
        f* = exp(μ₁ |v_agent| cos θ_agent + μ₂ |v_ego| cos θ_ego)
    which amplifies risk when agent and ego approach each other.
    """
    dx = Xg - agent['x']
    dy = Yg - agent['y']
    r = _elliptical_distance(Xg, Yg, agent)
    eps = 1e-12

    # Agent velocity projected onto agent→grid direction
    proj_agent = (dx * agent['speed'] * np.cos(agent['heading'])
                  + dy * agent['speed'] * np.sin(agent['heading'])) / (r + eps)

    # Ego velocity projected onto grid→agent direction (sign-reversed)
    proj_ego = (-dx * ego['speed'] * np.cos(ego['heading'])
                - dy * ego['speed'] * np.sin(ego['heading'])) / (r + eps)

    omega = np.exp(Cfg.mu1 * proj_agent + Cfg.mu2 * proj_ego)
    return omega, r


# ---------------------------------------------------------------------------
#  Single-agent risk contribution  (Eq. 12)
# ---------------------------------------------------------------------------
def compute_risk_single(Xg, Yg, agent, ego):
    """
    Evaluate the risk field contribution from one surrounding agent,
    as perceived by the ego vehicle, over the spatial grid.

        Ω = (P_agent / (2δ m_ego)) · f* · exp(−α r / m_agent)

    Returns
    -------
    risk : 2-D ndarray – risk intensity at each grid point
    """
    omega, r = _doppler_modulation(Xg, Yg, agent, ego)
    P = agent['mass'] * agent['speed']          # momentum of wave source

    risk = (P * omega
            * np.exp(-Cfg.alpha * np.maximum(r, 0) / agent['mass'])
            / (2 * Cfg.delta * ego['mass']))
    return risk


# ---------------------------------------------------------------------------
#  Superposed ADA from all surrounding agents
# ---------------------------------------------------------------------------
def construct_ADA(agents, ego, Xg=None, Yg=None):
    """
    Construct the full Kinetic Wave Field perceived by the ego vehicle
    from a list of surrounding agents (linear superposition).

    Parameters
    ----------
    agents : list[dict]  – surrounding vehicles (wave sources)
    ego    : dict         – ego vehicle (risk receiver)
    Xg, Yg : optional mesh grids; defaults to ADAConfig grids

    Returns
    -------
    ADA_total      : 2-D ndarray – total risk field
    ADA_individual : list[2-D ndarray] – per-agent contributions
    """
    if Xg is None or Yg is None:
        Xg, Yg = Cfg.X_mesh, Cfg.Y_mesh

    ADA_individual = []
    ADA_total = np.zeros_like(Xg, dtype=float)

    for agent in agents:
        risk = compute_risk_single(Xg, Yg, agent, ego)
        ADA_individual.append(risk)
        ADA_total += risk

    return ADA_total, ADA_individual
