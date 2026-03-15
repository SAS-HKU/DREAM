import numpy as np


class APFConfig:
    """
    Configuration for the Artificial Potential Field (APF) risk model.

    The APF constructs a scalar risk field over the ego vehicle's neighbourhood
    by superposing a goal-directed attractive potential with obstacle-induced
    repulsive potentials.  The negative gradient of the total field yields the
    virtual force that guides maneuver planning.

    References:
        Khatib, "Real-time obstacle avoidance for manipulators and mobile robots,"
        Int. J. Robotics Research, 1986.
    """

    # --- Spatial grid ---
    x_min, x_max = -30, 150
    y_min, y_max = -10, 10
    resolution = 0.25          # grid spacing (m)

    X_ = np.arange(x_min, x_max + resolution, resolution)
    Y_ = np.arange(y_min, y_max + resolution, resolution)
    X_mesh, Y_mesh = np.meshgrid(X_, Y_)

    # --- Attractive field parameters ---
    k_att = 0.5                # attractive gain  (ξ)
    d_att = 40.0               # distance threshold: quadratic when ρ ≤ d, conic when ρ > d

    # --- Repulsive field parameters (improved formulation) ---
    k_rep = 800.0              # repulsive gain  (η)
    rho_0 = 15.0               # influence radius of each obstacle (m)
    n_rep = 2                  # exponent on goal-distance term in improved repulsive field

    # --- Lane geometry ---
    lane_width = 3.75          # (m)

    # --- Visualisation ---
    cmap_name = 'cool'
    figsize_3d = (10, 5)
    figsize_contour = (10, 4)
