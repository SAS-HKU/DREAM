import numpy as np


class KWFConfig:
    """
    Configuration for the Kinetic Wave Field (KWF) model.

    The KWF quantifies the driving risk perceived by an ego vehicle from
    surrounding agents through a momentum-driven, wave-propagation analogy
    with Doppler-like frequency modulation and asymmetric spatial decay.

    References:
        Hu et al., "Socially Game-Theoretic Lane-Change for Autonomous Heavy Vehicle
        based on Asymmetric Driving Aggressiveness," IEEE Trans. Vehicular Technology, 2025.
    """

    # --- Spatial grid ---
    x_min, x_max = -30, 150
    y_min, y_max = -6, 6
    resolution = 0.25          # grid spacing (m)

    X_ = np.arange(x_min, x_max + resolution, resolution)
    Y_ = np.arange(y_min, y_max + resolution, resolution)
    X_mesh, Y_mesh = np.meshgrid(X_, Y_)

    # --- KWF model coefficients (Eq. 12 in paper) ---
    mu1 = 0.2                  # Doppler weight – source velocity projection
    mu2 = 0.21                 # Doppler weight – ego velocity projection
    sigma = 0.1                # spatial decay coefficient  (σ)
    delta = 0.0005             # normalisation constant     (δ)
    alpha = 600                # amplitude decay coefficient

    # --- Elliptical pseudo-distance parameters (Eq. 11) ---
    tau1 = 0.2                 # longitudinal safe-distance threshold
    tau2 = 0.1                 # lateral safe-distance threshold
    beta = 0.05                # shape coefficient for acceleration influence
    t0 = 2.0                   # acceleration influence time (s)

    # --- Minimum velocity offset ---
    v0 = 5                     # ensures non-zero risk when agent speed ≈ 0

    # --- Lane geometry ---
    lane_width = 3.75          # (m)

    # --- Visualisation ---
    contour_levels = [0.8e4, 2.8e4, 4.8e4, 6.0e4, 8.8e4, 10.8e4, 14e4]
    cmap_name = 'cool'
    figsize_3d = (10, 5)
    figsize_contour = (10, 4)
