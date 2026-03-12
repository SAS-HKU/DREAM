"""
DREAM Uncertainty Test — Two-Threat Occlusion Scenario (Proactive Risk)
========================================================================
Demonstrates DREAM's core advantage: proactive, uncertainty-aware risk planning
against two simultaneous occluded threats that only DRIFT can "see".

Scenario
--------
1. EGO starts in the LEFT lane (path1c, lane 0) at s = 20 m, vx = 8 m/s.
   A slow TRUCK in the CENTRE lane (vehicle_centre[1], vd = 5.5 m/s, s ≈ 50 m)
   blocks sensor rays across both lane-change directions.

2. An OCCLUDED MERGER starts in the RIGHT lane, 8 m ahead of the centre truck,
   always fed to DRIFT so risk accumulates in the centre-lane merge zone —
   but NEVER in IDEAM's MPC arrays while it rides in the truck's shadow.

3. A fast HIDDEN CAR (LeftLaneFastCar, vd = 14 m/s) lurks in the LEFT lane
   behind the left-lane truck, fed only to DRIFT.  When the ego performs a
   lane change toward the left lane, the hidden car's occlusion lifts and it
   is revealed into the MPC vehicle arrays.

4. When IDEAM first commands a lane change to the CENTRE lane (to pass the
   centre truck), the OccludedMerger simultaneously starts its own LC
   right → centre, creating a head-on merge conflict:
     · IDEAM:  enters centre → merger mid-LC → tight gap / hard braking.
     · DREAM:  DRIFT risk pre-built in centre merge zone → decision veto →
               holds left lane → no conflict.

DREAM vs IDEAM contrast
-----------------------
* IDEAM (baseline):  No risk field → blind to both hidden threats.  Returns
  to centre → encounters merger → merge conflict, tight TTC, hard braking.

* DREAM (ours):  DRIFT risk in centre zone + left shadow zone (proactive) →
  decision veto → holds left lane → smooth speed profile, safe gap.

Key metrics
-----------
  · Min TTC with merger   (IDEAM << DREAM)
  · Peak deceleration     (IDEAM > DREAM)
  · Min gap to merger     (IDEAM < DREAM)
  · Speed profile         (DREAM smoother)
"""

# ===========================================================================
# IMPORTS
# ===========================================================================
import sys
import os
import math
import time
import copy
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path as MplPath
from matplotlib.transforms import Affine2D
from scipy.ndimage import gaussian_filter as _gf
import scienceplots  # noqa: F401 — registers "science" style with matplotlib

sys.path.append("C:\\IDEAM_implementation-main\\decision_improve")
from Control.MPC import *
from Control.constraint_params import *
from Model.Dynamical_model import *
from Model.params import *
from Model.surrounding_params import *
from Model.Surrounding_model import *
from Model.surrounding_vehicles import *
from Control.HOCBF import *
from DecisionMaking.decision_params import *
from DecisionMaking.give_desired_path import *
from DecisionMaking.util import *
from DecisionMaking.util_params import *
from DecisionMaking.decision import *
from Prediction.surrounding_prediction import *
from progress.bar import Bar

# DRIFT / PRIDEAM
from config import Config as cfg
from pde_solver import (PDESolver, compute_total_Q, compute_velocity_field,
                        compute_diffusion_field, create_vehicle as drift_create_vehicle)
from Integration.drift_interface import DRIFTInterface
from Integration.prideam_controller import create_prideam_controller
from Integration.integration_config import get_preset

# ===========================================================================
# INTEGRATION MODE
# ===========================================================================
INTEGRATION_MODE = "conservative"
config_integration = get_preset(INTEGRATION_MODE)
config_integration.apply_mode()

print("=" * 70)
print(f"UNCERTAINTY TEST  |  DREAM mode: {INTEGRATION_MODE.upper()}")
print("=" * 70)

# ===========================================================================
# SCENARIO PARAMETERS
# ===========================================================================

# -- Truck designation (lane_idx, vehicle_idx): desired_speed [m/s] -----------
# (1, 1) = centre-lane slow truck — sits laterally between left-lane ego and
#           right-lane OccludedMerger, blocking sensor rays across centre lane.
#           Confirmed from 120_400 init: vehicle_centre[1] at s ≈ 50 m.
TRUCK_DESIGNATIONS = {
    (1, 1): 5.5,    # centre-lane slow truck (vd = 5.5 m/s)
}
_AGENT_TRUCK_LANE = 1   # lane 1 = centre
_AGENT_TRUCK_IDX  = 1   # vehicle_centre[1] at s ≈ 50 m

# -- Left-lane fast car (occluded, DRIFT-only until revealed) ---------------
# Hidden behind the left-lane truck.  Always in DRIFT (builds risk in the
# left shadow zone) but EXCLUDED from IDEAM's vehicle arrays until it exits
# the truck's occlusion shadow (geometric reveal) or fallback step fires.
LEFT_FAST_CAR_S_INIT          = 78.0   # [m] start ~20 m ahead of left truck
LEFT_FAST_CAR_VD              = 14.0   # [m/s] fast car desired speed
LEFT_FAST_CAR_REVEAL_FALLBACK = 90     # step — fires if occlusion never clears

# -- DRIFT risk boost around truck -----------------------------------------
TRUCK_RISK_BOOST    = 2.5
TRUCK_RISK_SIGMA    = 12.0
TRUCK_WEIGHT_SCALE  = 4.0
TRUCK_INFLUENCE_DIST = 70.0
TRUCK_PROACTIVE_DIST = 55.0
TRUCK_PROACTIVE_RISK = 0.35
TRUCK_SAFE_SPEED    = 7.5

# Asymmetric constraint scaling
TRUCK_LONG_EXTRA   = 8.0
TRUCK_TH_EXTRA     = 1.0
TRUCK_AL_SCALE     = 0.5
TRUCK_BL_SCALE     = 0.7
TRUCK_CENTER_RELAX = 4.0

# Squeeze avoidance
SQUEEZE_LON_DIST   = 25.0
SQUEEZE_MIN_ACCEL  = 0.8

# -- Occluded Merger (right lane → centre lane) ----------------------------
# Ghost-tracks s_offset metres ahead of the centre truck in the right lane.
# Always fed to DRIFT (risk builds in the centre-lane merge zone) but NEVER
# in IDEAM's MPC arrays while in the right lane.
# Trigger: when IDEAM first commands left→centre LC (path_now_i==0 →
# path_di_i==1), OccludedMerger simultaneously starts LC (right→centre).
MERGER_TRUCK_S_OFFSET = 8.0    # [m] how far ahead of truck merger rides
MERGER_VX             = 5.5    # [m/s] cruise speed (matches truck speed)
MERGER_LC_DURATION    = 35     # steps (3.5 s) for the right → centre LC
MERGER_LC_FALLBACK    = 220    # step fallback if IDEAM never commands centre

# Collision / near-collision thresholds
NEAR_COLLISION_DIST = 8.0          # [m]
COLLISION_DIST      = 3.0          # [m]

# -- Startup geometry constraints (checked/printed at init) ----------------
# Violation warnings are printed but do NOT halt the simulation.
EGO_FRONT_CLEAR_M    = 50.0  # [m] no left-lane vehicle in (ego_s, ego_s+50m)
TRUCK_SIDE_EXCL_M    = 25.0  # [m] no lane-0/2 vehicle within ±25m of truck s

# -- Visualization ---------------------------------------------------------
CAR_LENGTH   = 3.5
CAR_WIDTH    = 1.2
TRUCK_LENGTH = 12.0
TRUCK_WIDTH  = 2.0

EGO_DREAM_COLOR   = '#2196F3'   # Blue
EGO_IDEAM_COLOR   = '#4CAF50'   # Green
SURROUND_COLOR    = '#FFD600'   # Yellow
TRUCK_COLOR       = '#FF6F00'   # Dark orange
SHADOW_COLOR      = '#4A4A4A'   # Dark grey
AGENT_OCC_COLOR   = '#9C27B0'   # Purple (ghost / occluded)
AGENT_VIS_COLOR   = '#E91E63'   # Magenta (visible)

RISK_ALPHA  = 0.65
RISK_CMAP   = 'jet'
RISK_LEVELS = 40
RISK_VMAX   = 2.0

N_t = 400
dt  = 0.1
x_area = 50.0
y_area = 15.0

save_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                        "figsave_uncertainty_v1")
os.makedirs(save_dir, exist_ok=True)

# ===========================================================================
# HELPER: draw a rotated rectangle for one vehicle
# ===========================================================================

def draw_vehicle_rect(ax, x, y, yaw_rad, length, width,
                      facecolor, edgecolor='black', lw=0.8,
                      zorder=3, alpha=1.0, linestyle='-'):
    rect = mpatches.FancyBboxPatch(
        (-length / 2, -width / 2), length, width,
        boxstyle="round,pad=0.05",
        facecolor=facecolor, edgecolor=edgecolor,
        linewidth=lw, alpha=alpha, zorder=zorder,
        linestyle=linestyle
    )
    t = Affine2D().rotate(yaw_rad).translate(x, y) + ax.transData
    rect.set_transform(t)
    ax.add_patch(rect)
    return rect


# ===========================================================================
# HELPER: truck occlusion shadow polygon (same as emergency_test_prideam.py)
# ===========================================================================

def compute_truck_shadow(ego_x, ego_y, truck_state, shadow_length=55.0):
    """Shadow polygon from ego through truck corners — returns Nx2 array or None."""
    tx, ty, yaw = truck_state[3], truck_state[4], truck_state[5]
    dx, dy = tx - ego_x, ty - ego_y
    dist = math.sqrt(dx**2 + dy**2)
    if dist < 3:
        return None

    L, W = TRUCK_LENGTH, TRUCK_WIDTH
    corners_local = np.array([[-L/2, -W/2], [L/2, -W/2],
                               [L/2,  W/2], [-L/2,  W/2]])
    cos_h, sin_h = math.cos(yaw), math.sin(yaw)
    rot = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
    corners = (rot @ corners_local.T).T + np.array([tx, ty])

    angles = np.arctan2(corners[:, 1] - ego_y, corners[:, 0] - ego_x)
    left_corner  = corners[np.argmax(angles)]
    right_corner = corners[np.argmin(angles)]

    l_dir = left_corner  - np.array([ego_x, ego_y])
    l_dir /= (np.linalg.norm(l_dir) + 1e-9)
    r_dir = right_corner - np.array([ego_x, ego_y])
    r_dir /= (np.linalg.norm(r_dir) + 1e-9)

    return np.array([left_corner,
                     left_corner  + l_dir * shadow_length,
                     right_corner + r_dir * shadow_length,
                     right_corner])


def draw_shadow_polygon(ax, shadow_polygon, alpha=0.30):
    if shadow_polygon is None:
        return
    patch = plt.Polygon(shadow_polygon,
                        facecolor=SHADOW_COLOR, alpha=alpha,
                        edgecolor='red', linewidth=1.8,
                        linestyle='--', zorder=2)
    ax.add_patch(patch)


# ===========================================================================
# LEFT LANE FAST CAR  (occluded, DRIFT-only until geometrically revealed)
# ===========================================================================

class LeftLaneFastCar:
    """
    A fast car in the left lane hidden behind the left-lane truck.

    Pre-reveal:
        Rides at ``s_init`` in the left lane at high ``vd``, updated by IDM
        using left-lane vehicles ahead as leaders.  Always fed to DRIFT
        (builds risk in the truck shadow zone) but EXCLUDED from IDEAM's
        vehicle arrays → IDEAM sees the left lane as clear.

    Reveal trigger (primary — geometric):
        Each step ``check_occlusion(ego_x, ego_y, truck_state)`` tests whether
        the car is still inside the left-truck's sight-line shadow polygon.
        Once it exits (ego shifts laterally during LC, widening the gap), the
        car is marked ``revealed = True`` and injected into ``vl_mpc``.

    Reveal trigger (fallback):
        If the geometric check never fires, forced at
        ``LEFT_FAST_CAR_REVEAL_FALLBACK`` step.
    """

    def __init__(self, paths, path_data, steer_range, s_init, vd, lane_idx=0):
        self.paths       = paths
        self.path_data   = path_data
        self.steer_range = steer_range
        self.lane_idx    = lane_idx
        self.vd          = vd
        self.model       = Curved_Road_Vehicle(**surrounding_params())

        path = paths[lane_idx]
        x0, y0   = path(s_init)
        psi0     = path.get_theta_r(s_init)
        xc, yc, samps = path_data[lane_idx]
        try:
            s0, ey0, epsi0 = find_frenet_coord(path, xc, yc, samps,
                                               [x0, y0, psi0])
        except Exception:
            s0, ey0, epsi0 = s_init, 0.0, 0.0
        self.state = np.array([s0, ey0, epsi0, x0, y0, psi0, vd, 0.0])

        self.revealed    = False
        self.reveal_step = None
        self.occluded    = True   # start inside truck shadow

    # ------------------------------------------------------------------
    def update(self, vehicle_left):
        """Advance one step using IDM (left-lane vehicles as leaders)."""
        s  = self.state[0]
        vx = self.state[6]
        path = self.paths[self.lane_idx]

        # Find nearest left-lane vehicle ahead
        s_ahead, v_ahead = None, None
        for veh in vehicle_left:
            gap = veh[0] - s - 3.5
            if gap > 2.0 and (s_ahead is None or veh[0] < s_ahead):
                s_ahead, v_ahead = veh[0], veh[6]

        x_n, y_n, psi_n, vx_n, _, a = self.model.update_states(
            s, vx, self.vd, s_ahead, v_ahead, path, self.steer_range)

        xc, yc, samps = self.path_data[self.lane_idx]
        try:
            s_n, ey_n, epsi_n = find_frenet_coord(
                path, xc, yc, samps, [x_n, y_n, psi_n])
        except Exception:
            s_n, ey_n, epsi_n = s + vx * dt, 0.0, 0.0
        self.state = np.array([s_n, ey_n, epsi_n, x_n, y_n, psi_n,
                               max(vx_n, 0.0), a])

    # ------------------------------------------------------------------
    def check_occlusion(self, ego_x, ego_y, truck_state):
        """Update self.occluded using left-truck sight-line shadow."""
        shadow_poly = compute_truck_shadow(ego_x, ego_y, truck_state)
        if shadow_poly is None:
            self.occluded = False
            return
        poly_path     = MplPath(shadow_poly)
        ax, ay        = float(self.state[3]), float(self.state[4])
        self.occluded = bool(poly_path.contains_point([ax, ay]))

    # ------------------------------------------------------------------
    def to_drift_vehicle(self, vid=997):
        """Always provide to DRIFT (builds risk even when occluded)."""
        psi = self.state[5]
        vx  = self.state[6]
        v   = drift_create_vehicle(
            vid=vid,
            x=float(self.state[3]), y=float(self.state[4]),
            vx=vx * math.cos(psi), vy=vx * math.sin(psi),
            vclass='car')
        v['heading'] = psi
        v['a']       = float(self.state[7])
        return v

    # ------------------------------------------------------------------
    def to_ideam_row(self):
        """Vehicle array row [s, ey, epsi, x, y, psi, vx, a]."""
        return self.state.copy()

    # ------------------------------------------------------------------
    @property
    def x(self):   return float(self.state[3])
    @property
    def y(self):   return float(self.state[4])
    @property
    def psi(self): return float(self.state[5])
    @property
    def vx(self):  return float(self.state[6])


# ===========================================================================
# OCCLUDED MERGER  (right lane → centre lane, simultaneous merge conflict)
# ===========================================================================

class OccludedMerger:
    """
    A vehicle in the RIGHT lane laterally occluded from left-lane ego by the
    centre-lane truck.

    PRE-TRIGGER — KINEMATIC GHOST TRACKING:
        Rides ``s_offset`` metres ahead of the truck in the right lane at the
        truck's speed.  Always fed to DRIFT (builds risk in the centre-lane
        merge zone) but NEVER in IDEAM's MPC arrays → IDEAM sees clear.

    TRIGGER:
        When IDEAM first commands left→centre LC (path_now_i==0, path_di_i==1),
        the merger simultaneously starts its own LC (right → centre) over
        ``lc_duration`` steps.  Both IDEAM and the merger try to enter the
        centre lane at the same time:
          · IDEAM:  enters centre → merger mid-LC → tight gap / hard braking.
          · DREAM:  DRIFT risk in centre-lane merge zone (proactive, from
                    merger always being in DRIFT) → decision veto → stays
                    in left lane → no conflict.

    POST-TRIGGER — LANE CHANGE EXECUTION:
        Lateral position interpolates smoothly from right-lane centreline to
        centre-lane centreline over ``lc_duration`` steps.  Speed constant.
    """

    def __init__(self, paths, path_data, s_offset, lc_duration, vx_cruise):
        self.paths       = paths        # [path1c, path2c, path3c]
        self.path_data   = path_data    # [(x1c,y1c,s1c), ...]
        self.s_offset    = s_offset
        self.lc_duration = lc_duration
        self.vx          = vx_cruise

        self.state        = np.zeros(8)
        self.lc_triggered = False
        self.lc_progress  = 0.0
        self.lc_step      = None

    # ------------------------------------------------------------------
    def trigger_lc(self, step_idx):
        """Called once when IDEAM commands centre-lane (or fallback fires)."""
        if not self.lc_triggered:
            self.lc_triggered = True
            self.lc_step      = step_idx

    # ------------------------------------------------------------------
    def update(self, truck_state):
        """Advance one timestep — ghost tracking + smooth LC interpolation."""
        truck_s  = float(truck_state[0])
        truck_vx = float(truck_state[6])
        target_s = truck_s + self.s_offset

        # Advance LC progress
        if self.lc_triggered:
            self.lc_progress = min(1.0,
                                   self.lc_progress + 1.0 / self.lc_duration)

        # Cartesian interpolation: right lane → centre lane
        path_right  = self.paths[2]   # path3c
        path_centre = self.paths[1]   # path2c
        xr, yr = path_right(target_s)
        xc, yc = path_centre(target_s)
        psi_n  = path_right.get_theta_r(target_s)

        x_n = xr + self.lc_progress * (xc - xr)
        y_n = yr + self.lc_progress * (yc - yr)

        # Frenet coordinates on whichever lane is dominant
        if self.lc_progress >= 0.5:
            ref_path = path_centre
            rx, ry, rs = self.path_data[1]
        else:
            ref_path = path_right
            rx, ry, rs = self.path_data[2]
        try:
            s_n, ey_n, epsi_n = find_frenet_coord(
                ref_path, rx, ry, rs, [x_n, y_n, psi_n])
        except Exception:
            s_n, ey_n, epsi_n = target_s, 0.0, 0.0

        self.state = np.array([s_n, ey_n, epsi_n, x_n, y_n, psi_n,
                               max(truck_vx, 0.0), 0.0])

    # ------------------------------------------------------------------
    def in_centre_lane(self):
        """True once merger is substantially (≥ 50 %) into centre lane."""
        return self.lc_progress >= 0.5

    def is_occluded(self):
        """Occluded from IDEAM while still primarily in right lane."""
        return not self.in_centre_lane()

    # ------------------------------------------------------------------
    def to_drift_vehicle(self, vid=998):
        """Always provided to DRIFT to build risk in the merge zone."""
        psi = float(self.state[5])
        vx  = float(self.vx)
        v   = drift_create_vehicle(
            vid=vid,
            x=float(self.state[3]), y=float(self.state[4]),
            vx=vx * math.cos(psi), vy=vx * math.sin(psi),
            vclass='car')
        v['heading'] = psi
        v['a']       = 0.0
        return v

    # ------------------------------------------------------------------
    def to_ideam_row(self):
        """Vehicle array row [s, ey, epsi, x, y, psi, vx, a]."""
        return self.state.copy()

    # ------------------------------------------------------------------
    @property
    def x(self):   return float(self.state[3])
    @property
    def y(self):   return float(self.state[4])
    @property
    def psi(self): return float(self.state[5])


# ===========================================================================
# INITIALIZATION
# ===========================================================================

bar = Bar(max=N_t - 1)
boundary    = 1.0
steer_range = [math.radians(-8.0), math.radians(8.0)]

X0   = [8.0, 0.0, 0.0, 20.0, 0.0, 0.0]
X0_g = [path1c(X0[3])[0], path1c(X0[3])[1], path1c.get_theta_r(X0[3])]

path_center  = np.array([path1c, path2c, path3c], dtype=object)
sample_center = np.array([samples1c, samples2c, samples3c], dtype=object)
x_center = [x1c, x2c, x3c]
y_center = [y1c, y2c, y3c]
x_bound  = [x1, x2]
y_bound  = [y1, y2]
path_bound        = [path1, path2]
path_bound_sample = [samples1, samples2]

X_traj = [X0_g[0]]
Y_traj = [X0_g[1]]
oa, od  = 0.0, 0.0
path_desired = []
pathRecord   = [0]   # ego starts in left lane (index 0)

Params           = params()
Constraint_params = constraint_params()
dynamics         = Dynamic(**Params)
decision_param   = decision_params()
decision_maker   = decision(**decision_param)

# -- Expand DRIFT grid to cover full road extent ----------------------
# The default grid (x∈[-150, 255.2]) only covers part of the road.
# We compute the true road bounding box from loaded path coordinates
# and expand cfg before the PDESolver is instantiated.
_road_x_all = np.concatenate([x1c, x2c, x3c])
_road_y_all = np.concatenate([y1c, y2c, y3c])
_margin_m   = 25.0          # extra margin beyond road edge [m]
_dx_orig = (255.2 - (-150.0)) / (250 - 1)   # ≈ 1.62 m/cell (original)
_dy_orig = ((-45.3) - (-225.2)) / (80 - 1)  # ≈ 2.28 m/cell (original)

_new_x_min = min(float(np.min(_road_x_all)) - _margin_m, cfg.x_min)
_new_x_max = max(float(np.max(_road_x_all)) + _margin_m, cfg.x_max)
_new_y_min = min(float(np.min(_road_y_all)) - _margin_m, cfg.y_min)
_new_y_max = max(float(np.max(_road_y_all)) + _margin_m, cfg.y_max)

cfg.x_min, cfg.x_max = _new_x_min, _new_x_max
cfg.y_min, cfg.y_max = _new_y_min, _new_y_max
cfg.nx = max(250, int((_new_x_max - _new_x_min) / _dx_orig) + 2)
cfg.ny = max(80,  int((_new_y_max - _new_y_min) / _dy_orig) + 2)
cfg.dx = (_new_x_max - _new_x_min) / (cfg.nx - 1)
cfg.dy = (_new_y_max - _new_y_min) / (cfg.ny - 1)
cfg.x  = np.linspace(cfg.x_min, cfg.x_max, cfg.nx)
cfg.y  = np.linspace(cfg.y_min, cfg.y_max, cfg.ny)
cfg.X, cfg.Y = np.meshgrid(cfg.x, cfg.y)
print(f"[DRIFT Grid] x∈[{cfg.x_min:.0f}, {cfg.x_max:.0f}] m  "
      f"y∈[{cfg.y_min:.0f}, {cfg.y_max:.0f}] m  "
      f"nx={cfg.nx}, ny={cfg.ny}  "
      f"({cfg.nx*cfg.ny/1e3:.0f}k cells)")

# -- PRIDEAM controller ------------------------------------------------
controller = create_prideam_controller(
    paths={0: path1c, 1: path2c, 2: path3c},
    risk_weights={
        'mpc_cost':           config_integration.mpc_risk_weight,
        'cbf_modulation':     config_integration.cbf_alpha,
        'decision_threshold': config_integration.decision_risk_threshold,
    }
)
controller.get_path_curvature(path=path1c)   # ego starts in left lane

# -- Surrounding vehicles (from pickle for reproducibility) -----------
params_dir  = r"C:\DREAM\file_save\120_400"
surroundings = Surrounding_Vehicles(steer_range, dt, boundary, params_dir)

util_params_ = util_params()
utils        = LeaderFollower_Uitl(**util_params_)
controller.set_util(utils)
mpc_controller = controller.mpc

# -- Baseline IDEAM MPC (independent parallel planner) ----------------
baseline_mpc_viz  = LMPC(**constraint_params())
utils_ideam_viz   = LeaderFollower_Uitl(**util_params_)
baseline_mpc_viz.set_util(utils_ideam_viz)
baseline_mpc_viz.get_path_curvature(path=path1c)   # same lane as ego: left

# IDEAM and DREAM must share identical initial ego state and lane so that
# the only performance difference comes from the planner, not from state bias.
X0_ideam   = [8.0, 0.0, 0.0, 20.0, 0.0, 0.0]      # identical to DREAM X0
X0_g_ideam = [path1c(X0_ideam[3])[0],               # ← path1c (left lane)
               path1c(X0_ideam[3])[1],
               path1c.get_theta_r(X0_ideam[3])]
oa_ideam_viz, od_ideam_viz  = 0.0, 0.0
last_X_ideam_viz             = None
last_ideam_panel_horizon_viz = None
path_changed_ideam_viz       = 0   # starts in left lane (index 0)

# Panel snapshots (safe on frame 0)
ideam_panel_X0   = list(X0_ideam)
ideam_panel_X0_g = list(X0_g_ideam)

path_changed = 0   # ego starts in left lane (index 0)

# ── Speed designations ───────────────────────────────────────────────────
# Applied BEFORE truck designations so truck is not accidentally overwritten.
#
# Exclusion-zone clearance (reviewer constraint):
#   From the 120_400 init snapshot, left[1] (s≈51 m) and right[1] (s≈53 m)
#   sit within TRUCK_SIDE_EXCL_M of the truck (s≈50 m).  Setting them to a
#   high desired speed makes them accelerate away in the first few steps so
#   the merge zone is clean before the conflict window opens.
surroundings.vd_left_all[1]   = 30.0   # clears truck left-flank quickly
surroundings.vd_right_all[1]  = 30.0   # clears truck right-flank quickly

# Clear left[0] so the file-loaded vehicle vacates the zone ahead of the
# synthetic LeftLaneBlocker quickly (pre-advance will move it ≈ 60 m away).
surroundings.vd_left_all[0] = 30.0

# -- Designate trucks & set desired speed (must come AFTER general boost) --
for (_tl, _tv), _tvd in TRUCK_DESIGNATIONS.items():
    if _tl == 0:   surroundings.vd_left_all[_tv]   = _tvd
    elif _tl == 1: surroundings.vd_center_all[_tv] = _tvd
    elif _tl == 2: surroundings.vd_right_all[_tv]  = _tvd

# Path data tuple list for find_frenet_coord
_path_data = [
    (x1c, y1c, samples1c),
    (x2c, y2c, samples2c),
    (x3c, y3c, samples3c),
]

# Construct LeftLaneFastCar — fast occluded car in left lane behind the
# left-lane truck.  Always in DRIFT, revealed to IDEAM geometrically.
left_fast_car = LeftLaneFastCar(
    paths       = [path1c, path2c, path3c],
    path_data   = _path_data,
    steer_range = steer_range,
    s_init      = LEFT_FAST_CAR_S_INIT,
    vd          = LEFT_FAST_CAR_VD,
    lane_idx    = 0,   # left lane
)

# Construct OccludedMerger — ghost-tracks MERGER_TRUCK_S_OFFSET m ahead of
# the centre truck in the right lane.  Always in DRIFT; hidden from IDEAM
# until it substantially enters the centre lane.
merger = OccludedMerger(
    paths       = [path1c, path2c, path3c],
    path_data   = _path_data,
    s_offset    = MERGER_TRUCK_S_OFFSET,
    lc_duration = MERGER_LC_DURATION,
    vx_cruise   = MERGER_VX,
)

# ===========================================================================
# CONVERT IDEAM VEHICLES → DRIFT FORMAT
# ===========================================================================

def convert_to_drift(vehicle_left, vehicle_centre, vehicle_right,
                     truck_set=None, max_per_lane=5, exclude_set=None):
    """exclude_set: set of (lane_idx, vehicle_idx) to skip (LC agent slots)."""
    vehicles = []
    vid = 1
    for lane_idx, arr in enumerate([vehicle_left, vehicle_centre, vehicle_right]):
        if arr is None:
            continue
        count = 0
        for ri, row in enumerate(arr):
            if exclude_set and (lane_idx, ri) in exclude_set:
                continue   # this slot is now managed by an IdeamDecisionLCAgent
            is_truck = truck_set is not None and (lane_idx, ri) in truck_set
            if max_per_lane is not None and count >= max_per_lane and not is_truck:
                continue
            if len(row) < 7:
                continue
            psi   = row[5]
            v_lon = row[6]
            accel = row[7] if len(row) > 7 else 0.0
            vclass = 'truck' if is_truck else 'car'
            v = drift_create_vehicle(
                vid=vid,
                x=row[3], y=row[4],
                vx=v_lon * math.cos(psi),
                vy=v_lon * math.sin(psi),
                vclass=vclass)
            v['a']       = accel
            v['heading'] = psi
            vehicles.append(v)
            vid += 1
            count += 1
    return vehicles


# ===========================================================================
# DRIFT INITIALIZATION + ROAD MASK
# ===========================================================================

drift = controller.drift

print("Computing road boundary mask...")
_step    = 50
left_edge  = np.column_stack([x[::_step],  y[::_step]])
right_edge = np.column_stack([x3[::_step], y3[::_step]])
road_polygon  = np.vstack([left_edge, right_edge[::-1]])
road_mpl_path = MplPath(road_polygon)

grid_pts = np.column_stack([cfg.X.ravel(), cfg.Y.ravel()])
inside   = road_mpl_path.contains_points(grid_pts).reshape(cfg.X.shape).astype(float)
road_mask = np.clip(_gf(inside, sigma=1.5), 0, 1)
drift.set_road_mask(road_mask)
print(f"  Road mask: {np.sum(inside > 0.5)} / {inside.size} on-road "
      f"({100*np.mean(inside > 0.5):.1f}%)")

# ===========================================================================
# PRE-ADVANCE SURROUNDINGS — clear exclusion-zone positions before warm-up
# ===========================================================================
# Root cause of position bug: surroundings loads positions from the file
# (120_400) at init.  Setting vd_left_all[1]=30 and vd_right_all[1]=30
# only changes the IDM desired speed; it does NOT move left[1] (s≈51 m) and
# right[1] (s≈53 m) away from the truck (s≈50 m) at t=0.
# Fix: run total_update_emergency() for _N_PREADVANCE steps so that the two
# fast vehicles (vd=30 m/s) physically clear the truck's flanks before the
# DRIFT warm-up begins and before the main loop reads their positions.
_N_PREADVANCE = 20   # 2 s at dt=0.1 s; fast vehicles move ≈ 60 m away
print(f"Pre-advancing surroundings {_N_PREADVANCE} steps "
      f"({_N_PREADVANCE * dt:.1f} s) to clear exclusion zones ...")
for _pre in range(_N_PREADVANCE):
    surroundings.total_update_emergency(_pre)
print("  Pre-advance complete.")

# Warm-up: seed merger from truck position, then run DRIFT for 5 s so risk
# accumulates in the centre-lane merge zone before t=0.
print("DRIFT warm-up (5 s)...")
vl0, vc0, vr0 = surroundings.get_vehicles_states()

# Seed merger at truck_s + MERGER_TRUCK_S_OFFSET using initial truck state.
_init_truck_state = [vl0, vc0, vr0][_AGENT_TRUCK_LANE][_AGENT_TRUCK_IDX]
merger.update(_init_truck_state)
print(f"[MERGER INIT] truck_s={float(_init_truck_state[0]):.1f} m  "
      f"offset={MERGER_TRUCK_S_OFFSET:.1f} m  "
      f"→ merger at s={merger.state[0]:.1f} m  (right lane, vx={MERGER_VX} m/s)")
print(f"[LEFT_FAST_CAR INIT] left lane  s={left_fast_car.state[0]:.1f} m  "
      f"vd={LEFT_FAST_CAR_VD} m/s")

vd_init = convert_to_drift(vl0, vc0, vr0,
                           truck_set=set(TRUCK_DESIGNATIONS.keys()))
vd_init.append(merger.to_drift_vehicle(vid=998))        # merger always in DRIFT
vd_init.append(left_fast_car.to_drift_vehicle(vid=997)) # fast car always in DRIFT

_psi0 = X0_g[2]
ego_init = drift_create_vehicle(
    vid=0,
    x=X0_g[0], y=X0_g[1],
    vx=X0[0] * math.cos(_psi0) - X0[1] * math.sin(_psi0),
    vy=X0[0] * math.sin(_psi0) + X0[1] * math.cos(_psi0),
    vclass='car')
ego_init['heading'] = _psi0

drift.warmup(vd_init, ego_init, dt=dt, duration=5.0, substeps=3)
print()

# ===========================================================================
# STEP 1 — SCENARIO VERIFICATION  (printed once at startup)
# ===========================================================================

_lane_names = {0: 'left', 1: 'centre', 2: 'right'}
_truck_vd   = TRUCK_DESIGNATIONS[(_AGENT_TRUCK_LANE, _AGENT_TRUCK_IDX)]
_truck_arr0 = [vl0, vc0, vr0][_AGENT_TRUCK_LANE]
_truck_s    = float(_truck_arr0[_AGENT_TRUCK_IDX][0])
_ego_s      = float(X0[3])

print("=" * 70)
print("SCENARIO GEOMETRY VERIFICATION")
print("=" * 70)

# ── Actors ────────────────────────────────────────────────────────────────
print(f"  EGO          : left lane   s={_ego_s:.1f} m  vx={X0[0]:.1f} m/s")
print(f"  TRUCK        : {_lane_names[_AGENT_TRUCK_LANE]} lane  idx={_AGENT_TRUCK_IDX}"
      f"  s={_truck_s:.1f} m  vd={_truck_vd} m/s")
print(f"  MERGER       : right lane  s={merger.state[0]:.1f} m"
      f"  (offset={MERGER_TRUCK_S_OFFSET:.0f} m ahead of truck)  vx={MERGER_VX} m/s")
print(f"  LEFT_FAST_CAR: left lane   s={left_fast_car.state[0]:.1f} m  "
      f"vd={LEFT_FAST_CAR_VD:.1f} m/s  (occluded={left_fast_car.occluded})")

# ── Constraint checks ─────────────────────────────────────────────────────
_violations = []

# Check 1: merger is ahead of truck (s_offset check)
_mg_s = float(merger.state[0])
_expected_mg_s = _truck_s + MERGER_TRUCK_S_OFFSET
print(f"\n  Constraint 1 — merger ahead of truck "
      f"(expected s≈{_expected_mg_s:.1f} m):")
if abs(_mg_s - _expected_mg_s) < 3.0:
    print(f"    OK  merger at s={_mg_s:.1f} m")
else:
    _msg = f"    WARN  merger at s={_mg_s:.1f} m  (expected ~{_expected_mg_s:.1f} m)"
    print(_msg)
    _violations.append(_msg)

# Check 2: left fast car is ahead of ego
print(f"\n  Constraint 2 — left fast car ahead of ego "
      f"(s_fc={left_fast_car.state[0]:.1f} m > ego s={_ego_s:.1f} m):")
if left_fast_car.state[0] > _ego_s:
    print(f"    OK  fast car s={left_fast_car.state[0]:.1f} m  "
          f"(+{left_fast_car.state[0]-_ego_s:.1f} m ahead)")
else:
    _msg = f"    WARN  fast car s={left_fast_car.state[0]:.1f} m behind ego"
    print(_msg)
    _violations.append(_msg)

# Check 3: (placeholder for truck-side exclusion zone)
_ego_front_end = _ego_s + EGO_FRONT_CLEAR_M  # kept for compatibility
print(f"\n  Constraint 3 — truck-side exclusion zone "
      f"(±{TRUCK_SIDE_EXCL_M:.0f} m of truck s={_truck_s:.1f} m, lanes 0 & 2):")
_excl_lo, _excl_hi = _truck_s - TRUCK_SIDE_EXCL_M, _truck_s + TRUCK_SIDE_EXCL_M
for _li, _lr, _lname, _vd_arr in [
    (0, vl0, 'left',  surroundings.vd_left_all),
    (2, vr0, 'right', surroundings.vd_right_all),
]:
    for _vi, _veh in enumerate(_lr):
        _vs = float(_veh[0])
        if _excl_lo < _vs < _excl_hi:
            _vd = _vd_arr[_vi]
            _ok = _vd >= 20.0
            _tag = "OK (fast, will clear)" if _ok else "WARN"
            _msg = (f"    {_tag}  {_lname}[{_vi}] at s={_vs:.1f} m"
                    f"  (Δ={_vs-_truck_s:+.1f} m)  vd={_vd:.1f} m/s")
            print(_msg)
            if not _ok:
                _violations.append(_msg)

# Check 4: sight-line shadow (informational)
_shadow_init = compute_truck_shadow(X0_g[0], X0_g[1],
                                    _truck_arr0[_AGENT_TRUCK_IDX])
if _shadow_init is not None:
    _mg_in_sl = bool(MplPath(_shadow_init).contains_point([merger.x, merger.y]))
else:
    _mg_in_sl = False
print(f"\n  Constraint 4 — ego sight-line shadow covers merger: {_mg_in_sl}")
if not _mg_in_sl:
    print("    (DRIFT Q_occlusion cone is typically wider than the geometric shadow)")

# ── Summary ───────────────────────────────────────────────────────────────
print()
if _violations:
    print(f"  *** {len(_violations)} geometry violation(s) detected — "
          f"review vd settings or vehicle positions ***")
else:
    print("  All geometry constraints satisfied.")

_drift_trucks = [v for v in vd_init if v.get('class') == 'truck']
print(f"  DRIFT trucks: {len(_drift_trucks)} tagged vclass='truck'")
print("=" * 70)
print()

# ===========================================================================
# METRIC RECORDS
# ===========================================================================

risk_at_ego_list = []

# DREAM
dream_s          = []   # Frenet progress [m]
dream_vx         = []
dream_vy         = []
dream_acc        = []
dream_s_obs      = []   # min spacing to surrounding vehicles
dream_dist_agent = []   # distance to OccludedMerger

# IDEAM
ideam_s          = []
ideam_vx         = []
ideam_vy         = []
ideam_acc        = []
ideam_s_obs      = []
ideam_dist_agent = []   # distance to OccludedMerger

# Phantom occlusion flag per step
agent_occluded_record = []

# Merger LC tracking
_merger_triggered        = False  # gate: set once on true left→centre transition
_prev_path_di_i          = None   # IDEAM desired-path index from previous step
_prev_path_now_i         = None   # IDEAM actual lane from previous step
_ideam_first_lcc_step    = None   # step when IDEAM first commands centre lane
_merger_ttc_at_trigger   = None   # TTC (IDEAM-to-merger) at LC trigger moment

# ===========================================================================
# HELPER: plot risk overlay (same as emergency_test_prideam.py)
# ===========================================================================

def plot_risk_overlay(ax):
    """Overlay DRIFT risk field on current axis (using module-level risk_field)."""
    R_sm = _gf(risk_field, sigma=0.8)
    R_sm = np.clip(R_sm, 0, RISK_VMAX)
    if True:
        cf = ax.contourf(cfg.X, cfg.Y, R_sm,
                         levels=RISK_LEVELS, cmap=RISK_CMAP,
                         alpha=RISK_ALPHA, vmin=0, vmax=RISK_VMAX,
                         zorder=1, extend='max')
    ax.contour(cfg.X, cfg.Y, R_sm,
               levels=np.linspace(0.2, RISK_VMAX, 8),
               colors='darkred', linewidths=0.5, alpha=0.4, zorder=1)
    return cf


# ===========================================================================
# HELPER: draw one scenario panel
# ===========================================================================

def draw_panel(ax, ego_global, ego_state, vehicle_left, vehicle_centre,
               vehicle_right, x_range, y_range, title,
               horizon=None, risk_f=None, risk_val=None,
               show_merger=True):
    """Draw DREAM or IDEAM panel with road, vehicles, ego, and optional risk overlay."""
    plt.sca(ax)
    ax.cla()

    # Road geometry
    plot_env()

    # Risk overlay (DREAM panel only)
    cf = None
    if risk_f is not None:
        R_sm = _gf(risk_f, sigma=0.8)
        R_sm = np.clip(R_sm, 0, RISK_VMAX)
        cf = ax.contourf(cfg.X, cfg.Y, R_sm,
                         levels=RISK_LEVELS, cmap=RISK_CMAP,
                         alpha=RISK_ALPHA, vmin=0, vmax=RISK_VMAX,
                         zorder=1, extend='max')
        ax.contour(cfg.X, cfg.Y, R_sm,
                   levels=np.linspace(0.2, RISK_VMAX, 8),
                   colors='darkred', linewidths=0.5, alpha=0.4, zorder=1)

    # Horizon
    if horizon is not None and len(horizon) > 0:
        h = np.asarray(horizon)
        if h.ndim == 2 and h.shape[1] >= 2:
            ax.plot(h[:, 0], h[:, 1], color='#00BCD4', lw=1.8, ls='--', zorder=7)
            ax.scatter(h[:, 0], h[:, 1], color='#00BCD4', s=6, zorder=7)

    # Surrounding vehicles
    for li, (lane_arr, pr, xr, yr, sr) in enumerate([
        (vehicle_left,   path1c, x1c, y1c, samples1c),
        (vehicle_centre, path2c, x2c, y2c, samples2c),
        (vehicle_right,  path3c, x3c, y3c, samples3c),
    ]):
        for vi in range(len(lane_arr)):
            is_truck = (li, vi) in TRUCK_DESIGNATIONS
            veh = lane_arr[vi]
            in_view = (x_range[0] <= veh[3] <= x_range[1] and
                       y_range[0] <= veh[4] <= y_range[1])
            if not in_view:
                continue
            if is_truck:
                shadow_p = compute_truck_shadow(ego_global[0], ego_global[1], veh)
                draw_shadow_polygon(ax, shadow_p)
                draw_vehicle_rect(ax, veh[3], veh[4], veh[5],
                                  TRUCK_LENGTH, TRUCK_WIDTH, TRUCK_COLOR,
                                  edgecolor='darkred', lw=1.2, zorder=5)
                ax.text(veh[3] - 2, veh[4] + 2.5,
                        f"Truck {veh[6]:.1f} m/s",
                        rotation=np.rad2deg(veh[5]),
                        c='darkred', fontsize=5, style='oblique',
                        fontweight='bold')
            else:
                draw_vehicle_rect(ax, veh[3], veh[4], veh[5],
                                  CAR_LENGTH, CAR_WIDTH, SURROUND_COLOR,
                                  edgecolor='black', lw=0.6, zorder=4)
            if in_view:
                sx, ey_, _ = find_frenet_coord(pr, xr, yr, sr,
                                               [veh[3], veh[4], veh[5]])
                tx, ty = pr.get_cartesian_coords(sx - 4.75, ey_ - 1.0)
                ax.text(tx, ty, f"{veh[6]:.1f}", rotation=np.rad2deg(veh[5]),
                        c='k', fontsize=4, style='oblique')

    # OccludedMerger — ghost (dashed purple) while in right lane, solid magenta
    # once it has substantially entered the centre lane (lc_progress >= 0.5).
    if show_merger:
        _mx, _my = merger.x, merger.y
        _m_in_view = (x_range[0] <= _mx <= x_range[1] and
                      y_range[0] <= _my <= y_range[1])
        if _m_in_view:
            if merger.is_occluded():
                draw_vehicle_rect(ax, _mx, _my, merger.psi,
                                  CAR_LENGTH, CAR_WIDTH, AGENT_OCC_COLOR,
                                  edgecolor=AGENT_OCC_COLOR, lw=1.5, zorder=5,
                                  alpha=0.40, linestyle='--')
                ax.text(_mx + 1, _my + 1.2, "?",
                        c=AGENT_OCC_COLOR, fontsize=8, fontweight='bold', zorder=6)
            else:
                draw_vehicle_rect(ax, _mx, _my, merger.psi,
                                  CAR_LENGTH, CAR_WIDTH, AGENT_VIS_COLOR,
                                  edgecolor='darkred', lw=1.2, zorder=5)
                _lc_pct = int(merger.lc_progress * 100)
                ax.text(_mx + 1, _my + 1.2,
                        f"LC {_lc_pct}%",
                        c='darkred', fontsize=5, style='oblique',
                        fontweight='bold', zorder=6)

    # Ego vehicle
    facecolor = EGO_DREAM_COLOR if risk_f is not None else EGO_IDEAM_COLOR
    draw_vehicle_rect(ax, ego_global[0], ego_global[1], ego_global[2],
                      CAR_LENGTH, CAR_WIDTH, facecolor,
                      edgecolor='navy', lw=1.0, zorder=6)
    try:
        ego_sp = find_frenet_coord(path1c, x1c, y1c, samples1c, ego_global)
        _ref_path, _ref_s = path1c, ego_sp[0]
    except Exception:
        ego_sp = find_frenet_coord(path2c, x2c, y2c, samples2c, ego_global)
        _ref_path, _ref_s = path2c, ego_sp[0]
    tx, ty = _ref_path.get_cartesian_coords(_ref_s - 5.1, ego_sp[1] - 1.0)
    ax.text(tx, ty, f"{ego_state[0]:.1f} m/s",
            rotation=np.rad2deg(_ref_path.get_theta_r(_ref_s)),
            c='black', fontsize=5, style='oblique')

    # Risk value badge
    if risk_val is not None:
        col = 'red' if risk_val > 1.5 else ('orange' if risk_val > 0.5 else 'green')
        ax.text(0.985, 0.965, f"R={risk_val:.2f}",
                transform=ax.transAxes, ha='right', va='top',
                c=col, fontsize=7, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    return cf


# ===========================================================================
# HELPER: build MPC horizon for visualization
# ===========================================================================

def build_horizon(X0_seed, X0_g_seed, ctrl_a, ctrl_d,
                  path_d, sample, x_list, y_list):
    X_v = list(X0_seed)
    Xg_v = list(X0_g_seed)
    horizon = [list(Xg_v)]
    n = max(0, len(ctrl_a) - 1) if ctrl_a is not None else 0
    for k in range(n):
        u = [ctrl_a[k+1], ctrl_d[k+1]] if (ctrl_a and ctrl_d) else [0.0, 0.0]
        X_v, Xg_v, _, _ = dynamics.propagate(
            X_v, u, dt, Xg_v, path_d, sample, x_list, y_list, boundary)
        horizon.append(list(Xg_v))
    return np.array(horizon)


# ===========================================================================
# BASELINE PARAMETERS (restored each step)
# ===========================================================================

_d0_base  = utils.d0
_Th_base  = utils.Th
_al_base  = controller.mpc.a_l
_bl_base  = controller.mpc.b_l
_P_base   = controller.mpc.P.copy()

_proactive_lc_cooldown = 0

# ===========================================================================
# MAIN SIMULATION LOOP
# ===========================================================================

print(f"Running uncertainty test: {N_t} steps, dt={dt}s")
print()

risk_field = None          # initialised on first DRIFT step
current_colorbar = None    # track colorbar to avoid accumulation

# Pre-create figure for gcf/clf reuse (same pattern as emergency_test_prideam.py)
plt.figure(figsize=(12, 10))

for i in range(N_t):
    bar.next()

    # ──────────────────────────────────────────────────────────────────────
    # 1. GET CURRENT SURROUNDING VEHICLE STATES
    # ──────────────────────────────────────────────────────────────────────
    vehicle_left, vehicle_centre, vehicle_right = surroundings.get_vehicles_states()

    # Clamp truck speeds to their designated desired speed every step.
    # get_vehicles_states() returns REFERENCES to internal arrays, so this
    # also fixes the IDM input for the next surroundings.total_update_emergency().
    for (_tk_l, _tk_v), _tk_vd in TRUCK_DESIGNATIONS.items():
        _tk_arrs = [vehicle_left, vehicle_centre, vehicle_right]
        if len(_tk_arrs[_tk_l]) > _tk_v:
            _tk_arrs[_tk_l][_tk_v][6] = min(float(_tk_arrs[_tk_l][_tk_v][6]),
                                             float(_tk_vd))

    # ──────────────────────────────────────────────────────────────────────
    # 2. UPDATE SYNTHETIC AGENTS (left_fast_car + merger, every step)
    # ──────────────────────────────────────────────────────────────────────
    # 2a. Get truck state for ghost-tracking and occlusion check.
    _tk_arrs_cur  = [vehicle_left, vehicle_centre, vehicle_right]
    _truck_state_i = _tk_arrs_cur[_AGENT_TRUCK_LANE][_AGENT_TRUCK_IDX]

    # 2b. Advance the left-lane fast car (IDM on path1c, revealed when shadow clears).
    left_fast_car.update(vehicle_left)
    left_fast_car.check_occlusion(X0_g_ideam[0], X0_g_ideam[1], _truck_state_i)
    if not left_fast_car.revealed and not left_fast_car.occluded:
        left_fast_car.revealed    = True
        left_fast_car.reveal_step = i
        print(f"[FAST CAR REVEALED] Step {i} (t={i*dt:.1f}s) geometric trigger")
    elif not left_fast_car.revealed and i >= LEFT_FAST_CAR_REVEAL_FALLBACK:
        left_fast_car.revealed    = True
        left_fast_car.reveal_step = i
        print(f"[FAST CAR REVEALED] Step {i} (t={i*dt:.1f}s) fallback trigger")

    # 2c. Merger LC trigger — fires only on a TRUE left→centre intent.
    # Condition: IDEAM was physically in left lane (_prev_path_now_i == 0)
    #            AND commanded centre lane (_prev_path_di_i == 1) last step.
    _ideam_lc_intent = (_prev_path_now_i == 0 and _prev_path_di_i == 1)
    if not _merger_triggered and _ideam_lc_intent:
        _merger_triggered = True
        merger.trigger_lc(i)
        _mg_pos  = np.array([merger.x, merger.y])
        _id_pos  = np.array([X0_g_ideam[0], X0_g_ideam[1]])
        _mg_dist = float(np.linalg.norm(_mg_pos - _id_pos))
        _rel_spd = max(0.5, abs(MERGER_VX - float(X0_ideam[0])))
        _merger_ttc_at_trigger = _mg_dist / _rel_spd
        print(f"[MERGER LC TRIGGERED] Step {i} (t={i*dt:.1f}s)  "
              f"trigger=ideam_lc_intent (was_lane=0 → cmd_lane=1)")
        print(f"  merger: s={merger.state[0]:.1f}m  "
              f"pos=({merger.x:.1f},{merger.y:.1f})  vx={MERGER_VX:.1f} m/s")
        print(f"  IDEAM:  pos=({X0_g_ideam[0]:.1f},{X0_g_ideam[1]:.1f})  "
              f"vx={float(X0_ideam[0]):.1f} m/s")
        print(f"  dist_to_IDEAM={_mg_dist:.1f}m  TTC≈{_merger_ttc_at_trigger:.1f}s")
    elif not _merger_triggered and i >= MERGER_LC_FALLBACK:
        _merger_triggered = True
        merger.trigger_lc(i)
        print(f"[MERGER LC TRIGGERED] Step {i} (t={i*dt:.1f}s)  trigger=fallback")

    merger.update(_truck_state_i)   # ghost-track truck position
    agent_occluded_record.append(merger.is_occluded())

    # ──────────────────────────────────────────────────────────────────────
    # 3. BUILD DRIFT VEHICLE LIST (merger + left_fast_car ALWAYS included)
    # ──────────────────────────────────────────────────────────────────────
    vehicles_drift = convert_to_drift(
        vehicle_left, vehicle_centre, vehicle_right,
        truck_set=set(TRUCK_DESIGNATIONS.keys()))
    vehicles_drift.append(merger.to_drift_vehicle(vid=998))           # always in DRIFT
    vehicles_drift.append(left_fast_car.to_drift_vehicle(vid=997))    # always in DRIFT

    # ──────────────────────────────────────────────────────────────────────
    # 4. BUILD MPC VEHICLE LISTS
    #    left_fast_car injected into vl_mpc only after reveal.
    #    Merger: excluded while in right lane (lc_progress < 0.5); injected
    #    into vc_mpc once substantially in centre lane (lc_progress >= 0.5).
    # ──────────────────────────────────────────────────────────────────────
    vl_mpc = vehicle_left.copy()
    vc_mpc = vehicle_centre.copy()
    vr_mpc = vehicle_right.copy()

    # Inject fast car into left-lane array after reveal so IDEAM/DREAM react.
    if left_fast_car.revealed:
        vl_mpc = np.vstack([vl_mpc, left_fast_car.to_ideam_row()])

    if merger.in_centre_lane():
        # Inject merger into centre-lane array with path2c Frenet coordinates.
        try:
            _ms, _mey, _mepsi = find_frenet_coord(
                path2c, x2c, y2c, samples2c,
                [merger.x, merger.y, merger.psi])
            _mg_row = np.array([_ms, _mey, _mepsi,
                                merger.x, merger.y,
                                merger.psi, MERGER_VX, 0.0])
            vc_mpc = np.vstack([vc_mpc, _mg_row])
        except Exception:
            pass   # skip if Frenet projection fails

    # ──────────────────────────────────────────────────────────────────────
    # 5. STEP DRIFT
    # ──────────────────────────────────────────────────────────────────────
    ego_psi = X0_g[2]
    ego_drift = drift_create_vehicle(
        vid=0, x=X0_g[0], y=X0_g[1],
        vx=X0[0] * math.cos(ego_psi) - X0[1] * math.sin(ego_psi),
        vy=X0[0] * math.sin(ego_psi) + X0[1] * math.cos(ego_psi),
        vclass='car')
    ego_drift['heading'] = ego_psi

    risk_field = drift.step(vehicles_drift, ego_drift, dt=dt, substeps=3)
    risk_at_ego = drift.get_risk_cartesian(X0_g[0], X0_g[1])
    risk_at_ego_list.append(risk_at_ego)

    # ──────────────────────────────────────────────────────────────────────
    # 6. TRUCK RISK BOOST + DYNAMIC MPC WEIGHT
    # NOTE (architectural limitation): the local `risk_field` and `risk_at_ego`
    # variables are boosted here for visualization and the decision-veto check
    # (evaluate_decision_risk reads risk_at_ego via controller.drift.get_risk).
    # However solve_with_risk() (Section 9) internally queries controller.drift,
    # which holds the raw un-boosted DRIFT field from drift.step().
    # Consequence: MPC trajectory cost uses un-boosted risk; only the veto gate
    # and visualization use the boosted value.  To fully align them the boosted
    # field would need to be written back to drift.risk_field before Section 9.
    # ──────────────────────────────────────────────────────────────────────
    _ego_truck_dist   = float('inf')
    _nearest_truck_key = None
    if TRUCK_DESIGNATIONS:
        _lane_arrays = [vl_mpc, vc_mpc, vr_mpc]
        _boost_field = np.ones_like(risk_field)
        _ego_boost   = 1.0
        for (_tk_l, _tk_v) in TRUCK_DESIGNATIONS.keys():
            _tv = _lane_arrays[_tk_l][_tk_v]
            _tx, _ty = _tv[3], _tv[4]
            _d = float(np.sqrt((_tx - X0_g[0])**2 + (_ty - X0_g[1])**2))
            if _d < _ego_truck_dist:
                _ego_truck_dist    = _d
                _nearest_truck_key = (_tk_l, _tk_v)
            _dsq = (cfg.X - _tx)**2 + (cfg.Y - _ty)**2
            _boost_field += TRUCK_RISK_BOOST * np.exp(-_dsq / (2*TRUCK_RISK_SIGMA**2))
            _ego_boost   += TRUCK_RISK_BOOST * np.exp(-_d**2 / (2*TRUCK_RISK_SIGMA**2))
        risk_field  = risk_field * _boost_field
        risk_at_ego = risk_at_ego * _ego_boost

    _prox = max(0.0, 1.0 - _ego_truck_dist / TRUCK_INFLUENCE_DIST)
    controller.weights.mpc_cost = (
        config_integration.mpc_risk_weight * (
            1.0 + (TRUCK_WEIGHT_SCALE - 1.0) * _prox))

    # ──────────────────────────────────────────────────────────────────────
    # 7. DREAM PATH DECISION LOGIC
    # ──────────────────────────────────────────────────────────────────────
    path_now  = judge_current_position(
        X0_g[0:2], x_bound, y_bound, path_bound, path_bound_sample)
    path_ego  = surroundings.get_path_ego(path_now)

    # Asymmetric constraint scaling
    _truck_in_ego_lane = (_nearest_truck_key is not None and
                          _nearest_truck_key[0] == path_now)
    if _truck_in_ego_lane:
        utils.d0 = _d0_base + TRUCK_LONG_EXTRA * _prox
        utils.Th = _Th_base + TRUCK_TH_EXTRA  * _prox
    else:
        utils.d0 = _d0_base
        utils.Th = _Th_base
    controller.mpc.a_l = _al_base * (1.0 + TRUCK_AL_SCALE * _prox)
    controller.mpc.b_l = _bl_base * (1.0 + TRUCK_BL_SCALE * _prox)
    _relax = max(1.0, 1.0 + TRUCK_CENTER_RELAX * _prox)
    controller.mpc.P  = _P_base / _relax

    # Lateral squeeze detection
    _left_adj  = float('inf')
    _right_adj = float('inf')
    _ego_s     = X0[3]
    if path_now == 1:
        for _v in vl_mpc:
            _left_adj  = min(_left_adj,  abs(_v[0] - _ego_s))
        for _v in vr_mpc:
            _right_adj = min(_right_adj, abs(_v[0] - _ego_s))
    elif path_now == 0:
        for _v in vc_mpc:
            _right_adj = min(_right_adj, abs(_v[0] - _ego_s))
    elif path_now == 2:
        for _v in vc_mpc:
            _left_adj  = min(_left_adj,  abs(_v[0] - _ego_s))
    _squeezed            = (_left_adj < SQUEEZE_LON_DIST and
                            _right_adj < SQUEEZE_LON_DIST and
                            _ego_s > 35.0)
    _squeeze_escape      = False

    start_group_str = {0: "L1", 1: "C1", 2: "R1"}[path_now]

    if i == 0:
        ovx, ovy, owz, oS, oey, oepsi = clac_last_X(
            oa, od, mpc_controller.T, path_ego, dt, 6, X0, X0_g)
        last_X = [ovx, ovy, owz, oS, oey, oepsi]

    all_info   = utils.get_alllane_lf(
        path_ego, X0_g, path_now, vl_mpc, vc_mpc, vr_mpc)
    group_dict, ego_group = utils.formulate_gap_group(
        path_now, last_X, all_info, vl_mpc, vc_mpc, vr_mpc)

    _t_dec = time.time()
    desired_group = decision_maker.decision_making(group_dict, start_group_str)
    decision_dur  = time.time() - _t_dec

    path_d, path_dindex, C_label, sample, x_list, y_list, X0 = Decision_info(
        X0, X0_g, path_center, sample_center, x_center, y_center,
        boundary, desired_group, path_ego, path_now)

    C_label_additive = utils.inquire_C_state(C_label, desired_group)
    if C_label_additive == "Probe":
        path_d, path_dindex, C_label_virtual = path_ego, path_now, "K"
        _, xc, yc, samplesc = get_path_info(path_dindex)
        X0 = repropagate(path_d, samplesc, xc, yc, X0_g, X0)
    else:
        C_label_virtual = C_label

    # Decision veto
    if config_integration.enable_decision_veto and C_label != "K":
        ego_state_veto = list(X0)
        _rs, _allow, _ = controller.evaluate_decision_risk(
            ego_state_veto, path_now, path_dindex)
        if not _allow:
            path_d, path_dindex, C_label_virtual = path_ego, path_now, "K"
            _, xc, yc, samplesc = get_path_info(path_dindex)
            X0 = repropagate(path_d, samplesc, xc, yc, X0_g, X0)

    # Proactive LC away from truck
    if _proactive_lc_cooldown > 0:
        _proactive_lc_cooldown -= 1

    if (TRUCK_DESIGNATIONS and
            _ego_truck_dist < TRUCK_PROACTIVE_DIST and
            risk_at_ego > TRUCK_PROACTIVE_RISK and
            C_label_virtual == "K" and _proactive_lc_cooldown == 0):
        _tk_lane = _nearest_truck_key[0]
        if path_now == _tk_lane:
            # Ego is in same lane as truck — move away to overtake.
            # Truck in centre (1) → go LEFT (0), NOT right (merger is there).
            _alt = 0 if _tk_lane == 1 else (1 if _tk_lane != 1 else 0)
        elif path_now == 1:
            _alt = 2 if _tk_lane == 0 else 0
        else:
            _alt = path_now
        if _alt != path_now:
            path_d, path_dindex = path_center[_alt], _alt
            C_label_virtual = "L" if _alt < path_now else "R"
            _, xc, yc, samplesc = get_path_info(path_dindex)
            X0 = repropagate(path_d, samplesc, xc, yc, X0_g, X0)
            _proactive_lc_cooldown = 30

    # Squeeze escape
    if (_squeezed and C_label_virtual == "K" and _proactive_lc_cooldown == 0):
        _esc = -1
        if path_now == 1:
            _esc = 0 if _left_adj >= _right_adj else 2
        if _esc != -1 and _esc != path_now:
            path_d, path_dindex = path_center[_esc], _esc
            C_label_virtual = "L" if _esc < path_now else "R"
            _, xc, yc, samplesc = get_path_info(path_dindex)
            X0 = repropagate(path_d, samplesc, xc, yc, X0_g, X0)
            controller.mpc.b_l = _bl_base
            _proactive_lc_cooldown = 30
            _squeeze_escape = True
        elif path_now in (0, 2):
            _squeeze_escape = True
        if _squeeze_escape:
            controller.weights.mpc_cost = 0.0
    if _squeezed and not _squeeze_escape:
        controller.weights.mpc_cost = 0.0

    path_desired.append(path_d)
    if path_changed != path_dindex:
        controller.get_path_curvature(path=path_d)
        oS, oey = path_to_path_proj(oS, oey, path_changed, path_dindex)
        last_X = [ovx, ovy, owz, oS, oey, oepsi]
    path_changed = path_dindex

    # ──────────────────────────────────────────────────────────────────────
    # 8. IDEAM PARALLEL SIMULATION
    # ──────────────────────────────────────────────────────────────────────
    ideam_panel_X0    = list(X0_ideam)
    ideam_panel_X0_g  = list(X0_g_ideam)
    ideam_panel_horizon = None

    try:
        path_now_i = judge_current_position(
            X0_g_ideam[0:2], x_bound, y_bound, path_bound, path_bound_sample)
        path_ego_i = surroundings.get_path_ego(path_now_i)
        sgs_i = {0: "L1", 1: "C1", 2: "R1"}[path_now_i]

        if i == 0:
            _ovx_i, _ovy_i, _owz_i, _oS_i, _oey_i, _oepsi_i = clac_last_X(
                oa_ideam_viz, od_ideam_viz,
                baseline_mpc_viz.T, path_ego_i, dt, 6, X0_ideam, X0_g_ideam)
            last_X_ideam_viz = [_ovx_i, _ovy_i, _owz_i, _oS_i, _oey_i, _oepsi_i]

        all_info_i = utils_ideam_viz.get_alllane_lf(
            path_ego_i, X0_g_ideam, path_now_i, vl_mpc, vc_mpc, vr_mpc)
        gd_i, ego_grp_i = utils_ideam_viz.formulate_gap_group(
            path_now_i, last_X_ideam_viz, all_info_i, vl_mpc, vc_mpc, vr_mpc)
        dg_i = decision_maker.decision_making(gd_i, sgs_i)

        path_d_i, path_di_i, Cl_i, samp_i, xl_i, yl_i, X0_ideam = Decision_info(
            X0_ideam, X0_g_ideam, path_center, sample_center,
            x_center, y_center, boundary, dg_i, path_ego_i, path_now_i)

        Cla_i = utils_ideam_viz.inquire_C_state(Cl_i, dg_i)
        if Cla_i == "Probe":
            path_d_i, path_di_i, Clv_i = path_ego_i, path_now_i, "K"
            _, xci, yci, sci = get_path_info(path_di_i)
            X0_ideam = repropagate(path_d_i, sci, xci, yci, X0_g_ideam, X0_ideam)
        else:
            Clv_i = Cl_i

        # Track first true left→centre LC intent (same condition as merger trigger)
        if _ideam_first_lcc_step is None and path_now_i == 0 and path_di_i == 1:
            _ideam_first_lcc_step = i
            print(f"[IDEAM LC-CENTRE] Step {i} (t={i*dt:.1f}s): "
                  f"IDEAM first commands centre lane  (path {path_now_i}→{path_di_i})")

        if (path_changed_ideam_viz != path_di_i and last_X_ideam_viz is not None):
            _oS_i, _oey_i = path_to_path_proj(
                last_X_ideam_viz[3], last_X_ideam_viz[4],
                path_changed_ideam_viz, path_di_i)
            last_X_ideam_viz = [last_X_ideam_viz[0], last_X_ideam_viz[1],
                                 last_X_ideam_viz[2], _oS_i, _oey_i,
                                 last_X_ideam_viz[5]]
        baseline_mpc_viz.get_path_curvature(path=path_d_i)
        path_changed_ideam_viz = path_di_i

        res_i = baseline_mpc_viz.iterative_linear_mpc_control(
            X0_ideam, oa_ideam_viz, od_ideam_viz, dt,
            None, None, Cl_i, X0_g_ideam, path_d_i, last_X_ideam_viz,
            path_now_i, ego_grp_i, path_ego_i, dg_i,
            vl_mpc, vc_mpc, vr_mpc, path_di_i, Cla_i, Clv_i)

        if res_i is not None:
            oa_i, od_i, _ovx_i, _ovy_i, _owz_i, _oS_i, _oey_i, _oepsi_i = res_i
            last_X_ideam_viz = [_ovx_i, _ovy_i, _owz_i, _oS_i, _oey_i, _oepsi_i]
            oa_ideam_viz, od_ideam_viz = oa_i, od_i
            X0_ideam, X0_g_ideam, _, _ = dynamics.propagate(
                list(X0_ideam), [oa_i[0], od_i[0]], dt,
                list(X0_g_ideam), path_d_i, samp_i, xl_i, yl_i, boundary)
            ideam_panel_X0   = list(X0_ideam)
            ideam_panel_X0_g = list(X0_g_ideam)
            ideam_panel_horizon = build_horizon(
                ideam_panel_X0, ideam_panel_X0_g, oa_i, od_i,
                path_d_i, samp_i, xl_i, yl_i)
            last_ideam_panel_horizon_viz = ideam_panel_horizon

    except Exception as e:
        ideam_panel_horizon = last_ideam_panel_horizon_viz
        if i % 50 == 0:
            print(f"[IDEAM] Frame {i}: {e}")

    # Store IDEAM's actual lane and desired path for merger LC trigger next step.
    # Both are needed to detect a true left→centre transition event.
    try:
        _prev_path_di_i  = path_di_i
        _prev_path_now_i = path_now_i
    except NameError:
        _prev_path_di_i  = None
        _prev_path_now_i = None

    # ──────────────────────────────────────────────────────────────────────
    # 9. DREAM MPC SOLVE WITH RISK
    # ──────────────────────────────────────────────────────────────────────
    _t_solve = time.time()
    oa, od, ovx, ovy, owz, oS, oey, oepsi = controller.solve_with_risk(
        X0, oa, od, dt, None, None, C_label, X0_g, path_d, last_X,
        path_now, ego_group, path_ego, desired_group,
        vl_mpc, vc_mpc, vr_mpc, path_dindex, C_label_additive, C_label_virtual)

    # Speed cap and squeeze floor
    if (_truck_in_ego_lane and C_label_virtual == "K" and
            not _squeeze_escape and
            _ego_truck_dist < TRUCK_INFLUENCE_DIST and
            X0[0] > TRUCK_SAFE_SPEED):
        _os = X0[0] - TRUCK_SAFE_SPEED
        oa = list(oa)
        oa[0] = min(oa[0], -min(2.0, _os * 0.5))

    if _squeezed and oa[0] < SQUEEZE_MIN_ACCEL:
        oa = list(oa)
        oa[0] = SQUEEZE_MIN_ACCEL

    last_X = [ovx, ovy, owz, oS, oey, oepsi]
    X0, X0_g, _, _ = dynamics.propagate(
        X0, [oa[0], od[0]], dt, X0_g, path_d, sample, x_list, y_list, boundary)

    # ──────────────────────────────────────────────────────────────────────
    # 10. METRICS
    # ──────────────────────────────────────────────────────────────────────
    # DREAM — use path1c (ego's starting lane) for consistent Frenet progress
    try:
        _s_d, _, _ = find_frenet_coord(path1c, x1c, y1c, samples1c, X0_g)
    except Exception:
        _s_d, _, _ = find_frenet_coord(path2c, x2c, y2c, samples2c, X0_g)
    dream_s.append(float(_s_d))
    dream_vx.append(float(X0[0]))
    dream_vy.append(float(X0[1]))
    dream_acc.append(float(oa[0]))

    _dream_rect = create_rectangle(X0_g[0], X0_g[1],
                                   mpc_controller.vehicle_length,
                                   mpc_controller.vehicle_width, X0_g[2])
    dream_s_obs.append(float(surroundings.S_obs_calc(_dream_rect)))

    _d2ag = math.sqrt((X0_g[0] - merger.x)**2 +
                      (X0_g[1] - merger.y)**2)
    dream_dist_agent.append(_d2ag)

    # IDEAM — same reference path as DREAM for fair comparison
    if ideam_panel_X0_g is not None:
        try:
            _s_i, _, _ = find_frenet_coord(path1c, x1c, y1c, samples1c,
                                           ideam_panel_X0_g)
        except Exception:
            _s_i, _, _ = find_frenet_coord(path2c, x2c, y2c, samples2c,
                                           ideam_panel_X0_g)
        ideam_s.append(float(_s_i))
        ideam_vx.append(float(ideam_panel_X0[0]))
        ideam_vy.append(float(ideam_panel_X0[1]))
        _oa_i = float(oa_ideam_viz[0] if hasattr(oa_ideam_viz, '__len__')
                      else oa_ideam_viz)
        ideam_acc.append(_oa_i)
        _i_rect = create_rectangle(ideam_panel_X0_g[0], ideam_panel_X0_g[1],
                                   mpc_controller.vehicle_length,
                                   mpc_controller.vehicle_width,
                                   ideam_panel_X0_g[2])
        ideam_s_obs.append(float(surroundings.S_obs_calc(_i_rect)))
        _d2ag_i = math.sqrt((ideam_panel_X0_g[0] - merger.x)**2 +
                             (ideam_panel_X0_g[1] - merger.y)**2)
        ideam_dist_agent.append(_d2ag_i)
    else:
        ideam_s.append(ideam_s[-1] if ideam_s else 20.0)
        ideam_vx.append(0.0)
        ideam_vy.append(0.0)
        ideam_acc.append(0.0)
        ideam_s_obs.append(100.0)
        ideam_dist_agent.append(float('nan'))

    # ──────────────────────────────────────────────────────────────────────
    # 11. UPDATE SURROUNDING VEHICLES
    # ──────────────────────────────────────────────────────────────────────
    surroundings.total_update_emergency(i)

    # ──────────────────────────────────────────────────────────────────────
    # 12. VISUALIZATION — every frame, gcf/clf pattern matching reference
    # ──────────────────────────────────────────────────────────────────────
    # Build DREAM horizon rollout for visualization
    X0_vis, X0_g_vis = list(X0), list(X0_g)
    X0_g_vis_list = [list(X0_g_vis)]
    for _k in range(len(oa) - 1):
        _u_vis = [oa[_k + 1], od[_k + 1]]
        X0_vis, X0_g_vis, _, _ = dynamics.propagate(
            X0_vis, _u_vis, dt, X0_g_vis, path_d, sample, x_list, y_list, boundary)
        X0_g_vis_list.append(list(X0_g_vis))
    X0_g_vis_list = np.array(X0_g_vis_list)

    fig = plt.gcf()
    fig.clf()

    # Each panel is centred on its OWN ego (panels diverge as simulation goes)
    _ix = ideam_panel_X0_g[0] if (ideam_panel_X0_g and len(ideam_panel_X0_g) >= 2) else X0_g[0]
    _iy = ideam_panel_X0_g[1] if (ideam_panel_X0_g and len(ideam_panel_X0_g) >= 2) else X0_g[1]
    x_range_i = [_ix - x_area, _ix + x_area]
    y_range_i = [_iy - y_area, _iy + y_area]
    x_range_d = [X0_g[0] - x_area, X0_g[0] + x_area]
    y_range_d = [X0_g[1] - y_area, X0_g[1] + y_area]

    ax_left  = fig.add_subplot(2, 1, 1)   # top    = IDEAM (baseline)
    ax_right = fig.add_subplot(2, 1, 2)   # bottom = DREAM (ours)

    # ── IDEAM panel (top) — no risk overlay ────────────────────────────
    draw_panel(ax_left,
               ego_global=ideam_panel_X0_g,
               ego_state=ideam_panel_X0,
               vehicle_left=vl_mpc,
               vehicle_centre=vc_mpc,
               vehicle_right=vr_mpc,
               x_range=x_range_i, y_range=y_range_i,
               title="IDEAM (baseline) — no risk awareness",
               horizon=ideam_panel_horizon,
               risk_f=None, risk_val=None,
               show_merger=True)

    # ── DREAM panel (bottom) — DRIFT risk overlay ───────────────────────
    contourf_obj = draw_panel(ax_right,
                              ego_global=X0_g,
                              ego_state=X0,
                              vehicle_left=vl_mpc,
                              vehicle_centre=vc_mpc,
                              vehicle_right=vr_mpc,
                              x_range=x_range_d, y_range=y_range_d,
                              title="DREAM (ours) — occlusion-aware",
                              horizon=X0_g_vis_list,
                              risk_f=risk_field,
                              risk_val=risk_at_ego,
                              show_merger=True)

    # Colorbar on DREAM panel only (matches emergency_test_prideam.py)
    if contourf_obj is not None:
        cbar = fig.colorbar(contourf_obj, ax=ax_right,
                            orientation='vertical', pad=0.02, fraction=0.035)
        cbar.set_label('Risk Level', fontsize=9, weight='bold')
        cbar.ax.tick_params(labelsize=8, colors='black')

    # Remove x-tick labels on top panel (vertical layout)
    ax_left.tick_params(labelbottom=False)

    plt.savefig(os.path.join(save_dir, "{}.png".format(i)), dpi=600)

bar.finish()
print()
print("Simulation complete.")
print(f"  DREAM final s: {dream_s[-1]:.1f} m  |  IDEAM final s: {ideam_s[-1]:.1f} m")

# Collision / near-collision events
_nc_dream = [(k, dream_dist_agent[k])
             for k in range(len(dream_dist_agent))
             if dream_dist_agent[k] < NEAR_COLLISION_DIST]
_nc_ideam = [(k, ideam_dist_agent[k])
             for k in range(len(ideam_dist_agent))
             if not math.isnan(ideam_dist_agent[k]) and
             ideam_dist_agent[k] < NEAR_COLLISION_DIST]
_col_dream = [x for x in _nc_dream if x[1] < COLLISION_DIST]
_col_ideam = [x for x in _nc_ideam if x[1] < COLLISION_DIST]

print(f"  DREAM near-collisions (<{NEAR_COLLISION_DIST}m): {len(_nc_dream)}  "
      f"| collisions (<{COLLISION_DIST}m): {len(_col_dream)}")
print(f"  IDEAM near-collisions (<{NEAR_COLLISION_DIST}m): {len(_nc_ideam)}  "
      f"| collisions (<{COLLISION_DIST}m): {len(_col_ideam)}")

# ===========================================================================
# METRICS PLOT
# ===========================================================================

_t_arr = np.arange(N_t) * dt
_reveal_t = (merger.lc_step * dt) if merger.lc_step is not None else None

with plt.style.context(["science", "no-latex"]):
    fig_m, axes_m = plt.subplots(3, 2, figsize=(11, 12), constrained_layout=True)
    fig_m.suptitle("DREAM vs IDEAM — Simultaneous Merge Conflict Scenario",
                   fontsize=13)

    _C  = {"DREAM": "C0", "IDEAM": "C1"}
    _LS = {"DREAM": "-",  "IDEAM": "--"}

    def _shade(ax):
        """Mark merger LC trigger with a vertical line."""
        if _reveal_t is not None:
            ax.axvline(_reveal_t, color='magenta', lw=1.2, ls='--',
                       label='Merger LC triggered')

    # ── (0,0) Progress s(t) ──────────────────────────────────────────────
    ax = axes_m[0, 0]
    ax.plot(_t_arr, dream_s, color=_C["DREAM"], ls=_LS["DREAM"], label="DREAM")
    ax.plot(_t_arr, ideam_s, color=_C["IDEAM"], ls=_LS["IDEAM"], label="IDEAM")
    _shade(ax)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("s [m]")
    ax.set_title("Progress s(t)")
    ax.legend(fontsize=8)

    # ── (0,1) Speed vx(t) ────────────────────────────────────────────────
    ax = axes_m[0, 1]
    ax.plot(_t_arr, dream_vx, color=_C["DREAM"], ls=_LS["DREAM"], label="DREAM")
    ax.plot(_t_arr, ideam_vx, color=_C["IDEAM"], ls=_LS["IDEAM"], label="IDEAM")
    _shade(ax)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("$v_x$ [m/s]")
    ax.set_title("Speed $v_x$(t)")
    ax.legend(fontsize=8)

    # ── (1,0) Distance to occluded agent ─────────────────────────────────
    ax = axes_m[1, 0]
    ax.plot(_t_arr, dream_dist_agent, color=_C["DREAM"], ls=_LS["DREAM"],
            label="DREAM-to-agent")
    ax.plot(_t_arr, ideam_dist_agent, color=_C["IDEAM"], ls=_LS["IDEAM"],
            label="IDEAM-to-agent")
    ax.axhline(NEAR_COLLISION_DIST, color='orange', lw=1.2, ls='--',
               label=f"Near-collision ({NEAR_COLLISION_DIST}m)")
    ax.axhline(COLLISION_DIST, color='red', lw=1.2, ls='--',
               label=f"Collision ({COLLISION_DIST}m)")
    _shade(ax)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("Distance [m]")
    ax.set_title("Distance to OccludedMerger")
    ax.legend(fontsize=7)
    ax.set_ylim(bottom=0)

    # ── (1,1) DRIFT risk at ego ───────────────────────────────────────────
    ax = axes_m[1, 1]
    _t_risk = _t_arr[:len(risk_at_ego_list)]
    ax.plot(_t_risk, risk_at_ego_list, color=_C["DREAM"], ls=_LS["DREAM"])
    ax.fill_between(_t_risk, risk_at_ego_list, alpha=0.25, color=_C["DREAM"])
    _shade(ax)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("R(ego)")
    ax.set_title("DRIFT Risk at DREAM Ego")

    # ── (2,0) Acceleration ax(t) ──────────────────────────────────────────
    ax = axes_m[2, 0]
    ax.plot(_t_arr, dream_acc, color=_C["DREAM"], ls=_LS["DREAM"], label="DREAM")
    ax.plot(_t_arr, ideam_acc, color=_C["IDEAM"], ls=_LS["IDEAM"], label="IDEAM")
    _shade(ax)
    ax.axhline(0, color='black', lw=0.5)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("$a_x$ [m/s²]")
    ax.set_title("Longitudinal Acceleration")
    ax.legend(fontsize=8)

    # ── (2,1) Min spacing S_o(t) ─────────────────────────────────────────
    ax = axes_m[2, 1]
    ax.plot(_t_arr, dream_s_obs, color=_C["DREAM"], ls=_LS["DREAM"], label="DREAM")
    ax.plot(_t_arr, ideam_s_obs, color=_C["IDEAM"], ls=_LS["IDEAM"], label="IDEAM")
    ax.axhline(2.0, color='red', lw=0.8, ls=':', label="Safety threshold")
    _shade(ax)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("$S_o$ [m]")
    ax.set_title("Min Spacing to Surrounding Vehicles")
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)

    plt.savefig(os.path.join(save_dir, "metrics_uncertainty.png"),
                dpi=300, bbox_inches='tight')
    plt.close(fig_m)

# Save numeric metrics
np.save(os.path.join(save_dir, "metrics_uncertainty.npy"), {
    "dream_s":               dream_s,
    "ideam_s":               ideam_s,
    "dream_vx":              dream_vx,
    "ideam_vx":              ideam_vx,
    "dream_dist_agent":      dream_dist_agent,
    "ideam_dist_agent":      ideam_dist_agent,
    "risk_at_ego":           risk_at_ego_list,
    "agent_occluded":        agent_occluded_record,
    "merger_lc_step":        merger.lc_step,
    "merger_ttc_at_trigger": _merger_ttc_at_trigger,
    "near_collision_dist":   NEAR_COLLISION_DIST,
    "collision_dist":        COLLISION_DIST,
})

print(f"\nFrames saved to: {save_dir}")
print(f"Metrics plot:    {save_dir}/metrics_uncertainty.png")
