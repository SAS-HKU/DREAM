"""
DREAM Emergency Test
======================
FUNCTIONAL INTEGRATION of DRIFT risk field with IDEAM control.

Based on: emergency_test_with_risk_viz.py
Added:
- DREAMController for risk-aware MPC
- MPC cost integration (risk penalty in objective)
- Decision veto (gate lane changes by risk)
- CBF modulation (scale safety margins by risk)
- Configurable integration levels via integration_config
"""

import argparse
import os
import sys
import time

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

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

# DRIFT imports
from config import Config as cfg
from pde_solver import PDESolver, compute_total_Q, compute_velocity_field, compute_diffusion_field
from pde_solver import create_vehicle as drift_create_vehicle
from Integration.drift_interface import DRIFTInterface
import matplotlib.patches as mpatches
from matplotlib.transforms import Affine2D
import scienceplots  # noqa: F401  鈥?registers "science" style with matplotlib

# PRIDEAM imports - FUNCTIONAL INTEGRATION
from Integration.prideam_controller import create_prideam_controller
from Integration.integration_config import get_preset


def _str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _parse_cli_args():
    parser = argparse.ArgumentParser(
        description="Run DREAM emergency highway scenario benchmark."
    )
    parser.add_argument(
        "--integration-mode",
        default=os.environ.get("DREAM_INTEGRATION_MODE", "conservative"),
        help="Integration preset name for PRIDEAM controller.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=int(os.environ.get("DREAM_STEPS", "400")),
        help="Number of simulation steps.",
    )
    parser.add_argument(
        "--save-dir",
        default=os.path.join(SCRIPT_DIR, "figsave_DREAM_v9"),
        help="Directory where rendered frames and metrics are saved.",
    )
    parser.add_argument(
        "--scenario-file",
        default=os.path.join(SCRIPT_DIR, "file_save", "120_100"),
        help="Scenario pickle used to initialize surrounding vehicles.",
    )
    parser.add_argument(
        "--save-dpi",
        type=int,
        default=int(os.environ.get("DREAM_DPI", "600")),
        help="DPI used when saving frame images.",
    )
    parser.add_argument(
        "--save-frames",
        type=_str2bool,
        default=_str2bool(os.environ.get("DREAM_SAVE_FRAMES", "1")),
        help="Whether to save per-step visualization frames.",
    )
    parser.add_argument(
        "--ffmpeg-fps",
        type=int,
        default=int(os.environ.get("DREAM_FFMPEG_FPS", "10")),
        help="FPS value shown in the ffmpeg post-processing hint.",
    )
    return parser.parse_args()


CLI_ARGS = _parse_cli_args()

# =========================================================================
# INTEGRATION CONFIGURATION
# =========================================================================
# Choose integration mode: "baseline", "balanced", "conservative", "permissive"
# "conservative" is required to see meaningful DREAM vs IDEAM differences:
#   decision_risk_threshold=1.0, mpc_risk_weight=1.0, cbf_alpha=0.8
INTEGRATION_MODE = CLI_ARGS.integration_mode

config_integration = get_preset(INTEGRATION_MODE)
config_integration.apply_mode()

# =========================================================================
# TRUCK-AWARE RISK TUNING  (DREAM-only 鈥?IDEAM baseline is unaffected)
# =========================================================================
# Risk field boost: Gaussian amplifier applied to the DRIFT field around
# the truck.  The internal PDE state is NOT modified; this post-process
# is used only for DREAM's MPC cost and visualization.
TRUCK_RISK_BOOST    = 2.5    # peak multiplier at truck centre
TRUCK_RISK_SIGMA    = 12.0   # m  鈥?spatial extent (Gaussian std-dev)

# Dynamic MPC weight: risk_weight grows from base 鈫?base*TRUCK_WEIGHT_SCALE
# as ego closes within TRUCK_INFLUENCE_DIST of the truck.
TRUCK_WEIGHT_SCALE  = 4.0    # maximum multiplier on mpc_risk_weight
TRUCK_INFLUENCE_DIST = 70.0  # m  鈥?distance at which ramping begins

# Proactive lane change: DREAM overrides keep-lane and commands a lane
# change away from the truck when both conditions hold simultaneously.
TRUCK_PROACTIVE_DIST = 55.0  # m  鈥?ego-truck distance trigger
TRUCK_PROACTIVE_RISK = 0.35  # boosted risk level trigger (low threshold
                              #   so behaviour fires reliably)

# Efficiency trade-off: DREAM enforces a speed cap when the truck is directly
# ahead in the SAME lane and the ego is keeping lane (not overtaking).
# The cap is lifted during lane-change so the ego can accelerate to clear the truck.
# 7 m/s was too aggressive 鈥?it decelerates ego below overtaking speed and risks
# rear-end from following vehicles; 10.5 m/s still shows a meaningful penalty
# vs IDEAM while keeping the ego in a safe speed bracket.
TRUCK_SAFE_SPEED    = 10.5  # m/s  鈥?same-lane following cap (lane-keep only)

# 鈹€鈹€ ASYMMETRIC CONSTRAINT SCALING (DREAM-only) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# When near a truck, DREAM dynamically scales MPC parameters so that:
#   (a) Longitudinal following distance is larger (occlusion = more danger ahead)
#   (b) Lateral clearance during LC overtake is wider (truck is physically bigger)
#   (c) Center-keeping is relaxed (ego is free to offset away from truck's shadow)
# All scalings ramp linearly with proximity in [0, 1] (0=far, 1=at truck).
# IDEAM baseline uses fixed parameters throughout 鈥?this is the key difference.
TRUCK_LONG_EXTRA    = 8.0   # m  鈥?extra d0 at full proximity  (5 鈫?13 m min gap)
TRUCK_TH_EXTRA      = 1.0   # s  鈥?extra Th at full proximity  (0.5 鈫?1.5 s headway)
TRUCK_AL_SCALE      = 0.5   # 鈥? 鈥?a_l multiplier scale        (1.5 鈫?2.25)
TRUCK_BL_SCALE      = 0.7   # 鈥? 鈥?b_l multiplier scale        (2.2 鈫?3.74)
TRUCK_CENTER_RELAX  = 4.0   # 鈥? 鈥?P divisor at full proximity (0.5 鈫?0.1)

# 鈹€鈹€ LATERAL SQUEEZE AVOIDANCE (DREAM-only) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# When vehicles in BOTH adjacent lanes are within SQUEEZE_LON_DIST of the ego
# (measured in Frenet s along their respective paths), the ego is in a lateral
# squeeze.  DREAM responds by:
#   1. Overriding C_label_virtual to LC toward the side with more room
#   2. Zeroing the MPC risk weight so the solver behaves like baseline IDEAM
#      (forward DRIFT risk should NOT prevent escape from lateral danger)
#   3. Enforcing a minimum post-solve acceleration so ego actually pulls away
SQUEEZE_LON_DIST    = 25.0  # m  鈥?s-gap to count an adjacent vehicle as "beside"
SQUEEZE_MIN_ACCEL   = 0.8   # m/s虏 鈥?acceleration floor during squeeze escape

print("=" * 70)
print(f"DREAM INTEGRATION MODE: {INTEGRATION_MODE.upper()}")
print("=" * 70)
print(f"  Decision Veto:    {config_integration.enable_decision_veto}")
print(f"  MPC Cost:         {config_integration.enable_mpc_cost} (weight={config_integration.mpc_risk_weight})")
print(f"  CBF Modulation:   {config_integration.enable_cbf_modulation} (alpha={config_integration.cbf_alpha})")
print(f"  Decision Thresh:  {config_integration.decision_risk_threshold}")
print("=" * 70)
print()

# =========================================================================
# RECTANGLE-BASED VEHICLE RENDERING (replaces car image rendering)
# =========================================================================

def draw_vehicle_rect(ax, x, y, yaw_rad, length, width, facecolor, edgecolor='black', lw=0.8, zorder=3, alpha=1.0, label=None):
    """Draw a rotated rectangle for a vehicle."""
    # Rectangle centered at (x, y) with rotation
    rect = mpatches.FancyBboxPatch(
        (-length / 2, -width / 2), length, width,
        boxstyle="round,pad=0.05",
        facecolor=facecolor, edgecolor=edgecolor,
        linewidth=lw, alpha=alpha, zorder=zorder
    )
    t = Affine2D().rotate(yaw_rad).translate(x, y) + ax.transData
    rect.set_transform(t)
    ax.add_patch(rect)
    return rect

# Vehicle dimensions
CAR_LENGTH = 3.5
CAR_WIDTH = 1.2
TRUCK_LENGTH = 12.0
TRUCK_WIDTH = 2.0

# Colors
EGO_COLOR = '#2196F3'       # Blue
SURROUND_COLOR = '#FFD600'  # Yellow
TRUCK_COLOR = '#FF6F00'     # Dark orange
SHADOW_COLOR = '#4A4A4A'    # Grey for occlusion shadow polygon

# Show all 5 vehicles per lane for dense traffic (IDEAM sees all of them)
RENDER_CARS_PER_LANE = 5
# Include all vehicles per lane in DRIFT risk field
DRIFT_CARS_PER_LANE = 5


def compute_truck_shadow(ego_x, ego_y, truck_state, shadow_length=50.0):
    """
    Compute sight-line occlusion shadow polygon from ego to truck.
    Same geometry as drift_pde_visualization.py compute_shadow().

    Returns shadow_polygon (Nx2 array) or None if truck is behind ego.
    """
    tx, ty, yaw = truck_state[3], truck_state[4], truck_state[5]
    dx = tx - ego_x
    dy = ty - ego_y
    dist = np.sqrt(dx**2 + dy**2)

    if dist < 3:
        return None

    L, W = TRUCK_LENGTH, TRUCK_WIDTH
    # Truck corners in local frame
    corners_local = np.array([
        [-L / 2, -W / 2],
        [ L / 2, -W / 2],
        [ L / 2,  W / 2],
        [-L / 2,  W / 2],
    ])
    cos_h, sin_h = np.cos(yaw), np.sin(yaw)
    rot = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
    corners = (rot @ corners_local.T).T + np.array([tx, ty])

    # Find outermost corners from ego's view
    angles = np.arctan2(corners[:, 1] - ego_y, corners[:, 0] - ego_x)
    left_idx = np.argmax(angles)
    right_idx = np.argmin(angles)
    left_corner = corners[left_idx]
    right_corner = corners[right_idx]

    # Extend rays from ego through outermost corners
    left_dir = left_corner - np.array([ego_x, ego_y])
    left_dir = left_dir / (np.linalg.norm(left_dir) + 1e-6)
    right_dir = right_corner - np.array([ego_x, ego_y])
    right_dir = right_dir / (np.linalg.norm(right_dir) + 1e-6)

    left_far = left_corner + left_dir * shadow_length
    right_far = right_corner + right_dir * shadow_length

    shadow_polygon = np.array([left_corner, left_far, right_far, right_corner])
    return shadow_polygon


def draw_shadow_polygon(ax, shadow_polygon, alpha=0.35):
    """Draw occlusion shadow polygon with dashed red edge (like drift_pde_visualization.py)."""
    if shadow_polygon is None:
        return
    patch = plt.Polygon(
        shadow_polygon,
        facecolor=SHADOW_COLOR, alpha=alpha,
        edgecolor='red', linewidth=1.8, linestyle='--', zorder=2
    )
    ax.add_patch(patch)

time_now = int(time.time())
dirpath = SCRIPT_DIR

save_dir = os.path.abspath(CLI_ARGS.save_dir)
os.makedirs(save_dir, exist_ok=True)
SAVE_DPI = CLI_ARGS.save_dpi
SAVE_FRAMES = CLI_ARGS.save_frames
FFMPEG_FPS = CLI_ARGS.ffmpeg_fps

# =========================================================================
# INITIALIZATION - Original IDEAM
# =========================================================================

N_t = max(1, CLI_ARGS.steps)
bar = Bar(max=N_t-1)
dt = 0.1
boundary = 1.0

X0 = [8.0, 0.0, 0.0, 20.0, 0.0, 0.0]
X0_g = [path2c(X0[3])[0], path2c(X0[3])[1], path2c.get_theta_r(X0[3])]

path_center = np.array([path1c, path2c, path3c], dtype=object)
sample_center = np.array([samples1c, samples2c, samples3c], dtype=object)
x_center = [x1c, x2c, x3c]
y_center = [y1c, y2c, y3c]
x_bound = [x1, x2]
y_bound = [y1, y2]
path_bound = [path1, path2]
path_bound_sample = [samples1, samples2]

X = [X0_g[0]]
Y = [X0_g[1]]
Psi = [X0_g[2]]
vx = [8.0]
vy = [0.0]
w = []
s = []
ey = [0.0]
epsi = [0.0]
t = [0.0]
a = [0.0]
delta = [0.0]
oa, od = 0.0, 0.0
u0 = [oa, od]
pathRecord = [1]
path_desired = []

x_area = 50.0   # widened: need forward view to see truck + phantom cut-in ahead of ego
y_area = 15.0

steer_range = [math.radians(-8.0), math.radians(8.0)]

Params = params()
Constraint_params = constraint_params()
dynamics = Dynamic(**Params)

decision_param = decision_params()
decision_maker = decision(**decision_param)

# ===========================================================================
# PRIDEAM CONTROLLER INITIALIZATION (replaces LMPC)
# ===========================================================================
controller = create_prideam_controller(
    paths={0: path1c, 1: path2c, 2: path3c},
    risk_weights={
        'mpc_cost': config_integration.mpc_risk_weight,
        'cbf_modulation': config_integration.cbf_alpha,
        'decision_threshold': config_integration.decision_risk_threshold,
    }
)
controller.get_path_curvature(path=path2c)

params_dir = os.path.abspath(CLI_ARGS.scenario_file)
if not os.path.exists(params_dir):
    raise FileNotFoundError(
        f"Scenario file not found: {params_dir}. Use --scenario-file to provide a valid path."
    )
surroundings = Surrounding_Vehicles(steer_range, dt, boundary, params_dir)

util_params_ = util_params()
utils = LeaderFollower_Uitl(**util_params_)
controller.set_util(utils)

# For backward compatibility, keep mpc_controller reference
mpc_controller = controller.mpc

# Separate baseline IDEAM MPC (visualization-only comparator).
# This computes the nominal IDEAM plan from the same timestep state so we can
# render a side-by-side IDEAM vs DREAM comparison in each saved frame.
baseline_mpc_viz = LMPC(**constraint_params())
# Give IDEAM baseline its own util so DREAM's dynamic d0/Th scaling does not
# bleed into the IDEAM comparison panel.
utils_ideam_viz = LeaderFollower_Uitl(**util_params_)
baseline_mpc_viz.set_util(utils_ideam_viz)
baseline_mpc_viz.get_path_curvature(path=path2c)
# 鈹€鈹€ Independent IDEAM ego state (propagated every step, no DRIFT influence) 鈹€鈹€
X0_ideam     = [8.0, 0.0, 0.0, 20.0, 0.0, 0.0]   # same start as DREAM ego
X0_g_ideam   = [path2c(X0_ideam[3])[0],
                path2c(X0_ideam[3])[1],
                path2c.get_theta_r(X0_ideam[3])]
oa_ideam_viz, od_ideam_viz   = 0.0, 0.0
last_X_ideam_viz             = None
last_ideam_panel_horizon_viz = None
path_changed_ideam_viz       = 1   # lane index of current path for warm-start
# Panel snapshot variables 鈥?initialized here so the per-step metrics block
# can safely read them even on frame 0 (before the IDEAM section first runs).
ideam_panel_X0   = list(X0_ideam)
ideam_panel_X0_g = list(X0_g_ideam)

path_changed = 1

# =========================================================================
# TRUCK-TRAILER DESIGNATIONS
# =========================================================================
# Instead of separate truck dynamics, we designate specific surrounding
# vehicles as "trucks". They use the SAME Curved_Road_Vehicle motion model
# as all other cars (IDM + PID), but:
#   1. Rendered with larger dimensions (TRUCK_LENGTH x TRUCK_WIDTH)
#   2. Passed to DRIFT as vclass='truck' 鈫?triggers occlusion shadow modeling
#   3. Desired speed (vd) set slightly slower for realism
#
# 鈹€鈹€ USER-CONFIGURABLE TRUCK LIST 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# Add one entry per truck:  (lane_index, vehicle_index): desired_speed_m_s
#   lane_index   : 0=left  1=center  2=right
#   vehicle_index: 1鈥? recommended (index 0 is behind ego, avoid giving it high vd)
#   desired_speed: 6鈥? m/s creates a meaningful slow obstacle
#
# Two-truck example 鈥?blocks center lane and left-escape lane simultaneously:
TRUCK_DESIGNATIONS = {
    (1, 2): 7.0,   # center lane, vehicle 1 鈫?slow truck directly ahead of ego
    #(2, 2): 7.0,   # left   lane, vehicle 2 鈫?second truck blocks left escape
}

# Override desired speeds for truck-designated vehicles
for (_truck_lane, _truck_vidx), _truck_vd in TRUCK_DESIGNATIONS.items():
    if _truck_lane == 0:
        surroundings.vd_left_all[_truck_vidx] = _truck_vd
    elif _truck_lane == 1:
        surroundings.vd_center_all[_truck_vidx] = _truck_vd
    elif _truck_lane == 2:
        surroundings.vd_right_all[_truck_vidx] = _truck_vd

# Print truck info
_lane_names = {0: 'left', 1: 'center', 2: 'right'}
_lane_vehicles = [surroundings.vehicle_left, surroundings.vehicle_center, surroundings.vehicle_right]
print(f"[TRUCK INIT] Designated {len(TRUCK_DESIGNATIONS)} trucks from surrounding vehicles:")
for (_tl, _tv), _tvd in TRUCK_DESIGNATIONS.items():
    _ts = _lane_vehicles[_tl][_tv]
    print(f"  Truck: {_lane_names[_tl]} lane, vehicle {_tv}, "
          f"s={_ts[0]:.1f}m, vx={_ts[6]:.1f}m/s, vd={_tvd:.1f}m/s")

# =========================================================================
# ACCELERATING AGENTS 鈥?raise desired speed for selected IDM vehicles so
# they naturally accelerate past the crowd (IDM free-flow regime).
#
# IMPORTANT: vehicle[0] in every lane starts BEHIND the ego (s鈮?-10 vs ego
# s=20) and its IDM does NOT see the ego as a leader 鈥?giving it a high vd
# causes it to ram the ego from behind.  Only indices 1-4 (all ahead of ego
# at s鈮?0-180) get elevated desired speeds.
# =========================================================================
surroundings.vd_left_all[1]   = 13.0   # left lane: 1st car ahead, moderate push
surroundings.vd_left_all[3]   = 14.0   # left lane: 4th car, mid-convoy surge
surroundings.vd_right_all[1]  = 13.5   # right lane: 1st car ahead
surroundings.vd_right_all[2]  = 13.0   # right lane: mid car
surroundings.vd_center_all[2] = 12.5   # center lane: mid car (truck in center 鈥?keep moderate)
print("[ACCEL] Elevated desired speeds set for 5 surrounding vehicles (indices 1-4 only).")

hidden_vehicles = []

# =========================================================================
# METRICS RECORD - Original
# =========================================================================

S_obs_record = []
initial_params = [
    surroundings.left_initial_set,
    surroundings.center_initial_set,
    surroundings.right_initial_set
]
initial_vds = [
    surroundings.vd_left_all,
    surroundings.vd_center_all,
    surroundings.vd_right_all
]
progress_20, progress_40 = 0.0, 0.0
TTC_record = []
vel = []
vys = []
ys = []
acc = []
steer_record = []
path_record = []
lane_state_record = []
C_label_record = []

# 鈹€鈹€ EXTENDED METRICS (paper evaluation) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
progress_record       = []   # DREAM: s(t) projected onto center-lane reference
progress_ideam_record = []   # IDEAM: s(t) same reference
vel_ideam_record      = []   # IDEAM: vx(t)
vys_ideam_record      = []   # IDEAM: vy(t)
acc_ideam_record      = []   # IDEAM: acceleration command
S_obs_ideam_record    = []   # IDEAM: minimum spacing to surrounding vehicles
truck_lon_dist_record      = []   # DREAM: signed longitudinal distance ego鈫抧earest truck
truck_lat_dist_record      = []   # DREAM: lateral distance ego鈫抧earest truck (abs)
truck_lon_dist_ideam_record = []  # IDEAM: same geometry, different ego position
truck_lat_dist_ideam_record = []  # IDEAM: lateral distance
planning_time_record       = []   # DREAM: decision + MPC wall-clock time per step (s)

last_desired_group = None
# =========================================================================
# HELPER FUNCTION: Convert IDEAM vehicles to DRIFT format
# =========================================================================

def convert_ideam_to_drift(vehicle_left, vehicle_centre, vehicle_right,
                           max_per_lane=None, truck_set=None):
    """Convert IDEAM vehicle arrays to DRIFT vehicle list.

    Args:
        max_per_lane: If set, only include this many vehicles per lane in the
            DRIFT field to avoid saturating the entire region with risk.
        truck_set: Set of (lane_index, vehicle_index) tuples designating trucks.
            Trucks always get vclass='truck' for DRIFT occlusion modeling
            and are included even beyond the per-lane limit.
    """
    vehicles = []
    vid = 1

    for lane_idx, vehicle_array in enumerate([vehicle_left, vehicle_centre, vehicle_right]):
        if vehicle_array is None:
            continue

        count = 0
        for row_idx, row in enumerate(vehicle_array):
            is_truck = truck_set is not None and (lane_idx, row_idx) in truck_set
            # Skip non-truck vehicles beyond per-lane limit, but always include trucks
            if max_per_lane is not None and count >= max_per_lane and not is_truck:
                continue
            if len(row) < 7:
                continue

            # IDEAM format: [s, ey, epsi, x, y, psi, vx, a]
            x_cart = row[3]
            y_cart = row[4]
            psi = row[5]
            v_lon = row[6]
            accel = row[7] if len(row) > 7 else 0.0  # Acceleration

            # Decompose longitudinal speed into Cartesian components using heading
            vx_cart = v_lon * np.cos(psi)
            vy_cart = v_lon * np.sin(psi)

            # Trucks get vclass='truck' -> triggers DRIFT occlusion shadow modeling
            vclass = 'truck' if is_truck else 'car'

            # Create DRIFT vehicle with proper Cartesian velocity
            v = drift_create_vehicle(
                vid=vid,
                x=x_cart,
                y=y_cart,
                vx=vx_cart,
                vy=vy_cart,
                vclass=vclass
            )
            # Add acceleration for braking effect detection
            v['a'] = accel
            v['heading'] = psi

            vehicles.append(v)
            vid += 1
            count += 1

    return vehicles


# =========================================================================
# DRIFT INITIALIZATION (NEW)
# =========================================================================

print("=" * 70)
print("DRIFT Risk Field Visualization Overlay")
print("=" * 70)
print("Initializing DRIFT interface...")

# Use controller's DRIFT instance (already created with paths)
drift = controller.drift

# =========================================================================
# ROAD BOUNDARY MASK 鈥?Dirichlet BC: R(x,t) = 0 outside 惟_road
# =========================================================================
# Road corridor is bounded by path (left edge) and path3 (right edge).
# We form a closed polygon from these boundary coordinates, then create
# a smooth mask M(x) 鈭?[0,1] on the PDE grid.
#
# PDE boundary condition enforced each step:
#   R^{n+1}(x) 鈫?R^{n+1}(x) 路 M(x),   M(x) = 0 for x 鈭?惟_road
#
from matplotlib.path import Path as MplPath

print("Computing road boundary mask...")

# Sample road edge paths (path = left edge, path3 = right edge)
# Subsample every 50th point from the dense coordinate arrays
_step = 50
# x, y = left road edge coords;  x3, y3 = right road edge coords
# (imported from Path.path via surrounding_vehicles)
left_edge = np.column_stack([x[::_step], y[::_step]])
right_edge = np.column_stack([x3[::_step], y3[::_step]])

# Close the polygon: left edge forward 鈫?right edge backward
road_polygon = np.vstack([left_edge, right_edge[::-1]])
road_mpl_path = MplPath(road_polygon)

# Test every grid point
grid_pts = np.column_stack([cfg.X.ravel(), cfg.Y.ravel()])
inside = road_mpl_path.contains_points(grid_pts).reshape(cfg.X.shape).astype(float)

# Smooth the mask edges (taper over ~2 grid cells) to avoid sharp numerical artifacts
from scipy.ndimage import gaussian_filter as _gf
road_mask = _gf(inside, sigma=1.5)
road_mask = np.clip(road_mask, 0, 1)

drift.set_road_mask(road_mask)
print(f"  Road mask: {np.sum(inside > 0.5)} / {inside.size} grid points on-road "
      f"({100*np.mean(inside > 0.5):.1f}%)")

# =========================================================================
# DRIFT WARM-UP (NEW) - Fix cold start problem
# =========================================================================

print("=" * 70)
print("DRIFT Risk Field Warm-Up")
print("=" * 70)

# Get initial vehicle states
vehicle_left_init, vehicle_centre_init, vehicle_right_init = surroundings.get_vehicles_states()

# Convert to DRIFT format (cars + trucks via designations)
vehicles_drift_init = convert_ideam_to_drift(
    vehicle_left_init, vehicle_centre_init, vehicle_right_init,
    max_per_lane=DRIFT_CARS_PER_LANE,
    truck_set=set(TRUCK_DESIGNATIONS.keys())
)


# Create initial ego vehicle dict (decompose body-frame velocity to Cartesian)
_init_psi = X0_g[2]
ego_drift_init = drift_create_vehicle(
    vid=0,
    x=X0_g[0],
    y=X0_g[1],
    vx=X0[0] * np.cos(_init_psi) - X0[1] * np.sin(_init_psi),
    vy=X0[0] * np.sin(_init_psi) + X0[1] * np.cos(_init_psi),
    vclass='car'
)
ego_drift_init['heading'] = X0_g[2]

# Pre-evolve risk field for 5 seconds
drift.warmup(vehicles_drift_init, ego_drift_init, dt=dt, duration=5.0, substeps=3)

print()


# Risk field recording
risk_at_ego_list = []
risk_fields = []  # Store risk fields for post-processing if needed

# Visualization settings
RISK_ALPHA = 0.65  # Transparency of risk overlay
RISK_CMAP = 'jet'  # Colormap: Blue->Cyan->Green->Yellow->Red
RISK_LEVELS = 40   # Number of contour levels
# Lower RISK_VMAX so occlusion-level risk (~0.8-1.5) shows clearly on colour scale.
# Phantom braking creates Q >> 5; after PDE diffusion values still 1-2 range.
RISK_VMAX = 2.0
SHOW_CONTOUR = True  # Show contour lines to highlight risk gradients
SHOW_HEATMAP = True  # Show filled contours
SHOW_COLORBAR = True  # Show colorbar legend
SHOW_PARALLEL_COMPARE = True  # Left: IDEAM baseline plan, Right: DREAM/PRIDEAM
COMPARE_LAYOUT = "vertical"  # "horizontal" or "vertical"

# Track colorbar to avoid accumulation
current_colorbar = None

print(f"Risk visualization settings:")
print(f"  - Alpha: {RISK_ALPHA}")
print(f"  - Colormap: {RISK_CMAP}")
print(f"  - Max risk display: {RISK_VMAX}")
print(f"  - Contour levels: {RISK_LEVELS}")
print(f"  - Grid: {cfg.x.shape[0]} x {cfg.y.shape[0]}")
print()



# =========================================================================
# HELPER FUNCTION: Plot risk field overlay
# =========================================================================

def plot_risk_overlay(risk_field, ego_x, ego_y, x_range, y_range, frame_idx=0):
    """
    Plot risk field as overlay on the current figure.
    Uses the full grid so the contourf stays fixed 鈥?axis limits handle cropping.
    """
    from scipy.ndimage import gaussian_filter
    R_smooth = gaussian_filter(risk_field, sigma=0.8)
    R_smooth = np.clip(R_smooth, 0, RISK_VMAX)

    contourf = None
    if SHOW_HEATMAP:
        contourf = plt.contourf(
            cfg.X, cfg.Y, R_smooth,
            levels=RISK_LEVELS,
            cmap=RISK_CMAP,
            alpha=RISK_ALPHA,
            vmin=0,
            vmax=RISK_VMAX,
            zorder=1,
            extend='max'
        )

    if SHOW_CONTOUR:
        plt.contour(
            cfg.X, cfg.Y, R_smooth,
            levels=np.linspace(0.2, RISK_VMAX, 8),
            colors='darkred',
            linewidths=0.5,
            alpha=0.4,
            zorder=1
        )

    return contourf


def draw_scene_panel(ax, ego_state, ego_global, vehicle_left, vehicle_centre, vehicle_right,
                     x_range, y_range, title, horizon_global=None, risk_field=None,
                     risk_value=None, show_shadow=True):
    """
    Draw one comparison panel (IDEAM or DREAM) on the provided axis.

    If risk_field is provided, the DRIFT risk overlay is drawn first and the
    contourf handle is returned for use in a shared colorbar.
    """
    plt.sca(ax)
    ax.cla()

    # Plot road geometry on current axis
    plot_env()

    contourf_obj = None
    if risk_field is not None:
        contourf_obj = plot_risk_overlay(
            risk_field, ego_global[0], ego_global[1], x_range, y_range
        )

    # Predicted horizon trajectory (if available)
    if horizon_global is not None and len(horizon_global) > 0:
        h_arr = np.asarray(horizon_global)
        if h_arr.ndim == 2 and h_arr.shape[1] >= 2:
            ax.plot(h_arr[:, 0], h_arr[:, 1], color='#00BCD4', linewidth=1.8,
                    linestyle='--', zorder=7)
            ax.scatter(h_arr[:, 0], h_arr[:, 1], color='#00BCD4', s=6, zorder=7)

    # Ego vehicle
    draw_vehicle_rect(ax, ego_global[0], ego_global[1], ego_global[2],
                      CAR_LENGTH, CAR_WIDTH, EGO_COLOR, edgecolor='navy', lw=1.0, zorder=6)

    # Ego speed text
    ego_s_txt, ego_ey_txt, _ = find_frenet_coord(path2c, x2c, y2c, samples2c, ego_global)
    textx, texty = path2c.get_cartesian_coords(ego_s_txt - 5.1, ego_ey_txt - 1.0)
    ax.text(
        textx, texty, "{} m/s".format(round(ego_state[0], 1)),
        rotation=np.rad2deg(path2c.get_theta_r(ego_state[3])),
        c='black', fontsize=5, style='oblique'
    )

    # Optional local risk text (drawn in the top-right of each subplot)
    if risk_value is not None:
        if risk_value > 1.5:
            risk_color = 'red'
        elif risk_value > 0.5:
            risk_color = 'orange'
        else:
            risk_color = 'green'
        ax.text(
            0.985, 0.965, f"R={risk_value:.2f}",
            transform=ax.transAxes, ha='right', va='top',
            c=risk_color, fontsize=7, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
        )

    # Surrounding vehicles (cars + truck designations)
    for lane_idx, (lane_vehs, path_ref, xc_ref, yc_ref, samp_ref) in enumerate([
        (vehicle_left, path1c, x1c, y1c, samples1c),
        (vehicle_centre, path2c, x2c, y2c, samples2c),
        (vehicle_right, path3c, x3c, y3c, samples3c),
    ]):
        for vi in range(len(lane_vehs)):
            is_truck = (lane_idx, vi) in TRUCK_DESIGNATIONS
            if not is_truck and vi >= RENDER_CARS_PER_LANE:
                continue

            veh = lane_vehs[vi]
            veh_in_panel = (
                x_range[0] <= veh[3] <= x_range[1] and
                y_range[0] <= veh[4] <= y_range[1]
            )
            if is_truck:
                if show_shadow:
                    shadow_poly = compute_truck_shadow(ego_global[0], ego_global[1], veh)
                    draw_shadow_polygon(ax, shadow_poly, alpha=0.30)
                draw_vehicle_rect(
                    ax, veh[3], veh[4], veh[5],
                    TRUCK_LENGTH, TRUCK_WIDTH, TRUCK_COLOR,
                    edgecolor='darkred', lw=1.2, zorder=5
                )
                if veh_in_panel:
                    ax.text(
                        veh[3] - 2, veh[4] + 2.5,
                        "Truck: {} m/s".format(round(veh[6], 1)),
                        rotation=np.rad2deg(veh[5]), c='darkred', fontsize=5,
                        style='oblique', fontweight='bold'
                    )
            else:
                draw_vehicle_rect(
                    ax, veh[3], veh[4], veh[5],
                    CAR_LENGTH, CAR_WIDTH, SURROUND_COLOR,
                    edgecolor='black', lw=0.6, zorder=4
                )

            if veh_in_panel:
                sv_s, sv_ey, _ = find_frenet_coord(path_ref, xc_ref, yc_ref, samp_ref,
                                                   [veh[3], veh[4], veh[5]])
                tx, ty = path_ref.get_cartesian_coords(sv_s - 4.75, sv_ey - 1.0)
                ax.text(tx, ty, "{} m/s".format(round(veh[6], 1)),
                        rotation=np.rad2deg(veh[5]), c='k', fontsize=5, style='oblique')

    ax.set_title(title, fontsize=10, fontweight='bold')
    # Shared axes + axis('equal') crashes on savefig in Matplotlib.
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim((x_range[0], x_range[1]))
    ax.set_ylim((y_range[0], y_range[1]))

    return contourf_obj


def build_rollout_horizon_for_panel(X0_seed, X0_g_seed, control_seq_a, control_seq_d,
                                    path_d, sample, x_list, y_list, boundary, dt):
    """
    Build a Cartesian horizon for visualization by rolling out the vehicle model.

    If control sequences are missing, falls back to zero controls to still show
    a nominal trajectory in the panel.
    """
    X_vis = list(X0_seed)
    X_g_vis = list(X0_g_seed)
    horizon = [list(X_g_vis)]

    n_steps = 0
    if control_seq_a is not None:
        n_steps = max(0, len(control_seq_a) - 1)
    else:
        n_steps = max(0, baseline_mpc_viz.T - 1)

    for k in range(n_steps):
        if control_seq_a is None or control_seq_d is None:
            u_vis = [0.0, 0.0]
        else:
            u_vis = [control_seq_a[k + 1], control_seq_d[k + 1]]
        X_vis, X_g_vis, _, _ = dynamics.propagate(
            X_vis, u_vis, dt, X_g_vis, path_d, sample, x_list, y_list, boundary
        )
        horizon.append(list(X_g_vis))

    return np.array(horizon)

# =========================================================================
# MAIN SIMULATION LOOP
# =========================================================================

print(f"Running simulation for {N_t} timesteps with risk visualization...")
print()

# Cooldown counter 鈥?prevents oscillating proactive lane changes
_proactive_lc_cooldown = 0

# 鈹€鈹€ Baseline MPC/util parameters (captured once; restored each step by scaling) 鈹€鈹€
_d0_base  = utils.d0               # 5.0 m  鈥?minimum gap in longitudinal CBF
_Th_base  = utils.Th               # 0.5 s  鈥?time headway in longitudinal CBF
_al_base  = controller.mpc.a_l     # 1.5    鈥?longitudinal ellipse semi-axis (lead)
_bl_base  = controller.mpc.b_l     # 2.2    鈥?lateral ellipse semi-axis (lead)
_P_base   = controller.mpc.P.copy()  # [[0.5,0],[0,0]] 鈥?center-keeping slack weight

for i in range(N_t):
    bar.next()

    # =====================================================================
    # SURROUNDING VEHICLES
    # =====================================================================
    vehicle_left, vehicle_centre, vehicle_right = surroundings.get_vehicles_states()

    # =====================================================================
    # UPDATE DRIFT RISK FIELD (NEW)
    # =====================================================================

    # Convert IDEAM vehicles to DRIFT format (trucks auto-tagged via designations)
    vehicles_drift = convert_ideam_to_drift(vehicle_left, vehicle_centre, vehicle_right,
                                               max_per_lane=DRIFT_CARS_PER_LANE,
                                               truck_set=set(TRUCK_DESIGNATIONS.keys()))


    # Add hidden vehicles to DRIFT (they create risk in occlusion zones)
    # These are NOT added to IDEAM arrays until they emerge from occlusion
    for hv in hidden_vehicles:
        if not hv.visible:
            s_hidden, x_hidden, y_hidden = hv.get_position([path1c, path2c, path3c])
            vehicles_drift.append(drift_create_vehicle(
                vid=900 + hidden_vehicles.index(hv),  # High ID to avoid conflicts
                x=x_hidden,
                y=y_hidden,
                vx=hv.vx,
                vy=0.0,
                vclass='car'
            ))

    # Create ego vehicle dict for DRIFT
    # X0[0]=vx_body, X0[1]=vy_body -> decompose to Cartesian using heading
    ego_psi = X0_g[2]
    ego_vx_cart = X0[0] * np.cos(ego_psi) - X0[1] * np.sin(ego_psi)
    ego_vy_cart = X0[0] * np.sin(ego_psi) + X0[1] * np.cos(ego_psi)
    ego_drift = drift_create_vehicle(
        vid=0,
        x=X0_g[0],
        y=X0_g[1],
        vx=ego_vx_cart,
        vy=ego_vy_cart,
        vclass='car'
    )
    ego_drift['heading'] = ego_psi

    # Check if hidden vehicles have emerged from occlusion
    for hv in hidden_vehicles:
        if not hv.visible:
            hv.check_emergence(X0_g[0], X0_g[1])
            if hv.visible:
                print(f"[EMERGENCE] Frame {i}: Hidden vehicle emerged from occlusion!")

    # Step DRIFT PDE solver
    risk_field = drift.step(vehicles_drift, ego_drift, dt=dt, substeps=3)
    # Note: drift and controller.drift are the same instance, so risk_field is automatically shared

    # Detect braking vehicles and log for metrics
    braking_vehicles = []
    for v_drift in vehicles_drift:
        if 'a' in v_drift and v_drift['a'] < -0.5:  # Significant braking
            braking_vehicles.append({
                'id': v_drift['id'],
                'accel': v_drift['a'],
                'x': v_drift['x'],
                'y': v_drift['y'],
                'dist_to_ego': np.sqrt((v_drift['x'] - X0_g[0])**2 + (v_drift['y'] - X0_g[1])**2)
            })

    # Log braking events
    if braking_vehicles and i % 10 == 0:  # Print every 10 frames when braking detected
        print(f"\n[BRAKING] Frame {i}:")
        for bv in braking_vehicles:
            print(f"  Vehicle {bv['id']}: a={bv['accel']:.2f} m/s虏, "
                  f"distance={bv['dist_to_ego']:.1f}m")

    # Query risk at ego position
    risk_at_ego = drift.get_risk_cartesian(X0_g[0], X0_g[1])
    risk_at_ego_list.append(risk_at_ego)   # record raw DRIFT value

    # =====================================================================
    # TRUCK RISK AMPLIFICATION + DYNAMIC MPC WEIGHT  (DREAM-specific)
    # =====================================================================
    # 1. Gaussian boost on the local risk_field copy around the truck so the
    #    MPC cost term and visualization reflect truck-induced risk more clearly.
    #    The internal DRIFT PDE state is NOT changed 鈥?only the local variable.
    # 2. MPC risk weight ramps linearly from base to base*TRUCK_WEIGHT_SCALE as
    #    ego closes within TRUCK_INFLUENCE_DIST, creating proportionally stronger
    #    avoidance behaviour absent from the IDEAM baseline.
    _ego_truck_dist = float('inf')
    _nearest_truck_key = None
    if TRUCK_DESIGNATIONS:
        _lane_arrays = [vehicle_left, vehicle_centre, vehicle_right]
        _boost_field = np.ones_like(risk_field)
        _ego_boost = 1.0

        for (_tk_l, _tk_v) in TRUCK_DESIGNATIONS.keys():
            _tv_cur = _lane_arrays[_tk_l][_tk_v]
            _tx, _ty = _tv_cur[3], _tv_cur[4]
            _d = float(np.sqrt((_tx - X0_g[0])**2 + (_ty - X0_g[1])**2))

            # Track nearest truck for dynamic weight and proactive LC
            if _d < _ego_truck_dist:
                _ego_truck_dist = _d
                _nearest_truck_key = (_tk_l, _tk_v)

            # Accumulate Gaussian boost (additive per truck, then multiply once)
            _dist_sq_grid = (cfg.X - _tx)**2 + (cfg.Y - _ty)**2
            _boost_field += TRUCK_RISK_BOOST * np.exp(
                -_dist_sq_grid / (2.0 * TRUCK_RISK_SIGMA**2))
            _ego_boost += TRUCK_RISK_BOOST * np.exp(
                -_d**2 / (2.0 * TRUCK_RISK_SIGMA**2))

        risk_field = risk_field * _boost_field
        risk_at_ego = risk_at_ego * _ego_boost

    # Dynamic MPC weight (applies even if no trucks present, but stays at base)
    _prox = max(0.0, 1.0 - _ego_truck_dist / TRUCK_INFLUENCE_DIST)
    controller.weights.mpc_cost = (
        config_integration.mpc_risk_weight * (
            1.0 + (TRUCK_WEIGHT_SCALE - 1.0) * _prox)
    )
    if i % 50 == 0 and _ego_truck_dist < TRUCK_INFLUENCE_DIST:
        _ln = {0:'L', 1:'C', 2:'R'}
        _nk = f"{_ln.get(_nearest_truck_key[0],'?')}{_nearest_truck_key[1]}" if _nearest_truck_key else "?"
        print(f"[TRUCK] Frame {i}: nearest={_nk}  ego-truck={_ego_truck_dist:.1f}m  "
              f"prox={_prox:.2f}  mpc_w={controller.weights.mpc_cost:.2f}"
              f"  R_ego_boosted={risk_at_ego:.3f}")

    # Diagnostic: verify occlusion source is being generated
    if i % 50 == 0 and drift.last_Q is not None:
        Q_total = drift.last_Q
        # Count trucks in DRIFT vehicles
        n_trucks = sum(1 for vd in vehicles_drift if vd.get('class') == 'truck')
        print(f"\n[DRIFT DIAG] Frame {i}: Q_max={np.max(Q_total):.2f}, "
              f"R_max={np.max(risk_field):.2f}, R_ego={risk_at_ego:.3f}, "
              f"trucks_in_drift={n_trucks}")

    # Store risk field (optional - can be memory intensive)
    if i % 10 == 0:  # Store every 10th frame to save memory
        risk_fields.append(risk_field.copy())

    # =====================================================================
    # PATH POSITION INFO (Original IDEAM)
    # =====================================================================
    path_now = judge_current_position(
        X0_g[0:2], x_bound, y_bound, path_bound, path_bound_sample
    )
    path_ego = surroundings.get_path_ego(path_now)

    # 鈹€鈹€ ASYMMETRIC CONSTRAINT SCALING (DREAM-only) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    # path_now is now defined 鈥?scale MPC parameters based on truck proximity.
    # Detect whether the nearest truck is in the ego's current lane.
    _truck_in_ego_lane = (
        _nearest_truck_key is not None and _nearest_truck_key[0] == path_now)

    # (a) Longitudinal following distance: d0 and Th scaled up when truck leads
    if _truck_in_ego_lane:
        utils.d0 = _d0_base + TRUCK_LONG_EXTRA * _prox   # 5 鈫?13 m at full prox
        utils.Th = _Th_base + TRUCK_TH_EXTRA  * _prox   # 0.5 鈫?1.5 s at full prox
    else:
        utils.d0 = _d0_base
        utils.Th = _Th_base

    # (b) Lateral clearance: inflate safety ellipse around lead vehicle
    #     (affects HOCBF during LC 鈥?ego passes truck with more lateral room)
    controller.mpc.a_l = _al_base * (1.0 + TRUCK_AL_SCALE * _prox)
    controller.mpc.b_l = _bl_base * (1.0 + TRUCK_BL_SCALE * _prox)

    # (c) Center-keeping relaxation: reduce P penalty so ego can offset laterally
    #     away from the truck's occlusion shadow without fighting the cost function
    _relax = max(1.0, 1.0 + TRUCK_CENTER_RELAX * _prox)
    controller.mpc.P = _P_base / _relax

    if i % 50 == 0 and _ego_truck_dist < TRUCK_INFLUENCE_DIST:
        _ln = {0:'L', 1:'C', 2:'R'}
        _nk = f"{_ln.get(_nearest_truck_key[0],'?')}{_nearest_truck_key[1]}" if _nearest_truck_key else "?"
        print(f"[TRUCK-SCALE] Frame {i}: nearest={_nk}  prox={_prox:.2f}"
              f"  d0={utils.d0:.1f}m  Th={utils.Th:.2f}s"
              f"  bl={controller.mpc.b_l:.2f}  P_scale=1/{_relax:.1f}")

    # 鈹€鈹€ LATERAL SQUEEZE DETECTION (DREAM-only) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    # Measure the minimum Frenet-s gap from each adjacent lane to the ego.
    # Paths run roughly parallel, so cross-lane s values are comparable to
    # within a few metres for moderate curvature.
    _left_adj_min_dist  = float('inf')
    _right_adj_min_dist = float('inf')
    _ego_s = X0[3]          # Frenet s of ego in current path

    if path_now == 1:       # center lane 鈥?neighbors on both sides
        for _veh in vehicle_left:
            _left_adj_min_dist  = min(_left_adj_min_dist,  abs(_veh[0] - _ego_s))
        for _veh in vehicle_right:
            _right_adj_min_dist = min(_right_adj_min_dist, abs(_veh[0] - _ego_s))
    elif path_now == 0:     # left lane 鈥?center lane is to the right
        for _veh in vehicle_centre:
            _right_adj_min_dist = min(_right_adj_min_dist, abs(_veh[0] - _ego_s))
    elif path_now == 2:     # right lane 鈥?center lane is to the left
        for _veh in vehicle_centre:
            _left_adj_min_dist  = min(_left_adj_min_dist,  abs(_veh[0] - _ego_s))

    # Guard: don't flag squeeze in the first few seconds while vehicles sort
    # themselves out (vehicle[0] starts behind ego and may be within threshold).
    _squeezed = (_left_adj_min_dist  < SQUEEZE_LON_DIST and
                 _right_adj_min_dist < SQUEEZE_LON_DIST and
                 _ego_s > 35.0)       # only after ego has advanced past startup
    _squeeze_escape_active = False   # set True when escape LC is triggered


    if path_now == 0:
        start_group_str = "L1"
    elif path_now == 1:
        start_group_str = "C1"
    else:
        start_group_str = "R1"

    # =====================================================================
    # DECISION INFO (Original IDEAM)
    # =====================================================================
    if i == 0:
        ovx, ovy, owz, oS, oey, oepsi = clac_last_X(
            oa, od, mpc_controller.T, path_ego, dt, 6, X0, X0_g
        )
        last_X = [ovx, ovy, owz, oS, oey, oepsi]

    all_info = utils.get_alllane_lf(
        path_ego, X0_g, path_now, vehicle_left, vehicle_centre, vehicle_right
    )
    group_dict, ego_group = utils.formulate_gap_group(
        path_now, last_X, all_info, vehicle_left, vehicle_centre, vehicle_right
    )

    start_time = time.time()
    desired_group = decision_maker.decision_making(group_dict, start_group_str)
    end_time = time.time()
    decision_making_duration = end_time - start_time

    path_d, path_dindex, C_label, sample, x_list, y_list, X0 = Decision_info(
        X0, X0_g, path_center, sample_center, x_center, y_center,
        boundary, desired_group, path_ego, path_now
    )

    C_label_additive = utils.inquire_C_state(C_label, desired_group)

    if C_label_additive == "Probe":
        path_d = path_ego
        path_dindex = path_now
        C_label_virtual = "K"
        _, xc, yc, samplesc = get_path_info(path_dindex)
        X0 = repropagate(path_d, samplesc, xc, yc, X0_g, X0)
    else:
        C_label_virtual = C_label

    # (ideam_viz_ctx removed 鈥?IDEAM now runs as a fully independent simulation
    #  with its own state X0_ideam / X0_g_ideam, not from DREAM's captured state)

    # =====================================================================
    # DRIFT DECISION VETO (NEW)
    # =====================================================================
    # Check for veto AFTER C_label_additive is set but BEFORE path changes committed
    if config_integration.enable_decision_veto and C_label != "K":
        ego_state = [X0[0], X0[1], X0[2], X0[3], X0[4], X0[5]]
        risk_score, allow, details = controller.evaluate_decision_risk(
            ego_state, path_now, path_dindex
        )
        if not allow:
            print(f"[DRIFT VETO] Frame {i}: Lane change blocked (risk={risk_score:.2f})")
            # Follow same pattern as "Probe" case - DON'T modify C_label/C_label_additive
            # Only override path and virtual label, let MPC see original decision but execute keep-lane
            path_d = path_ego
            path_dindex = path_now
            C_label_virtual = "K"  # Tell MPC to execute keep-lane behavior
            # Repropagate state on current path (same as Probe case)
            _, xc, yc, samplesc = get_path_info(path_dindex)
            X0 = repropagate(path_d, samplesc, xc, yc, X0_g, X0)

    # =====================================================================
    # DREAM PROACTIVE AVOIDANCE  (DREAM-only 鈥?not in IDEAM baseline)
    # =====================================================================
    # When approaching the truck with elevated boosted risk, DREAM does NOT
    # insist on strict lane-keeping.  It overrides the keep-lane decision
    # with a lane change away from the truck zone.  IDEAM, lacking any risk
    # awareness, stays in its current lane and follows the truck blindly.
    # A 30-frame cooldown prevents oscillation once a lane change completes.
    if _proactive_lc_cooldown > 0:
        _proactive_lc_cooldown -= 1

    if (TRUCK_DESIGNATIONS and
            _ego_truck_dist < TRUCK_PROACTIVE_DIST and
            risk_at_ego > TRUCK_PROACTIVE_RISK and
            C_label_virtual == "K" and
            _proactive_lc_cooldown == 0):
        _truck_lane = _nearest_truck_key[0]
        # Choose target: move away from truck's side of the road
        if path_now == _truck_lane:
            _alt_index = 1 if _truck_lane != 1 else 2   # out of truck lane
        elif path_now == 1:                              # center lane
            _alt_index = 2 if _truck_lane == 0 else 0   # away from truck side
        else:
            _alt_index = path_now                        # already far side

        if _alt_index != path_now:
            path_d = path_center[_alt_index]
            path_dindex = _alt_index
            C_label_virtual = "L" if _alt_index < path_now else "R"
            _, xc, yc, samplesc = get_path_info(path_dindex)
            X0 = repropagate(path_d, samplesc, xc, yc, X0_g, X0)
            _proactive_lc_cooldown = 30   # 3 s at dt=0.1
            print(f"[DREAM PROACTIVE] Frame {i} (t={i*dt:.1f}s): "
                  f"proactive LC 鈫?lane {_alt_index}  "
                  f"ego-truck={_ego_truck_dist:.1f}m  R={risk_at_ego:.2f}")

    # =====================================================================
    # DREAM LATERAL SQUEEZE ESCAPE  (DREAM-only)
    # =====================================================================
    # When the ego is hemmed in by vehicles on BOTH adjacent sides, GSD
    # alone cannot escape because it evaluates longitudinal gaps 鈥?it may
    # still prefer the current lane.  DREAM adds an explicit squeeze-escape
    # that overrides keep-lane and commands a LC toward the side with MORE
    # longitudinal room (larger s-gap to nearest adjacent vehicle).
    #
    # During the escape LC:
    #   鈥?b_l is reset to baseline so the lateral CBF stays feasible
    #   鈥?The truck speed cap is lifted so ego can accelerate into the gap
    #   鈥?A 30-frame cooldown (3 s) prevents oscillation
    if (_squeezed and C_label_virtual == "K" and _proactive_lc_cooldown == 0):
        _esc_index = -1
        if path_now == 1:
            # Prefer the side with the larger s-gap (more room longitudinally)
            _esc_index = 0 if _left_adj_min_dist >= _right_adj_min_dist else 2
        # Left/right edge lanes: squeeze means the center lane is close 鈥?just
        # accelerate forward; no LC needed (would go off-road).
        # (_esc_index stays -1 for path_now 0 or 2)

        if _esc_index != -1 and _esc_index != path_now:
            path_d          = path_center[_esc_index]
            path_dindex     = _esc_index
            C_label_virtual = "L" if _esc_index < path_now else "R"
            _, xc, yc, samplesc = get_path_info(path_dindex)
            X0 = repropagate(path_d, samplesc, xc, yc, X0_g, X0)
            # Reset lateral safety ellipse so LC constraint is feasible
            controller.mpc.b_l = _bl_base
            _proactive_lc_cooldown  = 30   # 3 s cooldown
            _squeeze_escape_active  = True
            print(f"[DREAM SQUEEZE] Frame {i} (t={i*dt:.1f}s): "
                  f"squeeze escape 鈫?lane {_esc_index}  "
                  f"adj_L={_left_adj_min_dist:.1f}m  adj_R={_right_adj_min_dist:.1f}m")
        elif path_now in (0, 2):
            # Edge lane: no LC possible, but release speed cap so ego can
            # pull forward past the adjacent vehicle
            _squeeze_escape_active = True

        if _squeeze_escape_active:
            # Zero the MPC risk weight so forward DRIFT risk doesn't prevent
            # the ego from accelerating out of the lateral squeeze.
            # The weight is restored next frame by the truck-proximity ramp.
            controller.weights.mpc_cost = 0.0

    # Even if C_label_virtual is already an LC (GSD or proactive LC already
    # commands it), zero the risk weight while squeezed so the ongoing LC
    # executes at full speed instead of being held back by DRIFT penalty.
    if _squeezed and not _squeeze_escape_active:
        controller.weights.mpc_cost = 0.0

    path_desired.append(path_d)
    if path_changed != path_dindex:
        controller.get_path_curvature(path=path_d)
        oS, oey = path_to_path_proj(oS, oey, path_changed, path_dindex)
        last_X = [ovx, ovy, owz, oS, oey, oepsi]

    path_changed = path_dindex

    # =====================================================================
    # METRICS CALCULATION (Original IDEAM)
    # =====================================================================
    ego_rect = create_rectangle(
        X0_g[0], X0_g[1],
        mpc_controller.vehicle_length, mpc_controller.vehicle_width, X0_g[2]
    )
    s_obs_min = surroundings.S_obs_calc(ego_rect)
    S_obs_record.append(s_obs_min)

    vel.append(X0[0])
    vys.append(X0[1])
    ys.append(X0_g[1])

    lane_state_record.append(C_label_additive)
    C_label_record.append(C_label)
    path_record.append(path_now)

    # 鈹€鈹€ EXTENDED METRICS per step 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    # DREAM progress: project onto center-lane reference path (path2c) so both
    # planners share the same longitudinal ruler regardless of their lane.
    _path_ref, _x_ref, _y_ref, _samples_ref = get_path_info(1)
    _s_dream, _, _ = find_frenet_coord(_path_ref, _x_ref, _y_ref, _samples_ref, X0_g)
    progress_record.append(float(_s_dream))

    # IDEAM progress (uses ideam_panel_X0_g, which lags 1 step 鈥?acceptable)
    if ideam_panel_X0_g is not None and len(ideam_panel_X0_g) >= 2:
        _s_ideam, _, _ = find_frenet_coord(_path_ref, _x_ref, _y_ref, _samples_ref,
                                           ideam_panel_X0_g)
        progress_ideam_record.append(float(_s_ideam))
    else:
        progress_ideam_record.append(progress_ideam_record[-1] if progress_ideam_record else 20.0)

    # IDEAM kinematics
    vel_ideam_record.append(float(ideam_panel_X0[0]) if ideam_panel_X0 else 0.0)
    vys_ideam_record.append(float(ideam_panel_X0[1]) if ideam_panel_X0 else 0.0)
    _oa_i_scalar = float(oa_ideam_viz[0] if hasattr(oa_ideam_viz, '__len__') else oa_ideam_viz)
    acc_ideam_record.append(_oa_i_scalar if SHOW_PARALLEL_COMPARE else 0.0)

    # IDEAM minimum spacing (reuse surroundings.S_obs_calc with IDEAM ego rect)
    if ideam_panel_X0_g is not None and len(ideam_panel_X0_g) >= 3:
        _ideam_rect = create_rectangle(
            ideam_panel_X0_g[0], ideam_panel_X0_g[1],
            mpc_controller.vehicle_length, mpc_controller.vehicle_width,
            ideam_panel_X0_g[2]
        )
        S_obs_ideam_record.append(float(surroundings.S_obs_calc(_ideam_rect)))
    else:
        S_obs_ideam_record.append(S_obs_ideam_record[-1] if S_obs_ideam_record else 100.0)

    # Truck longitudinal / lateral distance 鈥?DREAM and IDEAM share the same
    # traffic scene so the truck position is identical; only the ego differs.
    def _truck_dists(ego_x, ego_y, ego_psi, truck_key):
        """Return (lon_dist, lat_dist) from ego to truck. lon>0 means truck ahead."""
        _lane_arrs = [vehicle_left, vehicle_centre, vehicle_right]
        _tv = _lane_arrs[truck_key[0]][truck_key[1]]
        _tx, _ty = _tv[3], _tv[4]
        _dx, _dy = _tx - ego_x, _ty - ego_y
        _lon = _dx * np.cos(ego_psi) + _dy * np.sin(ego_psi)
        _lat = abs(-_dx * np.sin(ego_psi) + _dy * np.cos(ego_psi))
        return float(_lon), float(_lat)

    if _nearest_truck_key is not None:
        _ld, _ltd = _truck_dists(X0_g[0], X0_g[1], X0_g[2], _nearest_truck_key)
        truck_lon_dist_record.append(_ld)
        truck_lat_dist_record.append(_ltd)
        # IDEAM ego truck distance
        if ideam_panel_X0_g is not None and len(ideam_panel_X0_g) >= 3:
            _ld_i, _ltd_i = _truck_dists(ideam_panel_X0_g[0], ideam_panel_X0_g[1],
                                          ideam_panel_X0_g[2], _nearest_truck_key)
        else:
            _ld_i, _ltd_i = float('nan'), float('nan')
        truck_lon_dist_ideam_record.append(_ld_i)
        truck_lat_dist_ideam_record.append(_ltd_i)
    else:
        truck_lon_dist_record.append(float('nan'))
        truck_lat_dist_record.append(float('nan'))
        truck_lon_dist_ideam_record.append(float('nan'))
        truck_lat_dist_ideam_record.append(float('nan'))

    if i == 200:
        path_m, x_m, y_m, samples_m = get_path_info(1)
        s_m, ey_m, epsi_m = find_frenet_coord(path_m, x_m, y_m, samples_m, X0_g)
        progress_20 = s_m
        metrics_save(
            S_obs_record, initial_params, progress_20, progress_40,
            TTC_record, vel, vys, ys, acc, steer_record,
            lane_state_record, path_record, C_label_record,
            initial_vds, "200", round_="risk_viz"
        )

    if i == 400:
        path_m, x_m, y_m, samples_m = get_path_info(1)
        s_m, ey_m, epsi_m = find_frenet_coord(path_m, x_m, y_m, samples_m, X0_g)
        progress_40 = s_m
        metrics_save(
            S_obs_record, initial_params, progress_20, progress_40,
            TTC_record, vel, vys, ys, acc, steer_record,
            lane_state_record, path_record, C_label_record,
            initial_vds, "400", round_="risk_viz"
        )

    # =====================================================================
    # IDEAM BASELINE 鈥?FULLY INDEPENDENT PARALLEL SIMULATION
    # =====================================================================
    # The IDEAM ego has its own state (X0_ideam, X0_g_ideam) that evolves
    # every step from its own MPC-CBF decisions.  No DRIFT, no proactive LC,
    # no truck-aware parameter scaling.  Same surrounding vehicles (shared
    # environment), same decision structure, fixed util and MPC parameters.
    # =====================================================================
    ideam_panel_X0    = list(X0_ideam)
    ideam_panel_X0_g  = list(X0_g_ideam)
    ideam_panel_horizon = None
    ideam_panel_status  = "MPC-CBF (baseline)"

    if SHOW_PARALLEL_COMPARE:
        try:
            # 1. Position detection
            path_now_i = judge_current_position(
                X0_g_ideam[0:2], x_bound, y_bound, path_bound, path_bound_sample
            )
            path_ego_i = surroundings.get_path_ego(path_now_i)
            if path_now_i == 0:
                sgs_i = "L1"
            elif path_now_i == 1:
                sgs_i = "C1"
            else:
                sgs_i = "R1"

            # 2. Warm-start on first frame
            if i == 0:
                _ovx_i, _ovy_i, _owz_i, _oS_i, _oey_i, _oepsi_i = clac_last_X(
                    oa_ideam_viz, od_ideam_viz,
                    baseline_mpc_viz.T, path_ego_i, dt, 6, X0_ideam, X0_g_ideam
                )
                last_X_ideam_viz = [_ovx_i, _ovy_i, _owz_i, _oS_i, _oey_i, _oepsi_i]

            # 3. Decision making (pure IDEAM 鈥?no DRIFT, no risk)
            all_info_i = utils_ideam_viz.get_alllane_lf(
                path_ego_i, X0_g_ideam, path_now_i,
                vehicle_left, vehicle_centre, vehicle_right
            )
            group_dict_i, ego_group_i = utils_ideam_viz.formulate_gap_group(
                path_now_i, last_X_ideam_viz, all_info_i,
                vehicle_left, vehicle_centre, vehicle_right
            )
            desired_group_i = decision_maker.decision_making(group_dict_i, sgs_i)

            path_d_i, path_dindex_i, C_label_i, sample_i, x_list_i, y_list_i, X0_ideam = \
                Decision_info(
                    X0_ideam, X0_g_ideam, path_center, sample_center, x_center, y_center,
                    boundary, desired_group_i, path_ego_i, path_now_i
                )

            C_label_additive_i = utils_ideam_viz.inquire_C_state(C_label_i, desired_group_i)
            if C_label_additive_i == "Probe":
                path_d_i      = path_ego_i
                path_dindex_i = path_now_i
                C_label_virt_i = "K"
                _, xc_i, yc_i, sc_i = get_path_info(path_dindex_i)
                X0_ideam = repropagate(path_d_i, sc_i, xc_i, yc_i, X0_g_ideam, X0_ideam)
            else:
                C_label_virt_i = C_label_i

            # 4. Warm-start path-change projection
            if (path_changed_ideam_viz != path_dindex_i and
                    last_X_ideam_viz is not None):
                _oS_i, _oey_i = path_to_path_proj(
                    last_X_ideam_viz[3], last_X_ideam_viz[4],
                    path_changed_ideam_viz, path_dindex_i
                )
                last_X_ideam_viz = [
                    last_X_ideam_viz[0], last_X_ideam_viz[1], last_X_ideam_viz[2],
                    _oS_i, _oey_i, last_X_ideam_viz[5]
                ]

            baseline_mpc_viz.get_path_curvature(path=path_d_i)
            path_changed_ideam_viz = path_dindex_i

            # 5. MPC-CBF solve (fixed params 鈥?no truck scaling)
            baseline_result = baseline_mpc_viz.iterative_linear_mpc_control(
                X0_ideam, oa_ideam_viz, od_ideam_viz, dt,
                None, None, C_label_i, X0_g_ideam,
                path_d_i, last_X_ideam_viz,
                path_now_i, ego_group_i, path_ego_i, desired_group_i,
                vehicle_left, vehicle_centre, vehicle_right,
                path_dindex_i, C_label_additive_i, C_label_virt_i
            )

            if baseline_result is not None:
                oa_i, od_i, ovx_i, ovy_i, owz_i, oS_i, oey_i, oepsi_i = baseline_result
                last_X_ideam_viz   = [ovx_i, ovy_i, owz_i, oS_i, oey_i, oepsi_i]
                oa_ideam_viz, od_ideam_viz = oa_i, od_i

                # 6. Propagate IDEAM ego state (true independent evolution)
                X0_ideam, X0_g_ideam, _, _ = dynamics.propagate(
                    list(X0_ideam), [oa_i[0], od_i[0]], dt,
                    list(X0_g_ideam),
                    path_d_i, sample_i, x_list_i, y_list_i, boundary
                )
                ideam_panel_X0   = list(X0_ideam)
                ideam_panel_X0_g = list(X0_g_ideam)

                ideam_panel_horizon = build_rollout_horizon_for_panel(
                    ideam_panel_X0, ideam_panel_X0_g, oa_i, od_i,
                    path_d_i, sample_i, x_list_i, y_list_i, boundary, dt
                )
                last_ideam_panel_horizon_viz = ideam_panel_horizon
            else:
                ideam_panel_status = "IDEAM (MPC fail)"
                ideam_panel_horizon = last_ideam_panel_horizon_viz

        except Exception as e:
            ideam_panel_status = "IDEAM (error)"
            ideam_panel_horizon = last_ideam_panel_horizon_viz
            if i % 50 == 0:
                print(f"[IDEAM] Frame {i}: solve error: {e}")

    # =====================================================================
    # PRIDEAM MPC SOLVE WITH RISK INTEGRATION
    # =====================================================================
    # Trucks are already in the surrounding vehicle arrays (same motion model),
    # so MPC naturally sees them for collision avoidance 鈥?no injection needed.
    _t_solve_start = time.time()
    oa, od, ovx, ovy, owz, oS, oey, oepsi = controller.solve_with_risk(
        X0, oa, od, dt, None, None, C_label, X0_g, path_d, last_X,
        path_now, ego_group, path_ego, desired_group,
        vehicle_left, vehicle_centre, vehicle_right,
        path_dindex, C_label_additive, C_label_virtual
    )
    # Total DREAM planning time: GSD decision + MPC solve (DRIFT step counted separately)
    planning_time_record.append(decision_making_duration + (time.time() - _t_solve_start))

    # =====================================================================
    # EFFICIENCY TRADE-OFF  (DREAM-only)
    # =====================================================================
    # Inside the truck influence zone, DREAM enforces a speed cap so it
    # accepts a lower travel speed in exchange for safety margin.  The
    # IDEAM baseline has no such mechanism and maintains higher speed.
    # Cap is ONLY applied when:
    #   (a) truck is in ego's current lane   鈫?direct following, not overtaking
    #   (b) ego is keeping lane (C_label_virtual == "K") 鈫?not mid-overtake
    #   (c) NOT in squeeze-escape mode 鈫?ego must accelerate to exit squeeze
    # During lane change or squeeze escape the ego must be free to accelerate.
    if (_truck_in_ego_lane and
            C_label_virtual == "K" and
            not _squeeze_escape_active and
            _ego_truck_dist < TRUCK_INFLUENCE_DIST and
            X0[0] > TRUCK_SAFE_SPEED):
        _overspeed = X0[0] - TRUCK_SAFE_SPEED
        _decel_cmd = -min(2.0, _overspeed * 0.5)   # proportional, 鈮?2 m/s虏
        oa = list(oa)
        oa[0] = min(oa[0], _decel_cmd)

    # 鈹€鈹€ SQUEEZE ACCELERATION FLOOR (DREAM-only) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    # The risk-weight zeroing above removes the MPC's DRIFT penalty, but the
    # solver may still return near-zero accel if already at a local optimum.
    # Enforce a minimum acceleration whenever squeezed so the ego physically
    # escapes regardless of whether an LC or K was commanded.
    if _squeezed and oa[0] < SQUEEZE_MIN_ACCEL:
        oa = list(oa)
        oa[0] = SQUEEZE_MIN_ACCEL

    last_X = [ovx, ovy, owz, oS, oey, oepsi]
    u0 = [oa[0], od[0]]

    X0, X0_g, a_lon_lat, ds = dynamics.propagate(
        X0, u0, dt, X0_g, path_d, sample, x_list, y_list, boundary
    )

    acc.append(oa[0])
    steer_record.append(od[0])

    # Ego trajectory visualization
    X0_vis = X0
    X0_g_vis = X0_g
    X0_vis_list = [X0_vis]
    X0_g_vis_list = [X0_g_vis]
    for ooooa in range(len(oa) - 1):
        u0_vis = [oa[ooooa + 1], od[ooooa + 1]]
        X0_vis, X0_g_vis, _, _ = dynamics.propagate(
            X0_vis, u0_vis, dt, X0_g_vis, path_d, sample, x_list, y_list, boundary
        )
        X0_vis_list.append(X0_vis)
        X0_g_vis_list.append(X0_g_vis)

    X0_g_vis_list = np.array(X0_g_vis_list)

    surroundings.total_update()

    # =====================================================================
    # VISUALIZATION WITH RISK OVERLAY (MODIFIED)
    # =====================================================================

    # Remove old colorbar if it exists (prevents accumulation)
    if current_colorbar is not None:
        current_colorbar.remove()
        current_colorbar = None

    fig = plt.gcf()
    fig.clf()

    x_range = [X0_g[0] - x_area, X0_g[0] + x_area]
    y_range = [X0_g[1] - y_area, X0_g[1] + y_area]

    if SHOW_PARALLEL_COMPARE:
        # Each panel is centred on its OWN ego 鈥?no sharex/sharey, because the
        # two egos diverge in position as the simulation progresses.
        if str(COMPARE_LAYOUT).lower() == "vertical":
            ax_left  = fig.add_subplot(2, 1, 1)
            ax_right = fig.add_subplot(2, 1, 2)
        else:
            ax_left  = fig.add_subplot(1, 2, 1)
            ax_right = fig.add_subplot(1, 2, 2)

        # View window for each panel follows its own ego
        _ix, _iy = (ideam_panel_X0_g[0], ideam_panel_X0_g[1]) \
                    if (ideam_panel_X0_g is not None and len(ideam_panel_X0_g) >= 2) \
                    else (X0_g[0], X0_g[1])
        x_range_i = [_ix - x_area, _ix + x_area]
        y_range_i = [_iy - y_area, _iy + y_area]

        # IDEAM panel 鈥?own ego centre, no risk overlay
        draw_scene_panel(
            ax_left,
            ideam_panel_X0,
            ideam_panel_X0_g,
            vehicle_left, vehicle_centre, vehicle_right,
            x_range_i, y_range_i,
            title=ideam_panel_status,
            horizon_global=ideam_panel_horizon,
            risk_field=None,
            risk_value=None,
            show_shadow=False,
        )

        # DREAM panel 鈥?own ego centre, DRIFT risk overlay + score
        contourf_obj = draw_scene_panel(
            ax_right,
            X0, X0_g,
            vehicle_left, vehicle_centre, vehicle_right,
            x_range, y_range,
            title="DREAM (ours)",
            horizon_global=X0_g_vis_list,
            risk_field=risk_field,
            risk_value=risk_at_ego,
            show_shadow=True,
        )

        # Colorbar for the DRIFT risk overlay (DREAM panel only)
        if SHOW_COLORBAR and contourf_obj is not None:
            cbar = fig.colorbar(
                contourf_obj,
                ax=ax_right,
                orientation='vertical',
                pad=0.02,
                fraction=0.035,
            )
            cbar.set_label('Risk Level', fontsize=9, weight='bold')
            cbar.ax.tick_params(labelsize=8, colors='black')
            current_colorbar = cbar

        # Remove redundant x-tick labels on the top panel (vertical layout only)
        if str(COMPARE_LAYOUT).lower() == "vertical":
            ax_left.tick_params(labelbottom=False)
    else:
        ax = fig.add_subplot(1, 1, 1)
        contourf_obj = draw_scene_panel(
            ax,
            X0, X0_g,
            vehicle_left, vehicle_centre, vehicle_right,
            x_range, y_range,
            title=f"DREAM (PRIDEAM)",
            horizon_global=X0_g_vis_list,
            risk_field=risk_field,
            risk_value=risk_at_ego,
            show_shadow=True,
        )

        if SHOW_COLORBAR and contourf_obj is not None:
            cbar = fig.colorbar(
                contourf_obj,
                ax=ax,
                orientation='horizontal',
                pad=0.08,
                fraction=0.046,
                aspect=30
            )
            cbar.set_label('Risk Level', fontsize=9, weight='bold')
            cbar.ax.tick_params(labelsize=8, colors='black')
            current_colorbar = cbar  # Track for next frame

    # Save figure
    if SAVE_FRAMES:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        plt.savefig(os.path.join(save_dir, "{}.png".format(i)), dpi=SAVE_DPI)

bar.finish()

# =========================================================================
# COMPREHENSIVE METRICS ANALYSIS  (paper evaluation)
# =========================================================================
import warnings
warnings.filterwarnings("ignore")

_t   = np.arange(N_t) * dt                         # time axis (s)
_dt  = dt

# 鈹€鈹€ numpy arrays 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
_vx_d    = np.array(vel)                            # DREAM vx
_vy_d    = np.array(vys)                            # DREAM vy
_ax_d    = np.array(acc)                            # DREAM ax (control cmd)
_so_d    = np.array(S_obs_record)                   # DREAM min spacing
_prog_d  = np.array(progress_record)               # DREAM progress s(t)
_risk_d  = np.array(risk_at_ego_list)              # DREAM DRIFT risk

_vx_i    = np.array(vel_ideam_record)               # IDEAM vx
_vy_i    = np.array(vys_ideam_record)               # IDEAM vy
_ax_i    = np.array(acc_ideam_record)               # IDEAM ax
_so_i    = np.array(S_obs_ideam_record)             # IDEAM min spacing
_prog_i  = np.array(progress_ideam_record)          # IDEAM progress s(t)

_pt          = np.array(planning_time_record)           # DREAM planning time
_trk_lon     = np.array(truck_lon_dist_record)          # DREAM lon dist to truck
_trk_lat     = np.array(truck_lat_dist_record)          # DREAM lat dist to truck
_trk_lon_i   = np.array(truck_lon_dist_ideam_record)    # IDEAM lon dist to truck
_trk_lat_i   = np.array(truck_lat_dist_ideam_record)    # IDEAM lat dist to truck
def _finite_diff_abs(arr, dt_val):
    arr = np.asarray(arr, dtype=float)
    if arr.size <= 1:
        return np.zeros(max(1, arr.size), dtype=float)
    diff = np.abs(np.diff(arr) / dt_val)
    return np.append(diff, diff[-1])


# 鈹€鈹€ jerk (finite difference of ax) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
_jx_d = _finite_diff_abs(_ax_d, _dt)
_jx_i = _finite_diff_abs(_ax_i, _dt)

# 鈹€鈹€ lateral acceleration (finite difference of vy) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
_ay_d = _finite_diff_abs(_vy_d, _dt)
_ay_i = _finite_diff_abs(_vy_i, _dt)

# 鈹€鈹€ helper: index closest to a time value 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
def _tidx(t_sec): return min(int(round(t_sec / _dt)), N_t - 1)

# 鈹€鈹€ SCALAR METRICS 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
D_NC   = 3.0    # near-collision distance threshold (m)
TTC_NC = 2.0    # near-collision TTC threshold (s) 鈥?placeholder, TTC_record empty for now

metrics = {}
for tag, vx, vy, ax, so, prog, risk, jx, ay in [
        ("DREAM", _vx_d, _vy_d, _ax_d, _so_d, _prog_d, _risk_d, _jx_d, _ay_d),
        ("IDEAM", _vx_i, _vy_i, _ax_i, _so_i, _prog_i, np.zeros(N_t), _jx_i, _ay_i)]:
    m = {}
    # Progress
    m["Prog_20"]    = float(prog[_tidx(20)]) if len(prog) > _tidx(20) else float('nan')
    m["Prog_40"]    = float(prog[_tidx(40)]) if len(prog) > _tidx(40) else float('nan')
    m["Prog_max"]   = float(np.nanmax(prog))
    # Efficiency
    m["vx_mean"]    = float(np.mean(vx))
    m["vx_max"]     = float(np.max(vx))
    # Safety
    m["So_min"]     = float(np.min(so))
    m["So_mean"]    = float(np.mean(so))
    m["collisions"] = int(np.sum(so < 0.5))               # overlap threshold 0.5 m
    m["near_coll"]  = int(np.sum(so < D_NC))
    # Comfort
    m["ax_max"]     = float(np.max(np.abs(ax)))
    m["ax_mean"]    = float(np.mean(np.abs(ax)))
    m["jx_mean"]    = float(np.mean(jx))
    # Motion consistency
    vy_bar = float(np.mean(vy))
    m["vy_var"]     = float(np.mean((vy - vy_bar) ** 2))
    m["ay_peak"]    = float(np.max(ay))
    # Risk
    m["risk_mean"]  = float(np.mean(risk))
    m["risk_max"]   = float(np.max(risk))
    metrics[tag] = m

# 鈹€鈹€ Computational efficiency (DREAM only) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
_rt_exceed = np.sum(_pt > _dt)
metrics["DREAM"]["t_plan_mean"]  = float(np.mean(_pt))
metrics["DREAM"]["t_plan_max"]   = float(np.max(_pt))
metrics["DREAM"]["rt_exceed_rt"] = float(_rt_exceed / N_t)

# 鈹€鈹€ Truck proximity (DREAM only) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
for _tag, _tlon, _tlat in [("DREAM", _trk_lon, _trk_lat), ("IDEAM", _trk_lon_i, _trk_lat_i)]:
    _valid = ~np.isnan(_tlon)
    if np.any(_valid):
        _pos = _tlon[_tlon > 0] if np.any(_tlon > 0) else _tlon[_valid]
        metrics[_tag]["truck_lon_min"] = float(np.nanmin(_pos))
        metrics[_tag]["truck_lat_min"] = float(np.nanmin(_tlat[_valid]))
    else:
        metrics[_tag]["truck_lon_min"] = float('nan')
        metrics[_tag]["truck_lat_min"] = float('nan')

# 鈹€鈹€ Print table 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
print("\n" + "=" * 72)
print(f"{'METRIC':<32}{'DREAM':>18}{'IDEAM':>18}")
print("=" * 72)
_rows = [
    ("Progress @ 20 s   [m]",     "Prog_20"),
    ("Progress @ 40 s   [m]",     "Prog_40"),
    ("Max progress      [m]",     "Prog_max"),
    ("Mean speed vx     [m/s]",   "vx_mean"),
    ("Max speed vx      [m/s]",   "vx_max"),
    ("Min spacing So    [m]",     "So_min"),
    ("Mean spacing So   [m]",     "So_mean"),
    ("Collisions (So<0.5m)",      "collisions"),
    (f"Near-collisions(So<{D_NC}m)", "near_coll"),
    ("Peak |ax|         [m/s虏]",  "ax_max"),
    ("Mean |ax|         [m/s虏]",  "ax_mean"),
    ("Mean |jerk|       [m/s鲁]",  "jx_mean"),
    ("Var(vy)           [m虏/s虏]", "vy_var"),
    ("Peak |ay|         [m/s虏]",  "ay_peak"),
    ("Mean DRIFT risk",            "risk_mean"),
    ("Max  DRIFT risk",            "risk_max"),
]
for label, key in _rows:
    d_val = metrics["DREAM"].get(key, float('nan'))
    i_val = metrics["IDEAM"].get(key, float('nan'))
    d_str = f"{d_val:>12.3f}" if np.isfinite(d_val) else f"{'--':>12}"
    i_str = f"{i_val:>12.3f}" if np.isfinite(i_val) else f"{'--':>12}"
    print(f"  {label:<30}{d_str}{i_str}")
print("-" * 72)
print(f"  {'Plan time mean  [ms]':<30}"
      f"{metrics['DREAM']['t_plan_mean']*1e3:>12.1f}{'  (DREAM only)':>18}")
print(f"  {'Plan time max   [ms]':<30}"
      f"{metrics['DREAM']['t_plan_max']*1e3:>12.1f}")
print(f"  {'RT exceedance ratio':<30}"
      f"{metrics['DREAM']['rt_exceed_rt']:>12.3f}")
for _tag, _tln, _tlt in [
        ("Truck lon dist min [m]", "truck_lon_min", "truck_lon_min"),
        ("Truck lat dist min [m]", "truck_lat_min", "truck_lat_min")]:
    dv = metrics["DREAM"].get(_tlt, float('nan'))
    iv = metrics["IDEAM"].get(_tlt, float('nan'))
    ds = f"{dv:>12.2f}" if np.isfinite(dv) else f"{'--':>12}"
    is_ = f"{iv:>12.2f}" if np.isfinite(iv) else f"{'--':>12}"
    print(f"  {_tag:<30}{ds}{is_}")
print("=" * 72)

# =========================================================================
# METRICS FIGURE  (8-panel, SciencePlots style)
# =========================================================================
_win = 10   # rolling window for jerk smoothing

with plt.style.context(["science", "no-latex"]):
    fig_m, axes_m = plt.subplots(4, 2, figsize=(10, 14),
                                  constrained_layout=True)
    fig_m.suptitle("DREAM vs IDEAM 鈥?Evaluation Metrics", fontsize=11)

    # colour / style per planner
    _C  = {"DREAM": "C0", "IDEAM": "C1"}   # SciencePlots cycles C0=blue, C1=red/orange
    _LS = {"DREAM": "-",  "IDEAM": "--"}

    def _fmt(ax, xlabel, ylabel, title, legend=True):
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xlim([0, _t[-1]])
        if legend:
            ax.legend()

    # 鈹€鈹€ (0,0) Longitudinal progress s(t) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    ax = axes_m[0, 0]
    ax.plot(_t, _prog_d, color=_C["DREAM"], ls=_LS["DREAM"], label="DREAM")
    ax.plot(_t, _prog_i, color=_C["IDEAM"], ls=_LS["IDEAM"], label="IDEAM")
    ax.axvline(20, color="gray", lw=0.7, ls=":", label="$T$=20,40 s")
    ax.axvline(40, color="gray", lw=0.7, ls=":")
    _fmt(ax, "Time (s)", r"$s(t)$ (m)", "Longitudinal Progress")

    # 鈹€鈹€ (0,1) Longitudinal speed v_x 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    ax = axes_m[0, 1]
    ax.plot(_t, _vx_d, color=_C["DREAM"], ls=_LS["DREAM"], label="DREAM")
    ax.plot(_t, _vx_i, color=_C["IDEAM"], ls=_LS["IDEAM"], label="IDEAM")
    ax.axhline(metrics["DREAM"]["vx_mean"], color=_C["DREAM"], lw=0.8, ls=":",
               alpha=0.8, label=fr"$\bar{{v}}$ DREAM={metrics['DREAM']['vx_mean']:.1f}")
    ax.axhline(metrics["IDEAM"]["vx_mean"], color=_C["IDEAM"], lw=0.8, ls=":",
               alpha=0.8, label=fr"$\bar{{v}}$ IDEAM={metrics['IDEAM']['vx_mean']:.1f}")
    _fmt(ax, "Time (s)", r"$v_x$ (m/s)", "Longitudinal Speed")

    # 鈹€鈹€ (1,0) Minimum spacing S_o(t) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    ax = axes_m[1, 0]
    ax.plot(_t, _so_d, color=_C["DREAM"], ls=_LS["DREAM"], label="DREAM")
    ax.plot(_t, _so_i, color=_C["IDEAM"], ls=_LS["IDEAM"], label="IDEAM")
    ax.axhline(D_NC, color="red", lw=0.8, ls="--",
               label=f"Near-collision ({D_NC} m)")
    _fmt(ax, "Time (s)", r"$S_o(t)$ (m)", "Minimum Spacing")

    # 鈹€鈹€ (1,1) DRIFT risk R(t) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    ax = axes_m[1, 1]
    ax.plot(_t, _risk_d, color=_C["DREAM"], ls=_LS["DREAM"],
            label=fr"DREAM  $\bar{{R}}$={metrics['DREAM']['risk_mean']:.3f}")
    ax.fill_between(_t, 0, _risk_d, color=_C["DREAM"], alpha=0.12)
    ax.axhline(metrics["DREAM"]["risk_mean"], color=_C["DREAM"], lw=0.7, ls=":", alpha=0.8)
    _fmt(ax, "Time (s)", r"$\mathcal{R}(t)$", "DRIFT Risk at Ego")

    # 鈹€鈹€ (2,0) Longitudinal |acceleration| |a_x(t)| 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    ax = axes_m[2, 0]
    ax.plot(_t, np.abs(_ax_d), color=_C["DREAM"], ls=_LS["DREAM"], label="DREAM")
    ax.plot(_t, np.abs(_ax_i), color=_C["IDEAM"], ls=_LS["IDEAM"], label="IDEAM")
    _fmt(ax, "Time (s)", r"$|a_x(t)|$ (m/s$^2$)", "Longitudinal Acceleration")

    # 鈹€鈹€ (2,1) Jerk |j_x| smoothed 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    ax = axes_m[2, 1]
    _win_d = max(1, min(_win, len(_jx_d)))
    _win_i = max(1, min(_win, len(_jx_i)))
    _jd_sm = np.convolve(_jx_d, np.ones(_win_d) / _win_d, mode="same")
    _ji_sm = np.convolve(_jx_i, np.ones(_win_i) / _win_i, mode="same")
    ax.plot(_t, _jd_sm, color=_C["DREAM"], ls=_LS["DREAM"], label="DREAM")
    ax.plot(_t, _ji_sm, color=_C["IDEAM"], ls=_LS["IDEAM"], label="IDEAM")
    _fmt(ax, "Time (s)", r"$|j_x(t)|$ (m/s$^3$)",
         f"Longitudinal Jerk ({_win}-step avg)")

    # 鈹€鈹€ (3,0) Truck distance 鈥?DREAM vs IDEAM (lon + lat) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    ax = axes_m[3, 0]
    _mk  = ~np.isnan(_trk_lon)
    _mki = ~np.isnan(_trk_lon_i)
    if np.any(_mk):
        ax.plot(_t[_mk],  _trk_lon[_mk],  color=_C["DREAM"],  ls=_LS["DREAM"],
                label="DREAM lon")
        ax.plot(_t[_mk],  _trk_lat[_mk],  color=_C["DREAM"],  ls=":",
                label="DREAM lat")
    if np.any(_mki):
        ax.plot(_t[_mki], _trk_lon_i[_mki], color=_C["IDEAM"], ls=_LS["IDEAM"],
                label="IDEAM lon")
        ax.plot(_t[_mki], _trk_lat_i[_mki], color=_C["IDEAM"], ls=":",
                label="IDEAM lat")
    ax.axhline(0, color="gray", lw=0.5, ls=":")
    _fmt(ax, "Time (s)", "Distance (m)", "Ego $\\leftrightarrow$ Nearest Truck")

    # 鈹€鈹€ (3,1) Planning time per step 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    ax = axes_m[3, 1]
    ax.plot(_t[:len(_pt)], _pt * 1e3, color=_C["DREAM"], ls=_LS["DREAM"],
            lw=0.7, alpha=0.7, label="DREAM (GSD+MPC)")
    ax.axhline(_dt * 1e3, color="red",     lw=0.9, ls="--",
               label=f"RT limit ({_dt*1e3:.0f} ms)")
    ax.axhline(np.mean(_pt) * 1e3, color=_C["DREAM"], lw=0.9, ls=":",
               label=f"Mean = {np.mean(_pt)*1e3:.1f} ms")
    _fmt(ax, "Time (s)", "Wall-clock (ms)", "Planning Time per Step")

    _metrics_fig_path = os.path.join(save_dir, "metrics_comparison.png")
    fig_m.savefig(_metrics_fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig_m)

print(f"\n[METRICS] Comparison figure saved 鈫?{_metrics_fig_path}")

# 鈹€鈹€ Save metrics dict as numpy for further analysis 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
_metrics_np_path = os.path.join(save_dir, "metrics_dict.npy")
np.save(_metrics_np_path, metrics)
print(f"[METRICS] Scalar dict saved         鈫?{_metrics_np_path}")

# =========================================================================
# SAVE RISK DATA FOR POST-ANALYSIS
# =========================================================================

print("\nSaving risk data...")

# Save risk at ego as numpy array
risk_data_path = os.path.join(save_dir, "risk_at_ego.npy")
np.save(risk_data_path, np.array(risk_at_ego_list))
print(f"Risk data saved to: {risk_data_path}")

# Get PRIDEAM controller statistics
prideam_stats = controller.get_stats()

# Save summary as text file
summary_path = os.path.join(save_dir, "risk_summary.txt")
with open(summary_path, 'w') as f:
    f.write("PRIDEAM Functional Integration Summary\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Integration Mode: {INTEGRATION_MODE.upper()}\n")
    f.write(f"\nDense Traffic Scenario:\n")
    f.write(f"  Vehicles per lane:  5 (all IDM+PID controlled)\n")
    f.write(f"  Truck designation:  {list(TRUCK_DESIGNATIONS.keys())}\n")
    f.write(f"  Accelerating vd:    left[1]=13.0, left[3]=14.0, "
            f"right[1]=13.5, right[2]=13.0, center[2]=12.5 m/s\n")
    f.write("\n")
    f.write(f"  Decision Veto: {config_integration.enable_decision_veto}\n")
    f.write(f"  MPC Cost: {config_integration.enable_mpc_cost} (weight={config_integration.mpc_risk_weight})\n")
    f.write(f"  CBF Modulation: {config_integration.enable_cbf_modulation} (alpha={config_integration.cbf_alpha})\n\n")

    f.write(f"Simulation Duration: {N_t * dt:.1f} seconds ({N_t} timesteps)\n")
    f.write(f"Timestep: {dt}s\n")
    f.write(f"Grid Size: {cfg.x.shape[0]} x {cfg.y.shape[0]}\n")
    f.write(f"PDE Substeps: 3\n\n")

    f.write("DREAM Controller Statistics:\n")
    f.write("-" * 60 + "\n")
    f.write(f"Risk Field Updates: {prideam_stats['risk_field_updates']}\n")
    f.write(f"MPC Solves: {prideam_stats['mpc_solves']}\n")
    f.write(f"Lane Change Attempts: {prideam_stats['total_lane_change_attempts']}\n")
    f.write(f"Lane Changes Blocked by DRIFT: {prideam_stats['lane_changes_blocked']}\n")
    f.write(f"Block Rate: {prideam_stats['block_rate']*100:.1f}%\n\n")

    risk_arr = np.array(risk_at_ego_list)
    f.write("Risk at Ego Statistics:\n")
    f.write("-" * 60 + "\n")
    f.write(f"Mean:   {np.mean(risk_arr):.4f}\n")
    f.write(f"Median: {np.median(risk_arr):.4f}\n")
    f.write(f"Std:    {np.std(risk_arr):.4f}\n")
    f.write(f"Min:    {np.min(risk_arr):.4f}\n")
    f.write(f"Max:    {np.max(risk_arr):.4f}\n")
    f.write(f"Q25:    {np.percentile(risk_arr, 25):.4f}\n")
    f.write(f"Q75:    {np.percentile(risk_arr, 75):.4f}\n\n")

    # Time in risk zones
    total_time = N_t * dt
    time_low = np.sum(risk_arr < 0.5) * dt
    time_mod = np.sum((risk_arr >= 0.5) & (risk_arr < 1.5)) * dt
    time_high = np.sum(risk_arr >= 1.5) * dt

    f.write("Time in Risk Zones:\n")
    f.write("-" * 60 + "\n")
    f.write(f"Low Risk (R<0.5):        {time_low:6.1f}s ({time_low/total_time*100:5.1f}%)\n")
    f.write(f"Moderate Risk (0.5-1.5): {time_mod:6.1f}s ({time_mod/total_time*100:5.1f}%)\n")
    f.write(f"High Risk (R>1.5):       {time_high:6.1f}s ({time_high/total_time*100:5.1f}%)\n")

print(f"Summary saved to: {summary_path}")

# =========================================================================
# FINAL STATISTICS
# =========================================================================

print("\n" + "=" * 70)
print("PRIDEAM Simulation Complete - Functional Integration")
print("=" * 70)

print(f"\nIntegration Mode: {INTEGRATION_MODE.upper()}")
print(f"  Decision Veto: {config_integration.enable_decision_veto}")
print(f"  MPC Cost: {config_integration.enable_mpc_cost} (weight={config_integration.mpc_risk_weight})")
print(f"  CBF Modulation: {config_integration.enable_cbf_modulation} (alpha={config_integration.cbf_alpha})")

print(f"\nPRIDEAM Controller Statistics:")
print(f"  Lane Change Attempts: {prideam_stats['total_lane_change_attempts']}")
print(f"  Blocked by DRIFT: {prideam_stats['lane_changes_blocked']} ({prideam_stats['block_rate']*100:.1f}%)")
print(f"  MPC Solves: {prideam_stats['mpc_solves']}")

risk_arr = np.array(risk_at_ego_list)
print(f"\nRisk at ego statistics:")
print(f"  Mean: {np.mean(risk_arr):.3f}")
print(f"  Max: {np.max(risk_arr):.3f}")
print(f"  Min: {np.min(risk_arr):.3f}")
print(f"  Std: {np.std(risk_arr):.3f}")

# Find high-risk moments
high_risk_threshold = 1.0
high_risk_frames = np.where(risk_arr > high_risk_threshold)[0]
if len(high_risk_frames) > 0:
    print(f"\nHigh risk moments (R > {high_risk_threshold}):")
    print(f"  Count: {len(high_risk_frames)}")
    print(f"  Frames: {high_risk_frames[:10]}..." if len(high_risk_frames) > 10 else f"  Frames: {high_risk_frames}")

print(f"\nOutput saved to: {save_dir}")
if SAVE_FRAMES:
    print(f"Total frames: {N_t}")
    print("\nTo create video, run:")
    print(
        f"  ffmpeg -r {FFMPEG_FPS} -i {save_dir}/%d.png "
        "-vcodec libx264 -crf 18 -pix_fmt yuv420p risk_viz.mp4"
    )
else:
    print("Frame saving disabled (--save-frames false).")


