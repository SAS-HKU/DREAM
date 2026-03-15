"""
Uncertainty Merger Scenario: Baseline IDEAM vs DREAM
====================================================

Top subplot (baseline):
- Ego: IDEAM (MPC-CBF + IDEAM decision).
- Merger: IDEAM (right -> centre lane change).

Bottom subplot (ours):
- Ego: DREAM controller.
- Merger: IDEAM (right -> centre lane change).

Blocker placement and lane-change timing are tuned so ego/merger do not start
LC before overtaking the truck.
"""

import argparse
import os
import math
import sys
import time
import traceback
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path as MplPath
from matplotlib.transforms import Affine2D
from scipy.ndimage import gaussian_filter as _gf
import scienceplots  # noqa: F401

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from Control.MPC import *  # noqa: F403
from Control.constraint_params import *  # noqa: F403
from Model.Dynamical_model import *  # noqa: F403
from Model.params import *  # noqa: F403
from Model.surrounding_params import *  # noqa: F403
from Model.Surrounding_model import *  # noqa: F403
from Control.HOCBF import *  # noqa: F403
from DecisionMaking.decision_params import *  # noqa: F403
from DecisionMaking.give_desired_path import *  # noqa: F403
from DecisionMaking.util import *  # noqa: F403
from DecisionMaking.util_params import *  # noqa: F403
from DecisionMaking.decision import *  # noqa: F403
from Prediction.surrounding_prediction import *  # noqa: F403
from progress.bar import Bar

from config import Config as cfg
from pde_solver import create_vehicle as drift_create_vehicle
from Integration.prideam_controller import create_prideam_controller
from Integration.integration_config import get_preset

# ADA source adapter (DREAM-ADA arm)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "Aggressiveness_Modeling"))
from Aggressiveness_Modeling.ADA_drift_source import compute_Q_ADA  # noqa: E402

# APF source adapter (DREAM-APF arm)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "APF_Modeling"))
from APF_Modeling.APF_drift_source import compute_Q_APF  # noqa: E402


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
        description="Run uncertainty merger benchmark: IDEAM baseline vs DREAM."
    )
    parser.add_argument(
        "--integration-mode",
        default=os.environ.get("DREAM_INTEGRATION_MODE", "conservative"),
        help="Integration preset name for PRIDEAM controller.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=int(os.environ.get("DREAM_STEPS", "100")),
        help="Number of simulation steps.",
    )
    parser.add_argument(
        "--save-dpi",
        type=int,
        default=int(os.environ.get("DREAM_DPI", "300")),
        help="DPI used when saving frame images.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=int(os.environ.get("DREAM_LOG_EVERY", "20")),
        help="Logging interval in steps.",
    )
    parser.add_argument(
        "--save-frames",
        type=_str2bool,
        default=_str2bool(os.environ.get("DREAM_SAVE_FRAMES", "1")),
        help="Whether to save per-step visualization frames.",
    )
    parser.add_argument(
        "--save-dir",
        default=os.path.join(SCRIPT_DIR, "figsave_uncertainty_merger_v4"),
        help="Directory where frames and metrics are saved.",
    )
    return parser.parse_args()


CLI_ARGS = _parse_cli_args()

# ============================================================================
# CONFIGURATION
# ============================================================================

INTEGRATION_MODE = CLI_ARGS.integration_mode
config_integration = get_preset(INTEGRATION_MODE)
config_integration.apply_mode()

N_t = max(1, CLI_ARGS.steps)
SAVE_DPI = CLI_ARGS.save_dpi
LOG_EVERY = max(1, CLI_ARGS.log_every)
SAVE_FRAMES = CLI_ARGS.save_frames
dt = 0.1
boundary = 1.0

# Vehicle geometry
CAR_LENGTH = 3.5
CAR_WIDTH = 1.2
TRUCK_LENGTH = 12.0
TRUCK_WIDTH = 2.0

# Colors
EGO_IDEAM_COLOR = "#4CAF50"
EGO_DREAM_COLOR = "#2196F3"
EGO_ADA_COLOR   = "#9C27B0"
EGO_APF_COLOR   = "#009688"
TRUCK_COLOR = "#FF6F00"
MERGER_COLOR = "#E91E63"
SHADOW_COLOR = "#4A4A4A"

# DRIFT visualization
RISK_ALPHA = 0.65
RISK_CMAP = "jet"
RISK_LEVELS = 40
RISK_VMAX = 2.0

# Metric thresholds
NEAR_COLLISION_DIST = 3.0
COLLISION_DIST = 1.0

# Scenario parameters
TRUCK_VD = 5.3
TRUCK_S0 = 45.0
TRUCK_V0 = 5.3

EGO_S0 = 20.0
EGO_V0 = 10.2

# Left-lane blocker (IDM-controlled slow car).
BLOCKER_S_INIT = 80.0
BLOCKER_VD = 10.0

# Mergers: IDEAM-controlled in both baseline and DREAM worlds.
MERGER_BASE_S0 = 24.0
MERGER_BASE_V0 = 10.5
MERGER_DREAM_S0 = 24.0
MERGER_DREAM_V0 = 10.5

# Scenario-level lane-change forcing for reproducibility.
EGO_FORCE_CENTER_MIN_STEP = 0
MERGER_FORCE_CENTER_MIN_STEP = 0
EGO_LC_OVERTAKE_MARGIN = 4.0
MERGER_LC_OVERTAKE_MARGIN = 4.0
EGO_BLOCKER_TRIGGER_GAP = 45.0
EGO_BASE_ASSIST_BLEND = 0.20
EGO_DREAM_ASSIST_BLEND = 0.20
MERGER_BASE_ASSIST_BLEND = 0.24
MERGER_DREAM_ASSIST_BLEND = 0.24
KEEP_LANE_ASSIST_BLEND = 0.30

x_area = 50.0
y_area = 15.0
steer_range = [math.radians(-8.0), math.radians(8.0)]

save_dir = os.path.abspath(CLI_ARGS.save_dir)
os.makedirs(save_dir, exist_ok=True)


# ============================================================================
# SYNTHETIC AGENTS
# ============================================================================

class LeftLaneBlocker:
    """Slow IDM-controlled car in left lane — always visible to both planners."""

    def __init__(self, path, path_data, s_init, vd, dt, steer_range):
        self.path        = path
        self.path_data   = path_data   # (x, y, s) arrays for find_frenet_coord
        self.s           = float(s_init)
        self.v           = float(vd)
        self.vd          = float(vd)
        self.dt          = dt
        self.steer_range = steer_range
        self.a           = 0.0
        self.x, self.y   = path(self.s)
        self.yaw         = path.get_theta_r(self.s)
        self._ctrl = Curved_Road_Vehicle(
            a_max=2.0, delta=4, s0=2.0, b=1.5, T=1.5,
            K_P=1.0, K_D=0.1, K_I=0.01,
            dt=dt, lf=1.5, lr=1.5, length=CAR_LENGTH)

    def update(self):
        self.x, self.y, self.yaw, v_next, _, self.a = \
            self._ctrl.update_states(
                self.s, self.v, self.vd, None, None, self.path, self.steer_range)
        self.s += self.v * self.dt
        self.v  = max(0.0, v_next)

    def to_mpc_row(self):
        px, py, ps = self.path_data
        try:
            s_f, ey_f, epsi_f = find_frenet_coord(
                self.path, px, py, ps, [self.x, self.y, self.yaw])
        except Exception:
            s_f, ey_f, epsi_f = self.s, 0.0, 0.0
        return np.array([s_f, ey_f, epsi_f, self.x, self.y, self.yaw,
                         max(0.0, self.v), self.a])

    def to_drift_vehicle(self, vid=3):
        psi = self.yaw
        v   = drift_create_vehicle(vid=vid, x=self.x, y=self.y,
                                   vx=self.v * math.cos(psi),
                                   vy=self.v * math.sin(psi), vclass="car")
        v["heading"] = psi
        v["a"]       = self.a
        return v


# ============================================================================
# VISUAL HELPERS
# ============================================================================

def draw_vehicle_rect(ax, x, y, yaw_rad, length, width, facecolor,
                      edgecolor="black", lw=0.8, zorder=3, alpha=1.0,
                      linestyle="-"):
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


def compute_truck_shadow(ego_x, ego_y, truck_row, shadow_length=55.0):
    tx, ty, yaw = truck_row[3], truck_row[4], truck_row[5]
    dx, dy = tx - ego_x, ty - ego_y
    dist = math.sqrt(dx**2 + dy**2)
    if dist < 2.0:
        return None

    corners_local = np.array([
        [-TRUCK_LENGTH / 2, -TRUCK_WIDTH / 2],
        [TRUCK_LENGTH / 2, -TRUCK_WIDTH / 2],
        [TRUCK_LENGTH / 2, TRUCK_WIDTH / 2],
        [-TRUCK_LENGTH / 2, TRUCK_WIDTH / 2],
    ])
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    corners = (rot @ corners_local.T).T + np.array([tx, ty])

    angles = np.arctan2(corners[:, 1] - ego_y, corners[:, 0] - ego_x)
    left_corner = corners[np.argmax(angles)]
    right_corner = corners[np.argmin(angles)]

    l_dir = left_corner - np.array([ego_x, ego_y])
    l_dir /= (np.linalg.norm(l_dir) + 1e-9)
    r_dir = right_corner - np.array([ego_x, ego_y])
    r_dir /= (np.linalg.norm(r_dir) + 1e-9)

    return np.array([
        left_corner,
        left_corner + l_dir * shadow_length,
        right_corner + r_dir * shadow_length,
        right_corner
    ])


def draw_shadow_polygon(ax, shadow_polygon, alpha=0.30):
    if shadow_polygon is None:
        return
    patch = plt.Polygon(
        shadow_polygon,
        facecolor=SHADOW_COLOR, alpha=alpha,
        edgecolor="red", linewidth=1.6, linestyle="--", zorder=2
    )
    ax.add_patch(patch)


def draw_panel(ax, ego_X0, ego_X0_g, truck_row, merger_row, blocker_row,
               title, x_range, y_range,
               risk_field=None, horizon=None, ego_color=EGO_IDEAM_COLOR):
    plt.sca(ax)
    ax.cla()
    plot_env()

    contourf_obj = None
    if risk_field is not None:
        r_sm = _gf(risk_field, sigma=0.8)
        r_sm = np.clip(r_sm, 0, RISK_VMAX)
        contourf_obj = ax.contourf(
            cfg.X, cfg.Y, r_sm,
            levels=RISK_LEVELS, cmap=RISK_CMAP,
            alpha=RISK_ALPHA, vmin=0, vmax=RISK_VMAX,
            zorder=1, extend="max"
        )
        ax.contour(
            cfg.X, cfg.Y, r_sm,
            levels=np.linspace(0.2, RISK_VMAX, 8),
            colors="darkred", linewidths=0.5, alpha=0.4, zorder=1
        )

    if horizon is not None and len(horizon) > 0:
        h = np.asarray(horizon)
        if h.ndim == 2 and h.shape[1] >= 2:
            ax.plot(h[:, 0], h[:, 1], color="#00BCD4", lw=1.8, ls="--", zorder=7)
            ax.scatter(h[:, 0], h[:, 1], color="#00BCD4", s=6, zorder=7)

    # Truck + shadow
    draw_shadow_polygon(ax, compute_truck_shadow(ego_X0_g[0], ego_X0_g[1], truck_row))
    draw_vehicle_rect(
        ax, truck_row[3], truck_row[4], truck_row[5],
        TRUCK_LENGTH, TRUCK_WIDTH, TRUCK_COLOR,
        edgecolor="darkred", lw=1.2, zorder=5
    )
    ax.text(
        truck_row[3] - 2.5, truck_row[4] + 2.8, f"Truck {truck_row[6]:.1f} m/s",
        rotation=np.rad2deg(truck_row[5]), c="darkred",
        fontsize=5, style="oblique", fontweight="bold"
    )

    # Merger
    draw_vehicle_rect(
        ax, merger_row[3], merger_row[4], merger_row[5],
        CAR_LENGTH, CAR_WIDTH, MERGER_COLOR,
        edgecolor="black", lw=0.9, zorder=5
    )
    ax.text(
        merger_row[3] - 1.5, merger_row[4] + 1.1, "Merger",
        rotation=np.rad2deg(merger_row[5]), c="black",
        fontsize=5, style="oblique", fontweight="bold"
    )

    # Left-lane blocker
    if blocker_row is not None:
        draw_vehicle_rect(
            ax, blocker_row[3], blocker_row[4], blocker_row[5],
            CAR_LENGTH, CAR_WIDTH, "#616161",
            edgecolor="black", lw=0.9, zorder=5
        )
        ax.text(
            blocker_row[3] - 1.7, blocker_row[4] + 1.1, "Blocker",
            rotation=np.rad2deg(blocker_row[5]), c="black",
            fontsize=5, style="oblique", fontweight="bold"
        )

    # Ego
    draw_vehicle_rect(
        ax, ego_X0_g[0], ego_X0_g[1], ego_X0_g[2],
        CAR_LENGTH, CAR_WIDTH, ego_color,
        edgecolor="navy", lw=1.0, zorder=6
    )
    ax.text(
        ego_X0_g[0] - 2.0, ego_X0_g[1] - 2.0, f"{ego_X0[0]:.1f} m/s",
        c="black", fontsize=6
    )

    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    return contourf_obj


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

path_center = np.array([path1c, path2c, path3c], dtype=object)
sample_center = np.array([samples1c, samples2c, samples3c], dtype=object)
x_center = [x1c, x2c, x3c]
y_center = [y1c, y2c, y3c]
x_bound = [x1, x2]
y_bound = [y1, y2]
path_bound = [path1, path2]
path_bound_sample = [samples1, samples2]


def lane_from_global(X0_g):
    return judge_current_position(X0_g[0:2], x_bound, y_bound, path_bound, path_bound_sample)


def empty_lane():
    return np.zeros((0, 8), dtype=float)


def sort_lane(arr):
    if arr is None or len(arr) == 0:
        return empty_lane()
    idx = np.argsort(arr[:, 0])
    return np.asarray(arr[idx], dtype=float)


def stack_rows(rows):
    valid = [np.asarray(r, dtype=float).reshape(1, 8) for r in rows if r is not None]
    if not valid:
        return empty_lane()
    return sort_lane(np.vstack(valid))


def state_to_row(X0, X0_g):
    return np.array([X0[3], X0[4], X0[5], X0_g[0], X0_g[1], X0_g[2], X0[0], 0.0], dtype=float)


def row_to_drift_vehicle(row, vid, vclass="car"):
    psi = float(row[5])
    v_lon = float(row[6])
    v = drift_create_vehicle(
        vid=vid,
        x=float(row[3]), y=float(row[4]),
        vx=v_lon * math.cos(psi), vy=v_lon * math.sin(psi),
        vclass=vclass
    )
    v["heading"] = psi
    v["a"] = float(row[7]) if len(row) > 7 else 0.0
    return v


def build_horizon(X0_seed, X0_g_seed, oa_cmd, od_cmd, path_d, sample, x_list, y_list,
                  dyn_obj, dt_, boundary_):
    if oa_cmd is None or od_cmd is None:
        return None
    Xv = list(X0_seed)
    Xgv = list(X0_g_seed)
    out = [list(Xgv)]
    n = max(0, len(oa_cmd) - 1)
    for k in range(n):
        u = [oa_cmd[k + 1], od_cmd[k + 1]]
        Xv, Xgv, _, _ = dyn_obj.propagate(Xv, u, dt_, Xgv, path_d, sample, x_list, y_list, boundary_)
        out.append(list(Xgv))
    return np.array(out)


def pick_group_for_lane(group_dict, lane_index):
    lane_to_groups = {
        0: ("L1", "L2"),
        1: ("C1", "C2"),
        2: ("R1", "R2"),
    }
    for name in lane_to_groups.get(lane_index, ()):
        if name in group_dict:
            return group_dict[name]
    return group_dict[next(iter(group_dict))]


def force_path_target(path_now, target_lane, X0, X0_g):
    target_lane = int(target_lane)
    path_d = [path1c, path2c, path3c][target_lane]
    _, x_list, y_list, sample = get_path_info(target_lane)

    if target_lane == path_now:
        C_label = "K"
    else:
        C_label = "R" if target_lane > path_now else "L"
    X0_forced = repropagate(path_d, sample, x_list, y_list, X0_g, X0)
    return path_d, target_lane, C_label, sample, x_list, y_list, X0_forced


def blend_state_toward_lane(X0, X0_g, target_lane, blend=0.25):
    blend = float(np.clip(blend, 0.0, 1.0))
    path_t = [path1c, path2c, path3c][int(target_lane)]
    s_ref = float(X0[3])
    x_t, y_t = path_t(s_ref)
    psi_t = path_t.get_theta_r(s_ref)

    X0_g_new = list(X0_g)
    X0_g_new[0] = (1.0 - blend) * X0_g[0] + blend * x_t
    X0_g_new[1] = (1.0 - blend) * X0_g[1] + blend * y_t
    dpsi = math.atan2(math.sin(psi_t - X0_g[2]), math.cos(psi_t - X0_g[2]))
    X0_g_new[2] = X0_g[2] + blend * dpsi

    _, x_list, y_list, sample = get_path_info(int(target_lane))
    X0_new = repropagate(path_t, sample, x_list, y_list, X0_g_new, list(X0))
    return X0_new, X0_g_new


def progress_on_reference(X0_g):
    """Project global pose to a common reference for fair s(t) comparison."""
    candidates = (
        (path1c, x1c, y1c, samples1c),
        (path2c, x2c, y2c, samples2c),
        (path3c, x3c, y3c, samples3c),
    )
    for path_ref, xr, yr, sr in candidates:
        try:
            s_ref, _, _ = find_frenet_coord(path_ref, xr, yr, sr, X0_g)
            if np.isfinite(s_ref):
                return float(s_ref)
        except Exception:
            continue
    return float("nan")


def min_center_distance(ego_g, rows):
    """Minimum Euclidean centre distance from ego to a list of vehicle rows."""
    ex, ey = float(ego_g[0]), float(ego_g[1])
    dists = []
    for row in rows:
        if row is None:
            continue
        rx, ry = float(row[3]), float(row[4])
        d = math.hypot(rx - ex, ry - ey)
        if np.isfinite(d):
            dists.append(d)
    return float(min(dists)) if dists else float("nan")


def ideam_agent_step(
    X0, X0_g, oa_prev, od_prev, last_X_prev, path_changed_prev,
    mpc_ctrl, util_obj, decision_obj, dynamics_obj,
    lane_left, lane_center, lane_right,
    force_target_lane=None, bypass_probe_guard=False
):
    path_now = lane_from_global(X0_g)
    path_ego = [path1c, path2c, path3c][path_now]
    start_group_str = {0: "L1", 1: "C1", 2: "R1"}[path_now]
    force_active = force_target_lane is not None

    try:
        if last_X_prev is None:
            ovx, ovy, owz, oS, oey, oepsi = clac_last_X(
                oa_prev, od_prev, mpc_ctrl.T, path_ego, dt, 6, X0, X0_g)
            last_X_prev = [ovx, ovy, owz, oS, oey, oepsi]

        all_info = util_obj.get_alllane_lf(path_ego, X0_g, path_now, lane_left, lane_center, lane_right)
        group_dict, ego_group = util_obj.formulate_gap_group(
            path_now, last_X_prev, all_info, lane_left, lane_center, lane_right)

        desired_group = decision_obj.decision_making(group_dict, start_group_str)
        if force_active:
            desired_group = pick_group_for_lane(group_dict, int(force_target_lane))
        path_d, path_dindex, C_label, sample, x_list, y_list, X0 = Decision_info(
            X0, X0_g, path_center, sample_center, x_center, y_center,
            boundary, desired_group, path_ego, path_now
        )
        if force_active and path_dindex != int(force_target_lane):
            path_d, path_dindex, C_label, sample, x_list, y_list, X0 = force_path_target(
                path_now, int(force_target_lane), X0, X0_g
            )

        C_label_additive = util_obj.inquire_C_state(C_label, desired_group)
        C_label_virtual = C_label

        if C_label_additive == "Probe" and not (force_active and bypass_probe_guard):
            path_d, path_dindex, C_label_virtual = path_ego, path_now, "K"
            _, xc, yc, samplesc = get_path_info(path_dindex)
            X0 = repropagate(path_d, samplesc, xc, yc, X0_g, X0)

        if path_changed_prev != path_dindex and last_X_prev is not None:
            mpc_ctrl.get_path_curvature(path=path_d)
            proj_s, proj_ey = path_to_path_proj(last_X_prev[3], last_X_prev[4], path_changed_prev, path_dindex)
            last_X_prev = [last_X_prev[0], last_X_prev[1], last_X_prev[2], proj_s, proj_ey, last_X_prev[5]]
        path_changed_prev = path_dindex

        res = mpc_ctrl.iterative_linear_mpc_control(
            X0, oa_prev, od_prev, dt, None, None, C_label, X0_g, path_d, last_X_prev,
            path_now, ego_group, path_ego, desired_group,
            lane_left, lane_center, lane_right, path_dindex, C_label_additive, C_label_virtual
        )
        if res is None:
            raise RuntimeError("IDEAM MPC returned None")

        oa_cmd, od_cmd, ovx, ovy, owz, oS, oey, oepsi = res
        if oa_cmd is None or od_cmd is None:
            raise RuntimeError("IDEAM MPC returned empty controls")

        last_X = [ovx, ovy, owz, oS, oey, oepsi]
        X0_new, X0_g_new, _, _ = dynamics_obj.propagate(
            X0, [oa_cmd[0], od_cmd[0]], dt, X0_g, path_d, sample, x_list, y_list, boundary
        )

        return {
            "ok": True,
            "X0": X0_new,
            "X0_g": X0_g_new,
            "oa": oa_cmd,
            "od": od_cmd,
            "last_X": last_X,
            "path_changed": path_changed_prev,
            "path_d": path_d,
            "sample": sample,
            "x_list": x_list,
            "y_list": y_list,
            "path_now": path_now,
            "path_dindex": path_dindex,
            "forced": force_active,
        }
    except Exception as e:
        # Robust fallback to keep simulation running
        _, x_list, y_list, sample = get_path_info(path_now)
        oa_cmd = [0.0] * mpc_ctrl.T
        od_cmd = [0.0] * mpc_ctrl.T
        X0_new, X0_g_new, _, _ = dynamics_obj.propagate(
            X0, [0.0, 0.0], dt, X0_g, path_ego, sample, x_list, y_list, boundary
        )
        return {
            "ok": False,
            "error": f"{e}\n{traceback.format_exc()}",
            "X0": X0_new,
            "X0_g": X0_g_new,
            "oa": oa_cmd,
            "od": od_cmd,
            "last_X": last_X_prev,
            "path_changed": path_now,
            "path_d": path_ego,
            "sample": sample,
            "x_list": x_list,
            "y_list": y_list,
            "path_now": path_now,
            "path_dindex": path_now,
            "forced": force_active,
        }


def dream_agent_step(
    X0, X0_g, oa_prev, od_prev, last_X_prev, path_changed_prev,
    controller_obj, util_obj, decision_obj, dynamics_obj,
    lane_left, lane_center, lane_right, enable_decision_veto=True,
    force_target_lane=None, bypass_probe_guard=False, force_ignore_veto=True
):
    path_now = lane_from_global(X0_g)
    path_ego = [path1c, path2c, path3c][path_now]
    start_group_str = {0: "L1", 1: "C1", 2: "R1"}[path_now]
    force_active = force_target_lane is not None

    mpc_ctrl = controller_obj.mpc

    try:
        if last_X_prev is None:
            ovx, ovy, owz, oS, oey, oepsi = clac_last_X(
                oa_prev, od_prev, mpc_ctrl.T, path_ego, dt, 6, X0, X0_g)
            last_X_prev = [ovx, ovy, owz, oS, oey, oepsi]

        all_info = util_obj.get_alllane_lf(path_ego, X0_g, path_now, lane_left, lane_center, lane_right)
        group_dict, ego_group = util_obj.formulate_gap_group(
            path_now, last_X_prev, all_info, lane_left, lane_center, lane_right)

        desired_group = decision_obj.decision_making(group_dict, start_group_str)
        if force_active:
            desired_group = pick_group_for_lane(group_dict, int(force_target_lane))
        path_d, path_dindex, C_label, sample, x_list, y_list, X0 = Decision_info(
            X0, X0_g, path_center, sample_center, x_center, y_center,
            boundary, desired_group, path_ego, path_now
        )
        if force_active and path_dindex != int(force_target_lane):
            path_d, path_dindex, C_label, sample, x_list, y_list, X0 = force_path_target(
                path_now, int(force_target_lane), X0, X0_g
            )

        C_label_additive = util_obj.inquire_C_state(C_label, desired_group)
        C_label_virtual = C_label

        if C_label_additive == "Probe" and not (force_active and bypass_probe_guard):
            path_d, path_dindex, C_label_virtual = path_ego, path_now, "K"
            _, xc, yc, samplesc = get_path_info(path_dindex)
            X0 = repropagate(path_d, samplesc, xc, yc, X0_g, X0)

        if enable_decision_veto and C_label != "K" and not (force_active and force_ignore_veto):
            risk_score, allow, _ = controller_obj.evaluate_decision_risk(list(X0), path_now, path_dindex)
            if not allow:
                _ = risk_score  # silence lint intent
                path_d, path_dindex, C_label_virtual = path_ego, path_now, "K"
                _, xc, yc, samplesc = get_path_info(path_dindex)
                X0 = repropagate(path_d, samplesc, xc, yc, X0_g, X0)

        if path_changed_prev != path_dindex and last_X_prev is not None:
            controller_obj.get_path_curvature(path=path_d)
            proj_s, proj_ey = path_to_path_proj(last_X_prev[3], last_X_prev[4], path_changed_prev, path_dindex)
            last_X_prev = [last_X_prev[0], last_X_prev[1], last_X_prev[2], proj_s, proj_ey, last_X_prev[5]]
        path_changed_prev = path_dindex

        oa_cmd, od_cmd, ovx, ovy, owz, oS, oey, oepsi = controller_obj.solve_with_risk(
            X0, oa_prev, od_prev, dt, None, None, C_label, X0_g, path_d, last_X_prev,
            path_now, ego_group, path_ego, desired_group,
            lane_left, lane_center, lane_right,
            path_dindex, C_label_additive, C_label_virtual
        )
        if oa_cmd is None or od_cmd is None:
            raise RuntimeError("DREAM MPC returned empty controls")

        last_X = [ovx, ovy, owz, oS, oey, oepsi]
        X0_new, X0_g_new, _, _ = dynamics_obj.propagate(
            X0, [oa_cmd[0], od_cmd[0]], dt, X0_g, path_d, sample, x_list, y_list, boundary
        )

        return {
            "ok": True,
            "X0": X0_new,
            "X0_g": X0_g_new,
            "oa": oa_cmd,
            "od": od_cmd,
            "last_X": last_X,
            "path_changed": path_changed_prev,
            "path_d": path_d,
            "sample": sample,
            "x_list": x_list,
            "y_list": y_list,
            "path_now": path_now,
            "path_dindex": path_dindex,
            "forced": force_active,
        }
    except Exception as e:
        _, x_list, y_list, sample = get_path_info(path_now)
        oa_cmd = [0.0] * mpc_ctrl.T
        od_cmd = [0.0] * mpc_ctrl.T
        X0_new, X0_g_new, _, _ = dynamics_obj.propagate(
            X0, [0.0, 0.0], dt, X0_g, path_ego, sample, x_list, y_list, boundary
        )
        return {
            "ok": False,
            "error": f"{e}\n{traceback.format_exc()}",
            "X0": X0_new,
            "X0_g": X0_g_new,
            "oa": oa_cmd,
            "od": od_cmd,
            "last_X": last_X_prev,
            "path_changed": path_now,
            "path_d": path_ego,
            "sample": sample,
            "x_list": x_list,
            "y_list": y_list,
            "path_now": path_now,
            "path_dindex": path_now,
            "forced": force_active,
        }


def update_truck_state(truck_row, truck_dyn, leaders_center):
    prev = np.asarray(truck_row, dtype=float).copy()
    s = float(prev[0])
    v = max(0.0, float(prev[6]))

    s_ahead = None
    v_ahead = None
    ahead = [r for r in leaders_center if r is not None and float(r[0]) > s + 1.0]
    if ahead:
        ahead = sorted(ahead, key=lambda r: float(r[0]))
        s_ahead = float(ahead[0][0])
        v_ahead = float(ahead[0][6])

    def _fallback_row(v_hint=None, a_hint=0.0):
        s_fb = s + v * dt
        x_fb, y_fb = path2c(s_fb)
        psi_fb = path2c.get_theta_r(s_fb)
        v_fb = v if v_hint is None else max(0.0, float(v_hint))
        a_fb = float(a_hint) if np.isfinite(a_hint) else 0.0
        return np.array([s_fb, 0.0, 0.0, x_fb, y_fb, psi_fb, v_fb, a_fb], dtype=float)

    try:
        x_next, y_next, psi_next, v_next, _, a = truck_dyn.update_states(
            s, v, TRUCK_VD, s_ahead, v_ahead, path2c, steer_range
        )
        raw_vals = np.array([x_next, y_next, psi_next, v_next, a], dtype=float)
        if not np.all(np.isfinite(raw_vals)):
            return _fallback_row()

        s_next, ey_next, epsi_next = find_frenet_coord(
            path2c, x2c, y2c, samples2c, [x_next, y_next, psi_next]
        )
        row = np.array([s_next, ey_next, epsi_next, x_next, y_next, psi_next, v_next, a], dtype=float)
        if not np.all(np.isfinite(row)):
            return _fallback_row(v_hint=v_next, a_hint=a)

        # Keep truck anchored to centre lane in this synthetic scenario.
        if abs(float(ey_next)) > 2.0:
            return _fallback_row(v_hint=v_next, a_hint=a)
        return row
    except Exception:
        return _fallback_row()


# ============================================================================
# INITIALIZATION
# ============================================================================

print("=" * 72)
print("UNCERTAINTY MERGER DREAM (baseline IDEAM vs DREAM)")
print("=" * 72)
print(f"Integration mode: {INTEGRATION_MODE}")
print(f"Steps: {N_t}")
print()

Params = params()
dynamics = Dynamic(**Params)
decision_param = decision_params()

decision_base = decision(**decision_param)
decision_merger_base = decision(**decision_param)
decision_dream = decision(**decision_param)
decision_merger_dream = decision(**decision_param)

util_cfg = util_params()

base_mpc = LMPC(**constraint_params())
base_utils = LeaderFollower_Uitl(**util_cfg)
base_mpc.set_util(base_utils)
base_mpc.get_path_curvature(path=path1c)

dream_controller = create_prideam_controller(
    paths={0: path1c, 1: path2c, 2: path3c},
    risk_weights={
        "mpc_cost": config_integration.mpc_risk_weight,
        "cbf_modulation": config_integration.cbf_alpha,
        "decision_threshold": config_integration.decision_risk_threshold,
    }
)
dream_utils = LeaderFollower_Uitl(**util_cfg)
dream_controller.set_util(dream_utils)
dream_controller.get_path_curvature(path=path1c)

# Baseline merger controller (IDEAM in top subplot).
base_merger_mpc = LMPC(**constraint_params())
base_merger_utils = LeaderFollower_Uitl(**util_cfg)
base_merger_mpc.set_util(base_merger_utils)
base_merger_mpc.get_path_curvature(path=path3c)

# DREAM-world merger controller (IDEAM).
dream_merger_mpc = LMPC(**constraint_params())
dream_merger_utils = LeaderFollower_Uitl(**util_cfg)
dream_merger_mpc.set_util(dream_merger_utils)
dream_merger_mpc.get_path_curvature(path=path3c)

# DREAM-ADA controller — same MPC/CBF/decision logic, ADA source field.
ada_controller = create_prideam_controller(
    paths={0: path1c, 1: path2c, 2: path3c},
    risk_weights={
        "mpc_cost": config_integration.mpc_risk_weight,
        "cbf_modulation": config_integration.cbf_alpha,
        "decision_threshold": config_integration.decision_risk_threshold,
    }
)
ada_utils = LeaderFollower_Uitl(**util_cfg)
ada_controller.set_util(ada_utils)
ada_controller.get_path_curvature(path=path1c)
decision_ada = decision(**decision_param)

# ADA-world merger controller (IDEAM).
ada_merger_mpc = LMPC(**constraint_params())
ada_merger_utils = LeaderFollower_Uitl(**util_cfg)
ada_merger_mpc.set_util(ada_merger_utils)
ada_merger_mpc.get_path_curvature(path=path3c)
decision_merger_ada = decision(**decision_param)

# DREAM-APF controller — same MPC/CBF/decision logic, APF source field.
apf_controller = create_prideam_controller(
    paths={0: path1c, 1: path2c, 2: path3c},
    risk_weights={
        "mpc_cost": config_integration.mpc_risk_weight,
        "cbf_modulation": config_integration.cbf_alpha,
        "decision_threshold": config_integration.decision_risk_threshold,
    }
)
apf_utils = LeaderFollower_Uitl(**util_cfg)
apf_controller.set_util(apf_utils)
apf_controller.get_path_curvature(path=path1c)
decision_apf = decision(**decision_param)

# APF-world merger controller (IDEAM).
apf_merger_mpc = LMPC(**constraint_params())
apf_merger_utils = LeaderFollower_Uitl(**util_cfg)
apf_merger_mpc.set_util(apf_merger_utils)
apf_merger_mpc.get_path_curvature(path=path3c)
decision_merger_apf = decision(**decision_param)

# Ego states
X0_base = [EGO_V0, 0.0, 0.0, EGO_S0, 0.0, 0.0]
X0_g_base = [path1c(EGO_S0)[0], path1c(EGO_S0)[1], path1c.get_theta_r(EGO_S0)]
oa_base, od_base = 0.0, 0.0
last_X_base = None
path_changed_base = 0

X0_dream = [EGO_V0, 0.0, 0.0, EGO_S0, 0.0, 0.0]
X0_g_dream = [path1c(EGO_S0)[0], path1c(EGO_S0)[1], path1c.get_theta_r(EGO_S0)]
oa_dream, od_dream = 0.0, 0.0
last_X_dream = None
path_changed_dream = 0

# Baseline merger states (IDEAM).
X0_merger_base = [MERGER_BASE_V0, 0.0, 0.0, MERGER_BASE_S0, 0.0, 0.0]
X0_g_merger_base = [path3c(MERGER_BASE_S0)[0], path3c(MERGER_BASE_S0)[1], path3c.get_theta_r(MERGER_BASE_S0)]
oa_merger_base, od_merger_base = 0.0, 0.0
last_X_merger_base = None
path_changed_merger_base = 2

# DREAM-world merger states (IDEAM).
X0_merger_dream = [MERGER_DREAM_V0, 0.0, 0.0, MERGER_DREAM_S0, 0.0, 0.0]
X0_g_merger_dream = [path3c(MERGER_DREAM_S0)[0], path3c(MERGER_DREAM_S0)[1], path3c.get_theta_r(MERGER_DREAM_S0)]
oa_merger_dream, od_merger_dream = 0.0, 0.0
last_X_merger_dream = None
path_changed_merger_dream = 2

# ADA-world ego states (same initial conditions as DREAM world).
X0_ada = [EGO_V0, 0.0, 0.0, EGO_S0, 0.0, 0.0]
X0_g_ada = [path1c(EGO_S0)[0], path1c(EGO_S0)[1], path1c.get_theta_r(EGO_S0)]
oa_ada, od_ada = 0.0, 0.0
last_X_ada = None
path_changed_ada = 0

# ADA-world merger states (same initial as DREAM merger).
X0_merger_ada = [MERGER_DREAM_V0, 0.0, 0.0, MERGER_DREAM_S0, 0.0, 0.0]
X0_g_merger_ada = [path3c(MERGER_DREAM_S0)[0], path3c(MERGER_DREAM_S0)[1], path3c.get_theta_r(MERGER_DREAM_S0)]
oa_merger_ada, od_merger_ada = 0.0, 0.0
last_X_merger_ada = None
path_changed_merger_ada = 2

# APF-world ego states (same initial conditions as DREAM world).
X0_apf = [EGO_V0, 0.0, 0.0, EGO_S0, 0.0, 0.0]
X0_g_apf = [path1c(EGO_S0)[0], path1c(EGO_S0)[1], path1c.get_theta_r(EGO_S0)]
oa_apf, od_apf = 0.0, 0.0
last_X_apf = None
path_changed_apf = 0

# APF-world merger states (same initial as DREAM merger).
X0_merger_apf = [MERGER_DREAM_V0, 0.0, 0.0, MERGER_DREAM_S0, 0.0, 0.0]
X0_g_merger_apf = [path3c(MERGER_DREAM_S0)[0], path3c(MERGER_DREAM_S0)[1], path3c.get_theta_r(MERGER_DREAM_S0)]
oa_merger_apf, od_merger_apf = 0.0, 0.0
last_X_merger_apf = None
path_changed_merger_apf = 2

# Trucks (separate baseline and dream worlds, same dynamics/initial states).
truck_dyn = Curved_Road_Vehicle(**surrounding_params())
truck_x0 = path2c(TRUCK_S0)
truck_psi0 = path2c.get_theta_r(TRUCK_S0)
truck_row_base  = np.array([TRUCK_S0, 0.0, 0.0, truck_x0[0], truck_x0[1], truck_psi0, TRUCK_V0, 0.0], dtype=float)
truck_row_dream = np.array([TRUCK_S0, 0.0, 0.0, truck_x0[0], truck_x0[1], truck_psi0, TRUCK_V0, 0.0], dtype=float)
truck_row_ada   = np.array([TRUCK_S0, 0.0, 0.0, truck_x0[0], truck_x0[1], truck_psi0, TRUCK_V0, 0.0], dtype=float)
truck_row_apf   = np.array([TRUCK_S0, 0.0, 0.0, truck_x0[0], truck_x0[1], truck_psi0, TRUCK_V0, 0.0], dtype=float)

# Left-lane blocker: slow IDM car forcing both planners to consider LC to centre
blocker = LeftLaneBlocker(
    path=path1c, path_data=(x1c, y1c, samples1c),
    s_init=BLOCKER_S_INIT, vd=BLOCKER_VD, dt=dt, steer_range=steer_range)

# DRIFT setup for DREAM
drift = dream_controller.drift
print("Computing DRIFT road mask...")
step_mask = 50
left_edge = np.column_stack([x[::step_mask], y[::step_mask]])
right_edge = np.column_stack([x3[::step_mask], y3[::step_mask]])
road_polygon = np.vstack([left_edge, right_edge[::-1]])
road_path = MplPath(road_polygon)
grid_pts = np.column_stack([cfg.X.ravel(), cfg.Y.ravel()])
inside = road_path.contains_points(grid_pts).reshape(cfg.X.shape).astype(float)
road_mask = np.clip(_gf(inside, sigma=1.5), 0, 1)
drift.set_road_mask(road_mask)

print("DRIFT warm-up (3 s)...")
ego_drift_init = drift_create_vehicle(
    vid=0, x=X0_g_dream[0], y=X0_g_dream[1],
    vx=X0_dream[0] * math.cos(X0_g_dream[2]),
    vy=X0_dream[0] * math.sin(X0_g_dream[2]),
    vclass="car"
)
ego_drift_init["heading"] = X0_g_dream[2]
warm_vehicles = [
    row_to_drift_vehicle(truck_row_dream, vid=1, vclass="truck"),
    row_to_drift_vehicle(state_to_row(X0_merger_dream, X0_g_merger_dream), vid=2, vclass="car"),
    blocker.to_drift_vehicle(vid=3),   # blocker in DRIFT
]
drift.warmup(warm_vehicles, ego_drift_init, dt=dt, duration=3.0, substeps=3)

# DRIFT (ADA source)
drift_ada = ada_controller.drift
drift_ada.set_road_mask(road_mask)
print("ADA-DRIFT warm-up (3 s)...")
ego_drift_init_ada = drift_create_vehicle(
    vid=0, x=X0_g_ada[0], y=X0_g_ada[1],
    vx=X0_ada[0] * math.cos(X0_g_ada[2]),
    vy=X0_ada[0] * math.sin(X0_g_ada[2]),
    vclass="car"
)
ego_drift_init_ada["heading"] = X0_g_ada[2]
warm_vehicles_ada = [
    row_to_drift_vehicle(truck_row_ada, vid=1, vclass="truck"),
    row_to_drift_vehicle(state_to_row(X0_merger_ada, X0_g_merger_ada), vid=2, vclass="car"),
    blocker.to_drift_vehicle(vid=3),
]
drift_ada.warmup(warm_vehicles_ada, ego_drift_init_ada, dt=dt, duration=3.0, substeps=3,
                 source_fn=compute_Q_ADA)
print()

# DRIFT (APF source)
drift_apf = apf_controller.drift
drift_apf.set_road_mask(road_mask)
print("APF-DRIFT warm-up (3 s)...")
ego_drift_init_apf = drift_create_vehicle(
    vid=0, x=X0_g_apf[0], y=X0_g_apf[1],
    vx=X0_apf[0] * math.cos(X0_g_apf[2]),
    vy=X0_apf[0] * math.sin(X0_g_apf[2]),
    vclass="car"
)
ego_drift_init_apf["heading"] = X0_g_apf[2]
warm_vehicles_apf = [
    row_to_drift_vehicle(truck_row_apf, vid=1, vclass="truck"),
    row_to_drift_vehicle(state_to_row(X0_merger_apf, X0_g_merger_apf), vid=2, vclass="car"),
    blocker.to_drift_vehicle(vid=3),
]
drift_apf.warmup(warm_vehicles_apf, ego_drift_init_apf, dt=dt, duration=3.0, substeps=3,
                 source_fn=compute_Q_APF)
print()

print(f"Ego init left lane     s={EGO_S0:.1f}  vx={EGO_V0:.1f}")
print(f"Truck init center lane s={TRUCK_S0:.1f}  vx={TRUCK_V0:.1f}  vd={TRUCK_VD:.1f}")
print(f"Blocker init left lane s={blocker.s:.1f}  vd={BLOCKER_VD:.1f}")
print(f"Baseline merger init right lane s={MERGER_BASE_S0:.1f}  vx={MERGER_BASE_V0:.1f}")
print(f"DREAM merger init right lane s={MERGER_DREAM_S0:.1f}  vx={MERGER_DREAM_V0:.1f}")
print()


# ============================================================================
# MAIN LOOP
# ============================================================================

bar = Bar(max=max(1, N_t - 1))
risk_field = risk_field_ada = risk_field_apf = None

plt.figure(figsize=(24, 8))

# ============================================================================
# METRIC BUFFERS
# ============================================================================
time_hist = []

base_s = []
base_vx = []
base_acc = []
base_lane_hist = []
base_dist_merger = []
base_min_dist_sur = []

dream_s = []
dream_vx = []
dream_acc = []
dream_lane_hist = []
dream_dist_merger = []
dream_min_dist_sur = []

base_merger_lane_hist = []
dream_merger_lane_hist = []
risk_at_ego_hist = []

ada_s = []
ada_vx = []
ada_acc = []
ada_lane_hist = []
ada_dist_merger = []
ada_min_dist_sur = []
ada_merger_lane_hist = []
risk_at_ego_ada_hist = []

apf_s = []
apf_vx = []
apf_acc = []
apf_lane_hist = []
apf_dist_merger = []
apf_min_dist_sur = []
apf_merger_lane_hist = []
risk_at_ego_apf_hist = []

for i in range(N_t):
    bar.next()

    # ------------------------------------------------------------------------
    # Advance synthetic agents
    # ------------------------------------------------------------------------
    blocker.update()

    # ------------------------------------------------------------------------
    # Build baseline world lanes (all IDEAM-controlled).
    # ------------------------------------------------------------------------
    _bl_row = blocker.to_mpc_row()
    merger_base_row = state_to_row(X0_merger_base, X0_g_merger_base)
    merger_base_lane = lane_from_global(X0_g_merger_base)

    lane_left_base = stack_rows([_bl_row] + ([merger_base_row] if merger_base_lane == 0 else []))
    lane_center_base = stack_rows([truck_row_base] + ([merger_base_row] if merger_base_lane == 1 else []))
    lane_right_base = stack_rows([merger_base_row] if merger_base_lane == 2 else [])

    # ------------------------------------------------------------------------
    # Baseline ego (IDEAM).
    # ------------------------------------------------------------------------
    ego_blocker_gap_base = float(blocker.s) - float(X0_base[3])
    ego_base_ready_for_center = (
        i >= EGO_FORCE_CENTER_MIN_STEP and
        X0_base[3] > float(truck_row_base[0]) + EGO_LC_OVERTAKE_MARGIN and
        0.0 < ego_blocker_gap_base < EGO_BLOCKER_TRIGGER_GAP
    )
    lane_base_now = lane_from_global(X0_g_base)
    if lane_base_now == 0:
        force_ego_base = 1 if ego_base_ready_for_center else 0
    else:
        force_ego_base = None
    out_base = ideam_agent_step(
        X0_base, X0_g_base, oa_base, od_base, last_X_base, path_changed_base,
        base_mpc, base_utils, decision_base, dynamics,
        lane_left_base, lane_center_base, lane_right_base,
        force_target_lane=force_ego_base,
        bypass_probe_guard=True
    )
    X0_base, X0_g_base = out_base["X0"], out_base["X0_g"]
    oa_base, od_base = out_base["oa"], out_base["od"]
    last_X_base, path_changed_base = out_base["last_X"], out_base["path_changed"]
    if force_ego_base is not None:
        tgt = int(force_ego_base)
        if lane_from_global(X0_g_base) != tgt:
            blend = EGO_BASE_ASSIST_BLEND if tgt == 1 else KEEP_LANE_ASSIST_BLEND
            X0_base, X0_g_base = blend_state_toward_lane(X0_base, X0_g_base, tgt, blend)

    # ------------------------------------------------------------------------
    # Baseline merger (IDEAM).
    # ------------------------------------------------------------------------
    base_row = state_to_row(X0_base, X0_g_base)
    base_lane = lane_from_global(X0_g_base)
    lane_left_merger = stack_rows([_bl_row] + ([base_row] if base_lane == 0 else []))
    lane_center_merger = stack_rows([truck_row_base] + ([base_row] if base_lane == 1 else []))
    lane_right_merger = stack_rows([base_row] if base_lane == 2 else [])

    merger_base_ready_for_center = (
        i >= MERGER_FORCE_CENTER_MIN_STEP and
        X0_merger_base[3] > float(truck_row_base[0]) + MERGER_LC_OVERTAKE_MARGIN and
        lane_from_global(X0_g_base) == 1
    )
    lane_merger_base_now = lane_from_global(X0_g_merger_base)
    if lane_merger_base_now == 2:
        force_merger_base = 1 if merger_base_ready_for_center else 2
    else:
        force_merger_base = None
    out_merger_base = ideam_agent_step(
        X0_merger_base, X0_g_merger_base,
        oa_merger_base, od_merger_base, last_X_merger_base, path_changed_merger_base,
        base_merger_mpc, base_merger_utils, decision_merger_base, dynamics,
        lane_left_merger, lane_center_merger, lane_right_merger,
        force_target_lane=force_merger_base,
        bypass_probe_guard=True
    )
    X0_merger_base, X0_g_merger_base = out_merger_base["X0"], out_merger_base["X0_g"]
    oa_merger_base, od_merger_base = out_merger_base["oa"], out_merger_base["od"]
    last_X_merger_base, path_changed_merger_base = out_merger_base["last_X"], out_merger_base["path_changed"]
    if force_merger_base is not None:
        tgt = int(force_merger_base)
        if lane_from_global(X0_g_merger_base) != tgt:
            blend = MERGER_BASE_ASSIST_BLEND if tgt == 1 else KEEP_LANE_ASSIST_BLEND
            X0_merger_base, X0_g_merger_base = blend_state_toward_lane(
                X0_merger_base, X0_g_merger_base, tgt, blend
            )

    # ------------------------------------------------------------------------
    # Build DREAM world lanes (DREAM ego + IDEAM merger).
    # ------------------------------------------------------------------------
    merger_dream_row = state_to_row(X0_merger_dream, X0_g_merger_dream)
    merger_dream_lane = lane_from_global(X0_g_merger_dream)
    lane_left_dream = stack_rows([_bl_row] + ([merger_dream_row] if merger_dream_lane == 0 else []))
    lane_center_dream = stack_rows([truck_row_dream] + ([merger_dream_row] if merger_dream_lane == 1 else []))
    lane_right_dream = stack_rows([merger_dream_row] if merger_dream_lane == 2 else [])

    # ------------------------------------------------------------------------
    # DRIFT step (GVF) for DREAM world.
    # ------------------------------------------------------------------------
    ego_drift = drift_create_vehicle(
        vid=0, x=X0_g_dream[0], y=X0_g_dream[1],
        vx=X0_dream[0] * math.cos(X0_g_dream[2]) - X0_dream[1] * math.sin(X0_g_dream[2]),
        vy=X0_dream[0] * math.sin(X0_g_dream[2]) + X0_dream[1] * math.cos(X0_g_dream[2]),
        vclass="car"
    )
    ego_drift["heading"] = X0_g_dream[2]

    drift_vehicles = [
        row_to_drift_vehicle(truck_row_dream, vid=1, vclass="truck"),
        row_to_drift_vehicle(merger_dream_row, vid=2, vclass="car"),
        blocker.to_drift_vehicle(vid=3),
    ]
    risk_field = drift.step(drift_vehicles, ego_drift, dt=dt, substeps=3)

    # ------------------------------------------------------------------------
    # Build ADA world lanes.
    # ------------------------------------------------------------------------
    merger_ada_row = state_to_row(X0_merger_ada, X0_g_merger_ada)
    merger_ada_lane = lane_from_global(X0_g_merger_ada)
    lane_left_ada    = stack_rows([_bl_row] + ([merger_ada_row] if merger_ada_lane == 0 else []))
    lane_center_ada  = stack_rows([truck_row_ada] + ([merger_ada_row] if merger_ada_lane == 1 else []))
    lane_right_ada   = stack_rows([merger_ada_row] if merger_ada_lane == 2 else [])

    # ------------------------------------------------------------------------
    # DRIFT step (ADA) for ADA world.
    # ------------------------------------------------------------------------
    ego_drift_ada_v = drift_create_vehicle(
        vid=0, x=X0_g_ada[0], y=X0_g_ada[1],
        vx=X0_ada[0] * math.cos(X0_g_ada[2]) - X0_ada[1] * math.sin(X0_g_ada[2]),
        vy=X0_ada[0] * math.sin(X0_g_ada[2]) + X0_ada[1] * math.cos(X0_g_ada[2]),
        vclass="car"
    )
    ego_drift_ada_v["heading"] = X0_g_ada[2]
    drift_vehicles_ada = [
        row_to_drift_vehicle(truck_row_ada, vid=1, vclass="truck"),
        row_to_drift_vehicle(merger_ada_row, vid=2, vclass="car"),
        blocker.to_drift_vehicle(vid=3),
    ]
    risk_field_ada = drift_ada.step(drift_vehicles_ada, ego_drift_ada_v,
                                    dt=dt, substeps=3, source_fn=compute_Q_ADA)

    # ------------------------------------------------------------------------
    # DREAM ego (risk-aware).
    # ------------------------------------------------------------------------
    ego_blocker_gap_dream = float(blocker.s) - float(X0_dream[3])
    ego_dream_ready_for_center = (
        i >= EGO_FORCE_CENTER_MIN_STEP and
        X0_dream[3] > float(truck_row_dream[0]) + EGO_LC_OVERTAKE_MARGIN and
        0.0 < ego_blocker_gap_dream < EGO_BLOCKER_TRIGGER_GAP
    )
    lane_dream_now = lane_from_global(X0_g_dream)
    if lane_dream_now == 0:
        force_ego_dream = 1 if ego_dream_ready_for_center else 0
    else:
        force_ego_dream = None
    out_dream = dream_agent_step(
        X0_dream, X0_g_dream, oa_dream, od_dream, last_X_dream, path_changed_dream,
        dream_controller, dream_utils, decision_dream, dynamics,
        lane_left_dream, lane_center_dream, lane_right_dream,
        enable_decision_veto=config_integration.enable_decision_veto,
        force_target_lane=force_ego_dream,
        bypass_probe_guard=True,
        force_ignore_veto=True
    )
    X0_dream, X0_g_dream = out_dream["X0"], out_dream["X0_g"]
    oa_dream, od_dream = out_dream["oa"], out_dream["od"]
    last_X_dream, path_changed_dream = out_dream["last_X"], out_dream["path_changed"]
    if force_ego_dream is not None:
        tgt = int(force_ego_dream)
        if lane_from_global(X0_g_dream) != tgt:
            blend = EGO_DREAM_ASSIST_BLEND if tgt == 1 else KEEP_LANE_ASSIST_BLEND
            X0_dream, X0_g_dream = blend_state_toward_lane(X0_dream, X0_g_dream, tgt, blend)

    # ------------------------------------------------------------------------
    # DREAM-world merger (IDEAM).
    # ------------------------------------------------------------------------
    dream_row = state_to_row(X0_dream, X0_g_dream)
    dream_lane = lane_from_global(X0_g_dream)
    lane_left_merger_dream = stack_rows([_bl_row] + ([dream_row] if dream_lane == 0 else []))
    lane_center_merger_dream = stack_rows([truck_row_dream] + ([dream_row] if dream_lane == 1 else []))
    lane_right_merger_dream = stack_rows([dream_row] if dream_lane == 2 else [])

    merger_dream_ready_for_center = (
        i >= MERGER_FORCE_CENTER_MIN_STEP and
        X0_merger_dream[3] > float(truck_row_dream[0]) + MERGER_LC_OVERTAKE_MARGIN and
        lane_from_global(X0_g_dream) == 1
    )
    lane_merger_dream_now = lane_from_global(X0_g_merger_dream)
    if lane_merger_dream_now == 2:
        force_merger_dream = 1 if merger_dream_ready_for_center else 2
    else:
        force_merger_dream = None
    out_merger_dream = ideam_agent_step(
        X0_merger_dream, X0_g_merger_dream,
        oa_merger_dream, od_merger_dream, last_X_merger_dream, path_changed_merger_dream,
        dream_merger_mpc, dream_merger_utils, decision_merger_dream, dynamics,
        lane_left_merger_dream, lane_center_merger_dream, lane_right_merger_dream,
        force_target_lane=force_merger_dream,
        bypass_probe_guard=True
    )
    X0_merger_dream, X0_g_merger_dream = out_merger_dream["X0"], out_merger_dream["X0_g"]
    oa_merger_dream, od_merger_dream = out_merger_dream["oa"], out_merger_dream["od"]
    last_X_merger_dream, path_changed_merger_dream = out_merger_dream["last_X"], out_merger_dream["path_changed"]
    if force_merger_dream is not None:
        tgt = int(force_merger_dream)
        if lane_from_global(X0_g_merger_dream) != tgt:
            blend = MERGER_DREAM_ASSIST_BLEND if tgt == 1 else KEEP_LANE_ASSIST_BLEND
            X0_merger_dream, X0_g_merger_dream = blend_state_toward_lane(
                X0_merger_dream, X0_g_merger_dream, tgt, blend
            )

    # ------------------------------------------------------------------------
    # ADA ego (risk-aware, ADA source).
    # ------------------------------------------------------------------------
    ego_blocker_gap_ada = float(blocker.s) - float(X0_ada[3])
    ego_ada_ready_for_center = (
        i >= EGO_FORCE_CENTER_MIN_STEP and
        X0_ada[3] > float(truck_row_ada[0]) + EGO_LC_OVERTAKE_MARGIN and
        0.0 < ego_blocker_gap_ada < EGO_BLOCKER_TRIGGER_GAP
    )
    lane_ada_now = lane_from_global(X0_g_ada)
    if lane_ada_now == 0:
        force_ego_ada = 1 if ego_ada_ready_for_center else 0
    else:
        force_ego_ada = None
    out_ada = dream_agent_step(
        X0_ada, X0_g_ada, oa_ada, od_ada, last_X_ada, path_changed_ada,
        ada_controller, ada_utils, decision_ada, dynamics,
        lane_left_ada, lane_center_ada, lane_right_ada,
        enable_decision_veto=config_integration.enable_decision_veto,
        force_target_lane=force_ego_ada,
        bypass_probe_guard=True,
        force_ignore_veto=True
    )
    X0_ada, X0_g_ada = out_ada["X0"], out_ada["X0_g"]
    oa_ada, od_ada = out_ada["oa"], out_ada["od"]
    last_X_ada, path_changed_ada = out_ada["last_X"], out_ada["path_changed"]
    if force_ego_ada is not None:
        tgt = int(force_ego_ada)
        if lane_from_global(X0_g_ada) != tgt:
            blend = EGO_DREAM_ASSIST_BLEND if tgt == 1 else KEEP_LANE_ASSIST_BLEND
            X0_ada, X0_g_ada = blend_state_toward_lane(X0_ada, X0_g_ada, tgt, blend)

    # ------------------------------------------------------------------------
    # ADA-world merger (IDEAM).
    # ------------------------------------------------------------------------
    ada_row = state_to_row(X0_ada, X0_g_ada)
    ada_lane = lane_from_global(X0_g_ada)
    lane_left_merger_ada   = stack_rows([_bl_row] + ([ada_row] if ada_lane == 0 else []))
    lane_center_merger_ada = stack_rows([truck_row_ada] + ([ada_row] if ada_lane == 1 else []))
    lane_right_merger_ada  = stack_rows([ada_row] if ada_lane == 2 else [])

    merger_ada_ready_for_center = (
        i >= MERGER_FORCE_CENTER_MIN_STEP and
        X0_merger_ada[3] > float(truck_row_ada[0]) + MERGER_LC_OVERTAKE_MARGIN and
        lane_from_global(X0_g_ada) == 1
    )
    lane_merger_ada_now = lane_from_global(X0_g_merger_ada)
    if lane_merger_ada_now == 2:
        force_merger_ada = 1 if merger_ada_ready_for_center else 2
    else:
        force_merger_ada = None
    out_merger_ada = ideam_agent_step(
        X0_merger_ada, X0_g_merger_ada,
        oa_merger_ada, od_merger_ada, last_X_merger_ada, path_changed_merger_ada,
        ada_merger_mpc, ada_merger_utils, decision_merger_ada, dynamics,
        lane_left_merger_ada, lane_center_merger_ada, lane_right_merger_ada,
        force_target_lane=force_merger_ada,
        bypass_probe_guard=True
    )
    X0_merger_ada, X0_g_merger_ada = out_merger_ada["X0"], out_merger_ada["X0_g"]
    oa_merger_ada, od_merger_ada = out_merger_ada["oa"], out_merger_ada["od"]
    last_X_merger_ada, path_changed_merger_ada = out_merger_ada["last_X"], out_merger_ada["path_changed"]
    if force_merger_ada is not None:
        tgt = int(force_merger_ada)
        if lane_from_global(X0_g_merger_ada) != tgt:
            blend = MERGER_DREAM_ASSIST_BLEND if tgt == 1 else KEEP_LANE_ASSIST_BLEND
            X0_merger_ada, X0_g_merger_ada = blend_state_toward_lane(
                X0_merger_ada, X0_g_merger_ada, tgt, blend)

    # ------------------------------------------------------------------------
    # Build APF world lanes.
    # ------------------------------------------------------------------------
    merger_apf_row = state_to_row(X0_merger_apf, X0_g_merger_apf)
    merger_apf_lane = lane_from_global(X0_g_merger_apf)
    lane_left_apf    = stack_rows([_bl_row] + ([merger_apf_row] if merger_apf_lane == 0 else []))
    lane_center_apf  = stack_rows([truck_row_apf] + ([merger_apf_row] if merger_apf_lane == 1 else []))
    lane_right_apf   = stack_rows([merger_apf_row] if merger_apf_lane == 2 else [])

    # ------------------------------------------------------------------------
    # DRIFT step (APF) for APF world.
    # ------------------------------------------------------------------------
    ego_drift_apf_v = drift_create_vehicle(
        vid=0, x=X0_g_apf[0], y=X0_g_apf[1],
        vx=X0_apf[0] * math.cos(X0_g_apf[2]) - X0_apf[1] * math.sin(X0_g_apf[2]),
        vy=X0_apf[0] * math.sin(X0_g_apf[2]) + X0_apf[1] * math.cos(X0_g_apf[2]),
        vclass="car"
    )
    ego_drift_apf_v["heading"] = X0_g_apf[2]
    drift_vehicles_apf = [
        row_to_drift_vehicle(truck_row_apf, vid=1, vclass="truck"),
        row_to_drift_vehicle(merger_apf_row, vid=2, vclass="car"),
        blocker.to_drift_vehicle(vid=3),
    ]
    risk_field_apf = drift_apf.step(drift_vehicles_apf, ego_drift_apf_v,
                                    dt=dt, substeps=3, source_fn=compute_Q_APF)

    # ------------------------------------------------------------------------
    # APF ego (risk-aware, APF source).
    # ------------------------------------------------------------------------
    ego_blocker_gap_apf = float(blocker.s) - float(X0_apf[3])
    ego_apf_ready_for_center = (
        i >= EGO_FORCE_CENTER_MIN_STEP and
        X0_apf[3] > float(truck_row_apf[0]) + EGO_LC_OVERTAKE_MARGIN and
        0.0 < ego_blocker_gap_apf < EGO_BLOCKER_TRIGGER_GAP
    )
    lane_apf_now = lane_from_global(X0_g_apf)
    if lane_apf_now == 0:
        force_ego_apf = 1 if ego_apf_ready_for_center else 0
    else:
        force_ego_apf = None
    out_apf = dream_agent_step(
        X0_apf, X0_g_apf, oa_apf, od_apf, last_X_apf, path_changed_apf,
        apf_controller, apf_utils, decision_apf, dynamics,
        lane_left_apf, lane_center_apf, lane_right_apf,
        enable_decision_veto=config_integration.enable_decision_veto,
        force_target_lane=force_ego_apf,
        bypass_probe_guard=True,
        force_ignore_veto=True
    )
    X0_apf, X0_g_apf = out_apf["X0"], out_apf["X0_g"]
    oa_apf, od_apf = out_apf["oa"], out_apf["od"]
    last_X_apf, path_changed_apf = out_apf["last_X"], out_apf["path_changed"]
    if force_ego_apf is not None:
        tgt = int(force_ego_apf)
        if lane_from_global(X0_g_apf) != tgt:
            blend = EGO_DREAM_ASSIST_BLEND if tgt == 1 else KEEP_LANE_ASSIST_BLEND
            X0_apf, X0_g_apf = blend_state_toward_lane(X0_apf, X0_g_apf, tgt, blend)

    # ------------------------------------------------------------------------
    # APF-world merger (IDEAM).
    # ------------------------------------------------------------------------
    apf_row = state_to_row(X0_apf, X0_g_apf)
    apf_lane = lane_from_global(X0_g_apf)
    lane_left_merger_apf   = stack_rows([_bl_row] + ([apf_row] if apf_lane == 0 else []))
    lane_center_merger_apf = stack_rows([truck_row_apf] + ([apf_row] if apf_lane == 1 else []))
    lane_right_merger_apf  = stack_rows([apf_row] if apf_lane == 2 else [])

    merger_apf_ready_for_center = (
        i >= MERGER_FORCE_CENTER_MIN_STEP and
        X0_merger_apf[3] > float(truck_row_apf[0]) + MERGER_LC_OVERTAKE_MARGIN and
        lane_from_global(X0_g_apf) == 1
    )
    lane_merger_apf_now = lane_from_global(X0_g_merger_apf)
    if lane_merger_apf_now == 2:
        force_merger_apf = 1 if merger_apf_ready_for_center else 2
    else:
        force_merger_apf = None
    out_merger_apf = ideam_agent_step(
        X0_merger_apf, X0_g_merger_apf,
        oa_merger_apf, od_merger_apf, last_X_merger_apf, path_changed_merger_apf,
        apf_merger_mpc, apf_merger_utils, decision_merger_apf, dynamics,
        lane_left_merger_apf, lane_center_merger_apf, lane_right_merger_apf,
        force_target_lane=force_merger_apf,
        bypass_probe_guard=True
    )
    X0_merger_apf, X0_g_merger_apf = out_merger_apf["X0"], out_merger_apf["X0_g"]
    oa_merger_apf, od_merger_apf = out_merger_apf["oa"], out_merger_apf["od"]
    last_X_merger_apf, path_changed_merger_apf = out_merger_apf["last_X"], out_merger_apf["path_changed"]
    if force_merger_apf is not None:
        tgt = int(force_merger_apf)
        if lane_from_global(X0_g_merger_apf) != tgt:
            blend = MERGER_DREAM_ASSIST_BLEND if tgt == 1 else KEEP_LANE_ASSIST_BLEND
            X0_merger_apf, X0_g_merger_apf = blend_state_toward_lane(
                X0_merger_apf, X0_g_merger_apf, tgt, blend)

    # ------------------------------------------------------------------------
    # Update trucks in their own worlds.
    # ------------------------------------------------------------------------
    leaders_center_base = []
    if lane_from_global(X0_g_base) == 1:
        leaders_center_base.append(state_to_row(X0_base, X0_g_base))
    if lane_from_global(X0_g_merger_base) == 1:
        leaders_center_base.append(state_to_row(X0_merger_base, X0_g_merger_base))
    truck_row_base = update_truck_state(truck_row_base, truck_dyn, leaders_center_base)

    leaders_center_dream = []
    if lane_from_global(X0_g_dream) == 1:
        leaders_center_dream.append(state_to_row(X0_dream, X0_g_dream))
    if lane_from_global(X0_g_merger_dream) == 1:
        leaders_center_dream.append(state_to_row(X0_merger_dream, X0_g_merger_dream))
    truck_row_dream = update_truck_state(truck_row_dream, truck_dyn, leaders_center_dream)

    leaders_center_ada = []
    if lane_from_global(X0_g_ada) == 1:
        leaders_center_ada.append(state_to_row(X0_ada, X0_g_ada))
    if lane_from_global(X0_g_merger_ada) == 1:
        leaders_center_ada.append(state_to_row(X0_merger_ada, X0_g_merger_ada))
    truck_row_ada = update_truck_state(truck_row_ada, truck_dyn, leaders_center_ada)

    leaders_center_apf = []
    if lane_from_global(X0_g_apf) == 1:
        leaders_center_apf.append(state_to_row(X0_apf, X0_g_apf))
    if lane_from_global(X0_g_merger_apf) == 1:
        leaders_center_apf.append(state_to_row(X0_merger_apf, X0_g_merger_apf))
    truck_row_apf = update_truck_state(truck_row_apf, truck_dyn, leaders_center_apf)

    # ------------------------------------------------------------------------
    # Metrics (sample after all agent/state updates).
    # ------------------------------------------------------------------------
    blocker_row_now = blocker.to_mpc_row()
    merger_row_now_base  = state_to_row(X0_merger_base, X0_g_merger_base)
    merger_row_now_dream = state_to_row(X0_merger_dream, X0_g_merger_dream)
    merger_row_now_ada   = state_to_row(X0_merger_ada, X0_g_merger_ada)
    merger_row_now_apf   = state_to_row(X0_merger_apf, X0_g_merger_apf)

    time_hist.append(i * dt)

    base_s.append(progress_on_reference(X0_g_base))
    base_vx.append(float(X0_base[0]))
    base_lane_hist.append(int(lane_from_global(X0_g_base)))
    base_merger_lane_hist.append(int(lane_from_global(X0_g_merger_base)))
    base_dist_merger.append(float(math.hypot(
        float(X0_g_base[0]) - float(merger_row_now_base[3]),
        float(X0_g_base[1]) - float(merger_row_now_base[4]),
    )))
    base_min_dist_sur.append(min_center_distance(
        X0_g_base, [truck_row_base, merger_row_now_base, blocker_row_now]
    ))
    if hasattr(oa_base, "__len__"):
        base_acc.append(float(oa_base[0]) if len(oa_base) > 0 else 0.0)
    else:
        base_acc.append(float(oa_base))

    dream_s.append(progress_on_reference(X0_g_dream))
    dream_vx.append(float(X0_dream[0]))
    dream_lane_hist.append(int(lane_from_global(X0_g_dream)))
    dream_merger_lane_hist.append(int(lane_from_global(X0_g_merger_dream)))
    dream_dist_merger.append(float(math.hypot(
        float(X0_g_dream[0]) - float(merger_row_now_dream[3]),
        float(X0_g_dream[1]) - float(merger_row_now_dream[4]),
    )))
    dream_min_dist_sur.append(min_center_distance(
        X0_g_dream, [truck_row_dream, merger_row_now_dream, blocker_row_now]
    ))
    if hasattr(oa_dream, "__len__"):
        dream_acc.append(float(oa_dream[0]) if len(oa_dream) > 0 else 0.0)
    else:
        dream_acc.append(float(oa_dream))

    try:
        risk_at_ego_hist.append(float(drift.get_risk_cartesian(float(X0_g_dream[0]), float(X0_g_dream[1]))))
    except Exception:
        risk_at_ego_hist.append(float("nan"))

    ada_s.append(progress_on_reference(X0_g_ada))
    ada_vx.append(float(X0_ada[0]))
    ada_lane_hist.append(int(lane_from_global(X0_g_ada)))
    ada_merger_lane_hist.append(int(lane_from_global(X0_g_merger_ada)))
    ada_dist_merger.append(float(math.hypot(
        float(X0_g_ada[0]) - float(merger_row_now_ada[3]),
        float(X0_g_ada[1]) - float(merger_row_now_ada[4]),
    )))
    ada_min_dist_sur.append(min_center_distance(
        X0_g_ada, [truck_row_ada, merger_row_now_ada, blocker_row_now]
    ))
    if hasattr(oa_ada, "__len__"):
        ada_acc.append(float(oa_ada[0]) if len(oa_ada) > 0 else 0.0)
    else:
        ada_acc.append(float(oa_ada))
    try:
        risk_at_ego_ada_hist.append(float(drift_ada.get_risk_cartesian(
            float(X0_g_ada[0]), float(X0_g_ada[1]))))
    except Exception:
        risk_at_ego_ada_hist.append(float("nan"))

    apf_s.append(progress_on_reference(X0_g_apf))
    apf_vx.append(float(X0_apf[0]))
    apf_lane_hist.append(int(lane_from_global(X0_g_apf)))
    apf_merger_lane_hist.append(int(lane_from_global(X0_g_merger_apf)))
    apf_dist_merger.append(float(math.hypot(
        float(X0_g_apf[0]) - float(merger_row_now_apf[3]),
        float(X0_g_apf[1]) - float(merger_row_now_apf[4]),
    )))
    apf_min_dist_sur.append(min_center_distance(
        X0_g_apf, [truck_row_apf, merger_row_now_apf, blocker_row_now]
    ))
    if hasattr(oa_apf, "__len__"):
        apf_acc.append(float(oa_apf[0]) if len(oa_apf) > 0 else 0.0)
    else:
        apf_acc.append(float(oa_apf))
    try:
        risk_at_ego_apf_hist.append(float(drift_apf.get_risk_cartesian(
            float(X0_g_apf[0]), float(X0_g_apf[1]))))
    except Exception:
        risk_at_ego_apf_hist.append(float("nan"))

    # ------------------------------------------------------------------------
    # Horizons for plotting
    # ------------------------------------------------------------------------
    h_base = build_horizon(
        X0_base, X0_g_base, oa_base, od_base,
        out_base["path_d"], out_base["sample"], out_base["x_list"], out_base["y_list"],
        dynamics, dt, boundary
    )
    h_dream = build_horizon(
        X0_dream, X0_g_dream, oa_dream, od_dream,
        out_dream["path_d"], out_dream["sample"], out_dream["x_list"], out_dream["y_list"],
        dynamics, dt, boundary
    )
    h_ada = build_horizon(
        X0_ada, X0_g_ada, oa_ada, od_ada,
        out_ada["path_d"], out_ada["sample"], out_ada["x_list"], out_ada["y_list"],
        dynamics, dt, boundary
    )
    h_apf = build_horizon(
        X0_apf, X0_g_apf, oa_apf, od_apf,
        out_apf["path_d"], out_apf["sample"], out_apf["x_list"], out_apf["y_list"],
        dynamics, dt, boundary
    )

    # ------------------------------------------------------------------------
    # Plot (four panels: IDEAM | DREAM-GVF | DREAM-ADA | DREAM-APF)
    # ------------------------------------------------------------------------
    fig = plt.gcf()
    fig.clf()
    ax_p1 = fig.add_subplot(1, 4, 1)
    ax_p2 = fig.add_subplot(1, 4, 2)
    ax_p3 = fig.add_subplot(1, 4, 3)
    ax_p4 = fig.add_subplot(1, 4, 4)

    xr_base  = [X0_g_base[0]  - x_area, X0_g_base[0]  + x_area]
    yr_base  = [X0_g_base[1]  - y_area, X0_g_base[1]  + y_area]
    xr_dream = [X0_g_dream[0] - x_area, X0_g_dream[0] + x_area]
    yr_dream = [X0_g_dream[1] - y_area, X0_g_dream[1] + y_area]
    xr_ada   = [X0_g_ada[0]   - x_area, X0_g_ada[0]   + x_area]
    yr_ada   = [X0_g_ada[1]   - y_area, X0_g_ada[1]   + y_area]
    xr_apf   = [X0_g_apf[0]   - x_area, X0_g_apf[0]   + x_area]
    yr_apf   = [X0_g_apf[1]   - y_area, X0_g_apf[1]   + y_area]

    merger_row_plot_base  = state_to_row(X0_merger_base,  X0_g_merger_base)
    merger_row_plot_dream = state_to_row(X0_merger_dream, X0_g_merger_dream)
    merger_row_plot_ada   = state_to_row(X0_merger_ada,   X0_g_merger_ada)
    merger_row_plot_apf   = state_to_row(X0_merger_apf,   X0_g_merger_apf)

    draw_panel(
        ax_p1, X0_base, X0_g_base, truck_row_base, merger_row_plot_base, _bl_row,
        title="Baseline MPC-CBF",
        x_range=xr_base, y_range=yr_base,
        risk_field=None, horizon=h_base, ego_color=EGO_IDEAM_COLOR
    )
    cf_gvf = draw_panel(
        ax_p2, X0_dream, X0_g_dream, truck_row_dream, merger_row_plot_dream, _bl_row,
        title="DREAM (GVF-DRIFT)",
        x_range=xr_dream, y_range=yr_dream,
        risk_field=risk_field, horizon=h_dream, ego_color=EGO_DREAM_COLOR
    )
    cf_ada = draw_panel(
        ax_p3, X0_ada, X0_g_ada, truck_row_ada, merger_row_plot_ada, _bl_row,
        title="DREAM (ADA-DRIFT)",
        x_range=xr_ada, y_range=yr_ada,
        risk_field=risk_field_ada, horizon=h_ada, ego_color=EGO_ADA_COLOR
    )
    cf_apf = draw_panel(
        ax_p4, X0_apf, X0_g_apf, truck_row_apf, merger_row_plot_apf, _bl_row,
        title="DREAM (APF-DRIFT)",
        x_range=xr_apf, y_range=yr_apf,
        risk_field=risk_field_apf, horizon=h_apf, ego_color=EGO_APF_COLOR
    )
    if cf_gvf is not None:
        cbar = fig.colorbar(cf_gvf, ax=ax_p2, orientation="vertical", pad=0.02, fraction=0.035)
        cbar.set_label("Risk Level", fontsize=9, weight="bold")
        cbar.ax.tick_params(labelsize=8, colors="black")
    if cf_ada is not None:
        cbar2 = fig.colorbar(cf_ada, ax=ax_p3, orientation="vertical", pad=0.02, fraction=0.035)
        cbar2.set_label("Risk Level", fontsize=9, weight="bold")
        cbar2.ax.tick_params(labelsize=8, colors="black")
    if cf_apf is not None:
        cbar3 = fig.colorbar(cf_apf, ax=ax_p4, orientation="vertical", pad=0.02, fraction=0.035)
        cbar3.set_label("Risk Level", fontsize=9, weight="bold")
        cbar3.ax.tick_params(labelsize=8, colors="black")

    if SAVE_FRAMES:
        plt.savefig(os.path.join(save_dir, f"{i}.png"), dpi=SAVE_DPI)

    if i % max(1, LOG_EVERY) == 0:
        print(
            f"[{i:03d}] base_lane={lane_from_global(X0_g_base)} "
            f"dream_lane={lane_from_global(X0_g_dream)} "
            f"ada_lane={lane_from_global(X0_g_ada)} "
            f"apf_lane={lane_from_global(X0_g_apf)} "
            f"base_merger={lane_from_global(X0_g_merger_base)} "
            f"dream_merger={lane_from_global(X0_g_merger_dream)} "
            f"ada_merger={lane_from_global(X0_g_merger_ada)} "
            f"apf_merger={lane_from_global(X0_g_merger_apf)}"
        )
        if not out_base["ok"]:
            print(f"  [warn] baseline fallback: {out_base.get('error','')}")
        if not out_dream["ok"]:
            print(f"  [warn] dream fallback: {out_dream.get('error','')}")
        if not out_ada["ok"]:
            print(f"  [warn] ada fallback: {out_ada.get('error','')}")
        if not out_apf["ok"]:
            print(f"  [warn] apf fallback: {out_apf.get('error','')}")
        if not out_merger_base["ok"]:
            print(f"  [warn] base merger fallback: {out_merger_base.get('error','')}")
        if not out_merger_dream["ok"]:
            print(f"  [warn] dream merger fallback: {out_merger_dream.get('error','')}")
        if not out_merger_ada["ok"]:
            print(f"  [warn] ada merger fallback: {out_merger_ada.get('error','')}")
        if not out_merger_apf["ok"]:
            print(f"  [warn] apf merger fallback: {out_merger_apf.get('error','')}")

bar.finish()
print()
print("Simulation complete.")
if SAVE_FRAMES:
    print(f"Frames saved to: {save_dir}")
else:
    print("Frame saving disabled (--save-frames false).")

# ============================================================================
# METRICS SUMMARY FIGURE (SciencePlots style)
# ============================================================================
with plt.style.context(["science", "no-latex"]):
    fig_m, axes_m = plt.subplots(3, 2, figsize=(11, 12), constrained_layout=True)
    fig_m.suptitle("Uncertainty Merger: IDEAM vs DREAM-GVF vs DREAM-ADA vs DREAM-APF", fontsize=13)

    _C  = {"IDEAM": "C1", "DREAM": "C0", "ADA": "#9C27B0", "APF": "#009688"}
    _LS = {"IDEAM": "--",  "DREAM": "-",   "ADA": "-.",     "APF": ":"}
    _t = np.asarray(time_hist, dtype=float)
    _risk_t     = _t[:len(risk_at_ego_hist)]
    _risk_ada_t = _t[:len(risk_at_ego_ada_hist)]
    _risk_apf_t = _t[:len(risk_at_ego_apf_hist)]

    # (0,0) Progress
    ax = axes_m[0, 0]
    ax.plot(_t, base_s,  color=_C["IDEAM"], ls=_LS["IDEAM"], label="IDEAM")
    ax.plot(_t, dream_s, color=_C["DREAM"], ls=_LS["DREAM"], label="DREAM (GVF)")
    ax.plot(_t, ada_s,   color=_C["ADA"],   ls=_LS["ADA"],   label="DREAM (ADA)")
    ax.plot(_t, apf_s,   color=_C["APF"],   ls=_LS["APF"],   label="DREAM (APF)")
    ax.set_xlabel("t [s]"); ax.set_ylabel("s [m]"); ax.set_title("Progress s(t)")
    ax.legend(fontsize=8)

    # (0,1) Speed
    ax = axes_m[0, 1]
    ax.plot(_t, base_vx,  color=_C["IDEAM"], ls=_LS["IDEAM"], label="IDEAM")
    ax.plot(_t, dream_vx, color=_C["DREAM"], ls=_LS["DREAM"], label="DREAM (GVF)")
    ax.plot(_t, ada_vx,   color=_C["ADA"],   ls=_LS["ADA"],   label="DREAM (ADA)")
    ax.plot(_t, apf_vx,   color=_C["APF"],   ls=_LS["APF"],   label="DREAM (APF)")
    ax.set_xlabel("t [s]"); ax.set_ylabel("vx [m/s]"); ax.set_title("Speed vx(t)")
    ax.legend(fontsize=8)

    # (1,0) Ego-to-merger distance
    ax = axes_m[1, 0]
    ax.plot(_t, base_dist_merger,  color=_C["IDEAM"], ls=_LS["IDEAM"], label="IDEAM")
    ax.plot(_t, dream_dist_merger, color=_C["DREAM"], ls=_LS["DREAM"], label="DREAM (GVF)")
    ax.plot(_t, ada_dist_merger,   color=_C["ADA"],   ls=_LS["ADA"],   label="DREAM (ADA)")
    ax.plot(_t, apf_dist_merger,   color=_C["APF"],   ls=_LS["APF"],   label="DREAM (APF)")
    ax.axhline(NEAR_COLLISION_DIST, color="orange", lw=1.0, ls="--",
               label=f"Near collision ({NEAR_COLLISION_DIST}m)")
    ax.axhline(COLLISION_DIST, color="red", lw=1.0, ls="--",
               label=f"Collision ({COLLISION_DIST}m)")
    ax.set_xlabel("t [s]"); ax.set_ylabel("distance [m]"); ax.set_title("Distance to Merger")
    ax.set_ylim(bottom=0.0); ax.legend(fontsize=7)

    # (1,1) DRIFT risk at ego (GVF vs ADA vs APF)
    ax = axes_m[1, 1]
    ax.plot(_risk_t, risk_at_ego_hist,
            color=_C["DREAM"], ls=_LS["DREAM"], label="GVF source")
    ax.fill_between(_risk_t, risk_at_ego_hist, color=_C["DREAM"], alpha=0.20)
    ax.plot(_risk_ada_t, risk_at_ego_ada_hist,
            color=_C["ADA"], ls=_LS["ADA"], label="ADA source")
    ax.plot(_risk_apf_t, risk_at_ego_apf_hist,
            color=_C["APF"], ls=_LS["APF"], label="APF source")
    ax.set_xlabel("t [s]"); ax.set_ylabel("R(ego)"); ax.set_title("DRIFT Risk at Ego")
    ax.legend(fontsize=8)

    # (2,0) Longitudinal acceleration
    ax = axes_m[2, 0]
    ax.plot(_t, base_acc,  color=_C["IDEAM"], ls=_LS["IDEAM"], label="IDEAM")
    ax.plot(_t, dream_acc, color=_C["DREAM"], ls=_LS["DREAM"], label="DREAM (GVF)")
    ax.plot(_t, ada_acc,   color=_C["ADA"],   ls=_LS["ADA"],   label="DREAM (ADA)")
    ax.plot(_t, apf_acc,   color=_C["APF"],   ls=_LS["APF"],   label="DREAM (APF)")
    ax.axhline(0.0, color="black", lw=0.6)
    ax.set_xlabel("t [s]"); ax.set_ylabel("ax [m/s^2]")
    ax.set_title("Longitudinal Acceleration"); ax.legend(fontsize=8)

    # (2,1) Minimum spacing to surrounding vehicles
    ax = axes_m[2, 1]
    ax.plot(_t, base_min_dist_sur,  color=_C["IDEAM"], ls=_LS["IDEAM"], label="IDEAM")
    ax.plot(_t, dream_min_dist_sur, color=_C["DREAM"], ls=_LS["DREAM"], label="DREAM (GVF)")
    ax.plot(_t, ada_min_dist_sur,   color=_C["ADA"],   ls=_LS["ADA"],   label="DREAM (ADA)")
    ax.plot(_t, apf_min_dist_sur,   color=_C["APF"],   ls=_LS["APF"],   label="DREAM (APF)")
    ax.axhline(NEAR_COLLISION_DIST, color="orange", lw=0.9, ls=":")
    ax.set_xlabel("t [s]"); ax.set_ylabel("min center dist [m]")
    ax.set_title("Minimum Distance to Nearby Vehicles")
    ax.set_ylim(bottom=0.0); ax.legend(fontsize=8)

    metrics_png = os.path.join(save_dir, "metrics_uncertainty_merger.png")
    plt.savefig(metrics_png, dpi=300, bbox_inches="tight")
    plt.close(fig_m)

metrics_npy = os.path.join(save_dir, "metrics_uncertainty_merger.npy")
np.save(metrics_npy, {
    "time": np.asarray(time_hist, dtype=float),
    "base_s": np.asarray(base_s, dtype=float),
    "dream_s": np.asarray(dream_s, dtype=float),
    "ada_s": np.asarray(ada_s, dtype=float),
    "apf_s": np.asarray(apf_s, dtype=float),
    "base_vx": np.asarray(base_vx, dtype=float),
    "dream_vx": np.asarray(dream_vx, dtype=float),
    "ada_vx": np.asarray(ada_vx, dtype=float),
    "apf_vx": np.asarray(apf_vx, dtype=float),
    "base_acc": np.asarray(base_acc, dtype=float),
    "dream_acc": np.asarray(dream_acc, dtype=float),
    "ada_acc": np.asarray(ada_acc, dtype=float),
    "apf_acc": np.asarray(apf_acc, dtype=float),
    "base_lane": np.asarray(base_lane_hist, dtype=int),
    "dream_lane": np.asarray(dream_lane_hist, dtype=int),
    "ada_lane": np.asarray(ada_lane_hist, dtype=int),
    "apf_lane": np.asarray(apf_lane_hist, dtype=int),
    "base_merger_lane": np.asarray(base_merger_lane_hist, dtype=int),
    "dream_merger_lane": np.asarray(dream_merger_lane_hist, dtype=int),
    "ada_merger_lane": np.asarray(ada_merger_lane_hist, dtype=int),
    "apf_merger_lane": np.asarray(apf_merger_lane_hist, dtype=int),
    "base_dist_merger": np.asarray(base_dist_merger, dtype=float),
    "dream_dist_merger": np.asarray(dream_dist_merger, dtype=float),
    "ada_dist_merger": np.asarray(ada_dist_merger, dtype=float),
    "apf_dist_merger": np.asarray(apf_dist_merger, dtype=float),
    "base_min_dist_sur": np.asarray(base_min_dist_sur, dtype=float),
    "dream_min_dist_sur": np.asarray(dream_min_dist_sur, dtype=float),
    "ada_min_dist_sur": np.asarray(ada_min_dist_sur, dtype=float),
    "apf_min_dist_sur": np.asarray(apf_min_dist_sur, dtype=float),
    "risk_at_ego": np.asarray(risk_at_ego_hist, dtype=float),
    "risk_at_ego_ada": np.asarray(risk_at_ego_ada_hist, dtype=float),
    "risk_at_ego_apf": np.asarray(risk_at_ego_apf_hist, dtype=float),
    "near_collision_dist": float(NEAR_COLLISION_DIST),
    "collision_dist": float(COLLISION_DIST),
})

_base_min_merger  = float(np.nanmin(base_dist_merger))  if len(base_dist_merger)  else float("nan")
_dream_min_merger = float(np.nanmin(dream_dist_merger)) if len(dream_dist_merger) else float("nan")
_ada_min_merger   = float(np.nanmin(ada_dist_merger))   if len(ada_dist_merger)   else float("nan")
_apf_min_merger   = float(np.nanmin(apf_dist_merger))   if len(apf_dist_merger)   else float("nan")
_base_min_sur     = float(np.nanmin(base_min_dist_sur)) if len(base_min_dist_sur) else float("nan")
_dream_min_sur    = float(np.nanmin(dream_min_dist_sur))if len(dream_min_dist_sur)else float("nan")
_ada_min_sur      = float(np.nanmin(ada_min_dist_sur))  if len(ada_min_dist_sur)  else float("nan")
_apf_min_sur      = float(np.nanmin(apf_min_dist_sur))  if len(apf_min_dist_sur)  else float("nan")

print(f"Metrics plot: {metrics_png}")
print(f"Metrics data: {metrics_npy}")
print(f"Min ego-merger distance   IDEAM={_base_min_merger:.2f}m  DREAM(GVF)={_dream_min_merger:.2f}m  DREAM(ADA)={_ada_min_merger:.2f}m  DREAM(APF)={_apf_min_merger:.2f}m")
print(f"Min ego-surround distance IDEAM={_base_min_sur:.2f}m  DREAM(GVF)={_dream_min_sur:.2f}m  DREAM(ADA)={_ada_min_sur:.2f}m  DREAM(APF)={_apf_min_sur:.2f}m")
