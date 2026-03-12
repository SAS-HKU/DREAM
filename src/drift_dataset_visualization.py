"""
uncertainty_DREAM_dataset.py
============================
DREAM vs IDEAM benchmark on real naturalistic vehicle trajectories.

Surrounding vehicles replay real recorded behaviour from the exiD / inD /
rounD datasets (CSV format).  Lane reference paths are fitted automatically
from the dominant vehicle flow via K-means clustering + cubic splines.
The ego vehicle is controlled by DREAM (DRIFT + IDEAM MPC) and an independent
IDEAM baseline simultaneously, starting from the recorded position of a chosen
dataset vehicle.

Usage
-----
Set DATASET_DIR, RECORDING_ID, and optionally EGO_TRACK_ID below, then run:
    python uncertainty_DREAM_dataset.py
"""

# ===========================================================================
# IMPORTS
# ===========================================================================
import sys
import os
import json
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
from scipy.interpolate import make_interp_spline
from sklearn.cluster import KMeans
import cv2
import scienceplots  # noqa: F401

sys.path.insert(0, ".")
sys.path.append("C:\\IDEAM_implementation-main\\decision_improve")

# IDEAM imports
from Control.MPC import *
from Control.constraint_params import *
from Model.Dynamical_model import *
from Model.params import *
from Model.surrounding_params import *
from Model.Surrounding_model import *
from Control.HOCBF import *
from DecisionMaking.decision_params import *
from DecisionMaking.give_desired_path import *
from DecisionMaking.util import *
import DecisionMaking.util as _dm_util_module
from DecisionMaking.util_params import *
from DecisionMaking.decision import *
from Prediction.surrounding_prediction import *
from progress.bar import Bar

# DRIFT / PRIDEAM
from config import Config as cfg
from pde_solver import (PDESolver, compute_total_Q, compute_velocity_field,
                        compute_diffusion_field,
                        create_vehicle as drift_create_vehicle)
from Integration.drift_interface import DRIFTInterface
from Integration.prideam_controller import create_prideam_controller
from Integration.integration_config import get_preset

# Dataset loader
from tracks_import import read_from_csv

# ===========================================================================
# PARAMETERS edit these to change scenario
# ===========================================================================

DATASET_DIR    = r"c:\field_modeling\data\rounD"
RECORDING_ID   = "03"            # zero-padded recording number, e.g. "00", "26"
EGO_TRACK_ID   = 45 # inD set 03: (track 49 58 66 88 95 115 148); inD set 32: (track 22 28) rounD set 03: (track 45 106)             # None = auto-select longest central car track
EGO_LANE       = 1               # Lane the ego starts on (0=left, 1=centre, 2=right)
N_LANES        = 3               # number of lanes to fit
LANE_WIDTH     = 3.75            # m used for parallel-path fallback
EGO_MIN_FRAMES = 80              # minimum track length (frames) for ego candidate

dt             = 0.1             # simulation step [s]
N_t            = 400             # number of simulation steps (20 s)
WARMUP_S       = 5.0             # DRIFT warm-up duration [s]

INTEGRATION_MODE  = "conservative"
DRIFT_CELL_M      = 2.0          # DRIFT grid cell size [m]
SCENE_MARGIN      = 60.0         # margin beyond track bbox [m]
SCENARIO_MODE     = "drift_overlay"   # "drift_overlay" or "benchmark"

# Risk visualisation
RISK_ALPHA   = 0.24    # max overlay alpha; keep road + vehicles readable
RISK_CMAP    = "jet"
RISK_LEVELS  = 80
RISK_VMAX    = 2.0
RISK_MIN_VIS = 0.08    # values below this are fully masked → road visible there
RISK_SMOOTH_SIGMA = 2.0
RISK_ALPHA_GAMMA = 0.8

# Optional manual map calibration (meters in world frame)
BG_SCALE_MULT = 1.0
BG_OFFSET_X_M = 0.0
BG_OFFSET_Y_M = 0.0
BG_AUTO_CENTER_SMALL_MAP = True  # helps inD/rounD backgrounds with local offsets

# Viewport half-extents around ego (m)
VIEW_X = 50.0   # match run_simulation.py x_area
VIEW_Y = 28.0   # match run_simulation.py y_area

# Visualization style
VIEW_STYLE = "run_simulation"   # "run_simulation" or "dataset_map"
USE_DATASET_BACKGROUND = True   # show orthophoto background (pixel-space aligned)
# Pixel-space scale used by TrackVisualizer (manual override; None=auto/preset)
VIS_SCALE_DOWN = None
# Known dataset visualization scales (same convention as TrackVisualizer).
VIS_SCALE_PRESETS = {"exid": 6.0, "ind": 12.0, "round": 10.0}
# Optional explicit path to visualizer_params.json.
VISUALIZER_PARAMS_PATH = None

save_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                        ("figsave_DRIFT_dataset_rounD_03"
                         if SCENARIO_MODE == "drift_overlay"
                         else "figsave_DREAM_dataset"))
os.makedirs(save_dir, exist_ok=True)

# ===========================================================================
# DATASET PATH CLASS cubic-spline path compatible with IDEAM interface
# ===========================================================================

class DatasetPath:
    """
    Cubic parametric spline path fitted to vehicle trajectory data.

    Implements the IDEAM path interface:
        path(s)               (x, y)  [called as path.__call__]
        path.get_theta_r(s)   heading [radians, math convention]
        path.get_cartesian_coords(s, ey) (x, y) at lateral offset

    Pre-samples (self.xc, self.yc, self.samples) are stored for direct use
    with IDEAM's find_frenet_coord(path, xc, yc, samples, pos).
    """

    def __init__(self, x_pts, y_pts, n_samples=600):
        # Remove duplicate/near-duplicate points
        diffs  = np.hypot(np.diff(x_pts), np.diff(y_pts))
        keep   = np.concatenate([[True], diffs > 0.05])
        x_pts  = np.asarray(x_pts)[keep]
        y_pts  = np.asarray(y_pts)[keep]

        if len(x_pts) < 4:
            raise ValueError("DatasetPath: need 4 unique control points")

        # Cumulative arc length parameterisation
        diffs  = np.hypot(np.diff(x_pts), np.diff(y_pts))
        s_raw  = np.concatenate([[0.0], np.cumsum(diffs)])

        self._spl_x   = make_interp_spline(s_raw, x_pts, k=3)
        self._spl_y   = make_interp_spline(s_raw, y_pts, k=3)
        self._spl_dx  = self._spl_x.derivative()
        self._spl_dy  = self._spl_y.derivative()
        self._spl_d2x = self._spl_dx.derivative()   # second derivative (for curvature)
        self._spl_d2y = self._spl_dy.derivative()
        self.s_min    = float(s_raw[0])
        self.s_max    = float(s_raw[-1])

        # Pre-sample for find_frenet_coord
        self.samples = np.linspace(self.s_min, self.s_max, n_samples)
        self.xc = np.array([float(self._spl_x(s)) for s in self.samples])
        self.yc = np.array([float(self._spl_y(s)) for s in self.samples])

    @staticmethod
    def _to_float(s):
        """Extract a plain Python float from any numpy array, scalar, or number.
        Uses .flat[0] which works for 0-d arrays, 1-element arrays, and scalars.
        """
        return float(np.asarray(s).flat[0])

    @staticmethod
    def _scalar(v):
        """Same extraction for spline output values."""
        return float(np.asarray(v).flat[0])

    def _clamp(self, s):
        """Clamp s to [s_min, s_max] returning a plain float."""
        s = self._to_float(s)
        return max(self.s_min, min(s, self.s_max))

    def __call__(self, s):
        s = self._clamp(s)
        return self._scalar(self._spl_x(s)), self._scalar(self._spl_y(s))

    def get_theta_r(self, s):
        s  = self._clamp(s)
        dx = self._scalar(self._spl_dx(s))
        dy = self._scalar(self._spl_dy(s))
        return math.atan2(dy, dx)

    def get_len(self):
        """Total arc length of the path [m]. Required by IDEAM's MPC."""
        return self.s_max

    def get_k(self, s):
        """Signed curvature 魏(s) at arc length s. Required by IDEAM's MPC."""
        s   = self._clamp(s)
        dx  = self._scalar(self._spl_dx(s))
        dy  = self._scalar(self._spl_dy(s))
        d2x = self._scalar(self._spl_d2x(s))
        d2y = self._scalar(self._spl_d2y(s))
        denom = (dx**2 + dy**2) ** 1.5
        return (dx * d2y - dy * d2x) / denom if denom > 1e-9 else 0.0

    def get_cartesian_coords(self, s, ey):
        """Return Cartesian (x, y) at arc-length s and lateral offset ey."""
        x, y = self(s)
        psi  = self.get_theta_r(s)
        return x - ey * math.sin(psi), y + ey * math.cos(psi)


# ===========================================================================
# HELPERS 鈥?coordinate conversion
# ===========================================================================

def heading_to_psi(heading_deg):
    """
    Dataset heading → math heading in radians.

    The inD / exiD / rounD datasets store heading in standard math convention:
        0° = East (+X), CCW positive, same as atan2(yVelocity, xVelocity).
    Verified: exiD Track 0, heading=128.95° ≈ atan2(23.33, -18.85) = 129° ✓

    No conversion needed — return directly in radians.
    (The TrackVisualizer flips it via headingVis=-heading because its pixel
    axes have Y increasing downward; world-metric axes do not need that flip.)
    """
    return math.radians(float(heading_deg))


def draw_vehicle_rect(ax, x, y, psi, length, width,
                      facecolor, edgecolor="black", lw=0.8, zorder=3,
                      alpha=1.0, linestyle="-"):
    rect = mpatches.FancyBboxPatch(
        (-length / 2, -width / 2), length, width,
        boxstyle="round,pad=0.05",
        facecolor=facecolor, edgecolor=edgecolor,
        linewidth=lw, alpha=alpha, zorder=zorder, linestyle=linestyle)
    t = Affine2D().rotate(psi).translate(x, y) + ax.transData
    rect.set_transform(t)
    ax.add_patch(rect)


def _offset_path_xy(path, ey_offset):
    """Sample one offset polyline from a centerline path at constant ey offset."""
    xs, ys = [], []
    for s in path.samples:
        x_o, y_o = path.get_cartesian_coords(s, ey_offset)
        xs.append(float(x_o))
        ys.append(float(y_o))
    return np.asarray(xs), np.asarray(ys)


def draw_dataset_road_layout(ax):
    """Draw a visible road layout from fitted lane centerlines."""
    if _paths is None or len(_paths) == 0:
        return

    half = 0.5 * LANE_WIDTH

    # Approximate outer boundaries + separators from lane centerlines.
    b0x, b0y = _offset_path_xy(_paths[0], +half)   # outer-left
    b1x, b1y = _offset_path_xy(_paths[0], -half)   # separator 0/1
    b2x, b2y = _offset_path_xy(_paths[1], -half) if len(_paths) > 2 else _offset_path_xy(_paths[-1], -half)
    b3x, b3y = _offset_path_xy(_paths[-1], -half)  # outer-right

    # Road surface band.
    poly_x = np.concatenate([b0x, b3x[::-1]])
    poly_y = np.concatenate([b0y, b3y[::-1]])
    ax.fill(poly_x, poly_y, color="#2B2B2B", alpha=0.20, zorder=1.6)

    # Lane boundaries.
    ax.plot(b0x, b0y, color="white", lw=1.4, alpha=0.85, zorder=2.2)
    ax.plot(b3x, b3y, color="white", lw=1.4, alpha=0.85, zorder=2.2)
    ax.plot(b1x, b1y, color="white", lw=1.0, ls="--", alpha=0.65, zorder=2.2)
    if len(_paths) > 2:
        ax.plot(b2x, b2y, color="white", lw=1.0, ls="--", alpha=0.65, zorder=2.2)


def draw_road_runsim_style(ax):
    """Draw lane lines in a run_simulation-like style (blue solids + dashdot centers)."""
    if _paths is None or len(_paths) == 0:
        return

    half = 0.5 * LANE_WIDTH
    b0x, b0y = _offset_path_xy(_paths[0], +half)   # upper/left outer boundary
    b1x, b1y = _offset_path_xy(_paths[0], -half)   # separator 0/1
    b2x, b2y = _offset_path_xy(_paths[1], -half) if len(_paths) > 2 else _offset_path_xy(_paths[-1], -half)
    b3x, b3y = _offset_path_xy(_paths[-1], -half)  # lower/right outer boundary

    # Solid boundaries + dashed separators
    ax.plot(b0x, b0y, color="b", lw=1.0, zorder=2.0)
    ax.plot(b3x, b3y, color="b", lw=1.0, zorder=2.0)
    ax.plot(b1x, b1y, color="b", lw=0.9, ls="dashdot", zorder=2.0)
    if len(_paths) > 2:
        ax.plot(b2x, b2y, color="b", lw=0.9, ls="dashdot", zorder=2.0)

    # Lane centers (dashdot) to match run_simulation visual intent
    for p in _paths:
        ax.plot(p.xc, p.yc, color="b", lw=0.8, ls="dashdot", alpha=0.85, zorder=2.0)


# ===========================================================================
# EGO AUTO-SELECTION
# ===========================================================================

def select_ego_track(tracks, tracks_meta, class_map, min_frames, ego_lane,
                     heading_ref_deg=None):
    """
    Pick a car track that:
    - has at least min_frames frames
    - travels in the same direction as the majority of traffic (if heading_ref_deg given)
    - starts nearest to the scene centre (as tie-breaker)
    """
    scene_cx = float(np.mean([t["xCenter"].mean() for t in tracks]))
    scene_cy = float(np.mean([t["yCenter"].mean() for t in tracks]))

    candidates = []
    for tm in tracks_meta:
        tid = tm["trackId"]
        cls = class_map.get(tid, "car")
        if cls not in ("car", "van"):
            continue
        if tm["numFrames"] < min_frames:
            continue
        tr = tracks[tid]
        # Same direction filter
        if heading_ref_deg is not None:
            h_mean = float(np.nanmean(tr["heading"]))
            angle_diff = abs(((h_mean - heading_ref_deg + 180) % 360) - 180)
            if angle_diff > 45:
                continue
        dist = math.hypot(tr["xCenter"].mean() - scene_cx,
                          tr["yCenter"].mean() - scene_cy)
        candidates.append((dist, tid))

    if not candidates:
        # Fallback: longest track regardless of class/direction
        candidates = [(0, tm["trackId"])
                      for tm in tracks_meta
                      if tm["numFrames"] >= min_frames]

    candidates.sort(key=lambda p: p[0])
    return candidates[0][1]


# ===========================================================================
# LANE PATH FITTING
# ===========================================================================

def fit_lane_paths(tracks, tracks_meta, class_map, ego_track_id,
                   n_lanes=3, bin_width=5.0):
    """
    Cluster all same-direction vehicles into n_lanes groups by their
    road-normal position, then fit a cubic spline through each cluster.

    Returns list of DatasetPath objects sorted left鈫抮ight
    (leftmost lateral offset = index 0).
    """
    ego_tr  = tracks[ego_track_id]
    h_ref   = float(np.nanmean(ego_tr["heading"]))          # reference compass heading
    h_ref_r = math.radians(h_ref)
    cos_h, sin_h = math.cos(h_ref_r), math.sin(h_ref_r)

    # Collect all (s_road, n_road, x_orig, y_orig) for same-direction vehicles
    pts = []
    for tm in tracks_meta:
        tid = tm["trackId"]
        cls = class_map.get(tid, "car")
        if cls not in ("car", "van", "truck"):
            continue
        tr = tracks[tid]
        h_mean = float(np.nanmean(tr["heading"]))
        angle_diff = abs(((h_mean - h_ref + 180) % 360) - 180)
        if angle_diff > 40:
            continue
        for xi, yi in zip(tr["xCenter"], tr["yCenter"]):
            s_r =  cos_h * xi + sin_h * yi
            n_r = -sin_h * xi + cos_h * yi
            pts.append((s_r, n_r, xi, yi))

    if len(pts) < n_lanes * 10:
        print("[LANE FIT] Too few same-direction points; using parallel fallback.")
        return _parallel_fallback(tracks[ego_track_id], n_lanes)

    pts = np.array(pts)   # (N, 4): s_road, n_road, x_orig, y_orig
    n_vals = pts[:, 1]

    # K-means on n_road to separate lanes
    km = KMeans(n_clusters=n_lanes, random_state=42, n_init=10)
    labels = km.fit_predict(n_vals.reshape(-1, 1))
    centres = km.cluster_centers_.ravel()
    order   = np.argsort(centres)          # leftmost 鈫?rightmost

    paths = []
    for li in range(n_lanes):
        cluster_idx = order[li]
        mask = labels == cluster_idx
        cluster_pts = pts[mask]            # (M, 4)

        # Sort by s_road and bin every bin_width m (use median x/y in each bin)
        s_vals = cluster_pts[:, 0]
        s_min, s_max = s_vals.min(), s_vals.max()
        bins   = np.arange(s_min, s_max + bin_width, bin_width)
        x_bin, y_bin = [], []
        for b in bins[:-1]:
            mask_b = (s_vals >= b) & (s_vals < b + bin_width)
            if mask_b.sum() < 2:
                continue
            x_bin.append(float(np.median(cluster_pts[mask_b, 2])))
            y_bin.append(float(np.median(cluster_pts[mask_b, 3])))

        if len(x_bin) < 4:
            print(f"[LANE FIT] Lane {li}: too few bins ({len(x_bin)}); using fallback.")
            paths.append(None)
            continue

        try:
            dp = DatasetPath(np.array(x_bin), np.array(y_bin))
            print(f"[LANE FIT] Lane {li}: s_max={dp.s_max:.0f} m  "
                  f"n_road_centre={centres[cluster_idx]:.1f} m  "
                  f"control_pts={len(x_bin)}")
            paths.append(dp)
        except Exception as e:
            print(f"[LANE FIT] Lane {li}: spline failed ({e}); using fallback.")
            paths.append(None)

    # Replace failed lanes with parallel offsets from nearest good lane
    good = [p for p in paths if p is not None]
    if not good:
        print("[LANE FIT] All lanes failed; using parallel fallback.")
        return _parallel_fallback(tracks[ego_track_id], n_lanes)

    for li in range(n_lanes):
        if paths[li] is None:
            ref = good[0]
            offset = (li - n_lanes // 2) * LANE_WIDTH
            x_off = ref.xc - offset * math.sin(ref.get_theta_r(ref.samples[0]))
            y_off = ref.yc + offset * math.cos(ref.get_theta_r(ref.samples[0]))
            paths[li] = DatasetPath(x_off, y_off)

    return paths


def _parallel_fallback(ego_track, n_lanes):
    """Create n_lanes parallel straight paths from ego track average heading."""
    h_ref = math.radians(90.0 - float(np.nanmean(ego_track["heading"])))
    x0 = float(ego_track["xCenter"].mean())
    y0 = float(ego_track["yCenter"].mean())
    s_span = float(np.max(ego_track["xCenter"]) - np.min(ego_track["xCenter"]))
    s_span = max(s_span, 200.0)
    cos_h, sin_h = math.cos(h_ref), math.sin(h_ref)

    paths = []
    for li in range(n_lanes):
        offset = (li - n_lanes // 2) * LANE_WIDTH
        x_start = x0 - offset * sin_h - 0.5 * s_span * cos_h
        y_start = y0 + offset * cos_h - 0.5 * s_span * sin_h
        x_end   = x0 - offset * sin_h + 0.5 * s_span * cos_h
        y_end   = y0 + offset * cos_h + 0.5 * s_span * sin_h
        n_pts   = max(10, int(s_span / 5))
        x_pts   = np.linspace(x_start, x_end, n_pts)
        y_pts   = np.linspace(y_start, y_end, n_pts)
        paths.append(DatasetPath(x_pts, y_pts))
        print(f"[LANE FIT] Fallback lane {li}: straight path, s_max={paths[-1].s_max:.0f} m")

    return paths


# ===========================================================================
# DATASET-AWARE REPLACEMENTS FOR IDEAM GLOBAL FUNCTIONS
# ===========================================================================

# These are set after path fitting (see INITIALIZATION section)
_paths   = None   # list of DatasetPath
_n_lanes = N_LANES
_vis_scale_down = 1.0
_vis_scale_source = "default"
_cfg_X_vis = None
_cfg_Y_vis = None

# Map IDEAM decision-group names to lane indices (for _Decision_info)
_GROUP_TO_LANE = {"L1": 0, "L2": 0, "C1": 1, "C2": 1, "R1": 2, "R2": 2}


def _find_visualizer_params_path(dataset_dir):
    """Find visualizer_params.json from common dataset-tool locations."""
    candidates = []
    if VISUALIZER_PARAMS_PATH:
        candidates.append(VISUALIZER_PARAMS_PATH)
    ds_parent = os.path.abspath(os.path.join(dataset_dir, os.pardir))
    candidates.append(os.path.join(ds_parent, "visualizer_params", "visualizer_params.json"))
    candidates.append(os.path.join(os.path.dirname(__file__), "..", "data",
                                   "visualizer_params", "visualizer_params.json"))
    candidates.append(r"c:\drone-dataset-tools\data\visualizer_params\visualizer_params.json")

    for p in candidates:
        try:
            if p and os.path.exists(p):
                return os.path.abspath(p)
        except Exception:
            pass
    return None


def _infer_vis_scale_down(tracks, bg_img, dataset_dir, manual_scale=None):
    """
    Pick TrackVisualizer-compatible pixel scale:
      1) manual override if provided
      2) dataset preset (exiD/inD/rounD)
      3) auto-fit so all track pixels fit inside the background image
    """
    if manual_scale is not None and float(manual_scale) > 0.0:
        return float(manual_scale), "manual"

    ds_key = os.path.basename(os.path.normpath(dataset_dir)).lower()

    # Prefer the exact scale-down factor used by run_track_visualization.py.
    vp_path = _find_visualizer_params_path(dataset_dir)
    if vp_path is not None:
        try:
            with open(vp_path, "r", encoding="utf-8") as f:
                vp = json.load(f)
            s = float(vp["datasets"][ds_key]["scale_down_factor"])
            if s > 0:
                return s, f"visualizer_params:{vp_path}"
        except Exception:
            pass

    if ds_key in VIS_SCALE_PRESETS:
        return float(VIS_SCALE_PRESETS[ds_key]), f"preset:{ds_key}"

    if bg_img is None:
        return 1.0, "no_background"

    try:
        x_vis = np.concatenate([np.asarray(t["xCenterVis"]).ravel() for t in tracks])
        y_vis = np.concatenate([np.asarray(t["yCenterVis"]).ravel() for t in tracks])
        h, w = bg_img.shape[:2]
        s_x = float(np.nanmax(x_vis)) / max(1.0, 0.98 * float(w))
        s_y = float(np.nanmax(y_vis)) / max(1.0, 0.98 * float(h))
        s = max(1.0, s_x, s_y)
        if s > 2.0:
            s = round(s * 2.0) / 2.0
        return float(s), f"auto-fit sx={s_x:.2f}, sy={s_y:.2f}"
    except Exception:
        return 1.0, "auto-fallback"


def _Decision_info(x0, x0_g, path_center_list, sample_center,
                   x_center, y_center, bound, desired_group,
                   path_now_obj, path_now_idx):
    """
    Dataset-aware replacement for IDEAM's Decision_info().

    The original function calls give_desired_path() which returns global
    highway paths (path1c/path2c/path3c), then uses np.where() to find their
    index in path_center_list.  With DatasetPath objects the comparison yields
    an empty array 鈫?IndexError.

    This version maps the desired-group name directly to a lane index via
    _GROUP_TO_LANE, bypassing any global-path lookup entirely.
    """
    target_name = desired_group.get("name", "C1")
    path_dindex = _GROUP_TO_LANE.get(target_name, path_now_idx)
    # Clamp to valid range
    path_dindex = max(0, min(path_dindex, len(_paths) - 1))
    path_d      = _paths[path_dindex]

    sample  = sample_center[path_dindex]
    x_list  = x_center[path_dindex]
    y_list  = y_center[path_dindex]

    # Repropagate ego state on the current (keep-lane) path to get post-state s
    sample_post = sample_center[path_now_idx]
    x_list_post = x_center[path_now_idx]
    y_list_post = y_center[path_now_idx]
    x0_post = repropagate(path_now_obj, sample_post, x_list_post, y_list_post,
                          x0_g, list(x0))

    # Replicate IDEAM's post_process: flag short-range target so we don't LC
    sl = desired_group.get("sl", None)
    target_ahead = sl[0] if (sl is not None and len(sl) > 0) else 1e4
    is_short = abs(target_ahead - x0_post[3]) <= 7.5

    if path_now_idx != path_dindex and not is_short:
        C_label   = "R" if path_dindex > path_now_idx else "L"
        x0_update = repropagate(path_d, sample, x_list, y_list, x0_g, list(x0))
        return path_d, path_dindex, C_label, sample, x_list, y_list, x0_update
    else:
        sample  = sample_center[path_now_idx]
        x_list  = x_center[path_now_idx]
        y_list  = y_center[path_now_idx]
        x0_update = repropagate(path_now_obj, sample, x_list, y_list, x0_g, list(x0))
        return path_now_obj, path_now_idx, "K", sample, x_list, y_list, x0_update


def _judge_lane(ego_xy):
    """
    Replace judge_current_position(X0_g[:2], x_bound, y_bound, ...).
    Returns lane index (0/1/2) of the path closest to ego_xy.
    """
    best_li, best_ey = 0, float("inf")
    for li, path in enumerate(_paths):
        try:
            _, ey, _ = find_frenet_coord(
                path, path.xc, path.yc, path.samples,
                [ego_xy[0], ego_xy[1], 0.0])
            if abs(ey) < abs(best_ey):
                best_li, best_ey = li, ey
        except Exception:
            pass
    return best_li


def _get_path_info(path_dindex):
    """Replace IDEAM's global get_path_info(idx)."""
    p = _paths[path_dindex]
    return p, p.xc, p.yc, p.samples


def _bind_dataset_path_adapters_to_utils(utils_obj, tag):
    """
    Bind DecisionMaking.util path lookup to dataset paths.

    DecisionMaking.util methods call module-level get_path_info(). Without
    rebinding they keep using Path.path highway references.
    """
    _dm_util_module.get_path_info = _get_path_info
    try:
        utils_obj.get_alllane_lf.__globals__["get_path_info"] = _get_path_info
    except Exception:
        pass
    try:
        utils_obj.get_remap_vehicles.__globals__["get_path_info"] = _get_path_info
    except Exception:
        pass
    _src = utils_obj.get_alllane_lf.__globals__.get("get_path_info", None)
    _mod = getattr(_src, "__module__", "unknown")
    print(f"[ADAPTER] {tag}: DecisionMaking.util.get_path_info -> {_mod}")


def _path_to_path_proj(s_old, ey_old, old_lane, new_lane):
    """
    Replace IDEAM's path_to_path_proj(s, ey, old, new).
    Converts Frenet coordinates from one DatasetPath to another.
    """
    p_old = _paths[old_lane]
    p_new = _paths[new_lane]

    s_arr = np.asarray(s_old)
    ey_arr = np.asarray(ey_old)
    s_is_scalar = (s_arr.ndim == 0)
    ey_is_scalar = (ey_arr.ndim == 0)

    s_vec = np.atleast_1d(s_arr).astype(float).reshape(-1)
    ey_vec = np.atleast_1d(ey_arr).astype(float).reshape(-1)

    # Match lengths for mixed scalar/vector inputs.
    if s_vec.size != ey_vec.size:
        if s_vec.size == 1:
            s_vec = np.full(ey_vec.shape, float(s_vec[0]), dtype=float)
        elif ey_vec.size == 1:
            ey_vec = np.full(s_vec.shape, float(ey_vec[0]), dtype=float)
        else:
            n = min(s_vec.size, ey_vec.size)
            s_vec = s_vec[:n]
            ey_vec = ey_vec[:n]

    s_proj, ey_proj = [], []
    for s_i, ey_i in zip(s_vec, ey_vec):
        x_w, y_w = p_old.get_cartesian_coords(float(s_i), float(ey_i))
        psi_old = p_old.get_theta_r(float(s_i))
        try:
            s_n, ey_n, _ = find_frenet_coord(
                p_new, p_new.xc, p_new.yc, p_new.samples,
                [x_w, y_w, psi_old])
        except Exception:
            s_n, ey_n = float(s_i), float(ey_i)
        s_proj.append(float(np.asarray(s_n).flat[0]))
        ey_proj.append(float(np.asarray(ey_n).flat[0]))

    if s_is_scalar and ey_is_scalar:
        return float(s_proj[0]), float(ey_proj[0])
    return np.asarray(s_proj, dtype=float), np.asarray(ey_proj, dtype=float)


# ===========================================================================
# SURROUNDING VEHICLE BUILDER (dataset 鈫?IDEAM + DRIFT format)
# ===========================================================================

def build_surrounding_arrays(frame_idx, ego_track_id, tracks, tracks_meta,
                              class_map, paths, frame_rate):
    """
    Read active vehicles at frame_idx, assign to lanes, return:
      vehicle_left, vehicle_centre, vehicle_right  鈥?IDEAM format arrays
      drift_vehicles                               鈥?DRIFT vehicle dicts
    Each IDEAM row: [s, ey, epsi, x, y, psi, vx, a]
    """
    lane_buckets = [[], [], []]
    drift_list   = []
    vid = 1

    for tm in tracks_meta:
        tid = tm["trackId"]
        if tid == ego_track_id:
            continue
        if not (tm["initialFrame"] <= frame_idx <= tm["finalFrame"]):
            continue

        tr = tracks[tid]
        fi = frame_idx - tm["initialFrame"]
        cls = class_map.get(tid, "car")

        x   = float(tr["xCenter"][fi])
        y   = float(tr["yCenter"][fi])
        psi = heading_to_psi(tr["heading"][fi])

        # Use longitudinal velocity (body frame) as vx
        lon_v = float(tr["lonVelocity"][fi])
        lon_a = float(tr["lonAcceleration"][fi]) if "lonAcceleration" in tr else 0.0

        # Skip stationary/very-slow vehicles (pedestrians, parked)
        if abs(lon_v) < 0.5 and cls not in ("car", "van", "truck"):
            continue

        # Assign to closest lane
        best_li, best_ey, best_s, best_epsi = -1, float("inf"), 0.0, 0.0
        for li, path in enumerate(paths):
            try:
                s_f, ey_f, ep_f = find_frenet_coord(
                    path, path.xc, path.yc, path.samples, [x, y, psi])
                if abs(ey_f) < abs(best_ey):
                    best_li, best_ey = li, ey_f
                    best_s, best_epsi = s_f, ep_f
            except Exception:
                pass

        if best_li < 0 or abs(best_ey) > 12.0:
            # Vehicle too far from all fitted lanes 鈥?still add to DRIFT
            vclass = "truck" if cls in ("truck", "van") else "car"
            vx_g = lon_v * math.cos(psi)
            vy_g = lon_v * math.sin(psi)
            v = drift_create_vehicle(vid=vid, x=x, y=y,
                                     vx=vx_g, vy=vy_g, vclass=vclass)
            v["heading"] = psi
            v["a"] = lon_a
            drift_list.append(v)
            vid += 1
            continue

        row = np.array([best_s, best_ey, best_epsi, x, y, psi, lon_v, lon_a])
        lane_buckets[best_li].append(row)

        # DRIFT vehicle
        vclass = "truck" if cls in ("truck", "van") else "car"
        vx_g = lon_v * math.cos(psi)
        vy_g = lon_v * math.sin(psi)
        v = drift_create_vehicle(vid=vid, x=x, y=y,
                                 vx=vx_g, vy=vy_g, vclass=vclass)
        v["heading"] = psi
        v["a"] = lon_a
        drift_list.append(v)
        vid += 1

    def sort_and_pad(bucket):
        if not bucket:
            return np.array([[1e4, 0.0, 0.0, 1e4, 0.0, 0.0, 0.01, 0.0]])
        arr = np.array(bucket)
        return arr[np.argsort(arr[:, 0])]

    return (sort_and_pad(lane_buckets[0]),
            sort_and_pad(lane_buckets[1]),
            sort_and_pad(lane_buckets[2]),
            drift_list)


# ===========================================================================
# VISUALIZATION HELPER
# ===========================================================================

def draw_frame(i, X0_g_dream, X0_dream, X0_g_ideam, X0_ideam,
               vl, vc, vr, risk_field, risk_at_ego,
               tracks, tracks_meta, class_map, frame_idx,
               horizon_dream=None, horizon_ideam=None,
               bg_img=None, bg_extent=None):
    """Render one frame: IDEAM (top) / DREAM (bottom)."""
    fig = plt.gcf()
    fig.clf()

    ax_top = fig.add_subplot(2, 1, 1)
    ax_bot = fig.add_subplot(2, 1, 2)

    for ax, ego_g, ego_s, horizon, panel_risk, panel_risk_val, title in [
        (ax_top, X0_g_ideam, X0_ideam, horizon_ideam, None, None,
         "IDEAM (baseline)"),
        (ax_bot, X0_g_dream, X0_dream, horizon_dream, risk_field, risk_at_ego,
         "DREAM (ours)"),
    ]:
        ax.cla()

        x_c, y_c = ego_g[0], ego_g[1]
        xlim = [x_c - VIEW_X, x_c + VIEW_X]
        ylim = [y_c - VIEW_Y, y_c + VIEW_Y]

        # Background image
        if USE_DATASET_BACKGROUND and bg_img is not None and bg_extent is not None:
            ax.imshow(bg_img, extent=bg_extent, origin="upper",
                      aspect="auto", zorder=0)

        # DRIFT risk overlay (DREAM panel only)
        cf_obj = None
        if panel_risk is not None:
            R_sm = _gf(panel_risk, sigma=RISK_SMOOTH_SIGMA)
            R_sm = np.clip(R_sm, 0, RISK_VMAX)
            R_vis = np.ma.masked_less(R_sm, RISK_MIN_VIS)
            if np.ma.count(R_vis) > 0:
                levels = np.linspace(RISK_MIN_VIS, RISK_VMAX, RISK_LEVELS)
                cf_obj = ax.contourf(cfg.X, cfg.Y, R_vis,
                                     levels=levels, cmap=RISK_CMAP,
                                     alpha=RISK_ALPHA, vmin=RISK_MIN_VIS,
                                     vmax=RISK_VMAX, zorder=1, extend="max")
                ax.contour(cfg.X, cfg.Y, R_vis,
                           levels=np.linspace(max(0.2, RISK_MIN_VIS), RISK_VMAX, 6),
                           colors="darkred", linewidths=0.4, alpha=0.35, zorder=1.1)

        # Draw road using selected view style.
        if VIEW_STYLE == "run_simulation":
            draw_road_runsim_style(ax)
        else:
            draw_dataset_road_layout(ax)

        # Dataset surrounding vehicles at this frame
        for tm in tracks_meta:
            tid = tm["trackId"]
            if not (tm["initialFrame"] <= frame_idx <= tm["finalFrame"]):
                continue
            tr = tracks[tid]
            fi  = frame_idx - tm["initialFrame"]
            vx_ = float(tr["xCenter"][fi])
            vy_ = float(tr["yCenter"][fi])
            if not (xlim[0] <= vx_ <= xlim[1] and ylim[0] <= vy_ <= ylim[1]):
                continue
            psi_ = heading_to_psi(tr["heading"][fi])
            cls_ = class_map.get(tid, "car")
            L = 8.0 if cls_ in ("truck", "van") else 3.5   # highway-style display size
            W = 2.5 if cls_ in ("truck", "van") else 1.2
            fc = "#FF6F00" if cls_ in ("truck", "van") else "#FFD600"
            draw_vehicle_rect(ax, vx_, vy_, psi_, L, W, fc,
                              edgecolor="black", lw=0.6, zorder=4)
            ax.text(vx_ + 0.5, vy_ + 0.5,
                    f"{float(tr['lonVelocity'][fi]):.1f}",
                    fontsize=3.5, color="k", style="oblique", zorder=5)

        # Horizon
        if horizon is not None and len(horizon) > 0:
            h = np.asarray(horizon)
            if h.ndim == 2 and h.shape[1] >= 2:
                ax.plot(h[:, 0], h[:, 1], color="#00BCD4",
                        lw=1.5, ls="--", zorder=7)
                ax.scatter(h[:, 0], h[:, 1], color="#00BCD4", s=4, zorder=7)

        # Ego
        ego_color = "#2196F3" if panel_risk is not None else "#4CAF50"
        draw_vehicle_rect(ax, ego_g[0], ego_g[1], ego_g[2],
                          4.5, 2.0, ego_color, edgecolor="navy",
                          lw=1.0, zorder=8)
        ax.text(ego_g[0] - 2, ego_g[1] + 1.5,
                f"{ego_s[0]:.1f} m/s",
                fontsize=4.5, color="black", style="oblique", zorder=9)

        # Risk badge
        if panel_risk_val is not None:
            rc = ("red"    if panel_risk_val > 1.5 else
                  "orange" if panel_risk_val > 0.5 else "green")
            ax.text(0.985, 0.965, f"R={panel_risk_val:.2f}",
                    transform=ax.transAxes, ha="right", va="top",
                    c=rc, fontsize=7, fontweight="bold",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{title}  |  t={i*dt:.1f} s  frame={frame_idx}",
                     fontsize=9, fontweight="bold")

        if cf_obj is not None:
            cbar = fig.colorbar(cf_obj, ax=ax, orientation="vertical",
                                pad=0.01, fraction=0.025)
            cbar.set_label("Risk", fontsize=7)
            cbar.ax.tick_params(labelsize=6)

    ax_top.tick_params(labelbottom=False)
    plt.savefig(os.path.join(save_dir, f"{i}.png"), dpi=300)


def draw_frame_drift_overlay(i, frame_idx, tracks, tracks_meta, class_map,
                             bg_img, risk_field, risk_at_ego,
                             ego_vx_hist=None, ego_ax_hist=None,
                             risk_hist=None):
    """
    Pixel-space rendering — traffic scene only (no telemetry strip in frames).
    Motion data is recorded externally and shown in metrics_drift_overlay.png.
    """
    fig = plt.gcf()
    fig.clf()
    ax = fig.add_subplot(1, 1, 1)
    ax.cla()

    vmax = RISK_VMAX

    # 1) Background in pixel coordinates.
    if bg_img is not None:
        ax.imshow(bg_img, origin="upper", zorder=0)
    else:
        ax.set_facecolor("#111111")

    # 2) DRIFT risk projected to pixel coordinates.
    if risk_field is not None and _cfg_X_vis is not None and _cfg_Y_vis is not None:
        R_sm = _gf(risk_field, sigma=RISK_SMOOTH_SIGMA)
        nonzero = R_sm[R_sm > RISK_MIN_VIS]
        vmax = float(np.percentile(nonzero, 95)) if nonzero.size > 50 else RISK_VMAX
        vmax = max(vmax, RISK_MIN_VIS + 1e-3)

        # Compress contrast slightly so the map and vehicles remain readable.
        Rn = (np.clip(R_sm, RISK_MIN_VIS, vmax) - RISK_MIN_VIS) / (vmax - RISK_MIN_VIS)
        Rn = np.power(np.clip(Rn, 0.0, 1.0), RISK_ALPHA_GAMMA)
        R_masked = np.ma.masked_less_equal(Rn, 0.0)
        if np.ma.count(R_masked) > 0:
            ax.contourf(_cfg_X_vis, _cfg_Y_vis, R_masked,
                        levels=np.linspace(0.02, 1.0, RISK_LEVELS),
                        cmap=RISK_CMAP, alpha=RISK_ALPHA,
                        zorder=2, antialiased=True)

    # 3) Vehicles using bboxVis/centerVis in the same pixel coordinate system.
    for tm in tracks_meta:
        tid = tm["trackId"]
        if not (tm["initialFrame"] <= frame_idx <= tm["finalFrame"]):
            continue
        tr = tracks[tid]
        fi = frame_idx - tm["initialFrame"]
        cls_ = class_map.get(tid, "car")
        is_ego = (tid == EGO_TRACK_ID)
        fc = "#F4511E" if is_ego else ("#FF8C00" if cls_ in ("truck", "van") else "#AED6F1")
        ec = "red" if is_ego else "black"
        lw = 1.0 if is_ego else 0.5
        alpha = 0.95 if is_ego else 0.82
        z = 5 if is_ego else 4

        if tr.get("bboxVis") is not None:
            bbox = np.asarray(tr["bboxVis"][fi], dtype=float) / _vis_scale_down
            poly = plt.Polygon(bbox, closed=True, facecolor=fc, edgecolor=ec,
                               linewidth=lw, alpha=alpha, zorder=z)
            ax.add_patch(poly)
        else:
            cx = float(tr["xCenterVis"][fi]) / _vis_scale_down
            cy = float(tr["yCenterVis"][fi]) / _vis_scale_down
            circ = plt.Circle((cx, cy), radius=max(1.4, 2.4 / _vis_scale_down),
                              facecolor=fc, edgecolor=ec, linewidth=lw,
                              alpha=alpha, zorder=z)
            ax.add_patch(circ)

    # 4) Ego-centric viewport (same physical window as run_simulation.py).
    if EGO_TRACK_ID is not None:
        ego_tm = tracks_meta[EGO_TRACK_ID]
        ego_tr = tracks[EGO_TRACK_ID]
        if ego_tm["initialFrame"] <= frame_idx <= ego_tm["finalFrame"]:
            fi_ego = frame_idx - ego_tm["initialFrame"]
            ex_px = float(ego_tr["xCenterVis"][fi_ego]) / _vis_scale_down
            ey_px = float(ego_tr["yCenterVis"][fi_ego]) / _vis_scale_down
        else:
            ex_px = float(np.mean(_track_x_all) / (_ortho_px_m * _vis_scale_down))
            ey_px = float(np.mean(-_track_y_all) / (_ortho_px_m * _vis_scale_down))
    else:
        ex_px = float(np.mean(_track_x_all) / (_ortho_px_m * _vis_scale_down))
        ey_px = float(np.mean(-_track_y_all) / (_ortho_px_m * _vis_scale_down))

    view_x_px = VIEW_X / (_ortho_px_m * _vis_scale_down)
    view_y_px = VIEW_Y / (_ortho_px_m * _vis_scale_down)
    x0, x1 = ex_px - view_x_px, ex_px + view_x_px
    y_top, y_bot = ey_px - view_y_px, ey_px + view_y_px
    if bg_img is not None:
        h, w = bg_img.shape[:2]
        if x0 < 0:
            x1 -= x0
            x0 = 0
        if x1 > (w - 1):
            x0 -= (x1 - (w - 1))
            x1 = w - 1
        if y_top < 0:
            y_bot -= y_top
            y_top = 0
        if y_bot > (h - 1):
            y_top -= (y_bot - (h - 1))
            y_bot = h - 1
        x0 = max(0, x0)
        x1 = min(w - 1, x1)
        y_top = max(0, y_top)
        y_bot = min(h - 1, y_bot)

    ax.set_xlim(x0, x1)
    # Keep TrackVisualizer's "y increases downward" pixel orientation.
    ax.set_ylim(y_bot, y_top)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    # 5) Labels / colorbar.
    rc = "red" if risk_at_ego > 1.5 else "orange" if risk_at_ego > 0.5 else "lime"
    ax.text(0.985, 0.965, f"R={risk_at_ego:.2f}",
            transform=ax.transAxes, ha="right", va="top",
            color=rc, fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="black", alpha=0.55))
    ax.set_title(f"DRIFT Risk Overlay  |  t={i*dt:.1f} s  frame={frame_idx}",
                 fontsize=9, fontweight="bold")

    _sm = plt.cm.ScalarMappable(
        norm=plt.Normalize(vmin=RISK_MIN_VIS, vmax=vmax),
        cmap=plt.colormaps[RISK_CMAP])
    _sm.set_array([])
    cbar = fig.colorbar(_sm, ax=ax, fraction=0.018, pad=0.005)
    cbar.set_label(f"Risk  (vmax={vmax:.2f})", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    plt.savefig(os.path.join(save_dir, f"{i}.png"), dpi=150,
                bbox_inches="tight")


# ===========================================================================
# INITIALIZATION
# ===========================================================================

print("=" * 70)
print(f"DREAM Dataset  |  {RECORDING_ID}  |  "
      f"integration={INTEGRATION_MODE}  scenario={SCENARIO_MODE}")
print("=" * 70)

config_integration = get_preset(INTEGRATION_MODE)
config_integration.apply_mode()

# Load dataset 
rec = f"{int(RECORDING_ID):02d}"
tracks_file       = os.path.join(DATASET_DIR, f"{rec}_tracks.csv")
tracks_meta_file  = os.path.join(DATASET_DIR, f"{rec}_tracksMeta.csv")
rec_meta_file     = os.path.join(DATASET_DIR, f"{rec}_recordingMeta.csv")

print(f"Loading recording {rec} from {DATASET_DIR} ...")
tracks, tracks_meta, recording_meta = read_from_csv(
    tracks_file, tracks_meta_file, rec_meta_file,
    include_px_coordinates=True)    # needed for bboxVis in pixel space

# Module-level ortho scale used for DRIFT grid → pixel conversion.
# x_px = x_m / _ortho_px_m,   y_px = -y_m / _ortho_px_m  (Y flipped)
_ortho_px_m = float(recording_meta["orthoPxToMeter"])

frame_rate   = float(recording_meta["frameRate"])    # e.g. 25 Hz
dt_dataset   = 1.0 / frame_rate
frame_stride = max(1, round(dt / dt_dataset))        # dataset frames per sim step
ortho_raw    = float(recording_meta["orthoPxToMeter"])

class_map = {tm["trackId"]: tm["class"] for tm in tracks_meta}
_track_x_all = np.concatenate([t["xCenter"] for t in tracks])
_track_y_all = np.concatenate([t["yCenter"] for t in tracks])
track_span_x = float(np.max(_track_x_all) - np.min(_track_x_all))
track_span_y = float(np.max(_track_y_all) - np.min(_track_y_all))

print(f"  Tracks: {len(tracks)}  |  frameRate={frame_rate} Hz  "
      f"  frame_stride={frame_stride}  (dt={dt}s)")

# 鈹€鈹€ Background image 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
bg_path = os.path.join(DATASET_DIR, f"{rec}_background.png")
bg_img, bg_extent = None, None
img_h, img_w = 0, 0
if not USE_DATASET_BACKGROUND:
    print("  Background raster disabled (run_simulation style view).")
elif os.path.exists(bg_path):
    _raw = cv2.imread(bg_path)
    bg_img = cv2.cvtColor(_raw, cv2.COLOR_BGR2RGB)
    img_h, img_w = bg_img.shape[:2]

    # Kept for benchmark/world view code path; drift_overlay uses pure pixel space.
    map_scale_m = _ortho_px_m * BG_SCALE_MULT
    bg_extent = [
        BG_OFFSET_X_M,
        BG_OFFSET_X_M + img_w * map_scale_m,
        BG_OFFSET_Y_M - img_h * map_scale_m,
        BG_OFFSET_Y_M,
    ]
else:
    print(f"  [WARN] Background image not found at {bg_path}")

_vis_scale_down, _vis_scale_source = _infer_vis_scale_down(
    tracks, bg_img, DATASET_DIR, manual_scale=VIS_SCALE_DOWN)
print(f"  Background: {img_w}x{img_h} px  |  ortho={_ortho_px_m:.6f} m/px")
print(f"  Pixel scale_down={_vis_scale_down:.2f}  ({_vis_scale_source})")

# Ego auto-selection 
# Compute mean heading from all car tracks (used for direction filter)
_all_hdg = [float(np.nanmean(t["heading"]))
            for t, tm in zip(tracks, tracks_meta)
            if class_map.get(tm["trackId"]) in ("car", "van")]
_hdg_ref = float(np.nanmean(_all_hdg)) if _all_hdg else 0.0

if EGO_TRACK_ID is None:
    EGO_TRACK_ID = select_ego_track(
        tracks, tracks_meta, class_map,
        min_frames=EGO_MIN_FRAMES, ego_lane=EGO_LANE,
        heading_ref_deg=_hdg_ref)
    print(f"  Auto-selected ego: trackId={EGO_TRACK_ID}  "
          f"class={class_map.get(EGO_TRACK_ID)}  "
          f"frames={tracks_meta[EGO_TRACK_ID]['numFrames']}")
else:
    print(f"  User-specified ego: trackId={EGO_TRACK_ID}  "
          f"class={class_map.get(EGO_TRACK_ID)}")

ego_meta  = tracks_meta[EGO_TRACK_ID]
ego_track = tracks[EGO_TRACK_ID]
ego_fi0   = ego_meta["initialFrame"]

# Lane path fitting
paths = fit_lane_paths(tracks, tracks_meta, class_map, EGO_TRACK_ID,
                       n_lanes=N_LANES)
_paths = paths   # set module-level for helper functions

# Precompute path data tuples
path_center   = np.array(paths, dtype=object)
sample_center = np.array([p.samples for p in paths], dtype=object)
x_center      = [p.xc for p in paths]
y_center      = [p.yc for p in paths]

# Expand DRIFT grid to cover full scene 
_all_x = np.concatenate([t["xCenter"] for t in tracks])
_all_y = np.concatenate([t["yCenter"] for t in tracks])
cfg.x_min = float(np.min(_all_x)) - SCENE_MARGIN
cfg.x_max = float(np.max(_all_x)) + SCENE_MARGIN
cfg.y_min = float(np.min(_all_y)) - SCENE_MARGIN
cfg.y_max = float(np.max(_all_y)) + SCENE_MARGIN
cfg.nx    = int((cfg.x_max - cfg.x_min) / DRIFT_CELL_M) + 2
cfg.ny    = int((cfg.y_max - cfg.y_min) / DRIFT_CELL_M) + 2
cfg.dx    = (cfg.x_max - cfg.x_min) / (cfg.nx - 1)
cfg.dy    = (cfg.y_max - cfg.y_min) / (cfg.ny - 1)
cfg.x     = np.linspace(cfg.x_min, cfg.x_max, cfg.nx)
cfg.y     = np.linspace(cfg.y_min, cfg.y_max, cfg.ny)
cfg.X, cfg.Y = np.meshgrid(cfg.x, cfg.y)
_cfg_X_vis = cfg.X / (_ortho_px_m * _vis_scale_down)
_cfg_Y_vis = -cfg.Y / (_ortho_px_m * _vis_scale_down)
print(f"[DRIFT Grid] x{cfg.x_min:.0f},{cfg.x_max:.0f}]  "
      f"y{cfg.y_min:.0f},{cfg.y_max:.0f}]  "
      f"nx={cfg.nx} ny={cfg.ny}  ({cfg.nx*cfg.ny//1000}k cells)")

# PRIDEAM controller 
controller = create_prideam_controller(
    paths={0: paths[0], 1: paths[1], 2: paths[2]},
    risk_weights={
        "mpc_cost":           config_integration.mpc_risk_weight,
        "cbf_modulation":     config_integration.cbf_alpha,
        "decision_threshold": config_integration.decision_risk_threshold,
    })
controller.get_path_curvature(path=paths[EGO_LANE])
drift = controller.drift

# Baseline IDEAM MPC 
baseline_mpc = LMPC(**constraint_params())
utils_ideam  = LeaderFollower_Uitl(**util_params())
_bind_dataset_path_adapters_to_utils(utils_ideam, "IDEAM")
baseline_mpc.set_util(utils_ideam)
baseline_mpc.get_path_curvature(path=paths[EGO_LANE])

# Shared decision-making objects
Params         = params()
dynamics       = Dynamic(**Params)
decision_param = decision_params()
decision_maker = decision(**decision_param)

utils_dream = LeaderFollower_Uitl(**util_params())
_bind_dataset_path_adapters_to_utils(utils_dream, "DREAM")
controller.set_util(utils_dream)
mpc_ctrl = controller.mpc

boundary    = 1.0
steer_range = [math.radians(-8.0), math.radians(8.0)]

# Ego initial state from dataset 
_ex0  = float(ego_track["xCenter"][0])
_ey0  = float(ego_track["yCenter"][0])
_eh0  = heading_to_psi(ego_track["heading"][0])
_ev0  = max(float(ego_track["lonVelocity"][0]), 1.0)

_ego_path = paths[EGO_LANE]
try:
    _s0, _ey_f0, _epsi0 = find_frenet_coord(
        _ego_path, _ego_path.xc, _ego_path.yc, _ego_path.samples,
        [_ex0, _ey0, _eh0])
except Exception:
    _s0, _ey_f0, _epsi0 = 0.0, 0.0, 0.0

X0     = [_ev0, 0.0, 0.0, _s0, _ey_f0, _epsi0]
X0_g   = [_ex0, _ey0, _eh0]

X0_ideam   = list(X0)
X0_g_ideam = list(X0_g)

print(f"[EGO INIT] s={_s0:.1f} m  vx={_ev0:.1f} m/s  "
      f"pos=({_ex0:.1f}, {_ey0:.1f})  psi={math.degrees(_eh0):.1f}")

# DRIFT warmup 
print("DRIFT warm-up ...")
_vl0, _vc0, _vr0, _drift_init = build_surrounding_arrays(
    ego_fi0, EGO_TRACK_ID, tracks, tracks_meta, class_map, paths, frame_rate)
_ego_init = drift_create_vehicle(
    vid=0, x=X0_g[0], y=X0_g[1],
    vx=X0[0]*math.cos(X0_g[2]), vy=X0[0]*math.sin(X0_g[2]),
    vclass="car")
_ego_init["heading"] = X0_g[2]
drift.warmup(_drift_init + [_ego_init], _ego_init,
             dt=dt, duration=WARMUP_S, substeps=3)
print()

if SCENARIO_MODE == "drift_overlay":
    print(f"Running DRIFT overlay only ({N_t} steps, dt={dt}s) ...")
    print()
    risk_at_ego_list = []
    ego_vx_list      = []   # longitudinal speed [m/s]
    ego_ax_list      = []   # longitudinal acceleration [m/s²]
    bar = Bar(max=N_t - 1)
    plt.figure(figsize=(12, 7))

    max_frame_all = max(tm["finalFrame"] for tm in tracks_meta)
    for i in range(N_t):
        bar.next()
        frame_idx = ego_fi0 + i * frame_stride

        if frame_idx > ego_meta["finalFrame"] or frame_idx > max_frame_all:
            print(f"\n[WARN] Reached end of recording at step {i}. Stopping.")
            N_t = i
            break

        fi_ego = frame_idx - ego_meta["initialFrame"]
        ex = float(ego_track["xCenter"][fi_ego])
        ey = float(ego_track["yCenter"][fi_ego])
        epsi = heading_to_psi(ego_track["heading"][fi_ego])
        ev = float(ego_track["lonVelocity"][fi_ego])
        ea = float(ego_track["lonAcceleration"][fi_ego]) \
             if "lonAcceleration" in ego_track else 0.0
        ego_vx_list.append(ev)
        ego_ax_list.append(ea)

        vl, vc, vr, drift_vehicles = build_surrounding_arrays(
            frame_idx, EGO_TRACK_ID, tracks, tracks_meta,
            class_map, paths, frame_rate)

        ego_drift_v = drift_create_vehicle(
            vid=0, x=ex, y=ey,
            vx=ev * math.cos(epsi), vy=ev * math.sin(epsi),
            vclass="car")
        ego_drift_v["heading"] = epsi

        risk_field = drift.step(drift_vehicles, ego_drift_v, dt=dt, substeps=3)
        risk_at_ego = float(drift.get_risk_cartesian(ex, ey))
        risk_at_ego_list.append(risk_at_ego)

        draw_frame_drift_overlay(
            i, frame_idx, tracks, tracks_meta, class_map,
            bg_img, risk_field, risk_at_ego,
            ego_vx_hist=list(ego_vx_list),
            ego_ax_hist=list(ego_ax_list),
            risk_hist=list(risk_at_ego_list))

    bar.finish()
    print()
    print("DRIFT overlay simulation complete.")

    if len(risk_at_ego_list) > 0:
        _t_r = np.arange(len(risk_at_ego_list)) * dt
        with plt.style.context(["science", "no-latex"]):
            fig_r, axes_r = plt.subplots(3, 1, figsize=(10, 8),
                                         constrained_layout=True, sharex=True)
            fig_r.suptitle(
                f"Ego Motion & DRIFT Risk  |  {rec} track {EGO_TRACK_ID}",
                fontsize=11)

            _panels = [
                (axes_r[0], ego_vx_list,      "C0", "$v_x$ [m/s]",  "Longitudinal Speed"),
                (axes_r[1], ego_ax_list,       "C1", "$a_x$ [m/s²]", "Longitudinal Acceleration"),
                (axes_r[2], risk_at_ego_list,  "C3", "Risk $R$",     "DRIFT Risk at Ego"),
            ]
            for axi, data, col, ylabel, title in _panels:
                axi.plot(_t_r, data, color=col, lw=1.4)
                axi.fill_between(_t_r, data, alpha=0.2, color=col)
                axi.set_ylabel(ylabel)
                axi.set_title(title, fontsize=9)
                axi.grid(True, lw=0.4, alpha=0.4)
            axes_r[1].axhline(0, color="black", lw=0.6, alpha=0.5)
            axes_r[2].set_xlabel("t [s]")

            plt.savefig(os.path.join(save_dir, "metrics_drift_overlay.png"),
                        dpi=200, bbox_inches="tight")
            plt.close(fig_r)

        np.save(os.path.join(save_dir, "metrics_drift_overlay.npy"), {
            "risk_at_ego": risk_at_ego_list,
            "ego_vx":      ego_vx_list,
            "ego_ax":      ego_ax_list,
            "recording":   rec,
            "ego_track_id": EGO_TRACK_ID,
            "scenario_mode": SCENARIO_MODE,
        })

    print(f"\nFrames  -> {save_dir}/")
    print(f"Metrics -> {save_dir}/metrics_drift_overlay.png")
    sys.exit(0)

# Misc state 
oa_i, od_i = 0.0, 0.0
last_X        = None
last_X_ideam  = None
path_changed  = EGO_LANE
path_changed_i = EGO_LANE

risk_at_ego_list = []
dream_s, dream_vx, dream_acc = [], [], []
ideam_s, ideam_vx, ideam_acc = [], [], []

ideam_X0_panel   = list(X0_ideam)
ideam_X0_g_panel = list(X0_g_ideam)
last_ideam_hor   = None

_d0_base = utils_dream.d0
_Th_base = utils_dream.Th
_al_base = mpc_ctrl.a_l
_bl_base = mpc_ctrl.b_l
_P_base  = mpc_ctrl.P.copy()

bar = Bar(max=N_t - 1)
plt.figure(figsize=(16, 10))


def _safe_first_scalar(val, default=0.0):
    """Return val[0] (or val) as float; fall back to default on None/invalid."""
    if val is None:
        return float(default)
    try:
        arr = np.asarray(val)
        if arr.size == 0:
            return float(default)
        return float(arr.flat[0])
    except Exception:
        return float(default)

# ===========================================================================
# MAIN SIMULATION LOOP
# ===========================================================================

print(f"Running {N_t} steps  (dt={dt}s, {N_t*dt:.0f}s total)  ...")
print()

for i in range(N_t):
    bar.next()

    frame_idx = ego_fi0 + i * frame_stride

    # Guard: if we've exceeded the recording, stop
    if frame_idx > max(tm["finalFrame"] for tm in tracks_meta):
        print(f"\n[WARN] Reached end of recording at step {i}. Stopping.")
        N_t = i
        break

    # ------------------------------------------------------------------
    # 1. GET SURROUNDING VEHICLES FROM DATASET
    # ------------------------------------------------------------------
    vl, vc, vr, drift_vehicles = build_surrounding_arrays(
        frame_idx, EGO_TRACK_ID, tracks, tracks_meta,
        class_map, paths, frame_rate)

    # ------------------------------------------------------------------
    # 2. STEP DRIFT
    # ------------------------------------------------------------------
    ego_psi = X0_g[2]
    ego_drift_v = drift_create_vehicle(
        vid=0, x=X0_g[0], y=X0_g[1],
        vx=X0[0]*math.cos(ego_psi) - X0[1]*math.sin(ego_psi),
        vy=X0[0]*math.sin(ego_psi) + X0[1]*math.cos(ego_psi),
        vclass="car")
    ego_drift_v["heading"] = ego_psi

    risk_field  = drift.step(drift_vehicles, ego_drift_v, dt=dt, substeps=3)
    risk_at_ego = drift.get_risk_cartesian(X0_g[0], X0_g[1])
    risk_at_ego_list.append(risk_at_ego)

    # ------------------------------------------------------------------
    # 3. DREAM PATH DECISION
    # ------------------------------------------------------------------
    path_now  = _judge_lane(X0_g[:2])
    path_ego  = paths[path_now]
    start_str = {0: "L1", 1: "C1", 2: "R1"}[path_now]

    if i == 0:
        ovx, ovy, owz, oS, oey_v, oepsi = clac_last_X(
            oa, od, mpc_ctrl.T, path_ego, dt, 6, X0, X0_g)
        last_X = [ovx, ovy, owz, oS, oey_v, oepsi]

    all_info   = utils_dream.get_alllane_lf(path_ego, X0_g, path_now, vl, vc, vr)
    group_dict, ego_group = utils_dream.formulate_gap_group(
        path_now, last_X, all_info, vl, vc, vr)
    desired_group = decision_maker.decision_making(group_dict, start_str)

    path_d, path_dindex, C_label, sample, x_list, y_list, X0 = _Decision_info(
        X0, X0_g, path_center, sample_center,
        x_center, y_center, boundary, desired_group, path_ego, path_now)

    C_label_additive = utils_dream.inquire_C_state(C_label, desired_group)

    if C_label_additive == "Probe":
        path_d, path_dindex = path_ego, path_now
        C_label_virtual = "K"
        _, xc, yc, samplesc = _get_path_info(path_dindex)
        X0 = repropagate(path_d, samplesc, xc, yc, X0_g, X0)
    else:
        C_label_virtual = C_label

    # Decision veto
    if config_integration.enable_decision_veto and C_label != "K":
        _rs, _allow, _ = controller.evaluate_decision_risk(
            list(X0), path_now, path_dindex)
        if not _allow:
            path_d, path_dindex = path_ego, path_now
            C_label_virtual = "K"
            _, xc, yc, samplesc = _get_path_info(path_dindex)
            X0 = repropagate(path_d, samplesc, xc, yc, X0_g, X0)

    if path_changed != path_dindex:
        controller.get_path_curvature(path=path_d)
        oS, oey_v = _path_to_path_proj(oS, oey_v, path_changed, path_dindex)
        last_X = [ovx, ovy, owz, oS, oey_v, oepsi]
    path_changed = path_dindex

    # ------------------------------------------------------------------
    # 4. DREAM MPC SOLVE
    # ------------------------------------------------------------------
    oa, od, ovx, ovy, owz, oS, oey_v, oepsi = controller.solve_with_risk(
        X0, oa, od, dt, None, None, C_label, X0_g,
        path_d, last_X, path_now, ego_group, path_ego, desired_group,
        vl, vc, vr, path_dindex, C_label_additive, C_label_virtual)

    last_X = [ovx, ovy, owz, oS, oey_v, oepsi]
    X0, X0_g, _, _ = dynamics.propagate(
        X0, [oa[0], od[0]], dt, X0_g, path_d, sample, x_list, y_list, boundary)

    # ------------------------------------------------------------------
    # 5. IDEAM BASELINE (independent ego)
    # ------------------------------------------------------------------
    ideam_X0_panel   = list(X0_ideam)
    ideam_X0_g_panel = list(X0_g_ideam)
    hor_ideam = last_ideam_hor

    try:
        path_now_i = _judge_lane(X0_g_ideam[:2])
        path_ego_i = paths[path_now_i]
        start_i    = {0: "L1", 1: "C1", 2: "R1"}[path_now_i]

        if i == 0:
            _oi = clac_last_X(oa_i, od_i, baseline_mpc.T,
                               path_ego_i, dt, 6, X0_ideam, X0_g_ideam)
            last_X_ideam = list(_oi)

        all_info_i = utils_ideam.get_alllane_lf(
            path_ego_i, X0_g_ideam, path_now_i, vl, vc, vr)
        gd_i, ego_grp_i = utils_ideam.formulate_gap_group(
            path_now_i, last_X_ideam, all_info_i, vl, vc, vr)
        dg_i = decision_maker.decision_making(gd_i, start_i)

        path_d_i, pdi_i, Cl_i, samp_i, xl_i, yl_i, X0_ideam = _Decision_info(
            X0_ideam, X0_g_ideam, path_center, sample_center,
            x_center, y_center, boundary, dg_i, path_ego_i, path_now_i)

        Cla_i = utils_ideam.inquire_C_state(Cl_i, dg_i)
        if Cla_i == "Probe":
            path_d_i, pdi_i, Clv_i = path_ego_i, path_now_i, "K"
            _, xci, yci, sci = _get_path_info(pdi_i)
            X0_ideam = repropagate(path_d_i, sci, xci, yci, X0_g_ideam, X0_ideam)
        else:
            Clv_i = Cl_i

        if path_changed_i != pdi_i and last_X_ideam is not None:
            _oS_i, _oey_i = _path_to_path_proj(
                last_X_ideam[3], last_X_ideam[4], path_changed_i, pdi_i)
            last_X_ideam = [last_X_ideam[0], last_X_ideam[1],
                            last_X_ideam[2], _oS_i, _oey_i, last_X_ideam[5]]
        baseline_mpc.get_path_curvature(path=path_d_i)
        path_changed_i = pdi_i

        res_i = baseline_mpc.iterative_linear_mpc_control(
            X0_ideam, oa_i, od_i, dt,
            None, None, Cl_i, X0_g_ideam, path_d_i, last_X_ideam,
            path_now_i, ego_grp_i, path_ego_i, dg_i,
            vl, vc, vr, pdi_i, Cla_i, Clv_i)

        if res_i is not None:
            oa_i, od_i, _oi0, _oi1, _oi2, _oi3, _oi4, _oi5 = res_i
            last_X_ideam = [_oi0, _oi1, _oi2, _oi3, _oi4, _oi5]
            X0_ideam, X0_g_ideam, _, _ = dynamics.propagate(
                list(X0_ideam), [oa_i[0], od_i[0]], dt,
                list(X0_g_ideam), path_d_i, samp_i, xl_i, yl_i, boundary)
            ideam_X0_panel   = list(X0_ideam)
            ideam_X0_g_panel = list(X0_g_ideam)

            # Build IDEAM horizon for visualization
            _Xi, _Xgi = list(X0_ideam), list(X0_g_ideam)
            _hor = [list(_Xgi)]
            for _k in range(len(oa_i) - 1):
                _Xi, _Xgi, _, _ = dynamics.propagate(
                    _Xi, [oa_i[_k+1], od_i[_k+1]], dt,
                    _Xgi, path_d_i, samp_i, xl_i, yl_i, boundary)
                _hor.append(list(_Xgi))
            hor_ideam = np.array(_hor)
            last_ideam_hor = hor_ideam

    except Exception as e:
        if i % 50 == 0:
            print(f"  [IDEAM] step {i}: {e}")

    # ------------------------------------------------------------------
    # 6. METRICS
    # ------------------------------------------------------------------
    try:
        _s_d, _, _ = find_frenet_coord(
            paths[EGO_LANE], paths[EGO_LANE].xc,
            paths[EGO_LANE].yc, paths[EGO_LANE].samples, X0_g)
    except Exception:
        _s_d = X0[3]
    dream_s.append(float(_s_d))
    dream_vx.append(float(X0[0]))
    dream_acc.append(_safe_first_scalar(oa, default=0.0))

    try:
        _s_i, _, _ = find_frenet_coord(
            paths[EGO_LANE], paths[EGO_LANE].xc,
            paths[EGO_LANE].yc, paths[EGO_LANE].samples, ideam_X0_g_panel)
    except Exception:
        _s_i = X0_ideam[3]
    ideam_s.append(float(_s_i))
    ideam_vx.append(float(ideam_X0_panel[0]))
    ideam_acc.append(_safe_first_scalar(oa_i, default=0.0))

    # ------------------------------------------------------------------
    # 7. DREAM HORIZON ROLLOUT
    # ------------------------------------------------------------------
    _Xv, _Xgv = list(X0), list(X0_g)
    _dream_hor = [list(_Xgv)]
    for _k in range(len(oa) - 1):
        _Xv, _Xgv, _, _ = dynamics.propagate(
            _Xv, [oa[_k+1], od[_k+1]], dt,
            _Xgv, path_d, sample, x_list, y_list, boundary)
        _dream_hor.append(list(_Xgv))
    _dream_hor = np.array(_dream_hor)

    # ------------------------------------------------------------------
    # 8. VISUALIZE
    # ------------------------------------------------------------------
    draw_frame(i, X0_g, X0, ideam_X0_g_panel, ideam_X0_panel,
               vl, vc, vr, risk_field, risk_at_ego,
               tracks, tracks_meta, class_map, frame_idx,
               horizon_dream=_dream_hor,
               horizon_ideam=hor_ideam,
               bg_img=bg_img, bg_extent=bg_extent)

bar.finish()
print()
print("Simulation complete.")
print(f"  DREAM  final s: {dream_s[-1]:.1f} m   vx: {dream_vx[-1]:.2f} m/s")
print(f"  IDEAM  final s: {ideam_s[-1]:.1f} m   vx: {ideam_vx[-1]:.2f} m/s")

# ===========================================================================
# METRICS PLOT
# ===========================================================================

_t = np.arange(len(dream_s)) * dt

with plt.style.context(["science", "no-latex"]):
    fig_m, axes_m = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    fig_m.suptitle(
        f"DREAM vs IDEAM 鈥?exiD recording {rec}  "
        f"(ego track {EGO_TRACK_ID})", fontsize=12)

    _C  = {"DREAM": "C0", "IDEAM": "C1"}
    _LS = {"DREAM": "-",  "IDEAM": "--"}

    axes_m[0, 0].plot(_t, dream_s,  color=_C["DREAM"], ls=_LS["DREAM"], label="DREAM")
    axes_m[0, 0].plot(_t, ideam_s,  color=_C["IDEAM"], ls=_LS["IDEAM"], label="IDEAM")
    axes_m[0, 0].set_xlabel("t [s]"); axes_m[0, 0].set_ylabel("s [m]")
    axes_m[0, 0].set_title("Progress s(t)"); axes_m[0, 0].legend(fontsize=8)

    axes_m[0, 1].plot(_t, dream_vx, color=_C["DREAM"], ls=_LS["DREAM"], label="DREAM")
    axes_m[0, 1].plot(_t, ideam_vx, color=_C["IDEAM"], ls=_LS["IDEAM"], label="IDEAM")
    axes_m[0, 1].set_xlabel("t [s]"); axes_m[0, 1].set_ylabel("vx [m/s]")
    axes_m[0, 1].set_title("Speed vx(t)"); axes_m[0, 1].legend(fontsize=8)

    axes_m[1, 0].plot(_t, dream_acc, color=_C["DREAM"], ls=_LS["DREAM"], label="DREAM")
    axes_m[1, 0].plot(_t, ideam_acc, color=_C["IDEAM"], ls=_LS["IDEAM"], label="IDEAM")
    axes_m[1, 0].axhline(0, color="black", lw=0.5)
    axes_m[1, 0].set_xlabel("t [s]"); axes_m[1, 0].set_ylabel("a [m/s虏]")
    axes_m[1, 0].set_title("Acceleration"); axes_m[1, 0].legend(fontsize=8)

    _t_r = np.arange(len(risk_at_ego_list)) * dt
    axes_m[1, 1].plot(_t_r, risk_at_ego_list, color=_C["DREAM"])
    axes_m[1, 1].fill_between(_t_r, risk_at_ego_list, alpha=0.2, color=_C["DREAM"])
    axes_m[1, 1].set_xlabel("t [s]"); axes_m[1, 1].set_ylabel("R(ego)")
    axes_m[1, 1].set_title("DRIFT Risk at DREAM Ego")

    plt.savefig(os.path.join(save_dir, "metrics_dataset.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig_m)

np.save(os.path.join(save_dir, "metrics_dataset.npy"), {
    "dream_s": dream_s, "ideam_s": ideam_s,
    "dream_vx": dream_vx, "ideam_vx": ideam_vx,
    "dream_acc": dream_acc, "ideam_acc": ideam_acc,
    "risk_at_ego": risk_at_ego_list,
    "recording": rec, "ego_track_id": EGO_TRACK_ID,
})

print(f"\nFrames {save_dir}/")
print(f"Metrics {save_dir}/metrics_dataset.png")
