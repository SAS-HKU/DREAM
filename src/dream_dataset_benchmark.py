"""
dream_dataset_benchmark.py
==========================
DREAM vs IDEAM benchmark with REAL dataset traffic as surrounding agents.

Architecture
------------
┌─────────────────────────────────────────────────────────┐
│  Dataset CSV  →  surrounding vehicles (replay)          │
│  DatasetPath  →  reference lanes (K-means + spline)     │
│                                                         │
│  DREAM ego  (risk-aware MPC + DRIFT veto)               │
│  IDEAM ego  (baseline MPC, no risk, independent)        │
│                                                         │
│  Both egos start at the same recorded position and      │
│  interact with the same dataset traffic each step.      │
└─────────────────────────────────────────────────────────┘

Visualization (uncertainty_merger_DREAM.py style)
-------------------------------------------------
  Top panel   : IDEAM baseline  — no risk overlay, green ego
  Bottom panel: DREAM (ours)    — DRIFT risk overlay, blue ego
  Both panels : orthophoto background + dataset vehicles + MPC horizon

Usage
-----
Set DATASET_DIR, RECORDING_ID, EGO_TRACK_ID then run:
    python dream_dataset_benchmark.py
"""

# ===========================================================================
# IMPORTS  (identical to drift_dataset_visualization.py)
# ===========================================================================
import argparse
import sys, os, math, time, copy, signal, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.transforms import Affine2D
from scipy.ndimage import gaussian_filter as _gf
from scipy.interpolate import make_interp_spline
from sklearn.cluster import KMeans
import cv2
import scienceplots  # noqa: F401

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

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

from config import Config as cfg
from pde_solver import (PDESolver, create_vehicle as drift_create_vehicle)
from Integration.prideam_controller import create_prideam_controller
from Integration.integration_config import get_preset
from tracks_import import read_from_csv

# ADA source adapter (DREAM-ADA benchmark arm)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "Aggressiveness_Modeling"))
from Aggressiveness_Modeling.ADA_drift_source import compute_Q_ADA  # noqa: E402


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
        description="Run DREAM vs IDEAM benchmark against real trajectory datasets."
    )
    parser.add_argument(
        "--dataset-dir",
        default=os.environ.get(
            "DREAM_DATASET_DIR",
            os.path.join(SCRIPT_DIR, "data", "inD"),
        ),
        help="Dataset directory containing XX_tracks.csv and related files.",
    )
    parser.add_argument(
        "--recording-id",
        default=os.environ.get("DREAM_RECORDING_ID", "01"),
        help="Recording id, e.g., 01.",
    )
    parser.add_argument(
        "--ego-track-id",
        type=int,
        default=int(os.environ.get("DREAM_EGO_TRACK_ID", "254")),
        help="Track id used as ego. Ignored when --auto-ego-track is set.",
    )
    parser.add_argument(
        "--auto-ego-track",
        action="store_true",
        help="Automatically choose ego track near scene center.",
    )
    parser.add_argument(
        "--save-dir",
        default=os.path.join(SCRIPT_DIR, "figsave_DREAM_inD_benchmark_04"),
        help="Directory where output frames and metrics are saved.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=int(os.environ.get("DREAM_STEPS", "200")),
        help="Number of simulation steps.",
    )
    parser.add_argument(
        "--integration-mode",
        default=os.environ.get("DREAM_INTEGRATION_MODE", "conservative"),
        help="Integration preset name for PRIDEAM controller.",
    )
    parser.add_argument(
        "--ego-lane",
        type=int,
        default=int(os.environ.get("DREAM_EGO_LANE", "1")),
        help="Reference lane index for the ego (0=left, 1=center, 2=right).",
    )
    parser.add_argument(
        "--save-frames",
        type=_str2bool,
        default=_str2bool(os.environ.get("DREAM_SAVE_FRAMES", "1")),
        help="Whether to save per-step visualization frames.",
    )
    parser.add_argument(
        "--frame-dpi",
        type=int,
        default=int(os.environ.get("DREAM_FRAME_DPI", "150")),
        help="DPI for per-step frame outputs.",
    )
    parser.add_argument(
        "--use-bg",
        type=_str2bool,
        default=_str2bool(os.environ.get("DREAM_USE_BG", "1")),
        help="Whether to use dataset orthophoto background.",
    )
    return parser.parse_args()


CLI_ARGS = _parse_cli_args()

# ===========================================================================
# PARAMETERS
# ===========================================================================

DATASET_DIR    = os.path.abspath(CLI_ARGS.dataset_dir)
RECORDING_ID   = str(CLI_ARGS.recording_id)
EGO_TRACK_ID   = None if CLI_ARGS.auto_ego_track else CLI_ARGS.ego_track_id
EGO_LANE       = CLI_ARGS.ego_lane
N_LANES        = 3
LANE_WIDTH     = 3.75         # m  — fallback parallel-path offset

dt             = 0.1          # simulation step [s]
N_t            = max(1, CLI_ARGS.steps)
WARMUP_S       = 0.5          # DRIFT warm-up [s]

INTEGRATION_MODE = CLI_ARGS.integration_mode
DRIFT_CELL_M     = 2.0        # DRIFT grid cell [m]
SCENE_MARGIN     = 60.0       # grid margin beyond track bbox [m]
EGO_MIN_FRAMES   = 80

# Risk visualisation
RISK_ALPHA        = 0.40
RISK_CMAP         = "jet"
RISK_LEVELS       = 40
RISK_VMAX         = 2.0
RISK_MIN_VIS      = 0.08
RISK_SMOOTH_SIGMA = 2.0    # Gaussian sigma for smoothing risk field
RISK_ALPHA_GAMMA  = 0.8    # gamma compress — keeps map readable

# View: each panel is ego-centred with this half-extent
VIEW_X = 50.0   # m
VIEW_Y = 28.0   # m

# Background
USE_BG  = CLI_ARGS.use_bg
BG_OFFSET_X_M = 0.0
BG_OFFSET_Y_M = 0.0
SAVE_FRAMES = CLI_ARGS.save_frames
FRAME_DPI = CLI_ARGS.frame_dpi

save_dir = os.path.abspath(CLI_ARGS.save_dir)
os.makedirs(save_dir, exist_ok=True)

# ===========================================================================
# HELPERS  (copied from drift_dataset_visualization.py)
# ===========================================================================

def heading_to_psi(heading_deg):
    """Dataset heading → math radians (heading is already CCW from +X)."""
    return math.radians(float(heading_deg))


def draw_vehicle_rect(ax, x, y, psi, length, width,
                      facecolor, edgecolor="black", lw=0.8,
                      zorder=3, alpha=1.0, linestyle="-"):
    rect = mpatches.FancyBboxPatch(
        (-length / 2, -width / 2), length, width,
        boxstyle="round,pad=0.05",
        facecolor=facecolor, edgecolor=edgecolor,
        linewidth=lw, alpha=alpha, zorder=zorder, linestyle=linestyle)
    t = Affine2D().rotate(psi).translate(x, y) + ax.transData
    rect.set_transform(t)
    ax.add_patch(rect)


# -- DatasetPath ---------------------------------------------------------------

class DatasetPath:
    """Cubic-spline path compatible with IDEAM path interface."""

    def __init__(self, x_pts, y_pts, n_samples=600):
        diffs = np.hypot(np.diff(x_pts), np.diff(y_pts))
        keep  = np.concatenate([[True], diffs > 0.05])
        x_pts, y_pts = np.asarray(x_pts)[keep], np.asarray(y_pts)[keep]
        if len(x_pts) < 4:
            raise ValueError("DatasetPath: need ≥ 4 unique control points")
        diffs = np.hypot(np.diff(x_pts), np.diff(y_pts))
        s_raw = np.concatenate([[0.0], np.cumsum(diffs)])
        self._spl_x  = make_interp_spline(s_raw, x_pts, k=3)
        self._spl_y  = make_interp_spline(s_raw, y_pts, k=3)
        self._spl_dx = self._spl_x.derivative()
        self._spl_dy = self._spl_y.derivative()
        self._spl_d2x= self._spl_dx.derivative()
        self._spl_d2y= self._spl_dy.derivative()
        self.s_min, self.s_max = float(s_raw[0]), float(s_raw[-1])
        self.samples = np.linspace(self.s_min, self.s_max, n_samples)
        self.xc = np.array([self(s)[0] for s in self.samples])
        self.yc = np.array([self(s)[1] for s in self.samples])

    @staticmethod
    def _f(v):
        return float(np.asarray(v).flat[0])

    def _c(self, s):
        s = self._f(s)
        return max(self.s_min, min(s, self.s_max))

    def __call__(self, s):
        s = self._c(s)
        return self._f(self._spl_x(s)), self._f(self._spl_y(s))

    def get_theta_r(self, s):
        s  = self._c(s)
        return math.atan2(self._f(self._spl_dy(s)), self._f(self._spl_dx(s)))

    def get_len(self):  return self.s_max

    def get_k(self, s):
        s   = self._c(s)
        dx, dy   = self._f(self._spl_dx(s)), self._f(self._spl_dy(s))
        d2x, d2y = self._f(self._spl_d2x(s)), self._f(self._spl_d2y(s))
        den = (dx**2 + dy**2)**1.5
        return (dx * d2y - dy * d2x) / den if den > 1e-9 else 0.0

    def get_cartesian_coords(self, s, ey):
        x, y = self(s); psi = self.get_theta_r(s)
        return x - ey * math.sin(psi), y + ey * math.cos(psi)


# -- Lane path fitting ---------------------------------------------------------

def fit_lane_paths(tracks, tracks_meta, class_map, ego_track_id, n_lanes, bin_w=5.0):
    ego_tr = tracks[ego_track_id]
    h_ref  = math.radians(float(np.nanmean(ego_tr["heading"])))
    ch, sh = math.cos(h_ref), math.sin(h_ref)
    pts = []
    for tm in tracks_meta:
        tid = tm["trackId"]
        if class_map.get(tid) not in ("car", "van", "truck"): continue
        h_m = float(np.nanmean(tracks[tid]["heading"]))
        if abs(((h_m - math.degrees(h_ref) + 180) % 360) - 180) > 40: continue
        for xi, yi in zip(tracks[tid]["xCenter"], tracks[tid]["yCenter"]):
            pts.append((ch*xi + sh*yi, -sh*xi + ch*yi, xi, yi))
    if len(pts) < n_lanes * 10:
        return _parallel_fallback(ego_tr, n_lanes)
    pts = np.array(pts)
    km = KMeans(n_clusters=n_lanes, random_state=42, n_init=10)
    lbl = km.fit_predict(pts[:, 1:2])
    order = np.argsort(km.cluster_centers_.ravel())
    paths = []
    for li in range(n_lanes):
        ci = order[li]; mask = lbl == ci; cp = pts[mask]
        s_v = cp[:, 0]; s_min, s_max = s_v.min(), s_v.max()
        bins = np.arange(s_min, s_max + bin_w, bin_w)
        xb, yb = [], []
        for b in bins[:-1]:
            mb = (s_v >= b) & (s_v < b + bin_w)
            if mb.sum() < 2: continue
            xb.append(float(np.median(cp[mb, 2])))
            yb.append(float(np.median(cp[mb, 3])))
        if len(xb) < 4:
            paths.append(None); continue
        try:
            paths.append(DatasetPath(np.array(xb), np.array(yb)))
            print(f"[LANE] lane {li}: s_max={paths[-1].s_max:.0f}m  pts={len(xb)}")
        except Exception:
            paths.append(None)
    good = [p for p in paths if p is not None]
    if not good:
        return _parallel_fallback(ego_tr, n_lanes)
    for li in range(n_lanes):
        if paths[li] is None:
            ref = good[0]; off = (li - n_lanes // 2) * LANE_WIDTH
            psi0 = ref.get_theta_r(ref.samples[0])
            xo = ref.xc - off * math.sin(psi0)
            yo = ref.yc + off * math.cos(psi0)
            paths[li] = DatasetPath(xo, yo)
    return paths


def _parallel_fallback(ego_tr, n_lanes):
    h = math.radians(float(np.nanmean(ego_tr["heading"])))
    x0, y0 = float(ego_tr["xCenter"].mean()), float(ego_tr["yCenter"].mean())
    span = max(float(ego_tr["xCenter"].max()-ego_tr["xCenter"].min()), 200.0)
    ch, sh = math.cos(h), math.sin(h)
    result = []
    for li in range(n_lanes):
        off = (li - n_lanes // 2) * LANE_WIDTH
        xs = np.linspace(x0 - off*sh - 0.5*span*ch, x0 - off*sh + 0.5*span*ch, 20)
        ys = np.linspace(y0 + off*ch - 0.5*span*sh, y0 + off*ch + 0.5*span*sh, 20)
        result.append(DatasetPath(xs, ys))
        print(f"[LANE] fallback lane {li}: s_max={result[-1].s_max:.0f}m")
    return result


def make_ego_track_paths(ego_track, n_lanes, lane_width, ego_lane_idx=1):
    """
    Build N reference paths centred on the ego's RECORDED trajectory.

    Why this matters for complex scenes (intersections, roundabouts):
      fit_lane_paths() clusters ALL vehicle trajectories and produces average
      paths that may not follow the actual road geometry.  For a roundabout
      the K-means centroid can be a straight line cutting across the loop.

      Here we fit a cubic spline DIRECTLY through the ego's GPS track
      (which is always on the road) and offset ±LANE_WIDTH for the adjacent
      lane reference paths.  The planner then follows the exact road shape.

    Parameters
    ----------
    ego_track   : dataset track dict (with "xCenter", "yCenter" arrays)
    n_lanes     : number of parallel paths to create
    lane_width  : lateral offset between adjacent lanes [m]
    ego_lane_idx: which path index is the ego's lane (default 1 = centre)
    """
    x_pts = np.asarray(ego_track["xCenter"], dtype=float)
    y_pts = np.asarray(ego_track["yCenter"], dtype=float)

    # Remove duplicate/near-stationary points (GPS noise)
    dist = np.hypot(np.diff(x_pts), np.diff(y_pts))
    keep = np.concatenate([[True], dist > 0.05])
    x_pts, y_pts = x_pts[keep], y_pts[keep]

    try:
        centre = DatasetPath(x_pts, y_pts)
    except Exception:
        print("[LANE] ego-track path failed; using parallel fallback")
        return _parallel_fallback(ego_track, n_lanes)

    print(f"[LANE] ego-track centre path: s_max={centre.s_max:.1f}m  "
          f"pts={len(x_pts)}")

    # Build parallel paths at each lane offset around centre
    paths = []
    for li in range(n_lanes):
        offset = (li - ego_lane_idx) * lane_width   # e.g. -3.75, 0, +3.75 m
        if abs(offset) < 0.01:
            paths.append(centre)
            continue
        # Sample perpendicular offsets along the centre path
        x_off = np.empty(len(centre.samples))
        y_off = np.empty(len(centre.samples))
        for k, s in enumerate(centre.samples):
            psi = centre.get_theta_r(s)
            xc, yc = centre(s)
            x_off[k] = xc - offset * math.sin(psi)
            y_off[k] = yc + offset * math.cos(psi)
        try:
            paths.append(DatasetPath(x_off, y_off))
            print(f"[LANE] lane {li}: offset={offset:+.2f}m  "
                  f"s_max={paths[-1].s_max:.1f}m")
        except Exception:
            paths.append(centre)   # fallback: reuse centre path

    return paths


def select_ego_track(tracks, tracks_meta, class_map, min_frames):
    cx = np.mean([t["xCenter"].mean() for t in tracks])
    cy = np.mean([t["yCenter"].mean() for t in tracks])
    cands = [(math.hypot(tracks[tm["trackId"]]["xCenter"].mean()-cx,
                         tracks[tm["trackId"]]["yCenter"].mean()-cy), tm["trackId"])
             for tm in tracks_meta
             if class_map.get(tm["trackId"]) in ("car","van")
             and tm["numFrames"] >= min_frames]
    if not cands:
        cands = [(0, tm["trackId"]) for tm in tracks_meta if tm["numFrames"] >= min_frames]
    return min(cands)[1]


def build_surrounding_arrays(frame_idx, ego_id, tracks, tracks_meta, class_map, paths, fr,
                             _diag=False, ego_xy=None, ego_s=None):
    """
    Build MPC lane arrays and DRIFT vehicle list from dataset at frame_idx.

    IMPORTANT — filtering behaviour:
      ALL active vehicles go into drift_l (feeds DRIFT risk field).
      Only vehicles with lateral offset |ey| ≤ 12 m from the nearest reference
      path go into vl/vc/vr (feeds MPC planners).

    For intersection / cross-road scenarios this means cross-road traffic is
    EXCLUDED from MPC but still contributes to the DRIFT risk field.
    Run with _diag=True (auto-set on step 0 and every 50 steps) to print
    how many vehicles reach each lane array vs. how many are filtered out.
    """
    buckets = [[], [], []]
    drift_l = []
    vid = 1
    n_active = 0
    n_filtered_ey = 0

    for tm in tracks_meta:
        tid = tm["trackId"]
        if tid == ego_id: continue
        if not (tm["initialFrame"] <= frame_idx <= tm["finalFrame"]): continue
        fi  = frame_idx - tm["initialFrame"]
        tr  = tracks[tid]
        x, y = float(tr["xCenter"][fi]), float(tr["yCenter"][fi])
        psi  = heading_to_psi(tr["heading"][fi])
        vl   = float(tr["lonVelocity"][fi])
        al   = float(tr["lonAcceleration"][fi]) if "lonAcceleration" in tr else 0.0
        cls  = class_map.get(tid, "car")
        if abs(vl) < 0.5 and cls not in ("car","van","truck"): continue
        n_active += 1
        bl, bey, bs, bep = -1, float("inf"), 0.0, 0.0
        for li, path in enumerate(paths):
            try:
                s, ey, ep = find_frenet_coord(path, path.xc, path.yc, path.samples, [x,y,psi])
                if abs(ey) < abs(bey): bl, bey, bs, bep = li, ey, s, ep
            except Exception: pass
        vc_ = "truck" if cls in ("truck","van") else "car"
        vxg, vyg = vl*math.cos(psi), vl*math.sin(psi)
        v = drift_create_vehicle(vid=vid, x=x, y=y, vx=vxg, vy=vyg, vclass=vc_)
        v["heading"], v["a"] = psi, al
        drift_l.append(v); vid += 1
        # Use road half-width + one extra lane as the MPC detection radius.
        # This is wider than the hard-coded 12 m when the road is narrow,
        # ensuring vehicles on adjacent lanes (up to one lane outside road edge)
        # are still seen by the planner's CBF.
        _mpc_ey_limit = max(12.0, ROAD_HALF_WIDTH + LANE_WIDTH)
        if bl < 0 or abs(bey) > _mpc_ey_limit:
            n_filtered_ey += 1
            # ── Cartesian emergency fallback ──────────────────────────────
            # Even if Frenet projection fails (cross-road, large ey), inject
            # the vehicle into the centre-lane array if it is physically close
            # AND ahead of the ego — this is the "planner has no information"
            # case that causes collisions.
            if ego_xy is not None and ego_s is not None:
                dist_m = float(np.hypot(x - ego_xy[0], y - ego_xy[1]))
                if dist_m < 20.0:          # within 20 m Cartesian
                    # Check it is roughly ahead (positive dot with ego heading)
                    _ego_heading = math.atan2(ego_xy[1] - y, ego_xy[0] - x)
                    _ahead = (x - ego_xy[0]) * math.cos(float(paths[EGO_LANE].get_theta_r(ego_s))) + \
                             (y - ego_xy[1]) * math.sin(float(paths[EGO_LANE].get_theta_r(ego_s)))
                    if _ahead > -3.0:      # ahead or alongside (not behind >3 m)
                        _s_proxy = ego_s + dist_m   # approximate s
                        buckets[EGO_LANE].append(
                            np.array([_s_proxy, 0.0, 0.0, x, y, psi, vl, al]))
            continue
        buckets[bl].append(np.array([bs, bey, bep, x, y, psi, vl, al]))

    if _diag:
        n_mpc = sum(len(b) for b in buckets)
        print(f"  [SURR DIAG] frame={frame_idx}: "
              f"active={n_active}  in_MPC={n_mpc}  "
              f"filtered_ey>{int(_mpc_ey_limit)}m={n_filtered_ey}  "
              f"lane sizes L={len(buckets[0])} C={len(buckets[1])} R={len(buckets[2])}  "
              f"ego_s={ego_s:.1f}" if ego_s is not None else "")
        if n_mpc == 0 and n_active > 0:
            print(f"  [SURR WARN] ALL {n_active} active vehicles filtered out of MPC arrays!")
            print(f"              Vehicles on cross-roads have |ey|>{int(_mpc_ey_limit)}m.")
            print(f"              Cartesian emergency fallback active (20m radius).")

    def _pad(b):
        if not b: return np.array([[1e4,0,0,1e4,0,0,0.01,0]])
        a = np.array(b); return a[np.argsort(a[:,0])]

    return _pad(buckets[0]), _pad(buckets[1]), _pad(buckets[2]), drift_l


# -- IDEAM adapters (same as drift_dataset_visualization.py) ------------------

_paths = None   # set after path fitting
_GROUP_TO_LANE = {"L1":0,"L2":0,"C1":1,"C2":1,"R1":2,"R2":2}

def _judge_lane(ego_xy):
    best_li, best_ey = 0, float("inf")
    for li, path in enumerate(_paths):
        try:
            _, ey, _ = find_frenet_coord(path, path.xc, path.yc, path.samples,
                                          [ego_xy[0], ego_xy[1], 0.0])
            if abs(ey) < abs(best_ey): best_li, best_ey = li, ey
        except Exception: pass
    return best_li

def _get_path_info(idx):
    p = _paths[idx]; return p, p.xc, p.yc, p.samples


def _bind_dataset_path_adapters_to_utils(utils_obj, tag):
    """
    Bind DecisionMaking.util path lookup to dataset paths.

    DecisionMaking.util.get_alllane_lf()/get_remap_vehicles() call the module-
    level get_path_info(). Without rebinding, they use Path.path highway refs
    and ignore the dataset-fitted paths.
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


def _proj_single(s_val, ey_val, old, new):
    """Project one (s, ey) scalar from path `old` onto path `new`."""
    po, pn = _paths[old], _paths[new]
    x_w, y_w = po.get_cartesian_coords(s_val, ey_val)
    psi_old  = po.get_theta_r(s_val)
    try:
        sn, eyn, _ = find_frenet_coord(pn, pn.xc, pn.yc, pn.samples,
                                        [x_w, y_w, psi_old])
    except Exception:
        sn, eyn = s_val, ey_val
    return float(np.asarray(sn).flat[0]), float(np.asarray(eyn).flat[0])


def _path_to_path_proj(s_old, ey_old, old, new):
    """
    Project Frenet coordinates from path `old` to path `new`.

    last_X[3] (oS) and last_X[4] (oey) are T-length numpy arrays
    (the MPC horizon trajectory), not scalars.  predict_motion() indexes
    them as os[i], so we must preserve the array shape.
    """
    s_arr  = np.asarray(s_old).ravel()
    ey_arr = np.asarray(ey_old).ravel()

    if s_arr.size <= 1:
        # Scalar path — return plain floats
        return _proj_single(float(s_arr.flat[0]), float(ey_arr.flat[0]), old, new)

    # Array path (T-length horizon) — project element-wise
    s_new  = np.empty_like(s_arr,  dtype=float)
    ey_new = np.empty_like(ey_arr, dtype=float)
    for k in range(s_arr.size):
        s_new[k], ey_new[k] = _proj_single(s_arr[k], ey_arr[k], old, new)
    return s_new, ey_new

def _Decision_info(x0, x0_g, pc, sc, xc, yc, bound, dg, path_now_obj, path_now_idx):
    target = dg.get("name","C1")
    pdi = max(0, min(_GROUP_TO_LANE.get(target, path_now_idx), len(_paths)-1))
    pd  = _paths[pdi]
    smp, xl, yl = sc[pdi], xc[pdi], yc[pdi]
    sp, xp, yp  = sc[path_now_idx], xc[path_now_idx], yc[path_now_idx]
    x0_post = repropagate(path_now_obj, sp, xp, yp, x0_g, list(x0))
    sl = dg.get("sl", None)
    ta = sl[0] if (sl is not None and len(sl)>0) else 1e4
    is_short = abs(ta - x0_post[3]) <= 7.5
    if path_now_idx != pdi and not is_short:
        C = "R" if pdi > path_now_idx else "L"
        x0u = repropagate(pd, smp, xl, yl, x0_g, list(x0))
        return pd, pdi, C, smp, xl, yl, x0u
    smp, xl, yl = sc[path_now_idx], xc[path_now_idx], yc[path_now_idx]
    x0u = repropagate(path_now_obj, smp, xl, yl, x0_g, list(x0))
    return path_now_obj, path_now_idx, "K", smp, xl, yl, x0u

# ===========================================================================
# VISUALIZATION  — three-panel benchmark
# ===========================================================================

def _render_panel(ax, ego_x_px, ego_y_px, ego_psi_px, ego_vx,
                  ego_col, ego_label,
                  frame_idx, tracks, tracks_meta, class_map,
                  bg_img, risk_field,
                  horizon=None, draw_risk=False,
                  vmax=RISK_VMAX,
                  is_gt=False):
    """
    Draw one panel in pixel-space (shared helper for all three panels).
    Mirrors draw_frame_drift_overlay exactly.
    """
    _sc = _ortho_px_m * _vis_scale_down

    ax.cla()
    if bg_img is not None:
        ax.imshow(bg_img, origin="upper", zorder=0)
    else:
        ax.set_facecolor("#111111")

    # DRIFT risk (DREAM panel only)
    if draw_risk and risk_field is not None and _cfg_X_vis is not None:
        R_sm = _gf(risk_field, sigma=RISK_SMOOTH_SIGMA)
        nz   = R_sm[R_sm > RISK_MIN_VIS]
        vmax = float(np.percentile(nz, 95)) if nz.size > 50 else RISK_VMAX
        vmax = max(vmax, RISK_MIN_VIS + 1e-3)
        Rn = (np.clip(R_sm, RISK_MIN_VIS, vmax) - RISK_MIN_VIS) / (vmax - RISK_MIN_VIS)
        Rn = np.power(np.clip(Rn, 0, 1), RISK_ALPHA_GAMMA)
        Rm = np.ma.masked_less_equal(Rn, 0.0)
        if np.ma.count(Rm) > 0:
            ax.contourf(_cfg_X_vis, _cfg_Y_vis, Rm,
                        levels=np.linspace(0.02, 1.0, RISK_LEVELS),
                        cmap=RISK_CMAP, alpha=RISK_ALPHA,
                        zorder=2, antialiased=True)

    # Dataset vehicles via bboxVis.
    # For ground truth (is_gt=True): include ALL tracks (ego too) using
    # their recorded bboxVis — pure dataset replay, ego highlighted in orange.
    # For planner panels: exclude EGO_TRACK_ID (drawn separately below).
    for tm in tracks_meta:
        tid = tm["trackId"]
        if not is_gt and tid == EGO_TRACK_ID: continue
        if not (tm["initialFrame"] <= frame_idx <= tm["finalFrame"]): continue
        tr   = tracks[tid]
        fi   = frame_idx - tm["initialFrame"]
        cls_ = class_map.get(tid, "car")
        is_ego_veh = (tid == EGO_TRACK_ID)
        fc = "#F4511E" if is_ego_veh else (
             "#FF8C00" if cls_ in ("truck","van") else "#AED6F1")
        ec  = "red" if is_ego_veh else "black"
        lw_ = 1.2  if is_ego_veh else 0.5
        z   = 6    if is_ego_veh else 4
        if tr.get("bboxVis") is not None:
            bbox = np.asarray(tr["bboxVis"][fi], dtype=float) / _vis_scale_down
            ax.add_patch(plt.Polygon(bbox, closed=True, facecolor=fc,
                                     edgecolor=ec, linewidth=lw_,
                                     alpha=0.90, zorder=z))
        elif "xCenterVis" in tr:
            cx = float(tr["xCenterVis"][fi]) / _vis_scale_down
            cy = float(tr["yCenterVis"][fi]) / _vis_scale_down
            ax.add_patch(plt.Circle((cx, cy), radius=2.0/_vis_scale_down,
                                     facecolor=fc, edgecolor=ec,
                                     linewidth=lw_, alpha=0.90, zorder=z))

    # MPC horizon (planner panels only)
    if not is_gt and horizon is not None and len(horizon) > 0:
        h = np.asarray(horizon)
        if h.ndim == 2 and h.shape[1] >= 2:
            hx, hy = h[:,0]/_sc, -h[:,1]/_sc
            ax.plot(hx, hy, color="#00BCD4", lw=1.5, ls="--", zorder=7)
            ax.scatter(hx, hy, color="#00BCD4", s=5, zorder=7)

    # Ego vehicle rectangle (planner panels only — GT uses bboxVis above)
    if not is_gt:
        draw_vehicle_rect(ax, ego_x_px, ego_y_px, ego_psi_px,
                          4.5/_sc, 2.0/_sc, ego_col,
                          edgecolor="navy", lw=1.2, zorder=8)
        ax.text(ego_x_px + 1.0/_sc, ego_y_px + 1.5/_sc,
                f"{ego_vx:.1f} m/s",
                fontsize=5, color="black", style="oblique", zorder=9)

    # Viewport centred on ego, clamped to image
    vx_px, vy_px = VIEW_X/_sc, VIEW_Y/_sc
    x0v, x1v   = ego_x_px - vx_px, ego_x_px + vx_px
    ytop, ybot  = ego_y_px - vy_px, ego_y_px + vy_px
    if bg_img is not None:
        _bh, _bw = bg_img.shape[:2]
        if x0v < 0:    x1v -= x0v;         x0v = 0
        if x1v > _bw:  x0v -= x1v - _bw;   x1v = _bw
        if ytop < 0:   ybot -= ytop;        ytop = 0
        if ybot > _bh: ytop -= ybot - _bh;  ybot = _bh
        x0v  = max(0, x0v);  x1v  = min(_bw, x1v)
        ytop = max(0, ytop); ybot = min(_bh, ybot)

    ax.set_xlim(x0v, x1v)
    ax.set_ylim(ybot, ytop)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.set_title(ego_label, fontsize=9, fontweight="bold")

    return vmax


def draw_three_panel(step, frame_idx,
                     X0_g_gt,   vx_gt,         # ground truth (replayed)
                     X0_g_ideam, X0_ideam,      # IDEAM planner
                     X0_g_dream,  X0_dream,     # DREAM planner
                     risk_field, risk_at_ego,
                     tracks, tracks_meta, class_map,
                     bg_img, bg_extent,
                     horizon_ideam=None, horizon_dream=None):
    """
    Three-panel benchmark visualisation.

    Left   : Ground Truth   — replays the ACTUAL recorded trajectory
    Middle : IDEAM baseline — MPC-CBF replaces the ego, no risk awareness
    Right  : DREAM (ours)   — risk-aware MPC replaces the ego

    All three panels share the same dataset surrounding vehicles (replay).
    This directly answers: "how does each controller interact with the same
    real-world traffic compared to what the human driver actually did?"
    """
    _sc = _ortho_px_m * _vis_scale_down

    fig = plt.gcf()
    fig.clf()
    ax_gt    = fig.add_subplot(1, 3, 1)
    ax_ideam = fig.add_subplot(1, 3, 2)
    ax_dream = fig.add_subplot(1, 3, 3)

    # ── Ground truth panel (no horizon — we just replay) ─────────────────
    gt_x_px   =  X0_g_gt[0] / _sc
    gt_y_px   = -X0_g_gt[1] / _sc
    gt_psi_px = -X0_g_gt[2]
    _render_panel(ax_gt, gt_x_px, gt_y_px, gt_psi_px, vx_gt,
                  "#F4511E",
                  f"Ground Truth  |  t={step*dt:.1f} s  ({vx_gt:.1f} m/s)",
                  frame_idx, tracks, tracks_meta, class_map,
                  bg_img, risk_field, horizon=None, draw_risk=False,
                  is_gt=True)    # ALL vehicles drawn via recorded bboxVis

    # ── IDEAM baseline panel ──────────────────────────────────────────────
    id_x_px   =  X0_g_ideam[0] / _sc
    id_y_px   = -X0_g_ideam[1] / _sc
    id_psi_px = -X0_g_ideam[2]
    _render_panel(ax_ideam, id_x_px, id_y_px, id_psi_px, X0_ideam[0],
                  "#4CAF50", "IDEAM (MPC-CBF baseline)",
                  frame_idx, tracks, tracks_meta, class_map,
                  bg_img, risk_field, horizon=horizon_ideam, draw_risk=False)

    # ── DREAM panel ───────────────────────────────────────────────────────
    dr_x_px   =  X0_g_dream[0] / _sc
    dr_y_px   = -X0_g_dream[1] / _sc
    dr_psi_px = -X0_g_dream[2]
    vmax = _render_panel(ax_dream, dr_x_px, dr_y_px, dr_psi_px, X0_dream[0],
                         "#2196F3", "DREAM (risk-aware)",
                         frame_idx, tracks, tracks_meta, class_map,
                         bg_img, risk_field, horizon=horizon_dream, draw_risk=True)

    # Risk badge on DREAM panel
    rc = "red" if risk_at_ego > 1.5 else "orange" if risk_at_ego > 0.5 else "lime"
    ax_dream.text(0.985, 0.965, f"R={risk_at_ego:.2f}",
                  transform=ax_dream.transAxes, ha="right", va="top",
                  color=rc, fontsize=7, fontweight="bold",
                  bbox=dict(boxstyle="round", facecolor="black", alpha=0.55))

    # Colorbar for DREAM risk
    if _cfg_X_vis is not None:
        import matplotlib.cm as _mcm
        _sm = _mcm.ScalarMappable(
            norm=plt.Normalize(vmin=0, vmax=vmax),
            cmap=plt.colormaps[RISK_CMAP])
        _sm.set_array([])
        cbar = fig.colorbar(_sm, ax=ax_dream, fraction=0.030, pad=0.010)
        cbar.set_label(f"Risk  (vmax={vmax:.1f})", fontsize=6)
        cbar.ax.tick_params(labelsize=5)

    if SAVE_FRAMES:
        plt.savefig(
            os.path.join(save_dir, f"{step}.png"),
            dpi=FRAME_DPI,
            bbox_inches="tight",
        )


def draw_four_panel(step, frame_idx,
                    X0_g_gt,    vx_gt,
                    X0_g_ideam, X0_ideam,
                    X0_g_dream, X0_dream,
                    X0_g_ada,   X0_ada,
                    risk_field,     risk_at_ego,
                    risk_field_ada, risk_at_ego_ada,
                    tracks, tracks_meta, class_map,
                    bg_img, bg_extent,
                    horizon_ideam=None, horizon_dream=None, horizon_ada=None):
    """
    Four-panel benchmark: Ground Truth | IDEAM | DREAM (GVF) | DREAM (ADA).

    The two DREAM panels share the same vmax so risk magnitudes are directly
    comparable.  Remaining geometry/shape differences between the GVF and ADA
    panels are structural — not amplitude artefacts.
    """
    _sc = _ortho_px_m * _vis_scale_down

    fig = plt.gcf()
    fig.clf()
    ax_gt    = fig.add_subplot(1, 4, 1)
    ax_ideam = fig.add_subplot(1, 4, 2)
    ax_dream = fig.add_subplot(1, 4, 3)
    ax_ada   = fig.add_subplot(1, 4, 4)

    # Shared vmax across both DREAM panels (computed from GVF field)
    _shared_vmax = RISK_VMAX
    if risk_field is not None:
        _nz = _gf(risk_field, sigma=RISK_SMOOTH_SIGMA)
        _nz = _nz[_nz > RISK_MIN_VIS]
        if _nz.size > 50:
            _shared_vmax = float(np.percentile(_nz, 95))
        _shared_vmax = max(_shared_vmax, RISK_MIN_VIS + 1e-3)

    # ── Ground Truth ──────────────────────────────────────────────────────
    gt_x_px   =  X0_g_gt[0] / _sc
    gt_y_px   = -X0_g_gt[1] / _sc
    gt_psi_px = -X0_g_gt[2]
    _render_panel(ax_gt, gt_x_px, gt_y_px, gt_psi_px, vx_gt,
                  "#F4511E",
                  f"Ground Truth  |  t={step*dt:.1f} s  ({vx_gt:.1f} m/s)",
                  frame_idx, tracks, tracks_meta, class_map,
                  bg_img, risk_field, horizon=None, draw_risk=False,
                  is_gt=True)

    # ── IDEAM baseline ────────────────────────────────────────────────────
    id_x_px   =  X0_g_ideam[0] / _sc
    id_y_px   = -X0_g_ideam[1] / _sc
    id_psi_px = -X0_g_ideam[2]
    _render_panel(ax_ideam, id_x_px, id_y_px, id_psi_px, X0_ideam[0],
                  "#4CAF50", "IDEAM (MPC-CBF baseline)",
                  frame_idx, tracks, tracks_meta, class_map,
                  bg_img, risk_field, horizon=horizon_ideam, draw_risk=False)

    # ── DREAM (GVF source) ────────────────────────────────────────────────
    dr_x_px   =  X0_g_dream[0] / _sc
    dr_y_px   = -X0_g_dream[1] / _sc
    dr_psi_px = -X0_g_dream[2]
    _render_panel(ax_dream, dr_x_px, dr_y_px, dr_psi_px, X0_dream[0],
                  "#2196F3", "DREAM (GVF-DRIFT)",
                  frame_idx, tracks, tracks_meta, class_map,
                  bg_img, risk_field, horizon=horizon_dream, draw_risk=True,
                  vmax=_shared_vmax)
    rc = "red" if risk_at_ego > 1.5 else "orange" if risk_at_ego > 0.5 else "lime"
    ax_dream.text(0.985, 0.965, f"R={risk_at_ego:.2f}",
                  transform=ax_dream.transAxes, ha="right", va="top",
                  color=rc, fontsize=7, fontweight="bold",
                  bbox=dict(boxstyle="round", facecolor="black", alpha=0.55))

    # ── DREAM (ADA source) ────────────────────────────────────────────────
    da_x_px   =  X0_g_ada[0] / _sc
    da_y_px   = -X0_g_ada[1] / _sc
    da_psi_px = -X0_g_ada[2]
    _render_panel(ax_ada, da_x_px, da_y_px, da_psi_px, X0_ada[0],
                  "#9C27B0", "DREAM (ADA-DRIFT)",
                  frame_idx, tracks, tracks_meta, class_map,
                  bg_img, risk_field_ada, horizon=horizon_ada, draw_risk=True,
                  vmax=_shared_vmax)
    rc_a = "red" if risk_at_ego_ada > 1.5 else "orange" if risk_at_ego_ada > 0.5 else "lime"
    ax_ada.text(0.985, 0.965, f"R={risk_at_ego_ada:.2f}",
                transform=ax_ada.transAxes, ha="right", va="top",
                color=rc_a, fontsize=7, fontweight="bold",
                bbox=dict(boxstyle="round", facecolor="black", alpha=0.55))

    # Shared colorbar on the ADA panel (rightmost DREAM panel)
    if _cfg_X_vis is not None:
        import matplotlib.cm as _mcm
        _sm = _mcm.ScalarMappable(
            norm=plt.Normalize(vmin=0, vmax=_shared_vmax),
            cmap=plt.colormaps[RISK_CMAP])
        _sm.set_array([])
        cbar = fig.colorbar(_sm, ax=ax_ada, fraction=0.030, pad=0.010)
        cbar.set_label(f"Risk  (vmax={_shared_vmax:.1f})", fontsize=6)
        cbar.ax.tick_params(labelsize=5)

    if SAVE_FRAMES:
        plt.savefig(
            os.path.join(save_dir, f"{step}.png"),
            dpi=FRAME_DPI,
            bbox_inches="tight",
        )


# keep old name as alias for callers that haven't been updated yet
def draw_comparison_panel(step, frame_idx,
                          X0_g_dream, X0_dream,
                          X0_g_ideam, X0_ideam,
                          vl, vc, vr,
                          risk_field, risk_at_ego,
                          tracks, tracks_meta, class_map,
                          bg_img, bg_extent,
                          horizon_dream=None, horizon_ideam=None):
    """
    Two-panel comparison rendered in the SAME pixel-space as
    drift_dataset_visualization.py / draw_frame_drift_overlay:
      Top    : IDEAM baseline  — no risk, green ego
      Bottom : DREAM (ours)    — DRIFT risk overlay, blue ego

    Layers per panel (all in pixel-space coordinates):
      0  background imshow         (no extent → native pixel axes)
      2  DRIFT risk contourf       (DREAM panel only, via _cfg_X_vis/_cfg_Y_vis)
      4  dataset vehicle bboxVis   (same as TrackVisualizer)
      7  MPC horizon               (world → pixel)
      8  ego rectangle             (world → pixel)
    """
    _sc = _ortho_px_m * _vis_scale_down   # metres-to-pixel factor

    fig = plt.gcf()
    fig.clf()
    ax_top = fig.add_subplot(2, 1, 1)
    ax_bot = fig.add_subplot(2, 1, 2)

    for ax, ego_g, ego_s, horizon, draw_risk, title_str, ego_col in [
        (ax_top, X0_g_ideam, X0_ideam, horizon_ideam, False,
         "IDEAM (baseline)", "#4CAF50"),
        (ax_bot, X0_g_dream,  X0_dream,  horizon_dream, True,
         "DREAM (ours)",     "#2196F3"),
    ]:
        ax.cla()

        # ── 1. Background (identical to draw_frame_drift_overlay) ─────────
        if bg_img is not None:
            ax.imshow(bg_img, origin="upper", zorder=0)
        else:
            ax.set_facecolor("#111111")

        # ── 2. DRIFT risk in pixel space (DREAM panel only) ───────────────
        vmax = RISK_VMAX
        if draw_risk and risk_field is not None and _cfg_X_vis is not None:
            R_sm = _gf(risk_field, sigma=RISK_SMOOTH_SIGMA)
            nz   = R_sm[R_sm > RISK_MIN_VIS]
            vmax = float(np.percentile(nz, 95)) if nz.size > 50 else RISK_VMAX
            vmax = max(vmax, RISK_MIN_VIS + 1e-3)
            Rn = (np.clip(R_sm, RISK_MIN_VIS, vmax) - RISK_MIN_VIS) / (vmax - RISK_MIN_VIS)
            Rn = np.power(np.clip(Rn, 0, 1), RISK_ALPHA_GAMMA)
            Rm = np.ma.masked_less_equal(Rn, 0.0)
            if np.ma.count(Rm) > 0:
                ax.contourf(_cfg_X_vis, _cfg_Y_vis, Rm,
                            levels=np.linspace(0.02, 1.0, RISK_LEVELS),
                            cmap=RISK_CMAP, alpha=RISK_ALPHA,
                            zorder=2, antialiased=True)

        # ── 3. Dataset vehicles via bboxVis (pixel-perfect alignment) ─────
        for tm in tracks_meta:
            tid = tm["trackId"]
            if tid == EGO_TRACK_ID: continue
            if not (tm["initialFrame"] <= frame_idx <= tm["finalFrame"]): continue
            tr   = tracks[tid]
            fi   = frame_idx - tm["initialFrame"]
            cls_ = class_map.get(tid, "car")
            fc   = "#FF8C00" if cls_ in ("truck","van") else "#AED6F1"
            if tr.get("bboxVis") is not None:
                bbox = np.asarray(tr["bboxVis"][fi], dtype=float) / _vis_scale_down
                ax.add_patch(plt.Polygon(bbox, closed=True, facecolor=fc,
                                         edgecolor="black", linewidth=0.5,
                                         alpha=0.85, zorder=4))
            elif "xCenterVis" in tr:
                cx = float(tr["xCenterVis"][fi]) / _vis_scale_down
                cy = float(tr["yCenterVis"][fi]) / _vis_scale_down
                ax.add_patch(plt.Circle((cx, cy), radius=2.0 / _vis_scale_down,
                                         facecolor=fc, edgecolor="black",
                                         linewidth=0.5, alpha=0.85, zorder=4))

        # ── 4. MPC horizon in pixel space ─────────────────────────────────
        if horizon is not None and len(horizon) > 0:
            h = np.asarray(horizon)
            if h.ndim == 2 and h.shape[1] >= 2:
                hx = h[:,0] / _sc
                hy = -h[:,1] / _sc    # Y flipped
                ax.plot(hx, hy, color="#00BCD4", lw=1.5, ls="--", zorder=7)
                ax.scatter(hx, hy, color="#00BCD4", s=5, zorder=7)

        # ── 5. Ego vehicle in pixel space ─────────────────────────────────
        ego_x_px   =  ego_g[0] / _sc
        ego_y_px   = -ego_g[1] / _sc          # Y flipped
        ego_psi_px = -ego_g[2]                 # heading sign flip for pixel Y
        ego_L_px   =  4.5 / _sc
        ego_W_px   =  2.0 / _sc
        draw_vehicle_rect(ax, ego_x_px, ego_y_px, ego_psi_px,
                          ego_L_px, ego_W_px, ego_col,
                          edgecolor="navy", lw=1.2, zorder=8)
        ax.text(ego_x_px + 1.0/_sc, ego_y_px + 1.5/_sc,
                f"{ego_s[0]:.1f} m/s",
                fontsize=5, color="black", style="oblique", zorder=9)

        # ── 6. Risk badge (DREAM only) ────────────────────────────────────
        if draw_risk:
            rc = ("red" if risk_at_ego > 1.5 else
                  "orange" if risk_at_ego > 0.5 else "lime")
            ax.text(0.985, 0.965, f"R={risk_at_ego:.2f}",
                    transform=ax.transAxes, ha="right", va="top",
                    color=rc, fontsize=7, fontweight="bold",
                    bbox=dict(boxstyle="round", facecolor="black", alpha=0.55))

        # ── 7. Viewport ego-centred in pixel space ────────────────────────
        ex_px   =  ego_g[0] / _sc
        ey_px   = -ego_g[1] / _sc
        vx_px   = VIEW_X / _sc
        vy_px   = VIEW_Y / _sc
        x0v, x1v = ex_px - vx_px, ex_px + vx_px
        ytop, ybot = ey_px - vy_px, ey_px + vy_px
        if bg_img is not None:
            _h, _w = bg_img.shape[:2]
            if x0v < 0:   x1v -= x0v;     x0v = 0
            if x1v > _w:  x0v -= x1v - _w; x1v = _w
            if ytop < 0:  ybot -= ytop;    ytop = 0
            if ybot > _h: ytop -= ybot - _h; ybot = _h
            x0v  = max(0, x0v);  x1v  = min(_w, x1v)
            ytop = max(0, ytop); ybot = min(_h, ybot)

        ax.set_xlim(x0v, x1v)
        ax.set_ylim(ybot, ytop)        # pixel Y: row 0 at top
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")
        ax.set_title(f"{title_str}  |  t={step*dt:.1f} s",
                     fontsize=9, fontweight="bold")

        if draw_risk and _cfg_X_vis is not None:
            import matplotlib.cm as _mcm
            _sm = _mcm.ScalarMappable(
                norm=plt.Normalize(vmin=0, vmax=vmax),
                cmap=plt.colormaps[RISK_CMAP])
            _sm.set_array([])
            cbar = fig.colorbar(_sm, ax=ax, fraction=0.018, pad=0.005)
            cbar.set_label(f"Risk  (vmax={vmax:.1f})", fontsize=6)
            cbar.ax.tick_params(labelsize=5)

    ax_top.tick_params(labelbottom=False)
    if SAVE_FRAMES:
        plt.savefig(
            os.path.join(save_dir, f"{step}.png"),
            dpi=FRAME_DPI,
            bbox_inches="tight",
        )


# ===========================================================================
# INITIALIZATION
# ===========================================================================

print("=" * 70)
print(f"DREAM Dataset Benchmark  |  {RECORDING_ID}  |  mode={INTEGRATION_MODE}")
print("=" * 70)

config_integration = get_preset(INTEGRATION_MODE)
config_integration.apply_mode()

# -- Load dataset -------------------------------------------------------------
if not os.path.isdir(DATASET_DIR):
    raise FileNotFoundError(
        f"Dataset directory not found: {DATASET_DIR}. "
        "Use --dataset-dir to point to rounD/inD data."
    )

rec = f"{int(RECORDING_ID):02d}"
tracks, tracks_meta, recording_meta = read_from_csv(
    os.path.join(DATASET_DIR, f"{rec}_tracks.csv"),
    os.path.join(DATASET_DIR, f"{rec}_tracksMeta.csv"),
    os.path.join(DATASET_DIR, f"{rec}_recordingMeta.csv"),
    include_px_coordinates=True)

frame_rate   = float(recording_meta["frameRate"])
frame_stride = max(1, round(dt / (1.0 / frame_rate)))
_ortho_px_m  = float(recording_meta["orthoPxToMeter"])
class_map    = {tm["trackId"]: tm["class"] for tm in tracks_meta}

print(f"Tracks: {len(tracks)}  frameRate={frame_rate}Hz  stride={frame_stride}")

# -- Background image ---------------------------------------------------------
_track_x_all = np.concatenate([t["xCenter"] for t in tracks])
_track_y_all = np.concatenate([t["yCenter"] for t in tracks])

bg_img, bg_extent = None, None
_bg_path = os.path.join(DATASET_DIR, f"{rec}_background.png")
if USE_BG and os.path.exists(_bg_path):
    _raw = cv2.imread(_bg_path)
    bg_img = cv2.cvtColor(_raw, cv2.COLOR_BGR2RGB)
    img_h, img_w = bg_img.shape[:2]
    bg_extent = [BG_OFFSET_X_M,
                 BG_OFFSET_X_M + img_w * _ortho_px_m,
                 BG_OFFSET_Y_M - img_h * _ortho_px_m,
                 BG_OFFSET_Y_M]

    # Auto-centre: shift bg_extent so the image centre aligns with the track
    # bounding-box centre.  Without this, for datasets where vehicle world
    # coordinates are far from 0 (e.g. rounD xCenter≈100–150m) the raw
    # bg_extent=[0,22m,...] never overlaps the ego-centred viewport.
    _img_cx  = 0.5 * (bg_extent[0] + bg_extent[1])
    _img_cy  = 0.5 * (bg_extent[2] + bg_extent[3])
    _trk_cx  = 0.5 * (float(np.min(_track_x_all)) + float(np.max(_track_x_all)))
    _trk_cy  = 0.5 * (float(np.min(_track_y_all)) + float(np.max(_track_y_all)))
    _dx, _dy = _trk_cx - _img_cx,  _trk_cy - _img_cy
    bg_extent = [bg_extent[0]+_dx, bg_extent[1]+_dx,
                 bg_extent[2]+_dy, bg_extent[3]+_dy]

    print(f"Background: {img_w}×{img_h}px  "
          f"world x=[{bg_extent[0]:.1f},{bg_extent[1]:.1f}]  "
          f"y=[{bg_extent[2]:.1f},{bg_extent[3]:.1f}]")

# -- Ego selection ------------------------------------------------------------
if EGO_TRACK_ID is None:
    EGO_TRACK_ID = select_ego_track(tracks, tracks_meta, class_map, EGO_MIN_FRAMES)
ego_meta  = tracks_meta[EGO_TRACK_ID]
ego_track = tracks[EGO_TRACK_ID]
ego_fi0   = ego_meta["initialFrame"]
print(f"Ego: trackId={EGO_TRACK_ID}  class={class_map[EGO_TRACK_ID]}  "
      f"frames={ego_meta['numFrames']}")

# -- Build reference paths from ego's recorded trajectory -------------------
# Use the ego track directly as the centre path so the planner follows the
# actual road geometry (critical for roundabouts, intersections, curved roads).
# fit_lane_paths() with K-means can produce paths that cut across curves.
paths = make_ego_track_paths(ego_track, N_LANES, LANE_WIDTH,
                             ego_lane_idx=EGO_LANE)
_paths = paths   # module-level for IDEAM adapters
path_center   = np.array(paths, dtype=object)
sample_center = np.array([p.samples for p in paths], dtype=object)
x_center      = [p.xc for p in paths]
y_center      = [p.yc for p in paths]

# -- Expand DRIFT grid --------------------------------------------------------
cfg.x_min = float(np.min(_track_x_all)) - SCENE_MARGIN
cfg.x_max = float(np.max(_track_x_all)) + SCENE_MARGIN
cfg.y_min = float(np.min(_track_y_all)) - SCENE_MARGIN
cfg.y_max = float(np.max(_track_y_all)) + SCENE_MARGIN
cfg.nx = int((cfg.x_max - cfg.x_min) / DRIFT_CELL_M) + 2
cfg.ny = int((cfg.y_max - cfg.y_min) / DRIFT_CELL_M) + 2
cfg.dx = (cfg.x_max - cfg.x_min) / (cfg.nx - 1)
cfg.dy = (cfg.y_max - cfg.y_min) / (cfg.ny - 1)
cfg.x  = np.linspace(cfg.x_min, cfg.x_max, cfg.nx)
cfg.y  = np.linspace(cfg.y_min, cfg.y_max, cfg.ny)
cfg.X, cfg.Y = np.meshgrid(cfg.x, cfg.y)
print(f"DRIFT grid: x=[{cfg.x_min:.0f},{cfg.x_max:.0f}]  "
      f"y=[{cfg.y_min:.0f},{cfg.y_max:.0f}]  {cfg.nx}×{cfg.ny}")

# -- Pixel-space scale (same presets as drift_dataset_visualization.py) -------
# These match the scale_down_factor used by run_track_visualization.py /
# TrackVisualizer so that bboxVis coordinates align with the background image.
_VIS_SCALE_PRESETS = {"exid": 6.0, "ind": 12.0, "round": 10.0}

_ds_key = os.path.basename(os.path.normpath(DATASET_DIR)).lower()
if _ds_key in _VIS_SCALE_PRESETS:
    _vis_scale_down = _VIS_SCALE_PRESETS[_ds_key]
    _vis_scale_src  = f"preset:{_ds_key}"
elif bg_img is not None and "xCenterVis" in tracks[0]:
    # Fallback: auto-fit so all pixel coords stay within the image
    try:
        _xv = np.concatenate([t["xCenterVis"].ravel() for t in tracks])
        _yv = np.concatenate([t["yCenterVis"].ravel() for t in tracks])
        _bh, _bw = bg_img.shape[:2]
        _sx = float(np.nanmax(np.abs(_xv))) / max(1.0, 0.98 * _bw)
        _sy = float(np.nanmax(np.abs(_yv))) / max(1.0, 0.98 * _bh)
        _vis_scale_down = max(1.0, _sx, _sy)
        _vis_scale_src  = f"auto-fit sx={_sx:.2f}, sy={_sy:.2f}"
    except Exception:
        _vis_scale_down, _vis_scale_src = 1.0, "auto-fallback"
else:
    _vis_scale_down, _vis_scale_src = 1.0, "default"
print(f"  vis_scale_down={_vis_scale_down:.3f}  ({_vis_scale_src})")

# DRIFT grid in pixel coordinates (used by contourf in draw_comparison_panel)
_cfg_X_vis = cfg.X / (_ortho_px_m * _vis_scale_down)
_cfg_Y_vis = -cfg.Y / (_ortho_px_m * _vis_scale_down)   # Y flipped for image space

# -- Road boundary from dataset laneWidth ------------------------------------
# tracks_import.py exposes `laneWidth` per frame (column from CSV).
# We use the ego's mean lane width to compute the half-width of the physical
# road (all lanes combined), which is the maximum allowed lateral Frenet offset
# before the planner goes off-road.
_ego_lane_width = 3.5   # default [m]
if "laneWidth" in ego_track:
    _w = np.asarray(ego_track["laneWidth"]).ravel()
    _w = _w[_w > 0.5]    # ignore zero/missing entries
    if len(_w) > 0:
        _ego_lane_width = float(np.nanmedian(_w))
# Road half-width = ego lane width * number of lanes / 2
# The planner may use lanes [0, 1, 2] offset by ±LANE_WIDTH from centre.
# The outermost allowable ey = half the total road width.
ROAD_HALF_WIDTH = (N_LANES / 2.0) * _ego_lane_width   # [m]
print(f"Road boundary: lane_width={_ego_lane_width:.2f}m  "
      f"road_half_width=±{ROAD_HALF_WIDTH:.2f}m  (N_LANES={N_LANES})")

# -- PRIDEAM + baseline IDEAM controllers ------------------------------------
controller = create_prideam_controller(
    paths={0: paths[0], 1: paths[1], 2: paths[2]},
    risk_weights={
        "mpc_cost":           config_integration.mpc_risk_weight,
        "cbf_modulation":     config_integration.cbf_alpha,
        "decision_threshold": config_integration.decision_risk_threshold,
    })
controller.get_path_curvature(path=paths[EGO_LANE])
drift = controller.drift

baseline_mpc = LMPC(**constraint_params())
utils_ideam  = LeaderFollower_Uitl(**util_params())
_bind_dataset_path_adapters_to_utils(utils_ideam, "IDEAM")
baseline_mpc.set_util(utils_ideam)
baseline_mpc.get_path_curvature(path=paths[EGO_LANE])

utils_dream = LeaderFollower_Uitl(**util_params())
_bind_dataset_path_adapters_to_utils(utils_dream, "DREAM")
controller.set_util(utils_dream)
mpc_ctrl = controller.mpc

# -- DREAM (ADA-DRIFT) controller — same MPC/CBF/decision logic, ADA source --
controller_ada = create_prideam_controller(
    paths={0: paths[0], 1: paths[1], 2: paths[2]},
    risk_weights={
        "mpc_cost":           config_integration.mpc_risk_weight,
        "cbf_modulation":     config_integration.cbf_alpha,
        "decision_threshold": config_integration.decision_risk_threshold,
    })
controller_ada.get_path_curvature(path=paths[EGO_LANE])
drift_ada = controller_ada.drift

utils_ada = LeaderFollower_Uitl(**util_params())
_bind_dataset_path_adapters_to_utils(utils_ada, "DREAM-ADA")
controller_ada.set_util(utils_ada)
mpc_ada = controller_ada.mpc

Params       = params()
dynamics     = Dynamic(**Params)
decision_p   = decision_params()
decision_maker = decision(**decision_p)
boundary     = 1.0

# -- Ego initial state from dataset ------------------------------------------
_fi0 = 0
_ex0 = float(ego_track["xCenter"][_fi0])
_ey0 = float(ego_track["yCenter"][_fi0])
_eh0 = heading_to_psi(ego_track["heading"][_fi0])
_ev0 = max(float(ego_track["lonVelocity"][_fi0]), 0.5)

_ego_path = paths[EGO_LANE]
try:
    _s0, _ey_f, _ep0 = find_frenet_coord(
        _ego_path, _ego_path.xc, _ego_path.yc, _ego_path.samples,
        [_ex0, _ey0, _eh0])
except Exception:
    _s0, _ey_f, _ep0 = 0.0, 0.0, 0.0

X0   = [_ev0, 0.0, 0.0, _s0, _ey_f, _ep0]
X0_g = [_ex0, _ey0, _eh0]
X0_ideam   = list(X0);  X0_g_ideam = list(X0_g)
print(f"Ego init: s={_s0:.1f}m  vx={_ev0:.1f}m/s  pos=({_ex0:.1f},{_ey0:.1f})")

# -- DRIFT warm-up ------------------------------------------------------------
print("DRIFT warm-up ...")
vl0, vc0, vr0, d0 = build_surrounding_arrays(
    ego_fi0, EGO_TRACK_ID, tracks, tracks_meta, class_map, paths, frame_rate)
_ei = drift_create_vehicle(vid=0, x=X0_g[0], y=X0_g[1],
                            vx=X0[0]*math.cos(X0_g[2]),
                            vy=X0[0]*math.sin(X0_g[2]), vclass="car")
_ei["heading"] = X0_g[2]
drift.warmup(d0 + [_ei], _ei, dt=dt, duration=WARMUP_S, substeps=3)
print("ADA-DRIFT warm-up ...")
drift_ada.warmup(d0 + [_ei], _ei, dt=dt, duration=WARMUP_S, substeps=3,
                 source_fn=compute_Q_ADA)
print()

# -- Misc state ---------------------------------------------------------------
oa, od         = 0.0, 0.0
oa_i, od_i     = 0.0, 0.0
oa_ada, od_ada = 0.0, 0.0
last_X = last_X_ideam = last_X_ada = None
path_changed = path_changed_i = path_changed_ada = EGO_LANE
last_ideam_hor = last_ada_hor = None

X0_ada   = list(X0);   X0_g_ada   = list(X0_g)

risk_list     = []
risk_list_ada = []
gt_vx_list    = []             # ground truth speed (from dataset)
dream_vx, dream_acc = [], []
ideam_vx,  ideam_acc  = [], []
ada_vx,    ada_acc    = [], []

METRICS_AUTOSAVE_EVERY = 5     # write metrics every N completed steps


def _save_metrics_outputs(interrupted=False, dpi=200):
    """
    Save current metrics figure/data to disk.

    This is used both for periodic autosave (crash-resilient) and final save.
    """
    if (len(gt_vx_list) + len(dream_vx) + len(ideam_vx) +
        len(dream_acc) + len(ideam_acc) + len(risk_list)) == 0:
        return

    _tgt    = np.arange(len(gt_vx_list)) * dt
    _td     = np.arange(len(dream_vx)) * dt
    _ti     = np.arange(len(ideam_vx)) * dt
    _tad    = np.arange(len(dream_acc)) * dt
    _tai    = np.arange(len(ideam_acc)) * dt
    _trisk  = np.arange(len(risk_list)) * dt
    _tada   = np.arange(len(ada_vx)) * dt
    _trisk_ada = np.arange(len(risk_list_ada)) * dt

    with plt.style.context(["science", "no-latex"]):
        fig_m, axes_m = plt.subplots(3, 1, figsize=(10, 9),
                                      constrained_layout=True, sharex=True)
        fig_m.suptitle(
            f"GT vs IDEAM vs DREAM-GVF vs DREAM-ADA  —  {rec} ego {EGO_TRACK_ID}",
            fontsize=11)

        # Speed
        if len(gt_vx_list) > 0:
            axes_m[0].plot(_tgt, gt_vx_list, color="C2", ls=":", lw=1.8, label="Ground Truth")
        if len(ideam_vx) > 0:
            axes_m[0].plot(_ti, ideam_vx, color="C1", ls="--", lw=1.4, label="IDEAM")
        if len(dream_vx) > 0:
            axes_m[0].plot(_td, dream_vx, color="C0", ls="-", lw=1.4, label="DREAM (GVF)")
        if len(ada_vx) > 0:
            axes_m[0].plot(_tada, ada_vx, color="#9C27B0", ls="-.", lw=1.4, label="DREAM (ADA)")
        axes_m[0].set_ylabel("$v_x$ [m/s]")
        axes_m[0].set_title("Speed")
        if len(axes_m[0].lines) > 0:
            axes_m[0].legend(fontsize=7)
        axes_m[0].grid(True, lw=0.4, alpha=0.4)

        # Acceleration
        if len(ideam_acc) > 0:
            axes_m[1].plot(_tai, ideam_acc, color="C1", ls="--", lw=1.4, label="IDEAM")
        if len(dream_acc) > 0:
            axes_m[1].plot(_tad, dream_acc, color="C0", ls="-", lw=1.4, label="DREAM (GVF)")
        if len(ada_acc) > 0:
            axes_m[1].plot(_tada, ada_acc, color="#9C27B0", ls="-.", lw=1.4, label="DREAM (ADA)")
        axes_m[1].axhline(0, color="k", lw=0.5, alpha=0.5)
        axes_m[1].set_ylabel("$a_x$ [m/s²]")
        axes_m[1].set_title("Acceleration")
        if len(axes_m[1].lines) > 0:
            axes_m[1].legend(fontsize=7)
        axes_m[1].grid(True, lw=0.4, alpha=0.4)

        # DRIFT risk — both sources
        if len(risk_list) > 0:
            axes_m[2].plot(_trisk, risk_list, color="C0", lw=1.4, label="GVF source")
            axes_m[2].fill_between(_trisk, risk_list, alpha=0.15, color="C0")
        if len(risk_list_ada) > 0:
            axes_m[2].plot(_trisk_ada, risk_list_ada,
                           color="#9C27B0", lw=1.4, ls="--", label="ADA source")
        axes_m[2].set_ylabel("Risk $R$")
        axes_m[2].set_title("DRIFT Risk at Ego")
        axes_m[2].set_xlabel("t [s]")
        if len(axes_m[2].lines) > 0:
            axes_m[2].legend(fontsize=7)
        axes_m[2].grid(True, lw=0.4, alpha=0.4)

        plt.savefig(os.path.join(save_dir, "metrics.png"), dpi=dpi, bbox_inches="tight")
        plt.close(fig_m)

    np.save(os.path.join(save_dir, "metrics.npy"), {
        "gt_vx": gt_vx_list,
        "dream_vx": dream_vx, "ideam_vx": ideam_vx, "ada_vx": ada_vx,
        "dream_acc": dream_acc, "ideam_acc": ideam_acc, "ada_acc": ada_acc,
        "risk": risk_list, "risk_ada": risk_list_ada,
        "rec": rec, "ego": EGO_TRACK_ID,
        "interrupted": bool(interrupted),
    })


bar = Bar(max=N_t - 1)
plt.figure(figsize=(25, 6.5))   # 4 panels side-by-side
max_frame  = max(tm["finalFrame"] for tm in tracks_meta)

# ===========================================================================
# MAIN SIMULATION LOOP
# ===========================================================================
print(f"Running {N_t} steps (dt={dt}s, {N_t*dt:.0f}s) ...")

# Graceful Ctrl+C: stop loop, then still save partial metrics/figures.
_stop_requested = False
_stop_reported = False
def _on_sigint(signum, frame):
    global _stop_requested
    _stop_requested = True
    print("\n[INTERRUPT] Stop requested. Finishing current step and saving outputs...")

_prev_sigint_handler = None
try:
    _prev_sigint_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, _on_sigint)
except Exception:
    _prev_sigint_handler = None

for i in range(N_t):
    if _stop_requested:
        if not _stop_reported:
            print(f"\n[INTERRUPT] Early stop at step {i}.")
            _stop_reported = True
        N_t = i
        break
    bar.next()
    frame_idx = ego_fi0 + i * frame_stride
    if frame_idx > max_frame:
        print(f"\n[WARN] End of recording at step {i}."); N_t = i; break

    # ── 0. Ground truth: read recorded ego position at this frame ────────
    fi_gt = frame_idx - ego_fi0
    if 0 <= fi_gt < len(ego_track["xCenter"]):
        gt_x  = float(ego_track["xCenter"][fi_gt])
        gt_y  = float(ego_track["yCenter"][fi_gt])
        gt_h  = heading_to_psi(ego_track["heading"][fi_gt])
        gt_vx = float(ego_track["lonVelocity"][fi_gt])
    else:
        gt_x, gt_y, gt_h, gt_vx = float(ego_track["xCenter"][-1]), \
                                   float(ego_track["yCenter"][-1]), \
                                   heading_to_psi(ego_track["heading"][-1]), 0.0
    X0_g_gt = [gt_x, gt_y, gt_h]
    gt_vx_list.append(gt_vx)

    # ── 1. Dataset surroundings ───────────────────────────────────────────
    _diag_step = (i == 0 or i % 50 == 0)
    _ego_xy_now = (X0_g[0], X0_g[1])
    _ego_s_now  = float(X0[3])   # current Frenet arc-length on reference path
    vl, vc, vr, drift_vehicles = build_surrounding_arrays(
        frame_idx, EGO_TRACK_ID, tracks, tracks_meta, class_map, paths, frame_rate,
        _diag=_diag_step, ego_xy=_ego_xy_now, ego_s=_ego_s_now)

    # ── 2. DRIFT step (GVF) ───────────────────────────────────────────────
    psi = X0_g[2]
    ego_dv = drift_create_vehicle(vid=0, x=X0_g[0], y=X0_g[1],
                                   vx=X0[0]*math.cos(psi)-X0[1]*math.sin(psi),
                                   vy=X0[0]*math.sin(psi)+X0[1]*math.cos(psi),
                                   vclass="car")
    ego_dv["heading"] = psi
    risk_field  = drift.step(drift_vehicles, ego_dv, dt=dt, substeps=3)
    risk_at_ego = float(drift.get_risk_cartesian(X0_g[0], X0_g[1]))
    risk_list.append(risk_at_ego)

    # ── 2b. DRIFT step (ADA) ──────────────────────────────────────────────
    psi_ada = X0_g_ada[2]
    ego_dv_ada = drift_create_vehicle(vid=0, x=X0_g_ada[0], y=X0_g_ada[1],
                                       vx=X0_ada[0]*math.cos(psi_ada)-X0_ada[1]*math.sin(psi_ada),
                                       vy=X0_ada[0]*math.sin(psi_ada)+X0_ada[1]*math.cos(psi_ada),
                                       vclass="car")
    ego_dv_ada["heading"] = psi_ada
    risk_field_ada  = drift_ada.step(drift_vehicles, ego_dv_ada, dt=dt, substeps=3,
                                     source_fn=compute_Q_ADA)
    risk_at_ego_ada = float(drift_ada.get_risk_cartesian(X0_g_ada[0], X0_g_ada[1]))
    risk_list_ada.append(risk_at_ego_ada)

    # ── 3. DREAM path decision ────────────────────────────────────────────
    path_now = _judge_lane(X0_g[:2])
    path_ego = paths[path_now]
    sgs = {0:"L1", 1:"C1", 2:"R1"}[path_now]

    if i == 0:
        ovx,ovy,owz,oS,oeyv,oepsi = clac_last_X(oa, od, mpc_ctrl.T, path_ego, dt, 6, X0, X0_g)
        last_X = [ovx,ovy,owz,oS,oeyv,oepsi]

    all_inf  = utils_dream.get_alllane_lf(path_ego, X0_g, path_now, vl, vc, vr)
    gd, eg   = utils_dream.formulate_gap_group(path_now, last_X, all_inf, vl, vc, vr)
    dg       = decision_maker.decision_making(gd, sgs)
    pd, pdi, CL, smp, xl, yl, X0 = _Decision_info(
        X0, X0_g, path_center, sample_center, x_center, y_center,
        boundary, dg, path_ego, path_now)
    CLa = utils_dream.inquire_C_state(CL, dg)
    if CLa == "Probe":
        pd, pdi, CLv = path_ego, path_now, "K"
        _, xc, yc, sc = _get_path_info(pdi)
        X0 = repropagate(pd, sc, xc, yc, X0_g, X0)
    else: CLv = CL
    if config_integration.enable_decision_veto and CL != "K":
        _, allow, _ = controller.evaluate_decision_risk(list(X0), path_now, pdi)
        if not allow:
            pd, pdi, CLv = path_ego, path_now, "K"
            _, xc, yc, sc = _get_path_info(pdi)
            X0 = repropagate(pd, sc, xc, yc, X0_g, X0)

    # ── Road-boundary guard (DREAM) ──────────────────────────────────────
    # If the proposed path would take ey beyond the physical road edge,
    # abort the lane change and keep current lane.
    # The target ey after LC ≈ (pdi - EGO_LANE) * LANE_WIDTH from path centre.
    _target_ey_dream = (pdi - EGO_LANE) * LANE_WIDTH
    if abs(_target_ey_dream) > ROAD_HALF_WIDTH + 0.5 and CLv != "K":
        pd, pdi, CLv = path_ego, path_now, "K"
        _, xc, yc, sc = _get_path_info(pdi)
        X0 = repropagate(pd, sc, xc, yc, X0_g, X0)
    if path_changed != pdi:
        controller.get_path_curvature(path=pd)
        oS, oeyv = _path_to_path_proj(oS, oeyv, path_changed, pdi)
        last_X = [ovx,ovy,owz,oS,oeyv,oepsi]
    path_changed = pdi

    # ── 4. DREAM MPC + propagate ──────────────────────────────────────────
    oa,od,ovx,ovy,owz,oS,oeyv,oepsi = controller.solve_with_risk(
        X0,oa,od,dt,None,None,CL,X0_g,pd,last_X,path_now,eg,path_ego,dg,
        vl,vc,vr,pdi,CLa,CLv)
    last_X = [ovx,ovy,owz,oS,oeyv,oepsi]
    X0, X0_g, _, _ = dynamics.propagate(X0, [oa[0],od[0]], dt, X0_g, pd, smp, xl, yl, boundary)
    dream_vx.append(float(X0[0]))
    dream_acc.append(float(oa[0]) if hasattr(oa,"__len__") else float(oa))

    # ── 5. IDEAM baseline (independent) ──────────────────────────────────
    Xi_p, Xgi_p, hor_i = list(X0_ideam), list(X0_g_ideam), last_ideam_hor
    try:
        pni = _judge_lane(X0_g_ideam[:2])
        pei = paths[pni]; sgi = {0:"L1",1:"C1",2:"R1"}[pni]
        if i == 0:
            _oi = clac_last_X(oa_i,od_i,baseline_mpc.T,pei,dt,6,X0_ideam,X0_g_ideam)
            last_X_ideam = list(_oi)
        ai = utils_ideam.get_alllane_lf(pei, X0_g_ideam, pni, vl, vc, vr)
        gdi,egi = utils_ideam.formulate_gap_group(pni, last_X_ideam, ai, vl, vc, vr)
        dgi = decision_maker.decision_making(gdi, sgi)
        pdi_i,pdii,Ci,si,xli,yli,X0_ideam = _Decision_info(
            X0_ideam,X0_g_ideam,path_center,sample_center,x_center,y_center,
            boundary,dgi,pei,pni)
        Cia = utils_ideam.inquire_C_state(Ci, dgi)
        if Cia == "Probe":
            pdi_i,pdii,Civ = pei,pni,"K"
            _,xci,yci,sci = _get_path_info(pdii)
            X0_ideam = repropagate(pdi_i,sci,xci,yci,X0_g_ideam,X0_ideam)
        else: Civ = Ci

        # ── Road-boundary guard (IDEAM) ───────────────────────────────────
        _target_ey_ideam = (pdii - EGO_LANE) * LANE_WIDTH
        if abs(_target_ey_ideam) > ROAD_HALF_WIDTH + 0.5 and Civ != "K":
            pdi_i,pdii,Civ = pei,pni,"K"
            _,xci,yci,sci = _get_path_info(pdii)
            X0_ideam = repropagate(pdi_i,sci,xci,yci,X0_g_ideam,X0_ideam)

        if path_changed_i != pdii and last_X_ideam:
            _oS2,_oey2 = _path_to_path_proj(last_X_ideam[3],last_X_ideam[4],path_changed_i,pdii)
            last_X_ideam[3],last_X_ideam[4] = _oS2,_oey2
        baseline_mpc.get_path_curvature(path=pdi_i); path_changed_i = pdii
        ri = baseline_mpc.iterative_linear_mpc_control(
            X0_ideam,oa_i,od_i,dt,None,None,Ci,X0_g_ideam,pdi_i,last_X_ideam,
            pni,egi,pei,dgi,vl,vc,vr,pdii,Cia,Civ)
        if ri:
            oa_i,od_i,*rest = ri; last_X_ideam = list(rest)
            X0_ideam,X0_g_ideam,_,_ = dynamics.propagate(
                list(X0_ideam),[oa_i[0],od_i[0]],dt,
                list(X0_g_ideam),pdi_i,si,xli,yli,boundary)
            Xi_p,Xgi_p = list(X0_ideam),list(X0_g_ideam)
            _h,_hg = list(X0_ideam),list(X0_g_ideam)
            _hor = [list(_hg)]
            for k in range(len(oa_i)-1):
                _h,_hg,_,_ = dynamics.propagate(_h,[oa_i[k+1],od_i[k+1]],dt,
                                                 _hg,pdi_i,si,xli,yli,boundary)
                _hor.append(list(_hg))
            hor_i = np.array(_hor); last_ideam_hor = hor_i
    except Exception as e:
        if i % 50 == 0: print(f"  [IDEAM] step {i}: {e}")
    ideam_vx.append(float(Xi_p[0]))
    ideam_acc.append(float(oa_i[0]) if hasattr(oa_i,"__len__") else float(oa_i))

    # ── 5b. DREAM-ADA planner (mirrors DREAM steps 3–4) ──────────────────
    Xa_p, Xga_p, hor_ada_cur = list(X0_ada), list(X0_g_ada), last_ada_hor
    try:
        pn_ada = _judge_lane(X0_g_ada[:2])
        pe_ada = paths[pn_ada]; sg_ada = {0:"L1",1:"C1",2:"R1"}[pn_ada]
        if i == 0:
            _oa_init = clac_last_X(oa_ada, od_ada, mpc_ada.T, pe_ada, dt, 6, X0_ada, X0_g_ada)
            last_X_ada = list(_oa_init)
        ai_ada = utils_ada.get_alllane_lf(pe_ada, X0_g_ada, pn_ada, vl, vc, vr)
        gd_ada, eg_ada = utils_ada.formulate_gap_group(pn_ada, last_X_ada, ai_ada, vl, vc, vr)
        dg_ada = decision_maker.decision_making(gd_ada, sg_ada)
        pd_ada, pdi_ada, CL_ada, smp_ada, xl_ada, yl_ada, X0_ada = _Decision_info(
            X0_ada, X0_g_ada, path_center, sample_center, x_center, y_center,
            boundary, dg_ada, pe_ada, pn_ada)
        CLa_ada = utils_ada.inquire_C_state(CL_ada, dg_ada)
        if CLa_ada == "Probe":
            pd_ada, pdi_ada, CLv_ada = pe_ada, pn_ada, "K"
            _, xc_ada, yc_ada, sc_ada = _get_path_info(pdi_ada)
            X0_ada = repropagate(pd_ada, sc_ada, xc_ada, yc_ada, X0_g_ada, X0_ada)
        else:
            CLv_ada = CL_ada
        if config_integration.enable_decision_veto and CL_ada != "K":
            _, allow_ada, _ = controller_ada.evaluate_decision_risk(
                list(X0_ada), pn_ada, pdi_ada)
            if not allow_ada:
                pd_ada, pdi_ada, CLv_ada = pe_ada, pn_ada, "K"
                _, xc_ada, yc_ada, sc_ada = _get_path_info(pdi_ada)
                X0_ada = repropagate(pd_ada, sc_ada, xc_ada, yc_ada, X0_g_ada, X0_ada)
        _target_ey_ada = (pdi_ada - EGO_LANE) * LANE_WIDTH
        if abs(_target_ey_ada) > ROAD_HALF_WIDTH + 0.5 and CLv_ada != "K":
            pd_ada, pdi_ada, CLv_ada = pe_ada, pn_ada, "K"
            _, xc_ada, yc_ada, sc_ada = _get_path_info(pdi_ada)
            X0_ada = repropagate(pd_ada, sc_ada, xc_ada, yc_ada, X0_g_ada, X0_ada)
        if path_changed_ada != pdi_ada and last_X_ada:
            _oSa, _oeya = _path_to_path_proj(last_X_ada[3], last_X_ada[4],
                                              path_changed_ada, pdi_ada)
            last_X_ada[3], last_X_ada[4] = _oSa, _oeya
        controller_ada.get_path_curvature(path=pd_ada)
        path_changed_ada = pdi_ada
        (oa_ada, od_ada,
         ovx_a, ovy_a, owz_a, oS_a, oeyv_a, oepsi_a) = controller_ada.solve_with_risk(
            X0_ada, oa_ada, od_ada, dt, None, None, CL_ada, X0_g_ada,
            pd_ada, last_X_ada, pn_ada, eg_ada, pe_ada, dg_ada,
            vl, vc, vr, pdi_ada, CLa_ada, CLv_ada)
        last_X_ada = [ovx_a, ovy_a, owz_a, oS_a, oeyv_a, oepsi_a]
        X0_ada, X0_g_ada, _, _ = dynamics.propagate(
            X0_ada, [oa_ada[0], od_ada[0]], dt, X0_g_ada,
            pd_ada, smp_ada, xl_ada, yl_ada, boundary)
        Xa_p, Xga_p = list(X0_ada), list(X0_g_ada)
        # ADA horizon rollout
        _Xva, _Xgva = list(X0_ada), list(X0_g_ada)
        _dha = [list(_Xgva)]
        for k in range(len(oa_ada)-1):
            _Xva, _Xgva, _, _ = dynamics.propagate(
                _Xva, [oa_ada[k+1], od_ada[k+1]], dt,
                _Xgva, pd_ada, smp_ada, xl_ada, yl_ada, boundary)
            _dha.append(list(_Xgva))
        hor_ada_cur = np.array(_dha); last_ada_hor = hor_ada_cur
    except Exception as e:
        if i % 50 == 0: print(f"  [ADA] step {i}: {e}")
    ada_vx.append(float(Xa_p[0]))
    ada_acc.append(float(oa_ada[0]) if hasattr(oa_ada, "__len__") else float(oa_ada))

    # Periodic autosave so Ctrl+C aborts inside native solvers still leave
    # a recent metrics figure/data on disk.
    if ((i + 1) % METRICS_AUTOSAVE_EVERY) == 0:
        _save_metrics_outputs(interrupted=False, dpi=150)

    # ── 6. DREAM (GVF) horizon rollout ────────────────────────────────────
    _Xv,_Xgv = list(X0),list(X0_g)
    _dh = [list(_Xgv)]
    for k in range(len(oa)-1):
        _Xv,_Xgv,_,_ = dynamics.propagate(_Xv,[oa[k+1],od[k+1]],dt,
                                            _Xgv,pd,smp,xl,yl,boundary)
        _dh.append(list(_Xgv))
    _dh = np.array(_dh)

    # ── 7. Visualize — four-panel: GT | IDEAM | DREAM-GVF | DREAM-ADA ────
    draw_four_panel(
        i, frame_idx,
        X0_g_gt, gt_vx_list[-1],
        Xgi_p, Xi_p,
        X0_g, X0,
        Xga_p, Xa_p,
        risk_field,     risk_at_ego,
        risk_field_ada, risk_at_ego_ada,
        tracks, tracks_meta, class_map,
        bg_img, bg_extent,
        horizon_ideam=hor_i, horizon_dream=_dh, horizon_ada=hor_ada_cur)

bar.finish()
print()
print("Simulation interrupted by user." if _stop_requested else "Simulation complete.")
if len(dream_vx) > 0:
    print(f"  DREAM (GVF) final vx={dream_vx[-1]:.2f}m/s")
if len(ada_vx) > 0:
    print(f"  DREAM (ADA) final vx={ada_vx[-1]:.2f}m/s")
if len(ideam_vx) > 0:
    print(f"  IDEAM       final vx={ideam_vx[-1]:.2f}m/s")

if _prev_sigint_handler is not None:
    try:
        signal.signal(signal.SIGINT, _prev_sigint_handler)
    except Exception:
        pass

# ===========================================================================
# METRICS PLOT
# ===========================================================================
_save_metrics_outputs(interrupted=bool(_stop_requested), dpi=200)
if SAVE_FRAMES:
    print(f"\nFrames  → {save_dir}/")
else:
    print("\nFrame saving disabled (--save-frames false).")
print(f"Metrics → {save_dir}/metrics.png")
