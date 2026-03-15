"""
Visualisation of the Artificial Potential Field (APF) risk model.

Produces four figures:
  1) 3-D surface of the total potential field
  2) Filled-contour risk map with vehicle patches and lane markings
  3) Per-obstacle repulsive field decomposition
  4) Total field with virtual-force quiver overlay
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
from APF_config import APFConfig as Cfg
from APF import make_ego, make_obstacle, make_goal, construct_APF


# ── drawing helpers (shared with KWF_vis) ─────────────────────────────────
def _draw_vehicle(ax, veh, facecolor='grey', edgecolor='black', zorder=5):
    w, h = veh['length'], veh['width']
    rect = mpatches.FancyBboxPatch(
        (-w / 2, -h / 2), w, h,
        boxstyle="round,pad=0.1",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=1.0, zorder=zorder)
    t = (mtransforms.Affine2D()
         .translate(veh['x'], veh['y'])
         + ax.transData)
    rect.set_transform(t)
    ax.add_patch(rect)


def _draw_lane_markings(ax, x_range, lane_centers, lane_width):
    y_bounds = sorted({c + s * lane_width / 2
                       for c in lane_centers for s in (-1, 1)})
    for i, yb in enumerate(y_bounds):
        style = '-w' if (i == 0 or i == len(y_bounds) - 1) else '--w'
        ax.plot(x_range, [yb, yb], style, linewidth=1.5, alpha=0.7)


def _draw_goal_marker(ax, goal, color='#f1c40f'):
    ax.plot(goal['x'], goal['y'], '*', color=color, markersize=18,
            markeredgecolor='black', markeredgewidth=0.8, zorder=7)
    ax.text(goal['x'] + 2, goal['y'] + 1, 'Goal',
            color=color, fontsize=10, fontweight='bold',
            bbox=dict(facecolor='black', alpha=0.4, boxstyle='round,pad=0.2'))


_OBS_COLORS = ['#3498db', '#e67e22', '#2ecc71', '#9b59b6', '#e74c3c']
_EGO_COLOR = '#e74c3c'


# ── main demo ─────────────────────────────────────────────────────────────
def APF_static_demo():
    """
    Highway scenario: ego vehicle navigates toward a goal while avoiding
    surrounding vehicles modelled as repulsive obstacles.
    """
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 12

    # ── Ego & goal ────────────────────────────────────────────────────────
    ego = make_ego(x=0, y=0, label='Ego')
    goal = make_goal(x=130, y=0, label='Goal')

    # ── Surrounding vehicles (obstacles) ──────────────────────────────────
    obs1 = make_obstacle(x=30, y=0, length=16.0, width=2.5, label='Obs 1 (Truck)')
    obs2 = make_obstacle(x=70, y=-4, length=13.0, width=2.5, label='Obs 2 (Truck)')
    obs3 = make_obstacle(x=100, y=4, length=4.8, width=2.2, label='Obs 3 (Car)')

    obstacles = [obs1, obs2, obs3]

    # ── Construct APF ─────────────────────────────────────────────────────
    Xg, Yg = Cfg.X_mesh, Cfg.Y_mesh
    U_total, U_att, U_rep_list, Fx, Fy = construct_APF(obstacles, goal, Xg, Yg)

    # Clip for cleaner visualisation (repulsive peaks are very sharp)
    U_clip = np.clip(U_total, 0, np.percentile(U_total, 99.5))

    # ══════════════════════════════════════════════════════════════════════
    #  Figure 1 – 3-D surface
    # ══════════════════════════════════════════════════════════════════════
    fig1 = plt.figure(figsize=Cfg.figsize_3d)
    ax1 = fig1.add_subplot(111, projection='3d')
    step = max(1, int(1.0 / Cfg.resolution))
    ax1.plot_surface(Xg[::step, ::step], Yg[::step, ::step],
                     U_clip[::step, ::step],
                     cmap=Cfg.cmap_name, edgecolor='none', alpha=0.9)
    ax1.set_xlabel('$x$ (m)')
    ax1.set_ylabel('$y$ (m)')
    ax1.set_zlabel('Potential')
    ax1.set_title('Artificial Potential Field – 3-D Surface')
    ax1.set_xlim(Cfg.x_min, Cfg.x_max)
    ax1.set_ylim(Cfg.y_min, Cfg.y_max)
    plt.tight_layout()

    # ══════════════════════════════════════════════════════════════════════
    #  Figure 2 – filled contour with vehicles
    # ══════════════════════════════════════════════════════════════════════
    fig2, ax2 = plt.subplots(figsize=Cfg.figsize_contour)

    levels = np.linspace(0, np.percentile(U_total, 98), 40)
    cf = ax2.contourf(Xg, Yg, U_clip, levels=levels, cmap=Cfg.cmap_name, extend='max')
    ax2.contour(Xg, Yg, U_clip, levels=levels[::5], colors='k',
                linewidths=0.3, alpha=0.4)

    _draw_lane_markings(ax2, [Cfg.x_min, Cfg.x_max], [-4, 0, 4], Cfg.lane_width)
    _draw_vehicle(ax2, ego, facecolor=_EGO_COLOR)
    ax2.text(ego['x'] - 2, ego['y'] + 2, 'Ego', color='white', fontsize=10,
             fontweight='bold',
             bbox=dict(facecolor='black', alpha=0.4, boxstyle='round,pad=0.2'))
    for k, obs in enumerate(obstacles):
        _draw_vehicle(ax2, obs, facecolor=_OBS_COLORS[k % len(_OBS_COLORS)])
    _draw_goal_marker(ax2, goal)

    cbar = plt.colorbar(cf, ax=ax2, pad=0.02, aspect=30)
    cbar.set_label('Potential (risk)', fontsize=11)

    ax2.set_xlim(Cfg.x_min, Cfg.x_max)
    ax2.set_ylim(Cfg.y_min, Cfg.y_max)
    ax2.set_xlabel('$x$ (m)')
    ax2.set_ylabel('$y$ (m)')
    ax2.set_title('Artificial Potential Field – Risk Map')
    ax2.set_aspect('auto')
    plt.tight_layout()

    # ══════════════════════════════════════════════════════════════════════
    #  Figure 3 – per-obstacle repulsive decomposition
    # ══════════════════════════════════════════════════════════════════════
    n_obs = len(obstacles)
    fig3, axes = plt.subplots(1, n_obs, figsize=(5 * n_obs, 3.5), sharey=True)
    if n_obs == 1:
        axes = [axes]
    for k, (ax, U_k) in enumerate(zip(axes, U_rep_list)):
        U_k_clip = np.clip(U_k, 0, np.percentile(U_k[U_k > 0], 99) if np.any(U_k > 0) else 1)
        cf_k = ax.contourf(Xg, Yg, U_k_clip, levels=30, cmap=Cfg.cmap_name)
        _draw_vehicle(ax, obstacles[k],
                      facecolor=_OBS_COLORS[k % len(_OBS_COLORS)])
        _draw_lane_markings(ax, [Cfg.x_min, Cfg.x_max], [-4, 0, 4], Cfg.lane_width)
        ax.set_xlim(Cfg.x_min, Cfg.x_max)
        ax.set_ylim(Cfg.y_min, Cfg.y_max)
        ax.set_xlabel('$x$ (m)')
        ax.set_title(f'Repulsive field – {obstacles[k]["label"]}')
        plt.colorbar(cf_k, ax=ax, pad=0.02)
    axes[0].set_ylabel('$y$ (m)')
    plt.tight_layout()

    # ══════════════════════════════════════════════════════════════════════
    #  Figure 4 – quiver (virtual force = −∇U)
    # ══════════════════════════════════════════════════════════════════════
    fig4, ax4 = plt.subplots(figsize=Cfg.figsize_contour)

    ax4.pcolormesh(Xg, Yg, U_clip, cmap='jet', shading='gouraud')

    F_mag = np.sqrt(Fx ** 2 + Fy ** 2) + 1e-12
    qs = max(1, int(3.0 / Cfg.resolution))
    ax4.quiver(Xg[::qs, ::qs], Yg[::qs, ::qs],
               Fx[::qs, ::qs] / F_mag[::qs, ::qs],
               Fy[::qs, ::qs] / F_mag[::qs, ::qs],
               color='white', alpha=0.6, scale=30)

    for k, obs in enumerate(obstacles):
        _draw_vehicle(ax4, obs, facecolor=_OBS_COLORS[k % len(_OBS_COLORS)])
    _draw_vehicle(ax4, ego, facecolor=_EGO_COLOR)
    _draw_goal_marker(ax4, goal)
    _draw_lane_markings(ax4, [Cfg.x_min, Cfg.x_max], [-4, 0, 4], Cfg.lane_width)

    ax4.set_xlim(Cfg.x_min, Cfg.x_max)
    ax4.set_ylim(Cfg.y_min, Cfg.y_max)
    ax4.set_xlabel('$x$ (m)')
    ax4.set_ylabel('$y$ (m)')
    ax4.set_title('Artificial Potential Field – Virtual Force Quiver')
    plt.tight_layout()

    return fig1, fig2, fig3, fig4


# ── entry point ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    figs = APF_static_demo()
    for i, fig in enumerate(figs, 1):
        fig.savefig(f'APF_fig{i}.png', dpi=200, bbox_inches='tight')
        print(f'Saved APF_fig{i}.png')
    plt.show()
