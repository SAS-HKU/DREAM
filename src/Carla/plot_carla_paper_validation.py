"""Create publication figures for the matched CARLA occlusion validation.

The script consumes only the aggregate analysis tables, the retained per-run
traces, and one outcome-blind representative visualization.  It uses
SciencePlots and writes both vector PDF and high-resolution PNG files.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401  # registers the SciencePlots styles


PALETTE = {"DREAM": "#2878B5", "IDEAM": "#F5A623"}


def _rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(row: Mapping[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in (None, "") else float("nan")


def _effect(effects: Iterable[Mapping[str, str]], metric: str) -> Mapping[str, str]:
    return next(row for row in effects if row["metric"] == metric)


def _run(runs: Iterable[Mapping[str, str]], seed: int, controller: str,
         condition: str) -> Mapping[str, str]:
    return next(
        row
        for row in runs
        if int(row["scene_seed"]) == int(seed)
        and row["controller"] == controller
        and row["condition"] == condition
    )


def _save(fig: plt.Figure, output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=400, bbox_inches="tight")
    plt.close(fig)


def validation_figure(analysis_dir: Path, visual_run_dir: Path,
                      output_dir: Path, representative_seed: int) -> None:
    runs = _rows(analysis_dir / "run_metrics.csv")
    effects = _rows(analysis_dir / "paired_effect_statistics.csv")
    frame_paths = [
        visual_run_dir / "frames" / "frame_00000.png",
        visual_run_dir / "frames" / "frame_00007.png",
        visual_run_dir / "frames" / "frame_00031.png",
    ]
    for path in frame_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    with plt.style.context(["science", "no-latex"]):
        plt.rcParams.update({
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.titlesize": 12,
        })
        fig = plt.figure(figsize=(13.2, 8.2), constrained_layout=True)
        grid = fig.add_gridspec(2, 3, height_ratios=(1.08, 1.0))
        titles = (
            "A1  Occluded approach ($t=0.25$ s)",
            "A2  Sensor reveal ($t=0.95$ s)",
            "A3  Converging-merge response ($t=3.35$ s)",
        )
        for index, (path, title) in enumerate(zip(frame_paths, titles)):
            ax = fig.add_subplot(grid[0, index])
            ax.imshow(plt.imread(path))
            ax.set_title(title, loc="left")
            ax.set_axis_off()

        safety_ax = fig.add_subplot(grid[1, 0])
        true_runs = [row for row in runs if row["condition"] == "true_threat"]
        seeds = sorted({int(row["scene_seed"]) for row in true_runs})
        for seed in seeds:
            ideam = _run(true_runs, seed, "IDEAM", "true_threat")
            dream = _run(true_runs, seed, "DREAM", "true_threat")
            values = [
                _float(ideam, "minimum_hidden_oriented_box_clearance_m"),
                _float(dream, "minimum_hidden_oriented_box_clearance_m"),
            ]
            safety_ax.plot([0, 1], values, color="#B7BDC6", lw=1.0, zorder=1)
            safety_ax.scatter(0, values[0], s=34, color=PALETTE["IDEAM"],
                              edgecolor="#6D4A00", linewidth=0.5, zorder=2)
            safety_ax.scatter(1, values[1], s=34, color=PALETTE["DREAM"],
                              edgecolor="#174A73", linewidth=0.5, zorder=2)
        safety_ax.axhline(0.0, color="#555B66", ls="--", lw=0.9)
        safety_ax.set_xticks([0, 1], ["IDEAM", "DREAM"])
        safety_ax.set_ylabel("Minimum hidden-vehicle clearance (m)")
        safety_ax.set_title("B  Paired true-threat safety margin", loc="left")
        dream_collision = sum(
            int(float(row["collision_incidence"]))
            for row in true_runs if row["controller"] == "DREAM"
        )
        ideam_collision = sum(
            int(float(row["collision_incidence"]))
            for row in true_runs if row["controller"] == "IDEAM"
        )
        dream_near = sum(
            int(float(row["near_collision_incidence"]))
            for row in true_runs if row["controller"] == "DREAM"
        )
        ideam_near = sum(
            int(float(row["near_collision_incidence"]))
            for row in true_runs if row["controller"] == "IDEAM"
        )
        safety_ax.text(
            0.03, 0.98,
            "collision: {} vs {}\nnear collision: {} vs {}".format(
                ideam_collision, dream_collision, ideam_near, dream_near
            ),
            transform=safety_ax.transAxes, ha="left", va="top",
            bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "none"},
        )

        speed_ax = fig.add_subplot(grid[1, 1:])
        for controller in ("DREAM", "IDEAM"):
            run = _run(runs, representative_seed, controller, "empty_shadow")
            trace = _rows(Path(run["run_directory"]) / "tick_trace.csv")
            time_s = np.asarray([_float(row, "time_s") for row in trace])
            ego_speed = np.asarray([_float(row, "ego_speed_mps") for row in trace])
            follower_speed = np.asarray([
                _float(row, "follower_1_speed_mps") for row in trace
            ])
            speed_ax.plot(
                time_s, ego_speed, color=PALETTE[controller], lw=1.8,
                label="{} ego".format(controller),
            )
            speed_ax.plot(
                time_s, follower_speed, color=PALETTE[controller], lw=1.4,
                ls="--", label="{} nearest follower".format(controller),
            )
        speed_ax.set_xlabel("Time (s)")
        speed_ax.set_ylabel("Speed (m s$^{-1}$)")
        speed_ax.set_title(
            "C  Empty-shadow ego and trailing-vehicle response", loc="left"
        )
        speed_ax.legend(ncol=2, loc="lower left")
        ct = _effect(effects, "empty_shadow_ct_v_mps")
        follower = _effect(
            effects,
            "empty_shadow__dream_minus_ideam__maximum_follower_speed_loss_mps",
        )
        speed_ax.text(
            0.98, 0.97,
            "$CT_v={:.2f}$ m s$^{{-1}}$\n"
            "$\\Delta$ follower speed loss $={:.2g}$ m s$^{{-1}}$".format(
                float(ct["mean"]), float(follower["mean"])
            ),
            transform=speed_ax.transAxes, ha="right", va="top",
            bbox={"facecolor": "white", "alpha": 0.90, "edgecolor": "#C9CDD3"},
        )
        fig.suptitle("Closed-loop CARLA validation of the occluded converging merge")
        _save(fig, output_dir / "fig_carla_closed_loop_validation")


def runtime_figure(analysis_dir: Path, output_dir: Path) -> None:
    runs = _rows(analysis_dir / "run_metrics.csv")
    group_order = [
        ("DREAM", "empty_shadow", "D--E"),
        ("IDEAM", "empty_shadow", "I--E"),
        ("DREAM", "true_threat", "D--T"),
        ("IDEAM", "true_threat", "I--T"),
    ]
    with plt.style.context(["science", "no-latex"]):
        plt.rcParams.update({"font.size": 9, "axes.titlesize": 10})
        fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8), constrained_layout=True)

        planner_ax = axes[0]
        for index, (controller, condition, label) in enumerate(group_order):
            subset = [
                row for row in runs
                if row["controller"] == controller and row["condition"] == condition
            ]
            values = 1000.0 * np.asarray([
                _float(row, "planner_mean_total_s") for row in subset
            ])
            planner_ax.scatter(
                np.full(values.shape, index), values, s=28,
                color=PALETTE[controller], alpha=0.9,
                edgecolor="#555555", linewidth=0.4,
            )
            planner_ax.plot(
                [index - 0.18, index + 0.18], [np.mean(values), np.mean(values)],
                color="black", lw=1.5,
            )
        planner_ax.axhline(100.0, color="#B13A3A", ls="--", lw=1.0,
                           label="100 ms nominal planning interval")
        planner_ax.set_xticks(range(len(group_order)), [item[2] for item in group_order])
        planner_ax.set_ylabel("Mean high-level planning time (ms)")
        planner_ax.set_title("A  High-level planning remains asynchronous", loc="left")
        planner_ax.legend(loc="upper left")

        loop_ax = axes[1]
        categories = []
        p95_values = []
        max_values = []
        colours = []
        for controller, condition, label in group_order:
            subset = [
                row for row in runs
                if row["controller"] == controller and row["condition"] == condition
            ]
            categories.extend(["{}\nlow".format(label), "{}\nphysics".format(label)])
            p95_values.extend([
                1000.0 * np.mean([_float(row, "low_level_p95_time_s") for row in subset]),
                1000.0 * np.mean([
                    _float(row, "physics_control_loop_p95_cycle_time_s") for row in subset
                ]),
            ])
            max_values.extend([
                1000.0 * max(_float(row, "low_level_maximum_time_s") for row in subset),
                1000.0 * max(
                    _float(row, "physics_control_loop_maximum_cycle_time_s")
                    for row in subset
                ),
            ])
            colours.extend([PALETTE[controller], PALETTE[controller]])
        positions = np.arange(len(categories))
        loop_ax.scatter(positions, p95_values, marker="o", s=34, color=colours,
                        label="mean P95")
        loop_ax.scatter(positions, max_values, marker="^", s=40, facecolors="none",
                        edgecolors=colours, linewidth=1.0, label="observed maximum")
        loop_ax.axhline(50.0, color="#7A4EAB", ls="--", lw=1.0,
                        label="50 ms physics deadline")
        loop_ax.axhline(100.0, color="#B13A3A", ls=":", lw=1.0,
                        label="100 ms low-level deadline")
        loop_ax.set_yscale("log")
        loop_ax.set_ylim(0.8, 140.0)
        loop_ax.set_xticks(positions, categories)
        loop_ax.set_ylabel("Execution time (ms, log scale)")
        loop_ax.set_title("B  Executed control loops meet their deadlines", loc="left")
        loop_ax.legend(loc="upper left", ncol=2)
        fig.suptitle("Measured timing of the asynchronous CARLA implementation")
        _save(fig, output_dir / "fig_carla_async_runtime")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--visual-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--representative-seed", type=int, default=1001)
    args = parser.parse_args()
    validation_figure(
        args.analysis_dir.resolve(), args.visual_run_dir.resolve(),
        args.output_dir.resolve(), args.representative_seed,
    )
    runtime_figure(args.analysis_dir.resolve(), args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
