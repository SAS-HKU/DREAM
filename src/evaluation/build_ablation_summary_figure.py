"""Build the paired occlusion-ablation summary figure.

The three panels deliberately have different evidential roles:

* physical outcomes under a true occluded threat;
* fixed-playback field-component distributions; and
* the progress cost of caution under an empty shadow.

The script reads only the active, deduplicated benchmark logs and exports the
plotted values as a long-form CSV. Plot styling is provided by SciencePlots.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")

SCRIPT_PATH = Path(__file__).resolve()


def _default_project_root() -> Path:
    """Return the repository root for local and ``src/evaluation`` copies."""

    if SCRIPT_PATH.parent.name == "evaluation" and SCRIPT_PATH.parent.parent.name == "src":
        return SCRIPT_PATH.parents[2]
    return SCRIPT_PATH.parents[1]


ROOT = _default_project_root()
LOCAL_SCIENCEPLOTS = ROOT / "SciencePlots" / "src"
if LOCAL_SCIENCEPLOTS.exists():
    sys.path.insert(0, str(LOCAL_SCIENCEPLOTS))

import scienceplots  # noqa: F401,E402  (registers Matplotlib styles)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


NAVY = "#1F4E79"
BLUE = "#5B8DB8"
LIGHT_BLUE = "#A9C5DC"
ORANGE = "#D97706"
LIGHT_ORANGE = "#F4C37D"
GREY = "#8A8F98"
LIGHT_GREY = "#D9DDE3"
TEXT = "#20252B"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def prepare_channel_panel(
    episode_paths: list[Path],
) -> tuple[list[str], list[str], np.ndarray, list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    for path in episode_paths:
        records.extend(read_jsonl(path))

    variants = [
        "coupling_full",
        "coupling_no_veto",
        "coupling_no_mpc_cost",
        "coupling_no_cbf",
        "coupling_none",
    ]
    labels = [
        "Full coupling",
        "No decision veto",
        "No MPC risk cost",
        "No CBF modulation",
        "No coupling channels",
    ]
    metrics = [
        ("Collision", "collision_incident"),
        ("Near\ncollision", "near_collision_incident"),
        ("TTC-\ncritical", "ttc_critical_incident"),
    ]

    counts = np.zeros((len(variants), len(metrics)), dtype=int)
    source_rows: list[dict[str, Any]] = []
    for i, variant in enumerate(variants):
        subset = [record for record in records if record["variant"] == variant]
        if len(subset) != 15:
            raise RuntimeError(f"Expected 15 active records for {variant}, found {len(subset)}")
        for j, (metric_label, metric_key) in enumerate(metrics):
            value = int(sum(bool(record["safety"][metric_key]) for record in subset))
            counts[i, j] = value
            source_rows.append(
                {
                    "panel": "A",
                    "suite": "true_occluded_threat",
                    "variant": variant,
                    "variant_label": labels[i],
                    "metric": metric_key,
                    "metric_label": metric_label,
                    "n_events": value,
                    "n_total": len(subset),
                }
            )
    return labels, [label for label, _ in metrics], counts, source_rows


def prepare_field_panel(
    episode_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return per-construction fixed-playback field measurements."""

    records = read_jsonl(episode_path)
    variant_order = [
        "field_full",
        "field_no_advection",
        "field_no_occ_source",
        "field_no_occ_diffusion",
        "field_static_trailer_occ",
    ]
    labels = {
        "field_full": "Full field",
        "field_no_advection": "No advection",
        "field_no_occ_source": r"No $Q_{\mathrm{occ}}$",
        "field_no_occ_diffusion": "No diffusion",
        "field_static_trailer_occ": "Static trailer source",
    }

    rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    for variant in variant_order:
        subset = [record for record in records if record["variant"] == variant]
        if len(subset) != 15:
            raise RuntimeError(
                f"Expected 15 active field records for {variant}, found {len(subset)}"
            )

        measurements: list[dict[str, Any]] = []
        for record in subset:
            playback = record["field"]["reference_playback"]
            if not playback.get("available", False):
                raise RuntimeError(
                    f"Fixed reference playback unavailable for {record['scenario_id']}"
                )
            value = float(playback["risk_mass_target_maneuver_tube"])
            measurement = {
                "scenario_id": record["scenario_id"],
                "pair_id": record["pair_id"],
                "severity": record["scenario_design"]["severity"],
                "value": value,
            }
            measurements.append(measurement)
            source_rows.append(
                {
                    "panel": "B",
                    "suite": "true_occluded_threat_fixed_reference_playback",
                    "scenario_id": record["scenario_id"],
                    "pair_id": record["pair_id"],
                    "severity": measurement["severity"],
                    "variant": variant,
                    "variant_label": labels[variant].replace("$", ""),
                    "metric": "risk_mass_target_maneuver_tube",
                    "value": value,
                }
            )

        rows.append(
            {
                "variant": variant,
                "label": labels[variant],
                "measurements": measurements,
                "median": float(
                    np.median([measurement["value"] for measurement in measurements])
                ),
            }
        )
    return rows, source_rows


def prepare_empty_panel(
    episode_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records = read_jsonl(episode_path)
    by_pair: dict[str, dict[str, dict[str, Any]]] = {}
    for record in records:
        by_pair.setdefault(record["pair_id"], {})[record["variant"]] = record

    severity_order = {"critical": 0, "moderate": 1, "mild": 2}
    pairs: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    for pair_id, variants in by_pair.items():
        if {"coupling_full", "coupling_no_veto"} - set(variants):
            raise RuntimeError(f"Incomplete empty-shadow pair: {pair_id}")
        full = variants["coupling_full"]
        no_veto = variants["coupling_no_veto"]
        severity = full["scenario_design"]["severity"]
        pair = {
            "pair_id": pair_id,
            "severity": severity,
            "scenario_id": full["scenario_id"],
            "full": float(full["tradeoff"]["progress_m"]),
            "no_veto": float(no_veto["tradeoff"]["progress_m"]),
        }
        pairs.append(pair)
        for variant, record in variants.items():
            source_rows.append(
                {
                    "panel": "C",
                    "suite": "empty_shadow",
                    "pair_id": pair_id,
                    "scenario_id": record["scenario_id"],
                    "severity": severity,
                    "variant": variant,
                    "metric": "progress_m",
                    "value": record["tradeoff"]["progress_m"],
                    "false_veto_incident": record["tradeoff"][
                        "false_veto_incident"
                    ],
                    "collision_incident": record["safety"]["collision_incident"],
                    "near_collision_incident": record["safety"][
                        "near_collision_incident"
                    ],
                }
            )
    pairs.sort(key=lambda row: (severity_order[row["severity"]], row["scenario_id"]))
    if len(pairs) != 15:
        raise RuntimeError(f"Expected 15 empty-shadow pairs, found {len(pairs)}")
    return pairs, source_rows


def export_source_data(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_figure(
    *,
    channel_episode_paths: list[Path],
    field_episode_path: Path,
    empty_episode_path: Path,
    figure_root: Path,
) -> None:
    figure_root.mkdir(parents=True, exist_ok=True)

    channel_labels, metric_labels, counts, source_a = prepare_channel_panel(
        channel_episode_paths
    )
    field_rows, source_b = prepare_field_panel(field_episode_path)
    empty_pairs, source_c = prepare_empty_panel(empty_episode_path)
    export_source_data(
        source_a + source_b + source_c,
        figure_root / "fig_ablation_summary_source_data.csv",
    )

    plt.style.use(["science", "no-latex", "grid"])
    plt.rcParams.update(
        {
            "font.size": 7.5,
            "axes.titlesize": 8.5,
            "axes.labelsize": 7.7,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "axes.linewidth": 0.7,
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "axes.edgecolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    figure = plt.figure(figsize=(7.5, 3.0), constrained_layout=False)
    grid = figure.add_gridspec(1, 3, width_ratios=(1.08, 1.34, 0.86))
    figure.subplots_adjust(
        left=0.135,
        right=0.99,
        top=0.89,
        bottom=0.245,
        wspace=0.56,
    )

    # Panel A: exact paired-benchmark event incidences.
    ax_a = figure.add_subplot(grid[0, 0])
    incidence_cmap = LinearSegmentedColormap.from_list(
        "incidence", ["#FAFBFC", LIGHT_ORANGE, ORANGE, "#8E3B00"]
    )
    ax_a.imshow(counts, cmap=incidence_cmap, vmin=0, vmax=15, aspect="auto")
    for row_index in range(counts.shape[0]):
        for column_index in range(counts.shape[1]):
            value = counts[row_index, column_index]
            ax_a.text(
                column_index,
                row_index,
                f"{value}/15",
                ha="center",
                va="center",
                color="white" if value >= 10 else TEXT,
                fontweight="bold" if value in {0, 15} else "normal",
                fontsize=7.2,
            )
    ax_a.set_xticks(range(len(metric_labels)), metric_labels)
    ax_a.set_yticks(range(len(channel_labels)), channel_labels)
    ax_a.tick_params(length=0, pad=4)
    ax_a.set_title(
        "A   True-threat event incidence",
        loc="left",
        fontweight="bold",
        pad=7,
    )

    # Panel B: cross-scenario fixed-playback distributions for field variants.
    ax_b = figure.add_subplot(grid[0, 1])
    field_values = [
        [measurement["value"] for measurement in row["measurements"]]
        for row in field_rows
    ]
    positions = np.arange(len(field_rows), dtype=float)
    boxplot = ax_b.boxplot(
        field_values,
        positions=positions,
        widths=0.56,
        patch_artist=True,
        showfliers=False,
        whis=1.5,
        medianprops={"color": TEXT, "linewidth": 1.1},
        whiskerprops={"color": GREY, "linewidth": 0.8},
        capprops={"color": GREY, "linewidth": 0.8},
    )
    box_face_colors = [LIGHT_BLUE, "#EEF1F4", LIGHT_ORANGE, "#EEF1F4", "#EEF1F4"]
    box_edge_colors = [NAVY, GREY, ORANGE, GREY, GREY]
    point_markers = ["o", "o", "D", "o", "o"]
    rng = np.random.default_rng(20260715)
    for index, (row, values) in enumerate(zip(field_rows, field_values)):
        boxplot["boxes"][index].set_facecolor(box_face_colors[index])
        boxplot["boxes"][index].set_edgecolor(box_edge_colors[index])
        boxplot["boxes"][index].set_linewidth(1.0)
        jitter = rng.uniform(-0.11, 0.11, size=len(values))
        primary_pair = index in {0, 2}
        ax_b.scatter(
            positions[index] + jitter,
            values,
            s=11 if primary_pair else 9,
            marker=point_markers[index],
            facecolor=box_edge_colors[index] if primary_pair else "white",
            edgecolor=box_edge_colors[index],
            linewidth=0.6,
            alpha=0.82,
            zorder=3,
        )
    ax_b.set_xticks(
        positions,
        [row["label"] for row in field_rows],
        rotation=24,
        ha="right",
        rotation_mode="anchor",
    )
    ax_b.set_ylabel("Target-tube risk mass")
    ax_b.set_xlim(-0.55, len(field_rows) - 0.45)
    ax_b.set_ylim(0.0, 2.8)
    ax_b.set_title(
        "B   Fixed-playback field response",
        loc="left",
        fontweight="bold",
        pad=7,
    )
    ax_b.grid(axis="y", color=LIGHT_GREY, linewidth=0.5, alpha=0.75)
    ax_b.grid(axis="x", visible=False)
    ax_b.set_axisbelow(True)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)

    # Panel C: paired progress under the empty-shadow condition.
    ax_c = figure.add_subplot(grid[0, 2])
    for pair in empty_pairs:
        ax_c.plot(
            [0, 1],
            [pair["full"], pair["no_veto"]],
            color=LIGHT_GREY,
            linewidth=0.85,
            zorder=1,
        )
        ax_c.scatter(0, pair["full"], s=19, color=NAVY, alpha=0.8, zorder=2)
        ax_c.scatter(1, pair["no_veto"], s=19, color=ORANGE, alpha=0.8, zorder=2)

    full_values = np.array([pair["full"] for pair in empty_pairs], dtype=float)
    no_veto_values = np.array([pair["no_veto"] for pair in empty_pairs], dtype=float)
    full_median = float(np.median(full_values))
    no_veto_median = float(np.median(no_veto_values))
    ax_c.scatter(
        [0, 1],
        [full_median, no_veto_median],
        marker="s",
        s=72,
        color=[NAVY, ORANGE],
        edgecolor="white",
        linewidth=1.1,
        zorder=4,
    )
    ax_c.text(-0.05, full_median - 0.8, f"{full_median:.2f}", ha="right", va="top", color=NAVY, fontweight="bold")
    ax_c.text(1.05, no_veto_median + 0.65, f"{no_veto_median:.2f}", ha="left", va="bottom", color=ORANGE, fontweight="bold")
    ax_c.set_xlim(-0.36, 1.36)
    ax_c.set_ylim(73.5, 94.0)
    ax_c.set_xticks([0, 1], ["Full\ncoupling", "No decision\nveto"])
    ax_c.set_ylabel("Progress (m)")
    ax_c.set_title(
        "C   Empty-shadow progress",
        loc="left",
        fontweight="bold",
        pad=7,
    )
    ax_c.text(
        0.5,
        0.975,
        r"$\Delta_{\mathrm{median}}=+8.34$ m",
        transform=ax_c.transAxes,
        ha="center",
        va="top",
        fontsize=6.7,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 1.5},
    )
    ax_c.grid(axis="y", color=LIGHT_GREY, linewidth=0.6, alpha=0.85)
    ax_c.set_axisbelow(True)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)

    for extension in ("pdf", "svg", "png"):
        output = figure_root / f"fig_ablation_summary.{extension}"
        save_kwargs: dict[str, Any] = {
            "bbox_inches": "tight",
            "facecolor": "white",
        }
        if extension == "png":
            save_kwargs["dpi"] = 600
        figure.savefig(output, **save_kwargs)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the SciencePlots-styled paired ablation figure."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=ROOT,
        help="Repository root used for default input and output paths.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Directory containing the heldout_v3_* benchmark outputs.",
    )
    parser.add_argument(
        "--figure-root",
        type=Path,
        default=None,
        help="Destination directory for PDF, SVG, PNG, and source CSV files.",
    )
    parser.add_argument(
        "--channels-true-episodes",
        action="append",
        type=Path,
        default=None,
        help=(
            "True-threat channel JSONL; repeat when the suite was split across "
            "multiple resumable output directories."
        ),
    )
    parser.add_argument(
        "--field-true-episodes",
        type=Path,
        default=None,
        help="True-threat field-component JSONL.",
    )
    parser.add_argument(
        "--channels-empty-episodes",
        type=Path,
        default=None,
        help="Empty-shadow channel JSONL.",
    )
    return parser.parse_args()


def _first_existing(candidates: list[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


if __name__ == "__main__":
    args = parse_args()
    project_root = args.project_root.resolve()
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else project_root / "outputs" / "revision_r1c1"
    )
    figure_root = (
        args.figure_root.resolve()
        if args.figure_root is not None
        else project_root / "figures" / "revision_r1c1"
    )
    if args.channels_true_episodes is not None:
        channel_episode_paths = [path.resolve() for path in args.channels_true_episodes]
    else:
        channel_episode_paths = [
            path
            for path in (
                output_root / "heldout_v3_channels_true" / "episodes.jsonl",
                output_root / "heldout_v3_channels_true_remaining" / "episodes.jsonl",
            )
            if path.exists()
        ]
        if not channel_episode_paths:
            channel_episode_paths = [
                output_root / "channels_true" / "episodes.jsonl"
            ]

    field_episode_path = (
        args.field_true_episodes.resolve()
        if args.field_true_episodes is not None
        else _first_existing(
            [
                output_root / "heldout_v3_field_true" / "episodes.jsonl",
                output_root / "field_true" / "episodes.jsonl",
            ]
        )
    )
    empty_episode_path = (
        args.channels_empty_episodes.resolve()
        if args.channels_empty_episodes is not None
        else _first_existing(
            [
                output_root / "heldout_v3_channels_empty" / "episodes.jsonl",
                output_root / "channels_empty" / "episodes.jsonl",
            ]
        )
    )
    build_figure(
        channel_episode_paths=channel_episode_paths,
        field_episode_path=field_episode_path,
        empty_episode_path=empty_episode_path,
        figure_root=figure_root,
    )
