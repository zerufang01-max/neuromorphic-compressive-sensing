"""Fig. 2: NMSE vs unfolded depth (a) and vs energy (b) at s = 28, NMSE vs sparsity
at L = 10 (c), AWGN robustness at s = 28, L = 10 (d), layer-wise spikes and
readout correction at s = 14, L = 6 (e), coefficient distributions (f).

Inputs: <output>/metrics/results.json and snr_results.json (evaluate.py) and
the checkpoints of EXAMPLE_CONDITION_ID.  All S-LISTA quantities come from
models.SpikingLISTA.trace, i.e. exactly the model that was trained.
"""

from __future__ import annotations

import argparse
import sys
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator, MultipleLocator, NullFormatter
import numpy as np
import torch

from config import (
    CONDITIONS, DEPTH_SWEEP_CONDITION, DEPTH_SWEEP_DEPTHS, METHODS, SPARSITY_SWEEP_DEPTH,
    DISTRIBUTION_SAMPLES, DISTRIBUTION_SEED, EXAMPLE_ANN_ID,
    EXAMPLE_CANDIDATES, EXAMPLE_CONDITION_ID, EXAMPLE_RULE, EXAMPLE_SELECTION_SEED,
    EXAMPLE_SLISTA_ID, NUMERICAL_EPS, SNR_MODEL_IDS, TRAINING_SEEDS,
)
from data import generator, sparse_batch
from utils import DEFAULT_OUTPUT, checkpoint_path, load_model, read_json, save_json


COLORS = {
    "lista": "#3976B8",
    "alista": "#3D9B73",
    "lamp": "#8067A8",
    "slista": "#D84A3A",
}
INK = "#253746"
TARGET = "#C2C8CE"
GRID = "#E8EAED"
DIAGONAL = "#D86A67"
ESTIMATE_BLUE = "#4278A9"
ESTIMATE_RED = "#D85C58"
SPIKE_POSITIVE = "#D84A3A"
SPIKE_NEGATIVE = "#2878B5"
CORRECTION = "#7968A6"
TOP_MARKER_AREA = 30.0

METHOD_LABELS = {
    "lista": "ANN LISTA",
    "alista": "ALISTA",
    "lamp": "LAMP",
    "slista": "S-LISTA",
}



def resolve_font_family(requested: str = "Aptos"):
    """Use Aptos when installed, with portable sans-serif fallbacks."""
    candidates = [requested, "Arial", "Helvetica", "DejaVu Sans"]
    seen = set()
    for family in candidates:
        if not family or family in seen:
            continue
        seen.add(family)
        try:
            font_manager.findfont(family, fallback_to_default=False)
            return family
        except ValueError:
            continue
    return "DejaVu Sans"


def configure_style(requested_font: str = "Aptos"):
    selected_font = resolve_font_family(requested_font)
    mpl.rcParams.update({
        "font.family": selected_font,
        "font.size": 11.0,
        "axes.titlesize": 11.0,
        "axes.labelsize": 11.0,
        "axes.linewidth": 0.78,
        "xtick.labelsize": 10.0,
        "ytick.labelsize": 10.0,
        "legend.fontsize": 10.0,
        "lines.linewidth": 1.50,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.025,
    })
    return selected_font



MARKERS = {"lista": "o", "alista": "o", "lamp": "o", "slista": "s"}


def clean_axis(ax, grid_axis="both"):
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.45, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2.6, width=0.6, pad=1.7)


def method_series(ax, x, y, yerr, method):
    """One method: line + markers + seed SD error bars."""
    ax.errorbar(x, y, yerr=yerr, color=COLORS[method], marker=MARKERS[method],
                markersize=math.sqrt(TOP_MARKER_AREA), markerfacecolor=COLORS[method],
                markeredgecolor=COLORS[method], markeredgewidth=0.65,
                linewidth=1.15, capsize=2, elinewidth=0.7, label=METHOD_LABELS[method],
                zorder=3 if method == "slista" else 2)


def rows_of_all(rows):
    """Depth-sweep rows of panel b (s = 28, every method)."""
    return [r for r in rows if r.get("condition_id") == DEPTH_SWEEP_CONDITION
            and int(r["depth"]) in DEPTH_SWEEP_DEPTHS]


def rows_of(rows, method, **match):
    return [r for r in rows if r["method"] == method
            and all(r.get(k) == v for k, v in match.items())]


def plot_depth_panel(ax, rows, key, ylabel):
    """Fig. 2a: s = 28, every method at every depth."""
    for method in METHODS:
        series = sorted(rows_of(rows, method, condition_id=DEPTH_SWEEP_CONDITION),
                        key=lambda r: r["depth"])
        series = [r for r in series if int(r["depth"]) in DEPTH_SWEEP_DEPTHS]
        method_series(ax, [int(r["depth"]) for r in series], [r[key] for r in series],
                      [r.get(key + "_std", 0.0) for r in series], method)
    ax.set_xticks(DEPTH_SWEEP_DEPTHS)
    ax.set_xlabel(r"Unfolded depth $L$", labelpad=2)
    ax.set_ylabel(ylabel)
    clean_axis(ax)


def plot_energy_nmse_panel(ax, rows):
    """Fig. 2b: the depth-sweep models of panel a in the energy-NMSE plane
    (s = 28).  For every method the points are ordered by depth from left to
    right, because the energy grows with L (state this in the caption)."""
    for method in METHODS:
        series = sorted(rows_of(rows, method, condition_id=DEPTH_SWEEP_CONDITION),
                        key=lambda r: r["depth"])
        series = [r for r in series if int(r["depth"]) in DEPTH_SWEEP_DEPTHS]
        x = [r["energy_uj"] for r in series]
        y = [r["nmse_db"] for r in series]
        method_series(ax, x, y, [r.get("nmse_db_std", 0.0) for r in series], method)
    ax.set_xscale("log")
    energies = [r["energy_uj"] for r in rows_of_all(rows) if r["energy_uj"] > 0]
    low, high = min(energies) / 1.4, max(energies) * 1.4
    ax.set_xlim(low, high)
    ax.set_xticks([t for t in (0.05, 0.1, 0.2, 0.5, 1, 2, 4, 10) if low <= t <= high])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r"Energy ($\mu$J/sample)", labelpad=2)
    ax.set_ylabel("NMSE (dB)")
    clean_axis(ax)


def plot_sparsity_panel(ax, rows):
    """Fig. 2c: NMSE vs s at L = SPARSITY_SWEEP_DEPTH."""
    sparsities = [c["s"] for c in CONDITIONS]
    for method in METHODS:
        series = sorted((r for r in rows_of(rows, method)
                         if int(r["depth"]) == SPARSITY_SWEEP_DEPTH), key=lambda r: r["s"])
        method_series(ax, [int(r["s"]) for r in series], [r["nmse_db"] for r in series],
                      [r.get("nmse_db_std", 0.0) for r in series], method)
    ax.set_xticks(sparsities)
    ax.set_xlabel(r"Sparsity $s$", labelpad=2)
    ax.set_ylabel("NMSE (dB)")
    clean_axis(ax)


def plot_snr_panel(ax, rows):
    """Fig. 2d: NMSE vs measurement SNR, s = 28, L = SPARSITY_SWEEP_DEPTH."""
    for method in METHODS:
        series = [r for r in rows if r["id"] == f"{method}_l{SPARSITY_SWEEP_DEPTH}"]
        series.sort(key=lambda r: 50.0 if r.get("snr_db") is None else float(r["snr_db"]))
        x = [50.0 if r.get("snr_db") is None else float(r["snr_db"]) for r in series]
        method_series(ax, x, [r["nmse_db"] for r in series],
                      [r.get("nmse_db_std", 0.0) for r in series], method)
    ax.set_xticks([0, 10, 20, 30, 40, 50], ["0", "10", "20", "30", "40", r"$\infty$"])
    ax.set_xlabel("Measurement SNR (dB)", labelpad=2)
    ax.set_ylabel("NMSE (dB)")
    clean_axis(ax)


def stems(ax, values, color, width, alpha=1.0, threshold=1e-10, zorder=3):
    values = np.asarray(values)
    active = np.flatnonzero(np.abs(values) > threshold)
    if active.size:
        ax.vlines(active, 0.0, values[active], color=color, linewidth=width,
                  alpha=alpha, zorder=zorder)


def signed_update_stems(ax, previous, current):
    """Draw only the new +/-1 segment added at the current S-LISTA layer."""
    previous = np.asarray(previous)
    current = np.asarray(current)
    delta = current - previous
    positive = np.flatnonzero(delta > 1e-8)
    negative = np.flatnonzero(delta < -1e-8)
    if positive.size:
        ax.vlines(
            positive, previous[positive], current[positive],
            color=SPIKE_POSITIVE, linewidth=1.55, alpha=1.0, zorder=5,
        )
    if negative.size:
        ax.vlines(
            negative, previous[negative], current[negative],
            color=SPIKE_NEGATIVE, linewidth=1.55, alpha=1.0, zorder=5,
        )


def coefficient_axis(ax, y_limit, show_y: bool, show_x: bool):
    ax.axhline(0.0, color="#9EA2A7", linewidth=0.45, zorder=0)
    ax.set_xlim(-3, 258)
    ax.set_ylim(-y_limit, y_limit)
    ax.yaxis.set_major_locator(MultipleLocator(2.0))
    ax.set_xticks([0, 128, 255])
    if not show_x:
        ax.set_xticklabels([])
    if not show_y:
        ax.set_yticklabels([])
    clean_axis(ax, "y")


def plot_layer_group(fig, spec, layer_data):
    """Spike accumulation over the L-1 spiking layers, then
    spike-only code + readout correction (eta * m * v) = final code."""
    target = np.asarray(layer_data["target"])
    ledger = np.asarray(layer_data["ledger"])
    correction = np.asarray(layer_data["correction"])
    output = np.asarray(layer_data["output"])
    layers = ledger.shape[0]
    y_limit = max(4.2, 1.05 * float(np.max(np.abs(np.concatenate([
        target, ledger.reshape(-1), correction, output,
    ])))))
    block = spec.subgridspec(4, 1, height_ratios=(0.14, 1.0, 0.26, 1.0),
                            hspace=0.24)
    header = fig.add_subplot(block[0])
    header.set_axis_off()
    handles = [
        Line2D([0], [0], color=TARGET, linewidth=1.8, label="Target"),
        Line2D([0], [0], color=INK, linewidth=1.4,
               label="Sparse code"),
        Line2D([0], [0], color=SPIKE_POSITIVE, linewidth=1.8,
               label="+1 update"),
        Line2D([0], [0], color=SPIKE_NEGATIVE, linewidth=1.8,
               label="−1 update"),
    ]
    header.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.1),
                  ncol=4, frameon=False, fontsize=9.5, columnspacing=1.5,
                  handlelength=1.5, handletextpad=0.5, borderaxespad=0)
    upper = block[1].subgridspec(1, layers, wspace=0.48)
    axes = []
    for k in range(layers):
        ax = fig.add_subplot(upper[0, k])
        axes.append(ax)
        previous = np.zeros_like(target) if k == 0 else ledger[k - 1]
        stems(ax, target, TARGET, 1.1, alpha=0.9, zorder=1)
        stems(ax, ledger[k], INK, 0.92, zorder=3)
        signed_update_stems(ax, previous, ledger[k])
        coefficient_axis(ax, y_limit, True, False)
        ax.set_title(f"Layer {k + 1}", pad=4, fontweight="semibold")
        if k == 0:
            ax.set_ylabel("Coefficient")
    for left, right in zip(axes[:-1], axes[1:]):
        lb, rb = left.get_position(), right.get_position()
        y = lb.y1 + 0.012
        left.annotate("", xy=(rb.x0 - 0.006, y),
                      xytext=(lb.x1 + 0.006, y),
                      xycoords=fig.transFigure, textcoords=fig.transFigure,
                      arrowprops=dict(arrowstyle="->", color=INK, lw=1.0),
                      annotation_clip=False)

    lower = block[3].subgridspec(1, 7,
        width_ratios=(1, 0.38, 1, 0.38, 1, 0.40, 1), wspace=0)
    titles = ["Spike-only\nsparse code", "Readout\ncorrection",
              "Final\nsparse code", "Target"]
    values = [ledger[-1], correction, output, target]
    for i, (title, value) in enumerate(zip(titles, values)):
        ax = fig.add_subplot(lower[0, 2 * i])
        axes.append(ax)
        if i == 0:
            stems(ax, target, TARGET, 1.1, alpha=0.9, zorder=1)
        stems(ax, value, CORRECTION if i == 1 else INK, 1.0, zorder=3)
        coefficient_axis(ax, y_limit, True, True)
        ax.set_title(title, pad=5, fontweight="semibold")
        if i == 0:
            ax.set_ylabel("Coefficient")
    # Leave the Target panel outside the arithmetic expression.
    for i, symbol in enumerate(("+", "=")):
        left, right = axes[layers + i].get_position(), axes[layers + 1 + i].get_position()
        fig.text((left.x1 + right.x0) / 2 - 0.012,
                 (left.y0 + left.y1) / 2, symbol,
                 fontsize=22, ha="center", va="center", color=INK)
    return axes


def distribution_extent(values: dict, amplitude_max: float):
    arrays = [np.asarray(value) for value in values.values()]
    maximum = max(float(amplitude_max), *(float(np.max(np.abs(x))) for x in arrays))
    return max(1.0, math.ceil((maximum + 0.15) * 2.0) / 2.0)


def plot_distribution_group(fig, spec, values: dict, amplitude_max: float,
                            max_points: int, seed: int):
    target = np.asarray(values["target"])
    columns = [
        ("ANN LISTA", np.asarray(values["ann"]), ESTIMATE_BLUE),
        ("S-LISTA, spikes only", np.asarray(values["slista_discrete"]),
         ESTIMATE_RED),
        ("S-LISTA, corrected", np.asarray(values["slista_corrected"]),
         ESTIMATE_RED),
    ]
    extent = distribution_extent(values, amplitude_max)
    bins = np.arange(-extent, extent + 0.2501, 0.25, dtype=np.float64)
    rng = np.random.default_rng(seed)
    if target.size > max_points:
        selected = np.sort(rng.choice(target.size, max_points, replace=False))
    else:
        selected = np.arange(target.size)

    # Use the unchanged bins and density normalization to find a shared scale.
    density_max = max(
        float(np.max(np.histogram(array, bins=bins, density=True)[0]))
        for array in [target] + [estimate for _, estimate, _ in columns]
    )
    density_upper = max(0.1, 1.10 * density_max)

    grid = spec.subgridspec(2, 3, height_ratios=(1.65, 1.0),
                            hspace=0.82, wspace=0.38)
    axes = []
    for column, (title, estimate, color) in enumerate(columns):
        scatter_ax = fig.add_subplot(grid[0, column])
        hist_ax = fig.add_subplot(grid[1, column])
        axes.extend((scatter_ax, hist_ax))

        scatter_ax.plot([-extent, extent], [-extent, extent], color=DIAGONAL,
                        linewidth=0.85, zorder=1, label=r"$y=x$")
        scatter_ax.scatter(
            target[selected], estimate[selected], s=2.0, color=color,
            alpha=0.28, linewidths=0, rasterized=True, zorder=2,
        )
        scatter_ax.axhline(0.0, color="#A7ABB0", linewidth=0.42,
                           linestyle=(0, (2, 2)), zorder=0)
        scatter_ax.axvline(0.0, color="#D6D8DB", linewidth=0.4, zorder=0)
        scatter_ax.set_xlim(-extent, extent)
        scatter_ax.set_ylim(-extent, extent)
        scatter_ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        scatter_ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        scatter_ax.set_aspect("equal", adjustable="box")
        scatter_ax.set_title(title, pad=3.0, fontweight="semibold")
        scatter_ax.set_xlabel("Target coefficient", labelpad=1.5)
        if column == 0:
            scatter_ax.set_ylabel("Estimated coefficient")
        clean_axis(scatter_ax)
        scatter_ax.legend(loc="upper left", frameon=False, fontsize=9,
                          borderaxespad=0.2, handlelength=1.2)

        hist_ax.hist(target, bins=bins, density=True, color=TARGET,
                     alpha=0.78, edgecolor="none", label="Target")
        hist_ax.hist(estimate, bins=bins, density=True, color=color,
                     alpha=0.56, edgecolor="none", label="Estimate")
        hist_ax.axvline(0.0, color="#989CA1", linewidth=0.42, zorder=0)
        hist_ax.set_xlim(-extent, extent)
        hist_ax.set_ylim(0.0, density_upper)
        hist_ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
        hist_ax.set_xlabel("Coefficient value", labelpad=1.5)
        if column == 0:
            hist_ax.set_ylabel("Density")
        clean_axis(hist_ax, "y")
        hist_ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01),
                       ncol=2, frameon=False, fontsize=9, columnspacing=1.0,
                       borderaxespad=0.1, handlelength=1.2)
    return axes


def panel_label(ax, text: str, x=-0.18, y=1.12):
    ax.text(x, y, text, transform=ax.transAxes, fontsize=15.5,
            fontweight="bold", va="top", ha="left", clip_on=False)


def build_figure(energy_rows, snr_rows, layer_data, distribution_values,
                 cfg: Protocol, args):
    """Render all panels on one canvas; inputs are plain Python/NumPy data."""
    # Reserve height for the layer sequence and larger square scatter plots.
    fig = plt.figure(figsize=(8.8, 8.3), facecolor="white")
    outer = fig.add_gridspec(
        3, 1, height_ratios=(0.90, 2.15, 1.90),
        left=0.075, right=0.975, bottom=0.050, top=0.965,
        hspace=0.33,
    )

    top = outer[0].subgridspec(1, 4, wspace=0.36)
    top_axes = [fig.add_subplot(top[0, index]) for index in range(4)]
    plot_depth_panel(top_axes[0], energy_rows, "nmse_db", "NMSE (dB)")
    plot_energy_nmse_panel(top_axes[1], energy_rows)
    plot_sparsity_panel(top_axes[2], energy_rows)
    plot_snr_panel(top_axes[3], snr_rows)
    # Panels a-d all show NMSE: one common y range and ticks; the y title is
    # written once, on panel a.
    values = [v for ax in top_axes for line in ax.lines
              for v in np.asarray(line.get_ydata(), dtype=float) if np.isfinite(v)]
    lower = 10.0 * math.floor((min(values) - 2.0) / 10.0)
    upper = 10.0 * math.ceil((max(values) + 2.0) / 10.0)
    for index, ax in enumerate(top_axes):
        ax.set_ylim(lower, upper)
        ax.yaxis.set_major_locator(MultipleLocator(20.0))
        if index:
            ax.set_ylabel("")
        panel_label(ax, chr(ord("a") + index), x=-0.30 if index else -0.40, y=1.12)
    # Legend in panel d without error-bar glyphs (seed SDs are drawn but are
    # smaller than the markers; state this in the caption).
    handles = [Line2D([0], [0], color=COLORS[m], marker=MARKERS[m], linewidth=1.15,
                      markersize=math.sqrt(TOP_MARKER_AREA), label=METHOD_LABELS[m])
               for m in METHODS]
    top_axes[3].legend(handles=handles, loc="lower left", frameon=False, fontsize=7.5,
                       handlelength=1.4, handletextpad=0.35, borderaxespad=0.2,
                       labelspacing=0.2)

    layer_axes = plot_layer_group(fig, outer[1], layer_data)
    layer_box = outer[1].get_position(fig)
    fig.text(0.016, layer_box.y1, "e", fontsize=15.5,
             fontweight="bold", ha="left", va="top")

    dist_axes = plot_distribution_group(
        fig, outer[2], distribution_values, cfg.amplitude_max,
        args.distribution_points, args.distribution_seed,
    )
    # The square scatter axes shrink inside their GridSpec cells, so an
    # axes-relative panel label can collide with the y ticks.  Anchor f to
    # the complete distribution block instead, aligned with the left edge
    # used by panel e.
    distribution_box = outer[2].get_position(fig)
    fig.text(
        0.016, distribution_box.y1 + 0.006, "f",
        fontsize=15.5, fontweight="bold", ha="left", va="top",
    )

    layer_left = min(ax.get_position().x0 for ax in layer_axes)
    layer_right = max(ax.get_position().x1 for ax in layer_axes)
    layer_bottom = min(ax.get_position().y0 for ax in layer_axes)
    fig.text((layer_left + layer_right) / 2, layer_bottom - 0.030,
             "Coefficient index", ha="center", va="top", fontsize=11.0)

    return fig


# ----------------------------------------------------------------- data part
def sample_nmse(estimate, target):
    return ((estimate - target).square().sum(dim=1)
            / target.square().sum(dim=1).clamp_min(NUMERICAL_EPS))


@torch.no_grad()
def choose_example(models, matrix, condition, cfg, rule):
    """Pick the panel-e signal from EXAMPLE_CANDIDATES test-distribution signals
    (shared seed) and the five trained models.

    rule = "median": the (seed, signal) pair whose final NMSE is the median of
    all pairs, i.e. a representative example.  rule = "best": the minimum."""
    rng = generator(EXAMPLE_SELECTION_SEED, matrix.device)
    target, measured = sparse_batch(EXAMPLE_CANDIDATES, condition["s"], matrix, rng,
                                    cfg.amplitude_min, cfg.amplitude_max)
    scores = torch.stack([sample_nmse(model(measured)[0], target) for model in models])
    flat = scores.flatten()
    index = int(flat.argsort()[len(flat) // 2]) if rule == "median" else int(flat.argmin())
    seed_index, sample = divmod(index, EXAMPLE_CANDIDATES)
    return seed_index, target[sample:sample + 1], measured[sample:sample + 1], {
        "rule": rule, "sample_index": sample, "candidates": EXAMPLE_CANDIDATES,
        "selection_seed": EXAMPLE_SELECTION_SEED}


@torch.no_grad()
def distribution_data(slista, ann, matrix, condition, cfg):
    """Estimates on the true support of DISTRIBUTION_SAMPLES fresh signals."""
    rng = generator(DISTRIBUTION_SEED, matrix.device)
    target, measured = sparse_batch(DISTRIBUTION_SAMPLES, condition["s"], matrix, rng,
                                    cfg.amplitude_min, cfg.amplitude_max)
    trace = slista.trace(measured)
    ann_output, _ = ann(measured)
    support = target != 0
    return {"target": target[support].cpu().numpy(),
            "ann": ann_output[support].cpu().numpy(),
            "slista_discrete": trace["spike_code"][support].cpu().numpy(),
            "slista_corrected": trace["output"][support].cpu().numpy()}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(TRAINING_SEEDS))
    parser.add_argument("--distribution-seed-model", type=int, default=TRAINING_SEEDS[0],
                        help="Training seed of the models shown in panel f.")
    parser.add_argument("--example-rule", choices=("median", "best"), default=EXAMPLE_RULE)
    parser.add_argument("--basename", default="fig2_synthetic")
    parser.add_argument("--results", type=Path, default=None,
                        help="Results JSON for panels a-c (default <output>/metrics/results.json); "
                             "use <output>/energy/results_energy.json for the energies of the full energy model.")
    parser.add_argument("--font-family", default="Aptos")
    parser.add_argument("--formats", nargs="+", choices=("png", "pdf", "svg"),
                        default=("png", "pdf"))
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--distribution-points", type=int, default=2000)
    args = parser.parse_args()
    args.distribution_seed = DISTRIBUTION_SEED
    return args


@torch.no_grad()
def main():
    args = parse_args()
    configure_style(args.font_family)
    device = torch.device(args.device)
    root = args.output.expanduser().resolve()

    slistas = [load_model(checkpoint_path(root, seed, EXAMPLE_CONDITION_ID, EXAMPLE_SLISTA_ID),
                          device) for seed in args.seeds]
    _, matrix, job, cfg = slistas[0]
    condition = job["condition"]
    seed_index, target, measured, metadata = choose_example(
        [entry[0] for entry in slistas], matrix, condition, cfg, args.example_rule)
    example = slistas[seed_index][0]
    trace = example.trace(measured)
    ledger = torch.stack(trace["codes"])[:, 0]

    panel_f = args.seeds.index(args.distribution_seed_model)
    ann = load_model(checkpoint_path(root, args.distribution_seed_model,
                                     EXAMPLE_CONDITION_ID, EXAMPLE_ANN_ID), device)[0]
    distribution = distribution_data(slistas[panel_f][0], ann, matrix, condition, cfg)

    energy_rows = read_json(args.results or root / "metrics" / "results.json")
    snr_rows = [r for r in read_json(root / "metrics" / "snr_results.json")
                if r["id"] in SNR_MODEL_IDS]
    layer_data = {"target": target[0].cpu().numpy(),
                  "ledger": ledger.cpu().numpy(),
                  "correction": trace["correction"][0].cpu().numpy(),
                  "output": trace["output"][0].cpu().numpy()}
    fig = build_figure(energy_rows, snr_rows, layer_data, distribution, cfg, args)
    out = root / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for extension in args.formats:
        fig.savefig(out / f"{args.basename}.{extension}", dpi=args.dpi)
    plt.close(fig)

    def db(value):
        return 10 * math.log10(max(float(value), 1e-30))
    metadata.update(
        example_model_seed=args.seeds[seed_index],
        spike_only_nmse_db=db(sample_nmse(trace["spike_code"], target)[0]),
        final_nmse_db=db(sample_nmse(trace["output"], target)[0]),
        theta=[float(t) for t in example.theta], eta=float(example.eta),
        distribution_model_seed=args.distribution_seed_model,
        distribution_samples=DISTRIBUTION_SAMPLES)
    save_json(out / f"{args.basename}_metadata.json", metadata)
    np.savez_compressed(out / f"{args.basename}_data.npz", **layer_data,
                        **{f"distribution_{k}": v for k, v in distribution.items()})
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
