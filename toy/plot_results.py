"""Composite sparse-recovery figure: energy, noise robustness, and coefficient analysis."""

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
from matplotlib.ticker import MaxNLocator, MultipleLocator
import numpy as np
import torch

from config import NUMERICAL_EPS, Protocol
from data import generator, measurement_matrix, sparse_batch
from models import build_model
from utils import DEFAULT_OUTPUT, PAPER_CONFIG, read_json


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

SNR_STYLES = {
    "lista_l5": ("lista", "o", "-", "ANN LISTA"),
    "alista_l5": ("alista", "o", "-", "ALISTA"),
    "lamp_l5": ("lamp", "o", "-", "LAMP"),
    "slista_l20_t1": ("slista", "s", "-", "S-LISTA"),
    "slista_l20_t4": ("slista", "s", "--", "S-LISTA, T=4"),
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


def load_json(path: Path):
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list):
        raise RuntimeError(f"Expected a list of rows in {path}.")
    return value


def load_payload(path: Path, device: torch.device):
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or "model" not in payload or "job" not in payload:
        raise RuntimeError(
            f"{path} is not a train.py checkpoint "
            "(expected keys 'model' and 'job')."
        )
    return payload, payload["job"]


def validate_job(job: dict, method: str, path: Path):
    if job.get("method") != method:
        raise RuntimeError(
            f"Checkpoint {path} contains method={job.get('method')!r}, "
            f"not {method!r}."
        )
    missing = [key for key in ("condition", "depth", "time_steps", "parameters")
               if key not in job]
    if missing:
        raise RuntimeError(f"Checkpoint job is missing: {', '.join(missing)}")


def infer_ann_checkpoint(slista_path: Path, ann_id: str) -> Path:
    condition_id = slista_path.name.split("__", 1)[0]
    candidate = slista_path.with_name(f"{condition_id}__{ann_id}.pth")
    if candidate.is_file():
        return candidate
    candidates = sorted(slista_path.parent.glob(f"{condition_id}__lista_l*.pth"))
    if len(candidates) == 1:
        return candidates[0]
    detail = "\n".join(f"  {path}" for path in candidates) or "  (none found)"
    raise FileNotFoundError(
        "Could not uniquely locate the companion LISTA checkpoint. "
        f"Pass --ann-checkpoint explicitly. Candidates:\n{detail}"
    )


def build_from_job(job: dict, payload: dict, device: torch.device, cfg: Protocol):
    condition = job["condition"]
    matrix = measurement_matrix(
        cfg.n, int(condition["m"]), cfg.max_m, cfg.matrix_seed, device
    )
    model = build_model(
        job["method"], matrix, int(job["depth"]), int(job["time_steps"]),
        job["parameters"],
    ).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    return model, matrix


@torch.no_grad()
def trace_slista(model, measurements: torch.Tensor, keep_ledger: bool = True):
    """Replay SpikingLISTA and expose the exact per-layer additive ledger."""
    initial = model.P(measurements)
    states = [torch.zeros_like(initial) for _ in range(model.depth)]
    cumulative = torch.zeros_like(initial)
    layer_spikes = (
        torch.zeros(model.depth, *initial.shape, device=initial.device,
                    dtype=initial.dtype)
        if keep_ledger else None
    )
    theta, tau = model.theta_snn, model.decoder_tau
    for _ in range(model.time_steps):
        code = torch.zeros_like(initial)
        next_states = []
        for layer_index in range(model.depth):
            drive = (initial if layer_index == 0
                     else initial - model.G[layer_index - 1](code))
            membrane = tau * states[layer_index] + drive
            spike = (membrane >= theta).to(initial.dtype)
            spike -= (-membrane >= theta).to(initial.dtype)
            code = code + spike
            next_states.append(membrane - theta * spike)
            if layer_spikes is not None:
                layer_spikes[layer_index] += spike
        states = next_states
        cumulative += code

    elapsed = float(model.time_steps)
    discrete = cumulative / elapsed
    support = (cumulative != 0).to(initial.dtype)
    correction = support * states[-1] / theta.clamp_min(1e-12) / elapsed
    output = discrete + correction
    ledger = None
    if layer_spikes is not None:
        ledger = torch.cumsum(layer_spikes, dim=0) / elapsed
        if not torch.equal(ledger[-1], discrete):
            raise RuntimeError("Internal trace error: layer ledger does not close.")
    return {
        "ledger": ledger,
        "discrete": discrete,
        "correction": correction,
        "output": output,
    }


def sample_nmse(estimate: torch.Tensor, target: torch.Tensor):
    return ((estimate - target).square().sum(dim=1)
            / target.square().sum(dim=1).clamp_min(NUMERICAL_EPS))


def batches(total: int, batch_size: int) -> Iterable[int]:
    remaining = int(total)
    while remaining > 0:
        size = min(remaining, int(batch_size))
        yield size
        remaining -= size


@torch.no_grad()
def select_sample(model, matrix, condition: dict, cfg: Protocol, args):
    """Select the lowest final (continuously corrected) NMSE among all candidates."""
    rng = generator(args.selection_seed, matrix.device)
    targets, measurements, records = [], [], []
    offset = 0
    for size in batches(args.candidate_count, args.selection_batch_size):
        target, measured = sparse_batch(
            size, int(condition["s"]), matrix, rng,
            cfg.amplitude_min, cfg.amplitude_max,
        )
        trace = trace_slista(model, measured, keep_ledger=False)
        final_nmse = sample_nmse(trace["output"], target)
        discrete_nmse = sample_nmse(trace["discrete"], target)
        correction_ratio = (
            trace["correction"].square().sum(dim=1)
            / trace["output"].square().sum(dim=1).clamp_min(NUMERICAL_EPS)
        )
        for local in range(size):
            records.append((
                offset + local,
                float(final_nmse[local]),
                float(discrete_nmse[local]),
                float(correction_ratio[local]),
            ))
        targets.append(target.cpu())
        measurements.append(measured.cpu())
        offset += size

    values = np.asarray(records, dtype=np.float64)
    if not np.all(np.isfinite(values[:, 1])):
        raise RuntimeError("Non-finite final NMSE encountered during sample selection.")
    chosen_row = int(np.argmin(values[:, 1]))
    chosen_index = int(values[chosen_row, 0])
    target_all = torch.cat(targets, dim=0)
    measurement_all = torch.cat(measurements, dim=0)
    metadata = {
        "sample_index": chosen_index,
        "selection_seed": int(args.selection_seed),
        "candidate_count": int(args.candidate_count),
        "selection_batch_size": int(args.selection_batch_size),
        "selection": "minimum final NMSE across all candidates",
        "selection_metric": "final_nmse",
        "final_nmse_db": 10.0 * math.log10(max(values[chosen_row, 1], 1e-30)),
        "spike_only_nmse_db": 10.0 * math.log10(max(values[chosen_row, 2], 1e-30)),
        "correction_energy_ratio": float(values[chosen_row, 3]),
    }
    return (
        target_all[chosen_index:chosen_index + 1].to(matrix.device),
        measurement_all[chosen_index:chosen_index + 1].to(matrix.device),
        metadata,
    )


@torch.no_grad()
def get_indexed_sample(matrix, condition: dict, cfg: Protocol, index: int, seed: int):
    rng = generator(seed, matrix.device)
    remaining = int(index) + 1
    target = measured = None
    while remaining > 0:
        size = min(2048, remaining)
        target, measured = sparse_batch(
            size, int(condition["s"]), matrix, rng,
            cfg.amplitude_min, cfg.amplitude_max,
        )
        remaining -= size
    return target[-1:], measured[-1:]


@torch.no_grad()
def collect_distribution_data(slista, ann, matrix, condition: dict,
                              cfg: Protocol, args):
    """Collect nontrivial coefficient pairs on the known true support."""
    rng = generator(args.distribution_seed, matrix.device)
    result = {key: [] for key in (
        "target", "ann", "slista_discrete", "slista_corrected"
    )}
    for size in batches(args.distribution_samples, args.distribution_batch_size):
        target, measured = sparse_batch(
            size, int(condition["s"]), matrix, rng,
            cfg.amplitude_min, cfg.amplitude_max,
        )
        slista_trace = trace_slista(slista, measured, keep_ledger=False)
        ann_output, _ = ann(measured)
        support = target != 0
        result["target"].append(target[support].cpu())
        result["ann"].append(ann_output[support].cpu())
        result["slista_discrete"].append(slista_trace["discrete"][support].cpu())
        result["slista_corrected"].append(slista_trace["output"][support].cpu())
    return {key: torch.cat(value).numpy() for key, value in result.items()}


def displayed(rows, method: str, time_steps: int | None = None):
    series = [row for row in rows if row.get("method") == method]
    if time_steps is not None:
        series = [row for row in series if int(row.get("time_steps", 1)) == time_steps]
    # Four fixed depths per method; independent of the measured performance.
    depths = (5, 10, 15, 20) if method == "slista" else (2, 3, 4, 5)
    series = [row for row in series if int(row["depth"]) in depths]
    return sorted(series, key=lambda row: int(row["depth"]))


def marker_bounds(rows, condition_ids, include_t4: bool):
    values = []
    for condition_id in condition_ids:
        subset = [row for row in rows if row.get("condition_id") == condition_id]
        for method in ("lista", "alista", "lamp"):
            values.extend(float(row["stored_parameters"])
                          for row in displayed(subset, method))
        for time_steps in ((1, 4) if include_t4 else (1,)):
            values.extend(float(row["stored_parameters"])
                          for row in displayed(subset, "slista", time_steps))
    if not values:
        raise RuntimeError("No rows match --energy-conditions.")
    return min(values), max(values)


def marker_area(_parameter_count: float, _bounds):
    """Use one marker size: model size is already reflected by energy."""
    return TOP_MARKER_AREA


def energy_series(ax, rows, method: str, bounds, time_steps: int | None = None):
    series = displayed(rows, method, time_steps)
    if not series:
        return []
    x = np.asarray([row["energy_uj"] for row in series], dtype=float)
    y = np.asarray([row["nmse_db"] for row in series], dtype=float)
    if method == "slista":
        dashed = time_steps == 4
        marker, linestyle = "s", "--" if dashed else "-"
        face = "white" if dashed else COLORS[method]
    else:
        marker, linestyle, face = "o", "-", COLORS[method]
    ax.plot(x, y, color=COLORS[method], linestyle=linestyle,
            linewidth=1.15, alpha=0.9, zorder=2)
    if any("nmse_db_std" in r for r in series):
        ax.errorbar(x, y, xerr=[r.get("energy_uj_std",0) for r in series],
                    yerr=[r.get("nmse_db_std",0) for r in series],
                    fmt="none", color=COLORS[method], capsize=2, linewidth=0.7)
    ax.scatter(
        x, y,
        s=[marker_area(row["stored_parameters"], bounds) for row in series],
        marker=marker, facecolor=face, edgecolor=COLORS[method],
        linewidth=0.75, alpha=0.95, zorder=3,
    )
    return series


def clean_axis(ax, grid_axis="both"):
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.45, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2.6, width=0.6, pad=1.7)


def synchronize_nmse_axes(axes):
    """Compare a--c on one scale, expanding it if any data would be clipped."""
    values = []
    for ax in axes:
        for line in ax.lines:
            y = np.asarray(line.get_ydata(), dtype=float)
            values.extend(y[np.isfinite(y)].tolist())
    lower, upper = -35.0, 0.0
    if values:
        # Keep a small margin for markers at the limits as well as outliers.
        lower = min(lower, 5.0 * math.floor((min(values) - 0.5) / 5.0))
        upper = max(upper, 5.0 * math.ceil((max(values) + 0.5) / 5.0))
    for ax in axes:
        ax.set_ylim(lower, upper)
        ax.yaxis.set_major_locator(MultipleLocator(5.0))


def condition_title(rows, condition_id: str):
    if rows:
        row = rows[0]
        sparsity = int(row.get("sparsity", condition_id.rsplit("s", 1)[-1]))
        return rf"$s={sparsity}$"
    return condition_id


def plot_energy_panel(ax, all_rows, condition_id: str, bounds, include_t4: bool):
    rows = [row for row in all_rows if row.get("condition_id") == condition_id]
    if not rows:
        raise RuntimeError(f"No energy rows found for {condition_id!r}.")
    shown = []
    for method in ("lista", "alista", "lamp"):
        shown.extend(energy_series(ax, rows, method, bounds))
    shown.extend(energy_series(ax, rows, "slista", bounds, 1))
    if include_t4:
        shown.extend(energy_series(ax, rows, "slista", bounds, 4))
    x = np.asarray([row["energy_uj"] for row in shown], dtype=float)
    y = np.asarray([row["nmse_db"] for row in shown], dtype=float)
    xspan = max(float(np.ptp(x)), max(float(np.max(x)), 1.0) * 0.05)
    yspan = max(float(np.ptp(y)), 1.0)
    ax.set_xlim(max(0.0, float(np.min(x)) - 0.07 * xspan),
                float(np.max(x)) + 0.07 * xspan)
    step = 2 if yspan < 10 else 5 if yspan < 25 else 10
    ax.set_ylim(step * math.floor((float(np.min(y)) - 0.07 * yspan) / step),
                step * math.ceil((float(np.max(y)) + 0.07 * yspan) / step))
    ax.yaxis.set_major_locator(MultipleLocator(step))
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.set_title(condition_title(rows, condition_id), pad=3.0, fontweight="semibold")
    ax.set_xlabel(r"Energy ($\mu$J/sample)", labelpad=2)
    clean_axis(ax)


def plot_snr_panel(ax, rows, include_t4: bool):
    for model_id, (method, marker, linestyle, _label) in SNR_STYLES.items():
        if model_id.endswith("t4") and not include_t4:
            continue
        series = [row for row in rows if row.get("id") == model_id]
        if not series:
            continue
        series.sort(key=lambda row: 50.0 if row.get("snr_db") is None
                    else float(row["snr_db"]))
        x = [50.0 if row.get("snr_db") is None else float(row["snr_db"])
             for row in series]
        face = "white" if model_id.endswith("t4") else COLORS[method]
        ax.plot(
            x, [row["nmse_db"] for row in series], color=COLORS[method],
            marker=marker, linestyle=linestyle, linewidth=1.15,
            markersize=math.sqrt(TOP_MARKER_AREA), markerfacecolor=face,
            markeredgecolor=COLORS[method], markeredgewidth=0.65,
            label=_label,
        )
        if any("nmse_db_std" in r for r in series):
            ax.errorbar(x,[r["nmse_db"] for r in series],
                        yerr=[r.get("nmse_db_std",0) for r in series],
                        fmt="none",color=COLORS[method],capsize=2,linewidth=0.7)
    ax.set_xticks([0, 10, 20, 30, 40, 50],
                  ["0", "10", "20", "30", "40", r"$\infty$"])
    ax.set_xlabel("Measurement SNR (dB)", labelpad=2)
    clean_axis(ax)
    ax.legend(
        loc="upper right", frameon=False, ncol=1,
        handlelength=1.1, handletextpad=0.35, borderaxespad=0.2,
        labelspacing=0.15, fontsize=8.0,
    )


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
    """Show spike accumulation, then the exact correction identity."""
    target = np.asarray(layer_data["target"])
    ledger = np.asarray(layer_data["ledger"])
    correction = np.asarray(layer_data["correction"])
    output = np.asarray(layer_data["output"])
    if ledger.shape[0] != 5:
        raise RuntimeError("The manuscript layout expects K=5.")
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
    upper = block[1].subgridspec(1, 5, wspace=0.48)
    axes = []
    for k in range(5):
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
    titles = ["Spike-only\nsparse code", "Continuous\ncorrection",
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
        left, right = axes[5 + i].get_position(), axes[6 + i].get_position()
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


def method_legend(include_t4: bool):
    handles = [
        Line2D([0], [0], marker="o", color=COLORS[method], linewidth=1.2,
               markersize=4.0, label=METHOD_LABELS[method])
        for method in ("lista", "alista", "lamp")
    ]
    handles.append(Line2D(
        [0], [0], marker="s", color=COLORS["slista"], linewidth=1.2,
        markersize=4.0, markerfacecolor=COLORS["slista"], label="S-LISTA (T=1)",
    ))
    if include_t4:
        handles.append(Line2D(
            [0], [0], marker="s", color=COLORS["slista"], linewidth=1.2,
            linestyle="--", markersize=4.0, markerfacecolor="white",
            label="S-LISTA (T=4)",
        ))
    return handles


def build_figure(energy_rows, snr_rows, layer_data, distribution_values,
                 cfg: Protocol, args):
    """Render all panels on one canvas; inputs are plain Python/NumPy data."""
    condition_ids = list(args.energy_conditions)
    bounds = marker_bounds(energy_rows, condition_ids, args.include_t4)

    # Reserve height for the layer sequence and larger square scatter plots.
    fig = plt.figure(figsize=(8.8, 8.3), facecolor="white")
    outer = fig.add_gridspec(
        3, 1, height_ratios=(0.90, 2.15, 1.90),
        left=0.068, right=0.993, bottom=0.050, top=0.974,
        hspace=0.33,
    )

    top = outer[0].subgridspec(1, 4, wspace=0.40)
    top_axes = [fig.add_subplot(top[0, index]) for index in range(4)]
    for ax, condition_id in zip(top_axes[:3], condition_ids):
        plot_energy_panel(ax, energy_rows, condition_id, bounds, args.include_t4)
    plot_snr_panel(top_axes[3], snr_rows, args.include_t4)
    synchronize_nmse_axes(top_axes[:3])
    top_axes[3].yaxis.set_major_locator(MultipleLocator(5.0))
    for index, ax in enumerate(top_axes):
        panel_label(ax, chr(ord("a") + index), x=-0.27)
    top_axes[0].set_ylabel("NMSE (dB)")
    top_axes[3].set_ylabel("NMSE (dB)")

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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slista-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=PAPER_CONFIG)
    parser.add_argument("--ann-checkpoint", type=Path)
    parser.add_argument("--ann-id", default="lista_l5")
    parser.add_argument("--energy-results", type=Path,
                        default=None)
    parser.add_argument("--snr-results", type=Path,
                        default=None)
    parser.add_argument("--energy-conditions", nargs=3,
                        default=("m141_s14", "m141_s28", "m141_s42"))
    parser.add_argument("--output-dir", type=Path,
                        default=None)
    parser.add_argument("--basename", default="toy_nc_composite")
    parser.add_argument(
        "--font-family", default="Aptos",
        help="Preferred figure font; falls back to Arial/Helvetica/DejaVu Sans.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--candidate-count", type=int, default=6000)
    parser.add_argument("--selection-batch-size", type=int, default=512)
    parser.add_argument("--selection-seed", type=int, default=20260829)
    parser.add_argument("--sample-index", type=int)
    parser.add_argument("--example-seeds", nargs="+", type=int,
                        help="Select panel e across sibling seed directories; panel f is unchanged.")
    parser.add_argument("--distribution-samples", type=int, default=2000)
    parser.add_argument("--distribution-batch-size", type=int, default=512)
    parser.add_argument("--distribution-seed", type=int, default=20260830)
    parser.add_argument("--distribution-points", type=int, default=2000)
    parser.add_argument("--include-t4", action="store_true",
                        help="Also draw T=4 quantitative curves (off by default).")
    parser.add_argument("--formats", nargs="+", choices=("png", "pdf", "svg"),
                        default=("png", "pdf"))
    parser.add_argument("--dpi", type=int, default=400)
    args = parser.parse_args()
    root = args.output.expanduser().resolve()
    args.slista_checkpoint = args.slista_checkpoint or root/'checkpoints/m141_s14__slista_l5_t1.pth'
    args.ann_checkpoint = args.ann_checkpoint or root/'checkpoints/m141_s14__lista_l5.pth'
    args.energy_results = args.energy_results or root/'metrics/results.json'
    args.snr_results = args.snr_results or root/'metrics/snr_results.json'
    args.output_dir = args.output_dir or root/'figures'
    figure_config = read_json(args.config)['figure']
    for key, value in figure_config.items():
        flag = '--' + key.replace('_', '-')
        if not any(arg == flag or arg.startswith(flag + '=') for arg in sys.argv[1:]):
            setattr(args, key, value)
    # Automatic multi-seed selection overrides a legacy fixed index from JSON.
    # Explicit conflicting CLI requests still fail in validate_args.
    explicit_index = any(arg == '--sample-index' or arg.startswith('--sample-index=')
                         for arg in sys.argv[1:])
    if args.example_seeds and not explicit_index:
        args.sample_index = None
    return args


def validate_args(args):
    if args.example_seeds and args.sample_index is not None:
        raise ValueError("Do not combine --example-seeds and --sample-index.")
    if args.candidate_count < 1 or args.selection_batch_size < 1:
        raise ValueError("Candidate counts must be positive.")
    if args.distribution_samples < 1 or args.distribution_batch_size < 1:
        raise ValueError("Distribution sample counts must be positive.")
    if args.distribution_points < 1:
        raise ValueError("--distribution-points must be positive.")


def main():
    args = parse_args()
    validate_args(args)
    selected_font = configure_style(args.font_family)
    print(f"Figure font: {selected_font}")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = Protocol()

    slista_path = args.slista_checkpoint.expanduser().resolve()
    if not slista_path.is_file():
        raise FileNotFoundError(slista_path)
    ann_path = (args.ann_checkpoint.expanduser().resolve()
                if args.ann_checkpoint is not None
                else infer_ann_checkpoint(slista_path, args.ann_id))
    if not ann_path.is_file():
        raise FileNotFoundError(ann_path)

    slista_payload, slista_job = load_payload(slista_path, device)
    ann_payload, ann_job = load_payload(ann_path, device)
    validate_job(slista_job, "slista", slista_path)
    validate_job(ann_job, "lista", ann_path)
    if slista_job["condition"] != ann_job["condition"]:
        raise RuntimeError("The S-LISTA and LISTA checkpoints use different conditions.")
    if int(slista_job["depth"]) != 5 or int(slista_job["time_steps"]) != 1:
        raise RuntimeError(
            "The coefficient panel requires depth=5 and T=1; "
            f"the supplied checkpoint is K={slista_job['depth']}, "
            f"T={slista_job['time_steps']}."
        )

    slista, matrix = build_from_job(slista_job, slista_payload, device, cfg)
    ann, ann_matrix = build_from_job(ann_job, ann_payload, device, cfg)
    if not torch.equal(matrix, ann_matrix):
        raise RuntimeError("The checkpoints do not use the same sensing matrix.")
    condition = slista_job["condition"]

    # Keep the distribution panel on its original seed, independently of panel e.
    distribution_slista, distribution_ann = slista, ann
    if args.example_seeds:
        seed_root = slista_path.parent.parent.parent
        best = None
        candidates = []
        for seed in args.example_seeds:
            candidate_path = seed_root / f"seed{seed}" / "checkpoints" / slista_path.name
            payload, job = load_payload(candidate_path, device)
            validate_job(job, "slista", candidate_path)
            if job["condition"] != condition or int(job["depth"]) != 5 or int(job["time_steps"]) != 1:
                raise ValueError(f"Incompatible example checkpoint: {candidate_path}")
            model, candidate_matrix = build_from_job(job, payload, device, cfg)
            if not torch.equal(matrix, candidate_matrix):
                raise ValueError("Example seeds must use the same sensing matrix.")
            x, y, info = select_sample(model, matrix, condition, cfg, args)
            score = info["final_nmse_db"]
            candidates.append({"seed": seed, "best_nmse_db": score, "sample_index": info["sample_index"]})
            print(f"Panel e seed={seed}: best final NMSE={score:.4f} dB", flush=True)
            if best is None or score < best[0]:
                best = (score, seed, candidate_path, model, x, y, info)
        _, chosen_seed, slista_path, slista, target, measured, metadata = best
        metadata.update(example_model_seed=chosen_seed, example_seed_candidates=candidates,
                        selection="minimum final NMSE across all candidate samples and requested model seeds")
        print(f"Panel e selected model seed={chosen_seed}", flush=True)
    elif args.sample_index is None:
        target, measured, metadata = select_sample(
            slista, matrix, condition, cfg, args,
        )
    else:
        target, measured = get_indexed_sample(
            matrix, condition, cfg, args.sample_index, args.selection_seed,
        )
        metadata = {
            "sample_index": int(args.sample_index),
            "selection_seed": int(args.selection_seed),
            "selection": "explicit index",
        }

    slista_trace = trace_slista(slista, measured, keep_ledger=True)
    ann_output, ann_layer_list = ann(measured)
    # LISTA returns its intermediate estimates as a Python list.  Stack them
    # once here so the diagnostic archive has a regular [K, B, N] array.
    ann_layers = torch.stack(ann_layer_list, dim=0)
    distribution_values = collect_distribution_data(
        distribution_slista, distribution_ann, matrix, condition, cfg, args,
    )
    energy_rows = load_json(args.energy_results)
    snr_rows = load_json(args.snr_results)

    layer_data = {
        "target": target.detach().cpu().numpy()[0],
        "ledger": slista_trace["ledger"].detach().cpu().numpy()[:, 0],
        "correction": slista_trace["correction"].detach().cpu().numpy()[0],
        "output": slista_trace["output"].detach().cpu().numpy()[0],
    }
    fig = build_figure(
        energy_rows, snr_rows, layer_data, distribution_values, cfg, args,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for extension in args.formats:
        output = args.output_dir / f"{args.basename}.{extension}"
        fig.savefig(output, dpi=args.dpi)
        outputs.append(output)
    plt.close(fig)

    final_nmse = float(sample_nmse(slista_trace["output"], target)[0])
    spike_nmse = float(sample_nmse(slista_trace["discrete"], target)[0])
    correction_ratio = float(
        slista_trace["correction"].square().sum()
        / slista_trace["output"].square().sum().clamp_min(NUMERICAL_EPS)
    )
    metadata.update({
        "condition": condition,
        "slista_checkpoint": str(slista_path),
        "ann_checkpoint": str(ann_path),
        "slista_depth": int(slista.depth),
        "slista_time_steps": int(slista.time_steps),
        "ann_depth": int(ann.depth),
        "final_nmse_db": 10.0 * math.log10(max(final_nmse, 1e-30)),
        "spike_only_nmse_db": 10.0 * math.log10(max(spike_nmse, 1e-30)),
        "correction_energy_ratio": correction_ratio,
        "distribution_samples": int(args.distribution_samples),
        "distribution_support_values": int(distribution_values["target"].size),
        "distribution_scope": "ground-truth support only",
        "include_t4": bool(args.include_t4),
        "energy_conditions": list(args.energy_conditions),
        "figure_outputs": [str(path) for path in outputs],
    })
    metadata_path = args.output_dir / f"{args.basename}_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    np.savez_compressed(
        args.output_dir / f"{args.basename}_data.npz",
        target=layer_data["target"],
        slista_ledger=layer_data["ledger"],
        slista_discrete=slista_trace["discrete"].detach().cpu().numpy()[0],
        slista_correction=layer_data["correction"],
        slista_output=layer_data["output"],
        ann_layers=ann_layers.detach().cpu().numpy(),
        ann_output=ann_output.detach().cpu().numpy(),
        distribution_target=distribution_values["target"],
        distribution_ann=distribution_values["ann"],
        distribution_slista_discrete=distribution_values["slista_discrete"],
        distribution_slista_corrected=distribution_values["slista_corrected"],
    )

    if args.sample_index is None:
        print(f"Selection: lowest final NMSE among {args.candidate_count} candidates")
    print(f"Selected sample index: {metadata['sample_index']}")
    actual_s = int(np.count_nonzero(layer_data["target"]))
    print(f"Figure e: s={actual_s} nonzero target coefficients, "
          f"N={layer_data['target'].size}, K={slista.depth}, "
          f"T={slista.time_steps} (checkpoint s={condition['s']})")
    print(f"Figure e NMSE: spikes only={metadata['spike_only_nmse_db']:.2f} dB; "
          f"final={metadata['final_nmse_db']:.2f} dB; "
          f"improvement={metadata['spike_only_nmse_db'] - metadata['final_nmse_db']:.2f} dB")
    for output in outputs:
        print(f"Figure: {output}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
