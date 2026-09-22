#!/usr/bin/env python3
import argparse
import csv
import json
import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import torch

from config import Config, PROJECT_DIR, DEFAULT_OUTPUT
from edge_compression import EdgeSensor
from wireless_channel import WirelessChannel
from snn_models import ALISTA, LAMP, HybridLISTA
from trainer import Trainer
import utils


# Shared typography for all panels, including the secondary y-axis.
FONT_AXIS = 16.8
FONT_TICK = 15.6
FONT_LEGEND = 14.0
FONT_ANNOTATION = 14.0
FONT_TITLE = 18.6
FONT_PANEL = 27.5

TARGETS = ["alista", "lamp", "lista_ann", "lista_snn_1"]
LABELS = {
    "alista": "ALISTA",
    "lamp": "LAMP",
    "lista_ann": "ANN LISTA",
    "lista_snn_1": "S-LISTA",
}
COLORS = {
    "alista": "#3a9d73",
    "lamp": "#7a5aa6",
    "lista_ann": "#3b83bd",
    "lista_snn_1": "#d84a3a",
}
MARKERS = {
    "alista": "o",
    "lamp": "D",
    "lista_ann": "o",
    "lista_snn_1": "s",
}
MODEL_SPECS = {
    "alista": (ALISTA, "ann"),
    "lamp": (LAMP, "ann"),
    "lista_ann": (HybridLISTA, "ann"),
    "lista_snn_1": (HybridLISTA, "spiking"),
}
SNR_GRID = [-5.0, -2.0, 0.0, 2.0, 5.0, 7.5, 10.0, 12.5, 15.0, 20.0]


def configure(args):
    Config.M = 78
    Config.TOTAL_MEAS_TARGET = 78
    Config.EFFECTIVE_TOTAL_MEAS = 78
    Config.NUM_LAYERS = 8
    Config.FISTA_ITERATIONS = 8
    Config.TIME_STEPS = 1
    Config.QUANT_BITS = 8
    Config.BITS_PER_SAMPLE = Config.M * Config.QUANT_BITS
    Config.TRAIN_SNR_DB = float(args.train_snr_db)
    Config.SNR_DB = float(args.train_snr_db)
    Config.CS_MODE = "learnable_dictionary"
    Config.EVAL_CHANNEL_MODE = "bit_awgn_hard"
    Config.TRAIN_CHANNEL_MODE = "bit_awgn_hard"
    if args.data_dir is not None:
        Config.DATA_DIR = str(args.data_dir)
    if torch.cuda.is_available():
        Config.DEVICE = torch.device(f"cuda:{args.gpu}")
    else:
        Config.DEVICE = torch.device("cpu")


def checkpoint_path(row):
    path = Path(row["Checkpoint"])
    if not path.is_absolute():
        path = Path(row["Trial_Dir"]) / path
    return path.resolve()


def read_best_rows(path):
    rows = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["Target"] in TARGETS:
                trial = Path(row["Trial_Dir"])
                if not trial.is_absolute():
                    row["Trial_Dir"] = str((path.parent / trial).resolve())
                rows[row["Target"]] = row
    return rows


def read_final_slista(results_dir):
    path = results_dir / "results.csv"
    frame = pd.read_csv(path)
    frame = frame[frame["Model"] == "Spiking LISTA"]
    row = frame.loc[frame["Val_NMSE_dB"].idxmin()]
    return {
        "Target": "lista_snn_1",
        "Checkpoint": str(row["checkpoint"]),
        "Trial_Dir": str(results_dir),
        "Val_NMSE_dB": float(row["Val_NMSE_dB"]),
        "Test_NMSE_dB": float(row["Test_NMSE_dB"]),
    }


def build_objects(best_rows, train_loader, eval_loader):
    objects = {}
    for target in TARGETS:
        model_class, mode = MODEL_SPECS[target]
        sensor = EdgeSensor().to(Config.DEVICE)
        model = model_class(phi=sensor.effective_phi().detach()).to(Config.DEVICE)
        if hasattr(model, "set_mode"):
            model.set_mode(mode)
        checkpoint = torch.load(
            checkpoint_path(best_rows[target]),
            map_location=Config.DEVICE,
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model"], strict=True)
        sensor.load_state_dict(checkpoint["sensor"], strict=True)
        channel = WirelessChannel(snr_db=Config.TRAIN_SNR_DB).to(Config.DEVICE)
        trainer = Trainer(
            model, sensor, channel, train_loader, eval_loader,
            custom_cfg=None,
        )
        model.eval()
        sensor.eval()
        objects[target] = {
            "model": model,
            "sensor": sensor,
            "channel": channel,
            "trainer": trainer,
            "mode": mode,
        }
    return objects


def limited_batches(loader, max_batches):
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        yield batch


@torch.no_grad()
def infer_object(obj, loader, snr_db, max_batches=None):
    obj["channel"].snr_db = float(snr_db)
    model = obj["model"]
    sensor = obj["sensor"]
    if hasattr(model, "set_mode"):
        model.set_mode(obj["mode"])
    targets = []
    estimates = []
    with utils.deterministic_eval(Config.EVAL_CHANNEL_SEED):
        for batch in limited_batches(loader, max_batches):
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            x = x.to(Config.DEVICE)
            y, _, _ = sensor(x, mode="shared")
            y_recv, _ = obj["channel"](
                y, mode="shared", channel_mode=Config.EVAL_CHANNEL_MODE
            )
            x_hat, _, _, _ = model(
                y_recv, states=None, return_debug=False,
                phi_override=sensor.effective_phi(),
            )
            targets.append(x.detach().cpu())
            estimates.append(x_hat.detach().cpu())
    return torch.cat(targets), torch.cat(estimates)


def aggregate_nmse(target, estimate):
    error = (estimate - target).square().sum().clamp_min(1e-12)
    energy = target.square().sum().clamp_min(1e-12)
    return float((10.0 * torch.log10(error / energy)).item())


def representative_indices(target, reconstructions, count=3):
    target_energy = target.square().sum(dim=1).clamp_min(1e-12)
    scores = []
    for target_name in TARGETS:
        estimate = reconstructions[target_name]
        sample_nmse = 10.0 * torch.log10(
            (estimate - target).square().sum(dim=1).clamp_min(1e-12)
            / target_energy
        )
        scores.append((sample_nmse - aggregate_nmse(target, estimate)).abs())
    ranking = torch.argsort(torch.stack(scores).mean(dim=0)).tolist()
    selected = []
    normalized = target / target.norm(dim=1, keepdim=True).clamp_min(1e-12)
    for index in ranking:
        if all(
            abs(float(torch.dot(normalized[index], normalized[old]))) < 0.985
            for old in selected
        ):
            selected.append(index)
        if len(selected) == count:
            break
    if len(selected) < count:
        for index in ranking:
            if index not in selected:
                selected.append(index)
            if len(selected) == count:
                break
    return selected


@torch.no_grad()
def firing_statistics(slista_obj, loader, max_batches=None):
    model = slista_obj["model"]
    sensor = slista_obj["sensor"]
    channel = slista_obj["channel"]
    model.set_mode("spiking")
    channel.snr_db = Config.TRAIN_SNR_DB
    positive = torch.zeros(Config.NUM_LAYERS, Config.N, dtype=torch.float64)
    negative = torch.zeros_like(positive)
    net_layer_atom = torch.zeros_like(positive)
    drive_ratio_sum = torch.zeros(Config.NUM_LAYERS, dtype=torch.float64)
    sample_count = 0

    with utils.deterministic_eval(Config.EVAL_CHANNEL_SEED):
        for batch in limited_batches(loader, max_batches):
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            x = x.to(Config.DEVICE)
            y, _, _ = sensor(x, mode="shared")
            y_recv, _ = channel(
                y, mode="shared", channel_mode=Config.EVAL_CHANNEL_MODE
            )
            drive0 = model.P_snn(y_recv)
            z = torch.zeros_like(drive0)
            for stage in range(Config.NUM_LAYERS):
                drive = (
                    drive0 if stage == 0
                    else drive0 - model.PD_snn_k[stage - 1](z)
                )
                theta = getattr(model, f"theta_snn_{stage}")
                spike_pos = (drive >= theta).to(drive.dtype)
                spike_neg = (-drive >= theta).to(drive.dtype)
                signed_spike = spike_pos - spike_neg
                positive[stage] += spike_pos.sum(dim=0).cpu().double()
                negative[stage] += spike_neg.sum(dim=0).cpu().double()
                z = z + signed_spike
                net_layer_atom[stage] += z.abs().sum(dim=0).cpu().double()
                drive_ratio_sum[stage] += (
                    drive.abs() / theta.clamp_min(1e-8)
                ).sum().cpu().double()
            sample_count += x.shape[0]

    positive /= max(sample_count, 1)
    negative /= max(sample_count, 1)
    atom_rate = (positive + negative).mean(dim=0).numpy()
    layer_positive = positive.mean(dim=1).numpy()
    layer_negative = negative.mean(dim=1).numpy()
    layer_atom_rate = (positive + negative).numpy()
    layer_numbers = torch.arange(
        1, Config.NUM_LAYERS + 1, dtype=torch.float64
    ).unsqueeze(1)
    net_layer_atom /= max(sample_count, 1) * layer_numbers
    net_layer_atom_rate = net_layer_atom.numpy()
    layer_drive_ratio = (
        drive_ratio_sum / max(sample_count * Config.N, 1)
    ).numpy()
    return (
        atom_rate, layer_positive, layer_negative,
        layer_atom_rate, net_layer_atom_rate, layer_drive_ratio,
    )


def compute_cache(objects, eval_loader, args):
    cache_tag = "full" if args.max_batches is None else f"draft_{args.max_batches}b"
    train_snr_tag = f"{Config.TRAIN_SNR_DB:g}".replace("-", "m").replace(".", "p")
    cache_file = args.cache_dir / f"evaluation_cache_{cache_tag}_snr{train_snr_tag}_awgn_v1.pt"
    robust_file = args.cache_dir / f"awgn_robustness_{cache_tag}_snr{train_snr_tag}_v1.csv"
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    if cache_file.exists() and not args.force_recompute:
        cache = torch.load(cache_file, map_location="cpu", weights_only=False)
    else:
        reconstructions = {}
        target = None
        energy = {}
        operation_counts = {}
        for target_name in TARGETS:
            current_target, estimate = infer_object(
                objects[target_name], eval_loader, Config.TRAIN_SNR_DB, args.max_batches
            )
            if target is None:
                target = current_target
            reconstructions[target_name] = estimate
            result = objects[target_name]["trainer"].evaluate(
                mode=objects[target_name]["mode"], loader=eval_loader
            )
            energy[target_name] = result["energy_uj"]
            operation_counts[target_name] = {
                "mac": result["dense_macs"],
                "ac": result["synaptic_acs"],
            }

        (
            atom_rate, layer_positive, layer_negative,
            layer_atom_rate, net_layer_atom_rate, layer_drive_ratio,
        ) = firing_statistics(objects["lista_snn_1"], eval_loader, args.max_batches)
        cache = {
            "target": target,
            "reconstructions": reconstructions,
            "energy": energy,
            "operation_counts": operation_counts,
            "atom_rate": atom_rate,
            "layer_positive": layer_positive,
            "layer_negative": layer_negative,
            "layer_atom_rate": layer_atom_rate,
            "net_layer_atom_rate": net_layer_atom_rate,
            "layer_drive_ratio": layer_drive_ratio,
        }
        torch.save(cache, cache_file)

    if robust_file.exists() and not args.force_recompute:
        robustness = pd.read_csv(robust_file)
    else:
        rows = []
        for target_name in TARGETS:
            for snr_db in SNR_GRID:
                target, estimate = infer_object(
                    objects[target_name], eval_loader, snr_db, args.max_batches
                )
                rows.append({
                    "Target": target_name,
                    "eval_snr_db": snr_db,
                    "NMSE_dB": aggregate_nmse(target, estimate),
                })
                print(
                    f"[Robustness] {target_name} | snr_db={snr_db:g} | "
                    f"NMSE={rows[-1]['NMSE_dB']:.2f} dB"
                )
        robustness = pd.DataFrame(rows)
        robustness.to_csv(robust_file, index=False)
    return cache, robustness


def best_values(path):
    values = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            target = row.get("Target", "")
            if target in TARGETS and row.get("Test_NMSE_dB", "") != "":
                values[target] = float(row["Test_NMSE_dB"])
    return values


def quantization_values(path):
    frame = pd.read_csv(path)
    frame = frame[frame["Status"].isin(["ok", "cached"])]
    return frame


def unify_legend_fonts(fig):
    legends = [ax.get_legend() for ax in fig.axes if ax.get_legend() is not None]
    legends.extend(fig.legends)
    sizes = []
    for legend in legends:
        for label in legend.get_texts():
            label.set_fontsize(FONT_LEGEND)
            label.set_fontweight("normal")
            label.set_fontfamily(plt.rcParams["font.family"])
            sizes.append(label.get_fontsize())
    if not sizes or any(abs(size - FONT_LEGEND) > 1e-6 for size in sizes):
        raise RuntimeError("Legend font verification failed")


def publication_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        # The ECG canvas is 14.8 in wide versus 8.35 in for the toy figure.
        # These sizes therefore produce the same type size after both figures
        # are scaled to the same manuscript width.
        "font.size": FONT_AXIS,
        "axes.labelsize": FONT_AXIS,
        "axes.titlesize": FONT_TITLE,
        "xtick.labelsize": FONT_TICK,
        "ytick.labelsize": FONT_TICK,
        "legend.fontsize": FONT_LEGEND,
        "axes.linewidth": 1.0,
        "lines.linewidth": 1.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def panel_label(axis, label, top_row=False):
    axis.text(
        -0.17 if top_row else -0.13,
        1.18 if top_row else 1.13,
        label, transform=axis.transAxes,
        fontsize=FONT_PANEL, fontweight="bold", va="top", ha="left",
        clip_on=False,
    )


def plot_composite(cache, robustness, fixed_nmse, learned_nmse, quant, args, rayleigh):
    publication_style()
    fig = plt.figure(figsize=(14.8, 13.4))
    outer = gridspec.GridSpec(
        3, 1, figure=fig, height_ratios=[0.75, 1.0, 1.0], hspace=0.24,
    )
    top_grid = outer[0].subgridspec(1, 4, wspace=0.32)
    middle_grid = outer[1].subgridspec(1, 3, wspace=0.55)
    bottom_grid = outer[2].subgridspec(1, 3, wspace=0.55)
    axes = [fig.add_subplot(top_grid[0, column]) for column in range(4)]
    axes += [fig.add_subplot(middle_grid[0, column]) for column in range(3)]
    axes += [fig.add_subplot(bottom_grid[0, column]) for column in range(3)]
    for axis in axes[4:]:
        axis.set_box_aspect(1.0)

    target = cache["target"]
    reconstructions = cache["reconstructions"]
    indices = representative_indices(target, reconstructions, count=3)
    reference = target[indices].numpy().reshape(-1)
    common_min = min(float(reference.min()), *(float(reconstructions[t][indices].min()) for t in TARGETS))
    common_max = max(float(reference.max()), *(float(reconstructions[t][indices].max()) for t in TARGETS))

    for index, target_name in enumerate(TARGETS):
        axis = axes[index]
        estimate = reconstructions[target_name][indices].numpy().reshape(-1)
        selected_nmse = aggregate_nmse(
            target[indices], reconstructions[target_name][indices]
        )
        axis.plot(reference, color="#222222", linewidth=1.35, label="Target")
        axis.plot(
            estimate, color=COLORS[target_name], linewidth=1.15,
            alpha=0.95, label="Estimate",
        )
        axis.axvline(256, color="#999999", linestyle=":", linewidth=0.8)
        axis.axvline(512, color="#999999", linestyle=":", linewidth=0.8)
        axis.set_title(LABELS[target_name], fontweight="bold", pad=3)
        axis.text(
            0.03, 0.06, f"NMSE = {selected_nmse:.1f} dB",
            transform=axis.transAxes, fontsize=FONT_ANNOTATION,
        )
        axis.set_xlim(0, len(reference) - 1)
        amplitude_span = common_max - common_min
        axis.set_ylim(
            common_min - 0.18 * amplitude_span,
            common_max + 0.05 * amplitude_span,
        )
        axis.set_xlabel("Sample index")
        if index == 0:
            axis.set_ylabel("Amplitude")
            axis.legend(frameon=False, loc="upper left", fontsize=FONT_LEGEND)
        axis.grid(True, alpha=0.20)

    axis = axes[8]
    comparison_x = np.arange(2)
    comparison_width = 0.19
    comparison_offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * comparison_width
    comparison_values = []
    for target_name, offset in zip(TARGETS, comparison_offsets):
        values = [fixed_nmse[target_name], learned_nmse[target_name]]
        comparison_values.extend(values)
        comparison_values.extend([values[0]-cache.get("fixed_std",{}).get(target_name,0),values[1]-cache.get("learned_std",{}).get(target_name,0)])
        axis.bar(
            comparison_x + offset, values, width=comparison_width,
            yerr=[cache.get("fixed_std", {}).get(target_name, 0), cache.get("learned_std", {}).get(target_name, 0)], capsize=3,
            color=COLORS[target_name], edgecolor="black", linewidth=0.5,
            label=LABELS[target_name],
        )
    axis.set_xticks(
        comparison_x,
        ["Wavelet\ndictionary", "Learned\ndictionary"],
    )
    axis.set_ylabel("NMSE (dB)")
    axis.axhline(0, color="black", linewidth=0.8)
    # Keep every negative-NMSE bar inside the panel.  Leave about 5 dB below
    # the best bar, round down to a 5-dB tick, and retain at least [-30, 0].
    comparison_lower = min(
        -35.0,
        5.0 * np.floor((min(comparison_values) - 5.0) / 5.0),
    )
    axis.set_ylim(comparison_lower, 0)
    axis.grid(axis="y", linestyle="--", alpha=0.35)
    # Algorithm legend shared by h, i and j is placed inside h below.

    axis = axes[5]
    for target_name in TARGETS:
        subset = robustness[robustness["Target"] == target_name].sort_values("eval_snr_db")
        if "std" in subset:
            axis.fill_between(subset["eval_snr_db"].to_numpy(), (subset["NMSE_dB"]-subset["std"]).to_numpy(), (subset["NMSE_dB"]+subset["std"]).to_numpy(), color=COLORS[target_name], alpha=.18, linewidth=0)
        axis.plot(
            subset["eval_snr_db"], subset["NMSE_dB"], marker=MARKERS[target_name],
            markersize=4, color=COLORS[target_name], label=LABELS[target_name],
        )
    axis.axvline(Config.TRAIN_SNR_DB, color="#555555", linestyle=":", linewidth=1.3)
    axis.set_xlabel("SNR (dB)")
    axis.set_ylabel("NMSE (dB)")
    axis.set_xticks([-5, 0, 5, 10, 15, 20])
    axis.grid(True, which="both", linestyle="--", alpha=0.3)
    axis.legend(frameon=False, ncol=1, fontsize=FONT_LEGEND, loc="upper right")

    # Same visual encoding as panel f; channel names identify the two scans.
    axes[5].text(0.04, 0.04, "AWGN", transform=axes[5].transAxes,
                 fontsize=FONT_ANNOTATION, ha="left", va="bottom")
    axes[5].set_xlabel(r"$E_b/N_0$ (dB)")
    axis = axes[6]
    for target_name in TARGETS:
        subset = rayleigh[rayleigh["target"] == target_name].sort_values("snr_db")
        if "std" in subset:
            axis.fill_between(subset["snr_db"].to_numpy(), (subset["nmse_db"]-subset["std"]).to_numpy(), (subset["nmse_db"]+subset["std"]).to_numpy(), color=COLORS[target_name], alpha=.18, linewidth=0)
        axis.plot(
            subset["snr_db"], subset["nmse_db"], marker=MARKERS[target_name],
            markersize=4, color=COLORS[target_name], label=LABELS[target_name],
        )
    axis.text(0.04, 0.04, "Rayleigh", transform=axis.transAxes,
              fontsize=FONT_ANNOTATION, ha="left", va="bottom")
    axis.set_xlabel(r"Average $E_b/N_0$ (dB)")
    axis.set_ylabel("NMSE (dB)")
    axis.set_xticks([-5, 0, 5, 10, 15, 20])
    axis.grid(True, which="both", linestyle="--", alpha=0.3)
    # Panel g shares the algorithm legend in f.
    # Common limits make AWGN and Rayleigh directly comparable.
    scan_values = np.concatenate([
        (robustness["NMSE_dB"]-robustness.get("std",0)).to_numpy(),
        (robustness["NMSE_dB"]+robustness.get("std",0)).to_numpy(),
        (rayleigh["nmse_db"]-rayleigh.get("std",0)).to_numpy(),
        (rayleigh["nmse_db"]+rayleigh.get("std",0)).to_numpy(),
        robustness["NMSE_dB"].to_numpy(dtype=float),
        rayleigh["nmse_db"].to_numpy(dtype=float),
    ])
    lower = 5.0 * np.floor((scan_values.min() - 1.0) / 5.0)
    upper = 5.0 * np.ceil((scan_values.max() + 1.0) / 5.0)
    for scan_axis in (axes[5], axes[6]):
        scan_axis.set_xlim(-5.5, 20.5)
        scan_axis.set_ylim(lower, upper)

    axis = axes[7]
    bit_values = [2, 4, 6, 8]
    group_x = np.arange(len(bit_values))
    bar_width = 0.19
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * bar_width
    all_quant_nmse = []
    for target_name, offset in zip(TARGETS, offsets):
        subset = quant[quant["Target"] == target_name]
        mapping = {
            int(row["Quant_Bits"]): float(row["Test_NMSE_dB"])
            for _, row in subset.iterrows()
        }
        values = [mapping.get(bits, np.nan) for bits in bit_values]
        errors = [dict(zip(subset["Quant_Bits"].astype(int),subset.get("std",np.zeros(len(subset))))).get(b,0) for b in bit_values]
        all_quant_nmse.extend(v-e for v,e in zip(values,errors))
        all_quant_nmse.extend(value for value in values if np.isfinite(value))
        axis.bar(
            group_x + offset, values, width=bar_width,
            yerr=[dict(zip(subset["Quant_Bits"].astype(int), subset.get("std", np.zeros(len(subset))))).get(bits, 0) for bits in bit_values], capsize=3,
            color=COLORS[target_name], edgecolor="black", linewidth=0.5,
            label=LABELS[target_name],
        )
    axis.set_xticks(group_x, [str(78 * int(bits)) for bits in bit_values])
    axis.set_xlabel("Bit/sample")
    axis.set_ylabel("NMSE (dB)")
    axis.axhline(0, color="black", linewidth=0.8)
    if all_quant_nmse:
        lower = min(all_quant_nmse)
        upper = max(all_quant_nmse)
        axis.set_ylim(lower - 1.0, upper + 1.5)
    axis.grid(axis="y", linestyle="--", alpha=0.3)

    shared_legend = axes[7].legend(
        frameon=False, ncol=2, fontsize=FONT_LEGEND, loc="lower center",
        handlelength=1.1, handletextpad=0.4, columnspacing=0.8,
        borderaxespad=0.5, labelspacing=0.35,
    )

    axis = axes[4]
    layer_index = np.arange(1, Config.NUM_LAYERS + 1)
    positive = np.asarray(cache["layer_positive"], dtype=float)
    negative = np.asarray(cache["layer_negative"], dtype=float)
    axis.bar(layer_index, positive, yerr=cache.get("layer_positive_std"), capsize=3, color="#D49A3A", label="Positive")
    axis.bar(
        layer_index, negative, bottom=positive,
        color="#4C78A8", label="Negative",
        yerr=cache.get("layer_total_std"), capsize=3,
    )
    axis.set_xticks(layer_index)
    axis.set_xlabel("Unfolded layer")
    axis.set_ylabel("Firing probability")
    total_firing = positive + negative
    axis.set_ylim(0, (total_firing + np.asarray(cache.get("layer_total_std",0))).max() * 1.35)
    axis.grid(axis="y", linestyle="--", alpha=0.3)
    drive_axis = axis.twinx()
    drive_line = drive_axis.plot(
        layer_index, np.asarray(cache["layer_drive_ratio"], dtype=float),
        color="#222222", marker="o", linewidth=1.25,
        label="Normalized drive",
    )
    if "layer_drive_ratio_std" in cache:
        sd=np.asarray(cache["layer_drive_ratio_std"]); mean=np.asarray(cache["layer_drive_ratio"])
        drive_axis.fill_between(layer_index,mean-sd,mean+sd,color="#222222",alpha=.15)
    drive_axis.set_ylabel(r"Normalized drive $|u_\ell|/\theta$")
    drive_axis.set_ylim(0, 2.0)
    drive_axis.set_yticks([0, 0.5, 1.0, 1.5, 2.0])
    drive_values = np.asarray(cache["layer_drive_ratio"], dtype=float)
    drive_padding = max(0.10, 0.30 * (drive_values.max() - drive_values.min()))
    drive_axis.set_ylim(
        0, 2.0
    )
    spike_handles, spike_labels = axis.get_legend_handles_labels()
    axis.legend(
        spike_handles + drive_line,
        spike_labels + [drive_line[0].get_label()],
        frameon=False, loc="upper left", fontsize=FONT_LEGEND,
    )

    axis = axes[9]
    energy_values = np.array([
        float(cache["energy"][target_name]) for target_name in TARGETS
    ])
    operation_x = np.arange(len(TARGETS))
    bars = axis.bar(
        operation_x, energy_values,
        yerr=[cache.get("energy_std", {}).get(t,0) for t in TARGETS], capsize=3,
        color=[COLORS[target_name] for target_name in TARGETS],
        edgecolor="black", linewidth=0.6,
    )
    axis.set_xticks(
        operation_x,
        [LABELS[target_name] for target_name in TARGETS],
        rotation=20, ha="right",
    )
    axis.set_ylabel(r"Energy ($\mu$J/sample)")
    axis.set_ylim(0, max(energy_values[i]+cache.get("energy_std",{}).get(t,0) for i,t in enumerate(TARGETS)) * 1.22)
    axis.grid(axis="y", linestyle="--", alpha=0.3)
    for bar, value in zip(bars, energy_values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + cache.get("energy_std",{}).get(TARGETS[list(bars).index(bar)],0) + 0.025 * energy_values.max(),
            f"{value:.2f}", ha="center", va="bottom", fontsize=FONT_ANNOTATION,
        )

    # Align NMSE scales within each plot type and row.
    for group_axes, values in (
        ((axes[5], axes[6]), scan_values),
        ((axes[7], axes[8]), np.concatenate([
            np.asarray(comparison_values), np.asarray(all_quant_nmse),
        ])),
    ):
        group_min = 5.0 * np.floor((values.min() - 1.0) / 5.0)
        group_max = max(0.0, 5.0 * np.ceil(values.max() / 5.0))
        if group_axes[0] is axes[7]:
            # Preserve room for the dictionary-comparison legend below bars.
            group_min = min(-35.0, group_min)
        group_ticks = np.arange(group_min, group_max + 0.1, 5.0)
        for current_axis in group_axes:
            current_axis.set_ylim(group_min, group_max)
            current_axis.set_yticks(group_ticks)
    # All waveform panels use the same tick locations and limits.
    waveform_ticks = axes[0].get_yticks()
    waveform_limits = axes[0].get_ylim()
    for waveform_axis in axes[:4]:
        waveform_axis.set_yticks(waveform_ticks)
        waveform_axis.set_ylim(waveform_limits)

    unify_legend_fonts(fig)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.978, bottom=0.050)

    # A twinned y-axis can alter the final axes box even when GridSpec cells
    # are identical. Reapply one common square size and baseline per row after
    # layout so the exported panels are geometrically aligned.
    fig.canvas.draw()

    def align_square_row(row_axes, twins=None):
        twins = twins or {}
        positions = [current.get_position() for current in row_axes]
        side = float(np.median([
            min(position.width, position.height) for position in positions
        ]))
        baseline = float(np.median([position.y0 for position in positions]))
        for current, position in zip(row_axes, positions):
            aligned = [
                position.x0 + 0.5 * (position.width - side),
                baseline,
                side,
                side,
            ]
            current.set_position(aligned)
            if current in twins:
                twins[current].set_position(aligned)

    align_square_row(axes[4:7], twins={axes[4]: drive_axis})
    align_square_row(axes[7:10])
    fig.canvas.draw()

    # Verify the shared legend stays fully inside h after final layout.
    renderer = fig.canvas.get_renderer()
    legend_box = shared_legend.get_window_extent(renderer)
    panel_box = axes[7].get_window_extent(renderer)
    if not (panel_box.x0 <= legend_box.x0 and legend_box.x1 <= panel_box.x1
            and panel_box.y0 <= legend_box.y0 and legend_box.y1 <= panel_box.y1):
        raise RuntimeError("Shared algorithm legend exceeds panel h bounds")

    label_rows = [
        (axes[0:4], "abcd"),
        (axes[4:7], "efg"),
        (axes[7:10], "hij"),
    ]
    for row_axes, labels in label_rows:
        row_top = max(current.get_position().y1 for current in row_axes)
        label_y = row_top + 0.012
        for current, label in zip(row_axes, labels):
            position = current.get_position()
            fig.text(
                position.x0 - 0.030, label_y, label,
                fontsize=FONT_PANEL, fontweight="bold",
                ha="left", va="bottom",
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = args.output_dir / "Fig_ECG.pdf"
    png_path = args.output_dir / "Fig_ECG.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")


def read_rayleigh_results(path):
    if not path.is_file():
        raise FileNotFoundError(
            f"Rayleigh results not found: {path}. Run eval_ecg_rayleigh_snr.py "
            "first, or supply --rayleigh-results-csv."
        )
    frame = pd.read_csv(path)
    required = {"target", "snr_db", "nmse_db"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{path}: expected columns {sorted(required)}")
    frame = frame[frame["target"].isin(TARGETS)].copy()
    for column in ("snr_db", "nmse_db"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if not np.isfinite(frame[["snr_db", "nmse_db"]].to_numpy()).all():
        raise ValueError(f"{path}: non-finite SNR or NMSE")
    frame = frame[frame["snr_db"].between(-5, 20)]
    for target_name in TARGETS:
        subset = frame[frame["target"] == target_name]
        if len(subset) < 2 or subset["snr_db"].duplicated().any():
            raise ValueError(f"{path}: missing or duplicate SNR results for {target_name}")
        if subset["snr_db"].min() != -5 or subset["snr_db"].max() != 20:
            raise ValueError(f"{path}: incomplete -5 to 20 dB scan for {target_name}")
    return frame


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--rayleigh-results-csv", type=Path,
        default=Path(PROJECT_DIR / "outputs/rayleigh/snr_scan.csv"),
        help="Existing Rayleigh scan produced by eval_ecg_rayleigh_snr.py",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=Config.NUM_WORKERS)
    parser.add_argument("--train-snr-db", type=float, default=5.0)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument(
        "--warm-best-csv", type=Path,
        default=Path(PROJECT_DIR / "outputs/learned/models.csv"),
    )
    parser.add_argument(
        "--slista-results-dir", type=Path,
        default=Path(PROJECT_DIR / "outputs/final_slista"),
    )
    parser.add_argument(
        "--fixed-best-csv", type=Path,
        default=Path(PROJECT_DIR / "outputs/fixed/models.csv"),
    )
    parser.add_argument(
        "--quant-results-csv", type=Path,
        default=Path(PROJECT_DIR / "outputs/quantization/results.csv"),
    )
    parser.add_argument(
        "--cache-dir", type=Path,
        default=Path(PROJECT_DIR / "outputs/cache"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(PROJECT_DIR / "outputs/figures"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = args.output.expanduser().resolve()
    for field, relative in {
        "warm_best_csv": "learned/models.csv", "fixed_best_csv": "fixed/models.csv",
        "slista_results_dir": "final_slista", "quant_results_csv": "quantization/results.csv",
        "cache_dir": "cache", "output_dir": "figures", "rayleigh_results_csv": "rayleigh/snr_scan.csv"
    }.items():
        if "--" + field.replace("_", "-") not in [v.split("=")[0] for v in __import__("sys").argv[1:]]:
            setattr(args, field, root / relative)

    print("ECG composite: e firing; f AWGN; g Rayleigh; h quantization; i dictionary; j energy")
    rayleigh = read_rayleigh_results(args.rayleigh_results_csv)
    configure(args)
    Config.EVAL_CHANNEL_SEED = Config.TEST_CHANNEL_SEED
    Config.NUM_WORKERS = args.num_workers
    utils.set_seed(Config.SEED)
    train_loader, eval_loader = utils.get_dataloaders(eval_split="test")
    warm_rows = read_best_rows(args.warm_best_csv)
    final_slista = read_final_slista(args.slista_results_dir)
    warm_rows["lista_snn_1"] = final_slista
    identity = {
        "checkpoints": {key: hashlib.sha256(checkpoint_path(row).read_bytes()).hexdigest()
                        for key, row in warm_rows.items()},
        "data": hashlib.sha256((Path(Config.DATA_DIR) / Config.TEST_FILE).read_bytes()).hexdigest(),
        "snr": args.train_snr_db, "max_batches": args.max_batches,
        "batch_size": Config.BATCH_SIZE, "channel_seed": Config.EVAL_CHANNEL_SEED,
    }
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    cache_identity = args.cache_dir / "sources.json"
    if not cache_identity.exists() or json.loads(cache_identity.read_text()) != identity:
        args.force_recompute = True
    objects = build_objects(warm_rows, train_loader, eval_loader)
    cache, robustness = compute_cache(objects, eval_loader, args)
    cache_identity.write_text(json.dumps(identity, indent=2) + "\n")
    fixed_nmse = best_values(args.fixed_best_csv)
    learned_nmse = best_values(args.warm_best_csv)
    learned_nmse["lista_snn_1"] = float(final_slista["Test_NMSE_dB"])
    quant = quantization_values(args.quant_results_csv)
    quant.loc[
        (quant["Target"] == "lista_snn_1")
        & (quant["Quant_Bits"].astype(int) == 8),
        "Test_NMSE_dB",
    ] = float(final_slista["Test_NMSE_dB"])
    utils.atomic_torch_save(dict(cache=cache, robustness=robustness, fixed_nmse=fixed_nmse,
        learned_nmse=learned_nmse, quant=quant, rayleigh=rayleigh, identity=identity), root / 'plot_data.pt')
    plot_composite(cache, robustness, fixed_nmse, learned_nmse, quant, args, rayleigh)


if __name__ == "__main__":
    main()
