# -*- coding: utf-8 -*-
import argparse
import glob
import os
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from config import Config
from edge_compression import EdgeSensor
import utils

warnings.filterwarnings("ignore")

VERSION = "PLOT_RESULTS_DVSG_CHANNEL_ENCODING_ABLATION_2026_07_14"

# -----------------------------------------------------------------------------
# Visual identity: match ECG figure where possible
# -----------------------------------------------------------------------------
MODEL_ALIASES: Dict[str, str] = {
    # Proposed / S-LISTA aliases
    "ConvLISTA_Img": "SPIKING LISTA",
    "ConvLISTA_Img_RelaxedAER": "SPIKING LISTA",
    "convlista_img": "SPIKING LISTA",
    "convlista_img_relaxedaer": "SPIKING LISTA",
    "SNN LISTA": "SPIKING LISTA",
    "SNN_LISTA": "SPIKING LISTA",
    "LISTA_SNN": "SPIKING LISTA",
    "Spiking LISTA": "SPIKING LISTA",
    "SPIKING LISTA": "SPIKING LISTA",
    "SpikingLISTA": "SPIKING LISTA",
    "ConvLISTA_Img_LowRate": "SPIKING LISTA LOW-RATE",
    "SPIKING LISTA (<1/2 ANN bits)": "SPIKING LISTA LOW-RATE",
    "SPIKING LISTA (<0.5 ANN bits)": "SPIKING LISTA LOW-RATE",

    # Spiking CNN baseline aliases
    "SpikingCNN": "SPIKING CNN",
    "SpikingCNN_RelaxedAER": "SPIKING CNN",
    "spikingcnn": "SPIKING CNN",
    "SCNN": "SPIKING CNN",
    "Spiking CNN": "SPIKING CNN",
    "SPIKING CNN": "SPIKING CNN",
    "SpikingCNN_LowRate": "SPIKING CNN LOW-RATE",
    "SPIKING CNN (<1/2 ANN bits)": "SPIKING CNN LOW-RATE",
    "SPIKING CNN (<0.5 ANN bits)": "SPIKING CNN LOW-RATE",

    # Dense ANN baselines
    "FISTA": "FISTA",
    "ALISTA": "ANN ALISTA",
    "ANN ALISTA": "ANN ALISTA",
    "LISTA_ANN": "ANN LISTA",
    "lista_ann": "ANN LISTA",
    "ANN LISTA": "ANN LISTA",
    "ANN_ConvLISTA": "ANN LISTA",
    "LAMP": "ANN LSTM",
    "lamp": "ANN LSTM",
    "ANN_LAMP": "ANN LSTM",
    "ANN LAMP": "ANN LSTM",
    "LSTM": "ANN LSTM",
    "lstm": "ANN LSTM",
    "ANN LSTM": "ANN LSTM",

    # Old DVSG CSVs aliases
    "ConvLSTM": "ANN LSTM",
    "convlstm": "ANN LSTM",
}

MODEL_ORDER: List[str] = [
    "FISTA",
    "ANN ALISTA",
    "ANN LSTM",
    "ANN LISTA",
    "SPIKING CNN",
    "SPIKING LISTA",
]

# ECG-style colors from the uploaded ECG plot.
# FISTA blue, ANN ALISTA orange, ANN LSTM green, ANN LISTA red, SNN/SPIKING LISTA purple.
# SPIKING CNN updated to a highly visible cyan for better contrast.
MODEL_COLOR: Dict[str, str] = {
    "FISTA": "#1f77b4",
    "ANN ALISTA": "#ff7f0e",
    "ANN LSTM": "#2ca02c",
    "ANN LISTA": "#d62728",
    "SPIKING LISTA": "#9467bd",
    "SPIKING CNN": "#17becf", 
    "SPIKING LISTA LOW-RATE": "#9467bd",
    "SPIKING CNN LOW-RATE": "#17becf",
}

MODEL_MARKER: Dict[str, str] = {
    "FISTA": "o",
    "ANN ALISTA": "s",
    "ANN LSTM": "^",
    "ANN LISTA": "D",
    "SPIKING LISTA": "v",
    "SPIKING CNN": "s",
    "SPIKING LISTA LOW-RATE": "v",
    "SPIKING CNN LOW-RATE": "s",
}


def is_lowrate_curve(model_name: str) -> bool:
    return "LOW-RATE" in model_name

# Final p-sweep range requested.
P_MIN = 1.0e-2
P_MAX = 1.0e-1
P_TICKS = np.round(np.arange(0.01, 0.1001, 0.01), 3)


def canonical_model_name(raw_name: object) -> str:
    if raw_name is None or pd.isna(raw_name):
        return "Unknown"
    name = str(raw_name).strip()
    if name in MODEL_ALIASES:
        return MODEL_ALIASES[name]
    compact = name.replace(" ", "_")
    if compact in MODEL_ALIASES:
        return MODEL_ALIASES[compact]
    lower = compact.lower()
    if lower in MODEL_ALIASES:
        return MODEL_ALIASES[lower]
    return name


def model_sort_key(name: str) -> Tuple[int, str]:
    return (MODEL_ORDER.index(name) if name in MODEL_ORDER else 999, name)


def apply_publication_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 15.0,         # Increased from 10.5
        "axes.labelsize": 15.0,    # Increased from 11.0
        "axes.titlesize": 15.0,
        "xtick.labelsize": 13.0,   # Increased from 9.5
        "ytick.labelsize": 13.0,   # Increased from 9.5
        "legend.fontsize": 10.0,   # Increased from 9.0
        "axes.linewidth": 1.0,     # Slightly thicker axes lines for clarity
        "lines.linewidth": 1.5,
        "lines.markersize": 5.0,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

def stylize_axis(ax: plt.Axes) -> None:
    ax.grid(True, which="major", linestyle="--", linewidth=0.45, alpha=0.25)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    ax.tick_params(direction="out", length=3.2, width=0.75, pad=1.5)


def save_figure(fig: plt.Figure, png_path: str) -> None:
    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    pdf_path = os.path.splitext(png_path)[0] + ".pdf"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {png_path}")
    print(f"[Saved] {pdf_path}")


def results_dir() -> str:
    return str(getattr(Config, "RESULTS_DIR", os.getcwd()))


def remove_old_duplicate_outputs() -> None:
    root = results_dir()
    old_names = [
        "plot_g2_psweep_baselines.png", "plot_g2_psweep_baselines.pdf",
        "plot_psweep_nmse_acc.png", "plot_psweep_nmse_acc.pdf",
        "plot_rate_sweep_nmse_acc.png", "plot_rate_sweep_nmse_acc.pdf",
    ]
    for name in old_names:
        path = os.path.join(root, name)
        if os.path.exists(path):
            try:
                os.remove(path)
                print(f"[Removed old figure] {path}")
            except OSError:
                pass


def first_existing_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def read_existing_csvs(paths: Sequence[str]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    seen = set()
    for path in paths:
        if path in seen or not os.path.exists(path):
            continue
        seen.add(path)
        try:
            tmp = pd.read_csv(path)
            tmp["__source_csv"] = os.path.basename(path)
            frames.append(tmp)
            print(f"[Read] {path} rows={len(tmp)}")
        except Exception as exc:
            print(f"[Warn] Failed to read {path}: {exc}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def attach_display_model(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["Display Model", "Model", "Base_Model", "Base Model", "model", "Method", "Algorithm", "Name"]:
        if col in df.columns:
            df["Display Model"] = df[col].map(canonical_model_name)
            return df
    df["Display Model"] = "Unknown"
    return df


def normalize_accuracy_series(values: pd.Series) -> pd.Series:
    vals = pd.to_numeric(values, errors="coerce")
    finite = vals[np.isfinite(vals)]
    if len(finite) > 0 and finite.max() <= 1.5:
        vals = vals * 100.0
    return vals


# -----------------------------------------------------------------------------
# Load p-sweep data
# -----------------------------------------------------------------------------
def load_psweep_results() -> pd.DataFrame:
    root = results_dir()
    primary = os.path.join(root, "benchmark_results_bsc_baseline.csv")
    legacy_candidates = [
        os.path.join(root, "bsc_fixed_weight_robustness_results.csv"),
        os.path.join(root, "psweep_results.csv"),
        os.path.join(root, "psweep_main_results.csv"),
    ]
    # main_robust.py writes the four-model main p-sweep. Do not merge older
    # p-sweep CSVs, which may still contain obsolete low-rate curves.
    candidates = [primary] if os.path.exists(primary) else legacy_candidates
    df = read_existing_csvs(candidates)
    if df.empty:
        return df
    df = attach_display_model(df)
    # Backward compatibility: explicitly remove low-rate rows if an older CSV
    # is plotted before the main p-sweep is rerun.
    df = df[~df["Display Model"].map(is_lowrate_curve)].copy()

    if "Experiment" in df.columns:
        exp = df["Experiment"].astype(str)
        mask = exp.str.contains("p", case=False, na=False) | exp.str.contains("robust", case=False, na=False)
        if mask.any():
            df = df[mask].copy()

    p_col = first_existing_col(df, ["BSC_p", "p_eval", "p", "Eval p", "BSC p", "eval_p"])
    nmse_col = first_existing_col(df, ["NMSE (dB)", "NMSE", "nmse", "Val NMSE", "Val NMSE (dB)"])
    acc_col = first_existing_col(df, [
        "Accuracy (%)", "Main Phase4 Acc (%)", "Main Phase4 Accuracy (%)",
        "Fixed Main Acc (%)", "Fixed Main Accuracy (%)", "Fixed main classifier accuracy (%)",
        "Fixed main classifier accuracy", "Fixed-source accuracy (%)", "Fixed Source Acc (%)",
        "Fixed Source Accuracy (%)", "Fixed-source accuracy", "FixedAcc", "FixedAcc (%)",
        "Acc(Img)", "Acc (%)", "accuracy", "Test accuracy", "test_acc",
    ])
    if p_col is None or nmse_col is None:
        print("[Skip] p-sweep CSV exists, but p/NMSE columns are missing.")
        print(f"       columns={list(df.columns)}")
        return pd.DataFrame()

    df["p_plot"] = pd.to_numeric(df[p_col], errors="coerce")
    df["NMSE_plot"] = pd.to_numeric(df[nmse_col], errors="coerce")
    df["ACC_plot"] = normalize_accuracy_series(df[acc_col]) if acc_col is not None else np.nan
    df = df[(df["p_plot"] >= P_MIN - 1e-12) & (df["p_plot"] <= P_MAX + 1e-12)].copy()
    df = df[np.isfinite(df["p_plot"]) & np.isfinite(df["NMSE_plot"])]
    df = df.drop_duplicates(subset=["Display Model", "p_plot"], keep="last")
    if not df.empty:
        print(f"[p-sweep] range={df['p_plot'].min():.4g}-{df['p_plot'].max():.4g}, rows={len(df)}")
    return df


# -----------------------------------------------------------------------------
# Load rate-sweep data
# -----------------------------------------------------------------------------
def load_rate_sweep_results() -> pd.DataFrame:
    root = results_dir()
    candidates = [os.path.join(root, "rate_penalty_results.csv")]
    candidates += sorted(glob.glob(os.path.join(root, "rate_penalty_result_*.csv")))
    df = read_existing_csvs(candidates)
    if df.empty:
        return df
    df = attach_display_model(df)

    bits_col = first_existing_col(df, [
        "Achieved Bits/Sample", "Bits/Sample", "Bits per sample", "bits_per_sample",
        "Avg Bits/Sample", "Average Bits/Sample", "AER Bits/Sample",
    ])
    nmse_col = first_existing_col(df, ["NMSE (dB)", "NMSE", "nmse", "Val NMSE", "Val NMSE (dB)"])
    acc_col = first_existing_col(df, [
        "Main Phase4 Acc (%)", "Main Phase4 Accuracy (%)", "Fixed Main Acc (%)", "Fixed Main Accuracy (%)",
        "Fixed main classifier accuracy (%)", "Fixed main classifier accuracy", "Fixed-source accuracy (%)",
        "Fixed Source Acc (%)", "Fixed Source Accuracy (%)", "Fixed-source accuracy",
        "Accuracy (%)", "Acc(Img)", "Acc (%)", "Image Acc (%)", "Test accuracy", "test_acc", "accuracy",
    ])
    alpha_col = first_existing_col(df, ["Alpha Rate", "alpha_rate", "Alpha", "alpha", "Rate Alpha"])

    if bits_col is None or nmse_col is None:
        print("[Skip] rate-sweep CSV exists, but bits/NMSE columns are missing.")
        print(f"       columns={list(df.columns)}")
        return pd.DataFrame()

    df["Bits_plot"] = pd.to_numeric(df[bits_col], errors="coerce")
    df["NMSE_plot"] = pd.to_numeric(df[nmse_col], errors="coerce")
    df["ACC_plot"] = normalize_accuracy_series(df[acc_col]) if acc_col is not None else np.nan
    df["Alpha_plot"] = pd.to_numeric(df[alpha_col], errors="coerce") if alpha_col is not None else np.nan
    df = df[np.isfinite(df["Bits_plot"]) & np.isfinite(df["NMSE_plot"])].copy()

    # Deduplicate repeated single-point + aggregate CSV rows.
    subset = ["Display Model", "Bits_plot", "NMSE_plot"]
    if "Alpha_plot" in df.columns and df["Alpha_plot"].notna().any():
        subset = ["Display Model", "Alpha_plot"]
    df = df.drop_duplicates(subset=subset, keep="last")

    if not df.empty:
        print(f"[rate-sweep] rows={len(df)}, models={sorted(df['Display Model'].unique(), key=model_sort_key)}")
    return df


# -----------------------------------------------------------------------------
# Compact 1x4 sweep figure
# -----------------------------------------------------------------------------
def plot_sweeps_1x4() -> None:
    ps = load_psweep_results()
    rt = load_rate_sweep_results()
    if ps.empty and rt.empty:
        print("[Skip] No sweep data found.")
        return

    # Adjusted figsize to better match the increased font sizes
    fig, axes = plt.subplots(1, 4, figsize=(14.5, 3.4))
    ax0, ax1, ax2, ax3 = axes

    if not ps.empty:
        for model_name in sorted(ps["Display Model"].unique(), key=model_sort_key):
            sub = ps[ps["Display Model"] == model_name].sort_values("p_plot")
            color = MODEL_COLOR.get(model_name)
            marker = MODEL_MARKER.get(model_name, "o")
            line_style = "-"
            marker_face = color
            ax0.plot(sub["p_plot"], sub["NMSE_plot"], marker=marker, color=color,
                     linestyle=line_style, markerfacecolor=marker_face,
                     markeredgecolor=color, markeredgewidth=0.8,
                     linewidth=1.25, markersize=3.7, label=model_name)
            if sub["ACC_plot"].notna().any():
                ax1.plot(sub["p_plot"], sub["ACC_plot"], marker=marker, color=color,
                         linestyle=line_style, markerfacecolor=marker_face,
                         markeredgecolor=color, markeredgewidth=0.8,
                         linewidth=1.25, markersize=3.7, label=model_name)
        for ax in [ax0, ax1]:
            ax.set_xlim(P_MIN, P_MAX)
            ax.set_xticks([0.01, 0.03, 0.05, 0.07, 0.09])
            ax.set_xticklabels(["0.01", "0.03", "0.05", "0.07", "0.09"])
            ax.set_xlabel("p", labelpad=4)
            stylize_axis(ax)
        ax0.set_ylabel("NMSE (dB)")
        ax1.set_ylabel("Accuracy (%)")
        
        # Brought labels closer to the plot to eliminate extra white space
        ax0.text(0.5, -0.25, "(a)", transform=ax0.transAxes, ha="center", va="top", fontweight="bold", clip_on=False)
        ax1.text(0.5, -0.25, "(b)", transform=ax1.transAxes, ha="center", va="top", fontweight="bold", clip_on=False)
        ax0.legend(
            loc="best", frameon=True, ncol=2, fontsize=7.8,
            handlelength=1.8, columnspacing=0.65, borderpad=0.35,
        )
    else:
        ax0.axis("off")
        ax1.axis("off")

    if not rt.empty:
        for model_name in sorted(rt["Display Model"].unique(), key=model_sort_key):
            sub = rt[rt["Display Model"] == model_name].sort_values("Bits_plot")
            color = MODEL_COLOR.get(model_name)
            marker = MODEL_MARKER.get(model_name, "o")
            x = sub["Bits_plot"] / 10000.0
            ax2.scatter(x, sub["NMSE_plot"], marker=marker, s=28, color=color,
                        edgecolor="black", linewidth=0.25, alpha=0.95, label=model_name, zorder=3)
            if sub["ACC_plot"].notna().any():
                ax3.scatter(x, sub["ACC_plot"], marker=marker, s=28, color=color,
                            edgecolor="black", linewidth=0.25, alpha=0.95, label=model_name, zorder=3)
        for ax in [ax2, ax3]:
            ax.set_xlabel(r"Achieved bits/sample ($\times 10^4$)", labelpad=4)
            stylize_axis(ax)
        ax2.set_ylabel("NMSE (dB)")
        ax3.set_ylabel("Accuracy (%)")
        
        # Brought labels closer to the plot to eliminate extra white space
        ax2.text(0.5, -0.25, "(c)", transform=ax2.transAxes, ha="center", va="top", fontweight="bold", clip_on=False)
        ax3.text(0.5, -0.25, "(d)", transform=ax3.transAxes, ha="center", va="top", fontweight="bold", clip_on=False)
        ax3.legend(loc="best", frameon=True, handlelength=1.2, borderpad=0.35)
    else:
        ax2.axis("off")
        ax3.axis("off")

    save_figure(fig, os.path.join(results_dir(), "plot_sweeps_1x4.png"))


# -----------------------------------------------------------------------------
# Same-measurement channel-encoding ablation
# -----------------------------------------------------------------------------

def load_channel_ablation_results() -> pd.DataFrame:
    path = os.path.join(results_dir(), "channel_encoding_ablation.csv")
    df = read_existing_csvs([path])
    if df.empty:
        return df

    required = [
        "Display Model", "Channel Encoding", "BSC_p", "Measurement BER",
        "Channel Recon Deviation (dB)", "NMSE (dB)", "Accuracy (%)",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        print(f"[Skip] Channel-ablation CSV is missing columns: {missing}")
        return pd.DataFrame()

    df["Display Model"] = df["Display Model"].map(canonical_model_name)
    numeric_cols = [
        "BSC_p", "Measurement BER", "Channel Recon Deviation (dB)",
        "NMSE (dB)", "Accuracy (%)", "AER BER Upper Bound",
        "Dense BER Theory", "AER/Dense Bit Ratio", "Exact Dominance Threshold",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[np.isfinite(df["BSC_p"])].copy()
    df = df.drop_duplicates(
        subset=["Display Model", "Channel Encoding", "BSC_p"], keep="last"
    )
    return df


def plot_channel_encoding_ablation() -> None:
    df = load_channel_ablation_results()
    if df.empty:
        print("[Skip] No same-measurement channel-ablation data found.")
        return

    fig, axes = plt.subplots(1, 4, figsize=(14.5, 3.4))
    metrics = [
        ("Measurement BER", "Measurement BER"),
        ("Channel Recon Deviation (dB)", "Recon. deviation (dB)"),
        ("NMSE (dB)", "NMSE (dB)"),
        ("Accuracy (%)", "Accuracy (%)"),
    ]
    channel_style = {
        "AER-BSC": {"linestyle": "-", "filled": True},
        "Dense-BSC": {"linestyle": "--", "filled": False},
    }

    for ax, (metric, ylabel) in zip(axes, metrics):
        for model_name in sorted(df["Display Model"].unique(), key=model_sort_key):
            for encoding in ["AER-BSC", "Dense-BSC"]:
                sub = df[
                    (df["Display Model"] == model_name)
                    & (df["Channel Encoding"] == encoding)
                ].sort_values("BSC_p")
                if sub.empty:
                    continue

                color = MODEL_COLOR.get(model_name, "#333333")
                marker = MODEL_MARKER.get(model_name, "o")
                style = channel_style[encoding]
                ax.plot(
                    sub["BSC_p"], sub[metric],
                    color=color,
                    linestyle=style["linestyle"],
                    marker=marker,
                    markerfacecolor=color if style["filled"] else "none",
                    markeredgecolor=color,
                    markeredgewidth=0.8,
                    linewidth=1.3,
                    markersize=3.8,
                    label=f"{model_name}, {encoding.replace('-BSC', '')}",
                )

        ax.set_xlim(P_MIN, P_MAX)
        ax.set_xticks([0.01, 0.03, 0.05, 0.07, 0.09])
        ax.set_xticklabels(["0.01", "0.03", "0.05", "0.07", "0.09"])
        ax.set_xlabel("p", labelpad=4)
        ax.set_ylabel(ylabel)
        stylize_axis(ax)

    for idx, ax in enumerate(axes):
        ax.text(
            0.5, -0.25, f"({chr(ord('a') + idx)})",
            transform=ax.transAxes, ha="center", va="top",
            fontweight="bold", clip_on=False,
        )

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(
            handles, labels, loc="best", frameon=True, fontsize=7.4,
            handlelength=1.8, borderpad=0.35,
        )

    save_figure(
        fig,
        os.path.join(results_dir(), "plot_channel_encoding_ablation_1x4.png"),
    )


# -----------------------------------------------------------------------------
# Transmitter-side power-law figure
# -----------------------------------------------------------------------------
def alpha_tag(alpha: float) -> str:
    return f"{alpha:.6g}".replace(".", "p").replace("-", "m")


def load_val_loader_for_powerlaw():
    try:
        _, val_loader = utils.get_dataloaders(Config.BATCH_SIZE_CLS)
    except TypeError:
        _, val_loader = utils.get_dataloaders()
    return val_loader


def sensor_checkpoint_candidates(powerlaw_alpha: float, prefer_main: bool) -> List[str]:
    seed = int(getattr(Config, "SEED", 42))
    rho = getattr(Config, "TX_RHO", 0.1)
    th = float(getattr(Config, "INIT_THRESHOLD", 0.10))
    th_tag = f"{th:.2f}"
    a_tag = alpha_tag(powerlaw_alpha)
    main_names = [
        f"dynamic_recon_spiking_convlista_img_relaxedaer_seed{seed}_rho{rho}_th{th_tag}.pth",
        f"dynamic_recon_spiking_convlista_img_relaxedaer_seed{seed}_th{th_tag}.pth",
    ]
    ratesweep_names = [
        f"dynamic_recon_spiking_convlista_img_relaxedaer_ratesweep_alpha{a_tag}_seed{seed}_rho{rho}_th{th_tag}.pth",
        f"dynamic_recon_spiking_convlista_img_relaxedaer_ratesweep_alpha0p0006_seed{seed}_rho{rho}_th{th_tag}.pth",
    ]
    return main_names + ratesweep_names if prefer_main else ratesweep_names + main_names


def load_tx_sensor(powerlaw_alpha: float, prefer_main_ckpt: bool, explicit_ckpt: Optional[str]) -> Tuple[Optional[EdgeSensor], Optional[str]]:
    device = getattr(Config, "DEVICE", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    Config.DEVICE = device
    sensor = EdgeSensor().to(device)

    candidate_paths: List[str] = []
    if explicit_ckpt:
        candidate_paths.append(explicit_ckpt if os.path.isabs(explicit_ckpt) else os.path.join(results_dir(), explicit_ckpt))
    candidate_paths.extend(os.path.join(results_dir(), name) for name in sensor_checkpoint_candidates(powerlaw_alpha, prefer_main_ckpt))

    for path in candidate_paths:
        if not os.path.exists(path):
            continue
        try:
            ckpt = torch.load(path, map_location=device, weights_only=False)
            if not isinstance(ckpt, dict) or "sensor" not in ckpt:
                continue
            state = ckpt["sensor"]
            state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
            sensor.load_state_dict(state, strict=False)
            sensor.eval()
            print(f"[Power-law] loaded transmitter sensor from: {path}")
            return sensor, path
        except Exception as exc:
            print(f"[Warn] Failed to load sensor from {path}: {exc}")
    return None, None


def collect_transmitter_activity(max_batches: int, powerlaw_alpha: float, prefer_main_ckpt: bool, explicit_ckpt: Optional[str]) -> Tuple[Optional[np.ndarray], Dict[str, float], Optional[str]]:
    sensor, ckpt_path = load_tx_sensor(powerlaw_alpha, prefer_main_ckpt, explicit_ckpt)
    if sensor is None:
        print("[Skip] No valid transmitter sensor checkpoint found for power-law plot.")
        return None, {}, None

    val_loader = load_val_loader_for_powerlaw()
    device = Config.DEVICE
    tx_counts: Optional[torch.Tensor] = None
    total_sample_frames = 0
    total_events = 0.0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= max_batches:
                break
            x_batch = batch[0].to(device)
            B, T = int(x_batch.shape[0]), int(x_batch.shape[1])
            state = None
            for t in range(T):
                x_t = x_batch[:, t]
                _, state, aux = sensor(x_t, state=state, mode="snn")
                if aux is None or "tx_hard" not in aux:
                    continue
                tx_hard = aux["tx_hard"].detach().float().view(B, -1)
                if tx_counts is None:
                    tx_counts = torch.zeros(tx_hard.shape[1], device=device)
                tx_counts += tx_hard.sum(dim=0)
                total_events += float(tx_hard.sum().item())
                total_sample_frames += B

    if tx_counts is None or total_sample_frames <= 0:
        return None, {}, ckpt_path

    avg_prob = (tx_counts / float(total_sample_frames)).detach().cpu().numpy()
    sorted_prob = np.sort(avg_prob)[::-1]
    positive_prob = sorted_prob[sorted_prob > 1e-10]
    stats: Dict[str, float] = {
        "total_aer_addresses": float(len(sorted_prob)),
        "active_aer_addresses": float(len(positive_prob)),
        "active_address_fraction": float(len(positive_prob) / max(1, len(sorted_prob))),
        "mean_emission_probability": float(np.mean(sorted_prob)),
        "max_emission_probability": float(np.max(sorted_prob)),
        "avg_events_per_frame": float(total_events / max(1, total_sample_frames)),
        "avg_bits_per_frame": float(total_events / max(1, total_sample_frames) * Config.RATE_BITS_PER_EVENT),
    }
    if len(positive_prob) > 0 and np.sum(positive_prob) > 0:
        top10 = max(1, int(0.10 * len(positive_prob)))
        top01 = max(1, int(0.01 * len(positive_prob)))
        stats["top_10_percent_traffic_share"] = float(np.sum(positive_prob[:top10]) / np.sum(positive_prob))
        stats["top_1_percent_traffic_share"] = float(np.sum(positive_prob[:top01]) / np.sum(positive_prob))
    return positive_prob, stats, ckpt_path


def plot_power_law(max_batches: int, powerlaw_alpha: float, prefer_main_ckpt: bool, explicit_ckpt: Optional[str]) -> None:
    print("[Power-law] transmitter-side AER tx_hard activity; scatter-only measured points.")
    positive_prob, stats, ckpt_path = collect_transmitter_activity(max_batches, powerlaw_alpha, prefer_main_ckpt, explicit_ckpt)
    if positive_prob is None or len(positive_prob) < 5:
        print("[Skip] Not enough transmitter activity to draw power-law plot.")
        return

    ranks = np.arange(1, len(positive_prob) + 1)
    fig, ax = plt.subplots(1, 1, figsize=(5.0, 3.75))

    ax.scatter(
        ranks, positive_prob,
        s=13,
        marker="x",
        color=MODEL_COLOR["SPIKING LISTA"],
        linewidth=0.65,
        alpha=0.82,
        label="SPIKING LISTA",
        zorder=3,
    )

    slope: Optional[float] = None
    if len(positive_prob) >= 12:
        log_x = np.log10(ranks)
        log_y = np.log10(positive_prob)
        fit_end = max(12, int(0.60 * len(log_x)))
        fit_end = min(fit_end, len(log_x))
        coeff = np.polyfit(log_x[:fit_end], log_y[:fit_end], 1)
        slope = float(coeff[0])
        fit_y = 10 ** np.polyval(coeff, log_x)
        exponent = abs(slope)
        ax.plot(
            ranks, fit_y,
            linestyle="--",
            linewidth=1.6,
            color=MODEL_COLOR["ANN ALISTA"],
            label=rf"$y \propto N^{{-{exponent:.2f}}}$",
            zorder=2,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Index of AER address")
    ax.set_ylabel("Average emission probability")
    y_min = max(float(np.min(positive_prob)) * 0.7, 1e-8)
    ax.set_ylim(y_min, 1.0)
    stylize_axis(ax)
    ax.legend(frameon=True, loc="best")
    save_figure(fig, os.path.join(results_dir(), "plot_power_law_distribution.png"))

    stats_path = os.path.join(results_dir(), "table_tx_power_law_stats.txt")
    with open(stats_path, "w") as f:
        f.write("Transmitter-side AER tx_hard activity distribution\n")
        f.write(f"version: {VERSION}\n")
        f.write(f"loaded_checkpoint: {ckpt_path}\n")
        for k, v in stats.items():
            f.write(f"{k}: {v:.8f}\n")
        if slope is not None:
            f.write(f"log_log_fit_slope: {slope:.8f}\n")
    print(f"[Saved] {stats_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Final DVSG plotting script: sweeps, channel ablation, and transmitter power-law.")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--tx-rho", type=float, default=None)
    parser.add_argument("--max-batches-powerlaw", type=int, default=10)
    parser.add_argument("--powerlaw-alpha", type=float, default=6e-4)
    parser.add_argument("--powerlaw-ckpt", type=str, default=None)
    parser.add_argument("--prefer-main-ckpt", action="store_true")
    parser.add_argument(
        "--figures",
        nargs="+",
        default=["sweeps", "channel", "powerlaw"],
        choices=["sweeps", "channel", "powerlaw"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    Config.setup(args)
    if args.results_dir is not None:
        Config.RESULTS_DIR = args.results_dir
    Config.make_dir()

    print(f"[Version] {VERSION}")
    print(f"[Results dir] {results_dir()}")
    apply_publication_style()
    remove_old_duplicate_outputs()

    if "sweeps" in args.figures:
        plot_sweeps_1x4()
    if "channel" in args.figures:
        plot_channel_encoding_ablation()
    if "powerlaw" in args.figures:
        plot_power_law(
            max_batches=int(args.max_batches_powerlaw),
            powerlaw_alpha=float(args.powerlaw_alpha),
            prefer_main_ckpt=bool(args.prefer_main_ckpt),
            explicit_ckpt=args.powerlaw_ckpt,
        )


if __name__ == "__main__":
    main()
