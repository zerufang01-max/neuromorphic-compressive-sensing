#!/usr/bin/env python3
"""Assemble the DVS-G and SHD results (Figs. 4 and 5, Tables 1-3).

A read-only script: it never loads checkpoints or trains; it only reads the
CSV/JSON files written by the scripts in shd/ and dvsg/.

Outputs (under --output-dir):
  figures/Fig4.{pdf,png}       4 x 3 panels, a-f DVS-G, g-l SHD
  figures/Fig5.{pdf,png}       CDFs a-d, task supervision e-h, class-wise NMSE i,j
  tables/Table1_main.tex       Rayleigh 20 dB, full-vector transmission
  tables/Table2_ablation.tex   block AER, Rayleigh 20 dB (no temperature row)
  tables/Table3_energy.tex     energy, AER models for the SNNs
  tables/SuppTable_int8.tex    INT8 post-training quantisation
  values/paper_values.md       every number the Results text needs
  values/*.csv                 the same numbers as tables
  sources.csv                  every file read (path, rows, modification time)

Typical use (from the repository root, after the experiments in shd/ and dvsg/):
  python assemble_event_results.py --output-dir results_event
or with the result folders elsewhere:
  python assemble_event_results.py --shd-root <shd folder> --dvsg-root <dvsg folder>

LaTeX tables need \\usepackage{booktabs} (and graphicx for \\resizebox).
"""

import argparse
import json
import logging
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
DEFAULT_SHD_ROOT = Path(__file__).resolve().parent / "shd"
DEFAULT_DVSG_ROOT = Path(__file__).resolve().parent / "dvsg"

DATASETS = ["DVS-G", "SHD"]
MODELS = ["S-LISTA", "Vanilla SNN", "ANN LISTA", "ANN LSTM"]
SNN_MODELS = MODELS[:2]
SEEDS = [42, 43, 44, 45, 46]
ANALYSIS_SEED = 42
SNR_GRID = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 7.5, 10, 15, 20]
TRAIN_SNR = 10.0
ABLATION_ORDER = [
    "Without rate regularization",
    "Without task supervision",
    "S-LISTA with AER",
]
DROPPED_ABLATION = "Without temperature relaxation"

# Same colours and markers as the synthetic and ECG figures.
MODEL_COLORS = {
    "S-LISTA": "#D94B3D",
    "Vanilla SNN": "#3BA275",
    "ANN LISTA": "#3E86BB",
    "ANN LSTM": "#7C5AAC",
}
MODEL_MARKERS = {"S-LISTA": "o", "Vanilla SNN": "s", "ANN LISTA": "^", "ANN LSTM": "D"}

NAME_MAP = {
    "SPIKING LISTA": "S-LISTA", "Spiking LISTA": "S-LISTA", "slista": "S-LISTA",
    "SPIKING CNN": "Vanilla SNN", "Spiking CNN": "Vanilla SNN", "SCNN": "Vanilla SNN",
    "Generic SNN": "Vanilla SNN", "SPIKING DENSE": "Vanilla SNN",
    "SpikingDenseRecon": "Vanilla SNN", "vanilla": "Vanilla SNN", "scnn": "Vanilla SNN",
    "ANN-LISTA": "ANN LISTA", "ann_lista": "ANN LISTA",
    "ANN-LSTM": "ANN LSTM", "ANN-ConvLSTM": "ANN LSTM", "ANN ConvLSTM": "ANN LSTM",
    "ann_lstm": "ANN LSTM",
}
KEY2MODEL = {
    "slista_dense": "S-LISTA", "slista_aer": "S-LISTA",
    "vanilla_dense": "Vanilla SNN", "vanilla_aer": "Vanilla SNN",
    "scnn_dense": "Vanilla SNN", "scnn_aer": "Vanilla SNN",
    "ann_lista": "ANN LISTA", "ann_lstm": "ANN LSTM",
}
ENERGY_KEYS = {
    "DVS-G": ["slista_aer", "vanilla_aer", "ann_lista", "ann_lstm"],
    "SHD": ["slista_aer", "scnn_aer", "ann_lista", "ann_lstm"],
}
DVSG_DENSE = ["slista_dense", "vanilla_dense", "ann_lista", "ann_lstm"]
VARIANT_MAP = {
    "Proposed": "S-LISTA with AER",
    "S-LISTA with AER": "S-LISTA with AER",
    "No rate regularization": "Without rate regularization",
    "Without rate regularization": "Without rate regularization",
    "No semantic supervision": "Without task supervision",
    "Without task supervision": "Without task supervision",
    "Without semantic supervision": "Without task supervision",
    "Hard surrogate": DROPPED_ABLATION,
    DROPPED_ABLATION: DROPPED_ABLATION,
}

# Reported values. Used only for a non-fatal consistency check
# (catches wrong files or wrong filters). (kind, dataset, item, metric): value.
DOC_VALUES = {
    ("table1", "DVS-G", "S-LISTA", "NMSE"): -13.19, ("table1", "DVS-G", "S-LISTA", "Acc"): 94.44,
    ("table1", "DVS-G", "Vanilla SNN", "NMSE"): -10.08, ("table1", "DVS-G", "Vanilla SNN", "Acc"): 90.97,
    ("table1", "DVS-G", "ANN LISTA", "NMSE"): -14.70, ("table1", "DVS-G", "ANN LISTA", "Acc"): 94.79,
    ("table1", "DVS-G", "ANN LSTM", "NMSE"): -15.04, ("table1", "DVS-G", "ANN LSTM", "Acc"): 95.07,
    ("table1", "SHD", "S-LISTA", "NMSE"): -16.22, ("table1", "SHD", "S-LISTA", "Acc"): 92.60,
    ("table1", "SHD", "Vanilla SNN", "NMSE"): -12.17, ("table1", "SHD", "Vanilla SNN", "Acc"): 91.73,
    ("table1", "SHD", "ANN LISTA", "NMSE"): -16.45, ("table1", "SHD", "ANN LISTA", "Acc"): 95.25,
    ("table1", "SHD", "ANN LSTM", "NMSE"): -16.19, ("table1", "SHD", "ANN LSTM", "Acc"): 95.19,
    ("robust", "DVS-G", "S-LISTA", "NMSE"): -9.37, ("robust", "DVS-G", "S-LISTA", "Acc"): 66.54,
    ("robust", "DVS-G", "ANN LSTM", "NMSE"): -8.34, ("robust", "DVS-G", "ANN LSTM", "Acc"): 48.49,
    ("robust", "DVS-G", "ANN LISTA", "NMSE"): -4.63, ("robust", "DVS-G", "ANN LISTA", "Acc"): 38.08,
    ("robust", "DVS-G", "Vanilla SNN", "NMSE"): -4.44, ("robust", "DVS-G", "Vanilla SNN", "Acc"): 36.26,
    ("robust", "SHD", "S-LISTA", "NMSE"): -13.16, ("robust", "SHD", "S-LISTA", "Acc"): 80.38,
    ("robust", "SHD", "ANN LSTM", "NMSE"): -11.00, ("robust", "SHD", "ANN LSTM", "Acc"): 77.49,
    ("robust", "SHD", "ANN LISTA", "NMSE"): -10.68, ("robust", "SHD", "ANN LISTA", "Acc"): 75.76,
    ("robust", "SHD", "Vanilla SNN", "NMSE"): -3.36, ("robust", "SHD", "Vanilla SNN", "Acc"): 73.14,
    ("table2", "DVS-G", "Without rate regularization", "NMSE"): -12.28,
    ("table2", "DVS-G", "Without rate regularization", "Acc"): 91.81,
    ("table2", "DVS-G", "Without rate regularization", "Bits"): 117330,
    ("table2", "DVS-G", "Without task supervision", "NMSE"): -11.50,
    ("table2", "DVS-G", "Without task supervision", "Acc"): 91.67,
    ("table2", "DVS-G", "Without task supervision", "Bits"): 15768,
    ("table2", "DVS-G", "S-LISTA with AER", "NMSE"): -11.58,
    ("table2", "DVS-G", "S-LISTA with AER", "Acc"): 92.15,
    ("table2", "DVS-G", "S-LISTA with AER", "Bits"): 17396,
    ("table2", "SHD", "Without rate regularization", "NMSE"): -14.99,
    ("table2", "SHD", "Without rate regularization", "Acc"): 92.00,
    ("table2", "SHD", "Without rate regularization", "Bits"): 6998,
    ("table2", "SHD", "Without task supervision", "NMSE"): -14.04,
    ("table2", "SHD", "Without task supervision", "Acc"): 88.01,
    ("table2", "SHD", "Without task supervision", "Bits"): 1263,
    ("table2", "SHD", "S-LISTA with AER", "NMSE"): -14.08,
    ("table2", "SHD", "S-LISTA with AER", "Acc"): 90.12,
    ("table2", "SHD", "S-LISTA with AER", "Bits"): 1994,
    ("energy", "DVS-G", "S-LISTA", "total"): 1151.3, ("energy", "DVS-G", "ANN LISTA", "total"): 48782.0,
    ("energy", "DVS-G", "S-LISTA", "recon_total"): 841.3,
    ("energy", "SHD", "S-LISTA", "total"): 11.39, ("energy", "SHD", "ANN LISTA", "total"): 267.33,
    ("energy", "SHD", "S-LISTA", "recon_total"): 9.10,
}

SOURCES = []
STYLE_SCALE = 1.0


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def read_csv(path, label):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"[{label}] missing file: {path}")
    frame = pd.read_csv(path)
    SOURCES.append({"label": label, "file": str(path.resolve()), "rows": len(frame),
                    "modified": pd.Timestamp(os.path.getmtime(path), unit="s").isoformat()})
    print(f"[read] {label}: {path} ({len(frame)} rows)")
    return frame


def pick(frame, candidates, label):
    for name in candidates:
        if name in frame.columns:
            return name
    lowered = {c.lower(): c for c in frame.columns}
    for name in candidates:
        if name.lower() in lowered:
            return lowered[name.lower()]
    raise KeyError(f"[{label}] none of the columns {candidates} found; columns are {list(frame.columns)}")


def model_names(series):
    return series.astype(str).str.strip().replace(NAME_MAP)


def as_bool(series):
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])


def mean_std(values):
    values = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(float)
    if values.size == 0:
        return np.nan, np.nan, 0
    std = float(values.std(ddof=1)) if values.size > 1 else 0.0
    return float(values.mean()), std, int(values.size)


def summarize(frame, groups, metrics):
    rows = []
    for keys, part in frame.groupby(groups, sort=False, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(groups, keys))
        for metric in metrics:
            m, s, n = mean_std(part[metric])
            row[f"{metric}_mean"], row[f"{metric}_std"], row[f"{metric}_n"] = m, s, n
        rows.append(row)
    return pd.DataFrame(rows)


def require_unique(frame, keys, label):
    dup = frame.duplicated(keys, keep=False)
    if dup.any():
        raise ValueError(f"[{label}] duplicated rows for keys {keys}:\n{frame.loc[dup, keys].head(10)}")


def require_seeds(frame, groups, label, seeds=SEEDS):
    for keys, part in frame.groupby(groups, sort=False):
        found = sorted(set(part["Seed"].astype(int)))
        if found != list(seeds):
            raise ValueError(f"[{label}] {keys}: expected seeds {seeds}, found {found}")


def order_models(frame, column="Model"):
    frame = frame.copy()
    frame["_o"] = frame[column].map({m: i for i, m in enumerate(MODELS)}).fillna(99)
    return frame.sort_values("_o", kind="stable").drop(columns="_o")


# ----------------------------------------------------------------------------
# Loading: every loader returns the same canonical columns for both datasets.
#   scans:    Dataset Model Seed Channel SNR NMSE Acc
#   semantic: Dataset Variant(With/Without) Seed Channel SNR NMSE Acc
#   main:     Dataset Model Seed NMSE Acc Bits           (Rayleigh 20 dB, dense)
#   main_awgn10: same at AWGN 10 dB
#   budget:   Dataset Model Seed Channels Bits NMSE Acc   (AWGN 10 dB, dense)
#   ablation: Dataset Setting Seed NMSE Acc Bits          (Rayleigh 20 dB, AER)
#   samples:  Dataset Model Sample Class NMSE             (Rayleigh 20 dB, seed 42)
#   aer_bits: Dataset Model Sample Class Bits             (AER models, seed 42)
#   classes:  Dataset Model Class NMSE
#   energy:   Dataset Model Key Transmission Bits {sender,recon}_{mac,ac,compute,
#             weight,state,total} total decoder
#   quant:    Dataset Model Key Transmission float_nmse int8_nmse float_acc int8_acc
# ----------------------------------------------------------------------------
def shd_standard(frame, label):
    out = pd.DataFrame()
    out["Model"] = model_names(frame[pick(frame, ["Display Model", "Model"], label)])
    out["Seed"] = pd.to_numeric(frame[pick(frame, ["Seed"], label)]).astype(int)
    for target, names in [("SNR", ["Eval SNR (dB)"]), ("NMSE", ["Mean NMSE (dB)", "NMSE (dB)"]),
                          ("Acc", ["Accuracy (%)"]), ("Bits", ["Bits/Sample", "Avg Bits/Sample"])]:
        try:
            out[target] = pd.to_numeric(frame[pick(frame, names, label)], errors="coerce")
        except KeyError:
            pass
    return out


def shd_scan(path, channel, label):
    raw = read_csv(path, label)
    if "Evaluation Channel" in raw.columns:
        chan = raw["Evaluation Channel"].astype(str).str.strip().str.lower()
        if not chan.eq(channel).all():
            raise ValueError(f"[{label}] expected only {channel} rows")
    frame = shd_standard(raw, label)
    frame["Channel"] = channel
    return frame


def load_shd(root):
    o = root / "outputs"
    d = {}
    scans = pd.concat([
        shd_scan(o / "snr_robustness_per_seed.csv", "awgn", "SHD scan AWGN"),
        shd_scan(o / "snr_robustness_rayleigh_per_seed.csv", "rayleigh", "SHD scan Rayleigh"),
    ], ignore_index=True)
    d["scans"] = scans[["Model", "Seed", "Channel", "SNR", "NMSE", "Acc"]]

    main = shd_standard(read_csv(o / "main_rayleigh20_per_seed.csv", "SHD Table 1"), "SHD Table 1")
    d["main"] = main[["Model", "Seed", "NMSE", "Acc", "Bits"]]
    awgn = shd_standard(read_csv(o / "main_results_per_seed.csv", "SHD AWGN 10 dB"), "SHD AWGN 10 dB")
    if "SNR" in awgn.columns:
        awgn = awgn[np.isclose(awgn["SNR"], TRAIN_SNR)]
    d["main_awgn10"] = awgn[["Model", "Seed", "NMSE", "Acc", "Bits"]]

    label = "SHD bit budget"
    raw = read_csv(root / "outputs_budget" / "budget_per_seed.csv", label)
    raw = raw[raw[pick(raw, ["Curve"], label)].astype(str).str.lower().eq("dense")]
    d["budget"] = pd.DataFrame({
        "Model": model_names(raw[pick(raw, ["Display Model", "Model", "Method"], label)]),
        "Seed": pd.to_numeric(raw[pick(raw, ["Seed"], label)]).astype(int),
        "Channels": pd.to_numeric(raw[pick(raw, ["Channels"], label)]).astype(int),
        "Bits": pd.to_numeric(raw[pick(raw, ["Bits/Sample", "Avg Bits/Sample"], label)]),
        "NMSE": pd.to_numeric(raw[pick(raw, ["AWGN NMSE @10"], label)]),
        "Acc": pd.to_numeric(raw[pick(raw, ["AWGN Acc @10"], label)]),
    })

    label = "SHD Table 2"
    raw = read_csv(o / "rayleigh_paper" / "core_ablation_per_seed.csv", label)
    abl = shd_standard(raw.assign(Model="S-LISTA"), label)
    abl["Setting"] = raw[pick(raw, ["Variant"], label)].astype(str).str.strip().map(VARIANT_MAP)
    if abl["Setting"].isna().any():
        raise ValueError(f"[{label}] unknown variant names: {sorted(set(raw['Variant']))}")
    d["ablation"] = abl.loc[abl["Setting"] != DROPPED_ABLATION, ["Setting", "Seed", "NMSE", "Acc", "Bits"]]

    sem = []
    for path, channel in [(o / "semantic_snr_robustness_per_seed.csv", "awgn"),
                          (o / "rayleigh_paper" / "semantic_snr_robustness_per_seed.csv", "rayleigh")]:
        label = f"SHD task supervision {channel}"
        raw = read_csv(path, label)
        part = shd_standard(raw.assign(Model="S-LISTA"), label)
        variant = raw[pick(raw, ["Variant"], label)].astype(str).str.lower()
        part["Variant"] = np.where(variant.str.startswith("without"), "Without", "With")
        part["Channel"] = channel
        sem.append(part)
    d["semantic"] = pd.concat(sem, ignore_index=True)[["Variant", "Seed", "Channel", "SNR", "NMSE", "Acc"]]

    rp = o / "rayleigh_paper"
    label = "SHD Fig 5c samples"
    raw = read_csv(rp / f"analysis_samples_seed{ANALYSIS_SEED}.csv", label)
    d["samples"] = pd.DataFrame({
        "Model": model_names(raw[pick(raw, ["Display Model", "Model"], label)]),
        "Sample": raw[pick(raw, ["Sample"], label)], "Class": raw[pick(raw, ["Class"], label)],
        "NMSE": pd.to_numeric(raw[pick(raw, ["NMSE (dB)"], label)])})
    label = "SHD Fig 5d AER bits"
    raw = read_csv(rp / f"analysis_aer_bits_seed{ANALYSIS_SEED}.csv", label)
    d["aer_bits"] = pd.DataFrame({
        "Model": model_names(raw[pick(raw, ["Display Model", "Model"], label)]),
        "Sample": raw[pick(raw, ["Sample"], label)], "Class": raw[pick(raw, ["Class"], label)],
        "Bits": pd.to_numeric(raw[pick(raw, ["Bits/Sample"], label)])})
    label = "SHD Fig 5j classes"
    raw = read_csv(rp / f"analysis_classes_seed{ANALYSIS_SEED}.csv", label)
    d["classes"] = pd.DataFrame({
        "Model": model_names(raw[pick(raw, ["Display Model", "Model"], label)]),
        "Class": pd.to_numeric(raw[pick(raw, ["Class"], label)]).astype(int),
        "NMSE": pd.to_numeric(raw[pick(raw, ["NMSE (dB)"], label)])})

    d["energy"] = load_shd_energy(o / "energy" / f"energy_table_aer_seed{ANALYSIS_SEED}.csv")
    d["quant"] = load_shd_quant(o / "energy" / f"quant_int8_seed{ANALYSIS_SEED}.csv")
    return d


def shd_key_column(raw, label):
    return raw[pick(raw, ["Experiment Key", "Experiment", "Key"], label)].astype(str).str.strip()


def load_shd_energy(path):
    label = "SHD Table 3"
    raw = read_csv(path, label)
    keys = shd_key_column(raw, label)
    raw = raw[keys.isin(ENERGY_KEYS["SHD"])].copy()
    keys = keys[raw.index]
    out = pd.DataFrame({"Key": keys.to_numpy()})
    out["Model"] = out["Key"].map(KEY2MODEL)
    for scope, prefix in [("sender", "sender"), ("recon", "reconstruction")]:
        for item, names in [("mac", ["MAC (M)", "MAC (MOP)"]), ("ac", ["AC (M)", "AC (MOP)"]),
                            ("compute", ["compute (uJ)"]), ("weight", ["weight (uJ)", "weights (uJ)"]),
                            ("state", ["state (uJ)"]), ("total", ["total (uJ)"])]:
            col = pick(raw, [f"{prefix} {n}" for n in names], label)
            out[f"{scope}_{item}"] = pd.to_numeric(raw[col]).to_numpy()
    out["total"] = pd.to_numeric(raw[pick(raw, ["total (uJ)"], label)]).to_numpy()
    out["Bits"] = pd.to_numeric(raw[pick(raw, ["Bits/Sample"], label)]).to_numpy()
    try:
        out["Transmission"] = raw[pick(raw, ["Transmission"], label)].astype(str).to_numpy()
    except KeyError:
        out["Transmission"] = np.where(out["Model"].isin(SNN_MODELS), "AER", "8-bit dense")
    out["decoder"] = np.nan  # SHD reads out through a linear dictionary.
    return out


def load_shd_quant(path):
    label = "SHD INT8"
    raw = read_csv(path, label)
    keys = shd_key_column(raw, label)
    raw = raw[keys.isin(ENERGY_KEYS["SHD"])]
    keys = keys[raw.index]
    return pd.DataFrame({
        "Key": keys.to_numpy(), "Model": keys.map(KEY2MODEL).to_numpy(),
        "float_nmse": pd.to_numeric(raw[pick(raw, ["float NMSE (dB)"], label)]).to_numpy(),
        "int8_nmse": pd.to_numeric(raw[pick(raw, ["network NMSE (dB)"], label)]).to_numpy(),
        "float_acc": pd.to_numeric(raw[pick(raw, ["float Accuracy (%)"], label)]).to_numpy(),
        "int8_acc": pd.to_numeric(raw[pick(raw, ["network Accuracy (%)"], label)]).to_numpy(),
    })


def dvsg_clean(frame):
    return frame.copy()


def load_dvsg(root):
    o = root / "outputs"
    t = o / "tables"
    d = {}
    raw = dvsg_clean(read_csv(t / "snr_scans_per_seed.csv", "DVS-G scans"))
    raw["channel"] = raw["channel"].astype(str).str.lower()
    base = pd.DataFrame({
        "Experiment": raw["experiment"].astype(str), "Seed": raw["seed"].astype(int),
        "Channel": raw["channel"], "SNR": pd.to_numeric(raw["snr_db"]),
        "NMSE": pd.to_numeric(raw["nmse"]), "Acc": pd.to_numeric(raw["acc"]),
        "Bits": pd.to_numeric(raw["bits"]) if "bits" in raw else np.nan,
    })
    dense = base[base["Experiment"].isin(DVSG_DENSE)].copy()
    dense["Model"] = dense["Experiment"].map(KEY2MODEL)
    d["scans"] = dense[["Model", "Seed", "Channel", "SNR", "NMSE", "Acc"]]
    pick_point = lambda ch, snr: dense[dense["Channel"].eq(ch) & np.isclose(dense["SNR"], snr)]
    d["main"] = pick_point("rayleigh", 20)[["Model", "Seed", "NMSE", "Acc", "Bits"]]
    d["main_awgn10"] = pick_point("awgn", TRAIN_SNR)[["Model", "Seed", "NMSE", "Acc", "Bits"]]

    sem = base[base["Experiment"].isin(["slista_dense", "slista_dense_no_semantic"])].copy()
    sem["Variant"] = np.where(sem["Experiment"].eq("slista_dense"), "With", "Without")
    d["semantic"] = sem[["Variant", "Seed", "Channel", "SNR", "NMSE", "Acc"]]

    raw = read_csv(t / "budget_per_seed.csv", "DVS-G bit budget")
    raw = raw[raw["channel"].astype(str).str.lower().eq("awgn") & np.isclose(raw["snr_db"], TRAIN_SNR)]
    d["budget"] = pd.DataFrame({
        "Model": model_names(raw["budget_method"]), "Seed": raw["seed"].astype(int),
        "Channels": raw["channels"].astype(int), "Bits": pd.to_numeric(raw["bits"]),
        "NMSE": pd.to_numeric(raw["nmse"]), "Acc": pd.to_numeric(raw["acc"])})

    rows = []
    names = {"slista_aer": "S-LISTA with AER", "slista_aer_no_rate": "Without rate regularization",
             "slista_aer_no_semantic": "Without task supervision"}
    for key, setting in names.items():
        for seed in SEEDS:
            ev = read_csv(o / "runs" / key / f"seed{seed}" / "eval.csv", f"DVS-G Table 2 {key} s{seed}")
            ev = ev[ev["channel"].astype(str).str.lower().eq("rayleigh") & np.isclose(ev["snr_db"], 20)]
            if len(ev) != 1:
                raise ValueError(f"[DVS-G Table 2] {key} seed {seed}: expected one Rayleigh 20 dB row")
            r = ev.iloc[0]
            rows.append({"Setting": setting, "Seed": seed, "NMSE": float(r["nmse"]),
                         "Acc": float(r["acc"]), "Bits": float(r["bits"])})
    d["ablation"] = pd.DataFrame(rows)

    a = o / "analysis"

    def keep_dense(frame):
        if "experiment" in frame.columns:
            return frame[frame["experiment"].astype(str).isin(DVSG_DENSE)]
        return frame

    raw = keep_dense(read_csv(a / f"samples_rayleigh20_seed{ANALYSIS_SEED}.csv", "DVS-G Fig 5a samples"))
    d["samples"] = pd.DataFrame({"Model": model_names(raw["method"]), "Sample": raw["sample"],
                                 "Class": raw["class"], "NMSE": pd.to_numeric(raw["nmse_db"])})
    raw = read_csv(a / f"aer_bits_seed{ANALYSIS_SEED}.csv", "DVS-G Fig 5b AER bits")
    if "experiment" in raw.columns:
        raw = raw[raw["experiment"].astype(str).isin(["slista_aer", "vanilla_aer"])]
    d["aer_bits"] = pd.DataFrame({"Model": model_names(raw["method"]), "Sample": raw["sample"],
                                  "Class": raw["class"], "Bits": pd.to_numeric(raw["aer_bits"])})
    raw = keep_dense(read_csv(a / f"classes_rayleigh20_seed{ANALYSIS_SEED}.csv", "DVS-G Fig 5i classes"))
    d["classes"] = pd.DataFrame({"Model": model_names(raw["method"]), "Class": raw["class"].astype(int),
                                 "NMSE": pd.to_numeric(raw["nmse_db"])})

    d["energy"] = load_dvsg_energy(o / "energy")
    label = "DVS-G INT8"
    raw = read_csv(o / "energy" / f"quant_seed{ANALYSIS_SEED}.csv", label)
    raw = raw[raw["experiment"].astype(str).isin(ENERGY_KEYS["DVS-G"])]
    d["quant"] = pd.DataFrame({
        "Key": raw["experiment"].astype(str).to_numpy(),
        "Model": raw["experiment"].astype(str).map(KEY2MODEL).to_numpy(),
        "float_nmse": raw["float_nmse"].to_numpy(float), "int8_nmse": raw["int8_nmse"].to_numpy(float),
        "float_acc": raw["float_acc"].to_numpy(float), "int8_acc": raw["int8_acc"].to_numpy(float)})

    d["robust_json"] = load_dvsg_robust_json(o / "runs")
    return d


def load_dvsg_energy(folder):
    label = "DVS-G Table 3"
    raw = read_csv(folder / f"table3_seed{ANALYSIS_SEED}.csv", label)
    raw = raw[as_bool(raw["in_table3"])].copy()
    comp = read_csv(folder / f"components_seed{ANALYSIS_SEED}.csv", "DVS-G Table 3 MAC/AC")
    comp["kind"] = comp["kind"].astype(str).str.lower()
    comp["scope"] = comp["scope"].astype(str).str.lower()
    ops = comp[comp["kind"].isin(["mac", "ac"])].groupby(["experiment", "scope", "kind"])[
        "amount_per_sample"].sum() / 1e6
    out = pd.DataFrame({"Key": raw["experiment"].astype(str).to_numpy()})
    out["Model"] = out["Key"].map(KEY2MODEL)
    for scope in ["sender", "recon"]:
        for item, col in [("compute", "compute_uj"), ("weight", "weights_uj"), ("state", "state_uj")]:
            out[f"{scope}_{item}"] = raw[f"{scope}_{col}"].to_numpy(float)
        out[f"{scope}_total"] = raw[f"{scope}_uj"].to_numpy(float)
        for kind in ["mac", "ac"]:
            out[f"{scope}_{kind}"] = [float(ops.get((k, scope, kind), 0.0)) for k in out["Key"]]
    out["total"] = raw["total_uj"].to_numpy(float)
    out["decoder"] = raw["readout_uj_not_counted"].to_numpy(float)
    out["Bits"] = raw["bits"].to_numpy(float) if "bits" in raw else np.nan
    out["Transmission"] = np.where(out["Model"].isin(SNN_MODELS), "AER", "8-bit dense")
    missing = set(ENERGY_KEYS["DVS-G"]) - set(out["Key"])
    if missing:
        raise ValueError(f"[{label}] missing rows {missing}")
    return out


def load_dvsg_robust_json(runs):
    rows = []
    for key in DVSG_DENSE:
        for seed in SEEDS:
            path = runs / key / f"seed{seed}" / "summary.json"
            if path.is_file():
                data = json.loads(path.read_text())
                rows.append({"Model": KEY2MODEL[key], "Seed": seed,
                             "NMSE": data.get("robust_nmse"), "Acc": data.get("robust_acc")})
    return pd.DataFrame(rows)


def validate(data):
    for ds, d in data.items():
        scans = d["scans"]
        require_unique(scans, ["Model", "Seed", "Channel", "SNR"], f"{ds} scans")
        require_seeds(scans, ["Model", "Channel", "SNR"], f"{ds} scans")
        for (model, channel), part in scans.groupby(["Model", "Channel"]):
            grid = sorted(set(part["SNR"].round(3)))
            if grid != sorted(SNR_GRID):
                raise ValueError(f"[{ds} scans] {model} {channel}: SNR grid {grid}")
        if set(scans["Model"]) != set(MODELS):
            raise ValueError(f"[{ds} scans] models {sorted(set(scans['Model']))}")
        for name in ["main", "main_awgn10"]:
            require_unique(d[name], ["Model", "Seed"], f"{ds} {name}")
            require_seeds(d[name], ["Model"], f"{ds} {name}")
        require_unique(d["budget"], ["Model", "Channels", "Seed"], f"{ds} budget")
        require_seeds(d["budget"], ["Model", "Channels"], f"{ds} budget")
        require_unique(d["ablation"], ["Setting", "Seed"], f"{ds} ablation")
        require_seeds(d["ablation"], ["Setting"], f"{ds} ablation")
        if sorted(set(d["ablation"]["Setting"])) != sorted(ABLATION_ORDER):
            raise ValueError(f"[{ds} ablation] settings {sorted(set(d['ablation']['Setting']))}")
        sem = d["semantic"]
        require_unique(sem, ["Variant", "Seed", "Channel", "SNR"], f"{ds} semantic")
        cov = [set(map(tuple, sem.loc[sem.Variant.eq(v), ["Seed", "Channel", "SNR"]].to_numpy()))
               for v in ["With", "Without"]]
        if cov[0] != cov[1]:
            raise ValueError(f"[{ds} semantic] With/Without cover different seeds or SNRs")
        if set(d["aer_bits"]["Model"]) != set(SNN_MODELS):
            raise ValueError(f"[{ds} AER bits] models {sorted(set(d['aer_bits']['Model']))}")
        require_unique(d["samples"], ["Model", "Sample"], f"{ds} samples")
        require_unique(d["classes"], ["Model", "Class"], f"{ds} classes")
        if sorted(d["energy"]["Model"]) != sorted(MODELS) or sorted(d["quant"]["Model"]) != sorted(MODELS):
            raise ValueError(f"[{ds}] energy / INT8 tables must have one row per method")
        for frame, name in [(d["scans"], "scans"), (d["budget"], "budget")]:
            if not frame["Acc"].between(0, 100).all():
                raise ValueError(f"[{ds} {name}] accuracy outside 0-100")
    print("[check] coverage and uniqueness checks passed")


# ----------------------------------------------------------------------------
# Derived numbers (shared by figures, tables and the values report)
# ----------------------------------------------------------------------------
def robust_points(scans):
    awgn = scans["Channel"].eq("awgn") & scans["SNR"].between(-5, 5)
    ray = scans["Channel"].eq("rayleigh") & scans["SNR"].between(-5, 20)
    return scans[awgn | ray]


def robust_per_seed(scans):
    pts = robust_points(scans)
    counts = pts.groupby(["Model", "Seed"]).size()
    if not (counts == 26).all():
        raise ValueError(f"robust average expects 26 points per model and seed, got {counts.unique()}")
    return pts.groupby(["Model", "Seed"], as_index=False)[["NMSE", "Acc"]].mean()


def flag_failed_budget(budget, threshold):
    budget = budget.copy()
    budget["Failed"] = budget["Acc"] < threshold
    return budget


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def publication_style(canvas_width):
    global STYLE_SCALE
    # Same effective type size as the 14.8-inch ECG composite after every
    # figure is scaled to the manuscript text width.
    STYLE_SCALE = float(canvas_width) / 14.8
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 16.8 * STYLE_SCALE, "axes.labelsize": 16.8 * STYLE_SCALE,
        "axes.titlesize": 18.6 * STYLE_SCALE, "xtick.labelsize": 15.6 * STYLE_SCALE,
        "ytick.labelsize": 15.6 * STYLE_SCALE, "legend.fontsize": 15.5 * STYLE_SCALE,
        "axes.linewidth": 1.0 * STYLE_SCALE, "lines.linewidth": 1.6 * STYLE_SCALE,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
    })


def scaled(value):
    return float(value) * STYLE_SCALE


def style_axis(axis):
    axis.grid(True, linestyle="--", linewidth=0.55, alpha=0.28)
    axis.tick_params(direction="out")


def add_panel_letters(axes, letters):
    for axis, letter in zip(axes, letters):
        axis.text(-0.17, 1.06, letter, transform=axis.transAxes, fontsize=scaled(27.5),
                  fontweight="bold", va="bottom", ha="left")


def save_figure(fig, folder, stem):
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(folder / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] {folder / stem}.pdf")


def padded(low, high, frac=0.06, minimum=0.15):
    margin = max((high - low) * frac, minimum)
    return low - margin, high + margin


def plot_scan(axis, frame, metric):
    summary = summarize(frame, ["Model", "SNR"], [metric])
    for model in MODELS:
        part = summary[summary["Model"] == model].sort_values("SNR")
        if part.empty:
            continue
        x = part["SNR"].to_numpy(float)
        y = part[f"{metric}_mean"].to_numpy(float)
        s = part[f"{metric}_std"].fillna(0).to_numpy(float)
        axis.plot(x, y, marker=MODEL_MARKERS[model], color=MODEL_COLORS[model],
                  markersize=5.3, label=model)
        axis.fill_between(x, y - s, y + s, color=MODEL_COLORS[model], alpha=0.08, linewidth=0)
    axis.set_xlim(-5, 20)
    axis.set_xticks([-5, 0, 5, 10, 15, 20])
    axis.set_xlabel(r"Average $E_b/N_0$ (dB)")
    style_axis(axis)


def budget_means(budget, include_failed=True):
    """Seed means per method and channel count over all seeds.
    include_failed=False is used only to set the axis range."""
    used = budget if include_failed else budget[~budget["Failed"]]
    return summarize(used, ["Model", "Channels"], ["Bits", "NMSE", "Acc"])


def plot_budget(axis, budget, metric):
    # Only the seed means are drawn (no per-seed markers, no lines).
    means = budget_means(budget)
    for model in SNN_MODELS:
        m = means[means["Model"] == model].sort_values("Bits_mean")
        axis.scatter(m["Bits_mean"] / 1e3, m[f"{metric}_mean"], s=62, marker=MODEL_MARKERS[model],
                     color=MODEL_COLORS[model], edgecolor="black", linewidth=0.8, label=model, zorder=4)
    axis.set_xlabel(r"Bits/sample ($\times10^3$)")
    style_axis(axis)


def budget_limits(budgets):
    # The range follows the means of the runs that trained normally, so a mean
    # pulled far off by diverged runs (DVS-G vanilla SNN, C = 1) falls outside
    # the axes and is not shown; it is still part of the plotted data.
    means = pd.concat([budget_means(b, include_failed=False) for b in budgets])
    nmse_lim = padded(float(means["NMSE_mean"].min()), float(means["NMSE_mean"].max()), minimum=0.4)
    acc_low = max(0, 5 * math.floor((float(means["Acc_mean"].min()) - 1) / 5))
    return nmse_lim, (acc_low, 100)


def build_fig4(data, args, outdir):
    scans = pd.concat([data[ds]["scans"] for ds in DATASETS])
    summary = summarize(scans, ["Model", "Channel", "SNR"], ["NMSE"])
    scan_nmse = padded(float((summary["NMSE_mean"] - summary["NMSE_std"]).min()),
                       float((summary["NMSE_mean"] + summary["NMSE_std"]).max()))
    budgets = {ds: flag_failed_budget(data[ds]["budget"], args.failed_acc) for ds in DATASETS}
    budget_nmse, budget_acc = budget_limits(list(budgets.values()))
    acc_step = 5 if budget_acc[1] - budget_acc[0] <= 30 else 10 if budget_acc[1] - budget_acc[0] <= 60 else 20

    publication_style(14.1)
    plt.rcParams.update({"font.size": scaled(20.0), "axes.labelsize": scaled(20.0),
                         "xtick.labelsize": scaled(18.0), "ytick.labelsize": scaled(18.0),
                         "legend.fontsize": scaled(15.0)})
    fig, axes = plt.subplots(4, 3, figsize=(14.1, 14.4), constrained_layout=True)
    fig.get_layout_engine().set(wspace=.012, hspace=.015, w_pad=.025, h_pad=.025)
    legend_kw = dict(loc="best", fontsize=scaled(15.0), handlelength=1.2, labelspacing=.15, borderpad=.25)
    for di, ds in enumerate(DATASETS):
        frame = data[ds]["scans"]
        for mi, metric in enumerate(["NMSE", "Acc"]):
            row = 2 * di + mi
            for col, channel in enumerate(["awgn", "rayleigh"]):
                ax = axes[row, col]
                plot_scan(ax, frame[frame["Channel"] == channel], metric)
                ax.set_ylim(scan_nmse if metric == "NMSE" else (0, 100))
                ax.set_ylabel(("NMSE (dB)" if metric == "NMSE" else "Accuracy (%)") if col == 0 else "")
                if row == 0 and col == 0:
                    ax.legend(**legend_kw)
            ax = axes[row, 2]
            ylim = budget_nmse if metric == "NMSE" else budget_acc
            ax.set_ylim(ylim)
            plot_budget(ax, budgets[ds], metric)
            ax.set_ylabel("NMSE (dB)" if metric == "NMSE" else "Accuracy (%)")
            if metric == "Acc":
                ax.set_yticks(np.arange(budget_acc[0], 100.1, acc_step))
            if row == 0:
                ax.legend(**legend_kw)
    add_panel_letters(axes.ravel(), "abcdefghijkl")
    save_figure(fig, outdir, "Fig4")


def ecdf(values):
    values = np.sort(pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(float))
    return values, np.arange(1, values.size + 1) / max(1, values.size)


def plot_ecdf(axis, frame, column, models, dataset, ylabel):
    for model in models:
        x, y = ecdf(frame.loc[frame["Model"] == model, column])
        axis.plot(x / 1e3 if column == "Bits" else x, y, color=MODEL_COLORS[model], label=model)
    axis.set_xlabel(r"Bits/sample ($\times10^3$)" if column == "Bits" else "Per-sample NMSE (dB)")
    axis.set_ylabel("CDF" if ylabel else "")
    axis.text(0.04, 0.94, dataset, transform=axis.transAxes, ha="left", va="top",
              fontsize=scaled(15.0), fontweight="bold")
    axis.set_ylim(-0.04, 1.04)
    axis.set_yticks([0, 0.5, 1])
    style_axis(axis)


def plot_semantic(axis, frame, dataset, show_legend):
    # "Without" is drawn first and "With" on top of it; bands sit
    # below both lines. The legend keeps the order With, Without.
    styles = [("Without", "#4D4D4D", "--", "white", 2),
              ("With", MODEL_COLORS["S-LISTA"], "-", MODEL_COLORS["S-LISTA"], 4)]
    for variant, color, linestyle, face, z in styles:
        part = summarize(frame[frame["Variant"] == variant], ["SNR"], ["Acc"]).sort_values("SNR")
        x, y = part["SNR"].to_numpy(float), part["Acc_mean"].to_numpy(float)
        s = part["Acc_std"].fillna(0).to_numpy(float)
        axis.fill_between(x, y - s, y + s, color=color, alpha=0.10, linewidth=0, zorder=1)
        axis.plot(x, y, marker="o", color=color, linestyle=linestyle, markerfacecolor=face,
                  markeredgecolor=color, label=variant, zorder=z)
    axis.set_xlabel(r"$E_b/N_0$ (dB)")
    axis.set_ylabel("Accuracy (%)")
    axis.set_xlim(-5, 20)
    axis.set_xticks(np.arange(-5, 21, 5))
    axis.set_ylim(0, 100)
    axis.set_yticks(np.arange(0, 101, 20))
    axis.tick_params(labelleft=True)
    axis.text(0.97, 0.05, dataset, transform=axis.transAxes, ha="right", va="bottom",
              fontsize=scaled(15.0), fontweight="bold")
    style_axis(axis)
    if show_legend:
        handles, labels = axis.get_legend_handles_labels()
        order = [labels.index("With"), labels.index("Without")]
        axis.legend([handles[i] for i in order], [labels[i] for i in order],
                    frameon=True, loc="upper right", bbox_to_anchor=(0.985, 0.84),
                    fontsize=scaled(13.8), borderpad=0.28, labelspacing=0.20,
                    handlelength=1.35, handletextpad=0.42)


def class_matrix(classes):
    order = [m for m in MODELS if m in set(classes["Model"])]
    cols = sorted(classes["Class"].unique())
    table = classes.pivot(index="Model", columns="Class", values="NMSE").reindex(index=order, columns=cols)
    return order, cols, table.to_numpy(float)


def build_fig5(data, outdir):
    publication_style(15.8)
    fig = plt.figure(figsize=(15.8, 10.9), constrained_layout=True)
    fig.get_layout_engine().set(h_pad=0.04, w_pad=0.025, hspace=0.012, wspace=0.02)
    rows = fig.subfigures(3, 1, height_ratios=[0.92, 1.16, 1.00], hspace=0.012)

    top = rows[0].subplots(1, 4)
    for i, ds in enumerate(DATASETS):
        plot_ecdf(top[2 * i], data[ds]["samples"], "NMSE", MODELS, ds, ylabel=(i == 0))
        plot_ecdf(top[2 * i + 1], data[ds]["aer_bits"], "Bits", SNN_MODELS, ds, ylabel=False)
    # Same NMSE axis for both datasets so the two distributions can be compared.
    # The range covers the pooled 0.5-99.5 % quantiles; a few extreme samples
    # (e.g. the worst vanilla-SNN SHD sample near +15 dB) leave the axes instead
    # of compressing every curve. The report lists the true extremes.
    pooled = pd.concat([data[ds]["samples"]["NMSE"] for ds in DATASETS])
    nmse_range = padded(float(pooled.quantile(0.005)), float(pooled.quantile(0.995)), minimum=0.5)
    for i in range(2):
        top[2 * i].set_xlim(nmse_range)
    handles = [Line2D([], [], color=MODEL_COLORS[m], linewidth=scaled(1.6), label=m) for m in MODELS]
    top[3].legend(handles=handles, frameon=True, fontsize=scaled(13.8), loc="lower right",
                  handlelength=1.25, handletextpad=0.45, borderpad=0.30, labelspacing=0.22)

    mid = rows[1].subplots(1, 4, sharey=True)
    for i, ds in enumerate(DATASETS):
        sem = data[ds]["semantic"]
        for j, (channel, title) in enumerate([("awgn", "AWGN"), ("rayleigh", "Rayleigh fading")]):
            ax = mid[2 * i + j]
            plot_semantic(ax, sem[sem["Channel"] == channel], ds, show_legend=(j == 0))
            ax.set_title(title, fontsize=scaled(16.0))

    grid = rows[2].add_gridspec(1, 4, width_ratios=[1., .04, 1., .04], wspace=.06)
    heat_axes, images = [], []
    mats = {ds: class_matrix(data[ds]["classes"]) for ds in DATASETS}
    lo = min(np.nanmin(m[2]) for m in mats.values())
    hi = max(np.nanmax(m[2]) for m in mats.values())
    for i, ds in enumerate(DATASETS):
        order, cols, matrix = mats[ds]
        ax = rows[2].add_subplot(grid[0, 2 * i])
        cax = rows[2].add_subplot(grid[0, 2 * i + 1])
        image = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=lo, vmax=hi)
        step = 2 if len(cols) > 12 else 1
        ticks = np.arange(0, len(cols), step)
        ax.set_xticks(ticks)
        ax.set_xticklabels([cols[k] for k in ticks], rotation=45 if len(cols) > 12 else 0,
                           ha="right" if len(cols) > 12 else "center")
        ax.set_yticks(np.arange(len(order)))
        ax.set_yticklabels(order)
        ax.set_xlabel("Gesture class" if ds == "DVS-G" else "Speech class")
        rows[2].colorbar(image, cax=cax, label="NMSE (dB)")
        heat_axes.append(ax)
        images.append(image)
    add_panel_letters(list(top) + list(mid) + heat_axes, "abcdefghij")
    save_figure(fig, outdir, "Fig5")


# ----------------------------------------------------------------------------
# LaTeX tables
# ----------------------------------------------------------------------------
def tex_num(value, digits):
    if not np.isfinite(value):
        return "--"
    return f"{value:,.{digits}f}".replace(",", "{,}")


def tex_pm(mean, std, digits=2):
    return f"${tex_num(mean, digits)} \\pm {tex_num(std, digits)}$"


def tex_auto(value, sig=3):
    """Energy / operation counts with a fixed number of significant figures
    (three): 42,467 -> 42,500; 0.0712 -> 0.0712; 0 -> 0."""
    if not np.isfinite(value):
        return "--"
    if value == 0:
        return "$0$"
    exponent = int(math.floor(math.log10(abs(value))))
    decimals = sig - 1 - exponent
    rounded = round(value, decimals)
    if rounded != 0 and int(math.floor(math.log10(abs(rounded)))) > exponent:  # e.g. 999.6 -> 1,000
        decimals -= 1
        rounded = round(value, decimals)
    return f"${tex_num(rounded, max(0, decimals))}$"


def tex_ratio(value):
    if not np.isfinite(value):
        return "--"
    return "$1$" if np.isclose(value, 1) else f"${value:.3f}$"


def write_tex(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[table] {path}")


def table1(data, outdir):
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{\textbf{Main reconstruction and downstream task results under Rayleigh fading at 20~dB.}"
        r" Results use full-vector transmission and are reported as means $\pm$ standard deviations"
        r" over five independent seeds.}",
        r"\label{tab:main_event}",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Dataset & Method & NMSE (dB) & Accuracy (\%) & Bits/sample \\",
        r"\midrule",
    ]
    for i, ds in enumerate(DATASETS):
        if i:
            lines.append(r"\midrule")
        s = order_models(summarize(data[ds]["main"], ["Model"], ["NMSE", "Acc", "Bits"]))
        for _, r in s.iterrows():
            bits = tex_num(r["Bits_mean"], 0) if r["Bits_std"] < 0.5 else tex_pm(r["Bits_mean"], r["Bits_std"], 0)
            lines.append(f"{ds} & {r['Model']} & {tex_pm(r['NMSE_mean'], r['NMSE_std'])} & "
                         f"{tex_pm(r['Acc_mean'], r['Acc_std'])} & {bits} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write_tex(outdir / "Table1_main.tex", lines)


def table2(data, outdir):
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{\textbf{Core ablation results under Rayleigh fading at 20~dB.} Results use block-based"
        r" AER transmission and are reported as means $\pm$ standard deviations over five independent seeds.}",
        r"\label{tab:ablation_event}",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Dataset & Setting & NMSE (dB) & Accuracy (\%) & Bits/sample \\",
        r"\midrule",
    ]
    for i, ds in enumerate(DATASETS):
        if i:
            lines.append(r"\midrule")
        s = summarize(data[ds]["ablation"], ["Setting"], ["NMSE", "Acc", "Bits"]).set_index("Setting")
        for setting in ABLATION_ORDER:
            r = s.loc[setting]
            lines.append(f"{ds} & {setting} & {tex_pm(r['NMSE_mean'], r['NMSE_std'])} & "
                         f"{tex_pm(r['Acc_mean'], r['Acc_std'])} & {tex_pm(r['Bits_mean'], r['Bits_std'], 0)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write_tex(outdir / "Table2_ablation.tex", lines)


def energy_frame(data):
    frames = []
    for ds in DATASETS:
        e = data[ds]["energy"].copy()
        e["Dataset"] = ds
        ref = e.set_index("Model")
        for col in ["sender_total", "recon_total", "total"]:
            e[f"{col}_vs_lista"] = e[col] / float(ref.loc["ANN LISTA", col])
            e[f"{col}_vs_lstm"] = e[col] / float(ref.loc["ANN LSTM", col])
        frames.append(order_models(e))
    return pd.concat(frames, ignore_index=True)


def table3(data, outdir):
    """Same layout as the manuscript: block a (sensing) and block b
    (reconstruction) with identical columns: MAC, AC and the energy split
    Compute / Weights / State / Total."""
    e = energy_frame(data)
    cols = ["mac", "ac", "compute", "weight", "state", "total"]

    def block(scope, letter, title):
        rows = [
            r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llrrrrrr@{}}",
            r"\multicolumn{8}{@{}l}{\textbf{" + letter + r"}\quad " + title + r"} \\",
            r"\toprule",
            r"& & & & \multicolumn{4}{c}{Energy ($\mu$J)} \\",
            r"\cmidrule(lr){5-8}",
            r"Dataset & Method & MAC (MOP) & AC (MOP) & Compute & Weights & State & Total \\",
            r"\midrule",
        ]
        for i, ds in enumerate(DATASETS):
            if i:
                rows.append(r"\midrule")
            part = e[e["Dataset"] == ds]
            # Lowest energy per column and dataset in bold (ties all bold). MAC and
            # AC are operation counts by type, so they are not marked.
            best = {k: part[f"{scope}_{k}"].min() for k in ["compute", "weight", "state", "total"]}
            for j, (_, r) in enumerate(part.iterrows()):
                cells = []
                for k in cols:
                    text = tex_auto(r[f"{scope}_{k}"]).strip("$")
                    if k in best and np.isclose(r[f"{scope}_{k}"], best[k], rtol=1e-6, atol=0):
                        text = r"\textbf{" + text + "}"
                    cells.append(text)
                rows.append(f"{ds if j == 0 else ''} & {r['Model']} & " + " & ".join(cells) + r" \\")
        rows += [r"\bottomrule", r"\end{tabular*}"]
        return rows

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{\textbf{Computational cost and estimated energy consumption per sample.}",
        r"\textbf{a}, Sensing. \textbf{b}, Reconstruction.}",
        r"\label{tab:event_computational_cost}",
        r"\begingroup",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        *block("sender", "a", "Sensing"),
        "",
        r"\bigskip",
        *block("recon", "b", "Reconstruction"),
        "",
        r"\smallskip",
        r"\begin{minipage}{\textwidth}",
        r"\footnotesize",
        r"Operation counts and energy cover all time steps of each input sample, using models trained with seed 42"
        r" and evaluated over AWGN at 10~dB.",
        r"One MOP denotes $10^6$ operations, with each multiply--accumulate",
        r"(MAC) or accumulate (AC) counted as one operation.",
        r"Energy is estimated as described in Methods and split into arithmetic operations (Compute), weight reads"
        r" (Weights), and neuron-state reads and writes (State).",
        r"S-LISTA and the vanilla SNN use their AER operating points in Table~\ref{tab:event_ablation}; the ANN"
        r" baselines transmit 8-bit measurements.",
        r"The readout-gain multiplications of S-LISTA are included in Compute and are not counted as MACs.",
        r"The final dictionary readout, communication, and downstream task evaluation are excluded.",
        r"\end{minipage}",
        r"\endgroup",
        r"\end{table*}",
    ]
    write_tex(outdir / "Table3_energy.tex", lines)


def table_int8(data, outdir):
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{\textbf{Effect of INT8 post-training quantization.} INT8 quantization is applied to all"
        r" weight matrices included in the energy estimates (per output channel, symmetric), without"
        r" retraining; thresholds, readout gains, biases, decoders and downstream classifiers remain in"
        r" floating point. Models of Table~\ref{tab:energy_event}, seed 42, AWGN at 10~dB.}",
        r"\label{tab:int8_event}",
        r"\begin{tabular}{lllcccc}",
        r"\toprule",
        r" & & & \multicolumn{2}{c}{NMSE (dB)} & \multicolumn{2}{c}{Accuracy (\%)} \\",
        r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r"Dataset & Method & Transmission & Float & INT8 & Float & INT8 \\",
        r"\midrule",
    ]
    for i, ds in enumerate(DATASETS):
        if i:
            lines.append(r"\midrule")
        q = order_models(data[ds]["quant"])
        for _, r in q.iterrows():
            trans = "AER" if r["Model"] in SNN_MODELS else "8-bit full vector"
            lines.append(f"{ds} & {r['Model']} & {trans} & ${tex_num(r['float_nmse'], 3)}$ & "
                         f"${tex_num(r['int8_nmse'], 3)}$ & ${tex_num(r['float_acc'], 2)}$ & "
                         f"${tex_num(r['int8_acc'], 2)}$ \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write_tex(outdir / "SuppTable_int8.tex", lines)


# ----------------------------------------------------------------------------
# Values report for writing the Results text
# ----------------------------------------------------------------------------
def runs(xs, labels):
    start = 0
    for i in range(1, len(xs) + 1):
        if i == len(xs) or labels[i] != labels[start]:
            yield xs[start], xs[i - 1], labels[start]
            start = i


def fmt_range(a, b):
    return f"{a:g} dB" if a == b else f"{a:g} to {b:g} dB"


def paired(frame, a_mask, b_mask, keys, metric):
    a = frame[a_mask].set_index(keys)[metric]
    b = frame[b_mask].set_index(keys)[metric]
    diff = (a - b).dropna()
    m, s, n = mean_std(diff)
    t = m / (s / math.sqrt(n)) if n > 1 and s > 0 else np.nan
    return m, s, n, t, int((diff > 0).sum())


class Report:
    def __init__(self):
        self.lines = []
        self.tables = {}

    def h(self, text):
        self.lines += ["", f"## {text}", ""]

    def p(self, text=""):
        self.lines.append(text)

    def csv(self, name, frame):
        self.tables[name] = frame

    def table(self, frame, floatfmt="{:.2f}"):
        cols = list(frame.columns)
        self.p("| " + " | ".join(cols) + " |")
        self.p("|" + "---|" * len(cols))
        for _, r in frame.iterrows():
            cells = []
            for c in cols:
                v = r[c]
                cells.append(floatfmt.format(v) if isinstance(v, (float, np.floating)) and np.isfinite(v)
                             else str(v))
            self.p("| " + " | ".join(cells) + " |")


def section_main(rep, data):
    rep.h("Table 1: Rayleigh 20 dB, full-vector transmission (mean ± sample std, 5 seeds)")
    allrows = []
    for ds in DATASETS:
        s = order_models(summarize(data[ds]["main"], ["Model"], ["NMSE", "Acc", "Bits"]))
        s.insert(0, "Dataset", ds)
        allrows.append(s)
        ref = s.set_index("Model").loc["S-LISTA"]
        for _, r in s.iterrows():
            rep.p(f"- {ds} | {r['Model']}: NMSE {r['NMSE_mean']:.2f} ± {r['NMSE_std']:.2f} dB; "
                  f"accuracy {r['Acc_mean']:.2f} ± {r['Acc_std']:.2f}%; bits/sample {r['Bits_mean']:,.0f}.")
        for _, r in s[s["Model"] != "S-LISTA"].iterrows():
            rep.p(f"  - {ds} S-LISTA vs {r['Model']}: NMSE {r['NMSE_mean'] - ref['NMSE_mean']:+.2f} dB "
                  f"(positive = S-LISTA lower error); accuracy {ref['Acc_mean'] - r['Acc_mean']:+.2f} pp; "
                  f"bits {100 * (1 - ref['Bits_mean'] / r['Bits_mean']):.1f}% fewer.")
    rep.csv("table1_rayleigh20", pd.concat(allrows))
    rep.h("AWGN 10 dB counterpart (text only)")
    for ds in DATASETS:
        s = order_models(summarize(data[ds]["main_awgn10"], ["Model"], ["NMSE", "Acc"]))
        for _, r in s.iterrows():
            rep.p(f"- {ds} | {r['Model']}: NMSE {r['NMSE_mean']:.2f} ± {r['NMSE_std']:.2f} dB; "
                  f"accuracy {r['Acc_mean']:.2f} ± {r['Acc_std']:.2f}%.")


def section_robust(rep, data, checks):
    rep.h("Robustness summary: mean of 26 points (AWGN -5 to 5 dB and Rayleigh -5 to 20 dB), per seed then across seeds")
    rows = []
    for ds in DATASETS:
        per_seed = robust_per_seed(data[ds]["scans"])
        s = order_models(summarize(per_seed, ["Model"], ["NMSE", "Acc"]))
        s.insert(0, "Dataset", ds)
        rows.append(s)
        idx = s.set_index("Model")
        best_ann_nmse = idx.loc[["ANN LISTA", "ANN LSTM"], "NMSE_mean"].idxmin()
        best_ann_acc = idx.loc[["ANN LISTA", "ANN LSTM"], "Acc_mean"].idxmax()
        for _, r in s.iterrows():
            rep.p(f"- {ds} | {r['Model']}: robust NMSE {r['NMSE_mean']:.2f} ± {r['NMSE_std']:.2f} dB; "
                  f"robust accuracy {r['Acc_mean']:.2f} ± {r['Acc_std']:.2f}%.")
            checks.append(("robust", ds, r["Model"], "NMSE", r["NMSE_mean"]))
            checks.append(("robust", ds, r["Model"], "Acc", r["Acc_mean"]))
        rep.p(f"  - {ds}: S-LISTA minus best ANN ({best_ann_acc}) robust accuracy "
              f"{idx.loc['S-LISTA', 'Acc_mean'] - idx.loc[best_ann_acc, 'Acc_mean']:+.2f} pp; robust NMSE vs best "
              f"ANN ({best_ann_nmse}) {idx.loc[best_ann_nmse, 'NMSE_mean'] - idx.loc['S-LISTA', 'NMSE_mean']:+.2f} dB "
              f"(positive = S-LISTA better).")
        js = data[ds].get("robust_json")
        if js is not None and len(js):
            merged = per_seed.merge(js, on=["Model", "Seed"], suffixes=("", "_json"))
            gap = (merged["NMSE"] - merged["NMSE_json"]).abs().max()
            gap_acc = (merged["Acc"] - merged["Acc_json"]).abs().max()
            rep.p(f"  - {ds}: max difference to runs/*/summary.json: NMSE {gap:.4f} dB, accuracy {gap_acc:.4f} pp.")
    rep.csv("robustness_26pt", pd.concat(rows))


def section_scans(rep, data):
    rep.h("Fig. 4 SNR scans (full-vector transmission; trained at AWGN 10 dB)")
    rep.p("Rankings compare seed means only; they are not significance tests. "
          "‘Best’ = lowest NMSE or highest accuracy among the four methods.")
    means_all, pair_rows = [], []
    for ds in DATASETS:
        scans = data[ds]["scans"]
        summary = summarize(scans, ["Channel", "Model", "SNR"], ["NMSE", "Acc"])
        summary.insert(0, "Dataset", ds)
        means_all.append(summary)
        for channel in ["awgn", "rayleigh"]:
            part = summary[summary["Channel"] == channel]
            for metric, sign, unit in [("NMSE", -1, "dB"), ("Acc", 1, "pp")]:
                table = part.pivot(index="SNR", columns="Model", values=f"{metric}_mean").sort_index()
                std = part.pivot(index="SNR", columns="Model", values=f"{metric}_std").sort_index()
                xs = list(table.index)
                labels = []
                for x in xs:
                    best = (table.loc[x] * sign).idxmax()
                    labels.append("best" if best == "S-LISTA" else f"not best ({best} best)")
                rep.p(f"### {ds} / {channel} / {metric}")
                spans = "; ".join(f"{lab} at {fmt_range(a, b)}" for a, b, lab in runs(xs, labels))
                rep.p(f"- S-LISTA among all methods: {spans}.")
                for base in MODELS[1:]:
                    gains = (table["S-LISTA"] - table[base]) * sign
                    lab = ["better" if g > 1e-9 else "worse" if g < -1e-9 else "tie" for g in gains]
                    spans = "; ".join(f"{l} at {fmt_range(a, b)}" for a, b, l in runs(xs, lab))
                    rep.p(f"- vs {base}: {spans}; advantage range {gains.min():+.2f} to {gains.max():+.2f} {unit} "
                          f"(max at {gains.idxmax():g} dB, min at {gains.idxmin():g} dB).")
                    for x in xs:
                        pair_rows.append({"Dataset": ds, "Channel": channel, "Metric": metric, "Baseline": base,
                                          "SNR": x, "S-LISTA": table.loc[x, "S-LISTA"],
                                          "Baseline value": table.loc[x, base],
                                          "Advantage": gains.loc[x], "Unit": unit,
                                          "S-LISTA std": std.loc[x, "S-LISTA"]})
                cols = [m for m in MODELS]
                show = table[cols].copy()
                show.insert(0, "SNR (dB)", show.index)
                for m in cols:
                    show[m] = [f"{a:.2f} ± {b:.2f}" for a, b in zip(table[m], std[m])]
                rep.table(show.reset_index(drop=True))
                rep.p()
    rep.csv("fig4_scans_mean_std", pd.concat(means_all))
    rep.csv("fig4_scans_pairwise", pd.DataFrame(pair_rows))


def section_budget(rep, data, args):
    rep.h("Fig. 4 bit budget (AWGN 10 dB, full-vector transmission, only the channel count changes)")
    rep.p(f"Failed seeds are those with accuracy below {args.failed_acc:g}%. Fig. 4 plots means over all seeds; "
          "the axis range follows the non-failed means, so a mean pulled off by failed runs lies outside the axes. "
          "Means are given both over all seeds and over non-failed seeds.")
    rows = []
    for ds in DATASETS:
        b = flag_failed_budget(data[ds]["budget"], args.failed_acc)
        for _, r in b[b["Failed"]].iterrows():
            rep.p(f"- {ds}: failed run {r['Model']} C={r['Channels']} seed {r['Seed']} "
                  f"(NMSE {r['NMSE']:.2f} dB, accuracy {r['Acc']:.2f}%).")
        full = summarize(b, ["Model", "Channels"], ["Bits", "NMSE", "Acc"])
        ok = summarize(b[~b["Failed"]], ["Model", "Channels"], ["NMSE", "Acc"]).rename(
            columns=lambda c: c if c in ("Model", "Channels") else c + "_nonfailed")
        s = full.merge(ok, on=["Model", "Channels"], how="left")
        s.insert(0, "Dataset", ds)
        rows.append(s)
        piv = s.pivot(index="Channels", columns="Model")
        rep.p(f"### {ds}")
        out = pd.DataFrame({"Channels": piv.index,
                            "Bits": piv[("Bits_mean", "S-LISTA")].round(0).astype(int).to_numpy()})
        for m in SNN_MODELS:
            out[f"{m} NMSE"] = [f"{a:.2f} ± {c:.2f}" for a, c in zip(piv[("NMSE_mean", m)], piv[("NMSE_std", m)])]
            out[f"{m} Acc"] = [f"{a:.2f} ± {c:.2f}" for a, c in zip(piv[("Acc_mean", m)], piv[("Acc_std", m)])]
        out["NMSE gap (dB, + = S-LISTA lower)"] = (piv[("NMSE_mean", "Vanilla SNN")]
                                                  - piv[("NMSE_mean", "S-LISTA")]).to_numpy()
        out["Acc gap (pp)"] = (piv[("Acc_mean", "S-LISTA")] - piv[("Acc_mean", "Vanilla SNN")]).to_numpy()
        out["NMSE gap non-failed"] = (piv[("NMSE_mean_nonfailed", "Vanilla SNN")]
                                      - piv[("NMSE_mean_nonfailed", "S-LISTA")]).to_numpy()
        out["Acc gap non-failed"] = (piv[("Acc_mean_nonfailed", "S-LISTA")]
                                     - piv[("Acc_mean_nonfailed", "Vanilla SNN")]).to_numpy()
        rep.table(out)
        for m in SNN_MODELS:
            series = piv[("NMSE_mean", m)].to_numpy()
            mono = bool(np.all(np.diff(series) <= 0))
            rep.p(f"- {ds} {m}: NMSE monotonically improves with bits: {mono}.")
    rep.csv("fig4_budget", pd.concat(rows))


def section_ablation(rep, data, checks):
    rep.h("Table 2: core ablation (block AER, Rayleigh 20 dB)")
    rows = []
    for ds in DATASETS:
        s = summarize(data[ds]["ablation"], ["Setting"], ["NMSE", "Acc", "Bits"]).set_index("Setting")
        ref = s.loc["S-LISTA with AER"]
        for setting in ABLATION_ORDER:
            r = s.loc[setting]
            rep.p(f"- {ds} | {setting}: NMSE {r['NMSE_mean']:.2f} ± {r['NMSE_std']:.2f} dB "
                  f"({r['NMSE_mean'] - ref['NMSE_mean']:+.2f} dB vs S-LISTA with AER); accuracy "
                  f"{r['Acc_mean']:.2f} ± {r['Acc_std']:.2f}% ({r['Acc_mean'] - ref['Acc_mean']:+.2f} pp); "
                  f"bits {r['Bits_mean']:,.0f} ± {r['Bits_std']:,.0f} ({100 * (r['Bits_mean'] / ref['Bits_mean'] - 1):+.1f}%).")
            for metric in ["NMSE", "Acc", "Bits"]:
                checks.append(("table2", ds, setting, metric, r[f"{metric}_mean"]))
        ann_bits = data[ds]["main"].loc[data[ds]["main"]["Model"] == "ANN LISTA", "Bits"].mean()
        rep.p(f"  - {ds}: S-LISTA with AER transmits {100 * (1 - ref['Bits_mean'] / ann_bits):.1f}% fewer bits than "
              f"the 8-bit ANN baselines ({ref['Bits_mean']:,.0f} vs {ann_bits:,.0f}).")
        s = s.reset_index()
        s.insert(0, "Dataset", ds)
        rows.append(s)
    rep.csv("table2_ablation", pd.concat(rows))


def section_semantic(rep, data):
    rep.h("Fig. 5e–h: with vs without task supervision (full-vector; downstream accuracy)")
    rep.p("Paired by seed: diff = with − without. t = mean/(std/√n), n = 5; |t| > 2.78 corresponds to p < 0.05 "
          "(two-sided). ‘Seeds +’ counts seeds where the supervised model is higher.")
    rows = []
    for ds in DATASETS:
        sem = data[ds]["semantic"]
        for channel in ["awgn", "rayleigh"]:
            part = sem[sem["Channel"] == channel]
            rep.p(f"### {ds} / {channel}")
            out = []
            for snr in sorted(part["SNR"].unique()):
                p = part[np.isclose(part["SNR"], snr)]
                w = mean_std(p.loc[p.Variant == "With", "Acc"])
                wo = mean_std(p.loc[p.Variant == "Without", "Acc"])
                m, s, n, t, pos = paired(p, p.Variant.eq("With"), p.Variant.eq("Without"), ["Seed"], "Acc")
                out.append({"SNR (dB)": snr, "With": f"{w[0]:.2f} ± {w[1]:.2f}",
                            "Without": f"{wo[0]:.2f} ± {wo[1]:.2f}", "Diff (pp)": m, "Diff std": s,
                            "t": t, "Seeds +": pos})
                rows.append({"Dataset": ds, "Channel": channel, "SNR": snr, "With": w[0], "With std": w[1],
                             "Without": wo[0], "Without std": wo[1], "Diff": m, "Diff std": s, "t": t,
                             "Seeds positive": pos})
            out = pd.DataFrame(out)
            rep.table(out)
            higher = [f"{x:g}" for x, d in zip(out["SNR (dB)"], out["Diff (pp)"]) if d > 0]
            lower = [f"{x:g}" for x, d in zip(out["SNR (dB)"], out["Diff (pp)"]) if d < 0]
            rep.p(f"- supervised mean higher at [{', '.join(higher)}] dB; lower at [{', '.join(lower)}] dB.")
        per_seed = {}
        for v in ["With", "Without"]:
            pts = sem[sem["Variant"] == v].assign(Model=v)
            per_seed[v] = robust_per_seed(pts.drop(columns="Variant"))
        both = per_seed["With"].merge(per_seed["Without"], on="Seed", suffixes=("_with", "_without"))
        d = both["Acc_with"] - both["Acc_without"]
        m, s, n = mean_std(d)
        rep.p(f"- {ds} robust (26-point) accuracy: with {both['Acc_with'].mean():.2f}, without "
              f"{both['Acc_without'].mean():.2f}; paired diff {m:+.2f} ± {s:.2f} pp "
              f"(t = {m / (s / math.sqrt(n)) if s > 0 else float('nan'):.2f}).")
        nm = summarize(sem[sem["Channel"].eq("rayleigh") & np.isclose(sem["SNR"], 20)], ["Variant"], ["NMSE"])
        nm = nm.set_index("Variant")
        rep.p(f"- {ds} Rayleigh 20 dB NMSE: with {nm.loc['With', 'NMSE_mean']:.2f}, without "
              f"{nm.loc['Without', 'NMSE_mean']:.2f} dB.")
    rep.csv("fig5_task_supervision", pd.DataFrame(rows))


def section_distributions(rep, data):
    rep.h("Fig. 5a–d: per-sample distributions (seed 42; NMSE at Rayleigh 20 dB full-vector; bits from AER models)")
    rows = []
    for ds in DATASETS:
        for name, frame, col in [("NMSE (dB)", data[ds]["samples"], "NMSE"),
                                 ("AER bits/sample", data[ds]["aer_bits"], "Bits")]:
            for model in MODELS:
                v = frame.loc[frame["Model"] == model, col].dropna()
                if v.empty:
                    continue
                row = {"Dataset": ds, "Metric": name, "Model": model, "N": len(v), "Mean": v.mean(),
                       "P10": v.quantile(.1), "P25": v.quantile(.25), "Median": v.median(),
                       "P75": v.quantile(.75), "P90": v.quantile(.9), "Worst": v.max(), "Best": v.min()}
                rows.append(row)
                rep.p(f"- {ds} | {name} | {model}: median {row['Median']:.2f}, IQR [{row['P25']:.2f}, "
                      f"{row['P75']:.2f}], P10–P90 [{row['P10']:.2f}, {row['P90']:.2f}], max {row['Worst']:.2f}, "
                      f"n = {row['N']}.")
    rep.csv("fig5_distributions", pd.DataFrame(rows))


def section_classes(rep, data):
    rep.h("Fig. 5i,j: class-wise NMSE (seed 42, Rayleigh 20 dB, full-vector)")
    rows = []
    for ds in DATASETS:
        c = data[ds]["classes"]
        piv = c.pivot(index="Class", columns="Model", values="NMSE")
        for model in MODELS:
            col = piv[model]
            rep.p(f"- {ds} | {model}: class mean {col.mean():.2f} dB; best class {col.idxmin()} ({col.min():.2f}); "
                  f"worst class {col.idxmax()} ({col.max():.2f}); spread {col.max() - col.min():.2f} dB.")
        ann_best = piv[["ANN LISTA", "ANN LSTM"]].min(axis=1)
        ann_worst = piv[["ANN LISTA", "ANN LSTM"]].max(axis=1)
        between = ((piv["S-LISTA"] >= ann_best) & (piv["S-LISTA"] <= piv["Vanilla SNN"])).sum()
        better_than_vanilla = (piv["S-LISTA"] < piv["Vanilla SNN"]).sum()
        better_than_ann = (piv["S-LISTA"] < ann_best).sum()
        rep.p(f"  - {ds}: S-LISTA lower than Vanilla in {better_than_vanilla}/{len(piv)} classes; lower than both "
              f"ANNs in {better_than_ann}/{len(piv)}; between best ANN and Vanilla in {between}/{len(piv)}; "
              f"between the two ANNs in {((piv['S-LISTA'] >= ann_best) & (piv['S-LISTA'] <= ann_worst)).sum()}"
              f"/{len(piv)}.")
        piv = piv.reset_index()
        piv.insert(0, "Dataset", ds)
        rows.append(piv)
    rep.csv("fig5_classes", pd.concat(rows))


def section_energy(rep, data, checks):
    rep.h("Table 3: energy (seed 42, AWGN 10 dB; SNNs at their AER operating points)")
    e = energy_frame(data)
    for ds in DATASETS:
        part = e[e["Dataset"] == ds].set_index("Model")
        s = part.loc["S-LISTA"]
        for model, r in part.iterrows():
            decoder = f"{r['decoder']:,.1f} µJ" if np.isfinite(r["decoder"]) else "--"
            rep.p(f"- {ds} | {model}: sensing {r['sender_total']:,.2f} µJ, reconstruction {r['recon_total']:,.2f} µJ, "
                  f"total {r['total']:,.2f} µJ; decoder (not counted) {decoder}; "
                  f"transmitted bits {r['Bits']:,.0f}.")
            checks.append(("energy", ds, model, "total", r["total"]))
            checks.append(("energy", ds, model, "recon_total", r["recon_total"]))
        for ref in ["ANN LISTA", "ANN LSTM"]:
            r = part.loc[ref]
            rep.p(f"  - {ds} S-LISTA vs {ref}: sensing ×{r['sender_total'] / s['sender_total']:.1f} lower "
                  f"(ratio {s['sender_total'] / r['sender_total']:.3f}); reconstruction 1/{r['recon_total'] / s['recon_total']:.1f} "
                  f"(ratio {s['recon_total'] / r['recon_total']:.4f}); total ratio {s['total'] / r['total']:.3f} "
                  f"({100 * (1 - s['total'] / r['total']):.1f}% lower).")
        v = part.loc["Vanilla SNN"]
        rep.p(f"  - {ds} (completeness only) S-LISTA reconstruction is 1/{v['recon_total'] / s['recon_total']:.2f} of "
              f"the vanilla SNN.")
        rep.p(f"  - {ds} S-LISTA reconstruction split: compute {100 * s['recon_compute'] / s['recon_total']:.0f}%, "
              f"weights {100 * s['recon_weight'] / s['recon_total']:.0f}%, state {100 * s['recon_state'] / s['recon_total']:.0f}%.")
        if np.isfinite(s["decoder"]):
            lista = part.loc["ANN LISTA"]
            rep.p(f"  - {ds} including the decoder, receiver S-LISTA / ANN LISTA = "
                  f"{(s['recon_total'] + s['decoder']) / (lista['recon_total'] + lista['decoder']):.3f}.")
    rep.csv("table3_energy", e)


def section_int8(rep, data):
    rep.h("Supplementary: INT8 post-training quantization (seed 42, AWGN 10 dB)")
    rows = []
    for ds in DATASETS:
        q = order_models(data[ds]["quant"])
        for _, r in q.iterrows():
            rep.p(f"- {ds} | {r['Model']}: NMSE {r['float_nmse']:.3f} -> {r['int8_nmse']:.3f} dB "
                  f"({r['int8_nmse'] - r['float_nmse']:+.3f}); accuracy {r['float_acc']:.2f} -> {r['int8_acc']:.2f}% "
                  f"({r['int8_acc'] - r['float_acc']:+.2f} pp).")
        q = q.copy()
        q.insert(0, "Dataset", ds)
        rows.append(q)
    rep.csv("supp_int8", pd.concat(rows))


def section_checks(rep, checks):
    rep.h("Consistency check against the final project documents")
    bad = 0
    for kind, ds, item, metric, value in checks:
        key = (kind, ds, item, metric)
        if key not in DOC_VALUES:
            continue
        doc = DOC_VALUES[key]
        tol = 0.006 if metric in ("NMSE", "Acc") else max(1.0, abs(doc) * 1e-3)
        if metric in ("total", "recon_total"):
            tol = max(0.006, abs(doc) * 1e-3)
        ok = abs(value - doc) <= tol
        bad += not ok
        rep.p(f"- {'OK ' if ok else 'MISMATCH'} {kind} {ds} {item} {metric}: computed {value:.4f}, document {doc}.")
    rep.p(f"\n{bad} mismatch(es).")
    print(f"[check] document comparison: {bad} mismatch(es)")


def write_values(data, args, outdir):
    rep = Report()
    rep.p("# DVS-G and SHD values for the Results text")
    rep.p()
    rep.p("Generated by assemble_event_results.py from the files in sources.csv. Rules for writing:")
    rep.p("- Use only numbers from this file; mean ± sample standard deviation over seeds 42–46 unless stated.")
    rep.p("- Accuracy differences are percentage points (pp). Differences under 1 dB or within one seed std: "
          "write ‘comparable’ or ‘similar’. No significance claims except the paired t values given.")
    rep.p("- Bit budget: call it ‘bit budget’ / ‘bits per sample’, never ‘measurement dimension’.")
    rep.p("- Table 1 / Fig. 4 scans / Fig. 5a,c,e–j use full-vector (dense) models; Table 2, Table 3 and "
          "Fig. 5b,d use AER models.")
    rep.p("- Energy conclusions refer to the reconstruction network (excluding the DVS-G decoder, which is the "
          "same for S-LISTA and the ANNs).")
    checks = []
    section_main(rep, data)
    for ds in DATASETS:
        for _, r in summarize(data[ds]["main"], ["Model"], ["NMSE", "Acc"]).iterrows():
            checks.append(("table1", ds, r["Model"], "NMSE", r["NMSE_mean"]))
            checks.append(("table1", ds, r["Model"], "Acc", r["Acc_mean"]))
    section_robust(rep, data, checks)
    section_scans(rep, data)
    section_budget(rep, data, args)
    section_ablation(rep, data, checks)
    section_semantic(rep, data)
    section_distributions(rep, data)
    section_classes(rep, data)
    section_energy(rep, data, checks)
    section_int8(rep, data)
    section_checks(rep, checks)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "paper_values.md").write_text("\n".join(rep.lines) + "\n", encoding="utf-8")
    for name, frame in rep.tables.items():
        frame.to_csv(outdir / f"{name}.csv", index=False)
    print(f"[values] {outdir / 'paper_values.md'} and {len(rep.tables)} csv files")


# ----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--shd-root", type=Path, default=DEFAULT_SHD_ROOT,
                   help="SHD folder containing outputs/ and outputs_budget/")
    p.add_argument("--dvsg-root", type=Path, default=DEFAULT_DVSG_ROOT,
                   help="DVS-G folder containing outputs/")
    p.add_argument("--output-dir", type=Path, default=Path("results_event"))
    p.add_argument("--only", nargs="+", choices=["fig4", "fig5", "tables", "values"],
                   default=["fig4", "fig5", "tables", "values"])
    p.add_argument("--failed-acc", type=float, default=50.0,
                   help="A bit-budget run counts as diverged below this accuracy (%%); such runs stay in the Fig. 4 means "
                        "but do not set the axis range")
    return p.parse_args()


def main():
    args = parse_args()
    out = args.output_dir.resolve()
    data = {"DVS-G": load_dvsg(args.dvsg_root.resolve()), "SHD": load_shd(args.shd_root.resolve())}
    validate(data)
    if "fig4" in args.only:
        build_fig4(data, args, out / "figures")
    if "fig5" in args.only:
        build_fig5(data, out / "figures")
    if "tables" in args.only:
        for fn in (table1, table2, table3, table_int8):
            fn(data, out / "tables")
    if "values" in args.only:
        write_values(data, args, out / "values")
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(SOURCES).to_csv(out / "sources.csv", index=False)
    print(f"[done] {out}")


if __name__ == "__main__":
    main()
