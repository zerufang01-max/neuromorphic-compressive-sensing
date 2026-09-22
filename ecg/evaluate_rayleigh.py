#!/usr/bin/env python3
"""Evaluate frozen ECG models over coherent Rayleigh fading.

The default channel draws one complex Gaussian gain per ECG sample and
independent conditional BPSK bit errors given that gain. SNR is average Eb/N0.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

LABELS = {"lista_ann": "ANN LISTA", "alista": "ALISTA", "lamp": "LAMP", "lista_snn_1": "S-LISTA"}
COLORS = {"lista_ann": "#3b83bd", "alista": "#3a9d73", "lamp": "#7a5aa6", "lista_snn_1": "#d84a3a"}


def parse_args():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-dir", type=Path, default=here)
    p.add_argument("--output", type=Path, default=here / "outputs")
    p.add_argument("--checkpoint-root", type=Path)
    p.add_argument("--data-dir", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--snr-min", type=float, default=-5.0)
    p.add_argument("--snr-max", type=float, default=20.0)
    p.add_argument("--snr-step", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-batches", type=int, help="Optional quick check; omitted means all test signals.")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--fading", choices=("sample", "measurement", "bit"), default="sample",
                   help="Fading granularity. Default 'sample' shares one gain within each ECG sample.")
    args = p.parse_args()
    if not all(math.isfinite(v) for v in (args.snr_min, args.snr_max, args.snr_step)):
        p.error("SNR values must be finite")
    if args.snr_step <= 0 or args.snr_max < args.snr_min:
        p.error("Require snr-step > 0 and snr-max >= snr-min")
    if args.batch_size < 1 or (args.max_batches is not None and args.max_batches < 1):
        p.error("Batch sizes/counts must be positive")
    if args.seed < 0:
        p.error("Seed must be nonnegative")
    return args


def snr_grid(low, high, step):
    grid = [low + i * step for i in range(int(math.floor((high - low) / step + 1e-10)) + 1)]
    if grid[-1] < high - 1e-9:
        grid.append(high)
    return [round(v, 10) for v in grid]


def quantize_indices(values, bits=8, clip=3.0):
    # Matches the original signed uniform quantizer, including round-to-even.
    values = np.asarray(values, dtype=np.float32)
    levels = (1 << bits) - 1
    scaled = (np.clip(values, -clip, clip) / np.float32(clip) + np.float32(1)) * np.float32(0.5 * levels)
    return np.clip(np.rint(scaled), 0, levels).astype(np.int64)


def transmit_rayleigh(values, snr_db, rng, fading="bit", bits=8, clip=3.0):
    """Quantization and conditional coherent-Rayleigh hard bit errors.

    One common RNG schedule is reused across models and SNRs. Sample fading
    shares one gain across all transmitted bits within each ECG sample.
    """
    q = quantize_indices(values, bits, clip)
    shifts = np.arange(bits, dtype=np.int64)
    sent_bits = (q[..., None] >> shifts) & 1
    if fading == "bit":
        shape = sent_bits.shape
    elif fading == "measurement":
        shape = (*q.shape, 1)
    elif fading == "sample":
        shape = (q.shape[0],) + (1,) * (sent_bits.ndim - 1)
    else:
        raise ValueError(f"Unknown fading granularity: {fading}")
    h_real = rng.standard_normal(shape, dtype=np.float32) * np.float32(math.sqrt(0.5))
    h_imag = rng.standard_normal(shape, dtype=np.float32) * np.float32(math.sqrt(0.5))
    gamma = np.float32(10.0 ** (float(snr_db) / 10.0))
    argument = np.sqrt(gamma * (h_real * h_real + h_imag * h_imag))
    flat_argument = argument.reshape(-1)
    conditional_ber = np.fromiter(
        (0.5 * math.erfc(float(v)) for v in flat_argument),
        dtype=np.float32,
        count=flat_argument.size,
    ).reshape(argument.shape)
    flips = rng.random(sent_bits.shape, dtype=np.float32) < conditional_ber
    decided = np.logical_xor(sent_bits.astype(bool), flips).astype(np.int64)
    errors = int(np.count_nonzero(decided != sent_bits))
    received_q = np.sum(decided << shifts, axis=-1)
    decoded = (received_q.astype(np.float32) / np.float32((1 << bits) - 1) * np.float32(2) - np.float32(1)) * np.float32(clip)
    return decoded, errors, int(sent_bits.size)


def theoretical_ber(snr_db):
    gamma = 10.0 ** (float(snr_db) / 10.0)
    return 0.5 * (1.0 - math.sqrt(gamma / (1.0 + gamma)))


def configure(Config, args, torch):
    Config.N = 256
    Config.DICT_SIZE = 256
    Config.M = 78
    Config.TOTAL_MEAS_TARGET = 78
    Config.EFFECTIVE_TOTAL_MEAS = 78
    Config.NUM_LAYERS = 8
    Config.TIME_STEPS = 1
    Config.CS_MODE = "learnable_dictionary"
    Config.QUANT_BITS = 8
    Config.QUANT_CLIP_VALUE = 3.0
    Config.BITS_PER_SAMPLE = 624
    Config.THETA_LISTA_SNN = 1.0
    Config.FINAL_THETA_SNN = 0.2
    Config.SURROGATE_WIDTH = 0.5
    Config.SLISTA_INIT = "random"
    Config.TRAIN_SNR_DB = 5.0
    Config.SEED = 42
    Config.BATCH_SIZE = args.batch_size
    Config.DATA_DIR = str(args.data_dir)
    if args.cpu or not torch.cuda.is_available():
        Config.DEVICE = torch.device("cpu")
    else:
        if not 0 <= args.gpu < torch.cuda.device_count():
            raise ValueError(f"GPU {args.gpu} is not available")
        Config.DEVICE = torch.device(f"cuda:{args.gpu}")
    return Config.DEVICE


def load_model(key, path, classes, Config, torch):
    from edge_compression import EdgeSensor
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not {"model", "sensor"} <= payload.keys():
        raise ValueError(f"Checkpoint lacks model/sensor: {path}")
    expected = {"code_version": Config.CODE_VERSION, "num_layers": 8,
                "cs_mode": "learnable_dictionary", "train_snr_db": 5.0,
                "quant_bits": 8}
    for name, value in expected.items():
        if payload.get(name) != value:
            raise ValueError(f"{path}: {name}={payload.get(name)!r}, expected {value!r}")
    if key == "lista_snn_1" and payload.get("slista_init") != "random":
        raise ValueError(f"Unexpected S-LISTA initialization metadata: {path}")
    torch.manual_seed(Config.SEED)
    sensor = EdgeSensor().to(Config.DEVICE)
    # Load Phi before construction; all persistent network buffers are then
    # restored with strict loading. phi_override also supplies the loaded Phi.
    sensor.load_state_dict(payload["sensor"], strict=True)
    model = classes[key](phi=sensor.effective_phi().detach()).to(Config.DEVICE)
    mode = "spiking" if key == "lista_snn_1" else "ann"
    if hasattr(model, "set_mode"):
        model.set_mode(mode)
    model.load_state_dict(payload["model"], strict=True)
    model.eval().requires_grad_(False)
    sensor.eval().requires_grad_(False)
    return model, sensor


def evaluate(model, sensor, loader, snr_db, args, Config, torch):
    # Identical channel/noise draws for every model and every SNR. SNR only
    # rescales the noise; input bit patterns may differ between learned sensors.
    rng = np.random.default_rng(args.seed)
    error_sum = signal_sum = 0.0
    bit_errors = bit_count = samples = clipped = measured_count = 0
    with torch.inference_mode():
        for index, batch in enumerate(loader):
            if args.max_batches is not None and index >= args.max_batches:
                break
            x = (batch[0] if isinstance(batch, (tuple, list)) else batch).to(Config.DEVICE)
            y, _, _ = sensor(x, mode="shared")
            measured = y.detach().cpu().numpy()
            received, errors, count = transmit_rayleigh(
                measured, snr_db, rng, args.fading, Config.QUANT_BITS, Config.QUANT_CLIP_VALUE)
            y_recv = torch.from_numpy(received).to(Config.DEVICE)
            x_hat, _, _, _ = model(y_recv, states=None, return_debug=False,
                                  phi_override=sensor.effective_phi())
            if not bool(torch.isfinite(x_hat).all()):
                raise RuntimeError(f"Non-finite reconstruction at SNR={snr_db}")
            # Aggregate error energy / aggregate target energy, as in the old plot.
            error_sum += float((x_hat - x).square().sum(dtype=torch.float64).item())
            signal_sum += float(x.square().sum(dtype=torch.float64).item())
            samples += int(x.shape[0])
            bit_errors += errors
            bit_count += count
            clipped += int(np.count_nonzero(np.abs(measured) > Config.QUANT_CLIP_VALUE))
            measured_count += int(measured.size)
    if samples == 0 or signal_sum <= 0 or bit_count == 0:
        raise RuntimeError("Evaluation has no samples or no positive target energy")
    return {"nmse_db": 10.0 * math.log10(max(error_sum, 1e-12) / max(signal_sum, 1e-12)),
            "ber": bit_errors / bit_count, "theoretical_ber": theoretical_ber(snr_db),
            "samples": samples, "bit_errors": bit_errors, "bit_count": bit_count,
            "measurement_clip_fraction": clipped / measured_count}


def plot_results(rows, output_dir, plt):
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    for key in LABELS:
        selected = sorted((r for r in rows if r["target"] == key), key=lambda r: r["snr_db"])
        ax.plot([r["snr_db"] for r in selected], [r["nmse_db"] for r in selected],
                marker="s" if key == "lista_snn_1" else "o", markersize=3.8,
                linewidth=1.4, color=COLORS[key], label=LABELS[key])
    ax.set_xlabel(r"Average $E_b/N_0$ (dB)")
    ax.set_ylabel("NMSE (dB)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"ecg_rayleigh_snr.{ext}", dpi=300)
    plt.close(fig)


def main():
    args = parse_args()
    args.project_dir = args.project_dir.expanduser().resolve()
    args.checkpoint_root = (args.checkpoint_root or args.output).expanduser().resolve()
    args.data_dir = (args.data_dir or args.project_dir / "data/mitdb_pt").expanduser().resolve()
    args.output_dir = (args.output_dir or args.output / "rayleigh").expanduser().resolve()
    from plot_results import read_best_rows, read_final_slista, checkpoint_path
    selected = read_best_rows(args.checkpoint_root / "learned" / "models.csv")
    selected["lista_snn_1"] = read_final_slista(args.checkpoint_root / "final_slista")
    paths = {key: checkpoint_path(selected[key]) for key in LABELS}
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError("Required AWGN checkpoints missing (no training fallback):\n" + "\n".join(missing))
    sys.path.insert(0, str(args.project_dir))
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from config import Config
    from snn_models import ALISTA, LAMP, HybridLISTA
    from utils import MITBIH_Dataset
    device = configure(Config, args, torch)
    dataset = MITBIH_Dataset("test")
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                         shuffle=False, num_workers=0)
    grid = snr_grid(args.snr_min, args.snr_max, args.snr_step)
    classes = {"lista_ann": HybridLISTA, "alista": ALISTA, "lamp": LAMP, "lista_snn_1": HybridLISTA}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "status": "running", "training_performed": False,
        "checkpoints": {k: str(v) for k, v in paths.items()},
        "data_file": str(args.data_dir / Config.TEST_FILE), "test_set_size": len(dataset),
        "snr_grid_db": grid, "snr_definition": "average Eb/N0 per uncoded BPSK bit",
        "fading_granularity": args.fading,
        "channel": "SISO quasi-static Rayleigh; h~CN(0,1); one h per ECG sample by default",
        "receiver": "perfect CSIR; no CSIT; coherent BPSK hard decisions sampled via conditional BER",
        "channel_seed": args.seed, "common_channel_draws": True,
        "realizations_per_snr": 1, "batch_size": args.batch_size,
        "max_batches": args.max_batches, "device": str(device),
        "train_snr_db": 5, "N": 256, "M": 78, "K": 8, "T": 1,
        "quant_bits": 8, "quant_clip_value": 3.0,
        "theta_snn": 1.0, "final_theta_snn": 0.2,
        "torch_version": torch.__version__, "numpy_version": np.__version__,
        "source_sha256": {name: hashlib.sha256((args.project_dir / name).read_bytes()).hexdigest()
                          for name in ("config.py", "snn_models.py", "edge_compression.py", "utils.py")},
    }
    metadata_file = args.output_dir / "run_metadata.json"
    metadata_file.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Device: {device}; test signals: {len(dataset)}; SNR points: {len(grid)}", flush=True)
    print(f"Rayleigh per {args.fading}; perfect CSI; hard BPSK decisions; AWGN-trained weights only", flush=True)
    rows = []
    fields = ["target", "model", "snr_db", "nmse_db", "ber", "theoretical_ber",
              "samples", "bit_errors", "bit_count", "measurement_clip_fraction", "checkpoint"]
    csv_path = args.output_dir / "snr_scan.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        handle.flush()
        for key, path in paths.items():
            print(f"Loading {LABELS[key]}: {path}", flush=True)
            model, sensor = load_model(key, path, classes, Config, torch)
            for snr in grid:
                row = {"target": key, "model": LABELS[key], "snr_db": snr,
                       **evaluate(model, sensor, loader, snr, args, Config, torch),
                       "checkpoint": str(path)}
                writer.writerow(row)
                handle.flush()
                rows.append(row)
                print(f"{LABELS[key]:9s} | SNR {snr:5.1f} dB | NMSE {row['nmse_db']:7.2f} dB | BER {row['ber']:.5f}", flush=True)
            del model, sensor
    plot_results(rows, args.output_dir, plt)
    metadata["status"] = "completed"
    metadata["rows"] = len(rows)
    metadata_file.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved results to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
