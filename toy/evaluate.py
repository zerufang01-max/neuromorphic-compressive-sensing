"""Held-out evaluation of all trained models (Fig. 2a-d).

For every seed and every (condition, model) of configs/paper.json:
  * NMSE on the 10,000-signal test set, analytical energy, firing rate,
    mean gated support size (S-LISTA)                       -> Fig. 2a-c
  * NMSE under AWGN on the measurements for SNR_MODEL_IDS    -> Fig. 2d
Results per seed and mean +/- sample SD over seeds are written to
<output>/metrics/{results,snr_results}{,_per_seed}.{json,csv}.

NMSE = 10 log10( sum ||x_hat - x||^2 / sum ||x||^2 ) over the whole test set.
SNR  = per-sample measurement power / noise power.
"""

import argparse
import csv
import math
import statistics
from pathlib import Path

import torch

from config import (
    NUMERICAL_EPS, SNR_CONDITION_ID, SNR_DB_VALUES, SNR_MODEL_IDS, TRAINING_SEEDS,
)
from data import generator, sparse_batch
from models import SpikingLISTA
from utils import DEFAULT_OUTPUT, PAPER_CONFIG, checkpoint_path, load_model, read_json, save_json


def add_awgn(measurements, snr_db, rng):
    if snr_db is None:
        return measurements
    power = measurements.square().mean(dim=1, keepdim=True)
    std = torch.sqrt(power * 10.0 ** (-float(snr_db) / 10.0))
    noise = torch.randn(measurements.shape, generator=rng, device=measurements.device,
                        dtype=measurements.dtype)
    return measurements + std * noise


@torch.no_grad()
def evaluate(model, matrix, job, cfg, snr_db=None):
    condition = job["condition"]
    signal_rng = generator(cfg.test_seed, matrix.device)
    noise_rng = generator(cfg.noise_seed, matrix.device)
    error = energy = 0.0
    counts = {"mac": 0.0, "ac": 0.0, "spikes": 0.0, "support": 0.0,
              "firing_rate_numerator": 0.0, "firing_rate_denominator": 0.0}
    remaining = cfg.test_samples
    while remaining:
        size = min(cfg.eval_batch_size, remaining)
        target, measurements = sparse_batch(size, condition["s"], matrix, signal_rng,
                                            cfg.amplitude_min, cfg.amplitude_max)
        estimate, _ = model(add_awgn(measurements, snr_db, noise_rng))
        error += float((estimate - target).square().sum())
        energy += float(target.square().sum())
        if isinstance(model, SpikingLISTA):
            for key, value in SpikingLISTA.operation_counts(
                    model.last_trace, model.n, model.m).items():
                counts[key] += value
        remaining -= size
    samples = cfg.test_samples
    row = {"seed": job.get("seed"), "condition_id": condition["id"], "id": job["id"],
           "method": job["method"], "depth": job["depth"], "s": condition["s"],
           "m": condition["m"], "snr_db": snr_db,
           "nmse_db": 10 * math.log10(max(error, NUMERICAL_EPS) / max(energy, NUMERICAL_EPS)),
           "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    if isinstance(model, SpikingLISTA):
        row.update(mac_per_sample=counts["mac"] / samples,
                   ac_per_sample=counts["ac"] / samples,
                   spikes_per_sample=counts["spikes"] / samples,
                   support_per_sample=counts["support"] / samples,
                   firing_rate=counts["firing_rate_numerator"] / counts["firing_rate_denominator"],
                   theta=[float(t) for t in model.theta],
                   eta=float(model.eta))
    else:
        row.update(mac_per_sample=float(model.macs_per_sample()), ac_per_sample=0.0)
    row["energy_uj"] = row["mac_per_sample"] * cfg.e_mac_uj + row["ac_per_sample"] * cfg.e_ac_uj
    return row


def aggregate(rows, keys):
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    summary = []
    for group in groups.values():
        out = {"seeds": [r["seed"] for r in group]}
        for key, value in group[0].items():
            if key == "seed" or isinstance(value, list):
                continue                      # per-seed lists (theta) stay in *_per_seed
            if key in keys or isinstance(value, (str, bool)) or value is None:
                out[key] = value
                continue
            values = [r[key] for r in group]
            out[key] = statistics.mean(values)
            out[key + "_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        summary.append(out)
    return summary


def write(rows, path):
    save_json(path.with_suffix(".json"), rows)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(args):
    device = torch.device(args.device)
    jobs = read_json(args.config)["jobs"]
    clean, noisy = [], []
    for seed in args.seeds:
        for job in jobs:
            path = checkpoint_path(args.output, seed, job["condition"]["id"], job["id"])
            model, matrix, trained_job, cfg = load_model(path, device)
            if trained_job.get("seed") != seed or trained_job["id"] != job["id"]:
                raise ValueError(f"Checkpoint does not match the requested job: {path}")
            row = evaluate(model, matrix, trained_job, cfg)
            clean.append(row)
            print(f"seed {seed} {row['condition_id']:>9} {row['id']:<11} "
                  f"{row['nmse_db']:7.2f} dB {row['energy_uj']:.4f} uJ", flush=True)
            if job["condition"]["id"] == SNR_CONDITION_ID and job["id"] in SNR_MODEL_IDS:
                for snr in SNR_DB_VALUES:
                    noisy.append(evaluate(model, matrix, trained_job, cfg, snr))
    out = Path(args.output) / "metrics"
    out.mkdir(parents=True, exist_ok=True)
    write(clean, out / "results_per_seed")
    write(aggregate(clean, ["condition_id", "id"]), out / "results")
    if noisy:
        reference = {(r["seed"], r["id"]): r["nmse_db"] for r in noisy if r["snr_db"] is None}
        for row in noisy:
            row["degradation_db"] = row["nmse_db"] - reference[(row["seed"], row["id"])]
        write(noisy, out / "snr_results_per_seed")
        write(aggregate(noisy, ["condition_id", "id", "snr_db"]), out / "snr_results")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--config", type=Path, default=PAPER_CONFIG)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(TRAINING_SEEDS))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    main(parser.parse_args())
