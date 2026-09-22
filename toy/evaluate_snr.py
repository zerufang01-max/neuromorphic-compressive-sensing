"""Frozen-weight robustness evaluation under AWGN-corrupted measurements."""

import argparse
import csv
import json
import math
from pathlib import Path

import torch

from config import (
    CONDITIONS, NUMERICAL_EPS, Protocol, SNR_CONDITION_ID,
    SNR_DB_VALUES, SNR_MODEL_IDS,
)
from data import generator, measurement_matrix, sparse_batch
from models import build_model
from utils import DEFAULT_OUTPUT, selected_models


def find_models(output):
    found = selected_models(output)
    wanted = [(SNR_CONDITION_ID, model_id) for model_id in SNR_MODEL_IDS]
    missing = [key for key in wanted if key not in found]
    if missing:
        raise ValueError(f'Missing robustness models: {missing}')
    return [found[key] for key in wanted]



def add_awgn(measurements, snr_db, rng):
    if snr_db is None:
        return measurements
    signal_power = measurements.square().mean(dim=1, keepdim=True)
    noise_std = torch.sqrt(signal_power * 10.0 ** (-float(snr_db) / 10.0))
    noise = torch.randn(measurements.shape, generator=rng,
                        device=measurements.device, dtype=measurements.dtype)
    return measurements + noise_std * noise


@torch.no_grad()
def evaluate_model(configuration, checkpoint, snr_db, device, cfg):
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if "protocol" in payload:
        cfg = Protocol(**payload["protocol"])
    for field in ("method", "depth", "time_steps", "condition", "parameters"):
        if payload["job"][field] != configuration[field]:
            raise ValueError(f"Checkpoint and manifest disagree: {field}: {checkpoint}")
    condition = next(row for row in CONDITIONS
                     if row["id"] == SNR_CONDITION_ID)
    matrix = measurement_matrix(
        cfg.n, condition["m"], cfg.max_m, cfg.matrix_seed, device)
    model = build_model(
        configuration["method"], matrix, configuration["depth"],
        configuration["time_steps"], configuration["parameters"]).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()

    signal_rng = generator(cfg.test_seed, device)
    noise_rng = generator(cfg.noise_seed, device)
    error = target_energy = 0.0
    remaining = cfg.test_samples
    while remaining:
        size = min(cfg.eval_batch_size, remaining)
        target, measurements = sparse_batch(
            size, condition["s"], matrix, signal_rng,
            cfg.amplitude_min, cfg.amplitude_max)
        noisy_measurements = add_awgn(measurements, snr_db, noise_rng)
        estimate, _ = model(noisy_measurements)
        error += float((estimate - target).square().sum())
        target_energy += float(target.square().sum())
        remaining -= size
    nmse_db = 10 * math.log10(
        max(error, NUMERICAL_EPS) / max(target_energy, NUMERICAL_EPS))
    return {
        "seed": configuration.get("seed", cfg.model_seed),
        "condition_id": condition["id"], "method": configuration["method"],
        "id": configuration["id"], "depth": configuration["depth"],
        "time_steps": configuration["time_steps"], "snr_db": snr_db,
        "nmse_db": nmse_db, "test_samples": cfg.test_samples,
    }


def main(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    cfg = Protocol()
    models = find_models(output)
    rows = []
    for configuration, checkpoint in models:
        for snr_db in SNR_DB_VALUES:
            row = evaluate_model(
                configuration, checkpoint, snr_db, device, cfg)
            rows.append(row)
            snr_label = "clean" if snr_db is None else f"{snr_db:g} dB"
            print(f"{row['id']:<18} | SNR={snr_label:>6} | "
                  f"NMSE={row['nmse_db']:.2f} dB")

    clean = {row["id"]: row["nmse_db"] for row in rows
             if row["snr_db"] is None}
    for row in rows:
        row["degradation_db"] = row["nmse_db"] - clean[row["id"]]
    output = output / "metrics"
    output.mkdir(parents=True, exist_ok=True)
    with open(output / "snr_results.json", "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    with open(output / "snr_results.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--device", default="cuda:0")
    main(parser.parse_args())
