"""Held-out evaluation and analytical energy accounting."""

import argparse
import csv
import json
import math
from pathlib import Path

import torch

from config import CONDITIONS, NUMERICAL_EPS, Protocol, architectures
from data import generator, measurement_matrix, sparse_batch
from models import build_model
from utils import DEFAULT_OUTPUT, selected_models


@torch.no_grad()
def evaluate(configuration, checkpoint, device, cfg):
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if "protocol" in payload:
        cfg = Protocol(**payload["protocol"])
    for field in ("method", "depth", "time_steps", "condition", "parameters"):
        if payload["job"][field] != configuration[field]:
            raise ValueError(f"Checkpoint and manifest disagree: {field}: {checkpoint}")
    condition = configuration["condition"]
    matrix = measurement_matrix(
        cfg.n, condition["m"], cfg.max_m, cfg.matrix_seed, device)
    model = build_model(
        configuration["method"], matrix, configuration["depth"],
        configuration["time_steps"], configuration["parameters"]).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()

    rng = generator(cfg.test_seed, device)
    remaining = cfg.test_samples
    error = target_energy = 0.0
    ac_total = firing_total = spike_total = 0.0
    while remaining:
        size = min(cfg.eval_batch_size, remaining)
        target, measurements = sparse_batch(
            size, condition["s"], matrix, rng,
            cfg.amplitude_min, cfg.amplitude_max)
        estimate, _ = model(measurements)
        error += float((estimate - target).square().sum())
        target_energy += float(target.square().sum())
        if configuration["method"] == "slista":
            spike_total += float(model.last_spike_count)
            ac_total += float(model.last_ac_count)
            firing_total += float(model.last_firing_rate) * size
        remaining -= size

    nmse_db = 10 * math.log10(
        max(error, NUMERICAL_EPS) / max(target_energy, NUMERICAL_EPS))
    if configuration["method"] == "slista":
        energy_uj = (condition["m"] * cfg.n * cfg.e_mac_uj
                     + ac_total / cfg.test_samples * cfg.e_ac_uj)
        firing_rate = firing_total / cfg.test_samples
    else:
        energy_uj = model.macs_per_sample() * cfg.e_mac_uj
        firing_rate = None

    trainable = sum(parameter.numel() for parameter in model.parameters()
                    if parameter.requires_grad)
    stored = sum(value.numel() for value in model.state_dict().values())
    return {
        "seed": configuration.get("seed", cfg.model_seed),
        "spikes_per_sample": spike_total / cfg.test_samples if configuration["method"] == "slista" else None,
        "test_samples": cfg.test_samples,
        "condition_id": condition["id"], "n": cfg.n,
        "m": condition["m"], "sparsity": condition["s"],
        "measurement_ratio": condition["m"] / cfg.n,
        "sparsity_over_m": condition["s"] / condition["m"],
        "id": configuration["id"], "method": configuration["method"],
        "depth": configuration["depth"],
        "time_steps": configuration["time_steps"],
        "nmse_db": nmse_db, "energy_uj": energy_uj,
        "firing_rate": firing_rate,
        "trainable_parameters": trainable,
        "stored_parameters": stored,
        "stage1_learning_rate": configuration["stage1_learning_rate"],
        "stage2_learning_rate": configuration["stage2_learning_rate"],
        "stage1_steps": cfg.stage1_steps,
        "stage2_steps": cfg.stage2_steps,
        "batch_size": cfg.batch_size,
    }


def main(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    cfg = Protocol()
    selected = selected_models(output)
    expected = [(condition["id"], architecture["id"])
                for condition in CONDITIONS for architecture in architectures()]
    missing = [key for key in expected if key not in selected]
    if missing:
        raise RuntimeError(
            "Missing selected checkpoints: "
            + ", ".join(f"{condition}/{model}" for condition, model in missing))

    rows = []
    for key in expected:
        configuration, checkpoint = selected[key]
        name = f"{configuration['condition']['id']}__{configuration['id']}.pth"
        result = evaluate(
            configuration, checkpoint, device, cfg)
        rows.append(result)
        print(f"{result['condition_id']:>10} / {result['id']:<18} | "
              f"{result['energy_uj']:.4f} uJ | {result['nmse_db']:.2f} dB")
    output = output / "metrics"
    output.mkdir(parents=True, exist_ok=True)
    with open(output / "results.json", "w") as handle:
        json.dump(rows, handle, indent=2)
    with open(output / "results.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--device", default="cuda:0")
    main(parser.parse_args())
