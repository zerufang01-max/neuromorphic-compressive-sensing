"""Energy of every trained model of the synthetic task (energy model of Supplementary Note 2).

  python energy_eval.py --output runs/final

Steps
  1. For every seed and job of configs/paper.json: S-LISTA is run on the
     10,000-signal test set and the per-sample means of ||xi^(l)||_0,
     ||z^(l)||_0, ||z^(l)||_1 and ||m||_0 are collected; ANN energies depend
     on the shapes only.  Energy is linear in these means, so the mean energy
     per sample equals the energy of the means (batch = 1 accounting).
  2. Consistency checks (written to selfcheck.json; the script stops if
     check 1 fails).  Check 1 compares the operation-count prices with energy_uj of
     metrics/results_per_seed.csv (written by evaluate.py).
  3. INT8 post-training quantisation of the s = 28, L = 10 models of
     the four methods: every weight matrix is quantised per output channel,
     symmetric, to INT8 and dequantised; NMSE is re-evaluated with
     evaluate.evaluate (same test set).  Scalars (thresholds, eta, gamma,
     alpha) stay in floating point.

Outputs in <output>/energy/:
  energy_per_seed.csv, energy_summary.csv (mean +/- SD over seeds and the
  ratio to ANN LISTA at the same condition and depth), slista_layers.csv
  (internal: per-layer statistics), quantisation.csv, selfcheck.json and
  results_energy.json (metrics/results.json with energy_uj replaced by the
  new total; used by plot_fig2.py --results).
"""

import argparse
import copy
import csv
import json
import statistics
from pathlib import Path

import torch

import energy_spec as es
from config import TRAINING_SEEDS
from data import generator, sparse_batch
from evaluate import evaluate
from models import SpikingLISTA
from utils import DEFAULT_OUTPUT, PAPER_CONFIG, checkpoint_path, load_model, read_json, save_json

QUANT_CONDITION = "m141_s28"
QUANT_IDS = ("lista_l10", "alista_l10", "lamp_l10", "slista_l10")


@torch.no_grad()
def slista_statistics(model, matrix, job, cfg):
    """Per-sample means over the test set (same signals as evaluate.py)."""
    rng = generator(cfg.test_seed, matrix.device)
    layers = model.spiking_layers
    sums = {"xi0": [0.0] * layers, "z0": [0.0] * layers, "z1": [0.0] * layers}
    mask = 0.0
    remaining = cfg.test_samples
    while remaining:
        size = min(cfg.eval_batch_size, remaining)
        _, measurements = sparse_batch(size, job["condition"]["s"], matrix, rng,
                                       cfg.amplitude_min, cfg.amplitude_max)
        trace = model.trace(measurements)
        for layer in range(layers):
            sums["xi0"][layer] += float((trace["spikes"][layer] != 0).sum())
            sums["z0"][layer] += float((trace["codes"][layer] != 0).sum())
            sums["z1"][layer] += float(trace["codes"][layer].abs().sum())
        mask += float(trace["mask"].sum())
        remaining -= size
    samples = cfg.test_samples
    stats = [{key: sums[key][layer] / samples for key in sums} for layer in range(layers)]
    return stats, mask / samples


def energy_row(model, matrix, job, cfg):
    m, n, depth = model.m, model.n, job["depth"]
    row = {"seed": job["seed"], "condition_id": job["condition"]["id"], "id": job["id"],
           "method": job["method"], "depth": depth, "s": job["condition"]["s"]}
    layers = None
    if job["method"] == "slista":
        layers, mask = slista_statistics(model, matrix, job, cfg)
        spec = es.slista(m, n, layers, mask, es.SPEC)
        opcount = es.slista(m, n, layers, mask, es.OPCOUNT)
        row["support_per_sample"] = mask
    else:
        spec = es.ANN[job["method"]](m, n, depth, es.SPEC)
        opcount = es.ANN[job["method"]](m, n, depth, es.OPCOUNT)
    row.update(spec.summary(es.SPEC))
    row["opcount_total_uj"] = opcount.total(es.OPCOUNT) * 1e-6
    return row, layers


def quantise_int8(model):
    """Per-output-channel symmetric INT8 fake quantisation of every 2-D weight."""
    q = copy.deepcopy(model)
    with torch.no_grad():
        for name, tensor in list(q.state_dict().items()):
            if tensor.dim() != 2 or not tensor.is_floating_point():
                continue
            scale = tensor.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 127.0
            tensor.copy_(torch.clamp(torch.round(tensor / scale), -127, 127) * scale)
    return q


def summarise(rows):
    keys = ("compute_uj", "weight_uj", "state_uj", "memory_uj", "total_uj", "opcount_total_uj")
    groups = {}
    for row in rows:
        groups.setdefault((row["condition_id"], row["id"]), []).append(row)
    out = []
    for (condition, model_id), group in groups.items():
        first = group[0]
        entry = {"condition_id": condition, "id": model_id, "method": first["method"],
                 "depth": first["depth"], "s": first["s"], "seeds": len(group)}
        for key in keys:
            values = [r[key] for r in group]
            entry[key] = statistics.mean(values)
            entry[key + "_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        out.append(entry)
    reference = {(e["condition_id"], e["depth"]): e["total_uj"] for e in out if e["method"] == "lista"}
    for entry in out:
        ref = reference.get((entry["condition_id"], entry["depth"]))
        entry["ratio_to_lista_same_depth"] = entry["total_uj"] / ref if ref else None
    return sorted(out, key=lambda e: (e["s"], e["method"], e["depth"]))


def write_csv(rows, path):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(args):
    device = torch.device(args.device)
    root = Path(args.output).resolve()
    out = root / "energy"
    out.mkdir(parents=True, exist_ok=True)
    jobs = read_json(args.config)["jobs"]

    old = {}
    old_path = root / "metrics" / "results_per_seed.csv"
    for r in csv.DictReader(open(old_path)):
        old[(int(r["seed"]), r["condition_id"], r["id"])] = float(r["energy_uj"])

    rows, layer_rows, quant_rows = [], [], []
    for seed in args.seeds:
        for job in jobs:
            path = checkpoint_path(root, seed, job["condition"]["id"], job["id"])
            model, matrix, trained, cfg = load_model(path, device)
            row, layers = energy_row(model, matrix, trained, cfg)
            row["stored_energy_uj"] = old[(seed, row["condition_id"], row["id"])]
            rows.append(row)
            if layers:
                for index, layer in enumerate(layers, start=1):
                    layer_rows.append({"seed": seed, "condition_id": row["condition_id"],
                                       "id": row["id"], "layer": index, **layer})
            if row["condition_id"] == QUANT_CONDITION and row["id"] in QUANT_IDS:
                fp = evaluate(model, matrix, trained, cfg)["nmse_db"]
                q = evaluate(quantise_int8(model), matrix, trained, cfg)["nmse_db"]
                quant_rows.append({"seed": seed, "id": row["id"], "fp32_nmse_db": fp,
                                   "int8_nmse_db": q, "change_db": q - fp})
            print(f"seed {seed} {row['condition_id']} {row['id']:<11} "
                  f"total {row['total_uj']:.4f} uJ (evaluate.py {row['stored_energy_uj']:.4f}, "
                  f"operation-count check {row['opcount_total_uj']:.4f})", flush=True)

    # Consistency checks.
    worst = max(abs(r["opcount_total_uj"] - r["stored_energy_uj"]) / r["stored_energy_uj"] for r in rows)
    checks = es.selfcheck()
    checks["check1_all_models_max_rel_err_vs_results_per_seed"] = worst
    checks["check1_passed"] = worst < 1e-6
    save_json(out / "selfcheck.json", checks)
    print(json.dumps(checks, indent=2))
    if not checks["check1_passed"] or not checks["check2_u2_equals_u1"]:
        raise SystemExit("Self-check failed; new energies were not written.")

    write_csv(rows, out / "energy_per_seed.csv")
    summary = summarise(rows)
    write_csv(summary, out / "energy_summary.csv")
    write_csv(layer_rows, out / "slista_layers.csv")
    if quant_rows:
        qsum = []
        for model_id in QUANT_IDS:
            group = [r for r in quant_rows if r["id"] == model_id]
            if group:
                qsum.append({"id": model_id, "seeds": len(group),
                             **{k: statistics.mean(r[k] for r in group)
                                for k in ("fp32_nmse_db", "int8_nmse_db", "change_db")},
                             "change_db_max_abs": max(abs(r["change_db"]) for r in group)})
        write_csv(quant_rows, out / "quantisation_per_seed.csv")
        write_csv(qsum, out / "quantisation.csv")

    # Fig. 2 input with the new energies (NMSE and everything else unchanged).
    totals = {(e["condition_id"], e["id"]): e for e in summary}
    results = read_json(root / "metrics" / "results.json")
    for r in results:
        entry = totals[(r["condition_id"], r["id"])]
        r["energy_uj"], r["energy_uj_std"] = entry["total_uj"], entry["total_uj_std"]
    save_json(out / "results_energy.json", results)
    print(f"Wrote {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--config", type=Path, default=PAPER_CONFIG)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(TRAINING_SEEDS))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    main(parser.parse_args())
