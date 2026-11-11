"""Energies of the S-LCA comparison (Supplementary Note 4) under the energy model of the paper.

  python slca_energy.py --output runs/final

Reads <output>/metrics/slca_five_signals.json (written by compare_slca.py; its
NMSE, spike counts and latencies are kept), recomputes the energy of
  * S-LCA from the stored spike counts and step counts, and
  * S-LISTA (s = 28, L = 10, five training seeds) by re-running the five shared
    signals through SpikingLISTA.trace and counting ||xi||_0, ||z||_0, ||z||_1
    per layer,
checks for both that the operation-count prices reproduce the stored
energies, and writes
  <output>/metrics/slca_five_signals_energy.json   (energy_uj / energies_uj
      replaced by the new values; the stored ones kept as *_opcount)
  <output>/figures/slca_comparison_energy.{pdf,png}
No quantisation.  compare_slca.py and its output files are not changed.
"""

import argparse
import copy
import json
import statistics
from pathlib import Path

import torch

import energy_spec as es
from config import SLCA_CONDITION_ID, SLCA_SIGNAL_SEEDS, SLCA_SLISTA_ID, TRAINING_SEEDS
from data import generator, sparse_batch
from utils import checkpoint_path, load_model

RESULT = "slca_five_signals.json"
OUT_RESULT = "slca_five_signals_energy.json"
TOL = 1e-6


@torch.no_grad()
def slista_signal_energies(model, matrix, cfg, sparsity, device):
    """Full-model and operation-count energy (uJ) of each of the five shared signals."""
    spec, opcount, layers_all = [], [], []
    for seed in SLCA_SIGNAL_SEEDS:      # same signals as compare_slca.py
        _, y = sparse_batch(1, sparsity, matrix, generator(seed, device),
                            cfg.amplitude_min, cfg.amplitude_max)
        trace = model.trace(y.contiguous())
        layers = [{"xi0": float((trace["spikes"][l] != 0).sum()),
                   "z0": float((trace["codes"][l] != 0).sum()),
                   "z1": float(trace["codes"][l].abs().sum())}
                  for l in range(model.spiking_layers)]
        mask = float(trace["mask"].sum())
        spec.append(es.slista(model.m, model.n, layers, mask, es.SPEC).total(es.SPEC) * 1e-6)
        opcount.append(es.slista(model.m, model.n, layers, mask, es.OPCOUNT).total(es.OPCOUNT) * 1e-6)
        layers_all.append(layers)
    return spec, opcount, layers_all


def plot(root, d):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    b, s = d["slca"], d["slista"]
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.0), sharey=True)
    for i, (ax, key, label) in enumerate(zip(
            axes, ["latency_ms", "spikes_per_sample", "energy_uj"],
            ["Inference time (ms)", "Spike count", "Estimated energy (μJ)"])):
        ax.plot([r[key] for r in b], [r["nmse_db"] for r in b], "o-", color="#3976B8", ms=4, lw=1.5)
        ax.errorbar(statistics.mean(r[key] for r in s), statistics.mean(r["nmse_db"] for r in s),
                    xerr=statistics.stdev(r[key] for r in s), yerr=statistics.stdev(r["nmse_db"] for r in s),
                    fmt="s", color="#D84A3A", ms=7, capsize=3)
        ax.set_xscale("log")
        ax.set_xlabel(label, fontsize=18)
        ax.tick_params(labelleft=True, labelsize=18)
        values = [r[key] for r in b] + [r[key] for r in s]
        ax.set_xlim(min(values) / 2.5, max(values) * 2.5)
        ax.grid(alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(-.10, 1.04, chr(97 + i), transform=ax.transAxes, fontweight="bold", fontsize=21)
    axes[0].set_ylabel("NMSE (dB)", fontsize=18)
    axes[0].legend(handles=[Line2D([], [], color="#3976B8", marker="o", label="S-LCA"),
                            Line2D([], [], color="#D84A3A", marker="s", ls="none", label="S-LISTA")],
                   loc="lower right", frameon=False, fontsize=17, handlelength=1.5, labelspacing=.3)
    fig.tight_layout()
    out = root / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"slca_comparison_energy.{ext}", dpi=400, bbox_inches="tight")
    plt.close(fig)
    print("Saved", out / "slca_comparison_energy.pdf", flush=True)


def main(args):
    root = args.output.resolve()
    d = json.loads((root / "metrics" / RESULT).read_text())
    new = copy.deepcopy(d)
    m, n = d["metadata"]["m"], d["metadata"]["n"]
    worst = 0.0

    for row in new["slca"]:
        spec = [es.slca(m, n, row["steps"], c, es.SPEC).total(es.SPEC) * 1e-6 for c in row["spike_counts"]]
        opcount = [es.slca(m, n, row["steps"], c, es.OPCOUNT).total(es.OPCOUNT) * 1e-6 for c in row["spike_counts"]]
        worst = max(worst, max(abs(a - b) / b for a, b in zip(opcount, row["energies_uj"])))
        row["energies_opcount_uj"], row["energy_opcount_uj"] = row["energies_uj"], row["energy_uj"]
        row["energies_uj"], row["energy_uj"] = spec, statistics.mean(spec)

    device = torch.device(args.device)
    by_seed = {r["seed"]: r for r in new["slista"]}
    layer_log = {}
    for seed in TRAINING_SEEDS:
        model, matrix, job, cfg = load_model(
            checkpoint_path(root, seed, SLCA_CONDITION_ID, SLCA_SLISTA_ID), device)
        spec, opcount, layers = slista_signal_energies(model, matrix, cfg, job["condition"]["s"], device)
        row = by_seed[seed]
        worst = max(worst, max(abs(a - b) / b for a, b in zip(opcount, row["energies_uj"])))
        row["energies_opcount_uj"], row["energy_opcount_uj"] = row["energies_uj"], row["energy_uj"]
        row["energies_uj"], row["energy_uj"] = spec, statistics.mean(spec)
        layer_log[seed] = layers
        print(f"S-LISTA seed {seed}: {row['energy_uj']:.4f} uJ (operation-count {row['energy_opcount_uj']:.4f})")

    print(f"Self-check 1 (operation-count prices reproduce stored energies): max rel err {worst:.2e}")
    if worst > TOL:
        raise SystemExit("Self-check 1 failed; nothing written.")

    sl = statistics.mean(r["energy_uj"] for r in new["slista"])
    lca = new["slca"][-1]
    new["metadata"]["energy_model"] = ("Supplementary Note 2 energy model; "
                                       "E_AC 0.1, E_MAC 0.3, E_ADD 0.1, E_MUL 3.1 pJ, 2.5 pJ/B")
    new["metadata"]["energy_selfcheck1_max_rel_err"] = worst
    new["metadata"]["slista_layer_statistics"] = layer_log
    new["metadata"]["summary"] = {
        "slista_energy_uj": sl,
        "slca_100k_energy_uj": lca["energy_uj"],
        "energy_reduction_vs_slca_100k": 1 - sl / lca["energy_uj"],
        "opcount_energy_reduction_vs_slca_100k": 1 - statistics.mean(r["energy_opcount_uj"] for r in new["slista"]) / lca["energy_opcount_uj"],
    }
    (root / "metrics" / OUT_RESULT).write_text(json.dumps(new, indent=2))
    print(json.dumps(new["metadata"]["summary"], indent=2))
    plot(root, new)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, default=Path("runs/final"))
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    main(p.parse_args())
