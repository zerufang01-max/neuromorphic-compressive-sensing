"""Bit-budget experiment: dense transmission with fewer binary channels per
frame (48, 64, 96, 128, 192; the formal models use 256), for S-LISTA and
Vanilla SNN with their formal settings unchanged.

Each job trains the reconstruction model and the downstream classifier and
evaluates AWGN and frozen Rayleigh over -5...20 dB (main.py --recon-only
--downstream), into outputs_budget/.  The 256-channel, AER and ANN reference
points are read from the formal results in outputs/.

    python budget.py --seeds 42 --gpus 0 1 2 3 4 5     # train + evaluate
    python budget.py --seeds 42 --summary               # table + figure only
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs_budget"
FORMAL = ROOT / "outputs"
CHANNELS = [48, 64, 96, 128, 192]
FRAMES = 25
MODELS = {"slista": "S-LISTA", "scnn": "Vanilla SNN"}
SNRS = [-5, 0, 5, 10, 20]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--gpus", nargs="+", type=int, default=[0])
    parser.add_argument("--jobs-per-gpu", type=int, default=1)
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--classifier-only", action="store_true",
                        help="reuse the reconstruction models; fail instead of retraining one")
    return parser.parse_args()


def key_of(model, channels):
    return f"{model}_dense_c{channels}"


def result_file(key, seed):
    return OUT / f"recon_point_{key}_seed{seed}.csv"


def protocol():
    """Downstream-classifier protocol tag of the current configuration."""
    sys.path.insert(0, str(ROOT))
    from config import Config
    return Config.classifier_protocol()


def done(key, seed):
    """Finished with a downstream classifier of the current protocol (a job
    whose classifier used another protocol is rerun; the reconstruction
    model is reused, only the classifier is retrained)."""
    path = result_file(key, seed)
    if not path.is_file():
        return False
    row = pd.read_csv(path).iloc[0]
    return (pd.notna(row.get("Robust Acc (%)", np.nan))
            and protocol() in str(row.get("Classifier Checkpoint", "")))


def run(args):
    jobs = [(key_of(m, c), s) for c in CHANNELS for m in MODELS for s in args.seeds
            if not done(key_of(m, c), s)]
    print(f"{len(jobs)} jobs: {[f'{k} seed{s}' for k, s in jobs]}", flush=True)
    if args.dry_run:
        return True
    OUT.mkdir(exist_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)
    slots = [g for g in args.gpus for _ in range(args.jobs_per_gpu)]
    running, failures = [], []
    while jobs or running:
        busy = {j["slot"] for j in running}
        for slot, gpu in enumerate(slots):
            if not jobs or slot in busy:
                continue
            key, seed = jobs.pop(0)
            log = (OUT / "logs" / f"{key}_seed{seed}.log").open("a")
            cmd = [sys.executable, "-u", str(ROOT / "main.py"), "--gpu", "0",
                   "--experiment", key, "--seed", str(seed), "--recon-only",
                   "--downstream", "--results-dir", str(OUT)]
            if args.classifier_only:
                cmd.append("--classifier-only")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
            proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            running.append({"proc": proc, "slot": slot, "key": key, "seed": seed, "log": log})
            print(f"[Start] GPU {gpu}: {key} seed{seed}", flush=True)
        time.sleep(10)
        still = []
        for job in running:
            code = job["proc"].poll()
            if code is None:
                still.append(job)
                continue
            job["log"].close()
            if code == 0 and done(job["key"], job["seed"]):
                print(f"[Done] {job['key']} seed{job['seed']}", flush=True)
            else:
                failures.append(f"{job['key']} seed{job['seed']}")
                print(f"[Failed] {job['key']} seed{job['seed']} -> "
                      f"{OUT / 'logs' / (job['key'] + '_seed%d.log' % job['seed'])}", flush=True)
        running = still
    if failures:
        print(f"{len(failures)} failed: {failures}", flush=True)
    return not failures


# ------------------------------------------------------------------ summary
def budget_rows(seeds):
    rows = []
    for c in CHANNELS:
        for m, name in MODELS.items():
            for s in seeds:
                path = result_file(key_of(m, c), s)
                if not path.is_file():
                    continue
                r = pd.read_csv(path).iloc[0]
                if not done(key_of(m, c), s):
                    continue
                row = {"Key": key_of(m, c), "Model": name, "Curve": "dense", "Channels": c, "Seed": s,
                       "Bits/Sample": float(r["Bits/Sample"]),
                       "Firing Rate": float(r.get("Firing Rate", np.nan))}
                for ch in ("AWGN", "Rayleigh"):
                    for x in SNRS:
                        row[f"{ch} NMSE @{x}"] = float(r[f"{ch} NMSE @{x:g}"])
                        row[f"{ch} Acc @{x}"] = float(r[f"{ch} Acc @{x:g}"])
                rows.append(row)
    return rows


def formal_rows(seeds):
    """256-channel dense, ANN and Vanilla-SNN AER points from outputs/."""
    rows = []
    awgn_files = {"slista_dense": "S-LISTA", "scnn_dense": "Vanilla SNN",
                  "ann_lista": "ANN LISTA", "ann_lstm": "ANN LSTM"}
    for key, name in awgn_files.items():
        for s in seeds:
            a = FORMAL / f"formal_point_{key}_seed{s}.csv"
            r = FORMAL / "rayleigh_frozen" / f"rayleigh_{key}_seed{s}.csv"
            if not (a.is_file() and r.is_file()):
                continue
            a = pd.read_csv(a).set_index("Eval SNR (dB)")
            r = pd.read_csv(r).set_index("Eval SNR (dB)")
            row = {"Key": key, "Model": name, "Curve": "dense" if "ann" not in key else "ann",
                   "Channels": 256, "Seed": s, "Bits/Sample": float(a.loc[10.0, "Bits/Sample"])}
            for x in SNRS:
                row[f"AWGN NMSE @{x}"] = float(a.loc[float(x), "Mean NMSE (dB)"])
                row[f"AWGN Acc @{x}"] = float(a.loc[float(x), "Accuracy (%)"])
                row[f"Rayleigh NMSE @{x}"] = float(r.loc[float(x), "Mean NMSE (dB)"])
                row[f"Rayleigh Acc @{x}"] = float(r.loc[float(x), "Accuracy (%)"])
            rows.append(row)
    aer = FORMAL / "performance_bit_rayleigh_per_seed.csv"
    if aer.is_file():
        d = pd.read_csv(aer)
        d = d[~d["Display Model"].eq("S-LISTA") & d["Seed"].isin(seeds)]
        for (key, s), part in d.groupby(["Experiment Key", "Seed"]):
            part = part.set_index("Eval SNR (dB)")
            row = {"Key": key, "Model": "Vanilla SNN", "Curve": "aer", "Channels": 256, "Seed": s,
                   "Bits/Sample": float(part["Bits/Sample"].iloc[0])}
            for x in SNRS:
                row[f"Rayleigh NMSE @{x}"] = float(part.loc[float(x), "Mean NMSE (dB)"])
                row[f"Rayleigh Acc @{x}"] = float(part.loc[float(x), "Accuracy (%)"])
            rows.append(row)
    # AWGN 10 dB of the AER points (their only AWGN evaluation).
    awgn = FORMAL / "performance_bit_per_seed.csv"
    if awgn.is_file():
        d = pd.read_csv(awgn)
        d = d[~d["Display Model"].eq("S-LISTA") & d["Seed"].isin(seeds)
              & np.isclose(d["Eval SNR (dB)"], 10)]
        lookup = {(r["Experiment Key"], int(r["Seed"])): r for r in d.to_dict("records")}
        for row in rows:
            hit = lookup.get((row["Key"], int(row["Seed"]))) if row["Curve"] == "aer" else None
            if hit is not None:
                row["AWGN NMSE @10"] = float(hit["Mean NMSE (dB)"])
                row["AWGN Acc @10"] = float(hit["Accuracy (%)"])
    return rows


def summarize(args):
    rows = budget_rows(args.seeds) + formal_rows(args.seeds)
    if not rows:
        print("No results yet.")
        return
    table = pd.DataFrame(rows)
    agg = (table.groupby(["Key", "Model", "Curve"]).mean(numeric_only=True)
           .reset_index().drop(columns=["Seed"]))
    agg = agg.sort_values(["Curve", "Model", "Bits/Sample"])
    OUT.mkdir(exist_ok=True)
    table.to_csv(OUT / "budget_per_seed.csv", index=False)
    agg.to_csv(OUT / "budget_summary.csv", index=False)
    pd.set_option("display.width", 250)
    for ch in ("Rayleigh", "AWGN"):
        cols = ["Model", "Curve", "Bits/Sample"] + \
            [f"{ch} NMSE @{x}" for x in SNRS] + [f"{ch} Acc @{x}" for x in SNRS]
        print(f"\n=== {ch} (seed mean over {args.seeds}; NMSE dB, accuracy %) ===")
        print(agg[[c for c in cols if c in agg]].to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    plot(agg, args.seeds, table)
    # Paper-style figures (same design as the performance-bit scatter).
    sys.path.insert(0, str(ROOT))
    import plot_results
    plot_results.plot_bit_budget(table, str(OUT), "AWGN", 10)
    plot_results.plot_bit_budget(table, str(OUT), "Rayleigh", 20, suffix="_rayleigh20")
    plot_results.plot_bit_budget(table, str(OUT), "Rayleigh", 5, suffix="_rayleigh5")


def plot(agg, seeds, table=None):
    """Main figure: AWGN 10 dB (training channel), seed mean +- std; ANN
    (51200 bits) as dashed reference lines.  A second figure keeps the
    Rayleigh panels for the supplement."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    spread = (table.groupby(["Key", "Model", "Curve"]).std(numeric_only=True).reset_index()
              if table is not None and table["Seed"].nunique() > 1 else None)
    style = {("S-LISTA", "dense"): ("#c0392b", "o", "-", "S-LISTA"),
             ("Vanilla SNN", "dense"): ("#2c6fbb", "s", "-", "Vanilla SNN"),
             ("Vanilla SNN", "aer"): ("#2c6fbb", "^", "--", "Vanilla SNN (AER)")}
    refs = {"ANN LISTA": "#555555", "ANN LSTM": "#999999"}

    def panel(ax, ch, snr, metric):
        col = f"{ch} {metric} @{snr}"
        for (model, curve), (color, marker, ls, label) in style.items():
            part = agg[(agg["Model"] == model) & (agg["Curve"] == curve)].sort_values("Bits/Sample")
            if part.empty or col not in part or part[col].isna().all():
                continue
            err = None
            if spread is not None:
                err = part[["Key"]].merge(spread, on="Key", how="left")[col].to_numpy()
            ax.errorbar(part["Bits/Sample"], part[col], yerr=err, marker=marker, linestyle=ls,
                        color=color, label=label, capsize=2, markersize=5)
        for model, color in refs.items():
            part = agg[agg["Model"] == model]
            if not part.empty and col in part and part[col].notna().all():
                ax.axhline(float(part[col].iloc[0]), color=color, linestyle=":", linewidth=1.2,
                           label=f"{model} (51200 bits)")
        ax.set_xlabel("Bits / sample")
        ax.grid(alpha=0.3)

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    panel(ax[0], "AWGN", 10, "NMSE")
    panel(ax[1], "AWGN", 10, "Acc")
    ax[0].set_ylabel("NMSE (dB)")
    ax[1].set_ylabel("Accuracy (%)")
    ax[0].legend(fontsize=7)
    fig.suptitle(f"SHD bit budget, AWGN 10 dB (seeds {seeds}; mean +- std)")
    fig.tight_layout()
    fig.savefig(OUT / "budget_tradeoff.png", dpi=150)
    plt.close(fig)

    panels = [("Rayleigh", 20), ("Rayleigh", 10), ("Rayleigh", 5)]
    fig, ax = plt.subplots(2, len(panels), figsize=(4 * len(panels), 6.4))
    for j, (ch, snr) in enumerate(panels):
        panel(ax[0, j], ch, snr, "NMSE")
        panel(ax[1, j], ch, snr, "Acc")
        ax[0, j].set_title(f"{ch} {snr} dB")
    ax[0, 0].set_ylabel("NMSE (dB)")
    ax[1, 0].set_ylabel("Accuracy (%)")
    ax[0, 0].legend(fontsize=7)
    fig.suptitle(f"SHD bit budget under unseen Rayleigh fading (seeds {seeds})")
    fig.tight_layout()
    fig.savefig(OUT / "budget_rayleigh.png", dpi=130)
    plt.close(fig)
    print(f"\nSaved: {OUT / 'budget_summary.csv'}, {OUT / 'budget_tradeoff.png'}, "
          f"{OUT / 'budget_rayleigh.png'}")


def main():
    args = parse_args()
    if args.summary:
        summarize(args)
        return
    ok = run(args)
    summarize(args)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
