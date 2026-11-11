"""DVS-Gesture: run, evaluate and summarise.

    python main.py run --group baselines --gpus 0 1              # baseline weights -> eval grid
    python main.py run --group slista budget --gpus 0 1          # S-LISTA + bit budget, one queue
    python main.py analysis --gpu 0                              # seed 42: samples, classes, energy
    python main.py summarize                                     # outputs/tables/*.csv

Every (experiment, seed) runs in its own process (``main.py worker``) and writes
to outputs/runs/<experiment>/seed<k>/: spec.json, recon.pth, classifier.pth,
history.csv, eval.csv (AWGN and Rayleigh, -5..20 dB), summary.json, log.txt.
A finished run is skipped unless its settings changed.  S-LISTA and the Vanilla
bit-budget points are trained here; the baselines (Vanilla SNN, ANN LISTA, ANN
LSTM) are trained by baselines/train.py and their reconstruction weights are
loaded.  The downstream classifier is trained here for every method, baselines
included, with one protocol (random translation + label smoothing,
config.CLASSIFIER_AUGMENT), so that all methods are classified the same way.
"""

import argparse
import csv
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import config as C
import experiments as E


# ----------------------------------------------------------------- helpers
def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys = []
    for row in rows:
        keys += [k for k in row if k not in keys]
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def spec_done(spec):
    """True if summary.json exists for exactly this spec (and has a classifier if requested)."""
    path = Path(spec["run_dir"]) / "summary.json"
    if not path.is_file():
        return False
    try:
        summary = json.loads(path.read_text())
    except json.JSONDecodeError:
        return False
    if summary.get("fingerprint") != E.recon_fingerprint(spec):
        return False
    wants_classifier = spec["with_classifier"] and not spec["recon_only"]
    if not wants_classifier:
        return True
    return bool(summary.get("has_classifier")) and summary.get("classifier_settings") == spec["classifier_settings"]


# ------------------------------------------------------------------ worker
def parameter_counts(experiment, model, sensor):
    excluded = ("semantic_head.",)
    if experiment.method == "ann_lista":
        excluded += ("P_snn.", "PD_snn.", "S_k.0.")
    recon = sum(p.numel() for n, p in model.named_parameters() if not n.startswith(excluded))
    sender = sum(p.numel() for p in sensor.parameters())
    return {"params_sender": int(sender), "params_recon": int(recon), "params_total": int(sender + recon)}


def load_run_models(experiment, spec, device, log=print):
    """Build and load the reconstruction (and classifier) of a finished run or a baseline."""
    import torch
    import trainer as T
    run_dir = Path(spec["run_dir"])
    stored = run_dir / "spec.json"
    if experiment.source == "train" and stored.is_file():
        spec = json.loads(stored.read_text())       # settings the weights were trained with
    model, sensor = T.build_models(experiment, spec["slista"], device)
    if experiment.source == "baseline":
        recon_path, _, _ = E.baseline_weights(experiment.baseline_key, spec["seed"])
    else:
        recon_path = run_dir / "recon.pth"
    cls_path = run_dir / "classifier.pth"      # the same classifier protocol for every method
    state = torch.load(recon_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    sensor.load_state_dict(state["sensor"], strict=True)
    spec = dict(spec, fingerprint=E.recon_fingerprint(spec))
    classifier = T.cached_classifier(cls_path, spec, device)
    log(f"loaded {recon_path}" + (f" and {cls_path}" if classifier is not None else " (no classifier)"))
    return model, sensor, classifier, state


def run_worker(spec_path, gpu):
    import torch
    device = C.setup_torch(gpu)
    import data
    import trainer as T

    spec = json.loads(Path(spec_path).read_text())
    spec["fingerprint"] = E.recon_fingerprint(spec)
    experiment = E.experiment_from_spec(spec)
    run_dir = Path(spec["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "spec.json").write_text(json.dumps(spec, indent=2))
    seed = int(spec["seed"])
    started = time.time()

    def log(message):
        print(f"[{time.strftime('%H:%M:%S')}] {experiment.key} seed{seed} | {message}", flush=True)

    T.set_seed(seed)
    loaders = data.make_loaders(seed)
    recon_path, cls_path = run_dir / "recon.pth", run_dir / "classifier.pth"

    if experiment.source == "baseline":       # weights from baselines/, only evaluated
        model, sensor, _, state = load_run_models(experiment, spec, device, log)
    else:
        model, sensor = T.build_models(experiment, spec["slista"], device)
        valid = False
        if recon_path.is_file():
            state = torch.load(recon_path, map_location="cpu", weights_only=False)
            valid = state.get("fingerprint") == spec["fingerprint"]
            if not valid:
                log("existing recon.pth has different settings -> retraining")
        if not valid:
            if cls_path.is_file():
                cls_path.unlink()
            T.train_reconstruction(spec, experiment, model, sensor, loaders, device, recon_path,
                                   run_dir / "history.csv", log)
        state = torch.load(recon_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"], strict=True)
        sensor.load_state_dict(state["sensor"], strict=True)
    classifier = None
    if spec["with_classifier"] and not spec["recon_only"]:
        classifier = T.cached_classifier(cls_path, spec, device)
        if classifier is None:
            classifier = T.train_classifier(spec, experiment, model, sensor, loaders, device, cls_path, log)

    rows = T.robustness_grid(model, sensor, loaders[1], experiment, device, classifier, log=log)
    for row in rows:
        row.update({"experiment": experiment.key, "tag": spec["tag"], "seed": seed,
                    "method": experiment.display})
    write_csv(run_dir / "eval.csv", rows)

    summary = {"experiment": experiment.key, "tag": spec["tag"], "seed": seed,
               "fingerprint": spec["fingerprint"], "has_classifier": classifier is not None,
               "classifier_settings": spec["classifier_settings"] if classifier is not None else None,
               "source": experiment.source, "best_epoch": state.get("epoch"),
               "calibration": state.get("calibration"), "slista": spec["slista"],
               "minutes": (time.time() - started) / 60.0, "code_version": C.CODE_VERSION}
    summary.update(T.robust_summary(rows))
    awgn10 = next(r for r in rows if r["channel"] == "awgn" and r["snr_db"] == C.TRAIN_SNR_DB)
    for key in ("tx_rate", "rx_rate", "support_per_frame", "rx_ac_per_sample", "bits"):
        if key in awgn10:
            summary[f"awgn10_{key}"] = awgn10[key]
    if experiment.method == "slista":
        summary["theta"] = [float(v) for v in model.theta.detach().cpu()]
        summary["eta"] = float(model.eta)
    summary.update(parameter_counts(experiment, model, sensor))
    T.atomic_json(run_dir / "summary.json", summary)
    log(f"done: robust NMSE {summary['robust_nmse']:.3f} dB | robust head {summary['robust_head_acc']:.2f}% "
        f"| robust acc {summary['robust_acc']:.2f}% | {summary['minutes']:.1f} min")


# -------------------------------------------------------------- dispatcher
def dispatch(specs, gpus, per_gpu=1, spec_dir=None, force=False):
    """Run specs in separate processes, ``per_gpu`` at a time on each GPU."""
    pending = [s for s in specs if force or not spec_done(s)]
    print(f"[dispatch] {len(pending)} of {len(specs)} runs to do on GPUs {gpus} x {per_gpu}", flush=True)
    if not pending:
        return []
    jobs = queue.Queue()
    for spec in pending:
        jobs.put(spec)
    failures, lock = [], threading.Lock()

    def worker(gpu):
        while True:
            try:
                spec = jobs.get_nowait()
            except queue.Empty:
                return
            run_dir = Path(spec["run_dir"])
            run_dir.mkdir(parents=True, exist_ok=True)
            spec_path = run_dir / "spec_request.json"
            spec_path.write_text(json.dumps(spec, indent=2))
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1")
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "worker",
                       "--spec", str(spec_path), "--gpu", "0"]
            label = f"{spec['tag']} seed{spec['seed']}"
            print(f"[start] GPU {gpu}: {label}", flush=True)
            with (run_dir / "log.txt").open("a") as handle:
                code = subprocess.run(command, cwd=str(C.ROOT), env=env, stdout=handle,
                                      stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL).returncode
            with lock:
                if code:
                    failures.append(str(run_dir / "log.txt"))
                print(f"[{'FAIL' if code else 'done'}] GPU {gpu}: {label}", flush=True)

    threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in gpus for _ in range(per_gpu)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        print("[dispatch] failed runs (see logs):\n  " + "\n  ".join(failures), flush=True)
    return failures


def group_keys(groups):
    if "all" in groups:
        groups = ["baselines", "slista", "budget"]
    return list(dict.fromkeys(k for g in groups for k in E.GROUPS[g]))


def formal_specs(groups, seeds, epochs=None):
    groups = [groups] if isinstance(groups, str) else list(groups)
    keys = group_keys(groups)
    needs_slista = any(E.REGISTRY[k].method == "slista" for k in keys)
    if needs_slista and not C.SLISTA_CONFIG_FILE.is_file():
        raise FileNotFoundError(f"{C.SLISTA_CONFIG_FILE} is missing "
                                "(or pass --allow-defaults to use config.SLISTA_DEFAULTS)")
    return [E.make_spec(key, seed, E.formal_run_dir(key, seed), epochs=epochs) for key in keys for seed in seeds]


# ---------------------------------------------------------------- analysis
def run_analysis(gpu, seed=42):
    """Seed-42 analysis: per-sample NMSE (Rayleigh 20 dB), class-wise NMSE, AER bits, energy."""
    device = C.setup_torch(gpu)
    import data
    import op_counter as energy
    import trainer as T
    from channels import make_channel

    _, test_loader = data.make_loaders(seed)
    out = C.OUTPUTS / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    samples, classes, bits_rows, energy_rows = [], [], [], []
    keys = list(dict.fromkeys(E.MAIN_KEYS + E.AER_BIT_KEYS + E.ENERGY_KEYS))
    for key in keys:
        experiment = E.REGISTRY[key]
        spec = E.make_spec(key, seed, E.formal_run_dir(key, seed))
        model, sensor, classifier, _ = load_run_models(experiment, spec, device)
        if key in E.MAIN_KEYS:
            result = T.evaluate(model, sensor, make_channel("rayleigh", 20.0, experiment.transport), test_loader,
                                experiment, device, classifier=classifier, per_sample=True)
            for i, row in enumerate(result["per_sample"]):
                samples.append({"method": experiment.display, "experiment": key, "sample": i, **row})
        if key in E.AER_BIT_KEYS:
            result = T.evaluate(model, sensor, make_channel("awgn", 10.0, experiment.transport), test_loader,
                                experiment, device, per_sample=True)
            for i, row in enumerate(result["per_sample"]):
                bits_rows.append({"method": experiment.display, "experiment": key, "sample": i,
                                  "class": row["class"], "aer_bits": row["aer_bits"]})
        if key in E.ENERGY_KEYS or key in ("slista_dense", "vanilla_dense"):
            energy_rows.append(energy.count_energy(experiment, model, sensor, test_loader, device))
            print(energy_rows[-1], flush=True)
    by_class = {}
    for row in samples:
        by_class.setdefault((row["method"], row["class"]), []).append(row["nmse_db"])
    for (method, cls), values in sorted(by_class.items()):
        classes.append({"method": method, "class": cls, "nmse_db": sum(values) / len(values), "n": len(values)})
    write_csv(out / f"samples_rayleigh20_seed{seed}.csv", samples)
    write_csv(out / f"classes_rayleigh20_seed{seed}.csv", classes)
    write_csv(out / f"aer_bits_seed{seed}.csv", bits_rows)
    write_csv(out / f"energy_seed{seed}.csv", energy_rows)
    print(f"[analysis] {out}", flush=True)


# ---------------------------------------------------------------- summaries
def _mean_std(values):
    values = [float(v) for v in values if v not in ("", None) and str(v) != "nan"]
    if not values:
        return float("nan"), float("nan"), 0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1) if len(values) > 1 else 0.0
    return mean, var ** 0.5, len(values)


def collect_eval(keys, seeds):
    rows, missing = [], []
    for key in keys:
        for seed in seeds:
            path = E.formal_run_dir(key, seed) / "eval.csv"
            if path.is_file():
                # the folder decides the experiment (a run copied from another key keeps its rows)
                rows += [dict(r, experiment=key) for r in read_csv(path)]
            else:
                missing.append(f"{key} seed{seed}")
    return rows, missing


def table(rows, group_keys, metrics):
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in group_keys), []).append(row)
    out = []
    for key, part in groups.items():
        entry = dict(zip(group_keys, key))
        entry["n_seeds"] = len({r["seed"] for r in part})
        for metric in metrics:
            mean, std, _ = _mean_std(r.get(metric, "") for r in part)
            entry[f"{metric}_mean"], entry[f"{metric}_std"] = mean, std
        out.append(entry)
    return out


def summarize(seeds):
    tables = C.OUTPUTS / "tables"
    metrics = ["nmse", "nmse_full", "nmse_framewise", "acc", "head_acc", "bits", "tx_rate", "rx_rate"]
    keys = [k for g in ("baselines", "slista", "budget") for k in E.GROUPS[g]]
    keys += [k for k in E.GROUPS["semweight"] if E.formal_run_dir(k, seeds[0]).parent.is_dir()]
    rows, missing = collect_eval(keys, seeds)
    for row in rows:
        row["snr_db"] = float(row["snr_db"])
    if missing:
        print(f"[summarize] missing runs ({len(missing)}): " + ", ".join(missing[:12])
              + (" ..." if len(missing) > 12 else ""))
    scan_keys = E.MAIN_KEYS + E.SEMANTIC_KEYS + E.GROUPS["semweight"]
    snr = table([r for r in rows if r["experiment"] in scan_keys], ["experiment", "channel", "snr_db"], metrics)
    write_csv(tables / "snr_scans_mean_std.csv", snr)
    write_csv(tables / "snr_scans_per_seed.csv", [r for r in rows if r["experiment"] in scan_keys])
    main = [r for r in snr if r["channel"] == "rayleigh" and r["snr_db"] == 20.0]
    write_csv(tables / "table1_rayleigh20.csv", main)
    ablation = table([r for r in rows if r["experiment"] in E.ABLATION_KEYS
                      and r["channel"] == "rayleigh" and r["snr_db"] == 20.0], ["experiment"], metrics)
    write_csv(tables / "table2_ablation_rayleigh20.csv", ablation)
    budget_rows = []
    for method in ("slista", "vanilla"):
        for c in C.BUDGET_CHANNELS:
            key = E.budget_key(method, c)
            if not any(r["experiment"] == key for r in rows) and key == "vanilla_dense_c4":
                key = "vanilla_dense"     # C = 4 not retrained: baseline weights
            for r in rows:
                if r["experiment"] == key:
                    budget_rows.append({**r, "budget_method": method, "channels": c})
    write_csv(tables / "budget_per_seed.csv", budget_rows)
    write_csv(tables / "budget_mean_std.csv",
              table(budget_rows, ["budget_method", "channels", "channel", "snr_db"], metrics))
    robust = []
    for key in keys:
        per_seed = {}
        for r in rows:
            if r["experiment"] == key and ((r["channel"] == "awgn" and r["snr_db"] <= C.ROBUST_AWGN_MAX_DB)
                                           or r["channel"] == "rayleigh"):
                per_seed.setdefault(r["seed"], []).append(r)
        if not per_seed:
            continue
        entry = {"experiment": key, "n_seeds": len(per_seed)}
        for metric in ("nmse", "acc", "head_acc"):
            seed_means = [_mean_std(r.get(metric, "") for r in part)[0] for part in per_seed.values()]
            entry[f"robust_{metric}_mean"], entry[f"robust_{metric}_std"], _ = _mean_std(seed_means)
        robust.append(entry)
    write_csv(tables / "robustness_mean_std.csv", robust)
    semantic_weight_table(rows, robust, tables)
    print(f"[summarize] tables in {tables}")
    for entry in main:
        print(f"  Table 1 {entry['experiment']:22s} NMSE {entry['nmse_mean']:7.2f} +- {entry['nmse_std']:.2f}"
              f" | acc {entry['acc_mean']:6.2f} +- {entry['acc_std']:.2f} | bits {entry['bits_mean']:.0f}")
    for entry in robust:
        print(f"  robust  {entry['experiment']:22s} NMSE {entry['robust_nmse_mean']:7.2f} | acc "
              f"{entry['robust_acc_mean']:6.2f} | head {entry['robust_head_acc_mean']:6.2f}")


def semantic_weight_table(rows, robust, tables):
    """Task-supervision weight study: one line per weight (0 = slista_dense_no_semantic)."""
    weights = {"slista_dense_no_semantic": 0.0, "slista_dense": C.SLISTA_TRAINING["semantic_weight"]}
    weights.update({E.semweight_key(w): w for w in C.SEMANTIC_SWEEP})
    robust_by_key = {r["experiment"]: r for r in robust}
    points = [("awgn", 10.0), ("rayleigh", 20.0)]
    low = {"awgn": (-5.0, 2.0), "rayleigh": (-5.0, 5.0)}      # low-SNR ranges (accuracy averaged)
    out = []
    for key, weight in sorted(weights.items(), key=lambda kv: (kv[1], kv[0])):
        part = [r for r in rows if r["experiment"] == key]
        if not part:
            continue
        entry = {"semantic_weight": weight, "experiment": key, "n_seeds": len({r["seed"] for r in part})}
        rob = robust_by_key.get(key, {})
        for metric in ("nmse", "acc", "head_acc"):
            entry[f"robust_{metric}"] = rob.get(f"robust_{metric}_mean")
            entry[f"robust_{metric}_std"] = rob.get(f"robust_{metric}_std")
        for channel, snr in points:
            for metric in ("nmse", "acc"):
                vals = [float(r[metric]) for r in part if r["channel"] == channel and r["snr_db"] == snr]
                entry[f"{channel}{snr:g}_{metric}"], entry[f"{channel}{snr:g}_{metric}_std"], _ = _mean_std(vals)
        for channel, (lo, hi) in low.items():
            per_seed = {}
            for r in part:
                if r["channel"] == channel and lo <= r["snr_db"] <= hi:
                    per_seed.setdefault(r["seed"], []).append(float(r["acc"]))
            means = [sum(v) / len(v) for v in per_seed.values()]
            entry[f"{channel}_low_acc"], entry[f"{channel}_low_acc_std"], _ = _mean_std(means)
        out.append(entry)
    if out:
        write_csv(tables / "semantic_weight.csv", out)
        print("[summarize] task-supervision weight (robust NMSE / acc | Rayleigh 20 NMSE / acc | "
              "low-SNR acc AWGN -5..2 / Rayleigh -5..5)")
        for e in out:
            print(f"  w={e['semantic_weight']:<5g} {e['experiment']:26s} n={e['n_seeds']} "
                  f"{e['robust_nmse']:7.2f} / {e['robust_acc']:5.2f} | {e['rayleigh20_nmse']:7.2f} / "
                  f"{e['rayleigh20_acc']:5.2f} | {e['awgn_low_acc']:5.2f} / {e['rayleigh_low_acc']:5.2f}")


# --------------------------------------------------------------------- CLI
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    w = sub.add_parser("worker")
    w.add_argument("--spec", required=True)
    w.add_argument("--gpu", type=int, default=0)
    r = sub.add_parser("run")
    r.add_argument("--group", nargs="+", choices=["baselines", "slista", "budget", "semweight", "all"], required=True,
                   help="one or more groups; all runs share one queue")
    r.add_argument("--gpus", type=int, nargs="+", default=C.DEFAULT_GPUS)
    r.add_argument("--per-gpu", type=int, default=1)
    r.add_argument("--seeds", type=int, nargs="+", default=C.SEEDS)
    r.add_argument("--keys", nargs="+", default=None, help="subset of the group's experiments")
    r.add_argument("--force", action="store_true", help="re-run finished runs (weights are reused if valid)")
    r.add_argument("--allow-defaults", action="store_true")
    a = sub.add_parser("analysis")
    a.add_argument("--gpu", type=int, default=C.DEFAULT_GPUS[0])
    a.add_argument("--seed", type=int, default=42)
    s = sub.add_parser("summarize")
    s.add_argument("--seeds", type=int, nargs="+", default=C.SEEDS)
    args = parser.parse_args()

    if args.command == "worker":
        run_worker(args.spec, args.gpu)
    elif args.command == "run":
        if args.allow_defaults and not C.SLISTA_CONFIG_FILE.is_file():
            print("[run] configs/slista.json missing; using config.SLISTA_DEFAULTS", flush=True)
            specs = [E.make_spec(k, s, E.formal_run_dir(k, s)) for k in group_keys(args.group) for s in args.seeds]
        else:
            specs = formal_specs(args.group, args.seeds)
        if args.keys:
            specs = [s for s in specs if s["experiment"]["key"] in args.keys]
        failures = dispatch(specs, args.gpus, args.per_gpu, force=args.force)
        sys.exit(1 if failures else 0)
    elif args.command == "analysis":
        run_analysis(args.gpu, args.seed)
    elif args.command == "summarize":
        summarize(args.seeds)


if __name__ == "__main__":
    main()
