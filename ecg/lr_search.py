#!/usr/bin/env python3
"""Learning-rate search for all four ECG methods on the validation split.

The searched quantity is one scale s per method and phase: every learning
rate of the method (config.py base rates; recovery weights, thresholds,
sensing matrix, dictionary) is multiplied by s, so the ratios between the
parameter groups stay fixed.  The same rule is used for every method
(configs/paper.json, "lr_search"):

  1. train once for every scale in "initial" and select the scale with the
     lowest validation NMSE (best checkpoint of each run);
  2. if the selected scale is the smallest or largest scale tested so far,
     also test the next scale of "ladder" beyond it, and repeat until the
     selected scale is interior or the ladder ends (reported as a boundary
     hit).  A run that diverges (no finite validation NMSE) counts as +inf.

Phase order: the fixed-wavelet phase is searched first (training from
scratch; S-LISTA calibrates its thresholds there).  The learned-dictionary
phase is then searched with every trial warm-started from the selected
fixed-phase model of the same method.  Quantization (2/4/6 bit) reuses the
learned-phase scale.  One seed ("lr_search.seed"); the test split is never
evaluated here.

Usage:
  python lr_search.py --output outputs_search --gpus 0 1 2 3 4 5 --workers-per-gpu 2 --write-config
  python lr_search.py --output outputs_search --targets lista_snn_1 --phases fixed
"""
import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from config import Config, PROJECT_DIR
from main import TARGETS, annotate_jobs, gpu_slots, method_settings, read_result, run_job


def trial_directory(phase, target, scale):
    return f"search/{phase}/{target}/s{scale:g}"


def make_trial(settings, phase, target, scale, warm_from=None):
    search = settings["lr_search"]
    l1, overrides = method_settings(settings, target)
    job = {
        "target": target, "phase": phase,
        "directory": trial_directory(phase, target, scale),
        "bits": 8, "epochs": int(search["epochs"][phase]), "lr_scale": float(scale),
        "cs_mode": "fixed_wavelet" if phase == "fixed" else "learnable_dictionary",
        "seed": int(search["seed"]), "train_snr_db": settings["train_snr_db"],
        "l1": l1, "skip_test": True,
    }
    if overrides:
        job["overrides"] = overrides
    if warm_from is not None:
        job["warm_start_from"] = warm_from
    return job


def validation_nmse(folder):
    try:
        value = float(read_result(folder)["Val_NMSE_dB"])
    except (FileNotFoundError, KeyError, ValueError, StopIteration):
        return math.inf
    return value if math.isfinite(value) else math.inf


def next_scales(results, ladder):
    """Expansion rule: one more ladder step beyond an edge optimum."""
    tested = sorted(results)
    best = min(tested, key=lambda v: results[v])
    if not math.isfinite(results[best]):
        raise RuntimeError("every tested scale diverged")
    index = ladder.index(best)
    if best == tested[0] and index > 0:
        return [ladder[index - 1]]
    if best == tested[-1] and index < len(ladder) - 1:
        return [ladder[index + 1]]
    return []


def summarize(phase, target, results, ladder):
    tested = sorted(results)
    best = min(tested, key=lambda v: results[v])
    return {
        "selected": best, "val_nmse_db": results[best],
        "boundary_hit": best in (ladder[0], ladder[-1]),
        "selected_is_interior": best not in (tested[0], tested[-1]),
        "tested": {f"{s:g}": results[s] for s in tested},
        "selected_directory": trial_directory(phase, target, best),
    }


def search_phase(settings, phase, targets, args, slots, warm_from=None):
    """Search all targets of one phase together, sharing the GPU slots."""
    search = settings["lr_search"]
    ladder = sorted(float(v) for v in search["ladder"])
    initial = sorted(float(v) for v in search["initial"])
    if any(v not in ladder for v in initial):
        raise ValueError("Every initial scale must be on the ladder")
    results = {target: {} for target in targets}
    pending = [(target, scale) for target in targets for scale in initial]

    def execute(item):
        target, scale = item
        source = None if warm_from is None else warm_from[target]
        job = make_trial(settings, phase, target, scale, source)
        annotate_jobs([job], args.data_dir, args.num_workers)
        folder = args.output / job["directory"]
        try:
            run_job(job, args, slots)
        except RuntimeError as error:
            # Only a run that never reached a finite validation NMSE counts as
            # divergence (+inf).  Any other failure (missing GPU, out of
            # memory, bug) stops the search instead of biasing it.
            log = folder / "train.log"
            text = log.read_text(errors="replace") if log.exists() else ""
            if "No finite validation checkpoint" not in text:
                raise RuntimeError(
                    f"{phase}/{target} s={scale:g} failed for a reason other than "
                    f"divergence; see {log}") from error
            print(f"[{phase}/{target}] s={scale:g} diverged (recorded as +inf)", flush=True)
        return target, scale, validation_nmse(folder)

    workers = max(1, len(args.gpus) * args.workers_per_gpu)
    while pending:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for target, scale, value in pool.map(execute, pending):
                results[target][scale] = value
                print(f"[{phase}/{target}] s={scale:g}: val NMSE {value:.3f} dB", flush=True)
        pending = [(target, scale) for target in targets
                   for scale in next_scales(results[target], ladder)]
    return {target: summarize(phase, target, results[target], ladder) for target in targets}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "outputs_search")
    parser.add_argument("--data-dir", type=Path, default=Path(Config.DATA_DIR))
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "configs/paper.json")
    parser.add_argument("--gpus", nargs="+", type=int, default=[0])
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--targets", nargs="+", choices=TARGETS, default=list(TARGETS))
    parser.add_argument("--phases", nargs="+", choices=["fixed", "learned"],
                        default=["fixed", "learned"])
    parser.add_argument("--write-config", action="store_true",
                        help="Write the selected scales into --config (lr_scale).")
    args = parser.parse_args()
    args.output = args.output.expanduser().resolve()
    args.data_dir = args.data_dir.expanduser().resolve()
    if args.workers_per_gpu < 1:
        parser.error("workers-per-gpu must be positive")
    settings = json.loads(args.config.read_text())
    summary_path = args.output / "search" / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    slots = gpu_slots(args)

    if "fixed" in args.phases:
        summary.setdefault("fixed", {}).update(
            search_phase(settings, "fixed", args.targets, args, slots))
    if "learned" in args.phases:
        missing = [t for t in args.targets if t not in summary.get("fixed", {})]
        if missing:
            raise RuntimeError(f"Search the fixed phase first for {missing}")
        warm_from = {t: summary["fixed"][t]["selected_directory"] for t in args.targets}
        summary.setdefault("learned", {}).update(
            search_phase(settings, "learned", args.targets, args, slots, warm_from))
    summary["rule"] = settings["lr_search"]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print("\nphase    target        selected  val NMSE (dB)  interior  tested")
    for phase in ("fixed", "learned"):
        for target, row in summary.get(phase, {}).items():
            print(f"{phase:8s} {target:12s} {row['selected']:9g} {row['val_nmse_db']:13.3f}  "
                  f"{str(row['selected_is_interior']):8s}  {', '.join(row['tested'])}")
            if row["boundary_hit"]:
                print(f"  WARNING: {phase}/{target} selected the end of the ladder")

    if args.write_config:
        for phase in ("fixed", "learned"):
            for target, row in summary.get(phase, {}).items():
                settings["lr_scale"][phase][target] = row["selected"]
        args.config.write_text(json.dumps(settings, indent=2) + "\n")
        print(f"Wrote lr_scale = {json.dumps(settings['lr_scale'])} to {args.config}")


if __name__ == "__main__":
    main()
