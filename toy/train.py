"""Training for the synthetic experiments.

Two modes:
  --mode search  learning-rate search on the hyper-parameter selection set
                 (stage 1 grid, expanded one ladder step at a time while the best
                 rate is at the edge of the tested range, then stage 2 at
                 stage2_lr_factor x the best stage-1 rate).  Writes
                 <output>/search/selected_jobs.json in the format of
                 configs/paper.json; copy it there to freeze the protocol.
  --mode fixed   train the frozen jobs of configs/paper.json for every seed in
                 TRAINING_SEEDS (stage 1 then stage 2, no search).

Within a run the checkpoint with the lowest validation NMSE is kept.  Test
data are never used here.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import torch

from config import (
    ALL_CONDITIONS, GRAD_CLIP, INITIAL_LEARNING_RATES, LEARNING_RATE_LADDER,
    NUMERICAL_EPS, TRAINING_SEEDS, Protocol, experiment_grid, model_parameters,
)
from data import generator, measurement_matrix, set_seed, sparse_batch
from models import build_model, nmse_loss
from utils import DEFAULT_OUTPUT, PAPER_CONFIG, read_json, save_json


# ------------------------------------------------------------------ helpers
@torch.no_grad()
def held_out_nmse_db(model, matrix, condition, samples, seed, cfg):
    model.eval()
    rng = generator(seed, matrix.device)
    error = energy = 0.0
    remaining = samples
    while remaining:
        size = min(cfg.eval_batch_size, remaining)
        target, measurements = sparse_batch(size, condition["s"], matrix, rng,
                                            cfg.amplitude_min, cfg.amplitude_max)
        estimate, _ = model(measurements)
        error += float((estimate - target).square().sum())
        energy += float(target.square().sum())
        remaining -= size
    return 10 * math.log10(max(error, NUMERICAL_EPS) / max(energy, NUMERICAL_EPS))


def snapshot(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def train_phase(model, matrix, condition, steps, learning_rate, data_seed, cfg,
                best_score=float("inf"), best_state=None, phase="stage1"):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps, eta_min=cfg.cosine_eta_min)
    rng = generator(data_seed, matrix.device)
    best_state = best_state or snapshot(model)
    history = []
    for step in range(1, steps + 1):
        model.train()
        target, measurements = sparse_batch(cfg.batch_size, condition["s"], matrix, rng,
                                            cfg.amplitude_min, cfg.amplitude_max)
        estimate, _ = model(measurements)
        loss = nmse_loss(estimate, target)
        if hasattr(model, "spike_penalty"):
            loss = loss + model.spike_penalty(model.last_trace)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % cfg.validation_interval == 0 or step == steps:
            score = held_out_nmse_db(model, matrix, condition, cfg.validation_samples,
                                     cfg.validation_seed, cfg)
            row = {"phase": phase, "step": step, "loss": float(loss.detach()),
                   "learning_rate": optimizer.param_groups[0]["lr"],
                   "validation_nmse_db": score}
            if hasattr(model, "theta"):
                row["theta"] = [round(float(t), 4) for t in model.theta.detach()]
                row["eta"] = float(model.eta.detach())
            history.append(row)
            print(json.dumps(row), flush=True)
            if score < best_score:
                best_score, best_state = score, snapshot(model)
    return best_score, best_state, history


def run_job(job, result_path, checkpoint_path, device):
    """Train one job.  job keys: condition, method, depth, parameters,
    stage1_learning_rate, stage2_learning_rate (None = stage 1 only),
    protocol (optional overrides), seed (optional)."""
    cfg = Protocol(**job.get("protocol", {}))
    condition = job["condition"]
    set_seed(cfg.model_seed)
    matrix = measurement_matrix(cfg.n, condition["m"], cfg.max_m, cfg.matrix_seed, device)
    model = build_model(job["method"], matrix, job["depth"], job["parameters"]).to(device)
    calibration = None
    if hasattr(model, "calibrate"):
        target, measurements = sparse_batch(
            cfg.batch_size, condition["s"], matrix, generator(cfg.calibration_seed, device),
            cfg.amplitude_min, cfg.amplitude_max)
        calibration = model.calibrate(target, measurements)
    best, state, history = train_phase(model, matrix, condition, cfg.stage1_steps,
                                       job["stage1_learning_rate"], cfg.train_seed, cfg)
    if job.get("stage2_learning_rate") is not None:
        model.load_state_dict(state)
        best, state, more = train_phase(model, matrix, condition, cfg.stage2_steps,
                                        job["stage2_learning_rate"], cfg.stage2_seed, cfg,
                                        best, state, "stage2")
        history += more
    model.load_state_dict(state)
    selection = held_out_nmse_db(model, matrix, condition, cfg.selection_samples,
                                 cfg.selection_seed, cfg)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "job": job, "protocol": asdict(cfg),
                "calibration": calibration}, checkpoint_path)
    save_json(result_path, {**job, "best_validation_nmse_db": best,
                            "selection_nmse_db": selection, "calibration": calibration,
                            "checkpoint": str(checkpoint_path), "history": history})


# --------------------------------------------------------- process launcher
def launch(jobs, work, gpus, workers_per_gpu, replace_stale=False):
    """Run jobs as subprocesses, workers_per_gpu at a time on each GPU.

    Resuming: a job whose result and checkpoint both exist and whose stored job
    is identical is skipped.  A job that was interrupted (no result yet) is
    rerun from the start.  If a finished result belongs to a *different* job
    with the same name, it is an error unless replace_stale=True (used only for
    search stage 2, whose job changes when the best stage-1 rate changes); the
    stale files are then moved to <work>/stale/ and the job is rerun."""
    for folder in ("jobs", "results", "checkpoints", "logs"):
        (work / folder).mkdir(parents=True, exist_ok=True)
    pending = []
    for job in jobs:
        result = work / "results" / f"{job['job_id']}.json"
        checkpoint = work / "checkpoints" / f"{job['job_id']}.pth"
        if result.is_file() and checkpoint.is_file():
            previous = read_json(result)
            if all(previous.get(k) == v for k, v in job.items()):
                continue
            if not replace_stale:
                raise ValueError(f"{result} was produced by a different job; use another --output")
            stale = work / "stale" / time.strftime("%Y%m%d-%H%M%S")
            stale.mkdir(parents=True, exist_ok=True)
            result.replace(stale / result.name)
            checkpoint.replace(stale / checkpoint.name)
            print(f"{work.name}: {job['job_id']} changed; old result moved to {stale}", flush=True)
        pending.append(job)
    slots = [gpu for gpu in gpus for _ in range(workers_per_gpu)]
    running = {}
    print(f"{work.name}: {len(jobs) - len(pending)}/{len(jobs)} already complete", flush=True)
    while pending or running:
        for slot, gpu in enumerate(slots):
            if slot in running or not pending:
                continue
            job = pending.pop(0)
            job_path = work / "jobs" / f"{job['job_id']}.json"
            save_json(job_path, job)
            log = open(work / "logs" / f"{job['job_id']}.log", "w")
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
            command = [sys.executable, str(Path(__file__).resolve()), "--worker",
                       "--job", str(job_path),
                       "--result", str(work / "results" / f"{job['job_id']}.json"),
                       "--checkpoint", str(work / "checkpoints" / f"{job['job_id']}.pth")]
            running[slot] = (subprocess.Popen(command, env=env, stdout=log,
                                              stderr=subprocess.STDOUT), job, log)
        time.sleep(1)
        for slot in list(running):
            process, job, log = running[slot]
            if process.poll() is None:
                continue
            log.close()
            del running[slot]
            if process.returncode:
                raise RuntimeError(f"{job['job_id']} failed; see {work / 'logs'}")
            print(f"{work.name}: finished {job['job_id']}", flush=True)
    return [read_json(work / "results" / f"{job['job_id']}.json") for job in jobs]


# --------------------------------------------------------------- search mode
def lr_tag(value):
    return f"{value:.0e}".replace("-", "m").replace("+", "")


def search_job(condition, architecture, learning_rate):
    return {"condition": condition, "method": architecture["method"],
            "depth": architecture["depth"], "id": architecture["id"],
            "parameters": model_parameters(architecture),
            "stage1_learning_rate": learning_rate, "stage2_learning_rate": None,
            "job_id": f"{condition['id']}__{architecture['id']}__lr{lr_tag(learning_rate)}"}


def search(args):
    root = Path(args.output).resolve()
    cfg = Protocol()
    wanted = {c["id"] for c in selected_conditions(args)}
    grid = [(c, a) for c, a in experiment_grid()
            if c["id"] in wanted and (not args.ids or a["id"] in args.ids)]
    stage1 = launch([search_job(c, a, lr) for c, a in grid for lr in INITIAL_LEARNING_RATES],
                    root / "search" / "stage1", args.gpus, args.workers_per_gpu)
    # Expand one ladder step at a time until every optimum is interior to the
    # tested range or sits at an end of LEARNING_RATE_LADDER.  Already finished
    # stage-1 jobs are skipped by launch(), so a rerun replays this quickly.
    while True:
        extra = []
        for c, a in grid:
            rows = [r for r in stage1 if r["condition"]["id"] == c["id"] and r["id"] == a["id"]]
            tested = sorted(r["stage1_learning_rate"] for r in rows)
            best = min(rows, key=lambda r: r["selection_nmse_db"])["stage1_learning_rate"]
            index = LEARNING_RATE_LADDER.index(best)
            if best == tested[0] and index > 0:
                extra.append(search_job(c, a, LEARNING_RATE_LADDER[index - 1]))
            elif best == tested[-1] and index + 1 < len(LEARNING_RATE_LADDER):
                extra.append(search_job(c, a, LEARNING_RATE_LADDER[index + 1]))
        if not extra:
            break
        stage1 += launch(extra, root / "search" / "stage1", args.gpus, args.workers_per_gpu)
    stage2_jobs, boundary = [], []
    for c, a in grid:
        rows = [r for r in stage1 if r["condition"]["id"] == c["id"] and r["id"] == a["id"]]
        best = min(rows, key=lambda r: r["selection_nmse_db"])
        tested = sorted(r["stage1_learning_rate"] for r in rows)
        if best["stage1_learning_rate"] in (tested[0], tested[-1]):
            # Still at the edge of the tested range after all expansions.
            boundary.append({"condition": c["id"], "id": a["id"],
                             "best_learning_rate": best["stage1_learning_rate"],
                             "tested": tested})
        job = {k: best[k] for k in ("condition", "method", "depth", "id", "parameters",
                                    "stage1_learning_rate")}
        job["stage2_learning_rate"] = best["stage1_learning_rate"] * cfg.stage2_lr_factor
        job["job_id"] = f"{c['id']}__{a['id']}"
        stage2_jobs.append(job)
    final = launch(stage2_jobs, root / "search" / "stage2", args.gpus, args.workers_per_gpu,
                   replace_stale=True)
    frozen = {(r["condition"]["id"], r["id"]): {
        k: r[k] for k in ("condition", "method", "depth", "id", "parameters",
                          "stage1_learning_rate", "stage2_learning_rate")} for r in final}
    # Merge into the current config: searched jobs replace their entries, jobs of
    # experiment_grid() that were not searched now are kept, anything else is dropped.
    base = read_json(args.config)
    needed = {(c["id"], a["id"]) for c, a in experiment_grid()}
    jobs = [frozen.pop((j["condition"]["id"], j["id"]), j) for j in base["jobs"]
            if (j["condition"]["id"], j["id"]) in needed]
    jobs += list(frozen.values())
    save_json(root / "search" / "selected_jobs.json",
              {**base, "description": "Output of train.py --mode search; review before "
                                      "copying to configs/paper.json.",
               "search_boundary_warnings": boundary, "jobs": jobs})
    for row in boundary:
        print(f"WARNING: {row['condition']} / {row['id']}: best learning rate "
              f"{row['best_learning_rate']} is at the edge of the tested range {row['tested']}")
    print(f"{len(boundary)} model(s) at the search boundary")
    print(f"Wrote {root / 'search' / 'selected_jobs.json'}; review and copy to configs/paper.json")


# ---------------------------------------------------------------- fixed mode
def fixed(args):
    config = read_json(args.config)
    wanted = {c["id"] for c in selected_conditions(args)}
    jobs = [j for j in config["jobs"] if j["condition"]["id"] in wanted
            and (not args.ids or j["id"] in args.ids)]
    if not jobs:
        raise ValueError("No jobs selected")
    missing = [j["id"] for j in jobs if not j.get("stage1_learning_rate")]
    if missing:
        raise ValueError(f"No learning rate for {missing}; run --mode search and copy "
                         "selected_jobs.json to configs/paper.json first")
    for seed in args.seeds or config.get("seeds", TRAINING_SEEDS):
        protocol = dict(config["protocol"], model_seed=seed,
                        train_seed=20260801 + seed, stage2_seed=20260811 + seed,
                        calibration_seed=20260815 + seed)
        prepared = [{**deepcopy(j), "protocol": protocol, "seed": seed,
                     "job_id": f"{j['condition']['id']}__{j['id']}"} for j in jobs]
        print(f"=== seed {seed} ===", flush=True)
        launch(prepared, Path(args.output).resolve() / f"seed{seed}",
               args.gpus, args.workers_per_gpu)


def selected_conditions(args):
    if not args.conditions:
        return list(ALL_CONDITIONS)
    known = {c["id"]: c for c in ALL_CONDITIONS}
    unknown = set(args.conditions) - set(known)
    if unknown:
        raise ValueError(f"Unknown conditions: {sorted(unknown)}")
    return [known[c] for c in args.conditions]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("fixed", "search"), default="fixed")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--config", type=Path, default=PAPER_CONFIG)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0])
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--conditions", nargs="+")
    parser.add_argument("--ids", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--job", help=argparse.SUPPRESS)
    parser.add_argument("--result", help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint", help=argparse.SUPPRESS)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.worker:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        run_job(read_json(arguments.job), Path(arguments.result),
                Path(arguments.checkpoint), device)
    elif arguments.mode == "search":
        search(arguments)
    else:
        fixed(arguments)
