"""Training for the synthetic experiments.

Trains the jobs of configs/paper.json (model, depth, condition and learning
rates) for every seed in TRAINING_SEEDS: stage 1, then stage 2 from the best
stage-1 checkpoint at stage2_lr_factor times the stage-1 rate.  Within a run
the checkpoint with the lowest validation NMSE is kept.  Test data are never
used here.
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
    ALL_CONDITIONS, GRAD_CLIP, NUMERICAL_EPS, TRAINING_SEEDS, Protocol,
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
    with the same name, it is an error unless replace_stale=True; the
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


# ------------------------------------------------------------------ training
def train_jobs(args):
    config = read_json(args.config)
    wanted = {c["id"] for c in selected_conditions(args)}
    jobs = [j for j in config["jobs"] if j["condition"]["id"] in wanted
            and (not args.ids or j["id"] in args.ids)]
    if not jobs:
        raise ValueError("No jobs selected")
    missing = [j["id"] for j in jobs if not j.get("stage1_learning_rate")]
    if missing:
        raise ValueError(f"No learning rate for {missing} in {args.config}")
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
    else:
        train_jobs(arguments)
