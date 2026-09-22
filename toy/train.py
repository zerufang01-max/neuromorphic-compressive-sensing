"""Two-stage sparse-recovery training with fixed configurations or learning-rate search."""

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

from config import (
    ALL_CONDITIONS as CONDITIONS, INITIAL_LEARNING_RATES, LEARNING_RATE_LADDER,
    NUMERICAL_EPS, Protocol, architectures, model_parameters,
)
from data import generator, measurement_matrix, set_seed, sparse_batch
from models import build_model
from utils import DEFAULT_OUTPUT, PAPER_CONFIG, read_json, save_json
from dataclasses import asdict


def nmse_loss(estimate, target):
    return ((estimate - target).square().sum(1)
            / target.square().sum(1).clamp_min(NUMERICAL_EPS)).mean()


@torch.no_grad()
def validation_nmse(model, matrix, condition, samples, seed, cfg):
    model.eval()
    rng = generator(seed, matrix.device)
    error = target_energy = 0.0
    remaining = samples
    while remaining:
        size = min(cfg.eval_batch_size, remaining)
        target, measurements = sparse_batch(
            size, condition["s"], matrix, rng,
            cfg.amplitude_min, cfg.amplitude_max)
        estimate, _ = model(measurements)
        error += float((estimate - target).square().sum())
        target_energy += float(target.square().sum())
        remaining -= size
    return 10 * math.log10(
        max(error, NUMERICAL_EPS) / max(target_energy, NUMERICAL_EPS))


def snapshot(model):
    return {name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()}


def train_phase(model, matrix, condition, parameters, steps, learning_rate,
                data_seed, cfg, initial_score=float("inf"), initial_state=None,
                phase="stage1"):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps, eta_min=cfg.cosine_eta_min)
    rng = generator(data_seed, matrix.device)
    best_score = initial_score
    best_state = initial_state or snapshot(model)
    history = []
    for step in range(1, steps + 1):
        model.train()
        target, measurements = sparse_batch(
            cfg.batch_size, condition["s"], matrix, rng,
            cfg.amplitude_min, cfg.amplitude_max)
        estimate, _ = model(measurements)
        loss = nmse_loss(estimate, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(parameters["grad_clip"]))
        optimizer.step()
        scheduler.step()
        if step == 1 or step % cfg.validation_interval == 0 or step == steps:
            score = validation_nmse(
                model, matrix, condition, cfg.validation_samples,
                cfg.validation_seed, cfg)
            print(f"{phase} step={step}/{steps} loss={float(loss.detach()):.6g} "
                  f"validation_nmse_db={score:.4f} lr={optimizer.param_groups[0]['lr']:.8g}", flush=True)
            history.append({"phase": phase, "step": step,
                            "learning_rate": optimizer.param_groups[0]["lr"],
                            "loss": float(loss.detach()),
                            "validation_nmse_db": score})
            if score < best_score:
                best_score = score
                best_state = snapshot(model)
    return best_score, best_state, history


def run_job(job, result_path, checkpoint_path, device):
    cfg = Protocol(**job.get("protocol", {}))
    condition = job["condition"]
    print(f"Job: {job['id']} condition={condition['id']} seed={cfg.model_seed} "
          f"parameters={job['parameters']} stage1_lr={job.get('stage1_learning_rate')} "
          f"stage2_lr={job.get('stage2_learning_rate')}", flush=True)
    set_seed(cfg.model_seed)
    matrix = measurement_matrix(
        cfg.n, condition["m"], cfg.max_m, cfg.matrix_seed, device)
    model = build_model(
        job["method"], matrix, job["depth"], job["time_steps"],
        job["parameters"]).to(device)

    if job["stage"] in ("stage1", "fixed"):
        best, state, history = train_phase(
            model, matrix, condition, job["parameters"], cfg.stage1_steps,
            job.get("stage1_learning_rate", job["parameters"]["learning_rate"]), cfg.train_seed, cfg)
    else:
        source = torch.load(
            job["source_checkpoint"], map_location=device,
            weights_only=False)
        model.load_state_dict(source["model"], strict=True)
        best = float(job["source_validation_nmse_db"])
        state = snapshot(model)
        best, state, history = train_phase(
            model, matrix, condition, job["parameters"], cfg.stage2_steps,
            job["parameters"]["learning_rate"], cfg.stage2_seed, cfg,
            best, state, "stage2")

    if job["stage"] == "fixed":
        model.load_state_dict(state, strict=True)
        best, state, second_history = train_phase(
            model, matrix, condition, job["parameters"], cfg.stage2_steps,
            job["stage2_learning_rate"], cfg.stage2_seed, cfg,
            best, state, "stage2")
        history.extend(second_history)
    model.load_state_dict(state, strict=True)
    selection = validation_nmse(
        model, matrix, condition, cfg.selection_samples,
        cfg.selection_seed, cfg)
    torch.save({"model": model.state_dict(), "job": job, "protocol": asdict(cfg)}, checkpoint_path)
    result = {**job, "best_validation_nmse_db": best,
              "selection_nmse_db": selection,
              "checkpoint": str(checkpoint_path)}
    save_json(result_path, result)
    with open(result_path.with_suffix(".csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)


def worker(args):
    with open(args.job, encoding="utf-8") as handle:
        job = json.load(handle)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    run_job(job, Path(args.result), Path(args.checkpoint), device)


def launch(jobs, stage, root, gpus, workers_per_gpu):
    work = root / "search" / stage
    for folder in ("jobs", "results", "checkpoints", "logs"):
        (work / folder).mkdir(parents=True, exist_ok=True)
    pending = []
    for job in jobs:
        result = work / "results" / f"{job['job_id']}.json"
        checkpoint = work / "checkpoints" / f"{job['job_id']}.pth"
        if result.is_file() and checkpoint.is_file():
            previous = read_json(result)
            if any(previous.get(k) != v for k, v in job.items()):
                raise ValueError(f'Existing job configuration differs: {result}; choose another --output')
        else:
            pending.append(job)
    if not gpus or workers_per_gpu < 1:
        raise ValueError("At least one GPU slot and one worker are required")
    slots = [gpu for gpu in gpus for _ in range(workers_per_gpu)]
    running = []
    completed = len(jobs) - len(pending)
    print(f"{stage}: {completed}/{len(jobs)} complete", flush=True)
    while pending or running:
        occupied = {entry[0] for entry in running}
        for slot, gpu in enumerate(slots):
            if slot in occupied or not pending:
                continue
            job = pending.pop(0)
            job_path = work / "jobs" / f"{job['job_id']}.json"
            result_path = work / "results" / f"{job['job_id']}.json"
            checkpoint_path = work / "checkpoints" / f"{job['job_id']}.pth"
            save_json(job_path, job)
            log = open(work / "logs" / f"{job['job_id']}.log", "w")
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            command = [
                sys.executable, str(Path(__file__).resolve()), "--worker",
                "--job", str(job_path), "--result", str(result_path),
                "--checkpoint", str(checkpoint_path),
            ]
            process = subprocess.Popen(
                command, env=environment, stdout=log,
                stderr=subprocess.STDOUT)
            running.append((slot, process, job, log))
        time.sleep(1)
        active = []
        for slot, process, job, log in running:
            if process.poll() is None:
                active.append((slot, process, job, log))
                continue
            log.close()
            if process.returncode:
                raise RuntimeError(
                    f"{job['job_id']} failed; see {work / 'logs'}")
            completed += 1
            print(f"{stage}: {completed}/{len(jobs)} complete", flush=True)
        running = active
    rows = []
    for job in jobs:
        with open(work / "results" / f"{job['job_id']}.json") as handle:
            rows.append(json.load(handle))
    return rows


def pair_key(row):
    return row["condition"]["id"], row["id"]


def lr_tag(value):
    return f"{value:.0e}".replace("-", "m").replace("+", "")


def stage1_job(condition, architecture, learning_rate, round_index=0):
    return {
        **architecture, "condition": condition, "stage": "stage1",
        "boundary_round": round_index,
        "parameters": model_parameters(architecture, learning_rate),
        "job_id": (f"{condition['id']}__{architecture['id']}__"
                   f"s1_lr{lr_tag(learning_rate)}"),
    }


def initial_jobs(conditions, architecture_list):
    return [stage1_job(condition, architecture, learning_rate)
            for condition in conditions
            for architecture in architecture_list
            for learning_rate in INITIAL_LEARNING_RATES]


def boundary_jobs(rows, conditions, architecture_list, round_index):
    grouped = {}
    for row in rows:
        grouped.setdefault(pair_key(row), []).append(row)
    ladder = list(LEARNING_RATE_LADDER)
    jobs = []
    for condition in conditions:
        for architecture in architecture_list:
            candidates = grouped[(condition["id"], architecture["id"])]
            tested = sorted({float(row["parameters"]["learning_rate"])
                             for row in candidates})
            best_lr = float(min(
                candidates, key=lambda row: row["selection_nmse_db"]
            )["parameters"]["learning_rate"])
            proposed = None
            if best_lr == tested[0]:
                index = ladder.index(tested[0])
                if index > 0:
                    proposed = ladder[index - 1]
            elif best_lr == tested[-1]:
                index = ladder.index(tested[-1])
                if index + 1 < len(ladder):
                    proposed = ladder[index + 1]
            if proposed is not None and proposed not in tested:
                jobs.append(stage1_job(
                    condition, architecture, proposed, round_index))
    return jobs


def stage2_jobs(stage1_rows, conditions, architecture_list, cfg):
    grouped = {}
    for row in stage1_rows:
        grouped.setdefault(pair_key(row), []).append(row)
    jobs = []
    for condition in conditions:
        for architecture in architecture_list:
            ranked = sorted(
                grouped[(condition["id"], architecture["id"])],
                key=lambda row: row["selection_nmse_db"])
            for rank, source in enumerate(ranked[:cfg.stage2_sources]):
                stage1_lr = float(source["parameters"]["learning_rate"])
                for factor in cfg.stage2_lr_factors:
                    parameters = dict(source["parameters"])
                    parameters["learning_rate"] = stage1_lr * factor
                    jobs.append({
                        **architecture, "condition": condition,
                        "stage": "stage2", "source_rank": rank,
                        "stage1_learning_rate": stage1_lr,
                        "stage2_lr_factor": factor,
                        "source_checkpoint": source["checkpoint"],
                        "source_validation_nmse_db":
                            source["best_validation_nmse_db"],
                        "parameters": parameters,
                        "job_id": (f"{condition['id']}__{architecture['id']}__"
                                   f"s2_r{rank}_x{factor:g}"),
                    })
    return jobs


def select_models(rows, conditions, architecture_list, root):
    grouped = {}
    for row in rows:
        grouped.setdefault(pair_key(row), []).append(row)
    checkpoint_dir = root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    selected = []
    for condition in conditions:
        for architecture in architecture_list:
            best = min(
                grouped[(condition["id"], architecture["id"])],
                key=lambda row: row["selection_nmse_db"])
            destination = checkpoint_dir / (
                f"{condition['id']}__{architecture['id']}.pth")
            if destination.exists():
                if destination.read_bytes() != Path(best["checkpoint"]).read_bytes():
                    raise ValueError(f"Existing checkpoint differs: {destination}; use another --output")
            else:
                shutil.copy2(best["checkpoint"], destination)
            selected.append({
                **architecture, "condition": condition,
                "parameters": best["parameters"],
                "stage1_learning_rate": best["stage1_learning_rate"],
                "stage2_learning_rate": best["parameters"]["learning_rate"],
                "best_validation_nmse_db":
                    best["best_validation_nmse_db"],
                "selection_nmse_db": best["selection_nmse_db"],
                "checkpoint": str(destination.relative_to(root)),
            })
            print(f"selected {condition['id']} / {architecture['id']}: "
                  f"{best['selection_nmse_db']:.2f} dB", flush=True)
    merge_selected(root, selected)


def controller(args):
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.conditions:
        requested_conditions = set(args.conditions)
        condition_list = [row for row in CONDITIONS
                          if row["id"] in requested_conditions]
        if len(condition_list) != len(requested_conditions):
            known = {row["id"] for row in CONDITIONS}
            raise ValueError(
                f"Unknown conditions: {sorted(requested_conditions - known)}")
    else:
        condition_list = list(CONDITIONS)
    all_architectures = architectures()
    if args.ids:
        requested = set(args.ids)
        architecture_list = [row for row in all_architectures
                             if row["id"] in requested]
        if len(architecture_list) != len(requested):
            known = {row["id"] for row in all_architectures}
            raise ValueError(f"Unknown IDs: {sorted(requested - known)}")
    else:
        architecture_list = all_architectures
    cfg = Protocol()
    print(f"Conditions: {', '.join(row['id'] for row in condition_list)}")
    print(f"Architectures: {len(architecture_list)}")
    print(f"Batch: {cfg.batch_size}; updates: "
          f"{cfg.stage1_steps}+{cfg.stage2_steps}", flush=True)

    signature = {"conditions": condition_list, "architectures": architecture_list,
                 "protocol": asdict(cfg)}
    signature = json.loads(json.dumps(signature))
    signature_path = root / "search" / "search_config.json"
    if signature_path.exists() and read_json(signature_path) != signature:
        raise ValueError('Search configuration differs; use a separate --output')
    save_json(signature_path, signature)
    stage1_path = root / "search" / "stage1_results.json"
    if stage1_path.is_file():
        with open(stage1_path) as handle:
            stage1_rows = json.load(handle)
    else:
        stage1_rows = launch(
            initial_jobs(condition_list, architecture_list), "stage1", root,
            args.gpus, args.workers_per_gpu)
        save_json(stage1_path, stage1_rows)
    for round_index in range(1, cfg.max_boundary_expansions + 1):
        extra = boundary_jobs(
            stage1_rows, condition_list, architecture_list, round_index)
        if not extra:
            break
        stage1_rows.extend(launch(
            extra, "stage1", root, args.gpus, args.workers_per_gpu))
        save_json(stage1_path, stage1_rows)

    stage2_path = root / "search" / "stage2_results.json"
    if stage2_path.is_file():
        with open(stage2_path) as handle:
            stage2_rows = json.load(handle)
    else:
        jobs = stage2_jobs(stage1_rows, condition_list, architecture_list, cfg)
        stage2_rows = launch(
            jobs, "stage2", root, args.gpus, args.workers_per_gpu)
        save_json(stage2_path, stage2_rows)
    select_models(stage2_rows, condition_list, architecture_list, root)
    print(f"Training complete: {root}", flush=True)


def merge_selected(root, rows):
    path = root / 'configs' / 'selected_configs.json'
    previous = read_json(path) if path.exists() else []
    merged = {(r['condition']['id'], r['id']): r for r in previous}
    for row in rows:
        key = (row['condition']['id'], row['id'])
        if key in merged and merged[key] != row:
            raise ValueError(f'Conflicting exported model: {key}; use another --output')
        merged[key] = row
    save_json(path, list(merged.values()))


def fixed_controller_single(args):
    root = Path(args.output).expanduser().resolve()
    config = read_json(args.config)
    jobs = [r for r in config['jobs']
            if r['method'] != 'slista' or r['time_steps'] == 1]
    if args.conditions:
        known = {r['condition']['id'] for r in jobs}
        if set(args.conditions) - known:
            raise ValueError('Unknown conditions')
        jobs = [r for r in jobs if r['condition']['id'] in args.conditions]
    if args.ids:
        known = {r['id'] for r in jobs}
        if set(args.ids) - known:
            raise ValueError('Unknown model IDs for these conditions')
        jobs = [r for r in jobs if r['id'] in args.ids]
    if not jobs:
        raise ValueError('No selected training jobs')
    prepared = [{**r, 'stage': 'fixed', 'protocol': config['protocol'],
                 'job_id': r['condition']['id'] + '__' + r['id']} for r in jobs]
    keys = [r['job_id'] for r in prepared]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate training jobs')
    save_json(root / 'configs' / 'requested_jobs.json', prepared)
    rows = launch(prepared, 'fixed', root, args.gpus, args.workers_per_gpu)
    exported = []
    for row in rows:
        destination = root / 'checkpoints' / (row['job_id'] + '.pth')
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            import hashlib
            if hashlib.sha256(destination.read_bytes()).digest() != hashlib.sha256(Path(row['checkpoint']).read_bytes()).digest():
                raise ValueError(f'Existing final checkpoint differs: {destination}')
        else:
            shutil.copy2(row['checkpoint'], destination)
        exported.append({**row, 'checkpoint': str(destination.relative_to(root))})
    merge_selected(root, exported)


def fixed_controller(args):
    from copy import copy, deepcopy
    base = read_json(args.config)
    seeds = args.seeds if args.seeds is not None else base.get('seeds', [42,43,44,45,46])
    if len(set(seeds)) != len(seeds):
        raise ValueError('Duplicate seeds')
    for seed in seeds:
        cfg = deepcopy(base)
        cfg.pop('seeds', None)
        # Fixed validation/test/matrix; independent initialization and train streams.
        cfg['protocol']['model_seed'] = seed
        cfg['protocol']['train_seed'] = 20260801 + seed
        cfg['protocol']['stage2_seed'] = 20260811 + seed
        for job in cfg['jobs']:
            job['seed'] = seed
        root = Path(args.output).expanduser().resolve() / f'seed{seed}'
        path = root / 'configs' / 'run_config.json'
        if path.exists() and read_json(path) != cfg:
            previous = read_json(path)
            if previous['protocol'] != cfg['protocol']:
                raise ValueError(f'Training protocol differs: {path}')
            old_jobs = {(j['condition']['id'],j['id']):j for j in previous['jobs']}
            for job in cfg['jobs']:
                key = (job['condition']['id'],job['id'])
                if key in old_jobs and old_jobs[key] != job:
                    raise ValueError(f'Existing model settings differ: {key}')
            # Model inventory/figure settings may change; common training jobs may not.
            history = path.parent / 'previous_run_configs'
            history.mkdir(exist_ok=True)
            import hashlib
            digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
            shutil.copy2(path, history / (digest + '.json'))
        save_json(path, cfg)
        sub = copy(args)
        sub.output, sub.config = str(root), path
        print(f'=== Seed {seed} ===', flush=True)
        fixed_controller_single(sub)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--gpus", nargs="+", type=int,
                        default=[0])
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--mode", choices=["fixed", "search"], default="fixed")
    parser.add_argument("--config", type=Path, default=PAPER_CONFIG)
    parser.add_argument("--ids", nargs="+")
    parser.add_argument("--conditions", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--job")
    parser.add_argument("--result")
    parser.add_argument("--checkpoint")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.mode == "search" and arguments.seeds is not None:
        raise ValueError("--seeds applies to fixed training; search uses its own output root")
    worker(arguments) if arguments.worker else (fixed_controller(arguments) if arguments.mode == "fixed" else controller(arguments))
