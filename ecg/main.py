"""ECG experiment scheduling and single-job training."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import torch
from config import Config, PROJECT_DIR, DEFAULT_OUTPUT
from edge_compression import EdgeSensor
from wireless_channel import WirelessChannel
from snn_models import LAMP, HybridLISTA, ALISTA, FISTA
from trainer import Trainer
import utils

TARGETS = ("alista", "lamp", "lista_ann", "lista_snn_1")
def build_experiments(target: str = "all"):
    exps = [
        {"id": "fista", "name": "FISTA", "label": "FISTA", "model_class": FISTA, "mode": "ann", "hp": Config.FISTA_CONFIG},
        {"id": "alista", "name": "ALISTA", "label": "ANN ALISTA", "model_class": ALISTA, "mode": "ann", "hp": Config.ALISTA_CONFIG},
        {"id": "lamp", "name": "LAMP", "label": "ANN LAMP", "model_class": LAMP, "mode": "ann", "hp": Config.LAMP_ANN_CONFIG},
        {"id": "lista_ann", "name": "LISTA_ANN", "label": "ANN LISTA", "model_class": HybridLISTA, "mode": "ann", "hp": Config.LISTA_ANN_CONFIG},
        {"id": "lista_snn_1", "name": "LISTA_SNN", "label": "Spiking LISTA", "model_class": HybridLISTA, "mode": "spiking", "hp": Config.LISTA_SNN_CONFIG, "t_val": 1},
    ]
    if target != "all":
        exps = [e for e in exps if e["id"] == target]
    return exps

def configure_channel(snr_db: float):
    Config.SNR_DB = float(snr_db)

def l1_for_mode(mode: str) -> float:
    canonical_mode = "spiking" if mode == "snn" else mode
    return (
        Config.ALPHA_L1_SNN if canonical_mode == "spiking"
        else Config.ALPHA_L1_ANN
    )

def apply_training_overrides(epochs=None, lr_scale=1.0):
    train_configs = (
        Config.LAMP_ANN_CONFIG,
        Config.LISTA_ANN_CONFIG,
        Config.LISTA_SNN_CONFIG,
        Config.ALISTA_CONFIG,
    )
    if epochs is not None:
        for hp in train_configs:
            hp["epochs"] = epochs

    Config.LR_SCALE = lr_scale
    for hp in train_configs:
        for key in ("lr_model", "lr_theta", "lr_phi", "lr_dictionary"):
            if key in hp:
                hp[key] *= lr_scale

def build_checkpoint_name(exp_cfg, model, train_snr_db: float) -> str:
    canonical_mode = "spiking" if exp_cfg["mode"] == "snn" else exp_cfg["mode"]
    model_name_lower = exp_cfg["name"].lower()
    threshold = model.get_threshold_value() if hasattr(model, "get_threshold_value") else 0.0
    suffix = (
        f"_th{threshold:g}_snr{train_snr_db:g}_q{Config.QUANT_BITS}"
        f"_{Config.CS_MODE}"
    )
    if Config.LR_SCALE != 1.0:
        suffix += f"_lrs{Config.LR_SCALE:g}"
    if canonical_mode == "spiking":
        suffix += (
            f"_T{Config.TIME_STEPS}_fth{Config.FINAL_THETA_SNN:g}"
            f"_sw{Config.SURROGATE_WIDTH:g}"
        )
    if Config.WARM_START_CHECKPOINT:
        suffix += "_warmft"

    return (
        f"ecg_M{Config.M}_K{Config.NUM_LAYERS}_"
        f"{canonical_mode}_{model_name_lower}_"
        f"l1{l1_for_mode(canonical_mode):.0e}{suffix}.pth"
    )

def resolve_compatible_checkpoint(
    checkpoint_root, expected_path, exp_cfg, model, sensor
):
    """Load only the checkpoint with the current concise filename."""
    expected = Path(expected_path)
    if expected.is_file() and Path(str(expected) + ".complete").exists():
        return expected, torch.load(
            expected, map_location=Config.DEVICE, weights_only=False
        )
    return expected, None

def train_or_load_fixed_p(exp_cfg, train_loader, val_loader, train_snr_db: float):
    configure_channel(train_snr_db)

    if "t_val" in exp_cfg:
        Config.TIME_STEPS = exp_cfg["t_val"]

    channel = WirelessChannel().to(Config.DEVICE)
    channel.snr_db = train_snr_db
    sensor = EdgeSensor().to(Config.DEVICE)

    if exp_cfg["model_class"] in (LAMP, HybridLISTA, ALISTA, FISTA):
        model = exp_cfg["model_class"](
            phi=sensor.effective_phi().detach()
        ).to(Config.DEVICE)
    else:
        model = exp_cfg["model_class"]().to(Config.DEVICE)

    canonical_mode = "spiking" if exp_cfg["mode"] == "snn" else exp_cfg["mode"]
    if hasattr(model, "set_mode"):
        model.set_mode(canonical_mode)

    trainer = Trainer(model, sensor, channel, train_loader, val_loader, custom_cfg=exp_cfg["hp"])
    ckpt_name = build_checkpoint_name(exp_cfg, model, train_snr_db)
    checkpoint_root = Config.RESULTS_DIR
    if exp_cfg["id"] == "lista_snn_1":
        checkpoint_root = getattr(
            Config, "SLISTA_RESULTS_DIR", None
        ) or checkpoint_root
    else:
        checkpoint_root = getattr(
            Config, "BASELINE_RESULTS_DIR", None
        ) or checkpoint_root
    ckpt_path = os.path.join(checkpoint_root, ckpt_name)

    if exp_cfg["name"] != "FISTA":
        resolved_path, ckpt = resolve_compatible_checkpoint(
            checkpoint_root, ckpt_path, exp_cfg, model, sensor
        )
        if ckpt is not None:
            ckpt_path = str(resolved_path)
            ckpt_name = resolved_path.name
            print(f"    [Found] Loaded weights: {ckpt_path}")
            if ckpt.get("code_version") != Config.CODE_VERSION:
                raise RuntimeError(
                    f"Checkpoint {ckpt_name} was produced by "
                    f"{ckpt.get('code_version', 'an unknown code version')}; "
                    f"expected {Config.CODE_VERSION}."
                )
            if canonical_mode == "spiking" and ckpt.get(
                "slista_init", "ista"
            ) != Config.SLISTA_INIT:
                raise RuntimeError(
                    f"Checkpoint initialization is "
                    f"{ckpt.get('slista_init', 'ista')}, expected "
                    f"{Config.SLISTA_INIT}."
                )
            model.load_state_dict(ckpt["model"], strict=True)
            sensor.load_state_dict(ckpt["sensor"], strict=True)
        elif getattr(Config, "EVAL_ONLY", False):
            raise FileNotFoundError(
                f"Strict eval-only checkpoint is missing and no compatible "
                f"fallback was found: {ckpt_path}\n"
                f"Available .pth files in {checkpoint_root}:\n  "
                + "\n  ".join(
                    str(path) for path in sorted(
                        Path(checkpoint_root).rglob("*.pth")
                    )
                )
            )
        elif exp_cfg["hp"].get("epochs", 0) > 0:
            if Config.WARM_START_CHECKPOINT:
                warm_path = Path(Config.WARM_START_CHECKPOINT)
                warm_checkpoint = torch.load(
                    warm_path, map_location=Config.DEVICE, weights_only=False
                )
                model.load_state_dict(warm_checkpoint["model"], strict=True)
                sensor.load_state_dict(warm_checkpoint["sensor"], strict=True)
                if hasattr(model, "synchronize_dictionary_operator"):
                    model.synchronize_dictionary_operator(
                        sensor.effective_phi()
                    )
                print(f"    [Warm start] Loaded fixed-CS weights: {warm_path}")
            print(f"    [Train] {ckpt_name} not found. Training once at train_snr_db={train_snr_db:g}...")
            trainer.train_dynamic_recon(
                hp=exp_cfg["hp"],
                model_name=exp_cfg["name"],
                mode=exp_cfg["mode"],
                save_path=ckpt_path,
            )
            if os.path.exists(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location=Config.DEVICE, weights_only=False)
                model.load_state_dict(ckpt["model"], strict=True)
                sensor.load_state_dict(ckpt["sensor"], strict=True)
        else:
            print(f"    [Warn] {ckpt_name} is missing and epochs=0. Skipping.")
            return None

    return {
        "exp": exp_cfg,
        "trainer": trainer,
        "model": model,
        "sensor": sensor,
        "channel": channel,
        "checkpoint": ckpt_name,
        "mode": exp_cfg["mode"],
        "label": exp_cfg["label"],
        "train_snr_db": train_snr_db,
        "alpha_l1": l1_for_mode(exp_cfg["mode"]),
    }

def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def read_result(folder):
    with (folder / 'results.csv').open(newline='') as stream:
        return next(csv.DictReader(stream))


def run_single(job, root, data_dir, cpu=False):
    """Train one configuration in a fresh process, then evaluate its selected weights."""
    folder = root / job['directory']
    folder.mkdir(parents=True, exist_ok=True)
    if cpu:
        Config.DEVICE = torch.device('cpu')
    Config.DATA_DIR = str(data_dir)
    Config.RESULTS_DIR = str(folder)
    Config.SEED = job['seed']
    Config.TRAIN_SNR_DB = job['train_snr_db']
    Config.SNR_DB = job['train_snr_db']
    Config.CS_MODE = job['cs_mode']
    Config.QUANT_BITS = job['bits']
    Config.BITS_PER_SAMPLE = Config.M * job['bits']
    Config.ALPHA_L1_ANN = Config.ALPHA_L1_SNN = job['l1']
    Config.WARM_START_CHECKPOINT = str(root / job['warm_start']) if job.get('warm_start') else None
    apply_training_overrides(job['epochs'], job['lr_scale'])
    for key, value in job.get('explicit_lrs', {}).items():
        Config.LISTA_SNN_CONFIG[key] = value
    Config.NUM_WORKERS = job.get('num_workers', Config.NUM_WORKERS)
    train_loader, eval_loader = utils.get_dataloaders()
    utils.set_seed(Config.SEED)
    exp = build_experiments(job['target'])[0]
    obj = train_or_load_fixed_p(exp, train_loader, eval_loader, Config.TRAIN_SNR_DB)
    if not (folder / obj['checkpoint']).is_file():
        raise RuntimeError(f'Training produced no selected checkpoint: {folder}')
    validation = obj['trainer'].validate_recon(mode=obj['mode'])
    _, test_loader = utils.get_dataloaders(eval_split='test')
    result = obj['trainer'].evaluate(mode=obj['mode'], loader=test_loader)
    hp = exp['hp']
    row = {
        'Target': job['target'], 'Model': obj['label'], 'M': Config.M,
        'K': Config.NUM_LAYERS, 'train_snr_db': Config.TRAIN_SNR_DB,
        'L1_Penalty': job['l1'], 'LR_Scale': job['lr_scale'], 'Epochs': job['epochs'],
        'LR_Model': hp.get('lr_model', 0),
        'LR_Threshold': hp.get('lr_theta', hp.get('lr_model', 0)),
        'LR_P': hp.get('lr_p', hp.get('lr_model', 0)),
        'LR_PD': hp.get('lr_pd', hp.get('lr_model', 0)),
        'LR_Phi': hp.get('lr_phi', 0),
        'LR_Dictionary': hp.get('lr_dictionary', Config.DICTIONARY_LR),
        'Val_NMSE_dB': validation, 'Test_NMSE_dB': result['nmse'], 'Seed': Config.SEED, 'checkpoint': obj['checkpoint'],
        'Quant_Bits': job['bits'], 'Bits_Per_Sample': Config.BITS_PER_SAMPLE,
    }
    write_csv(folder / 'results.csv', [row])


def make_jobs(settings, targets, stages, epoch_override=None):
    jobs = []
    for phase in stages:
        phase_targets = ['lista_snn_1'] if phase == 'final_slista' else targets
        if phase == 'final_slista' and 'lista_snn_1' not in targets:
            continue
        for target in phase_targets:
            values = settings[phase] if phase == 'final_slista' else settings[phase][target]
            variants = values.items() if phase == 'quantization' else [(8, values)]
            for bits, hp in variants:
                directory = phase if phase == 'final_slista' else f'{phase}/{target}'
                if phase == 'quantization':
                    directory += f'/q{bits}'
                explicit = {key: hp[key] for key in ('lr_p', 'lr_pd', 'lr_phi', 'lr_dictionary')} if phase == 'final_slista' else {}
                if any(v is None or v <= 0 for v in explicit.values()):
                    raise ValueError('Fill final_slista.lr_p and lr_pd in the configuration, or recover them with --recover-final-lrs CHECKPOINT. Other stages can run separately.')
                jobs.append({
                    'target': target, 'phase': phase, 'directory': directory,
                    'bits': int(bits), 'epochs': epoch_override or hp['epochs'],
                    'lr_scale': hp.get('lr_scale', 1.0), 'explicit_lrs': explicit,
                    'cs_mode': 'fixed_wavelet' if phase == 'fixed' else 'learnable_dictionary',
                    'seed': settings['seed'], 'train_snr_db': settings['train_snr_db'],
                    'l1': settings['l1'],
                })
    return jobs


def launch_jobs(jobs, args):
    slots = queue.Queue()
    for gpu in args.gpus:
        for _ in range(args.workers_per_gpu):
            slots.put(gpu)

    def execute(job):
        folder = args.output / job['directory']
        folder.mkdir(parents=True, exist_ok=True)
        if job['phase'] != 'fixed':
            seed_root = args.output / f"seed{job['seed']}" if job['directory'].startswith('seed') else args.output
            fixed_dir = seed_root / 'fixed' / job['target']
            fixed = read_result(fixed_dir)
            source = fixed_dir / fixed['checkpoint']
            if not source.is_file():
                raise FileNotFoundError(source)
            job['warm_start'] = str(source.relative_to(args.output))
            job['warm_start_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
        record = folder / 'job.json'
        if record.exists() and json.loads(record.read_text()) != job:
            raise ValueError(f'Configuration differs from existing job: {folder}; use a separate --output.')
        if not record.exists() and list(folder.glob('*.pth')):
            raise ValueError(f'Untracked checkpoints in {folder}; use a fresh output directory.')
        record.write_text(json.dumps(job, indent=2) + '\n')
        if (folder / 'results.csv').exists():
            row = read_result(folder)
            if not (folder / row['checkpoint']).is_file():
                raise FileNotFoundError(folder / row['checkpoint'])
            print(f"Complete: {job['directory']}", flush=True)
            return
        # Partial jobs resume from the last atomically saved epoch.
        gpu = slots.get()
        try:
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = '' if args.cpu else str(gpu)
            command = [sys.executable, '-u', str(Path(__file__).resolve()), '--worker-job', str(record),
                       '--output', str(args.output), '--data-dir', str(args.data_dir)]
            if args.cpu:
                command.append('--cpu')
            print(f"Start GPU {gpu}: {job['directory']}", flush=True)
            with (folder / 'train.log').open('a') as log:
                status = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            if status.returncode:
                raise RuntimeError(f"Training failed: {folder / 'train.log'}")
            if not (folder / 'results.csv').is_file():
                raise RuntimeError(f'Missing result: {folder}')
        finally:
            slots.put(gpu)
    with ThreadPoolExecutor(max_workers=len(args.gpus) * args.workers_per_gpu) as pool:
        list(pool.map(execute, jobs))


def aggregate(root):
    """Build portable selected-model tables from completed jobs."""
    for phase in ('fixed', 'learned', 'quantization'):
        rows = []
        for path in sorted((root / phase).glob('**/job.json')):
            folder = path.parent
            if not (folder / 'results.csv').is_file():
                continue
            r = read_result(folder)
            rows.append({
                'Target': r['Target'], 'Model': r['Model'], 'Status': 'ok',
                'LR_Scale': r['LR_Scale'], 'Epochs': r['Epochs'],
                'Val_NMSE_dB': r['Val_NMSE_dB'], 'Test_NMSE_dB': r['Test_NMSE_dB'], 'Seed': r['Seed'], 'Checkpoint': r['checkpoint'],
                'Trial_Dir': str(folder.relative_to(root / phase)),
                'Quant_Bits': r['Quant_Bits'], 'Bits_Per_Sample': r['Bits_Per_Sample'],
            })
        if rows:
            write_csv(root / phase / ('results.csv' if phase == 'quantization' else 'models.csv'), rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--data-dir', type=Path, default=Path(Config.DATA_DIR))
    parser.add_argument('--config', type=Path, default=PROJECT_DIR / 'configs/paper.json')
    parser.add_argument('--gpus', nargs='+', type=int, default=[0])
    parser.add_argument('--workers-per-gpu', type=int, default=1)
    parser.add_argument('--targets', nargs='+', choices=TARGETS, default=list(TARGETS))
    parser.add_argument('--stages', nargs='+', choices=['fixed', 'learned', 'final_slista', 'quantization'],
                        default=['fixed', 'learned', 'final_slista', 'quantization'])
    parser.add_argument('--seed', type=int)
    parser.add_argument('--epochs', type=int, help='Override every selected job duration.')
    parser.add_argument('--num-workers', type=int, default=Config.NUM_WORKERS)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--list-jobs', action='store_true', help='List selected configurations without training.')
    parser.add_argument('--recover-final-lrs', type=Path, metavar='CHECKPOINT',
                        help='Read lr_p and lr_pd from a trusted local checkpoint into --config, then exit.')
    parser.add_argument('--worker-job', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.output = args.output.expanduser().resolve()
    args.data_dir = args.data_dir.expanduser().resolve()
    if args.workers_per_gpu < 1 or (args.epochs is not None and args.epochs < 1):
        parser.error('Worker count and epoch count must be positive.')
    if args.worker_job:
        run_single(json.loads(args.worker_job.read_text()), args.output, args.data_dir, args.cpu)
        return
    settings = json.loads(args.config.read_text())
    if args.recover_final_lrs:
        ckpt = torch.load(args.recover_final_lrs, map_location='cpu', weights_only=False)
        for key in ('lr_p', 'lr_pd'):
            value = ckpt.get(key)
            if value is None or float(value) <= 0:
                raise ValueError(f'Checkpoint lacks {key}')
            settings['final_slista'][key] = float(value)
        args.config.write_text(json.dumps(settings, indent=2) + '\n')
        print(json.dumps(settings['final_slista'], indent=2))
        return
    if args.seed is not None: settings['seed'] = args.seed
    stages = [p for p in ('fixed', 'learned', 'final_slista', 'quantization') if p in args.stages]
    jobs = make_jobs(settings, args.targets, stages, args.epochs)
    source_hashes = {name: hashlib.sha256((PROJECT_DIR / name).read_bytes()).hexdigest()
                     for name in ('config.py', 'snn_models.py', 'trainer.py', 'edge_compression.py', 'wireless_channel.py', 'utils.py')}
    data_hashes = {name: hashlib.sha256((args.data_dir / name).read_bytes()).hexdigest() for name in ('train_data.pt','val_data.pt','test_data.pt','split_manifest.json')}
    for job in jobs:
        job['data_sha256'] = data_hashes
        job['data_dir'] = str(args.data_dir)
        job['num_workers'] = args.num_workers
        job['source_sha256'] = source_hashes
    if args.list_jobs:
        print(json.dumps(jobs, indent=2))
        return
    for phase in stages:
        launch_jobs([j for j in jobs if j['phase'] == phase], args)
        aggregate(args.output)
    print(f'Completed. Results: {args.output}')


if __name__ == '__main__':
    main()
