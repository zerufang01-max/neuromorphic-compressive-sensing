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
from snn_models import LAMP, LISTA, SpikingLISTA, ALISTA, FISTA
from trainer import Trainer
import utils

TARGETS = ("alista", "lamp", "lista_ann", "lista_snn_1")
def build_experiments(target: str = "all"):
    exps = [
        {"id": "fista", "name": "FISTA", "label": "FISTA", "model_class": FISTA, "mode": "ann", "hp": Config.FISTA_CONFIG},
        {"id": "alista", "name": "ALISTA", "label": "ANN ALISTA", "model_class": ALISTA, "mode": "ann", "hp": Config.ALISTA_CONFIG},
        {"id": "lamp", "name": "LAMP", "label": "ANN LAMP", "model_class": LAMP, "mode": "ann", "hp": Config.LAMP_ANN_CONFIG},
        {"id": "lista_ann", "name": "LISTA_ANN", "label": "ANN LISTA", "model_class": LISTA, "mode": "ann", "hp": Config.LISTA_ANN_CONFIG},
        {"id": "lista_snn_1", "name": "LISTA_SNN", "label": "Spiking LISTA", "model_class": SpikingLISTA, "mode": "spiking", "hp": Config.LISTA_SNN_CONFIG},
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
    if canonical_mode == "spiking":
        suffix = (
            f"_snr{train_snr_db:g}_q{Config.QUANT_BITS}_{Config.CS_MODE}"
            f"_lrs{Config.LR_SCALE:g}"
            f"_b{Config.SLISTA_BETA:g}_a{Config.SLISTA_ALPHA:g}"
        )
    else:
        threshold = model.get_threshold_value() if hasattr(model, "get_threshold_value") else 0.0
        suffix = (
            f"_th{threshold:g}_snr{train_snr_db:g}_q{Config.QUANT_BITS}"
            f"_{Config.CS_MODE}"
        )
        if Config.LR_SCALE != 1.0:
            suffix += f"_lrs{Config.LR_SCALE:g}"
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

    channel = WirelessChannel().to(Config.DEVICE)
    channel.snr_db = train_snr_db
    sensor = EdgeSensor().to(Config.DEVICE)

    if exp_cfg["model_class"] in (LAMP, LISTA, SpikingLISTA, ALISTA, FISTA):
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
    apply_method_overrides(job.get('overrides', {}))
    Config.WARM_START_CHECKPOINT = str(root / job['warm_start']) if job.get('warm_start') else None
    apply_training_overrides(job['epochs'], job['lr_scale'])
    if job['target'] == 'lista_snn_1':
        hp = Config.LISTA_SNN_CONFIG
        hp['lr_phi'] = Config.SLISTA_PHI_LR_RATIO * hp['lr_model']
        hp['lr_dictionary'] = Config.SLISTA_DICTIONARY_LR_RATIO * hp['lr_model']
        hp['lr_theta'] = Config.SLISTA_THETA_LR_RATIO * hp['lr_model']
        hp['lr_pd'] = Config.SLISTA_G_LR_RATIO * hp['lr_model']
    Config.NUM_WORKERS = job.get('num_workers', Config.NUM_WORKERS)
    train_loader, eval_loader = utils.get_dataloaders()
    utils.set_seed(Config.SEED)
    exp = build_experiments(job['target'])[0]
    obj = train_or_load_fixed_p(exp, train_loader, eval_loader, Config.TRAIN_SNR_DB)
    if not (folder / obj['checkpoint']).is_file():
        raise RuntimeError(f'Training produced no selected checkpoint: {folder}')
    validation = obj['trainer'].validate_recon(mode=obj['mode'])
    if job.get('skip_test'):
        # Searches and tuning: selection uses the validation split only.  With
        # val_stats the operation counts and firing statistics are measured on
        # the validation split as well (columns prefixed Val_).
        result = {'nmse': ''}
        if job.get('val_stats'):
            result = dict(obj['trainer'].evaluate(mode=obj['mode'], loader=eval_loader), nmse='')
    else:
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
    if exp['mode'] == 'spiking':
        model = obj['model']
        row.update({
            'Sigma_Init': float(model.calibration_sigma),
            'Theta_Learned': ' '.join(f'{v:.6g}' for v in model.threshold_values()),
            'Eta_Learned': float(model.eta),
        })
    if result.get('energy_uj') is not None:
        prefix = 'Val_' if job.get('skip_test') else ''
        row.update({
            prefix + 'Energy_uJ': result['energy_uj'], prefix + 'MACs': result['dense_macs'],
            prefix + 'ACs': result['synaptic_acs'], prefix + 'Firing_Rate': result['firing_rate'],
            prefix + 'Support_Per_Sample': result.get('support_per_sample', ''),
            prefix + 'Layer_Firing_Rates': ' '.join(f'{v:.5g}' for v in result.get('layer_firing_rates', [])),
        })
    if job.get('overrides'):
        row['Overrides'] = json.dumps(job['overrides'], sort_keys=True)
    write_csv(folder / 'results.csv', [row])


PHASES = ('fixed', 'learned', 'quantization')


# Per-method settings that configs/paper.json ("overrides") or a tuning job
# may change.  Everything else in Config stays fixed.
OVERRIDABLE = {
    'SLISTA_KAPPA': float, 'SLISTA_BETA': float, 'SLISTA_ALPHA': float,
    'SLISTA_PHI_LR_RATIO': float, 'SLISTA_DICTIONARY_LR_RATIO': float,
    'SLISTA_THETA_LR_RATIO': float, 'SLISTA_G_LR_RATIO': float,
    'THETA_LAMP': float, 'VAL_EVERY': int,
}


def apply_method_overrides(overrides):
    for key, value in overrides.items():
        if key not in OVERRIDABLE:
            raise ValueError(f'{key} cannot be overridden (allowed: {sorted(OVERRIDABLE)})')
        setattr(Config, key, OVERRIDABLE[key](value))


def method_settings(settings, target):
    """(l1, overrides) of one method from configs/paper.json; the method may
    replace the common l1 (S-LISTA spike-rate weight)."""
    overrides = dict(settings.get('overrides', {}).get(target, {}))
    l1 = float(overrides.pop('l1', settings['l1']))
    return l1, overrides


def lr_scale_for(settings, phase, target):
    """Selected learning-rate scale: fixed uses lr_scale.fixed; learned and
    quantization use lr_scale.learned (written by lr_search.py)."""
    key = 'fixed' if phase == 'fixed' else 'learned'
    value = settings.get('lr_scale', {}).get(key, {}).get(target)
    if value is None or float(value) <= 0:
        raise ValueError(
            f'configs/paper.json: lr_scale.{key}.{target} is not set. Run lr_search.py '
            f'(it writes the selected scales with --write-config).')
    return float(value)


def make_jobs(settings, targets, stages, epoch_override=None):
    """One job per (phase, target[, bits]).  The 8-bit quantization point is
    the learned-dictionary model itself (identical configuration), so the
    quantization phase lists only the lower bit widths."""
    jobs = []
    for phase in stages:
        if phase not in PHASES:
            raise ValueError(f'Unknown phase {phase}')
        for target in targets:
            values = settings[phase][target]
            variants = values.items() if phase == 'quantization' else [(8, values)]
            for bits, hp in variants:
                if phase == 'quantization' and int(bits) == 8:
                    raise ValueError('8 bit is the learned phase; remove it from quantization')
                directory = f'{phase}/{target}'
                if phase == 'quantization':
                    directory += f'/q{bits}'
                if 'lr_scale' in hp:
                    raise ValueError('Learning-rate scales belong in the lr_scale section')
                l1, overrides = method_settings(settings, target)
                job = {
                    'target': target, 'phase': phase, 'directory': directory,
                    'bits': int(bits), 'epochs': epoch_override or hp['epochs'],
                    'lr_scale': lr_scale_for(settings, phase, target),
                    'cs_mode': 'fixed_wavelet' if phase == 'fixed' else 'learnable_dictionary',
                    'seed': settings['seed'], 'train_snr_db': settings['train_snr_db'],
                    'l1': l1,
                }
                if overrides:
                    job['overrides'] = overrides
                jobs.append(job)
    return jobs


SOURCE_FILES = ('config.py', 'snn_models.py', 'trainer.py', 'edge_compression.py',
                'wireless_channel.py', 'utils.py', 'main.py')
DATA_FILES = ('train_data.pt', 'val_data.pt', 'test_data.pt', 'split_manifest.json')


def annotate_jobs(jobs, data_dir, num_workers):
    """Record data and source hashes so a job cannot silently change."""
    source_hashes = {name: hashlib.sha256((PROJECT_DIR / name).read_bytes()).hexdigest()
                     for name in SOURCE_FILES}
    data_hashes = {name: hashlib.sha256((Path(data_dir) / name).read_bytes()).hexdigest()
                   for name in DATA_FILES}
    for job in jobs:
        job['data_sha256'] = data_hashes
        job['data_dir'] = str(data_dir)
        job['num_workers'] = num_workers
        job['source_sha256'] = source_hashes
    return jobs


def gpu_slots(args):
    slots = queue.Queue()
    for gpu in args.gpus:
        for _ in range(args.workers_per_gpu):
            slots.put(gpu)
    return slots


def run_job(job, args, slots):
    """Train one job in a fresh process (or report it complete)."""
    folder = args.output / job['directory']
    folder.mkdir(parents=True, exist_ok=True)
    if job['phase'] != 'fixed':
        seed_root = args.output / f"seed{job['seed']}" if job['directory'].startswith('seed') else args.output
        # The learning-rate search names its warm-start trial explicitly.
        fixed_dir = (args.output / job['warm_start_from']) if job.get('warm_start_from') \
            else seed_root / 'fixed' / job['target']
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


def launch_jobs(jobs, args):
    slots = gpu_slots(args)
    with ThreadPoolExecutor(max_workers=len(args.gpus) * args.workers_per_gpu) as pool:
        list(pool.map(lambda job: run_job(job, args, slots), jobs))


def aggregate(root):
    """Build portable selected-model tables from completed jobs."""
    for phase in PHASES:
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
    parser.add_argument('--stages', nargs='+', choices=list(PHASES), default=list(PHASES))
    parser.add_argument('--seed', type=int)
    parser.add_argument('--epochs', type=int, help='Override every selected job duration.')
    parser.add_argument('--num-workers', type=int, default=Config.NUM_WORKERS)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--list-jobs', action='store_true', help='List selected configurations without training.')
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
    if args.seed is not None: settings['seed'] = args.seed
    stages = [p for p in PHASES if p in args.stages]
    jobs = make_jobs(settings, args.targets, stages, args.epochs)
    annotate_jobs(jobs, args.data_dir, args.num_workers)
    if args.list_jobs:
        print(json.dumps(jobs, indent=2))
        return
    for phase in stages:
        launch_jobs([j for j in jobs if j['phase'] == phase], args)
        aggregate(args.output)
    print(f'Completed. Results: {args.output}')


if __name__ == '__main__':
    main()
