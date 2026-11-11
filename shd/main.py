"""SHD experiments: single-job training, GPU scheduling and frozen evaluation."""

import argparse
import subprocess
import sys
import time
import glob
import math
import os
import random
import re
from pathlib import Path
import dataclasses
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from config import Config
from edge_compression import EdgeSensor, BinaryEdgeSensor
from snn_models import (
    HybridLISTA,
    LSTMRecon,
    SpikingDenseRecon,
    SpikingLISTA,
    TemporalConvClassifier,
)
from trainer import Trainer
from wireless_channel import (
    DenseQuantizedAWGNChannel,
    DenseSpikeAWGNChannel,
    BlockAERAWGNChannel,
)
import utils


VERSION = "SHD_SLISTA_V2"
TRAIN_SNR_DB = 10.0
SNR_GRID = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 7.5, 10, 15, 20]
SNR_KEYS = [
    "slista_dense", "scnn_dense", "ann_lista", "ann_lstm"
]
MAIN_KEYS = list(SNR_KEYS)
# "Without temperature relaxation": the TX temperature of Eq. (9) is held at
# tau_final for the whole run instead of being annealed; everything else
# (shape of the TX surrogate, channel relaxation, RX, hyperparameters) is
# identical to slista_aer.
CORE_KEYS = [
    "slista_aer",
    "slista_aer_fixed_tau",
    "slista_aer_no_rate",
    "slista_aer_no_semantic",
]
CORE_VARIANTS = {
    "slista_aer": "S-LISTA with AER",
    "slista_aer_fixed_tau": "Without temperature relaxation",
    "slista_aer_no_rate": "Without rate regularization",
    "slista_aer_no_semantic": "Without task supervision",
}
SEMANTIC_SNR_KEYS = ["slista_dense", "slista_dense_no_semantic"]
# Bit budgets (binary channels per frame; 25 frames per sample).  Multiples
# of the AER block size 16, which the Rayleigh channel class requires.
BUDGET_CHANNELS = [48, 64, 96, 128, 192]
SLISTA_RATE_VALUES = [0.02, 0.05, 0.10, 0.20, 0.30]
# AER operating points controlled by spike-rate regularization.
SCNN_RATE_VALUES = [0.06, 0.09, 0.12, 0.25, 0.50]
SLISTA_SELECTED_RATE = 0.10
SCNN_SELECTED_RATE = 0.12
RATE_KEYS = [
    "slista_rate_0p02", "slista_rate_0p05", "slista_aer",
    "slista_rate_0p20", "slista_rate_0p30",
    "scnn_rate_0p06", "scnn_rate_0p09", "scnn_aer",
    "scnn_rate_0p25", "scnn_rate_0p50",
    "ann_lista", "ann_lstm",
]
RATE_JOB_KEYS = [
    key for key in RATE_KEYS
    if key not in {"ann_lista", "ann_lstm"}
]
# Baselines.  Existing checkpoints can be re-evaluated with --eval-only.
BASELINE_KEYS = [
    "ann_lista", "ann_lstm",
    "scnn_dense", "scnn_aer",
    "scnn_rate_0p06", "scnn_rate_0p09",
    "scnn_rate_0p25", "scnn_rate_0p50",
]
SLISTA_KEYS = [
    "slista_dense",
    "slista_aer",
    "slista_rate_0p02",
    "slista_rate_0p05",
    "slista_rate_0p20",
    "slista_rate_0p30",
    "slista_aer_fixed_tau",
    "slista_aer_no_rate",
    "slista_aer_no_semantic",
    "slista_dense_no_semantic",
]


@dataclass(frozen=True)
class Experiment:
    key: str
    display_name: str
    model_class: type
    mode: str
    transport: str
    quant_bits: int | None
    encoder_threshold: float | None
    decoder_tau: float | None
    alpha_rate: float
    semantic_weight: float
    tx_gradient: str = "temperature"
    # Binary channels transmitted per frame; a dense sample carries
    # channels * TIME_STEPS bits (bit-budget experiments vary it).
    channels: int = 256


def experiment_registry():
    S = SpikingLISTA
    tx_slista = Config.SLISTA_TX_THRESHOLD
    sem = float(Config.SLISTA_SEMANTIC_WEIGHT)
    registry = {
        "slista_aer": Experiment(
            "slista_aer", "S-LISTA", S, "spiking",
            "block_aer_awgn_hard", None, tx_slista, None,
            SLISTA_SELECTED_RATE, sem,
        ),
        "scnn_aer": Experiment(
            "scnn_aer", "Vanilla SNN", SpikingDenseRecon, "spiking",
            "block_aer_awgn_hard", None,
            Config.SCNN_TX_THRESHOLD, Config.SCNN_DECODER_TAU,
            SCNN_SELECTED_RATE, 3.0,
        ),
        "ann_lista": Experiment(
            "ann_lista", "ANN LISTA", HybridLISTA, "ann",
            "dense_q8_bitawgn_hard", 8, None, None, 1e-2, 3.0,
        ),
        "ann_lstm": Experiment(
            "ann_lstm", "ANN LSTM", LSTMRecon, "ann",
            "dense_q8_bitawgn_hard", 8, None, None, 1e-2, 3.0,
        ),
        "slista_dense": Experiment(
            "slista_dense", "S-LISTA", S, "spiking",
            "dense_spike_awgn_hard", None, tx_slista, None,
            0.0, sem,
        ),
        "scnn_dense": Experiment(
            "scnn_dense", "Vanilla SNN", SpikingDenseRecon, "spiking",
            "dense_spike_awgn_hard", None,
            Config.SCNN_TX_THRESHOLD, Config.SCNN_DECODER_TAU,
            0.0, 3.0,
        ),
        "slista_aer_fixed_tau": Experiment(
            "slista_aer_fixed_tau", "S-LISTA", S, "spiking",
            "block_aer_awgn_hard", None, tx_slista, None,
            SLISTA_SELECTED_RATE, sem, "fixed_temperature",
        ),
        "slista_aer_no_rate": Experiment(
            "slista_aer_no_rate", "S-LISTA", S, "spiking",
            "block_aer_awgn_hard", None, tx_slista, None,
            0.0, sem,
        ),
        "slista_aer_no_semantic": Experiment(
            "slista_aer_no_semantic", "S-LISTA", S, "spiking",
            "block_aer_awgn_hard", None, tx_slista, None,
            SLISTA_SELECTED_RATE, 0.0,
        ),
        "slista_dense_no_semantic": Experiment(
            "slista_dense_no_semantic", "S-LISTA", S, "spiking",
            "dense_spike_awgn_hard", None, tx_slista, None,
            0.0, 0.0,
        ),
    }
    # Bit-budget points: dense transmission with fewer binary channels per
    # frame; every other setting is that of slista_dense / scnn_dense.
    for channels in BUDGET_CHANNELS:
        for base, key in ((registry["slista_dense"], f"slista_dense_c{channels}"),
                          (registry["scnn_dense"], f"scnn_dense_c{channels}")):
            registry[key] = dataclasses.replace(base, key=key, channels=int(channels))
    for value in SLISTA_RATE_VALUES:
        if np.isclose(value, SLISTA_SELECTED_RATE):
            continue
        tag = f"{value:.2f}".replace(".", "p")
        key = f"slista_rate_{tag}"
        registry[key] = Experiment(
            key, "S-LISTA", S, "spiking",
            "block_aer_awgn_hard", None, tx_slista, None,
            float(value), sem,
        )
    for value in SCNN_RATE_VALUES:
        if np.isclose(value, SCNN_SELECTED_RATE):
            continue
        tag = f"{value:.2f}".replace(".", "p")
        key = f"scnn_rate_{tag}"
        registry[key] = Experiment(
            key, "Vanilla SNN", SpikingDenseRecon, "spiking",
            "block_aer_awgn_hard", None,
            Config.SCNN_TX_THRESHOLD, Config.SCNN_DECODER_TAU,
            float(value), 3.0,
        )
    return registry


def set_seed(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def atomic_csv(frame, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.tmp.{os.getpid()}"
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def configure(experiment, seed):
    Config.SEED = int(seed)
    Config.MEAS_DIM = Config.M = int(experiment.channels)
    Config.RATE_BITS_PER_EVENT = max(1, math.ceil(math.log2(Config.MEAS_DIM)))
    Config.TRAIN_SNR_DB = TRAIN_SNR_DB
    Config.EVAL_SNR_DB_LIST = list(SNR_GRID)
    Config.ALPHA_SEMANTIC = float(experiment.semantic_weight)
    Config.SEMANTIC_DETACH_BACKBONE = False
    # The temperature ablation only changes the TX annealing schedule.
    Config.TX_ABLATION_MODE = "none"
    if experiment.tx_gradient == "fixed_temperature":
        Config.TAU_ANNEAL_TYPE = "fixed"
        Config.TAU_FIXED = float(Config.TAU_FINAL)
    elif experiment.tx_gradient == "temperature":
        Config.TAU_ANNEAL_TYPE = "exp"
    else:
        raise ValueError(f"Unknown TX gradient setting: {experiment.tx_gradient}")
    Config.SLISTA_P_WEIGHT_DECAY = 0.0
    Config.DOWNSTREAM_LR_OVERRIDE = getattr(Config, "_CLI_CLASSIFIER_LR", 1e-3)
    Config.RECON_LR_OVERRIDE = getattr(Config, "_CLI_RECON_LR", None)
    Config.ANN_QUANT_BITS = 8
    Config.ANN_RX_MODE = "hard"
    if experiment.mode == "spiking":
        Config.SNN_TRANSPORT = experiment.transport
        if experiment.decoder_tau is not None:
            Config.DECODER_TAU = float(experiment.decoder_tau)
        if experiment.model_class is SpikingDenseRecon:
            Config.ALPHA_RATE_SCNN = float(experiment.alpha_rate)
        else:
            Config.ALPHA_RATE_SLISTA = float(experiment.alpha_rate)


def experiment_hp(experiment, epochs=None):
    if experiment.key.startswith("slista_"):
        hp = dict(Config.SLISTA_PHASE3_CONFIG)
        # One overall multiplier for the reconstruction path (P, G, eta,
        # thresholds via their ratios, sensor and dictionary).  The task head
        # keeps its own learning rate.
        scale = float(Config.SLISTA_LR_SCALE)
        for name in ("lr_lista_snn", "lr_sensor", "lr_dict"):
            hp[name] = float(hp[name]) * scale
    elif experiment.key.startswith("scnn_"):
        hp = dict(Config.SCNN_PHASE3_CONFIG)
    else:
        hp = dict(Config.PHASE3_CONFIG)
    if epochs is not None:
        hp["epochs"] = int(epochs)
    if (
        experiment.key.startswith("scnn_")
        and Config.RECON_LR_OVERRIDE is not None
    ):
        hp["lr_spiking_dense"] = float(Config.RECON_LR_OVERRIDE)
    return hp


def build_components(experiment):
    if experiment.mode == "ann":
        sensor = EdgeSensor()
        channel = DenseQuantizedAWGNChannel(
            snr_db=TRAIN_SNR_DB, receiver="hard"
        )
    else:
        sensor = BinaryEdgeSensor(
            threshold=float(experiment.encoder_threshold),
            normalization=Config.SENSING_NORMALIZATION,
        )
        channel_class = (
            BlockAERAWGNChannel
            if experiment.transport == "block_aer_awgn_hard"
            else DenseSpikeAWGNChannel
        )
        channel = channel_class(snr_db=TRAIN_SNR_DB)
    model = experiment.model_class()
    if hasattr(model, "mode"):
        model.mode = "rate" if experiment.mode == "ann" else "spiking"
    return (
        model.to(Config.DEVICE),
        sensor.to(Config.DEVICE),
        channel.to(Config.DEVICE),
    )


def experiment_artifact_key(experiment):
    # Tags used in checkpoint and result file names.
    if experiment.model_class is SpikingDenseRecon:
        return f"{experiment.key}_compact3x700_tx1p5_nth0p8_leak0p06_stau0p3_lr6em4"
    if experiment.model_class is SpikingLISTA:
        # S-LISTA hyperparameters are recorded inside the checkpoint and in
        # every result row.
        return experiment.key
    return experiment.key


def recon_model_name(experiment, seed):
    tag = experiment_artifact_key(experiment)
    if Config.RECON_LR_OVERRIDE is not None:
        tag += f"_rlr{float(Config.RECON_LR_OVERRIDE):g}".replace(".", "p")
    return f"formal_{tag}_seed{int(seed)}"


def recon_checkpoint_path(experiment, seed, model=None):
    if model is None:
        model = experiment.model_class()
        if hasattr(model, "mode"):
            model.mode = "rate" if experiment.mode == "ann" else "spiking"
    threshold = (
        model.get_threshold_value()
        if hasattr(model, "get_threshold_value") else None
    )
    name = Config.get_recon_ckpt_name(
        experiment.mode,
        recon_model_name(experiment, seed),
        threshold,
        experiment.alpha_rate,
        hp=experiment_hp(experiment),
    )
    return os.path.join(Config.RESULTS_DIR, name)


def classifier_checkpoint_path(experiment, seed):
    tag = experiment_artifact_key(experiment)
    return os.path.join(Config.RESULTS_DIR,
        f'downstream_recon_x_{tag}_seed{seed}_{Config.classifier_protocol()}.pth')


def point_path(experiment, seed):
    suffix = ""
    if Config.RECON_LR_OVERRIDE is not None:
        suffix = f"_rlr{float(Config.RECON_LR_OVERRIDE):g}_clr{float(Config.DOWNSTREAM_LR_OVERRIDE):g}".replace(".", "p")
    return os.path.join(
        Config.RESULTS_DIR,
        f"formal_point_{experiment.key}_seed{int(seed)}{suffix}.csv",
    )


def attach_downstream_classifier(trainer, experiment, seed):
    set_seed(int(seed) + int(Config.DOWNSTREAM_INIT_SEED_OFFSET))
    classifier = TemporalConvClassifier(
        input_dim=Config.INPUT_DIM,
        num_classes=Config.NUM_CLASSES,
        frame_dim=Config.DOWNSTREAM_TCN_DIM,
        dropout=Config.DOWNSTREAM_DROPOUT,
        norm_groups=Config.DOWNSTREAM_NORM_GROUPS,
        channel_pool=Config.DOWNSTREAM_CHANNEL_POOL,
        log_input=Config.DOWNSTREAM_LOG_INPUT,
    ).to(Config.DEVICE)
    trainer.set_downstream_classifier(classifier)
    return classifier


def resolve_existing_recon(recorded):
    path = Path(str(recorded))
    for candidate in (Path(Config.RESULTS_DIR)/path.name, path):
        if candidate.is_file():
            return candidate.resolve()
    pattern = r'_th[-+]?(?:\d+(?:\.\d*)?|\.\d+)\.pth$'
    prefix = re.sub(pattern, '', path.name)
    candidates = []
    if prefix != path.name:
        for folder in {Path(Config.RESULTS_DIR), path.parent}:
            candidates.extend(p.resolve() for p in folder.glob('*.pth')
                              if re.sub(pattern, '', p.name) == prefix)
    candidates = sorted(set(candidates))
    if len(candidates) != 1:
        raise FileNotFoundError(f'Expected one existing reconstruction for {path}; found {candidates}')
    return candidates[0]


# Values assumed for S-LISTA settings that a checkpoint does not record.
SLISTA_UNRECORDED_DEFAULTS = {
    "SLISTA_TX_THRESHOLD": 0.75, "SLISTA_SEMANTIC_WEIGHT": 3.0,
    "SLISTA_WEIGHT_DECAY_P": 0.0, "SLISTA_WEIGHT_DECAY_G": 0.0,
}


def slista_settings_mismatch(path):
    """For an S-LISTA checkpoint, the settings that differ from the current
    configuration ({} if they match or the file is not an S-LISTA model)."""
    if not os.path.isfile(path):
        return {}
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    info = checkpoint.get("slista")
    if not info:
        return {}
    stored = dict(SLISTA_UNRECORDED_DEFAULTS, **(info.get("settings") or {}))
    current = Config.slista_settings()
    return {name: (stored.get(name), value) for name, value in current.items()
            if stored.get(name) is None or abs(float(stored[name]) - value) > 1e-12}


def train_or_load(experiment, seed, args, train_loader, val_loader):
    configure(experiment, seed)
    recorded_recon = None
    source_csv = Path(point_path(experiment, seed))
    if source_csv.is_file() and not args.force_retrain:
        frame = pd.read_csv(source_csv)
        paths = frame['Recon Checkpoint'].dropna().unique()
        if len(paths) != 1:
            raise RuntimeError(f'Ambiguous reconstruction provenance: {source_csv}')
        recorded_recon = resolve_existing_recon(paths[0])
    set_seed(seed)
    model, sensor, channel = build_components(experiment)
    trainer = Trainer(model, sensor, channel, train_loader, val_loader)
    # With --force-retrain the recorded path is ignored, so the freshly
    # trained checkpoint (not an older file) is the one loaded below.
    recon_path = str(recorded_recon) if recorded_recon else recon_checkpoint_path(experiment, seed, model)
    if args.force_retrain_classifier and not os.path.isfile(recon_path):
        raise FileNotFoundError(f'Classifier refresh requires an existing reconstruction: {recon_path}')
    retrained_recon = False
    stale = slista_settings_mismatch(recon_path) if not args.force_retrain else {}
    if stale:
        print(f"[Stale S-LISTA checkpoint] {recon_path} was trained with other settings "
              f"(stored, current): {stale}; retraining", flush=True)
        if args.eval_only:
            raise RuntimeError(f"Stale S-LISTA checkpoint in --eval-only: {recon_path}")
        recon_path = recon_checkpoint_path(experiment, seed, model)
    if args.force_retrain or stale or not os.path.exists(recon_path):
        if args.eval_only:
            raise FileNotFoundError(recon_path)
        if getattr(args, "classifier_only", False):
            raise RuntimeError(f"--classifier-only: {experiment.key} seed{seed} would retrain "
                               f"its reconstruction model ({recon_path}; stale settings: {stale})")
        trainer.train_dynamic_recon(
            hp=experiment_hp(experiment, args.epochs),
            model_name=recon_model_name(experiment, seed),
            mode=experiment.mode,
            freeze_sensor=False,
            alpha_rate=experiment.alpha_rate,
        )
        retrained_recon = True
        if not os.path.isfile(recon_path):
            raise FileNotFoundError(
                f'Training did not write the expected checkpoint: {recon_path}'
            )
    checkpoint = torch.load(
        recon_path, map_location=Config.DEVICE, weights_only=False
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    sensor.load_state_dict(checkpoint["sensor"], strict=True)
    channel.set_snr_db(TRAIN_SNR_DB)
    for module in (model, sensor):
        module.eval()
        module.requires_grad_(False)

    classifier = attach_downstream_classifier(trainer, experiment, seed)
    classifier_path = classifier_checkpoint_path(experiment, seed)
    # A classifier trained on an older reconstruction model is never reused.
    if (
        args.force_retrain_classifier or retrained_recon
        or not os.path.exists(classifier_path)
    ):
        if args.eval_only:
            raise FileNotFoundError(classifier_path)
        hp = dict(Config.PHASE4_CONFIG)
        if args.epochs is not None:
            hp["epochs"] = int(args.epochs)
        trainer.train_phase4(
            hp=hp,
            model_name=recon_model_name(experiment, seed),
            mode=experiment.mode,
            target="recon",
            save_path=classifier_path,
        )
    classifier_state = torch.load(
        classifier_path, map_location=Config.DEVICE, weights_only=False
    )
    classifier.load_state_dict(classifier_state["classifier"], strict=True)
    classifier.eval()
    return trainer, checkpoint, recon_path, classifier_path


def evaluation_snrs(experiment):
    if getattr(Config, "_CLI_EVAL_10_ONLY", False):
        return [TRAIN_SNR_DB]
    if experiment.key in SNR_KEYS or experiment.key in SEMANTIC_SNR_KEYS:
        return SNR_GRID
    return [TRAIN_SNR_DB]


def save_z_features(trainer, experiment, seed):
    if experiment.key not in SEMANTIC_SNR_KEYS:
        return
    trainer.channel.set_snr_db(TRAIN_SNR_DB)
    sequence, labels = trainer.collect_z_sequence(
        mode=experiment.mode,
        max_samples=Config.TSNE_COMPARISON_MAX_PER_REGIME,
    )
    features = sequence.reshape(sequence.shape[0], -1)
    np.savez_compressed(
        os.path.join(
            Config.RESULTS_DIR,
            f"z_features_{experiment.key}_seed{int(seed)}.npz",
        ),
        features=features,
        labels=labels,
        seed=np.asarray([int(seed)]),
        semantic_weight=np.asarray([float(experiment.semantic_weight)]),
    )


def slista_columns(checkpoint):
    """Provenance of an S-LISTA checkpoint (empty for other models)."""
    info = checkpoint.get("slista") if isinstance(checkpoint, dict) else None
    if not info:
        return {}
    columns = {f"S-LISTA {name}": value for name, value in info["settings"].items()}
    calibration = info.get("calibration") or {}
    columns["S-LISTA Calibration Sigma"] = calibration.get("sigma", np.nan)
    columns["S-LISTA Active Frame Fraction"] = calibration.get(
        "active_frame_fraction", np.nan
    )
    columns["TAU Anneal Type"] = checkpoint.get("tau_anneal_type", "")
    return columns


def evaluate_grid(trainer, experiment, checkpoint, seed, repeats,
                  recon_path=None, classifier_path=None):
    trainer.channel.set_snr_db(TRAIN_SNR_DB)
    bits = trainer.evaluate_bits(mode=experiment.mode)
    extra = slista_columns(checkpoint)
    if trainer._is_spiking_lista():
        extra.update(trainer.slista_diagnostics(mode=experiment.mode))
    parameter_counts = utils.count_reconstruction_parameters(
        trainer.model, trainer.sensor, mode=experiment.mode
    )
    rows = []
    original_channel_seed = int(Config.EVAL_CHANNEL_SEED)
    for snr_db in evaluation_snrs(experiment):
        trainer.channel.set_snr_db(float(snr_db))
        metrics = []
        for repeat in range(int(repeats)):
            Config.EVAL_CHANNEL_SEED = original_channel_seed + repeat
            joint = trainer.validate_joint_metrics(mode=experiment.mode)
            joint["Task Accuracy (%)"] = trainer.validate_cls(
                mode=experiment.mode, target="recon"
            )
            metrics.append(joint)

        def mean_std(name):
            values = np.asarray([float(item[name]) for item in metrics])
            return float(values.mean()), float(values.std(ddof=0))

        nmse, nmse_noise_std = mean_std("Temporal-Mean NMSE (dB)")
        full_nmse, _ = mean_std("Full-Sequence NMSE (dB)")
        frame_nmse, _ = mean_std("Framewise Mean NMSE (dB)")
        accuracy, accuracy_noise_std = mean_std("Task Accuracy (%)")
        sigma = Config.awgn_sigma(snr_db)
        ber = 0.5 * math.erfc(0.5 / (math.sqrt(2.0) * sigma))
        row = {
            "Experiment Key": experiment.key,
            "Display Model": experiment.display_name,
            "Mode": experiment.mode,
            "Transport": experiment.transport,
            "Seed": int(seed),
            "Train SNR (dB)": TRAIN_SNR_DB,
            "Eval SNR (dB)": float(snr_db),
            "Quant Bits": 8 if experiment.mode == "ann" else 1,
            "Semantic Weight": float(experiment.semantic_weight),
            "Alpha Rate": float(experiment.alpha_rate),
            "TX Gradient": experiment.tx_gradient,
            "Mean NMSE (dB)": nmse,
            "Noise-repeat NMSE Std": nmse_noise_std,
            "Full-Sequence NMSE (dB)": full_nmse,
            "Framewise Mean NMSE (dB)": frame_nmse,
            "Accuracy (%)": accuracy,
            "Noise-repeat Accuracy Std": accuracy_noise_std,
            "Bits/Sample": float(bits["bits_per_sample"]),
            "Measurements/Sample": float(bits["measurements"]),
            "Tx Firing Rate": (
                float(bits["measurements"])
                / float(Config.MEAS_DIM * Config.TIME_STEPS)
                if experiment.mode == "spiking" else 0.0
            ),
            "Noise Sigma": float(sigma),
            "Theoretical BER": float(ber),
            "Selected Epoch": int(checkpoint["epoch"]),
            # The files actually loaded.
            "Recon Checkpoint": str(
                recon_path or recon_checkpoint_path(experiment, seed)
            ),
            "Classifier Checkpoint": str(
                classifier_path or classifier_checkpoint_path(experiment, seed)
            ),
            "Code Version": Config.CODE_VERSION,
        }
        row.update(parameter_counts)
        row.update(extra)
        rows.append(row)
    Config.EVAL_CHANNEL_SEED = original_channel_seed
    trainer.channel.set_snr_db(TRAIN_SNR_DB)
    return rows


def run_one(args):
    registry = experiment_registry()
    experiment = registry[args.experiment]
    train_loader, val_loader = utils.get_dataloaders(Config.BATCH_SIZE_RECON)
    trainer, checkpoint, recon_path, classifier_path = train_or_load(
        experiment, args.seed, args, train_loader, val_loader
    )
    rows = evaluate_grid(
        trainer, experiment, checkpoint, args.seed, args.noise_repeats,
        recon_path=recon_path, classifier_path=classifier_path,
    )
    result = pd.DataFrame(rows)
    result['Classifier Protocol'] = Config.classifier_protocol()
    if experiment.key in CORE_VARIANTS:
        result['Ablation Label'] = CORE_VARIANTS[experiment.key]
    atomic_csv(result, point_path(experiment, args.seed))
    save_z_features(trainer, experiment, args.seed)
    print(
        f"[Done] {experiment.key} | seed={args.seed} | "
        f"{point_path(experiment, args.seed)}",
        flush=True,
    )


def recon_point_path(experiment, seed):
    return os.path.join(
        Config.RESULTS_DIR,
        f"recon_point_{experiment.key}_seed{int(seed)}.csv",
    )


# Robustness summary: the low-SNR AWGN band (where the
# AWGN curves bend) and the full Rayleigh grid.
ROBUST_AWGN_BAND = [snr for snr in SNR_GRID if snr <= 5]


def robustness_metrics(trainer, experiment, downstream=False):
    """NMSE and joint task-head accuracy over the full AWGN and quasi-static
    Rayleigh grids, with the same fixed noise/fading realisations as the
    formal evaluations (evaluate_grid / evaluate_rayleigh)."""
    from wireless_channel import FrozenRayleighChannel
    mode = experiment.mode
    original = trainer.channel
    original_seed = int(Config.EVAL_CHANNEL_SEED)
    curves = {"AWGN": {}, "Rayleigh": {}}
    downstream_acc = {"AWGN": {}, "Rayleigh": {}}
    try:
        for snr in SNR_GRID:
            original.set_snr_db(float(snr))
            joint = trainer.validate_joint_metrics(mode=mode)
            curves["AWGN"][snr] = (joint["Temporal-Mean NMSE (dB)"], joint["Joint Accuracy"])
            if downstream:
                # Same protocol as evaluate_grid (deterministic AWGN draw).
                downstream_acc["AWGN"][snr] = trainer.validate_cls(mode=mode, target="recon")
        fading = FrozenRayleighChannel(20.0, experiment.transport).to(Config.DEVICE)
        trainer.channel = fading
        for snr in SNR_GRID:
            fading.set_snr_db(float(snr))
            fading.reset_rng(original_seed)
            joint = trainer.validate_joint_metrics(mode=mode)
            curves["Rayleigh"][snr] = (joint["Temporal-Mean NMSE (dB)"], joint["Joint Accuracy"])
            if downstream:
                # Same protocol as evaluate_rayleigh (fades/bit errors reset).
                fading.eval()
                fading.reset_rng(original_seed)
                downstream_acc["Rayleigh"][snr] = trainer.validate_cls(mode=mode, target="recon")
    finally:
        trainer.channel = original
        original.set_snr_db(TRAIN_SNR_DB)
        Config.EVAL_CHANNEL_SEED = original_seed
    result = {}
    for channel, curve in curves.items():
        for snr, (nmse, head_acc) in curve.items():
            result[f"{channel} NMSE @{snr:g}"] = float(nmse)
            result[f"{channel} Head Acc @{snr:g}"] = float(head_acc)
    low = [curves["AWGN"][snr] for snr in ROBUST_AWGN_BAND]
    fade = list(curves["Rayleigh"].values())
    result["AWGN Low NMSE (dB)"] = float(np.mean([v[0] for v in low]))
    result["Rayleigh Mean NMSE (dB)"] = float(np.mean([v[0] for v in fade]))
    result["Robust NMSE (dB)"] = float(np.mean([v[0] for v in low + fade]))
    result["AWGN Low Head Acc (%)"] = float(np.mean([v[1] for v in low]))
    result["Rayleigh Mean Head Acc (%)"] = float(np.mean([v[1] for v in fade]))
    result["Robust Head Acc (%)"] = float(np.mean([v[1] for v in low + fade]))
    if downstream:
        for channel, curve in downstream_acc.items():
            for snr, value in curve.items():
                result[f"{channel} Acc @{snr:g}"] = float(value)
        low_acc = [downstream_acc["AWGN"][snr] for snr in ROBUST_AWGN_BAND]
        fade_acc = list(downstream_acc["Rayleigh"].values())
        result["AWGN Low Acc (%)"] = float(np.mean(low_acc))
        result["Rayleigh Mean Acc (%)"] = float(np.mean(fade_acc))
        result["Robust Acc (%)"] = float(np.mean(low_acc + fade_acc))
    return result


def run_recon_only(args):
    """Train (or load) the reconstruction model only and evaluate it on the
    evaluation split: AWGN 10 dB (training channel), S-LISTA diagnostics and
    the robustness grids.  The downstream classifier is not
    trained, the jointly trained task head provides an accuracy proxy.  If
    the checkpoint already exists, nothing is retrained (re-evaluation only)."""
    experiment = experiment_registry()[args.experiment]
    train_loader, val_loader = utils.get_dataloaders(Config.BATCH_SIZE_RECON)
    configure(experiment, args.seed)
    set_seed(args.seed)
    model, sensor, channel = build_components(experiment)
    trainer = Trainer(model, sensor, channel, train_loader, val_loader)
    recon_path = recon_checkpoint_path(experiment, args.seed, model)
    point = recon_point_path(experiment, args.seed)
    previous = pd.read_csv(point).iloc[0].to_dict() if os.path.isfile(point) else {}
    started = time.time()
    trained = False
    if args.force_retrain or not os.path.exists(recon_path):
        if getattr(args, "classifier_only", False):
            raise RuntimeError(f"--classifier-only: {experiment.key} seed{args.seed} would retrain "
                               f"its reconstruction model ({recon_path})")
        trainer.train_dynamic_recon(
            hp=experiment_hp(experiment, args.epochs),
            model_name=recon_model_name(experiment, args.seed),
            mode=experiment.mode,
            freeze_sensor=False,
            alpha_rate=experiment.alpha_rate,
        )
        trained = True
    train_minutes = (
        (time.time() - started) / 60.0 if trained
        else float(previous.get("Train Minutes", np.nan))
    )
    checkpoint = torch.load(recon_path, map_location=Config.DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    sensor.load_state_dict(checkpoint["sensor"], strict=True)
    channel.set_snr_db(TRAIN_SNR_DB)
    joint = trainer.validate_joint_metrics(mode=experiment.mode)
    bits = trainer.evaluate_bits(mode=experiment.mode)
    row = {
        "Experiment Key": experiment.key,
        "Seed": int(args.seed),
        "Epochs": int(experiment_hp(experiment, args.epochs)["epochs"]),
        "Selected Epoch": int(checkpoint["epoch"]),
        "TX Threshold": float(experiment.encoder_threshold),
        "Mean NMSE (dB)": float(joint["Temporal-Mean NMSE (dB)"]),
        "Full-Sequence NMSE (dB)": float(joint["Full-Sequence NMSE (dB)"]),
        "Framewise Mean NMSE (dB)": float(joint["Framewise Mean NMSE (dB)"]),
        "Task Head Accuracy (%)": float(joint["Joint Accuracy"]),
        "Bits/Sample": float(bits["bits_per_sample"]),
        "Tx Firing Rate": float(bits["measurements"]) / float(Config.MEAS_DIM * Config.TIME_STEPS),
        "Train Minutes": train_minutes,
        "Recon Checkpoint": str(recon_path),
        "Code Version": Config.CODE_VERSION,
    }
    row.update(slista_columns(checkpoint))
    if trainer._is_spiking_lista():
        row.update(trainer.slista_diagnostics(mode=experiment.mode))
    if args.downstream:
        # Downstream classifier on reconstructed x, exactly as in the formal
        # protocol (frozen sensor/reconstruction, AWGN 10 dB, 150 epochs).
        for module in (model, sensor):
            module.eval()
            module.requires_grad_(False)
        classifier = attach_downstream_classifier(trainer, experiment, args.seed)
        classifier_path = classifier_checkpoint_path(experiment, args.seed)
        if trained or args.force_retrain_classifier or not os.path.exists(classifier_path):
            hp = dict(Config.PHASE4_CONFIG)
            if args.epochs is not None:
                hp["epochs"] = int(args.epochs)
            trainer.train_phase4(
                hp=hp, model_name=recon_model_name(experiment, args.seed),
                mode=experiment.mode, target="recon", save_path=classifier_path,
            )
        state = torch.load(classifier_path, map_location=Config.DEVICE, weights_only=False)
        classifier.load_state_dict(state["classifier"], strict=True)
        classifier.eval()
        channel.set_snr_db(TRAIN_SNR_DB)
        row["Accuracy (%)"] = float(trainer.validate_cls(mode=experiment.mode, target="recon"))
        row["Classifier Checkpoint"] = str(classifier_path)
    started = time.time()
    row.update(robustness_metrics(trainer, experiment, downstream=args.downstream))
    row["Robustness Eval Minutes"] = (time.time() - started) / 60.0
    atomic_csv(pd.DataFrame([row]), point)
    print(
        f"[Recon-only] {experiment.key} seed={args.seed}: "
        f"AWGN10 {row['Mean NMSE (dB)']:.3f} dB | robust "
        f"{row['Robust NMSE (dB)']:.3f} dB | Rayleigh mean "
        f"{row['Rayleigh Mean NMSE (dB)']:.3f} dB | head acc "
        f"{row['Task Head Accuracy (%)']:.2f}% | {point}",
        flush=True,
    )


def read_points():
    frames = [
        pd.read_csv(path)
        for path in sorted(path for path in glob.glob(
            os.path.join(Config.RESULTS_DIR, "formal_point_*_seed*.csv")
        ) if "_rlr" not in path)
    ]
    if not frames:
        return pd.DataFrame()
    raw = pd.concat(frames, ignore_index=True)
    if not raw['Classifier Checkpoint'].astype(str).str.contains(Config.classifier_protocol(), regex=False).all():
        stale = sorted(raw.loc[~raw['Classifier Checkpoint'].astype(str).str.contains(
            Config.classifier_protocol(), regex=False), 'Experiment Key'].unique())
        raise RuntimeError(f'Formal points still use another classifier protocol than '
                           f'{Config.classifier_protocol()} ({stale}); finish all classification jobs before summary.')
    raw.loc[
        raw["Display Model"].isin(
            ["Spiking CNN", "Generic SNN", "SCNN"]
        ), "Display Model"
    ] = "Vanilla SNN"
    return raw.drop_duplicates(
        ["Experiment Key", "Seed", "Eval SNR (dB)"], keep="last"
    )


def mean_std_table(raw, group_columns):
    metrics = [
        "Mean NMSE (dB)", "Accuracy (%)", "Bits/Sample",
        "Measurements/Sample", "Tx Firing Rate",
    ]
    rows = []
    for keys, part in raw.groupby(group_columns, sort=False, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_columns, keys))
        row["N Seeds"] = int(part["Seed"].nunique())
        for metric in metrics:
            values = pd.to_numeric(part[metric], errors="coerce").dropna()
            row[f"{metric} Mean"] = float(values.mean())
            row[f"{metric} Std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
        rows.append(row)
    return pd.DataFrame(rows)


def save_subset(raw, keys, stem, variants=None):
    subset = raw[raw["Experiment Key"].isin(keys)].copy()
    if variants:
        subset["Variant"] = subset["Experiment Key"].map(variants)
    atomic_csv(subset, os.path.join(Config.RESULTS_DIR, f"{stem}_per_seed.csv"))
    groups = (["Variant"] if variants else ["Display Model"]) + ["Eval SNR (dB)"]
    atomic_csv(
        mean_std_table(subset, groups),
        os.path.join(Config.RESULTS_DIR, f"{stem}_mean_std.csv"),
    )


def summarize():
    raw = read_points()
    if raw.empty:
        raise RuntimeError("No formal point files were found.")
    main = raw[
        raw["Experiment Key"].isin(MAIN_KEYS)
        & np.isclose(raw["Eval SNR (dB)"], TRAIN_SNR_DB)
    ].copy()
    atomic_csv(main, os.path.join(Config.RESULTS_DIR, "main_results_per_seed.csv"))
    atomic_csv(
        mean_std_table(main, ["Display Model"]),
        os.path.join(Config.RESULTS_DIR, "main_results_mean_std.csv"),
    )
    rate = raw[
        raw["Experiment Key"].isin(RATE_KEYS)
        & np.isclose(raw["Eval SNR (dB)"], TRAIN_SNR_DB)
    ].copy()
    atomic_csv(
        rate,
        os.path.join(Config.RESULTS_DIR, "performance_bit_per_seed.csv"),
    )
    atomic_csv(
        mean_std_table(
            rate,
            ["Display Model", "Alpha Rate", "Eval SNR (dB)"],
        ),
        os.path.join(Config.RESULTS_DIR, "performance_bit_mean_std.csv"),
    )
    save_subset(raw, SNR_KEYS, "snr_robustness")
    save_subset(
        raw, CORE_KEYS, "core_ablation", CORE_VARIANTS,
    )
    save_subset(
        raw, SEMANTIC_SNR_KEYS, "semantic_snr_robustness",
        {
            "slista_dense": "With semantic supervision",
            "slista_dense_no_semantic": "Without semantic supervision",
        },
    )
    print(f"[Summary] {Config.RESULTS_DIR}")


class OperationCounter:
    def __init__(self):
        self.counts = {
            "sender": {"MAC": 0.0, "AC": 0.0},
            "reconstruction": {"MAC": 0.0, "AC": 0.0},
        }
        self.handles = []

    def add(self, layer, operation, scope):
        def hook(module, inputs, output):
            x = inputs[0]
            dense = float(
                x.shape[0] * module.in_features * module.out_features
            )
            if operation == "MAC":
                self.counts[scope]["MAC"] += dense
            else:
                self.counts[scope]["AC"] += dense * float(
                    x.detach().abs().sum().item() / max(1, x.numel())
                )
        self.handles.append(layer.register_forward_hook(hook))

    def close(self):
        for handle in self.handles:
            handle.remove()


def attach_operation_hooks(counter, experiment, model, sensor):
    if experiment.mode == "ann":
        counter.add(sensor.fc, "MAC", "sender")
    if isinstance(model, SpikingLISTA):
        # P on binary received spikes and all L-1 G matrices on integer
        # codes are ACs (n per unit of input magnitude).  The readout MAC
        # (eta on the gated support) is added from the trace in
        # collect_analysis.
        counter.add(model.P, "AC", "reconstruction")
        for layer in model.G:
            counter.add(layer, "AC", "reconstruction")
    elif experiment.key == "ann_lista":
        counter.add(model.W_e, "MAC", "reconstruction")
        for layer in list(model.S_k)[1:]:
            counter.add(layer, "MAC", "reconstruction")
    elif experiment.key.startswith("scnn"):
        counter.add(model.head_1, "AC", "reconstruction")
        counter.add(model.head_2, "AC", "reconstruction")
        for block in model.res_blocks:
            counter.add(block.fc1, "AC", "reconstruction")
            counter.add(block.fc2, "AC", "reconstruction")


def recorded_checkpoints(experiment, seed):
    """Checkpoint files recorded in the AWGN result of this run, if any."""
    source = Path(point_path(experiment, seed))
    if not source.is_file():
        return recon_checkpoint_path(experiment, seed), classifier_checkpoint_path(experiment, seed)
    frame = pd.read_csv(source)
    recon = frame["Recon Checkpoint"].dropna().unique()
    head = frame["Classifier Checkpoint"].dropna().unique()
    if len(recon) != 1 or len(head) != 1:
        raise RuntimeError(f"Ambiguous checkpoint provenance: {source}")
    return str(resolve_existing_recon(recon[0])), str(head[0])


def load_for_analysis(experiment, seed, loader):
    configure(experiment, seed)
    model, sensor, channel = build_components(experiment)
    recon_file, classifier_file = recorded_checkpoints(experiment, seed)
    recon = torch.load(
        recon_file,
        map_location=Config.DEVICE,
        weights_only=False,
    )
    model.load_state_dict(recon["model"], strict=True)
    sensor.load_state_dict(recon["sensor"], strict=True)
    trainer = Trainer(model, sensor, channel, loader, loader)
    classifier = attach_downstream_classifier(trainer, experiment, seed)
    state = torch.load(
        classifier_file,
        map_location=Config.DEVICE,
        weights_only=False,
    )
    classifier.load_state_dict(state["classifier"], strict=True)
    model.eval(); sensor.eval(); channel.eval(); classifier.eval()
    channel.set_snr_db(TRAIN_SNR_DB)
    return trainer


def collect_analysis(seed, max_batches=0):
    _, loader = utils.get_dataloaders(Config.BATCH_SIZE_CLS)
    sample_frames = []
    energy_rows = []
    for key in MAIN_KEYS:
        experiment = experiment_registry()[key]
        trainer = load_for_analysis(experiment, seed, loader)
        counter = OperationCounter()
        attach_operation_hooks(
            counter, experiment, trainer.model, trainer.sensor
        )
        rows = []
        sample_offset = 0
        total_samples = 0
        set_seed(Config.EVAL_CHANNEL_SEED)
        with torch.no_grad():
            for batch_index, (x_batch, labels) in enumerate(loader):
                if max_batches > 0 and batch_index >= max_batches:
                    break
                x_batch = trainer._preprocess_batch(x_batch)
                batch_size, time_steps, _ = x_batch.shape
                total_samples += batch_size
                sensor_state = channel_state = model_states = None
                recon_sequence = []
                sample_bits = torch.zeros(batch_size, device=Config.DEVICE)
                if experiment.mode == "ann":
                    sample_bits.fill_(time_steps * Config.MEAS_DIM * 8)
                for time_index in range(time_steps):
                    if experiment.mode == "spiking":
                        active = float(
                            x_batch[:, time_index].detach().abs().sum().item()
                        )
                        counter.counts["sender"]["AC"] += (
                            active * Config.MEAS_DIM
                        )
                    transmitted, sensor_state, auxiliary = trainer.sensor(
                        x_batch[:, time_index],
                        state=sensor_state,
                        mode=experiment.mode,
                    )
                    hard = auxiliary.get("tx_hard")
                    if experiment.mode == "spiking":
                        if experiment.transport == "block_aer_awgn_hard":
                            sample_bits += trainer.channel.block_aer_bits(hard)
                        else:
                            sample_bits += Config.MEAS_DIM
                    received, channel_state = trainer.channel(
                        transmitted,
                        state=channel_state,
                        mode=experiment.mode,
                        tx_prob=auxiliary.get("tx_prob"),
                        tx_hard=hard,
                        tx_aux=auxiliary,
                    )
                    reconstructed, _, model_states, _ = trainer.model(
                        received, states=model_states
                    )
                    if isinstance(trainer.model, SpikingLISTA):
                        counter.counts["reconstruction"]["MAC"] += float(
                            trainer.model.last_trace["mask"].sum().item()
                        )
                    recon_sequence.append(reconstructed)
                reconstructed = torch.stack(recon_sequence, dim=1)
                predictions = trainer.downstream_classifier(
                    reconstructed
                ).argmax(dim=1)
                target_mean = x_batch.mean(dim=1)
                recon_mean = reconstructed.mean(dim=1)
                sample_nmse = 10.0 * torch.log10(
                    torch.sum((target_mean - recon_mean).square(), dim=1)
                    / torch.sum(target_mean.square(), dim=1).clamp_min(1e-12)
                    + 1e-10
                )
                for local in range(batch_size):
                    rows.append({
                        "Model": experiment.display_name,
                        "Sample": sample_offset + local,
                        "Class": int(labels[local].item()),
                        "Prediction": int(predictions[local].item()),
                        "NMSE (dB)": float(sample_nmse[local].item()),
                        "Bits/Sample": float(sample_bits[local].item()),
                    })
                sample_offset += batch_size
        counter.close()
        if key == "ann_lstm" and total_samples > 0:
            hidden = int(trainer.model.hidden_dim)
            counter.counts["reconstruction"]["MAC"] += (
                total_samples * Config.TIME_STEPS * 4.0 * hidden
                * (Config.MEAS_DIM + hidden)
            )
        sample_frames.append(pd.DataFrame(rows))
        sender_mac = counter.counts["sender"]["MAC"] / total_samples
        sender_ac = counter.counts["sender"]["AC"] / total_samples
        recon_mac = counter.counts["reconstruction"]["MAC"] / total_samples
        recon_ac = counter.counts["reconstruction"]["AC"] / total_samples
        energy_rows.append({
            "Model": experiment.display_name,
            "Sender MAC/Sample": sender_mac,
            "Sender AC/Sample": sender_ac,
            "Reconstruction MAC/Sample": recon_mac,
            "Reconstruction AC/Sample": recon_ac,
            "Total Energy (uJ/Sample)": (
                4.6 * (sender_mac + recon_mac)
                + 0.9 * (sender_ac + recon_ac)
            ) / 1e6,
        })
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    aer_bit_frames = []
    for key in ("slista_aer", "scnn_aer"):
        experiment = experiment_registry()[key]
        trainer = load_for_analysis(experiment, seed, loader)
        rows = []
        sample_offset = 0
        with torch.no_grad():
            for batch_index, (x_batch, labels) in enumerate(loader):
                if max_batches > 0 and batch_index >= max_batches:
                    break
                x_batch = trainer._preprocess_batch(x_batch)
                batch_size, time_steps, _ = x_batch.shape
                sensor_state = None
                sample_bits = torch.zeros(batch_size, device=Config.DEVICE)
                for time_index in range(time_steps):
                    _, sensor_state, auxiliary = trainer.sensor(
                        x_batch[:, time_index], state=sensor_state,
                        mode=experiment.mode,
                    )
                    sample_bits += trainer.channel.block_aer_bits(
                        auxiliary["tx_hard"]
                    )
                for local in range(batch_size):
                    rows.append({
                        "Model": experiment.display_name,
                        "Sample": sample_offset + local,
                        "Class": int(labels[local].item()),
                        "Bits/Sample": float(sample_bits[local].item()),
                    })
                sample_offset += batch_size
        aer_bit_frames.append(pd.DataFrame(rows))
    samples = pd.concat(sample_frames, ignore_index=True)
    classes = (
        samples.groupby(["Model", "Class"], as_index=False)
        .agg({"NMSE (dB)": "mean", "Bits/Sample": "mean"})
    )
    atomic_csv(
        samples,
        os.path.join(Config.RESULTS_DIR, f"analysis_samples_seed{seed}.csv"),
    )
    atomic_csv(
        classes,
        os.path.join(Config.RESULTS_DIR, f"analysis_classes_seed{seed}.csv"),
    )
    atomic_csv(
        pd.DataFrame(energy_rows),
        os.path.join(Config.RESULTS_DIR, f"analysis_energy_seed{seed}.csv"),
    )
    atomic_csv(
        pd.concat(aer_bit_frames, ignore_index=True),
        os.path.join(
            Config.RESULTS_DIR, f"analysis_aer_bits_seed{seed}.csv"
        ),
    )
    print(f"[Analysis] seed={seed}")


RAYLEIGH_VERSION = "shd-frozen-awgn10-rayleigh-v1"


def rayleigh_sources():
    """Use exact original CSV rows, never search for a 'similar' weight."""
    root = Path(Config.RESULTS_DIR)
    parts = []
    for key in SNR_KEYS:
        for seed in Config.DEFAULT_SEEDS:
            path = root / f"formal_point_{key}_seed{seed}.csv"
            frame = pd.read_csv(path)
            if set(frame["Eval SNR (dB)"]) != set(SNR_GRID):
                raise ValueError(f"Incomplete original AWGN scan: {path}")
            frame = frame[np.isclose(frame["Eval SNR (dB)"], 10)].copy()
            if len(frame) != 1 or not frame["Experiment Key"].eq(key).all() or not frame["Seed"].eq(seed).all():
                raise ValueError(f"Ambiguous checkpoint source: {path}")
            frame["Source CSV"] = str(path.resolve())
            parts.append(frame)
    # Rebuild the scatter from current operating points, not a stale summary.
    rate_path = root / "performance_bit_per_seed.csv"
    selected = []
    missing = []
    for key in RATE_KEYS:
        for seed in Config.DEFAULT_SEEDS:
            path = root / f"formal_point_{key}_seed{seed}.csv"
            if not path.is_file():
                missing.append(f"{key} seed{seed}: {path}")
                continue
            frame = pd.read_csv(path)
            frame = frame[np.isclose(frame["Eval SNR (dB)"], 10)].copy()
            if len(frame) != 1 or not frame["Experiment Key"].eq(key).all() or not frame["Seed"].eq(seed).all():
                raise ValueError(f"Invalid AWGN10 source row: {path}")
            if not frame['Classifier Checkpoint'].astype(str).str.contains(Config.classifier_protocol(), regex=False).all():
                raise ValueError(f"Old classifier source: {path}")
            frame["Source CSV"] = str(path.resolve())
            selected.append(frame)
    if missing:
        raise FileNotFoundError("Rate evaluation is incomplete; finish the missing jobs before Rayleigh evaluation:\n" + "\n".join(missing))
    all_rate = pd.concat(selected, ignore_index=True)
    all_rate.loc[all_rate["Display Model"].isin(["Spiking CNN", "Generic SNN", "SCNN"]), "Display Model"] = "Vanilla SNN"
    atomic_csv(all_rate, str(rate_path))
    atomic_csv(mean_std_table(all_rate, ["Display Model", "Alpha Rate", "Eval SNR (dB)"]),
               str(root / "performance_bit_mean_std.csv"))
    rate = all_rate[all_rate["Experiment Key"].isin(RATE_JOB_KEYS)].copy()
    print(f"[Updated AER scatter] {len(rate)} SNN rows from current RATE_KEYS", flush=True)
    parts.append(rate)
    sources = pd.concat(parts, ignore_index=True)
    if sources.duplicated(["Experiment Key", "Seed"]).any():
        raise ValueError("Duplicate experiment/seed source rows")
    if not np.isclose(sources["Train SNR (dB)"], 10).all():
        raise ValueError("Expected AWGN10-trained checkpoints")
    return sources


def resolved_weights(row):
    if Config.classifier_protocol() not in str(row['Classifier Checkpoint']):
        raise RuntimeError(f"Old classifier result for {row['Experiment Key']} seed{row['Seed']} "
                           f"(expected protocol {Config.classifier_protocol()}): "
                           "rerun formal classification before Rayleigh evaluation.")
    resolved = []
    for field in ("Recon Checkpoint", "Classifier Checkpoint"):
        path = Path(str(row[field]))
        if not path.is_file():
            path = Path(Config.RESULTS_DIR) / path.name
        if not path.is_file() and field == "Recon Checkpoint":
            # Trainer fixes the filename before training. Historical formal
            # CSVs regenerate it using the trained (learned) threshold.
            # Match ONLY that final numeric suffix; never relax seed, rate,
            # model, transport or any other experiment tag.
            pattern = r"_th[-+]?(?:\d+(?:\.\d*)?|\.\d+)\.pth$"
            prefix = re.sub(pattern, "", path.name)
            if prefix != path.name:
                candidates = sorted(
                    candidate for candidate in path.parent.glob("*.pth")
                    if candidate.is_file()
                    and re.sub(pattern, "", candidate.name) == prefix
                )
                if len(candidates) > 1:
                    raise RuntimeError(
                        f"Ambiguous threshold filenames for {row['Experiment Key']} seed{row['Seed']}:\n"
                        + "\n".join(str(candidate) for candidate in candidates)
                    )
                if len(candidates) == 1:
                    replacement = candidates[0]
                    print(f"[Checkpoint filename] {path.name} -> {replacement.name}", flush=True)
                    path = replacement
        if not path.is_file():
            raise FileNotFoundError(f"{row['Experiment Key']} seed{row['Seed']}: {field}: {path}")
        resolved.append(path.resolve())
    return resolved


def rayleigh_path(key, seed):
    return Path(Config.RESULTS_DIR) / "rayleigh_frozen" / f"rayleigh_{key}_seed{int(seed)}.csv"


def rayleigh_grid(key):
    """Frozen-Rayleigh SNR points of a key: the full grid for the robustness
    keys and for the AER rate points, whose
    performance-bit trade-off is also reported at low SNR."""
    return SNR_GRID if key in SNR_KEYS or key in RATE_JOB_KEYS else [20.0]


def valid_rayleigh(frame, key, seed):
    grid = rayleigh_grid(key)
    required = {"Experiment Key", "Seed", "Evaluation Channel", "Eval SNR (dB)",
                "Code Version", "Mean NMSE (dB)", "Accuracy (%)", "Bits/Sample"}
    return (required.issubset(frame.columns) and len(frame) == len(grid)
            and set(frame["Eval SNR (dB)"]) == set(grid)
            and frame["Experiment Key"].eq(key).all() and frame["Seed"].eq(seed).all()
            and frame["Evaluation Channel"].eq("rayleigh").all()
            and frame["Code Version"].eq(RAYLEIGH_VERSION).all()
            and np.isfinite(frame[["Mean NMSE (dB)", "Accuracy (%)", "Bits/Sample"]].to_numpy(float)).all()
            and frame["Accuracy (%)"].between(0, 100).all())


def evaluate_rayleigh(args):
    from wireless_channel import FrozenRayleighChannel
    sources = rayleigh_sources()
    selected = sources[sources["Experiment Key"].eq(args.experiment) & sources["Seed"].eq(args.seed)]
    if len(selected) != 1:
        raise ValueError("Requested experiment/seed is absent from original AWGN results")
    row = selected.iloc[0]
    recon, classifier_path = resolved_weights(row)
    experiment = experiment_registry()[args.experiment]
    if not np.isclose(float(row["Alpha Rate"]), experiment.alpha_rate):
        raise ValueError("Alpha Rate differs from the recorded training configuration")
    configure(experiment, args.seed)
    set_seed(args.seed)
    _, loader = utils.get_dataloaders(Config.BATCH_SIZE_RECON)
    model, sensor, _ = build_components(experiment)
    checkpoint = torch.load(recon, map_location=Config.DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    sensor.load_state_dict(checkpoint["sensor"], strict=True)
    channel = FrozenRayleighChannel(20.0, experiment.transport).to(Config.DEVICE)
    trainer = Trainer(model, sensor, channel, loader, loader)
    classifier = attach_downstream_classifier(trainer, experiment, args.seed)
    classifier.load_state_dict(torch.load(classifier_path, map_location=Config.DEVICE,
                                          weights_only=False)["classifier"], strict=True)
    model.eval(); sensor.eval(); classifier.eval(); channel.eval()
    bits = trainer.evaluate_bits(mode=experiment.mode)
    rows = []
    for snr in rayleigh_grid(args.experiment):
        channel.set_snr_db(snr)
        channel.reset_rng(Config.EVAL_CHANNEL_SEED)
        joint = trainer.validate_joint_metrics(mode=experiment.mode)
        # validate_cls does not call channel.eval(); reset explicitly per pass.
        channel.eval()
        channel.reset_rng(Config.EVAL_CHANNEL_SEED)
        accuracy = trainer.validate_cls(mode=experiment.mode, target="recon")
        result = row.to_dict()
        result.update({"Training Channel": "AWGN", "Evaluation Channel": "rayleigh",
                       "Eval SNR (dB)": float(snr), "Mean NMSE (dB)": joint["Temporal-Mean NMSE (dB)"],
                       "Full-Sequence NMSE (dB)": joint["Full-Sequence NMSE (dB)"],
                       "Framewise Mean NMSE (dB)": joint["Framewise Mean NMSE (dB)"],
                       "Accuracy (%)": float(accuracy), "Bits/Sample": float(bits["bits_per_sample"]),
                       "Measurements/Sample": float(bits["measurements"]),
                       "Recon Checkpoint": str(recon), "Classifier Checkpoint": str(classifier_path),
                       "Code Version": RAYLEIGH_VERSION, "Noise Sigma": np.nan,
                       "Noise-repeat NMSE Std": np.nan, "Noise-repeat Accuracy Std": np.nan,
                       "Theoretical BER": 0.5 * (1.0 - math.sqrt(10**(snr/10) / (1+10**(snr/10)))),
                       "Fading coherence": "one sample / full sequence", "CSI": "perfect CSIR; no CSIT"})
        rows.append(result)
        print(f"[Rayleigh] {args.experiment} seed{args.seed} {snr:g} dB: ACC={accuracy:.2f}%, NMSE={result['Mean NMSE (dB)']:.3f}", flush=True)
    frame = pd.DataFrame(rows)
    if not valid_rayleigh(frame, args.experiment, args.seed):
        raise ValueError("Invalid Rayleigh evaluation output")
    atomic_csv(frame, str(rayleigh_path(args.experiment, args.seed)))


def summarize_rayleigh():
    frames = []
    for row in rayleigh_sources().to_dict("records"):
        key, seed = row["Experiment Key"], int(row["Seed"])
        frame = pd.read_csv(rayleigh_path(key, seed))
        if not valid_rayleigh(frame, key, seed):
            raise ValueError(f"Invalid/incomplete Rayleigh result: {key} seed{seed}")
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    atomic_csv(raw[raw["Experiment Key"].isin(SNR_KEYS)],
               os.path.join(Config.RESULTS_DIR, "snr_robustness_rayleigh_per_seed.csv"))
    rate = raw[raw["Experiment Key"].isin(RATE_JOB_KEYS)]
    atomic_csv(rate[np.isclose(rate["Eval SNR (dB)"], 20)],
               os.path.join(Config.RESULTS_DIR, "performance_bit_rayleigh20_per_seed.csv"))
    # All SNR points of the AER rate keys (performance-bit trade-off at low SNR).
    atomic_csv(rate, os.path.join(Config.RESULTS_DIR, "performance_bit_rayleigh_per_seed.csv"))
    main = raw[raw["Experiment Key"].isin(SNR_KEYS) & np.isclose(raw["Eval SNR (dB)"], 20)]
    atomic_csv(main, os.path.join(Config.RESULTS_DIR, "main_rayleigh20_per_seed.csv"))
    print(f"[Rayleigh summary] {Config.RESULTS_DIR}")


def dispatch_rayleigh(args):
    from wireless_channel import check_frozen_rayleigh_channel
    check_frozen_rayleigh_channel()
    import queue
    import subprocess
    import sys
    from concurrent.futures import ThreadPoolExecutor
    sources = rayleigh_sources()
    jobs = queue.Queue()
    for row in sources.to_dict("records"):
        weights = resolved_weights(row)
        key, seed = row["Experiment Key"], int(row["Seed"])
        target = rayleigh_path(key, seed)
        if target.exists() and not args.redo_rayleigh:
            cached = pd.read_csv(target)
            if (valid_rayleigh(cached, key, seed)
                    and all(cached[field].eq(str(path)).all() for field, path in
                            zip(("Recon Checkpoint", "Classifier Checkpoint"), weights))
                    and all(target.stat().st_mtime >= path.stat().st_mtime for path in weights)):
                print(f"SKIP complete: {key} seed{seed}", flush=True)
                continue
        jobs.put((key, seed))
    log_dir = Path(Config.RESULTS_DIR) / "rayleigh_frozen" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    def worker(gpu):
        failed = []
        while True:
            try:
                key, seed = jobs.get_nowait()
            except queue.Empty:
                return failed
            log = log_dir / f"{key}_seed{seed}.log"
            print(f"START GPU {gpu}: {key} seed{seed}", flush=True)
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "--rayleigh-eval-only",
                       "--experiment", key, "--seed", str(seed), "--gpu", str(gpu)]
            with log.open("w") as output:
                status = subprocess.run(command, cwd=Path(__file__).resolve().parent,
                                        stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT).returncode
            print(f"{'FAIL' if status else 'DONE'} GPU {gpu}: {key} seed{seed}; {log}", flush=True)
            if status:
                failed.append(str(log))
    gpus = list(dict.fromkeys(args.gpus))
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        failures = [p for result in pool.map(worker, gpus) for p in result]
    if failures:
        raise RuntimeError("Failed evaluation jobs:\n" + "\n".join(failures))
    summarize_rayleigh()


PAPER_DATASET = 'shd'

# Integrated frozen-weight paper evaluation. Outputs are isolated from AWGN.
def paper_source(key, seed):
    source = Path(Config.RESULTS_DIR) / f'formal_point_{key}_seed{seed}.csv'
    if source.is_file():
        frame = pd.read_csv(source)
    else:
        source = Path(Config.RESULTS_DIR) / 'performance_bit_per_seed.csv'
        frame = pd.read_csv(source)
        frame = frame[frame['Experiment Key'].eq(key) & frame['Seed'].eq(seed)]
    frame = frame[np.isclose(frame['Eval SNR (dB)'], 10)]
    if len(frame) != 1:
        raise ValueError(f'Need exactly one AWGN10 source for {key} seed{seed}: {source}')
    row = frame.iloc[0].to_dict()
    ex = experiment_registry()[key]
    if row['Experiment Key'] != key or int(row['Seed']) != seed or not np.isclose(row['Train SNR (dB)'],10):
        raise ValueError(f'Wrong source identity: {source}')
    if 'Semantic Weight' in row and not np.isclose(row['Semantic Weight'],ex.semantic_weight):
        raise ValueError(f'Wrong semantic weight: {source}')
    if 'Transport' in row and row['Transport'] != ex.transport:
        raise ValueError(f'Wrong source transport: {source}')
    if not np.isclose(row['Alpha Rate'],ex.alpha_rate):
        raise ValueError(f'Wrong source rate: {source}')
    if PAPER_DATASET == 'shd':
        recon, head = resolved_weights(row)
    else:
        _, recon, head, _ = resolve_rayleigh_pair(None,key,seed)
    return row, recon, head


def paper_trainer(key, seed, loader):
    ex = experiment_registry()[key]
    row,recon,head = paper_source(key,seed)
    configure(ex,seed);set_seed(seed)
    model,sensor,_ = build_components(ex)
    state = torch.load(recon,map_location=Config.DEVICE,weights_only=False)
    model.load_state_dict(state['model'],strict=True)
    sensor.load_state_dict(state['sensor'],strict=True)
    if PAPER_DATASET == 'shd':
        from wireless_channel import FrozenRayleighChannel
        channel = FrozenRayleighChannel(20.,ex.transport).to(Config.DEVICE)
    else:
        channel = FadingBitChannel(20.,10.,10.).to(Config.DEVICE)
    trainer = Trainer(model,sensor,channel,loader,loader)
    classifier = attach_downstream_classifier(trainer,ex,seed)
    classifier.load_state_dict(torch.load(head,map_location=Config.DEVICE,weights_only=False)['classifier'],strict=True)
    for module in (model,sensor,channel,classifier):
        module.eval();module.requires_grad_(False)
    row.update({'Recon Checkpoint':str(recon),'Classifier Checkpoint':str(head)})
    return trainer,ex,row


def paper_reset(trainer,snr):
    trainer.channel.set_snr_db(float(snr));trainer.channel.eval()
    trainer.channel.reset_rng(int(Config.EVAL_CHANNEL_SEED))


def paper_samples(trainer,ex,seed,target):
    rows=[];offset=0;paper_reset(trainer,20)
    with torch.no_grad():
        for x,labels in trainer.val_loader:
            x=trainer._preprocess_batch(x) if PAPER_DATASET=='shd' else x.to(Config.DEVICE)
            sensor_state=channel_state=model_state=None
            reconstructed=[];bits=torch.zeros(x.shape[0],device=Config.DEVICE)
            for t in range(x.shape[1]):
                sent,sensor_state,aux=trainer.sensor(x[:,t],state=sensor_state,mode=ex.mode)
                hard=aux.get('tx_hard')
                if ex.mode=='ann':
                    bits += (Config.MEAS_DIM if PAPER_DATASET=='shd' else Config.LATENT_DIM)*8
                elif 'aer' in ex.transport:
                    bits += trainer.channel.block_aer_bits(hard)
                else:
                    bits += Config.MEAS_DIM if PAPER_DATASET=='shd' else Config.LATENT_DIM
                kwargs=dict(state=channel_state,mode=ex.mode,tx_prob=aux.get('tx_prob'),tx_hard=hard)
                if PAPER_DATASET=='shd':kwargs['tx_aux']=aux
                received,channel_state=trainer.channel(sent,**kwargs)
                rec,_,model_state,_=trainer.model(received,states=model_state)
                reconstructed.append(rec)
            reconstructed=torch.stack(reconstructed,dim=1)
            prediction=trainer.downstream_classifier(reconstructed).argmax(dim=1)
            truth_mean=x.mean(dim=1);rec_mean=reconstructed.mean(dim=1)
            dims=tuple(range(1,truth_mean.ndim))
            nmse=10*torch.log10((rec_mean-truth_mean).square().sum(dim=dims)/(truth_mean.square().sum(dim=dims)+1e-10)+1e-10)
            for i in range(x.shape[0]):
                rows.append({'Model':ex.display_name,'Sample':offset+i,'Class':int(labels[i]),
                    'Prediction':int(prediction[i]),'NMSE (dB)':float(nmse[i]),'Bits/Sample':float(bits[i]),
                    'Experiment Key':ex.key,'Seed':seed,'Evaluation Channel':'rayleigh','Eval SNR (dB)':20.,
                    'Transport':ex.transport})
            offset+=x.shape[0]
    atomic_csv(pd.DataFrame(rows),str(target))


def run_paper_rayleigh(args):
    folder=Path(Config.RESULTS_DIR)/'rayleigh_paper';folder.mkdir(parents=True, exist_ok=True)
    trainer,ex,source=paper_trainer(args.experiment,args.seed,utils.get_dataloaders(Config.BATCH_SIZE_RECON)[1])
    rows=[]
    with torch.no_grad():
        paper_reset(trainer,20)
        bits=trainer.evaluate_bits(mode=ex.mode)
        for snr in (SNR_GRID if ex.key in SEMANTIC_SNR_KEYS else [20.]):
            paper_reset(trainer,snr);joint=trainer.validate_joint_metrics(mode=ex.mode)
            paper_reset(trainer,snr)
            acc=(trainer.validate_cls(mode=ex.mode,target='recon') if PAPER_DATASET=='shd' else
                 trainer.validate_downstream_classifier(mode=ex.mode)['Accuracy (%)'])
            row=dict(source)
            row.update({'Evaluation Channel':'rayleigh','Training Channel':'AWGN','Eval SNR (dB)':float(snr),
                'Mean NMSE (dB)':float(joint['Temporal-Mean NMSE (dB)']),'Accuracy (%)':float(acc),
                'Bits/Sample':float(bits['bits_per_sample'] if PAPER_DATASET=='shd' else bits[0]),
                'Code Version':'rayleigh-paper-v1'})
            if not np.isfinite([row[k] for k in ['Mean NMSE (dB)','Accuracy (%)','Bits/Sample']]).all() or not 0 <= row['Accuracy (%)'] <= 100:
                raise ValueError('Invalid paper evaluation metrics')
            row['Display Model'] = ex.display_name
            rows.append(row)
    atomic_csv(pd.DataFrame(rows),str(folder/f'paper_{ex.key}_seed{args.seed}.csv'))
    if args.seed==args.paper_analysis_seed:
        if ex.key in SNR_KEYS or ex.key in paper_aer_keys():
            paper_samples(trainer,ex,args.seed,folder/f'samples_{ex.key}_seed{args.seed}.csv')
        if ex.key in SEMANTIC_SNR_KEYS:
            paper_reset(trainer,20)
            z,labels=trainer.collect_z_sequence(mode=ex.mode,max_samples=Config.TSNE_COMPARISON_MAX_PER_REGIME)
            np.savez_compressed(folder/f'z_features_{ex.key}_seed{args.seed}.npz',
                features=z.reshape(z.shape[0],-1),labels=labels,seed=np.asarray([args.seed]),
                semantic_weight=np.asarray([ex.semantic_weight]),evaluation_channel='rayleigh',eval_snr_db=20.)
    print(f'[Paper Rayleigh] {ex.key} seed{args.seed}',flush=True)


def paper_aer_keys():
    if PAPER_DATASET=='shd':return ['slista_aer','scnn_aer']
    selected=load_report_operating_points()
    return [selected['slista_key'],selected['scnn_key']]


def dispatch_paper_rayleigh(args):
    import subprocess,sys,queue
    from concurrent.futures import ThreadPoolExecutor
    folder=Path(Config.RESULTS_DIR)/'rayleigh_paper';folder.mkdir(parents=True, exist_ok=True)
    jobs=list(dict.fromkeys([(key,seed) for key in CORE_KEYS+SEMANTIC_SNR_KEYS for seed in Config.DEFAULT_SEEDS]
              +[(key,args.paper_analysis_seed) for key in SNR_KEYS+paper_aer_keys()]))
    for key,seed in jobs:paper_source(key,seed)
    tasks=queue.Queue()
    for job in jobs:tasks.put(job)
    def work(gpu):
        failed=[]
        while True:
            try:key,seed=tasks.get_nowait()
            except queue.Empty:return failed
            log=folder/f'{key}_seed{seed}.log'
            cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--paper-rayleigh-one',
                 '--experiment',key,'--seed',str(seed),'--gpu',str(gpu),
                 '--paper-source-results',str(Path(Config.RESULTS_DIR).resolve()),
                 '--paper-analysis-seed',str(args.paper_analysis_seed)]
            print(f'START GPU {gpu}: {key} seed{seed}',flush=True)
            with log.open('w') as handle:
                code=subprocess.run(cmd,stdout=handle,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL).returncode
            print(f'{"FAIL" if code else "DONE"} GPU {gpu}: {key} seed{seed}',flush=True)
            if code:failed.append(str(log))
    with ThreadPoolExecutor(max_workers=len(set(args.gpus))) as pool:
        failures=[x for batch in pool.map(work,dict.fromkeys(args.gpus)) for x in batch]
    if failures:raise RuntimeError('Failed paper evaluation:\n'+'\n'.join(failures))
    variants=dict(CORE_VARIANTS)
    core=[];semantic=[]
    for key,seed in jobs:
        f=pd.read_csv(folder/f'paper_{key}_seed{seed}.csv')
        if key in CORE_KEYS:
            f['Variant']=variants[key];core.append(f)
        if key in SEMANTIC_SNR_KEYS:
            f['Variant']='Without semantic supervision' if 'no_semantic' in key else 'With semantic supervision'
            semantic.append(f)
    atomic_csv(pd.concat(core,ignore_index=True),str(folder/'core_ablation_per_seed.csv'))
    atomic_csv(pd.concat(semantic,ignore_index=True),str(folder/'semantic_snr_robustness_per_seed.csv'))
    samples=pd.concat([pd.read_csv(folder/f'samples_{k}_seed{args.paper_analysis_seed}.csv') for k in SNR_KEYS],ignore_index=True)
    atomic_csv(samples,str(folder/f'analysis_samples_seed{args.paper_analysis_seed}.csv'))
    atomic_csv(samples.groupby(['Model','Class'],as_index=False)[['NMSE (dB)','Bits/Sample']].mean(),str(folder/f'analysis_classes_seed{args.paper_analysis_seed}.csv'))
    aer=pd.concat([pd.read_csv(folder/f'samples_{k}_seed{args.paper_analysis_seed}.csv') for k in paper_aer_keys()],ignore_index=True)
    atomic_csv(aer[['Model','Sample','Class','Bits/Sample']],str(folder/f'analysis_aer_bits_seed{args.paper_analysis_seed}.csv'))
    # Reuse each project's established feature processing and t-SNE plotting.
    import plot_results
    plot_results.plot_tsne(str(folder),args.paper_analysis_seed)
    print(f'[Done] Rayleigh paper data: {folder}',flush=True)



GROUPS = {
    "main": MAIN_KEYS,
    "rate": RATE_JOB_KEYS,
    "snr": SNR_KEYS,
    "ablation": CORE_KEYS,
    "semantic_snr": SEMANTIC_SNR_KEYS,
    "slista": SLISTA_KEYS,
    "baselines": BASELINE_KEYS,
}

def selected_experiments(run):
    groups = list(GROUPS) if run == "all" else [run]
    ordered = []
    for group in groups:
        for key in GROUPS[group]:
            if key not in ordered:
                ordered.append(key)
    return ordered

def worker_command(root, experiment, seed, args):
    command = [
        sys.executable,
        str(root / "main.py"),
        "--gpu", "0",
        "--experiment", experiment,
        "--seed", str(seed),
        "--noise-repeats", str(args.noise_repeats),
    ]
    if args.epochs is not None:
        command += ["--epochs", str(args.epochs)]
    if args.eval_only:
        command.append("--eval-only")
    if args.force_retrain:
        command.append("--force-retrain")
    if args.force_retrain_classifier:
        command.append("--force-retrain-classifier")
    if args.classifier_only:
        command.append("--classifier-only")
    command += slista_override_args(args)
    return command

def run_followup(root, gpu, arguments, log_name):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    log_path = Path(Config.RESULTS_DIR) / "logs" / log_name
    with log_path.open("w") as log:
        result = subprocess.run(
            [sys.executable, *arguments],
            cwd=root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    if result.returncode != 0:
        raise RuntimeError(f"Follow-up failed; see {log_path}")

def dispatch_training(args):
    root = Path(__file__).resolve().parent
    (Path(Config.RESULTS_DIR) / "logs").mkdir(parents=True, exist_ok=True)
    slots = list(args.gpus) * int(args.processes_per_gpu)
    queue = [
        (experiment, seed)
        for experiment in selected_experiments(args.run)
        for seed in args.seeds
    ]
    if args.dry_run:
        print(f"{len(queue)} training jobs on GPUs {args.gpus}")
        for key, seed in queue: print(f"  {key} seed{seed}")
        return
    running = []
    failures = []

    while queue or running:
        occupied = {item["slot"] for item in running}
        for slot, gpu in enumerate(slots):
            if not queue or slot in occupied:
                continue
            experiment, seed = queue.pop(0)
            log_path = Path(Config.RESULTS_DIR) / "logs" / f"{experiment}_seed{seed}.log"
            log = log_path.open("w")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            process = subprocess.Popen(
                worker_command(root, experiment, seed, args),
                cwd=root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            running.append({
                "process": process,
                "slot": slot,
                "gpu": gpu,
                "experiment": experiment,
                "seed": seed,
                "log": log,
                "log_path": log_path,
            })
            print(
                f"[Start] GPU {gpu} slot {slot}: "
                f"{experiment}, seed={seed}",
                flush=True,
            )

        time.sleep(5)
        survivors = []
        for item in running:
            returncode = item["process"].poll()
            if returncode is None:
                survivors.append(item)
                continue
            item["log"].close()
            if returncode == 0:
                print(
                    f"[Done] GPU {item['gpu']}: {item['experiment']}, "
                    f"seed={item['seed']}",
                    flush=True,
                )
            else:
                failures.append(item)
                print(
                    f"[Failed] {item['experiment']}, seed={item['seed']} "
                    f"-> {item['log_path']}",
                    flush=True,
                )
        running = survivors

    if failures:
        raise RuntimeError(f"{len(failures)} training jobs failed")

    first_gpu = args.gpus[0]
    run_followup(
        root,
        first_gpu,
        [str(root / "main.py"), "--gpu", "0", "--summarize-only"],
        "summarize.log",
    )
    if args.run in ("all", "main", "slista") and not args.skip_analysis:
        run_followup(
            root,
            first_gpu,
            [
                str(root / "main.py"), "--gpu", "0",
                "--analysis-only", "--seed", str(args.seeds[0]),
            ],
            "analysis.log",
        )
    if not args.skip_plot:
        run_followup(
            root,
            first_gpu,
            [str(root / "plot_results.py")],
            "plot_results.log",
        )
    print(f"[Complete] results={Config.RESULTS_DIR}", flush=True)

SLISTA_FLAGS = {
    "slista-kappa": "SLISTA_KAPPA",
    "slista-beta": "SLISTA_BETA",
    "slista-alpha": "SLISTA_ALPHA",
    "slista-rho": "SLISTA_RHO",
    "slista-lambda": "SLISTA_LAMBDA_S",
    "slista-lr-scale": "SLISTA_LR_SCALE",
    "slista-theta-lr-ratio": "SLISTA_THETA_LR_RATIO",
    "slista-g-lr-ratio": "SLISTA_G_LR_RATIO",
    "slista-tx-threshold": "SLISTA_TX_THRESHOLD",
    "slista-semantic-weight": "SLISTA_SEMANTIC_WEIGHT",
    "slista-weight-decay-p": "SLISTA_WEIGHT_DECAY_P",
    "slista-weight-decay-g": "SLISTA_WEIGHT_DECAY_G",
}


def slista_override_args(args):
    """CLI flags for the S-LISTA overrides given on this command line."""
    flags = []
    for flag, name in SLISTA_FLAGS.items():
        value = getattr(args, name, None)
        if value is not None:
            flags += [f"--{flag}", repr(float(value))]
    return flags


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0])
    rayleigh = parser.add_mutually_exclusive_group()
    rayleigh.add_argument("--rayleigh-all", action="store_true", help="Frozen scans + AER scatter, one job per GPU, then summarize")
    rayleigh.add_argument("--rayleigh-eval-only", action="store_true")
    rayleigh.add_argument("--rayleigh-summarize-only", action="store_true")
    parser.add_argument("--redo-rayleigh", action="store_true", help="Re-evaluate completed jobs; never retrain")
    parser.add_argument("--experiment", choices=sorted(experiment_registry()))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--noise-repeats", type=int, default=1)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--force-retrain-classifier", action="store_true")
    parser.add_argument("--classifier-only", action="store_true",
                        help="fail instead of retraining any reconstruction model")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--analysis-only", action="store_true")
    parser.add_argument("--max-batches-analysis", type=int, default=0)
    parser.add_argument("--recon-lr", type=float, default=None)
    parser.add_argument("--classifier-lr", type=float, default=None)
    parser.add_argument("--eval-10db-only", action="store_true")
    parser.add_argument('--paper-rayleigh-all', action='store_true')
    parser.add_argument('--paper-rayleigh-one', action='store_true')
    parser.add_argument('--paper-source-results', type=Path)
    parser.add_argument('--paper-analysis-seed', type=int, default=42)
    parser.add_argument("--downstream", action="store_true",
                        help="with --recon-only: also train/evaluate the downstream classifier")
    parser.add_argument("--recon-only", action="store_true",
                        help="Train/evaluate the reconstruction model only")
    # S-LISTA hyperparameter overrides (otherwise config.py and
    # configs/slista.json are used).
    for flag, name in SLISTA_FLAGS.items():
        parser.add_argument(f"--{flag}", type=float, default=None, dest=name)
    parser.add_argument("--run", choices=["all"] + list(GROUPS))
    parser.add_argument("--processes-per-gpu", type=int, default=2)
    parser.add_argument("--seeds", nargs="+", type=int, default=Config.DEFAULT_SEEDS)
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument("--skip-plot", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--results-dir", default=Config.RESULTS_DIR)
    parser.add_argument("--data-dir", default=Config.PROCESSED_DIR)
    parser.add_argument("--num-workers", type=int, default=Config.NUM_WORKERS)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.classifier_lr is not None and not np.isclose(args.classifier_lr, 1e-3):
        raise ValueError('The selected downstream protocol fixes classifier LR at 1e-3.')
    Config.RESULTS_DIR = str(Path(args.results_dir).resolve())
    Config.PROCESSED_DIR = str(Path(args.data_dir).resolve())
    Config.NUM_WORKERS = args.num_workers
    for name in SLISTA_FLAGS.values():
        value = getattr(args, name, None)
        if value is not None:
            setattr(Config, name, float(value))
    os.environ["SHD_OUTPUT_DIR"] = Config.RESULTS_DIR
    os.environ["SHD_DATA_DIR"] = Config.PROCESSED_DIR
    os.environ["SHD_NUM_WORKERS"] = str(args.num_workers)
    Config.make_dir()
    if args.run:
        if args.processes_per_gpu < 1 or any(g < 0 for g in args.gpus):
            raise ValueError("Use non-negative GPUs and a positive processes-per-gpu")
        dispatch_training(args)
        return
    Config.DEVICE = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )
    print(f"[Version] {VERSION}")
    Config._CLI_RECON_LR = args.recon_lr
    Config._CLI_CLASSIFIER_LR = args.classifier_lr or 1e-3
    Config._CLI_EVAL_10_ONLY = args.eval_10db_only
    if args.paper_rayleigh_all or args.paper_rayleigh_one:
        if args.paper_source_results:
            Config.RESULTS_DIR = str(args.paper_source_results.resolve())
        if args.paper_rayleigh_all:
            dispatch_paper_rayleigh(args)
        else:
            run_paper_rayleigh(args)
        return
    if args.rayleigh_all:
        dispatch_rayleigh(args)
    elif args.rayleigh_eval_only:
        evaluate_rayleigh(args)
    elif args.rayleigh_summarize_only:
        summarize_rayleigh()
    elif args.summarize_only:
        summarize()
    elif args.analysis_only:
        collect_analysis(args.seed, args.max_batches_analysis)
    elif args.experiment and args.recon_only:
        run_recon_only(args)
    elif args.experiment:
        run_one(args)
    else:
        raise ValueError("Specify --experiment, --summarize-only or --analysis-only")


if __name__ == "__main__":
    main()
