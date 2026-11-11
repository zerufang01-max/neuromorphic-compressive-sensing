"""Training of the DVS-Gesture baselines (Vanilla SNN, ANN LISTA, ANN ConvLSTM)
under bit-input AWGN.

Each invocation owns one (experiment, seed) pair:

    python train.py --experiment ann_lista_q8 --seed 42 --gpu 0

It trains the sensing and reconstruction models and the downstream classifier,
evaluates them and writes formal_point_<experiment>_seed<seed>.csv, which
records the checkpoint files.  ../main.py reads these files (experiment group
"baselines").
"""

import argparse
import glob
import json
import math
import os
import random
import re
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from config import Config
from edge_compression import EdgeSensor
from snn_models import (
    ConvolutionalLISTA,
    ConvLSTMRecon,
    SpikingCNNRecon,
    VideoClassifier,
)
from trainer import Trainer
from wireless_channel import WirelessChannel
import utils


VERSION = "DVSG_AWGN_FORMAL_5SEED"

TRAIN_SNR_DB = 10.0
ANN_ALPHA_RATE = float(Config.ALPHA_RATE_LISTA)
FORMAL_LSTM_LR = 5e-4
SNR_GRID = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 7.5, 10, 15, 20]
SNR_KEYS = ["scnn_dense", "ann_lista_q8", "ann_lstm_q8"]
SCNN_RATE_VALUES = [1e-3, 1.4e-3, 2e-3, 3e-3, 4e-3]
SCNN_SELECTED_RATE = 1.4e-3


def rate_key(prefix, value):
    return f"{prefix}_rate_{float(value):g}".replace(".", "p").replace("-", "m")


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


def experiment_registry():
    semantic = float(Config.ALPHA_SEMANTIC)
    registry = {
        "scnn_aer": Experiment(
            "scnn_aer", "Vanilla SNN", SpikingCNNRecon, "spiking",
            "block_aer_awgn_hard", None,
            Config.ENCODER_THRESHOLD, Config.DECODER_TAU,
            SCNN_SELECTED_RATE, semantic,
        ),
        "ann_lista_q8": Experiment(
            "ann_lista_q8", "ANN LISTA", ConvolutionalLISTA, "ann",
            "dense_q8_bitawgn_hard", 8, None, None, ANN_ALPHA_RATE, semantic,
        ),
        "ann_lstm_q8": Experiment(
            "ann_lstm_q8", "ANN ConvLSTM", ConvLSTMRecon, "ann",
            "dense_q8_bitawgn_hard", 8, None, None, ANN_ALPHA_RATE, semantic,
        ),
        "scnn_dense": Experiment(
            "scnn_dense", "Vanilla SNN", SpikingCNNRecon, "spiking",
            "dense_binary_awgn_hard", None,
            Config.ENCODER_THRESHOLD, Config.DECODER_TAU,
            0.0, semantic,
        ),
    }
    for value in SCNN_RATE_VALUES:
        key = rate_key("scnn", value)
        registry[key] = Experiment(
            key, "Vanilla SNN", SpikingCNNRecon, "spiking",
            "block_aer_awgn_hard", None,
            Config.ENCODER_THRESHOLD, Config.DECODER_TAU,
            float(value), semantic,
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
    Config.TRAIN_SNR_DB = TRAIN_SNR_DB
    Config.EVAL_SNR_DB_LIST = list(SNR_GRID)
    Config.ALPHA_SEMANTIC = float(experiment.semantic_weight)
    Config.TX_ABLATION_MODE = (
        "hard_spike" if experiment.tx_gradient == "hard_surrogate" else "none"
    )
    Config.DECODER_SURROGATE_MODE = (
        "hard" if experiment.tx_gradient == "hard_surrogate" else "sigmoid"
    )
    classifier_lr = getattr(Config, "_CLI_CLASSIFIER_LR", None)
    if classifier_lr is None:
        classifier_key = experiment.key
        if experiment.key.startswith("scnn"):
            classifier_key = "scnn_dense"
        classifier_lr = Config.SELECTED_CLASSIFIER_LR.get(
            classifier_key, 5e-4
        )
    Config.DOWNSTREAM_LR_OVERRIDE = float(classifier_lr)
    Config.RECON_LR_OVERRIDE = getattr(Config, "_CLI_RECON_LR", None)
    Config.ANN_QUANT_BITS = 8
    Config.ANN_RX_MODE = "hard"
    if experiment.mode == "spiking":
        selected = Config.SELECTED_SNN_PARAMS.get("scnn", {})
        Config.ENCODER_THRESHOLD = float(
            getattr(Config, "_CLI_TX_THRESHOLD", None)
            if getattr(Config, "_CLI_TX_THRESHOLD", None) is not None
            else selected.get("tx_threshold", Config.ENCODER_THRESHOLD)
        )
        Config.DECODER_THRESHOLD = float(
            getattr(Config, "_CLI_NEURON_THRESHOLD", None)
            if getattr(Config, "_CLI_NEURON_THRESHOLD", None) is not None
            else selected.get("neuron_threshold", Config.DECODER_THRESHOLD)
        )
        Config.DECODER_TAU = float(
            getattr(Config, "_CLI_LEAK", None)
            if getattr(Config, "_CLI_LEAK", None) is not None
            else selected.get("leak", Config.DECODER_TAU)
        )
        Config.DECODER_SURROGATE_TAU = float(
            getattr(Config, "_CLI_SURROGATE_TAU", None)
            if getattr(Config, "_CLI_SURROGATE_TAU", None) is not None
            else selected.get("surrogate_tau", Config.DECODER_SURROGATE_TAU)
        )
        Config.SNN_TRANSPORT = experiment.transport
        Config.ALPHA_RATE_SCNN = float(experiment.alpha_rate)


def experiment_hp(experiment, epochs=None):
    hp = dict(Config.PHASE3_CONFIG)
    if epochs is not None:
        hp["epochs"] = int(epochs)
    if (
        experiment.key.startswith("scnn_")
        and Config.RECON_LR_OVERRIDE is not None
    ):
        hp["lr_spiking_cnn"] = float(Config.RECON_LR_OVERRIDE)
    elif experiment.key == "ann_lista_q8" and Config.RECON_LR_OVERRIDE is not None:
        hp["lr_lista_ann"] = float(Config.RECON_LR_OVERRIDE)
    elif experiment.key == "ann_lstm_q8":
        hp["lr_lstm"] = (
            float(Config.RECON_LR_OVERRIDE)
            if Config.RECON_LR_OVERRIDE is not None
            else FORMAL_LSTM_LR
        )
    return hp


def build_components(experiment):
    sensor = EdgeSensor()
    channel = WirelessChannel(snr_db=TRAIN_SNR_DB)
    model = experiment.model_class()
    if hasattr(model, "mode"):
        model.mode = "rate" if experiment.mode == "ann" else "spiking"
    return (
        model.to(Config.DEVICE),
        sensor.to(Config.DEVICE),
        channel.to(Config.DEVICE),
    )


def experiment_artifact_key(experiment):
    scan_tag = getattr(Config, "_CLI_ARTIFACT_TAG", None)
    return f"{experiment.key}_{scan_tag}" if scan_tag else experiment.key


def recon_model_name(experiment, seed):
    tag = experiment_artifact_key(experiment)
    if Config.RECON_LR_OVERRIDE is not None:
        tag += f"_rlr{float(Config.RECON_LR_OVERRIDE):g}".replace(".", "p")
    elif experiment.key == "ann_lstm_q8":
        tag += "_preawgninit_lr5em4"
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
    checkpoint_alpha = float(experiment.alpha_rate)
    name = Config.get_recon_ckpt_name(
        experiment.mode,
        recon_model_name(experiment, seed),
        threshold,
        checkpoint_alpha,
    )
    return os.path.join(Config.RESULTS_DIR, name)


def classifier_checkpoint_path(experiment, seed):
    tag = experiment_artifact_key(experiment)
    if experiment.key == "ann_lstm_q8" and Config.RECON_LR_OVERRIDE is None:
        tag += "_preawgninit_lr5em4"
    lr_tag = f"{float(Config.DOWNSTREAM_LR_OVERRIDE):g}".replace(".", "p")
    tag += f"_clr{lr_tag}"
    return os.path.join(
        Config.RESULTS_DIR,
        f"downstream_recon_x_{tag}_seed{int(seed)}_"
        f"trainsnr{TRAIN_SNR_DB:g}.pth",
    )


def point_path(experiment, seed):
    suffix = ""
    if getattr(Config, "_CLI_ARTIFACT_TAG", None):
        suffix += f"_{Config._CLI_ARTIFACT_TAG}"
    if Config.RECON_LR_OVERRIDE is not None:
        suffix += f"_rlr{float(Config.RECON_LR_OVERRIDE):g}_clr{float(Config.DOWNSTREAM_LR_OVERRIDE):g}".replace(".", "p")
    return os.path.join(
        Config.RESULTS_DIR,
        f"formal_point_{experiment.key}_seed{int(seed)}{suffix}.csv",
    )


def attach_downstream_classifier(trainer, experiment, seed):
    set_seed(int(seed) + 10000)
    classifier = VideoClassifier(
        Config.NUM_CLASSES, Config.TIME_STEPS
    ).to(Config.DEVICE)
    trainer.set_downstream_classifier(classifier)
    return classifier


def train_or_load(experiment, seed, args, train_loader, val_loader):
    configure(experiment, seed)
    set_seed(seed)
    model, sensor, channel = build_components(experiment)
    trainer = Trainer(model, sensor, channel, train_loader, val_loader)
    recon_path = recon_checkpoint_path(experiment, seed, model)
    if args.force_retrain or not os.path.exists(recon_path):
        if args.eval_only:
            raise FileNotFoundError(recon_path)
        trainer.train_dynamic_recon(
            hp=experiment_hp(experiment, args.epochs),
            model_name=recon_model_name(experiment, seed),
            mode=experiment.mode,
        )
    checkpoint = torch.load(
        recon_path, map_location=Config.DEVICE, weights_only=False
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    sensor.load_state_dict(checkpoint["sensor"], strict=True)
    channel.set_snr_db(TRAIN_SNR_DB)

    classifier = attach_downstream_classifier(trainer, experiment, seed)
    classifier_path = classifier_checkpoint_path(experiment, seed)
    if args.force_retrain_classifier or not os.path.exists(classifier_path):
        if args.eval_only:
            raise FileNotFoundError(classifier_path)
        hp = dict(Config.PHASE4_CONFIG)
        hp["lr"] = float(Config.DOWNSTREAM_LR_OVERRIDE)
        if args.epochs is not None:
            hp["epochs"] = int(args.epochs)
        trainer.train_downstream_classifier(
            hp=hp, mode=experiment.mode, save_path=classifier_path
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
    if experiment.key in SNR_KEYS:
        return SNR_GRID
    return [TRAIN_SNR_DB]


def evaluate_grid(trainer, experiment, checkpoint, seed, repeats):
    trainer.channel.set_snr_db(TRAIN_SNR_DB)
    bits_per_sample, firing_rate = trainer.evaluate_bits(
        mode=experiment.mode
    )
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
            joint["Task Accuracy (%)"] = trainer.validate_downstream_classifier(
                mode=experiment.mode
            )["Accuracy (%)"]
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
            "Bits/Sample": float(bits_per_sample),
            "Measurements/Sample": float(
                firing_rate * Config.LATENT_DIM * Config.TIME_STEPS
                if experiment.mode == "spiking"
                else Config.LATENT_DIM * Config.TIME_STEPS
            ),
            "Tx Firing Rate": float(firing_rate),
            "Noise Sigma": float(sigma),
            "Theoretical BER": float(ber),
            "Selected Epoch": int(checkpoint["epoch"]),
            "Recon Checkpoint": recon_checkpoint_path(
                experiment, seed, trainer.model
            ),
            "Classifier Checkpoint": classifier_checkpoint_path(
                experiment, seed
            ),
            "Code Version": Config.CODE_VERSION,
        }
        row.update(parameter_counts)
        rows.append(row)
    Config.EVAL_CHANNEL_SEED = original_channel_seed
    trainer.channel.set_snr_db(TRAIN_SNR_DB)
    return rows


def run_one(args):
    registry = experiment_registry()
    experiment = registry[args.experiment]
    train_loader, val_loader = utils.get_dataloaders(Config.BATCH_SIZE_RECON)
    trainer, checkpoint, _, _ = train_or_load(
        experiment, args.seed, args, train_loader, val_loader
    )
    rows = evaluate_grid(
        trainer, experiment, checkpoint, args.seed, args.noise_repeats
    )
    atomic_csv(pd.DataFrame(rows), point_path(experiment, args.seed))
    print(
        f"[Done] {experiment.key} | seed={args.seed} | "
        f"{point_path(experiment, args.seed)}",
        flush=True,
    )

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--experiment", choices=sorted(experiment_registry()), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--noise-repeats", type=int, default=1)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--force-retrain-classifier", action="store_true")
    parser.add_argument("--recon-lr", type=float, default=None)
    parser.add_argument("--classifier-lr", type=float, default=None)
    parser.add_argument("--eval-10db-only", action="store_true")
    parser.add_argument("--artifact-tag", type=str, default=None)
    parser.add_argument("--tx-threshold", type=float, default=None)
    parser.add_argument("--neuron-threshold", type=float, default=None)
    parser.add_argument("--leak", type=float, default=None)
    parser.add_argument("--surrogate-tau", type=float, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    Config.DEVICE = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )
    print(f"[Version] {VERSION}")
    Config._CLI_RECON_LR = args.recon_lr
    Config._CLI_CLASSIFIER_LR = args.classifier_lr
    Config._CLI_EVAL_10_ONLY = args.eval_10db_only
    Config._CLI_ARTIFACT_TAG = args.artifact_tag
    Config._CLI_TX_THRESHOLD = args.tx_threshold
    Config._CLI_NEURON_THRESHOLD = args.neuron_threshold
    Config._CLI_LEAK = args.leak
    Config._CLI_SURROGATE_TAU = args.surrogate_tau
    run_one(args)


if __name__ == "__main__":
    main()
