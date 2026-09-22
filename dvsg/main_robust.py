import argparse
import os
import random
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from config import Config
from edge_compression import EdgeSensor
from snn_models import (
    ConvolutionalLISTA,
    ConvolutionalLISTA_ImageSpace,
    ConvLSTMRecon,
    SpikingCNNRecon,
)
from trainer import Trainer
from wireless_channel import WirelessChannel
import utils

warnings.filterwarnings("ignore")


# -----------------------------------------------------------------------------
# Default grids
# -----------------------------------------------------------------------------

# Final paper p-sweep: evaluate around and above the training channel p=1e-2.
# We omit p=0 and very high-p collapse points; this focuses on the degradation region.
EVAL_P_GRID = [
    1.0e-2, 1.5e-2,
    2.0e-2, 2.5e-2,
    3.0e-2, 3.5e-2,
    4.0e-2, 4.5e-2,
    5.0e-2, 5.5e-2,
    6.0e-2, 6.5e-2,
    7.0e-2, 7.5e-2,
    8.0e-2, 8.5e-2,
    9.0e-2, 9.5e-2,
    1.0e-1,
]

DEFAULT_ALPHA_GRID = [
    2e-4, 4e-4, 6e-4, 8e-4,
    1e-3, 1.2e-3, 1.4e-3, 1.6e-3, 1.8e-3, 2e-3,
]

# Low-rate checkpoints used only by the channel-encoding ablation.  The
# ordinary p-sweep intentionally excludes these operating points.
DEFAULT_LOWRATE_LISTA_CKPT = (
    "/workspace0/zf925/spiking_lista_dvsg/results_dvsg_recon_final/"
    "dynamic_recon_spiking_convlista_img_relaxedaer_ratesweep_"
    "alpha0p0008_seed42_rho0.1_th0.10.pth"
)
DEFAULT_LOWRATE_SCNN_CKPT = (
    "/workspace0/zf925/spiking_lista_dvsg/results_dvsg_recon_final/"
    "dynamic_recon_spiking_spikingcnn_relaxedaer_ratesweep_"
    "alpha0p0012_seed42_rho0.1.pth"
)


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}


def alpha_tag(alpha: float) -> str:
    """Filename-safe alpha tag, e.g. 0.0002 -> 0p0002."""
    return f"{alpha:.6g}".replace(".", "p").replace("-", "m")


def threshold_suffix(model: torch.nn.Module) -> str:
    if hasattr(model, "get_threshold_value"):
        try:
            return f"_th{float(model.get_threshold_value()):.2f}"
        except Exception:
            return ""
    return ""


def rho_suffix() -> str:
    return f"_rho{Config.TX_RHO}"


def dynamic_recon_checkpoint_name(model: torch.nn.Module, model_name: str, mode: str) -> str:
    return f"dynamic_recon_{mode}_{model_name.lower()}{rho_suffix()}{threshold_suffix(model)}.pth"


def checkpoint_path(filename: str) -> str:
    return os.path.join(Config.RESULTS_DIR, filename)


def safe_load_state(model: torch.nn.Module, sensor: Optional[torch.nn.Module], path_or_name: str,
                    device: torch.device, strict_model: bool = False) -> bool:
    path = path_or_name
    if not os.path.isabs(path):
        path = checkpoint_path(path_or_name)

    if not os.path.exists(path):
        print(f"    [Missing] {path}")
        return False

    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model_state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        if isinstance(model_state, dict):
            model.load_state_dict(strip_module_prefix(model_state), strict=strict_model)

        if isinstance(ckpt, dict) and sensor is not None and "sensor" in ckpt:
            sensor.load_state_dict(strip_module_prefix(ckpt["sensor"]), strict=False)

        print(f"    [Loaded] {path}")
        return True
    except Exception as exc:
        print(f"    [Error] Failed to load {path}: {exc}")
        return False


def load_first_existing(model: torch.nn.Module, sensor: Optional[torch.nn.Module], candidates: List[str],
                        device: torch.device, strict_model: bool = False) -> Optional[str]:
    for name in candidates:
        if safe_load_state(model, sensor, name, device, strict_model=strict_model):
            return name
    return None


def load_fixed_classifier(model_class, candidates: List[str], device: torch.device
                          ) -> Tuple[Optional[torch.nn.Module], Optional[str]]:
    """Load a classifier in a separate holder so reconstruction weights stay intact."""
    holder = model_class().to(device)
    dummy_sensor = EdgeSensor().to(device)
    loaded_name = load_first_existing(
        holder,
        dummy_sensor,
        candidates,
        device,
        strict_model=False,
    )
    if loaded_name is None:
        return None, None

    classifier = getattr(holder, "classifier_image", None)
    if classifier is None:
        print(f"    [Skip] Classifier holder {model_class.__name__} has no classifier_image.")
        return None, loaded_name

    classifier.eval()
    for param in classifier.parameters():
        param.requires_grad = False
    return classifier, loaded_name


def reset_channel_cache(channel: WirelessChannel, p: float) -> None:
    channel.p = float(p)
    if hasattr(channel, "_K"):
        channel._K = None
    if hasattr(channel, "_K_p"):
        channel._K_p = -1.0


def model_display_name(name: str) -> str:
    mapping = {
        "ConvLISTA_Img": "SPIKING LISTA",
        "ANN_ConvLISTA": "ANN LISTA",
        "ConvLSTM": "ANN LAMP",
        "SpikingCNN": "SCNN",
    }
    return mapping.get(name, name)


# -----------------------------------------------------------------------------
# Model registry
# -----------------------------------------------------------------------------

def build_psweep_models() -> List[Dict]:
    th = f"{Config.INIT_THRESHOLD:.2f}"
    rho = f"{Config.TX_RHO}"
    models = [
        {
            "name": "ConvLISTA_Img",
            "class": ConvolutionalLISTA_ImageSpace,
            "mode": "spiking",
            "recon_candidates": [
                f"dynamic_recon_spiking_convlista_img_relaxedaer_seed{Config.SEED}_rho{rho}_th{th}.pth",
                f"dynamic_recon_spiking_convlista_img_relaxedaer_seed{Config.SEED}_th{th}.pth",
                f"dynamic_recon_spiking_convlista_img_relaxedaer_seed{Config.SEED}_th0.50.pth",
            ],
            "cls_candidates": [
                f"phase4_cls_spiking_convlista_img_relaxedaer_seed{Config.SEED}_rho{rho}_th{th}_image.pth",
                f"phase4_cls_spiking_convlista_img_relaxedaer_seed{Config.SEED}_th{th}_image.pth",
                f"phase4_cls_spiking_convlista_img_relaxedaer_seed{Config.SEED}_th0.50_image.pth",
                f"phase4_cls_spiking_convlista_img_relaxedaer_seed{Config.SEED}_th0.50.pth",
            ],
        },
        {
            "name": "ConvLSTM",
            "class": ConvLSTMRecon,
            "mode": "ann",
            "recon_candidates": [
                f"dynamic_recon_ann_convlstm_seed{Config.SEED}_rho{rho}.pth",
                f"dynamic_recon_ann_convlstm_seed{Config.SEED}.pth",
            ],
            "cls_candidates": [
                f"phase4_cls_ann_convlstm_seed{Config.SEED}_rho{rho}_image.pth",
                f"phase4_cls_ann_convlstm_seed{Config.SEED}_image.pth",
                f"phase4_cls_ann_convlstm_seed{Config.SEED}.pth",
            ],
        },
        {
            "name": "SpikingCNN",
            "class": SpikingCNNRecon,
            "mode": "spiking",
            "recon_candidates": [
                f"dynamic_recon_spiking_spikingcnn_relaxedaer_seed{Config.SEED}_rho{rho}.pth",
                f"dynamic_recon_spiking_spikingcnn_relaxedaer_seed{Config.SEED}.pth",
            ],
            "cls_candidates": [
                f"phase4_cls_spiking_spikingcnn_relaxedaer_seed{Config.SEED}_rho{rho}_image.pth",
                f"phase4_cls_spiking_spikingcnn_relaxedaer_seed{Config.SEED}_image.pth",
                f"phase4_cls_spiking_spikingcnn_relaxedaer_seed{Config.SEED}.pth",
            ],
        },
        {
            "name": "ANN_ConvLISTA",
            "class": ConvolutionalLISTA,
            "mode": "ann",
            "recon_candidates": [
                f"dynamic_recon_ann_ann_convlista_seed{Config.SEED}_rho{rho}_th{th}.pth",
                f"dynamic_recon_ann_ann_convlista_seed{Config.SEED}_th{th}.pth",
                f"dynamic_recon_ann_ann_convlista_seed{Config.SEED}_th0.50.pth",
            ],
            "cls_candidates": [
                f"phase4_cls_ann_ann_convlista_seed{Config.SEED}_rho{rho}_th{th}_image.pth",
                f"phase4_cls_ann_ann_convlista_seed{Config.SEED}_th{th}_image.pth",
                f"phase4_cls_ann_ann_convlista_seed{Config.SEED}_th0.50_image.pth",
                f"phase4_cls_ann_ann_convlista_seed{Config.SEED}_th0.50.pth",
            ],
        },
    ]

    return models


def build_rate_models() -> List[Dict]:
    th = f"{Config.INIT_THRESHOLD:.2f}"
    rho = f"{Config.TX_RHO}"
    return [
        {
            "name": "ConvLISTA_Img",
            "label": "SPIKING LISTA",
            "class": ConvolutionalLISTA_ImageSpace,
            "mode": "spiking",
            "stem": "convlista_img_relaxedaer",
            "alpha_attr": "ALPHA_RATE_LISTA",
            "main_cls_candidates": [
                f"phase4_cls_spiking_convlista_img_relaxedaer_seed{Config.SEED}_rho{rho}_th{th}_image.pth",
                f"phase4_cls_spiking_convlista_img_relaxedaer_seed{Config.SEED}_th{th}_image.pth",
                f"phase4_cls_spiking_convlista_img_relaxedaer_seed{Config.SEED}_th0.50_image.pth",
            ],
        },
        {
            "name": "SpikingCNN",
            "label": "SCNN",
            "class": SpikingCNNRecon,
            "mode": "spiking",
            "stem": "spikingcnn_relaxedaer",
            "alpha_attr": "ALPHA_RATE_SCNN",
            "main_cls_candidates": [
                f"phase4_cls_spiking_spikingcnn_relaxedaer_seed{Config.SEED}_rho{rho}_image.pth",
                f"phase4_cls_spiking_spikingcnn_relaxedaer_seed{Config.SEED}_image.pth",
                f"phase4_cls_spiking_spikingcnn_relaxedaer_seed{Config.SEED}.pth",
            ],
        },
    ]


def build_channel_ablation_models(args) -> List[Dict]:
    """Low-rate models used to compare AER and dense bitmap transmission."""
    th = f"{Config.INIT_THRESHOLD:.2f}"
    rho = f"{Config.TX_RHO}"

    lista_ckpt = args.lowrate_lista_ckpt
    scnn_ckpt = args.lowrate_scnn_ckpt

    def with_basename(path: str) -> List[str]:
        candidates = [path]
        if path:
            candidates.append(os.path.basename(path))
        return candidates

    models = [
        {
            "name": "ConvLISTA_Img",
            "label": "SPIKING LISTA",
            "class": ConvolutionalLISTA_ImageSpace,
            "recon_candidates": with_basename(lista_ckpt),
            "cls_candidates": [
                f"phase4_cls_spiking_convlista_img_relaxedaer_seed{Config.SEED}_rho{rho}_th{th}_image.pth",
                f"phase4_cls_spiking_convlista_img_relaxedaer_seed{Config.SEED}_th{th}_image.pth",
                f"phase4_cls_spiking_convlista_img_relaxedaer_seed{Config.SEED}_th0.50_image.pth",
            ],
        },
        {
            "name": "SpikingCNN",
            "label": "SPIKING CNN",
            "class": SpikingCNNRecon,
            "recon_candidates": with_basename(scnn_ckpt),
            "cls_candidates": [
                f"phase4_cls_spiking_spikingcnn_relaxedaer_seed{Config.SEED}_rho{rho}_image.pth",
                f"phase4_cls_spiking_spikingcnn_relaxedaer_seed{Config.SEED}_image.pth",
                f"phase4_cls_spiking_spikingcnn_relaxedaer_seed{Config.SEED}.pth",
            ],
        },
    ]
    if args.channel_model == "all":
        return models
    return [cfg for cfg in models if cfg["name"] == args.channel_model]


def rate_model_name(cfg: Dict, alpha: float) -> str:
    return f"{cfg['stem']}_ratesweep_alpha{alpha_tag(alpha)}_seed{Config.SEED}"


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------

def nmse_energy(mean_gt: torch.Tensor, mean_rec: torch.Tensor) -> Tuple[float, float]:
    """
    Return linear error energy and target energy.

    Formal global NMSE should aggregate energies over the whole
    validation set before taking the ratio and log.
    """
    err_energy = torch.sum((mean_gt - mean_rec) ** 2).item()
    target_energy = torch.sum(mean_gt ** 2).item()
    return float(err_energy), float(target_energy)


def energy_to_nmse_db(total_error_energy: float, total_target_energy: float) -> float:
    eps_den = 1e-12
    eps_log = 1e-10
    if total_target_energy <= 0:
        return float("nan")

    nmse_linear = total_error_energy / max(total_target_energy, eps_den)
    return float(10.0 * np.log10(nmse_linear + eps_log))


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return float((preds == labels).float().mean().item() * 100.0)


def evaluate_reconstruction_and_classifier(
    model: torch.nn.Module,
    sensor: EdgeSensor,
    channel: WirelessChannel,
    val_loader,
    mode: str,
    classifier: Optional[torch.nn.Module] = None,
    bsc_p: float = 0.01,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """Evaluate NMSE, measured AER bits, and optional fixed classifier accuracy."""
    Config.USE_WIRELESS = True
    reset_channel_cache(channel, bsc_p)

    model.eval()
    sensor.eval()
    if classifier is not None:
        classifier.eval()

    sensor.mode = "ann" if mode == "ann" else "snn"
    if hasattr(model, "mode"):
        model.mode = "rate" if mode == "ann" else "spiking"

    total_error_energy = 0.0
    total_target_energy = 0.0
    total_acc = 0.0
    total_batches = 0
    total_samples = 0
    total_frames = 0
    total_bits = 0.0
    total_events = 0.0
    total_elements = 0.0

    set_seed(Config.SEED)

    with torch.no_grad():
        for batch_idx, (x_batch, labels) in enumerate(val_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            x_batch = x_batch.to(Config.DEVICE)
            labels = labels.to(Config.DEVICE)
            B, T = x_batch.shape[0], x_batch.shape[1]

            total_samples += B
            total_frames += B * T
            total_batches += 1

            model_states = None
            sensor_state = None
            channel_state = None
            rec_seq = []

            if mode == "ann":
                total_bits += float(B * Config.LATENT_DIM * T)

            for t in range(T):
                x_frame = x_batch[:, t]
                y_enc, sensor_state, aux_dict = sensor(x_frame, state=sensor_state, mode=mode)

                tx_prob = aux_dict.get("tx_prob") if aux_dict else None
                tx_hard = aux_dict.get("tx_hard") if aux_dict else None

                if mode != "ann" and tx_hard is not None:
                    events = tx_hard.sum().item()
                    total_events += events
                    total_elements += B * Config.LATENT_DIM
                    total_bits += events * Config.RATE_BITS_PER_EVENT

                y_noisy, channel_state = channel(
                    y_enc,
                    state=channel_state,
                    mode=mode,
                    tx_prob=tx_prob,
                    tx_hard=tx_hard,
                )

                x_rec, _, model_states, _ = model(y_noisy, states=model_states)
                rec_seq.append(x_rec if x_rec is not None else x_frame)

            rec_stack = torch.stack(rec_seq, dim=1)

            batch_err, batch_tgt = nmse_energy(
                x_batch.mean(dim=1),
                rec_stack.mean(dim=1)
            )
            total_error_energy += batch_err
            total_target_energy += batch_tgt

            if classifier is not None:
                logits = classifier(rec_stack)
                total_acc += accuracy_from_logits(logits, labels)

    avg_bits = total_bits / max(1, total_samples)
    avg_bits_frame = total_bits / max(1, total_frames)
    firing_rate = total_events / max(1.0, total_elements) if mode != "ann" else 0.0

    return {
        "NMSE (dB)": energy_to_nmse_db(total_error_energy, total_target_energy),
        "Classifier Acc (%)": total_acc / max(1, total_batches) if classifier is not None else np.nan,
        "Avg Bits/Sample": avg_bits,
        "Avg Bits/Frame": avg_bits_frame,
        "Tx Firing Rate": firing_rate,
    }


def evaluate_channel_encoding_pair(
    model: torch.nn.Module,
    sensor: EdgeSensor,
    channel: WirelessChannel,
    val_loader,
    classifier: Optional[torch.nn.Module],
    bsc_p: float,
    max_batches: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compare two physical representations of the same hard measurement tensor.

    AER-BSC transmits only active addresses. Dense-BSC bypasses AER and sends
    the complete binary measurement bitmap, including all zero positions.
    The sensor, hard measurements, reconstruction model, and classifier are
    shared, so only the channel representation changes.
    """
    Config.USE_WIRELESS = True
    reset_channel_cache(channel, bsc_p)
    model.eval()
    sensor.eval()
    sensor.mode = "snn"
    if hasattr(model, "mode"):
        model.mode = "spiking"
    if classifier is not None:
        classifier.eval()

    branch_names = ("AER-BSC", "Dense-BSC")
    error_energy = {name: 0.0 for name in branch_names}
    target_energy = {name: 0.0 for name in branch_names}
    deviation_energy = {name: 0.0 for name in branch_names}
    clean_recon_energy = {name: 0.0 for name in branch_names}
    correct = {name: 0 for name in branch_names}
    hamming_errors = {name: 0.0 for name in branch_names}

    clean_error_energy = 0.0
    clean_target_energy = 0.0
    clean_correct = 0
    total_samples = 0
    total_elements = 0.0
    total_events = 0.0

    set_seed(Config.SEED)

    with torch.no_grad():
        for batch_idx, (x_batch, labels) in enumerate(val_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            x_batch = x_batch.to(Config.DEVICE)
            labels = labels.to(Config.DEVICE)
            batch_size, num_steps = x_batch.shape[0], x_batch.shape[1]
            total_samples += int(batch_size)

            sensor_state = None
            channel_state_aer = None
            channel_state_dense = None
            clean_states = None
            aer_states = None
            dense_states = None
            clean_seq = []
            aer_seq = []
            dense_seq = []

            for t in range(num_steps):
                x_frame = x_batch[:, t]
                _, sensor_state, aux = sensor(x_frame, state=sensor_state, mode="snn")
                if aux is None or aux.get("tx_hard") is None:
                    raise RuntimeError("Channel ablation requires EdgeSensor aux['tx_hard'].")

                tx_hard = aux["tx_hard"].detach().float()
                total_events += float(tx_hard.sum().item())
                total_elements += float(tx_hard.numel())

                # No-channel reference: the exact same hard measurement enters
                # the exact same receiver without any bit flips.
                x_clean, _, clean_states, _ = model(tx_hard, states=clean_states)

                # Address-event representation: only active indices are sent.
                y_aer, channel_state_aer = channel(
                    tx_hard,
                    state=channel_state_aer,
                    mode="spiking",
                    tx_hard=tx_hard,
                )
                x_aer, _, aer_states, _ = model(y_aer, states=aer_states)

                # Dense bitmap representation: every zero and one is sent as a
                # channel bit. This deliberately bypasses AER address coding.
                y_dense, channel_state_dense = channel(
                    tx_hard,
                    state=channel_state_dense,
                    mode="ann",
                )
                x_dense, _, dense_states, _ = model(y_dense, states=dense_states)

                clean_seq.append(x_clean if x_clean is not None else x_frame)
                aer_seq.append(x_aer if x_aer is not None else x_frame)
                dense_seq.append(x_dense if x_dense is not None else x_frame)

                hamming_errors["AER-BSC"] += float((y_aer != tx_hard).sum().item())
                hamming_errors["Dense-BSC"] += float((y_dense != tx_hard).sum().item())

            clean_stack = torch.stack(clean_seq, dim=1)
            branch_stacks = {
                "AER-BSC": torch.stack(aer_seq, dim=1),
                "Dense-BSC": torch.stack(dense_seq, dim=1),
            }

            target_mean = x_batch.mean(dim=1)
            clean_mean = clean_stack.mean(dim=1)
            clean_err, clean_tgt = nmse_energy(target_mean, clean_mean)
            clean_error_energy += clean_err
            clean_target_energy += clean_tgt

            if classifier is not None:
                clean_logits = classifier(clean_stack)
                clean_correct += int((clean_logits.argmax(dim=1) == labels).sum().item())

            for name, rec_stack in branch_stacks.items():
                rec_mean = rec_stack.mean(dim=1)
                branch_err, branch_tgt = nmse_energy(target_mean, rec_mean)
                error_energy[name] += branch_err
                target_energy[name] += branch_tgt
                deviation_energy[name] += float(torch.sum((rec_mean - clean_mean) ** 2).item())
                clean_recon_energy[name] += float(torch.sum(clean_mean ** 2).item())

                if classifier is not None:
                    logits = classifier(rec_stack)
                    correct[name] += int((logits.argmax(dim=1) == labels).sum().item())

    if total_samples == 0 or total_elements == 0:
        raise RuntimeError("Channel ablation evaluated zero samples.")

    bits_per_event = float(Config.RATE_BITS_PER_EVENT)
    aer_bits_per_sample = total_events * bits_per_event / total_samples
    dense_bits_per_sample = total_elements / total_samples
    rate_ratio = aer_bits_per_sample / max(dense_bits_per_sample, 1e-12)
    firing_rate = total_events / total_elements
    q_address = 1.0 - (1.0 - float(bsc_p)) ** int(Config.RATE_BITS_PER_EVENT)
    exact_threshold = (
        bits_per_event * float(bsc_p) / max(2.0 * q_address, 1e-12)
        if bsc_p > 0 else 0.5
    )
    aer_ber_bound = 2.0 * firing_rate * q_address

    clean_nmse = energy_to_nmse_db(clean_error_energy, clean_target_energy)
    clean_acc = 100.0 * clean_correct / total_samples if classifier is not None else np.nan
    results: Dict[str, Dict[str, float]] = {}
    for name in branch_names:
        results[name] = {
            "NMSE (dB)": energy_to_nmse_db(error_energy[name], target_energy[name]),
            "Accuracy (%)": 100.0 * correct[name] / total_samples if classifier is not None else np.nan,
            "Measurement BER": hamming_errors[name] / total_elements,
            "Channel Recon Deviation (dB)": energy_to_nmse_db(
                deviation_energy[name], clean_recon_energy[name]
            ),
            "Clean NMSE (dB)": clean_nmse,
            "Clean Accuracy (%)": clean_acc,
            "AER Bits/Sample": aer_bits_per_sample,
            "Dense Bits/Sample": dense_bits_per_sample,
            "AER/Dense Bit Ratio": rate_ratio,
            "Tx Firing Rate": firing_rate,
            "Address Error Probability": q_address,
            "AER BER Upper Bound": aer_ber_bound,
            "Dense BER Theory": float(bsc_p),
            "Exact Dominance Threshold": exact_threshold,
            "Below Half Payload": float(rate_ratio < 0.5),
            "Theorem Condition Met": float(rate_ratio < exact_threshold),
        }
    return results


# -----------------------------------------------------------------------------
# p-sweep: preserved logic, no retraining
# -----------------------------------------------------------------------------

def run_psweep(args) -> None:
    print("\n" + "=" * 80)
    print(">>> RUNNING BSC P-SWEEP")
    print("=" * 80)

    _, val_loader = utils.get_dataloaders(Config.BATCH_SIZE_CLS)
    channel = WirelessChannel(p=Config.BSC_CROSSOVER_PROB).to(Config.DEVICE)

    rows = []
    loaded = []

    for cfg in build_psweep_models():
        print(f"\n[Loading p-sweep model] {cfg['name']}")
        sensor = EdgeSensor().to(Config.DEVICE)
        model = cfg["class"]().to(Config.DEVICE)

        recon_loaded = load_first_existing(model, sensor, cfg["recon_candidates"], Config.DEVICE, strict_model=False)
        classifier, cls_loaded = load_fixed_classifier(
            cfg["class"], cfg["cls_candidates"], Config.DEVICE
        )

        if recon_loaded is None or cls_loaded is None or classifier is None:
            print(f"    [Skip] Missing reconstruction or classifier checkpoint for {cfg['name']}")
            continue

        loaded.append((cfg, model, sensor, classifier, recon_loaded, cls_loaded))

    if not loaded:
        print("[p-sweep] No checkpoints loaded. Nothing to evaluate.")
        return

    for cfg, model, sensor, classifier, recon_ckpt, cls_ckpt in loaded:
        for p in EVAL_P_GRID:
            print(f"[P-SWEEP] {cfg['name']} | p={p:g}")
            res = evaluate_reconstruction_and_classifier(
                model=model,
                sensor=sensor,
                channel=channel,
                val_loader=val_loader,
                mode=cfg["mode"],
                classifier=classifier,
                bsc_p=p,
                max_batches=args.max_batches_eval,
            )
            rows.append({
                "Experiment": "BSC_P_Sweep",
                "Model": cfg["name"],
                "Display Model": model_display_name(cfg["name"]),
                "Mode": cfg["mode"],
                "Seed": Config.SEED,
                "BSC_p": p,
                "Classifier": "adapted_image",
                "Recon Checkpoint": recon_ckpt,
                "Classifier Checkpoint": cls_ckpt,
                "NMSE (dB)": res["NMSE (dB)"],
                "Accuracy (%)": res["Classifier Acc (%)"],
                "Avg Bits/Sample": res["Avg Bits/Sample"],
                "Avg Bits/Frame": res["Avg Bits/Frame"],
                "Tx Firing Rate": res["Tx Firing Rate"],
            })

    df = pd.DataFrame(rows)
    out = checkpoint_path("benchmark_results_bsc_baseline.csv")
    df.to_csv(out, index=False)
    print(f"\n[p-sweep] Saved: {out}")


# -----------------------------------------------------------------------------
# Same-measurement channel-encoding ablation: no retraining
# -----------------------------------------------------------------------------

def run_channel_encoding_ablation(args) -> None:
    print("\n" + "=" * 80)
    print(">>> RUNNING SAME-MEASUREMENT CHANNEL-ENCODING ABLATION")
    print(">>> AER-BSC: active addresses only | Dense-BSC: complete binary bitmap")
    print("=" * 80)

    _, val_loader = utils.get_dataloaders(Config.BATCH_SIZE_CLS)
    channel = WirelessChannel(p=Config.BSC_CROSSOVER_PROB).to(Config.DEVICE)
    rows = []

    for cfg in build_channel_ablation_models(args):
        print(f"\n[Loading low-rate channel-ablation model] {cfg['name']}")
        sensor = EdgeSensor().to(Config.DEVICE)
        model = cfg["class"]().to(Config.DEVICE)
        recon_loaded = load_first_existing(
            model, sensor, cfg["recon_candidates"], Config.DEVICE, strict_model=False
        )
        classifier, cls_loaded = load_fixed_classifier(
            cfg["class"], cfg["cls_candidates"], Config.DEVICE
        )
        if recon_loaded is None or cls_loaded is None or classifier is None:
            print(f"    [Skip] Missing reconstruction or classifier checkpoint for {cfg['name']}")
            continue

        for p in args.channel_p_grid:
            p = float(p)
            print(f"[CHANNEL ABLATION] {cfg['name']} | p={p:g} | shared TX and receiver")
            pair = evaluate_channel_encoding_pair(
                model=model,
                sensor=sensor,
                channel=channel,
                val_loader=val_loader,
                classifier=classifier,
                bsc_p=p,
                max_batches=args.max_batches_eval,
            )

            for encoding, res in pair.items():
                row = {
                    "Experiment": "Same_Measurement_Channel_Encoding_Ablation",
                    "Model": cfg["name"],
                    "Display Model": cfg["label"],
                    "Channel Encoding": encoding,
                    "Seed": Config.SEED,
                    "BSC_p": p,
                    "Recon Checkpoint": recon_loaded,
                    "Classifier Checkpoint": cls_loaded,
                    **res,
                }
                rows.append(row)
                print(
                    f"    {encoding:9s} | BER={res['Measurement BER']:.5f} | "
                    f"NMSE={res['NMSE (dB)']:.3f} dB | Acc={res['Accuracy (%)']:.2f}%"
                )

            shared = pair["AER-BSC"]
            print(
                f"    Payload: AER={shared['AER Bits/Sample']:.2f}, "
                f"Dense={shared['Dense Bits/Sample']:.2f}, "
                f"ratio={shared['AER/Dense Bit Ratio']:.4f}, "
                f"threshold={shared['Exact Dominance Threshold']:.4f}, "
                f"condition={'YES' if shared['Theorem Condition Met'] > 0.5 else 'NO'}"
            )

    if not rows:
        print("[Channel Ablation] No checkpoints loaded. Nothing to save.")
        return

    out = checkpoint_path("channel_encoding_ablation.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n[Channel Ablation] Saved: {out}")



# -----------------------------------------------------------------------------
# Main-experiment classifier loader for rate-sweep evaluation
# -----------------------------------------------------------------------------

def load_main_phase4_classifier(cfg: Dict) -> Tuple[Optional[torch.nn.Module], Optional[str]]:
    """
    Load the fixed Phase-4 image classifier trained in the main experiment.

    The holder model is separate from the rate-sweep reconstruction model, so the
    classifier weights are fixed and are not overwritten when each alpha checkpoint
    is loaded. This avoids both clean-source domain shift and per-alpha classifier
    retraining noise.
    """
    holder = cfg["class"]().to(Config.DEVICE)
    dummy_sensor = EdgeSensor().to(Config.DEVICE)

    loaded_name = load_first_existing(
        holder,
        dummy_sensor,
        cfg.get("main_cls_candidates", []),
        Config.DEVICE,
        strict_model=False,
    )
    if loaded_name is None:
        print(f"[Rate Sweep] WARNING: no main Phase-4 classifier found for {cfg['name']}.")
        return None, None

    classifier = getattr(holder, "classifier_image", None)
    if classifier is None:
        print(f"[Rate Sweep] WARNING: {cfg['name']} holder has no classifier_image.")
        return None, loaded_name

    classifier.eval()
    for param in classifier.parameters():
        param.requires_grad = False

    print(f"[Rate Sweep] Using fixed main Phase-4 classifier for {cfg['name']}: {loaded_name}")
    return classifier, loaded_name

# -----------------------------------------------------------------------------
# Rate-penalty sweep: uses the main Trainer.train_dynamic_recon()
# -----------------------------------------------------------------------------

def set_rate_alpha_for_model(cfg: Dict, alpha: float) -> None:
    if cfg["alpha_attr"] == "ALPHA_RATE_LISTA":
        Config.ALPHA_RATE_LISTA = float(alpha)
    elif cfg["alpha_attr"] == "ALPHA_RATE_SCNN":
        Config.ALPHA_RATE_SCNN = float(alpha)
    else:
        raise ValueError(f"Unknown alpha_attr: {cfg['alpha_attr']}")


def select_rate_models(rate_model_arg: str) -> List[Dict]:
    models = build_rate_models()
    if rate_model_arg == "all":
        return models
    return [m for m in models if m["name"] == rate_model_arg]


def run_rate_penalty_sweep(args) -> None:
    print("\n" + "=" * 80)
    print(">>> RUNNING RATE-PENALTY SWEEP VIA MAIN TRAINING CODE")
    print("=" * 80)

    train_loader_recon, val_loader_recon = utils.get_dataloaders(Config.BATCH_SIZE_RECON)
    _, val_loader_cls = utils.get_dataloaders(Config.BATCH_SIZE_CLS)

    rows = []
    selected_models = select_rate_models(args.rate_model)

    for cfg in selected_models:
        fixed_classifier, fixed_classifier_ckpt = load_main_phase4_classifier(cfg)

        for alpha in args.alpha_rates:
            alpha = float(alpha)
            if alpha < 2e-4:
                print(f"[Skip] alpha={alpha:g} is below 2e-4.")
                continue

            print("\n" + "=" * 80)
            print(f">>> RATE SWEEP POINT | Model: {cfg['name']} | alpha_rate={alpha:g}")
            print("=" * 80)

            set_seed(Config.SEED)
            set_rate_alpha_for_model(cfg, alpha)
            Config.BSC_CROSSOVER_PROB = float(args.rate_train_p)

            sensor = EdgeSensor().to(Config.DEVICE)
            model = cfg["class"]().to(Config.DEVICE)
            channel_train = WirelessChannel(p=args.rate_train_p).to(Config.DEVICE)
            trainer = Trainer(model, sensor, channel_train, train_loader_recon, val_loader_recon)

            model_name = rate_model_name(cfg, alpha)
            ckpt_name = dynamic_recon_checkpoint_name(model, model_name, cfg["mode"])
            ckpt_path = checkpoint_path(ckpt_name)

            if os.path.exists(ckpt_path) and not args.overwrite:
                print(f"[Rate Sweep] Found existing reconstruction checkpoint: {ckpt_path}")
                safe_load_state(model, sensor, ckpt_path, Config.DEVICE, strict_model=False)
            else:
                print(f"[Rate Sweep] Training reconstruction with main Trainer.train_dynamic_recon().")
                print(f"[Rate Sweep] {cfg['alpha_attr']} = {alpha:g}")
                trainer.train_dynamic_recon(model_name=model_name, mode=cfg["mode"], freeze_sensor=False)
                if not safe_load_state(model, sensor, ckpt_path, Config.DEVICE, strict_model=False):
                    print(f"[Warning] Could not reload saved checkpoint after training: {ckpt_path}")

            # Evaluate using the fixed main Phase-4 image classifier.
            eval_res = evaluate_reconstruction_and_classifier(
                model=model,
                sensor=sensor,
                channel=channel_train,
                val_loader=val_loader_cls,
                mode=cfg["mode"],
                classifier=fixed_classifier,
                bsc_p=args.rate_eval_p,
                max_batches=args.max_batches_eval,
            )

            row = {
                "Experiment": "Rate_Penalty_Sweep",
                "Model": cfg["name"],
                "Display Model": cfg["label"],
                "Mode": cfg["mode"],
                "Seed": Config.SEED,
                "Alpha Rate": alpha,
                "Alpha Attr": cfg["alpha_attr"],
                "Train BSC p": args.rate_train_p,
                "Eval BSC p": args.rate_eval_p,
                "Classifier": "fixed_main_phase4_image",
                "Classifier Checkpoint": fixed_classifier_ckpt,
                "Recon Checkpoint": ckpt_name,
                "NMSE (dB)": eval_res["NMSE (dB)"],
                "Main Phase4 Acc (%)": eval_res["Classifier Acc (%)"],
                "Avg Bits/Sample": eval_res["Avg Bits/Sample"],
                "Avg Bits/Frame": eval_res["Avg Bits/Frame"],
                "Tx Firing Rate": eval_res["Tx Firing Rate"],
            }
            rows.append(row)

            single_csv = checkpoint_path(
                f"rate_penalty_result_{cfg['stem']}_alpha{alpha_tag(alpha)}_seed{Config.SEED}.csv"
            )
            pd.DataFrame([row]).to_csv(single_csv, index=False)
            print(f"[Rate Sweep] Saved point CSV: {single_csv}")
            print(
                f"[Rate Sweep Result] {cfg['name']} alpha={alpha:g} | "
                f"Bits={row['Avg Bits/Sample']:.2f} | "
                f"NMSE={row['NMSE (dB)']:.3f} dB | "
                f"MainPhase4Acc={row['Main Phase4 Acc (%)']:.2f}%"
            )

    if rows:
        df_new = pd.DataFrame(rows)
        agg_path = checkpoint_path("rate_penalty_results.csv")

        if os.path.exists(agg_path) and not args.overwrite_rate_csv:
            old = pd.read_csv(agg_path)
            key_cols = ["Model", "Alpha Rate", "Seed", "Train BSC p", "Eval BSC p"]
            combined = pd.concat([old, df_new], ignore_index=True)
            combined = combined.drop_duplicates(subset=key_cols, keep="last")
        else:
            combined = df_new

        combined = combined.sort_values(["Model", "Alpha Rate"])
        combined.to_csv(agg_path, index=False)
        print(f"\n[Rate Sweep] Saved aggregate CSV: {agg_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", choices=["all", "psweep", "channel", "rate"], default="all")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--tx_rho", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max-batches-eval", type=int, default=None)
    parser.add_argument("--max-batches-train", type=int, default=None)

    parser.add_argument("--rate-model", choices=["ConvLISTA_Img", "SpikingCNN", "all"], default="all")
    parser.add_argument("--alpha-rates", nargs="+", type=float, default=DEFAULT_ALPHA_GRID)
    parser.add_argument("--rate-train-p", type=float, default=0.01)
    parser.add_argument("--rate-eval-p", type=float, default=0.01)

    parser.add_argument(
        "--lowrate-lista-ckpt",
        type=str,
        default=DEFAULT_LOWRATE_LISTA_CKPT,
        help="Evaluation-only S-LISTA checkpoint below one half of ANN bits.",
    )
    parser.add_argument(
        "--lowrate-scnn-ckpt",
        type=str,
        default=DEFAULT_LOWRATE_SCNN_CKPT,
        help="Evaluation-only Spiking CNN checkpoint below one half of ANN bits.",
    )
    parser.add_argument(
        "--channel-model",
        choices=["ConvLISTA_Img", "SpikingCNN", "all"],
        default="all",
        help="Low-rate receiver(s) used in the same-measurement channel ablation.",
    )
    parser.add_argument("--channel-p-grid", nargs="+", type=float, default=EVAL_P_GRID)

    parser.add_argument("--overwrite", action="store_true", help="Retrain reconstruction checkpoints even if they exist.")
    parser.add_argument("--overwrite-rate-csv", action="store_true", help="Overwrite aggregate rate_penalty_results.csv.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    Config.SEED = int(args.seed)
    Config.setup(args)
    Config.DEVICE = torch.device(Config.DEVICE if torch.cuda.is_available() else "cpu")
    Config.print_config()
    set_seed(Config.SEED)

    if args.run in ["all", "psweep"]:
        run_psweep(args)

    if args.run in ["all", "channel"]:
        run_channel_encoding_ablation(args)

    if args.run in ["all", "rate"]:
        run_rate_penalty_sweep(args)


if __name__ == "__main__":
    main()
