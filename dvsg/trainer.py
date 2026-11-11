"""Model construction, training, calibration, downstream classifier and evaluation.

Protocol (the same as for the baselines in baselines/ unless noted):
- per-frame MSE + sparse-code term (+ AER rate term), averaged over T = 16;
  task term alpha_sem * CE / log(11) on the stacked codes (the codes are shifted
  by up to 10 %, one shift per sample; for S-LISTA this is the setting
  "semantic_shift", 0 = no shift); Adam, cosine annealing, clip 1.0;
- checkpoint selection by temporal-mean NMSE on the test split at AWGN 10 dB,
  evaluated after every epoch;
- downstream classifier: VideoClassifier on reconstructed video, 100 epochs,
  AWGN 10 dB, best test accuracy; for every method: random translation
  of the reconstructed video (up to 10 %, one shift per sample) and label
  smoothing 0.1 (config.CLASSIFIER_AUGMENT);
- evaluation noise is reseeded before every pass (AWGN: global generator,
  seed 2026; Rayleigh: channel generator, seed 2026).  One pass yields the
  NMSE, the task-head accuracy and the downstream accuracy.
- the per-epoch validation reseeds the global generator, so every
  training epoch starts its channel noise and latent shifts from the same
  state (the data order still changes).  Kept for parity with the baselines.
"""

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import config as C
from channels import AWGNChannel, block_aer_bits, expected_block_bits, make_channel
from baseline_models import ConvLSTMRecon, ConvolutionalLISTA, SpikingCNNRecon, VideoClassifier
from sensor import EdgeSensor
from slista import SpikingLISTA


# --------------------------------------------------------------- utilities
def set_seed(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def deterministic_eval(seed=C.EVAL_CHANNEL_SEED):
    set_seed(seed)


def atomic_json(path, payload):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=float))
    tmp.replace(path)


def tau_schedule(training, epoch):
    if training.get("tau_mode", "anneal") == "fixed":
        return float(training["tau_final"])
    epochs = int(training["epochs"])
    t0, t1 = float(training["tau_init"]), float(training["tau_final"])
    return t0 * (t1 / t0) ** (epoch / max(1, epochs - 1))


# -------------------------------------------------------------- construction
def build_models(experiment, slista_settings, device):
    """(model, sensor) for one experiment.  Baseline classes read config.Config."""
    channels = int(experiment.channels)
    C.Config.set_measurement_channels(channels)
    if experiment.method == "slista":
        s = slista_settings
        sensor = EdgeSensor(channels, tx_threshold=s["tx_threshold"], tx_rho=C.Config.TX_RHO,
                            internal_threshold=C.SLISTA_INTERNAL_LIF_THRESHOLD)
        model = SpikingLISTA(channels, kappa=s["kappa"], beta=s["beta"], alpha=s["alpha"], rho=s["rho"],
                             lambda_s=s["lambda_s"], threshold_floor=s["threshold_floor"])
    elif experiment.method == "vanilla":
        C.Config.use_vanilla_settings()
        sensor = EdgeSensor(channels, tx_threshold=C.Config.VANILLA["tx_threshold"], tx_rho=C.Config.TX_RHO)
        model = SpikingCNNRecon()
        model.mode = "spiking"
    elif experiment.method == "ann_lista":
        sensor = EdgeSensor(channels, tx_rho=C.Config.TX_RHO)
        model = ConvolutionalLISTA()
        model.mode = "rate"
    elif experiment.method == "ann_lstm":
        sensor = EdgeSensor(channels, tx_rho=C.Config.TX_RHO)
        model = ConvLSTMRecon()
    else:
        raise ValueError(experiment.method)
    return model.to(device), sensor.to(device)


def step(model, y, states):
    """Uniform per-frame call: (x_hat, z, states, trace)."""
    if isinstance(model, SpikingLISTA):
        return model(y, states)
    x_rec, layers, states, _ = model(y, states=states)
    return x_rec, layers[-1], states, {"layers": layers}


def shift_sequence(sequence, ratio, probability=1.0):
    """Zero-padded integer translation by up to ``ratio`` of H and W; one shift per sample,
    shared by all frames of the sequence [B, T, C, H, W]."""
    batch, _, _, height, width = sequence.shape
    max_dx, max_dy = int(ratio * width), int(ratio * height)
    if ratio <= 0 or probability <= 0 or (max_dx == 0 and max_dy == 0):
        return sequence
    out = torch.zeros_like(sequence)
    for b in range(batch):
        if random.random() > probability:
            out[b] = sequence[b]
            continue
        dx = random.randint(-max_dx, max_dx) if max_dx > 0 else 0
        dy = random.randint(-max_dy, max_dy) if max_dy > 0 else 0
        out[b, :, :, max(0, dy):min(height, height + dy), max(0, dx):min(width, width + dx)] = \
            sequence[b, :, :, max(0, -dy):min(height, height - dy), max(0, -dx):min(width, width - dx)]
    return out


def augment_codes(z_sequence, ratio=None):
    """Task-head input during reconstruction training (baselines: ratio 0.1, probability 1)."""
    ratio = C.Config.SEMANTIC_SHIFT_RATIO if ratio is None else float(ratio)
    return shift_sequence(z_sequence, ratio, C.Config.SEMANTIC_SHIFT_PROB)


# -------------------------------------------------------------- calibration
@torch.no_grad()
def calibrate_slista(model, sensor, train_set, device, samples=C.CALIBRATION_SAMPLES):
    """sigma = std of P * y over all coordinates, frames with >= 1 received spike.

    Runs the freshly initialised transmitter in training mode (batch statistics)
    and AWGN at 10 dB; the transmitter state (BatchNorm running statistics) and
    the RNG state of the training run are restored afterwards.  (weight_norm
    modules cannot be deep-copied, hence the snapshot instead of a copy.)
    """
    snapshot = {k: v.detach().clone() for k, v in sensor.state_dict().items()}
    was_training = sensor.training
    with torch.random.fork_rng(devices=[device.index if device.index is not None else 0]):
        torch.manual_seed(C.CALIBRATION_SEED)
        index = torch.randperm(len(train_set), generator=torch.Generator().manual_seed(C.CALIBRATION_SEED))
        subset = torch.utils.data.Subset(train_set, index[:samples].tolist())
        loader = torch.utils.data.DataLoader(subset, batch_size=C.BATCH_SIZE, shuffle=False,
                                             num_workers=C.NUM_WORKERS)
        probe = sensor.train()
        channel = AWGNChannel(C.TRAIN_SNR_DB, "dense").to(device).eval()
        total = total_sq = count = 0.0
        frames_used = frames_all = 0
        tx_spikes = tx_elements = 0.0
        for x, _ in loader:
            x = x.to(device)
            state = None
            for t in range(x.shape[1]):
                y, state, aux = probe(x[:, t], state=state, mode="spiking", tau=1.0)
                tx_spikes += float(aux["tx_hard"].sum())
                tx_elements += aux["tx_hard"].numel()
                y, _ = channel(y, mode="spiking", tx_hard=aux["tx_hard"])
                current = model.P(y.view(y.shape[0], model.meas_channels, model.grid, model.grid)).double()
                keep = y.sum(dim=1) > 0
                frames_all += int(keep.numel())
                frames_used += int(keep.sum())
                if keep.any():
                    c = current[keep]
                    total += float(c.sum())
                    total_sq += float(c.square().sum())
                    count += c.numel()
        mean = total / count
        sigma = math.sqrt(max(total_sq / count - mean * mean, 0.0))
    sensor.load_state_dict(snapshot, strict=True)
    sensor.train(was_training)
    info = model.set_initial_thresholds(sigma)
    info.update({"frames_used": frames_used, "frames_total": frames_all,
                 "tx_rate_at_init": tx_spikes / max(1.0, tx_elements)})
    return info


# ---------------------------------------------------------------- optimiser
def make_optimizer(experiment, training, settings, model, sensor):
    semantic = experiment.semantic_weight > 0
    for p in model.semantic_head.parameters():
        p.requires_grad_(semantic)
    if experiment.method == "slista":
        scale = float(settings["lr_scale"])
        lr = float(training["lr_recon"]) * scale
        g = model.parameter_groups()
        groups = [
            {"params": g["P"], "lr": lr, "weight_decay": float(settings["p_weight_decay"]), "name": "P"},
            {"params": g["eta"], "lr": lr, "name": "eta"},
            {"params": g["G"], "lr": lr * float(settings["g_lr_ratio"]), "name": "G"},
            {"params": g["theta"], "lr": lr * float(settings["theta_lr_ratio"]), "name": "theta"},
            {"params": g["dict"], "lr": float(training["lr_dict"]) * scale, "name": "dict"},
            {"params": list(sensor.parameters()), "lr": float(training["lr_sensor"]) * scale, "name": "sensor"},
        ]
        if semantic:
            groups.append({"params": g["semantic"], "lr": float(training["lr_semantic"]),
                           "weight_decay": C.Config.SEMANTIC_WEIGHT_DECAY, "name": "semantic"})
    elif experiment.method == "vanilla":
        groups = [g for g in model.get_optimizer_groups(training) if len(g["params"]) > 0]
        groups.append({"params": list(sensor.parameters()), "lr": float(training["lr_sensor"])})
    else:
        raise ValueError("Only S-LISTA and the Vanilla bit-budget runs are trained here; see baselines/train.py")
    for group in groups:
        group["initial_lr"] = group["lr"]
    return optim.Adam(groups, weight_decay=0.0)


def sparsity_term(experiment, training, model, z_trace, batch):
    if isinstance(model, SpikingLISTA):
        return model.sparsity_penalty(z_trace)
    layers = z_trace["layers"]           # form used for the Vanilla runs
    l1 = sum(layer.abs().view(batch, -1).sum(dim=1).mean() for layer in layers) / len(layers)
    return float(training["alpha_l1"]) * l1


# -------------------------------------------------------------------- train
def train_reconstruction(spec, experiment, model, sensor, loaders, device, ckpt_path, history_path, log):
    training, settings = spec["training"], spec["slista"]
    train_loader, test_loader = loaders
    mode = experiment.mode
    channel = AWGNChannel(C.TRAIN_SNR_DB, experiment.transport).to(device)
    # S-LISTA: setting semantic_shift; Vanilla budget runs: 0.1.
    semantic_shift = float(settings.get("semantic_shift", C.Config.SEMANTIC_SHIFT_RATIO))
    calibration = None
    if experiment.method == "slista":
        calibration = calibrate_slista(model, sensor, train_loader.dataset, device)
        log(f"calibration: {calibration}")
    optimizer = make_optimizer(experiment, training, settings, model, sensor)
    epochs = int(training["epochs"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=float(training["lr_min"]))
    params = [p for group in optimizer.param_groups for p in group["params"]]
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()
    semantic_weight = float(experiment.semantic_weight)
    rate_weight = float(experiment.rate_weight)
    history, best = [], float("inf")
    partial = ckpt_path.with_name(ckpt_path.stem + ".partial.pth")
    if partial.exists():
        partial.unlink()
    start = time.time()
    for epoch in range(epochs):
        tau = tau_schedule(training, epoch)
        model.train(); sensor.train(); channel.train()
        sums = {"loss": torch.zeros((), device=device), "recon": torch.zeros((), device=device),
                "ce": torch.zeros((), device=device), "correct": torch.zeros((), device=device),
                "tx": torch.zeros((), device=device)}
        samples = tx_elements = batches = 0
        for x, labels in train_loader:
            x, labels = x.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            batch, frames = x.shape[:2]
            states = sensor_state = None
            loss_frames = 0.0
            codes = []
            for t in range(frames):
                y, sensor_state, aux = sensor(x[:, t], state=sensor_state, mode=mode, tau=tau)
                y, _ = channel(y, mode=mode, tx_prob=aux["tx_prob"], tx_hard=aux["tx_hard"])
                x_rec, z, states, trace = step(model, y, states)
                loss = mse(x_rec, x[:, t]) + sparsity_term(experiment, training, model, trace, batch)
                if rate_weight > 0:
                    if experiment.transport != "aer":
                        raise ValueError("rate regularisation is defined for AER transport")
                    bits = expected_block_bits(aux["tx_prob"], C.NUM_BLOCKS)
                    loss = loss + rate_weight * bits.mean() / aux["tx_prob"].shape[1]
                loss_frames = loss_frames + loss
                codes.append(z)
                sums["tx"] += aux["tx_hard"].detach().sum()
                tx_elements += aux["tx_hard"].numel()
            recon = loss_frames / frames
            total = recon
            if semantic_weight > 0:
                logits = model.semantic_head(augment_codes(torch.stack(codes, dim=1), semantic_shift))
                semantic_ce = ce(logits, labels)
                total = recon + semantic_weight * semantic_ce / math.log(C.Config.NUM_CLASSES)
                sums["ce"] += semantic_ce.detach() * batch
                sums["correct"] += (logits.detach().argmax(1) == labels).sum()
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            sums["loss"] += total.detach()
            sums["recon"] += recon.detach()
            samples += batch
            batches += 1
        scheduler.step()

        val = evaluate(model, sensor, make_channel("awgn", C.TRAIN_SNR_DB, experiment.transport),
                       test_loader, experiment, device)
        row = {"epoch": epoch + 1, "tau": tau,
               "train_loss": float(sums["loss"]) / batches, "train_recon": float(sums["recon"]) / batches,
               "train_ce": float(sums["ce"]) / samples if semantic_weight > 0 else float("nan"),
               "train_head_acc": 100.0 * float(sums["correct"]) / samples if semantic_weight > 0 else float("nan"),
               "train_tx_rate": float(sums["tx"]) / max(1, tx_elements),
               "minutes": (time.time() - start) / 60.0}
        row.update({f"val_{k}": v for k, v in val.items() if not isinstance(v, (list, dict))})
        if isinstance(model, SpikingLISTA):
            row["theta"] = " ".join(f"{v:.4f}" for v in model.theta.detach().cpu().tolist())
            row["eta"] = float(model.eta)
        history.append(row)
        _write_history(history_path, history)
        if val["nmse"] < best:
            best = val["nmse"]
            torch.save({"model": model.state_dict(), "sensor": sensor.state_dict(), "epoch": epoch + 1,
                        "val": val, "fingerprint": spec["fingerprint"], "calibration": calibration,
                        "code_version": C.CODE_VERSION}, partial)
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == epochs:
            extra = f" | theta {row['theta']} | eta {row['eta']:.3f}" if "theta" in row else ""
            log(f"[ep {epoch + 1:03d}] NMSE {val['nmse']:.2f} dB (best {best:.2f}) | head acc "
                f"{val['head_acc']:.2f}% | tx {val['tx_rate'] * 100:.2f}% | rx {val.get('rx_rate', float('nan')):.4f}"
                f" | {row['minutes']:.1f} min{extra}")
    # Only a run that reached its last epoch produces recon.pth; an interrupted run
    # leaves recon.partial.pth, which is never reused.
    if not partial.exists():      # NMSE never finite (diverged): keep the last epoch, flagged
        log("WARNING: no finite validation NMSE; saving the last epoch")
        torch.save({"model": model.state_dict(), "sensor": sensor.state_dict(), "epoch": epochs,
                    "val": val, "fingerprint": spec["fingerprint"], "calibration": calibration,
                    "code_version": C.CODE_VERSION, "diverged": True}, partial)
    partial.replace(ckpt_path)
    return best


def _write_history(path, rows):
    import csv
    keys = list(rows[0].keys())
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


# ----------------------------------------------------------- reconstruction
def run_sequence(model, sensor, channel, x, mode):
    """Reconstruct one batch of sequences in eval mode.  Returns (x_hat, codes, stats)."""
    states = sensor_state = channel_state = None
    recon, codes = [], []
    stats = {"tx": x.new_zeros(()), "aer_bits": x.new_zeros(x.shape[0])}
    slista_counts = []
    for t in range(x.shape[1]):
        y, sensor_state, aux = sensor(x[:, t], state=sensor_state, mode=mode)
        if mode == "spiking":
            stats["tx"] = stats["tx"] + aux["tx_hard"].sum()
            if channel.transport == "aer":
                stats["aer_bits"] = stats["aer_bits"] + block_aer_bits(aux["tx_hard"], channel.num_blocks)
        y, channel_state = channel(y, state=channel_state, mode=mode,
                                   tx_prob=aux["tx_prob"], tx_hard=aux["tx_hard"])
        x_rec, z, states, trace = step(model, y, states)
        if isinstance(model, SpikingLISTA):
            slista_counts.append(model.frame_counts(trace))
        recon.append(x_rec)
        codes.append(z)
    if slista_counts:
        stats["slista"] = {k: sum(c[k] for c in slista_counts) for k in slista_counts[0]}
    return torch.stack(recon, dim=1), torch.stack(codes, dim=1), stats


def bits_per_sample(experiment, aer_bits_mean):
    dim = C.latent_dim(experiment.channels) * C.Config.TIME_STEPS
    if experiment.mode == "ann":
        return float(dim * 8)
    if experiment.transport == "aer":
        return float(aer_bits_mean)
    return float(dim)


@torch.no_grad()
def evaluate(model, sensor, channel, loader, experiment, device, classifier=None, per_sample=False,
             seed=C.EVAL_CHANNEL_SEED):
    """One deterministic pass: NMSE (temporal-mean, full, framewise), head and downstream accuracy."""
    channel = channel.to(device)
    deterministic_eval(seed)
    model.eval(); sensor.eval(); channel.eval()
    if classifier is not None:
        classifier.eval()
    mode = experiment.mode
    err_t = tgt_t = err_f = tgt_f = 0.0
    frame_err = frame_tgt = None
    head_correct = down_correct = total = 0
    tx_spikes = 0.0
    tx_elements = 0
    aer_sum = 0.0
    sl = None
    rows = []
    for x, labels in loader:
        x, labels = x.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        recon, codes, stats = run_sequence(model, sensor, channel, x, mode)
        delta = x - recon
        target_mean, recon_mean = x.mean(dim=1), recon.mean(dim=1)
        err_t += float((target_mean - recon_mean).square().sum())
        tgt_t += float(target_mean.square().sum())
        err_f += float(delta.square().sum())
        tgt_f += float(x.square().sum())
        fe = delta.square().sum(dim=(0, 2, 3, 4)).double().cpu().numpy()
        ft = x.square().sum(dim=(0, 2, 3, 4)).double().cpu().numpy()
        frame_err = fe if frame_err is None else frame_err + fe
        frame_tgt = ft if frame_tgt is None else frame_tgt + ft
        head_pred = model.semantic_head(codes).argmax(dim=1)
        head_correct += int((head_pred == labels).sum())
        down_pred = None
        if classifier is not None:
            down_pred = classifier(recon).argmax(dim=1)
            down_correct += int((down_pred == labels).sum())
        total += int(labels.numel())
        if mode == "spiking":
            tx_spikes += float(stats["tx"])
            tx_elements += x.shape[0] * x.shape[1] * C.latent_dim(experiment.channels)
            aer_sum += float(stats["aer_bits"].sum())
        if "slista" in stats:
            batch_counts = {k: v.double().cpu() for k, v in stats["slista"].items()}
            sl = batch_counts if sl is None else {k: sl[k] + batch_counts[k] for k in sl}
        if per_sample:
            dims = tuple(range(1, target_mean.ndim))
            nmse = 10 * torch.log10((target_mean - recon_mean).square().sum(dim=dims)
                                    / target_mean.square().sum(dim=dims).clamp_min(1e-12) + 1e-10)
            for i in range(x.shape[0]):
                rows.append({"class": int(labels[i]), "nmse_db": float(nmse[i]),
                             "head_pred": int(head_pred[i]),
                             "pred": int(down_pred[i]) if down_pred is not None else -1,
                             "aer_bits": float(stats["aer_bits"][i])})

    def db(err, tgt):
        return float(10.0 * np.log10(err / tgt + 1e-10)) if tgt > 0 else float("nan")

    frames = [db(e, t) for e, t in zip(frame_err, frame_tgt) if t > 0]
    result = {"nmse": db(err_t, tgt_t), "nmse_full": db(err_f, tgt_f),
              "nmse_framewise": float(np.mean(frames)) if frames else float("nan"),
              "head_acc": 100.0 * head_correct / max(1, total),
              "acc": 100.0 * down_correct / max(1, total) if classifier is not None else float("nan"),
              "samples": total,
              "tx_rate": tx_spikes / max(1, tx_elements) if mode == "spiking" else float("nan"),
              "bits": bits_per_sample(experiment, aer_sum / max(1, total))}
    if sl is not None:
        frames_batch = total * C.Config.TIME_STEPS
        result.update({
            "rx_rate": float(sl["layer_spikes"].sum()) / model.firing_denominator(frames_batch),
            "rx_rate_layers": [float(v) / (2.0 * frames_batch * model.code_size) for v in sl["layer_spikes"]],
            "support_per_frame": float(sl["support"]) / frames_batch,
            "rx_ac_per_sample": float(sl["ac"]) / total,
            "rx_mac_per_sample": float(sl["mac"]) / total,
        })
    if per_sample:
        result["per_sample"] = rows
    return result


# --------------------------------------------------------------- classifier
@torch.no_grad()
def classifier_accuracy(classifier, model, sensor, channel, loader, experiment, device):
    """Reseed, reconstruct the test split, classify.

    The reseed also fixes the noise of the following training epoch, exactly as in
    baselines/trainer.py (whose training noise therefore starts from the same state every epoch).
    """
    deterministic_eval()
    model.eval(); sensor.eval(); channel.eval(); classifier.eval()
    correct = total = 0
    for x, labels in loader:
        x, labels = x.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        recon, _, _ = run_sequence(model, sensor, channel, x, experiment.mode)
        correct += int((classifier(recon).argmax(1) == labels).sum())
        total += int(labels.numel())
    return 100.0 * correct / max(1, total)


def train_classifier(spec, experiment, model, sensor, loaders, device, path, log):
    """Downstream classifier (frozen sensing/reconstruction, AWGN 10 dB, best test accuracy) with the
    augmentation of spec["classifier_settings"]: random translation of the reconstructed
    video and label smoothing."""
    train_loader, test_loader = loaders
    set_seed(int(spec["seed"]) + 10000)
    classifier = VideoClassifier(C.Config.NUM_CLASSES, C.Config.TIME_STEPS).to(device)
    hp = C.CLASSIFIER_TRAINING
    settings = spec["classifier_settings"]
    lr = float(settings["lr"])
    shift = float(settings["shift_ratio"])
    optimizer = optim.Adam(classifier.parameters(), lr=lr, weight_decay=hp["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hp["epochs"], eta_min=hp["lr_min"])
    channel = make_channel("awgn", C.TRAIN_SNR_DB, experiment.transport).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(settings["label_smoothing"]))
    best = -1.0
    partial = path.with_name(path.stem + ".partial.pth")
    if partial.exists():
        partial.unlink()
    for epoch in range(hp["epochs"]):
        model.eval(); sensor.eval(); channel.eval()
        classifier.train()
        for x, labels in train_loader:
            x, labels = x.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with torch.no_grad():
                recon, _, _ = run_sequence(model, sensor, channel, x, experiment.mode)
            loss = criterion(classifier(shift_sequence(recon, shift)), labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        scheduler.step()
        accuracy = classifier_accuracy(classifier, model, sensor, channel, test_loader, experiment, device)
        if accuracy > best:
            best = accuracy
            torch.save({"classifier": classifier.state_dict(), "epoch": epoch + 1, "val_accuracy": accuracy,
                        "fingerprint": spec["fingerprint"], "settings": settings}, partial)
        if epoch == 0 or (epoch + 1) % 10 == 0:
            log(f"[cls ep {epoch + 1:03d}] acc {accuracy:.2f}% (best {best:.2f}%)")
    partial.replace(path)
    state = torch.load(path, map_location=device, weights_only=False)
    classifier.load_state_dict(state["classifier"], strict=True)
    return classifier.eval()


def cached_classifier(path, spec, device):
    """The classifier at ``path`` if it was trained for exactly these weights and settings."""
    path = Path(path)
    if not path.is_file():
        return None
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("fingerprint") != spec["fingerprint"] or state.get("settings") != spec.get("classifier_settings"):
        return None
    return load_classifier(path, device)


def load_classifier(path, device):
    classifier = VideoClassifier(C.Config.NUM_CLASSES, C.Config.TIME_STEPS).to(device)
    state = torch.load(path, map_location=device, weights_only=False)
    classifier.load_state_dict(state["classifier"], strict=True)
    return classifier.eval()


# ---------------------------------------------------------- robustness grid
def robustness_grid(model, sensor, loader, experiment, device, classifier=None, snrs=None, log=None):
    rows = []
    for kind in ("awgn", "rayleigh"):
        for snr in (snrs or C.SNR_GRID):
            channel = make_channel(kind, snr, experiment.transport)
            result = evaluate(model, sensor, channel, loader, experiment, device, classifier=classifier)
            row = {"channel": kind, "snr_db": float(snr)}
            row.update({k: v for k, v in result.items() if not isinstance(v, (list, dict))})
            rows.append(row)
            if log:
                log(f"{kind:8s} {snr:5.1f} dB: NMSE {result['nmse']:7.3f} dB | head {result['head_acc']:6.2f}%"
                    f" | acc {result['acc']:6.2f}%")
    return rows


def robust_summary(rows):
    """Mean over AWGN -5..5 dB and Rayleigh -5..20 dB (26 points), as for SHD."""
    pick = [r for r in rows if (r["channel"] == "awgn" and r["snr_db"] <= C.ROBUST_AWGN_MAX_DB)
            or r["channel"] == "rayleigh"]
    out = {"robust_points": len(pick)}
    for metric in ("nmse", "head_acc", "acc"):
        values = [r[metric] for r in pick]
        out[f"robust_{metric}"] = float(np.mean(values)) if values else float("nan")
    for kind in ("awgn", "rayleigh"):
        for snr in (10.0, 20.0):
            match = [r for r in rows if r["channel"] == kind and r["snr_db"] == snr]
            if match:
                for metric in ("nmse", "head_acc", "acc"):
                    out[f"{kind}{int(snr)}_{metric}"] = match[0][metric]
    return out
