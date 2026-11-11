"""Self-test of the SHD code.  Run it before starting long training jobs.

    python selftest.py --gpu 0              # unit + synthetic-data tests, ~1 min
    python selftest.py --gpu 0 --real-data  # also 1-epoch runs on real SHD data

Nothing is written outside a temporary directory (printed at the end).
"""

import argparse
import dataclasses
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TMP = Path(tempfile.mkdtemp(prefix="shd_selftest_"))
os.environ["SHD_OUTPUT_DIR"] = str(TMP / "synthetic")   # before importing config

import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.utils.data as data  # noqa: E402

from config import Config  # noqa: E402
import main as shd  # noqa: E402
from snn_models import BoxcarSpike, SpikingLISTA, TemporalConvClassifier  # noqa: E402
from trainer import Trainer, channel_shift, random_channel_shift  # noqa: E402

FAILURES = []


def check(condition, message):
    status = "ok  " if condition else "FAIL"
    print(f"  [{status}] {message}", flush=True)
    if not condition:
        FAILURES.append(message)


def close(a, b, rel=1e-4, abs_tol=1e-6):
    return math.isclose(float(a), float(b), rel_tol=rel, abs_tol=abs_tol)


def test_boxcar(device):
    print("\n[1] Boxcar surrogate")
    x = torch.tensor([0.0, 0.9, 1.05, 1.3, 2.0], device=device, requires_grad=True)
    theta = torch.tensor(1.0, device=device, requires_grad=True)
    width = torch.tensor(0.2, device=device)
    out = BoxcarSpike.apply(x, theta, width)
    check(out.tolist() == [0.0, 0.0, 1.0, 1.0, 1.0], f"forward is H(x-theta): {out.tolist()}")
    out.sum().backward()
    check(x.grad.tolist() == [0.0, 1.0, 1.0, 0.0, 0.0], f"d/dx is the boxcar: {x.grad.tolist()}")
    check(close(theta.grad, -2.0), f"d/dtheta = -sum(boxcar): {float(theta.grad)}")


def random_frames(count, device, rate=0.1, seed=0):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.rand(count, Config.MEAS_DIM, generator=generator) < rate).float().to(device)


def test_model(device):
    print("\n[2] SpikingLISTA forward, calibration, gradients, op counts")
    torch.manual_seed(0)
    model = SpikingLISTA().to(device)
    L1 = model.spiking_layers
    check(L1 == Config.NUM_LAYERS - 1 and len(model.G) == L1,
          f"{L1} spiking layers and {len(model.G)} G matrices for L={Config.NUM_LAYERS}")
    names = [name for name, _ in model.named_parameters()]
    check(not any(("theta" in n or "thresh" in n) for n in names if not n.startswith("semantic_head.")),
          "no learnable parameter name contains 'theta'/'thresh'")

    stats = model.calibrate(random_frames(512, device))
    theta = model.threshold_values()
    sigma = stats["sigma"]
    check(close(theta[0], model.kappa * sigma), f"theta_1 = kappa*sigma ({theta[0]:.4f})")
    check(close(theta[-1], model.beta * sigma), f"theta_L-1 = beta*sigma ({theta[-1]:.4f})")
    if model.kappa > model.beta:
        check(all(a > b for a, b in zip(theta, theta[1:])), "initial thresholds decrease")
    check(close(float(model.eta), 1.0 / theta[-1]), "eta_init = 1/theta_L-1")
    check(close(float(model.calibration_sigma), sigma), "sigma stored in the checkpoint buffer")

    batch = 16
    states = None
    total_spikes = 0.0
    for t in range(3):
        y = random_frames(batch, device, seed=10 + t)
        x, layers, states, info = model(y, states)
        trace = model.last_trace
        values = torch.cat([s.flatten() for s in trace["spikes"]]).unique().tolist()
        check(set(values) <= {-1.0, 0.0, 1.0}, f"t={t}: spikes are ternary {sorted(values)}")
        total_spikes += sum(float(s.abs().sum()) for s in trace["spikes"])
        code = trace["spike_code"]
        check(bool((code == code.round()).all()), f"t={t}: accumulated code is integer")
        check(len(layers) == Config.NUM_LAYERS and layers[-1] is trace["output"],
              f"t={t}: forward returns {L1} codes + readout code")
        off = trace["mask"] == 0
        check(bool((trace["output"][off] == 0).all()), f"t={t}: readout is zero off the gated support")
        on = ~off
        expected = code + model.eta * trace["v"]
        check(bool(torch.allclose(trace["output"][on], expected[on], atol=1e-5)),
              f"t={t}: readout = z^(L-1) + eta*I^(L) on the support")
        check(tuple(states.shape) == (L1, batch, model.n), f"t={t}: state shape {tuple(states.shape)}")
        check(bool((x >= 0).all()), f"t={t}: reconstruction is non-negative (ReLU)")
    check(total_spikes > 0, f"spikes occur ({int(total_spikes)} in 3 frames)")

    y = random_frames(batch, device, seed=99)
    noise = torch.randn(L1, batch, model.n, device=device) * 5
    with torch.no_grad():
        saved = float(model.rho)
        model.rho.fill_(0.0)
        a = model(y, None)[0]
        b = model(y, noise)[0]
        check(bool(torch.allclose(a, b)), "rho=0: previous membranes have no effect")
        model.rho.fill_(0.5)
        c = model(y, noise)[0]
        check(not bool(torch.allclose(a, c)), "rho>0: previous membranes change the output")
        model.rho.fill_(saved)

    model.zero_grad(set_to_none=True)
    states, loss = None, 0.0
    for t in range(3):
        x, layers, states, _ = model(random_frames(batch, device, seed=20 + t), states)
        loss = loss + x.square().mean() + layers[-1].square().mean() + model.sparsity_penalty(layers)
    loss.backward()
    for name, param in model.named_parameters():
        if name in ("raw_level", "raw_eta", "P.weight", "G.0.weight",
                    f"G.{L1 - 1}.weight", "synthesis_dict.weight"):
            grad = param.grad
            ok = grad is not None and bool(torch.isfinite(grad).all()) and float(grad.abs().sum()) > 0
            check(ok, f"gradient reaches {name}")

    with torch.no_grad():
        counter = shd.OperationCounter()
        shd.attach_operation_hooks(counter, shd.experiment_registry()["slista_dense"], model, None)
        y = random_frames(batch, device, seed=30)
        model(y, None)
        counter.close()
        reference = SpikingLISTA.operation_counts(y, model.last_trace, model.n)
        check(close(counter.counts["reconstruction"]["AC"], reference["ac"], rel=1e-5),
              f"hook AC count = n*(||y||_1 + sum_l ||z^(l)||_1) ({reference['ac']:.0f})")
        check(close(reference["mac"], float(model.last_trace["mask"].sum())), "readout MAC = ||m||_0")

    model.lambda_s = 0.7
    layers = model(random_frames(batch, device, seed=40), None)[1]
    manual = 0.7 * torch.stack([c.abs().sum(1) for c in layers[:L1]]).mean(0).div(model.n).mean()
    check(close(model.sparsity_penalty(layers), manual), "sparsity penalty uses the L-1 spiking codes only")


class Synthetic(data.Dataset):
    def __init__(self, count, device, seed):
        generator = torch.Generator().manual_seed(seed)
        rate = torch.rand(count, 1, Config.INPUT_DIM, generator=generator) * 0.6
        active = (torch.rand(count, Config.TIME_STEPS, 1, generator=generator) > 0.3).float()
        self.x = torch.poisson(rate * active, generator=generator).to(device)
        self.y = torch.randint(0, Config.NUM_CLASSES, (count,), generator=generator).to(device)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


def test_training(device):
    print("\n[3] Trainer integration on synthetic data (2 epochs)")
    train = data.DataLoader(Synthetic(96, device, 1), batch_size=32, shuffle=True)
    val = data.DataLoader(Synthetic(64, device, 2), batch_size=32)
    registry = shd.experiment_registry()
    for key, tau_type in (("slista_dense", "exp"), ("slista_aer_fixed_tau", "fixed")):
        experiment = registry[key]
        shd.configure(experiment, 42)
        check(Config.TAU_ANNEAL_TYPE == tau_type, f"{key}: TX temperature schedule '{tau_type}'")
        check(close(Config.get_tau(0, 100), 1.0 if tau_type == "exp" else Config.TAU_FINAL),
              f"{key}: tau at epoch 0 = {Config.get_tau(0, 100):.3f}")
        shd.set_seed(42)
        model, sensor, channel = shd.build_components(experiment)
        trainer = Trainer(model, sensor, channel, train, val)
        trainer._unfreeze_all_params("spiking")
        check(model.raw_level.requires_grad and model.raw_eta.requires_grad,
              f"{key}: thresholds and eta are trainable")
        hp = shd.experiment_hp(experiment, epochs=2)
        groups = model.get_optimizer_groups(hp)
        theta_group = [g for g in groups if any(p is model.raw_level for p in g["params"])]
        check(len(theta_group) == 1 and close(theta_group[0]["lr"],
              hp["lr_lista_snn"] * Config.SLISTA_THETA_LR_RATIO),
              f"{key}: threshold LR = ratio * recon LR")
        saved = (Config.SLISTA_WEIGHT_DECAY_P, Config.SLISTA_WEIGHT_DECAY_G)
        Config.SLISTA_WEIGHT_DECAY_P, Config.SLISTA_WEIGHT_DECAY_G = 1e-4, 2e-4
        wd_groups = model.get_optimizer_groups(hp)
        Config.SLISTA_WEIGHT_DECAY_P, Config.SLISTA_WEIGHT_DECAY_G = saved
        p_ids = {id(p) for p in model.P.parameters()}
        g_ids = {id(p) for p in model.G.parameters()}
        sem_ids = {id(p) for p in model.semantic_head.parameters()}
        decay_ok = True
        for g in wd_groups:
            ids = {id(p) for p in g["params"]}
            wd = float(g.get("weight_decay", 0.0))
            if not ids or ids <= sem_ids:
                continue                      # semantic head keeps its own decay
            if ids <= p_ids:
                decay_ok &= close(wd, 1e-4)
            elif ids <= g_ids:
                decay_ok &= close(wd, 2e-4)
            else:
                decay_ok &= wd == 0.0 and not (ids & (p_ids | g_ids))
        covered = set().union(*({id(p) for p in g["params"]} for g in wd_groups))
        check(decay_ok and p_ids <= covered and g_ids <= covered,
              f"{key}: weight decay only on P (1e-4) and G (2e-4)")
        trainer.train_dynamic_recon(hp=hp, model_name=shd.recon_model_name(experiment, 42),
                                    mode="spiking", freeze_sensor=False,
                                    alpha_rate=experiment.alpha_rate)
        path = shd.recon_checkpoint_path(experiment, 42)
        check(os.path.isfile(path), f"{key}: checkpoint written at the path main.py loads")
        same = shd.slista_settings_mismatch(path)
        saved_lambda = Config.SLISTA_LAMBDA_S
        Config.SLISTA_LAMBDA_S = saved_lambda + 0.123
        changed = shd.slista_settings_mismatch(path)
        Config.SLISTA_LAMBDA_S = saved_lambda
        check(same == {} and "SLISTA_LAMBDA_S" in changed,
              f"{key}: checkpoints trained with other S-LISTA settings are detected as stale")
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        info = checkpoint.get("slista") or {}
        check(bool(info.get("calibration")), f"{key}: calibration recorded "
              f"(sigma={info.get('calibration', {}).get('sigma', float('nan')):.4f}, "
              f"active frames {info.get('calibration', {}).get('active_frame_fraction', float('nan')):.2f})")
        init = info.get("calibration", {}).get("theta_init", [])
        moved = max((abs(a - b) for a, b in zip(info.get("thresholds", []), init)), default=0.0)
        check(moved > 0, f"{key}: thresholds were updated by training (max change {moved:.2e})")
        check(checkpoint.get("tau_anneal_type") == tau_type, f"{key}: schedule stored in checkpoint")
        fresh = SpikingLISTA().to(device)
        fresh.load_state_dict(checkpoint["model"], strict=True)
        check(close(float(fresh.rho), Config.SLISTA_RHO), f"{key}: rho restored from checkpoint")
        model.load_state_dict(checkpoint["model"], strict=True)
        joint = trainer.validate_joint_metrics(mode="spiking")
        check(math.isfinite(joint["Temporal-Mean NMSE (dB)"]),
              f"{key}: evaluation NMSE {joint['Temporal-Mean NMSE (dB)']:.2f} dB")
        diagnostics = trainer.slista_diagnostics(mode="spiking")
        check(all(math.isfinite(v) for v in diagnostics.values()),
              f"{key}: diagnostics finite (rate {diagnostics['Firing Rate']:.4f}, "
              f"oscillation {diagnostics['Oscillation Ratio']:.3f})")
        if key == "slista_dense":
            robust = shd.robustness_metrics(trainer, experiment)
            check(len(robust) == 66 and all(math.isfinite(v) for v in robust.values()),
                  f"{key}: robustness grid finite (robust NMSE {robust['Robust NMSE (dB)']:.2f} dB, "
                  f"Rayleigh @-5 {robust['Rayleigh NMSE @-5']:.2f}, @20 {robust['Rayleigh NMSE @20']:.2f})")
            check(trainer.channel is channel and close(channel.snr_db, shd.TRAIN_SNR_DB),
                  f"{key}: training channel restored after the robustness evaluation")
            legacy = {("raw_vartheta" if k == "raw_level" else k): v
                      for k, v in checkpoint["model"].items()}
            fresh = SpikingLISTA().to(device)
            fresh.load_state_dict(legacy, strict=True)
            check(torch.equal(fresh.raw_level, model.raw_level),
                  f"{key}: checkpoints with the parameter name raw_vartheta still load")


def test_budget(device):
    print("\n[3b] Bit-budget experiments: channel count follows the experiment")
    registry = shd.experiment_registry()
    for key, channels in (("slista_dense_c48", 48), ("scnn_dense_c96", 96), ("slista_dense", 256)):
        shd.configure(registry[key], 42)
        model, sensor, channel = shd.build_components(registry[key])
        widths = {m.out_features for m in sensor.modules() if isinstance(m, torch.nn.Linear)}
        inputs = {m.in_features for m in model.modules() if isinstance(m, torch.nn.Linear)}
        check(Config.MEAS_DIM == channels and channels in widths and channels in inputs,
              f"{key}: {channels} channels per frame (sensor out {sorted(widths)}, "
              f"receiver inputs include {channels})")
    base, small = registry["slista_dense"], registry["slista_dense_c48"]
    same = {f.name for f in dataclasses.fields(base)
            if getattr(base, f.name) == getattr(small, f.name)}
    check(same == {f.name for f in dataclasses.fields(base)} - {"key", "channels"},
          "budget keys differ from the formal dense setting only in key and channel count")


def test_classifier_protocol(device):
    print("\n[3c] Downstream-classifier protocol options")
    check((Config.DOWNSTREAM_CHANNEL_POOL, Config.DOWNSTREAM_LOG_INPUT,
           Config.DOWNSTREAM_CHANNEL_SHIFT, Config.DOWNSTREAM_LABEL_SMOOTHING) == (1, False, 40, 0.1),
          "config defaults are the formal protocol (no pooling, linear input, shift 40, smoothing 0.1)")
    formal_tag = Config.classifier_protocol()
    saved = (Config.DOWNSTREAM_CHANNEL_SHIFT, Config.DOWNSTREAM_LABEL_SMOOTHING)
    Config.DOWNSTREAM_CHANNEL_SHIFT, Config.DOWNSTREAM_LABEL_SMOOTHING = 0, 0.0
    plain_tag = Config.classifier_protocol()
    Config.DOWNSTREAM_CHANNEL_SHIFT, Config.DOWNSTREAM_LABEL_SMOOTHING = saved
    experiment = shd.experiment_registry()["ann_lista"]
    path = shd.classifier_checkpoint_path(experiment, 42)
    check(formal_tag == "gn32_w384_shift40_ls0p1_lr1em3_min1em7_ep150"
          and plain_tag == "gn32_w384_default_lr1em3_min1em7_ep150"
          and path.endswith(f"downstream_recon_x_ann_lista_seed42_{formal_tag}.pth"),
          f"classifier checkpoints carry the protocol tag ({formal_tag})")
    x = torch.arange(1, 2 * 3 * 8 + 1, dtype=torch.float32, device=device).reshape(2, 3, 8)
    y = channel_shift(x, torch.tensor([2, -3]))
    check(torch.equal(y[0, :, 2:], x[0, :, :6]) and not y[0, :, :2].any()
          and torch.equal(y[1, :, :5], x[1, :, 3:]) and not y[1, :, 5:].any(),
          "channel shift moves all channels of a sample by its integer, zero fill, same for every frame")
    state = torch.get_rng_state()
    same = random_channel_shift(x, 0)
    check(same is x and torch.equal(state, torch.get_rng_state()),
          "shift 0 is the identity and draws no random numbers (formal training unchanged)")
    shifted = random_channel_shift(x.repeat(50, 1, 1), 3)
    check(bool((shifted.sum(dim=(1, 2)) <= x.repeat(50, 1, 1).sum(dim=(1, 2)) + 1e-6).all()),
          "random shift never creates mass")
    reference = TemporalConvClassifier(input_dim=Config.INPUT_DIM, num_classes=Config.NUM_CLASSES)
    default = TemporalConvClassifier(input_dim=Config.INPUT_DIM, num_classes=Config.NUM_CLASSES,
                                     channel_pool=1, log_input=False)
    shapes = lambda m: {k: tuple(v.shape) for k, v in m.state_dict().items()}  # noqa: E731
    check(shapes(reference) == shapes(default) and default.frame_projection.in_features == Config.INPUT_DIM,
          "default classifier has the formal parameters (formal checkpoints load unchanged)")
    pooled = TemporalConvClassifier(input_dim=Config.INPUT_DIM, num_classes=Config.NUM_CLASSES,
                                    channel_pool=5, log_input=True).to(device)
    sample = torch.randn(4, Config.TIME_STEPS, Config.INPUT_DIM, device=device) * 3
    front = pooled.preprocess(sample)
    manual = sample.reshape(4, Config.TIME_STEPS, Config.INPUT_DIM // 5, 5).sum(-1)
    manual = torch.sign(manual) * torch.log1p(manual.abs())
    check(pooled.frame_projection.in_features == Config.INPUT_DIM // 5 and torch.allclose(front, manual)
          and bool(torch.isfinite(front).all()),
          f"pool 5 + log: {Config.INPUT_DIM} -> {Config.INPUT_DIM // 5} channels, finite for negative inputs")
    check(tuple(pooled(sample).shape) == (4, Config.NUM_CLASSES), "pooled classifier output shape")
    try:
        TemporalConvClassifier(input_dim=Config.INPUT_DIM, num_classes=Config.NUM_CLASSES, channel_pool=3)
        check(False, "pool that does not divide the channel count is rejected")
    except ValueError:
        check(True, "pool that does not divide the channel count is rejected")

    train = data.DataLoader(Synthetic(64, device, 3), batch_size=32, shuffle=True)
    val = data.DataLoader(Synthetic(32, device, 4), batch_size=32)
    experiment = shd.experiment_registry()["slista_dense"]
    shd.configure(experiment, 42)
    shd.set_seed(42)
    model, sensor, channel = shd.build_components(experiment)
    trainer = Trainer(model, sensor, channel, train, val)
    saved = (Config.DOWNSTREAM_CHANNEL_POOL, Config.DOWNSTREAM_LOG_INPUT,
             Config.DOWNSTREAM_CHANNEL_SHIFT, Config.DOWNSTREAM_LABEL_SMOOTHING)
    try:
        (Config.DOWNSTREAM_CHANNEL_POOL, Config.DOWNSTREAM_LOG_INPUT,
         Config.DOWNSTREAM_CHANNEL_SHIFT, Config.DOWNSTREAM_LABEL_SMOOTHING) = (5, True, 40, 0.1)
        classifier = shd.attach_downstream_classifier(trainer, experiment, 42)
        path = TMP / "synthetic" / "cls_protocol_test.pth"
        # train_phase4 saves only when validation accuracy beats 0; on random
        # synthetic labels it can stay at 0 %, so a fixed value is returned
        # here (the real validation path is exercised by the smoke run).
        trainer.validate_cls = lambda **kwargs: 50.0
        p_before = [p.detach().clone() for p in model.parameters()]
        trainer.train_phase4(hp={"epochs": 1, "lr": 1e-3, "lr_min": 1e-7, "weight_decay": 1e-4},
                             model_name="cls_protocol_test", mode="spiking", target="recon",
                             save_path=str(path))
        state = torch.load(path, map_location=device, weights_only=False)
        check(classifier.channel_pool == 5 and classifier.log_input
              and close(trainer.criterion_cls.label_smoothing, 0.1),
              "classifier built and trained with the configured protocol")
        check(state.get("downstream_channel_pool") == 5 and state.get("downstream_log_input") is True
              and state.get("downstream_channel_shift") == 40
              and close(state.get("downstream_label_smoothing"), 0.1),
              "protocol recorded in the classifier checkpoint")
        check(all(torch.equal(a, b) for a, b in zip(p_before, model.parameters())),
              "reconstruction parameters unchanged by classifier training")
    finally:
        (Config.DOWNSTREAM_CHANNEL_POOL, Config.DOWNSTREAM_LOG_INPUT,
         Config.DOWNSTREAM_CHANNEL_SHIFT, Config.DOWNSTREAM_LABEL_SMOOTHING) = saved


def test_real_data(gpu):
    print("\n[4] Real SHD data, 1 epoch each (also gives the time per epoch)")
    runs = [
        ("recon-only", ["--experiment", "slista_dense", "--recon-only"]),
        ("recon-only + downstream classifier",
         ["--experiment", "slista_dense", "--recon-only", "--downstream"]),
        ("full run (recon + classifier + evaluation)",
         ["--experiment", "slista_dense_no_semantic", "--eval-10db-only"]),
        ("bit budget, S-LISTA 48 channels (recon-only + downstream)",
         ["--experiment", "slista_dense_c48", "--recon-only", "--downstream"]),
        ("bit budget, Vanilla SNN 48 channels (recon-only + downstream)",
         ["--experiment", "scnn_dense_c48", "--recon-only", "--downstream"]),
    ]
    for label, extra in runs:
        out = TMP / "real" / (extra[1] + ("_downstream" if "--downstream" in extra else ""))
        cmd = [sys.executable, "-u", str(ROOT / "main.py"), "--gpu", "0", "--seed", "42",
               "--epochs", "1", "--results-dir", str(out)] + extra
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        env.pop("SHD_OUTPUT_DIR", None)
        log = out.with_suffix(".log")
        out.parent.mkdir(parents=True, exist_ok=True)
        started = time.time()
        with log.open("w") as handle:
            code = subprocess.run(cmd, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT).returncode
        minutes = (time.time() - started) / 60
        check(code == 0, f"{label}: exit code {code} after {minutes:.1f} min (log: {log})")
        if code == 0 and extra[1].endswith("_c48"):
            point = pd.read_csv(out / f"recon_point_{extra[1]}_seed42.csv").iloc[0]
            check(close(point["Bits/Sample"], 48 * Config.TIME_STEPS),
                  f"{label}: {point['Bits/Sample']:.0f} bits/sample = 48 x {Config.TIME_STEPS}")
        if code == 0:
            for line in log.read_text().splitlines():
                if "S-LISTA calibration" in line or "[Recon-only]" in line or "[Done]" in line:
                    print("        " + line.strip())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--real-data", action="store_true")
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    Config.DEVICE = device
    Config.RESULTS_DIR = str(TMP / "synthetic")
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    print(f"Device {device}; S-LISTA settings {Config.slista_settings()}")
    test_boxcar(device)
    test_model(device)
    test_training(device)
    test_budget(device)
    test_classifier_protocol(device)
    if args.real_data:
        test_real_data(args.gpu)
    print(f"\nTemporary files: {TMP}")
    if FAILURES:
        print(f"\n{len(FAILURES)} check(s) FAILED:")
        for message in FAILURES:
            print("  - " + message)
        sys.exit(1)
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
