"""Self-test for the DVS-Gesture code.  Run it before starting long training jobs.

    python selftest.py                 # unit checks + synthetic end-to-end run (~5 min)
    python selftest.py --real-data     # also: 1 epoch on the real data, timing and peak memory

Writes nothing outside selftest_tmp/ (deleted at the end unless --keep).
"""

import argparse
import math
import shutil
import sys
import time
import traceback

import config as C

RESULTS = []


def check(name):
    """Decorator that runs the test immediately and records the outcome."""
    def wrap(fn):
        start = time.time()
        try:
            info = fn()
            RESULTS.append((name, True, info or "", time.time() - start))
            print(f"PASS {name} {info or ''} ({time.time() - start:.1f}s)", flush=True)
        except Exception as error:
            RESULTS.append((name, False, f"{type(error).__name__}: {error}", time.time() - start))
            print(f"FAIL {name}: {error}", flush=True)
            traceback.print_exc()
        return fn
    return wrap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=C.DEFAULT_GPUS[0])
    parser.add_argument("--real-data", action="store_true")
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()

    device = C.setup_torch(args.gpu)
    import torch
    import torch.nn as nn

    import channels as CH
    import op_counter as energy
    import experiments as E
    import trainer as T
    from baseline_models import ConvolutionalLISTA, SpikingCNNRecon
    from sensor import EdgeSensor
    from slista import BoxcarSpike, SpikingLISTA

    tmp = C.ROOT / "selftest_tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()

    # ------------------------------------------------------------ S-LISTA core
    @check("boxcar surrogate (value, d/dx, d/dtheta)")
    def _():
        x = torch.tensor([0.0, 0.9, 1.0, 1.05, 2.0], requires_grad=True)
        theta = torch.tensor(1.0, requires_grad=True)
        y = BoxcarSpike.apply(x, theta, torch.tensor(0.1))
        assert y.tolist() == [0.0, 0.0, 1.0, 1.0, 1.0]
        y.sum().backward()
        assert x.grad.tolist() == [0.0, 0.0, 1.0, 1.0, 0.0], x.grad
        assert abs(float(theta.grad) + 2.0) < 1e-6, theta.grad

    @check("S-LISTA forward: ternary spikes, integer codes, gated readout, memory")
    def _():
        torch.manual_seed(0)
        m = SpikingLISTA(4, rho=0.3, lambda_s=0.01).to(device)
        m.set_initial_thresholds(0.5)
        theta = [float(v) for v in m.theta]
        assert all(a > b for a, b in zip(theta, theta[1:])), theta
        assert abs(theta[0] - 1.25) < 1e-4 and abs(theta[-1] - 1.0) < 1e-4, theta
        assert abs(float(m.eta) - 1.0) < 1e-4
        y = (torch.rand(3, 4 * 32 * 32, device=device) < 0.3).float()
        x1, z1, s1, tr = m(y)
        assert x1.shape == (3, 2, 128, 128) and z1.shape == (3, 128, 32, 32) and len(s1) == 7
        for spike in tr["spikes"]:
            assert set(torch.unique(spike).tolist()) <= {-1.0, 0.0, 1.0}
        code = tr["spike_code"]
        assert torch.equal(code, code.round())
        current = m.P(y.view(3, 4, 32, 32)) - m.G[-1](code)
        expected = code + m.eta * (code != 0).float() * current
        assert torch.allclose(z1, expected, atol=1e-5)
        x2, z2, _, _ = m(y, s1)
        assert not torch.allclose(z1, z2), "membrane memory has no effect"
        m0 = SpikingLISTA(4, rho=0.0).to(device)
        assert m0(y)[2] is None
        return f"theta {theta[0]:.3f}->{theta[-1]:.3f}"

    @check("S-LISTA gradients reach P, G, theta, eta, dictionary and transmitter")
    def _():
        torch.manual_seed(1)
        sensor = EdgeSensor(4, tx_threshold=0.25).to(device).train()
        m = SpikingLISTA(4, rho=0.3, lambda_s=0.01).to(device).train()
        m.set_initial_thresholds(0.3)
        channel = CH.AWGNChannel(10.0, "dense").to(device).train()
        x = torch.rand(2, 3, 2, 128, 128, device=device)
        loss, states, s_state = 0.0, None, None
        for t in range(3):
            y, s_state, aux = sensor(x[:, t], s_state, mode="spiking", tau=0.5)
            y, _ = channel(y, mode="spiking", tx_prob=aux["tx_prob"], tx_hard=aux["tx_hard"])
            x_rec, z, states, trace = m(y, states)
            loss = loss + ((x_rec - x[:, t]) ** 2).mean() + m.sparsity_penalty(trace)
        loss.backward()
        grads = {"P": m.P.weight.grad, "G": m.G[0].weight.grad, "theta": m.raw_level.grad,
                 "eta": m.raw_eta.grad, "tail": m.tail[-1].weight.grad,
                 "sensor": sensor.encoder[0].weight_v.grad}
        dead = [k for k, g in grads.items() if g is None or float(g.abs().sum()) == 0.0]
        assert not dead, f"no gradient: {dead}"

    @check("S-LISTA AC/MAC counts equal the convolution-hook convention")
    def _():
        torch.manual_seed(2)
        m = SpikingLISTA(4).to(device).eval()
        m.set_initial_thresholds(0.2)
        y = (torch.rand(2, 4096, device=device) < 0.2).float()
        _, _, _, tr = m(y)
        counts = m.frame_counts(tr)
        dense_per_unit = 128 * 9
        manual = dense_per_unit * (float(y.sum()) + sum(float(c.abs().sum()) for c in tr["codes"]))
        assert abs(float(counts["ac"]) - manual) < 1e-3 * manual
        assert float(counts["mac"]) == float((tr["spike_code"] != 0).sum())

    @check("sparsity penalty = lambda * mean_l ||z_l||_1 / n")
    def _():
        m = SpikingLISTA(4, lambda_s=0.5)
        codes = [torch.ones(2, 128, 32, 32) * k for k in (1, 2, 3)]
        value = float(m.sparsity_penalty({"codes": codes, "spike_code": codes[-1]}))
        assert abs(value - 0.5 * 2.0) < 1e-6, value

    # -------------------------------------------------------------- channels
    @check("AWGN channel: BER, determinism, AER payload")
    def _():
        ch = CH.AWGNChannel(0.0, "dense").to(device).eval()
        bits = (torch.rand(4, 200000, device=device) < 0.5).float()
        T.deterministic_eval()
        a = ch(bits, mode="spiking", tx_hard=bits)[0]
        T.deterministic_eval()
        b = ch(bits, mode="spiking", tx_hard=bits)[0]
        assert torch.equal(a, b)
        ber = float((a != bits).float().mean())
        assert abs(ber - 0.5 * math.erfc(1.0)) < 2e-3, ber
        hard = torch.zeros(1, 4096, device=device)
        hard[0, :3] = 1
        hard[0, 100] = 1
        assert float(CH.block_aer_bits(hard, 256)[0]) == 2 * 8 + 4 * 4
        return f"BER {ber:.4f}"

    @check("Rayleigh channel: reproducible after eval(), one fade per sample")
    def _():
        ch = CH.RayleighChannel(5.0, "dense").to(device)
        bits = torch.zeros(8, 4096, device=device)
        ch.eval()
        out1, state = ch(bits, mode="spiking", tx_hard=bits)
        out2, _ = ch(bits, state=state, mode="spiking", tx_hard=bits)
        ch.eval()
        again, _ = ch(bits, mode="spiking", tx_hard=bits)
        assert torch.equal(out1, again)
        assert state["bit_error_probability"].shape == (8,)

    # --------------------------------------------------------------- energy
    @check("energy convention: first TX conv MAC, ANN LISTA 7 recurrent convs")
    def _():
        C.Config.set_measurement_channels(4)
        exp_ann = E.REGISTRY["ann_lista"]
        model, sensor = T.build_models(exp_ann, {}, device)
        counter = energy.OperationCounter().attach(exp_ann, model, sensor)
        x = torch.rand(1, 2, 128, 128, device=device)
        with torch.no_grad():
            y, _, _ = sensor(x, mode="ann")
            model(y)
        counter.close()
        sender = 64 * 64 * 16 * 2 * 9 + 64 * 64 * 32 * 16 * 9 + 32 * 32 * 4 * 32 * 9
        recon = 32 * 32 * 128 * 4 * 9 + 7 * 32 * 32 * 128 * 128 * 9
        assert counter.counts[("sender", "mac")] == sender, counter.counts
        assert counter.counts[("recon", "mac")] == recon, counter.counts
        exp_v = E.REGISTRY["vanilla_dense"]
        model, sensor = T.build_models(exp_v, {}, device)
        counter = energy.OperationCounter().attach(exp_v, model, sensor)
        with torch.no_grad():
            y, _, _ = sensor(x, mode="spiking")
            model(y)
        counter.close()
        assert counter.counts[("sender", "mac")] == 64 * 64 * 16 * 2 * 9
        return f"ANN LISTA recon {recon / 1e6:.1f} MMAC/frame"

    @check("baseline classes keep their parameter names")
    def _():
        C.Config.set_measurement_channels(4)
        keys = set(ConvolutionalLISTA().state_dict())
        need = {"P_snn.weight", "PD_snn.0.weight", "fixed_theta_snn", "S_k.0.weight", "theta_ann.0"}
        assert need <= keys, f"missing {sorted(need - keys)}"
        C.Config.use_vanilla_settings()
        v = SpikingCNNRecon()
        assert "thresh" not in v.state_dict(), "Vanilla thresh must stay a non-persistent buffer"
        assert abs(float(v.thresh) - 0.2) < 1e-6, float(v.thresh)
        s = EdgeSensor(4, 0.2).state_dict()
        assert "encoder.0.weight_g" in s and "lif_out.thresh" in s, sorted(s)[:6]

    # -------------------------------------------------- synthetic end-to-end
    @check("synthetic end-to-end: train S-LISTA (dense, AER), Vanilla C=2, classifier, grid")
    def _():
        import data
        torch.manual_seed(3)

        class Synthetic(torch.utils.data.Dataset):
            def __init__(self, n, seed):
                g = torch.Generator().manual_seed(seed)
                self.x = (torch.rand(n, 16, 2, 128, 128, generator=g) < 0.05).float()
                self.y = torch.randint(0, 11, (n,), generator=g)

            def __len__(self):
                return len(self.x)

            def __getitem__(self, i):
                return self.x[i], int(self.y[i])

        def loaders(seed):
            mk = lambda ds, sh: torch.utils.data.DataLoader(ds, batch_size=4, shuffle=sh, num_workers=0)  # noqa: E731
            return mk(Synthetic(8, seed), True), mk(Synthetic(8, seed + 1), False)

        import json
        import main as M
        original = (data.make_loaders, dict(C.CLASSIFIER_TRAINING), list(C.SNR_GRID))
        data.make_loaders = lambda seed, **_: loaders(seed)
        C.CLASSIFIER_TRAINING["epochs"] = 2
        C.SNR_GRID[:] = [0.0, 10.0, 20.0]
        lines = []
        try:
            for key in ("slista_dense", "slista_aer", "vanilla_dense_c2"):
                spec = E.make_spec(key, 42, tmp / key, slista=dict(C.SLISTA_DEFAULTS, rho=0.3), epochs=2)
                spec_path = tmp / f"{key}.json"
                spec_path.write_text(json.dumps(spec))
                M.run_worker(spec_path, args.gpu)
                summary = json.loads((tmp / key / "summary.json").read_text())
                assert summary["has_classifier"] and math.isfinite(summary["robust_nmse"])
                assert M.spec_done(spec), "finished run not recognised"
                changed = E.make_spec(key, 42, tmp / key, slista=dict(C.SLISTA_DEFAULTS, rho=0.5), epochs=2)
                assert key.startswith("vanilla") or not M.spec_done(changed), "settings change not detected"
                lines.append(f"{key}: {summary['minutes']:.1f} min")
        finally:
            data.make_loaders = original[0]
            C.CLASSIFIER_TRAINING.clear()
            C.CLASSIFIER_TRAINING.update(original[1])
            C.SNR_GRID[:] = original[2]
        return "; ".join(lines)

    @check("augmentation: shared shift per sample, semantic_shift 0 = none, default fingerprint, classifier settings")
    def _():
        import random
        random.seed(0)
        x = torch.rand(6, 16, 2, 128, 128)
        y = T.shift_sequence(x, 0.1)
        for b in range(6):            # every frame of a sample moved by the same (dx, dy)
            ref = None
            for dy in range(-12, 13):
                for dx in range(-12, 13):
                    h0, h1, w0, w1 = max(0, dy), min(128, 128 + dy), max(0, dx), min(128, 128 + dx)
                    src = x[b, :, :, max(0, -dy):min(128, 128 - dy), max(0, -dx):min(128, 128 - dx)]
                    if torch.equal(y[b, :, :, h0:h1, w0:w1], src):
                        ref = (dx, dy)
                        break
                if ref:
                    break
            assert ref is not None, f"sample {b}: not a shared translation within 12 px"
        z = torch.rand(2, 16, 128, 32, 32)
        assert torch.equal(T.augment_codes(z, 0.0), z), "semantic_shift 0 must leave the codes unchanged"
        base = dict(C.SLISTA_DEFAULTS)
        reduced = dict(base)
        reduced.pop("semantic_shift")
        f_new = E.recon_fingerprint(E.make_spec("slista_dense", 42, tmp / "fp", slista=base))
        f_old = E.recon_fingerprint(E.make_spec("slista_dense", 42, tmp / "fp", slista=reduced))
        f_off = E.recon_fingerprint(E.make_spec("slista_dense", 42, tmp / "fp", slista=dict(base, semantic_shift=0.0)))
        assert f_new == f_old and f_off != f_new, "semantic_shift 0.1 must not change the fingerprint"
        lrs = {k: E.make_spec(k, 42, tmp / "fp")["classifier_settings"]["lr"]
               for k in ("slista_dense", "vanilla_dense", "ann_lista", "ann_lstm")}
        assert lrs == {"slista_dense": 5e-4, "vanilla_dense": 8e-4, "ann_lista": 8e-4, "ann_lstm": 5e-4}, lrs
        return f"classifier {C.CLASSIFIER_AUGMENT}, lr {lrs}"

    @check("S-LISTA transmitter: output threshold set by tx_threshold, internal LIFs fixed at 0.25")
    def _():
        exp = E.REGISTRY["slista_dense"]
        _, sensor = T.build_models(exp, dict(C.SLISTA_DEFAULTS, tx_threshold=0.5), device)
        values = [float(m.thresh) for m in sensor.modules() if hasattr(m, "thresh")]
        assert values == [0.25, 0.25, 0.5], values
        _, vanilla = T.build_models(E.REGISTRY["vanilla_dense_c4"], {}, device)
        values = [round(float(m.thresh), 6) for m in vanilla.modules() if hasattr(m, "thresh")]
        assert values == [0.2, 0.2, 0.2], values

    @check("calibration restores the transmitter state and the RNG")
    def _():
        torch.manual_seed(5)
        sensor = EdgeSensor(4, tx_threshold=0.25).to(device).eval()
        m = SpikingLISTA(4).to(device)
        before = {k: v.clone() for k, v in sensor.state_dict().items()}

        class Tiny(torch.utils.data.Dataset):
            def __len__(self):
                return 6

            def __getitem__(self, i):
                g = torch.Generator().manual_seed(i)
                return (torch.rand(16, 2, 128, 128, generator=g) < 0.05).float(), 0

        torch.manual_seed(9)
        expected_next = torch.rand(3, device=device)
        torch.manual_seed(9)
        info = T.calibrate_slista(m, sensor, Tiny(), device)
        assert torch.equal(torch.rand(3, device=device), expected_next), "calibration changed the RNG stream"
        after = sensor.state_dict()
        changed = [k for k in before if not torch.equal(before[k], after[k])]
        assert not changed and not sensor.training, f"transmitter changed: {changed}"
        assert info["sigma"] > 0 and bool(m.calibrated)
        return f"sigma {info['sigma']:.3f}, tx rate at init {info['tx_rate_at_init']:.3f}"

    # --------------------------------------------------------- memory check
    @check("peak GPU memory of one S-LISTA training step (batch 64, 16 frames, rho 0.3)")
    def _():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        sensor = EdgeSensor(4).to(device).train()
        m = SpikingLISTA(4, rho=0.3).to(device).train()
        m.set_initial_thresholds(0.3)
        channel = CH.AWGNChannel(10.0, "dense").to(device).train()
        x = (torch.rand(64, 16, 2, 128, 128, device=device) < 0.05).float()
        times = []
        for _ in range(3):
            torch.cuda.synchronize(device)
            start = time.time()
            loss, states, s_state, codes = 0.0, None, None, []
            for t in range(16):
                y, s_state, aux = sensor(x[:, t], s_state, mode="spiking", tau=0.5)
                y, _ = channel(y, mode="spiking", tx_prob=aux["tx_prob"], tx_hard=aux["tx_hard"])
                x_rec, z, states, trace = m(y, states)
                loss = loss + ((x_rec - x[:, t]) ** 2).mean() + m.sparsity_penalty(trace)
                codes.append(z)
            loss = loss + nn.functional.cross_entropy(m.semantic_head(torch.stack(codes, 1)),
                                                      torch.zeros(64, dtype=torch.long, device=device))
            loss.backward()
            m.zero_grad(set_to_none=True)
            sensor.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            times.append(time.time() - start)
        peak = torch.cuda.max_memory_allocated(device) / 2 ** 30
        total = torch.cuda.get_device_properties(device).total_memory / 2 ** 30
        return (f"{peak:.1f} GiB of {total:.0f} GiB; step time first {times[0]:.2f} s, "
                f"then {times[1]:.2f} / {times[2]:.2f} s (19 steps per epoch)")

    # ----------------------------------------------------------- optional
    if args.real_data:
        @check("real data: 1 epoch S-LISTA dense (calibration, timing)")
        def _():
            import json
            import main as M
            spec = E.make_spec("slista_dense", 42, tmp / "real", slista=dict(C.SLISTA_DEFAULTS), epochs=1,
                               recon_only=True, with_classifier=False)
            path = tmp / "real.json"
            path.write_text(json.dumps(spec))
            start = time.time()
            M.run_worker(path, args.gpu)
            summary = json.loads((tmp / "real" / "summary.json").read_text())
            cal = summary["calibration"]
            return (f"sigma {cal['sigma']:.3f}, theta0 {cal['theta_init'][0]:.3f}, tx rate at init "
                    f"{cal['tx_rate_at_init']:.3f}, frames used {cal['frames_used']}/{cal['frames_total']}, "
                    f"1 epoch + 30-point grid {(time.time() - start) / 60:.1f} min")

    failed = [r for r in RESULTS if not r[1]]
    print("\n" + "=" * 70)
    for name, ok, info, _seconds in RESULTS:
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {info}")
    print(f"{len(RESULTS) - len(failed)}/{len(RESULTS)} passed")
    if not args.keep:
        shutil.rmtree(tmp, ignore_errors=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
