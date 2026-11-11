"""Selftest for sn3_fig3.py.

1. 30 small random networks: r_star from sn3_fig3.rstar_summary (the function
   used by the main script) against brute force over all error patterns, using
   an independent numpy implementation of Algorithm S1.  Also checks the
   Theorem 4 criterion for every pattern.  Half of the networks have weights,
   thresholds and rho on a 0.25 grid, so sums land exactly on endpoints.
2. Real model, 20 test samples: flipping the r_star bits found for the argmin
   neuron changes a spike (float32 model and float64 re-implementation).
3. Real model, same samples, r_star >= 2: all 2 x 25 x 700 directional
   (r_star - 1)-bit patterns and 1,000 random (r_star - 1)-bit patterns leave
   every spike unchanged.
4. Identities S26, S31, S32 on the 20 samples under AWGN 0 dB (< 1e-9).

Usage:   python sn3_fig3_selftest.py --gpu 0
                   python sn3_fig3_selftest.py --small-only     (item 1, CPU)
Writes outputs/sn3_fig3_selftest/selftest.json.
"""

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch

import sn3_fig3 as S

OUT = S.ROOT / "outputs" / "sn3_fig3_selftest"


# ----------------------------------------------------------------- item 1
def np_forward(P, G, theta, rho, y):
    """Independent numpy Algorithm S1; returns spikes [T, L-1, N]."""
    T, N, layers = y.shape[0], P.shape[0], len(G)
    states = np.zeros((layers, N))
    spikes = np.zeros((T, layers, N))
    for t in range(T):
        initial = P @ y[t]
        code = np.zeros(N)
        current = initial
        for l in range(layers):
            u = rho * states[l] + current
            xi = np.where(u >= theta[l], 1.0, np.where(u <= -theta[l], -1.0, 0.0))
            states[l] = u - theta[l] * xi
            code = code + xi
            spikes[t, l] = xi
            current = initial - G[l] @ code
    return spikes


def random_network(rng, grid):
    N, M = int(rng.integers(1, 7)), int(rng.integers(1, 7))
    T = int(rng.integers(1, 3))
    while T * M > 12:
        M -= 1
    layers = int(rng.integers(1, 4))
    if grid:
        def draw(shape, lo, hi):
            return rng.integers(lo, hi + 1, size=shape) * 0.25
        P = draw((N, M), -3, 3)
        G = [draw((N, N), -2, 2) for _ in range(layers)]
        theta = draw(layers, 1, 5)
        rho = float(rng.choice([0.0, 0.5]))
    else:
        P = rng.normal(0, 0.6, (N, M))
        G = [rng.normal(0, 0.3, (N, N)) for _ in range(layers)]
        theta = rng.uniform(0.2, 1.2, layers)
        rho = float(rng.choice([0.0, rng.uniform(0.1, 0.9)]))
    s_tx = rng.integers(0, 2, (T, M)).astype(float)
    return P, G, theta, rho, s_tx


def item1(count=30, seed=7):
    rng = np.random.default_rng(seed)
    results, failures, endpoint_nets = [], [], 0
    for net in range(count):
        grid = net % 2 == 0
        P, G, theta, rho, s_tx = random_network(rng, grid)
        T, M = s_tx.shape
        params = S.Params(torch.tensor(P), [torch.tensor(g) for g in G], torch.tensor(theta), 1.0, rho)
        ref = S.forward(params, torch.tensor(s_tx)[None])
        ubar, xi = ref["ubar"][0], ref["xi"][0]
        np_ref = np_forward(P, G, theta, rho, s_tx)
        forward_ok = bool(np.array_equal(np_ref, xi.numpy().astype(float)))
        summary = S.rstar_summary(ubar, xi, params.theta, params.P, torch.tensor(s_tx), rho)
        r_fast = summary["r_star"]
        # brute force over all patterns, and Theorem 4 for each pattern
        n = T * M
        r_brute, thm4_bad = S.INF, 0
        for mask in range(1, 2 ** n):
            flips = np.array([(mask >> q) & 1 for q in range(n)], dtype=float).reshape(T, M)
            y = np.abs(s_tx - flips)
            changed = not np.array_equal(np_forward(P, G, theta, rho, y), np_ref)
            if changed:
                r_brute = min(r_brute, int(flips.sum()))
            delta = torch.tensor(y - s_tx)
            eps = torch.zeros(T, P.shape[0], dtype=torch.float64)
            for t in range(T):
                eps[t] = (rho * eps[t - 1] if t else 0.0) + params.P @ delta[t]
            predicted = not bool(S.inside_J(ubar, xi, params.theta, eps[:, :, None]).all())
            thm4_bad += int(predicted != changed)
        # do prefix sums touch an endpoint exactly?
        tables = S.r_tables(ubar, xi, params.theta, params.P, torch.tensor(s_tx), rho)
        touch = False
        for sums in (tables["s_plus"], tables["s_minus"]):
            for l in range(len(G)):
                u = ubar[:, l, :, None] + sums
                touch |= bool(((u == params.theta[l]) | (u == -params.theta[l])).any())
        endpoint_nets += int(touch)
        ok = forward_ok and r_fast == r_brute and thm4_bad == 0
        results.append({"net": net, "grid": grid, "N": P.shape[0], "M": M, "T": T, "layers": len(G),
                        "rho": rho, "r_fast": r_fast if r_fast < S.INF else "inf",
                        "r_brute": r_brute if r_brute < S.INF else "inf", "forward_ok": forward_ok,
                        "thm4_mismatches": thm4_bad, "endpoint_touched": touch, "ok": ok})
        if not ok:
            failures.append(results[-1])
    passed = not failures
    print(f"[1] small networks: {count - len(failures)}/{count} agree; "
          f"{endpoint_nets} with prefix sums exactly on an endpoint; "
          f"r_brute values {sorted(set(str(r['r_brute']) for r in results))}", flush=True)
    return {"passed": passed, "networks": count, "agree": count - len(failures),
            "endpoint_networks": endpoint_nets, "failures": failures, "results": results}


# ------------------------------------------------------------- items 2-4
def spikes_float32(model, frames):
    """Frame-wise float32 trace (rho = 0, zero initial state): [B, L-1, N] int8."""
    trace = model.trace(frames)
    return torch.stack(trace["spikes"], 1).to(torch.int8)


@torch.no_grad()
def items_real(gpu, n_samples=20, n_random=1000, seed=11):
    ctx = S.setup(gpu)
    torch.backends.cuda.matmul.allow_tf32 = False      # strict float32 for the checks
    model = ctx["model"]
    p64 = S.params_from_model(model)
    if p64.rho != 0.0:
        raise RuntimeError("items 2-3 assume rho = 0 (frame-wise patterns)")
    dataset = ctx["loader"].dataset
    rng = np.random.default_rng(seed)
    picks = np.sort(rng.choice(len(dataset), n_samples, replace=False))
    x = dataset.x[torch.as_tensor(picks, device=dataset.x.device)]
    awgn = S.make_channel(ctx, "awgn", 0.0)
    tx, rx, _ = S.transmit(ctx, awgn, x)
    T, M = tx.shape[1], tx.shape[2]
    item2, item3 = [], []
    for b, sample in enumerate(picks.tolist()):
        s_tx = tx[b].to(torch.float64)
        ref = S.forward(p64, tx[b:b + 1])
        ubar, xi = ref["ubar"][0], ref["xi"][0]
        summary = S.rstar_summary(ubar, xi, p64.theta, p64.P, s_tx, 0.0)
        r_star = summary["r_star"]
        # item 2
        if r_star < S.INF:
            k, i, layer = summary["frame"], summary["neuron"], summary["layer"]
            y = tx[b].clone()
            for frame, channel in summary["bits"]:
                y[frame, channel] = 1.0 - y[frame, channel]
            pert64 = S.forward(p64, y[None])
            sp32 = spikes_float32(model, y[k][None])[0]
            ref32 = spikes_float32(model, tx[b, k][None])[0]
            any64 = bool((pert64["xi"][0] != xi).any())
            any32 = bool((sp32 != ref32).any())
            neuron64 = [l + 1 for l in range(p64.layers) if pert64["xi"][0, k, l, i] != xi[k, l, i]]
            delta = (y - tx[b]).to(torch.float64)
            eps = delta @ p64.P.T
            first_pred = int(S.predicted_changes({"ubar": ubar[None], "xi": xi[None]}, eps[None], p64.theta)[0, k])
            first_act = int(S.actual_changes({"xi": xi[None]}, {"xi": pert64["xi"]})[0, k])
            item2.append({"sample": sample, "r_star": r_star, "frame": k + 1, "neuron": i,
                          "direction": summary["direction"], "recorded_layer": layer,
                          "spike_changed_float64": any64, "spike_changed_float32": any32,
                          "neuron_first_changed_layer_float64": neuron64[0] if neuron64 else 0,
                          "frame_first_layer_pred": first_pred, "frame_first_layer_actual": first_act})
        else:
            item2.append({"sample": sample, "r_star": "inf", "note": "no error pattern changes a spike"})
        # item 3
        if r_star >= 2 and r_star < S.INF:
            q = r_star - 1
            tables = S.r_tables(ubar, xi, p64.theta, p64.P, s_tx, 0.0)
            c = tables["c"]                                           # [T, N, M]
            frames_list, keys = [], []
            for direction, order in (("+", tables["pos_idx"]), ("-", tables["neg_idx"])):
                top = order[:, :, :q]                                 # [T, N, q]
                sign_ok = torch.gather(c, 2, top)
                sign_ok = sign_ok > 0 if direction == "+" else sign_ok < 0
                for k in range(T):
                    base = tx[b, k].repeat(c.shape[1], 1)             # [N, M]
                    flips = torch.zeros_like(base)
                    flips.scatter_(1, top[k], sign_ok[k].to(base.dtype))
                    frames_list.append((base - flips).abs())
                    keys.append(k)
            frames = torch.cat(frames_list)                          # [2 T N, M]
            frame_index = torch.tensor(keys, device=tx.device).repeat_interleave(c.shape[1])
            changed32 = changed64 = 0
            for start in range(0, frames.shape[0], 4096):
                chunk = frames[start:start + 4096]
                kk = frame_index[start:start + 4096]
                sp = spikes_float32(model, chunk)
                ref32 = spikes_float32(model, tx[b, kk])
                bad = (sp != ref32).flatten(1).any(1)
                changed32 += int(bad.sum())
                f64 = S.forward(p64, chunk[:, None])["xi"][:, 0]
                changed64 += int((f64 != xi[kk]).flatten(1).any(1).sum())
            # random (r_star - 1)-bit patterns over all T*M bits
            rand_changed32 = rand_changed64 = 0
            ref_seq32 = None
            for start in range(0, n_random, 100):
                count = min(100, n_random - start)
                ys = tx[b].repeat(count, 1, 1).reshape(count, -1)
                for row in range(count):
                    picks_bits = torch.as_tensor(rng.choice(T * M, q, replace=False), device=ys.device)
                    ys[row, picks_bits] = 1.0 - ys[row, picks_bits]
                ys = ys.reshape(count, T, M)
                _, sp = S.float32_run(model, ys)
                if ref_seq32 is None:
                    _, ref_seq32 = S.float32_run(model, tx[b:b + 1])
                rand_changed32 += int((sp != ref_seq32).flatten(1).any(1).sum())
                f64 = S.forward(p64, ys)["xi"]
                rand_changed64 += int((f64 != xi[None]).flatten(1).any(1).sum())
            item3.append({"sample": sample, "r_star": r_star, "directional_patterns": int(frames.shape[0]),
                          "directional_changed_float32": changed32, "directional_changed_float64": changed64,
                          "random_patterns": n_random, "random_changed_float32": rand_changed32,
                          "random_changed_float64": rand_changed64})
        else:
            item3.append({"sample": sample, "r_star": r_star if r_star < S.INF else "inf",
                          "note": "skipped (r_star = 1)" if r_star == 1 else "skipped (r_star = inf)"})
    # item 4
    f = S.analyze_frames(p64, tx, rx, x)
    resid = {name: float(f[name].max()) for name in ("s26", "s31", "s32")}
    classes = np.bincount(f["cls"].flatten().cpu().numpy(), minlength=4).tolist()
    thm4 = int(((f["first_actual"] == f["first_pred"])).sum())
    item4 = {"residuals": resid, "frame_classes": classes, "thm4_first_layer_agree": thm4,
             "frames": int(f["cls"].numel()), "passed": all(v < 1e-9 for v in resid.values())}

    item2_ok = all(r.get("spike_changed_float64", True) for r in item2)
    item3_ok = all(r.get("directional_changed_float64", 0) == 0 and r.get("random_changed_float64", 0) == 0
                   for r in item3)
    f32_notes = {
        "item2_float32_no_change": sum(1 for r in item2 if r.get("spike_changed_float32") is False),
        "item3_float32_changes": sum(r.get("directional_changed_float32", 0) + r.get("random_changed_float32", 0)
                                     for r in item3),
    }
    print(f"[2] r_star bits change a spike (float64): {item2_ok}; "
          f"r_star values {[r['r_star'] for r in item2]}", flush=True)
    print(f"[3] (r_star - 1)-bit patterns leave spikes unchanged (float64): {item3_ok}; "
          f"tested samples {sum(1 for r in item3 if 'directional_patterns' in r)}", flush=True)
    print(f"[4] identities max residual {resid} -> {item4['passed']}", flush=True)
    print(f"    strict float32 notes: {f32_notes}", flush=True)
    return {"item2": {"passed": item2_ok, "rows": item2}, "item3": {"passed": item3_ok, "rows": item3},
            "item4": item4, "float32_notes": f32_notes, "samples": picks.tolist(), "checkpoint": ctx["recon"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpu", type=str)
    parser.add_argument("--small-only", action="store_true")
    args = parser.parse_args()
    if not args.small_only and args.gpu is None:
        parser.error("--gpu is required unless --small-only")
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.time()
    report = {"item1": item1()}
    if not args.small_only:
        report.update(items_real(args.gpu))
    passed = report["item1"]["passed"] and (args.small_only or (
        report["item2"]["passed"] and report["item3"]["passed"] and report["item4"]["passed"]))
    report["passed"] = passed
    (OUT / "selftest.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"{'PASS' if passed else 'FAIL'} ({time.time() - start:.0f}s); {OUT / 'selftest.json'}", flush=True)


if __name__ == "__main__":
    main()
