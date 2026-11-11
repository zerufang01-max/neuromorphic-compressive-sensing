"""Self-test of sn3_analysis.py / sn3_figs.py (run before sn3_figs.py).

  CUDA_VISIBLE_DEVICES=0 python -u sn3_selftest.py --output runs/final --device cuda:0 \
      > runs/final/sn3_selftest.log 2>&1

Item 1  Small random networks (30 networks, n <= 8, m <= 6, L = 2..5, 40 signals
        each; half of them with every weight, signal value and threshold on a
        0.25 grid so that currents hit the thresholds exactly).  Every function
        of sn3_analysis is compared with a direct scalar implementation written
        here (Python loops, explicit index sets and sub-matrices), and the float64
        reference forward with models.SpikingLISTA.trace run in float64.
Item 2  Real models (seed 42, L = 10, s = 14/28/42/56, first 1,000 test signals),
        through sn3_figs.run_condition: code conventions, float32 vs float64
        spikes, sn3_analysis.forward vs SpikingLISTA.trace in float64,
        S14 residual < 1e-10, S15 relative residual < 1e-12, E_0 constant and
        equal to the closed form, no S18 violation.
Item 3  Same test set as evaluate.py: float32 NMSE over all 10,000 test signals
        equals metrics/results_per_seed.csv (seed 42) within 1e-6 dB.
Writes <output>/sn3_selftest.json (outside sn3_figs/, which sn3_figs.py empties).
--skip-real runs item 1 only (and skips its torch part if torch is missing).
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

import energy_spec as es
import sn3_analysis as sa

NETWORKS = 30
SIGNALS = 40
TOL = 1e-12


# ------------------------------------------------------- direct implementations
def direct_forward(P, G, theta, eta, y):
    n, m = P.shape
    out = {"currents": [], "spikes": [], "codes": []}
    B = y.shape[0]
    cur_all = [np.zeros((B, n)) for _ in range(len(G) + 1)]
    xi_all = [np.zeros((B, n)) for _ in range(len(G))]
    z_all = [np.zeros((B, n)) for _ in range(len(G))]
    output = np.zeros((B, n))
    mask = np.zeros((B, n))
    for b in range(B):
        I1 = [sum(P[i, j] * y[b, j] for j in range(m)) for i in range(n)]
        z = [0.0] * n
        I = list(I1)
        for l in range(len(G)):
            for i in range(n):
                cur_all[l][b, i] = I[i]
                xi = 1.0 if I[i] >= theta[l] else (-1.0 if -I[i] >= theta[l] else 0.0)
                xi_all[l][b, i] = xi
                z[i] += xi
                z_all[l][b, i] = z[i]
            I = [I1[i] - sum(G[l][i, j] * z[j] for j in range(n)) for i in range(n)]
        for i in range(n):
            cur_all[-1][b, i] = I[i]
            mask[b, i] = 1.0 if z[i] != 0 else 0.0
            output[b, i] = z[i] + eta * mask[b, i] * I[i]
    out.update(currents=cur_all, spikes=xi_all, codes=z_all, mask=mask,
               spike_code=z_all[-1], output=output)
    return out


def direct_terms(tr, t, D, Gs, eta, theta, m):
    """Every per-sample quantity of sn3_analysis, from index sets and sub-matrices."""
    n = t.shape[1]
    G = Gs[-1]
    L1 = len(Gs)
    depth = L1 + 1
    const = sa.energy_constants(m, n, depth)
    rows = []
    for b in range(t.shape[0]):
        z, zL, zs = tr["output"][b], tr["spike_code"][b], t[b]
        lam = [i for i in range(n) if zL[i] != 0]
        S = [i for i in range(n) if zs[i] != 0]
        LS = [i for i in lam if i in S]
        miss = [i for i in S if i not in lam]
        false = [i for i in lam if i not in S]
        r = {"n_lambda": len(lam), "n_miss": len(miss), "n_false": len(false),
             "zstar_sq": sum(v * v for v in zs),
             "err_total": sum((z[i] - zs[i]) ** 2 for i in range(n)),
             "E_amp": sum((z[i] - zs[i]) ** 2 for i in LS),
             "E_miss": sum(zs[i] ** 2 for i in miss),
             "E_false": sum(z[i] ** 2 for i in false),
             "before_LS": sum((zL[i] - zs[i]) ** 2 for i in LS)}
        if lam:
            e = np.array([zL[i] - zs[i] for i in lam])
            T1 = (np.eye(len(lam)) - eta * G[np.ix_(lam, lam)]) @ e
            T2 = eta * (D - G)[np.ix_(lam, lam)] @ zs[lam]
            T3 = (eta * D[np.ix_(lam, miss)] @ zs[miss]) if miss else np.zeros(len(lam))
            lhs = z[lam] - zs[lam]
            r.update(T1_sq=float(T1 @ T1), T2_sq=float(T2 @ T2), T3_sq=float(T3 @ T3),
                     T12_sq=float((T1 + T2) @ (T1 + T2)),
                     err_lambda_sq=float(lhs @ lhs), zstar_lambda_sq=float(zs[lam] @ zs[lam]),
                     s14_resid=float(np.abs(T1 + T2 + T3 - lhs).max()))
            den = np.linalg.norm(D[np.ix_(lam, lam)], 2)
            r["dg_lambda_ratio"] = (np.linalg.norm((D - G)[np.ix_(lam, lam)], 2) / den
                                    if den > 0 else float("nan"))
        else:
            r.update(T1_sq=0.0, T2_sq=0.0, T3_sq=0.0, T12_sq=0.0, err_lambda_sq=0.0, zstar_lambda_sq=0.0,
                     s14_resid=0.0, dg_lambda_ratio=float("nan"))
        # reference-case condition
        triples, ok_S, ok_notS, all_ok = 0, 0, 0, True
        for i in range(n):
            coord_ok = True
            for k in range(L1):
                prev = 0.0 if k == 0 else tr["codes"][k - 1][b, i]
                w = tr["currents"][k][b, i] - (zs[i] - prev)
                ok = abs(w) < theta[k] - 0.5
                triples += ok
                coord_ok &= ok
            if coord_ok:
                ok_S += i in S
                ok_notS += i not in S
            all_ok &= coord_ok
        r.update(lemma_triples_ok=triples, lemma_coord_ok_S=ok_S, lemma_coord_ok_notS=ok_notS,
                 lemma_sample_ok=int(all_ok))
        # lost coordinates
        ever = [i for i in range(n) if any(tr["codes"][k][b, i] != 0 for k in range(L1))]
        r["lost_S"] = sum(1 for i in ever if zL[i] == 0 and i in S)
        r["lost_notS"] = sum(1 for i in ever if zL[i] == 0 and i not in S)
        # energy: energy_spec.slista per sample, E_0 from the closed form, F_l
        layers = [{"xi0": float(np.count_nonzero(tr["spikes"][k][b])),
                   "z0": float(np.count_nonzero(tr["codes"][k][b])),
                   "z1": float(np.abs(tr["codes"][k][b]).sum())} for k in range(L1)]
        mnz = float(np.count_nonzero(zL))
        r["E_spec_pJ"] = es.slista(m, n, layers, mnz, es.SPEC).total(es.SPEC)
        Fl = [sum(1 for i in range(n) if tr["codes"][k][b, i] != 0 and i not in S)
              for k in range(L1)]
        for k in range(L1):
            r[f"F_{k + 1}"] = Fl[k]
        r["F"] = sum(Fl)
        r["E_minus_E0_pJ"] = r["E_spec_pJ"] - const["E0_formula"]
        r["B_pJ"] = const["C_L"] * len(S) + const["C_L_prime"] * r["F"]
        # first-layer bound, Eq. (S20)
        bound = 0.0
        for i in range(n):
            if i in S:
                continue
            w2 = sum(D[i, j] ** 2 * zs[j] ** 2 for j in S)
            if w2 > 0:
                bound += math.exp(-theta[0] ** 2 / (2 * w2))
        r["first_layer_bound"] = 2 * bound
        rows.append(r)
    return rows


def direct_persistence(tr, t):
    L1 = len(tr["codes"])
    hist = [0] * (L1 + 1)
    for b in range(t.shape[0]):
        for i in range(t.shape[1]):
            if t[b, i] == 0:
                hist[sum(1 for k in range(L1) if tr["codes"][k][b, i] != 0)] += 1
    return hist


# ------------------------------------------------------------- random networks
def random_network(rng, grid):
    n = int(rng.integers(4, 9))
    m = int(rng.integers(3, 7))
    depth = int(rng.integers(2, 6))
    s = int(rng.integers(1, n))
    q = lambda shape, lo, hi: (rng.integers(lo, hi + 1, size=shape) * 0.25)
    if grid:
        A = q((m, n), -4, 4)
        P = q((n, m), -4, 4)
        G = [q((n, n), -4, 4) for _ in range(depth - 1)]
        theta = np.full(depth - 1, 0.25 * int(rng.integers(1, 7)))     # uniform, on the grid
        amp = lambda size: q(size, 1, 12)
    else:
        A = rng.standard_normal((m, n)) / math.sqrt(m)
        P = rng.standard_normal((n, m))
        G = [rng.standard_normal((n, n)) * 0.5 for _ in range(depth - 1)]
        theta = rng.uniform(0.2, 1.5, size=depth - 1)
        amp = lambda size: rng.uniform(0.0, 3.0, size=size)
    t = np.zeros((SIGNALS, n))
    for b in range(SIGNALS):
        idx = rng.choice(n, size=s, replace=False)
        t[b, idx] = rng.choice([-1.0, 1.0], size=s) * amp(s)
        t[b, idx[t[b, idx] == 0]] = 0.25            # keep exactly s nonzeros
    return {"A": A, "P": P, "G": G, "theta": theta, "eta": 0.5 if grid else rng.uniform(0.1, 1),
            "t": t, "n": n, "m": m, "depth": depth, "s": s, "grid": grid}


def max_diff(a, b):
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


def torch_trace(net, theta, eta):
    """models.SpikingLISTA in float64 with the given weights; returns trace and the
    theta / eta the model actually uses."""
    import torch
    from models import SpikingLISTA, inverse_softplus
    A = torch.tensor(net["A"], dtype=torch.float64)
    grid = net["grid"]
    floor = float(theta[0]) if grid else 1e-2
    model = SpikingLISTA(A, net["depth"], threshold_floor=floor).double()
    with torch.no_grad():
        model.P.weight.copy_(torch.tensor(net["P"]))
        for g, w in zip(model.G, net["G"]):
            g.weight.copy_(torch.tensor(w))
        if grid:      # softplus(-1e4) = 0 exactly: theta = floor, exactly on the grid
            model.raw_theta.fill_(-1e4)
        else:
            model.raw_theta.copy_(torch.tensor([inverse_softplus(v - floor) for v in theta]))
        model.raw_eta.fill_(inverse_softplus(eta))
        y = torch.tensor(net["t"] @ net["A"].T)
        tr = model.trace(y)
        th = model.theta.numpy().copy()
        et = float(model.eta)
    tr = {k: ([x.numpy() for x in v] if isinstance(v, list) else v.numpy()) for k, v in tr.items()}
    return tr, th, et, model


def item1(use_torch):
    rng = np.random.default_rng(20261110)
    worst = {}
    failures, ties, torch_rows = [], 0, []

    def track(name, value):
        worst[name] = max(worst.get(name, 0.0), float(value))

    for index in range(NETWORKS):
        net = random_network(rng, grid=index % 2 == 0)
        theta, eta = net["theta"], net["eta"]
        if use_torch:
            ttr, theta, eta, model = torch_trace(net, theta, eta)
            if net["grid"]:
                assert np.all(theta == net["theta"]), "grid thresholds not exact"
        y = net["t"] @ net["A"].T
        D = net["P"] @ net["A"]
        tr = sa.forward(net["P"], net["G"], theta, eta, y)
        ref = direct_forward(net["P"], net["G"], theta, eta, y)
        ties += int(sum((np.abs(c) == th).sum() for c, th in zip(tr["currents"][:-1], theta)))
        spikes_equal = all(np.array_equal(a, b) for a, b in zip(tr["spikes"], ref["spikes"]))
        track("forward_currents_vs_direct", max(max_diff(a, b) for a, b in
                                                 zip(tr["currents"], ref["currents"])))
        track("forward_output_vs_direct", max_diff(tr["output"], ref["output"]))
        if use_torch:
            from sn3_figs import code_checks
            same = all(np.array_equal(a, b) for a, b in zip(tr["spikes"], ttr["spikes"]))
            track("forward_currents_vs_torch_trace",
                  max(max_diff(a, b) for a, b in zip(tr["currents"], ttr["currents"])))
            track("forward_output_vs_torch_trace", max_diff(tr["output"], ttr["output"]))
            torch_rows.append(same)
            if not code_checks(model)["all_passed"]:
                failures.append(f"net {index}: code_checks failed on the small model")
        if not spikes_equal:
            failures.append(f"net {index}: spikes differ from the direct forward")

        got = sa.error_split(tr, net["t"], D, net["G"][-1], eta)
        lemma, _ = sa.reference_condition(tr, net["t"], theta)
        got.update(lemma)
        got.update(sa.lost_coordinates(tr, net["t"]))
        got["dg_lambda_ratio"] = sa.restricted_ratio(tr, D, net["G"][-1])
        got.update(sa.energy_terms(tr, net["t"], net["m"], net["n"]))
        got["first_layer_bound"] = sa.first_layer_bound(net["t"], D, theta[0])
        rows = direct_terms(tr, net["t"], D, net["G"], eta, theta, net["m"])
        for key in rows[0]:
            want = np.array([r[key] for r in rows], dtype=np.float64)
            have = np.asarray(got[key], dtype=np.float64)
            both_nan = np.isnan(want) & np.isnan(have)
            diff = np.where(both_nan, 0.0, np.abs(want - have))
            scale = np.maximum(1.0, np.abs(want))
            track(f"{key}", np.nanmax(diff / scale) if diff.size else 0.0)
        track("s14_identity_resid", got["s14_resid"].max())
        track("s15_identity_resid_rel", (got["s15_resid"] / np.maximum(got["err_total"], 1e-300)).max())
        const = sa.energy_constants(net["m"], net["n"], net["depth"])
        track("E0_rel_dev_from_formula", np.abs(got["E0_pJ"] - const["E0_formula"]).max()
              / const["E0_formula"])
        if (got["E_minus_E0_pJ"] > got["B_pJ"] * (1 + 1e-12)).any():
            failures.append(f"net {index}: S18 violated")
        if sa.persistence_counts(tr, net["t"]).tolist() != direct_persistence(tr, net["t"]):
            failures.append(f"net {index}: persistence histogram differs")

    limits = {k: TOL for k in worst}
    limits["s14_identity_resid"] = 1e-10
    bad = {k: v for k, v in worst.items() if v > limits[k]}
    passed = not failures and not bad and (not use_torch or all(torch_rows))
    return {"networks": NETWORKS, "signals_per_network": SIGNALS,
            "threshold_ties_hit_exactly": ties, "torch_compared": use_torch,
            "torch_spikes_identical_networks": sum(torch_rows) if use_torch else None,
            "max_relative_differences": worst, "over_tolerance": bad,
            "failures": failures, "passed": passed}


# -------------------------------------------------------------- real models
def item2_3(root, device_name):
    import torch
    import sn3_figs as sf
    from config import CONDITIONS
    device = torch.device(device_name)
    out = {}
    for condition in CONDITIONS:
        model, matrix, cfg = sf.load_condition(root, condition, device)
        checks = sf.code_checks(model)
        prm = sf.parameters64(model, matrix)
        col, ext = sf.run_condition(model, matrix, cfg, condition, prm, limit=1000)
        const = sa.energy_constants(model.m, model.n, model.depth)
        # sn3_analysis.forward vs SpikingLISTA.trace, both float64, on real signals
        from data import generator, sparse_batch
        import copy
        with torch.no_grad():
            target, _ = sparse_batch(200, condition["s"], matrix, generator(cfg.test_seed, device),
                                     cfg.amplitude_min, cfg.amplitude_max)
            m64 = copy.deepcopy(model).double()
            t = target.double().cpu().numpy()
            y64 = t @ prm["A"].T
            ttr = m64.trace(torch.tensor(y64, device=device))
            tr = sa.forward(prm["P"], prm["G"], prm["theta"], prm["eta"], y64)
            spikes_same = all(np.array_equal(a.cpu().numpy(), b)
                              for a, b in zip(ttr["spikes"], tr["spikes"]))
            out_diff = max_diff(ttr["output"].cpu().numpy(), tr["output"])
        e0 = col["E0_pJ"]
        r2 = {"code_checks": checks,
              "spike_mismatch_f32_vs_f64_elements": int(col["spike_mismatch"].sum()),
              "forward_vs_trace_float64_spikes_identical": spikes_same,
              "forward_vs_trace_float64_max_output_diff": out_diff,
              "s14_max_abs_resid": float(col["s14_resid"].max()),
              "s15_max_rel_resid": float((col["s15_resid"]
                                          / np.maximum(col["err_total"], 1e-300)).max()),
              "E0_rel_std": float(e0.std() / e0.mean()),
              "E0_mean_uj": float(e0.mean()) * 1e-6,
              "E0_formula_uj": const["E0_formula"] * 1e-6,
              "C_L_nj": const["C_L"] * 1e-3, "C_L_prime_nj": const["C_L_prime"] * 1e-3,
              "s18_violations": int((col["E_minus_E0_pJ"] > col["B_pJ"] * (1 + 1e-12)).sum())}
        r2["passed"] = (checks["all_passed"] and spikes_same and out_diff < 1e-9
                        and r2["s14_max_abs_resid"] < 1e-10 and r2["s15_max_rel_resid"] < 1e-12
                        and r2["E0_rel_std"] < 1e-12
                        and abs(r2["E0_mean_uj"] - r2["E0_formula_uj"]) < 1e-12 * r2["E0_formula_uj"]
                        and r2["s18_violations"] == 0)
        # mismatches are reported, not a failure (spec: float64 is authoritative)
        # item 3: float32 NMSE on the whole test set vs evaluate.py
        err = energy = 0.0
        rng = generator(cfg.test_seed, device)
        remaining = cfg.test_samples
        with torch.no_grad():
            while remaining:
                size = min(cfg.eval_batch_size, remaining)
                target, y = sparse_batch(size, condition["s"], matrix, rng,
                                         cfg.amplitude_min, cfg.amplitude_max)
                est, _ = model(y)
                err += float((est - target).square().sum())
                energy += float(target.square().sum())
                remaining -= size
        nmse = 10 * math.log10(err / energy)
        ref = sf.reference_nmse(root, condition["id"])
        r3 = {"nmse_db": nmse, "evaluate_py_nmse_db": ref,
              "passed": ref is not None and abs(nmse - ref) < 1e-6}
        out[condition["id"]] = {"item2": r2, "item3": r3}
        print(f"{condition['id']}: item2 {'PASS' if r2['passed'] else 'FAIL'} "
              f"(mismatches {r2['spike_mismatch_f32_vs_f64_elements']}, "
              f"S14 {r2['s14_max_abs_resid']:.1e}, E0 relstd {r2['E0_rel_std']:.1e}, "
              f"E0 {r2['E0_mean_uj']:.6f} uJ, S18 viol {r2['s18_violations']}) | "
              f"item3 {'PASS' if r3['passed'] else 'FAIL'} ({nmse:.6f} vs {ref})", flush=True)
    return out


def main(args):
    use_torch = True
    try:
        import torch  # noqa: F401
    except ImportError:
        if not args.skip_real:
            raise
        use_torch = False
    result = {"item1": item1(use_torch)}
    print("item1:", json.dumps(result["item1"], indent=2), flush=True)
    if not args.skip_real:
        result["items2_3"] = item2_3(Path(args.output).resolve(), args.device)
    result["all_passed"] = result["item1"]["passed"] and all(
        v["item2"]["passed"] and v["item3"]["passed"] for v in result.get("items2_3", {}).values())
    path = Path(args.output) / "sn3_selftest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, default=float) + "\n")
    print("ALL PASSED" if result["all_passed"] else "SOME CHECKS FAILED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default="runs/final")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-real", action="store_true")
    main(parser.parse_args())
