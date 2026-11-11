"""Supplementary Figs. 1 and 2 of Supplementary Note 3 (synthetic task).

  CUDA_VISIBLE_DEVICES=0 nohup python -u sn3_figs.py --output runs/final \
      --device cuda:0 > runs/final/sn3_figs.log 2>&1 &

Evaluation only:
  * models: S-LISTA, L = 10, s = 14/28/42/56, training seed 42
    (<output>/seed42/checkpoints/m141_s*__slista_l10.pth), noise-free;
  * signals: the 10,000 test signals of evaluate.py (test_seed, batches of
    eval_batch_size, same device), so every number refers to the same test set;
  * the float32 forward (model.trace) is run as in evaluate.py; every analysed
    quantity is recomputed in float64 (sn3_analysis.forward, y = A z* in float64)
    and the spikes of both runs are compared bit by bit;
  * Supp. Fig. 1: Eqs. (S14), (S15), Reference-case condition (Section 3.2), ||D - G||_2 / ||D||_2,
    amplitudes of missed / detected coefficients, coordinates lost by the end;
  * Supp. Fig. 2: self-check of Eq. (S17) against energy_spec.slista,
    C_L and C'_L, Eq. (S18) violations, F = sum_l |supp(z^(l)) \\ S|,
    bound of Eq. (S20), persistence of false detections.

Outputs (the directory is emptied first): <output>/sn3_figs/
  summary.json                    every number quoted in Supplementary Note 3
  per_sample_s{14,28,42,56}.csv   one row per test signal
  supp_fig1.{pdf,png}, supp_fig2.{pdf,png}   (sn3_plot.py; can be redrawn alone)
No existing file or result is modified.
"""

import argparse
import csv
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as tF

import energy_spec as es
import sn3_analysis as sa
import sn3_plot
from config import CONDITIONS, SPARSITY_SWEEP_DEPTH
from data import generator, sparse_batch
from models import SurrogateSpike
from utils import checkpoint_path, load_model, save_json

SEED = 42
MODEL_ID = f"slista_l{SPARSITY_SWEEP_DEPTH}"
OUT_NAME = "sn3_figs"
S14_TOL = 1e-10
E0_REL_STD_TOL = 1e-12


# --------------------------------------------------------------- code checks
@torch.no_grad()
def code_checks(model):
    """The three conventions SN3 relies on."""
    out = {}
    x = torch.tensor([0.75, -0.75, 0.7499999, 0.0], dtype=torch.float64)
    t = torch.tensor(0.75, dtype=torch.float64)
    w = 0.4 * t
    spikes = SurrogateSpike.apply(x, t, w) - SurrogateSpike.apply(-x, t, w)
    out["quantizer_fires_at_equality"] = spikes.tolist() == [1.0, -1.0, 0.0, 0.0]
    dev, dtype = model.P.weight.device, model.P.weight.dtype
    rng = torch.Generator(device="cpu").manual_seed(0)
    z = torch.randn(4, model.n, generator=rng).to(dev, dtype)
    yy = torch.randn(4, model.m, generator=rng).to(dev, dtype)
    out["linear_G_is_weight"] = all(
        torch.allclose(g(z), z @ g.weight.T, rtol=1e-5, atol=1e-6) for g in model.G)
    out["linear_P_is_weight"] = bool(torch.allclose(model.P(yy), yy @ model.P.weight.T,
                                                    rtol=1e-5, atol=1e-6))
    out["theta_is_floor_plus_softplus"] = bool(torch.allclose(
        model.theta, model.threshold_floor + tF.softplus(model.raw_theta)))
    out["eta_is_softplus"] = bool(torch.allclose(model.eta, tF.softplus(model.raw_eta)))
    out["all_passed"] = all(out.values())
    return out


def parameters64(model, matrix):
    """Weights, thresholds and eta as float64 numpy arrays (floor + softplus in float64)."""
    raw_theta = model.raw_theta.detach().double()
    theta = (model.threshold_floor.detach().double() + tF.softplus(raw_theta)).cpu().numpy()
    eta = float(tF.softplus(model.raw_eta.detach().double()))
    P = model.P.weight.detach().double().cpu().numpy()
    G = [g.weight.detach().double().cpu().numpy() for g in model.G]
    A = matrix.detach().double().cpu().numpy()
    return {"P": P, "G": G, "theta": theta, "eta": eta, "A": A, "D": P @ A}


# ------------------------------------------------------------ one condition
@torch.no_grad()
def run_condition(model, matrix, cfg, condition, prm, limit=None):
    """Analyse the first `limit` (default: all) test signals; returns per-sample
    columns (dict of arrays) and the aggregate extras."""
    s = condition["s"]
    total = cfg.test_samples if limit is None else int(limit)
    if total % cfg.eval_batch_size and total != cfg.test_samples:
        raise ValueError("limit must be a multiple of eval_batch_size")
    rng = generator(cfg.test_seed, matrix.device)
    P, G, theta, eta, A, D = (prm[k] for k in ("P", "G", "theta", "eta", "A", "D"))
    columns = {}
    extras = {"err32": 0.0, "energy32": 0.0, "lemma_layer_ok": np.zeros(len(G), np.int64),
              "persistence": np.zeros(len(G) + 1, np.int64), "amp_missed": [],
              "amp_detected": [], "max_out_diff_f32_f64": 0.0}
    done = 0
    while done < total:
        size = min(cfg.eval_batch_size, total - done)
        target, y32 = sparse_batch(size, s, matrix, rng, cfg.amplitude_min, cfg.amplitude_max)
        tr32 = model.trace(y32)
        extras["err32"] += float((tr32["output"].double() - target.double()).square().sum())
        extras["energy32"] += float(target.double().square().sum())

        t = target.double().cpu().numpy()
        tr = sa.forward(P, G, theta, eta, t @ A.T)
        mismatch = sum((tr32["spikes"][l].double().cpu().numpy() != tr["spikes"][l]).sum(1)
                       for l in range(len(G)))
        extras["max_out_diff_f32_f64"] = max(
            extras["max_out_diff_f32_f64"],
            float(np.abs(tr32["output"].double().cpu().numpy() - tr["output"]).max()))

        batch = {"sample": np.arange(done, done + size), "spike_mismatch": mismatch}
        batch.update(sa.error_split(tr, t, D, G[-1], eta))
        lemma, per_layer = sa.reference_condition(tr, t, theta)
        batch.update(lemma)
        extras["lemma_layer_ok"] += np.asarray(per_layer, dtype=np.int64)
        batch.update(sa.lost_coordinates(tr, t))
        batch["dg_lambda_ratio"] = sa.restricted_ratio(tr, D, G[-1])
        batch.update(sa.energy_terms(tr, t, model.m, model.n))
        batch["first_layer_bound"] = sa.first_layer_bound(t, D, theta[0])
        extras["persistence"] += sa.persistence_counts(tr, t)
        lam = tr["spike_code"] != 0
        sup = t != 0
        extras["amp_missed"].append(np.abs(t[sup & ~lam]))
        extras["amp_detected"].append(np.abs(t[sup & lam]))
        for key, value in batch.items():
            columns.setdefault(key, []).append(np.asarray(value))
        done += size
    columns = {k: np.concatenate(v) for k, v in columns.items()}
    extras["amp_missed"] = np.concatenate(extras["amp_missed"])
    extras["amp_detected"] = np.concatenate(extras["amp_detected"])
    return columns, extras


# ----------------------------------------------------------------- summaries
def reference_nmse(root, condition_id):
    """Seed-42 NMSE from evaluate.py (read only), to confirm the same test set."""
    path = root / "metrics" / "results_per_seed.csv"
    if not path.exists():
        return None
    for r in csv.DictReader(open(path)):
        if int(r["seed"]) == SEED and r["condition_id"] == condition_id and r["id"] == MODEL_ID:
            return float(r["nmse_db"])
    return None


def reference_energy(root, condition_id):
    """Seed-42 S-LISTA total energy (uJ) from energy_eval.py (read only)."""
    path = root / "energy" / "energy_per_seed.csv"
    if not path.exists():
        return None
    for r in csv.DictReader(open(path)):
        if int(r["seed"]) == SEED and r["condition_id"] == condition_id and r["id"] == MODEL_ID:
            return float(r["total_uj"])
    return None


def summarise(col, ext, prm, s, const, root, condition_id):
    N = col["sample"].size
    n = prm["D"].shape[0]
    L1 = len(prm["G"])
    sums = {k: float(col[k].sum()) for k in (
        "zstar_sq", "err_total", "E_amp", "E_miss", "E_false", "before_LS",
        "T1_sq", "T2_sq", "T3_sq", "T12_sq", "err_lambda_sq", "zstar_lambda_sq")}
    tot, zs, zl = sums["err_total"], sums["zstar_sq"], sums["zstar_lambda_sq"]
    theta = [float(v) for v in prm["theta"]]
    norm_D = np.linalg.norm(prm["D"], 2)
    nmse_ref = reference_nmse(root, condition_id)
    nmse32 = sa.db(ext["err32"], ext["energy32"])
    fig1 = {
        "sums": sums,
        "pct_amp": 100 * sums["E_amp"] / tot, "pct_miss": 100 * sums["E_miss"] / tot,
        "pct_false": 100 * sums["E_false"] / tot,
        "nmse_db": sa.db(tot, zs), "nmse_db_float32": nmse32,
        "nmse_db_evaluate_py": nmse_ref,
        "nmse_db_float32_minus_evaluate_py": None if nmse_ref is None else nmse32 - nmse_ref,
        "before_db": sa.db(sums["before_LS"], zs), "after_db": sa.db(sums["E_amp"], zs),
        "reduction_db": sa.db(sums["before_LS"], zs) - sa.db(sums["E_amp"], zs),
        "T1_db": sa.db(sums["T1_sq"], zl), "T2_db": sa.db(sums["T2_sq"], zl),
        "T3_db": sa.db(sums["T3_sq"], zl),
        "T12_db": sa.db(sums["T12_sq"], zl), "err_lambda_db": sa.db(sums["err_lambda_sq"], zl),
        "s15_max_abs_resid": float(col["s15_resid"].max()),
        "s15_max_rel_resid": float((col["s15_resid"] / np.maximum(col["err_total"], 1e-300)).max()),
        "s14_max_abs_resid": float(col["s14_resid"].max()),
        "s14_passed": bool(col["s14_resid"].max() < S14_TOL),
        "mean_n_lambda": float(col["n_lambda"].mean()),
        "mean_n_miss": float(col["n_miss"].mean()),
        "mean_n_false": float(col["n_false"].mean()),
        "reference_condition": {
            "triples": sa.clopper_pearson(col["lemma_triples_ok"].sum(), N * n * L1),
            "coords_in_S_all_layers": sa.clopper_pearson(col["lemma_coord_ok_S"].sum(), N * s),
            "coords_not_in_S_all_layers": sa.clopper_pearson(col["lemma_coord_ok_notS"].sum(),
                                                            N * (n - s)),
            "samples_all": sa.clopper_pearson(col["lemma_sample_ok"].sum(), N),
            "per_layer_fraction": [float(k) / (N * n) for k in ext["lemma_layer_ok"]],
            "layers_with_theta_le_half": [k + 1 for k, v in enumerate(theta) if v <= 0.5],
        },
        "theta": theta, "eta": prm["eta"],
        "dg_ratio_full": [float(np.linalg.norm(prm["D"] - g, 2) / norm_D) for g in prm["G"]],
        "dg_ratio_lambda": sa.describe(col["dg_lambda_ratio"]),
        "amplitude_missed": sa.quantiles(ext["amp_missed"]),
        "amplitude_detected": sa.quantiles(ext["amp_detected"]),
        "lost_S_mean": float(col["lost_S"].mean()),
        "lost_notS_mean": float(col["lost_notS"].mean()),
    }
    ratio = col["E_minus_E0_pJ"] / col["B_pJ"]
    tol = 1e-12 * col["B_pJ"]
    bound_mean, f1_mean = float(col["first_layer_bound"].mean()), float(col["F_1"].mean())
    e_ref = reference_energy(root, condition_id)
    fig2 = {
        "ratio_code_energy_to_bound": sa.describe(ratio),
        "max_ratio": float(ratio.max()),
        "s18_violations": int((col["E_minus_E0_pJ"] > col["B_pJ"] + tol).sum()),
        "F_over_s": sa.describe(col["F"] / s),
        "F_mean": float(col["F"].mean()),
        "F_layer_mean": [float(col[f"F_{l + 1}"].mean()) for l in range(L1)],
        "false_term_share_of_bound": sa.describe(const["C_L_prime"] * col["F"] / col["B_pJ"]),
        "first_layer": {"mean_bound": bound_mean, "mean_F1": f1_mean,
                  "mean_F1_over_mean_bound": f1_mean / bound_mean if bound_mean > 0 else None,
                  "fraction_samples_F1_gt_bound": float((col["F_1"] > col["first_layer_bound"]).mean())},
        "persistence_hist_layers_0_to_Lm1": [int(v) for v in ext["persistence"]],
        "E_spec_mean_uj": float(col["E_spec_pJ"].mean()) * 1e-6,
        "E_spec_mean_uj_energy_eval_py": e_ref,
        "E_minus_E0_mean_nj": float(col["E_minus_E0_pJ"].mean()) * 1e-3,
        "B_mean_nj": float(col["B_pJ"].mean()) * 1e-3,
    }
    flt = {"spike_mismatch_elements": int(col["spike_mismatch"].sum()),
           "samples_with_mismatch": int((col["spike_mismatch"] > 0).sum()),
           "max_abs_output_diff_f32_f64": ext["max_out_diff_f32_f64"]}
    return {"s": s, "samples": N, "fig1": fig1, "fig2": fig2, "float_check": flt}


def clean_json(value):
    """numpy scalars -> Python; non-finite floats -> strings (save_json forbids NaN)."""
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else str(float(value))
    return value


def write_columns(columns, s, path):
    keys = list(columns)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["s"] + keys)
        for i in range(columns["sample"].size):
            writer.writerow([s] + [format(float(columns[k][i]), ".12g") if columns[k].dtype.kind == "f"
                                   else int(columns[k][i]) for k in keys])


def e0_check(all_e0, const):
    e0 = np.concatenate(all_e0)
    mean = float(e0.mean())
    rel_std = float(e0.std() / abs(mean))
    rel_formula = abs(mean - const["E0_formula"]) / const["E0_formula"]
    return {"E0_mean_uj": mean * 1e-6, "E0_formula_uj": const["E0_formula"] * 1e-6,
            "E0_rel_std": rel_std, "E0_max_abs_dev_pJ": float(np.abs(e0 - mean).max()),
            "E0_mean_rel_err_vs_formula": rel_formula, "samples": int(e0.size),
            "passed": bool(rel_std < E0_REL_STD_TOL and rel_formula < 1e-12)}


# ---------------------------------------------------------------------- main
def load_condition(root, condition, device):
    path = checkpoint_path(root, SEED, condition["id"], MODEL_ID)
    model, matrix, job, cfg = load_model(path, device)
    if job.get("seed") != SEED or job["id"] != MODEL_ID or job["depth"] != SPARSITY_SWEEP_DEPTH:
        raise ValueError(f"Unexpected checkpoint {path}: {job.get('seed')} {job['id']}")
    return model, matrix, cfg


def main(args):
    root = Path(args.output).resolve()
    out = root / OUT_NAME
    if args.plot_only:
        sn3_plot.make_figures(out)
        return
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    device = torch.device(args.device)
    summary = {"settings": {"seed": SEED, "model_id": MODEL_ID, "device": str(device),
                            "conditions": [c["id"] for c in CONDITIONS], "noise": "none"},
               "per_s": {}}
    all_e0, const = [], None
    start = time.time()
    for condition in CONDITIONS:
        model, matrix, cfg = load_condition(root, condition, device)
        checks = code_checks(model)
        summary.setdefault("code_checks", {})[condition["id"]] = checks
        if not checks["all_passed"]:
            save_json(out / "summary.json", clean_json(summary))
            raise SystemExit(f"Code check failed for {condition['id']}: {checks}")
        prm = parameters64(model, matrix)
        const = sa.energy_constants(model.m, model.n, model.depth)
        summary["constants"] = {**const, "C_L_nj": const["C_L"] * 1e-3,
                                "C_L_prime_nj": const["C_L_prime"] * 1e-3,
                                "C_L_prime_over_C_L": const["C_L_prime"] / const["C_L"],
                                "m": model.m, "n": model.n, "depth": model.depth}
        columns, extras = run_condition(model, matrix, cfg, condition, prm)
        s = condition["s"]
        write_columns(columns, s, out / f"per_sample_s{s}.csv")
        row = summarise(columns, extras, prm, s, const, root, condition["id"])
        summary["per_s"][str(s)] = row
        all_e0.append(columns["E0_pJ"])
        f1, f2, fc = row["fig1"], row["fig2"], row["float_check"]
        print(f"[{time.time() - start:6.1f}s] s={s:2d} NMSE {f1['nmse_db']:.2f} dB "
              f"(float32 {f1['nmse_db_float32']:.4f}, evaluate.py {f1['nmse_db_evaluate_py']}) | "
              f"amp/miss/false {f1['pct_amp']:.1f}/{f1['pct_miss']:.1f}/{f1['pct_false']:.1f}% | "
              f"readout -{f1['reduction_db']:.2f} dB | S14 resid {f1['s14_max_abs_resid']:.1e} | "
              f"F/s {f2['F_over_s']['mean']:.3f} | (E-E0)/B med {f2['ratio_code_energy_to_bound']['median']:.3f} "
              f"| S18 viol {f2['s18_violations']} | spike mismatches {fc['spike_mismatch_elements']}",
              flush=True)
    summary["fig2_selfcheck"] = e0_check(all_e0, const)
    summary["fig2_selfcheck"]["C_L_nj"] = const["C_L"] * 1e-3
    summary["fig2_selfcheck"]["C_L_prime_nj"] = const["C_L_prime"] * 1e-3
    summary["all_checks"] = {
        "code_checks": all(v["all_passed"] for v in summary["code_checks"].values()),
        "s14_residual_below_1e-10": all(r["fig1"]["s14_passed"] for r in summary["per_s"].values()),
        "s15_relative_residual_below_1e-12": all(r["fig1"]["s15_max_rel_resid"] < 1e-12
                                                 for r in summary["per_s"].values()),
        "E0_constant": summary["fig2_selfcheck"]["passed"],
        "s18_no_violation": all(r["fig2"]["s18_violations"] == 0 for r in summary["per_s"].values()),
        "no_spike_mismatch": all(r["float_check"]["spike_mismatch_elements"] == 0
                                 for r in summary["per_s"].values()),
    }
    save_json(out / "summary.json", clean_json(summary))
    print(json.dumps({"fig2_selfcheck": summary["fig2_selfcheck"],
                      "all_checks": summary["all_checks"]}, indent=2), flush=True)
    sn3_plot.make_figures(out)
    print(f"Wrote {out} in {time.time() - start:.1f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default="runs/final")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--plot-only", action="store_true",
                        help="redraw the figures from summary.json and the per-sample csv files")
    main(parser.parse_args())
