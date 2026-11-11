"""Supplementary Note 3, Supplementary Fig. 3 (SHD): propagation of channel errors.

Evaluation only: the seed-42 ``slista_dense`` model is
loaded, nothing is trained and no existing file is changed.

A. r_star (Eqs. S24, S29; reference run only, channel independent).
B. Monte Carlo under AWGN and frozen Rayleigh fading at the 15 SNRs of Fig. 4,
   with exactly the noise / fading realisations of the formal evaluation
   (checked by reproducing the formal temporal-mean NMSE of seed 42).

All analysis quantities are recomputed in float64 (Algorithm S1 re-implemented
below); the float32 forward of the model is used only to reproduce the formal
NMSE and to count float32/float64 spike mismatches.

Usage:
    python sn3_fig3_selftest.py --gpu 0          # first
    nohup python -u sn3_fig3.py --gpu 0 > sn3_fig3.log 2>&1 &
    python sn3_fig3_plot.py
Outputs: outputs/sn3_fig3/ (cleared at start).  --smoke: first batch, two SNRs,
outputs/sn3_fig3_smoke/.
"""

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "sn3_fig3"
SNR_GRID = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 7.5, 10, 15, 20]
SEED = 42
KEY = "slista_dense"
INF = 10 ** 9
CHUNK = 64
CLASS_NAMES = ["i_unchanged", "ii_spikes_changed_d0", "iii_same_support", "iv_support_changed"]


# ======================================================================
# Pure math (no project imports; used by the selftest on small networks)
# ======================================================================
def quantize(u, theta):
    """Ternary quantizer Q_theta (Eq. S11): H(u - theta) - H(-u - theta), H(0) = 1."""
    return (u >= theta).to(u.dtype) - ((-u) >= theta).to(u.dtype)


def cp_interval(k, n):
    """Clopper-Pearson 95% interval for k successes in n trials."""
    from scipy.stats import beta
    k, n = int(k), int(n)
    if n == 0:
        return float("nan"), float("nan")
    lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
    return lo, hi


class Params:
    """S-LISTA parameters in one dtype.  G[l] acts on z^(l+1):
    I^(l+2) = I^(1) - G[l] z^(l+1); the readout uses G[-1]."""

    def __init__(self, P, G, theta, eta, rho, D=None):
        self.P, self.G, self.theta = P, list(G), theta
        self.eta, self.rho, self.D = float(eta), float(rho), D
        self.layers = len(self.G)


def params_from_model(model, dtype=torch.float64):
    with torch.no_grad():
        return Params(
            model.P.weight.detach().to(dtype),
            [g.weight.detach().to(dtype) for g in model.G],
            model.vartheta.detach().to(dtype),
            float(model.eta.detach()), float(model.rho),
            model.synthesis_dict.weight.detach().to(dtype),
        )


def forward(params, y):
    """Algorithm S1 (multi-step, membrane memory rho, soft reset).

    y: [B, T, M].  Returns ubar [B,T,L-1,N] (pre-spike membranes), xi int8
    [B,T,L-1,N], code z^(L-1) [B,T,N], IL = I^(L) [B,T,N], z (readout) [B,T,N].
    """
    y = y.to(params.P.dtype)
    B, T, _ = y.shape
    N, Lm = params.P.shape[0], params.layers
    states = y.new_zeros(Lm, B, N)
    ubar = y.new_zeros(B, T, Lm, N)
    xi = torch.zeros(B, T, Lm, N, dtype=torch.int8, device=y.device)
    codes, ils, zs = [], [], []
    for t in range(T):
        initial = y[:, t] @ params.P.T
        code = torch.zeros_like(initial)
        current = initial
        for layer in range(Lm):
            membrane = params.rho * states[layer] + current
            spike = quantize(membrane, params.theta[layer])
            states[layer] = membrane - params.theta[layer] * spike
            ubar[:, t, layer] = membrane
            xi[:, t, layer] = spike.to(torch.int8)
            code = code + spike
            current = initial - code @ params.G[layer].T
        mask = (code != 0).to(code.dtype)
        codes.append(code)
        ils.append(current)
        zs.append(code + params.eta * mask * current)
    return {"ubar": ubar, "xi": xi, "code": torch.stack(codes, 1),
            "IL": torch.stack(ils, 1), "z": torch.stack(zs, 1)}


def contributions(P, s_tx, rho):
    """c[k, i, q]: term that flipping bit q contributes to eps_{k,i} (Sec. 3.4).

    rho == 0: bits of frame k only, q = channel j, c = P_ij (1 - 2 s_kj).
    rho != 0: bits (k', j), k' <= k, q = k' M + j, c = rho^(k-k') P_ij (1 - 2 s_k'j);
    bits of later frames contribute 0 (and are not counted)."""
    T, M = s_tx.shape
    sign = 1.0 - 2.0 * s_tx.to(P.dtype)
    if rho == 0.0:
        return P[None, :, :] * sign[:, None, :]
    out = P.new_zeros(T, P.shape[0], T * M)
    for k in range(T):
        for kp in range(k + 1):
            out[k, :, kp * M:(kp + 1) * M] = (rho ** (k - kp)) * P * sign[kp][None, :]
    return out


def inside_J(ubar, xi, theta, b):
    """b in J_{k,i} (Eq. S24), decided with the quantizer itself so that open and
    closed endpoints are exact.  ubar, xi: [T, L-1, N]; b: [T, N, Q] -> bool."""
    ok = None
    for layer in range(ubar.shape[1]):
        same = quantize(ubar[:, layer, :, None] + b, theta[layer]) == xi[:, layer, :, None].to(b.dtype)
        ok = same if ok is None else ok & same
    return ok


def first_true(mask):
    """1 + index of the first True along the last axis; INF if none."""
    hit = mask.any(-1)
    index = torch.argmax(mask.to(torch.float32), dim=-1) + 1
    return torch.where(hit, index, torch.full_like(index, INF))


def r_tables(ubar, xi, theta, P, s_tx, rho):
    """Eq. (S29): r_plus, r_minus [T, N] (INF if none) for one sample, with the
    sorted prefix sums S+(q), S-(q) and the per-bit contributions."""
    c = contributions(P, s_tx, rho)
    pos_vals, pos_idx = torch.where(c > 0, c, torch.zeros_like(c)).sort(dim=-1, descending=True)
    neg_vals, neg_idx = torch.where(c < 0, c, torch.zeros_like(c)).sort(dim=-1)
    s_plus, s_minus = pos_vals.cumsum(-1), neg_vals.cumsum(-1)
    r_plus = first_true(~inside_J(ubar, xi, theta, s_plus))
    r_minus = first_true(~inside_J(ubar, xi, theta, s_minus))
    return {"r_plus": r_plus, "r_minus": r_minus, "s_plus": s_plus, "s_minus": s_minus,
            "pos_idx": pos_idx, "neg_idx": neg_idx, "c": c}


def margins(ubar, theta):
    """delta_{k,i} = min_l | |ubar| - theta_l | (Eq. S27); ubar [T, L-1, N] -> [T, N]."""
    return (ubar.abs() - theta[None, :, None]).abs().min(dim=1).values


def first_changed_layer(ubar_ki, xi_ki, theta, b):
    """1-based first layer l with Q(ubar + b) != xi for one neuron; 0 if none."""
    for layer in range(ubar_ki.shape[0]):
        if float(quantize(ubar_ki[layer] + b, theta[layer])) != float(xi_ki[layer]):
            return layer + 1
    return 0


def rstar_summary(ubar, xi, theta, P, s_tx, rho):
    """r_star (Eq. S29) of one sample and the pattern that attains it."""
    tables = r_tables(ubar, xi, theta, P, s_tx, rho)
    r = torch.minimum(tables["r_plus"], tables["r_minus"])
    r_star = int(r.min())
    result = {"r": r, "r_star": r_star, "frame": -1, "neuron": -1, "direction": "",
              "layer": 0, "bits": []}
    if r_star >= INF:
        return result
    k, i = [int(v) for v in (r == r_star).nonzero()[0]]
    plus = int(tables["r_plus"][k, i]) == r_star
    both = plus and int(tables["r_minus"][k, i]) == r_star
    sums = tables["s_plus"] if plus else tables["s_minus"]
    order = tables["pos_idx"] if plus else tables["neg_idx"]
    b = sums[k, i, r_star - 1]
    M = s_tx.shape[1]
    columns = [int(q) for q in order[k, i, :r_star]]
    bits = [(k, q) if rho == 0.0 else (q // M, q % M) for q in columns]
    result.update(frame=k, neuron=i, direction=("+-" if both else "+" if plus else "-"),
                  layer=first_changed_layer(ubar[k, :, i], xi[k, :, i], theta, b), bits=bits)
    return result


def predicted_changes(ref, eps, theta):
    """Theorem 4 test per frame: first layer l (1-based) at which
    Q(ubar + eps) != xi for some neuron; 0 if the frame is predicted unchanged."""
    first = torch.zeros(eps.shape[:2], dtype=torch.long, device=eps.device)
    for layer in range(ref["ubar"].shape[2]):
        changed = (quantize(ref["ubar"][:, :, layer] + eps, theta[layer])
                   != ref["xi"][:, :, layer].to(eps.dtype)).any(-1)
        first = torch.where((first == 0) & changed, torch.full_like(first, layer + 1), first)
    return first


def actual_changes(ref, pert):
    first = torch.zeros(ref["xi"].shape[:2], dtype=torch.long, device=ref["xi"].device)
    for layer in range(ref["xi"].shape[2]):
        changed = (ref["xi"][:, :, layer] != pert["xi"][:, :, layer]).any(-1)
        first = torch.where((first == 0) & changed, torch.full_like(first, layer + 1), first)
    return first


def analyze_frames(params, tx, rx, x=None):
    """Reference (tx) vs perturbed (rx) run in params' dtype.  Per-frame tensors
    [B, T]."""
    ref, pert = forward(params, tx), forward(params, rx)
    delta = (rx - tx).to(params.P.dtype)
    p_delta = delta @ params.P.T                                  # P Delta_t
    eps = torch.zeros_like(p_delta)
    for t in range(p_delta.shape[1]):                             # Eq. (S22)
        eps[:, t] = (params.rho * eps[:, t - 1] if t else 0.0) + p_delta[:, t]
    first_actual = actual_changes(ref, pert)
    first_pred = predicted_changes(ref, eps, params.theta)
    changed = first_actual > 0
    # With rho != 0 the eps model holds only up to the first change of a sample.
    valid = torch.ones_like(changed)
    if params.rho != 0.0:
        seen = torch.cumsum(changed.long(), dim=1)
        valid = (seen - changed.long()) == 0
    d = pert["code"] - ref["code"]
    lam, lam_p = ref["code"] != 0, pert["code"] != 0
    same_support = (lam == lam_p).all(-1)
    d_zero = (d == 0).all(-1)
    cls = torch.full(changed.shape, 3, dtype=torch.long, device=tx.device)
    cls[same_support] = 2
    cls[d_zero] = 1
    cls[~changed] = 0
    dz = pert["z"] - ref["z"]
    eta, G = params.eta, params.G[-1]
    mask = lam.to(dz.dtype)
    # Eq. (S26): frames (i), (ii).
    s26 = (dz - eta * mask * p_delta).abs().amax(-1)
    s26 = torch.where(cls <= 1, s26, torch.zeros_like(s26))
    # Eq. (S31): every frame, on Lambda cap Lambda'.
    rhs31 = d - eta * (d @ G.T) + eta * p_delta
    both = lam & lam_p
    s31 = torch.where(both, (dz - rhs31).abs(), torch.zeros_like(dz)).amax(-1)
    # Eq. (S32): frames (iii), on Lambda, with G restricted to Lambda x Lambda.
    d_lam = d * mask
    contracted = d_lam - eta * (d_lam @ G.T)                      # (I - eta G_LL) d_L on L
    rhs32 = contracted + eta * p_delta
    s32 = torch.where(lam, (dz - rhs32).abs(), torch.zeros_like(dz)).amax(-1)
    s32 = torch.where(cls == 2, s32, torch.zeros_like(s32))
    d_norm = (d_lam ** 2).sum(-1).sqrt()
    ratio = (torch.where(lam, contracted, torch.zeros_like(contracted)) ** 2).sum(-1).sqrt() / d_norm
    ratio = torch.where(cls == 2, ratio, torch.full_like(ratio, float("nan")))
    sq = dz ** 2
    out = {
        "bit_errors": delta.abs().sum(-1),
        "changed": changed, "cls": cls, "first_actual": first_actual, "first_pred": first_pred,
        "valid": valid, "s26": s26, "s31": s31, "s32": s32, "ratio": ratio,
        "out_energy": sq.sum(-1),
        "out_common": torch.where(both, sq, torch.zeros_like(sq)).sum(-1),
        "out_symdiff": torch.where(lam ^ lam_p, sq, torch.zeros_like(sq)).sum(-1),
        "enter": (lam_p & ~lam).sum(-1), "leave": (lam & ~lam_p).sum(-1),
        "xi_ref": ref["xi"], "xi_pert": pert["xi"],
    }
    if x is not None and params.D is not None:
        target = x.to(params.P.dtype).mean(1)
        power = (target ** 2).sum(-1)
        for name, run in (("nmse_ref", ref), ("nmse_pert", pert)):
            recon = torch.relu(run["z"] @ params.D.T).mean(1)
            err = ((target - recon) ** 2).sum(-1)
            out[name] = torch.where(power > 0, 10 * torch.log10(err / power),
                                    torch.full_like(err, float("nan")))
    return out


# ======================================================================
# Project-dependent part
# ======================================================================
def setup(gpu):
    """Load the seed-42 slista_dense model, sensor and the official test loader."""
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    sys.path.insert(0, str(ROOT))
    import pandas as pd
    import main as M
    import utils
    from config import Config
    Config.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ex = M.experiment_registry()[KEY]
    M.configure(ex, SEED)
    M.set_seed(SEED)
    row = pd.read_csv(M.point_path(ex, SEED)).iloc[0]
    recon = Path(str(row["Recon Checkpoint"]))
    if not recon.is_file():
        recon = Path(Config.RESULTS_DIR) / recon.name
    if not recon.is_file():
        raise FileNotFoundError(recon)
    model, sensor, channel = M.build_components(ex)
    state = torch.load(recon, map_location=Config.DEVICE, weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    sensor.load_state_dict(state["sensor"], strict=True)
    for module in (model, sensor, channel):
        module.eval()
        module.requires_grad_(False)
    # Same as the evaluation loader of utils.get_dataloaders (official test
    # set, same batch size, no shuffling), without loading the training set.
    loader = torch.utils.data.DataLoader(utils.SHDDataset(split="test"),
                                         batch_size=Config.BATCH_SIZE_RECON, shuffle=False,
                                         num_workers=0, pin_memory=False)
    return {"M": M, "utils": utils, "Config": Config, "ex": ex, "model": model, "sensor": sensor,
            "awgn": channel, "loader": loader, "recon": str(recon), "pd": pd}


def make_channel(ctx, name, snr):
    """Channel with the RNG state of the formal evaluation at this SNR
    (evaluate_grid / evaluate_rayleigh -> validate_joint_metrics)."""
    from wireless_channel import FrozenRayleighChannel
    Config = ctx["Config"]
    Config.EVAL_CHANNEL_SEED = 2026
    if name == "awgn":
        channel = ctx["awgn"]
        channel.set_snr_db(float(snr))
    else:
        channel = FrozenRayleighChannel(20.0, ctx["ex"].transport).to(Config.DEVICE)
        channel.set_snr_db(float(snr))
        channel.reset_rng(Config.EVAL_CHANNEL_SEED)
    channel.eval()
    ctx["utils"].deterministic_eval()
    return channel


@torch.no_grad()
def transmit(ctx, channel, x):
    """Same calls as Trainer._process_reconstruction.  Returns tx, rx [B,T,M]
    and, for the Rayleigh channel, whether the fade is shared by all frames."""
    sensor = ctx["sensor"]
    sensor_state = channel_state = None
    tx, rx, shared = [], [], True
    gain0 = None
    for t in range(x.shape[1]):
        y_enc, sensor_state, aux = sensor(x[:, t], state=sensor_state, mode="spiking", tau=None)
        y_noisy, channel_state = channel(y_enc, state=channel_state, mode="spiking",
                                         tx_prob=aux.get("tx_prob"), tx_hard=aux.get("tx_hard"),
                                         tx_aux=aux)
        tx.append(aux["tx_hard"])
        rx.append(y_noisy)
        if isinstance(channel_state, dict) and "gain_power" in channel_state:
            gain = channel_state["gain_power"]
            gain0 = gain.clone() if gain0 is None else gain0
            shared &= bool(torch.equal(gain0, gain))
    return torch.stack(tx, 1), torch.stack(rx, 1), shared


@torch.no_grad()
def float32_run(model, y):
    """The model's own forward (float32, project matmul settings) over a sequence.
    Returns x_rec [B,T,700] and spikes int8 [B,T,L-1,N]."""
    states, recs, spikes = None, [], []
    for t in range(y.shape[1]):
        x_rec, _, states, _ = model(y[:, t], states=states)
        recs.append(x_rec)
        spikes.append(torch.stack(model.last_trace["spikes"], 1).to(torch.int8))
    return torch.stack(recs, 1), torch.stack(spikes, 1)


def formal_nmse(ctx, channel_name):
    """Seed-42 S-LISTA temporal-mean NMSE of the formal csv, per SNR."""
    pd, Config = ctx["pd"], ctx["Config"]
    name = "snr_robustness_per_seed.csv" if channel_name == "awgn" else "snr_robustness_rayleigh_per_seed.csv"
    path = Path(Config.RESULTS_DIR) / name
    if not path.is_file():
        return {}
    frame = pd.read_csv(path)
    frame = frame[frame["Seed"].eq(SEED) & frame["Experiment Key"].eq(KEY)]
    return {float(r["Eval SNR (dB)"]): float(r["Mean NMSE (dB)"]) for _, r in frame.iterrows()}


def bounds(r_star, p, n):
    """Eq. (S30) per sample: lower, upper (0 for r_star = inf)."""
    from scipy.special import gammaln
    r = np.asarray(r_star, dtype=np.float64)
    finite = r < INF
    lower = np.zeros_like(r)
    upper = np.zeros_like(r)
    if p <= 0:
        return lower, upper
    rf = r[finite]
    lower[finite] = np.exp(rf * np.log(p) + (n - rf) * np.log1p(-p))
    log_choose = gammaln(n + 1) - gammaln(rf + 1) - gammaln(n - rf + 1)
    upper[finite] = np.minimum(1.0, np.exp(log_choose + rf * np.log(p)))
    return lower, upper


def code_checks(ctx):
    """Checks 1, 4 (3 is checked during the Rayleigh runs, 2 from BER)."""
    from snn_models import BoxcarSpike
    model = ctx["model"]
    theta = model.vartheta.detach()[:1]
    at_threshold = BoxcarSpike.apply(theta.clone(), theta, theta * 0.3)
    return {
        "quantizer_ge (H(0)=1)": bool(float(at_threshold) == 1.0),
        "rho": float(model.rho),
        "theta": [float(v) for v in model.vartheta.detach().cpu()],
        "eta": float(model.eta.detach()),
        "N_z": int(model.P.weight.shape[0]), "M": int(model.P.weight.shape[1]),
        "spiking_layers": int(model.spiking_layers),
        "G_convention": "I^(l+1) = I^(1) - G[l-1] z^(l) (snn_models.SpikingLISTA.trace); readout uses G[L-2]",
        "P_input": "trace(): initial = P(y_frame), no scaling or bias",
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
    }


# ----------------------------------------------------------------- part A
@torch.no_grad()
def part_a(ctx, out_dir, max_batches=None):
    model = ctx["model"]
    p64 = params_from_model(model)
    P, theta, rho = p64.P, p64.theta, p64.rho
    rows, all_margins = [], []
    mismatch = {"f32_tf32": 0, "f32_strict": 0, "elements": 0}
    sample = 0
    awgn = make_channel(ctx, "awgn", 20.0)
    for batch_index, (x, labels) in enumerate(ctx["loader"]):
        if max_batches is not None and batch_index >= max_batches:
            break
        tx, _, _ = transmit(ctx, awgn, x)
        _, spikes_tf32 = float32_run(model, tx)
        previous = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        _, spikes_strict = float32_run(model, tx)
        torch.backends.cuda.matmul.allow_tf32 = previous
        for start in range(0, x.shape[0], CHUNK):
            sl = slice(start, start + CHUNK)
            ref = forward(p64, tx[sl])
            mismatch["f32_tf32"] += int((ref["xi"] != spikes_tf32[sl]).sum())
            mismatch["f32_strict"] += int((ref["xi"] != spikes_strict[sl]).sum())
            mismatch["elements"] += int(ref["xi"].numel())
            for b in range(ref["xi"].shape[0]):
                ubar, xi = ref["ubar"][b], ref["xi"][b]
                summary = rstar_summary(ubar, xi, theta, P, tx[sl][b].to(torch.float64), rho)
                r = summary["r"]
                delta = margins(ubar, theta)
                all_margins.append(delta.flatten().float().cpu())
                row = {"Sample": sample, "Label": int(labels[sl][b]),
                       "r_star": summary["r_star"] if summary["r_star"] < INF else np.inf,
                       "Argmin Frame": summary["frame"] + 1 if summary["frame"] >= 0 else -1,
                       "Argmin Neuron": summary["neuron"], "Direction": summary["direction"],
                       "First Changed Layer": summary["layer"],
                       "Flip Bits (frame:channel)": ";".join(f"{k + 1}:{j}" for k, j in summary["bits"]),
                       "n r_ki=1": int((r == 1).sum()), "frac r_ki=1": float((r == 1).float().mean()),
                       "Min Margin": float(delta.min()), "Active Frames": int((tx[sl][b].sum(-1) > 0).sum())}
                for q in range(1, 11):
                    row[f"r_ki={q}"] = int((r == q).sum())
                row["r_ki>10"] = int(((r > 10) & (r < INF)).sum())
                row["r_ki=inf"] = int((r >= INF).sum())
                rows.append(row)
                sample += 1
        print(f"[A] batch {batch_index}: {sample} samples", flush=True)
    frame = ctx["pd"].DataFrame(rows)
    frame.to_csv(out_dir / "rstar_per_sample.csv", index=False)
    margin_all = torch.cat(all_margins).double()
    qs = [0.0, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    values = np.quantile(margin_all.numpy(), qs)
    finite = np.isfinite(frame["r_star"].to_numpy())
    rs = frame["r_star"].to_numpy()
    ge2 = int((rs >= 2).sum())
    summary = {
        "samples": int(len(frame)),
        "margin_quantiles": {f"{q:g}": float(v) for q, v in zip(qs, values)},
        "r_star_counts": {str(int(v)) if np.isfinite(v) else "inf": int((rs == v).sum())
                          for v in sorted(set(rs.tolist()))},
        "r_star>=2": {"k": ge2, "n": int(len(rs)), "frac": ge2 / max(1, len(rs)), "cp95": cp_interval(ge2, len(rs))},
        "r_star_finite": int(finite.sum()),
        "first_changed_layer_counts": {str(int(k)): int(v) for k, v in frame["First Changed Layer"].value_counts().items()},
        "direction_counts": {str(k): int(v) for k, v in frame["Direction"].value_counts().items()},
        "mean frac r_ki=1": float(frame["frac r_ki=1"].mean()),
        "reference_spike_mismatch_float32_vs_float64": mismatch,
    }
    (out_dir / "rstar_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return frame


# ----------------------------------------------------------------- part B
@torch.no_grad()
def part_b(ctx, out_dir, rstar, snrs, max_batches=None):
    model = ctx["model"]
    p64 = params_from_model(model)
    n_bits = None
    per_sample, summary_rows = [], []
    for channel_name in ("awgn", "rayleigh"):
        formal = formal_nmse(ctx, channel_name)
        for snr in snrs:
            t0 = time.time()
            channel = make_channel(ctx, channel_name, snr)
            acc = {"temporal_error": 0.0, "temporal_target": 0.0, "flips": 0, "bits": 0,
                   "frames": 0, "cls": np.zeros(4, np.int64), "agree": 0, "agree_layer": 0, "valid": 0,
                   "s26": 0.0, "s31": 0.0, "s32": 0.0, "energy": np.zeros(4), "enter": 0, "leave": 0,
                   "first_layer": np.zeros(p64.layers + 1, np.int64), "f32_mismatch": 0,
                   "f32_elements": 0, "fade_shared": True}
            ratios = []
            sample = 0
            for batch_index, (x, labels) in enumerate(ctx["loader"]):
                if max_batches is not None and batch_index >= max_batches:
                    break
                tx, rx, shared = transmit(ctx, channel, x)
                acc["fade_shared"] &= shared
                n_bits = tx.shape[1] * tx.shape[2]
                # Formal NMSE reproduction (float32, as validate_joint_metrics).
                x_rec, spikes32 = float32_run(model, rx)
                acc["temporal_error"] += float(torch.sum((x.mean(1) - x_rec.mean(1)) ** 2))
                acc["temporal_target"] += float(torch.sum(x.mean(1) ** 2))
                acc["flips"] += int((rx != tx).sum())
                acc["bits"] += int(tx.numel())
                for start in range(0, x.shape[0], CHUNK):
                    sl = slice(start, start + CHUNK)
                    f = analyze_frames(p64, tx[sl], rx[sl], x[sl])
                    acc["f32_mismatch"] += int((f["xi_pert"] != spikes32[sl]).sum())
                    acc["f32_elements"] += int(spikes32[sl].numel())
                    cls = f["cls"]
                    acc["frames"] += int(cls.numel())
                    acc["cls"] += np.bincount(cls.flatten().cpu().numpy(), minlength=4)
                    valid = f["valid"]
                    acc["valid"] += int(valid.sum())
                    acc["agree"] += int((((f["first_actual"] > 0) == (f["first_pred"] > 0)) & valid).sum())
                    acc["agree_layer"] += int(((f["first_actual"] == f["first_pred"]) & valid).sum())
                    for name in ("s26", "s31", "s32"):
                        acc[name] = max(acc[name], float(f[name].max()))
                    energy = f["out_energy"]
                    acc["energy"][0] += float(energy[cls <= 1].sum())
                    acc["energy"][1] += float(energy[cls == 2].sum())
                    acc["energy"][2] += float(f["out_common"][cls == 3].sum())
                    acc["energy"][3] += float(f["out_symdiff"][cls == 3].sum())
                    acc["enter"] += int(f["enter"][cls == 3].sum())
                    acc["leave"] += int(f["leave"][cls == 3].sum())
                    acc["first_layer"] += np.bincount(f["first_actual"].flatten().cpu().numpy(),
                                                      minlength=p64.layers + 1)
                    ratios.append(f["ratio"][cls == 2].cpu())
                    bit_err = f["bit_errors"].sum(1).cpu().numpy()
                    unchanged = (~f["changed"].any(1)).cpu().numpy()
                    counts = [(cls == c).sum(1).cpu().numpy() for c in range(4)]
                    for b in range(cls.shape[0]):
                        per_sample.append({
                            "Channel": channel_name, "Eb/N0 (dB)": float(snr), "Sample": sample,
                            "Bit Errors": int(bit_err[b]), "Spikes Unchanged": bool(unchanged[b]),
                            **{f"Frames {CLASS_NAMES[c]}": int(counts[c][b]) for c in range(4)},
                            "NMSE ref (dB)": float(f["nmse_ref"][b]), "NMSE pert (dB)": float(f["nmse_pert"][b]),
                        })
                        sample += 1
            n_samples = sample
            rows = [r for r in per_sample if r["Channel"] == channel_name and r["Eb/N0 (dB)"] == float(snr)]
            unchanged = np.array([r["Spikes Unchanged"] for r in rows])
            inc = np.array([r["NMSE pert (dB)"] - r["NMSE ref (dB)"] for r in rows])
            k_s = int(unchanged.sum())
            frames_unchanged = int(acc["cls"][0])
            nmse = 10.0 * np.log10(acc["temporal_error"] / acc["temporal_target"] + 1e-10)
            ber = acc["flips"] / max(1, acc["bits"])
            ber_ci = cp_interval(acc["flips"], acc["bits"])
            ratio_all = torch.cat(ratios).numpy() if ratios else np.zeros(0)
            ratio_all = ratio_all[np.isfinite(ratio_all)]
            row = {
                "Channel": channel_name, "Eb/N0 (dB)": float(snr), "Samples": n_samples,
                "Frames": acc["frames"],
                "Samples Unchanged": k_s, "Frac Samples Unchanged": k_s / max(1, n_samples),
                "CP lo Samples": cp_interval(k_s, n_samples)[0], "CP hi Samples": cp_interval(k_s, n_samples)[1],
                "Frames Unchanged": frames_unchanged, "Frac Frames Unchanged": frames_unchanged / max(1, acc["frames"]),
                "CP lo Frames": cp_interval(frames_unchanged, acc["frames"])[0],
                "CP hi Frames": cp_interval(frames_unchanged, acc["frames"])[1],
                **{f"Frames {CLASS_NAMES[c]}": int(acc["cls"][c]) for c in range(4)},
                **{f"Frac {CLASS_NAMES[c]}": float(acc["cls"][c]) / max(1, acc["frames"]) for c in range(4)},
                "BER measured": ber, "BER CP lo": ber_ci[0], "BER CP hi": ber_ci[1],
                "BER formula (AWGN, h=1)": 0.5 * math.erfc(math.sqrt(10.0 ** (snr / 10.0))),
                "Thm4 agree (changed?)": acc["agree"], "Thm4 agree (first layer)": acc["agree_layer"],
                "Thm4 frames compared": acc["valid"],
                "Max resid S26": acc["s26"], "Max resid S31": acc["s31"], "Max resid S32": acc["s32"],
                "Out energy total": float(acc["energy"].sum()),
                "Out energy (i)+(ii)": acc["energy"][0], "Out energy (iii)": acc["energy"][1],
                "Out energy (iv) common": acc["energy"][2], "Out energy (iv) symdiff": acc["energy"][3],
                "(iv) entering coords": acc["enter"], "(iv) leaving coords": acc["leave"],
                "R_t n": int(ratio_all.size),
                "R_t median": float(np.median(ratio_all)) if ratio_all.size else np.nan,
                "R_t q25": float(np.quantile(ratio_all, 0.25)) if ratio_all.size else np.nan,
                "R_t q75": float(np.quantile(ratio_all, 0.75)) if ratio_all.size else np.nan,
                "R_t frac<1": float((ratio_all < 1).mean()) if ratio_all.size else np.nan,
                **{f"First change layer {l}": int(acc["first_layer"][l]) for l in range(1, p64.layers + 1)},
                "Mean NMSE increment unchanged (dB)": float(np.nanmean(inc[unchanged])) if unchanged.any() else np.nan,
                "Mean NMSE increment changed (dB)": float(np.nanmean(inc[~unchanged])) if (~unchanged).any() else np.nan,
                "NMSE reproduced (dB)": nmse, "NMSE formal csv (dB)": formal.get(float(snr), np.nan),
                "float32 vs float64 spike mismatches (perturbed)": acc["f32_mismatch"],
                "Spike elements": acc["f32_elements"],
                "Fade shared by all frames": bool(acc["fade_shared"]) if channel_name == "rayleigh" else "",
            }
            if channel_name == "awgn":
                formula = row["BER formula (AWGN, h=1)"]
                use_formula = ber_ci[0] <= formula <= ber_ci[1]
                p = formula if use_formula else ber
                r_values = rstar["r_star"].to_numpy()[:n_samples]
                r_values = np.where(np.isfinite(r_values), r_values, INF)
                lower, upper = bounds(r_values, p, n_bits)
                row.update({"p_e used for bound": p, "p_e source": "formula" if use_formula else "measured",
                            "Mean lower_s": float(lower.mean()), "Mean upper_s": float(upper.mean()),
                            "Unchanged band lo (1-mean upper)": 1 - float(upper.mean()),
                            "Unchanged band hi (1-mean lower)": 1 - float(lower.mean())})
                for r, lo, hi in zip(rows, lower, upper):
                    r["lower_s"], r["upper_s"] = float(lo), float(hi)
            summary_rows.append(row)
            print(f"[B] {channel_name} {snr:g} dB: unchanged samples {k_s}/{n_samples}, "
                  f"frames {frames_unchanged}/{acc['frames']}, classes {acc['cls'].tolist()}, "
                  f"NMSE {nmse:.6f} (formal {row['NMSE formal csv (dB)']}), BER {ber:.3e}, "
                  f"resid {acc['s26']:.1e}/{acc['s31']:.1e}/{acc['s32']:.1e}, "
                  f"thm4 {acc['agree']}/{acc['valid']}, {time.time() - t0:.0f}s", flush=True)
            pd = ctx["pd"]
            pd.DataFrame(summary_rows).to_csv(out_dir / "mc_summary.csv", index=False)
            pd.DataFrame(per_sample).to_csv(out_dir / "mc_per_sample.csv", index=False)
    return summary_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpu", type=str, required=True)
    parser.add_argument("--smoke", action="store_true", help="first test batch, 2 SNRs")
    args = parser.parse_args()
    out_dir = ROOT / "outputs" / ("sn3_fig3_smoke" if args.smoke else "sn3_fig3")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    ctx = setup(args.gpu)
    max_batches = 1 if args.smoke else None
    snrs = [0, 20] if args.smoke else SNR_GRID
    info = {"checkpoint": ctx["recon"], "seed": SEED, "experiment": KEY,
            "code_checks": code_checks(ctx), "smoke": args.smoke}
    (out_dir / "run_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2), flush=True)
    if info["code_checks"]["rho"] != 0.0:
        print("WARNING: rho != 0; frames are not independent.", flush=True)
    start = time.time()
    rstar = part_a(ctx, out_dir, max_batches)
    part_b(ctx, out_dir, rstar, snrs, max_batches)
    print(f"Done in {time.time() - start:.0f}s; outputs in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
