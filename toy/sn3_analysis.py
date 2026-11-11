"""Quantities of Supplementary Note 3 (SN3) used by Supplementary Figs. 1 and 2.

Pure numpy, float64.  No torch: sn3_figs.py loads the trained models and
converts every weight, threshold and signal to float64 numpy arrays first; the
self-test (sn3_selftest.py) calls the same functions on small random networks.

Notation:
  P   (n, m)    read-in matrix  = model.P.weight
  G   list of L-1 (n, n) matrices, G[l-1] = model.G[l-1].weight, so that
                I^(l+1) = I^(1) - G^(l) z^(l)  and  (G z)_i = sum_j G_ij z_j
  theta (L-1,)  thresholds = model.theta (floor + softplus), eta = model.eta
  A   (m, n)    sensing matrix, y = A z*  (data.sparse_batch: y = z* @ A.T)
  D   = P A     (n, n)
Batches are row-major: one sample per row.
"""

import math

import numpy as np

import energy_spec as es

QUANTILES = (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)


# ------------------------------------------------------------------ forward
def forward(P, G, theta, eta, y):
    """float64 reference of models.SpikingLISTA.trace (same operations, same >=)."""
    initial = y @ P.T
    code = np.zeros_like(initial)
    currents, spikes, codes = [], [], []
    current = initial
    for layer in range(len(G)):
        currents.append(current)
        spike = ((current >= theta[layer]).astype(np.float64)
                 - (-current >= theta[layer]).astype(np.float64))
        code = code + spike
        spikes.append(spike)
        codes.append(code)
        current = initial - code @ G[layer].T
    currents.append(current)
    mask = (code != 0).astype(np.float64)
    output = code + eta * mask * current
    return {"currents": currents, "spikes": spikes, "codes": codes,
            "mask": mask, "spike_code": code, "output": output}


# ------------------------------------------------------- Supp. Fig. 1 (S14, S15)
def error_split(trace, target, D, G, eta):
    """Per-sample terms of Eq. (S15) and Eq. (S14).  G = G_s^(L-1)."""
    zL = trace["spike_code"]
    z = trace["output"]
    lam = zL != 0
    sup = target != 0
    lamf = lam.astype(np.float64)
    diff = z - target
    e = zL - target
    out = {
        "n_lambda": lam.sum(1),
        "n_miss": (sup & ~lam).sum(1),
        "n_false": (lam & ~sup).sum(1),
        "zstar_sq": (target ** 2).sum(1),
        "err_total": (diff ** 2).sum(1),
        "E_amp": np.where(lam & sup, diff ** 2, 0.0).sum(1),
        "E_miss": np.where(sup & ~lam, target ** 2, 0.0).sum(1),
        "E_false": np.where(lam & ~sup, z ** 2, 0.0).sum(1),
        "before_LS": np.where(lam & sup, e ** 2, 0.0).sum(1),
    }
    # Eq. (S14), restricted to Lambda; masked products equal the sub-matrix products.
    e_lam = e * lamf
    t1 = lamf * (e_lam - eta * (e_lam @ G.T))
    t2 = eta * lamf * ((target * lamf) @ (D - G).T)
    t3 = eta * lamf * ((target * (1.0 - lamf)) @ D.T)
    lhs = lamf * diff
    out.update(T1_sq=(t1 ** 2).sum(1), T2_sq=(t2 ** 2).sum(1), T3_sq=(t3 ** 2).sum(1),
               T12_sq=((t1 + t2) ** 2).sum(1),
               err_lambda_sq=(lhs ** 2).sum(1),
               zstar_lambda_sq=((target * lamf) ** 2).sum(1),
               s14_resid=np.abs(t1 + t2 + t3 - lhs).max(1),
               s15_resid=np.abs(out["E_amp"] + out["E_miss"] + out["E_false"]
                                - out["err_total"]))
    return out


def reference_condition(trace, target, theta):
    """Reference-case condition (Section 3.2) |w^(k)| < theta_k - 1/2, w^(k) = I^(k) - (z* - z^(k-1)),
    z^(0) = 0, k = 1..L-1.  Returns per-sample counts and per-layer counts."""
    sup = target != 0
    prev = np.zeros_like(target)
    ok_all = np.ones(target.shape, dtype=bool)
    triples = np.zeros(target.shape[0], dtype=np.int64)
    per_layer = []
    for k, current in enumerate(trace["currents"][:-1]):
        w = current - (target - prev)
        ok = np.abs(w) < theta[k] - 0.5
        ok_all &= ok
        triples += ok.sum(1)
        per_layer.append(int(ok.sum()))
        prev = trace["codes"][k]
    return {"lemma_triples_ok": triples,
            "lemma_coord_ok_S": (ok_all & sup).sum(1),
            "lemma_coord_ok_notS": (ok_all & ~sup).sum(1),
            "lemma_sample_ok": ok_all.all(1).astype(np.int64)}, per_layer


def lost_coordinates(trace, target):
    """Coordinates nonzero in some layer but zero in z^(L-1)."""
    sup = target != 0
    ever = np.zeros(target.shape, dtype=bool)
    for code in trace["codes"]:
        ever |= code != 0
    lost = ever & (trace["spike_code"] == 0)
    return {"lost_S": (lost & sup).sum(1), "lost_notS": (lost & ~sup).sum(1)}


def restricted_ratio(trace, D, G):
    """||(D - G)_{Lambda Lambda}||_2 / ||D_{Lambda Lambda}||_2 per sample (nan if Lambda empty)."""
    lam = trace["spike_code"] != 0
    diff = D - G
    out = np.full(lam.shape[0], np.nan)
    for b in range(lam.shape[0]):
        idx = np.flatnonzero(lam[b])
        if idx.size == 0:
            continue
        sub = np.ix_(idx, idx)
        den = np.linalg.norm(D[sub], 2)
        out[b] = np.linalg.norm(diff[sub], 2) / den if den > 0 else np.nan
    return out


# ------------------------------------------------------- Supp. Fig. 2 (S17, S18)
def energy_constants(m, n, depth, p=es.SPEC):
    """c0..c3 of Eq. (S17), C_L and C'_L of Eq. (S18) and the closed form of E_0 (pJ)."""
    c0 = (n + 1) * p.byte
    c1 = n * p.ac
    c2 = 2 * p.byte + p.add
    c3 = p.mul + p.add + p.byte
    L = depth
    return {"c0": c0, "c1": c1, "c2": c2, "c3": c3,
            "C_L": (L - 1) * (c0 + c2) + 0.5 * L * (L - 1) * c1 + c3,
            "C_L_prime": c0 + (L - 1) * c1 + 2 * c2 + c3,
            "E0_formula": (m * n * (p.mac + p.byte) + m * p.byte
                           + es.BIAS_BYTES * n * L * p.byte + 2 * n * (L - 1) * p.add)}


def energy_terms(trace, target, m, n, p=es.SPEC):
    """Per-sample E_spec (energy_spec.slista), the four code terms of (S17),
    F_l = |supp(z^(l)) \\ S| and the bound terms of (S18).  Energies in pJ."""
    depth = len(trace["codes"]) + 1
    const = energy_constants(m, n, depth, p)
    sup = target != 0
    xi0 = np.stack([(x != 0).sum(1) for x in trace["spikes"]], 1)        # (B, L-1)
    z0 = np.stack([(c != 0).sum(1) for c in trace["codes"]], 1)
    z1 = np.stack([np.abs(c).sum(1) for c in trace["codes"]], 1)
    false = np.stack([((c != 0) & ~sup).sum(1) for c in trace["codes"]], 1)
    mask = trace["mask"].sum(1)
    e_spec = np.empty(target.shape[0])
    for b in range(target.shape[0]):
        layers = [{"xi0": float(xi0[b, l]), "z0": float(z0[b, l]), "z1": float(z1[b, l])}
                  for l in range(depth - 1)]
        e_spec[b] = es.slista(m, n, layers, float(mask[b]), p).total(p)
    code_part = (const["c0"] * z0.sum(1) + const["c1"] * z1.sum(1)
                 + const["c2"] * xi0.sum(1) + const["c3"] * mask)
    s = sup.sum(1)
    F = false.sum(1)
    out = {"E_spec_pJ": e_spec, "E0_pJ": e_spec - code_part,
           "E_minus_E0_pJ": code_part,
           "B_pJ": const["C_L"] * s + const["C_L_prime"] * F,
           "F": F, "s_true": s}
    for l in range(depth - 1):
        out[f"F_{l + 1}"] = false[:, l]
    return out


def first_layer_bound(target, D, theta1):
    """2 sum_{i not in S} exp(-theta_1^2 / (2 omega_i^2)), omega_i^2 = sum_{j in S} D_ij^2 z*_j^2
    (noise-free); terms with omega_i^2 = 0 count 0."""
    omega2 = (target ** 2) @ (D ** 2).T
    outside = (target == 0) & (omega2 > 0)
    safe = np.where(outside, omega2, 1.0)
    return 2.0 * np.where(outside, np.exp(-theta1 ** 2 / (2.0 * safe)), 0.0).sum(1)


def persistence_counts(trace, target):
    """For every i not in S: number of layers with z_i^(l) != 0 (0..L-1), as a histogram."""
    layers = len(trace["codes"])
    count = np.zeros(target.shape, dtype=np.int64)
    for code in trace["codes"]:
        count += code != 0
    return np.bincount(count[target == 0], minlength=layers + 1)


# ------------------------------------------------------------------ statistics
def clopper_pearson(k, N):
    from scipy.stats import beta
    k, N = int(k), int(N)
    lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, N - k + 1))
    hi = 1.0 if k == N else float(beta.ppf(0.975, k + 1, N - k))
    return {"k": k, "N": N, "fraction": k / N if N else float("nan"), "ci95": [lo, hi]}


def describe(values):
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"count": 0}
    q25, med, q75 = np.quantile(v, [0.25, 0.5, 0.75])
    return {"count": int(v.size), "mean": float(v.mean()), "median": float(med),
            "q25": float(q25), "q75": float(q75)}


def quantiles(values):
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return {"count": 0}
    return {"count": int(v.size), "mean": float(v.mean()),
            **{f"q{int(round(100 * q)):02d}": float(np.quantile(v, q)) for q in QUANTILES}}


def db(num, den):
    return 10.0 * math.log10(num / den) if num > 0 and den > 0 else float("-inf")
