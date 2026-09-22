#!/usr/bin/env python3
"""Compare the joint-secant S-LISTA recovery certificate with the old regional bound.

Standalone: no earlier audit script is required. No training. Toy checkpoints: T=1 only.

  python audit_slista_joint_bound.py --sources outputs/seed42/checkpoints \
    --conditions m141_s28 m141_s42 m141_s56 --depths 5 25 --samples 10 --device cuda:0

Fast reuse of old audit_slista_pattern_bounds.py certificate.npz files:
  python audit_slista_joint_bound.py --certificates pattern_bound_audit

  python audit_slista_joint_bound.py --self-test

Dependencies: numpy, scipy; torch for checkpoints. Optional noise-radius > 0
requires clarabel (python -m pip install clarabel). LP/SOCP calculations use CPU.

WHAT IS COMPARED
For each reference: actual SSE, old coordinate/basis regional SSE upper bound,
new joint-secant upper bound, and attained same-pattern witness SSE (a lower
bound on the region's worst SSE). All comparisons use the SAME support, prior,
pulse pattern, previous-frame state, and measurement/noise model. No fitting of
the region to the observed reference error. Positive/negative coefficient signs
are NOT fixed. Reference support is oracle information, NOT a receiver-known
certificate. Only support and the declared amplitude/noise budgets define priors.

If e_i lies in [l_i,u_i], e_i**2 <= (l_i+u_i)*e_i-l_i*u_i. We maximize the SUM
of these secants over ONE common candidate, then retain the minimum of this
valid upper certificate and the old bound. Report raw secant results too: a tie
or solver fallback must not be called a strict improvement. Both original
coordinates and the same thin error-subspace basis used by the old bound are
evaluated, including numerical projection residuals. The basis includes noise
columns whenever noise is present. Quarter-squared interval widths quantify
the exact secant relaxation gap; solver certificate gaps are separate.

MEMORY / NOISE / FINAL SOFT THRESHOLD
--frame-npz accepts one or more npz files containing a single streaming frame:
  A: (m,n), P: (n,m), G: (K-1,n,n), z_true: (n,), B: positive scalar,
  theta: scalar, (K,), or (K,n), previous_membrane: (K,n), rho: scalar.
Optional noise: (m,), noise_radius: nonnegative scalar; output_threshold: scalar.
previous_membrane is the ACTUAL post-reset state of the PREVIOUS frame, before
decay. It is fixed, not optimized, and remains inside the signed affine formula.
If noise is supplied, noise_radius must cover it; we never infer a budget from
the observed noise. Missing noise/radius/threshold defaults to zero. This is a
state-conditional frame certificate, not a full-sequence noise stability theorem.
This frame adapter matches the paper's per-frame support-gated readout; it is
NOT the toy T>1 rate-averaged readout. T=4 checkpoints are never audited.

--noise-radius on toy checkpoints draws independent noise uniformly in the
measurement-domain L2 ball of this declared radius. This is a bounded-noise
probe, NOT an AWGN or Rayleigh simulation. --post-threshold is an explicit
optional readout ablation on new toy checkpoints; it is not silently loaded
from old output_threshold checkpoints, whose forward semantics are unknown.

All upper certificates use nonnegative dual multipliers plus the support
function of the box/noise ball to cover dual residuals. Numerical guards are
included. These are float64/longdouble numerical certificates, not rigorous
directed-rounding interval certificates. Native float32 paths/errors are
reported separately; the certificate is for the float64 mathematical replay.
Summary ratios are ratios of SUMS of squared errors, not averages of dB/ratios.
No figures are generated. See samples.csv, summary.csv, report.json.
"""

import argparse
import csv
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')
import numpy as np
from scipy.optimize import linprog


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def ratio(a, b):
    return float(a/b) if b > 1e-24 else None


def db(a, energy):
    return float(10*np.log10(max(a/energy, 1e-300))) if energy > 0 else None


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def soft(x, threshold):
    return np.sign(x)*np.maximum(np.abs(x)-threshold, 0)


class Region:
    """R v <= d; |v[:s]| <= B; ||v[s:]||_2 <= 1 if noise variables exist."""
    def __init__(self, R, d, B, s, guard=1e-10, seconds=30.):
        self.R, self.d = np.asarray(R, float), np.asarray(d, float)
        self.B, self.s, self.guard, self.seconds = float(B), int(s), guard, seconds
        self.dim = self.R.shape[1]
        if self.s <= 0 or self.s > self.dim or not (self.B > 0 and np.isfinite(self.B)):
            raise ValueError('Need nonempty support, positive B, and compatible dimensions.')
        if not np.isfinite(self.R).all() or not np.isfinite(self.d).all():
            raise ValueError('Nonfinite region coefficients')
        self.calls = 0
        self.solver = None
        if self.dim > self.s:
            try:
                import clarabel
            except ImportError as exc:
                raise RuntimeError('Noise-ball audits require: python -m pip install clarabel') from exc
            from scipy import sparse
            box = sparse.hstack([sparse.eye(self.s), sparse.csc_matrix((self.s, self.dim-self.s))])
            soc = sparse.vstack([sparse.csc_matrix((1, self.dim)),
                sparse.hstack([sparse.csc_matrix((self.dim-self.s, self.s)), -sparse.eye(self.dim-self.s)])])
            matrix = sparse.vstack([sparse.csc_matrix(self.R), box, -box, soc], format='csc')
            rhs = np.r_[self.d, np.full(2*self.s, self.B), 1., np.zeros(self.dim-self.s)]
            cones = [clarabel.NonnegativeConeT(len(self.d)+2*self.s),
                     clarabel.SecondOrderConeT(self.dim-self.s+1)]
            settings = clarabel.DefaultSettings()
            settings.verbose = False
            settings.presolve_enable = False
            settings.chordal_decomposition_enable = False
            settings.time_limit = seconds
            settings.max_iter = 200
            settings.tol_gap_abs = settings.tol_gap_rel = settings.tol_feas = 1e-9
            if hasattr(settings, 'max_threads'):
                settings.max_threads = 2
            self.solver = clarabel.DefaultSolver(sparse.csc_matrix((self.dim, self.dim)),
                np.zeros(self.dim), matrix, rhs, cones, settings)

    def support(self, h):
        h = np.asarray(h, np.longdouble)
        return self.B*np.abs(h[:self.s]).sum()+np.sqrt(np.sum(h[self.s:]**2))

    def dual(self, h, c, lam):
        lam = np.maximum(np.asarray(lam, np.longdouble), 0)
        h = np.asarray(h, np.longdouble)
        d, R = self.d.astype(np.longdouble), self.R.astype(np.longdouble)
        correction = self.support(h-R.T@lam)
        value = np.longdouble(c)+d@lam+correction
        scale = 1+abs(c)+np.abs(d*lam).sum()+self.support(h)+correction
        return float(value+self.guard*scale), float(correction)

    def maximize(self, h, c=0.):
        h = np.asarray(h, float)
        if not np.any(h):
            return dict(upper=float(c), primal=float(c), x=None,
                        lam=np.zeros(len(self.d)), residual=0., gap=0.)
        self.calls += 1
        if self.solver is None:
            result = linprog(-h, A_ub=self.R, b_ub=self.d,
                bounds=[(-self.B, self.B)]*self.dim, method='highs', options={
                'time_limit': self.seconds, 'primal_feasibility_tolerance': 1e-9,
                'dual_feasibility_tolerance': 1e-9})
            if not result.success:
                raise RuntimeError('LP failed: '+result.message)
            x = result.x
            lam = np.maximum(-np.asarray(result.ineqlin.marginals), 0)
        else:
            self.solver.update(q=-h)
            result = self.solver.solve()
            if str(result.status) not in ('Solved', 'AlmostSolved'):
                raise RuntimeError('SOCP failed: '+str(result.status))
            x = np.asarray(result.x)
            lam = np.maximum(np.asarray(result.z[:len(self.d)]), 0)
        if not np.isfinite(x).all() or not np.isfinite(lam).all():
            raise ArithmeticError('Nonfinite optimizer/dual multipliers')
        upper, residual = self.dual(h, c, lam)
        fallback, fallback_residual = self.dual(h, c, np.zeros(len(self.d)))
        if fallback < upper:
            upper, residual, lam = fallback, fallback_residual, np.zeros(len(self.d))
        primal = float(h@x+c)
        if primal > upper+1e-6*(1+abs(primal)):
            raise ArithmeticError('Primal objective exceeds numerical dual certificate')
        return dict(upper=upper, primal=primal, x=x, lam=lam, residual=residual,
                    gap=max(0., upper-primal))


def coordinate_and_joint(H, b, region, cached=None):
    """Old interval bound and new secant bound with identical coordinate intervals."""
    started = time.perf_counter()
    candidates = []
    endpoint_gaps = []
    if cached is None:
        signed = np.zeros((len(b), 2))
        last_progress = time.perf_counter()
        for i in range(len(b)):
            for j, sign in enumerate((1., -1.)):
                answer = region.maximize(sign*H[i], sign*b[i])
                signed[i, j] = answer['upper']
                endpoint_gaps.append(answer['gap'])
                if answer['x'] is not None:
                    candidates.append(answer['x'])
            if time.perf_counter()-last_progress > 20:
                print(f'      error intervals: {i+1}/{len(b)} coordinates', flush=True)
                last_progress = time.perf_counter()
    else:
        signed = np.asarray(cached, float).copy()
        if signed.shape != (len(b), 2) or not np.isfinite(signed).all():
            raise ValueError('Invalid cached coordinate certificates')
    lower, upper = -signed[:, 1], signed[:, 0]
    if np.any(lower > upper):
        raise ArithmeticError('Inconsistent error interval endpoints')
    old = float(np.sum(np.maximum(np.abs(lower), np.abs(upper))**2))
    secant_started = time.perf_counter()
    # Form the new affine objective in extended precision and cover conversion error.
    hl = H.astype(np.longdouble)
    bl = b.astype(np.longdouble)
    l, u = lower.astype(np.longdouble), upper.astype(np.longdouble)
    h_exact, c_exact = (l+u)@hl, (l+u)@bl-l@u
    h, c = np.asarray(h_exact, float), float(c_exact)
    conversion_guard = float(region.support(h_exact-h)+abs(c_exact-c))
    answer = region.maximize(h, c)
    raw_joint = answer['upper']+conversion_guard
    if raw_joint < -1e-7*(1+old):
        raise ArithmeticError('Negative SSE certificate; region/objective is inconsistent')
    raw_joint = max(0., raw_joint)
    if answer['x'] is not None:
        candidates.append(answer['x'])
    # Each operand is independently an upper certificate. Rounding/solver error
    # can make the raw secant certificate marginally worse, so retain the old one.
    kept = min(old, raw_joint)
    return dict(old=old, joint=kept, raw_joint=raw_joint, signed=signed,
        joint_h=h, joint_c=c, joint_lam=answer['lam'],
        joint_dual_residual=answer['residual'], joint_solver_gap=answer['gap']+conversion_guard,
        objective_conversion_guard=conversion_guard,
        interval_width_slack=float(np.sum((upper-lower)**2)/4),
        endpoint_gap_max=max(endpoint_gaps, default=0.),
        candidates=candidates, seconds=time.perf_counter()-started,
        secant_seconds=time.perf_counter()-secant_started,
        cache_reused=cached is not None)


def combined_bounds(H, b, region, cache=None):
    cache = cache or {}
    direct = coordinate_and_joint(H, b, region, cache.get('full_signed_upper'))
    basis = cache.get('full_basis_U')
    if basis is None:
        basis, _, _ = np.linalg.svd(H, full_matrices=False)
    projected_H, projected_b = basis.T@H, basis.T@b
    # Old saved reduced endpoints refer to the saved affine coefficients exactly.
    reduced_H = cache.get('full_basis_H', projected_H)
    reduced_b = cache.get('full_basis_b', projected_b)
    reduced = coordinate_and_joint(reduced_H, reduced_b, region, cache.get('full_basis_signed_upper'))
    perpendicular = b-basis@reduced_b
    residual = H-basis@reduced_H
    gain_sq = 1+np.linalg.norm(basis.T@basis-np.eye(basis.shape[1]), 'fro')
    cross = np.linalg.norm(basis.T@perpendicular)
    # Coefficient box plus optional noise ball; do not truncate noise directions.
    residual_bound = region.B*np.linalg.norm(residual[:, :region.s], axis=0).sum()
    residual_bound += np.linalg.norm(residual[:, region.s:], 'fro')

    def lift(value):
        value = max(0., value)
        norm = math.sqrt(max(0., gain_sq*value+2*cross*math.sqrt(value)+perpendicular@perpendicular))
        return float((norm+residual_bound+region.guard*(1+norm+residual_bound))**2)

    old_basis, new_basis = lift(reduced['old']), lift(reduced['joint'])
    old = min(direct['old'], old_basis)
    new = min(direct['joint'], new_basis)
    return dict(old=old, joint=new, direct=direct, reduced=reduced, basis=basis,
        reduced_H=reduced_H, reduced_b=reduced_b,
        old_basis=old_basis, joint_basis=new_basis,
        candidates=direct['candidates']+reduced['candidates'])


def thresholds(theta, K, n):
    theta = np.asarray(theta, float)
    if theta.shape == (K,):
        theta = theta[:, None]
    theta = np.broadcast_to(theta, (K, n)).copy()
    if not np.isfinite(theta).all() or np.any(theta <= 0):
        raise ValueError('Every firing threshold must be positive and finite')
    return theta


def replay(current, G, theta, previous, rho):
    K, n = len(G)+1, len(current)
    theta = thresholds(theta, K, n)
    previous = np.asarray(previous, float)
    if previous.shape != (K, n):
        raise ValueError('previous_membrane must have shape (K,n)')
    q = np.zeros(n)
    spikes, shifts, states = [], [], []
    for k in range(K):
        before = q.copy()
        feedback = np.zeros(n) if k == 0 else G[k-1]@before
        # pre = current - shift; memory is kept with its sign, not norm-bounded.
        shift = feedback-rho*previous[k]
        pre = current-shift
        spike = (pre >= theta[k]).astype(float)-(pre <= -theta[k]).astype(float)
        q += spike
        states.append(pre-theta[k]*spike)
        spikes.append(spike)
        shifts.append(shift)
    mask = q != 0
    output = mask*(before+(current-shifts[-1])/theta[-1])
    direct_output = q+mask*np.asarray(states)[-1]/theta[-1]
    if not np.allclose(output, direct_output, atol=1e-10, rtol=1e-10):
        raise AssertionError('Continuous readout cancellation identity failed')
    return dict(q=before, mask=mask, output=output, spikes=np.asarray(spikes),
                shifts=np.asarray(shifts), states=np.asarray(states), theta=theta)


def build_region(J, trace, B, s, guard):
    lo, hi = np.full(J.shape[0], -np.inf), np.full(J.shape[0], np.inf)
    for spike, shift, theta in zip(trace['spikes'], trace['shifts'], trace['theta']):
        lo = np.maximum(lo, np.where(spike >= 0, shift+np.where(spike > 0, theta, -theta), -np.inf))
        hi = np.minimum(hi, np.where(spike <= 0, shift+np.where(spike < 0, -theta, theta), np.inf))
    positive, negative = np.isfinite(hi), np.isfinite(lo)
    R = np.r_[J[positive], -J[negative]]
    d = np.r_[hi[positive], -lo[negative]]
    extent = B*np.abs(R[:, :s]).sum(axis=1)+np.linalg.norm(R[:, s:], axis=1)
    d += guard*(1+np.abs(d)+extent)
    scale = np.maximum(1., np.maximum(np.abs(d), np.max(np.abs(R), axis=1)))
    return R/scale[:, None], d/scale


def same_pattern(v, J, trace):
    pre = (J@v)[None, :]-trace['shifts']
    spike = (pre >= trace['theta']).astype(float)-(pre <= -trace['theta']).astype(float)
    return np.array_equal(spike, trace['spikes'])


def audit_region(H, b, J, support, anchor, trace, region, threshold, cache=None):
    start = time.perf_counter()
    if np.max(np.abs(anchor[:region.s])) > region.B+1e-12 or np.linalg.norm(anchor[region.s:]) > 1+1e-10:
        raise ValueError('Reference violates declared amplitude/noise budget; no bound is reported')
    if np.max(region.R@anchor-region.d, initial=0.) > 1e-8 or not same_pattern(anchor, J, trace):
        raise ValueError('Reference is not in the specified pulse region')
    truth = np.zeros(len(b))
    truth[support] = anchor[:region.s]
    actual_error = trace['output']-truth
    identity_error = float(np.max(np.abs(actual_error-(H@anchor+b))))
    if identity_error > 1e-8*(1+np.max(np.abs(actual_error))):
        raise AssertionError('Affine error identity failed')
    bounds = combined_bounds(H, b, region, cache)
    witness, witness_sse = anchor.copy(), float(actual_error@actual_error)
    if cache is not None and 'witness_x' in cache:
        bounds['candidates'].append(cache['witness_x'])
    for candidate in bounds['candidates']:
        candidate = np.asarray(candidate).copy()
        candidate[:region.s] = np.clip(candidate[:region.s], -region.B, region.B)
        candidate[region.s:] /= max(1., np.linalg.norm(candidate[region.s:]))
        for weight in (1., 1-1e-6, 1-1e-4, .99, .9, .5):
            v = weight*candidate+(1-weight)*anchor
            if same_pattern(v, J, trace):
                sse = float(np.sum((H@v+b)**2))
                if sse > witness_sse:
                    witness, witness_sse = v, sse
                break
    actual = float(actual_error@actual_error)
    old, new = bounds['old'], bounds['joint']
    if witness_sse > new+1e-7*(1+new) or new > old+1e-7*(1+old):
        raise AssertionError('Actual/witness <= joint <= old certificate check failed')
    final_actual = float(np.sum((soft(trace['output'], threshold)-truth)**2))
    final_new = float((math.sqrt(new)+threshold*math.sqrt(region.s))**2)
    final_old = float((math.sqrt(old)+threshold*math.sqrt(region.s))**2)
    if final_actual > final_new+1e-7*(1+final_new):
        raise AssertionError('Final soft-threshold error exceeds certificate')
    row = dict(actual_sse=actual, target_energy=float(truth@truth), actual_nmse_db=db(actual, truth@truth),
        old_bound_sse=old, joint_bound_sse=new,
        old_to_actual_sse=ratio(old, actual), joint_to_actual_sse=ratio(new, actual),
        joint_over_old=ratio(new, old), reduction_percent=100*(1-new/old) if old > 0 else 0.,
        witness_sse=witness_sse, joint_to_witness_sse=ratio(new, witness_sse),
        original_coordinate_old_sse=bounds['direct']['old'],
        original_coordinate_raw_secant_sse=bounds['direct']['raw_joint'],
        basis_old_sse=bounds['old_basis'], basis_joint_sse=bounds['joint_basis'],
        basis_raw_secant_projected_sse=bounds['reduced']['raw_joint'],
        coordinate_interval_width_slack=bounds['direct']['interval_width_slack'],
        basis_interval_width_slack=bounds['reduced']['interval_width_slack'],
        max_secant_solver_gap=max(bounds['direct']['joint_solver_gap'], bounds['reduced']['joint_solver_gap']),
        max_endpoint_solver_gap=max(bounds['direct']['endpoint_gap_max'], bounds['reduced']['endpoint_gap_max']),
        output_threshold=threshold, final_actual_sse=final_actual,
        final_old_bound_sse=final_old, final_joint_bound_sse=final_new,
        affine_identity_max_abs=identity_error, upper_bound_violation=False,
        witness_same_pattern=True, support_size=region.s, basis_dimensions=bounds['basis'].shape[1],
        endpoint_cache_reused=bounds['direct']['cache_reused'], optimization_calls=region.calls,
        extra_secant_seconds=bounds['direct']['secant_seconds']+bounds['reduced']['secant_seconds'],
        seconds=time.perf_counter()-start)
    arrays = dict(H=H, b=b, J=J, support=support, reference=anchor, witness=witness,
        R=region.R, d=region.d, B=np.array(region.B), spikes=trace['spikes'],
        effective_feedback=trace['shifts'], theta=trace['theta'], basis=bounds['basis'],
        reduced_H=bounds['reduced_H'], reduced_b=bounds['reduced_b'])
    for label, result in [('coordinate', bounds['direct']), ('basis', bounds['reduced'])]:
        for name in ('signed', 'joint_h', 'joint_c', 'joint_lam', 'objective_conversion_guard'):
            arrays[label+'_'+name] = result[name]
    return row, arrays


def audit_frame(P, A, G, theta, z_true, B, previous, rho, noise, delta, threshold, args):
    P, A, G, z_true, previous, noise = [np.asarray(x, float) for x in (P, A, G, z_true, previous, noise)]
    n, m = P.shape
    K = len(G)+1
    if A.shape != (m, n) or G.shape != (K-1, n, n) or z_true.shape != (n,) or noise.shape != (m,):
        raise ValueError('Incompatible frame array shapes')
    if any(not np.isfinite(x).all() for x in (P, A, G, z_true, previous, noise)):
        raise ValueError('Nonfinite frame data')
    if not (np.isfinite(rho) and np.isfinite(delta) and delta >= 0 and threshold >= 0 and np.isfinite(threshold)):
        raise ValueError('Invalid rho/noise radius/final threshold')
    if np.linalg.norm(noise) > delta+1e-12:
        raise ValueError('Observed noise exceeds the declared radius')
    support = np.flatnonzero(z_true)
    J = P@A[:, support]
    anchor = z_true[support]
    if delta > 0:
        J = np.c_[J, delta*P]
        anchor = np.r_[anchor, noise/delta]
    trace = replay(J@anchor, G, theta, previous, rho)
    H = trace['mask'][:, None]*J/trace['theta'][-1, :, None]
    H[support, np.arange(len(support))] -= 1
    b = trace['mask']*(trace['q']-trace['shifts'][-1]/trace['theta'][-1])
    R, d = build_region(J, trace, B, len(support), args.guard)
    region = Region(R, d, B, len(support), args.guard, args.lp_seconds)
    row, arrays = audit_region(H, b, J, support, anchor, trace, region, threshold)
    # Independent sequential replay of the witness, including all feedback matrices.
    witness_trace = replay(J@arrays['witness'], G, theta, previous, rho)
    witness_truth = np.zeros(n); witness_truth[support] = arrays['witness'][:len(support)]
    if not np.array_equal(witness_trace['spikes'], trace['spikes']) or not np.isclose(
            np.sum((witness_trace['output']-witness_truth)**2), row['witness_sse'], rtol=1e-8, atol=1e-8):
        raise AssertionError('Independent sequential witness replay failed')
    row.update(noise_radius=delta, noise_norm=float(np.linalg.norm(noise)), rho=float(rho),
               nonzero_previous_membrane=bool(np.any(previous)), depth=K,
               sequential_witness_replay=True)
    arrays.update(previous_membrane=previous, rho=np.array(rho), noise=noise,
                  noise_radius=np.array(delta), output=trace['output'], states=trace['states'])
    return row, arrays, trace


def resolve_module(root, name):
    exact = root/(name+'.py')
    choices = [exact] if exact.is_file() else sorted(root.glob(name+'(*).py'))
    if len(choices) != 1:
        raise ValueError(f'Need exactly one {name}.py (or uploaded-name variant) in {root}')
    spec = importlib.util.spec_from_file_location(name, choices[0])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module, choices[0]


def matches(condition, depth, args):
    return (not args.conditions or condition in args.conditions) and (not args.depths or depth in args.depths)


def checkpoint_samples(path, args, modules, cfg):
    import torch
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if 'protocol' in payload:
        cfg = modules['config'].Protocol(**payload['protocol'])
    job = payload['job']
    if job['method'] != 'slista' or int(job['time_steps']) != 1:
        raise ValueError('Only T=1 S-LISTA toy checkpoints are supported')
    if 'output_threshold' in payload['model']:
        raise NotImplementedError('Incompatible checkpoint: output_threshold is not supported by this model.')
    condition, depth = job['condition'], int(job['depth'])
    if not matches(condition['id'], depth, args):
        return
    data, models = modules['data'], modules['models']
    device = torch.device(args.device)
    A = data.measurement_matrix(cfg.n, int(condition['m']), cfg.max_m, cfg.matrix_seed, device)
    if args.matrix_file:
        A = torch.as_tensor(np.load(args.matrix_file, allow_pickle=False), dtype=A.dtype, device=device)
    if tuple(A.shape) != (int(condition['m']), cfg.n):
        raise ValueError('Measurement matrix has wrong dimensions')
    model = models.build_model('slista', A, depth, 1, job['parameters']).to(device)
    model.load_state_dict(payload['model'], strict=True)
    model.eval()
    if any(not torch.isfinite(v).all() for v in model.state_dict().values()):
        raise ValueError('Nonfinite model parameter')
    to_np = lambda x: x.detach().cpu().double().numpy()
    P, A64 = to_np(model.P.weight), to_np(A)
    checkpoint_hash = sha(path)
    matrix_hash = hashlib.sha256(A64.tobytes()).hexdigest()
    G = np.asarray([to_np(layer.weight) for layer in model.G]).reshape(depth-1, cfg.n, cfg.n)
    theta = float(model.theta_snn)
    if theta < 1e-6:
        raise ValueError('Threshold clamp case is excluded')
    rng = data.generator(args.seed if args.seed is not None else cfg.test_seed, device)
    noise_rng = np.random.default_rng(args.noise_seed)
    batch_size = args.batch_size or cfg.eval_batch_size
    sample = 0
    with torch.no_grad():
        while sample < args.samples:
            # Preserve the original full RNG batch shape, even when auditing 1 sample.
            targets, observations = data.sparse_batch(batch_size, int(condition['s']), A, rng,
                                                      cfg.amplitude_min, cfg.amplitude_max)
            count = min(batch_size, args.samples-sample)
            noises = np.zeros((count, A.shape[0]))
            if args.noise_radius > 0:
                noises = noise_rng.normal(size=noises.shape)
                noises /= np.linalg.norm(noises, axis=1, keepdims=True)
                noises *= args.noise_radius*noise_rng.random((count, 1))**(1/A.shape[0])
            inputs = observations[:count]+torch.as_tensor(noises, dtype=observations.dtype, device=device)
            native, _ = model(inputs)
            native = soft(to_np(native), args.post_threshold)
            current = model.P(inputs)
            q = torch.zeros_like(current)
            native_spikes = []
            for k in range(depth):
                pre = current if k == 0 else current-model.G[k-1](q)
                spike = (pre >= model.theta_snn).to(pre.dtype)-(pre <= -model.theta_snn).to(pre.dtype)
                q += spike
                native_spikes.append(to_np(spike))
            for j in range(count):
                truth = to_np(targets[j])
                print(f'  {condition["id"]} L={depth}: sample {sample+1}/{args.samples}', flush=True)
                row, arrays, trace = audit_frame(P, A64, G, theta, truth, cfg.amplitude_max,
                    np.zeros((depth, cfg.n)), float(model.decoder_tau), noises[j],
                    args.noise_radius, args.post_threshold, args)
                row.update(checkpoint=str(path), condition=condition['id'], sample=sample,
                    checkpoint_sha256=checkpoint_hash, matrix_sha256=matrix_hash,
                    generated_batch_size=batch_size,
                    test_seed=args.seed if args.seed is not None else cfg.test_seed,
                    mode='toy_T1', original_float32_sse=float(np.sum((native[j]-truth)**2)),
                    original_float32_same_spike_pattern=bool(np.array_equal(
                        np.asarray(native_spikes)[:, j], trace['spikes'])),
                    original_float32_vs_float64_max_abs=float(np.max(np.abs(
                        native[j]-soft(trace['output'], args.post_threshold)))))
                yield row, arrays
                sample += 1


def cached_sample(path, args):
    with np.load(path, allow_pickle=False) as file:
        cache = {key: file[key] for key in file.files}
    needed = ['H', 'b', 'W', 'B', 'theta', 'reference_x', 'support', 'spikes', 'feedbacks',
              'q', 'mask', 'reference_output', 'full_R', 'full_d', 'full_signed_upper']
    if any(key not in cache for key in needed):
        raise ValueError('Not an old audit_slista_pattern_bounds.py certificate archive')
    metadata_path = path.with_name('metrics.json')
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    condition = metadata.get('condition', 'unknown')
    depth = int(metadata.get('depth', len(cache['spikes'])))
    if not matches(condition, depth, args):
        return None
    support = cache['support'].astype(int)
    trace = dict(q=cache['q'], mask=cache['mask'], output=cache['reference_output'],
        shifts=cache['feedbacks'], spikes=cache['spikes'],
        theta=thresholds(cache['theta'], depth, len(cache['b'])))
    region = Region(cache['full_R'], cache['full_d'], float(cache['B']), len(support), args.guard, args.lp_seconds)
    row, arrays = audit_region(cache['H'], cache['b'], cache['W'], support, cache['reference_x'],
                               trace, region, 0., cache)
    row.update(mode='cached_toy_T1', source_certificate=str(path),
        checkpoint=metadata.get('checkpoint', str(path.parent.parent)), condition=condition,
        depth=depth, sample=metadata.get('sample', path.parent.name),
        noise_radius=0., nonzero_previous_membrane=False,
        original_float32_same_spike_pattern=metadata.get('original_float32_same_spike_pattern'),
        original_float32_sse=metadata.get('original_float32_sse'),
        previous_reported_old_bound_sse=metadata.get('full_path_bound_sse'))
    if row['previous_reported_old_bound_sse'] is not None and not np.isclose(
            row['old_bound_sse'], row['previous_reported_old_bound_sse'], rtol=1e-6, atol=1e-8):
        raise AssertionError('Recomputed old bound does not match saved audit result')
    return row, arrays


def frame_sample(path, args):
    with np.load(path, allow_pickle=False) as file:
        a = {key: file[key] for key in file.files}
    required = ['P', 'A', 'G', 'theta', 'z_true', 'B', 'previous_membrane', 'rho']
    if any(key not in a for key in required):
        raise ValueError('Frame NPZ missing keys: '+', '.join(key for key in required if key not in a))
    row, arrays, _ = audit_frame(a['P'], a['A'], a['G'], a['theta'], a['z_true'], float(a['B']),
        a['previous_membrane'], float(a['rho']), a.get('noise', np.zeros(a['A'].shape[0])),
        float(a.get('noise_radius', 0.)), float(a.get('output_threshold', 0.)), args)
    row.update(mode='state_conditional_frame', checkpoint=str(path), condition='frame', sample=0)
    return row, arrays


def summarize(rows):
    groups = {}
    for row in rows:
        key = (row['checkpoint'], row['mode'], row.get('noise_radius', 0.), row['output_threshold'])
        groups.setdefault(key, []).append(row)
    result = []
    for group in groups.values():
        total = lambda key: float(sum(row[key] for row in group))
        actual, energy = total('actual_sse'), total('target_energy')
        old, new, witness = total('old_bound_sse'), total('joint_bound_sse'), total('witness_sse')
        result.append(dict(checkpoint=group[0]['checkpoint'], condition=group[0]['condition'],
            depth=group[0]['depth'], mode=group[0]['mode'], samples=len(group),
            noise_radius=group[0].get('noise_radius', 0.), output_threshold=group[0]['output_threshold'],
            actual_nmse_db=db(actual, energy), old_to_actual_sse=ratio(old, actual),
            joint_to_actual_sse=ratio(new, actual), joint_over_old=ratio(new, old),
            reduction_percent=100*(1-new/old) if old > 0 else 0.,
            joint_to_witness_sse=ratio(new, witness), witness_to_actual_sse=ratio(witness, actual),
            old_bound_db_on_reference_energy=db(old, energy),
            joint_bound_db_on_reference_energy=db(new, energy),
            final_actual_nmse_db=db(total('final_actual_sse'), energy),
            final_joint_to_actual_sse=ratio(total('final_joint_bound_sse'), total('final_actual_sse')),
            upper_bound_violations=sum(row['upper_bound_violation'] for row in group),
            improved_samples=sum(row['joint_bound_sse'] < row['old_bound_sse']*(1-1e-6) for row in group),
            original_float32_path_mismatches=sum(row.get('original_float32_same_spike_pattern') is False for row in group),
            optimization_calls=sum(row['optimization_calls'] for row in group),
            extra_secant_seconds=total('extra_secant_seconds'), seconds=total('seconds')))
    return result


def self_test():
    # Joint maxima matter: on x>=0,y>=0,x+y<=1, old=2, joint=true worst=1.
    R, d = np.array([[-1., 0.], [0., -1.], [1., 1.]]), np.array([0., 0., 1.])
    reg = Region(R, d, 1., 2)
    answer = coordinate_and_joint(np.eye(2), np.zeros(2), reg)
    assert np.isclose(answer['old'], 2., atol=1e-7)
    assert np.isclose(answer['joint'], 1., atol=1e-7)
    # Enumerate vertices of random 2D bounded regions to know the exact worst SSE.
    rng = np.random.default_rng(925)
    for _ in range(20):
        R = np.r_[np.eye(2), -np.eye(2), rng.normal(size=(4, 2))]
        d = np.r_[np.ones(4), rng.uniform(.2, 1., 4)]
        H, b = rng.normal(size=(4, 2)), rng.normal(size=4)
        reg = Region(R, d, 1., 2)
        answer = coordinate_and_joint(H, b, reg)
        vertices = []
        for i in range(len(d)):
            for j in range(i):
                matrix = R[[i, j]]
                if abs(np.linalg.det(matrix)) < 1e-10:
                    continue
                x = np.linalg.solve(matrix, d[[i, j]])
                if np.all(R@x <= d+1e-9):
                    vertices.append(x)
        exact = max(float(np.sum((H@v+b)**2)) for v in vertices)
        assert exact <= answer['joint']+1e-7
        assert answer['joint'] <= answer['old']+1e-7
        assert answer['joint']-exact <= answer['interval_width_slack']+answer['joint_solver_gap']+1e-7
    # Nonzero, genuinely carried membrane states across successive frames.
    args = argparse.Namespace(guard=1e-10, lp_seconds=30.)
    n, m, K = 5, 3, 3
    A, P = rng.normal(size=(m, n)), rng.normal(size=(n, m))*.4
    G = rng.normal(size=(K-1, n, n))*.15
    previous = np.zeros((K, n))
    for t in range(3):
        z = np.array([1.2, 0., -.7, 0., 0.])
        row, _, trace = audit_frame(P, A, G, np.array([.7, 1.1, .9]), z, 2.,
            previous, .8, np.zeros(m), 0., .1, args)
        assert not row['upper_bound_violation']
        assert row['nonzero_previous_membrane'] == (t > 0)
        previous = trace['states']
    try:
        import clarabel  # noqa: F401
    except ImportError:
        print('Noise SOCP self-test skipped: clarabel is not installed.')
    else:
        row, _, _ = audit_frame(P, A, G, 1., z, 2., previous, .8,
            np.array([.03, -.02, .01]), .1, .1, args)
        assert not row['upper_bound_violation']
        print('Noise-ball + nonzero-memory + soft-threshold check passed.')
    print('PASS: exact triangle example, 20 exact vertex maxima/gaps, 3 carried-state frames, soft threshold.')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sources', nargs='+')
    parser.add_argument('--checkpoints', nargs='+')
    parser.add_argument('--certificates', nargs='+', help='Old certificate.npz file(s) or audit directories')
    parser.add_argument('--frame-npz', nargs='+', help='Single-frame array exports including actual previous membrane')
    parser.add_argument('--project-dir', type=Path, default=Path('.'))
    parser.add_argument('--conditions', nargs='+')
    parser.add_argument('--depths', nargs='+', type=int)
    parser.add_argument('--samples', type=int, default=10, help='Per checkpoint; cache mode uses all matching saved samples')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--noise-seed', type=int, default=20260828)
    parser.add_argument('--noise-radius', type=float, default=0.)
    parser.add_argument('--post-threshold', type=float, default=0.)
    parser.add_argument('--matrix-file', type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--guard', type=float, default=1e-10)
    parser.add_argument('--lp-seconds', type=float, default=30.)
    parser.add_argument('--output', type=Path, default=Path('joint_bound_audit'))
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.samples <= 0 or args.threads <= 0 or args.guard < 0 or args.lp_seconds <= 0:
        parser.error('Samples/threads/time must be positive; guard must be nonnegative')
    if any(not math.isfinite(v) for v in (args.guard, args.lp_seconds, args.noise_radius, args.post_threshold)):
        parser.error('Numerical arguments must be finite')
    if args.noise_radius < 0 or args.post_threshold < 0 or (args.batch_size is not None and args.batch_size <= 0):
        parser.error('Invalid noise radius, post-threshold, or batch size')
    if args.self_test:
        self_test()
    modes = sum(bool(x) for x in (args.sources or args.checkpoints, args.certificates, args.frame_npz))
    if modes == 0 and args.self_test:
        return
    if modes != 1:
        parser.error('Choose one mode: checkpoints (--sources/--checkpoints), --certificates, or --frame-npz')
    if (args.certificates or args.frame_npz) and (args.noise_radius or args.post_threshold):
        parser.error('Noise/final threshold overrides only apply to checkpoint mode; frame exports carry their own values')
    paths, source_files = [], {}
    modules = {}
    if args.certificates:
        for source in args.certificates:
            path = Path(source).expanduser().resolve()
            if not path.exists():
                parser.error('Missing certificate source: '+str(path))
            paths.extend(sorted(path.rglob('certificate.npz')) if path.is_dir() else [path])
    elif args.frame_npz:
        paths = [Path(p).expanduser().resolve() for p in args.frame_npz]
    else:
        import torch
        torch.set_num_threads(args.threads)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        for name in ('config', 'data', 'models'):
            module, path = resolve_module(args.project_dir.expanduser().resolve(), name)
            modules[name] = module
            source_files[name] = dict(path=str(path), sha256=sha(path))
        cfg = modules['config'].Protocol()
        paths = [Path(p).expanduser().resolve() for p in args.checkpoints or []]
        regex = re.compile(r'(.+)__slista_l(\d+)_t1\.pth$')
        for source in args.sources or []:
            path = Path(source).expanduser().resolve()
            if not path.is_dir():
                parser.error('Missing checkpoint source: '+str(path))
            for checkpoint in sorted(path.rglob('*__slista_l*_t1.pth')):
                match = regex.fullmatch(checkpoint.name)
                if match and matches(match[1], int(match[2]), args):
                    paths.append(checkpoint)
    paths = list(dict.fromkeys(paths))
    if not paths:
        parser.error('No matching inputs found')
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out = args.output.expanduser().resolve()/stamp
    out.mkdir(parents=True, exist_ok=False)
    rows, errors, skipped = [], [], []

    def save():
        summaries = summarize(rows)
        write_csv(out/'samples.csv', rows)
        write_csv(out/'summary.csv', summaries)
        write_json(out/'report.json', dict(summaries=summaries, errors=errors, skipped=skipped,
            total_sample_evaluations=len(rows), source_files=source_files,
            arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            scope='Known-support/prior, pulse- and state-conditional SSE certificate; float64 replay; no convergence claim.',
            notes=['A candidate generally has a different measurement with the same pulses; exact observed measurement equality is not imposed.',
                   'Reference amplitudes are used for error/feasibility and witnesses, never to fit the upper bound.',
                   'Joint bounds retain the old bound as a valid numerical fallback; raw secant results are also saved.',
                   'Width slack refers to exact secant relaxation in the indicated basis, not upper-minus-reference error.',
                   'Native float32 outputs are not automatically certified by the float64 bound.',
                   'No trained streaming performance is inferred from toy T=1 or synthetic self-tests.']))

    def record(result):
        if result is None:
            return
        row, arrays = result
        index = len(rows)
        folder = out/f'sample_{index:05d}'
        folder.mkdir()
        np.savez_compressed(folder/'joint_certificate.npz', **arrays)
        row['artifact'] = str(folder)
        write_json(folder/'metrics.json', row)
        rows.append(row)
        save()
        print(f'    old/actual={row["old_to_actual_sse"]}; joint/actual={row["joint_to_actual_sse"]}; '
              f'reduction={row["reduction_percent"]:.2f}%; joint/witness={row["joint_to_witness_sse"]}', flush=True)

    print(f'Found {len(paths)} input(s). Bounds run on CPU; output: {out}', flush=True)
    for path in paths:
        try:
            if args.certificates:
                print('  Reusing '+str(path), flush=True)
                record(cached_sample(path, args))
            elif args.frame_npz:
                record(frame_sample(path, args))
            else:
                for result in checkpoint_samples(path, args, modules, cfg):
                    record(result)
        except NotImplementedError as exc:
            skipped.append(dict(path=str(path), reason=str(exc)))
            print('SKIPPED: '+str(path)+': '+str(exc), flush=True)
        except Exception as exc:
            errors.append(dict(path=str(path), error=f'{type(exc).__name__}: {exc}'))
            print('FAILED: '+str(path)+': '+str(exc), file=sys.stderr, flush=True)
        save()
    print(f'Done: {len(rows)} sample evaluations, {len(errors)} failed inputs, {len(skipped)} skipped.\nAudit output: {out}', flush=True)
    if errors or not rows:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
