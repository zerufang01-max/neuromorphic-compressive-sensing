#!/usr/bin/env python3
"""Fixed-weight sparsity sweep for S-LISTA firing/feedback activity bounds.

Run inside the original toy project. This file is standalone; no earlier audit
script is required. It does NOT train, tune thresholds, or process T=4 weights.

  python audit_slista_activity.py --sources outputs/seed42/checkpoints \
    --conditions m141_s28 --depths 5 25 --samples 1000 --device cuda:0

Every selected checkpoint stays FIXED across the complete test-sparsity grid.
--conditions selects TRAINING conditions/checkpoints; --sparsities selects TEST
sparsities. Default: 0 1 2 4 7 14 28 42 56. All nonzero amplitudes retain the
original configured distribution; there is no fixed-total-energy normalization.
The original data.sparse_batch generator is reused, with the original full batch
size. Each sparsity restarts the same declared seed for comparisons across
checkpoints. Different sparsities are not claimed to be paired/nested signals.

MATHEMATICS (one frame, q[0]=0, xi[k] in {-1,0,1}, q[k]=q[k-1]+xi[k])
Let S be the true support, s=|S|, N=code dimension and L=depth.
F=sum_k ||xi[k]||_1; r=F/(2*N*L). Both signed channels are counted, max r=0.5.
F <= L*s + 2*sum_{k=1}^{L-1} ||q[k]_{S^c}||_2^2 + ||q[L]_{S^c}||_2^2.
The toy feedback counter C=N*sum_{k=1}^{L-1} ||q[k]||_0 satisfies
C <= N*((L-1)*s + sum_{k=1}^{L-1} ||q[k]_{S^c}||_2^2).
Both bounds are reported raw AND capped by the trivial maxima N*L and N*N*(L-1).
The cap must not hide an uninformative raw bound; ratios of both are in CSV.

r/(s/(2N)) = F_on/(L*s) + F_off/(L*s), for s>0.
Thus absence of off-support activity implies r<=s/(2N), NOT equality: equality
also needs every true-support coordinate to spike at every layer. The script
reports the two contributions, ratios and absolute differences, without claiming
an asymptotic limit from a finite integer sparsity sweep. s=0 ratios are null.

Counts and their slack decomposition are checked per sample with integer
arithmetic. Native model output, model.last_firing_rate and model.last_ac_count
are compared against an independent native-dtype replay. The legacy feedback_ac fields count active cumulative connections (L0),
not energy ACs. feedback_magnitude_ac counts integer magnitudes (L1) and is
compared with the current native model counter. Neither is
not measured hardware instructions/latency or a full-system energy model.
Dense embedding cost and dictionary readout are excluded from this feedback
counter. Noise and streaming memories are not exercised by this T=1 sweep.
The counting lemma itself does apply framewise with memory if q resets per frame.

Outputs: summary.csv (one checkpoint/test-s pair), report.json, layers.csv,
per-checkpoint samples.csv, and per-checkpoint activity.pdf/activity.png.
Use --no-plots if matplotlib is unavailable. Plotting errors are recorded.
Dependencies: torch, numpy; matplotlib for plots. --self-test needs only numpy.
Legacy output_threshold checkpoints are skipped explicitly: no state key or
unknown readout semantics is silently discarded.
"""

import argparse
import csv
import datetime as dt
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import re
import sys
import time

import numpy as np


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def ratio(a, b):
    return float(a/b) if b > 0 else None


def nmse(sse, energy):
    return float(10*np.log10(max(sse/energy, 1e-300))) if energy > 0 else None


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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


def count_paths(spikes, true_support):
    """Pure NumPy kernel: spikes shape (L,B,N), support shape (B,N)."""
    spikes = np.asarray(spikes)
    if spikes.ndim != 3 or not np.isin(spikes, (-1, 0, 1)).all():
        raise ValueError('Expected ternary spikes with shape (L,batch,N)')
    spikes = spikes.astype(np.int64)
    L, batch, N = spikes.shape
    support = np.asarray(true_support, bool)
    if support.shape != (batch, N):
        raise ValueError('Support shape mismatch')
    q = np.cumsum(spikes, axis=0, dtype=np.int64)
    on = support[None, :, :]
    off = ~on
    event = np.abs(spikes)
    s = support.sum(axis=1, dtype=np.int64)
    on_spikes = (event*on).sum(axis=(0, 2))
    off_spikes = (event*off).sum(axis=(0, 2))
    off_energy = (q*q*off).sum(axis=2)
    off_middle_energy = off_energy[:-1].sum(axis=0)
    off_bound = 2*off_middle_energy+off_energy[-1]
    actual_F = on_spikes+off_spikes
    bound_F = L*s+off_bound
    on_active = ((q != 0)&on).sum(axis=2)
    off_active = ((q != 0)&off).sum(axis=2)
    ac_on = N*on_active[:-1].sum(axis=0)
    ac_off = N*off_active[:-1].sum(axis=0)
    ac = ac_on+ac_off
    ac_bound = N*((L-1)*s+off_middle_energy)
    if np.any(actual_F > bound_F) or np.any(ac > ac_bound):
        raise AssertionError('Activity inequality violated')
    if np.any(on_spikes > L*s) or np.any(off_spikes > off_bound):
        raise AssertionError('Support-part counting inequality violated')
    on_unused = L*s-on_spikes
    off_slack = off_bound-off_spikes
    if not np.array_equal(bound_F-actual_F, on_unused+off_slack):
        raise AssertionError('Firing-bound slack decomposition failed')
    off_net = (np.abs(q[-1])*~support).sum(axis=1)
    if np.any((off_spikes-off_net) % 2):
        raise AssertionError('Signed cancellation parity failed')
    sample = dict(s_actual=s, spike_count=actual_F, on_support_spikes=on_spikes,
        off_support_spikes=off_spikes, spike_bound_raw=bound_F,
        spike_bound_capped=np.minimum(bound_F, N*L),
        off_support_count_bound=off_bound,
        on_support_unused_spike_capacity=on_unused, off_support_bound_slack=off_slack,
        off_support_middle_energy=off_middle_energy, off_support_final_energy=off_energy[-1],
        off_support_cancellation_pairs=(off_spikes-off_net)//2,
        ever_off_support=np.any((q != 0)&off, axis=(0, 2)).astype(int),
        feedback_magnitude_ac=N*np.abs(q[:-1]).sum(axis=(0, 2)),
        feedback_ac=ac, feedback_ac_on=ac_on, feedback_ac_off=ac_off,
        feedback_ac_bound_raw=ac_bound, feedback_ac_bound_capped=np.minimum(ac_bound,N*N*(L-1)))
    layer = dict(on_spikes=(event*on).sum(axis=2), off_spikes=(event*off).sum(axis=2),
        on_cumulative_active=on_active, off_cumulative_active=off_active,
        off_cumulative_energy=off_energy)
    return sample, layer


def native_trace(model, measurements):
    """Replay the uploaded toy's T=1 forward; return output and ternary path."""
    import torch
    initial = model.P(measurements)
    q = torch.zeros_like(initial)
    path = []
    for k in range(model.depth):
        pre = initial if k == 0 else initial-model.G[k-1](q)
        spike = (pre >= model.theta_snn).to(pre.dtype)-(pre <= -model.theta_snn).to(pre.dtype)
        q = q+spike
        membrane = pre-model.theta_snn*spike
        path.append(spike)
    # Same operation order/clamp as original toy; no soft threshold is inserted.
    output = q+(q != 0).to(q.dtype)*(membrane/model.theta_snn.clamp_min(1e-6))
    return output, torch.stack(path)


def summarize(samples, metadata, s_requested, L, N):
    n = len(samples)
    total = lambda key: sum(row[key] for row in samples)
    mean = lambda key: float(total(key)/n)
    F, on, off, sum_s = [total(key) for key in ('spike_count','on_support_spikes','off_support_spikes','s_actual')]
    Braw, Bcap = total('spike_bound_raw'), total('spike_bound_capped')
    AC = total('feedback_ac')
    rate_scale = 2*N*L*n
    ideal_count = L*sum_s
    ideal_ac = N*(L-1)*sum_s
    normalized = ratio(F, ideal_count)
    inside = ratio(on, ideal_count)
    outside = ratio(off, ideal_count)
    if normalized is not None and not math.isclose(normalized, inside+outside, abs_tol=1e-12):
        raise AssertionError('Normalized-rate decomposition failed')
    no_off = [row for row in samples if row['ever_off_support'] == 0]
    zero_events = sum(row['spike_count'] == 0 for row in samples)
    row = dict(**metadata, test_s=s_requested, samples=n, actual_s_min=min(r['s_actual'] for r in samples),
        actual_s_max=max(r['s_actual'] for r in samples),
        firing_rate=F/rate_scale, firing_rate_std=float(np.std([r['spike_count']/(2*N*L) for r in samples],ddof=1)) if n>1 else 0.,
        firing_ideal=ideal_count/rate_scale, firing_bound_raw=Braw/rate_scale,
        firing_bound_capped=Bcap/rate_scale,
        firing_raw_bound_to_actual=ratio(Braw,F), firing_capped_bound_to_actual=ratio(Bcap,F),
        firing_over_ideal=normalized,
        firing_minus_ideal=(F-ideal_count)/rate_scale,
        firing_abs_relative_distance_from_ideal=abs(normalized-1) if normalized is not None else None,
        on_support_firing_utilization=inside, off_support_spikes_over_Ls=outside,
        off_support_spike_fraction=ratio(off,F),
        firing_bound_correction_over_ideal=ratio(total('off_support_count_bound'),ideal_count),
        on_support_unused_spike_capacity_mean=mean('on_support_unused_spike_capacity'),
        off_support_bound_slack_mean=mean('off_support_bound_slack'),
        firing_cap_used_fraction=sum(r['spike_bound_raw']>N*L for r in samples)/n,
        no_off_support_activity_fraction=len(no_off)/n,
        no_off_support_firing_over_ideal=ratio(sum(r['spike_count'] for r in no_off),L*sum(r['s_actual'] for r in no_off)),
        zero_spike_fraction=zero_events/n,
        off_support_cancellation_event_fraction=ratio(2*total('off_support_cancellation_pairs'),off),
        feedback_ac_mean=AC/n, feedback_ac_on_mean=mean('feedback_ac_on'),
        feedback_ac_off_mean=mean('feedback_ac_off'), feedback_ac_ideal=ideal_ac/n,
        feedback_ac_bound_raw=mean('feedback_ac_bound_raw'), feedback_ac_bound_capped=mean('feedback_ac_bound_capped'),
        feedback_ac_raw_bound_to_actual=ratio(total('feedback_ac_bound_raw'),AC),
        feedback_ac_capped_bound_to_actual=ratio(total('feedback_ac_bound_capped'),AC),
        feedback_ac_over_ideal=ratio(AC,ideal_ac), feedback_ac_off_fraction=ratio(total('feedback_ac_off'),AC),
        actual_nmse_db=nmse(total('sse'),total('target_energy')),
        support_recall=ratio(total('true_positive_count'),sum_s),
        support_precision=ratio(total('true_positive_count'),total('predicted_support_count')),
        support_false_positive_mean=mean('false_positive_count'),
        firing_bound_violations=0, feedback_bound_violations=0,
        native_output_max_abs=max(r['native_output_max_abs'] for r in samples),
        native_firing_max_abs=max(r['native_firing_abs_diff'] for r in samples),
        native_ac_max_abs=max(r['native_ac_abs_diff'] for r in samples))
    return row


def plot_checkpoint(rows, out, font_size):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font = 'DejaVu Sans'
    for requested in ('Aptos','Arial','DejaVu Sans'):
        try:
            font_manager.findfont(requested, fallback_to_default=False)
            font = requested
            break
        except ValueError:
            pass
    plt.rcParams.update({'font.family':font,'font.size':font_size,
        'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,
        'ps.fonttype':42,'legend.frameon':False,'axes.linewidth':.8})
    rows = sorted(rows, key=lambda r:r['test_s'])
    positions = np.arange(len(rows))
    x = np.array([r['test_s'] for r in rows])
    get = lambda key: np.array([np.nan if r[key] is None else r[key] for r in rows],float)
    fig, axes = plt.subplots(2,2,figsize=(12.2,8.3),layout='constrained')
    blue, orange, green = '#0072B2','#D55E00','#009E73'
    a,b,c,d = axes.ravel()
    a.plot(x,get('firing_rate'),'-o',color=blue,label='Observed rate',ms=4)
    a.plot(x,get('firing_bound_capped'),'--s',color=orange,label='Bound, capped at 0.5',ms=4)
    a.plot(x,get('firing_ideal'),':',color='black',label=r'$s/(2N)$',lw=2)
    a.set(xlabel=r'Test sparsity $s$',ylabel='Firing rate',ylim=(0,None))
    a.legend(fontsize=font_size-2)
    positive = x>0
    b.plot(x[positive],get('firing_over_ideal')[positive],'-o',color='black',label='Total',ms=4)
    b.plot(x[positive],get('on_support_firing_utilization')[positive],'-',color=blue,label='On support')
    b.plot(x[positive],get('off_support_spikes_over_Ls')[positive],'--',color=orange,label='Off support')
    b.axhline(1,color='#888888',ls=':',lw=1)
    b.set(xlabel=r'Test sparsity $s$',ylabel=r'Rate / $[s/(2N)]$',ylim=(0,None))
    b.legend(fontsize=font_size-2)
    c.bar(positions,get('feedback_ac_on_mean')/1e3,color=blue,label='On support')
    c.bar(positions,get('feedback_ac_off_mean')/1e3,bottom=get('feedback_ac_on_mean')/1e3,color=orange,label='Off support')
    c.plot(positions,get('feedback_ac_ideal')/1e3,'k--',label=r'$N(L-1)s$')
    c.set_xticks(positions,[str(v) for v in x])
    c.set(xlabel=r'Test sparsity $s$',ylabel='Active feedback connections (thousands)')
    c.legend(fontsize=font_size-2)
    d.plot(x,get('firing_raw_bound_to_actual'),'-o',color=blue,label='Firing bound',ms=4)
    d.plot(x,get('feedback_ac_raw_bound_to_actual'),'-s',color=green,label='Feedback bound',ms=4)
    d.axhline(1,color='#888888',ls=':',lw=1)
    d.set(xlabel=r'Test sparsity $s$',ylabel='Raw bound / observed count')
    finite = np.r_[get('firing_raw_bound_to_actual'),get('feedback_ac_raw_bound_to_actual')]
    finite = finite[np.isfinite(finite)]
    if finite.size and finite.max()>20:
        d.set_yscale('log')
    d.legend(fontsize=font_size-2)
    for letter,ax in zip('abcd',axes.ravel()):
        ax.text(-.12,1.04,letter,transform=ax.transAxes,fontweight='bold',fontsize=font_size+2)
        ax.grid(axis='y',alpha=.2,lw=.5)
        ax.set_axisbelow(True)
    fig.savefig(out/'activity.png',dpi=200)
    fig.savefig(out/'activity.pdf')
    plt.close(fig)


def audit_checkpoint(path, args, modules, cfg, root, save_callback):
    import torch
    payload = torch.load(path,map_location='cpu',weights_only=True)
    if 'protocol' in payload:
        cfg = modules['config'].Protocol(**payload['protocol'])
    job = payload['job']
    if job['method']!='slista' or int(job['time_steps'])!=1:
        raise NotImplementedError('Only S-LISTA T=1 checkpoints are supported; no T=4 replay.')
    if 'output_threshold' in payload['model']:
        raise NotImplementedError('Incompatible checkpoint: output_threshold is not supported by this model.')
    condition, L = job['condition'],int(job['depth'])
    if (args.conditions and condition['id'] not in args.conditions) or (args.depths and L not in args.depths):
        return None
    N = int(cfg.n)
    if any(s<0 or s>N for s in args.sparsities):
        raise ValueError(f'Every requested sparsity must be between 0 and N={N}')
    device = torch.device(args.device)
    data,models = modules['data'],modules['models']
    A = data.measurement_matrix(N,int(condition['m']),cfg.max_m,cfg.matrix_seed,device)
    if args.matrix_file:
        A = torch.as_tensor(np.load(args.matrix_file,allow_pickle=False),dtype=A.dtype,device=device)
    if tuple(A.shape)!=(int(condition['m']),N) or not torch.isfinite(A).all():
        raise ValueError('Invalid measurement matrix')
    model = models.build_model('slista',A,L,1,job['parameters']).to(device)
    model.load_state_dict(payload['model'],strict=True)
    model.eval()
    if int(model.time_steps)!=1 or int(model.depth)!=L:
        raise ValueError('Unexpected architecture')
    if any(not torch.isfinite(v).all() for v in model.state_dict().values()) or float(model.theta_snn)<=0:
        raise ValueError('Nonfinite parameters or nonpositive threshold')
    checkpoint_hash = sha(path)
    key = f'{condition["id"]}__l{L}__{checkpoint_hash[:8]}_{hashlib.sha256(str(path).encode()).hexdigest()[:6]}'
    out = root/key;out.mkdir()
    metadata = dict(checkpoint=str(path),checkpoint_sha256=checkpoint_hash,
        training_condition=condition['id'],training_s=int(condition['s']),depth=L,N=N,M=int(A.shape[0]))
    batch_size = args.batch_size or cfg.eval_batch_size
    seed = args.seed if args.seed is not None else cfg.test_seed
    write_json(out/'metadata.json',dict(**metadata,job=job,protocol=vars(cfg),
        generated_batch_size=batch_size,test_seed=seed,
        amplitude_prior=[cfg.amplitude_min,cfg.amplitude_max],amplitude_signs='symmetric',
        matrix_sha256=hashlib.sha256(A.detach().cpu().double().numpy().tobytes()).hexdigest(),
        frozen_weights_across_s=True,no_input_noise=True,initial_membrane_zero=True))
    all_samples, summaries, layer_rows = [],[],[]
    with torch.no_grad():
        for s in sorted(set(args.sparsities)):
            started=time.perf_counter()
            generator=data.generator(seed,device)
            samples=[]
            layer_sums={}
            while len(samples)<args.samples:
                targets,y=data.sparse_batch(batch_size,s,A,generator,cfg.amplitude_min,cfg.amplitude_max)
                # Preserve full generation batch; evaluate only needed samples.
                count=min(batch_size,args.samples-len(samples))
                targets,y=targets[:count],y[:count]
                reference,_=model(y)
                native_rate=float(model.last_firing_rate)
                native_ac=float(model.last_ac_count)
                reconstructed,path_tensor=native_trace(model,y)
                output_diff=float((reference-reconstructed).abs().max())
                if not torch.allclose(reference,reconstructed,rtol=1e-6,atol=1e-6):
                    raise AssertionError('Native forward disagrees with T=1 replay; source semantics differ')
                path_np=path_tensor.cpu().numpy()
                support=(targets!=0).cpu().numpy()
                counts,layers=count_paths(path_np,support)
                rate_difference=abs(native_rate-float(counts['spike_count'].sum())/(2*count*N*L))
                ac_difference=abs(native_ac-float(counts['feedback_magnitude_ac'].sum()))
                if rate_difference>1e-6 or ac_difference>1.+1e-6*abs(native_ac):
                    raise AssertionError('Native firing/magnitude-AC counters do not match replay')
                for name,values in layers.items():
                    layer_sums[name]=layer_sums.get(name,np.zeros(L,dtype=np.int64))+values.sum(axis=1)
                output=reference.cpu().double().numpy()
                truth=targets.cpu().double().numpy()
                predicted=output!=0
                sses=np.sum((output-truth)**2,axis=1)
                energies=np.sum(truth**2,axis=1)
                for j in range(count):
                    item={name:int(values[j]) for name,values in counts.items()}
                    item.update(test_s=s,sample=len(samples),sse=float(sses[j]),target_energy=float(energies[j]),
                        true_positive_count=int(np.sum(predicted[j]&support[j])),
                        predicted_support_count=int(predicted[j].sum()),
                        false_positive_count=int(np.sum(predicted[j]&~support[j])),
                        native_output_max_abs=output_diff,native_firing_abs_diff=rate_difference,
                        native_ac_abs_diff=ac_difference)
                    samples.append(item)
            summary=summarize(samples,metadata,s,L,N)
            summary['seconds']=time.perf_counter()-started
            summaries.append(summary)
            all_samples.extend(samples)
            for k in range(L):
                item=dict(**metadata,test_s=s,layer=k+1,samples=len(samples))
                item.update({name+'_mean':float(values[k]/len(samples)) for name,values in layer_sums.items()})
                layer_rows.append(item)
            write_csv(out/'samples.csv',all_samples)
            write_csv(out/'summary.csv',summaries)
            write_csv(out/'layers.csv',layer_rows)
            save_callback(summary,layer_rows[-L:])
            print(f'  {condition["id"]} L={L}, test s={s}: r={summary["firing_rate"]:.6g}, '
                f'ideal={summary["firing_ideal"]:.6g}, r/ideal={summary["firing_over_ideal"]}, '
                f'raw bound/r={summary["firing_raw_bound_to_actual"]}, NMSE={summary["actual_nmse_db"]}',flush=True)
    return out,summaries


def self_test():
    checked=0
    for L in range(1,8):
        paths=np.asarray(list(itertools.product((-1,0,1),repeat=L)),dtype=np.int64).T[:,:,None]
        for is_on in (False,True):
            support=np.full((paths.shape[1],1),is_on)
            stats,_=count_paths(paths,support)
            assert np.all(stats['spike_count']<=stats['spike_bound_raw'])
            assert np.all(stats['feedback_ac']<=stats['feedback_ac_bound_raw'])
            checked+=paths.shape[1]
    # Ideal-support example that does NOT approach s/(2N): each true location
    # fires once, then remains silent; r/(s/(2N)) = 1/L, not 1.
    path=np.zeros((5,1,8),dtype=int);path[0,0,:2]=1
    support=np.zeros((1,8),bool);support[0,:2]=True
    stats,_=count_paths(path,support)
    assert stats['spike_count'][0]/(5*2)==.2
    assert stats['off_support_spikes'][0]==0
    assert stats['feedback_ac'][0]==8*4*2
    # Off-support +1,-1 cancellation remains counted even when final q is zero.
    path=np.tile(np.array([1,-1,1,-1])[:,None,None],(1,2,3))
    stats,_=count_paths(path,np.zeros((2,3),bool))
    assert np.all(stats['off_support_final_energy']==0)
    assert np.all(stats['spike_count']==12)
    assert np.all(stats['spike_bound_raw']==12)
    print(f'PASS: {checked} signed support/path cases, both counting bounds, slack identities, cancellation, non-equality example.')


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sources',nargs='+')
    parser.add_argument('--checkpoints',nargs='+')
    parser.add_argument('--conditions',nargs='+',help='Select checkpoint TRAINING conditions')
    parser.add_argument('--depths',nargs='+',type=int)
    parser.add_argument('--sparsities',nargs='+',type=int,default=[0,1,2,4,7,14,28,42,56])
    parser.add_argument('--samples',type=int,default=1000,help='Per checkpoint and test sparsity')
    parser.add_argument('--project-dir',type=Path,default=Path('.'))
    parser.add_argument('--batch-size',type=int)
    parser.add_argument('--seed',type=int)
    parser.add_argument('--matrix-file',type=Path)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--output',type=Path,default=Path('activity_audit'))
    parser.add_argument('--no-plots',action='store_true')
    parser.add_argument('--font-size',type=float,default=14.)
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.samples<=0 or args.threads<=0 or args.font_size<=0 or not math.isfinite(args.font_size):
        parser.error('Samples, threads, font size must be positive')
    if args.batch_size is not None and args.batch_size<=0:
        parser.error('Batch size must be positive')
    if args.self_test:
        self_test()
        if not (args.sources or args.checkpoints):return
    if not (args.sources or args.checkpoints):
        parser.error('Supply --sources outputs/seed42/checkpoints or --checkpoints PATH')
    import torch
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    modules,source_files={},{}
    for name in ('config','data','models'):
        module,path=resolve_module(args.project_dir.expanduser().resolve(),name)
        modules[name]=module;source_files[name]=dict(path=str(path),sha256=sha(path))
    cfg=modules['config'].Protocol()
    paths=[Path(p).expanduser().resolve() for p in args.checkpoints or []]
    pattern=re.compile(r'(.+)__slista_l(\d+)_t1\.pth$')
    for source in args.sources or []:
        directory=Path(source).expanduser().resolve()
        if not directory.is_dir():parser.error('Missing source: '+str(directory))
        for path in sorted(directory.rglob('*__slista_l*_t1.pth')):
            match=pattern.fullmatch(path.name)
            if match and (not args.conditions or match[1] in args.conditions) and (not args.depths or int(match[2]) in args.depths):
                paths.append(path)
    paths=list(dict.fromkeys(paths))
    if not paths:parser.error('No matching T=1 checkpoints')
    root=args.output.expanduser().resolve()/dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    root.mkdir(parents=True,exist_ok=False)
    summaries,layers,errors,skipped,plot_errors=[],[],[],[],[]

    def save(summary=None,new_layers=None):
        if summary is not None:
            summaries.append(summary);layers.extend(new_layers)
        write_csv(root/'summary.csv',summaries)
        write_csv(root/'layers.csv',layers)
        write_json(root/'report.json',dict(summaries=summaries,errors=errors,skipped=skipped,plot_errors=plot_errors,
            source_files=source_files,arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            total_sample_evaluations=sum(r['samples'] for r in summaries),
            notes=['Every checkpoint is fixed throughout its input sparsity sweep; different checkpoints are reported separately.',
                'No retraining, T=4, injected noise, energy renormalization, or streaming experiment.',
                'r<=s/(2N) under no off-support intermediate activity; equality/convergence is NOT promised.',
                'Normalized rate equals on-support firing utilization plus off-support spikes divided by L*s.',
                'Reported bound ratios are ratios of sums, not mean per-sample ratios. Zero denominators are null.',
                'Raw bounds and physical-cap bounds are both reported; panel d shows RAW-bound tightness.',
                'NMSE and support recall help distinguish low activity from failure to respond.',
                'Legacy feedback_ac fields are L0 active-connection counts, not energy ACs. feedback_magnitude_ac is the L1 count checked against the native model; both exclude embedding and dictionary readout.',
                'Statistics use the native-dtype forward path; no independent double-precision spike pattern is substituted.']))

    print(f'Selected {len(paths)} fixed checkpoint(s), test s={sorted(set(args.sparsities))}, {args.samples} samples per s.',flush=True)
    for path in paths:
        try:
            result=audit_checkpoint(path,args,modules,cfg,root,save)
            if result is not None and not args.no_plots:
                folder,rows=result
                try:plot_checkpoint(rows,folder,args.font_size)
                except Exception as exc:
                    plot_errors.append(dict(checkpoint=str(path),error=f'{type(exc).__name__}: {exc}'))
                    print('PLOT FAILED (CSV results retained): '+str(exc),file=sys.stderr,flush=True)
        except NotImplementedError as exc:
            skipped.append(dict(checkpoint=str(path),reason=str(exc)))
            print('SKIPPED: '+str(path)+': '+str(exc),flush=True)
        except Exception as exc:
            errors.append(dict(checkpoint=str(path),error=f'{type(exc).__name__}: {exc}'))
            print('FAILED: '+str(path)+': '+str(exc),file=sys.stderr,flush=True)
        save()
    print(f'Done: {len(summaries)} checkpoint/sparsity pairs; {len(errors)} failures, {len(skipped)} skipped.\nAudit output: {root}',flush=True)
    if errors or plot_errors or not summaries:raise SystemExit(1)


if __name__=='__main__':
    main()
