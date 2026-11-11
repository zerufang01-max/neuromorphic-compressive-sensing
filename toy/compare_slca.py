"""Supp. Fig. 4: LIF S-LCA (fixed lambda, 100 to 100,000 steps) vs the five
trained S-LISTA models (L = 10, s = 28) on five shared test signals.

Energy uses the same analytical model as evaluate.py
(models.SpikingLISTA.operation_counts for S-LISTA; slca.synaptic_cost for S-LCA).
Latency is synchronized GPU wall-clock time, median over the five signals."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

import torch
from config import (SLCA_CONDITION_ID, SLCA_DT, SLCA_HORIZONS, SLCA_PENALTY,
                    SLCA_SIGNAL_SEEDS, SLCA_SLISTA_ID, TRAINING_SEEDS)
from data import generator, sparse_batch
from models import SpikingLISTA
from slca import SignedSLCA
from utils import checkpoint_path, load_model

SEEDS = list(SLCA_SIGNAL_SEEDS)
HORIZONS = list(SLCA_HORIZONS)
RESULT = 'slca_five_signals.json'


def save(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(path)


@torch.inference_mode()
def latency(fn, inputs, device):
    # One warm-up, then exactly one timed call per signal; no monitoring.
    fn(inputs[0]);torch.cuda.synchronize(device)
    samples=[]
    for x in inputs:
        torch.cuda.synchronize(device);start=time.perf_counter_ns()
        result=fn(x)
        torch.cuda.synchronize(device);samples.append((time.perf_counter_ns()-start)/1e6)
        if not torch.isfinite(result).all():raise ValueError('Nonfinite timed output')
    return samples


def metrics(error,power,spikes,energy,times,**extra):
    return dict(nmse_db=10*math.log10(max(sum(error),1e-30)/sum(power)),
                spikes_per_sample=statistics.mean(spikes),energy_uj=statistics.mean(energy),
                latency_ms=statistics.median(times),latency_samples_ms=times,
                squared_errors=error,target_energies=power,spike_counts=spikes,
                energies_uj=energy,**extra)


@torch.inference_mode()
def run(args):
    device=torch.device(args.device)
    if device.type!='cuda' or not torch.cuda.is_available():raise ValueError('GPU timing requires CUDA')
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    root=args.output.resolve()
    # Validate all checkpoints before starting the expensive recurrence.
    models=[];fingerprints={}
    for seed in TRAINING_SEEDS:
        path=checkpoint_path(root,seed,SLCA_CONDITION_ID,SLCA_SLISTA_ID)
        model,matrix,job,cfg=load_model(path,device)
        if job.get('seed')!=seed or job['id']!=SLCA_SLISTA_ID:
            raise ValueError(f'Checkpoint identity mismatch: {path}')
        models.append(model);fingerprints[str(seed)]=hashlib.sha256(path.read_bytes()).hexdigest()
    condition=job['condition'];m,s=condition['m'],condition['s']
    targets=[];inputs=[]
    for seed in SEEDS:
        z,y=sparse_batch(1,s,matrix,generator(seed,device),cfg.amplitude_min,cfg.amplitude_max)
        targets.append(z);inputs.append(y.contiguous())
    power=[float(z.square().sum()) for z in targets]
    meta=dict(data_seeds=SEEDS,model_seeds=list(TRAINING_SEEDS),samples=len(SEEDS),n=cfg.n,m=m,s=s,
              lambda_value=SLCA_PENALTY,dt=SLCA_DT,horizons=HORIZONS,gpu=torch.cuda.get_device_name(device),
              torch=torch.__version__,cuda=torch.version.cuda,batch_size=1,dtype='float32',
              warmup_calls=1,timed_calls_per_configuration=len(SEEDS),
              timing='synchronized wall clock; median of one call per signal',
              backend={'slca':'TorchScript recurrent loop','slista':'native PyTorch'},
              nmse='10 log10(sum squared error / sum target energy) over the signals',
              scope='small-sample illustrative comparison; not the main 10000-signal benchmark',
              protocol=cfg.__dict__,checkpoint_sha256=fingerprints)
    raw={h:dict(error=[],spikes=[],energy=[]) for h in HORIZONS}
    baseline=SignedSLCA(matrix,penalty=SLCA_PENALTY,dt=SLCA_DT).eval()
    print(f'Statistics: {len(SEEDS)} shared signals; fixed lambda={SLCA_PENALTY}; no scan.',flush=True)
    for seed,z,y in zip(SEEDS,targets,inputs):
        for h,estimate,events in baseline.snapshots(y,HORIZONS):
            if not torch.isfinite(estimate).all():raise ValueError('Nonfinite S-LCA output')
            count=float(events.sum());macs,acs=baseline.synaptic_cost(count)
            raw[h]['error'].append(float((estimate-z).square().sum()))
            raw[h]['spikes'].append(count);raw[h]['energy'].append(macs*cfg.e_mac_uj+acs*cfg.e_ac_uj)
            print(f'data seed={seed}, steps={h}, NMSE={10*math.log10(max(raw[h]["error"][-1],1e-30)/float(z.square().sum())):.3f} dB',flush=True)
    learned=[]
    for seed,model in zip(TRAINING_SEEDS,models):
        errors=[];spikes=[];energies=[]
        for z,y in zip(targets,inputs):
            estimate,_=model(y);errors.append(float((estimate-z).square().sum()))
            ops=SpikingLISTA.operation_counts(model.last_trace,model.n,model.m)
            spikes.append(ops['spikes']);energies.append(ops['mac']*cfg.e_mac_uj+ops['ac']*cfg.e_ac_uj)
        learned.append(dict(seed=seed,error=errors,spikes=spikes,energy=energies))
    # Save statistical work before timing so an interruption does not erase it.
    save(root/'metrics/slca_five_signals_statistics.json',dict(metadata=meta,slca=raw,slista=learned,power=power))
    print('Timing: keep GPU free of other workloads; 1 warm-up + 5 timed calls per configuration.',flush=True)
    slca=[];slista=[]
    for h in HORIZONS:
        times=latency(lambda y:baseline(y,h),inputs,device);v=raw[h]
        slca.append(metrics(v['error'],power,v['spikes'],v['energy'],times,steps=h))
        print(f'S-LCA steps={h}: median {statistics.median(times):.3f} ms',flush=True)
    for model,v in zip(models,learned):
        times=latency(lambda y:model(y)[0],inputs,device)
        slista.append(metrics(v['error'],power,v['spikes'],v['energy'],times,seed=v['seed']))
    save(root/'metrics'/RESULT,dict(metadata=meta,slca=slca,slista=slista))


def plot(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    d=json.loads((root/'metrics'/RESULT).read_text());b=d['slca'];s=d['slista']
    fig,axes=plt.subplots(1,3,figsize=(12.6,4.0),sharey=True)
    for i,(ax,key,label) in enumerate(zip(axes,['latency_ms','spikes_per_sample','energy_uj'],
          ['Inference time (ms)','Spike count','Synaptic energy (μJ)'])):
        ax.plot([r[key] for r in b],[r['nmse_db'] for r in b],'o-',color='#3976B8',ms=4,lw=1.5)
        ax.errorbar(statistics.mean(r[key] for r in s),statistics.mean(r['nmse_db'] for r in s),
                    xerr=statistics.stdev(r[key] for r in s),yerr=statistics.stdev(r['nmse_db'] for r in s),
                    fmt='s',color='#D84A3A',ms=7,capsize=3)
        ax.set_xscale('log');ax.set_xlabel(label,fontsize=18);ax.tick_params(labelleft=True,labelsize=18)
        # Pad the log x-range so neither method sits on the axis edge.
        values=[r[key] for r in b]+[r[key] for r in s]
        ax.set_xlim(min(values)/2.5,max(values)*2.5)
        ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
        ax.text(-.10,1.04,chr(97+i),transform=ax.transAxes,fontweight='bold',fontsize=21)
    axes[0].set_ylabel('NMSE (dB)',fontsize=18)
    axes[0].legend(handles=[Line2D([],[],color='#3976B8',marker='o',label='S-LCA'),
                           Line2D([],[],color='#D84A3A',marker='s',ls='none',label='S-LISTA')],loc='lower right',frameon=False,fontsize=17,handlelength=1.5,labelspacing=.3)
    fig.tight_layout();out=root/'figures';out.mkdir(parents=True,exist_ok=True)
    for ext in ('pdf','png'):fig.savefig(out/f'slca_comparison.{ext}',dpi=400,bbox_inches='tight')
    plt.close(fig)
    print('Saved',out/'slca_comparison.pdf',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('outputs'))
    p.add_argument('--device',default='cuda:0');p.add_argument('--plot-only',action='store_true')
    args=p.parse_args()
    if not args.plot_only:run(args)
    plot(args.output)
