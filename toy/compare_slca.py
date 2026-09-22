"""Five shared test signals: fixed-lambda LIF S-LCA vs five trained S-LISTA models."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

import torch
from config import Protocol
from data import generator, measurement_matrix, sparse_batch
from models import build_model
from slca import SignedSLCA

SEEDS = [42,43,44,45,46]
HORIZONS = [100,200,500,1000,2000,5000,10000,20000,50000,100000]
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
    root=args.output.resolve(); cfg=Protocol(**json.loads((root/'seed42/configs/run_config.json').read_text())['protocol'])
    matrix=measurement_matrix(cfg.n,141,cfg.max_m,cfg.matrix_seed,device)
    targets=[];inputs=[]
    for seed in SEEDS:
        z,y=sparse_batch(1,28,matrix,generator(seed,device),cfg.amplitude_min,cfg.amplitude_max)
        targets.append(z);inputs.append(y.contiguous())
    power=[float(z.square().sum()) for z in targets]
    meta=dict(data_seeds=SEEDS,model_seeds=SEEDS,samples=5,n=cfg.n,m=141,s=28,
              lambda_value=.1,dt=.01,horizons=HORIZONS,gpu=torch.cuda.get_device_name(device),
              torch=torch.__version__,cuda=torch.version.cuda,batch_size=1,dtype='float32',
              warmup_calls=1,timed_calls_per_configuration=5,
              timing='synchronized wall clock; median of one call per signal',
              backend={'slca':'TorchScript recurrent loop','slista':'native PyTorch'},
              nmse='10 log10(sum squared error / sum target energy) over five signals',
              scope='small-sample illustrative comparison; not the main 10000-signal benchmark',
              protocol=cfg.__dict__)
    # Validate all checkpoints before starting the expensive recurrence.
    models=[];fingerprints={}
    fields=['n','max_m','matrix_seed','amplitude_min','amplitude_max','e_mac_uj','e_ac_uj']
    for seed in SEEDS:
        path=root/f'seed{seed}/checkpoints/m141_s28__slista_l20_t1.pth'
        payload=torch.load(path,map_location=device,weights_only=False);job=payload['job']
        if (job.get('seed'),job['method'],job['depth'],job['time_steps'],job['condition']['id'])!=(seed,'slista',20,1,'m141_s28'):
            raise ValueError(f'Checkpoint identity mismatch: {path}')
        for field in fields:
            if payload['protocol'][field]!=getattr(cfg,field):raise ValueError(f'Protocol mismatch: {field}')
        model=build_model('slista',matrix,20,1,job['parameters']).to(device).eval()
        model.load_state_dict(payload['model'],strict=True);models.append(model)
        fingerprints[str(seed)]=hashlib.sha256(path.read_bytes()).hexdigest()
    meta['checkpoint_sha256']=fingerprints
    raw={h:dict(error=[],spikes=[],energy=[]) for h in HORIZONS}
    baseline=SignedSLCA(matrix,penalty=.1,dt=.01).eval()
    print('Statistics: five shared signals; fixed lambda=0.1; no scan.',flush=True)
    for seed,z,y in zip(SEEDS,targets,inputs):
        for h,estimate,events in baseline.snapshots(y,HORIZONS):
            if not torch.isfinite(estimate).all():raise ValueError('Nonfinite S-LCA output')
            count=float(events.sum());macs,acs=baseline.synaptic_cost(count)
            raw[h]['error'].append(float((estimate-z).square().sum()))
            raw[h]['spikes'].append(count);raw[h]['energy'].append(macs*cfg.e_mac_uj+acs*cfg.e_ac_uj)
            print(f'data seed={seed}, steps={h}, NMSE={10*math.log10(max(raw[h]["error"][-1],1e-30)/float(z.square().sum())):.3f} dB',flush=True)
    learned=[]
    for seed,model in zip(SEEDS,models):
        errors=[];spikes=[];energies=[];model.collect_stats=True
        for z,y in zip(targets,inputs):
            estimate,_=model(y);errors.append(float((estimate-z).square().sum()))
            spikes.append(float(model.last_spike_count));energies.append(141*cfg.n*cfg.e_mac_uj+float(model.last_ac_count)*cfg.e_ac_uj)
        model.collect_stats=False
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
        ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
        ax.text(-.10,1.04,chr(97+i),transform=ax.transAxes,fontweight='bold',fontsize=21)
    axes[0].set_ylabel('NMSE (dB)',fontsize=18)
    axes[0].legend(handles=[Line2D([],[],color='#3976B8',marker='o',label='S-LCA'),
                           Line2D([],[],color='#D84A3A',marker='s',ls='none',label='S-LISTA')],loc='upper right',frameon=False,fontsize=17,handlelength=1.5,labelspacing=.3)
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
