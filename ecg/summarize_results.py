"""Aggregate per-seed test metrics; no averaging reconstructed waveforms."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from plot_results import plot_composite, TARGETS


def grouped(frames, keys, value):
    data=pd.concat([f.assign(Seed=42+i) for i,f in enumerate(frames)],ignore_index=True)
    if data.duplicated(keys+['Seed']).any(): raise ValueError('Duplicate seed/metric rows')
    g=data.groupby(keys)[value]
    if not (g.count()==5).all(): raise ValueError('Every plotted condition must have all five seeds')
    return g.agg(['mean','std']).reset_index().rename(columns={'mean':value})


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('outputs'));a=p.parse_args()
    runs=[]
    for seed in range(42,47):
        root=a.output/f'seed{seed}'
        for phase in ('fixed','learned','quantization','final_slista'):
            expected=16 if phase=='quantization' else (1 if phase=='final_slista' else 4)
            completed=list((root/phase).rglob('job.json'))
            if len(completed)!=expected: raise ValueError(f'{root/phase}: expected {expected} jobs')
            for job in completed:
                if not (job.parent/'results.csv').is_file(): raise ValueError(f'Incomplete: {job.parent}')
        runs.append(torch.load(root/'plot_data.pt',map_location='cpu',weights_only=False))
    if len({d['identity']['data'] for d in runs})!=1: raise ValueError('Test sets differ')
    cache=dict(runs[0]['cache']) # example waveforms always seed42; never ensemble predictions
    stats={}
    for key in ('fixed_nmse','learned_nmse'):
        mean={t:float(np.mean([d[key][t] for d in runs])) for t in TARGETS}
        sd={t:float(np.std([d[key][t] for d in runs],ddof=1)) for t in TARGETS}
        stats[key]=mean;cache[key.replace('_nmse','_std')]=sd
    for key in ('energy','operation_counts'):
        if key=='operation_counts':
            cache[key]={t:{op:float(np.mean([d['cache'][key][t][op] for d in runs])) for op in ('mac','ac')} for t in TARGETS}
        else:
            cache[key]={t:float(np.mean([d['cache'][key][t] for d in runs])) for t in TARGETS}
            cache[key+'_std']={t:float(np.std([d['cache'][key][t] for d in runs],ddof=1)) for t in TARGETS}
    for key in ('layer_positive','layer_negative','layer_drive_ratio'):
        values=np.asarray([d['cache'][key] for d in runs]);cache[key]=values.mean(axis=0);cache[key+'_std']=values.std(axis=0,ddof=1)
    totals=np.asarray([np.asarray(d['cache']['layer_positive'])+np.asarray(d['cache']['layer_negative']) for d in runs])
    cache['layer_total_std']=totals.std(axis=0,ddof=1)
    robust=grouped([d['robustness'] for d in runs],['Target','eval_snr_db'],'NMSE_dB')
    ray=grouped([d['rayleigh'] for d in runs],['target','snr_db'],'nmse_db')
    quant=grouped([d['quant'] for d in runs],['Target','Quant_Bits'],'Test_NMSE_dB')
    out=a.output/'figures';out.mkdir(parents=True,exist_ok=True)
    metrics=a.output/'metrics';metrics.mkdir(exist_ok=True)
    for name,df in [('awgn',robust),('rayleigh',ray),('quantization',quant)]: df.to_csv(metrics/f'{name}.csv',index=False)
    summary=dict(stats,energy=cache['energy'],energy_std=cache['energy_std'],fixed_std=cache['fixed_std'],learned_std=cache['learned_std'],seeds=list(range(42,47)),example_seed=42,uncertainty='sample standard deviation across five training seeds; firing stacked upper error = SD of total')
    (metrics/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    args=SimpleNamespace(output_dir=out)
    plot_composite(cache,robust,stats['fixed_nmse'],stats['learned_nmse'],quant,args,ray)
    print('Saved five-seed figures:',out)

if __name__=='__main__':main()
