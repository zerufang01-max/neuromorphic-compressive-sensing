"""Evaluate all five independent trainings and aggregate paired fixed-test results."""
import argparse
import csv
import statistics
from copy import copy
from pathlib import Path
from utils import DEFAULT_OUTPUT, read_json, save_json


def aggregate(rows, keys):
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row.get(k) for k in keys), []).append(row)
    output = []
    for group in groups.values():
        row = dict(group[0])
        row.pop('seed', None)
        row['seeds'] = [r['seed'] for r in group]
        if len(row['seeds']) != len(set(row['seeds'])):
            raise ValueError('Duplicate seed in aggregate')
        row['n_seeds'] = len(group)
        for key in group[0]:
            values = [r.get(key) for r in group]
            if key not in keys and key != 'seed' and all(isinstance(v, (int,float)) for v in values):
                row[key] = statistics.mean(values)
                row[key+'_std'] = statistics.stdev(values) if len(values)>1 else 0.0
        output.append(row)
    return output


def main(args):
    import torch
    import evaluate
    import evaluate_snr
    from config import Protocol
    from utils import selected_models
    all_rows, all_snr = [], []
    expected = None
    for seed in args.seeds:
        root = Path(args.output) / f'seed{seed}'
        selected = selected_models(root)
        keys = set(selected)
        if expected is None:
            expected = keys
        if keys != expected:
            raise ValueError(f'Model grids differ across seeds: seed{seed}')
        rows = []
        snr_rows = []
        for configuration, checkpoint in selected.values():
            if configuration.get('seed') != seed:
                raise ValueError(f'Wrong seed in manifest: {checkpoint}')
            row = evaluate.evaluate(configuration, checkpoint, torch.device(args.device), Protocol())
            rows.append(row)
            print(seed, row['condition_id'], row['id'], row['nmse_db'], flush=True)
            if args.snr and configuration['condition']['id'] == evaluate_snr.SNR_CONDITION_ID and configuration['id'] in evaluate_snr.SNR_MODEL_IDS:
                for snr in evaluate_snr.SNR_DB_VALUES:
                    snr_rows.append(evaluate_snr.evaluate_model(configuration, checkpoint, snr,
                                                               torch.device(args.device), Protocol()))
        save_json(root/'metrics/results.json', rows)
        if snr_rows:
            clean = {r['id']:r['nmse_db'] for r in snr_rows if r['snr_db'] is None}
            for r in snr_rows:
                r['degradation_db'] = r['nmse_db']-clean[r['id']]
            save_json(root/'metrics/snr_results.json', snr_rows)
        all_rows.extend(rows); all_snr.extend(snr_rows)
    out = Path(args.output)/'metrics'
    for name, rows, keys in [('results',all_rows,['condition_id','id']),
                              ('snr_results',all_snr,['condition_id','id','snr_db'])]:
        if not rows:
            continue
        save_json(out/(name+'_per_seed.json'),rows)
        summary = aggregate(rows,keys)
        save_json(out/(name+'.json'),summary)
        for suffix, values in [('_per_seed',rows),('',summary)]:
            with (out/(name+suffix+'.csv')).open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in values for k in r)))
                writer.writeheader();writer.writerows(values)
    save_json(out/'aggregation_protocol.json',{
        'seeds':args.seeds,'nmse':'mean and sample SD of per-seed aggregate NMSE in dB',
        'matrix_and_test':'fixed across seeds; training initialization/data change',
        'device':args.device})

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',default=str(DEFAULT_OUTPUT))
    p.add_argument('--seeds',type=int,nargs='+',default=[42,43,44,45,46])
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--snr',action='store_true')
    a=p.parse_args()
    if len(set(a.seeds))!=len(a.seeds): p.error('Duplicate seeds')
    main(a)
