"""Validate the exact fixed model inventory and checkpoint/manifest correspondence."""
import argparse
import json
from pathlib import Path
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('outputs'))
    p.add_argument('--config',type=Path,default=Path(__file__).resolve().parent/'configs/paper.json')
    a=p.parse_args();cfg=json.loads(a.config.read_text())
    expected={j['condition']['id']+'__'+j['id']:j for j in cfg['jobs']}
    if len(expected)!=len(cfg['jobs']):raise ValueError('Duplicate configured model')
    for seed in cfg['seeds']:
        root=a.output/f'seed{seed}'
        rows=json.loads((root/'configs/selected_configs.json').read_text())
        actual={j['condition']['id']+'__'+j['id']:j for j in rows}
        if len(rows)!=len(actual) or set(actual)!=set(expected):
            raise ValueError(f'seed{seed}: missing={set(expected)-set(actual)}, extra={set(actual)-set(expected)}')
        files={f.stem for f in (root/'checkpoints').glob('*.pth')}
        if files!=set(expected):raise ValueError(f'seed{seed}: checkpoint file inventory differs')
        for key,j in actual.items():
            want=expected[key]
            for field in ('parameters','stage1_learning_rate','stage2_learning_rate','depth','time_steps','method','condition'):
                if j[field]!=want[field]:raise ValueError(f'{key}: {field} differs from paper config')
            if j.get('seed')!=seed:raise ValueError(f'{key}: seed mismatch')
            path=root/j['checkpoint']
            if path.resolve()!=(root/'checkpoints'/f'{key}.pth').resolve():raise ValueError('Unexpected checkpoint path')
            payload=torch.load(path,map_location='cpu',weights_only=False)
            for field in ('parameters','depth','time_steps','condition','seed'):
                if payload['job'].get(field)!=j.get(field):raise ValueError(f'{key}: checkpoint metadata mismatch')
        print(f'seed{seed}: exactly {len(expected)} matching final models',flush=True)
    print('Fixed model inventory verified.')

if __name__=='__main__':main()
