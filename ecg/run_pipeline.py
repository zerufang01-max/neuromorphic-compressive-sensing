"""Five-seed ECG pipeline: train, evaluate on DS2, aggregate, plot."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
import queue

ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'outputs')
    p.add_argument('--data-dir',type=Path,default=ROOT/'data/mitdb_pt')
    p.add_argument('--config',type=Path,default=ROOT/'configs/paper.json')
    p.add_argument('--gpus',nargs='+',type=int,default=list(range(6)))
    p.add_argument('--workers-per-gpu',type=int,default=2)
    p.add_argument('--stage',choices=['all','train','evaluate','figures'],default='all')
    p.add_argument('--cpu',action='store_true')
    p.add_argument('--final-slista-only', action='store_true')
    a=p.parse_args();a.output=a.output.resolve();a.data_dir=a.data_dir.resolve()
    from main import make_jobs, launch_jobs, aggregate
    settings=json.loads(a.config.read_text());seeds=settings['seeds']
    if seeds != [42,43,44,45,46]: raise ValueError('Public five-seed protocol requires seeds 42--46')
    a.num_workers=0
    os.environ.setdefault('OMP_NUM_THREADS','2')
    os.environ.setdefault('MKL_NUM_THREADS','2')
    if a.workers_per_gpu<1: p.error('workers-per-gpu must be positive')
    if a.stage in ('all','train'):
        import hashlib
        phases=['final_slista'] if a.final_slista_only else ['fixed','learned','final_slista','quantization']
        jobs=[]
        for seed in seeds:
            cfg=dict(settings,seed=seed)
            for job in make_jobs(cfg,['alista','lamp','lista_ann','lista_snn_1'],phases):
                job['directory']=f'seed{seed}/'+job['directory']
                job['num_workers']=0
                job['data_dir']=str(a.data_dir)
                job['data_sha256']={n:hashlib.sha256((a.data_dir/n).read_bytes()).hexdigest() for n in ('train_data.pt','val_data.pt','test_data.pt','split_manifest.json')}
                job['source_sha256']={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ('config.py','snn_models.py','trainer.py','utils.py','edge_compression.py','wireless_channel.py','main.py')}
                jobs.append(job)
        for phase in phases:
            launch_jobs([j for j in jobs if j['phase']==phase],a)
            for seed in seeds: aggregate(a.output/f'seed{seed}')
    if a.stage in ('all','evaluate'):
        slots=queue.Queue()
        for gpu in a.gpus: slots.put(gpu)
        def evaluate(seed):
            gpu=slots.get();root=a.output/f'seed{seed}'
            try:
                root.mkdir(parents=True,exist_ok=True)
                env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
                if a.cpu: env['CUDA_VISIBLE_DEVICES']=''
                with (root/'evaluation.log').open('a') as log:
                    for script in ['evaluate_rayleigh.py','plot_results.py']:
                        cmd=[sys.executable,'-u',str(ROOT/script),'--output',str(root),'--data-dir',str(a.data_dir),'--gpu',str(gpu)]
                        if script=='plot_results.py': cmd+=['--force-recompute']
                        if a.cpu and script=='evaluate_rayleigh.py': cmd+=['--cpu']
                        subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            finally: slots.put(gpu)
        with ThreadPoolExecutor(max_workers=len(a.gpus)) as pool: list(pool.map(evaluate,seeds))
    if a.stage in ('all','evaluate','figures'):
        subprocess.run([sys.executable,str(ROOT/'summarize_results.py'),'--output',str(a.output)],check=True,cwd=ROOT)

if __name__=='__main__': main()
