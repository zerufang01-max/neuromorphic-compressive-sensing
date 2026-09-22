#!/usr/bin/env python3
"""Publication figures for the validated S-LISTA recovery and activity bounds.

One-command audit + plotting in the original toy project (no retraining):
  python make_slista_theory_figures.py --run-audits \
    --sources outputs/seed42/checkpoints --device cuda:0

Keep audit_slista_joint_bound.py and audit_slista_activity.py beside this file.
Defaults: recovery L=10, training s=28/42/56, 1000 samples per model; activity
fixed s=28-trained L=10 model, 5000 samples per test sparsity. Bounds use CPU;
the recovery audit is substantially slower than the firing-rate sweep.

Plot existing results without any model evaluation:
  python make_slista_theory_figures.py \
    --error-results joint_bound_audit/EXACT_RUN \
    --activity-results activity_audit/EXACT_RUN

Either input is optional. --activity-results accepts summary.csv, report.json,
a run directory or the returned ZIP. --error-results needs samples.csv (not summary.csv), a run
directory with per-sample metrics.json files, or a zip of that run. A zip is
read directly, never extracted. Missing sample-level errors cannot be inferred
from aggregate averages. Multiple runs are never silently pooled.

FIGURES
recovery_bounds.pdf/png: per model, sort by the final-output squared-error
certificate; use exactly the same permutation for actual-error bars. Every
sample is retained, no smoothing, interpolation, clipping or re-ranking of bars.
Axes show squared recovery error in linear units; x is sorted sample rank.
Rows from this audit use float64 mathematical replay, not native float32 error.

firing_ideal_reference.pdf/png: empirical mean rate and the conditional
ideal-support reference s/(2N). All data, including rates above the reference,
remain visible. This reference is not asserted to bound arbitrary trained models.

firing_bounds.pdf/png: empirical mean rate as bars, s/(2N) as a dashed line,
and the mean FULL RAW firing bound as a line. The raw bound is not replaced
with the 0.5 physical cap. Lower panels enlarge low sparsities; no data are
altered. All samples contribute to each mean, including silent/poor recoveries.
Equal weights are used for individual samples, not for separately trained models.
When sample standard deviations are available, error bars show one standard
error of the observed mean (not a bound on individual samples).

Exact plotted values, original sample identifiers and metadata/captions are
saved next to the figures. Model groups are never silently selected based on
performance. Larger sample counts improve evaluation precision; they are not
a theorem about convergence in depth or sparsity.

Dependencies: numpy, matplotlib for plotting; torch, scipy for --run-audits.
"""

import argparse
import csv
import datetime as dt
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator

BLUE, ORANGE = '#0072B2', '#D55E00'


def json_write(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def csv_write(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def numeric(row, key):
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f'Nonfinite {key}')
    return value


def read_csv_text(text):
    return list(csv.DictReader(io.StringIO(text)))


def check_report(report):
    if report.get('errors'):
        raise ValueError('Audit report contains failures; inspect report.json before publication plotting.')


def error_rows(path):
    if path.suffix.lower() == '.zip':
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            candidates=[]
            for name in names:
                if name.startswith('__MACOSX/') or not (name.endswith('/samples.csv') or name=='samples.csv'):
                    continue
                rows=read_csv_text(z.read(name).decode('utf-8-sig'))
                if rows and 'joint_bound_sse' in rows[0] and 'actual_sse' in rows[0]:
                    candidates.append((name,rows))
            if len(candidates)>1:
                raise ValueError('Zip includes multiple recovery sample tables; provide one audit run at a time.')
            if candidates:
                name,rows=candidates[0]
                report_name=str(Path(name).with_name('report.json'))
                if report_name in names:check_report(json.loads(z.read(report_name)))
                return rows
            reports=[name for name in names if name.endswith('report.json') and not name.startswith('__MACOSX/')]
            if len(reports)>1:raise ValueError('Zip has no unique recovery sample table; provide an exact audit run.')
            for name in reports:check_report(json.loads(z.read(name)))
            rows=[json.loads(z.read(name)) for name in sorted(names) if name.endswith('/metrics.json')]
            return rows
    if path.is_dir():
        reports = list(path.glob('report.json'))
        if not reports:
            candidates = sorted(path.glob('*/report.json'))
            if len(candidates)==1:
                return error_rows(candidates[0].parent)
            if len(candidates)>1:
                raise ValueError('Multiple runs found; pass an exact joint_bound_audit/<timestamp> directory.')
        for report in reports: check_report(json.loads(report.read_text()))
        if (path/'samples.csv').is_file(): return read_csv_text((path/'samples.csv').read_text())
        return [json.loads(p.read_text()) for p in sorted(path.rglob('metrics.json'))]
    if path.suffix.lower()=='.csv': return read_csv_text(path.read_text(encoding='utf-8-sig'))
    if path.name=='report.json':
        return error_rows(path.parent)
    raise ValueError('Recovery plot requires sample-level samples.csv or an audit run; aggregate summary is insufficient.')


def activity_rows(path):
    if path.suffix.lower()=='.zip':
        candidates=[]
        with zipfile.ZipFile(path) as z:
            for name in z.namelist():
                if not name.endswith('report.json') or name.startswith('__MACOSX/'):continue
                report=json.loads(z.read(name))
                rows=report.get('summaries',[])
                if rows and 'firing_bound_raw' in rows[0]:
                    check_report(report);candidates.append(rows)
        if len(candidates)!=1:raise ValueError('ZIP must contain exactly one activity report')
        return candidates[0]
    if path.is_dir():
        if (path/'report.json').is_file(): return activity_rows(path/'report.json')
        candidates = sorted(path.glob('*/report.json'))
        if len(candidates)==1:return activity_rows(candidates[0])
        if len(candidates)>1:raise ValueError('Multiple activity runs found; pass an exact run directory.')
        if (path/'summary.csv').is_file():return activity_rows(path/'summary.csv')
        raise ValueError('No activity summary/report in directory')
    if path.suffix.lower()=='.json':
        report=json.loads(path.read_text());check_report(report)
        return report['summaries']
    return read_csv_text(path.read_text(encoding='utf-8-sig'))


def style(size):
    selected='DejaVu Sans'
    for family in ('Aptos','Arial','DejaVu Sans'):
        try:
            font_manager.findfont(family,fallback_to_default=False)
            selected=family;break
        except ValueError:pass
    plt.rcParams.update({'font.family':selected,'font.size':size,'axes.labelsize':size,
        'xtick.labelsize':size-1,'ytick.labelsize':size-1,'legend.fontsize':size-1,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.8,
        'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
    return selected


def finish(fig, out, name):
    fig.savefig(out/(name+'.pdf'),bbox_inches='tight',pad_inches=.06)
    fig.savefig(out/(name+'.png'),dpi=300,bbox_inches='tight',pad_inches=.06)
    plt.close(fig)


def recovery_figure(rows,out,size):
    groups={}
    for row in rows:
        if 'joint_bound_sse' not in row or 'actual_sse' not in row or 'sample' not in row:
            raise ValueError('Recovery input contains aggregate rows. Need the joint audit root samples.csv.')
        group=(str(row['checkpoint']),int(row['depth']),str(row['condition']))
        groups.setdefault(group,[]).append(row)
    if not groups:raise ValueError('No recovery samples found')
    if len(groups)>6:raise ValueError('More than six model groups; provide a specific audit run for a readable figure.')
    keys=sorted(groups,key=lambda k:(k[1],int(k[2].split('_s')[-1]) if '_s' in k[2] else 0,k[0]))
    ncols=min(3,len(keys));nrows=math.ceil(len(keys)/ncols)
    fig,axes=plt.subplots(nrows,ncols,figsize=(5.1*ncols,3.9*nrows),squeeze=False,layout='constrained')
    plotted,stats=[],[]
    for index,key in enumerate(keys):
        ax=axes.ravel()[index]; group=groups[key]
        ids=[str(r['sample']) for r in group]
        if len(set(ids))!=len(ids):raise ValueError('Duplicate sample IDs within model; multiple runs may have been mixed.')
        thresholds={numeric(r,'output_threshold') if 'output_threshold' in r else 0. for r in group}
        if len(thresholds)!=1:raise ValueError('Mixed final thresholds within one recovery curve')
        final=all('final_joint_bound_sse' in r and 'final_actual_sse' in r for r in group)
        bk='final_joint_bound_sse' if final else 'joint_bound_sse'
        ek='final_actual_sse' if final else 'actual_sse'
        upper=np.array([numeric(r,bk) for r in group]); actual=np.array([numeric(r,ek) for r in group])
        if np.any(upper<0) or np.any(actual<0):raise ValueError('Negative squared errors')
        if np.any(actual>upper+1e-7*(1+upper)):raise ValueError('Actual recovery error exceeds its bound')
        order=np.argsort(upper,kind='stable')
        upper,actual=upper[order],actual[order]
        rank=np.arange(1,len(group)+1)
        if np.any(np.diff(upper)<0):raise AssertionError('Recovery bound sorting failed')
        ax.bar(rank,actual,width=.90,color=BLUE,alpha=.85,linewidth=0,label='Observed squared error',zorder=2)
        ax.plot(rank,upper,color=ORANGE,lw=1.8,label='Recovery upper bound',zorder=3)
        ax.set(xlabel='Sample (ordered by bound)',ylabel='Squared recovery error',xlim=(.4,len(group)+.6),ylim=(0,None))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True,nbins=5))
        ax.set_title(rf'$s={key[2].split("_s")[-1]},\ L={key[1]}$',fontsize=size,pad=9)
        ax.text(-.12,1.04,chr(ord('a')+index),transform=ax.transAxes,fontweight='bold',fontsize=size+2)
        ax.grid(axis='y',alpha=.18,lw=.5);ax.set_axisbelow(True)
        if index==0:ax.legend(loc='upper left',fontsize=size-3)
        count_mismatch=sum(str(r.get('original_float32_same_spike_pattern','')).lower()=='false' for r in group)
        stats.append(dict(checkpoint=key[0],depth=key[1],condition=key[2],samples=len(group),
            output_threshold=next(iter(thresholds)),original_float32_path_mismatches=count_mismatch,
            bound_to_actual_sum=float(upper.sum()/actual.sum()) if actual.sum()>0 else None,
            comparison='float64 mathematical replay',bound_key=bk,error_key=ek))
        for j,original in enumerate(order):
            r=group[int(original)]
            plotted.append(dict(checkpoint=key[0],condition=key[2],depth=key[1],sorted_rank=j+1,
                original_sample=r['sample'],actual_sse=float(actual[j]),upper_sse=float(upper[j]),
                target_energy=numeric(r,'target_energy') if 'target_energy' in r else None))
    for ax in axes.ravel()[len(keys):]:ax.set_visible(False)
    finish(fig,out,'recovery_bounds')
    csv_write(out/'recovery_bounds_data.csv',plotted)
    return stats


def firing_figure(rows,out,size,zoom_s,ideal_only=False):
    groups={}
    for row in rows:
        for key in ('training_condition','depth','test_s','firing_rate','firing_bound_raw','firing_ideal'):
            if key not in row:raise ValueError('Activity input lacks '+key)
        if int(row.get('firing_bound_violations',0)) or int(row.get('feedback_bound_violations',0)):
            raise ValueError('Activity report contains inequality violations')
        key=(str(row['checkpoint']),int(row['depth']),str(row['training_condition']))
        groups.setdefault(key,[]).append(row)
    if not groups:raise ValueError('No activity rows')
    if len(groups)>4:raise ValueError('More than four fixed models; choose a narrower activity run.')
    keys=sorted(groups,key=lambda k:(k[1],k[2],k[0]))
    fig,axes=plt.subplots(2,len(keys),figsize=(5.5*len(keys),7.5),squeeze=False,layout='constrained')
    plotted,stats=[],[]
    for column,key in enumerate(keys):
        group=sorted(groups[key],key=lambda r:int(r['test_s']))
        x=np.array([int(r['test_s']) for r in group])
        if len(set(x))!=len(x):raise ValueError('Duplicate test sparsities; do not pool runs/checkpoints')
        actual=np.array([numeric(r,'firing_rate') for r in group])
        upper=np.array([numeric(r,'firing_bound_raw') for r in group])
        ideal=np.array([numeric(r,'firing_ideal') for r in group])
        counts=np.array([int(r['samples']) for r in group])
        if np.any(counts<=0):raise ValueError('Nonpositive activity sample count')
        has_sem=all('firing_rate_std' in r and int(r['samples'])>1 for r in group)
        sem=np.array([numeric(r,'firing_rate_std') for r in group])/np.sqrt(counts) if has_sem else None
        if has_sem and np.any(sem<0):raise ValueError('Negative standard deviation')
        dimensions={int(r['N']) for r in group if 'N' in r}
        if len(dimensions)>1:raise ValueError('Inconsistent signal dimension within a model')
        N=next(iter(dimensions)) if dimensions else None
        if N is not None and not np.allclose(ideal,x/(2*N),rtol=1e-10,atol=1e-12):
            raise ValueError('Ideal-rate values do not match s/(2N)')
        if np.any(actual>upper+1e-10) or np.any(actual<0):raise ValueError('Invalid activity bound/mean')
        if np.any(actual>.5+1e-10):raise ValueError('Firing-rate normalization disagrees with two-channel convention')
        if len(x)>1:
            gap=np.diff(x).astype(float)
            widths=.7*np.minimum(np.r_[gap[0],gap],np.r_[gap,gap[-1]])
            widths=np.minimum(widths,2.5)
        else:widths=np.ones(len(x))*.7
        for row_index,mask in enumerate((np.ones(len(x),bool),x<=zoom_s)):
            ax=axes[row_index,column]
            if not np.any(mask):raise ValueError('No data within the requested sparse-regime zoom')
            xx=x[mask]
            ax.bar(xx,actual[mask],width=widths[mask],color=BLUE,alpha=.85,linewidth=0,
                   label='Observed firing rate',zorder=2)
            if has_sem:
                ax.errorbar(xx,actual[mask],yerr=sem[mask],fmt='none',ecolor='#222222',
                            elinewidth=.7,capsize=2,capthick=.7,zorder=5)
            ax.plot(xx,ideal[mask],'--',color='#333333',lw=1.6,
                    label=r'Ideal-support reference: $s/(2N)$' if ideal_only else r'$s/(2N)$',zorder=3)
            if not ideal_only:
                ax.plot(xx,upper[mask],'-o',color=ORANGE,lw=1.6,ms=3.3,label='Full firing-rate bound',zorder=4)
            ax.set(xlabel=r'Sparsity $s$',ylabel='Firing rate',ylim=(0,None))
            if row_index==0:
                ax.set_title(rf'$L={key[1]}$',fontsize=size,pad=9)
            else:
                ax.set_title(rf'Low sparsity: $s\leq {zoom_s}$',fontsize=size-1,pad=9)
                ax.set_xticks(xx)
            ax.set_xlim(float(xx.min())-.8,float(xx.max())+max(.8,float(widths[mask][-1])))
            ax.ticklabel_format(axis='y',style='plain',useOffset=False)
            ax.text(-.12,1.04,chr(ord('a')+row_index*len(keys)+column),transform=ax.transAxes,
                    fontweight='bold',fontsize=size+2)
            ax.grid(axis='y',alpha=.18,lw=.5);ax.set_axisbelow(True)
            if row_index==0 and column==0:ax.legend(loc='upper left',fontsize=size-3)
        for j,(r,a,u,i) in enumerate(zip(group,actual,upper,ideal)):
            plotted.append(dict(checkpoint=key[0],depth=key[1],training_condition=key[2],test_s=int(r['test_s']),
                samples=int(r['samples']),observed_rate=float(a),observed_rate_sem=float(sem[j]) if has_sem else None,
                full_raw_bound=float(u),ideal_s_over_2N=float(i),N=N))
        stats.append(dict(checkpoint=key[0],depth=key[1],training_condition=key[2],
            test_sparsities=x.tolist(),samples_per_s=[int(r['samples']) for r in group],
            full_bound_exceeds_physical_rate_max_at=x[upper>.5].tolist(),weights_fixed=True,N=N,
            mean_rate_exceeds_ideal_reference_at=x[actual>ideal+1e-12].tolist(),
            error_bars='one standard error of mean' if has_sem else None))
    name='firing_ideal_reference' if ideal_only else 'firing_bounds'
    finish(fig,out,name)
    csv_write(out/(name+'_data.csv'),plotted)
    return stats


def run_audit(script, argv, destination):
    path=Path(__file__).resolve().with_name(script)
    if not path.is_file():raise FileNotFoundError('Keep '+script+' beside this plotting script (included in the ZIP).')
    command=[sys.executable,str(path),*argv,'--output',str(destination)]
    print('Running '+script+' ...',flush=True)
    subprocess.run(command,check=True)
    reports=sorted(destination.glob('*/report.json'))
    if len(reports)!=1:raise RuntimeError('Expected exactly one audit output in '+str(destination))
    report=json.loads(reports[0].read_text());check_report(report)
    return reports[0].parent


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run-audits',action='store_true')
    parser.add_argument('--sources',nargs='+',default=['outputs/seed42/checkpoints'])
    parser.add_argument('--project-dir',type=Path,default=Path('.'))
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--error-conditions',nargs='+',default=['m141_s14','m141_s28','m141_s42'])
    parser.add_argument('--error-depths',nargs='+',type=int,default=[10])
    parser.add_argument('--error-samples',type=int,default=1000)
    parser.add_argument('--activity-conditions',nargs='+',default=['m141_s28'])
    parser.add_argument('--activity-depths',nargs='+',type=int,default=[10])
    parser.add_argument('--activity-samples',type=int,default=5000)
    parser.add_argument('--sparsities',nargs='+',type=int,default=[0,1,2,3,4,5,7,10,14,20,28,35,42,49,56])
    parser.add_argument('--seed',type=int)
    parser.add_argument('--error-results',type=Path)
    parser.add_argument('--activity-results',type=Path)
    parser.add_argument('--zoom-s',type=int,default=7)
    parser.add_argument('--font-size',type=float,default=14.)
    parser.add_argument('--output',type=Path,default=Path('slista_paper_figures'))
    args=parser.parse_args()
    if args.error_samples<=0 or args.activity_samples<=0 or args.threads<=0 or args.zoom_s<1 or not (args.font_size>0 and math.isfinite(args.font_size)):
        parser.error('Samples, threads, zoom limit, font size must be positive')
    if args.run_audits and (args.error_results or args.activity_results):
        parser.error('Choose --run-audits or existing --error-results/--activity-results')
    if not args.run_audits and not (args.error_results or args.activity_results):
        parser.error('Supply --run-audits or existing audit results')
    root=args.output.expanduser().resolve()/dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    root.mkdir(parents=True,exist_ok=False)
    if args.run_audits:
        common=['--sources',*args.sources,'--project-dir',str(args.project_dir.resolve()),
                '--device',args.device,'--threads',str(args.threads)]
        if args.seed is not None:common+=['--seed',str(args.seed)]
        # Activity sweep is inexpensive and immediately supplies its figure.
        args.activity_results=run_audit('audit_slista_activity.py',common+[
            '--conditions',*args.activity_conditions,'--depths',*map(str,args.activity_depths),
            '--sparsities',*map(str,args.sparsities),'--samples',str(args.activity_samples),'--no-plots'],root/'activity_audit')
        font=style(args.font_size)
        activities=activity_rows(args.activity_results)
        expected={(c,L,s) for c in args.activity_conditions for L in args.activity_depths for s in set(args.sparsities)}
        obtained={(r['training_condition'],int(r['depth']),int(r['test_s'])) for r in activities}
        if not expected.issubset(obtained):raise RuntimeError('Some requested activity models/sparsities were not audited')
        if any(int(r['samples'])!=args.activity_samples for r in activities):raise RuntimeError('Activity sample count mismatch')
        firing_figure(activities,root,args.font_size,args.zoom_s)
        firing_figure(activities,root,args.font_size,args.zoom_s,ideal_only=True)
        print('Activity figure complete. Starting recovery certificates; these optimizations are the slow part.',flush=True)
        args.error_results=run_audit('audit_slista_joint_bound.py',common+[
            '--conditions',*args.error_conditions,'--depths',*map(str,args.error_depths),
            '--samples',str(args.error_samples)],root/'recovery_audit')
    font=style(args.font_size)
    metadata=dict(font=font,arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    if args.error_results:
        rows=error_rows(args.error_results.expanduser().resolve())
        if args.run_audits:
            expected={(c,L) for c in args.error_conditions for L in args.error_depths}
            obtained={(r['condition'],int(r['depth'])) for r in rows}
            if not expected.issubset(obtained):raise RuntimeError('Some requested recovery models were not audited')
            counts={}
            for r in rows:counts[str(r['checkpoint'])]=counts.get(str(r['checkpoint']),0)+1
            if any(n!=args.error_samples for n in counts.values()):raise RuntimeError('Recovery sample count mismatch')
        metadata['recovery']=recovery_figure(rows,root,args.font_size)
    if args.activity_results:
        activities=activity_rows(args.activity_results.expanduser().resolve())
        metadata['activity']=firing_figure(activities,root,args.font_size,args.zoom_s)
        firing_figure(activities,root,args.font_size,args.zoom_s,ideal_only=True)
    metadata['notes']=[
        'Recovery curves are sorted by the plotted bound; actual-error bars share exactly that order.',
        'Recovery values are per-sample squared error, not dB or individually normalized NMSE.',
        'The recovery numerical certificate uses the float64 mathematical replay and oracle support/amplitude priors.',
        'Firing bars and lines are sample means; the full raw bound is displayed even when it exceeds 0.5.',
        'Where available, activity error bars show one sample standard deviation divided by sqrt(sample count).',
        'The s/(2N) line is an ideal-support upper benchmark; equality and convergence toward it are not assumed.',
        'The ideal-reference figure retains every observed mean, including values above s/(2N); it does not claim training reduces the correction.',
        'Low-sparsity panels enlarge existing values without changing sample selection.',
        'Each activity curve uses fixed weights and unchanged nonzero-amplitude distribution across s.',
        'These plots concern static, noiseless toy S-LISTA T=1; they do not establish wireless/streaming robustness.']
    json_write(root/'figure_metadata.json',metadata)
    captions=[]
    if 'recovery' in metadata:
        counts=', '.join(f"{v['condition']}, L={v['depth']}: {v['samples']} samples" for v in metadata['recovery'])
        captions.append('Recovery figure. Per-sample squared recovery error (bars) and its conditional upper certificate (line). '
            'Samples are ordered independently within each panel by increasing certificate value; bars use the same order. '
            'The certificate uses the trained operators, observed pulse pattern, and true-support/amplitude priors. '
            'All samples are retained. Values correspond to the float64 mathematical replay. '+counts+'.')
    if 'activity' in metadata:
        counts='; '.join(f"training {v['training_condition']}, N={v['N']}, L={v['depth']}, samples per s={v['samples_per_s']}" for v in metadata['activity'])
        captions.append('Ideal-support reference figure. Mean firing rate (bars) and s/(2N) (dashed line). '
            'The dashed line is an upper bound under the sufficient condition that all intermediate cumulative codes '
            'remain on the true support; it is not an unconditional upper bound for freely trained networks. '
            'Models and nonzero-amplitude distributions are held fixed across the sparsity sweep. All observations '
            'are retained, including means exceeding the dashed reference. The lower panel enlarges low sparsities. '
            'Error bars, where available, denote one standard error of the observed mean across test signals. '+counts+'.')
        captions.append('Activity figure. Mean firing rate (bars), ideal-support benchmark s/(2N) (dashed line), and full '
            'activity upper bound including off-support cumulative activity (solid line). Each model is held fixed throughout '
            'the sparsity sweep. The lower row enlarges the low-sparsity region. Rates are normalized over N coordinates, L '
            'layers and two signed channels. Error bars, where available, denote one standard error of the observed mean '
            'across test signals for a fixed trained model. The full raw bound is shown without the trivial 0.5 cap. '+counts+'.')
    (root/'captions.txt').write_text('\n\n'.join(captions)+'\n',encoding='utf-8')
    # Return a compact deliverable for review: figures, exact plot data, metadata,
    # captions, audit summaries, and recovery sample rows; exclude large NPZ files.
    with zipfile.ZipFile(root/'figures_and_data.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(root.iterdir()):
            if path.is_file() and path.suffix in ('.pdf','.png','.csv','.json','.txt'):
                z.write(path,path.name)
        for label,source in [('recovery_audit',args.error_results),('activity_audit',args.activity_results)]:
            if source and Path(source).is_dir():
                for name in ('report.json','summary.csv','samples.csv'):
                    path=Path(source)/name
                    if path.is_file():z.write(path,label+'/'+name)
    print('Figures and exact plotted data:',root,flush=True)
    print('Upload figures_and_data.zip for the next manuscript step.',flush=True)


if __name__=='__main__':
    main()
