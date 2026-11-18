"""Summarize and select only from completed revision 14 experiments."""
import argparse
import csv
import shutil
import subprocess

from route3_g3_r14 import C, OUT, PREFIX, REC, ROOT, TRUTH, read, write
from route3_upper_bound_evaluate import pool
from route3_g3_r14_data import BASE

VERSIONS = ['original','flow','depthsplat-official']
T4 = OUT/'T4'


def table(path, rows):
    if rows:
        with path.open('w') as f:
            writer = csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)


def choose_geometry():
    by_version = {}
    for v in VERSIONS:
        p = T4/f'r14-T4-{v}.json'
        if p.exists():
            stats = read(p)
            if len(stats['rows']) != 720:
                raise ValueError(f'Incomplete T4 scoring for {v}')
            by_version[v] = stats
    if set(by_version) != set(VERSIONS):
        return None
    base = by_version['original']['pooled']
    sel = lambda r:r['scene'].startswith('metric-pilot-') and r['focus']==5 and r['fstop']==1.4 and r['region']=='recoverable'
    score_a = pool([r for r in base if sel(r) and r['method']=='A'])['psnr']
    score_gt = pool([r for r in base if sel(r) and r['method']=='GT_D1'])['psnr']
    original = pool([r for r in base if sel(r) and r['method']=='D1'])['psnr']
    rows = []
    details = []
    zero = []
    for v, values in by_version.items():
        refocus = pool([r for r in values['pooled'] if sel(r) and r['method']=='D1'])['psnr']
        d0 = pool([r for r in values['pooled'] if sel(r) and r['method']=='D0'])['psnr']
        rows.append(dict(version=v,D0_PSNR=d0,D1_PSNR=refocus,A_PSNR=score_a,GT_D1_PSNR=score_gt,
                         D1_minus_A_dB=refocus-score_a, D1_minus_D0_dB=refocus-d0,
                         geometry_gap_recovered=(refocus-original)/(score_gt-original)))
        for r in values['pooled']:
            if r['method'] not in ['D0','D1']:
                continue
            keys = ['scene','focus','fstop','region']
            o = {x['method']:x for x in base if all(x[k]==r[k] for k in keys)}
            method = dict(version=v, **r)
            decline = r['psnr']-o['A']['psnr'] if r['psnr'] is not None else None
            gap = o['GT_D1']['psnr']-o['D1']['psnr'] if o['GT_D1']['psnr'] is not None else None
            gain = r['psnr']-o['D1']['psnr'] if r['psnr'] is not None else None
            details.append(dict(**method, A_PSNR=o['A']['psnr'], vs_A_dB=decline,
                                GT_selected_focus_PSNR=o['GT_D1']['psnr'], original_D1_PSNR=o['D1']['psnr'],
                                raw_gap_ratio=gain/gap if r['method']=='D1' and gap and abs(gap)>1e-8 else None,
                                gap_recovery_ratio=gain/gap if r['method']=='D1' and gap and gap>1e-8 else None))
        for r in values['rows']:
            if r['method'] in ['D0','D1','A'] and r['region']=='full_image':
                same = next(x for x in values['rows'] if all(x[k]==r[k] for k in ['scene','frame','id'])
                            and x['region']=='full_image' and x['method']=='A')
                zero.append(dict(version=v, scene=r['scene'], frame=r['frame'],id=r['id'],
                                 method=r['method'], PSNR=r['zero_PSNR'], vs_A_dB=r['zero_PSNR']-same['zero_PSNR']))
    chosen = max((r for r in rows if r['version']!='original'),key=lambda r:r['D1_PSNR'])['version']
    report = dict(default_geometry=chosen, comparison=rows, pixels='pooled SSE and counts',configs='2 G2 scenes, f/1.4, P2 far, all3 frames',
                  same_ray_F2=True, A_focus_unchanged=True, original_comparator='original explicit depth with its own refocused P2',
                  depth_fits='placement only;GT depth not used by estimator', branches=VERSIONS,
                  selection_candidates=VERSIONS[1:],original_retained_for_comparison=True)
    write(T4/'selection.json',report)
    write(REC/f'{PREFIX}-T4-selection.json',report)
    table(REC/f'{PREFIX}-T4-pooled.csv',details)
    table(REC/f'{PREFIX}-T4-zero.csv',zero)
    perframe = [read(p) for v in VERSIONS for p in sorted((T4/v).glob('*/*/geometry.json'))]
    first = ROOT/'outputs/route3-order-fallback-2026-11-17-01'
    for row in perframe:
        name,f=row['scene'],row['frame']
        key=f'{name}-' if name=='B2-high' else ''
        row['A_and_GT_P2_m']={}
        for nominal in [1.4,5.]:
            ref=read(first/name/f'f{f:03d}'/'P2'/f'{key}f{f:03d}-N1.4-F{nominal:g}'/'render.json')
            row['A_and_GT_P2_m'][str(nominal)]=dict(A=ref['focus_m'],D_GT=ref['GT_focus_m'])
    write(REC/f'{PREFIX}-T4-depth-focus.json',dict(rows=perframe))
    timing=[]
    for v in VERSIONS:
        frames=[read(p) for p in (T4/v).glob('*/*/*/render.json')]
        calls=[t for row in frames for t in row['timings']]
        timing.append(dict(version=v, configurations=len(frames), render_call_seconds=sum(t['seconds'] for t in calls),
                           peak_torch_MiB=max(t['peak_MiB'] for t in calls)))
    inference=[read(p) for p in (T4/'depthsplat-official-raw').glob('*/*.json')]
    write(REC/f'{PREFIX}-T4-runtime.json',dict(render=timing,depthsplat_inference=inference,
          official_bounds_m=[.5,200.],DepthSplat_code_and_weight_record='route3-g3-r13-2026-11-18-model-licenses.json',
          timings_exclude_GPU_queue_and_CPU_scoring=True))
    return report


def get_split():
    done = [OUT/f'{v}-done.json' for v in ['broad','multilayer','thin']]
    if not all(p.exists() for p in done[:2]):
        return None
    progress = OUT/'thin-progress.json'
    if progress.exists() and any(not r['passed'] for r in read(progress)):
        thin_ok = False
    elif done[-1].exists():
        thin_ok = read(done[-1])['passed']
    else:
        return None
    split = {k:[name for name in C['split'][k] if thin_ok or not name.endswith('-thin')] for k in ['train','heldout']}
    write(OUT/'split.json',dict(**split, thin_excluded=not thin_ok))
    return split


def fit_plot(folder, fit):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(9,3.5))
    curve=fit['curve']
    axes[0].plot([r['step'] for r in curve],[r['total'] for r in curve],alpha=.45,label='Batch loss')
    steps=[0]+[r['step'] for r in fit['validation']]
    review=[fit['initial']]+[r['rows'][0] for r in fit['validation']]
    axes[0].plot(steps,[r['loss'] for r in review],marker='o',label='All active queries')
    for key,label in [('selection_correct_rate','Correct selection'),('false_activation_rate','False activation')]:
        axes[1].plot(steps,[r['metrics'][key] for r in review],marker='o',label=label)
    for axis in axes:
        axis.set_xlabel('Step')
        axis.legend()
        axis.grid(alpha=.2)
    axes[0].set_ylabel('Direct supervision loss')
    axes[1].set_ylim(0,1.02)
    fig.suptitle(folder.name+' | frame 30, f/1.4, far focus')
    fig.tight_layout()
    path=folder/'fit-curves.png'
    fig.savefig(path,dpi=140)
    plt.close(fig)
    return str(path)


def report():
    selected = choose_geometry()
    split = get_split()
    rows = []
    floors = []
    sampling = []
    costs = []
    def size(path):
        return int(subprocess.check_output(['du','-sb',str(path)],text=True).split()[0])/2**30 if path.exists() else 0.
    for name in C['scenes']:
        p = TRUTH/name/'quality.json'
        if not p.exists():
            rows.append(dict(scene=name, passed=None, spp=None, phase_RMSE=None, period_p01=None,
                             zero_PSNR=None, seconds=None, new_GiB=None, reused=False, truth_aux=False,
                             original_cache=False))
            continue
        d = read(p)
        c = d['surface_checks']
        rows.append(dict(scene=name,passed=d['passed'],spp=d['final_spp'],
                         phase_RMSE=max(x.get('excess_RMSE',0) for x in c['alias']),
                         period_p01=min(x['period_p01_px'] for x in c['periods'] if x['period_p01_px'] is not None),
                         zero_PSNR=c['zero_aperture_repeat_PSNR'], seconds=d['seconds'],new_GiB=d['own_disk_GiB'],
                         reused='reused' in d, truth_aux=(p.parent/'truth-aux-ready.json').exists(),
                         original_cache=(BASE/name/'training-data-ready.json').exists()))
        floors.extend([dict(scene=name,**x) for x in d.get('texture_floors',[])])
        sampling.extend(dict(scene=name,spp=l['spp'],**s) for l in d['sampling'] for s in l['checks'])
        qdir=p.parent
        jobs=[j for file in sorted(qdir.glob('*-jobs-results.json')) for j in read(file)]
        seconds_by_output={j['output']:j['seconds'] for j in jobs if 'seconds' in j and not j.get('reused_existing')}
        aux=qdir/'truth-aux-ready.json'
        temporal=qdir/'temporal-ready.json'
        costs.append(dict(scene=name, truth_wall_seconds=d['seconds'],
                          Blender_job_seconds=sum(seconds_by_output.values()),
                          excludes_historical_GPU_time='reused' in d,
                          aux_wall_seconds=read(aux)['seconds'] if aux.exists() else None,
                          temporal_wall_seconds=read(temporal)['seconds'] if temporal.exists() else None,
                          truth_GiB=size(qdir),native_frontend_GiB=size(OUT/'geometry'/'native'/name),
                          original_geometry_GiB=size(BASE/name)+size(BASE/'learning-data'/name),
                          selected_geometry_GiB=(size(OUT/'geometry'/selected['default_geometry']/name)
                            +size(OUT/'geometry'/selected['default_geometry']/'learning-data'/name)) if selected else None))
    for label, data in [('scene-quality',rows),('bar-floors',floors),('sampling',sampling),('costs',costs)]:
        table(REC/f'{PREFIX}-{label}.csv',data)
    fits = []
    for name in C['scenes'][:2]:
        folder = OUT/'overfit'/name
        p = folder/'training.json'
        render = folder/'render.json'
        fit = read(p) if p.exists() else None
        if fit:
            write(REC/f'{PREFIX}-overfit-{name}.json',dict(**fit,render=read(render) if render.exists() else None,
                                                         plot=fit_plot(folder,fit)))
        fits.append(dict(scene=name,fit_completed=bool(fit),passed=fit['passed'] if fit else None,
                         render_completed=render.exists(),path=str(folder)))
    storage = shutil.disk_usage(OUT)
    blocked = OUT/'S2a-blocked.json'
    ready = OUT/'selected-data-ready.json'
    runtime = OUT/'overlap-budget.json'
    initial = T4/'initial-depthsplat-probe.json'
    valid={r['scene'] for r in rows if r['passed']}
    available={k:[n for n in names if n in valid] for k,names in split.items()} if split else None
    missing={k:[n for n in names if n not in valid] for k,names in split.items()} if split else None
    d = dict(base_revision='1b5787c', output=str(OUT), order='O1',fallback='F2',
             split=split, valid_scenes_by_split=available, unavailable_scenes_by_split=missing,
             scenes=rows, geometry=selected, overfit=fits,
             selected_data=read(ready) if ready.exists() else None,
             S2a_block=read(blocked) if blocked.exists() else None,
             minibatch_GPU_overlap=read(runtime) if runtime.exists() else None,
             initial_depthsplat_probe=read(initial) if initial.exists() else None,
             temporal_complete=[p.parent.name for p in TRUTH.glob('*/temporal-ready.json')],
             training=(OUT/'S2a/training.json').exists(), holdout_metrics=(OUT/'S2a/evaluation.json').exists(),
             free_GiB=storage.free/2**30, output_GiB=size(OUT))
    write(REC/f'{PREFIX}-summary.json',d)
    print(d['geometry'],d['split'],d['overfit'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['report','select'])
    a=p.parse_args()
    if a.mode=='report':
        report()
    else:
        print(choose_geometry(),flush=True)
