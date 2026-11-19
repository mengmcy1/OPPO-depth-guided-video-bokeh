"""Produce machine-readable stage 2 milestone tables from completed artifacts."""
from ast import literal_eval
from collections import defaultdict, Counter
import csv
import re
import shutil

import numpy as np

from route3_ovg_1b_common import REC, ROOT, read
from route3_ovg_stage2_data import OUT, PREFIX
from route3_ovg_stage2_evaluate import FLOORS, STAGES, path_for
from route3_ovg_subpixel import save_json
from route3_upper_bound_evaluate import pool

OPTICAL=('stage','scene','protocol','aperture','nominal_focus_m','method','weights','split','second','region')
SPLIT_MARKS=('boundary_2px','zero_boundary_2px','zero_full_image','in_focus','recoverable')


def csv_file(suffix,rows):
    rows=list(rows)
    fn=REC/f'{PREFIX}-{suffix}.csv'
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with fn.open('w',newline='') as f:
        writer=csv.DictWriter(f,keys,lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    return str(fn.relative_to(ROOT))


def summary(rows):
    sums=defaultdict(list)
    for r in rows:
        for s in (r['scene'],'pooled'):
            dimensions=tuple(r[k] if k!='scene' else s for k in OPTICAL)
            sums[dimensions].append(r)
    return {k:dict(zip(OPTICAL,k),cases=len(values),**pool(values))
            for k,values in sorted(sums.items())}


def compare(aggregate,key,reference):
    a,b=aggregate[key],aggregate[reference]
    if a['pixels']!=b['pixels'] or a['cases']!=b['cases']:
        raise ValueError(f'Photometric pairing uses different masks {key}')
    return dict({k:a[k] for k in OPTICAL},reference_split=b['split'],reference_second=b['second'],
                cases=a['cases'],pixels=a['pixels'],psnr=a['psnr'],reference_psnr=b['psnr'],
                PSNR_delta=a['psnr']-b['psnr'],patch_lpips=a['patch_lpips'],
                reference_patch_LPIPS=b['patch_lpips'],LPIPS_delta=a['patch_lpips']-b['patch_lpips'])


def optical_pairs(data):
    query,split=[],[]
    for k,v in data.items():
        if not v['pixels'] or v['split']=='standalone':
            continue
        if v['second'] and (*k[:8],False,k[9]) in data:
            query.append(compare(data,k,(*k[:8],False,k[9])))
        if v['split'] in ('retrieved','oracle') and v['region'] in SPLIT_MARKS:
            other=(*k[:7],'none',*k[8:])
            if other in data:
                split.append(compare(data,k,other))
    return query,split


def optics_guards(rows):
    for stage in STAGES:
        for weights in ('EMA','raw'):
            for name,region in (('visible','zero_visible_full_image'),('combined','zero_full_image')):
                values=[r for r in rows if r['stage']==stage and r['method']=='OVG' and r['weights']==weights
                        and r['split']=='retrieved' and r['second'] and r['region']==region]
                if values:
                    yield dict(stage=stage,weights=weights,check=name,threshold_db=FLOORS[name],cases=len(values),
                        passed=sum(r['psnr']>=FLOORS[name] for r in values),min_db=min(r['psnr'] for r in values),
                        pooled_PSNR_db=pool(values)['psnr'])


def interior_rates(rows):
    from route3_ovg_1b_common import OUT as DATA
    regions={}
    sums=defaultdict(Counter)
    for r in rows:
        if not (r['protocol']=='P2' and r['aperture']==1.4 and r['nominal_focus_m']==1.4
                and r['second'] and r['region']=='full_image' and r['split']=='retrieved'
                and r['method']=='OVG'):
            continue
        scene,frame=r['scene'],r['frame']
        if (scene,frame) not in regions:
            path=DATA/'B'/scene/'pinhole-subpixels'/f'f{frame:03d}.npz'
            with np.load(path) as y:
                regions[scene,frame]=int(((y['alpha']>=.9999)&y['valid_alpha']&~y['depth_boundary']).sum())
        count=regions[scene,frame]
        for name in (scene,'pooled'):
            sums[r['stage'],r['weights'],name].update(frames=1,interior_pixels=count,
                false_pixels=round(r['false_transparent']*count),total_pixels=r['pixels'],
                split_pixels=round(r['b']*r['pixels']))
    return [dict(stage=s,weights=w,scene=n,**v,
                 false_semitransparent_rate=v['false_pixels']/v['interior_pixels'],
                 first_split_rate=v['split_pixels']/v['total_pixels'])
            for (s,w,n),v in sorted(sums.items())]


def leakage_stats(diagnostics):
    groups=defaultdict(list)
    for r in diagnostics:
        for scope in (r['scene'],'pooled'):
            groups[scope,r['method'],r['nominal_focus_m']].append(r)
    results,classification=[],[]
    for (scope,method,focus),cases in sorted(groups.items()):
        per_case=[]
        totals=defaultdict(Counter)
        for row in cases:
            if not row['pixels']:
                continue
            source=path_for(row['stage'],row['scene'],row['frame'])/row['protocol']/row['id']
            with np.load(source/f'leakage-{method}.npz') as data:
                per_case.append({k:data[k].ravel() for k in ('T','query_weight','query_front')})
            for cat in row['classes']:
                totals[cat['cause']].update({k:cat[k] for k in ('rays','positive_SSE','net_SSE')})
        result=dict(scene=scope,method=method,focus=focus,frames=len(per_case),
                    pixels=sum(r['pixels'] for r in cases),rays=sum(r['rays'] for r in cases),
                    delta_SSE=sum(r.get('delta_SSE',0) for r in cases),
                    positive_SSE=sum(r.get('positive_SSE',0) for r in cases),
                    q_rays=sum(r.get('significant_query_rays',0) for r in cases))
        if per_case:
            for field in ('T','query_weight','query_front'):
                values=np.concatenate([r[field] for r in per_case])
                result.update({f'{field}_{name}':x for name,x in (
                    ('mean',float(values.mean())),('median',float(np.median(values))),
                    ('p95',float(np.quantile(values,.95))),('max',float(values.max())))})
        results.append(result)
        for cat,v in totals.items():
            classification.append(dict(scene=scope,method=method,focus=focus,cause=cat,**v,
                                       ray_fraction=v['rays']/result['rays'],
                                       positive_SSE_fraction=v['positive_SSE']/max(1e-24,result['positive_SSE'])))
    return results,classification


def terminal_summary(aggregate,query_pairs,parts):
    main=[r for r in query_pairs if r['stage']=='2b' and r['method']=='OVG'
          and r['weights']=='EMA' and r['split']=='retrieved' and r['scene']=='pooled'
          and r['aperture']==1.4 and r['protocol']=='P2']
    guard=next(r for r in main if r['nominal_focus_m']==1.4 and r['region']=='in_focus')
    tags=dict(stage='2b',scene='pooled',protocol='P2',aperture=1.4,method='OVG',weights='EMA',
              second=True,region='boundary_2px')
    optical={}
    for split in ('none','oracle','retrieved'):
        rows=[aggregate[tuple(dict(tags,split=split,nominal_focus_m=f)[k] for k in OPTICAL)] for f in (1.4,5.)]
        optical[split]=pool(rows)
    baseline=optical['none']['psnr']
    oracle=optical['oracle']['psnr']
    actual=optical['retrieved']['psnr']
    if oracle<=baseline:
        decision='B_gt does not improve pooled boundary PSNR; retain unsplit'
    elif actual<oracle and actual<=baseline:
        decision='B_gt improves pooled boundary PSNR, retrieved background loses it; memory/candidate quality'
    elif actual<oracle:
        decision='B_gt and retrieval both improve pooled boundary PSNR; retrieval preserves only part of the gain'
    else:
        decision='Retrieved background preserves the pooled boundary PSNR gain'
    evidence=[r for r in parts if r['scene']=='pooled' and r['method']=='2b' and r['focus']==1.4
              and r['cause']!='below_weight_or_nonharmful']
    dominant=max(evidence,key=lambda x:x['positive_SSE'],default=None)
    proposal={
        'front_order':'inspect placement and depth ordering',
        'surface_gap':'use both directions of L_occ to constrain learned footprint and opacity',
        'color_vs_F2':'candidate and memory appearance quality'
    }
    return dict(second_query_near_guard_passed=guard['PSNR_delta']>=-.1,
                second_query_near_PSNR_delta_db=guard['PSNR_delta'],
                edge_reference_cases='two nominal foci, P2 f/1.4, EMA, second Query enabled',
                B_gt_effect=optical,split_decision=decision,
                largest_attributed_near_positive_SSE=dominant,
                next_check_from_ray_evidence=proposal.get(dominant['cause']) if dominant and dominant['positive_SSE']>0 else None)


def curves():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    stats,monitors=[],[]
    for stage,color in (('2a','#2067a5'),('2b','#c26b21')):
        dest=OUT/stage
        report=read(dest/'training.json')
        curve=read(dest/f'step-{report["steps"]:06d}-curve.json')
        offset=0 if stage=='2a' else 10000
        axes[0,0].plot([offset+i for i in range(200,len(curve)+1,200)],
                       [np.mean([x['loss'] for x in curve[i:i+200]]) for i in range(0,len(curve),200)],
                       color=color,label=stage)
        stats.append(dict(stage=stage,**report))
        series=defaultdict(list)
        for file in sorted(dest.glob('step-*-monitor.json')):
            step=int(file.name.split('-')[1])
            for weights,rows in read(file).items():
                for row in rows:
                    record=dict(stage=stage,weights=weights,step=step,scene=row['scene'],
                                b=row['b'],false_transparent=row['false_transparent'])
                    for kind,checks in ((k,row[k]) for k in ('visible','combined')):
                        record.update({f'{kind}_{region}':v for region,v in checks.items()})
                    monitors.append(record)
                for field in ('visible','combined'):
                    series[weights,field].append((step+offset,min(r[field]['full'] for r in rows)))
                series[weights,'b'].append((step+offset,np.mean([r['b'] for r in rows])))
        for (weights,field),points in series.items():
            ax={'visible':axes[0,1],'combined':axes[1,0],'b':axes[1,1]}[field]
            ax.plot(*zip(*points),'-' if weights=='EMA' else '--',color=color,label=f'{stage} {weights}')
    for ax,title in zip(axes.flat,['Mean training loss per 200 steps','Visible only: worst dev zero aperture (dB)',
                    'With hidden + F2: worst dev zero aperture (dB)','Mean first-pass split fraction (dev)']):
        ax.set(title=title,xlabel='Optimizer step')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    axes[0,1].axhline(50,color='grey')
    axes[1,0].axhline(60,color='grey')
    path=OUT/'S2-training-curves.png'
    fig.savefig(path,dpi=150)
    plt.close(fig)
    return stats,monitors,str(path.relative_to(ROOT))


def restart_evidence():
    logs=ROOT/'logs'/PREFIX
    previous=logs/'S2-formal.log'
    resumed=sorted(logs.glob('S2-resume-*.log'))
    if not previous.is_file() or not resumed:
        return None

    def points(path):
        values={}
        for line in path.read_text().splitlines():
            if line.startswith('STAGE2 {') and "'stage': '2b'" in line:
                row=literal_eval(re.sub(r'Counter\((\{[^}]*\})\)',r'\1',line[len('STAGE2 '):]))
                values[row['step']]=row
        return values

    before=points(previous)
    after={}
    for path in resumed:
        after.update(points(path))
    common=sorted(before.keys() & after.keys())
    return dict(last_logged_step_before_interruption=max(before,default=None),
                restart_logs=[str(p.relative_to(ROOT)) for p in resumed],
                restarts=[entry for p in resumed for entry in p.read_text().splitlines() if entry.startswith('RESUME ')],
                replay_logged_steps=common,
                replay_tier_counts_match=all(before[i]['counts']==after[i]['counts'] for i in common),
                replay_max_abs_logged_loss_difference=max((abs(after[i]['loss']-before[i]['loss']) for i in common),default=None),
                peak_allocated_MiB=max((r['peak_MiB'] for r in (*before.values(),*after.values())),default=None),
                note='Model/EMA/optimizer and explicit batch/pixel RNG restored. Cache hits and CUDA rendering may preclude bitwise-identical loss replay.')


def report():
    reports=[read(OUT/f'S3-{s}.json') for s in STAGES]
    rows=[r for part in reports for r in part['rows']]
    aggregate=summary(rows)
    query,split=optical_pairs(aggregate)
    leaks=[r for part in reports for r in part['leakage']]
    leak,causes=leakage_stats(leaks)
    fit,mon,plot=curves()
    frozen=read(OUT/'S3-frozen-boundary.json')
    interior=interior_rates(rows)
    files=dict(optical={s:csv_file(f'{s}-optical',(r for r in aggregate.values() if r['stage']==s))
                        for s in STAGES},
               primary=csv_file('S3-primary',(r for r in aggregate.values() if r['protocol']=='P2'
                                  and r['aperture']==1.4 and r['weights']!='raw')),
               query=csv_file('S3-query-pairs',query),split=csv_file('S3-boundary-pairs',split),
               interior=csv_file('S3-interior-and-b',interior),
               leakage=csv_file('S3-leakage',leak),causes=csv_file('S3-leakage-classes',causes),
               frozen_boundary=csv_file('S3-frozen-boundaries',frozen['rows']),
               zero=csv_file('S3-zero-floors',optics_guards(rows)),monitor=csv_file('S2-monitor',mon))
    save_json(REC/f'{PREFIX}-summary.json',dict(
        starting_revision='dba02b4',output=str(OUT.relative_to(ROOT)),S0=read(OUT/'S0.json'),
        S1=read(OUT/'S1.json'),geometry_frozen_after=read(OUT/'S1-frozen-after.json'),
        S2=dict(stages=fit,curves=plot,recovery=restart_evidence()),
        disk_free_GiB=shutil.disk_usage(OUT).free/2**30,S3_runtime=[dict(stage=s,seconds=d['seconds'],peak_MiB=d['peak_MiB'])
            for s,d in zip(STAGES,reports)],S3_scope='three development scenes, exact K64 finite aperture and separately K16 for all blocked-ray checks',
        S3_guards=list(optics_guards(rows)),S3_interior=interior,milestone=terminal_summary(aggregate,query,causes),files=files))


if __name__=='__main__':
    report()
