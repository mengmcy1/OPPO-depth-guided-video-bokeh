"""Write the fixed-step milestone and paired footprint/geometry evidence."""
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

from route3_ovg_r2_common import CONFIG, OUT, REC, ROOT, ROUND1
from route3_ovg_subpixel import save_json

PREFIX='route3-ovg-1a-r2-2026-11-18'


def read(filename):
    return json.loads(filename.read_text())


def write_table(label,rows):
    path=REC/f'{PREFIX}-{label}.csv'
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w') as f:
        w=csv.DictWriter(f,fields,lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
    return str(path.relative_to(ROOT))


def disk(path):
    return int(subprocess.check_output(['du','-sB1',str(path)],text=True).split()[0])


def footprint():
    root=OUT/'V1'
    val=read(root/'evaluation.json')
    train=read(root/'retrained/training.json')
    decision=read(root/'decision.json')
    assert train['steps']==20000 and len(val['rows'])==12
    entries=[]
    for x in val['rows']:
        rec=x['regions']['recoverable']
        entries.append(dict(scene=x['scene'],case=x['case'],focus=x['nominal_focus'],rule=x['method'],
                            zero_PSNR=x['zero_PSNR'],zero_vs_A_dB=x['zero_vs_A'],
                            below_60_dB=x['below_absolute_60'],
                            recoverable_psnr=rec['psnr'],recoverable_LPIPS=rec['patch_lpips'],
                            **{k:x['audit'].get(k,0) for k in ('total','moved','shrunk','dropped')}))
    labels=[]
    for p in sorted((root/'footprint-labels').glob('*/**/audit.json')):
        report=read(p)
        role='GT_warmup' if p.parent.name=='256-5-GTwarmup' else 'estimated'
        labels.extend(dict(scene=report['scene'],geometry=role,**r) for r in report['rows'])
    c=Counter()
    for r in labels:
        c.update({k:r[k] for k in ('true_background','dropped_true_background','shrunk_true_background')})
    epochs=dict(steps=train['steps'],seconds=train['seconds'],peak_allocated_MiB=train['peak_allocated_MiB'],
                 GT_warmup_steps=train['GT_warmup_steps'],
                 probe_initial=round(float(np.mean(train['train_probe'][0]['losses'])),8),
                 probe_final=round(float(np.mean(train['train_probe'][-1]['losses'])),8),
                 seed=train['seed'],training_manifest='seven estimated-geometry scenes plus existing warmup, all labels recomputed',
                 checkpoint=str((root/'retrained/step-020000.pt').relative_to(ROOT)),
                 curve=write_table('V1-Query-training',train['curve']))
    return dict(decision=decision,paired_table=write_table('V1-paired',entries),paired_rows=entries,
                label_counts=c,placement_table=write_table('V1-labels',labels),
                selected_after_retrain=val['measured_rule'],footprint_lost_configs=val['footprint_lost_configs'],
                combined=val['combined'],retrain=epochs,
                scale_handling='label footprint factor cached and inconsistent placements excluded; Query uses fixed raw surfel scale followed by deterministic footprint shrink in the renderer')


def corpus():
    sources={}
    for tier in ('G','C'):
        rows=[]
        for p in sorted((OUT/tier/'windows').glob('*/manifest.json')):
            raw=read(p)
            r=dict(scene=raw['scene'],frames=len(raw['rows']),linked_first_round=p.parent.is_symlink(),
                   prior_complete=all((p.parent/'prior'/f'f{i:03d}.npz').exists()
                                      for i in range(len(raw['rows']))))
            if tier=='G':
                r['environment']=raw['environment']
            elif 'family' in raw:
                r.update(family=raw['family'],seed=raw['seed'],baseline_m=raw['baseline_m'],
                         foreground_width_m=raw['foreground_width_m'],holes=len(raw['holes']),geometry_seconds=raw['seconds'])
            prior=p.parent/'prior/summary.json'
            if prior.exists():
                value=read(prior)
                r.update(prior_elapsed_s=value['seconds'],prior_GPU_s=value['GPU_forward_seconds'],
                         prior_bytes=value['bytes'],mean_prior_support=value['mean_supported_fraction'])
            elif r['linked_first_round']:
                r['prior_bytes']=sum(t.stat().st_size for t in (p.parent/'prior').glob('*.npz'))
            rows.append(r)
        sources[tier]=dict(windows=len(rows),frames=sum(r['frames'] for r in rows if r['prior_complete']),
                          new_windows=sum(not r['linked_first_round'] for r in rows),
                          incomplete_windows=[r['scene'] for r in rows if not r['prior_complete']],
                          prior_bytes=sum(r.get('prior_bytes',0) for r in rows),
                          prior_elapsed_s=sum(r.get('prior_elapsed_s',0) for r in rows),
                          prior_GPU_s=sum(r.get('prior_GPU_s',0) for r in rows),
                          disk_bytes_new_content=disk(OUT/tier),detail=write_table('V3-'+tier+'-shards',rows))
        if tier=='G':
            by=Counter()
            for r in rows:
                if r['prior_complete']:by[r['environment']]+=r['frames']
            public=read(OUT/'G/acquisition.json')
            sources[tier].update(environments=by,range_bytes_new=sum(r.get('range_bytes',0) for r in public['environments']),
                               acquisition=public)
        else:
            generation=read(OUT/'C/generation.json')
            sources[tier].update(generation=generation,families=Counter(r['family'] for r in rows if 'family' in r),
                         optical_scope='K16 sampled at each new clip middle frame, both focuses; subpixel labels and hidden layers on all frames')
    sources['B']=dict(read(ROUND1/'B/manifest.json')['split'],inputs_per_scene=13,train_frames=91,dev_frames=39,
                      labels_source=str((ROUND1/'B/manifest.json').relative_to(ROOT)),assets_sealed_test_excluded=True)
    thin=read(OUT/'B-thin-texture/summary.json')
    sources['thin_texture_mask']=dict(**{k:v for k,v in thin.items() if k!='rows'},
                     frames_table=write_table('V3-thin-mask',[dict(scene=r['scene'],frame=r['frame'],
                         pixels_blocked=r['pixels_blocked'],candidate_objects='|'.join(r['candidate_objects'])) for r in thin['rows']]))
    return sources


def figures(curve,validation):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    fig,axes=plt.subplots(1,2,figsize=(11.4,4.2))
    for tier in ('G','C','B'):
        values=[row for row in curve if row['tier']==tier]
        width=100
        y=np.convolve([r['total'] for r in values],np.ones(width)/width,mode='valid')
        steps=[r['step'] for r in values[width//2:width//2+len(y)]]
        axes[0].plot(steps,y,label=tier)
    for name in sorted({r['scene'] for r in validation[-1]['paired']}):
        rows=[]
        for d in validation:
            r=next(p for p in d['paired'] if p['scene']==name and p['prior']=='DepthSplat')
            rows.append((d['step'],r['delta_boundary_front_CoC_px']))
        axes[1].plot(*zip(*rows),marker='o',label=name.replace('metric-scene-',''))
    axes[1].axhline(0.,linestyle='--',color='grey',linewidth=1)
    axes[0].set(xlabel='Step',ylabel='Training loss (100 batches within each tier)')
    axes[1].set(xlabel='Fixed validation step',ylabel='Boundary-front CoC MAE minus DepthSplat (px)')
    for a in axes:a.legend(fontsize=8)
    fig.tight_layout()
    t=OUT/'training-curves.png';fig.savefig(t,dpi=150);plt.close(fig)
    return str(t.relative_to(ROOT))


def final_validation():
    entries=[read(OUT/'validation'/f'step-{step:06d}.json') for step in CONFIG['checkpoint_steps']]
    last=entries[-1]
    first=read(ROUND1/'validation/step-002400.json')
    measures=['front_inverse_mae','front_CoC_mae_px','boundary_front_CoC_mae_px',
              'boundary_back_CoC_mae_px','boundary_coverage_MAE','false_semitransparent']
    joined=[]
    lookup={(x['scene'],x['method']):x for x in first['pooled']}
    for v in last['pooled']:
        old=lookup.get((v['scene'],v['method']))
        if v['method']!='selected-prior-base':
            assert old is not None
            for group in ('front_n','edge_front_n','edge_back_n','alpha_edge_n','interior_n'):
                assert old[group]==v[group], (v['scene'],v['method'],group)
        joined.append(dict(scene=v['scene'],method=v['method'],domain=v['domain'],
                    **{k+'_r1':old[k] if old else None for k in measures},
                    **{k+'_r2':v[k] for k in measures},
                    **{k+'_r2_minus_r1':(v[k]-old[k]) if old and v[k] is not None and old[k] is not None else None for k in measures},
                    **{k:v[k] for k in ('front_n','edge_front_n','edge_back_n','alpha_edge_n','interior_n')}))
    old_pair={(r['scene'],r['prior']):r for r in first['paired']}
    pair=[]
    for v in last['paired']:
        old=old_pair[v['scene'],v['prior']]
        assert v['pixels']==old['pixels']
        pair.append(dict(**v,**{k+'_r1':old[k] for k in
            ('delta_inverse_mae','delta_CoC_px','delta_boundary_front_CoC_px')}))
    ds=[r for r in pair if r['prior']=='DepthSplat']
    return entries,dict(comparison=write_table('V4-vs-round1',joined),
                        absolute=write_table('V4-development',last['pooled']),
                        paired=write_table('V4-same-masks',pair),
                        weights=write_table('V4-prior-selection',last['prior_selection']),
                        check_history=write_table('V4-validation-intervals',[dict(step=e['step'],**r) for e in entries for r in e['paired']]),
                        per_frame=write_table('V4-per-frame',last['per_frame']),
                        methods=joined,paired_rows=pair,selection=last['prior_selection'],
                        frames=last['frames'],same_mask_counts_verified=True,
                        goal_boundary_not_worse=dict(rule='equal or lower paired boundary-front CoC MAE',rows=ds,
                                                     by_scene={r['scene']:r['delta_boundary_front_CoC_px']<=0 for r in ds}))


def main():
    v1=footprint()
    v2=read(OUT/'single-batch.json')
    v3=corpus()
    training=read(OUT/'training.json')
    assert training['steps']==30000 and training['evaluations'][-1]['step']==30000
    validation,dev=final_validation()
    curve=training.pop('curve')
    assert len(curve)==30000 and all(r['frames']==39 for r in validation)
    bins=defaultdict(list)
    for row in curve:
        bins[(row['step']-1)//100,row['tier']].append(row)
    curve_rows=[
        dict(first_step=block*100+1,last_step=(block+1)*100,tier=tier,batches=len(rows),
             **{key:float(np.mean([r[key] for r in rows])) for key in rows[0] if key not in ('step','tier')})
        for (block,tier),rows in sorted(bins.items())]
    shape=training['architecture']
    shape.update(prior_choice='learned per-pixel masked softmax on three inverse-depth priors, quarter-scale + pixel logits',
                 boundary_sampling='equal-size strata where available up to 8192 per label, weighting 2:1 for all three terms',
                 all_missing_depth_m=5.0,neighbor_visibility='retains round1 supporting view evidence; target basis is learned')
    training.update(curve=write_table('V4-training-curve',curve_rows),plot=figures(curve,validation),
                    curve_source=str((OUT/'training.json').relative_to(ROOT)),
                    curve_recording='CSV means by tier and fixed 100-step block, raw per-step data in run output')
    v4=dict(training=training,checkpoint=training['evaluations'][-1]['checkpoint'],development=dev)
    headline=dict(base_git=CONFIG['protocol_revision'],outputs=CONFIG['output'],V1=v1,V2=v2,V3=v3,V4=v4,
                  resources=dict(output_GiB=disk(OUT)/2**30,data_free_GiB=shutil.disk_usage(OUT).free/2**30,
                                 CUDA_memory_fractions=dict(V1=.18,V4=.45,priors=.20),machine='RTX 4090D 48GB'),
                  independent_test_inventory='03_实验记录/records/route3-ovg-1a-2026-11-18-independent-test.json',
                  conclusion_scope='given poses, same three development scenes, geometry only; joint data/network/loss changes cannot be individually attributed')
    save_json(REC/f'{PREFIX}-summary.json',headline)


if __name__=='__main__':
    main()
