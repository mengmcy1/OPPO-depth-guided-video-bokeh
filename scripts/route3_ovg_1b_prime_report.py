"""Extract audited 1b′ tables from the frozen terminal and checkpoint artifacts."""
import csv
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from io import StringIO

import torch

from route3_ovg_1b_common import CONFIG, REC, ROOT, SOURCE, read
from route3_ovg_1b_prime import OUT, PROFILE
from route3_ovg_1b_prime_evaluate import A
from route3_ovg_subpixel import save_json

PREFIX = 'route3-ovg-1b-prime-2026-11-18'
OPTICAL_COLS = ('scene','protocol','nominal_focus_m','fstop','hidden','visible','region')
SUBSETS = {'high_conf_DepthSplat_boundary': 4603, 'high_conf_single_prior': 2566,
           'high_conf_multiple_priors': 2037}


def table(name, rows):
    path = REC/f'{PREFIX}-{name}.csv'
    fields = list(dict.fromkeys(k for row in rows for k in row))
    stream = StringIO(newline='')
    writer = csv.DictWriter(stream, fields, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    contents = stream.getvalue()
    if not path.exists():
        path.write_text(contents, encoding='utf-8')
    elif path.read_text(encoding='utf-8')!=contents:
        raise ValueError(f'Existing table differs: {path}')
    return str(path.relative_to(ROOT))


def summarize():
    train_path=OUT/'training.json'
    optical_path=OUT/'Q3/terminal/results.json'
    train=read(train_path)
    check=read(OUT/'Q1/check.json')
    optical=read(optical_path)
    validation=OUT/'validation'
    assert check['passed'] and train['steps']==CONFIG['steps'] and train['profile']==PROFILE
    curve=train['curve']
    assert len(curve)==30000 and all(math.isfinite(row['loss']) and math.isfinite(row['grad_norm']) for row in curve)
    assert Counter(row['tier'] for row in curve)==Counter(train['counts'])
    assert all('normal' not in row['terms'] for row in curve)
    assert all(set(row['terms'])=={'mv','zero'} for row in curve if row['tier']=='R')
    checkpoint=OUT/'checkpoints/step-030000.pt'
    terminal=torch.load(checkpoint,map_location='cpu',weights_only=True)
    before=torch.load(SOURCE/'checkpoints/step-030000.pt',map_location='cpu',weights_only=True)['model']
    assert terminal['step']==30000 and terminal['profile']==PROFILE
    for key in ('model','ema'):
        current=terminal[key]
        assert all(torch.isfinite(v).all().item() for v in current.values())
        for param in ('weight','bias'):
            torch.testing.assert_close(before[f'footprint.{param}'],current[f'footprint.{param}'],rtol=0,atol=0)
            torch.testing.assert_close(before[f'geometry.{param}'][2:5],current[f'geometry.{param}'][2:5],rtol=0,atol=0)
    assert optical['K']==64 and len(optical['rows'])==72*(8*2*8+7)
    assert len(optical['placements'])==72*8
    assert len(optical['rows'])==len({(r['frame'],*(r[k] for k in OPTICAL_COLS)) for r in optical['rows']})
    assert all(not r.get('excluded_nonfinite_pixels',0) for r in optical['rows'])
    monitors,geom_curves,high_conf=[],[],[]
    prior_fields=('scene','focus_m','subset','method','edge_front_n','boundary_front_CoC_mae_px')
    for step in CONFIG['eval_steps']:
        stat=read(validation/f'step-{step:06d}.json')
        for weight,evidence in stat.items():
            zero=evidence['zero_aperture']
            for row in zero['frames']:
                for arm in ('visible_only','with_GT_hidden_and_F2'):
                    threshold=PROFILE['zero_aperture']['visible_only_min_db' if arm=='visible_only' else 'merged_with_f2_min_db']
                    psnr=row[arm]['psnr_db']
                    monitors.append(dict(step=step,weights=weight,scene=row['scene'],arm=arm,psnr_db=psnr,
                                         floor_db=threshold,below_floor=psnr<threshold))
            for arm,psnr in zero['pooled_psnr_db'].items():
                threshold=PROFILE['zero_aperture']['visible_only_min_db' if arm=='visible_only' else 'merged_with_f2_min_db']
                monitors.append(dict(step=step,weights=weight,scene='pooled',arm=arm,psnr_db=psnr,
                                     floor_db=threshold,below_floor=psnr<threshold))
            for row in evidence['rows']:
                if row['scene']=='pooled' and row['method']=='OVG-Net' and row['focus_m']==1.4:
                    geom_curves.append(dict(step=step,weights=weight,**{k:row[k] for k in prior_fields}))
                if step==30000 and row['subset'] in SUBSETS:
                    if row['scene']=='pooled':
                        assert row['edge_front_n']==SUBSETS[row['subset']]
                    high_conf.append(dict(weights=weight,**{k:row[k] for k in prior_fields}))
    main_rows=[r for r in optical['pooled'] if r['scene']=='pooled' and r['protocol']=='P2' and r['fstop']==1.4
               and r['region'] in ('in_focus','recoverable','zero_full_image','zero_visible_full_image')]
    paired=optical['paired']
    primary=tuple(r for r in paired if r['protocol']=='P2' and r['fstop']==1.4 and r['scene']=='pooled')
    geometry_comparison={}
    for w in ('EMA','raw'):
        rows=[r for r in high_conf if r['weights']==w and r['scene']=='pooled' and r['focus_m']==1.4]
        grouping=defaultdict(dict)
        for r in rows:
            grouping[r['subset']][r['method']]=r['boundary_front_CoC_mae_px']
        geometry_comparison[w]={key:dict(pixels=SUBSETS[key],**results,
                meets_fusion=results['OVG-Net']<=results['selected-prior-base'])
                for key,results in grouping.items()}
    direction=[]
    for focus in (1.4,5.):
        for fstop in (1.4,2.):
            view=[r for r in optical['pooled'] if r['scene']=='pooled' and r['protocol']=='P2'
                  and r['nominal_focus_m']==focus and r['fstop']==fstop and r['region']=='recoverable']
            sharp=next(r for r in view if r['visible']==A)['psnr']
            actual={r['visible']:r['psnr']-sharp for r in view if r['hidden']=='Query'}
            direction.append(dict(focus_m=focus,fstop=fstop,A_psnr=sharp,Query_minus_A_db=actual))
    data={
        'zero_monitor':table('Q2-zero-monitor',monitors),
        'geometry_curve':table('Q2-checkpoint-geometry',geom_curves),
        'optical_pooled':table('Q3-optical',optical['pooled']),
        'paired_differences':table('Q3-paired',optical['paired']),
        'high_conf_boundary':table('Q3-high-confidence',high_conf),
    }
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes=plt.subplots(1,2,figsize=(10,4),sharex=True)
    arms=[('visible_only',50,'Visible only'),('with_GT_hidden_and_F2',60,'With GT hidden layer and F2')]
    for ax,(arm,floor,title) in zip(axes,arms):
        for weight in ('raw','EMA'):
            sequence=[(r['step'],r['psnr_db']) for r in monitors if r['arm']==arm and r['scene']=='pooled' and r['weights']==weight]
            ax.plot(*zip(*sequence),marker='o',label=weight)
        ax.axhline(floor,linestyle='--',color='gray',label='Floor')
        ax.set(title=title,xlabel='Optimizer step',ylabel='Pinhole PSNR (dB)')
        ax.legend()
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(OUT/'zero-aperture-curves.png',dpi=150)
    plt.close(fig)
    training_curves=OUT/'training-curves.png'
    assert training_curves.is_file()
    summary=dict(protocol=PROFILE['protocol_revision'],known_original_run='a180611',Q1=check,
        Q2=dict(steps=30000,seconds=train['seconds'],peak_allocated_MiB=train['peak_MiB'],
                counts=train['counts'],sampled_fractions=train['sampled_fractions'],
                active_fractions=train['active_fractions'],data_pool=train['joins'][-1]['counts'],
                R_availability='all 291 observations from the beginning',
                normal_footprint_opacity_parameters_unchanged=True,
                training_plot=str(training_curves.relative_to(ROOT)),
                zero_monitor_plot=str((OUT/'zero-aperture-curves.png').relative_to(ROOT)),
                terminal_checkpoint=str(checkpoint.relative_to(ROOT)),
                completed_utc=datetime.fromtimestamp(train_path.stat().st_mtime,timezone.utc).isoformat()),
        Q3=dict(K=optical['K'],cases=72,scenario_count=3,rows=len(optical['rows']),
                EMA_primary_pooled=main_rows,primary_pairs=primary,
                high_confidence=geometry_comparison,direction_relative_to_A=direction,
                terminal_seconds=optical['seconds'],terminal_peak_allocated_MiB=optical['peak_allocated_MiB'],
                paired_reference_seconds=read(OUT/'Q3/reference/results.json')['seconds'],
                scope='three synthetic development scenes; not an independent generalization score',
                optical_artifact=str(optical_path.relative_to(ROOT))),tables=data)
    path=REC/f'{PREFIX}-summary.json'
    assert not path.exists()
    save_json(path,summary)
    print('PRIME_REPORT_READY',path,flush=True)


if __name__=='__main__':
    summarize()
