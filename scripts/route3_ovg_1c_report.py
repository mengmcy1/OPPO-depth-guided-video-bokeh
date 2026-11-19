"""Write the 1c selection and measurements without touching the historical runs."""
from collections import Counter, defaultdict
import csv

import numpy as np

from route3_ovg_1b_common import REC, ROOT, read
from route3_ovg_1b_prime import OUT as PRIME
from route3_ovg_1b_prime_evaluate import A, ORIGINAL, PRIME_EMA
from route3_ovg_1c import INITIAL, OUT
from route3_ovg_1c_evaluate import ALIGNED_FIXED, TRUE, NO_EXPLICIT
from route3_ovg_refocus import OUT as REFOCUS
from route3_ovg_refocus_report import KEYS, by, coverage_summary, normalize
from route3_ovg_subpixel import save_json
from route3_upper_bound_evaluate import pool

PREFIX = 'route3-ovg-1c-2026-11-19'
CONTROL = '1c-control-EMA'
ALIGNED = '1c-aligned-EMA'
TWO_FLOORS = {'zero_visible_full_image': 50., 'zero_full_image': 60.}


def table(label, rows):
    path = REC/f'{PREFIX}-{label}.csv'
    rows = list(rows)
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as out:
        writer = csv.DictWriter(out,columns,lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    return str(path.relative_to(ROOT))


def matched_delta(rows, base, method, base_hidden='Query', hidden='Query'):
    columns = ('frame','id',*KEYS[:4],KEYS[-1])
    reference = {by(r,columns):r for r in rows if r['method']==base and r['hidden']==base_hidden}
    treatment = {by(r,columns):r for r in rows if r['method']==method and r['hidden']==hidden}
    common = sorted(reference.keys() & treatment.keys())
    if not all(reference[k]['pixels']==treatment[k]['pixels'] for k in common):
        raise ValueError(f'Comparison changed scoring support for {method} against {base}')
    before_pool = normalize(reference[k] for k in common)
    result = []
    for key,row in normalize(treatment[k] for k in common).items():
        if row['pixels'] == 0:
            continue
        before = before_pool[(*key[:4],base,base_hidden,key[-1])]
        result.append(dict({k:row[k] for k in KEYS},reference=base, reference_hidden=base_hidden,
                           pixels=row['pixels'],cases=row['cases'], method_psnr=row['psnr'],
                           reference_psnr=before['psnr'],PSNR_delta=row['psnr']-before['psnr'],
                           method_patch_LPIPS=row['patch_lpips'],reference_patch_LPIPS=before['patch_lpips'],
                           LPIPS_delta=row['patch_lpips']-before['patch_lpips']))
    return result


def is_primary(row):
    return row['protocol']=='P2' and row['aperture']==1.4 and row['hidden']=='Query'


def select(rows):
    paired = matched_delta(rows,CONTROL,ALIGNED)
    near = [r for r in paired if is_primary(r) and r['名义对焦距离_m']==1.4 and r['region']=='in_focus']
    far = [r for r in paired if is_primary(r) and r['名义对焦距离_m']==5. and r['region']=='recoverable']
    assert len(near)==len(far)==4 and all(r['cases']==(9 if r['scene']=='pooled' else 3) for r in near+far)
    dn = next(r['PSNR_delta'] for r in near if r['scene']=='pooled')
    df = next(r['PSNR_delta'] for r in far if r['scene']=='pooled')
    improved = sum(r['PSNR_delta']>0 for r in near if r['scene']!='pooled')
    floors = {}
    for region,minimum in TWO_FLOORS.items():
        values = [r['psnr'] for r in rows if r['method']==ALIGNED and r['region']==region and is_primary(r)]
        assert len(values)==18
        floors[region] = dict(floor_db=minimum,min_db=min(values),passed=sum(x>=minimum for x in values),cases=len(values))
    checks = dict(near_focus=dn>.1 and improved>=2, far_recoverable=df>=-.1,
                  zero_aperture=all(x['passed']==x['cases'] for x in floors.values()))
    method = ALIGNED if all(checks.values()) else PRIME_EMA
    checkpoint = OUT/'train/aligned/checkpoints/step-005000.pt' if method==ALIGNED else INITIAL
    return dict(criteria=checks,near_pool_delta_db=dn,near_improved_scenes=improved,
                far_pool_delta_db=df,per_scene_near=near,per_scene_far=far,
                primary_zero_floors=floors,stage2_method=method,
                stage2_checkpoint=str(checkpoint.relative_to(ROOT)),weights_key='ema',
                conditional_input_alignment=(method==ALIGNED),stage1_closed=True)


def summarize_fit(rows):
    groups = defaultdict(list)
    for row in rows:
        for name in (row['scene'],row['split'],'all-B'):
            groups[name].append(row)
    summary = []
    for group,items in groups.items():
        reasons = dict(Counter(r['fallback'] or 'accepted' for r in items))
        eligible = [r for r in items if 'holdout_before' in r]
        denominator = sum(r['holdout_pixels'] for r in eligible)
        rec = dict(group=group,frames=len(items),accepted=reasons.get('accepted',0),classification=reasons)
        for label in ('before','after_applied'):
            moment = 'holdout_'+label
            if denominator:
                rec['heldout_inverse_RMSE_'+label] = (
                    sum(r['holdout_pixels']*r[moment]['inverse_RMSE']**2 for r in eligible)/denominator)**.5
                rec['heldout_depth_AbsRel_'+label] = (
                    sum(r['holdout_pixels']*r[moment]['sparse_depth_AbsRel'] for r in eligible)/denominator)
        rec['heldout_report_frames'] = len(eligible)
        rec['heldout_report_pixels'] = denominator
        summary.append(rec)
    return summary


def zero_rows(rows):
    groups = defaultdict(list)
    for r in rows:
        if r['region'] in TWO_FLOORS and r['method']!=A:
            groups[by(r,('scene','protocol','aperture','名义对焦距离_m','method','hidden','region'))].append(r)
    for k,records in sorted(groups.items()):
        minimum = TWO_FLOORS[k[-1]]
        # without an explicit layer scoped as a control, not as a "with-hidden" floor
        if k[-2]=='none' and k[-1]=='zero_full_image':
            continue
        yield dict(zip(KEYS,k),floor_db=minimum,cases=len(records),
                   cases_below=sum(r['psnr']<minimum for r in records),min_db=min(r['psnr'] for r in records),
                   pooled_psnr=pool(records)['psnr'])


def training_curves():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    overview, monitor, geometry = [], [], []
    fig, axes = plt.subplots(2, 2, figsize=(11,7), constrained_layout=True)
    for arm,color in (('control','#1774aa'),('aligned','#c45e21')):
        folder = OUT/'train'/arm
        report = read(folder/'training.json')
        blocks = [np.mean([r['loss'] for r in report['curve'][i:i+100]]) for i in range(0,5000,100)]
        axes[0,0].plot(np.arange(100,5001,100),blocks,color=color,label=arm)
        overview.append(dict(arm=arm,steps=report['steps'],counts=report['counts'],
                       sampled_fractions=report['sampled_fractions'],active=report['active'],
                       identical_sample_sequence=report['batch_sequence_sha256'],
                       started_from=report['initialization'],seconds=report['seconds'],peak_MiB=report['peak_MiB'],
                       first_100_mean_loss=float(blocks[0]),last_100_mean_loss=float(blocks[-1]),
                       checkpoint=str((folder/'checkpoints/step-005000.pt').relative_to(ROOT))))
        for suffix,style in (('EMA','-'),('raw','--')):
            end = []
            for step in range(1000,5001,1000):
                r = read(folder/'validation'/f'step-{step:06d}.json')[suffix]
                z = r['zero_aperture']['pooled_psnr_db']
                frames = r['zero_aperture']['frames']
                geo = next(x for x in r['rows'] if x['scene']=='pooled' and x['method']=='OVG-Net'
                           and x['focus_m']==1.4 and x['subset']=='any_prior')
                row = dict(arm=arm,weights=suffix,step=step,visible_only=z['visible_only'],
                           with_GT_hidden_and_F2=z['with_GT_hidden_and_F2'],
                           visible_min_db=min(x['visible_only']['psnr_db'] for x in frames),
                           merged_min_db=min(x['with_GT_hidden_and_F2']['psnr_db'] for x in frames),
                           visible_below=sum(x['visible_below_floor'] for x in frames),
                           merged_below=sum(x['merged_below_floor'] for x in frames),
                           front_CoC_mae_px=geo['front_CoC_mae_px'],
                           boundary_front_CoC_mae_px=geo['boundary_front_CoC_mae_px'])
                end.append(row)
                if step==5000:
                    geometry.extend(dict(x,checkpoint=f'1c-{arm}-{suffix}') for x in r['rows'])
            monitor.extend(end)
            for axis,key in ((axes[0,1],'boundary_front_CoC_mae_px'),
                             (axes[1,0],'visible_only'),(axes[1,1],'with_GT_hidden_and_F2')):
                axis.plot([x['step'] for x in end],[x[key] for x in end],style,color=color,
                          marker='o',label=f'{arm} {suffix}')
    axes[1,0].axhline(50,color='grey',linewidth=.8)
    axes[1,1].axhline(60,color='grey',linewidth=.8)
    for axis,title,ylabel in zip(axes.flat,['100-step average training loss','Dev boundary front CoC MAE',
                'Zero aperture: visible only','Zero aperture: GT hidden + F2'],['loss','px','dB','dB']):
        axis.set(title=title,ylabel=ylabel,xlabel='continuation step')
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    fig.savefig(OUT/'C2-paired-curves.png',dpi=150)
    plt.close(fig)
    baseline = read(PRIME/'validation/step-030000.json')
    for suffix in ('EMA','raw'):
        geometry.extend(dict(x,checkpoint=f'1b-prime-{suffix}') for x in baseline[suffix]['rows'])
    return overview, monitor, geometry


def main():
    alignment = read(OUT/'C1.json')
    optics = read(OUT/'C3-optical.json')
    cache = read(OUT/'C3-query.json')
    summaries, monitor, geometry = training_curves()
    labels = read(OUT/'phase2-B/labels-summary.json')
    coverage = read(OUT/'phase2-B/coverage-summary.json')
    labels['coverage_valid_pixels'] = sum(r['valid'] for r in coverage['rows'])
    labels['coverage_max_abs_error'] = max(r['max_abs_error'] for r in coverage['rows'])
    labels['audit_boundary_pixels'] = sum(r['edge_pixels'] for r in labels['audits'])
    labels['manifest'] = str((OUT/'phase2-B/render-jobs.json').relative_to(ROOT))
    ref = read(REFOCUS/'R3.json')
    old = [r for r in ref['rows'] if r['method'] in (PRIME_EMA,ORIGINAL,A)]
    rows = optics['rows'] + old
    by_case = ('frame','id',*KEYS)
    assert len(rows)==len({by(r,by_case) for r in rows})
    assert not any(r.get('excluded_nonfinite_pixels',0) for r in rows)
    aggregate = normalize(rows)
    selection = select(rows)
    pair_rows = matched_delta(rows,CONTROL,ALIGNED)
    pair_rows += matched_delta(rows,'1c-control-raw','1c-aligned-raw')
    pair_rows += matched_delta(rows,ORIGINAL,ALIGNED_FIXED)
    pair_rows += matched_delta(rows,PRIME_EMA,TRUE)
    gains = []
    for method in NO_EXPLICIT:
        gains.extend(matched_delta(rows,method,method,'none','Query'))
    gaps = []
    for method in (CONTROL,ALIGNED,PRIME_EMA,ORIGINAL,ALIGNED_FIXED,TRUE):
        gaps.extend(matched_delta(rows,A,method,'standalone'))
    agreements = read(OUT/'C3-agreement.json')['rows']
    pooled_agreement = defaultdict(Counter)
    for r in agreements:
        for scope in (r['scene'],'pooled'):
            pooled_agreement[scope,r['arm']].update({k:v for k,v in r.items() if k.startswith(('edge_','all_'))})
    for arm in ('before','after'):
        assert pooled_agreement['pooled',arm]['edge_single']==2566
        assert pooled_agreement['pooled',arm]['edge_multiple']==2037
    geom_sums = defaultdict(Counter)
    geom_cols = ('scene','method','nominal_focus_distance_m','region')
    for r in optics['visible_geometry']:
        for scope in (r['scene'],'pooled'):
            geom_sums[(scope,*by(r,geom_cols[1:]))].update({k:r[k] for k in (
                'pixels','inverse_absolute_error_sum','CoC_absolute_error_sum_px')})
    rendered_geometry = [dict(zip(geom_cols,k),**v,
                              inverse_depth_MAE=v['inverse_absolute_error_sum']/v['pixels'],
                              CoC_MAE_px=v['CoC_absolute_error_sum_px']/v['pixels'])
                         for k,v in sorted(geom_sums.items()) if v['pixels']]
    files = dict(
        C1_alignment=table('C1-alignment',[
            dict({k:v for k,v in r.items() if not isinstance(v,dict)},
                 **{f'{phase}_{k}':v for phase in ('holdout_before','holdout_after_proposed','holdout_after_applied')
                    for k,v in r.get(phase,{}).items()}) for r in alignment['rows']]),
        C2_zero_and_geometry=table('C2-monitor',monitor),
        C3_focus=table('C3-focus',read(OUT/'C3-focus.json')['rows']),
        C3_query_cases=table('C3-query',cache['rows']),
        C3_query_pooled=table('C3-query-pooled',coverage_summary(cache['rows'])),
        C3_optical=table('C3-optical',aggregate.values()),
        C3_pairs=table('C3-pairs',pair_rows),
        C3_hidden_grid=table('C3-hidden-four-cells',gains),
        C3_near_gaps=table('C3-vs-A',gaps),
        C3_zero=table('C3-zero',zero_rows(rows)),
        C3_geometry=table('C3-geometry',[
            {k:r[k] for k in ('checkpoint','scene','subset','method','focus_m','front_n','edge_front_n',
                    'front_inverse_mae','front_CoC_mae_px','boundary_front_inverse_mae','boundary_front_CoC_mae_px')}
            for r in geometry if r['subset'] in ('any_prior','high_conf_single_prior','high_conf_multiple_priors')]),
        C3_rendered_geometry=table('C3-rendered-geometry',rendered_geometry),
        C3_agreement=table('C3-agreement',agreements),
        phase2_B_audit=table('phase2-B-audit',labels['audits']))
    save_json(REC/f'{PREFIX}-summary.json',dict(
        starting_revision='5bfc678',starting_pipeline='1b′ aeda3af + formal focus 27b1e88',
        output=str(OUT.relative_to(ROOT)),scope='3 synthetic development scenes, 9 representative frames; no independent test',
        C1=dict(fixed_check=alignment['fixed_check'],stays_DepthSplat_only=alignment['untouched_tiers'],
                summary=summarize_fit(alignment['rows']),seed=alignment['seed'],seconds=alignment['seconds'],
                relative_error=alignment['relative_error']),
        C2=dict(arms=summaries,monitor=files['C2_zero_and_geometry'],curves=str((OUT/'C2-paired-curves.png').relative_to(ROOT)),
                latest_selection_only=True,
                environment=dict(device='RTX 4090 D',torch='2.7.1+cu128',opencv='4.11.0',numpy='1.26.4',
                                 matmul_precision_both_arms='highest, default',
                                 historical_1b_prime_train_matmul_precision='high'),
                initialization='same 1b′ raw, EMA and AdamW for both arms'),
        C3=dict(agreement=[dict(scene=s,arm=a,**d) for (s,a),d in sorted(pooled_agreement.items())],
                render_seconds=optics['seconds'],query_cache_seconds=cache['seconds'],
                peak_MiB=optics['peak_allocated_MiB'],
                P1_failure_count=sum(r['失败'] for r in read(OUT/'C3-focus.json')['rows'] if r['protocol']=='P1'),
                GT_reference_exact_hidden_anchor=PRIME_EMA,
                identity_metrics_reused='A, 1b′ EMA and fixed original from 27b1e88',
                primary=[r for r in aggregate.values() if r['protocol']=='P2' and r['aperture']==1.4 and
                          (r['名义对焦距离_m'],r['region']) in ((1.4,'in_focus'),(5.,'recoverable'))],
                hidden_grid_primary=[r for r in gains if is_primary(r) and
                          (r['名义对焦距离_m'],r['region']) in ((1.4,'in_focus'),(5.,'recoverable'))]),
        C4=selection,stage2_B=labels, files=files))
    print(selection['criteria'],selection['stage2_method'],selection['near_pool_delta_db'],selection['far_pool_delta_db'])


if __name__=='__main__':
    main()
