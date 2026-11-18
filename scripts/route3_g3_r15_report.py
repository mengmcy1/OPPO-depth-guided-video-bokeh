"""Pool revision 15 scores by SSE and emit the experiment records."""
import csv
from collections import Counter, defaultdict
import platform
import shutil
import subprocess

import numpy as np
import torch

from route3_g3_r15 import C, DEFAULT, ORIGINAL, OUT, PREFIX, PREV, REC, read, write
from route3_upper_bound_evaluate import pool

METHODS = ('main', 'balanced_activation', 'original_geometry')
STATS = ('queries', 'candidate_correct_available_queries', 'matched_heads', 'selected_correct',
         'false_activated_queries', 'unneeded_queries', 'activated_needed_queries', 'needed_queries')


def csv_record(label, rows):
    if rows:
        path = REC / (PREFIX + '-' + label + '.csv')
        keys = list(dict.fromkeys(k for row in rows for k in row))
        with path.open('w') as stream:
            writer = csv.DictWriter(stream, keys, lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
        return str(path.relative_to(REC.parent.parent))
    return None


def group_scores(rows, fields):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in fields)].append(row)
    return [dict(zip(fields, k), **pool(v), configurations=len(v)) for k, v in sorted(groups.items())]


def order_summary():
    value = read(REC / (PREFIX + '-R2.json'))
    counts = defaultdict(Counter)
    for row in value['audits']:
        group = counts[row['scene'], row['focus'], row['rule']]
        for field in ('total', 'violating', 'moved', 'dropped'):
            group[field] += row[field]
    count_rows = [dict(scene=s, focus=f, rule=r, **c, violation_fraction=c['violating']/max(1,c['total']))
                  for (s,f,r),c in sorted(counts.items())]
    pooled = group_scores(value['rows'], ('scene', 'focus', 'region', 'method'))
    return dict(decision=value['rule'], in_focus_delta_dB=value['in_focus_delta_dB'], far_PSNR=value['far_PSNR'],
                counts=csv_record('R2-counts', count_rows), metrics=csv_record('R2-pooled', pooled),
                placement_audit=read(OUT/'R2/placements512-ready.json'))


def render_summary(split):
    scores, zero, queries, candidate = [], [], [], []
    reports = sorted((OUT/'R3/renders').glob('*/*/*/*/scores.json'))
    for path in reports:
        value = read(path)
        geometry, scene, protocol, cid = path.parts[-5:-1]
        shared = dict(geometry=geometry, scene=scene, protocol=protocol, id=cid)
        scores.extend(dict(r,geometry=geometry) for r in value['rows'])
        zero.extend(dict(r,geometry=geometry) for r in value['zero'])
        info = dict(shared, focus=value['rows'][0]['focus'], fstop=value['rows'][0]['fstop'])
        for mode, q in value['query'].items():
            if q['metrics'] is not None:
                queries.append(dict(info, mode=mode, **q['metrics']))
        candidate.append(dict(info, **value['candidate_oracle']))
    pooled = group_scores(scores, ('geometry','scene','protocol','focus','region','method'))
    pooled += group_scores([dict(r,scene='pooled') for r in scores],
                           ('geometry','scene','protocol','focus','region','method'))
    allfocus = [dict(r,focus='both') for r in scores]
    pooled += group_scores(allfocus, ('geometry','scene','protocol','focus','region','method'))
    pooled += group_scores([dict(r,scene='pooled') for r in allfocus],
                           ('geometry','scene','protocol','focus','region','method'))
    comparators = {mode:[r for r in pooled if r['method']==mode] for mode in METHODS}
    index = {(r['geometry'],r['scene'],r['protocol'],r['focus'],r['region'],r['method']):r for r in pooled}
    comparison = []
    for mode, selected in comparators.items():
        for r in selected:
            key = (r['geometry'],r['scene'],r['protocol'],r['focus'],r['region'])
            def get(method):
                return index.get((*key,method),{})
            def psnr(method):
                return get(method).get('psnr')
            def difference(first, second):
                return first-second if first is not None and second is not None else None
            d0, d1, cand = [psnr(k) for k in ('D0','D1_Oracle','Candidate_Oracle')]
            change = difference(r['psnr'],d0)
            oracle_gain = difference(d1,d0)
            cand_gain = difference(cand,d0)
            learning = (change/cand_gain if change is not None and cand_gain is not None
                        and ((r['scene']=='pooled' and abs(cand_gain)>1e-6) or cand_gain>=.3) else None)
            coverage = cand_gain/oracle_gain if cand_gain is not None and oracle_gain is not None and abs(oracle_gain)>1e-6 else None
            comparison.append(dict(zip(('geometry','scene','protocol','focus','region'),key),
                                   mode=mode,learned_PSNR=r['psnr'],D0_PSNR=d0,D1_Oracle_PSNR=d1,Candidate_Oracle_PSNR=cand,
                                   A_PSNR=psnr('A'),R1_R2_PSNR=psnr('R1_R2'),
                                   D_R1_R2_same_candidates_PSNR=psnr('D_R1_R2_same_candidates'),
                                   learned_minus_D0_dB=change,D1_Oracle_minus_D0_dB=oracle_gain,candidate_minus_D0_dB=cand_gain,
                                   learned_minus_A_dB=difference(r['psnr'],psnr('A')),
                                   learned_minus_R1_R2_dB=difference(r['psnr'],psnr('R1_R2')),
                                   learned_minus_matched_rule_dB=difference(r['psnr'],psnr('D_R1_R2_same_candidates')),
                                   learning_efficiency=learning,candidate_coverage=coverage,
                                   scene_efficiency_eligible=cand_gain is not None and cand_gain>=.3,
                                   D1_positive_headroom=oracle_gain is not None and oracle_gain>0,
                                   learned_LPIPS=r['patch_lpips'],A_LPIPS=get('A').get('patch_lpips'),
                                   D0_LPIPS=get('D0').get('patch_lpips'),D1_Oracle_LPIPS=get('D1_Oracle').get('patch_lpips'),
                                   Candidate_Oracle_LPIPS=get('Candidate_Oracle').get('patch_lpips'),R1_R2_LPIPS=get('R1_R2').get('patch_lpips')))
    bucket = defaultdict(Counter)
    for r in queries:
        for scene in (r['scene'],'pooled'):
            for focus in (r['focus'],'both'):
                key = r['geometry'],scene,r['protocol'],focus,r['mode']
                bucket[key].update({k:r[k] for k in STATS})
    qrows = []
    for key,t in bucket.items():
        qrows.append(dict(zip(('geometry','scene','protocol','focus','mode'),key),
                          **t,correct_candidate_fraction=ratio(t['candidate_correct_available_queries'],t['queries']),
                          selection_accuracy=ratio(t['selected_correct'],t['matched_heads']),
                          false_activation=ratio(t['false_activated_queries'],t['unneeded_queries']),
                          missed_activation=ratio(t['needed_queries']-t['activated_needed_queries'],t['needed_queries'])))
    fkey = lambda r: (r['geometry'],r['scene'],r['protocol'],r['id'],r['region'])
    zx = {(fkey(r),r['method']):r['PSNR'] for r in zero}
    ix = {(fkey(r),r['method']):r['psnr'] for r in scores}
    guards = []
    for rows, like, key, criterion in ((zero,zx,'PSNR','zero'),(scores,ix,'psnr','defocus')):
        for r in rows:
            if r['method'] not in METHODS or r['region'] not in ('full_image','in_focus','invalid_geometry'):
                continue
            def delta(other):
                a, b = r[key], like.get((fkey(r),other))
                return a-b if a is not None and b is not None else None
            guards.append(dict(geometry=r['geometry'],scene=r['scene'],protocol=r['protocol'],focus=r['focus'],
                               fstop=r['fstop'],frame=r['frame'],id=r['id'],mode=r['method'],region=r['region'],
                               kind=criterion,PSNR=r[key],relative_A=delta('A'),relative_D0=delta('D0')))
    return dict(configurations=len(reports), expected_P2_configurations_per_geometry=len(split['heldout'])*12,
                completed_configurations=[dict(geometry=g,protocol=p,configurations=n) for (g,p),n in
                                          sorted(Counter((r['geometry'],r['protocol']) for r in candidate).items())],
                per_config=csv_record('R3-per-config',scores), pooled=csv_record('R3-pooled',pooled),
                comparison=csv_record('R3-learning-system',comparison), query=csv_record('R3-query',qrows),
                query_per_config=csv_record('R3-query-per-config',queries),
                oracle_query=csv_record('R3-oracle-query',candidate),
                zero=csv_record('R3-zero',zero),guards=csv_record('R3-guards',guards),
                direction_rows=[r for r in comparison if r['mode']=='main' and r['protocol']=='P2'
                                and r['region']=='recoverable'],
                query_primary=[r for r in qrows if r['mode']=='main' and r['protocol']=='P2'],
                denominator='Signed differences of pooled PSNR. Per-scene efficiency needs candidate gain >=0.3 dB. Raw pooled ratio and candidate coverage retain sign; negative headroom is not a recoverable fraction.',
                actual_oracle='One supported positive candidate per query, highest K16 hit fraction and sorted-order tiebreak. Unavailable GT placement or color is tracked separately. Candidate comes from the same eight observations used by inference.',
                same_candidate_ablation='R1+R2 on D uses the same eight candidates, O1 ordering and F2 background as learned. System R1+R2 uses A and the full geometry-specific rule pool.',
                query_definitions='All method-band queries include empty queries. Correctness = K16 world-point match; selection = top weighted candidate on GT-supported heads; active = any predicted alpha > 0.1; hidden need = at least one mesh-labeled hidden ray at the query pixel.',
                P1_fits=[read(p) for p in sorted((OUT/'evaluation-input').glob('*/*/ready.json'))])


def ratio(x,y):
    return x/y if y else None


def coverage_summary():
    raw = OUT/'R4/summary.json'
    if not raw.exists():
        data = dict(rows=[read(p) for p in sorted((OUT/'R4').glob('*/coverage.json'))],
                    incomplete=True)
    else:
        data = read(raw)
    data['pilot_source_geometry'] = ('The three representative input views reuse T4 dominant/DepthSplat depths. '
                                     'The original comparison uses calibrated SHARP depth at other input frames. '
                                     'The DepthSplat comparison runs the official depth checkpoint at those frames, '
                                     'with the same input-only consistency and registered-static validity rules.')
    cases, groups = [], []
    for row in data['rows']:
        for geom, result in row['geometries'].items():
            for r in result['rows']:
                cases.append(dict(scene=row['scene'], geometry=geom,
                                  **{k:v for k,v in r.items() if k not in ('missing_causes','correct_observations_by_source')},
                                  **r['missing_causes'], **{'source_'+k:v for k,v in r['correct_observations_by_source'].items()}))
            for focus in ('both',1.4,5.0):
                selection = [r for r in result['rows'] if focus=='both' or r['focus']==focus]
                counts = {k:sum(r[k] for r in selection) for k in ('band_queries','queries_with_correct','recoverable',
                                                                 'recoverable_in_band','recoverable_correct')}
                cause, source = Counter(), Counter()
                for r in selection:
                    cause.update(r['missing_causes'])
                    source.update(r['correct_observations_by_source'])
                groups.append(dict(scene=row['scene'],geometry=geom,focus=focus,**counts,
                                   availability=ratio(counts['queries_with_correct'],counts['band_queries']),
                                   recoverable_availability=ratio(counts['recoverable_correct'],counts['recoverable']),
                                   ring_miss=counts['recoverable']-counts['recoverable_in_band'],
                                   **dict(cause), **{'source_'+k:v for k,v in source.items()}))
    return dict(per_config=csv_record('R4-per-config',cases),pooled=csv_record('R4-pooled',groups),
                rows=groups,complete=raw.exists(),notes={k:v for k,v in data.items() if k!='rows'})


def training_summary():
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    result = {}
    curves, probes = [], []
    for mode in METHODS:
        folder = OUT/'R3'/mode
        path = folder/'training.json'
        if not path.exists():
            continue
        d = read(path)
        log = d['train_probe']
        stability = [float(np.mean(r['losses'])) for r in log]
        ax[0].plot([r['step'] for r in log],stability,marker='.',label=mode)
        x = [r['step'] for r in d['curve']]
        y = [r['total'] for r in d['curve']]
        if len(y)>=9:
            ax[1].plot(x[4:-4],np.convolve(y,np.ones(9)/9,mode='valid'),label=mode)
        result[mode] = dict(protocol=read(folder/'protocol.json'),initial_fixed_probe=stability[0],
                            final_fixed_probe=stability[-1],**{k:d[k] for k in
                            ['steps','trainable_parameters','seconds','seed','peak_allocated_MiB','GT_warmup_steps']},
                            fixed_probe=log,heldout_monitor=d['validation'])
        curves.extend(dict(mode=mode,**r) for r in d['curve'])
        probes.extend(dict(mode=mode,step=r['step'],mean=v) for r,v in zip(log,stability))
    if result:
        for axis,title in zip(ax,('Fixed training-sample probe','Training minibatches (9-sample mean)')):
            axis.set(xlabel='Optimization step',ylabel='Direct-supervision loss',title=title)
            axis.legend(fontsize=8)
        fig.tight_layout()
        path = OUT/'R3/training-curves.png'
        fig.savefig(path,dpi=150)
        csv_record('R3-training-curve',curves)
        csv_record('R3-training-probe',probes)
    else:
        path = None
    plt.close(fig)
    return dict(runs=result,plot=str(path) if path else None,
                original_geometry_resumed_at_step=2000,
                timing_note='Original geometry run reports elapsed time after resume from step 2000. Parallel ablations and evaluation had independent CUDA memory budgets, so these times are not hardware benchmarks.',
                engineering_check=dict(prior_revision='663fd35',
                    multilayer_selection_accuracy=.975,
                    broad_selection_accuracy=.803,broad_false_activation_rate=.804))


def main():
    qa = read(OUT/'R1-quality.json')
    previous = {p.name:read(p/'quality.json') for p in (PREV/'truth').glob('metric-scene-*') if (p/'quality.json').exists()}
    scene_rows = []
    for role, names in qa['split'].items():
        for name in names:
            d = qa['quality'] if name==qa['scene'] else previous[name]
            surf = d.get('surface_checks',{})
            alias = [r['excess_RMSE'] for r in surf.get('alias',[]) if r.get('excess_RMSE') is not None]
            period = [r['period_p01_px'] for r in surf.get('periods',[]) if r.get('period_p01_px') is not None]
            sampling = {kind:max([r['patch_lpips'] for trial in d.get('sampling',[]) for r in trial['checks']
                                 if r['kind']==kind and trial['spp']==d['final_spp']],default=None)
                        for kind in ('convergence','independent')}
            scene_rows.append(dict(scene=name,split=role,seed=d['seed'],spp=d['final_spp'],passed=d['passed'],
                                   geometry_original=str(ORIGINAL/name),geometry_depthsplat=str(DEFAULT/name),
                                   alias_pass=surf.get('alias_pass'),phase_excess_RMSE_max=max(alias,default=None),
                                   period_p01_min_px=min(period,default=None),period_pass=surf.get('period_pass'),
                                   zero_repeat_PSNR=surf.get('zero_aperture_repeat_PSNR'),zero_pass=surf.get('zero_pass'),
                                   final_spp_convergence_LPIPS_max=sampling['convergence'],
                                   final_spp_independent_LPIPS_max=sampling['independent'],
                                   seconds_truth=d['seconds'],own_truth_GiB=d['own_disk_GiB']))
    csv_record('R1-scene-split',scene_rows)
    own_disk = int(subprocess.check_output(['du','-sB1',str(OUT)],text=True).split()[0])/2**30
    write(REC/(PREFIX+'-summary.json'),dict(base_commit='e87a75a',outputs=str(OUT),
          disk_available_GiB=shutil.disk_usage(OUT).free/2**30,new_outputs_GiB=own_disk,
          environment=dict(python=platform.python_version(),torch=str(torch.__version__),torch_cuda=torch.version.cuda,
                           gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version',
                                                        '--format=csv,noheader'],text=True).strip(),
                           blender='4.5.3',pixel_filter='BOX',filter_width_px=1.0),
          R1=dict(quality=qa,scenes=scene_rows,excluded=['122103-thin','122106-thin','122109-thin','122112-thin',
                   '122115-thin','122113-broad'],cache_ready=(OUT/'R1-ready.json').exists()),
          R2=order_summary(),R3=dict(training=training_summary(),evaluation=render_summary(qa['split'])),
          R4=coverage_summary()))


if __name__ == '__main__':
    main()
