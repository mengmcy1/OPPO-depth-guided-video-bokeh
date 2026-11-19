"""Audit and summarize the protocol-corrected evaluation without re-scoring old files."""
from collections import Counter, defaultdict
import csv
from io import StringIO
import math

from route3_ovg_refocus import OUT, PREFIX, METHODS, FIRST, case_frames, read
from route3_ovg_1b_common import OUT as STAGE1B, REC, ROOT
from route3_ovg_1b_prime import OUT as PRIME
from route3_ovg_1b_prime_evaluate import A, DEPTHSPLAT, OLD, OLD_FIXED, ORIGINAL, PRIME_EMA, PRIME_RAW
from route3_ovg_subpixel import save_json
from route3_upper_bound_evaluate import pool

KEYS=('scene','protocol','aperture','名义对焦距离_m','method','hidden','region')
ZERO_FLOORS={'zero_visible_full_image':50.,'zero_full_image':60.}
NEW_PAIRS=[
    ('primitive',OLD_FIXED,OLD),
    ('gate-EMA',PRIME_EMA,PRIME_EMA+'-gate-off'),
    ('gate-raw',PRIME_RAW,PRIME_RAW+'-gate-off'),
    ('relative-to-1a-EMA',PRIME_EMA,FIRST),
    ('relative-to-1a-raw',PRIME_RAW,FIRST),
    ('system',PRIME_EMA,ORIGINAL),
    ('system',PRIME_EMA,DEPTHSPLAT),
    ('system',PRIME_EMA,A),
    ('system',OLD,ORIGINAL),
    ('system',OLD,DEPTHSPLAT),
    ('system',OLD,A),
]


def by(r,k):
    return tuple(r[name] for name in k)


def table(name,rows):
    p=REC/f'{PREFIX}-{name}.csv'
    f=StringIO(newline='')
    keys=list(dict.fromkeys(k for r in rows for k in r))
    writer=csv.DictWriter(f,keys,lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    content=f.getvalue()
    if not p.exists():
        p.write_text(content,encoding='utf-8')
    elif p.read_text(encoding='utf-8')!=content:
        raise ValueError(f'Existing table differs: {p}')
    return str(p.relative_to(ROOT))


def old_rows():
    prime=read(PRIME/'Q3/terminal/results.json')['rows']
    first=read(STAGE1B/'P3/terminal/results.json')['rows']
    for row in [r for r in prime if r['visible'] in METHODS]+[r for r in first if r['visible']==FIRST]:
        keep={k:row[k] for k in ('scene','frame','id','protocol','hidden','region','psnr','sse','pixels','patch_lpips')}
        yield dict(keep,method=row['visible'],aperture=row['fstop'],名义对焦距离_m=row['nominal_focus_m'],
                   实际对焦距离_m=row['focus_m'])


def normalize(rows):
    groups=defaultdict(list)
    for row in rows:
        dims=by(row,KEYS[1:])
        groups[(row['scene'],*dims)].append(row)
        groups[('pooled',*dims)].append(row)
    return {key:dict(zip(KEYS,key),**pool(vals),cases=len(vals)) for key,vals in sorted(groups.items())}


def compare_reference(old,new):
    # Match every valid arm/configuration first, then pool the same cases.
    by_case=('frame','id',*KEYS)
    recorded={by(r,by_case):r for r in old}
    shared=[r for r in new if by(r,by_case) in recorded]
    original=[recorded[by(r,by_case)] for r in shared]
    assert all(before['pixels']==after['pixels'] for before,after in zip(original,shared))
    baseline=normalize(original)
    corrected=normalize(shared)
    series=[]
    for key,c in corrected.items():
        a=baseline[key]
        focal=c['名义对焦距离_m']
        if (focal,c['region']) not in ((1.4,'in_focus'),(5.,'recoverable')) or not c['pixels']:
            continue
        series.append(dict({k:c[k] for k in KEYS},pixels=c['pixels'],cases=c['cases'],
                           old_psnr=a['psnr'],new_psnr=c['psnr'],PSNR_delta=c['psnr']-a['psnr'],
                           old_patch_lpips=a['patch_lpips'],new_patch_lpips=c['patch_lpips'],
                           LPIPS_delta=c['patch_lpips']-a['patch_lpips']))
    return series


def deltas(pooled):
    series=[]
    for label,treatment,reference in NEW_PAIRS:
        for key,row in pooled.items():
            if row['method']!=treatment or not row['pixels']:
                continue
            hidden='standalone' if reference==A else row['hidden']
            base=pooled.get((*key[:4],reference,hidden,key[-1]))
            if base is None:
                continue
            if base['pixels']!=row['pixels'] or base['cases']!=row['cases']:
                raise ValueError('Paired comparison has a different scoring support')
            series.append(dict({k:row[k] for k in KEYS},comparison=label,reference=reference,
                               pixels=row['pixels'],cases=row['cases'],PSNR_delta=row['psnr']-base['psnr'],
                               LPIPS_delta=row['patch_lpips']-base['patch_lpips']))
    return series


def zero_summary(all_rows):
    grouped=defaultdict(Counter)
    for row in all_rows:
        if row['region'] in ZERO_FLOORS and row['method']!=A:
            key=by(row,('protocol','method','hidden','region'))
            grouped[key].update(cases=1,below_floor=int(row['psnr']<ZERO_FLOORS[row['region']]))
    return [dict(zip(('protocol','method','hidden','region'),key),floor_db=ZERO_FLOORS[key[-1]],**t)
            for key,t in sorted(grouped.items())]


def coverage_summary(all_rows):
    counts=('band_pixels','old_positions_with_candidate','new_positions_with_candidate','supplemented_positions',
            'supplemented_outside_old_cached_positions','previously_covered_now_filtered','new_positions_without_candidate')
    fields=('scene','protocol','aperture','名义对焦距离_m','method')
    grouped=defaultdict(Counter)
    for row in all_rows:
        if not row['query_applicable']:
            continue
        for scene in (row['scene'],'pooled'):
            grouped[(scene,*by(row,fields[1:]))].update({k:row[k] for k in counts})
    stats=[]
    for k,v in sorted(grouped.items()):
        n=v['band_pixels']
        stats.append(dict(zip(fields,k),**v,old_candidate_coverage=v['old_positions_with_candidate']/n if n else None,
                          new_candidate_coverage=v['new_positions_with_candidate']/n if n else None))
    return stats


def main():
    focuses=read(OUT/'R1.json')
    support=read(OUT/'R2.json')
    optics=read(OUT/'R3.json')
    rows=optics['rows']
    assert len(focuses['fixed_check'])==3 and all(r['passed'] for r in focuses['fixed_check'])
    background_sets=defaultdict(set)
    for row in focuses['rows']:
        if not row['P1拟合失败']:
            sample_counts=by(row,('P1共同背景像素数','P1拟合训练像素数','P1拟合留出像素数'))
            background_sets[row['scene'],row['frame']].add(sample_counts)
    assert all(len(v)==1 for v in background_sets.values())
    assert len({by(r,('frame','id',*KEYS)) for r in rows})==len(rows)
    assert all(not r.get('excluded_nonfinite_pixels',0) for r in rows)
    if not any(r['失败'] for r in focuses['rows']):
        assert len(rows)==72*(9*2*8+7) and len(optics['cache_checks'])==72*8
        assert len(optics['placements'])==72*9
    for r in rows:
        if r['pixels']:
            assert abs(r['psnr']+10*math.log10(max(r['sse']/(3*r['pixels']),1e-12)))<1e-8
    files={}
    frame_index=['scene','frame','id','protocol','method']
    placed={by(r,frame_index):r for r in optics['placements']}
    for k,v in placed.items():
        if v['method']==OLD:
            other=placed[(*k[:-1],OLD_FIXED)]
            for field in ('GT','Query','band_pixels','实际对焦距离_m'):
                assert v[field]==other[field]
    pooled=normalize(rows)
    old_new=compare_reference(list(old_rows()),rows)
    matched=deltas(pooled)
    coverage=coverage_summary(support['rows'])
    zero=zero_summary(rows)
    for title,vals in [('R1-focus-distances',focuses['rows']),('R2-query-coverage',support['rows']),
                       ('R2-coverage-pooled',coverage),('R3-optical',list(pooled.values())),
                       ('R3-old-new',old_new),('R3-pairs',matched),('R3-zero-floors',zero)]:
        files[title]=table(title,vals)
    def main_focus(row):
        return row['protocol']=='P2' and row['aperture']==1.4
    high_conf='03_实验记录/records/route3-ovg-1b-prime-2026-11-18-Q3-high-confidence.csv'
    summary=dict(protocol_revision='4a6ac37',trained=False,changed_weights_or_gate=False,
        result_scope='three same synthetic development scenes, 1,30,60 frames, 72 cases per method when valid',
        outputs=str(OUT.relative_to(ROOT)),终点评价组数=len({by(r,('scene','frame','id','protocol')) for r in rows}),
        R1=dict(fixed_check=focuses['fixed_check'],case_method_rows=len(focuses['rows']),
               per_frame_processing_seconds=sum(read(f[-1]/'R1.json')['seconds'] for f in case_frames()),
               failed=[r for r in focuses['rows'] if r['失败']],
               P2_low_support_rows=sum(r['protocol']=='P2' and r['P2有效率低于20pct'] for r in focuses['rows']),
               P1_RMSE_range_inv_m=[min(r['P1拟合留出RMSE_逆米'] for r in focuses['rows'] if r['P1拟合留出RMSE_逆米'] is not None),
                                  max(r['P1拟合留出RMSE_逆米'] for r in focuses['rows'] if r['P1拟合留出RMSE_逆米'] is not None)],
               common_background=True),
        R2=dict(seconds=support['seconds'],methods=['unchanged source context','regenerated target-conditioned candidate cache'],
                main_EMA=[r for r in coverage if r['method']==PRIME_EMA and r['scene']=='pooled' and main_focus(r)],
                cache_checks=len(optics['cache_checks'])),
        R3=dict(seconds=optics['seconds'],peak_torch_MiB=optics['peak_allocated_MiB'],K=optics['K'],
                full_metrics=str((OUT/'R3.json').relative_to(ROOT)),
                main_old_new=[r for r in old_new if main_focus(r) and r['method'] in (PRIME_EMA,OLD,FIRST)],
                main_pairs=[r for r in matched if main_focus(r) and r['scene']=='pooled'
                        and (r['名义对焦距离_m'],r['region']) in ((1.4,'in_focus'),(5.,'recoverable'))],
                zero_floors=zero),
        high_confidence_reference=dict(commit='aeda3af',table=high_conf),tables=files)
    save_json(REC/f'{PREFIX}-summary.json',summary)
    print('REFOCUS_REPORT_READY',len(rows),'optical scores',len(old_new),'old/new pooled rows',flush=True)


if __name__=='__main__':
    main()
