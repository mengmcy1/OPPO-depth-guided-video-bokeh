"""Compare the frozen v1 and interval-v2 endpoints, including preservation costs."""
import csv
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OLD=json.loads((ROOT/'configs/ours-student-pilot-v1-20261023.json').read_text())
NEW=json.loads((ROOT/'configs/ours-student-interval-v2-20261023.json').read_text())
PREFIX=ROOT/NEW['record_prefix']


def equal_source(rows,field):
    xs=[r[field] for r in rows if r[field] is not None]
    return float(np.mean(xs)) if xs else None


def main():
    records={name:json.loads((ROOT/(cfg['record_prefix']+'-results.json')).read_text()) for name,cfg in [('v1',OLD),('v2',NEW)]}
    allowed={'name','output','record_prefix','keep_loss','keep_mode','prior_config','single_factor','acceptance','scope'}
    assert all(OLD[k]==NEW[k] for k in OLD if k not in allowed)
    rows=[];summary=[];acceptance=[]
    for variant in OLD['variants']:
        a,b=records['v1']['runs'][variant],records['v2']['runs'][variant]
        for key in ['init_sha256','schedule_sha256','steps','exposures','branch_counts','actual_perturbation_changed_crop_pixels','perturbation_footprint_crop_pixels','parameters']:
            assert a[key]==b[key],(variant,key)
        x=records['v1']['per_scene'];y=records['v2']['per_scene']
        aa=[r for r in x if r['variant']==variant];bb=[r for r in y if r['variant']==variant]
        assert len(aa)==len(bb)
        for old,new in zip(aa,bb):
            for key in ['scene','split','branch','region','pixels','baseline_mae']:assert old[key]==new[key],key
            row={k:old[k] for k in ['variant','scene','split','branch','region','pixels']}
            row.update(baseline_mae=old['baseline_mae'],v1_mae=old['mae'],v2_mae=new['mae'],v1_mean_change=old['mean_change'],v2_mean_change=new['mean_change'],
                       v1_saturation=old['outside_0_1_fraction'],v2_saturation=new['outside_0_1_fraction'],
                       v2_vs_v1_mae_delta=new['mae']-old['mae'] if new['mae'] is not None else None)
            rows.append(row)
        for split in ['pilot_train','pilot_dev']:
            for branch in ['natural','perturbed']:
                for region in sorted({r['region'] for r in aa}):
                    rs=[r for r in rows if r['variant']==variant and r['split']==split and r['branch']==branch and r['region']==region]
                    row=dict(variant=variant,split=split,branch=branch,region=region,sources=sum(r['pixels']>0 for r in rs))
                    for key in ['baseline_mae','v1_mae','v2_mae','v1_mean_change','v2_mean_change','v1_saturation','v2_saturation']:row[key]=equal_source(rs,key)
                    for name in ['v1','v2']:
                        row[name+'_vs_baseline_percent']=(row[name+'_mae']/row['baseline_mae']-1)*100 if row['baseline_mae'] else None
                    row['v2_vs_v1_percent']=(row['v2_mae']/row['v1_mae']-1)*100 if row['v1_mae'] else None
                    summary.append(row)
        dev=[r for r in rows if r['variant']==variant and r['split']=='pilot_dev' and r['branch']=='natural']
        near=[r for r in dev if r['region']=='near_reference'];overall=[r for r in dev if r['region']=='valid']
        acceptance.append(dict(variant=variant,near_better_than_v1_on_both_sources=all(r['v2_mae']<r['v1_mae'] for r in near),
            near_not_worse_than_input_on_both_sources=all(r['v2_mae']<=r['baseline_mae'] for r in near),
            overall_not_worse_than_input_on_both_sources=all(r['v2_mae']<=r['baseline_mae'] for r in overall),
            overall_mean_not_worse_than_v1=equal_source(overall,'v2_mae')<=equal_source(overall,'v1_mae'),
            full_success=all(r['v2_mae']<=r['baseline_mae'] for r in near+overall)))
    manifest=json.loads((ROOT/NEW['data_manifest']).read_text());behavior=[]
    for scene in ['cave_2','market_2']:
        sample=next(r for r in manifest['frames'] if r['sequence']==scene)
        with np.load(ROOT/sample['vda_npz']) as z:d=z['disparity_normalized']
        with np.load(ROOT/sample['pair_npz']) as z:ref=z['reference'];near=z['near_reference']
        lo=np.minimum(d,ref);hi=np.maximum(d,ref);baseline=np.abs(d-ref)
        for variant in NEW['variants']:
            for version,cfg in [('v1',OLD),('v2',NEW)]:
                p=np.load(ROOT/cfg['output']/variant/'predictions'/scene/'natural.npz')['prediction']
                error=np.abs(p-ref);violation=np.maximum(lo-p,0)+np.maximum(p-hi,0)
                behavior.append(dict(scene=scene,variant=variant,version=version,near_pixels=int(near.sum()),
                    harm_fraction=float((error>baseline)[near].mean()),interval_violation_fraction=float((violation>0)[near].mean()),
                    interval_violation_mean=float(violation[near].mean(dtype=np.float64)),
                    note='clipped output; no inference GT gate applied; interval defined only for loss/diagnostics'))
    result=dict(status='INTERVAL_SINGLE_FACTOR_FOLLOWUP_COMPLETE',old_config=OLD,new_config=NEW,controlled_fields_equal=True,
                summary=summary,per_scene=rows,acceptance=acceptance,behavior=behavior,
                caveats=['Same development sources used to motivate and inspect this follow-up; not independent confirmation.',
                         'One seed and fixed final step; no significance claim or post-hoc best checkpoint selection.',
                         'Geometry metrics only; no new ATB or real-video result.'])
    with Path(str(PREFIX)+'-comparison.json').open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    for suffix,data in [('-comparison.csv',summary),('-comparison-per-scene.csv',rows),('-interval-behavior.csv',behavior)]:
        with Path(str(PREFIX)+suffix).open('x',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]),lineterminator='\n');writer.writeheader();writer.writerows(data)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,region,title in zip(axes,['valid','near_reference'],['Overall geometry MAE','Originally near-reference MAE']):
        rs=[r for r in summary if r['split']=='pilot_dev' and r['branch']=='natural' and r['region']==region]
        pos=np.arange(3);v1=[r['v1_mae'] for r in rs];v2=[r['v2_mae'] for r in rs]
        ax.bar(pos-.18,v1,.36,label='v1 dead zone',color='#80b1d3');ax.bar(pos+.18,v2,.36,label='v2 reference interval',color='#1261a0')
        ax.axhline(rs[0]['baseline_mae'],color='black',linestyle='--',label='unchanged input')
        ax.set(xticks=pos,xticklabels=[r['variant'] for r in rs],title=title,ylabel='Aligned-geometry MAE',ylim=(0,max(v1+v2+[rs[0]['baseline_mae']])*1.2))
        ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Two reused development sources; same seed/budget; fixed step2000')
    dest=ROOT/NEW['output']/'report';dest.mkdir(exist_ok=True)
    fig.savefig(dest/'preservation-v1-v2.png',dpi=160);fig.savefig(dest/'preservation-v1-v2.pdf');plt.close(fig)
    print(json.dumps([r for r in summary if r['split']=='pilot_dev' and r['branch']=='natural' and r['region'] in ['valid','near_reference']],indent=2),flush=True)
    print('ACCEPTANCE',json.dumps(acceptance),flush=True)


if __name__=='__main__':main()
