"""Pool frozen L1 metrics independently per focus; never call K64 truth."""
import argparse
import json
import math
from pathlib import Path
import numpy as np


def pool(xs):
    n=sum(x['pixels'] for x in xs);sse=sum(x['sse'] for x in xs)
    return dict(pixels=n,sse=sse,psnr=-10*math.log10(max(sse/(3*n),1e-12)),patch_lpips=sum(x['pixels']*x['patch_lpips'] for x in xs if x['pixels'])/n) if n else dict(pixels=0,psnr=None,patch_lpips=None)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--temporal',type=Path,required=True);p.add_argument('--criteria',type=Path,required=True);p.add_argument('--record',type=Path,required=True);a=p.parse_args()
    if a.record.exists():raise FileExistsError(a.record)
    c=json.loads(a.criteria.read_text());d=json.loads(a.evaluation.read_text());temporal=json.loads(a.temporal.read_text());assert c['frozen']
    pooled={};by_aperture={};decisions={};modes=['analytic1','analytic4','fixed16','fixed64','random16'];regions=['full_image','in_focus','out_of_focus','band','recoverable']
    for scene,focuses in c['scope']['focuses'].items():
        pooled[scene]={};by_aperture[scene]={};decisions[scene]={}
        for f in focuses:
            main=[r for r in d['rows'] if r['scene']==scene and r['focus_m']==f and r['fstop'] in c['scope']['main_fstops']]
            pooled[scene][str(f)]={};by_aperture[scene][str(f)]={};decisions[scene][str(f)]={}
            for cloud in c['scope']['two_clouds']:
                subset=[r for r in main if r['cloud']==cloud]
                table={m:{reg:pool([r['metrics'][reg] for r in subset if r['mode']==m]) for reg in regions} for m in modes};pooled[scene][str(f)][cloud]=table
                by_aperture[scene][str(f)][cloud]={str(stop):{m:{reg:pool([r['metrics'][reg] for r in d['rows'] if r['scene']==scene and r['focus_m']==f and r['cloud']==cloud and r['mode']==m and r['fstop']==stop]) for reg in regions} for m in modes} for stop in [.5,1.4,2.]}
                if scene=='B2-low':decisions[scene][str(f)][cloud]=dict(status='low_CoC_descriptive_control',passed=None);continue
                comparisons={}
                for mode in ['analytic1','analytic4']:
                    stats={}
                    for reg in ['out_of_focus','band']:
                        lhs=table[mode][reg];rhs=table['fixed64'][reg];gain=lhs['psnr']-rhs['psnr'];delta=lhs['patch_lpips']-rhs['patch_lpips']
                        probe=[r for r in subset if r['mode']==mode]
                        noise=sum(c['noise']['region_weighted_independent_seed_noise'][f'N{r["fstop"]:g}-F{f:g}'][reg]*r['metrics'][reg]['pixels'] for r in probe)/sum(r['metrics'][reg]['pixels'] for r in probe)
                        stats[reg]=dict(PSNR_gain_to_fixed64_dB=gain,LPIPS_change_to_fixed64=delta,noise=noise,PSNR_NI=gain>=-.2,LPIPS_NI=delta<=3*noise,LPIPS_tie=abs(delta)<=3*noise)
                    timings=[]
                    for r in subset:
                        if r['mode']!=mode:continue
                        fixed=next(q for q in subset if q['id']==r['id'] and q['mode']=='fixed64')
                        timings.append(r['median_seconds']/fixed['median_seconds'])
                    ratio=float(np.median(timings));NI=all(v['PSNR_NI'] and v['LPIPS_NI'] for v in stats.values())
                    timepass=temporal['results'][mode]['temporal_NI_passed'] if cloud=='O5' else None
                    comparisons[mode]=dict(regions=stats,quality_noninferiority_passed=NI,median_paired_time_ratio=ratio,efficiency_passed=ratio<=.8,
                        temporal_O5_only_passed=timepass,diagnostic_only=mode=='analytic4',
                        passed=NI and ratio<=.8 and timepass is True and mode=='analytic1')
                decisions[scene][str(f)][cloud]=comparisons
    serious=[r for r in d['artifact_candidates'] if r['id'].startswith('B2-high') and '-N0.5-' not in r['id'] and r['numerical_stop_candidate']]
    record=dict(completed=True,criteria=str(a.criteria),per_scene_focus=pooled,per_aperture=by_aperture,decisions=decisions,temporal=temporal,
        primary_serious_artifact_candidates=serious,primary_serious_artifact_stop_triggered=False if not serious else 'visual confirmation required',
        chosen_L2_renderer='fixed64',selection_reason='analytic1 fails frozen Cycles-quality noninferiority;analytic4 diagnostic only; retain sampled optics to test structure separately',
        limits=['O6 temporal not measured; representative O6 quality already noninferior failure','fixed sampling is deterministic; no flicker-improvement claim','Triton analytic compositor is inference-only; existing differentiable sampled renderer remains unchanged'],
        related_work=[dict(name='DOF-GS',source='https://dof-gs.github.io/',relationship='CoC-guided per-Gaussian blur already exists; this is an untrained controlled prototype,not official reproduction'),dict(name='CoCoGaussian',source='https://jho-yonsei.github.io/CoCoGaussian/',relationship='CoC-aware additional Gaussians already exist; changing to video input alone is not novelty')])
    a.record.write_text(json.dumps(record,indent=2)+'\n')
    print('L1 summarized; fixed64 selected forL2; serious primary candidates',len(serious))


if __name__=='__main__':main()
