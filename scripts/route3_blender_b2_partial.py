"""Summarize already-rendered B2 data after a stop; launches no new renders."""
import argparse
import csv
import json
from pathlib import Path
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from route3_blender_b1_analyze import load_exr,metrics,passes


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--record',type=Path,required=True)
    a=p.parse_args();high=a.output/'B2-high';rows=json.loads((high/'convergence/timings.json').read_text());preflight=json.loads((high/'geometry-preflight/summary.json').read_text())
    groups=sorted({(r['frame'],r['fstop'],r['focus_m']) for r in rows});analysis=[];checks=[]
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for frame,stop,focus in groups:
        group=sorted([r for r in rows if (r['frame'],r['fstop'],r['focus_m'])==(frame,stop,focus)],key=lambda x:x['samples'])
        reference=next(r for r in group if r['samples']==1024);image=load_exr(high/'convergence'/(reference['key']+'.exr'));curve=[]
        for r in group:
            m=metrics(load_exr(high/'convergence'/(r['key']+'.exr')),image)
            value={**r,**m,'reference_samples':1024,'same_seed_reference_check_passed':passes(m)};analysis.append(value);curve.append(value)
        checks.append(dict(frame=frame,fstop=stop,focus_m=focus,reference_samples=1024,
                           independent_seed_rendered=False,terminal_convergence_certified=False,
                           candidate_same_seed_samples=[r['samples'] for r in curve if r['samples']<1024 and r['same_seed_reference_check_passed']]))
        for ax,key in zip(axes,['linear_rgb_rmse','boundary_linear_rgb_rmse']):
            ax.plot([r['samples'] for r in curve if r['samples']<1024],[r[key] for r in curve if r['samples']<1024],'o-',label=f't{frame} N{stop} F{focus}')
    for ax,title,threshold in zip(axes,['Whole-image linear RGB RMSE','Boundary linear RGB RMSE'],[.005,.01]):
        ax.axhline(threshold,color='black',linestyle='--',linewidth=1);ax.set_title(title);ax.set_xlabel('Samples per pixel');ax.set_ylabel('Error vs same-seed1024')
        ax.set_xscale('log',base=2);ax.set_yscale('log');ax.set_xticks([128,256,512],labels=['128','256','512']);ax.grid(alpha=.2)
    axes[1].legend(ncol=2,fontsize=7);fig.suptitle('B2-high partial convergence: independent terminal noise test NOT performed')
    fig.savefig(high/'convergence/sampling-partial.png',dpi=170);plt.close(fig)
    for r in preflight:
        scale=512/r['resolution'][0]
        for field in ['coc_hidden_p10_p50_p90','coc_blocking_p10_p50_p90','coc_pairmax_p10_p50_p90']:
            r[field+'_at512']=(np.asarray(r[field])*scale).tolist() if r[field] else None
        r['low_coc_pairmax_median_below3_at512']=r['coc_pairmax_p10_p50_p90_at512'][1]<3 if r['coc_pairmax_p10_p50_p90_at512'] else True
    timing={str(s):dict(renders=sum(r['samples']==s for r in rows),mean_seconds=float(np.mean([r['elapsed_seconds'] for r in rows if r['samples']==s])),
                       min_seconds=float(min(r['elapsed_seconds'] for r in rows if r['samples']==s)),max_seconds=float(max(r['elapsed_seconds'] for r in rows if r['samples']==s))) for s in [128,256,512,1024]}
    result=dict(phase='B2 partial stopped with A-line background-depth failure',status='paused_before_independent_noise_and_formal_inputs',
        stop_reason='Parent reported CamVFX cat reliable-background depth support73/95/0 pixels across views; user stop condition: do not expand',
        software=dict(blender='4.5.3 LTS',binary='/root/autodl-tmp/tools/blender-4.5.3-linux-x64/blender',GPU='RTX4090D GPU0, OptiX',CPU_threads=12),
        scene_design=dict(low='Unmodified B1 scene saved to new B2-low directory; original preserved',
            high='camera near y=-1.5m; static occluders y=-.1/0m; moving sphere y=.2m; background y=3.4m; focus1.4/5.0m; altered framing, rigid sphere/camera motion, thin stems',
            assets='original procedural primitives; checker retained with fine nonperiodic noise, procedural mineral/woodlike color and bumps; no third-party assets',assets_license='CC0-1.0 generated assets'),
        completed=['low/high.blend and stable local topology/60 rigid object and camera matrices','4 high low-sample256² previews',
                   '64²K16 geometric aperture/center/other12-input visibility preflight at frames1/30/60, N1.4/2.8, both focuses',
                   '48 high main-seed512² renders:2frames x3fstops x2focuses x4SPP128/256/512/1024'],
        not_completed=dict(independent_seed_noise='not started after stop',sampling_configuration='not frozen; same-seed curves alone do not certify1024',
            pinhole60='not started',full_resolution_auxiliary_and_visibility_GT='not started',representative_formal_DOF='not started',hidden_layer_reference='not started',
            criteria_file='not created; method evaluation has not begun',O3_O5_O6_DrBokeh='not evaluated',formal_representative_frame_list='not frozen'),
        preflight=dict(rows=preflight,CoC_reporting='native64px and scaled512px CoC both recorded; no method results used',
            real_visibility='Embree first-hit surface; test center invisibility by recasting same surface point; cross-frame recoverability tested by stable object-local point, not object ID alone',
            visibility_tolerance_m=1e-4,input_frames=[1,6,11,16,21,26,30,35,40,45,50,55,60],
            candidate_representative_frames=[1,30,60],selection_reason='left,center,right sphere/camera trajectory positions; selected from input list and geometry, not method scores',
            cross_input_visible_definition='surface appears in another one of12 input frames',
            hidden_foreground_support=dict(rays=sum(r['includes_hidden_foreground_rays'] for r in preflight),
                fraction_of_hidden=sum(r['includes_hidden_foreground_rays'] for r in preflight)/max(sum(r['hidden_rays'] for r in preflight),1),
                limitation='Some hidden first-hit surfaces are foreground; resumed B2 needs additional selectively hidden reference layers/support.512² confirmation not performed')),
        sampling=dict(rows=analysis,terminal_checks=checks,timing_by_spp=timing,
            independent_noise_estimate=None,main_seed=20261110,denoising=False,motion_blur=False,adaptive_sampling=False,
            thresholds=dict(full_linear_rmse=.005,boundary_linear_rmse=.01,highlight_luminance_relative=.02,highlight_centroid_px=.25),
            status='same-seed partial curve only; no formal convergence or method judgment'),
        outputs=dict(root=str(a.output.resolve()),preview=str(high/'preview'),curve=str(high/'convergence/sampling-partial.png'),geometry=str(high/'scene-geometry.npz')),
        GPU_released=True,method_evaluation_started=False,criteria_frozen=False,model_training=False,formal60frame_DOF=False)
    a.record.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    with a.record.with_suffix('.csv').open('w',newline='',encoding='utf-8') as f:
        keys=[k for k in analysis[0] if k!='resolution'];w=csv.DictWriter(f,fieldnames=keys,lineterminator='\n');w.writeheader();w.writerows({k:r.get(k) for k in keys} for r in analysis)
    print(json.dumps(dict(status=result['status'],completed_renders=len(rows),timing_by_spp=timing,geometry_probe_seconds=sum(r['elapsed_seconds'] for r in preflight)),indent=2))


if __name__=='__main__':main()
