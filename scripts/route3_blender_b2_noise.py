"""Independent-seed Cycles noise, before B2 geometry/method evaluation."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
import cv2
import numpy as np
import torch
import lpips

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from route3_blender_b1_analyze import load_exr,metrics


def srgb(linear):
    x=np.maximum(linear,0)
    return np.clip(np.where(x<=.0031308,12.92*x,1.055*np.maximum(x,1e-12)**(1/2.4)-.055),0,1).astype(np.float32)


def patch_noise(model,a,b,reference):
    a=srgb(a);b=srgb(b);luma=reference@np.array([.2126,.7152,.0722])
    grad=np.hypot(cv2.Sobel(luma,cv2.CV_64F,1,0,ksize=3)/8,cv2.Sobel(luma,cv2.CV_64F,0,1,ksize=3)/8)
    edge=cv2.dilate((grad>=.05).astype(np.uint8),np.ones((7,7),np.uint8))>0
    positions=[(y,x) for y in range(0,512,64) for x in range(0,512,64)]
    pa=torch.tensor(np.stack([a[y:y+64,x:x+64] for y,x in positions])).permute(0,3,1,2)
    pb=torch.tensor(np.stack([b[y:y+64,x:x+64] for y,x in positions])).permute(0,3,1,2)
    scores=[]
    with torch.inference_mode():
        for start in range(0,len(pa),16):scores.extend(model(2*pa[start:start+16]-1,2*pb[start:start+16]-1).flatten().tolist())
    scores=np.asarray(scores);weights=np.array([edge[y:y+64,x:x+64].sum() for y,x in positions])
    return dict(mean=float(scores.mean()),p50_p90_p95_max=np.quantile(scores,[.5,.9,.95,1]).tolist(),
                boundary_weighted=float(np.sum(scores*weights)/max(weights.sum(),1)),
                scores=scores.tolist(),patch_positions_yx=positions,boundary_support_pixels=weights.tolist())


def collect(root):
    sources=[('B2-high',root/'outputs/route3-a2b-2026-11-10-01/blender-b2/B2-high/convergence'),
             ('B2-low',root/'outputs/route3-c1-2026-11-10-01/blender-b1'),
             ('B2-high',root/'outputs/route3-b2only-2026-11-10-01/noise/B2-high'),
             ('B2-low',root/'outputs/route3-b2only-2026-11-10-01/noise/B2-low')]
    result={}
    for label,p in sources:
        if not (p/'timings.json').exists():continue
        for row in json.loads((p/'timings.json').read_text()):
            if row.get('pinhole',False) or row.get('resolution')!=[512,512] or row.get('hidden',False):continue
            if row.get('device','GPU')!='GPU':continue
            key=(label,row['frame'],float(row['fstop']),float(row['focus_m']),row['samples'],row.get('seed',20261110))
            result[key]={**row,'path':str(p/(row['key']+'.exr'))}
    return result


def plot_noise(rows,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    groups=sorted({(r['scene'],r['fstop'],r['focus_m']) for r in rows})
    fig,axes=plt.subplots(1,2,figsize=(13,4.8),constrained_layout=True)
    for scene,stop,focus in groups:
        values=[r for r in rows if (r['scene'],r['fstop'],r['focus_m'])==(scene,stop,focus)]
        spp=sorted({r['samples'] for r in values})
        boundary=[max(r['boundary_linear_rgb_rmse'] for r in values if r['samples']==s) for s in spp]
        lp=[max(max(r['noise_patch_lpips']['mean'],r['noise_patch_lpips']['boundary_weighted']) for r in values if r['samples']==s) for s in spp]
        label=f'{scene} N{stop:g} F{focus:g}'
        axes[0].plot(spp,boundary,'o-',label=label);axes[1].plot(spp,lp,'o-',label=label)
    for ax,title,threshold in zip(axes,['Maximum boundary linear-RGB noise','Maximum full/boundary patch-LPIPS noise'],[.01,.002]):
        ax.axhline(threshold,color='black',ls='--');ax.set_xscale('log',base=2);ax.set_yscale('log');ax.set_xticks([512,1024,2048],labels=['512','1024','2048'])
        ax.set_title(title);ax.set_xlabel('Samples per pixel');ax.set_ylabel('Independent seed difference');ax.grid(alpha=.2)
    axes[1].legend(ncol=2,fontsize=6);fig.suptitle('Independent-seed noise: worst of probe frames1/30, before method evaluation')
    path.parent.mkdir(parents=True,exist_ok=True);fig.savefig(path,dpi=170);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--record',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(8);cv2.setNumThreads(4)
    model=lpips.LPIPS(net='alex',version='0.1').eval();allrows=collect(a.project);results=[]
    pairs=sorted({key[:5] for key in allrows if key[-1]==20261111})
    for key in pairs:
        ka=(*key,20261110);kb=(*key,20261111)
        if ka not in allrows or kb not in allrows:continue
        left=load_exr(allrows[ka]['path']);right=load_exr(allrows[kb]['path']);m=metrics(right,left);lp=patch_noise(model,left,right,left)
        passed=m['linear_rgb_rmse']<=.005 and m['boundary_linear_rgb_rmse']<=.01 and max(lp['mean'],lp['boundary_weighted'])<=.002
        row=dict(scene=key[0],frame=key[1],fstop=key[2],focus_m=key[3],samples=key[4],
            main_seed=20261110,independent_seed=20261111,main_path=allrows[ka]['path'],independent_path=allrows[kb]['path'],
            **m,noise_patch_lpips=lp,passed=bool(passed))
        results.append(row);print('NOISE_CHECK',json.dumps({k:v for k,v in row.items() if k not in ['noise_patch_lpips','main_path','independent_path']}),lp['mean'],lp['boundary_weighted'],flush=True)
    configs={}
    for row in results:
        config=(row['scene'],row['fstop'],row['focus_m']);configs.setdefault(config,[]).append(row)
    selected=[]
    for (scene,stop,focus),rows in sorted(configs.items()):
        spp=sorted(set(r['samples'] for r in rows));accepted=[s for s in spp if all(any(r['frame']==f and r['samples']==s and r['passed'] for r in rows) for f in [1,30])]
        selected.append(dict(scene=scene,fstop=stop,focus_m=focus,selected_samples=min(accepted) if accepted else None,
                             assessed_samples=spp,passed_for_both_representative_probe_frames=bool(accepted)))
    required_high={(stop,focus) for stop in [.5,1.4,2.,2.8] for focus in [1.4,5.]}
    passed_high={(r['fstop'],r['focus_m']) for r in selected if r['scene']=='B2-high' and r['selected_samples'] is not None}
    complete=required_high<=passed_high and all(any(r['scene']=='B2-low' and r['fstop']==2 and r['focus_m']==f and r['selected_samples'] is not None for r in selected) for f in [5.3,8.4])
    formal=[]
    for scene,focuses in [('B2-high',[1.4,5.]),('B2-low',[5.3,8.4])]:
        for stop in [.5,1.4,2.]:
            for focus in focuses:
                selected_row=next((r for r in selected if r['scene']==scene and r['fstop']==stop and r['focus_m']==focus),None)
                spp=selected_row['selected_samples'] if selected_row else None
                if scene=='B2-low':spp=2048 if stop==.5 else 512
                formal.append(dict(scene=scene,fstop=stop,focus_m=focus,samples=spp,
                    formal_role='pressure' if stop==.5 else 'primary',
                    source='B1 validated sample configuration reused' if scene=='B2-low' and stop in [.5,1.4] else 'current independent-pair checks'))
    result=dict(stage='B2-only step1 before formal inputs/GT/methods',passed=complete,phase1_noise_passed=complete,
        criteria=dict(linear_RGB_RMSE_max=.005,boundary_RMSE_max=.01,patch_LPIPS_mean_and_boundary_weighted_max=.002,
                      LPIPS='alexnet version0.1,64x64 fixed grid, whole unmasked patches; CPU8threads',
                      LPIPS_domain='scene-linear EXR -> standard sRGB clipped0..1; linear RMSE remains unclipped',
                      official_method_delta_LPIPS_candidate=-.01,noise_ratio_rationale='noise threshold<=one fifth of the method LPIPS change candidate; set before method results',
                      independent_seeds=[20261110,20261111],reference='independent pair at same SPP;1024 not treated as noise-free GT'),
        rows=results,selected_samples=selected,formal_samples=formal,low_B1_reuse=dict(f1_4=512,f0_5=2048,
             note='prior B1 independent-seed checks retained; formal f2.0 is assessed here'),
        method_evaluation_started=False,GT_generation_started=False,formal_criteria_frozen=False)
    plot=a.project/'outputs/route3-b2only-2026-11-10-01/noise/independent-noise.png';plot_noise(results,plot);result['noise_plot']=str(plot)
    newrows=[]
    for scene in ['B2-high','B2-low']:
        runtime=a.project/'outputs/route3-b2only-2026-11-10-01/noise'/scene/'timings.json'
        if runtime.exists():newrows.extend(json.loads(runtime.read_text()))
    result['new_noise_render_runtime']=dict(images=len(newrows),sum_render_EXR_write_seconds=sum(r['elapsed_seconds_render_and_EXR_write'] for r in newrows),
        by_samples={str(s):dict(images=sum(r['samples']==s for r in newrows),mean_seconds=float(np.mean([r['elapsed_seconds_render_and_EXR_write'] for r in newrows if r['samples']==s]))) for s in sorted({r['samples'] for r in newrows})},
        boundary='render+EXR write only; excludes startup, PNG previews and CPU LPIPS')
    a.record.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    with a.record.with_suffix('.csv').open('w',newline='',encoding='utf-8') as f:
        fields=['scene','frame','fstop','focus_m','samples','linear_rgb_rmse','boundary_linear_rgb_rmse','noise_lpips_mean','noise_lpips_boundary','passed']
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader()
        for r in results:w.writerow({**{k:r[k] for k in fields if k in r},'noise_lpips_mean':r['noise_patch_lpips']['mean'],'noise_lpips_boundary':r['noise_patch_lpips']['boundary_weighted']})
    print('NOISE_COMPLETE',complete,selected,flush=True)


if __name__=='__main__':main()
