"""Optional physical stratified K16 diagnosis, without residual covariance."""
import argparse
import json
import os
import sys
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from route3_sharp_s1c import OUT,PREFIX,RECORDS,CRITERIA,write
from route3_sharp_s1b_residual import cloud_inputs,gradient_cost
from route3_upper_bound_evaluate import items,display,exr,score,pool
from route3_ldr_u1r import model
from src.subaperture_residual import render_residual
from src.dual_layer import depth_band
SEEDS=[2026111011,2026111012,2026111013,2026111014]


def render(c):
    subprocess.run(['nvidia-smi'],check=True)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise ValueError('Explicit GPU0')
    root=OUT/'random16';root.mkdir(exist_ok=False);oldc=json.loads((RECORDS/'route3-sharp-s1b-criteria-2026-11-11.json').read_text());baseline=json.loads((RECORDS/'route3-sharp-s1b-2026-11-11-residual-render.json').read_text())['rows'];rows=[];profiles=[]
    for frame in [1,30,60]:
        for cloud in ['O5','O6-GT-rec']:
            d,K,E,pc,fill,bg=cloud_inputs(frame,cloud,oldc)
            for case in [x for x in items() if x['frame_id']==frame]:
                for seed in SEEDS:
                    with torch.inference_mode():
                        torch.cuda.synchronize();start=time.monotonic();r=render_residual(pc,K,E,512,512,case['focus_distance_m'],case['aperture_radius_m'],16,fill,bg,residual=False,seed=seed);torch.cuda.synchronize();seconds=time.monotonic()-start
                    folder=root/case['id']/cloud;folder.mkdir(parents=True,exist_ok=True);path=folder/f'seed{seed}.npz';np.savez_compressed(path,prediction_srgb=display(r['rgb_linear'].cpu().numpy()),alpha=r['alpha'].cpu().numpy())
                    rows.append(dict(id=case['id'],cloud=cloud,seed=seed,path=str(path),forward_seconds=seconds))
                    if frame==30 and seed==SEEDS[0]:
                        for mode,k,s in [('fixed64',64,None),('stratified16',16,seed)]:profiles.append(dict(id=case['id'],cloud=cloud,mode=mode,**gradient_cost(pc,K,E,case['focus_distance_m'],case['aperture_radius_m'],k,False,s,fill,bg)))
                    print('S1C_RANDOM16',case['id'],cloud,seed,flush=True)
            del pc;torch.cuda.empty_cache()
    write(root/'predictions.json',dict(completed=True,rows=rows,profiles=profiles,baseline=[r for r in baseline if r['mode']=='fixed64'],residual_covariance=False,sample_center_contraction=False));write(RECORDS/(PREFIX+'-random16-render.json'),dict(completed=True,profiles=profiles,quality_outputs=len(rows),residual_covariance=False,GT_clouds_original_L1=True))


def evaluate(c):
    run=json.loads((OUT/'random16/predictions.json').read_text());network=model();lookup={x['id']:x for x in items()};baseline=json.loads((RECORDS/'route3-sharp-s1b-2026-11-11-residual-evaluation.json').read_text())['metrics'];metrics={};variance=[]
    for case in items():
        d=np.load(case['input_npz']);band,_,r=depth_band(d['center_depth'],d['K'][0,0],case['focus_distance_m'],case['aperture_radius_m']);masks=dict(full_image=np.ones((512,512),bool),in_focus=d['center_surface_valid']&(r<=1),out_of_focus=d['center_surface_valid']&(r>=2),band=band);truth=display(exr(case['gt_rgb_exr']));metrics[case['id']]={}
        for cloud in ['O5','O6-GT-rec']:
            group=[x for x in run['rows'] if x['id']==case['id'] and x['cloud']==cloud];images=[np.load(x['path'])['prediction_srgb'] for x in group];values=[score(image,truth,masks,network) for image in images];var=np.var(np.stack(images).astype(np.float64),axis=0,ddof=1)
            metrics[case['id']][cloud]={str(x['seed']):v for x,v in zip(group,values)}
            variance.append(dict(id=case['id'],cloud=cloud,variance_mean_by_region={reg:float(var[mask].mean()) for reg,mask in masks.items()},single_draw_PSNR_std={reg:float(np.std([v[reg]['psnr'] for v in values],ddof=1)) for reg in masks}))
    pooled={}
    for cloud in ['O5','O6-GT-rec']:
        pooled[cloud]={}
        for focus in [1.4,5.]:
            ids=[x['id'] for x in items() if x['focus_distance_m']==focus];table={}
            for reg in ['full_image','in_focus','out_of_focus','band']:
                a=pool([metrics[i][cloud][str(seed)][reg] for i in ids for seed in SEEDS]);b=pool([baseline[i][cloud]['fixed64'][reg] for i in ids]);table[reg]=dict(expected_single_draw=a,K64=b,PSNR_change_dB=a['psnr']-b['psnr'],LPIPS_change=a['patch_lpips']-b['patch_lpips'])
            pooled[cloud][str(focus)]=table
    write(RECORDS/(PREFIX+'-random16-evaluation.json'),dict(completed=True,descriptive_only=True,seeds=SEEDS,pooled=pooled,per_case_metrics=metrics,single_draw_variance=variance,profiles=run['profiles'],quality_is_expected_single_K16_MSE_not_averaged64ray_image=True,variance_ddof=1,no_formal_main_gate=True))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['render','evaluate'],required=True);a=p.parse_args();torch.set_num_threads(8);globals()[a.mode](json.loads(CRITERIA.read_text()))


if __name__=='__main__':main()
