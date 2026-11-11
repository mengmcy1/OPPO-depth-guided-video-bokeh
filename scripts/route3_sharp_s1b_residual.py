"""Independent S1b GT-cloud quadrature/residual quality and gradient cost."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
import cv2
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from src.aperture_renderer import GaussianScene
from src.layered_aperture import pixel_layer,merge_layers
from src.subaperture_residual import render_residual
from route3_b2_oracle_3d import sparse_pixel_layer
from route3_upper_bound_evaluate import BASE,RECORDS,items,display,exr,score,pool
from route3_ldr_u1r import model
OUT=ROOT/'outputs/route3-sharp-s1b-2026-11-11-01'
PREFIX='route3-sharp-s1b-2026-11-11'
CRITERIA=RECORDS/'route3-sharp-s1b-criteria-2026-11-11.json'
RUN_NAME='residual-hoistedcov'
SEEDS=[2026111123,2026111124,2026111125,2026111126]
MODES=[('fixed64',64,False,None),('fixed16',16,False,None),('fixed16_residual',16,True,None)]+[(f'stratified16_residual_seed{s}',16,True,s) for s in SEEDS]


def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def cloud_inputs(frame,kind,c):
    d=np.load(BASE/'common-inputs/B2-high'/f'f{frame:03d}.npz');K=torch.tensor(d['K'],device='cuda',dtype=torch.float32)
    camera=torch.tensor(d['camera_c2w'],device='cuda',dtype=torch.float32);E=torch.linalg.inv(camera)
    visible=pixel_layer(torch.tensor(d['visible_rgb_linear'],device='cuda'),torch.tensor(d['center_depth'],device='cuda'),K,camera,torch.tensor(d['center_surface_valid'],device='cuda'))
    if kind=='O6-GT-rec':
        parent=np.load(BASE/'methods-3d/prepared'/f'B2-high-f{frame:03d}'/'parents.npz')
        hidden=sparse_pixel_layer(parent,parent['recoverable'],K,camera,c['residual']['profile'],'cuda');visible=merge_layers(visible,hidden)
    fill=torch.tensor(np.load(BASE/'methods-2d/fill'/f'B2-high-f{frame:03d}'/'fill-image.npz')['fill_rgb_linear'],device='cuda')
    bg=torch.tensor(d['background_depth'],device='cuda')
    return d,K,E,visible,fill,bg


def gradient_cost(cloud,K,E,focus,aperture,k,residual,seed,fill,bg):
    keys=['means','quats','scales','opacities','colors']
    pc=GaussianScene(*[getattr(cloud,key).detach().clone().requires_grad_(True) for key in keys],cloud.source_frames)
    forward=[];backward=[];peak=[];extra=[]
    for repeat in range(4):
        for key in keys:getattr(pc,key).grad=None
        torch.cuda.synchronize();baseline=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();start=time.monotonic()
        result=render_residual(pc,K,E,512,512,focus,aperture,k,fill,bg,residual=residual,seed=seed,checkpoint_rays=True)
        loss=result['rgb_linear'].square().mean();torch.cuda.synchronize();middle=time.monotonic();loss.backward();torch.cuda.synchronize();end=time.monotonic()
        if not all(getattr(pc,key).grad is not None and bool(torch.isfinite(getattr(pc,key).grad).all()) for key in keys):raise RuntimeError('S1B_STOP residual profiling nonfinite gradient')
        if repeat:
            forward.append(middle-start);backward.append(end-middle);peak.append(torch.cuda.max_memory_allocated()/2**20);extra.append((torch.cuda.max_memory_allocated()-baseline)/2**20)
        del result,loss
    return dict(forward_seconds=forward,backward_seconds=backward,median_forward_seconds=float(np.median(forward)),median_backward_seconds=float(np.median(backward)),
        median_total_seconds=float(np.median(np.array(forward)+backward)),peak_allocated_MiB=max(peak),incremental_peak_MiB=max(extra),
        all_attribute_gradients_finite=True,gradient_checkpoint_each_ray=True,parameter_values_updated=False)


def render(c):
    root=OUT/RUN_NAME;root.mkdir(exist_ok=False);rows=[];profiles=[]
    # This profile subset is fixed by frame/time, never selected by quality.
    write(root/'execution-plan.json',dict(frozen_before_residual_metrics=True,quality_frames=[1,30,60],profile_frames=[30],profile_all_focuses_and_apertures_and_clouds=True,
        random_quality_seeds=SEEDS,random_gradient_profile_seed=SEEDS[0],gradient_checkpoint_all_modes=True,timed_repetitions=3,warmup=1))
    for frame in [1,30,60]:
        for kind in ['O5','O6-GT-rec']:
            d,K,E,cloud,fill,bg=cloud_inputs(frame,kind,c)
            for case in [x for x in items() if x['frame_id']==frame]:
                folder=root/case['id']/kind;folder.mkdir(parents=True)
                for name,k,residual,seed in MODES:
                    with torch.inference_mode():
                        torch.cuda.synchronize();started=time.monotonic();result=render_residual(cloud,K,E,512,512,case['focus_distance_m'],case['aperture_radius_m'],k,fill,bg,residual=residual,seed=seed)
                        torch.cuda.synchronize();seconds=time.monotonic()-started
                    path=folder/(name+'.npz');np.savez_compressed(path,rgb_linear=result['rgb_linear'].cpu().numpy(),prediction_srgb=display(result['rgb_linear'].cpu().numpy()),alpha=result['alpha'].cpu().numpy())
                    rows.append(dict(id=case['id'],frame=frame,focus_m=case['focus_distance_m'],fstop=case['fstop'],cloud=kind,mode=name,K=k,residual=residual,seed=seed,
                        Gaussians=len(cloud.means),path=str(path),forward_inference_seconds=seconds))
                    if frame==30 and (seed is None or seed==SEEDS[0]):
                        profile=gradient_cost(cloud,K,E,case['focus_distance_m'],case['aperture_radius_m'],k,residual,seed,fill,bg)
                        profiles.append(dict(id=case['id'],cloud=kind,mode=name,**profile))
                    print('RESIDUAL_RENDER',case['id'],kind,name,seconds,flush=True)
                    write(root/'progress.json',dict(completed=False,rows=rows,profiles=profiles))
                del result
            del cloud;torch.cuda.empty_cache()
    report=dict(completed=True,rows=rows,profiles=profiles,profile_frame=30,ordinary_default_physical=True,residual_is_moment_approximation=True,determinant_implementation='explicit2x2',covariance_implementation='camera_rotation_and_residual_hoisted_once',
        L1_original_cloud_and_fill_convention=True,no_optimization_or_network_training=True)
    write(root/'predictions.json',report);write(RECORDS/(PREFIX+'-residual-render.json'),report)


def evaluate(c):
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    root=OUT/(RUN_NAME+'-evaluation');root.mkdir(exist_ok=False);network=model();run=json.loads((OUT/RUN_NAME/'predictions.json').read_text());lookup={x['id']:x for x in items()}
    metrics={};flat=[];plots=[]
    from src.dual_layer import depth_band
    for row in run['rows']:
        case=lookup[row['id']];d=np.load(case['input_npz']);truth=display(exr(case['gt_rgb_exr']));band,_,radius=depth_band(d['center_depth'],d['K'][0,0],case['focus_distance_m'],case['aperture_radius_m'])
        masks=dict(full_image=np.ones((512,512),bool),in_focus=d['center_surface_valid']&(radius<=1),out_of_focus=d['center_surface_valid']&(radius>=2),band=band)
        pred=np.load(row['path'])['prediction_srgb'];values=score(pred,truth,masks,network)
        metrics.setdefault(row['id'],{}).setdefault(row['cloud'],{})[row['mode']]=values
        for region,v in values.items():flat.append(dict(id=row['id'],cloud=row['cloud'],mode=row['mode'],region=region,**v))
        print('RESIDUAL_SCORE',row['id'],row['cloud'],row['mode'],values['band']['psnr'],flush=True)
    pooled={};decisions={}
    for kind in ['O5','O6-GT-rec']:
        pooled[kind]={};decisions[kind]={}
        for focus in [1.4,5.]:
            ids=[x['id'] for x in items() if x['focus_distance_m']==focus];table={}
            for mode in [m[0] for m in MODES]:table[mode]={reg:pool([metrics[i][kind][mode][reg] for i in ids]) for reg in ['full_image','in_focus','out_of_focus','band']}
            # Expected single-draw quality pools independent seed SSE, rather
            # than presenting an effective K64 image average as K16 quality.
            table['stratified16_residual_expected']={reg:pool([metrics[i][kind][f'stratified16_residual_seed{s}'][reg] for i in ids for s in SEEDS]) for reg in ['full_image','in_focus','out_of_focus','band']}
            pooled[kind][str(focus)]=table;decisions[kind][str(focus)]={}
            for mode in ['fixed16','fixed16_residual','stratified16_residual_expected']:
                drop={reg:table[mode][reg]['psnr']-table['fixed64'][reg]['psnr'] for reg in ['out_of_focus','band']}
                decisions[kind][str(focus)][mode]=dict(Cycles_PSNR_change_vs_fixed64_dB=drop,noninferior=all(v>=-.2 for v in drop.values()))
    for case in items():
        truth=display(exr(case['gt_rgb_exr']));d=np.load(case['input_npz']);band=depth_band(d['center_depth'],d['K'][0,0],case['focus_distance_m'],case['aperture_radius_m'])[0]
        y,x=np.nonzero(band);index=np.argmin((y-np.median(y))**2+(x-np.median(x))**2);cy=int(np.clip(y[index]-64,0,384));cx=int(np.clip(x[index]-64,0,384))
        for kind in ['O5','O6-GT-rec']:
            chosen=[next(r for r in run['rows'] if r['id']==case['id'] and r['cloud']==kind and r['mode']==mode) for mode in ['fixed64','fixed16','fixed16_residual',f'stratified16_residual_seed{SEEDS[0]}']]
            images=[truth]+[np.load(r['path'])['prediction_srgb'] for r in chosen];fig,axes=plt.subplots(2,5,figsize=(15,6))
            for column,(image,name) in enumerate(zip(images,['Cycles']+[r['mode'] for r in chosen])):
                axes[0,column].imshow(image);axes[1,column].imshow(image[cy:cy+128,cx:cx+128]);axes[0,column].set_title(name,fontsize=8)
                axes[0,column].axis('off');axes[1,column].axis('off')
            fig.suptitle(case['id']+' '+kind+'; same GT-depth-band crop');fig.tight_layout();path=root/(case['id']+'-'+kind+'.png');fig.savefig(path,dpi=110);plt.close(fig);plots.append(str(path))
    write(RECORDS/(PREFIX+'-residual-evaluation.json'),dict(completed=True,metrics=metrics,pooled=pooled,decisions=decisions,profiles=run['profiles'],visualizations=plots,
        random_score_is_expected_single_K16_draw=True,not_an_optical_ground_truth=True,GT_geometry_setting=True))
    import csv
    with (RECORDS/(PREFIX+'-residual-evaluation.csv')).open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]),lineterminator='\n');writer.writeheader();writer.writerows(flat)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['render','evaluate'],required=True);a=p.parse_args();c=json.loads(CRITERIA.read_text());assert c['frozen']
    torch.set_num_threads(8);cv2.setNumThreads(1)
    if a.mode=='render':
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise ValueError('Explicit GPU0')
        subprocess.run(['nvidia-smi'],check=True)
    globals()[a.mode](c)


if __name__=='__main__':main()
