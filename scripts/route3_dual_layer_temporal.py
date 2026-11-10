"""Prepare/render/evaluate the preregistered eight-frame O5 temporal diagnosis."""
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
from src.c1_regions import display_rgb,linear_rgb
from src.layered_aperture import pixel_layer
from src.analytic_aperture import analytic,sampled
from route3_dual_layer_l1 import MODES,display


def prepare(a,c):
    a.output.mkdir(parents=True,exist_ok=False);items=[];fills=[];inputs=a.output/'inputs';inputs.mkdir()
    rows=json.loads((a.base/'inputs/B2-high/pinhole/timings.json').read_text());by={r['frame']:r for r in rows}
    gt=json.loads((a.base/'manifests/oracle.json').read_text())['items'];old30=next(x for x in gt if x['scene']=='B2-high' and x['frame_id']==30 and x['fstop']==1.4 and x['focus_distance_m']==5.)
    for frame in c['scope']['temporal_frames']:
        if frame==30:
            source=old30['input_npz'];truth=old30['gt_rgb_exr'];fill=str(a.base/'methods-2d/fill/B2-high-f030/fill-image.npz')
        else:
            geom=np.load(a.base/'B2-high/geometry'/f'f{frame:03d}-center.npz');bg=np.load(a.base/'B2-high/geometry'/f'f{frame:03d}-background.npz')
            color=a.base/'inputs/B2-high/pinhole'/(by[frame]['key']+'.exr');raw=cv2.imread(str(color),cv2.IMREAD_UNCHANGED)[...,:3][...,::-1];rgb=display_rgb(raw).astype(np.float32)
            mask=cv2.dilate((geom['foreground_alpha']>.1).astype(np.uint8),np.ones((5,5),np.uint8),iterations=2)>0
            source=str(inputs/f'f{frame:03d}.npz');np.savez_compressed(source,rgb_srgb=rgb,visible_rgb_linear=linear_rgb(rgb),center_depth=np.where(geom['center_visible'],geom['depth_axial_m'],10000.).astype(np.float32),
                center_surface_valid=geom['center_visible'],K=geom['K'],camera_c2w=geom['camera_c2w'],background_depth=bg['depth_axial_m'],fill_mask=mask)
            truth=str(a.output/'Cycles'/f'f{frame:03d}-N1.4-F5-s1024-seed20261110.exr');fill=str(a.output/'fill'/f'B2-high-f{frame:03d}'/'fill.npz')
            fills.append(dict(id=f'B2-high-f{frame:03d}',input_npz=source,uses_GT_segmentation=True,uses_true_hidden_depth=True))
        items.append(dict(id=f'B2-high-f{frame:03d}-N1.4-F5',frame=frame,input_npz=source,gt_rgb_exr=truth,fill=fill))
    (a.output/'fill-manifest.json').write_text(json.dumps(dict(items=fills),indent=2)+'\n')
    jobs=[dict(scene='B2-high',blend=str((ROOT/'outputs/route3-a2b-2026-11-10-01/blender-b2/B2-high/B2-high.blend').resolve()),frames=[x['frame'] for x in items if x['frame']!=30],output=str((a.output/'Cycles').resolve()))]
    (a.output/'cycles-jobs.json').write_text(json.dumps(jobs,indent=2)+'\n')
    record=dict(items=items,criteria=str(a.criteria),frames=c['scope']['temporal_frames'],cloud='O5',focus_m=5.,fstop=1.4,
        fill='same B2 L1 GT foreground mask/true deep background convention; oracle temporal diagnosis, never L3 inputs',GT_frame30_reused=True,model_training=False)
    a.record.write_text(json.dumps(record,indent=2)+'\n')


def render(a,c):
    a.output.mkdir(parents=True,exist_ok=False);items=json.loads(a.input.read_text())['items'];rows=[]
    for x in items:
        d=np.load(x['input_npz']);C=torch.tensor(d['visible_rgb_linear'],device='cuda');Z=torch.tensor(d['center_depth'],device='cuda');K=torch.tensor(d['K'],device='cuda',dtype=torch.float32);cam=torch.tensor(d['camera_c2w'],device='cuda',dtype=torch.float32)
        pc=pixel_layer(C,Z,K,cam,torch.tensor(d['center_surface_valid'],device='cuda'));fill_data=np.load(x['fill']);fill=torch.tensor(fill_data['fill_rgb_linear'],device='cuda');bg=torch.tensor(d['background_depth'],device='cuda');E=torch.linalg.inv(cam)
        for mode in MODES:
            def run():
                if mode.startswith('analytic'):return analytic(pc,K,E,512,512,5.,.05/(2*1.4),int(mode[-1]),fill,bg)
                return sampled(pc,K,E,512,512,5.,.05/(2*1.4),64 if mode=='fixed64' else 16,fill,bg,random_seed=20261110+x['frame'] if mode=='random16' else None)
            with torch.inference_mode():
                warm=run();del warm;torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();times=[]
                for _ in range(c['efficiency']['timed_repetitions']):
                    start=time.monotonic();out=run();torch.cuda.synchronize();times.append(time.monotonic()-start)
                pred=display(out['rgb_linear'].cpu().numpy())
            dest=a.output/x['id']/mode;dest.mkdir(parents=True);np.savez_compressed(dest/'prediction.npz',prediction_srgb=pred)
            rows.append(dict(id=x['id'],frame=x['frame'],mode=mode,prediction=str(dest/'prediction.npz'),gt_rgb_exr=x['gt_rgb_exr'],median_seconds=float(np.median(times)),seconds=times,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20))
            print('TEMPORAL_RENDER',x['frame'],mode,flush=True)
        del pc;torch.cuda.empty_cache()
    a.record.write_text(json.dumps(dict(rows=rows,completed=True,cloud='O5',criteria=str(a.criteria)),indent=2)+'\n')


def evaluate(a,c):
    rows=json.loads(a.input.read_text())['rows'];results={};gt=[]
    for frame in c['scope']['temporal_frames']:
        r=next(r for r in rows if r['frame']==frame);gt.append(display(cv2.imread(r['gt_rgb_exr'],cv2.IMREAD_UNCHANGED)[...,:3][...,::-1]))
    gt=np.stack(gt);difference=np.diff(gt,axis=0)
    for mode in MODES:
        pred=np.stack([np.load(next(r['prediction'] for r in rows if r['frame']==frame and r['mode']==mode))['prediction_srgb'] for frame in c['scope']['temporal_frames']])
        error=np.diff(pred,axis=0)-difference
        results[mode]=dict(temporal_difference_error_RMSE=float(np.sqrt(np.mean(error.astype(np.float64)**2))),per_transition_RMSE=np.sqrt(np.mean(error.astype(np.float64)**2,axis=(1,2,3))).tolist(),
            Cycles_full_image_PSNR=float(-10*np.log10(np.mean((pred.astype(np.float64)-gt)**2))),median_render_seconds=float(np.median([r['median_seconds'] for r in rows if r['mode']==mode])))
    for mode in ['analytic1','analytic4']:
        results[mode]['ratio_to_fixed64']=results[mode]['temporal_difference_error_RMSE']/results['fixed64']['temporal_difference_error_RMSE']
        results[mode]['ratio_to_fixed16']=results[mode]['temporal_difference_error_RMSE']/results['fixed16']['temporal_difference_error_RMSE']
        results[mode]['temporal_NI_passed']=all(results[mode]['temporal_difference_error_RMSE']<=results[b]['temporal_difference_error_RMSE']*c['temporal']['relative_noninferiority_max'] for b in ['fixed16','fixed64'])
    a.record.write_text(json.dumps(dict(completed=True,frames=c['scope']['temporal_frames'],results=results,cloud='O5',motion_aligned=False,flicker_claim=False,
        definition='unaligned displayRGB frame-difference minus independent Cycles frame-difference; includes geometry,appearance,motion and integration errors',criteria=str(a.criteria)),indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['prepare','render','evaluate'],required=True);p.add_argument('--base',type=Path,default=ROOT/'outputs/route3-b2only-2026-11-10-01')
    p.add_argument('--criteria',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--record',type=Path,required=True);p.add_argument('--input',type=Path);a=p.parse_args()
    c=json.loads(a.criteria.read_text());assert c['frozen'];torch.set_num_threads(4);cv2.setNumThreads(1)
    if a.record.exists():raise FileExistsError(a.record)
    if a.mode=='render':
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise ValueError('Explicit GPU0 required')
        subprocess.run(['nvidia-smi'],check=True);render(a,c)
    elif a.mode=='prepare':prepare(a,c)
    else:evaluate(a,c)


if __name__=='__main__':main()
