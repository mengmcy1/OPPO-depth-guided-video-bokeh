"""Input-video-supervised U1 proxy; aperture views are monitor-only.

The primary fit uses GT poses/depth, dynamic IDs and fixed GT-rec hidden
Gaussians. No defocused target, scoring mask or evaluation aperture position
is loaded by prepare/fit. Historical R2 checkpoints/results stay untouched.
"""
import argparse
import datetime
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
import torch.nn.functional as F
from gsplat import rasterization
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from src.aperture_renderer import GaussianScene,linear_to_srgb
from src.layered_aperture import merge_layers
from src.upper_bound_optimization import FittedVisible,ssim
from route3_upper_bound_u1 import inputs,hidden_selection,translated_fill,pinhole
from route3_b2_oracle_3d import sparse_pixel_layer
from route3_upper_bound_evaluate import BASE,OUT as UPPER,RECORDS,exr,display
OUT=ROOT/'outputs/route3-u1video-2026-11-11-01'
PREFIX='route3-u1video-2026-11-11'
CRITERIA=RECORDS/'route3-u1video-criteria-2026-11-11.json'
FRAMES=[1,6,11,16,21,26,30,35,40,45,50,55,60]


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def rgb_depth(pc,K,E,background,colors=None):
    image,alpha,_=rasterization(pc.means,pc.quats,pc.scales,pc.opacities,
        pc.colors if colors is None else colors,E[None],K[None],512,512,packed=False,
        render_mode='RGB+ED',rasterize_mode='antialiased',near_plane=1e-4,far_plane=1e5,eps2d=.0015)
    pred=linear_to_srgb(image[0,...,:3]+(1-alpha[0])*background).clamp(0,1)
    return pred,image[0,...,3].detach(),alpha[0,...,0].detach(),image[0,...,0].detach()


def reproject_fill(fill,bg,K,target_cam,view_cam):
    """Same local-depth ray/plane C_fill lookup, generalized to input poses."""
    y,x=torch.meshgrid(torch.arange(512,device='cuda'),torch.arange(512,device='cuda'),indexing='ij')
    ray=torch.stack([x+.5,y+.5,torch.ones_like(x)],-1).float()@torch.linalg.inv(K).T
    relative=torch.linalg.inv(target_cam)@view_cam
    ray=ray@relative[:3,:3].T;origin=relative[:3,3]
    distance=(bg-origin[2])/ray[...,2].clamp_min(1e-6)
    point=origin+distance[...,None]*ray
    uv=point@K.T;uv=uv[...,:2]/uv[...,2:].clamp_min(1e-6)
    return F.grid_sample(fill.permute(2,0,1)[None],(2*uv/512-1)[None],align_corners=False,padding_mode='border')[0].permute(1,2,0)


def consistent(depth,alpha,truth):
    return (torch.isfinite(depth)&(truth>0)&(alpha>=.95)&
        ((depth-truth).abs()<=torch.maximum(.02*truth,torch.full_like(truth,.01))))


def windows(mask):
    return F.avg_pool2d(mask[None,None].float(),11,1)[0,0]>=1


def valid_supervision(mask):
    count=int(mask.sum());nwindow=int(windows(mask).sum())
    return count>=26215 and nwindow/(502**2)>=.05,count,nwindow


def masked_loss(pred,target,mask=None):
    if mask is None:return .8*(pred-target).abs().mean()+.2*(1-ssim(pred,target))
    support=windows(mask);local=ssim(pred,target,return_map=True)[0]
    return .8*(pred-target).abs()[mask].mean()+.2*(1-local[:,support].mean())


def prepare(c):
    dest=OUT/'supervision';dest.mkdir(parents=True,exist_ok=False);rows=[];baseline=[];stop=None
    pin=BASE/'inputs/B2-high/pinhole';pinrows={r['frame']:r for r in json.loads((pin/'timings.json').read_text())}
    for frame in [1,30,60]:
        started=time.monotonic();d,K,cam,visible,parent,fill,bg,profile=inputs(frame)
        chosen,_=hidden_selection(d,parent);hidden=sparse_pixel_layer(parent,chosen,K,cam,profile,'cuda');pc=merge_layers(visible,hidden)
        targetgeom=np.load(BASE/'B2-high/geometry'/f'f{frame:03d}-center.npz')
        dynamic=np.isin(targetgeom['object_id'],[7]).repeat(5,0).repeat(5,1).reshape(-1)
        dynamic=np.concatenate([dynamic,np.repeat(np.isin(parent['object_id'][chosen],[7]),int(profile['subdivisions'])**2)])
        assert len(dynamic)==len(pc.means)
        labels=torch.tensor(dynamic,device='cuda',dtype=torch.float32)[:,None].expand(-1,3).contiguous()
        folder=dest/f'f{frame:03d}';folder.mkdir();views=[]
        for source in FRAMES:
            geometry=BASE/'B2-high/geometry'/f'f{source:03d}-center.npz';g=np.load(geometry)
            sourcecam=torch.tensor(g['camera_c2w'],device='cuda',dtype=torch.float32);E=torch.linalg.inv(sourcecam)
            background=fill if source==frame else reproject_fill(fill,bg,K,cam,sourcecam)
            rgbpath=pin/(pinrows[source]['key']+'.exr');rgb=display(np.clip(exr(rgbpath),0,1))
            if source==frame:
                mask=np.ones((512,512),bool);count=512**2;nwindow=502**2;valid=True
                assert np.allclose(rgb,d['rgb_srgb'],atol=1e-6)
                with torch.inference_mode():
                    legacy=pinhole(pc,K,E,fill);extended,_,_,_=rgb_depth(pc,K,E,fill)
                    render_difference=float((legacy-extended).abs().max())
                    assert render_difference<2e-6,render_difference
                coverage=dict(target=True,retained_fraction=1.,retained_pixels=count,valid_SSIM_windows=nwindow)
                coverage['RGB_plus_expected_depth_vs_R2_RGB_max_difference']=render_difference
            else:
                with torch.inference_mode():
                    _,depth,alpha,contribution=rgb_depth(pc,K,E,torch.zeros_like(background),labels)
                    own_dynamic=cv2.dilate(np.isin(g['object_id'],[7]).astype(np.uint8),np.ones((5,5),np.uint8))>0
                    projected_dynamic=cv2.dilate((contribution.cpu().numpy()>.01).astype(np.uint8),np.ones((5,5),np.uint8))>0
                    excluded=own_dynamic|projected_dynamic
                    good=consistent(depth,alpha,torch.tensor(g['depth_axial_m'],device='cuda'))&~torch.tensor(excluded,device='cuda')
                    valid,count,nwindow=valid_supervision(good);mask=good.cpu().numpy()
                    coverage=dict(target=False,retained_fraction=count/512**2,retained_pixels=count,valid_SSIM_windows=nwindow,
                        own_dynamic_excluded_pixels=int(own_dynamic.sum()),projected_target_dynamic_excluded_pixels=int(projected_dynamic.sum()),
                        own_GT_surface_pixels=int((g['depth_axial_m']>0).sum()),alpha_covered_pixels=int((alpha>=.95).sum()),valid=valid)
                if not valid:stop=f'f{frame:03d} source{source}: usable supervision below frozen threshold'
            path=folder/f'source{source:03d}.npz'
            np.savez_compressed(path,rgb_srgb=rgb,initial_mask=mask,GT_depth=g['depth_axial_m'],E=E.cpu().numpy(),background=background.cpu().numpy())
            relative=np.linalg.inv(d['camera_c2w'])@g['camera_c2w'];offset=relative[:3,3];lateral=float(np.linalg.norm(offset[:2]));distance=float(np.linalg.norm(offset))
            view=dict(frame=source,target=source==frame,path=str(path),RGB_source=str(rgbpath),GT_geometry=str(geometry),**{k:v for k,v in coverage.items() if k!='target'})
            views.append(view)
            if source!=frame:baseline.append(dict(target_frame=frame,source_frame=source,offset_in_target_camera_m=offset.tolist(),distance_m=distance,lateral_m=lateral,
                ratio_to_f1_4_radius=distance/(.05/(2*1.4)),ratio_to_f2_radius=distance/(.05/4),lateral_ratio_to_f1_4_radius=lateral/(.05/(2*1.4))))
            print('VIDEO_PREPARE',frame,source,count,valid,flush=True)
        rows.append(dict(frame=frame,views=views,seconds=time.monotonic()-started,hidden_parents=int(chosen.sum()),visible_gaussians=len(visible.means)))
        del pc,visible,hidden,labels;torch.cuda.empty_cache()
        if stop:break
    result=dict(completed=stop is None,stop_reason=stop,rows=rows,baseline=baseline,GT_dynamic_ids=[7],
        loss_masks_use_GT_geometry_and_dynamic_IDs=True,aperture_images_or_positions_read=False,DOF_GT_or_evaluation_masks_read=False)
    write(OUT/'supervision.json',result);write(RECORDS/(PREFIX+'-supervision.json'),result)
    if stop:raise RuntimeError('U1_VIDEO_STOP '+stop)


def fit(frame,c):
    prep=json.loads((OUT/'supervision.json').read_text());assert prep['completed'];entry=next(r for r in prep['rows'] if r['frame']==frame)
    folder=OUT/'fit'/f'f{frame:03d}';folder.mkdir(parents=True,exist_ok=False)
    d,K,cam,initial,parent,fill,bg,profile=inputs(frame);chosen,_=hidden_selection(d,parent)
    hidden=sparse_pixel_layer(parent,chosen,K,cam,profile,'cuda');train=[]
    for row in entry['views']:
        cached=np.load(row['path']);train.append(dict(frame=row['frame'],target=row['target'],
            **{k:torch.tensor(cached[k],device='cuda') for k in ['rgb_srgb','initial_mask','GT_depth','E','background']}))
    held=[]
    for row in json.loads((UPPER/'U1-positions.json').read_text())['positions']:
        if row['frame']!=frame or row['role']!='U1-heldout':continue
        rgb=torch.tensor(display(exr(UPPER/'U1-supervision'/(row['key']+'.exr'))),device='cuda')
        E=torch.linalg.inv(cam);E[:2,3]-=torch.tensor(row['xy_m'],device='cuda')
        held.append(dict(rgb=rgb,E=E,background=translated_fill(fill,bg,K,row['xy_m'])))
    assert len(train)==13 and len(held)==16
    z=torch.tensor(d['center_depth'],device='cuda').repeat_interleave(5,0).repeat_interleave(5,1).reshape(-1)
    fitted=FittedVisible(initial,z);optimizer=fitted.optimizer();del initial,z
    probabilities=np.array([.5 if v['target'] else .5/12 for v in train]);rng=np.random.default_rng(2026111105+frame)
    config=dict(frame=frame,steps=8000,seed=2026111105+frame,code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        criteria=str(CRITERIA),learning_rates={g['name']:g['lr'] for g in optimizer.param_groups},visible_gaussians=len(fitted.means),hidden_parents=int(chosen.sum()),
        train_frames=FRAMES,train_probabilities=probabilities.tolist(),heldout_views=16,monitor_only_aperture_images=True,
        DOF_GT_read=False,evaluation_positions_read=False,evaluation_masks_read=False,GT_geometry_and_dynamic_masks_Oracle=True,
        fixed_hidden_GT_rec=True,heldout_selects_checkpoint=False,stopping='R2 historicalbest decline0.5dB for3 consecutive monitors, or nonfinite loss/parameters')
    write(folder/'config.json',config)
    counts={str(v['frame']):0 for v in train};min_counts={};rows=[];times=[];best=-float('inf');declines=0;stop=None;first_gradients={}
    started=time.monotonic();torch.cuda.reset_peak_memory_stats()

    def prediction(pc,v):
        pred,dep,alpha,_=rgb_depth(pc,K,v['E'],v['background'])
        mask=None if v['target'] else v['initial_mask']&consistent(dep,alpha,v['GT_depth'])
        if mask is not None:
            ok,n,nw=valid_supervision(mask)
            min_counts[str(v['frame'])]=min(min_counts.get(str(v['frame']),n),n)
            if not ok:raise RuntimeError(f'U1_VIDEO_STOP live supervision f{frame} source{v["frame"]}: {n} pixels, {nw} SSIM windows')
        return pred,mask

    for step in range(8001):
        if step%500==0 or step in [0,100]:
            with torch.inference_mode():
                if not all(bool(torch.isfinite(p).all()) for p in fitted.parameters()):stop='nonfinite optimization parameter';break
                pc=merge_layers(fitted.scene(),hidden);weighted_mse=0.;weighted_loss=0.;neighbor_mse=[];target_stats=None
                for v,weight in zip(train,probabilities):
                    pred,mask=prediction(pc,v);error=(pred-v['rgb_srgb']).square();mse=float(error.double().mean() if mask is None else error[mask].double().mean())
                    loss=float(masked_loss(pred,v['rgb_srgb'],mask));weighted_mse+=weight*mse;weighted_loss+=weight*loss
                    if v['target']:target_stats=dict(psnr=float(-10*np.log10(max(mse,1e-12))),loss=loss)
                    else:neighbor_mse.append(mse)
                held_sse=0.
                for v in held:
                    pred=pinhole(pc,K,v['E'],v['background']);held_sse+=float((pred-v['rgb']).square().double().sum())
                held_psnr=float(-10*np.log10(max(held_sse/(16*512*512*3),1e-12)));del pc
            if not np.isfinite(held_psnr+weighted_loss):stop='nonfinite monitor loss';break
            best=max(best,held_psnr);decline=best-held_psnr;declines=declines+1 if decline>=.5 else 0
            monitor=dict(train=dict(psnr=float(-10*np.log10(max(weighted_mse,1e-12))),loss=float(weighted_loss),views=13),
                target=target_stats,neighbors=dict(psnr=float(-10*np.log10(max(np.mean(neighbor_mse),1e-12))),views=12),heldout=dict(psnr=held_psnr,views=16))
            row=dict(step=step,seconds=time.monotonic()-started,metrics=monitor,train_minus_heldout_PSNR_dB=monitor['train']['psnr']-held_psnr,
                best_heldout_PSNR=best,heldout_decline_dB=decline,consecutive_declines=declines,minimum_live_neighbor_pixels=dict(min_counts),
                median_step_seconds=float(np.median(times[-100:])) if times else None,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20)
            rows.append(row);write(folder/'monitor.json',dict(completed=False,rows=rows,config=config));print('VIDEO_MONITOR',frame,json.dumps(row),flush=True)
            if declines>=3:stop='heldout PSNR declined>=0.5dB from historical best for3 consecutive monitors';break
            if step>0 and step%1000==0:torch.save(dict(step=step,parameters=fitted.state_dict(),optimizer=optimizer.state_dict(),rng=rng.bit_generator.state),folder/'latest.pt')
        if step==8000:break
        factor=.01+.99*.5*(1+np.cos(np.pi*step/8000))
        for group in optimizer.param_groups:group['lr']=config['learning_rates'][group['name']]*factor
        v=train[int(rng.choice(13,p=probabilities))];counts[str(v['frame'])]+=1;optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize();begin=time.monotonic();pc=merge_layers(fitted.scene(),hidden);pred,mask=prediction(pc,v)
        loss=masked_loss(pred,v['rgb_srgb'],mask)
        if not bool(torch.isfinite(loss)):stop='nonfinite training loss';break
        loss.backward()
        if step==0:
            for name,param in fitted.named_parameters():
                if param.grad is None or not bool(torch.isfinite(param.grad).all()):raise RuntimeError('U1_VIDEO_STOP missing/nonfinite first gradient '+name)
                first_gradients[name]=dict(norm=float(param.grad.norm()),max_abs=float(param.grad.abs().max()),parameters=param.numel())
        optimizer.step();fitted.project_constraints();torch.cuda.synchronize();times.append(time.monotonic()-begin);del pc,pred,loss
    with torch.no_grad():
        scene=fitted.scene();torch.save({k:getattr(scene,k).detach().cpu() for k in scene.__dataclass_fields__},folder/'visible-final.pt')
    result=dict(completed=stop is None and step==8000,stop_reason=stop,config=config,rows=rows,wall_seconds=time.monotonic()-started,
        output=str(folder/'visible-final.pt'),selected_step=step,training_view_counts=counts,minimum_live_neighbor_pixels=min_counts,
        median_step_seconds=float(np.median(times)) if times else None,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
        learned_linear_color_max=float(scene.colors.max()),learned_Gaussian_fraction_any_color_above1=float((scene.colors>1).any(-1).float().mean()),first_backward_gradients=first_gradients)
    write(folder/'result.json',result);write(RECORDS/(PREFIX+f'-fit-f{frame:03d}.json'),result)
    if stop:raise RuntimeError('U1_VIDEO_STOP '+stop)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['prepare','fit'],required=True);p.add_argument('--frame',type=int,choices=[1,30,60]);a=p.parse_args()
    c=json.loads(CRITERIA.read_text());assert c['frozen'];cv2.setNumThreads(1);torch.set_num_threads(4)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise ValueError('Use explicit authorized GPU0')
    subprocess.run(['nvidia-smi'],check=True)
    try:
        if a.mode=='fit':fit(a.frame,c)
        else:prepare(c)
    except RuntimeError as exc:
        if 'U1_VIDEO_STOP' in str(exc):
            write(RECORDS/(PREFIX+'-stop.json'),dict(completed=False,stage=a.mode,frame=a.frame,reason=str(exc),
                stopped_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope_not_expanded=True))
        raise


if __name__=='__main__':main()
