"""Frozen SHARP/DA3/RAFT inference and E1 calibration for two static pilots.

No network is trained. Optional target-only color anchoring changes output GS
attributes exactly as S1b/S1c, never checkpoint weights.
"""
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
from scipy.ndimage import distance_transform_edt

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT.parent/'ml-sharp-s1-20261111/src'),str(ROOT.parent/'Depth-Anything-3/src')]
from route3_pilot_fields import OUT,RECORDS,PREFIX,read,write
from route3_sharp_s1 import raster
from route3_upper_bound_da3 import sim3
from src.aperture_renderer import GaussianScene
from src.dense_flow_geometry import pair_depth,multiview_depth,stratified_points,sample
from route3_sharp_s1c import support_stats
from route3_sharp_s1b_calibrate import fit_inverse,inverse_stats,bilinear
from route3_upper_bound_evaluate import display
FRAMES=[1,6,11,16,21,26,30,35,40,45,50,55,60]
K=np.array([[512*50/36,0,256],[0,512*50/36,256],[0,0,1]],np.float32)


def inference(folder):
    from depth_anything_3.api import DepthAnything3
    root=folder/'models';root.mkdir(exist_ok=True);rows=[]
    if not (root/'DA3.json').exists():
        net=DepthAnything3.from_pretrained(str(ROOT/'models/DA3-BASE')).to('cuda').eval();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.monotonic()
        pred=net.inference([str(folder/'inputs'/f'f{f:03d}.png') for f in FRAMES],process_res=518,process_res_method='upper_bound_resize',export_dir=None)
        torch.cuda.synchronize();seconds=time.monotonic()-t;peak=torch.cuda.max_memory_allocated()/2**20
        poses=np.broadcast_to(np.eye(4),(13,4,4)).copy();poses[:,:3]=pred.extrinsics[:,:3];poses=np.linalg.inv(poses);gt=np.load(folder/'scene-geometry.npz')['camera_c2w_opencv'][np.array(FRAMES)-1]
        scale,R,trans,singular=sim3(poses[:,:3,3],gt[:,:3,3]);aligned=poses.copy();aligned[:,:3,:3]=R@poses[:,:3,:3];aligned[:,:3,3]=scale*(poses[:,:3,3]@R.T)+trans
        rotation=np.swapaxes(gt[:,:3,:3],-1,-2)@aligned[:,:3,:3];rot=np.rad2deg(np.arccos(np.clip((np.trace(rotation,axis1=1,axis2=2)-1)/2,-1,1)))
        dest=root/'DA3';dest.mkdir(exist_ok=True)
        for i,f in enumerate(FRAMES):np.savez_compressed(dest/f'f{f:03d}.npz',depth=cv2.resize(pred.depth[i].astype(np.float32),(512,512))*scale,K=K,c2w=aligned[i])
        np.savez_compressed(dest/'native.npz',depth=pred.depth,extrinsics=pred.extrinsics,intrinsics=pred.intrinsics)
        write(root/'DA3.json',dict(completed=True,seconds=seconds,peak_allocated_MiB=peak,license='Apache-2.0',GT_conditioned_inference=False,global_sim3=dict(scale=scale,R=R.tolist(),translation=trans.tolist(),translation_RMSE_m=float(np.sqrt(np.mean(np.sum((aligned[:,:3,3]-gt[:,:3,3])**2,axis=-1)))),rotation_deg=rot.tolist()),per_frame_GT_depth_fit=False));del net,pred;torch.cuda.empty_cache()
    if (root/'SHARP.json').exists():return
    from sharp.models import create_predictor,PredictorParams
    from sharp.cli.predict import predict_image
    net=create_predictor(PredictorParams());net.load_state_dict(torch.load(ROOT/'models/SHARP/sharp_2572gikvuh.pt',map_location='cpu',weights_only=True),strict=True);net.eval().cuda();dest=root/'SHARP';dest.mkdir(exist_ok=True)
    for f in FRAMES:
        raw=read(folder/'inputs'/f'f{f:03d}.exr')[...,:3].clip(0,1);rgb=display(raw);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.monotonic()
        cloud=predict_image(net,rgb*255,float(K[0,0]),torch.device('cuda'));torch.cuda.synchronize();elapsed=time.monotonic()-start;n=cloud.mean_vectors.shape[1];half=n//2
        values=dict(means=cloud.mean_vectors[0],quats=cloud.quaternions[0],scales=cloud.singular_values[0],opacities=cloud.opacities[0].reshape(-1),colors=cloud.colors[0],source_frames=torch.arange(2,device='cuda').repeat_interleave(half))
        front=GaussianScene(**{k:v[:half] for k,v in values.items()})
        with torch.inference_mode():_,z,a=raster(front,torch.tensor(K,device='cuda'))
        z=z.cpu().numpy();a=a.cpu().numpy();valid=np.isfinite(z)&(z>0)&(a>=.1)
        if not valid.all():index=distance_transform_edt(~valid,return_distances=False,return_indices=True);z[~valid]=z[tuple(index[:,~valid])]
        np.savez_compressed(dest/f'f{f:03d}.npz',depth=z,alpha=a,valid=valid,K=K,rgb_linear=raw,rgb_srgb=rgb)
        if f in [1,30,60]:torch.save({k:v.detach().cpu() for k,v in values.items()},dest/f'f{f:03d}.pt')
        rows.append(dict(frame=f,seconds=elapsed,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,gaussians=n));print('PILOT_SHARP',folder.name,f,elapsed,flush=True)
        del cloud,values,front;torch.cuda.empty_cache()
    write(root/'SHARP.json',dict(completed=True,rows=rows,code_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT.parent/'ml-sharp-s1-20261111',text=True).strip(),license='Apple research-only;see S1 license record',network_updated=False));del net;torch.cuda.empty_cache()


def calibrate(folder):
    root=folder/'models';dest=root/'calibration';dest.mkdir(exist_ok=True)
    if (dest/'summary.json').exists() and json.loads((dest/'summary.json').read_text()).get('completed'):return
    gtposes=np.load(folder/'scene-geometry.npz')['camera_c2w_opencv'];poses={f:gtposes[f-1] for f in FRAMES};plan={};pairs=set()
    for f in FRAMES:
        # Match S1c largest-baseline policy, with six valid input neighbors.
        neighbors=sorted([g for g in FRAMES if g!=f],key=lambda g:(-np.linalg.norm(poses[g][:3,3]-poses[f][:3,3]),g))[:6];plan[f]=neighbors
        for g in neighbors:pairs.add((f,g));pairs.add((g,f))
    write(dest/'flow-plan.json',dict(selected=plan,directed=sorted(pairs),GT_depth_or_masks=False))
    from torchvision.models.optical_flow import raft_large,Raft_Large_Weights
    net=raft_large(weights=None,progress=False).eval().cuda();net.load_state_dict(torch.load(ROOT/'models/metrics/hub/checkpoints/raft_large_C_T_SKHT_V2-ff5fadd5.pth',map_location='cpu',weights_only=True));transform=Raft_Large_Weights.C_T_SKHT_V2.transforms();flow=dest/'flow';flow.mkdir(exist_ok=True)
    images={f:torch.tensor(cv2.imread(str(folder/'inputs'/f'f{f:03d}.png'))[...,::-1].copy(),device='cuda').permute(2,0,1)[None] for f in FRAMES};timings=[]
    for f,g in sorted(pairs):
        path=flow/f'f{f:03d}-f{g:03d}.npz'
        if path.exists():continue
        a,b=transform(images[f],images[g]);torch.cuda.synchronize();t=time.monotonic()
        with torch.inference_mode():v=net(a,b,num_flow_updates=32)[-1][0].permute(1,2,0)
        torch.cuda.synchronize();timings.append(time.monotonic()-t);np.savez_compressed(path,flow=v.cpu().numpy());print('PILOT_FLOW',folder.name,f,g,flush=True)
    del net,images;torch.cuda.empty_cache();rows=[]
    for f in FRAMES:
        references=[]
        for g in plan[f]:
            v=np.load(flow/f'f{f:03d}-f{g:03d}.npz')['flow'];back=np.load(flow/f'f{g:03d}-f{f:03d}.npz')['flow'];z,uv,stats=pair_depth(v,back,K,poses[f],poses[g]);references.append(dict(frame=g,depth=z,other_xy=uv))
        z,valid,support,count=multiview_depth(references,K,poses[f],poses);xy,tri,held=stratified_points(z,valid,f,0);stats=support_stats(xy,tri,held)
        np.savez_compressed(dest/f'f{f:03d}-reference.npz',depth=z,valid=valid,support=support,xy=xy,triangulated_depth=tri,heldout=held)
        result=dict(frame=f,support=stats,models={});rows.append(result)
        if not stats['supported']:
            write(dest/'summary.json',dict(completed=False,rows=rows,reason='S1c support insufficient;no threshold relaxation'));raise RuntimeError('PILOT_CALIBRATION_SUPPORT '+folder.name+str(f))
        for name in ['SHARP','DA3']:
            native=np.load(root/name/f'f{f:03d}.npz')['depth'];observed=bilinear(native,xy);fit=fit_inverse(observed[~held],tri[~held],native);before=inverse_stats(observed[held],tri[held]);after=inverse_stats(observed[held],tri[held],fit['a'],fit['b']);passed=after['inverse_RMSE']<=before['inverse_RMSE']+1e-10 and after['sparse_depth_AbsRel']<=before['sparse_depth_AbsRel']+1e-10
            attempted=dict(**fit,heldout_before=before,heldout_after=after)
            if not passed:
                # Development data frontend: reject a harmful input-only fit.
                # Identity is an explicit uncalibrated fallback, never called
                # successful affine correction, and never selected using GT.
                fit=dict(a=1.,b=0.,success=True,identity_fallback=True)
                after=before.copy()
            result['models'][name]=dict(**fit,attempted_affine_fit=attempted,calibration_applied=passed,heldout_before=before,heldout_after=after,heldout_not_worse=True,fit_rejected_and_native_depth_retained=not passed)
            target=dest/name;target.mkdir(exist_ok=True);np.savez_compressed(target/f'f{f:03d}.npz',depth=(1/np.maximum(fit['a']/native+fit['b'],1e-4)).astype(np.float32),K=K,c2w=poses[f])
        # Input-only confidence fields, retained for training interface.
        errors=[];cycles=[];angles=[];yy,xx=np.mgrid[:512,:512];world=(np.dstack([xx+.5,yy+.5,np.ones((512,512))])@np.linalg.inv(K).T*z[...,None])@poses[f][:3,:3].T+poses[f][:3,3]
        for ref in references:
            g=ref['frame'];v=np.load(flow/f'f{f:03d}-f{g:03d}.npz')['flow'];back=np.load(flow/f'f{g:03d}-f{f:03d}.npz')['flow'];uv=ref['other_xy'];fb=np.linalg.norm(v+sample(back,uv),axis=-1);cam=(world-poses[g][:3,3])@poses[g][:3,:3];q=cam@K.T;err=np.linalg.norm(q[...,:2]/np.maximum(q[...,2:],1e-8)-(uv+.5),axis=-1);a=world-poses[f][:3,3];b=world-poses[g][:3,3];theta=np.arccos(np.clip((a*b).sum(-1)/np.maximum(np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1),1e-8),-1,1));ok=valid&(fb<=1)&(err<=1);errors.append(np.where(ok,err,np.nan));cycles.append(np.where(ok,fb,np.nan));angles.append(np.where(ok,theta,np.nan))
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning);re=np.nanmedian(errors,axis=0);fb=np.nanmedian(cycles,axis=0);angle=np.nanmedian(angles,axis=0)
        good=valid&np.isfinite(re+fb+angle);confidence=np.where(good,np.exp(-.5*re**2-.5*fb**2)*np.minimum(1,np.sin(angle)/np.sin(np.deg2rad(2))),.05).astype(np.float32)
        np.savez_compressed(dest/f'f{f:03d}-confidence.npz',confidence=confidence,reprojection=re,FB=fb,angle_radians=angle,supported=good)
        write(dest/'progress.json',dict(rows=rows));print('PILOT_CALIBRATION',folder.name,f,stats['points'],flush=True)
    write(dest/'summary.json',dict(completed=True,rows=rows,flow_seconds=sum(timings),flow_pairs=len(pairs),E1_GT_poses=True,GT_depth_or_masks_used=False))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scene',required=True);p.add_argument('--mode',choices=['inference','calibrate'],required=True);a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise RuntimeError('Explicit GPU0 required')
    subprocess.run(['nvidia-smi'],check=True);torch.set_num_threads(8);cv2.setNumThreads(1)
    globals()[a.mode](OUT/a.scene)


if __name__=='__main__':main()
