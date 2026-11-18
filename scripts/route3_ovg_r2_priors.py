"""Precompute one float16 DepthSplat stack per ready RGB window."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
import torch

from route3_depthsplat_depth import network
from route3_ovg_r2_common import CONFIG, OUT, ROOT, guard
from route3_ovg_subpixel import save_json, save_npz

MODEL='depthsplat-depth-small-352x640-randview2-8-e807bd82.pth'


def resize_inputs(path, rows):
    dest=path.parent/'input-resized'
    dest.mkdir(exist_ok=True)
    images,matrices,pose=[],[],[]
    for i,row in enumerate(rows):
        src=ROOT/row['image']
        image=cv2.imread(str(src))
        if image is None:
            raise ValueError(f'Missing image in complete window {src}')
        k=np.array(row['K'],np.float32)
        k[:2]*=np.array([256/image.shape[1],256/image.shape[0]])[:,None]
        p=dest/f'f{i:03d}.png'
        if not p.exists():
            image=cv2.resize(image,(256,256),interpolation=cv2.INTER_AREA)
            if not cv2.imwrite(str(p),image):
                raise OSError(f'Could not encode stage 1a input {p}')
        else:
            image=cv2.imread(str(p))
        images.append(image[...,::-1].copy())
        matrices.append(k)
        pose.append(row['c2w'])
    return np.stack(images),np.stack(matrices),np.asarray(pose,np.float32)


@torch.inference_mode()
def stack_depth(net, images,K,pose):
    N,H,W=images.shape[:3]
    rgb=torch.tensor(images,device='cuda').permute(0,3,1,2).float()[None]/255.
    intr=torch.tensor(K/np.array([W,H,1.],np.float32)[None,:,None],device='cuda')[None]
    P=torch.tensor(pose,device='cuda')[None]
    neighbors=torch.cdist(P[:,:,:3,3],P[:,:,:3,3]).argsort(-1)[:,:,:3]
    out=net(rgb,attn_splits_list=[2],intrinsics=intr,min_depth=torch.full((1,N),1/200.,device='cuda'),
            max_depth=torch.full((1,N),1/.5,device='cuda'),extrinsics=P,nn_matrix=neighbors)
    depth=out['depth_preds'][-1][0].cpu().numpy()
    if depth.shape!=(N,H,W) or not (np.isfinite(depth)&(depth>0)).all():
        raise ValueError(f'Unexpected DepthSplat window output {depth.shape}')
    return depth


def support_map(depth, K, poses, i):
    h,w=depth.shape[-2:]
    y,x=np.mgrid[:h,:w]
    cam=(np.dstack([x,y,np.ones_like(x)])@np.linalg.inv(K[i]).T)*depth[i,...,None]
    world=cam@poses[i,:3,:3].T+poses[i,:3,3]
    support=np.zeros((h,w),np.uint8)
    for j in range(len(poses)):
        if j==i:
            continue
        v=(world-poses[j,:3,3])@poses[j,:3,:3]
        q=v@K[j].T
        uv=(q[...,:2]/np.maximum(q[...,2:],1e-8)).astype(np.float32)
        proj=cv2.remap(depth[j],uv[...,0],uv[...,1],cv2.INTER_LINEAR)
        support+=((v[...,2]>0)&(proj>0)&(abs(proj-v[...,2])<=.03*v[...,2])).astype(np.uint8)
    return support


def infer(net,path):
    if (path.parent/'prior/summary.json').exists():
        return
    meta=json.loads(path.read_text())
    t0=time.monotonic()
    imgs,K,P=resize_inputs(path,meta['rows'])
    t1=time.monotonic()
    depth=stack_depth(net,imgs,K,P)
    inference_s=time.monotonic()-t1
    total=0
    reports=[]
    for i in range(len(imgs)):
        supported=support_map(depth,K,P,i)
        dst=path.parent/'prior'/f'f{i:03d}.npz'
        if not dst.exists():
            save_npz(dst,depth=depth[i].astype(np.float16),consistent=supported>=2,support=supported)
        total+=dst.stat().st_size
        reports.append(float((supported>=2).mean()))
    save_json(path.parent/'prior/summary.json',dict(window=meta['scene'],tier=meta['tier'],frames=len(imgs),
                bytes=total,seconds=time.monotonic()-t0,GPU_forward_seconds=inference_s,
                mean_supported_fraction=float(np.mean(reports)),checkpoint=MODEL,dtype='float16',
                given_poses=True,GT_depth_input=False,model_one_full_context_forward=True))
    print('PRIOR_READY',meta['tier'],meta['scene'],len(imgs),flush=True)


def main(watch):
    guard()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
    subprocess.run(['nvidia-smi'],check=True)
    if torch.cuda.mem_get_info()[0]<24*2**30:
        raise RuntimeError('Concurrent depth-precompute memory budget unavailable')
    torch.cuda.set_per_process_memory_fraction(.20)
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    net,_=network(ROOT.parent/'depthsplat-r13-20261117',ROOT.parent/'dinov2-depthsplat-r13',ROOT/'models/DepthSplat'/MODEL)
    net=net.cuda().eval()
    failed=set()
    while True:
        guard()
        remaining=[p for tier in ('G','C') for p in sorted((OUT/tier/'windows').glob('*/manifest.json'))
                   if not p.parent.is_symlink() and not (p.parent/'prior/summary.json').exists() and p not in failed]
        if remaining:
            for p in remaining:
                try:
                    infer(net,p)
                except (OSError,ValueError,RuntimeError) as exc:
                    failed.add(p)
                    save_json(p.parent/'prior-error.json',dict(error=str(exc),scene=p.parent.name))
                    print('PRIOR_ERROR',p.parent.name,type(exc).__name__,str(exc),flush=True)
            continue
        g=OUT/'G/acquisition.json'
        gdone=g.exists() and len(json.loads(g.read_text())['environments'])==len(CONFIG['G_round2']['environments'])
        if not watch or (gdone and (OUT/'C/generation.json').exists()):
            break
        time.sleep(4)
    save_json(OUT/'prior-worker.json',dict(done=True,errors=[str(p.relative_to(ROOT)) for p in sorted(failed)]))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--watch',action='store_true')
    main(p.parse_args().watch)
