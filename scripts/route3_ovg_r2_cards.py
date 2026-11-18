"""Extend analytic card clips and preserve texture-only thin-rod exclusions."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import os
import time

import cv2
import numpy as np

from route3_ovg_r2_common import CONFIG, OUT, REC, ROOT, ROUND1, guard
from route3_ovg_card import lens_offsets, load_texture, uv
from route3_ovg_subpixel import TRAIN_FRAMES, offsets, save_json, save_npz, two_layers

FAMILIES=['broad','narrow','perforated','deep_multilayer','shallow_multilayer','wide_parallax']


def setup(seed):
    rng=np.random.default_rng(seed)
    family=FAMILIES[(seed-CONFIG['C_round2']['first_seed'])%len(FAMILIES)]
    front=rng.uniform(1.15,1.95)
    rear=rng.uniform(3.8,7.6)
    mid=front+(rear-front)*rng.uniform(.24,.65)
    if family=='shallow_multilayer':
        mid=front+rng.uniform(.18,.36)
    elif family=='deep_multilayer':
        mid=min(rear-.3,front+rng.uniform(1.3,1.9))
    width=rng.uniform(.20,.60)
    if family in ('narrow','perforated'):
        width=rng.uniform(.035,.115) if family=='narrow' else rng.uniform(.46,.64)
    left=rng.uniform(-.28,-.06)
    y=rng.uniform(-.40,-.22)
    right=left+width
    top=y+rng.uniform(.42,.84)
    cards=np.array([[front,left,right,y,top],[mid,.04,rng.uniform(.35,.75),-.64,.55],[rear,-20.,20.,-20.,20.]])
    holes=[]
    if family=='perforated':
        for iy in range(4):
            for ix in range(3):
                x=left+width*(ix+.55)/3
                yy=y+(top-y)*(iy+.6)/4
                a,b=width*.075,(top-y)*rng.uniform(.045,.07)
                holes.append((0,x-a,x+a,yy-b,yy+b))
    base=rng.uniform(.115,.48) if family!='wide_parallax' else rng.uniform(.46,.78)
    count=CONFIG['C_round2']['frames_per_sequence']
    t=np.linspace(-1,1,count)
    direction=rng.choice([-1,1])
    origins=np.stack([direction*base*t/2,rng.uniform(.009,.060)*np.sin(t*np.pi),
                      rng.uniform(.012,.08)*np.sin(t*1.5)],-1)
    texture_transform=rng.uniform([.35,-.5,-.5],[1.9,.5,.5],(3,3))
    return family,cards,holes,origins,texture_transform


def cast(rays, origin, rectangles, holes):
    depth=(rectangles[:,0]-origin[2])/np.maximum(rays[...,2,None],1e-8)
    x=origin[0]+rays[...,0,None]*depth
    y=origin[1]+rays[...,1,None]*depth
    hits=(depth>0)&(x>=rectangles[:,1])&(x<rectangles[:,2])&(y>=rectangles[:,3])&(y<rectangles[:,4])
    for i,l,r,b,t in holes:
        hits[...,i]&=~((x[...,i]>=l)&(x[...,i]<r)&(y[...,i]>=b)&(y[...,i]<t))
    chosen=hits.argmax(-1)
    okay=hits.any(-1)
    def gather(a):
        return np.take_along_axis(a,chosen[...,None],-1)[...,0]
    z=np.where(okay,rectangles[chosen,0]-origin[2],0).astype(np.float32)
    points=np.stack([gather(x),gather(y),rectangles[chosen,0]],-1).astype(np.float32)
    return z,np.where(okay,chosen+1,0).astype(np.uint8),points,hits


def aperture_trace(dest, frame, ray, K, rect, holes, origins, focus):
    center=origins[frame]
    fields=[]
    for lens in lens_offsets():
        z,oid,points,_=cast(ray*focus-lens,center+lens,rect,holes)
        xy=uv(points,center,K)
        inside=(xy>=0).all(-1)&(xy<ray.shape[0]).all(-1)
        target,blocker,_,_=cast(points-center,center,rect,holes)
        hidden=inside&(oid>0)&(blocker>0)&(target+1e-4<z)
        ids=np.flatnonzero(hidden)
        sliced=points.reshape(-1,3)[ids]
        obj=oid.ravel()[ids]
        witness=np.zeros(oid.size,np.uint8)
        for j,p in enumerate(origins):
            if j==frame or not len(ids):continue
            xy_j=uv(sliced,p,K)
            view=(xy_j>=0).all(-1)&(xy_j<ray.shape[0]).all(-1)
            _,other,_,_=cast(sliced-p,p,rect,holes)
            witness[ids]|=((view&(other==obj)).astype(np.uint8)<<j)
        witness=witness.reshape(oid.shape)
        fields.append(dict(depth=z,surface_id=oid,world=points,light_valid=oid>0,center_in_view=inside,
                           hidden=hidden,visible_source_frames=witness,recoverable=witness>0,
                           never_seen=hidden&(witness==0)))
    save_npz(dest/'K16-aperture'/f'f{frame:02d}-N1.4-F{focus:g}.npz',
             **{k:np.stack([r[k] for r in fields]) for k in fields[0]})
    return dict(focus_m=focus,frame=frame,hidden=sum(int(d['hidden'].sum()) for d in fields),
                recoverable=sum(int(d['recoverable'].sum()) for d in fields),K=16)


def make_clip(seed,textures,assets):
    guard()
    name=f'card-r2-{seed}'
    dest=OUT/'C/windows'/name
    if (dest/'manifest.json').exists():
        return json.loads((dest/'manifest.json').read_text())
    start=time.monotonic()
    dest.mkdir(parents=True,exist_ok=True)
    family,rect,holes,origins,transform=setup(seed)
    size=CONFIG['C_round2']['size']
    K=np.array([[size*50/36,0,size/2],[0,size*50/36,size/2],[0,0,1]],dtype=np.float32)
    x=np.arange(size,dtype=np.float32)+.5
    pix=np.stack([*np.meshgrid(x,x),np.ones((size,size),np.float32)],-1)
    radius=(pix[...,None,:]+offsets(4).astype(np.float32))@np.linalg.inv(K).T
    ray=pix@np.linalg.inv(K).T
    order=np.random.default_rng(seed+193).permutation(3)
    rows=[]
    for frame,point in enumerate(origins):
        z,obj,pts,_=cast(radius,point,rect,holes)
        normals=np.zeros((*z.shape,3),np.float32)
        normals[...,2]=-(obj>0).astype(np.float32)
        label=two_layers(z,normals,obj)
        dual=label['valid_dual']
        if not label['valid'].all() or not np.all(label['depth2'][dual]>label['depth1'][dual]):
            raise ValueError(f'Bad geometric layer order {seed} {frame}')
        rgb=np.zeros((*z.shape,3),np.float32)
        for j,t in enumerate(order):
            tex=textures[t]
            mask=obj==j+1
            if not mask.any():
                continue
            coord=(pts[mask,:2]/transform[j,0]+transform[j,1:])%1
            xy=(coord*np.array([tex.shape[1]-1,tex.shape[0]-1])).astype(np.float32)
            rgb[mask]=np.concatenate([cv2.remap(tex,sub[:,None,0],sub[:,None,1],cv2.INTER_LINEAR)[:,0]
                                     for sub in np.array_split(xy,int(np.ceil(len(xy)/16_000)))])
        image=rgb.mean(-2)
        display=np.where(image<=.0031308,12.92*image,1.055*image**(1/2.4)-.055)
        out=dest/f'frame-{frame:02d}.png'
        if not out.exists() and not cv2.imwrite(str(out),np.rint(display.clip(0,1)[...,::-1]*255).astype(np.uint8)):
            raise OSError(out)
        p=dest/f'labels-{frame:02d}.npz'
        if not p.exists():
            save_npz(p,**label)
        observed,center_id,_,hits=cast(ray,point,rect,holes)
        np.put_along_axis(hits,(center_id.astype(np.int64)-1)[...,None],False,axis=-1)
        second=hits.argmax(-1)
        hidden=hits.any(-1)
        world=(point+ray*(rect[second,0]-point[2])[...,None]).astype(np.float32)
        save_npz(dest/'peeled'/f'f{frame:02d}.npz',hidden_valid=hidden,surface_id=np.where(hidden,second+1,0).astype(np.uint8),
                 depth=np.where(hidden,rect[second,0]-point[2],0).astype(np.float32),world=world)
        pose=np.eye(4);pose[:3,3]=point
        rows.append(dict(frame=frame,image=str(out.relative_to(ROOT)),label=str(p.relative_to(ROOT)),
                         c2w=pose.tolist(),K=K.tolist(),dual_pixels=int(dual.sum())))
    mid=len(origins)//2
    lens=[aperture_trace(dest,mid,ray,K,rect,holes,origins,f) for f in [float(rect[0,0]),float(rect[-1,0])]]
    if not any(r['hidden'] for r in lens):
        raise ValueError(f'Missing aperture hiding evidence {seed}')
    result=dict(tier='C',scene=name,asset_order=order.tolist(),assets=assets,seed=seed,family=family,rows=rows,
                camera_origins=origins.tolist(),baseline_m=float(np.ptp(origins[:,0])),foreground_width_m=float(rect[0,2]-rect[0,1]),
                plane_depths_m=rect[:,0].tolist(),rectangles=rect.tolist(),holes=holes,
                K16_aperture=lens,geometry_masks_full_sequence=True,K16_scope='single middle frame, two focuses',
                texture_map_per_plane=transform.tolist(),static_scene=True,seconds=time.monotonic()-start)
    save_json(dest/'manifest.json',result)
    print('C_SHARD_READY',name,family,len(rows),flush=True)
    return result


def cards():
    guard()
    config=CONFIG['C_round2']
    sealed=json.loads((REC/'route3-ovg-1a-2026-11-18-independent-test.json').read_text())['forbidden_train']
    original=next((ROUND1/'C/windows').glob('*/manifest.json'))
    meta=json.loads(original.read_text())
    assets=meta['assets']
    seeds=list(range(config['first_seed'],config['first_seed']+config['sequences']))
    if set(seeds)&set(sealed['seeds']) or {u['asset'] for u in assets}&set(sealed['asset_ids']):
        raise ValueError('C train data overlaps independent testing inventory')
    textures=[load_texture(u['asset'],ROUND1/'C/assets')[0] for u in assets]
    completed,errors=[],[]
    cv2.setNumThreads(1)
    with ThreadPoolExecutor(max_workers=4) as workers:
        wait={workers.submit(make_clip,s,textures,assets):s for s in seeds}
        for f in as_completed(wait):
            try:
                result=f.result()
                completed.append(dict(scene=result['scene'],family=result['family'],frames=len(result['rows']),
                                      seconds=result['seconds'],baseline_m=result['baseline_m'],
                                      foreground_width_m=result['foreground_width_m'],holes=len(result['holes'])))
            except (OSError,ValueError) as e:
                errors.append(dict(seed=wait[f],reason=str(e)))
                print('C_SHARD_ERROR',wait[f],str(e),flush=True)
    save_json(OUT/'C/generation.json',dict(completed=sorted(completed,key=lambda a:a['scene']),
              errors=errors,old_clips=1,license='CC0',with_camera_motion=True,distinct_seeds_and_layouts=True))


def thin_masks():
    guard()
    os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
    historical=ROOT/'outputs/route3-g3-r14-2026-11-17-01/truth'
    source=REC/'route3-ovg-1a-2026-11-18-thin-check.csv'
    offending={}
    with source.open() as stream:
        for row in csv.DictReader(stream):
            if row['interior_phase_pass']=='False':
                offending.setdefault(row['scene'], {})[int(row['object_id'])]=row['object']
    rows=[]
    shape=np.ones((3,3),np.uint8)
    for scene in sorted(historical.glob('*-thin')):
        ids=offending.get(scene.name,{})
        for frame in TRAIN_FRAMES:
            p=scene/'inputs'/f'f{frame:03d}-IndexOB-{frame:04d}.exr'
            raw=cv2.imread(str(p),cv2.IMREAD_UNCHANGED)
            if raw is None:
                raise FileNotFoundError(p)
            labels=np.rint(raw[...,0] if raw.ndim==3 else raw).astype(np.int16)
            reject=np.zeros_like(labels,bool)
            for oid in ids:
                reject|=cv2.erode((labels==oid).astype(np.uint8),shape)>0
            dest=OUT/'B-thin-texture'/scene.name/f'f{frame:03d}.npz'
            if not dest.exists():
                save_npz(dest,texture_supervision_valid=~reject,rejected_pure_interior=reject)
            rows.append(dict(scene=scene.name,frame=frame,pixels_blocked=int(reject.sum()),candidate_objects=list(ids.values())))
    save_json(OUT/'B-thin-texture/summary.json',dict(rows=rows,masked_pixel_observations=sum(r['pixels_blocked'] for r in rows),
              affected_objects=sum(len(x) for x in offending.values()),frames=len(rows),
              geometry_alpha_visibility_kept=True,no_split_change=True,
              consumer='13.5(6) masks only zero/LPIPS-context-independent pixel error and L_bokeh pixel losses; 1a has no optical loss',
              source=str(source.relative_to(ROOT))))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('job',choices=['cards','thin'])
    {'cards':cards,'thin':thin_masks}[p.parse_args().job]()
