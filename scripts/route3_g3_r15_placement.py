"""Recompute the 512-resolution target placement fields only."""
import cv2
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor

from route3_g3_r15 import OUT, DEFAULT, qa, read, write
from route3_s2a_data import save_npz


def scene(name,rule):
    root=DEFAULT/name
    audits=[]
    for entry in read(root/'prepared.json')['rows']:
        f=entry['frame']
        geo=np.load(root/'geometry'/f'f{f:03d}-center.npz')
        pose,K=geo['camera_c2w'],geo['K'].astype(np.float32)
        depth=np.load(entry['input_npz'])['depth']
        fit=np.load(root/'placement'/f'f{f:03d}.npz')
        a,b,supported=fit['a'],fit['b'],fit['local_supported']
        background=cv2.dilate(depth,np.ones((49,49),np.uint8)) if rule=='O2' else None
        def place(gt, uv, valid):
            clipped=np.floor(np.nan_to_num(uv)).clip(0,511).astype(np.int32)
            x,y=clipped[...,0],clipped[...,1]
            inverse=a[y,x]/np.maximum(gt,1e-6)+b[y,x]
            valid=valid&supported[y,x]&np.isfinite(inverse)&(inverse>1e-4)
            raw=1/np.maximum(inverse,1e-4)
            floor=1.03*depth[y,x]
            conflict=valid&(raw<floor)
            back=floor if background is None else background[y,x]
            keep=valid&(~conflict|(np.isfinite(back)&(back>=floor)))
            placed=np.where(conflict&keep,back,raw)
            return placed,keep,conflict
        label=np.load(root/'labels'/f'f{f:03d}'/(entry['cases'][0]['id']+'.npz'))
        cam=(label['GT_source_world']-pose[:3,3])@pose[:3,:3]
        uvz=cam@K.T
        uv=uvz[:,:2]/np.maximum(uvz[:,2:],1e-8)
        inside=(cam[:,2]>0)&(uv>=0).all(-1)&(uv<512).all(-1)
        z,valid,conflicts=place(cam[:,2],uv,inside)
        dest=OUT/'R2/placement512'/name/f'f{f:03d}'
        dest.mkdir(parents=True,exist_ok=True)
        save_npz(dest/'candidates.npz',dict(depth=z.astype(np.float32),valid=valid,
               camera=(cam*(z/np.maximum(cam[:,2],1e-6))[:,None]).astype(np.float32),violating=conflicts))
        for case in entry['cases']:
            cid=case['id']
            with np.load(root/'geometry'/(cid+'-K16.npz')) as lens:
                gt,z_uv,hidden=lens['depth_axial_m'],lens['reference_uv_center'],lens['hidden']
            zray,ok,violated=place(gt,z_uv,hidden)
            save_npz(dest/(cid+'.npz'),dict(depth=np.where(ok,zray,0).astype(np.float32)))
            audits.append(dict(case=cid,placement_supported=int(ok.sum()),
                               violating_rays=int(violated.sum()),dropped_rays=int((violated&~ok).sum())))
    return dict(scene=name,audits=audits)


def main():
    while not (OUT/'R2/labels-ready.json').exists():
        qa.guard()
        time.sleep(4)
    meta=read(OUT/'R2/labels-ready.json')
    with ThreadPoolExecutor(max_workers=3) as pool:
        rows=list(pool.map(lambda name:scene(name,meta['rule']),meta['scenes']))
    write(OUT/'R2/placements512-ready.json',dict(rule=meta['rule'],rows=rows))
    print('PLACEMENTS_512_READY',len(rows),flush=True)


if __name__=='__main__':
    main()
