"""S1d fixed-visible preparation and input-only candidate evidence."""
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
from route3_upper_bound_evaluate import BASE,OUT as UPPER,RECORDS,items
from route3_sharp_s1 import OUT as S1,FRAMES,raster
from route3_sharp_s1c_apply import anchored
from src.aperture_renderer import aperture_disk
from src.dual_layer import depth_band
from src.hidden_retrieval_rules import collect,aggregate
from src.dense_flow_geometry import sample
S1C=ROOT/'outputs/route3-sharp-s1c-2026-11-10-01';OUT=ROOT/'outputs/route3-sharp-s1d-2026-11-11-01';PREFIX='route3-sharp-s1d-2026-11-11';CRITERIA=RECORDS/'route3-sharp-s1d-criteria-2026-11-11.json'


def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def entries():return [e for e in json.loads((S1C/'prepared.json').read_text())['rows'] if e['geometry']=='cal_E1']


def source_edge(z):
    inv=1/np.maximum(z,1e-6);hi=cv2.dilate(inv,np.ones((3,3),np.uint8));lo=cv2.erode(inv,np.ones((3,3),np.uint8))
    return cv2.dilate(((hi-lo)>.03*hi).astype(np.uint8),np.ones((7,7),np.uint8))>0


def visible(c):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise ValueError('Explicit GPU0')
    subprocess.run(['nvidia-smi'],check=True);root=OUT/'visible';root.mkdir(exist_ok=False);rows=[]
    for entry in entries():
        assert (S1C/'geometry-mapping'/(entry['id']+'.json')).exists();cloud=anchored(entry,'A_color');d=np.load(entry['input_npz']);K=torch.tensor(d['K'],device='cuda')
        with torch.inference_mode():rgb,depth,alpha=raster(cloud,K)
        z=depth.cpu().numpy();a=alpha.cpu().numpy();fallback=~np.isfinite(z)|(z<=0)|(a<.1);z=np.where(fallback,d['depth'],z)
        np.savez_compressed(root/f'f{entry["frame"]:03d}.npz',depth=z,alpha=a,fallback=fallback)
        rows.append(dict(frame=entry['frame'],fallback_pixels=int(fallback.sum()),fixed_S1c_attributes=True))
        for case in entry['cases']:
            disk=aperture_disk(64,.05/(2*case['fstop']),device=K.device,dtype=K.dtype).cpu().numpy();np.save(root/(case['id']+'-aperture.npy'),disk)
        del cloud;torch.cuda.empty_cache()
    write(RECORDS/(PREFIX+'-visible.json'),dict(completed=True,rows=rows,no_anchoring_or_geometry_update=True))


def confidence(c):
    root=OUT/'confidence';root.mkdir(exist_ok=False);plan=json.loads((S1C/'flow-plan.json').read_text())['selected']['E1'];poses={f:np.load(UPPER/'U2-da3/E1'/f'f{f:03d}.npz')['c2w'] for f in FRAMES};rows=[];yy,xx=np.mgrid[:512,:512];uv=np.stack([xx,yy],-1)
    import warnings
    for f in FRAMES:
        ref=np.load(S1C/'calibration/E1'/f'f{f:03d}-reference.npz');z=ref['depth'];valid=ref['valid'];K=np.load(S1/'depth'/f'f{f:03d}.npz')['K'];rays=np.dstack([xx+.5,yy+.5,np.ones((512,512))])@np.linalg.inv(K).T;world=(rays*z[...,None])@poses[f][:3,:3].T+poses[f][:3,3];errors=[];cycles=[];angles=[]
        for pair in plan[str(f)]:
            g=pair['frame'];flow=np.load(S1C/'flow'/f'f{f:03d}-f{g:03d}.npz')['flow'];back=np.load(S1C/'flow'/f'f{g:03d}-f{f:03d}.npz')['flow'];q=uv+flow;fb=np.linalg.norm(flow+sample(back,q),axis=-1);camera=(world-poses[g][:3,3])@poses[g][:3,:3];p=camera@K.T;error=np.linalg.norm(p[...,:2]/np.maximum(p[...,2:],1e-12)-(q+.5),axis=-1)
            a=world-poses[f][:3,3];b=world-poses[g][:3,3];cos=(a*b).sum(-1)/np.maximum(np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1),1e-12);theta=np.arccos(np.clip(cos,-1,1));ok=valid&(fb<=1)&(error<=1)&(camera[...,2]>0)&(theta>=np.deg2rad(.1))
            errors.append(np.where(ok,error,np.nan));cycles.append(np.where(ok,fb,np.nan));angles.append(np.where(ok,theta,np.nan))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning);e=np.nanmedian(np.stack(errors),axis=0);fb=np.nanmedian(np.stack(cycles),axis=0);angle=np.nanmedian(np.stack(angles),axis=0)
        supported=(np.isfinite(np.stack(errors)).sum(0)>=2)&np.isfinite(e+fb+angle);weight=np.where(supported,np.exp(-.5*e**2)*np.exp(-.5*fb**2)*np.minimum(1,np.sin(angle)/np.sin(np.deg2rad(2))),.05).astype(np.float32)
        np.savez_compressed(root/f'f{f:03d}.npz',weight=weight,reprojection=e,FB=fb,angle_radians=angle,supported=supported);rows.append(dict(frame=f,supported_pixels=int(supported.sum()),weight_p10_p50_p90=np.quantile(weight,[.1,.5,.9]).tolist()));print('S1D_CONFIDENCE',f,flush=True)
    write(RECORDS/(PREFIX+'-confidence.json'),dict(completed=True,rows=rows,GT_depth_or_masks_read=False,no_new_calibration=True))


def prepare(c):
    root=OUT/'candidates';root.mkdir(exist_ok=False);views=[];rows=[]
    for f in FRAMES:
        d=np.load(S1/'depth'/f'f{f:03d}.npz');cal=np.load(S1C/'calibration/E1/SHARP'/f'f{f:03d}.npz');views.append(dict(frame=f,depth=cal['depth'],K=d['K'],c2w=cal['c2w'],rgb_linear=d['rgb_linear'],edge=source_edge(cal['depth']),confidence=np.load(OUT/'confidence'/f'f{f:03d}.npz')['weight']))
    for entry in entries():
        f=entry['frame'];target=next(v for v in views if v['frame']==f);union=np.zeros((512,512),bool);d=np.load(entry['input_npz'])
        for case in entry['cases']:union|=depth_band(d['depth'],d['K'][0,0],case['focus_m'],.05/(2*case['fstop']))[0]
        pool=collect(d['depth'],d['K'],target['c2w'],union,views,f);folder=root/f'f{f:03d}';folder.mkdir();np.savez_compressed(folder/'pool.npz',**pool);parent,audit=aggregate(pool,0,d['depth']);np.savez_compressed(folder/'R0.npz',**parent)
        old=np.load(entry['parents']['L3_cal_E1'])
        for key in ['uv','depth','colors_linear','source_mask','source_count']:assert np.array_equal(parent[key],old[key]),(f,key)
        rows.append(dict(frame=f,**audit,old_parent_arrays_identical=True));print('S1D_POOL',f,audit,flush=True)
    write(RECORDS/(PREFIX+'-candidates.json'),dict(completed=True,rows=rows,no_GT_depth_or_masks=True))


def rules(c):
    if (OUT/'STOP-unclassified.json').exists():raise RuntimeError('S1D_STOP unclassified support')
    rows=[]
    for entry in entries():
        f=entry['frame'];folder=OUT/'candidates'/f'f{f:03d}'
        with np.load(folder/'pool.npz') as archive:pool={key:archive[key] for key in archive.files}
        z=np.load(OUT/'visible'/f'f{f:03d}.npz')['depth']
        cache={}
        for stage in [1,2,3,4]:
            p,a=aggregate(pool,stage,z,cluster_cache=cache);path=folder/f'R{stage}.npz';assert not path.exists();np.savez_compressed(path,**p);rows.append(dict(frame=f,**a));print('S1D_RULE',f,stage,a,flush=True)
    write(RECORDS/(PREFIX+'-rules.json'),dict(completed=True,rows=rows,final_method='R4 regardless scores',GT_depth_or_masks_used=False))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['visible','confidence','prepare','rules'],required=True);a=p.parse_args();torch.set_num_threads(8);cv2.setNumThreads(1);c=json.loads(CRITERIA.read_text());assert c['frozen'];globals()[a.mode](c)


if __name__=='__main__':main()
