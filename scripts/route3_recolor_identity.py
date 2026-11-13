"""Appendix11 Oracle identities/colors from existing meshes and input images.

CPU/Embree statistics only: no Blender, image rendering, baking or learning.
The 768x768 row-major grid is SHARP's source grid, not projected learned XYZ.
"""
import argparse,json,sys,time,warnings,shutil
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from route3_blender_b2_full_geometry import FullWorld
CFG=ROOT/'configs/route3-recolor-2026-11-13.json';C=json.loads(CFG.read_text());OUT=ROOT/C['output']

def save(p,**d):
    if p.exists():raise FileExistsError(p)
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.part')
    with tmp.open('wb') as h:np.savez_compressed(h,**d)
    tmp.replace(p)

def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

from src.surface_identity import local_footprint,sample_identity_rgb

def run(name,only_frame=None):
    sc=C['scenes'][name];w=FullWorld(Path(sc['geometry']),768);mesh=w.d;camera_count=768**2;cache={}
    for f in C['source_frames']:
        with np.load(Path(sc['GTmaps'])/f'f{f:03d}-center.npz') as a:gt={k:a[k] for k in ['object_id','local_surface_point','depth_axial_m','K','camera_c2w']}
        foot=local_footprint(gt['local_surface_point'],gt['object_id'],gt['depth_axial_m'],mesh,f,gt['K'],gt['camera_c2w'])
        with np.load(Path(sc['RGB_source'])/f'f{f:03d}.npz') as a:rgb=a['rgb_linear']
        cache[f]=(gt,foot,rgb)
    for f in ([only_frame] if only_frame else C['frames']):
        dest=OUT/name/f'f{f:03d}';dest.mkdir(parents=True,exist_ok=True);path=dest/'identity-colors.npz'
        if path.exists():continue
        if shutil.disk_usage(OUT).free/2**30<60:raise RuntimeError('STOP disk below60GiB')
        start=time.monotonic();center=w.center(f,1);ids0=center['object_id'].ravel();local0=center['local_surface_point'].reshape(-1,3);ids1=np.zeros(camera_count,np.int16);local1=np.full((camera_count,3),np.nan,np.float32);cam=w.cameras[f-1]
        for oid in np.unique(ids0):
            if oid<=0:continue
            take=np.flatnonzero(ids0==oid);hit,tri=w.cast_masked(f,cam[:3,3],w.rays[take]@cam[:3,:3].T,(int(oid),));ok=tri>=0;hid=np.zeros(len(take),np.int16);hid[ok]=w.tri_obj[tri[ok]];ids1[take]=hid;local1[take]=w.local(hit,hid,f).astype(np.float32)
        ids=np.r_[ids0,ids1].astype(np.int16);local=np.r_[local0,local1].astype(np.float32);observations=np.full((2*camera_count,12,3),np.nan,np.float32);counts=np.zeros(len(ids),np.uint8);exact_visible_counts=np.zeros(len(ids),np.uint8);source_index=0;per_source=[]
        good=np.flatnonzero((ids>0)&np.isfinite(local).all(-1))
        for source in C['source_frames']:
            if source==f:continue
            gt,foot,rgb=cache[source];visible_total=0;color_total=0;beg=time.monotonic()
            for b in range(0,len(good),131072):
                ix=good[b:b+131072];world=w.world(local[ix],ids[ix],source);visible,inside,hit,obj,tri,uv=w.visible_at(source,source,world,ids[ix],tolerance=1e-4);take=ix[visible];visible_total+=len(take);exact_visible_counts[take]+=1
                color,supported=sample_identity_rgb(uv[visible]*(512/768),local[take],ids[take],gt,rgb,foot);take=take[supported];observations[take,source_index]=color[supported];counts[take]+=1;color_total+=len(take)
            per_source.append(dict(frame=source,GT_exact_visible=visible_total,color_observed=color_total,seconds=time.monotonic()-beg));source_index+=1
            print('R3_ID_SOURCE',name,f,source,visible_total,color_total,round(time.monotonic()-beg,2),flush=True)
        with warnings.catch_warnings():warnings.simplefilter('ignore',RuntimeWarning);colors=np.nanmedian(observations,axis=1)
        available=counts>0;colors[~available]=0
        save(path,object_id=ids,local_point=local,oracle_color=colors,oracle_color_valid=available,observation_count=counts,exact_visible_count=exact_visible_counts)
        write(dest/'identity-colors.json',dict(scene=name,frame=f,source_grid=[768,768],gaussians=2*camera_count,layer0_identified=int((ids0>0).sum()),layer1_identified=int((ids1>0).sum()),layer0_otherframe_color=int(available[:camera_count].sum()),layer1_otherframe_color=int(available[camera_count:].sum()),sources=per_source,seconds=time.monotonic()-start,source_pixel_mapping='official rowmajor layer768x768;image512 sourcepixel centers=(i+.5)*512/768',peel='remove the center firsthit object only,retain following object;sky/nohit unassigned',visibility='exactmesh visibility1e-4m,GTlocalpoint tracked with objecttransform',color='same LDRinputlinearRGB;GT-identityconsistentbilinear,medianover12otherframes;no targetframe color',path=str(path)))
        print('R3_ID_DONE',name,f,round(time.monotonic()-start,2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scene',choices=list(C['scenes']));p.add_argument('--frame',type=int,choices=C['frames']);a=p.parse_args()
    for name in ([a.scene] if a.scene else C['scenes']):run(name,a.frame)
