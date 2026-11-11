"""Derived K64 geometry only; no Blender or new radiance ground truth."""
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from route3_blender_b2_full_geometry import FullWorld
OUT=ROOT/'outputs/route3-sharp-s1d-2026-11-11-01';BASE=ROOT/'outputs/route3-b2only-2026-11-10-01';RECORDS=ROOT/'03_实验记录/records';PREFIX='route3-sharp-s1d-2026-11-11'


def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def gt_color_match(parent,uv,z,obj):
    matched=np.full(len(z),-1,np.int32)
    for oid in np.unique(obj):
        qids=np.flatnonzero(obj==oid);pids=np.flatnonzero(parent['recoverable']&(parent['object_id']==oid))
        if not len(pids):continue
        feature=np.c_[parent['uv'][pids],50*np.log(parent['depth'][pids])];tree=cKDTree(feature);query=np.c_[uv[qids],50*np.log(z[qids])];dist,idx=tree.query(query,k=8,workers=8);inrange=idx<len(pids);chosen=pids[np.minimum(idx,len(pids)-1)]
        pixel_distance=np.linalg.norm(parent['uv'][chosen]-uv[qids,None],axis=-1);depth_error=abs(parent['depth'][chosen]-z[qids,None]);valid=inrange&(pixel_distance<=1.5)&(depth_error<=np.maximum(.02,.02*z[qids,None]));score=np.where(valid,pixel_distance+depth_error/np.maximum(.02,.02*z[qids,None]),np.inf);best=np.argmin(score,axis=1);ok=np.isfinite(score[np.arange(len(qids)),best]);matched[qids[ok]]=chosen[np.arange(len(qids))[ok],best[ok]]
    return matched


def main():
    criteria=json.loads((RECORDS/'route3-sharp-s1d-criteria-2026-11-11.json').read_text());assert criteria['frozen'];root=OUT/'geometry';root.mkdir(exist_ok=False);manifest=json.loads((BASE/'manifests/oracle.json').read_text())['items'];cases=[x for x in manifest if x['scene']=='B2-high' and x['fstop'] in [1.4,2.]]
    world=FullWorld(ROOT/'outputs/route3-a2b-2026-11-10-01/blender-b2/B2-high/scene-geometry.npz',512);rows=[]
    for case in cases:
        frame=case['frame_id'];old=np.load(case['geometry_gt_npz']);scoremask=old['mask_recoverable_total']&~old['residual_pixels'];data=np.load(case['input_npz'])
        # Same frozen GT-depth/CoC band as the scoring protocol.
        import cv2
        sys.path.insert(0,str(ROOT));from src.dual_layer import depth_band
        band=depth_band(data['center_depth'],data['K'][0,0],case['focus_distance_m'],case['aperture_radius_m'])[0];scoremask&=band
        pixels=np.flatnonzero(scoremask);rays=world.rays[pixels];cam=world.cameras[frame-1];disk=np.load(OUT/'visible'/(case['id']+'-aperture.npy'));parts=[];start=time.monotonic();unknown=0
        for k,xy in enumerate(disk):
            offset=np.r_[xy.astype(np.float64),0.];hit,tri=world.cast(frame,cam[:3,3]+cam[:3,:3]@offset,(rays*case['focus_distance_m']-offset)@cam[:3,:3].T)
            obj=np.zeros(len(tri),np.int16);good=tri>=0;obj[good]=world.tri_obj[tri[good]];vis,inside,first,block,_,uv=world.visible_at(frame,frame,hit,obj);known=inside&good&np.isfinite(first).all(-1);hidden=known&(np.linalg.norm(hit-cam[:3,3],axis=-1)-np.linalg.norm(first-cam[:3,3],axis=-1)>1e-4);unknown+=int((~known).sum());ids=np.flatnonzero(hidden)
            if len(ids):
                local=world.local(hit[ids],obj[ids],frame);depth=((hit[ids]-cam[:3,3])@cam[:3,:3])[:,2]
                parts.append(dict(pixel=pixels[ids].astype(np.int32),sample=np.full(len(ids),k,np.uint8),uv=uv[ids].astype(np.float32),depth=depth.astype(np.float32),object_id=obj[ids],blocker=block[ids],local_point=local.astype(np.float32)))
        values={key:np.concatenate([p[key] for p in parts]) for key in parts[0]};n=len(values['depth']);rec=np.zeros(n,bool);source_bits=np.zeros(n,np.uint16)
        for slot,other in enumerate(world.d['input_frames']):
            other=int(other)
            if other==frame:continue
            point=world.world(values['local_point'].astype(np.float64),values['object_id'],other);visible,*_=world.visible_at(other,other,point,values['object_id']);rec|=visible;source_bits|=visible.astype(np.uint16)*(1<<slot)
        parent=np.load(BASE/'methods-3d/prepared'/f'B2-high-f{frame:03d}'/'parents.npz');idx=np.flatnonzero(rec);matched=np.full(n,-1,np.int32);matched[idx]=gt_color_match(parent,values['uv'][idx],values['depth'][idx],values['object_id'][idx]);colors=np.zeros((n,3),np.float32);ok=matched>=0;colors[ok]=parent['colors_linear'][matched[ok]]
        values.update(recoverable=rec,source_bits=source_bits,GT_parent=matched,GT_color=colors,motion_related=np.isin(values['object_id'],world.dynamic)|np.isin(values['blocker'],world.dynamic),scoremask=scoremask,score_pixels=pixels,aperture_xy=disk)
        path=root/(case['id']+'.npz');np.savez_compressed(path,**values)
        row=dict(id=case['id'],frame=frame,focus_m=case['focus_distance_m'],score_pixels=len(pixels),traced_K64_rays=64*len(pixels),geometry_unknown_rays=unknown,hidden_rays=n,recoverable_hidden_rays=int(rec.sum()),color_supported_rec_rays=int((rec&ok).sum()),seconds=time.monotonic()-start,path=str(path));rows.append(row);write(root/'progress.json',dict(completed=False,rows=rows));print('S1D_GEOMETRY',case['id'],row,flush=True)
    write(RECORDS/(PREFIX+'-geometry.json'),dict(completed=True,rows=rows,existing_mesh_read_only=True,actual_renderer_K64_positions=True,new_Blender_or_DOF_truth=False))


if __name__=='__main__':main()
