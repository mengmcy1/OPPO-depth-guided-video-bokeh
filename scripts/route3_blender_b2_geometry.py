"""Batch rigid-surface visibility for B2 with Embree; no model outputs used."""
import argparse
import json
import math
from pathlib import Path
import time
import cv2
import numpy as np
import trimesh
from trimesh.ray.ray_pyembree import RayMeshIntersector


class World:
    def __init__(self,path,size):
        self.d=np.load(path);self.names=self.d['object_names'].tolist();self.ids=self.d['object_ids'].tolist();self.size=size
        self.K=self.d['intrinsics'].copy()*np.array([[size/512,size/512,size/512],[size/512,size/512,size/512],[1,1,1]])
        self.K[2]=[0,0,1];self.cameras=self.d['camera_c2w_opencv'];self.transforms=self.d['object_world_matrices']
        self.fg=np.array(self.d['foreground_object_ids']);self.dynamic=np.array(self.d['dynamic_object_ids']);self.cache={}
        self.tri_obj=self.d['triangle_object_id'];self.tri_poly=self.d['triangle_polygon_id']
        y,x=np.mgrid[:size,:size];self.pixels=np.stack([x.ravel()+.5,y.ravel()+.5,np.ones(x.size)],-1)
        self.rays=self.pixels@np.linalg.inv(self.K).T
    def geometry(self,frame,background=False):
        key=(frame,background)
        if key in self.cache:return self.cache[key]
        vertices=[];faces=[];offset=0
        for index,obj_id in enumerate(self.ids):
            v=self.d[f'object_{obj_id}_vertices'];mat=self.transforms[frame-1,index]
            vertices.append(v@mat[:3,:3].T+mat[:3,3]);faces.append(self.d[f'object_{obj_id}_faces']+offset);offset+=len(v)
        faces=np.concatenate(faces);tri_map=np.arange(len(faces))
        if background:
            select=~np.isin(self.tri_obj,self.fg);faces=faces[select];tri_map=tri_map[select]
        mesh=trimesh.Trimesh(vertices=np.concatenate(vertices),faces=faces,process=False);ray=RayMeshIntersector(mesh)
        self.cache[key]=(ray,tri_map)
        return ray,tri_map
    def cast(self,frame,origins,directions,background=False):
        ray,tri_map=self.geometry(frame,background);origins=np.broadcast_to(origins,directions.shape).copy()
        points,indices,triangles=ray.intersects_location(origins,directions,multiple_hits=False)
        result=np.full(directions.shape,np.nan);faces=np.full(len(directions),-1,np.int32)
        result[indices]=points;faces[indices]=tri_map[triangles]
        return result,faces
    def local(self,points,obj_ids,frame):
        result=np.full(points.shape,np.nan)
        for i,obj_id in enumerate(self.ids):
            mask=obj_ids==obj_id
            if mask.any():
                inv=np.linalg.inv(self.transforms[frame-1,i]);result[mask]=points[mask]@inv[:3,:3].T+inv[:3,3]
        return result
    def world(self,local,obj_ids,frame):
        result=np.full(local.shape,np.nan)
        for i,obj_id in enumerate(self.ids):
            mask=obj_ids==obj_id
            if mask.any():
                mat=self.transforms[frame-1,i];result[mask]=local[mask]@mat[:3,:3].T+mat[:3,3]
        return result
    def center(self,frame,alpha_samples=16,background=False):
        cam=self.cameras[frame-1];points,tri=self.cast(frame,cam[:3,3],self.rays@cam[:3,:3].T,background)
        valid=tri>=0;obj=np.zeros(len(tri),np.int16);obj[valid]=self.tri_obj[tri[valid]]
        depth=np.zeros(len(tri),np.float32);depth[valid]=((points[valid]-cam[:3,3])@cam[:3,:3])[:,2]
        local=self.local(points,obj,frame);alpha=np.isin(obj,self.fg).astype(np.float32)
        if alpha_samples>1:
            grid=int(math.sqrt(alpha_samples));assert grid*grid==alpha_samples
            offsets=np.array([(x,y,0) for y in (np.arange(grid)+.5)/grid-.5 for x in (np.arange(grid)+.5)/grid-.5])@np.linalg.inv(self.K).T
            total=np.zeros(len(tri),np.float32);ray,tri_map=self.geometry(frame)
            for i in range(0,len(offsets),4):
                directions=((self.rays[None]+offsets[i:i+4,None])@cam[:3,:3].T).reshape(-1,3)
                hit=ray.intersects_first(np.broadcast_to(cam[:3,3],directions.shape).copy(),directions)
                object_id=np.zeros(len(hit),np.int16);ok=hit>=0;object_id[ok]=self.tri_obj[tri_map[hit[ok]]]
                total+=np.isin(object_id,self.fg).reshape(-1,len(tri)).sum(0)
            alpha=total/alpha_samples
        size=self.size
        return dict(depth_axial_m=depth.reshape(size,size),object_id=obj.reshape(size,size),triangle_id=tri.reshape(size,size),
                    local_surface_point=local.reshape(size,size,3).astype(np.float32),foreground_alpha=alpha.reshape(size,size),
                    center_visible=valid.reshape(size,size),K=self.K,camera_c2w=cam)
    def visible(self,frame,points,obj_ids,tolerance=1e-4):
        cam=self.cameras[frame-1];cp=(points-cam[:3,3])@cam[:3,:3];uvz=cp@self.K.T
        with np.errstate(invalid='ignore',divide='ignore'):uv=uvz[:,:2]/uvz[:,2:]
        inside=np.isfinite(points).all(-1)&(cp[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<self.size)&(uv[:,1]>=0)&(uv[:,1]<self.size)
        result=np.zeros(len(points),bool);first=np.full(points.shape,np.nan);hit_ids=np.zeros(len(points),np.int16)
        ids=np.flatnonzero(inside)
        hit,tri=self.cast(frame,cam[:3,3],points[ids]-cam[:3,3]);valid=tri>=0
        first[ids]=hit;hit_ids[ids[valid]]=self.tri_obj[tri[valid]]
        result[ids]=valid&(hit_ids[ids]==obj_ids[ids])&(np.linalg.norm(hit-points[ids],axis=-1)<=tolerance)
        return result,inside,first,hit_ids
    def aperture(self,frame,samples,fstop,focus,track=True):
        start=time.monotonic();size=self.size;cam=self.cameras[frame-1];half=samples//2;i=np.arange(half)
        r=np.sqrt((i+.5)/half);theta=i*math.pi*(3-math.sqrt(5));xy=np.stack([r*np.cos(theta),r*np.sin(theta)],-1);xy=np.concatenate([xy,-xy])*.05/(2*fstop)
        shape=(samples,size,size);objects=np.zeros(shape,np.int16);triangles=np.full(shape,-1,np.int32);locals=np.full((*shape,3),np.nan,np.float32)
        hidden=np.zeros(shape,bool);known=np.zeros(shape,bool);recover=np.zeros(shape,bool);dynamic_occlusion=np.zeros(shape,bool)
        depth=np.zeros(shape,np.float32);block_depth=np.zeros(shape,np.float32);first_seen=np.full(shape,-1,np.int16);visible_counts=np.zeros(shape,np.uint8)
        for k,offset in enumerate(xy):
            o=np.r_[offset,0.];points,tri=self.cast(frame,cam[:3,3]+cam[:3,:3]@o,(self.rays*focus-o)@cam[:3,:3].T)
            valid=tri>=0;obj=np.zeros(len(tri),np.int16);obj[valid]=self.tri_obj[tri[valid]]
            local=self.local(points,obj,frame);vis,inside,first,blocking_obj=self.visible(frame,points,obj)
            is_hidden=inside&valid&~vis&np.isfinite(first).all(-1)
            objects[k]=obj.reshape(size,size);triangles[k]=tri.reshape(size,size);locals[k]=local.reshape(size,size,3)
            hidden[k]=is_hidden.reshape(size,size);known[k]=(inside&valid&np.isfinite(first).all(-1)).reshape(size,size)
            dynamic_occlusion[k]=(is_hidden&np.isin(blocking_obj,self.dynamic)).reshape(size,size)
            depth[k]=np.nan_to_num(((points-cam[:3,3])@cam[:3,:3])[:,2],nan=0).reshape(size,size)
            block_depth[k]=np.nan_to_num(((first-cam[:3,3])@cam[:3,:3])[:,2],nan=0).reshape(size,size)
        if track:
            all_hidden=hidden.reshape(-1);ids=np.flatnonzero(all_hidden);obj=objects.reshape(-1)[ids];local=locals.reshape(-1,3)[ids]
            for other in self.d['input_frames']:
                if other==frame:continue
                points=self.world(local,obj,int(other));visible,_,_,_=self.visible(int(other),points,obj)
                recover.reshape(-1)[ids[visible]]=True;visible_counts.reshape(-1)[ids[visible]]+=1
                unset=visible&(first_seen.reshape(-1)[ids]<0);first_seen.reshape(-1)[ids[unset]]=other
        scale=self.K[0,0]*.05/(2*fstop)
        coc=np.zeros(shape,np.float32);coc[depth>0]=scale*np.abs(1/depth[depth>0]-1/focus)
        block_coc=np.zeros(shape,np.float32);block_coc[block_depth>0]=scale*np.abs(1/block_depth[block_depth>0]-1/focus)
        max_coc=np.maximum(coc,block_coc)
        summary=dict(frame=frame,resolution=[size,size],samples=samples,fstop=fstop,focus_m=focus,
            image_disocclusion_pixels=int(hidden.any(0).sum()),image_disocclusion_fraction=float(hidden.any(0).mean()),
            dynamic_disocclusion_pixels=int(dynamic_occlusion.any(0).sum()),static_disocclusion_pixels=int((hidden&~dynamic_occlusion).any(0).sum()),
            recoverable_pixels=int(recover.any(0).sum()),known_ray_fraction=float(known.mean()),
            hidden_rays=int(hidden.sum()),recoverable_rays=int(recover.sum()),recoverable_ray_fraction_among_hidden=float(recover.sum()/max(hidden.sum(),1)),
            coc_hidden_p10_p50_p90=np.quantile(coc[hidden],[.1,.5,.9]).tolist() if hidden.any() else None,
            coc_blocking_p10_p50_p90=np.quantile(block_coc[hidden],[.1,.5,.9]).tolist() if hidden.any() else None,
            coc_pairmax_p10_p50_p90=np.quantile(max_coc[hidden],[.1,.5,.9]).tolist() if hidden.any() else None,
            coc_pairmax_p10_p50_p90_at512=(np.quantile(max_coc[hidden],[.1,.5,.9])*(512/size)).tolist() if hidden.any() else None,
            low_coc_pairmax_median_below3_at512=bool(np.median(max_coc[hidden])*(512/size)<3) if hidden.any() else True,
            elapsed_seconds=time.monotonic()-start,visibility_tolerance_m=1e-4,
            includes_hidden_foreground_rays=int((hidden&np.isin(objects,self.fg)).sum()))
        arrays=dict(object_id=objects,triangle_id=triangles,local_surface_point=locals,hidden=hidden,known=known,recoverable=recover,
                    dynamic_occlusion=dynamic_occlusion,depth_axial_m=depth,blocking_depth_axial_m=block_depth,
                    coc_hidden=coc,coc_blocking=block_coc,coc_pairmax=max_coc,first_visible_other_frame=first_seen,
                    other_input_visible_count=visible_counts,aperture_xy=xy,K=self.K,camera_c2w=cam)
        return arrays,summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scene',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=['center','aperture'],required=True);p.add_argument('--frames',type=int,nargs='+',default=[1,30,60]);p.add_argument('--size',type=int,default=512)
    p.add_argument('--samples',type=int,default=16);p.add_argument('--alpha-samples',type=int,default=16);p.add_argument('--fstops',type=float,nargs='+',default=[1.4]);p.add_argument('--focuses',type=float,nargs='+',default=[1.4,5])
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);w=World(a.scene/'scene-geometry.npz',a.size);rows=[]
    for frame in a.frames:
        if a.mode=='center':
            start=time.monotonic();values=w.center(frame,a.alpha_samples);np.savez_compressed(a.output/f'f{frame:03d}-center.npz',**values)
            bg=w.center(frame,1,True);np.savez_compressed(a.output/f'f{frame:03d}-background.npz',**bg)
            row=dict(frame=frame,foreground_fraction=float(values['foreground_alpha'].mean()),center_hit_fraction=float(values['center_visible'].mean()),elapsed_seconds=time.monotonic()-start)
            rows.append(row);print(json.dumps(row),flush=True)
        else:
            for stop in a.fstops:
                for focus in a.focuses:
                    values,row=w.aperture(frame,a.samples,stop,focus);key=f'f{frame:03d}-N{stop:g}-F{focus:g}-K{a.samples}'
                    np.savez_compressed(a.output/(key+'.npz'),**values);rows.append(row);print(key,json.dumps(row),flush=True)
        (a.output/'summary.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
