"""B2 full-resolution rigid-surface GT, independent of all method outputs.

Run only after the sampling-noise stage passes. Counterfactual visibility
separates camera-parallax-sufficient recovery from motion-required recovery.
Object-wise first/second peel support is checked at each hidden surface point.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time
import cv2
import numpy as np
import trimesh
from trimesh.ray.ray_pyembree import RayMeshIntersector

sys.path.insert(0,str(Path(__file__).resolve().parent))
from route3_blender_b2_geometry import World


class FullWorld(World):
    def __init__(self,path,size=512):
        super().__init__(path,size)
        self.hidden_cache={}
        self.triangle_local=np.zeros(len(self.tri_obj),np.int32)
        for obj_id in self.ids:
            indices=np.flatnonzero(self.tri_obj==obj_id)
            self.triangle_local[indices]=np.arange(len(indices))

    def masked_geometry(self,frame,exclude=()):
        exclude=tuple(sorted(set(int(x) for x in exclude)))
        if not exclude:return self.geometry(frame)
        key=(frame,exclude)
        if key in self.hidden_cache:return self.hidden_cache[key]
        vertices=[];faces=[];offset=0
        for i,obj_id in enumerate(self.ids):
            v=self.d[f'object_{obj_id}_vertices'];mat=self.transforms[frame-1,i]
            vertices.append(v@mat[:3,:3].T+mat[:3,3]);faces.append(self.d[f'object_{obj_id}_faces']+offset);offset+=len(v)
        select=~np.isin(self.tri_obj,exclude);faces=np.concatenate(faces)[select]
        mesh=trimesh.Trimesh(vertices=np.concatenate(vertices),faces=faces,process=False)
        result=(RayMeshIntersector(mesh),np.flatnonzero(select))
        self.hidden_cache[key]=result
        return result

    def cast_masked(self,frame,origin,direction,exclude=()):
        ray,mapping=self.masked_geometry(frame,exclude)
        points,indices,triangles=ray.intersects_location(np.broadcast_to(origin,direction.shape).copy(),direction,multiple_hits=False)
        hit=np.full(direction.shape,np.nan);tri=np.full(len(direction),-1,np.int32)
        hit[indices]=points;tri[indices]=mapping[triangles]
        return hit,tri

    def visible_at(self,geometry_frame,camera_frame,points,obj_ids,tolerance=1e-4,exclude=()):
        cam=self.cameras[camera_frame-1];cp=(points-cam[:3,3])@cam[:3,:3]
        uvz=cp@self.K.T
        with np.errstate(invalid='ignore',divide='ignore'):uv=uvz[:,:2]/uvz[:,2:]
        inside=np.isfinite(points).all(-1)&(cp[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<self.size)&(uv[:,1]>=0)&(uv[:,1]<self.size)
        ids=np.flatnonzero(inside);hit=np.full(points.shape,np.nan);tri=np.full(len(points),-1,np.int32)
        if len(ids):
            h,t=self.cast_masked(geometry_frame,cam[:3,3],points[ids]-cam[:3,3],exclude)
            hit[ids]=h;tri[ids]=t
        obj=np.zeros(len(points),np.int16);good=tri>=0;obj[good]=self.tri_obj[tri[good]]
        visible=good&(obj==obj_ids)&(np.linalg.norm(hit-points,axis=-1)<=tolerance)
        return visible,inside,hit,obj,tri,uv

    def barycentric(self,local,triangles):
        result=np.full(local.shape,np.nan,np.float32)
        good=triangles>=0
        obj=np.zeros(len(triangles),np.int16);obj[good]=self.tri_obj[triangles[good]]
        for obj_id in self.ids:
            select=(obj==obj_id)&good
            if not select.any():continue
            vertices=self.d[f'object_{obj_id}_vertices'];faces=self.d[f'object_{obj_id}_faces']
            tri=vertices[faces[self.triangle_local[triangles[select]]]]
            result[select]=trimesh.triangles.points_to_barycentric(tri,local[select]).astype(np.float32)
        return result

    def center_full(self,frame,alpha_samples=16,exclude=()):
        if not exclude:
            values=self.center(frame,alpha_samples)
        else:
            cam=self.cameras[frame-1];hit,tri=self.cast_masked(frame,cam[:3,3],self.rays@cam[:3,:3].T,exclude)
            good=tri>=0;obj=np.zeros(len(tri),np.int16);obj[good]=self.tri_obj[tri[good]]
            dep=np.zeros(len(tri),np.float32);dep[good]=((hit[good]-cam[:3,3])@cam[:3,:3])[:,2]
            local=self.local(hit,obj,frame)
            values=dict(depth_axial_m=dep.reshape(self.size,self.size),object_id=obj.reshape(self.size,self.size),
                triangle_id=tri.reshape(self.size,self.size),local_surface_point=local.reshape(self.size,self.size,3).astype(np.float32),
                foreground_alpha=np.isin(obj,self.fg).reshape(self.size,self.size).astype(np.float32),
                center_visible=good.reshape(self.size,self.size),K=self.K,camera_c2w=cam)
        values['surface_barycentric']=self.barycentric(values['local_surface_point'].reshape(-1,3),values['triangle_id'].reshape(-1)).reshape(self.size,self.size,3)
        values['input_frame_id']=np.array(frame);values['excluded_object_ids']=np.array(exclude,np.int16)
        return values

    def peel(self,frame,points,obj_ids,first_blocker,tolerance=1e-4):
        n=len(points);peel_count=np.zeros(n,np.uint8);first_exclude=first_blocker.astype(np.int16).copy()
        second_exclude=np.zeros(n,np.int16);residual=np.ones(n,bool);requirements=set()
        cam=self.cameras[frame-1]
        for blocker in np.unique(first_blocker):
            if blocker<=0:continue
            ids=np.flatnonzero(first_blocker==blocker)
            requirements.add((int(blocker),))
            visible,inside,hit,hitobj,tri,uv=self.visible_at(frame,frame,points[ids],obj_ids[ids],tolerance,(int(blocker),))
            # The desired surface must survive removal of the blocking object.
            visible&=obj_ids[ids]!=blocker
            peel_count[ids[visible]]=1;residual[ids[visible]]=False
            left=~visible&inside&(hitobj>0)&(hitobj!=obj_ids[ids])&(obj_ids[ids]!=blocker)
            for blocker2 in np.unique(hitobj[left]):
                sub=ids[left&(hitobj==blocker2)]
                exclusions=tuple(sorted({int(blocker),int(blocker2)}));requirements.add(exclusions)
                second_exclude[sub]=blocker2
                ok,*_=self.visible_at(frame,frame,points[sub],obj_ids[sub],tolerance,exclusions)
                ok&=~np.isin(obj_ids[sub],exclusions)
                peel_count[sub[ok]]=2;residual[sub[ok]]=False
        return peel_count,first_exclude,second_exclude,residual,requirements

    def aperture_full(self,frame,samples,fstop,focus,tolerance=1e-4):
        start=time.monotonic();size=self.size;cam=self.cameras[frame-1]
        i=np.arange(samples//2);r=np.sqrt((i+.5)/(samples//2));angle=i*math.pi*(3-math.sqrt(5))
        disk=np.stack([r*np.cos(angle),r*np.sin(angle)],-1);disk=np.concatenate([disk,-disk])*.05/(2*fstop)
        shape=(samples,size,size);objects=np.zeros(shape,np.int16);triangles=np.full(shape,-1,np.int32)
        local=np.full((*shape,3),np.nan,np.float32);depth=np.zeros(shape,np.float32);block_depth=np.zeros(shape,np.float32)
        hidden=np.zeros(shape,bool);known=np.zeros(shape,bool);blocking=np.zeros(shape,np.int16)
        projected_uv=np.full((*shape,2),np.nan,np.float32)
        for k,xy in enumerate(disk):
            offset=np.r_[xy,0.];points,tri=self.cast(frame,cam[:3,3]+cam[:3,:3]@offset,(self.rays*focus-offset)@cam[:3,:3].T)
            good=tri>=0;obj=np.zeros(len(tri),np.int16);obj[good]=self.tri_obj[tri[good]]
            vis,inside,first,bobj,_,uv=self.visible_at(frame,frame,points,obj,tolerance)
            # Coplanar object-ID ties are not disocclusion: the central first
            # surface must be strictly in front of the aperture-hit surface.
            center_range=np.linalg.norm(points-cam[:3,3],axis=-1)
            first_range=np.linalg.norm(first-cam[:3,3],axis=-1)
            h=inside&good&np.isfinite(first).all(-1)&(center_range-first_range>tolerance)
            objects[k]=obj.reshape(size,size);triangles[k]=tri.reshape(size,size)
            local[k]=self.local(points,obj,frame).reshape(size,size,3)
            depth[k]=np.nan_to_num(((points-cam[:3,3])@cam[:3,:3])[:,2],nan=0).reshape(size,size)
            block_depth[k]=np.nan_to_num(((first-cam[:3,3])@cam[:3,:3])[:,2],nan=0).reshape(size,size)
            hidden[k]=h.reshape(size,size);known[k]=(inside&good&np.isfinite(first).all(-1)).reshape(size,size)
            blocking[k]=bobj.reshape(size,size);projected_uv[k]=uv.reshape(size,size,2)
        ids=np.flatnonzero(hidden.reshape(-1));hid_obj=objects.reshape(-1)[ids];hid_local=local.reshape(-1,3)[ids].astype(np.float64)
        point_t=self.world(hid_local,hid_obj,frame)
        peel_count,peel1,peel2,residual,requirements=self.peel(frame,point_t,hid_obj,blocking.reshape(-1)[ids],tolerance)
        recover=np.zeros(len(ids),bool);motion_sufficient=np.zeros(len(ids),bool);parallax_sufficient=np.zeros(len(ids),bool)
        first_seen=np.full(len(ids),-1,np.int16);visible_count=np.zeros(len(ids),np.uint8)
        for other in self.d['input_frames']:
            other=int(other)
            if other==frame:continue
            point_s=self.world(hid_local,hid_obj,other)
            actual,*_=self.visible_at(other,other,point_s,hid_obj,tolerance)
            recover|=actual;visible_count+=actual.astype(np.uint8)
            first_seen[(first_seen<0)&actual]=other
            # Counterfactuals are evaluated for the same actual observing input.
            motion,*_=self.visible_at(other,frame,point_s,hid_obj,tolerance)
            parallax,*_=self.visible_at(frame,other,point_t,hid_obj,tolerance)
            motion_sufficient|=actual&motion;parallax_sufficient|=actual&parallax
        motion_required=recover&~parallax_sufficient
        mixed=recover&~motion_sufficient&~parallax_sufficient
        overlap=recover&motion_sufficient&parallax_sufficient
        def ray_array(value,dtype,fill=0):
            a=np.full(np.prod(shape),fill,dtype=dtype);a[ids]=value;return a.reshape(shape)
        residual_rays=ray_array(residual,bool);rec_rays=ray_array(recover,bool)
        parallax_rays=ray_array(recover&parallax_sufficient,bool);motion_rays=ray_array(motion_required,bool)
        usable=hidden&~residual_rays;residual_pixels=residual_rays.any(0)
        raw_rec=rec_rays.any(0);effective_rec=(rec_rays&usable).any(0)&~residual_pixels
        parallax_pixels=(parallax_rays&usable).any(0)&effective_rec
        motion_pixels=effective_rec&~parallax_pixels
        unrecoverable=usable.any(0)&~effective_rec&~residual_pixels
        assert np.all(~rec_rays|hidden)
        assert not np.any(motion_pixels&parallax_pixels)
        assert np.array_equal(motion_pixels|parallax_pixels,effective_rec)
        assert not np.any(effective_rec&unrecoverable)
        factor=self.K[0,0]*.05/(2*fstop)
        coc=np.zeros(shape,np.float32);coc[depth>0]=factor*np.abs(1/depth[depth>0]-1/focus)
        bcoc=np.zeros(shape,np.float32);bcoc[block_depth>0]=factor*np.abs(1/block_depth[block_depth>0]-1/focus)
        pair=np.maximum(coc,bcoc)
        current_depth=self.center(frame,1)['depth_axial_m'];current_coc=np.zeros((size,size),np.float32)
        ok=current_depth>0;current_coc[ok]=factor*np.abs(1/current_depth[ok]-1/focus)
        def quantile(x):return np.quantile(x,[.1,.5,.9]).tolist() if len(x) else None
        summary=dict(frame=frame,resolution=[size,size],samples=samples,fstop=fstop,focus_m=focus,
            hidden_rays=len(ids),residual_rays=int(residual.sum()),residual_fraction_of_hidden=float(residual.mean()) if len(ids) else None,
            self_occluded_hidden_rays=int((hid_obj==blocking.reshape(-1)[ids]).sum()),
            residual_self_occluded_rays=int((residual&(hid_obj==blocking.reshape(-1)[ids])).sum()),
            residual_pixels=int(residual_pixels.sum()),peel1_supported_rays=int((peel_count==1).sum()),peel2_supported_rays=int((peel_count==2).sum()),
            raw_hidden_pixels=int(hidden.any(0).sum()),raw_recoverable_pixels=int(raw_rec.sum()),
            effective_recoverable_total_pixels=int(effective_rec.sum()),motion_required_pixels=int(motion_pixels.sum()),
            parallax_sufficient_pixels=int(parallax_pixels.sum()),unrecoverable_pixels=int(unrecoverable.sum()),
            actual_recoverable_rays=int(recover.sum()),motion_sufficient_rays=int(motion_sufficient.sum()),parallax_sufficient_rays=int(parallax_sufficient.sum()),
            motion_parallax_overlap_rays=int(overlap.sum()),joint_only_mixed_rays=int(mixed.sum()),
            known_aperture_ray_fraction=float(known.mean()),image_disocclusion_fraction=float(hidden.any(0).mean()),
            coc_hidden_p10_p50_p90=quantile(coc[hidden]),coc_blocking_p10_p50_p90=quantile(bcoc[hidden]),
            coc_pairmax_p10_p50_p90=quantile(pair[hidden]),coc_current_in_effective_rec_p10_p50_p90=quantile(current_coc[effective_rec]),
            low_coc_median_below3=bool(np.median(pair[hidden])<3) if hidden.any() else True,
            elapsed_seconds=time.monotonic()-start,visibility_tolerance_m=tolerance,
            hidden_definition='central first-hit range is at least1e-4m nearer than aperture hit; coplanar object-ID ties are not hidden',
            classification='only actual-visible sources count; parallax-sufficient priority, remaining recovered rays motion-required; joint-only mixed separately retained',
            pixel_classification='union over16 aperture rays; drop every pixel with any residual ray; parallax priority yields disjoint pixel classes',
            required_exclusion_sets=[list(x) for x in sorted(requirements)])
        summary['invariants']=dict(recoverable_subset_of_hidden=True,display_classes_disjoint=True,
            motion_or_parallax_equals_effective_recoverable=True,unrecoverable_disjoint_from_recoverable=True)
        arrays=dict(object_id=objects,triangle_id=triangles,local_surface_point=local,depth_axial_m=depth,
            blocking_object_id=blocking,blocking_depth_axial_m=block_depth,reference_uv_center=projected_uv,
            hidden=hidden,known=known,recoverable=rec_rays,motion_required=motion_rays,parallax_sufficient=parallax_rays,
            motion_sufficient=ray_array(motion_sufficient,bool),joint_only_mixed=ray_array(mixed,bool),motion_parallax_overlap=ray_array(overlap,bool),
            first_visible_other_frame=ray_array(first_seen,np.int16,-1),visible_other_input_count=ray_array(visible_count,np.uint8),
            peel_count=ray_array(peel_count,np.uint8),peel_first_excluded_id=ray_array(peel1,np.int16),peel_second_excluded_id=ray_array(peel2,np.int16),
            residual=residual_rays,residual_pixels=residual_pixels,mask_recoverable_total=effective_rec,mask_motion_required=motion_pixels,
            mask_parallax_sufficient=parallax_pixels,mask_unrecoverable=unrecoverable,mask_disocclusion=hidden.any(0)&~residual_pixels,
            coc_hidden=coc,coc_blocking=bcoc,coc_pairmax=pair,coc_current=current_coc,
            aperture_xy=disk,K=self.K,camera_c2w=cam,input_frames=self.d['input_frames'],frame_id=np.array(frame))
        return arrays,summary


def save_json(path,value):
    temporary=path.with_name(path.name+'.part')
    temporary.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    temporary.replace(path)


def save_npz(path,values):
    temporary=path.with_name(path.name+'.part')
    with temporary.open('wb') as handle:np.savez_compressed(handle,**values)
    temporary.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scene',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=['benchmark','center','aperture','references'],required=True);p.add_argument('--size',type=int,default=512)
    p.add_argument('--frames',type=int,nargs='+',default=[1,30,60]);p.add_argument('--fstops',type=float,nargs='+',default=[1.4,2.,.5])
    p.add_argument('--focuses',type=float,nargs='+',default=[1.4,5.]);p.add_argument('--samples',type=int,default=16)
    p.add_argument('--alpha-samples',type=int,default=16);p.add_argument('--noise-record',type=Path,required=True)
    p.add_argument('--max-residual-fraction',type=float,default=.018)
    a=p.parse_args();noise=json.loads(a.noise_record.read_text())
    if not noise.get('passed',False) and noise.get('status') not in ['passed','noise_passed','sampling_noise_passed']:
        raise RuntimeError('Sampling noise must be explicitly passed before formal geometry')
    a.output.mkdir(parents=True,exist_ok=True);cv2.setNumThreads(4);w=FullWorld(a.scene/'scene-geometry.npz',a.size)
    rows=[];requirements={};existing=a.output/'summary.json'
    if existing.exists():rows=json.loads(existing.read_text())
    if (a.output/'required-exclusions.json').exists():requirements=json.loads((a.output/'required-exclusions.json').read_text())['frames']
    if a.mode=='references':
        req=json.loads((a.output/'required-exclusions.json').read_text())
        for frame,sets in req['frames'].items():
            for exclusion in sets:
                key=f'f{int(frame):03d}-exclude-'+ '-'.join(map(str,exclusion));path=a.output/(key+'.npz')
                if not path.exists():save_npz(path,w.center_full(int(frame),1,exclusion))
        return
    for frame in a.frames[:1] if a.mode=='benchmark' else a.frames:
        if a.mode!='center' and frame not in w.d['input_frames']:
            raise ValueError('Representative GT frames must belong to the fixed13 inputs')
        if a.mode=='center':
            path=a.output/f'f{frame:03d}-center.npz'
            if path.exists():continue
            start=time.monotonic();values=w.center_full(frame,a.alpha_samples);save_npz(path,values)
            background=w.center_full(frame,1,tuple(w.fg.tolist()));save_npz(a.output/f'f{frame:03d}-background.npz',background)
            row=dict(frame=frame,mode='center',elapsed_seconds=time.monotonic()-start,foreground_alpha_samples=a.alpha_samples,
                alpha_definition=f'{int(math.sqrt(a.alpha_samples))}x{int(math.sqrt(a.alpha_samples))} regular subpixel first-hit foreground fraction',center_valid_fraction=float(values['center_visible'].mean()))
            rows.append(row);save_json(existing,rows);print(json.dumps(row),flush=True)
            continue
        for stop in a.fstops[:1] if a.mode=='benchmark' else a.fstops:
            for focus in a.focuses[:1] if a.mode=='benchmark' else a.focuses:
                key=f'f{frame:03d}-N{stop:g}-F{focus:g}-K{a.samples}';path=a.output/(key+'.npz')
                if path.exists():continue
                arrays,row=w.aperture_full(frame,a.samples,stop,focus)
                write_start=time.monotonic();save_npz(path,arrays);row['write_seconds']=time.monotonic()-write_start
                row['key']=key;row['mode']=a.mode;rows.append(row);save_json(existing,rows)
                req=sorted({tuple(x) for x in requirements.get(str(frame),[])}|{tuple(x) for x in row['required_exclusion_sets']})
                requirements[str(frame)]=[list(x) for x in req]
                save_json(a.output/'required-exclusions.json',dict(object_names={str(i):n for i,n in zip(w.ids,w.names)},frames=requirements))
                print(json.dumps(row),flush=True)
                if row['residual_fraction_of_hidden'] is not None and row['residual_fraction_of_hidden']>a.max_residual_fraction:
                    save_json(a.output/'stop.json',dict(condition='two-peel residual substantially above0.9% preflight',threshold=a.max_residual_fraction,observation=row))
                    raise SystemExit(3)


if __name__=='__main__':main()
