"""Fixed mesh surfels from all 50 views at the verified mesh time.

Only an initialization for O1: fitting colors is a separate operation. No
geometry, opacity or scale is fitted, and no learned model is trained.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
import trimesh
from trimesh.ray.ray_pyembree import RayMeshIntersector

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.camvfx4d import load_camera_frames,intrinsics_from_fov,read_rgb
from src.c1_regions import linear_rgb


def center_visibility(scene, output):
    mesh=trimesh.load(scene/'pinhole/points3d_opencv.ply',process=False)
    mesh.vertices=np.asarray(mesh.vertices)*[1,-1,1];ray=RayMeshIntersector(mesh)
    cameras,meta=load_camera_frames(scene/'pinhole/transforms_train.json');K=intrinsics_from_fov(meta)
    d=np.load(output/'fixed-geometry.npz');points=d['means'];frame=int(d['mesh_frame_id'])
    for view in [0,16,33]:
        origin=cameras[(view,frame)]['c2w'][:3,3];visible=np.zeros(len(points),bool)
        for start in range(0,len(points),250000):
            chunk=points[start:start+250000];directions=chunk-origin
            hits,ids,_=ray.intersects_location(np.broadcast_to(origin,directions.shape).copy(),directions,multiple_hits=False)
            distance=np.linalg.norm(directions[ids],axis=-1)
            tolerance=np.maximum(1e-4,distance*1e-4)
            cam=cameras[(view,frame)]['c2w'];cp=(chunk[ids]-origin)@cam[:3,:3];uvz=cp@K.T;uv=uvz[:,:2]/uvz[:,2:]
            inside=(cp[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<512)&(uv[:,1]>=0)&(uv[:,1]<512)
            visible[start+ids]=inside&(np.linalg.norm(hits-chunk[ids],axis=-1)<=tolerance)
        np.save(output/f'center-visible-frustum-{view}.npy',visible)
        print('CENTER_VISIBLE',scene.name,view,int(visible.sum()),len(points),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scene',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--time-probe',type=Path,required=True)
    p.add_argument('--size',type=int,default=384);p.add_argument('--visibility-only',action='store_true');a=p.parse_args()
    if a.visibility_only:
        center_visibility(a.scene,a.output);return
    a.output.mkdir(parents=True,exist_ok=False)
    match=next(s for s in json.loads(a.time_probe.read_text())['scenes'] if s['scene']==a.scene.name)
    if not match['temporal_correspondence_confirmed']:raise ValueError('Mesh time not confirmed: no dynamic Oracle')
    frame=int(np.median([r['minimum_error_frame'] for r in match['views']]))
    started=time.monotonic();cv2.setNumThreads(4)
    mesh=trimesh.load(a.scene/'pinhole/points3d_opencv.ply',process=False)
    mesh.vertices=np.asarray(mesh.vertices)*[1,-1,1];ray=RayMeshIntersector(mesh)
    frames,meta=load_camera_frames(a.scene/'pinhole/transforms_train.json');K=intrinsics_from_fov(meta,a.size,a.size)
    y,x=np.mgrid[:a.size,:a.size];rays=np.stack([x.ravel()+.5,y.ravel()+.5,np.ones(x.size)],-1)@np.linalg.inv(K).T
    means=[];colors=[];scales=[];quats=[];counts=[]
    for view in range(50):
        cam=frames[(view,frame)]['c2w'];dirs=rays@cam[:3,:3].T
        points,ids,faces=ray.intersects_location(np.broadcast_to(cam[:3,3],dirs.shape).copy(),dirs,multiple_hits=False)
        depth=((points-cam[:3,3])@cam[:3,:3])[:,2];valid=depth>1e-4
        points,ids,faces,depth=points[valid],ids[valid],faces[valid],depth[valid]
        normals=np.asarray(mesh.face_normals)[faces];normals=np.where((normals[:,2]<0)[:,None],-normals,normals)
        q=np.stack([1+normals[:,2],-normals[:,1],normals[:,0],np.zeros(len(normals))],-1)
        q/=np.linalg.norm(q,axis=-1,keepdims=True)
        sigma=depth/K[0,0]
        means.append(points.astype(np.float32));quats.append(q.astype(np.float32))
        scales.append(np.stack([sigma*.8,sigma*.8,sigma*.01],-1).astype(np.float32))
        rgb=cv2.resize(linear_rgb(read_rgb(a.scene/f'pinhole/{view}/{frame:06d}.png')),(a.size,a.size),interpolation=cv2.INTER_AREA)
        colors.append(rgb.reshape(-1,3)[ids]);counts.append(len(ids))
    means=np.concatenate(means);colors=np.concatenate(colors);scales=np.concatenate(scales);quats=np.concatenate(quats)
    # Prefer finer surfels when several views sample the same small surface cell.
    voxel=float(np.max(mesh.extents)/8192)
    order=np.argsort(scales[:,0],kind='stable');grid=np.rint(means[order]/voxel).astype(np.int32)
    _,first=np.unique(grid,axis=0,return_index=True);keep=order[first]
    np.savez(a.output/'fixed-geometry.npz',means=means[keep],quats=quats[keep],scales=scales[keep],
             colors_linear=colors[keep],opacities=np.full(len(keep),.999,np.float32),source_frames=np.zeros(len(keep),np.int64),
             mesh_frame_id=frame,mesh_axis_transform=[1,-1,1],sample_views=np.arange(50),
             initializer=np.array('50 posed mesh first-hit views; metric coordinates; no geometry optimization'))
    record=dict(scene=a.scene.name,mesh_frame_id=frame,temporal_evidence='leave-view-out probe agrees across 0/16/33',
        initialization_views=50,resolution_per_view=[a.size,a.size],view_point_counts=counts,
        points_before_dedup=len(means),points_after_dedup=len(keep),dedup_voxel_m=voxel,
        fixed_attributes=['means','quats','scales','opacities'],allowed_optimization='linear RGB colors only',
        elapsed_seconds=time.monotonic()-started)
    (a.output/'geometry.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)


if __name__=='__main__':main()
