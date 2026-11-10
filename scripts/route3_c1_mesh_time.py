"""Leave-target-view-out mesh reprojection probe for temporal correspondence.

The supplied mesh has no textures. For each candidate time, transfer colors
from nearby views onto fixed first-hit mesh surfaces, then render the held-out
view. Minima are evidence of agreement, not proof of dynamic geometry GT.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.camvfx4d import load_camera_frames, intrinsics_from_fov, read_rgb
from src.c1_regions import linear_rgb, display_rgb
from route3_mesh_probe import cast


def fixed_appearance_probe(a,c):
    result=dict(method='fixed O1 center image, colors fitted at the independently selected frame1; compare to all50 target frames',
        independence='secondary appearance check, not independent temporal selection because colors were fitted at frame1',scenes=[])
    for name in c['development_scenes']:
        views=[]
        for view in c['views']:
            render=linear_rgb(np.load(a.fixed_oracle_root/name/str(view)/'oracle-renders.npz')['center_srgb'])
            support=np.load(a.regions/name/str(view)/'background.npz')['motion_masks'].any(0)
            curves=[]
            for frame in range(1,51):
                target=linear_rgb(read_rgb(a.raw/name/f'pinhole/{view}/{frame:06d}.png'))
                error=((render-target)**2).mean(-1)
                curves.append(dict(frame_id=frame,full_image_mse=float(error.mean()),
                                   dynamic_region_mse=float(error[support].mean()) if support.any() else None))
            best=int(np.argmin([r['dynamic_region_mse'] for r in curves]))+1
            row=dict(view=view,minimum_error_frame=best,dynamic_pixels=int(support.sum()),curves=curves)
            views.append(row);print(name,view,'fixed appearance minimum',best,flush=True)
        result['scenes'].append(dict(scene=name,views=views))
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',type=Path,default=Path('data/raw/CamVFX-4D'))
    p.add_argument('--regions',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path('configs/route3-c1-a1-20261110.json'))
    p.add_argument('--fixed-oracle-root',type=Path,help='Secondary fixed-appearance curve after independent correspondence probe')
    a=p.parse_args();c=json.loads(a.config.read_text());a.output.mkdir(parents=True,exist_ok=False);cv2.setNumThreads(4)
    if a.fixed_oracle_root:
        fixed_appearance_probe(a,c);return
    size=c['resolution'];result=dict(config=c,scenes=[],method='candidate-time colors from four nearest non-target views; mesh geometry fixed',
        confirmation_rule='>=1000 dynamic-region pixels, >=5% relative nonlocal minimum margin, best times across 3 views within two frames',
        limitations=['textureless mesh precludes meaningful raw vertex-color RGB matching',
                     'view-dependent reflectance and unmodelled motion can change reprojection error',
                     'candidate colors are nuisance variables; this probe alone cannot certify mesh correctness'])
    yy,xx=np.mgrid[:size,:size]
    for name in c['development_scenes']:
        started=time.monotonic();scene=a.raw/name;rows=[]
        cameras,meta=load_camera_frames(scene/'pinhole/transforms_train.json');K=intrinsics_from_fov(meta,size,size)
        rays=np.stack([xx.ravel()+.5,yy.ravel()+.5,np.ones(xx.size)],-1)@np.linalg.inv(K).T
        mesh=trimesh.load(scene/'pinhole/points3d_opencv.ply',process=False)
        mesh.vertices=np.asarray(mesh.vertices)*np.array(c['mesh_axis_transform']);ray=RayMeshIntersector(mesh)
        for view in c['views']:
            saved=np.load(a.regions/name/str(view)/'background.npz')
            motion_union=saved['motion_masks'].any(0)
            camera=cameras[(view,1)]['c2w'];points,good=cast(ray,camera[:3,3],rays@camera[:3,:3].T)
            near=sorted([v for v in range(50) if v!=view],key=lambda v:np.linalg.norm(cameras[(v,1)]['c2w'][:3,3]-camera[:3,3]))[:4]
            mappings=[];coverage=np.zeros(xx.size,np.float32)
            for source in near:
                cam=cameras[(source,1)]['c2w'];cp=(points-cam[:3,3])@cam[:3,:3];uvz=cp@K.T
                uv=uvz[:,:2]/uvz[:,2:]
                valid=good&(cp[:,2]>0)&(uv[:,0]>=.5)&(uv[:,0]<size-.5)&(uv[:,1]>=.5)&(uv[:,1]<size-.5)
                ids=np.flatnonzero(valid);hits,ok=cast(ray,cam[:3,3],points[ids]-cam[:3,3])
                distance=np.linalg.norm(points[ids]-cam[:3,3],axis=-1)
                visible=ok&(np.linalg.norm(hits-points[ids],axis=-1)<=np.maximum(1e-4,distance*1e-4))
                valid[:]=False;valid[ids[visible]]=True
                uv=np.nan_to_num(uv,nan=-1e4,posinf=-1e4,neginf=-1e4)
                mappings.append((source,uv.reshape(size,size,2).astype(np.float32)-.5,valid.reshape(size,size)))
                coverage+=valid
            region=motion_union&(coverage.reshape(size,size)>=2)
            curves=[];images=[]
            for frame in range(1,51):
                render=np.zeros((size,size,3),np.float32);weights=np.zeros((size,size),np.float32)
                for source,uv,valid in mappings:
                    rgb=linear_rgb(read_rgb(scene/f'pinhole/{source}/{frame:06d}.png'))
                    warped=cv2.remap(rgb,uv[...,0],uv[...,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
                    render+=warped*valid[...,None];weights+=valid
                render/=np.maximum(weights[...,None],1)
                target=linear_rgb(read_rgb(scene/f'pinhole/{view}/{frame:06d}.png'))
                error=((render-target)**2).mean(-1)
                curve=dict(frame_id=frame,full_valid_mse=float(error[weights>=2].mean()),
                           dynamic_region_mse=float(error[region].mean()) if region.any() else None)
                curves.append(curve);images.append(render)
            values=np.array([r['dynamic_region_mse'] for r in curves]);best=int(values.argmin())
            exclusion=c['time_correspondence_nonlocal_exclusion_frames']
            nonlocal_values=values[np.abs(np.arange(50)-best)>exclusion]
            runner=float(nonlocal_values.min());margin=(runner-float(values[best]))/max(float(values[best]),1e-12)
            pixels=int(region.sum());confident=pixels>=c['time_correspondence_min_dynamic_pixels'] and margin>=c['time_correspondence_nonlocal_margin_fraction']
            out=a.output/name/str(view);out.mkdir(parents=True)
            for label,idx in [('first',0),('minimum',best)]:
                cv2.imwrite(str(out/f'{label}-mesh-reprojected.png'),np.uint8(display_rgb(images[idx])[...,::-1]*255))
                ref=read_rgb(scene/f'pinhole/{view}/{idx+1:06d}.png')
                cv2.imwrite(str(out/f'{label}-reference.png'),np.uint8(ref[...,::-1]*255))
            np.savez_compressed(out/'correspondence-support.npz',dynamic_region=region,source_mesh_visibility_count=coverage.reshape(size,size))
            row=dict(view=view,color_source_views=near,valid_dynamic_pixels=pixels,
                minimum_error_frame=best+1,minimum_dynamic_mse=float(values[best]),nonlocal_runner_up_mse=runner,
                nonlocal_relative_margin=margin,confident_temporal_minimum=bool(confident),curves=curves)
            rows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='curves'}),flush=True)
        bests=[r['minimum_error_frame'] for r in rows]
        confirmed=all(r['confident_temporal_minimum'] for r in rows) and max(bests)-min(bests)<=2
        result['scenes'].append(dict(scene=name,views=rows,temporal_correspondence_confirmed=confirmed,
            dynamic_G_allowed=confirmed,elapsed_seconds=time.monotonic()-started,
            failure_flags=[] if confirmed else ['mesh_time_correspondence_unconfirmed']))
        (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    result['stop_condition_triggered']=any(not s['temporal_correspondence_confirmed'] for s in result['scenes'])
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
