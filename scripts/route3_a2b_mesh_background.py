"""CamVFX frame1 mesh-derived regions and conservative background-depth audit.

Component labels use existing method-independent A1 motion masks only. We do
not skip ambiguous surfaces to manufacture a background hit. Missing depths
remain unknown and are excluded from O6's supported-pixel analysis.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
import trimesh
from scipy.sparse.csgraph import connected_components
from trimesh.ray.ray_pyembree import RayMeshIntersector

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.camvfx4d import load_camera_frames,intrinsics_from_fov,read_rgb,effect_parameters


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--record',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    cv2.setNumThreads(4);size=512;yy,xx=np.mgrid[:size,:size]
    settings=dict(dynamic_min_current_overlap=.25,dynamic_min_ever_overlap=.35,
        static_max_current_overlap=.05,static_max_ever_overlap=.2,minimum_component_support_pixels=64,
        minimum_supported_hidden_pixels_per_scene=1000,minimum_background_depth_fraction_on_dynamic_center=.8,
        background_agreement_frames_min=10,background_depth_separation_m=.001)
    record=dict(phase='A2b CamVFX conservative background-depth feasibility',thresholds=settings,
        frame_id=1,views=[0,16,33],scenes=[],GT_foreground_segmentation_used=False,
        region_names=['mesh_near_motion','mesh_static_static','original_M_C1_proxy'],K=16,
        classification='connected-face-component projection overlap with A1 M1 and ever-motion masks across views0/16/33',
        limitations=['component labels are conservative proxies; material seams can split one object',
                     '50-frame median is not hidden-surface color GT','static certainty does not prove exact mesh geometry'])
    for name in ['plant','cat']:
        start=time.monotonic();raw=Path('data/raw/CamVFX-4D')/name
        mesh=trimesh.load(raw/'pinhole/points3d_opencv.ply',process=False);mesh.vertices=np.asarray(mesh.vertices)*[1,-1,1]
        n,labels=connected_components(trimesh.graph.edges_to_coo(mesh.face_adjacency,count=len(mesh.faces)),directed=False)
        ray=RayMeshIntersector(mesh);cameras,meta=load_camera_frames(raw/'pinhole/transforms_train.json');K=intrinsics_from_fov(meta)
        directions=np.stack([xx.ravel()+.5,yy.ravel()+.5,np.ones(xx.size)],-1)@np.linalg.inv(K).T
        totals=[];moved=[];ever_counts=[];cache=[]
        for view in [0,16,33]:
            board=np.load(f'outputs/route3-c1-2026-11-10-01/regions-dev/{name}/{view}/background.npz')
            regions=np.load(f'outputs/route3-c1-2026-11-10-01/regions-dev/{name}/{view}/regions.npz')
            cam=cameras[(view,1)]['c2w'];dirs=directions@cam[:3,:3].T;origin=cam[:3,3]
            face,ids,hit=ray.intersects_id(np.broadcast_to(origin,dirs.shape).copy(),dirs,multiple_hits=False,return_locations=True)
            component=np.full(xx.size,-1,np.int32);component[ids]=labels[face]
            m=board['motion_masks'][0].ravel();ever=board['motion_masks'].any(0).ravel()
            totals.append(np.bincount(component[ids],minlength=n))
            moved.append(np.bincount(component[ids[m[ids]]],minlength=n))
            ever_counts.append(np.bincount(component[ids[ever[ids]]],minlength=n))
            cache.append((view,cam,dirs,component,board,regions))
        totals=np.stack(totals);moved=np.stack(moved);ever_counts=np.stack(ever_counts)
        ratio=moved/np.maximum(totals,1);eratio=ever_counts/np.maximum(totals,1)
        sufficient=totals.sum(0)>=settings['minimum_component_support_pixels']
        dynamic=sufficient&(ratio.max(0)>=settings['dynamic_min_current_overlap'])&(eratio.max(0)>=settings['dynamic_min_ever_overlap'])
        static=sufficient&(ratio.max(0)<=settings['static_max_current_overlap'])&(eratio.max(0)<=settings['static_max_ever_overlap'])
        states=np.where(dynamic,1,np.where(static,0,-1));rows=[];supported_total=0
        scene_dir=a.output/name;scene_dir.mkdir()
        np.savez_compressed(scene_dir/'components.npz',face_component=labels,component_state=states,visible_counts=totals,current_overlap=ratio,ever_overlap=eratio)
        for view,cam,dirs,front_component,board,regions in cache:
            origin=cam[:3,3];face,ids,hits=ray.intersects_id(np.broadcast_to(origin,dirs.shape).copy(),dirs,multiple_hits=True,return_locations=True)
            depth=((hits-origin)@cam[:3,:3])[:,2]
            # First non-dynamic surface; ambiguous intervening surfaces invalidate it.
            non_dynamic=(states[labels[face]]!=1)&(depth>0)
            order=np.lexsort((depth[non_dynamic],ids[non_dynamic]));indices=np.flatnonzero(non_dynamic)[order]
            _,first=np.unique(ids[indices],return_index=True);chosen=indices[first]
            bg=np.zeros(xx.size,np.float32);bg_component=np.full(xx.size,-1,np.int32)
            bg[ids[chosen]]=depth[chosen];bg_component[ids[chosen]]=labels[face[chosen]]
            valid=(bg_component>=0)&(states[np.maximum(bg_component,0)]==0)
            front_known=front_component>=0;front_dynamic=front_known&(states[np.maximum(front_component,0)]==1)
            center=board['mesh_depth'].ravel()
            valid&=(~front_dynamic)|(bg>center+settings['background_depth_separation_m'])
            bg=np.where(valid,bg,0).reshape(size,size);valid=valid.reshape(size,size);fg=front_dynamic.reshape(size,size)
            radius=regions['D_t'][0]  # A1 used the conservative disk at this view.
            mesh_region=board['static_disocclusion']
            near=mesh_region&radius;static_region=mesh_region&~radius
            board_valid=board['agreement_frames']>=settings['background_agreement_frames_min']
            hidden=fg&valid&board_valid
            supported_total+=int(hidden.sum())
            known_on_dynamic=float(valid[fg].mean()) if fg.any() else None
            mask_summary={}
            for label,mask in [('mesh_all',mesh_region),('mesh_near_motion',near),('mesh_static_static',static_region),('original_M_C1_proxy',regions['M_C1_proxy'][0])]:
                union=(mask|regions['M_C1_proxy'][0]).sum()
                mask_summary[label]=dict(pixels=int(mask.sum()),image_fraction=float(mask.mean()),
                    overlap_with_proxy_pixels=int((mask&regions['M_C1_proxy'][0]).sum()),
                    IoU_with_proxy=float((mask&regions['M_C1_proxy'][0]).sum()/union) if union else None,
                    supported_background_fraction=float(valid[mask].mean()) if mask.any() else None)
            out=scene_dir/str(view);out.mkdir()
            param=effect_parameters(raw)[0]
            np.savez_compressed(out/'layer-inputs.npz',rgb_srgb=read_rgb(raw/f'pinhole/{view}/000001.png'),center_depth=board['mesh_depth'],
                background_depth=bg,background_depth_valid=valid,foreground_alpha=fg.astype(np.float32),fill_mask=fg.astype(np.float32),
                hidden_rgb_linear=board['linear_rgb'],hidden_layer_valid=hidden,K=K,
                focus_distance_m=param['focus_distance_m'],aperture_radius_m=param['aperture_radius_m'],
                mesh_all=mesh_region,mesh_near_motion=near,mesh_static_static=static_region,original_M_C1_proxy=regions['M_C1_proxy'][0])
            overlay=read_rgb(raw/f'pinhole/{view}/000001.png').copy();overlay[near]=[1,0,0];overlay[static_region]=[0,.5,1]
            cv2.imwrite(str(out/'mesh-regions.png'),np.uint8(overlay[...,::-1]*255))
            row=dict(view=view,dynamic_center_pixels=int(fg.sum()),supported_hidden_pixels=int(hidden.sum()),
                background_depth_valid_fraction=float(valid.mean()),background_known_on_dynamic_fraction=known_on_dynamic,
                failed_background_pixels=int((~valid).sum()),ambiguous_center_component_pixels=int((front_known&(states[np.maximum(front_component,0)]<0)).sum()),regions=mask_summary)
            rows.append(row);print(name,json.dumps(row),flush=True)
        reliable=supported_total>=settings['minimum_supported_hidden_pixels_per_scene'] and all(r['background_known_on_dynamic_fraction'] is not None and r['background_known_on_dynamic_fraction']>=settings['minimum_background_depth_fraction_on_dynamic_center'] for r in rows)
        s=dict(scene=name,mesh_faces=len(mesh.faces),components=n,dynamic_components=int(dynamic.sum()),static_certain_components=int(static.sum()),ambiguous_components=int((states<0).sum()),
               supported_hidden_pixels=supported_total,views=rows,background_depth_reliable_for_O6=bool(reliable),elapsed_seconds=time.monotonic()-start)
        record['scenes'].append(s);a.record.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    record['stop_condition_triggered']=any(not s['background_depth_reliable_for_O6'] for s in record['scenes'])
    a.record.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8');(a.output/'summary.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
