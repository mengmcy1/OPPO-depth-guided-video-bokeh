"""S2a full observations and mesh-only labels; native K16 renders not required."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt,uniform_filter
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from route3_pilot_fields import OUT,read,write,RECORDS,PREFIX
from route3_pilot_models import FRAMES,K
from route3_sharp_s1d import source_edge
from src.hidden_retrieval_rules import collect,aggregate
from src.dual_layer import depth_band
from src.observation_memory import build_memory


def load_views(folder):
    views=[]
    for f in FRAMES:
        d=np.load(folder/'models/SHARP'/f'f{f:03d}.npz');c=np.load(folder/'models/calibration/SHARP'/f'f{f:03d}.npz');confidence=np.load(folder/'models/calibration'/f'f{f:03d}-confidence.npz');edge=source_edge(c['depth'])
        views.append(dict(frame=f,depth=c['depth'],K=K,c2w=c['c2w'],rgb_linear=d['rgb_linear'],rgb_srgb=d['rgb_srgb'],edge=edge,edge_distance=distance_transform_edt(~edge).astype(np.float32),confidence=confidence['confidence'],reprojection=confidence['reprojection'],FB=confidence['FB'],angle_radians=confidence['angle_radians']))
    return views


def prepare(folder):
    root=folder/'candidates';root.mkdir(exist_ok=True);views=load_views(folder);entries=[];fillitems=[]
    for frame in [1,30,60]:
        target=next(v for v in views if v['frame']==frame);z=target['depth'];g=np.load(folder/'geometry'/f'f{frame:03d}-center.npz');gt=g['depth_axial_m'];union=np.zeros((512,512),bool);cases=[];max_radius=0.
        for nominal in [1.4,5.]:
            for stop in [1.4,2.]:
                r=.05/(2*stop);focusmask=(gt>0)&(K[0,0]*r*abs(1/np.maximum(gt,1e-6)-1/nominal)<=1);focus=float(np.median(z[focusmask]));band,edge,radius=depth_band(z,K[0,0],focus,r);union|=band;max_radius=max(max_radius,float(radius.max()));cases.append(dict(id=f'f{frame:03d}-N{stop:g}-F{nominal:g}',frame=frame,focus_m=focus,nominal_focus_m=nominal,fstop=stop,band_pixels=int(band.sum())))
        dest=root/f'f{frame:03d}';dest.mkdir(exist_ok=True)
        if not (dest/'pool.npz').exists():
            pool=collect(z,K,target['c2w'],union,views,frame);n=len(pool['depth']);xyz=np.empty((n,3),np.float32);patch=np.empty((n,3,3,3),np.uint8);extra={k:np.empty(n,np.float32) for k in ['edge_distance','reprojection','FB','angle_radians']}
            for slot,v in enumerate(views):
                at=pool['slot']==slot;xy=pool['source_xy'][at];sx,sy=xy[:,0],xy[:,1];zz=v['depth'][sy,sx];rays=np.c_[sx+.5,sy+.5,np.ones(len(sx))]@np.linalg.inv(K).T;xyz[at]=(rays*zz[:,None])@v['c2w'][:3,:3].T+v['c2w'][:3,3]
                for key in extra:extra[key][at]=v[key][sy,sx]
                for dy in [-1,0,1]:
                    for dx in [-1,0,1]:patch[at,dy+1,dx+1]=np.uint8(np.rint(v['rgb_srgb'][np.clip(sy+dy,0,511),np.clip(sx+dx,0,511)]*255))
            np.savez_compressed(dest/'pool.npz',**pool,position_world_estimated=xyz,source_frame_id=np.array(FRAMES,np.int16)[pool['slot']],rgb_patch=patch,**extra)
        mask=cv2.dilate(union.astype(np.uint8),np.ones((5,5),np.uint8),iterations=2)>0;extension=int(np.ceil(2*max_radius))+4;bg=cv2.dilate(z,np.ones((2*extension+1,2*extension+1),np.uint8));inputpath=dest/'inputs.npz'
        if not inputpath.exists():np.savez_compressed(inputpath,depth=z,K=K,c2w=target['c2w'],rgb_srgb=target['rgb_srgb'],rgb_linear=target['rgb_linear'],fill_mask=mask,background_depth=bg)
        entry=dict(scene=folder.name,frame=frame,input_npz=str(inputpath),cases=cases,fill=str(folder/'fill'/f'f{frame:03d}'/'fill.npz'),pool=str(dest/'pool.npz'),input_fields_have_GT=False,GT_only_for_focusmask=True);entries.append(entry);fillitems.append(dict(id=f'f{frame:03d}',input_npz=str(inputpath),uses_GT_segmentation=False,uses_true_hidden_depth=False));print('S2A_CANDIDATES',folder.name,frame,flush=True)
    write(folder/'prepared.json',dict(rows=entries));write(folder/'fill-inputs.json',dict(items=fillitems))
    # Compact memory index, separate from all target-dependent labels.
    memory=folder/'memory';memory.mkdir(exist_ok=True)
    if not (memory/'memory.npz').exists():
        xyz=[];rgb=[];conf=[];refs=[];yy,xx=np.mgrid[:512,:512]
        for v in views:
            z=v['depth'];rays=np.dstack([xx+.5,yy+.5,np.ones(z.shape)])@np.linalg.inv(K).T;xyz.append(((rays*z[...,None])@v['c2w'][:3,:3].T+v['c2w'][:3,3]).reshape(-1,3));rgb.append(v['rgb_linear'].reshape(-1,3));conf.append(v['confidence'].ravel());refs.append(np.c_[np.full(xx.size,v['frame']),xx.ravel(),yy.ravel()].astype(np.int16))
        xyz=np.concatenate(xyz).astype(np.float32);rgb=np.concatenate(rgb).astype(np.float32);conf=np.concatenate(conf);refs=np.concatenate(refs);m,stats=build_memory(xyz,rgb,conf,refs)
        np.savez_compressed(memory/'observations.npz',position_world_estimated=xyz,colors_linear=rgb.astype(np.float16),confidence=conf,source_reference=refs);np.savez_compressed(memory/'memory.npz',**m);write(memory/'summary.json',stats);print('S2A_MEMORY',folder.name,stats,flush=True)


def local_placement(gt_depth,estimated,background,radius=24,return_valid=False):
    """GT-label-only disparity mapping; fit visible background, never hidden GT."""
    valid=background&(gt_depth>0)&np.isfinite(estimated)&(estimated>0);yy,xx=np.mgrid[:512,:512];held=((xx//8+yy//8)%5==0)&valid;train=valid&~held;g=1/np.maximum(gt_depth,1e-6);e=1/np.maximum(estimated,1e-6);size=2*radius+1
    box=lambda v:uniform_filter(v.astype(np.float64),size=size,mode='constant')*size*size
    n=box(train);sx=box(g*train);sy=box(e*train);sxx=box(g*g*train);sxy=box(g*e*train);var=sxx-sx*sx/np.maximum(n,1);cov=sxy-sx*sy/np.maximum(n,1)
    a=np.where(var>1e-5,cov/np.maximum(var,1e-12),1.);a=np.clip(a,.1,3.);b=(sy-a*sx)/np.maximum(n,1);supported=n>=32
    # Label-only fallback to the nearest supported local visible-background fit.
    if supported.any():index=distance_transform_edt(~supported,return_distances=False,return_indices=True);a=np.where(supported,a,a[tuple(index)]);b=np.where(supported,b,b[tuple(index)])
    else:a=np.ones_like(g);b=np.zeros_like(g)
    error=(a*g+b-e)[held];result=(a.astype(np.float32),b.astype(np.float32),dict(local_fit_supported_fraction=float(supported.mean()),heldout_pixels=int(held.sum()),heldout_inverse_depth_RMSE=float(np.sqrt(np.mean(error**2))) if len(error) else None,heldout_relative_inverse_error=float(np.mean(abs(error)/np.maximum(e[held],1e-6))) if len(error) else None,window_radius_px=radius,estimated_geometry_consistency_validation_not_true_hidden_depth_error=True));return (*result,supported) if return_valid else result


def labels(folder,training=False):
    label_root=folder/('labels-training-focus' if training else 'labels')
    if (label_root/'summary.json').exists():return
    started=time.monotonic()
    views=load_views(folder);native={};metadata=json.loads((folder/'scene.json').read_text());fg=np.array([o['id'] for o in metadata['objects'] if o['name'] in metadata['foreground_names']]);rows=[]
    for v in views:
        f=v['frame'];p=folder/'input-passes-v2';world=read(p/f'f{f:03d}-Position-{f:04d}.exr');obj=read(p/f'f{f:03d}-IndexOB-{f:04d}.exr')[...,0].round().astype(np.int16);delta=[]
        for axis in [0,1]:
            d=np.linalg.norm(np.diff(world,axis=axis),axis=-1);same=np.diff(obj,axis=axis)==0;d=np.where(same,d,0);pads=[(0,0),(0,0)];pads[axis]=(0,1);delta.append(np.pad(d,pads))
        native[f]=dict(world=world,object=obj,footprint=np.maximum(np.maximum(*delta),1e-4))
    training_plan=json.loads((folder/'geometry/training-focus-plan.json').read_text())['rows'] if training else []
    for entry in json.loads((folder/'prepared.json').read_text())['rows']:
        if training:entry=dict(entry,cases=next(r['cases'] for r in training_plan if r['frame']==entry['frame']))
        f=entry['frame'];d=np.load(entry['input_npz']);pool={k:v for k,v in np.load(entry['pool']).items()};n=len(pool['depth']);src=np.empty((n,3),np.float32);srcobj=np.empty(n,np.int16);radius=np.empty(n,np.float32)
        for slot,v in enumerate(views):
            at=pool['slot']==slot;xy=pool['source_xy'][at];raw=native[v['frame']];src[at]=raw['world'][xy[:,1],xy[:,0]];srcobj[at]=raw['object'][xy[:,1],xy[:,0]];radius[at]=raw['footprint'][xy[:,1],xy[:,0]]
        current=np.load(folder/'geometry'/f'f{f:03d}-center.npz');a,b,audit=local_placement(current['depth_axial_m'],d['depth'],~np.isin(current['object_id'],fg)&current['center_visible']);labeldir=label_root/f'f{f:03d}';labeldir.mkdir(parents=True,exist_ok=True)
        pose=current['camera_c2w'];true_camera=(src-pose[:3,3])@pose[:3,:3];true_z=true_camera[:,2];central=true_camera@K.T;central=central[:,:2]/np.maximum(central[:,2:],1e-8);cx=np.floor(central[:,0]).astype(int);cy=np.floor(central[:,1]).astype(int);in_center=(true_z>0)&(cx>=0)&(cx<512)&(cy>=0)&(cy<512);cx=np.clip(cx,0,511);cy=np.clip(cy,0,511);central_world=native[f]['world'][cy,cx]
        center_hidden=in_center&(srcobj>0)&(current['depth_axial_m'][cy,cx]>0)&(true_z>current['depth_axial_m'][cy,cx]+1e-4)&(np.linalg.norm(src-central_world,axis=-1)>1.5*radius+1e-4)
        placement_depth=1/np.maximum(a[cy,cx]/np.maximum(true_z,1e-6)+b[cy,cx],1e-4);placement_camera=true_camera*(placement_depth/np.maximum(true_z,1e-6))[:,None]
        for case in entry['cases']:
            if (labeldir/(case['id']+'.npz')).exists():continue
            geo=np.load(folder/'geometry'/(case['id']+'-K16.npz'));hidden=geo['hidden'];bits=np.zeros(n,np.uint16);match_count=np.zeros(n,np.uint8);surface_matches=0;positions=[]
            if 'world_position' in geo.files:mesh_world=geo['world_position']
            else:
                uv=geo['reference_uv_center'];rays=np.concatenate([uv,np.ones((*uv.shape[:-1],1))],-1)@np.linalg.inv(K).T;cam=rays*geo['depth_axial_m'][...,None];mesh_world=(cam@pose[:3,:3].T+pose[:3,3]).astype(np.float32)
            mesh_objects=geo['object_id']
            for k in range(16):
                hit=mesh_world[k];obj=mesh_objects[k]
                # Content is checked at the source point's TRUE off-axis
                # projection, not at its estimated target-pixel placement.
                # Thus changing the estimated depth cannot change content
                # labels for an unchanged source observation.
                xy=geo['aperture_xy'][k];camera=true_camera-np.r_[xy,0.];intr=K.copy();intr[:2,2]+=np.diag(K)[:2]*xy/case['nominal_focus_m'];project=camera@intr.T;project=project[:,:2]/np.maximum(project[:,2:],1e-8);px=np.floor(project[:,0]).astype(int);py=np.floor(project[:,1]).astype(int);inside=(camera[:,2]>0)&(px>=0)&(px<512)&(py>=0)&(py<512);px=np.clip(px,0,511);py=np.clip(py,0,511)
                h=hit[py,px];oid=obj[py,px];need=hidden[k,py,px];good=center_hidden&inside&(srcobj==oid)&need&(np.linalg.norm(src-h,axis=-1)<=1.5*radius+1e-4)
                # Also require that the hit projects inside this source pixel.
                for slot,v in enumerate(views):
                    at=np.flatnonzero(good&(pool['slot']==slot));c=(h[at]-v['c2w'][:3,3])@v['c2w'][:3,:3];q=c@K.T;q=q[:,:2]/np.maximum(q[:,2:],1e-8);good[at]&=(np.abs(q-(pool['source_xy'][at]+.5))<=.75).all(-1)
                bits|=good.astype(np.uint16)*(1<<k);match_count+=good.astype(np.uint8);surface_matches+=int(good.sum())
                hitz=geo['depth_axial_m'][k];mapped=1/np.maximum(a/np.maximum(hitz,1e-6)+b,1e-4);positions.append(np.where(hidden[k],mapped,0).astype(np.float32))
            np.savez_compressed(labeldir/(case['id']+'.npz'),content_hit_bits=bits,content_hit_fraction=match_count.astype(np.float32)/16,placement_depth=np.stack(positions),candidate_placement_depth=placement_depth.astype(np.float32),candidate_placement_camera=placement_camera.astype(np.float32),candidate_GT_center_projection=central.astype(np.float32),placement_affine_a=a,placement_affine_b=b,hidden_ray_mask=hidden,recoverable_ray_mask=geo['recoverable'],GT_source_world=src,GT_source_object_id=srcobj)
            correct=match_count>0;candidate_error=np.linalg.norm(pool['position_world_estimated']-((placement_camera@pose[:3,:3].T)+pose[:3,3]),axis=-1)
            rows.append(dict(frame=f,id=case['id'],candidates=n,correct_any_ray_candidates=int(correct.sum()),content_correct_rate=float(correct.mean()) if n else None,weighted_hit_fraction_mean=float(match_count.mean()/16) if n else None,matched_candidate_rays=surface_matches,placement_alignment=audit,current_candidate_to_aligned_GT_target_distance_p50_p90_m=np.quantile(candidate_error[correct],[.5,.9]).tolist() if correct.any() else None,no_estimated_depth_tolerance_used_in_content_label=True,content_independent_of_estimated_placement=True));print('S2A_LABELS',folder.name,case['id'],rows[-1]['content_correct_rate'],flush=True)
    # World-coordinate-based memory diagnostics, never fed back into memory.
    memory_index=folder/'memory/verified-index.npz'
    if not memory_index.exists():memory_index=folder/'memory/memory.npz'
    m=np.load(memory_index);entry=m['observation_entry_id'];ids=[];world=[]
    for f in FRAMES:ids.append(native[f]['object'].ravel());world.append(native[f]['world'].reshape(-1,3))
    ids=np.concatenate(ids);world=np.concatenate(world);valid=(ids>0)&(entry>=0);isfg=np.isin(ids,fg);ne=len(m['position']);fgcount=np.bincount(entry[valid&isfg],minlength=ne);bgcount=np.bincount(entry[valid&~isfg],minlength=ne);mixed=(fgcount>0)&(bgcount>0)
    gtcell=np.c_[ids[valid],np.floor(world[valid]/.02).astype(np.int32)];pairs=np.unique(np.c_[gtcell,entry[valid]],axis=0);cells,count=np.unique(pairs[:,:4],axis=0,return_counts=True)
    memory_stats=dict(entries=ne,index_path=str(memory_index),foreground_background_wrong_merge_entries=int(mixed.sum()),foreground_background_wrong_merge_fraction=float(mixed.mean()) if len(mixed) else None,GT_surface_cells=len(cells),GT_surface_cells_multiple_entries=int((count>1).sum()),duplicate_GT_cell_fraction=float((count>1).mean()) if len(count) else None,definition='sameobject+2cmGTworldvoxel;duplicatecellcountisresolutiondependentproxy,notexactcontinuoussurfaceidentity',GT_only_diagnostic=True)
    write(label_root/'summary.json',dict(rows=rows,memory=memory_stats,seconds_total=time.monotonic()-started,training_focus=training,target_firsthit_source='exportedBlender mesh+Embree;native K16 omitted after pilot validation',shared_appearance_and_placement_support=str(folder/'labels'),content_definition='native source ownGTworld projected through trueK16subcameras;matchfirsthitusingworldpixel-footprintandobject/sourcepixel;independentofestimatedprojection;neverestimated-depth tolerance',placement_definition='GT disparity mapped by local visible-background affine into sharedestimatedcoordinates;heldout visiblepoints onlyforvalidation'))


def training_labels(folder):return labels(folder,training=True)


def appearance(folder):
    """Separate peeled-surface RGB labels; never changes an input observation."""
    from scipy.spatial import cKDTree
    rows=[]
    for entry in json.loads((folder/'prepared.json').read_text())['rows']:
        f=entry['frame'];dest=folder/'labels'/f'f{f:03d}'/'appearance-targets.npz';g=np.load(folder/'labels'/f'f{f:03d}'/(entry['cases'][0]['id']+'.npz'));world=g['GT_source_world'];obj=g['GT_source_object_id'];oracle=np.load(folder/'oracle'/f'f{f:03d}.npz');gt=np.load(folder/'geometry'/f'f{f:03d}-center.npz');camera=(world-gt['camera_c2w'][:3,3])@gt['camera_c2w'][:3,:3];radius=3*np.maximum(camera[:,2],0)/gt['K'][0,0]+1e-4;distance=np.full(len(world),np.inf);rgb=np.zeros((len(world),3),np.float32)
        for oid in np.unique(obj):
            if oid<=0:continue
            at=np.flatnonzero(obj==oid);parents=np.flatnonzero((oracle['object_id']==oid)&oracle['recoverable'])
            if not len(parents):continue
            dist,near=cKDTree(oracle['world_position'][parents]).query(world[at],workers=8);distance[at]=dist;rgb[at]=oracle['colors_linear'][parents[near]]
        valid=distance<=radius
        if not dest.exists():np.savez_compressed(dest,hidden_rgb_linear=rgb,valid=valid,GT_surface_world_distance=distance.astype(np.float32),GT_world_pixel_footprint_radius=radius.astype(np.float32))
        for case in entry['cases']:
            correct=np.load(folder/'labels'/f'f{f:03d}'/(case['id']+'.npz'))['content_hit_fraction']>0;rows.append(dict(id=case['id'],correct_candidates=int(correct.sum()),appearance_supported_correct=int((correct&valid).sum()),appearance_valid_fraction_of_correct=float((correct&valid).sum()/max(correct.sum(),1))))
    write(folder/'labels/appearance-summary.json',dict(rows=rows,color_source='existing256spp camera-only peeled references,matchedbyobjectandworldpixel-footprint;center-viewappearanceproxy',GT_only_supervision=True,no_training_input_changed=True))


def placement_support(folder):
    """Expose label uncertainty; extrapolated unsupported labels are not losses."""
    metadata=json.loads((folder/'scene.json').read_text());fg=[o['id'] for o in metadata['objects'] if o['name'] in metadata['foreground_names']];rows=[]
    for e in json.loads((folder/'prepared.json').read_text())['rows']:
        f=e['frame'];g=np.load(folder/'geometry'/f'f{f:03d}-center.npz');d=np.load(e['input_npz']);a,b,stats,valid=local_placement(g['depth_axial_m'],d['depth'],~np.isin(g['object_id'],fg)&g['center_visible'],return_valid=True);label=np.load(folder/'labels'/f'f{f:03d}'/(e['cases'][0]['id']+'.npz'));uv=label['candidate_GT_center_projection'];x=np.clip(np.floor(uv[:,0]).astype(int),0,511);y=np.clip(np.floor(uv[:,1]).astype(int),0,511);world=label['GT_source_world'];cp=(world-g['camera_c2w'][:3,3])@g['camera_c2w'][:3,:3];inverse=a[y,x]/np.maximum(cp[:,2],1e-6)+b[y,x];candidate_valid=valid[y,x]&(inverse>1e-4)
        p=folder/'labels'/f'f{f:03d}'/'placement-support.npz'
        if not p.exists():np.savez_compressed(p,local_fit_supported=valid,candidate_placement_valid=candidate_valid,local_fit_count_requirement=np.int32(32),window_radius_px=np.int32(24))
        for c in e['cases']:
            q=np.load(folder/'labels'/f'f{f:03d}'/(c['id']+'.npz'));correct=q['content_hit_fraction']>0;rows.append(dict(id=c['id'],correct_candidates=int(correct.sum()),placement_valid_correct=int((correct&candidate_valid).sum()),placement_supported_fraction_of_correct=float((correct&candidate_valid).sum()/max(correct.sum(),1)),invalid_excluded_from_depth_loss=True,affine_inverse_floor_candidates=int((inverse<=1e-4).sum()),**stats))
    write(folder/'labels/placement-summary.json',dict(rows=rows,loss_support='contentmatchedray ANDcandidate_placement_valid;RGBalsoappearancevalid',unsupported_nearest_fit_depths_are_diagnostic_only=True))


def placement_safety(folder):
    """Flag local-affine targets conflicting with the fixed visible geometry.

    Do not clamp a contradictory label into a made-up depth or change content
    labels. Preserve the proposed target and exclude it from placement losses.
    The behind margin is the existing S1d R1 rule, not a GT-depth tolerance.
    """
    rows=[]
    for e in json.loads((folder/'prepared.json').read_text())['rows']:
        f=e['frame'];d=np.load(e['input_npz']);label=np.load(folder/'labels'/f'f{f:03d}'/(e['cases'][0]['id']+'.npz'));support=np.load(folder/'labels'/f'f{f:03d}'/'placement-support.npz');uv=label['candidate_GT_center_projection'];x=np.clip(np.floor(uv[:,0]).astype(int),0,511);y=np.clip(np.floor(uv[:,1]).astype(int),0,511);z=label['candidate_placement_depth'];visible=d['depth'][y,x];front=z<=visible+np.maximum(.005,.02*visible);inside=(uv[:,0]>=0)&(uv[:,0]<512)&(uv[:,1]>=0)&(uv[:,1]<512);safe=support['candidate_placement_valid']&inside&(label['GT_source_object_id']>0)&~front;p=folder/'labels'/f'f{f:03d}'/'placement-support-v2.npz'
        if not p.exists():np.savez_compressed(p,candidate_placement_valid=safe,local_fit_supported=support['local_fit_supported'],fixed_visible_front_conflict=front,previous_support=support['candidate_placement_valid'])
        for root in [folder/'labels',folder/'labels-training-focus']:
            for path in (root/f'f{f:03d}').glob('f*.npz'):
                q=np.load(path);correct=q['content_hit_fraction']>0;rows.append(dict(id=path.stem,training_focus=root.name.endswith('training-focus'),correct_candidates=int(correct.sum()),safe_placement_correct=int((correct&safe).sum()),safe_fraction_of_correct=float((correct&safe).sum()/max(correct.sum(),1)),front_conflict_fraction_of_correct=float((correct&front).sum()/max(correct.sum(),1)),invalid_targets_masked_not_clamped=True))
    write(folder/'labels/placement-safety-summary.json',dict(rows=rows,margin='sameS1dR1:max5mm,2%estimatedvisibledepth;notGTdepthtolerance',targets_and_content_labels_unchanged=True,loss_mask_uses_v2=True,initial_aligned_reference_without_this_guard_is_not_approved_training_supervision=True))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scene',required=True);p.add_argument('--mode',choices=['prepare','labels','training_labels','appearance','placement_support','placement_safety'],required=True);a=p.parse_args();cv2.setNumThreads(1);globals()[a.mode](OUT/a.scene)


if __name__=='__main__':main()
