"""Appendix10 exact-ray prefix statistics, factored by row/hidden layer.

GT labels identify the hidden surface. A same-object nearest parent gives its
actual raw/aligned placement. We query the continuous ray through that placed
parent, avoiding the focus/depth-induced ray mismatch of fixed pixel queries.
All GT operations here are diagnostics, absent from method-side masks/images.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from route3_local_replacement import (OUT,SCENES,ROWS,COLS,desc_for,frame_dir,case_dir,
    load,write,save,tensor,guard,hidden_layers,reference,visible_cloud)
from src.local_gaussian_replacement import projection,ray_query_plan,blocker_ray_queries

def true_parent_world(parents,mesh,frame):
    if 'world_position' in parents:return parents['world_position']
    result=np.zeros((len(parents['depth']),3),np.float32)
    for oi,oid in enumerate(mesh['object_ids']):
        at=parents['object_id']==oid;matrix=mesh['object_world_matrices'][frame-1,oi]
        result[at]=parents['local_surface_point'][at]@matrix[:3,:3].T+matrix[:3,3]
    return result

def ray_labels(name,desc,entry,case,parents,alignment):
    dest=frame_dir(name,entry['frame']);path=dest/(case['id']+'-ray-reference.npz')
    if path.exists():return load(path)
    f=entry['frame'];mesh=np.load(desc['scene']/'scene-geometry.npz');fg=mesh['foreground_object_ids']
    key=f'f{f:03d}-N{case["fstop"]:g}-F{case["nominal_focus_m"]:g}-K16.npz';g=np.load(desc['geometry']/key)
    _,regions=reference(name,desc,case,f);objects=g['object_id'];raw_depth=g['depth_axial_m'];hidden=g['hidden']&g['recoverable']&~g['residual_pixels'][None]&regions['recoverable'][None]
    lens,pixel_y,pixel_x=np.nonzero(hidden);obj=objects[lens,pixel_y,pixel_x];uv=g['reference_uv_center'][lens,pixel_y,pixel_x];z=raw_depth[lens,pixel_y,pixel_x];rays=np.c_[uv,np.ones(len(uv))]@np.linalg.inv(g['K']).T;cam=rays*z[:,None];world=cam@g['camera_c2w'][:3,:3].T+g['camera_c2w'][:3,3]
    wp=true_parent_world(parents,mesh,f);index=np.full(len(z),-1,np.int32);distance=np.full(len(z),np.inf)
    for oid in np.unique(obj):
        ids=np.flatnonzero((parents['object_id']==oid)&parents['recoverable']);at=np.flatnonzero(obj==oid)
        if not len(ids):continue
        dist,nearest=cKDTree(wp[ids]).query(world[at],workers=4);ok=dist<=3*z[at]/g['K'][0,0]+1e-4;index[at[ok]]=ids[nearest[ok]];distance[at[ok]]=dist[ok]
    good=index>=0;index=index[good];lens=lens[good];pixel_y=pixel_y[good];pixel_x=pixel_x[good];rdepth=parents['depth'][index];adepth=alignment['parent_depth_aligned'][index];av=alignment['parent_valid'][index]
    count_map=np.zeros((16,512,512),bool);count_map[lens,pixel_y,pixel_x]=True
    # Control rays go towards the hidden plane from other K16 rays at the
    # same canonical pixel, rather than stopping at the foreground itself.
    plane_raw=np.full((16,512,512),np.nan,np.float32);plane_aligned=plane_raw.copy();plane_raw[lens,pixel_y,pixel_x]=rdepth;plane_aligned[lens[av],pixel_y[av],pixel_x[av]]=adepth[av]
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning);pr=np.nanmedian(plane_raw,axis=0);pa=np.nanmedian(plane_aligned,axis=0)
    control=np.isin(objects,fg)&~g['hidden']&regions['recoverable'][None]&np.isfinite(pr)[None]
    ck,cy,cx=np.nonzero(control)
    result=dict(hidden_lens=lens.astype(np.uint8),hidden_y=pixel_y.astype(np.int16),hidden_x=pixel_x.astype(np.int16),parent_index=index,raw_depth=rdepth,aligned_depth=adepth,aligned_valid=av,control_lens=ck.astype(np.uint8),control_y=cy.astype(np.int16),control_x=cx.astype(np.int16),control_raw_depth=pr[cy,cx],control_aligned_depth=pa[cy,cx],aperture_xy=g['aperture_xy'],raw_label_count=np.int64(len(good)),matched_label_count=np.int64(good.sum()),matched_distance_m=distance[good].astype(np.float32))
    save(path,**result);return result

def query_for(lab,k,plane,parents,K,xy,focus,nominal):
    take=np.flatnonzero(lab['hidden_lens']==k)
    if plane=='aligned':take=take[lab['aligned_valid'][take]]
    ids=lab['parent_index'][take];z=lab[plane+'_depth'][take];central=parents['uv'][ids]
    # The actual parent center after placement: x/z,y/z unchanged by alignment.
    uv=central+np.diag(K)[:2]*xy*(1/focus-1/z[:,None])
    control=np.flatnonzero((lab['control_lens']==k)&np.isfinite(lab['control_'+plane+'_depth']))
    cz=lab['control_'+plane+'_depth'][control]
    cuv=np.c_[lab['control_x'][control]+.5,lab['control_y'][control]+.5]+np.diag(K)[:2]*xy*(1/focus-1/nominal)
    # Controls preserve the nominal physical foreground-blocked ray. Their
    # stopping depth is the placed hidden plane from other aperture samples.
    alluv=np.concatenate([uv,cuv]).astype(np.float32);limits=np.r_[z,cz].astype(np.float32)
    return alluv,limits,len(take),len(control)

def run(names):
    guard(True)
    for name in names:
      desc=desc_for(name)
      for entry in desc['entries']:
        f=entry['frame'];dest=frame_dir(name,f);d=load(entry['input_npz']);K=tensor(d['K']);cloud=visible_cloud(desc,entry);parents=load(desc['parents'](f));retrieved=load(dest/'retrieval.npz');alignment=load(dest/'alignment.npz')
        for rule in ['P2','P1']:
          for case in entry['cases']:
            folder=case_dir(name,f,case,rule);done=folder/'transmittance.json'
            if done.exists():continue
            if not (folder/'render.json').exists():continue
            guard();meta=json.loads((folder/'render.json').read_text());masks=load(folder/'support-masks.npz');focus=meta['focus_m'];lab=ray_labels(name,desc,entry,case,parents,alignment);hidden,_=hidden_layers(parents,retrieved,alignment,masks['band'],K)
            states={'full':None,**{s:tensor(masks[s]) for s in ['oracle_R1','oracle_R2','method_R1','method_R2']}}
            buffers={};domain=[]
            for k,xy in enumerate(lab['aperture_xy']):
                E=torch.eye(4,device='cuda');E[:2,3]-=tensor(xy);intr=K.clone();intr[:2,2]+=K.diag()[:2]*tensor(xy)/focus
                visible_projection=projection(cloud,intr,E)
                hidden_projections={col:projection(h,intr,E) for col,h in hidden.items() if h is not None}
                for plane in ['aligned','raw']:
                    uv,limit,nh,nc=query_for(lab,k,plane,parents,d['K'],xy,focus,case['nominal_focus_m']);plan=ray_query_plan(tensor(uv));valid=plan[-1].cpu().numpy();valid_h=valid[:nh];valid_c=valid[nh:];limits=tensor(limit);x=np.clip(uv[:,0].astype(int),0,511);y=np.clip(uv[:,1].astype(int),0,511);inband=masks['band'][y,x]&valid
                    domain.append(dict(plane=plane,lens=k,hidden_queries=nh,hidden_inside=int(valid_h.sum()),control_queries=nc,control_inside=int(valid_c.sum())))
                    def append(state,array):
                        a=array.cpu().numpy();buffers.setdefault(plane+'__'+state+'__hidden',[]).append(a[:,np.r_[valid_h,np.zeros(nc,bool)]].T)
                        buffers.setdefault(plane+'__'+state+'__control',[]).append(a[:,np.r_[np.zeros(nh,bool),valid_c]].T)
                    for state,drop in states.items():append(state,blocker_ray_queries(cloud,visible_projection,limits,plan,drop))
                    for col,h in hidden.items():
                        if h is None:part=torch.zeros((3,len(uv)),device='cuda')
                        else:
                            hh=blocker_ray_queries(h,hidden_projections[col],limits,plan);part=(hh[:3]+hh[3:])*tensor(inband)[None]
                        append(col,part)
            arrays={key:np.concatenate(value) for key,value in buffers.items()};path=folder/'prefix-components.npz';save(path,**arrays)
            write(done,dict(scene=name,frame=f,id=case['id'],nominal_focus_m=case['nominal_focus_m'],fstop=case['fstop'],rule=rule,path=str(path),domain=domain,raw_label_rays=int(lab['raw_label_count']),matched_parent_rays=int(lab['matched_label_count']),aligned_valid_parent_rays=int(lab['aligned_valid'].sum()),reference='raw and aligned placed-parent center,continuous projected ray;no pixel rounding;control keeps true foreground-blocked nominal ray',visible_prefix='six bins:SHARP layer0/1 x<=5%,5-20%,>20%;no early termination',replacement_prefix='three additional bins,stored separately;visible-only and total merged prefixes both reported',same_row_support='oracle fornone,GTraw,GTaligned;method forretrieval',inference_GT_leak=False,statistic_GT_only=True))
            print('X_PREFIX',name,f,rule,case['id'],len(arrays['aligned__full__hidden']),flush=True)
            del hidden,buffers,arrays;torch.cuda.empty_cache()
        del cloud;torch.cuda.empty_cache()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scene',choices=SCENES);a=p.parse_args();run([a.scene] if a.scene else SCENES)
