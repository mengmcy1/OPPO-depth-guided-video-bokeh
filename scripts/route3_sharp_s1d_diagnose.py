"""GT-only diagnostic association and independent replacement datasets."""
import json
import os
import sys
from pathlib import Path
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
import cv2
import numpy as np
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from route3_sharp_s1d import OUT,PREFIX,CRITERIA,RECORDS,S1C,S1,UPPER,BASE,entries,write,FRAMES
from route3_upper_bound_evaluate import items,display,exr
from src.dual_layer import depth_band
LABELS=['unclassified','ring_miss','candidate_missing','motion_related','depth_infront_visible','depth_too_far','depth_too_near_behind_visible','color_source_edge','color_view_exposure_proxy','color_other','depth_color_correct']


def associate(parent,q,band):
    uv=parent['uv'];x=np.clip(uv[:,0].astype(int),0,511);y=np.clip(uv[:,1].astype(int),0,511);ids=np.flatnonzero(band[y,x])
    if not len(ids):return np.full(len(q),-1,np.int32)
    dist,idx=cKDTree(uv[ids]).query(q,workers=8);return np.where(dist<=1.5,ids[idx],-1).astype(np.int32)


def parent_flags(parent,pool,gtviews,dynamic):
    count=parent['contrib_count'];group=np.repeat(np.arange(len(count)),count);ids=parent['contrib_indices'];slots=pool['slot'][ids];xy=pool['source_xy'][ids];moving=np.zeros(len(ids),bool)
    for slot in np.unique(slots):
        at=slots==slot;g=gtviews[int(slot)];moving[at]=np.isin(g['object_id'][xy[at,1],xy[at,0]],dynamic)
    edge=np.bincount(group,weights=pool['edge'][ids],minlength=len(count))/np.maximum(count,1);motion=np.bincount(group,weights=moving,minlength=len(count))/np.maximum(count,1)
    return edge,motion


def appearance_support(parent,pool,assoc,q,z,obj,cam,K,gtviews,poses):
    n=len(assoc);fraction=np.zeros(n,np.float32);valid=np.flatnonzero(assoc>=0)
    if not len(valid):return fraction
    p=assoc[valid];count=parent['contrib_count'][p];ray=np.repeat(valid,count);start=np.repeat(parent['contrib_start'][p],count);offset=np.arange(len(ray))-np.repeat(np.cumsum(count)-count,count);ids=parent['contrib_indices'][start+offset];slots=pool['slot'][ids];sxy=pool['source_xy'][ids]
    world=(np.c_[q,np.ones(n)]@np.linalg.inv(K).T*z[:,None])@cam[:3,:3].T+cam[:3,3];matched=np.zeros(len(ray),bool)
    for slot in np.unique(slots):
        selected=np.flatnonzero(slots==slot);r=ray[selected];camera=(world[r]-poses[int(slot)][:3,3])@poses[int(slot)][:3,:3];uvz=camera@K.T;uv=uvz[:,:2]/np.maximum(uvz[:,2:],1e-12);x=sxy[selected,0];y=sxy[selected,1];gt=gtviews[int(slot)]
        matched[selected]=(np.linalg.norm(uv-(sxy[selected]+.5),axis=-1)<=1.5)&(gt['object_id'][y,x]==obj[r])&(abs(gt['depth_axial_m'][y,x]-camera[:,2])<=np.maximum(.02,.02*camera[:,2]))&~pool['edge'][ids[selected]]
    fraction=np.bincount(ray,weights=matched,minlength=n)/np.maximum(np.bincount(ray,minlength=n),1);return fraction


def replacement_parents(parent,gt,band):
    fields=['uv','depth','colors_linear'];p={k:parent[k].copy() for k in fields};gids=np.flatnonzero(gt['recoverable']);tree=cKDTree(gt['uv'][gids]);dist,idx=tree.query(p['uv'],k=16,workers=8);inside=idx<len(gids);chosen=gids[np.minimum(idx,len(gids)-1)];valid=inside&(dist<=1.5);z=np.where(valid,gt['depth'][chosen],np.inf);best=np.argmin(z,axis=1);ok=np.isfinite(z[np.arange(len(z)),best]);matched=chosen[np.arange(len(z)),best];uv=p['uv'];active=band[np.clip(uv[:,1].astype(int),0,511),np.clip(uv[:,0].astype(int),0,511)];ok&=active
    depth={k:v.copy() for k,v in p.items()};depth['depth'][ok]=gt['depth'][matched[ok]];color={k:v.copy() for k,v in p.items()};color['colors_linear'][ok]=gt['colors_linear'][matched[ok]]
    guv=gt['uv'];gactive=gt['recoverable']&band[np.clip(guv[:,1].astype(int),0,511),np.clip(guv[:,0].astype(int),0,511)];activeids=np.flatnonzero(active)
    nearest=cKDTree(p['uv'][activeids]).query(guv,workers=8)[0] if len(activeids) else np.full(len(guv),np.inf);missing=gactive&(nearest>1.5);added={k:np.concatenate([p[k],gt[k][missing]]) for k in fields}
    return dict(Oracle_depth_only=depth,Oracle_color_only=color,Oracle_missing_only=added),dict(active_R0_parents=int(active.sum()),GTmatched_replacement_parents=int(ok.sum()),GTmissing_parents_added=int(missing.sum()))


def main():
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    c=json.loads(CRITERIA.read_text());assert c['frozen'];root=OUT/'decomposition';root.mkdir(exist_ok=False);replacement=OUT/'replacements';replacement.mkdir();old=json.loads((RECORDS/'route3-sharp-s1c-2026-11-10-evaluation.json').read_text());refs=json.loads((ROOT/'outputs/route3-ldr-u1r-2026-11-10-01/references.json').read_text());refmap={x['id']:x['path'] for x in refs['rows']};geometry=json.loads((RECORDS/(PREFIX+'-geometry.json')).read_text())['rows'];gtviews=[np.load(BASE/'B2-high/geometry'/f'f{f:03d}-center.npz') for f in FRAMES];poses=[g['camera_c2w'] for g in gtviews];meta=np.load(ROOT/'outputs/route3-a2b-2026-11-10-01/blender-b2/B2-high/scene-geometry.npz');dynamic=meta['dynamic_object_ids'];rows=[];figures=[];failed=[]
    for case in items():
        frame=case['frame_id'];entry=next(e for e in entries() if e['frame']==frame);d=np.load(entry['input_npz']);lens=next(l for l in entry['cases'] if l['id']==case['id']);band=depth_band(d['depth'],d['K'][0,0],lens['focus_m'],case['aperture_radius_m'])[0];parent=np.load(OUT/'candidates'/f'f{frame:03d}'/'R0.npz');pool=np.load(OUT/'candidates'/f'f{frame:03d}'/'pool.npz');g=np.load(OUT/'geometry'/(case['id']+'.npz'));select=g['recoverable'];pixel=g['pixel'][select];q=g['uv'][select];z=g['depth'][select];obj=g['object_id'][select];color=g['GT_color'][select];colorvalid=g['GT_parent'][select]>=0;n=len(z);assoc=associate(parent,q,band);found=assoc>=0;safe=np.maximum(assoc,0);x=np.clip(q[:,0].astype(int),0,511);y=np.clip(q[:,1].astype(int),0,511);label=np.zeros(n,np.uint8);edge,motion=parent_flags(parent,pool,gtviews,dynamic);motion_flag=g['motion_related'][select]|(found&(motion[safe]>=.5));truevisible=gtviews[FRAMES.index(frame)]['depth_axial_m'][y,x];candz=parent['depth'][safe];delta=candz-z;front=found&(truevisible>0)&(candz<truevisible-np.maximum(.001,.005*truevisible));tau=np.maximum(.02,.02*z);colorerror=found&colorvalid&(abs(parent['colors_linear'][safe]-color).max(-1)>.05)
        ring=~band.flat[pixel]|~band[y,x];label[ring]=1;label[(label==0)&~found]=2;label[(label==0)&motion_flag]=3;label[(label==0)&front]=4;label[(label==0)&(delta>tau)]=5;label[(label==0)&(delta<-tau)]=6;label[(label==0)&colorerror&(edge[safe]>=.5)]=7
        need=(label==0)&colorerror;appearance=np.zeros(n,np.float32)
        if need.any():
            narrow=np.where(need,assoc,-1);appearance=appearance_support(parent,pool,narrow,q,z,obj,gtviews[FRAMES.index(frame)]['camera_c2w'],d['K'],gtviews,poses)
        label[(label==0)&colorerror&(appearance>=.5)]=8;label[(label==0)&colorerror]=9;label[(label==0)&colorvalid]=10
        unknown=int((label==0).sum());ratio=unknown/max(n,1);np.savez_compressed(root/(case['id']+'.npz'),pixel=pixel,sample=g['sample'][select],label=label,parent_index=assoc,motion_overlap=motion_flag,front_overlap=front,color_error_overlap=colorerror)
        counts=np.bincount(pixel,minlength=512*512);validpixels=counts>0;pred=np.load(old['predictions'][case['id']]['cal_E1__A_color_L3_cal_E1'])['prediction_srgb'];truth=display(exr(case['gt_rgb_exr']));ldr=np.load(refmap[case['id']])['reference_srgb'];categories=[];maps=[]
        for code,name in enumerate(LABELS):
            per=np.bincount(pixel[label==code],minlength=512*512);fraction=per/np.maximum(counts,1);scores={ref:float((fraction*((pred.astype(np.float64)-t.astype(np.float64))**2).sum(-1).reshape(-1)).sum()) for ref,t in [('Cycles',truth),('LDR',ldr)]}
            categories.append(dict(category=name,rays=int(per.sum()),fractional_pixel_mass=float(fraction.sum()),eligible_pixels=int(validpixels.sum()),any_ray_pixels=int((per>0).sum()),SSE=scores));maps.append(fraction.reshape(512,512))
        row=dict(id=case['id'],focus_m=case['focus_distance_m'],eligible_rays=n,eligible_pixels=int(validpixels.sum()),original_score_pixels=int(g['scoremask'].sum()),score_pixels_without_K64_rec=int((g['scoremask'].reshape(-1)&~validpixels).sum()),unclassified_rays=unknown,unclassified_fraction=ratio,categories=categories,geometry_unknown_rays=next(r['geometry_unknown_rays'] for r in geometry if r['id']==case['id']));rows.append(row)
        gt=np.load(BASE/'methods-3d/prepared'/f'B2-high-f{frame:03d}'/'parents.npz');parents,audit=replacement_parents(parent,gt,band);folder=replacement/case['id'];folder.mkdir()
        for name,p in parents.items():np.savez_compressed(folder/(name+'.npz'),**p)
        write(folder/'audit.json',audit)
        if frame in [1,30,60] and case['fstop']==1.4:
            fig,axes=plt.subplots(3,4,figsize=(14,10));axes.flat[0].imshow(pred);axes.flat[0].set_title('R0 prediction')
            for ax,image,name in zip(list(axes.flat)[1:],maps,LABELS):ax.imshow(image,vmin=0,vmax=1);ax.set_title(name,fontsize=8)
            for ax in axes.flat:ax.axis('off')
            fig.suptitle(case['id']+' conditional recoverable-ray class fractions');fig.tight_layout();path=root/(case['id']+'-classes.png');fig.savefig(path,dpi=130);plt.close(fig);figures.append(str(path))
        if ratio>.05:failed.append(case['id'])
        write(RECORDS/(PREFIX+'-decomposition-progress.json'),dict(completed=False,rows=rows,unclassified_failed=failed));print('S1D_CLASSIFY',case['id'],'unknown',ratio,flush=True)
        if failed:write(OUT/'STOP-unclassified.json',dict(id=case['id'],fraction=ratio,threshold=.05));raise RuntimeError('S1D_STOP unclassified rays exceed frozen5%')
    pooled={}
    for focus in [1.4,5.]:
        group=[r for r in rows if r['focus_m']==focus];pooled[str(focus)]={};npix=sum(r['eligible_pixels'] for r in group)
        for code,name in enumerate(LABELS):
            selected=[r['categories'][code] for r in group];pooled[str(focus)][name]=dict(rays=sum(r['rays'] for r in selected),fractional_pixel_share=sum(r['fractional_pixel_mass'] for r in selected)/npix,any_ray_pixel_share=sum(r['any_ray_pixels'] for r in selected)/npix,SSE_share={ref:sum(r['SSE'][ref] for r in selected)/sum(a['SSE'][ref] for row in group for a in row['categories']) for ref in ['Cycles','LDR']})
    write(RECORDS/(PREFIX+'-decomposition.json'),dict(completed=True,rows=rows,pooled=pooled,visualizations=figures,classification_commit='634b30f',pixel_SSE_frequency_attribution_not_unique_cause=True,GT_only_diagnostic=True))


if __name__=='__main__':main()
