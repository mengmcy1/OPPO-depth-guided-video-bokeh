"""Formal P1/P2 optical pairings and K16 ray attribution for appearance stage."""
import argparse
import time

import cv2
import numpy as np
import torch

import route3_ovg_data as data
from route3_ldr_u1r import model as lpips
from route3_ovg_1b_common import CONFIG, read
from route3_ovg_1b_optical_data import background_samples, geometry_root
from route3_ovg_1b_prime_evaluate import A, ORIGINAL, PRIME_EMA
from route3_ovg_1c_evaluate import case_frames, old_folder
from route3_ovg_refocus_query import cache_path
from route3_ovg_refocus_render import check_cache, merged_metrics
from route3_ovg_stage2_train import (OUT, Model, Q_MIN, guard, load_checkpoint, original_B,
                                    label_appearance, tensor)
from route3_ovg_subpixel import save_json, save_npz
from route3_g3_r14_train import learned_hidden
from route3_s2a_evaluate import masks
from route3_upper_bound_evaluate import display, exr, score
from src.aperture_renderer import aperture_disk
from src.footprint_order import apply_footprint_order
from src.layered_aperture import merge_layers
from src.ovg_appearance import boundary_candidate, boundary_layer, visible_layer
from src.ovg_leakage import query_contribution
from src.ovg_optics import fixed_visible_surfels, raster, subcamera
from src.prefix_transmittance import prefix_transmittance

SELECTED='1c-aligned-EMA'
TRUE='GT-visible-reference'
STAGES=('1c','2a','2b')
ATOL=dict(query_weight=.001,transmittance=.01)
FLOORS=dict(visible=50.,combined=60.)


def path_for(stamp, name, frame):
    return OUT/'evaluation'/stamp/name/f'f{frame:03d}'


def canonical_shapes(name, frame, y, case):
    root=geometry_root(name)
    center=data.load(root/'geometry'/f'f{frame:03d}-center.npz',
                     ('center_visible','object_id','depth_axial_m'))
    pixel_region=masks(root,case)
    boundary=y['depth_boundary'][0].cpu().numpy()
    footprint=cv2.dilate(boundary.astype(np.uint8),np.ones((5,5),np.uint8))>0
    opaque=(y['alpha'][0].cpu().numpy()>=.9999)&y['valid_alpha'][0].cpu().numpy()
    pixel_region['boundary_2px']=footprint
    pixel_region['interior']=opaque&~boundary
    return center,pixel_region


def formal_layer(query, item, manifest, folder, row, depth):
    case=item['case']
    band=data.load(folder/'R2-bands.npz',('__'.join((row['method'],case['protocol'],case['id'])),))
    band=next(iter(band.values()))
    path=str(cache_path(folder,row['method']))
    focus=row['实际对焦距离_m']
    active=check_cache(query.store,path,focus,case['fstop'],band,depth.cpu().numpy())
    spec=dict(item,input=path,case=dict(case,focus_m=focus))
    K=tensor(data.load(path,('target_K',))['target_K'])
    cloud=learned_hidden(query.network,query.store,spec,manifest,K,band)
    cloud,metadata=apply_footprint_order(cloud,depth,K)
    return cloud,dict(eligible_pixels=active,**metadata)


def diagnostic_leakage(visible, first, hidden, K, entry, case, center, regions, target):
    """Use the exact K16 rays and label focal plane for a single fixed comparison."""
    root=geometry_root(entry['scene'])
    labels=data.load(root/'geometry'/(case['id']+'-K16.npz'),
                     ('known','hidden','object_id','depth_axial_m','aperture_xy'))
    object_key=center['object_id']
    fg=data.load(root/'scene-geometry.npz',('foreground_object_ids',))['foreground_object_ids']
    same=(labels['object_id']==object_key[None])&labels['known']&~labels['hidden']
    region=regions['interior']&center['center_visible']&np.isin(object_key,fg)&same.all(0)
    region &= np.isfinite(labels['depth_axial_m']).all(0)&(labels['depth_axial_m']>0).all(0)
    if not region.any():
        return dict(pixels=0,rays=0,nominal_focus_m=case['nominal_focus_m']), {}
    focus=case['nominal_focus_m']
    offsets=tensor(labels['aperture_xy'].astype(np.float32))
    spec=dict(tiers='B',entry=entry)
    filler=background_samples(spec,K,focus,offsets,(512,512))
    base=merge_layers(visible,first)
    combined=merge_layers(base,hidden)
    rgb0=raster(base,K,focus,offsets,512,512,filler).cpu().numpy()[:,region]
    rgb1=raster(combined,K,focus,offsets,512,512,filler).cpu().numpy()[:,region]
    T,wq,w_front=[],[],[]
    for i,shift in enumerate(offsets):
        intr,view=subcamera(K,focus,shift)
        z=tensor(np.where(region,labels['depth_axial_m'][i],0).astype(np.float32))
        T.append(prefix_transmittance(visible,intr,view,(z*1.005).contiguous()).cpu().numpy()[region])
        q,front=query_contribution(combined,intr,view,z)
        wq.append(q.cpu().numpy()[region])
        w_front.append(front.cpu().numpy()[region])
    T,wq,w_front=map(np.stack,(T,wq,w_front))
    # The sequential attribution implementation is checked independently
    # against the actual renderer with a Query-indicator color.
    from dataclasses import replace
    first_ray=raster(replace(combined,colors=(combined.source_frames==1).float()[:,None].expand(-1,3).contiguous()),
                     K,focus,offsets[:1],512,512)[0,...,0].cpu().numpy()[region]
    discrepancy=float(np.abs(first_ray-wq[0]).max())
    if discrepancy>2e-4:
        raise ValueError(f'Query attribution differs from the gsplat indicator: {discrepancy}')
    d0,d1=rgb0.mean(0),rgb1.mean(0)
    s0,s1=display(d0),display(d1)
    truth=target[region]
    delta=((s1-truth)**2-(s0-truth)**2).sum(-1)
    linear=d1-d0
    # Secant of the display transform apportions the exact pixel SSE
    # change to rays, including clipping and cancellation across rays.
    secant=np.divide(s1-s0,linear,out=np.zeros_like(linear),where=abs(linear)>1e-12)
    effect=((rgb1-rgb0)*secant[None]*(s1+s0-2*truth)[None]/len(T)).sum(-1)
    if not np.allclose(effect.sum(0),delta,atol=2e-6,rtol=2e-3):
        raise ValueError('Per-ray SSE attribution fails conservation')
    ordered=w_front>=ATOL['query_weight']
    exposed=(T>=ATOL['transmittance'])&~ordered
    meaningful=wq>=ATOL['query_weight']
    # Priority is physical mis-order, then appreciable foreground leakage.
    # Remaining harmful observed contribution is the color comparison.
    selection=meaningful & (effect>0)
    conditions=(('front_order',selection&ordered),('surface_gap',selection&exposed),
                ('color_vs_F2',selection&~ordered&~exposed),('below_weight_or_nonharmful',~selection))
    groups=[dict(cause=label,rays=int(m.sum()),positive_SSE=float(np.maximum(effect[m],0).sum()),
                  net_SSE=float(effect[m].sum())) for label,m in conditions]
    def stats(x):
        return dict(mean=float(x.mean()),median=float(np.median(x)),p95=float(np.quantile(x,.95)),
                    max=float(x.max()))
    return dict(pixels=int(region.sum()),rays=int(region.sum())*len(T),nominal_focus_m=focus,
                q_indicator_max_abs=discrepancy,T_after_front=stats(T),query_weight=stats(wq),
                query_before_front=stats(w_front),delta_SSE=float(delta.sum()),classes=groups,
                no_hidden_SSE=float(np.square(s0-truth).sum()),with_hidden_SSE=float(np.square(s1-truth).sum()),
                significant_query_rays=int(meaningful.sum()),positive_SSE=float(np.maximum(effect,0).sum()),
                thresholds=ATOL),dict(mask=region,T=T,query_weight=wq,query_front=w_front,
                pixel_SSE_change=delta,per_ray_SSE_change=effect)


def load_geometry(name,frame,folder):
    prior=old_folder(folder)
    maps=data.load(folder/'visible-depths.npz',(SELECTED,TRUE))
    maps.update(data.load(prior/'visible-depths.npz',(ORIGINAL,PRIME_EMA)))
    return {k:np.where(v>0,v,1e4).astype(np.float32) for k,v in maps.items()}


def method_row(rows,method):
    return next(x for x in rows if x['method']==method)


@torch.no_grad()
def evaluate(stage):
    guard(gpu=True)
    torch.cuda.reset_peak_memory_stats()
    model=Model()
    if stage=='1c':
        variants=[('EMA',{k:v.detach().cpu().clone() for k,v in model.heads.state_dict().items()})]
    else:
        path=OUT/stage/f'step-{10000 if stage=="2a" else 30000:06d}.pt'
        ck=torch.load(path,map_location='cpu',weights_only=True)
        variants=[('EMA',ck['ema']),('raw',ck['model'])]
    metric=lpips()
    start=time.monotonic()
    rows, leaks, geometry_checks = [],[],[]
    for name,frame,entry,_,_,_,manifest,items,folder_1c in case_frames():
        guard()
        saved_focus=read(folder_1c/'focus.json')['rows']
        old=old_folder(folder_1c)
        past_focus=read(old/'R1.json')['rows']
        trial=original_B(name,frame)
        x,y,g,first=model.observations(trial,native=True)
        np.testing.assert_array_equal(g['depth1'][0].cpu().numpy(),
                                      data.load(folder_1c/'visible-depths.npz',(SELECTED,))[SELECTED])
        # The same frozen model and 1c prior mapping are exercised by every endpoint.
        geometry_checks.append(dict(scene=name,frame=frame,mu1_matches_1c=True))
        K=x['K'][0]
        zero=K.new_zeros((1,2))
        spec=dict(tiers='B',entry=entry)
        zero_fill=background_samples(spec,K,1.4,zero,(512,512))
        pinhole=data.load(entry['input_npz'],('rgb_srgb',))['rgb_srgb']
        maps=load_geometry(name,frame,folder_1c)
        official_rgb=label_appearance('B',name,trial[2])
        og_bg=tensor(official_rgb['background'])
        original_cloud=fixed_visible_surfels(dict(depth1=tensor(maps[ORIGINAL])[None]),x)
        true_cloud=fixed_visible_surfels(dict(depth1=tensor(maps[TRUE])[None]),x)
        heads={}
        for weight,state in variants:
            model.heads.load_state_dict(state)
            ap=model.appearance(g,x,first)
            candidate=boundary_candidate(ap)
            b=candidate&(tensor(first['q'])>=Q_MIN)
            bgt=candidate&tensor(official_rgb['comp_valid'])
            heads[weight]=(ap,b,bgt)
        for item in items:
            case=item['case']
            cid,protocol,fstop=case['id'],case['protocol'],case['fstop']
            path=path_for(stage,name,frame)/protocol/cid/'case.json'
            if path.exists():
                prev=read(path)
                rows.extend(prev['rows'])
                leaks.extend(prev.get('leakage',[]))
                continue
            root=geometry_root(name)
            target=display(exr(root/'dof'/(cid+'.exr')))
            center,regions=canonical_shapes(name,frame,y,case)
            zero_masks={k:regions[k] for k in ('full_image','interior','boundary_2px')}
            scoring={k:regions[k] for k in ('full_image','in_focus','recoverable','boundary_2px')}
            primary=protocol=='P2' and fstop==1.4
            case_focus=[r for r in saved_focus if r['id']==cid and r['protocol']==protocol]
            plan=method_row(case_focus,SELECTED)
            if plan['失败']:
                raise ValueError(f'1c official focus unavailable: {name} {cid} {protocol}')
            focus=plan['实际对焦距离_m']
            offsets=aperture_disk(64,.05/(2*fstop),device='cuda',dtype=K.dtype)
            second,place= formal_layer(model.query,item,manifest,folder_1c,plan,g['depth1'][0])
            f2=background_samples(spec,K,focus,offsets,(512,512))
            local_rows,local_leaks=[],[]
            common=dict(stage=stage,scene=name,frame=frame,id=cid,protocol=protocol,aperture=fstop,
                        nominal_focus_m=case['nominal_focus_m'])

            def paired(method,weights,split_type,vis,front_layer,query_layer,focus_here,fill,
                       split_fraction=0.,fraction_invalid=0.):
                total=merge_layers(vis,front_layer)
                z_visible=raster(vis,K,focus_here,zero,512,512)[0]
                zstats=score(display(z_visible.cpu().numpy()),pinhole,zero_masks,metric)
                for second_on in (False,True):
                    sc=merge_layers(total,query_layer if second_on else None)
                    prediction=raster(sc,K,focus_here,offsets,512,512,fill).mean(0)
                    z=raster(sc,K,focus_here,zero,512,512,zero_fill)[0]
                    im,zero_im,values=merged_metrics(prediction,z,target,pinhole,scoring,zero_masks,metric)
                    values.update({'zero_visible_'+k:v for k,v in zstats.items()})
                    local_rows.extend(dict(common,method=method,weights=weights,split=split_type,
                        second=second_on,region=region,actual_focus_m=focus_here,b=split_fraction,
                        false_transparent=fraction_invalid,**v) for region,v in values.items())
                    save_npz(path.parent/f'{method}-{weights}-{split_type}-Q{int(second_on)}.npz',
                             rgb=im,zero=zero_im)

            for weight,(ap,b,bgt) in heads.items():
                if weight=='raw' and not primary:
                    continue
                false=float((ap['alpha'][0]<.95)[tensor(regions['interior'])].float().mean())
                empty=torch.zeros_like(b)
                modes=['none','oracle','retrieved'] if weight=='EMA' else ['retrieved']
                for split_type in modes:
                    mask={'none':empty,'oracle':bgt,'retrieved':b}[split_type]
                    vis=visible_layer(ap,x,mask)
                    if split_type=='none' or stage=='1c':
                        front_layer=None
                    elif split_type=='oracle':
                        front_layer=boundary_layer(ap,x,mask,og_bg,ap['depth2'][0])
                    else:
                        front_layer=boundary_layer(ap,x,mask,tensor(first['color']),tensor(first['depth']))
                    if stage=='1c' and split_type!='none':
                        # Zero exposure and unchanged surfaces imply exact identity.
                        n=len(local_rows)
                        base_rows=[r for r in local_rows if r['method']=='OVG' and r['weights']==weight and r['split']=='none']
                        local_rows[n:n]=[dict(r,split=split_type,b=float(mask.float().mean())) for r in base_rows]
                    else:
                        paired('OVG',weight,split_type,vis,front_layer,second,focus,f2,float(mask.float().mean()),false)
                    if primary and (stage,weight,split_type) in (('1c','EMA','none'),('2b','EMA','retrieved')):
                        stats,rays=diagnostic_leakage(vis,front_layer,second,K,entry,case,center,regions,target)
                        local_leaks.append(dict(common,method=stage,**stats))
                        save_npz(path.parent/f'leakage-{stage}.npz',**rays)

            if stage=='1c':
                old_rows=[r for r in past_focus if r['id']==cid and r['protocol']==protocol]
                for method,anchor,vis in ((ORIGINAL,ORIGINAL,original_cloud),(TRUE,PRIME_EMA,true_cloud)):
                    row=method_row(case_focus,method)
                    row_old=method_row(old_rows,anchor)
                    depth=tensor(maps[anchor])
                    query,_=formal_layer(model.query,item,manifest,old,row_old,depth)
                    local_focus=row['实际对焦距离_m']
                    fill=background_samples(spec,K,local_focus,offsets,(512,512))
                    paired(method,'fixed','none',vis,None,query,local_focus,fill)
                    if primary and method==ORIGINAL:
                        stats,rays=diagnostic_leakage(vis,None,query,K,entry,case,center,regions,target)
                        local_leaks.append(dict(common,method=method,**stats))
                        save_npz(path.parent/f'leakage-{method}.npz',**rays)
                baseline_focus=method_row(case_focus,A)['实际对焦距离_m']
                baseline=background_samples(spec,K,baseline_focus,offsets,(512,512)).mean(0)
                im,z,values=merged_metrics(baseline,zero_fill[0],target,pinhole,scoring,zero_masks,metric)
                local_rows.extend(dict(common,method=A,weights='fixed',split='standalone',second=False,region=k,
                    actual_focus_m=baseline_focus,b=0.,false_transparent=0.,**v) for k,v in values.items())
            save_json(path,dict(rows=local_rows,leakage=local_leaks,order=place))
            rows.extend(local_rows)
            leaks.extend(local_leaks)
            print('STAGE2_EVAL',stage,name,frame,cid,protocol,flush=True)
        model.query.store.inputs.cache_clear()
        model.query.store.active.cache_clear()
    if stage=='2b':
        from route3_ovg_1b_train import evaluate as geometry_measure
        from route3_ovg_1c import aligned_sample, OUT as START
        before=read(START/'train/aligned/validation/step-005000.json')['EMA']
        current=geometry_measure(model.backbone,30000,fixed=True,sample=aligned_sample)
        selectors=('high_conf_single_prior','high_conf_multiple_priors')
        fields=('front_n','edge_front_n','front_CoC_mae_px','boundary_front_CoC_mae_px')
        def profile(report):
            return {(r['scene'],r['subset'],r['method'],r['focus_m']):{k:r[k] for k in fields}
                    for r in report['rows'] if r['subset'] in selectors and r['method']!='OVG-Net-gate-off'}
        start_values,end_values=profile(before),profile(current)
        if start_values != end_values:
            raise ValueError('High confidence frozen boundary geometry differs from 1c')
        save_json(OUT/'S3-frozen-boundary.json',dict(passed=True,
                  rows=[dict(scene=k[0],subset=k[1],method=k[2],nominal_focus_m=k[3],**v)
                        for k,v in end_values.items()]))
    result=dict(rows=rows,leakage=leaks,geometry_checks=geometry_checks,seconds=time.monotonic()-start,
                peak_MiB=torch.cuda.max_memory_allocated()/2**20,
                truth_visibility_ray_focus='K16 exact nominal-focus pupil projection; OVG evaluation otherwise official P2/P1',
                GT_reference='same placed second-pass layer as 1b′ EMA, as in 1c',
                boundary_oracle='at predicted boundary candidates with valid captured B_gt; depth anchor unchanged')
    save_json(OUT/f'S3-{stage}.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=STAGES)
    evaluate(p.parse_args().stage)
