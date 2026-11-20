"""One-off P2 pairing and K16 attribution for b=0 opacity (13.16.6)."""
import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import torch

from route3_ovg_stage2_evaluate import (
    OUT, DIAGNOSTICS, FLOORS, Model, ORIGINAL, Q_MIN, SELECTED, aperture_disk,
    background_samples, boundary_candidate, boundary_layer, canonical_shapes,
    case_frames, data, display, exr, fixed_visible_surfels, formal_layer,
    geometry_root, guard, leakage_region, load_geometry, merge_layers, method_row,
    old_folder, original_B, path_for, pixel_layer, prefix_transmittance, raster,
    ray_effect, read, save_json, save_npz, subcamera, tensor, visible_layer,
)
from route3_ovg_1b_common import REC
from route3_upper_bound_evaluate import region_sse
from src.ovg_opaque_eval import opaque_view, render, subpixel_flags

DEST = OUT/'opaque-b0'
PREFIX = 'route3-ovg-stage2-2026-11-20-opaque-b0'
METHODS = ('1c', '2a', '2b', ORIGINAL)
RAY_CLASSES = ('front_order', 'T<0.02', '0.02<=T<0.1', 'T>=0.1', 'color_vs_F2')
REGIONS = ('full_image', 'in_focus', 'recoverable')


@dataclass
class Layers:
    depth: torch.Tensor
    b0: torch.Tensor
    flags: torch.Tensor
    visible: object
    first: object
    query: object

    def scenes(self):
        base=merge_layers(self.visible,self.first)
        return base,merge_layers(base,self.query),len(base.means)


def make_layers(x, g, ap, first, name, query, original=None):
    if original is None:
        b=boundary_candidate(ap)&(tensor(first['q'])>=Q_MIN)
        depth=ap['depth1'][0]
        visible=visible_layer(ap,x,b)
        front=None if name=='1c' else boundary_layer(ap,x,b,tensor(first['color']),tensor(first['depth']))
        valid=torch.isfinite(depth)&(depth>0)&torch.isfinite(x['rgb'][0]).all(0)
        b0=valid&~b
    else:
        depth=original
        visible=fixed_visible_surfels(dict(depth1=depth[None]),x)
        front=None
        valid=torch.isfinite(depth)&(depth>0)&torch.isfinite(x['rgb'][0]).all(0)
        b0=valid
    flags=subpixel_flags(valid,b0,5)
    if len(flags)!=len(visible.means):
        raise ValueError('b=0 flags must follow the original visible subpixel order')
    return Layers(depth,b0,flags,visible,front,query)


def old_image_dir(method, name, frame, case):
    split='none' if method=='1c' else 'retrieved'
    return path_for(method,name,frame)/case['protocol']/case['id'],split


def appearance_states(model):
    init={k:v.detach().cpu().clone() for k,v in model.heads.state_dict().items()}
    return {'1c':init,**{m:torch.load(OUT/m/f'step-{n:06d}.pt',map_location='cpu',weights_only=True)['ema']
                          for m,n in (('2a',10000),('2b',30000))}}


def frame_models(model, states, trial):
    x,y,g,first=model.observations(trial,native=True)
    predictions={}
    for stamp,state in states.items():
        model.heads.load_state_dict(state)
        predictions[stamp]=model.appearance(g,x,first)
    return x,y,g,first,predictions


def optics(layers, K, focus, stop, spec, pinhole, target, regions, closed):
    zero=K.new_zeros((1,2))
    offsets=aperture_disk(64,.05/(2*stop),device=K.device,dtype=K.dtype)
    zfill=background_samples(spec,K,focus,zero,(512,512))
    fill=background_samples(spec,K,focus,offsets,(512,512))
    base,merged,q0=layers.scenes()
    def evaluate(scene, rays, bg=None):
        return render(scene,K,focus,rays,layers.depth,layers.b0,layers.flags,q0,bg,opaque_b0=closed)
    zd=display(evaluate(layers.visible,zero)[0].cpu().numpy())
    zeros=dict(visible_only=region_sse(zd,pinhole,regions['full_image']))
    images={'visible_zero':zd}
    rows=[]
    for state,scene in ((False,base),(True,merged)):
        rgb=display(evaluate(scene,offsets,fill).mean(0).cpu().numpy())
        z=display(evaluate(scene,zero,zfill)[0].cpu().numpy())
        images.update({f'rgb_{int(state)}':rgb,f'zero_{int(state)}':z})
        rows.append(dict(second_query=state,
            regions={k:region_sse(rgb,target,regions[k]) for k in REGIONS},
            zero_combined=region_sse(z,pinhole,regions['full_image'])))
    return dict(zero=zeros,pairs=rows),images


def bins(wq,zq,zv,T,effect):
    """A precedence partition; non-order finite T rays enter the three bins."""
    harmful=(wq>=.001)&(effect>0)
    order=harmful&(zv>0)&(zq<zv)
    usable=harmful&~order&np.isfinite(T)
    masks=(order,usable&(T<.02),usable&(T>=.02)&(T<.1),usable&(T>=.1),harmful&~order&~usable)
    if sum(int(m.sum()) for m in masks)!=int(harmful.sum()):
        raise ValueError('Leakage bins do not partition the harmful ray population')
    return [dict(category=c,rays=int(m.sum()),positive_SSE=float(effect[m].sum(dtype=np.float64)))
            for c,m in zip(RAY_CLASSES,masks)]


def old_rays(stage, name, frame, case, method, region):
    with np.load(DIAGNOSTICS/stage/name/f'f{frame:03d}'/case['id']/f'leakage-{method}.npz') as f:
        np.testing.assert_array_equal(f['mask'],region)
        return dict(wq=f['query_weight'],zq=f['query_depth_m'],zv=f['dominant_visible_depth_m'],
                    T=f['T'],effect=f['per_ray_SSE_change'],
                    inverse=f['dominant_visible_inverse_depth_minus_GT_inv_m'])


def leakage(layers, K, entry, case, center, regions, target, legacy):
    labels,region=leakage_region(entry,case,center,regions)
    if not region.any():
        return dict(domain_rays=0,closed_rays=0,off=[],on=[]),{}
    old=old_rays(*legacy,region)
    focus=case['nominal_focus_m']  # The K16 label rays use the nominal focal plane.
    rays=tensor(labels['aperture_xy'].astype(np.float32))
    fill=background_samples(dict(tiers='B',entry=entry),K,focus,rays,(512,512))
    base,merged,q0=layers.scenes()
    rgb0,rgb1,qws,qzs,ts,closed=[],[],[],[],[],[]
    mask=tensor(region)
    for i,o in enumerate(rays):
        args=(K,focus,o,layers.depth,layers.b0,layers.flags,q0,fill[i])
        base_im,_=opaque_view(base,*args)
        full_im,d=opaque_view(merged,*args)
        rgb0.append(base_im[mask].cpu().numpy())
        rgb1.append(full_im[mask].cpu().numpy())
        close=d['closed'][mask].cpu().numpy()
        wq=np.where(close,d['query_weight'][mask].cpu().numpy(),old['wq'][i])
        zq=np.where(close,d['query_depth'][mask].cpu().numpy(),old['zq'][i])
        # For an unclosed ray, the ordinary path and its optical weights
        # are unchanged. A surviving Query on a closed ray is in the prefix.
        if np.any(close&(wq>=.001)):
            cutoff=torch.zeros_like(layers.depth)
            cutoff[mask]=tensor(np.nan_to_num(zq,nan=0.).astype(np.float32))
            intr,view=subcamera(K,focus,o)
            T=prefix_transmittance(layers.visible,intr,view,cutoff)[mask].cpu().numpy()
            T=np.where(close,T,old['T'][i])
        else:
            T=np.where(close,np.nan,old['T'][i])
        T=np.where(wq>0,T,np.nan)
        qws.append(wq)
        qzs.append(zq)
        ts.append(T)
        closed.append(close)
    _,effect,_,_=ray_effect(np.stack(rgb0),np.stack(rgb1),target[region])
    wq,zq,T,closed=map(np.stack,(qws,qzs,ts,closed))
    zr=np.nan_to_num(old['zv'],nan=0.)
    old_bins=bins(old['wq'],old['zq'],zr,old['T'],old['effect'])
    new_bins=bins(wq,zq,zr,T,effect)
    return dict(domain_rays=int(wq.size),closed_rays=int(closed.sum()),off=old_bins,on=new_bins,
                off_total_positive_SSE=float(np.maximum(old['effect'],0).sum(dtype=np.float64)),
                on_total_positive_SSE=float(np.maximum(effect,0).sum(dtype=np.float64))),dict(
        mask=region,closed=closed,T=T,query_weight=wq,query_depth_m=zq,dominant_visible_depth_m=old['zv'],
        dominant_visible_inverse_depth_minus_GT_inv_m=old['inverse'],per_ray_SSE_change=effect)


@torch.no_grad()
def toy():
    size=40
    K=tensor(np.array([[40,0,20],[0,40,20],[0,0,1]],np.float32))
    color=K.new_full((size,size,3),.2)
    depth=K.new_full((size,size),2.)
    sites=[(20,5),(20,17),(20,31)]
    cases=torch.zeros_like(depth,dtype=torch.bool)
    for y,x in sites:
        cases[y,x]=True
    valid=cases.clone();valid[sites[-1]]=False
    b0=torch.zeros_like(cases);b0[sites[0]]=True
    I=torch.eye(4,device=K.device)
    front=pixel_layer(color,depth,K,I,valid,sigma_pixels=1.,opacity=.9,subdivisions=1)
    query=pixel_layer(color+.6,depth+1,K,I,cases,1,sigma_pixels=1.,opacity=.75,subdivisions=1)
    full=merge_layers(front,query)
    offsets=K.new_zeros((1,2))
    bg=color[None]*.3
    flags=subpixel_flags(valid,b0,1)
    kwargs=(depth,b0,flags,len(front.means))
    old=raster(full,K,2.,offsets,size,size,bg)[0]
    off=render(full,K,2.,offsets,*kwargs,bg,opaque_b0=False)[0]
    on,meta=opaque_view(full,K,2.,offsets[0],*kwargs,bg[0])
    alone,_=opaque_view(front,K,2.,offsets[0],*kwargs,bg[0])
    assert torch.equal(old,off)
    assert [bool(meta['closed'][p]) for p in sites]==[True,False,False]
    assert float(meta['query_weight'][sites[0]])==0.
    assert torch.equal(on[sites[0]],alone[sites[0]])
    torch.testing.assert_close(on[sites[0]],color[sites[0]],rtol=0,atol=1e-6)
    assert torch.equal(on[sites[1]],old[sites[1]]) and torch.equal(on[sites[2]],old[sites[2]])
    return dict(pass_=True,disable_bitwise=True,b0_query_weight=0.,b1_max_abs=0.,hole_max_abs=0.)


def write_csv(path, rows):
    cols=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        wr=csv.DictWriter(f,cols,lineterminator='\n')
        wr.writeheader()
        wr.writerows(rows)


def image_agreement(a,b):
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Invalid migration image')
    delta=a.astype(np.float64)-b
    return dict(max_abs=float(np.abs(delta).max()),
                psnr_db=float(-10*np.log10(max(float(np.square(delta).mean()),1e-16))))


def compare_old(method, name, frame, case, stats, images):
    root,split=old_image_dir(method,name,frame,case)
    old=read(root/'case.json')['rows']
    visible_old=next(r for r in old if r['method']=='OVG' and r['weights']=='EMA' and
                     r['split']==split and not r['second'] and r['region']=='zero_visible_full_image')
    visible_new=stats['zero']['visible_only']['psnr']
    comparisons=[dict(region='visible_zero_score',old_dB=visible_old['psnr'],new_dB=visible_new,
         abs_score_difference_dB=abs(visible_old['psnr']-visible_new),
         pixel_reference='old evaluation saved this score without a separate visible-only image')]
    for q in (0,1):
        with np.load(root/f'OVG-EMA-{split}-Q{q}.npz') as saved:
            for k in ('rgb','zero'):
                comparisons.append(dict(region=f'{k}_q{q}',**image_agreement(images[f'{k}_{q}'],saved[k])))
    comparisons[0]['passed']=comparisons[0]['abs_score_difference_dB']<=.1
    for r in comparisons[1:]:
        r['passed']=r['psnr_db']>=48.
    return comparisons


def formal_inputs(model, name, frame, folder, manifest, item, g, x, y, entry):
    case=item['case']
    center,regions=canonical_shapes(name,frame,y,case)
    target=display(exr(geometry_root(name)/'dof'/(case['id']+'.exr')))
    pinhole=data.load(entry['input_npz'],('rgb_srgb',))['rgb_srgb']
    rows=read(folder/'focus.json')['rows']
    focus_row=method_row([r for r in rows if r['id']==case['id'] and r['protocol']=='P2'],SELECTED)
    if focus_row['失败']:
        raise ValueError(f'No method focus for {case["id"]}')
    # The actual front is recomputed; the frozen Query cache, projection and
    # footprint order use the exact depth snapshot with which they were made.
    # Different CUDA architectures can differ at the last bits in depth.
    cached=data.load(folder/'visible-depths.npz',(SELECTED,))[SELECTED]
    predicted=g['depth1'][0].cpu().numpy()
    if predicted.shape!=cached.shape or not np.isfinite(predicted).all() or not (predicted>0).all():
        raise ValueError('New visible depth is invalid for the frozen Query geometry')
    query,_=formal_layer(model.query,item,manifest,folder,focus_row,tensor(cached))
    return center,regions,target,pinhole,focus_row['实际对焦距离_m'],query


@torch.no_grad()
def migration_optics(dest):
    guard(gpu=True)
    model=Model()
    states=appearance_states(model)
    states.pop('2a')
    comparisons=[]
    for name,frame,entry,_,_,_,manifest,items,folder in case_frames():
        if name!='metric-scene-122104-broad':
            continue
        trial=original_B(name,frame)
        x,y,g,first,aps=frame_models(model,states,trial)
        frozen=data.load(folder/'visible-depths.npz',(SELECTED,))[SELECTED]
        depth=g['depth1'][0].cpu().numpy()
        relative=np.abs(depth.astype(np.float64)/frozen-1)
        for item in items:
            case=item['case']
            if (case['protocol'],case['fstop'])!=('P2',1.4):
                continue
            _,regions,target,pinhole,focus,query=formal_inputs(model,name,frame,folder,manifest,item,g,x,y,entry)
            for method,ap in aps.items():
                if method=='1c' and (frame,case['nominal_focus_m'])!=(1,1.4):
                    continue
                out=dest/'optical'/method/f'f{frame:03d}'/case['id']
                if (out/'off.json').is_file() and (out/'off.npz').is_file():
                    stats=read(out/'off.json')
                    with np.load(out/'off.npz') as f:images={k:f[k] for k in f.files}
                else:
                    layers=make_layers(x,g,ap,first,method,query)
                    stats,images=optics(layers,x['K'][0],focus,1.4,dict(tiers='B',entry=entry),pinhole,target,regions,False)
                    save_npz(out/'off.npz',**images)
                    save_json(out/'off.json',stats)
                saved=compare_old(method,name,frame,case,stats,images)
                comparisons.extend(dict(stage=method,scene=name,frame=frame,case=case['id'],
                    visible_depth_max_abs_m=float(np.abs(depth-frozen).max()),
                    visible_depth_max_relative=float(relative.max()),
                    visible_depth_p99_relative=float(np.quantile(relative,.99)),**r) for r in saved)
                print('MIGRATION_OPTICS',method,frame,case['id'],all(r['passed'] for r in saved),flush=True)
        model.query.store.inputs.cache_clear()
        model.query.store.active.cache_clear()
    assert len(comparisons)==35
    save_json(dest/'optical.json',dict(passed=all(r['passed'] for r in comparisons),rows=comparisons))


def pooled(values):
    sse=sum(x['sse'] for x in values); n=sum(x['pixels'] for x in values)
    return float(-10*np.log10(max(sse/(3*n),1e-12))) if n else None


def publish(cases):
    pair,zrows,rays=[],[],[]
    original={m:read(OUT/f'S3-{m}.json')['rows'] for m in ('1c','2b')}
    for method,old in original.items():
        batch=[c for c in cases if c['method']==method]
        for scope in ('pooled',*sorted({c['scene'] for c in batch})):
            selected=[c for c in batch if scope=='pooled' or c['scene']==scope]
            for f,region in ((1.4,'in_focus'),(1.4,'recoverable'),(5.,'recoverable')):
                focus=[c for c in selected if c['focus']==f]
                def pair_score(mode,q):
                    return pooled([c['optical'][mode]['pairs'][q]['regions'][region] for c in focus])
                old_split='none' if method=='1c' else 'retrieved'
                def old_score(q):
                    return pooled([r for r in old if r['scene'] in {c['scene'] for c in focus}
                                   and r['protocol']=='P2' and r['aperture']==1.4 and r['nominal_focus_m']==f
                                   and r['method']=='OVG' and r['weights']=='EMA' and r['split']==old_split
                                   and r['second']==bool(q) and r['region']==region])
                for tag,fun in (('4090D-original',old_score),('5090-off',lambda q:pair_score('off',q)),
                                ('5090-opaque-b0',lambda q:pair_score('on',q))):
                    q0,q1=fun(0),fun(1)
                    pair.append(dict(stage=method,scene=scope,focus_m=f,region=region,mode=tag,
                        query_off_PSNR=q0,query_on_PSNR=q1,delta_PSNR=q1-q0,case_count=len(focus)))
            for mode in ('off','on'):
                for tag,threshold in FLOORS.items():
                    values=[c['optical'][mode]['zero']['visible_only'] if tag=='visible' else
                            c['optical'][mode]['pairs'][1]['zero_combined'] for c in selected]
                    zrows.append(dict(stage=method,scene=scope,mode=mode,kind=tag,threshold_dB=threshold,
                        pooled_PSNR=pooled(values),passed=sum(r['psnr']>=threshold for r in values),total=len(values)))
    for method in METHODS:
        all_cases=[c for c in cases if c['method']==method]
        for scope in ('pooled',*sorted({c['scene'] for c in all_cases})):
            for f in (1.4,5.):
                selected=[c['leakage'] for c in all_cases if c['focus']==f and (scope=='pooled' or c['scene']==scope)]
                for mode in ('off','on'):
                    for tag in RAY_CLASSES:
                        matched=[v for c in selected for v in c[mode] if v['category']==tag]
                        rays.append(dict(stage=method,scene=scope,nominal_focus_m=f,mode=mode,category=tag,
                                       ray_count=sum(v['rays'] for v in matched),
                                       positive_SSE=sum(v['positive_SSE'] for v in matched),
                                       domain_rays=sum(c['domain_rays'] for c in selected),
                                       b0_closed_rays=sum(c['closed_rays'] for c in selected) if mode=='on' else 0))
    write_csv(REC/f'{PREFIX}-query.csv',pair)
    write_csv(REC/f'{PREFIX}-zero.csv',zrows)
    write_csv(REC/f'{PREFIX}-leakage.csv',rays)
    save_json(DEST/'summary.json',dict(
                                      protocol='P2 f/1.4 both foci, 3 scenes x 3 frames; EMA and same F2, b and first-pass background',
                                      ray_protocol='K16 label rays at the nominal focus; pre-closure dominant visible depth is the shared order reference',
                                      ray_off_source='frozen old K16 composites, optical weights and per-ray SSE; reclassified with 13.16.4',
                                      color_vs_F2='13.16.4 residual after ordering and the three finite-T bins',
                                      coverage_rule='translated camera-facing unit parent pixel; close only if the b=0 splat group contributes at its depth',
                                      query=[r for r in pair if r['scene']=='pooled'],
                                      zero=[r for r in zrows if r['scene']=='pooled'],
                                      leakage=[r for r in rays if r['scene']=='pooled']))


@torch.no_grad()
def run():
    guard(gpu=True)
    DEST.mkdir(exist_ok=True)
    save_json(DEST/'toy.json',toy())
    model=Model()
    states=appearance_states(model)
    all_cases=[]
    for name,frame,entry,_,_,_,manifest,items,folder in case_frames():
        trial=original_B(name,frame)
        x,y,g,first,aps=frame_models(model,states,trial)
        old=old_folder(folder)
        old_focus=read(old/'R1.json')['rows']
        depths=load_geometry(name,frame,folder)
        for item in items:
            case=item['case']
            if (case['protocol'],case['fstop'])!=('P2',1.4):
                continue
            cid=case['id']
            if all((DEST/'cases'/m/name/f'{cid}.json').is_file() for m in METHODS):
                all_cases.extend(read(DEST/'cases'/m/name/f'{cid}.json') for m in METHODS)
                continue
            center,regions,target,pinhole,focus,query=formal_inputs(model,name,frame,folder,manifest,item,g,x,y,entry)
            prior=method_row([r for r in old_focus if r['id']==cid and r['protocol']=='P2'],ORIGINAL)
            fixed_depth=tensor(depths[ORIGINAL])
            fixed_query,_=formal_layer(model.query,item,manifest,old,prior,fixed_depth)
            for m in METHODS:
                save=DEST/'cases'/m/name/f'{cid}.json'
                if save.is_file():
                    all_cases.append(read(save))
                    continue
                if m==ORIGINAL:
                    layers=make_layers(x,g,None,first,m,fixed_query,original=fixed_depth)
                    legacy=('1c',name,frame,case,m)
                else:
                    layers=make_layers(x,g,aps[m],first,m,query)
                    legacy=(m,name,frame,case,m)
                result=dict(method=m,scene=name,frame=frame,case=cid,focus=case['nominal_focus_m'],
                            actual_focus_m=focus if m!=ORIGINAL else prior['实际对焦距离_m'])
                if m in ('1c','2b'):
                    result['optical']={}
                    for mode in ('off','on'):
                        stats,images=optics(layers,x['K'][0],focus,1.4,dict(tiers='B',entry=entry),
                                            pinhole,target,regions,mode=='on')
                        result['optical'][mode]=stats
                        save_npz(save.parent/cid/f'optical-{mode}.npz',**images)
                result['leakage'],arrays=leakage(layers,x['K'][0],entry,case,center,regions,target,legacy)
                if arrays:
                    save_npz(save.parent/cid/'leakage-on.npz',**arrays)
                save_json(save,result)
                all_cases.append(result)
                print('OPAQUE_B0',m,name,cid,flush=True)
        model.query.store.inputs.cache_clear()
        model.query.store.active.cache_clear()
    assert len(all_cases)==72
    publish(all_cases)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('task',choices=('migration','toy','evaluate'))
    p.add_argument('--migration-dir',type=Path)
    a=p.parse_args()
    if a.task=='toy':
        guard(gpu=True)
        print(toy())
    elif a.task=='migration':
        migration_optics(a.migration_dir)
    else:
        run()
