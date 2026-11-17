"""Appendix12 tenth revision: fixed explicit surfels and shared input-only fill."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
import cv2
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from route3_local_replacement import desc_for, load, tensor, PROFILE, reference
from route3_depth_interface import write, save
from route3_sharp_s1 import raster
from src.visible_depth_interface import dominant_surface
from src.layered_aperture import pixel_layer
from src.layered_aperture import merge_layers
from src.analytic_aperture import sampled
from src.dual_layer import depth_band
from route3_s2a_predict import visible_cloud
from route3_s2a_proxy import display
from route3_b2_oracle_3d import sparse_pixel_layer
from src.explicit_layer_interface import replacement_footprints, subset_scene

PREFIX = 'route3-explicit-layer-2026-11-17'
OUT = ROOT/'outputs'/f'{PREFIX}-01'
REC = ROOT/'03_实验记录/records'
PREV = ROOT/'outputs/route3-local-replacement-2026-11-13-01'
SCENES = ['metric-pilot-122001-broad', 'metric-pilot-122002-multilayer', 'B2-high']
METHODS = ['A', 'B', 'Bprime', 'C0', 'C1', 'D0', 'D1', 'DGT0', 'DGT1']


def fd(name, frame):
    return OUT/name/f'f{frame:03d}'


def cd(name, frame, case, rule):
    return fd(name, frame)/rule/case['id']


def guard(gpu=False):
    if shutil.disk_usage(OUT).free / 2**30 < 60:
        raise RuntimeError('STOP: data disk below 60 GiB; no cleanup authorized')
    if gpu:
        assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
        subprocess.run(['nvidia-smi'], check=True)
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    torch.backends.cuda.matmul.allow_tf32 = False


def validation():
    """B2 frame27 is not one of the three evaluation frames; no parameter fitting."""
    guard(True)
    from route3_sharp_s1b_temporal import TEMP, full_scene
    d = load(TEMP/'sharp-depth/f027.npz')
    K = tensor(d['K'])
    cloud = full_scene(27)
    start = time.monotonic()
    clusters = dominant_surface(cloud, K, np.ones((512,512), bool))
    depth = clusters['cluster_depth']
    valid = (clusters['cluster_fraction'] >= .9) & (clusters['contribution_alpha'] > 1e-6) & np.isfinite(depth) & (depth > 0)
    no_cluster = ~np.isfinite(depth) | (depth <= 0)
    depth[no_cluster] = 1e4
    del cloud
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    explicit = pixel_layer(tensor(d['rgb_linear']), tensor(depth), K, torch.eye(4, device='cuda'))
    with torch.inference_mode():
        rgb, z, alpha = raster(explicit, K)
    from route3_s2a_proxy import display
    # Black residual here avoids hiding a zero-aperture representation failure.
    image = display(rgb.cpu().numpy())
    error = ((image.astype(np.float64)-d['rgb_srgb'])**2).sum()
    psnr = float(-10*np.log10(error/(512*512*3)))
    save(OUT/'validation-f027.npz', prediction_srgb=image, rgb_linear=rgb.cpu().numpy(), alpha=alpha.cpu().numpy(), depth=depth, dominant_valid=valid)
    result = dict(scene='B2-high', frame=27, evaluation_frames=[1,30,60], non_evaluation_frame=True, zero_PSNR=psnr, threshold_dB=50, passed=psnr>=50, dominant_fraction=float(valid.mean()), no_cluster_pixels=int(no_cluster.sum()), Gaussian_count=len(explicit.means), parameters=dict(subdivisions=5,sigma_pixels=.7,opacity=.999,eps2d=.0015), residual_background='black, conservative zero-aperture check; formal E2 fill is shared', seconds=time.monotonic()-start, peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20)
    write(OUT/'validation.json',result)
    write(REC/(PREFIX+'-validation.json'),result)
    print('E1_VALIDATION',json.dumps(result),flush=True)
    if not result['passed']:
        raise RuntimeError('STOP: non-evaluation D0 zero-aperture PSNR below50dB; no tuning authorized')


def prepare():
    guard(True)
    assert json.loads((OUT/'validation.json').read_text())['passed']
    fills = []
    for name in SCENES:
        desc = desc_for(name)
        for entry in desc['entries']:
            f = entry['frame']
            dest = fd(name, f)
            dest.mkdir(parents=True, exist_ok=True)
            if not (dest/'prepared.json').exists():
                start = time.monotonic()
                guard()
                d = load(entry['input_npz'])
                K = tensor(d['K'])
                cloud = visible_cloud(desc, entry)
                clusters = dominant_surface(cloud, K, np.ones((512,512), bool))
                own = load(PREV/name/f'f{f:03d}'/'visible-depth.npz')
                old = json.loads((PREV/name/f'f{f:03d}'/'prepared.json').read_text())
                depth = clusters['cluster_depth'].copy()
                invalid_empty = ~np.isfinite(depth) | (depth <= 0)
                dominant = (clusters['cluster_fraction'] >= .9) & (clusters['contribution_alpha'] > 1e-6) & ~invalid_empty
                depth[invalid_empty] = 1e4
                # Existing A focus/ED bands are copied, not selected anew.
                union = np.zeros((512,512), bool)
                max_radius = 0.
                bands = {}
                for rule in ['P2','P1']:
                    for case in entry['cases']:
                        row = next(r for r in old['focus'] if r['rule']==rule and r['nominal_focus_m']==case['nominal_focus_m'])
                        if not row['success']:
                            raise RuntimeError('Existing A focus unavailable; no refocus authorized')
                        band, edge, coc = depth_band(own['depth'], d['K'][0,0], row['focus_m'], .05/(2*case['fstop']))
                        previous = load(PREV/name/f'f{f:03d}'/rule/case['id']/'support-masks.npz')['band']
                        assert np.array_equal(band, previous)
                        bands[rule+'__'+case['id']] = band
                        union |= band
                        max_radius = max(max_radius, float(coc.max()))
                mask = cv2.dilate(union.astype(np.uint8), np.ones((5,5),np.uint8), iterations=2)>0
                radius = int(np.ceil(2*max_radius))+4
                bg = cv2.dilate(own['depth'].astype(np.float32), np.ones((2*radius+1,2*radius+1),np.uint8))
                save(dest/'explicit-depth.npz', depth=depth, dominant_valid=dominant, no_cluster=invalid_empty, **clusters)
                save(dest/'fill-input.npz', rgb_linear=d['rgb_linear'], rgb_srgb=d['rgb_srgb'], fill_mask=mask, background_depth=bg, K=d['K'])
                save(dest/'bands.npz', **bands)
                write(dest/'prepared.json', dict(scene=name,frame=f,focus=old['focus'],dominant_fraction=float(dominant.mean()),invalid_fraction=float((~dominant).mean()),no_cluster_pixels=int(invalid_empty.sum()),fill_mask_fraction=float(mask.mean()),fill_mask_pixels=int(mask.sum()),max_CoC_px=max_radius,background_filter_radius_px=radius,depth_source='largest contribution cluster of complete A,not first layer;fallback largest cluster marked invalid',focus_and_band_unchanged=True,seconds=time.monotonic()-start))
                print('E_PREPARE',name,f,float(dominant.mean()),radius,flush=True)
                del cloud
                torch.cuda.empty_cache()
            fills.append(dict(id=name+f'-f{f:03d}', input_npz=str(dest/'fill-input.npz'), uses_GT_segmentation=False,uses_true_hidden_depth=False))
    write(OUT/'fill-manifest.json',dict(items=fills))


def validation_prepare_fill():
    """Exact E2 background for the same non-evaluation validation frame.

    The earlier black check remains saved. Only its residual background is
    recomposed; Gaussian parameters and camera are identical, without fitting.
    """
    guard(True)
    from route3_sharp_s1b_temporal import TEMP,full_scene
    from route3_depth_interface import focus_row,subject_mask
    from route3_step2_diagnostic import fit_background
    d=load(TEMP/'sharp-depth/f027.npz')
    v=load(OUT/'validation-f027.npz')
    K=tensor(d['K'])
    cloud=full_scene(27)
    with torch.inference_mode():
        _,depth,alpha=raster(cloud,K)
    own=depth.cpu().numpy()
    own=np.where((own>0)&np.isfinite(own)&(alpha.cpu().numpy()>1e-6),own,1e4).astype(np.float32)
    desc=desc_for('B2-high')
    g=load(desc['geometry']/'f027-center.npz')
    fg=np.load(desc['scene']/'scene-geometry.npz')['foreground_object_ids']
    fit=fit_background(g['depth_axial_m'],own,g['center_visible']&~np.isin(g['object_id'],fg))
    clusters=dict(cluster_depth=v['depth'],cluster_fraction=v['dominant_valid'].astype(np.float32),contribution_alpha=np.ones((512,512),np.float32))
    union=np.zeros((512,512),bool)
    maximum=0.
    focuses=[]
    for nominal in [1.4,5.]:
      for rule in ['P2','P1']:
        row=focus_row(own,subject_mask(g,nominal),nominal,None,fit,rule,clusters)
        if not row['success']:
            raise RuntimeError('Validation A focus unavailable;no substitute allowed')
        focuses.append(row)
        for stop in [1.4,2.]:
            band,_,coc=depth_band(own,d['K'][0,0],row['focus_m'],.05/(2*stop))
            union|=band
            maximum=max(maximum,float(coc.max()))
    mask=cv2.dilate(union.astype(np.uint8),np.ones((5,5),np.uint8),iterations=2)>0
    radius=int(np.ceil(2*maximum))+4
    background=cv2.dilate(own,np.ones((2*radius+1,2*radius+1),np.uint8))
    save(OUT/'validation-fill-input.npz',rgb_linear=d['rgb_linear'],rgb_srgb=d['rgb_srgb'],fill_mask=mask,background_depth=background)
    write(OUT/'validation-fill-manifest.json',dict(items=[dict(id='B2-high-f027',input_npz=str(OUT/'validation-fill-input.npz'),uses_GT_segmentation=False,uses_true_hidden_depth=False)]))
    write(OUT/'validation-fill-preparation.json',dict(focus=focuses,fill_mask_fraction=float(mask.mean()),background_filter_radius_px=radius,no_parameters_changed=True))


def validation_finalize():
    v=load(OUT/'validation-f027.npz')
    from route3_sharp_s1b_temporal import TEMP
    d=load(TEMP/'sharp-depth/f027.npz')
    fill=load(OUT/'validation-fill/B2-high-f027/fill.npz')['fill_rgb_linear']
    rgb=v['rgb_linear']+(1-v['alpha'][...,None])*fill
    image=display(rgb)
    error=float(((image.astype(np.float64)-d['rgb_srgb'])**2).sum())
    psnr=float(-10*np.log10(error/(512*512*3)))
    save(OUT/'validation-f027-E2.npz',rgb_linear=rgb,prediction_srgb=image)
    result=json.loads((OUT/'validation.json').read_text())
    result.update(zero_PSNR_black=result['zero_PSNR'],zero_PSNR=psnr,residual_background='formal shared E2 input-only LaMa rule;exact residual recomposition of same non-evaluation raster/alpha',passed=psnr>=50)
    write(OUT/'validation-E2.json',result)
    write(REC/(PREFIX+'-validation-E2.json'),result)
    print('E1_FORMAL_E2_VALIDATION',psnr,flush=True)
    if not result['passed']:
        raise RuntimeError('STOP: formal E2 validation below50dB;no tuning authorized')


def zero(cloud,K,fill):
    with torch.inference_mode():
        rgb,depth,alpha=raster(cloud,K)
    return dict(zero_rgb_linear=(rgb+(1-alpha[...,None])*fill).cpu().numpy(), zero_srgb=display((rgb+(1-alpha[...,None])*fill).cpu().numpy()), zero_alpha=alpha.cpu().numpy())


def do_render(cloud,K,focus,stop,fill,bg):
    start=time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        result=sampled(cloud,K,torch.eye(4,device='cuda'),512,512,focus,.05/(2*stop),64,fill,bg,far_plane=1e5,return_fill_contribution=True)
    values={k:v.cpu().numpy() for k,v in result.items()}
    values['prediction_srgb']=display(values['rgb_linear'])
    values.update(zero(cloud,K,fill))
    return values,dict(seconds=time.monotonic()-start,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,gaussians=len(cloud.means))


def raw_hidden(parents,K):
    selection=parents['recoverable']
    if not selection.any():
        return None
    return sparse_pixel_layer(parents,selection,K,torch.eye(4,device='cuda'),PROFILE,'cuda')


def upper():
    guard(True)
    rows=[]
    for name in SCENES[:2]:
        desc=desc_for(name)
        for entry in desc['entries']:
            guard()
            f=entry['frame']
            case=next(c for c in entry['cases'] if c['nominal_focus_m']==5 and c['fstop']==1.4)
            folder=cd(name,f,case,'P2')
            folder.mkdir(parents=True,exist_ok=True)
            d=load(entry['input_npz'])
            K=tensor(d['K'])
            g=load(desc['geometry']/f'f{f:03d}-center.npz')
            truth,regions=reference(name,desc,case,f)
            prep=json.loads((fd(name,f)/'prepared.json').read_text())
            focus=next(r['focus_m'] for r in prep['focus'] if r['rule']=='P2' and r['nominal_focus_m']==5)
            fill=tensor(load(OUT/'fill'/(name+f'-f{f:03d}')/'fill.npz')['fill_rgb_linear'])
            bg=tensor(load(fd(name,f)/'fill-input.npz')['background_depth'])
            parents=load(desc['parents'](f))
            for method in ['A','DGT0','DGT1']:
                path=folder/(method+'.npz')
                if not path.exists():
                    if method=='A':
                        cloud=visible_cloud(desc,entry)
                    else:
                        depth=np.where(g['center_visible']&(g['depth_axial_m']>0),g['depth_axial_m'],1e4).astype(np.float32)
                        cloud=pixel_layer(tensor(d['rgb_linear']),tensor(depth),K,torch.eye(4,device='cuda'))
                        if method=='DGT1':
                            hidden=raw_hidden(parents,K)
                            cloud=merge_layers(cloud,hidden)
                            del hidden
                    values,timing=do_render(cloud,K,focus,1.4,fill,bg)
                    save(path,**values)
                    write(folder/(method+'-timing.json'),timing)
                    del cloud,values
                    torch.cuda.empty_cache()
                prediction=load(path)['prediction_srgb']
                mask=regions['recoverable']
                error=float(((prediction[mask].astype(np.float64)-truth[mask])**2).sum())
                n=int(mask.sum())
                rows.append(dict(scene=name,frame=f,method=method,pixels=n,sse=error,psnr=float(-10*np.log10(max(error/(3*n),1e-30)))))
                print('E_UPPER',name,f,method,rows[-1]['psnr'],flush=True)
    lookup={(r['scene'],r['frame'],r['method']):r for r in rows}
    gains=[dict(scene=name,frame=f,H_GT_dB=lookup[name,f,'DGT1']['psnr']-lookup[name,f,'DGT0']['psnr'],DGT1_minus_A_dB=lookup[name,f,'DGT1']['psnr']-lookup[name,f,'A']['psnr']) for name in SCENES[:2] for f in [1,30,60]]
    def pooled(method):
        rr=[r for r in rows if r['method']==method]
        return -10*np.log10(sum(r['sse'] for r in rr)/(3*sum(r['pixels'] for r in rr)))
    gain=float(pooled('DGT1')-pooled('DGT0'))
    frame_pass={name:sum(r['H_GT_dB']>=1 for r in gains if r['scene']==name) for name in SCENES[:2]}
    passed=gain>=1 and all(v>=2 for v in frame_pass.values())
    result=dict(rows=rows,frame_gains=gains,pooled_H_GT_dB=gain,each_scene_frames_ge1dB=frame_pass,passed=passed,criteria='G2 pooled far f1.4>=1dB and each pilot>=2/3 frames>=1dB',same_A_P2_focus=True,next_step='complete matrix' if passed else 'STOP: H_GT gate failed; user decides scene/recoverability review; do not judge C/D representation')
    write(OUT/'upper-gate.json',result)
    write(REC/(PREFIX+'-upper-gate.json'),result)
    print('E_UPPER_GATE',json.dumps(result),flush=True)


def aligned_hidden(parents,alignment,band,K,raw=False):
    uv=parents['uv']
    x=np.clip(uv[:,0].astype(int),0,511)
    y=np.clip(uv[:,1].astype(int),0,511)
    select=parents['recoverable']&band[y,x]
    values=parents
    if not raw:
        select&=alignment['parent_valid']
        values=dict(parents,depth=alignment['parent_depth_aligned'])
    if not select.any():
        return None,select
    return sparse_pixel_layer(values,select,K,torch.eye(4,device='cuda'),PROFILE,'cuda'),select


def replacement(name,entry,cloud,K,depth,rgb,rule,nominal):
    f=entry['frame']
    dest=fd(name,f)/rule/f'replacement-F{nominal:g}.npz'
    desc=desc_for(name)
    case=next(c for c in entry['cases'] if c['nominal_focus_m']==nominal and c['fstop']==1.4)
    band=load(fd(name,f)/'bands.npz')[rule+'__'+case['id']]
    prep=json.loads((fd(name,f)/'prepared.json').read_text())
    p2=next(r['focus_m'] for r in prep['focus'] if r['rule']=='P2' and r['nominal_focus_m']==nominal)
    if not dest.exists():
        deleted,coverage,overlap,audit=replacement_footprints(cloud,K,band,p2)
        ids=cloud.source_frames.cpu().numpy()
        rm=deleted.cpu().numpy()
        op=overlap.cpu().numpy()
        seam=(cv2.dilate(coverage.astype(np.uint8),np.ones((17,17),np.uint8))>0)&~(cv2.erode(coverage.astype(np.uint8),np.ones((17,17),np.uint8))>0)
        save(dest,deleted=rm,coverage=coverage,retained_overlap=op,seam=seam)
        write(dest.with_suffix('.json'),dict(**audit,scene=name,frame=f,rule=rule,nominal_focus_m=nominal,p2_focus_m=p2,layer0_deleted=int((rm&(ids==0)).sum()),layer1_deleted=int((rm&(ids==1)).sum()),layer0_retained_overlap=int((op&(ids==0)).sum()),layer1_retained_overlap=int((op&(ids==1)).sum()),seam_pixels=int(seam.sum()),f2_reuses_f14_set=True,band_source='A rule-specific f1.4 band;CoC expansion always uses fixed A P2 focus;reuse same set at f2.0'))
    masks=load(dest)
    retained=subset_scene(cloud,tensor(~masks['deleted']))
    explicit=pixel_layer(tensor(rgb),tensor(depth),K,torch.eye(4,device='cuda'),valid=tensor(masks['coverage']))
    return merge_layers(retained,explicit),masks


def render():
    guard(True)
    if not json.loads((OUT/'upper-gate.json').read_text())['passed']:
        raise RuntimeError('H_GT stop condition: remaining matrix not authorized')
    for name in SCENES:
      desc=desc_for(name)
      for entry in desc['entries']:
        guard()
        f=entry['frame']
        d=load(entry['input_npz'])
        K=tensor(d['K'])
        explicit_depth=load(fd(name,f)/'explicit-depth.npz')
        own=explicit_depth['depth']
        g=load(desc['geometry']/f'f{f:03d}-center.npz')
        zgt=np.where(g['center_visible']&(g['depth_axial_m']>0),g['depth_axial_m'],1e4).astype(np.float32)
        prep=json.loads((fd(name,f)/'prepared.json').read_text())
        parents=load(desc['parents'](f))
        alignment=load(PREV/name/f'f{f:03d}'/'alignment.npz')
        cloud=visible_cloud(desc,entry)
        full=pixel_layer(tensor(d['rgb_linear']),tensor(own),K,torch.eye(4,device='cuda'))
        fullgt=pixel_layer(tensor(d['rgb_linear']),tensor(zgt),K,torch.eye(4,device='cuda'))
        allhidden=raw_hidden(parents,K)
        fill=tensor(load(OUT/'fill'/(name+f'-f{f:03d}')/'fill.npz')['fill_rgb_linear'])
        bg=tensor(load(fd(name,f)/'fill-input.npz')['background_depth'])
        for rule in ['P2','P1']:
          for nominal in [1.4,5.]:
            local,masks=replacement(name,entry,cloud,K,own,d['rgb_linear'],rule,nominal)
            focus=next(r['focus_m'] for r in prep['focus'] if r['rule']==rule and r['nominal_focus_m']==nominal)
            for case in [c for c in entry['cases'] if c['nominal_focus_m']==nominal]:
                folder=cd(name,f,case,rule)
                folder.mkdir(parents=True,exist_ok=True)
                if (folder/'render.json').exists():
                    continue
                guard()
                band=load(fd(name,f)/'bands.npz')[rule+'__'+case['id']]
                hidden,selection=aligned_hidden(parents,alignment,band,K)
                hiddenraw,rawselection=aligned_hidden(parents,alignment,band,K,True)
                if hidden is not None:
                    with torch.inference_mode():
                        _,_,a=raster(hidden,K)
                    support=(a.cpu().numpy()>0)&band&alignment['local_supported']
                else:
                    support=np.zeros((512,512),bool)
                save(folder/'masks.npz',band=band,support=support,seam=masks['seam'],coverage=masks['coverage'],invalid=~explicit_depth['dominant_valid'])
                states=dict(A=(cloud,None),B=(cloud,hidden),Bprime=(cloud,hiddenraw),C0=(local,None),C1=(local,hidden),D0=(full,None),D1=(full,hidden),DGT0=(fullgt,None),DGT1=(fullgt,allhidden))
                rr=[]
                for method,(visible,hid) in states.items():
                    path=folder/(method+'.npz')
                    if not path.exists():
                        values,timing=do_render(merge_layers(visible,hid),K,focus,case['fstop'],fill,bg)
                        save(path,**values)
                        write(folder/(method+'-timing.json'),timing)
                        del values
                        torch.cuda.empty_cache()
                    rr.append(dict(method=method,path=str(path),**json.loads((folder/(method+'-timing.json')).read_text())))
                    if method in ['C0','C1','D0','D1']:
                        black=folder/(method+'-black.npz')
                        if not black.exists():
                            v=load(path)
                            # Exact linear compositing identity, same rays/weights:
                            # black = accumulated surface RGB, no residual T*Cfill.
                            b=v['rgb_linear']-v['fill_contribution_linear']
                            zb=v['zero_rgb_linear']-(1-v['zero_alpha'][...,None])*fill.cpu().numpy()
                            save(black,prediction_srgb=display(b),rgb_linear=b,alpha=v['alpha'],zero_srgb=display(zb))
                    print('E_RENDER',name,f,rule,case['id'],method,flush=True)
                _,regions=reference(name,desc,case,f)
                uv=parents['uv'][selection]
                xx=np.clip(uv[:,0].astype(int),0,511)
                yy=np.clip(uv[:,1].astype(int),0,511)
                ahead=(alignment['parent_depth_aligned'][selection]<=own[yy,xx])
                write(folder/'render.json',dict(scene=name,frame=f,id=case['id'],rule=rule,nominal_focus_m=nominal,fstop=case['fstop'],focus_m=focus,rows=rr,focus_and_band_common=True,aligned_parents=int(selection.sum()),raw_band_parents=int(rawselection.sum()),raw_all_recoverable_parents=int(parents['recoverable'].sum()),support_recoverable_fraction=float(support[regions['recoverable']].mean()) if regions['recoverable'].any() else None,support_recoverable_pixels=int((support&regions['recoverable']).sum()),recoverable_pixels=int(regions['recoverable'].sum()),aligned_parent_not_behind_explicit_center_fraction=float(ahead.mean()) if len(ahead) else None,hidden_order='actual front-to-back depth sort;no hidden-depth clamping or artificial alpha override',black_diagnostic='exact surface accumulated RGB from same rays;subtract saved residual-fill contribution,not an image mask'))
                del hidden,hiddenraw,states
                torch.cuda.empty_cache()
            del local
            torch.cuda.empty_cache()
        del cloud,full,fullgt,allhidden
        torch.cuda.empty_cache()


def evaluate(net=None):
    guard()
    from route3_upper_bound_evaluate import score
    from route3_ldr_u1r import model
    if net is None:
        net=model()
    allrows=[]
    for name in SCENES:
      desc=desc_for(name)
      for entry in desc['entries']:
        f=entry['frame']
        target=load(entry['input_npz'])['rgb_srgb']
        for rule in ['P2','P1']:
          for case in entry['cases']:
            folder=cd(name,f,case,rule)
            if not (folder/'render.json').exists():
                continue
            methods=METHODS+[m+'-black' for m in ['C0','C1','D0','D1']]
            if not all((folder/(method+'-scores.json')).exists() for method in methods):
                truth,regions=reference(name,desc,case,f)
                m=load(folder/'masks.npz')
                regions=dict(regions,invalid_geometry=m['invalid'],invalid_infocus=m['invalid']&regions['in_focus'],seam=m['seam'],seam_infocus=m['seam']&regions['in_focus'],recoverable_supported=m['support']&regions['recoverable'])
            common=dict(scene=name,frame=f,id=case['id'],rule=rule,nominal_focus_m=case['nominal_focus_m'],fstop=case['fstop'])
            for method in methods:
                cache=folder/(method+'-scores.json')
                if cache.exists():
                    result=json.loads(cache.read_text())
                else:
                    v=load(folder/(method+'.npz'))
                    result=dict(dof=score(v['prediction_srgb'],truth,regions,net),zero={})
                    for region,mask in regions.items():
                        n=int(mask.sum())
                        error=float(((v['zero_srgb'][mask].astype(np.float64)-target[mask])**2).sum())
                        result['zero'][region]=dict(pixels=n,sse=error,psnr=float(-10*np.log10(max(error/max(3*n,1),1e-30))) if n else None)
                    write(cache,result)
                for region,value in result['dof'].items():
                    z=result['zero']['full_image']
                    allrows.append(dict(**common,method=method,region=region,**value,zero_PSNR=z['psnr'],zero_sse=z['sse'],zero_pixels=z['pixels'],region_zero_PSNR=result['zero'][region]['psnr'],region_zero_sse=result['zero'][region]['sse']))
            print('E_SCORE',name,f,rule,case['id'],flush=True)
    write(OUT/'scores.json',dict(rows=allrows,reference='Cycles;frozen GT evaluation regions;fixed64px AlexNet LPIPS0.1 patches'))


def follow_evaluate():
    """CPU-only LPIPS scoring can follow serial GPU rendering without a second GPU job."""
    guard()
    from route3_ldr_u1r import model
    net=model()
    last=-1
    while True:
        count=sum(1 for p in OUT.glob('*/f*/P*/*/render.json'))
        if count!=last:
            evaluate(net)
            last=count
        if count==72:
            return
        time.sleep(10)


def statistics():
    """Prefix transmittance only, computed before the full image matrix.

    Reuse appendix10 K16-to-parent correspondences. Queries go through placed
    parent centers at continuous image coordinates; raw/aligned domains and
    foreground-blocked controls are separate, never silently mixed.
    """
    guard(True)
    if not json.loads((OUT/'upper-gate.json').read_text())['passed']:
        raise RuntimeError('H_GT gate failed')
    from route3_local_replacement_stats import query_for
    from src.local_gaussian_replacement import projection,ray_query_plan,blocker_ray_queries
    from route3_step2_diagnostic import numpy_stat
    for name in SCENES:
      desc=desc_for(name)
      for entry in desc['entries']:
        f=entry['frame']
        d=load(entry['input_npz'])
        K=tensor(d['K'])
        depth=load(fd(name,f)/'explicit-depth.npz')['depth']
        g=load(desc['geometry']/f'f{f:03d}-center.npz')
        zgt=np.where(g['center_visible']&(g['depth_axial_m']>0),g['depth_axial_m'],1e4).astype(np.float32)
        parents=load(desc['parents'](f))
        alignment=load(PREV/name/f'f{f:03d}'/'alignment.npz')
        prep=json.loads((fd(name,f)/'prepared.json').read_text())
        cloud=visible_cloud(desc,entry)
        full=pixel_layer(tensor(d['rgb_linear']),tensor(depth),K,torch.eye(4,device='cuda'))
        fullgt=pixel_layer(tensor(d['rgb_linear']),tensor(zgt),K,torch.eye(4,device='cuda'))
        allhidden=raw_hidden(parents,K)
        for rule in ['P2','P1']:
          for nominal in [1.4,5.]:
            local,_=replacement(name,entry,cloud,K,depth,d['rgb_linear'],rule,nominal)
            focus=next(r['focus_m'] for r in prep['focus'] if r['rule']==rule and r['nominal_focus_m']==nominal)
            for case in [c for c in entry['cases'] if c['nominal_focus_m']==nominal]:
                folder=cd(name,f,case,rule)
                done=folder/'transmittance.json'
                if done.exists():
                    continue
                guard()
                lab=load(PREV/name/f'f{f:03d}'/(case['id']+'-ray-reference.npz'))
                band=load(fd(name,f)/'bands.npz')[rule+'__'+case['id']]
                hidden,_=aligned_hidden(parents,alignment,band,K)
                hiddenraw,_=aligned_hidden(parents,alignment,band,K,True)
                vis=dict(A=cloud,C=local,D=full,DGT=fullgt)
                hides=dict(aligned=hidden,raw=hiddenraw,allraw=allhidden)
                buffers={}
                domains=[]
                for k,xy in enumerate(lab['aperture_xy']):
                    view=torch.eye(4,device='cuda')
                    view[:2,3]-=tensor(xy)
                    intr=K.clone()
                    intr[:2,2]+=K.diag()[:2]*tensor(xy)/focus
                    plans={}
                    for plane in ['aligned','raw']:
                        uv,limit,nh,nc=query_for(lab,k,plane,parents,d['K'],xy,focus,nominal)
                        plan=ray_query_plan(tensor(uv))
                        valid=plan[-1].cpu().numpy()
                        plans[plane]=(plan,tensor(limit),nh,nc,valid)
                        domains.append(dict(plane=plane,lens=k,hidden_queries=nh,hidden_inside=int(valid[:nh].sum()),control_queries=nc,control_inside=int(valid[nh:].sum())))
                    for kind,states in [('visible',vis),('hidden',hides)]:
                      for state,s in states.items():
                        projected=projection(s,intr,view) if s is not None else None
                        for plane,(plan,limit,nh,nc,valid) in plans.items():
                            if projected is None:
                                logs=np.zeros(nh+nc,np.float32)
                            else:
                                logs=blocker_ray_queries(s,projected,limit,plan).sum(0).cpu().numpy()
                            for domain,take in [('hidden',np.r_[valid[:nh],np.zeros(nc,bool)]),('control',np.r_[np.zeros(nh,bool),valid[nh:]])]:
                                buffers.setdefault(plane+'__'+kind+'_'+state+'__'+domain,[]).append(logs[take])
                        del projected
                arrays={key:np.concatenate(value) for key,value in buffers.items()}
                mapping=dict(A=('A',None),B=('A','aligned'),Bprime=('A','raw'),C0=('C',None),C1=('C','aligned'),D0=('D',None),D1=('D','aligned'),DGT0=('DGT',None),DGT1=('DGT','allraw'))
                rows=[]
                for plane in ['aligned','raw']:
                  for domain in ['hidden','control']:
                    for method,(vv,hh) in mapping.items():
                        logs=arrays[plane+'__visible_'+vv+'__'+domain].copy()
                        if hh is not None:
                            logs+=arrays[plane+'__hidden_'+hh+'__'+domain]
                        T=np.exp(-np.minimum(logs.astype(np.float64),745))
                        rows.append(dict(method=method,plane=plane,domain=domain,preferred_plane=('raw' if method in ['Bprime','DGT0','DGT1'] else 'aligned'),**numpy_stat(T)))
                save(folder/'prefix-log-components.npz',**arrays)
                write(done,dict(scene=name,frame=f,id=case['id'],rule=rule,nominal_focus_m=nominal,fstop=case['fstop'],focus_m=focus,rows=rows,domain=domains,raw_label_rays=int(lab['raw_label_count']),matched_parent_rays=int(lab['matched_label_count']),query='same appendix10 matched K16 hidden surface parent,continuous projected placed-parent center;raw/aligned domains separate',prefix='native projection AA/cutoff;full product,no early truncation;includes all Gaussian centers strictly before query surface;colors/fill independent',black_has_same_T=True,diagnostic_GT_only=True))
                print('E_PREFIX',name,f,rule,case['id'],flush=True)
                del hidden,hiddenraw,buffers,arrays,hides
                torch.cuda.empty_cache()
            del local
        del cloud,full,fullgt,allhidden
        torch.cuda.empty_cache()


def zero_categories(gt,foreground_ids,invalid):
    """Disjoint diagnostic categories, with GT restricted to analysis."""
    ids=gt['object_id']
    fg=np.isin(ids,foreground_ids)&gt['center_visible']
    edge=np.zeros_like(fg)
    for axis in [0,1]:
        prev=np.roll(ids,1,axis)
        prevfg=np.roll(fg,1,axis)
        diff=(ids!=prev)&(fg|prevfg)
        if axis==0:
            diff[0]=False
        else:
            diff[:,0]=False
        edge|=diff|np.roll(diff,-1,axis)
    boundary=cv2.dilate(edge.astype(np.uint8),np.ones((5,5),np.uint8))>0
    n,labels,stats,_=cv2.connectedComponentsWithStats(fg.astype(np.uint8),8)
    large=np.zeros_like(fg)
    for component in range(1,n):
        if stats[component,cv2.CC_STAT_AREA]>=1024:
            large|=labels==component
    categories=np.full(fg.shape,3,np.uint8)
    categories[large]=2
    categories[invalid]=1
    categories[boundary]=0
    return categories


def zero_diagnostics():
    """User-added zero-error localization: statistics only, no new render groups."""
    guard(True)
    from src.explicit_layer_interface import hidden_depth_contributions
    from route3_step2_diagnostic import numpy_stat
    names=['foreground_boundary_pm2','invalid_geometry','large_occluder_interior','other']
    settings=dict(category_priority=names,foreground_boundary='GT object-ID transitions with at least one foreground side,5x5 dilation (2px);report only',large_occluder='GT foreground connected component>=1024px,then boundary/invalid excluded',positive_pixel_SSE_threshold=1e-12,depth_bins='front<-3%,within±3%,behind>3%;also strictly-front fraction',depth_reference='fixed E1 explicit cluster depth at that pixel;C pixels outside explicit coverage separately flagged',depth_distribution='strongest actually composited hidden Gaussian,plus all-hidden optical-weight bin masses;no new RGB renders',SSE='per-pixel sum of squared display-RGB error to input;delta=with-hidden minus no-hidden;positive increase and signed net both reported')
    write(OUT/'zero-diagnostic-settings.json',settings)
    write(REC/(PREFIX+'-zero-diagnostic-settings.json'),settings)
    for name in SCENES:
      desc=desc_for(name)
      fg=np.load(desc['scene']/'scene-geometry.npz')['foreground_object_ids']
      for entry in desc['entries']:
        f=entry['frame']
        d=load(entry['input_npz'])
        K=tensor(d['K'])
        target=d['rgb_srgb'].astype(np.float64)
        gt=load(desc['geometry']/f'f{f:03d}-center.npz')
        explicit=load(fd(name,f)/'explicit-depth.npz')
        depth=explicit['depth']
        categories=zero_categories(gt,fg,~explicit['dominant_valid'])
        zgt=np.where(gt['center_visible']&(gt['depth_axial_m']>0),gt['depth_axial_m'],1e4).astype(np.float32)
        cloud=visible_cloud(desc,entry)
        full=pixel_layer(tensor(d['rgb_linear']),tensor(depth),K,torch.eye(4,device='cuda'))
        fullgt=pixel_layer(tensor(d['rgb_linear']),tensor(zgt),K,torch.eye(4,device='cuda'))
        parents=load(desc['parents'](f))
        alignment=load(PREV/name/f'f{f:03d}'/'alignment.npz')
        allhidden=raw_hidden(parents,K)
        for rule in ['P2','P1']:
          for nominal in [1.4,5.]:
            local,replace=replacement(name,entry,cloud,K,depth,d['rgb_linear'],rule,nominal)
            for case in [c for c in entry['cases'] if c['nominal_focus_m']==nominal]:
                folder=cd(name,f,case,rule)
                done=folder/'zero-diagnostics.json'
                if done.exists():
                    continue
                guard()
                band=load(folder/'masks.npz')['band']
                hidden,_=aligned_hidden(parents,alignment,band,K)
                rows=[]
                arrays=dict(categories=categories)
                for label,v,hid,ref,coverage in [('C',local,hidden,depth,replace['coverage']),('D',full,hidden,depth,np.ones((512,512),bool)),('DGT',fullgt,allhidden,zgt,np.ones((512,512),bool))]:
                    lo=load(folder/(label+'0.npz'))['zero_srgb'].astype(np.float64)
                    hi=load(folder/(label+'1.npz'))['zero_srgb'].astype(np.float64)
                    old=np.sum((lo-target)**2,-1)
                    new=np.sum((hi-target)**2,-1)
                    delta=new-old
                    pos=delta>1e-12
                    neg=delta< -1e-12
                    positive=np.where(pos,delta,0.)
                    positive_total=float(positive.sum())
                    net=float(delta.sum())
                    if hid is not None and pos.any():
                        values=hidden_depth_contributions(merge_layers(v,hid),K,len(v.means),ref,pos)
                    else:
                        values=np.zeros((7,512,512),np.float32)
                    total_weight=values[0]
                    has_hidden=total_weight>0
                    use=pos&coverage&has_hidden
                    rel=values[1]
                    depthrows=[]
                    for category,mask in [('all',np.ones((512,512),bool))]+[(c,categories==j) for j,c in enumerate(names)]:
                        pp=pos&mask
                        uu=use&mask
                        ww=total_weight[uu].sum(dtype=np.float64)
                        bins=[]
                        for condition,channel in [(rel<-.03,3),(abs(rel)<=.03,4),(rel>.03,5)]:
                            bins.append(dict(pixel_count=int((uu&condition).sum()),pixel_fraction=float((uu&condition).sum()/max(int(uu.sum()),1)),positive_SSE_fraction=float(positive[uu&condition].sum()/max(float(positive[uu].sum()),1e-30)),hidden_optical_weight_fraction=float(values[channel][uu].sum(dtype=np.float64)/max(ww,1e-30))))
                        depthrows.append(dict(category=category,positive_pixels=int(pp.sum()),valid_hidden_and_explicit_reference_pixels=int(uu.sum()),no_hidden_contribution_pixels=int((pp&~has_hidden).sum()),outside_explicit_coverage_pixels=int((pp&~coverage).sum()),hidden_optical_total_weight=float(ww),valid_positive_SSE_total=float(positive[uu].sum()),relative_depth=numpy_stat(rel[uu]),weight_mean_relative_depth=numpy_stat(values[2][uu]),front=bins[0],within3percent=bins[1],behind=bins[2],strictly_front_pixel_fraction=float((rel[uu]<0).mean()) if uu.any() else None,strictly_front_optical_fraction=float(values[6][uu].sum(dtype=np.float64)/max(ww,1e-30))))
                    spatial=[]
                    for j,c in enumerate(names):
                        at=categories==j
                        signed=float(delta[at].sum())
                        inc=float(positive[at].sum())
                        spatial.append(dict(category=c,pixels=int(at.sum()),positive_pixels=int((pos&at).sum()),old_SSE=float(old[at].sum()),new_SSE=float(new[at].sum()),net_SSE_delta=signed,positive_SSE_delta=inc,negative_SSE_delta=float(np.where(neg&at,delta,0.).sum()),positive_increment_fraction=inc/positive_total if positive_total>0 else None,signed_net_fraction=signed/net if abs(net)>1e-12 else None))
                    row=dict(pair=label+'1-'+label+'0',old_SSE=float(old.sum()),new_SSE=float(new.sum()),net_SSE_delta=net,positive_SSE_delta=positive_total,negative_SSE_delta=float(np.where(neg,delta,0.).sum()),old_zero_PSNR=float(-10*np.log10(max(float(old.sum())/(3*512**2),1e-30))),new_zero_PSNR=float(-10*np.log10(max(float(new.sum())/(3*512**2),1e-30))),spatial=spatial,depth=depthrows)
                    rows.append(row)
                    if f==30 and rule=='P2' and case['fstop']==1.4:
                        arrays[label+'_SSE_delta']=delta.astype(np.float32)
                        arrays[label+'_hidden_dominant_relative_depth']=np.where(use,rel,np.nan)
                        arrays[label+'_positive_pixels']=pos
                    print('E_ZERO_DIAG',name,f,rule,case['id'],label,net,flush=True)
                if len(arrays)>1:
                    save(folder/'zero-error-maps.npz',**arrays)
                write(done,dict(scene=name,frame=f,id=case['id'],rule=rule,nominal_focus_m=nominal,fstop=case['fstop'],rows=rows,settings=str(OUT/'zero-diagnostic-settings.json'),no_new_render_groups=True))
                del hidden
                torch.cuda.empty_cache()
            del local
        del cloud,full,fullgt,allhidden
        torch.cuda.empty_cache()


def verify_rendering():
    """Direct native checks of the added statistics and black-fill identity."""
    guard(True)
    from src.aperture_renderer import GaussianScene
    from src.explicit_layer_interface import hidden_depth_contributions
    K=tensor(np.array([[711.1111,0,256],[0,711.1111,256],[0,0,1]],np.float32))
    means=tensor(np.array([[0,0,2.],[0,0,1.9],[0,0,3.2]],np.float32))
    quat=tensor(np.array([[1,0,0,0]]*3,np.float32))
    scale=tensor(np.array([[.02,.02,.0001]]*3,np.float32))
    opacity=tensor(np.array([.2,.4,.6],np.float32))
    colors=tensor(np.array([[0,0,0],[1,0,0],[0,0,1]],np.float32))
    s=GaussianScene(means,quat,scale,opacity,colors,torch.zeros(3,device='cuda',dtype=torch.long))
    selection=np.zeros((512,512),bool)
    selection[256,256]=True
    values=hidden_depth_contributions(s,K,1,np.full((512,512),2.,np.float32),selection)
    with torch.inference_mode():
        rgb,_,_=raster(s,K)
    direct=rgb[256,256].cpu().numpy()
    error=float(abs(values[3:6,256,256]-direct).max())
    assert error<2e-6,error
    assert abs(values[1,256,256]+.05)<1e-6
    name=SCENES[0]
    desc=desc_for(name)
    entry=next(r for r in desc['entries'] if r['frame']==30)
    case=next(c for c in entry['cases'] if c['nominal_focus_m']==5 and c['fstop']==1.4)
    folder=cd(name,30,case,'P2')
    d=load(entry['input_npz'])
    K=tensor(d['K'])
    depth=load(fd(name,30)/'explicit-depth.npz')['depth']
    full=pixel_layer(tensor(d['rgb_linear']),tensor(depth),K,torch.eye(4,device='cuda'))
    focus=json.loads((folder/'render.json').read_text())['focus_m']
    bg=tensor(load(fd(name,30)/'fill-input.npz')['background_depth'])
    with torch.inference_mode():
        direct=sampled(full,K,torch.eye(4,device='cuda'),512,512,focus,.05/2.8,64,torch.zeros((512,512,3),device='cuda'),bg,far_plane=1e5)
    saved=load(folder/'D0-black.npz')
    black_error=float(abs(direct['rgb_linear'].cpu().numpy()-saved['rgb_linear']).max())
    assert black_error<2e-6,black_error
    result=dict(passed=True,hidden_depth_bin_weight_vs_native_RGB_max_error=error,dominant_hidden_relative_depth=float(values[1,256,256]),black_D0_exact_residual_subtraction_vs_native_max_linear_error=black_error,scene=name,frame=30,no_new_method_groups=True)
    write(OUT/'render-statistic-verification.json',result)
    print('E_VERIFY',json.dumps(result),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['validation','prepare','upper','render','evaluate','statistics','follow_evaluate','validation_prepare_fill','validation_finalize','zero_diagnostics','verify_rendering'])
    a=p.parse_args()
    OUT.mkdir(exist_ok=True)
    globals()[a.mode]()
