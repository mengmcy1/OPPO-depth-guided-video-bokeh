"""Freeze stage-1b placement convention on a paired zero-aperture guard."""
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

from route3_ovg_subpixel import CONFIG, OUT, PREVIOUS, ROOT, save_json, save_npz

sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from src.footprint_order import apply_footprint_order, footprint_scale
from src.ordered_layers import order_hidden, ray_samples, render_fallbacks
from src.layered_aperture import pixel_layer, merge_layers
from src.light_candidate_fusion import LightCandidateFusion
from src.aperture_renderer import aperture_disk

from route3_g3_r14_train import train_network, learned_hidden
from route3_g3_geometry import load, zero_parts
from route3_s2a_baselines import scene
from route3_s2a_learn import Store
from route3_s2a_evaluate import masks
from route3_s2a_proxy import read as read_exr
from route3_upper_bound_evaluate import display, score
from route3_ldr_u1r import model as lpips
from src.dual_layer import depth_band

WORK=OUT/'W4'
SOURCE=PREVIOUS/'supervision'/'original'
CORE=PREVIOUS/'geometry'/'original'
LOG=ROOT/'logs/route3-ovg-1a-2026-11-18'


def read(p):
    return json.loads(p.read_text())


def guard(gpu=False):
    if shutil.disk_usage(OUT).free/2**30<60:
        raise RuntimeError('Data disk below 60 GiB')
    if gpu:
        assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
        subprocess.run(['nvidia-smi'],check=True)
        free,_=torch.cuda.mem_get_info()
        if free<24*2**30:
            raise RuntimeError('Concurrent GPU memory requirement unmet')
        torch.cuda.set_per_process_memory_fraction(.18)


def link(source,dest):
    dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists():
        dest.symlink_to(source.resolve(),target_is_directory=source.is_dir())


def query_net(path):
    net=LightCandidateFusion().cuda().eval()
    net.load_state_dict(torch.load(path,weights_only=True,map_location='cpu')['model'])
    return net


def context(name):
    base=CORE/name
    a=next(e for e in read(base/'prepared.json')['rows'] if e['frame']==30)
    d=load(a['input_npz'])
    T=lambda x:torch.as_tensor(x,device='cuda')
    K,z,fill,bg=map(T,[d['K'],d['depth'],load(a['fill'])['fill_rgb_linear'],d['background_depth']])
    A=scene(base/'visible'/'f030.pt')
    D=pixel_layer(T(d['rgb_linear']),z,K,torch.eye(4,device='cuda'))
    r,alpha=zero_parts(A,K)
    a_zero=r+(1-alpha[...,None])*fill
    rgb=np.asarray(d['rgb_srgb'])
    return a,d,A,D,K,z,fill,bg,a_zero,rgb


def zero(cloud, K, background, target):
    color,alpha=zero_parts(cloud,K)
    pred=display((color+(1-alpha[...,None])*background).cpu().numpy())
    mse=np.mean((pred.astype(np.float64)-target)**2)
    return float(-10*np.log10(max(mse,1e-30)))


def ray_coverage(hidden, K, focus, radius, geometry):
    """Average hidden-only opacity on lens rays labelled recoverable at edges."""
    from gsplat import rasterization
    masks=geometry['recoverable']
    # The existing K16 recoverable mask selects labeled hidden surface rays.
    total=int(masks.sum())
    if hidden is None or not total:
        return dict(rays=total,mean_hidden_alpha=0.)
    device=K.device
    S,H,W=masks.shape
    projected=aperture_disk(S,radius,device=device,dtype=K.dtype)
    alpha_sum=0.
    for i,xy in enumerate(projected):
        V=torch.eye(4,device=device)
        V[:2,3]-=xy
        J=K.clone(); J[:2,2]+=J.diag()[:2]*xy/focus
        _,alpha,_=rasterization(hidden.means,hidden.quats,hidden.scales,hidden.opacities,
                                hidden.colors,V[None],J[None],W,H,packed=False,
                                render_mode='RGB',rasterize_mode='antialiased',eps2d=.0015,
                                near_plane=1e-4,far_plane=1e5)
        alpha_sum+=float(alpha[0,:,:,0][torch.as_tensor(masks[i],device=device)].sum())
    return dict(rays=total,mean_hidden_alpha=alpha_sum/total)


@torch.inference_mode()
def paired_check():
    result=WORK/'decision.json'
    if result.exists():
        return read(result)
    guard(True)
    split=read(PREVIOUS/'split.json')
    net=query_net(PREVIOUS/'R3/original_geometry/step-020000.pt')
    data=Store()
    count=[]
    for name in split['heldout']:
        a,d,A,D,K,z,fill,bg,a_zero,rgb=context(name)
        manifest=read(PREVIOUS/'evaluation-input/original'/name/'512-13/manifest.json')
        d0=zero(D,K,a_zero,rgb)
        for c in a['cases']:
            if c['fstop']!=1.4:
                continue
            cid=c['id']
            row=next(x for x in manifest['rows'] if x['case']['id']==cid and x['case']['protocol']=='P2')
            radius=.05/(2*c['fstop'])
            band=depth_band(d['depth'],d['K'][0,0],c['focus_m'],radius)[0]
            h=learned_hidden(net,data,row,manifest,K,band)
            base,stats0=order_hidden(h,z,K,'O1')
            safe,stat=apply_footprint_order(h,z,K,CONFIG['footprint_min_sigma_px_512'])
            zero0=zero(merge_layers(D,base),K,a_zero,rgb)
            znew=zero(merge_layers(D,safe),K,a_zero,rgb)
            geom=load(CORE/name/'geometry'/(cid+'-K16.npz'))
            a1=ray_coverage(base,K,c['focus_m'],radius,geom)
            a2=ray_coverage(safe,K,c['focus_m'],radius,geom)
            count.append(dict(scene=name,case=cid,focus=c['nominal_focus_m'],D0_zero_PSNR=d0,center_zero_PSNR=zero0,
                              footprint_zero_PSNR=znew,drop_dB=znew-d0,center_coverage=a1,footprint_coverage=a2,**stat))
            print('W4_GUARD',name,cid,znew-d0,stat['shrunk'],stat['dropped'],flush=True)
    passed=all(r['drop_dB']>=-CONFIG['zero_aperture_max_drop_db'] for r in count)
    result_data=dict(selected='footprint' if passed else 'O1',passed=passed,rows=count,
                     threshold_DB=CONFIG['zero_aperture_max_drop_db'],
                     threshold_definition='each dev scene f030, both P2 f/1.4 focus cases; compare D+hidden to D0, all b=0',
                     reference_checkpoint=str((PREVIOUS/'R3/original_geometry/step-020000.pt').relative_to(ROOT)),
                     optical_coverage_definition='hidden-only alpha on GT recoverable K16 rays in the evaluation aperture')
    save_json(result,result_data)
    return result_data


def update_targets(path,query_depth,K,minimum,dest,native_size):
    labels=load(path)
    correct=(labels['content_hit_fraction']>0)&labels['placement_support_v2'].astype(bool)
    # Use the same reference pixel footprint as the actual hidden renderer,
    # scaling pixels with resolution so 256 labels describe that geometry.
    camera=labels['candidate_camera_target'][correct]
    T=torch.tensor(camera,dtype=torch.float32)
    k=torch.as_tensor(K,dtype=torch.float32)
    z= T[:,2]
    sigma=(z/K[0,0])*.7*native_size/512
    scale,keep=footprint_scale(T,sigma[:,None].repeat(1,2),sigma*.01,torch.as_tensor(query_depth),k,minimum)
    nominal_support=labels['placement_support_v2'].copy()
    nominal_support[correct]=keep.cpu().numpy()
    factors=np.ones_like(labels['candidate_depth_target'],np.float16)
    factors[correct]=scale.cpu().numpy().astype(np.float16)
    save_npz(dest,**dict(labels,placement_support_v2=nominal_support,footprint_tangent_factor=factors))
    return dict(source=str(path.relative_to(ROOT)),true_background=int(correct.sum()),
                dropped_true_background=int((~keep).sum()),shrunk_true_background=int((keep&(scale<1)).sum()))


def relabel(name,kind='256-5'):
    root=WORK/'footprint-labels'/name
    if kind!='256-5':
        root=root/kind
    ready=root/'manifest.json'
    if ready.exists():
        return read(ready)
    rowdata=read(SOURCE/'learning-data'/name/kind/'manifest.json')
    audit=[]
    current=None
    for row in rowdata['rows']:
        if row['input']!=current:
            z=load(row['input']) # only observed depth and intrinsics are used below
            current=row['input']
        p=Path(row['target'])
        target=root/f'f{row["frame"]:03d}'/p.name
        result=update_targets(p,z['target_depth'],z['target_K'],
                             CONFIG['footprint_min_sigma_px_512']/2,target,256)
        audit.append(dict(id=row['case']['id'],**result))
        row['target']=str(target)
    save_json(ready,rowdata)
    save_json(root/'audit.json',dict(scene=name,rows=audit))
    return rowdata


def make_training(choice):
    split=read(PREVIOUS/'split.json')
    train=WORK/'retrain-data'
    torch.set_num_threads(2)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(relabel,split['train']+split['heldout']))
    for n in split['train']+split['heldout']:
        root=train/'learning-data'/n
        chosen=WORK/'footprint-labels'/n if choice=='footprint' else SOURCE/'learning-data'/n/'256-5'
        link(chosen,root/'256-5')
    first=split['train'][0]
    warmup='256-5-GTwarmup'
    if choice=='footprint':
        relabel(first,warmup)
        warmroot=WORK/'footprint-labels'/first/warmup
    else:
        warmroot=SOURCE/'learning-data'/first/warmup
    link(warmroot,train/'learning-data'/first/warmup)
    return train,split


@torch.inference_mode()
def report_retrain():
    raw=read(WORK/'decision.json')
    selected=raw['selected']
    net=query_net(WORK/'retrained/step-020000.pt')
    metric=lpips()
    store=Store()
    rows=[]
    for row in raw['rows']:
        name,cid=row['scene'],row['case']
        a,d,A,D,K,z,fill,bg,a_zero,rgb=context(name)
        manifest=read(PREVIOUS/'evaluation-input/original'/name/'512-13/manifest.json')
        p=next(x for x in manifest['rows'] if x['case']['id']==cid and x['case']['protocol']=='P2')
        c=p['case']; radius=.05/(2*c['fstop']);zf=c['focus_m']
        band=depth_band(d['depth'],d['K'][0,0],zf,radius)[0]
        h=learned_hidden(net,store,p,manifest,K,band)
        if selected=='footprint':
            h,stats=apply_footprint_order(h,z,K,CONFIG['footprint_min_sigma_px_512'])
        else:
            h,stats=order_hidden(h,z,K,'O1')
        coupled=merge_layers(D,h)
        reference=PREVIOUS/'truth'/name
        gt=display(read_exr(reference/'dof'/(cid+'.exr')))
        areas=masks(reference,dict(c,frame=30))
        samples=ray_samples(A,K,zf,radius,64,fill,bg)
        image=render_fallbacks(coupled,K,zf,radius,64,fill,bg,samples)['F2']
        pred=display(image.cpu().numpy())
        before=read(PREVIOUS/'R3/renders/original'/name/'P2'/cid/'scores.json')
        val=score(pred,gt,{k:areas[k] for k in ['recoverable','band','in_focus','full_image']},metric)
        a0={r['region']:r for r in before['rows'] if r['method']=='original_geometry'}
        rows.append(dict(scene=name,case=cid,zero_PSNR=zero(coupled,K,a_zero,rgb),**stats,
                        regions={k:dict(metrics=v,previous=r,
                                        PSNR_change=v['psnr']-r['psnr'] if v['psnr'] is not None and r['psnr'] is not None else None)
                                 for k,v in val.items() if (r:=a0[k])}))
        save_npz(WORK/'retrained-predictions'/name/(cid+'.npz'),prediction_srgb=pred)
    save_json(WORK/'retrain-evaluation.json',dict(selected=selected,rows=rows))


def main():
    WORK.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(3)
    with (LOG/'gpu0-W4.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        chosen=paired_check()['selected']
        train,split=make_training(chosen)
        guard(True)
        dest=WORK/'retrained'
        if not (dest/'training.json').exists():
            train_network(train,dest,split['train'],split['heldout'],require_descent=True)
        if not (WORK/'retrain-evaluation.json').exists():
            report_retrain()


if __name__=='__main__':
    main()
