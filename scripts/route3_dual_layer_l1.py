"""Frozen L1 renderer checks, representative predictions and CPU evaluation.

The probe subset is fixed by geometry/time before scoring. A numerical
artifact stop prevents generating the temporal extension or starting L2.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
import cv2
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from src.aperture_renderer import GaussianScene,linear_to_srgb
from src.layered_aperture import pixel_layer,merge_layers
from src.analytic_aperture import analytic,analytic_single,sampled,expanded_projection
from src.dual_layer import depth_band,warp_visible,compose_visible_hidden
from src.c1_bokehme import region_sse,patch_metrics
from route3_b2_oracle_3d import sparse_pixel_layer
from route3_c1_bokehme_oracle import lpips_cpu
from types import SimpleNamespace

MODES=['analytic1','analytic4','fixed16','fixed64','random16']


def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')


def display(x):return np.where(x<=.0031308,12.92*x,1.055*np.maximum(x,1e-12)**(1/2.4)-.055).clip(0,1).astype(np.float32)


def checks(a):
    a.output.mkdir(parents=True,exist_ok=False)
    from gsplat import rasterization
    K=torch.tensor([[32.,0,16.],[0,32.,16.],[0,0,1.]],device='cuda');E=torch.eye(4,device='cuda')
    cloud=GaussianScene(torch.tensor([[0.,0,2.]],device='cuda'),torch.tensor([[1.,0,0,0]],device='cuda'),
        torch.tensor([[.04,.04,.001]],device='cuda'),torch.tensor([.6],device='cuda'),torch.tensor([[.2,.4,.7]],device='cuda'),torch.zeros(1,device='cuda',dtype=torch.long))
    out=analytic_single(cloud,K,E,32,32,2.,0.)
    native,alpha,_=rasterization(cloud.means,cloud.quats,cloud.scales,cloud.opacities,cloud.colors,E[None],K[None],32,32,
        packed=False,render_mode='RGB+ED',rasterize_mode='antialiased',eps2d=.0015,near_plane=1e-4,far_plane=1e4)
    support=alpha[0,...,0]>.004
    core=float(abs(out['rgb_linear'][support]-native[0,...,:3][support]).max())
    maximum=float(abs(out['rgb_linear']-native[0,...,:3]).max())
    if core>2e-5 or maximum>1/255+.0002:raise ValueError(('Triton/native mismatch',core,maximum))
    _,_,conic,_,_,_=expanded_projection(cloud,K,E,32,32,4.,.2)
    v=conic[0,0];cov=torch.linalg.inv(torch.tensor([[v[0],v[1]],[v[1],v[2]]],device='cuda'))
    expected=.64**2+.0015+1.6**2/4
    cocerr=float(abs(cov.diag()-expected).max())
    fill=torch.zeros(32,32,3,device='cuda');background=torch.full((32,32),6.,device='cuda')
    scalar=analytic(cloud,K,E,32,32,4.,.2,1,fill,background)
    field=analytic(cloud,K,E,32,32,torch.full((32,32),4.,device='cuda'),.2,1,fill,background)
    field_error=float(abs(scalar['rgb_linear']-field['rgb_linear']).max());assert field_error<1e-6
    m=4/(3*math.pi);quad_total=(.25-m*m)+m*m
    assert abs(quad_total-.25)<1e-12 and cocerr<1e-5
    plane=torch.ones(32,32,device='cuda')*4;blue=torch.tensor([0.,0,1.],device='cuda').expand(32,32,3)
    hidden=pixel_layer(blue,plane,K,E)
    xy=torch.tensor([.03,-.02],device='cuda');view=E.clone();view[:2,3]-=xy;intr=K.clone();intr[:2,2]+=K.diag()[:2]*xy/3
    image,ha,_=rasterization(hidden.means,hidden.quats,hidden.scales,hidden.opacities,hidden.colors,view[None],intr[None],32,32,
        packed=False,render_mode='RGB+ED',rasterize_mode='antialiased',eps2d=.0015)
    ha=ha[0,...,0];hp=image[0,...,:3];hz=image[0,...,3]
    source_z=torch.ones_like(plane)*2;source_z[:,16:]=-1
    red=torch.tensor([1.,0,0.],device='cuda').expand(32,32,3);green=torch.tensor([0.,1.,0.],device='cuda').expand(32,32,3)
    vp=warp_visible(red,source_z,K,3.,xy);composed,reverse=compose_visible_hidden(vp,hp,hz,1-ha,green)
    expected_vp=vp['alpha'][...,None]*red+(1-vp['alpha'][...,None])*(hp+(1-ha[...,None])*green)
    interface_err=float(abs(composed-expected_vp).max());energy_err=float(abs(composed.sum(-1)-1).max())
    assert interface_err<1e-6 and energy_err<1e-6 and not bool(reverse.any())
    record=dict(passed=True,Triton_native_unblurred_core_RGB_max_error=core,Triton_native_all_RGB_max_error=maximum,
        threshold_difference='Triton1e-6 vs native1/255; core comparison excludes native threshold tails',
        CoC_covariance_diagonal_max_error=cocerr,constant_focus_field_vs_scalar_max_error=field_error,quadrant_total_variance=.25,quadrant_centroid=m,
        V_P_same_subaperture_hidden_GS_RGB_max_error=interface_err,V_P_hidden_GS_unit_energy_max_error=energy_err,
        V_P_warp_hole_pixels=int((vp['alpha']==0).sum()),V_P_geometry='foreground halfplanez2, hidden GSz4, same off-axis lens sample; composited then checked against alpha formula',
        differentiable_existing_renderer_unchanged=True,analytic_prototype_backward=False,Cycles_metrics_computed=False)
    write(a.record,record);print('RENDERER_CHECKS',record,flush=True)


def get_items(a):
    xs=[x for x in json.loads((a.base/'manifests/oracle.json').read_text())['items'] if x['fstop'] in ([.5,1.4,2.] if a.include_pressure else [1.4,2.])]
    if a.probe:xs=[x for x in xs if x['scene']=='B2-high' and x['frame_id'] in [1,30] and x['fstop']==1.4]
    return xs


def render(a,c):
    a.output.mkdir(parents=True,exist_ok=False);xs=get_items(a);groups={};rows=[]
    reuse={}
    if a.reuse:
        reuse={(r['id'],r['cloud'],r['mode']):r for r in json.loads(a.reuse.read_text())['rows']}
    for x in xs:groups.setdefault((x['scene'],x['frame_id']),[]).append(x)
    profile=c['profile'];repetitions=c['efficiency']['timed_repetitions']
    for (scene,frame),items in sorted(groups.items()):
        d=np.load(items[0]['input_npz']);C=torch.tensor(d['visible_rgb_linear'],device='cuda');Z=torch.tensor(d['center_depth'],device='cuda')
        K=torch.tensor(d['K'],device='cuda',dtype=torch.float32);cam=torch.tensor(d['camera_c2w'],device='cuda',dtype=torch.float32);E=torch.linalg.inv(cam)
        h,w=Z.shape;visible=pixel_layer(C,Z,K,cam,torch.tensor(d['center_surface_valid'],device='cuda'))
        parent=np.load(a.base/'methods-3d/prepared'/f'{scene}-f{frame:03d}'/'parents.npz')
        hidden=sparse_pixel_layer(parent,parent['recoverable'],K,cam,profile,'cuda')
        fill_data=np.load(a.base/'methods-2d/fill'/f'{scene}-f{frame:03d}'/'fill-image.npz');fill=torch.tensor(fill_data['fill_rgb_linear'],device='cuda');bg=torch.tensor(d['background_depth'],device='cuda')
        for x in items:
            focus=x['focus_distance_m'];aperture=x['aperture_radius_m']
            for kind,pc in [('O5',visible),('O6-GT-rec',merge_layers(visible,hidden))]:
                for mode in MODES:
                    if (x['id'],kind,mode) in reuse:
                        previous=reuse[(x['id'],kind,mode)]
                        if not Path(previous['prediction']).is_file():raise FileNotFoundError(previous['prediction'])
                        rows.append(dict(previous,reused_same_configuration=True));continue
                    def run():
                        if mode.startswith('analytic'):return analytic(pc,K,E,w,h,focus,aperture,int(mode[-1]),fill,bg)
                        return sampled(pc,K,E,w,h,focus,aperture,64 if mode=='fixed64' else 16,fill,bg,
                            random_seed=20261110+frame if mode=='random16' else None)
                    with torch.inference_mode():
                        warm=run();del warm;torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                        times=[]
                        for _ in range(repetitions):
                            start=time.monotonic();result=run();torch.cuda.synchronize();times.append(time.monotonic()-start)
                        pred=linear_to_srgb(result['rgb_linear']).clamp(0,1).cpu().numpy()
                    dest=a.output/x['id']/kind/mode;dest.mkdir(parents=True)
                    np.savez_compressed(dest/'prediction.npz',prediction_srgb=pred,rgb_linear=result['rgb_linear'].cpu().numpy(),alpha=result['alpha'].cpu().numpy())
                    cv2.imwrite(str(dest/'prediction.png'),np.uint8(pred[...,::-1]*255))
                    row=dict(id=x['id'],scene=scene,frame=frame,focus_m=focus,fstop=x['fstop'],cloud=kind,mode=mode,
                        Gaussians=len(pc.means),render_seconds=times,median_seconds=float(np.median(times)),peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
                        component_statistics=result.get('component_statistics'),prediction=str(dest/'prediction.npz'))
                    rows.append(row);print('L1_RENDERED',x['id'],kind,mode,row['median_seconds'],flush=True)
                    del result,pred
                write(a.output/'runtime.json',dict(completed=False,rows=rows,probe=a.probe))
        del visible,hidden,parent;torch.cuda.empty_cache()
    write(a.record,dict(completed=True,rows=rows,probe=a.probe,criteria=str(a.criteria),modes=c['modes'],training=False))


def evaluate(a,c):
    runtime=json.loads(a.input.read_text());items={x['id']:x for x in get_items(a)}
    model=lpips_cpu(SimpleNamespace(alexnet_weights=Path('/root/autodl-tmp/.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth')))
    rows=[];images={};masks_by={};GT={};proofs=[];a.output.mkdir(parents=True,exist_ok=False)
    reuse={(r['id'],r['cloud'],r['mode']):r['metrics'] for r in json.loads(a.reuse.read_text())['rows']} if a.reuse else {}
    for r in runtime['rows']:
        x=items[r['id']];d=np.load(x['input_npz']);g=np.load(x['geometry_gt_npz'])
        if r['id'] not in GT:
            target=display(cv2.imread(x['gt_rgb_exr'],cv2.IMREAD_UNCHANGED)[...,:3][...,::-1]);GT[r['id']]=target
            band,edge,R=depth_band(d['center_depth'],d['K'][0,0],x['focus_distance_m'],x['aperture_radius_m'])
            masks_by[r['id']]=dict(full_image=np.ones(band.shape,bool),in_focus=d['center_surface_valid']&(R<=1),out_of_focus=d['center_surface_valid']&(R>=2),band=band,recoverable=g['mask_recoverable_total'])
        target=GT[r['id']];masks=masks_by[r['id']];p=np.load(r['prediction']);pred=p['prediction_srgb']
        if not np.isfinite(pred).all():raise ValueError('Nonfinite method output')
        stats=reuse.get((r['id'],r['cloud'],r['mode']))
        if stats is None:
            pm=patch_metrics(pred,target,np.ones((512,512),bool),model,patch_size=64,device='cpu');score=np.array(pm['patch_lpips_scores'])
            stats={}
            for name,m in masks.items():
                support=np.array([m[y:y+64,z:z+64].sum() for y in range(0,512,64) for z in range(0,512,64)])
                stats[name]=dict(**region_sse(pred,target,m),patch_lpips=float(score@support/support.sum()) if support.sum() else None)
        rows.append(dict(**r,metrics=stats));images.setdefault((r['id'],r['cloud']),{})[r['mode']]=pred
        print('L1_SCORED',r['id'],r['cloud'],r['mode'],stats['full_image']['psnr'],stats['band']['psnr'],flush=True)
    # Per-case artifact diagnosis is frozen, independent of pooled quality.
    for (ident,kind),views in images.items():
        x=items[ident];band=masks_by[ident]['band'];base=next(r for r in rows if r['id']==ident and r['cloud']==kind and r['mode']=='fixed64')
        for mode in ['analytic1','analytic4']:
            current=next(r for r in rows if r['id']==ident and r['cloud']==kind and r['mode']==mode)
            linear=np.load(current['prediction'])['rgb_linear'];raw=cv2.imread(x['gt_rgb_exr'],cv2.IMREAD_UNCHANGED)[...,:3][...,::-1]
            err=np.abs(linear-np.clip(raw,0,1)).mean(-1);p90=float(np.quantile(err[band],.9))
            drop=base['metrics']['band']['psnr']-current['metrics']['band']['psnr']
            proofs.append(dict(id=ident,cloud=kind,mode=mode,band_PSNR_drop_to_fixed64_dB=drop,linear_band_MAE_p90=p90,numerical_stop_candidate=drop>=3 and p90>=.10))
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    plots=[]
    for (ident,kind),views in images.items():
        m=masks_by[ident]['band'];y,x=np.nonzero(m);cy,cx=int(np.median(y)),int(np.median(x))
        closest=int(np.argmin((y-cy)**2+(x-cx)**2));cy,cx=int(y[closest]),int(x[closest]);y0=max(0,min(cy-64,384));x0=max(0,min(cx-64,384))
        fig,axes=plt.subplots(2,6,figsize=(16,6));allviews={'GT':GT[ident],**views}
        for j,name in enumerate(['GT']+MODES):
            axes[0,j].imshow(allviews[name]);axes[1,j].imshow(allviews[name][y0:y0+128,x0:x0+128]);axes[0,j].set_title(name)
            for i in [0,1]:axes[i,j].axis('off')
        fig.suptitle(ident+' '+kind+'; identical GT-band-selected crop');fig.tight_layout();path=a.output/(ident+'-'+kind+'.png');fig.savefig(path,dpi=120);plt.close(fig);plots.append(str(path))
    write(a.record,dict(completed=True,probe=runtime['probe'],rows=rows,artifact_candidates=proofs,visualizations=plots,
        criteria=str(a.criteria),criteria_changed=False,stop_requires_visual_confirmation=True,old_B2_results_changed=False))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['check','render','evaluate'],required=True)
    p.add_argument('--base',type=Path,default=ROOT/'outputs/route3-b2only-2026-11-10-01');p.add_argument('--criteria',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--record',type=Path,required=True);p.add_argument('--input',type=Path);p.add_argument('--probe',action='store_true');p.add_argument('--include-pressure',action='store_true');p.add_argument('--reuse',type=Path);a=p.parse_args()
    c=json.loads(a.criteria.read_text());assert c['frozen'] is True
    cv2.setNumThreads(1);torch.set_num_threads(8 if a.mode=='evaluate' else 4)
    if a.record.exists():raise FileExistsError(a.record)
    if a.mode!='evaluate':
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise ValueError('Explicit GPU0 required')
        subprocess.run(['nvidia-smi'],check=True)
    if a.mode=='check':checks(a)
    elif a.mode=='render':render(a,c)
    else:evaluate(a,c)


if __name__=='__main__':main()
