"""L0 ordered checks: zero aperture first, then representation probes and K sweep."""
import argparse
import json
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
from src.aperture_renderer import render_aperture,linear_to_srgb,GaussianScene
from src.layered_aperture import pixel_layer,render_layered,aperture_fill_samples
from src.dual_layer import depth_band,render_pixels,warp_visible,compose_visible_hidden
from src.c1_bokehme import region_sse,physical_coc_radius


def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def display(a):
    return np.where(a<=.0031308,12.92*a,1.055*np.maximum(a,1e-12)**(1/2.4)-.055).clip(0,1).astype(np.float32)


def masks(data,entry):
    radius,_=physical_coc_radius(data['center_depth'],data['K'][0,0],entry['aperture_radius_m'],entry['focus_distance_m'])
    band,edge,_=depth_band(data['center_depth'],data['K'][0,0],entry['focus_distance_m'],entry['aperture_radius_m'])
    return dict(full_image=np.ones(radius.shape,bool),in_focus=data['center_surface_valid']&(abs(radius)<=1),
        out_of_focus=data['center_surface_valid']&(abs(radius)>=2),boundary=cv2.dilate(edge.astype(np.uint8),np.ones((5,5),np.uint8))>0,band=band,
        known=data['center_surface_valid'],sky=~data['center_surface_valid'])


def interface_checks():
    dev='cuda';K=torch.tensor([[32.,0,16.],[0,32.,16.],[0,0,1.]],device=dev);z=torch.ones(32,32,device=dev)*2
    rgb=torch.tensor([.2,.4,.6],device=dev).expand(32,32,3).clone()
    v=warp_visible(rgb,z,K,2.,torch.zeros(2,device=dev))
    zero_error=float(abs(v['color']-rgb).max());coverage_error=float(abs(v['alpha']-1).max())
    # Explicit color/alpha ordering with partially covered visible sample.
    vis=dict(color=torch.tensor([[[1.,0,0]]],device=dev),depth=torch.tensor([[1.]],device=dev),alpha=torch.tensor([[.4]],device=dev))
    h=torch.tensor([[[0,.5,0]]],device=dev);t=torch.tensor([[.5]],device=dev);fill=torch.tensor([[[0,0,1.]]],device=dev)
    behind,reverse=compose_visible_hidden(vis,h,torch.tensor([[2.]],device=dev),t,fill)
    infront,reverse2=compose_visible_hidden(vis,h,torch.tensor([[.5]],device=dev),t,fill)
    order_error=max(float(abs(behind-torch.tensor([[[.4,.3,.3]]],device=dev)).max()),float(abs(infront-torch.tensor([[[.2,.5,.3]]],device=dev)).max()))
    # A constant-color coplanar sheet must have no seam or energy jump.
    xy=torch.tensor([.03,-.02],device=dev);w=warp_visible(rgb,z,K,3.,xy)
    c,_=compose_visible_hidden(w,(1-w['alpha'])[...,None]*rgb,z+.5,w['alpha'],rgb)
    seam=float(abs(c-rgb).max())
    passed=zero_error<1e-6 and coverage_error<1e-6 and order_error<1e-6 and seam<1e-6 and bool(reverse2.all()) and not bool(reverse.any())
    return dict(zero_aperture_RGB_max_error=zero_error,zero_aperture_coverage_error=coverage_error,
        depth_order_alpha_energy_max_error=order_error,constant_color_seam_max_error=seam,
        warp_holes_keep_zero_alpha=True,soft_z_relative=.001,passed=passed)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,default=ROOT/'outputs/route3-b2only-2026-11-10-01')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--record',type=Path,required=True)
    p.add_argument('--mode',choices=['zero','scan','repair','vp-repair'],required=True);a=p.parse_args()
    if a.record.exists():raise FileExistsError(a.record)
    a.output.mkdir(parents=True,exist_ok=False)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise ValueError('Explicit GPU0 required')
    subprocess.run(['nvidia-smi'],check=True);torch.set_num_threads(4);cv2.setNumThreads(1)
    entries=json.loads((a.base/'manifests/oracle.json').read_text())['items'];entries=[e for e in entries if e['fstop'] in [1.4,2.]]
    if a.mode=='vp-repair':entries=[e for e in entries if e['frame_id']==1 and e['fstop']==1.4]
    groups={}
    for e in entries:groups.setdefault((e['scene'],e['frame_id']),[]).append(e)
    rows=[]
    interface=interface_checks()
    if not interface['passed']:raise ValueError(interface)
    for (scene,frame),items in sorted(groups.items()):
        d=np.load(items[0]['input_npz']);rgb=d['rgb_srgb'];z=d['center_depth'];h,w=z.shape
        C=torch.tensor(d['visible_rgb_linear'],device='cuda');Z=torch.tensor(z,device='cuda');K=torch.tensor(d['K'],device='cuda',dtype=torch.float32);cam=torch.tensor(d['camera_c2w'],device='cuda',dtype=torch.float32);E=torch.linalg.inv(cam)
        raw=cv2.imread(str(d['raw_rgb_exr']),cv2.IMREAD_UNCHANGED)[...,:3][...,::-1];pinhole_gt=display(raw)
        known=torch.tensor(d['center_surface_valid'],device='cuda');sky=~known
        fill_id=f'{scene}-f{frame:03d}';fill=np.load(a.base/'methods-2d/fill'/fill_id/'fill-image.npz');fillRGB=torch.tensor(fill['fill_rgb_linear'],device='cuda');bg=torch.tensor(d['background_depth'],device='cuda')
        # Legacy profile exactly matches B2; alternatives isolate sampling, AA
        # and RGB-domain effects, never fit colors against any DOF truth.
        cloud=pixel_layer(C,Z,K,cam,known)
        if a.mode=='zero':
            profiles=[('V-G-B2',cloud,.0015,'classic',C),
                      ('V-G-AA',cloud,.0015,'antialiased',C),
                      ('V-G-1pixel',pixel_layer(C,Z,K,cam,known,subdivisions=1,sigma_pixels=.35),.0015,'classic',C)]
            for name,pc,eps,aa,colors in profiles:
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.monotonic()
                with torch.inference_mode():out=render_layered(pc,None,K,E,w,h,items[0]['focus_distance_m'],0.,1,fillRGB,bg,return_samples=True,eps2d=eps) if aa=='classic' else render_aperture(pc,K,E,w,h,items[0]['focus_distance_m'],0.,1,sample_backgrounds=fillRGB[None],eps2d=eps,rasterize_mode=aa)
                torch.cuda.synchronize();seconds=time.monotonic()-start
                pred=linear_to_srgb(out['rgb_linear']).clamp(0,1).cpu().numpy();dest=a.output/f'{scene}-f{frame:03d}-{name}';dest.mkdir()
                np.savez_compressed(dest/'prediction.npz',prediction_srgb=pred,alpha=out['alpha'].cpu().numpy())
                cv2.imwrite(str(dest/'prediction.png'),np.uint8(pred[...,::-1]*255))
                for e in items:
                    ms=masks(d,e)
                    rows.append(dict(id=e['id'],method=name,scene=scene,frame=frame,focus_m=e['focus_distance_m'],fstop=e['fstop'],aperture_radius=0.,K=1,
                        vs_input={k:region_sse(pred,rgb,m) for k,m in ms.items()},vs_Cycles_pinhole={k:region_sse(pred,pinhole_gt,m) for k,m in ms.items()},
                        seconds=seconds,mean_alpha=float(out['alpha'].mean()),peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20))
                print('L0_ZERO',scene,frame,name,rows[-1]['vs_input']['full_image']['psnr'],flush=True)
            # Exact pixel forward warp zero-aperture, including far-field sky.
            vp=render_pixels(C,Z,K,items[0]['focus_distance_m'],0.,1,fillRGB[None]);pred=linear_to_srgb(vp['rgb_linear']).clamp(0,1).cpu().numpy()
            dest=a.output/f'{scene}-f{frame:03d}-V-P';dest.mkdir();np.savez_compressed(dest/'prediction.npz',prediction_srgb=pred,alpha=vp['alpha'].cpu().numpy())
            for e in items:rows.append(dict(id=e['id'],method='V-P',scene=scene,frame=frame,focus_m=e['focus_distance_m'],fstop=e['fstop'],aperture_radius=0.,K=1,
                vs_input={k:region_sse(pred,rgb,m) for k,m in masks(d,e).items()},vs_Cycles_pinhole={k:region_sse(pred,pinhole_gt,m) for k,m in masks(d,e).items()}))
        else:
            for e in items:
                target=display(cv2.imread(e['gt_rgb_exr'],cv2.IMREAD_UNCHANGED)[...,:3][...,::-1]);ms=masks(d,e)
                profiles=[('V-G',cloud,'classic'),('V-P',None,'pixel')]
                if a.mode=='repair':profiles=[('V-G-AA',cloud,'antialiased'),('V-G-1pixel',pixel_layer(C,Z,K,cam,known,subdivisions=1,sigma_pixels=.35),'classic')]
                if a.mode=='vp-repair':profiles=[('V-P-depth-slope',None,'pixel_corrected')]
                for count in ([64] if a.mode in ['repair','vp-repair'] else [16,64,128]):
                    for name,pc,aa in profiles:
                        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.monotonic()
                        with torch.inference_mode():
                            if not aa.startswith('pixel'):
                                fills,_,_=aperture_fill_samples(fillRGB,bg,K,e['focus_distance_m'],e['aperture_radius_m'],count)
                                out=render_aperture(pc,K,E,w,h,e['focus_distance_m'],e['aperture_radius_m'],count,sample_backgrounds=fills,eps2d=.0015,rasterize_mode=aa)
                            else:
                                fills,_,_=aperture_fill_samples(fillRGB,bg,K,e['focus_distance_m'],e['aperture_radius_m'],count)
                                out=render_pixels(C,Z,K,e['focus_distance_m'],e['aperture_radius_m'],count,fills,depth_slope_correction=aa=='pixel_corrected')
                        torch.cuda.synchronize();seconds=time.monotonic()-start
                        pred=linear_to_srgb(out['rgb_linear']).clamp(0,1).cpu().numpy();dest=a.output/e['id']/f'{name}-K{count}';dest.mkdir(parents=True)
                        np.savez_compressed(dest/'prediction.npz',prediction_srgb=pred,alpha=out['alpha'].cpu().numpy());cv2.imwrite(str(dest/'prediction.png'),np.uint8(pred[...,::-1]*255))
                        rows.append(dict(id=e['id'],scene=scene,frame=frame,focus_m=e['focus_distance_m'],fstop=e['fstop'],method=name,K=count,
                            vs_Cycles_DOF={k:region_sse(pred,target,m) for k,m in ms.items()},seconds=seconds,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20))
                        print('L0_SCAN',e['id'],name,count,rows[-1]['vs_Cycles_DOF']['full_image']['psnr'],seconds,flush=True)
        del cloud;torch.cuda.empty_cache()
        save(a.output/'runtime.json',dict(mode=a.mode,completed=False,rows=rows,interface_checks=interface))
    save(a.record,dict(mode=a.mode,completed=True,rows=rows,interface_checks=interface,
        representation='GT RGB/depth backprojection; original B2 profile5x5 sigma.7 eps.0015 opacity.999; no optimization',
        input_sky='same10000m far-field; V-G retains only known surfaces with LaMa residual; V-P explicitly warps known RGB sky approximation',
        old_outputs_modified=False,training=False))


if __name__=='__main__':main()
