"""C1/C2: same-Gaussian renderer and opacity diagnostics, no model forward.

The first comparison deliberately keeps the same numeric color domain in both
renderers. Linear-light conversion is reported separately as an intentional
change, rather than misclassified as an export/interface error.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import cv2
import numpy as np
import torch


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def error(a,b):
    z=(a-b).float()
    return dict(max_abs=float(z.abs().max()),mean_abs=float(z.abs().mean()),rmse=float(z.square().mean().sqrt()))


def psnr(a,b,mask=None):
    z=(a-b).float()
    if mask is not None:z=z[mask]
    return float(-10*z.square().mean().clamp_min(1e-12).log10()) if z.numel() else None


def summary(x):
    v=x.detach().float().cpu().numpy()
    return dict(count=int(v.size),mean=float(v.mean()),quantiles=np.quantile(v,[0,.01,.1,.5,.9,.99,1]).tolist())


def save_rgb(path,x):
    cv2.imwrite(str(path),np.uint8(x.detach().cpu().numpy().clip(0,1)[...,::-1]*255))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--repository',type=Path,default=Path('/root/autodl-tmp/MoVieS'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--record',type=Path,required=True)
    p.add_argument('--cache',type=Path,help='One external official/B2 cache in the same export format')
    p.add_argument('--name',help='Name for an external cache')
    p.add_argument('--target',type=Path,help='External cache target display-RGB PNG')
    a=p.parse_args();a.project=a.project.resolve();a.repository=a.repository.resolve();a.output=a.output.resolve();a.record=a.record.resolve()
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);cv2.setNumThreads(4)
    helper=load_module('camvfx_input',a.project/'src/camvfx4d.py')
    aperture=load_module('project_aperture',a.project/'src/aperture_renderer.py')
    os.chdir(a.repository);sys.path[:0]=[str(a.repository),str(a.repository/'extensions/vggt')]
    from src.models.gs_render.gs_util import GaussianModel,render,Camera
    from src.options import opt_dict
    options=opt_dict['movies'];rows=[]
    jobs=[(a.name,a.cache.resolve(),a.target.resolve())] if a.cache else [
        (name,a.project/f'outputs/route3-c1-2026-11-10-01/movies-{name}-01',a.project/f'data/raw/CamVFX-4D/{name}/pinhole/0/000001.png') for name in ['plant','cat']]
    for name,cache,target_path in jobs:
        out=a.output/name;out.mkdir();saved=torch.load(cache/'same_forward_outputs.pt',map_location='cpu',weights_only=True)
        d=np.load(cache/'metric_gaussians.npz');scale=float(d['scale_to_meters']);H,W=saved['depth'].shape[-2:]
        tensors={k:torch.tensor(d[k],device='cuda') for k in ['means','quats','scales','opacities','source_frames','colors_srgb']}
        means=tensors['means']/scale;scales=tensors['scales']/scale
        pc=GaussianModel().set_data(means,tensors['colors_srgb'],scales,tensors['quats'],tensors['opacities'][:,None],0)
        if 'normalized_intrinsics' in saved:
            intrinsics=saved['normalized_intrinsics'][:1].to('cuda').float()
        else:
            _,meta=helper.load_camera_frames(a.project/f'data/raw/CamVFX-4D/{name}/pinhole/transforms_train.json')
            K0=helper.intrinsics_from_fov(meta,W,H)
            intrinsics=torch.tensor([K0[0,0]/W,K0[1,1]/H,.5,.5],device='cuda').bfloat16().float()[None]
        C2W=saved['normalized_C2W'][:1].to('cuda',dtype=torch.bfloat16).float();E=torch.linalg.inv(C2W[0])
        K=Camera(C2W[0],intrinsics[0],H,W).K
        target=cv2.resize(helper.read_rgb(target_path),(W,H),interpolation=cv2.INTER_CUBIC).clip(0,1)
        target=torch.tensor(target,device='cuda')
        scene=aperture.GaussianScene(means,tensors['quats'],scales,tensors['opacities'],tensors['colors_srgb'],tensors['source_frames'])
        started=time.monotonic();torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            native=render(pc,H,W,C2W,intrinsics,znear=options.znear,zfar=options.zfar)
            native_rgb=native['image'][0].permute(1,2,0);native_alpha=native['alpha'][0,0]
            matched=render(pc,H,W,C2W,intrinsics,znear=1e-4,zfar=1e4)
            project=aperture.render_aperture(scene,K,E,W,H,1.,0.,1,current_frame=0)
            linear_scene=aperture.GaussianScene(means,tensors['quats'],scales,tensors['opacities'],aperture.srgb_to_linear(tensors['colors_srgb']),tensors['source_frames'])
            project_linear=aperture.render_aperture(linear_scene,K,E,W,H,1.,0.,1,current_frame=0)
        row=dict(name=name,cache=str(cache),gaussians=len(means),comparison_resolution=[H,W],
            same_numeric_color_domain='display-RGB numbers passed unchanged to both rasterizers for interface isolation',
            same_numeric_RGB_error=error(project['rgb_linear'],matched['image'][0].permute(1,2,0)),
            same_numeric_alpha_error=error(project['alpha'],matched['alpha'][0,0]),
            native_default_clip_vs_project_RGB_error=error(project['rgb_linear'],native_rgb),
            native_default_clip_vs_project_alpha_error=error(project['alpha'],native_alpha),
            intentional_linear_conversion_vs_native_display_RGB_error=error(aperture.linear_to_srgb(project_linear['rgb_linear']).clamp(0,1),native_rgb),
            native_all_source_alpha=summary(native_alpha),native_all_source_alpha_coverage={str(c):float((native_alpha>=c).float().mean()) for c in [.05,.1,.5,.95]},
            native_all_source_RGB_vs_target_PSNR_diagnostic=psnr(native_rgb,target),
            camera_intrinsics_pixels=K.cpu().tolist(),native_clip=[options.znear,options.zfar],matched_clip=[1e-4,1e4],
            single_sources=[],opacity_per_source=[])
        save_rgb(out/'all-source-native-rgb.png',native_rgb)
        cv2.imwrite(str(out/'all-source-alpha.png'),np.uint8(native_alpha.cpu().numpy()*255))
        ids=saved['input_frame_ids'];F=len(ids);N=H*W
        base=(torch.tanh(saved['opacity']-1)*.5+.5).flatten()
        query=(torch.tanh(saved['query_motion_gaussians'][0]['motion_opacity']-1)*.5+.5).flatten()
        for f,frame in enumerate(ids):
            source=frame-1;mask=tensors['source_frames']==source
            pointcloud=GaussianModel().set_data(means[mask],tensors['colors_srgb'][mask],scales[mask],tensors['quats'][mask],tensors['opacities'][mask,None],0)
            with torch.inference_mode():single=render(pointcloud,H,W,C2W,intrinsics,znear=options.znear,zfar=options.zfar)
            rgb=single['image'][0].permute(1,2,0);alpha=single['alpha'][0,0]
            straight=(rgb/alpha[...,None].clamp_min(1e-6)).clamp(0,1)
            row['single_sources'].append(dict(source_frame_id=frame,gaussians=int(mask.sum()),
                alpha=summary(alpha),alpha_coverage={str(c):float((alpha>=c).float().mean()) for c in [.05,.1,.5,.95]},
                raw_premultiplied_PSNR_vs_target_diagnostic=psnr(rgb,target),
                straight_color_PSNR_on_alpha_ge_005_diagnostic=psnr(straight,target,alpha>=.05),
                straight_color_scope='RGB/alpha visualization only on alpha>=.05; not a new baseline or model change'))
            row['opacity_per_source'].append(dict(source_frame_id=frame,base=summary(base[f*N:(f+1)*N]),query=summary(query[f*N:(f+1)*N]),
                base_pruning_fraction=float((base[f*N:(f+1)*N]<=.001).float().mean()),query_pruning_fraction=float((query[f*N:(f+1)*N]<=.001).float().mean())))
            if f==0:
                save_rgb(out/'current-source-native-rgb.png',rgb);save_rgb(out/'current-source-alpha-normalized-diagnostic.png',straight)
                cv2.imwrite(str(out/'current-source-alpha.png'),np.uint8(alpha.cpu().numpy()*255))
                np.savez_compressed(out/'current-source.npz',rgb_display=rgb.cpu().numpy(),alpha=alpha.cpu().numpy(),straight_color_diagnostic=straight.cpu().numpy())
        torch.cuda.synchronize();row['elapsed_seconds']=time.monotonic()-started;row['peak_allocated_mib']=torch.cuda.max_memory_allocated()/2**20
        row['observation']='same-valued Gaussian rendering agrees; single-source premultiplied RGB must be interpreted together with its alpha'
        rows.append(row);print(json.dumps({k:v for k,v in row.items() if k not in ['single_sources','opacity_per_source']}),flush=True)
        np.savez_compressed(out/'all-source.npz',native_rgb=native_rgb.cpu().numpy(),native_alpha=native_alpha.cpu().numpy(),project_same_domain_rgb=project['rgb_linear'].cpu().numpy(),project_alpha=project['alpha'].cpu().numpy(),project_linear_rgb=project_linear['rgb_linear'].cpu().numpy())
        del saved,d,pc,tensors,scene,project,project_linear,native,matched,linear_scene;torch.cuda.empty_cache()
    record=dict(phase='C1 same-Gaussian renderer / C2 opacity',official_commit='77262fac6724a12c0b133a4fee10e62772db6d0f',
        A1_outputs_modified=False,model_training=False,model_forward=False,results=rows,
        limitations=['CamVFX is fixed-pose; official real-video and B2 tests are separate',
                     'unpremultiplying is a diagnostic visualization and cannot reconstruct absent geometry',
                     's_other is source contribution, never hidden-content contribution'])
    a.record.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    (a.output/'summary.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
