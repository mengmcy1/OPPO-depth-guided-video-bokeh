"""Frozen S1 inference/export and input-only SHARP depth preparation.

Official model files are imported from the data disk and remain unmodified.
Depth/pose retrieval uses no target hidden data. GT focus selection and the
single declared Oracle scale are explicit, separate from native geometry.
"""
import argparse
import hashlib
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
from gsplat import rasterization
from scipy.ndimage import distance_transform_edt
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
OFFICIAL=ROOT.parent/'ml-sharp-s1-20261111'
sys.path.insert(0,str(OFFICIAL/'src'))
from src.aperture_renderer import GaussianScene
from src.layered_aperture import pixel_layer,merge_layers
from src.analytic_aperture import sampled
from src.dual_layer import depth_band
from src.dual_layer_retrieval import retrieve
from route3_upper_bound_evaluate import BASE,OUT as UPPER,RECORDS,display,exr,items
from route3_target_focus import target_mask
from route3_b2_oracle_3d import sparse_pixel_layer
OUT=ROOT/'outputs/route3-sharp-s1-2026-11-11-01'
PREFIX='route3-sharp-s1-2026-11-11'
CRITERIA=RECORDS/'route3-sharp-s1-criteria-2026-11-11.json'
WEIGHT=ROOT/'models/SHARP/sharp_2572gikvuh.pt'
FRAMES=[1,6,11,16,21,26,30,35,40,45,50,55,60]


def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def scene(frame,layer='full',scale=1.):
    values=torch.load(OUT/'gaussians'/f'f{frame:03d}.pt',map_location='cuda',weights_only=True)
    selection=slice(None) if layer=='full' else values['source_frames']==(0 if layer=='front' else 1)
    fields={k:v[selection] for k,v in values.items()}
    fields['means']=fields['means']*scale;fields['scales']=fields['scales']*scale
    return GaussianScene(**fields)


def raster(cloud,K,mode='antialiased',eps=.0015):
    image,alpha,_=rasterization(cloud.means,cloud.quats,cloud.scales,cloud.opacities,cloud.colors,
        torch.eye(4,device='cuda')[None],K[None],512,512,packed=False,render_mode='RGB+ED',
        rasterize_mode=mode,eps2d=eps,near_plane=1e-4,far_plane=1e5)
    return image[0,...,:3],image[0,...,3],alpha[0,...,0]


def inference(c):
    from sharp.models import create_predictor,PredictorParams
    from sharp.cli.predict import predict_image
    from sharp.utils.gaussians import Gaussians3D
    from sharp.utils.gsplat import GSplatRenderer
    (OUT/'gaussians').mkdir(parents=True,exist_ok=False);(OUT/'depth').mkdir()
    start=time.monotonic();state=torch.load(WEIGHT,map_location='cpu',weights_only=True)
    model=create_predictor(PredictorParams());model.load_state_dict(state,strict=True);del state
    model.eval().to('cuda');torch.cuda.synchronize();load_seconds=time.monotonic()-start
    pin=BASE/'inputs/B2-high/pinhole';timing={r['frame']:r for r in json.loads((pin/'timings.json').read_text())}
    K=torch.tensor([[512*50/36,0,256],[0,512*50/36,256],[0,0,1]],device='cuda',dtype=torch.float32);rows=[]
    for frame in FRAMES:
        source=pin/(timing[frame]['key']+'.exr');raw=np.clip(exr(source),0,1).astype(np.float32);rgb=display(raw)
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();started=time.monotonic()
        cloud=predict_image(model,rgb*255.,float(K[0,0]),torch.device('cuda'));torch.cuda.synchronize();elapsed=time.monotonic()-started
        count=cloud.mean_vectors.shape[1];half=count//2
        assert count==2*768*768
        for value in cloud:assert bool(torch.isfinite(value).all())
        values=dict(means=cloud.mean_vectors[0],quats=cloud.quaternions[0],scales=cloud.singular_values[0],
            opacities=cloud.opacities[0].reshape(-1),colors=cloud.colors[0],source_frames=torch.arange(2,device='cuda').repeat_interleave(half))
        if frame in [1,30,60]:torch.save({k:v.detach().cpu() for k,v in values.items()},OUT/'gaussians'/f'f{frame:03d}.pt')
        front=GaussianScene(**{k:v[:half] for k,v in values.items()})
        with torch.inference_mode():
            color,depth,alpha=raster(front,K)
        depth=depth.cpu().numpy();a=alpha.cpu().numpy();valid=np.isfinite(depth)&(depth>0)&(a>=.1)
        if not valid.any():raise RuntimeError('S1_STOP front depth has no valid pixels')
        filled=depth.copy()
        if not valid.all():
            index=distance_transform_edt(~valid,return_distances=False,return_indices=True);filled[~valid]=depth[tuple(index[:,~valid])]
        path=OUT/'depth'/f'f{frame:03d}.npz';np.savez_compressed(path,depth=filled,raw_depth=depth,valid=valid,alpha=a,rgb_srgb=rgb,rgb_linear=raw,K=K.cpu().numpy())
        agreement=None
        if frame in [1,30,60]:
            # Match both renderers' parameters to isolate coordinate/export,
            # before comparing the project's common antialiased renderer.
            with torch.inference_mode():
                official=GSplatRenderer(color_space='linearRGB',low_pass_filter_eps=0.)(cloud,torch.eye(4,device='cuda')[None],K[None],512,512)
                project,dep,al=raster(GaussianScene(**values),K,'classic',0.)
                difference=float((official.color[0].permute(1,2,0)-torch.tensor(display(project.cpu().numpy()),device='cuda')).abs().max())
                alpha_difference=float((official.alpha[0,0]-al).abs().max())
            agreement=dict(max_abs_sRGB=difference,max_abs_alpha=alpha_difference)
            if difference>2e-5 or alpha_difference>2e-5:raise RuntimeError('S1_STOP official/export camera or RGB mismatch')
        rows.append(dict(frame=frame,input=str(source),depth=str(path),seconds_official_predict_and_unproject=elapsed,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
            Gaussian_count=count,front_count=half,filled_depth_pixels=int((~valid).sum()),
            block1_not_behind_paired_block0_fraction=float((values['means'][half:,2]<=values['means'][:half,2]).float().mean()),
            official_export_agreement=agreement,gaussians_saved=frame in [1,30,60]))
        write(OUT/'inference-progress.json',dict(completed=False,rows=rows));print('SHARP_INFER',frame,elapsed,rows[-1]['peak_allocated_MiB'],flush=True)
        del cloud,values,front;torch.cuda.empty_cache()
    record=dict(completed=True,rows=rows,model_loading_seconds=load_seconds,strict_load=True,model_or_architecture_modified=False,
        code_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=OFFICIAL,text=True).strip(),input_f_px=512*50/36,inference_resolution=[1536,1536],original_resolution=[512,512])
    write(OUT/'inference.json',record);write(RECORDS/(PREFIX+'-inference.json'),record)


def prepare(c):
    # Scalar Oracle fitting is isolated; native maps remain byte-for-byte.
    gt=[];pred=[]
    for frame in [1,30,60]:
        g=np.load(BASE/'B2-high/geometry'/f'f{frame:03d}-center.npz');d=np.load(OUT/'depth'/f'f{frame:03d}.npz')
        valid=g['center_visible']&np.isfinite(d['depth'])&(d['depth']>0)
        gt.append(g['depth_axial_m'][valid]);pred.append(d['depth'][valid])
    oracle=float(np.median(np.concatenate(gt))/np.median(np.concatenate(pred)))
    write(OUT/'scale.json',dict(native=1.,oracle_global=oracle,formula='median GT finite visible depth pooled3 / median paired SHARP depth',per_frame_or_region_fitting=False,diagnostic_only=True))
    root=OUT/'prepared';root.mkdir(exist_ok=False);rows=[];fills=[]
    for label,scale in [('native',1.),('oracle_scale',oracle)]:
        views={}
        for variant in ['E1','E2']:
            views[variant]=[]
            for frame in FRAMES:
                d=np.load(OUT/'depth'/f'f{frame:03d}.npz')
                if variant=='E1':pose=np.load(BASE/'B2-high/geometry'/f'f{frame:03d}-center.npz')['camera_c2w']
                else:pose=np.load(UPPER/'U2-da3/E2'/f'f{frame:03d}.npz')['c2w']
                views[variant].append(dict(frame=frame,depth=d['depth']*scale,K=d['K'],c2w=pose,rgb_linear=d['rgb_linear']))
        for frame in [1,30,60]:
            data=np.load(OUT/'depth'/f'f{frame:03d}.npz');z=data['depth']*scale;K=data['K'];union=np.zeros(z.shape,bool);maxradius=0.;cases=[]
            for case in [x for x in items() if x['frame_id']==frame]:
                # No GT depth value enters native geometry or band. This mask
                # only simulates selecting the subject to focus on.
                selection=target_mask(case);focus=float(np.median(z[selection]));band,_,radius=depth_band(z,K[0,0],focus,case['aperture_radius_m'])
                union|=band;maxradius=max(maxradius,float(radius.max()))
                cases.append(dict(id=case['id'],focus_m=focus,nominal_focus_m=case['focus_distance_m'],fstop=case['fstop'],band_pixels=int(band.sum())))
            mask=cv2.dilate(union.astype(np.uint8),np.ones((5,5),np.uint8),iterations=2)>0
            extension=int(np.ceil(2*maxradius))+4;bg=cv2.dilate(z,np.ones((2*extension+1,2*extension+1),np.uint8))
            name=f'{label}-f{frame:03d}';folder=root/name;folder.mkdir()
            path=folder/'inputs.npz';np.savez_compressed(path,depth=z,K=K,rgb_srgb=data['rgb_srgb'],rgb_linear=data['rgb_linear'],fill_mask=mask,background_depth=bg)
            parents={};audits={}
            for variant in ['E1','E2']:
                target=next(x for x in views[variant] if x['frame']==frame)
                parent,audit=retrieve(z,K,target['c2w'],union,views[variant],frame)
                p=folder/('parents-'+variant+'.npz');np.savez_compressed(p,**parent);parents[variant]=str(p);audits[variant]=audit
            rows.append(dict(id=name,frame=frame,scale=scale,scale_label=label,input_npz=str(path),cases=cases,parents=parents,audits=audits,
                fill_mask_pixels=int(mask.sum()),background_extension_radius_px=extension,GT_hidden_used=False,GT_scale_used=label=='oracle_scale'))
            fills.append(dict(id=name,input_npz=str(path),uses_GT_segmentation=False,uses_true_hidden_depth=False,Oracle_scale=label=='oracle_scale'))
            print('SHARP_PREPARE',name,[(x['nominal_focus_m'],x['fstop'],x['focus_m']) for x in cases],flush=True)
    write(OUT/'prepared.json',dict(completed=True,rows=rows,scale=oracle));write(OUT/'fill-inputs.json',dict(items=fills))
    write(RECORDS/(PREFIX+'-preparation.json'),dict(completed=True,rows=rows,single_global_Oracle_scale=oracle))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['inference','prepare'],required=True);a=p.parse_args()
    c=json.loads(CRITERIA.read_text());assert c['frozen'];torch.set_num_threads(8);cv2.setNumThreads(1)
    if a.mode=='inference':
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise ValueError('Explicit GPU0')
        subprocess.run(['nvidia-smi'],check=True)
    globals()[a.mode](c)


if __name__=='__main__':main()
