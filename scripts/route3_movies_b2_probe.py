"""C3/C4: official posed sample and GT-normalized B2 diagnostic input.

Preparation is CPU-only. Inference uses the same complete pretrained MoVieS
checkpoint, no training or source-fusion design. B2 units follow the actual
dataset canonical-camera + mean-input-XYZ normalization, using GT solely to
make the known-motion diagnostic interpretable in meters.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
import cv2
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
FRAMES=np.array([1,6,11,16,21,26,30,35,40,45,50,55,60])


def write(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def read_rgb_exr(path):
    x=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
    if x is None:raise FileNotFoundError(path)
    x=np.clip(x[...,:3][...,::-1],0,1)
    return np.where(x<=.0031308,12.92*x,1.055*np.maximum(x,1e-12)**(1/2.4)-.055).clip(0,1).astype(np.float32)


def prepare(a):
    a.output.mkdir(parents=True,exist_ok=False);base=a.base.resolve();scene='B2-high';size=518
    rows=json.loads((base/'inputs'/scene/'pinhole/timings.json').read_text())
    byframe={int(r['frame']):r for r in rows};geometries=[];images=[];cameras=[];sum_norm=0.;count=0
    first=np.load(base/scene/'geometry/f001-center.npz')['camera_c2w'];canonical=np.linalg.inv(first)
    for frame in FRAMES:
        gpath=base/scene/f'geometry/f{frame:03d}-center.npz';g=np.load(gpath);geometries.append(str(gpath))
        image=read_rgb_exr(base/'inputs'/scene/'pinhole'/(byframe[int(frame)]['key']+'.exr'))
        images.append(cv2.resize(image,(size,size),interpolation=cv2.INTER_CUBIC).clip(0,1).transpose(2,0,1))
        camera=canonical@g['camera_c2w'];cameras.append(camera)
        depth=g['depth_axial_m'];K=g['K'];valid=g['center_visible']
        y,x=np.mgrid[:depth.shape[0],:depth.shape[1]]
        ray=np.stack([x+.5,y+.5,np.ones_like(x)],-1)@np.linalg.inv(K).T
        xyz=(ray*depth[...,None])@camera[:3,:3].T+camera[:3,3]
        sum_norm+=float(np.linalg.norm(xyz[valid],axis=-1).sum());count+=int(valid.sum())
    mean_norm=sum_norm/(count+1e-6);scaling_factor=1/(mean_norm+1e-6)
    cameras=np.stack(cameras);cameras[:,:3,3]*=scaling_factor
    original=np.load(geometries[0]);K=original['K'];intrinsics=np.tile([K[0,0]/512,K[1,1]/512,K[0,2]/512,K[1,2]/512],(13,1))
    source_scene=ROOT/'outputs/route3-a2b-2026-11-10-01/blender-b2'/scene/'scene-geometry.npz'
    np.savez_compressed(a.output/'posed-input.npz',images=np.stack(images),C2W=cameras.astype(np.float32),
        fxfycxcy=intrinsics.astype(np.float32),frame_ids=FRAMES,times=(FRAMES-1)/59,
        metric_scale_to_world=1/scaling_factor,canonical_to_world=first,geometry_paths=np.array(geometries),source_scene_geometry=np.array(str(source_scene)),
        dataset=np.array('B2-high'),GT_geometry_normalization=np.array(True))
    record=dict(dataset=scene,input_frame_ids=FRAMES.tolist(),resolution=[size,size],finite_GT_points=count,
        canonical_transform=np.asarray(canonical).tolist(),mean_canonical_XYZ_norm_m=mean_norm,
        world_to_model_scale=scaling_factor,model_to_world_scale_m=1/scaling_factor,
        normalization_source='MoVieS src/data/base_dataset.py:_camera_normalize; dynamicreplica_dataset.py:134-145 norm_xyz with camera_norm_unit=1',
        depth_mask='all finite first-hit center surfaces in13frames; no sky; no dataset-specific depth quantile filtering',
        input_geometry_permission='GT poses and GT scale used to normalize diagnostic input; GT depths are not fed as image/depth features',
        GT_RGB_policy='same clipped scene-linear -> standard-sRGB input as B2 methods',
        posthoc_depth_only_scale_fit=False,inference_run=False)
    restored=cameras.astype(np.float32).astype(np.float64);restored[:,:3,3]/=scaling_factor
    restored=first[None]@restored;original_cameras=np.stack([np.load(p)['camera_c2w'] for p in geometries])
    record['stored_pose_inverse_check']=dict(scope='storedFP32 normalized poses before intentional modelBF16 conditioning',
        inverse_camera_translation_max_error_m=float(abs(restored[:,:3,3]-original_cameras[:,:3,3]).max()),
        inverse_camera_rotation_max_abs_error=float(abs(restored[:,:3,:3]-original_cameras[:,:3,:3]).max()))
    write(a.output/'normalization.json',record);write(a.record,record);print(json.dumps(record),flush=True)


def motion_check(saved,input_data,output,record):
    source=np.load(str(input_data['source_scene_geometry']));ids=source['object_ids'].tolist();dynamic=source['dynamic_object_ids'].tolist()
    matrices=source['object_world_matrices'];R0=input_data['canonical_to_world'][:3,:3];scale=float(input_data['metric_scale_to_world'])
    motion=saved['query_motion'][0,:,:,:3].numpy();H,W=motion.shape[-2:];frame_ids=input_data['frame_ids'].tolist()
    query_indices=[0,6,12];results=[]
    # Run this CPU stage in the existing dofgs environment with Embree.
    spec=importlib.util.spec_from_file_location('b2_gt_geometry',ROOT/'scripts/route3_blender_b2_geometry.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    world=module.World(Path(str(input_data['source_scene_geometry'])),512)
    if H!=W:raise ValueError('B2 rigid-surface motion validation expects the square diagnostic input')
    source_world=module.World(Path(str(input_data['source_scene_geometry'])),W)
    for source_index,frame in enumerate(frame_ids):
        # Recast source GT at nominal model pixel centers instead of assigning a
        # neighboring512px surface to a518px model pixel at object boundaries.
        g=source_world.center(frame,alpha_samples=1);obj=g['object_id'];local=g['local_surface_point']
        for qi,query_index in enumerate(query_indices):
            query=frame_ids[query_index];pred=motion[qi,source_index].transpose(1,2,0)*scale
            pred=pred@R0.T
            for category,object_ids in [('dynamic',dynamic),('static',[i for i in ids if i not in dynamic])]:
                mask=np.isin(obj,object_ids)&np.isfinite(local).all(-1)
                if not mask.any():continue
                delta=np.zeros((int(mask.sum()),3));query_points=np.zeros_like(delta);labels=obj[mask];points=local[mask]
                for oid in object_ids:
                    subset=labels==oid
                    if not subset.any():continue
                    i=ids.index(oid);old=matrices[frame-1,i];new=matrices[query-1,i]
                    query_points[subset]=points[subset]@new[:3,:3].T+new[:3,3]
                    delta[subset]=query_points[subset]-(points[subset]@old[:3,:3].T+old[:3,3])
                error=np.linalg.norm(pred[mask]-delta,axis=-1);gt_norm=np.linalg.norm(delta,axis=-1)
                row=dict(source_frame=frame,query_frame=query,category=category,pixels=int(mask.sum()),
                    metric_displacement_endpoint_error_m_p50_p90=np.quantile(error,[.5,.9]).tolist(),
                    GT_displacement_norm_m_p50_p90=np.quantile(gt_norm,[.5,.9]).tolist(),
                    predicted_displacement_norm_m_p50_p90=np.quantile(np.linalg.norm(pred[mask],axis=-1),[.5,.9]).tolist())
                if category=='dynamic':
                    visible,_,_,_=world.visible(query,query_points,labels,tolerance=1e-4)
                    row['query_visible_dynamic_points']=int(visible.sum())
                    row['query_occluded_or_outside_dynamic_points']=int((~visible).sum())
                    row['query_visible_endpoint_error_m_p50_p90']=np.quantile(error[visible],[.5,.9]).tolist() if visible.any() else None
                    # Isolate motion on known source surface points; source-depth
                    # error is deliberately not added to this correspondence probe.
                    predicted_points=query_points-delta+pred[mask]
                    camera=world.cameras[query-1];K=world.K
                    def project(x):
                        cp=(x-camera[:3,3])@camera[:3,:3];uvz=cp@K.T
                        with np.errstate(divide='ignore',invalid='ignore'):uv=uvz[:,:2]/uvz[:,2:]
                        return uv,cp[:,2]>0
                    uv,positive=project(predicted_points);gtuv,_=project(query_points)
                    good=visible&positive&np.isfinite(uv).all(-1)
                    row['supported_2D_correspondence_points']=int(good.sum())
                    row['motion_on_GT_source_point_reprojection_error_pixels_p50_p90']=np.quantile(np.linalg.norm(uv[good]-gtuv[good],axis=-1),[.5,.9]).tolist() if good.any() else None
                results.append(row)
    record['motion_GT_comparison']=dict(results=results,definition='same source-pixel rigid surface local point transformed at source/query frames; query-source world displacement',
        units='meters, using the same known model-to-world scale and first-camera rotation as input normalization',
        no_M2_design=True,source_support='only source-visible finite surface points; GT displacement from rigid transforms even when target occluded, never filled with zero for invisibility',
        target_visibility='dynamic correspondence only: exact mesh first-hit/GT-object visibility, tolerance1e-4m, inside512image',
        source_sampling='Exact mesh first-hit source rays at nominal518pixel centers; RGB was bicubic-resized512-to518; target visibility/reprojection at native512',
        limitation='BF16 conditioning of model poses/intrinsics remains; displacement/GT-source reprojection do not by themselves validate predicted source depth')


def infer(a):
    a.output.mkdir(parents=True,exist_ok=False);input_data=np.load(a.input.resolve());images=np.asarray(input_data['images'],np.float32)
    if images.shape[0]!=13 or max(images.shape[-2:])>518:raise ValueError('Use exactly13 posed frames and long edge<=518')
    if not np.isfinite(images).all() or images.min()<0 or images.max()>1:raise ValueError('Display RGB[0,1] required')
    frames=input_data['frame_ids'].tolist() if 'frame_ids' in input_data else list(range(1,14))
    times=np.asarray(input_data['times'],np.float32) if 'times' in input_data else np.linspace(0,1,13,dtype=np.float32)
    query_indices=[0,6,12];query_times=times[query_indices];C2W=np.asarray(input_data['C2W'],np.float32);intr=np.asarray(input_data['fxfycxcy'],np.float32)
    data=dict(images=torch.from_numpy(images)[None],C2W=torch.from_numpy(C2W)[None],intrinsics=torch.from_numpy(intr)[None],
              times=torch.from_numpy(times)[None],queries=torch.from_numpy(query_times)[None])
    os.chdir(a.repository);sys.path[:0]=[str(a.repository),str(a.repository/'extensions/vggt')]
    from src.options import opt_dict
    from src.models import SplatRecon
    from src.models.gs_render.gs_util import GaussianModel,render
    from extensions.vggt.vggt.models.vggt import VGGT
    from safetensors.torch import load_file
    VGGT.from_pretrained=classmethod(lambda cls,*args,**kwargs:cls());opt=opt_dict['movies'];opt.memory_efficient_attention=False
    start=time.monotonic();model=SplatRecon(opt,load_lpips=False);loading=model.load_state_dict(load_file(str(a.weights)),strict=True)
    model.eval().to('cuda:0');load_seconds=time.monotonic()-start
    data={k:v.to('cuda:0',dtype=torch.bfloat16) for k,v in data.items()}
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.monotonic()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        outputs,motion,motion_gs=model.backbone(data['images'],data['C2W'],data['intrinsics'],data['times'],data['queries'],frames_chunk_size=4)
    torch.cuda.synchronize();forward_seconds=time.monotonic()-start;peak=torch.cuda.max_memory_allocated()/2**20
    saved={k:v.float().cpu() for k,v in outputs.items()};saved.update(query_motion=motion.float().cpu(),
        query_motion_gaussians=[{k:v.float().cpu() for k,v in g.items()} for g in motion_gs],
        normalized_C2W=torch.from_numpy(C2W),normalized_intrinsics=data['intrinsics'][0].float().cpu(),input_frame_ids=frames,
        query_times=torch.from_numpy(query_times),input_times=torch.from_numpy(times),original_head_dtypes={k:str(v.dtype) for k,v in outputs.items()})
    torch.save(saved,a.output/'same-forward-outputs.pt')
    H,W=images.shape[-2:];renders=[]
    for qi,index in enumerate(query_indices):
        render_inputs=dict(outputs,offset=motion[:,qi,:,:3]);render_inputs.update(motion_gs[qi])
        with torch.inference_mode():result=model.gs_renderer.render(render_inputs,data['C2W'],data['intrinsics'],data['C2W'][:,index:index+1],data['intrinsics'][:,index:index+1],return_pc=True)
        pc=result['pc'][0];op=model.gs_renderer.opacity_activation(render_inputs['motion_opacity'])[0].permute(0,2,3,1).reshape(-1)
        keep=op>opt.opacity_threshold;sources=(torch.tensor(frames,device='cuda')-1).repeat_interleave(H*W)[keep];current=sources==frames[index]-1
        part=GaussianModel().set_data(pc.xyz[current],pc.color[current],pc.scale[current],pc.rotation[current],pc.opacity[current],0)
        with torch.inference_mode():single=render(part,H,W,data['C2W'][0,index:index+1],data['intrinsics'][0,index:index+1],znear=opt.znear,zfar=opt.zfar)
        full_rgb=result['image'][0,0].permute(1,2,0).float();full_alpha=result['alpha'][0,0,0].float()
        one_rgb=single['image'][0].permute(1,2,0).float();one_alpha=single['alpha'][0,0].float()
        target=torch.tensor(images[index].transpose(1,2,0),device='cuda');error=lambda x:float(-10*(x-target).square().mean().clamp_min(1e-12).log10())
        for label,rgb,alpha in [('all',full_rgb,full_alpha),('current',one_rgb,one_alpha)]:
            stem=f'query-f{frames[index]:03d}-{label}';cv2.imwrite(str(a.output/(stem+'.png')),np.uint8(rgb.clamp(0,1).cpu().numpy()[...,::-1]*255))
            cv2.imwrite(str(a.output/(stem+'-alpha.png')),np.uint8(alpha.cpu().numpy()*255))
            np.savez_compressed(a.output/(stem+'.npz'),rgb_srgb=rgb.cpu().numpy(),alpha=alpha.cpu().numpy())
        renders.append(dict(query_frame=frames[index],query_time=float(query_times[qi]),all_source_gaussians=len(pc.xyz),current_source_gaussians=int(current.sum()),
            all_source_alpha_mean=float(full_alpha.mean()),current_source_alpha_mean=float(one_alpha.mean()),
            all_source_RGB_PSNR_diagnostic=error(full_rgb),current_source_premultiplied_RGB_PSNR_diagnostic=error(one_rgb),
            current_source_alpha_ge_05_fraction=float((one_alpha>=.5).float().mean()),
            all_source_alpha_ge_095_fraction=float((full_alpha>=.95).float().mean())))
    record=dict(dataset=a.name,input_path=str(a.input),input_shape=list(images.shape),input_frame_ids=frames,query_frame_ids=[frames[i] for i in query_indices],
        frame_number_semantics='B2 original one-based frame IDs' if 'frame_ids' in input_data else 'one-based slots in author NPZ; original DAVIS video frame numbers not supplied',
        input_times=times.tolist(),query_times=query_times.tolist(),strict_checkpoint=str(loading),official_commit='77262fac6724a12c0b133a4fee10e62772db6d0f',
        model_load_seconds=load_seconds,three_query_forward_seconds=forward_seconds,forward_peak_allocated_mib=peak,renders=renders,
        camera_input='authors preprocessed posed sample unchanged' if 'source_scene_geometry' not in input_data else 'GT poses canonicalized and scaled by actual dataset norm_xyz formula',
        current_source_note='current-source ablation retains cross-frame attention; not a single-frame reconstruction baseline',
        color_domain='official numeric display-RGB alpha composition; RGB and alpha reported separately',
        model_training=False,M2_design=False,existing_results_modified=False)
    record['C4_status']='separate CPU motion mode after this same forward; no second model inference'
    write(a.record,record);write(a.output/'summary.json',record);print(json.dumps(record),flush=True)


def render_error(a):
    """CPU metrics from saved C3 RGB/alpha on an independent GT dynamic mask."""
    data=np.load(a.input);frames=data['frame_ids'].tolist();H,W=data['images'].shape[-2:]
    spec=importlib.util.spec_from_file_location('b2_gt_render_geometry',ROOT/'scripts/route3_blender_b2_geometry.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    world=module.World(Path(str(data['source_scene_geometry'])),W);rows=[]
    if H!=W:raise ValueError('B2 dynamic-object render diagnostic expects square input')
    a.output.mkdir(parents=True,exist_ok=False)
    for index in [0,6,12]:
        frame=frames[index];g=world.center(frame,alpha_samples=1)
        mask=np.isin(g['object_id'],world.dynamic)&g['center_visible']
        target=data['images'][index].transpose(1,2,0).astype(np.float64)
        np.savez_compressed(a.output/f'f{frame:03d}-GT-dynamic-mask.npz',mask_dynamic=mask,object_id=g['object_id'],surface_valid=g['center_visible'])
        for mode in ['all','current']:
            path=a.cache/f'query-f{frame:03d}-{mode}.npz';prediction=np.load(path)
            rgb=prediction['rgb_srgb'].astype(np.float64);alpha=prediction['alpha']
            n=int(mask.sum());sse=float(((rgb[mask]-target[mask])**2).sum())
            mse=sse/(3*n) if n else None
            full_mse=float(((rgb-target)**2).mean())
            rows.append(dict(query_frame=frame,source_mode=mode,pixels=n,RGB_sse=sse,
                RGB_psnr_dB=float(-10*np.log10(max(mse,1e-12))) if mse is not None else None,
                alpha_mean=float(alpha[mask].mean()) if n else None,
                alpha_p10_p50_p90=np.quantile(alpha[mask],[.1,.5,.9]).tolist() if n else None,
                full_image_RGB_psnr_crosscheck_dB=float(-10*np.log10(max(full_mse,1e-12))),
                prediction_cache=str(path)))
    record=dict(phase='C4 known-dynamic-object rendering error',source_cache=str(a.cache),input_path=str(a.input),rows=rows,
        GT_mask='Exact first-hit mesh at nominal518pixel centers; declared dynamic-object IDs, source/query camera at that frame; hard center-ray mask, not antialiased alpha',
        dynamic_object_ids=world.dynamic.tolist(),color_domain='saved native premultiplied display RGB[0,1], no unpremultiply; alpha reported separately',
        target='same bicubic512-to518 clipped-linear/standard-sRGB B2 input used for C3 quality diagnostics',
        limits=['Bicubic RGB resampling can mix foreground/background at the hard518GT boundary',
                'Model poses/intrinsics use BF16 conditioning while GTmask uses nominal full-precision geometry',
                'RGB error includes coverage, source depth, geometry, appearance and motion; not an isolated motion-head score'],
        new_model_forward=False,new_model_render=False,model_training=False,M2_design=False)
    write(a.record,record);write(a.output/'summary.json',record);print(json.dumps(record),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['prepare-b2','infer','motion','render-error'],required=True)
    p.add_argument('--base',type=Path,default=ROOT/'outputs/route3-b2only-2026-11-10-01');p.add_argument('--input',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--record',type=Path,required=True);p.add_argument('--name',default='B2-high')
    p.add_argument('--repository',type=Path,default=Path('/root/autodl-tmp/MoVieS'));p.add_argument('--weights',type=Path,default=ROOT/'models/MoVieS/movies_ckpt.safetensors');p.add_argument('--cache',type=Path)
    a=p.parse_args();a.output=a.output.resolve();a.record=a.record.resolve();a.repository=a.repository.resolve();a.weights=a.weights.resolve()
    torch.set_num_threads(4);cv2.setNumThreads(4)
    if a.mode=='prepare-b2':prepare(a)
    elif a.mode=='render-error':
        if not a.input or not a.cache:raise ValueError('InputNPZ and saved C3 RGB/alpha cache required')
        render_error(a)
    elif a.mode=='motion':
        if not a.input or not a.cache:raise ValueError('InputNPZ and same-forward cache required')
        a.output.mkdir(parents=True,exist_ok=False)
        saved=torch.load(a.cache/'same-forward-outputs.pt',map_location='cpu',weights_only=True);data=np.load(a.input)
        record=dict(phase='C4 known-motion validation',source_cache=str(a.cache),input_path=str(a.input),new_model_inference=False,model_training=False,M2_design=False)
        motion_check(saved,data,a.output,record);write(a.record,record);write(a.output/'summary.json',record)
    else:
        if not a.input:raise ValueError('inputNPZ required')
        a.input=a.input.resolve();infer(a)


if __name__=='__main__':main()
