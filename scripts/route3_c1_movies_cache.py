"""Development-only MoVieS pass, with three query times and metric alignment.

No test-scene inference. Input RGB preserves 16-bit cat PNG values. The same
source IDs across queries permit motion-range classification without using a
single query's absolute offset as a dynamic label.
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
from safetensors.torch import load_file


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scene',type=Path,required=True)
    p.add_argument('--regions',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--repository',type=Path,default=Path('/root/autodl-tmp/MoVieS'))
    p.add_argument('--weights',type=Path,default=Path('models/MoVieS/movies_ckpt.safetensors'))
    p.add_argument('--view',type=int,default=0);a=p.parse_args()
    if a.scene.name not in ['plant','cat']:raise ValueError('Only the fixed development scenes are allowed')
    root=Path(__file__).resolve().parents[1]
    a.scene=a.scene.resolve();a.regions=a.regions.resolve();a.output=a.output.resolve();a.repository=a.repository.resolve();a.weights=a.weights.resolve()
    a.output.mkdir(parents=True,exist_ok=False)
    spec=importlib.util.spec_from_file_location('camvfx_input',root/'src/camvfx4d.py');helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    chosen=[1,5,9,13,17,21,26,30,34,38,42,46,50];size=518
    frames,metadata=helper.load_camera_frames(a.scene/'pinhole/transforms_train.json')
    cameras=np.stack([frames[(a.view,i)]['c2w'] for i in chosen]);original=cameras[0].copy();cameras=np.linalg.inv(cameras[:1])@cameras
    K=helper.intrinsics_from_fov(metadata,size,size).astype(np.float32)
    normalized_K=np.array([K[0,0]/size,K[1,1]/size,.5,.5],np.float32)
    images=np.stack([cv2.resize(helper.read_rgb(a.scene/f'pinhole/{a.view}/{i:06d}.png'),(size,size),interpolation=cv2.INTER_CUBIC).clip(0,1).transpose(2,0,1) for i in chosen])
    times=np.array([frames[(a.view,i)]['time'] for i in chosen],np.float32);times=(times-times.min())/(times.max()-times.min())
    queries=times[[0,6,12]]
    data=dict(images=torch.from_numpy(images)[None],C2W=torch.from_numpy(cameras).float()[None],
              intrinsics=torch.from_numpy(np.broadcast_to(normalized_K,(13,4)).copy())[None],
              times=torch.from_numpy(times)[None],query=torch.from_numpy(queries)[None])
    os.chdir(a.repository);sys.path[:0]=[str(a.repository),str(a.repository/'extensions/vggt')]
    from src.options import opt_dict
    from src.models import SplatRecon
    from extensions.vggt.vggt.models.vggt import VGGT
    VGGT.from_pretrained=classmethod(lambda cls,*args,**kwargs:cls())
    opt=opt_dict['movies'];opt.memory_efficient_attention=False
    start=time.monotonic();model=SplatRecon(opt,load_lpips=False);state=load_file(str(a.weights))
    loading=model.load_state_dict(state,strict=True);del state;model.eval().to('cuda:0');load_seconds=time.monotonic()-start
    data={k:v.to('cuda:0',dtype=torch.bfloat16) for k,v in data.items()}
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.monotonic()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        outputs,motion,motion_gs=model.backbone(data['images'],data['C2W'],data['intrinsics'],data['times'],data['query'],frames_chunk_size=4)
    torch.cuda.synchronize();seconds=time.monotonic()-start;peak=torch.cuda.max_memory_allocated()/2**20
    saved={k:v.float().cpu() for k,v in outputs.items()}
    saved.update(query_motion=motion.float().cpu(),query_motion_gaussians=[{k:v.float().cpu() for k,v in d.items()} for d in motion_gs],
                 input_frame_ids=chosen,input_frame_indices=[i-1 for i in chosen],input_times=torch.from_numpy(times),
                 query_times=torch.from_numpy(queries),original_camera=torch.from_numpy(original),normalized_C2W=torch.from_numpy(cameras))
    torch.save(saved,a.output/'same_forward_outputs.pt')
    region=np.load(a.regions/a.scene.name/str(a.view)/'background.npz')
    gt=cv2.resize(region['mesh_depth'],(size,size),interpolation=cv2.INTER_NEAREST)
    static=cv2.resize((~region['motion_masks'].any(0)).astype(np.uint8),(size,size),interpolation=cv2.INTER_NEAREST)>0
    prediction=outputs['depth'][0,0,0].float().cpu().numpy()
    valid=static&(gt>0)&np.isfinite(prediction)&(prediction>0)
    alignment=valid&(gt>=np.median(gt[valid]));scale=float(np.median(gt[alignment]/prediction[alignment]))
    relative=np.abs(prediction*scale-gt)/np.maximum(gt,1e-6)
    render_inputs=dict(outputs,offset=motion[:,0,:,:3]);render_inputs.update(motion_gs[0])
    with torch.inference_mode():
        rendered=model.gs_renderer.render(render_inputs,data['C2W'],data['intrinsics'],data['C2W'][:,:1],data['intrinsics'][:,:1],return_pc=True)
    pc=rendered['pc'][0]
    opacity=model.gs_renderer.opacity_activation(render_inputs['motion_opacity'])[0].permute(0,2,3,1).reshape(-1)
    keep=opacity>opt.opacity_threshold
    source_ids=(torch.tensor(chosen,device='cuda:0')-1).repeat_interleave(size*size)[keep]
    offsets=motion[0,:,:,:3].permute(0,1,3,4,2).reshape(len(queries),-1,3)[:,keep].float().cpu().numpy()
    np.savez(a.output/'metric_gaussians.npz',means=pc.xyz.float().cpu().numpy()*scale,quats=pc.rotation.float().cpu().numpy(),
        scales=pc.scale.float().cpu().numpy()*scale,opacities=pc.opacity.float().cpu().numpy().reshape(-1),colors_srgb=pc.color.float().cpu().numpy(),
        source_frames=source_ids.cpu().numpy(),source_flat_indices=keep.nonzero().flatten().cpu().numpy(),query_offsets_normalized=offsets,
        query_times=queries,input_frame_indices=np.array(chosen)-1,scale_to_meters=scale,canonical_to_dataset_world=original,
        K=helper.intrinsics_from_fov(metadata).astype(np.float32),world_to_camera=np.eye(4,dtype=np.float32))
    rgb=rendered['image'][0,0].float().permute(1,2,0).clamp(0,1).cpu().numpy()
    cv2.imwrite(str(a.output/'native-pinhole.png'),np.uint8(rgb[...,::-1]*255))
    np.savez_compressed(a.output/'depth-alignment.npz',metric_depth=prediction*scale,mesh_depth=gt,alignment_mask=alignment)
    record=dict(scene=a.scene.name,view=a.view,input_frame_ids=chosen,input_size=[size,size],query_times=queries.tolist(),
        strict_loading=str(loading),model_load_seconds=load_seconds,forward_seconds=seconds,peak_allocated_mib=peak,
        scale_to_meters=scale,alignment_pixels=int(alignment.sum()),static_background_relative_error_p50=float(np.median(relative[alignment])),
        static_background_relative_error_p90=float(np.quantile(relative[alignment],.9)),
        input_rgb='cv2 unchanged uint8/uint16 -> float32 display RGB -> bicubic518; no PIL 16-bit conversion',
        time_convention='dataset times normalized to [0,1]; query0/25/49 normalized; offsets added to source-camera-unprojected canonical positions',
        moving_geometry_accuracy_validated=False,training=False)
    (a.output/'inference.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)


if __name__=='__main__':main()
