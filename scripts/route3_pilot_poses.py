"""Nonblocking E2-prime start: frozen VGGT vs DA3 pose-only comparison."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT.parent/'vggt-stage0')]
from route3_pilot_fields import OUT,write,RECORDS,PREFIX
from route3_upper_bound_da3 import sim3
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
FRAMES=[1,6,11,16,21,26,30,35,40,45,50,55,60]


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise RuntimeError('ExplicitGPU0')
    subprocess.run(['nvidia-smi'],check=True);torch.set_num_threads(8);model=VGGT(enable_point=False,enable_depth=False,enable_track=False).eval();state=torch.load(ROOT/'models/VGGT-stage0/model.pt',map_location='cpu',weights_only=True);missing,unexpected=model.load_state_dict(state,strict=False);assert not missing and all(x.startswith(('point_head.','depth_head.','track_head.')) for x in unexpected);del state;model.cuda();rows=[];dest=OUT/'E2-prime';dest.mkdir(exist_ok=True)
    for label in ['B2-high','pilot-multilayer']:
        if label=='B2-high':
            rgbdir=ROOT/'outputs/route3-upper-bound-2026-11-11-01/U2-da3/input-images';gt=np.load(ROOT/'outputs/route3-a2b-2026-11-10-01/blender-b2/B2-high/scene-geometry.npz')['camera_c2w_opencv'][np.array(FRAMES)-1];da3=json.loads((RECORDS/'route3-upper-bound-2026-11-11-da3.json').read_text())
        else:rgbdir=OUT/label/'inputs';gt=np.load(OUT/label/'scene-geometry.npz')['camera_c2w_opencv'][np.array(FRAMES)-1];da3=json.loads((OUT/label/'models/DA3.json').read_text())
        images=load_and_preprocess_images([str(rgbdir/f'f{f:03d}.png') for f in FRAMES]).cuda();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.monotonic()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):pred=model(images)
        extr,intr=pose_encoding_to_extri_intri(pred['pose_enc'],images.shape[-2:]);torch.cuda.synchronize();seconds=time.monotonic()-t;p=np.broadcast_to(np.eye(4),(13,4,4)).copy();p[:,:3]=extr[0].cpu().numpy();p=np.linalg.inv(p);scale,R,translation,sv=sim3(p[:,:3,3],gt[:,:3,3]);aligned=p.copy();aligned[:,:3,:3]=R@p[:,:3,:3];aligned[:,:3,3]=scale*p[:,:3,3]@R.T+translation;delta=np.swapaxes(gt[:,:3,:3],-1,-2)@aligned[:,:3,:3];angle=np.rad2deg(np.arccos(np.clip((np.trace(delta,axis1=1,axis2=2)-1)/2,-1,1)));err=np.linalg.norm(aligned[:,:3,3]-gt[:,:3,3],axis=-1)
        np.savez_compressed(dest/(label+'-VGGT.npz'),c2w_aligned=aligned,c2w_native=p,intrinsics=intr[0].cpu().numpy());rows.append(dict(scene=label,model='VGGT-1B',seconds=seconds,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,translation_RMSE_m=float(np.sqrt(np.mean(err**2))),rotation_mean_deg=float(angle.mean()),rotation_max_deg=float(angle.max()),per_frame_rotation_deg=angle.tolist(),global_sim3_scale=scale,camera_center_singular_values=sv.tolist(),DA3=da3));print('E2_PRIME_POSE',label,rows[-1]['translation_RMSE_m'],rows[-1]['rotation_mean_deg'],flush=True)
    write(RECORDS/(PREFIX+'-E2-prime.json'),dict(status='pose_comparison_completed_calibration_and_hidden_retention_not_yet_run',rows=rows,VGGT_license='local code:Meta VGGT Licensev1July29,2025+AUP;fixedHF860abecweightcard:CC-BY-NC-4.0;research permitted;not Apache',license_original=(ROOT.parent/'vggt-stage0/LICENSE.txt').read_text(),code_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT.parent/'vggt-stage0',text=True).strip(),weights_reference='models/VGGT-stage0/download.json;existing5GBweightsreused',DA3_license='Apache-2.0',no_new_weight_download=True,nonblocking_for_pilot=True,training=False))


if __name__=='__main__':main()
