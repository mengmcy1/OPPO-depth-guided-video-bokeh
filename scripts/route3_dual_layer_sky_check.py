"""Check the full-image visible contract before the observed-sky bugfix rerun."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
import cv2
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.layered_aperture import pixel_layer
from src.analytic_aperture import sampled
from src.aperture_renderer import linear_to_srgb
from src.c1_bokehme import region_sse,physical_coc_radius


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--shared',type=Path,required=True);p.add_argument('--fill-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--record',type=Path,required=True);a=p.parse_args()
    if a.record.exists():raise FileExistsError(a.record)
    a.output.mkdir(parents=True,exist_ok=False);assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0';subprocess.run(['nvidia-smi'],check=True);torch.set_num_threads(4);rows=[]
    for x in json.loads(a.shared.read_text())['items']:
        d=np.load(x['input_npz']);K=torch.tensor(d['K'],device='cuda',dtype=torch.float32);cam=torch.tensor(d['camera_c2w'],device='cuda',dtype=torch.float32);E=torch.linalg.inv(cam)
        C=torch.tensor(d['visible_rgb_linear'],device='cuda');Z=torch.tensor(d['center_depth'],device='cuda');known=d['center_surface_valid']
        fill=torch.tensor(np.load(a.fill_root/x['id']/'fill.npz')['fill_rgb_linear'],device='cuda');bg=torch.tensor(d['background_depth'],device='cuda')
        focus=1.4 if x['scene']=='B2-high' else 5.3
        for name,valid,far in [('omitted_observed_sky',torch.tensor(known,device='cuda'),1e4),('full_observed_visible_layer',None,1e5)]:
            cloud=pixel_layer(C,Z,K,cam,valid)
            with torch.inference_mode():out=sampled(cloud,K,E,512,512,focus,0.,1,fill,bg,far_plane=far);pred=linear_to_srgb(out['rgb_linear']).clamp(0,1).cpu().numpy()
            sky_interior=cv2.erode((~known).astype(np.uint8),np.ones((7,7),np.uint8))>0
            row=dict(scene=x['scene'],frame=x['frame'],mode=name,full=region_sse(pred,d['rgb_srgb'],np.ones(known.shape,bool)),observed_sky_interior=region_sse(pred,d['rgb_srgb'],sky_interior),
                depth_m=10000.,depth_is_GT=False,far_clip_m=far,source='current RGB only; method-only far field,never recovered GT geometry')
            np.savez_compressed(a.output/(x['id']+'-'+name+'.npz'),prediction_srgb=pred)
            rows.append(row);del cloud;torch.cuda.empty_cache();print('SKY_CONTRACT',x['id'],name,row['observed_sky_interior']['psnr'],flush=True)
    passed=all(r['observed_sky_interior']['psnr'] is None or r['observed_sky_interior']['psnr']>=50 for r in rows if r['mode']=='full_observed_visible_layer')
    record=dict(passed=passed,rows=rows,no_DOF_GT_metrics_used=True,unchanged=['K64','bandmask','LaMa lookup/mask','background proxy','frozen criteria','Gaussian kernel/AA'],
        reason='source RGB exists on no-depth sky; previous implementation wrongly treated missing depth as missing visible color. Add current observed RGB at existing10km method-only depth and allow farclip>10km.',
        first_L2_outputs_preserved=True,parameter_fitting_or_search=False)
    a.record.write_text(json.dumps(record,indent=2)+'\n')
    if not passed:raise ValueError('Observed-sky visible contract still fails')


if __name__=='__main__':main()
