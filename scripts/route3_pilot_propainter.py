"""Run unmodified official ProPainter on the13 input frames; no GT masks."""
import argparse
import gc
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time
import cv2
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from route3_pilot_fields import OUT,write
FRAMES=[1,6,11,16,21,26,30,35,40,45,50,55,60]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scene',required=True);a=p.parse_args();folder=OUT/'C2'/a.scene;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'result.json').exists():return
    images=folder/'input';masks=folder/'mask';images.mkdir(exist_ok=True);masks.mkdir(exist_ok=True);rows=[]
    for index,f in enumerate(FRAMES):
        if a.scene=='B2-high':
            d=np.load(ROOT/'outputs/route3-sharp-s1-2026-11-11-01/depth'/f'f{f:03d}.npz');depth=np.load(ROOT/'outputs/route3-sharp-s1c-2026-11-10-01/calibration/E1/SHARP'/f'f{f:03d}.npz')['depth']
        else:
            d=np.load(OUT/a.scene/'models/SHARP'/f'f{f:03d}.npz');depth=np.load(OUT/a.scene/'models/calibration/SHARP'/f'f{f:03d}.npz')['depth']
        mask=np.isfinite(depth)&(depth>0)&(depth<3.);cv2.imwrite(str(images/f'{index:04d}.png'),np.uint8(np.rint(d['rgb_srgb'][...,::-1]*255)));cv2.imwrite(str(masks/f'{index:04d}.png'),mask.astype(np.uint8)*255)
        rows.append(dict(index=index,source_frame=f,foreground_mask_fraction=float(mask.mean())))
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise RuntimeError('Explicit GPU0')
    subprocess.run(['nvidia-smi'],check=True);torch.set_num_threads(8);cv2.setNumThreads(1);torch.manual_seed(20261112);torch.cuda.reset_peak_memory_stats();t=time.monotonic()
    repo=ROOT.parent/'ProPainter-pilot-proxy-20261112';sys.path.insert(0,str(repo));os.chdir(repo)
    args=['inference_propainter.py','--video',str(images),'--mask',str(masks),'--output',str(folder/'results'),'--height','512','--width','512','--mask_dilation','4','--ref_stride','3','--neighbor_length','6','--subvideo_length','16','--raft_iter','20','--save_frames']
    sys.argv=args;runpy.run_path(str(repo/'inference_propainter.py'),run_name='__main__');torch.cuda.synchronize();seconds=time.monotonic()-t
    outputs=sorted((folder/'results/input/frames').glob('*.png'));assert len(outputs)==13
    write(folder/'result.json',dict(completed=True,scene=a.scene,seconds=seconds,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,parameters=args[1:],rows=rows,background_images=[str(x) for x in outputs],mask_definition='calibration-pipeline SHARP depth0<z<3m;official dilation4px;no GT segmentation/depth',GT_used=False,network_training=False,source_script_unmodified=True,license='NTU S-Lab noncommercialresearch',code_archive=json.loads((repo/'SOURCE_ARCHIVE.json').read_text())))


if __name__=='__main__':main()
