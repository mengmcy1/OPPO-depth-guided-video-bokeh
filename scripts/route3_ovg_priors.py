"""Prepare compact official DepthSplat priors from RGB and given poses only."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np
from route3_ovg_tartan import CONFIG, ROOT, json_write

OUT=ROOT/CONFIG['output']


def build(tier):
    root=OUT/tier
    raw=root/'depthsplat-raw'
    windows=sorted((root/'windows').glob('*/manifest.json'))
    if not windows:
        raise FileNotFoundError(f'No {tier} windows')
    plan=[]
    compact=[]
    size=CONFIG['size']
    for filename in windows:
        window=json.loads(filename.read_text())
        rows=window['rows']
        dest=filename.parent
        work=dest/'input-resized';work.mkdir(exist_ok=True)
        paths, Ks = [], []
        for i,row in enumerate(rows):
            png=work/f'f{i:03d}.png'
            K=np.array(row['K'],np.float32)
            image=cv2.imread(str(ROOT/row['image']))
            assert image is not None
            K[:2]*=np.array([size/image.shape[1],size/image.shape[0]])[:,None]
            if not png.exists():
                cv2.imwrite(str(png),cv2.resize(image,(size,size),interpolation=cv2.INTER_AREA))
            paths.append(png)
            Ks.append(K)
        poses=np.array([row['c2w'] for row in rows])
        for i,row in enumerate(rows):
            chosen=[i]+sorted([j for j in range(len(rows)) if j!=i],
                              key=lambda j:(np.linalg.norm(poses[i,:3,3]-poses[j,:3,3]),j))[:7]
            z=raw/window['scene']/f'f{i:03d}.npz'
            plan.append(dict(scene=window['scene'],target_frame=row['frame'],target_index=0,frames=[rows[j]['frame'] for j in chosen],
                             images=[str(paths[j]) for j in chosen],K=[Ks[j].tolist() for j in chosen],
                             c2w=poses[chosen].tolist(),bounds_m=[.5,200.],output=str(z)))
            compact.append((z,dest/'prior'/f'f{i:03d}.npz'))
    json_write(root/'depthsplat-plan.json',plan)
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
    if shutil.disk_usage(OUT).free/2**30<60:
        raise RuntimeError('Data disk below 60 GiB')
    subprocess.run(['nvidia-smi'],check=True)
    start=time.monotonic()
    subprocess.run([sys.executable,'scripts/route3_depthsplat_depth.py','--plan',str(root/'depthsplat-plan.json')],
                   cwd=ROOT,check=True)
    for src,dst in compact:
        if dst.exists():
            continue
        dst.parent.mkdir(exist_ok=True)
        with np.load(src) as data:
            np.savez_compressed(dst,depth=data['depth'].astype(np.float16),consistent=data['consistent'],
                                support=data['support'])
    json_write(root/'priors.json',dict(tier=tier,windows=len(windows),frames=len(plan),dtype='float16',
                      elapsed_s=time.monotonic()-start,compact_bytes=sum(dst.stat().st_size for _,dst in compact),
                      prior='official DepthSplat small inference from RGB in given pose coordinates',
                      other_priors_confidence_zero=True,GT_depth_input=False,
                      checkpoint='depthsplat-depth-small-352x640-randview2-8-e807bd82.pth',
                      provenance='03_实验记录/records/route3-g3-r13-2026-11-18-model-licenses.json'))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('tier',choices=['C','G'])
    build(parser.parse_args().tier)
