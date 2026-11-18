"""Ancillary 2x BOX comparison on the existing thin scenes, no new assets."""
import json
import os
import shutil
import subprocess

import numpy as np
import cv2

from route3_ovg_subpixel import OUT, ROOT, save_json


def main():
    os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
    dest=OUT/'B/thin-2x'
    old=ROOT/'outputs/route3-g3-r14-2026-11-17-01/truth'
    jobs=[]
    for scene in sorted(old.glob('metric-scene-*-thin')):
        meta=json.loads((scene/'alias/manifest.json').read_text())
        same=meta['rows'][0]
        if same['label']!='center':
            raise ValueError(scene)
        jobs.append(dict(scene=scene.name,blend=str(scene/'scene.blend'),id=scene.name+'-2x',
                         frame=30,size=1024,spp=meta['samples'],seed=same['seed'],
                         omit_reserved_passes=True,dof=False,output=str(dest/(scene.name+'.exr'))))
    if shutil.disk_usage(OUT).free/2**30<60:
        raise RuntimeError('Data disk below 60 GiB')
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
    plan=dest/'jobs.json'
    save_json(plan,jobs)
    subprocess.run(['nvidia-smi'],check=True)
    blender=ROOT.parent/'tools/blender-4.5.3-linux-x64/blender'
    subprocess.run([str(blender),'-b','-t','6','--python','scripts/route3_metric_blender.py',
                    '--','--jobs',str(plan)],cwd=ROOT,check=True)
    rows=[]
    for job in jobs:
        scene=old/job['scene']
        control=cv2.imread(str(scene/'alias/center.exr'),-1)[...,:3]
        img=cv2.imread(job['output'],-1)[...,:3]
        label=cv2.imread(str(scene/'alias/center-IndexOB-0030.exr'),-1)[...,0].round().astype(np.int32)
        graph=json.loads((scene/'scene.json').read_text())
        front={x['id']:x['name'] for x in graph['objects'] if x['name'] in graph['foreground_names']}
        ref=cv2.resize(img,(512,512),interpolation=cv2.INTER_AREA)
        error=(ref-control).astype(np.float64)**2
        stats=[]
        for q,name in front.items():
            core=cv2.erode((label==q).astype(np.uint8),np.ones((3,3),np.uint8))>0
            stats.append(dict(object=name,pure_interior_pixels=int(core.sum()),
                              linear_RGB_RMSE=float(np.sqrt(error[core].mean())) if core.any() else None))
        rows.append(dict(scene=scene.name,size=1024,base_size=512,frame=30,spp=job['spp'],
                         linear_RGB_full_RMSE=float(np.sqrt(error.mean())),front_objects=stats))
    save_json(dest/'summary.json',dict(rows=rows,Blender='4.5.3 LTS',filter='BOX 1.0',
                     diagnostic_only=True,meaning='1024 BOX render area-averaged to 512; independent from half-pixel phase and coverage gate'))


if __name__=='__main__':
    main()
