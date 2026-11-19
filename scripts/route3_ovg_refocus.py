"""Re-evaluate frozen 1b/1b′ models with their own P2 and P1 focus distances."""
import argparse
import os
import shutil
import subprocess
import time

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
# Affine fitting and the 3-column CPU backprojections are small BLAS jobs.
os.environ['OPENBLAS_NUM_THREADS']='1'

import cv2
import numpy as np
import torch

from route3_depth_interface import focus_row, subject_mask
from route3_ovg_1b_common import CONFIG, OUT as DATA_RUN, ROOT, SOURCE, read
from route3_ovg_1b_optical_data import geometry_root
from route3_ovg_1b_prime import OUT as PRIME_RUN
from route3_ovg_1b_prime_evaluate import A, DEPTHSPLAT, OLD, OLD_FIXED, ORIGINAL, PRIME_EMA, PRIME_RAW
from route3_ovg_1b_train import load_model
import route3_ovg_data as data
from route3_ovg_subpixel import save_json, save_npz
from route3_step2_diagnostic import fit_background

OUT = ROOT/'outputs/route3-ovg-refocus-2026-11-18-01'
PREFIX = 'route3-ovg-refocus-2026-11-18'
FIRST = 'OVG-1a'
METHODS = (ORIGINAL, DEPTHSPLAT, FIRST, OLD, OLD_FIXED, PRIME_EMA, PRIME_RAW,
           PRIME_EMA+'-gate-off', PRIME_RAW+'-gate-off', A)
LEGACY_HEADS = ('normal', 'sigma_px', 'opacity')


def guard():
    OUT.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(OUT).free<60*2**30:
        raise RuntimeError('Data disk available space is below 60 GiB')


def gpu_setup():
    subprocess.run(['nvidia-smi'],check=True)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':
        raise RuntimeError('Use the assigned GPU 0')
    torch.set_num_threads(5)
    cv2.setNumThreads(2)
    data.OUT=DATA_RUN


def case_frames():
    for name in read(data.PREVIOUS/'split.json')['heldout']:
        source=data.PREVIOUS/'geometry/original'/name
        manifest=read(data.PREVIOUS/'evaluation-input/original'/name/'512-13/manifest.json')
        target=data.PREVIOUS/'geometry/depthsplat-official'/name
        matching={e['frame']:e for e in read(target/'prepared.json')['rows']}
        for entry in read(source/'prepared.json')['rows']:
            frame=entry['frame']
            cases=[r for r in manifest['rows'] if r['frame']==frame]
            folder=OUT/name/f'f{frame:03d}'
            yield name,frame,entry,matching[frame],source,target,manifest,cases,folder


def valid_depth(z):
    return np.isfinite(z)&(z>0)


def per_frame_depths(frame,entry,other,source,target,name,model_specs):
    original=data.load(entry['input_npz'],('depth',))['depth']
    depths={ORIGINAL:original,DEPTHSPLAT:data.load(other['input_npz'],('depth',))['depth']}
    valid={}
    f=f'f{frame:03d}.npz'
    for method,path in ((ORIGINAL,source),(DEPTHSPLAT,target)):
        keep=data.load(path/'models/calibration/SHARP'/f,('dominant_valid',))['dominant_valid']
        valid[method]=valid_depth(depths[method]) & keep
    surface=data.load(source/'A-depth'/f,('depth','valid','cluster_depth','cluster_fraction','contribution_alpha'))
    depths[A]=surface['depth']
    valid[A]=valid_depth(depths[A]) & surface['valid']
    x,_=data.sample(('B',name,data.TRAIN_FRAMES.index(frame)),'cuda')
    heads={}
    for method,net,gate in model_specs:
        use_heads=method in (FIRST,OLD)
        pred=net(x,protect_residual=gate,render_heads=use_heads)
        z=pred['depth1'][0].cpu().numpy()
        depths[method]=z
        valid[method]=valid_depth(z)
        for k in LEGACY_HEADS if use_heads else ():
            heads[method+'__'+k]=pred[k].cpu().numpy()
    return depths,valid,surface,heads


def focus_records(name, frame, cases, depths, valid, surface):
    gt_dir=geometry_root(name)
    g=data.load(gt_dir/'geometry'/f'f{frame:03d}-center.npz',('depth_axial_m','K','center_visible','object_id'))
    foreground=data.load(gt_dir/'scene-geometry.npz',('foreground_object_ids',))['foreground_object_ids']
    background=g['center_visible'] & ~np.isin(g['object_id'],foreground) & valid_depth(g['depth_axial_m'])
    # Use a single common fitting set before the predetermined train/holdout split.
    background &= np.logical_and.reduce(list(valid.values()))
    fits={k:fit_background(g['depth_axial_m'],z,background) for k,z in depths.items()}
    subjects={f:subject_mask(g,f) for f in (1.4,5.)}
    results=[]
    for item in cases:
        c=item['case']
        nominal,protocol=c['nominal_focus_m'],c['protocol']
        for method in METHODS:
            key=OLD if method==OLD_FIXED else method
            depth=depths[key]
            fit=fits[key]
            eligible=np.where(valid[key], depth, 0.)
            row=focus_row(eligible,subjects[nominal],nominal,c['focus_m'],fit,protocol,
                          surface if method==A else None)
            value=row['focus_m']
            passed=value is not None and np.isfinite(value) and value>0
            results.append(dict(scene=name,frame=frame,id=c['id'],protocol=protocol,aperture=c['fstop'],
                method=method,名义对焦距离_m=nominal,实际对焦距离_m=float(value) if passed else None,
                旧实际对焦距离_m=c['focus_m'],P2主体总像素数=row['subject_pixels'],
                P2有效像素数=row['dominant_pixels'],P2有效率低于20pct=row['dominant_below_20percent'],
                P1共同背景像素数=int(background.sum()),P1拟合训练像素数=fit.get('train'),
                P1拟合留出像素数=fit.get('heldout'),P1拟合斜率=fit.get('a'),P1拟合截距_逆米=fit.get('b'),
                P1拟合留出RMSE_逆米=fit.get('heldout_inverse_RMSE'),
                P1拟合留出相对MAE=fit.get('heldout_inverse_relative_MAE'),
                P1拟合失败=not fit['success'],P1拟合失败原因=fit.get('reason'),
                失败=not passed,失败原因=None if passed else (
                    'no valid subject surface' if protocol=='P2' else fit.get('reason') or 'invalid mapped focus distance')))
    return results,subjects,background


@torch.inference_mode()
def prepare():
    specs=[
        (FIRST,load_model(SOURCE/'checkpoints/step-030000.pt').eval(),False),
        (OLD,load_model(DATA_RUN/'checkpoints/step-030000.pt','ema').eval(),False),
    ]
    file=PRIME_RUN/'checkpoints/step-030000.pt'
    for label,key in ((PRIME_EMA,'ema'),(PRIME_RAW,'model')):
        net=load_model(file,key).eval()
        specs.extend([(label,net,True),(label+'-gate-off',net,False)])
    all_rows=[]
    checked=[]
    for name,frame,entry,other,source,target,_,cases,folder in case_frames():
        guard()
        destination=folder/'R1.json'
        if destination.exists():
            cached=read(destination)
            all_rows.extend(cached['rows'])
            checked.extend(cached['fixed_check'])
            continue
        start=time.monotonic()
        depth,valid,surface,heads=per_frame_depths(frame,entry,other,source,target,name,specs)
        rows,subjects,bg=focus_records(name,frame,cases,depth,valid,surface)
        fixed=[]
        if '122105' in name:
            check=next(r for r in rows if r['method']==PRIME_EMA and r['protocol']=='P2'
                       and r['名义对焦距离_m']==1.4 and r['aperture']==1.4)
            median=float(np.median(depth[PRIME_EMA][subjects[1.4]&valid[PRIME_EMA]]))
            expected={1:1.300,30:1.409,60:1.324}[frame]
            selected=check['实际对焦距离_m']
            passed=selected is not None and abs(selected-median)<=1e-7 and abs(median-expected)<=.003
            fixed.append(dict(scene=name,frame=frame,实际对焦距离_m=check['实际对焦距离_m'],
                              主体深度中位数_m=median,历史核对约值_m=expected,passed=passed))
            if not passed:
                save_json(folder/'R1-fixed-check-failure.json',dict(fixed_check=fixed,row=check))
                raise RuntimeError(f'R1 subject distance check failed for 122105 frame {frame}')
        save_npz(folder/'visible-depths.npz',**depth,**heads)
        save_npz(folder/'R1-masks.npz',subject_near=subjects[1.4],subject_far=subjects[5.],
                 common_visible_background=bg,**{k+'__valid':v for k,v in valid.items()})
        save_json(destination,dict(rows=rows,fixed_check=fixed,seconds=time.monotonic()-start))
        all_rows.extend(rows)
        checked.extend(fixed)
        print('REFOCUS_R1',name,frame,'failures',sum(r['失败'] for r in rows),fixed,flush=True)
    assert len(checked)==3 and all(r['passed'] for r in checked)
    save_json(OUT/'R1.json',dict(protocol_revision='4a6ac37',固定镜头焦距_mm=50,rows=all_rows,
                                fixed_check=checked,methods=list(METHODS)))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('job',choices=['prepare'])
    args=parser.parse_args()
    guard()
    gpu_setup()
    prepare()
