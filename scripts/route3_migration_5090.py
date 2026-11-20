"""Verify only the designated, already-cached inputs after migrating to RTX 5090."""
import argparse
from importlib import metadata
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from route3_ovg_1b_common import ROOT, REC, read
from route3_ovg_stage2_evaluate import DIAGNOSTICS, guard, save_json
from route3_ovg_opaque_b0 import image_agreement
from route3_upper_bound_evaluate import display, exr

STAMP='route3-migration-5090-2026-11-20'
DEST=ROOT/'outputs'/f'{STAMP}-01'
LOG=ROOT/'logs'/STAMP
SCENE='metric-scene-122104-broad'
BLENDER=ROOT.parent/'tools/blender-4.5.13-linux-x64/blender'
DEPTH=ROOT/'outputs/route3-g3-r14-2026-11-17-01/geometry/depthsplat-official'/SCENE
PHASE2=ROOT/'outputs/route3-ovg-1c-2026-11-19-01/phase2-B'


def compare_depth(a,b,confidence):
    with np.load(a) as old, np.load(b) as new:
        x,z=old['depth'].astype(np.float64),new['depth'].astype(np.float64)
        v,w=old[confidence].astype(bool),new[confidence].astype(bool)
        valid=np.isfinite(x)&np.isfinite(z)&(x>0)&(z>0)
        joint=valid&v&w
        if not joint.any():
            raise ValueError('No common positive confident depth')
        rel=np.abs(x[joint]/z[joint]-1)
        global_rel=np.abs(x[valid]/z[valid]-1)
        p95=float(np.quantile(rel,.95)); frac=float((v==w).mean())
        return dict(common_confident_pixels=int(joint.sum()),confidence_agreement=frac,
            confidently_valid_old=int(v.sum()),confidently_valid_new=int(w.sum()),
            inverse_depth_absrel_mean=float(rel.mean()),inverse_depth_absrel_p95=p95,
            inverse_depth_absrel_max=float(rel.max()),inverse_depth_absrel_max_all_valid=float(global_rel.max()),
            inverse_depth_relative_gt_5pct_pixels=int((rel>.05).sum()),
            depth_max_abs_m=float(np.abs(x[joint]-z[joint]).max()),
            inverse_depth_rel_PSNR_db=float(-10*np.log10(max(np.square(rel).mean(),1e-16))),
            passed=bool(p95<.005 and frac>=.99))


def sharp():
    import route3_sharp_s1 as original
    dest=DEST/'sharp'
    original.inference(None,frames=(1,),output=dest,publish=False)
    r=compare_depth(original.OUT/'depth/f001.npz',dest/'depth/f001.npz','valid')
    with np.load(original.OUT/'depth/f001.npz') as a, np.load(dest/'depth/f001.npz') as b:
        r['alpha']=image_agreement(a['alpha'],b['alpha'])
    save_json(DEST/'sharp.json',dict(input='B2-high frame 1 (original S1 EXR, no remapping)',**r))


def depthsplat():
    plan=next(r for r in read(DEPTH/'depthsplat-plan.json') if r['target_frame']==1)
    baseline=Path(plan['output'])
    dest=DEST/'depthsplat/f001.npz'
    path=DEST/'depthsplat-plan.json'
    save_json(path,[dict(plan,output=str(dest))])
    subprocess.run([sys.executable,'-u','scripts/route3_depthsplat_depth.py','--plan',str(path)],check=True,cwd=ROOT)
    r=compare_depth(baseline,dest,'consistent')
    save_json(DEST/'depthsplat.json',dict(input=f'{SCENE} frame 1',input_context=plan['frames'],**r))


def cycles():
    example=next(r for r in read(PHASE2/'render-jobs.json') if r['scene']==SCENE and r['frame']==1)
    ref=Path(example['source_image'])
    root=ref.parent.parent
    previous=next(r for f in ('inputs-jobs-results.json','inputs-r13-jobs-results.json')
                  if (root/f).is_file() for r in read(root/f)
                  if Path(r['output']).resolve()==ref)
    local=DEST/'cycles'
    local.mkdir(exist_ok=True)
    plan=local/'jobs.json'
    out=local/'f001.exr'
    job={k:v for k,v in previous.items() if k not in ('passes','seconds')}
    save_json(plan,[dict(job,output=str(out))])
    if not out.is_file():
        subprocess.run([str(BLENDER),'-b','-t','12','--python-exit-code','2','--python',
                        'scripts/route3_metric_blender.py','--','--jobs',str(plan)],check=True,cwd=ROOT)
    old,new=exr(ref),exr(out)
    quality=image_agreement(display(old),display(new))
    save_json(DEST/'cycles.json',dict(input=f'{SCENE} f001 pinhole',spp=previous['spp'],
        seed=previous['seed'],linear_max_abs=float(np.abs(old-new).max()),
        display=quality,passed=quality['psnr_db']>=40.))


def verify():
    checks={name:read(DEST/f'{name}.json') for name in ('sharp','depthsplat','optical','cycles')}
    checks['sharp']['localization']=read(DEST/'sharp-localize.json')
    toy_new=read(DEST/'toy.json')
    toy_old=read(DIAGNOSTICS/'toy.json')
    cols=('query_weight','query_depth_m','dominant_visible_depth_m','dominant_visible_weight',
          'T_at_query_depth','SSE_increase')
    a=np.array([r[k] for r in toy_old['cases'] for k in cols if r[k] is not None])
    b=np.array([r[k] for r in toy_new['cases'] for k in cols if r[k] is not None])
    checks['leakage_toy']=dict(
        **image_agreement(a,b),
        passed=toy_new['passed'] and [r['cause'] for r in toy_old['cases']]==[r['cause'] for r in toy_new['cases']]
        and bool(np.allclose(a,b,atol=2e-5,rtol=2e-4)),cases=toy_new['cases'])
    environment={p:metadata.version(p) for p in ('torch','triton','gsplat','xformers')}
    assert environment['torch']=='2.7.1+cu128'
    environment.update(gpu='NVIDIA GeForce RTX 5090',capability='sm_120',driver='580.105.08',
        cuda_toolkit='12.8.93',cuda_build=torch.version.cuda,vcpu=25,cgroup_RAM_GiB=90,
        blender=subprocess.check_output([str(BLENDER),'--version'],text=True).splitlines()[0])
    installation=read(DEST/'installation.json')
    result=dict(environment=environment,installation=installation,
        checks=checks,all_passed=all(r['passed'] for r in checks.values()),
        comparison_rules=dict(optical_image_min_PSNR=48.,visible_zero_max_score_delta_dB=.1,
                             cycles_display_min_PSNR=40.,inverse_depth_p95_max=.005,
                             confidence_agreement_min=.99),
        output=str(DEST.relative_to(ROOT)),
        focus_and_query='Use original P2 focus and exact archived Query conditioning/placement depth; render the visible layer with the new prediction and compare to old images and scores.',
        scope='single SHARP and DepthSplat target, one Cycles frame, 1c one P2 case and 2b one development scene, and saved toy')
    save_json(REC/f'{STAMP}.json',result)
    if not result['all_passed']:
        raise SystemExit('Migration comparison failed; do not start the one-off evaluation.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('task',choices=('sharp','depthsplat','cycles','verify'))
    task=p.parse_args().task
    if task!='verify':
        guard(gpu=True)
    globals()[task]()
