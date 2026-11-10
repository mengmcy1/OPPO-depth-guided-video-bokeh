"""Frozen U0-U2 region scoring; GT and baseline metrics stay in evaluator only."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
import cv2
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from src.dual_layer import depth_band
from src.c1_bokehme import region_sse,patch_metrics
from route3_c1_bokehme_oracle import lpips_cpu
BASE=ROOT/'outputs/route3-b2only-2026-11-10-01'
OUT=ROOT/'outputs/route3-upper-bound-2026-11-11-01'
RECORDS=ROOT/'03_实验记录/records'


def display(x):
    return np.where(x<=.0031308,12.92*x,1.055*np.maximum(x,0)**(1/2.4)-.055).clip(0,1).astype(np.float32)


def exr(path):
    value=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
    if value is None:raise FileNotFoundError(path)
    return value[...,:3][...,::-1].copy()


def items():
    return [x for x in json.loads((BASE/'manifests/oracle.json').read_text())['items']
            if x['scene']=='B2-high' and x['fstop'] in [1.4,2.]]


def masks_for(x,c):
    d=np.load(x['input_npz']);g=np.load(x['geometry_gt_npz'])
    band,edge,R=depth_band(d['center_depth'],d['K'][0,0],x['focus_distance_m'],x['aperture_radius_m'],c['band']['multiplier'],c['band']['relative_inverse_depth_jump'])
    residual=g['residual_pixels'].astype(bool)
    return dict(full_image=np.ones(band.shape,bool),in_focus=d['center_surface_valid']&(R<=1),
        out_of_focus=d['center_surface_valid']&(R>=2),band=band&~residual,
        recoverable=band&g['mask_recoverable_total'],motion_required=band&g['mask_motion_required'],
        parallax_sufficient=band&g['mask_parallax_sufficient'])


def score(pred,target,masks,model):
    ps=patch_metrics(pred,target,np.ones(pred.shape[:2],bool),model,patch_size=64,device='cpu')
    scores=np.array(ps['patch_lpips_scores']);result={}
    for region,mask in masks.items():
        weights=np.array([mask[y:y+64,x:x+64].sum() for y in range(0,512,64) for x in range(0,512,64)])
        result[region]=dict(**region_sse(pred,target,mask),patch_lpips=float(scores@weights/weights.sum()) if weights.sum() else None)
    return result


def pool(rows):
    n=sum(r['pixels'] for r in rows);s=sum(r['sse'] for r in rows)
    return dict(pixels=n,sse=s,psnr=float(-10*np.log10(max(s/(3*n),1e-12))),
        patch_lpips=sum(r['pixels']*r['patch_lpips'] for r in rows if r['pixels'])/n) if n else dict(pixels=0,sse=0.,psnr=None,patch_lpips=None)


def baseline_metrics():
    two=json.loads((RECORDS/'route3-dual-layer-2026-11-11-L2-observed-sky-evaluation.json').read_text())['per_case_metrics']
    three=json.loads((RECORDS/'route3-dual-layer-2026-11-11-L3-evaluation.json').read_text())['per_case_metrics']
    return {x['id']:{'O3_GT':two[x['id']]['O3'],'L2':two[x['id']]['L2'],'L3_GT':three[x['id']]['L3']} for x in items()}


def full_noise(focus,stop):
    d=json.loads((RECORDS/'route3-c1-b2-2026-11-10-noise.json').read_text())
    spp=next(r['samples'] for r in d['formal_samples'] if r['scene']=='B2-high' and r['fstop']==stop and r['focus_m']==focus)
    return max(r['noise_patch_lpips']['mean'] for r in d['rows'] if r['scene']=='B2-high' and r['fstop']==stop and r['focus_m']==focus and r['samples']==spp)


def average(a):
    plan=json.loads((OUT/'view-plan.json').read_text());timings=json.loads((OUT/'U0-subviews/timings.json').read_text());bykey={x['key']:x for x in timings}
    dest=OUT/'U0-averages';dest.mkdir(exist_ok=False);rows=[]
    for case,jobs in plan['evaluation_groups'].items():
        if not all(j['key'] in bykey for j in jobs):continue
        raw=sum(exr(bykey[j['key']]['path']).astype(np.float64) for j in jobs)/64
        path=dest/(case+'.npz');np.savez_compressed(path,rgb_linear=raw.astype(np.float32),prediction_srgb=display(raw))
        cv2.imwrite(str(path.with_suffix('.png')),np.uint8(display(raw)[...,::-1]*255))
        rows.append(dict(id=case,method='U0',path=str(path),subviews=64))
    noise=[];npairs=OUT/'U0-noise/timings.json'
    if npairs.exists():
        nindex={r['key']:r for r in json.loads(npairs.read_text())}
        for case,jobs in plan['evaluation_groups'].items():
            if not all(j['key']+'-noise' in nindex for j in jobs):continue
            raw=sum(exr(nindex[j['key']+'-noise']['path']).astype(np.float64) for j in jobs)/64
            path=dest/(case+'-noise.npz');np.savez_compressed(path,rgb_linear=raw.astype(np.float32),prediction_srgb=display(raw));noise.append(dict(id=case,path=str(path)))
    manifest=dict(rows=rows,noise=noise,render_images=len(timings),sum_render_write_seconds=sum(r['seconds'] for r in timings),
        linear_unclipped_average=True,geometry_checks=str(OUT/'U0-subviews/geometry-checks.json'))
    (OUT/'U0-predictions.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest),flush=True)


def evaluate(a):
    c=json.loads((RECORDS/'route3-upper-bound-criteria-2026-11-11.json').read_text());assert c['frozen']
    model=lpips_cpu(SimpleNamespace(alexnet_weights=ROOT.parent/'.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth'))
    manifest=json.loads(a.manifest.read_text());lookup={x['id']:x for x in items()};metrics=baseline_metrics();flat=[];plots=[]
    previous={r['id']:r for r in json.loads((RECORDS/'route3-dual-layer-2026-11-11-L2-observed-sky-render.json').read_text())['rows']}
    dest=OUT/(a.stage+'-evaluation');dest.mkdir(exist_ok=False)
    for r in manifest['rows']:
        x=lookup[r['id']];pred=np.load(r['path'])['prediction_srgb'];target=display(exr(x['gt_rgb_exr']));masks=masks_for(x,c)
        if not np.isfinite(pred).all():raise ValueError('Nonfinite prediction')
        stats=score(pred,target,masks,model)
        for region,value in stats.items():
            if value['pixels']!=metrics[x['id']]['O3_GT'][region]['pixels']:
                raise ValueError('Scoring mask differs from frozen L2: '+x['id']+' '+region)
        metrics[x['id']][r['method']]=stats
        for region,v in stats.items():flat.append(dict(id=x['id'],method=r['method'],region=region,**v))
        import matplotlib
        matplotlib.use('Agg');import matplotlib.pyplot as plt
        O3=np.load((OUT/'U2-O3'/(x['id']+'.npz')) if a.stage=='U2' else (BASE/'methods-2d/bokehme'/x['id']/'prediction.npz'))['prediction_srgb']
        imgs=[target,O3,pred,np.minimum(abs(pred-target)*15,1)]
        titles=['Cycles','O3 DA3-depth' if a.stage=='U2' else 'O3 GT-depth',r['method'],'absolute error x15']
        if a.stage=='U1':
            imgs=[target,O3,np.load(previous[x['id']]['outputs']['L2'])['prediction_srgb'],pred]
            titles=['Cycles','O3 GT-depth','unoptimized L2',r['method']]
        ys,xs=np.nonzero(masks['band']);index=np.argmin((ys-np.median(ys))**2+(xs-np.median(xs))**2)
        cy=int(np.clip(ys[index]-64,0,384));cx=int(np.clip(xs[index]-64,0,384))
        fig,ax=plt.subplots(2,4,figsize=(16,8))
        for column,(img,title) in enumerate(zip(imgs,titles)):
            ax[0,column].imshow(img);ax[1,column].imshow(img[cy:cy+128,cx:cx+128]);ax[0,column].set_title(title)
            ax[0,column].axis('off');ax[1,column].axis('off')
        fig.suptitle(x['id']);fig.tight_layout();path=dest/(x['id']+'-'+r['method']+'.png');fig.savefig(path,dpi=110);plt.close(fig);plots.append(str(path))
        print('UPPER_SCORE',r['method'],r['id'],stats['full_image']['psnr'],stats['band']['psnr'],flush=True)
    noise=[]
    for r in manifest.get('noise',[]):
        reference=next(t for t in manifest['rows'] if t['id']==r['id'] and t['method']=='U0')
        x=lookup[r['id']];p=np.load(r['path'])['prediction_srgb'];q=np.load(reference['path'])['prediction_srgb']
        stats=score(p,q,masks_for(x,c),model)
        threshold=full_noise(x['focus_distance_m'],x['fstop'])
        noise.append(dict(id=r['id'],statistics=stats,existing_Cycles_noise_patch_LPIPS=threshold,
            below_existing_noise=stats['full_image']['patch_lpips']<threshold))
    pooled={};comparisons={};per_aperture={}
    for focus in [1.4,5.]:
        selected=[x for x in items() if x['focus_distance_m']==focus];ids=[x['id'] for x in selected]
        methods=set.intersection(*(set(metrics[i]) for i in ids));table={m:{reg:pool([metrics[i][m][reg] for i in ids]) for reg in c['regions']} for m in sorted(methods)}
        pooled[str(focus)]=table;comparisons[str(focus)]={}
        per_aperture[str(focus)]={}
        for stop in [1.4,2.]:
            group=[x['id'] for x in selected if x['fstop']==stop]
            per_aperture[str(focus)][str(stop)]={m:{reg:pool([metrics[i][m][reg] for i in group])
                for reg in c['regions']} for m in sorted(methods)}
        for m in methods-{'O3_GT','L2','L3_GT'}:
            comparisons[str(focus)][m]={b:{reg:dict(
                PSNR_gain_dB=table[m][reg]['psnr']-table[b][reg]['psnr'] if table[m][reg]['pixels'] else None,
                LPIPS_change=table[m][reg]['patch_lpips']-table[b][reg]['patch_lpips'] if table[m][reg]['pixels'] else None)
                for reg in c['regions']} for b in ['O3_GT','L2','L3_GT']}
    report=dict(completed=len({r['id'] for r in manifest['rows']})==12,stage=a.stage,criteria='11c78ca',
        manifest=str(a.manifest),per_case_metrics=metrics,pooled=pooled,per_aperture=per_aperture,
        comparisons=comparisons,noise=noise,visualizations=plots)
    if a.record.exists():raise FileExistsError(a.record)
    a.record.write_text(json.dumps(report,indent=2)+'\n')
    with a.record.with_suffix('.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]),lineterminator='\n');writer.writeheader();writer.writerows(flat)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['average','evaluate'],required=True)
    p.add_argument('--manifest',type=Path);p.add_argument('--stage',default='U0');p.add_argument('--record',type=Path);a=p.parse_args()
    torch.set_num_threads(8);cv2.setNumThreads(1)
    if a.mode=='average':average(a)
    else:evaluate(a)


if __name__=='__main__':main()
