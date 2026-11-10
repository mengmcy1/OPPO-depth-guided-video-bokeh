"""Apply precommitted criteria; do not tune after reading method results."""
import argparse
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from src.dual_layer import depth_band
from route3_upper_bound_evaluate import OUT,BASE,RECORDS,items,masks_for,full_noise


def geometry():
    rows=[]
    for frame in [1,30,60]:
        estimated=np.load(OUT/'U2-da3/E1'/f'f{frame:03d}.npz');z=estimated['depth'];K=estimated['K']
        gt=np.load(BASE/'B2-high/geometry'/f'f{frame:03d}-center.npz');truth=gt['depth_axial_m'];known=truth>0
        inv=np.where(known,1/np.maximum(truth,1e-6),0)
        edge=(cv2.dilate(inv,np.ones((3,3),np.uint8))-cv2.erode(inv,np.ones((3,3),np.uint8)))>.03*np.maximum(inv,1e-6)
        boundary=cv2.dilate(edge.astype(np.uint8),np.ones((5,5),np.uint8))>0
        errors={}
        for name,mask in [('finite_full_image',known),('finite_depth_boundary',known&boundary)]:
            delta=z[mask]-truth[mask];ratio=np.maximum(z[mask]/truth[mask],truth[mask]/z[mask])
            errors[name]=dict(pixels=int(mask.sum()),abs_relative=float(np.mean(abs(delta)/truth[mask])),
                RMSE_m=float(np.sqrt(np.mean(delta.astype(np.float64)**2))),median_error_m=float(np.median(delta)),
                delta1_fraction=float(np.mean(ratio<1.25)),predicted_to_GT_depth_median_ratio=float(np.median(z[mask]/truth[mask])))
        for focus in [1.4,5.]:
            for stop in [1.4,2.]:
                Z=np.where(known,truth,1e4);real,_,_=depth_band(Z,K[0,0],focus,.05/(2*stop))
                pred,_,_=depth_band(z,K[0,0],focus,.05/(2*stop))
                TP=int((real&pred).sum());FP=int((~real&pred).sum());FN=int((real&~pred).sum())
                rows.append(dict(frame=frame,focus_m=focus,fstop=stop,depth_errors=errors,
                    GT_band_pixels=int(real.sum()),estimated_band_pixels=int(pred.sum()),band_TP=TP,band_FP=FP,band_FN=FN,
                    band_recall=TP/max(TP+FN,1),band_precision=TP/max(TP+FP,1)))
    report=dict(completed=True,rows=rows,geometry_corrections_from_depth_GT=False,
        unknown_GT_sky_excluded_from_depth_error=True,GT_used_here='evaluation only;predictions fixed beforehand')
    (RECORDS/'route3-upper-bound-2026-11-11-U2-geometry-evaluation.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def u1(d,c):
    decisions={}
    for focus in [1.4,5.]:
        key=str(focus);table=d['pooled'][key];records=[x for x in items() if x['focus_distance_m']==focus]
        l,r=table['U1']['band'],table['O3_GT']['band'];gain=l['psnr']-r['psnr'];delta=l['patch_lpips']-r['patch_lpips']
        weights=[d['per_case_metrics'][x['id']]['U1']['band']['pixels'] for x in records]
        noises=[c['noise']['per_lens_band'][f'N{x["fstop"]:g}-F{focus:g}'] for x in records]
        noise=float(np.average(noises,weights=weights));quality=gain>=.5 or (-delta>=.1*r['patch_lpips'] and -delta>3*noise)
        secondary=gain>=-.2 and delta<=3*noise;valid=[];directions=0;ties=0;groups=[]
        for x,n in zip(records,noises):
            a=d['per_case_metrics'][x['id']]['U1']['band'];b=d['per_case_metrics'][x['id']]['O3_GT']['band']
            enough=a['pixels']>=1000;direction=a['psnr']>b['psnr'] and a['patch_lpips']-b['patch_lpips']<-3*n
            if enough:valid.append(x['id']);directions+=int(direction);ties+=int(abs(a['patch_lpips']-b['patch_lpips'])<=3*n)
            groups.append(dict(id=x['id'],pixels=a['pixels'],PSNR_gain_dB=a['psnr']-b['psnr'],LPIPS_change=a['patch_lpips']-b['patch_lpips'],significant_same_direction=direction))
        guards={reg:table['U1'][reg]['psnr']-table['O3_GT'][reg]['psnr'] for reg in ['full_image','in_focus']}
        formal=quality and secondary and len(valid)==6 and directions>=5 and all(v>=-.2 for v in guards.values())
        fnoise=float(np.mean([full_noise(focus,x['fstop']) for x in records]))
        fdelta=table['U1']['full_image']['patch_lpips']-table['O3_GT']['full_image']['patch_lpips']
        worth=formal and guards['full_image']>=.5 and fdelta<=3*fnoise
        decisions[key]=dict(band_PSNR_gain_dB=gain,band_LPIPS_change=delta,band_noise=noise,quality_passed=quality,
            secondary_passed=secondary,valid_groups=len(valid),significant_same_direction=directions,LPIPS_ties=ties,
            guardrails_dB=guards,formal_method_value_passed=formal,full_LPIPS_change=fdelta,full_LPIPS_noise=fnoise,
            worth_training=worth,groups=groups)
    return dict(per_focus=decisions,at_least_one_focus_worth_training=any(x['worth_training'] for x in decisions.values()))


def u2(d,c):
    result={}
    for variant in ['E1','E2']:
        result[variant]={}
        for focus in [1.4,5.]:
            table=d['pooled'][str(focus)];ours='ours_'+variant;o3='O3_'+variant;degradation={};direct={}
            for region in ['band','recoverable','full_image','in_focus']:
                l,g=table[ours][region],table['L3_GT'][region];b,t=table[o3][region],table['O3_GT'][region]
                do=l['psnr']-g['psnr'];db=b['psnr']-t['psnr']
                degradation[region]=dict(Delta_ours_PSNR_dB=do,Delta_O3_PSNR_dB=db,advantage_dB=do-db,
                    Delta_ours_LPIPS=l['patch_lpips']-g['patch_lpips'],Delta_O3_LPIPS=b['patch_lpips']-t['patch_lpips'])
                direct[region]=dict(PSNR_gain_dB=l['psnr']-b['psnr'],LPIPS_change=l['patch_lpips']-b['patch_lpips'])
            groups=[];count=0;valid=0
            for x in [x for x in items() if x['focus_distance_m']==focus]:
                m=d['per_case_metrics'][x['id']];a=m[ours]['band'];b=m[o3]['band']
                advantage=(a['psnr']-m['L3_GT']['band']['psnr'])-(b['psnr']-m['O3_GT']['band']['psnr'])
                enough=a['pixels']>=1000;valid+=int(enough);count+=int(enough and advantage>0)
                groups.append(dict(id=x['id'],pixels=a['pixels'],degradation_advantage_dB=advantage,
                    direct_PSNR_gain_dB=a['psnr']-b['psnr'],direct_LPIPS_change=a['patch_lpips']-b['patch_lpips']))
            guard=all(direct[k]['PSNR_gain_dB']>=-.2 for k in ['full_image','in_focus'])
            result[variant][str(focus)]=dict(degradation=degradation,direct=direct,valid_groups=valid,
                same_direction=count,guardrails_passed=guard,
                passed=degradation['band']['advantage_dB']>=.3 and valid==6 and count>=5 and guard,groups=groups)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['U1','U2','geometry'],required=True);a=p.parse_args()
    cv2.setNumThreads(1)
    if a.stage=='geometry':geometry();return
    c=json.loads((RECORDS/'route3-upper-bound-criteria-2026-11-11.json').read_text())
    d=json.loads((RECORDS/f'route3-upper-bound-2026-11-11-{a.stage}-evaluation.json').read_text())
    result=globals()[a.stage.lower()](d,c)
    report=dict(stage=a.stage,criteria_commit='11c78ca',criteria_modified=False,decisions=result)
    (RECORDS/f'route3-upper-bound-2026-11-11-{a.stage}-decisions.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
