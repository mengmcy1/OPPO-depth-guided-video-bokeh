"""S1 frozen scoring and evaluator-only GT diagnostics."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
import cv2
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from route3_sharp_s1 import OUT,PREFIX,CRITERIA,scene,raster
from route3_upper_bound_evaluate import BASE,OUT as UPPER,RECORDS,items,score,pool,display,exr
from route3_ldr_u1r import model,evaluation_masks,aggregate
R1=ROOT/'outputs/route3-ldr-u1r-2026-11-10-01'
R3=ROOT/'outputs/route3-target-focus-2026-11-10-01'


def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def write_csv(path,rows):
    with path.open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)


def edge_maps(frame):
    g=np.load(BASE/'B2-high/geometry'/f'f{frame:03d}-center.npz');z=g['depth_axial_m'];known=g['center_visible']
    lo=cv2.erode(np.where(known,z,1e30),np.ones((3,3),np.uint8));hi=cv2.dilate(np.where(known,z,0),np.ones((3,3),np.uint8))
    edge=known&((1/np.maximum(lo,1e-8)-1/np.maximum(hi,1e-8))>.03/np.maximum(lo,1e-8))
    edge=cv2.dilate(edge.astype(np.uint8),np.ones((7,7),np.uint8))>0
    lo=cv2.erode(np.where(known,z,1e30),np.ones((7,7),np.uint8));hi=cv2.dilate(np.where(known,z,0),np.ones((7,7),np.uint8))
    return g,edge&known,lo,hi


def depth_stats(pred,truth,mask):
    a=pred[mask].astype(np.float64);b=truth[mask].astype(np.float64)
    return dict(pixels=len(a),AbsRel=float(np.mean(abs(a-b)/b)),RMSE_m=float(np.sqrt(np.mean((a-b)**2))),
        median_pred_to_GT=float(np.median(a/b)),delta1=float(np.mean(np.maximum(a/b,b/a)<1.25)))


def zero(c):
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    folder=OUT/'zero-evaluation';folder.mkdir(exist_ok=False);network=model();metrics=[];depth=[];figures=[]
    predictions=json.loads((OUT/'zero-predictions.json').read_text())['rows']
    cases=items()
    for frame in [1,30,60]:
        g,edge,_,_=edge_maps(frame);d=np.load(OUT/'depth'/f'f{frame:03d}.npz');truth=d['rgb_srgb']
        near=next(x for x in cases if x['frame_id']==frame and x['focus_distance_m']==1.4 and x['fstop']==1.4)
        far=next(x for x in cases if x['frame_id']==frame and x['focus_distance_m']==5. and x['fstop']==1.4)
        from route3_target_focus import target_mask
        masks=dict(full_image=np.ones(edge.shape,bool),near_in_focus=target_mask(near),far_in_focus=target_mask(far),boundary=edge)
        for label in ['native','oracle_scale']:
            selected=[p for p in predictions if p['frame']==frame and p['scale_label']==label];images={}
            for pred in selected:
                a=np.load(pred['path']);images[pred['method']]=a['prediction_srgb']
                for fill in ['LaMa','black']:
                    img=a['prediction_srgb' if fill=='LaMa' else 'black_srgb'];values=score(img,truth,masks,network)
                    metrics.append(dict(frame=frame,scale_label=label,method=pred['method'],fill=fill,metrics=values))
            fig,axes=plt.subplots(2,3,figsize=(12,8));imgs=[truth,images['SHARP_front'],images['SHARP_full'],images['VG_SHARP'],np.load(selected[1]['path'])['black_srgb'],abs(images['SHARP_full']-truth).clip(0,1)*5]
            for ax,img,name in zip(axes.flat,imgs,['Input LDR','SHARP front + fill','SHARP full + fill','VG SHARP depth','SHARP front black','Full error x5']):ax.imshow(img.clip(0,1));ax.set_title(name);ax.axis('off')
            fig.suptitle(f'frame{frame} {label}; aperture0 K1');fig.tight_layout();path=folder/f'f{frame:03d}-{label}.png';fig.savefig(path,dpi=120);plt.close(fig);figures.append(str(path))
        S=json.loads((OUT/'scale.json').read_text())['oracle_global'];da3=np.load(UPPER/'U2-da3/E1'/f'f{frame:03d}.npz')['depth']
        for name,z in [('SHARP_native',d['depth']),('SHARP_Oracle_scale',d['depth']*S),('DA3_original_global_Sim3',da3)]:
            depth.append(dict(frame=frame,method=name,full=depth_stats(z,g['depth_axial_m'],g['center_visible']),boundary=depth_stats(z,g['depth_axial_m'],edge),GT_for_scoring_only=True))
    record=dict(completed=True,criteria=str(CRITERIA),metrics=metrics,depth_errors=depth,visualizations=figures,
        projection_export_check='official/project classic eps0 agreed before this score; common AA eps0.0015 evaluated here',old_results_modified=False)
    write(RECORDS/(PREFIX+'-zero-and-depth.json'),record)
    for r in metrics:
        if r['scale_label']=='native' and r['fill']=='LaMa':print('SHARP_ZERO',r['frame'],r['method'],r['metrics']['full_image']['psnr'],flush=True)


def comparisons(metrics,pooled,left,right,guard,c,noise,reference,region='band'):
    result={}
    for focus in [1.4,5.]:
        cases=[x for x in items() if x['focus_distance_m']==focus];table=pooled[str(focus)]
        a,b=table[left][region],table[right][region];gain=a['psnr']-b['psnr'];delta=a['patch_lpips']-b['patch_lpips']
        ns=[noise[region] if reference=='LDR' else c['noise']['per_lens_band'][f'N{x["fstop"]:g}-F{focus:g}'] for x in cases]
        n=float(np.average(ns,weights=[metrics[x['id']][left][region]['pixels'] for x in cases]));groups=[]
        for x,ni in zip(cases,ns):
            a1,b1=metrics[x['id']][left][region],metrics[x['id']][right][region]
            groups.append(dict(id=x['id'],pixels=a1['pixels'],PSNR_gain_dB=a1['psnr']-b1['psnr'],LPIPS_change=a1['patch_lpips']-b1['patch_lpips'],noise=ni,
                same_direction=bool(a1['psnr']>b1['psnr'] and a1['patch_lpips']-b1['patch_lpips']<-3*ni)))
        valid=sum(g['pixels']>=1000 for g in groups);same=sum(g['pixels']>=1000 and g['same_direction'] for g in groups)
        quality=bool(gain>=.5 or (-delta>=.1*b['patch_lpips'] and -delta>3*n));secondary=bool(gain>=-.2 and delta<=3*n)
        guards={r:table[left][r]['psnr']-table[guard][r]['psnr'] for r in ['full_image','in_focus']}
        local=quality and secondary and valid==6 and same>=5
        result[str(focus)]=dict(left=left,right=right,region=region,PSNR_gain_dB=gain,LPIPS_change=delta,noise=n,quality_passed=quality,secondary_passed=secondary,
            valid_groups=valid,same_direction=same,groups=groups,guardrail_O3=guard,guardrails_dB=guards,local_passed=local,passed=local and all(v>=-.2 for v in guards.values()))
    return result


def quality(c):
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    network=model();refs=json.loads((R1/'references.json').read_text());reference={x['id']:x['path'] for x in refs['rows']};noise=refs['max_noise_by_region']
    predictions=json.loads((OUT/'predictions.json').read_text())['rows'];cached=json.loads((R3/'predictions.json').read_text())['rows'];lookup={x['id']:x for x in items()}
    old=json.loads((RECORDS/'route3-target-focus-2026-11-10-evaluation.json').read_text())
    for label in ['native','oracle_scale']:
        methods={key:{} for key in lookup};metrics={'Cycles':{},'LDR':{}};flat=[];plots=[]
        for pred in predictions:
            if pred['scale_label']==label:methods[pred['id']][pred['method']]=pred['path']
        for pred in cached:
            if pred['method'] in ['O3_E1','O5_E1']:methods[pred['id']]['O3_DA3' if pred['method']=='O3_E1' else 'VG_DA3']=pred['path']
        folder=OUT/('evaluation-'+label);folder.mkdir(exist_ok=False)
        for case,paths in methods.items():
            x=lookup[case];ldr=np.load(reference[case])['reference_srgb'];target,masks,_=evaluation_masks(x,c,ldr);masks={k:masks[k] for k in c['regions']}
            metrics['Cycles'][case]={};metrics['LDR'][case]={}
            for name,path in paths.items():
                pred=np.load(path)['prediction_srgb'];assert pred.shape==(512,512,3) and np.isfinite(pred).all()
                for ref,truth in [('Cycles',target),('LDR',ldr)]:
                    values=score(pred,truth,masks,network);metrics[ref][case][name]=values
                    if name in ['O3_DA3','VG_DA3']:
                        oldname='O3_E1' if name=='O3_DA3' else 'O5_E1'
                        for region in ['full_image','in_focus','out_of_focus','band','recoverable']:
                            assert abs(values[region]['psnr']-old['per_case_metrics'][ref][case][oldname][region]['psnr'])<1e-9
                    for region,v in values.items():flat.append(dict(scale_label=label,reference=ref,id=case,method=name,region=region,PSNR=v['psnr'],LPIPS=v['patch_lpips'],pixels=v['pixels']))
                print('SHARP_SCORE',label,case,name,flush=True)
            names=['O3_DA3','O3_SHARP','VG_SHARP','SHARP_front','SHARP_full','SHARP_front_L3_E1','SHARP_full_L3_E1','SHARP_front_L3_E2','SHARP_full_L3_E2']
            images=[target,ldr]+[np.load(paths[name])['prediction_srgb'] for name in names]
            # Both crops follow fixed GT score regions, solely for display.
            for region in ['in_focus','band']:
                y,z=np.nonzero(masks[region]);cy=int(np.clip(np.median(y)-64,0,384));cx=int(np.clip(np.median(z)-64,0,384))
                fig,axes=plt.subplots(2,len(images),figsize=(3*len(images),6))
                for col,(img,name) in enumerate(zip(images,['Cycles','LDR reference']+names)):
                    axes[0,col].imshow(img);axes[1,col].imshow(img[cy:cy+128,cx:cx+128]);axes[0,col].set_title(name,fontsize=8)
                    axes[0,col].axis('off');axes[1,col].axis('off')
                fig.suptitle(case+' '+label+' crop:'+region);fig.tight_layout();path=folder/(case+'-'+region+'.png');fig.savefig(path,dpi=100);plt.close(fig);plots.append(str(path))
        pooled,aperture=aggregate(metrics,c);decisions={}
        pairs=[('primary','SHARP_front','O3_SHARP','O3_SHARP'),('secondary','SHARP_front','O3_DA3','O3_DA3')]
        pairs +=[(name+'_vs_full',name,'SHARP_full','O3_SHARP') for name in ['SHARP_front_L3_E1','SHARP_front_L3_E2','SHARP_full_L3_E1','SHARP_full_L3_E2']]
        for ref in ['Cycles','LDR']:
            decisions[ref]={key:{reg:comparisons(metrics[ref],pooled[ref],left,right,guard,c,noise,ref,reg) for reg in ['band','recoverable']} for key,left,right,guard in pairs}
        record=dict(completed=True,scale_label=label,Oracle_diagnostic_only=label!='native',criteria_commit='627ee78',per_case_metrics=metrics,pooled=pooled,per_aperture=aperture,
            decisions=decisions,predictions=methods,visualizations=plots,reference_noise=noise,old_R3_baseline_PSNR_identical=True,old_results_and_decisions_changed=False,
            recoverable_is_secondary_local_report=True,Cycles_recoverable_direction_uses_frozen_band_noise=True)
        write(RECORDS/(PREFIX+'-'+label+'-evaluation.json'),record);write_csv(RECORDS/(PREFIX+'-'+label+'-evaluation.csv'),flat)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['zero','quality'],required=True);a=p.parse_args()
    c=json.loads(CRITERIA.read_text());assert c['frozen'];torch.set_num_threads(8);cv2.setNumThreads(1)
    globals()[a.mode](c)


if __name__=='__main__':main()
