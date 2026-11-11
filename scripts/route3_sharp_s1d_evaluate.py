"""S1d frozen P6, unselected rule sequence and MSE-gap recovery."""
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from route3_sharp_s1d import OUT,PREFIX,CRITERIA,RECORDS,write
from route3_upper_bound_evaluate import items,score,display,exr
from route3_ldr_u1r import model,evaluation_masks,aggregate
from route3_sharp_s1_evaluate import comparisons,write_csv
METHODS=['R1','R2','R3','R4','Oracle_depth_only','Oracle_color_only','Oracle_missing_only']


def recovery(method,visible,oracle):
    a=method['sse']/(3*method['pixels']);b=visible['sse']/(3*visible['pixels']);g=oracle['sse']/(3*oracle['pixels']);gain=oracle['psnr']-visible['psnr'];den=b-g
    if gain<.1 or den<=0 or den<.01*b:return dict(ratio=None,reason='Oracle not sufficiently better than visible',Oracle_gain_dB=gain)
    return dict(ratio=float((b-a)/den),reason=None,Oracle_gain_dB=gain)


def main():
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    torch.set_num_threads(8);c=json.loads(CRITERIA.read_text());assert c['frozen'];root=OUT/'evaluation';root.mkdir(exist_ok=False);network=model();old=json.loads((RECORDS/'route3-sharp-s1c-2026-11-10-evaluation.json').read_text());refs=json.loads((ROOT/'outputs/route3-ldr-u1r-2026-11-10-01/references.json').read_text());refmap={x['id']:x['path'] for x in refs['rows']};lookup={x['id']:x for x in items()};metrics={ref:{case:{} for case in lookup} for ref in ['Cycles','LDR']};paths={case:{} for case in lookup};figures=[];flat=[];stages={m:json.loads((OUT/'render'/m/'predictions.json').read_text())['rows'] for m in METHODS}
    aliases=dict(A_color='cal_E1__A_color',O3_DA3='cal_E1__O3_DA3',O3_SHARP='cal_E1__O3_SHARP',R0='cal_E1__A_color_L3_cal_E1',GTrec='cal_E1__A_color_O6_GT_rec')
    for case in lookup:
        for name,oldname in aliases.items():
            paths[case][name]=old['predictions'][case][oldname]
            for ref in metrics:metrics[ref][case][name]=old['per_case_metrics'][ref][case][oldname]
    for case,x in lookup.items():
        ldr=np.load(refmap[case])['reference_srgb'];truth,masks,_=evaluation_masks(x,c,ldr);masks={k:masks[k] for k in c['regions']}
        for method in METHODS:
            row=next(r for r in stages[method] if r['id']==case);paths[case][method]=row['path'];p=np.load(row['path'])['prediction_srgb'];assert np.isfinite(p).all()
            for ref,t in [('Cycles',truth),('LDR',ldr)]:metrics[ref][case][method]=score(p,t,masks,network)
            print('S1D_SCORE',case,method,flush=True)
        for family,names in [('rules',['A_color','O3_DA3','R0','R1','R2','R3','R4','GTrec']),('replacements',['A_color','R0','Oracle_depth_only','Oracle_color_only','Oracle_missing_only','GTrec'])]:
            images=[truth]+[np.load(paths[case][m])['prediction_srgb'] for m in names];y,z=np.nonzero(masks['recoverable']);i=np.argmin((y-np.median(y))**2+(z-np.median(z))**2);cy=int(np.clip(y[i]-64,0,384));cx=int(np.clip(z[i]-64,0,384));fig,axes=plt.subplots(2,len(images),figsize=(3*len(images),6))
            for col,(image,name) in enumerate(zip(images,['Cycles']+names)):
                axes[0,col].imshow(image);axes[1,col].imshow(image[cy:cy+128,cx:cx+128]);axes[0,col].set_title(name,fontsize=8)
                for ax in axes[:,col]:ax.axis('off')
            fig.suptitle(case+' '+family+'; GT-only crop');fig.tight_layout();path=root/(case+'-'+family+'.png');fig.savefig(path,dpi=110);plt.close(fig);figures.append(str(path))
    pooled,aperture=aggregate(metrics,c);decisions={};gap=[];direct=[]
    for ref in metrics:
        decisions[ref]={region:comparisons(metrics[ref],pooled[ref],'R4','A_color','A_color',c,refs['max_noise_by_region'],ref,region) for region in ['recoverable','band']}
        for focus,table in pooled[ref].items():
            for method,regions in table.items():
                for region,value in regions.items():
                    flat.append(dict(reference=ref,focus_m=focus,method=method,region=region,**value));gap.append(dict(reference=ref,focus_m=focus,method=method,region=region,**recovery(value,table['A_color'][region],table['GTrec'][region])))
                    if region in ['full_image','in_focus','band']:direct.append(dict(reference=ref,focus_m=focus,method=method,region=region,versus_calibrated_O3_DA3_PSNR_dB=value['psnr']-table['O3_DA3'][region]['psnr'],versus_calibrated_O3_DA3_LPIPS=value['patch_lpips']-table['O3_DA3'][region]['patch_lpips'],descriptive_only=True))
        for focus in ['1.4','5.0']:decisions[ref].setdefault('P6_joint',{})[focus]=all(decisions[ref][reg][focus]['passed'] for reg in ['recoverable','band'])
    write(RECORDS/(PREFIX+'-evaluation.json'),dict(completed=True,criteria_commit='634b30f',per_case_metrics=metrics,pooled=pooled,per_aperture=aperture,P6=decisions,gap_recovery=gap,baseline_O3_DA3_descriptive=direct,predictions=paths,visualizations=figures,
        fixed_visible_no_refit=True,final_R4_not_result_selected=True,Oracle_replacements_not_methods=True,old_baselines_reused_exactly=True,E1_GT_poses=True))
    write_csv(RECORDS/(PREFIX+'-tables.csv'),flat);write_csv(RECORDS/(PREFIX+'-gap-recovery.csv'),gap);write_csv(RECORDS/(PREFIX+'-O3-DA3-descriptive.csv'),direct)


if __name__=='__main__':main()
