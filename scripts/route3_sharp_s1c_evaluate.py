"""Frozen E1-only P3/P4/P5 and descriptive four-stage retrieval ladder."""
import json
import os
import sys
from pathlib import Path
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from route3_sharp_s1c import OUT,PREFIX,CRITERIA,RECORDS,S1
from route3_upper_bound_evaluate import items,score,display,exr
from route3_ldr_u1r import model,evaluation_masks,aggregate
from route3_sharp_s1_evaluate import comparisons,write_csv,edge_maps
from route3_target_focus import target_mask


def write(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def main():
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    torch.set_num_threads(8);c=json.loads(CRITERIA.read_text());assert c['frozen'];root=OUT/'evaluation';root.mkdir(exist_ok=False);network=model()
    run=json.loads((OUT/'predictions.json').read_text());refs=json.loads((ROOT/'outputs/route3-ldr-u1r-2026-11-10-01/references.json').read_text());reference={r['id']:r['path'] for r in refs['rows']};old=json.loads((RECORDS/'route3-sharp-s1b-2026-11-11-native-evaluation.json').read_text());lookup={x['id']:x for x in items()}
    paths={x['id']:{} for x in items()}
    for row in run['rows']:paths[row['id']][row['geometry']+'__'+row['method']]=row['path']
    for case in paths:
        for name in ['O3_SHARP','O3_DA3']:paths[case]['native__'+name]=old['predictions'][case][name]
    metrics={'Cycles':{},'LDR':{}};flat=[];figures=[];cracks=[]
    for case,methods in paths.items():
        x=lookup[case];ldr=np.load(reference[case])['reference_srgb'];target,masks,_=evaluation_masks(x,c,ldr);masks={k:masks[k] for k in c['regions']};focusmask=target_mask(x)
        for ref in metrics:metrics[ref][case]={}
        for name,path in methods.items():
            p=np.load(path);image=p['prediction_srgb'];assert np.isfinite(image).all()
            for ref,truth in [('Cycles',target),('LDR',ldr)]:metrics[ref][case][name]=score(image,truth,masks,network)
            if name=='native__A_color':
                for region in c['regions']:assert abs(metrics['Cycles'][case][name][region]['psnr']-old['per_case_metrics']['Cycles'][case]['A_color'][region]['psnr'])<1e-9
            if 'alpha' in p:
                T=1-p['alpha'];fill=p['fill_contribution_linear'];rgb=p['rgb_linear'];cracks.append(dict(id=case,method=name,focus_m=x['focus_distance_m'],pixels=int(focusmask.sum()),leaky_pixels=int((focusmask&(T>.1)).sum()),mean_transmittance=float(T[focusmask].mean()),fill_L1_sum=float(abs(fill[focusmask]).sum()),RGB_L1_sum=float(abs(rgb[focusmask]).sum())))
            print('S1C_SCORE',case,name,flush=True)
        for geometry in ['native','cal_E1']:
            levels=['O3_SHARP','A_color','A_color_O6_GT_rec','A_color_L3_GTgeometry','A_color_L3_cal_E1','A_color_L3_native_E1','A_color_edge'];images=[target,ldr]+[np.load(methods[geometry+'__'+m])['prediction_srgb'] for m in levels]
            for region in ['band','recoverable']:
                y,z=np.nonzero(masks[region]);i=np.argmin((y-np.median(y))**2+(z-np.median(z))**2);cy=int(np.clip(y[i]-64,0,384));cx=int(np.clip(z[i]-64,0,384));fig,axes=plt.subplots(2,len(images),figsize=(3*len(images),6))
                for col,(image,title) in enumerate(zip(images,['Cycles','LDR']+levels)):
                    axes[0,col].imshow(image);axes[1,col].imshow(image[cy:cy+128,cx:cx+128]);axes[0,col].set_title(title,fontsize=7)
                    for ax in axes[:,col]:ax.axis('off')
                fig.suptitle(case+' '+geometry+'; '+region);fig.tight_layout();path=root/(case+'-'+geometry+'-'+region+'.png');fig.savefig(path,dpi=105);plt.close(fig);figures.append(str(path))
    pooled,aperture=aggregate(metrics,c);decisions={};ladder=[]
    pairs=[('P3_E1','cal_E1__A_color_L3_cal_E1','cal_E1__A_color','band')]
    for method in ['O3_SHARP','O3_DA3','A_color']:pairs.append(('P4_'+method,'cal_E1__'+method,'native__'+method,'band'))
    pairs.append(('P4_A_color_L3','cal_E1__A_color_L3_cal_E1','native__A_color_L3_native_E1','band'))
    for geometry in ['native','cal_E1']:
        pairs += [('P5_'+geometry,geometry+'__A_color_edge',geometry+'__A_color','recoverable'),('P1_edge_'+geometry,geometry+'__A_color_edge',geometry+'__O3_SHARP','band'),('P1_A_color_'+geometry,geometry+'__A_color',geometry+'__O3_SHARP','band')]
    for ref in metrics:
        decisions[ref]={label:comparisons(metrics[ref],pooled[ref],left,right,right,c,refs['max_noise_by_region'],ref,region) for label,left,right,region in pairs}
        for focus,table in pooled[ref].items():
            for method,regions in table.items():
                for region,value in regions.items():flat.append(dict(reference=ref,focus_m=focus,method=method,region=region,**value))
            levels=['A_color_O6_GT_rec','A_color_L3_GTgeometry','A_color_L3_cal_E1','A_color_L3_native_E1']
            for geometry in ['native','cal_E1']:
                for index,level in enumerate(levels):
                    for region in ['band','recoverable','in_focus','full_image']:
                        a=table[geometry+'__'+level][region];b=table[geometry+'__A_color'][region];previous=table[geometry+'__'+levels[index-1]][region] if index else None
                        ladder.append(dict(reference=ref,focus_m=focus,geometry=geometry,level=level,region=region,PSNR=a['psnr'],LPIPS=a['patch_lpips'],versus_visible_PSNR_dB=a['psnr']-b['psnr'],versus_visible_LPIPS=a['patch_lpips']-b['patch_lpips'],
                            next_minus_previous_PSNR_dB=a['psnr']-previous['psnr'] if previous else None,next_minus_previous_LPIPS=a['patch_lpips']-previous['patch_lpips'] if previous else None,descriptive_Oracle_only=True))
    zero=[]
    for row in run['zeros']:
        f=row['frame'];truth=np.load(S1/'depth'/f'f{f:03d}.npz')['rgb_srgb'];_,edge,_,_=edge_maps(f);near=next(x for x in items() if x['frame_id']==f and x['focus_distance_m']==1.4 and x['fstop']==1.4);far=next(x for x in items() if x['frame_id']==f and x['focus_distance_m']==5. and x['fstop']==1.4)
        masks=dict(full_image=np.ones((512,512),bool),near_in_focus=target_mask(near),far_in_focus=target_mask(far),boundary=edge);zero.append(dict(geometry=row['geometry'],frame=f,method=row['method'],metrics=score(np.load(row['path'])['prediction_srgb'],truth,masks,network)))
    report=dict(completed=True,E1_GT_poses=True,E2='not completed: insufficient support blocked',criteria_commit='4569f66',continuation_commit='276e7cd',per_case_metrics=metrics,pooled=pooled,per_aperture=aperture,decisions=decisions,ladder=ladder,cracks=cracks,zero=zero,predictions=paths,visualizations=figures,
        claims_conditional_on_accurate_poses=True,GT_depth_used_only_scoring_focus_Oracle_diagnostics=True,old_results_unchanged=True)
    write(RECORDS/(PREFIX+'-evaluation.json'),report);write_csv(RECORDS/(PREFIX+'-tables.csv'),flat);write_csv(RECORDS/(PREFIX+'-ladder.csv'),ladder)


if __name__=='__main__':main()
