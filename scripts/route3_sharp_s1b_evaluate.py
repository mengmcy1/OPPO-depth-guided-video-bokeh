"""S1b native-only scores; calibration gates remain uncomputed."""
import argparse
import json
import os
import sys
from pathlib import Path
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from route3_sharp_s1b import OUT,PREFIX,CRITERIA,write
from route3_sharp_s1 import OUT as S1,PREFIX as S1PREFIX
from route3_sharp_s1_evaluate import comparisons,edge_maps,write_csv
from route3_upper_bound_evaluate import BASE,RECORDS,items,score,display,exr
from route3_ldr_u1r import model,evaluation_masks,aggregate
from route3_target_focus import target_mask


def quality(c):
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    root=OUT/'evaluation-native';root.mkdir(exist_ok=False);network=model()
    prior=json.loads((RECORDS/(S1PREFIX+'-native-evaluation.json')).read_text())
    refs=json.loads((ROOT/'outputs/route3-ldr-u1r-2026-11-10-01/references.json').read_text());refmap={r['id']:r['path'] for r in refs['rows']}
    cached=['O3_DA3','O3_SHARP','VG_SHARP','SHARP_full','SHARP_full_L3_E1','SHARP_full_L3_E2']
    metrics={ref:{case:{m:v for m,v in values.items() if m in cached} for case,values in bycase.items()} for ref,bycase in prior['per_case_metrics'].items()}
    paths={case:{m:p for m,p in bymethod.items() if m in cached} for case,bymethod in prior['predictions'].items()}
    new=json.loads((OUT/'native-predictions.json').read_text());figures=[];flat=[]
    for x in items():
        case=x['id'];ldr=np.load(refmap[case])['reference_srgb'];truth,masks,_=evaluation_masks(x,c,ldr);masks={r:masks[r] for r in c['regions']}
        for row in [r for r in new['rows'] if r['id']==case]:
            pred=np.load(row['path'])['prediction_srgb'];assert np.isfinite(pred).all();paths[case][row['method']]=row['path']
            for ref,t in [('Cycles',truth),('LDR',ldr)]:metrics[ref][case][row['method']]=score(pred,t,masks,network)
            print('S1B_SCORE',case,row['method'],flush=True)
        names=['O3_SHARP','SHARP_full','A_color','A_color_opacity','A_color_L3_E1','A_color_L3_E2'];images=[truth,ldr]+[np.load(paths[case][n])['prediction_srgb'] for n in names]
        for region in ['in_focus','band']:
            y,z=np.nonzero(masks[region]);cy=int(np.clip(np.median(y)-64,0,384));cx=int(np.clip(np.median(z)-64,0,384))
            fig,axes=plt.subplots(2,len(images),figsize=(3*len(images),6))
            for col,(image,title) in enumerate(zip(images,['Cycles','LDR']+names)):
                axes[0,col].imshow(image);axes[1,col].imshow(image[cy:cy+128,cx:cx+128]);axes[0,col].set_title(title,fontsize=8)
                for ax in axes[:,col]:ax.axis('off')
            fig.suptitle(case+' native; crop:'+region);fig.tight_layout();p=root/(case+'-'+region+'.png');fig.savefig(p,dpi=105);plt.close(fig);figures.append(str(p))
    pooled,aperture=aggregate(metrics,c);decisions={};hidden=[]
    for ref in ['Cycles','LDR']:
        decisions[ref]={}
        for method in ['A_color','A_color_opacity']:
            for label,right in [('P1','O3_SHARP'),('P1_secondary_DA3','O3_DA3'),('P2','SHARP_full')]:
                guard='O3_DA3' if label=='P1_secondary_DA3' else 'O3_SHARP'
                decisions[ref][label+'_'+method]={r:comparisons(metrics[ref],pooled[ref],method,right,guard,c,refs['max_noise_by_region'],ref,r) for r in ['band','recoverable']}
        for focus,table in pooled[ref].items():
            for method,regions in table.items():
                for region,v in regions.items():flat.append(dict(reference=ref,focus_m=focus,method=method,region=region,**v))
            for method in ['A_color','A_color_opacity']:
                for E in ['E1','E2']:
                    for region in c['regions']:
                        a=table[method+'_L3_'+E][region];b=table[method][region];old=table['SHARP_full_L3_'+E][region]
                        hidden.append(dict(reference=ref,focus_m=focus,method=method,poses=E,region=region,PSNR_increment_dB=a['psnr']-b['psnr'],LPIPS_increment=a['patch_lpips']-b['patch_lpips'],
                            versus_S1_full_L3_PSNR_dB=a['psnr']-old['psnr'],versus_S1_full_L3_LPIPS=a['patch_lpips']-old['patch_lpips'],descriptive_only=True))
    write(RECORDS/(PREFIX+'-native-evaluation.json'),dict(completed=True,per_case_metrics=metrics,pooled=pooled,per_aperture=aperture,decisions=decisions,hidden_descriptive=hidden,
        predictions=paths,visualizations=figures,baseline_metrics_reused_exactly=True,P3='not completed: calibration paused',P4='not completed: calibration paused',criteria_commits=['9dc09ef','36011d7']))
    write_csv(RECORDS/(PREFIX+'-native-tables.csv'),flat);write_csv(RECORDS/(PREFIX+'-hidden-descriptive.csv'),hidden)


def zero_diagnostics(c):
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    network=model();new=json.loads((OUT/'native-predictions.json').read_text());zeros=[];cracks=[];figures=[]
    root=OUT/'zero-diagnostics';root.mkdir(exist_ok=False)
    oldzero=json.loads((RECORDS/(S1PREFIX+'-zero-and-depth.json')).read_text());oldfly=json.loads((RECORDS/(S1PREFIX+'-diagnostics.json')).read_text())
    for row in new['zero']:
        frame=row['frame'];d=np.load(S1/'depth'/f'f{frame:03d}.npz');_,edge,_,_=edge_maps(frame)
        near=next(x for x in items() if x['frame_id']==frame and x['focus_distance_m']==1.4 and x['fstop']==1.4);far=next(x for x in items() if x['frame_id']==frame and x['focus_distance_m']==5. and x['fstop']==1.4)
        masks=dict(full_image=np.ones((512,512),bool),near_in_focus=target_mask(near),far_in_focus=target_mask(far),boundary=edge)
        values=score(np.load(row['path'])['prediction_srgb'],d['rgb_srgb'],masks,network)
        baseline=next(r['metrics'] for r in oldzero['metrics'] if r['frame']==frame and r['scale_label']=='native' and r['fill']=='LaMa' and r['method']=='SHARP_full')
        zeros.append(dict(frame=frame,method=row['method'],before=baseline,after=values))
        fit=json.loads((OUT/'anchor/native'/f'f{frame:03d}'/row['method']/'result.json').read_text())
        fig,axes=plt.subplots(1,2,figsize=(8,3));axes[0].plot([r['step'] for r in fit['rows']],[r['input_PSNR_dB'] for r in fit['rows']]);axes[0].set_title('Target-only input reconstruction PSNR');axes[0].set_xlabel('fixed step')
        axes[1].imshow(np.clip(abs(np.load(row['path'])['prediction_srgb']-d['rgb_srgb'])*20,0,1));axes[1].set_title('Final target error x20');axes[1].axis('off');fig.tight_layout();p=root/f'f{frame:03d}-{row["method"]}.png';fig.savefig(p,dpi=130);plt.close(fig);figures.append(str(p))
    lookup={x['id']:x for x in items()};prior=json.loads((S1/'predictions.json').read_text())['rows']
    for row in new['rows']:
        a=np.load(row['path']);mask=target_mask(lookup[row['id']]);T=1-a['alpha'];leak=mask&(T>.1);rgb=a['rgb_linear'];fill=a['fill_contribution_linear']
        if row['method']=='A_color':
            old=next(p for p in prior if p['scale_label']=='native' and p['id']==row['id'] and p['method']=='SHARP_full')
            assert np.array_equal(a['alpha'],np.load(old['path'])['alpha'])
        cracks.append(dict(id=row['id'],focus_m=lookup[row['id']]['focus_distance_m'],method=row['method'],pixels=int(mask.sum()),leaky_pixels=int(leak.sum()),leaky_fraction=float(leak.sum()/mask.sum()),
            mean_transmittance=float(T[mask].mean()),fill_L1_share=float(abs(fill[mask]).sum()/max(abs(rgb[mask]).sum(),1e-12))))
    write(RECORDS/(PREFIX+'-zero-diagnostics.json'),dict(completed=True,zero=zeros,cracks=cracks,geometry_fly_membership_unchanged=True,
        fly_counts_reused=[r for r in oldfly['flypoints'] if r['scale_label']=='native' and r['method']=='SHARP_full'],
        A_color_alpha_and_fly_alpha_contributions_unchanged=True,A_color_opacity_fly_membership_same_but_contribution_can_change=True,
        visualizations=figures,transparency_is_support_proxy_not_certified_geometric_crack=True))


def flypoints(c):
    import subprocess
    from route3_sharp_s1b import fitted_scene
    from route3_sharp_s1 import raster
    from src.aperture_renderer import GaussianScene
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise ValueError('Explicit GPU0')
    subprocess.run(['nvidia-smi'],check=True);rows=[]
    for frame in [1,30,60]:
        g,edge,lo,hi=edge_maps(frame);d=np.load(S1/'depth'/f'f{frame:03d}.npz');cloud=fitted_scene(frame,'A_color_opacity')
        means=cloud.means.cpu().numpy();z=means[:,2];uvz=means@d['K'].T;uv=uvz[:,:2]/np.maximum(uvz[:,2:],1e-8);x=np.floor(uv[:,0]).astype(int);y=np.floor(uv[:,1]).astype(int)
        inside=(z>0)&(x>=0)&(x<512)&(y>=0)&(y<512);ids=np.flatnonzero(inside);selected=np.zeros(len(z),bool);a,b=lo[y[ids],x[ids]],hi[y[ids],x[ids]]
        selected[ids]=edge[y[ids],x[ids]]&(z[ids]>a+np.maximum(.005,.01*a))&(z[ids]<b-np.maximum(.005,.01*b))
        tag=torch.tensor(selected,device='cuda',dtype=torch.float32)[:,None].expand(-1,3).contiguous();marked=GaussianScene(cloud.means,cloud.quats,cloud.scales,cloud.opacities,tag,cloud.source_frames)
        with torch.inference_mode():image,_,_=raster(marked,torch.tensor(d['K'],device='cuda'))
        heat=image[...,0].cpu().numpy();rows.append(dict(frame=frame,method='A_color_opacity',fly_count=int(selected.sum()),Gaussians=len(z),mean_tag_contribution=float(heat.mean()),mean_tag_contribution_near_edge=float(heat[edge].mean())))
        del cloud,marked;torch.cuda.empty_cache()
    write(RECORDS/(PREFIX+'-fly-opacity.json'),dict(completed=True,rows=rows,tagged_K1_proxy_not_certified_outlier_or_K64_contribution=True,GT_evaluator_only=True))


def temporal(c):
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    root=OUT/'temporal/evaluation';root.mkdir(exist_ok=False);network=model();run=json.loads((OUT/'temporal/predictions.json').read_text())['rows']
    old=json.loads((RECORDS/'route3-dual-layer-2026-11-10-temporal-inputs.json').read_text());print('OLD_TEMP_KEYS',old.keys(),flush=True)
    # The existing temporal manifest is the source of the 5m reference paths.
    oldrows=old['items'];truth5={r['frame']:r['gt_rgb_exr'] for r in oldrows}
    truth14={f:str(OUT/'temporal/Cycles-F1.4'/f'f{f:03d}-N1.4-F1.4-s2048-seed20261110.exr') for f in range(27,35)}
    result={};flat=[];figures=[]
    for focus,truthmap in [(5.,truth5),(1.4,truth14)]:
        table={};gt={f:display(exr(truthmap[f])) for f in range(27,35)}
        for method in sorted({r['method'] for r in run}):
            predictions={r['frame']:np.load(r['path'])['prediction_srgb'] for r in run if r['method']==method and r['focus_m']==focus};per=[];changes=[]
            for f in range(27,35):
                v=score(predictions[f],gt[f],dict(full_image=np.ones((512,512),bool)),network)['full_image'];per.append(dict(frame=f,**v));flat.append(dict(focus_m=focus,method=method,frame=f,**v))
            for f in range(28,35):
                err=(predictions[f].astype(np.float64)-predictions[f-1])-(gt[f].astype(np.float64)-gt[f-1]);changes.append(dict(from_frame=f-1,to_frame=f,RMSE=float(np.sqrt(np.mean(err**2)))))
            from route3_upper_bound_evaluate import pool
            table[method]=dict(per_frame=per,pooled=pool(per),transitions=changes,temporal_change_RMSE=float(np.sqrt(np.mean([v['RMSE']**2 for v in changes]))))
        ratio=table['A_color']['temporal_change_RMSE']/table['SHARP_full']['temporal_change_RMSE']
        result[str(focus)]=dict(methods=table,A_color_to_full_RMSE_ratio=ratio,diagnostic_le_1p05=ratio<=1.05,focuses_not_pooled=True)
        fig,axes=plt.subplots(1,3,figsize=(16,4))
        for name,v in table.items():
            axes[0].plot(range(27,35),[r['psnr'] for r in v['per_frame']],label=name);axes[1].plot(range(27,35),[r['patch_lpips'] for r in v['per_frame']],label=name);axes[2].plot(range(28,35),[r['RMSE'] for r in v['transitions']],label=name)
        for ax,title in zip(axes,['Cycles PSNR','Cycles patch LPIPS','Frame-change error RMSE']):ax.set_title(title);ax.set_xlabel('frame');ax.legend(fontsize=6)
        fig.suptitle(f'nominal focus {focus:g}m; input-side target focus per method');fig.tight_layout();p=root/f'F{focus:g}-curves.png';fig.savefig(p,dpi=140);plt.close(fig);figures.append(str(p))
        # Identical transition/frame for all methods, chosen by time, not error.
        names=list(table);fig,axes=plt.subplots(2,len(names)+1,figsize=(3*(len(names)+1),6))
        for col,name in enumerate(['Cycles']+names):
            ims=gt if name=='Cycles' else {r['frame']:np.load(r['path'])['prediction_srgb'] for r in run if r['method']==name and r['focus_m']==focus}
            axes[0,col].imshow(ims[30]);axes[1,col].imshow(np.clip((ims[31]-ims[30])*5+.5,0,1));axes[0,col].set_title(name,fontsize=8)
            for ax in axes[:,col]:ax.axis('off')
        fig.suptitle('frame30 and signed frame31-30 x5 + .5');fig.tight_layout();p=root/f'F{focus:g}-differences.png';fig.savefig(p,dpi=120);plt.close(fig);figures.append(str(p))
    write(RECORDS/(PREFIX+'-temporal-evaluation.json'),dict(completed=True,focus_groups=result,visualizations=figures,calibrated_variants_not_run=True,not_a_flicker_claim=True,unaligned_changes_include_motion_and_geometry=True))
    write_csv(RECORDS/(PREFIX+'-temporal-tables.csv'),flat)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['quality','zero_diagnostics','flypoints','temporal'],required=True);a=p.parse_args();torch.set_num_threads(8)
    c=json.loads(CRITERIA.read_text());assert c['frozen'];globals()[a.mode](c)


if __name__=='__main__':main()
