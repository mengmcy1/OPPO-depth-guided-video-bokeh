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


def diagnostics(c):
    import subprocess
    from src.aperture_renderer import GaussianScene
    from src.layered_aperture import pixel_layer
    from route3_target_focus import target_mask
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise ValueError('Explicit GPU0')
    subprocess.run(['nvidia-smi'],check=True)
    folder=OUT/'diagnostics';folder.mkdir(exist_ok=False);fly=[];backrows=[];cracks=[];figures=[]
    prepared=json.loads((OUT/'prepared.json').read_text())['rows'];preds=json.loads((OUT/'predictions.json').read_text())['rows'];lookup={x['id']:x for x in items()}
    for entry in prepared:
        frame=entry['frame'];label=entry['scale_label'];d=np.load(entry['input_npz']);g,edge,lo,hi=edge_maps(frame)
        K=torch.tensor(d['K'],device='cuda');eye=torch.eye(4,device='cuda');maps={}
        clouds=dict(VG_SHARP=pixel_layer(torch.tensor(d['rgb_linear'],device='cuda'),torch.tensor(d['depth'],device='cuda'),K,eye),
            SHARP_front=scene(frame,'front',entry['scale']),SHARP_full=scene(frame,'full',entry['scale']))
        for name,cloud in clouds.items():
            means=cloud.means.cpu().numpy();z=means[:,2];uvz=means@d['K'].T;uv=uvz[:,:2]/np.maximum(uvz[:,2:],1e-8)
            x=np.floor(uv[:,0]).astype(np.int32);y=np.floor(uv[:,1]).astype(np.int32)
            inside=(z>0)&(x>=0)&(x<512)&(y>=0)&(y<512);ids=np.flatnonzero(inside);selected=np.zeros(len(z),bool)
            a,b=lo[y[ids],x[ids]],hi[y[ids],x[ids]]
            selected[ids]=edge[y[ids],x[ids]]&(z[ids]>a+np.maximum(.005,.01*a))&(z[ids]<b-np.maximum(.005,.01*b))
            tag=torch.tensor(selected,device='cuda',dtype=torch.float32)[:,None].expand(-1,3).contiguous()
            marked=GaussianScene(cloud.means,cloud.quats,cloud.scales,cloud.opacities,tag,cloud.source_frames)
            with torch.inference_mode():contribution,_,_=raster(marked,K)
            heat=contribution[...,0].cpu().numpy();maps[name]=heat
            op=cloud.opacities.cpu().numpy()
            fly.append(dict(frame=frame,scale_label=label,method=name,Gaussians=len(z),in_image=int(inside.sum()),fly_count=int(selected.sum()),fly_fraction=float(selected.mean()),
                fly_intrinsic_opacity_fraction=float(op[selected].sum()/max(op.sum(),1e-12)),
                K1_mean_rendered_fly_contribution=float(heat.mean()),K1_mean_contribution_near_GT_edge=float(heat[edge].mean()),
                K1_pixels_fly_contribution_gt_001=int((heat>.01).sum()),diagnostic_only=True))
            if name=='SHARP_full':
                half=len(z)//2;back=np.arange(half,len(z));inback=back[inside[back]]
                parent=np.load(BASE/'methods-3d/prepared'/f'B2-high-f{frame:03d}'/'parents.npz')
                gtdepth=np.full(512*512,np.inf,np.float32);puv=parent['uv'];px=np.clip(puv[:,0].astype(int),0,511);py=np.clip(puv[:,1].astype(int),0,511)
                np.minimum.at(gtdepth,py*512+px,parent['depth']);gtdepth=gtdepth.reshape(512,512)
                at=gtdepth[y[inback],x[inback]];support=np.isfinite(at);matches=abs(z[inback]-at)<=np.maximum(.005,.02*at)
                # inf GT pixels are not matches and are outside the denominator.
                matches &=support
                paired_delta=z[half:]-z[:half]
                structure=dict(paired_nonpositive_depth_gap_fraction=float((paired_delta<=0).mean()),
                    paired_back_at_least_2pct_or_5mm_nearer_fraction=float((paired_delta<-np.maximum(.005,.02*z[:half])).mean()),
                    paired_depth_gap_p10_p50_p90=np.quantile(paired_delta,[.1,.5,.9]).tolist(),
                    front_opacity_p10_p50_p90=np.quantile(op[:half],[.1,.5,.9]).tolist(),back_opacity_p10_p50_p90=np.quantile(op[half:],[.1,.5,.9]).tolist(),
                    known_hidden_supported_back_Gaussians=int(support.sum()),matched_hidden_back_Gaussians=int(matches.sum()),
                    hidden_match_fraction_of_supported=float(matches.sum()/support.sum()) if support.any() else None,
                    GT_hidden_used_only_for_diagnostic=True)
                for case in entry['cases']:
                    from src.dual_layer import depth_band
                    band=depth_band(d['depth'],d['K'][0,0],case['focus_m'],.05/(2*case['fstop']))[0]
                    active=band[y[inback],x[inback]]
                    backrows.append(dict(id=case['id'],scale_label=label,back_Gaussians=half,back_in_image=len(inback),back_projected_in_band=int(active.sum()),
                        fraction_all_back_in_band=float(active.sum()/half),intrinsic_opacity_fraction_in_band=float(op[inback[active]].sum()/max(op[half:].sum(),1e-12)),**structure))
            del marked,cloud,tag
        fig,axes=plt.subplots(1,5,figsize=(15,3))
        for ax,img,title in zip(axes,[d['rgb_srgb'],edge,maps['VG_SHARP'],maps['SHARP_front'],maps['SHARP_full']],['Input','GT edge','VG intermediate depth weight','SHARP front weight','SHARP full weight']):
            ax.imshow(img,vmin=0,vmax=1);ax.set_title(title,fontsize=8);ax.axis('off')
        fig.suptitle(f'f{frame:03d} {label}; K1 tagged Gaussian alpha contributions');fig.tight_layout();p=folder/(entry['id']+'-flypoints.png');fig.savefig(p,dpi=120);plt.close(fig);figures.append(str(p))
        del clouds;torch.cuda.empty_cache()
    for row in preds:
        if row['method']=='O3_SHARP':continue
        pred=np.load(row['path']);mask=target_mask(lookup[row['id']]);T=1-pred['alpha'];leak=mask&(T>.1);fill=pred['fill_contribution_linear'];rgb=pred['rgb_linear']
        cracks.append(dict(id=row['id'],scale_label=row['scale_label'],method=row['method'],nominal_focus_m=lookup[row['id']]['focus_distance_m'],pixels=int(mask.sum()),
            mean_transmittance=float(T[mask].mean()),leaky_pixels=int(leak.sum()),leaky_fraction=float(leak.sum()/mask.sum()),
            fill_linear_RGB_mean=fill[mask].mean(0).tolist(),fill_L1_sum=float(abs(fill[mask]).sum()),RGB_L1_sum=float(abs(rgb[mask]).sum()),
            fill_L1_share=float(abs(fill[mask]).sum()/max(abs(rgb[mask]).sum(),1e-12)),
            leak_fill_L1_share=float(abs(fill[leak]).sum()/max(abs(rgb[leak]).sum(),1e-12)) if leak.any() else None))
    pooled={}
    for label in ['native','oracle_scale']:
        pooled[label]={}
        for focus in [1.4,5.]:
            pooled[label][str(focus)]={}
            for name in sorted({r['method'] for r in cracks}):
                group=[r for r in cracks if r['scale_label']==label and r['nominal_focus_m']==focus and r['method']==name];n=sum(r['pixels'] for r in group)
                pooled[label][str(focus)][name]=dict(pixels=n,leaky_pixels=sum(r['leaky_pixels'] for r in group),
                    leaky_fraction=sum(r['leaky_pixels'] for r in group)/n,mean_transmittance=sum(r['pixels']*r['mean_transmittance'] for r in group)/n,
                    fill_L1_share=sum(r['fill_L1_sum'] for r in group)/max(sum(r['RGB_L1_sum'] for r in group),1e-12))
    write(RECORDS/(PREFIX+'-diagnostics.json'),dict(completed=True,flypoints=fly,back_layer=backrows,cracks=cracks,pooled_cracks=pooled,visualizations=figures,
        GT_used_only_for_diagnostics=True,crack_metric_is_transparency_support_proxy_not_certified_geometric_crack=True,
        K1_fly_contribution_not_K64_DOF_fraction=True,raw_fly_counts_have_different_Gaussian_densities=True,
        intermediate_depth_proxy_can_include_global_depth_bias_and_slanted_surfaces_not_certified_outlier_GT=True,method_parameters_tuned=False))


def report(c):
    import datetime
    import shutil
    import subprocess
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    evaluations={label:json.loads((RECORDS/(PREFIX+'-'+label+'-evaluation.json')).read_text()) for label in ['native','oracle_scale']}
    diagnostic=json.loads((RECORDS/(PREFIX+'-diagnostics.json')).read_text());zero_result=json.loads((RECORDS/(PREFIX+'-zero-and-depth.json')).read_text())
    infer=json.loads((RECORDS/(PREFIX+'-inference.json')).read_text());prep=json.loads((OUT/'prepared.json').read_text())
    license_record=json.loads((RECORDS/(PREFIX+'-model-licenses.json')).read_text());rendered=json.loads((RECORDS/(PREFIX+'-render.json')).read_text());tables=[];contrasts=[];concise={}
    extra_pairs=[('representation_same_depth','SHARP_front','VG_SHARP'),('learned_full_vs_VG_same_depth','SHARP_full','VG_SHARP'),
        ('depth_source_VG','VG_SHARP','VG_DA3'),('depth_source_O3','O3_SHARP','O3_DA3'),('network_second_layer','SHARP_full','SHARP_front'),
        ('retrieval_added_front_E1','SHARP_front_L3_E1','SHARP_front'),('retrieval_added_front_E2','SHARP_front_L3_E2','SHARP_front')]
    for label,ev in evaluations.items():
        concise[label]={}
        for ref,byfocus in ev['pooled'].items():
            concise[label][ref]={}
            for focus,t in byfocus.items():
                for name,regions in t.items():
                    for region,a in regions.items():
                        b=t['O3_SHARP'][region];old=t['O3_DA3'][region];full=t['SHARP_full'][region]
                        tables.append(dict(scale_label=label,Oracle_diagnostic_only=label!='native',reference=ref,nominal_focus_m=focus,method=name,region=region,
                            pixels=a['pixels'],PSNR_dB=a['psnr'],patch_LPIPS=a['patch_lpips'],
                            versus_O3_SHARP_PSNR_dB=a['psnr']-b['psnr'],versus_O3_SHARP_LPIPS=a['patch_lpips']-b['patch_lpips'],
                            versus_O3_DA3_PSNR_dB=a['psnr']-old['psnr'],versus_SHARP_full_PSNR_dB=a['psnr']-full['psnr']))
                for key,left,right in extra_pairs:
                    for region in c['regions']:
                        a,b=t[left][region],t[right][region]
                        contrasts.append(dict(scale_label=label,reference=ref,nominal_focus_m=focus,comparison=key,left=left,right=right,region=region,
                            PSNR_gain_dB=a['psnr']-b['psnr'],LPIPS_change=a['patch_lpips']-b['patch_lpips'],descriptive_not_new_formal_gate=True))
            for key,region_decisions in ev['decisions'][ref].items():
                concise[label][ref][key]={reg:{focus:{k:v for k,v in result.items() if k!='groups'} for focus,result in group.items()} for reg,group in region_decisions.items()}
    write_csv(RECORDS/(PREFIX+'-report-tables.csv'),tables);write_csv(RECORDS/(PREFIX+'-descriptive-contrasts.csv'),contrasts)
    names=['O3_DA3','VG_DA3','VG_SHARP','SHARP_front','SHARP_full','SHARP_front_L3_E1','SHARP_front_L3_E2','SHARP_full_L3_E1','SHARP_full_L3_E2']
    fig,axes=plt.subplots(2,3,figsize=(16,9))
    for row,focus in enumerate(['1.4','5.0']):
        t=evaluations['native']['pooled']['Cycles'][focus]
        for ax,region in zip(axes[row],['full_image','in_focus','band']):
            values=[t[name][region]['psnr']-t['O3_SHARP'][region]['psnr'] for name in names]
            ax.bar(np.arange(len(names)),values);ax.axhline(0,color='black',linewidth=.8)
            ax.axhline(-.2 if region!='band' else .5,color='red',linestyle='--',linewidth=.8)
            ax.set_xticks(np.arange(len(names)),names,rotation=70,fontsize=7);ax.set_ylabel('PSNR relative to O3 SHARP-depth (dB)');ax.set_title(f'Nominal focus {focus} m; {region}')
    fig.tight_layout();overview=OUT/'native-Cycles-overview.png';fig.savefig(overview,dpi=140);plt.close(fig)
    timings={name:[row['seconds'][name] for row in rendered['rows']] for name in rendered['rows'][0]['seconds']}
    depth=zero_result['depth_errors'];back_native=[row for row in diagnostic['back_layer'] if row['scale_label']=='native']
    back_summary=[]
    for frame in [1,30,60]:
        selected=[row for row in back_native if row['id'].startswith(f'B2-high-f{frame:03d}')];first=selected[0]
        back_summary.append(dict(frame=frame,band_fraction_range=[min(x['fraction_all_back_in_band'] for x in selected),max(x['fraction_all_back_in_band'] for x in selected)],
            paired_depth_gap_p10_p50_p90=first['paired_depth_gap_p10_p50_p90'],paired_nonpositive_fraction=first['paired_nonpositive_depth_gap_fraction'],
            significant_nearer_fraction=first['paired_back_at_least_2pct_or_5mm_nearer_fraction'],GT_hidden_supported_back_Gaussians=first['known_hidden_supported_back_Gaussians'],
            GT_hidden_match_fraction=first['hidden_match_fraction_of_supported']))
    def env_version(python):
        return json.loads(subprocess.check_output([python,'-c','import json,sys,importlib.metadata as m;print(json.dumps({"Python":sys.version.split()[0],"torch":m.version("torch")}))'],text=True))
    summary=dict(status='completed_pause_for_user',reported_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),criteria=dict(path=str(CRITERIA.relative_to(ROOT)),commit='627ee78',unchanged=True),
        code=dict(official_revision=license_record['code']['revision'],inference_adapter_commit='da4d89d',render_commit='de181ac',score_commit='e775e55'),
        output=str(OUT.relative_to(ROOT)),scope=c['scope'],license_record=str((RECORDS/(PREFIX+'-model-licenses.json')).relative_to(ROOT)),
        weights_sha256=license_record['weights']['sha256'],research_use_allowed=True,commercial_or_product_use_allowed=False,
        inference=dict(model_loading_seconds=infer['model_loading_seconds'],official_predict_and_unproject_total_seconds=sum(x['seconds_official_predict_and_unproject'] for x in infer['rows']),
            peak_allocated_MiB=max(x['peak_allocated_MiB'] for x in infer['rows']),inference_resolution=[1536,1536],Gaussians_per_frame=1179648,Gaussians_per_output_layer=589824,strict_load=True),
        zero_aperture=dict(metrics=zero_result['metrics'],official_export_checks=[dict(frame=x['frame'],**x['official_export_agreement']) for x in infer['rows'] if x['official_export_agreement']]),
        depth_errors=depth,single_global_Oracle_scale=prep['scale'],target_focus_plan=str((RECORDS/(PREFIX+'-preparation.json')).relative_to(ROOT)),
        primary_native_decisions=concise['native'],Oracle_scale_diagnostic_decisions=concise['oracle_scale'],
        pooled_scores={label:ev['pooled'] for label,ev in evaluations.items()},
        tables=dict(all_focus_regions=str((RECORDS/(PREFIX+'-report-tables.csv')).relative_to(ROOT)),descriptive_contrasts=str((RECORDS/(PREFIX+'-descriptive-contrasts.csv')).relative_to(ROOT))),
        diagnostics=dict(record=str((RECORDS/(PREFIX+'-diagnostics.json')).relative_to(ROOT)),native_pooled_cracks=diagnostic['pooled_cracks']['native'],
            native_flypoints=[x for x in diagnostic['flypoints'] if x['scale_label']=='native'],native_second_layer=back_summary,
            caveats=['T>0.1 is transparency-support proxy; block0 alone may intentionally rely on block1, not certified geometric crack GT',
                'intermediate-depth near-GT-edge tags can include depth bias and slanted surfaces; raw counts compare different Gaussian densities; K1 contributions are not K64 DOF fractions',
                'the network output layers are jointly predicted; replacing block1 removes visible-reconstruction contributions as well as any hidden content']),
        visualization=dict(overview=str(overview),zero=zero_result['visualizations'],native=evaluations['native']['visualizations'],Oracle=evaluations['oracle_scale']['visualizations'],diagnostics=diagnostic['visualizations']),
        runtime=dict(synchronized_render_seconds_by_method={name:dict(total=float(sum(v)),median=float(np.median(v)),minimum=float(min(v)),maximum=float(max(v))) for name,v in timings.items()},
            render_peak_allocated_MiB=max(x['peak_allocated_MiB'] for x in rendered['rows']),queue_log='logs/route3-sharp-s1-2026-11-11/queue.log',
            GPU_stages_serial=True,diagnostic_GPU_overlap_with_quality_CPU_only=True,not_a_new_timing_measurement_for_cached_R3=True),
        software=dict(route3=license_record['environment'],O3_env='/root/autodl-tmp/envs/dofgaussian',O3=env_version('/root/autodl-tmp/envs/dofgaussian/bin/python'),
            LaMa_env='/root/miniconda3/envs/magicbokeh',LaMa=env_version('/root/miniconda3/envs/magicbokeh/bin/python'),GPU='RTX4090D',GPU_total_MiB=49140,driver='580.105.08'),
        issues_resolved=['LaMa first launch in dofgaussian lacked easydict; reused original magicbokeh LaMa environment without package/config changes; empty failed output and log preserved',
            'moved gsplat import to render function so O3 adapter runs in its original environment without installing gsplat there'],
        observations=['Native SHARP front does not pass either focus versus same-depth O3 or DA3-depth O3 under Cycles or LDR criteria.',
            'The full two-layer learned representation improves several VG metrics, but its full-image quality remains below corresponding O3; front-only and full behave differently.',
            'Video retrieval does not stably outperform SHARP full under the frozen comparisons. Added retrieval has small near-focus gains and far-focus degradation in native settings.',
            'Depth-source and representation contrasts are both reported; global Oracle scale does not replace native verdicts.',
            'This component diagnosis does not invalidate R2 GT-geometry upper-bound results or establish that all learned explicit GS designs fail.'],
        limits=c['limits'],old_results_and_decisions_changed=False,SHARP_modified_or_finetuned=False,network_training=False,new_Blender_render=False,U1_video_resumed=False,
        resources_after={p:dict(free_bytes=shutil.disk_usage(p).free) for p in ['/','/root/autodl-tmp']},
        next_step='Pause for user decision. No basis from S1 for treating SHARP-front as a successful O3 replacement or claiming video hidden retrieval beats the network block; retain 2.5D visible +3D hidden comparison and diagnose representation/depth/retrieval if separately authorized.',
        validation_record='03_实验记录/records/'+PREFIX+'-validation.json')
    summary['interpretation_limits']=[
        '同一深度来源不等于逐高斯几何完全相同；O3/V-G 使用第一层光栅化期望深度，SHARP 保留各高斯的分布和学习属性。',
        'V-G 的深度来源对照也随规则改变 CoC、输入侧环带和填充，不把它当作唯一独立因素的因果估计。',
        'SHARP 两个输出块联合承担可见重建；截取 block0 的结果不能单独证明所有学习式显式 GS 可见层无效。',
        '第二层 GT 隐藏深度一致性受原生深度偏差和几何匹配阈值影响，不等于网络真实隐藏内容的整体准确率。']
    summary['software']['metrics']=dict(package='lpips0.1.4',network='alex',metric_version='0.1',patch_size=64,patch_context='whole64px grid, regional pixel support weights')
    summary['runtime']['timing_boundaries']='inference includes official1536 preprocessing, network,CPU covariance unprojection; excludes checkpoint loading. Render times include K64 projection/composition/fill lookup, exclude inference and NPZ I/O; O3 timing record includes NPZ write, not used for speed ratios.'
    with (RECORDS/(PREFIX+'-summary.json')).open('x') as f:f.write(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print('SHARP_REPORT',len(tables),'pooled regional scores',len(contrasts),'descriptive contrasts',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['zero','quality','diagnostics','report'],required=True);a=p.parse_args()
    c=json.loads(CRITERIA.read_text());assert c['frozen'];torch.set_num_threads(8);cv2.setNumThreads(1)
    globals()[a.mode](c)


if __name__=='__main__':main()
