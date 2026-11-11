"""R3 target-region focus: keep U2 geometry and reevaluate both references.

GT in-focus pixels are allowed only to select the focus scalar. Rendering
uses cached DA3 geometry, input-only retrieval/fill and a derived depth band.
"""
import argparse
import csv
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from src.dual_layer import depth_band
from src.dual_layer_retrieval import retrieve
from route3_upper_bound_evaluate import BASE,OUT as UPPER,RECORDS,items,masks_for,score,pool,display,exr
from route3_c1_bokehme_oracle import lpips_cpu
OUT=ROOT/'outputs/route3-target-focus-2026-11-10-01'
LDR=ROOT/'outputs/route3-ldr-u1r-2026-11-10-01'
PREFIX='route3-target-focus-2026-11-10'
CRITERIA=RECORDS/'route3-target-focus-criteria-2026-11-10.json'
REGIONS=['full_image','in_focus','out_of_focus','band','recoverable']
FRAMES=[1,6,11,16,21,26,30,35,40,45,50,55,60]


def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def method_band(depth,fx,focus,aperture):return depth_band(depth,fx,focus,aperture,1.5,.03)[0]


def target_mask(case):
    # Exactly the original in-focus indicator; do not read hidden GT here.
    gt=np.load(case['input_npz'])
    radius=abs(gt['K'][0,0]*case['aperture_radius_m']*(1/gt['center_depth']-1/case['focus_distance_m']))
    return gt['center_surface_valid']&(radius<=1)


def prepare():
    start=time.monotonic();dest=OUT/'parameters';dest.mkdir(exist_ok=False);rows=[];audit=[]
    original=json.loads((UPPER/'U2-inputs.json').read_text())['items'];oldc=json.loads((RECORDS/'route3-dual-layer-criteria-L2-2026-11-11.json').read_text())
    for inp in original:
        d=np.load(inp['input_npz']);z=d['center_depth'];K=d['K'];old_union=np.zeros(z.shape,bool);new_union=np.zeros(z.shape,bool);group=[]
        for focus in [1.4,5.]:
            for stop in [.5,1.4,2.]:old_union|=method_band(z,K[0,0],focus,.05/(2*stop))
        for case in [x for x in items() if x['frame_id']==inp['frame']]:
            # The only new privileged input: target-region selection, not a
            # GT depth value, scale fit, output band or geometry correction.
            gt_focus_mask=target_mask(case);valid=gt_focus_mask&np.isfinite(z)&(z>0)
            if not valid.any():raise RuntimeError('R3_STOP empty focus selection: '+case['id'])
            focus=float(np.median(z[valid]));band=method_band(z,K[0,0],focus,case['aperture_radius_m']);new_union|=band
            path=dest/(case['id']+'-'+inp['variant']+'.npz');np.savez_compressed(path,estimated_band=band,render_focus_m=focus)
            row=dict(id=case['id'],variant=inp['variant'],frame=inp['frame'],fstop=case['fstop'],nominal_focus_m=case['focus_distance_m'],
                render_focus_m=focus,focus_mask_pixels=int(gt_focus_mask.sum()),focus_valid_pixels=int(valid.sum()),
                estimated_focus_region_depth_p10_p50_p90=np.quantile(z[valid],[.1,.5,.9]).tolist(),
                parameters=str(path),input_npz=inp['input_npz'],parents=inp['parents'],fill_id=inp['fill_id'],
                band_pixels=int(band.sum()),band_outside_original_retrieval_union=int((band&~old_union).sum()),
                band_outside_original_fill_mask=int((band&~d['fill_mask']).sum()))
            group.append(row)
        missing=int((new_union&~old_union).sum());regenerated=False
        if missing:
            # Per-pixel aggregation is unchanged; enlarge only the cache domain.
            timing=json.loads((BASE/'inputs/B2-high/pinhole/timings.json').read_text());byframe={r['frame']:r for r in timing};views=[]
            for frame in FRAMES:
                source=np.load(UPPER/'U2-da3'/inp['variant']/f'f{frame:03d}.npz')
                color=np.clip(exr(BASE/'inputs/B2-high/pinhole'/(byframe[frame]['key']+'.exr')),0,1)
                views.append(dict(frame=frame,depth=source['depth'],K=source['K'],c2w=source['c2w'],rgb_linear=color))
            parent,details=retrieve(z,K,d['camera_c2w'],new_union|old_union,views,inp['frame'])
            parent_dir=OUT/'expanded-parents'/inp['id'];parent_dir.mkdir(parents=True,exist_ok=False)
            parent_path=parent_dir/'parents.npz';np.savez_compressed(parent_path,**parent)
            for row in group:row['parents']=str(parent_path)
            regenerated=True
        rows.extend(group);audit.append(dict(id=inp['id'],new_band_missing_cache_pixels=missing,parent_cache_regenerated=regenerated,
            original_fill_reused=True,original_background_depth_reused=True));print('R3_FOCUS',inp['id'],[(x['nominal_focus_m'],x['fstop'],x['render_focus_m']) for x in group],flush=True)
    for case in items():
        selected=[r for r in rows if r['id']==case['id']];assert len(selected)==2
        assert selected[0]['render_focus_m']==selected[1]['render_focus_m']
    record=dict(completed=True,rows=rows,cache_audit=audit,GT_used_only_for_focus_region=True,
        GT_depth_values_not_fitted=True,global_scale_or_pose_changed=False,method_input_Gaussians_not_optimized=True,
        score_masks_remain_nominal=True,prepare_seconds=time.monotonic()-start)
    write(OUT/'focus-plan.json',record);write(RECORDS/(PREFIX+'-focus.json'),record)


def bokehme():
    import torch
    from src.baselines.bokehme import setup,predict
    torch.set_num_threads(4);pipeline,settings,nets,_=setup(ROOT.parent/'BokehMe',ROOT/'models/BokehMe',dict(gamma=2.2,highlight=False,defocus_scale=10.))
    dest=OUT/'O3';dest.mkdir(exist_ok=False);rows=[];runtime=[]
    for row in json.loads((OUT/'focus-plan.json').read_text())['rows']:
        if row['variant']!='E1':continue
        d=np.load(row['input_npz']);control=(d['K'][0,0]*.05/(2*row['fstop'])*(1/d['center_depth']-1/row['render_focus_m'])/10).astype(np.float32)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.monotonic()
        pred,_=predict(pipeline,settings,nets,np.ascontiguousarray(d['rgb_srgb']),np.ascontiguousarray(control))
        torch.cuda.synchronize();seconds=time.monotonic()-start
        path=dest/(row['id']+'.npz');np.savez_compressed(path,prediction_srgb=pred.clip(0,1).astype(np.float32),render_focus_m=row['render_focus_m'])
        for variant in ['E1','E2']:rows.append(dict(id=row['id'],method='O3_'+variant,path=str(path)))
        runtime.append(dict(id=row['id'],seconds=seconds,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20));print('R3_O3',row['id'],flush=True)
    write(OUT/'O3-predictions.json',dict(rows=rows));write(RECORDS/(PREFIX+'-O3.json'),dict(completed=True,rows=runtime,
        same_depth_same_focus_same_O3_for_E1_E2=True,CoC_calibration_changed=False))


def render():
    import torch
    from src.layered_aperture import pixel_layer,merge_layers
    from src.analytic_aperture import sampled
    from route3_b2_oracle_3d import sparse_pixel_layer
    torch.set_num_threads(4);dest=OUT/'3d';dest.mkdir(exist_ok=False);rows=[];runtime=[]
    profile=json.loads((RECORDS/'route3-dual-layer-criteria-L2-2026-11-11.json').read_text())['profile']
    planned=json.loads((OUT/'focus-plan.json').read_text())['rows'];groups={}
    for row in planned:groups.setdefault((row['variant'],row['frame']),[]).append(row)
    for (variant,frame),cases in groups.items():
        d=np.load(cases[0]['input_npz']);K=torch.tensor(d['K'],device='cuda',dtype=torch.float32);cam=torch.tensor(d['camera_c2w'],device='cuda',dtype=torch.float32);E=torch.linalg.inv(cam)
        visible=pixel_layer(torch.tensor(d['visible_rgb_linear'],device='cuda'),torch.tensor(d['center_depth'],device='cuda'),K,cam)
        fill=torch.tensor(np.load(UPPER/'U2-fill'/cases[0]['fill_id']/'fill.npz')['fill_rgb_linear'],device='cuda');bg=torch.tensor(d['background_depth'],device='cuda')
        parent=np.load(cases[0]['parents'])
        for case in cases:
            band=np.load(case['parameters'])['estimated_band'];uv=parent['uv'];x=np.clip(uv[:,0].astype(int),0,511);y=np.clip(uv[:,1].astype(int),0,511)
            chosen=band[y,x];hidden=sparse_pixel_layer(parent,chosen,K,cam,profile,'cuda');values={};times={};torch.cuda.reset_peak_memory_stats()
            for kind,cloud in [('O5',visible),('joint',merge_layers(visible,hidden))]:
                with torch.inference_mode():
                    torch.cuda.synchronize();start=time.monotonic();result=sampled(cloud,K,E,512,512,case['render_focus_m'],.05/(2*case['fstop']),64,fill,bg,far_plane=1e5)
                    torch.cuda.synchronize();times[kind]=time.monotonic()-start
                values[kind]=result['rgb_linear'].cpu().numpy()
            combined=np.where(band[...,None],values['joint'],values['O5'])
            for label,raw in [('O5',values['O5']),('ours',combined)]:
                path=dest/(case['id']+'-'+label+'_'+variant+'.npz');np.savez_compressed(path,prediction_srgb=display(raw),rgb_linear=raw,estimated_band=band,render_focus_m=case['render_focus_m'])
                rows.append(dict(id=case['id'],method=label+'_'+variant,path=str(path)))
            o3=np.load(OUT/'O3'/(case['id']+'.npz'))['prediction_srgb'];ours=display(combined)
            path=dest/(case['id']+'-descriptive_'+variant+'.npz');np.savez_compressed(path,prediction_srgb=np.where(band[...,None],ours,o3))
            rows.append(dict(id=case['id'],method='descriptive_'+variant,path=str(path)))
            runtime.append(dict(id=case['id'],variant=variant,focus_m=case['render_focus_m'],seconds=times,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,hidden_parents=int(chosen.sum())))
            print('R3_RENDER',variant,case['id'],flush=True);del hidden
        del visible;torch.cuda.empty_cache()
    rows+=json.loads((OUT/'O3-predictions.json').read_text())['rows'];write(OUT/'predictions.json',dict(rows=rows))
    write(RECORDS/(PREFIX+'-render.json'),dict(completed=True,rows=runtime,GT_focus_mask_used_only_as_scalar=True,GT_evaluation_masks_used_in_band_or_fill=False,
        K=64,old_fill_and_background_depth_reused=True,old_gaussian_profile_and_colors_unchanged=True))


def comparisons(pooled,oldpool,gtpool,cases,metrics,old,c):
    results={};gates={};recovery={}
    for reference,tables in pooled.items():
        results[reference]={};gates[reference]={};recovery[reference]={}
        for variant in ['E1','E2']:
            results[reference][variant]={};gates[reference][variant]={};recovery[reference][variant]={}
            for focus in [1.4,5.]:
                key=str(focus);t=tables[key];base=gtpool[reference][key];previous=oldpool[reference][key];degradation={};direct={};changes={}
                for reg in REGIONS:
                    a,b=t['ours_'+variant][reg],t['O3_'+variant][reg];ga,gb=base['L3'][reg],base['O3'][reg]
                    da=a['psnr']-ga['psnr'];db=b['psnr']-gb['psnr']
                    degradation[reg]=dict(Delta_ours_PSNR_dB=da,Delta_O3_PSNR_dB=db,advantage_dB=da-db,
                        Delta_ours_LPIPS=a['patch_lpips']-ga['patch_lpips'],Delta_O3_LPIPS=b['patch_lpips']-gb['patch_lpips'])
                    direct[reg]=dict(PSNR_gain_dB=a['psnr']-b['psnr'],LPIPS_change=a['patch_lpips']-b['patch_lpips'])
                for method in ['O3','O5','ours','descriptive']:
                    name=method+'_'+variant;changes[method]={}
                    for reg in REGIONS:
                        a,b=t[name][reg],previous['U2_'+name][reg]
                        changes[method][reg]=dict(PSNR_recovered_dB=a['psnr']-b['psnr'],LPIPS_change=a['patch_lpips']-b['patch_lpips'])
                groups=[];same=0;valid=0
                for x in cases:
                    if x['focus_distance_m']!=focus:continue
                    m=metrics[reference][x['id']];g=old[reference][x['id']]
                    a,b=m['ours_'+variant]['band'],m['O3_'+variant]['band']
                    advantage=(a['psnr']-g['L3']['band']['psnr'])-(b['psnr']-g['O3']['band']['psnr'])
                    enough=a['pixels']>=1000;valid+=int(enough);same+=int(enough and advantage>0)
                    groups.append(dict(id=x['id'],pixels=a['pixels'],degradation_advantage_dB=advantage))
                guard=all(direct[reg]['PSNR_gain_dB']>=-.2 for reg in ['full_image','in_focus'])
                passed=degradation['band']['advantage_dB']>=.3 and valid==6 and same>=5 and guard
                results[reference][variant][key]=dict(degradation=degradation,direct=direct,change_from_U2=changes)
                gates[reference][variant][key]=dict(passed=passed,valid_groups=valid,same_direction=same,guardrails_passed=guard,
                    band_advantage_dB=degradation['band']['advantage_dB'],groups=groups,primary=reference=='Cycles')
                recovery[reference][variant][key]={m:x['in_focus'] for m,x in changes.items()}
    return results,gates,recovery


def evaluate(c):
    import torch
    torch.set_num_threads(8)
    network=lpips_cpu(SimpleNamespace(alexnet_weights=ROOT.parent/'.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth'))
    old=json.loads((RECORDS/'route3-ldr-u1r-2026-11-10-R1-evaluation.json').read_text());refs=json.loads((LDR/'references.json').read_text())
    refmap={x['id']:x['path'] for x in refs['rows']};planned=json.loads((OUT/'focus-plan.json').read_text())['rows']
    bandspec=json.loads((RECORDS/'route3-dual-layer-criteria-L2-2026-11-11.json').read_text());lookup={x['id']:x for x in items()}
    predictions=json.loads((OUT/'predictions.json').read_text())['rows'];metrics={'Cycles':{},'LDR':{}};flat=[];plots=[]
    for row in predictions:
        x=lookup[row['id']];masks=masks_for(x,bandspec);masks={k:masks[k] for k in REGIONS};pred=np.load(row['path'])['prediction_srgb']
        assert np.isfinite(pred).all() and pred.shape==(512,512,3)
        for reference,target in [('Cycles',display(exr(x['gt_rgb_exr']))),('LDR',np.load(refmap[x['id']])['reference_srgb'])]:
            values=score(pred,target,masks,network);metrics[reference].setdefault(x['id'],{})[row['method']]=values
            for region,v in values.items():
                previous=old['per_case_metrics'][reference][x['id']]['U2_'+row['method']][region];assert v['pixels']==previous['pixels']
                flat.append(dict(reference=reference,id=x['id'],frame=x['frame_id'],fstop=x['fstop'],nominal_focus_m=x['focus_distance_m'],method=row['method'],region=region,
                    PSNR=v['psnr'],patch_LPIPS=v['patch_lpips'],U2_PSNR=previous['psnr'],U2_LPIPS=previous['patch_lpips'],
                    PSNR_change=v['psnr']-previous['psnr'],LPIPS_change=v['patch_lpips']-previous['patch_lpips'],pixels=v['pixels']))
        print('R3_SCORE',row['id'],row['method'],metrics['Cycles'][x['id']][row['method']]['in_focus']['psnr'],flush=True)
    pooled={};apertures={}
    for reference,cases in metrics.items():
        pooled[reference]={};apertures[reference]={}
        for focus in [1.4,5.]:
            ids=[x['id'] for x in items() if x['focus_distance_m']==focus];methods=sorted(cases[ids[0]])
            pooled[reference][str(focus)]={m:{reg:pool([cases[i][m][reg] for i in ids]) for reg in REGIONS} for m in methods}
            apertures[reference][str(focus)]={}
            for stop in [1.4,2.]:
                group=[x['id'] for x in items() if x['focus_distance_m']==focus and x['fstop']==stop]
                apertures[reference][str(focus)][str(stop)]={m:{reg:pool([cases[i][m][reg] for i in group]) for reg in REGIONS} for m in methods}
    contrast,gates,recovery=comparisons(pooled,old['pooled'],old['pooled'],items(),metrics,old['per_case_metrics'],c)
    dest=OUT/'evaluation';dest.mkdir(exist_ok=False)
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    for row in planned:
        x=lookup[row['id']];v=row['variant'];new={p['method']:p['path'] for p in predictions if p['id']==x['id']}
        source={p['method']:p['path'] for p in json.loads((UPPER/'U2-predictions.json').read_text())['rows'] if p['id']==x['id']}
        images=[display(exr(x['gt_rgb_exr'])),np.load(source['O3_'+v])['prediction_srgb'],np.load(new['O3_'+v])['prediction_srgb'],
                np.load(refmap[x['id']])['reference_srgb'],np.load(source['ours_'+v])['prediction_srgb'],np.load(new['ours_'+v])['prediction_srgb']]
        fig,axes=plt.subplots(2,3,figsize=(12,8))
        for ax,img,label in zip(axes.flat,images,['Cycles','U2 O3','R3 O3','LDRconsistent','U2 ours','R3 ours']):ax.imshow(img);ax.set_title(label);ax.axis('off')
        fig.suptitle(x['id']+' '+v+f' renderF={row["render_focus_m"]:.4f}m');fig.tight_layout();path=dest/(x['id']+'-'+v+'.png');fig.savefig(path,dpi=120);plt.close(fig);plots.append(str(path))
    report=dict(completed=True,criteria=str(CRITERIA),primary_reference='Cycles',auxiliary_reference='LDR',per_case_metrics=metrics,pooled=pooled,per_aperture=apertures,
        original_U2_and_GT_pooled=old['pooled'],comparisons=contrast,decisions=gates,focal_recovery=recovery,visualizations=plots,
        masks_unchanged=True,GT_mask_used_only_to_select_focus=True,old_U2_decisions_unchanged=True)
    write(RECORDS/(PREFIX+'-evaluation.json'),report)
    with (RECORDS/(PREFIX+'-evaluation.csv')).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]),lineterminator='\n');writer.writeheader();writer.writerows(flat)


def report():
    """Summarize existing scores; never rerender or alter frozen decisions."""
    import shutil
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    evaluation=json.loads((RECORDS/(PREFIX+'-evaluation.json')).read_text())
    focus=json.loads((OUT/'focus-plan.json').read_text())
    diagnostic=json.loads((RECORDS/(PREFIX+'-CoC-diagnostic.json')).read_text())
    o3=json.loads((RECORDS/(PREFIX+'-O3.json')).read_text())
    rendered=json.loads((RECORDS/(PREFIX+'-render.json')).read_text())
    old_summary=json.loads((RECORDS/'route3-upper-bound-2026-11-11-summary.json').read_text())
    flat=[];figures=[];focus_stats={};coc_stats={}
    for reference,tables in evaluation['pooled'].items():
        fig,axes=plt.subplots(2,2,figsize=(11,8))
        for vi,variant in enumerate(['E1','E2']):
            for fi,nominal in enumerate(['1.4','5.0']):
                t=tables[nominal];old=evaluation['original_U2_and_GT_pooled'][reference][nominal]
                ax=axes[vi,fi];names=['O3','O5','ours'];x=np.arange(3)
                for label,offset,values in [
                    ('Original U2',-.16,[old['U2_'+m+'_'+variant]['in_focus']['psnr'] for m in names]),
                    ('R3 target focus',0,[t[m+'_'+variant]['in_focus']['psnr'] for m in names]),
                    ('GT geometry baseline',.16,[old['L3' if m=='ours' else m]['in_focus']['psnr'] for m in names])]:
                    ax.plot(x+offset,values,'o',label=label)
                ax.set_xticks(x,names);ax.set_ylabel('In-focus pooled PSNR (dB)')
                ax.set_title(f'{variant}, nominal focus {nominal} m');ax.grid(alpha=.2)
                if vi==fi==0:ax.legend(fontsize=8)
        fig.suptitle(reference+' reference: original U2 vs target focus');fig.tight_layout()
        path=OUT/('focal-recovery-'+reference+'.png');fig.savefig(path,dpi=140);plt.close(fig);figures.append(str(path))
        for variant in ['E1','E2']:
            for nominal,t in tables.items():
                old=evaluation['original_U2_and_GT_pooled'][reference][nominal]
                for method in ['O3','O5','ours','descriptive']:
                    for region in REGIONS:
                        a=t[method+'_'+variant][region];b=old['U2_'+method+'_'+variant][region]
                        baseline=old['L3' if method=='ours' else 'descriptive_O3_outside_L2_inside' if method=='descriptive' else method][region]
                        flat.append(dict(reference=reference,variant=variant,nominal_focus_m=nominal,method=method,region=region,
                            pixels=a['pixels'],U2_PSNR_dB=b['psnr'],R3_PSNR_dB=a['psnr'],PSNR_change_dB=a['psnr']-b['psnr'],
                            U2_patch_LPIPS=b['patch_lpips'],R3_patch_LPIPS=a['patch_lpips'],LPIPS_change=a['patch_lpips']-b['patch_lpips'],
                            GT_geometry_PSNR_dB=baseline['psnr'],GT_geometry_patch_LPIPS=baseline['patch_lpips'],
                            U2_degradation_PSNR_dB=b['psnr']-baseline['psnr'],R3_degradation_PSNR_dB=a['psnr']-baseline['psnr'],
                            descriptive_only=method=='descriptive'))
    table=RECORDS/(PREFIX+'-region-tables.csv')
    with table.open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]),lineterminator='\n');writer.writeheader();writer.writerows(flat)
    selected=[r for r in focus['rows'] if r['variant']=='E1'];fig,axes=plt.subplots(3,4,figsize=(13,10));lookup={x['id']:x for x in items()}
    for ax,row in zip(axes.flat,selected):
        d=np.load(row['input_npz']);ax.imshow(d['rgb_srgb'])
        mask=target_mask(lookup[row['id']]);overlay=np.zeros((*mask.shape,4));overlay[mask]=[1.,.15,.1,.25]
        ax.imshow(overlay);ax.set_title(f'f{row["frame"]} N{row["fstop"]}: {row["nominal_focus_m"]:g} -> {row["render_focus_m"]:.3f} m',fontsize=9);ax.axis('off')
    fig.suptitle('Original GT in-focus target selection (red); shared E1/E2 focus');fig.tight_layout()
    path=OUT/'focus-selection.png';fig.savefig(path,dpi=120);plt.close(fig);figures.append(str(path))
    for nominal in [1.4,5.]:
        rows=[r for r in selected if r['nominal_focus_m']==nominal]
        focus_stats[str(nominal)]=dict(min_m=min(r['render_focus_m'] for r in rows),max_m=max(r['render_focus_m'] for r in rows))
        rows=[r for r in diagnostic['rows'] if r['nominal_focus_m']==nominal];pixels=sum(r['pixels'] for r in rows)
        coc_stats[str(nominal)]=dict(pixels=pixels,
            estimated_CoC_le1_fraction_before=sum(r['pixels']*r['fraction_estimated_CoC_le1_before'] for r in rows)/pixels,
            estimated_CoC_le1_fraction_after=sum(r['pixels']*r['fraction_estimated_CoC_le1_after'] for r in rows)/pixels,
            local_depth_ratio_median_range=[min(r['DA3_to_GT_depth_ratio_p10_p50_p90'][1] for r in rows),max(r['DA3_to_GT_depth_ratio_p10_p50_p90'][1] for r in rows)])
    runtime=dict(O3_prediction_seconds=sum(r['seconds'] for r in o3['rows']),
        O3_peak_allocated_MiB=max(r['peak_allocated_MiB'] for r in o3['rows']),
        sampled_render_seconds=sum(sum(r['seconds'].values()) for r in rendered['rows']),
        sampled_render_peak_allocated_MiB=max(r['peak_allocated_MiB'] for r in rendered['rows']),
        queue_UTC=dict(bokehme=['2026-11-11T03:28:47Z','2026-11-11T03:28:55Z'],render=['2026-11-11T03:28:55Z','2026-11-11T03:29:57Z'],evaluate=['2026-11-11T03:29:57Z','2026-11-11T03:31:34Z']),
        timing_note='Prediction CUDA synchronization timings exclude loading and I/O; queue wall times include them. Two GPU stages ran sequentially; LPIPS scored on CPU.')
    resources={path:dict(total_bytes=shutil.disk_usage(path).total,free_bytes=shutil.disk_usage(path).free) for path in ['/','/root/autodl-tmp']}
    summary=dict(status='completed_pause_for_user',reported_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        criteria=dict(path=str(CRITERIA.relative_to(ROOT)),commit='be99e50',implementation_commit='d1d64d8'),
        basis=dict(U2_criteria='11c78ca',U2_summary='5933ccc',R1_R2_summary='84f77ad'),output=str(OUT.relative_to(ROOT)),
        scope=dict(scene='B2-high',frames=[1,30,60],nominal_focuses_m=[1.4,5.],fstops=[1.4,2.],groups_per_focus=6,variants=['E1','E2']),
        focus_selection=dict(formula='median of unchanged DA3 estimated depth inside original GT in-focus mask',ranges=focus_stats,
            plan_record=str((RECORDS/(PREFIX+'-focus.json')).relative_to(ROOT)),E1_E2_and_all_methods_identical=True,GT_mask_is_user_target_proxy_only=True),
        decisions=evaluation['decisions'],comparisons=evaluation['comparisons'],focal_recovery=evaluation['focal_recovery'],
        original_U2_decisions=old_summary['U2']['decisions'],old_U2_decisions_changed=False,
        cache_audit=focus['cache_audit'],CoC_diagnostic=dict(descriptive=True,weighted=coc_stats,record=str((RECORDS/(PREFIX+'-CoC-diagnostic.json')).relative_to(ROOT))),
        tables=dict(pooled_focus_regions=str(table.relative_to(ROOT)),per_case_regions=str((RECORDS/(PREFIX+'-evaluation.csv')).relative_to(ROOT)),
            all_scores_and_per_aperture=str((RECORDS/(PREFIX+'-evaluation.json')).relative_to(ROOT))),
        visualizations=figures+evaluation['visualizations'],runtime=runtime,resources_after=resources,
        software=dict(GPU='RTX 4090 D',GPU_index=0,GPU_total_MiB=49140,driver='580.105.08',renderer_env='/root/autodl-tmp/envs/route3',python='3.10.22',torch='2.7.1+cu128',gsplat='1.5.0',LPIPS_package='0.1.4',LPIPS_model='alex metric 0.1',patch=64,
            O3_env='/root/autodl-tmp/envs/dofgaussian',O3_python='3.9.25',O3_torch='2.1.2+cu121',O3_gamma=2.2,O3_defocus_scale=10,physical_focal_length_m=.05,K=64),
        observations=[
            'Target focus improves O3 focal PSNR by about 1.69/1.01 dB and focal LPIPS. It does not recover all previous 9-10 dB focal loss.',
            'Ours focal PSNR decreases about 0.76/0.72 dB near-focus (E1/E2), improves about 0.83 dB far-focus, while focal LPIPS improves or is nearly unchanged.',
            'Near-focus out-of-focus and full-image errors increase after selecting the subject. Moving a scalar focus plane does not correct spatially nonuniform depth or CoC magnitude.',
            'All E1/E2 nominal focuses fail the prospectively frozen Cycles R3 gate and auxiliary LDR gate; the prior far-focus degradation advantage shrinks substantially.',
            'Residual errors can include depth/CoC, edge representation and retrieval. This experiment does not isolate their individual causal contributions.',
            'Original U2 decisions and the positive R2 GT-geometry/true-hidden upper-bound observations remain unchanged.'
        ],limits=[
            'One controlled scene, three frames, six lens groups per focus; GT mask simulates target selection, not automatic focusing.',
            'Unoptimized V-G plus input-only cross-frame retrieval; no R2 fitted visible layer or hidden GT is used by the R3 method.',
            'Descriptive splice is not a method or formal decision; fixed original GT scoring masks and nominal Cycles/LDR references.',
            'No new Blender rendering, DA3 inference, training, global scale correction or aperture/CoC retuning.'
        ],next_step='Pause for user decision; do not start training or geometry/CoC retuning automatically.',
        validation_record=str((RECORDS/(PREFIX+'-validation.json')).relative_to(ROOT)))
    path=RECORDS/(PREFIX+'-summary.json')
    with path.open('x') as f:f.write(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print('R3_REPORT',str(path),len(flat),'pooled regional rows',len(figures),'overview plots',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--mode',choices=['prepare','bokehme','render','evaluate','report'],required=True);a=parser.parse_args()
    cv2.setNumThreads(1);criteria=json.loads(CRITERIA.read_text());assert criteria['frozen']
    if a.mode in ['bokehme','render']:
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise ValueError('Explicit GPU0')
        subprocess.run(['nvidia-smi'],check=True)
    if a.mode=='evaluate':evaluate(criteria)
    else:globals()[a.mode]()


if __name__=='__main__':main()
