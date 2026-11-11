"""R1/R2 frozen LDR-reference experiment. Historical outputs are read-only.

References and masks are read only by the scoring/analysis branches. Fitting
uses the original U1 supervision and implementation with a new stop policy.
"""
import argparse
import csv
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
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from route3_upper_bound_evaluate import BASE,OUT as UPPER,RECORDS,display,exr,items,masks_for,score,pool,full_noise
from route3_c1_bokehme_oracle import lpips_cpu
from src.c1_bokehme import region_sse
OUT=ROOT/'outputs/route3-ldr-u1r-2026-11-10-01'
PREFIX='route3-ldr-u1r-2026-11-10'
CRITERIA=RECORDS/'route3-ldr-u1r-criteria-2026-11-10.json'


def write(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def model():return lpips_cpu(SimpleNamespace(alexnet_weights=ROOT.parent/'.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth'))


def references(c):
    dest=OUT/'references';dest.mkdir(exist_ok=False);start=time.monotonic()
    plan=json.loads((UPPER/'view-plan.json').read_text());timing=json.loads((UPPER/'U0-subviews/timings.json').read_text())
    bykey={x['key']:x for x in timing};noise_index={x['key']:x for x in json.loads((UPPER/'U0-noise/timings.json').read_text())};rows=[]
    for x in items():
        jobs=plan['evaluation_groups'][x['id']];assert len(jobs)==64
        linear=sum(np.clip(exr(bykey[j['key']]['path']).astype(np.float64),0,1) for j in jobs)/64
        rgb=display(linear);path=dest/(x['id']+'.npz');np.savez_compressed(path,rgb_linear=linear.astype(np.float32),reference_srgb=rgb)
        cv2.imwrite(str(path.with_suffix('.png')),np.uint8(rgb[...,::-1]*255));row=dict(id=x['id'],path=str(path),samples=64)
        if x['id'] in c['reference_noise']['complete_pairs']:
            independent=sum(np.clip(exr(noise_index[j['key']+'-noise']['path']).astype(np.float64),0,1) for j in jobs)/64
            path=dest/(x['id']+'-independent.npz');np.savez_compressed(path,rgb_linear=independent.astype(np.float32),reference_srgb=display(independent));row['noise_path']=str(path)
        rows.append(row);print('LDR_REFERENCE',x['id'],flush=True)
    network=model();lookup={x['id']:x for x in items()};byid={x['id']:x for x in rows}
    old=json.loads((RECORDS/'route3-upper-bound-2026-11-11-HDR-diagnostic.json').read_text())['rows'][0]
    assert old['id']==c['reference']['frame1_check']['id']
    pred=np.load(byid[old['id']]['path']);previous=np.load(old['prediction'])
    difference=float(abs(pred['reference_srgb']-previous['prediction_srgb']).max())
    x=lookup[old['id']];check=score(pred['reference_srgb'],display(exr(x['gt_rgb_exr'])),masks_for(x,c),network)
    psnr_difference=abs(check['full_image']['psnr']-old['per_subview_clipped_before_mean']['full_image']['psnr'])
    lpips_difference=abs(check['full_image']['patch_lpips']-old['per_subview_clipped_before_mean']['full_image']['patch_lpips'])
    tolerance=c['reference']['frame1_check']
    passed=difference<=tolerance['max_RGB_abs_tolerance'] and psnr_difference<=tolerance['max_PSNR_abs_tolerance_dB'] and lpips_difference<=tolerance['max_LPIPS_abs_tolerance']
    if not passed:raise RuntimeError('R1_STOP frame1 reference does not match HDR diagnostic')
    noises=[]
    for row in rows:
        if 'noise_path' not in row:continue
        x=lookup[row['id']];stats=score(np.load(row['path'])['reference_srgb'],np.load(row['noise_path'])['reference_srgb'],masks_for(x,c),network)
        noises.append(dict(id=x['id'],metrics=stats))
    maximum={reg:max(n['metrics'][reg]['patch_lpips'] or 0 for n in noises) for reg in c['regions']}
    result=dict(completed=True,criteria=str(CRITERIA),rows=rows,
        frame1_check=dict(passed=passed,max_RGB_difference=difference,PSNR_difference_dB=psnr_difference,LPIPS_difference=lpips_difference,Cycles_PSNR=check['full_image']['psnr']),
        independent_noise_pairs=noises,max_noise_by_region=maximum,generation_and_check_seconds=time.monotonic()-start,
        old_outputs_modified=False,new_Blender_renders=0)
    write(OUT/'references.json',result);write(RECORDS/(PREFIX+'-references.json'),result)
    if any(n['metrics']['full_image']['psnr']<50 for n in noises):raise RuntimeError('R1_STOP reference noise PSNR<50dB')


def cached_methods(x,dest):
    two={r['id']:r for r in json.loads((RECORDS/'route3-dual-layer-2026-11-11-L2-observed-sky-render.json').read_text())['rows']}
    three={r['id']:r for r in json.loads((RECORDS/'route3-dual-layer-2026-11-11-L3-render.json').read_text())['rows']}
    t=two[x['id']]['outputs'];methods={'O3':str(BASE/'methods-2d/bokehme'/x['id']/'prediction.npz'),'O5':t['O5_shared'],'L2':t['L2'],'L3':three[x['id']]['outputs']['L3']}
    a=np.load(t['L2']);band=a['band'];pred=np.where(band[...,None],a['prediction_srgb'],np.load(methods['O3'])['prediction_srgb'])
    path=dest/(x['id']+'-descriptive.npz');np.savez_compressed(path,prediction_srgb=pred);methods['descriptive_O3_outside_L2_inside']=str(path)
    for row in json.loads((UPPER/'U2-predictions.json').read_text())['rows']:
        if row['id']==x['id']:methods['U2_'+row['method']]=row['path']
    return methods


def evaluation_masks(x,c,ldr):
    masks=masks_for(x,c);target=display(exr(x['gt_rgb_exr']))
    influenced=abs(target-ldr).max(-1)>c['high_light_influence']['threshold']
    masks['highlight_influence']=influenced;masks['outside_highlight_influence']=~influenced
    count,labels,stats,centers=cv2.connectedComponentsWithStats(influenced.astype(np.uint8),8)
    order=sorted(range(1,count),key=lambda k:int(stats[k,4]),reverse=True)[:10]
    distribution=dict(pixels=int(influenced.sum()),fraction=float(influenced.mean()),components=count-1,
        components_top10=[dict(xywh=stats[k,:4].tolist(),pixels=int(stats[k,4]),centroid_xy=centers[k].tolist()) for k in order],
        region_overlap_pixels={k:int((mask&influenced).sum()) for k,mask in masks.items() if k not in ['highlight_influence','outside_highlight_influence']})
    return target,masks,distribution


def aggregate(metrics,c):
    pooled={};apertures={}
    for ref,cases in metrics.items():
        pooled[ref]={};apertures[ref]={}
        for focus in [1.4,5.]:
            chosen=[x for x in items() if x['focus_distance_m']==focus];ids=[x['id'] for x in chosen]
            methods=set.intersection(*(set(cases[i]) for i in ids));regions=list(cases[ids[0]][next(iter(methods))])
            pooled[ref][str(focus)]={m:{r:pool([cases[i][m][r] for i in ids]) for r in regions} for m in sorted(methods)}
            apertures[ref][str(focus)]={}
            for stop in [1.4,2.]:
                ids2=[x['id'] for x in chosen if x['fstop']==stop]
                apertures[ref][str(focus)][str(stop)]={m:{r:pool([cases[i][m][r] for i in ids2]) for r in regions} for m in sorted(methods)}
    return pooled,apertures


def gate(metrics,pooled,method,c,noise,worth=False,reference='LDR'):
    result={}
    for focus in [1.4,5.]:
        cases=[x for x in items() if x['focus_distance_m']==focus];table=pooled[str(focus)];left=table[method]['band'];right=table['O3']['band']
        weights=[metrics[x['id']][method]['band']['pixels'] for x in cases]
        ns=[noise['band'] if reference=='LDR' else c['R2']['Cycles']['noise']['per_lens_band'][f'N{x["fstop"]:g}-F{focus:g}'] for x in cases]
        n=float(np.average(ns,weights=weights));gain=left['psnr']-right['psnr'];delta=left['patch_lpips']-right['patch_lpips']
        quality=gain>=.5 or (-delta>=.1*right['patch_lpips'] and -delta>3*n);secondary=gain>=-.2 and delta<=3*n
        groups=[];same=0;valid=0;ties=0
        for x,ni in zip(cases,ns):
            a,b=metrics[x['id']][method]['band'],metrics[x['id']]['O3']['band'];ok=a['pixels']>=1000
            direction=a['psnr']>b['psnr'] and a['patch_lpips']-b['patch_lpips']<-3*ni
            valid+=int(ok);same+=int(ok and direction);ties+=int(ok and abs(a['patch_lpips']-b['patch_lpips'])<=3*ni)
            groups.append(dict(id=x['id'],pixels=a['pixels'],PSNR_gain_dB=a['psnr']-b['psnr'],LPIPS_change=a['patch_lpips']-b['patch_lpips'],noise=ni,same_direction=direction))
        guards={reg:table[method][reg]['psnr']-table['O3'][reg]['psnr'] for reg in ['full_image','in_focus']}
        passed=quality and secondary and valid==6 and same>=5 and all(v>=-.2 for v in guards.values())
        fn=noise['full_image'] if reference=='LDR' else float(np.mean([full_noise(focus,x['fstop']) for x in cases]))
        full_delta=table[method]['full_image']['patch_lpips']-table['O3']['full_image']['patch_lpips']
        result[str(focus)]=dict(method=method,reference=reference,PSNR_gain_dB=gain,LPIPS_change=delta,noise=n,quality_passed=quality,secondary_passed=secondary,
            valid_groups=valid,same_direction=same,LPIPS_noise_ties=ties,guardrails_dB=guards,passed=passed,
            full_LPIPS_change=full_delta,full_noise=fn,worth_training=(passed and guards['full_image']>=.5 and full_delta<=3*fn) if worth else None,groups=groups)
    return result


def evaluate(c,stage):
    folder=OUT/(stage+'-evaluation');folder.mkdir(exist_ok=False)
    refs=json.loads((OUT/'references.json').read_text());refmap={x['id']:x for x in refs['rows']};noise=refs['max_noise_by_region'];network=model()
    methods={};metrics={'Cycles':{},'LDR':{}};distributions=[];flat=[];figures=[]
    if stage=='R1':
        cache=OUT/'R1-derived';cache.mkdir(exist_ok=False)
        methods={x['id']:cached_methods(x,cache) for x in items()}
    else:
        methods={x['id']:{} for x in items()}
        for row in json.loads((OUT/('HDR-predictions.json' if stage=='HDR' else 'R2-predictions.json')).read_text())['rows']:methods[row['id']][row['method']]=row['path']
        for x in items():
            methods[x['id']]['O3']=str(BASE/'methods-2d/bokehme'/x['id']/'prediction.npz')
            old={r['id']:r for r in json.loads((RECORDS/'route3-dual-layer-2026-11-11-L2-observed-sky-render.json').read_text())['rows']}
            methods[x['id']]['L2']=old[x['id']]['outputs']['L2']
    for x in items():
        ldr=np.load(refmap[x['id']]['path'])['reference_srgb'];target,masks,dist=evaluation_masks(x,c,ldr);distributions.append(dict(id=x['id'],**dist))
        metrics['Cycles'][x['id']]={};metrics['LDR'][x['id']]={}
        for name,path in methods[x['id']].items():
            pred=np.load(path)['prediction_srgb'];assert np.isfinite(pred).all() and pred.shape==(512,512,3)
            for reference,truth in [('Cycles',target),('LDR',ldr)]:
                value=score(pred,truth,masks,network);metrics[reference][x['id']][name]=value
                for region,v in value.items():flat.append(dict(stage=stage,reference=reference,id=x['id'],method=name,region=region,**v))
            print('LDR_SCORE',stage,x['id'],name,metrics['LDR'][x['id']][name]['band']['psnr'],flush=True)
        import matplotlib
        matplotlib.use('Agg');import matplotlib.pyplot as plt
        names=['O3','O5','L2','L3'] if stage=='R1' else (['O3','L2','HDR_input_descriptive'] if stage=='HDR' else ['O3','L2','U1','U1_clip'])
        images=[target,ldr]+[np.load(methods[x['id']][name])['prediction_srgb'] for name in names]
        fig,axes=plt.subplots(2,3,figsize=(12,8))
        for ax,img,title in zip(axes.flat,images,['Cycles','LDR consistent']+names):ax.imshow(img);ax.set_title(title);ax.axis('off')
        fig.suptitle(x['id']);fig.tight_layout();path=folder/(x['id']+'.png');fig.savefig(path,dpi=120);plt.close(fig);figures.append(str(path))
        if stage=='R1':cv2.imwrite(str(folder/(x['id']+'-highlight-mask.png')),np.uint8(masks['highlight_influence'])*255)
    pooled,apertures=aggregate(metrics,c)
    if stage=='R1':
        if any(3*noise['band']>=.1*pooled['LDR'][str(f)]['O3']['band']['patch_lpips'] for f in [1.4,5.]):
            raise RuntimeError('R1_STOP reference noise cannot resolve10%LPIPS gate')
        decisions={m:gate(metrics['LDR'],pooled['LDR'],m,c,noise) for m in ['L2','L3']}
    elif stage=='R2':
        decisions={'Cycles_U1':gate(metrics['Cycles'],pooled['Cycles'],'U1',c,noise,True,'Cycles'),
                   'LDR_U1_clip':gate(metrics['LDR'],pooled['LDR'],'U1_clip',c,noise,True,'LDR')}
    else:decisions={}
    report=dict(completed=True,stage=stage,criteria_commit='c18e323',criteria=str(CRITERIA),per_case_metrics=metrics,pooled=pooled,per_aperture=apertures,
        decisions=decisions,highlight_influence=distributions,visualizations=figures,predictions=methods,reference_noise=noise,
        historical_decisions_unchanged=True,descriptive_outputs_are_methods=False,U2_new_decisions_computed=False)
    path=RECORDS/(PREFIX+'-'+stage+'-evaluation.json');assert not path.exists();write(path,report)
    with path.with_suffix('.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]),lineterminator='\n');writer.writeheader();writer.writerows(flat)


def replay_frame1(c):
    source=json.loads((RECORDS/'route3-upper-bound-2026-11-11-U1-f001.json').read_text())
    best=-float('inf');consecutive=0;rows=[];trigger=False
    for row in source['rows']:
        value=row['metrics']['heldout']['psnr'];best=max(best,value);drop=best-value;consecutive=consecutive+1 if drop>=.5 else 0
        rows.append(dict(step=row['step'],heldout_PSNR=value,best=best,drop_dB=drop,consecutive=consecutive))
        if consecutive>=3:trigger=True;break
    assert source['rows'][-1]['step']==8000
    if trigger:raise RuntimeError('R2_STOP reused frame1 would trigger new decline rule')
    result=dict(completed=True,frame=1,reused=True,source_record='route3-upper-bound-2026-11-11-U1-f001.json',source_checkpoint=source['output'],
        checkpoint_step=8000,old_decision_unchanged=True,checkpoint_not_heldout_selected=True,new_decline_rule_triggered=False,
        replay=rows,config=source['config'],monitor_rows=source['rows'])
    write(RECORDS/(PREFIX+'-R2-f001-reuse.json'),result)


def checkpoint(frame):
    if frame==1:return Path(json.loads((RECORDS/'route3-upper-bound-2026-11-11-U1-f001.json').read_text())['output'])
    report=json.loads((RECORDS/(PREFIX+f'-R2-f{frame:03d}.json')).read_text())
    if not report['completed'] or report['rows'][-1]['step']!=8000:raise RuntimeError('R2 fit incomplete;no scoring')
    return Path(report['output'])


def render(c,hdr_only=False):
    from src.aperture_renderer import GaussianScene
    from src.layered_aperture import pixel_layer,merge_layers
    from src.analytic_aperture import sampled
    from route3_b2_oracle_3d import sparse_pixel_layer
    from route3_upper_bound_u1 import inputs,hidden_selection
    stage='HDR' if hdr_only else 'R2'
    if hdr_only:assert (RECORDS/(PREFIX+'-R2-evaluation.json')).exists()
    dest=OUT/(stage+'-render');dest.mkdir(exist_ok=False);rows=[];runtime=[]
    for frame in [1,30,60]:
        d,K,cam,unused,parent,fill,bg,profile=inputs(frame);del unused;E=torch.linalg.inv(cam)
        if hdr_only:
            timings=json.loads((BASE/'inputs/B2-high/pinhole/timings.json').read_text());source=next(r for r in timings if r['frame']==frame)
            raw=exr(BASE/'inputs/B2-high/pinhole'/(source['key']+'.exr'))
            visible=pixel_layer(torch.tensor(raw,device='cuda'),torch.tensor(d['center_depth'],device='cuda'),K,cam)
            variants=[('HDR_input_descriptive',visible)]
        else:
            data=torch.load(checkpoint(frame),weights_only=True,map_location='cuda');visible=GaussianScene(**data)
            clipped=GaussianScene(visible.means,visible.quats,visible.scales,visible.opacities,visible.colors.clamp(0,1),visible.source_frames)
            variants=[('U1',visible),('U1_clip',clipped)]
        for x in [x for x in items() if x['frame_id']==frame]:
            chosen,band=hidden_selection(d,parent,x['focus_distance_m'],x['fstop']);hidden=sparse_pixel_layer(parent,chosen,K,cam,profile,'cuda')
            case_times={};torch.cuda.reset_peak_memory_stats()
            for label,pc in variants:
                values={}
                for kind,cloud in [('visible',pc),('joint',merge_layers(pc,hidden))]:
                    torch.cuda.synchronize();start=time.monotonic()
                    with torch.inference_mode():value=sampled(cloud,K,E,512,512,x['focus_distance_m'],x['aperture_radius_m'],64,fill,bg,far_plane=1e5)
                    torch.cuda.synchronize();case_times[label+'_'+kind]=time.monotonic()-start;values[kind]=value['rgb_linear'].cpu().numpy()
                mixed=np.where(band[...,None],values['joint'],values['visible'])
                predictions=[(label,mixed)] if hdr_only else [(label,mixed),(label+'_visible',values['visible'])]
                for name,linear in predictions:
                    path=dest/(x['id']+'-'+name+'.npz');np.savez_compressed(path,rgb_linear=linear,prediction_srgb=display(linear),band=band)
                    rows.append(dict(id=x['id'],method=name,path=str(path)))
            runtime.append(dict(id=x['id'],seconds=case_times,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,hidden_parents=int(chosen.sum())))
            print('R2_RENDER',stage,x['id'],flush=True);del hidden
        del visible,variants;torch.cuda.empty_cache()
    write(OUT/(stage+'-predictions.json'),dict(rows=rows))
    write(RECORDS/(PREFIX+'-'+stage+'-render.json'),dict(completed=True,rows=runtime,K=64,
        hidden_geometry_unchanged=True,fill_unchanged=True,clip_visible_colors_before_render=not hdr_only,
        HDR_input_descriptive_only=hdr_only,old_results_modified=False))


def radiance(c):
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    dest=OUT/'radiance-analysis';dest.mkdir(exist_ok=False);rows=[]
    for frame in [1,30,60]:
        data=torch.load(checkpoint(frame),map_location='cpu',weights_only=True);means=data['means'].numpy();colors=data['colors'].numpy()
        current=np.load(ROOT/'outputs/route3-dual-layer-2026-11-10-01/shared-inputs'/f'B2-high-f{frame:03d}.npz')
        K=current['K'];cam=current['camera_c2w'];timings=json.loads((BASE/'inputs/B2-high/pinhole/timings.json').read_text());source=next(r for r in timings if r['frame']==frame)
        raw=exr(BASE/'inputs/B2-high/pinhole'/(source['key']+'.exr'));mask=(raw>1).any(-1);near=cv2.dilate(mask.astype(np.uint8),np.ones((7,7),np.uint8))>0
        high=(colors>1).any(-1);all_map=np.zeros(512*512,np.int32);high_map=np.zeros_like(all_map);high_xyz=[];high_rgb=[];high_uv=[]
        for start in range(0,len(means),131072):
            m=means[start:start+131072];pc=(m-cam[:3,3])@cam[:3,:3];projected=pc@K.T;uv=projected[:,:2]/np.maximum(projected[:,2:],1e-12)
            xx=np.floor(uv[:,0]).astype(np.int32);yy=np.floor(uv[:,1]).astype(np.int32)
            valid=(pc[:,2]>0)&(xx>=0)&(xx<512)&(yy>=0)&(yy<512);selected=high[start:start+len(m)]&valid
            np.add.at(all_map,yy[valid]*512+xx[valid],1);np.add.at(high_map,yy[selected]*512+xx[selected],1)
            high_xyz.append(m[selected]);high_rgb.append(colors[start:start+len(m)][selected]);high_uv.append(uv[selected].astype(np.float32))
        total=int(high.sum());near_count=int(high_map.reshape(512,512)[near].sum());base_count=int(all_map.reshape(512,512)[near].sum())
        fraction=near_count/max(total,1);baseline=base_count/len(means);xyz=np.concatenate(high_xyz)
        path=dest/f'f{frame:03d}.npz';np.savez_compressed(path,high_xyz=xyz,high_rgb=np.concatenate(high_rgb),high_uv=np.concatenate(high_uv),
            all_count_map=all_map.reshape(512,512),high_count_map=high_map.reshape(512,512),input_HDR_mask=mask,emitter_neighborhood=near)
        fig,axes=plt.subplots(1,3,figsize=(12,4));axes[0].imshow(current['rgb_srgb']);axes[0].set_title('Input LDR')
        axes[1].imshow(near);axes[1].set_title('Input HDR pixels +3px (analysis only)')
        axes[2].imshow(np.log1p(high_map.reshape(512,512)),cmap='magma');axes[2].set_title('log1p projected count,Gaussian RGB>1')
        for ax in axes:ax.axis('off')
        fig.tight_layout();figure=dest/f'f{frame:03d}.png';fig.savefig(figure,dpi=120);plt.close(fig)
        rows.append(dict(frame=frame,gaussians=len(means),high_gaussians=total,high_fraction=total/len(means),linear_color_max=float(colors.max()),
            high_fraction_near_input_HDR_pixels=fraction,all_gaussian_fraction_near_HDR_pixels=baseline,enrichment=fraction/max(baseline,1e-12),
            high_world_xyz_p10_p50_p90=np.quantile(xyz,[.1,.5,.9],axis=0).tolist() if len(xyz) else None,
            source_input_HDR_pixels=int(mask.sum()),neighborhood_radius_px=3,data=str(path),visualization=str(figure)))
        print('R2_RADIANCE',frame,total,fraction,flush=True)
        del data,means,colors
    write(RECORDS/(PREFIX+'-radiance.json'),dict(completed=True,rows=rows,
        input_HDR_used_only_for_analysis=True,not_a_highlight_recovery_method=True,
        interpretation='learned Gaussian color is opacity/scale-dependent;values>1 alone do not certify physicalHDR recovery'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['references','R1','replay','render','radiance','R2-evaluate','HDR-render','HDR-evaluate'],required=True);a=p.parse_args()
    torch.set_num_threads(8);cv2.setNumThreads(1);c=json.loads(CRITERIA.read_text());assert c['frozen']
    if a.mode=='references':references(c)
    elif a.mode=='replay':replay_frame1(c)
    elif a.mode=='radiance':radiance(c)
    elif a.mode in ['render','HDR-render']:
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise ValueError('Explicit GPU0 required')
        subprocess.run(['nvidia-smi'],check=True);torch.set_num_threads(4);render(c,a.mode=='HDR-render')
    else:evaluate(c,{'R1':'R1','R2-evaluate':'R2','HDR-evaluate':'HDR'}[a.mode])


if __name__=='__main__':main()
