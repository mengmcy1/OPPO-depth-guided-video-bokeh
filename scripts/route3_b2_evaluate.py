"""Evaluate B2 only after immutable criteria and all five method outputs exist.

All decisions are per scene and per focus. Pixels and rendering combinations
are not independent scenes. Fixed patch LPIPS sees complete RGB patches.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
import cv2
import numpy as np
import torch

PROJECT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(PROJECT),str(PROJECT/'scripts')]
from src.c1_bokehme import region_sse,patch_metrics,physical_coc_radius
from route3_c1_bokehme_oracle import lpips_cpu
from route3_blender_b1_analyze import load_exr

METHODS=['O3','O5','O6-GT-all','O6-GT-rec','DrBokeh_GT']
PRIMARY_REGIONS=['recoverable_total','motion_required','parallax_sufficient','unrecoverable','disocclusion']
REGIONS=PRIMARY_REGIONS+['residual_excluded','full_image','in_focus']


def display_rgb(x):
    x=np.maximum(x,0)
    return np.clip(np.where(x<=.0031308,12.92*x,1.055*np.maximum(x,1e-12)**(1/2.4)-.055),0,1).astype(np.float32)


def method_path(root,item,method):
    if method=='O3':return root/'methods-2d/bokehme'/item['id']/'prediction.npz'
    if method=='DrBokeh_GT':return root/'methods-2d/drbokeh'/item['id']/'prediction.npz'
    return root/'methods-3d'/item['id']/method/'prediction.npz'


def masks_for(item,source,geometry):
    residual=geometry[item['evaluation_mask_keys']['residual']].astype(bool)
    masks={name:geometry[item['evaluation_mask_keys'][name]].astype(bool) for name in PRIMARY_REGIONS}
    for name,mask in masks.items():
        if np.any(mask&residual):raise ValueError('Provided primary mask contains residual pixels: '+name)
    if not np.array_equal(masks['recoverable_total'],masks['motion_required']|masks['parallax_sufficient']):
        raise ValueError('Recoverable subclasses do not reproduce supplied total mask')
    depth=source['center_depth_raw'];valid=source['center_surface_valid'].astype(bool)&np.isfinite(depth)&(depth>0)
    radius,_=physical_coc_radius(depth,source['K'][0,0],item['aperture_radius_m'],item['focus_distance_m'])
    # Guardrails intentionally include residual pixels; full is really full.
    masks.update(residual_excluded=residual,full_image=np.ones(residual.shape,bool),in_focus=valid&(abs(radius)<=1.))
    return masks


def pooled(rows):
    n=sum(r['pixels'] for r in rows);sse=sum(r['sse'] for r in rows)
    weighted=sum(r['pixels']*r['patch_lpips'] for r in rows if r['patch_lpips'] is not None)
    return dict(pixels=n,sse=sse,psnr=-10*math.log10(max(sse/(3*n),1e-12)) if n else None,
                patch_lpips=weighted/n if n else None,combinations_with_region=sum(r['pixels']>0 for r in rows))


def difference(left,right):
    return dict(psnr_gain_dB=left['psnr']-right['psnr'] if left['psnr'] is not None and right['psnr'] is not None else None,
                patch_lpips_change=left['patch_lpips']-right['patch_lpips'] if left['patch_lpips'] is not None and right['patch_lpips'] is not None else None)


def comparison_gate(diff,per_case,guardrails,criteria):
    gain,lp=diff['psnr_gain_dB'],diff['patch_lpips_change']
    if gain is None or lp is None:return dict(passed=None,status='empty_region')
    quality=gain>=criteria['psnr_gain_dB'] or lp<=criteria['lpips_change_max']
    secondary=gain>=-criteria['guardrail_max_psnr_drop_dB'] and lp<=criteria['secondary_lpips_worsening_max']
    same=sum(d['psnr_gain_dB'] is not None and d['patch_lpips_change'] is not None and d['psnr_gain_dB']>0 and d['patch_lpips_change']<0 for d in per_case)
    required=math.ceil(criteria['same_direction_fraction']*len(per_case))
    guard_ok=all(d['psnr_gain_dB'] is not None and d['psnr_gain_dB']>=-criteria['guardrail_max_psnr_drop_dB'] for d in guardrails.values())
    return dict(**diff,quality_gate_OR=quality,secondary_metric_gate=secondary,
                psnr_worsened=gain<0,patch_lpips_worsened=lp>0,same_direction_count=same,
                eligible_count=len(per_case),required_same_direction_count=required,
                same_direction_fraction=same/len(per_case) if per_case else None,
                stable_gate=bool(per_case) and same>=required,guardrails=guardrails,guardrails_passed=guard_ok,
                improvement_and_stability_passed=quality and secondary and bool(per_case) and same>=required,
                passed=quality and secondary and bool(per_case) and same>=required and guard_ok)


def gt_supported_crop(mask):
    """Place a display crop on actual GT support, without looking at methods."""
    n,labels,stats,centers=cv2.connectedComponentsWithStats(mask.astype(np.uint8),8)
    if n>1:
        component=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]));cx,cy=centers[component]
        yy,xx=np.nonzero(labels==component)
        nearest=int(np.argmin((xx-cx)**2+(yy-cy)**2))
        sx,sy=int(xx[nearest]),int(yy[nearest]);area=int(stats[component,cv2.CC_STAT_AREA])
        assert mask[sy,sx]
    else:
        cx=cy=256.;sx=sy=256;area=0
    x=int(np.clip(sx-32,0,448));y=int(np.clip(sy-32,0,448))
    return y,x,dict(component_pixels=area,component_centroid_yx=[float(cy),float(cx)],
                    GT_support_pixel_yx=[sy,sx],crop_GT_pixels=int(mask[y:y+64,x:x+64].sum()),
                    selection='nearest actual pixel within largest GT-recoverable component to its centroid; deterministic row-major tie; no method scores')


def visualize(root,items,criteria,case_images,case_masks,output_directory=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    out=Path(output_directory) if output_directory else root/'evaluation/visualizations'
    out.mkdir(parents=True,exist_ok=True)
    if any(out.glob('*-patches.png')):raise FileExistsError('Preserve existing figures; use a fresh visualization directory: '+str(out))
    plots=[]
    for scene,focuses in criteria['scope']['focuses'].items():
        for focus in focuses:
            for stop in criteria['primary_fstops']+criteria['pressure_fstops']:
                group=[x for x in items if x['scene']==scene and x['focus_distance_m']==focus and x['fstop']==stop]
                fig,axes=plt.subplots(len(group),len(METHODS)+1,figsize=(15,2.8*len(group)),squeeze=False)
                locator,location_axes=plt.subplots(1,len(group),figsize=(4*len(group),4),squeeze=False)
                entries=[]
                for row,item in enumerate(sorted(group,key=lambda x:x['frame_id'])):
                    mask=case_masks[item['id']]['recoverable_total'];y,x,selection=gt_supported_crop(mask)
                    entries.append(dict(id=item['id'],frame=item['frame_id'],crop_yx=[y,x],**selection))
                    for col,method in enumerate(['GT']+METHODS):
                        image=case_images[item['id']][method];axes[row,col].imshow(image[y:y+64,x:x+64],interpolation='nearest')
                        axes[row,col].set_title(f't{item["frame_id"]} {method}',fontsize=9);axes[row,col].axis('off')
                    ax=location_axes[0,row];ax.imshow(case_images[item['id']]['GT'])
                    if mask.any():ax.contour(mask.astype(np.uint8),levels=[.5],colors=['lime'],linewidths=.5)
                    ax.add_patch(Rectangle((x-.5,y-.5),64,64,fill=False,edgecolor='yellow',linewidth=1.2))
                    sy,sx=selection['GT_support_pixel_yx'];ax.scatter([sx],[sy],s=24,c='red',marker='+')
                    ax.set_title(f't{item["frame_id"]}: green=GT, yellow=crop',fontsize=9);ax.axis('off')
                fig.suptitle(f'{scene}, focus{focus:g}m, f/{stop:g}: fixed frames, GT-selected crop',fontsize=11);fig.tight_layout()
                path=out/f'{scene}-N{stop:g}-F{focus:g}-patches.png';fig.savefig(path,dpi=140);plt.close(fig)
                location_path=out/f'{scene}-N{stop:g}-F{focus:g}-GT-locator.png'
                locator.suptitle(f'{scene}, focus{focus:g}m, f/{stop:g}: GT-only crop location');locator.tight_layout();locator.savefig(location_path,dpi=140);plt.close(locator)
                plots.append(dict(scene=scene,focus_distance_m=focus,fstop=stop,path=str(path),GT_locator_path=str(location_path),cases=entries))
    return plots


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--criteria',type=Path,required=True)
    p.add_argument('--manifest',type=Path);p.add_argument('--record',type=Path,required=True)
    p.add_argument('--alexnet-weights',type=Path,default=Path('/root/autodl-tmp/.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth'))
    p.add_argument('--visualize-only',action='store_true',help='Redraw figures without loading LPIPS or computing any quality scores')
    p.add_argument('--visualization-directory',type=Path,help='Fresh figure directory; existing montage files are never replaced')
    a=p.parse_args();a.root=a.root.resolve();a.manifest=a.manifest or a.root/'manifests/oracle.json'
    frozen_bytes=a.criteria.read_bytes();criteria=json.loads(frozen_bytes)
    if criteria.get('frozen') is not True or criteria.get('method_outputs_existed_at_freeze') is not False:raise ValueError('Immutable pre-method frozen criteria required')
    if a.record.exists():raise FileExistsError('Preserve previous evaluation: '+str(a.record))
    if criteria['statistics']['LPIPS_net']!='alex' or criteria['statistics']['LPIPS_version']!='0.1' or criteria['statistics']['LPIPS_spatial']:
        raise ValueError('This evaluator implements the frozen AlexNet0.1 nonspatial patch protocol')
    items=json.loads(a.manifest.read_text())['items']
    if len(items)!=criteria['scope']['planned_group_count']:raise ValueError('Manifest group count differs from frozen scope')
    missing=[str(method_path(a.root,item,m)) for item in items for m in METHODS if not method_path(a.root,item,m).is_file()]
    if missing:raise FileNotFoundError('Method outputs incomplete; no scores computed: '+str(missing[:8]))
    if a.visualize_only:
        cv2.setNumThreads(1);images={};masks={}
        for item in items:
            images[item['id']]={'GT':display_rgb(load_exr(item['gt_rgb_exr']))}
            images[item['id']].update({m:np.load(method_path(a.root,item,m))['prediction_srgb'] for m in METHODS})
            geometry=np.load(item['geometry_gt_npz']);masks[item['id']]={'recoverable_total':geometry[item['evaluation_mask_keys']['recoverable_total']].astype(bool)}
        plots=visualize(a.root,items,criteria,images,masks,a.visualization_directory)
        if a.criteria.read_bytes()!=frozen_bytes:raise RuntimeError('Frozen criteria changed during figure generation')
        record=dict(mode='visualization-only GT-support correction',plots=plots,fixed_frames=criteria['scope']['representative_frames'],
                    crop_size=64,selection_uses_only_GT=True,method_scores_used=False,quality_scores_recomputed=False,
                    LPIPS_patch_grid_changed=False,method_outputs_changed=False,frozen_criteria_changed=False,
                    previous_figures_preserved=str(a.root/'evaluation/visualizations'))
        a.record.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8');print('VISUALIZATIONS_ONLY_COMPLETE',len(plots),flush=True);return
    torch.set_num_threads(8);cv2.setNumThreads(1);model=lpips_cpu(SimpleNamespace(alexnet_weights=a.alexnet_weights))
    patch=int(criteria['statistics']['patch_size']);positions=[(y,x) for y in range(0,512,patch) for x in range(0,512,patch)]
    case_metrics={};case_meta=[];flat=[];patch_arrays={};case_images={};case_masks={}
    geometry_summaries={s:json.loads((a.root/s/'geometry/summary.json').read_text()) for s in criteria['scope']['focuses']}
    for item in items:
        raw=load_exr(item['gt_rgb_exr']);target=display_rgb(raw);source=np.load(item['input_npz']);geometry=np.load(item['geometry_gt_npz'])
        masks=masks_for(item,source,geometry);case_masks[item['id']]=masks;case_images[item['id']]={'GT':target}
        coc_median=float(item['coc_pairmax_p10_p50_p90'][1]);low_coc=coc_median<criteria['eligibility']['low_CoC_threshold_px']
        eligible=item['fstop'] in criteria['primary_fstops'] and not low_coc and int(masks['recoverable_total'].sum())>=criteria['min_pixels']
        g=next(g for g in geometry_summaries[item['scene']] if 'fstop' in g and g['frame']==item['frame_id'] and abs(g['fstop']-item['fstop'])<1e-7 and abs(g['focus_m']-item['focus_distance_m'])<1e-7)
        counts={name:int(mask.sum()) for name,mask in masks.items()}
        meta=dict(id=item['id'],scene=item['scene'],frame=item['frame_id'],fstop=item['fstop'],focus_distance_m=item['focus_distance_m'],
            role='main' if item['fstop'] in criteria['primary_fstops'] else 'pressure',low_CoC=low_coc,CoC_pairmax_median_px=coc_median,
            eligible_primary=eligible,region_pixels=counts,region_image_fractions={k:v/(512*512) for k,v in counts.items()},
            geometry_visibility={k:g[k] for k in ['hidden_rays','actual_recoverable_rays','residual_rays','residual_pixels','known_aperture_ray_fraction','raw_hidden_pixels','raw_recoverable_pixels','image_disocclusion_fraction','coc_hidden_p10_p50_p90','coc_blocking_p10_p50_p90','coc_current_in_effective_rec_p10_p50_p90']},
            recoverable_image_fraction_among_supported_hidden=counts['recoverable_total']/max(counts['disocclusion'],1),
            recoverable_ray_fraction=g['actual_recoverable_rays']/max(g['hidden_rays'],1),GT_clipped_component_fraction=float(((raw<0)|(raw>1)).mean()))
        case_meta.append(meta);case_metrics[item['id']]={}
        for method in METHODS:
            prediction=np.load(method_path(a.root,item,method))['prediction_srgb'].astype(np.float32)
            if prediction.shape!=target.shape or not np.isfinite(prediction).all() or prediction.min()<-1e-5 or prediction.max()>1+1e-5:
                raise ValueError('Expected finite display RGB[0,1]: '+item['id']+' '+method)
            prediction=np.clip(prediction,0,1);case_images[item['id']][method]=prediction
            pm=patch_metrics(prediction,target,np.ones((512,512),bool),model,patch_size=patch,device='cpu');scores=np.asarray(pm['patch_lpips_scores'])
            patch_arrays[item['id']+'__'+method]=scores.astype(np.float32);case_metrics[item['id']][method]={}
            for region,mask in masks.items():
                if region=='residual_excluded':
                    metrics=dict(pixels=int(mask.sum()),sse=0.,psnr=None,patch_lpips=None,patch_count=0,status='excluded_from_primary_and_subregion_scores; counted only')
                else:
                    metrics=region_sse(prediction,target,mask);support=np.array([mask[y:y+patch,x:x+patch].sum() for y,x in positions])
                    metrics.update(patch_lpips=float(scores@support/support.sum()) if support.sum() else None,
                                   patch_count=int((support>0).sum()),status='scored' if support.sum() else 'empty_region')
                case_metrics[item['id']][method][region]=metrics
                flat.append(dict(id=item['id'],scene=item['scene'],frame=item['frame_id'],fstop=item['fstop'],focus_distance_m=item['focus_distance_m'],
                                 eligible_primary=eligible,low_CoC=low_coc,method=method,region=region,**metrics))
        print('EVALUATED',item['id'],flush=True)
    def tables(ids):
        return {m:{r:pooled([case_metrics[i][m][r] for i in ids]) if r!='residual_excluded' else dict(pixels=sum(case_metrics[i][m][r]['pixels'] for i in ids),psnr=None,patch_lpips=None,status='excluded') for r in REGIONS} for m in METHODS}
    by_scene_focus={};by_aperture={};decisions={};primary=criteria['decision']['primary_method']
    for scene,focuses in criteria['scope']['focuses'].items():
        by_scene_focus[scene]={};by_aperture[scene]={};decisions[scene]={}
        for focus in focuses:
            group=[r for r in case_meta if r['scene']==scene and r['focus_distance_m']==focus]
            main=[r['id'] for r in group if r['role']=='main'];eligible=[r['id'] for r in group if r['eligible_primary']]
            by_scene_focus[scene][str(focus)]=dict(primary_eligible=tables(eligible),all_main_descriptive=tables(main),
                eligible_count=len(eligible),planned_count=len(main),low_CoC_main_count=sum(r['low_CoC'] and r['role']=='main' for r in group))
            by_aperture[scene][str(focus)]={str(stop):tables([r['id'] for r in group if r['fstop']==stop]) for stop in sorted({r['fstop'] for r in group})}
            controls=scene==criteria['scope']['control_scene'];enough=len(eligible)>=criteria['min_groups'];comparisons={}
            if enough and not controls:
                table=by_scene_focus[scene][str(focus)]['primary_eligible']
                for baseline in [criteria['decision']['mechanism_comparator']]+criteria['decision']['method_value_comparators']:
                    gain=difference(table[primary]['recoverable_total'],table[baseline]['recoverable_total'])
                    per_case=[difference(case_metrics[i][primary]['recoverable_total'],case_metrics[i][baseline]['recoverable_total']) for i in eligible]
                    guard={r:difference(table[primary][r],table[baseline][r]) for r in ['full_image','in_focus']}
                    comparisons[baseline]=comparison_gate(gain,per_case,guard,criteria)
                mechanism=comparisons[criteria['decision']['mechanism_comparator']]['improvement_and_stability_passed']
                value=all(comparisons[m]['improvement_and_stability_passed'] for m in criteria['decision']['method_value_comparators'])
                guards=all(v['guardrails_passed'] for v in comparisons.values())
                decision=dict(status='evaluated_under_frozen_criteria',mechanism_passed=mechanism,method_value_passed=value,all_guardrails_passed=guards,passed=mechanism and value and guards,comparisons=comparisons)
            else:decision=dict(status='low_CoC_control_descriptive_only' if controls else 'insufficient_eligible_groups',passed=None,comparisons={},eligible_count=len(eligible),required_count=criteria['min_groups'])
            descriptive=by_scene_focus[scene][str(focus)]['all_main_descriptive']
            decision['all_minus_rec_diagnostic']={r:difference(descriptive['O6-GT-all'][r],descriptive['O6-GT-rec'][r]) for r in REGIONS if r!='residual_excluded'}
            decisions[scene][str(focus)]=decision
    high=decisions[criteria['scope']['primary_scene']];passes=sum(d.get('passed') is True for d in high.values());assessed=sum(d.get('passed') is not None for d in high.values())
    interpretation='conditional_effectiveness_for_one_focus' if passes==1 else ('passes_both_tested_focuses_in_one_controlled_scene' if passes==len(high) else ('insufficient_valid_groups' if not assessed else 'no_focus_passes_frozen_criteria'))
    output=a.root/'evaluation';output.mkdir(parents=True,exist_ok=True);np.savez_compressed(output/'patch-distances.npz',**patch_arrays)
    plots=visualize(a.root,items,criteria,case_images,case_masks,a.visualization_directory)
    if a.criteria.read_bytes()!=frozen_bytes:raise RuntimeError('Frozen criteria changed during evaluation')
    record=dict(phase='B2-only frozen evaluation',criteria=str(a.criteria.resolve()),criteria_frozen_at_utc=criteria['frozen_at_utc'],
        no_cross_focus_pooling_for_decisions=True,method_display_names={'DrBokeh_GT':'Oracle-input Dr.Bokeh','O6-GT-rec':'GT Oracle for recoverable surfaces'},
        statistics=criteria['statistics'],cases=case_meta,by_scene_focus=by_scene_focus,by_aperture=by_aperture,decisions=decisions,
        interpretation=interpretation,plots=plots,per_patch_distances=str(output/'patch-distances.npz'),
        limitations=criteria['scope']['limitation'],all_minus_rec_is_recoverability_fraction=False,model_training=False)
    a.record.parent.mkdir(parents=True,exist_ok=True);a.record.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    fields=list(flat[0])
    with a.record.with_suffix('.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows({k:r.get(k) for k in fields} for r in flat)
    print('EVALUATION_COMPLETE',interpretation,json.dumps(decisions),flush=True)


if __name__=='__main__':main()
