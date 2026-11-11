"""Canonical float64 diagnostic tables and S1d evidence summary."""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from route3_sharp_s1d import OUT,PREFIX,CRITERIA,RECORDS,S1C,BASE,entries,write
from route3_upper_bound_evaluate import items,display,exr
from route3_sharp_s1_evaluate import write_csv as first_write_csv
from route3_sharp_s1d_diagnose import LABELS
from src.dual_layer import depth_band
from scipy.spatial import cKDTree


def load(suffix):return json.loads((RECORDS/(PREFIX+'-'+suffix+'.json')).read_text())


def write_csv(path,rows):
    if not path.exists():return first_write_csv(path,rows)
    buf=io.StringIO();writer=csv.DictWriter(buf,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    assert path.read_text()==buf.getvalue(),f'Existing derived table differs: {path}'


def decomposition():
    old=load('decomposition');cache=json.loads((RECORDS/'route3-sharp-s1c-2026-11-10-evaluation.json').read_text());refs=json.loads((ROOT/'outputs/route3-ldr-u1r-2026-11-10-01/references.json').read_text());refmap={x['id']:x['path'] for x in refs['rows']};rows=[];overlaps=[]
    for case in items():
        lab=np.load(OUT/'decomposition'/(case['id']+'.npz'));geo=np.load(OUT/'geometry'/(case['id']+'.npz'));parent=np.load(OUT/'candidates'/f'f{case["frame_id"]:03d}'/'R0.npz');pix=lab['pixel'];labels=lab['label'];count=np.bincount(pix,minlength=512*512);valid=count>0;pred=np.load(cache['predictions'][case['id']]['cal_E1__A_color_L3_cal_E1'])['prediction_srgb'];truths=dict(Cycles=display(exr(case['gt_rgb_exr'])),LDR=np.load(refmap[case['id']])['reference_srgb']);errors={ref:((pred.astype(np.float64)-t.astype(np.float64))**2).sum(-1).reshape(-1) for ref,t in truths.items()};cats=[]
        for code,name in enumerate(LABELS):
            per=np.bincount(pix[labels==code],minlength=512*512);fraction=per/np.maximum(count,1);cats.append(dict(category=name,rays=int(per.sum()),fractional_pixel_mass=float(fraction.sum()),any_ray_pixels=int((per>0).sum()),SSE={ref:float(fraction@error) for ref,error in errors.items()}))
        row=next(r.copy() for r in old['rows'] if r['id']==case['id']);row['categories']=cats;rows.append(row)
        association=lab['parent_index'];found=association>=0;safe=np.maximum(association,0);depthbad=found&((abs(parent['depth'][safe]-geo['depth'][geo['recoverable']])>np.maximum(.02,.02*geo['depth'][geo['recoverable']]))|lab['front_overlap']);colorbad=lab['color_error_overlap'];motion=lab['motion_overlap'];pool_edge=np.load(OUT/'candidates'/f'f{case["frame_id"]:03d}'/'pool.npz')['edge'];counts=parent['contrib_count'];pid=np.repeat(np.arange(len(counts)),counts);edge_fraction=np.bincount(pid,weights=pool_edge[parent['contrib_indices']],minlength=len(counts))/np.maximum(counts,1);edgecolor=colorbad&found&(edge_fraction[safe]>=.5)
        flags=dict(depth_any=depthbad,color_any=colorbad,motion_any=motion,depth_and_color=depthbad&colorbad,depth_and_motion=depthbad&motion,color_and_motion=colorbad&motion,depth_color_motion=depthbad&colorbad&motion,color_and_source_edge=edgecolor)
        for name,flag in flags.items():
            mass=np.bincount(pix[flag],minlength=512*512)/np.maximum(count,1);overlaps.append(dict(id=case['id'],focus_m=case['focus_distance_m'],flag=name,rays=int(flag.sum()),eligible_pixels=int(valid.sum()),fractional_pixel_mass=float(mass.sum()),SSE={ref:float(mass@error) for ref,error in errors.items()}))
    pooled={};overlap_pool={};table=[]
    for focus in [1.4,5.]:
        group=[r for r in rows if r['focus_m']==focus];n=sum(r['eligible_pixels'] for r in group);total={ref:sum(c['SSE'][ref] for r in group for c in r['categories']) for ref in ['Cycles','LDR']};pooled[str(focus)]={};overlap_pool[str(focus)]={}
        for code,name in enumerate(LABELS):
            selected=[r['categories'][code] for r in group];v=dict(rays=sum(c['rays'] for c in selected),fractional_pixel_share=sum(c['fractional_pixel_mass'] for c in selected)/n,any_ray_pixel_share=sum(c['any_ray_pixels'] for c in selected)/n,SSE_share={ref:sum(c['SSE'][ref] for c in selected)/total[ref] for ref in total});pooled[str(focus)][name]=v;table.append(dict(focus_m=focus,category=name,**{k:v[k] for k in ['rays','fractional_pixel_share','any_ray_pixel_share']},Cycles_SSE_share=v['SSE_share']['Cycles'],LDR_SSE_share=v['SSE_share']['LDR']))
        for name in sorted({r['flag'] for r in overlaps}):
            selected=[r for r in overlaps if r['focus_m']==focus and r['flag']==name];overlap_pool[str(focus)][name]=dict(rays=sum(c['rays'] for c in selected),fractional_pixel_share=sum(c['fractional_pixel_mass'] for c in selected)/n,SSE_share={ref:sum(c['SSE'][ref] for c in selected)/total[ref] for ref in total})
    write(RECORDS/(PREFIX+'-decomposition-report.json'),dict(completed=True,canonical_SSE_float64=True,classification_labels_unchanged=True,initial_record_preserved=True,rows=rows,pooled=pooled,overlapping_flags=overlap_pool,visualizations=old['visualizations'],category_priority_does_not_remove_joint_errors=True,
        SSE_frequency_attribution_not_causal=True,coverage_is_projected_parent_presence_not_guaranteed_optical_contribution=True,original_K16_scoremask_and_actual_K64_geometry=True));write_csv(RECORDS/(PREFIX+'-decomposition.csv'),table)


def replacement_audit():
    from route3_sharp_s1 import FRAMES
    gtviews=[np.load(BASE/'B2-high/geometry'/f'f{f:03d}-center.npz')['object_id'] for f in FRAMES];rows=[]
    for entry in entries():
        frame=entry['frame'];p=np.load(OUT/'candidates'/f'f{frame:03d}'/'R0.npz');gt=np.load(BASE/'methods-3d/prepared'/f'B2-high-f{frame:03d}'/'parents.npz');gids=np.flatnonzero(gt['recoverable']);distance,index=cKDTree(gt['uv'][gids]).query(p['uv'],k=16,workers=8);valid=(index<len(gids))&(distance<=1.5);ids=gids[np.minimum(index,len(gids)-1)];depths=gt['depth'][ids];nearest=np.where(valid,depths,np.inf);first=np.argmin(nearest,axis=1);matched=ids[np.arange(len(first)),first];ok=np.isfinite(nearest[np.arange(len(first)),first]);low=np.min(nearest,axis=1);high=np.max(np.where(valid,depths,-np.inf),axis=1);multilayer=ok&((high-low)>np.maximum(.02,.02*low))
        pool=np.load(OUT/'candidates'/f'f{frame:03d}'/'pool.npz');contrib=p['contrib_indices'];parent_id=np.repeat(np.arange(len(p['depth'])),p['contrib_count']);slots=pool['slot'][contrib];xy=pool['source_xy'][contrib];objects=np.zeros(len(contrib),np.int16)
        for slot in np.unique(slots):
            at=slots==slot;objects[at]=gtviews[int(slot)][xy[at,1],xy[at,0]]
        mismatched=objects!=gt['object_id'][matched[parent_id]];mismatch_fraction=np.bincount(parent_id,weights=mismatched,minlength=len(p['depth']))/np.maximum(p['contrib_count'],1);d=np.load(entry['input_npz']);vis=np.load(OUT/'visible'/f'f{frame:03d}.npz')['depth'];x=np.clip(p['uv'][:,0].astype(int),0,511);y=np.clip(p['uv'][:,1].astype(int),0,511);newz=gt['depth'][matched]
        for case in entry['cases']:
            band=depth_band(d['depth'],d['K'][0,0],case['focus_m'],.05/(2*case['fstop']))[0];selected=ok&band[y,x];n=int(selected.sum());rows.append(dict(id=case['id'],matched_replaced_parents=n,GT_neighborhood_multiple_depths_fraction=float((selected&multilayer).sum()/max(n,1)),majority_source_GT_object_differs_from_first_GT_layer_fraction=float((selected&(mismatch_fraction>=.5)).sum()/max(n,1)),replacement_violates_R1_visible_order_fraction=float((selected&(newz<=vis[y,x]+np.maximum(.005,.02*vis[y,x]))).sum()/max(n,1))))
    write(RECORDS/(PREFIX+'-replacement-audit.json'),dict(completed=True,rows=rows,read_only_post_observation_check=True,no_new_render_or_changed_GT_pairing=True,
        frozen_pairing='first recoverable hidden layer within1.5px from16nearestGTparents',interpretation='Single-field substitutions can mix geometry from a target hidden surface with color from another source surface, or conflict with fixed biased visible geometry. Negative outcomes do not show correct physical depth is harmful; interventions are not an unambiguous additive causal decomposition.'))


def summary():
    evaluation=load('evaluation');decomp=load('decomposition-report');renders={m:load('render-'+m) for m in ['R1','R2','R3','R4','Oracle_depth_only','Oracle_color_only','Oracle_missing_only']};diagnostic={};focus_checks={};oldrender=json.loads((RECORDS/'route3-sharp-s1c-2026-11-10-render.json').read_text());baseline=[r for r in oldrender['hidden_diagnostics'] if r['geometry']=='cal_E1' and r['level']=='L3_cal_E1'];case_lookup={x['id']:x for x in items()};flat=[]
    for method,report in renders.items():
        focus_checks[method]=report['focus_quick_check'];diagnostic[method]={}
        for focus in [1.4,5.]:
            group=[r for r in report['diagnostics'] if r['focus_m']==focus];nr=sum(r['rec_rays'] for r in group);diagnostic[method][str(focus)]=dict(parents=sum(r['parents'] for r in group),wrong_front_fraction=sum(r['wrong_parents'] for r in group)/max(sum(r['known_GT_parents'] for r in group),1),coverage=sum(r['covered_rec_rays'] for r in group)/nr,depth_correct_coverage=sum(r['depth_correct_rec_rays'] for r in group)/nr,depth_color_correct_coverage=sum(r['depth_color_correct_rec_rays'] for r in group)/nr,tag_wrong_mean_band_unweighted=float(np.mean([r['tag_wrong_mean_band'] for r in group])))
            flat.append(dict(method=method,focus_m=focus,**diagnostic[method][str(focus)]))
    r0_rows=[]
    for x in items():
        parents=np.load(OUT/'candidates'/f'f{x["frame_id"]:03d}'/'R0.npz');old=np.load(S1C/'prepared'/f'cal_E1-f{x["frame_id"]:03d}'/'L3_cal_E1.npz')
        for key in ['uv','depth','colors_linear','source_mask','source_count']:assert np.array_equal(parents[key],old[key])
        geo=np.load(OUT/'geometry'/(x['id']+'.npz'));lab=np.load(OUT/'decomposition'/(x['id']+'.npz'));assoc=lab['parent_index'];found=assoc>=0;safe=np.maximum(assoc,0);entry=next(e for e in entries() if e['frame']==x['frame_id']);d=np.load(entry['input_npz']);lens=next(v for v in entry['cases'] if v['id']==x['id']);band=depth_band(d['depth'],d['K'][0,0],lens['focus_m'],x['aperture_radius_m'])[0];q=geo['uv'][geo['recoverable']];pix=geo['pixel'][geo['recoverable']];found&=band.flat[pix]&band[np.clip(q[:,1].astype(int),0,511),np.clip(q[:,0].astype(int),0,511)];good=found&(abs(parents['depth'][safe]-geo['depth'][geo['recoverable']])<=np.maximum(.02,.02*geo['depth'][geo['recoverable']]));col=good&(geo['GT_parent'][geo['recoverable']]>=0)&(abs(parents['colors_linear'][safe]-geo['GT_color'][geo['recoverable']]).max(-1)<=.05);original=next(r for r in baseline if r['id']==x['id'])
        r0_rows.append(dict(focus_m=x['focus_distance_m'],parents=original['parents'],known_GT_parents=original['known_GT_supported_parents'],wrong_parents=original['wrong_parents'],rec_rays=len(q),covered_rec_rays=int(found.sum()),depth_correct_rec_rays=int(good.sum()),depth_color_correct_rec_rays=int(col.sum()),tag_wrong_mean_band=original['tag_mean_contribution_band']))
    diagnostic['R0']={}
    for focus in [1.4,5.]:
        group=[r for r in r0_rows if r['focus_m']==focus];nr=sum(r['rec_rays'] for r in group);v=dict(parents=sum(r['parents'] for r in group),wrong_front_fraction=sum(r['wrong_parents'] for r in group)/max(sum(r['known_GT_parents'] for r in group),1),coverage=sum(r['covered_rec_rays'] for r in group)/nr,depth_correct_coverage=sum(r['depth_correct_rec_rays'] for r in group)/nr,depth_color_correct_coverage=sum(r['depth_color_correct_rec_rays'] for r in group)/nr,tag_wrong_mean_band_unweighted=float(np.mean([r['tag_wrong_mean_band'] for r in group])));diagnostic['R0'][str(focus)]=v;flat.append(dict(method='R0',focus_m=focus,**v))
    assert all(r['unclassified_fraction']<=.05 for r in decomp['rows'])
    assert all(len(report['rows'])==12 for report in renders.values()) and len(evaluation['per_case_metrics']['Cycles'])==12
    validation=dict(passed=True,R0_all5_arrays_identical_to_S1c=True,unclassified_below_frozen5percent=True,fixed_visible_no_anchoring=True,all_method_changes_inside_fixed_band=all(d['all_RGB_changes_inside_fixed_band'] for r in renders.values() for d in r['diagnostics']),method_outputs=12*len(renders),other_processes_killed=False,GT_not_used_in_R1_R4=True,rule_final_R4_not_selected_by_scores=True)
    write(RECORDS/(PREFIX+'-validation.json'),validation);write(RECORDS/(PREFIX+'-diagnostics.json'),dict(completed=True,pooled=diagnostic,focus_checks=focus_checks,all_per_case=[d for r in renders.values() for d in r['diagnostics']],wrong_front_versus_true_visible_surface_is_not_all_hidden_depth_error=True));write_csv(RECORDS/(PREFIX+'-diagnostics.csv'),flat)
    def space(path):
        s=os.statvfs(path);return dict(available_GiB=s.f_bavail*s.f_frsize/2**30,available_bytes=s.f_bavail*s.f_frsize)
    report=dict(status='completed_pause_for_user',client_date='2026-11-11',client_timezone='America/Los_Angeles',criteria=dict(path=str(CRITERIA.relative_to(ROOT)),commit='634b30f',implementation_commit='8d0b613',unchanged=True),report_base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        scope=dict(scene='B2-high',frames=[1,30,60],focuses_m=[1.4,5.],fstops=[1.4,2.],K=64,visible='unchanged S1c calibrated A-color',E1_GT_poses=True,no_training=True),
        decomposition=decomp['pooled'],overlapping_errors=decomp['overlapping_flags'],unknown_max=max(r['unclassified_fraction'] for r in decomp['rows']),SSE_is_association_not_causal=True,
        P6=evaluation['P6'],main_pooled=evaluation['pooled'],per_aperture=evaluation['per_aperture'],gap_recovery=evaluation['gap_recovery'],O3_DA3_descriptive=evaluation['baseline_O3_DA3_descriptive'],rule_diagnostics=diagnostic,
        dataset_geometry=load('geometry'),confidence=load('confidence'),rule_preparation=load('rules'),replacement_matching_audit=load('replacement-audit'),replacement_support=[dict(id=x['id'],**json.loads((OUT/'replacements'/x['id']/'audit.json').read_text())) for x in items()],
        render_timings={m:dict(prediction_seconds_sum=sum(row['seconds'] for row in report['rows']),prediction_seconds_median=float(np.median([row['seconds'] for row in report['rows']])),peak_allocated_MiB=max(row['peak_allocated_MiB'] for row in report['rows']),excludes_retrieval_load_IO_GT_tag_pass=True) for m,report in renders.items()},
        interpretation_record=PREFIX+'-interpretation.json',
        resources=dict(data_disk=space('/root/autodl-tmp'),system_disk=space('/'),outputs_bytes=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()),GPU_stages_serial=True,no_environment_installation=True),
        files=dict(output=str(OUT),class_table=PREFIX+'-decomposition.csv',class_report=PREFIX+'-decomposition-report.json',method_table=PREFIX+'-tables.csv',gap_table=PREFIX+'-gap-recovery.csv',O3_DA3=PREFIX+'-O3-DA3-descriptive.csv',diagnostics=PREFIX+'-diagnostics.json',visualizations=dict(classes=str(OUT/'decomposition'),methods=str(OUT/'evaluation'))),
        boundaries=['onecontrolledscene,3frames,6groups/focus;accurateGTposesassumed','diagnosticcategoryshares notuniquephysicalcausality;pixelSSEapportionedbyGTrec-rayfrequencies','sourceedge/view/exposure flags areproxies;noisolatedexposureintervention','coveragewithin1.5pxparentassociation isnotguaranteedopticalsurfacecoverage','depth/colorGTreplacementonlyexistingmatchedsurfacepoints;missingrepairdoesnotrepairring','GTOracle/replacementsnotmethods','R4allrulesfixedbeforemetrics,notbestofobservedstages','failedsmallrulefamilycannotprovelearningstrictlynecessary'],after_report='pause;userdecidestrainingornewmethods')
    write(RECORDS/(PREFIX+'-summary.json'),report);print('S1D_SUMMARY_WRITTEN',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['decomposition','replacement_audit','summary'],required=True);a=p.parse_args();globals()[a.mode]()


if __name__=='__main__':main()
