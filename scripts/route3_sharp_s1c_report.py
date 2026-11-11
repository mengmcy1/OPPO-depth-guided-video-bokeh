"""Final S1c evidence summary; accurate E1 poses are an explicit condition."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from route3_sharp_s1c import OUT,PREFIX,RECORDS,CRITERIA,write
from route3_upper_bound_evaluate import items
from src.dual_layer import depth_band
from route3_upper_bound_evaluate import masks_for


def read(suffix):return json.loads((RECORDS/(PREFIX+'-'+suffix+'.json')).read_text())


def main():
    ev=read('evaluation');cal=read('calibration');geo=read('geometry-evaluation');analysis=read('geometry-analysis');e2=read('E2-block-analysis');render=read('render');random=read('random16-evaluation')
    lookup={x['id']:x for x in items()};prepared=json.loads((OUT/'prepared.json').read_text())['rows'];tags=[];tagpool={};cracks={};cost={};edge_coverage=[];c=json.loads(CRITERIA.read_text())
    for entry in prepared:
        valid=np.load(entry['input_npz'])['edge_loss_valid']
        for case in entry['cases']:
            masks=masks_for(lookup[case['id']],c)
            edge_coverage.append(dict(geometry=entry['geometry'],id=case['id'],focus_m=lookup[case['id']]['focus_distance_m'],excluded_by_scoring_region={r:float((~valid&mask).sum()/mask.sum()) for r,mask in masks.items() if mask.any()},GT_only_evaluator_not_used_for_loss_mask=True))
    for row in render['hidden_diagnostics']:
        x=lookup[row['id']];entry=next(e for e in prepared if e['geometry']==row['geometry'] and e['frame']==x['frame_id']);d=np.load(entry['input_npz']);case=next(c for c in entry['cases'] if c['id']==row['id']);band=depth_band(d['depth'],d['K'][0,0],case['focus_m'],x['aperture_radius_m'])[0]
        tags.append(dict(**row,focus_m=x['focus_distance_m'],band_pixels=int(band.sum()),hidden_Gaussians=row['parents']*25,wrong_hidden_Gaussians=row['wrong_parents']*25,
            wrong_fraction_all_parents=row['wrong_parents']/max(row['parents'],1),unknown_GT_parents=row['parents']-row['known_GT_supported_parents']))
    for geom in ['native','cal_E1']:
        tagpool[geom]={}
        for focus in [1.4,5.]:
            tagpool[geom][str(focus)]={}
            for level in ['O6_GT_rec','L3_GTgeometry','L3_cal_E1','L3_native_E1']:
                rows=[r for r in tags if r['geometry']==geom and r['focus_m']==focus and r['level']==level];npix=sum(r['band_pixels'] for r in rows)
                tagpool[geom][str(focus)][level]=dict(hidden_Gaussians=sum(r['hidden_Gaussians'] for r in rows),wrong_hidden_Gaussians=sum(r['wrong_hidden_Gaussians'] for r in rows),
                    wrong_fraction_known_GT=sum(r['wrong_parents'] for r in rows)/max(sum(r['known_GT_supported_parents'] for r in rows),1),
                    wrong_fraction_all=sum(r['wrong_parents'] for r in rows)/max(sum(r['parents'] for r in rows),1),tag_mean_render_contribution_in_predicted_band=sum(r['tag_mean_contribution_band']*r['band_pixels'] for r in rows)/npix)
    for focus in [1.4,5.]:
        cracks[str(focus)]={}
        for method in sorted({r['method'] for r in ev['cracks']}):
            rows=[r for r in ev['cracks'] if r['focus_m']==focus and r['method']==method];n=sum(r['pixels'] for r in rows)
            cracks[str(focus)][method]=dict(pixels=n,leaky_pixels=sum(r['leaky_pixels'] for r in rows),leaky_fraction=sum(r['leaky_pixels'] for r in rows)/n,
                mean_transmittance=sum(r['mean_transmittance']*r['pixels'] for r in rows)/n,fill_L1_share=sum(r['fill_L1_sum'] for r in rows)/max(sum(r['RGB_L1_sum'] for r in rows),1e-12))
    for cloud in ['O5','O6-GT-rec']:
        cost[cloud]={}
        for mode in ['fixed64','stratified16']:
            rows=[r for r in random['profiles'] if r['cloud']==cloud and r['mode']==mode];cost[cloud][mode]=dict(forward_seconds_median=float(np.median([r['median_forward_seconds'] for r in rows])),backward_seconds_median=float(np.median([r['median_backward_seconds'] for r in rows])),
                total_seconds_median=float(np.median([r['median_total_seconds'] for r in rows])),peak_allocated_MiB=max(r['peak_allocated_MiB'] for r in rows),all5_gradients_finite=all(r['all_attribute_gradients_finite'] for r in rows))
    fits=json.loads((OUT/'anchor.json').read_text())['rows'];mapping=[json.loads(p.read_text()) for p in sorted((OUT/'geometry-mapping').glob('*.json'))]
    native=json.loads((ROOT/'outputs/route3-sharp-s1b-2026-11-11-01/native-predictions.json').read_text())['rows'];new=json.loads((OUT/'predictions.json').read_text());differences=[]
    for row in new['rows']:
        if row['geometry']=='native' and row['method']=='A_color_L3_native_E1':
            old=next(r for r in native if r['id']==row['id'] and r['method']=='A_color_L3_E1');delta=float(abs(np.load(row['path'])['prediction_srgb']-np.load(old['path'])['prediction_srgb']).max());differences.append(dict(id=row['id'],max_RGB_difference=delta));assert delta<2e-5
    assert len(new['rows'])==192 and len(new['zeros'])==18 and len(tags)==96 and len(fits)==12 and all(f['completed'] and f['completed_updates']==1000 for f in fits)
    write(RECORDS/(PREFIX+'-diagnostics.json'),dict(completed=True,wrong_hidden=tags,wrong_hidden_pooled=tagpool,cracks_pooled=cracks,zero=ev['zero'],projection_mapping=mapping,
        edge_mask_evaluation_coverage=edge_coverage,GT_depth_never_used_to_delete_retrieved_points=True,GT_retrieval_zero_wrong_count_is_expected_by_same_GT_filter_not_independent_proof=True,
        tag_RGB_is_contribution_not_count_and_is_in_linear_RGB=True,one_parent_has25_equal_attribute_Gaussians=True))
    write(RECORDS/(PREFIX+'-validation.json'),dict(passed=True,method_outputs192=True,zero_outputs18=True,tag_cases96=True,all12_fits_final1000=True,native_A_color_old_PSNR_identical=True,native_L3_against_S1b=differences,
        mapping=mapping,all_optional_gradients_finite=all(r['all_attribute_gradients_finite'] for r in random['profiles']),E2_no_new_fits_after_stop=True,criteria_changed_after_metrics=False))
    def space(path):
        s=os.statvfs(path);return dict(available_bytes=s.f_bavail*s.f_frsize,available_GiB=s.f_bavail*s.f_frsize/2**30)
    representative=[r for r in geo['rows'] if r['variant']=='E1' and r['frame'] in [1,30,60]]
    summary=dict(status='E1 authorized scope completed;E2 insufficient-support blocked;pause_for_user',client_date='2026-11-10',client_timezone='America/Los_Angeles',reported_at_host_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        criteria=dict(path=str(CRITERIA.relative_to(ROOT)),commit='4569f66',continuation='276e7cd',application_and_optional_plan='29a6124',unchanged=True),code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        scope=dict(scene='B2-high',frames=[1,30,60],focuses_m=[1.4,5.],fstops=[1.4,2.],K=64,groups_per_focus=6,E1_uses_GT_poses=True,claims_conditional_on_accurate_poses=True,E2_all_method_results_not_completed=True,network_weights_trained=False),
        flow=analysis['flow_runtime'],flow_model_licenses=PREFIX+'-model-licenses.json',E1_calibration=dict(frames13_completed=True,representative_depth=representative,cross_frame_scale=analysis['cross_frame_scale']['E1'],all_heldout_errors_not_worse=True),
        E2_block=dict(reason=cal['stop_reason'],analysis_record=PREFIX+'-E2-block-analysis.json',far_stratum_analysis=analysis['E2_far_diagnosis'],frame6=e2['same_frame_support'],no_threshold_or_bin_or_algorithm_change=True),
        main_pooled=ev['pooled'],per_aperture=ev['per_aperture'],decisions=ev['decisions'],ladder=ev['ladder'],wrong_hidden_pooled=tagpool,cracks_pooled=cracks,
        anchoring=[{k:f[k] for k in ['geometry','frame','method','wall_seconds','peak_allocated_MiB','color_changed_fraction','opacity_max_abs_delta','loss_mask_valid_pixels','valid_SSIM_windows']} for f in fits],
        optional6=dict(completed=True,quality=random['pooled'],single_draw_variance=random['single_draw_variance'],cost=cost,descriptive_not_main_gate=True,edge_downweight_variant_not_executed='only full exclusion was preregistered'),
        observations=['Dense flow provides near/far reference support and improves heldout errors for all13E1frames. E1 assumes accurate GTposes; this is not an estimated-pose solution.',
            'E1 calibration recovers much of previous far-focus retrieval loss, but additive hidden retrieval P3 still fails at both focuses.',
            'True hidden GT reference helps after visible geometry calibration; GT-geometry rule retrieval still loses quality in several regions, so depthaccuracy alone does not finish visibility/aggregation/appearance/dynamic handling.',
            'Full edge exclusion improves far-focus recoverable PSNR but fails direction/guardrails and loses in-focus accuracy; this test does not establish the mixed-edge-color hypothesis as the sole cause.',
            'Optional random physicalK16 lowers measured cost but has integration variance and lower expected single-draw quality thanK64; no inference NI or automatic adoption claim.'],
        boundaries=['one controlledscene,3frames,6groups/focus;no cross-scene claim','E1 usesGTposes,calibration and retrieval conclusions require accurate poses','Oracle ladder descriptive,not methodresults or guaranteed attainablebound','coloranchoring updates outputattributes only,targetRGBonly','GT restricted to poseconditioning,focusclick,declaredOracle andevaluation/diagnostics; noGTdepth inflowfilter/calibration/edgedetection','SHARP research-only; originalresults/frozen decisions unchanged'],
        resources=dict(data_disk=space('/root/autodl-tmp'),system_disk=space('/'),output_bytes=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()),GPU_stages_serial=True,no_new_environment_installation=True),
        files=dict(output=str(OUT),tables=PREFIX+'-tables.csv',ladder=PREFIX+'-ladder.csv',point_table=PREFIX+'-triangulation.csv',geometry=PREFIX+'-geometry-evaluation.json',diagnostics=PREFIX+'-diagnostics.json',optional6=PREFIX+'-random16-evaluation.json',visualizations=str(OUT/'evaluation')),
        next_step='Pause for user. Keep learned GS+anchoring visible candidate; hidden retrieval needs reliability/aggregation/visibility work under accurateposes. E2 pose improvement is a separate future authorized round,not retriedhere.')
    write(RECORDS/(PREFIX+'-summary.json'),summary);print('S1C_SUMMARY_WRITTEN',flush=True)


if __name__=='__main__':main()
