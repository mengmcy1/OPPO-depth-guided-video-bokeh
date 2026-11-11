"""Summarize completed S1b branches while preserving the calibration stop."""
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from route3_sharp_s1b import OUT,PREFIX,CRITERIA,write
from route3_upper_bound_evaluate import RECORDS


def load(suffix):return json.loads((RECORDS/(PREFIX+'-'+suffix+'.json')).read_text())


def main():
    ev=load('native-evaluation');res=load('residual-evaluation');temp=load('temporal-evaluation');cal=load('calibration');zero=load('zero-diagnostics')
    fits=[load(f'anchor-native-f{f:03d}-{m}') for f in [1,30,60] for m in ['A_color','A_color_opacity']]
    cost={};recommendations={}
    for cloud in ['O5','O6-GT-rec']:
        cost[cloud]={};recommendations[cloud]={}
        for mode in ['fixed64','fixed16','fixed16_residual','stratified16_residual_seed2026111123']:
            rows=[r for r in res['profiles'] if r['cloud']==cloud and r['mode']==mode]
            cost[cloud][mode]=dict(profile_cases=len(rows),forward_seconds_median=float(np.median([r['median_forward_seconds'] for r in rows])),
                backward_seconds_median=float(np.median([r['median_backward_seconds'] for r in rows])),total_seconds_median=float(np.median([r['median_total_seconds'] for r in rows])),
                max_peak_allocated_MiB=max(r['peak_allocated_MiB'] for r in rows),max_incremental_peak_MiB=max(r['incremental_peak_MiB'] for r in rows),all_attribute_gradients_finite=all(r['all_attribute_gradients_finite'] for r in rows))
        for focus,by_mode in res['decisions'][cloud].items():
            recommendations[cloud][focus]={}
            for mode,v in by_mode.items():
                profile=mode.replace('_expected','_seed2026111123')
                lens_suffix='-F'+format(float(focus),'g')
                candidate=[r['median_total_seconds'] for r in res['profiles'] if r['cloud']==cloud and r['mode']==profile and r['id'].endswith(lens_suffix)]
                reference=[r['median_total_seconds'] for r in res['profiles'] if r['cloud']==cloud and r['mode']=='fixed64' and r['id'].endswith(lens_suffix)]
                efficient=float(np.median(candidate))<float(np.median(reference))
                recommendations[cloud][focus][mode]=dict(quality_noninferior=v['noninferior'],all5_gradients_finite=cost[cloud][profile]['all_attribute_gradients_finite'],measured_cost_lower=efficient,
                    candidate_total_seconds_median=float(np.median(candidate)),fixed64_total_seconds_median=float(np.median(reference)),
                    eligible_under_registered_rule=bool(v['noninferior'] and efficient and cost[cloud][profile]['all_attribute_gradients_finite']))
    temporal={focus:dict(A_color_to_full_RMSE_ratio=v['A_color_to_full_RMSE_ratio'],diagnostic_le_1p05=v['diagnostic_le_1p05'],
        methods={m:dict(PSNR_dB=a['pooled']['psnr'],patch_LPIPS=a['pooled']['patch_lpips'],temporal_change_RMSE=a['temporal_change_RMSE']) for m,a in v['methods'].items()}) for focus,v in temp['focus_groups'].items()}
    def space(path):
        v=os.statvfs(path);return dict(available_bytes=v.f_bavail*v.f_frsize,available_GiB=v.f_bavail*v.f_frsize/2**30)
    runtime={}
    for mode in ['A_color','A_color_opacity']:
        selected=[r for r in fits if r['variant']==mode]
        runtime[mode]=dict(seconds_per_frame=[r['wall_seconds'] for r in selected],median_seconds=float(np.median([r['wall_seconds'] for r in selected])),max_peak_allocated_MiB=max(r['peak_allocated_MiB'] for r in selected),
            zero_PSNR_before=[r['rows'][0]['input_PSNR_dB'] for r in selected],zero_PSNR_after=[r['rows'][-1]['input_PSNR_dB'] for r in selected],changed_Gaussian_fraction=[r['color_changed_fraction'] for r in selected],
            Gaussian_RGB_max_delta_quantiles=[r['Gaussian_RGB_max_delta_quantiles'] for r in selected],opacity_max_abs_delta=[r['opacity_max_abs_delta'] for r in selected])
    output_bytes=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
    summary=dict(status='authorized_native_branches_completed;calibration_paused;pause_for_user',reported_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        criteria=dict(path=str(CRITERIA.relative_to(ROOT)),initial_commit='9dc09ef',user_temporal_exception_commit='36011d7',native_continuation_commit='62516ca',unchanged_after_scoring=True),
        implementation_commit='5c25d6d',current_report_code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        scope=dict(scene='B2-high',representative_frames=[1,30,60],focuses_m=[1.4,5.],main_fstops=[1.4,2.],K=64,groups_per_focus=6,calibrated_results_not_computed=True,network_weights_updated=False),
        versions=dict(official_SHARP_revision='aed6527499ef91cba3b54c18d49a870f25947190',SHARP_weights='models/SHARP/sharp_2572gikvuh.pt',license='research-only; S1 record unchanged',
            route3='Python3.10.22 torch2.7.1+cu128 gsplat1.5.0',O3='dofgaussian Python3.9.25 torch2.1.2+cu121',LaMa='magicbokeh Python3.10.21 torch2.8.0+cu128',Blender='4.5.3 LTS portable',GPU='RTX4090D 49140MiB GPU0 driver580.105.08'),
        main_pooled=ev['pooled'],main_per_aperture=ev['per_aperture'],P1_P2_decisions=ev['decisions'],hidden_uncalibrated_descriptive=ev['hidden_descriptive'],
        color_anchor=runtime,zero_partition_scores=zero['zero'],
        calibration=dict(status='paused by frozen support check;user says no retry',stop_reason=cal['stop_reason'],E1_fitted_frames=0,E2_attempted=False,P3='not completed because calibration paused',P4='not completed because calibration paused',analysis_record=PREFIX+'-calibration-analysis.json'),
        temporal=temporal,new_temporal_GT=load('temporal-new-GT'),residual_quality=res['pooled'],residual_quality_decisions=res['decisions'],residual_gradient_cost=cost,residual_recommendations=recommendations,
        resource=dict(data_disk=space('/root/autodl-tmp'),system_disk=space('/'),new_output_bytes=output_bytes,GPU_stages_serial=True,GPU_checks_before_each_job=True,
            CPU_quality_scoring_can_overlap_GPU_rendering=True,all_new_cache_and_products_on_data_disk=True,no_environment_install_or_global_configuration_change=True),
        files=dict(output=str(OUT),main_metrics=PREFIX+'-native-evaluation.json',main_tables=PREFIX+'-native-tables.csv',hidden_descriptive=PREFIX+'-hidden-descriptive.csv',
            zero=PREFIX+'-zero-diagnostics.json',fly_opacity=PREFIX+'-fly-opacity.json',edges=PREFIX+'-edge-artifacts.json',temporal=PREFIX+'-temporal-evaluation.json',residual=PREFIX+'-residual-evaluation.json',
            validation=PREFIX+'-validation.json',residual_transparency=PREFIX+'-residual-artifacts.json',visualizations=dict(main=str(OUT/'evaluation-native'),foreground_edges=str(OUT/'edge-artifacts'),temporal=str(OUT/'temporal/evaluation'),residual=str(OUT/'residual-hoistedcov-evaluation'))),
        observations=['Target-only A-color passes same-depth O3 P1 at both focuses under both references; no geometry or network update is required for this local evidence.',
            'Native uncalibrated retrieval still hurts far-focus quality; P3/P4 were not run, so this does not isolate geometry calibration from retrieval aggregation.',
            'Temporal change-error includes motion, geometry, per-frame focus selection and appearance; it is not an isolated flicker test.',
            'All tested K16 candidates fail the registered <=0.2dB NI gate in at least one required region. Fixed residual lowers forward+backward cost but far-focus foreground transparency artifacts are observed; retain K64 for this protocol.'],
        boundaries=['One controlled scene,3representative frames,6lens groups per focus; no cross-scene generalization claim.',
            'Output-parameter test-time color optimization uses target RGB; it is not network training or proof a feed-forward model reaches these scores.',
            'GT scoring/focus-click masks allowed; no GTdepth used to fit calibration or color.',
            'Transparency and intermediate-depth tags are diagnostic proxies,not certified crack/outlier GT.',
            'All old results and frozen decisions remain unchanged; SHARP is research-only.'],
        next_step='Pause for user. Supported candidate: learned explicit full visible representation with pixel anchoring. Calibration alternatives are estimates only; a new calibration/training round requires user decision.')
    write(RECORDS/(PREFIX+'-summary.json'),summary)
    print('S1B_REPORT',recommendations,flush=True)


if __name__=='__main__':main()
