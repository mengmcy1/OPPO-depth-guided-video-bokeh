"""Summarize completed truth checks/T4a; never resumes stopped experiments."""
import csv
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
import cv2
import numpy as np
import torch
from route3_g3_r13 import C, OUT, REC, write
from route3_s2a_proxy import read, display


def table(path, rows):
    if not rows:
        return
    with path.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def alias_report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    reports = []
    dest = OUT.parent/'diagnostics'
    dest.mkdir(exist_ok=True)
    for name in ['metric-scene-122103-thin', 'metric-scene-122106-thin']:
        scene = OUT/name
        manifest = json.loads((scene/'alias/manifest.json').read_text())
        jobs = {r['label']: r for r in manifest['rows']}
        center, phase, noise = [read(jobs[k]['path']) for k in ['center', 'shift', 'noise']]
        ids = read(jobs['center']['passes']['IndexOB'])[..., 0].round().astype(int)
        shifted = cv2.warpAffine(center, np.array([[1, 0, -.5], [0, 1, 0]], np.float32),
                                 (512, 512), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT101)
        phase_mse = np.square(display(phase).astype(np.float64)-display(shifted)).mean(-1)
        noise_mse = np.square(display(noise).astype(np.float64)-display(center)).mean(-1)
        failed = [x for x in json.loads((scene/'surface-checks-r13.json').read_text())['alias'] if not x['passed']]
        failed_ids = [x['id'] for x in manifest['objects'] if x['name'] in {r['object_name'] for r in failed}]
        interior = np.zeros(ids.shape, bool)
        per_object = []
        for oid in failed_ids:
            mask = cv2.erode((ids == oid).astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
            interior |= mask
            per_object.append(dict(object_id=oid,
                object_name=next(x['name'] for x in manifest['objects'] if x['id'] == oid),
                pixels=int(mask.sum()), phase_PSNR=float(-10*np.log10(max(phase_mse[mask].mean(), 1e-30))),
                noise_PSNR=float(-10*np.log10(max(noise_mse[mask].mean(), 1e-30))),
                excess_RMSE=float(max(phase_mse[mask].mean()-noise_mse[mask].mean(), 0)**.5)))
        fig, axes = plt.subplots(1, 4, figsize=(15, 4))
        axes[0].imshow(display(center)); axes[0].set_title('Pinhole input')
        axes[1].imshow(interior, cmap='gray'); axes[1].set_title('Failed object interiors')
        axes[2].imshow(np.sqrt(phase_mse), vmin=0, vmax=.08, cmap='magma'); axes[2].set_title('Phase error RMSE (0 to .08)')
        axes[3].imshow(np.sqrt(noise_mse), vmin=0, vmax=.08, cmap='magma'); axes[3].set_title('Independent noise (same scale)')
        for ax in axes: ax.axis('off')
        fig.suptitle(name+'; BOX1.0,256spp,0.5px principal-point shift')
        fig.tight_layout()
        path = dest/(name+'-phase.png')
        fig.savefig(path, dpi=130); plt.close(fig)
        reports.append(dict(scene=name, failed_objects=per_object,
            max_camera_extrinsic_change=max(r['max_extrinsic_change'] for r in manifest['rows']),
            materials_unchanged=True, thresholds_unchanged=True, figure=str(path),
            observation='phase residual exceeds fixed threshold inside eroded thin-bar objects;period and same-camera repeat pass;not DOF sampling noise',
            causal_limit='does not isolate texture alias from same-object face/shading transitions or Lanczos phase interpolation;no additional render or rule change'))
    return reports


def report():
    prefix = C['record_prefix']
    truth = [json.loads(p.read_text()) for p in sorted(OUT.glob('*/truth-ready-r13.json'))]
    sample_rows, scene_rows = [], []
    for row in truth:
        check = row['surface_checks']
        for level in row['sampling']:
            for s in level['checks']:
                sample_rows.append(dict(scene=row['scene'], spp=level['spp'], **s))
        samples = [s for level in row['sampling'] for s in level['checks']]
        scene_rows.append(dict(scene=row['scene'], seed=row['seed'], split=row['split'],
            final_spp=row['final_spp'], sampling_passed=all(x['passed'] for x in samples),
            convergence_max_patch_LPIPS=max(s['patch_lpips'] for s in samples if s['kind']=='convergence'),
            independent_max_patch_LPIPS=max(s['patch_lpips'] for s in samples if s['kind']=='independent'),
            period_p01_px_min=min(x['period_p01_px'] for x in check['periods'] if x['period_p01_px'] is not None),
            alias_excess_RMSE_max=max(x.get('excess_RMSE', 0) for x in check['alias']),
            zero_repeat_PSNR=check['zero_aperture_repeat_PSNR'], period_pass=check['period_pass'],
            alias_pass=check['alias_pass'], zero_pass=check['zero_pass'], valid=row['passed'],
            truth_stage_wall_seconds=row['seconds'], new_own_files_GiB=row['own_disk_GiB']))
    table(REC/(prefix+'-sampling.csv'), sample_rows)
    table(REC/(prefix+'-truth-scenes.csv'), scene_rows)
    flow = json.loads((REC/(prefix+'-T4-flow.json')).read_text())
    for key in ['rows', 'pooled', 'geometry_gap_recovery']:
        table(REC/(prefix+'-T4a-'+key+'.csv'), flow[key])
    geometry = [json.loads(p.read_text()) for p in (OUT.parent/'T4/flow').glob('*/f*/geometry.json')]
    depth = []
    for name in sorted({x['scene'] for x in geometry}):
        chosen = [x for x in geometry if x['scene'] == name]
        n = sum(x['depth_error']['before']['pixels'] for x in chosen)
        depth.append(dict(scene=name, band_pixels_across3frames=n,
             before_AbsRel=sum(x['depth_error']['before']['AbsRel']*x['depth_error']['before']['pixels'] for x in chosen)/n,
             after_AbsRel=sum(x['depth_error']['after']['AbsRel']*x['depth_error']['after']['pixels'] for x in chosen)/n,
             replaced_pixels=sum(x['replaced_pixels'] for x in chosen),
             expanded_pixels=sum(x['expanded_pixels'] for x in chosen)))
    table(REC/(prefix+'-T4a-depth.csv'), depth)
    summary_table = []
    for name in sorted({x['scene'] for x in geometry}):
        for focus in [1.4, 5.]:
            for region in ['recoverable', 'in_focus', 'full_image', 'band']:
                rows = {x['method']: x for x in flow['pooled'] if x['scene']==name and x['focus']==focus and x['fstop']==1.4 and x['region']==region}
                gap = next(x for x in flow['geometry_gap_recovery'] if x['scene']==name and x['focus']==focus and x['fstop']==1.4 and x['region']==region)
                summary_table.append(dict(scene=name, focus_m=focus, region=region,
                    A_PSNR=rows['A']['psnr'], D0_flow_PSNR=rows['D0']['psnr'],
                    original_D1_PSNR=rows['original_D1']['psnr'], flow_D1_PSNR=rows['D1']['psnr'],
                    GT_D1_PSNR=rows['GT_D1']['psnr'], flow_D1_minus_original_dB=gap['gain_dB'],
                    flow_D1_minus_A_dB=gap['versus_A_dB'], positive_GT_gap=gap['positive_GT_gap'],
                    geometry_gap_recovery=gap['ratio'], raw_formula_ratio=gap['raw_formula_ratio']))
    table(REC/(prefix+'-T4a-main.csv'), summary_table)
    zero_rows = []
    for name in sorted({x['scene'] for x in geometry}):
        for focus in [1.4, 5.]:
            for stop in [1.4, 2.]:
                z = {}
                infocus = {}
                for method in ['A', 'D0', 'D1', 'original_D1', 'GT_D1']:
                    selected = [r for r in flow['rows'] if r['scene']==name and r['focus']==focus
                                and r['fstop']==stop and r['method']==method and r['region']=='full_image']
                    z[method] = float(-10*np.log10(np.mean([10**(-r['zero_PSNR']/10) for r in selected])))
                    infocus[method] = next(r['psnr'] for r in flow['pooled'] if r['scene']==name
                        and r['focus']==focus and r['fstop']==stop and r['method']==method and r['region']=='in_focus')
                zero_rows.append(dict(scene=name, focus_m=focus, fstop=stop,
                    A_zero_PSNR=z['A'], D0_zero_PSNR=z['D0'], D1_zero_PSNR=z['D1'],
                    D0_zero_minus_A_dB=z['D0']-z['A'], D1_zero_minus_A_dB=z['D1']-z['A'],
                    D0_infocus_minus_A_dB=infocus['D0']-infocus['A'],
                    D1_infocus_minus_A_dB=infocus['D1']-infocus['A'],
                    worst_D1_zero_frame_PSNR=min(r['zero_PSNR'] for r in flow['rows'] if r['scene']==name
                        and r['focus']==focus and r['fstop']==stop and r['method']=='D1' and r['region']=='full_image')))
    table(REC/(prefix+'-T4a-zero-guards.csv'), zero_rows)
    timings = [dict(scene=p.parents[2].name, frame=p.parents[1].name, id=p.parent.name,
                   **t) for p in (OUT.parent/'T4/flow').glob('*/f*/*/render.json')
               for t in json.loads(p.read_text())['timings']]
    table(REC/(prefix+'-T4a-timing.csv'), timings)
    aliases = alias_report()
    write(REC/(prefix+'-alias-diagnostics.json'), dict(rows=aliases, render_groups_added=0))
    usage = __import__('shutil').disk_usage(OUT)
    # Exclude immutable file/dir references, count bytes newly allocated by
    # this run's files. No history hashes or data-wide audit is performed.
    disk = sum(p.stat().st_size for p in OUT.parent.rglob('*') if p.is_file() and not p.is_symlink())/2**30
    env = dict(python=platform.python_version(), torch=str(torch.__version__),
               code_revision=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
               Blender='4.5.3 LTS', filter_type='BOX', filter_width=1.,
               GPU='RTX4090D48GB', GPU_index=0,
               data_disk_free_GiB=usage.free/2**30, system_disk_free_GiB=__import__('shutil').disk_usage('/').free/2**30,
               output_new_regular_files_GiB=disk,
               initial_df_available='332G (rounded df -h at run start)',
               new_package_installation=False, new_training_machine_used=False,
               no_old_output_deleted_or_overwritten=True,
               CPU_label_orchestration_stopped_at_batch_failure=True,
               in_flight_CPU_aperture_job_finished_without_new_followup=True,
               GPU_schedule='Own T2 Blender overlapped T4a renderer after capacity/process check;combined observed~5640MiB;not formal efficiency timings',
               GPU_budget=str(OUT.parent/'gpu-budget-T2-T4a.json'))
    write(REC/(prefix+'-environment.json'), env)
    result = dict(status='STOP_T2_BATCH_ALIAS_VALIDITY_FAILURE', protocol_revision='ccfca5d or later;appendix13r13',
        T1=dict(state='fixed;not rerun', order='O1', fallback='F2'), output=str(OUT.parent),
        T2=dict(prespecified_split=C['split'], scenes_rendered=len(truth), all_scenes=16,
                valid_scenes=[x['scene'] for x in scene_rows if x['valid']],
                failed_scenes=[x['scene'] for x in scene_rows if not x['valid']],
                not_generated=[x for x in C['scenes'] if x not in {r['scene'] for r in truth}],
                final_spp={x['scene']:x['final_spp'] for x in truth},
                sampling_all_passed=all(x['sampling_passed'] for x in scene_rows),
                higher_spp_not_needed=True, threshold_changed=False,
                formal_DOF_configurations=12*len(truth), pinhole_inputs=13*len(truth),
                first2_historical_inputs_and_assets_referenced_read_only=True,
                truth_stage_wall_seconds_sum=sum(x['truth_stage_wall_seconds'] for x in scene_rows),
                geometry_independent_K16_state='firstscene13centers+12aperturecases completed;4trainingfoci,peels/oracle/temporal remain pending',
                geometry_dependent_precomputation='not started;separate geometry root reserved',
                alias_diagnostics=aliases),
        T4a=dict(state='completed rendering and scoring on3scenes/3frames/2foci/2apertures,P2',
                 evaluation_rows=len(flow['rows']), reference='Cycles',
                 depth=depth, main_f1p4=summary_table,
                 original_A_and_F2_unchanged=True, P2_scalar_fixed_to_T1=True,
                 GT_reference_uses_GT_selected_focus=True,
                 geometry_gap_formula='dB gain ofnewD1overoriginalD1 / dB GTgap;no clipping;nonpositiveGTgap=N/A,raw ratio retained',
                 render_call_wall_seconds_sum=sum(x['seconds'] for x in timings),
                 timing_scope='D0/D1 calls only;excludes prep,A-sample generation and scoring;GPU contention logged;not formal efficiency benchmark',
                 peak_Pytorch_allocated_MiB=max(x['peak_MiB'] for x in timings),
                 zero_and_infocus_comparison=zero_rows,
                 interpretation='banddepth improves on G2 but not B2;render gains mixed;not a uniform replacement;no results-based stopping gate',
                 full_results=str(REC/(prefix+'-T4-flow.json'))),
        T4b=dict(state='code/license/weights/strict CPU construction complete;GPU inference/render not run before T2stop',
                 strict_state_load='all keys matched;35534721parameters', license_record=str(REC/(prefix+'-model-licenses.json'))),
        T3=dict(overfit='not started', S2a='not started', reason='T2 batch validity stop;dataset not ready', checkpoints_created=0),
        environment=env, stop_condition='other data validity checks fail in batch;2/6phase failures;sampling itself passes',
        conclusions_unchanged=True, next_step='await user decision on failed fixed thin scenes/phase checks;do not loosen thresholds,replace scenes,change textures or start training automatically')
    write(REC/(prefix+'-summary.json'), result)
    print(json.dumps(dict(status=result['status'], free_GiB=env['data_disk_free_GiB'],
                          output_GiB=disk, T4a_peak_MiB=result['T4a']['peak_Pytorch_allocated_MiB']), indent=2))


if __name__ == '__main__':
    report()
