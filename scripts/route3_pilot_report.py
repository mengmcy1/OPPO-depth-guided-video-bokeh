"""Pilot data inventory, input/label checks, cost and first-step report."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from route3_pilot_fields import OUT,RECORDS,PREFIX,read,write
from src.candidate_query_data import observation_flat_index


def load(path):return json.loads(path.read_text())


def depth_stats(pred,gt,mask):
    use=mask&np.isfinite(pred)&(pred>0)&np.isfinite(gt)&(gt>0);x=pred[use];y=gt[use]
    return dict(pixels=int(use.sum()),AbsRel=float(np.mean(abs(x-y)/y)),RMSE_m=float(np.sqrt(np.mean((x-y)**2))),median_scale_ratio=float(np.median(x/y))) if len(x) else dict(pixels=0)


def geometry():
    rows=[]
    for label in ['pilot-thin','pilot-multilayer']:
        s=OUT/label;cal=load(s/'models/calibration/summary.json')
        for f in [1,6,11,16,21,26,30,35,40,45,50,55,60]:
            g=np.load(s/'geometry'/f'f{f:03d}-center.npz');gt=g['depth_axial_m'];known=g['center_visible'];inv=1/np.maximum(gt,1e-6)
            import cv2
            edge=(cv2.dilate(inv,np.ones((3,3),np.uint8))-cv2.erode(inv,np.ones((3,3),np.uint8)))>.03*cv2.dilate(inv,np.ones((3,3),np.uint8));edge=cv2.dilate(edge.astype(np.uint8),np.ones((7,7),np.uint8))>0;ref=np.load(s/'models/calibration'/f'f{f:03d}-reference.npz');r=dict(scene=label,frame=f,triangulated_reference=depth_stats(ref['depth'],gt,known&ref['valid']),estimated={})
            for source in ['SHARP','DA3']:
                native=np.load(s/'models'/source/f'f{f:03d}.npz')['depth'];mapped=np.load(s/'models/calibration'/source/f'f{f:03d}.npz')['depth'];r['estimated'][source]={name:dict(before=depth_stats(native,gt,mask),after=depth_stats(mapped,gt,mask)) for name,mask in dict(full=known,boundary=known&edge,near=known&(gt<2.5),middle=known&(gt>=2.5)&(gt<4),far=known&(gt>=4)).items()};row=next(x for x in cal['rows'] if x['frame']==f)['models'][source];r['estimated'][source]['identity_fallback']=row.get('fit_rejected_and_native_depth_retained',False)
            rows.append(r)
    write(RECORDS/(PREFIX+'-geometry.json'),dict(rows=rows,GT_only_after_inference_and_calibration=True,calibration_identity_fallback_explicit=True,E1_GTposes=True))


def interfaces():
    inventories=[];validation=[]
    for label in ['pilot-thin','pilot-multilayer']:
        s=OUT/label;entries=load(s/'prepared.json')['rows'];inputs=[];targets=[]
        for f in [1,6,11,16,21,26,30,35,40,45,50,55,60]:
            inputs.append(dict(frame=f,linear_EXR=str(s/'inputs'/f'f{f:03d}.exr'),display_RGB=str(s/'inputs'/f'f{f:03d}.png'),native_firsthit_world_position=str(s/'input-passes-v2'/f'f{f:03d}-Position-{f:04d}.exr'),native_object_ID=str(s/'input-passes-v2'/f'f{f:03d}-IndexOB-{f:04d}.exr'),native_object_local_AOV=str(s/'input-passes-v2'/f'f{f:03d}-ObjectLocal-{f:04d}.exr'),pinhole_AA_Cryptomatte=str(s/'inputs'/f'f{f:03d}-crypto-{f:04d}.exr'),GT_geometry=str(s/'geometry'/f'f{f:03d}-center.npz'),SHARP_depth=str(s/'models/SHARP'/f'f{f:03d}.npz'),DA3_depth=str(s/'models/DA3'/f'f{f:03d}.npz'),calibration_SHARP=str(s/'models/calibration/SHARP'/f'f{f:03d}.npz'),calibration_DA3=str(s/'models/calibration/DA3'/f'f{f:03d}.npz')))
        for e in entries:
            pool=np.load(e['pool']);q=np.load(Path(e['pool']).with_name('query-M8.npz'));mem=np.load(s/'memory/verified-index.npz');index=observation_flat_index(pool['slot'],pool['source_xy']);evidence=np.load(Path(e['pool']).with_name('memory-evidence.npz'));assert np.array_equal(evidence['memory_entry_id'],mem['observation_entry_id'][index]);assert not any(k.startswith('GT') for k in pool.files);assert not np.any(pool['source_frame_id']==e['frame']);assert q['candidate_indices'].shape[1]==8;assert all(Path(x['path']).is_file() for x in load(s/'predictions.json')['rows'])
            appearance=np.load(s/'labels'/f'f{e["frame"]:03d}'/'appearance-targets.npz');support=np.load(s/'labels'/f'f{e["frame"]:03d}'/'placement-support-v2.npz');assert len(appearance['valid'])==len(pool['depth'])==len(support['candidate_placement_valid'])
            feature_path=Path(e['pool']).with_name('query-M8-features.npz');features=np.load(feature_path);assert features['features'].shape==(*q['candidate_indices'].shape,19) and np.isfinite(features['features']).all()
            qmem=Path(e['pool']).with_name('query-M8-memory-evidence.npz');safe_index=np.maximum(q['candidate_indices'],0)
            if not qmem.exists():np.savez_compressed(qmem,memory_verified=q['candidate_valid']&evidence['verified'][safe_index],memory_entry_id=np.where(q['candidate_valid'],evidence['memory_entry_id'][safe_index],-1))
            n=len(pool['depth']);source_sample=np.random.default_rng(20261112+e['frame']).choice(n,min(5000,n),replace=False)
            for slot,f in enumerate([1,6,11,16,21,26,30,35,40,45,50,55,60]):
                at=source_sample[pool['slot'][source_sample]==slot];xy=pool['source_xy'][at];rgb=np.load(s/'models/SHARP'/f'f{f:03d}.npz')['rgb_linear'];assert np.array_equal(pool['colors'][at],rgb[xy[:,1],xy[:,0]])
            for case in e['cases']:
                lp=s/'labels'/f'f{e["frame"]:03d}'/(case['id']+'.npz');lab=np.load(lp);bits=lab['content_hit_bits'];pop=np.unpackbits(bits.view(np.uint8).reshape(-1,2),axis=1).sum(1);assert np.allclose(pop/16,lab['content_hit_fraction']);assert lab['placement_depth'].shape==(16,512,512)
                targets.append(dict(**case,GT_DOF=str(s/'dof'/(case['id']+'.exr')),native_K16_world_pass_pattern=str(s/'aperture-passes-v2'/(case['id']+'-kXX-Position-'+f'{e["frame"]:04d}'+'.exr')),GT_visibility=str(s/'geometry'/(case['id']+'-K16.npz')),candidate_pool=e['pool'],query_M8=str(Path(e['pool']).with_name('query-M8.npz')),query_features=str(feature_path),query_memory_evidence=str(qmem),labels=str(lp),appearance_labels=str(s/'labels'/f'f{e["frame"]:03d}'/'appearance-targets.npz'),placement_validity=str(s/'labels'/f'f{e["frame"]:03d}'/'placement-support-v2.npz'),visible_GS=str(s/'visible'/f'f{e["frame"]:03d}.pt'),label_files_must_not_enter_model_inputs=True))
            validation.append(dict(scene=label,frame=e['frame'],candidates=n,query_shape=list(q['candidate_indices'].shape),RGB_original_source_exact=True,source_references_int64_correct=True,target_frame_excluded=True,GT_absent_from_input_pool=True,content_bits_match_hit_fraction=True))
        training_plan=load(s/'geometry/training-focus-plan.json');training_targets=[]
        for group in training_plan['rows']:
            for case in group['cases']:
                path=s/'labels-training-focus'/f'f{group["frame"]:03d}'/(case['id']+'.npz');assert path.is_file();lab=np.load(path);bits=lab['content_hit_bits'];pop=np.unpackbits(bits.view(np.uint8).reshape(-1,2),axis=1).sum(1);assert np.allclose(pop/16,lab['content_hit_fraction']);training_targets.append(dict(**case,labels=str(path),GT_mesh_ray_hit=str(s/'geometry'/(case['id']+'-K16.npz')),appearance_and_placement_validity_shared_with_evaluation_labels=True,labels_depend_on_focus=True,not_a_new_DOF_render=True))
        assert len(training_targets)==12
        fields=load(s/'field-checks.json');assert len(fields['center_native_passes'])==13 and len(fields['aperture_native_passes'])==192
        assert max(x['world_error_same_ID_p50_p99_max_m'][1] for x in fields['aperture_native_passes'])<1e-3
        assert max(x['local_error_same_ID_p99_m'] for x in fields['center_native_passes'])<1e-3
        paths=[p for p in s.rglob('*') if p.is_file()];sizes={name:sum(p.stat().st_size for p in (s/name).rglob('*') if p.is_file()) if (s/name).exists() else 0 for name in ['inputs','input-passes-v2','aperture-passes-v2','geometry','models','candidates','memory','labels','labels-training-focus','oracle','references','visible','dof','continuous','rack','predictions','LDR-subviews','LDR-reference','aperture-passes','labels-query-conditioned-v1']}
        # Do not automatically copy large rendering/diagnostic caches to a trainer.
        training=sum(sizes[n] for n in ['inputs','models','candidates','labels','labels-training-focus','visible','dof'])+sum((s/'memory'/n).stat().st_size for n in ['observations.npz','verified-index.npz','verification.npz'])
        inventory=dict(scene=label,scene_metadata=load(s/'scene.json'),split='pipeline-development;not an untouched publication test scene',inputs=inputs,targets=targets,training_focus_targets=training_targets,training_focus_plan=training_plan,continuous_GT=[str(s/'continuous'/f'f{f:03d}.exr') for f in range(27,35)],focus_rack_GT=[dict(frame=f,focus_m=1/(1/1.4+(f-27)/7*(1/5-1/1.4)),path=str(s/'rack'/f'f{f:03d}.exr')) for f in range(27,35)],foreground_only_trial=str(s/'preflight/foreground.exr'),background_only_trial=str(s/'preflight/background.exr'),long_term_memory=str(s/'memory/verified-index.npz'),raw_observations=str(s/'memory/observations.npz'),optional_feature_dimension=0,tensor_schema=dict(input_RGB='13x512x512x3',depth='13x512x512',poses='13x4x4',candidate_RGB_patch='Nx3x3x3 uint8',candidate_world_position='Nx3 float32,estimated',source_reference='Nx(frame,source_x,source_y)',query='Qx8 indices+valid;Q<=32768',query_features='Qx8x19 float32+missing flags;finite',query_memory_evidence='Qx8 verified + memory entryID',lens_condition='estimatedfocus_m + aperture_radius_m suppliedpercase',content_hit_weight='Nx1 float32;K16 bitset',placement_ray_depth='16x512x512 float32',candidate_placement='Nx3 alignedcameraXYZ,labels only',hidden_RGB_label='Nx3 float32+valid,labels only',optional_feature='entriesx0',future_hidden_outputs='Qx2xGSattributes,no decoder implemented'),S2a_interface=dict(resolution=256,input_frames=[1,16,30,45,60],intrinsics_scale=.5,pixel_CoC_scale=.5,projected_footprint_scale=.5,five_frame_recoverability_requires_fresh_recomputation=True,not_computed_or_trained_in_step1=True),disk_bytes=sum(p.stat().st_size for p in paths),disk_by_component=sizes,conservative_training_sync_bytes=training,invalid_development_outputs_excluded=['aperture-passes/','labels-query-conditioned-v1/','memory/memory.npz','memory-evidence-int16-invalid-v1.npz'],no_neural_network_implemented=True,no_network_training=True)
        write(s/'dataset-manifest.json',inventory);inventories.append(inventory)
    write(RECORDS/(PREFIX+'-data-inventory.json'),dict(rows=inventories,B2_high_excluded_from_training=True,formal_future_holdout='reserve3-4new independent asset combinations;pilotsarealreadyinspected'))
    repeated=[];noise=load(RECORDS/(PREFIX+'-sampling.json'))['rows']
    for label in ['pilot-thin','pilot-multilayer']:
        diff=abs(read(OUT/label/'dof/f030-N1.4-F5.exr')[...,:3]-read(OUT/label/'continuous/f030.exr')[...,:3]);rms=float(np.sqrt(np.mean(diff.astype(np.float64)**2)));n=next(x['linear_noise_RMSE'] for x in noise if x['scene']==label and x['focus_m']==5.);assert rms<.1*n;repeated.append(dict(scene=label,max_linear_error=float(diff.max()),RMSE=rms,independent_seed_linear_RMSE=n,ratio_to_independent_seed_noise=rms/n,exact_bitwise_equal=False))
    write(RECORDS/(PREFIX+'-validation.json'),dict(passed=True,passed_means='interfaces/nativepasses/provenance checked,notallestimatedgeometryorOraclequalitypassed',rows=validation,field_crosschecks='nativeBlenderfirsthit-vsindependentEmbree;allK16configs',unit_tests='4 focused source-reference/memory/query/finite-evidence contracts passed',continuous_truth_consistency=repeated,placement_loss_requires_content_ray_and_safe_v2_mask=True,old_experiment_outputs_unchanged=True,no_network_training=True,no_network_implementation=True,phase2_not_started=True))


def c2_diagnostics():
    import cv2
    from route3_upper_bound_evaluate import display
    rows=[]
    for label in ['pilot-thin','pilot-multilayer','B2-high']:
        if label=='B2-high':
            geometry=ROOT/'outputs/route3-b2only-2026-11-10-01/B2-high/geometry';meta=np.load(ROOT/'outputs/route3-a2b-2026-11-10-01/blender-b2/B2-high/scene-geometry.npz');entry=next(x for x in load(ROOT/'outputs/route3-sharp-s1c-2026-11-10-01/prepared.json')['rows'] if x['geometry']=='cal_E1' and x['frame']==30)
        else:geometry=OUT/label/'geometry';meta=np.load(OUT/label/'scene-geometry.npz');entry=next(x for x in load(OUT/label/'prepared.json')['rows'] if x['frame']==30)
        g=np.load(geometry/'f030-center.npz');background=np.load(geometry/'f030-background.npz');d=np.load(entry['input_npz']);mask=cv2.imread(str(OUT/'C2'/label/'mask/0006.png'),0)>0;effective=cv2.dilate(mask.astype(np.uint8),np.ones((9,9),np.uint8))>0;foreground=np.isin(g['object_id'],meta['foreground_object_ids']);both=(foreground|effective)&g['center_visible'];bgvalid=background['center_visible']&effective
        row=dict(scene=label,frame=30,mask_fraction=float(effective.mean()),foreground_mask_IoU=float((foreground&effective).sum()/max(both.sum(),1)),foreground_mask_recall=float((foreground&effective).sum()/max(foreground.sum(),1)),foreground_mask_precision=float((foreground&effective).sum()/max(effective.sum(),1)),extension_depth_error_vs_GT_background=depth_stats(d['background_depth'],background['depth_axial_m'],bgvalid),GT_only_after_predictions=True)
        if label!='B2-high':
            pred=cv2.imread(str(OUT/'C2'/label/'results/input/frames/0006.png'))[...,::-1].astype(np.float32)/255;truth=display(read(OUT/label/'preflight/background.exr')[...,:3]);mse=float(np.mean((pred[foreground].astype(np.float64)-truth[foreground])**2));row['inpaint_RGB_vs_peeled_background_in_GT_foreground_PSNR']=float(-10*np.log10(max(mse,1e-12)))
        rows.append(row)
    write(RECORDS/(PREFIX+'-C2-diagnostics.json'),dict(rows=rows,interpretation='baseline includesinput-depthmask andS1ddepth-extensionerrors;notanisolatedtestofProPainterRGBability;nomask/depthtuningafterqualityscores'))


def summary():
    inventory=load(RECORDS/(PREFIX+'-data-inventory.json'));groups={name:load(OUT/(name+'-jobs-results.json')) for name in ['formal','aperture-corrected','center-data','peel','preflight','ldr-subviews']};cost=[]
    for row in inventory['rows']:
        label=row['scene'];counts={name:dict(jobs=sum(j['scene']==label for j in jobs),seconds=sum(j['seconds'] for j in jobs if j['scene']==label),bytes=sum(j['bytes'] for j in jobs if j['scene']==label)) for name,jobs in groups.items()};old=sum(j['seconds'] for j in groups['formal'] if j['scene']==label and 'aperture_xy' in j);canonical=sum(v['seconds'] for v in counts.values())-old
        labels=load(OUT/label/'labels/summary.json');legacy=labels.pop('memory')
        cost.append(dict(scene=label,render_stages=counts,valid_render_seconds=canonical,invalid_animated_camera_render_seconds_retained=old,total_render_seconds_including_development_attempts=canonical+old,disk_bytes=row['disk_bytes'],training_sync_bytes=row['conservative_training_sync_bytes'],SHARP_inference=load(OUT/label/'models/SHARP.json'),DA3_inference=load(OUT/label/'models/DA3.json'),color_anchor=load(OUT/label/'visible/summary.json'),memory=load(OUT/label/'memory/verified-summary.json'),memory_GT=load(OUT/label/'memory/GT-evaluation.json'),labels=labels,legacy_unverified_index_diagnostic_not_active=legacy,training_focus_labels=load(OUT/label/'labels-training-focus/summary.json'),placement_validity=load(OUT/label/'labels/placement-summary.json'),placement_safety=load(OUT/label/'labels/placement-safety-summary.json'),appearance_labels=load(OUT/label/'labels/appearance-summary.json')))
    seconds=max(c['valid_render_seconds'] for c in cost);perdisk=max(c['disk_bytes'] for c in cost);trainingdisk=max(c['training_sync_bytes'] for c in cost)
    record=dict(status='step1_completed_pause_for_user',scope='A1-A2 two staticpilots+A3required/trialfields+A4+B0R1R2/C2baselines;E2primeposecomparisonstarted',code_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),client_date='2026-11-12',client_timezone='America/Los_Angeles',outputs=str(OUT),cost=cost,baseline_Cycles=PREFIX+'-baseline-Cycles.json',baseline_LDR=PREFIX+'-baseline-LDR.json',fields=PREFIX+'-fields.json',sampling=PREFIX+'-sampling.json',geometry=PREFIX+'-geometry.json',licenses=PREFIX+'-model-licenses.json',E2_prime=PREFIX+'-E2-prime.json',validation=PREFIX+'-validation.json',next_scale_recommendation=dict(next_data_pilot_scenes_total='3-5 only after user confirmation',initial_training_scenes=12,fresh_heldout_scenes=4,expand_training_to_24_only_after_first_learning_curve=True,full28scene_cache_exceeds_current_free_disk=perdisk*28>os.statvfs('/root/autodl-tmp').f_bavail*os.statvfs('/root/autodl-tmp').f_frsize,future_training_scene_can_skip_nativeK16trial_and_LDRbenchmark_subviews=True,B2_high_never_train=True,pilot_scenes_are_not_untouched_test_data=True,render_only_16scenes_hours=seconds*16/3600,render_only_28scenes_hours=seconds*28/3600,all_outputs_16scenes_GiB=perdisk*16/2**30,conservative_training_sync_16scenes_GiB=trainingdisk*16/2**30,one_week_render_budget_passed=seconds*28<7*24*3600,cost_not_training_runtime=True,identity_fallback_and_large_alignment_residuals_must_be_reviewed_before_expansion=True),no_network_implemented=True,no_network_training=True,phase2_not_started=True,after_report='pause await user',limitations=['twoproceduralcontrolledstatic scenes;nogeneralizationclaim','E1usesGTposes;E2primecalibration/hiddenbenefitretentionnotcompleted','rawGT hidden depth withestimatedvisiblegeometry isnotaperformanceceiling;alignedlabelreferenceisaseparatediagnostic','contentlabelpixel-footprinttolerance isGTworld/sourcepixelbased,notestimateddepthtolerance','memoryGTduplicateproxy depends2cmcellresolution','one-weekprojectionrender-only;development/IO/calibrationandtrainingseparate'])
    def space(path):
        s=os.statvfs(path);return dict(available_GiB=s.f_bavail*s.f_frsize/2**30)
    record.update(resources=dict(data_disk=space('/root/autodl-tmp'),system_disk=space('/'),GPU_stages_serial=True,no_96GB_training_machine_used=True,no_environment_installation=True),scheme_update_during_execution='688bc47:meshlabels+4inverse-depthtrainingfocusesperframe;nativeK16passesretainedastrialvalidation',data_quality_flags=['thinSHARP10of13affinefitsrejected,nativedepthretainedexplicitly','rawGTdepthwithestimatedvisibleisnotanopticalceiling;negativeor>100percentgapratiosnotclipped','localaffineplacementunsupported/front-conflictingtargetsmaskedwithplacement-support-v2;contentlabelsnotchanged','aligned-allGTparentrenderfailedonthinscene;notanapprovedcompleteupperbound','verifiedmemoryhaslarge2cmGT-cellrepetitionproxy;notallmemoryincorrect','C2includesdepthmask/background-extensionerrors;notanisolatedProPainterqualitytest'],training_readiness='filesandinterfacescompleted;qualityflagsandcorrectionrangesshouldbereviewedasdataengineeringbeforephase2;noneuralnetworktested',new_network_trainable_parameters=0,A4_placement_render_records=[PREFIX+'-A4-placement-render-Cycles.json',PREFIX+'-A4-placement-render-LDR.json'],C2_diagnostics=PREFIX+'-C2-diagnostics.json',environment_record=PREFIX+'-environment.json')
    write(RECORDS/(PREFIX+'-summary.json'),record)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['geometry','interfaces','c2_diagnostics','summary'],required=True);a=p.parse_args();globals()[a.mode]()


if __name__=='__main__':main()
