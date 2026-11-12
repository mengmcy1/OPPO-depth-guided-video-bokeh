"""Existing-output data/headroom inventory, before learned-method evaluation.

These are rendering references and R1+R2 controls, not fusion-network results.
Selection remains the prespecified GT-rec farfocus criterion.
"""
import csv,json,os,sys
from pathlib import Path
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from route3_s2a_screen import OUT,RECORDS,write
from route3_s2a_evaluate import masks
from route3_s2a_proxy import display,read

def main():
    cfg=json.loads((ROOT/'configs/route3-network-s2a-2026-11-12.json').read_text());inventory=[];cells=[];pooled=[]
    for name in cfg['scenes']:
        s=OUT/name
        if not (s/'formal-interface.json').exists():continue
        meta=json.loads((s/'scene.json').read_text());check=json.loads((s/'actual-gain-check.json').read_text()) if(s/'actual-gain-check.json').exists() else None
        if check is None:continue
        entries=json.loads((s/'prepared.json').read_text())['rows'];render_jobs=[]
        for key in ['screen-input','native-data','formal-peels','formal-dof']:
            p=s/(key+'-jobs-results.json')
            if p.exists():render_jobs+=json.loads(p.read_text())
        cal=json.loads((s/'models/calibration/summary.json').read_text())['rows'];cal_by_frame={r['frame']:r['models']['SHARP']['calibration_applied'] for r in cal}
        seen=set();logical=allocated=0
        for p in s.rglob('*'):
            if not p.is_file():continue
            st=p.stat();key=(st.st_dev,st.st_ino)
            if key in seen:continue
            seen.add(key);logical+=st.st_size;allocated+=st.st_blocks*512
        inventory.append(dict(scene=name,seed=meta['seed'],family=meta['family'],low_texture=meta['low_texture'],foreground_depth_design_m=meta['foreground_depth_design_m'],background_depth_design_m=meta['background_depth_design_m'],foreground_background_depth_ratio=meta['foreground_background_depth_ratio'],camera_baseline_design_m=meta['camera_translation_baseline_design_m'],RGB_render_seconds=sum(r['seconds'] for r in render_jobs),RGB_render_jobs=len(render_jobs),pinhole_spp=256,DOF_spp=1024,native_data_spp=1,K_labels=16,logical_file_bytes=logical,allocated_file_bytes=allocated,SHARP_calibration_applied_frames=sum(r['models']['SHARP']['calibration_applied'] for r in cal),native_fallback_frames=sum(r['models']['SHARP']['fit_rejected_and_native_depth_retained'] for r in cal),actual_holdout_qualified=check['passed'],far_f1p4_gains=[r['GTrec_minus_Acolor_dB'] for r in check['rows']],GT_gain_only_used_for_split=True))
        for e in entries:
            for c in e['cases']:
                target=display(read(s/'dof'/(c['id']+'.exr')));regions=masks(s,c)
                for method in ['A_color','R1_R2','GTrec']:
                    pred=np.load(s/'predictions'/(c['id']+'-'+method+'.npz'))['prediction_srgb'];error=((pred.astype(np.float64)-target)**2).sum(-1)
                    for region in ['recoverable','band','in_focus','full_image']:
                        mask=regions[region];n=int(mask.sum());sse=float(error[mask].sum());psnr=float(-10*np.log10(max(sse/(3*n),1e-12))) if n else None;cells.append(dict(scene=name,id=c['id'],frame=e['frame'],focus_m=c['nominal_focus_m'],fstop=c['fstop'],region=region,method=method,pixels=n,sse=sse,PSNR_dB=psnr))
        local=[r for r in cells if r['scene']==name]
        index={(r['id'],r['region'],r['method']):r for r in local}
        for r in local:
            v=index[r['id'],r['region'],'A_color'];g=index[r['id'],r['region'],'GTrec'];rule=index[r['id'],r['region'],'R1_R2'];gain=g['PSNR_dB']-v['PSNR_dB'] if v['PSNR_dB'] is not None else None
            qualified=gain is not None and gain>=1. and r['pixels']>=1000
            r.update(calibration='accepted' if cal_by_frame[r['frame']] else 'rejected-native-retained',GTrec_minus_Acolor_dB=gain,qualified_for_gap_ratio=qualified,MSE_gap_recovery=(v['sse']-r['sse'])/(v['sse']-g['sse']) if qualified else None,PSNR_vs_R1R2_dB=r['PSNR_dB']-rule['PSNR_dB'] if r['PSNR_dB'] is not None else None,focus_guardrail_pass=r['PSNR_dB']-v['PSNR_dB']>=-.2 if r['region']=='in_focus' and r['PSNR_dB'] is not None else None)
        for focus in [1.4,5.]:
            for region in ['recoverable','band','in_focus','full_image']:
                values={}
                for method in ['A_color','R1_R2','GTrec']:
                    chosen=[r for r in local if r['focus_m']==focus and r['region']==region and r['method']==method];n=sum(r['pixels'] for r in chosen);sse=sum(r['sse'] for r in chosen);values[method]=-10*np.log10(max(sse/(3*n),1e-12)) if n else None
                pooled.append(dict(scene=name,focus_m=focus,region=region,Acolor_PSNR_dB=values['A_color'],R1R2_PSNR_dB=values['R1_R2'],GTrec_PSNR_dB=values['GTrec'],R1R2_minus_Acolor_dB=values['R1_R2']-values['A_color'] if values['A_color'] is not None else None,GTrec_minus_Acolor_dB=values['GTrec']-values['A_color'] if values['A_color'] is not None else None))
    result=dict(completed_formal_scenes=len(inventory),planned_formal_scenes=16,qualified_for_holdout=sum(r['actual_holdout_qualified'] for r in inventory),inventory=inventory,cells=cells,pooled=pooled,reference='Cycles',method_comparison='existing A-color and R1+R2 controls and GTrec;no learned fusion scores',holdout_selection_not_changed=True,network_trained=(OUT/'S2a').exists() and any((OUT/'S2a').glob('step-*.pt')),GTrec_not_perfect_optical_upper_bound=True)
    write(RECORDS/'route3-network-s2a-2026-11-12-data-headroom.json',result)
    if cells:
        with (RECORDS/'route3-network-s2a-2026-11-12-data-headroom.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(cells[0]),lineterminator='\n');w.writeheader();w.writerows(cells)
    print('DATA_REPORT',len(inventory),result['qualified_for_holdout'],flush=True)
    for r in pooled:
        if r['focus_m']==5. and r['region']=='recoverable':print(r,flush=True)

if __name__=='__main__':main()
