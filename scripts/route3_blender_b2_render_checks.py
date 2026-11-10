"""Inventory Cycles inputs/GT and test per-object references on unchanged surfaces."""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
import cv2
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from route3_blender_b1_analyze import load_exr


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--record',type=Path,required=True)
    p.add_argument('--references',action='store_true');p.add_argument('--scenes',nargs='+',default=['B2-high','B2-low']);p.add_argument('--allow-pending-geometry',action='store_true');a=p.parse_args();cv2.setNumThreads(1);rows=[]
    result=dict(mode='reference check' if a.references else 'Cycles formal image inventory',scenes={})
    for scene in a.scenes:
        pin=a.root/'inputs'/scene/'pinhole';gt=a.root/'ground-truth'/scene
        pinrows=json.loads((pin/'timings.json').read_text());gtrows=json.loads((gt/'timings.json').read_text())
        if len(pinrows)!=60 or sorted(r['frame'] for r in pinrows)!=list(range(1,61)):raise ValueError('Pinhole frames incomplete: '+scene)
        if len(gtrows)!=18 or any(r['frame'] not in [1,30,60] for r in gtrows):raise ValueError('Representative DOF incomplete: '+scene)
        for folder,images in [(pin,pinrows),(gt,gtrows)]:
            for r in images:
                image=load_exr(folder/(r['key']+'.exr'))
                assert image.shape==(512,512,3) and np.isfinite(image).all()
                assert (folder/(r['key']+'.png')).exists()
        details=dict(pinhole_frames=60,representative_DOF_images=18,representative_frames=[1,30,60],
            pinhole_samples=1024,seed=20261110,primary_fstops=[1.4,2.],pressure_fstop=.5,
            reused_DOF_images=sum(r.get('reused_completed_noise_render',False) for r in gtrows),
            new_pinhole_render_seconds=sum(r['elapsed_seconds_render_and_EXR_write'] for r in pinrows),
            new_DOF_render_seconds=sum(r.get('elapsed_seconds_render_and_EXR_write') or 0 for r in gtrows))
        if a.references:
            refdir=a.root/'references'/scene;refs=json.loads((refdir/'timings.json').read_text());checks=[];pending=[]
            by_frame={r['frame']:r for r in pinrows}
            for r in refs:
                frame=r['frame'];hide=r['hidden_object_ids'];suffix='-'.join(map(str,hide));geometry=a.root/scene/'geometry'
                if not (geometry/f'f{frame:03d}-exclude-{suffix}.npz').exists() and a.allow_pending_geometry:
                    pending.append(dict(frame=frame,hidden_object_ids=hide));continue
                full=np.load(geometry/f'f{frame:03d}-center.npz');reference=np.load(geometry/f'f{frame:03d}-exclude-{suffix}.npz')
                same=(full['triangle_id']==reference['triangle_id'])&(full['object_id']==reference['object_id'])&(full['depth_axial_m']>0)
                same&=np.abs(full['depth_axial_m']-reference['depth_axial_m'])<1e-4
                same=cv2.erode(same.astype(np.uint8),np.ones((7,7),np.uint8))>0
                fullrgb=load_exr(pin/(by_frame[frame]['key']+'.exr'));refrgb=load_exr(refdir/(r['key']+'.exr'))
                diff=refrgb[same]-fullrgb[same];rmse=float(np.sqrt((diff**2).mean())) if same.any() else None
                check=dict(scene=scene,frame=frame,hidden_object_ids=hide,visible_unchanged_pixels=int(same.sum()),
                    linear_RGB_RMSE=rmse,linear_RGB_MAE=float(np.abs(diff).mean()) if same.any() else None,
                    max_abs=float(np.abs(diff).max()) if same.any() else None,
                    threshold_RMSE=.005,passed=bool(same.sum()>=1000 and rmse is not None and rmse<=.005),
                    reference_RGB=str(refdir/(r['key']+'.exr')),reference_geometry=str(geometry/f'f{frame:03d}-exclude-{suffix}.npz'),
                    physical_condition='identical center triangle/object/depth,7x7 erosion; same seed/SPP; only selected objects camera-ray visibility disabled')
                checks.append(check);print('REFERENCE_CHECK',json.dumps(check),flush=True)
            required=json.loads((a.root/scene/'geometry/required-exclusions.json').read_text())
            expected={(int(frame),tuple(ids)) for frame,sets in required['frames'].items() for ids in sets}
            rendered={(r['frame'],tuple(r['hidden_object_ids'])) for r in refs}
            missing=[dict(frame=frame,hidden_object_ids=list(ids)) for frame,ids in sorted(expected-rendered)]
            details['reference_checks']=checks;details['checked_reference_count']=len(checks);details['rendered_reference_count']=len(refs);details['pending_reference_geometry']=pending
            details['required_reference_count']=len(expected);details['missing_required_RGB']=missing
            details['reference_samples']=1024;details['reference_seed']=20261110
            details['reference_render_seconds']=sum(r['elapsed_seconds_render_and_EXR_write'] for r in refs)
        result['scenes'][scene]=details
    result.update(passed=not a.references or all(c['passed'] for s in result['scenes'].values() for c in s['reference_checks']),
        all_outputs_scene_linear_32bit_EXR=True,PNG_preview='Standard display transform',model_training=False,
        full60frame_DOF=False,method_evaluation=False,
        complete=not a.references or all(not s['pending_reference_geometry'] and not s['missing_required_RGB'] for s in result['scenes'].values()),
        reference_RMSE_tolerance='0.005 scene-linear, same full-frame tolerance as independent-noise phase; same seed/SPP and unchanged first-hit geometry expected much closer')
    a.record.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print('RENDER_CHECK_COMPLETE',result['passed'],flush=True)


if __name__=='__main__':main()
