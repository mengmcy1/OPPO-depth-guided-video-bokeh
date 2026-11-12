"""S2a formal mesh labels; reuse existing screening geometry, never overwrite it.

CPU Embree from existing Blender-exported meshes; independently checked against
native Blender Position/Z/Index/AOV passes. Never consumes predicted geometry.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
from route3_blender_b2_full_geometry import FullWorld,save_npz,save_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scene',type=Path,required=True)
    p.add_argument('--mode',choices=['center','aperture','training_focus'],required=True);a=p.parse_args()
    out=a.scene/'geometry';out.mkdir(exist_ok=True);w=FullWorld(a.scene/'scene-geometry.npz',512)
    rows=[];record=out/(a.mode+'-summary.json')
    if record.exists():rows=json.loads(record.read_text())
    if a.mode=='center':
        for frame in w.d['input_frames']:
            path=out/f'f{frame:03d}-center.npz'
            if path.exists():continue
            t=time.monotonic();v=w.center_full(int(frame),16);v['world_position']=w.world(v['local_surface_point'].reshape(-1,3),v['object_id'].reshape(-1),int(frame)).reshape(512,512,3).astype(np.float32)
            save_npz(path,v);save_npz(out/f'f{frame:03d}-background.npz',w.center_full(int(frame),1,tuple(w.fg.tolist())))
            row=dict(frame=int(frame),seconds=time.monotonic()-t,valid_pixels=int(v['center_visible'].sum()),static=True);rows.append(row);save_json(record,rows);print('S2A_CENTER',row,flush=True)
    elif a.mode=='training_focus':
        plan=[]
        for frame in [1,30,60]:
            current=np.load(out/f'f{frame:03d}-center.npz');depth=current['depth_axial_m'][current['center_visible']];low,high=float(depth.min()),float(depth.max());centers=(np.arange(4)+.5)/4;focuses=1/((1-centers)/low+centers/high);cases=[]
            for focus in focuses:
                key=f'f{frame:03d}-N1.4-F{focus:.6f}';path=out/(key+'-K16.npz')
                if not path.exists():
                    values,row=w.aperture_full(frame,16,1.4,float(focus));values['world_position']=w.world(values['local_surface_point'].reshape(-1,3),values['object_id'].reshape(-1),frame).reshape(16,512,512,3).astype(np.float32);save_npz(path,values);row['key']=key;rows.append(row);save_json(record,rows);print('S2A_TRAIN_FOCUS',key,row['elapsed_seconds'],flush=True)
                cases.append(dict(id=key,frame=frame,focus_m=float(focus),nominal_focus_m=float(focus),fstop=1.4,training_focus_only=True))
            plan.append(dict(frame=frame,GT_visible_depth_range_m=[low,high],cases=cases))
        save_json(out/'training-focus-plan.json',dict(rows=plan,K=16,initial_training_focus_count=4,selection='centersof4equalinverse-depthbinsinthisframevisibleGTdepthrange',GT_label_generation_only=True,sample_training_only_from_precomputed_focuses=True,additional_continuous_random_focus_requires_recomputed_ray_labels=True,scheme_update='688bc47'))
    else:
        requirements={}
        for frame in [1,30,60]:
            for focus in [1.4,5.]:
                for fstop in [1.4,2.]:
                    key=f'f{frame:03d}-N{fstop:g}-F{focus:g}-K16';path=out/(key+'.npz')
                    if path.exists():
                        row=next((x for x in rows if x['key']==key),None)
                        if row is None:row=dict(key=key,**json.loads(path.with_suffix('.json').read_text()));rows.append(row);save_json(record,rows)
                    else:
                        values,row=w.aperture_full(frame,16,fstop,focus)
                        values['world_position']=w.world(values['local_surface_point'].reshape(-1,3),values['object_id'].reshape(-1),frame).reshape(16,512,512,3).astype(np.float32)
                        save_npz(path,values);row['key']=key;rows.append(row);save_json(record,rows);print('S2A_APERTURE',key,row['elapsed_seconds'],row['effective_recoverable_total_pixels'],flush=True)
                    requirements.setdefault(str(frame),set()).update(tuple(x) for x in row['required_exclusion_sets'])
        save_json(out/'required-exclusions.json',dict(frames={k:[list(x) for x in sorted(v)] for k,v in requirements.items()}))
        # Center-camera reference geometry for every first/second peel subset.
        for frame,sets in requirements.items():
            for ids in sets:
                path=out/(f'f{int(frame):03d}-exclude-'+'-'.join(map(str,ids))+'.npz')
                if not path.exists():save_npz(path,w.center_full(int(frame),1,ids))


if __name__=='__main__':main()
