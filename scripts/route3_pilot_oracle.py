"""GT-only peeled hidden reference, never a candidate/memory source."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
import cv2
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from route3_blender_b2_full_geometry import FullWorld,save_json,save_npz
from route3_b2_oracle_3d import deduplicate


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scene',type=Path,required=True);a=p.parse_args();folder=a.scene;w=FullWorld(folder/'scene-geometry.npz',512);out=folder/'oracle';out.mkdir(exist_ok=True)
    jobs=[j for j in json.loads((folder.parent/'peel-jobs-results.json').read_text()) if j['scene']==folder.name];rows=[]
    for frame in [1,30,60]:
        path=out/f'f{frame:03d}.npz'
        if path.exists():continue
        started=time.monotonic();center=np.load(folder/'geometry'/f'f{frame:03d}-center.npz');parts={k:[] for k in ['pixel','depth','colors','object','local','triangle','priority','reference']};refs=sorted([j for j in jobs if j['frame']==frame],key=lambda j:(len(j['hidden_object_ids']),j['hidden_object_ids']))
        for index,j in enumerate(refs):
            key=f'f{frame:03d}-exclude-'+'-'.join(map(str,j['hidden_object_ids']));g=np.load(folder/'geometry'/(key+'.npz'));good=center['center_visible']&g['center_visible']&(g['depth_axial_m']>center['depth_axial_m']+1e-4);ids=np.flatnonzero(good);rgb=cv2.imread(j['output'],-1)[...,:3][...,::-1].clip(0,1)
            for name,value in dict(pixel=ids,depth=g['depth_axial_m'].ravel()[ids],colors=rgb.reshape(-1,3)[ids],object=g['object_id'].ravel()[ids],local=g['local_surface_point'].reshape(-1,3)[ids],triangle=g['triangle_id'].ravel()[ids],priority=np.full(len(ids),len(j['hidden_object_ids']),np.uint8),reference=np.full(len(ids),index,np.int16)).items():parts[name].append(value)
        values={k:np.concatenate(v) for k,v in parts.items()};keep=deduplicate(values['pixel'],values['depth'],values['priority'],values['reference']);v={k:a[keep] for k,a in values.items()};world=w.world(v['local'],v['object'],frame);recover=np.zeros(len(world),bool);count=np.zeros(len(world),np.uint8)
        for source in w.d['input_frames']:
            if int(source)==frame:continue
            visible,*_=w.visible_at(int(source),int(source),world,v['object']);recover|=visible;count+=visible.astype(np.uint8)
        uv=np.c_[v['pixel']%512+.5,v['pixel']//512+.5].astype(np.float32);save_npz(path,dict(uv=uv,depth=v['depth'],colors_linear=v['colors'],object_id=v['object'],world_position=world.astype(np.float32),local_surface_point=v['local'],triangle_id=v['triangle'],recoverable=recover,visible_other_input_count=count))
        row=dict(frame=frame,parents=len(world),recoverable_parents=int(recover.sum()),references=len(refs),seconds=time.monotonic()-started,GT_Oracle_only=True);rows.append(row);print('PILOT_ORACLE',folder.name,row,flush=True)
    save_json(out/'summary.json',dict(rows=rows,GT_never_used_in_ordinary_candidates_or_memory=True))


if __name__=='__main__':main()
