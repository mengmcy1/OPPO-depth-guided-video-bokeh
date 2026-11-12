"""CPU geometry for the retained formal peel colors; no new RGB rendering."""
import argparse,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from route3_blender_b2_full_geometry import FullWorld,save_npz,save_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',type=Path,required=True);a=p.parse_args();s=a.scene;w=FullWorld(s/'scene-geometry.npz',512);jobs=json.loads((s/'formal-peels-jobs.json').read_text())
    for j in jobs:
        ids=j['hidden_object_ids'];f=j['frame'];key=f'f{f:03d}-exclude-'+ '-'.join(map(str,ids));path=s/'geometry'/(key+'.npz')
        if not path.exists():save_npz(path,w.center_full(f,1,tuple(ids)))
    # Screening's alpha=1 center geometry is unchanged; separately expose alpha16.
    rows=[]
    for f in [1,30,60]:
        path=s/'geometry'/f'f{f:03d}-alpha16.npz'
        if not path.exists():save_npz(path,dict(foreground_alpha=w.center_full(f,16)['foreground_alpha']))
        rows.append(dict(frame=f,path=str(path),previous_center_alpha_sample_count=1,formal_alpha_sample_count=16))
    save_json(s/'geometry/alpha-supplement.json',dict(rows=rows))

if __name__=='__main__':main()
