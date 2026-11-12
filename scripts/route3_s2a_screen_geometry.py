"""Mesh-only K16 recoverability and sparse peel plan for pre-DOF screening."""
import argparse,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from route3_blender_b2_full_geometry import FullWorld,save_json,save_npz

def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',type=Path,required=True);a=p.parse_args();s=a.scene;out=s/'geometry';out.mkdir(exist_ok=True);w=FullWorld(s/'scene-geometry.npz',512);requirements={};rows=[]
    for f in [1,30,60]:
        path=out/f'f{f:03d}-center.npz'
        if not path.exists():save_npz(path,w.center_full(f,1))
        requirements[str(f)]=set()
        for focus in [1.4,5.]:
            key=f'f{f:03d}-N1.4-F{focus:g}-K16';path=out/(key+'.npz')
            if path.exists():row=json.loads(path.with_suffix('.json').read_text())
            else:
                g,row=w.aperture_full(f,16,1.4,focus);save_npz(path,g);save_json(path.with_suffix('.json'),row)
            requirements[str(f)].update(tuple(x) for x in row['required_exclusion_sets']);rows.append(dict(key=key,**row));print('SCREEN_MESH',s.name,key,row['effective_recoverable_total_pixels'],flush=True)
    meta=json.loads((s/'scene.json').read_text());names={x['id']:x['name'] for x in meta['objects']};jobs=[]
    for fs,sets in requirements.items():
        for ids in sorted(sets):
            key=f'f{int(fs):03d}-exclude-'+ '-'.join(map(str,ids));jobs.append(dict(id=s.name+'-screen-'+key,scene=s.name,blend=str(s/'scene.blend'),size=512,frame=int(fs),spp=32,seed=20261112,dof=False,hidden_object_ids=list(ids),hide_names=[names[i] for i in ids],output=str(s/'screen-peels'/(key+'.exr'))))
    save_json(s/'screen-peel-jobs.json',jobs);save_json(s/'screen-mesh.json',dict(rows=rows,K=16,DOF_rendered=False,requirements={k:[list(v) for v in sorted(values)] for k,values in requirements.items()}))

if __name__=='__main__':main()
