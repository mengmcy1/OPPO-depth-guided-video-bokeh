"""Only actual input-side fill preparation; no candidate memory or scores."""
import argparse,json,sys
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from src.dual_layer import depth_band

def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',type=Path,required=True);a=p.parse_args();s=a.scene;entries=[];fillitems=[];K=np.array([[512*50/36,0,256],[0,512*50/36,256],[0,0,1]],np.float32);dest=s/'screen-inputs';dest.mkdir(exist_ok=True)
    for f in [1,30,60]:
        raw=np.load(s/'models/SHARP'/f'f{f:03d}.npz');cal=np.load(s/'models/calibration/SHARP'/f'f{f:03d}.npz');z=cal['depth'];g=np.load(s/'geometry'/f'f{f:03d}-center.npz');gt=g['depth_axial_m'];union=np.zeros((512,512),bool);cases=[];maxradius=0.
        for nominal in [1.4,5.]:
            for stop in [1.4,2.]:
                r=.05/(2*stop);clicked=(gt>0)&(K[0,0]*r*abs(1/np.maximum(gt,1e-6)-1/nominal)<=1)
                if not clicked.any():raise RuntimeError('No focus selection pixels '+s.name+str((f,nominal)))
                focus=float(np.median(z[clicked]));band,edge,radius=depth_band(z,K[0,0],focus,r);union|=band;maxradius=max(maxradius,float(radius.max()));cases.append(dict(id=f'f{f:03d}-N{stop:g}-F{nominal:g}',frame=f,nominal_focus_m=nominal,focus_m=focus,fstop=stop))
        mask=cv2.dilate(union.astype(np.uint8),np.ones((5,5),np.uint8),iterations=2)>0;extension=int(np.ceil(2*maxradius))+4;bg=cv2.dilate(z,np.ones((2*extension+1,2*extension+1),np.uint8));path=dest/f'f{f:03d}.npz'
        if not path.exists():np.savez_compressed(path,depth=z,K=K,c2w=cal['c2w'],rgb_srgb=raw['rgb_srgb'],rgb_linear=raw['rgb_linear'],fill_mask=mask,background_depth=bg)
        entries.append(dict(scene=s.name,frame=f,input_npz=str(path),fill=str(s/'screen-fill'/f'f{f:03d}/fill.npz'),cases=cases));fillitems.append(dict(id=f'f{f:03d}',input_npz=str(path),uses_GT_segmentation=False,uses_true_hidden_depth=False))
    (s/'screen-prepared.json').write_text(json.dumps(dict(rows=entries),indent=2)+'\n');(s/'screen-fill-inputs.json').write_text(json.dumps(dict(items=fillitems),indent=2)+'\n')
if __name__=='__main__':main()
