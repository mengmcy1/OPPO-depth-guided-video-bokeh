"""Read-only L0 error attribution by visible depth, coverage and input texture."""
import argparse
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.dual_layer import depth_band
from src.c1_bokehme import region_sse
from src.c1_regions import display_rgb


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,default=ROOT/'outputs/route3-b2only-2026-11-10-01')
    p.add_argument('--experiment',type=Path,required=True);p.add_argument('--record',type=Path,required=True);a=p.parse_args()
    if a.record.exists():raise FileExistsError(a.record)
    rows=[];cv2.setNumThreads(1)
    for x in json.loads((a.base/'manifests/oracle.json').read_text())['items']:
        if x['fstop'] not in [1.4,2.]:continue
        d=np.load(x['input_npz']);v=np.load(a.experiment/'L0-scan'/x['id']/'V-G-K64/prediction.npz');pred=v['prediction_srgb'];alpha=v['alpha']
        O3=np.load(a.base/'methods-2d/bokehme'/x['id']/'prediction.npz')['prediction_srgb']
        raw=cv2.imread(x['gt_rgb_exr'],cv2.IMREAD_UNCHANGED)[...,:3][...,::-1];gt=display_rgb(raw).astype(np.float32)
        band,edge,R=depth_band(d['center_depth'],d['K'][0,0],x['focus_distance_m'],x['aperture_radius_m']);known=d['center_surface_valid']
        total=region_sse(pred,gt,np.ones(band.shape,bool))['sse']
        gray=cv2.cvtColor(d['visible_rgb_linear'],cv2.COLOR_RGB2GRAY)
        grad=cv2.magnitude(cv2.Sobel(gray,cv2.CV_32F,1,0),cv2.Sobel(gray,cv2.CV_32F,0,1));threshold=float(np.quantile(grad[known],.75))
        masks=dict(smooth_depth_high_alpha=known&~band&(alpha>=.99),
            texture_edges_in_smooth_depth=known&~band&(alpha>=.99)&(grad>=threshold),
            depth_band=band,low_alpha=known&(alpha<.95),sky=~known)
        metrics={k:dict(VG=region_sse(pred,gt,m),O3=region_sse(O3,gt,m),VG_fraction_of_whole_SSE=region_sse(pred,gt,m)['sse']/total) for k,m in masks.items()}
        rows.append(dict(id=x['id'],scene=x['scene'],focus_m=x['focus_distance_m'],fstop=x['fstop'],gradient_threshold=threshold,regions=metrics))
    record=dict(stage='L0 finite-aperture error budget',rows=rows,uses_L1_outputs=False,
        mask_definition='current visible depth band,known mesh surfaces,rendered alpha and input texture gradients; no hiddenGT-selected masks; subsets can overlap',
        interpretation='high-alpha smooth-depth errors cannot be explained solely by missing coverage/hidden content; texture association is not unique causal proof',old_results_modified=False)
    a.record.write_text(json.dumps(record,indent=2)+'\n')
    for scene in ['B2-low','B2-high']:
        for k in masks:
            subset=[r['regions'][k] for r in rows if r['scene']==scene]
            print(scene,k,'mean SSE fraction',np.mean([r['VG_fraction_of_whole_SSE'] for r in subset]))


if __name__=='__main__':main()
