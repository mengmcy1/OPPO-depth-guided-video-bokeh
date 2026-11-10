"""Input-only shared LaMa setup or L3 retrieval, preserving all B2 outputs."""
import argparse
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.c1_regions import linear_rgb,display_rgb
from src.dual_layer import depth_band
from src.dual_layer_retrieval import retrieve
FRAMES=[1,6,11,16,21,26,30,35,40,45,50,55,60]


def shared(a):
    a.output.mkdir(parents=True,exist_ok=False);entries=[]
    for scene,focuses in [('B2-high',[1.4,5.]),('B2-low',[5.3,8.4])]:
        for frame in [1,30,60]:
            old=a.base/'common-inputs'/scene/f'f{frame:03d}.npz';d=np.load(old)
            # Only current visible depth, calibration, RGB and poses are read.
            z=d['center_depth'];known=d['center_surface_valid'];union=np.zeros(z.shape,bool);maxrad=0
            for focus in focuses:
                for stop in [.5,1.4,2.]:
                    band,_,r=depth_band(z,d['K'][0,0],focus,.05/(2*stop));union|=band;maxrad=max(maxrad,float(r.max()))
            fillmask=cv2.dilate(union.astype(np.uint8),np.ones((5,5),np.uint8),iterations=2)>0
            radius=int(np.ceil(2*maxrad))+4
            finite_depth=np.where(known,z,0).astype(np.float32)
            extended=cv2.dilate(finite_depth,np.ones((2*radius+1,2*radius+1),np.uint8))
            bg=np.where(known,np.maximum(extended,z),10000.).astype(np.float32)
            dest=a.output/f'{scene}-f{frame:03d}.npz'
            np.savez_compressed(dest,rgb_srgb=d['rgb_srgb'],visible_rgb_linear=d['visible_rgb_linear'],center_depth=z,
                center_depth_raw=d['center_depth_raw'],center_surface_valid=known,K=d['K'],camera_c2w=d['camera_c2w'],
                fill_mask=fillmask,background_depth=bg)
            entries.append(dict(id=f'{scene}-f{frame:03d}',scene=scene,frame=frame,input_npz=str(dest),
                fill_mask_pixels=int(fillmask.sum()),background_extension_radius_px=radius,
                uses_GT_segmentation=False,uses_true_hidden_depth=False,background_rule='max of current visible finite depth in fixed2*maxCoC+4 neighborhood;sky stays10km; proxy not true hidden depth',
                fillmask_rule='union of visible-depth/CoC bands over predetermined focal/aperture choices, then5x5 dilation2; no object labels'))
    record=dict(mode='shared input-only Cfill setup',items=entries,
        L3_allowed_input_fields=['RGB','visible depth','K','camera_c2w'],target_hidden_GT_used=False,GT_alpha_used=False,
        difference_from_B2='B2 used GT semantic foreground mask and true deep background; replace both with visible-depth-only proxies for L2/L3 to avoid L3 leakage',
        ablation='L2 includes O5 and full-hidden with this same fill before applying the band; no single-factor claim against old B2 fill')
    a.record.write_text(json.dumps(record,indent=2)+'\n')


def retrieval(a):
    a.output.mkdir(parents=True,exist_ok=False);records=[]
    shared_manifest=json.loads(a.shared.read_text())['items']
    for scene in ['B2-high','B2-low']:
        timings=json.loads((a.base/'inputs'/scene/'pinhole/timings.json').read_text());byframe={r['frame']:r for r in timings};views=[];input_files=[]
        for frame in FRAMES:
            geom=a.base/scene/'geometry'/f'f{frame:03d}-center.npz';g=np.load(geom)
            color=a.base/'inputs'/scene/'pinhole'/(byframe[frame]['key']+'.exr')
            raw=cv2.imread(str(color),cv2.IMREAD_UNCHANGED)[...,:3][...,::-1];rgb=linear_rgb(display_rgb(np.clip(raw,0,1)))
            views.append(dict(frame=frame,depth=g['depth_axial_m'],K=g['K'],c2w=g['camera_c2w'],rgb_linear=rgb))
            input_files.extend([str(geom),str(color)])
        for frame in [1,30,60]:
            item=next(r for r in shared_manifest if r['scene']==scene and r['frame']==frame);target=np.load(item['input_npz'])
            # Cache is focal-independent; render-time band activates its subset.
            union=np.zeros((512,512),bool)
            for focus in ([1.4,5.] if scene=='B2-high' else [5.3,8.4]):
                for stop in [.5,1.4,2.]:union|=depth_band(target['center_depth'],target['K'][0,0],focus,.05/(2*stop))[0]
            parent,audit=retrieve(target['center_depth'],target['K'],target['camera_c2w'],union,views,frame)
            dest=a.output/f'{scene}-f{frame:03d}';dest.mkdir();np.savez_compressed(dest/'parents.npz',**parent)
            records.append(dict(scene=scene,frame=frame,parents=len(parent['depth']),audit=audit,output=str(dest/'parents.npz'),input_files=input_files,
                fields_read=['source RGB','source visible axial depth','source K/camera_c2w','target current visible depth','visible-depth band'],
                forbidden_fields_read=[],target_hidden_geometry_or_texture_used=False,GT_recoverability_used=False,GT_segmentation_used=False,
                target_excluded=True,source_count_p10_p50_p90=np.quantile(parent['source_count'],[.1,.5,.9]).tolist() if len(parent['depth']) else None))
    a.record.write_text(json.dumps(dict(mode='L3 input-only retrieval',completed=True,rows=records,input_frames=FRAMES,
        aggregation='nearest depth behind target visible surface; median RGB/depth within1% of nearest depth',
        target_depth_margin='max1mm,0.5%visibledepth',motion='no motion compensation; dynamic errors evaluated separately using GT only after prediction',
        no_hidden_GT_reads=True,training=False),indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['shared','retrieval'],required=True)
    p.add_argument('--base',type=Path,default=ROOT/'outputs/route3-b2only-2026-11-10-01');p.add_argument('--output',type=Path,required=True);p.add_argument('--record',type=Path,required=True);p.add_argument('--shared',type=Path);a=p.parse_args()
    cv2.setNumThreads(1)
    if a.record.exists():raise FileExistsError(a.record)
    if a.mode=='shared':shared(a)
    else:retrieval(a)


if __name__=='__main__':main()
