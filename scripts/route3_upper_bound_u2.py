"""U2 predictions from allowed inputs, with E1/E2 sharing estimated depth.

The prediction branches never load GT geometry, hidden surfaces, recoverability
or scoring masks. E1's explicitly authorized GT poses were exported by the
alignment stage. RGB comes only from the thirteen input pinhole images.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from src.dual_layer import depth_band
from src.dual_layer_retrieval import retrieve
OUT=ROOT/'outputs/route3-upper-bound-2026-11-11-01'
BASE=ROOT/'outputs/route3-b2only-2026-11-10-01'
RECORDS=ROOT/'03_实验记录/records'
FRAMES=[1,6,11,16,21,26,30,35,40,45,50,55,60]


def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def lens_cases(frame):
    for focus in [1.4,5.]:
        for stop in [1.4,2.]:
            yield dict(id=f'B2-high-f{frame:03d}-N{stop:g}-F{focus:g}',focus_m=focus,fstop=stop,aperture=.05/(2*stop))


def prepare():
    dest=OUT/'U2-inputs';dest.mkdir(exist_ok=False);rows=[];fills=[]
    timing=json.loads((BASE/'inputs/B2-high/pinhole/timings.json').read_text());index={r['frame']:r for r in timing};colors={}
    for frame in FRAMES:
        raw=cv2.imread(str(BASE/'inputs/B2-high/pinhole'/(index[frame]['key']+'.exr')),cv2.IMREAD_UNCHANGED)[...,:3][...,::-1].copy()
        colors[frame]=np.clip(raw,0,1).astype(np.float32)
    for variant in ['E1','E2']:
        views=[]
        for frame in FRAMES:
            d=np.load(OUT/'U2-da3'/variant/f'f{frame:03d}.npz')
            views.append(dict(frame=frame,depth=d['depth'],K=d['K'],c2w=d['c2w'],rgb_linear=colors[frame]))
        for frame in [1,30,60]:
            v=next(r for r in views if r['frame']==frame);z=v['depth'];K=v['K'];union=np.zeros(z.shape,bool);maxrad=0.
            for focus in [1.4,5.]:
                for stop in [.5,1.4,2.]:
                    b,_,R=depth_band(z,K[0,0],focus,.05/(2*stop));union|=b;maxrad=max(maxrad,float(R.max()))
            mask=cv2.dilate(union.astype(np.uint8),np.ones((5,5),np.uint8),iterations=2)>0
            radius=int(np.ceil(2*maxrad))+4
            bg=cv2.dilate(z,np.ones((2*radius+1,2*radius+1),np.uint8))
            parent,audit=retrieve(z,K,v['c2w'],union,views,frame)
            name=f'{variant}-B2-high-f{frame:03d}';folder=dest/name;folder.mkdir()
            rgb=np.where(colors[frame]<=.0031308,12.92*colors[frame],1.055*colors[frame]**(1/2.4)-.055).clip(0,1).astype(np.float32)
            np.savez_compressed(folder/'inputs.npz',center_depth=z,center_depth_raw=z,center_surface_valid=np.ones(z.shape,bool),
                K=K,camera_c2w=v['c2w'],visible_rgb_linear=colors[frame],rgb_srgb=rgb,fill_mask=mask,background_depth=bg)
            np.savez_compressed(folder/'parents.npz',**parent)
            rows.append(dict(id=name,variant=variant,frame=frame,input_npz=str(folder/'inputs.npz'),parents=str(folder/'parents.npz'),
                hidden_parents=len(parent['depth']),audit=audit,fill_id=f'E1-B2-high-f{frame:03d}',
                geometry='DA3 depth + '+('explicit E1 GT poses' if variant=='E1' else 'one Sim3-aligned DA3 poses'),
                source_mask_no_GT=True,unknown_sky_GT_not_used=True,background_extension_radius_px=radius))
            if variant=='E1':fills.append(dict(id=name,input_npz=str(folder/'inputs.npz'),uses_GT_segmentation=False,uses_true_hidden_depth=False))
            print('U2_RETRIEVED',name,len(parent['depth']),flush=True)
    write(OUT/'U2-inputs.json',dict(completed=True,items=rows,input_frames=FRAMES,target_hidden_GT_used=False,
        retrieval_rule='unchangedL3 nearest behind-visible;median within1%;target excluded;no dynamics compensation'))
    write(OUT/'U2-fill-inputs.json',dict(items=fills,E1_E2_fill_identical=True))
    write(RECORDS/'route3-upper-bound-2026-11-11-U2-prepare.json',dict(completed=True,rows=rows,forbidden_GT_fields_read=[]))


def bokehme():
    import torch
    torch.set_num_threads(4)
    from src.baselines.bokehme import setup,predict
    pipeline,settings,nets,_=setup(ROOT.parent/'BokehMe',ROOT/'models/BokehMe',dict(gamma=2.2,highlight=False,defocus_scale=10.))
    dest=OUT/'U2-O3';dest.mkdir(exist_ok=False);rows=[];timings=[]
    for x in json.loads((OUT/'U2-inputs.json').read_text())['items']:
        if x['variant']!='E1':continue
        d=np.load(x['input_npz']);rgb=d['rgb_srgb'];depth=d['center_depth'];fx=d['K'][0,0]
        for case in lens_cases(x['frame']):
            control=(fx*case['aperture']*(1/depth-1/case['focus_m'])/10).astype(np.float32)
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.monotonic()
            pred,_=predict(pipeline,settings,nets,np.ascontiguousarray(rgb),np.ascontiguousarray(control))
            torch.cuda.synchronize();seconds=time.monotonic()-start
            path=dest/(case['id']+'.npz');np.savez_compressed(path,prediction_srgb=pred.clip(0,1).astype(np.float32))
            for variant in ['E1','E2']:rows.append(dict(id=case['id'],method='O3_'+variant,path=str(path)))
            timings.append(dict(id=case['id'],seconds=seconds,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20))
            print('U2_O3',case['id'],seconds,flush=True)
    write(OUT/'U2-O3-predictions.json',dict(rows=rows))
    write(RECORDS/'route3-upper-bound-2026-11-11-U2-O3.json',dict(completed=True,rows=timings,
        CoC='unchanged fx*a*(1/z-1/F)/10;gamma2.2;defocus_scale10',
        GT_depth_used=False,E1_E2_predictions_identical='sameDA3 depth andRGB;O3 doesnotuseposes'))


def render():
    import torch
    from src.layered_aperture import pixel_layer,merge_layers
    from src.analytic_aperture import sampled
    from route3_b2_oracle_3d import sparse_pixel_layer
    torch.set_num_threads(4)
    c=json.loads((RECORDS/'route3-upper-bound-criteria-2026-11-11.json').read_text())
    profile=json.loads((RECORDS/'route3-dual-layer-criteria-L2-2026-11-11.json').read_text())['profile']
    dest=OUT/'U2-3d';dest.mkdir(exist_ok=False);rows=[];stats=[]
    for x in json.loads((OUT/'U2-inputs.json').read_text())['items']:
        d=np.load(x['input_npz']);parent=np.load(x['parents']);K=torch.tensor(d['K'],device='cuda',dtype=torch.float32)
        cam=torch.tensor(d['camera_c2w'],device='cuda',dtype=torch.float32);E=torch.linalg.inv(cam)
        visible=pixel_layer(torch.tensor(d['visible_rgb_linear'],device='cuda'),torch.tensor(d['center_depth'],device='cuda'),K,cam)
        filled=np.load(OUT/'U2-fill'/x['fill_id']/'fill.npz')
        fill=torch.tensor(filled['fill_rgb_linear'],device='cuda');bg=torch.tensor(d['background_depth'],device='cuda')
        for case in lens_cases(x['frame']):
            band,_,_=depth_band(d['center_depth'],d['K'][0,0],case['focus_m'],case['aperture'],c['band']['multiplier'],c['band']['relative_inverse_depth_jump'])
            uv=parent['uv'];xx=np.clip(uv[:,0].astype(int),0,511);yy=np.clip(uv[:,1].astype(int),0,511)
            chosen=band[yy,xx];hidden=sparse_pixel_layer(parent,chosen,K,cam,profile,'cuda')
            values={};runtime={};torch.cuda.reset_peak_memory_stats()
            for name,pc in [('O5',visible),('joint',merge_layers(visible,hidden))]:
                with torch.inference_mode():
                    torch.cuda.synchronize();start=time.monotonic()
                    value=sampled(pc,K,E,512,512,case['focus_m'],case['aperture'],64,fill,bg,far_plane=1e5)
                    torch.cuda.synchronize();runtime[name]=time.monotonic()-start
                values[name]=value['rgb_linear'].cpu().numpy()
            combined=np.where(band[...,None],values['joint'],values['O5'])
            predictions={'O5_'+x['variant']:values['O5'],'ours_'+x['variant']:combined}
            for name,raw in predictions.items():
                pred=np.where(raw<=.0031308,12.92*raw,1.055*np.maximum(raw,0)**(1/2.4)-.055).clip(0,1).astype(np.float32)
                path=dest/(case['id']+'-'+name+'.npz');np.savez_compressed(path,rgb_linear=raw,prediction_srgb=pred,estimated_band=band)
                rows.append(dict(id=case['id'],method=name,path=str(path)))
            o3=np.load(OUT/'U2-O3'/(case['id']+'.npz'))['prediction_srgb'];ours=np.load(rows[-1]['path'])['prediction_srgb']
            path=dest/(case['id']+'-descriptive-'+x['variant']+'.npz');np.savez_compressed(path,prediction_srgb=np.where(band[...,None],ours,o3))
            rows.append(dict(id=case['id'],method='descriptive_'+x['variant'],path=str(path)))
            stats.append(dict(id=case['id'],variant=x['variant'],band_pixels=int(band.sum()),hidden_parents_selected=int(chosen.sum()),
                runtime=runtime,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20))
            print('U2_RENDER',x['variant'],case['id'],flush=True);del hidden
        del visible;torch.cuda.empty_cache()
    write(OUT/'U2-3d-predictions.json',dict(rows=rows))
    o3=json.loads((OUT/'U2-O3-predictions.json').read_text())['rows']
    write(OUT/'U2-predictions.json',dict(rows=rows+o3))
    write(RECORDS/'route3-upper-bound-2026-11-11-U2-render.json',dict(completed=True,rows=stats,
        GT_depth_or_hidden_geometry_or_masks_used=False,estimated_depth_band=True,renderer='unchangednative deterministic64',
        composition='depthsorted jointvisible+hidden, remainingT*sameinputonlyLaMa;hidden inactive outside outputband'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['prepare','bokehme','render'],required=True);a=p.parse_args()
    cv2.setNumThreads(1)
    if a.mode!='prepare':
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise ValueError('Explicit GPU0')
        subprocess.run(['nvidia-smi'],check=True)
    globals()[a.mode]()


if __name__=='__main__':main()
