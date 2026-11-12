"""Fixed A-color, cumulative R1+R2, video-inpaint layer and GT-rec reference."""
import argparse
import gc
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
import cv2
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from route3_pilot_fields import OUT,write
from src.aperture_renderer import GaussianScene
from src.sharp_anchor import anchor
from src.dual_layer import depth_band
from src.c1_regions import linear_rgb
from src.hidden_retrieval_rules import aggregate
from src.layered_aperture import merge_layers
from src.analytic_aperture import sampled
from route3_sharp_s1 import raster
from route3_b2_oracle_3d import sparse_pixel_layer
from route3_upper_bound_evaluate import display


def scene(path):return GaussianScene(**torch.load(path,map_location='cuda',weights_only=True))


def anchor_visible(folder):
    entries=json.loads((folder/'prepared.json').read_text())['rows']
    for entry in entries:
        if not Path(entry['fill']).exists():raise FileNotFoundError('Generate LaMa using existing magicbokeh environment: '+entry['fill'])
    dest=folder/'visible';dest.mkdir(exist_ok=True);rows=[];cal=json.loads((folder/'models/calibration/summary.json').read_text())['rows']
    for entry in entries:
        f=entry['frame'];path=dest/f'f{f:03d}.pt'
        if path.exists():rows.append(dict(frame=f,**json.loads((dest/f'f{f:03d}-anchor.json').read_text())));continue
        cloud=scene(folder/'models/SHARP'/f'f{f:03d}.pt');fit=next(x for x in cal if x['frame']==f)['models']['SHARP'];z=cloud.means[:,2];mapped=1/(fit['a']/z.clamp_min(1e-6)+fit['b']).clamp_min(1e-4);ratio=torch.where(z>1e-6,mapped/z.clamp_min(1e-6),torch.ones_like(z));base=GaussianScene(cloud.means*ratio[:,None],cloud.quats,cloud.scales*ratio[:,None],cloud.opacities,cloud.colors,cloud.source_frames);d=np.load(entry['input_npz']);K=torch.tensor(d['K'],device='cuda');fill=torch.tensor(np.load(entry['fill'])['fill_rgb_linear'],device='cuda');target=torch.tensor(d['rgb_srgb'],device='cuda');torch.manual_seed(2026111121+f)
        fitted,result=anchor(base,K,fill,target,False,lambda x:write(dest/f'f{f:03d}-monitor.json',x));assert result['completed'],result['stop_reason']
        torch.save({key:getattr(fitted,key).detach().cpu() for key in fitted.__dataclass_fields__},path);write(dest/f'f{f:03d}-anchor.json',result);rows.append(dict(frame=f,**result));print('PILOT_ANCHOR',folder.name,f,result['rows'][-1]['input_PSNR_dB'],flush=True)
        del base,cloud,fitted;torch.cuda.empty_cache()
    write(dest/'summary.json',dict(rows=rows,SHARP_weights_frozen=True,target_RGB_only=True,steps=1000,loss='same S1b .8L1+.2SSIM',new_network_implemented=False))


def render(folder):
    if folder.name=='B2-high':return render_b2()
    if (folder/'predictions.json').exists():return
    entries=json.loads((folder/'prepared.json').read_text())['rows'];output=folder/'predictions';output.mkdir(exist_ok=True);profile=json.loads((ROOT/'03_实验记录/records/route3-dual-layer-criteria-L2-2026-11-11.json').read_text())['profile'];rows=[]
    c2root=OUT/'C2'/folder.name/'results/input/frames'
    for entry in entries:
        f=entry['frame'];d=np.load(entry['input_npz']);K=torch.tensor(d['K'],device='cuda');E=torch.eye(4,device='cuda');cloud=scene(folder/'visible'/f'f{f:03d}.pt');fill=torch.tensor(np.load(entry['fill'])['fill_rgb_linear'],device='cuda');bg=torch.tensor(d['background_depth'],device='cuda');pool={k:v for k,v in np.load(entry['pool']).items()}
        with torch.inference_mode():_,z,a=raster(cloud,K)
        own=np.where((a.cpu().numpy()>=.1)&(z.cpu().numpy()>0),z.cpu().numpy(),d['depth']);r2,audit=aggregate(pool,2,own);np.savez_compressed(folder/'candidates'/f'f{f:03d}'/'R1_R2.npz',**r2);oracle=np.load(folder/'oracle'/f'f{f:03d}.npz');image=cv2.imread(str(c2root/f'{[1,6,11,16,21,26,30,35,40,45,50,55,60].index(f):04d}.png'))[...,::-1].astype(np.float32)/255;mask=cv2.imread(str(OUT/'C2'/folder.name/'mask'/f'{[1,6,11,16,21,26,30,35,40,45,50,55,60].index(f):04d}.png'),0)>0;mask=cv2.dilate(mask.astype(np.uint8),np.ones((9,9),np.uint8))>0;yy,xx=np.nonzero(mask&(d['background_depth']>own+np.maximum(.005,.02*own)));c2=dict(uv=np.c_[xx+.5,yy+.5].astype(np.float32),depth=d['background_depth'][yy,xx],colors_linear=linear_rgb(image)[yy,xx])
        for case in entry['cases']:
            band=depth_band(d['depth'],d['K'][0,0],case['focus_m'],.05/(2*case['fstop']))[0]
            torch.cuda.synchronize();base_start=time.monotonic()
            with torch.inference_mode():base=sampled(cloud,K,E,512,512,case['focus_m'],.05/(2*case['fstop']),64,fill,bg,far_plane=1e5,return_fill_contribution=True)
            torch.cuda.synchronize();base_seconds=time.monotonic()-base_start
            baseline={k:v.cpu().numpy() for k,v in base.items()};del base
            for name,parent in [('A_color',None),('R1_R2',r2),('C2_ProPainter',c2),('GTrec',oracle)]:
                path=output/(case['id']+'-'+name+'.npz')
                if path.exists():rows.append(dict(**case,method=name,path=str(path)));continue
                torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.monotonic();hidden=None
                if parent is None:values=baseline
                else:
                    uv=parent['uv'];selected=band[np.clip(uv[:,1].astype(int),0,511),np.clip(uv[:,0].astype(int),0,511)]
                    if name=='GTrec':selected&=parent['recoverable']
                    hidden=sparse_pixel_layer(parent,selected,K,E,profile,'cuda')
                    with torch.inference_mode():pred=sampled(merge_layers(cloud,hidden),K,E,512,512,case['focus_m'],.05/(2*case['fstop']),64,fill,bg,far_plane=1e5,return_fill_contribution=True)
                    values={k:np.where(band[...,None] if v.ndim==3 else band,v.cpu().numpy(),baseline[k]) for k,v in pred.items()};del pred
                torch.cuda.synchronize();elapsed=base_seconds if parent is None else time.monotonic()-t;np.savez_compressed(path,prediction_srgb=display(values['rgb_linear']),**values);rows.append(dict(**case,method=name,path=str(path),seconds=elapsed,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,hidden_Gaussians=0 if hidden is None else len(hidden.means)));del hidden;print('PILOT_BASELINE',folder.name,case['id'],name,flush=True)
        del cloud;torch.cuda.empty_cache()
    write(folder/'predictions.json',dict(rows=rows,geometry='E1;validated-affine-or-explicitidentityfallback',K=64,C2_input_depth_mask_and_S1d_background_extension=True,GTrec_reference_only=True))


def render_b2():
    from route3_sharp_s1d import entries
    from route3_sharp_s1c_apply import anchored
    folder=OUT/'B2-high';folder.mkdir(exist_ok=True);profile=json.loads((ROOT/'03_实验记录/records/route3-dual-layer-criteria-L2-2026-11-11.json').read_text())['profile'];rows=[];s1c=ROOT/'outputs/route3-sharp-s1c-2026-11-10-01';lookup=json.loads((ROOT/'03_实验记录/records/route3-sharp-s1c-2026-11-10-evaluation.json').read_text())['predictions']
    for entry in entries():
        f=entry['frame'];d=np.load(entry['input_npz']);K=torch.tensor(d['K'],device='cuda');E=torch.eye(4,device='cuda');cloud=anchored(entry,'A_color');fill=torch.tensor(np.load(entry['fill'])['fill_rgb_linear'],device='cuda');bg=torch.tensor(d['background_depth'],device='cuda');idx=[1,6,11,16,21,26,30,35,40,45,50,55,60].index(f);image=cv2.imread(str(OUT/'C2/B2-high/results/input/frames'/f'{idx:04d}.png'))[...,::-1].astype(np.float32)/255;mask=cv2.imread(str(OUT/'C2/B2-high/mask'/f'{idx:04d}.png'),0)>0;mask=cv2.dilate(mask.astype(np.uint8),np.ones((9,9),np.uint8))>0
        with torch.inference_mode():_,z,a=raster(cloud,K)
        own=np.where(a.cpu().numpy()>=.1,z.cpu().numpy(),d['depth']);yy,xx=np.nonzero(mask&(d['background_depth']>own+np.maximum(.005,.02*own)));parent=dict(uv=np.c_[xx+.5,yy+.5].astype(np.float32),depth=d['background_depth'][yy,xx],colors_linear=linear_rgb(image)[yy,xx])
        for case in entry['cases']:
            band=depth_band(d['depth'],d['K'][0,0],case['focus_m'],.05/(2*case['fstop']))[0];uv=parent['uv'];hidden=sparse_pixel_layer(parent,band[uv[:,1].astype(int),uv[:,0].astype(int)],K,E,profile,'cuda');base=np.load(lookup[case['id']]['cal_E1__A_color'])
            with torch.inference_mode():pred=sampled(merge_layers(cloud,hidden),K,E,512,512,case['focus_m'],.05/(2*case['fstop']),64,fill,bg,far_plane=1e5)
            rgb=np.where(band[...,None],pred['rgb_linear'].cpu().numpy(),base['rgb_linear']);path=folder/(case['id']+'-C2.npz');np.savez_compressed(path,prediction_srgb=display(rgb),rgb_linear=rgb);rows.append(dict(**case,method='C2_ProPainter',path=str(path)));del hidden,pred
        del cloud;torch.cuda.empty_cache()
    write(folder/'predictions.json',dict(rows=rows,K=64,prior_results_unmodified=True))


def aligned_reference(folder):
    """A4 placement-label rendering check; additional Oracle, never a method."""
    profile=json.loads((ROOT/'03_实验记录/records/route3-dual-layer-criteria-L2-2026-11-11.json').read_text())['profile'];rows=[]
    for e in json.loads((folder/'prepared.json').read_text())['rows']:
        f=e['frame'];d=np.load(e['input_npz']);K=torch.tensor(d['K'],device='cuda');E=torch.eye(4,device='cuda');cloud=scene(folder/'visible'/f'f{f:03d}.pt');fill=torch.tensor(np.load(e['fill'])['fill_rgb_linear'],device='cuda');bg=torch.tensor(d['background_depth'],device='cuda');p={k:v for k,v in np.load(folder/'oracle'/f'f{f:03d}.npz').items()};label=np.load(folder/'labels'/f'f{f:03d}'/(e['cases'][0]['id']+'.npz'));support=np.load(folder/'labels'/f'f{f:03d}'/'placement-support.npz')['local_fit_supported'];x=np.clip(p['uv'][:,0].astype(int),0,511);y=np.clip(p['uv'][:,1].astype(int),0,511);inv=label['placement_affine_a'][y,x]/p['depth']+label['placement_affine_b'][y,x];valid=(inv>1e-4)&support[y,x];p['depth']=(1/np.maximum(inv,1e-4)).astype(np.float32)
        for c in e['cases']:
            band=depth_band(d['depth'],d['K'][0,0],c['focus_m'],.05/(2*c['fstop']))[0];selected=p['recoverable']&valid&band[y,x];hidden=sparse_pixel_layer(p,selected,K,E,profile,'cuda');base=np.load(folder/'predictions'/(c['id']+'-A_color.npz'))
            with torch.inference_mode():pred=sampled(merge_layers(cloud,hidden),K,E,512,512,c['focus_m'],.05/(2*c['fstop']),64,fill,bg,far_plane=1e5)
            rgb=np.where(band[...,None],pred['rgb_linear'].cpu().numpy(),base['rgb_linear']);path=folder/'predictions'/(c['id']+'-GTrec_aligned-label-check.npz');np.savez_compressed(path,prediction_srgb=display(rgb),rgb_linear=rgb);rows.append(dict(**c,method='GTrec_aligned_label_check',path=str(path),parents=int(selected.sum()),local_fit_supported=True,Oracle_only=True));del hidden,pred
        del cloud;torch.cuda.empty_cache()
    write(folder/'aligned-reference-predictions.json',dict(rows=rows,scope='A4 label placement check,additionalOracle;originalfourgroupsunchanged',unsupported_localfit_hidden_parents_excluded=True))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scene',required=True);p.add_argument('--mode',choices=['anchor_visible','render','aligned_reference'],required=True);a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise RuntimeError('ExplicitGPU0')
    subprocess.run(['nvidia-smi'],check=True);torch.set_num_threads(8);cv2.setNumThreads(1);globals()[a.mode](OUT/a.scene)


if __name__=='__main__':main()
