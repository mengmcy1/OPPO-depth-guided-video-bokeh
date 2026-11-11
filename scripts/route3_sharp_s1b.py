"""S1b output-only anchoring; all products use a fresh experiment directory."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from route3_sharp_s1 import OUT as S1,scene as original_scene,FRAMES
from route3_upper_bound_evaluate import BASE,OUT as UPPER,RECORDS
from src.sharp_anchor import anchor
from src.aperture_renderer import GaussianScene
OUT=ROOT/'outputs/route3-sharp-s1b-2026-11-11-01'
PREFIX='route3-sharp-s1b-2026-11-11'
CRITERIA=RECORDS/'route3-sharp-s1b-criteria-2026-11-11.json'


def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def anchor_native(c):
    root=OUT/'anchor/native';root.mkdir(parents=True,exist_ok=True)
    prior=json.loads((S1/'prepared.json').read_text())['rows']
    for frame in [1,30,60]:
        entry=next(r for r in prior if r['scale_label']=='native' and r['frame']==frame)
        d=np.load(entry['input_npz']);K=torch.tensor(d['K'],device='cuda');fill=torch.tensor(np.load(S1/'fill'/entry['id']/'fill.npz')['fill_rgb_linear'],device='cuda')
        target=torch.tensor(d['rgb_srgb'],device='cuda');base=original_scene(frame,'full')
        for name,op in [('A_color',False),('A_color_opacity',True)]:
            if (OUT/'stop.json').exists():
                stopped=json.loads((OUT/'stop.json').read_text())
                if stopped['stage']!='calibration' or not (OUT/'native-continuation-authorized.json').exists():
                    raise RuntimeError('S1B_STOP another authorized stage triggered stop; do not launch new fits')
            torch.manual_seed(2026111121+frame);folder=root/f'f{frame:03d}'/name
            if (folder/'result.json').exists():
                prior_result=json.loads((folder/'result.json').read_text())
                assert prior_result['completed'] and prior_result['completed_updates']==1000 and Path(prior_result['attributes']).exists()
                print('ANCHOR_REUSE_COMPLETED',frame,name,flush=True);continue
            folder.mkdir(parents=True,exist_ok=False)
            def monitor(value):write(folder/'monitor.json',value)
            fitted,result=anchor(base,K,fill,target,op,monitor)
            torch.save(dict(colors=fitted.colors.detach().cpu(),opacities=fitted.opacities.detach().cpu(),completed_updates=result['completed_updates']),folder/'attributes-final.pt')
            result.update(frame=frame,variant=name,geometry='native',code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                attributes=str(folder/'attributes-final.pt'),source_Gaussians=str(S1/'gaussians'/f'f{frame:03d}.pt'),input_RGB=entry['input_npz'],fill=str(S1/'fill'/entry['id']/'fill.npz'))
            write(folder/'result.json',result);write(RECORDS/(PREFIX+f'-anchor-native-f{frame:03d}-{name}.json'),result)
            if not result['completed']:
                write(OUT/'stop.json',dict(stage='anchor_native',frame=frame,variant=name,reason=result['stop_reason'],completed_updates=result['completed_updates']))
                raise RuntimeError('S1B_STOP '+str(result['stop_reason']))
            print('ANCHOR_DONE',frame,name,result['rows'][0]['input_PSNR_dB'],result['rows'][-1]['input_PSNR_dB'],flush=True)
        del base;torch.cuda.empty_cache()


def fitted_scene(frame,name,source=None,folder=None):
    base=original_scene(frame,'full') if source is None else source
    path=(folder or OUT/'anchor/native'/f'f{frame:03d}'/name)/'attributes-final.pt'
    attributes=torch.load(path,map_location='cuda',weights_only=True);assert attributes['completed_updates']==1000
    return GaussianScene(base.means,base.quats,base.scales,attributes['opacities'],attributes['colors'],base.source_frames)


def render_native(c):
    from src.analytic_aperture import sampled
    from src.layered_aperture import merge_layers
    from src.dual_layer import depth_band
    from route3_b2_oracle_3d import sparse_pixel_layer
    from route3_sharp_s1 import raster
    from route3_upper_bound_evaluate import display
    root=OUT/'render-native';root.mkdir(exist_ok=False);zero=OUT/'zero-native';zero.mkdir(exist_ok=False);rows=[];zeros=[];timing=[]
    profile=json.loads((RECORDS/'route3-dual-layer-criteria-L2-2026-11-11.json').read_text())['profile']
    prior=json.loads((S1/'prepared.json').read_text())['rows']
    for frame in [1,30,60]:
        entry=next(r for r in prior if r['scale_label']=='native' and r['frame']==frame);d=np.load(entry['input_npz']);K=torch.tensor(d['K'],device='cuda');E=torch.eye(4,device='cuda')
        fill=torch.tensor(np.load(S1/'fill'/entry['id']/'fill.npz')['fill_rgb_linear'],device='cuda');bg=torch.tensor(d['background_depth'],device='cuda')
        for name in ['A_color','A_color_opacity']:
            cloud=fitted_scene(frame,name)
            with torch.inference_mode():raw,depth,alpha=raster(cloud,K)
            path=zero/f'f{frame:03d}-{name}.npz';np.savez_compressed(path,prediction_srgb=display((raw+(1-alpha[...,None])*fill).cpu().numpy()),alpha=alpha.cpu().numpy())
            zeros.append(dict(frame=frame,method=name,path=str(path)))
            for case in entry['cases']:
                band=depth_band(d['depth'],d['K'][0,0],case['focus_m'],.05/(2*case['fstop']))[0];torch.cuda.reset_peak_memory_stats();times={}
                with torch.inference_mode():
                    torch.cuda.synchronize();started=time.monotonic();r=sampled(cloud,K,E,512,512,case['focus_m'],.05/(2*case['fstop']),64,fill,bg,far_plane=1e5,return_fill_contribution=True)
                    torch.cuda.synchronize();times[name]=time.monotonic()-started
                values={name:{k:v.cpu().numpy() for k,v in r.items()}}
                for variant,parentpath in entry['parents'].items():
                    parent=np.load(parentpath);uv=parent['uv'];x=np.clip(uv[:,0].astype(int),0,511);y=np.clip(uv[:,1].astype(int),0,511)
                    hidden=sparse_pixel_layer(parent,band[y,x],K,E,profile,'cuda')
                    with torch.inference_mode():
                        torch.cuda.synchronize();started=time.monotonic();joint=sampled(merge_layers(cloud,hidden),K,E,512,512,case['focus_m'],.05/(2*case['fstop']),64,fill,bg,far_plane=1e5,return_fill_contribution=True)
                        torch.cuda.synchronize();times[name+'_L3_'+variant]=time.monotonic()-started
                    values[name+'_L3_'+variant]={k:np.where(band[...,None] if v.ndim==3 else band,v.cpu().numpy(),values[name][k]) for k,v in joint.items()}
                    del hidden
                for method,value in values.items():
                    p=root/(case['id']+'-'+method+'.npz');np.savez_compressed(p,prediction_srgb=display(value['rgb_linear']),**value,band=band,focus_m=case['focus_m'])
                    rows.append(dict(id=case['id'],method=method,path=str(p)))
                timing.append(dict(id=case['id'],method=name,seconds=times,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20))
                print('S1B_RENDER',case['id'],name,flush=True)
            del cloud;torch.cuda.empty_cache()
    write(OUT/'native-predictions.json',dict(completed=True,rows=rows,zero=zeros));write(RECORDS/(PREFIX+'-native-render.json'),dict(completed=True,rows=timing,K=64,hidden_results_descriptive_only=True,
        GT_depth_or_hidden_surfaces_used_in_prediction=False,E1_retrieval_uses_given_GT_poses=True,E2_retrieval_uses_globally_aligned_DA3_poses=True,calibration_used=False))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['anchor-native','render-native'],required=True);a=p.parse_args()
    c=json.loads(CRITERIA.read_text());assert c['frozen'];torch.set_num_threads(8)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise ValueError('Explicit authorized GPU0')
    subprocess.run(['nvidia-smi'],check=True)
    if a.mode=='anchor-native':anchor_native(c)
    else:render_native(c)


if __name__=='__main__':main()
