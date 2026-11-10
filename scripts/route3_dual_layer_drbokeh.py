"""L0 prerequisite: examine/reuse native Dr.Bokeh layers without changing B2.

The only repair restores supplied visible GT disparities after inpainting.
The public scatter kernel (+1 soft radius), gamma and hidden fill stay intact.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.c1_bokehme import physical_coc_radius,region_sse
from src.baselines.drbokeh import map_geometry


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,default=ROOT/'outputs/route3-b2only-2026-11-10-01')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--record',type=Path,required=True)
    a=p.parse_args()
    if a.record.exists():raise FileExistsError(a.record)
    a.output.mkdir(parents=True,exist_ok=False)
    # Declared before the first new baseline measurement.
    rule=dict(in_focus='known visible GT surfaces, abs(physical CoC radius)<=1px',
              focus_identity_PSNR_vs_O3_allowed_drop_dB=.5,
              judgement='pool RGB SSE independently per scene/focus over f1.4/f2 and frames1/30/60; every primary high focus must pass',
              no_native_kernel_or_CoC_retuning=True)
    (a.output/'precheck-rule.json').write_text(json.dumps(rule,indent=2)+'\n')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise ValueError('Explicit GPU0 required')
    subprocess.run(['nvidia-smi'],check=True)
    import torch
    import cv2
    from src.baselines.drbokeh_cuda import load_grid2d
    torch.set_num_threads(4);cv2.setNumThreads(1)
    repo=ROOT.parent/'DrBokeh';sys.path.insert(0,str(repo/'app/Render/DScatter'))
    import GPU_scatter
    GPU_scatter.scatter_cuda=load_grid2d()
    from scatter import Multi_Layer_Renderer
    model=Multi_Layer_Renderer(71).cuda().eval()
    entries=[x for x in json.loads((a.base/'manifests/oracle.json').read_text())['items'] if x['fstop'] in [1.4,2.]]
    rows=[]
    for x in entries:
        d=np.load(x['input_npz']);cached=np.load(a.base/'methods-2d/drbokeh'/x['id']/'prediction.npz')
        rgb=d['rgb_srgb'];depth=d['center_depth'];known=d['center_surface_valid'];alpha=d['GTforeground_alpha']
        dn,fn,kn,mapping=map_geometry(1/depth.astype(np.float64),1/x['focus_distance_m'],float(d['K'][0,0])*x['aperture_radius_m'])
        radius,_=physical_coc_radius(depth,d['K'][0,0],x['aperture_radius_m'],x['focus_distance_m'])
        focus=known&(abs(radius)<=1.)
        O3=np.load(a.base/'methods-2d/bokehme'/x['id']/'prediction.npz')['prediction_srgb']
        original=[cached[k].copy() for k in ['fg_rgbad_render','bg_rgbad_render']]
        repaired=[v.copy() for v in original]
        # Keep native hidden estimates, but never overwrite known input depth.
        fg=alpha>.1;bg=~d['fill_mask']
        repaired[0][...,4]=np.where(fg,dn,repaired[0][...,4])
        repaired[1][...,4]=np.where(bg,dn,repaired[1][...,4])
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.monotonic()
        with torch.inference_mode():
            check=model.inference(original,kn,fn)**(1/2.2)
            pred=model.inference(repaired,kn,fn)**(1/2.2)
        torch.cuda.synchronize();seconds=time.monotonic()-started
        discrepancy=float(abs(np.clip(check,0,1)-cached['prediction_srgb']).max())
        if discrepancy>3e-6:raise ValueError('Native layer replay differs from old output')
        out=a.output/x['id'];out.mkdir();pred=np.clip(pred,0,1).astype(np.float32)
        np.savez_compressed(out/'prediction.npz',prediction_srgb=pred,center_surface_valid=known)
        cv2.imwrite(str(out/'prediction.png'),np.uint8(pred[...,::-1]*255))
        rows.append(dict(id=x['id'],scene=x['scene'],frame=x['frame_id'],focus_m=x['focus_distance_m'],fstop=x['fstop'],
            unknown_sky_pixels=int((~known).sum()),sky_depth_m=10000.,focus_pixels=int(focus.sum()),
            O3_to_input=region_sse(O3,rgb,focus),native_to_input=region_sse(cached['prediction_srgb'],rgb,focus),
            repaired_to_input=region_sse(pred,rgb,focus),mapping=mapping,replay_max_error=discrepancy,
            seconds=seconds,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20))
        print('DR_PRECHECK',x['id'],rows[-1]['native_to_input']['psnr'],rows[-1]['repaired_to_input']['psnr'],rows[-1]['O3_to_input']['psnr'],flush=True)
    pooled={}
    for scene in ['B2-high','B2-low']:
        pooled[scene]={}
        for f in sorted({r['focus_m'] for r in rows if r['scene']==scene}):
            selected=[r for r in rows if r['scene']==scene and r['focus_m']==f];stats={}
            for name in ['O3_to_input','native_to_input','repaired_to_input']:
                count=sum(r[name]['pixels'] for r in selected);sse=sum(r[name]['sse'] for r in selected)
                stats[name]=dict(pixels=count,sse=sse,psnr=float(-10*np.log10(max(sse/(3*count),1e-12))))
            stats['repaired_minus_O3_dB']=stats['repaired_to_input']['psnr']-stats['O3_to_input']['psnr']
            stats['passed']=stats['repaired_minus_O3_dB']>=-.5;pooled[scene][str(f)]=stats
    accepted=all(v['passed'] for v in pooled['B2-high'].values())
    record=dict(stage='L0 prerequisite Oracle-input Dr.Bokeh',rule=rule,rows=rows,pooled=pooled,included_in_L1_L2_L3=accepted,
        inspected=dict(resolution=[512,512],input='same standard sRGBfloat[0,1]',native_gamma=2.2,offset=0,
            radius='fx*aperture*(1/depth-1/focus), algebraic mapping checked; radius not diameter',
            sky='raw0 preserved; method-only10000m sky; no treating sky as zero CoC',
            kernel='official radius abs(relative_disparity)*strength+1 and soft tail retained',
            repair='restore visible GT disparity on foreground alpha>.1 and background outside common fill mask; no GT hidden geometry/RGB'),
        training=False,old_B2_modified=False,next_step='continue L0; DrBokeh excluded if precheck failed')
    a.record.write_text(json.dumps(record,indent=2)+'\n');print('DRBOKEH_ACCEPTED',accepted,flush=True)


if __name__=='__main__':main()
