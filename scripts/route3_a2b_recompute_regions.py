"""Rescore existing A1 images on frame1 mesh-derived regions; no rendering.

This remains useful when the background-depth feasibility probe stops O6.
Original A1 images/results are read-only inputs and are never overwritten.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.camvfx4d import read_rgb,effect_parameters
from src.c1_bokehme import physical_coc_radius,region_sse,patch_metrics
from route3_c1_bokehme_oracle import lpips_cpu


def distribution(values):
    v=np.asarray(values);v=v[np.isfinite(v)]
    return dict(pixels=int(v.size),absolute_quantiles_p0_p25_p50_p75_p95_p100=np.quantile(abs(v),[0,.25,.5,.75,.95,1]).tolist()) if len(v) else dict(pixels=0,absolute_quantiles_p0_p25_p50_p75_p95_p100=None)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--record',type=Path,required=True)
    p.add_argument('--alexnet-weights',type=Path,default=Path('/root/autodl-tmp/.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth'))
    a=p.parse_args();model=lpips_cpu(a);pools={};rows=[];record=dict(phase='A2b rescore existing A1 outputs; no new Oracle rendering',
        original_A1_results_modified=False,methods=['O1','O2','O3_reused_A1','O4'],unperformed_methods=['O5','O6-50'],
        regions=['mesh_near_motion','mesh_static_static','original_M_C1_proxy'],extra_regions=['mesh_all','full_image'],
        protocol=dict(RGB='display sRGB[0,1]',PSNR='scene-pooled SSE/(3*pixels)',patch='64x64 fixed-grid whole RGB patches weighted by region pixel count; no zero outside mask',LPIPS='alex,version0.1,spatialFalse',K=16),by_view=[])
    previous=Path('outputs/route3-c1-2026-11-10-01')
    for scene in ['plant','cat']:
        pools[scene]={};param=effect_parameters(Path('data/raw/CamVFX-4D')/scene)[0]
        for view in [0,16,33]:
            inputs=np.load(a.inputs/scene/str(view)/'layer-inputs.npz')
            oracle=np.load(previous/f'oracle-final/{scene}/{view}/oracle-renders.npz')
            bokeh=np.load(previous/f'bokehme-oracle-02/{scene}/{view}/bokehme-predictions.npz')
            target=read_rgb(f'data/raw/CamVFX-4D/{scene}/dof/{view}/000001.png')
            images=dict(O1=oracle['O1_srgb'],O2=bokeh['O2_srgb'],O3_reused_A1=bokeh['O3_srgb'],O4=oracle['O4_srgb'])
            masks={r:inputs[r].astype(bool) for r in ['mesh_near_motion','mesh_static_static','original_M_C1_proxy','mesh_all']}
            masks['full_image']=np.ones((512,512),bool)
            coc,valid=physical_coc_radius(inputs['center_depth'],inputs['K'][0,0],param['aperture_radius_m'],param['focus_distance_m'])
            bg_coc,bg_valid=physical_coc_radius(inputs['background_depth'],inputs['K'][0,0],param['aperture_radius_m'],param['focus_distance_m'])
            metrics={};co={};coverage={}
            for region,mask in masks.items():
                co[region]=dict(center_first_hit_radius_px=distribution(coc[mask&valid]),background_certain_radius_px=distribution(bg_coc[mask&bg_valid]),background_unknown_pixels=int((mask&~bg_valid).sum()))
                coverage[region]=dict(pixels=int(mask.sum()),image_fraction=float(mask.mean()),O1_s_hole_mean=float(oracle['O1_s_hole'][mask].mean()) if mask.any() else None,
                    O1_effective_aperture_sample_fraction=float(oracle['effective_sample_fraction'][mask].mean()) if mask.any() else None)
            for method,image in images.items():
                metrics[method]={};pools[scene].setdefault(method,{})
                for region,mask in masks.items():
                    score=region_sse(image,target,mask);score.update(patch_metrics(image,target,mask,model));metrics[method][region]=score
                    q=pools[scene][method].setdefault(region,dict(pixels=0,sse=0.,patch_lpips_weighted_sum=0.,patch_ssim_weighted_sum=0.,views=0))
                    q['pixels']+=score['pixels'];q['sse']+=score['sse'];q['views']+=bool(score['pixels'])
                    if score['pixels']:
                        q['patch_lpips_weighted_sum']+=score['patch_lpips']*score['pixels'];q['patch_ssim_weighted_sum']+=score['patch_ssim']*score['pixels']
            record['by_view'].append(dict(scene=scene,view=view,metrics=metrics,CoC=co,coverage=coverage))
            print('RESCORED',scene,view,flush=True)
    for scene,methods in pools.items():
        for method,regions in methods.items():
            for region,q in regions.items():
                n=q['pixels'];q.update(psnr=-10*math.log10(max(q['sse']/(3*n),1e-12)) if n else None,
                    patch_lpips=q['patch_lpips_weighted_sum']/n if n else None,patch_ssim=q['patch_ssim_weighted_sum']/n if n else None,
                    image_fraction=n/(3*512*512),distinct_times=1)
                rows.append(dict(scene=scene,method=method,region=region,pixels=n,PSNR=q['psnr'],patch_LPIPS=q['patch_lpips'],image_fraction=q['image_fraction']))
    record['by_scene']=pools;record['O1_minus_O4']={s:{r:dict(PSNR_dB=pools[s]['O1'][r]['psnr']-pools[s]['O4'][r]['psnr'],patch_LPIPS=pools[s]['O1'][r]['patch_lpips']-pools[s]['O4'][r]['patch_lpips']) for r in pools[s]['O1']} for s in pools}
    record['first_level_decision']='Not evaluable: background-depth stop prevented O5/O6; no O6-minus-O5 or O6-minus-O3 values fabricated.'
    a.record.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    with a.record.with_suffix('.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


if __name__=='__main__':main()
