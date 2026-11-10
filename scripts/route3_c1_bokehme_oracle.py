"""Development-only C1 CoC calibration, BokehMe O2/O3/M0 and region metrics.

Run calibrate/evaluate in route3 (LPIPS installed), render in dofgaussian
(existing CuPy). No environment installation or disparity min/max fitting.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from src.camvfx4d import effect_parameters,read_rgb
from src.c1_bokehme import (physical_coc_radius,pipeline_defocus,plane_impulse_checks,
                          region_sse,patch_metrics,region_protocol)


def caches(base):
    mapping=dict(TMPDIR='tmp',TMP='tmp',TEMP='tmp',HF_HOME='huggingface',HF_HUB_CACHE='huggingface/hub',
                 TORCH_HOME='torch',XDG_CACHE_HOME='xdg',PIP_CACHE_DIR='pip',CUPY_CACHE_DIR='cupy',
                 CUDA_CACHE_PATH='cuda',TORCH_EXTENSIONS_DIR='torch_extensions',MPLCONFIGDIR='matplotlib')
    for key,suffix in mapping.items():
        path=base/suffix;path.mkdir(parents=True,exist_ok=True);os.environ[key]=str(path)
    return {k:os.environ[k] for k in mapping}


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def oracle_root(a,scene):
    return a.plant_oracle if scene=='plant' else a.cat_oracle


def calibrate(a):
    checks=plane_impulse_checks();ranges=[]
    for scene in a.scenes:
        params=effect_parameters(a.raw/scene)[0]
        for view in a.views:
            d=np.load(a.regions/scene/str(view)/'background.npz')
            radius,valid=physical_coc_radius(d['mesh_depth'],d['K'][0,0],params['aperture_radius_m'],params['focus_distance_m'])
            values=radius[valid]
            ranges.append(dict(scene=scene,view=view,geometry='independent static mesh, time1 correspondence checked separately',
                               valid_pixels=int(valid.sum()),signed_radius_px_range=[float(values.min()),float(values.max())],
                               signed_radius_px_quantiles=np.quantile(values,[0,.01,.5,.99,1]).tolist(),
                               pipeline_control_range=[float(values.min()/10),float(values.max()/10)],lens=params))
    result=dict(status='physical mapping frozen analytically; CUDA soft-disk impulse check pending render phase',
                development_scenes=a.scenes,test_scenes_used=False,frame=1,
                mapping=dict(formula='signed_radius_px=fx*aperture_radius_m*(1/depth_m-1/focus_m)',
                             pipeline_input='signed_radius_px/10',slope=1.,offset_px=0.,gamma=2.2,defocus_scale=10.,
                             sign='near foreground positive; far background negative; focus axial camera depth',
                             aperture='radius=focal_length_m/(2*fstop)',rgb='sRGB input/output float[0,1]; official gamma2.2 internal convention',
                             invalid_depth_rule='control=0 (focus fallback), invalid count reported; full-image scoring retains those pixels',
                             normalization='none; no per-scene disparity normalization or GT fitting'),
                analytic_point_plane_checks=checks,development_physical_ranges=ranges,
                kernel_limit='Official soft-tanh disk and neural refinement differ from thin-lens disk integration; matching radius does not make kernels identical.',
                parameter_selection='Unit mapping from code and camera geometry only; no method or dataset DOF scores used for fitting.',
                source=dict(repository=str(a.repository),commit=subprocess.check_output(['git','-C',str(a.repository),'rev-parse','HEAD'],text=True).strip(),
                            code_radius='classical_renderer/scatter.py: fltRadius=fabsf(fltDefocus)',
                            code_scale='demo.py pipeline: classical_renderer(image**gamma,defocus*args.defocus_scale)'))
    if a.calibration.exists():raise FileExistsError(a.calibration)
    save(a.calibration,result);print(json.dumps(result,ensure_ascii=False,indent=2))


def gpu_check():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise RuntimeError('Set CUDA_VISIBLE_DEVICES=0 explicitly for the granted device')
    subprocess.run(['nvidia-smi'],check=True)


def cuda_impulse(classical):
    import torch
    image=torch.zeros((1,3,64,64),device='cuda');image[:,:,32,32]=1
    rows=[];positive={}
    for signed in [0.,4.,-4.,8.,-8.]:
        output,_=classical(image,torch.full((1,1,64,64),signed,device='cuda'))
        kernel=output[0,0].float().cpu().numpy();peak=float(kernel[32,32]);axis=kernel[32]
        positions=np.flatnonzero(axis>=peak*.5);half=float((positions.max()-positions.min())/2)
        row=dict(signed_radius_px=signed,half_max_radius_px=half,peak=peak,energy=float(kernel.sum()))
        if signed>0:positive[signed]=kernel
        if signed<0:
            row['sign_flip_kernel_max_abs_error']=float(abs(kernel-positive[-signed]).max())
            assert row['sign_flip_kernel_max_abs_error']<1e-6
        if signed:assert abs(half-abs(signed))<=1
        rows.append(row)
    return dict(checks_passed=True,rows=rows,interpretation='Homogeneous-plane impulse confirms pixel radius, not diameter; sign changes ordering convention, not homogeneous-plane kernel. Tanh tail and neural refinement remain different kernels.')


def render(a):
    gpu_check();import torch
    from src.baselines.bokehme import setup,predict
    frozen=json.loads(a.calibration.read_text(encoding='utf-8'));mapping=frozen['mapping']
    assert mapping['slope']==1 and mapping['offset_px']==0 and mapping['defocus_scale']==10 and mapping['gamma']==2.2
    a.output.mkdir(parents=True,exist_ok=False)
    pipeline,settings,nets,counts=setup(a.repository,a.checkpoints,dict(gamma=2.2,highlight=False,defocus_scale=10.))
    impulse=cuda_impulse(nets['classical']);frozen['cuda_impulse_check']=impulse
    frozen['status']='analytic physical mapping frozen; CUDA soft-disk impulse check passed';save(a.calibration,frozen)
    rows=[]
    for scene in a.scenes:
        params=effect_parameters(a.raw/scene)[0]
        for view in a.views:
            directory=oracle_root(a,scene)/str(view);oracle=np.load(directory/'oracle-renders.npz')
            bg=np.load(a.regions/scene/str(view)/'background.npz');K=bg['K'];depth=bg['mesh_depth']
            control,valid=pipeline_defocus(depth,K[0,0],params['aperture_radius_m'],params['focus_distance_m'])
            pinhole=read_rgb(a.raw/scene/f'pinhole/{view}/000001.png')
            saved=dict(shared_mesh_first_hit_depth=depth,oracle_gaussian_expected_depth=oracle['center_depth'],physical_signed_radius=control*10,depth_valid=valid)
            sources=dict(O2=oracle['center_srgb'],O3=pinhole);controls=dict(O2=control,O3=control)
            movies_path=a.base/f'movies-{scene}-01/depth-alignment.npz'
            if view==0 and movies_path.exists():
                import cv2
                md=np.load(movies_path)['metric_depth'];md=cv2.resize(md,(depth.shape[1],depth.shape[0]),interpolation=cv2.INTER_NEAREST)
                mc,mv=pipeline_defocus(md,K[0,0],params['aperture_radius_m'],params['focus_distance_m'])
                sources['M0']=pinhole;controls['M0']=mc;saved.update(M0_metric_depth=md,M0_physical_signed_radius=mc*10,M0_depth_valid=mv)
            for name,image in sources.items():
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.monotonic()
                prediction,stats=predict(pipeline,settings,nets,np.ascontiguousarray(image,dtype=np.float32),np.ascontiguousarray(controls[name],dtype=np.float32))
                torch.cuda.synchronize();seconds=time.monotonic()-started;saved[name+'_srgb']=prediction
                from PIL import Image
                dest=a.output/scene/str(view);dest.mkdir(parents=True,exist_ok=True)
                Image.fromarray(np.uint8(np.round(prediction*255))).save(dest/(name+'.png'))
                row=dict(scene=scene,view=view,frame=1,method=name,seconds=seconds,torch_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                         invalid_shared_depth_pixels=int((~valid).sum()),stats=stats)
                rows.append(row);print(json.dumps(row),flush=True)
            np.savez_compressed(a.output/scene/str(view)/'bokehme-predictions.npz',**saved)
            save(a.output/'runtime.json',dict(models_parameter_counts=counts,calibration=str(a.calibration),rows=rows,training=False))


def lpips_cpu(a):
    import torch
    import lpips
    torch.set_num_threads(8)
    source=a.alexnet_weights
    if not source.is_file():raise FileNotFoundError('Existing LPIPS AlexNet backbone required; implicit download disabled: '+str(source))
    destination=Path(os.environ['TORCH_HOME'])/'hub/checkpoints/alexnet-owt-7be5be79.pth';destination.parent.mkdir(parents=True,exist_ok=True)
    if not destination.exists():destination.symlink_to(source.resolve())
    # lpips ships its small learned calibration weights; pretrained backbone
    # is the existing file linked above, so constructor needs no network.
    model=lpips.LPIPS(net='alex',version='0.1',spatial=False).eval().cpu()
    return model


def evaluate(a):
    import cv2
    if a.evaluation.exists():raise FileExistsError(a.evaluation)
    model=lpips_cpu(a);views=[];aggregate={};guard=[]
    for scene in a.scenes:
        aggregate[scene]={};params=effect_parameters(a.raw/scene)[0]
        for view in a.views:
            oracle=np.load(oracle_root(a,scene)/str(view)/'oracle-renders.npz')
            prediction=np.load(a.output/scene/str(view)/'bokehme-predictions.npz')
            regions=np.load(a.regions/scene/str(view)/'regions.npz');bg=np.load(a.regions/scene/str(view)/'background.npz')
            masks,radius,valid=region_protocol(regions,1,bg['mesh_depth'],bg['K'],params)
            target=read_rgb(a.raw/scene/f'dof/{view}/000001.png')
            methods=dict(O1=oracle['O1_srgb'],O2=prediction['O2_srgb'],O3=prediction['O3_srgb'],O4=oracle['O4_srgb'])
            if 'M0_srgb' in prediction:methods['M0']=prediction['M0_srgb']
            metrics={};coverage={}
            for region,mask in masks.items():
                if mask.any():
                    coverage[region]=dict(pixels=int(mask.sum()),O1_s_hole_mean=float(oracle['O1_s_hole'][mask].mean()),
                                          O1_s_hole_quantiles=np.quantile(oracle['O1_s_hole'][mask],[0,.5,.9,1]).tolist(),
                                          effective_aperture_sample_fraction_mean=float(oracle['effective_sample_fraction'][mask].mean()),
                                          center_depth_geometry_coverage=float(valid[mask].mean()),
                                          independent_mesh_known_aperture_ray_fraction=float(bg['known_ray_fraction'][mask].mean()))
                else:coverage[region]=dict(pixels=0,status='empty_region')
            for method,image in methods.items():
                metrics[method]={};aggregate[scene].setdefault(method,{})
                for region,mask in masks.items():
                    score=region_sse(image,target,mask);score.update(patch_metrics(image,target,mask,model))
                    metrics[method][region]=score
                    pool=aggregate[scene][method].setdefault(region,dict(sse=0.,pixels=0,valid_views=0,frame_ids=set(),patch_weight_sum=0.,patch_lpips_sum=0.,patch_ssim_sum=0.))
                    pool['sse']+=score['sse'];pool['pixels']+=score['pixels']
                    if score['pixels']:
                        pool['valid_views']+=1;pool['frame_ids'].add(1)
                        pool['patch_weight_sum']+=score['pixels'];pool['patch_lpips_sum']+=score['patch_lpips']*score['pixels'];pool['patch_ssim_sum']+=score['patch_ssim']*score['pixels']
            if 'M0' in methods:
                pinhole=read_rgb(a.raw/scene/f'pinhole/{view}/000001.png')
                own=prediction['M0_depth_valid']&(abs(prediction['M0_physical_signed_radius'])<=1.)
                guard.append(dict(scene=scene,view=view,frame=1,
                                  M0_vs_input_common_oracle_focus_mask=region_sse(methods['M0'],pinhole,masks['in_focus']),
                                  M0_vs_input_own_predicted_depth_focus_mask_diagnostic=region_sse(methods['M0'],pinhole,own),
                                  common_focus_mask_source='independent triangle-mesh first-hit depth (not Gaussian expected depth), shared for M1-vs-M0 comparisons',
                                  prediction_npz=str(a.output/scene/str(view)/'bokehme-predictions.npz')))
            views.append(dict(scene=scene,view=view,frame=1,metrics=metrics,coverage=coverage))
            print('EVALUATED',scene,view,flush=True)
    differences={}
    for scene,methods in aggregate.items():
        for method,regions in methods.items():
            for name,pool in regions.items():
                pool['frame_ids']=sorted(pool['frame_ids']);pool['valid_times']=len(pool['frame_ids']);n=pool['pixels']
                pool['psnr']=-10*math.log10(max(pool['sse']/(3*n),1e-12)) if n else None
                pool['patch_lpips']=pool['patch_lpips_sum']/pool['patch_weight_sum'] if n else None
                pool['patch_ssim']=pool['patch_ssim_sum']/pool['patch_weight_sum'] if n else None
                pool['candidate_minimum_1000pixels_3times_passed']=n>=1000 and pool['valid_times']>=3
        differences[scene]={}
        for region in methods['O1']:
            def delta(left,right):
                l=methods[left][region]['psnr'];r=methods[right][region]['psnr'];return l-r if l is not None and r is not None else None
            differences[scene][region]=dict(O1_minus_O2_dB=delta('O1','O2'),O1_minus_O4_dB=delta('O1','O4'),O2_minus_O3_dB=delta('O2','O3'))
    macro={}
    for method in ['O1','O2','O3','O4']:
        macro[method]={}
        for region in aggregate[a.scenes[0]][method]:
            values=[aggregate[s][method][region]['psnr'] for s in a.scenes if aggregate[s][method][region]['psnr'] is not None]
            macro[method][region]=dict(descriptive_scene_mean_psnr_dB=float(np.mean(values)) if values else None,scenes_with_nonempty_region=len(values),
                                       formally_effective_scenes=sum(aggregate[s][method][region]['candidate_minimum_1000pixels_3times_passed'] for s in a.scenes))
    G=[differences[s]['M_C1_proxy']['O1_minus_O2_dB'] for s in a.scenes if differences[s]['M_C1_proxy']['O1_minus_O2_dB'] is not None]
    result=dict(phase='A1 development only',matched_mesh_time=1,oracle_valid_times=1,calibration=str(a.calibration),scenes=a.scenes,test_method_results=False,
                protocol=dict(rgb='sRGB float[0,1]; shared target and masks for O1-O4; BokehMe official gamma2.2',psnr='scene-pooled RGB SSE/(3*pixels), then equal scene means',
                              patch='64x64 fixed grid, full RGB patches, no mask-outside zeroing; per-patch scores weighted by region support',
                              patch_limitation='LPIPS/SSIM measure context outside a narrow mask; patch support ratios retained for interpretation',
                              patch_ssim='11x11Gaussian sigma1.5,population covariance,valid5pxcrop of each full64px patch',
                              LPIPS='alex,version0.1,spatialFalse; pretrained backbone reused read-only; CPU8threads',
                              regions='full;boundary D_t;in_focus|CoC radius|<=1px;out_of_focus>=2px;M_C1_proxy;dynamic_unrecoverable;static_static',
                              shared_ideal_depth='independent triangle-mesh first intersection for O2,O3 and all common focus masks; Gaussian expected depth retained only as diagnostic; M0 uses own predicted depth for synthesis',
                              M0_comparison_scope='M0 onlyview0 per development scene; use by_view for matched-view comparisons, not M0 versus three-view pooled Oracle',
                              minimum_amount='candidate1000 cumulative pixels and3 distinct times per scene; six-view pixels never imply extra times'),
                by_view=views,by_scene=aggregate,scene_mean_descriptive=macro,differences= differences,
                G_descriptive_dB=float(np.mean(G)) if G else None,G_reliable_for_freezing_threshold=False,
                reliability_reason='Only matched frame1 has known dynamic mesh geometry; cannot satisfy candidate>=3times. Numeric differences are descriptive and do not establish formal C1 success threshold or a hidden-content upper bound.',
                M0_input_focus_guardrails=guard,training=False,
                method_output_directory=str(a.output),oracle_directories={s:str(oracle_root(a,s)) for s in a.scenes},
                runtime=json.loads((a.output/'runtime.json').read_text(encoding='utf-8')))
    save(a.evaluation,result);print(json.dumps(dict(G_descriptive_dB=result['G_descriptive_dB'],differences=differences),indent=2))
    csvrows=[]
    for scene,methods in aggregate.items():
        for method,regions in methods.items():
            for region,q in regions.items():
                csvrows.append(dict(scene=scene,method=method,region=region,pixels=q['pixels'],valid_times=q['valid_times'],valid_views=q['valid_views'],pooled_psnr=q['psnr'],patch_lpips=q['patch_lpips'],patch_ssim=q['patch_ssim'],candidate_minimum_1000pixels_3times_passed=q['candidate_minimum_1000pixels_3times_passed']))
    with a.evaluation.with_suffix('.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(csvrows[0]),lineterminator='\n');writer.writeheader();writer.writerows(csvrows)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['calibrate','render','evaluate'],required=True)
    p.add_argument('--base',type=Path,default=ROOT/'outputs/route3-c1-2026-11-10-01');p.add_argument('--raw',type=Path,default=ROOT/'data/raw/CamVFX-4D')
    p.add_argument('--regions',type=Path);p.add_argument('--plant-oracle',type=Path);p.add_argument('--cat-oracle',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--scenes',choices=['plant','cat'],nargs='+',default=['plant','cat']);p.add_argument('--views',type=int,nargs='+',default=[0,16,33])
    p.add_argument('--repository',type=Path,default=ROOT.parent/'BokehMe');p.add_argument('--checkpoints',type=Path,default=ROOT/'models/BokehMe')
    p.add_argument('--cache',type=Path,default=ROOT.parent/'route3-cache')
    p.add_argument('--alexnet-weights',type=Path,default=ROOT.parent/'.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth')
    p.add_argument('--calibration',type=Path,default=ROOT/'03_实验记录/records/route3-c1-2026-11-10-coc-calibration.json')
    p.add_argument('--evaluation',type=Path,default=ROOT/'03_实验记录/records/route3-c1-2026-11-10-oracle-evaluation.json')
    a=p.parse_args();a.base=a.base.resolve();a.regions=a.regions or a.base/'regions-dev';a.output=a.output or a.base/'bokehme-oracle-01'
    a.plant_oracle=a.plant_oracle or a.base/'oracle-final/plant';a.cat_oracle=a.cat_oracle or a.base/'oracle-final/cat'
    if any(v not in [0,16,33] for v in a.views):raise ValueError('Only A0 development views0/16/33 are authorized')
    caches(a.cache.resolve())
    {'calibrate':calibrate,'render':render,'evaluate':evaluate}[a.mode](a)


if __name__=='__main__':main()
