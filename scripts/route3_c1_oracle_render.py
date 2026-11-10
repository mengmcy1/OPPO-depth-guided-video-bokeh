"""O1/O4 fixed-geometry development Oracle: color fitting and lens rendering.

This fits only per-surfel linear RGB, using all 50 views of the matched time.
No learned-model parameters, geometry, scales, opacity or poses are optimized.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
import torch
from gsplat import rasterization

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.aperture_renderer import GaussianScene,render_aperture,linear_to_srgb
from src.camvfx4d import load_camera_frames,intrinsics_from_fov,read_rgb,effect_parameters
from src.c1_regions import linear_rgb


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scene',type=Path,required=True)
    p.add_argument('--geometry',type=Path,required=True);p.add_argument('--regions',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--colors-from',type=Path,help='Reuse previously fitted colors without rerunning optimization')
    p.add_argument('--fit-reused-colors',action='store_true',help='Continue fixed-geometry color fitting from the reused values')
    p.add_argument('--color-iterations',type=int,default=8)
    a=p.parse_args()
    if a.scene.name not in ['plant','cat']:raise ValueError('Development scenes only')
    a.output.mkdir(parents=True,exist_ok=False);cv2.setNumThreads(4);torch.manual_seed(20261110)
    frames,meta=load_camera_frames(a.scene/'pinhole/transforms_train.json');K=torch.tensor(intrinsics_from_fov(meta),device='cuda',dtype=torch.float32)
    d=np.load(a.geometry);frame=int(d['mesh_frame_id']);c=json.loads(Path('configs/route3-c1-a1-20261110.json').read_text())
    tensors={k:torch.tensor(d[k],device='cuda') for k in ['means','quats','scales','opacities','source_frames','colors_linear']}
    colors=torch.nn.Parameter(torch.tensor(np.load(a.colors_from),device='cuda') if a.colors_from else tensors['colors_linear']);scene=GaussianScene(tensors['means'],tensors['quats'],tensors['scales'],tensors['opacities'],colors,tensors['source_frames'])
    parameters=effect_parameters(a.scene)[0];F=parameters['focus_distance_m'];A=parameters['aperture_radius_m'];sample_count=c['aperture_samples']
    coverage=[]
    for view in c['views']:
        E=torch.tensor(np.linalg.inv(frames[(view,frame)]['c2w']),device='cuda',dtype=torch.float32)
        masks=np.load(a.regions/a.scene.name/str(view)/'regions.npz');idx=masks['frame_ids'].tolist().index(frame)
        region=torch.tensor(masks['M_C1_proxy'][idx],device='cuda')
        with torch.no_grad():r=render_aperture(scene,K,E,512,512,F,A,sample_count,return_samples=True)
        hole=float(r['s_hole'][region].mean()) if region.any() else None
        effective=float((r['sample_alphas'][:,region]>=.95).float().mean()) if region.any() else None
        row=dict(view=view,region_pixels=int(region.sum()),mean_s_hole=hole,effective_aperture_sample_fraction_alpha_ge_095=effective,
                 geometry_coverage_fraction=float((r['O_coverage'][region]>=.95).float().mean()) if region.any() else None)
        coverage.append(row);print('COVERAGE',json.dumps(row),flush=True)
        del r
    record=dict(scene=a.scene.name,mesh_frame_id=frame,gaussians=len(colors),K=sample_count,coverage=coverage,
        coverage_gate=dict(maximum_mean_hole=.02,minimum_effective_sample_fraction=.95),color_fitting=[])
    if any(r['mean_s_hole'] is None or r['mean_s_hole']>.02 or r['effective_aperture_sample_fraction_alpha_ge_095']<.95 for r in coverage):
        record['stop_condition']='O1 coverage insufficient';(a.output/'summary.json').write_text(json.dumps(record,indent=2)+'\n');return
    views=list(range(50));torch.cuda.reset_peak_memory_stats();start=time.monotonic()
    targets=[torch.tensor(linear_rgb(read_rgb(a.scene/f'pinhole/{v}/{frame:06d}.png')),device='cuda') for v in views]
    cameras=[torch.tensor(np.linalg.inv(frames[(v,frame)]['c2w']),device='cuda',dtype=torch.float32) for v in views]
    def image_at(view):
        return rasterization(scene.means,scene.quats,scene.scales,scene.opacities,colors,cameras[view][None],K[None],512,512,
                             packed=False,render_mode='RGB',near_plane=1e-4,far_plane=1e4)[0][0]
    do_fit=not a.colors_from or a.fit_reused_colors
    if do_fit:
        # Fixed alpha weights make image=A*color linear and nonnegative.
        # A^T 1 majorizes the Hessian row sum (alpha<=1). Full-view projected
        # Jacobi steps avoid sequential-view Adam color oscillations.
        colors.grad=None
        for view in views:image_at(view).sum().backward()
        denominator=colors.grad.detach().clone().clamp_min(1e-6)
        previous=None
        for iteration in range(a.color_iterations):
            colors.grad=None;loss_sum=0
            for view in views:
                residual=image_at(view)-targets[view]
                loss=.5*residual.square().sum();loss.backward();loss_sum+=float(residual.detach().square().mean())
            mse=loss_sum/50
            if previous is not None and mse>previous+1e-7:raise RuntimeError('Full-view color objective increased')
            with torch.no_grad():colors.sub_(colors.grad/denominator).clamp_(0,1)
            previous=mse
            record['color_fitting'].append(dict(iteration=iteration+1,all_50_views_mean_linear_mse_before_step=mse))
            print('COLOR_FIT',json.dumps(record['color_fitting'][-1]),flush=True)
    torch.cuda.synchronize();record['color_optimization_seconds']=time.monotonic()-start if do_fit else None;record['peak_allocated_mib']=torch.cuda.max_memory_allocated()/2**20
    record['color_solver']='full-50-view projected Jacobi, A^T1 preconditioner; step1; fixed nonnegative alpha weights'
    record['reused_optimized_colors']=str(a.colors_from) if a.colors_from else None
    record.update(fixed=['means','quats','scales','opacities','poses','lens'],optimized=['linear RGB colors'],model_training=False,
                  O4_hole_background_linear_rgb=[0,0,0],RGB_evaluation='display sRGB float32 [0,1]; 3D integration linear RGB',
                  scale='dataset metric mesh coordinates, same lens as all 3D groups; no depth shift')
    np.save(a.output/'optimized-colors-linear.npy',colors.detach().cpu().numpy())
    (a.output/'summary.json').write_text(json.dumps(record,indent=2)+'\n')
    for view in c['views']:
        E=torch.tensor(np.linalg.inv(frames[(view,frame)]['c2w']),device='cuda',dtype=torch.float32)
        out=a.output/str(view);out.mkdir()
        visibility_path=a.geometry.parent/f'center-visible-frustum-{view}.npy'
        if not visibility_path.exists():raise FileNotFoundError('Exact first-hit surface visibility required for O4: '+str(visibility_path))
        visible=torch.tensor(np.load(visibility_path),device='cuda')
        partial=GaussianScene(*(getattr(scene,k)[visible] for k in scene.__dataclass_fields__))
        with torch.no_grad():
            center=render_aperture(scene,K,E,512,512,F,0.,1)
            full=render_aperture(scene,K,E,512,512,F,A,sample_count,return_samples=True)
            ablated=render_aperture(partial,K,E,512,512,F,A,sample_count)
        values=dict(O1_srgb=linear_to_srgb(full['rgb_linear']).clamp(0,1).cpu().numpy(),
                    O4_srgb=linear_to_srgb(ablated['rgb_linear']).clamp(0,1).cpu().numpy(),
                    center_srgb=linear_to_srgb(center['rgb_linear']).clamp(0,1).cpu().numpy(),
                    center_depth=center['depth'].cpu().numpy(),O1_s_hole=full['s_hole'].cpu().numpy(),
                    effective_sample_fraction=(full['sample_alphas']>=.95).float().mean(0).cpu().numpy())
        np.savez_compressed(out/'oracle-renders.npz',**values)
        for key in ['O1_srgb','O4_srgb','center_srgb']:cv2.imwrite(str(out/(key+'.png')),np.uint8(values[key][...,::-1]*255))
    (a.output/'summary.json').write_text(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':main()
