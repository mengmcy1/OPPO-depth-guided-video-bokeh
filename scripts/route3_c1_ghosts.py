"""Source/ghost development diagnostics from saved same-forward Gaussians.

Box is an explicitly allowed test-scene diagnosis: images and source masses
only, with no PSNR/LPIPS, ranking or M2 evaluation. Its one-query cache cannot
establish a multiquery dynamic label; source-pixel motion masks are used there.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.aperture_renderer import GaussianScene,render_aperture,srgb_to_linear,linear_to_srgb
from src.camvfx4d import effect_parameters,read_rgb
from src.c1_fusion import multiquery_dynamic,select_gaussians,FusionSettings


def subset(scene, keep):
    return GaussianScene(*(getattr(scene,k)[keep] for k in scene.__dataclass_fields__))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scene',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--regions',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--view',type=int,default=0)
    p.add_argument('--visibility-relative-margin',type=float,default=.1)
    p.add_argument('--only-m2b',action='store_true',help='Minimal development follow-up; reuse earlier M1 results')
    a=p.parse_args();name=a.scene.name
    if name not in ['plant','cat','box']:raise ValueError('Only development scenes and explicit cached box diagnosis')
    if name=='box' and a.only_m2b:raise ValueError('No M2 test-scene evaluation in A1')
    a.output.mkdir(parents=True,exist_ok=False);cv2.setNumThreads(4);d=np.load(a.cache/'metric_gaussians.npz')
    t={k:torch.tensor(d[k],device='cuda') for k in ['means','quats','scales','opacities','source_frames','K','world_to_camera','colors_srgb']}
    scene=GaussianScene(t['means'],t['quats'],t['scales'],t['opacities'],srgb_to_linear(t['colors_srgb']),t['source_frames'])
    background=np.load(a.regions/name/str(a.view)/'background.npz');masks=np.load(a.regions/name/str(a.view)/'regions.npz')
    frame_ids=masks['frame_ids'].tolist();input_indices=np.array(frame_ids)-1
    motion=background['motion_masks'];current=torch.tensor(motion[0],device='cuda');board=torch.tensor(background['linear_rgb'],device='cuda')
    target=torch.tensor(read_rgb(a.scene/f'pinhole/{a.view}/000001.png'),device='cuda');target_linear=srgb_to_linear(target)
    if 'source_flat_indices' in d:
        flat=d['source_flat_indices'];offsets=torch.tensor(d['query_offsets_normalized'],device='cuda')
        dynamic=multiquery_dynamic(offsets,.005);label='same-Gaussian pairwise displacement across queries0,25/49,1; threshold .005 normalized units'
        depth=torch.tensor(np.load(a.cache/'depth-alignment.npz')['metric_depth'],device='cuda')[None,None]
        depth=torch.nn.functional.interpolate(depth,size=(512,512),mode='nearest')[0,0]
    else:
        saved=torch.load(a.cache/'same_forward_outputs.pt',map_location='cpu',weights_only=True)
        opacity=saved['query_motion_gaussians'][0]['motion_opacity'][0].permute(0,2,3,1).reshape(-1)
        flat=((torch.tanh(opacity-1)*.5+.5)>.001).nonzero().flatten().numpy()
        assert len(flat)==len(scene.means)
        resized=np.stack([cv2.resize(motion[i].astype(np.uint8),(518,518),interpolation=cv2.INTER_NEAREST)>0 for i in input_indices])
        dynamic=torch.tensor(resized.reshape(-1)[flat],device='cuda')
        label='diagnostic source-pixel motion-mask labels; one-query box cache cannot determine temporal Gaussian motion range'
        depth=None
    parameters=effect_parameters(a.scene)[0];K=t['K'];E=t['world_to_camera'];lens=(parameters['focus_distance_m'],parameters['aperture_radius_m'])
    methods={'M1-cur':scene.select([0]),'M1':scene,'static-sources':subset(scene,~dynamic),'dynamic-sources':subset(scene,dynamic)}
    fusion_stats={}
    if name in ['plant','cat']:
        for steps in [1,2]:
            for visibility in [False,True]:
                settings=FusionSettings(True,visibility,neighbor_steps=steps,visibility_relative_margin=a.visibility_relative_margin)
                keep,stats=select_gaussians(scene.means,scene.source_frames,dynamic,torch.tensor(input_indices,device='cuda'),0,K,E,depth,settings,current)
                if visibility:
                    before,_=select_gaussians(scene.means,scene.source_frames,dynamic,torch.tensor(input_indices,device='cuda'),0,K,E,depth,FusionSettings(True,False,neighbor_steps=steps))
                    cp=scene.means@E[:3,:3].T+E[:3,3];uvz=cp@K.T;uv=(uvz[:,:2]/uvz[:,2:].clamp_min(1e-8)).floor().long()
                    inside=(cp[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<512)&(uv[:,1]>=0)&(uv[:,1]<512)
                    u,v=uv[:,0].clamp(0,511),uv[:,1].clamp(0,511)
                    mesh_depth=torch.tensor(background['mesh_depth'],device='cuda')[v,u]
                    stable=torch.tensor(~motion.any(0)&background['valid'],device='cuda')[v,u]
                    consistent=inside&stable&(mesh_depth>0)&((cp[:,2]-mesh_depth).abs()<=.01*mesh_depth+.001)
                    stats['mesh_consistent_stable_background_candidates']=int((before&consistent).sum())
                    stats['deleted_mesh_consistent_static_proxy']=int((before&~keep&consistent).sum())
                    stats['false_deletion_proxy_tolerance']='independent stable background and mesh first-hit depth within1%+1mm; not dynamic GT'
                key=f'M2{"b" if visibility else "a"}-window{steps}';methods[key]=subset(scene,keep);fusion_stats[key]=stats
    if a.only_m2b:methods={k:v for k,v in methods.items() if k.startswith('M2b')}
    record=dict(scene=name,view=a.view,query_frame_id=1,split='test_diagnostic_only' if name=='box' else 'development',
        dynamic_label=label,dynamic_fraction=float(dynamic.float().mean()),K=16,
        scale_to_meters=float(d['scale_to_meters']),source_indexing='zero-based global frame index',
        quality_metrics_on_test=False,source_contribution_interpretation='alpha mass in disagreement regions, not causal attribution of error',
        fusion_switches=fusion_stats,renders=[])
    record['visibility_relative_margin']=a.visibility_relative_margin
    record['diagnostic_thresholds']=dict(ghost_proxy_max_channel_linear_rgb_error=.05,
        dynamic_projection_minimum_alpha=.1,multiquery_motion_range_normalized=.005,
        visibility_absolute_margin_m=.01,native_opacity_minimum=.001,
        mesh_consistent_static_proxy_relative_tolerance=.01,mesh_consistent_static_proxy_absolute_tolerance_m=.001)
    torch.cuda.reset_peak_memory_stats();start=time.monotonic()
    for key,representation in methods.items():
        for mode,samples,aperture in [('pinhole',1,0.),('dof',16,lens[1])]:
            with torch.no_grad():r=render_aperture(representation,K,E,512,512,lens[0],aperture,samples,current_frame=0)
            rgb=linear_to_srgb(r['rgb_linear']).clamp(0,1)
            cv2.imwrite(str(a.output/f'{key}-{mode}.png'),np.uint8(rgb.cpu().numpy()[...,::-1]*255))
            np.savez_compressed(a.output/f'{key}-{mode}.npz',rgb_srgb=rgb.cpu().numpy(),s_cur=r['s_cur'].cpu().numpy(),
                s_other=r['s_other'].cpu().numpy(),s_hole=r['s_hole'].cpu().numpy())
            row=dict(method=key,mode=mode,gaussians=len(representation.means))
            if name!='box':
                for region_name in ['M_C1_proxy','dynamic_unrecoverable','static_static']:
                    region=torch.tensor(masks[region_name][0],device='cuda')
                    row[region_name+'_source_fractions']={field:float(r[field][region].mean()) if region.any() else None for field in ['s_cur','s_other','s_hole']}
            if key in ['M1','M1-cur']:
                ever=torch.tensor(motion.any(0),device='cuda')
                disagreement=(r['rgb_linear']-target_linear).abs().amax(-1)>.05
                ghost_proxy=ever&~current&disagreement
                row['sources']=[dict(source_frame_id=int(source)+1,alpha_mass_in_current_motion=float(r['t_ref_weights'][...,j][current].sum()),
                    alpha_mass_in_ghost_disagreement_proxy=float(r['t_ref_weights'][...,j][ghost_proxy].sum())) for j,source in enumerate(r['source_ids'])]
                if name!='box':row['ghost_disagreement_proxy_pixels']=int(ghost_proxy.sum())
            if name!='box':
                if mode=='pinhole' and key=='static-sources':
                    row['static_source_vs_background_linear_MAE_in_Mt']=float((r['rgb_linear'][current]-board[current]).abs().mean()) if current.any() else None
                if mode=='pinhole' and key=='dynamic-sources':
                    projection=r['O_coverage']>.1;union=(projection|current).sum()
                    row['dynamic_projection_IoU_with_Mt']=float((projection&current).sum()/union) if union else None
                mask=torch.tensor(masks['M_C1_proxy'][0],device='cuda')
                for region_name,region in [('full',torch.ones_like(current)),('M_C1_proxy',mask)]:
                    truth=target if mode=='pinhole' else torch.tensor(read_rgb(a.scene/f'dof/{a.view}/000001.png'),device='cuda')
                    row[region_name+'_PSNR_development_only']=float(-10*((rgb[region]-truth[region]).square().mean()).clamp_min(1e-12).log10()) if region.any() else None
            record['renders'].append(row);print('GHOST_RENDER',key,mode,flush=True)
    torch.cuda.synchronize();record['render_seconds']=time.monotonic()-start;record['peak_allocated_mib']=torch.cuda.max_memory_allocated()/2**20
    record['time_convention_audit']=dict(official_commit='77262fac6724a12c0b133a4fee10e62772db6d0f',
        official_files=['src/data/*_dataset.py (times normalized0..1)','src/models/networks/vggsplat.py:300-350','src/models/splatrecon.py:110-129','src/models/gs_render/gs_renderer.py:80-88'],
        query_axis='B,F_out,F_in,3+conf,H,W; query0 chosen, xyz channels only',
        application='offset is identity-activated and ADDED to source unprojected xyz; same canonical world frame',
        scale='one scalar multiplies means, scales, offsets and depth; lens dimensions in meters',
        conclusion='matches official API indexing and normalization; this does not rule out network time-conditioning errors')
    (a.output/'summary.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)


if __name__=='__main__':main()
