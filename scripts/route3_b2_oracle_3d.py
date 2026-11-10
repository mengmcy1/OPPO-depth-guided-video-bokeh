"""Frozen B2 O5/O6 preparation and shared three-dimensional lens rendering.

Prepare in the existing CPU mesh environment, render in route3. Both modes
require frozen criteria. Hidden parents come from per-object peeled references
and are filtered by actual visibility of the same local surface in12 inputs.
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

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.c1_regions import linear_rgb,display_rgb


def save_json(path,value):
    temp=path.with_name(path.name+'.part');temp.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8');temp.replace(path)


def save_npz(path,**values):
    temp=path.with_name(path.name+'.part')
    with temp.open('wb') as handle:np.savez_compressed(handle,**values)
    temp.replace(path)


def cache_paths():
    base=Path('/root/autodl-tmp/route3-cache')
    for key,folder in dict(TMPDIR='tmp',HF_HOME='hf',TORCH_HOME='torch',XDG_CACHE_HOME='xdg',MPLCONFIGDIR='matplotlib',CUDA_CACHE_PATH='cuda',TORCH_EXTENSIONS_DIR='torch_extensions').items():
        path=base/folder;path.mkdir(parents=True,exist_ok=True);os.environ[key]=str(path)


def input_path(entry):
    path=Path(entry['input_npz']);return path if path.is_absolute() else ROOT/path


def read_exr(path):
    image=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim!=3 or image.shape[-1]<3:raise ValueError('Missing RGB EXR: '+str(path))
    rgb=np.ascontiguousarray(image[...,:3][...,::-1],dtype=np.float32)
    if not np.isfinite(rgb).all():raise ValueError('Nonfinite reference EXR')
    return rgb


def deduplicate(pixel,depth,priority,reference,tolerance=1e-5):
    """Same pixel, anchored depth clusters; prefer fewer hidden objects."""
    if not len(pixel):return np.empty(0,np.int64)
    order=np.lexsort((reference,priority,depth,pixel));selected=[];group=[];anchor=0.;last_pixel=-1
    for index in order:
        if pixel[index]!=last_pixel or depth[index]-anchor>tolerance:
            if group:selected.append(min(group,key=lambda i:(priority[i],reference[i])))
            group=[int(index)];anchor=float(depth[index]);last_pixel=int(pixel[index])
        else:group.append(int(index))
    if group:selected.append(min(group,key=lambda i:(priority[i],reference[i])))
    return np.asarray(selected,np.int64)


def prepare(a,entries,criteria):
    from route3_blender_b2_full_geometry import FullWorld
    from route3_b2_oracle_2d import shared_input
    groups={}
    for entry in entries:groups.setdefault((entry['scene'],int(entry['frame_id'])),entry)
    rows=[]
    for (scene,frame),entry in sorted(groups.items()):
        started=time.monotonic();out=a.output/'prepared'/f'{scene}-f{frame:03d}';out.mkdir(parents=True,exist_ok=True)
        if (out/'parents.npz').exists():raise FileExistsError(out/'parents.npz')
        data=np.load(input_path(entry));shared_input(data)
        center=np.asarray(data['center_depth_raw'],np.float32);known=np.asarray(data['center_surface_valid'],bool)
        K=np.asarray(data['K'],np.float64);camera=np.asarray(data['camera_c2w'],np.float64);h,w=center.shape
        y,x=np.mgrid[:h,:w];uv=np.stack([x.ravel()+.5,y.ravel()+.5],-1).astype(np.float32)
        rays=np.c_[uv,np.ones(h*w)]@np.linalg.inv(K).T;ray_norm=np.linalg.norm(rays,axis=-1)
        world=FullWorld(a.scene_root/scene/'scene-geometry.npz',h)
        refdir=a.base/'references'/scene;geometry=a.base/scene/'geometry'
        metadata=json.loads((refdir/'timings.json').read_text());refs=sorted([r for r in metadata if int(r['frame'])==frame],key=lambda r:(len(r['hidden_object_ids']),r['hidden_object_ids']))
        if not refs:raise ValueError('No peeled references for '+scene)
        parts={key:[] for key in ['pixel','depth','colors','object_id','triangle_id','local_point','priority','reference']};source_rows=[]
        for index,ref in enumerate(refs):
            excluded=ref['hidden_object_ids'];key=f'f{frame:03d}-exclude-'+'-'.join(map(str,excluded))
            geom_path=geometry/(key+'.npz');g=np.load(geom_path);depth=np.asarray(g['depth_axial_m'],np.float32)
            if not np.allclose(g['K'],K) or not np.allclose(g['camera_c2w'],camera):raise ValueError('Reference camera differs from common input')
            if sorted(g['excluded_object_ids'].tolist())!=sorted(excluded):raise ValueError('Reference exclusion mismatch')
            good=known&g['center_visible']&(depth>0)
            good&=((depth-center)*ray_norm.reshape(h,w)>1e-4)
            ids=np.flatnonzero(good.ravel())
            raw=read_exr(refdir/(ref['key']+'.exr'))
            color=linear_rgb(display_rgb(np.clip(raw,0,1))).reshape(-1,3)[ids]
            parts['pixel'].append(ids);parts['depth'].append(depth.ravel()[ids]);parts['colors'].append(color)
            parts['object_id'].append(g['object_id'].ravel()[ids]);parts['triangle_id'].append(g['triangle_id'].ravel()[ids])
            parts['local_point'].append(g['local_surface_point'].reshape(-1,3)[ids])
            parts['priority'].append(np.full(len(ids),len(excluded),np.uint8));parts['reference'].append(np.full(len(ids),index,np.int16))
            source_rows.append(dict(reference=index,key=key,EXR=str(refdir/(ref['key']+'.exr')),geometry=str(geom_path),excluded_object_ids=excluded,
                selected_hidden_pixels=len(ids),HDR_clipped_fraction_full=float(((raw<0)|(raw>1)).any(-1).mean()),
                HDR_clipped_fraction_selected=float(((raw.reshape(-1,3)[ids]<0)|(raw.reshape(-1,3)[ids]>1)).any(-1).mean()) if len(ids) else None))
        values={key:np.concatenate(v,axis=0) for key,v in parts.items()}
        keep=deduplicate(values['pixel'],values['depth'],values['priority'],values['reference'])
        before=len(values['pixel']);values={key:v[keep] for key,v in values.items()}
        obj=values['object_id'];local=values['local_point'].astype(np.float64);n=len(obj)
        recover=np.zeros(n,bool);first_seen=np.full(n,-1,np.int16);count=np.zeros(n,np.uint8)
        for other in world.d['input_frames']:
            other=int(other)
            if other==frame:continue
            point=world.world(local,obj,other);visible,*_=world.visible_at(other,other,point,obj,tolerance=1e-4)
            recover|=visible;count+=visible.astype(np.uint8);first_seen[(first_seen<0)&visible]=other
        assert np.array_equal(count>0,recover) and np.array_equal(first_seen>=0,recover)
        assert not np.any(first_seen[recover]==frame)
        save_npz(out/'parents.npz',uv=uv[values['pixel']],depth=values['depth'],colors_linear=values['colors'],
            pixel_index=values['pixel'],object_id=obj,triangle_id=values['triangle_id'],local_surface_point=values['local_point'],
            reference_index=values['reference'],reference_hide_count=values['priority'],recoverable=recover,
            first_visible_other_frame=first_seen,visible_other_input_count=count,K=K,camera_c2w=camera,input_frame_id=frame)
        repeated_pixels=int(n-len(np.unique(values['pixel'])))
        row=dict(scene=scene,frame_id=frame,input_npz=str(input_path(entry)),reference_count=len(refs),references=source_rows,
            parents_before_dedup=before,hidden_parent_count=n,recoverable_hidden_parent_count=int(recover.sum()),
            additional_hidden_depths_at_same_pixel=repeated_pixels,dedup_depth_tolerance_m=1e-5,
            dedup='same pixel/depth cluster, prefer fewer excluded objects; distinct depths at same pixel retained',
            hidden_condition='original center has mesh hit; peeled reference first-hit range>original range+1e-4m',
            visibility='same local point visible in one of12 other actual input poses/geometries; no object-ID-only or aperture-mask flag',
            preparation_seconds=time.monotonic()-started,criteria=str(a.criteria),parameters=criteria['optics']['pixel_initialization'])
        rows.append(row);save_json(out/'metadata.json',row);print('PREPARED',scene,frame,n,int(recover.sum()),flush=True)
    save_json(a.record,dict(mode='prepare',completed=True,criteria=str(a.criteria),manifest=str(a.manifest),rows=rows,
        discretization='all25 child splats inherit parent center color/depth and actual-source support; finite pixel surfels, no surface baking or optimization'))


def sparse_pixel_layer(parent,selected,K,c2w,parameters,device):
    import torch
    from scipy.spatial.transform import Rotation
    from src.aperture_renderer import GaussianScene
    subdivision=int(parameters['subdivisions']);dtype=torch.float32
    uv=torch.tensor(parent['uv'][selected],device=device,dtype=dtype);depth=torch.tensor(parent['depth'][selected],device=device,dtype=dtype)
    color=torch.tensor(parent['colors_linear'][selected],device=device,dtype=dtype)
    axis=(torch.arange(subdivision,device=device,dtype=dtype)+.5)/subdivision-.5
    yy,xx=torch.meshgrid(axis,axis,indexing='ij');offset=torch.stack([xx.flatten(),yy.flatten()],-1)
    sample_uv=(uv[:,None,:]+offset[None]).reshape(-1,2);z=depth.repeat_interleave(subdivision**2)
    rays=torch.cat([sample_uv,torch.ones_like(sample_uv[:,:1])],-1)@torch.linalg.inv(K).T
    points=rays*z[:,None];means=points@c2w[:3,:3].T+c2w[:3,3]
    sigma=z/K[0,0]*(float(parameters['sigma_pixels'])/subdivision)
    scales=torch.stack([sigma,sigma,sigma*.01],-1)
    q=Rotation.from_matrix(c2w[:3,:3].cpu().double().numpy()).as_quat()[[3,0,1,2]]
    quats=torch.tensor(np.array(q),device=device,dtype=dtype)[None].expand(len(means),4).contiguous()
    return GaussianScene(means,quats,scales,torch.full_like(sigma,float(parameters['opacity'])),
        color.repeat_interleave(subdivision**2,0),torch.ones(len(means),device=device,dtype=torch.long))


def render(a,entries,criteria):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise ValueError('Use parent-granted CUDA_VISIBLE_DEVICES=0')
    subprocess.run(['nvidia-smi'],check=True)
    import torch
    from src.aperture_renderer import linear_to_srgb
    from src.layered_aperture import pixel_layer,render_layered
    from route3_b2_oracle_2d import shared_input,depth_protocol
    torch.set_num_threads(4);groups={};rows=[];parameters=criteria['optics']['pixel_initialization'];samples=int(criteria['optics']['K'])
    fill_entries=json.loads(a.fill_manifest.read_text())['items'];fill_ids={(r['scene'],int(r['frame_id'])):r['id'] for r in fill_entries}
    for entry in entries:groups.setdefault((entry['scene'],int(entry['frame_id'])),[]).append(entry)
    for (scene,frame),items in sorted(groups.items()):
        frame_started=time.monotonic();data=np.load(input_path(items[0]));rgb,visible_rgb,_=shared_input(data);depth,valid,_=depth_protocol(data)
        h,w=depth.shape;K=torch.tensor(data['K'],device='cuda',dtype=torch.float32);c2w=torch.tensor(data['camera_c2w'],device='cuda',dtype=torch.float32);E=torch.linalg.inv(c2w)
        visible=pixel_layer(torch.tensor(visible_rgb,device='cuda'),torch.tensor(depth,device='cuda'),K,c2w,
            valid=torch.tensor(valid,device='cuda'),subdivisions=int(parameters['subdivisions']),sigma_pixels=float(parameters['sigma_pixels']),opacity=float(parameters['opacity']))
        parent=np.load(a.output/'prepared'/f'{scene}-f{frame:03d}'/'parents.npz')
        if int(parent['input_frame_id'])!=frame or not np.allclose(parent['K'],data['K']) or not np.allclose(parent['camera_c2w'],data['camera_c2w']):
            raise ValueError('Prepared hidden parents do not match the common frame camera')
        all_hidden=sparse_pixel_layer(parent,np.ones(len(parent['depth']),bool),K,c2w,parameters,'cuda')
        rec_hidden=sparse_pixel_layer(parent,parent['recoverable'],K,c2w,parameters,'cuda')
        fill_id=fill_ids[(scene,frame)];fill_path=a.fill_root/fill_id/'fill-image.npz';fill=np.load(fill_path)
        if not np.array_equal(fill['fill_mask'],np.asarray(data['fill_mask'],bool)):
            raise ValueError('LaMa fill mask differs from the common O5/O6 mask')
        fill_rgb=torch.tensor(fill['fill_rgb_linear'],device='cuda',dtype=torch.float32);bgdepth=torch.tensor(data['background_depth'],device='cuda',dtype=torch.float32)
        torch.cuda.synchronize();build_seconds=time.monotonic()-frame_started
        for entry in items:
            gt=np.load(entry['geometry_gt_npz']);mask=torch.tensor(gt['mask_recoverable_total'],device='cuda')
            for method,hidden in [('O5',None),('O6-GT-all',all_hidden),('O6-GT-rec',rec_hidden)]:
                out=a.output/entry['id']/method;out.mkdir(parents=True,exist_ok=False)
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.monotonic()
                with torch.inference_mode():
                    result=render_layered(visible,hidden,K,E,w,h,float(entry['focus_distance_m']),float(entry['aperture_radius_m']),samples,
                        fill_rgb,bgdepth,return_samples=True,eps2d=float(parameters['eps2d']))
                    prediction=linear_to_srgb(result['rgb_linear']).clamp(0,1)
                torch.cuda.synchronize();seconds=time.monotonic()-started
                if not np.allclose(result['aperture_xy'].cpu().numpy(),gt['aperture_xy'],atol=1e-7,rtol=1e-6):raise ValueError('Lens sample order differs from geometric GT')
                arrays=dict(prediction_srgb=prediction.cpu().numpy(),alpha=result['alpha'].cpu().numpy(),
                    visible_contribution=result['s_cur'].cpu().numpy(),hidden_contribution=result['s_other'].cpu().numpy(),fill_fraction=result['s_hole'].cpu().numpy(),
                    fill_background_depth_known=result['fill_background_depth_known'].cpu().numpy(),
                    fill_projection_inside_fraction=result['fill_projection_inside_fraction'].cpu().numpy())
                save_npz(out/'prediction.npz',**arrays);cv2.imwrite(str(out/'prediction.png'),np.uint8(arrays['prediction_srgb'][...,::-1]*255))
                raw_domain=gt['known']&~gt['hidden'] if method=='O5' else (gt['hidden'] if method=='O6-GT-all' else gt['recoverable'])&~gt['residual']
                domain=torch.tensor(raw_domain,device='cuda');alpha=result['sample_alphas'][domain]
                coverage=dict(rays=int(domain.sum()),domain='known visible' if method=='O5' else ('all supported hidden' if method=='O6-GT-all' else 'actual recovered supported hidden'),
                    mean_hole=float((1-alpha).mean()) if alpha.numel() else None,
                    alpha_ge_095_fraction=float((alpha>=.95).float().mean()) if alpha.numel() else None)
                coverage['old_O1_reference_thresholds_passed']=(coverage['mean_hole']<=.02 and coverage['alpha_ge_095_fraction']>=.95) if alpha.numel() else None
                row=dict(id=entry['id'],scene=scene,frame_id=frame,method=method,focus_distance_m=entry['focus_distance_m'],aperture_radius_m=entry['aperture_radius_m'],
                    visible_Gaussians=len(visible.means),hidden_Gaussians=0 if hidden is None else len(hidden.means),render_seconds=seconds,
                    peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,frame_build_seconds=build_seconds,K=samples,
                    shared_fill=str(fill_path),GT_fill_mask=True,GT_background_depth=True,geometry_oracle_setting=True,
                    fill_outside_fraction=float(1-result['fill_projection_inside_fraction'].mean()),
                    primary_region_mean_hole=float(result['s_hole'][mask].mean()) if mask.any() else None,coverage_diagnostic=coverage,
                    coverage_note='diagnostic/reference thresholds only; no new post-freeze stopping rule; missing hidden layers in O5 are intentional',
                    composition='jointly depth-sort visible+hidden per subcamera, alpha-composite, then residualT*sameCfill; averageK in linear RGB',
                    criteria=str(a.criteria),output=str(out/'prediction.npz'))
                rows.append(row);save_json(out/'metadata.json',row);save_json(a.output/'runtime.json',dict(completed=False,rows=rows));print('RENDERED',entry['id'],method,seconds,flush=True)
                del result,prediction
        del visible,all_hidden,rec_hidden,parent;torch.cuda.empty_cache()
    record=dict(mode='render',completed=True,criteria=str(a.criteria),manifest=str(a.manifest),parameters=parameters,K=samples,rows=rows,
        color_policy='shared sRGB0..1 input and inverse transfer; no extraHDR energy for3D; integrate linearRGB then sRGB',
        rec_label='可恢复表面的 GT Oracle',surface_baking=False,color_optimization=False,model_training=False)
    save_json(a.record,record);save_json(a.output/'runtime.json',record)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['prepare','render'],required=True)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--criteria',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--record',type=Path,required=True)
    p.add_argument('--base',type=Path,default=ROOT/'outputs/route3-b2only-2026-11-10-01')
    p.add_argument('--scene-root',type=Path,default=ROOT/'outputs/route3-a2b-2026-11-10-01/blender-b2')
    p.add_argument('--fill-manifest',type=Path);p.add_argument('--fill-root',type=Path);a=p.parse_args()
    cache_paths();cv2.setNumThreads(4)
    for key in ['manifest','criteria','output','record','base','scene_root']:
        setattr(a,key,getattr(a,key).resolve())
    criteria=json.loads(a.criteria.read_text());entries=json.loads(a.manifest.read_text())['items']
    if criteria.get('frozen') is not True:raise ValueError('Criteria must be frozen before any B2 method preparation/output')
    params=criteria['optics']['pixel_initialization']
    if (params['subdivisions'],params['sigma_pixels'],params['opacity'],params['eps2d'],criteria['optics']['K'])!=(5,.7,.999,.0015,16):raise ValueError('Unexpected frozen initialization; do not silently change parameters')
    if any(e.get('dataset','').lower()!='blender' or e['scene'] not in ['B2-low','B2-high'] for e in entries):raise ValueError('Authorized Blender scenes only')
    if any(Path(e['id']).name!=e['id'] for e in entries) or len({e['id'] for e in entries})!=len(entries):raise ValueError('Unique single-directory manifest IDs required')
    a.output.mkdir(parents=True,exist_ok=True)
    if a.mode=='prepare':prepare(a,entries,criteria)
    else:
        if a.fill_manifest is None or a.fill_root is None:raise ValueError('Shared six-frame LaMa fill manifest/root required')
        a.fill_manifest=a.fill_manifest.resolve();a.fill_root=a.fill_root.resolve();render(a,entries,criteria)


if __name__=='__main__':main()
