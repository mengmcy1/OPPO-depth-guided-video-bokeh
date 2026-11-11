"""Authorized E1-only S1c application; E2 calibration remains blocked."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import cv2
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from route3_sharp_s1c import OUT,PREFIX,CRITERIA,write,S1,UPPER,BASE,RECORDS,FRAMES
from route3_sharp_s1 import scene,raster
from route3_sharp_s1b import OUT as S1B,fitted_scene
from route3_upper_bound_evaluate import items,display
from route3_target_focus import target_mask
from src.dual_layer import depth_band
from src.dual_layer_retrieval import retrieve
from src.aperture_renderer import GaussianScene
from src.sharp_anchor import anchor


def prepare(c):
    root=OUT/'prepared';root.mkdir(exist_ok=False);rows=[];fills=[];views={name:[] for name in ['native','cal_E1','GT']}
    for f in FRAMES:
        rgb=np.load(S1/'depth'/f'f{f:03d}.npz');g=np.load(BASE/'B2-high/geometry'/f'f{f:03d}-center.npz');z=np.load(OUT/'calibration/E1/SHARP'/f'f{f:03d}.npz')['depth']
        for name,depth in [('native',rgb['depth']),('cal_E1',z),('GT',g['depth_axial_m'])]:views[name].append(dict(frame=f,depth=depth,K=rgb['K'],c2w=g['camera_c2w'],rgb_linear=rgb['rgb_linear']))
    prior=json.loads((S1/'prepared.json').read_text())['rows']
    for name in ['native','cal_E1']:
        for f in [1,30,60]:
            old=next(x for x in prior if x['scale_label']=='native' and x['frame']==f);rgb=np.load(old['input_npz']);target=next(v for v in views[name] if v['frame']==f);z=target['depth'];K=target['K'];union=np.zeros((512,512),bool);maxradius=0.;cases=[]
            for case in [x for x in items() if x['frame_id']==f]:
                focus=float(np.median(z[target_mask(case)]));band,edge,radius=depth_band(z,K[0,0],focus,case['aperture_radius_m']);union|=band;maxradius=max(maxradius,float(radius.max()));cases.append(dict(id=case['id'],focus_m=focus,fstop=case['fstop'],nominal_focus_m=case['focus_distance_m']))
            mask=cv2.dilate(union.astype(np.uint8),np.ones((5,5),np.uint8),iterations=2)>0;extension=int(np.ceil(2*maxradius))+4;bg=cv2.dilate(z,np.ones((2*extension+1,2*extension+1),np.uint8))
            inv=1/np.maximum(z,1e-6);mx=cv2.dilate(inv,np.ones((3,3),np.uint8));mn=cv2.erode(inv,np.ones((3,3),np.uint8));edge=(mx-mn)>.03*mx;excluded=cv2.dilate(edge.astype(np.uint8),np.ones((7,7),np.uint8))>0
            folder=root/f'{name}-f{f:03d}';folder.mkdir();path=folder/'inputs.npz';np.savez_compressed(path,depth=z,K=K,rgb_srgb=rgb['rgb_srgb'],rgb_linear=rgb['rgb_linear'],fill_mask=mask,background_depth=bg,edge_loss_valid=~excluded)
            if name=='native':
                fill=str(S1/'fill'/old['id']/'fill.npz');assert np.array_equal(mask,rgb['fill_mask']) and np.array_equal(bg,rgb['background_depth'])
            else:fill=str(OUT/'fill'/f'{name}-f{f:03d}'/'fill.npz');fills.append(dict(id=f'{name}-f{f:03d}',input_npz=str(path),uses_GT_segmentation=False,uses_true_hidden_depth=False))
            parents={};audits={}
            for level,source in [('L3_GTgeometry','GT'),('L3_cal_E1','cal_E1'),('L3_native_E1','native')]:
                target_view=next(v for v in views[source] if v['frame']==f);tz=target_view['depth']
                if source=='GT':tz=np.where(tz>0,tz,10000.)
                p,a=retrieve(tz,K,target_view['c2w'],union,views[source],f);dest=folder/(level+'.npz');np.savez_compressed(dest,**p);parents[level]=str(dest);audits[level]=a
            rows.append(dict(id=f'{name}-f{f:03d}',geometry=name,frame=f,input_npz=str(path),fill=fill,cases=cases,parents=parents,audits=audits,edge_loss_valid_pixels=int((~excluded).sum())))
            print('S1C_PREPARED',name,f,flush=True)
    write(OUT/'prepared.json',dict(completed=True,rows=rows,E1_GT_poses=True));write(OUT/'fill-inputs.json',dict(items=fills));write(RECORDS/(PREFIX+'-preparation.json'),dict(completed=True,rows=rows,E1_GT_poses=True,E2_incomplete=True))


def base_scene(entry):
    cloud=scene(entry['frame'],'full')
    if entry['geometry']=='native':return cloud
    fitted=next(x for x in json.loads((OUT/'calibration.json').read_text())['rows'] if x['variant']=='E1' and x['frame']==entry['frame'])['models']['SHARP'];z=cloud.means[:,2];inverse=fitted['a']/z.clamp_min(1e-6)+fitted['b'];mapped=1/inverse.clamp_min(1e-4);ratio=torch.where(z>1e-6,mapped/z.clamp_min(1e-6),torch.ones_like(z))
    folder=OUT/'geometry-mapping';folder.mkdir(exist_ok=True);path=folder/(entry['id']+'.json')
    if not path.exists():
        write(path,dict(frame=entry['frame'],Gaussians=len(z),floored_inverse_depth_Gaussians=int(((inverse<1e-4)&(z>0)).sum()),nonpositive_depth_Gaussians_kept=int((z<=1e-6).sum()),ratio_p01_p50_p99=torch.quantile(ratio,torch.tensor([.01,.5,.99],device=z.device)).cpu().tolist(),projection_preserved_by_ray_scaling=True))
    return GaussianScene(cloud.means*ratio[:,None],cloud.quats,cloud.scales*ratio[:,None],cloud.opacities,cloud.colors,cloud.source_frames)


def fit(c):
    rows=[]
    for entry in json.loads((OUT/'prepared.json').read_text())['rows']:
        f=entry['frame'];d=np.load(entry['input_npz']);K=torch.tensor(d['K'],device='cuda');fill=torch.tensor(np.load(entry['fill'])['fill_rgb_linear'],device='cuda');target=torch.tensor(d['rgb_srgb'],device='cuda');cloud=base_scene(entry)
        modes=['A_color_edge'] if entry['geometry']=='native' else ['A_color','A_color_opacity','A_color_edge']
        for method in modes:
            folder=OUT/'anchor'/entry['id']/method;folder.mkdir(parents=True,exist_ok=False);torch.manual_seed(2026111121+f);mask=torch.tensor(d['edge_loss_valid'],device='cuda') if method=='A_color_edge' else None
            fitted,result=anchor(cloud,K,fill,target,method=='A_color_opacity',lambda x:write(folder/'monitor.json',x),loss_mask=mask)
            path=folder/'attributes-final.pt';torch.save(dict(colors=fitted.colors.detach().cpu(),opacities=fitted.opacities.detach().cpu(),completed_updates=result['completed_updates']),path)
            result.update(frame=f,geometry=entry['geometry'],method=method,attributes=str(path));write(folder/'result.json',result);write(RECORDS/(PREFIX+f'-anchor-{entry["id"]}-{method}.json'),result)
            if not result['completed']:raise RuntimeError('S1C_STOP '+str(result['stop_reason']))
            rows.append(result);print('S1C_ANCHOR',entry['id'],method,result['rows'][-1]['input_PSNR_dB'],flush=True)
        del cloud,fitted;torch.cuda.empty_cache()
    write(OUT/'anchor.json',dict(completed=True,rows=rows,E1_GT_poses=True))


def anchored(entry,method):
    if entry['geometry']=='native' and method in ['A_color','A_color_opacity']:return fitted_scene(entry['frame'],method)
    base=base_scene(entry);p=torch.load(OUT/'anchor'/entry['id']/method/'attributes-final.pt',map_location='cuda',weights_only=True);assert p['completed_updates']==1000
    return GaussianScene(base.means,base.quats,base.scales,p['opacities'],p['colors'],base.source_frames)


def bokehme(c):
    from src.baselines.bokehme import setup,predict
    net,settings,models,_=setup(ROOT.parent/'BokehMe',ROOT/'models/BokehMe',dict(gamma=2.2,highlight=False,defocus_scale=10.));folder=OUT/'O3';folder.mkdir(exist_ok=False);rows=[]
    for entry in json.loads((OUT/'prepared.json').read_text())['rows']:
        if entry['geometry']=='native':continue
        d=np.load(entry['input_npz'])
        for source in ['SHARP','DA3']:
            z=np.load(OUT/'calibration/E1'/source/f'f{entry["frame"]:03d}.npz')['depth']
            for case in entry['cases']:
                original=next(x for x in items() if x['id']==case['id']);focus=float(np.median(z[target_mask(original)]));control=(d['K'][0,0]*original['aperture_radius_m']*(1/z-1/focus)/10).astype(np.float32)
                p,_=predict(net,settings,models,np.ascontiguousarray(d['rgb_srgb']),np.ascontiguousarray(control));path=folder/(case['id']+'-'+source+'.npz');np.savez_compressed(path,prediction_srgb=p.clip(0,1));rows.append(dict(geometry='cal_E1',id=case['id'],method='O3_'+source,path=str(path)))
    write(OUT/'O3-predictions.json',dict(completed=True,rows=rows))


def render(c):
    from src.analytic_aperture import sampled
    from src.layered_aperture import merge_layers
    from route3_b2_oracle_3d import sparse_pixel_layer
    root=OUT/'render';root.mkdir(exist_ok=False);zeros=OUT/'zero';zeros.mkdir(exist_ok=False);rows=[];zero_rows=[];diagnostics=[];timing=[]
    profile=json.loads((RECORDS/'route3-dual-layer-criteria-L2-2026-11-11.json').read_text())['profile'];cache=json.loads((S1B/'native-predictions.json').read_text())['rows']
    for entry in json.loads((OUT/'prepared.json').read_text())['rows']:
        d=np.load(entry['input_npz']);K=torch.tensor(d['K'],device='cuda');E=torch.eye(4,device='cuda');fill=torch.tensor(np.load(entry['fill'])['fill_rgb_linear'],device='cuda');bg=torch.tensor(d['background_depth'],device='cuda')
        for method in ['A_color','A_color_opacity','A_color_edge']:
            cloud=anchored(entry,method)
            with torch.inference_mode():rgb,_,alpha=raster(cloud,K)
            path=zeros/(entry['id']+'-'+method+'.npz');np.savez_compressed(path,prediction_srgb=display((rgb+(1-alpha[...,None])*fill).cpu().numpy()),alpha=alpha.cpu().numpy());zero_rows.append(dict(geometry=entry['geometry'],frame=entry['frame'],method=method,path=str(path)))
            for case in entry['cases']:
                path=root/(entry['id']+'-'+case['id']+'-'+method+'.npz');band=depth_band(d['depth'],d['K'][0,0],case['focus_m'],.05/(2*case['fstop']))[0]
                if entry['geometry']=='native' and method!='A_color_edge':
                    cached=next(x for x in cache if x['id']==case['id'] and x['method']==method);path=Path(cached['path'])
                else:
                    with torch.inference_mode():p=sampled(cloud,K,E,512,512,case['focus_m'],.05/(2*case['fstop']),64,fill,bg,far_plane=1e5,return_fill_contribution=True)
                    np.savez_compressed(path,prediction_srgb=display(p['rgb_linear'].cpu().numpy()),**{k:v.cpu().numpy() for k,v in p.items()})
                rows.append(dict(geometry=entry['geometry'],id=case['id'],method=method,path=str(path)));base_result=np.load(path)
                if method!='A_color':continue
                for level in ['O6_GT_rec','L3_GTgeometry','L3_cal_E1','L3_native_E1']:
                    source=BASE/'methods-3d/prepared'/f'B2-high-f{entry["frame"]:03d}'/'parents.npz' if level=='O6_GT_rec' else Path(entry['parents'][level]);parent=np.load(source);uv=parent['uv'];x=np.clip(uv[:,0].astype(int),0,511);y=np.clip(uv[:,1].astype(int),0,511);selected=band[y,x]
                    if level=='O6_GT_rec':selected&=parent['recoverable']
                    hidden=sparse_pixel_layer(parent,selected,K,E,profile,'cuda');torch.cuda.synchronize();start=time.monotonic()
                    with torch.inference_mode():p=sampled(merge_layers(cloud,hidden),K,E,512,512,case['focus_m'],.05/(2*case['fstop']),64,fill,bg,far_plane=1e5,return_fill_contribution=True)
                    torch.cuda.synchronize();seconds=time.monotonic()-start
                    values={k:np.where(band[...,None] if v.ndim==3 else band,v.cpu().numpy(),base_result[k]) for k,v in p.items()};path=root/(entry['id']+'-'+case['id']+'-'+level+'.npz');np.savez_compressed(path,prediction_srgb=display(values['rgb_linear']),**values)
                    rows.append(dict(geometry=entry['geometry'],id=case['id'],method='A_color_'+level,path=str(path)));timing.append(dict(geometry=entry['geometry'],id=case['id'],level=level,seconds=seconds))
                    # Separate evaluator-only GT tag pass; never filters output.
                    g=np.load(BASE/'B2-high/geometry'/f'f{entry["frame"]:03d}-center.npz');known=g['center_visible'][y,x];gt=g['depth_axial_m'][y,x];wrong=known&(parent['depth']<gt-np.maximum(.001,.005*gt));tagparent={k:parent[k] for k in parent.files};tagparent['colors_linear']=np.repeat(wrong[:,None].astype(np.float32),3,axis=1);taghidden=sparse_pixel_layer(tagparent,selected,K,E,profile,'cuda');tagvisible=GaussianScene(cloud.means,cloud.quats,cloud.scales,cloud.opacities,torch.zeros_like(cloud.colors),cloud.source_frames)
                    with torch.inference_mode():tag=sampled(merge_layers(tagvisible,taghidden),K,E,512,512,case['focus_m'],.05/(2*case['fstop']),64,torch.zeros_like(fill),bg,far_plane=1e5)
                    heat=np.where(band,tag['rgb_linear'][...,0].cpu().numpy(),0);diagnostics.append(dict(geometry=entry['geometry'],id=case['id'],level=level,parents=int(selected.sum()),known_GT_supported_parents=int((known&selected).sum()),wrong_parents=int((wrong&selected).sum()),wrong_fraction=float((wrong&selected).sum()/max((known&selected).sum(),1)),tag_mean_contribution_full=float(heat.mean()),tag_mean_contribution_band=float(heat[band].mean()),GT_used_only_after_method_prediction=True))
                    del hidden,taghidden,tagvisible,p,tag
                print('S1C_RENDER',entry['id'],case['id'],method,flush=True)
            del cloud;torch.cuda.empty_cache()
    rows+=json.loads((OUT/'O3-predictions.json').read_text())['rows'];write(OUT/'predictions.json',dict(completed=True,rows=rows,zeros=zero_rows));write(RECORDS/(PREFIX+'-render.json'),dict(completed=True,rows=timing,hidden_diagnostics=diagnostics,K=64,E1_GT_poses=True,E2='blocked'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['prepare','fit','bokehme','render'],required=True);a=p.parse_args();c=json.loads(CRITERIA.read_text());assert c['frozen'] and (OUT/'E1-continuation-authorized.json').exists();torch.set_num_threads(8);cv2.setNumThreads(1)
    if a.mode!='prepare':
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise ValueError('Explicit GPU0')
        subprocess.run(['nvidia-smi'],check=True)
    globals()[a.mode](c)


if __name__=='__main__':main()
