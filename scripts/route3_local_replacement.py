"""Appendix10 seventh revision: full fixed-rule3x4 matrix, existing assets only."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
import cv2
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from route3_depth_interface import SCENES as OLD_SCENES,OUT as FOUT,subject_mask,focus_row,write,save
from route3_step2_diagnostic import fit_background,numpy_stat
from route3_s2a_data import descriptor,views_for,ALL
from route3_s2a_predict import visible_cloud
from route3_s2a_candidates import local_placement
from route3_s2a_evaluate import masks as scene_masks
from route3_upper_bound_evaluate import items,score,pool
from route3_ldr_u1r import evaluation_masks,model
from route3_s2a_proxy import display,read
from route3_b2_oracle_3d import sparse_pixel_layer
from route3_sharp_s1 import raster
from src.visible_depth_interface import visible_depth_interface
from src.local_gaussian_replacement import input_observation_weights,local_suppression_masks,suppress,projection,blocker_categories_projected
from src.hidden_retrieval_rules import collect,aggregate
from src.dual_layer import depth_band
from src.layered_aperture import merge_layers
from src.analytic_aperture import sampled
from src.aperture_renderer import GaussianScene

PREFIX='route3-local-replacement-2026-11-13';OUT=ROOT/'outputs'/f'{PREFIX}-01';REC=ROOT/'03_实验记录/records'
SCENES=list(OLD_SCENES)+['B2-high'];REGIONS=['full_image','in_focus','band','recoverable'];COLS=['none','retrieval','GTraw','GTaligned'];ROWS=['R0','R1','R2']
PROFILE=json.loads((REC/'route3-dual-layer-criteria-L2-2026-11-11.json').read_text())['profile']

def guard(gpu=False):
    if shutil.disk_usage(OUT).free/2**30<60:raise RuntimeError('STOP disk below60GiB;no cleanup authorized')
    if gpu:
        assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
        subprocess.run(['nvidia-smi'],check=True)
    torch.set_num_threads(4);cv2.setNumThreads(2);torch.backends.cuda.matmul.allow_tf32=False

def desc_for(name):
    if name=='B2-high':return descriptor(name)
    s=OLD_SCENES[name]
    return dict(scene=s,entries=json.loads((s/'prepared.json').read_text())['rows'],geometry=s/'geometry',model=s/'models/SHARP',calibration=s/'models/calibration/SHARP',confidence=s/'models/calibration',parents=lambda f:s/'oracle'/f'f{f:03d}.npz',source_root=s)

def frame_dir(name,f):return OUT/name/f'f{f:03d}'
def case_dir(name,f,case,rule):return frame_dir(name,f)/rule/case['id']
def tensor(a):return torch.as_tensor(a,device='cuda')
def load(path):
    with np.load(path) as a:return {k:a[k] for k in a.files}

def reference(name,desc,case,f):
    if name!='B2-high':
        c=dict(case,frame=f);truth=display(read(desc['source_root']/'dof'/(case['id']+'.exr')));m=scene_masks(desc['source_root'],c)
    else:
        x=next(x for x in items() if x['id']==case['id']);criteria=json.loads((REC/'route3-sharp-s1d-criteria-2026-11-11.json').read_text());refs=json.loads((ROOT/'outputs/route3-ldr-u1r-2026-11-10-01/references.json').read_text())['rows'];ldr=np.load(next(x['path'] for x in refs if x['id']==case['id']))['reference_srgb'];truth,m,_=evaluation_masks(x,criteria,ldr)
    return truth,{k:m[k] for k in REGIONS}

def fit_and_map(g,own,foreground_ids,parents):
    background=g['center_visible']&~np.isin(g['object_id'],foreground_ids)
    a,b,audit,supported=local_placement(g['depth_axial_m'],own,background,radius=24,return_valid=True)
    uv=parents['uv'];x=np.clip(np.floor(uv[:,0]).astype(int),0,511);y=np.clip(np.floor(uv[:,1]).astype(int),0,511)
    inverse=a[y,x]/parents['depth']+b[y,x];valid=supported[y,x]&np.isfinite(inverse)&(inverse>1e-4)&parents['recoverable']
    mapped=np.full(len(inverse),np.nan,np.float32);mapped[valid]=1/inverse[valid]
    # Unsupported values from the legacy helper's nearest-fit fill are never
    # used. No GT-front correction, depth clamp, or raw-depth fallback.
    return a,b,supported,mapped,valid,dict(**audit,parent_count=len(uv),valid_recoverable_parents=int(valid.sum()),invalid_inverse_recoverable=int((parents['recoverable']&supported[y,x]&~(inverse>1e-4)).sum()),no_fallback=True,fit_target='renderer visible-depth,not inputfirstlayer',window_radius=24,minimum_training_background_pixels=32,slope_rule='unchanged placementv2 clip[.1,3],identityslope atvariance<=1e-5;offset fitted locally')

def prepare(names):
    guard(True)
    for name in names:
      desc=desc_for(name);views=None;mesh=np.load(desc['scene']/'scene-geometry.npz');fgids=mesh['foreground_object_ids']
      for entry in desc['entries']:
        f=entry['frame'];dest=frame_dir(name,f);manifest=dest/'prepared.json'
        if manifest.exists():continue
        guard();t=time.monotonic();d=load(entry['input_npz']);g=load(desc['geometry']/f'f{f:03d}-center.npz');cloud=visible_cloud(desc,entry);K=tensor(d['K']);old=FOUT/name/f'f{f:03d}';rows=[]
        if name!='B2-high':
            maps=load(old/'visible-depth.npz');previous=json.loads((old/'interface.json').read_text());rows=[r for r in previous['focus'] if r['method']=='visible' and r['rule'] in ['P2','P1']];globalfit=previous['background_fits']['visible']
        else:
            select=subject_mask(g,1.4)|subject_mask(g,5.);raw=visible_depth_interface(cloud,K,select);maps={k:v for k,v in raw.items() if isinstance(v,np.ndarray)};globalfit=fit_background(g['depth_axial_m'],maps['depth'],g['center_visible']&~np.isin(g['object_id'],fgids))
            for nominal in [1.4,5.]:
                subject=subject_mask(g,nominal)
                for rule in ['P2','P1']:rows.append(dict(method='visible',**focus_row(maps['depth'],subject,nominal,None,globalfit,rule,maps)))
        save(dest/'visible-depth.npz',**maps)
        maximum,mass,audit=input_observation_weights(cloud,K)
        native_alpha=float(maps['alpha'].sum());mass_sum=float(mass.sum());assert abs(mass_sum-native_alpha)/max(native_alpha,1)<2e-5,(mass_sum,native_alpha)
        save(dest/'observation.npz',max_weight=maximum.cpu().numpy(),sum_weight=mass.cpu().numpy(),unobserved=(maximum<.01).cpu().numpy())
        parents=load(desc['parents'](f));aa,bb,supported,mapped,valid,fit=fit_and_map(g,maps['depth'],fgids,parents)
        save(dest/'alignment.npz',a=aa,b=bb,local_supported=supported,parent_depth_aligned=mapped,parent_valid=valid)
        union=np.zeros((512,512),bool)
        for row in rows:
            if row['success']:union|=depth_band(maps['depth'],d['K'][0,0],row['focus_m'],.05/2.8)[0]
        if views is None:views=views_for(desc,ALL,512)
        rawpool=collect(maps['depth'],d['K'],d['c2w'] if 'c2w' in d else g['camera_c2w'],union,views,f);retrieved,ra=aggregate(rawpool,2,maps['depth']);save(dest/'retrieval.npz',**retrieved)
        write(manifest,dict(scene=name,frame=f,focus=rows,global_fit=globalfit,local_alignment=fit,observation=dict(**audit,sum_weight=mass_sum,rendered_alpha_sum=native_alpha,total_gaussians=len(cloud.means),unobserved_gaussians=int((maximum<.01).sum()),unobserved_cumulative_contribution_fraction=float(mass[maximum<.01].sum()/mass.sum())),retrieval=ra,seconds=time.monotonic()-t))
        print('X_PREPARE',name,f,'unobserved',float((maximum<.01).float().mean()),'mass',float(mass[maximum<.01].sum()/mass.sum()),'support',float(supported.mean()),flush=True)
        del cloud;torch.cuda.empty_cache()

def hidden_layers(parents,retrieved,alignment,band,K):
    uv=parents['uv'];x=np.clip(uv[:,0].astype(int),0,511);y=np.clip(uv[:,1].astype(int),0,511);raw_select=parents['recoverable']&band[y,x];aligned_select=raw_select&alignment['parent_valid'];aligned=dict(parents,depth=alignment['parent_depth_aligned'])
    rv=retrieved['uv'];rx=np.clip(rv[:,0].astype(int),0,511);ry=np.clip(rv[:,1].astype(int),0,511);rsel=band[ry,rx]
    eye=torch.eye(4,device='cuda')
    make=lambda p,sel:sparse_pixel_layer(p,sel,K,eye,PROFILE,'cuda') if sel.any() else None
    return dict(GTraw=make(parents,raw_select),GTaligned=make(aligned,aligned_select),retrieval=make(retrieved,rsel)),dict(raw_parent_count=int(raw_select.sum()),aligned_parent_count=int(aligned_select.sum()),retrieved_parent_count=int(rsel.sum()))

def support_map(hidden,K,valid=None):
    if hidden is None:return np.zeros((512,512),bool),np.zeros((512,512),np.float32)
    with torch.inference_mode():_,z,a=raster(hidden,K)
    support=a.cpu().numpy()>0
    if valid is not None:support&=valid
    return support,z.cpu().numpy()

def image_zero(cloud,K,fill):
    with torch.inference_mode():rgb,z,a=raster(cloud,K)
    return dict(rgb_linear=(rgb+(1-a[...,None])*fill).cpu().numpy(),alpha=a.cpu().numpy(),depth=z.cpu().numpy())

def render(names):
    guard(True)
    for name in names:
      desc=desc_for(name)
      for entry in desc['entries']:
        f=entry['frame'];dest=frame_dir(name,f);prep=json.loads((dest/'prepared.json').read_text());d=load(entry['input_npz']);own=load(dest/'visible-depth.npz')['depth'];obs=load(dest/'observation.npz');alignment=load(dest/'alignment.npz');parents=load(desc['parents'](f));retrieved=load(dest/'retrieval.npz');cloud=visible_cloud(desc,entry);K=tensor(d['K']);fill=tensor(np.load(entry['fill'])['fill_rgb_linear']);bg=tensor(d['background_depth']);E=torch.eye(4,device='cuda')
        for rule in ['P2','P1']:
          for case in entry['cases']:
            folder=case_dir(name,f,case,rule);manifest=folder/'render.json'
            if manifest.exists():continue
            guard();focusrow=next(r for r in prep['focus'] if r['rule']==rule and r['nominal_focus_m']==case['nominal_focus_m'])
            if not focusrow['success']:write(folder/'unavailable.json',focusrow);continue
            focus=focusrow['focus_m'];aperture=.05/(2*case['fstop']);band,edge,coc=depth_band(own,d['K'][0,0],focus,aperture);hidden,counts=hidden_layers(parents,retrieved,alignment,band,K)
            oracle_support,oracle_depth=support_map(hidden['GTaligned'],K,band&alignment['local_supported']);method_support,method_depth=support_map(hidden['retrieval'],K,band)
            masks={};suppression=[]
            for branch,support,depth in [('oracle',oracle_support,oracle_depth),('method',method_support,method_depth)]:
                r1,r2=local_suppression_masks(cloud,K,tensor(support),tensor(own),tensor(depth),tensor(obs['unobserved']))
                for row,mask in [('R1',r1),('R2',r2)]:
                    mn=mask.cpu().numpy();masks[branch+'_'+row]=mn
                    if row=='R2':assert not np.any(mn&~obs['unobserved'])
                    removed=suppress(cloud,~mask)
                    with torch.inference_mode():_,_,covered=raster(removed,K)
                    foot=covered.cpu().numpy()>0;ids=cloud.source_frames.cpu().numpy();mass=obs['sum_weight'];den=max(float(mass.sum()),1e-30)
                    suppression.append(dict(branch=branch,row=row,count=int(mn.sum()),layer0_count=int((mn&(ids==0)).sum()),layer1_count=int((mn&(ids==1)).sum()),input_contribution_fraction=float(mass[mn].sum()/den),layer0_input_contribution_fraction=float(mass[mn&(ids==0)].sum()/den),layer1_input_contribution_fraction=float(mass[mn&(ids==1)].sum()/den),max_input_weight_of_removed=float(obs['max_weight'][mn].max()) if mn.any() else 0.,footprint_pixels=int(foot.sum()),footprint_fraction=float(foot.mean()),footprint_outside_support_pixels=int((foot&~support).sum()),support_pixels=int(support.sum())))
            save(folder/'support-masks.npz',band=band,oracle_support=oracle_support,oracle_hidden_depth=oracle_depth,method_support=method_support,method_hidden_depth=method_depth,**masks)
            visible={('R0','oracle'):cloud}
            for branch in ['oracle','method']:
                for row in ['R1','R2']:visible[(row,branch)]=suppress(cloud,tensor(masks[branch+'_'+row]))
            bases={};zeros={};records=[]
            for row,branch in [('R0','oracle'),('R1','oracle'),('R2','oracle'),('R1','method'),('R2','method')]:
                key=(row,branch);t=time.monotonic();torch.cuda.reset_peak_memory_stats()
                with torch.inference_mode():p=sampled(visible[key],K,E,512,512,focus,aperture,64,fill,bg,far_plane=1e5,return_fill_contribution=True)
                bases[key]={k:v.cpu().numpy() for k,v in p.items()};zeros[key]=image_zero(visible[key],K,fill)
                if branch=='method':
                    # Matched suppression-only method control, auxiliary to3x4.
                    save(folder/(row+'-none-method-support.npz'),prediction_srgb=display(bases[key]['rgb_linear']),zero_srgb=display(zeros[key]['rgb_linear']))
            for row in ROWS:
              for column in COLS:
                branch='method' if column=='retrieval' and row!='R0' else 'oracle';key=(row,branch);base=bases[key];zero=zeros[key];hid=hidden.get(column);t=time.monotonic();torch.cuda.reset_peak_memory_stats()
                if hid is None:values=base;zero_values=zero
                else:
                    merged=merge_layers(visible[key],hid)
                    with torch.inference_mode():p=sampled(merged,K,E,512,512,focus,aperture,64,fill,bg,far_plane=1e5,return_fill_contribution=True)
                    values={k:np.where(band[...,None] if v.ndim==3 else band,v.cpu().numpy(),base[k]) for k,v in p.items()};z=image_zero(merged,K,fill);zero_values={k:np.where(band[...,None] if v.ndim==3 else band,v,zero[k]) for k,v in z.items()}
                path=folder/(row+'-'+column+'.npz');save(path,prediction_srgb=display(values['rgb_linear']),**values,zero_srgb=display(zero_values['rgb_linear']))
                old_error=None
                if row=='R0' and column in ['none','GTraw'] and name!='B2-high' and case['fstop']==1.4:
                    oldpath=FOUT/name/f'f{f:03d}'/(case['id']+'-'+rule+'-'+('A_color' if column=='none' else 'GTrec')+'.npz');old_error=float(abs(values['rgb_linear']-np.load(oldpath)['rgb_linear']).max())
                records.append(dict(row=row,column=column,branch=branch,path=str(path),seconds=time.monotonic()-t,peak_MiB=torch.cuda.max_memory_allocated()/2**20,old_result_max_linear_error=old_error))
            # CoC diagnostics use the same fixed camera focus as every matrix
            # cell, not a post-suppression autofocus adjustment.
            g=np.load(desc['geometry']/f'f{f:03d}-center.npz');_,regions=reference(name,desc,case,f);inv=alignment['a']/np.maximum(g['depth_axial_m'],1e-6)+alignment['b'];good=alignment['local_supported']&g['center_visible']&np.isfinite(inv)&(inv>1e-4);gt_signed=d['K'][0,0]*aperture*(1/np.maximum(g['depth_axial_m'],1e-6)-1/case['nominal_focus_m']);mapped_signed=d['K'][0,0]*aperture*(inv-1/focus);raw_signed=d['K'][0,0]*aperture*(1/np.maximum(g['depth_axial_m'],1e-6)-1/focus);mm=good&regions['in_focus'];CoC=dict(valid_infocus_pixels=int(mm.sum()),GT_radius_px=numpy_stat(abs(gt_signed[mm])),aligned_radius_px=numpy_stat(abs(mapped_signed[mm])),aligned_minus_GT_radius_px=numpy_stat(abs(mapped_signed[mm])-abs(gt_signed[mm])),alignment_change_radius_px=numpy_stat(abs(mapped_signed[mm])-abs(raw_signed[mm])))
            write(manifest,dict(scene=name,frame=f,id=case['id'],nominal_focus_m=case['nominal_focus_m'],fstop=case['fstop'],rule=rule,focus_m=focus,rows=records,suppression=suppression,parent_counts=counts,oracle_support_recoverable_pixels=int((oracle_support&regions['recoverable']).sum()),recoverable_pixels=int(regions['recoverable'].sum()),oracle_support_recoverable_fraction=float(oracle_support[regions['recoverable']].mean()) if regions['recoverable'].any() else None,CoC=CoC,optics_fixed_from_R0=True,none_column_uses_oracle_suppression=True,method_retrieval_uses_own_support_no_GT=True))
            print('X_RENDER',name,f,rule,case['id'],'suppression',[(r['branch'],r['row'],r['count'],round(r['input_contribution_fraction'],5)) for r in suppression],flush=True)
            del hidden,visible,bases,zeros;torch.cuda.empty_cache()
        del cloud;torch.cuda.empty_cache()

def evaluate(names):
    guard();net=model();allrows=[];auxiliary=[]
    for name in names:
      desc=desc_for(name)
      for entry in desc['entries']:
       for rule in ['P2','P1']:
        for case in entry['cases']:
          folder=case_dir(name,entry['frame'],case,rule);manifest=folder/'render.json'
          if not manifest.exists():continue
          truth,regions=reference(name,desc,case,entry['frame']);support=np.load(folder/'support-masks.npz')['oracle_support'];regions=dict(regions,recoverable_supported=regions['recoverable']&support);target=np.load(entry['input_npz'])['rgb_srgb'];common=dict(scene=name,frame=entry['frame'],id=case['id'],nominal_focus_m=case['nominal_focus_m'],fstop=case['fstop'],rule=rule)
          for row in ROWS:
            for column in COLS:
              path=folder/(row+'-'+column+'.npz');cache=folder/(row+'-'+column+'-scores.json')
              if cache.exists():result=json.loads(cache.read_text())
              else:
                p=np.load(path);result=dict(dof=score(p['prediction_srgb'],truth,regions,net),zero={})
                for region,mask in dict(full_image=regions['full_image'],in_focus=regions['in_focus']).items():
                    error=((p['zero_srgb'][mask].astype(np.float64)-target[mask])**2).sum();n=int(mask.sum());result['zero'][region]=dict(pixels=n,sse=float(error),psnr=float(-10*np.log10(max(error/max(3*n,1),1e-30))))
                write(cache,result)
              for region,m in result['dof'].items():allrows.append(dict(**common,row=row,column=column,region=region,**m,zero_full_PSNR=result['zero']['full_image']['psnr'],zero_infocus_PSNR=result['zero']['in_focus']['psnr'],zero_full_sse=result['zero']['full_image']['sse'],zero_full_pixels=result['zero']['full_image']['pixels'],zero_infocus_sse=result['zero']['in_focus']['sse'],zero_infocus_pixels=result['zero']['in_focus']['pixels']))
          for row in ['R1','R2']:
              path=folder/(row+'-none-method-support.npz');cache=folder/(row+'-none-method-support-scores.json')
              if cache.exists():result=json.loads(cache.read_text())
              else:
                  p=np.load(path);result=score(p['prediction_srgb'],truth,regions,net);write(cache,result)
              for region,m in result.items():auxiliary.append(dict(**common,row=row,column='none-method-support',region=region,**m))
          print('X_SCORE',name,entry['frame'],rule,case['id'],flush=True)
    write(OUT/'scores.json',dict(rows=allrows,reference='Cycles;frozen GT masks;AlexNetLPIPS0.1,64pxfixedpatches'))
    write(OUT/'method-control-scores.json',dict(rows=auxiliary,description='matched suppression-only controls for input-only retrieval support;outside main3x4'))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','render','evaluate']);p.add_argument('--scene',choices=SCENES);a=p.parse_args();OUT.mkdir(exist_ok=True);names=[a.scene] if a.scene else SCENES;globals()[a.mode](names)

if __name__=='__main__':main()
