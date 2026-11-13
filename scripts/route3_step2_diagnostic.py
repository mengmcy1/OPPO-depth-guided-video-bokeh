"""Appendix8 D: existing five scenes, fixed appearance, explicit geometry edits."""
import argparse,json,os,sys,time,subprocess,shutil
from pathlib import Path
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
import cv2,numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from route3_s2a_data import descriptor
from route3_s2a_predict import visible_cloud,baseline_path
from route3_s2a_evaluate import masks as scene_masks
from route3_s2a_proxy import display,read
from route3_upper_bound_evaluate import items,score,pool
from route3_ldr_u1r import model,evaluation_masks
from src.aperture_renderer import GaussianScene
from src.analytic_aperture import sampled
from src.layered_aperture import merge_layers
from src.dual_layer import depth_band
from src.prefix_transmittance import prefix_transmittance
from route3_sharp_s1 import raster
from route3_b2_oracle_3d import sparse_pixel_layer
CFG=ROOT/'configs/route3-step2-diagnostic-2026-11-12.json';C=json.loads(CFG.read_text());OUT=Path(C['output']);REC=ROOT/'03_实验记录/records';PREFIX='route3-step2-diagnostic-2026-11-12'
REGIONS=['full_image','in_focus','band','recoverable']
def write(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def save(p,**d):
    if p.exists():raise FileExistsError(p)
    p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,**d)
def numpy_stat(v):
    v=np.asarray(v);v=v[np.isfinite(v)]
    return dict(n=int(len(v)),mean=float(v.mean()),median=float(np.median(v)),quantiles=np.quantile(v,[.01,.1,.5,.9,.99]).tolist(),fraction_gt_half=float((v>.5).mean())) if len(v) else dict(n=0,mean=None,median=None,quantiles=None,fraction_gt_half=None)

def original(name,c):
    if name.startswith('scene-'):return {m:str(descriptor(name)['source_root']/'predictions'/(c['id']+'-'+m+'.npz')) for m in ['A_color','GTrec']}
    p=json.loads((REC/'route3-network-pilot-2026-11-12-baseline-Cycles.json').read_text());return {m:p['predictions'][name][c['id']][m] for m in ['A_color','GTrec']}

def references(name,c,desc):
    if name=='B2-high':
        x=next(x for x in items() if x['id']==c['id']);ldrrow=next(x for x in json.loads((ROOT/'outputs/route3-ldr-u1r-2026-11-10-01/references.json').read_text())['rows'] if x['id']==c['id']);ldr=np.load(ldrrow['path'])['reference_srgb'];crit=json.loads((REC/'route3-sharp-s1d-criteria-2026-11-11.json').read_text());truth,regions,_=evaluation_masks(x,crit,ldr)
    else:
        truth=display(read(desc['source_root']/'dof'/(c['id']+'.exr')));regions=scene_masks(desc['source_root'],c);path=desc['source_root']/'LDR-reference'/(c['id']+'-main.npz');ldr=np.load(path)['reference_srgb'] if path.exists() else None
    return dict(Cycles=truth,**({'LDR':ldr} if ldr is not None else {})),{k:regions[k] for k in REGIONS}

def fit_background(gt,est,background):
    y,x=np.mgrid[:512,:512];good=background&np.isfinite(gt)&(gt>0)&np.isfinite(est)&(est>0);held=good&((x//8+y//8)%5==0);train=good&~held
    if train.sum()<1000 or held.sum()<100:return dict(success=False,reason='insufficient background samples',train=int(train.sum()),heldout=int(held.sum()))
    xx=1/gt[train].astype(np.float64);yy=1/est[train].astype(np.float64);X=np.c_[xx,np.ones(len(xx))]
    if np.ptp(np.quantile(xx,[.05,.95]))<.005:return dict(success=False,reason='inverse depth span below.005',train=int(train.sum()))
    weights=np.ones(len(xx));ab=np.linalg.lstsq(X,yy,rcond=None)[0]
    for _ in range(12):
        err=X@ab-yy;delta=max(.002,1.345*1.4826*np.median(abs(err-np.median(err))));weights=np.minimum(1.,delta/np.maximum(abs(err),1e-12));ab=np.linalg.lstsq(X*np.sqrt(weights[:,None]),yy*np.sqrt(weights),rcond=None)[0]
    residual=ab[0]/gt[held]+ab[1]-1/est[held];pred=ab[0]/np.array([1.4,5.])+ab[1]
    success=bool(ab[0]>0 and np.isfinite(ab).all() and (pred>0).all())
    return dict(success=success,reason=None if success else 'nonpositive affine slope or mapped focus',a=float(ab[0]),b=float(ab[1]),train=int(train.sum()),heldout=int(held.sum()),heldout_inverse_RMSE=float(np.sqrt(np.mean(residual**2))),heldout_inverse_relative_MAE=float(np.mean(abs(residual)*est[held])),fitted_from_GT_visible_background_only=True)

def replace_depth(cloud,K,gt,fg,fit,mode,estimated_visible_depth=None):
    uvz=cloud.means@K.T;uv=uvz[:,:2]/uvz[:,2:].clamp_min(1e-8);xy=uv.floor().long();inside=(cloud.means[:,2]>1e-5)&(xy[:,0]>=0)&(xy[:,0]<512)&(xy[:,1]>=0)&(xy[:,1]<512);x=xy[:,0].clamp(0,511);y=xy[:,1].clamp(0,511);target=torch.tensor(gt,device='cuda')[y,x];good=inside&(target>0)
    if mode=='foreground':good&=torch.tensor(fg,device='cuda')[y,x];target=(fit['a']/target.clamp_min(1e-6)+fit['b']).clamp_min(1e-8).reciprocal()
    denominator=cloud.means[:,2] if estimated_visible_depth is None else torch.tensor(estimated_visible_depth,device='cuda')[y,x]
    ratio=torch.where(good,target/denominator.clamp_min(1e-6),torch.ones_like(target));result=GaussianScene(cloud.means*ratio[:,None],cloud.quats,cloud.scales*ratio[:,None],cloud.opacities,cloud.colors,cloud.source_frames)
    return result,dict(changed_gaussians=int(good.sum()),total=len(good),projection_preserved=True,quats_opacity_colors_unchanged=True,relative_layer_structure_preserved=estimated_visible_depth is not None,both_layers_collapsed=estimated_visible_depth is None,unknown_or_outside_unchanged=True)

def transmittance(cloud,K,geometry,fg_ids,focus):
    a={k:v for k,v in geometry.items()};hidden=a['hidden'].astype(bool)&a['recoverable'].astype(bool)&~a['residual_pixels'][None];pixel=hidden.any(0);z=a['depth_axial_m'];masked=np.where(hidden,z,np.nan)
    import warnings
    with warnings.catch_warnings():warnings.simplefilter('ignore',RuntimeWarning);plane=np.nanmedian(masked,axis=0)
    plane=np.nan_to_num(plane,nan=0.);allhidden=[];allcontrol=[];maps=[]
    for k,xy in enumerate(a['aperture_xy']):
        view=torch.eye(4,device='cuda');view[:2,3]-=torch.tensor(xy,device='cuda');intr=K.clone();intr[:2,2]+=K.diag()[:2]*torch.tensor(xy,device='cuda')/focus
        control=np.isin(a['object_id'][k],fg_ids)&~a['hidden'][k]&pixel;limits=np.where(hidden[k],z[k],plane).astype(np.float32);T=prefix_transmittance(cloud,intr,view,torch.tensor(limits,device='cuda')).cpu().numpy();allhidden.append(T[hidden[k]]);allcontrol.append(T[control]);maps.append(T)
    return dict(hidden=numpy_stat(np.concatenate(allhidden)),foreground_blocked_control=numpy_stat(np.concatenate(allcontrol)),control_depth_definition='median hidden-surface depth from other K16 rays at same pixel;not the foreground hit depth',sample_count=16,GT_labels_nominal_focus_and_method_actual_focus_reported=True),np.stack(maps)

def render():
    subprocess.run(['nvidia-smi'],check=True);torch.set_num_threads(6);profile=json.loads((REC/'route3-dual-layer-criteria-L2-2026-11-11.json').read_text())['profile'];rows=[];fits=[];zeros=[];stats=[]
    relative=C.get('relative_geometry',False);tag='render-relative' if relative else 'render'
    for name in C['scenes']:
      desc=descriptor(name);mesh=np.load(desc['scene']/'scene-geometry.npz');fg_ids=mesh['foreground_object_ids']
      for entry in desc['entries']:
        f=entry['frame'];folder=OUT/tag/name/f'f{f:03d}';folder.mkdir(parents=True,exist_ok=True);manifest=folder/'manifest.json'
        if manifest.exists():m=json.loads(manifest.read_text());rows+=m['rows'];fits.append(m['fit']);zeros+=m['zeros'];stats+=m['transmittance'];continue
        if shutil.disk_usage(OUT).free/2**30<60:raise RuntimeError('DISK BELOW60GiB')
        d=np.load(entry['input_npz']);g=np.load(desc['geometry']/f'f{f:03d}-center.npz');gt=g['depth_axial_m'];fg=np.isin(g['object_id'],fg_ids);fit=fit_background(gt,d['depth'],g['center_visible']&~fg);fit.update(scene=name,frame=f);fits.append(fit)
        cloud=visible_cloud(desc,entry);K=torch.tensor(d['K'],device='cuda');E=torch.eye(4,device='cuda');fill=torch.tensor(np.load(entry['fill'])['fill_rgb_linear'],device='cuda');bg=torch.tensor(d['background_depth'],device='cuda');parents=np.load(desc['parents'](f));local=[];zz=[];tt=[]
        with torch.no_grad():rgb,depth,alpha=raster(cloud,K)
        own=np.where(alpha.cpu().numpy()>.1,depth.cpu().numpy(),d['depth'])
        zpath=folder/'zero-original.npz';save(zpath,prediction_srgb=display((rgb+(1-alpha[...,None])*fill).cpu().numpy()),alpha=alpha.cpu().numpy());zz.append(dict(scene=name,frame=f,geometry='original',path=str(zpath)))
        for geometry in ['focus-only','foreground','full-GT']:
            if geometry!='full-GT' and not fit['success']:continue
            new=cloud;change={}
            if geometry!='focus-only':new,change=replace_depth(cloud,K,gt,fg,fit,'foreground' if geometry=='foreground' else 'full',own if relative else None)
            if geometry!='focus-only':
                with torch.no_grad():rgb,depth,alpha=raster(new,K)
                p=folder/('zero-'+geometry+'.npz');save(p,prediction_srgb=display((rgb+(1-alpha[...,None])*fill).cpu().numpy()),alpha=alpha.cpu().numpy());zz.append(dict(scene=name,frame=f,geometry=geometry,path=str(p),**change))
            for case in entry['cases']:
                if case['fstop']!=1.4:continue
                nominal=case['nominal_focus_m'];focus=nominal if geometry=='full-GT' else 1/(fit['a']/nominal+fit['b']);band=depth_band(d['depth'],d['K'][0,0],case['focus_m'],.05/2.8)[0]
                uv=parents['uv'];sel=parents['recoverable']&band[np.clip(uv[:,1].astype(int),0,511),np.clip(uv[:,0].astype(int),0,511)];hid=sparse_pixel_layer(parents,sel,K,E,profile,'cuda')
                for method,hidden in [('A_color',None),('GTrec',hid)]:
                    start=time.monotonic()
                    with torch.no_grad():p=sampled(merge_layers(new,hidden),K,E,512,512,focus,.05/2.8,64,fill,bg,far_plane=1e5)
                    values={k:v.cpu().numpy() for k,v in p.items()}
                    if hidden is not None:
                        base=np.load(folder/(case['id']+'-'+geometry+'-A_color.npz'));values={k:np.where(band[...,None] if v.ndim==3 else band,v,base[k]) for k,v in values.items()}
                    path=folder/(case['id']+'-'+geometry+'-'+method+'.npz');save(path,prediction_srgb=display(values['rgb_linear']),**values);local.append(dict(scene=name,frame=f,id=case['id'],nominal_focus_m=nominal,original_predicted_focus_m=case['focus_m'],focus_m=focus,geometry=geometry,method=method,path=str(path),seconds=time.monotonic()-start,hidden_unchanged=True,frozen_original_method_band=True))
                a=np.load(desc['geometry']/f'f{f:03d}-N1.4-F{nominal:g}-K16.npz')
                with torch.no_grad():stat,T=transmittance(new,K,a,fg_ids,focus)
                tp=folder/(case['id']+'-'+geometry+'-T.npz');save(tp,transmittance=T);tt.append(dict(scene=name,frame=f,nominal_focus_m=nominal,focus_m=focus,geometry=geometry,path=str(tp),**stat));print('D2_RENDER',name,f,geometry,nominal,stat['hidden']['mean'],flush=True)
            del new;torch.cuda.empty_cache()
        write(manifest,dict(rows=local,fit=fit,zeros=zz,transmittance=tt));rows+=local;zeros+=zz;stats+=tt;del cloud;torch.cuda.empty_cache()
    write(OUT/(tag+'.json'),dict(rows=rows,fits=fits,zeros=zeros,transmittance=stats,relative_layer_structure=relative));write(REC/(PREFIX+'-D2-transmittance'+('-relative' if relative else '')+'.json'),dict(rows=stats,fits=fits,depth_sort_and_alpha_cutoff='gsplat AA,eps2d.0015,alpha>=1/255,alpha cap.999;productbeforepoint;no training',GTrec_geometry_fixed_across_D2=True,relative_layer_structure=relative))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['render'],default='render');p.add_argument('--relative-geometry',action='store_true');a=p.parse_args();C['relative_geometry']=a.relative_geometry;render()
