"""Appendix9 F1--F5: read-only existing scenes and frozen Gaussian attributes."""
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
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from src.aperture_renderer import GaussianScene
from src.visible_depth_interface import dominant_surface,cluster_reference,visible_depth_interface
from src.blocker_attribution import blocker_log_categories
from src.prefix_transmittance import prefix_transmittance
from src.dual_layer import depth_band
from src.analytic_aperture import sampled
from src.layered_aperture import merge_layers
from src.hidden_retrieval_rules import collect,aggregate
from route3_sharp_s1 import raster
from route3_s2a_candidates import load_views
from route3_s2a_evaluate import masks as score_masks
from route3_s2a_proxy import read,display
from route3_step2_diagnostic import fit_background,numpy_stat
from route3_b2_oracle_3d import sparse_pixel_layer
from route3_upper_bound_evaluate import score,pool
from route3_ldr_u1r import model

PREFIX='route3-depth-interface-2026-11-12'
OUT=ROOT/'outputs'/f'{PREFIX}-01'
REC=ROOT/'03_实验记录/records'
DIAG=ROOT/'outputs/route3-step2-diagnostic-2026-11-12-01'
SCENES={
    'metric-pilot-122001-broad':DIAG/'G2/metric-pilot-122001-broad',
    'metric-pilot-122002-multilayer':DIAG/'G2/metric-pilot-122002-multilayer',
    'scene-121305-broad':ROOT/'outputs/route3-network-s2a-2026-11-12-01/scene-121305-broad',
    'pilot-multilayer':ROOT/'outputs/route3-network-pilot-2026-11-12-01/pilot-multilayer'}
REGIONS=['full_image','in_focus','band','recoverable']

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.part')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    tmp.replace(path)

def save(path,**value):
    if path.exists():raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,**value)

def guard(gpu=False):
    if shutil.disk_usage(OUT).free/2**30<60:raise RuntimeError('STOP:disk below60GiB')
    if gpu:
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise RuntimeError('Explicit authorized GPU0 required')
        subprocess.run(['nvidia-smi'],check=True)
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32=False

def entries(folder):return json.loads((folder/'prepared.json').read_text())['rows']
def local(name,frame):return OUT/name/f'f{frame:03d}'
def cloud_for(folder,frame):return GaussianScene(**torch.load(folder/'visible'/f'f{frame:03d}.pt',map_location='cuda',weights_only=True))

def subject_mask(g,nominal):
    z=g['depth_axial_m'];coc=g['K'][0,0]*.05/2.8*abs(1/np.maximum(z,1e-6)-1/nominal)
    return g['center_visible']&(z>0)&(coc<=1)

def minimum_coc_focus(z,subject,fx):
    # Diagnostic search only. Uniform grid in inverse-depth plus exact median.
    v=1/z[subject];v=np.sort(v[np.isfinite(v)&(v>0)].astype(np.float64))
    if not len(v):return None
    grid=np.unique(np.r_[np.linspace(v[0],v[-1],2049),np.median(v)])
    p=np.r_[0.,np.cumsum(v)];n=np.searchsorted(v,grid)
    loss=(grid*n-p[n]+p[-1]-p[n]-grid*(len(v)-n))/len(v)*fx*.05/2.8
    at=int(np.argmin(loss));return dict(focus_m=float(1/grid[at]),mean_subject_CoC_px=float(loss[at]),objective='mean absolute CoC of all valid subject pixels;inverse-depth grid2049 plus exactmedian;descriptive only')

def focus_row(z,subject,nominal,old,fit,rule,clusters=None):
    valid=subject&np.isfinite(z)&(z>0)
    if clusters is None:
        dom=valid;cd=z
    else:
        cd=clusters['cluster_depth'];dom=valid&(clusters['cluster_fraction']>=.9)&(clusters['contribution_alpha']>1e-6)&(cd>0)
    base=dict(nominal_focus_m=nominal,subject_pixels=int(subject.sum()),valid_subject_pixels=int(valid.sum()),dominant_pixels=int(dom.sum()),dominant_fraction=float(dom.sum()/max(subject.sum(),1)),dominant_below_20percent=bool(dom.sum()<.2*subject.sum()))
    if rule=='old':focus=old
    elif rule=='P2':focus=float(np.median(cd[dom])) if dom.any() else None
    else:focus=float(1/(fit['a']/nominal+fit['b'])) if fit['success'] else None
    base.update(rule=rule,focus_m=focus,success=focus is not None)
    if focus is not None:
        base['subject_CoC_px']=numpy_stat((711.111111*.05/2.8*abs(1/z[valid]-1/focus)).ravel())
    return base

def verify():
    guard(True)
    assert cluster_reference(np.r_[np.ones(100),3.],np.r_[np.ones(100)*.0095,.05])[1]>.949
    assert cluster_reference([1.,3.],[.5,.5])[1]==.5
    assert cluster_reference([1.,1.025,1.05,1.075],[.25]*4)[1]<1
    # Tiny controlled cloud: verify color/ED alpha weights and six blocker bins.
    K=torch.tensor([[50.,0,8],[0,50.,8],[0,0,1]],device='cuda')
    means=torch.tensor([[0.,0,1],[0,0,4.9],[0,0,5.1]],device='cuda')
    quats=torch.tensor([[1.,0,0,0]]*3,device='cuda');scales=torch.tensor([[.1,.1,.01]]*3,device='cuda')
    s=GaussianScene(means,quats,scales,torch.ones(3,device='cuda')*.5,torch.ones((3,3),device='cuda'),torch.tensor([0,1,1],device='cuda'))
    from gsplat import rasterization
    _,a,_=rasterization(s.means,s.quats,s.scales,s.opacities,s.colors,torch.eye(4,device='cuda')[None],K[None],16,16,render_mode='RGB+ED',rasterize_mode='antialiased',eps2d=.0015,packed=False,near_plane=1e-4,far_plane=1e5)
    d=dominant_surface(s,K,np.ones((16,16),bool));diff=float(np.max(abs(d['contribution_alpha']-a[0,...,0].cpu().numpy())))
    assert diff<2e-5,diff
    lim=torch.ones((16,16),device='cuda')*5;sel=torch.ones((16,16),device='cuda',dtype=torch.bool)
    logs=blocker_log_categories(s,K,torch.eye(4,device='cuda'),lim,sel)
    old=prefix_transmittance(s,K,torch.eye(4,device='cuda'),lim)
    error=float((logs.sum(0).neg().exp()-old).abs().max())
    assert error<2e-5,error
    assert logs[2].sum()>0 and logs[3].sum()>0
    assert logs[[0,1,4,5]].abs().max()==0
    write(OUT/'verification.json',dict(passed=True,synthetic_cluster_contracts=3,alpha_max_abs_error=diff,category_product_vs_existing_prefix_max_abs_error=error,source_layer_and_margin_bins_correct=True))
    print('F_VERIFY',diff,error,flush=True)

def prepare(names):
    guard(True)
    for name in names:
      folder=SCENES[name];mesh=np.load(folder/'scene-geometry.npz');fgids=mesh['foreground_object_ids']
      for entry in entries(folder):
        f=entry['frame'];dest=local(name,f);path=dest/'interface.json'
        if path.exists():continue
        guard();start=time.monotonic();d=np.load(entry['input_npz']);g=np.load(folder/'geometry'/f'f{f:03d}-center.npz')
        subjects={nom:subject_mask(g,nom) for nom in [1.4,5.]};select=subjects[1.4]|subjects[5.]
        cloud=cloud_for(folder,f);K=torch.tensor(d['K'],device='cuda')
        interface_maps=visible_depth_interface(cloud,K,select)
        zz,aa,valid=[interface_maps[k] for k in ['depth','alpha','valid']]
        clusters={k:v for k,v in interface_maps.items() if k not in ['depth','alpha','valid']}
        error=float(np.max(abs(clusters['contribution_alpha'][select]-aa[select]))) if select.any() else 0.
        if error>2e-4:raise RuntimeError(f'Contribution alpha mismatch {name} {f}: {error}')
        save(dest/'visible-depth.npz',depth=zz,alpha=aa,valid=valid,**{k:v for k,v in clusters.items() if isinstance(v,np.ndarray)})
        background=g['center_visible']&~np.isin(g['object_id'],fgids)&valid
        maps={'visible':zz,'O3_SHARP':d['depth'],'O3_DA3':np.load(folder/'models/calibration/DA3'/f'f{f:03d}.npz')['depth']}
        fits={m:fit_background(g['depth_axial_m'],zmap,background) for m,zmap in maps.items()}
        rows=[];search=[]
        for method,zmap in maps.items():
          for nominal,subject in subjects.items():
            old=next(c['focus_m'] for c in entry['cases'] if c['nominal_focus_m']==nominal and c['fstop']==1.4) if method=='visible' else float(np.median(zmap[subject]))
            search.append(dict(method=method,nominal_focus_m=nominal,**(minimum_coc_focus(zmap,subject,d['K'][0,0]) or {})))
            for rule in ['old','P2','P1']:
                rows.append(dict(method=method,**focus_row(zmap,subject,nominal,old,fits[method],rule,clusters if method=='visible' else None)))
        write(path,dict(scene=name,frame=f,focus=rows,background_fits=fits,minimum_CoC_control=search,contribution_alpha_max_abs_error=error,rare_exact_CPU_pixels=clusters['rare_exact_CPU_pixels'],seconds=time.monotonic()-start,source_depths_and_gaussians_unchanged=True,visible_depth='ED=sum(w_i*z_i)/sum(w_i),actual front-to-back compositor;unsupported pixels1e4,flagged invalid',focus_cluster_median='ordinary depth median inside maximum contribution contiguous valid cluster;unweighted pixel median for focus'))
        print('F_INTERFACE',name,f,round(time.monotonic()-start,2),[(r['nominal_focus_m'],r['focus_m'],r['dominant_fraction']) for r in rows if r['rule']=='P2' and r['method']=='visible'],flush=True)
        del cloud;torch.cuda.empty_cache()

def render(names):
    guard(True);profile=json.loads((REC/'route3-dual-layer-criteria-L2-2026-11-11.json').read_text())['profile']
    for name in names:
      folder=SCENES[name];views=None
      for entry in entries(folder):
        f=entry['frame'];dest=local(name,f);interface=json.loads((dest/'interface.json').read_text());d=np.load(entry['input_npz']);own=np.load(dest/'visible-depth.npz')['depth'];K=torch.tensor(d['K'],device='cuda');E=torch.eye(4,device='cuda');fill=torch.tensor(np.load(entry['fill'])['fill_rgb_linear'],device='cuda');bg=torch.tensor(d['background_depth'],device='cuda');cloud=cloud_for(folder,f);oracle=np.load(folder/'oracle'/f'f{f:03d}.npz')
        for rule in ['old','P2','P1']:
          # Target pool regenerated from the unified map; source observations and
          # frozen R1+R2 algorithm unchanged. No hidden labels in this branch.
          foc={r['nominal_focus_m']:r['focus_m'] for r in interface['focus'] if r['method']=='visible' and r['rule']==rule and r['success']}
          union=np.zeros((512,512),bool)
          for nom,focus in foc.items():union|=depth_band(d['depth'] if rule=='old' else own,d['K'][0,0],focus,.05/2.8)[0]
          rpath=dest/('R1_R2-'+rule+'.npz')
          if rule=='old':r2=np.load(folder/'candidates'/f'f{f:03d}'/'R1_R2.npz')
          elif rpath.exists():r2=np.load(rpath)
          else:
            if views is None:views=load_views(folder)
            pooldata=collect(own,d['K'],d['c2w'],union,views,f);r2,audit=aggregate(pooldata,2,own);save(rpath,**r2);write(dest/('R1_R2-'+rule+'-audit.json'),audit)
          for case in entry['cases']:
            if case['fstop']!=1.4 or case['nominal_focus_m'] not in foc:continue
            guard();nom=case['nominal_focus_m'];focus=foc[nom];band,edge,coc=depth_band(d['depth'] if rule=='old' else own,d['K'][0,0],focus,.05/2.8)
            bandpath=dest/(case['id']+'-'+rule+'-band.npz')
            if not bandpath.exists():save(bandpath,band=band,edge=edge,coc=coc)
            manifest=dest/(case['id']+'-'+rule+'-render.json')
            if manifest.exists():continue
            rows=[];basepath=dest/(case['id']+'-'+rule+'-A_color.npz')
            for method,parent in [('A_color',None),('R1_R2',r2),('GTrec',oracle)]:
                path=dest/(case['id']+'-'+rule+'-'+method+'.npz')
                if path.exists():continue
                start=time.monotonic();torch.cuda.reset_peak_memory_stats();hidden=None
                if parent is not None:
                    uv=parent['uv'];sel=band[np.clip(uv[:,1].astype(int),0,511),np.clip(uv[:,0].astype(int),0,511)]
                    if method=='GTrec':sel&=parent['recoverable']
                    hidden=sparse_pixel_layer(parent,sel,K,E,profile,'cuda')
                with torch.inference_mode():p=sampled(merge_layers(cloud,hidden),K,E,512,512,focus,.05/2.8,64,fill,bg,far_plane=1e5,return_fill_contribution=True)
                values={k:v.cpu().numpy() for k,v in p.items()};outside_error=0.
                if parent is not None:
                    baseline=np.load(basepath);values={k:np.where(band[...,None] if v.ndim==3 else band,v,baseline[k]) for k,v in values.items()};outside_error=float(np.max(abs(values['rgb_linear'][~band]-baseline['rgb_linear'][~band]))) if (~band).any() else 0.
                save(path,prediction_srgb=display(values['rgb_linear']),**values)
                old_error=None;oldpath=folder/'predictions'/(case['id']+'-'+method+'.npz')
                if rule=='old' and oldpath.exists():old_error=float(np.max(abs(values['rgb_linear']-np.load(oldpath)['rgb_linear'])))
                rows.append(dict(scene=name,frame=f,id=case['id'],nominal_focus_m=nom,rule=rule,method=method,focus_m=focus,path=str(path),seconds=time.monotonic()-start,peak_MiB=torch.cuda.max_memory_allocated()/2**20,hidden_gaussians=0 if hidden is None else len(hidden.means),outside_band_max_error=outside_error,old_result_max_linear_error=old_error))
                print('F_RENDER',name,f,nom,rule,method,round(time.monotonic()-start,2),'old_error',old_error,flush=True)
                del p,hidden
            write(manifest,dict(rows=rows,focus_m=focus,band=str(bandpath),visible_GS_and_GTrec_unchanged=True))
        del cloud;torch.cuda.empty_cache()

def normals(mesh,frame,g):
    blocks=[]
    for oi,oid in enumerate(mesh['object_ids']):
        matrix=mesh['object_world_matrices'][frame-1,oi];v=mesh[f'object_{oid}_vertices']@matrix[:3,:3].T+matrix[:3,3];faces=mesh[f'object_{oid}_faces'];n=np.cross(v[faces[:,1]]-v[faces[:,0]],v[faces[:,2]]-v[faces[:,0]]);n/=np.maximum(np.linalg.norm(n,axis=-1,keepdims=True),1e-12);blocks.append(n)
    face=np.concatenate(blocks);tid=g['triangle_id'];known=(tid>=0)&(tid<len(face))&(g['object_id']>0);n=face[np.clip(tid,0,len(face)-1)]
    yy,xx=np.mgrid[:512,:512];r=np.dstack([xx+.5,yy+.5,np.ones((512,512))])@np.linalg.inv(g['K']).T;r=r@g['camera_c2w'][:3,:3].T;r/=np.linalg.norm(r,axis=-1,keepdims=True)
    return known,np.abs((n*r).sum(-1))<np.cos(np.deg2rad(75))

def f4(names):
    rows=[]
    for name in names:
      folder=SCENES[name];mesh=np.load(folder/'scene-geometry.npz')
      for entry in entries(folder):
        f=entry['frame'];g=np.load(folder/'geometry'/f'f{f:03d}-center.npz');d=np.load(entry['input_npz'])
        cached=folder/'zero-evaluation'/f'f{f:03d}.npz' if name.startswith('metric-') else DIAG/'render'/name/f'f{f:03d}'/'zero-original.npz'
        p=np.load(cached)['prediction_srgb'];sse=((p.astype(np.float64)-d['rgb_srgb'])**2).sum(-1)
        boundary=np.zeros((512,512),bool)
        for oid in np.unique(g['object_id']):
            mask=g['object_id']==oid;boundary|=mask&~(cv2.erode(mask.astype(np.uint8),np.ones((7,7),np.uint8))>0)
        known,grazing=normals(mesh,f,g)
        partition=dict(boundary=boundary,interior_grazing=~boundary&known&grazing,interior_non_grazing=~boundary&known&~grazing,interior_unknown_sky=~boundary&~known)
        assert np.all(sum(partition.values())==1)
        for region,mask in dict(full_image=np.ones((512,512),bool),**partition).items():
            n=int(mask.sum());ss=float(sse[mask].sum());rows.append(dict(scene=name,frame=f,region=region,pixels=n,pixel_fraction=n/512**2,sse=ss,SSE_fraction=ss/max(float(sse.sum()),1e-30),psnr=float(-10*np.log10(max(ss/max(3*n,1),1e-30))) if n else None,source=str(cached)))
        dest=local(name,f);dest.mkdir(parents=True,exist_ok=True)
        if not (dest/'F4-masks.npz').exists():save(dest/'F4-masks.npz',**partition,SSE=sse.astype(np.float32))
    write(OUT/'F4.json',dict(rows=rows,zero_images='existing cached predictions only;no new zero-aperture render',boundary='ID-wise7x7 erosion ring',grazing='absolute dot(world face normal,view direction)<cos75deg;cached GT triangle geometry'))
    print('F4_DONE',len(rows),flush=True)

def limits_for(folder,f,nominal,parents):
    """Match a cached true surface hit to unchanged GTrec parent placement."""
    from scipy.spatial import cKDTree
    path=folder/'geometry'/f'f{f:03d}-N1.4-F{nominal:g}-K16.npz'
    with np.load(path) as archive:
        wanted=['object_id','hidden','recoverable','residual_pixels','depth_axial_m','aperture_xy','K']
        wanted+=['world_position'] if 'world_position' in archive.files else ['local_surface_point']
        a={k:archive[k] for k in wanted}
    h=a['hidden']&a['recoverable']&~a['residual_pixels'][None]
    limits=np.zeros(h.shape,np.float32);matched=np.zeros(h.shape,bool);distances=[]
    for obj in np.unique(a['object_id'][h]):
        pp=np.flatnonzero((parents['object_id']==obj)&parents['recoverable']);ray=np.nonzero(h&(a['object_id']==obj))
        if not len(pp):continue
        if 'world_position' in a:points=a['world_position'][ray]
        else:
            mesh=np.load(folder/'scene-geometry.npz');oi=int(np.flatnonzero(mesh['object_ids']==obj)[0]);matrix=mesh['object_world_matrices'][f-1,oi]
            points=a['local_surface_point'][ray]@matrix[:3,:3].T+matrix[:3,3]
        dist,ix=cKDTree(parents['world_position'][pp]).query(points,workers=4)
        # Pixel footprint bound only for diagnostic hit-to-parent lookup, not
        # a change to hidden geometry, retrieval or evaluation masks.
        tol=3*a['depth_axial_m'][ray]/a['K'][0,0]+1e-4;ok=dist<=tol
        index=pp[ix];limits[ray]=parents['depth'][index];matched[ray]=ok;distances.extend(dist[ok].tolist())
    return a,limits,matched,numpy_stat(np.array(distances))

def stats(names,label_reference=False):
    guard(True)
    for name in names:
      folder=SCENES[name];fgids=np.load(folder/'scene-geometry.npz')['foreground_object_ids']
      for entry in entries(folder):
        f=entry['frame'];dest=local(name,f);d=np.load(entry['input_npz']);K=torch.tensor(d['K'],device='cuda');cloud=cloud_for(folder,f);parents=np.load(folder/'oracle'/f'f{f:03d}.npz');inter=json.loads((dest/'interface.json').read_text())
        for nominal in [1.4,5.]:
          a,limits,matched,dist=limits_for(folder,f,nominal,parents);case=next(c for c in entry['cases'] if c['fstop']==1.4 and c['nominal_focus_m']==nominal);regions=score_masks(folder,case);positive=a['hidden']&a['recoverable']&~a['residual_pixels'][None]&regions['recoverable'][None]&matched
          masked=np.where(positive,limits,np.nan)
          import warnings
          with warnings.catch_warnings():warnings.simplefilter('ignore',RuntimeWarning);plane=np.nanmedian(masked,axis=0)
          plane=np.nan_to_num(plane,nan=0.)
          for rule in (['label-ray'] if label_reference else ['P2','P1']):
            output=dest/(case['id']+'-'+rule+'-blockers.json')
            if output.exists():continue
            fr=dict(success=True,focus_m=nominal) if label_reference else next(r for r in inter['focus'] if r['method']=='visible' and r['rule']==rule and r['nominal_focus_m']==nominal)
            if not fr['success']:continue
            focus=fr['focus_m'];hitlogs=[];controllogs=[];lens=[];positions=[];maps=[]
            for k,xy in enumerate(a['aperture_xy']):
                control=np.isin(a['object_id'][k],fgids)&~a['hidden'][k]&regions['recoverable']&(plane>0)
                select=positive[k]|control;lim=np.where(positive[k],limits[k],plane).astype(np.float32)
                E=torch.eye(4,device='cuda');E[:2,3]-=torch.tensor(xy,device='cuda');intr=K.clone();intr[:2,2]+=K.diag()[:2]*torch.tensor(xy,device='cuda')/focus
                logs=blocker_log_categories(cloud,intr,E,torch.tensor(lim,device='cuda'),torch.tensor(select,device='cuda')).cpu().numpy()
                hitlogs.append(logs[:,positive[k]].T);controllogs.append(logs[:,control].T);lens.append(np.full(int(positive[k].sum()),k,np.uint8));positions.append(np.flatnonzero(positive[k]).astype(np.int32))
                maps.append(logs.sum(0))
            hit=np.concatenate(hitlogs);control=np.concatenate(controllogs)
            pp=dest/(case['id']+'-'+rule+'-blockers.npz');save(pp,hidden_log_attenuation=hit,control_log_attenuation=control,lens_index=np.concatenate(lens),pixel=np.concatenate(positions),total_log_maps=np.stack(maps))
            def summarize(logs):
                total=logs.sum(-1);shares=logs.sum(0)/max(total.sum(),1e-30);groups={f'layer{layer}_margin{label}':[layer*3+j] for layer in [0,1] for j,label in enumerate(['le5pct','5to20pct','gt20pct'])};groups.update(layer0=[0,1,2],layer1=[3,4,5],margin_le5pct=[0,3],margin_5to20pct=[1,4],margin_gt20pct=[2,5])
                return dict(T=numpy_stat(np.exp(-total)),category_log_attenuation_fraction=shares.tolist(),category_mean_ray_fraction=(logs/np.maximum(total[:,None],1e-30)).mean(0).tolist() if len(logs) else None,without_category={label:numpy_stat(np.exp(-(total-logs[:,ii].sum(-1)).clip(0))) for label,ii in groups.items()})
            band=np.load(dest/(case['id']+'-'+('P2' if label_reference else rule)+'-band.npz'))['band'];raw=a['hidden']&a['recoverable']&~a['residual_pixels'][None]&regions['recoverable'][None]
            write(output,dict(scene=name,frame=f,nominal_focus_m=nominal,focus_m=focus,rule=rule,hidden=summarize(hit),foreground_blocked_control=summarize(control),raw_hidden_label_rays=int(raw.sum()),matched_GTrec_rays=int((raw&matched).sum()),parent_lookup_distance=dist,method_band_coverage=float(band[regions['recoverable']].mean()) if regions['recoverable'].any() else None,band_outside_rays_included_in_T_statistics=True,data=str(pp),category_order=['layer0_le5pct','layer0_5to20pct','layer0_gt20pct','layer1_le5pct','layer1_5to20pct','layer1_gt20pct'],limit='nearest same-object true surface parent,unchanged GTrec placement axial depth;match<=3GTpixel footprints+0.1mm',sampling='nominal GT K16 pixel/ray labels;actual method subcamera focus used for projection',label_to_actual_focus_shift_px_max=float(np.max(np.abs(a['aperture_xy']*np.diag(d['K'])[:2]*(1/focus-1/nominal)))),counterfactual='statistics only,no Gaussian or rendered result modified'))
            print('F_BLOCKERS',name,f,nominal,rule,len(hit),float(np.median(np.exp(-hit.sum(-1)))) if len(hit) else None,flush=True)
        del cloud;torch.cuda.empty_cache()

def evaluate(names):
    net=model();rows=[]
    for name in names:
      folder=SCENES[name]
      for entry in entries(folder):
        dest=local(name,entry['frame'])
        for case in entry['cases']:
          if case['fstop']!=1.4:continue
          truth=display(read(folder/'dof'/(case['id']+'.exr')));region={k:v for k,v in score_masks(folder,case).items() if k in REGIONS}
          for rule in ['old','P2','P1']:
            for method in ['A_color','R1_R2','GTrec']:
              path=dest/(case['id']+'-'+rule+'-'+method+'.npz')
              if not path.exists():continue
              cache=dest/(case['id']+'-'+rule+'-'+method+'-score.json')
              if cache.exists():values=json.loads(cache.read_text())
              else:values=score(np.load(path)['prediction_srgb'],truth,region,net);write(cache,values)
              for reg,v in values.items():rows.append(dict(scene=name,frame=entry['frame'],nominal_focus_m=case['nominal_focus_m'],id=case['id'],rule=rule,method=method,region=reg,**v))
        print('F_SCORE',name,entry['frame'],flush=True)
    groups={}
    for r in rows:groups.setdefault((r['scene'],r['nominal_focus_m'],r['rule'],r['method'],r['region']),[]).append(r)
    pooled=[dict(scene=k[0],nominal_focus_m=k[1],rule=k[2],method=k[3],region=k[4],**pool(v)) for k,v in groups.items()]
    write(OUT/'F3.json',dict(cells=rows,pooled=pooled,reference='Cycles;existing nominal truth;fstop1.4;K64',scoring_regions='unchanged GT definitions,not method bands',LPIPS='frozen AlexNet LPIPS0.1;64x64 fixed grid,pixel-weighted patches',legacy_thresholds_39p92_57p05_unchanged=True))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['verify','prepare','render','f4','stats','label-rays','evaluate']);p.add_argument('--scene',choices=list(SCENES));a=p.parse_args();names=[a.scene] if a.scene else list(SCENES);OUT.mkdir(exist_ok=True)
    if a.mode=='verify':verify()
    elif a.mode=='label-rays':stats(names,label_reference=True)
    else:dict(prepare=prepare,render=render,f4=f4,stats=stats,evaluate=evaluate)[a.mode](names)

if __name__=='__main__':main()
