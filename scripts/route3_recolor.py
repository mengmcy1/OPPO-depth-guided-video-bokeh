"""Appendix11 P0 contribution analysis, mandatory calibration and P1 colors.

Means, scales, quaternions, opacities and Gaussian count are immutable.
Oracle identity/color paths and input-only retrieval paths remain separate.
"""
import argparse,json,sys,os,time,subprocess,shutil
from pathlib import Path
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
import cv2,numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from route3_local_replacement import desc_for,SCENES,reference,load,write,save,visible_cloud
from src.surface_identity import local_footprint
from src.aperture_renderer import GaussianScene,aperture_disk
from src.local_gaussian_replacement import projection,ray_query_plan
from src.recolor_statistics import surface_statistics,delta_color_statistics
from src.analytic_aperture import sampled
from route3_s2a_proxy import display
from route3_sharp_s1 import raster
from route3_upper_bound_evaluate import score,pool
from route3_ldr_u1r import model
from route3_step2_diagnostic import numpy_stat

PREFIX='route3-recolor-2026-11-13';OUT=ROOT/'outputs'/f'{PREFIX}-01';REC=ROOT/'03_实验记录/records';C=json.loads((ROOT/'configs'/f'{PREFIX}.json').read_text());PREV=ROOT/C['previous']
METHODS=['A_color','Oracle_main','Oracle_aux','Oracle_retrieval_set','Retrieval','Control'];REGIONS=['full_image','in_focus','band','recoverable']

def guard(gpu=False):
    if shutil.disk_usage(OUT).free/2**30<60:raise RuntimeError('STOP disk below60GiB')
    if gpu:
        assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0';subprocess.run(['nvidia-smi'],check=True)
    torch.set_num_threads(4);cv2.setNumThreads(2);torch.backends.cuda.matmul.allow_tf32=False

def t(a):return torch.as_tensor(a,device='cuda')
def fd(name,f):return OUT/name/f'f{f:03d}'
def cd(name,f,case,rule):return fd(name,f)/rule/case['id']
def old_case(name,f,case,rule):return PREV/name/f'f{f:03d}'/rule/case['id']
def sse(pred,truth,mask):
    n=int(mask.sum());err=float(((pred[mask].astype(np.float64)-truth[mask])**2).sum())
    return dict(pixels=n,sse=err,psnr=float(-10*np.log10(max(err/max(n*3,1),1e-30))) if n else None)
def changes(a,b,mask):
    d=a[mask].astype(np.float64)-b[mask];return dict(pixels=int(mask.sum()),MAE=float(abs(d).mean()) if len(d) else None,RMSE=float(np.sqrt((d*d).mean())) if len(d) else None,max_abs=float(abs(d).max()) if len(d) else None,fraction_pixels_max_change_gt_1_255=float((abs(d).max(-1)>1/255).mean()) if len(d) else None)

def weighted_stats(value,weight):
    good=np.isfinite(value)&np.isfinite(weight)&(weight>0);v=value[good].astype(np.float64);w=weight[good].astype(np.float64)
    if not len(v):return dict(n=0,weight=0.,mean=None,quantiles=None)
    order=np.argsort(v,kind='stable');v=v[order];w=w[order];cs=w.cumsum();qs=np.array([.01,.1,.25,.5,.75,.9,.99]);return dict(n=len(v),weight=float(cs[-1]),mean=float(v@w/cs[-1]),quantiles=v[np.minimum(np.searchsorted(cs,qs*cs[-1]),len(v)-1)].tolist(),fraction_error_gt_0p05=float(w[v>.05].sum()/cs[-1]))

def recolor(cloud,colors):
    # Sharing these exact tensors makes geometry/opacity/count invariance
    # structural, rather than relying on an approximately equal export.
    return GaussianScene(cloud.means,cloud.quats,cloud.scales,cloud.opacities,colors,cloud.source_frames)

def match_case(name,desc,entry,case,cloud,identity,obs):
    f=entry['frame'];dest=fd(name,f)/'K16'/case['id'];path=dest/'matching.npz'
    if path.exists():return load(path)
    start=time.monotonic();key=f'f{f:03d}-N{case["fstop"]:g}-F{case["nominal_focus_m"]:g}-K16.npz'
    with np.load(desc['geometry']/key) as a:g={k:a[k] for k in ['object_id','local_surface_point','depth_axial_m','hidden','recoverable','residual_pixels','reference_uv_center','aperture_xy','K','camera_c2w']}
    _,regions=reference(name,desc,case,f);roi=g['hidden']&g['recoverable']&~g['residual_pixels'][None]&regions['recoverable'][None]
    mesh=np.load(desc['scene']/'scene-geometry.npz');alignment=load(PREV/name/f'f{f:03d}'/'alignment.npz');K=t(g['K'].astype(np.float32));mass=torch.zeros(len(cloud.means),device='cuda');raystats=[];lens=[];pixels=[]
    sid=t(identity['object_id']);slocal=t(identity['local_point']);hascolor=t(identity['oracle_color_valid']);unobs=t(obs['unobserved'])
    for k,xy in enumerate(g['aperture_xy']):
        y,x=np.nonzero(roi[k]);foot=local_footprint(g['local_surface_point'][k],g['object_id'][k],g['depth_axial_m'][k],mesh,f,g['K'],g['camera_c2w']);uv=g['reference_uv_center'][k,y,x];cx=np.clip(uv[:,0].astype(int),0,511);cy=np.clip(uv[:,1].astype(int),0,511);raw=g['depth_axial_m'][k,y,x];inverse=alignment['a'][cy,cx]/raw+alignment['b'][cy,cx];valid=alignment['local_supported'][cy,cx]&(inverse>1e-4);az=np.zeros(len(raw),np.float32);az[valid]=1/inverse[valid]
        # These are the actual physical GT-labelled aperture rays. Changing
        # focus only shifts image coordinates; matching is shared by P2/P1.
        view=torch.eye(4,device='cuda');view[:2,3]-=t(xy);intr=K.clone();intr[:2,2]+=K.diag()[:2]*t(xy)/case['nominal_focus_m'];query=np.c_[x+.5,y+.5].astype(np.float32);plan=ray_query_plan(t(query))
        r=surface_statistics(cloud,projection(cloud,intr,view),plan,t(g['object_id'][k,y,x]),t(g['local_surface_point'][k,y,x]),t(foot[y,x]),t(raw),t(az),sid,slocal,hascolor,unobs,mass)
        raystats.append(r.cpu().numpy().T);lens.append(np.full(len(y),k,np.uint8));pixels.append((y*512+x).astype(np.int32))
    weights=mass.cpu().numpy();aux=(weights>0)&identity['oracle_color_valid'];main=aux&obs['unobserved'];r=np.concatenate(raystats);den=np.maximum(r[:,0],1e-30)
    save(path,gaussian_optical_weight=weights,main_mask=main,aux_mask=aux,ray_statistics=r,ray_lens=np.concatenate(lens),ray_pixel=np.concatenate(pixels))
    error_linear=abs(cloud.colors.cpu().numpy()-identity['oracle_color']).mean(-1);error_display=abs(display(cloud.colors.cpu().numpy())-display(identity['oracle_color'])).mean(-1)
    detail=dict(scene=name,frame=f,id=case['id'],nominal_focus_m=case['nominal_focus_m'],fstop=case['fstop'],GT_hidden_rays=len(r),top8_match_fraction=numpy_stat(r[:,7]/np.maximum(r[:,6],1)),top8_any_match_fraction=float((r[:,7]>0).mean()) if len(r) else None,top8_count_weighted_match_fraction=float(r[:,7].sum()/max(r[:,6].sum(),1)),top8_close_depth_wrong_identity_aligned_fraction=float(r[:,10].sum()/max(r[:,6].sum(),1)),top8_close_depth_wrong_identity_raw_fraction=float(r[:,11].sum()/max(r[:,6].sum(),1)),optical_surface_match=numpy_stat(r[:,1]/den),optical_with_other_observation=numpy_stat(r[:,2]/den),optical_main=numpy_stat(r[:,3]/den),close_depth_wrong_identity_raw=numpy_stat(r[:,4]/den),close_depth_wrong_identity_aligned=numpy_stat(r[:,5]/den),matched_gaussians=int((weights>0).sum()),main_gaussians=int(main.sum()),aux_gaussians=int(aux.sum()),main_layer0=int((main&(cloud.source_frames.cpu().numpy()==0)).sum()),main_layer1=int((main&(cloud.source_frames.cpu().numpy()==1)).sum()),error={label:dict(linear=weighted_stats(error_linear,weights*sel),display=weighted_stats(error_display,weights*sel)) for label,sel in [('main',main),('aux',aux)]},seconds=time.monotonic()-start,ray_geometry='nominal physical K16 rays;P2/P1 principalpoint shifts cancel for same physical ray;full K64 prediction uses actual methodcamera',path=str(path))
    write(dest/'matching.json',detail);print('R3_MATCH',name,f,case['id'],len(r),int(main.sum()),int(aux.sum()),round(time.monotonic()-start,2),flush=True)
    return load(path)

def p0(names):
    guard(True)
    for name in names:
      desc=desc_for(name)
      for entry in desc['entries']:
        f=entry['frame'];dest=fd(name,f);identity=load(dest/'identity-colors.npz');obs=load(PREV/name/f'f{f:03d}'/'observation.npz');d=load(entry['input_npz']);cloud=visible_cloud(desc,entry);K=t(d['K']);E=torch.eye(4,device='cuda');fill=t(np.load(entry['fill'])['fill_rgb_linear']);assert len(cloud.means)==2*768**2==len(identity['object_id'])
        assert torch.all(cloud.source_frames[:768**2]==0) and torch.all(cloud.source_frames[768**2:]==1)
        oracle=t(identity['oracle_color']);oracle=torch.where(t(identity['oracle_color_valid'])[:,None],oracle,cloud.colors);difference=oracle-cloud.colors
        zero_path=dest/'zero-baseline-linear.npz'
        if zero_path.exists():zero_base=load(zero_path)['rgb_linear']
        else:
            _,current,alpha=delta_color_statistics(cloud,K,E,torch.zeros((len(cloud.means),6),device='cuda'),include_current=True);nativealpha=load(PREV/name/f'f{f:03d}'/'visible-depth.npz')['alpha'];zero_base=current.cpu().numpy()+(1-nativealpha[...,None])*fill.cpu().numpy();save(zero_path,rgb_linear=zero_base,alpha_sum_statistics=alpha.cpu().numpy())
        for case in entry['cases']:
          match=match_case(name,desc,entry,case,cloud,identity,obs);dc=torch.cat([difference*t(match['main_mask'])[:,None],difference*t(match['aux_mask'])[:,None]],-1).contiguous()
          for rule in ['P2','P1']:
            folder=cd(name,f,case,rule);done=folder/'P0.json'
            if done.exists():continue
            guard();start=time.monotonic();previous=json.loads((old_case(name,f,case,rule)/'render.json').read_text());focus=previous['focus_m'];base=np.load(old_case(name,f,case,rule)/'R0-none.npz');avg=torch.zeros((512,512,6),device='cuda')
            for xy in aperture_disk(64,.05/(2*case['fstop']),device=K.device,dtype=K.dtype):
                view=E.clone();view[:2,3]-=xy;intr=K.clone();intr[:2,2]+=K.diag()[:2]*xy/focus;delta,_,_=delta_color_statistics(cloud,intr,view,dc);avg+=delta
            avg=(avg/64).cpu().numpy();zero,_,_=delta_color_statistics(cloud,K,E,dc);zero=zero.cpu().numpy();truth,regions=reference(name,desc,case,f);arrays={};rows=[]
            for ii,label in enumerate(['Oracle_main','Oracle_aux']):
                linear=base['rgb_linear']+avg[...,ii*3:ii*3+3];pred=display(linear);z=display(zero_base+zero[...,ii*3:ii*3+3]);arrays[label+'_linear']=linear;arrays[label+'_srgb']=pred;arrays[label+'_zero_srgb']=z
                rows.append(dict(method=label,regions={r:sse(pred,truth,m) for r,m in regions.items()},zero_full=sse(z,d['rgb_srgb'],np.ones((512,512),bool)),zero_infocus=sse(z,d['rgb_srgb'],regions['in_focus']),outside_change=changes(pred,base['prediction_srgb'],~regions['recoverable']),zero_outside_change=changes(z,display(zero_base),~regions['recoverable'])))
            save(folder/'P0-predictions.npz',**arrays)
            write(done,dict(scene=name,frame=f,id=case['id'],nominal_focus_m=case['nominal_focus_m'],fstop=case['fstop'],rule=rule,focus_m=focus,rows=rows,baseline_regions={r:sse(base['prediction_srgb'],truth,m) for r,m in regions.items()},baseline_zero_full=sse(display(zero_base),d['rgb_srgb'],np.ones((512,512),bool)),seconds=time.monotonic()-start,baseline=str(old_case(name,f,case,rule)/'R0-none.npz'),GTset_fixed_globally=True,K_prediction=64,K_matching=16,RGB_renderer_called=False))
            print('R3_P0',name,f,rule,case['id'],round(time.monotonic()-start,2),flush=True)
        del cloud;torch.cuda.empty_cache()

def actual_cloud(cloud,identity,match,method,retrieval_mask=None,retrieval_colors=None,seed=0):
    colors=cloud.colors.clone()
    if method in ['Oracle_main','Oracle_aux','Control']:
        mask=match['main_mask' if method in ['Oracle_main','Control'] else 'aux_mask'];values=identity['oracle_color'][mask].copy()
        if method=='Control':values=values[np.random.default_rng(seed).permutation(len(values))]
        colors[t(mask)]=t(values)
    elif method=='Retrieval':colors[t(retrieval_mask)]=t(retrieval_colors[retrieval_mask])
    elif method=='Oracle_retrieval_set':
        mask=retrieval_mask&identity['oracle_color_valid'];colors[t(mask)]=t(identity['oracle_color'][mask])
    return recolor(cloud,colors)

def render_one(cloud,K,entry,focus,stop):
    fill=t(np.load(entry['fill'])['fill_rgb_linear']);d=load(entry['input_npz']);E=torch.eye(4,device='cuda')
    with torch.inference_mode():
        p=sampled(cloud,K,E,512,512,focus,.05/(2*stop),64,fill,t(d['background_depth']),far_plane=1e5);rgb,_,alpha=raster(cloud,K)
    values={k:v.cpu().numpy() for k,v in p.items()};values['prediction_srgb']=display(values['rgb_linear']);values['zero_rgb_linear']=(rgb+(1-alpha[...,None])*fill).cpu().numpy();values['zero_srgb']=display(values['zero_rgb_linear']);return values

def calibration():
    guard(True);name=SCENES[0];desc=desc_for(name);entry=next(e for e in desc['entries'] if e['frame']==30);case=next(c for c in entry['cases'] if c['nominal_focus_m']==5 and c['fstop']==1.4);folder=cd(name,30,case,'P2');p0=json.loads((folder/'P0.json').read_text());identity=load(fd(name,30)/'identity-colors.npz');match=load(fd(name,30)/'K16'/case['id']/'matching.npz');cloud=visible_cloud(desc,entry);K=t(np.load(entry['input_npz'])['K']);new=actual_cloud(cloud,identity,match,'Oracle_main');values=render_one(new,K,entry,p0['focus_m'],1.4);save(OUT/'P0-single-frame-check.npz',**values);truth,regions=reference(name,desc,case,30);pred=load(folder/'P0-predictions.npz');target=np.load(entry['input_npz'])['rgb_srgb'];actual=sse(values['prediction_srgb'],truth,regions['recoverable']);expected=sse(pred['Oracle_main_srgb'],truth,regions['recoverable']);az=sse(values['zero_srgb'],target,np.ones((512,512),bool));ez=sse(pred['Oracle_main_zero_srgb'],target,np.ones((512,512),bool));dofdiff=abs(actual['psnr']-expected['psnr']);zdiff=abs(az['psnr']-ez['psnr'])
    write(OUT/'P0-single-frame-check.json',dict(scene=name,frame=30,id=case['id'],rule='P2',method='Oracle_main',actual_recoverable=actual,expected_recoverable=expected,actual_zero=az,expected_zero=ez,PSNR_difference_recoverable_dB=dofdiff,PSNR_difference_zero_dB=zdiff,max_display_abs_error=float(abs(values['prediction_srgb']-pred['Oracle_main_srgb']).max()),passed=bool(dofdiff<=.1 and zdiff<=.1)))
    print('R3_CALIBRATION',dofdiff,zdiff,flush=True)

def sample_bilinear(image,uv):
    x=np.clip(uv[:,0]-.5,0,511);y=np.clip(uv[:,1]-.5,0,511);x0=np.floor(x).astype(int);y0=np.floor(y).astype(int);x1=np.minimum(x0+1,511);y1=np.minimum(y0+1,511);fx=x-x0;fy=y-y0
    if image.ndim==3:fx=fx[:,None];fy=fy[:,None]
    return (1-fx)*(1-fy)*image[y0,x0]+fx*(1-fy)*image[y0,x1]+(1-fx)*fy*image[y1,x0]+fx*fy*image[y1,x1]

def retrieval_frame(name,desc,entry,cloud,obs):
    """Input-only: no Oracle identity, source GT maps, labels or target hidden GT."""
    path=fd(name,entry['frame'])/'retrieval-colors.npz'
    if path.exists():return load(path)
    start=time.monotonic();target=np.load(desc['calibration']/f'f{entry["frame"]:03d}.npz');pose=target['c2w'];index=np.flatnonzero(obs['unobserved']);xyz=cloud.means.detach().cpu().numpy()[index].astype(np.float64);world=xyz@pose[:3,:3].T+pose[:3,3];observations=np.full((len(index),12,3),np.nan,np.float32);count=np.zeros(len(index),np.uint8);per_source=[];j=0;K=np.load(entry['input_npz'])['K']
    for source in C['source_frames']:
        if source==entry['frame']:continue
        estimated=np.load(desc['calibration']/f'f{source:03d}.npz');camera=(world-estimated['c2w'][:3,3])@estimated['c2w'][:3,:3];pp=camera@K.T;uv=pp[:,:2]/np.maximum(pp[:,2:],1e-8);inside=(camera[:,2]>0)&np.isfinite(uv).all(-1)&(uv[:,0]>=0)&(uv[:,0]<512)&(uv[:,1]>=0)&(uv[:,1]<512);depth=sample_bilinear(estimated['depth'],uv);error=abs(camera[:,2]-depth);ok=inside&np.isfinite(depth)&(depth>0)&(error<=np.maximum(.01,.02*depth));rgb=np.load(desc['model']/f'f{source:03d}.npz')['rgb_linear'];color=sample_bilinear(rgb,uv);observations[ok,j]=color[ok];count[ok]+=1;per_source.append(dict(frame=source,inside=int(inside.sum()),estimated_depth_consistent=int(ok.sum())));j+=1
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning);median=np.nanmedian(observations,axis=1);mad=np.nanmedian(abs(observations-median[:,None,:]).mean(-1),axis=1)
    valid=(count>=2)&np.isfinite(mad)&(mad<=.05)&np.isfinite(median).all(-1);median[~np.isfinite(median)]=0;confidence=np.where(np.isfinite(mad),np.minimum(count/3,1)*np.exp(-mad/.05),0).astype(np.float32)
    colors=cloud.colors.detach().cpu().numpy().copy();colors[index]=median;accepted=np.zeros(len(cloud.means),bool);accepted[index]=valid;conf=np.zeros(len(cloud.means),np.float32);conf[index]=confidence;counts=np.zeros(len(cloud.means),np.uint8);counts[index]=count
    save(path,colors=colors,valid=accepted,confidence=conf,source_count=counts)
    write(fd(name,entry['frame'])/'retrieval-colors.json',dict(scene=name,frame=entry['frame'],unobserved_candidates=len(index),accepted_before_band=int(valid.sum()),source=per_source,minimum_sources=2,color_MAD_L1_max=.05,visibility='two-sided axial depth agreement<=max1cm,2%source depth',source_geometry='existing E1-calibrated SHARP estimated visible-depth maps,as prior retrieval;no new geometry/inference',GT_inputs='E1 given poses and shared P2subjectfocus only;no GTdepth,identity,hiddenGT or evaluation masks',seconds=time.monotonic()-start))
    return load(path)

def render(names):
    guard(True);decision=json.loads((OUT/'P0-decision.json').read_text())
    if decision['early_stop']:raise RuntimeError('P0 early stop: P1 not authorized')
    for name in names:
      desc=desc_for(name)
      for entry in desc['entries']:
        f=entry['frame'];identity=load(fd(name,f)/'identity-colors.npz');obs=load(PREV/name/f'f{f:03d}'/'observation.npz');cloud=visible_cloud(desc,entry);K=t(np.load(entry['input_npz'])['K']);retrieved=retrieval_frame(name,desc,entry,cloud,obs);uvz=cloud.means.detach().cpu().numpy()@K.cpu().numpy().T;uv=uvz[:,:2]/np.maximum(uvz[:,2:],1e-8);x=np.clip(np.floor(uv[:,0]).astype(int),0,511);y=np.clip(np.floor(uv[:,1]).astype(int),0,511);inside=(uvz[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<512)&(uv[:,1]>=0)&(uv[:,1]<512)
        for case in entry['cases']:
          match=load(fd(name,f)/'K16'/case['id']/'matching.npz')
          for rule in ['P2','P1']:
            folder=cd(name,f,case,rule);done=folder/'P1.json'
            if done.exists():continue
            guard();p0=json.loads((folder/'P0.json').read_text());band=np.load(old_case(name,f,case,rule)/'support-masks.npz')['band'];candidate=obs['unobserved']&inside&band[y,x];selection=candidate&retrieved['valid'];seed=C['control_seed']+10000*SCENES.index(name)+100*f+10*(case['nominal_focus_m']==5)+int(round(case['fstop']*10))+(0 if rule=='P2' else 1);save(folder/'retrieval-selection.npz',selected=selection,candidates=candidate)
            rows=[]
            for method in METHODS:
                path=folder/(method+'.npz')
                if path.exists():continue
                start=time.monotonic();new=actual_cloud(cloud,identity,match,method,selection,retrieved['colors'],seed);values=render_one(new,K,entry,p0['focus_m'],case['fstop']);save(path,**values);old_error=None
                if method=='A_color':old_error=float(abs(values['rgb_linear']-np.load(old_case(name,f,case,rule)/'R0-none.npz')['rgb_linear']).max())
                rows.append(dict(method=method,path=str(path),seconds=time.monotonic()-start,Acolor_previous_max_linear_error=old_error));print('R3_P1',name,f,rule,case['id'],method,round(time.monotonic()-start,2),flush=True)
            occupied=np.zeros((512,512),bool);occupied[y[selection],x[selection]]=True;correct_mass=match['gaussian_optical_weight'];write(done,dict(scene=name,frame=f,id=case['id'],rule=rule,nominal_focus_m=case['nominal_focus_m'],fstop=case['fstop'],focus_m=p0['focus_m'],rows=rows,control_seed=int(seed),retrieval=dict(candidates=int(candidate.sum()),colored=int(selection.sum()),acceptance_fraction=float(selection.sum()/max(candidate.sum(),1)),band_pixel_coverage=float(occupied[band].mean()) if band.any() else None,confidence=numpy_stat(retrieved['confidence'][selection]),source_count=numpy_stat(retrieved['source_count'][selection]),input_alpha_contribution_fraction=float(obs['sum_weight'][selection].sum()/max(obs['sum_weight'].sum(),1e-30)),GTmatched_optical_weight_coverage_diagnostic=float(correct_mass[selection].sum()/max(correct_mass.sum(),1e-30)),Oracle_at_retrieval_unchanged_without_identity_or_color=int((selection&~identity['oracle_color_valid']).sum())),geometry_opacity_and_count_unchanged=True,color_only=True))
        del cloud;torch.cuda.empty_cache()

def evaluate(names):
    guard();net=model();allrows=[]
    for name in names:
      desc=desc_for(name)
      for entry in desc['entries']:
        f=entry['frame'];d=np.load(entry['input_npz'])
        for case in entry['cases']:
          truth,regions=reference(name,desc,case,f)
          for rule in ['P2','P1']:
            folder=cd(name,f,case,rule)
            if not (folder/'P1.json').exists():continue
            base=np.load(folder/'A_color.npz');predicted=json.loads((folder/'P0.json').read_text());forecast={r['method']:r for r in predicted['rows']}
            for method in METHODS:
              cache=folder/(method+'-scores.json')
              if cache.exists():result=json.loads(cache.read_text())
              else:
                p=np.load(folder/(method+'.npz'));result=dict(regions=score(p['prediction_srgb'],truth,regions,net),zero_full=sse(p['zero_srgb'],d['rgb_srgb'],np.ones((512,512),bool)),zero_infocus=sse(p['zero_srgb'],d['rgb_srgb'],regions['in_focus']),outside_change=changes(p['prediction_srgb'],base['prediction_srgb'],~regions['recoverable']),zero_outside_change=changes(p['zero_srgb'],base['zero_srgb'],~regions['recoverable']),alpha_max_change=float(abs(p['alpha']-base['alpha']).max()));write(cache,result)
              for region,m in result['regions'].items():
                expected=forecast[method]['regions'][region]['psnr'] if method in forecast else None
                expected_zero=forecast[method]['zero_full']['psnr'] if method in forecast else None
                allrows.append(dict(scene=name,frame=f,id=case['id'],rule=rule,nominal_focus_m=case['nominal_focus_m'],fstop=case['fstop'],method=method,region=region,**m,zero_PSNR=result['zero_full']['psnr'],zero_sse=result['zero_full']['sse'],zero_pixels=result['zero_full']['pixels'],zero_infocus_PSNR=result['zero_infocus']['psnr'],alpha_max_change=result['alpha_max_change'],P0_expected_PSNR=expected,actual_minus_expected_PSNR=m['psnr']-expected if expected is not None else None,P0_expected_zero_PSNR=expected_zero,zero_actual_minus_expected_PSNR=result['zero_full']['psnr']-expected_zero if expected_zero is not None else None,outside_MAE=result['outside_change']['MAE'],outside_max_change=result['outside_change']['max_abs'],outside_fraction_gt1_255=result['outside_change']['fraction_pixels_max_change_gt_1_255']))
            print('R3_SCORE',name,f,rule,case['id'],flush=True)
    write(OUT/'P1-scores.json',dict(rows=allrows,reference='Cycles;frozen GT regions,AlexNetLPIPS0.1 fixed64patch'))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['p0','calibration','render','evaluate']);p.add_argument('--scene',choices=SCENES);a=p.parse_args()
    if a.mode=='calibration':calibration()
    else:dict(p0=p0,render=render,evaluate=evaluate)[a.mode]([a.scene] if a.scene else SCENES)

if __name__=='__main__':main()
