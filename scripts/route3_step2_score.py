"""Appendix8 D1-D3: original regions, fixed patch metrics and phase probes."""
import argparse,csv,json,sys
from pathlib import Path
import cv2,numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from route3_step2_diagnostic import C,OUT,REC,PREFIX,REGIONS,write,descriptor,original,references,model,score,pool,display,read,numpy_stat

def alias():
    from scipy.ndimage import distance_transform_edt
    rows=[];figs=[]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for name in C['scenes']:
        folder=OUT/'alias'/name;meta=json.loads((folder/'manifest.json').read_text())
        if (folder/'materials-corrected.json').exists():meta['objects']=json.loads((folder/'materials-corrected.json').read_text())['objects']
        jobs={r['label']:r for r in meta['rows']};base=read(jobs['center']['path']);shift=read(jobs['shift']['path']);noise=read(jobs['noise']['path']);P=read(jobs['center']['passes']['Position']);ids=read(jobs['center']['passes']['IndexOB'])[...,0].round().astype(int);desc=descriptor(name);geo=np.load(desc['geometry']/'f030-center.npz');cam=geo['camera_c2w'][:3,3]
        py,px=np.gradient(P.astype(np.float64),axis=(0,1));normal=np.cross(px,py);normal/=np.maximum(np.linalg.norm(normal,axis=-1,keepdims=True),1e-12);view=cam-P;view/=np.maximum(np.linalg.norm(view,axis=-1,keepdims=True),1e-12);grazing=abs((normal*view).sum(-1))<np.cos(np.deg2rad(75));period=np.full(ids.shape,np.inf);delta=jobs['shift']['expected_image_delta_xy'];matrix=np.array([[1,0,delta[0]],[0,1,delta[1]]],np.float32);warped=cv2.warpAffine(base,matrix,(512,512),flags=cv2.INTER_LANCZOS4,borderMode=cv2.BORDER_REFLECT101)
        for obj in meta['objects']:
            mask=ids==obj['id'];interior=cv2.erode(mask.astype(np.uint8),np.ones((11,11),np.uint8))>0;local=np.full(ids.shape,np.inf);pattern_errors=[]
            for pat in obj['patterns']:
                if 'error' in pat:pattern_errors.append(pat);continue
                G=np.array(pat['world_phase_gradient_cycles_m']);gx=px@G.T;gy=py@G.T;freq=np.sqrt(gx*gx+gy*gy).max(-1);pp=1/np.maximum(freq,1e-12);local=np.minimum(local,pp)
            valid=interior&~grazing&np.isfinite(local)&(local<1e10);period[mask]=local[mask];stats={}
            for label,valid_pixels in [('interior',interior),('non_grazing',interior&~grazing),('grazing',interior&grazing)]:
                n=int(valid_pixels.sum());err=(display(shift)-display(warped))**2;nr=(display(noise)-display(base))**2
                phase=float(err[valid_pixels].mean()) if n else None;noise_mse=float(nr[valid_pixels].mean()) if n else None
                stats[label]=dict(pixels=n,phase_PSNR_dB=float(-10*np.log10(max(phase,1e-12))) if n else None,independent_seed_noise_PSNR_dB=float(-10*np.log10(max(noise_mse,1e-12))) if n else None,phase_to_noise_SSE_ratio=phase/max(noise_mse,1e-12) if n else None,linear_phase_RMSE=float(np.sqrt(np.mean((shift[valid_pixels]-warped[valid_pixels])**2))) if n else None)
            rows.append(dict(scene=name,frame=30,object_id=obj['id'],object_name=obj['name'],projected_period_pixels=numpy_stat(local[valid]),grazing_projected_period_pixels=numpy_stat(local[interior&grazing&np.isfinite(local)]),fraction_period_below8=float((local[valid]<8).mean()) if valid.any() else None,alias=stats,patterns=obj['patterns'],unsupported=pattern_errors,world_position_source=jobs['center']['passes']['Position'],all_extrinsic_changes=[r['max_extrinsic_change'] for r in meta['rows']]))
        np.savez_compressed(folder/'period-map.npz',period_pixels=period,grazing=grazing,object_id=ids)
        fig,ax=plt.subplots(1,3,figsize=(15,5));ax[0].imshow(display(base));p=ax[1].imshow(np.log2(np.clip(period,.0625,64)),vmin=-4,vmax=6,cmap='viridis');fig.colorbar(p,ax=ax[1],label='log2 projected cycle/feature pixels');ax[2].imshow(np.sqrt(np.mean((display(shift)-display(warped))**2,-1)),vmin=0,vmax=.15,cmap='magma')
        for a in ax:a.axis('off')
        fig.suptitle(name+';center / projected feature scale / halfpixel phase error');path=folder/'alias-period.png';fig.savefig(path,dpi=120);plt.close(fig);figs.append(str(path))
    write(REC/(PREFIX+'-alias.json'),dict(rows=rows,figures=figs,phase='lens shift_x+.5/512;image delta -.5px;no extrinsic motion;Lanczos4;11x11 object erosion',period='world-coordinate Jacobian,material node phase gradients,smallest component period;noise finest-octave feature proxy,not literal periodicity;hard-edge harmonics unbounded',criteria_not_fitted_to_method_results=True));print('D_ALIAS_SCORED',len(rows),flush=True)

def metrics(relative=False):
    suffix='-relative' if relative else '';torch.set_num_threads(4);net=model();render=json.loads((OUT/('render'+suffix+'.json')).read_text());cells=[];focus_selection=[];zero_rows=[];per_case={};cachefolder=OUT/('score-cache'+suffix);cachefolder.mkdir(exist_ok=True)
    for name in C['scenes']:
      desc=descriptor(name);mesh=np.load(desc['scene']/'scene-geometry.npz');fg_ids=mesh['foreground_object_ids']
      for entry in desc['entries']:
        f=entry['frame'];d=np.load(entry['input_npz']);g=np.load(desc['geometry']/f'f{f:03d}-center.npz');ids=g['object_id'];boundary=np.zeros((512,512),bool)
        for i in np.unique(ids):boundary|=(ids==i)&~(cv2.erode((ids==i).astype(np.uint8),np.ones((7,7),np.uint8))>0)
        zr=dict(full_image=np.ones(ids.shape,bool),occlusion_boundary=boundary)
        for focus in [1.4,5.]:zr['in_focus_'+str(focus)]=g['center_visible']&(g['K'][0,0]*.05/2.8*abs(1/np.maximum(g['depth_axial_m'],1e-6)-1/focus)<=1)
        for zrow in [r for r in render['zeros'] if r['scene']==name and r['frame']==f]:
            p=np.load(zrow['path'])['prediction_srgb'];v=score(p,d['rgb_srgb'],zr,net)
            for reg,m in v.items():zero_rows.append(dict(scene=name,frame=f,geometry=zrow['geometry'],region=reg,**m))
        for case in entry['cases']:
            if case['fstop']!=1.4:continue
            focus=case['nominal_focus_m'];mask=zr['in_focus_'+str(focus)];fg=np.isin(ids,fg_ids)
            focus_selection.append(dict(scene=name,frame=f,nominal_focus_m=focus,predicted_focus_m=case['focus_m'],selected_pixels=int(mask.sum()),GT_depth_median=float(np.median(g['depth_axial_m'][mask])),estimated_depth_median=float(np.median(d['depth'][mask])),foreground_share=float(fg[mask].mean()),selection='all GT center-visible pixels with physical CoC radius<=1px at N1.4;median of estimated first-layer depth;not a single pixel'))
            refs,regions=references(name,case,desc);paths=original(name,case);candidates=[dict(geometry='original',method=m,path=p) for m,p in paths.items()]+[r for r in render['rows'] if r['scene']==name and r['id']==case['id']]
            for predrow in candidates:
                pred=np.load(predrow['path'])['prediction_srgb'];key=name+'-'+case['id']+'-'+predrow['geometry']+'-'+predrow['method'];cache=cachefolder/(key+'.json');data=json.loads(cache.read_text()) if cache.exists() else {}
                for ref,truth in refs.items():
                    for sigma in [0,1,2,4]:
                        ck=ref+'-'+str(sigma)
                        if ck not in data:
                            pp=pred if sigma==0 else cv2.GaussianBlur(pred,(0,0),sigma,borderType=cv2.BORDER_REFLECT101);tt=truth if sigma==0 else cv2.GaussianBlur(truth,(0,0),sigma,borderType=cv2.BORDER_REFLECT101);data[ck]=score(pp,tt,regions,net)
                        for region,v in data[ck].items():cells.append(dict(scene=name,frame=f,id=case['id'],nominal_focus_m=focus,reference=ref,geometry=predrow['geometry'],method=predrow['method'],sigma_px=sigma,region=region,**v))
                write(cache,data)
            print('D_SCORE',name,case['id'],flush=True)
    grouped={}
    for row in cells:
        key=(row['scene'],row['nominal_focus_m'],row['reference'],row['geometry'],row['method'],row['sigma_px'],row['region']);grouped.setdefault(key,[]).append(row)
    pooled=[]
    for key,rows in grouped.items():
        name,focus,reference,geometry,method,sigma,region=key;pooled.append(dict(scene=name,focus_m=focus,reference=reference,geometry=geometry,method=method,sigma_px=sigma,region=region,**pool(rows)))
    result=dict(cells=cells,pooled=pooled,zero_aperture=zero_rows,focus_selection=focus_selection,lowpass_space='displayRGB;both prediction and reference same sigma;GT region masks fixed',LDR_missing_scenes=[n for n in C['scenes'] if n.startswith('scene-')],frozen_original_results_not_changed=True,statistics='pooled SSE PSNR,64x64AlexNet LPIPS0.1,region pixel weighted')
    result['relative_layer_structure_preserved']=relative;write(REC/(PREFIX+'-D-metrics'+suffix+'.json'),result)
    for tag,rows in [('D-cells',cells),('D-pooled',pooled),('D-zero',zero_rows),('focus-selection',focus_selection)]:
        with (REC/(PREFIX+'-'+tag+suffix+'.csv')).open('w') as h:w=csv.DictWriter(h,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['alias','metrics'],required=True);p.add_argument('--relative-geometry',action='store_true');a=p.parse_args();metrics(a.relative_geometry) if a.mode=='metrics' else alias()
