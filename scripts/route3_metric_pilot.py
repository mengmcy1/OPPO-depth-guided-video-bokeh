"""G2 truth checks precede any method inference; stop rather than reject pilots."""
import argparse,json,os,sys,time,subprocess,shutil,csv
from pathlib import Path
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
ROOT=Path(__file__).resolve().parents[1];CONFIG=ROOT/'configs/route3-step2-G2-2026-11-12.json';os.environ['ROUTE3_PILOT_CONFIG']=str(CONFIG);sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import cv2,numpy as np,torch
from route3_s2a_proxy import display,read
from src.dual_layer import depth_band
C=json.loads(CONFIG.read_text());OUT=Path(C['output']);LOG=ROOT/'logs/route3-step2-diagnostic-2026-11-12';REC=ROOT/'03_实验记录/records';PREFIX='route3-step2-diagnostic-2026-11-12-G2';PY='/root/autodl-tmp/envs/route3/bin/python';CPU='/root/autodl-tmp/envs/dofgs/bin/python';LAMA='/root/autodl-tmp/envs/conda-magicbokeh/bin/python';BLENDER='/root/autodl-tmp/tools/blender-4.5.3-linux-x64/blender';FRAMES=C['input_frames']
def write(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def run(cmd,name,gpu=False):
    if shutil.disk_usage(OUT).free/2**30<60:raise RuntimeError('DISK BELOW60GiB')
    if gpu:subprocess.run(['nvidia-smi'],stdout=subprocess.DEVNULL,check=True)
    start=time.monotonic()
    with (LOG/(name+'.log')).open('a') as f:subprocess.run(cmd,cwd=ROOT,env=os.environ.copy(),stdout=f,stderr=subprocess.STDOUT,check=True)
    return time.monotonic()-start
def jobs(s,name,rows):
    path=s/(name+'-jobs.json')
    if path.exists():assert json.loads(path.read_text())==rows
    else:write(path,rows)
    run([BLENDER,'-b','-t','12','--python','scripts/route3_metric_blender.py','--','--jobs',str(path)],s.name+'-'+name,True);return json.loads(path.with_name(path.stem+'-results.json').read_text())

def truth_render():
    OUT.mkdir(exist_ok=True);plans=[]
    for index,name in enumerate(C['scenes'],1):
        s=OUT/name;s.mkdir(exist_ok=True)
        if not(s/'scene.blend').exists():run([BLENDER,'-b','-t','12','--python','scripts/route3_metric_blender.py','--','--build',str(s),'--seed',str(122000+index),'--index',str(index)],name+'-build')
        assert (s/'scene.blend').exists() and (s/'scene.json').exists(),'Blender build did not finish;keep draft and stop'
        common=dict(scene=name,blend=str(s/'scene.blend'),size=512,seed=20261112)
        inputs=[dict(**common,id=name+f'-input-{f}',frame=f,spp=256,dof=False,data_passes=True,cryptomatte=True,output=str(s/'inputs'/f'f{f:03d}.exr')) for f in FRAMES];jobs(s,'inputs',inputs)
        for f in FRAMES:
            p=s/'inputs'/f'f{f:03d}.png'
            if not p.exists():cv2.imwrite(str(p),np.uint8(np.rint(display(read(s/'inputs'/f'f{f:03d}.exr'))[...,::-1]*255)))
        dof=[]
        for f in [1,30,60]:
          for focus in [1.4,5.]:
           for stop in [1.4,2.]:
            key=f'f{f:03d}-N{stop:g}-F{focus:g}';dof.append(dict(**common,id=name+'-'+key,frame=f,spp=1024,dof=True,focus=focus,fstop=stop,output=str(s/'dof'/(key+'.exr'))))
        for focus in [1.4,5.]:
          for tag,spp,seed in [('independent',1024,2026111202),('lower-spp',512,20261112)]:
            dof.append(dict(scene=name,blend=str(s/'scene.blend'),size=512,seed=seed,id=name+f'-noise-{focus:g}-{tag}',frame=30,spp=spp,dof=True,focus=focus,fstop=1.4,output=str(s/'noise'/f'F{focus:g}-{tag}.exr')))
        jobs(s,'dof',dof);m=json.loads((s/'scene.json').read_text());plans.append(dict(scene=name,blend=str(s/'scene.blend'),objects=m['objects'],output=str(s/'alias')))
    p=OUT/'alias-plan.json';write(p,plans);run([BLENDER,'-b','-t','12','--python','scripts/route3_step2_alias_blender.py','--','--plan',str(p)],'G2-alias',True)

def truth_score():
    from route3_ldr_u1r import model
    from route3_upper_bound_evaluate import score
    torch.set_num_threads(4);net=model();results=[]
    for name in C['scenes']:
        s=OUT/name;meta=json.loads((s/'alias/manifest.json').read_text());objects={r['id']:r for r in meta['objects']};geom=np.load(s/'scene-geometry.npz');stats=[];period_pass=True
        for f in FRAMES:
            P=read(s/'inputs'/f'f{f:03d}-Position-{f:04d}.exr');ids=read(s/'inputs'/f'f{f:03d}-IndexOB-{f:04d}.exr')[...,0].round().astype(int);dy,dx=np.gradient(P.astype(np.float64),axis=(0,1));normal=np.cross(dx,dy);normal/=np.maximum(np.linalg.norm(normal,axis=-1,keepdims=True),1e-12);v=geom['camera_c2w_opencv'][f-1,:3,3]-P;v/=np.maximum(np.linalg.norm(v,axis=-1,keepdims=True),1e-12);grazing=abs((normal*v).sum(-1))<np.cos(np.deg2rad(75))
            for oid,obj in objects.items():
                p=np.full(ids.shape,np.inf)
                for pat in obj['patterns']:
                    if 'error' in pat:raise RuntimeError('Unsupported G2 texture coordinate '+str(pat))
                    G=np.array(pat['world_phase_gradient_cycles_m']);frequency=np.sqrt((dx@G.T)**2+(dy@G.T)**2).max(-1);p=np.minimum(p,1/np.maximum(frequency,1e-12))
                interior=cv2.erode((ids==oid).astype(np.uint8),np.ones((3,3),np.uint8))>0;valid=interior&~grazing&np.isfinite(p)&(p<1e10);n=int(valid.sum());q=float(np.quantile(p[valid],.01)) if n else None;ok=q is None or q>=8;period_pass&=ok;stats.append(dict(frame=f,object_id=oid,object_name=obj['name'],valid_pixels=n,grazing_pixels=int((interior&grazing).sum()),period_p01_px=q,period_p50_px=float(np.median(p[valid])) if n else None,passed=ok))
        aliasjobs={r['label']:r for r in meta['rows']};base=read(aliasjobs['center']['path']);phase=read(aliasjobs['shift']['path']);noise=read(aliasjobs['noise']['path']);ids=read(aliasjobs['center']['passes']['IndexOB'])[...,0].round().astype(int);shift=cv2.warpAffine(base,np.array([[1,0,-.5],[0,1,0]],np.float32),(512,512),flags=cv2.INTER_LANCZOS4,borderMode=cv2.BORDER_REFLECT101);aliasrows=[];aliaspass=True
        for oid,obj in objects.items():
            valid=cv2.erode((ids==oid).astype(np.uint8),np.ones((11,11),np.uint8))>0;n=int(valid.sum())
            if n<25:aliasrows.append(dict(object_name=obj['name'],pixels=n,insufficient_interior=True));continue
            e=float(np.mean((display(phase)[valid]-display(shift)[valid])**2));nr=float(np.mean((display(noise)[valid]-display(base)[valid])**2));excess=max(e-nr,0)**.5;ok=excess<=C['G2_truth_checks']['alias_excess_RMSE_limit'];aliaspass&=ok;aliasrows.append(dict(object_name=obj['name'],pixels=n,phase_PSNR_dB=float(-10*np.log10(max(e,1e-12))),noise_PSNR_dB=float(-10*np.log10(max(nr,1e-12))),phase_to_noise_SSE_ratio=e/max(nr,1e-12),excess_RMSE=excess,passed=ok))
        samples=[];converged=True;regions=dict(full_image=np.ones((512,512),bool))
        for focus in [1.4,5.]:
            target=display(read(s/'dof'/f'f030-N1.4-F{focus:g}.exr'))
            for tag in ['independent','lower-spp']:
                result=score(display(read(s/'noise'/f'F{focus:g}-{tag}.exr')),target,regions,net)['full_image'];ok=result['psnr']>=45 and result['patch_lpips']<=.002;converged&=ok;samples.append(dict(focus_m=focus,variant=tag,passed=ok,**result))
        row=dict(scene=name,periods=stats,alias=aliasrows,sampling=samples,period_pass=bool(period_pass),alias_pass=bool(aliaspass),convergence_pass=bool(converged),passed=bool(period_pass and aliaspass and converged));results.append(row);m=json.loads((s/'scene.json').read_text());m.update(texture_period_p01_px_min=min(x['period_p01_px'] for x in stats if x['period_p01_px'] is not None),texture_periods=stats,texture_truth_checks_passed=row['passed']);write(s/'scene.json',m);write(s/'truth-check.json',row);print('G2_TRUTH',name,row['passed'],m['texture_period_p01_px_min'],flush=True)
    write(OUT/'truth-checks.json',dict(rows=results,all_passed=all(r['passed'] for r in results),method_results_not_read=True));write(REC/(PREFIX+'-truth.json'),dict(rows=results,all_passed=all(r['passed'] for r in results),checks=C['G2_truth_checks']))
    if not all(r['passed'] for r in results):write(OUT/'STOP-truth.json',dict(reason='G2 truth-side checks failed;do not run method inference;do not discard scenes'))

def methods():
    if not json.loads((OUT/'truth-checks.json').read_text())['all_passed']:raise RuntimeError('Truth checks must pass before methods')
    master=[]
    for name in C['scenes']:
        s=OUT/name
        for mode in ['center','aperture']:run([CPU,'scripts/route3_s2a_geometry.py','--scene',str(s),'--mode',mode],name+'-geometry-'+mode)
        req=json.loads((s/'geometry/required-exclusions.json').read_text())['frames'];names={o['id']:o['name'] for o in json.loads((s/'scene.json').read_text())['objects']};peels=[]
        for frame,sets in req.items():
            for ids in sets:
                key=f'f{int(frame):03d}-exclude-'+ '-'.join(map(str,ids));peels.append(dict(scene=name,id=name+'-'+key,blend=str(s/'scene.blend'),frame=int(frame),spp=256,size=512,seed=20261112,dof=False,hidden_object_ids=ids,hide_names=[names[i] for i in ids],output=str(s/'references'/(key+'.exr'))))
        master+=jobs(s,'formal-peels',peels);write(OUT/'peel-jobs-results.json',master);run([CPU,'scripts/route3_s2a_reference_geometry.py','--scene',str(s)],name+'-reference-geometry');run([CPU,'scripts/route3_pilot_oracle.py','--scene',str(s)],name+'-oracle')
        for mode in ['inference','calibrate']:run([PY,'scripts/route3_s2a_models.py','--scene',name,'--mode',mode],name+'-'+mode,True)
        run([PY,'scripts/route3_s2a_candidates.py','--scene',name,'--mode','prepare'],name+'-candidates')
        if not(s/'fill-record.json').exists():run([LAMA,'scripts/route3_dual_layer_fill.py','--manifest',str(s/'fill-inputs.json'),'--output',str(s/'fill'),'--record',str(s/'fill-record.json')],name+'-LaMa',True)
        for mode in ['anchor_visible','render']:run([PY,'scripts/route3_s2a_baselines.py','--scene',name,'--mode',mode],name+'-'+mode,True)
        print('G2_METHODS_COMPLETE',name,flush=True)

def score_methods():
    from route3_ldr_u1r import model
    from route3_upper_bound_evaluate import score,pool
    from route3_b2_evaluate import gt_supported_crop
    import route3_s2a_evaluate as visualization
    visualization.OUT=OUT
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    torch.set_num_threads(4);net=model();cells=[];percase={};quality=[];figures=[]
    for name in C['scenes']:
        s=OUT/name;entries=json.loads((s/'prepared.json').read_text())['rows'];percase[name]={};anchors=json.loads((s/'visible/summary.json').read_text())['rows'];zero=[x['rows'][-1]['input_PSNR_dB'] for x in anchors];inputPSNR=float(np.median(zero))
        for entry in entries:
          f=entry['frame'];g=np.load(s/'geometry'/f'f{f:03d}-center.npz');z=np.where(g['center_visible'],g['depth_axial_m'],1e4)
          for case in entry['cases']:
            cid=case['id'];a=np.load(s/'geometry'/(cid+'-K16.npz'));band,_,coc=depth_band(z,g['K'][0,0],case['nominal_focus_m'],.05/(2*case['fstop']));regions=dict(full_image=np.ones((512,512),bool),in_focus=g['center_visible']&(coc<=1),band=band&~a['residual_pixels'],recoverable=band&a['mask_recoverable_total']);target=display(read(s/'dof'/(cid+'.exr')));percase[name][cid]={}
            for method in ['A_color','R1_R2','GTrec']:
                pred=np.load(s/'predictions'/(cid+'-'+method+'.npz'))['prediction_srgb'];m=score(pred,target,regions,net);percase[name][cid][method]=m
                for region,v in m.items():cells.append(dict(scene=name,id=cid,frame=f,focus_m=case['nominal_focus_m'],fstop=case['fstop'],method=method,region=region,**v))
            if f==30 and case['fstop']==1.4:
                ims={m:np.load(s/'predictions'/(cid+'-'+m+'.npz'))['prediction_srgb'] for m in ['A_color','R1_R2','GTrec']};figures.append(visualization.figure(name,case,target,regions,ims))
        infocus={}
        for focus in [1.4,5.]:
            chosen=[r for r in cells if r['scene']==name and r['focus_m']==focus and r['fstop']==1.4 and r['method']=='A_color' and r['region']=='in_focus'];infocus[str(focus)]=pool(chosen)['psnr']
        q=C['G2_method_diagnostic_reference'];passed=inputPSNR>=q['zeroaperture_median_full_PSNR_min'] and infocus['1.4']>=q['near_infocus_pooled_PSNR_min'] and infocus['5.0']>=q['far_infocus_pooled_PSNR_min'];cal=json.loads((s/'models/calibration/summary.json').read_text())['rows'];quality.append(dict(scene=name,zeroaperture_perframe_input_PSNR=zero,zeroaperture_median_input_PSNR=inputPSNR,N1p4_pooled_infocus=infocus,generator_level_reference_pass=bool(passed),SHARP_calibration_accepted_frames=sum(r['models']['SHARP']['calibration_applied'] for r in cal),scene_not_rejected_on_method_score=True))
    groups={}
    for r in cells:groups.setdefault((r['scene'],r['focus_m'],r['fstop'],r['method'],r['region']),[]).append(r)
    pooled=[dict(scene=k[0],focus_m=k[1],fstop=k[2],method=k[3],region=k[4],**pool(v)) for k,v in groups.items()];result=dict(cells=cells,pooled=pooled,per_case=percase,generator_quality=quality,all_generator_references_passed=all(x['generator_level_reference_pass'] for x in quality),reference='Cycles',LDR='not rendered for G2;available auxiliary D references remain separate',figures=figures,method_scores_not_used_for_scene_selection=True,diagnostic_reference=C['G2_method_diagnostic_reference']);write(REC/(PREFIX+'-methods.json'),result);write(OUT/'method-checks.json',result)
    for tag,rows in [('cells',cells),('pooled',pooled)]:
        with (REC/(PREFIX+'-'+tag+'.csv')).open('w') as h:w=csv.DictWriter(h,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    if not result['all_generator_references_passed']:write(OUT/'STOP-method.json',dict(reason='G2 zero-aperture or infocus remains clearly below controls;stop generator expansion,do not discard either pilot',quality=quality))
    print('G2_METHOD_CHECK',quality,flush=True)

def zero_regions():
    """Complete G2 zero-aperture/infocus report from frozen attributes only."""
    from route3_s2a_baselines import scene
    from route3_sharp_s1 import raster
    from route3_ldr_u1r import model
    from route3_upper_bound_evaluate import score
    subprocess.run(['nvidia-smi'],check=True);torch.set_num_threads(4);net=model();rows=[]
    def depth_stats(estimate,gt,mask):
        valid=mask&np.isfinite(estimate)&(estimate>0)&(gt>0)
        return dict(pixels=int(valid.sum()),AbsRel=float(np.mean(abs(estimate[valid]-gt[valid])/gt[valid])),median_ratio=float(np.median(estimate[valid]/gt[valid]))) if valid.any() else dict(pixels=0)
    for name in C['scenes']:
        s=OUT/name;fgids=np.load(s/'scene-geometry.npz')['foreground_object_ids']
        for e in json.loads((s/'prepared.json').read_text())['rows']:
            f=e['frame'];d=np.load(e['input_npz']);g=np.load(s/'geometry'/f'f{f:03d}-center.npz');z=g['depth_axial_m'];fg=cv2.erode(np.isin(g['object_id'],fgids).astype(np.uint8),np.ones((7,7),np.uint8))>0;boundary=np.zeros(z.shape,bool)
            for oid in np.unique(g['object_id']):boundary|=(g['object_id']==oid)&~(cv2.erode((g['object_id']==oid).astype(np.uint8),np.ones((7,7),np.uint8))>0)
            cloud=scene(s/'visible'/f'f{f:03d}.pt');K=torch.tensor(d['K'],device='cuda');fill=torch.tensor(np.load(e['fill'])['fill_rgb_linear'],device='cuda')
            with torch.no_grad():rgb,depth,alpha=raster(cloud,K)
            own=depth.cpu().numpy();a=alpha.cpu().numpy();pred=display((rgb+(1-alpha[...,None])*fill).cpu().numpy());masks=dict(full_image=np.ones(z.shape,bool),boundary=boundary);cocs=[]
            for focus in [1.4,5.]:
                case=next(c for c in e['cases'] if c['fstop']==1.4 and c['nominal_focus_m']==focus);GTcoc=g['K'][0,0]*.05/2.8*abs(1/np.maximum(z,1e-6)-1/focus);predcoc=g['K'][0,0]*.05/2.8*abs(1/np.maximum(own,1e-6)-1/case['focus_m']);mask=g['center_visible']&(GTcoc<=1);masks['in_focus_'+str(focus)]=mask;valid=mask&fg&(a>=.1);cocs.append(dict(focus_m=focus,predicted_focus_m=case['focus_m'],GT_infocus_foreground_pixels=int(valid.sum()),GT_CoC_median_px=float(np.median(GTcoc[valid])) if valid.any() else None,full_visible_CoC_median_px=float(np.median(predcoc[valid])) if valid.any() else None,fraction_full_visible_CoC_gt1=float((predcoc[valid]>1).mean()) if valid.any() else None))
            dest=s/'zero-evaluation';dest.mkdir(exist_ok=True);path=dest/f'f{f:03d}.npz'
            if not path.exists():np.savez_compressed(path,prediction_srgb=pred,visible_expected_depth=own,alpha=a)
            rows.append(dict(scene=name,frame=f,path=str(path),input_reconstruction=score(pred,d['rgb_srgb'],masks,net),full_visible_depth_foreground=depth_stats(own,z,fg&(a>=.1)),first_layer_depth_foreground=depth_stats(d['depth'],z,fg),CoC_in_GT_infocus=cocs,no_parameter_updated=True,GT_used_for_report_only=True))
    write(REC/(PREFIX+'-zero-regions.json'),dict(rows=rows,scope='required G2 zero-aperture/infocus report;cached attributes;no reoptimization or new method'));print('G2_ZERO_REPORT',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['truth-render','truth-score','methods','score-methods','zero-regions'],required=True);a=p.parse_args();{'truth-render':truth_render,'truth-score':truth_score,'methods':methods,'score-methods':score_methods,'zero-regions':zero_regions}[a.phase]()
