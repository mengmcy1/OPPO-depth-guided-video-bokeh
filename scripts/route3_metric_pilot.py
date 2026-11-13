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

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['truth-render','truth-score','methods'],required=True);a=p.parse_args();{'truth-render':truth_render,'truth-score':truth_score,'methods':methods}[a.phase]()
