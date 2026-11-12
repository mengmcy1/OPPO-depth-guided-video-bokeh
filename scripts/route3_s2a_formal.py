"""Formal batch preparation after input-only screening, before neural training.

Split only by prespecified GT-rec benefit and scene seed order. No learned
model results participate. All 512 master inputs/labels remain immutable.
"""
import argparse,json,os,shutil,sys,time
from pathlib import Path
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from route3_s2a_screen import run,OUT,RECORDS,PY,CPU,BLENDER,FRAMES,write
CONFIG=ROOT/'configs/route3-network-s2a-2026-11-12.json'

def jobs_run(s,name,jobs):
    p=s/(name+'-jobs.json')
    if p.exists():assert json.loads(p.read_text())==jobs
    else:write(p,jobs)
    t=run([BLENDER,'-b','-t','12','--python','scripts/route3_s2a_blender.py','--','--jobs',str(p)],s.name+'-'+name,True)
    rows=json.loads(p.with_name(p.stem+'-results.json').read_text());assert len(rows)==len(jobs);return t,rows

def render_scene(s):
    result=s/'formal-data.json'
    if result.exists():return json.loads(result.read_text())
    t=time.monotonic();timings={};meta=json.loads((s/'scene.json').read_text());common=dict(scene=s.name,blend=str(s/'scene.blend'),size=512,seed=20261112)
    def job(tag,f,spp,sub,stem,**kw):return dict(**common,id=s.name+'-'+tag,frame=f,spp=spp,output=str(s/sub/(stem+'.exr')),**kw)
    native=[job(f'native-{f}',f,1,'input-passes-v2',f'f{f:03d}',dof=False,data_passes=True,cryptomatte=True) for f in FRAMES];timings['native_data'],_=jobs_run(s,'native-data',native)
    for mode in ['center','aperture','training_focus']:timings['geometry-'+mode]=run([CPU,'scripts/route3_s2a_geometry.py','--scene',str(s),'--mode',mode],s.name+'-formal-geometry-'+mode)
    req=json.loads((s/'geometry/required-exclusions.json').read_text())['frames']
    for row in json.loads((s/'geometry/training_focus-summary.json').read_text()):
        frame=str(row['frame']);req[frame]=[list(x) for x in sorted({tuple(x) for x in req[frame]}|{tuple(x) for x in row['required_exclusion_sets']})]
    # The CPU oracle helper derives missing reference geometry itself below.
    names={x['id']:x['name'] for x in meta['objects']};peels=[]
    for frame,sets in req.items():
        for ids in sets:
            f=int(frame);key=f'f{f:03d}-exclude-'+ '-'.join(map(str,ids));peels.append(job('peel-'+key,f,256,'references',key,dof=False,hide_names=[names[i] for i in ids],hidden_object_ids=ids))
    timings['peels'],peelrows=jobs_run(s,'formal-peels',peels)
    master=OUT/'peel-jobs-results.json';old=json.loads(master.read_text()) if master.exists() else [];write(master,[x for x in old if x['scene']!=s.name]+peelrows)
    timings['reference_geometry']=run([CPU,'scripts/route3_s2a_reference_geometry.py','--scene',str(s)],s.name+'-formal-reference-geometry')
    timings['GT_parents']=run([CPU,'scripts/route3_pilot_oracle.py','--scene',str(s)],s.name+'-formal-oracle')
    gt=[]
    for f in [1,30,60]:
        for focus in [1.4,5.]:
            for stop in [1.4,2.]:
                key=f'f{f:03d}-N{stop:g}-F{focus:g}';gt.append(job('dof-'+key,f,1024,'dof',key,dof=True,fstop=stop,focus=focus))
    for f in range(27,35):
        gt.append(job(f'continuous-{f}',f,1024,'continuous',f'f{f:03d}',dof=True,fstop=1.4,focus=5.));focus=1/(1/1.4+(f-27)/7*(1/5-1/1.4));gt.append(job(f'rack-{f}',f,1024,'rack',f'f{f:03d}',dof=True,fstop=1.4,focus=focus))
    timings['DOF'],_=jobs_run(s,'formal-dof',gt)
    r=dict(scene=s.name,timings_seconds=timings,total_seconds=time.monotonic()-t,pinhole_spp=256,pinhole_reused_from_screening=True,peel_spp=256,DOF_spp=1024,formal_K16_native_passes_omitted=True,disk_bytes=sum(p.stat().st_size for p in s.rglob('*') if p.is_file()));write(result,r);return r

def frontend_labels(s):
    if (s/'formal-interface.json').exists():return
    stages=['prepare']
    for mode in stages:run([PY,'scripts/route3_s2a_candidates.py','--scene',s.name,'--mode',mode],s.name+'-candidates-'+mode)
    # The exact same mask/depth settings generated this existing fill.
    for e in json.loads((s/'prepared.json').read_text())['rows']:
        d=np.load(e['input_npz']);screen=np.load(s/'screen-inputs'/f'f{e["frame"]:03d}.npz')
        for k in ['fill_mask','background_depth','rgb_srgb','depth']:assert np.array_equal(d[k],screen[k]),k
    if not(s/'fill').exists():(s/'fill').symlink_to(s/'screen-fill',target_is_directory=True)
    run([PY,'scripts/route3_s2a_memory.py','--scene',s.name],s.name+'-memory')
    for mode in ['labels','training_labels','appearance','placement_support','placement_safety']:run([PY,'scripts/route3_s2a_candidates.py','--scene',s.name,'--mode',mode],s.name+'-candidates-'+mode)
    run([PY,'scripts/route3_s2a_baselines.py','--scene',s.name,'--mode','anchor_visible'],s.name+'-anchor',True)
    run([PY,'scripts/route3_s2a_baselines.py','--scene',s.name,'--mode','render'],s.name+'-baselines',True)
    write(s/'formal-interface.json',dict(completed=True,placement_validity=str(s/'labels'),memory_verification=str(s/'memory/verification.npz'),calibration_flags=str(s/'models/calibration/summary.json'),new_neural_network_trained=False))

def gain_check(s):
    from route3_s2a_proxy import display,read
    from route3_s2a_evaluate import masks
    gains=[];allrows=[]
    for entry in json.loads((s/'prepared.json').read_text())['rows']:
        for case in entry['cases']:
            cid=case['id'];regions=masks(s,case);truth=display(read(s/'dof'/(cid+'.exr')));visible=np.load(s/'predictions'/(cid+'-A_color.npz'))['prediction_srgb'];oracle=np.load(s/'predictions'/(cid+'-GTrec.npz'))['prediction_srgb']
            for reg in ['recoverable','band','in_focus','full_image']:
                mask=regions[reg];n=int(mask.sum());v=float(np.mean((visible[mask].astype(np.float64)-truth[mask])**2)) if n else None;o=float(np.mean((oracle[mask].astype(np.float64)-truth[mask])**2)) if n else None;gain=float(10*np.log10(max(v,1e-12)/max(o,1e-12))) if n else None
                row=dict(id=cid,frame=entry['frame'],focus_m=case['nominal_focus_m'],fstop=case['fstop'],region=reg,pixels=n,Acolor_PSNR_dB=float(-10*np.log10(max(v,1e-12))) if n else None,GTrec_PSNR_dB=float(-10*np.log10(max(o,1e-12))) if n else None,GTrec_minus_Acolor_dB=gain,Acolor_SSE=None if n==0 else v*3*n,GTrec_SSE=None if n==0 else o*3*n);allrows.append(row)
                if case['nominal_focus_m']==5. and case['fstop']==1.4 and reg=='recoverable':gains.append(dict(frame=entry['frame'],pixels=n,GTrec_minus_Acolor_dB=gain,passed=bool(n>=1000 and gain>=1.)))
    r=dict(scene=s.name,seed=json.loads((s/'scene.json').read_text())['seed'],rows=gains,all_case_regions=allrows,passed=sum(x['passed'] for x in gains)>=2,selection_uses_only_GTrec_and_Acolor=True,model_outputs_not_used=True,Oracle_reference_not_perfect_optical_upper_bound=True);write(s/'actual-gain-check.json',r);return r

def main():
    os.environ['ROUTE3_PILOT_CONFIG']=str(CONFIG);cv2.setNumThreads(1);screen=json.loads((OUT/'screening.json').read_text());assert screen.get('screen_complete');selected=[r for r in screen['rows'] if r['passed']][:16];names=[r['scene'] for r in selected];checks=[];cost=[]
    if (OUT/'split.json').exists():raise FileExistsError('Formal split already fixed;do not overwrite')
    # Conservative remaining estimate, including label caches/checkpoints/LDR.
    estimate=len(names)*6+4*5+15;free=shutil.disk_usage(ROOT).free/2**30
    write(OUT/'formal-disk-budget.json',dict(free_GiB=free,conservative_additional_GiB=estimate,projected_free_GiB=free-estimate,stop_if_projected_below_GiB=60))
    if free-estimate<60:write(OUT/'STOP.json',dict(reason='projected formal free disk below60GiB'));return
    c=json.loads(CONFIG.read_text());c['scenes']=names;c['seeds']=[json.loads((OUT/name/'scene.json').read_text())['seed'] for name in names];write(CONFIG,c)
    for name in names:
        s=OUT/name;cost.append(render_scene(s));frontend_labels(s);checks.append(gain_check(s));cost[-1]['disk_bytes']=sum(p.stat().st_size for p in s.rglob('*') if p.is_file())
        measured_extra=[max(0,c['disk_bytes']-next(x['disk_bytes'] for x in selected if x['scene']==c['scene']))/2**30 for c in cost]
        remaining=len(names)-len(cost);current_free=shutil.disk_usage(ROOT).free/2**30;projected=current_free-(max(measured_extra)*remaining+20+15)
        write(OUT/'formal-progress.json',dict(cost=cost,actual_gain_checks=checks,current_free_GiB=current_free,projected_free_GiB=projected))
        if projected<60:write(OUT/'STOP.json',dict(reason='measured formal cost projects free disk below60GiB',projected_free_GiB=projected));return
        print('S2A_FORMAL',name,checks[-1]['passed'],flush=True)
    qualified=sorted([c for c in checks if c['passed']],key=lambda x:x['seed'])
    if len(qualified)<4:write(OUT/'STOP.json',dict(reason='fewer than4 actual-gain-qualified scenes;cannot freeze requested heldout split',actual_gain_checks=checks));return
    held=[r['scene'] for r in qualified[:4]];train=[n for n in names if n not in held];split=dict(train=train,heldout=held,excluded=['B2-high','pilot-thin','pilot-multilayer'],selection='first4 seeds passing >=1dB farfocusf1.4 >=2/3 reps;before neural training;no learned model metrics used',actual_gain_checks=checks,training_started=False);assert len(train)==12;write(OUT/'split.json',split);write(RECORDS/'route3-network-s2a-2026-11-12-split.json',split)

if __name__=='__main__':main()
