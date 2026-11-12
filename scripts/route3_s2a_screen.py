"""Sequential input-only scene screening before any formal DOF or training.

All commands/logs and quality fallbacks are retained. First-30 pass-count and
60-GiB disk stop are applied without changing thresholds.
"""
import argparse,json,os,shutil,subprocess,sys,time
from pathlib import Path
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from route3_s2a_proxy import proxy,write,OUT,RECORDS
BLENDER='/root/autodl-tmp/tools/blender-4.5.3-linux-x64/blender';PY='/root/autodl-tmp/envs/route3/bin/python';CPU='/root/autodl-tmp/envs/dofgs/bin/python';LAMA='/root/autodl-tmp/envs/conda-magicbokeh/bin/python';LOG=ROOT/'logs/route3-network-s2a-2026-11-12'
FRAMES=[1,6,11,16,21,26,30,35,40,45,50,55,60]

def run(cmd,name,gpu=False):
    if gpu:subprocess.run(['nvidia-smi'],stdout=subprocess.DEVNULL,check=True)
    if shutil.disk_usage(ROOT).free/2**30<60:raise RuntimeError('STOP_DISK_BELOW_60_GiB')
    p=LOG/(name+'.log');t=time.monotonic()
    with p.open('a') as f:
        f.write('\n'+json.dumps(cmd)+'\n');f.flush();subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    return time.monotonic()-t

def candidate(index):
    seed=121301+index;family=['thin','broad','multilayer'][index%3];s=OUT/f'scene-{seed}-{family}';timings={};start=time.monotonic()
    if (s/'screen-result.json').exists():return json.loads((s/'screen-result.json').read_text())
    if not(s/'scene.blend').exists():timings['build']=run([BLENDER,'-b','-t','12','--python','scripts/route3_s2a_blender.py','--','--build',str(s),'--seed',str(seed),'--index',str(index)],s.name+'-build');assert(s/'scene.blend').exists()
    common=dict(scene=s.name,blend=str(s/'scene.blend'),size=512,spp=256,seed=20261112,dof=False)
    jobs=[dict(**common,id=s.name+f'-input-{f}',frame=f,output=str(s/'inputs'/f'f{f:03d}.exr'),data_passes=True,cryptomatte=True) for f in FRAMES];jp=s/'screen-input-jobs.json'
    if not jp.exists():write(jp,jobs)
    timings['pinhole']=run([BLENDER,'-b','-t','12','--python','scripts/route3_s2a_blender.py','--','--jobs',str(jp)],s.name+'-inputs',True)
    assert len(json.loads(jp.with_name(jp.stem+'-results.json').read_text()))==13
    from route3_s2a_proxy import read,display
    for f in FRAMES:
        p=s/'inputs'/f'f{f:03d}.png'
        if not p.exists():cv2.imwrite(str(p),np.uint8(np.rint(display(read(s/'inputs'/f'f{f:03d}.exr').clip(0,1))[...,::-1]*255)))
    timings['mesh']=run([CPU,'scripts/route3_s2a_screen_geometry.py','--scene',str(s)],s.name+'-mesh')
    # A nominal focus group must have an actual point-selection region.
    focus_support=[]
    for f in [1,30,60]:
        g=np.load(s/'geometry'/f'f{f:03d}-center.npz');z=g['depth_axial_m']
        for focus in [1.4,5.]:focus_support.append(dict(frame=f,focus=focus,pixels=int(((z>0)&(g['K'][0,0]*.05/2.8*abs(1/np.maximum(z,1e-6)-1/focus)<=1)).sum())))
    if any(x['pixels']==0 for x in focus_support):
        r=dict(scene=s.name,seed=seed,index=index,family=family,passed=False,far_frames_passed=0,rows=[],rejection_reason='empty GT in-focus click mask;invalid candidate',focus_support=focus_support,timings_seconds=timings,total_seconds=time.monotonic()-start,formal_DOF_rendered=False)
        write(s/'screen-result.json',r);return r
    timings['peeled_color']=run([BLENDER,'-b','-t','12','--python','scripts/route3_s2a_blender.py','--','--jobs',str(s/'screen-peel-jobs.json')],s.name+'-screen-peels',True)
    peel=json.loads((s/'screen-peel-jobs-results.json').read_text());assert len(peel)==len(json.loads((s/'screen-peel-jobs.json').read_text()))
    for mode in ['inference','calibrate']:
        timings[mode]=run([PY,'scripts/route3_s2a_models.py','--scene',s.name,'--mode',mode],s.name+'-'+mode,True)
    timings['fill_prepare']=run([PY,'scripts/route3_s2a_screen_prepare.py','--scene',str(s)],s.name+'-fill-prepare')
    if not(s/'screen-fill-record.json').exists():timings['LaMa']=run([LAMA,'scripts/route3_dual_layer_fill.py','--manifest',str(s/'screen-fill-inputs.json'),'--output',str(s/'screen-fill'),'--record',str(s/'screen-fill-record.json')],s.name+'-LaMa',True)
    rows=[]
    for e in json.loads((s/'screen-prepared.json').read_text())['rows']:
        for focus in [1.4,5.]:rows.append(proxy(s/'geometry'/f'f{e["frame"]:03d}-N1.4-F{focus:g}-K16.npz',dict(e,proxy_nominal_focus_m=focus),refs=[j for j in peel if j['frame']==e['frame']]))
    crit=json.loads((OUT/'screen-thresholds.json').read_text());n=0
    for r in rows:
        r['difficulty_D']=r['recoverable_image_fraction']*10**(-r['fill_hidden_PSNR_dB']/10) if r['fill_hidden_PSNR_dB'] is not None else 0.;r['frame_pass']=r['difficulty_D']>=crit['threshold_D'] and r['recoverable_image_fraction']>=crit['minimum_recoverable_image_fraction'] and r['color_support_fraction']>=crit['minimum_color_supported_ray_fraction']
        if r['focus_m']==5.:n+=r['frame_pass']
    cal=json.loads((s/'models/calibration/summary.json').read_text());quality={name:{'accepted':sum(r['models'][name]['calibration_applied'] for r in cal['rows']),'native_fallback':sum(r['models'][name]['fit_rejected_and_native_depth_retained'] for r in cal['rows'])} for name in ['SHARP','DA3']};result=dict(scene=s.name,seed=seed,index=index,family=family,passed=n>=2,far_frames_passed=n,rows=rows,calibration=quality,timings_seconds=timings,total_seconds=time.monotonic()-start,disk_bytes=sum(p.stat().st_size for p in s.rglob('*') if p.is_file()),formal_DOF_rendered=False)
    write(s/'screen-result.json',result);return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--one',type=int);a=p.parse_args();os.environ['ROUTE3_PILOT_CONFIG']=str(ROOT/'configs/route3-network-s2a-2026-11-12.json');os.environ['OPENBLAS_NUM_THREADS']='4';os.environ['OMP_NUM_THREADS']='8';cv2.setNumThreads(1)
    if a.one is not None:r=candidate(a.one);print(json.dumps(r),flush=True);return
    rows=[];i=0
    while sum(x['passed'] for x in rows)<16:
        r=candidate(i);rows.append(r);i+=1;state=dict(rows=rows,candidates=len(rows),passed=sum(x['passed'] for x in rows),screen_complete=False,thresholds=str(OUT/'screen-thresholds.json'),free_GiB=shutil.disk_usage(ROOT).free/2**30)
        if i==30 and state['passed']<12:
            state.update(status='STOP_FIRST30_FEWER_THAN12',stop_condition_triggered=True);write(OUT/'screening.json',state);write(RECORDS/'route3-network-s2a-2026-11-12-screening.json',state);print('S2A_STOP',state['status'],state['passed'],flush=True);return
        write(OUT/'screening.json',state);print('S2A_SCREEN',r['scene'],r['passed'],r['far_frames_passed'],'total',state['passed'],'of',i,flush=True)
    state.update(screen_complete=True,status='16_passed;formal rendering next');write(OUT/'screening.json',state);write(RECORDS/'route3-network-s2a-2026-11-12-screening.json',state)

if __name__=='__main__':main()
