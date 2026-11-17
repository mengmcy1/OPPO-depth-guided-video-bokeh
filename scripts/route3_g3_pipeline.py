"""Appendix13 G3: prespecified independent scenes, truth-only data checks."""
import argparse,json,os,sys,shutil,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];CONFIG=ROOT/'configs/route3-g3-s2a-2026-11-17.json';os.environ['ROUTE3_PILOT_CONFIG']=str(CONFIG);os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1');sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import cv2,numpy as np
C=json.loads(CONFIG.read_text());OUT=Path(C['output']);LOG=ROOT/'logs/route3-g3-s2a-2026-11-17';REC=ROOT/'03_实验记录/records';PREFIX=C['record_prefix'];BLENDER=C['cycles']['binary'];PY='/root/autodl-tmp/envs/route3/bin/python';CPU='/root/autodl-tmp/envs/dofgs/bin/python';FRAMES=C['input_frames']

def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def run(cmd,name,gpu=False):
 free=shutil.disk_usage(OUT).free/2**30
 if free<60:write(OUT/'STOP.json',dict(reason='disk below60GiB',free_GiB=free));raise RuntimeError('STOP disk below60GiB')
 if gpu:
  assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0';subprocess.run(['nvidia-smi'],stdout=subprocess.DEVNULL,check=True)
 start=time.monotonic()
 with (LOG/(name+'.log')).open('a') as f:subprocess.run(cmd,cwd=ROOT,env=os.environ.copy(),stdout=f,stderr=subprocess.STDOUT,check=True)
 return time.monotonic()-start

def jobs(s,tag,items):
 p=s/(tag+'-jobs.json')
 if p.exists():assert json.loads(p.read_text())==items
 else:write(p,items)
 t=run([BLENDER,'-b','-t','12','--python','scripts/route3_metric_blender.py','--','--jobs',str(p)],s.name+'-'+tag,True)
 return t,json.loads(p.with_name(p.stem+'-results.json').read_text())

def truth_scene(name,index):
 s=OUT/name;s.mkdir(exist_ok=True);timings={};start=time.monotonic()
 if not (s/'scene.blend').exists():timings['build']=run([BLENDER,'-b','-t','12','--python','scripts/route3_metric_blender.py','--','--build',str(s),'--seed',str(C['seeds'][index-1]),'--index',str(index)],name+'-build')
 common=dict(scene=name,blend=str(s/'scene.blend'),size=512,seed=20261112,omit_reserved_passes=True)
 inputs=[dict(**common,id=name+f'-input-{f}',frame=f,spp=256,dof=False,data_passes=True,data_pass_names=['Position','Depth','IndexOB'],cryptomatte=False,output=str(s/'inputs'/f'f{f:03d}.exr')) for f in FRAMES]
 timings['pinhole'],_=jobs(s,'inputs',inputs)
 from route3_s2a_proxy import read,display
 for f in FRAMES:
  p=s/'inputs'/f'f{f:03d}.png'
  if not p.exists():cv2.imwrite(str(p),np.uint8(np.rint(display(read(s/'inputs'/f'f{f:03d}.exr'))[...,::-1]*255)))
 dof=[]
 for f in [1,30,60]:
  for focus in [1.4,5.]:
   for stop in [1.4,2.]:
    key=f'f{f:03d}-N{stop:g}-F{focus:g}';dof.append(dict(**common,id=name+'-'+key,frame=f,spp=1024,dof=True,focus=focus,fstop=stop,output=str(s/'dof'/(key+'.exr'))))
 for focus in [1.4,5.]:
  for tag,spp,seed in [('independent',1024,2026111202),('lower-spp',512,20261112)]:dof.append(dict(scene=name,blend=str(s/'scene.blend'),size=512,seed=seed,omit_reserved_passes=True,id=name+f'-noise-{focus:g}-{tag}',frame=30,spp=spp,dof=True,focus=focus,fstop=1.4,output=str(s/'noise'/f'F{focus:g}-{tag}.exr')))
 timings['dof'],_=jobs(s,'dof',dof)
 meta=json.loads((s/'scene.json').read_text());plan=s/'alias-plan.json';write(plan,[dict(scene=name,blend=str(s/'scene.blend'),objects=meta['objects'],output=str(s/'alias'))]);timings['alias']=run([BLENDER,'-b','-t','12','--python','scripts/route3_step2_alias_blender.py','--','--plan',str(plan)],name+'-alias',True)
 import route3_metric_pilot as checker
 checker.C=dict(C,scenes=[name]);checker.OUT=OUT;checker.PREFIX=PREFIX+'-'+name;checker.LOG=LOG;os.environ['ROUTE3_PILOT_CONFIG']=str(CONFIG)
 checker.truth_score();proof=json.loads((s/'truth-check.json').read_text());zero=read(s/'alias/center.exr');original=read(s/'inputs/f030.exr');err=float(np.mean((display(zero).astype(np.float64)-display(original))**2));zpsnr=float(-10*np.log10(max(err,1e-30)));proof['zero_aperture_repeat_PSNR']=zpsnr;proof['zero_aperture_repeat_pass']=zpsnr>=50;proof['zero_aperture_repeat_definition']='same scene/camera/seed/256spp,no method;independent render pipeline input vs alias center';proof['passed']=proof['passed'] and proof['zero_aperture_repeat_pass'];write(s/'truth-check.json',proof)
 cost=dict(scene=name,seconds=time.monotonic()-start,timings=timings,disk_GiB=sum(p.stat().st_size for p in s.rglob('*') if p.is_file())/2**30,passed=proof['passed'],zero_PSNR=zpsnr,period_pass=proof['period_pass'],alias_pass=proof['alias_pass'],convergence_pass=proof['convergence_pass'],native_reserved_AOV_or_cryptomatte=False,scene_split_fixed_before_methods=True);write(s/'truth-ready.json',cost);print('G3_TRUTH',name,json.dumps(cost),flush=True);return cost

def truth():
 selection=ROOT/'outputs/route3-order-fallback-2026-11-17-01/selection.json'
 assert selection.exists(),'T1 selection must be fixed first'
 budget=C['disk_budget'];projected=shutil.disk_usage(OUT).free/2**30-budget['conservative_master_GiB']-budget['estimated_geometry_labels_and_augmentation_GiB']-budget['training_and_evaluation_reserve_GiB'];write(OUT/'disk-budget.json',dict(**budget,projected_free_GiB=projected))
 if projected<60:write(OUT/'STOP.json',dict(reason='projected disk below60GiB'));return
 rows=[]
 for index,name in enumerate(C['scenes'],1):
  p=OUT/name/'truth-ready.json';rows.append(json.loads(p.read_text()) if p.exists() else truth_scene(name,index));write(OUT/'truth-progress.json',dict(rows=rows,selection_uses_no_method=True))
  if sum(not r['passed'] for r in rows)>=2:
   result=dict(reason='T2 truth-side data validity failed for multiple scenes',rows=rows,failed=[r['scene'] for r in rows if not r['passed']],no_scene_replaced_or_deleted=True);write(OUT/'STOP.json',result);write(REC/(PREFIX+'-STOP.json'),result);return
 write(OUT/'truth-ready.json',dict(rows=rows,all_passed=all(r['passed'] for r in rows),split=C['split'],no_method_selection=True))
 write(REC/(PREFIX+'-truth.json'),dict(rows=rows,all_passed=all(r['passed'] for r in rows),split=C['split'],no_method_selection=True))

def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['truth']);a=p.parse_args();globals()[a.mode]()
if __name__=='__main__':main()
