"""Appendix13 T1: select order and per-ray fallback once, then proceed to G3."""
import argparse,json,os,sys,time
from pathlib import Path
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR','1')
import cv2,numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from route3_explicit_layer import SCENES,PREV as ALIGN_ROOT,OUT as EXPLICIT,desc_for,load,tensor,guard as oldguard,aligned_hidden,raw_hidden
from route3_depth_interface import write,save,subject_mask
from route3_s2a_predict import visible_cloud
from route3_sharp_s1 import raster
from route3_local_replacement import reference
from route3_upper_bound_evaluate import score,pool
from route3_ldr_u1r import model
from src.layered_aperture import pixel_layer,merge_layers
from src.ordered_layers import order_hidden,ray_samples,render_fallbacks
from src.aperture_renderer import linear_to_srgb
PREFIX='route3-order-fallback-2026-11-17';OUT=ROOT/'outputs'/f'{PREFIX}-01';REC=ROOT/'03_实验记录/records';CONFIG=ROOT/'configs'/f'{PREFIX}.json'
METHODS=['A']+[f'D0_{f}' for f in ['F1','F2']]+[f'D1_{o}_{f}' for o in ['none','O1','O2'] for f in ['F1','F2']]+[f'DGT{i}_{f}' for i in [0,1] for f in ['F1','F2']]

def guard(gpu=False):
 import shutil,subprocess
 if shutil.disk_usage(OUT).free/2**30<60:raise RuntimeError('STOP disk below60GiB')
 if gpu:
  assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0';subprocess.run(['nvidia-smi'],check=True)
 torch.set_num_threads(4);cv2.setNumThreads(2);torch.backends.cuda.matmul.allow_tf32=False

def fd(name,f):return OUT/name/f'f{f:03d}'
def cd(name,f,c,rule):return fd(name,f)/rule/c['id']
def display(a):return np.where(a<=.0031308,12.92*a,1.055*np.maximum(a,0)**(1/2.4)-.055).clip(0,1).astype(np.float32)
def zero_parts(cloud,K):
 with torch.inference_mode():r,z,a=raster(cloud,K)
 return r,a

def verify():
 guard(True)
 from src.aperture_renderer import GaussianScene
 K=tensor(np.array([[100,0,256],[0,100,256],[0,0,1]],np.float32));means=tensor(np.array([[0,0,1],[.1,0,3]],np.float32));q=tensor(np.array([[1,0,0,0]]*2,np.float32));s=tensor(np.ones((2,3),np.float32)*.01);c=tensor(np.ones((2,3),np.float32)*.3);a=tensor(np.array([.5,.5],np.float32));cloud=GaussianScene(means,q,s,a,c,torch.ones(2,device='cuda',dtype=torch.long));v=torch.full((512,512),2.,device='cuda');bg=torch.full_like(v,5.)
 r1,a1=order_hidden(cloud,v,K,'O1',bg);r2,a2=order_hidden(cloud,v,K,'O2',bg)
 assert float(abs(r1.means[0,2]-2.06))<1e-6 and torch.equal(r1.means[1],means[1]);assert float(r2.means[0,2])==5.
 for key in ['quats','scales','opacities','colors','source_frames']:assert torch.equal(getattr(r1,key),getattr(cloud,key))
 r3,a3=order_hidden(cloud,v,K,'O2',v);assert len(r3.means)==1
 fill=torch.full((512,512,3),.2,device='cuda');samples=ray_samples(cloud,K,2.,.01,4,fill,bg);p=render_fallbacks(cloud,K,2.,.01,4,fill,bg,samples)
 from src.aperture_renderer import render_aperture
 native=render_aperture(cloud,K,torch.eye(4,device='cuda'),512,512,2.,.01,4,sample_backgrounds=samples,rasterize_mode='antialiased',eps2d=.0015)
 err=float((p['F2']-native['rgb_linear']).abs().max());assert err<2e-6,err
 write(OUT/'verification.json',dict(passed=True,F2_sample_backgrounds_native_max_error=err,O1=a1,O2=a2,O2_drop=a3,scales_unchanged=True))
 print('T1_VERIFY',err,flush=True)

def render():
 guard(True)
 for name in SCENES:
  desc=desc_for(name)
  for entry in desc['entries']:
   f=entry['frame'];d=load(entry['input_npz']);K=tensor(d['K']);data=load(EXPLICIT/name/f'f{f:03d}'/'explicit-depth.npz');depth=data['depth'];fill=tensor(load(EXPLICIT/'fill'/(name+f'-f{f:03d}')/'fill.npz')['fill_rgb_linear']);bg=tensor(load(EXPLICIT/name/f'f{f:03d}'/'fill-input.npz')['background_depth']);zbg=tensor(cv2.dilate(depth,np.ones((49,49),np.uint8)));A=visible_cloud(desc,entry);D=pixel_layer(tensor(d['rgb_linear']),tensor(depth),K,torch.eye(4,device='cuda'));g=load(desc['geometry']/f'f{f:03d}-center.npz');gtdepth=np.where(g['center_visible']&(g['depth_axial_m']>0),g['depth_axial_m'],1e4).astype(np.float32);DGT=pixel_layer(tensor(d['rgb_linear']),tensor(gtdepth),K,torch.eye(4,device='cuda'));parents=load(desc['parents'](f));align=load(ALIGN_ROOT/name/f'f{f:03d}'/'alignment.npz');HGT=raw_hidden(parents,K);ra,aa=zero_parts(A,K);Azero=ra+(1-aa[...,None])*fill
   for rule in ['P2','P1']:
    for case in entry['cases']:
     dest=cd(name,f,case,rule);dest.mkdir(parents=True,exist_ok=True)
     if (dest/'render.json').exists():continue
     guard();start=time.monotonic();old=EXPLICIT/name/f'f{f:03d}'/rule/case['id'];meta=json.loads((old/'render.json').read_text());focus=meta['focus_m'];band=load(old/'masks.npz')['band'];h,selected=aligned_hidden(parents,align,band,K);ordered={};audits={}
     for o in ['none','O1','O2']:ordered[o],audits[o]=order_hidden(h,tensor(depth),K,o,zbg)
     mask=subject_mask(g,case['nominal_focus_m']);GTfocus=float(np.median(gtdepth[mask])) if rule=='P2' else float(case['nominal_focus_m'])
     with torch.inference_mode():As=ray_samples(A,K,focus,.05/(2*case['fstop']),64,fill,bg)
     if not(dest/'A.npz').exists():save(dest/'A.npz',rgb_linear=As.mean(0).cpu().numpy(),prediction_srgb=display(As.mean(0).cpu().numpy()),zero_srgb=display(Azero.cpu().numpy()))
     timings=[];Agt=None
     for label,visible,hidden,ffocus in [('D0',D,None,focus)]+[(f'D1_{o}',D,ordered[o],focus) for o in ['none','O1','O2']]+[('DGT0',DGT,None,GTfocus),('DGT1',DGT,HGT,GTfocus)]:
      if all((dest/(label+'_'+fall+'.npz')).exists() for fall in ['F1','F2']):continue
      torch.cuda.reset_peak_memory_stats();t=time.monotonic();cloud=merge_layers(visible,hidden)
      if label.startswith('DGT') and Agt is None:
       with torch.inference_mode():Agt=ray_samples(A,K,GTfocus,.05/(2*case['fstop']),64,fill,bg)
      fallback=Agt if label.startswith('DGT') else As
      with torch.inference_mode():p=render_fallbacks(cloud,K,ffocus,.05/(2*case['fstop']),64,fill,bg,fallback);r,a=zero_parts(cloud,K)
      for fall,back in [('F1',fill),('F2',Azero)]:
       path=dest/(label+'_'+fall+'.npz')
       if not path.exists():save(path,rgb_linear=p[fall].cpu().numpy(),prediction_srgb=display(p[fall].cpu().numpy()),alpha=p['alpha'].cpu().numpy(),zero_srgb=display((r+(1-a[...,None])*back).cpu().numpy()))
      timings.append(dict(label=label,seconds=time.monotonic()-t,peak_MiB=torch.cuda.max_memory_allocated()/2**20,focus_m=ffocus,gaussians=len(cloud.means)));del cloud,p
      print('T1_RENDER',name,f,rule,case['id'],label,flush=True)
     write(dest/'render.json',dict(scene=name,frame=f,id=case['id'],nominal_focus_m=case['nominal_focus_m'],fstop=case['fstop'],rule=rule,focus_m=focus,GT_focus_m=GTfocus,GT_focus_subject_pixels=int(mask.sum()),GT_old_A_focus_reference=str(old),order=audits,timings=timings,background_radius_px=24,seconds=time.monotonic()-start,all_groups=METHODS))
     del ordered,As,Agt;torch.cuda.empty_cache()
   del A,D,DGT,HGT;torch.cuda.empty_cache()

def evaluate(net=None):
 guard();net=model() if net is None else net;rows=[]
 for name in SCENES:
  desc=desc_for(name)
  for entry in desc['entries']:
   f=entry['frame'];target=load(entry['input_npz'])['rgb_srgb']
   for rule in ['P2','P1']:
    for case in entry['cases']:
     dest=cd(name,f,case,rule)
     if not(dest/'render.json').exists():continue
     truth,regions=reference(name,desc,case,f);invalid=~load(EXPLICIT/name/f'f{f:03d}'/'explicit-depth.npz')['dominant_valid'];regions=dict(regions,invalid_geometry=invalid)
     for method in METHODS:
      cache=dest/(method+'-scores.json')
      if cache.exists():v=json.loads(cache.read_text())
      else:
       p=load(dest/(method+'.npz'));n=512**2;s=float(np.square(p['zero_srgb'].astype(np.float64)-target).sum());v=dict(dof=score(p['prediction_srgb'],truth,regions,net),zero=dict(pixels=n,sse=s,psnr=float(-10*np.log10(max(s/(3*n),1e-30)))));write(cache,v)
      for region,stats in v['dof'].items():rows.append(dict(scene=name,frame=f,id=case['id'],rule=rule,nominal_focus_m=case['nominal_focus_m'],fstop=case['fstop'],method=method,region=region,**stats,zero_sse=v['zero']['sse'],zero_pixels=v['zero']['pixels'],zero_PSNR=v['zero']['psnr']))
     print('T1_SCORE',name,f,rule,case['id'],flush=True)
 write(OUT/'scores.json',dict(rows=rows))

def follow():
 guard();net=model();last=-1
 while True:
  count=len(list(OUT.glob('*/f*/P*/*/render.json')))
  if count!=last:evaluate(net);last=count
  if count==72:return
  time.sleep(10)

def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['verify','render','evaluate','follow']);a=p.parse_args();OUT.mkdir(exist_ok=True);globals()[a.mode]()
if __name__=='__main__':main()
