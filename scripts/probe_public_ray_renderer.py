"""Check public ray interfaces and compare matched scenes; do not infer author identity."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from synthetic.public_ray_renderer import load_renderer,render
from synthetic.common import srgb_to_linear,linear_to_srgb
p=argparse.ArgumentParser();p.add_argument('--scene',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
torch.set_num_threads(4)
modules={k:load_renderer(k) for k in ['MPIB','BokehDiff']}
def tensor(x):return torch.from_numpy(x).permute(0,3,1,2)[None].contiguous().cuda().float()
def evaluate(x,y):
 d=x.astype(float)-y.astype(float);m=float(np.mean(d*d));return {'MAE':float(np.mean(abs(d))),'PSNR':float(-10*np.log10(m)) if m else None,'max_abs':float(abs(d).max())}
# Flat layers at the same depth must reduce to ordinary alpha composition.
h,w=96,128;images=np.zeros((2,h,w,3),np.float32);images[0]=[.8,.2,.1];images[1]=[.1,.3,.7]
alpha=np.ones((2,h,w,1),np.float32);alpha[0]=0;alpha[0,20:76,24:104]=.6
expected=images[0]*alpha[0]+images[1]*(1-alpha[0]);controls={}
for k,mod in modules.items():
 out=render(mod,tensor(images),tensor(alpha),[[2.,2.]],16.,2.,101,k)[0].permute(1,2,0).cpu().numpy();controls[k]=evaluate(out,expected);assert controls[k]['MAE']<1e-3,controls[k]
 # Constant layer under defocus checks finite normalized ray accumulation including borders.
 constant=render(mod,tensor(images[1:]),tensor(np.ones((1,h,w,1),np.float32)),[[6.]],16.,2.,101,k)[0].permute(1,2,0).cpu().numpy()
 controls[k]['constant_defocus_max_error']=float(abs(constant-images[1]).max());assert controls[k]['constant_defocus_max_error']<1e-3
m=json.loads((a.scene/'metadata.json').read_text());results=[]
for index in [0,12,24]:
 frame=m['frames'][index];rgba=[np.asarray(Image.open(a.scene/f),dtype=np.float32)/255 for f in frame['foreground_files']]
 bg=np.asarray(Image.open(a.scene/'layers/background.png'),dtype=np.float32)[64:-64,64:-64]/255
 images=np.stack([x[:,:,:3] for x in rgba]+[bg]);alpha=np.stack([x[:,:,3:] for x in rgba]+[np.ones((*bg.shape[:2],1),np.float32)])
 depths=[frame['foreground_depths_m']+[frame['background_depth_m']]]
 import cv2
 cycles=cv2.imread(str(a.scene/'cycles_optix_s1024_f1'/f'{index:05d}.png'),cv2.IMREAD_UNCHANGED)[:,:,::-1].astype(np.float32)/65535
 outputs={}
 for color in ['srgb','linear']:
  x=images if color=='srgb' else srgb_to_linear(images)
  for kind,mod in modules.items():
   torch.cuda.synchronize();start=time.perf_counter();out=render(mod,tensor(x),tensor(alpha),depths,m['config']['K'],frame['focus_distance_m'],101,kind);torch.cuda.synchronize();seconds=time.perf_counter()-start
   out=out[0].permute(1,2,0).cpu().numpy();out=out if color=='srgb' else linear_to_srgb(out)
   label=kind+'_'+color;outputs[label]=out
   Image.fromarray(np.uint8(np.round(np.clip(out,0,1)*255))).save(a.output/f'{index:05d}_{label}.png')
   results.append({'frame':index,'renderer':kind,'working_space':color,'seconds':seconds,'vs_Cycles':evaluate(out,cycles)})
 for color in ['srgb','linear']:results.append({'frame':index,'comparison':'MPIB_vs_BokehDiff_'+color,**evaluate(outputs['MPIB_'+color],outputs['BokehDiff_'+color])})
 print('completed frame',index,flush=True)
report={'status':'PUBLIC_RENDERER_PROBE_COMPLETE_NOT_AUTHOR_IDENTITY_PROOF','samples_per_side':101,'BokehDiff_Eta':0,'controls':controls,'rows':results,'limits':['Any-to-Bokeh kernel remains missing; matching public interface does not prove exact implementation.','MPIB kernel used unchanged, only cupy.int compatibility alias and tensor parameter reshape.','Color-space variants diagnose differences; do not choose by Any-to-Bokeh score.']}
with (a.output/'report.json').open('x') as f:json.dump(report,f,indent=2)
print(json.dumps(report,indent=2))
