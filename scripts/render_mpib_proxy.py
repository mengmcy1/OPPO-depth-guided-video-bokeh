"""Render the fixed proxy branch using unchanged official MPIB ray kernels."""
import argparse,json,sys,time,subprocess
from pathlib import Path
import numpy as np
from PIL import Image
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from synthetic.public_ray_renderer import load_renderer,render
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--only',default='');p.add_argument('--wait-for-inputs',action='store_true');a=p.parse_args()
cfg=json.loads((a.input/'config.json').read_text());commit=subprocess.check_output(['git','-C',str(ROOT.parent/'MPIB'),'rev-parse','HEAD'],text=True).strip();assert commit==cfg['renderer']['commit']
torch.set_num_threads(4);module=load_renderer('MPIB')
def tensor(x):return torch.from_numpy(x).permute(0,3,1,2)[None].contiguous().cuda().float()
for scene in cfg['scenes']:
    if a.only and scene['id'] not in a.only.split(','):continue
    folder=a.input/scene['id'];start=time.monotonic()
    while a.wait_for_inputs and not (folder/'metadata.json').exists():
        if time.monotonic()-start>7200:raise TimeoutError(folder)
        time.sleep(5)
    m=json.loads((folder/'metadata.json').read_text());assert m['source']==scene
    out=folder/'gt_mpib_public_srgb';out.mkdir(exist_ok=False)
    bg=np.asarray(Image.open(folder/'layers/background.png').convert('RGB'),dtype=np.float32)[64:-64,64:-64]/255
    times=[];ranges=[]
    with torch.inference_mode():
        for frame in m['frames']:
            rgba=[np.asarray(Image.open(folder/f).convert('RGBA'),dtype=np.float32)/255 for f in frame['foreground_files']]
            images=np.stack([x[:,:,:3] for x in rgba]+[bg]);alphas=np.stack([x[:,:,3:] for x in rgba]+[np.ones((*bg.shape[:2],1),np.float32)])
            depths=[frame['foreground_depths_m']+[frame['background_depth_m']]]
            torch.cuda.synchronize();start=time.perf_counter()
            value=render(module,tensor(images),tensor(alphas),depths,scene['K'],frame['focus_distance_m'],cfg['renderer']['samples_per_side'])
            torch.cuda.synchronize();times.append(time.perf_counter()-start)
            x=value[0].permute(1,2,0).cpu().numpy();assert x.shape==(576,1024,3)
            ranges.append([float(x.min()),float(x.max())]);Image.fromarray(np.uint8(np.round(np.clip(x,0,1)*255))).save(out/f"{frame['index']:05d}.png")
    report={'renderer':'MPIB-public proxy','commit':commit,'samples_per_side':101,'working_space':'sRGB','frames':25,'seconds':times,'total_seconds':sum(times),'pre_clamp_ranges':ranges,'scope':'Render calls with transfers, may overlap other work; not standardized efficiency'}
    (out/'timing.json').write_text(json.dumps(report,indent=2));print(scene['id'],25,sum(times),flush=True)
