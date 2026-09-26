"""Render known pilot layers with the author Dr.Bokeh core, bypassing estimated preprocessing."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
sys.path.insert(0,str(root.parent/'DrBokeh/app/Render/DScatter'))
from synthetic.common import srgb_to_linear,linear_to_srgb
from scatter import Multi_Layer_Renderer
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);a=p.parse_args()
torch.set_num_threads(4);renderer=Multi_Layer_Renderer(41).cuda().eval()
for folder in sorted(a.input.glob('*')):
 if not (folder/'metadata.json').exists():continue
 meta=json.loads((folder/'metadata.json').read_text());W,H=meta['config']['width'],meta['config']['height'];K=meta['config']['K']
 out=folder/'bokeh_dr';out.mkdir(exist_ok=False)
 bg=srgb_to_linear(np.asarray(Image.open(folder/'layers/background.png')).astype(np.float32)/255)[64:64+H,64:64+W]
 times=[]
 for f in meta['frames']:
  layers=[]
  for file,z in zip(f['foreground_files'],f['foreground_depths_m']):
   q=np.asarray(Image.open(folder/file)).astype(np.float32)/255
   layers.append(np.dstack((srgb_to_linear(q[:,:,:3]),q[:,:,3],np.full((H,W),1/z))).astype(np.float32))
  layers.append(np.dstack((bg,np.ones((H,W)),np.full((H,W),1/meta['config']['background_depth_m']))).astype(np.float32))
  torch.cuda.synchronize();start=time.perf_counter()
  with torch.inference_mode():result=renderer.inference(layers,K,f['focus_disparity'])
  torch.cuda.synchronize();times.append(time.perf_counter()-start)
  assert result.shape==(H,W,3) and np.isfinite(result).all()
  Image.fromarray(np.uint8(np.round(linear_to_srgb(result)*255))).save(out/f"{f['index']:05d}.png")
 (out/'timing.json').write_text(json.dumps({'seconds':times,'total_seconds':sum(times),'scope':'known-layer author core + transfers; excludes PNG IO; no learned depth/matting/inpainting','lens':41},indent=2))
 print(folder.name,sum(times),flush=True)
