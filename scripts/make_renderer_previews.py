"""Build review contact sheets and 2x2 videos from pilot outputs."""
import argparse
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
import imageio.v2 as imageio
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path);a=p.parse_args();out=a.output or a.input/'previews_optix1024';out.mkdir(parents=True,exist_ok=False)
try:font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
except OSError:font=ImageFont.load_default()
items=[('Input RGB','rgb'),('Any-to-Bokeh','pred_any2bokeh'),('Dr.Bokeh reference','bokeh_dr'),('Cycles reference (1024 spp)','cycles_optix_s1024_f0.945244')]
def tile(folder,sub,i,label):
 x=cv2.imread(str(folder/sub/f'{i:05d}.png'),-1);assert x is not None
 scale=65535 if x.dtype==np.uint16 else 255;x=np.uint8(np.round(x[:,:,::-1].astype(np.float32)/scale*255))
 t=Image.new('RGB',(512,320),'white');t.paste(Image.fromarray(x).resize((512,288)),(0,32));ImageDraw.Draw(t).text((8,6),label,fill='black',font=font);return t
for folder in sorted(a.input.glob('scene_*')):
 sheet=Image.new('RGB',(2048,960),'white')
 for row,i in enumerate([0,12,24]):
  for col,(label,sub) in enumerate(items):sheet.paste(tile(folder,sub,i,f'{label} | frame {i}'),(col*512,row*320))
 sheet.save(out/f'{folder.name}.png')
 writer=imageio.get_writer(out/f'{folder.name}.mp4',fps=20,codec='libx264',quality=8)
 for i in range(25):
  frame=Image.new('RGB',(1024,640),'white')
  for j,(label,sub) in enumerate(items):frame.paste(tile(folder,sub,i,label),(j%2*512,j//2*320))
  writer.append_data(np.asarray(frame))
 writer.close();print(folder.name,flush=True)
