"""Make fixed-stage diagnostic contact sheets from saved preflight arrays."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--run',type=Path,default=Path('outputs/drbokeh-preflight-20260928'))
p.add_argument('--only',default='')
p.add_argument('--frames',default='0,6,12,18,24')
a=p.parse_args();protocol=json.loads((a.run/'protocol.json').read_text())
names=a.only.split(',') if a.only else protocol['preflight_scenes'];frames=[int(t) for t in a.frames.split(',')]
labels=['Input RGB','Input disparity','LDF alpha','FG (alpha on gray)','FG disparity','LaMa BG RGB','LaMa BG depth','Final Dr.Bokeh']
for sid in names:
    canvas=Image.new('RGB',(8*256,len(frames)*172+22),'white');draw=ImageDraw.Draw(canvas)
    for c,label in enumerate(labels):draw.text((c*256+4,4),label,fill='black')
    for row,t in enumerate(frames):
        folder=a.run/'clips'/sid
        with np.load(folder/'diagnostics'/f'{t:05d}.npz') as q:
            inp=np.asarray(Image.open(Path(protocol['input'])/sid/'rgb'/f'{t:05d}.png')).astype(np.float32)/255
            alpha=q['ldf_alpha'];fg=q['fg_rgbad_render'];bg=q['bg_rgbad_render']
            fg_rgb=np.maximum(fg[...,:3],0)**(1/protocol['gamma'])-protocol['offset']
            views=[inp,q['input_disparity_normalized'],alpha[...,0],fg_rgb*alpha+.5*(1-alpha),fg[...,4],q['RGB_inpainting_output'],bg[...,4],q['final_float_RGB']]
            for c,view in enumerate(views):
                im=Image.fromarray(np.uint8(np.round(np.clip(view,0,1)*255))).convert('RGB').resize((256,144))
                canvas.paste(im,(c*256,row*172+22))
        audit=json.loads((a.run/'timing'/sid/f'{t:05d}.json').read_text())
        draw.text((4,row*172+166),f"{sid} frame {t}; K={audit['geometry']['K_manifest']}; focus'={audit['geometry']['focus_prime']:.4f}; K_Dr={audit['geometry']['K_Dr']:.4f}",fill='black')
    out=a.run/'review';out.mkdir(exist_ok=True)
    canvas.save(out/f"{sid}-{'_'.join(map(str,frames))}.jpg",quality=95)
    print(out/f"{sid}-{'_'.join(map(str,frames))}.jpg")
