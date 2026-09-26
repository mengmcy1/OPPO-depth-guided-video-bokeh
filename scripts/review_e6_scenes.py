"""Create per-scene keyframe evidence while E6 generation/rendering proceeds."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


def rgb(path):
    x = cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
    assert x is not None,path
    if x.ndim == 2:
        x = np.repeat(x[...,None],3,axis=2)
    else:
        x = x[...,:3][...,::-1]
    return np.uint8(np.round(x.astype(float)/np.iinfo(x.dtype).max*255))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stage',choices=['inputs','reference','complete'],default='inputs')
    p.add_argument('--only',default='')
    a = p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    config = json.loads((a.input/'config.json').read_text())
    chosen = set(a.only.split(',')) if a.only else {s['id'] for s in config['scenes']}
    for scene in config['scenes']:
        if scene['id'] not in chosen:
            continue
        folder = a.input/scene['id']
        meta = json.loads((folder/'metadata.json').read_text())
        dest = a.output/(folder.name+'.jpg')
        if dest.exists():
            raise FileExistsError(dest)
        sheet = Image.new('RGB',(1280,5*210),'white')
        draw = ImageDraw.Draw(sheet)
        for row,index in enumerate([0,6,12,18,24]):
            name = f'{index:05d}.png'
            source = rgb(folder/'rgb'/name)
            alpha = rgb(folder/'alpha'/name)
            if a.stage == 'inputs':
                depth = np.load(folder/'depth'/f'{index:05d}.npy')
                radius = scene['K']*np.abs(1/depth-1/scene['focus_distance_m'])
                heat = cv2.applyColorMap(np.uint8(np.clip(radius/12,0,1)*255),cv2.COLORMAP_TURBO)[...,::-1]
                panels = [source,alpha,heat,source]
                labels = 'RGB / Alpha / nominal CoC (0..12px) / edge crop'
            else:
                reference = rgb(folder/'cycles_optix_s1024_f1'/name)
                panels = [source,reference,alpha if a.stage=='reference' else rgb(folder/'pred_any2bokeh'/name),reference]
                labels = 'RGB / Cycles / '+('Alpha' if a.stage=='reference' else 'Any-to-Bokeh')+' / Cycles edge crop'
            # Crop around the strongest Alpha boundary row near the upper half of the subject.
            mask = alpha[:,:,0]
            band = cv2.dilate(mask,np.ones((3,3),np.uint8)) != cv2.erode(mask,np.ones((3,3),np.uint8))
            ys,xs = np.where(band)
            if len(ys):
                target = int(np.quantile(ys,.25))
                ids = np.abs(ys-target)<8
                cx,cy = int(np.median(xs[ids])),target
            else:
                cx,cy = source.shape[1]//2,source.shape[0]//2
            x = int(np.clip(cx-160,0,source.shape[1]-320));y = int(np.clip(cy-90,0,source.shape[0]-180))
            panels[-1] = panels[-1][y:y+180,x:x+320]
            for col,panel in enumerate(panels):
                sheet.paste(Image.fromarray(panel).resize((320,180)),(col*320,row*210+30))
            draw.text((3,row*210+4),f'{folder.name} frame {index} K={scene["K"]} focus={scene["focus_distance_m"]}m | {labels}',fill='black')
        sheet.save(dest,quality=92)
        print(dest,flush=True)


if __name__ == '__main__':
    main()
