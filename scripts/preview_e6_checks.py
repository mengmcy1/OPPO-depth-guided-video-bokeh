"""Small visual evidence for E6 controls and full-clip smoke tests."""
import argparse
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


def load(path):
    x = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    assert x is not None, path
    return np.uint8(np.round(x[..., ::-1].astype(float)/np.iinfo(x.dtype).max*255))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--mode', choices=['edges','clips','noise'], required=True)
    a = p.parse_args()
    if a.mode != 'edges':
        import imageio_ffmpeg  # Use the existing any2bokeh environment for MP4 previews.
    out = a.input/'previews'/a.mode
    out.mkdir(parents=True, exist_ok=False)
    if a.mode == 'edges':
        for prefix in ['am_05','vm_02']:
            folders = sorted((a.input/'controls/edges').glob(prefix+'_*'))
            sheet = Image.new('RGB',(1024,168*len(folders)),'white')
            draw = ImageDraw.Draw(sheet)
            for i, folder in enumerate(folders):
                rgb, bokeh = [load(folder/s/'00000.png') for s in ['rgb','cycles_optix_s1024_f1']]
                # Same crop coordinates for all backgrounds and depth orders.
                x, y = (400,120) if prefix == 'vm_02' else (440,190)
                panels = [rgb,bokeh,rgb[y:y+256,x:x+256],bokeh[y:y+256,x:x+256]]
                for j, panel in enumerate(panels):
                    im = Image.fromarray(panel).resize((256,144))
                    sheet.paste(im,(j*256,i*168+24))
                draw.text((2,i*168+4),folder.name+' : RGB / Cycles / RGB crop / Cycles crop',fill='black')
            sheet.save(out/(prefix+'.png'))
        return
    base = a.input/'clips' if a.mode=='clips' else a.input/'controls/noise'
    for folder in sorted(base.iterdir()):
        if not folder.is_dir():
            continue
        primary = folder/'cycles_optix_s1024_f1'
        secondary = folder/('pred_any2bokeh' if a.mode=='clips' else 'cycles_optix_s1024_f1_seed100000')
        files = sorted(primary.glob('*.png'))
        assert len(files)==25
        sheet = Image.new('RGB',(1024,7*212),'white')
        draw = ImageDraw.Draw(sheet)
        with imageio.get_writer(out/(folder.name+'.mp4'),format='FFMPEG',fps=30,codec='libx264',quality=7,macro_block_size=1) as writer:
            row = 0
            for i,file in enumerate(files):
                x,y = load(file),load(secondary/file.name)
                if a.mode=='clips':
                    panels = [load(folder/'rgb'/file.name),x,y]
                else:
                    panels = [x,y,np.uint8(np.clip(np.abs(x.astype(float)-y)*20,0,255))]
                panel = np.concatenate([cv2.resize(v,(512,288)) for v in panels],axis=1)
                writer.append_data(panel)
                if i in [0,4,8,12,16,20,24]:
                    sheet.paste(Image.fromarray(panel).resize((1024,192)),(0,row*212+20))
                    draw.text((2,row*212+3),f'{folder.name} frame {i}: '+('RGB / Cycles / Any-to-Bokeh' if a.mode=='clips' else 'seed0 / seed100000 / abs difference x20'),fill='black')
                    row+=1
        sheet.save(out/(folder.name+'.jpg'))


if __name__ == '__main__':
    main()
