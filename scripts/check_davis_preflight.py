"""Common eight-sequence integrity checks and fixed visual review sheets; no VEPI."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--method',required=True);a=p.parse_args()
    protocol=json.loads((ROOT/'configs/davis-baselines.json').read_text())
    c=json.loads((ROOT/protocol['data_protocol']).read_text());out=ROOT/protocol['output']/a.method
    summaries=[]
    for name in protocol['preflight_sequences']:
        inputs=ROOT/c['derived']/'eval'/name;pred=out/'predictions'/name;audit=out/'audit'/name
        done=json.loads((audit/'complete.json').read_text());assert done['frames']==25 and done['finite']
        paths=sorted(pred.glob('*.png'));assert [p.name for p in paths]==[f'{t:05d}.png' for t in range(25)]
        geometry=json.loads((ROOT/c['derived']/'sequences'/name/'geometry.json').read_text())
        rows=[]
        for t,path in enumerate(paths):
            with Image.open(path) as im:
                assert im.mode=='RGB' and im.size==(1024,576);im.verify()
            with Image.open(path) as im:q=np.asarray(im)
            assert q.max()>q.min()
            row=done['outputs'][t] if a.method=='any2bokeh' else json.loads((audit/f'{t:05d}.json').read_text())
            assert row['frame']==t and row['focus']==geometry['frames'][t]['focus'] and row['K']==16
            rows.append(row)
        if a.method=='any2bokeh':
            x=json.loads((audit/'input.json').read_text());assert x['author_decoder_identity_VAE_index_check'] and not x['GT_mask_argument_supplied']
        else:assert all(r['finite'] and not r['GT_mask_argument_supplied'] for r in rows)
        summary=dict(id=name,frames=25,finite=True,PNG_complete=True,focus_K_correct=True,GT_mask_argument_supplied=False)
        if a.method in ('bokehme','bokehdiff'):
            summary.update(span_range=[min(r['span'] for r in rows),max(r['span'] for r in rows)],
                effective_K_range=[min(r['effective_K'] for r in rows),max(r['effective_K'] for r in rows)])
        if a.method=='drbokeh':
            summary.update(LDF_coverage_range=[min(r['full_public']['LDF_mask']['coverage'] for r in rows),max(r['full_public']['LDF_mask']['coverage'] for r in rows)],
                LDF_empty_frames=[r['frame'] for r in rows if r['full_public']['LDF_mask']['threshold_empty']],
                LDF_full_frames=[r['frame'] for r in rows if r['full_public']['LDF_mask']['threshold_full']])
        summaries.append(summary)
        canvas=Image.new('RGB',(320*4,205*5),'white');draw=ImageDraw.Draw(canvas)
        for j,t in enumerate(protocol['keyframes']):
            rgb=Image.open(inputs/'rgb'/f'{t:05d}.jpg').convert('RGB')
            dp=next((inputs/'disparity').glob(f'{t:05d}_zf_*.png'));d=cv2.imread(str(dp),0);f=float(dp.stem.split('_zf_')[1])
            depth=Image.fromarray(cv2.applyColorMap(d,cv2.COLORMAP_TURBO)[:,:,::-1])
            coc=Image.fromarray(cv2.applyColorMap(np.uint8(np.abs(d.astype(np.float32)/255-f)*255),cv2.COLORMAP_TURBO)[:,:,::-1])
            frames=[rgb,depth,coc,Image.open(pred/f'{t:05d}.png')]
            for k,im in enumerate(frames):
                canvas.paste(im.resize((320,180)),(k*320,j*205+25))
                draw.text((k*320+4,j*205+4),f'{name} f{t} '+['RGB','source VDA','source CoC/16',a.method][k],fill='black')
        dest=out/'previews'/f'{name}.jpg';dest.parent.mkdir(exist_ok=True)
        if not dest.exists():canvas.save(dest,quality=93)
        if a.method=='drbokeh':
            canvas=Image.new('RGB',(320*4,205*5),'white');draw=ImageDraw.Draw(canvas)
            for j,t in enumerate(protocol['keyframes']):
                with np.load(out/'diagnostics'/name/f'{t:05d}.npz') as z:
                    arrays=[np.repeat(z['ldf_alpha'],3,axis=-1),
                        np.repeat(z['fg_rgbad_render'][...,4:5],3,axis=-1),
                        np.repeat(z['bg_rgbad_render'][...,4:5],3,axis=-1),z['RGB_inpainting_output']]
                    for k,x in enumerate(arrays):
                        img=Image.fromarray(np.uint8(np.round(np.clip(x,0,1)*255)))
                        canvas.paste(img.resize((320,180)),(k*320,j*205+25))
                        draw.text((k*320+4,j*205+4),f'{name} f{t} '+['LDF alpha','FG repaired d','BG repaired d','LaMa RGB'][k],fill='black')
            dest=out/'previews'/f'{name}-layers.jpg'
            if not dest.exists():canvas.save(dest,quality=93)
    report=dict(status='NUMERICAL_PREFLIGHT_PASSED_VISUAL_REVIEW_PENDING',method=a.method,scenes=8,frames=200,
        keyframes=protocol['keyframes'],VEPI_computed=False,checks=summaries,
        scope='Implementation/input checks only; model failures retained, no score threshold')
    dest=ROOT/'03_实验记录/records'/f'davis-{a.method}-preflight-20260929.json'
    if dest.exists():assert json.loads(dest.read_text())==report
    else:
        with dest.open('x') as f:json.dump(report,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
    print('PREFLIGHT_NUMERICAL_COMPLETE',a.method,flush=True)


if __name__=='__main__':main()
