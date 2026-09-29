"""Prepare selected visual groups and paired intermediate diagnostics; no inference."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--config',default='configs/drbokeh-protocol-recovery-v2.json')
p.add_argument('--only',default='',help='Comma-separated scene IDs, for explicitly labelled posthoc review')
p.add_argument('--output',type=Path)
args=p.parse_args()
config=json.loads((ROOT/args.config).read_text())
base=json.loads((ROOT/config['base_protocol']).read_text())
selected=args.only.split(',') if args.only else config['scene_ids']
assert len(set(selected))==len(selected) and set(selected)<=set(config['scene_ids'])
run=ROOT/config['output'];out=ROOT/args.output if args.output else run/'visual-review-20260929';out.mkdir(exist_ok=True)
rows=[]


def thumbnail(x,size=(256,144)):
    if isinstance(x,Path):im=Image.open(x).convert('RGB')
    else:
        x=np.squeeze(x)
        im=Image.fromarray(np.uint8(np.round(np.clip(x,0,1)*255))).convert('RGB')
    return im.resize(size,Image.Resampling.LANCZOS)


for sid in selected:
    final=Image.new('RGB',(7*256,5*168),'white');fd=ImageDraw.Draw(final)
    layers=Image.new('RGB',(8*224,5*150),'white');ld=ImageDraw.Draw(layers)
    for row,t in enumerate([0,6,12,18,24]):
        src=ROOT/base['input']/sid
        with np.load(run/'full_public/clips'/sid/'diagnostics'/f'{t:05d}.npz') as z:
            a={k:z[k] for k in z.files}
        with np.load(run/'gt_alpha/clips'/sid/'diagnostics'/f'{t:05d}.npz') as z:
            b={k:z[k] for k in z.files}
        pairs=[('AIF',src/'rgb'/f'{t:05d}.png'),('GT',src/'gt_mpib_public_srgb'/f'{t:05d}.png'),
               ('LDF alpha',a['ldf_alpha']),('GT union alpha',b['provided_alpha'])]
        pairs += [(branch,run/branch/'clips'/sid/'pred_drbokeh'/f'{t:05d}.png') for branch in ('full_public','gt_alpha','oracle_layers')]
        for col,(name,value) in enumerate(pairs):
            final.paste(thumbnail(value),(col*256,row*168+24));fd.text((col*256+3,row*168+3),f'{sid}/{t:02d} {name}',fill='black')
        intermediates=[]
        for key,label in [('RGB_inpainting_mask','LaMa mask'),('fg_rgbad_render','FG disparity'),
                          ('bg_rgbad_render','BG disparity'),('RGB_inpainting_output','LaMa RGB')]:
            for tag,values in [('LDF',a),('GTalpha',b)]:
                x=values[key][...,4] if 'rgbad' in key else values[key]
                intermediates.append((label+' '+tag,x))
        for col,(name,value) in enumerate(intermediates):
            layers.paste(thumbnail(value,(224,126)),(col*224,row*150+24));ld.text((col*224+3,row*150+3),f'{t:02d} {name}',fill='black')
        aa=a['ldf_alpha'];bb=b['provided_alpha'];ha=aa>.1;hb=bb>.1
        numeric=dict(scene=sid,frame=t,alpha_MAE=float(np.abs(aa-bb).mean()),
                     missed_union_fraction_of_image=float((hb&~ha).mean()),extra_LDF_fraction_of_image=float((ha&~hb).mean()),
                     inpaint_mask_changed_fraction=float((a['RGB_inpainting_mask']!=b['RGB_inpainting_mask']).mean()))
        for name,key in [('FG','fg_rgbad_render'),('BG','bg_rgbad_render')]:
            numeric[name+'_repaired_disparity_change_MAE']=float(np.abs(a[key][...,4]-b[key][...,4]).mean())
            numeric[name+'_disparity_ranges']=[[float(z[key][...,4].min()),float(z[key][...,4].max())] for z in (a,b)]
        numeric['LaMa_RGB_change_MAE']=float(np.abs(a['RGB_inpainting_output']-b['RGB_inpainting_output']).mean())
        rows.append(numeric)
    for kind,im in [('final',final),('layers',layers)]:
        with (out/f'{sid}-{kind}.jpg').open('xb') as f:im.save(f,format='JPEG',quality=95)
    print('PREPARED',sid,flush=True)
with (out/'intermediate-comparison.json').open('x') as f:
    json.dump(dict(scene_count=len(selected),groups=len(rows),rows=rows,scope='Paired changes, not independent layer accuracy; visual depth clamped to [0,1] only for display'),f,indent=2)
