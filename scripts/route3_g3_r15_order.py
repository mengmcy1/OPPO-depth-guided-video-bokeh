"""One paired DepthSplat re-render for the revision 15 ordering decision."""
import fcntl
import time
import cv2
import numpy as np
import torch

from route3_g3_r15 import LOG, OUT, PREV, REC, PREFIX, qa, read, write
from route3_g3_geometry import SCENES, EXPLICIT, desc_for, load, tensor, visible_cloud, reference
from route3_explicit_layer import aligned_hidden
from route3_ldr_u1r import model
from route3_upper_bound_evaluate import display, pool, score
from src.dual_layer import depth_band
from src.layered_aperture import merge_layers, pixel_layer
from src.ordered_layers import order_hidden, ray_samples, render_fallbacks

DEST = OUT/'R2'


def main():
    torch.set_num_threads(4)
    net=model()
    rows, audits = [], []
    for name in SCENES:
        desc=desc_for(name)
        for e in desc['entries']:
            f=e['frame']
            old=PREV/'T4/depthsplat-official'/name/f'f{f:03d}'
            save=DEST/name/f'f{f:03d}'
            save.mkdir(parents=True,exist_ok=True)
            jobs=[c for c in e['cases'] if c['fstop']==1.4]
            d=load(e['input_npz'])
            z=load(old/'depth.npz')['depth']
            parents=load(desc['parents'](f))
            align=load(old/'alignment.npz')
            focus_table=read(old/'focus.json')
            with (LOG/'gpu0.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                qa.guard(True)
                depth,K=tensor(z),tensor(d['K'])
                bg_order=tensor(cv2.dilate(z,np.ones((49,49),np.uint8)))
                A=visible_cloud(desc,e)
                D=pixel_layer(tensor(d['rgb_linear']),depth,K,torch.eye(4,device='cuda'))
                fill=tensor(load(EXPLICIT/'fill'/(name+f'-f{f:03d}')/'fill.npz')['fill_rgb_linear'])
                bg=tensor(load(EXPLICIT/name/f'f{f:03d}'/'fill-input.npz')['background_depth'])
                for c in jobs:
                    cid=c['id']
                    focus=focus_table[str(c['nominal_focus_m'])]['focus_m']
                    band=depth_band(z,d['K'][0,0],focus,.05/2.8)[0]
                    hidden,_=aligned_hidden(parents,align,band,K)
                    with torch.inference_mode():
                        fallback=ray_samples(A,K,focus,.05/2.8,64,fill,bg)
                        for rule in ('O1','O2'):
                            h,result=order_hidden(hidden,depth,K,rule,bg_order)
                            t=time.monotonic()
                            rgb=render_fallbacks(merge_layers(D,h),K,focus,.05/2.8,64,fill,bg,fallback)['F2']
                            np.savez_compressed(save/f'{cid}-{rule}.npz',prediction_srgb=display(rgb.cpu().numpy()))
                            audits.append(dict(scene=name,frame=f,id=cid,focus=c['nominal_focus_m'],
                                               focus_m=focus,seconds=time.monotonic()-t,
                                               violation_fraction=result['violating']/max(result['total'],1),**result))
            for c in jobs:
                cid=c['id']
                truth, masks=reference(name,desc,c,f)
                for m,p in [('D0',old/cid/'D0.npz')]+[(r,save/f'{cid}-{r}.npz') for r in ('O1','O2')]:
                    values=score(load(p)['prediction_srgb'],truth,masks,net)
                    rows.extend(dict(scene=name,frame=f,focus=c['nominal_focus_m'],id=cid,
                                     method=m,region=k,**v) for k,v in values.items())
            print('R2',name,f,flush=True)
    def aggregate(method,region,focus=None):
        return pool([r for r in rows if r['scene']!='B2-high' and r['method']==method and r['region']==region
                     and (focus is None or r['focus']==focus)])['psnr']
    d0=aggregate('D0','in_focus')
    diff={r:aggregate(r,'in_focus')-d0 for r in ('O1','O2')}
    far={r:aggregate(r,'recoverable',5.) for r in ('O1','O2')}
    chosen=max(diff,key=diff.get) if abs(diff['O1']-diff['O2'])>=.1 else ('O1' if far['O1']>far['O2'] else 'O2')
    summary=dict(rule=chosen,in_focus_delta_dB=diff,far_PSNR=far,audits=audits,rows=rows,
                 selection_definition='2 G2 scenes pooled SSE,focus guard both foci,tie <0.1dB,far recoverable,tie O2')
    write(REC/f'{PREFIX}-R2.json',summary)
    write(DEST/'selection.json',dict(rule=chosen,in_focus_delta_dB=diff,far_PSNR=far))
    print('ORDER_SELECTED',chosen,diff,far,flush=True)


if __name__=='__main__':
    main()
