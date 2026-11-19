"""Appearance labels from the existing A, C and B training splits."""
from concurrent.futures import ThreadPoolExecutor
import time

import cv2
import numpy as np

from route3_ovg_1b_cards import clip_parameters, shade
from route3_ovg_r2_cards import cast
import route3_ovg_data as data
from route3_ovg_1b_common import OUT as PRIOR, ROOT, REC, read
from route3_ovg_subpixel import offsets, save_json, save_npz

OUT = ROOT/'outputs/route3-ovg-stage2-2026-11-19-01'
PREFIX = 'route3-ovg-stage2-2026-11-19'
SIZE = 256
SEED = 20261119
MEMORY = dict(shift_px=4, hole_count=3, hole_side_px=[12,44], gaussian_rgb_sigma=.025)
SOURCES = (
    dict(name='AM-2k', split='train', role='estimated foreground and human-labeled matte', license='MIT per release agreement',
         agreement='https://jizhizili.github.io/files/gfm_datasets_agreements/AM-2k_Dataset_Release_Agreement.pdf'),
    dict(name='BG-20k', split='train', role='real backgrounds', license='MIT per release agreement',
         agreement='https://jizhizili.github.io/files/gfm_datasets_agreements/BG-20k_Dataset_Release_Agreement.pdf'),
)


def rgb(path):
    raw = cv2.imread(str(path),cv2.IMREAD_COLOR)
    if raw is None:
        raise OSError(path)
    return data.srgb_to_linear(raw[...,::-1].astype(np.float32)/255)


def save_array(path, array):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_suffix('.part')
    with tmp.open('wb') as f:
        np.save(f,array.astype(np.float16))
    tmp.replace(path)


def asset(job):
    role,path,companion=job
    dest=OUT/'A'/role/(path.stem+'.npy')
    if not dest.is_file():
        frame=rgb(path)
        if companion is not None:
            mask=cv2.imread(str(companion),cv2.IMREAD_UNCHANGED)
            if mask is None:
                raise OSError(companion)
            if mask.ndim==3:
                mask=mask[...,0]
            alpha=mask.astype(np.float32)/np.iinfo(mask.dtype).max
            area=lambda im: cv2.resize(im,(SIZE,SIZE),interpolation=cv2.INTER_AREA)
            a=area(alpha)
            premult=area(frame*alpha[...,None])
            frame=np.concatenate((np.divide(premult,a[...,None],out=np.zeros_like(premult),where=a[...,None]>0),
                                  a[...,None]),axis=-1)
        else:
            h,w=frame.shape[:2]
            span=min(h,w)
            y,x=(h-span)//2,(w-span)//2
            frame=cv2.resize(frame[y:y+span,x:x+span],(SIZE,SIZE),interpolation=cv2.INTER_AREA)
        save_array(dest,frame)
    return str(dest.relative_to(ROOT))


def compose(fg,bg,rng,simulate):
    fg=fg.astype(np.float32)
    bg=bg.astype(np.float32)
    if rng.integers(2):
        fg=fg[:,::-1].copy()
    if rng.integers(2):
        bg=bg[:,::-1].copy()
    color,a=fg[...,:3],fg[...,3]
    image=a[...,None]*color+(1-a[...,None])*bg
    mem=np.zeros_like(bg)
    known=np.zeros(a.shape,bool)
    if simulate:
        h,w=a.shape
        dx,dy=rng.integers(-MEMORY['shift_px'],MEMORY['shift_px']+1,size=2)
        matrix=np.array([[1,0,dx],[0,1,dy]],np.float32)
        mem=cv2.warpAffine(bg,matrix,(w,h))
        known=cv2.warpAffine(np.ones_like(a),matrix,(w,h))>.5
        for _ in range(MEMORY['hole_count']):
            y,x=rng.integers(0,h,size=2)
            size=int(rng.integers(*MEMORY['hole_side_px']))
            known[y:y+size,x:x+size]=False
        mem=np.clip(mem+rng.normal(0,MEMORY['gaussian_rgb_sigma'],mem.shape).astype(np.float32),0,1)
        mem[~known]=0
    return image,color,a,bg,mem,known


def c_labels(path):
    meta=read(path)
    count=0
    max_alpha,max_rgb,max_png=0.,0.,0.
    for index,row in enumerate(meta['rows']):
        destination=OUT/'C'/meta['scene']/f'f{index:03d}-appearance.npz'
        if destination.is_file():
            d=data.load(destination)
            max_alpha=max(max_alpha,float(d['max_alpha']))
            max_rgb=max(max_rgb,float(d['max_rgb']))
            max_png=max(max_png,float(d['max_png']))
            count+=1
            continue
        rect,holes,origin,K,materials=clip_parameters(meta,index)
        target=data.load(ROOT/row['label'])
        front,back=target['depth1'],target['depth2']
        cut=np.where(back>front,2/(1/np.maximum(front,1e-6)+1/np.maximum(back,1e-6)),1e4)
        yy,xx=np.mgrid[:front.shape[0],:front.shape[1]].astype(np.float32)+.5
        pixels=np.stack((xx,yy,np.ones_like(xx)),-1)
        ray=(pixels[...,None,:]+offsets(4).astype(np.float32))@np.linalg.inv(K).T
        z,ids,xyz,_=cast(ray,origin,rect,holes)
        color=shade(ids,xyz,materials)
        first=(z>0)&(z<cut[...,None])
        n=first.sum(-1)
        a=n.astype(np.float32)/16
        f=np.where(first[...,None],color,0).sum(-2)
        b=np.where(first[...,None],0,color).sum(-2)
        f=np.divide(f,n[...,None],out=np.zeros_like(f),where=n[...,None]>0)
        b=np.divide(b,(16-n)[...,None],out=np.zeros_like(b),where=(n<16)[...,None])
        total=color.mean(-2)
        delta=np.abs(a-target['alpha'])[target['valid_alpha']].max(initial=0)
        recomposed=np.abs(a[...,None]*f+(1-a[...,None])*b-total).max()
        from route3_ovg_stage2_query import to_display
        native=cv2.imread(str(ROOT/row['image']),cv2.IMREAD_COLOR)[...,::-1].astype(np.int16)
        computed=np.rint(to_display(total)*255).astype(np.int16)
        codes=np.abs(computed-native).max()
        png=np.abs(total-rgb(ROOT/row['image'])).max()
        if delta or recomposed>3e-7 or codes>1:
            raise ValueError(f'C label mismatch {meta["scene"]} {index}: {delta}, {recomposed}, {png}')
        save_npz(destination,alpha=a,foreground=f,background=b,full=total,
                 max_alpha=delta,max_rgb=recomposed,max_png=png,png_codes=codes)
        max_alpha,max_rgb,max_png=max(max_alpha,float(delta)),max(max_rgb,float(recomposed)),max(max_png,float(png))
        count+=1
    return dict(scene=meta['scene'],frames=count,source=str(path.relative_to(ROOT)),
                coverage_max_error=max_alpha,composition_max_error=max_rgb,source_PNG_max_error=max_png)


def prepare():
    from shutil import disk_usage
    start=time.monotonic()
    OUT.mkdir(parents=True,exist_ok=True)
    if disk_usage(OUT).free<60*2**30:
        raise RuntimeError('Data disk below 60 GiB')
    raws=ROOT/'data/raw'
    matte=raws/'AM-2k/train'
    fg=sorted((matte/'fg').glob('*.png'))
    assert all((matte/'mask'/f.name).is_file() for f in fg)
    bg=sorted((raws/'BG-20k/train').glob('*.jpg'))
    rng=np.random.default_rng(SEED)
    selected=sorted(rng.choice(len(bg),2048,replace=False))
    background=[bg[i] for i in selected]
    save_json(REC/f'{PREFIX}-materials.json',dict(sources=SOURCES,
        foreground_pairs=len(fg),backgrounds=len(background),available_train_backgrounds=len(bg),
        selection_seed=SEED, attribution='Li, Zhang, Maybank and Tao, Bridging Composite and Real: Towards End-to-end Deep Image Matting, IJCV 2021',
        local_originals_read_only=True, original_owner_copyright=True, memory_2a=MEMORY,
        memory_2b_A='invalid, no multi-view capture'))
    cv2.setNumThreads(1)
    card_paths=sorted((PRIOR/'C/windows').glob('card-r2-*/manifest.json'))
    with ThreadPoolExecutor(max_workers=6) as pool:
        card_future=[pool.submit(c_labels,p) for p in card_paths[:6]]
        todo=[('foreground',p,matte/'mask'/p.name) for p in fg]
        todo.extend(('background',p,None) for p in background)
        refs=list(pool.map(asset,todo))
        assets=dict(foreground=refs[:len(fg)],background=refs[len(fg):])
        checks=[]
        for i in range(20):
            f=np.load(ROOT/assets['foreground'][int(rng.integers(len(fg)))])
            b=np.load(ROOT/assets['background'][int(rng.integers(len(background)))])
            im,color,a,b,mem,valid=compose(f,b,rng,True)
            ref=a.astype(np.float64)[...,None]*color+(1-a.astype(np.float64)[...,None])*b
            error=float(np.abs(im-ref).max())
            if error>2e-7 or np.any(mem[~valid]):
                raise ValueError('A composition or memory mask mismatch')
            checks.append(dict(sample=i,max_linear_error=error,memory_valid=float(valid.mean())))
        save_json(OUT/'A/manifest.json',dict(**assets,composition_checks=checks))
        cards=[job.result() for job in card_future]
        cards.extend(pool.map(c_labels,card_paths[6:]))
    save_json(OUT/'S0.json',dict(A_count=[len(fg),len(background)],A_checks=checks,C=cards,
        C_frames=sum(r['frames'] for r in cards),C_grid=4,seconds=time.monotonic()-start,
        C_legacy='original geometry pilot excluded; stage 1b expanded cards supply matching appearance and optical geometry schema'))
    print('STAGE2_S0',len(fg),len(background),len(cards),flush=True)


if __name__=='__main__':
    prepare()
