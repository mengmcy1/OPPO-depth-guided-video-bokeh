"""Fixed native-DepthNet response preflight; deliberately does not compute VEPI."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from prepare_davis_v2 import write_new,png


def worker(c,names,limit):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and os.environ.get('TF_ENABLE_ONEDNN_OPTS')=='0'
    sys.path.insert(0,str(ROOT/'src'))
    from baselines.deeplens_native import DeepLensNative
    cfg=c['deeplens'];out=ROOT/cfg['output'];cv2.setNumThreads(1)
    model=DeepLensNative((ROOT/cfg['repository']).resolve(),ROOT/cfg['checkpoint'],cfg['threads_per_worker'])
    write_new(out/'runtime'/f'{os.getpid()}.json',model.audit)
    for name in names:
        dest=out/name;inputs=ROOT/c['derived']/'eval'/name
        for i in range(limit or 25):
            record=dest/f'{i:05d}.json'
            if record.exists():
                r=json.loads(record.read_text());assert r['sequence']==name and len(r['apertures'])==3
                for a in r['apertures']:assert (dest/f"aperture-{a['aperture']:g}"/f'{i:05d}.png').exists()
                continue
            high=cv2.resize(cv2.imread(str(inputs/'rgb'/f'{i:05d}.jpg')),(1024,576),interpolation=cv2.INTER_LINEAR).astype(np.float32)/255
            low=cv2.resize(high,(512,288),interpolation=cv2.INTER_AREA)
            mask=cv2.resize(cv2.imread(str(inputs/'mask'/f'{i:05d}.png'),cv2.IMREAD_GRAYSCALE),(512,288),interpolation=cv2.INTER_NEAREST)>128
            assert mask.any() and (~mask).any()
            start=time.monotonic();d=model.predict_depth(low)
            mean=float(d[mask].mean());cost=np.where(mask,np.abs(d-mean),np.inf)
            y,x=np.unravel_index(np.argmin(cost),d.shape)
            assert mask[y,x]
            delta=np.abs(d-d[y,x]);aps=[];previous=None
            for ap in cfg['aperture_probe']:
                value,audit=model.predict(low,high,int(x),int(y),ap,d)
                png(dest/f'aperture-{ap:g}'/f'{i:05d}.png',np.uint8(np.round(np.clip(value,0,1)*255)))
                audit.update(aperture=ap,MAE_vs_input=float(np.abs(value-high).mean()),
                             MAE_vs_previous_aperture=None if previous is None else float(np.abs(value-previous).mean()))
                previous=value;aps.append(audit)
            depthfile=dest/'depth'/f'{i:05d}.npy';depthfile.parent.mkdir(parents=True,exist_ok=True)
            if depthfile.exists():assert np.array_equal(np.load(depthfile),d)
            else:
                with depthfile.open('xb') as f:np.save(f,d)
            result=dict(sequence=name,frame=i,seconds=time.monotonic()-start,depth_range=[float(d.min()),float(d.max())],
                foreground_depth=dict(mean=mean,min=float(d[mask].min()),max=float(d[mask].max())),
                background_depth=dict(mean=float(d[~mask].mean()),min=float(d[~mask].min()),max=float(d[~mask].max())),
                focus_pixel_low_xy=[int(x),int(y)],focus_depth=float(d[y,x]),focus_mean_error=float(cost[y,x]),
                delta_depth_quantiles=np.quantile(delta,[0,.25,.5,.75,1]).tolist(),
                foreground_delta_quantiles=np.quantile(delta[mask],[0,.25,.5,.75,1]).tolist(),
                background_delta_quantiles=np.quantile(delta[~mask],[0,.25,.5,.75,1]).tolist(),
                apertures=aps,metric_selection=False)
            write_new(record,result)
            print('DL_FRAME',name,i,round(result['seconds'],2),flush=True)
        print('DL_SEQUENCE_DONE',name,flush=True)


def launch(c,config,names):
    out=ROOT/c['deeplens']['output'];write_new(out/'protocol.json',c)
    native=ROOT.parent/'envs/deeplens-tf1-cpu/bin/python'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',TF_ENABLE_ONEDNN_OPTS='0',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1')
    workers=c['deeplens']['workers'];logs=ROOT/'logs/davis-v2-20260929';logs.mkdir(exist_ok=True,parents=True)
    def part(i):
        group=names[i::workers]
        if not group:return
        with (logs/f'deeplens-worker-{i}.log').open('a') as f:
            subprocess.run([str(native),str(Path(__file__).resolve()),'worker','--config',config,'--sequences',*group],
                           env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    with ThreadPoolExecutor(workers) as pool:list(pool.map(part,range(workers)))
    summarize(c)


def summarize(c):
    from PIL import Image,ImageDraw
    cfg=c['deeplens'];out=ROOT/cfg['output'];rows=[]
    for name in cfg['preflight_sequences']:
        for i in range(25):
            r=json.loads((out/name/f'{i:05d}.json').read_text())
            for ap in cfg['aperture_probe']:
                p=out/name/f'aperture-{ap:g}'/f'{i:05d}.png'
                with Image.open(p) as im:assert im.size==(1024,576) and im.mode=='RGB';im.verify()
            rows.append(r)
        canvas=Image.new('RGB',(320*5,205*5),'white');draw=ImageDraw.Draw(canvas)
        for j,t in enumerate(cfg['keyframes']):
            rgb=Image.open(ROOT/c['derived']/'eval'/name/'rgb'/f'{t:05d}.jpg').convert('RGB')
            d=np.load(out/name/'depth'/f'{t:05d}.npy');color=cv2.applyColorMap(np.uint8(np.clip(d,0,1)*255),cv2.COLORMAP_TURBO)
            r=json.loads((out/name/f'{t:05d}.json').read_text());x,y=r['focus_pixel_low_xy']
            cv2.drawMarker(color,(x,y),(255,255,255),cv2.MARKER_CROSS,14,2)
            imgs=[rgb,Image.fromarray(color[:,:,::-1])]+[Image.open(out/name/f'aperture-{ap:g}'/f'{t:05d}.png') for ap in cfg['aperture_probe']]
            for k,im in enumerate(imgs):
                canvas.paste(im.resize((320,180)),(k*320,j*205+25))
                draw.text((k*320+5,j*205+5),f'{name} f{t} '+['RGB','Native depth + click','ap0.25','ap0.5','ap1.0'][k],fill='black')
        target=out/'previews'/f'{name}.jpg';target.parent.mkdir(exist_ok=True)
        if not target.exists():canvas.save(target,quality=92)
    result=dict(status='EIGHT_SEQUENCE_NATIVE_RESPONSE_COMPLETE_VISUAL_AND_APERTURE_DECISION_PENDING',
        sequences=cfg['preflight_sequences'],frames=len(rows),predictions=len(rows)*3,
        apertures=cfg['aperture_probe'],aperture_rule=None,VEPI_computed=False,
        max_focus_mean_error=max(r['focus_mean_error'] for r in rows),
        max_signed_error=max(a['signed_error'] for r in rows for a in r['apertures']),
        all_depth_repeat_bitwise_equal=all(a['depth_repeat_bitwise_equal'] for r in rows for a in r['apertures']),
        runtime_records=[json.loads(p.read_text()) for p in sorted((out/'runtime').glob('*.json'))],
        rows=rows)
    write_new(out/'summary.json',result)
    write_new(ROOT/'03_实验记录/records/deeplens-davis-native-preflight-20260929.json',result)
    print('DL_PREFLIGHT_COMPLETE',len(rows),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['worker','launch','summarize'])
    p.add_argument('--config',default='configs/davis-eval-v2.json');p.add_argument('--sequences',nargs='*');p.add_argument('--limit',type=int)
    a=p.parse_args();c=json.loads((ROOT/a.config).read_text());names=a.sequences or c['deeplens']['preflight_sequences']
    assert set(names)<=set(c['deeplens']['preflight_sequences'])
    if a.phase=='worker':worker(c,names,a.limit)
    elif a.phase=='launch':launch(c,a.config,names)
    else:summarize(c)
