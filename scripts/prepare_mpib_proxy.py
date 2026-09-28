"""Derive sRGB inputs from the fixed 200-scene layers without changing geometry."""
import argparse,json,os,sys
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]

def prepare(scene,cfg,out):
    base=ROOT/cfg['geometry_run']
    sub='input-repair-v2' if scene['id']==cfg['input_repair']['scene'] else 'clips'
    source=base/sub/scene['id'];meta=json.loads((source/'metadata.json').read_text())
    assert meta['source']==scene and len(meta['frames'])==25
    dest=out/scene['id'];dest.mkdir();(dest/'rgb').mkdir()
    for name in ['layers','disparity','depth','alpha','mask']:(dest/name).symlink_to((source/name).resolve(),target_is_directory=True)
    bg=np.asarray(Image.open(source/'layers/background.png').convert('RGB'),dtype=np.float32)[64:-64,64:-64]/255
    differences=[]
    for frame in meta['frames']:
        rgb=bg.copy()
        for file in reversed(frame['foreground_files']):
            layer=np.asarray(Image.open(source/file).convert('RGBA'),dtype=np.float32)/255
            alpha=layer[:,:,3:];rgb=layer[:,:,:3]*alpha+rgb*(1-alpha)
        assert np.isfinite(rgb).all() and rgb.min()>=0 and rgb.max()<=1.00001
        saved=np.uint8(np.round(np.clip(rgb,0,1)*255));Image.fromarray(saved).save(dest/'rgb'/f"{frame['index']:05d}.png")
        old=np.asarray(Image.open(source/'rgb'/f"{frame['index']:05d}.png"),dtype=np.float32)/255
        differences.append(float(np.abs(saved.astype(np.float32)/255-old).mean()))
    meta.update(role='MPIB-public proxy reproduction, not official GT or held-out benchmark',color_space='sRGB-space alpha composition and ray integration, no gamma conversion',geometry_source=str(source.relative_to(ROOT)),input_srgb_vs_old_linear_mae=differences)
    (dest/'metadata.json').write_text(json.dumps(meta,indent=2));return scene['id']

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=ROOT/'configs/any2bokeh-mpib-public-proxy.json');p.add_argument('--output',type=Path,required=True);p.add_argument('--only',default='');p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    cfg=json.loads(a.config.read_text());a.output.mkdir(parents=True,exist_ok=True);cp=a.output/'config.json'
    if cp.exists():assert json.loads(cp.read_text())==cfg
    else:cp.write_text(json.dumps(cfg,indent=2))
    scenes=[s for s in cfg['scenes'] if not a.only or s['id'] in a.only.split(',')];assert scenes
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        jobs=[pool.submit(prepare,s,cfg,a.output) for s in scenes]
        for i,job in enumerate(as_completed(jobs)):print(i+1,len(scenes),job.result(),flush=True)
if __name__=='__main__':main()
