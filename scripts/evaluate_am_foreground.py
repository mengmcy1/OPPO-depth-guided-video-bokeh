"""Train-only proxy consistency and controlled foreground-color recovery."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('NUMBA_THREADING_LAYER', 'workqueue')
os.environ.setdefault('NUMBA_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import pymatting
from pymatting import estimate_foreground_ml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.synthetic.common import srgb_to_linear, linear_to_srgb


def dump(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)


def load(path, mode='RGB'):
    return np.asarray(Image.open(path).convert(mode), dtype=np.float32) / 255


def background(spec, shape):
    h, w = shape
    if 'value_srgb' in spec:
        return np.full((h,w,3), spec['value_srgb'], dtype=np.float32)
    if spec['name'] == 'checker':
        y,x = np.ogrid[:h,:w]
        v = ((x//spec['period_px'] + y//spec['period_px']) % 2).astype(np.float32)
        return np.repeat(v[...,None],3,axis=2)
    return np.asarray(ImageOps.fit(Image.open(ROOT/spec['raw_path']).convert('RGB'),
                                  (w,h),method=Image.Resampling.BILINEAR),dtype=np.float32)/255


def errors(diff, mask):
    d = diff[mask].astype(np.float64)
    mse = float(np.mean(d*d))
    return {'mae':float(np.mean(np.abs(d))), 'mse':mse,
            'psnr':float(-10*np.log10(mse)) if mse else None}


def gradient_error(diff, mask):
    total, count = 0., 0
    for axis in [0,1]:
        d = np.diff(diff,axis=axis)
        m = (mask[1:] & mask[:-1]) if axis == 0 else (mask[:,1:] & mask[:,:-1])
        total += float(np.abs(d[m]).sum(dtype=np.float64))
        count += int(m.sum())*3
    return total/count if count else None


def crop_box(alpha):
    soft = (alpha > .05) & (alpha < .95)
    h,w = soft.shape
    # Select before running any candidate, using alpha alone.
    size = min(256,h,w)
    positions = [(int(soft[y:y+size,x:x+size].sum()),x,y)
                 for y in sorted(set(list(range(0,h-size+1,128))+[h-size]))
                 for x in sorted(set(list(range(0,w-size+1,128))+[w-size]))]
    _,x,y = max(positions,key=lambda p:p[0])
    return x,y,x+size,y+size


def panel_image(arr):
    return Image.fromarray(np.uint8(np.clip(arr,0,1)*255+.5))


def run_sample(index, sample, config, output):
    assert pymatting.__version__ == config['pymatting_version']
    name = sample['name']
    dest = output/name
    dest.mkdir()
    original = load(ROOT/f'data/raw/AM-2k/train/original/{name}.jpg')
    reference = load(ROOT/f'data/raw/AM-2k/train/fg/{name}.png')
    alpha = load(ROOT/f'data/raw/AM-2k/train/mask/{name}.png','L')
    assert original.shape == reference.shape == (*alpha.shape,3)
    assert np.isfinite(original).all() and np.isfinite(reference).all()
    soft = (alpha>.05)&(alpha<.95)
    transition = (alpha>0)&(alpha<1)
    assert soft.any()
    a = alpha[...,None]
    ref_lin = srgb_to_linear(reference)
    backgrounds = [background(b,alpha.shape) for b in config['backgrounds']]
    bg = backgrounds[index % len(backgrounds)]
    inputs = {'natural_proxy':original,
              'controlled_srgb':a*reference+(1-a)*bg,
              'controlled_linear':linear_to_srgb(a*ref_lin+(1-a)*srgb_to_linear(bg))}
    box = crop_box(alpha)
    x0,y0,x1,y1 = box
    crop = np.s_[y0:y1,x0:x1]
    # Warm up fixed dtype/signature outside all measured timings.
    estimate_foreground_ml(np.full((32,32,3),.5,dtype=np.float32),
                           np.full((32,32),.5,dtype=np.float32),**config['ml_parameters'])
    result = {'name':name,'category':sample['category'],'shape':list(alpha.shape),
              'source_id':'AM-2k:'+name,'soft_pixels':int(soft.sum()),'crop_xyxy':box,
              'controlled_background':config['backgrounds'][index%len(backgrounds)]['name'],
              'inputs':{}}
    for kind, image in inputs.items():
        measured = {}
        views = {'reference':reference[crop]}
        for method in config['candidates']:
            start = time.perf_counter()
            if method == 'naive':
                raw = image.copy()
            else:
                raw = estimate_foreground_ml(srgb_to_linear(image) if method == 'ml_linear' else image,
                                             alpha,**config['ml_parameters'])
            runtime = time.perf_counter()-start
            assert raw.shape == reference.shape and np.isfinite(raw).all()
            extrema = [float(raw.min()),float(raw.max())]
            clipped_fraction = float(np.mean((raw<0)|(raw>1)))
            raw = np.clip(raw,0,1)
            pred = linear_to_srgb(raw) if method == 'ml_linear' else raw
            lin = raw if method == 'ml_linear' else srgb_to_linear(pred)
            diff = pred-reference
            ldiff = lin-ref_lin
            stats = {'runtime_seconds':runtime,'pre_clamp_range':extrema,
                     'clipped_fraction':clipped_fraction,
                     'fg_srgb_soft':errors(diff,soft),'fg_linear_soft':errors(ldiff,soft),
                     'fg_srgb_transition':errors(diff,transition),
                     'linear_composite_soft':errors(a*ldiff,soft),
                     'alpha_weighted_srgb_full_mae':float(np.mean(np.abs(a*diff),dtype=np.float64)),
                     'fg_srgb_soft_gradient_mae':gradient_error(diff,soft),
                     'display_composite_soft':{}}
            # All numerical recompositions use linear light, including sRGB-input controls.
            for spec, new_bg in zip(config['backgrounds'],backgrounds):
                new_lin = srgb_to_linear(new_bg)
                composite = linear_to_srgb(a*lin+(1-a)*new_lin)
                target = linear_to_srgb(a*ref_lin+(1-a)*new_lin)
                stats['display_composite_soft'][spec['name']] = errors(composite-target,soft)
            measured[method] = stats
            views[method] = pred[crop]
        result['inputs'][kind] = measured
        # Native-size fixed crop panels; do not use JPEG previews for scores.
        canvas = Image.new('RGB',(4*256,8*280),(35,35,35))
        draw = ImageDraw.Draw(canvas)
        for col, (label, patch) in enumerate(views.items()):
            draw.text((col*256+5,4),label,fill='white')
            canvas.paste(panel_image(patch),(col*256,24))
            for row, (spec, new_bg) in enumerate(zip(config['backgrounds'],backgrounds),1):
                comp = linear_to_srgb(a[crop]*srgb_to_linear(patch)+(1-a[crop])*srgb_to_linear(new_bg[crop]))
                draw.text((col*256+5,row*280+4),spec['name'],fill='white')
                canvas.paste(panel_image(comp),(col*256,row*280+24))
        canvas.save(dest/(kind+'-edges.jpg'),quality=94)
    overview = Image.new('RGB',(768,276),(30,30,30))
    draw = ImageDraw.Draw(overview)
    for i,(label,im) in enumerate([('original',original),('alpha',np.repeat(a,3,axis=2)),('official proxy FG',reference)]):
        thumb = panel_image(im)
        if i == 0:
            ImageDraw.Draw(thumb).rectangle(box,outline='red',width=4)
        thumb.thumbnail((256,256))
        overview.paste(thumb,(i*256,20))
        draw.text((i*256+4,3),label,fill='white')
    overview.save(dest/'overview.jpg',quality=94)
    dump(dest/'metrics.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/am-foreground-estimation.json')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    args=p.parse_args()
    config=json.loads(args.config.read_text())
    development=json.loads((ROOT/config['development_config']).read_text())
    categories=json.loads((ROOT/'data/raw/AM-2k/am2k_split_category.json').read_text())
    samples=development['samples']
    assert len({s['name'] for s in samples})==len(samples)
    for sample in samples:
        assert categories[sample['name']]['split']=='train'
        assert 'AM-2k:'+sample['name'] not in development['prior_development_ids']
    args.output.mkdir(parents=True,exist_ok=False)
    dump(args.output/'config.json',config)
    start=time.perf_counter()
    rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs=[pool.submit(run_sample,i,s,config,args.output) for i,s in enumerate(samples)]
        for job in as_completed(jobs):
            row=job.result()
            rows.append(row)
            print(f'{len(rows)}/{len(samples)} {row["name"]}',flush=True)
    rows.sort(key=lambda r:r['name'])
    summary={}
    for kind in rows[0]['inputs']:
        summary[kind]={}
        for method in config['candidates']:
            values=[r['inputs'][kind][method] for r in rows]
            summary[kind][method]={
                'primary_linear_composite_soft_mae':float(np.mean([v['linear_composite_soft']['mae'] for v in values])),
                'fg_srgb_soft_mae':float(np.mean([v['fg_srgb_soft']['mae'] for v in values])),
                'fg_srgb_soft_psnr':float(np.mean([v['fg_srgb_soft']['psnr'] for v in values])),
                'fg_srgb_soft_gradient_mae':float(np.mean([v['fg_srgb_soft_gradient_mae'] for v in values])),
                'display_composite_soft_mae_7bg':float(np.mean([b['mae'] for v in values for b in v['display_composite_soft'].values()])),
                'runtime_mean_seconds':float(np.mean([v['runtime_seconds'] for v in values])),
                'primary_better_than_naive_sources':sum(r['inputs'][kind][method]['linear_composite_soft']['mae'] < r['inputs'][kind]['naive']['linear_composite_soft']['mae'] for r in rows)}
    report={'status':'NUMERICAL_COMPLETE_VISUAL_REVIEW_PENDING','config':str(args.config),
            'versions':{'pymatting':pymatting.__version__,'numpy':np.__version__},
            'wall_seconds':time.perf_counter()-start,'workers':args.workers,
            'validation_images_read':0,'summary':summary,'rows':rows}
    dump(args.output/'metrics.json',report)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
