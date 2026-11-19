"""B-tier geometric coverage and camera-matched pure foreground targets."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import shutil
import time

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')

import cv2
import numpy as np

from route3_ovg_subpixel import OUT as B_LABELS, PREVIOUS, ROOT, TRAIN_FRAMES, intersect, save_json, save_npz

OUT = ROOT/'outputs/route3-ovg-1c-2026-11-19-01'/'phase2-B'
AUDIT_FRAMES = (1, 30, 60)
# Grid quadrature and stochastic Cycles do not share subpixel sample locations.
TOLERANCE = dict(coverage_max=0., same_seed_complement_linear_MAE=.002,
                 composed_display_MAE=.025, composed_display_p95=.10,
                 rendered_vs_mesh_coverage_MAE=.075)


def scene_coverage(name):
    from route3_blender_b2_geometry import World
    start = time.monotonic()
    base = PREVIOUS/'truth'/name
    folder = OUT/name
    folder.mkdir(parents=True, exist_ok=True)
    world = World(base/'scene-geometry.npz', 512)
    h = w = 512
    yy, xx = np.mgrid[:h, :w]
    xy = np.c_[xx.ravel(), yy.ravel()]
    rows = []
    for frame in TRAIN_FRAMES:
        original = B_LABELS/'B'/name/'pinhole-subpixels'/f'f{frame:03d}.npz'
        dest = folder/f'f{frame:03d}-coverage.npz'
        with np.load(original) as d:
            front, back = d['depth1'], d['depth2']
            valid, dual, grid = d['valid_alpha'], d['valid_dual'], int(d['subpixel_grid'])
            cutoff = np.full((h, w), 1000., np.float32)
            split = (front>0) & (back>front)
            cutoff[split] = 2/(1/front[split]+1/back[split])
            if not dest.is_file():
                counts = []
                for row in range(0,h,16):
                    group = slice(row*w,(row+16)*w)
                    depth, _, _ = intersect(world,frame,xy[group],grid)
                    threshold = cutoff.ravel()[group, None]
                    counts.append(((depth>0) & (depth<threshold)).sum(-1))
                coverage = (np.concatenate(counts).reshape(h,w)/grid**2).astype(np.float32)
                save_npz(dest,coverage=coverage,cutoff_m=cutoff,valid_coverage=valid,
                         valid_dual=dual,subpixel_grid=np.int16(grid),K=d['K'],camera_c2w=d['camera_c2w'])
            else:
                with np.load(dest) as cached:
                    coverage = cached['coverage']
            error = np.abs(coverage[valid]-d['alpha'][valid])
            rows.append(dict(scene=name,frame=frame,valid=int(valid.sum()),grid=grid,dual=int(dual.sum()),
                             boundary=int((valid&(coverage>0)&(coverage<1)).sum()),
                             max_abs_error=float(error.max(initial=0)),different=int((error>0).sum())))
    passed = all(r['max_abs_error'] <= TOLERANCE['coverage_max'] for r in rows)
    save_json(folder/'coverage-audit.json',dict(passed=passed,seconds=time.monotonic()-start,rows=rows))
    print('PHASE2_COVERAGE',name,passed,flush=True)
    if not passed:
        raise ValueError(f'Saved alpha differs from axial cutoff coverage: {name}')
    return rows


def pinhole_render_params(base, frame):
    # Resolve the immutable original render job for the linked camera assets.
    image = (base/'inputs'/f'f{frame:03d}.exr').resolve()
    source = image.parent.parent
    manifests = (source/'inputs-jobs-results.json', source/'inputs-r13-jobs-results.json')
    for manifest in manifests:
        if manifest.is_file():
            for row in json.loads(manifest.read_text()):
                if row['frame']==frame and Path(row['output']).resolve()==image:
                    return dict(spp=row['spp'],seed=row['seed'],source_image=str(image))
    raise FileNotFoundError(f'No original pinhole seed for {image}')


def prepare(names, workers, audit_only):
    OUT.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(OUT).free < 60*2**30:
        raise RuntimeError('Data disk below 60 GiB')
    with ProcessPoolExecutor(max_workers=workers) as pool:
        groups = list(pool.map(scene_coverage, names))
    jobs = []
    for name in names:
        base = PREVIOUS/'truth'/name
        for frame in AUDIT_FRAMES if audit_only else TRAIN_FRAMES:
            config = pinhole_render_params(base, frame)
            folder = OUT/name
            for kind in ('foreground','visible-background'):
                if kind == 'visible-background' and frame not in AUDIT_FRAMES:
                    continue
                key = f'f{frame:03d}-{kind}'
                jobs.append(dict(config, id=name+'-'+key, scene=name, frame=frame,
                                 blend=str(base/'scene.blend'), coverage=str(folder/f'f{frame:03d}-coverage.npz'),
                                 output=str(folder/(key+'.exr')), **{'pass':kind}))
    save_json(OUT/('audit-jobs.json' if audit_only else 'render-jobs.json'),jobs)
    save_json(OUT/'coverage-summary.json',dict(rows=[r for group in groups for r in group],
              thresholds=TOLERANCE, cutoff='inverse-depth midpoint, use a far plane when no second mode'))


def image(path):
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(path)
    return raw[..., [2,1,0,3] if raw.shape[-1]==4 else [2,1,0]]


def display(rgb):
    x = np.clip(rgb,0.,1.)
    return np.where(x<=.0031308,12.92*x,1.055*x**(1/2.4)-.055)


def finish(jobs):
    render_rows = {r['id']:r for r in jobs}
    checks = []
    for row in jobs:
        if row['pass'] != 'foreground':
            continue
        folder = Path(row['coverage']).parent
        frame = row['frame']
        f = image(row['output'])
        if f.shape != (512,512,4):
            raise ValueError('Read the native RGBA EXR with an alpha-preserving OpenCV build')
        alpha = f[...,3]
        with np.load(row['coverage']) as d:
            a, reliable, dual = d['coverage'],d['valid_coverage'],d['valid_dual']
        full = image(row['source_image'])[...,:3]
        interior = reliable & (a == 1)
        valid = reliable & (a>0) & (alpha>0)
        foreground = np.divide(f[...,:3],alpha[...,None],out=np.zeros_like(f[...,:3]),where=valid[...,None])
        foreground[interior] = full[interior]
        valid |= interior
        if not np.isfinite(foreground).all():
            raise ValueError(f'Nonfinite foreground color: {row["id"]}')
        save_npz(folder/f'f{frame:03d}-foreground-label.npz',linear_rgb=foreground,
                 valid_color=valid, valid_boundary_color=valid&dual,rendered_coverage=alpha)
        other = render_rows.get(row['id'].removesuffix('foreground')+'visible-background')
        if other is None:
            continue
        b = image(other['output'])
        b_alpha = b[...,3]
        bg = np.divide(b[...,:3],b_alpha[...,None],out=np.zeros_like(b[...,:3]),where=b_alpha[...,None]>0)
        estimate = a[...,None]*foreground+(1-a[...,None])*bg
        edges = dual & (a>0) & (a<1)
        complementary = (f[...,:3]+b[...,:3]-full)[edges]
        error = np.abs((display(estimate)-display(full))[edges])
        coverage_error = np.abs((alpha-a)[edges])
        metrics = dict(same_seed_complement_linear_MAE=float(np.abs(complementary).mean()),
                       composed_display_MAE=float(error.mean()),composed_display_p95=float(np.quantile(error,.95)),
                       rendered_vs_mesh_coverage_MAE=float(coverage_error.mean()))
        passed = all(value <= TOLERANCE[key] for key,value in metrics.items())
        checks.append(dict(scene=row['scene'],frame=frame,edge_pixels=int(edges.sum()),
                           both_color_samples=int((edges&valid&(b_alpha>0)).sum()),passed=passed,**metrics))
        print('PHASE2_COMPOSITION',row['id'],passed,metrics,flush=True)
    scenes = sorted({r['scene'] for r in jobs})
    reports = []
    for name in scenes:
        folder = OUT/name
        coverage = json.loads((folder/'coverage-audit.json').read_text())
        foreground_rows = [r for r in jobs if r['scene']==name and r['pass']=='foreground']
        reports.append(dict(scene=name, foreground_frames=len(foreground_rows),
                            coverage_frames=len(coverage['rows']), coverage_passed=coverage['passed'],
                            coverage_seconds=coverage['seconds'],
                            total_bytes=sum(p.stat().st_size for p in folder.iterdir() if p.suffix in ('.exr','.npz')),
                            spp=sorted({r['spp'] for r in foreground_rows})))
    save_json(OUT/('pilot-audit.json' if len(jobs)<=6 else 'labels-summary.json'),dict(
        passed=all(r['passed'] for r in checks) and all(r['coverage_passed'] for r in reports),
        audits=checks,tolerances=TOLERANCE, scenes=reports,
        foreground_frames=sum(r['pass']=='foreground' for r in jobs),
        render_seconds=sum(r['seconds'] for r in jobs),EXR_bytes=sum(r['bytes'] for r in jobs),
        total_bytes=sum(r['total_bytes'] for r in reports),
        blender=sorted({r['blender'] for r in jobs}),filter='BOX 1.0',
        layers='Cycles camera-ray holdouts at pixelwise axial cutoff; shadows/reflections unchanged',
        background_for_composition='conditional color from first hits on the complementary camera subrays',
        RGB='boundary: linear unassociated camera pass; opaque interior: exact input radiance; RGBA EXR retained'))
    if not all(r['passed'] for r in checks):
        raise ValueError('B phase 2 label composition audit failed')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=('prepare','finish'))
    parser.add_argument('--scene',action='append')
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--audit-only',action='store_true')
    parser.add_argument('--results',type=Path)
    args = parser.parse_args()
    if args.operation == 'prepare':
        split = json.loads((PREVIOUS/'split.json').read_text())
        prepare(args.scene or split['train']+split['heldout'],args.workers,args.audit_only)
    else:
        finish(json.loads(args.results.read_text()))
