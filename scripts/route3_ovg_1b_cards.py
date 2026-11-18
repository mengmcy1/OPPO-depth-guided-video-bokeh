"""Independent CPU thin-lens integration for tier C, with a matched Cycles audit."""
import argparse
from functools import lru_cache
import json
import os
import time

import cv2
import numpy as np

from route3_ovg_1b_common import CONFIG, OUT, SOURCE, ROOT, guard, read
from route3_ovg_r2_cards import cast
from route3_ovg_card import load_texture
from route3_ovg_subpixel import save_json, save_npz


def sample_disk(count, radius):
    index = np.arange(count, dtype=np.float64)
    theta = index * np.pi * (3. - np.sqrt(5.))
    r = radius * np.sqrt((index + .5) / count)
    return np.stack((r * np.cos(theta), r * np.sin(theta), np.zeros_like(r)), axis=-1)


def camera_rays(K, height, width, dx, dy):
    y, x = np.mgrid[:height, :width]
    pixels = np.stack((x + dx, y + dy, np.ones_like(x)), axis=-1)
    return (pixels @ np.linalg.inv(K).T).astype(np.float32)


def sample_texture(tex, points, transform):
    coord = (points[..., :2] / transform[0] + transform[1:]) % 1
    xy = (coord * [tex.shape[1] - 1, tex.shape[0] - 1]).astype(np.float32)
    # remap's destination axes are limited to signed 16-bit coordinates.
    return np.concatenate([cv2.remap(tex, sub[:, None, 0], sub[:, None, 1],
                                     cv2.INTER_LINEAR)[:, 0]
                           for sub in np.array_split(xy, max(1, (len(xy) + 15999) // 16000))])


def shade(oid, xyz, materials):
    rgb = np.zeros((*oid.shape, 3), np.float32)
    for j, material in enumerate(materials, start=1):
        mask = oid == j
        if mask.any():
            color, transform = material
            rgb[mask] = color if transform is None else sample_texture(color, xyz[mask], transform)
    return rgb


def integrate(rectangles, holes, origin, K, size, focus, fstop, materials, *, disk_count, pixel_grid):
    """Trace true intersections for each lens ray, independently of GS and K16 labels."""
    total = np.zeros((size, size, 3), np.float64)
    pupil = sample_disk(disk_count, .05 / (2 * fstop)) if fstop else np.zeros((1, 3))
    for y in range(pixel_grid):
        for x in range(pixel_grid):
            ray = camera_rays(K, size, size, (x + .5) / pixel_grid, (y + .5) / pixel_grid)
            for shift in pupil:
                _, ids, points, _ = cast(ray * focus - shift, origin + shift, rectangles, holes)
                total += shade(ids, points, materials)
    return (total / (len(pupil) * pixel_grid**2)).astype(np.float32)


@lru_cache(maxsize=8)
def texture(asset):
    return load_texture(asset, SOURCE / 'C/assets')[0] if (SOURCE / 'C/assets').exists() else load_texture(
        asset, ROOT / 'outputs/route3-ovg-1a-2026-11-18-01/C/assets')[0]


def clip_parameters(meta, index):
    row = meta['rows'][index]
    origin = np.asarray(row['c2w'], dtype=np.float32)[:3, 3]
    rectangles = np.asarray(meta['rectangles'], dtype=np.float32)
    materials = [(texture(meta['assets'][int(i)]['asset']), t)
                 for i, t in zip(meta['asset_order'], meta['texture_map_per_plane'])]
    return rectangles, meta['holes'], origin, np.asarray(row['K'], dtype=np.float32), materials


def prepare(manifest):
    """Generate independent references only on the two K16-labeled configurations."""
    guard()
    meta = read(manifest)
    if 'rectangles' not in meta:
        return
    opts = CONFIG['card_reference']
    size = 256
    dest = OUT / 'C/optical' / meta['scene']
    start = time.monotonic()
    rows = []
    for entry in meta['K16_aperture']:
        frame = entry['frame']
        focus = entry['focus_m']
        cfg = clip_parameters(meta, frame)
        file = dest / f'f{frame:02d}-N1.4-F{focus:g}.npz'
        if not file.exists():
            rgb = integrate(*cfg[:-1], size, focus, opts['fstop'], cfg[-1],
                            disk_count=opts['lens_samples'], pixel_grid=opts['pixel_grid'])
            save_npz(file, optical_rgb=rgb)
        rows.append(dict(frame=frame, focus=focus, fstop=opts['fstop'], path=str(file.relative_to(ROOT))))
    save_json(dest / 'manifest.json', dict(rows=rows, seconds=time.monotonic()-start, samples=opts,
                                         independent_exact_planes=True, linear=True))
    print('C_OPTICAL_READY', meta['scene'], flush=True)


def audit_trace():
    path = OUT / 'P1/card-audit'
    spec = read(path / 'spec.json')
    rect = np.asarray([p['rectangle'] for p in spec['planes']])
    mats = [(np.asarray(p['color']), None) for p in spec['planes']]
    K = np.array(spec['K'], np.float32)
    rows = []
    os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
    for focus in spec['focus_m']:
        result = integrate(rect, [], np.zeros(3), K, spec['size'], focus, spec['fstop'], mats,
                           disk_count=512, pixel_grid=4)
        ref = cv2.imread(str(path / f'F{focus:g}.exr'), cv2.IMREAD_UNCHANGED)
        if ref is None:
            raise FileNotFoundError(path / f'F{focus:g}.exr')
        ref = ref[..., :3][..., ::-1]
        err = (result-ref).astype(np.float64)
        psnr = float(-10*np.log10(max(np.mean(err**2), 1e-30)))
        rows.append(dict(focus_m=focus, psnr_db=psnr, mae=float(np.mean(abs(err))),
                         max_abs=float(abs(err).max())))
        save_npz(path / f'F{focus:g}-analytic.npz', optical_rgb=result)
    passed = all(row['psnr_db'] >= CONFIG['card_check']['psnr_min_db'] for row in rows)
    save_json(path / 'result.json', dict(passed=passed, matched_B_optics=spec, rows=rows,
              scope='matched flat emissive planes; no indirect illumination or aperture transmission variation'))
    print(json.dumps(rows), flush=True)
    if not passed:
        raise RuntimeError('P1 paired C/Cycles comparison failed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('job', choices=['audit', 'references'])
    options = parser.parse_args()
    if options.job == 'audit':
        audit_trace()
    else:
        from concurrent.futures import ProcessPoolExecutor
        from route3_ovg_1b_data import setup

        setup()
        cv2.setNumThreads(1)
        manifests = sorted((OUT / 'C/windows').glob('*/manifest.json'))
        with ProcessPoolExecutor(max_workers=8) as pool:
            for _ in pool.map(prepare, manifests):
                pass
