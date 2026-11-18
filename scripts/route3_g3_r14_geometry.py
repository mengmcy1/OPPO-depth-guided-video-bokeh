"""Whole-frame geometry versions with registered motion masks and independent P2 focus."""
import argparse
import os
from pathlib import Path

from route3_g3_r14 import C, LOG, OUT, PY, ROOT, qa, read, write
import route3_g3_geometry as evaluation
from route3_g3_geometry import desc_for, load, fit_and_map, stats
from route3_depth_interface import focus_row, subject_mask
from src.c1_regions import background_and_motion
from src.dense_flow_geometry import pair_depth, multiview_depth, sample
from src.dual_layer import depth_band
import cv2
import numpy as np

T4 = OUT/'T4'
FRAMES = C['input_frames']
PREV = ROOT/'outputs/route3-g3-r13-2026-11-18-01/T4'
S1C = ROOT/'outputs/route3-sharp-s1c-2026-11-10-01'
MOTION = read(ROOT/'configs/route3-c1-a1-20261110.json')


def registered_masks(views):
    """Compare a temporal stack registered using input-side depth and pose."""
    masks = {}
    y, x = np.mgrid[:512, :512]
    pixels = np.dstack([x+.5, y+.5, np.ones_like(x)])
    for target, v in views.items():
        world = (pixels@np.linalg.inv(v['K']).T*v['depth'][..., None])@v['c2w'][:3, :3].T+v['c2w'][:3, 3]
        aligned = []
        for u in views.values():
            camera = (world-u['c2w'][:3, 3])@u['c2w'][:3, :3]
            q = camera@u['K'].T
            uv = q[..., :2]/np.maximum(q[..., 2:], 1e-8)-.5
            warped = sample(u['rgb_linear'], uv)
            valid = (camera[..., 2] > 0)&(uv >= 0).all(-1)&(uv <= 511).all(-1)
            aligned.append(np.where(valid[..., None], warped, v['rgb_linear']))
        _, _, _, motion, _ = background_and_motion(np.stack(aligned), MOTION)
        masks[target] = motion[list(views).index(target)]
    return masks


def flow_depth(frame, views, masks, cache, neighbors):
    pairs = []
    v = views[frame]
    for g in neighbors:
        fwd = np.load(cache/f'f{frame:03d}-f{g:03d}.npz')['flow']
        back = np.load(cache/f'f{g:03d}-f{frame:03d}.npz')['flow']
        # Remove moving correspondences before constructing the ray equations.
        yy, xx = np.mgrid[:512, :512]
        matched = np.dstack([xx, yy])+fwd
        eligible = ~(masks[frame] | (sample(masks[g].astype(np.float32), matched) > 0))
        z, uv, _ = pair_depth(fwd, back, v['K'], v['c2w'], views[g]['c2w'], eligible=eligible)
        pairs.append(dict(frame=g, depth=z, other_xy=uv))
    z, valid, support, _ = multiview_depth(pairs, v['K'], v['c2w'], {f:v['c2w'] for f,v in views.items()})
    return z, valid & (support >= 2)


def old_views(desc):
    views = {}
    for f in FRAMES:
        raw = load(desc['model']/f'f{f:03d}.npz')
        calibrated = load(desc['calibration']/f'f{f:03d}.npz')
        views[f] = dict(rgb_linear=raw['rgb_linear'], K=raw['K'], c2w=calibrated['c2w'],
                        depth=calibrated['depth'])
    return views


def run_depthsplat(plan, tag):
    qa.run([PY, 'scripts/route3_depthsplat_depth.py', '--plan', str(plan)], tag, True)


def execute(version):
    evaluation.OUT = T4
    evaluation.LOCK = LOG/'gpu0.lock'
    if version.startswith('depthsplat'):
        plan = read(PREV/'depthsplat-plan.json')
        for row in plan:
            row['bounds_m'] = [.5, 200.]
            row['output'] = str(T4/(version+'-raw')/row['scene']/f'f{row["target_frame"]:03d}.npz')
        path = T4/(version+'-plan.json')
        write(path, plan)
        run_depthsplat(path, 'T4b-'+version+'-inference')
    for name in evaluation.SCENES:
        desc = desc_for(name)
        views = old_views(desc)
        masks = registered_masks(views) if version != 'original' else None
        if name == 'B2-high':
            cache = S1C/'flow'
            poses = {f:v['c2w'] for f,v in views.items()}
            neighbors = {f:sorted([g for g in FRAMES if g != f],
                         key=lambda g:(-np.linalg.norm(poses[g][:3,3]-poses[f][:3,3]),g))[:6] for f in FRAMES}
        else:
            cache = desc['confidence']/'flow'
            neighbors = {int(f): gs for f,gs in read(desc['confidence']/'flow-plan.json')['selected'].items()}
        for entry in desc['entries']:
            frame = entry['frame']
            dest = T4/version/name/f'f{frame:03d}'
            if (dest/'geometry.json').exists():
                continue
            base = load(evaluation.EXPLICIT/name/f'f{frame:03d}'/'explicit-depth.npz')
            z = base['depth']
            if version == 'flow':
                candidate, support = flow_depth(frame, views, masks, cache, neighbors[frame])
            elif version.startswith('depthsplat'):
                d = load(T4/(version+'-raw')/name/f'f{frame:03d}.npz')
                candidate, support = d['depth'], d['consistent']
            else:
                candidate, support = z, np.zeros_like(z, bool)
            replace = support & ~masks[frame] if masks is not None else support
            depth = np.where(replace, candidate, z).astype(np.float32)
            g = load(desc['geometry']/f'f{frame:03d}-center.npz')
            valid = replace | base['dominant_valid']
            focus = {str(f):focus_row(np.where(valid,depth,np.nan), subject_mask(g,f),
                                     f,None,None,'P2') for f in [1.4,5.]}
            if any(not row['success'] for row in focus.values()):
                raise ValueError(f'{name} frame {frame} {version}: empty P2 subject support')
            band = np.zeros_like(z, bool)
            for row in focus.values():
                band |= depth_band(depth, views[frame]['K'][0,0], row['focus_m'], .05/2.8)[0]
            parents = load(desc['parents'](frame))
            fg = np.load(desc['scene']/'scene-geometry.npz')['foreground_object_ids']
            a,b,supported,mapped,pvalid,audit = fit_and_map(g,depth,fg,parents)
            evaluation.save(dest/'depth.npz', depth=depth, replaced=replace, band=band, dominant_valid=valid)
            evaluation.save(dest/'alignment.npz',a=a,b=b,local_supported=supported,parent_depth_aligned=mapped,parent_valid=pvalid)
            write(dest/'focus.json',focus)
            truth, known = g['depth_axial_m'], g['center_visible']
            write(dest/'geometry.json',dict(scene=name,frame=frame,version=version,
                  depth_error={k:dict(original=stats(z,truth,m),version=stats(depth,truth,m)) for k,m in
                               [('full',known),('band',known&band)]},
                  motion_pixels=int(masks[frame].sum()) if masks is not None else None,
                  replaced_pixels=int(replace.sum()), focus=focus, placement=audit))
            print('T4_READY',version,name,frame,flush=True)
    evaluation.render(version)
    evaluation.REC = T4
    evaluation.PREFIX = 'r14'
    evaluation.evaluate(version)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('version', choices=['original','flow','depthsplat-official'])
    execute(p.parse_args().version)
