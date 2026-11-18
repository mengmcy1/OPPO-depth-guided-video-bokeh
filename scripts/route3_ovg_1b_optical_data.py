"""Render-only hidden substitutes and separation of pinned optical targets."""
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch

from route3_ovg_1b_cards import clip_parameters, shade
from route3_ovg_1b_common import CONFIG, OUT, read
import route3_ovg_data as data
from src.aperture_renderer import GaussianScene, aperture_disk
from src.footprint_order import apply_footprint_order
from route3_s2a_candidates import local_placement
from src.ovg_optics import raster


def geometry_root(scene):
    return data.PREVIOUS / 'truth' / scene


@lru_cache(maxsize=48)
def synthetic_optical(item):
    tier, key, index = item
    if tier == 'B':
        frame = data.TRAIN_FRAMES[index]
        base = geometry_root(key)
        work = data.PREVIOUS / 'geometry/original' / key
        cams = read(work / 'prepared.json')
        entry = next((row for row in cams['rows'] if row['frame'] == frame), None)
        if entry is None:
            return []
        specs = []
        for case in entry['cases']:
            if case['fstop'] != 1.4:
                continue
            cid = case['id']
            labels = base / 'geometry' / (cid + '-K16.npz')
            truth = base / 'dof' / (cid + '.exr')
            if labels.exists() and truth.exists():
                specs.append(dict(focus=case['nominal_focus_m'], fstop=case['fstop'], label=labels,
                                  truth=truth, entry=entry, tiers='B'))
        return specs
    if tier == 'C':
        meta = read(key)
        if 'rectangles' not in meta:
            return []
        specs = []
        for row in meta.get('K16_aperture', []):
            if row['frame'] != index:
                continue
            focus = row['focus_m']
            filename = f'f{index:02d}-N1.4-F{focus:g}.npz'
            label = Path(key).parent / 'K16-aperture' / filename
            truth = OUT / 'C/optical' / meta['scene'] / filename
            if label.exists():
                specs.append(dict(focus=focus, fstop=1.4, label=label, truth=truth,
                                  tiers='C', manifest=key, index=index))
        return specs
    return []


def read_optical(spec):
    file = spec['truth']
    if not file.is_file():
        return None
    if spec['tiers'] == 'C':
        return data.load(file, ('optical_rgb',))['optical_rgb']
    z = cv2.imread(str(file), cv2.IMREAD_UNCHANGED)
    if z is None:
        raise FileNotFoundError(file)
    return z[..., :3][..., ::-1].copy()


@lru_cache(maxsize=20)
def load_ray_labels(label, tier):
    if tier == 'B':
        d = data.load(label, ('depth_axial_m', 'known', 'hidden', 'object_id', 'aperture_xy'))
        return d['depth_axial_m'], d['known'], d['hidden'], d['object_id'], d['aperture_xy']
    d = data.load(label, ('depth', 'light_valid', 'center_in_view', 'hidden', 'surface_id'))
    return d['depth'], d['light_valid'] & d['center_in_view'], d['hidden'], d['surface_id'], None


def occupancy_labels(spec, h, w, target, generator):
    label, tier = spec['label'], spec['tiers']
    depth, known, hidden, ids, pupil = load_ray_labels(label, tier)
    device = target['alpha'].device
    positions = torch.randperm(len(depth), generator=generator, device=device)[:CONFIG['optical_train']['occ_ray_count']]
    radius = .05 / (2 * spec['fstop'])
    offsets = (aperture_disk(len(depth), radius, device=device, dtype=torch.float32) if pupil is None
               else torch.from_numpy(pupil.astype(np.float32)).to(device))
    indices = positions.cpu().numpy()
    z, good, invisible, obj = (torch.as_tensor(v[indices], device=device)
                               for v in (depth, known, hidden, ids))
    if tier == 'B':
        scene_mesh = geometry_root(spec['entry']['scene']) / 'scene-geometry.npz'
        occluders = data.load(scene_mesh, ('foreground_object_ids',))['foreground_object_ids']
        fore = torch.isin(obj, torch.as_tensor(occluders, device=device).to(obj.dtype))
    else:
        fore = (obj > 0) & (obj < 3)
    pure = ~target['depth_boundary'][0] & (target['alpha'][0] >= .9999) & target['valid_alpha'][0]
    good = good & pure[None] & torch.isfinite(z) & (z > 0)
    reveal = good & invisible
    block = good & ~invisible & fore
    if pure.shape != (h, w):
        # Keep the coarse label only if each native ray in its footprint has
        # the same target class. Any mixed center-surface pixel excludes it.
        pool = torch.nn.functional.adaptive_avg_pool2d
        reveal = pool(reveal.float(), (h, w)) == 1
        block = pool(block.float(), (h, w)) == 1
        z = pool(z.nan_to_num(), (h, w))
    # Include the front surface in its own controlled cutoff. The clipped
    # depth operation stays strict and all coordinates remain axial meters.
    cutoff = torch.where(block, z*(1 + .005), z).nan_to_num()
    return offsets[positions], cutoff.float(), reveal, block


@lru_cache(maxsize=16)
def b_oracle(scene, frame):
    root = geometry_root(scene)
    parents = data.load(root / 'oracle' / f'f{frame:03d}.npz')
    center = data.load(root / 'geometry' / f'f{frame:03d}-center.npz',
                       ('depth_axial_m', 'center_visible', 'object_id'))
    occluders = data.load(root / 'scene-geometry.npz', ('foreground_object_ids',))['foreground_object_ids']
    background = center['center_visible'] & ~np.isin(center['object_id'], occluders)
    return parents, center['depth_axial_m'], background


def rescale_nearest(value, size):
    return cv2.resize(value, size[::-1], interpolation=cv2.INTER_NEAREST_EXACT)


def place(parents, front_gt, background, visible_z, K, reference_width):
    """Local disparity fit and fixed footprint order for this visible geometry."""
    own = visible_z.detach().float().cpu().numpy()
    h, w = own.shape
    scale = w / reference_width
    if front_gt.shape != own.shape:
        front_gt = rescale_nearest(front_gt, (h, w))
        background = rescale_nearest(background.astype(np.uint8), (h, w)).astype(bool)
    a, b, _, valid = local_placement(front_gt, own, background, radius=max(1, round(24 * scale)),
                                     return_valid=True)
    uv = parents['uv'] * scale
    x = np.floor(uv[:, 0]).astype(int)
    y = np.floor(uv[:, 1]).astype(int)
    inside = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    x, y = x.clip(0, w-1), y.clip(0, h-1)
    inverse = a[y, x] / parents['depth'] + b[y, x]
    keep = inside & parents['recoverable'] & valid[y, x] & np.isfinite(inverse) & (inverse > 1e-4)
    if not keep.any():
        return None, {'eligible': int(parents['recoverable'].sum()), 'placed': 0, 'moved': 0,
                      'shrunk': 0, 'dropped': 0}
    tensor = lambda v: torch.as_tensor(v, dtype=K.dtype, device=K.device)
    uv, depth = tensor(uv[keep]), tensor(1/inverse[keep])
    if scale == 1:
        # Reuse the reference representation for the native matched evaluation.
        from route3_explicit_layer import PROFILE, sparse_pixel_layer
        t = dict(uv=uv.cpu().numpy(), depth=depth.cpu().numpy(),
                 colors_linear=parents['colors_linear'][keep])
        scene = sparse_pixel_layer(t, np.ones(len(uv), bool), K, torch.eye(4, device=K.device),
                                   PROFILE, K.device)
    else:
        xy = torch.cat((uv, torch.ones_like(depth[:, None])), -1)
        xyz = (xy @ torch.linalg.inv(K).T) * depth[:, None]
        q = torch.zeros((len(xyz), 4), dtype=K.dtype, device=K.device)
        q[:, 0] = 1.
        sigma = depth[:, None] * .7 / K.diag()[:2]
        scene = GaussianScene(xyz, q, torch.cat((sigma, depth[:, None] * 2e-4), -1),
                              torch.full_like(depth, .999), tensor(parents['colors_linear'][keep]),
                              torch.ones_like(depth, dtype=torch.long))
    result, stats = apply_footprint_order(scene, visible_z.detach(), K)
    stats.update(eligible=int(parents['recoverable'].sum()), placed=int(keep.sum()))
    return result, stats


@lru_cache(maxsize=24)
def c_oracle(label, manifest, index):
    from route3_ovg_r2_cards import cast
    from route3_ovg_1b_cards import camera_rays

    arrays = data.load(label, ('world', 'recoverable', 'surface_id'))
    selected = arrays['recoverable']
    meta = read(manifest)
    rect, holes, origin, K, materials = clip_parameters(meta, index)
    pts = arrays['world'][selected]
    cam = pts - origin
    proj = cam @ K.T
    uv = proj[:, :2] / np.maximum(proj[:, 2:], 1e-6)
    flat = np.floor(uv[:, 1]).astype(np.int64) * 256 + np.floor(uv[:, 0]).astype(np.int64)
    # Keep one representative per same surface and center pixel.
    unique = np.unique(flat * 4 + arrays['surface_id'][selected], return_index=True)[1]
    parents = dict(uv=uv[unique].astype(np.float32), depth=cam[unique, 2],
                   recoverable=np.ones(len(unique), bool),
                   colors_linear=shade(arrays['surface_id'][selected][unique], pts[unique], materials))
    z, ids, _, _ = cast(camera_rays(K, 256, 256, .5, .5), origin, rect, holes)
    return parents, z, ids == 3


def hidden_substitute(spec, z, K):
    if spec['tiers'] == 'B':
        params = b_oracle(spec['entry']['scene'], spec['entry']['frame'])
        return place(*params, z, K, 512)
    params = c_oracle(spec['label'], spec['manifest'], spec['index'])
    return place(*params, z, K, 256)


def background_samples(spec, K, focus, offsets, size):
    """B's input-only F2 fallback evaluated at identically shifted rays."""
    if spec is None or spec['tiers'] != 'B':
        return None
    from route3_s2a_baselines import scene
    e = spec['entry']
    directory = data.PREVIOUS / 'geometry/original' / e['scene']
    A = scene(directory / 'visible' / f'f{e["frame"]:03d}.pt')
    p = data.load(e['input_npz'], ('background_depth',))
    h, w = size
    bg = cv2.resize(p['background_depth'], (w, h), interpolation=cv2.INTER_NEAREST_EXACT)
    image = data.load(e['fill'], ('fill_rgb_linear',))['fill_rgb_linear']
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    ten = lambda a: torch.tensor(a, device=K.device, dtype=K.dtype)
    bg, image = ten(bg), ten(image)
    bg = torch.where(torch.isfinite(bg) & (bg > 0), bg, 1e4)
    y, x = torch.meshgrid(torch.arange(h, device=K.device)+.5, torch.arange(w, device=K.device)+.5,
                          indexing='ij')
    uv = torch.stack((x, y), dim=-1)
    shifts = offsets[:, None, None] * K.diag()[:2] * (bg.reciprocal() - 1 / focus)[None, ..., None]
    grid = 2 * (uv[None] + shifts) / K.new_tensor([w, h]) - 1
    src = image.permute(2, 0, 1)[None].expand(len(offsets), -1, -1, -1)
    fill = torch.nn.functional.grid_sample(src, grid, padding_mode='border', align_corners=False).permute(0, 2, 3, 1)
    with torch.no_grad():
        return raster(A, K, focus, offsets, h, w, fill)
