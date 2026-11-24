"""Expanded B/C optical labels, with GT confined to render supervision."""
from functools import lru_cache
import hashlib

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from route3_ovg_1b_optical_data import place, render_f2
from route3_ovg_card import lens_offsets
from route3_v2data_B_inputs import cameras, observed_rgb
from route3_v2data_loader import FORMAL_DATA, ROOT, legacy, read
from src.aperture_renderer import GaussianScene


def linear_reference(spec):
    field = 'rgb_linear' if spec['tiers'] == 'B' else 'optical_rgb'
    return legacy.load(spec['truth'], (field,))[field].astype(np.float32)


@lru_cache(maxsize=2)
def trace_b(unit, frame, focus, fstop):
    from route3_v2data_B_labels import StaticWorld

    world = StaticWorld(unit/'labels/scene-geometry.npz', 512)
    fields, _ = world.aperture(frame, 16, fstop, focus, track=True)
    return {k: fields[k] for k in ('depth_axial_m', 'known', 'hidden', 'object_id',
                                  'aperture_xy', 'recoverable', 'first_visible_other_frame')}


def ray_fields(spec):
    if spec['tiers'] == 'B':
        keys = ('depth_axial_m', 'known', 'hidden', 'object_id', 'aperture_xy',
                'recoverable', 'first_visible_other_frame')
        d = legacy.load(spec['label'], keys) if spec['label'] else trace_b(
            spec['unit'], spec['frame'], spec['focus'], spec['fstop'])
        geom = spec['unit']/'labels/scene-geometry.npz'
        ids = legacy.load(geom, ('foreground_object_ids',))['foreground_object_ids']
        return dict(d, depth=d['depth_axial_m'], ids=d['object_id'],
                    foreground=np.isin(d['object_id'], ids))
    keys = ('depth', 'light_valid', 'center_in_view', 'hidden', 'surface_id', 'foreground_hit',
            'recoverable', 'world', 'visible_source_frames')
    d = legacy.load(spec['label'], keys)
    return dict(d, ids=d['surface_id'], known=d['light_valid'] & d['center_in_view'],
                foreground=d['foreground_hit'], aperture_xy=lens_offsets()[:, :2])


def occupancy(rays, target, size, generator, ray_count):
    device = target['alpha'].device
    indices = torch.randperm(len(rays['depth']), generator=generator, device=device)[:ray_count].cpu().numpy()
    z, known, hidden, foreground = (torch.as_tensor(rays[k][indices], device=device)
                                    for k in ('depth', 'known', 'hidden', 'foreground'))
    z = z.float()
    pure = target['valid_alpha'][0] & ~target['depth_boundary'][0] & (target['alpha'][0] >= .9999)
    good = known & pure[None] & torch.isfinite(z) & (z > 0)
    reveal, block = good & hidden, good & ~hidden & foreground
    if pure.shape != size:
        reveal = F.adaptive_avg_pool2d(reveal.float(), size) == 1
        block = F.adaptive_avg_pool2d(block.float(), size) == 1
        z = F.adaptive_avg_pool2d(z.nan_to_num(), size)
    cutoff = torch.where(block, z*1.005, z).nan_to_num()
    offsets = torch.as_tensor(rays['aperture_xy'][indices], dtype=torch.float32, device=device)
    return offsets, cutoff, reveal, block


def project(points, pose, intrinsics):
    camera = (points-pose[:3,3]) @ pose[:3,:3]
    q = camera @ intrinsics.T
    return q[:,:2]/np.maximum(q[:,2:3], 1e-6), camera[:,2]


def witness_colors(points, witness, intrinsics, poses, image):
    rgb = np.zeros((len(points),3), np.float32)
    kept = np.zeros(len(points), bool)
    for index in np.unique(witness):
        ids = np.flatnonzero(witness==index)
        uv, depth = project(points[ids], poses[index], intrinsics[index])
        src = image(int(index))
        h,w = src.shape[:2]
        good = (depth>0)&(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
        xy = np.floor(uv[good]).astype(np.int64)
        rgb[ids[good]] = src[xy[:,1],xy[:,0]]
        kept[ids[good]] = True
    return rgb, kept


def hidden_parents(spec, rays, reference):
    use = rays['recoverable']
    if spec['tiers'] == 'B':
        unit = spec['unit']
        frame = spec['frame']-1
        intr, poses = cameras(unit)
        intrinsics = np.broadcast_to(intr,(len(poses),3,3))
        lens, y, x = np.nonzero(use)
        depths = rays['depth'][use].astype(np.float32)
        grid = np.stack((x+.5,y+.5,np.ones_like(x)), -1) @ np.linalg.inv(intr).T
        xy = rays['aperture_xy'][lens] * (1-depths[:,None]/spec['focus'])
        points = (grid*depths[:,None]+np.column_stack((xy,np.zeros(len(xy))))) @ poses[frame,:3,:3].T + poses[frame,:3,3]
        witness = rays['first_visible_other_frame'][use].astype(np.int64)-1
        def image(index):
            return observed_rgb(unit,index+1)
    else:
        frame = spec['index']
        meta = read(spec['manifest'])
        rows = meta['rows']
        poses = np.array([r['c2w'] for r in rows], np.float32)
        intrinsics = np.array([r['K'] for r in rows], np.float32)
        points = rays['world'][use].astype(np.float32)
        bits = rays['visible_source_frames'][use].astype(np.int64)
        witness = np.log2(bits&-bits).astype(np.int64)
        def image(index):
            src = cv2.imread(str(ROOT/rows[index]['image']))
            return legacy.srgb_to_linear(src[...,::-1].astype(np.float32)/255)
    if not use.any():
        return None
    h,w = reference['gt'].shape
    uv, depth = project(points,poses[frame],intrinsics[frame])
    good = (depth>0)&(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)&(witness>=0)&(witness<len(poses))
    # Bound placement to one point per projected pixel and surface.
    pixel = np.floor(uv[good]).astype(np.int64)
    ids = rays['ids'][use][good].astype(np.int64)
    packed = np.stack((pixel[:,0],pixel[:,1],ids),axis=-1)
    unique = np.flatnonzero(good)[np.unique(packed,axis=0,return_index=True)[1]]
    colors, visible = witness_colors(points[unique],witness[unique],intrinsics,poses,image)
    return dict(uv=uv[unique].astype(np.float32),depth=depth[unique].astype(np.float32),
                recoverable=visible,colors_linear=colors)


def hidden_substitute(spec, rays, reference, visible_z, K, *, parents=None):
    if parents is None:
        parents = hidden_parents(spec, rays, reference)
        if parents is None:
            return None, dict(eligible=0, placed=0)
    h,w = reference["gt"].shape
    background = reference['valid'] & ~np.isin(reference['object_id'],reference['foreground'])
    return place(parents, reference['gt'], background, visible_z, K, w)


def f2_entry(unit, frame, directory=None):
    root = directory if directory is not None else FORMAL_DATA/'derived-f2'
    return root/unit.name/f'f{frame:03d}'


@lru_cache(maxsize=4)
def read_f2(entry, unit):
    if not (entry/'manifest.json').is_file():
        raise FileNotFoundError(f"B optical frame has no derived LaMa F2 cache: {entry.name}")
    record = read(entry/'manifest.json')
    geometry_sha = hashlib.sha256((unit/'manifest.json').read_bytes()).hexdigest()
    if record['geometry_manifest_sha256'] != geometry_sha:
        raise ValueError(f"derived LaMa F2 cache points to a different geometry release: {unit.name}")
    fields = legacy.load(entry/'background.npz', ('background_depth','fill_rgb_linear'))
    cloud = torch.load(entry/'a-color.pt', map_location='cpu', weights_only=True)
    return fields, cloud


@torch.no_grad()
def observed_background(spec, inputs, focus, offsets):
    """Use the fixed derived legacy F2 lookup for B optical training only."""
    if not spec or spec['tiers'] != 'B':
        return None
    entry = f2_entry(spec['unit'], spec['frame'], spec.get('f2_root'))
    fields, cloud = read_f2(entry, spec['unit'])
    K = inputs['K'][0]
    scene = GaussianScene(**{key:value.to(K.device) for key,value in cloud.items()})
    return render_f2(scene, fields['background_depth'], fields['fill_rgb_linear'],
                     K, focus, offsets, tuple(inputs['rgb'].shape[-2:]))
