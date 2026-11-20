"""Validate DL3DV view roles and preserve the existing L_mv interface."""
import fcntl
from pathlib import Path

import cv2
import numpy as np

import route3_ovg_1b_dl3dv as dl3dv
from route3_ovg_r2_priors import resize_inputs
from route3_ovg_subpixel import save_npz
from route3_v2data_G_common import ROOT, SPLITS, check_space, read, write
from route3_v2data_R_acquire import OUT


def geometric_common(source_depth, target_depth, ka, kb, a, b):
    height, width = source_depth.shape
    y, x = np.mgrid[:height, :width].astype(np.float32)
    rays = np.stack((x+.5, y+.5, np.ones_like(x)), axis=-1) @ np.linalg.inv(ka).T
    world = (rays * source_depth[..., None]) @ a[:3, :3].T + a[:3, 3]
    camera = (world - b[:3, 3]) @ b[:3, :3]
    projected = camera @ kb.T
    uv = (projected[..., :2] / np.maximum(camera[..., 2:], 1e-6)).astype(np.float32) - .5
    target = cv2.remap(target_depth, uv[..., 0], uv[..., 1], cv2.INTER_NEAREST)
    z = camera[..., 2]
    return ((uv[..., 0] >= 0) & (uv[..., 0] < width-1) & (uv[..., 1] >= 0) & (uv[..., 1] < height-1) &
            (z > 0) & (target > 0) & (np.abs(target-z) <= .04*z))


def precalculate(net, path):
    with (path.parent/'validation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not (path.parent/'readiness.json').exists():
            verify(net, path)


def verify(net, path):
    check_space(OUT, 16 * 2**20)
    meta = read(path)
    inputs, heldout = set(meta['inputs']), set(meta['heldout'])
    if inputs & heldout or inputs | heldout != set(range(len(meta['rows']))) or len(meta['rows']) != 8:
        raise ValueError('R input and held-out views violate the existing eight-view partition')
    result = dl3dv.validate(net, path)
    if not result['passed']:
        write(path.parent/'readiness.json', dict(passed=False, reason='No static common view pair at the fixed threshold'))
        return
    _, intrinsics, poses = resize_inputs(path, meta['rows'])
    depths = []
    for i, row in enumerate(meta['rows']):
        raw = cv2.imread(str(ROOT/row['image']))
        if raw is None or max(raw.shape[:2]) != 480:
            raise ValueError('DL3DV frame is not the native 480P source used by the existing loader')
        with np.load(path.parent/'prior'/f'f{i:03d}.npz') as value:
            if value['depth'].dtype != np.float16:
                raise ValueError('R target geometry prior is not float16')
            depths.append(value['depth'].astype(np.float32))
        observed = path.parent/'observed-prior'/f'f{i:03d}.npz'
        if (i in inputs) != observed.exists():
            raise ValueError('R observed-prior files violate the input-view partition')
        if i in inputs:
            with np.load(observed) as value:
                if value['depth'].dtype != np.float16 or value['static'].dtype != np.bool_:
                    raise ValueError('R observed prior or static mask dtype mismatch')
    common_pixels = eligible_pixels = total_pixels = pairs = 0
    for row in result['accepted']:
        i = row['index']
        neighbors = row['heldout_neighbors']
        if i not in inputs or not set(neighbors) <= heldout:
            raise ValueError('R scoring pairs cross input/held-out roles')
        with np.load(ROOT/row['mask']) as accepted:
            if accepted['neighbors'].tolist() != neighbors:
                raise ValueError('R pair index disagrees with the L_mv mask')
            joint = accepted['mask']
        common = np.stack([geometric_common(depths[i], depths[j], intrinsics[i], intrinsics[j],
                                             poses[i], poses[j]) for j in neighbors])
        # Old runs compared unquantized or quantized depths depending on their cache.
        # Reusing the original joint labels keeps every old positive pixel valid.
        common |= joint
        save_npz(path.parent/'common-visible'/f'f{i:03d}.npz', neighbors=np.array(neighbors), mask=common)
        common_pixels += int(common.sum())
        eligible_pixels += int(joint.sum())
        total_pixels += common.size
        pairs += len(neighbors)
    write(path.parent/'readiness.json', dict(passed=True, scene=meta['scene'], frame_count=len(meta['rows']),
          input_count=len(inputs), heldout_count=len(heldout), accepted_pairs=pairs,
          common_pixels=common_pixels, static_common_pixels=eligible_pixels, pair_pixels=total_pixels,
          common_visible_fraction=common_pixels/total_pixels,
          static_common_fraction=eligible_pixels/total_pixels,
          masks=dict(common='common-visible', l_mv='static-common', input_static='observed-prior')))


def inventory(role):
    if role not in SPLITS:
        raise ValueError(role)
    for unit in read(OUT/'manifest.json')['units']:
        if unit['split'] == role:
            path = OUT / unit['window']
            for row in read(path.parent/'visibility.json')['accepted']:
                yield ('R', str(path), row['index'])


def training_input(manifest, index, device):
    return dl3dv.training_input(str(manifest), index, device)


if __name__ == '__main__':
    cv2.setNumThreads(1)
    for retained in sorted((OUT/'windows').glob('*/retained.json')):
        precalculate(None, retained.parent/'manifest.json')
