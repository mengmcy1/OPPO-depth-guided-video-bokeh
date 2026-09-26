"""Validate pilot/E6 inputs without changing the author's PNG8 conditioning protocol."""
import json
from pathlib import Path

import numpy as np
from PIL import Image


def discover(root, only=''):
    config = json.loads((root / 'config.json').read_text())
    ids = {s['id'] for s in config['scenes']} if isinstance(config.get('scenes'), list) else None
    candidates = [p for p in sorted(root.iterdir()) if p.is_dir() and
                  (p.name in ids if ids is not None else p.name.startswith('scene_'))]
    if only:
        requested = set(only.split(','))
        assert requested <= {p.name for p in candidates}, 'Requested scene missing from input/manifest'
        candidates = [p for p in candidates if p.name in requested]
    assert candidates, 'No input scenes found; refusing an empty successful run'
    return candidates


def validate(folder):
    meta = json.loads((folder / 'metadata.json').read_text())
    c = meta['config']
    frames = meta['frames']
    assert len(frames) == 25, 'This runner validates full 25-frame clips only'
    assert [f['index'] for f in frames] == list(range(25))
    assert [f.get('timeline_frame', f['index']) for f in frames] == list(range(25)), 'Nonconsecutive samples are not a full clip'
    rgbs, disparities = sorted((folder/'rgb').glob('*.png')), sorted((folder/'disparity').glob('*.png'))
    assert len(rgbs) == len(disparities) == 25
    errors = []
    for i, (rgb, disp, frame) in enumerate(zip(rgbs, disparities, frames)):
        assert rgb.stem == f'{i:05d}' and disp.name.startswith(f'{i:05d}_zf_')
        color = np.array(Image.open(rgb))
        png = np.array(Image.open(disp))
        depth = np.load(folder/'depth'/f'{i:05d}.npy')
        assert color.shape == (c['height'], c['width'], 3)
        assert png.shape == depth.shape == color.shape[:2] and png.dtype == np.uint8
        assert np.isfinite(depth).all() and (depth > 0).all()
        inverse = 1/depth.astype(np.float64)
        assert (inverse <= 1).all(), 'PNG8 adapter would clip physical disparity'
        focus = float(disp.stem.split('_zf_')[1])
        assert abs(focus - 1/frame['focus_distance_m']) <= 5.1e-9
        assert abs(focus - frame['focus_disparity']) <= 5.1e-9
        actual = np.abs(png.astype(float)/255-focus)*c['K']
        expected = np.abs(inverse-1/frame['focus_distance_m'])*c['K']
        error = float(np.max(np.abs(actual-expected)))
        assert error <= c['K']/510 + 1e-5
        errors.append(error)
    return dict(scene=folder.name, frames=25, resolution=[c['width'], c['height']], K=c['K'],
                disparity='inverse meters; PNG8, no per-scene normalization',
                max_coc_radius_quantization_error_px=max(errors),
                quantization_bound_px=c['K']/510,
                caveat='Author PNG8 conditioning is retained; float depth remains the physical reference')


def check_batch(ds, batch, folder):
    import torch
    files = sorted(p.name for p in (folder/'disparity').glob('*.png'))
    groups, _ = ds.group_frames_with_overlap(files)
    K = json.loads((folder/'metadata.json').read_text())['config']['K']
    for group_index, group in enumerate(groups):
        for index, name in enumerate(group):
            q = np.array(Image.open(folder/'disparity'/name)).astype(np.float32)
            focus = float(Path(name).stem.split('_zf_')[1])
            expected = (np.abs(q-focus*255)/255)*2-1
            actual = batch['coc'][group_index, index].numpy()
            assert np.allclose(actual, expected[None], atol=2e-6)
            assert float(batch['k'][group_index,index]) == K
    assert all(torch.isfinite(batch[k]).all() for k in ['pixel_values','coc','k','mpi_mask','weight'])
    return {k: list(batch[k].shape) for k in ['pixel_values','coc','k','mpi_mask','weight']}
