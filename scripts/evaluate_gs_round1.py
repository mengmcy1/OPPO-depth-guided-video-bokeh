#!/usr/bin/env python3
"""Evaluate explicit prediction/GT pairs on CPU without interrupting GPU training.

Unified metrics follow 3DGS: RGB [0,1], Gaussian-window SSIM, LPIPS VGG v0.1
with the same [0,1] call convention as the official metrics script. Native
metrics use each repository's function and its saved camera GT separately.
"""
import argparse
from collections import defaultdict
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
import lpips
from skimage import metrics as skmetrics

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent


def module_from_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def tensor(path):
    rgb = np.asarray(Image.open(path).convert('RGB')).copy()
    return torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0)/255


def native_function(method):
    paths = {'dofgs': 'DOFGS_implementation/src/metrics/__init__.py',
             'dofgaussian': 'DoF-Gaussian/metric.py', 'cocogaussian': 'CoCoGaussian/metrics.py'}
    return module_from_file('native_'+method, WORKSPACE/paths[method]).compute_img_metric if method in paths else None


def native_scores(function, prediction, truth):
    """Restore pre-channel_axis skimage API semantics, not a different metric."""
    from skimage import metrics
    original = metrics.structural_similarity
    def compatibility(*args, **kwargs):
        if kwargs.pop('multichannel', False): kwargs.setdefault('channel_axis', -1)
        kwargs.setdefault('data_range', 2.)
        return original(*args, **kwargs)
    metrics.structural_similarity = compatibility
    values = {}
    try:
        for metric in ['psnr', 'ssim', 'lpips']:
            try: values[metric] = float(function(prediction[0], truth[0], metric))
            except Exception as exc: values[metric+'_error'] = str(exc)
    finally:
        metrics.structural_similarity = original
    return values


def means(rows):
    keys = set().union(*(row.keys() for row in rows)) if rows else set()
    return {key: sum(float(row[key]) for row in rows if isinstance(row.get(key), (float, int)))/sum(isinstance(row.get(key), (float, int)) for row in rows)
            for key in keys if any(isinstance(row.get(key), (float, int)) for row in rows)}


@torch.no_grad()
def evaluate(manifest, output, regions=None):
    torch.set_num_threads(8)
    data = json.loads(manifest.read_text())
    official_loss = module_from_file('gs_ssim', WORKSPACE/'mip-splatting/utils/loss_utils.py')
    criterion = lpips.LPIPS(net='vgg', version='0.1').cpu().eval()
    spatial = lpips.LPIPS(net='vgg', version='0.1', spatial=True).cpu().eval() if regions else None
    function = native_function(data['method'])
    results = []; grouped = defaultdict(list)
    for pair in data['pairs']:
        if pair.get('role', 'test') != 'test': continue
        prediction, truth = tensor(pair['prediction']), tensor(pair['truth'])
        if prediction.shape != truth.shape:
            raise ValueError(f"Resolution mismatch for {pair['image']}: {prediction.shape} / {truth.shape}")
        error = (prediction-truth).square()
        mse = error.mean().item()
        scores = {'psnr': -10*math.log10(max(mse, 1e-16)),
                  'ssim': float(official_loss.ssim(prediction, truth)),
                  'lpips': float(criterion(prediction, truth))}
        row = dict(pair, unified=scores, resolution=list(prediction.shape[-2:][::-1]))
        native_gt = tensor(pair.get('native_truth', pair['truth']))
        native_prediction=tensor(pair.get('native_prediction',pair['prediction']))
        if pair['protocol'] in {'A', 'r0'}:
            row['native'] = native_scores(function, native_prediction, native_gt) if function else {
                'psnr': -10*math.log10(max(float((native_prediction-native_gt).square().mean()), 1e-16)),
                'ssim': float(official_loss.ssim(native_prediction, native_gt)), 'lpips': float(criterion(native_prediction, native_gt))}
        if regions and pair['protocol'].startswith('r1-'):
            mode = pair['protocol'].split('-')[-1]
            masks = np.load(regions/data['scene']/(pair['image']+'.npz'))
            _, ssim_map = skmetrics.structural_similarity(
                prediction[0].permute(1, 2, 0).numpy(), truth[0].permute(1, 2, 0).numpy(),
                channel_axis=-1, data_range=1., gaussian_weights=True, sigma=1.5,
                use_sample_covariance=False, full=True)
            lpips_map = spatial(prediction, truth).squeeze().cpu().numpy()
            pixel_mse = error.mean(1).squeeze().cpu().numpy()
            regional = {}
            for label, key in [('boundary', 'boundary'), ('in_focus', mode+'_in_focus'), ('out_of_focus', mode+'_out_of_focus')]:
                mask = masks[key].astype(bool)
                regional[label] = {'pixels': int(mask.sum()), 'fraction': float(mask.mean())}
                if mask.any():
                    regional[label].update(psnr=-10*math.log10(max(float(pixel_mse[mask].mean()), 1e-16)),
                                           ssim=float(ssim_map.mean(-1)[mask].mean()), lpips=float(lpips_map[mask].mean()))
            row['regions'] = regional
        grouped[pair['protocol']].append(row)
        results.append(row)
    summary = {protocol: {'views': len(rows), 'unified': means([r['unified'] for r in rows]),
                           'native': means([r.get('native', {}) for r in rows])}
               for protocol, rows in grouped.items()}
    for protocol, rows in grouped.items():
        if any('regions' in row for row in rows):
            summary[protocol]['regions'] = {region: means([r['regions'][region] for r in rows if 'regions' in r])
                                            for region in ['boundary', 'in_focus', 'out_of_focus']}
    report = {'method': data['method'], 'dataset': data['dataset'], 'scene': data['scene'], 'status': 'evaluated',
              'summary': summary, 'per_view': results,
              'protocol': {'psnr': 'RGB [0,1], per-image MSE then mean image PSNR',
                           'ssim': '3DGS Gaussian 11x11 sigma1.5, constants0.01/0.03',
                           'lpips': 'VGG v0.1, official 3DGS [0,1] input call convention; no extra normalize flag',
                           'native': 'official repository compute_img_metric; legacy skimage channel_axis/data_range compatibility only',
                           'ground_truth': 'unified uses canonical images_4 or resized original focus image; native uses camera GT when available',
                           'partial_scene_averages': 'computed later across completed scenes only'},
              'device': 'cpu', 'manifest': str(manifest)}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.json.tmp'); temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n'); temporary.replace(output)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--regions', type=Path)
    args = parser.parse_args(); evaluate(args.manifest, args.output, args.regions)


if __name__ == '__main__': main()
