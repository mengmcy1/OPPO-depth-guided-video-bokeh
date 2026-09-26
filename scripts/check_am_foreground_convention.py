"""Check official estimated AM layers' encoding; this does not recover physical GT."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.synthetic.common import srgb_to_linear, linear_to_srgb


def metrics(error):
    mse = float(np.mean(error.astype(np.float64) ** 2))
    return {'mae': float(np.mean(np.abs(error), dtype=np.float64)),
            'mse': mse, 'psnr': float(-10 * np.log10(mse)) if mse else None}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT/'configs/am-foreground-development.json')
    p.add_argument('--report', type=Path, required=True)
    args = p.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    config = json.loads(args.config.read_text())
    excluded = set(config['prior_development_ids'])
    categories = json.loads((ROOT/'data/raw/AM-2k/am2k_split_category.json').read_text())
    rows = []
    for sample in config['samples']:
        name = sample['name']
        assert categories[name]['split'] == 'train'
        assert 'AM-2k:' + name not in excluded
        folder = ROOT/'data/raw/AM-2k/train'
        def read(role, extension, mode):
            return np.array(Image.open(folder/role/(name+extension)).convert(mode), dtype=np.float32)/255
        original = read('original', '.jpg', 'RGB')
        fg, bg = read('fg', '.png', 'RGB'), read('bg', '.jpg', 'RGB')
        alpha = read('mask', '.png', 'L')
        assert fg.shape == bg.shape == original.shape == (*alpha.shape, 3)
        soft = (alpha > .05) & (alpha < .95)
        assert soft.any(), name
        a = alpha[..., None]
        composites = {'srgb': a*fg+(1-a)*bg,
                      'linear': linear_to_srgb(a*srgb_to_linear(fg)+(1-a)*srgb_to_linear(bg))}
        row = dict(sample, shape=list(alpha.shape), soft_pixels=int(soft.sum()))
        row['reconstruction'] = {space: {'full': metrics(c-original),
                                        'soft': metrics((c-original)[soft])}
                                 for space, c in composites.items()}
        rows.append(row)
        print(f'{len(rows)}/{len(config["samples"])} {name}', flush=True)
    summary = {space: {region: {metric: float(np.mean([
        row['reconstruction'][space][region][metric] for row in rows]))
        for metric in ['mae', 'mse', 'psnr']} for region in ['full', 'soft']}
        for space in ['srgb', 'linear']}
    report = {'status': 'COLOR_CONVENTION_CHECK_COMPLETE_NOT_ESTIMATOR_SELECTION',
              'reference_status': config['reference_status'], 'config': str(args.config),
              'sample_count': len(rows), 'category_count': len({r['category'] for r in rows}),
              'summary_equal_weight_per_image': summary,
              'srgb_lower_soft_mae_count': sum(r['reconstruction']['srgb']['soft']['mae'] <
                                              r['reconstruction']['linear']['soft']['mae'] for r in rows),
              'validation_images_read': 0, 'rows': rows}
    with args.report.open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
