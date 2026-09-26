"""Evaluate edge-spread radius against a predeclared Cycles camera relationship."""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from synthetic.common import srgb_to_linear


def variance(file, radius):
    image = cv2.imread(str(file), cv2.IMREAD_UNCHANGED)
    assert image is not None, file
    linear = srgb_to_linear(image.astype(np.float64) / np.iinfo(image.dtype).max)
    h, w = image.shape[:2]
    # Average many rows; remove non-edge regions so quantization cannot pollute moments.
    line = linear[h//4:3*h//4].mean(axis=(0, 2))
    half = int(np.ceil(radius + 6))
    gradient = np.diff(line)[w//2-half:w//2+half]
    x = np.arange(len(gradient), dtype=float)
    weight = gradient / gradient.sum()
    mean = np.sum(x*weight)
    return float(np.sum((x-mean)**2*weight))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--config', type=Path, default=ROOT / 'configs/renderer-e6.json')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    cfg = json.loads(a.config.read_text())
    rows = []
    for case in cfg['calibration_cases']:
        folder = a.input / case['id']
        nominal = case['K']*abs(1/case['object_depth_m']-1/case['focus_distance_m'])
        pinhole = variance(folder/'pinhole_optix/00000.png', nominal)
        row = dict(case, nominal_radius_px=nominal)
        for tag, name in [('cycles_optix_s1024_f1','physical'), ('cycles_optix_s1024_f0.945244','historical'), ('bokeh_dr','dr')]:
            file = folder/tag/'00000.png'
            assert file.exists(), file
            # Uniform circular aperture: one-dimensional marginal variance = R^2/4.
            measured = 2*np.sqrt(max(0., variance(file, nominal / .945244)-pinhole))
            row[name+'_radius_px'] = float(measured)
            row[name+'_error_px'] = float(measured-nominal)
        tolerance = max(cfg['criteria']['calibration_radius_tolerance_px'], cfg['criteria']['calibration_radius_tolerance_relative']*nominal)
        row['tolerance_px'] = tolerance
        row['physical_status'] = ('pass' if abs(row['physical_error_px']) <= tolerance else 'fail') if nominal >= 2 else ('resolution_limited' if nominal else ('pass' if row['physical_radius_px'] <= cfg['criteria']['in_focus_excess_radius_max_px'] else 'fail'))
        rows.append(row)
    result = dict(method='2*sqrt(edge derivative variance minus same-case pinhole variance), linear light, central rows; diagnostic effective disk radius',
                  criteria=cfg['criteria'], cases=rows,
                  scope='Calibration only; does not establish dynamic stability or complete E6',
                  physical_failures=[r['id'] for r in rows if r['physical_status']=='fail'])
    with a.output.open('x') as f:
        json.dump(result, f, indent=2)
    for row in rows:
        print(row['id'], 'nominal',round(row['nominal_radius_px'],4), 'physical',round(row['physical_radius_px'],4), row['physical_status'])


if __name__ == '__main__':
    main()
