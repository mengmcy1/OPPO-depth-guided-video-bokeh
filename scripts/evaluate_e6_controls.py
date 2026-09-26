"""Measure independent-seed noise, static flicker and pinhole edge alignment."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    x = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    assert x is not None, path
    return x.astype(np.float64) / np.iinfo(x.dtype).max


def measure(a, b):
    d = a-b
    return dict(PSNR=float(-10*np.log10(max(np.mean(d*d), 1e-16))), MAE=float(np.abs(d).mean()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--config', type=Path, default=ROOT/'configs/renderer-e6.json')
    p.add_argument('--mode', choices=['noise','edges'], required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    criteria = json.loads(a.config.read_text())['criteria']
    rows = []
    for folder in sorted((a.input/a.mode).iterdir()):
        if not folder.is_dir():
            continue
        if a.mode == 'noise':
            primary = folder/'cycles_optix_s1024_f1'
            secondary = folder/'cycles_optix_s1024_f1_seed100000'
            files = sorted(primary.glob('*.png'))
            assert len(files) == len(list(secondary.glob('*.png'))) == 25
            values, temporal, difference_temporal = [], [], []
            previous = previous_difference = None
            for file in files:
                x, y = load(file), load(secondary/file.name)
                values.append(measure(x,y))
                if previous is not None:
                    temporal.append(float(np.abs(x-previous).mean()))
                    difference_temporal.append(float(np.abs((x-y)-previous_difference).mean()))
                previous, previous_difference = x, x-y
            row = dict(scene=folder.name, independent_seed_frames=values,
                       min_seed_psnr_db=min(v['PSNR'] for v in values),
                       max_seed_mae=max(v['MAE'] for v in values),
                       mean_temporal_L1=float(np.mean(temporal)),
                       mean_seed_difference_temporal_L1=float(np.mean(difference_temporal)))
            row['independent_seed_pass'] = row['min_seed_psnr_db'] >= criteria['independent_seed_psnr_min_db'] and row['max_seed_mae'] <= criteria['independent_seed_mae_max']
            if folder.name.startswith('static'):
                row['static_flicker_pass'] = row['mean_temporal_L1'] <= criteria['static_repeated_render_mean_temporal_L1_max']
            else:
                row['note'] = 'Raw temporal L1 includes real motion; static threshold is not applied to it'
        else:
            row = dict(scene=folder.name, **measure(load(folder/'rgb/00000.png'),load(folder/'pinhole_optix/00000.png')))
            row['alignment_pass'] = row['PSNR'] >= criteria['alignment_pinhole_psnr_min_db']
            row['note'] = 'Pinhole agreement alone does not validate bokeh edges; inspect matched bokeh views and source alpha'
        rows.append(row)
    assert rows
    with a.output.open('x') as f:
        json.dump(dict(criteria=criteria, color_space='sRGB [0,1]', rows=rows), f, indent=2)
    for row in rows:
        print(json.dumps({k:v for k,v in row.items() if k!='independent_seed_frames'}), flush=True)


if __name__ == '__main__':
    main()
