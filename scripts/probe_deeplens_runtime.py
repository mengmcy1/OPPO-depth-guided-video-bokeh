"""CPU-only full-resolution DeepLens runtime check; no benchmark control mapping is frozen."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from baselines.deeplens import DeepLensExternal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='NPZ with official BGR/INTER_AREA preprocessing')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert os.environ.get('TF_ENABLE_ONEDNN_OPTS') == '0'
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'result.json').exists()
    input_path = ROOT / args.input
    paths = sorted(input_path.glob('*.npz')) if input_path.is_dir() else [input_path]
    assert paths
    model = DeepLensExternal(ROOT.parent / 'DeepLens', ROOT / 'models/DeepLens/model')
    print('LOADED', model.audit, flush=True)
    if input_path.is_dir():
        rows = []
        for path in paths:
            with np.load(path) as inputs:
                low, high, d = inputs['bgr_low'], inputs['bgr_high'], inputs['disparity']
                focus, aperture, K = float(inputs['focus']), float(inputs['aperture']), int(inputs['K'])
            assert high.shape == (576, 1024, 3) and low.shape == (288, 512, 3)
            assert aperture == K / 32 and K in (4, 8, 16, 32)
            start = time.monotonic()
            value, audit = model.predict(low, high, d, focus, aperture)
            row = dict(input_file=str(path.relative_to(ROOT)), focus=focus, aperture=aperture, K=K,
                       forward_seconds=time.monotonic() - start, result=audit)
            with (out / (path.stem + '-bgr.npz')).open('xb') as f:
                np.savez_compressed(f, output=value)
            rows.append(row)
            print('FRAME', path.stem, row['forward_seconds'], flush=True)
        report = dict(status='PROVISIONAL_EXTERNAL_GEOMETRY_40_KEYFRAMES_CPU_COMPLETE', load=model.audit,
            rows=rows, frames=len(rows), threads=8,
            scope='Proposed raw disparity / K32 engineering candidate only, not a frozen adapter or paper reproduction. No DepthNet/mask, no metrics, no GPU.')
        with (out / 'result.json').open('x') as f:
            json.dump(report, f, indent=2, allow_nan=False)
            f.write('\n')
        print('COMPLETE', len(rows), flush=True)
        return
    inputs = np.load(input_path)
    low, high, d = inputs['bgr_low'], inputs['bgr_high'], inputs['disparity']
    focus = float(inputs['focus'])
    assert high.shape == (576, 1024, 3) and low.shape == (288, 512, 3)
    # Fixed public demo midpoint; intentionally independent of manifest K.
    start = time.monotonic()
    value, audit = model.predict(low, high, d, focus, aperture=0.5)
    seconds = time.monotonic() - start
    with (out / 'output_bgr_float.npy').open('xb') as f:
        np.save(f, value)
    report = dict(status='EXTERNAL_LENSBLUR_GUIDED_SR_CPU_FORWARD_PASSED', load=model.audit,
        input_file=str(args.input), focus=focus, aperture=0.5,
        forward_seconds=seconds, result=audit, threads=8,
        scope='Runtime feasibility only; public demo aperture0.5, no K mapping chosen, no metrics, no DepthNet or mask. Existing GPU jobs untouched.')
    with (out / 'result.json').open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write('\n')
    print('COMPLETE', report, flush=True)


if __name__ == '__main__':
    main()
