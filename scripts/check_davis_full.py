"""Read-only checks of all frozen DAVIS outputs before encoding; never rerun a model."""
import argparse
import fcntl
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from prepare_davis_v2 import write_new

ROOT = Path(__file__).resolve().parents[1]
METHODS = ['any2bokeh', 'mpib', 'bokehme', 'drbokeh', 'bokehdiff', 'deeplens']


def read(path):
    return json.loads(path.read_text())


def check_method(method, base=None):
    p = read(ROOT / 'configs/davis-baselines.json')
    c = read(ROOT / p['data_protocol'])
    base = ROOT / p['output'] if base is None else Path(base)
    out = base / method
    names = sorted(s['id'] for s in read(ROOT / c['derived'] / 'source-manifest.json')['sequences'])
    order = p['preflight_sequences'] + [s for s in names if s not in p['preflight_sequences']]
    assert sorted(d.name for d in (out / 'predictions').iterdir() if d.is_dir()) == names
    rng = []; ranges = []; rows = []; coverage = []; radii = []; empty_masks = []; full_masks = []
    if method != 'deeplens':
        import torch
        saved = read(out / 'protocol.json')
        assert saved['configuration'] == p and saved['data_protocol'] == c and saved['scene_order'] == order
    else:
        assert read(out / 'complete.json')['frames'] == 1250
        assert read(out / 'protocol.json')['selected_aperture'] == .5
    for index, name in enumerate(order):
        pred = out / 'predictions' / name
        files = sorted(pred.glob('*.png'))
        assert [f.name for f in files] == [f'{i:05d}.png' for i in range(25)]
        geometry = read(ROOT / c['derived'] / 'sequences' / name / 'geometry.json')
        if method != 'deeplens':
            audit = out / 'audit' / name; done = read(audit / 'complete.json')
            assert done['scene'] == name and done['frames'] == 25 and done['finite']
            state = torch.load(audit / 'rng-after.pt', map_location='cpu', weights_only=True)
            assert state['next_scene_index'] == index + 1
            assert all(state[k].dtype == torch.uint8 and state[k].ndim == 1 and state[k].numel() > 0 for k in ['cpu', 'cuda'])
            rng.append(dict(scene=name, next_scene_index=index + 1))
            if method == 'any2bokeh':
                a = read(audit / 'input.json')
                assert not a['GT_mask_argument_supplied'] and a['author_decoder_identity_VAE_index_check']
                assert sum(a['decoder_source_indices'], []) == list(range(25))
        for t, path in enumerate(files):
            with Image.open(path) as im:
                assert im.mode == 'RGB' and im.size == (1024, 576); im.verify()
            with Image.open(path) as im: pixels = np.asarray(im)
            assert pixels.dtype == np.uint8 and pixels.max() > pixels.min()
            if method == 'deeplens':
                native = out / 'native' / name
                r = read(native / f'{t:05d}.json')
                assert r['sequence'] == name and r['frame'] == t and not r['metric_selection']
                a = next(x for x in r['apertures'] if x['aperture'] == .5)
                assert a['signed_error'] <= 1e-6 and a['depth_repeat_bitwise_equal']
                assert np.isfinite(a['raw_range']).all()
                d = np.load(native / 'depth' / f'{t:05d}.npy')
                assert d.shape == (288, 512) and np.isfinite(d).all()
                m = cv2.resize(cv2.imread(str(ROOT / c['derived'] / 'eval' / name / 'mask' / f'{t:05d}.png'), 0), (512, 288), interpolation=cv2.INTER_NEAREST) > 128
                y, x = np.unravel_index(np.argmin(np.where(m, np.abs(d - d[m].mean()), np.inf)), d.shape)
                assert [int(x), int(y)] == r['focus_pixel_low_xy']
                assert float(d[y, x]) == r['focus_depth'] == a['focal_depth']
                assert pred.resolve() == (native / 'aperture-0.5').resolve()
            else:
                r = done['outputs'][t] if method == 'any2bokeh' else read(audit / f'{t:05d}.json')
                g = geometry['frames'][t]
                assert g['frame'] == r['frame'] == t and r['focus'] == g['focus'] and r['K'] == 16
                assert float(Path(g['file']).stem.split('_zf_')[1]) == r['focus']
                stats = r if method == 'any2bokeh' else r['output']
                assert np.isfinite(stats['raw_range']).all() and stats['shape'] == [576, 1024, 3]
                assert [int(pixels.min()), int(pixels.max())] == stats['PNG_range']
                if method != 'any2bokeh':
                    assert r['finite'] and not r['GT_mask_argument_supplied']
                if method in ['bokehme', 'bokehdiff']:
                    assert abs(r['effective_K'] - 16 / r['span']) < 1e-10
                    ranges.append(r['effective_K'])
                if method == 'mpib': assert r['gamma'] == 2.2 and not r['disparity_normalization']
                if method == 'bokehme': assert r['gamma'] == 2.2
                if method == 'bokehdiff':
                    assert r['bokehdiff']['seed'] == 42 and not r['bokehdiff']['reset_global_seed']
                    assert r['geometry']['candidate'] == 'normalized'
                if method == 'drbokeh':
                    q = r['full_public']
                    assert q['calls']['LDF'] == 1 and q['calls']['LaMa'] == 2 and q['calls']['MiDaS'] == 0
                    assert r['geometry']['signed_CoC_error_float64'] <= 1e-10
                    coverage.append(q['LDF_mask']['coverage'])
                    if q['LDF_mask']['threshold_empty']: empty_masks.append([name, t])
                    if q['LDF_mask']['threshold_full']: full_masks.append([name, t])
                    radii.extend(v['actual_radius_max'] for v in q['actual_layers'].values())
        rows.append(dict(scene=name, frames=25, PNG_RGB_size_order_passed=True))
    result = dict(status='FULL_50_INTEGRITY_PASSED', method=method, scenes=50, frames=1250,
        finite_evidence='Float outputs were checked during inference; persisted raw ranges and PNG integrity rechecked here.',
        focus_control='native DepthNet foreground-matched pixel / aperture0.5' if method == 'deeplens' else 'VDA filename focus / K16',
        preflight_reused_scenes=p['preflight_sequences'], rng_boundaries=rng, rows=rows)
    if ranges: result['effective_K_range'] = [min(ranges), max(ranges)]
    if coverage: result.update(LDF_coverage_range=[min(coverage), max(coverage)], actual_layer_radius_max=max(radii),
        LDF_threshold_empty_frames=empty_masks, LDF_threshold_full_frames=full_masks)
    if method in ['any2bokeh', 'bokehdiff']:
        log = (ROOT / 'logs/davis-baselines-20260929' / f'{method}-full.log').read_text(errors='replace')
        assert ('ATB_LOADED_RESUME_INDEX 8' if method == 'any2bokeh' else 'MODEL_LOADED bokehdiff resume_index 8') in log
        result['preflight_to_full_resume_index'] = 8
        result['rng_note'] = 'Stored CPU/CUDA boundary states and logged restoration checked; no re-inference or new seed.'
    write_new(out / 'integrity.json', result)
    print('FULL_INTEGRITY_PASSED', method, 1250, flush=True)
    return result


def check_all():
    base = ROOT / read(ROOT / 'configs/davis-baselines.json')['output']
    # Three independent metric workers may arrive together; only the first scans.
    with (base / 'integrity.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        record = ROOT / '03_实验记录/records/davis-six-method-integrity-20260929.json'
        if record.exists():
            result = read(record)
            assert result['status'] == 'ALL_SIX_FULL_INTEGRITY_PASSED'
            return result
        results = {m: check_method(m, base) for m in METHODS}
        result = dict(status='ALL_SIX_FULL_INTEGRITY_PASSED', frames=7500, methods=results)
        write_new(record, result)
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--method', choices=METHODS)
    args = parser.parse_args()
    if args.method: check_method(args.method)
    else: check_all()
