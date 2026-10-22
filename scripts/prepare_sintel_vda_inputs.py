"""Frozen VDA input preparation; no GT fitting, student training or rendering."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/ours-geometry-data-pilot-20261022.json'
OUT = ROOT / 'data/processed/ours-geometry-pilot-20261022/vda-native'
RECORD = ROOT / '03_实验记录/records/ours-geometry-pilot-vda-native-20261022.json'


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def infer():
    import torch
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
    assert not OUT.exists(), 'Existing outputs are immutable; check rather than overwrite.'
    baseline = json.loads((ROOT / 'configs/davis-eval-v2.json').read_text())['vda']
    repo = (ROOT / baseline['repository']).resolve()
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    assert commit == baseline['commit']
    assert not subprocess.check_output(['git', '-C', str(repo), 'diff', '--name-only'], text=True).strip()
    checkpoint = ROOT / baseline['checkpoint']
    assert checkpoint.stat().st_size == baseline['checkpoint_bytes']
    sys.path.insert(0, str(repo))
    from video_depth_anything.video_depth import VideoDepthAnything, INFER_LEN, OVERLAP
    from video_depth_anything.util.transform import Resize
    import xformers
    torch.set_num_threads(6)
    cv2.setNumThreads(2)
    torch.manual_seed(20261022)
    np.random.seed(20261022)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    fa3_disabled = torch.cuda.get_device_capability(0)[0] >= 12
    if fa3_disabled:
        from xformers.ops.fmha import _set_use_fa3
        _set_use_fa3(False)
    model = VideoDepthAnything(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024])
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    model.eval().to('cuda')
    c = json.loads(CONFIG.read_text())
    raw = ROOT / c['raw_root'] / 'training/clean'
    runtime = dict(source_commit=commit, checkpoint=str(checkpoint.relative_to(ROOT)),
                   checkpoint_sha256=sha(checkpoint), checkpoint_revision=baseline['checkpoint_revision'],
                   python=sys.version, torch=torch.__version__, numpy=np.__version__,
                   opencv=cv2.__version__, xformers=xformers.__version__, gpu=torch.cuda.get_device_name(0),
                   seed=20261022, tf32=False, fp32=False, input_size_argument=518,
                   blackwell_fa3_disabled=fa3_disabled, context='only the selected 25-frame clip; not full source movie',
                   gt_used=False, focus=None, K_bokeh=None, training=False,
                   normalization='single min/max over all 25 native float predictions; no per-frame fit',
                   q8='floor(normalized*255), diagnostic copy; float prediction retained',
                   target_fps_argument=1, target_fps_note='pass-through return metadata only, no frame sampling')
    save_json(OUT / 'runtime.json', runtime)
    rows = []
    for s in c['sequences']:
        name = s['sequence']
        paths = [raw / name / f'frame_{i:04d}.png' for i in s['frames']]
        frames = np.stack([cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in paths])
        assert frames.shape == (25, 436, 1024, 3)
        h, w = frames.shape[1:3]
        effective = round(int(518 * 1.777 / (w / h)) / 14) * 14
        actual_w, actual_h = Resize(effective, effective, keep_aspect_ratio=True,
                                   ensure_multiple_of=14, resize_method='lower_bound').get_size(w, h)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.monotonic()
        pred, _ = model.infer_video_depth(frames, 1, input_size=518, device='cuda', fp32=False)
        torch.cuda.synchronize()
        elapsed = time.monotonic() - start
        assert pred.shape == frames.shape[:3] and np.isfinite(pred).all()
        pred = pred.astype(np.float32)
        lo, hi = float(pred.min()), float(pred.max())
        assert hi > lo
        norm = (pred - lo) / (hi - lo)
        q8 = (norm * 255).astype(np.uint8)
        dest = OUT / name
        dest.mkdir()
        np.savez_compressed(dest / 'prediction.npz', disparity_raw=pred, disparity_normalized=norm, disparity_q8=q8)
        row = dict(sequence=name, split=s['split'], frames=s['frames'], native_wh=[w,h],
                   effective_input_size=effective, network_wh=[int(actual_w),int(actual_h)],
                   input_sha256=[sha(p) for p in paths], raw_min=lo, raw_max=hi,
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(), seconds=elapsed,
                   infer_len=INFER_LEN, overlap=OVERLAP, inference_windows=2,
                   temporal_padding_copies=(INFER_LEN-OVERLAP-len(frames)%(INFER_LEN-OVERLAP))%(INFER_LEN-OVERLAP)+OVERLAP, native_coordinates_preserved=True,
                   camera_intrinsics_changed=False, gt_used=False)
        save_json(dest / 'summary.json', row)
        rows.append(row)
        print('DONE', name, round(elapsed, 2), flush=True)
        del frames, pred, norm, q8
        torch.cuda.empty_cache()
    save_json(OUT / 'summary.json', dict(runtime=runtime, sequences=rows))


def check():
    c = json.loads(CONFIG.read_text())
    summary = json.loads((OUT / 'summary.json').read_text())
    assert [s['sequence'] for s in c['sequences']] == [r['sequence'] for r in summary['sequences']]
    for r in summary['sequences']:
        p = OUT / r['sequence'] / 'prediction.npz'
        with np.load(p) as z:
            d, n, q = z['disparity_raw'], z['disparity_normalized'], z['disparity_q8']
            assert d.shape == n.shape == q.shape == (25,436,1024)
            assert d.dtype == n.dtype == np.float32 and q.dtype == np.uint8
            assert np.isfinite(d).all() and np.isfinite(n).all()
            assert np.array_equal(n, (d-r['raw_min'])/(r['raw_max']-r['raw_min']))
            assert np.array_equal(q, (n*255).astype(np.uint8))
            assert n.min() == 0 and n.max() == 1
        r['prediction_sha256'] = sha(p)
        r['prediction_bytes'] = p.stat().st_size
    summary.update(status='150_NATIVE_VDA_FRAMES_CHECKED_NOT_TRAINING_READY', output_root=str(OUT.relative_to(ROOT)),
                   scope='Only VDA predictions and normalization; GT alignment, supervision validity and corruption still pending.')
    save_json(RECORD, summary)
    print('CHECKED', len(summary['sequences'])*25, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['infer', 'check'])
    args = parser.parse_args()
    infer() if args.phase == 'infer' else check()
