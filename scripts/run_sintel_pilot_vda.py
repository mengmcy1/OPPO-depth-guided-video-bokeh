"""Run frozen VDA on the Sintel geometry pilot and compare with GT inverse depth.

Same VDA protocol as scripts/prepare_davis_v2.py (vitl, input_size 518, fp32 False,
per-sequence min/max quantization). GT is never used for the VDA input; it is only
used afterwards for one sequence-level scale/shift fit (least squares on valid
pixels) to place VDA in GT inverse-depth units for comparison figures.
"""
import json, os, struct, sys, time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/ours-geometry-data-pilot-20261022.json'
OUT = ROOT / 'outputs/ours-geometry-pilot-vda-20261022'
VDA_REPO = (ROOT / '../Video-Depth-Anything').resolve()
VDA_COMMIT = '4f5ae23172ba60fd7bc11ef671cca678842c7072'
CKPT = ROOT / 'models/Video-Depth-Anything/video_depth_anything_vitl.pth'


def depth_read(path):
    b = path.read_bytes()
    assert b[:4] == b'PIEH'
    w, h = struct.unpack('<ii', b[4:12])
    return np.frombuffer(b, dtype='<f4', offset=12).reshape(h, w).copy()


def main():
    import subprocess, torch
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
    assert subprocess.check_output(['git', '-C', str(VDA_REPO), 'rev-parse', 'HEAD'], text=True).strip() == VDA_COMMIT
    sys.path.insert(0, str(VDA_REPO))
    from video_depth_anything.video_depth import VideoDepthAnything
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = VideoDepthAnything(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024])
    model.load_state_dict(torch.load(CKPT, map_location='cpu', weights_only=True), strict=True)
    model.eval().to('cuda')

    c = json.loads(CONFIG.read_text())
    raw = ROOT / c['raw_root'] / 'training'
    summary = []
    for s in c['sequences']:
        name = s['sequence']
        dest = OUT / name
        assert not dest.exists(), f'Preserve existing {dest}'
        dest.mkdir(parents=True)
        files = [f'frame_{i:04d}' for i in s['frames']]
        frames = np.stack([cv2.cvtColor(cv2.imread(str(raw / 'clean' / name / f'{f}.png')), cv2.COLOR_BGR2RGB) for f in files])
        gt = np.stack([depth_read(raw / 'depth' / name / f'{f}.dpt') for f in files])
        t0 = time.monotonic()
        pred, _ = model.infer_video_depth(frames, 1, input_size=518, device='cuda', fp32=False)
        seconds = time.monotonic() - t0
        assert pred.shape == gt.shape and np.isfinite(pred).all()
        lo, hi = float(pred.min()), float(pred.max())
        q = ((pred - lo) / (hi - lo) * 255).astype(np.uint8)  # ATB pre_process quantization
        valid = np.isfinite(gt) & (gt > 0)
        inv = np.where(valid, 1.0 / np.where(valid, gt, 1), 0).astype(np.float32)
        # One sequence-level affine fit of VDA disparity to GT inverse depth.
        a, b = np.polyfit(pred[valid].astype(np.float64), inv[valid].astype(np.float64), 1)
        aligned = (a * pred + b).astype(np.float32)
        np.savez_compressed(dest / 'depth.npz', vda=pred.astype(np.float32), q=q, gt_inv=inv,
                            valid=valid, aligned=aligned)
        err = np.abs(aligned - inv)[valid]
        rel = err / inv[valid]
        row = dict(sequence=name, split=s['split'], frames=len(files), seconds=round(seconds, 2),
                   vda_min=lo, vda_max=hi, fit_scale=float(a), fit_shift=float(b),
                   valid_fraction=float(valid.mean()), absrel_inv_median=float(np.median(rel)),
                   absrel_inv_mean=float(rel.mean()), absrel_inv_gt_0p25=float((rel > 0.25).mean()))
        (dest / 'summary.json').write_text(json.dumps(row, indent=2) + '\n')
        summary.append(row)
        print('VDA_DONE', name, round(seconds, 2), flush=True)
    (OUT / 'summary.json').write_text(json.dumps(dict(
        vda_commit=VDA_COMMIT, input_size=518, fp32=False, render_pass='clean',
        alignment='per-sequence least-squares scale+shift of VDA output to GT inverse depth (valid=finite,>0)',
        sequences=summary), indent=2) + '\n')


if __name__ == '__main__':
    main()
