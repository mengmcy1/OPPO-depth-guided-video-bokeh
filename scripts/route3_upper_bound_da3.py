"""U2 joint DA3 inference; GT is read only afterward for one global Sim(3).

No GT depth fit, per-frame rescaling, object labels or hidden data is used.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
import cv2
import numpy as np
import torch
from depth_anything_3.api import DepthAnything3

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.dual_layer import depth_band
FRAMES = [1, 6, 11, 16, 21, 26, 30, 35, 40, 45, 50, 55, 60]


def sim3(source, target):
    x, y = source - source.mean(0), target - target.mean(0)
    u, singular, vh = np.linalg.svd(y.T @ x / len(x))
    signs = np.ones(3)
    signs[-1] = np.linalg.det(u @ vh)
    R = (u * signs) @ vh
    variance = np.square(x).sum() / len(x)
    if variance < 1e-12:
        raise ValueError('Degenerate predicted camera centers')
    scale = float(singular @ signs / variance)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid global Sim3 scale')
    translation = target.mean(0) - scale * R @ source.mean(0)
    return scale, R, translation, singular


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--record', type=Path, required=True)
    p.add_argument('--base', type=Path, default=ROOT/'outputs/route3-b2only-2026-11-10-01')
    a = p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise ValueError('Explicit GPU0 required')
    subprocess.run(['nvidia-smi'], check=True)
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    cv2.setNumThreads(1)
    timings = json.loads((a.base/'inputs/B2-high/pinhole/timings.json').read_text())
    sources = {r['frame']:a.base/'inputs/B2-high/pinhole'/(r['key']+'.exr') for r in timings}
    image_dir = a.output/'input-images'
    image_dir.mkdir()
    paths = {}
    for frame in FRAMES:
        raw = cv2.imread(str(sources[frame]), cv2.IMREAD_UNCHANGED)[..., :3]
        raw = np.clip(raw, 0, 1)
        display = np.where(raw<=.0031308, 12.92*raw, 1.055*raw**(1/2.4)-.055)
        paths[frame] = image_dir/f'f{frame:03d}.png'
        cv2.imwrite(str(paths[frame]), np.uint8(np.rint(display.clip(0,1)*255)))
    # Model sees input images only. No GT camera conditioning.
    model = DepthAnything3.from_pretrained(str(ROOT/'models/DA3-BASE')).to('cuda:0').eval()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    prediction = model.inference([str(paths[f]) for f in FRAMES], process_res=518,
                                 process_res_method='upper_bound_resize', export_dir=None)
    torch.cuda.synchronize()
    seconds = time.monotonic() - start
    peak = torch.cuda.max_memory_allocated() / 2**20
    np.savez_compressed(a.output/'raw-prediction.npz', depth=prediction.depth,
        extrinsics=prediction.extrinsics, intrinsics=prediction.intrinsics,
        confidence=prediction.conf, processed_images=prediction.processed_images)
    if not (np.isfinite(prediction.depth).all() and (prediction.depth > 0).all()
            and np.isfinite(prediction.extrinsics).all()):
        raise ValueError('Invalid joint DA3 output')
    predicted = np.broadcast_to(np.eye(4), (13, 4, 4)).copy()
    predicted[:, :3, :] = prediction.extrinsics[:, :3, :]
    predicted = np.linalg.inv(predicted)
    # This is the sole GT use in geometry preparation: one sequence Sim3.
    geometries = [np.load(a.base/'B2-high/geometry'/f'f{f:03d}-center.npz') for f in FRAMES]
    gt = np.stack([d['camera_c2w'] for d in geometries])
    scale, R, t, singular = sim3(predicted[:, :3, 3], gt[:, :3, 3])
    aligned = predicted.copy()
    aligned[:, :3, :3] = R @ predicted[:, :3, :3]
    aligned[:, :3, 3] = scale * (predicted[:, :3, 3] @ R.T) + t
    errors = np.linalg.norm(aligned[:, :3, 3] - gt[:, :3, 3], axis=-1)
    rotation = np.swapaxes(gt[:, :3, :3], -1, -2) @ aligned[:, :3, :3]
    angles = np.rad2deg(np.arccos(np.clip((np.trace(rotation, axis1=1, axis2=2)-1)/2, -1, 1)))
    # Known fixed lens calibration is shared by E1/E2 and their baselines.
    # It comes from the input camera specification, never a depth correction.
    K = np.array([[512*50/36, 0, 256], [0, 512*50/36, 256], [0, 0, 1]], np.float32)
    rows = []
    for index, frame in enumerate(FRAMES):
        depth = cv2.resize(prediction.depth[index].astype(np.float32), (512,512), interpolation=cv2.INTER_LINEAR) * scale
        for variant, poses in [('E1', gt), ('E2', aligned)]:
            dest = a.output/variant
            dest.mkdir(exist_ok=True)
            np.savez_compressed(dest/f'f{frame:03d}.npz', depth=depth, K=K, c2w=poses[index])
        if frame in [1,30,60]:
            for focus in [1.4,5.]:
                for stop in [1.4,2.]:
                    band, _, coc = depth_band(depth,K[0,0],focus,.05/(2*stop))
                    rows.append(dict(frame=frame, focus_m=focus, fstop=stop,
                        band_pixels=int(band.sum()), depth_p01_p50_p99=np.quantile(depth,[.01,.5,.99]).tolist(),
                        CoC_p50_p99=np.quantile(coc,[.5,.99]).tolist()))
    record = dict(completed=True, model='DA3-BASE', weights_license='Apache-2.0', input_frames=FRAMES,
        input_paths=[str(paths[f]) for f in FRAMES], original_linear_sources=[str(sources[f]) for f in FRAMES],
        input_transfer='same clipped linear EXR -> standard displayRGB as baselines, rounded8bit only at DA3 image input',
        process_res=518, output_shape=list(prediction.depth.shape),
        inference_seconds=seconds, peak_allocated_MiB=peak, GT_conditioned_inference=False,
        global_sim3=dict(scale=scale, rotation=R.tolist(), translation=t.tolist(), singular_values=singular.tolist(),
            camera_center_rmse_m=float(np.sqrt(np.mean(errors**2))), camera_center_max_error_m=float(errors.max()),
            rotation_error_deg=angles.tolist()),
        depth_alignment='same one global camera-derived scale, bilinear resize only; no depth fitting',
        intrinsics='known fixed input lens50mm/sensor36mm/512px shared by all methods; DA3 predicted intrinsics retained for diagnosis',
        geometry_rows=rows, band_construction_valid=all(r['band_pixels']>=1000 for r in rows),
        raw_prediction=str(a.output/'raw-prediction.npz'), output=str(a.output))
    a.record.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)


if __name__ == '__main__':
    main()
