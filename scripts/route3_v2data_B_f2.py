"""Build the observed-input-only legacy A-color plus LaMa F2 cache."""
import argparse
from dataclasses import fields
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np
import torch

from route3_v2data_B_batch import GPU_LOCK, GIB, reserve, sha256
from route3_v2data_B_front import ROOT, write_json
from route3_v2data_B_inputs import FORMAL_DATA, observation, to_srgb8
from route3_v2data_B_plan import load
from route3_v2data_G_priors import SHARP_SOURCE, SHARP_WEIGHTS
from route3_v2data_optics import f2_entry
from route3_ovg_subpixel import save_npz
from src.aperture_renderer import GaussianScene, linear_to_srgb
from src.dual_layer import depth_band
from src.sharp_anchor import anchor


FILL_PYTHON = ROOT.parent/'envs/conda-magicbokeh/bin/python'
FILL_SCRIPT = ROOT/'scripts/route3_dual_layer_fill.py'
FRAMES = (1, 30, 60)


def prepare_input(unit, frame, cases, path):
    observed = observation(unit, frame, aligned=True)
    depth = np.where(observed['confidence'][0] > 0, observed['depth'][0], observed['depth'][1])
    depth = np.where(np.isfinite(depth) & (depth > 0), depth, 1e4).astype(np.float32)
    rgb = linear_to_srgb(torch.from_numpy(observed['rgb'])).numpy().clip(0, 1)
    union = np.zeros(depth.shape, bool)
    max_radius = 0.
    for focus, fstop in cases:
        band, _, radius = depth_band(depth, observed['K'][0,0], focus, .05/(2*fstop))
        union |= band
        max_radius = max(max_radius, float(radius.max()))
    mask = cv2.dilate(union.astype(np.uint8), np.ones((5,5),np.uint8), iterations=2)>0
    extension = int(np.ceil(2*max_radius))+4
    background = cv2.dilate(depth, np.ones((2*extension+1,2*extension+1),np.uint8))
    save_npz(path, rgb_srgb=rgb, fill_mask=mask, background_depth=background)
    return observed, background, extension


def lama_fill(target, source):
    manifest = target/'fill-input.json'
    write_json(manifest, dict(items=[dict(id='frame', input_npz=str(source),
                                         uses_GT_segmentation=False, uses_true_hidden_depth=False)]))
    result = target/'lama-result.json'
    fill = target/'lama/frame/fill.npz'
    if not result.is_file():
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES='0', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
        with (target/'lama.log').open('wb') as log:
            subprocess.run(["flock", str(GPU_LOCK), str(FILL_PYTHON), str(FILL_SCRIPT),
                            "--manifest", str(manifest), "--output", str(target/'lama'),
                            "--record", str(result)], env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    with np.load(fill) as values:
        return values['fill_rgb_linear']


@torch.no_grad()
def sharp_scene(observed, alignment):
    sys.path.insert(0, str(SHARP_SOURCE/'src'))
    from sharp.cli.predict import predict_image
    from sharp.models import PredictorParams, create_predictor

    model = create_predictor(PredictorParams())
    model.load_state_dict(torch.load(SHARP_WEIGHTS, map_location='cpu', weights_only=True), strict=True)
    model.cuda().eval()
    cloud = predict_image(model, to_srgb8(observed['rgb']), float(observed['K'][0,0]), torch.device('cuda'))
    means = cloud.mean_vectors[0]
    z = means[:,2]
    inverse = alignment['slope']/z.clamp_min(1e-6)+alignment['intercept']
    ratio = torch.where(z>1e-6, inverse.clamp_min(1e-4).reciprocal()/z, torch.ones_like(z))
    layers = (torch.arange(len(z),device=z.device)>=len(z)//2).long()
    return GaussianScene(means*ratio[:,None], cloud.quaternions[0], cloud.singular_values[0]*ratio[:,None],
                         cloud.opacities[0].reshape(-1), cloud.colors[0], layers)


def build_frame(unit, frame, cases, directory=None):
    if frame not in FRAMES:
        raise ValueError("B F2 is defined on the three training optical frames")
    geometry_sha = sha256(unit/'manifest.json')
    dest = f2_entry(unit,frame,directory)
    if (dest/'manifest.json').is_file():
        previous = load(dest/'manifest.json')
        if previous['geometry_manifest_sha256'] != geometry_sha:
            raise ValueError(f"stale F2 geometry for {unit.name} f{frame:03d}")
        return previous
    dest.parent.mkdir(parents=True,exist_ok=True)
    reserve(dest.parent, GIB)
    stage = dest.with_name(f".{dest.name}-attempt-{time.time_ns()}")
    stage.mkdir()
    observed, background, extension = prepare_input(unit,frame,cases,stage/'source.npz')
    fill = lama_fill(stage,stage/'source.npz')
    summary = load(unit/'observations/priors/summary.json')
    alignment = next(row for row in summary['rows'] if row['frame']==frame)
    start = time.monotonic()
    with GPU_LOCK.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        torch.cuda.set_per_process_memory_fraction(.69,0)
        torch.cuda.reset_peak_memory_stats()
        cloud = sharp_scene(observed, alignment)
        intrinsics = torch.as_tensor(observed['K'],device='cuda')
        fill_gpu = torch.as_tensor(fill,device='cuda')
        target = linear_to_srgb(torch.as_tensor(observed['rgb'],device='cuda')).clamp(0,1)
        torch.manual_seed(2026111121+frame)
        anchored, audit = anchor(cloud,intrinsics,fill_gpu,target,False,
                                 lambda row:write_json(stage/'anchor-progress.json',row))
        if not audit['completed']:
            raise RuntimeError(f"legacy A-color failed for {unit.name} f{frame:03d}: {audit['stop_reason']}")
        torch.save({field.name:getattr(anchored,field.name).detach().cpu() for field in fields(anchored)},
                   stage/'a-color.pt')
        peak = torch.cuda.max_memory_reserved()
        del anchored,cloud,fill_gpu,target,intrinsics
        torch.cuda.empty_cache()
    save_npz(stage/'background.npz',background_depth=background,fill_rgb_linear=fill)
    receipt = dict(scene_id=unit.name,frame=frame,geometry_manifest_sha256=geometry_sha,
                   input='observed linear RGB and the 1c SHARP/DepthSplat prior only',
                   fill='legacy depth-band mask, LaMa, aligned frozen SHARP and 1000-step A-color RGB fit',
                   dilation_radius_px=extension, focus_aperture_cases=[list(case) for case in cases],
                   network_weights_updated=False, seconds=time.monotonic()-start, peak_vram_bytes=peak,
                   product_sha256={file:sha256(stage/file) for file in ('background.npz','a-color.pt')})
    write_json(stage/'manifest.json',receipt)
    stage.rename(dest)
    return receipt


def formal_cases(optical, frame):
    rows = load(optical/'labels/optical-audit.json')['optical']
    return [(row['focus_m'],row['fstop']) for row in rows if row['id'].startswith(f'f{frame:03d}-')]


def cache_optical(unit, optical):
    return [build_frame(unit, frame, formal_cases(optical,frame)) for frame in FRAMES]


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene',type=Path,required=True)
    parser.add_argument('--isolated-destination',type=Path,required=True)
    parser.add_argument('--frame',type=int,choices=FRAMES,default=30)
    args = parser.parse_args()
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    unit = args.scene.resolve()
    meta = load(unit/'scene-spec.json')
    near = meta['near_anchor_m']
    cases = [(focus,fstop) for focus in (near,near*meta['depth_ratio']) for fstop in (1.4,2.)]
    result = build_frame(unit,args.frame,cases,args.isolated_destination.resolve())
    print('B_F2', result['scene_id'], result['frame'], result['peak_vram_bytes'], flush=True)
