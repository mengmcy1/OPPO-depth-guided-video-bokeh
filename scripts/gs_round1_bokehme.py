#!/usr/bin/env python3
"""Calibrate the existing BokehMe adapter on reconstructed training views only."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from baselines.bokehme import setup, predict
from gs_round1_method import Sampler, write
from gs_round1_resources import gpu_slot


def rgb(path):
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB).astype(np.float32)/255


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True)
    args = parser.parse_args(); job = json.loads(args.job.read_text())
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
    subprocess.run(['nvidia-smi'], check=True)
    lock = (Path(os.environ['TMPDIR'])/'dof-gs-gpu.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX if job.get('exclusive') else fcntl.LOCK_SH)
    with gpu_slot(job.get('reserved_memory_mib',16384),job['id']+':train',exclusive=bool(job.get('exclusive'))):
        run(job)


def run(job):
    config = json.loads((ROOT/'configs/dof-gs-round1-20261105.json').read_text())['bokehme']
    model_output = Path(job['dofgs_output'])
    source = ROOT/'data/processed/dofnerf-scenes/mix'/job['scene']
    manifest = json.loads((source/'manifest.json').read_text())
    by_name = {Path(v['output_filename']).stem: v for v in manifest['views']}
    pairs = json.loads((model_output/'rendered/pairs.json').read_text())['pairs']
    views = {}
    all_valid_disparities = []
    for pair in pairs:
        if pair['protocol'] != 'r0': continue
        name = pair['image']; base = Path(pair['prediction']).parent
        depth = np.load(base/(name+'-depth.npy')); alpha = np.load(base/(name+'-alpha.npy'))
        valid = np.isfinite(depth) & (depth > 1e-6) & (alpha > .01)
        disparity = np.zeros_like(depth, dtype=np.float32); disparity[valid] = 1/depth[valid]
        if pair['role'] == 'train': all_valid_disparities.append(disparity[valid])
        views[name] = {'role': pair['role'], 'image': rgb(pair['prediction']), 'truth': rgb(pair['truth']) if pair['role']=='train' else None,
                       'disparity': disparity, 'valid': valid, 'selected_mode': by_name[name]['selected_mode']}
    assert all_valid_disparities
    low, high = np.percentile(np.concatenate(all_valid_disparities), config['disparity_percentiles'])
    assert high > low
    for view in views.values():
        view['disparity'] = np.where(view['valid'], np.clip((view['disparity']-low)/(high-low), 0, 1), 0).astype(np.float32)
    pipeline, settings, nets, _ = setup(ROOT.parent/'BokehMe', ROOT/'models/BokehMe', config)
    sampler = Sampler(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); sampler.start(); started = time.perf_counter()
    def render(view, focus, strength):
        defocus = strength*(view['disparity']-focus)/config['defocus_scale']
        return predict(pipeline, settings, nets, view['image'], defocus.astype(np.float32))[0]
    chosen = {}; records = []
    for mode in ['fg', 'bg']:
        training = [(name, value) for name, value in views.items() if value['role'] == 'train' and value['selected_mode'] == mode]
        assert training
        scores = []
        for focus in config['focus_grid']:
            for strength in config['K_grid']:
                error = sum(float(np.abs(render(value, focus, strength)-value['truth']).mean()) for _, value in training)/len(training)
                scores.append({'focus': focus, 'K': strength, 'mean_training_l1': error})
        best = min(scores, key=lambda value: value['mean_training_l1'])
        chosen[mode] = dict(best, training_images=[name for name, _ in training], grid=scores)
        destination = Path(job['output'])/'rendered'/mode; destination.mkdir(parents=True, exist_ok=True)
        for name, value in views.items():
            if value['role'] != 'test': continue
            prediction = render(value, best['focus'], best['K'])
            pred = destination/(name+'.png')
            cv2.imwrite(str(pred), cv2.cvtColor((prediction*255).round().astype(np.uint8), cv2.COLOR_RGB2BGR))
            raw = ROOT/'data/raw/DoF-NeRF/DoF_RealWorld_Data'/job['scene']/by_name[name]['source_files'][mode]
            truth = destination/(name+'-gt.png')
            cv2.imwrite(str(truth), cv2.resize(cv2.imread(str(raw)), (497, 331), interpolation=cv2.INTER_AREA))
            records.append({'protocol': 'r1-'+mode, 'role': 'test', 'image': name, 'prediction': str(pred), 'truth': str(truth)})
    torch.cuda.synchronize(); seconds = time.perf_counter()-started
    observed = sampler.finish()
    output = Path(job['output'])
    write(output/'bokehme-efficiency.json', {'seconds': seconds, 'torch_peak_bytes': torch.cuda.max_memory_allocated(),
                                           'nvidia_peak_mib': observed['gaussian']['nvidia_peak_mib'], 'exclusive': job.get('exclusive', False)})
    write(output/'calibration.json', {'scene': job['scene'], 'grid_config': config, 'chosen': chosen,
                                     'depth_conversion': 'alpha-normalized Gaussian z expectation -> reciprocal depth; alpha>0.01; global1/99 percentiles from training views only; clip0..1, invalid0',
                                     'training_disparity_bounds': [float(low), float(high)], 'test_images_used_for_fitting': False,
                                     'aif_input': 'DOF-GS PNG8 all-in-focus reconstruction, float32 depth/alpha',
                                     'adapter': 'existing src/baselines/bokehme.py, official gamma2.2 and defocus_scale10'})
    write(output/'rendered/pairs.json', {'method': 'bokehme', 'dataset': 'dofnerf-mix', 'scene': job['scene'], 'pairs': records})


if __name__ == '__main__': main()
