#!/usr/bin/env python3
"""Build fixed, method-independent evaluation regions using existing DA-V2 weights.

CPU inference uses exactly the repository's image transform and forward path;
it avoids infer_image's automatic CUDA selection during exclusive training.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose

ROOT = Path(__file__).resolve().parents[1]


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/dof-gs-round1-20261105.json')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/dof-gs-repro-2026-11-05/regions')
    parser.add_argument('--scene', nargs='*')
    args = parser.parse_args(); config = json.loads(args.config.read_text()); settings = config['regions']
    torch.set_num_threads(8); cv2.setNumThreads(4)
    repository = ROOT.parent/'MagicBokeh'; sys.path.insert(0, str(repository))
    from depth_anything_v2.dpt import DepthAnythingV2
    from depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet
    model = DepthAnythingV2(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024])
    state = torch.load(ROOT/settings['weights'], map_location='cpu')
    if 'model' in state: state = state['model']
    model.load_state_dict({k.replace('module.', ''): v for k, v in state.items()}, strict=True)
    model.cpu().eval()
    transform = Compose([Resize(width=settings['input_size'], height=settings['input_size'], resize_target=False,
                                keep_aspect_ratio=True, ensure_multiple_of=14, resize_method='lower_bound', image_interpolation_method=cv2.INTER_CUBIC),
                         NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), PrepareForNet()])
    rows = []
    for scene in args.scene or config['dofnerf_scenes']:
        source = ROOT/'data/processed/dofnerf-scenes/mix'/scene
        if not (source/'manifest.json').exists(): continue
        manifest = json.loads((source/'manifest.json').read_text())
        destination = args.output/scene; destination.mkdir(parents=True, exist_ok=True)
        previews = []
        for view in manifest['views']:
            if view['role'] != 'test': continue
            name = Path(view['output_filename']).stem
            sharp = cv2.imread(str(source/'images_4'/view['output_filename']))
            height, width = sharp.shape[:2]
            image = transform({'image': cv2.cvtColor(sharp, cv2.COLOR_BGR2RGB)/255.})['image']
            image = torch.from_numpy(image).unsqueeze(0).cpu()
            depth = model(image)
            depth = F.interpolate(depth[:, None], (height, width), mode='bilinear', align_corners=True)[0, 0].cpu().numpy()
            assert np.isfinite(depth).all()
            normalized = (depth-depth.min())/max(float(np.ptp(depth)), 1e-6)
            gx = cv2.Sobel(normalized, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(normalized, cv2.CV_32F, 0, 1, ksize=3)
            gradient = np.hypot(gx, gy)
            threshold = float(np.quantile(gradient, settings['gradient_quantile']))
            radius = settings['dilation_radius']
            boundary = cv2.dilate((gradient > threshold).astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius*2+1, radius*2+1))).astype(bool)
            masks = {'boundary': boundary, 'relative_depth': depth}
            for mode in ['fg', 'bg']:
                raw = ROOT/'data/raw/DoF-NeRF/DoF_RealWorld_Data'/scene/view['source_files'][mode]
                focus = cv2.resize(cv2.imread(str(raw)), (width, height), interpolation=cv2.INTER_AREA)
                blur = lambda x: cv2.GaussianBlur(x.astype(np.float32)/255, (0, 0), settings['difference_sigma'])
                difference = np.abs(blur(focus)-blur(sharp)).mean(-1)
                masks[mode+'_in_focus'] = (difference < settings['focus_difference_threshold']) & ~boundary
                masks[mode+'_out_of_focus'] = (difference >= settings['focus_difference_threshold']) & ~boundary
            np.savez_compressed(destination/(name+'.npz'), **masks)
            overlay = sharp.copy(); overlay[boundary] = (0.5*overlay[boundary]+np.array([0, 0, 127.5])).astype(np.uint8)
            focus_overlay = sharp.copy()
            focus_overlay[masks['fg_in_focus']] = (0.5*focus_overlay[masks['fg_in_focus']]+np.array([0, 127.5, 0])).astype(np.uint8)
            previews.append(np.concatenate([sharp, cv2.applyColorMap((normalized*255).astype(np.uint8), cv2.COLORMAP_VIRIDIS), overlay, focus_overlay], axis=1))
            rows.append({'scene': scene, 'image': name, 'gradient_threshold': threshold,
                         'fractions': {key: float(value.mean()) for key, value in masks.items() if value.dtype == bool}})
        if previews: cv2.imwrite(str(destination/'preview.png'), np.concatenate(previews, axis=0))
        print(scene, 'regions ready', flush=True)
    report = {'settings': settings, 'source_repository': str(repository),
              'source_commit': subprocess.check_output(['git', '-C', str(repository), 'rev-parse', 'HEAD'], text=True).strip(),
              'weights': str(ROOT/settings['weights']), 'views': rows, 'visually_reviewed': False,
              'interpretation': 'fixed relative-depth boundary and photometric focus proxies, not semantic ground-truth segmentation'}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'regions-candidate.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__': main()
