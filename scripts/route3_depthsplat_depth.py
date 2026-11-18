"""Official DepthSplat depth-only inference; no Gaussian network is loaded.

Uses the released complete small checkpoint strictly. DINOv2 is constructed
locally without downloading duplicate initial weights. Architecture is intact.
"""
import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import time
import types
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
import cv2
import numpy as np
import torch


def network(code, dino, weights):
    sys.path.insert(0, str(dino))
    package = types.ModuleType('depthsplat_unimatch')
    package.__path__ = [str(code/'src/model/encoder/unimatch')]
    sys.modules[package.__name__] = package
    module = importlib.import_module('depthsplat_unimatch.mv_unimatch')
    original = torch.hub.load
    def local_load(repo, name, *args, **kwargs):
        if repo != 'facebookresearch/dinov2':
            raise ValueError('Unexpected backbone '+repo)
        from dinov2.hub import backbones
        return getattr(backbones, name)(pretrained=False)
    torch.hub.load = local_load
    try:
        # Small release uses one scale / 8x upsampling; the official base
        # example's two-scale / 4x settings do not match the small checkpoint.
        net = module.MultiViewUniMatch(num_scales=1, upsample_factor=8,
                                       lowest_feature_resolution=8, vit_type='vits')
    finally:
        torch.hub.load = original
    state = torch.load(weights, map_location='cpu', weights_only=True)['model']
    result = net.load_state_dict(state, strict=True)
    return net.eval(), str(result)


def infer(args):
    torch.set_num_threads(4)
    net, keys = network(args.code, args.dino, args.weights)
    if args.construct_only:
        print('DEPTHSPLAT_STRICT_LOAD', keys, sum(p.numel() for p in net.parameters()))
        return
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
    import subprocess
    subprocess.run(['nvidia-smi'], check=True)
    net.cuda()
    plan = json.loads(args.plan.read_text())
    for entry in plan:
        dest = Path(entry['output'])
        if dest.exists():
            continue
        imgs = [cv2.imread(p)[..., ::-1].copy() for p in entry['images']]
        images = torch.tensor(np.stack(imgs), device='cuda').permute(0, 3, 1, 2).float()[None]/255
        Ks = np.stack(entry['K']).astype(np.float32)
        Ks[:, 0] /= images.shape[-1]
        Ks[:, 1] /= images.shape[-2]
        intr = torch.tensor(Ks, device='cuda')[None]
        poses = torch.tensor(np.array(entry['c2w']), dtype=torch.float32, device='cuda')[None]
        neighbors = torch.cdist(poses[:, :, :3, 3], poses[:, :, :3, 3]).argsort(-1)[:, :, :3]
        n = len(imgs)
        near = torch.full((1, n), .1, device='cuda')
        far = torch.full((1, n), 1000., device='cuda')
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.monotonic()
        with torch.inference_mode():
            out = net(images, attn_splits_list=[2], intrinsics=intr,
                      min_depth=1/far, max_depth=1/near, extrinsics=poses,
                      nn_matrix=neighbors)
        torch.cuda.synchronize()
        seconds = time.monotonic()-start
        depths = out['depth_preds'][-1][0].cpu().numpy()
        assert np.isfinite(depths).all() and (depths > 0).all()
        # Input-only multiview consistency; no GT depth is read here.
        target = entry['target_index']
        z = depths[target]
        h, w = z.shape
        y, x = np.mgrid[:h, :w]
        pixelK = np.array(entry['K'][target])
        # Match official pixel coordinates (integer meshgrid in warp helper).
        pts = np.dstack([x, y, np.ones_like(x)])@np.linalg.inv(pixelK).T*z[..., None]
        Ps = np.array(entry['c2w'])
        world = pts@Ps[target, :3, :3].T+Ps[target, :3, 3]
        support = np.zeros(z.shape, np.uint8)
        for i in range(n):
            if i == target:
                continue
            cam = (world-Ps[i, :3, 3])@Ps[i, :3, :3]
            q = cam@np.array(entry['K'][i]).T
            uv = q[..., :2]/np.maximum(q[..., 2:], 1e-8)
            observed = cv2.remap(depths[i], uv[..., 0].astype(np.float32),
                                 uv[..., 1].astype(np.float32), cv2.INTER_LINEAR)
            support += ((cam[..., 2] > 0) & (observed > 0) &
                        (abs(observed-cam[..., 2]) <= .03*cam[..., 2])).astype(np.uint8)
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dest, depth=z, consistent=support >= 2, support=support)
        result = dict(seconds=seconds, peak_MiB=torch.cuda.max_memory_allocated()/2**20,
                      frames=entry['frames'], target=entry['target_frame'],
                      strict_load=keys, resolution=[h, w], near=.1, far=1000.,
                      confidence='>=2other input predictions agree within3% axialdepth',
                      pose_scale_preserved=True, GT_depth_used=False)
        dest.with_suffix('.json').write_text(json.dumps(result, indent=2)+'\n')
        print('DEPTHSPLAT_DEPTH', entry['scene'], entry['target_frame'], seconds, flush=True)
        del out, images, poses
        torch.cuda.empty_cache()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--code', type=Path, default=Path('/root/autodl-tmp/depthsplat-r13-20261117'))
    p.add_argument('--dino', type=Path, default=Path('/root/autodl-tmp/dinov2-depthsplat-r13'))
    p.add_argument('--weights', type=Path, default=Path('models/DepthSplat/depthsplat-depth-small-352x640-randview2-8-e807bd82.pth'))
    p.add_argument('--plan', type=Path)
    p.add_argument('--construct-only', action='store_true')
    infer(p.parse_args())
