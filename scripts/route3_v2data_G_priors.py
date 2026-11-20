"""Compute G priors from RGB and poses, and expose the OVG sample interface."""
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import route3_ovg_data as ovg
from route3_ovg_r2_priors import MODEL, resize_inputs, stack_depth, support_map
from route3_ovg_subpixel import save_npz
from route3_ovg_tartan import depth_from_png
from route3_v2data_G_acquire import OUT, WINDOW
from route3_v2data_G_common import ROOT, SPLITS, check_space, read, write
from src.dense_flow_geometry import multiview_depth, pair_depth


SHARP_SOURCE = ROOT.parent / 'ml-sharp-s1-20261111'
SHARP_WEIGHTS = ROOT / 'models/SHARP/sharp_2572gikvuh.pt'
RAFT_WEIGHTS = ROOT / 'models/metrics/hub/checkpoints/raft_large_C_T_SKHT_V2-ff5fadd5.pth'


def models():
    from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
    sys.path.insert(0, str(SHARP_SOURCE / 'src'))
    from sharp.models import PredictorParams, create_predictor
    sharp = create_predictor(PredictorParams())
    sharp.load_state_dict(torch.load(SHARP_WEIGHTS, map_location='cpu', weights_only=True), strict=True)
    raft = raft_large(weights=None, progress=False)
    raft.load_state_dict(torch.load(RAFT_WEIGHTS, map_location='cpu', weights_only=True), strict=True)
    transform = Raft_Large_Weights.C_T_SKHT_V2.transforms()
    return sharp.cuda().eval(), raft.cuda().eval(), transform


@torch.inference_mode()
def sharp_prior(model, image, k, *, reference=False):
    from gsplat import rasterization
    height, width = image.shape[:2]
    if reference:
        from sharp.cli.predict import predict_image
        cloud = predict_image(model, image, float(k[0, 0]), torch.device('cuda'))
    else:
        if not np.array_equal(k, [[width/2, 0, width/2], [0, height/2, height/2], [0, 0, 1]]):
            raise ValueError('The native G intrinsics must give identity SHARP unprojection')
        rgb = torch.as_tensor(image, device='cuda').float().permute(2, 0, 1)[None] / 255.0
        rgb = F.interpolate(rgb, size=(1536, 1536), mode='bilinear', align_corners=True)
        # With the centered 90-degree G pinhole, SHARP's original NDC
        # camera-to-metric transform is identity. The CPU covariance SVD
        # would only refactor the same ellipsoids.
        cloud = model(rgb, torch.full((1,), .5, device='cuda'))
    half = cloud.mean_vectors.shape[1] // 2
    means, scales, rotations, colors, opacity = (
        getattr(cloud, name)[0, :half] for name in
        ('mean_vectors', 'singular_values', 'quaternions', 'colors', 'opacities'))
    intrinsic = torch.tensor(k, device='cuda')[None]
    rendered, alpha, _ = rasterization(means, rotations, scales, opacity.reshape(-1), colors,
                                       torch.eye(4, device='cuda')[None], intrinsic,
                                       width, height, packed=False,
                                       render_mode='RGB+ED', rasterize_mode='antialiased', eps2d=.0015)
    depth = rendered[0, ..., 3].cpu().numpy()
    confidence = alpha[0, ..., 0].cpu().numpy()
    valid = np.isfinite(depth) & (depth >= .05) & (depth <= 200) & (confidence >= .1)
    return np.where(valid, depth, 0.), np.where(valid, confidence, 0.)


def check_sharp_reference(model, path):
    meta = read(path)
    images, matrices, _ = resize_inputs(path, meta['rows'])
    image, k = images[0], matrices[0]
    before = time.monotonic()
    oracle_depth, oracle_alpha = sharp_prior(model, image, k, reference=True)
    reference_seconds = time.monotonic() - before
    before = time.monotonic()
    depth, alpha = sharp_prior(model, image, k)
    fast_seconds = time.monotonic() - before
    valid = (oracle_alpha > 0) & (alpha > 0)
    relative = np.abs(depth[valid]/oracle_depth[valid] - 1) if valid.any() else np.ones(1)
    median_relative, p99_relative = np.quantile(relative, [.50, .99]).tolist()
    mask_agreement = float(((oracle_alpha > 0) == (alpha > 0)).mean())
    write(OUT/'sharp-identity-check.json', dict(passed=p99_relative <= .01 and mask_agreement >= .99,
          window=meta['scene'], frame=meta['rows'][0]['frame'], median_relative_depth_error=median_relative,
          p99_relative_depth_error=p99_relative, valid_agreement=mask_agreement,
          reference_seconds=reference_seconds, identity_seconds=fast_seconds))


@torch.inference_mode()
def triangulation(raft, transform, images, intrinsics, poses):
    n = len(images)
    input_images = torch.as_tensor(images, device='cuda').permute(0, 3, 1, 2).contiguous()
    adjacent = {(i, j) for i in range(n) for j in range(max(0, i-2), min(n, i+3)) if i != j}
    pair_ids = sorted(adjacent)
    flows = {}
    for start in range(0, len(pair_ids), 4):
        pairs = pair_ids[start:start+4]
        a, b = zip(*pairs)
        first, second = transform(input_images[list(a)], input_images[list(b)])
        result = raft(first, second, num_flow_updates=32)[-1].permute(0, 2, 3, 1).cpu().numpy()
        flows.update(zip(pairs, result))
    results = []
    for i in range(n):
        neighbors = []
        for j in sorted(b for a, b in adjacent if a == i):
            depth, uv, _ = pair_depth(flows[i, j], flows[j, i], intrinsics[i], poses[i], poses[j])
            neighbors.append(dict(frame=j, depth=depth, other_xy=uv))
        depth, valid, support, _ = multiview_depth(neighbors, intrinsics[i], poses[i], poses)
        valid &= np.isfinite(depth) & (depth >= .05) & (depth <= 200)
        results.append((np.where(valid, depth, 0), np.where(valid, support / len(neighbors), 0)))
    return results


def validate_source(meta):
    rows = meta['rows']
    first = rows[0]['frame']
    if len(rows) != WINDOW or [row['frame'] for row in rows] != list(range(first, first + WINDOW)):
        raise ValueError('TartanAir window is not eight consecutive capture frames')
    for row in rows:
        k = np.asarray(row['K'])
        pose = np.asarray(row['c2w'])
        if not np.allclose(k, [[320, 0, 320], [0, 320, 320], [0, 0, 1]]):
            raise ValueError('Unexpected native TartanAir intrinsics')
        if not np.isfinite(pose).all() or not np.allclose(pose[3], [0, 0, 0, 1]) or not np.isclose(np.linalg.det(pose[:3, :3]), 1):
            raise ValueError('Unexpected OpenCV camera-to-world pose')
        if 'labels' not in Path(row['depth']).parts or row['image'] == row['depth']:
            raise ValueError('G depth label is not physically separate from RGB')


def precalculate(net, sharp, raft, transform, path):
    meta = read(path)
    validate_source(meta)
    check_space(OUT, 16 * 2**20)
    start = time.monotonic()
    images, intrinsics, poses = resize_inputs(path, meta['rows'])
    if images.shape != (WINDOW, 256, 256, 3) or not np.allclose(intrinsics[:, :2], [[128, 0, 128], [0, 128, 128]]):
        raise ValueError('G loader image size or scaled intrinsics differ from the fixed contract')
    ds = stack_depth(net, images, intrinsics, poses)
    tri = triangulation(raft, transform, images, intrinsics, poses)
    use_reference = not read(OUT/'sharp-identity-check.json')['passed']
    for i, (image, k) in enumerate(zip(images, intrinsics)):
        sharp_depth, sharp_confidence = sharp_prior(sharp, image, k, reference=use_reference)
        tri_depth, tri_confidence = tri[i]
        supported = support_map(ds, intrinsics, poses, i)
        save_npz(path.parent/'prior'/f'f{i:03d}.npz', depth=ds[i].astype(np.float16),
                 support=supported, consistent=supported >= 2,
                 sharp_depth=sharp_depth.astype(np.float16), sharp_confidence=sharp_confidence.astype(np.float16),
                 triangulation_depth=tri_depth.astype(np.float16),
                 triangulation_confidence=tri_confidence.astype(np.float16))
    validate_loader(path)
    write(path.parent/'prior'/'summary.json', dict(window=meta['scene'], tier='G', frames=WINDOW,
          dtype='float16', seconds=time.monotonic()-start, checkpoint=MODEL,
          channels=list(ovg.PRIORS), given_poses=True, GT_depth_input=False, loader_checked=True,
          sharp_unprojection='official' if use_reference else 'identity pinhole'))


def observation(path, index):
    observed = ovg.gc_input(str(path), index)
    priors = ovg.load(path.parent/'prior'/f'f{index:03d}.npz')
    depth, confidence = observed['depth'].copy(), observed['confidence'].copy()
    for slot, name in ((0, 'sharp'), (2, 'triangulation')):
        depth[slot] = priors[f'{name}_depth']
        confidence[slot] = priors[f'{name}_confidence']
    return dict(observed, depth=depth, confidence=confidence)


def sample(item, device):
    tier, filename, index = item
    path = Path(filename)
    meta = read(path)
    if tier != 'G':
        raise ValueError(tier)
    current = observation(path, index)
    neighbors = [observation(path, j) for j in ovg.neighbors(index, len(meta['rows']))]
    inputs = ovg.input_record(current, neighbors)
    native_depth = depth_from_png(ROOT / meta['rows'][index]['depth'])
    depth = cv2.resize(native_depth, (256, 256), interpolation=cv2.INTER_NEAREST_EXACT)
    target = ovg.no_label(depth, current['K'])
    target['normal'] = target['normal'].transpose(2, 0, 1)
    return tuple({key: torch.as_tensor(array[None].copy(), device=device) for key, array in row.items()}
                 for row in (inputs, target))


def validate_loader(path):
    item = ('G', str(path), 0)
    original, labels = ovg.sample(item, 'cpu')
    observed, target = sample(item, 'cpu')
    if observed.keys() != original.keys() or target.keys() != labels.keys():
        raise ValueError('OVG G input or supervision schema mismatch')
    if not all(torch.equal(target[key], labels[key]) for key in target):
        raise ValueError('OVG G supervision differs from the existing axial depth loader')
    if not torch.equal(observed['prior_depth'][:, 1], original['prior_depth'][:, 1]):
        raise ValueError('DepthSplat prior differs from the existing loader')
    if not torch.isfinite(observed['prior_depth']).all() or not torch.isfinite(observed['prior_conf']).all():
        raise ValueError('Non-finite G priors')


def inventory(role):
    if role not in SPLITS:
        raise ValueError(role)
    for unit in read(OUT/'manifest.json')['units']:
        if unit['split'] == role:
            for window in unit['windows']:
                path = OUT / window
                yield from (('G', str(path), index) for index in range(WINDOW))
