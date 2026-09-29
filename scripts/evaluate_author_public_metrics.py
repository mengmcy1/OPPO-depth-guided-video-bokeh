#!/usr/bin/env python3
"""Evaluate existing PNG pairs with the public Any-to-Bokeh metric kernels.

Upstream model/metric source is read, never modified. The adapter replaces
missing/private imports and broken video indexing, not metric/model formulas.
"""
import argparse
import ast
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset


def write_json(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def git_head(path):
    return subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()


def read_clip(directory):
    paths = sorted(directory.glob('*.png'))
    assert [p.name for p in paths] == [f'{i:05d}.png' for i in range(25)], directory
    frames = np.stack([np.array(Image.open(p).convert('RGB')) for p in paths])
    assert frames.shape == (25, 576, 1024, 3), (directory, frames.shape)
    return frames


class Pairs(Dataset):
    def __init__(self, clips, ids, prediction_root=None, prediction_dir='pred_any2bokeh'):
        self.clips, self.ids = clips, ids
        self.prediction_root, self.prediction_dir = prediction_root or clips, prediction_dir

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        scene = self.clips / self.ids[index]
        paths = [scene / 'gt_mpib_public_srgb', self.prediction_root / self.ids[index] / self.prediction_dir]
        return tuple(torch.from_numpy(read_clip(path)).permute(3, 0, 1, 2).float() / 255 for path in paths)


class FVDFrames(Dataset):
    def __init__(self, clips, ids, role):
        self.clips, self.ids, self.role = clips, ids, role

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        from cdfvd.utils.data_utils import preprocess
        # Identical to cdfvd video_folder spatial/temporal preprocessing, without
        # lossy MP4 re-encoding or an extra video-reader dependency.
        return preprocess(torch.from_numpy(read_clip(self.clips / self.ids[index] / self.role)),
                          resolution=256, sequence_length=25, sample_every_n_frames=1)


def public_image_functions(author):
    """Execute the exact public PSNR/SSIM function ASTs, with their dependencies.

    Importing all of vid_metrics.py fails on unpublished metrics.utils and the
    legacy z_train package. Neither is needed for the two metric functions.
    """
    from einops import rearrange
    from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
    from tqdm import tqdm
    source = author / 'metrics/vid_metrics.py'
    tree = ast.parse(source.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in ('psnr', 'ssim')]
    assert len(nodes) == 2
    namespace = dict(torch=torch, rearrange=rearrange, tqdm=tqdm,
                     PeakSignalNoiseRatio=PeakSignalNoiseRatio,
                     StructuralSimilarityIndexMeasure=StructuralSimilarityIndexMeasure)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), namespace)
    return namespace


def torchmetrics_scores(args, ids):
    funcs = public_image_functions(args.author)
    loader = DataLoader(Pairs(args.clips, ids, args.prediction_root, args.prediction_dir), batch_size=1, shuffle=False,
                        num_workers=args.workers, prefetch_factor=2 if args.workers else None)
    values = {name.upper(): float(funcs[name](loader).cpu()) for name in ('psnr', 'ssim')}
    return dict(values=values, video_count=len(ids), frame_count=25 * len(ids),
                aggregation='public function; batch_size=1 clip; average of clip metric forward values',
                input='RGB PNG8 / 255; original 1024x576; all 25 frames; no encoding/resize')


def vfid_scores(args, ids):
    from metrics.fid_metrics.dataset import ImageSequenceDataset
    from metrics.fid_metrics.fid import build_inception3d, calculate_fid
    from tqdm import tqdm
    assert args.i3d.is_file() and args.i3d.stat().st_size == 50883138, 'Finish the specified I3D download first'
    model = build_inception3d(str(args.i3d)).cuda().eval()
    features = []
    for base, role in [(args.clips, 'gt_mpib_public_srgb'), (args.prediction_root or args.clips, args.prediction_dir)]:
        if role == 'gt_mpib_public_srgb' and args.reference_features:
            cached = np.load(args.reference_features / 'vfid-i-features.npz')
            assert list(cached['scene_ids']) == ids and cached['gt'].shape == (2*len(ids), 1024)
            features.append(cached['gt'])
            continue
        # One public dataset per scene: no cross-scene windows. Public defaults
        # remain sequence_length=10, resize=(224,224), no_overlap=True.
        datasets = [ImageSequenceDataset(str(base / sid / role), sequence_length=10,
                                        resize_shape=(224, 224)) for sid in ids]
        assert all(len(ds) == 2 for ds in datasets)
        loader = DataLoader(ConcatDataset(datasets), batch_size=4, shuffle=False, num_workers=args.workers)
        rows = []
        with torch.no_grad():
            for x in tqdm(loader, desc=f'VFID-I {role}'):
                x = x.cuda() * 2 - 1
                y = model.extract_features(x).squeeze(3).squeeze(3).mean(2)
                assert y.shape[1] == 1024 and torch.isfinite(y).all()
                rows.append(y.cpu().numpy())
        features.append(np.concatenate(rows))
    assert all(x.shape == (2 * len(ids), 1024) for x in features)
    with (args.output / 'vfid-i-features.npz').open('xb') as feature_file:
        np.savez(feature_file, gt=features[0], pred=features[1], scene_ids=ids)
    score = float(calculate_fid(*features))
    return dict(values={'VFID_I': score}, video_count=len(ids), feature_count_per_side=2 * len(ids),
                feature_dim=1024, sequence_length=10, starts=[0, 10], unused_tail_frames=[20, 21, 22, 23, 24],
                preprocessing='public RGB ImageSequenceDataset: ToTensor, Resize 224x224, then [-1,1]',
                adaptation='per-scene datasets avoid public flattened-folder cross-scene clips and broken VideoDataset indices',
                covariance='public calculate_fid / numpy.cov ddof=1 / scipy sqrtm', checkpoint=str(args.i3d))


def fvd_scores(args, ids):
    from cdfvd import fvd
    assert args.videomae.is_file() and args.videomae.stat().st_size == 2023804201, 'Finish the specified VideoMAE download first'
    evaluator = fvd.cdfvd('videomae', ckpt_path=str(args.videomae), n_real='full', n_fake='full',
                          seed=42, compute_feats=True, half_precision=False, device='cuda')
    for base, role, function in [(args.clips, 'gt_mpib_public_srgb', evaluator.compute_real_stats),
                                (args.prediction_root or args.clips, args.prediction_dir, evaluator.compute_fake_stats)]:
        if role == 'gt_mpib_public_srgb' and args.reference_features:
            cached = np.load(args.reference_features / 'fvd-videomae-features.npz')
            assert list(cached['scene_ids']) == ids and cached['gt'].shape == (len(ids), 1408)
            evaluator.real_stats.append(cached['gt'])
            continue
        loader = DataLoader(FVDFrames(base, ids, role), batch_size=16, shuffle=False,
                            num_workers=args.workers)
        function(loader)
    assert evaluator.real_stats.num_items == evaluator.fake_stats.num_items == len(ids)
    gt, pred = evaluator.real_stats.get_all(), evaluator.fake_stats.get_all()
    assert gt.shape == pred.shape == (len(ids), 1408)
    assert np.isfinite(gt).all() and np.isfinite(pred).all()
    with (args.output / 'fvd-videomae-features.npz').open('xb') as feature_file:
        np.savez(feature_file, gt=gt, pred=pred, scene_ids=ids)
    score = float(evaluator.compute_fvd_from_stats())
    return dict(values={'FVD_VideoMAE': score}, video_count=len(ids), feature_count_per_side=len(ids),
                feature_dim=1408, sequence_length=25, batch_size=16, seed=42,
                preprocessing='cdfvd video preprocess: short side 256 + center crop; byte conversion; public VideoMAE resize 224; [0,1]',
                adaptation="n_fake='full' prevents default 2048 repeated samples; PNG decoding replaces video decoding",
                temporal_note='public tubelet_size=2: 25 input frames yield 12 temporal tokens; final unmatched frame is unused',
                covariance='public FeatureStats population covariance / scipy sqrtm', checkpoint=str(args.videomae))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--clips', type=Path, default=Path('outputs/any2bokeh-mpib-public-proxy-200-20260928/clips'))
    p.add_argument('--author', type=Path, default=Path('../any-to-bokeh'))
    p.add_argument('--cdfvd', type=Path, default=Path('../content-debiased-fvd'))
    p.add_argument('--i3d', type=Path, default=Path('models/metrics/VFID/i3d.pt'))
    p.add_argument('--videomae', type=Path, default=Path('models/metrics/VideoMAE2/mae-g/vit_g_hybrid_pt_1200e_ssv2_ft.pth'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--limit', type=int)
    p.add_argument('--only', help='Fixed scene subset, kept in original manifest order; mutually exclusive with --limit')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--prediction-root', type=Path)
    p.add_argument('--prediction-dir', default='pred_any2bokeh')
    p.add_argument('--reference-features', type=Path, help='Reuse verified GT features on identical scenes and preprocessing')
    p.add_argument('--phase', choices=['all', 'torchmetrics', 'vfid', 'fvd'], default='all')
    args = p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0', 'Select the authorized GPU explicitly'
    args.author, args.cdfvd = args.author.resolve(), args.cdfvd.resolve()
    assert git_head(args.author) == 'f7a16959dd219080fb8eec0d32bb9602d8970c22'
    assert git_head(args.cdfvd) == 'a1e037ab7cb087debd2221d14ae4a001ec054201'
    sys.path.insert(0, str(args.author))
    sys.path.insert(0, str(args.cdfvd))
    torch.set_num_threads(8)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    ids = [s['id'] for s in json.loads((args.clips / 'config.json').read_text())['scenes']]
    assert len(ids) == len(set(ids)) == 200
    if args.only:
        assert args.limit is None
        selected=args.only.split(',')
        assert len(selected)==len(set(selected)) and set(selected)<=set(ids)
        ids=[sid for sid in ids if sid in selected]
    if args.limit:
        ids = ids[:args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = dict(scene_ids=ids, author_commit=git_head(args.author), cdfvd_commit=git_head(args.cdfvd),
                    torchmetrics=importlib.metadata.version('torchmetrics'), torch=torch.__version__,
                    torchvision=importlib.metadata.version('torchvision'), precision='float32; TF32 disabled',
                    inputs=str(args.clips), scope='public metric kernels with documented PNG/count/scene-boundary adapters; not exact paper replication')
    if args.prediction_root is not None or args.prediction_dir != 'pred_any2bokeh':
        protocol.update(prediction_root=str(args.prediction_root or args.clips), prediction_dir=args.prediction_dir)
    if args.reference_features:
        previous = json.loads((args.reference_features/'protocol.json').read_text())
        for key in ('scene_ids','author_commit','cdfvd_commit','torchmetrics','torch','torchvision','precision','inputs'):
            assert previous[key] == protocol[key], f'GT feature protocol mismatch: {key}'
        protocol['reference_features'] = str(args.reference_features)
    plan = args.output / 'protocol.json'
    if plan.exists():
        assert json.loads(plan.read_text()) == protocol
    else:
        write_json(plan, protocol)
    methods = {'torchmetrics': torchmetrics_scores, 'vfid': vfid_scores, 'fvd': fvd_scores}
    for name, method in methods.items():
        if args.phase not in ('all', name):
            continue
        result = args.output / f'{name}.json'
        if result.exists():
            saved = json.loads(result.read_text())
            assert saved['video_count'] == len(ids) and saved['protocol'] == protocol
            print('REUSE', result, flush=True)
            continue
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        print('START', name, 'scenes', len(ids), flush=True)
        if name == 'fvd':
            # The 24 GiB VideoMAE model must not overlap another FVD extraction
            # while the two diffusion workers and image metrics are active.
            lock=Path(__file__).resolve().parents[1]/'logs/fvd-gpu0.lock'
            lock.parent.mkdir(exist_ok=True)
            with lock.open('a') as handle:
                fcntl.flock(handle,fcntl.LOCK_EX)
                value = method(args, ids)
        else:
            value = method(args, ids)
        assert all(np.isfinite(x) for x in value['values'].values())
        value.update(protocol=protocol, elapsed_seconds=time.monotonic() - start,
                     peak_allocated_GiB=torch.cuda.max_memory_allocated() / 2**30)
        write_json(result, value)
        torch.cuda.empty_cache()
        print('DONE', name, value['values'], 'seconds', value['elapsed_seconds'], flush=True)


if __name__ == '__main__':
    main()
