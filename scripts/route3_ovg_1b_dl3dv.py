"""Prepare a small, authorized DL3DV shard with input/held-out view pairs."""
import argparse
from pathlib import Path
from functools import lru_cache
import json
import os
import subprocess
from zipfile import ZipFile

import cv2
import numpy as np

from route3_ovg_1b_common import CONFIG, OUT, ROOT, guard, read, link
from route3_ovg_r2_priors import infer, resize_inputs, stack_depth, support_map, MODEL
from route3_ovg_subpixel import save_json, save_npz

REPO = CONFIG['dl3dv']['repo']
TERMS = 'https://github.com/DL3DV-10K/Dataset/blob/main/License.md'


def metadata(meta):
    """Nerfstudio uses OpenGL camera axes. OVG uses OpenCV and axial depth."""
    flip = np.diag([1., -1., -1., 1.])
    rows = []
    for row in sorted(meta['frames'], key=lambda x: x['file_path']):
        fields = dict(meta, **row)
        K = [[fields['fl_x'], 0., fields['cx']], [0., fields['fl_y'], fields['cy']], [0., 0., 1.]]
        pose = np.asarray(row['transform_matrix'], np.float64) @ flip
        if (pose.shape != (4, 4) or not np.isfinite(pose).all() or
                abs(np.linalg.det(pose[:3, :3]) - 1.) > .01 or not np.allclose(pose[3], [0, 0, 0, 1.])):
            continue
        rows.append(dict(filename=Path(row['file_path']).name, K=K, c2w=pose.tolist(),
                         resolution=[fields['w'], fields['h']]))
    return rows


def choose(rows, count):
    if len(rows) < count:
        raise ValueError('Fewer than one complete 480P pose window')
    centers = np.array([r['c2w'] for r in rows])[:, :3, 3]
    baseline = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    candidates = [(np.median(baseline[i:i+count-1]), i) for i in range(len(rows)-count+1)
                  if np.min(baseline[i:i+count-1]) > 1e-6]
    if not candidates:
        raise ValueError('No moving-camera window')
    mid = np.median([value for value, _ in candidates])
    start = min(candidates, key=lambda r: (abs(r[0]-mid), r[1]))[1]
    return rows[start:start+count]


def color_and_depth_mask(a, b, za, zb, Ka, Kb, Pa, Pb):
    """Cross-view estimated-depth, rigid flow and forward/backward consistency."""
    h, w = za.shape
    y, x = np.mgrid[:h, :w].astype(np.float32)
    v = np.stack((x + .5, y + .5, np.ones_like(x)), -1) @ np.linalg.inv(Ka).T
    world = (v * za[..., None]) @ Pa[:3, :3].T + Pa[:3, 3]
    cam = (world - Pb[:3, 3]) @ Pb[:3, :3]
    projected = cam @ Kb.T
    uv = (projected[..., :2] / np.maximum(cam[..., 2:], 1e-6)).astype(np.float32) - .5
    before = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    after = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
    estimator = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    forward = estimator.calc(before, after, None)
    backward = estimator.calc(after, before, None)
    dst = np.stack((x, y), -1) + forward
    back_at_dst = cv2.remap(backward, dst[..., 0], dst[..., 1], cv2.INTER_LINEAR)
    observed_depth = cv2.remap(zb, uv[..., 0], uv[..., 1], cv2.INTER_NEAREST)
    accepted = ((uv[..., 0] >= 0) & (uv[..., 0] < w-1) & (uv[..., 1] >= 0) & (uv[..., 1] < h-1) &
                (cam[..., 2] > 0) & (observed_depth > 0) &
                (abs(observed_depth-cam[..., 2]) <= .04*cam[..., 2]) &
                (np.linalg.norm(back_at_dst+forward, axis=-1) <= 1.) &
                (np.linalg.norm(dst-uv, axis=-1) <= 1.5))
    return accepted


def stage(source, scene):
    guard()
    dest = OUT / 'R/windows' / scene
    p = dest / 'manifest.json'
    if p.exists():
        return p
    archive = ZipFile(source) if source.is_file() else None
    try:
        if archive is not None:
            names = archive.namelist()
            transforms = [m for m in names if Path(m).name == 'transforms.json']
            if len(transforms) != 1:
                raise ValueError('DL3DV archive needs one transforms.json')
            meta = json.loads(archive.read(transforms[0]))
            images = {Path(n).name: n for n in names if Path(n).parent.name == 'images_8'}
        else:
            meta = read(source / 'transforms.json')
        picked = choose(metadata(meta), CONFIG['dl3dv']['frames_per_window'])
        rows = []
        for r in picked:
            name = r.pop('filename')
            image = dest / '480p' / name
            if not image.exists():
                image.parent.mkdir(parents=True, exist_ok=True)
                if archive is None:
                    native = source / 'images_8' / name
                    if not native.is_file():
                        raise FileNotFoundError(native)
                    link(native, image)
                else:
                    image.write_bytes(archive.read(images[name]))
            decoded = cv2.imread(str(image))
            if decoded is None or max(decoded.shape[:2]) != 480:
                raise ValueError(f'Missing 480P RGB at {name}')
            h, w = decoded.shape[:2]
            srcw, srch = r.pop('resolution')
            K = np.asarray(r['K'])
            K[:2] *= np.array([w/srcw, h/srch])[:, None]
            rows.append(dict(r, K=K.tolist(), image=str(image.relative_to(ROOT))))
    finally:
        if archive is not None:
            archive.close()
    heldout = list(range(1, len(rows), 3))
    inputs = [i for i in range(len(rows)) if i not in heldout]
    save_json(p, dict(tier='R', scene=scene, rows=rows, inputs=inputs, heldout=heldout,
                      license='CC BY-NC 4.0 and DL3DV terms (noncommercial research and education)',
                      official_repository=REPO, terms=TERMS))
    return p


def validate(net, p):
    done = p.parent / 'visibility.json'
    if done.exists():
        return read(done)
    m = read(p)
    infer(net, p)
    images, K, poses = resize_inputs(p, m['rows'])
    depths = np.stack([np.load(p.parent / 'prior' / f'f{j:03d}.npz')['depth'].astype(np.float32)
                       for j in range(len(poses))])
    inputs = np.array(m['inputs'])
    observed_depths = stack_depth(net, images[inputs], K[inputs], poses[inputs])
    for k, i in enumerate(inputs):
        support = support_map(observed_depths, K[inputs], poses[inputs], k)
        static = np.zeros_like(support, dtype=bool)
        neighbor_index = sorted(range(len(inputs)), key=lambda l: (abs(l-k), l))
        for l in [l for l in neighbor_index if l != k][:2]:
            j = inputs[l]
            static |= color_and_depth_mask(images[i], images[j], observed_depths[k],
                                           observed_depths[l], K[i], K[j], poses[i], poses[j])
        save_npz(p.parent / 'observed-prior' / f'f{i:03d}.npz',
                 depth=observed_depths[k].astype(np.float16), support=support, static=static)
    thresholds = CONFIG['dl3dv']['min_common_fraction']
    accepted = []
    for i in m['inputs']:
        closest = sorted(m['heldout'], key=lambda j: abs(i-j))
        pair, masks, fractions = [], [], []
        for j in closest:
            mask = color_and_depth_mask(images[i], images[j], depths[i], depths[j], K[i], K[j],
                                         poses[i], poses[j])
            if mask.mean() >= thresholds:
                pair.append(j)
                masks.append(mask)
                fractions.append(float(mask.mean()))
        if pair:
            file = p.parent / 'static-common' / f'f{i:03d}.npz'
            save_npz(file, neighbors=np.array(pair), mask=np.stack(masks))
            accepted.append(dict(index=i, heldout_neighbors=pair, common_fractions=fractions,
                                 mask=str(file.relative_to(ROOT))))
    result = dict(scene=m['scene'], accepted=accepted, passed=bool(accepted),
                  static_common_definition='positive cheirality, depth reprojection 4%, DIS FB ≤1 px, rigid flow disagreement ≤1.5 px',
                  min_fraction=thresholds, input_count=len(m['inputs']), heldout_count=len(m['heldout']))
    save_json(done, result)
    return result


def inventory():
    for p in sorted((OUT / 'R/windows').glob('*/manifest.json')):
        vis = p.parent / 'visibility.json'
        if vis.exists():
            yield from (('R', str(p), row['index']) for row in read(vis)['accepted'])


@lru_cache(maxsize=96)
def observed_input(manifest, index):
    import route3_ovg_data as data

    observed = data.gc_input(manifest, index)
    native = data.load(Path(manifest).parent / 'observed-prior' / f'f{index:03d}.npz')
    depths = np.zeros_like(observed['depth'])
    confidence = np.zeros_like(observed['confidence'])
    depths[1] = native['depth'].astype(np.float32)
    confidence[1] = data.confidence(native['support'] / 4, native['support'] >= 2)
    return dict(observed, depth=depths, confidence=confidence, static=native['static'])


def training_input(manifest, index, device):
    import torch
    import route3_ovg_data as data

    path = Path(manifest)
    meta = read(path)
    ids = [j for j in sorted(meta['inputs'], key=lambda j: (abs(j-index), j)) if j != index][:2]
    current = observed_input(manifest, index)
    given = data.input_record(current, [observed_input(manifest, j) for j in ids])
    given['motion'] = ~current['static']
    pairs = data.load(path.parent / 'static-common' / f'f{index:03d}.npz')
    scoring = data.input_record(current, [data.gc_input(manifest, int(j)) for j in pairs['neighbors']])
    scoring['common_neighbor'] = pairs['mask']
    scoring['motion'] = np.zeros_like(current['static'])
    convert = lambda record: {k: torch.as_tensor(v[None].copy(), device=device) for k, v in record.items()}
    common = torch.as_tensor(pairs['mask'].any(0), device=device)
    return convert(given), convert(scoring), common


def sources(root, budget):
    if root is not None:
        if root.is_file() and root.suffix.lower() == '.zip':
            yield root, root.stem
        elif (root / 'transforms.json').exists():
            yield root, root.name
        else:
            entries = [p.parent for p in root.rglob('transforms.json')]
            entries.extend(root.rglob('*.zip'))
            for p in sorted(entries)[:budget]:
                yield p, p.stem
        return
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    api.auth_check(REPO, repo_type='dataset')
    revision = api.dataset_info(REPO).sha
    for i, entry in enumerate(p for p in api.list_repo_tree(REPO, '1K', revision=revision, repo_type='dataset')
                              if getattr(p, 'path', '').endswith('.zip')):
        if i == budget:
            break
        guard()
        archive = hf_hub_download(REPO, entry.path, repo_type='dataset', revision=revision,
                                  local_dir=OUT / 'R/archives')
        yield Path(archive), Path(entry.path).stem


def run(root, budget):
    import torch
    from route3_depthsplat_depth import network

    guard()
    pending = iter(sources(root, budget))
    first = next(pending, None)
    if first is None:
        raise FileNotFoundError('No authorized DL3DV image+pose source found')
    subprocess.run(['nvidia-smi'], check=True)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0' or torch.cuda.mem_get_info()[0] < 6*2**30:
        raise RuntimeError('DL3DV prior worker GPU memory unavailable')
    cv2.setNumThreads(1)
    torch.cuda.set_per_process_memory_fraction(.20)
    net, _ = network(ROOT.parent/'depthsplat-r13-20261117', ROOT.parent/'dinov2-depthsplat-r13',
                     ROOT/'models/DepthSplat'/MODEL)
    net = net.cuda().eval()
    reports = []
    from itertools import chain
    for source, scene in chain([first], pending):
        try:
            reports.append(validate(net, stage(source, scene)))
        except (OSError, ValueError, KeyError) as exc:
            reports.append(dict(scene=scene, passed=False, reason=str(exc)))
        save_json(OUT / 'R/acquisition.json', dict(repo=REPO, terms=TERMS, scenes=reports))
        print('R_VISIBILITY', scene, reports[-1]['passed'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path)
    parser.add_argument('--budget', type=int, default=CONFIG['dl3dv']['first_scene_budget'])
    args = parser.parse_args()
    run(args.source, args.budget)
