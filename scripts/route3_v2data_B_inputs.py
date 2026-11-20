"""Read formal B observations and prepare their input-only geometry priors."""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import time

import numpy as np

from route3_v2data_B_front import DATA, ROOT
from route3_ovg_subpixel import TRAIN_FRAMES, save_json, save_npz


FORMAL_DATA = DATA/"v2"
PARTITIONS = ("training", "development", "test_unseen_assets", "test_unseen_occlusions")


def formal_unit(name):
    if Path(name).name != name:
        return None
    role = name.split("-", 1)[0]
    if role in PARTITIONS:
        path = FORMAL_DATA/"units"/role/name
        if (path/"manifest.json").is_file():
            return path
    return None


@lru_cache(maxsize=16)
def cameras(unit):
    with (unit/"observations/cameras.json").open() as stream:
        rows = json.load(stream)
    k = np.asarray(rows["K"], dtype=np.float32)
    poses = np.asarray(rows["poses"], dtype=np.float32)
    if k.shape != (3, 3) or poses.shape != (60, 4, 4):
        raise ValueError("B camera matrix or trajectory has the wrong dimensions")
    if not (np.isfinite(k).all() and np.isfinite(poses).all()
            and np.allclose(poses[:, 3], [0, 0, 0, 1])
            and np.allclose(np.linalg.det(poses[:, :3, :3]), 1, atol=1e-4)):
        raise ValueError("B camera contains invalid OpenCV transforms")
    return k, poses


def observed_rgb(unit, frame):
    with np.load(unit/"observations"/f"f{frame:03d}.npz") as fields:
        return fields["rgb_linear"].astype(np.float32)


@lru_cache(maxsize=24)
def observation(unit, frame, *, aligned=False):
    k, poses = cameras(unit)
    rgb = observed_rgb(unit, frame)
    with np.load(unit/"observations/priors"/f"f{frame:03d}.npz") as arrays:
        depth = arrays["depth"].astype(np.float32)
        confidence = arrays["confidence"].astype(np.float32)
        if aligned:
            depth[0] = arrays["aligned_sharp"]
    return dict(rgb=rgb, K=k, pose=poses[frame-1], depth=depth, confidence=confidence)


def supervision(unit, frame, keys):
    with np.load(unit/"labels"/f"f{frame:03d}-pinhole.npz") as arrays:
        fields = {key: arrays[key] for key in keys}
    return {key: value.astype(np.float32) if value.dtype.kind == "f" else value
            for key, value in fields.items()}


def library_unit_paths(role):
    partition = {"train": "training", "dev": "development"}[role]
    return sorted(path.parent for path in (FORMAL_DATA/"units"/partition).glob("*/manifest.json"))


def to_srgb8(linear):
    x = np.clip(linear, 0, 1)
    srgb = np.where(x <= .0031308, 12.92*x, 1.055*np.power(x,1/2.4)-.055)
    return np.rint(srgb*255).astype(np.uint8)


def calculate_depths(net, images, matrices, poses):
    from route3_ovg_r2_priors import stack_depth

    # A five-view window supplies each frame's three nearest model views
    # while bounding memory for the native 512-pixel B frames.
    width = min(5, len(images))
    windows = {}
    depths = []
    for index in range(len(images)):
        start = min(max(0, index-width//2), len(images)-width)
        selection = slice(start,start+width)
        if start not in windows:
            windows[start] = stack_depth(net,images[selection],matrices[selection],poses[selection])
        depths.append(windows[start][index-start])
    return np.stack(depths)


def precalculate(unit):
    import cv2
    import torch
    from route3_depthsplat_depth import network
    from route3_ovg_1c import fit
    from route3_ovg_r2_priors import MODEL, support_map
    from route3_v2data_G_priors import models, sharp_prior, triangulation

    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    spec = json.loads((unit/"scene-spec.json").read_text())
    seed = int(spec["seed"])
    name = spec.get("scene_id", unit.name)
    out = unit/"observations/priors"
    if (out/"summary.json").is_file():
        return
    start = time.monotonic()
    k, poses = cameras(unit)
    poses = poses[np.asarray(TRAIN_FRAMES)-1]
    matrices = np.repeat(k[None,...], len(poses), axis=0)
    images = np.stack([to_srgb8(observed_rgb(unit, frame)) for frame in TRAIN_FRAMES])
    net, _ = network(ROOT.parent/"depthsplat-r13-20261117", ROOT.parent/"dinov2-depthsplat-r13",
                     ROOT/"models/DepthSplat"/MODEL)
    net = net.cuda().eval()
    ds = calculate_depths(net, images, matrices, poses)
    del net
    torch.cuda.empty_cache()
    sharp, raft, transform = models()
    tri = triangulation(raft, transform, images, matrices, poses)
    del raft
    torch.cuda.empty_cache()
    reports = []
    for index, frame in enumerate(TRAIN_FRAMES):
        sharp_z, sharp_q = sharp_prior(sharp, images[index], k, reference=True)
        tri_z, tri_q = tri[index]
        support = support_map(ds, matrices, poses, index)
        ds_valid = np.isfinite(ds[index]) & (ds[index] >= .05) & (ds[index] <= 200) & (support >= 2)
        ds_q = np.where(ds_valid, np.minimum(support/7.,1.), 0.)
        depth = np.stack([sharp_z, np.where(ds_valid, ds[index],0), tri_z]).astype(np.float32)
        confidence = np.stack([sharp_q, ds_q, tri_q]).astype(np.float32)
        aligned, report = fit(name, frame, dict(depth=depth,confidence=confidence), scene_seed=seed)
        save_npz(out/f"f{frame:03d}.npz", depth=depth.astype(np.float16),
                 confidence=confidence.astype(np.float16), aligned_sharp=aligned.astype(np.float16))
        reports.append(report)
    save_json(out/"summary.json",dict(scene=name, frames=TRAIN_FRAMES, input="observations RGB and cameras",
                                    alignment="1c SHARP inverse-depth", rows=reports,
                                    seconds=time.monotonic()-start))
    print("B priors",name,len(reports),"seconds",round(time.monotonic()-start,2),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    args = parser.parse_args()
    precalculate(args.scene)
