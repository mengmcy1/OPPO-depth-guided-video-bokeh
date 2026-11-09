"""Initialize static Gaussians from supplied mesh depths and three RGB views.

No fitting/training. The single exported mesh's time is unspecified; use only
after checking its projections against the selected RGB frames.
"""
import argparse
from pathlib import Path
import sys
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import trimesh
from trimesh.ray.ray_pyembree import RayMeshIntersector

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.camvfx4d import load_camera_frames, intrinsics_from_fov, read_rgb


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("scene", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--views", nargs="+", type=int, default=[0, 16, 33])
    p.add_argument("--size", type=int, default=512)
    a = p.parse_args()
    scene = a.scene / "pinhole"
    mesh = trimesh.load(scene / "points3d_opencv.ply", process=False)
    mesh.vertices = np.asarray(mesh.vertices) * np.array([1., -1., 1.])
    ray = RayMeshIntersector(mesh)
    frames, metadata = load_camera_frames(scene / "transforms_train.json")
    K = intrinsics_from_fov(metadata, a.size, a.size)
    yy, xx = np.mgrid[:a.size, :a.size]
    camera_rays = np.stack((xx.ravel() + .5, yy.ravel() + .5, np.ones(xx.size)), -1) @ np.linalg.inv(K).T
    means, colors, scales, quats, sources = [], [], [], [], []
    reference_depth = None
    for view_id in a.views:
        camera = frames[(view_id, 1)]["c2w"]
        directions = camera_rays @ camera[:3, :3].T
        points, indices, _ = ray.intersects_location(np.broadcast_to(camera[:3, 3], directions.shape).copy(),
                                                    directions, multiple_hits=False)
        depth = ((points - camera[:3, 3]) @ np.linalg.inv(camera[:3, :3]).T)[:, 2]
        valid = depth > 0
        points, indices, depth = points[valid], indices[valid], depth[valid]
        if reference_depth is None:
            reference_depth = np.zeros(xx.size, np.float32)
            reference_depth[indices] = depth
        rgb = read_rgb(scene / str(view_id) / "000001.png")
        if rgb.shape[:2] != (a.size, a.size):
            import cv2
            rgb = cv2.resize(rgb, (a.size, a.size), interpolation=cv2.INTER_AREA)
        sigma = depth / K[0, 0]
        quaternion = Rotation.from_matrix(camera[:3, :3]).as_quat()[[3, 0, 1, 2]]
        means.append(points); colors.append(rgb.reshape(-1, 3)[indices])
        scales.append(np.stack((sigma * .8, sigma * .8, sigma * .01), -1))
        quats.append(np.broadcast_to(quaternion, (len(points), 4)))
        sources.append(np.full(len(points), view_id))
        print(view_id, "points", len(points), "coverage", len(points) / xx.size, flush=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.output, means=np.concatenate(means).astype(np.float32), colors_srgb=np.concatenate(colors),
             scales=np.concatenate(scales).astype(np.float32), quats=np.concatenate(quats).astype(np.float32),
             source_frames=np.concatenate(sources), K=K.astype(np.float32),
             world_to_camera=np.linalg.inv(frames[(a.views[0], 1)]["c2w"]).astype(np.float32),
             reference_depth=reference_depth.reshape(a.size, a.size),
             initializer=np.array("provided Y-reflected mesh first-hit depths + 3 posed RGB images; no optimization"))


if __name__ == "__main__":
    main()
