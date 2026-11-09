"""Estimate aperture disocclusion from the supplied single-time triangle mesh.

This is a geometry approximation, not per-frame GT. Cast aperture rays and
recast their first intersections from the central camera to test visibility.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
import trimesh
from trimesh.ray.ray_pyembree import RayMeshIntersector

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.camvfx4d import load_camera_frames, intrinsics_from_fov, effect_parameters


def cast(intersector, origins, directions):
    origins = np.broadcast_to(origins, directions.shape).copy()
    locations, indices, _ = intersector.intersects_location(origins, directions, multiple_hits=False)
    points = np.full(directions.shape, np.nan)
    points[indices] = locations
    return points, np.isfinite(points).all(-1)


def probe(scene_dir, size, views, samples, mesh_y_flip):
    started = time.monotonic()
    parameters = effect_parameters(scene_dir)[0]
    mesh = trimesh.load(scene_dir / "pinhole/points3d_opencv.ply", process=False)
    if mesh_y_flip:
        mesh.vertices = np.asarray(mesh.vertices) * np.array([1., -1., 1.])
    ray = RayMeshIntersector(mesh)
    frames, data = load_camera_frames(scene_dir / "pinhole/transforms_train.json")
    K = intrinsics_from_fov(data, size, size)
    yy, xx = np.mgrid[:size, :size]
    pixels = np.stack((xx.ravel() + .5, yy.ravel() + .5, np.ones(xx.size)), -1)
    half = samples // 2
    index = np.arange(half)
    radius = np.sqrt((index + .5) / half)
    theta = index * math.pi * (3 - math.sqrt(5))
    aperture = np.stack((radius * np.cos(theta), radius * np.sin(theta)), -1)
    aperture = np.concatenate((aperture, -aperture)) * parameters["aperture_radius_m"]
    rows = []
    for view_id in views:
        camera = frames[(view_id, 1)]["c2w"]
        focal_points = pixels @ np.linalg.inv(K).T * parameters["focus_distance_m"]
        any_occluded = np.zeros(xx.size, bool)
        known_count = np.zeros(xx.size, np.int32)
        occluded_count = np.zeros(xx.size, np.int32)
        aperture_hit_count = 0
        for xy in aperture:
            offset = np.array([xy[0], xy[1], 0.])
            origins = camera[:3, 3] + camera[:3, :3] @ offset
            directions = (focal_points - offset) @ camera[:3, :3].T
            hit, good = cast(ray, origins, directions)
            aperture_hit_count += int(good.sum())
            central_camera_points = (hit - camera[:3, 3]) @ camera[:3, :3]
            projected = central_camera_points @ K.T
            uv = projected[:, :2] / projected[:, 2:]
            inside = good & (central_camera_points[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < size)
            inside &= (uv[:, 1] >= 0) & (uv[:, 1] < size)
            relevant = np.flatnonzero(inside)
            central_hit, central_good = cast(ray, camera[:3, 3], hit[relevant] - camera[:3, 3])
            hit_distance = np.linalg.norm(hit[relevant] - camera[:3, 3], axis=-1)
            first_distance = np.linalg.norm(central_hit - camera[:3, 3], axis=-1)
            occluded_local = central_good & (hit_distance - first_distance > np.maximum(1e-4, hit_distance * 1e-4))
            known = relevant[central_good]
            occluded = relevant[occluded_local]
            known_count[known] += 1
            occluded_count[occluded] += 1
            any_occluded[occluded] = True
        row = {"view": view_id, "image_disocclusion_fraction_any_sample": float(any_occluded.mean()),
               "aperture_ray_disocclusion_fraction": float(occluded_count.sum() / (samples * xx.size)),
               "known_aperture_ray_fraction": float(known_count.sum() / (samples * xx.size)),
               "aperture_mesh_hit_fraction": aperture_hit_count / (samples * xx.size),
               "unknown_aperture_ray_fraction": float(1 - known_count.sum() / (samples * xx.size))}
        rows.append(row)
        print(scene_dir.name, json.dumps(row), flush=True)
    return {"scene": scene_dir.name, "parameters": parameters, "resolution": [size, size], "samples": samples,
            "mesh_vertices": len(mesh.vertices), "mesh_faces": len(mesh.faces), "views": rows,
            "mean_image_disocclusion_fraction": float(np.mean([r["image_disocclusion_fraction_any_sample"] for r in rows])),
            "geometry_source": "provided PLY triangle mesh; Blender camera axes converted to OpenCV",
            "mesh_axis_transform": [1, -1 if mesh_y_flip else 1, 1],
            "occlusion_tolerance": {"absolute_m": 1e-4, "relative_to_hit_range": 1e-4, "combination": "maximum"},
            "temporal_correspondence": "mesh time not specified; NOT validated per-frame GT",
            "elapsed_seconds": time.monotonic() - started}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, default=Path("data/raw/CamVFX-4D"))
    p.add_argument("--scenes", nargs="*")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--views", nargs="+", type=int, default=[0, 16, 33])
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--mesh-y-flip", action="store_true", help="Explicit candidate PLY axis conversion; validate against multiple RGB views")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.samples < 2 or a.samples % 2:
        raise ValueError("Use an even sample count >= 2")
    result = {"method": "triangle first-hit visibility with central-camera recasting", "scenes": []}
    for scene in sorted(a.raw.iterdir()):
        if a.scenes and scene.name not in a.scenes:
            continue
        if not (scene / "pinhole/points3d_opencv.ply").exists():
            continue
        result["scenes"].append(probe(scene, a.size, a.views, a.samples, a.mesh_y_flip))
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
