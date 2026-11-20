"""Compute B-tier subpixel, peeled center, and K16 labels from the saved mesh."""
import argparse
from pathlib import Path
import shutil
import time

import numpy as np

from route3_blender_b2_geometry import World
from route3_ovg_subpixel import (
    SPEC, TRAIN_FRAMES, check_pass, compare, intersect, rectangle_pixels, save_json, save_npz, two_layers,
)


WIDTH = 512
CHECK_FRAMES = [1, 30, 60]


class StaticWorld(World):
    def geometry(self, frame, background=False):
        # v2 B scenes use rigid background and foreground meshes.
        return super().geometry(1, background)


def compressed_fields(data):
    return {
        key: value.astype(np.float16)
        if value.dtype == np.float32 and key not in {"camera_c2w", "K", "aperture_xy"}
        else value
        for key, value in data.items()
    }


def sample_intersections(world, frame, xy, grid):
    return two_layers(*intersect(world, frame, xy, grid))


def choose_grid(world, name):
    compar = []
    thin_compar = []
    for frame in CHECK_FRAMES:
        center = world.center(frame, 1)
        xy = rectangle_pixels(center["depth_axial_m"], 8192)
        old, high = [sample_intersections(world, frame, xy, g) for g in (4, 8)]
        values = compare(old, high)
        compar.append(dict(frame=frame, passed=check_pass(values), **values))

        # Include projected parts of a narrow structure even if no pixel center hits it.
        thin_ids = [oid for oid, label in zip(world.ids, world.names) if label == "foreground_structure"]
        if not thin_ids:
            continue
        import cv2

        oid = thin_ids[0]
        points = world.d[f"object_{oid}_vertices"]
        transform = world.transforms[frame-1, world.ids.index(oid)]
        points = points @ transform[:3, :3].T + transform[:3, 3]
        camera = world.cameras[frame-1]
        cam = (points-camera[:3, 3]) @ camera[:3, :3]
        image = cam @ world.K.T
        xy_bounds = image[:, :2] / image[:, 2:]
        low = np.floor(xy_bounds.min(0)-2).astype(int).clip(0, WIDTH-1)
        high = np.ceil(xy_bounds.max(0)+2).astype(int).clip(0, WIDTH-1)
        box = np.zeros((WIDTH, WIDTH), bool)
        box[low[1]:high[1]+1, low[0]:high[0]+1] = True
        interior = cv2.erode(np.uint8(center["object_id"] == oid), np.ones((3, 3), np.uint8)).astype(bool)
        y, x = np.nonzero(box & ~interior)
        ids = np.linspace(0, len(x)-1, min(8192, len(x)), dtype=int)
        xx = np.c_[x[ids], y[ids]]
        if not len(xx):
            continue
        for shift in [(0, 0), (.5, 0)]:
            counts = [(intersect(world, frame, xx, n, shift)[2] == oid).mean(-1) for n in (8, 16)]
            boundary = (counts[1] > 0) & (counts[1] < 1)
            error = float(np.abs(counts[0]-counts[1])[boundary].mean()) if boundary.any() else None
            thin_compar.append(dict(
                frame=frame, shift=list(shift), boundary=int(boundary.sum()),
                coverage_mae=error, passed=error is not None and error <= SPEC["thin_max_mean_coverage_error"],
                interior_pixels=int(interior.sum()), interior_texture_phase="requires matched Cycles images" if interior.any() else "not_applicable",
            ))
    grid = 4 if all(row["passed"] for row in compar) else 8
    if thin_compar:
        grid = 8 if all(row["passed"] for row in thin_compar) else 16
    save_json(name / "labels/geometry-fixed-checks.json", dict(grid=grid, subpixel=compar, thin=thin_compar))
    return grid


def pinhole(world, frame, grid, scene):
    w = world.size
    center = world.center(frame, 1)
    peeled = world.center(frame, 1, background=True)
    parts = {}
    true_coverage_error = 0.
    for y in range(0, w, 16):
        yy, xx = np.mgrid[y:min(y+16, w), :w]
        xy = np.c_[xx.ravel(), yy.ravel()]
        depths, normal, object_ids = intersect(world, frame, xy, grid)
        labels = two_layers(depths, normal, object_ids)
        a, b = labels["depth1"], labels["depth2"]
        valid = labels["valid_alpha"]
        dual = b > a
        cutoff = np.full_like(a, 1000.)
        cutoff[dual] = 2 / (1/a[dual] + 1/b[dual])
        coverage = ((depths > 0) & (depths < cutoff[:, None])).mean(-1)
        true_coverage_error = max(true_coverage_error, float(np.abs(coverage[valid]-labels["alpha"][valid]).max(initial=0)))
        if (b[labels["valid_dual"]] <= a[labels["valid_dual"]]).any() or not np.isfinite(labels["normal"]).all():
            raise ValueError(f"invalid ordered depth or normals in frame {frame}")
        labels["cutoff_m"] = cutoff
        for key, value in labels.items():
            parts.setdefault(key, []).append(value)
    if true_coverage_error != 0:
        raise ValueError(f"cutoff coverage disagrees with existing B convention in frame {frame}: {true_coverage_error}")
    data = {key: np.concatenate(rows).reshape(w,w,*rows[0].shape[1:]) for key, rows in parts.items()}
    if not data["valid"].any():
        raise ValueError(f"empty pinhole labels in frame {frame}")
    # Keep the axial cutter at float32 so the foreground and complement use the same render boundary.
    values = compressed_fields(dict(data, subpixel_grid=np.asarray(grid, dtype=np.int16), **center))
    values["cutoff_m"] = data["cutoff_m"]
    save_npz(scene/"labels"/f"f{frame:03d}-pinhole.npz", **values)
    save_npz(scene/"labels"/f"f{frame:03d}-peeled.npz", **compressed_fields(peeled))
    boundary = data["valid_alpha"] & (data["alpha"] > 0) & (data["alpha"] < 1)
    dual = data["valid_dual"]
    relative = data["depth2"][dual] / data["depth1"][dual]
    return dict(
        frame=frame, pixels=w*w, boundary=int(boundary.sum()), valid_dual=int(dual.sum()),
        relative_depth_histogram=np.histogram(relative, [1., 1.15, 1.65, np.inf])[0].tolist(),
        alpha_exact=True,
    )


def labels(scene, frames, focus, aperture):
    if shutil.disk_usage(scene).free < 60 * 2**30 + max(1, len(frames)) * 0.5 * 2**30:
        raise RuntimeError("Disk reserve would be exceeded for geometry label batch")
    start = time.monotonic()
    world = StaticWorld(scene/"labels/scene-geometry.npz", WIDTH)
    transforms = world.transforms
    if not np.allclose(transforms, transforms[0], atol=0, rtol=0):
        raise ValueError("Static geometry cannot be reused for moving mesh transforms")
    grid = choose_grid(world, scene)
    records = []
    for frame in frames:
        row = pinhole(world, frame, grid, scene)
        if aperture:
            row["K16"] = []
            for index, focus_m in enumerate(focus):
                data, stats = world.aperture(frame, 16, 1.4, focus_m, track=True)
                save_npz(scene/"labels"/f"f{frame:03d}-focus{index}-K16.npz", **compressed_fields(data))
                stats["never_visible_rays"] = stats["hidden_rays"] - stats["recoverable_rays"]
                stats["disoccluded_aperture_ray_fraction"] = stats["hidden_rays"]/(16*WIDTH**2)
                row["K16"].append(stats)
                del data
        records.append(row)
        print("B_LABEL", frame, row["boundary"], row["valid_dual"], flush=True)
    save_json(scene/"labels/scene-supervision.json", dict(
        rows=records, grid=grid, K16=aperture, seconds=time.monotonic()-start,
        uses_all_60_observed_frames_for_visibility=True,
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--focus", type=float, nargs=2, required=True)
    parser.add_argument("--frames", type=int, nargs="+", default=TRAIN_FRAMES)
    parser.add_argument("--skip-aperture", action="store_true", help="Run just the existing fixed pinhole checks and labels")
    args = parser.parse_args()
    labels(args.scene, args.frames, args.focus, not args.skip_aperture)


if __name__ == "__main__":
    main()
