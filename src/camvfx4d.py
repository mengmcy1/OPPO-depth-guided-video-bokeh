"""CamVFX metadata and OpenCV camera/depth helpers; never writes raw data."""
import csv
import json
import struct
from pathlib import Path
import numpy as np


def read_rgb(path):
    """Read display RGB as float32 [0,1], preserving both 8/16-bit PNG inputs."""
    import cv2
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim != 3 or image.shape[-1] not in (3, 4):
        raise ValueError(f"Expected RGB/RGBA image: {path}")
    if image.dtype not in (np.uint8, np.uint16):
        raise ValueError(f"Expected 8/16-bit integer RGB: {image.dtype}")
    maximum = np.iinfo(image.dtype).max
    return np.ascontiguousarray(image[..., :3][..., ::-1], dtype=np.float32) / maximum


def load_camera_frames(path):
    data = json.loads(Path(path).read_text())
    frames = {}
    for item in data["frames"]:
        name = Path(item["file_path"])
        key = (int(name.parent.name), int(name.stem))
        c2w = np.array(item["transform_matrix"], dtype=np.float64)
        # NeRF/Blender c2w: camera axes right/up/back -> OpenCV right/down/front.
        c2w = c2w @ np.diag([1., -1., -1., 1.])
        frames[key] = {"c2w": c2w, "time": item.get("time"), "path": str(name)}
    return frames, data


def intrinsics_from_fov(metadata, width=512, height=512):
    focal = width / (2 * np.tan(metadata["camera_angle_x"] / 2))
    return np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1.]])


def effect_parameters(scene_dir):
    scene_dir = Path(scene_dir)
    manifest = json.loads((scene_dir / "self_dataset_manifest.json").read_text())
    scene_key = manifest["scenes"][scene_dir.name]["effect_param_scene"]
    with (scene_dir / "effect_params.csv").open() as handle:
        rows = [r for r in csv.DictReader(handle) if r["effect_scene"] == scene_key]
    result = []
    for row in rows:
        focal_mm = float(row["pinhole_focal_length_mm"])
        fstop = float(row["dof_fstop"])
        result.append({"effect_scene": scene_key, "focal_length_mm": focal_mm,
                       "sensor_size_mm": float(row["sensor_size_mm"]),
                       "focus_distance_m": float(row["dof_focus_distance_m"]), "fstop": fstop,
                       "aperture_radius_m": focal_mm / 1000 / (2 * fstop),
                       "blades": int(row["dof_blades"]), "ratio": float(row["dof_ratio"])})
    return result


def audit_scene(scene_dir, verify_images=False):
    scene_dir = Path(scene_dir)
    result = {"scene": scene_dir.name, "dof_parameters": effect_parameters(scene_dir), "effects": {}}
    result["one_dof_parameter_group"] = len(result["dof_parameters"]) == 1
    for effect in ("pinhole", "dof"):
        path = scene_dir / effect
        if not (path / "transforms_train.json").exists():
            continue
        frames, metadata = load_camera_frames(path / "transforms_train.json")
        views = sorted({k[0] for k in frames})
        deltas = {}
        for view in views:
            matrices = np.stack([v["c2w"] for k, v in sorted(frames.items()) if k[0] == view])
            deltas[str(view)] = float(np.max(np.abs(matrices - matrices[:1])))
        train = json.loads((path / "transforms_train.json").read_text())
        test = json.loads((path / "transforms_test.json").read_text())
        images = list(path.glob("*/*.png"))
        result["effects"][effect] = {"camera_count": len(views), "pose_records": len(frames),
                                      "frames_per_view": {str(v): sum(k[0] == v for k in frames) for v in views},
                                      "max_extrinsic_abs_delta_by_view": deltas,
                                      "all_views_fixed": all(x == 0 for x in deltas.values()),
                                      "train_test_json_identical": train == test,
                                      "intrinsics_512": intrinsics_from_fov(metadata).tolist(),
                                      "image_files_present": len(images),
                                      "payload_bytes_on_disk": sum(x.stat().st_size for x in path.rglob("*") if x.is_file())}
        if verify_images:
            dimensions, formats, by_view, bad_headers = {}, {}, {}, []
            actual_pairs = set()
            for image in images:
                with image.open("rb") as handle:
                    header = handle.read(26)
                if len(header) < 26 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
                    bad_headers.append(str(image.relative_to(scene_dir)))
                    continue
                width, height = struct.unpack(">II", header[16:24])
                key = f"{width}x{height}"
                dimensions[key] = dimensions.get(key, 0) + 1
                fmt = f"bit_depth={header[24]},color_type={header[25]}"
                formats[fmt] = formats.get(fmt, 0) + 1
                pair = (int(image.parent.name), int(image.stem))
                actual_pairs.add(pair)
                by_view[str(pair[0])] = by_view.get(str(pair[0]), 0) + 1
            result["effects"][effect].update(image_dimension_histogram=dimensions,
                                               png_format_histogram=formats,
                                               image_count_by_view=by_view, bad_png_headers=bad_headers,
                                               all_pose_image_pairs_present=set(frames) == actual_pairs,
                                               all_raw_files_read_only=all(not x.stat().st_mode & 0o222 for x in path.rglob("*") if x.is_file()))
    return result


def project_pointcloud_depth(points, world_to_camera, intrinsics, width, height,
                             splat_radius=1):
    """Nearest-point z-buffer. Holes stay invalid; no implicit depth filling."""
    camera = points @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
    keep = np.isfinite(camera).all(-1) & (camera[:, 2] > 1e-4)
    camera = camera[keep]
    projection = camera @ intrinsics.T
    xy = np.floor(projection[:, :2] / projection[:, 2:]).astype(np.int64)
    depth = np.full((height, width), np.inf, dtype=np.float32)
    for dy in range(-splat_radius, splat_radius + 1):
        for dx in range(-splat_radius, splat_radius + 1):
            u, v = xy[:, 0] + dx, xy[:, 1] + dy
            good = (u >= 0) & (u < width) & (v >= 0) & (v < height)
            np.minimum.at(depth, (v[good], u[good]), camera[good, 2])
    valid = np.isfinite(depth)
    return np.where(valid, depth, 0), valid
