"""Plan B geometry scenes deterministically from a frozen registry and quota plan."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from route3_v2data_B_front import DATA, ROOT, write_json
from route3_v2data_B_sources import canonical_pool


QUOTA = ROOT/"configs/route3-v2data-B-2026-11-20.json"
MODES = {
    "training": ("train", False),
    "development": ("dev", False),
    "test_unseen_assets": ("test", False),
    "test_unseen_occlusions": ("train", True),
}


def integer_seed(*parts):
    raw = "/".join(map(str,parts)).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def is_reserved(spec, reserved):
    return (
        spec["depth_bands"] >= reserved["configured_depth_bands_min"]
        or spec["depth_ratio"] > reserved["background_to_near_anchor_ratio_gt"]
        or spec["hole_area_fraction"] > reserved["hole_area_fraction_gt"]
        or spec["camera_baseline_m"] > reserved["camera_baseline_m_gt"]
    )


def draw_asset(rows, kind, pool, rng):
    matching = sorted((row for row in rows if row["category"] == kind and row["pool"] == pool), key=lambda row: row["key"])
    if not matching:
        raise ValueError(f"No eligible {kind} from {pool}")
    selected = matching[int(rng.integers(len(matching)))]
    if canonical_pool(selected["key"]) != pool:
        raise ValueError(f"Canonical hash mismatch: {selected['key']}")
    return selected


def scene_spec(config, registry, mode, cell, index, *, special=None):
    quota = config["geometry"]
    pool, held_out = MODES[mode]
    category, distance, motion = (cell[key] for key in ("occlusion","distance","motion"))
    seed = integer_seed(config["seed"], mode, category, distance, motion, index)
    layout, objects, materials, camera = map(np.random.default_rng, np.random.SeedSequence(seed).spawn(4))
    motion_range = quota["motion"][motion]
    ratio = float(layout.uniform(*quota["distance_bins"][distance]))
    near = float(layout.uniform(1.25,1.85))
    baseline = float(camera.uniform(*motion_range["baseline_m"]))
    spin = float(camera.uniform(*motion_range["rotation_span_deg"]))
    distant_layer = 3 if category == "multiple_layers" else 2
    holes = 0.
    if special == "large_scale_small_disparity":
        near, baseline = float(layout.uniform(4.7,5.2)), float(camera.uniform(.026,.035))
    if held_out:
        regime = index % 4
        if regime == 0:
            distant_layer = int(layout.integers(4,6))
        elif regime == 1:
            ratio = float(layout.uniform(7.2,9.2))
        elif regime == 2:
            baseline = float(camera.uniform(.50,.64))
        else:
            holes = float(layout.uniform(.38,.62))
    result = dict(
        scene_id=f"{mode}-{index:03d}-{seed:016x}", category=category, distance=distance, motion=motion,
        partition=mode, seed=seed, near_anchor_m=near, depth_ratio=ratio, depth_bands=distant_layer,
        hole_area_fraction=holes, camera_baseline_m=baseline, rotation_span_deg=spin,
        camera_trajectory={"frames":quota["camera_trajectory_frames"], "baseline_m":baseline,
                           "rotation_span_deg":spin, "profile":"centered linear translation and yaw"},
        sloped_surface=index%2 == 0, highlights=index%7 == 0, priority=special,
        assets=[], material_pool=pool,
    )
    rows = registry["usable"]
    hdr = draw_asset(rows,"hdri",pool,layout)
    tex = draw_asset(rows,"material",pool,materials)
    result["hdri_path"] = str(DATA/hdr["render_path"])
    result["texture_path"] = str(DATA/tex["render_path"])
    assets = [hdr,tex]
    if category == "thin_structure":
        structure = draw_asset(rows,"procedural_structure",pool,objects)
        result["structure"] = {
            name: structure[name] for name in ("shape","width_px_at_512","parameter_segment")
        }
        assets.append(structure)
    if category == "complex_layout":
        mesh = draw_asset(rows,"mesh",pool,objects)
        result["model_path"] = str(DATA/mesh["mesh_path"])
        assets.append(mesh)
    result["assets"] = [row["key"] for row in assets]
    if is_reserved(result, config["reserved_configuration"]) != held_out:
        raise ValueError(f"Reserved configuration mismatch for {result['scene_id']}")
    return result


def plan(config, registry):
    quota = config["geometry"]["quota_cells"]
    assert len(quota) == 54 and sum(row["training_scenes"] for row in quota) == 300
    specs = {mode: [] for mode in MODES}
    for cell in quota:
        for index in range(cell["training_scenes"]):
            priority = None
            if cell["distance"] == "small" and cell["motion"] == "small_translation" and index < 2:
                priority = "large_scale_small_disparity"
            spec = scene_spec(config,registry,"training",cell,len(specs["training"]),special=priority)
            specs["training"].append(spec)
    for category, count in config["development"]["count_by_occlusion"].items():
        cells = [cell for cell in quota if cell["occlusion"] == category]
        for offset in range(count):
            draft = scene_spec(config,registry,"development",cells[3*offset+offset],len(specs["development"]))
            specs["development"].append(draft)
    for mode, case in (("test_unseen_assets","unseen_assets"),("test_unseen_occlusions","unseen_occlusions")):
        for index in range(config["test"]["groups"][case]["count"]):
            cell = quota[(index*13 + (7 if case == "unseen_occlusions" else 0)) % len(quota)]
            specs[mode].append(scene_spec(config,registry,mode,cell,index))
    identifiers = Counter(item["scene_id"] for rows in specs.values() for item in rows)
    if any(count != 1 for count in identifiers.values()):
        raise ValueError("Scene identity collision")
    return specs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--generator-version", required=True)
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text())
    if registry.get("status") != "frozen-v1":
        raise ValueError("Formal plans require the reviewed material pool v1")
    config = json.loads(QUOTA.read_text())
    specs = plan(config,registry)
    sealed = DATA/"sealed-scene-plans"
    sealed.mkdir(exist_ok=True)
    sha256 = {}
    for mode, rows in specs.items():
        path = sealed/(mode+".json")
        if path.exists():
            raise FileExistsError(path)
        sha256[mode] = write_json(path, dict(
            generator_version=args.generator_version, scenes=rows,
            quota_sha256=hashlib.sha256(QUOTA.read_bytes()).hexdigest(),
            registry_sha256=hashlib.sha256(args.registry.read_bytes()).hexdigest(),
        ))
    write_json(sealed/"sha256.json", sha256)
    print(sha256, flush=True)


if __name__ == "__main__":
    main()
