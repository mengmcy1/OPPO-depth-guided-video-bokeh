"""Plan B geometry scenes deterministically from a frozen registry and quota plan."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from tempfile import mkdtemp

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
GENERATION_CODE = (
    "route3_v2data_B_plan.py", "route3_v2data_B_batch.py", "route3_v2data_B_scene.py",
    "route3_v2data_B_labels.py", "route3_v2data_B_render.py", "route3_v2data_B_inputs.py",
    "route3_v2data_B_layout.py", "route3_ovg_1c.py", "route3_ovg_data.py",
    "route3_ovg_subpixel.py", "route3_ovg_phase2_blender.py", "route3_blender_b2_geometry.py",
    "route3_v2data_G_priors.py", "route3_ovg_r2_priors.py", "route3_depthsplat_depth.py",
)


def load(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generator_code():
    return {name:digest(ROOT/"scripts"/name) for name in GENERATION_CODE}


def qualification():
    pilot = DATA/"pilot"
    checks = DATA/"finalization-validation"
    frame_audits = [pilot/"thin-reviewed/labels/radiance-audit.json", *sorted(pilot.glob("case-*/labels/radiance-audit.json"))]
    row_audits = [load(a).get("audits",[]) for a in frame_audits]
    queue = load(pilot/"radiance-scenes.json")
    failures = [row for row in queue if row.get("status") == "failed"]
    geometry_file = pilot/"full-geometry-scenes.json"
    geometry = load(geometry_file)
    raw = load(DATA/"3d-front-status.json")
    indoor = DATA/"sources/infinigen-indoor-cost-0/native-45-render"
    indoor_cost = load(indoor/"cost.json")
    room = load(checks/"3d-front-room-validation.json")
    room_review = load(DATA/"source-candidates/3d-front.json")
    optics = pilot/"thin-optical-30/labels/optical-audit.json"
    priors = checks/"formal-input-contract/validation.json"
    registry = load(DATA/"asset-registry-provisional.json")
    complete_radiance = sum(len({row["frame"] for row in rows}) == 13 for rows in row_audits)
    blockers = []
    for condition, reason in (
        (len(geometry) == 10 and all(row["status"] in {"passed", "failed"} for row in geometry), "10 pilot geometry and K16 verdicts"),
        (complete_radiance + sum(not any(path.parents[1].name == row["scene"] for path in frame_audits)
                                 for row in failures) == 10, "remaining pilot radiance dispositions"),
        (len(load(optics).get("optical",[])) == 4, "first two-focus, two-aperture optical reference"),
        (load(priors).get("passed"), "observed-only B loader and three predicted priors"),
        (indoor_cost.get("one_frame_render",{}).get("exit_code") == 0 and (indoor/"frame-001.png").is_file(), "Infinigen native render"),
        (raw.get("sha256_verified"), "3D-FUTURE archive SHA-256"),
        (room.get("passed") and room.get("preview") and (DATA/room["preview"]).is_file(), "3D-FRONT room dependency and load preview"),
        (room_review.get("candidates"), "3D-FRONT layout mesh admission decision"),
        (registry.get("usable") and not registry.get("pending"), "material pool reviews are outstanding"),
    ):
        if not condition:
            blockers.append(reason)
    return dict(blockers=blockers, complete_radiance=complete_radiance,
        radiance_passed=sum(len(rows)==13 and all(row["passed"] for row in rows) for rows in row_audits),
        radiance_failed=failures,
        composition_not_evaluable=sum(row.get("composition_status")=="not_evaluable" or row.get("valid_boundary")==0
                                     for rows in row_audits for row in rows))


def freeze_registry():
    evidence = qualification()
    if evidence["blockers"]:
        raise ValueError("; ".join(evidence["blockers"]))
    source = DATA/"asset-registry-provisional.json"
    registry = load(source)
    frozen = DATA/"asset-registry-v1.json"
    if frozen.exists():
        existing = load(frozen)
        if existing.get("provisional_sha256") != digest(source):
            raise ValueError("frozen v1 has a different source snapshot")
        registry = existing
    else:
        registry.update(status="frozen-v1",provisional_sha256=digest(source),qualification=evidence)
        plan(load(QUOTA),registry)
        write_json(frozen,registry)
    return frozen



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
    result["hdri_path"] = str((DATA/hdr["render_path"]).resolve())
    result["texture_path"] = str((DATA/tex["render_path"]).resolve())
    assets = [hdr,tex]
    if category == "thin_structure":
        structure = draw_asset(rows,"procedural_structure",pool,objects)
        result["structure"] = {
            name: structure[name] for name in ("shape","width_px_at_512","parameter_segment")
        }
        assets.append(structure)
    if category == "complex_layout":
        mesh = draw_asset(rows,"mesh",pool,objects)
        result["model_path"] = str((DATA/mesh["mesh_path"]).resolve())
        assets.append(mesh)
        if any(row["category"] == "layout" and row["pool"] == pool for row in rows) and index % 2 == 0:
            backdrop = draw_asset(rows,"layout",pool,layout)
            result["layout_path"] = str((DATA/backdrop["mesh_path"]).resolve())
            assets.append(backdrop)
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
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--registry", type=Path)
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--freeze", action="store_true")
    parser.add_argument("--generator-version")
    args = parser.parse_args()
    if args.preflight:
        print(json.dumps(qualification(),ensure_ascii=False,indent=2),flush=True)
        return
    path = freeze_registry() if args.freeze else args.registry
    registry = load(path)
    if registry.get("status") != "frozen-v1":
        raise ValueError("Formal plans require the reviewed material pool v1")
    attribution = [
        {key: row[key] for key in ("key","source","asset_id","author","license","page","terms")}
        for row in registry["usable"] if row["license"].startswith("CC-BY-")
    ]
    authors = DATA/"asset-attribution-v1.json"
    if not authors.is_file():
        write_json(authors,attribution)
    elif load(authors) != attribution:
        raise ValueError("frozen v1 attribution does not match its material snapshot")
    config = json.loads(QUOTA.read_text())
    specs = plan(config,registry)
    code = generator_code()
    version_id = hashlib.sha256(json.dumps(code,sort_keys=True).encode()).hexdigest()
    version = args.generator_version or f"B-v1-{version_id[:12]}"
    sealed = DATA/"sealed-scene-plans"
    quota_sha = digest(QUOTA)
    registry_sha = digest(path)
    if sealed.exists():
        sha256 = load(sealed/"sha256.json")
        for mode in specs:
            file = sealed/(mode+".json")
            saved = load(file)
            if (digest(file) != sha256.get(mode) or saved["generator_version"] != version
                    or saved["generator_code"] != code or saved["registry_sha256"] != registry_sha
                    or saved["quota_sha256"] != quota_sha):
                raise ValueError(f"sealed {mode} plan does not match its original version and checksums")
    else:
        temporary = Path(mkdtemp(dir=DATA,prefix=".sealing-B-"))
        sha256 = {}
        for mode, rows in specs.items():
            sha256[mode] = write_json(temporary/(mode+".json"), dict(
                generator_version=version, generator_code=code, scenes=rows,
                quota_sha256=quota_sha, registry_sha256=registry_sha,
            ))
        write_json(temporary/"sha256.json", sha256)
        temporary.rename(sealed)
    manifest_path = DATA/"manifest.json"
    state = load(manifest_path)
    state.update(generator_version=version,material_registry=dict(path=str(path.resolve().relative_to(DATA.resolve())),
        sha256=registry_sha,status="frozen-v1",attributions="asset-attribution-v1.json"),scene_plan_sha256=sha256,
        asset_attribution_sha256=digest(authors))
    write_json(manifest_path,state)
    print(version,sha256,flush=True)


if __name__ == "__main__":
    main()
