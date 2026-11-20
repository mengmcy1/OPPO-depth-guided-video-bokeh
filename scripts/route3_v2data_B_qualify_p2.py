"""Run the preregistered B p2 qualification in an isolated sibling directory."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from threading import Event
import time

import numpy as np

from route3_v2data_B_batch import (
    FORMAL_DATA, RECORD, finish_unit, prepare_geometry, reserve, sha256, write_json,
)
from route3_v2data_B_dispatch import RunMeter, assigned_cpus, execute_on, production_lock
from route3_v2data_B_plan import load
from route3_v2data_B_release import PLAN_DIR, sealed_plan
from route3_ovg_subpixel import TRAIN_FRAMES


def label_files():
    names = ["scene-geometry.npz"]
    for frame in TRAIN_FRAMES:
        names.extend(f"f{frame:03d}-{suffix}.npz" for suffix in
                     ("pinhole", "peeled", "focus0-K16", "focus1-K16"))
    return names


def structure_checksums(unit):
    path = unit/"labels/scene-geometry.npz"
    names = ("object_names", "object_ids", "object_world_matrices")
    import hashlib
    with np.load(path) as data:
        meshes = sorted(name for name in data.files if name.endswith(("_vertices", "_faces")))
        return {name: hashlib.sha256(data[name].tobytes()).hexdigest() for name in (*names, *meshes)}


def invariance(new, old, spec, estimate, execute):
    prepare_geometry(new, spec, estimate, execute=execute)
    base_structure, patch_structure = structure_checksums(old), structure_checksums(new)
    hashes = {name: [sha256(old/"labels"/name), sha256(new/"labels"/name)] for name in label_files()}
    different = [name for name, pair in hashes.items() if pair[0] != pair[1]]
    structure_passed = base_structure == patch_structure
    return dict(
        scene_id=spec["scene_id"], category=spec["category"], partition=spec["partition"],
        structure_equal=structure_passed, label_file_count=len(hashes),
        differing_labels=different, passed=structure_passed and not different,
        structure_before=base_structure, structure_after=patch_structure, label_sha256=hashes,
    )


def final_smoke(unit, spec, version, execute):
    result = finish_unit(unit, spec, version, execute=execute)
    digest = write_json(unit/"manifest.json", result)
    presence = load(unit/"labels/foreground-presence.json")
    placement = load(unit/"labels/multilayer-placement.json")
    policy = placement.get("layers", [])
    rear_layers = policy[1:]
    in_range = all(.3 <= row["middle_overlap"] <= .7 for row in rear_layers)
    if spec["category"] == "multiple_layers" and (len(rear_layers) < 1 or not in_range):
        raise ValueError("multilayer smoke qualification needs every midpoint overlap in [0.3, 0.7]")
    return dict(
        scene_id=spec["scene_id"], category=spec["category"], passed=True, manifest_sha256=digest,
        audited_frames=len(presence["frames"]), minima=presence["foreground_minima"],
        fallback_count=placement.get("fallback_count", 0),
        fallback_by_layer={row["name"]: row["fallback"] for row in policy}, bytes=result["bytes"],
    )


def qualify(dest, estimate):
    config = load(RECORD)["v2"]["p2_qualification_policy"]
    from route3_v2data_B_release import p2_generation_code
    code = p2_generation_code()
    specs = {}
    for mode in ("training", "development", "test_unseen_assets", "test_unseen_occlusions"):
        plan, _ = sealed_plan(PLAN_DIR/(mode+".json"))
        specs.update((row["scene_id"], row) for row in plan["scenes"])
    dest.mkdir(parents=True)
    (dest/"smoke").mkdir()
    (dest/"invariance").mkdir()
    reserve(dest, estimate * len(config["smoke_training_scene_ids"]))
    (dest/"proof.json").touch()
    smoke, checks = {}, {}
    start = time.monotonic()
    test_ids = config["smoke_training_scene_ids"]
    controls = config["invariance_scene_ids"]
    version = config["version"]

    def publish(status):
        proof = dict(status=status, version=version, generation_code=code, policy=config,
                     smoke=smoke, invariance=checks, seconds=time.monotonic()-start)
        write_json(dest/"proof.json", proof)

    with production_lock():
        if not (FORMAL_DATA/"stop-after-current").is_file():
            raise ValueError("qualification requires the formal dispatcher to remain drained")
        allocation = assigned_cpus()
        abort = Event()
        meter = RunMeter(allocation, abort)
        cpu_exec = partial(execute_on, allocation[1], 8, abort)
        gpu_exec = partial(execute_on, allocation[2], 18, abort)
        review = iter([(scene_id, "smoke") for scene_id in test_ids]
                      + [(scene_id, "invariance") for scene_id in controls])
        cpu = gpu = ready = None
        pending = next(review, None)
        publish("running")
        with ThreadPoolExecutor(max_workers=2) as pool:
            while cpu or gpu or ready or pending:
                if ready and gpu is None:
                    unit, spec = ready
                    gpu = (pool.submit(final_smoke, unit, spec, version, gpu_exec), spec)
                    ready = None
                if pending and not cpu and not ready:
                    scene_id, kind = pending
                    spec = specs[scene_id]
                    unit = dest/kind/scene_id
                    if kind == "smoke":
                        if spec["partition"] != "training":
                            raise ValueError("qualification scene must be from the sealed training plan")
                        job = pool.submit(prepare_geometry, unit, spec, estimate, execute=cpu_exec)
                    else:
                        old = FORMAL_DATA/"units"/spec["partition"]/scene_id
                        job = pool.submit(invariance, unit, old, spec, estimate, cpu_exec)
                    cpu = (job, kind, spec, unit)
                    pending = next(review, None)
                meter.sample(labels_busy=cpu is not None, gpu_busy=gpu is not None)
                if cpu and cpu[0].done():
                    work, kind, spec, unit = cpu
                    scene_id = spec["scene_id"]
                    try:
                        result = work.result()
                        if kind == "smoke":
                            ready = (unit, spec)
                        else:
                            checks[scene_id] = result
                    except (RuntimeError, ValueError, OSError) as exc:
                        (smoke if kind == "smoke" else checks)[scene_id] = dict(
                            scene_id=scene_id, category=spec["category"], passed=False,
                            failure=str(exc), stage="geometry",
                        )
                    cpu = None
                    publish("running")
                if gpu and gpu[0].done():
                    work, spec = gpu
                    scene_id = spec["scene_id"]
                    try:
                        smoke[scene_id] = work.result()
                    except (RuntimeError, ValueError, OSError) as exc:
                        smoke[scene_id] = dict(scene_id=scene_id, category=spec["category"],
                                               passed=False, failure=str(exc), stage="render_and_accept")
                    print(scene_id, smoke[scene_id]["passed"], flush=True)
                    gpu = None
                    publish("running")
                if abort.is_set():
                    pending = ready = None
                time.sleep(.5)
    passed = len(smoke) == len(test_ids) and len(checks) == len(controls) and all(
        row["passed"] for row in (*smoke.values(), *checks.values())
    ) and not abort.is_set()
    publish("passed" if passed else "failed")
    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--measured-peak-unit-bytes", type=int, required=True)
    args = parser.parse_args()
    if args.measured_peak_unit_bytes <= 0:
        parser.error("the unit size must be positive")
    raise SystemExit(0 if qualify(args.output.resolve(), args.measured_peak_unit_bytes) else 1)
