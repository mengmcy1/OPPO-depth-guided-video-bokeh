"""Check and report the world-box placement of every sealed multilayer B scene."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from itertools import combinations
import os
from pathlib import Path
from threading import Event

import numpy as np
import time

from route3_v2data_B_batch import blender
from route3_v2data_B_dispatch import RunMeter, execute_on, production_lock
from route3_v2data_B_front import RECORD, write_json
from route3_v2data_B_inputs import FORMAL_DATA
from route3_v2data_B_placement import policy
from route3_v2data_B_presence import appearance_checks, designated_foreground
from route3_v2data_B_plan import MODES, load
from route3_v2data_B_release import PLAN_DIR, code_hash, p2_generation_code, sealed_plan
from route3_ovg_subpixel import TRAIN_FRAMES


def planned_scenes():
    plans = [sealed_plan(PLAN_DIR/(mode+".json"))[0] for mode in MODES]
    sealed = [spec for plan in plans for spec in plan["scenes"]]
    rows = [spec for spec in sealed if len(spec.get("occluder_assets", [])) > 1 or spec["depth_bands"] > 2]
    spec = load(RECORD)["v2"]["p2_qualification_policy"]["synthetic_multilayer_curve"]
    source = next(row for row in sealed if row["scene_id"] == spec["source"])
    if any(row["seed"] == spec["seed"] or row["scene_id"] == spec["scene_id"] for row in sealed):
        raise ValueError("the isolated synthetic case must not reuse a sealed ID or seed")
    synthetic = dict(source, scene_id=spec["scene_id"], seed=spec["seed"], depth_bands=spec["depth_bands"])
    return rows, synthetic


def native_checks(path, spec, cpus):
    from route3_v2data_B_labels import StaticWorld
    os.sched_setaffinity(0, set(cpus))
    world = StaticWorld(path, 512)
    groups = designated_foreground(spec, world.names)
    objects = {name:(int(obj), index) for index,(name,obj) in enumerate(zip(world.names,world.ids))}
    groups_ids = {actor:[objects[name][0] for name in names] for actor,names in groups.items()}
    depth_ranges = {}
    for actor, names in groups.items():
        ranges = []
        for name in names:
            identifier, index = objects[name]
            transform = world.transforms[0, index]
            vertex = world.d[f"object_{identifier}_vertices"]
            y = vertex @ transform[1,:3] + transform[1,3]
            ranges.append((float(y.min()), float(y.max())))
        depth_ranges[actor] = [min(value[0] for value in ranges), max(value[1] for value in ranges)]
    pairs = []
    for left, right in combinations(depth_ranges, 2):
        a, b = depth_ranges[left], depth_ranges[right]
        overlap = min(a[1], b[1])-max(a[0], b[0])
        pairs.append(dict(actors=[left,right], depth_overlap_m=max(0.,overlap),
                          triangle_intersection_pairs=0 if overlap<0 else None))
    ray, triangle_map = world.geometry(1)
    hits = {key:[] for key in groups}
    for frame in TRAIN_FRAMES:
        camera = world.cameras[frame-1]
        directions = world.rays @ camera[:3,:3].T
        hit = ray.intersects_first(np.broadcast_to(camera[:3,3],directions.shape).copy(),directions)
        valid = hit >= 0
        by_id = np.bincount(world.tri_obj[triangle_map[hit[valid]]])
        for actor, ids in groups_ids.items():
            hits[actor].append(sum(int(by_id[obj]) for obj in ids if obj < len(by_id)))
    seen, failures = appearance_checks(hits, TRAIN_FRAMES)
    for actor,row in seen.items():
        row["hits_by_frame"] = dict(zip((f"f{x:03d}" for x in TRAIN_FRAMES), hits[actor]))
    collision_free = all(row["triangle_intersection_pairs"] == 0 for row in pairs)
    return dict(
        appearance_passed=not failures, appearance_failures=failures, native_center=seen,
        evaluated_actor_depth_ranges=depth_ranges, actor_triangle_pairs=pairs, triangle_intersection_pairs=(
            0 if collision_free else None
        ), triangle_method="disjoint world-Y intervals of all triangulated actor vertices certify zero",
    )


def run_preflight(output):
    archive = load(RECORD)["v2"]
    if archive["multilayer_placement_policy_p2"] != policy(TRAIN_FRAMES):
        raise ValueError("the preregistered policy does not match the source")
    code = p2_generation_code()
    if archive["generator_patch_p2"]["generator_code_sha256"] != code_hash(code):
        raise ValueError("preflight source hash changed after preregistration")
    rows, synthetic = planned_scenes()
    known = {spec["scene_id"]:spec for spec in rows}
    training, _ = sealed_plan(PLAN_DIR/"training.json")
    sample_ids = set(archive["p2_qualification_policy"]["smoke_training_scene_ids"])
    single = [spec for spec in training["scenes"] if spec["scene_id"] in sample_ids and spec["scene_id"] not in known]
    output.mkdir(parents=True)
    cpus = sorted(os.sched_getaffinity(0))[:14]
    if len(cpus) < 13:
        raise RuntimeError("four CPU workers need 12 worker CPUs and one coordinator CPU")
    allocation = (cpus[12:], cpus[:6], cpus[6:12])
    abort = Event()
    with production_lock(), ThreadPoolExecutor(max_workers=4) as workers:
        os.sched_setaffinity(0, {cpus[12]})
        meter = RunMeter(allocation, abort)
        tasks = []
        scenes = [*rows, *single, synthetic]
        for shard in range(4):
            dest = output/f"shard-{shard:02d}"
            dest.mkdir()
            source = dest/"input.json"
            write_json(source, scenes[shard::4])
            path = dest/"preflight.json"
            execute = partial(execute_on, cpus[shard*3:(shard+1)*3], 6, abort)
            command = blender("route3_v2data_B_scene.py", "--preflight", source, "--output", path, threads=3)
            tasks.append((workers.submit(execute, command, dest/"blender.log"), path))
        while any(not task.done() for task, _ in tasks):
            meter.sample()
            time.sleep(.5)
        collected = {}
        for task, result in tasks:
            task.result()
            collected.update((row["scene_id"], row) for row in load(result))
    locations = {spec["scene_id"]:output/f"shard-{index%4:02d}"/"geometry"/spec["scene_id"]/"labels/scene-geometry.npz"
                 for index,spec in enumerate(scenes)}
    with production_lock(), ProcessPoolExecutor(max_workers=4) as workers:
        checks = []
        for index,spec in enumerate(scenes):
            key = spec["scene_id"]
            if collected[key].get("built"):
                mask = cpus[3*(index%4):3*(index%4)+3]
                checks.append((key,workers.submit(native_checks, locations[key], spec, mask)))
        for key, job in checks:
            native = job.result()
            collected[key].update(native)
            collected[key]["passed"] &= native["appearance_passed"] and native["triangle_intersection_pairs"] == 0
            print("NATIVE", key, collected[key]["passed"], flush=True)
    entries = [collected[spec["scene_id"]] for spec in rows]
    special_failures = [row for row in entries if row["scene_id"].startswith("training-129-") and not row.get("built")
                        and row.get("failure","").startswith("no feasible framed placement")]
    rest_passes = all(row["passed"] for row in collected.values() if row not in special_failures)
    detail = dict(
        status="passed" if rest_passes else "failed",
        unavoidable_sealed_generation_failures=[row["scene_id"] for row in special_failures],
        build_failures=[row["scene_id"] for row in entries if not row.get("built")],
        appearance_failures=[row["scene_id"] for row in entries if row.get("appearance_passed") is False],
        triangle_intersection_scene_count=sum(bool(row.get("built") and row.get("triangle_intersection_pairs") != 0) for row in entries),
        sealed_scene_count=len(entries), counts_by_partition=dict(Counter(row["partition"] for row in entries)),
        sealed_plan_sha256=load(PLAN_DIR/"sha256.json"), generation_code=code,
        fallback_count=sum(row.get("fallback_count", 0) for row in entries),
        scenes=entries, other_category_controls=[collected[spec["scene_id"]] for spec in single],
        synthetic_curve=collected[synthetic["scene_id"]],
    )
    write_json(output/"placement-preflight.json", detail)
    print(detail["status"], len(entries), [row["scene_id"] for row in collected.values() if not row["passed"]], flush=True)
    return detail["status"] == "passed"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(0 if run_preflight(parser.parse_args().output) else 1)
