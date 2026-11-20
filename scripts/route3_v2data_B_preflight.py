"""Check and report the world-box placement of every sealed multilayer B scene."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import os
from pathlib import Path
from threading import Event
import time

from route3_v2data_B_batch import blender
from route3_v2data_B_dispatch import RunMeter, execute_on, production_lock
from route3_v2data_B_front import RECORD, write_json
from route3_v2data_B_inputs import FORMAL_DATA
from route3_v2data_B_placement import policy
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


def run_preflight(output):
    archive = load(RECORD)["v2"]
    if archive["multilayer_placement_policy_p2"] != policy(TRAIN_FRAMES):
        raise ValueError("the preregistered policy does not match the source")
    code = p2_generation_code()
    if archive["generator_patch_p2"]["generator_code_sha256"] != code_hash(code):
        raise ValueError("preflight source hash changed after preregistration")
    rows, synthetic = planned_scenes()
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
        scenes = [*rows, synthetic]
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
    entries = [collected[spec["scene_id"]] for spec in rows]
    detail = dict(
        status="passed" if all(row["passed"] for row in collected.values()) else "failed",
        sealed_scene_count=len(entries), counts_by_partition=dict(Counter(row["partition"] for row in entries)),
        sealed_plan_sha256=load(PLAN_DIR/"sha256.json"), generation_code=code,
        fallback_count=sum(row.get("fallback_count", 0) for row in entries),
        scenes=entries, synthetic_curve=collected[synthetic["scene_id"]],
    )
    write_json(output/"placement-preflight.json", detail)
    print(detail["status"], len(entries), [row["scene_id"] for row in collected.values() if not row["passed"]], flush=True)
    return detail["status"] == "passed"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(0 if run_preflight(parser.parse_args().output) else 1)
