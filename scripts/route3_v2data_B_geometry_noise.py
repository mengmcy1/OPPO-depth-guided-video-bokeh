"""Measure scene-specific repeatability of the frozen B geometry input priors."""
from datetime import datetime, timezone
from functools import partial
from itertools import combinations
import time

import numpy as np

from route3_ovg_subpixel import TRAIN_FRAMES
from route3_v2data_B_batch import GIB, RECORD, finish_unit, reserve, sha256, write_json
from route3_v2data_B_geometry_pair import (
    FORMAL_DATA, REVIEW, SCENES, SINGLE_CPU, TRAINING, TRIAL, GeometryGPU, gpu_lease, stage_geometry,
)
from route3_v2data_B_plan import load
from route3_v2data_B_release import PLAN_DIR, code_hash, load_production_plan, p2_generation_code


RESULT = REVIEW/"geometry-pair-noise-result.json"
METRICS = ("mae", "maximum_absolute_difference")
DEPTH = ("depth_sharp", "depth_ds", "depth_tri")
FIELDS = (*DEPTH, "confidence", "aligned_sharp")
POLICY = dict(
    factor=1.25, scene_ids=list(SCENES), frames=list(TRAIN_FRAMES),
    serial_repeats_per_scene=2, additional_serial_order=[SCENES[1], SCENES[1], SCENES[0]],
    comparison="original admission, serial 1 and serial 2, all three unordered pairs, per-scene maxima",
    absolute_metrics={field:list(METRICS) for field in FIELDS},
    relative_metric="maximum over 13 frames of the pixel median of 2*abs(a-b)/(abs(a)+abs(b)) for aligned_sharp, both zero maps to zero",
    zero_bound="a zero serial maximum requires an exact match for that metric",
    parallel_source="reuse original geometry pair qualification outputs",
    calibration_holds_global_gpu_lock=True,
)


def checked_field(a, b, field, frame):
    x, y = a[field], b[field]
    if x.shape!=y.shape or x.dtype!=np.float16 or y.dtype!=np.float16:
        raise ValueError(f"mismatched stored priors for {frame} {field}")
    expected = (512,512) if field=="aligned_sharp" else (3,512,512)
    if x.shape!=expected or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError(f"invalid stored priors for {frame} {field}")
    return x.astype(np.float32), y.astype(np.float32)


def comparison(first, second):
    sums = dict.fromkeys(FIELDS,0.)
    maxima = dict.fromkeys(FIELDS,0.)
    counts = dict.fromkeys(FIELDS,0)
    medians = {}
    for frame in TRAIN_FRAMES:
        name = f"observations/priors/f{frame:03d}.npz"
        with np.load(first/name) as a, np.load(second/name) as b:
            x,y = checked_field(a,b,"depth",frame)
            delta = dict(zip(DEPTH,np.abs(x-y),strict=True))
            x,y = checked_field(a,b,"confidence",frame)
            delta["confidence"] = np.abs(x-y)
            x,y = checked_field(a,b,"aligned_sharp",frame)
            delta["aligned_sharp"] = np.abs(x-y)
            scale = np.abs(x)+np.abs(y)
            relative = np.divide(2*delta["aligned_sharp"],scale,out=np.zeros_like(x),where=scale!=0)
            medians[f"f{frame:03d}"] = float(np.median(relative))
            for field,values in delta.items():
                sums[field] += float(values.sum(dtype=np.float64))
                maxima[field] = max(maxima[field],float(values.max()))
                counts[field] += values.size
    metrics = {field:dict(mae=sums[field]/counts[field], maximum_absolute_difference=maxima[field])
               for field in FIELDS}
    metrics["aligned_sharp"]["max_frame_median_relative_difference"] = max(medians.values())
    return dict(metrics=metrics, aligned_sharp_median_relative_by_frame=medians)


def compare_scene(admitted, repeats, parallel):
    copies = dict(admitted=admitted, serial_1=repeats[0], serial_2=repeats[1])
    pairs = {f"{a}__{b}":comparison(copies[a],copies[b]) for a,b in combinations(copies,2)}
    measured = comparison(admitted,parallel)
    envelope = {
        field:{metric:max(pair["metrics"][field][metric] for pair in pairs.values()) for metric in values}
        for field,values in measured["metrics"].items()
    }
    decisions = {
        field:{metric:value<=POLICY["factor"]*envelope[field][metric] for metric,value in values.items()}
        for field,values in measured["metrics"].items()
    }
    return dict(serial_pairs=pairs, serial_maxima=envelope, parallel_vs_admitted=measured,
                metric_passed=decisions, passed=all(all(row.values()) for row in decisions.values()))


def main():
    preregistration = load(RECORD)["v2"]["geometry_pair_noise_policy"]
    if preregistration["policy"]!=POLICY:
        raise ValueError("the noise policy differs from the pre-registered B record")
    original = load(TRIAL)
    plan = load_production_plan(PLAN_DIR/"training.json")
    version = plan["effective_generator_version"]
    if (original["generator_version"]!=version or
            original["generator_code_sha256"]!=code_hash(p2_generation_code())):
        raise ValueError("the frozen generator differs from the original parallel run")
    original_dir = FORMAL_DATA/original["directory"]
    reserve(FORMAL_DATA,3*GIB)
    trial_dir = REVIEW/f"geometry-pair-noise-{time.time_ns()}"
    destinations = []
    for index,scene in zip((1,2,2),POLICY["additional_serial_order"],strict=True):
        unit = TRAINING/scene
        if sha256(unit/"manifest.json")!=original["scenes"][scene]["manifest_sha256"]:
            raise ValueError(f"{scene} has changed since the parallel qualification")
        target = trial_dir/f"serial-{index}"/scene
        stage_geometry(unit,target)
        destinations.append(target)

    reference = dict(version=version, generator_code_sha256=original["generator_code_sha256"],
                     directory=str(trial_dir.relative_to(FORMAL_DATA)), policy=preregistration,
                     original_parallel_report_sha256=sha256(TRIAL))
    repeats = {}
    queued = time.monotonic()
    with gpu_lease():
        started = time.monotonic()
        for target in destinations:
            meter = GeometryGPU()
            error = None
            try:
                finish_unit(target,load(target/"scene-spec.json"),version,execute=partial(meter.execute,SINGLE_CPU))
            except (KeyError,OSError,RuntimeError,ValueError) as exception:
                error = str(exception)
            repeats[str(target.relative_to(FORMAL_DATA))] = dict(**meter.report(), error=error)
            write_json(trial_dir/"serial-progress.json",repeats)
            print(target.parent.name,target.name,"finished",error or "",flush=True)
    reference.update(serial_repeats=repeats, gpu_lock_seconds=time.monotonic()-started,
                     gpu_lock_wait_seconds=started-queued, finished_at=datetime.now(timezone.utc).isoformat())
    if any(row["error"] for row in repeats.values()):
        write_json(RESULT,dict(reference,adopted=False,calibration_complete=False))
        return
    results = {}
    for scene in SCENES:
        first = (original_dir/"serial" if scene==original["serial_calibration"] else trial_dir/"serial-1")/scene
        second = trial_dir/"serial-2"/scene
        if sha256(TRAINING/scene/"manifest.json")!=original["scenes"][scene]["manifest_sha256"]:
            raise ValueError(f"{scene} was modified during calibration")
        results[scene] = compare_scene(TRAINING/scene,(first,second),original_dir/"parallel"/scene)
    adopted = all(row["passed"] for row in results.values()) and original["speed_passed"] and original["resources_passed"]
    final = dict(reference, calibration_complete=True, scenes=results, adopted=adopted,
                 original_speed_passed=original["speed_passed"], original_resources_passed=original["resources_passed"])
    write_json(trial_dir/"result.json",final)
    write_json(RESULT,final)
    print("adopted",adopted,flush=True)


if __name__=="__main__":
    main()
