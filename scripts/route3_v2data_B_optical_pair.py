"""Measure and, if qualified, dispatch paired B optical references under one GPU lock."""
from collections import Counter
from contextlib import ExitStack
import fcntl
import os
import subprocess
import time

import numpy as np
import psutil

from route3_v2data_B_batch import (
    FORMAL_DATA, GIB, GPU_LOCK, RECORD, archive_status, blender, optical_selection,
    reserve, sha256, verify_optical, write_json,
)
from route3_v2data_B_plan import load
from route3_v2data_B_presence import attach_optical_visibility
from route3_v2data_B_release import OPTICAL_FRAMES, PLAN_DIR, load_production_plan


SEED = 20261120
SAMPLES = 1024
RSS_CAP = 28 * GIB
VRAM_CAP = 24 * GIB
PASS_RATIO = .85
RECEIPT = FORMAL_DATA / "p2-review/paired-optical-result.json"


def trial_policy(ids):
    return dict(
        scene_ids=ids, frames=list(OPTICAL_FRAMES), focus_count=2, apertures=[1.4, 2.],
        subviews=64, samples=SAMPLES, seed=SEED, max_ratio=PASS_RATIO,
        rss_limit_bytes=RSS_CAP, vram_limit_bytes=VRAM_CAP,
        timing="from acquired shared lock through the acceptance of both scenes, with queue time reported separately",
        noise="An exact match passes. Otherwise, independently render f030 with seed+1 at the same sample "
              "count, and require each LDR/HDR product's MAE and maximum absolute difference to stay "
              "below the corresponding peak independent-seed discrepancy for that scene and representation.",
    )


def stage_optical(source, unit):
    unit.mkdir(parents=True, exist_ok=False)
    labels = unit/"labels"
    labels.mkdir()
    for name in ("scene.blend", "scene-spec.json"):
        (unit/name).symlink_to(source/name)
    for frame in OPTICAL_FRAMES:
        name = f"f{frame:03d}-pinhole.npz"
        (labels/name).symlink_to(source/"labels"/name)


def resources():
    root = psutil.Process()
    pids, rss = set(), 0
    for process in [root, *root.children(recursive=True)]:
        try:
            rss += process.memory_info().rss
            pids.add(process.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    reply = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
        check=True, text=True, capture_output=True,
    )
    vram = Counter()
    for line in reply.stdout.splitlines():
        pid, _, mib = line.partition(",")
        if pid.strip().isdigit() and int(pid) in pids:
            if not mib.strip().isdigit():
                raise RuntimeError("could not measure combined VRAM for the B optical workers")
            vram[int(pid)] += int(mib) * 2**20
    return rss, sum(vram.values())


def stop_children(processes):
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def render_group(units, allocation, *, parallel, frames=OPTICAL_FRAMES, seed=SEED):
    workers = [*allocation[1], *allocation[2]]
    half = min(6, len(workers)//2)
    if half < 1:
        raise RuntimeError("paired optical qualification requires two disjoint worker CPU sets")
    affinity = (workers[:half], workers[half:2*half])
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="0", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    peak_rss = peak_vram = 0
    result = {unit.name:dict(passed=False) for unit in units}
    before_lock = time.monotonic()
    with GPU_LOCK.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        started = time.monotonic()
        groups = [units] if parallel else [[unit] for unit in units]
        for group in groups:
            processes = []
            with ExitStack() as streams:
                try:
                    for index, unit in enumerate(group):
                        cpus = affinity[index]
                        command = blender(
                            "route3_v2data_B_render.py", "--scene", unit, "--optical-only", "--frames", *frames,
                            "--seed", seed, "--reference-samples", SAMPLES, threads=len(cpus),
                        )
                        stream = streams.enter_context((unit/"render.log").open("wb"))
                        process = subprocess.Popen(
                            ["taskset", "-c", ",".join(map(str, cpus)), *command],
                            stdout=stream, stderr=subprocess.STDOUT, env=environment,
                        )
                        processes.append(process)
                    while any(process.poll() is None for process in processes):
                        rss, vram = resources()
                        peak_rss, peak_vram = max(peak_rss, rss), max(peak_vram, vram)
                        if rss > RSS_CAP or vram > VRAM_CAP:
                            raise RuntimeError("paired optical render exceeded its combined RSS or VRAM cap")
                        time.sleep(1)
                except (OSError, RuntimeError, psutil.Error, subprocess.SubprocessError) as error:
                    for unit in group:
                        result[unit.name]["failure"] = str(error)
                    stop_children(processes)
                    continue
                for unit, process in zip(group, processes):
                    row = result[unit.name]
                    row["exit_code"] = process.returncode
                    if process.returncode:
                        row["failure"] = f"optical render returned {process.returncode}"
                        continue
                    try:
                        labels = unit/"labels"
                        verify_optical(labels, load(labels/"optical-audit.json")["optical"], frames)
                    except (OSError, ValueError, KeyError) as error:
                        row["failure"] = str(error)
                    else:
                        row["passed"] = True
        elapsed = time.monotonic() - started
    return dict(
        seconds=elapsed, gpu_queue_seconds=started-before_lock, peak_combined_rss_bytes=peak_rss,
        peak_combined_vram_bytes=peak_vram, scenes=result, worker_cpu_counts=[len(mask) for mask in affinity],
    )


def pixel_difference(first, second):
    comparison = {}
    for path in sorted((second/"labels").glob("*.npz")):
        kind = path.stem.rsplit("-", 1)[-1]
        if kind not in {"optics", "hdr"}:
            continue
        with np.load(first/"labels"/path.name) as left, np.load(path) as right:
            delta = np.abs(left["rgb_linear"].astype(np.float32)-right["rgb_linear"].astype(np.float32))
        comparison[path.name] = dict(kind=kind, mae=float(delta.mean(dtype=np.float64)),
                                     maximum_absolute_difference=float(delta.max()))
    return comparison


def within_noise(measured, calibration):
    metrics = ("mae", "maximum_absolute_difference")
    maxima = {(kind, key):max(row[key] for row in calibration.values() if row["kind"] == kind)
              for kind in ("optics", "hdr") for key in metrics}
    return all(row[key] <= maxima[row["kind"], key] for row in measured.values() for key in metrics)


def paired_trial(estimate, allocation, version):
    config = load(RECORD)["v2"]["p2_optical_parallel_policy"]
    if config != trial_policy(config["scene_ids"]) or len(set(config["scene_ids"])) != 2:
        raise ValueError("the paired optical experiment differs from its preregistered policy")
    existing = load(RECEIPT)
    if existing.get("generator_version") == version and existing.get("policy") == config:
        return existing
    ids = config["scene_ids"]
    path = RECEIPT.parent / f"paired-optical-{time.time_ns()}"
    report = dict(generator_version=version, policy=config, adopted=False,
                  products=str(path.relative_to(FORMAL_DATA)))
    sources = {key:FORMAL_DATA/"units/training"/key for key in ids}
    release = load(FORMAL_DATA/"generator-patch-p2.json")
    proof_path = FORMAL_DATA/release["qualification"]["path"]
    proof = load(proof_path)
    source_origin = {}
    for scene_id, source in sources.items():
        if load(source/"manifest.json").get("generator_version") != version:
            fallback_path = proof_path
            previous = proof.get("unchanged_category_proof")
            if previous and scene_id in previous["unchanged_smoke_ids"]:
                fallback_path = FORMAL_DATA/previous["path"]
            source = fallback_path.parent/"smoke"/scene_id
            manifest = source/"manifest.json"
            if manifest.is_file() and sha256(manifest) == proof["smoke"][scene_id]["manifest_sha256"]:
                sources[scene_id] = source
                source_origin[scene_id] = "preregistered isolated training qualification"
        else:
            source_origin[scene_id] = "formally admitted training unit"
    report["training_scene_sources"] = source_origin
    if any(load(source/"manifest.json").get("generator_version") != version for source in sources.values()):
        report["blocking_reason"] = "one or both preregistered training scenes have no verified p2 manifest"
        return report
    reserve(FORMAL_DATA, estimate * 6)
    path.mkdir(parents=True)
    report["geometry_manifest_sha256"] = {key:sha256(source/"manifest.json") for key, source in sources.items()}
    for mode in ("serial", "parallel"):
        units = [path/mode/key for key in ids]
        for unit in units:
            stage_optical(sources[unit.name], unit)
        report[mode] = render_group(units, allocation, parallel=mode == "parallel")
        write_json(path/"progress.json", report)
    serial, parallel = report["serial"], report["parallel"]
    passed = all(row["passed"] for mode in (serial, parallel) for row in mode["scenes"].values())
    report["parallel_to_serial_ratio"] = parallel["seconds"] / serial["seconds"]
    if passed:
        differences = {key:pixel_difference(path/"serial"/key, path/"parallel"/key) for key in ids}
        report["pixel_differences"] = differences
        noise_passes = {}
        for key, differences_for_scene in differences.items():
            noise_passes[key] = all(row["maximum_absolute_difference"] == 0 for row in differences_for_scene.values())
        if not all(noise_passes.values()):
            units = [path/"independent-seed"/key for key in ids if not noise_passes[key]]
            for unit in units:
                stage_optical(sources[unit.name], unit)
            noise = render_group(units, allocation, parallel=False, frames=(30,), seed=SEED+1)
            report["independent_noise_calibration"] = noise
            for unit in units:
                if noise["scenes"][unit.name]["passed"]:
                    calibration = pixel_difference(path/"serial"/unit.name, unit)
                    noise["scenes"][unit.name]["pixel_differences"] = calibration
                    noise_passes[unit.name] = within_noise(differences[unit.name], calibration)
        report["within_cycles_noise"] = noise_passes
        report["adopted"] = report["parallel_to_serial_ratio"] <= PASS_RATIO and all(noise_passes.values()) and (
            0 < parallel["peak_combined_rss_bytes"] <= RSS_CAP
            and 0 < parallel["peak_combined_vram_bytes"] <= VRAM_CAP
        )
    write_json(RECEIPT, report)
    return report


def record_trial(report):
    value = {key:value for key, value in report.items()
             if key not in ("pixel_differences", "independent_noise_calibration")}
    if RECEIPT.is_file():
        value.update(path=str(RECEIPT.relative_to(FORMAL_DATA)), sha256=sha256(RECEIPT))
    if "pixel_differences" in report:
        value["worst_pixel_difference"] = {
            scene: {metric:max(row[metric] for row in rows.values())
                    for metric in ("mae", "maximum_absolute_difference")}
            for scene, rows in report["pixel_differences"].items()
        }
    record = load(RECORD)
    from route3_v2data_B_dispatch import update_p2_eta
    v2 = record["v2"]
    v2["p2_optical_parallel_trial"] = value
    if v2["p2_dispatch"].get("production_measurements"):
        update_p2_eta(v2)
    write_json(RECORD, record)


def produce_pairs(estimate, allocation, version):
    target = FORMAL_DATA / "optical"
    target.mkdir(exist_ok=True)
    selection = target/"selection.json"
    if not selection.exists():
        admitted = [load(path) for path in sorted((FORMAL_DATA/"units/training").glob("*/manifest.json"))]
        write_json(selection, optical_selection(admitted))
    selection_sha = sha256(selection)
    failures_path = target/"failures.json"
    failures = load(failures_path) or []
    excluded = {row["scene_id"] for row in failures}
    counts = Counter(row["category"] for row in failures if row.get("counts_toward_pause", True))
    pending = iter(load(selection))

    def failure(scene_id, category, reason, count=True):
        failures.append(dict(scene_id=scene_id, category=category, reason=reason, counts_toward_pause=count))
        counts[category] += count
        write_json(failures_path, failures)

    while not (FORMAL_DATA/"stop-after-current").is_file():
        group = []
        reserve(FORMAL_DATA, estimate * 2)
        for item in pending:
            scene_id = item["scene_id"]
            unit = target/scene_id
            category = item["primary_difficulty"]
            if scene_id in excluded or (unit/"manifest.json").is_file():
                continue
            if counts[category] >= 2:
                failure(scene_id, category, "two failed optical scenes for this difficulty", False)
                continue
            source = FORMAL_DATA/"units/training"/scene_id
            try:
                stage_optical(source, unit)
            except OSError as error:
                failure(scene_id, category, str(error))
                continue
            group.append((item, unit, source))
            if len(group) == 2:
                break
        if not group:
            break
        run = render_group([unit for _, unit, _ in group], allocation, parallel=True)
        for item, unit, source in group:
            row = run["scenes"][unit.name]
            if not row["passed"]:
                failure(unit.name, item["primary_difficulty"], row["failure"])
                continue
            labels = unit/"labels"
            attach_optical_visibility(labels/"optical-audit.json", source/"labels/foreground-presence.json")
            products = sorted(path for path in labels.glob("*.npz") if not path.is_symlink())
            outputs = [dict(path=str(path.relative_to(FORMAL_DATA)), bytes=path.stat().st_size, sha256=sha256(path))
                       for path in products]
            write_json(unit/"manifest.json", dict(
                **item, generator_version=version, optical_frames=list(OPTICAL_FRAMES), partition="training",
                geometry_manifest_sha256=sha256(source/"manifest.json"), selection_sha256=selection_sha,
                bytes=sum(row["bytes"] for row in outputs), files=outputs,
            ))
        archive_status()
    archive_status()


def try_p2_pairs(estimate, allocation):
    version = load_production_plan(PLAN_DIR/"training.json")["effective_generator_version"]
    if not version.endswith("-p2"):
        return False
    report = paired_trial(estimate, allocation, version)
    record_trial(report)
    if report["adopted"]:
        produce_pairs(estimate, allocation, version)
    return report["adopted"]



def execute_serial_optical(execute, command, log, *, uses_gpu=False):
    usage = execute(command, log, uses_gpu=uses_gpu)
    if "--optical-only" in command:
        unit = log.parent
        origin = FORMAL_DATA/"units/training"/unit.name/"labels/foreground-presence.json"
        attach_optical_visibility(unit/"labels/optical-audit.json", origin)
    return usage
