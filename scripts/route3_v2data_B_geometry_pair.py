"""Isolated B geometry GPU qualification against admitted training artifacts."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
from functools import partial
import os
from pathlib import Path
import shutil
import subprocess
from threading import Lock
import time

import numpy as np

from route3_v2data_B_batch import (
    FORMAL_DATA, GIB, GPU_LOCK, finish_unit, reserve, sha256, write_json,
)
from route3_v2data_B_optical_pair import resources, stop_children
from route3_v2data_B_plan import load
from route3_v2data_B_release import PLAN_DIR, code_hash, load_production_plan, p2_generation_code


TRAINING = FORMAL_DATA/"units/training"
REVIEW = FORMAL_DATA/"p2-review"
TRIAL = REVIEW/"geometry-pair-result.json"
SCENES = ("training-001-7a9b2de81e9aa875", "training-061-ff0a4ff42915123f")
PHASES = ("render", "input-priors")
SINGLE_CPU = tuple(range(7, 19))
PAIR_CPUS = (SINGLE_CPU[:6], SINGLE_CPU[6:])
GPU_CAP = 28 * GIB
RSS_CAP = 70 * GIB
MAX_RATIO = .85
PRIORS = ("depth", "confidence", "aligned_sharp")


def stage_geometry(source, target):
    """Carry only products that the frozen CPU stage prepares."""
    target.mkdir(parents=True, exist_ok=False)
    for folder in ("observations", "labels"):
        (target/folder).mkdir()
    for filename in ("scene.blend", "scene-spec.json", "scene-build.json", "build.resources.json",
                     "geometry.resources.json", "observations/cameras.json"):
        path = source/filename
        if path.is_file():
            os.link(path, target/filename)
    for path in (source/"labels").iterdir():
        if path.suffix == ".json" and path.name != "radiance-audit.json":
            shutil.copy2(path, target/"labels"/path.name)
        elif path.name == "scene-geometry.npz" or path.name.endswith(
                ("-pinhole.npz", "-peeled.npz", "-K16.npz")):
            os.link(path, target/"labels"/path.name)


@contextmanager
def gpu_lease():
    with GPU_LOCK.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


class GeometryGPU:
    """Track both slots under the caller's one GPU lease."""

    def __init__(self):
        self.guard = Lock()
        self.active = {}
        self.phases = {}
        self.peak_rss = self.peak_vram = 0
        self.error = None

    def execute(self, cpus, command, log, *, uses_gpu=False):
        if not uses_gpu or log.stem not in PHASES:
            raise ValueError(f"unexpected GPU geometry stage {log.name}")
        unit = log.parent.name
        phase = dict(start_monotonic=time.monotonic(), cpu_ids=list(cpus))
        command = ["taskset", "-c", ",".join(map(str, cpus)), *command]
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES="0", OPENBLAS_NUM_THREADS="1",
                           OMP_NUM_THREADS="1")
        with log.open("xb") as stream:
            with self.guard:
                if self.error:
                    raise RuntimeError(self.error)
                job = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=environment)
                phase["pid"] = job.pid
                self.active[job.pid] = job
                self.phases.setdefault(unit, {})[log.stem] = phase
            try:
                while job.poll() is None:
                    with self.guard:
                        try:
                            rss, vram = resources(self.active.values())
                        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
                            self.error = f"GPU resource measurement failed: {error}"
                        else:
                            self.peak_rss = max(self.peak_rss, rss)
                            self.peak_vram = max(self.peak_vram, vram)
                            if rss > RSS_CAP or vram > GPU_CAP:
                                self.error = "geometry exceeded the combined RSS or VRAM cap"
                        if self.error:
                            stop_children(self.active.values())
                            break
                    time.sleep(.5)
            finally:
                with self.guard:
                    del self.active[job.pid]
            phase["end_monotonic"] = time.monotonic()
            phase["seconds"] = phase["end_monotonic"] - phase["start_monotonic"]
            phase["exit_code"] = job.returncode
            write_json(log.with_suffix(".resources.json"), phase)
        if self.error or job.returncode:
            raise RuntimeError(self.error or f"geometry {log.stem} returned {job.returncode}")

    def report(self):
        stages = [row for scene in self.phases.values() for row in scene.values()]
        seconds = max(row["end_monotonic"] for row in stages) - min(
            row["start_monotonic"] for row in stages) if stages else 0
        return dict(seconds=seconds, phases=self.phases, peak_combined_rss_bytes=self.peak_rss,
                    peak_combined_vram_bytes=self.peak_vram, resource_error=self.error)


def rerun(units, version, affinity):
    meter = GeometryGPU()
    errors = {}
    before_lock = time.monotonic()
    with gpu_lease(), ThreadPoolExecutor(max_workers=len(units)) as pool:
        wait = time.monotonic() - before_lock
        jobs = {
            pool.submit(finish_unit, unit, load(unit/"scene-spec.json"), version,
                        execute=partial(meter.execute, cpus)): unit
            for unit, cpus in zip(units, affinity, strict=True)
        }
        for task, unit in jobs.items():
            try:
                task.result()
            except (OSError, RuntimeError, ValueError, KeyError) as error:
                errors[unit.name] = str(error)
    return dict(**meter.report(), gpu_queue_seconds=wait, failures=errors)


def ordinal(array):
    bits = array.view(np.uint16).astype(np.int32)
    return np.where(bits >= 32768, 32768-bits, bits)


def array_differences(reference, repeat, files, fields, *, half_ulp=False):
    totals = {name: dict(samples=0, changed=0, sum=0., max=0., max_half_ulp=0) for name in fields}
    for filename in files:
        with np.load(reference/filename) as original, np.load(repeat/filename) as candidate:
            for name, stats in totals.items():
                a, b = original[name], candidate[name]
                if a.shape != b.shape or a.dtype != b.dtype or not np.isfinite(a).all() or not np.isfinite(b).all():
                    raise ValueError(f"invalid comparison inputs: {filename} {name}")
                delta = np.abs(a.astype(np.float32)-b.astype(np.float32))
                stats["samples"] += a.size
                stats["changed"] += np.count_nonzero(delta)
                stats["sum"] += float(delta.sum(dtype=np.float64))
                stats["max"] = max(stats["max"], float(delta.max()))
                if half_ulp:
                    if a.dtype != np.float16:
                        raise ValueError(f"{filename} {name} is not stored as float16")
                    stats["max_half_ulp"] = max(stats["max_half_ulp"], int(np.abs(ordinal(a)-ordinal(b)).max()))
    return {
        name: dict(samples=stats["samples"], changed=int(stats["changed"]),
                   mae=stats["sum"]/stats["samples"], maximum_absolute_difference=stats["max"],
                   **({"max_half_ulp":stats["max_half_ulp"]} if half_ulp else {}))
        for name, stats in totals.items()
    }


def products(reference, repeat):
    def files(folder, count):
        baseline = sorted(p.name for p in (reference/folder).glob("f*.npz"))
        if len(baseline) != count or baseline != sorted(p.name for p in (repeat/folder).glob("f*.npz")):
            raise ValueError(f"mismatched evaluation files in {folder}")
        return baseline
    return dict(
        observations=array_differences(reference/"observations", repeat/"observations",
                                      files("observations", 60), ("rgb_linear",), half_ulp=True)["rgb_linear"],
        priors=array_differences(reference/"observations/priors", repeat/"observations/priors",
                                 files("observations/priors", 13), PRIORS),
    )


def original_timing(unit):
    durations = {}
    for phase in PHASES:
        measurements = unit/f"{phase}.resources.json"
        markers = list((FORMAL_DATA/"production-runs").glob(f"*/{unit.name}-{phase}.pid"))
        if len(markers) != 1:
            raise ValueError(f"expected one original GPU phase marker for {unit.name} {phase}")
        durations[phase] = dict(seconds=(measurements.stat().st_mtime_ns-markers[0].stat().st_mtime_ns)/1e9,
                                measurement=str(measurements.relative_to(FORMAL_DATA)),
                                marker=str(markers[0].relative_to(FORMAL_DATA)))
    return durations


def trial():
    plan = load_production_plan(PLAN_DIR/"training.json")
    generator = plan["effective_generator_version"]
    sources = [TRAINING/name for name in SCENES]
    original = {}
    stamp = time.time_ns()
    work = REVIEW/f"geometry-pair-{stamp}"
    pairs = [work/"parallel"/source.name for source in sources]
    serial = work/"serial"/sources[0].name
    reserve(FORMAL_DATA, 3*GIB)
    for source, target in zip(sources, pairs, strict=True):
        manifest = source/"manifest.json"
        row = load(manifest)
        if row["generator_version"] != generator or row["partition"] != "training":
            raise ValueError(f"{source.name} is not an admitted unit of the current version")
        original[source.name] = dict(category=row["category"], manifest_sha256=sha256(manifest),
                                     phases=original_timing(source))
        stage_geometry(source, target)
    if len({row["category"] for row in original.values()}) != 2:
        raise ValueError("take the benchmark scenes from distinct categories")
    stage_geometry(sources[0], serial)
    receipt = dict(scenes=original, serial_calibration=serial.name, directory=str(work.relative_to(FORMAL_DATA)),
                   generator_version=generator, generator_code_sha256=code_hash(p2_generation_code()),
                   limits=dict(speed_ratio=MAX_RATIO, rss_bytes=RSS_CAP, vram_bytes=GPU_CAP, observation_half_ulp=1))
    parallel = rerun(pairs, generator, PAIR_CPUS)
    receipt["parallel"] = parallel
    write_json(work/"parallel-result.json", parallel)
    calibration = rerun([serial], generator, [SINGLE_CPU])
    receipt["serial"] = calibration
    original_seconds = sum(phase["seconds"] for row in original.values() for phase in row["phases"].values())
    ratio = parallel["seconds"] / original_seconds
    receipt.update(original_gpu_phase_sum_seconds=original_seconds, parallel_to_original_ratio=ratio)
    equivalence = False
    if not parallel["failures"] and not calibration["failures"]:
        for source in sources:
            if sha256(source/"manifest.json") != original[source.name]["manifest_sha256"]:
                raise RuntimeError(f"admitted manifest changed: {source.name}")
        differences = {source.name:products(source, repeat) for source, repeat in zip(sources, pairs, strict=True)}
        noise = products(sources[0], serial)
        equivalence = all(
            row["observations"]["max_half_ulp"] <= 1 and
            all(stats[key] <= noise["priors"][field][key] for field, stats in row["priors"].items()
                for key in ("mae", "maximum_absolute_difference"))
            for row in differences.values()
        )
        receipt["differences"] = dict(parallel_against_admitted=differences, serial_against_admitted=noise)
    valid_resources = all(group["peak_combined_vram_bytes"] <= GPU_CAP and
                          group["peak_combined_rss_bytes"] <= RSS_CAP and
                          not group["resource_error"] and not group["failures"]
                          for group in (parallel, calibration))
    receipt.update(equivalence_passed=equivalence, speed_passed=ratio <= MAX_RATIO,
                   resources_passed=valid_resources, adopted=equivalence and ratio <= MAX_RATIO and valid_resources,
                   recorded_at=datetime.now(timezone.utc).isoformat())
    write_json(work/"result.json", receipt)
    write_json(TRIAL, receipt)
    return receipt


def guard_pending_optics():
    """Let the legacy dispatcher finish geometry, then hold optics if a sealed unit failed."""
    from route3_v2data_B_release import sealed_plan
    plan, _ = sealed_plan(PLAN_DIR/"training.json")
    stop = FORMAL_DATA/"stop-after-current"
    specs = plan["scenes"]
    failures_path = TRAINING/"failures.json"
    evidence = REVIEW/"optical-geometry-gate.json"
    while True:
        failures = load(failures_path) or []
        failed = {row["scene_id"] for row in failures}
        blocked = Counter(row["category"] for row in failures if row.get("counts_toward_pause", True))
        outstanding = [
            spec for spec in specs if spec["scene_id"] not in failed and blocked[spec["category"]]<2
            and not (TRAINING/spec["scene_id"]/"manifest.json").is_file()
        ]
        if not failures and not outstanding:
            return
        if failures and len(outstanding)<=1 and not stop.exists():
            active = outstanding[0]["scene_id"] if outstanding else None
            if active is None or (TRAINING/active/"render.log").exists():
                try:
                    with stop.open("x") as stream:
                        stream.write("training optics requires admission of all sealed geometry scenes\n")
                except FileExistsError:
                    continue
                write_json(evidence,dict(required=len(specs), remaining_active_geometry=active, failures=failures,
                                         requested_at_utc=datetime.now(timezone.utc).isoformat()))
                return
        time.sleep(1 if len(outstanding)<5 else 20)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guard-pending-optics", action="store_true")
    args = parser.parse_args()
    if args.guard_pending_optics:
        guard_pending_optics()
    else:
        print(trial()["adopted"], flush=True)
