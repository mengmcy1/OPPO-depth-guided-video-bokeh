"""One manifest writer for the two-slot B v2 CPU and GPU production queue."""
import argparse
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from functools import partial
import fcntl
import os
from pathlib import Path
from threading import Event
import time

import psutil

from route3_v2data_B_batch import (
    FORMAL_DATA, GIB, GPU_LOCK, RECORD, archive_status, finish_unit, prepare_geometry,
    produce_optical, reserve, run, sha256, write_json,
)
from route3_v2data_B_plan import load
from route3_v2data_B_release import PLAN_DIR, install_optical_patch, load_production_plan


PARTITIONS = ("test_unseen_assets","test_unseen_occlusions","development","training")
STOP = FORMAL_DATA/"stop-after-current"


@contextmanager
def production_lock():
    with (FORMAL_DATA/"production-dispatch.lock").open("a") as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another B production dispatcher owns the archive") from None
        yield


def assigned_cpus():
    allowed = sorted(os.sched_getaffinity(0))[:14]
    if len(allowed)<3:
        raise RuntimeError("two disjoint worker slots require a CPU for each slot and the dispatcher")
    # The controller and its monitoring commands share one CPU. Every
    # subprocess then binds to its own stage's disjoint budget.
    for thread in psutil.Process().threads():
        try:
            os.sched_setaffinity(thread.id,{allowed[0]})
        except ProcessLookupError:
            continue
    split=1+min(5,(len(allowed)-1)//2)
    return allowed[:1],allowed[1:split],allowed[split:]


def execute_on(cpus, rss_gib, abort, command, log, *, uses_gpu=False):
    if abort.is_set():
        raise RuntimeError("B dispatch reached the combined resource limit")
    affinity = ",".join(map(str,cpus))
    return run(["taskset","-c",affinity,*command],log,uses_gpu=uses_gpu,
               cpu_limit_bytes=rss_gib*GIB,abort=abort)


class RunMeter:
    def __init__(self, allocation, abort):
        self.process = psutil.Process()
        self.allocation = allocation
        self.abort = abort
        self.start = self.last = time.monotonic()
        self.previous_cpu = {}
        self.previous_wait = None
        self.previous_overlap = False
        self.overlap_seconds = 0.
        self.peak_rss = 0
        self.peak_cpu_equivalents = 0.
        self.wait_seconds = Counter()
        self.gpu_started = False

    def sample(self, *, labels_busy=False, gpu_busy=False):
        now = time.monotonic()
        interval = now-self.last
        if self.previous_wait:
            self.wait_seconds[self.previous_wait] += interval
        if self.previous_overlap:
            self.overlap_seconds += interval
        self.previous_overlap = labels_busy and gpu_busy
        self.last = now
        self.gpu_started |= gpu_busy
        self.previous_wait = None
        if labels_busy and not gpu_busy:
            stage = "steady" if self.gpu_started else "startup"
            with GPU_LOCK.open("a") as lock:
                try:
                    fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    self.previous_wait = stage+"_shared_lock_busy"
                else:
                    self.previous_wait = stage+"_gpu_free"
                    fcntl.flock(lock,fcntl.LOCK_UN)
        rss = 0
        current_cpu = {}
        for process in [self.process,*self.process.children(recursive=True)]:
            try:
                with process.oneshot():
                    rss += process.memory_info().rss
                    cpu = process.cpu_times()
                    current_cpu[process.pid] = cpu.user+cpu.system
            except (psutil.NoSuchProcess,psutil.AccessDenied):
                continue
        self.peak_rss = max(self.peak_rss,rss)
        if interval>=.1:
            occupied = sum(max(0,value-self.previous_cpu.get(pid,value)) for pid,value in current_cpu.items())/interval
            self.peak_cpu_equivalents = max(self.peak_cpu_equivalents,occupied)
        self.previous_cpu = current_cpu
        if rss>28*GIB:
            self.abort.set()

    def report(self, mode, attempts, accepted):
        self.sample()
        elapsed = time.monotonic()-self.start
        coordinator,cpu,gpu = self.allocation
        return dict(partition=mode,scene_slots=2,attempted_scenes=attempts,accepted_scenes=accepted,
                    seconds=elapsed,accepted_per_hour=accepted*3600/elapsed,
                    cpu_affinity={"dispatcher":coordinator,"geometry":cpu,"gpu":gpu},
                    max_compute_threads=sum(map(len,self.allocation)),
                    sampled_peak_cpu_equivalents=self.peak_cpu_equivalents,
                    peak_combined_rss_bytes=self.peak_rss,
                    label_and_gpu_lane_overlap_seconds=self.overlap_seconds,
                    gpu_idle_for_b_labels_seconds={
                        phase:self.wait_seconds[phase+"_gpu_free"] for phase in ("startup","steady")
                    },
                    shared_lock_busy_while_b_labels_pending_seconds={
                        phase:self.wait_seconds[phase+"_shared_lock_busy"] for phase in ("startup","steady")
                    },
                    combined_memory_limit_hit=self.abort.is_set())


def publish_measurement(report, benchmark):
    path = FORMAL_DATA/"production-runs"/f"{time.time_ns()}-{report['partition']}.json"
    path.parent.mkdir(exist_ok=True)
    value = dict(report,path=str(path.relative_to(FORMAL_DATA)),sha256=write_json(path,report))
    record = load(RECORD)
    key = "two_scene_benchmark" if benchmark else "last_two_scene_production_run"
    record["v2"][key] = value
    write_json(RECORD,record)
    print("B dispatch",report["partition"],report["accepted_scenes"],"accepted",
          f"{report['accepted_per_hour']:.2f}/hour",
          "GPU idle for B labels",report["gpu_idle_for_b_labels_seconds"],flush=True)
    return report


def run_partition(path, estimate, limit, allocation):
    plan = load_production_plan(path)
    mode = path.stem
    units = FORMAL_DATA/"units"/mode
    units.mkdir(parents=True,exist_ok=True)
    failure_path = units/"failures.json"
    failures = load(failure_path) or []
    blocked = Counter(row["category"] for row in failures if row.get("counts_toward_pause",True))
    excluded = {row["scene_id"] for row in failures}
    pending = deque(spec for spec in plan["scenes"] if spec["scene_id"] not in excluded
                    and not (units/spec["scene_id"]/"manifest.json").is_file())
    prepared = deque()
    abort = Event()
    meter = RunMeter(allocation,abort)
    cpu_exec = partial(execute_on,allocation[1],8,abort)
    gpu_exec = partial(execute_on,allocation[2],18,abort)
    attempted = accepted = 0
    disk_failure = None

    def failure(spec, reason, *, count=True):
        failures.append(dict(scene_id=spec["scene_id"],category=spec["category"],
                             reason=reason,counts_toward_pause=count))
        if count:
            blocked[spec["category"]] += 1
        write_json(failure_path,failures)

    with ThreadPoolExecutor(max_workers=2) as jobs:
        cpu = gpu = None
        while pending or prepared or cpu or gpu:
            if gpu is None and prepared:
                spec,unit = prepared.popleft()
                if abort.is_set() or blocked[spec["category"]]>=2:
                    failure(spec,"B resource limit or two failed scenes in category",count=False)
                else:
                    gpu=(jobs.submit(finish_unit,unit,spec,plan["effective_generator_version"],execute=gpu_exec),spec,unit)
            if (cpu is None and len(prepared)+int(gpu is not None)<2 and pending
                    and not STOP.is_file() and not abort.is_set()
                    and disk_failure is None and (limit is None or attempted<limit)):
                spec=pending.popleft()
                unit=units/spec["scene_id"]
                if blocked[spec["category"]]>=2:
                    failure(spec,"two failed scenes in category",count=False)
                    continue
                attempted += 1
                if unit.exists():
                    failure(spec,"prior incomplete attempt was preserved")
                    continue
                try:
                    reserve(units,estimate*(1+len(prepared)+int(gpu is not None)))
                except RuntimeError as exc:
                    disk_failure=exc
                else:
                    cpu=(jobs.submit(prepare_geometry,unit,spec,estimate,execute=cpu_exec),spec,unit)
            running=[job[0] for job in (cpu,gpu) if job]
            if not running:
                if prepared:
                    continue
                if pending and not STOP.is_file() and not abort.is_set() and disk_failure is None and (
                    limit is None or attempted<limit
                ):
                    continue
                break
            meter.sample(labels_busy=cpu is not None and (cpu[2]/"geometry.log").is_file(),
                         gpu_busy=gpu is not None)
            done,_=wait(running,timeout=.5,return_when=FIRST_COMPLETED)
            for stage in ("gpu","cpu"):
                job=gpu if stage=="gpu" else cpu
                if job is None or job[0] not in done:
                    continue
                task,spec,unit=job
                try:
                    result=task.result()
                    if stage=="cpu":
                        prepared.append((spec,unit))
                    else:
                        write_json(unit/"manifest.json",result)
                        accepted+=1
                except (RuntimeError,ValueError,OSError) as exc:
                    disk="disk reserve" in str(exc).lower() or "insufficient disk space" in str(exc).lower()
                    failure(spec,str(exc),count=not disk)
                    if disk:
                        disk_failure=exc
                if stage=="gpu":
                    gpu=None
                    archive_status()
                else:
                    cpu=None
    write_json(failure_path,failures)
    archive_status()
    report=publish_measurement(meter.report(mode,attempted,accepted),benchmark=limit is not None)
    if abort.is_set():
        raise RuntimeError("B production paused after hitting its combined resident memory limit")
    if disk_failure:
        raise disk_failure
    return report


def dispatch(plans, estimate, max_new_scenes=None, *, optics=False):
    with production_lock():
        allocation = assigned_cpus()
        reports=[]
        for path in plans:
            if STOP.is_file():
                return reports
            reports.append(run_partition(path,estimate,max_new_scenes,allocation))
        if optics and not STOP.is_file():
            abort=Event()
            execute=partial(execute_on,allocation[2],18,abort)
            produce_optical(estimate,execute=execute)
        return reports


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--sealed-plan",type=Path)
    mode.add_argument("--all",action="store_true")
    mode.add_argument("--optical",action="store_true")
    mode.add_argument("--summarize",action="store_true")
    mode.add_argument("--install-optical-frame-patch",action="store_true")
    parser.add_argument("--measured-peak-unit-bytes",type=int)
    parser.add_argument("--max-new-scenes",type=int)
    args=parser.parse_args()
    if args.summarize or args.install_optical_frame_patch:
        with production_lock():
            if args.summarize:
                archive_status()
            else:
                print(install_optical_patch(),flush=True)
        return
    budget=args.measured_peak_unit_bytes
    if not budget or budget<=0:
        parser.error("a measured positive unit size from the pilot is required")
    if args.max_new_scenes is not None and (not args.sealed_plan or args.max_new_scenes<=0):
        parser.error("--max-new-scenes requires --sealed-plan and a positive limit")
    paths=([PLAN_DIR/(p+".json") for p in PARTITIONS] if args.all else [args.sealed_plan] if args.sealed_plan else [])
    dispatch(paths,budget,args.max_new_scenes,optics=args.all or args.optical)


if __name__=="__main__":
    main()
