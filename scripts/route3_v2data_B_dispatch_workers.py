"""Prepare B training geometry with three CPU workers and a single GPU lane."""
import argparse
from collections import Counter, defaultdict, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import csv
from datetime import datetime, timezone
import fcntl
from functools import partial
import os
from pathlib import Path
import shutil
import subprocess
import sys
from threading import Event
import time

import psutil

from route3_v2data_B_batch import (
    FORMAL_DATA, GIB, GPU_LOCK, RECORD, RESERVE_BYTES, ROOT, archive_status, finish_unit, prepare_geometry, produce_optical,
    reserve, sha256, write_json,
)
from route3_v2data_B_dispatch import STOP, RunMeter, execute_on, production_lock, publish_measurement
from route3_v2data_B_optical_pair import execute_serial_optical, paired_trial, produce_pairs, record_trial
from route3_v2data_B_plan import load
from route3_v2data_B_release import (
    DISPATCH_PATCH_VERSION, PLAN_DIR, code_hash, cpu_worker_dispatch_code, load_production_plan,
    sealed_plan, validate_p2,
)


CPU_WORKERS = 3
READY_LIMIT = 30
RSS_GIB = 70
MAX_THREADS = 23
PHASES = ("render", "input-priors")
DEVELOPMENT = FORMAL_DATA/"units/development"
TRAINING = FORMAL_DATA/"units/training"
PREPARED = FORMAL_DATA/"dispatch-prepared"
HANDOFF = "development finished; p2-cpu-workers-3 takeover"
DISPATCH_RECEIPT = FORMAL_DATA/"dispatch-cpu-workers-p2.json"


def affinity():
    cpus = sorted(os.sched_getaffinity(0))[:MAX_THREADS]
    if len(cpus) < 19:
        raise RuntimeError("B requires a coordinator, six geometry CPU cores and 12 available GPU-worker CPU cores")
    for thread in psutil.Process().threads():
        try:
            os.sched_setaffinity(thread.id, {cpus[0]})
        except ProcessLookupError:
            continue
    return cpus[:1], [cpus[i:i+2] for i in (1,3,5)], cpus[7:19]


def current_phase(unit, marker_dir):
    if unit is None:
        return "idle"
    for phase in reversed(PHASES):
        marker = marker_dir/f"{unit.name}-{phase}.pid"
        if marker.is_file() and not (unit/f"{phase}.resources.json").is_file():
            try:
                pid = int(marker.read_text())
                if psutil.Process(pid).is_running():
                    return phase
            except (ValueError, psutil.NoSuchProcess):
                continue
    return "waiting_or_acceptance"


class GeometryTelemetry:
    def __init__(self, path, stream, markers):
        self.path, self.markers = path, markers
        self.output = csv.writer(stream)
        self.output.writerow(("elapsed_seconds","scene_id","phase","gpu_utilization_percent","ready_scenes"))
        self.stream = stream
        self.start = time.monotonic()
        self.next_sample = self.start
        self.stage_gpu = defaultdict(list)
        self.stage_seconds = defaultdict(dict)
        self.ready_on_sample = []
        self.ready_at_gpu_dispatch = []
        self.samples = 0

    def sample(self, unit, ready_count):
        now = time.monotonic()
        if now < self.next_sample:
            return
        self.next_sample += 1
        if self.next_sample <= now:
            self.next_sample = now+1
        phase = current_phase(unit, self.markers)
        value = subprocess.run(
            ["nvidia-smi","-i","0","--query-gpu=utilization.gpu","--format=csv,noheader,nounits"],
            text=True,capture_output=True,check=True,
        ).stdout.strip()
        usage = int(value)
        self.stage_gpu[phase].append(usage)
        self.ready_on_sample.append(ready_count)
        self.output.writerow((round(now-self.start,3),unit.name if unit else "",phase,usage,ready_count))
        self.samples += 1
        if self.samples%15==0:
            self.stream.flush()

    def complete(self, unit):
        self.stage_seconds[unit.name] = {}
        for phase in PHASES:
            marker = self.markers/f"{unit.name}-{phase}.pid"
            end = unit/f"{phase}.resources.json"
            if marker.is_file() and end.is_file():
                self.stage_seconds[unit.name][phase] = (end.stat().st_mtime_ns-marker.stat().st_mtime_ns)/1e9

    def result(self, accepted):
        self.stream.flush()
        elapsed = time.monotonic()-self.start
        stages = {name:dict(samples=len(values),mean_utilization_percent=sum(values)/len(values))
                  for name,values in self.stage_gpu.items() if values}
        duration = [sum(row.values()) for row in self.stage_seconds.values() if set(row)==set(PHASES)]
        mean = sum(duration)/len(duration) if duration else None
        active = [x for phase in PHASES for x in self.stage_gpu.get(phase, [])]
        utilization = sum(active)/len(active) if active else None
        ready_nonempty = bool(self.ready_on_sample) and all(self.ready_on_sample)
        return dict(
            accepted_scenes=accepted, sampling_seconds=elapsed,sample_interval_seconds=1,
            samples=self.samples,per_stage=stages,per_scene_gpu_seconds=self.stage_seconds,
            active_gpu_mean_utilization_percent=utilization,
            accepted_per_hour=accepted*3600/elapsed if elapsed else 0,
            serial_gpu_upper_bound_per_hour=3600/mean if mean else None,
            ready_queue_always_nonempty=ready_nonempty,
            ready_queue_floor=min(self.ready_on_sample,default=0),
            ready_queue_floor_before_dispatch=min(self.ready_at_gpu_dispatch,default=0),
            path=str(self.path.relative_to(FORMAL_DATA)), sha256=sha256(self.path),
        )


def execute_cpu(cpus, abort, command, log, *, uses_gpu=False):
    command=list(command)
    if "--threads" in command:
        command[command.index("--threads")+1]=str(len(cpus))
    return execute_on(cpus,6,abort,command,log,uses_gpu=uses_gpu)


def execute_gpu(cpus, abort, markers, command, log, *, uses_gpu=False):
    if not uses_gpu:
        return execute_on(cpus,46,abort,command,log)
    marker = markers/f"{log.parent.name}-{log.stem}.pid"
    # The process marker is written inside the existing global flock.
    entry = ["sh","-c",'printf "%s\\n" "$$" > "$1"; shift; exec "$@"',"B GPU phase",str(marker),*command]
    return execute_on(cpus,46,abort,entry,log,uses_gpu=True)


def prepared_receipt(unit, version):
    return dict(generator_version=version,spec_sha256=sha256(unit/"scene-spec.json"),
                presence_sha256=sha256(unit/"labels/foreground-presence.json"))


def already_prepared(unit, spec, version):
    receipt = load(PREPARED/f"{unit.name}.json")
    if not receipt or not (unit/"scene-spec.json").is_file() or load(unit/"scene-spec.json") != spec:
        return False
    if any((unit/f"{phase}.log").exists() for phase in PHASES):
        return False
    return receipt==prepared_receipt(unit,version)


def setup_receipt(plan, state, allocation):
    release_ref = state["generator_patch"]
    original_path = FORMAL_DATA/release_ref["path"]
    release = load(original_path)
    if sha256(original_path) != release_ref["sha256"]:
        raise ValueError("the sealed generator receipt has changed")
    if state.get("dispatch_cpu_workers"):
        validate_p2(plan,state,release)
        return load(DISPATCH_RECEIPT)
    coordinator, geometry, gpu = allocation
    patch = dict(
        version=DISPATCH_PATCH_VERSION, generation_release_sha256=release_ref["sha256"],
        generator_code_sha256=release["generator_code_sha256"], base_dispatch_code=release["dispatch_code"],
        dispatch_code=cpu_worker_dispatch_code(),
        workers=CPU_WORKERS, maximum_ready_units=READY_LIMIT, max_compute_threads=MAX_THREADS,
        peak_rss_limit_bytes=RSS_GIB*GIB,
        cpu_affinity=dict(coordinator=coordinator,geometry=geometry,gpu=gpu),
        optical_trial_before_geometry_gpu=True,
    )
    patch["dispatch_code_sha256"]=code_hash(patch["dispatch_code"])
    state["dispatch_cpu_workers"]=dict(version=patch["version"],path=DISPATCH_RECEIPT.name,
                                       sha256=write_json(DISPATCH_RECEIPT,patch))
    validate_p2(plan,state,release)
    manifest_digest = write_json(FORMAL_DATA/"manifest.json",state)
    record = load(RECORD)
    record["v2"]["dispatch_cpu_workers"] = dict(patch,receipt=state["dispatch_cpu_workers"])
    record["v2"]["manifest_sha256"] = manifest_digest
    record["v2"]["p2_dispatch"].update(status="running_cpu_worker_dispatch",stop_after_current=False)
    write_json(RECORD,record)
    return patch


def exclusive_trial(estimate, allocation, version, abort):
    before_lock=time.monotonic()
    with GPU_LOCK.open("a") as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        wait_seconds=time.monotonic()-before_lock
        report = paired_trial(estimate,allocation,version,held_lock=stream,force_qualification=True,abort=abort)
    report["exclusive_gpu_lock_wait_seconds"]=wait_seconds
    return report


def gpu_resources(report, trial):
    return dict(report, paired_optical_passed_equivalence=bool(trial.get("within_cycles_noise")) and all(
        trial["within_cycles_noise"].values()
    ), paired_optical_adopted=trial.get("adopted",False))


def save_first_thirty(telemetry, accepted, trial):
    result = gpu_resources(telemetry.result(accepted), trial)
    mean=result["active_gpu_mean_utilization_percent"]
    result["eligible_to_propose_geometry_pair"] = (
        mean is not None and mean<60 and result["paired_optical_passed_equivalence"] and result["ready_queue_always_nonempty"]
    )
    record=load(RECORD)
    record["v2"]["dispatch_cpu_workers"]["training_first_30"]=result
    write_json(RECORD,record)
    return result


def dispatch_training(plan, estimate, allocation, publish_worktree=None):
    coordinator, geometry, gpu = allocation
    version=plan["effective_generator_version"]
    TRAINING.mkdir(parents=True,exist_ok=True)
    PREPARED.mkdir(exist_ok=True)
    run_dir=FORMAL_DATA/"production-runs"/f"{time.time_ns()}-cpu-workers"
    run_dir.mkdir(parents=True)
    failures_path=TRAINING/"failures.json"
    failures=load(failures_path) or []
    blocked=Counter(row["category"] for row in failures if row.get("counts_toward_pause",True))
    excluded={row["scene_id"] for row in failures}
    pending=deque()
    ready=deque()
    for spec in plan["scenes"]:
        unit=TRAINING/spec["scene_id"]
        if spec["scene_id"] in excluded or (unit/"manifest.json").is_file():
            continue
        (ready if already_prepared(unit,spec,version) else pending).append((spec,unit))

    abort=Event()
    meter=RunMeter((coordinator,[cpu for mask in geometry for cpu in mask],gpu),abort,rss_gib=RSS_GIB)
    devices=deque(geometry)
    attempted=accepted=0
    peak_ready=len(ready)
    failed_disk=False
    trial_result=None
    telemetry=None
    first_thirty=None
    start_geometry=None

    def fail(spec, error, count=True):
        failures.append(dict(scene_id=spec["scene_id"],category=spec["category"],
                             reason=str(error),counts_toward_pause=count))
        blocked[spec["category"]] += bool(count)
        write_json(failures_path,failures)

    def summarize(completed=False):
        elapsed=time.monotonic()-start_geometry
        snapshot=meter.report("training",attempted,accepted)
        snapshot.update(scene_slots=CPU_WORKERS+1,seconds=elapsed,accepted_per_hour=accepted*3600/elapsed,
                        generator_version=version,partition_dispatch_finished=completed,
                        cpu_workers=CPU_WORKERS,ready_queue_limit=READY_LIMIT,peak_ready_queue=peak_ready)
        publish_measurement(snapshot,False)
        record=load(RECORD)
        record["v2"]["dispatch_cpu_workers"].update(
            ready_queue_peak=peak_ready,peak_combined_rss_bytes=meter.peak_rss,
            sampled_peak_cpu_equivalents=meter.peak_cpu_equivalents,
        )
        write_json(RECORD,record)

    with (run_dir/"gpu-utilization-first30.csv").open("x",newline="") as samples, ThreadPoolExecutor(max_workers=4) as pool:
        prepared_jobs={}
        active=None
        trial=pool.submit(exclusive_trial,estimate,(coordinator,[],gpu),version,abort)
        while pending or ready or prepared_jobs or active or trial:
            if trial and trial.done():
                trial_result=trial.result()
                record_trial(trial_result)
                report_snapshot(publish_worktree,run_dir/"publish-optical-trial.log")
                start_geometry=time.monotonic()
                telemetry=GeometryTelemetry(run_dir/"gpu-utilization-first30.csv",samples,run_dir)
                trial=None
            if telemetry and first_thirty is None and accepted>=30 and time.monotonic()-telemetry.start>=1800:
                first_thirty=save_first_thirty(telemetry,accepted,trial_result)
                summarize()
                report_snapshot(publish_worktree,run_dir/"publish-training-first30.log")
            running_allowed=not STOP.is_file() and not abort.is_set()
            if trial is None and active is None and ready and running_allowed:
                if telemetry and first_thirty is None:
                    telemetry.ready_at_gpu_dispatch.append(len(ready))
                spec,unit=ready.popleft()
                if blocked[spec["category"]]>=2:
                    fail(spec,"two failed scenes in category",False)
                else:
                    execute=partial(execute_gpu,gpu,abort,run_dir)
                    active=(pool.submit(finish_unit,unit,spec,version,execute=execute),spec,unit)
            while devices and pending and running_allowed and not failed_disk and len(ready)+len(prepared_jobs)<READY_LIMIT:
                budget_slots=len(ready)+len(prepared_jobs)+int(active is not None)+1
                free=shutil.disk_usage(FORMAL_DATA).free
                if free < RESERVE_BYTES+estimate*budget_slots:
                    failed_disk = free<RESERVE_BYTES+estimate
                    break
                spec,unit=pending.popleft()
                if blocked[spec["category"]]>=2:
                    fail(spec,"two failed scenes in category",False)
                    continue
                attempted+=1
                if unit.exists():
                    fail(spec,"prior incomplete attempt was preserved")
                    continue
                cpu=devices.popleft()
                execute=partial(execute_cpu,cpu,abort)
                future=pool.submit(prepare_geometry,unit,spec,estimate,execute=execute)
                prepared_jobs[future]=(spec,unit,cpu)
            futures=[*prepared_jobs]
            if active:
                futures.append(active[0])
            if trial:
                futures.append(trial)
            meter.sample(labels_busy=bool(prepared_jobs),gpu_busy=bool(active or trial))
            if telemetry and first_thirty is None:
                telemetry.sample(active[2] if active else None,len(ready))
            if not futures:
                if running_allowed and ready:
                    continue
                break
            done,_=wait(futures,timeout=.4,return_when=FIRST_COMPLETED)
            for future in [task for task in prepared_jobs if task in done]:
                spec,unit,cpu=prepared_jobs.pop(future)
                devices.append(cpu)
                try:
                    future.result()
                except (OSError,RuntimeError,ValueError) as error:
                    fail(spec,error)
                else:
                    write_json(PREPARED/f"{unit.name}.json",prepared_receipt(unit,version))
                    ready.append((spec,unit))
                    peak_ready=max(peak_ready,len(ready))
            if active and active[0] in done:
                task,spec,unit=active
                try:
                    result=task.result()
                    write_json(unit/"manifest.json",result)
                    accepted+=1
                    if telemetry and first_thirty is None:
                        telemetry.complete(unit)
                except (OSError,RuntimeError,ValueError) as error:
                    fail(spec,error)
                active=None
                archive_status()
                if accepted and accepted%4==0:
                    summarize()
    archive_status()
    if start_geometry is not None:
        if telemetry and first_thirty is None and telemetry.samples:
            # Partial evidence remains available if a genuine production stop occurs.
            with (run_dir/"gpu-utilization-first30.csv").open("a") as stream:
                telemetry.stream=stream
                save_first_thirty(telemetry,accepted,trial_result)
        summarize(completed=not pending and not ready and not abort.is_set())
    if abort.is_set():
        raise RuntimeError("B exceeded the combined 70 GiB RSS cap")
    if failed_disk or (pending and not STOP.is_file()):
        raise RuntimeError("B disk reserve prevented the remaining geometry from being issued")
    return trial_result


def report_snapshot(checkout, log):
    if checkout is None:
        return
    command=[sys.executable,str(Path(__file__)),"--publish-snapshot","--publish-worktree",str(checkout)]
    with log.open("wb") as stream:
        subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)


def publish_snapshot(checkout):
    if checkout == ROOT:
        raise ValueError("publish production snapshots from the isolated worktree")
    record_path=RECORD.relative_to(ROOT)
    progress=Path("03_实验记录/当前实验进度与方法论.md")
    subprocess.run(["git","pull","--rebase","origin","main"],cwd=checkout,check=True)
    live=load(RECORD)
    v2=live["v2"]
    tracked=load(checkout/record_path)
    for key in ("p2_qualification","p2_invariance","p2_dispatch","p2_rebuild","formal_counts","manifest_sha256",
                "target_gap","failures","dispatch_cpu_workers","optical_parallel_trial","p2_optical_parallel_trial","p2_finish_eta"):
        if key in v2:
            tracked["v2"][key]=v2[key]
    write_json(checkout/record_path,tracked)
    first=v2["dispatch_cpu_workers"].get("training_first_30")
    trial=v2.get("optical_parallel_trial",{})
    note="**B v2 p2 正式派单**。开发集结束后启动三个 CPU 几何工人，GPU 几何保持单槽。"
    if "parallel_to_serial_ratio" in trial:
        mode="双场并行" if trial["adopted"] else "串行"
        note+=f"双场光学实测并行用时为串行的 {trial['parallel_to_serial_ratio']:.3f} 倍，训练光学采用{mode}。"
    if first and first["accepted_scenes"]>=30:
        note+=f"前 30 个训练几何单元入列速度为 {first['accepted_per_hour']:.2f} 场/小时。"
    note+="[B 档记录](records/route3-v2data-B-2026-11-20.json)"
    path=checkout/progress
    text=path.read_text()
    lines=text.splitlines(keepends=True)
    index=next((i for i,line in enumerate(lines) if line.startswith("**B v2 p2 正式派单**")),None)
    if index is None:
        i=next(i for i,line in enumerate(lines) if line.startswith("**B v2 p2 多层放置与资格验证**"))
        lines.insert(i+1,"\n"+note+"\n")
    else:
        lines[index]=note+"\n"
    path.write_text("".join(lines))
    for command in (
        ["git","diff","--check"],
        ["git","add","--",str(record_path),str(progress)],
        ["git","diff","--cached","--check"],
        ["git","commit","-m","记录 B v2 p2 并行光学实测与训练派单性能"],
        ["git","pull","--rebase","origin","main"],
        ["git","push","origin","HEAD:main"],
    ):
        subprocess.run(command,cwd=checkout,check=True)


def run(estimate, publish_worktree=None):
    with production_lock():
        plan,state=sealed_plan(PLAN_DIR/"training.json")
        if STOP.is_file() and STOP.read_text().strip()!=HANDOFF:
            raise RuntimeError("a different dispatch stop is still in effect")
        allocation=affinity()
        setup_receipt(plan,state,allocation)
        if STOP.is_file():
            STOP.rename(FORMAL_DATA/"p2-review"/f"cpu-workers-handoff-{time.time_ns()}")
        plan=load_production_plan(PLAN_DIR/"training.json")
        trial=dispatch_training(plan,estimate,allocation,publish_worktree)
        if STOP.is_file():
            return
        coordinator,_,gpu=allocation
        if trial.get("adopted"):
            produce_pairs(estimate,(coordinator,[],gpu),plan["effective_generator_version"])
        else:
            execute=partial(execute_on,gpu,46,Event())
            produce_optical(estimate,execute=partial(execute_serial_optical,execute))
        record=load(RECORD)
        v2=record["v2"]
        status=("drained_after_stop" if STOP.is_file() else
                "queue_exhausted_with_shortfalls" if any(v2["target_gap"].values()) else "completed")
        v2["p2_dispatch"].update(status=status,stop_after_current=STOP.is_file())
        v2["p2_finish_eta"]=dict(status=status,remaining_units=v2["target_gap"],
                                 measured_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(RECORD,record)


def completed_development():
    plan=load(PLAN_DIR/"development.json")
    failures={row["scene_id"] for row in (load(DEVELOPMENT/"failures.json") or [])}
    return [(DEVELOPMENT/spec["scene_id"],spec["scene_id"] in failures) for spec in plan["scenes"]]


def handoff(old_checkout, commit, estimate):
    if not commit:
        raise ValueError("the dispatch handoff requires a committed production entry")
    while True:
        if STOP.is_file() and STOP.read_text().strip()!=HANDOFF:
            raise RuntimeError("a different dispatch stop is active")
        # All scheduled development scenes have been handed to the old CPU/GPU
        # workers. Its stop file drains in-flight work and blocks the next partition.
        issued=completed_development()
        if not STOP.is_file() and issued and all(unit.is_dir() or failed for unit,failed in issued):
            with STOP.open("x") as stream:
                stream.write(HANDOFF+"\n")
            print("B development issued all scenes; waiting for old dispatcher to drain",flush=True)
        if STOP.is_file():
            try:
                with production_lock():
                    old_record=load(old_checkout/RECORD.relative_to(Path(__file__).resolve().parents[1]))
                    measured=old_record["v2"]["p2_dispatch"]["production_measurements"].get("development",{})
                    if not measured.get("partition_dispatch_finished") or not all(
                        (unit/"manifest.json").is_file() or failed for unit,failed in completed_development()
                    ):
                        raise ValueError("development did not finish before the dispatch handoff")
                    if any(TRAINING.glob("training-*/scene-spec.json")):
                        raise ValueError("training was already issued by the old dispatch")
                    subprocess.run(["git","merge","--ff-only",commit],cwd=old_checkout,check=True)
                os.chdir(old_checkout)
                script=old_checkout/"scripts"/Path(__file__).name
                publisher=Path(__file__).resolve().parents[1]
                os.execv(sys.executable,[sys.executable,str(script),"--measured-peak-unit-bytes",str(estimate),
                                         "--publish-worktree",str(publisher)])
            except RuntimeError as error:
                if str(error)!="another B production dispatcher owns the archive":
                    raise
        time.sleep(3)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measured-peak-unit-bytes",type=int)
    parser.add_argument("--publish-snapshot",action="store_true")
    parser.add_argument("--publish-worktree",type=Path)
    parser.add_argument("--old-checkout",type=Path)
    parser.add_argument("--handoff-commit")
    args=parser.parse_args()
    if args.publish_snapshot:
        if not args.publish_worktree:
            parser.error("publishing needs an isolated worktree")
        publish_snapshot(args.publish_worktree.resolve())
        raise SystemExit(0)
    if not args.measured_peak_unit_bytes or args.measured_peak_unit_bytes<=0:
        parser.error("the disk estimate must be positive")
    if args.old_checkout:
        handoff(args.old_checkout.resolve(),args.handoff_commit,args.measured_peak_unit_bytes)
    else:
        run(args.measured_peak_unit_bytes,args.publish_worktree)
