"""Prepare B training geometry with a bounded CPU queue and qualified GPU slots."""
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
from route3_v2data_B_geometry_pair import PAIR_CPUS, finish_group
from route3_v2data_B_geometry_slots import (
    EXTRA_CPU, NOISE_RESULT, PAIR_HANDOFF, QUEUE_MIN, SURVEY_UNITS, Survey, install_gpu_dispatch,
    active_geometry_fallback, apply_geometry_requeue, optical_gate, preserve_for_retry, process_gpu_samples,
    record_geometry_requeue_outcome, record_geometry_slots, rejected_geometry_radiance, update_slot_eta,
)
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


def dispatch_training(plan, estimate, allocation, publish_worktree=None, *, gpu_patch=None, extra_cores=()):
    coordinator, geometry, gpu = allocation
    version=plan["effective_generator_version"]
    TRAINING.mkdir(parents=True,exist_ok=True)
    PREPARED.mkdir(exist_ok=True)
    run_dir=FORMAL_DATA/"production-runs"/f"{time.time_ns()}-cpu-workers"
    run_dir.mkdir(parents=True)
    failures_path=TRAINING/"failures.json"
    failures=load(failures_path) or []
    category_pause = not (gpu_patch and gpu_patch.get("geometry_requeue"))
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
    live = load(RECORD)["v2"]
    switch = active_geometry_fallback(live)
    recovery = live.get("geometry_pair_fallback_recovery")
    pair_enabled = bool(gpu_patch and gpu_patch["adopted"] and not switch)
    reported_survey = live.get("geometry_pair_first_30")
    keep_measuring = bool(pair_enabled and recovery and reported_survey and reported_survey["accepted"]<SURVEY_UNITS)
    survey = Survey(reported_survey if keep_measuring else None)
    survey_done = bool(reported_survey and not keep_measuring)
    if live.get("geometry_pair_extra_worker") and extra_cores:
        survey.added_cpu = live["geometry_pair_extra_worker"]

    def enable_extra_worker():
        devices.append(extra_cores)
        meter.allocation = (coordinator, [*meter.allocation[1],*extra_cores], gpu)

    if survey.added_cpu:
        enable_extra_worker()

    def save_pair_record(**values):
        record = load(RECORD)
        record["v2"].update(values)
        write_json(RECORD,record)

    def complete_pair_survey():
        nonlocal survey_done
        if survey.started is not None and not survey_done:
            survey_done = True
            fallback = switch or live.get("geometry_pair_fallback")
            save_pair_record(geometry_pair_first_30=survey.report(meter, fallback=fallback, recovery=recovery))
            summarize()
            report_snapshot(publish_worktree,run_dir/"publish-geometry-pair-first30.log")
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
                        cpu_workers=CPU_WORKERS+bool(survey.added_cpu),ready_queue_limit=READY_LIMIT,
                        peak_ready_queue=peak_ready,gpu_geometry_slots=2 if pair_enabled else 1)
        publish_measurement(snapshot,False)
        record=load(RECORD)
        record["v2"]["dispatch_cpu_workers"].update(
            ready_queue_peak=peak_ready,peak_combined_rss_bytes=meter.peak_rss,
            sampled_peak_cpu_equivalents=meter.peak_cpu_equivalents,
        )
        if gpu_patch:
            update_slot_eta(record["v2"],failures)
        write_json(RECORD,record)

    with (run_dir/"gpu-utilization-first30.csv").open("x",newline="") as samples, ThreadPoolExecutor(max_workers=6) as pool:
        prepared_jobs={}
        active=None
        if gpu_patch:
            trial_result = load(FORMAL_DATA/"p2-review/paired-optical-result.json")
            start_geometry = time.monotonic()
            trial = None
        else:
            trial=pool.submit(exclusive_trial,estimate,(coordinator,[],gpu),version,abort)
        retries = set()
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
            if pair_enabled:
                if not survey_done:
                    survey.ready(len(ready))
                if len(ready)<QUEUE_MIN and not survey.added_cpu and extra_cores and pending and running_allowed:
                    enable_extra_worker()
                    survey.added_cpu = dict(cpu_ids=extra_cores,ready_at_activation=len(ready),
                                            activated_at_utc=datetime.now(timezone.utc).isoformat())
                    save_pair_record(geometry_pair_extra_worker=survey.added_cpu)
            if trial is None and active is None and ready and running_allowed:
                if telemetry and first_thirty is None:
                    telemetry.ready_at_gpu_dispatch.append(len(ready))
                width=2 if pair_enabled else 1
                group=[]
                while len(group)<width and ready:
                    spec,unit=ready.popleft()
                    if category_pause and blocked[spec["category"]]>=2 and unit.name not in retries:
                        fail(spec,"two failed scenes in category",False)
                    else:
                        group.append((spec,unit))
                if group:
                    if pair_enabled:
                        if not survey_done:
                            survey.ready(len(ready))
                        trace = run_dir/f"{group[0][1].name}-pmon.txt" if not survey_done and len(group)==2 else None
                        if trace and survey.started is None:
                            survey.started=time.monotonic()
                        units=[unit for _,unit in group]
                        future=pool.submit(finish_group,units,version,PAIR_CPUS[:len(units)],abort=abort,
                                           monitor=partial(process_gpu_samples,trace) if trace else None)
                        active=(future,group,trace)
                    else:
                        spec,unit=group[0]
                        execute=partial(execute_gpu,gpu,abort,run_dir)
                        active=(pool.submit(finish_unit,unit,spec,version,execute=execute),group,None)
            while devices and pending and running_allowed and not failed_disk and len(ready)+len(prepared_jobs)<READY_LIMIT:
                budget_slots=len(ready)+len(prepared_jobs)+(len(active[1]) if active else 0)+1
                free=shutil.disk_usage(FORMAL_DATA).free
                if free < RESERVE_BYTES+estimate*budget_slots:
                    failed_disk = free<RESERVE_BYTES+estimate
                    break
                spec,unit=pending.popleft()
                if category_pause and blocked[spec["category"]]>=2:
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
                telemetry.sample(active[1][0][1] if active else None,len(ready))
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
                task,group,trace=active
                report=None
                try:
                    if pair_enabled:
                        results,report=task.result()
                        errors=report["failures"]
                    else:
                        results={group[0][1].name:task.result()}
                        errors={}
                except (OSError,RuntimeError,ValueError,KeyError,subprocess.SubprocessError) as error:
                    results,errors={},dict.fromkeys((unit.name for _,unit in group),str(error))
                dual = pair_enabled
                runtime_errors = dict(errors)
                if dual and not category_pause:
                    for _,unit in group:
                        if unit.name in errors and rejected_geometry_radiance(unit,errors[unit.name]):
                            del runtime_errors[unit.name]
                if dual and runtime_errors:
                    pair_enabled=False
                    switch=dict(at_utc=datetime.now(timezone.utc).isoformat(), failures=runtime_errors, archived_attempts={})
                for spec,unit in group:
                    if unit.name in results:
                        result = results[unit.name]
                        if gpu_patch:
                            result["geometry_gpu_slots"] = "dual" if dual and len(group)==2 else "single"
                        write_json(unit/"manifest.json",result)
                        accepted+=1
                        if telemetry and first_thirty is None:
                            telemetry.complete(unit)
                    elif dual and unit.name in runtime_errors:
                        switch["archived_attempts"][unit.name]=preserve_for_retry(unit,run_dir)
                        write_json(PREPARED/f"{unit.name}.json",prepared_receipt(unit,version))
                        retries.add(unit.name)
                        ready.appendleft((spec,unit))
                    else:
                        fail(spec,errors[unit.name])
                if dual and runtime_errors:
                    save_pair_record(geometry_pair_fallback=switch)
                    print("B geometry GPU slots fell back to one slot",runtime_errors,flush=True)
                if dual and trace and report:
                    survey.track(report,trace,len(ready),len(results))
                active=None
                units=archive_status()
                if gpu_patch and results:
                    record_geometry_slots(units)
                if accepted and (dual or accepted%4==0):
                    summarize()
                if dual and (not pair_enabled or survey.accepted>=SURVEY_UNITS):
                    complete_pair_survey()
    units=archive_status()
    if gpu_patch:
        record_geometry_slots(units)
        if gpu_patch.get("geometry_requeue"):
            record_geometry_requeue_outcome(units,failures)
        complete_pair_survey()
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
                "target_gap","failures","dispatch_cpu_workers","optical_parallel_trial","p2_optical_parallel_trial",
                "p2_finish_eta","dispatch_gpu_slots","geometry_parallel_trial","geometry_pair_first_30",
                "geometry_pair_extra_worker","geometry_pair_fallback","optical_slot_decision","optical_geometry_gate",
                "geometry_pair_noise_policy","geometry_pair_noise_trial","depth_prior_repeatability",
                "target_gap_details","geometry_pair_planner_override","geometry_gpu_slot_postcheck",
                "geometry_gpu_slot_provenance","geometry_pair_fallback_recovery","large_foreground_requeue",
                "training_optical_selection"):
        if key in v2:
            tracked["v2"][key]=v2[key]
    write_json(checkout/record_path,tracked)
    first=v2["dispatch_cpu_workers"].get("training_first_30")
    trial=v2.get("optical_parallel_trial",{})
    pair = v2.get("dispatch_gpu_slots", {})
    mode = "双槽" if pair.get("adopted") and not active_geometry_fallback(v2) else "单槽"
    note=f"**B v2 p2 正式派单**。开发集结束后启动三个 CPU 几何工人，GPU 几何使用{mode}。"
    if v2.get("geometry_pair_noise_trial"):
        state = "通过" if v2["geometry_pair_noise_trial"].get("adopted") else "未通过"
        note+=f"按场景补测的深度一致性判据{state}。"
    if pair.get("planner_override"):
        note+="经用户同意，采用规划方改判。"
    if v2.get("geometry_pair_fallback"):
        note+="双槽运行曾因场景验收失败退回单槽。"
        if not active_geometry_fallback(v2):
            note+="单槽重放复现后排除该场景，并恢复双槽。"
    rerun=v2.get("large_foreground_requeue",{})
    if rerun.get("status")=="drained":
        note+=f"大面积前景补跑入列 {len(rerun['admitted'])}/22 场。"
        if v2["optical_geometry_gate"].get("allowed"):
            note+="按各类别至少 90% 覆盖且真实失败至多 10 场的门槛进入训练光学。"
    if "geometry_pair_first_30" in v2:
        measured = v2["geometry_pair_first_30"]
        note+=f"几何双槽实测 {measured['accepted']} 场，入列速度为 {measured['accepted_per_hour']:.2f} 场/小时。"
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
        ["git","commit","-m","记录 B v2 p2 训练派单实测",
         "-m","Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"],
        ["git","pull","--rebase","origin","main"],
        ["git","push","origin","HEAD:main"],
    ):
        subprocess.run(command,cwd=checkout,check=True)


def try_publish_snapshot(checkout):
    try:
        publish_snapshot(checkout)
    except subprocess.CalledProcessError as error:
        print("B production snapshot could not be pushed",error,flush=True)


def run(estimate, publish_worktree=None, *, gpu_dispatch=False):
    with production_lock():
        plan,state=sealed_plan(PLAN_DIR/"training.json")
        gpu_dispatch = gpu_dispatch or bool(state.get("dispatch_gpu_slots"))
        expected_stop = PAIR_HANDOFF if gpu_dispatch else HANDOFF
        if STOP.is_file() and STOP.read_text().strip()!=expected_stop:
            raise RuntimeError("a different dispatch stop is still in effect")
        available = os.sched_getaffinity(0)
        extra = EXTRA_CPU if set(EXTRA_CPU).issubset(available) else ()
        allocation=affinity()
        gpu_patch = None
        if gpu_dispatch:
            if tuple(allocation[2]) != PAIR_CPUS[0]+PAIR_CPUS[1] or not extra:
                raise RuntimeError("the geometry and optical slot CPU layout is unavailable")
            gpu_patch = install_gpu_dispatch(plan,state)
        else:
            setup_receipt(plan,state,allocation)
        if STOP.is_file():
            STOP.rename(FORMAL_DATA/"p2-review"/f"dispatch-handoff-{time.time_ns()}")
        plan=load_production_plan(PLAN_DIR/"training.json")
        if gpu_patch:
            if gpu_patch.get("geometry_requeue"):
                apply_geometry_requeue(gpu_patch,TRAINING/"failures.json")
                archive_status()
            record=load(RECORD)
            update_slot_eta(record["v2"],load(TRAINING/"failures.json") or [])
            write_json(RECORD,record)
            print("B geometry dispatcher",gpu_patch["dispatch_code_sha256"],
                  "dual" if gpu_patch["adopted"] else "single",
                  gpu_patch["switched_at_utc"],flush=True)
        trial=dispatch_training(plan,estimate,allocation,publish_worktree,gpu_patch=gpu_patch,extra_cores=extra)
        if STOP.is_file():
            return
        coordinator,_,gpu=allocation
        if gpu_patch:
            archive=load(FORMAL_DATA/"manifest.json")
            failed=load(TRAINING/"failures.json") or []
            decision=optical_gate(plan["scenes"],archive["units"],failed)
            record=load(RECORD)
            v2=record["v2"]
            decision["status"]="starting_optical_selection" if decision["allowed"] else "blocked"
            decision["checked_at_utc"]=datetime.now(timezone.utc).isoformat()
            v2["optical_geometry_gate"]=decision
            v2["p2_dispatch"].update(status="starting_training_optical" if decision["allowed"]
                                     else "training_optical_blocked_by_geometry_gate", stop_after_current=False)
            if not decision["allowed"]:
                v2["p2_finish_eta"].update(status="training_optical_blocked_by_geometry_gate")
            write_json(RECORD,record)
            if not decision["allowed"]:
                if publish_worktree:
                    try_publish_snapshot(publish_worktree)
                print("B training optical gate blocked",decision,flush=True)
                return
            print("B training optical gate passed",decision,flush=True)
        if trial.get("adopted"):
            def progress():
                record=load(RECORD)
                update_slot_eta(record["v2"],load(TRAINING/"failures.json") or [])
                write_json(RECORD,record)
                if publish_worktree:
                    try_publish_snapshot(publish_worktree)

            def selection_ready(candidates, path):
                record=load(RECORD)
                record["v2"]["training_optical_selection"]=dict(
                    count=len(candidates), path=str(path.relative_to(FORMAL_DATA)), sha256=sha256(path),
                    primary_difficulty=dict(Counter(item["primary_difficulty"] for item in candidates)),
                    difficulty_labels=dict(Counter(label for item in candidates for label in item["difficulties"])),
                )
                record["v2"]["p2_dispatch"]["status"]="dispatching_training_optical"
                write_json(RECORD,record)
                print("B training optical selection sealed",len(candidates),flush=True)

            produce_pairs(estimate,allocation,plan["effective_generator_version"],survey_slots=bool(gpu_patch),
                          on_policy_decision=progress if gpu_patch else None,
                          on_selection=selection_ready if gpu_patch else None)
        elif gpu_patch:
            raise RuntimeError("the adopted two-slot optical qualification is required")
        else:
            execute=partial(execute_on,gpu,46,Event())
            produce_optical(estimate,execute=partial(execute_serial_optical,execute))
        record=load(RECORD)
        v2=record["v2"]
        gap = v2["target_gap"]
        accepted_gaps = v2.get("target_gap_details", {}).get("train_geometry", {}).get("accepted", [])
        remaining = dict(gap,train_geometry=max(0,gap["train_geometry"]-len(accepted_gaps)))
        status=("drained_after_stop" if STOP.is_file() else
                "queue_exhausted_with_shortfalls" if any(remaining.values()) else
                "completed_with_accepted_geometry_gap" if accepted_gaps else "completed")
        v2["p2_dispatch"].update(status=status,stop_after_current=STOP.is_file())
        v2["p2_finish_eta"]=dict(status=status,remaining_units=v2["target_gap"],
                                 measured_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(RECORD,record)
        if gpu_patch and publish_worktree:
            try_publish_snapshot(publish_worktree)


def handoff_geometry(old_checkout, commit, estimate, publisher):
    if not commit:
        raise ValueError("the GPU handoff requires a committed production entry")
    training = load(PLAN_DIR/"training.json")["scenes"]
    version = load(FORMAL_DATA/"generator-patch-p2.json")["patch_version"]
    if not load(NOISE_RESULT):
        raise ValueError("the geometry noise qualification report is required before takeover")
    old_head = subprocess.check_output(["git","rev-parse","HEAD"],cwd=old_checkout,text=True).strip()
    subprocess.run(["git","merge-base","--is-ancestor",old_head,commit],cwd=old_checkout,check=True)
    while True:
        if STOP.is_file() and STOP.read_text().strip()!=PAIR_HANDOFF:
            raise RuntimeError("a different dispatch stop is active")
        try:
            with production_lock():
                subprocess.run(["git","merge","--ff-only",commit],cwd=old_checkout,check=True)
            os.chdir(old_checkout)
            script=old_checkout/"scripts"/Path(__file__).name
            os.execv(sys.executable,[sys.executable,str(script),"--gpu-dispatch",
                                     "--measured-peak-unit-bytes",str(estimate),
                                     "--publish-worktree",str(publisher)])
        except RuntimeError as error:
            if str(error)!="another B production dispatcher owns the archive":
                raise
        if not STOP.is_file():
            excluded={row["scene_id"] for row in load(TRAINING/"failures.json") or []}
            ready=sum(already_prepared(TRAINING/spec["scene_id"],spec,version) for spec in training
                      if spec["scene_id"] not in excluded)
            if ready>=READY_LIMIT:
                with STOP.open("x") as stream:
                    stream.write(PAIR_HANDOFF+"\n")
                print(f"B geometry takeover requested with {ready} prepared scenes",flush=True)
        time.sleep(2)


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
    parser.add_argument("--gpu-dispatch",action="store_true")
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
        if args.gpu_dispatch:
            if not args.publish_worktree:
                parser.error("a GPU handoff needs an isolated snapshot worktree")
            handoff_geometry(args.old_checkout.resolve(),args.handoff_commit,args.measured_peak_unit_bytes,
                             args.publish_worktree.resolve())
        else:
            handoff(args.old_checkout.resolve(),args.handoff_commit,args.measured_peak_unit_bytes)
    else:
        run(args.measured_peak_unit_bytes,args.publish_worktree,gpu_dispatch=args.gpu_dispatch)
