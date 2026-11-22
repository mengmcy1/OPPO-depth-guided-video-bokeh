"""B dispatch GPU-slot lineage, failover and first-30 measurements."""
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import subprocess
import time

from route3_v2data_B_batch import FORMAL_DATA, RECORD, sha256, write_json
from route3_v2data_B_geometry_pair import GPU_CAP, PAIR_CPUS, REVIEW, TRIAL, stage_geometry
from route3_v2data_B_plan import load
from route3_v2data_B_production_policy import POLICY, ProductionStop
from route3_v2data_B_release import (
    GEOMETRY_REQUEUE, GPU_DISPATCH_RECEIPT, GPU_DISPATCH_VERSION, PLANNER_OVERRIDE,
    approved_geometry_override, code_hash, gpu_dispatch_code, previous_gpu_dispatch, validate_p2,
)


PAIR_HANDOFF = "training GPU slots and optical gate dispatch takeover"
NOISE_RESULT = REVIEW/"geometry-pair-noise-result.json"
POSTCHECK_PREREGISTRATION = REVIEW/"geometry-gpu-slot-postcheck-preregistration.json"
SURVEY_UNITS = 30
EXTRA_CPU = (19, 20, 21, 22)
QUEUE_MIN = 5


def install_gpu_dispatch(plan, state):
    release = state["generator_patch"]
    release_path = FORMAL_DATA/release["path"]
    if sha256(release_path) != release["sha256"]:
        raise ValueError("the sealed generator receipt has changed")
    generator = load(release_path)
    result = load(NOISE_RESULT)
    if (not result or result["version"] != generator["patch_version"] or
            result["generator_code_sha256"] != generator["generator_code_sha256"] or
            result["original_parallel_report_sha256"] != sha256(TRIAL)):
        raise ValueError("GPU dispatcher does not match the tested generator and qualification decision")
    previous_ref = state.get("dispatch_gpu_slots")
    if previous_ref and previous_ref["version"]==GPU_DISPATCH_VERSION:
        validate_p2(plan, state, generator)
        return load(GPU_DISPATCH_RECEIPT)
    if not state.get("dispatch_cpu_workers") or not previous_ref:
        raise ValueError("the preceding geometry dispatcher receipts are required")
    qualification = dict(path=str(NOISE_RESULT.relative_to(FORMAL_DATA)),sha256=sha256(NOISE_RESULT))
    previous = previous_gpu_dispatch(previous_ref,state,generator)
    if not previous or previous["qualification"]!=qualification:
        raise ValueError("the preceding GPU dispatch receipt or its qualification changed")
    approval = approved_geometry_override(qualification,result)
    approval_ref = dict(path=str(PLANNER_OVERRIDE.relative_to(FORMAL_DATA)),
                        sha256=sha256(PLANNER_OVERRIDE)) if approval else None
    record = load(RECORD)
    requeue = load(GEOMETRY_REQUEUE)
    if not (requeue and requeue["authority"]=="规划方决定，经用户同意"
            and requeue["decision"]=="resume_large_foreground_with_two_gpu_slots"
            and requeue["original_fallback"]==record["v2"].get("geometry_pair_fallback")
            and (result["adopted"] or approval)):
        raise ValueError("the approved geometry requeue does not match the live single-slot fallback")
    requeue_ref = dict(path=str(GEOMETRY_REQUEUE.relative_to(FORMAL_DATA)),sha256=sha256(GEOMETRY_REQUEUE))
    code = gpu_dispatch_code()
    patch = dict(version=GPU_DISPATCH_VERSION, parent=state["dispatch_cpu_workers"],
                 generation_release_sha256=release["sha256"], generator_code_sha256=result["generator_code_sha256"],
                 dispatch_code=code, dispatch_code_sha256=code_hash(code),
                 qualification=qualification, qualification_adopted=result["adopted"], planner_override=approval_ref,
                 previous_gpu_dispatch=previous_ref, fallback_recovery=previous.get("fallback_recovery"),
                 geometry_requeue=requeue_ref, adopted=True,
                 production_policy=POLICY, maximum_vram_bytes=GPU_CAP,
                 gpu_slot_cpus=PAIR_CPUS,
                 extra_cpu_worker_cores=EXTRA_CPU, extra_worker_ready_queue_threshold=QUEUE_MIN,
                 switched_at_utc=datetime.now(timezone.utc).isoformat())
    state["dispatch_gpu_slots"] = dict(version=patch["version"], path=GPU_DISPATCH_RECEIPT.name,
                                     sha256=write_json(GPU_DISPATCH_RECEIPT, patch))
    validate_p2(plan, state, generator)
    manifest = write_json(FORMAL_DATA/"manifest.json", state)
    record["v2"].update(
        dispatch_gpu_slots=patch, geometry_pair_noise_policy=result["policy"],
        geometry_pair_noise_trial=dict(result, receipt=patch["qualification"]), manifest_sha256=manifest,
        optical_geometry_gate=dict(status="waiting_for_geometry_queue"),
        production_policy=POLICY,
        depth_prior_repeatability=dict(
            statement="B 输入深度先验的串行复跑结果存在数值波动。",
            per_scene_serial_maxima={scene:rows["serial_maxima"] for scene,rows in result.get("scenes", {}).items()},
        ),
    )
    if approval:
        record["v2"]["geometry_pair_planner_override"] = dict(approval,receipt=approval_ref)
    postcheck = load(POSTCHECK_PREREGISTRATION)
    if postcheck:
        postcheck["production_effect"] = POLICY["geometry_slot_postcheck_zh"]
        postcheck["result_handling"]["any_failure"] = "记录失败场景及超限值，不阻塞生产，不自动重算。"
        record["v2"]["geometry_gpu_slot_postcheck"] = postcheck
    provenance = record["v2"].setdefault("geometry_gpu_slot_provenance", dict(
        backfilled_at_utc=patch["switched_at_utc"], per_scene={},
    ))
    for unit in state["units"]:
        if unit["partition"]=="training":
            provenance["per_scene"].setdefault(unit["scene_id"],"single")
    record["v2"]["p2_dispatch"].update(status="running_geometry_gpu_slots", stop_after_current=False)
    write_json(RECORD, record)
    return patch


def apply_geometry_requeue(patch, failures_path):
    request = load(GEOMETRY_REQUEUE)
    ref = patch["geometry_requeue"]
    if sha256(GEOMETRY_REQUEUE)!=ref["sha256"]:
        raise ValueError("the geometry requeue receipt changed")
    record = load(RECORD)
    v2 = record["v2"]
    original = request["preserved_original_failures"]
    skipped = request["original_skips"]
    live_failures = load(failures_path)
    if any(row not in live_failures for row in original):
        raise ValueError("the four original geometry failures must remain unchanged")
    if any(row in live_failures for row in skipped):
        if sha256(failures_path)!=request["original_failures_sha256"]:
            raise ValueError("the failure registry differs from the approved requeue")
        history = [dict(row, status="类别暂停导致跳过，已重新派发") for row in skipped]
        v2["large_foreground_requeue"] = dict(
            request, receipt=ref, status="redispatched",
            started_at_utc=patch["switched_at_utc"], history=history,
        )
        v2.setdefault("target_gap_details", {}).setdefault("train_geometry", {})["accepted"] = request["accepted_geometry_gaps"]
        write_json(RECORD,record)
        write_json(failures_path,[row for row in live_failures if row not in skipped])
    elif v2.get("large_foreground_requeue", {}).get("receipt")!=ref:
        raise ValueError("the archived category skips do not have a matching production record")


def record_geometry_requeue_outcome(units, failures):
    record = load(RECORD)
    rerun = record["v2"]["large_foreground_requeue"]
    scope = {row["scene_id"] for row in rerun["original_skips"]}
    admitted = {row["scene_id"]:row["geometry_gpu_slots"] for row in units if row["scene_id"] in scope}
    failed = [row for row in failures if row["scene_id"] in scope and row.get("counts_toward_pause", True)]
    pending = sorted(scope-admitted.keys()-{row["scene_id"] for row in failed})
    rerun.update(status="pending" if pending else "drained", admitted=admitted, new_failures=failed, pending=pending,
                 completed_at_utc=datetime.now(timezone.utc).isoformat() if not pending else None)
    write_json(RECORD,record)


def record_geometry_slots(units):
    record = load(RECORD)
    by_scene = record["v2"]["geometry_gpu_slot_provenance"]["per_scene"]
    for unit in units:
        if unit["partition"]=="training" and "geometry_gpu_slots" in unit:
            by_scene[unit["scene_id"]] = unit["geometry_gpu_slots"]
    write_json(RECORD,record)


def geometry_outcomes(specs, units, failures):
    planned = {spec["scene_id"]:spec["category"] for spec in specs}
    admitted = {unit["scene_id"] for unit in units if unit["partition"]=="training"}
    counted = {row["scene_id"] for row in failures if row.get("counts_toward_pause", True)}
    plan_counts = Counter(planned.values())
    admitted_counts = Counter(planned[scene] for scene in admitted & planned.keys())
    failed_counts = Counter(planned[scene] for scene in counted & planned.keys())
    if planned.keys()!=admitted|counted or admitted&counted:
        raise ProductionStop("admitted_data_corruption","geometry outcomes are missing or contradictory after drain")
    return dict(
        status="drained", planned=len(planned), admitted=len(admitted), observed_failures=len(counted),
        admitted_by_category=dict(admitted_counts), planned_by_category=dict(plan_counts),
        failures_by_category=dict(failed_counts), failed_scene_ids=sorted(counted),
    )


def active_geometry_fallback(v2):
    fallback = v2.get("geometry_pair_fallback")
    recovery = v2.get("geometry_pair_fallback_recovery", {})
    requeue = v2.get("large_foreground_requeue", {})
    resolved = (recovery.get("resumed_at_utc") and fallback and fallback["at_utc"]==recovery["fallback_at_utc"]
                or requeue.get("started_at_utc") and requeue["original_fallback"]==fallback)
    return fallback if not resolved else None


def update_slot_eta(v2, training_failures):
    counts = v2["target_gap"]
    terminal_failures = len({row["scene_id"] for row in training_failures})
    geometry_remaining = max(0,counts["train_geometry"]-terminal_failures)
    dual = v2["dispatch_gpu_slots"]["adopted"] and not active_geometry_fallback(v2)
    single_rate = v2["dispatch_cpu_workers"]["training_first_30"]["accepted_per_hour"]
    survey = v2.get("geometry_pair_first_30", {})
    if dual and survey.get("accepted",0)>=SURVEY_UNITS:
        rate, basis = survey["accepted_per_hour"], "first 30 production geometry units with two GPU slots"
    elif dual:
        rate = 7200/load(TRIAL)["parallel"]["seconds"]
        basis = "isolated paired qualification, before first 30 production units"
    else:
        rate, basis = single_rate, "first 30 production geometry units with one GPU slot"
    optical = v2.get("optical_slot_decision", {}).get("probe", {})
    if optical.get("adopted") and v2["optical_slot_decision"]["slots"]==3:
        optical_seconds, optical_basis = optical["per_scene_seconds"], "production three-slot trial"
    else:
        optical_seconds = v2["p2_optical_parallel_trial"]["parallel"]["seconds"]/2
        optical_basis = "qualified two-slot optical wall time"
    hours = dict(training_geometry=geometry_remaining/rate,
                 training_optical=counts["train_optical"]*optical_seconds/3600)
    total = sum(hours.values())
    now = datetime.now(timezone.utc)
    v2["p2_finish_eta"] = dict(
        status="conditional_on_optical_gate_and_future_acceptance",
        measured_at_utc=now.isoformat(), estimated_hours=total,
        estimated_finish_utc=(now+timedelta(hours=total)).isoformat(),
        remaining_hours_by_stage=hours, geometry_basis=basis, geometry_per_hour=rate,
        training_geometry_remaining_runnable=geometry_remaining, training_geometry_failed=terminal_failures,
        training_optical_basis=optical_basis,
    )


def preserve_for_retry(unit, run_dir):
    if (unit/"manifest.json").exists():
        raise RuntimeError("an admitted training scene cannot be retried")
    previous = run_dir/"failed-pair-attempts"/unit.name
    previous.parent.mkdir(exist_ok=True)
    unit.rename(previous)
    stage_geometry(previous, unit)
    return str(previous.relative_to(FORMAL_DATA))


@contextmanager
def process_gpu_samples(path):
    with path.open("x") as stream:
        sampler = subprocess.Popen(
            ["nvidia-smi", "pmon", "-i", "0", "-s", "u", "-d", "1", "-o", "DT"],
            stdout=stream, stderr=subprocess.STDOUT, env={"TZ": "UTC"},
        )
        try:
            yield
        finally:
            sampler.terminate()
            sampler.wait(timeout=5)


def gather_slot_samples(path, phases):
    processes = {row["pid"]: (tuple(row["cpu_ids"]), phase)
                 for scene in phases.values() for phase, row in scene.items()}
    result = defaultdict(list)
    for line in path.read_text().splitlines():
        columns = line.split()
        if len(columns) < 6 or not columns[3].isdigit() or not columns[5].isdigit():
            continue
        phase = processes.get(int(columns[3]))
        if phase:
            result[phase].append(int(columns[5]))
    return result


class Survey:
    def __init__(self, previous=None):
        self.started = None
        previous = previous or {}
        self.accepted = previous.get("total_admitted_in_measured_pairs",previous.get("accepted",0))
        self.elapsed_before = previous.get("seconds",0)
        self.queue_floor = previous.get("ready_queue_minimum")
        self.vram = previous.get("peak_combined_vram_bytes",0)
        self.rss_before = previous.get("peak_rss_bytes",0)
        self.samples = defaultdict(list)
        self.groups = list(previous.get("groups",[]))
        self.added_cpu = previous.get("added_cpu_worker")
        for group in self.groups:
            path = FORMAL_DATA/group["path"]
            if sha256(path)!=group["sha256"]:
                raise ValueError("the first dual-slot survey trace has changed")
            self.include_samples(path,group["phases"])

    def include_samples(self,path,phases):
        for key, readings in gather_slot_samples(path,phases).items():
            self.samples[key].extend(readings)

    def ready(self, ready_count):
        self.queue_floor = ready_count if self.queue_floor is None else min(self.queue_floor, ready_count)

    def track(self, report, path, ready_count, accepted):
        self.ready(ready_count)
        self.accepted += accepted
        self.vram = max(self.vram, report["peak_combined_vram_bytes"])
        self.include_samples(path,report["phases"])
        self.groups.append(dict(path=str(path.relative_to(FORMAL_DATA)), sha256=sha256(path), **report))

    def report(self, meter, *, fallback, recovery=None):
        elapsed = self.elapsed_before+time.monotonic()-self.started
        measured = min(self.accepted,SURVEY_UNITS)
        slots = {
            "-".join(map(str, cpus)): {
                phase:dict(samples=len(values), mean_gpu_sm_percent=sum(values)/len(values))
                for (mask, phase), values in self.samples.items() if cpus == mask and values
            } for cpus in PAIR_CPUS
        }
        return dict(
            accepted=measured, total_admitted_in_measured_pairs=self.accepted,
            seconds=elapsed, accepted_per_hour=3600*measured/elapsed,
            gpu_utilization_per_slot_and_phase=slots, measurement="nvidia-smi pmon process SM percentage, 1 second",
            peak_rss_bytes=max(self.rss_before,meter.peak_rss), peak_combined_vram_bytes=self.vram,
            ready_queue_minimum=self.queue_floor, added_cpu_worker=self.added_cpu, groups=self.groups,
            fallback=fallback, recovery=recovery,
        )
