"""B dispatch GPU-slot lineage, failover and first-30 measurements."""
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import subprocess
import time

from route3_v2data_B_batch import FORMAL_DATA, RECORD, sha256, write_json
from route3_v2data_B_geometry_pair import GPU_CAP, PAIR_CPUS, REVIEW, TRIAL, stage_geometry
from route3_v2data_B_plan import load
from route3_v2data_B_release import (
    GPU_DISPATCH_RECEIPT, GPU_DISPATCH_VERSION, PLANNER_OVERRIDE, PREVIOUS_GPU_DISPATCH_RECEIPT,
    PREVIOUS_GPU_DISPATCH_VERSION, approved_geometry_override, code_hash, gpu_dispatch_code,
    gpu_dispatch_receipt, validate_p2,
)


PAIR_HANDOFF = "training GPU slots and optical gate dispatch takeover"
NOISE_RESULT = REVIEW/"geometry-pair-noise-result.json"
POSTCHECK_PREREGISTRATION = REVIEW/"geometry-gpu-slot-postcheck-preregistration.json"
ACCEPTED_GEOMETRY_GAP = dict(
    scene_id="training-038-4df143ee5d0b1f26", label="已接受的缺口：training-038",
    failed_checks="13 帧同种子分层互补误差为 0.20–0.26，上限 0.002",
    coverage_passed=True, complement_error_other_completed_units_at_registration=0.0,
)
OPTICAL_GATE_POLICY = dict(maximum_training_failures=3, category_pause_at=2,
                           requires_drained_sealed_training_queue=True,
                           selection_pool="admitted_training_geometry")
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
    if not state.get("dispatch_cpu_workers"):
        raise ValueError("the preceding CPU dispatcher receipt is required")
    qualification = dict(path=str(NOISE_RESULT.relative_to(FORMAL_DATA)),sha256=sha256(NOISE_RESULT))
    if previous_ref:
        previous = gpu_dispatch_receipt(previous_ref,state,generator,PREVIOUS_GPU_DISPATCH_RECEIPT,
                                        PREVIOUS_GPU_DISPATCH_VERSION)
        if not previous or previous["qualification"]!=qualification:
            raise ValueError("the preceding GPU dispatch receipt or its qualification changed")
    approval = approved_geometry_override(qualification,result)
    approval_ref = dict(path=str(PLANNER_OVERRIDE.relative_to(FORMAL_DATA)),
                        sha256=sha256(PLANNER_OVERRIDE)) if approval else None
    code = gpu_dispatch_code()
    patch = dict(version=GPU_DISPATCH_VERSION, parent=state["dispatch_cpu_workers"],
                 generation_release_sha256=release["sha256"], generator_code_sha256=result["generator_code_sha256"],
                 dispatch_code=code, dispatch_code_sha256=code_hash(code),
                 qualification=qualification, qualification_adopted=result["adopted"], planner_override=approval_ref,
                 previous_gpu_dispatch=previous_ref, adopted=bool(result["adopted"] or approval),
                 optical_gate_policy=OPTICAL_GATE_POLICY, maximum_vram_bytes=GPU_CAP,
                 gpu_slot_cpus=PAIR_CPUS,
                 extra_cpu_worker_cores=EXTRA_CPU, extra_worker_ready_queue_threshold=QUEUE_MIN,
                 switched_at_utc=datetime.now(timezone.utc).isoformat())
    state["dispatch_gpu_slots"] = dict(version=patch["version"], path=GPU_DISPATCH_RECEIPT.name,
                                     sha256=write_json(GPU_DISPATCH_RECEIPT, patch))
    validate_p2(plan, state, generator)
    manifest = write_json(FORMAL_DATA/"manifest.json", state)
    record = load(RECORD)
    record["v2"].update(
        dispatch_gpu_slots=patch, geometry_pair_noise_policy=result["policy"],
        geometry_pair_noise_trial=dict(result, receipt=patch["qualification"]), manifest_sha256=manifest,
        optical_geometry_gate=dict(policy=OPTICAL_GATE_POLICY,status="waiting_for_geometry_queue"),
        depth_prior_repeatability=dict(
            statement="B 输入深度先验的串行复跑结果存在数值波动。",
            per_scene_serial_maxima={scene:rows["serial_maxima"] for scene,rows in result.get("scenes", {}).items()},
        ),
    )
    if approval:
        record["v2"]["geometry_pair_planner_override"] = dict(approval,receipt=approval_ref)
    postcheck = load(POSTCHECK_PREREGISTRATION)
    if postcheck:
        record["v2"]["geometry_gpu_slot_postcheck"] = postcheck
    provenance = record["v2"].setdefault("geometry_gpu_slot_provenance", dict(
        backfilled_at_utc=patch["switched_at_utc"], per_scene={},
    ))
    for unit in state["units"]:
        if unit["partition"]=="training":
            provenance["per_scene"].setdefault(unit["scene_id"],"single")
    record["v2"].setdefault("target_gap_details", {})["train_geometry"] = dict(
        accepted=[ACCEPTED_GEOMETRY_GAP],
    )
    record["v2"]["p2_dispatch"].update(status="running_geometry_gpu_slots", stop_after_current=False)
    write_json(RECORD, record)
    return patch


def record_geometry_slots(units):
    record = load(RECORD)
    by_scene = record["v2"]["geometry_gpu_slot_provenance"]["per_scene"]
    for unit in units:
        if unit["partition"]=="training" and "geometry_gpu_slots" in unit:
            by_scene[unit["scene_id"]] = unit["geometry_gpu_slots"]
    write_json(RECORD,record)


def optical_gate(specs, units, failures):
    planned = {spec["scene_id"]:spec["category"] for spec in specs}
    admitted = {unit["scene_id"] for unit in units if unit["partition"]=="training"}
    failed = {row["scene_id"] for row in failures}
    counted = {row["scene_id"] for row in failures if row.get("counts_toward_pause", True)}
    by_category = Counter(planned[scene] for scene in counted & planned.keys())
    paused = sorted(name for name,count in by_category.items() if count>=OPTICAL_GATE_POLICY["category_pause_at"])
    unaccounted = sorted(planned.keys()-admitted-failed)
    contradictory = sorted((admitted & failed) | ((admitted | failed)-planned.keys()))
    failure_limit_passed = len(counted)<=OPTICAL_GATE_POLICY["maximum_training_failures"]
    return dict(
        allowed=failure_limit_passed and not paused and not unaccounted and not contradictory,
        policy=OPTICAL_GATE_POLICY, planned=len(planned), admitted=len(admitted), observed_failures=len(counted),
        failed_scene_ids=sorted(failed), failure_count_by_category=dict(by_category), paused_categories=paused,
        unaccounted_scene_ids=unaccounted, contradictory_scene_ids=contradictory,
    )


def update_slot_eta(v2, training_failures):
    counts = v2["target_gap"]
    terminal_failures = len({row["scene_id"] for row in training_failures})
    geometry_remaining = max(0,counts["train_geometry"]-terminal_failures)
    dual = v2["dispatch_gpu_slots"]["adopted"] and not v2.get("geometry_pair_fallback")
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
    def __init__(self):
        self.started = None
        self.accepted = 0
        self.queue_floor = None
        self.vram = 0
        self.samples = defaultdict(list)
        self.groups = []
        self.added_cpu = None

    def ready(self, ready_count):
        self.queue_floor = ready_count if self.queue_floor is None else min(self.queue_floor, ready_count)

    def track(self, report, path, ready_count, accepted):
        self.ready(ready_count)
        self.accepted += accepted
        self.vram = max(self.vram, report["peak_combined_vram_bytes"])
        for key, readings in gather_slot_samples(path, report["phases"]).items():
            self.samples[key].extend(readings)
        self.groups.append(dict(path=str(path.relative_to(FORMAL_DATA)), sha256=sha256(path), **report))

    def report(self, meter, *, fallback):
        elapsed = time.monotonic() - self.started
        slots = {
            "-".join(map(str, cpus)): {
                phase:dict(samples=len(values), mean_gpu_sm_percent=sum(values)/len(values))
                for (mask, phase), values in self.samples.items() if cpus == mask and values
            } for cpus in PAIR_CPUS
        }
        return dict(
            accepted=self.accepted, seconds=elapsed, accepted_per_hour=3600*self.accepted/elapsed,
            gpu_utilization_per_slot_and_phase=slots, measurement="nvidia-smi pmon process SM percentage, 1 second",
            peak_rss_bytes=meter.peak_rss, peak_combined_vram_bytes=self.vram, ready_queue_minimum=self.queue_floor,
            added_cpu_worker=self.added_cpu, groups=self.groups, fallback=fallback,
        )
