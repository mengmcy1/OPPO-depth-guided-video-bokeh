"""Measure and, if qualified, dispatch paired B optical references under one GPU lock."""
from collections import Counter
from contextlib import ExitStack, nullcontext
import fcntl
import os
import subprocess
import time

import numpy as np
import psutil

from route3_v2data_B_batch import (
    FORMAL_DATA, GIB, GPU_LOCK, OPTICAL_TAGS, RECORD, archive_status, blender, optical_selection,
    reserve, sha256, verify_optical, write_json,
)
from route3_v2data_B_plan import load
from route3_v2data_B_presence import attach_optical_visibility
from route3_v2data_B_release import OPTICAL_FRAMES, PLAN_DIR, load_production_plan
from route3_v2data_B_production_policy import (
    OpticalQueue, ProductionHealth, ProductionStop, ranked_optical_candidates, system_failure,
)


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


def resources(renderers=None):
    root = psutil.Process()
    selected = root.children(recursive=True) if renderers is None else []
    for job in renderers or ():
        try:
            process = psutil.Process(job.pid)
            selected.extend([process, *process.children(recursive=True)])
        except psutil.NoSuchProcess:
            continue
    pids, rss = set(), 0
    for process in [root, *selected]:
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


def render_group(units, allocation, *, parallel, frames=OPTICAL_FRAMES, seed=SEED,
                 held_lock=None, abort=None, slot_cpus=None, sample_gpu=False, vram_cap=VRAM_CAP):
    if slot_cpus is None:
        workers = [*allocation[1], *allocation[2]]
        half = min(6, len(workers)//2)
        if half < 1:
            raise RuntimeError("paired optical qualification requires two disjoint worker CPU sets")
        slot_cpus = (workers[:half], workers[half:2*half])
    if parallel and len(units)>len(slot_cpus):
        raise ValueError("optical units exceed the configured GPU slot CPUs")
    gpu_samples = []
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="0", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    peak_rss = peak_vram = 0
    result = {unit.name:dict(passed=False) for unit in units}
    before_lock = time.monotonic()
    with (GPU_LOCK.open("a") if held_lock is None else nullcontext(held_lock)) as lock:
        if held_lock is None:
            fcntl.flock(lock, fcntl.LOCK_EX)
        started = time.monotonic()
        groups = [units] if parallel else [[unit] for unit in units]
        for group in groups:
            processes = []
            with ExitStack() as streams:
                try:
                    for index, unit in enumerate(group):
                        cpus = slot_cpus[index]
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
                    next_sample = time.monotonic()
                    while any(process.poll() is None for process in processes):
                        rss, vram = resources(processes)
                        peak_rss, peak_vram = max(peak_rss, rss), max(peak_vram, vram)
                        if abort is not None and abort.is_set():
                            raise RuntimeError("combined dispatch resource limit")
                        if rss > RSS_CAP or vram > vram_cap:
                            raise RuntimeError("optical render exceeded its combined RSS or VRAM cap")
                        if sample_gpu:
                            utilization = subprocess.check_output(
                                ["nvidia-smi", "-i", "0", "--query-gpu=utilization.gpu",
                                 "--format=csv,noheader,nounits"], text=True,
                            )
                            gpu_samples.append(int(utilization.strip()))
                        next_sample += 1
                        time.sleep(max(0,next_sample-time.monotonic()))
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
        peak_combined_vram_bytes=peak_vram, scenes=result, worker_cpu_counts=[len(mask) for mask in slot_cpus],
        gpu_utilization_percent=gpu_samples,
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


def paired_trial(estimate, allocation, version, *, held_lock=None, force_qualification=False, abort=None):
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
        if force_qualification or load(source/"manifest.json").get("generator_version") != version:
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
        report[mode] = render_group(units, allocation, parallel=mode == "parallel", held_lock=held_lock, abort=abort)
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
            noise = render_group(units, allocation, parallel=False, frames=(30,), seed=SEED+1,
                                 held_lock=held_lock, abort=abort)
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
    v2["optical_parallel_trial"] = value
    if v2["p2_dispatch"].get("production_measurements"):
        update_p2_eta(v2)
    write_json(RECORD, record)


def optical_slot_probe(group, allocation, tri_cpus, baseline, estimate, run):
    from route3_v2data_B_geometry_pair import array_differences

    report = dict(probe=run, adopted=False, equivalence_passed=False)
    if len(group)!=3 or not all(row["passed"] for row in run["scenes"].values()):
        report["reason"] = "expected three successful optical renders"
        return report
    _, witness, source = group[0]
    reserve(FORMAL_DATA, estimate)
    repeat = FORMAL_DATA/"p2-review"/f"optical-third-slot-{time.time_ns()}"/witness.name
    repeat.parent.mkdir(parents=True)
    stage_optical(source, repeat)
    serial = render_group([repeat], allocation, parallel=False, frames=(30,), slot_cpus=(tri_cpus[0],))
    report.update(serial=serial, witness=witness.name, serial_directory=str(repeat.relative_to(FORMAL_DATA)))
    if all(row["passed"] for row in serial["scenes"].values()):
        diffs = {}
        files = sorted(p.name for p in (repeat/"labels").glob("f030-*.npz")
                       if p.stem.endswith(("-optics", "-hdr")))
        if len(files)!=8:
            raise ValueError("f030 comparison must include four LDR and four HDR products")
        for suffix in ("optics", "hdr"):
            selected = [name for name in files if name.endswith(f"-{suffix}.npz")]
            diffs[suffix] = array_differences(repeat/"labels", witness/"labels", selected, ("rgb_linear",),
                                              half_ulp=True)["rgb_linear"]
        report["differences"] = diffs
        report["equivalence_passed"] = all(row["max_half_ulp"]<=1 for row in diffs.values())
    trial_seconds = run["seconds"]/3
    baseline_seconds = sum(row["seconds"] for row in baseline)/4
    report.update(per_scene_seconds=trial_seconds, baseline_per_scene_seconds=baseline_seconds,
                  time_ratio=trial_seconds/baseline_seconds,
                  speed_passed=trial_seconds<=.9*baseline_seconds,
                  resources_passed=run["peak_combined_vram_bytes"]<=28*GIB and run["peak_combined_rss_bytes"]<=RSS_CAP)
    report["adopted"] = all(report[key] for key in ("equivalence_passed","speed_passed","resources_passed"))
    return report


def retry_optical_group(group, allocation, cpus, *, parallel, health, attempt):
    archive = FORMAL_DATA/"p2-review"/f"optical-recovery-{time.time_ns()}"
    archive.mkdir(parents=True)
    for _, unit, source in group:
        unit.rename(archive/unit.name)
        stage_optical(source,unit)
    report = render_group([unit for _,unit,_ in group],allocation,parallel=parallel,slot_cpus=cpus)
    for _,unit,_ in group:
        row = report["scenes"][unit.name]
        health.add("optical",unit.name,row["passed"],attempt=attempt,reason=row.get("failure"))
    return report["scenes"]


def produce_pairs(estimate, allocation, version, *, survey_slots=False, on_policy_decision=None,
                  on_selection=None, health=None):
    health = health or ProductionHealth()
    target = FORMAL_DATA/"optical"
    target.mkdir(exist_ok=True)
    selection = target/"selection.json"
    admitted = [load(path) for path in sorted((FORMAL_DATA/"units/training").glob("*/manifest.json"))]
    rankings = ranked_optical_candidates(admitted)
    if not selection.exists():
        try:
            chosen = optical_selection(admitted)
        except ValueError as error:
            if not str(error).startswith(("fewer eligible training geometry", "insufficient training geometry")):
                raise
            chosen, reserved = [], set()
            for index in range(80):
                tag = OPTICAL_TAGS[index%len(OPTICAL_TAGS)]
                item = next((row for row in rankings[tag] if row["scene_id"] not in reserved), None)
                if item:
                    chosen.append(item)
                    reserved.add(item["scene_id"])
        write_json(selection,chosen)
    selection_sha = sha256(selection)
    candidates = load(selection)
    queue = OpticalQueue(candidates,rankings,target/"ranked-replacements.json")
    failures_path = target/"failures.json"
    failures = load(failures_path) or []
    excluded = {row["scene_id"] for row in failures if row.get("counts_toward_pause",True)}
    accepted_ids = {path.parent.name for path in target.glob("*/manifest.json")}
    if on_selection:
        on_selection(candidates,selection)
    slots_path = target/"slot-decision.json"
    policy = load(slots_path) or dict(baseline=[],baseline_accepted=0,slots=2)
    coordinator, geometry, gpu = allocation
    workers = gpu if survey_slots else [*geometry,*gpu]
    half = min(6,len(workers)//2)
    double_cpus = (workers[:half],workers[half:2*half])
    tri_cpus = (tuple(cpu for mask in geometry for cpu in mask), *double_cpus) if survey_slots else ()
    if survey_slots and any(len(cpus)!=6 for cpus in tri_cpus):
        raise ProductionStop("unrecoverable_gpu_or_environment","optical production requires three groups of six CPU cores")

    def save_policy():
        write_json(slots_path,policy)
        record=load(RECORD)
        record["v2"]["optical_slot_decision"] = policy
        write_json(RECORD,record)

    def save_replacements():
        record=load(RECORD)
        latest=record["v2"]["training_optical_selection"] if on_selection else record["v2"].setdefault("training_optical_selection",{})
        latest.update(replacements=queue.replacements(), remaining_target=80-len(accepted_ids),
                      exhausted_slots=queue.exhausted(excluded))
        write_json(RECORD,record)

    def failure(item, reason):
        scene_id=item["scene_id"]
        failures.append(dict(scene_id=scene_id,category=item["primary_difficulty"],reason=reason))
        excluded.add(scene_id)
        write_json(failures_path,failures)

    while not (FORMAL_DATA/"stop-after-current").is_file():
        health.check()
        group = []
        baseline_pending = survey_slots and policy["baseline_accepted"]<4
        probe_pending = survey_slots and policy.get("two_slot_mean_gpu_percent",100)<75 and "probe" not in policy
        width = (min(2,4-policy["baseline_accepted"]) if baseline_pending
                 else 3 if probe_pending else policy["slots"])
        reserve(FORMAL_DATA,estimate*width)
        for _,item in queue.pending(excluded,accepted_ids):
            scene_id = item["scene_id"]
            unit = target/scene_id
            source = FORMAL_DATA/"units/training"/scene_id
            if unit.exists():
                archive = FORMAL_DATA/"p2-review"/f"optical-interrupted-{time.time_ns()}"
                archive.mkdir(parents=True)
                unit.rename(archive/unit.name)
            stage_optical(source,unit)
            group.append((item,unit,source))
            if len(group)==width:
                break
        if not group:
            break
        save_replacements()

        is_triple = width==3 and len(group)==3
        run = render_group([unit for _,unit,_ in group],(coordinator,[],gpu),parallel=True,
                           slot_cpus=tri_cpus if is_triple else double_cpus,sample_gpu=baseline_pending,
                           vram_cap=28*GIB if is_triple else VRAM_CAP)
        rows=run["scenes"].copy()
        for _,unit,_ in group:
            row=rows[unit.name]
            health.add("optical",unit.name,row["passed"],attempt="triple" if is_triple else "paired",
                       reason=row.get("failure"))
        fallback = False
        if probe_pending:
            try:
                policy["probe"] = optical_slot_probe(group,allocation,tri_cpus,policy["baseline"],estimate,run)
            except (OSError,RuntimeError,ValueError,KeyError) as error:
                policy["probe"] = dict(probe=run,adopted=False,equivalence_passed=False,reason=str(error))
            policy["slots"] = 3 if policy["probe"]["adopted"] else 2
            fallback = not policy["probe"]["equivalence_passed"]
            save_policy()
        elif is_triple and not all(row["passed"] for row in rows.values()):
            policy["slots"] = 2
            policy["production_fallback"] = run
            fallback = True
            save_policy()

        def failed_systemically(unit):
            row=rows[unit.name]
            return not row["passed"] and system_failure(row.get("failure"),unit/"render.log",
                                                         exit_code=row.get("exit_code"))

        # Failed scene checks become gaps. Successful but unqualified triple outputs
        # and GPU/environment errors use the established double-slot recovery path.
        if fallback:
            retry = [(item,unit,source) for item,unit,source in group
                     if rows[unit.name]["passed"] or failed_systemically(unit)]
            for index in range(0,len(retry),2):
                rows.update(retry_optical_group(retry[index:index+2],allocation,double_cpus,
                                               parallel=True,health=health,attempt="paired_recovery"))
        for item,unit,source in group:
            if failed_systemically(unit):
                rows.update(retry_optical_group([(item,unit,source)],allocation,double_cpus,
                                               parallel=False,health=health,attempt="single_recovery"))

        unrecovered=[]
        for item, unit, source in group:
            row = rows[unit.name]
            if not row["passed"]:
                failure(item,row["failure"])
                if failed_systemically(unit):
                    unrecovered.append(f"{unit.name}: {row['failure']}")
                continue
            labels = unit/"labels"
            attach_optical_visibility(labels/"optical-audit.json",source/"labels/foreground-presence.json")
            products = sorted(path for path in labels.glob("*.npz") if not path.is_symlink())
            outputs = [dict(path=str(path.relative_to(FORMAL_DATA)),bytes=path.stat().st_size,sha256=sha256(path))
                       for path in products]
            write_json(unit/"manifest.json",dict(
                **item,generator_version=version,optical_frames=list(OPTICAL_FRAMES),partition="training",
                geometry_manifest_sha256=sha256(source/"manifest.json"),selection_sha256=selection_sha,
                bytes=sum(row["bytes"] for row in outputs),files=outputs,
            ))
            accepted_ids.add(unit.name)
        if baseline_pending and any(row["passed"] for row in rows.values()):
            accepted = sum(row["passed"] for row in rows.values())
            policy["baseline"].append(run)
            policy["baseline_accepted"] += accepted
            samples = [usage for batch in policy["baseline"] for usage in batch["gpu_utilization_percent"]]
            if policy["baseline_accepted"]==4 and samples:
                policy["two_slot_mean_gpu_percent"] = sum(samples)/len(samples)
            save_policy()
        archive_status()
        save_replacements()
        if unrecovered:
            raise ProductionStop("unrecoverable_gpu_or_environment","; ".join(unrecovered))
        health.check()
        if on_policy_decision and (probe_pending or fallback or
                                   baseline_pending and policy.get("two_slot_mean_gpu_percent",0)>=75):
            on_policy_decision()
    archive_status()
    save_replacements()


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
