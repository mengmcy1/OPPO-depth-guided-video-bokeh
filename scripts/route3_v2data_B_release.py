"""Validate B v2 sealed plans and their explicit generator patch chain."""
import hashlib
import json

from route3_v2data_B_front import RECORD, ROOT, write_json
from route3_v2data_B_inputs import FORMAL_DATA
from route3_v2data_B_plan import MODES, digest, generator_code, load


OPTICAL_FRAMES = (1, 30, 60)
PATCHABLE = {"route3_v2data_B_batch.py", "route3_v2data_B_render.py"}
DISPATCH_CODE = ("route3_v2data_B_dispatch.py", "route3_v2data_B_release.py")
P2_DISPATCH_CODE = (*DISPATCH_CODE, "route3_v2data_B_deploy_p2.py", "route3_v2data_B_optical_pair.py")
WORKER_DISPATCH_FILE = "route3_v2data_B_dispatch_workers.py"
DISPATCH_PATCH_VERSION = "p2-cpu-workers-3"
PREVIOUS_GPU_DISPATCHES = {
    "p2-training-gpu-slots-optical-gate": "dispatch-gpu-slots-p2.json",
    "p2-training-gpu-slots-planner-decision": "dispatch-gpu-slots-planner-p2.json",
}
GPU_DISPATCH_VERSION = "p2-training-gpu-slots-scene-failure-recovery"
GPU_DISPATCH_FILES = ("route3_v2data_B_geometry_pair.py", "route3_v2data_B_geometry_slots.py")
GPU_DISPATCH_RECEIPT = FORMAL_DATA/"dispatch-gpu-slots-recovery-p2.json"
PLANNER_OVERRIDE = FORMAL_DATA/"p2-review/geometry-pair-planner-override.json"
FALLBACK_RECOVERY = FORMAL_DATA/"p2-review/geometry-slot-fallback-recovery-130.json"
P2_CHANGED_FILES = {"route3_v2data_B_scene.py", "route3_v2data_B_batch.py"}
PLAN_DIR = FORMAL_DATA/"sealed-scene-plans"
P2_ADDITIONAL_GENERATION_FILES = ("route3_v2data_B_placement.py", "route3_v2data_B_presence.py")


def p2_generation_code():
    return {**generator_code(), **{name:digest(ROOT/"scripts"/name) for name in P2_ADDITIONAL_GENERATION_FILES}}


def code_hash(code):
    return hashlib.sha256(json.dumps(code,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def patched_code(base, current):
    if set(base)!=set(current):
        raise ValueError("the sealed generator dependency list changed")
    changes = {name:dict(before=base[name],after=current[name]) for name in base if base[name]!=current[name]}
    if set(changes)!=PATCHABLE:
        raise ValueError(f"unexpected files changed from the sealed generator: {sorted(changes)}")
    return changes


def dispatch_code(*, p2=False):
    return {name:digest(ROOT/"scripts"/name) for name in (P2_DISPATCH_CODE if p2 else DISPATCH_CODE)}


def cpu_worker_dispatch_code():
    return {**dispatch_code(p2=True), WORKER_DISPATCH_FILE:digest(ROOT/"scripts"/WORKER_DISPATCH_FILE)}


def gpu_dispatch_code():
    return {**cpu_worker_dispatch_code(),
            **{name:digest(ROOT/"scripts"/name) for name in GPU_DISPATCH_FILES}}


def gpu_dispatch_receipt(ref, state, generator, path, version):
    if not ref or ref["version"]!=version or ref["path"]!=path.name:
        return None
    gpu = load(path)
    if not (
        gpu and digest(path)==ref["sha256"] and gpu["version"]==version
        and gpu["parent"]==state["dispatch_cpu_workers"]
        and gpu["generation_release_sha256"]==state["generator_patch"]["sha256"]
        and gpu["generator_code_sha256"]==generator["generator_code_sha256"]
        and gpu["dispatch_code_sha256"]==code_hash(gpu["dispatch_code"])
        and digest(FORMAL_DATA/gpu["qualification"]["path"])==gpu["qualification"]["sha256"]
    ):
        return None
    return gpu


def previous_gpu_dispatch(ref, state, generator):
    name = PREVIOUS_GPU_DISPATCHES.get(ref["version"])
    return gpu_dispatch_receipt(ref,state,generator,FORMAL_DATA/name,ref["version"]) if name else None


def approved_geometry_override(qualification, result):
    approval = load(PLANNER_OVERRIDE)
    if not approval:
        return None
    if not (
        approval["decision"]=="enable_dual_geometry"
        and approval["authority"]=="规划方经用户同意"
        and approval["qualification"]==dict(qualification,adopted=result["adopted"])
        and result["calibration_complete"] and result["original_speed_passed"] and result["original_resources_passed"]
    ):
        raise ValueError("the planner approval does not reference the frozen geometry qualification")
    return approval


def verifies_dispatch(state, receipt):
    ref = state.get("dispatch_cpu_workers")
    if not ref:
        return receipt["dispatch_code"] == dispatch_code(p2=True)
    path = FORMAL_DATA/ref["path"]
    patch = load(path)
    valid_base = (
        patch and ref["version"] == DISPATCH_PATCH_VERSION and digest(path) == ref["sha256"]
        and patch["version"] == ref["version"] and patch["base_dispatch_code"] == receipt["dispatch_code"]
        and patch["generation_release_sha256"] == state["generator_patch"]["sha256"]
        and patch["generator_code_sha256"] == receipt["generator_code_sha256"]
        and patch["dispatch_code_sha256"] == code_hash(patch["dispatch_code"])
    )
    if not valid_base:
        return False
    gpu_ref = state.get("dispatch_gpu_slots")
    if not gpu_ref:
        return patch["dispatch_code"] == cpu_worker_dispatch_code()
    gpu = gpu_dispatch_receipt(gpu_ref,state,receipt,GPU_DISPATCH_RECEIPT,GPU_DISPATCH_VERSION)
    if not gpu or gpu["dispatch_code"] != gpu_dispatch_code():
        return False
    previous = gpu.get("previous_gpu_dispatch")
    if previous:
        previous_receipt = previous_gpu_dispatch(previous,state,receipt)
        if not previous_receipt or previous_receipt["qualification"]!=gpu["qualification"]:
            return False
    result = load(FORMAL_DATA/gpu["qualification"]["path"])
    if not result or gpu["qualification_adopted"]!=result["adopted"]:
        return False
    approval = gpu.get("planner_override")
    if approval:
        if (approval["path"]!=str(PLANNER_OVERRIDE.relative_to(FORMAL_DATA))
                or digest(PLANNER_OVERRIDE)!=approval["sha256"]
                or not approved_geometry_override(gpu["qualification"],result)):
            return False
    recovery = gpu.get("fallback_recovery")
    if recovery and (recovery["path"]!=str(FALLBACK_RECOVERY.relative_to(FORMAL_DATA))
                     or digest(FALLBACK_RECOVERY)!=recovery["sha256"]):
        return False
    return gpu["adopted"]==bool(result["adopted"] or approval)


def sealed_plan(path):
    if path.parent.resolve()!=PLAN_DIR.resolve() or path.stem not in MODES:
        raise ValueError("only the v2 production plans can be dispatched")
    plan = load(path)
    state = load(FORMAL_DATA/"manifest.json")
    expected = load(PLAN_DIR/"sha256.json")
    if expected.get(path.stem)!=digest(path) or state["scene_plan_sha256"]!=expected:
        raise ValueError("sealed plan checksum was changed")
    if not state.get("eligible_for_formal_use") or state["generator_version"]!=plan["generator_version"]:
        raise ValueError("production requires the accepted sealed generator version")
    return plan,state


def parent_p1_code(plan, state, release):
    parent_path = FORMAL_DATA / "generator-patch-p1.json"
    parent = load(parent_path)
    base = plan["generator_code"]
    code = {**base, **{name: change["after"] for name, change in parent["changes"].items()}}
    if (
        release["path"] != parent_path.name or digest(parent_path) != release["sha256"]
        or parent["base_version"] != plan["generator_version"]
        or release["version"] != parent["patch_version"]
        or parent["base_generator_code_sha256"] != code_hash(base)
        or parent["generator_code_sha256"] != code_hash(code)
        or parent["changes"] != patched_code(base, code)
        or parent["sealed_plan_sha256"] != state["scene_plan_sha256"]
        or parent["optical_frames"] != list(OPTICAL_FRAMES)
    ):
        raise ValueError("the parent B p1 release no longer matches the sealed plans")
    return code


def p2_changes(parent, current):
    if set(current) - set(parent) != set(P2_ADDITIONAL_GENERATION_FILES) or set(parent) - set(current):
        raise ValueError("unexpected generator dependency change in p2")
    changes = {name:dict(before=parent.get(name), after=value)
               for name, value in current.items() if parent.get(name) != value}
    if set(changes) != P2_CHANGED_FILES | set(P2_ADDITIONAL_GENERATION_FILES):
        raise ValueError(f"unexpected B p2 code changes: {sorted(changes)}")
    return changes


def validate_p2(plan, state, receipt):
    current = p2_generation_code()
    parent = parent_p1_code(plan, state, receipt["parent_patch"])
    archive = receipt.get("archive")
    if archive and digest(FORMAL_DATA / archive["path"]) != archive["sha256"]:
        raise ValueError("the p2 archive inventory checksum has changed")
    if (
        receipt["base_version"] != plan["generator_version"]
        or receipt["patch_version"] != plan["generator_version"] + "-p2"
        or receipt["base_generator_code_sha256"] != code_hash(plan["generator_code"])
        or receipt["parent_generator_code_sha256"] != code_hash(parent)
        or receipt["generator_code_sha256"] != code_hash(current)
        or receipt["changes"] != p2_changes(parent, current)
        or receipt["sealed_plan_sha256"] != state["scene_plan_sha256"]
        or not verifies_dispatch(state, receipt)
        or receipt["optical_frames"] != list(OPTICAL_FRAMES)
        or digest(FORMAL_DATA / receipt["qualification"]["path"]) != receipt["qualification"]["sha256"]
        or digest(FORMAL_DATA / receipt["preflight"]["path"]) != receipt["preflight"]["sha256"]
    ):
        raise ValueError("the installed B p2 release differs from the qualified code or sealed plans")


def load_production_plan(path):
    plan,state = sealed_plan(path)
    base,current = plan["generator_code"],generator_code()
    effective = plan["generator_version"]
    if state.get("generator_patch"):
        release = state["generator_patch"]
        record = load(FORMAL_DATA/release["path"])
        if release["version"] == effective + "-p2":
            if digest(FORMAL_DATA/release["path"]) != release["sha256"]:
                raise ValueError("B p2 release checksum has changed")
            validate_p2(plan, state, record)
            return dict(plan, effective_generator_version=record["patch_version"])
        if (
            digest(FORMAL_DATA/release["path"])!=release["sha256"]
            or record["base_version"]!=effective
            or record["patch_version"]!=release["version"]
            or record["base_generator_code_sha256"]!=code_hash(base)
            or record["sealed_plan_sha256"]!=state["scene_plan_sha256"]
            or record["generator_code_sha256"]!=code_hash(current)
            or record["changes"]!=patched_code(base,current)
            or record["dispatch_code"]!=dispatch_code()
            or record["optical_frames"]!=list(OPTICAL_FRAMES)
        ):
            raise ValueError("the installed B optical patch does not match the sealed release")
        effective = record["patch_version"]
    elif current!=base:
        raise ValueError("generator source differs from the sealed plan")
    return dict(plan,effective_generator_version=effective)


def install_optical_patch():
    plan,state = sealed_plan(PLAN_DIR/"training.json")
    if state.get("generator_patch"):
        return load_production_plan(PLAN_DIR/"training.json")["effective_generator_version"]
    if not (FORMAL_DATA/"stop-after-current").is_file():
        raise ValueError("drain the existing queue before installing the optical patch")
    if state["counts"]["development"] or state["counts"]["train_optical"]:
        raise ValueError("existing optical production requires a separately reviewed migration")
    for directory in ("units/development","optical"):
        if any(item.is_dir() for item in (FORMAL_DATA/directory).glob("*")):
            raise ValueError("an incomplete optical unit already exists")
    for mode in MODES:
        other,_ = sealed_plan(PLAN_DIR/(mode+".json"))
        if other["generator_code"]!=plan["generator_code"]:
            raise ValueError("the sealed producer code differs across partitions")
    current=generator_code()
    version=plan["generator_version"]+"-p1"
    patch=dict(
        base_version=plan["generator_version"],patch_version=version,
        generator_code_sha256=code_hash(current),base_generator_code_sha256=code_hash(plan["generator_code"]),
        changes=patched_code(plan["generator_code"],current),dispatch_code=dispatch_code(),
        optical_frames=list(OPTICAL_FRAMES),previous_optical_frame_count=13,
        scope="optical reference frames only in generated data; two-slot dispatch is operational",
        sealed_plan_sha256=state["scene_plan_sha256"],preserved_unit_counts=state["counts"],
    )
    path=FORMAL_DATA/"generator-patch-p1.json"
    if path.is_file() and load(path)!=patch:
        raise ValueError("an interrupted patch installation has a different receipt")
    patch_sha=write_json(path,patch)
    state["generator_patch"]=dict(path=path.name,sha256=patch_sha,version=version)
    state["optical_reference_frames"]=list(OPTICAL_FRAMES)
    manifest_sha=write_json(FORMAL_DATA/"manifest.json",state)
    record=load(RECORD)
    record["v2"].update(generator_patch=dict(patch,sha256=patch_sha),manifest_sha256=manifest_sha)
    write_json(RECORD,record)
    return version
