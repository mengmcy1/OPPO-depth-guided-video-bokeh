"""Check sealed B v2 plans and the explicitly recorded optical frame patch."""
import hashlib
import json

from route3_v2data_B_front import RECORD, ROOT, write_json
from route3_v2data_B_inputs import FORMAL_DATA
from route3_v2data_B_plan import MODES, digest, generator_code, load


OPTICAL_FRAMES = (1, 30, 60)
PATCHABLE = {"route3_v2data_B_batch.py", "route3_v2data_B_render.py"}
DISPATCH_CODE = ("route3_v2data_B_dispatch.py", "route3_v2data_B_release.py")
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


def dispatch_code():
    return {name:digest(ROOT/"scripts"/name) for name in DISPATCH_CODE}


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


def load_production_plan(path):
    plan,state = sealed_plan(path)
    base,current = plan["generator_code"],generator_code()
    effective = plan["generator_version"]
    if state.get("generator_patch"):
        release = state["generator_patch"]
        record = load(FORMAL_DATA/release["path"])
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
