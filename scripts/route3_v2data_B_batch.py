"""Generate sealed B units and summarize counts without rerendering sealed tests."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import psutil

from route3_v2data_B_front import DATA, RECORD, RESERVE_BYTES, ROOT, write_json


BLENDER = ROOT.parent/"tools/blender-4.5.13-linux-x64/blender"
GPU_LOCK = ROOT.parent/"route3-cache/gpu-data.lock"
SCRIPTS = ROOT/"scripts"
GIB = 2**30
PRODUCTION_MODES = {"training", "development", "test_unseen_assets", "test_unseen_occlusions"}
OPTICAL_TAGS = (
    "foreground_focus", "background_focus_with_disocclusion", "thin_and_complex_edge_transmission",
    "multiple_layers_and_local_occlusion", "highlight_and_bokeh",
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(2**20):
            digest.update(block)
    return digest.hexdigest()


def blender(script, *args):
    return [str(BLENDER), "--background", "--disable-autoexec", "--threads", "6",
            "--python-exit-code", "1", "--python", str(SCRIPTS/script), "--", *map(str,args)]


def reserve(directory, bytes_for_job):
    if shutil.disk_usage(directory).free < RESERVE_BYTES + bytes_for_job:
        raise RuntimeError("insufficient disk space for the batch plus the 60 GiB reserve")


def run(command, log, *, uses_gpu=False):
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="0" if uses_gpu else "",
                       OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    if uses_gpu:
        command = ["flock", str(GPU_LOCK), *command]
    start = time.monotonic()
    peak_rss = peak_device = 0
    with log.open("wb") as stream:
        job = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=environment)
        process = psutil.Process(job.pid)
        problem = None
        while job.poll() is None:
            try:
                children = [process, *process.children(recursive=True)]
                peak_rss = max(peak_rss, sum(p.memory_info().rss for p in children))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                time.sleep(.5)
                continue
            if uses_gpu:
                by_pid = {p.pid for p in children}
                reply = subprocess.run(
                    ["nvidia-smi","--query-compute-apps=pid,used_memory","--format=csv,noheader,nounits"],
                    text=True,capture_output=True,check=True,
                )
                for line in reply.stdout.splitlines():
                    pid, _, mib = line.partition(",")
                    if pid.strip().isdigit() and int(pid) in by_pid and mib.strip().isdigit():
                        peak_device = max(peak_device,int(mib)*2**20)
            if peak_rss > 28*GIB or peak_device > 24*GIB:
                for child in reversed(children):
                    try:
                        child.terminate()
                    except psutil.NoSuchProcess:
                        pass
                problem = "scene exceeded the allowed CPU or GPU resident memory"
                break
            time.sleep(.5)
        exit_code = job.wait()
    usage = dict(seconds=time.monotonic()-start, peak_rss_bytes=peak_rss, peak_vram_bytes=peak_device,
                 exit_code=exit_code)
    write_json(log.with_suffix(".resources.json"),usage)
    if problem or exit_code:
        raise RuntimeError(problem or f"scene process returned {exit_code}: {log.name}")
    return usage


def verified_unit(unit, spec, generator, optical):
    observations = sorted((unit/"observations").glob("*.npz"))
    labels = unit/"labels"
    geometry = json.loads((labels/"scene-supervision.json").read_text())
    radiance = json.loads((labels/"radiance-audit.json").read_text())
    if (radiance.get("layer_protocol") != "same-seed complementary camera-ray holdouts"
            or radiance.get("observed_reference") != "same raw render used for the saved input"):
        raise ValueError("B radiance labels must audit the saved observation and its same-seed complements")
    expected_rgb = 60
    expected_labels = 13
    need = {
        "RGB": (len(observations), expected_rgb),
        "geometry": (len(list(labels.glob("*-pinhole.npz"))), expected_labels),
        "peeled_geometry": (len(list(labels.glob("*-peeled.npz"))), expected_labels),
        "peeled_color": (len(list(labels.glob("*-peeled-color.npz"))), expected_labels),
        "layer_colors": (len(list(labels.glob("*-colors.npz"))), expected_labels),
        "radiance_audits": (len({row["frame"] for row in radiance["audits"]}), expected_labels),
        "K16": (len(list(labels.glob("*-K16.npz"))), expected_labels*2),
        "optical_pairs": (len(radiance["optical"]), expected_labels*4 if optical else 0),
    }
    bad = [name for name, (found, required) in need.items() if found != required]
    if bad or not all(row["passed"] for row in radiance["audits"]):
        raise ValueError(f"fixed B label/observation check failed: {bad}")
    files = sorted(p for p in unit.rglob("*") if p.is_file() and p.suffix not in {".log",".part"})
    assets = spec["assets"]
    identity = {key: spec[key] for key in ("scene_id","seed","partition","category","distance","motion","camera_trajectory")}
    output_files = [dict(path=str(path.relative_to(unit)),bytes=path.stat().st_size,sha256=sha256(path)) for path in files]
    result = dict(
        **identity,generator_version=generator,assets=assets,files=output_files,
        bytes=sum(row["bytes"] for row in output_files),observations="observations/",labels="labels/",
        scene_supervision=geometry["rows"],radiance_audit=radiance["audits"],depth_bands=spec["depth_bands"],
    )
    return result


def produce(unit, spec, generator, estimate):
    optical = spec["partition"] == "development"
    reserve(DATA, estimate)
    unit.mkdir(parents=True, exist_ok=False)
    description = unit/"scene-spec.json"
    write_json(description,spec)
    run(blender("route3_v2data_B_scene.py","--spec",description,"--output",unit),unit/"build.log")
    near = spec["near_anchor_m"]
    run(
        [sys.executable,str(SCRIPTS/"route3_v2data_B_labels.py"),"--scene",str(unit),"--focus",
         str(near),str(near*spec["depth_ratio"])], unit/"geometry.log",
    )
    opts = ["--optical"] if optical else []
    run(blender("route3_v2data_B_render.py","--scene",unit,*opts),unit/"render.log",uses_gpu=True)
    status = verified_unit(unit,spec,generator,optical)
    write_json(unit/"manifest.json",status)
    return status


def combine_geometry(units):
    cells = defaultdict(lambda: dict(
        scenes=0, valid_boundary_pixels=0, valid_dual_pixels=0, label_pixels=0,
        recoverable_aperture_rays=0, hidden_aperture_rays=0, never_visible_rays=0,
        relative_depth_histogram=[0,0,0], aperture_configurations=Counter(),
    ))
    for unit in units:
        if unit["partition"] != "training":
            continue
        cell = cells["/".join(unit[key] for key in ("category","distance","motion"))]
        cell["scenes"] += 1
        for frame in unit["scene_supervision"]:
            cell["valid_boundary_pixels"] += frame["boundary"]
            cell["valid_dual_pixels"] += frame["valid_dual"]
            cell["label_pixels"] += frame["pixels"]
            cell["relative_depth_histogram"] = [
                a+b for a,b in zip(cell["relative_depth_histogram"],frame["relative_depth_histogram"])
            ]
            for focus, lens in enumerate(frame["K16"]):
                cell["aperture_configurations"][f"focus{focus}/f{lens['fstop']}"] += 1
                for key,value in (
                    ("hidden_aperture_rays",lens["hidden_rays"]),
                    ("recoverable_aperture_rays",lens["recoverable_rays"]),
                    ("never_visible_rays",lens["never_visible_rays"]),
                ):
                    cell[key] += value
    return dict(cells)


def optical_selection(units, count=80):
    """Choose from validated training geometry by fixed physical difficulty scores."""
    candidates=[]
    for unit in units:
        if unit["partition"] != "training":
            continue
        aperture=[case for frame in unit["scene_supervision"] for case in frame["K16"]]
        boundary=sum(f["boundary"] for f in unit["scene_supervision"])
        scores = {
            "foreground_focus": sum(f["pixels"] for f in unit["scene_supervision"]) - boundary,
            "background_focus_with_disocclusion": sum(case["recoverable_rays"] for case in aperture),
            "thin_and_complex_edge_transmission": boundary if unit["category"] in {"thin_structure","curved_self_occlusion","complex_layout"} else 0,
            "multiple_layers_and_local_occlusion": boundary if unit["depth_bands"] >= 3 else 0,
            "highlight_and_bokeh": sum(f["saturated_boundary"] for f in unit["radiance_audit"]),
        }
        candidates.append((unit,scores))
    if len(candidates) < count:
        raise ValueError("fewer eligible training geometry scenes than the optical target")
    chosen, selected = [], set()
    for index in range(count):
        difficulty = OPTICAL_TAGS[index % len(OPTICAL_TAGS)]
        available = [(u,s) for u,s in candidates if u["scene_id"] not in selected and s[difficulty] > 0]
        if not available:
            raise ValueError(f"insufficient training geometry for optical difficulty {difficulty}")
        unit,scores = min(available,key=lambda item:(-item[1][difficulty],item[0]["scene_id"]))
        selected.add(unit["scene_id"])
        chosen.append(dict(scene_id=unit["scene_id"],primary_difficulty=difficulty,
                           difficulties=[name for name in OPTICAL_TAGS if scores[name]>0],scores=scores))
    return chosen


def batch(plan_path, checked_archive_bytes, max_new_scenes=None):
    base = json.loads(plan_path.read_text())
    folder = plan_path.parent
    expected = json.loads((folder/"sha256.json").read_text())
    if expected[plan_path.stem] != sha256(plan_path):
        raise ValueError("sealed plan checksum was changed")
    mode = plan_path.stem
    if mode not in PRODUCTION_MODES:
        raise ValueError("not a production B partition")
    state = json.loads((DATA/"manifest.json").read_text())
    if state.get("generator_version") != base["generator_version"]:
        raise ValueError("production requires the same formally accepted generator version as the sealed plan")
    units = DATA/"units"/mode
    units.mkdir(parents=True, exist_ok=True)
    failures_path = units/"failures.json"
    failures = json.loads(failures_path.read_text()) if failures_path.is_file() else []
    counts = Counter(row["category"] for row in failures if row.get("counts_toward_pause",True))
    noted={row["scene_id"] for row in failures}
    attempted = 0
    for spec in base["scenes"]:
        unit = units/spec["scene_id"]
        if (unit/"manifest.json").is_file() or spec["scene_id"] in noted:
            continue
        if counts[spec["category"]] >= 2:
            failures.append(dict(scene_id=spec["scene_id"],category=spec["category"],
                                 reason="two failed scenes in category",counts_toward_pause=False))
            continue
        if max_new_scenes is not None and attempted >= max_new_scenes:
            break
        attempted += 1
        try:
            if unit.exists():
                raise RuntimeError("prior incomplete attempt was preserved")
            produce(unit,spec,base["generator_version"],checked_archive_bytes)
        except (RuntimeError,ValueError,OSError) as exc:
            metadata = dict(scene_id=spec["scene_id"],category=spec["category"],reason=str(exc))
            failures.append(metadata)
            if "insufficient disk space" in str(exc):
                write_json(failures_path,failures)
                raise
            counts[spec["category"]] += 1
        write_json(failures_path,failures)
    write_json(failures_path,failures)


def archive_status():
    units = []
    for file in sorted((DATA/"units").glob("*/*/manifest.json")):
        unit = json.loads(file.read_text())
        unit.update(manifest=str(file.relative_to(DATA)), manifest_sha256=sha256(file))
        units.append(unit)
    optics = []
    for file in sorted((DATA/"optical").glob("*/manifest.json")):
        item = json.loads(file.read_text())
        item.update(manifest=str(file.relative_to(DATA)), manifest_sha256=sha256(file))
        optics.append(item)
    state_path = DATA/"manifest.json"
    state = json.loads(state_path.read_text())
    counts = Counter(item["partition"] for item in units)
    state["counts"] = dict(
        train_geometry=counts["training"], development=counts["development"],
        test=counts["test_unseen_assets"]+counts["test_unseen_occlusions"], train_optical=len(optics),
    )
    state["units"] = [
        {name: item[name] for name in ("scene_id","partition","generator_version","assets",
                                       "bytes","manifest","manifest_sha256")} for item in units
    ]
    state["optical_units"] = optics
    state["quota_actual"] = combine_geometry(units)
    state["verified_geometry_and_optical_bytes"] = sum(item["bytes"] for item in [*units,*optics])
    split_failures = {}
    for path in sorted((DATA/"units").glob("*/failures.json")):
        split_failures[path.parent.name] = dict(path=str(path.relative_to(DATA)),sha256=sha256(path),
                                               count=len(json.loads(path.read_text())))
    state["generation_failures"] = split_failures
    manifest_digest = write_json(state_path,state)
    record = json.loads(RECORD.read_text())
    record["manifest_sha256"] = manifest_digest
    record["completed"] = state["counts"]
    record["target_gap"] = {mode:record["targets"][mode]-count for mode,count in state["counts"].items()}
    record["verified_geometry_and_optical_bytes"] = state["verified_geometry_and_optical_bytes"]
    record["failures"] = split_failures
    record["optical_difficulty_counts"] = dict(Counter(tag for item in optics for tag in item["difficulties"]))
    write_json(RECORD,record)
    return units


def produce_optical(estimate):
    files = sorted((DATA/"units/training").glob("*/manifest.json"))
    candidates={file.parent.name:json.loads(file.read_text()) for file in files}
    target = DATA/"optical"
    target.mkdir(exist_ok=True)
    selection = target/"selection.json"
    if not selection.is_file():
        chosen = optical_selection(list(candidates.values()))
        write_json(selection,chosen)
    chosen = json.loads(selection.read_text())
    for item in chosen:
        scene_id = item["scene_id"]
        unit = target/scene_id
        if (unit/"manifest.json").is_file():
            continue
        if unit.exists():
            raise RuntimeError(f"optical attempt requires review after failure: {scene_id}")
        reserve(DATA,estimate)
        source = DATA/"units/training"/scene_id
        unit.mkdir()
        labels = unit/"labels"
        labels.mkdir()
        for name in ("scene.blend","scene-spec.json"):
            (unit/name).symlink_to(source/name)
        for frame in (source/"labels").glob("*-pinhole.npz"):
            (labels/frame.name).symlink_to(frame)
        run(blender("route3_v2data_B_render.py","--scene",unit,"--optical-only"),unit/"render.log",uses_gpu=True)
        audit = json.loads((labels/"optical-audit.json").read_text())
        if len(audit["optical"]) != 52:
            raise ValueError(f"incomplete optical configurations: {scene_id}")
        render_products = sorted(p for p in labels.glob("*.npz") if not p.is_symlink())
        outputs = [dict(path=str(p.relative_to(DATA)),bytes=p.stat().st_size,sha256=sha256(p)) for p in render_products]
        write_json(unit/"manifest.json",dict(
            **item, generator_version=candidates[scene_id]["generator_version"],
            partition="training",geometry_manifest_sha256=sha256(source/"manifest.json"),
            selection_sha256=sha256(selection),bytes=sum(file["bytes"] for file in outputs),files=outputs,
        ))
        archive_status()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--sealed-plan", type=Path)
    mode.add_argument("--optical", action="store_true")
    mode.add_argument("--summarize", action="store_true")
    parser.add_argument("--measured-peak-unit-bytes", type=int)
    parser.add_argument("--max-new-scenes", type=int, help="Limit this launch to the first measured production batch")
    args = parser.parse_args()
    if args.summarize:
        archive_status()
        return
    if not args.measured_peak_unit_bytes or args.measured_peak_unit_bytes <= 0:
        parser.error("a measured positive unit size from the pilot is required")
    if args.max_new_scenes is not None and (args.sealed_plan is None or args.max_new_scenes <= 0):
        parser.error("--max-new-scenes requires --sealed-plan and a positive limit")
    if args.optical:
        produce_optical(args.measured_peak_unit_bytes)
    else:
        batch(args.sealed_plan,args.measured_peak_unit_bytes,args.max_new_scenes)
        archive_status()


if __name__ == "__main__":
    main()
