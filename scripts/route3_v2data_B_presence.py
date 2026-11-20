"""Count designated foreground surface identities in saved B geometry labels."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from route3_v2data_B_front import write_json
from route3_v2data_B_inputs import FORMAL_DATA
from route3_ovg_subpixel import TRAIN_FRAMES


MIN_CENTER_HITS = 1
MID_FRAME = 30
MIN_VISIBLE_FRAMES = 9
OPTICAL_FRAMES = (1, 30, 60)


def appearance_checks(actors, frames):
    """Summarize already counted native pinhole hits per designated actor."""
    if MID_FRAME not in frames or len(frames) != 13:
        raise ValueError("foreground admission requires the 13 native label frames")
    failures, summary = [], {}
    for name, counts in actors.items():
        visible = [frame for frame, hits in zip(frames, counts) if hits >= MIN_CENTER_HITS]
        absent = [frame for frame in frames if frame not in visible]
        passed = MID_FRAME in visible and len(visible) >= MIN_VISIBLE_FRAMES
        summary[name] = dict(visible_frames=len(visible), absent_frames=absent,
                             f030_center_hits=int(counts[frames.index(MID_FRAME)]), passed=passed)
        if not passed:
            failures.append(dict(foreground=name, **summary[name]))
    return summary, failures


def reference_foregrounds(presence):
    return {f"f{row['frame']:03d}":{
        key:dict(visible=actor["center_hits"]>=MIN_CENTER_HITS, center_hits=actor["center_hits"])
        for key,actor in row["foreground"].items()
    } for row in presence["frames"] if row["frame"] in OPTICAL_FRAMES}


def attach_optical_visibility(audit_path, presence_path):
    audit = json.loads(audit_path.read_text())
    presence = json.loads(presence_path.read_text())
    audit["foreground_visibility"] = reference_foregrounds(presence)
    write_json(audit_path, audit)


def designated_foreground(spec, names):
    """Treat the mesh parts of one imported actor as one designated object."""
    category = spec["category"]
    imported = spec.get("occluder_assets",[])
    if category=="thin_structure":
        result = {"foreground_structure":["foreground_structure"]}
    elif imported:
        result = {}
        for index in range(len(imported)):
            name = f"foreground_import_{index:02d}"
            result[name] = [item for item in names if item.startswith(name+"_")]
    elif category=="curved_self_occlusion":
        name = "foreground_sloped" if spec.get("sloped_surface") else "foreground_curved"
        result = {name:[name]}
    elif category=="complex_layout" and spec.get("model_path"):
        result = {"foreground_import":[item for item in names if item.startswith("foreground_import_")]}
    else:
        result = {"foreground_main":["foreground_main"]}
    for layer in range(max(2,1+len(imported)),spec["depth_bands"]):
        name = f"foreground_middle_{layer}"
        result[name] = [name]
    return result


def foreground_presence(unit, spec, frames=TRAIN_FRAMES):
    labels = unit/"labels"
    with np.load(labels/"scene-geometry.npz") as geom:
        foreground = set(geom["foreground_object_ids"].tolist())
        objects = {name:int(index) for name,index in zip(geom["object_names"],geom["object_ids"])
                   if index in foreground}
    groups = designated_foreground(spec,objects)
    own_ids = {name:[objects[part] for part in parts if part in objects] for name,parts in groups.items()}
    by_frame = []
    for frame in frames:
        path = labels/f"f{frame:03d}-pinhole.npz"
        with np.load(path) as truth:
            pixels = np.bincount(truth["first_hit_object"][truth["valid"]].astype(np.int64))
            centers = np.bincount(truth["object_id"][truth["center_visible"]].astype(np.int64))
        actors = {}
        for name,identities in own_ids.items():
            pixels_count = sum(int(pixels[i]) for i in identities if i<len(pixels))
            hit_count = sum(int(centers[i]) for i in identities if i<len(centers))
            actors[name] = dict(surface_pixels=pixels_count,center_hits=hit_count)
        by_frame.append(dict(frame=frame,foreground=actors))
    summary = {
        name: {f"minimum_{key}": min(row["foreground"][name][key] for row in by_frame)
               for key in ("surface_pixels", "center_hits")}
        for name in groups
    }
    appearances, failures = appearance_checks(
        {name:[item["foreground"][name]["center_hits"] for item in by_frame] for name in groups}, list(frames)
    )
    for name, row in appearances.items():
        summary[name].update(row)
    placement = labels/"multilayer-placement.json"
    if placement.is_file():
        report = json.loads(placement.read_text())
        for layer in report["layers"]:
            summary[layer["name"]].update(
                midpoint_projected_overlap=layer["middle_overlap"],
                minimum_projected_overlap=min(row["overlap"] for row in layer["frames"]),
                maximum_projected_overlap=max(row["overlap"] for row in layer["frames"]),
            )
    return dict(scene_id=spec["scene_id"],partition=spec["partition"],category=spec["category"],
                foreground_minima=summary,
                minimum_center_hits=MIN_CENTER_HITS,foreground_surface_ids=own_ids,
                frames=by_frame,failed_checks=failures,passed=not failures)


def require_presence(unit,spec):
    audit=foreground_presence(unit,spec)
    write_json(unit/"labels/foreground-presence.json",audit)
    if not audit["passed"]:
        failure=audit["failed_checks"][0]
        raise ValueError(f"designated foreground {failure['foreground']} has {failure['f030_center_hits']} f030 "
                         f"center hits and appears in {failure['visible_frames']}/13 frames")
    reference = unit/"labels/radiance-audit.json"
    if reference.is_file() and json.loads(reference.read_text()).get("optical"):
        attach_optical_visibility(reference, unit/"labels/foreground-presence.json")
    return audit


def audit_archive(output):
    state = json.loads((FORMAL_DATA/"manifest.json").read_text())
    rows=[]
    counts={}
    for item in state["units"]:
        unit=(FORMAL_DATA/item["manifest"]).parent
        spec=json.loads((unit/"scene-spec.json").read_text())
        verdict=foreground_presence(unit,spec)
        rows.append(verdict)
        tally=counts.setdefault(spec["partition"],Counter())
        tally["passed" if verdict["passed"] else "failed"]+=1
    write_json(output,dict(counts_by_partition=counts,
                           failed_outside_curved=any(not v["passed"] and v["category"]!="curved_self_occlusion" for v in rows),
                           units=rows))
    print({k:dict(v) for k,v in counts.items()},flush=True)
    for row in rows:
        if not row["passed"] and row["category"]!="curved_self_occlusion":
            print(row["partition"],row["scene_id"],row["category"],row["failed_checks"][0],
                  len(row["failed_checks"]),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-archive",type=Path,required=True)
    audit_archive(parser.parse_args().audit_archive)
