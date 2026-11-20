"""Review a deterministic batch of real 3D-FRONT rooms from the local archives."""
import argparse
from collections import Counter
from contextlib import closing
from itertools import islice
import time
import json
import sys

from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parent))

from route3_v2data_B_front import DATA, TERMS, write_json
from route3_v2data_B_layout import extract_candidate, room_candidates


V2 = DATA/"v2"


def check(room,architecture_only=False):
    from route3_v2data_B_layout import build_room, review_layout
    from route3_v2data_B_topology import audit_mesh
    if architecture_only:
        layout = json.loads((room/"room.json").read_text())
        names = {row["uid"].partition(".")[0] for row in layout["architecture"]}
        review_layout(room/"room.blend",room/"architecture-review",mesh_audit=audit_mesh,architecture_meshes=names)
    else:
        build_room(room)
        review_layout(room/"room.blend",room/"opaque-review",mesh_audit=audit_mesh)


def stage(count, supplement=False):
    from route3_v2data_B_batch import GIB, blender, run, sha256

    output = V2/"source-candidates/3d-front.json"
    workspace = V2/"source-layouts"
    output.parent.mkdir(parents=True,exist_ok=True)
    workspace.mkdir(parents=True,exist_ok=True)
    existing = json.loads(output.read_text()) if output.is_file() else {}
    candidates = {row["selection_index"]:row for row in existing.get("candidates",[])}
    if supplement:
        seen = {row["asset_id"] for row in candidates.values()}
        start = max(candidates,default=-1)+1
    else:
        seen = set()
        start = 0
    with closing(room_candidates(partial=supplement)) as selected:
        if supplement:
            selected = (item for item in selected if f"{item[1]}/{item[2][0]['instanceid']}" not in seen)
        for index,candidate in enumerate(islice(selected,max(0,count-start)),start=start):
            if index in candidates:
                continue
            room = workspace/f"room-{index:03d}"
            row = dict(source="3D-FRONT",selection_index=index,category="layout",
                       author="3D-FRONT and 3D-FUTURE dataset authors",
                       license="3D-FRONT / 3D-FUTURE Data Sets Use License Agreements",
                       terms=TERMS["3D-FRONT"],furniture_terms=TERMS["3D-FUTURE"],
                       license_terms_version={"3D-FRONT":"2020-06-18","3D-FUTURE":"2020-03-28"},
                       page="https://tianchi.aliyun.com/specials/promotion/alibaba-3d-scene-dataset",
                       download_channel="authorized local archives",can_redistribute=False)
            try:
                if not (room/"room.json").is_file():
                    if room.is_dir():
                        room.rename(room.with_name(f"{room.name}-interrupted-{time.time_ns()}"))
                    extract_candidate(room,candidate)
                spec = json.loads((room/"room.json").read_text())
                row.update(asset_id=f"{spec['layout_id']}/{spec['room_id']}",archive_sha256=spec["source_sha256"],
                           omitted_unresolved_references=len(spec.get("omitted_unresolved_references",[])))
                review = room/"opaque-review/mesh-review.json"
                if not review.is_file():
                    run(blender("route3_v2data_B_v2layouts.py","--room",room,threads=2),
                        room/"build-and-review.log",cpu_limit_bytes=4*GIB)
                result = json.loads(review.read_text())
                row.update(status=result["status"],mesh_review=str(review.relative_to(V2)))
                if result["status"] == "passed":
                    render = review.with_name("layout.blend")
                    row.update(mesh_path=str(render.relative_to(V2)),geometry_sha256=sha256(render),
                               face_count=result["faces"],components=result["accepted"])
                else:
                    row["reason"] = "no opaque and unambiguous mesh component survived"
            except (RuntimeError,ValueError,OSError,KeyError) as exc:
                row.setdefault("asset_id",f"unresolved-{index:03d}")
                row.update(status="excluded",reason=str(exc))
            candidates[index]=row
            write_json(output,dict(source="3D-FRONT",status="reviewing",candidates=list(candidates.values())))
            print("3D-FRONT",len(candidates),"/",count,dict(Counter(r["status"] for r in candidates.values())),flush=True)
    write_json(output,dict(source="3D-FRONT",status="complete",catalog_complete=True,
                           candidates=list(candidates.values())))


def isolate_layout_geometry():
    from route3_v2data_B_batch import GIB, blender, run, sha256

    output = V2/"source-candidates/3d-front.json"
    state = json.loads(output.read_text())
    if state.get("status") not in {"complete","checking_partition_dependencies"}:
        raise ValueError("finish the 3D-FRONT room audits before isolating geometry")
    state.update(status="checking_partition_dependencies",catalog_complete=False)
    write_json(output,state)
    for row in state["candidates"]:
        if row["status"] != "passed" or row.get("asset_role")=="architectural_layout":
            continue
        room=V2/"source-layouts"/f"room-{row['selection_index']:03d}"
        review=room/"architecture-review"/"mesh-review.json"
        try:
            if not review.is_file():
                run(blender("route3_v2data_B_v2layouts.py","--room",room,"--architecture-only",threads=2),
                    room/"architecture-review.log",cpu_limit_bytes=4*GIB)
            data=json.loads(review.read_text())
            row.update(asset_role="architectural_layout",full_room_review=row["mesh_review"],
                       mesh_review=str(review.relative_to(V2)),status=data["status"])
            if data["status"] == "passed":
                model=review.with_name("layout.blend")
                row.update(mesh_path=str(model.relative_to(V2)),geometry_sha256=sha256(model),
                           face_count=data["faces"],components=data["accepted"])
            else:
                row["reason"]="no architectural component survived the independent layout audit"
        except (RuntimeError,ValueError,OSError) as exc:
            row.update(status="excluded",reason=str(exc))
        write_json(output,state)
    state.update(status="complete",catalog_complete=True)
    write_json(output,state)
    print("isolated layout geometry",dict(Counter(r["status"] for r in state["candidates"])),flush=True)


if __name__ == "__main__":
    arguments = sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else None
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count",type=int,default=40)
    parser.add_argument("--room",type=Path)
    parser.add_argument("--supplement",action="store_true")
    parser.add_argument("--architecture-only",action="store_true")
    opts=parser.parse_args(arguments)
    if opts.room:
        check(opts.room,opts.architecture_only)
    elif opts.architecture_only:
        isolate_layout_geometry()
    else:
        stage(opts.count,opts.supplement)
