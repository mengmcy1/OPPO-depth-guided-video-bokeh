"""Review a deterministic batch of real 3D-FRONT rooms from the local archives."""
import argparse
from collections import Counter
import json
import sys

from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parent))

from route3_v2data_B_front import DATA, TERMS, write_json
from route3_v2data_B_layout import extract_room


V2 = DATA/"v2"


def check(room):
    from route3_v2data_B_layout import build_room, review_layout
    from route3_v2data_B_topology import audit_mesh
    build_room(room)
    review_layout(room/"room.blend",room/"opaque-review",mesh_audit=audit_mesh)


def stage(count):
    from route3_v2data_B_batch import GIB, blender, run, sha256

    output = V2/"source-candidates/3d-front.json"
    workspace = V2/"source-layouts"
    output.parent.mkdir(parents=True,exist_ok=True)
    workspace.mkdir(parents=True,exist_ok=True)
    existing = json.loads(output.read_text()) if output.is_file() else {}
    candidates = {row["selection_index"]:row for row in existing.get("candidates",[])}
    for index in range(count):
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
                extract_room(room,index)
            spec = json.loads((room/"room.json").read_text())
            row.update(asset_id=f"{spec['layout_id']}/{spec['room_id']}",archive_sha256=spec["source_sha256"])
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


if __name__ == "__main__":
    arguments = sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else None
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count",type=int,default=40)
    parser.add_argument("--room",type=Path)
    opts=parser.parse_args(arguments)
    if opts.room:
        check(opts.room)
    else:
        stage(opts.count)
