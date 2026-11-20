"""Stage a deterministic furniture sample from the local verified 3D-FUTURE ZIP."""
import argparse
import hashlib
import json
import shutil
from zipfile import ZipFile

from route3_v2data_B_batch import sha256
from route3_v2data_B_front import ARCHIVES, DATA, RESERVE_BYTES, TERMS, write_json
from route3_v2data_B_layout import group_files
from route3_v2data_B_sources import stable_limit


V2 = DATA/"v2"


def files_from_archive(archive, group, root):
    root.mkdir(parents=True,exist_ok=True)
    files = []
    for name in ("raw_model.obj","model.mtl","texture.png"):
        member = group[name]
        dest = root/name
        if not dest.is_file():
            temp = dest.with_suffix(dest.suffix+".part")
            with archive.open(member) as source, temp.open("wb") as output:
                shutil.copyfileobj(source,output)
            if temp.stat().st_size != member.file_size:
                raise ValueError("unexpected extracted 3D-FUTURE file size")
            temp.replace(dest)
        if dest.stat().st_size != member.file_size:
            raise ValueError("cached 3D-FUTURE file size changed")
        files.append(dict(path=str(dest.relative_to(V2)),bytes=member.file_size,sha256=sha256(dest)))
    return files


def declared_opaque(value, material):
    if any(name in material.casefold() for name in ("glass","acrylic","transparent")):
        return False
    for line in value.splitlines():
        pieces = line.partition("#")[0].split()
        if not pieces:
            continue
        key,*args = pieces
        if key in {"map_d","map_Tr"}:
            return False
        if key == "d" and float(args[-1]) < 1-1e-6:
            return False
        if key == "Tr" and float(args[-1]) > 1e-6:
            return False
        if key == "illum" and int(args[0]) in {4,6,7,9}:
            return False
    return True


def stage(limit):
    source = json.loads((DATA/"3d-front-status.json").read_text())
    assert source["sha256_verified"]
    container = DATA/"sources/3d-front"/ARCHIVES[2]
    receipt = next(row for row in source["downloaded_files"] if row["file"]==ARCHIVES[2])
    if not receipt["verified"] or container.stat().st_size != receipt["bytes"]:
        raise ValueError("missing verified archive or changed archive size")
    output = V2/"source-candidates/3d-future.json"
    output.parent.mkdir(parents=True,exist_ok=True)
    cached = json.loads(output.read_text()) if output.is_file() else {}
    have = {row["asset_id"]:row for row in cached.get("candidates",[])}
    with ZipFile(container) as archive:
        catalog = group_files(archive,{".obj",".png",".mtl"})
        metadata = {row["model_id"]:row for row in json.loads(archive.read("3D-FUTURE-model/model_info.json"))}
        expected = {"raw_model.obj","model.mtl","texture.png"}
        ordered = stable_limit(catalog,limit,lambda uid:f"3D-FUTURE:{uid}")
        to_stage = [uid for uid in ordered if uid not in have and expected <= catalog[uid].keys()]
        size = sum(catalog[uid][name].file_size for uid in to_stage for name in expected)
        if shutil.disk_usage(output.parent).free < RESERVE_BYTES+size:
            raise ValueError("insufficient disk space for staging and 60 GiB reserve")
        state = dict(source="3D-FUTURE",status="extracting",catalog_count=len(catalog),
                     staged_selection=len(ordered),archive_sha256=receipt["sha256"])
        for index,uid in enumerate(ordered):
            if uid in have:
                continue
            model = metadata.get(uid,{})
            row = dict(source="3D-FUTURE",asset_id=uid,category="mesh",
                       author="3D-FUTURE dataset authors",license="3D-FUTURE Data Sets Use License Agreement",
                       terms=TERMS["3D-FUTURE"],terms_version="2020-03-28",can_redistribute=False,
                       page="https://tianchi.aliyun.com/specials/promotion/alibaba-3d-scene-dataset",
                       download_channel="authorized 3D-FUTURE model archive",
                       taxonomy=[model.get("super-category",""),model.get("category","")],
                       source_material=model.get("material"),archive_sha256=receipt["sha256"])
            if not expected <= catalog[uid].keys() or not model:
                row.update(status="excluded",reason="missing official model component or metadata")
            elif not declared_opaque(archive.read(catalog[uid]["model.mtl"]).decode(errors="replace"),model.get("material") or ""):
                row.update(status="excluded",reason="transparent or refractive material declaration")
            else:
                key = hashlib.sha256(f"3D-FUTURE:{uid}".encode()).hexdigest()[:20]
                row.update(files=files_from_archive(archive,catalog[uid],V2/"source-files"/key),
                           status="pending_mesh_checks")
            have[uid] = row
            if index%12==11:
                write_json(output,dict(state,candidates=list(have.values())))
        write_json(output,dict(state,status="complete",catalog_complete=True,candidates=list(have.values())))
    print("3D-FUTURE staged",len(have),"/",len(catalog),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit",type=int,default=480)
    stage(parser.parse_args().limit)
