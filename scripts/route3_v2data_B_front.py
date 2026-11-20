"""Download the three authorized 3D-FRONT archives and verify Drive SHA-256."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/route3-v2data-2026-11-20/B"
RECORD = ROOT / "03_实验记录/records/route3-v2data-B-2026-11-20.json"
ARCHIVES = ("3D-FRONT.zip", "3D-FRONT-texture.zip", "3D-FUTURE-model.zip")
RESERVE_BYTES = 60 * 2**30
TERMS = {
    "3D-FRONT": "https://gw.alicdn.com/bao/uploaded/TB1ZJUfK.z1gK0jSZLeXXb9kVXa.pdf",
    "3D-FUTURE": "https://terms.aliyun.com/legal-agreement/terms/suit_bu1_ali_cloud/suit_bu1_ali_cloud202004171628_60052.html",
}


def write_json(path, value):
    temp = path.with_suffix(".part")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record_status(status, **updates):
    status_path = DATA / "3d-front-status.json"
    status_record = json.loads(status_path.read_text())
    status_record.update(updates, status=status)
    status_hash = write_json(status_path, status_record)
    manifest_path = DATA / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    source = next(row for row in manifest["sources"] if row["name"] == "3D-FRONT")
    source.update(
        status=status,
        license="3D-FRONT / 3D-FUTURE Data Sets Use License Agreements",
        sha256=status_hash,
        redistribute=False,
    )
    manifest_hash = write_json(manifest_path, manifest)
    record = json.loads(RECORD.read_text())
    summary = {
        key: value for key, value in status_record.items()
        if key not in ("authorization_terms_original", "remote_archive_inventory")
    }
    summary["authorization_terms_recorded"] = bool(status_record.get("authorization_terms_original"))
    summary["status_record_sha256"] = status_hash
    record.update(manifest_sha256=manifest_hash, **{"3d_front": summary})
    write_json(RECORD, record)


def download(remote, folder):
    listing = json.loads((DATA / "3d-front-target-listing.json").read_text())["files"]
    rows = {row["Path"]: row for row in listing}
    expected = []
    for name in ARCHIVES:
        row = rows[name]
        expected_hash = (row.get("Hashes") or {}).get("sha256")
        if not expected_hash or row["Size"] <= 0:
            raise ValueError(f"Missing source size or SHA-256 for {name}")
        expected.append({
            "file": name,
            "bytes": row["Size"],
            "drive_sha256": expected_hash,
        })

    dest = DATA / "sources" / "3d-front"
    dest.mkdir(parents=True, exist_ok=True)
    remaining = sum(row["bytes"] for row in expected if not (dest / row["file"]).is_file())
    free = shutil.disk_usage(dest).free
    if free < RESERVE_BYTES + remaining:
        record_status("磁盘空间不足", reason="可用空间小于本批预估加 60 GiB。")
        raise RuntimeError("Disk reserve would be exceeded")

    inventory = DATA / "3d-front-download-files.txt"
    inventory.write_text("\n".join(ARCHIVES) + "\n")
    record_status(
        "正在下载",
        reason="已确认授权目录含场景、家具与纹理三个归档。",
        remote_archive_inventory=expected,
        expected_bytes=sum(row["bytes"] for row in expected),
        free_bytes_at_download_start=free,
        license_agreements=TERMS,
        post_expiry_use_restrictions=(
            "邮件确认过期后账号无法通过该链接继续访问，需要续办。"
            "邮件未规定本地已下载副本的期满处理，使用继续受适用的数据协议约束。"
        ),
    )
    result = subprocess.Popen(
        ["rclone", "copy", f"{remote}:{folder}", str(dest), "--drive-shared-with-me",
         "--files-from-raw", str(inventory), "--immutable", "--checksum",
         "--transfers", "3", "--checkers", "3", "--contimeout", "20s", "--timeout", "2m",
         "--retries", "3", "--low-level-retries", "4", "--stats", "20s",
         "--stats-one-line", "--stats-log-level", "NOTICE", "--log-level", "NOTICE"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    for line in result.stdout:
        if "ETA" in line and "NOTICE" in line:
            print(line.strip(), flush=True)
    if result.wait():
        record_status("下载受阻", reason=f"rclone copy 失败，退出码 {result.returncode}。")
        raise RuntimeError(f"rclone copy failed with exit code {result.returncode}")

    completed = []
    for row in expected:
        path = dest / row["file"]
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        verified = path.stat().st_size == row["bytes"] and digest == row["drive_sha256"]
        completed.append(dict(row, sha256=digest, verified=verified))
        if not verified:
            record_status("SHA-256 校验失败", reason=row["file"], downloaded_files=completed)
            raise ValueError(f"Source checksum mismatch: {row['file']}")
        print("SHA256_OK", row["file"], row["bytes"], flush=True)
    record_status(
        "下载完成并校验",
        reason="三个归档的字节数与 SHA-256 均与已授权 Google Drive 文件一致。",
        downloaded_files=completed,
        sha256_verified=True,
    )


def copy_range(remote_path, offset, size, path):
    if path.is_file() and path.stat().st_size == size:
        return
    pending = path.with_suffix(".part")
    command = [
        "rclone","cat",remote_path,"--drive-shared-with-me","--offset",str(offset),"--count",str(size),
        "--retries","1","--low-level-retries","2","--contimeout","15s","--timeout","90s",
        "--log-level","ERROR",
    ]
    for attempt in range(12):
        try:
            with pending.open("wb") as stream:
                result = subprocess.run(command,stdout=stream,stderr=subprocess.PIPE,timeout=420)
            if result.returncode == 0 and pending.stat().st_size == size:
                pending.replace(path)
                return
        except subprocess.TimeoutExpired:
            pass
        time.sleep(min(45, 4+attempt*4))
    raise RuntimeError(f"Range {offset} failed after bounded transport retries")


def download_ranges(remote, folder, names):
    listing = json.loads((DATA/"3d-front-target-listing.json").read_text())["files"]
    inventory = {row["Path"]: row for row in listing}
    dest = DATA/"sources"/"3d-front"
    dest.mkdir(parents=True,exist_ok=True)
    cache = DATA/"sources"/"3d-front-range-cache"
    cache.mkdir(exist_ok=True)
    for name in names:
        size = inventory[name]["Size"]
        expected_hash = inventory[name]["Hashes"]["sha256"]
        target = dest/name
        if target.is_file():
            with target.open("rb") as stream:
                digest = hashlib.file_digest(stream,"sha256").hexdigest()
            if target.stat().st_size == size and digest == expected_hash:
                continue
            raise ValueError(f"Existing archive fails SHA-256: {name}")
        staging = cache/name
        staging.mkdir(exist_ok=True)
        committed = sum(f.stat().st_size for f in staging.glob("*.range"))
        if shutil.disk_usage(cache).free < RESERVE_BYTES + size*2-committed:
            raise RuntimeError(f"Disk reserve would be exceeded for ranged transfer: {name}")
        chunk = 64*2**20
        pieces = [
            (offset,min(chunk,size-offset),staging/f"{offset:012d}.range")
            for offset in range(0,size,chunk)
        ]
        remote_path = f"{remote}:{folder}/{name}"
        with ThreadPoolExecutor(max_workers=4) as pool:
            waiting = [pool.submit(copy_range,remote_path,offset,length,path) for offset,length,path in pieces]
            for index,future in enumerate(waiting,1):
                future.result()
                if index % 4 == 0 or index == len(pieces):
                    print("RANGE_PROGRESS",name,index,len(pieces),flush=True)
        digest = hashlib.sha256()
        assembled = staging/"complete-archive.part"
        with assembled.open("wb") as output:
            for _offset,_length,path in pieces:
                with path.open("rb") as stream:
                    while block := stream.read(2**20):
                        output.write(block)
                        digest.update(block)
        if assembled.stat().st_size != size or digest.hexdigest() != expected_hash:
            raise ValueError(f"Ranged transfer SHA-256 mismatch: {name}")
        if not target.exists():
            assembled.replace(target)
        else:
            with target.open("rb") as stream:
                if hashlib.file_digest(stream,"sha256").hexdigest() != expected_hash:
                    raise ValueError(f"Concurrent full transfer produced an unexpected checksum: {name}")
            assembled.unlink()
        # A verified final archive substitutes for these rebuildable transport pieces.
        for _offset,_length,path in pieces:
            path.unlink()
        status_path = DATA/"3d-front-status.json"
        current = json.loads(status_path.read_text())
        confirmed = {row["file"]:row for row in current.get("downloaded_files",[]) if row.get("verified")}
        confirmed[name] = dict(file=name,bytes=size,sha256=expected_hash,drive_sha256=expected_hash,verified=True)
        finished = set(confirmed) == set(ARCHIVES)
        record_status(
            "下载完成并校验" if finished else "正在下载，部分归档已校验",
            reason="已按 Google Drive 提供的 SHA-256 核验完成的原始归档。",
            downloaded_files=list(confirmed.values()),sha256_verified=finished,
        )
        print("SHA256_OK",name,size,flush=True)


def index_layouts():
    """Index verified layouts while the separate model and texture dependencies download."""
    name = "3D-FRONT.zip"
    raw = DATA/"sources"/"3d-front"/name
    status = json.loads((DATA/"3d-front-status.json").read_text())
    verified = next((row for row in status["downloaded_files"] if row["file"]==name and row["verified"]),None)
    if verified is None or raw.stat().st_size!=verified["bytes"]:
        raise ValueError("layout archive must have a verified remote SHA-256")
    seen = {}
    rows, excluded = [], []
    room_counts=Counter()
    with ZipFile(raw) as source:
        for member in sorted(source.namelist()):
            if not member.lower().endswith(".json"):
                continue
            try:
                payload=source.read(member)
                layout=json.loads(payload)
                rooms=layout["scene"]["room"]
                if not rooms or any(not isinstance(room.get("children"),list) for room in rooms):
                    raise ValueError("missing layout rooms or placement rows")
                room_types=Counter(room.get("type","unknown") for room in rooms)
                refs = {child.get("ref") for room in rooms for child in room["children"]}
                furniture = sorted({
                    item["jid"] for item in layout["furniture"] if item.get("jid") and item["uid"] in refs
                })
            except (KeyError,TypeError,ValueError) as exc:
                excluded.append(dict(member=member,reason=type(exc).__name__))
                continue
            digest=hashlib.sha256(payload).hexdigest()
            if digest in seen:
                excluded.append(dict(member=member,reason="exact duplicate layout",canonical_id=seen[digest]))
                continue
            asset_id=Path(member).stem
            seen[digest]=asset_id
            room_counts.update(room_types)
            rows.append(dict(
                source="3D-FRONT",asset_id=asset_id,author="3D-FRONT dataset authors and Alibaba",
                category="layout",source_version=layout.get("version"),member=member,sha256=digest,
                license="3D-FRONT Data Sets Use License Agreement",terms=TERMS["3D-FRONT"],
                terms_version="2020-06-18",can_redistribute=False,
                download_channel="approved Google Drive folder / rclone",
                room_counts=dict(room_types),furniture_ids=furniture,
                mesh_count=len(layout.get("mesh",[])),material_count=len(layout.get("material",[])),
                status="pending_furniture_texture_and_geometry_review",
            ))
    index=DATA/"3d-front-layout-index.json"
    digest=write_json(index,dict(archive_sha256=verified["sha256"],rows=rows,exclusions=excluded,
                                room_counts=dict(room_counts)))
    record_status(status["status"],layout_index=dict(path=index.name,sha256=digest,
                    candidate_layouts=len(rows),excluded=len(excluded)))
    print("LAYOUT_INDEX",len(rows),len(excluded),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", default="3dfront", help="Name of the configured read-only Drive remote")
    parser.add_argument("--folder", default="3D-FRONT", help="Verified name of the shared folder")
    parser.add_argument("--ranges", nargs="+", choices=ARCHIVES, help="Recover specified archives using durable byte ranges")
    parser.add_argument("--index-layouts", action="store_true")
    args = parser.parse_args()
    if args.index_layouts:
        index_layouts()
    elif args.ranges:
        download_ranges(args.remote.removesuffix(":"),args.folder,args.ranges)
    else:
        download(args.remote.removesuffix(":"),args.folder)


if __name__ == "__main__":
    main()
