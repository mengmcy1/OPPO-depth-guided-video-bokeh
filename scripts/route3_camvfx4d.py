"""Inspect/extract selected CamVFX ZIP members using bounded HTTP Range requests.

The scene archives interleave effects. Only pinhole/dof and shared metadata
payloads are fetched. Raw files are made read-only; download slices live in
data/archives so interrupted transfers can resume without overwriting raw data.
"""
import argparse
import concurrent.futures
import io
import json
import os
from pathlib import Path, PurePosixPath
import struct
import threading
import time
import urllib.request
import zipfile
import zlib


class RemoteZip(io.RawIOBase):
    def __init__(self, url, size, use_proxy=False):
        self.url, self.size, self.pos, self.bytes_read = url, size, 0, 0
        self.use_proxy = use_proxy

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        return self.pos

    def tell(self):
        return self.pos

    def read(self, size=-1):
        size = min(self.size - self.pos, size if size >= 0 else self.size)
        if size <= 0:
            return b""
        data = fetch_range(self.url, self.pos, self.pos + size - 1, self.size, self.use_proxy)
        self.pos += len(data)
        self.bytes_read += len(data)
        return data


def fetch_range(url, first, last, total, use_proxy=False):
    # User preference: direct connections; proxy is an explicit per-run opt-in.
    handler = urllib.request.ProxyHandler() if use_proxy else urllib.request.ProxyHandler({})
    opener = urllib.request.build_opener(handler)
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={first}-{last}"})
            with opener.open(req, timeout=90) as response:
                expected = f"bytes {first}-{last}/{total}"
                if response.status != 206 or response.headers.get("Content-Range") != expected:
                    raise RuntimeError("HTTP server did not honor the exact byte range")
                data = response.read(last - first + 2)
            if len(data) != last - first + 1:
                raise RuntimeError("Truncated byte range")
            return data
        except Exception as exc:
            if attempt == 3:
                # Redirect URLs can contain credentials: retain only the type.
                reason = getattr(exc, "reason", None)
                error_number = getattr(reason, "errno", None)
                status = getattr(exc, "code", None)
                raise RuntimeError(f"Range {first}:{last} failed: {type(exc).__name__}; "
                                   f"HTTP={status}, errno={error_number}") from None
            time.sleep(attempt + 1)


def destination(raw, scene, member):
    parts = PurePosixPath(member).parts
    if parts[:2] != ("data", "self") or ".." in parts:
        raise ValueError("Unexpected archive path")
    if len(parts) == 3:
        return raw / scene / parts[2]
    for effect in ("pinhole", "dof"):
        if parts[2] == f"{scene}_{effect}_50cams":
            return raw / scene / effect / Path(*parts[3:])
    return None


def write_raw(path, data, crc):
    if zlib.crc32(data) & 0xFFFFFFFF != crc:
        raise RuntimeError(f"CRC mismatch: {path.name}")
    if path.exists():
        if path.stat().st_size != len(data) or zlib.crc32(path.read_bytes()) & 0xFFFFFFFF != crc:
            raise RuntimeError(f"Refusing to overwrite existing raw file: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    with part.open("xb") as handle:
        handle.write(data)
    os.chmod(part, 0o444)
    part.rename(path)


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".tmp")
    part.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    part.replace(path)


def run_scene(row, revision, raw, cache, metadata_only, workers, use_proxy=False, endpoint="https://huggingface.co"):
    scene = Path(row["path"]).stem
    started = time.monotonic()
    url = f"{endpoint.rstrip('/')}/datasets/Shigon/CamVFX-4D/resolve/{revision}/{row['path']}"
    remote = RemoteZip(url, row["size"], use_proxy=use_proxy)
    with zipfile.ZipFile(remote) as archive:
        infos = sorted(archive.infolist(), key=lambda x: x.header_offset)
        selected = [i for i in infos if not i.is_dir() and destination(raw, scene, i.filename) is not None]
        metadata = [i for i in selected if not i.filename.lower().endswith(".png")]
        local_by_crc = {}
        for item in metadata:
            target = destination(raw, scene, item.filename)
            key = (item.file_size, item.CRC)
            if target.exists():
                content = target.read_bytes()
                if len(content) != item.file_size or zlib.crc32(content) & 0xFFFFFFFF != item.CRC:
                    raise RuntimeError(f"Existing raw metadata differs: {target}")
            elif key in local_by_crc:
                content = local_by_crc[key].read_bytes()
                write_raw(target, content, item.CRC)
            else:
                write_raw(target, archive.read(item), item.CRC)
            local_by_crc[key] = target
        plan = {"scene": scene, "archive_bytes": row["size"], "revision": revision,
                "selected_files": len(selected), "selected_payload_bytes": sum(i.file_size for i in selected),
                "metadata_files": len(metadata), "metadata_http_body_bytes": remote.bytes_read,
                "metadata_only": metadata_only, "image_files_written": 0, "range_body_bytes": 0}
        plan.update(endpoint=endpoint, use_proxy=use_proxy)
        save_json(cache / f"{scene}-metadata.json", plan)
        if metadata_only:
            print(json.dumps(plan), flush=True)
            return plan
        groups = []
        selected_names = {i.filename for i in selected}
        for index, item in enumerate(infos):
            if item.filename not in selected_names:
                continue
            if groups and groups[-1]["last_index"] == index - 1:
                groups[-1]["members"].append(item)
                groups[-1]["last_index"] = index
            else:
                groups.append({"members": [item], "last_index": index, "first": item.header_offset})
        for group_id, group in enumerate(groups):
            last_index = group["last_index"]
            # Stop before the next local header, including no other effect's payload.
            end = infos[last_index + 1].header_offset if last_index + 1 < len(infos) else archive.start_dir
            first, length = group["first"], end - group["first"]
            path = cache / f"{scene}-selected-{group_id}.range"
            progress = cache / f"{scene}-selected-{group_id}-chunks.json"
            chunk_size = 32 * 1024 * 1024
            chunks = list(range((length + chunk_size - 1) // chunk_size))
            done = set(json.loads(progress.read_text())) if progress.exists() else set()
            fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
            os.ftruncate(fd, length)
            lock = threading.Lock()
            def transfer(index):
                if index in done:
                    return
                offset = index * chunk_size
                data = fetch_range(url, first + offset, first + min(length, offset + chunk_size) - 1, row["size"], use_proxy)
                written = 0
                while written < len(data):
                    written += os.pwrite(fd, data[written:], offset + written)
                with lock:
                    done.add(index)
                    plan["range_body_bytes"] += len(data)
                    save_json(progress, sorted(done))
                    print(f"{scene} group={group_id} chunks={len(done)}/{len(chunks)}", flush=True)
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                    list(pool.map(transfer, chunks))
            finally:
                os.close(fd)
            with path.open("rb") as handle:
                for item in group["members"]:
                    if not item.filename.lower().endswith(".png"):
                        continue
                    handle.seek(item.header_offset - first)
                    header = handle.read(30)
                    fields = struct.unpack("<4s5H3L2H", header)
                    if fields[0] != b"PK\x03\x04":
                        raise RuntimeError("Invalid local ZIP header")
                    handle.seek(fields[-2] + fields[-1], 1)
                    data = handle.read(item.compress_size)
                    if item.compress_type == zipfile.ZIP_DEFLATED:
                        data = zlib.decompress(data, -15)
                    elif item.compress_type != zipfile.ZIP_STORED:
                        raise RuntimeError("Unsupported compression")
                    if len(data) != item.file_size:
                        raise RuntimeError("Decoded member length mismatch")
                    write_raw(destination(raw, scene, item.filename), data, item.CRC)
                    plan["image_files_written"] += 1
        plan["elapsed_seconds"] = round(time.monotonic() - started, 3)
        plan["status"] = "complete"
        save_json(cache / f"{scene}-download.json", plan)
        print(json.dumps(plan), flush=True)
        return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--raw", type=Path, default=Path("data/raw/CamVFX-4D"))
    parser.add_argument("--cache", type=Path, default=Path("data/archives/CamVFX-4D-selected"))
    parser.add_argument("--scene", nargs="*")
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--endpoint", default="https://huggingface.co", help="HF endpoint or mirror; revision remains pinned")
    parser.add_argument("--use-proxy", action="store_true", help="Explicitly use environment proxy; default connects directly")
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    rows = [r for r in json.loads(args.tree.read_text()) if r["path"].endswith(".zip") and
            (not args.scene or Path(r["path"]).stem in args.scene)]
    if args.scene:
        rows.sort(key=lambda row: args.scene.index(Path(row["path"]).stem))
    for row in rows:
        run_scene(row, args.revision, args.raw, args.cache, args.metadata_only, args.workers, args.use_proxy, args.endpoint)


if __name__ == "__main__":
    main()
