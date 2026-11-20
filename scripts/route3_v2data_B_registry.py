"""Collect reviewed B assets and deterministic splits, with a separate exclusions list."""
import hashlib
from collections import Counter, defaultdict
import json
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np

from route3_v2data_B_sources import DATA, canonical_pool, write_json


PRIORITY = {"Poly Haven": 0, "ambientCG": 1, "Sketchfab": 2, "Objaverse": 3}


def absorb_sources(data_root):
    audits=data_root/"source-audits"
    candidates_dir=data_root/"source-candidates"
    failures, candidates = [], []
    for path in sorted(candidates_dir.glob("*.json")):
        if (candidates_dir/(path.stem+"-review.json")).exists():
            continue
        source = json.loads(path.read_text())
        if source.get("reason"):
            failures.append(dict(source=source["source"], status=source["status"], reason=source["reason"]))
        reviewed_path = audits / path.name
        if path.stem == "gso" and (audits/"gso-reviewed.json").is_file():
            reviewed_path = audits/"gso-reviewed.json"
        reviewed = json.loads(reviewed_path.read_text()) if reviewed_path.is_file() else []
        keys = {(r["asset_id"],r["category"]) for r in reviewed}
        bundles = {r["bundle"] for r in reviewed if "bundle" in r}
        candidates.extend(row for row in source["candidates"]
                          if (row["asset_id"],row["category"]) not in keys and row["asset_id"] not in bundles)
        candidates.extend(reviewed)
    return failures, candidates


def file_fingerprint(row, data_root):
    files = row["files"]
    if row["category"] == "hdri":
        return files[0]["sha256"], None, files[0]["path"]
    path = next(
        (data_root/file["path"] for file in files if Path(file["path"]).name in {"diffuse.jpg", "diff.jpg"}),
        data_root/files[0]["path"],
    )
    if path.suffix.lower() == ".zip":
        with ZipFile(path) as archive:
            name = next((n for n in archive.namelist() if n.lower().endswith(("_color.jpg", "_color.png"))), None)
            if name is None:
                raise ValueError("no diffuse image in the material")
            raw = archive.read(name)
        decoded = path.with_name("validated-diffuse" + Path(name).suffix.lower())
        if not decoded.is_file():
            decoded.write_bytes(raw)
        path = decoded
    else:
        raw = path.read_bytes()
    image = cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("diffuse image cannot be decoded")
    small = cv2.resize(image,(32,32),interpolation=cv2.INTER_AREA).astype(np.int16)
    return hashlib.sha256(raw).hexdigest(), small, str(path.relative_to(data_root))


def register(data_root=DATA, inherited=()):
    cv2.setNumThreads(1)
    source_failures, candidates = absorb_sources(data_root)
    candidates.extend(inherited)
    order = sorted(candidates, key=lambda row: (PRIORITY.get(row["source"], 4), row["source"], row["asset_id"]))
    usable, exclusions, waiting, images = [], [], [], []
    seen = {}
    for row in order:
        status = row["status"]
        key = f'{row["source"]}:{row["asset_id"]}'
        if status == "excluded":
            exclusions.append(dict(row, key=key))
            continue
        if status not in {"passed", "downloaded"}:
            waiting.append(dict(row, key=key))
            continue
        category = row["category"]
        image = None
        render_path = row.get("mesh_path")
        if category in {"mesh","procedural_structure","layout"}:
            signature = row["geometry_sha256"]
        else:
            try:
                signature, image, render_path = file_fingerprint(row,data_root)
            except (ValueError, StopIteration, OSError) as exc:
                exclusions.append(dict(row, key=key, reason=f"material validation: {type(exc).__name__}"))
                continue
        aliases = [("source asset ID", key), (category, signature)]
        if row.get("platform_model_id"):
            aliases.insert(0, ("Sketchfab model identity",row["platform_model_id"]))
        duplicate = next((seen[alias] for alias in aliases if alias in seen), None)
        if image is not None and duplicate is None:
            duplicate = next((old_key for old_key, old in images if np.abs(image-old).mean() <= 2), None)
        if duplicate:
            exclusions.append(dict(row, key=key, canonical_key=duplicate, reason="cross-source asset duplicate"))
            continue
        if image is not None:
            images.append((key,image))
        for alias in aliases:
            seen[alias] = key
        # Assign the pool only after deduplication has selected the canonical ID.
        usable.append(dict(row, key=key, pool=canonical_pool(key), fingerprint=signature, render_path=render_path))
    by_source = defaultdict(Counter)
    for row in usable:
        by_source[row["source"]][row["category"] + "/" + row["pool"]] += 1
    attribution = [
        {key: row[key] for key in ("key","source","asset_id","author","license","page","terms")}
        for row in usable if row["license"].startswith("CC-BY-")
    ]
    stats = Counter(row["license"] for row in usable)
    registered = dict(
        status="provisional", usable=usable, pending=waiting, exclusions=exclusions,
        source_failures=source_failures, counts_by_source_category_pool=dict(by_source),
        counts_by_license=dict(stats),
    )
    write_json(data_root/"asset-registry-provisional.json", registered)
    write_json(data_root/"asset-attribution-provisional.json", attribution)
    return registered


if __name__ == "__main__":
    report = register()
    print("usable", len(report["usable"]), "pending", len(report["pending"]), "excluded", len(report["exclusions"]))
