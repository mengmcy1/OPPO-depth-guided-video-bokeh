"""Acquire and split licensed diffuse textures for route 3 card clips."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
from pathlib import Path
import shutil
from time import sleep
from zipfile import ZipFile

import cv2
import numpy as np
import requests

from route3_ovg_subpixel import ROOT, save_json

OUT = ROOT / "data/processed/route3-v2data-2026-11-20/C"
CC0 = "CC0-1.0"
GIB = 2**30
AGENT = {"User-Agent": "oppo-bokeh-route3-research/2.0"}


def canonical_split(key):
    score = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 1000
    return "train" if score < 800 else "dev" if score < 900 else "test"


def fetch(url):
    for attempt in range(3):
        try:
            response = requests.get(url, headers=AGENT, timeout=(10, 45))
            response.raise_for_status()
            return response
        except requests.RequestException:
            if attempt == 2:
                raise
            sleep(1 + attempt)


def listings(source):
    if source == "Poly Haven":
        url = "https://api.polyhaven.com/assets?t=textures"
        return [
            dict(source=source, asset_id=key, author=meta["authors"], catalog=url,
                 page=f"https://polyhaven.com/a/{key}", terms="https://polyhaven.com/license")
            for key, meta in fetch(url).json().items() if meta.get("authors")
        ]
    url = "https://ambientcg.com/api/v2/full_json?type=Material&limit=100&include=downloadData"
    rows = []
    data = fetch(url).json()
    for meta in data["foundAssets"]:
        maps = meta["downloadFolders"].get("default", {}).get("downloadFiletypeCategories", {})
        download = next((v["downloadLink"] for v in maps.get("zip", {}).get("downloads", [])
                         if v["attribute"] == "1K-JPG"), None)
        if download:
            rows.append(dict(source=source, asset_id=meta["assetId"], author="ambientCG (Lennart Demes)",
                             catalog=url, page=meta["shortLink"], terms="https://docs.ambientcg.com/license/",
                             download=download))
    return rows


def acquire(row):
    key = row["asset_id"]
    url = row.get("download")
    if url is None:
        maps = fetch(f"https://api.polyhaven.com/files/{key}").json()
        diffuse = maps.get("Diffuse", maps.get("diff", {}))
        url = diffuse["1k"]["jpg"]["url"]
        payload = fetch(url).content
    else:
        with ZipFile(io.BytesIO(fetch(url).content)) as archive:
            name = next(n for n in archive.namelist() if n.lower().endswith("_color.jpg"))
            payload = archive.read(name)
    image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("diffuse map decode failed")
    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    thumbnail = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
    return dict(row, download=url, license=CC0, terms_version=CC0,
                source_file_sha256=hashlib.sha256(payload).hexdigest()), image, thumbnail


def main():
    cv2.setNumThreads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "assets.json"
    if path.exists():
        return
    if shutil.disk_usage(OUT).free < 60*GIB + GIB:
        raise RuntimeError("disk reserve for texture batch")
    rows, excluded = [], []
    with ThreadPoolExecutor(max_workers=5) as pool:
        sources = ("Poly Haven", "ambientCG")
        futures = {provider: pool.submit(listings, provider) for provider in sources}
        for provider, task in futures.items():
            try:
                candidates = sorted(task.result(), key=lambda v: hashlib.sha256(
                    ("route3-card-v2/" + v["asset_id"]).encode()).digest())
                rows.extend(candidates[:55])
            except (ValueError, requests.RequestException, KeyError) as exc:
                excluded.append(dict(source=provider, reason=str(exc)))
        assets, seen = [], []
        for row, task in zip(rows, [pool.submit(acquire, row) for row in rows]):
            try:
                meta, image, thumb = task.result()
            except (ValueError, requests.RequestException, KeyError, StopIteration, OSError) as exc:
                excluded.append(dict(source=row["source"], asset_id=row["asset_id"], reason=str(exc)))
                continue
            # Compare a common decoded representation across both catalogs before pool assignment.
            thumb = thumb.astype(np.int16)
            duplicate = next((key for key, old in seen if np.abs(thumb-old).mean() <= 2), None)
            key = f'{row["source"]}:{row["asset_id"]}'
            if duplicate:
                excluded.append(dict(key=key, canonical_key=duplicate, reason="duplicate diffuse image"))
                continue
            seen.append((key, thumb))
            relative = Path("textures") / meta["source"].replace(" ", "-") / (meta["asset_id"] + ".png")
            target = OUT / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            ok, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            if not ok:
                raise OSError(target)
            payload = encoded.tobytes()
            target.write_bytes(payload)
            assets.append(dict(meta, key=key, split=canonical_split(key), path=str(relative),
                               sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload)))
    save_json(path, dict(assets=assets, exclusions=excluded, attribution=[]))
    print("C textures", len(assets), "excluded", len(excluded), flush=True)


if __name__ == "__main__":
    main()
