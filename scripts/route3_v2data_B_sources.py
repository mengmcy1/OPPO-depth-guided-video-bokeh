"""Acquire B-tier source candidates before mesh checks and global pool assignment."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import itertools
import json
from pathlib import Path, PurePosixPath
import re
import shutil
from urllib.parse import quote, urljoin, urlsplit
from uuid import NAMESPACE_URL, uuid5

import requests

from route3_v2data_B_front import DATA, RESERVE_BYTES, write_json


CC0 = "CC0-1.0"
CC_BY = "CC-BY-4.0"
CRED = Path("/root/.config/oppo-bokeh")
POLY = "https://api.polyhaven.com"
AMBIENT = "https://ambientcg.com/api/v2/full_json"
FUEL = "https://fuel.gazebosim.org/1.0/GoogleResearch/models"
BLENDKIT = "https://www.blenderkit.com/api/v1/search/"
SKETCHFAB = "https://api.sketchfab.com/v3"
OBJAVERSE_PATHS = "https://huggingface.co/datasets/allenai/objaverse/resolve/main/object-paths.json.gz"
SMITHSONIAN = "https://api.si.edu/openaccess/api/v1.0/search"
HUMAN_BUNDLE = (
    "https://mirror.blender.org/demo/asset-bundles/human-base-meshes/"
    "human-base-meshes-bundle-v1.4.1.zip"
)


class SourceError(RuntimeError):
    """An error whose message is safe to record."""


def request(url, *, params=None, auth=None, stream=False):
    headers = {"User-Agent": "OPPO-Bokeh-Research/2.0"}
    if auth:
        headers["Authorization"] = f"Bearer {auth}"
    try:
        response = requests.get(url, params=params, headers=headers, stream=stream, timeout=(12, 45))
        if response.status_code in (401, 403) and (auth or "api_key" in (params or {})):
            response.close()
            raise SourceError(f"待凭据或访问授权 (HTTP {response.status_code})")
        if response.status_code != 200:
            response.close()
            raise SourceError(f"HTTP {response.status_code}")
        return response
    except requests.RequestException as exc:
        raise SourceError(type(exc).__name__) from None


def metadata(url, **kwargs):
    with request(url, **kwargs) as response:
        try:
            return response.json()
        except ValueError:
            raise SourceError("响应不是有效 JSON") from None


def token(name):
    path = CRED / (name + ".token")
    value = path.read_text().strip() if path.is_file() else ""
    if not value:
        raise SourceError("待凭据")
    return value


def stable_limit(items, count, key):
    order = sorted(items, key=lambda item: hashlib.sha256(key(item).encode()).digest())
    return order[:count]


def canonical_pool(key):
    h = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 1000
    return "train" if h < 800 else "dev" if h < 900 else "test"


def asset(source, asset_id, author, license_id, category, channel, terms, **kwargs):
    return dict(
        source=source, asset_id=asset_id, author=author, license=license_id,
        category=category, download_channel=channel, terms=terms, terms_version=license_id,
        can_redistribute=True, **kwargs,
    )


def poly_catalog(limit):
    for catalog, category in (("models", "mesh"), ("textures", "material"), ("hdris", "hdri")):
        rows = metadata(f"{POLY}/assets", params={"t": catalog})
        for key in stable_limit(rows, limit, lambda item: f"Poly Haven:{item}"):
            meta = rows[key]
            yield asset(
                "Poly Haven", key, meta.get("authors"), CC0, category,
                f"{POLY}/files/{key}", "https://polyhaven.com/license",
                scale_m=meta.get("dimensions"), source_face_count=meta.get("polycount"),
                taxonomy=meta.get("categories"), page=f"https://polyhaven.com/a/{key}",
            )


def ambient_catalog(limit):
    for kind, category in (("Material", "material"), ("3DModel", "mesh")):
        catalog = metadata(
            AMBIENT, params={"limit": 100, "include": "downloadData", "type": kind}
        )
        rows = stable_limit(catalog["foundAssets"], limit, lambda row: f"ambientCG:{row['assetId']}")
        for meta in rows:
            groups = meta.get("downloadFolders", {}).get("default", {}).get("downloadFiletypeCategories", {})
            formats = groups.get("zip", {}).get("downloads", [])
            choice = next((row for row in formats if row["attribute"] == "1K-JPG"), None)
            if category == "mesh" and not choice:
                choice = next((row for row in formats if "1K" in row["attribute"]), None)
            yield asset(
                "ambientCG", meta["assetId"], "ambientCG (Lennart Demes)", CC0, category,
                AMBIENT, "https://docs.ambientcg.com/license/", page=meta.get("shortLink"),
                files=[dict(url=choice["downloadLink"], name=choice["fileName"], size=choice["size"])]
                if choice else [],
            )


def human_catalog(_limit):
    yield asset(
        "Blender Studio Human Base Meshes", "v1.4.1", "Blender Studio and community contributors",
        CC0, "mesh_bundle", HUMAN_BUNDLE, "https://www.blender.org/download/demo-files/#assets",
        files=[dict(url=HUMAN_BUNDLE, name="human-base-meshes-bundle-v1.4.1.zip")],
        asset_id_policy="individual Blender asset name after bundle inspection",
    )


def gso_catalog(limit):
    rows = metadata(FUEL, params={"page": 1, "per_page": 100})
    for meta in stable_limit(rows, limit, lambda row: f"Google Scanned Objects:{row['name']}"):
        key = meta["name"]
        escaped = quote(key, safe="")
        uri = f"{FUEL}/{escaped}/1/{escaped}.zip"
        cc_by = meta["license_url"].rstrip("/").endswith("creativecommons.org/licenses/by/4.0")
        yield asset(
            "Google Scanned Objects", key, meta["owner"], CC_BY if cc_by else None, "mesh",
            uri, meta["license_url"], page=f"https://app.gazebosim.org/GoogleResearch/fuel/models/{escaped}",
            files=[dict(url=uri, name=key + ".zip", size=meta["filesize"])],
            decimate=True,
        )


def smithsonian_catalog(limit):
    rows = metadata(
        SMITHSONIAN, params={
            "api_key": token("smithsonian"),
            "q": 'online_media_type:"3D Models"',
            "rows": max(50, limit),
        },
    )["response"]["rows"]
    items = {}
    for item in rows:
        media = item["content"].get("descriptiveNonRepeating", {}).get("online_media", {}).get("media", [])
        for entry in media:
            if entry.get("type") == "3d_voyager":
                items.setdefault(entry["id"], (item, entry))
    found, unavailable = [], []
    for item, entry in items.values():
        resources=[]
        for resource in entry.get("resources", []):
            name = resource.get("filename", "").lower()
            if not resource.get("url") or not (name.endswith(".glb") or "gltf_std.zip" in name):
                continue
            attrs = {key: value for part in resource.get("attributes",[]) for key,value in part.items()}
            size = attrs.get("FILE_SIZE")
            if size and size > 128*2**20:
                continue
            resources.append((resource, attrs))
        options = sorted(resources,key=lambda r: (
            r[0].get("category") != "Low Resolution",
            r[1].get("DRACO_COMPRESSED", False),
            r[1].get("FILE_SIZE", 0),
            r[0]["filename"],
        ))
        files = []
        if options:
            choice, attrs = options[0]
            files.append(dict(url=choice["url"],name=choice["filename"],size=attrs.get("FILE_SIZE")))
        candidate=asset(
            "Smithsonian Open Access", entry["id"], f"Smithsonian Institution ({item['unitCode']})",
            CC0 if entry.get("usage", {}).get("access") == "CC0" else None, "mesh",
            "Smithsonian Open Access API / 3D API", "https://www.si.edu/openaccess/faq",
            page=entry.get("content"), decimate=True, files=files,
        )
        (found if files and candidate["license"] else unavailable).append(candidate)
    yield from stable_limit(found, limit, lambda row: row["asset_id"])
    yield from unavailable


def blendkit_catalog(limit):
    auth = token("blendkit")
    search = metadata(
        BLENDKIT, params={"query": "chair+asset_type:model+license:cc_zero",
                          "page_size": min(50, max(24, limit))}, auth=auth,
    )
    for meta in stable_limit(search["results"], limit, lambda row: row["assetBaseId"]):
        author = meta.get("author") or {}
        key = meta["assetBaseId"]
        files = [item for item in meta.get("files", []) if item.get("fileType") == "blend"]
        endpoints = [urljoin(BLENDKIT, item["downloadUrl"]) for item in files if item.get("downloadUrl")]
        yield asset(
            "Blendkit", key, author.get("fullName") or author.get("id"),
            CC0 if meta.get("license") == "cc_zero" else None, "mesh",
            "Blendkit licensed asset API", "https://www.blenderkit.com/terms-of-service/",
            page=f"https://www.blenderkit.com/asset-gallery-detail/{key}/",
            file_api=endpoints[0] if endpoints and meta.get("canDownload") else None,
        )


def sketchfab_auth():
    auth = token("sketchfab")
    metadata(f"{SKETCHFAB}/me", auth=auth)
    return auth


def sketchfab_catalog(limit):
    auth = sketchfab_auth()
    response = metadata(
        f"{SKETCHFAB}/search",
        params={"type": "models", "downloadable": "true", "q": "household furniture"},
        auth=auth,
    )
    for item in stable_limit(response["results"], limit, lambda row: row["uid"]):
        yield sketchfab_model(item["uid"], auth, "Sketchfab")


def objaverse_catalog(limit):
    auth = sketchfab_auth()
    with request(OBJAVERSE_PATHS) as response:
        catalog = json.loads(gzip.decompress(response.content))
    for uid in stable_limit(catalog, limit, lambda item: item):
        yield sketchfab_model(uid, auth, "Objaverse")


def sketchfab_model(uid, auth, source):
    if not re.fullmatch(r"[0-9a-f]{32}", uid):
        raise SourceError("无效的 Sketchfab model ID")
    model_api = f"{SKETCHFAB}/models/{uid}"
    try:
        model = metadata(model_api, auth=auth)
    except SourceError as exc:
        return asset(source, uid, None, None, "mesh", model_api, "https://sketchfab.com/terms",
                     exclusion=str(exc))
    tags = model.get("tags")
    tagged = tags is not None and any(
        str(tag.get(name, "")).lower().replace("-", "") == "noai"
        for tag in tags for name in ("name", "slug")
    )
    blocked = tagged or model.get("isAIRestricted", False)
    license_info = model.get("license") or {}
    url = license_info.get("url") or ""
    kind = url.rstrip("/").split("/")
    license_id = None
    if tags is not None and not blocked and model.get("isDownloadable"):
        if "creativecommons.org/publicdomain/zero/1.0" in url:
            license_id = CC0
        elif "creativecommons.org/licenses/by/" in url and kind[-1] in {"2.0", "2.5", "3.0", "4.0"}:
            license_id = "CC-BY-" + kind[-1]
    author = model.get("user") or {}
    return asset(
        source, uid, author.get("displayName") or author.get("username"),
        license_id, "mesh", f"{model_api}/download", url or "https://sketchfab.com/terms",
        platform_model_id=uid, current_status_url=model_api,
        source_face_count=model.get("faceCount"), page=model.get("viewerUrl"),
        platform_terms_version="2026-10-27",
        exclusion="NoAI" if blocked else "当前标签、下载资格或许可不可核实或不适用",
    )


def protected_files(row):
    if row["source"] == "Blendkit":
        endpoint = row.get("file_api")
        if not endpoint or urlsplit(endpoint).hostname not in {"www.blenderkit.com", "www.blendkit.com"}:
            raise SourceError("缺少可信 Blendkit 文件下载入口")
        info = metadata(
            endpoint, auth=token("blendkit"),
            params={"scene_uuid": str(uuid5(NAMESPACE_URL, "oppo-bokeh/" + row["asset_id"]))},
        )
        url, name = info["filePath"], row["asset_id"] + ".blend"
    else:
        info = metadata(row["download_channel"], auth=token("sketchfab"))
        file = info.get("glb") or info.get("gltf")
        if not file:
            raise SourceError("官方 Download API 未提供 glb 或 glTF")
        url = file["url"]
        name = row["asset_id"] + (".glb" if "glb" in info else ".zip")
    if urlsplit(url).scheme != "https":
        raise SourceError("非 HTTPS 资产地址")
    return [dict(name=name, url=url)]


def open_movie_catalog(_limit):
    films = (
        (
            "Spring", "autumn-v1", "CC-BY-4.0",
            "https://studio.blender.org/projects/spring/pages/about/",
            "https://studio.blender.org/characters/autumn/v1/",
            "https://studio.blender.org/download-source/files/97/97108c7a70aef830ef9b3f185442307e/97108c7a70aef830ef9b3f185442307e.zip",
            "autumn.zip",
        ),
        (
            "Sprite Fright", "sprite-v1", "CC-BY-1.0",
            "https://studio.blender.org/projects/sprite-fright/pages/about/",
            "https://studio.blender.org/characters/sprite/v1/",
            "https://studio.blender.org/download-source/files/d9/d985a2f5f303ad2f1abe49c316b566b1/d985a2f5f303ad2f1abe49c316b566b1.blend",
            "sprite.blend",
        ),
    )
    for movie, key, license_id, terms, page, url, file in films:
        yield asset(
            "Blender Studio Open Movies", f"{movie}/{key}", "Blender Foundation | studio.blender.org",
            license_id, "mesh", page, terms, page=page, movie=movie,
            files=[dict(url=url, name=file)],
        )


def procedural_catalog(_limit):
    bands = ((0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 8.0))
    shapes = ("railing", "wire", "branch", "fence", "mesh")
    for shape, (low, high), variant in itertools.product(shapes, bands, range(12)):
        yield asset(
            "Procedural Geometry Nodes", f"{shape}/{low:g}-{high:g}px/{variant:02d}",
            "Project authors", "project-owned", "procedural_structure",
            "Blender Geometry Nodes", "13.5 and 13.15.1",
            shape=shape, width_px_at_512=[low, high], parameter_segment=variant, files=[],
        )


def poly_files(row):
    raw = metadata(row["download_channel"])
    kind = row["category"]
    if kind == "mesh":
        main = raw["gltf"]["1k"]["gltf"]
        relative = PurePosixPath(urlsplit(main["url"]).path).name
        return [dict(main, name=relative)] + [dict(data, name=name) for name, data in main.get("include", {}).items()]
    if kind == "hdri":
        main = raw["hdri"]["1k"]["hdr"]
        return [dict(main, name=row["asset_id"] + ".hdr")]
    files = []
    for name, variants in raw.items():
        fmt = variants.get("1k", {}).get("jpg")
        if fmt and name.lower() in {"diffuse", "diff", "rough", "nor_gl", "metal"}:
            files.append(dict(fmt, name=name.lower() + ".jpg"))
    if not any(file["name"] in ("diffuse.jpg", "diff.jpg") for file in files):
        raise SourceError("没有 1K 漫反射贴图")
    return files


def save_file(base, file):
    name = PurePosixPath(file["name"])
    if name.is_absolute() or ".." in name.parts or not name.name:
        raise SourceError("无效的资源路径")
    dest = base.joinpath(*name.parts)
    dest.parent.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256()
    md5 = hashlib.md5(usedforsecurity=False)
    temp = dest.with_suffix(dest.suffix + ".part")
    if not dest.exists():
        with request(file["url"], stream=True) as response:
            expected_size = file.get("size") or int(response.headers.get("Content-Length") or 0)
            if shutil.disk_usage(dest.parent).free < RESERVE_BYTES + expected_size:
                raise SourceError("磁盘余量不满足 60 GiB 加下载大小")
            with temp.open("wb") as stream:
                for block in response.iter_content(2**20):
                    stream.write(block)
            temp.replace(dest)
    with dest.open("rb") as stream:
        while block := stream.read(2**20):
            sha.update(block)
            md5.update(block)
    if file.get("md5") and md5.hexdigest() != file["md5"]:
        raise SourceError("源 MD5 不匹配")
    if file.get("size") and dest.stat().st_size != file["size"]:
        raise SourceError("源文件大小不匹配")
    return dict(path=str(dest.relative_to(DATA)), bytes=dest.stat().st_size, sha256=sha.hexdigest())


def prepare(row):
    result = {k: v for k, v in row.items() if k not in {"files", "file_api", "exclusion"}}
    if not row["license"] or not row["author"]:
        return dict(result, status="excluded", reason=row.get("exclusion", "许可证或作者不可核实"))
    if row["category"] == "procedural_structure":
        return dict(result, status="parameter_segment_pending_geometry_nodes")
    try:
        if row["source"] == "Poly Haven":
            files = poly_files(row)
        elif row["source"] in {"Blendkit", "Sketchfab", "Objaverse"}:
            files = protected_files(row)
        else:
            files = row["files"]
        if not files:
            raise SourceError("未找到适用的下载文件")
        key = hashlib.sha256(f'{row["source"]}:{row["asset_id"]}'.encode()).hexdigest()[:20]
        local = DATA / "source-files" / key
        stored = [save_file(local, file) for file in files]
        status = "pending_mesh_checks" if row["category"].startswith("mesh") else "downloaded"
        return dict(result, status=status, files=stored)
    except (SourceError, KeyError, ValueError, OSError) as exc:
        reason = str(exc) if isinstance(exc, SourceError) else type(exc).__name__
        return dict(result, status="excluded", reason=reason)


CATALOGS = {
    "poly-haven": ("Poly Haven", poly_catalog),
    "ambientcg": ("ambientCG", ambient_catalog),
    "human-base-meshes": ("Blender Studio Human Base Meshes", human_catalog),
    "gso": ("Google Scanned Objects", gso_catalog),
    "smithsonian": ("Smithsonian Open Access", smithsonian_catalog),
    "procedural": ("Procedural Geometry Nodes", procedural_catalog),
    "blendkit": ("Blendkit", blendkit_catalog),
    "sketchfab": ("Sketchfab", sketchfab_catalog),
    "objaverse": ("Objaverse", objaverse_catalog),
    "open-movies": ("Blender Studio Open Movies", open_movie_catalog),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", choices=CATALOGS)
    parser.add_argument("--limit", type=int, default=36, help="Maximum candidate records per external category")
    parser.add_argument("--workers", type=int, default=6, choices=range(1, 9))
    args = parser.parse_args()
    output = DATA / "source-candidates"
    output.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for slug in args.sources:
            name, catalog = CATALOGS[slug]
            path = output / (slug + ".json")
            if path.exists():
                continue
            try:
                rows = list(catalog(args.limit))
                values = list(pool.map(prepare, rows))
                result = dict(source=name, candidates=values, catalog_count=len(rows))
            except (SourceError, KeyError, ValueError, OSError) as exc:
                reason = str(exc) if isinstance(exc, SourceError) else type(exc).__name__
                status = "待凭据" if "待凭据" in reason else "catalog_unavailable"
                result = dict(source=name, candidates=[], status=status, reason=reason)
            write_json(path, result)
            print(slug, len(result["candidates"]), result.get("status", "catalogued"), flush=True)


if __name__ == "__main__":
    main()
