"""Enumerate official B v2 source catalogs and checkpoint individual downloads."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import gzip
import json
import shutil
import time

from route3_v2data_B_front import DATA, RESERVE_BYTES, write_json
from route3_v2data_B_sources import (
    CATALOGS, OBJAVERSE_PATHS, SKETCHFAB, SourceError, blendkit_catalog, gso_catalog,
    metadata, poly_catalog, prepare, request, sketchfab_auth, sketchfab_model, smithsonian_catalog,
    stable_limit,
)

V2 = DATA/"v2"
SOURCES = ("gso","poly-haven","smithsonian","blendkit","sketchfab","objaverse")
SEARCHES = ("household furniture","stone sculpture","household appliance","ceramic vase",
            "plant pot","office furniture","curved chair")


def unchecked_catalog(source, limit):
    if source == "gso":
        yield from gso_catalog(None,full=True)
    elif source == "poly-haven":
        yield from poly_catalog(None,categories=("models",))
    elif source == "blendkit":
        for term in ("chair","sofa","table","cabinet","vase","sculpture"):
            yield from blendkit_catalog(limit,term=term)
    elif source == "smithsonian":
        for start in range(0,limit,100):
            yield from smithsonian_catalog(100,start=start)


def sketchfab_ids(source, limit, auth):
    if source == "objaverse":
        with request(OBJAVERSE_PATHS) as response:
            catalog = json.loads(gzip.decompress(response.content))
        return stable_limit(catalog,limit,str)
    ids = set()
    failures = []
    for term in SEARCHES:
        try:
            response = metadata(f"{SKETCHFAB}/search",params={
                "type":"models","downloadable":"true","q":term,
            },auth=auth)
        except SourceError as exc:
            failures.append(exc)
            print(source,"search term unavailable",term,str(exc),flush=True)
            continue
        ids.update(row["uid"] for row in response["results"])
    if not ids and failures:
        raise failures[0]
    return stable_limit(ids,limit,str)


def acquire(row):
    value = prepare(row,data_root=V2)
    if value["status"] == "download_deferred":
        time.sleep(value.get("retry_after_seconds") or 12)
        value = prepare(row,data_root=V2)
    if value["status"] == "download_deferred":
        value.update(status="excluded",reason="official API rate limited after paced retry")
    return value


def acquire_protected(uid, auth, source):
    model = sketchfab_model(uid,auth,source)
    if model.get("model_detail_http_status") == 429:
        time.sleep(12)
        model = sketchfab_model(uid,auth,source)
    return acquire(model)


def fetch(source, limit, workers, retry=False):
    output = V2/"source-candidates"
    output.mkdir(exist_ok=True,parents=True)
    name = CATALOGS[source][0]
    receipt = output/(source+".json")
    status = json.loads(receipt.read_text()) if receipt.is_file() else {}
    if status.get("catalog_complete") and not retry:
        return
    auth = None
    try:
        if source in {"sketchfab","objaverse"}:
            auth = sketchfab_auth()
            catalog = {uid:uid for uid in sketchfab_ids(source,limit,auth)}
            status["account"] = "verified"
        else:
            catalog = {row["asset_id"]:row for row in unchecked_catalog(source,limit)}
        sizes = sum(sum(file.get("size") or 0 for file in row.get("files",[]))
                    for row in catalog.values() if isinstance(row,dict))
        if shutil.disk_usage(output).free < RESERVE_BYTES+sizes:
            raise SourceError("insufficient disk space for catalog download and 60 GiB reserve")
    except (SourceError,OSError,KeyError,ValueError) as exc:
        reason = str(exc) if isinstance(exc,SourceError) else type(exc).__name__
        waiting = "待凭据" in reason or (isinstance(exc,SourceError) and exc.status_code in {401,403})
        status.update(source=name,status="待凭据" if waiting else "catalog_unavailable",reason=reason,
                      candidates=status.get("candidates",[]))
        write_json(receipt,status)
        print(source,status["status"],reason,flush=True)
        return
    previous = {row["asset_id"]:row for row in status.get("candidates",[])}
    def lookup(key):
        return acquire_protected(key,auth,name) if auth else acquire(catalog[key])

    todo = [key for key in catalog if key not in previous or (
        retry and previous[key]["status"]=="excluded" and
        any(term in previous[key]["reason"] for term in ("HTTP 5","HTTP 429","SSLError","Timeout","ConnectionError","rate limited","ChunkedEncodingError")))]
    checkpoint = dict(source=name,status="downloading",catalog_count=len(catalog),estimated_source_bytes=sizes,
                      account=status.get("account"))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for offset in range(0,len(todo),12):
            for row in pool.map(lookup,todo[offset:offset+12]):
                previous[row["asset_id"]] = row
            checkpoint["candidates"] = [previous[key] for key in catalog if key in previous]
            write_json(receipt,checkpoint)
            print(source,len(checkpoint["candidates"]),"/",len(catalog),dict(Counter(
                row["status"] for row in checkpoint["candidates"])),flush=True)
    checkpoint.update(catalog_complete=True,status="complete",candidates=[previous[key] for key in catalog])
    write_json(receipt,checkpoint)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source",choices=SOURCES)
    parser.add_argument("--limit",type=int,default=240)
    parser.add_argument("--workers",type=int,choices=range(1,7),default=4)
    parser.add_argument("--retry",action="store_true")
    args = parser.parse_args()
    fetch(args.source,args.limit,args.workers,args.retry)


if __name__ == "__main__":
    main()
