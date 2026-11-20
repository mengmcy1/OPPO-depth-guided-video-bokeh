"""Enumerate official B v2 source catalogs and checkpoint individual downloads."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import fcntl
import gzip
from itertools import count
import json
import shutil
import time

import requests

from route3_v2data_B_front import DATA, RESERVE_BYTES, write_json
from route3_v2data_B_sources import (
    CATALOGS, FUEL, OBJAVERSE_PATHS, SKETCHFAB, SourceError, blendkit_catalog, gso_asset,
    metadata, poly_catalog, prepare, request, sketchfab_auth, sketchfab_model, smithsonian_catalog,
    stable_limit,
)

V2 = DATA/"v2"
SOURCES = ("gso","poly-haven","smithsonian","blendkit","sketchfab","objaverse")
SEARCHES = ("household furniture","stone sculpture","household appliance","ceramic vase",
            "plant pot","office furniture","curved chair")


def fuel_page(page):
    params={"page":page,"per_page":100}
    # A temporary transport failure must not invalidate earlier official pages.
    for attempt in range(8):
        with requests.Session() as client:
            client.trust_env=bool(attempt%2)
            try:
                with client.get(FUEL,params=params,timeout=(6,14)) as response:
                    if response.status_code==200:
                        rows=response.json()
                        if isinstance(rows,list):
                            return rows
                    if response.status_code==404:
                        return []
                    retry = response.headers.get("Retry-After","")
                    delay = int(retry) if retry.isdecimal() else min(12,1+attempt)
            except (requests.RequestException,ValueError):
                delay = min(12,1+attempt)
        time.sleep(delay)
    raise SourceError(f"official Fuel catalog page {page} is temporarily unavailable")


def fuel_catalog():
    cache=V2/"source-inventory"/"gso-fuel-pages"
    cache.mkdir(parents=True,exist_ok=True)
    names=set()
    for page in count(1):
        path=cache/f"{page:04d}.json"
        rows=json.loads(path.read_text()) if path.is_file() else fuel_page(page)
        ids={row["name"] for row in rows}
        if len(ids)!=len(rows) or ids.intersection(names):
            raise SourceError("official Fuel pages repeated model IDs")
        if not path.exists():
            write_json(path,rows)
            print("gso catalog page",page,len(rows),flush=True)
        names.update(ids)
        yield from (gso_asset(row) for row in rows)
        if len(rows)<100:
            break


def unchecked_catalog(source, limit):
    if source == "gso":
        yield from fuel_catalog()
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
            uids = ([row["asset_id"] for row in status["candidates"]]
                    if retry and status.get("catalog_complete") else sketchfab_ids(source,limit,auth))
            catalog = {uid:uid for uid in uids}
            status["account"] = "verified"
        else:
            catalog = {row["asset_id"]:row for row in unchecked_catalog(source,limit)}
        if source=="gso" and len(catalog)<status.get("catalog_count",0):
            raise SourceError("official Fuel listing is shorter than its previous full snapshot")
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
        (any(term in previous[key]["reason"] for term in
             ("HTTP 5","HTTP 429","SSLError","Timeout","ConnectionError","rate limited","ChunkedEncodingError"))
         or (source=="poly-haven" and previous[key]["reason"]=="KeyError")))]
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
    locks = V2/"source-candidates"
    locks.mkdir(parents=True,exist_ok=True)
    with (locks/f".{args.source}.lock").open("a") as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(args.source,"another acquisition is active",flush=True)
            return
        fetch(args.source,args.limit,args.workers,args.retry)


if __name__ == "__main__":
    main()
