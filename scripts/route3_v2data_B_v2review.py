"""Review staged B v2 meshes concurrently, with a resumable audit per source."""
import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib
import json
import time

from route3_v2data_B_batch import DATA, GIB, blender, run, write_json


V2 = DATA/"v2"


def review(source, rows):
    name = hashlib.sha256("\n".join(row["asset_id"] for row in rows).encode()).hexdigest()[:16]
    directory = V2/"mesh-audit-jobs"/source/name
    directory.mkdir(parents=True,exist_ok=True)
    inputs = directory/"input.json"
    output = directory/"output.json"
    if output.is_file():
        value = json.loads(output.read_text())
        if {row["asset_id"] for row in value} == {row["asset_id"] for row in rows}:
            return value
    write_json(inputs,dict(candidates=rows))
    try:
        run(blender("route3_v2data_B_mesh_check.py","--base",V2,"--candidates",inputs,
                    "--output",output,"--allow-open",threads=2),
            directory/"audit.log",cpu_limit_bytes=4*GIB,timeout_seconds=240)
    except RuntimeError as exc:
        if len(rows)>1:
            pivot=len(rows)//2
            return review(source,rows[:pivot])+review(source,rows[pivot:])
        return [dict(rows[0],status="excluded",reason=str(exc))]
    return json.loads(output.read_text())


def watch(sources, workers, retry_worker_errors):
    audits = V2/"source-audits"
    audits.mkdir(parents=True,exist_ok=True)
    cache = {}
    for source in sources:
        path = audits/(source+".json")
        rows=json.loads(path.read_text()) if path.is_file() else []
        cache[source] = {
            row["asset_id"]:row for row in rows
            if not (retry_worker_errors and row.get("reason","").startswith("scene process returned "))
        }
    with ThreadPoolExecutor(max_workers=workers) as pool:
        running = {}
        next_source = 0
        while True:
            snapshots = {}
            inflight = {(source,row["asset_id"]) for source,rows in running.values() for row in rows}
            todo = {}
            for source in sources:
                path = V2/"source-candidates"/(source+".json")
                if not path.is_file():
                    continue
                snapshot = json.loads(path.read_text())
                snapshots[source] = snapshot
                todo[source] = [row for row in snapshot["candidates"] if row["status"]=="pending_mesh_checks"
                                and row["asset_id"] not in cache[source] and (source,row["asset_id"]) not in inflight]
            for offset in range(len(sources)):
                if len(running) == workers:
                    break
                source = sources[(next_source+offset)%len(sources)]
                waiting = todo.get(source,[])
                if not waiting:
                    continue
                if (waiting[0].get("source_face_count") or 0) > 250000:
                    shard = waiting[:1]
                else:
                    boundary = next((i for i,r in enumerate(waiting[:6])
                                     if (r.get("source_face_count") or 0)>250000),6)
                    shard = waiting[:boundary]
                task = pool.submit(review,source,shard)
                running[task] = (source,shard)
            next_source = (next_source+1)%len(sources)
            if not running:
                if all(source in snapshots and not todo[source] and
                       (snapshots[source].get("catalog_complete") or snapshots[source]["status"] in {
                           "待凭据","catalog_unavailable","complete",
                       }) for source in sources):
                    return
                time.sleep(12)
                continue
            done,_ = wait(running,timeout=12,return_when=FIRST_COMPLETED)
            for task in done:
                source,_ = running.pop(task)
                cache[source].update((row["asset_id"],row) for row in task.result())
                write_json(audits/(source+".json"),list(cache[source].values()))
                print(source,"reviewed",len(cache[source]),dict(Counter(
                    row["status"] for row in cache[source].values())),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources",nargs="+")
    parser.add_argument("--workers",type=int,choices=range(1,5),default=4)
    parser.add_argument("--retry-worker-errors",action="store_true")
    args=parser.parse_args()
    watch(args.sources,args.workers,args.retry_worker_errors)
