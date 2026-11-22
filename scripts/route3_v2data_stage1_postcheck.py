"""Execute the registered geometry-slot postcheck after all 80 training optics."""
import argparse
from collections import defaultdict
import fcntl
from itertools import combinations
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from route3_v2data_B_batch import FORMAL_DATA,GPU_LOCK,ROOT,sha256,write_json
from route3_v2data_B_geometry_noise import comparison
from route3_v2data_B_plan import load
from route3_v2data_stage1 import OUTPUT,verify_disk
from route3_v2data_stage1_dispatch import stamp
from route3_ovg_subpixel import TRAIN_FRAMES

POSTCHECK_POLICY="geometry_gpu_slot_postcheck"


def observations_only(unit,destination):
    destination.mkdir(parents=True,exist_ok=False)
    (destination/"observations").mkdir()
    (destination/"scene-spec.json").symlink_to(unit/"scene-spec.json")
    for name in ["cameras.json",*(f"f{f:03d}.npz" for f in TRAIN_FRAMES)]:
        (destination/"observations"/name).symlink_to(unit/"observations"/name)


def rerun_priors(unit,root):
    out=[]
    for index in (1,2):
        destination=root/unit.name/f"serial-{index}"
        if (destination/"observations/priors/summary.json").is_file():
            out.append(destination)
            continue
        for retry in range(2):
            attempt=destination if not destination.exists() else destination.with_name(f"serial-{index}-attempt-{time.time_ns()}")
            observations_only(unit,attempt)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES="0",OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",
                     PYTHONPATH=f"{ROOT}/scripts:{ROOT}")
            command=["taskset","-c","7-18",sys.executable,"-u",str(ROOT/"scripts/route3_v2data_B_inputs.py"),
                     "--scene",str(attempt)]
            with (attempt/"input-priors.log").open("xb") as log:
                result=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode==0 and (attempt/"observations/priors/summary.json").is_file():
                out.append(attempt)
                break
        else:
            raise RuntimeError(f"postcheck serial prior computation failed twice for {unit.name}")
    return out


def compare(unit,repeats):
    dirs={"dual":unit,"serial_1":repeats[0],"serial_2":repeats[1]}
    pairs={f"{a}__{b}":comparison(dirs[a],dirs[b]) for a,b in combinations(dirs,2)}
    basis=pairs["dual__serial_1"]["metrics"]
    reference={field:{stat:max(pair["metrics"][field][stat] for pair in pairs.values()) for stat in row}
               for field,row in basis.items()}
    checks={pair:{field:{stat:value<=1.5*reference[field][stat] for stat,value in row.items()}
                  for field,row in values["metrics"].items()} for pair,values in pairs.items() if pair.startswith("dual__")}
    return dict(scene=unit.name,serial_directories=[str(d) for d in repeats],pairs=pairs,reference=reference,
                passed=all(v for pair in checks.values() for field in pair.values() for v in field.values()),checks=checks)


def describe(unit):
    valid_pixels=pixels=terms=0
    summed_confidence=0.
    priors=unit/"observations/priors"
    for frame in TRAIN_FRAMES:
        with np.load(priors/f"f{frame:03d}.npz") as arrays:
            depth,conf=arrays["depth"],arrays["confidence"]
            valid=np.isfinite(depth)&(depth>0)&(conf>0)
            valid_pixels+=int(valid.any(axis=0).sum())
            pixels+=int(np.prod(depth.shape[1:]))
            terms+=conf.size
            summed_confidence+=float(conf.sum(dtype=np.float64))
    align=load(priors/"summary.json")["rows"]
    scale=[row["slope"] for row in align if row.get("slope") is not None]
    residual=[row["holdout_after_applied"]["inverse_MAE"] for row in align
              if row.get("holdout_after_applied",{}).get("inverse_MAE") is not None]
    return dict(scene=unit.name,prior_valid_pixel_fraction=valid_pixels/pixels,
                mean_prior_confidence=summed_confidence/terms,
                alignment_scale_median=float(np.median(scale)) if scale else None,
                alignment_residual_median=float(np.median(residual)) if residual else None)


def publish_postcheck(checkout,result,path):
    relative=Path("03_实验记录/records/route3-v2data-B-2026-11-20.json")
    subprocess.run(["git","pull","--rebase","origin","main"],cwd=checkout,check=True)
    record=load(checkout/relative)
    record["v2"]["geometry_gpu_slot_postcheck_result"]=dict(
        path=str(path),sha256=sha256(path),passed=result["passed"],missing_categories=result["missing_categories"],
        measured_categories=list(result["comparisons"]),hard_optical_available=result["hard_optical_available"],groups={
            mode:{key:row[key] for key in ("scenes","statistics")} for mode,row in result["descriptive"].items()
        },note=result["note"],
    )
    write_json(checkout/relative,record)
    subprocess.run(["git","diff","--check"],cwd=checkout,check=True)
    subprocess.run(["git","add","--",str(relative)],cwd=checkout,check=True)
    subprocess.run(["git","diff","--cached","--check"],cwd=checkout,check=True)
    if subprocess.run(["git","diff","--cached","--quiet"],cwd=checkout).returncode==1:
        subprocess.run(["git","commit","-m","记录 B 单双槽事后复核",
                        "-m","Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"],cwd=checkout,check=True)
        subprocess.run(["git","pull","--rebase","origin","main"],cwd=checkout,check=True)
        subprocess.run(["git","push","origin","HEAD:main"],cwd=checkout,check=True)


def run_postcheck(run, production_record, checkout):
    outcome=run/"geometry-slot-postcheck"
    outcome.mkdir(parents=True,exist_ok=True)
    with (outcome/"lock").open("a") as task_lock:
        fcntl.flock(task_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        result_path=outcome/"result.json"
        if result_path.is_file():
            publish_postcheck(checkout,load(result_path),result_path)
            return
        while True:
            hard=load(FORMAL_DATA/"optical/hard-milestone.json")
            if hard.get("rendered")==80:
                ready=all((FORMAL_DATA/"derived-f2"/name/f"f{frame:03d}"/"manifest.json").is_file()
                          for name in hard["scene_ids"] for frame in (1,30,60))
                if ready:
                    break
            time.sleep(45)
        record=load(production_record)["v2"]
        policy=record[POSTCHECK_POLICY]
        tags=record["geometry_gpu_slot_provenance"]["per_scene"]
        units=[FORMAL_DATA/entry["manifest"] for entry in load(FORMAL_DATA/"manifest.json")["units"]
               if entry["partition"]=="training"]
        groups=defaultdict(list)
        categories=defaultdict(list)
        for path in units:
            unit=path.parent
            slot=tags[unit.name]
            groups[slot].append(unit)
            if slot=="dual":
                categories[load(path)["category"]].append(unit)
        chosen={category:min(members,key=lambda item:item.name) for category,members in categories.items()}
        expected={row["category"] for row in load(FORMAL_DATA/"sealed-scene-plans/training.json")["scenes"]}
        missing=sorted(expected-chosen.keys())
        measures={}
        started=time.monotonic()
        with GPU_LOCK.open("a") as gpu:
            fcntl.flock(gpu,fcntl.LOCK_EX)
            for category,unit in sorted(chosen.items()):
                verify_disk(outcome)
                measures[category]=rerun_priors(unit,outcome/"recapture")
        gpu_seconds=time.monotonic()-started
        tests={category:compare(unit,measures[category]) for category,unit in sorted(chosen.items())}
        description={}
        for mode,items in sorted(groups.items()):
            rows=[describe(unit) for unit in items]
            fields={}
            for key in ("prior_valid_pixel_fraction","mean_prior_confidence","alignment_scale_median","alignment_residual_median"):
                values=[row[key] for row in rows if row[key] is not None]
                fields[key]=dict(n=len(values),quantiles_q1_median_q3=np.quantile(values,[.25,.5,.75]).tolist() if values else None)
            description[mode]=dict(scenes=len(rows),statistics=fields,per_scene=rows)
        result=dict(
            preregistered_policy=policy,finished_at_utc=stamp(),GPU_and_queue_seconds=gpu_seconds,
            passed=not missing and all(row["passed"] for row in tests.values()),comparisons=tests,descriptive=description,
            missing_categories=missing,hard_optical_available=hard["rendered"],hard_scene_ids=hard["scene_ids"],
            note="无双槽单元的类别记缺失；波动参照包含两项被判定的差异，结果只记录，不作为生产门槛。",
        )
        write_json(result_path,result)
        publish_postcheck(checkout,result,result_path)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,default=OUTPUT)
    parser.add_argument("--production-record",type=Path,required=True)
    parser.add_argument("--publish-worktree",type=Path,required=True)
    args=parser.parse_args()
    run_postcheck(args.run.resolve(),args.production_record.resolve(),args.publish_worktree.resolve())
