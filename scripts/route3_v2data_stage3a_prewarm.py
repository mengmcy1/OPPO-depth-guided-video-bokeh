"""Prepare immutable f/2 GT ray parents on CPU while Stage 3a trains."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import os
from pathlib import Path

import cv2
import numpy as np

from route3_ovg_data import b_foreground_reference
from route3_v2data_B_batch import write_json
from route3_v2data_loader import TrainingData, b_optical_cases
from route3_v2data_optics import hidden_parents, ray_fields
from route3_v2data_stage3a import OUT, optical_pool


def warm(args):
    cache, unit = args
    cv2.setNumThreads(1)
    written = {}
    for spec in b_optical_cases(unit):
        if spec["fstop"] != 2.:
            continue
        destination = cache/unit.name/f'{spec["truth"].stem}.npz'
        if destination.is_file():
            continue
        ref = b_foreground_reference(unit.name,spec["frame"])
        parents = hidden_parents(spec,ray_fields(spec),ref)
        if parents is None:
            parents = dict(uv=np.empty((0,2),np.float32),depth=np.empty(0,np.float32),
                           recoverable=np.empty(0,bool),colors_linear=np.empty((0,3),np.float32))
        destination.parent.mkdir(parents=True,exist_ok=True)
        # The on-demand trainer can publish the same case first. Only the
        # fixed labeled parents are written, with an atomic no-clobber link.
        temp = destination.with_name(f".{destination.name}.{os.getpid()}.part")
        try:
            with temp.open("xb") as f:
                np.savez_compressed(f,**parents)
            try:
                os.link(temp,destination)
            except FileExistsError:
                continue
            written[spec["truth"].stem] = len(parents["depth"])
        finally:
            if temp.is_file():
                temp.unlink()
    return dict(scene=unit.name,cases=written)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=Path,default=OUT)
    args=p.parse_args()
    run=args.run.resolve()
    inventory=TrainingData()
    pool=optical_pool(inventory)
    order=pool["hard"]+sorted(set(pool["scenes"])-set(pool["hard"]))
    jobs=[(run/"hidden-parents",inventory.b[s]) for s in order]
    results=[]
    with ProcessPoolExecutor(max_workers=2,mp_context=get_context("spawn")) as workers:
        for item in workers.map(warm,jobs):
            results.append(item)
            print("PARENTS_READY",item["scene"],len(item["cases"]),flush=True)
    write_json(run/"hidden-parent-prewarm.json",dict(scene_count=len(order),rows=results,cpu_workers=2))


if __name__=="__main__":
    main()
