"""Generate the registered B training tear labels, using the seven-arm protocol."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
import time

import numpy as np

from route3_v2data_B_inputs import FORMAL_DATA
from route3_v2data_module2_labels import FRAMES, PROTOCOL, ROOT, save_json
import route3_v2data_module2_labels as midpoint

RUN = ROOT/"outputs/route3-v2data-module2-tear-d-2026-11-25-01"
CACHE = FORMAL_DATA/"derived-module2-tears-training-2026-11-25-v2"
CONFIG = ROOT/"03_实验记录/records/route3-v2data-module2-tear-d-config-2026-11-25.json"
PILOT = ("thin_structure", "multiple_layers", "curved_self_occlusion")


def label(unit, frames, pilot):
    midpoint.CACHE = CACHE
    midpoint.OUT = RUN
    job = ("training", unit.name, unit/"labels/scene-geometry.npz", unit)
    return midpoint.scene(job, frames, pilot)


def run(pilot, workers):
    if not 1 <= workers <= len(os.sched_getaffinity(0))//2:
        raise ValueError("CPU midpoint labels may use at most half the available vCPUs")
    config = json.loads(CONFIG.read_text())
    plan = config["data"]["training_scenes"]
    selected = ([next(n for n, cat in plan.items() if cat == c) for c in PILOT]
                if pilot else [n for n in plan if n not in config["data"].get("excluded_after_alignment", {})])
    root = FORMAL_DATA/"units/training"
    units = [root/name for name in selected]
    if any(not (p/"manifest.json").is_file() for p in units):
        raise ValueError("Training label plan contains a non-admitted unit")
    if not pilot and not (RUN/"training-label-visual-review.json").is_file():
        raise ValueError("Review the three training overlay images before labeling the full plan")
    start = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(label, unit, (30,) if pilot else FRAMES, pilot) for unit in units]
        rows = [job.result() for job in jobs]
    for row in rows:
        row["category"] = plan[row["scene"]]
    save_json(RUN/("training-label-pilot.json" if pilot else "training-labels.json"),
              dict(protocol=PROTOCOL, rows=rows, excluded=config["data"].get("excluded_after_alignment", {}),
                   workers=workers, seconds=time.monotonic()-start))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    run(args.pilot, args.workers)
