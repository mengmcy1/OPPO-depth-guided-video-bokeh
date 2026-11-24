"""Recount representation links without rendering or changing archived evaluation outputs."""
import argparse
from pathlib import Path
import subprocess

import torch

import route3_v2data_stage1_visualize as gallery
from route3_v2data_baseline_dev3 import read, write
from route3_v2data_stage1_evaluate import load_paired_checkpoint
from route3_v2data_stage1_visualize_repr import frame_setup
from route3_v2data_stage2_train import INITIAL, configuration, fixed_inputs


@torch.inference_mode()
def count(source, output):
    if output.exists():
        raise FileExistsError(output)
    subprocess.run(["nvidia-smi","--query-gpu=memory.used,memory.total,utilization.gpu","--format=csv"],check=True)
    configuration()
    torch.cuda.set_per_process_memory_fraction(4*2**30/torch.cuda.get_device_properties(0).total_memory)
    net = load_paired_checkpoint(INITIAL,"ema").eval()
    frozen = fixed_inputs()
    cases = {r["file"] for r in read(source)["rows"] if "error" not in r}
    rows = []
    for task in gallery.planned_frames():
        planned = [v["file"] for v in task["files"] if v["file"] in cases]
        if not planned:
            continue
        state = frame_setup(task,net,None,None,None)
        links, samples = state[5:7]
        gt_pairs = int(samples["same"][1].sum())
        counts = {k:sum(int(axis.sum()) for axis in axes) for k,axes in links.items()}
        if gt_pairs != counts["GT"]:
            raise ValueError("GT-link replay disagrees with the eligible same-surface support")
        rows.append(dict(scene=task["name"],frame=task["frame"],cases=planned,links=counts))
        print("LINK_COUNTS",task["name"],task["frame"],counts,flush=True)
        del state
    seen = {f for r in rows for f in r["cases"]}
    if seen != cases:
        raise ValueError("Replayed connection grid differs from the archived case set")
    totals = {}
    for cohort in ("development","legacy"):
        members = [r for r in rows if r["scene"].startswith("development-")==(cohort=="development")]
        totals[cohort] = {k:sum(r["links"][k] for r in members) for k in rows[0]["links"]}
    write(output,dict(source=str(source),frozen=frozen,cohorts=totals,rows=rows,
                      mask="Uses archived representation code. Eligible endpoints are valid & ~split; GT additionally requires labeled continuity."))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a = p.parse_args()
    count(a.source.resolve(),a.output.resolve())
