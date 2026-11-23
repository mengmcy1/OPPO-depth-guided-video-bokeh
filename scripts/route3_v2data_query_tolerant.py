"""Evaluate the extended first-pass grid with the project's square ±2 px boundary band."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
from fractions import Fraction
import os
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from route3_v2data_B_batch import GIB, ROOT, sha256, write_json
from route3_v2data_B_plan import load
from route3_v2data_query_retrain import RECORD, TIME_ZONE, first_pass_frames

EPSILON = (.02, .05, .10, .20, .30, .40)
Q_MIN = (.3, .5, .7, .8, .9)
BINS = ("0", "1_to_2", "3_to_5", "over_5")
KERNEL_2 = np.ones((5, 5), np.uint8)
KERNEL_5 = np.ones((11, 11), np.uint8)


def tolerant_f1(n):
    precision_hits = n["precision_hits"]
    recall_hits = n["recall_hits"]
    denominator = precision_hits*n["gt_boundary"] + recall_hits*n["split"]
    return Fraction(2*precision_hits*recall_hits, max(1, denominator))


def selection_key(row):
    n = row["counts"]
    distance_to_original = ((Fraction(str(row["epsilon"]))-Fraction(".05"))**2
                            + (Fraction(str(row["q_min"]))-Fraction(".5"))**2)
    return (-tolerant_f1(n), Fraction(n["over_5"], max(1, n["split"])), distance_to_original)


def evaluate(checkpoint, query, reference):
    start = time.monotonic()
    geometry_sha, query_sha = sha256(checkpoint), sha256(query)
    if (geometry_sha, query_sha) != (reference["stage1_geometry_sha256"], reference["new_query_sha256"]):
        raise ValueError("The checkpoints do not match the recorded frozen geometry and adopted Query")
    if reference["decision"]["selected"] != "new":
        raise ValueError("The recorded decision does not adopt the new Query")

    counts = {(e, q): Counter() for e in EPSILON for q in Q_MIN}
    frames_by_scene = defaultdict(list)
    far_alpha, far_quality = [], []
    for scene, frame, alpha, quality, gate, valid, boundary, _ in first_pass_frames(checkpoint, query):
        frames_by_scene[scene].append(frame)
        near_2 = cv2.dilate(boundary.astype(np.uint8), KERNEL_2).astype(bool)
        near_5 = cv2.dilate(boundary.astype(np.uint8), KERNEL_5).astype(bool)
        precision_band = near_2 & valid
        distance_bands = (boundary, near_2 & ~boundary, near_5 & ~near_2, ~near_5)
        possible_far = distance_bands[-1] & gate & (alpha < 1-min(EPSILON)) & (quality >= min(Q_MIN))
        far_alpha.append(alpha[possible_far])
        far_quality.append(quality[possible_far])
        gt_count = int(boundary.sum())
        depth_and_coverage = {e: gate & (alpha < 1-e) for e in EPSILON}
        by_quality = {q: quality >= q for q in Q_MIN}
        for (e, q), n in counts.items():
            b = depth_and_coverage[e] & by_quality[q]
            hits = int((b & boundary).sum())
            recall_band = cv2.dilate(b.astype(np.uint8), KERNEL_2).astype(bool)
            n.update(tp=hits, fp=int((b & valid & ~boundary).sum()), fn=gt_count-hits,
                     gt_boundary=gt_count, split=int(b.sum()), pixels=b.size,
                     precision_hits=int((b & precision_band).sum()), recall_hits=int((boundary & recall_band).sum()))
            n.update({k: int((b & mask).sum()) for k, mask in zip(BINS, distance_bands)})

    old = reference["first_pass"]
    if len(frames_by_scene) != 15 or any(frames != old["frames"] for frames in frames_by_scene.values()):
        raise ValueError("Expected all 13 recorded label frames for each of 15 development scenes")
    previous = old["selected"]
    old_pair = (previous["epsilon"], previous["q_min"])
    observed = {key: counts[old_pair][key] for key in ("tp", "fp", "fn")}
    expected = {key: previous[key] for key in observed}
    if observed != expected:
        raise ValueError(f"The original strict counts do not reproduce: {observed} instead of {expected}")
    print("QUERY_TOLERANT_STRICT_REPRODUCED", observed, flush=True)

    alpha = np.concatenate(far_alpha)
    quality = np.concatenate(far_quality)
    grid = []
    for (e, q), n in counts.items():
        selected_alpha = alpha[(alpha < 1-e) & (quality >= q)]
        if sum(n[k] for k in BINS) != n["split"] or len(selected_alpha) != n["over_5"]:
            raise ValueError("Split distance bins or alpha support do not match the split count")
        quantiles = np.quantile(selected_alpha, [.1, .5, .9]).tolist() if len(selected_alpha) else [None]*3
        grid.append(dict(
            epsilon=e, q_min=q,
            strict_f1=2*n["tp"]/max(1, 2*n["tp"]+n["fp"]+n["fn"]),
            tolerant_f1=float(tolerant_f1(n)), precision=n["precision_hits"]/max(1, n["split"]),
            recall=n["recall_hits"]/max(1, n["gt_boundary"]), b1_fraction=n["split"]/n["pixels"],
            b1_distance_fractions={k: n[k]/max(1, n["split"]) for k in BINS},
            b1_over_5_alpha_hat=dict(zip(("p10", "p50", "p90"), quantiles)),
            counts=dict(n),
        ))
    selected = min(grid, key=selection_key)
    upper_boundary = [key for key, upper in (("epsilon", max(EPSILON)), ("q_min", max(Q_MIN)))
                      if selected[key] == upper]
    return dict(
        grid=grid, selected=selected, upper_grid_boundary=upper_boundary,
        selection_note="最优点位于网格边界" if upper_boundary else "最优点未触及 ε=0.40 或 q_min=0.90",
        previous_selected=dict(epsilon=old_pair[0], q_min=old_pair[1]),
        strict_reproduction=dict(passed=True, **observed),
        epsilon_grid=list(EPSILON), q_min_grid=list(Q_MIN),
        scenes=len(frames_by_scene), frames_per_scene=list(old["frames"]), resolution=512,
        stage1_geometry_sha256=geometry_sha, query_sha256=query_sha,
        distance="Chebyshev pixel distance (5x5 square dilation at 2 px), matching the project boundary band",
        precision_support="b=1 within valid GT boundary dilation / all b=1",
        recall_support="valid GT boundary pixels within 2 px of a b=1 pixel / valid GT boundary pixels",
        pooling="sum pixel counts before calculating precision, recall and F1",
        tie_break="lowest proportion over 5 px, then squared distance to epsilon=.05 and q_min=.5",
        seconds=time.monotonic()-start, recorded_at=datetime.now(TIME_ZONE).isoformat(timespec="seconds"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--query-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        parser.error("assign the shared CUDA device before evaluation")
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.cuda.set_per_process_memory_fraction(10*GIB/torch.cuda.get_device_properties(0).total_memory, 0)
    write_json(args.output, evaluate(args.checkpoint, args.query_checkpoint, load(ROOT/RECORD)))
