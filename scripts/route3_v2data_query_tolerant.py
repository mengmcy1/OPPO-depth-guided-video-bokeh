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
from src.ovg_first_pass import boundary_candidate, observed_split

EPSILON = (.02, .05, .10, .20, .30, .40)
Q_MIN = (.3, .5, .7, .8, .9)
FINAL_EPSILON = (*EPSILON, .50, .60, .70)
FINAL_Q_MIN = (0., .1, .2, *Q_MIN)
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


def distance_masks(boundary):
    near_2 = cv2.dilate(boundary.astype(np.uint8), KERNEL_2).astype(bool)
    near_5 = cv2.dilate(boundary.astype(np.uint8), KERNEL_5).astype(bool)
    return near_2, (boundary, near_2 & ~boundary, near_5 & ~near_2, ~near_5)


def add_case(n, split, valid, boundary, near_2, distance_bands):
    hits = int((split & boundary).sum())
    gt_count = int(boundary.sum())
    recall_band = cv2.dilate(split.astype(np.uint8), KERNEL_2).astype(bool)
    n.update(tp=hits, fp=int((split & valid & ~boundary).sum()), fn=gt_count-hits,
             gt_boundary=gt_count, split=int(split.sum()), pixels=split.size,
             precision_hits=int((split & near_2 & valid).sum()),
             recall_hits=int((boundary & recall_band).sum()))
    n.update({k:int((split & mask).sum()) for k,mask in zip(BINS,distance_bands)})


def q_positive(checkpoint, query, reference):
    start = time.monotonic()
    final = reference["first_pass_final"]
    geometry_sha,query_sha=sha256(checkpoint),sha256(query)
    if (geometry_sha,query_sha)!=(final["stage1_geometry_sha256"],final["query_sha256"]):
        raise ValueError("Frozen geometry or Query changed")
    epsilon=final["selected"]["epsilon"]
    if (epsilon,final["selected"]["q_min"])!=(.40,0.):
        raise ValueError("The q-positive reevaluation expects the fixed epsilon and the q=0 grid baseline")
    selected,baseline,removed=Counter(),Counter(),Counter()
    frames_by_scene=defaultdict(list)
    for scene,frame,alpha,quality,gate,valid,boundary,_ in first_pass_frames(checkpoint,query):
        frames_by_scene[scene].append(frame)
        candidate=boundary_candidate(alpha,gate,epsilon)
        if not np.isfinite(quality[candidate]).all() or (quality[candidate]<0).any():
            raise ValueError("The observed-background existence marker is invalid")
        split=observed_split(alpha,gate,quality,epsilon)
        missing=candidate & ~split
        near_2,dist=distance_masks(boundary)
        add_case(selected,split,valid,boundary,near_2,dist)
        add_case(baseline,candidate,valid,boundary,near_2,dist)
        removed.update(pixels=int(missing.sum()))
        removed.update({k:int((missing&mask).sum()) for k,mask in zip(BINS,dist)})
    if len(frames_by_scene)!=15 or any(frames!=final["frames_per_scene"] for frames in frames_by_scene.values()):
        raise ValueError("Expected 15 development scenes and all their recorded label frames")
    if dict(baseline)!=final["selected"]["counts"]:
        raise ValueError("The q=0 grid baseline counts changed during reevaluation")
    if selected["split"]+removed["pixels"]!=baseline["split"] or sum(removed[k] for k in BINS)!=removed["pixels"]:
        raise ValueError("Missing candidates do not partition the threshold-free splits")
    return dict(
        epsilon=epsilon, observed_background_required=True, rule="candidate_mask and q > 0",
        decision_note="q_min 不由 F1 选择，按方案 4.1 取 q > 0（存在背景候选）。",
        tolerant_f1=float(tolerant_f1(selected)),
        precision=selected["precision_hits"]/max(1,selected["split"]),
        recall=selected["recall_hits"]/max(1,selected["gt_boundary"]),
        b1_fraction=selected["split"]/selected["pixels"],
        b1_distance_fractions={k:selected[k]/max(1,selected["split"]) for k in BINS},
        removed_q_zero=dict(pixels=removed["pixels"],previous_b1_pixels=baseline["split"],
                            fraction_of_previous_b1=removed["pixels"]/max(1,baseline["split"]),
                            distance_counts={k:removed[k] for k in BINS},
                            distance_fractions={k:removed[k]/max(1,removed["pixels"]) for k in BINS}),
        counts=dict(selected),grid_baseline_reproduced=True,
        scenes=len(frames_by_scene),frames_per_scene=final["frames_per_scene"],resolution=512,
        stage1_geometry_sha256=geometry_sha,query_sha256=query_sha,
        recorded_at=datetime.now(TIME_ZONE).isoformat(timespec="seconds"),seconds=time.monotonic()-start,
    )


def evaluate(checkpoint, query, reference, *, final=False):
    start = time.monotonic()
    geometry_sha, query_sha = sha256(checkpoint), sha256(query)
    if (geometry_sha, query_sha) != (reference["stage1_geometry_sha256"], reference["new_query_sha256"]):
        raise ValueError("The checkpoints do not match the recorded frozen geometry and adopted Query")
    if reference["decision"]["selected"] != "new":
        raise ValueError("The recorded decision does not adopt the new Query")

    eps_grid, q_grid = (FINAL_EPSILON, FINAL_Q_MIN) if final else (EPSILON, Q_MIN)
    previous_run = reference["first_pass_tolerant"] if final else None
    reused = {(r["epsilon"], r["q_min"]): r for r in previous_run["grid"]} if final else {}
    if final and ((previous_run["stage1_geometry_sha256"],previous_run["query_sha256"]) !=
                  (geometry_sha,query_sha) or len(reused)!=30):
        raise ValueError("The 30 cached evaluation points must match the checkpoint and the Query")
    points = [(e,q) for e in eps_grid for q in q_grid]
    counts = {key: Counter() for key in points if key not in reused}
    false_inside = Counter()
    inner_pixels = 0
    frames_by_scene = defaultdict(list)
    far_alpha, far_quality = [], []
    for scene, frame, alpha, quality, gate, valid, boundary, interior in first_pass_frames(checkpoint, query):
        frames_by_scene[scene].append(frame)
        near_2,distance_bands=distance_masks(boundary)
        candidate = (alpha < 1-min(eps_grid)) & gate
        if not final:
            candidate &= quality >= min(q_grid)
        possible_far = distance_bands[-1] & candidate
        far_alpha.append(alpha[possible_far])
        far_quality.append(quality[possible_far])
        coverage = {e: alpha < 1-e for e in eps_grid}
        if final:
            interior = interior & ~near_2
            inner_pixels += int(interior.sum())
            false_inside.update({e:int((interior & below).sum()) for e,below in coverage.items()})
        depth_and_coverage = {e: gate & below for e,below in coverage.items()}
        by_quality = {q: True if q == 0 else quality >= q for q in q_grid}
        for (e, q), n in counts.items():
            b = depth_and_coverage[e] & by_quality[q]
            add_case(n,b,valid,boundary,near_2,distance_bands)

    old = reference["first_pass"]
    if len(frames_by_scene) != 15 or any(frames != old["frames"] for frames in frames_by_scene.values()):
        raise ValueError("Expected all 13 recorded label frames for each of 15 development scenes")
    previous = old["selected"]
    old_pair = (previous["epsilon"], previous["q_min"])
    old_counts = reused[old_pair]["counts"] if final else counts[old_pair]
    observed = {key: old_counts[key] for key in ("tp", "fp", "fn")}
    expected = {key: previous[key] for key in observed}
    if observed != expected:
        raise ValueError(f"The original strict counts do not reproduce: {observed} instead of {expected}")
    print("QUERY_TOLERANT_STRICT_CACHED" if final else "QUERY_TOLERANT_STRICT_REPRODUCED",
          observed, flush=True)

    alpha = np.concatenate(far_alpha)
    quality = np.concatenate(far_quality)
    grid = []
    for (e, q), n in counts.items():
        take = alpha < 1-e
        if q != 0:
            take &= quality >= q
        selected_alpha = alpha[take]
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
    if final:
        measured = {(r["epsilon"],r["q_min"]):r for r in grid}
        grid = []
        for e,q in points:
            row = reused[e,q] if (e,q) in reused else measured[e,q]
            grid.append(dict(row,gt_interior_false_semitransparent_rate=false_inside[e]/max(1,inner_pixels),
                             gt_interior_false_semitransparent_pixels=false_inside[e],
                             gt_interior_pixels=inner_pixels))
        old_pair = (previous_run["selected"]["epsilon"],previous_run["selected"]["q_min"])
    selected = min(grid, key=selection_key)
    upper_boundary = [key for key, upper in (("epsilon", max(eps_grid)), ("q_min", max(q_grid)))
                      if selected[key] == upper]
    lower_boundary = [key for key, lower in (("epsilon", min(eps_grid)), ("q_min", min(q_grid)))
                      if selected[key] == lower]
    boundary = upper_boundary or (final and lower_boundary)
    result = dict(
        grid=grid, selected=selected, upper_grid_boundary=upper_boundary,
        selection_note="最优点位于网格边界" if boundary else "最优点位于网格内部",
        previous_selected=dict(epsilon=old_pair[0], q_min=old_pair[1]),
        strict_reproduction=dict(passed=True, **observed),
        epsilon_grid=list(eps_grid), q_min_grid=list(q_grid),
        scenes=len(frames_by_scene), frames_per_scene=list(old["frames"]), resolution=512,
        stage1_geometry_sha256=geometry_sha, query_sha256=query_sha,
        distance="Chebyshev pixel distance (5x5 square dilation at 2 px), matching the project boundary band",
        precision_support="b=1 within valid GT boundary dilation / all b=1",
        recall_support="valid GT boundary pixels within 2 px of a b=1 pixel / valid GT boundary pixels",
        pooling="sum pixel counts before calculating precision, recall and F1",
        tie_break="lowest proportion over 5 px, then squared distance to epsilon=.05 and q_min=.5",
        seconds=time.monotonic()-start, recorded_at=datetime.now(TIME_ZONE).isoformat(timespec="seconds"),
    )
    if final:
        result.update(lower_grid_boundary=lower_boundary,reused_points=len(reused),new_points=len(counts),
                      strict_reproduction=dict(source="first_pass_tolerant",passed=True,**observed),
                      gt_interior="valid opaque GT outside the ±2 px square boundary band",
                      zero_q_min="第一遍拆分只由 α̂ 决定，Query 可信度不作门限。原深度约束保持。",
                      first_pass_uses_query_confidence_gate=selected["q_min"]!=0)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--query-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument("--final-grid", action="store_true")
    mode.add_argument("--q-positive",action="store_true")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        parser.error("assign the shared CUDA device before evaluation")
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.cuda.set_per_process_memory_fraction(10*GIB/torch.cuda.get_device_properties(0).total_memory, 0)
    reference=load(ROOT/RECORD)
    result=q_positive(args.checkpoint,args.query_checkpoint,reference) if args.q_positive else (
        evaluate(args.checkpoint,args.query_checkpoint,reference,final=args.final_grid))
    write_json(args.output,result)
