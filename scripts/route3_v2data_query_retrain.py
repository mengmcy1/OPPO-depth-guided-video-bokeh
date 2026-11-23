"""Train the fixed-size Query, compare paired dev scores, and choose first-pass gates."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
from fractions import Fraction
from functools import lru_cache
import math
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

import cv2
import numpy as np
import torch

import route3_ovg_data as ovg
from route3_g3_r14_train import SETTINGS, train_network
from route3_ovg_1c import aligned_sample
from route3_ovg_stage2_query import FrozenQuery
from route3_s2a_learn import Store
from route3_v2data_B_batch import FORMAL_DATA, GIB, ROOT, sha256, write_json
from route3_v2data_B_inputs import library_unit_paths, observation
from route3_v2data_B_plan import load
from route3_v2data_stage1_evaluate import (
    METRICS, QUERY, load_paired_checkpoint, paired_psnr, pooled, summary, view_from_observation,
)

FLOORS = ("zero_visible", "zero_combined")
EPS = (.02, .05, .10, .20)
Q = (.3, .5, .7)
RECORDS = Path("03_实验记录/records")
RECORD = RECORDS/"route3-v2data-query-2026-11-23.json"
B_RECORD = RECORDS/"route3-v2data-B-2026-11-20.json"
PROGRESS = Path("03_实验记录/当前实验进度与方法论.md")
EVALUATOR = ROOT/"scripts/route3_v2data_stage1_evaluate.py"
TIME_ZONE = ZoneInfo("Asia/Shanghai")


class TrainingStore(Store):
    @lru_cache(maxsize=16)
    def inputs(self, path):
        return Store.inputs.__wrapped__(self, path)

    @lru_cache(maxsize=16)
    def targets(self, path, input_path):
        return Store.targets.__wrapped__(self, path, input_path)

    @lru_cache(maxsize=32)
    def active(self, input_path, focus, fstop):
        return Store.active.__wrapped__(self, input_path, focus, fstop)

    @lru_cache(maxsize=16)
    def context_file(self, path):
        value = super().context({"context": path})
        del self.contexts[path]
        return value

    def context(self, manifest):
        return self.context_file(manifest["context"])


def monitor_row(manifest):
    middle = manifest["frames"][2]
    return min(manifest["rows"], key=lambda r: (abs(r["frame"]-middle), -r["case"]["focus_index"]))


def train(data, output):
    state = load(data/"prepared.json")
    plan = load(data/"inventory.json")
    if "seconds" not in state or len(state["accepted"])+len(state["skipped"]) != len(plan["jobs"]):
        raise ValueError("Query labels have not been accounted for")
    tiers = {name: row["tier"] for name, row in state["accepted"].items() if row["split"] == "train"}
    monitors = sorted(name for name, row in state["accepted"].items() if row["split"] == "dev")
    if not (output/"training.json").is_file():
        train_network(data, output, sorted(tiers), monitors, store=TrainingStore(),
                      scene_tiers=tiers, monitor_row=monitor_row)
    result = load(output/"training.json")
    if result["sample_tiers"] != {"B": 14000, "C": 6000} or result["trainable_parameters"] != 38281:
        raise ValueError("Query run did not match the registered batch and architecture contract")
    return result


def run_evaluation(checkpoint, query, f2_root, output):
    args = [sys.executable, "-u", str(EVALUATOR), "--checkpoint", str(checkpoint),
            "--query-checkpoint", str(query), "--output", str(output), "--f2-root", str(f2_root),
            "--phase", "1b", "--branch", "same-surface", "--weight", "20", "--ema-only"]
    output.mkdir(parents=True, exist_ok=True)
    with (output/"evaluation.log").open("a") as log:
        subprocess.run(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    return load(output/"ema.json")


def distribution(values):
    return dict(n=len(values), median_db=float(np.median(values)) if values else None,
                sd_db=float(np.std(values, ddof=1)) if len(values) > 1 else None,
                positive=sum(x > 0 for x in values), negative=sum(x < 0 for x in values),
                tied=sum(x == 0 for x in values))


def paired_metric(old, new, get, score):
    def per_scene(rows):
        grouped = defaultdict(list)
        for row in rows:
            if row["status"] == "evaluated" and get(row) is not None:
                grouped[row["scene"]].append(get(row))
        return grouped

    a, b = per_scene(old), per_scene(new)
    if a.keys() != b.keys():
        raise ValueError("Query comparison changed scorable scenes")
    scenes = {}
    for scene in sorted(a):
        first, second = score(a[scene]), score(b[scene])
        if first[1] is not None and second[1] is not None:
            scenes[scene] = dict(old_db=first[1], new_db=second[1], difference_db=second[1]-first[1])
    first = score([x for values in a.values() for x in values])
    second = score([x for values in b.values() for x in values])
    return dict(scenes=scenes, paired=distribution([v["difference_db"] for v in scenes.values()]),
                old_distribution=distribution([v["old_db"] for v in scenes.values()]),
                new_distribution=distribution([v["new_db"] for v in scenes.values()]),
                pooled=dict(old=first[0], new=second[0], old_db=first[1], new_db=second[1],
                            difference_db=second[1]-first[1] if None not in (first[1], second[1]) else None))


def absolute_score(rows):
    value = pooled(rows)
    return value, value["psnr_db"]


def increment_score(rows):
    value = paired_psnr([row["on"] for row in rows], [row["off"] for row in rows])
    return value, value["delta_db"]


def compare(before, after):
    result = {}
    for split in ("development", "legacy"):
        old, new = before[split+"_rows"], after[split+"_rows"]
        keys = lambda rows: [(x["scene"], x["case"], x["status"]) for x in rows]
        if keys(old) != keys(new):
            raise ValueError("Query comparison changed P2 case support")
        differences = []
        for a, b in zip(old, new):
            if a.get("actual_focus_m") != b.get("actual_focus_m"):
                raise ValueError("Query comparison changed actual focus")
            for metric, pair in a.get("query_pair", {}).items():
                other = b["query_pair"][metric]
                if any(pair[k]["pixels"] != other[k]["pixels"] for k in ("on", "off")):
                    raise ValueError("Query comparison changed scoring pixels")
                differences.append(abs(pair["off"]["sse"]-other["off"]["sse"]))
        result[split] = dict(max_query_off_sse_abs_delta=max(differences, default=0.),
                             zero_floors={}, regions={})
        for metric in METRICS:
            result[split]["regions"][metric] = dict(
                open=paired_metric(old, new, lambda r: r[metric], absolute_score),
                query_increment=paired_metric(old, new, lambda r: r["query_pair"].get(metric), increment_score),
            )
        old_summary, new_summary = summary(old), summary(new)
        result[split]["unscored_case_ids"] = new_summary["unscored_case_ids"]
        for metric in FLOORS:
            result[split]["zero_floors"][metric] = dict(
                old=old_summary[metric], new=new_summary[metric],
                psnr=paired_metric(old, new, lambda r: r[metric], absolute_score),
            )
    return result


def adopt(comparison):
    dev = comparison["development"]
    near, far = (dev["regions"][m]["query_increment"] for m in METRICS)
    far_median = far["paired"]["median_db"]
    far_pooled = far["pooled"]["difference_db"]
    near_pooled = near["pooled"]["difference_db"]
    floors = {k: dev["zero_floors"][k]["new"]["all_passed"] for k in FLOORS}
    checks = dict(far_median_nonnegative=far_median is not None and far_median >= 0,
                  near_pooled_delta_ge_minus_point_one=near_pooled is not None and near_pooled >= -.1,
                  both_zero_floors=all(floors.values()))
    alternative = dict(far_median_ge_minus_point_zero_five=far_median is not None and far_median >= -.05,
                       far_pooled_delta_ge_minus_point_one=far_pooled is not None and far_pooled >= -.1,
                       near_pooled_delta_ge_minus_point_one=checks["near_pooled_delta_ge_minus_point_one"],
                       zero_combined=floors["zero_combined"])
    return dict(selected="new" if all(checks.values()) else "old", checks=checks,
                alternative=dict(all_passed=all(alternative.values()), checks=alternative),
                far_scene_median_delta_db=far_median, far_pooled_delta_db=far_pooled,
                near_pooled_delta_db=near_pooled, zero_floors=floors)


@torch.inference_mode()
def first_pass_frames(checkpoint, query_path):
    net = load_paired_checkpoint(checkpoint, "ema").eval().requires_grad_(False)
    query = FrozenQuery(query_path)
    for unit in library_unit_paths("dev"):
        views = [view_from_observation(observation(unit, frame, aligned=True), frame)
                 for frame in ovg.TRAIN_FRAMES]
        for i, view in enumerate(views):
            x, y = aligned_sample(("B", unit.name, i), "cuda")
            pred = net(x, protect_residual=True, render_heads=False)
            front, back, alpha = (pred[k][0].cpu().numpy() for k in ("depth1", "depth2", "alpha"))
            quality = query.first_from_views(views, None, view["frame"], front, back)["q"]
            valid, boundary, valid_alpha, gt_alpha = (
                y[k][0].cpu().numpy() for k in ("valid", "depth_boundary", "valid_alpha", "alpha")
            )
            boundary = boundary & valid
            interior = valid & valid_alpha & (gt_alpha >= .9999) & ~boundary
            depth_gate = back >= 1.03*front
            yield unit.name, view["frame"], alpha, quality, depth_gate, valid, boundary, interior
        print("QUERY_FIRST_PASS", unit.name, flush=True)


def first_pass(checkpoint, query_path):
    sums = {(eps, q): Counter() for eps in EPS for q in Q}
    for _, _, alpha, quality, depth_gate, valid, boundary, interior in first_pass_frames(checkpoint, query_path):
        for (eps, q), count in sums.items():
            b = depth_gate & (alpha < 1-eps) & (quality >= q)
            count.update(tp=int((b&boundary).sum()), fp=int((b&valid&~boundary).sum()),
                         fn=int((~b&boundary).sum()), split=int(b.sum()), pixels=b.size,
                         interior_splits=int((b&interior).sum()))
    grid = []
    for (eps, q), n in sums.items():
        tp, fp, fn = (n[k] for k in ("tp", "fp", "fn"))
        grid.append(dict(epsilon=eps, q_min=q, f1=2*tp/max(1, 2*tp+fp+fn),
                         precision=tp/max(1, tp+fp), recall=tp/max(1, tp+fn),
                         b1_fraction=n["split"]/n["pixels"],
                         b1_in_gt_interior_fraction=n["interior_splits"]/max(1, n["split"]), **n))

    def select(row):
        distance = (Fraction(str(row["epsilon"]))-Fraction(".05"))**2+(Fraction(str(row["q_min"]))-Fraction(".5"))**2
        return (-Fraction(2*row["tp"], max(1, 2*row["tp"]+row["fp"]+row["fn"])),
                Fraction(row["interior_splits"], max(1, row["split"])), distance)

    return dict(grid=grid, selected=min(grid, key=select), resolution=512, frames=list(ovg.TRAIN_FRAMES),
                ground_truth_boundary="B depth_boundary within valid geometry",
                ground_truth_interior="valid opaque coverage outside depth_boundary",
                pooling="sum TP, FP and FN over all 15 development scenes and 13 label frames",
                tie_break="less GT-interior b=1, then squared distance to epsilon=.05 and q_min=.5")


def data_summary(state):
    result = {}
    for tier in ("B", "C"):
        result[tier] = {}
        for split in ("train", "dev"):
            rows = [r for r in state["accepted"].values() if (r["tier"],r["split"]) == (tier,split)]
            stats = Counter()
            for row in rows:
                stats.update(row["stats"])
            result[tier][split] = dict(scenes=len(rows), labeled_cases=sum(r["cases"] for r in rows), **stats)
    return result


def optical_status():
    manifest = load(FORMAL_DATA/"manifest.json")
    return dict(admitted=manifest["counts"]["train_optical"],
                f2_complete=sum(all((FORMAL_DATA/"derived-f2"/r["scene_id"]/f"f{f:03d}"/"manifest.json").is_file()
                                    for f in (1, 30, 60)) for r in manifest["optical_units"]))


def publish(report):
    def git(*args):
        subprocess.run(["git", *args], cwd=ROOT, check=True)

    git("diff", "--check")
    git("pull", "--rebase", "origin", "main")
    today = datetime.now(TIME_ZONE)
    optical = optical_status()
    report["recorded_at"] = today.isoformat(timespec="seconds")
    report["B_training_optical"] = optical
    write_json(ROOT/RECORD, report)
    b_record = load(ROOT/B_RECORD)
    v2 = b_record["v2"]
    v2["formal_counts"]["train_optical"] = optical["admitted"]
    v2["target_gap"]["train_optical"] = v2["optical_full_scope"]["total_target"]-optical["admitted"]
    v2["optical_full_scope"]["status_at_query_decision"] = dict(optical, recorded_at=report["recorded_at"])
    write_json(ROOT/B_RECORD, b_record)
    path = ROOT/PROGRESS
    lines = path.read_text().splitlines(keepends=True)
    timestamp = next(i for i,line in enumerate(lines) if line.startswith("> 更新时间"))
    lines[timestamp] = f"> 更新时间：{today.date()}（客户端日期，阶段 2 前 Query 重训与第一遍解码选定）\n"
    decision = report["decision"]
    grid = report["first_pass"]["selected"]
    note = (f"**阶段 2 前 Query 重训**。在冻结的同表面 30k EMA 几何上完成 20,000 步，"
            f"按 §13.16.3 采用{'新' if decision['selected']=='new' else '原'} Query。"
            f"第一遍选定 ε={grid['epsilon']:g}、q_min={grid['q_min']:g}。"
            f"B 光学 {optical['admitted']} 场入列，{optical['f2_complete']} 场 F2 齐备。"
            f"下一步为阶段 2。[重训记录](records/{RECORD.name})\n")
    section = next((i for i,line in enumerate(lines) if line.startswith("**阶段 2 前 Query 重训**")), None)
    if section is None:
        section = next(i for i,line in enumerate(lines) if line.startswith("**新阶段 1 选支修正**"))
        lines[section+1:section+1] = ["\n", note]
    else:
        lines[section] = note
    path.write_text("".join(lines))
    git("diff", "--check")
    git("add", "--", str(RECORD), str(B_RECORD), str(PROGRESS))
    git("diff", "--cached", "--check")
    git("commit", "-m", "记录扩充数据 Query 重训和第一遍解码选定",
        "-m", "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>")
    git("pull", "--rebase", "origin", "main")
    git("push", "origin", "HEAD:main")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage1-run", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    stage1 = args.stage1_run.resolve()
    checkpoint = stage1/"1b/same-surface/checkpoints/step-030000.pt"
    fitted_query = out/"train/step-020000.pt"
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.cuda.set_per_process_memory_fraction(10*GIB/torch.cuda.get_device_properties(0).total_memory, 0)
    write_json(out/"status.json", {"phase": "training"})
    training = train(out/"data", out/"train")
    torch.cuda.empty_cache()
    write_json(out/"status.json", {"phase": "paired_evaluation"})
    old, new = [run_evaluation(checkpoint, path, stage1/"development-F2", out/label)
                for label, path in (("old-query", QUERY), ("new-query", fitted_query))]
    comparison = compare(old, new)
    decision = adopt(comparison)
    write_json(out/"comparison.json", dict(decision=decision, comparisons=comparison))
    write_json(out/"status.json", {"phase": "first_pass", "query": decision["selected"]})
    grid_path = out/"first-pass.json"
    grid = load(grid_path)
    if not grid:
        grid = first_pass(checkpoint, fitted_query if decision["selected"] == "new" else QUERY)
        write_json(grid_path, grid)
    state = load(out/"data/prepared.json")
    loss = [dict(step=m["step"], by_tier={
        tier: float(np.mean([r["loss"] for r in m["rows"] if r["scene"].startswith(prefix)]))
        for tier, prefix in (("B", "development-"), ("C", "card-"))
        if any(r["scene"].startswith(prefix) for r in m["rows"])
    }) for m in training["validation"]]
    smoke = load(out/"old-smoke/ema.json")
    baseline = load(stage1/"evaluation/1b/step-030000/same-surface/ema.json")
    smoke_scene = smoke["development_rows"][0]["scene"]
    expected = [r for r in baseline["development_rows"] if r["scene"] == smoke_scene]
    if expected != smoke["development_rows"]:
        raise ValueError("Old checkpoint numerical reproduction changed after the smoke")
    reached = Counter(state["accepted"][name]["tier"] for name in training["sample_scenes"])
    report = dict(
        stage1_geometry_sha256=sha256(checkpoint), old_query_sha256=sha256(QUERY),
        new_query_sha256=sha256(fitted_query),
        scope=dict(scheme="13.16.3(4,6), 13.15 fixed row 3", geometry="frozen same-surface 1b 30k EMA",
                   benchmark="P2 f/1.4, same saved geometry/observations/F2 for both Query versions"),
        training={k: training[k] for k in ("steps", "seed", "seconds", "peak_allocated_MiB", "trainable_parameters",
                                          "sample_tiers", "sample_scenes", "curve", "GT_warmup_steps")},
        monitor_losses=loss, label_totals=data_summary(state), skipped_scenes=state["skipped"],
        sampled_unique_scenes=dict(reached),
        B_registered_exclusions=load(out/"data/inventory.json")["B_registered_exclusions"],
        prep=dict(seconds=state["seconds"], peak_allocated_MiB=state["peak_allocated_MiB"],
                  input="B native frozen depth sampled to 256, C native 256 frozen depth, 5 observed source frames",
                  labels="256/5 fields with B peeled color/identity and native K16 nearest-center labels; C peeled labels and exact analytic K16 recast with five views",
                  B_recoverability="five selected GT raster witness views, compared by surface identity and projected depth",
                  reveal_cases="two archived nominal f/1.4 focal distances per frame; no additional focal sweep",
                  training_change="old GT-input warmup omitted; exact balanced 70/30 interleaving from step one",
                  monitor="B 15 development scenes and 10 deterministic C development scenes"),
        optical_smoke=dict(scene=smoke_scene, exact=True, scored_cases=len(expected)),
        decision=decision, comparisons=comparison, first_pass=grid,
    )
    write_json(out/"result.json", report)
    write_json(out/"status.json", {"phase": "publishing", "decision": decision["selected"]})
    publish(report)
    write_json(out/"result.json", report)
    write_json(out/"status.json", {"phase": "complete", "decision": decision["selected"],
                                  "commit": subprocess.check_output(["git","rev-parse","--short","HEAD"],
                                                                    cwd=ROOT, text=True).strip()})


if __name__ == "__main__":
    main()
