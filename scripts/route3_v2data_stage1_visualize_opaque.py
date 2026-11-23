"""Add evaluation-only b=0 opacity closure to the existing Stage 1 gallery."""
import argparse
from collections import defaultdict
from pathlib import Path
import time

import numpy as np
import torch

import route3_ovg_data as ovg
import route3_v2data_stage1_visualize as gallery
import route3_v2data_stage2_train as stage2
from route3_ovg_1b_common import CONFIG
from route3_ovg_stage2_query import FrozenQuery
from route3_upper_bound_evaluate import display
from route3_v2data_B_batch import GIB, reserve, sha256, write_json
from route3_v2data_B_plan import load
from route3_v2data_query_retrain import distribution
from route3_v2data_stage1_evaluate import load_paired_checkpoint, pooled, region_errors
from route3_v2data_stage2_eval import f2_for, k16, optical_target, oracle_hidden, peeled_truth, render_offsets
from src.aperture_renderer import aperture_disk
from src.layered_aperture import merge_layers
from src.ovg_opaque_eval import opaque_view, subpixel_flags
from src.ovg_optics import raster


COLUMNS = {"input": "输入 RGB", "GT": "GT · Cycles", "S3": "S3 · 新 Query",
           "S3o": "S3o · 新 Query + 不透明", "S4": "S4 · F2", "S5": "S5 · GT 隐藏层",
           "S5o": "S5o · GT 隐藏层 + 不透明"}
SYSTEMS = ("S3", "S3o", "S4", "S5", "S5o")
PAIRS = ("S5o", "S3o")
TILE = 252


def first_pass(plan, prediction, query, views, memory):
    frame = plan["frame"]
    cache = stage2.RUN/"observed-first/B"/plan["name"]/f"f{frame:03d}.npz"
    if cache.is_file():
        return ovg.load(cache, ("q",)), "stage2_observed_first_cache"
    z1, z2 = (prediction[k][0].cpu().numpy() for k in ("depth1", "depth2"))
    return query.first_from_views(views, memory, frame, z1, z2), "recomputed_from_observations"


@torch.inference_mode()
def frame_setup(plan, net, query, views, memory):
    x, ref, masks, subjects = gallery.frame_reference(plan)
    new = gallery.prepare_geometry(net, x, views, memory, plan["frame"], plan["cases"], subjects)
    depth = new["depth"]
    first, first_source = first_pass(plan, new["prediction"], query, views, memory)
    split = stage2.split_mask(new["prediction"], first)
    valid = torch.isfinite(depth) & (depth > 0) & torch.isfinite(x["rgb"][0]).all(0)
    b0 = valid & ~split
    flags = subpixel_flags(valid, b0, 5)
    visible_count = len(new["visible"].means)
    if flags.numel() != visible_count:
        raise ValueError("Subpixel flags do not match the fixed visible layers")
    return x, ref, masks, new, (depth, b0, flags, visible_count), float(split.float().mean()), first_source


def paired_transmission(visible, K, focus, offsets, opaque):
    ordinary = torch.zeros_like(opaque[0])
    closed = torch.zeros_like(ordinary)
    for part in offsets.split(4):
        residual = 1-raster(visible, K, focus, part, *ordinary.shape, return_alpha=True)[1]
        ordinary += residual.sum(0)
        for r, offset in zip(residual, part):
            _, diagnostics = opaque_view(visible, K, focus, offset, *opaque)
            closed += r.masked_fill(diagnostics["closed"], 0)
    return tuple((v/len(offsets)).clamp(0, 1).cpu().numpy() for v in (ordinary, closed))


@torch.inference_mode()
def render_case(plan, state, query, index, original):
    x, ref, masks, new, opaque, b1, first_source = state
    case, K = plan["cases"][index], x["K"][0]
    frame, unit, name = (plan[k] for k in ("frame", "unit", "name"))
    focus = original["actual_focus_m"]["S3"]
    if new["focuses"][index] is None or abs(new["focuses"][index]-focus) > 1e-6:
        raise ValueError("Frozen P2 focus differs from the original gallery")
    mask = masks[index]
    gt = optical_target(unit, name, case)
    offsets = aperture_disk(CONFIG["native_eval_rays"], .05/(2*case["fstop"]), device=K.device, dtype=K.dtype)
    fill = f2_for(unit, plan["entry"], frame, x, focus, offsets)
    visible = new["visible"]
    q_layer = gallery.query_layer(query.network, new, K, frame, case, index)
    gt_layer = oracle_hidden(peeled_truth(unit, name, frame), ref, new["depth"], K, focus, case["fstop"])[0]
    images = {"input": x["rgb"][0].permute(1, 2, 0)}
    for key, hidden in (("S3", q_layer), ("S5", gt_layer)):
        merged = merge_layers(visible, hidden)
        images[key] = render_offsets(merged, K, focus, offsets, fill)
        images[key+"o"] = render_offsets(merged, K, focus, offsets, fill, opaque)
    images["S4"] = render_offsets(visible, K, focus, offsets, fill)
    scores = {k: pooled([region_errors(rgb, gt, mask)]) for k, rgb in images.items()}
    scores["GT"] = pooled([dict(pixels=int(mask.sum()), sse=0.)])
    for key in ("input", "GT", "S3", "S4", "S5"):
        reference, actual = original["region_scores"][key], scores[key]
        if reference["pixels"] != actual["pixels"] or abs(reference["psnr_db"]-actual["psnr_db"]) > .001:
            raise ValueError(f"{key} rating differs from the original gallery")
    rgb = {key: display(v.cpu().numpy()) for key, v in images.items()}
    rgb["GT"] = gt
    lens = k16(unit, name, frame, case)["aperture_xy"]
    if lens.shape != (16, 2):
        raise ValueError(f"Expected 16 K16 pupil offsets, got {lens.shape}")
    transmission = paired_transmission(visible, K, focus, K.new_tensor(lens), opaque)
    info = {key: original[key] for key in ("file", "short", "scene", "frame", "focus", "case", "why",
                                           "region", "crop_xyxy", "crop_support")}
    focuses = dict(input=None, GT=original["actual_focus_m"]["GT"], **dict.fromkeys(SYSTEMS, focus))
    row = dict(info, region_scores=scores, actual_focus_m=focuses, b1_fraction=b1, first_pass_source=first_source,
               mean_transmittance={key: dict(full=float(v.mean()), region=float(v[mask].mean()))
                                   for key, v in zip(("S3", "S3o"), transmission)})
    maps = dict(zip(("S3 · 可见层透射率", "S3o · 不透明后透射率"), transmission))
    return row, rgb, maps


def statistics(rows):
    result = {}
    for group in ("development", "legacy"):
        result[group] = {}
        for region in ("near_in_focus", "far_recoverable"):
            per_scene = defaultdict(list)
            for row in rows:
                if row.get("error") or row["region"] != region:
                    continue
                is_development = row["short"].startswith("dev")
                if is_development == (group == "development"):
                    per_scene[row["scene"]].append(row)
            data = {}
            for key in SYSTEMS:
                case_scores = [r["region_scores"][key] for scene in per_scene.values() for r in scene]
                scores = {name: pooled([r["region_scores"][key] for r in cases]) for name, cases in per_scene.items()}
                values = [v["psnr_db"] for v in scores.values()]
                data[key] = dict(pooled=pooled(case_scores), per_scene=scores,
                                 median_db=float(np.median(values)) if values else None)
            pairs = {}
            for key in PAIRS:
                a, b = (data[k]["per_scene"] for k in (key, "S4"))
                diff = {name: a[name]["psnr_db"]-b[name]["psnr_db"] for name in per_scene}
                pairs[key+"-S4"] = dict(per_scene_db=diff, **distribution(list(diff.values())))
            result[group][region] = dict(scenes=len(per_scene), cases=sum(map(len, per_scene.values())),
                                         scores=data, pairs=pairs)
    return result


def db(value, *, signed=False):
    if value is None:
        return "—"
    return f"{value:+.3f}" if signed else f"{value:.3f}"


def summarize(output, rows, stats):
    md = ["# b=0 合成不透明对照", "",
          "同期使用同一 P2 实际焦距、f/1.4、F2、裁剪框和评分区域。"
          "S3o 与 S5o 按 §13.16.3 在命中的 b=0 可见面上关闭剩余透射。"
          "按阶段 2 的 ε=0.40、深度比 ≥1.03 且 q>0 划分 b。", "",
          "表内单位为 dB。绝对值按 RGB SSE 与像素数合并，逐场景中位数先在场景内合并。"
          "配对差的标准差按场景计算，方向数依次为正／负／零。", ""]
    for group, title in (("development", "新开发集"), ("legacy", "原三个场景")):
        near, far = (stats[group][r] for r in ("near_in_focus", "far_recoverable"))
        md.extend([f"## {title}", "",
                   f"| 系统或配对 | 近焦焦内（{near['scenes']} 场／{near['cases']} case） | "
                   f"远焦可恢复（{far['scenes']} 场／{far['cases']} case） |", "|---|---:|---:|"])
        for key in SYSTEMS:
            values = [f"{db(s['scores'][key]['pooled']['psnr_db'])} / {db(s['scores'][key]['median_db'])}"
                      for s in (near, far)]
            md.append(f"| {key}（合并／场景中位数） | {' | '.join(values)} |")
        for key in PAIRS:
            values = []
            for s in (near, far):
                pair = s["pairs"][key+"-S4"]
                values.append(f"{db(pair['median_db'], signed=True)} / {db(pair['sd_db'])} / "
                              f"{pair['positive']}／{pair['negative']}／{pair['tied']}")
            md.append(f"| {key}−S4（中位数／标准差／方向数） | {' | '.join(values)} |")
        md.append("")
    md += ["## 跳过的 case", ""]
    errors = [r for r in rows if r.get("error")]
    for row in errors:
        md.append(f"- {row['file']}，{row['error'].replace(chr(10), ' ')}")
    if not errors:
        md.append("无。")
    (output/"汇总.md").write_text("\n".join(md)+"\n", encoding="utf-8")


def run(source, pilot):
    stage2.configuration()
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(6*GIB/total, 0)
    torch.backends.cuda.matmul.allow_tf32 = False
    frozen = stage2.fixed_inputs()
    parent = source/"metrics.json"
    originals = load(parent)
    if (originals["checkpoints"]["new_1b"]["sha256"] != frozen["geometry_sha256"] or
            originals["checkpoints"]["new_Query"]["sha256"] != frozen["query_sha256"]):
        raise ValueError("Source gallery and Stage 2 frozen geometry or Query are different")
    output = source/"opaque"
    output.mkdir(exist_ok=True)
    path = output/"metrics.json"
    source_sha = sha256(parent)
    started = time.monotonic()
    report = load(path) or dict(source_metrics_sha256=source_sha, started_at=gallery.stamp(), seconds=0.,
                                threshold=frozen["rule"], epsilon=frozen["epsilon"],
                                scopes="P2 f/1.4, 64 pupil offsets; K16 mean visible transmittance before/after closure",
                                rows=[dict(r, inherited_skip=True) for r in originals["rows"] if r.get("error")])
    if report["source_metrics_sha256"] != source_sha:
        raise ValueError("Source gallery changed between runs")
    previous = report["seconds"]
    rows = {r["file"]: r for r in report["rows"]}
    base = {r["file"]: r for r in originals["rows"] if not r.get("error")}
    pilot_name = "019_dev001_f001_near.png"
    net = load_paired_checkpoint(stage2.INITIAL, "ema").eval()
    query = FrozenQuery(stage2.QUERY)
    scene, views, memory = None, None, None

    def publish():
        report["rows"] = sorted(rows.values(), key=lambda r: r["file"])
        report["seconds"] = previous+time.monotonic()-started
        report["completed_cases"] = sum(not row.get("error") for row in rows.values())
        report["skipped_cases"] = sum(bool(row.get("error")) for row in rows.values())
        report["peak_allocated_MiB"] = max(report.get("peak_allocated_MiB", 0),
                                           torch.cuda.max_memory_allocated()/2**20)
        write_json(path, report)

    for plan in gallery.planned_frames():
        pending = [(i, row["file"]) for i, row in enumerate(plan["files"])
                   if row["file"] in base and row["file"] not in rows and (not pilot or row["file"] == pilot_name)]
        if not pending:
            continue
        reserve(output, GIB)
        try:
            if scene != plan["name"]:
                views, memory = gallery.scene_views(plan)
                scene = plan["name"]
            state = frame_setup(plan, net, query, views, memory)
        except Exception as error:
            for _, name in pending:
                rows[name] = dict(base[name], error=f"{type(error).__name__}: {error}")
                print("OPAQUE_SKIP", name, rows[name]["error"], flush=True)
            torch.cuda.empty_cache()
        else:
            for index, name in pending:
                tick = time.monotonic()
                try:
                    row, rgb, maps = render_case(plan, state, query, index, base[name])
                    gallery.draw_panel(output/name, row, rgb, state[2], maps, columns=COLUMNS, tile=TILE)
                    rows[name] = dict(row, seconds=time.monotonic()-tick)
                    print("OPAQUE_DONE", name, round(rows[name]["seconds"], 2), flush=True)
                except Exception as error:
                    rows[name] = dict(base[name], error=f"{type(error).__name__}: {error}")
                    print("OPAQUE_SKIP", name, rows[name]["error"], flush=True)
                    torch.cuda.empty_cache()
        publish()
    if not pilot:
        report["summary"] = statistics(report["rows"])
        keys = ("GT", *SYSTEMS)
        for focus in ("near", "far"):
            gallery.overview(output, report["rows"], focus, columns=COLUMNS, keys=keys, tile=TILE)
        summarize(output, report["rows"], report["summary"])
        report["completed_at"] = gallery.stamp()
        publish()
    print("OPAQUE_TOTAL", report["completed_cases"], report["skipped_cases"], round(report["seconds"], 2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=gallery.OUT)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    run(args.source.resolve(), args.pilot)
