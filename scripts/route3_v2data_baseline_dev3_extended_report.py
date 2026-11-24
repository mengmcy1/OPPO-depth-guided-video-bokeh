"""Matched-scene summaries and image panels for the expanded P2 baseline evaluation."""
from collections import defaultdict
import csv
from html import escape
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from route3_v2data_baseline_dev3 import ROOT, read, write
import route3_v2data_stage1_visualize as gallery
from route3_v2data_query_retrain import distribution
from route3_upper_bound_evaluate import score
from src.c1_bokehme import region_sse


LABELS = {"stage3":"阶段 3 起点", "stage3_f2":"起点 (F2)", "A":"A · SHARP",
          "R1_R2":"R1+R2", "DrBokeh":"Dr.Bokeh · GT 深度/α", "GT_oracle":"GT 几何 Oracle"}
METHODS = tuple(LABELS)
REGIONS = ("full_image","near_in_focus","far_recoverable","boundary_2px")
ZERO_FLOORS = {"zero_visible":50., "zero_combined":60.}


def percentile(value):
    return dict(zip(("p50","p95"),np.percentile(value,[50,95]).tolist())) if value.size else dict(p50=None,p95=None)


def box_for(image, truth, selected):
    if not selected.any():
        return (208,208,304,304)
    return gallery.largest_error_crop(image,truth,selected)


def draw_panel(path, row, state, truth, masks, predictions, coc):
    inputs = {"input":state["rgb"], "GT":truth}
    images = dict(inputs,**{name:r["rgb"] for name,r in predictions.items()})
    source = images.get("stage3",state["rgb"])
    boxes = (box_for(source,truth,masks["boundary_2px"]),
             box_for(source,truth,masks.get("near_in_focus",masks.get("far_recoverable")) & ~masks["boundary_2px"]))
    row["crop_xyxy"] = [list(b) for b in boxes]
    names = list(inputs)+list(METHODS)
    tile, gap, left = 192,10,14
    grid = 7 if coc else 5
    out = Image.new("RGB",(left*2+len(names)*(tile+gap)-gap,83+grid*(tile+29)),(18,23,31))
    draw = ImageDraw.Draw(out)
    mode = "近焦" if row["focus_index"]==0 else "远焦"
    draw.text((14,8),f'{row["scene"]} · f{row["frame"]:03d} · {mode} · f/{row["fstop"]:g}',font=gallery.font(25),fill="white")
    captions = ("全图","边界放大","区域放大","光学绝对误差 0–0.25","零光圈合成误差 0–0.05",
                "可见深度 · m","前景 CoC 误差 0–2 px")
    labels = dict(LABELS,input="输入 RGB",GT="Cycles LDR 参考")
    size = (tile,tile)
    top = float(np.percentile(state["gt"][state["known"]],99))
    row["depth_display_max_m"] = top
    for i,name in enumerate(names):
        x = left+i*(tile+gap)
        draw.text((x,49),labels[name],font=gallery.font(15),fill="white")
        if name not in images:
            draw.text((x,105),"该方法缺测",font=gallery.font(16),fill=(204,156,144))
            continue
        value = images[name]
        img = gallery.rgb_image(value)
        if name == "GT":
            img = img.copy()
            pen = ImageDraw.Draw(img)
            pen.rectangle(boxes[0],outline=(235,146,50),width=2)
            pen.rectangle(boxes[1],outline=(95,209,206),width=2)
        error = np.abs(value-truth).mean(-1)
        tiles = [img.resize(size,Image.Resampling.BOX), *[
            gallery.rgb_image(value).crop(b).resize(size,Image.Resampling.NEAREST) for b in boxes],
            gallery.scalar_image(error,.25,cv2.COLORMAP_INFERNO).resize(size)]
        layers = predictions.get(name,{})
        z = layers.get("zero_combined")
        tiles.append(None if z is None else
            gallery.scalar_image(np.abs(z-state["rgb"]).mean(-1),.05,cv2.COLORMAP_INFERNO).resize(size))
        if coc:
            depth = state["gt"] if name=="GT" else layers.get("depth")
            tiles.append(None if depth is None else gallery.scalar_image(
                np.where(np.isfinite(depth),depth,0),top,cv2.COLORMAP_VIRIDIS).resize(size))
            tiles.append(gallery.scalar_image(coc[name],2.,cv2.COLORMAP_INFERNO).resize(size) if name in coc else None)
        for j,box in enumerate(tiles):
            if box is None:
                continue
            top_y = 73+j*(tile+29)
            if i==0 or name=="GT":
                draw.text((x,top_y),captions[j],font=gallery.font(12),fill=(194,208,225))
            out.paste(box,(x,top_y+19))
    out.save(path,quality=95,subsampling=0)


def save_case(output, row, state, truth, masks, predictions, skips, lpips):
    case_scores, coc_maps, coc_samples = {}, {}, {}
    detail = row["category"]=="thin_structure" or row["scene"].startswith("development-008-")
    for name,values in predictions.items():
        metrics = score(values["rgb"],truth,masks,lpips)
        for label in ZERO_FLOORS:
            img = values[label]
            if img is not None:
                metrics[label] = region_sse(img,state["rgb"],masks["full_image"])
        extra = {}
        if detail:
            depth = values["depth"]
            valid = state["foreground"] & np.isfinite(depth) & (depth > 0) & (depth < 1e4)
            focus = values["focus"]
            factor = state["intr"][0,0]*.05/(2*row["fstop"])*focus/(focus-.05)
            error = np.zeros_like(depth,dtype=np.float32)
            error[valid] = np.abs(1/depth[valid].astype(np.float64)-1/state["main_gt"][valid])*factor
            extra["foreground_coc_px"] = dict(pixels=int(valid.sum()),gt_pixels=int(state["foreground"].sum()),
                                              **percentile(error[valid]))
            coc_maps[name] = error
            coc_samples[name] = error[valid]
        case_scores[name] = dict(metrics=metrics,focus_m=values["focus"],audit=values["audit"],**extra)
    short = row["scene"].split("-")[1]
    stem = f'dev{short}-{row["id"]}'
    row.update(scores=case_scores,skips=skips,subject_pixels=int(
        (state["subject"][next(i for i,c in enumerate(state["cases"]) if c["id"]==row["id"])]).sum()))
    if coc_samples:
        dest = output/"coc"
        dest.mkdir(exist_ok=True)
        np.savez_compressed(dest/(stem+".npz"),**coc_samples)
        row["coc_samples"] = f"coc/{stem}.npz"
    if not predictions:
        return
    panel = output/"panels"
    panel.mkdir(exist_ok=True)
    file = panel/(stem+".jpg")
    try:
        draw_panel(file,row,state,truth,masks,predictions,coc_maps)
    except (ValueError,OSError) as error:
        row["panel_failure"] = str(error)
    else:
        row["panel"] = str(file.relative_to(output))


def pooled(block):
    n = sum(row["pixels"] for row in block)
    sse = sum(row["sse"] for row in block)
    weighted = [r for r in block if r.get("patch_lpips") is not None and r["pixels"]]
    patch = sum(r["pixels"]*r["patch_lpips"] for r in weighted)/sum(r["pixels"] for r in weighted) if weighted else None
    return dict(pixels=n,sse=sse,psnr=float(-10*np.log10(max(sse/(3*n),1e-12))) if n else None,
                patch_lpips=patch)


def median(values):
    return float(np.median(values)) if len(values) else None


def entries(rows, method, metric):
    return [(row["scene"],row["scores"].get(method,{}).get("metrics",{}).get(metric))
            for row in rows]


def group_regions(rows, method, metric):
    if metric in ("near_in_focus","far_recoverable"):
        index = int(metric=="far_recoverable")
        rows = [r for r in rows if r["focus_index"]==index]
    grouped = defaultdict(list)
    before, after = defaultdict(list), defaultdict(list)
    active = [value for _,value in entries(rows,method,metric) if value and value["pixels"]]
    for (scene,value),(_,base) in zip(entries(rows,method,metric), entries(rows,"stage3",metric)):
        if value and value["pixels"]:
            grouped[scene].append(value)
            if base and base["pixels"]:
                if base["pixels"] != value["pixels"]:
                    raise ValueError("Methods used unequal scoring masks in a matched case")
                before[scene].append(base)
                after[scene].append(value)
    scene_rows = {name:pooled(block) for name,block in grouped.items()}
    abs_metrics = dict(psnr=median([r["psnr"] for r in scene_rows.values()]),
                       patch_lpips=median([r["patch_lpips"] for r in scene_rows.values() if r["patch_lpips"] is not None]))
    differences = {}
    for key in ("psnr","patch_lpips"):
        deltas = {}
        for scene in before:
            a, b = pooled(before[scene])[key], pooled(after[scene])[key]
            if a is not None and b is not None:
                deltas[scene] = b-a
        stats = distribution(list(deltas.values()))
        if key=="patch_lpips":
            stats = {k.removesuffix("_db"):v for k,v in stats.items()}
        differences[key] = dict(per_scene=deltas,**stats)
    summary = dict(scored=len(active),unscored=len(entries(rows,method,metric))-len(active),
                   combined=pooled(active),scene_median=abs_metrics,paired_vs_stage3=differences,
                   per_scene=scene_rows)
    if metric in ZERO_FLOORS:
        summary.update(floor_db=ZERO_FLOORS[metric],pass_cases=sum(r["psnr"]>=ZERO_FLOORS[metric] for r in active))
    return summary


def subgroup(output, rows, *, include_coc=True):
    result = {}
    coc = defaultdict(list)
    for row in rows:
        if include_coc and "coc_samples" in row:
            with np.load(output/row["coc_samples"]) as arrays:
                for name in arrays.files:
                    coc[name].append(arrays[name])
    for name in METHODS:
        result[name] = {metric:group_regions(rows,name,metric) for metric in (*REGIONS,*ZERO_FLOORS)}
        if coc[name]:
            depths = np.concatenate(coc[name])
            result[name]["foreground_coc_px"] = dict(pixels=len(depths),**percentile(depths))
    return result


def legacy_sources():
    records = ROOT/"03_实验记录/records"
    prior = ROOT.parent/"route3-query-v2-20261123/outputs/route3-query-v2-20261123/new-query/ema.json"
    references = dict(
        stage3=dict(file=str(prior),scope="P2，1b 30k + 新 Query，原三场景，仅 f/1.4"),
        stage3_f2=dict(file=str(prior),scope="P2，同一归档的 Query 关闭配对，仅 f/1.4"),
        A=dict(file=str((records/"route3-ovg-1c-2026-11-19-C3-optical.csv").relative_to(ROOT)),
               scope="原三场景 P2，SHARP + A-color"),
        R1_R2=dict(file=str((records/"route3-g3-r15-2026-11-17-R3-per-config.csv").relative_to(ROOT)),
                   scope="原三场景 P2，历史原几何 R1+R2，归档掩码"),
        DrBokeh=dict(file=str((records/"route3-baseline-dev3-2026-11-20-auxiliary.csv").relative_to(ROOT)),
                     scope="原三场景 P1，1c 深度与 LDF 分割，仅 f/1.4。新场景用 B2 Oracle 输入"),
        GT_oracle=dict(file=str((records/"route3-ovg-1c-2026-11-19-C3-optical.csv").relative_to(ROOT)),
                       scope="原三场景 P2，GT 可见几何与 GT 隐藏层归档"),
    )
    combined = defaultdict(list)

    def put(stop, method, region, row):
        if row["pixels"]:
            combined[(stop,method,region)].append(row)

    for row in read(prior)["legacy_rows"]:
        if row["status"]=="evaluated":
            for region, scores in row["query_pair"].items():
                put(row["fstop"],"stage3",region,scores["on"])
                put(row["fstop"],"stage3_f2",region,scores["off"])

    def archived_rows(method):
        with (ROOT/references[method]["file"]).open() as f:
            return list(csv.DictReader(f))

    def score_row(row):
        return dict(pixels=int(row["pixels"]),sse=float(row["sse"]),
                    patch_lpips=float(row["patch_lpips"]) if row["patch_lpips"] else None)

    def region_for(name, focus):
        if name=="in_focus":
            return "near_in_focus" if focus==1.4 else None
        if name=="recoverable":
            return "far_recoverable" if focus==5. else None
        return name if name in REGIONS else None

    for method, old_name, hidden in (("A","A-SHARP-A-color","standalone"),
                                      ("GT_oracle","GT-visible-reference","GT")):
        for row in archived_rows(method):
            if (row["scene"].startswith("metric-scene-") and row["protocol"]=="P2"
                    and row["method"]==old_name and row["hidden"]==hidden):
                metric = region_for(row["region"],float(row["名义对焦距离_m"]))
                if metric:
                    put(float(row["aperture"]),method,metric,score_row(row))
    for row in archived_rows("R1_R2"):
        if (row["protocol"],row["geometry"],row["method"])==("P2","original","R1_R2"):
            metric = region_for(row["region"],float(row["focus"]))
            if metric and int(row["frame"]) in (1,30,60):
                put(float(row["fstop"]),"R1_R2",metric,score_row(row))
    for row in archived_rows("DrBokeh"):
        if (row["scene"].startswith("metric-scene-") and row["protocol"]=="P1"
                and row["method"]=="drbokeh" and row["arm"]=="1c-depth" and float(row["aperture"])==1.4):
            put(1.4,"DrBokeh",row["region"],score_row(row))
    results = {f"f{stop:g}":{method:{area:pooled(combined[(stop,method,area)])
                          for area in REGIONS} for method in METHODS} for stop in (1.4,2.)}
    return dict(references=references, pooled=results)


def fmt(value, digits=2):
    return "—" if value is None else f"{value:.{digits}f}"


def review_set(rows, report, fixed):
    cases = {(r["scene"].split("-")[1],r["frame"],r["focus_index"]):r
             for r in rows if r["fstop"]==1.4 and "panel" in r}
    selected = {cases[tuple(k)]["panel"]:"固定清单" for k in fixed if tuple(k) in cases}
    for area,focus_index in (("near_in_focus",0),("far_recoverable",1)):
        delta = report["f1.4"]["A"][area]["paired_vs_stage3"]["psnr"]["per_scene"]
        order = sorted(delta,key=delta.get)
        for scene in dict.fromkeys(order[:2]+order[-2:]):
            key = (scene.split("-")[1],30,focus_index)
            if key in cases:
                panel = cases[key]["panel"]
                if panel not in selected:
                    selected[panel] = f'A 相对起点场景差值 {delta[scene]:+.2f} dB'
    return selected


def make_tables(summary):
    lines = []
    for tag,data in summary.items():
        lines += [f"## {tag}","",
            "| 方法 | 全图 PSNR | 全图 patch LPIPS | 近焦焦内 | 远焦可恢复 | 边界 ±2 px | 零光圈仅可见 | 零光圈完整 | 前景 CoC p50/p95 (px) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for name,values in data.items():
            def metric(region,key):
                item = values[region]
                return "/".join(fmt(x,4 if key=="patch_lpips" else 2)
                                for x in (item["combined"][key],item["scene_median"][key]))
            def zero(region):
                v=values[region]
                return f"{metric(region,'psnr')} ({v['pass_cases']}/{v['scored']})" if v["scored"] else "—"
            entries = [LABELS[name],metric("full_image","psnr"),metric("full_image","patch_lpips"),
                       *[metric(r,"psnr") for r in REGIONS[1:]],*map(zero,ZERO_FLOORS)]
            coco = values.get("foreground_coc_px")
            entries.append("/".join(fmt(coco[k]) for k in ("p50","p95")) if coco else "—")
            lines.append("| "+" | ".join(entries)+" |")
        lines.extend(["","| 方法 | 全图 PSNR 差 | 全图 patch LPIPS 差 | 近焦差 | 远焦差 | 边界差 |",
                           "|---|---:|---:|---:|---:|---:|"])
        for name,values in data.items():
            if name=="stage3":
                continue
            items = [(region,"psnr") for region in REGIONS]
            items.insert(1,("full_image","patch_lpips"))
            cells = [LABELS[name]]
            for region,key in items:
                p = values[region]["paired_vs_stage3"][key]
                suffix = "_db" if key=="psnr" else ""
                good = p["positive"] if key=="psnr" else p["negative"]
                cells.append(f'{fmt(p["median"+suffix],4)} / {fmt(p["sd"+suffix],4)} / {good}/{p["n"]}')
            lines.append("| "+" | ".join(cells)+" |")
        lines.append("")
    return lines


def finish(output, seconds):
    protocol = read(output/"protocol.json")
    rows = [read(path) for path in sorted((output/"cases").glob("*/*.json"))]
    converted = {}
    for stop in protocol["stops"]:
        active = [r for r in rows if r["fstop"]==stop]
        converted[f"f{stop:g}"] = subgroup(output,active,include_coc=False)
        converted[f"f{stop:g} 细结构"] = subgroup(output,[r for r in active if r["category"]=="thin_structure"])
        converted[f"f{stop:g} development-008"] = subgroup(output,[r for r in active
                                                                     if r["scene"].startswith("development-008-")])
    fixed = review_set(rows,converted,protocol["fixed_review"])
    summary = dict(protocol=protocol,seconds=seconds,rows=len(rows),
                   case_panels=sum("panel" in r for r in rows),summaries=converted,
                   review_cases=fixed,legacy=legacy_sources(),
                   skips=[dict(scene=r["scene"],case=r["id"],methods=r["skips"]) for r in rows if r["skips"]])
    write(output/"summary.json",summary)
    text = ["# 阶段 3 起点外部基线 · 新开发集","",
            "全图及分区列为合并值／逐场景中位数。配对列为相对起点的逐场景中位数／样本标准差／向优场景数。"
            "PSNR 向优为上升，patch LPIPS 向优为下降。Dr.Bokeh 使用 B2 的 GT 深度与覆盖率。"
            "零光圈适用时列合并 PSNR、逐场景中位数和过线数，Dr.Bokeh 没有独立可见层。"
            "CoC 为前景有效深度的像素级分位数，统一采用阶段 1 的逆深度绝对误差乘 CoC 系数。",
            "",f"{len(rows)} 个计划 case 已登记，出图 {summary['case_panels']} 张。",*make_tables(converted),
            "## 原 3 场景的历史引用",
            "",
            "以下为原归档的分区口径。Dr.Bokeh 行使用 1c 深度、LDF 分割与 P1。空白代表该档无对应结果。",
            "", "| 方法 | 光圈 | 全图 PSNR | 全图 patch LPIPS | 近焦焦内 | 远焦可恢复 | 边界 ±2 px |",
            "|---|---|---:|---:|---:|---:|---:|"]
    for stop,methods in summary["legacy"]["pooled"].items():
        for method,data in methods.items():
            label = "Dr.Bokeh · 1c 深度 P1" if method=="DrBokeh" else LABELS[method]
            values = (fmt(data["full_image"]["psnr"]),fmt(data["full_image"]["patch_lpips"],4),
                      *[fmt(data[r]["psnr"]) for r in REGIONS[1:]])
            text.append("| "+" | ".join((label,stop,*values))+" |")
    text.append("")
    text.extend(f'- {line["scope"]} · {line["file"]}' for line in summary["legacy"]["references"].values())
    text.append("- [连续性参考图 · 122104 f030 近焦](../visual-20261123/003_122104_f030_near.png)")
    text.extend(["","## 图像观察清单",""])
    text.extend(f"- [{Path(name).stem}]({name}) · {tag}" for name,tag in fixed.items())
    text.extend(["","缺测条目保留在 [summary.json](summary.json)。"])
    (output/"汇总.md").write_text("\n".join(text)+"\n")
    html = ['<!doctype html><meta charset="utf-8"><title>阶段 3 起点外部基线</title>',
            '<h1>阶段 3 起点外部基线</h1><p><a href="汇总.md">汇总与观察清单</a></p>',
            '<p>图中各行依次为全图、边界放大、区域放大、绝对误差（0–0.25）、零光圈合成误差（0–0.05）。'
            '细结构追加可见深度和前景 CoC 误差（0–2 px）。</p>',
            '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,35rem),1fr));gap:1rem">']
    html.extend(f'<figure style="margin:0"><a href="{escape(r["panel"])}"><img loading="lazy" '
                f'width="100%" src="{escape(r["panel"])}"></a><figcaption>'
                f'{escape(r["scene"])} · {escape(r["id"])}</figcaption></figure>'
                for r in rows if "panel" in r)
    (output/"index.html").write_text("\n".join(html+["</div>"])+"\n")
