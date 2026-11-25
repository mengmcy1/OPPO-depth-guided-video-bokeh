"""Write the figure-linked Module 2 comparison without retaining large images in Git."""
import argparse
import csv
from html import escape
from pathlib import Path

import numpy as np
from PIL import Image

from route3_ovg_subpixel import ROOT, save_json
from route3_v2data_B_plan import load
from route3_v2data_module2_infer import (
    ARMS, ARM_NAMES, COMPETITOR, GROUPS, OUT, PAIR, THIN_SCENES, ZERO, acc_stats, score_psnr, title,
)


NAMES = dict(near_gt_minus_f2="近焦 GT 替身减 F2",far_recoverable="远焦可恢复",
             boundary_2px="边界 ±2 px",full_image="全图",
             zero_visible="零光圈仅可见层",zero_combined="零光圈完整",
             zero_visible_common="共同双层区仅可见",zero_combined_common="共同双层区完整")
RAY_LABEL = "近焦 K16 主体内部"
ZONE_NAMES = dict(foreground="杆或指定前景",gap_background="框内缝隙",other="其余")


def fmt(n,k=3):
    return "—" if n is None else f"{n:.{k}f}"


def pct(n):
    return "—" if n is None else f"{100*n:.3f}%"


def fmt_guard(value):
    text = fmt(value["psnr_db"])
    return text+(f'（{value["passed_cases"]}/{value["scored_cases"]}）' if value["passed_cases"] is not None else "")


def markdown(report):
    out = ["# 模块二只推理对照","",
           "各档使用同一输入颜色、0.999 不透明度与 0.14 px 的子高斯尺度。"
           "有光圈的光学指标按 64 条子光圈采样。近焦诊断另用标称焦距的 K16，限定在 "
           "GT 不透明主体内部及 16 条均命中同一可见物体的像素。"
           "有效隐藏权重与前后错序分档沿用 V0–V3 的正向 SSE 定义。",""]
    for cohort,entry in report["summary"].items():
        out.extend([f"## {cohort}","",
                    f'已评分 {entry["scored"]} 个 case，无同一对焦主体 {len(entry["unscored"])} 个。',"",
                    "| 档 | 近焦增量 | 远焦 | 边界 | 全图 | 仅可见过线 | 完整过线 | 双层区仅可见 | 双层区完整 | T ≥ 0.1 | 高 T 正向 SSE |",
                    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
        for arm in ARMS:
            v=entry["methods"][arm]
            vals = [fmt(v[k]["psnr_db"]) for k in ("near_gt_minus_f2","far_recoverable","boundary_2px","full_image")]
            vals += [fmt_guard(v[k]) for k in ZERO]
            vals += [pct(v["transmission"][k]) for k in ("high_fraction","high_positive_sse_share")]
            out.append("| "+" | ".join([ARM_NAMES[arm],*vals])+" |")
        out.extend(["","配对格为合并差 / 逐场景中位 / 样本标准差 / 正向场景数，单位 dB。","",
                    "| 档与同几何 V0 配对 | 指标 | 合并 / 中位 / 标准差 / 正向 |","|---|---|---:|"])
        for arm,comp in entry["paired"].items():
            for key,v in comp.items():
                out.append(f'| {ARM_NAMES[arm]} − {ARM_NAMES[PAIR[arm]]} | {NAMES[key]} | '
                           f'{fmt(v["pooled_delta_db"])} / {fmt(v["median_db"])} / {fmt(v["sd_db"])} / {v["positive"]}/{v["n"]} |')
        out.append("")
    out.extend(["## GT 撕裂标签","",
                "| 划分 | 边总数 | 预筛 | 连通 | 断开 | 无效 | 跨物体连通 |",
                "|---|---:|---:|---:|---:|---:|---:|"])
    for group,v in report["labels"]["totals"].items():
        col=[f'{v[k]:,}' for k in ("edges","prescreen")]
        col += [f'{v[k]:,}（{pct(v["fractions"][k])}）' for k in ("connected","torn","invalid")]
        out.append("| "+" | ".join([group,*col,f'{v["different_id_connected"]:,}'])+" |")
    for name,stats in report["diagnostics_by_cohort"].items():
        out.extend(["",f"## {name} · 撕裂误判与分级退化","",
                    "| 档 | 区域 | 误连 / GT 断开 | 误断 / GT 连通 |","|---|---|---:|---:|"])
        for arm,block in stats["tear_accuracy"].items():
            for zone in GROUPS:
                v=block[zone]
                out.append(f'| {ARM_NAMES[arm]} | {ZONE_NAMES[zone]} | '
                           f'{v["false_connected"]:,}/{v["GT_torn"]:,}（{pct(v["false_connected_fraction"])}） | '
                           f'{v["false_torn"]:,}/{v["GT_connected"]:,}（{pct(v["false_torn_fraction"])}） |')
        out.extend(["","| 区域 | b 的错位边 | GT 连通边 |","|---|---:|---:|"])
        for zone,v in stats["alignment"].items():
            out.append(f'| {ZONE_NAMES[zone]} | {v["edges"]:,} | {v["GT_connected"]:,} |')
        out.extend(["","| 档 | 区域 | 三角形 | 一维 | 常数退回 | 固定中心 |","|---|---|---:|---:|---:|---:|"])
        for arm,v in stats["interpolation"].items():
            for zone,r in v["zones"].items():
                f=r["fractions"]
                out.append("| "+" | ".join((ARM_NAMES[arm],ZONE_NAMES[zone],
                           *(pct(f[k]) for k in ("triangle","linear","constant","fixed_center"))))+" |")
    out.extend(["","[焦距逐 case CSV](focus.csv) · [面板](index.html) · [观察清单](观察清单.md)",""])
    return "\n".join(out)+"\n"


def panel_observation(path,row):
    # Evaluate the existing aligned zoom tile instead of comparing different crops.
    tile,gap=182,8
    top=76+2*(tile+25)
    with Image.open(path) as p:
        p.load()
        truth=np.asarray(p.crop((5,top+5,tile-5,top+tile-5)),dtype=np.float32)
        errors={}
        for i,arm in enumerate(ARMS,1):
            offset=i*(tile+gap)
            view=np.asarray(p.crop((offset+5,top+5,offset+tile-5,top+tile-5)),dtype=np.float32)
            errors[arm]=float(np.abs(view-truth).mean()/255.)
    key = "near_in_focus" if row["focus_index"]==0 else "far_recoverable"
    base=score_psnr(row["methods"]["V0_pred"]["optical"][key])
    ref="近焦焦内" if row["focus_index"]==0 else "远焦可恢复区"
    values="、".join(f'{ARM_NAMES[k]} {fmt(errors[k],4)}' for k in ("V0_pred","b","b_union","c_003","c_010"))
    out=f'同框边界放大块相对参考的显示 RGB 平均绝对误差为 {values}，V0 预测的{ref} PSNR 为 {fmt(base)} dB。'
    return out


def compact(report):
    small = {k:v for k,v in report.items() if k!="case_rows"}
    small["summary"]={group:{k:v for k,v in stats.items() if k!="per_scene"}
                      for group,stats in report["summary"].items()}
    small["focus_cases"]=[dict(scene=r["scene"],frame=r["frame"],case=r["case"],status=r["status"],
                               GT_focus_m=r["GT_focus_m"],nominal_focus_m=r["nominal_focus_m"],
                               actual_focus_m={arm:x["focus_m"] for arm,x in r["methods"].items()})
                          for r in report["case_rows"]]
    return small


def measured_pace(report):
    start,end = (report[k] for k in ("competitor_start","competitor_end"))
    log = [(int(v.split()[1]),float(v.split()[-1])) for v in (COMPETITOR/"train.log").read_text().splitlines()
           if v.startswith("STAGE3B ")]
    early = [p for p in log if p[0]<=start["step"]-250]
    ds = end["step"]-start["step"]
    during = (end["seconds"]-start["seconds"])/ds if ds>0 else None
    base = (start["seconds"]-early[-1][1])/(start["step"]-early[-1][0]) if early else None
    return dict(before_seconds_per_step=base,during_seconds_per_step=during,step_delta=ds,
                during_to_before=during/base if base and during else None,
                interpretation="Observed adjacent time windows with B optical production also active. This is not an isolated speed test.")


def make(root,record):
    source=root/"report.json"
    report=load(source)
    review=load(OUT/"full-visual-review.json")
    if not report or not review:
        raise ValueError("Complete the seven arms and open the representative panels before summarizing")
    frames=[load(p) for p in sorted((root/"frames").glob("*/*.json"))]
    scopes={k:[] for k in ("development","legacy","thin_structure","dev008")}
    for frame in frames:
        row=frame["rows"][0]
        scopes[row["cohort"]].append(frame)
        slug=title(row["scene"])
        if slug in THIN_SCENES:
            scopes["thin_structure"].append(frame)
        elif slug=="dev008":
            scopes["dev008"].append(frame)
    report["diagnostics_by_cohort"]={k:acc_stats(v) for k,v in scopes.items() if v}
    report["shared_training_pace"]=measured_pace(report)
    panels=[r for r in report["case_rows"] if r["status"]=="evaluated"]
    if len(panels)!=sum(report["summary"][k]["scored"] for k in ("development","legacy") if k in report["summary"]):
        raise ValueError("Visual index and paired case support differ")
    observations = ["# 模块二图像观察清单","",
                    "目视核对记录按固定框比较各档。逐 case 的裁块颜色差从已生成面板直接计算。",""]
    observations.extend([f"- {x['case']}。{x['observation']} [{x['group']}]({x['panel']})" for x in review["observations"]])
    observations.extend(["","## 逐 case 裁块",""])
    html=['<!doctype html><meta charset="utf-8"><title>模块二只推理对照</title>',
          '<h1>模块二只推理对照</h1><p>每行对比同焦点、同放大框与同一色标。</p>']
    for r in panels:
        path=Path(r["panel"])
        rel=path.relative_to(root)
        label=f'{title(r["scene"])} · {r["case"]}'
        note=panel_observation(path,r)
        observations.append(f"- {label}。{note} [对照图]({rel})")
        html.append(f'<figure><a href="{escape(str(rel))}"><img width="1200" loading="lazy" src="{escape(str(rel))}"></a>'
                    f'<figcaption>{escape(label)}。{escape(note)}</figcaption></figure>')
    (root/"index.html").write_text("\n".join(html)+"\n")
    (root/"观察清单.md").write_text("\n".join(observations)+"\n")
    (root/"汇总.md").write_text(markdown(report))
    with (root/"focus.csv").open("w",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(("cohort","scene","case","frame","nominal_focus_m","method","actual_focus_m","GT_focus_m"))
        for row in panels:
            for arm,v in row["methods"].items():
                writer.writerow((row["cohort"],row["scene"],row["case"],row["frame"],row["nominal_focus_m"],
                                 arm,v["focus_m"],v["GT_focus_m"]))
    if record:
        small=compact(report)
        small.update(artifact=str(source),visual_review=review,figures=str(root/"index.html"),
                     observations=str(root/"观察清单.md"),focus_csv=str(root/"focus.csv"),
                     whole_report=str(root/"汇总.md"),
                     label_visual_review=load(OUT/"label-visual-review.json"))
        save_json(record,small)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=OUT/"all-arms")
    p.add_argument("--write-record",action="store_true")
    a=p.parse_args()
    target=ROOT/"03_实验记录/records/route3-v2data-module2-infer-2026-11-25.json"
    make(a.root,target if a.write_record else None)
