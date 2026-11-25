"""Summarize and archive the v2 d-full comparison after the panel review."""
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from route3_v2data_B_plan import load
from route3_v2data_module2_d_labels import ROOT, RUN as V1, save_json
from route3_v2data_module2_d_report import compact_summary, display_report, gallery, pace
from route3_v2data_module2_d_v2_train import RUN
from route3_v2data_module2_infer import COMPETITOR, GROUPS
from route3_v2data_module2_report import fmt, pct, ZONE_NAMES

RECORD = ROOT/"03_实验记录/records/route3-v2data-module2-tear-d-v2-2026-11-25.json"
V1_RECORD = ROOT/"03_实验记录/records/route3-v2data-module2-tear-d-2026-11-25.json"
RESULTS = RUN/"evaluation"
METHODS = ("V0_pred", "v1", "v2a", "v2b", "b", "c_010")
COLUMNS = dict(zip(METHODS,(3,8,10,11,4,7)))
HEADS = ("v1", "v2a", "v2b")
HEADING = "模块二 d-full 第二版"


def observed_pace(train, optics):
    train_start = train["competitor_start"]
    end = train["competitor_end"]
    log = [line.split() for line in (COMPETITOR/"train.log").read_text().splitlines()
           if line.startswith("STAGE3B ")]
    earlier = next((p for p in log if int(p[1]) == train_start["step"]-225), None)
    reference = dict(step=int(earlier[1]),seconds=float(earlier[-1])) if earlier else {}
    smoke = load(RUN/"evaluation-smoke/report.json")
    optical_start = smoke["competitor_start"] if smoke else optics["competitor_start"]
    stop = optics["competitor_end"]
    return dict(before=pace(reference,train_start),training=pace(train_start,end),
                evaluation=pace(optical_start,stop), windows=dict(before=[reference,train_start],
                training=[train_start,end], evaluation=[optical_start,stop]),
                interpretation="同时有 B 光学在运行。训练窗口含给 3b 正式评价让行的时间。")


def train_curves(v1, train):
    setup = dict(v1=(v1,"d_full","0.40"),v2a=(train,"v2a","tab:blue"),v2b=(train,"v2b","tab:orange"))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    curves = {}
    for name,(run,key,color) in setup.items():
        items = run["history"]
        curves[name] = []
        for row in items:
            curved = dict(step=row["step"])
            for scope in ("training","development"):
                curved[scope] = row[scope][key]["at_half"]
            curves[name].append(curved)
        for scope,style in (("training","--"),("development","-")):
            osi = [row[scope]["weighted_error"]*100 for row in curves[name]]
            axes[0].plot([row["step"] for row in items], osi, style, color=color, label=f"{name} {scope}")
        selected = run["selected"][key]["step"]
        points = next(r["development"][key]["curve"] for r in items if r["step"]==selected)
        p,r = [[v[n] for v in points] for n in ("precision","recall")]
        axes[1].plot(r,p,".-",color=color,label=f"{name} · step {selected}")
        axes[1].scatter(r[4],p[4],color=color,marker="s",zorder=3)
    axes[0].set(xlabel="Training step", ylabel="Weighted edge error (%)")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Development PR (square: t > 0.5)")
    for ax in axes:
        ax.grid(alpha=.25)
        ax.legend(fontsize=8,frameon=False)
    fig.savefig(RESULTS/"edge-curves.png",dpi=160)
    plt.close(fig)
    return curves


def details(accuracy, far_stats, curves):
    lines = ["## 远焦逐场景汇总","",
             "| 划分 | 档 | 合并 PSNR | 逐场景 PSNR 中位 | 相对 V0 合并差 | 相对 V0 逐场景配对中位 |",
             "|---|---|---:|---:|---:|---:|"]
    for scope,methods in far_stats.items():
        for arm,v in methods.items():
            lines.append(f"| {scope} | {arm} | {fmt(v['pooled_db'])} | {fmt(v['scene_median_db'])} | "
                         f"{fmt(v['paired_to_V0'].get('pooled_delta_db'))} | {fmt(v['paired_to_V0'].get('median_db'))} |")
    lines.extend(["","## 撕裂专项","",
                  "| 档 | 区域 | GT 断边误连 | 假台阶误断 |","|---|---|---:|---:|"])
    for arm in HEADS:
        for zone in GROUPS:
            v = accuracy[arm][zone]
            lines.append(f"| {arm} | {ZONE_NAMES[zone]} | {pct(v['false_connected_fraction'])} "
                         f"({v['false_connected']}/{v['GT_torn']}) | {pct(v['false_step_torn_fraction'])} "
                         f"({v['false_step_torn']}/{v['false_step_connected']}) |")
    lines.extend(["","## 加权边误差","","[训练曲线与 PR](edge-curves.png)","",
                  "| 档 | 步数 | 训练 | 开发 |","|---|---:|---:|---:|"])
    for arm,rows in curves.items():
        for row in rows:
            lines.append(f"| {arm} | {row['step']} | {pct(row['training']['weighted_error'])} | "
                         f"{pct(row['development']['weighted_error'])} |")
    lines.extend(["", "[图像索引](index.html) · [观察清单](观察清单.md) · [逐 case 焦距](focus.csv)", ""])
    return lines


def report():
    train, optical, review = [load(p) for p in (RUN/"training-report.json",
                                RESULTS/"report.json", RESULTS/"visual-review.json")]
    previous = load(V1_RECORD)
    if not (train and optical and review and review.get("observations")):
        raise FileNotFoundError("The new heads, six-arm comparison and reviewed panels are required")
    accuracy = {"v1":previous["tear_accuracy"]["d_full"],
                **{arm:optical["diagnostics"]["tear_accuracy"][arm] for arm in HEADS[1:]}}
    summary = optical["summary"]
    far_stats = {}
    for scope, row in summary.items():
        far_stats[scope] = {}
        for arm in METHODS:
            points = [sc[arm]["far_recoverable"]["psnr_db"] for sc in row["per_scene"].values()]
            actual = [x for x in points if x is not None]
            far_stats[scope][arm] = dict(pooled_db=row["methods"][arm]["far_recoverable"]["psnr_db"],
                scene_median_db=float(np.median(actual)) if actual else None,
                paired_to_V0={k:v for k,v in row["paired"].get(arm,{}).get("far_recoverable",{}).items()
                              if k!="per_scene_db"})
    old_train = previous["training"]
    curves = train_curves(old_train,train)
    actual = dict(optical_wall_seconds=optical["seconds"],
                  optical_peak_reserved_gib=optical["peak_reserved_gib"],
                  optical_gpu_wait_seconds=optical["gpu_wait_seconds"],
                  optical_and_smoke_wall_seconds=(RESULTS/"report.json").stat().st_mtime
                       -datetime.fromisoformat(load(RUN/"training-progress.json")["at"]).timestamp())
    record = dict(configuration=optical["configuration"],training=train,
                  v1_curve=curves["v1"], summary=compact_summary(summary),
                  far_scene_summary=far_stats, tear_accuracy=accuracy,visual_review=review,
                  timing=actual, stage3b_observed_seconds_per_step=observed_pace(train,optical),
                  stage3_integration=False, independent_test_cases_used=0,
                  artifacts={k:str(RESULTS/file) for k,file in
                      (("report","report.json"),("summary","汇总.md"),("gallery","index.html"),
                       ("observations","观察清单.md"),("curves","edge-curves.png"),("focus","focus.csv"))})
    gallery(optical["cases"], [], review, output=RESULTS, columns=COLUMNS, heading=HEADING)
    (RESULTS/"汇总.md").write_text("\n".join(display_report(summary,METHODS,HEADING)+
                                          details(accuracy,far_stats,curves))+"\n")
    save_json(RECORD,record)


if __name__=="__main__":
    report()
