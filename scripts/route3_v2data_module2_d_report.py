"""Publish compact evidence for the independently evaluated learned tear heads."""
import csv
from datetime import datetime
from html import escape
from pathlib import Path

import numpy as np
from PIL import Image

from route3_v2data_B_plan import load
from route3_v2data_module2_d_evaluate import ARMS, LEARNED, RESULTS
from route3_v2data_module2_d_labels import CONFIG, ROOT, RUN, save_json
from route3_v2data_module2_infer import COMPETITOR, score_psnr, title
from route3_v2data_module2_report import fmt, fmt_guard, pct, ZONE_NAMES

RECORD = ROOT/"03_实验记录/records/route3-v2data-module2-tear-d-2026-11-25.json"
KEYS = ("near_in_focus", "near_gt_minus_f2", "far_recoverable", "boundary_2px", "full_image")
DISPLAY = ("V0_pred", "b", "c_010", *LEARNED)


def label_counts(labels, timing):
    frames = [f for row in labels["rows"] for f in row["frames"]]
    categories = sorted({row["category"] for row in labels["rows"]})
    counts = [n for f in frames for n in f["directions"]]
    totals = {key: sum(v[key] for v in counts) for key in counts[0]}
    totals["fractions"] = {key: totals[key]/totals["edges"] for key in ("connected", "torn", "invalid")}
    return dict(training_scenes=len(labels["rows"]), frames=len(frames), totals=totals,
                allocation={c: sum(row["category"]==c for row in labels["rows"]) for c in categories},
                protocol=labels["protocol"], excluded=labels["excluded"], workers=labels["workers"],
                cached_final_pass_seconds=labels["seconds"], measured_generation=timing,
                training_visual_review=load(RUN/"training-label-visual-review.json"))


def pace(a, b):
    first, last = a.get("step"), b.get("step")
    if first is None or last is None or last <= first:
        return None
    return (b["seconds"]-a["seconds"])/(last-first)


def competitor_pace(train, render):
    baseline_end = train["competitor_start"]
    log = [s.split() for s in (COMPETITOR/"train.log").read_text().splitlines() if s.startswith("STAGE3B ")]
    earlier = [row for row in log if int(row[1]) == baseline_end["step"]-225]
    reference = dict(step=int(earlier[-1][1]), seconds=float(earlier[-1][-1])) if earlier else {}
    smoke = load(RUN/"evaluation-smoke/report.json")
    optical_start = smoke["competitor_start"] if smoke else render["competitor_start"]
    return dict(baseline=pace(reference, baseline_end),
                features_and_training=pace(baseline_end, train["competitor_end"]),
                optical_evaluation=pace(optical_start, render["competitor_end"]),
                windows=dict(baseline=[reference, baseline_end],
                             training=[baseline_end, train["competitor_end"]],
                             optical=[optical_start, render["competitor_end"]]),
                interpretation="3b 自身进度日志的相邻时间窗，B 光学渲染在各窗共同运行。差值是共同运行时的观测变化。")


def display_report(summary, arms=ARMS, heading="模块二学习档 (d)"):
    lines = [f"# {heading}","",
             "P2、f/1.4、K64 的光学输出沿用七臂口径。透射与正向 SSE 用标称焦距下的 K16 真值光线。"
             "PSNR 合并按 SSE ÷ (3 × 像素数)。",""]
    for scope, row in summary.items():
        lines.extend([f"## {scope} · {row['scored']} 个有效 case","",
                      "| 档 | 近焦 | 替身−F2 | 远焦 | 边界 ±2 px | 全图 | 零光圈仅可见 | 零光圈完整 | T≥0.1 占比 | T≥0.1 SSE | 0.02≤T<0.1 SSE | front_order SSE |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
        for method in arms:
            rec = row["methods"][method]
            ray = rec["transmission"]
            v = [fmt(rec[k]["psnr_db"]) for k in KEYS]
            v.extend(fmt_guard(rec[k]) for k in ("zero_visible","zero_combined"))
            v.append(pct(ray["high_fraction"]))
            v.extend(fmt(ray["positive_sse_bins"].get(k,0.))
                     for k in ("T>=0.1","0.02<=T<0.1","front_order"))
            lines.append("| "+" | ".join([method,*v])+" |")
        lines.extend(["","| 档 | 指标 | 合并差 | 逐场景中位 | SD | 正向/全部 |",
                      "|---|---|---:|---:|---:|---:|"])
        for method, pair in row["paired"].items():
            for key in KEYS:
                v = pair[key]
                lines.append(f"| {method} − V0 | {key} | {fmt(v['pooled_delta_db'])} | "
                             f"{fmt(v['median_db'])} | {fmt(v['sd_db'])} | {v['positive']}/{v['n']} |")
        lines.extend(["","| 场景 | " + " | ".join(f"{arm} 远焦" for arm in arms) + " |",
                      "|---|" + "---:|"*len(arms)])
        for scene, values in row["per_scene"].items():
            cols = [fmt(values[arm]["far_recoverable"]["psnr_db"]) for arm in arms]
            lines.append("| "+" | ".join([title(scene), *cols])+" |")
        lines.append("")
    return lines


def edge_curves(train):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    steps = [p["step"] for p in train["history"]]
    for method, color in zip(LEARNED, ("tab:blue", "tab:orange")):
        label = method.replace("_","-")
        for scope, style in (("training","--"),("development","-")):
            values = [p[scope][method]["at_half"]["weighted_error"] for p in train["history"]]
            axes[0].plot(steps, np.array(values)*100, style, color=color, label=f"{label} {scope}")
        step = train["selected"][method]["step"]
        chosen = next(p["development"][method] for p in train["history"] if p["step"]==step)
        p, r = [[v[key] for v in chosen["curve"]] for key in ("precision","recall")]
        axes[1].plot(r, p, ".-", color=color, label=f"{label} · step {step}")
        axes[1].scatter(r[4], p[4], color=color, marker="s", zorder=3)
    axes[0].set(xlabel="Training step", ylabel="Weighted edge error (%)")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Development PR (square: t > 0.5)")
    for ax in axes:
        ax.grid(alpha=.25)
        ax.legend(fontsize=8, frameon=False)
    fig.savefig(RESULTS/"edge-curves.png", dpi=160)
    plt.close(fig)

    lines = ["## 边级训练曲线","","[加权边误差与开发集 PR 曲线](edge-curves.png)","",
             "| 步数 | d-full 训练 / 开发 | d-nodepth 训练 / 开发 |",
             "|---|---:|---:|"]
    for row in train["history"]:
        values = [" / ".join(pct(row[scope][method]["at_half"]["weighted_error"])
                             for scope in ("training","development")) for method in LEARNED]
        lines.append(f"| {row['step']} | {' | '.join(values)} |")
    lines.append("")
    return lines


def extra_tables(record):
    lines = ["## 撕裂专项","", "| 档 | 区域 | 假台阶误断 | GT 断边误连 |",
             "|---|---|---:|---:|"]
    for method in LEARNED:
        for zone,v in record["tear_accuracy"][method].items():
            lines.append(f"| {method} | {ZONE_NAMES[zone]} | {v['false_step_torn']}/{v['false_step_connected']} "
                         f"({pct(v['false_step_torn_fraction'])}) | {v['false_connected']}/{v['GT_torn']} "
                         f"({pct(v['false_connected_fraction'])}) |")
    lines.extend(["","## dev005 远焦缝隙","",
                  "T 是隐藏高斯合成权重平均深度处的可见层前缀透射。未命中隐藏层的光线在全组平均中按零处理。"
                  "另记命中条件均值以及可见层末端剩余透射。RGB 是隐藏层在线性合成中的直接颜色贡献。","",
                  "| 帧 | 档 | 缝隙像素 | 有效隐藏命中 | 平均 T 全组 | 平均 T 有命中 | 末端 T | 隐藏层线性 RGB |",
                  "|---|---|---:|---:|---:|---:|---:|---|"])
    for frame in record["dev005_far_gaps"]:
        for method,v in frame["methods"].items():
            color = "/".join(fmt(i,6) for i in v["mean_hidden_linear_RGB_contribution"])
            lines.append(f"| f{frame['frame']:03d} | {method} | {v['gap_pixels']} | {pct(v['hidden_hit_fraction'])} | "
                         f"{fmt(v['mean_visible_prefix_T_hidden_unreached_as_zero'],6)} | "
                         f"{fmt(v['mean_visible_prefix_T_given_hidden_hit'],6)} | "
                         f"{fmt(v['mean_visible_residual_T'],6)} | {color} |")
    lines.extend(["","[逐 case 焦距](focus.csv) · [对照图](index.html) · [观察清单](观察清单.md)",""])
    return lines


def crop_observation(row, columns=None):
    region = "near_in_focus" if row["focus_index"]==0 else "far_recoverable"
    tile,pitch,top = 182,190,76+207
    with Image.open(row["panel"]) as fig:
        im = np.array(fig, dtype=np.float32)/255
    truth = im[top:top+tile, :tile]
    values = []
    for arm,i in (columns or dict(zip(DISPLAY, (3,4,7,8,9)))).items():
        offset = i*pitch
        image = im[top:top+tile, offset:offset+tile]
        difference = float(np.abs(image-truth).mean())
        db = score_psnr(row["methods"][arm]["optical"][region])
        values.append(f"{arm} {difference:.4f} / {fmt(db)}")
    return ("目标区放大框显示 RGB MAE / 该 case 区域 PSNR (dB) 为 "+"、".join(values)+"。")


def gallery(cases, gaps, review, *, output=RESULTS, columns=None, heading="模块二学习档 (d)"):
    doc = [f"# {heading} 图像观察清单",""]
    doc += [f"- {row['observation']} [{row['case']}]({row['panel']})" for row in review["observations"]]
    doc += ["","## 逐 case 同框读数",""]
    html = [f"<!doctype html><meta charset='utf-8'><title>{escape(heading)}</title>",
            f"<h1>{escape(heading)}</h1>"]
    with (output/"focus.csv").open("w", newline="") as f:
        out = csv.writer(f)
        out.writerow(("cohort","scene","frame","case","arm","P2_focus_m","GT_focus_m"))
        for row in cases:
            if row["status"]!="evaluated":
                continue
            rel = Path(row["panel"]).relative_to(output)
            label = f"{title(row['scene'])} · {row['case']}"
            note = crop_observation(row,columns)
            doc.append(f"- {label}。{note} [面板]({rel})")
            html.append(f"<figure><a href='{escape(str(rel))}'><img width='1300' loading='lazy' "
                        f"src='{escape(str(rel))}'></a><figcaption>{escape(label)}。"
                        f"{escape(note)}</figcaption></figure>")
            for arm,v in row["methods"].items():
                out.writerow((row["cohort"],row["scene"],row["frame"],row["case"],arm,v["focus_m"],row["GT_focus_m"]))
    for frame in gaps:
        if "panel" in frame:
            rel = Path(frame["panel"]).relative_to(output)
            html.append(f"<figure><a href='{rel}'><img width='876' loading='lazy' src='{rel}'></a>"
                        f"<figcaption>{escape(frame['case'])} · 远焦缝隙</figcaption></figure>")
    (output/"观察清单.md").write_text("\n".join(doc)+"\n")
    (output/"index.html").write_text("\n".join(html)+"\n")


def compact_summary(summary):
    small = {}
    for scope, row in summary.items():
        result = {k:v for k,v in row.items() if k!="per_scene"}
        result["per_scene_psnr_db"] = {
            scene:{arm:{metric:values[metric]["psnr_db"] for metric in KEYS}
                   for arm,values in methods.items()}
            for scene,methods in row["per_scene"].items()
        }
        small[scope] = result
    return small


def report():
    evaluation = load(RESULTS/"report.json")
    train = load(RUN/"training-report.json")
    review = load(RESULTS/"visual-review.json")
    if not evaluation or not train or not review:
        raise FileNotFoundError("Complete training and rendering, then open the fixed diagnostic panels")
    config = load(CONFIG)
    summary = evaluation["summary"]
    t = summary["development"]["methods"]
    delta_b = t["b"]["near_in_focus"]["psnr_db"]-t["V0_pred"]["near_in_focus"]["psnr_db"]
    high = {name: r["transmission"]["positive_sse_bins"]["T>=0.1"] for name,r in t.items()}
    optical_recovery = {arm: dict(
        near_fraction=(t[arm]["near_in_focus"]["psnr_db"]-t["V0_pred"]["near_in_focus"]["psnr_db"])/delta_b,
        high_T_SSE_fraction=(high["V0_pred"]-high[arm])/(high["V0_pred"]-high["b"])) for arm in LEARNED}
    label_report = label_counts(load(RUN/"training-labels.json"), load(RUN/"training-label-timing.json"))
    cases = evaluation["cases"]
    timings = {key:evaluation[key] for key in ("seconds","gpu_wait_seconds","peak_reserved_gib","peak_allocated_gib")}
    done = datetime.fromisoformat(load(RUN/"training-progress.json")["at"]).timestamp()
    timings["optical_pipeline_wall_after_training_seconds"] = (RESULTS/"report.json").stat().st_mtime-done
    timings["optical_pipeline_includes"] = "冒烟、首次评价、从缓存续跑和远焦缝隙诊断"
    record = dict(configuration=evaluation["configuration"], labels=label_report, training=train,
        summary=compact_summary(summary), optical_recovery=optical_recovery, tear_accuracy=evaluation["diagnostics"]["tear_accuracy"],
        interpolation=evaluation["diagnostics"]["interpolation"], dev005_far_gaps=evaluation["gap"],
        timing=timings, recovery=load(RUN/"evaluation-recovery.json"),
        stage3b_observed_seconds_per_step=competitor_pace(train,evaluation), visual_review=review,
        focus_cases=[dict(scene=r["scene"],case=r["case"],frame=r["frame"],status=r["status"],
                          gt_focus_m=r["GT_focus_m"],
                          actual_focus_m={arm:m["focus_m"] for arm,m in r["methods"].items()}) for r in cases],
        artifacts=dict(full_report=str(RESULTS/"report.json"), gallery=str(RESULTS/"index.html"),
                       markdown=str(RESULTS/"汇总.md"), observations=str(RESULTS/"观察清单.md"),
                       focus_csv=str(RESULTS/"focus.csv"),edge_curves=str(RESULTS/"edge-curves.png")),
        independent_test_cases_used=0, stage3_integration=False, followup=config["retrain_after_E"])
    gallery(cases, evaluation["gap"], review)
    (RESULTS/"汇总.md").write_text("\n".join(display_report(summary)+edge_curves(train)+extra_tables(record))+"\n")
    save_json(RECORD, record)


if __name__ == "__main__":
    report()
