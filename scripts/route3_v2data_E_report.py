"""Publish the compact E preparation evidence and local panel index."""
from datetime import datetime
from html import escape

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from route3_v2data_B_inputs import FORMAL_DATA
from route3_v2data_E_cache import cache_path
from route3_v2data_E_labels import REGIONS, ROOT, RUN, load, save_json
from route3_v2data_module2_infer import COMPETITOR, PARENTS
from route3_v2data_stage1_evaluate import pooled

RECORD=ROOT/"03_实验记录/records/route3-v2data-E-prep-2026-11-26.json"


def fmt(value,digits=4):
    return "—" if value is None else f"{value:.{digits}f}"


def pct(value):
    return "—" if value is None else f"{100*value:.2f}%"


def error(v):
    return f"{fmt(v['p50'])}／{fmt(v['p95'])}"


def pace(a,b):
    first,last=a.get("step"),b.get("step")
    return (b["seconds"]-a["seconds"])/(last-first) if first is not None and last is not None and last>first else None


def schedules(phases):
    first=phases[0][1]["competitor_start"]
    rows=[(int(p[1]),float(p[-1])) for line in (COMPETITOR/"train.log").read_text().splitlines()
          if line.startswith("STAGE3B ") for p in (line.split(),)]
    stop=first.get("step")
    previous=next((dict(step=step,seconds=sec) for step,sec in rows if stop and step==stop-200),None)
    reference=pace(previous,first) if previous else None
    intervals={}
    for phase,item in phases:
        before,after=(item[k] for k in ("competitor_start","competitor_end"))
        intervals[phase]=dict(seconds_per_step=pace(before,after),before=before,after=after)
    return dict(baseline_seconds_per_step=reference,before=previous,anchor=first,phases=intervals,
                stage3c_sampled=False,interpretation="同期有 B 光学运行。这里给出 3b 日志相邻时间窗的实际步时。")


def render_stats(panels):
    results={}
    for group in ("training","development"):
        group_rows=[p for p in panels if p["cohort"]==group]
        bundles={}
        for focus_index,tag in enumerate(("near","far")):
            cases=[r for p in group_rows for r in p["optical"] if r["focus_index"]==focus_index]
            bundles[tag]={method:{key:pooled([case["scores"][method][key] for case in cases])
                          for key in ("full","focus_region","rod","gap")} for method in ("OVG","E","GT")}
        results[group]=bundles
    return results


def plot_curves(tr):
    x=[r["step"] for r in tr["curves"]]
    y=[r["train_loss"] for r in tr["curves"]]
    best=[r["best"] for r in tr["curves"]]
    fig,ax=plt.subplots(figsize=(7,4),layout="constrained")
    ax.plot(x,y,label="Native training objective",marker=".")
    ax.plot(x,best,label="Running best",alpha=.7)
    ax.axvline(tr["best_step"],color="gray",linestyle=":")
    ax.set(xlabel="Training step",ylabel="Loss",title="E-L memorization check")
    ax.grid(alpha=.2)
    ax.legend(frameon=False)
    fig.savefig(RUN/"overfit-curve.png",dpi=155)
    plt.close(fig)


def gallery(panels,notes):
    md=["# E 准备检查观察清单",""]
    html=["<!doctype html><meta charset='utf-8'><title>E-L 准备检查</title>","<h1>E-L 准备检查</h1>"]
    for row in panels:
        name=row["scene"]
        paths=[row["depth_panel"],*(c["panel"] for c in row["optical"])]
        note=notes[name]
        links=" · ".join(f"[{label}](panels/{p})" for label,p in zip(("深度","近焦","远焦"),paths))
        md.append(f"- {name} f030。{note} {links}")
        html.append(f"<section><h2>{escape(name)} f030</h2><p>{escape(note)}</p>")
        for p in paths:
            html.append(f"<a href='panels/{escape(p)}'><img src='panels/{escape(p)}' loading='lazy' width='960'></a>")
        html.append("</section>")
    (RUN/"观察清单.md").write_text("\n".join(md)+"\n")
    (RUN/"index.html").write_text("\n".join(html)+"\n")


def markdown(record):
    lines=["# E-L 准备工作","","| 档 | 场景 | 帧 | thin_structure 场景 | α<0.5 冲突 | α≥0.5 冲突 |",
           "|---|---:|---:|---:|---:|---:|"]
    for scope,x in record["labels"]["scopes"].items():
        row=x["conflicts"]
        lines.append(f"| {scope} | {x['scenes']} | {x['frames']} | {x['categories'].get('thin_structure',0)} | "
                     f"{pct(row['alpha_lt_half']['conflict_fraction'])} | {pct(row['alpha_ge_half']['conflict_fraction'])} |")
    lines.extend(["",f"后分量间隔 m = {record['labels']['m']:.6f}。误差单位为 m⁻¹。","",
                  "| 档 | 区域 | 多视角触发 | p50／p95 | 带符号中位数 |","|---|---|---:|---:|---:|"])
    for scope,x in record["anchors"]["stats"].items():
        for region in REGIONS:
            y=x[region]
            z=y["error_inv_m"]
            lines.append(f"| {scope} | {region} | {pct(y['triggered_fraction'])} | {error(z)} | {fmt(z['signed_p50'])} |")
    lines.extend(["","## 逆深度与门控","",
                  "| 划分 | 区域 | OVG ρ₁ p50／p95 | E ρc p50／p95 | OVG ρ₂ p50／p95 | E 后分量 p50／p95 | 前／后饱和 | M 召回 | g p10／p50／p90 | 选后比例 |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for group,views in record["depth"].items():
        for zone,row in views.items():
            gates=row["g_quantiles"]
            gate_col="／".join(fmt(gates[k],3) for k in ("p10","p50","p90")) if gates else "—"
            sat_col="／".join(pct(row.get(k+"_saturated_fraction")) for k in ("front","back"))
            lines.append("| "+" | ".join((group,zone,error(row["OVG_rho1"]),error(row["E_decoded"]),
                error(row["OVG_rho2"]),error(row["E_back"]),sat_col,pct(row.get("M_hit_fraction")),
                gate_col,pct(row.get("selected_back_fraction"))))+" |")
    lines.extend(["","## 原光圈渲染口径","","| 划分 | 焦点 | 档 | 全图 PSNR | 焦点评分区 PSNR | 杆 | 缝 |",
                    "|---|---|---|---:|---:|---:|---:|"])
    for cohort,row in record["optical"]["pooled"].items():
        for focus,methods in row.items():
            for method,values in methods.items():
                vals=[fmt(values[k]["psnr_db"],3) for k in ("full","focus_region","rod","gap")]
                lines.append("| "+" | ".join((cohort,focus,method,*vals))+" |")
    lines.extend(["",record["dev005_f001"]["conclusion"],"",
                  "[图像索引](index.html) · [逐帧观察](观察清单.md) · [训练曲线](overfit-curve.png)",""])
    return "\n".join(lines)


def check_dev005():
    """Isolate the archived empty placement using native arrays on CPU."""
    from route3_v2data_fp16_fit_audit import archived_local_placement

    historical_fit=archived_local_placement()
    name="development-005-35442efa18790d0a"
    labels=FORMAL_DATA/"units/development"/name/"labels"
    rows=[]
    for frame in (1,30,60):
        tag=f"f{frame:03d}"
        with (np.load(labels/f"{tag}-pinhole.npz") as gt,
              np.load(labels/"scene-geometry.npz") as ids,
              np.load(PARENTS/name/f"{tag}-focus1-N1.4-optics.npz") as parents,
              np.load(cache_path("ovg",name,frame)) as geometry):
            raw=gt["depth_axial_m"]
            bg=(raw>0)&gt["valid"]&gt["center_visible"]&~np.isin(gt["object_id"],ids["foreground_object_ids"])
            x,y=np.floor(parents["uv"]).astype(int).T
            inside=(x>=0)&(x<raw.shape[1])&(y>=0)&(y<raw.shape[0])
            known=parents["recoverable"]&inside
            x=x.clip(0,raw.shape[1]-1)
            y=y.clip(0,raw.shape[0]-1)
            row=dict(frame=frame,eligible=int(parents["recoverable"].sum()),inside=int(known.sum()),
                     zero_depth_pixels=int((raw<=0).sum()),casts={})
            for dtype in (np.float16,np.float32):
                with np.errstate(over="ignore",invalid="ignore",divide="ignore"):
                    a,b,_,mask=historical_fit(raw.astype(dtype),1/geometry["rho1"],bg,return_valid=True)
                    inverse=a[y,x]/parents["depth"]+b[y,x]
                finite=np.isfinite(inverse)
                supported=known&mask[y,x]
                row["casts"][np.dtype(dtype).name]=dict(supported=int(supported.sum()),
                    nonfinite=int((known&~finite).sum()),passing=int((supported&finite&(inverse>1e-4)).sum()))
            rows.append(row)
    return dict(evidence="原模块二日志提供资格与放置计数。用其 GT 候选缓存、中心标签和本轮已落盘同一 OVG EMA 深度核对局部仿射，不重跑光学。",
                rows=rows)

def main():
    plan=load(RUN/"inventory.json")
    labels=load(RUN/"labels.json")
    anchor=load(RUN/"ovg-audit.json")
    cache=load(RUN/"cache-report.json")
    tr=load(RUN/"overfit-training.json")
    depth=load(RUN/"depth-diagnostic.json")
    optical=load(RUN/"panels.json")
    cast=check_dev005()
    notes=load(RUN/"visual-review.json")
    count=len(optical["rows"])
    if count!=9 or set(notes)!={r["scene"] for r in optical["rows"]} or len(cache["checks"])!=3:
        raise ValueError("The fixed figures and all three cache equivalence checks require review")
    dismissed=FORMAL_DATA/"units/training/failures.json"
    original=load(dismissed) if dismissed.is_file() else {}
    labels.pop("frames")
    for f in ("metadata","geometry","optical","development"):
        plan.pop(f)
    labels["inventory"]=plan
    labels["accepted_existing_geometry_gaps"]=original
    observed=schedules([("OVG",anchor),("DA2",cache),("E_L",tr),("depth_diagnostic",depth),("optical",optical)])
    cast["conclusion"]="dev005 f001 的 FP16 中心深度含 7,148 个零值，取倒数溢出并污染局部拟合；13,191 个合格候选的拟合逆深度全部非有限，故放置 0。"
    record=dict(
        at=datetime.now().astimezone().isoformat(timespec="seconds"),
        basis=["dc3c736","dc6314c"], independent_test_cases_used=0,stage3_integration=False,
        labels=labels, anchors=anchor, caches=cache, overfit=tr, depth=depth["summaries"],
        optical=dict(panel_color_range=optical["shared_inverse_color_range"], rows=optical["rows"],
                     pooled=render_stats(optical["rows"]),elapsed_seconds=optical["elapsed_seconds"]),
        observed_3b_pace=observed,dev005_f001=cast,observations=notes,
        artifacts={key:str((RUN/f).relative_to(ROOT)) for key,f in
           (("scope","inventory.json"),("cache","cache"),("exact","cache-report.json"),("figures","index.html"),
            ("notes","观察清单.md"),("summary","汇总.md"),("curves","overfit-curve.png"))})
    gallery(optical["rows"],notes)
    plot_curves(tr)
    (RUN/"汇总.md").write_text(markdown(record))
    save_json(RECORD,record)


if __name__=="__main__":
    main()
