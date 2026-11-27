"""Aggregate the E0 v4 optical and holdout outputs and update the experiment record."""
import argparse
from collections import defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import route3_v2data_E0_v4_optics as optics
from route3_v2data_B_batch import sha256
from route3_v2data_E_labels import load, save_json

RUN = optics.RUN
RECORD = optics.RECORD
LEARNED = ("F2", "reference", "Query")
RELEASE = ("development", "legacy")
SUBJECT_ZONES = ("gap_in_subject", "normal_in_subject")


def key(row):
    return row["scene"], row["case"], row["focus_mode"]


def evaluated(path):
    return [r for f in sorted((path/"frames").glob("*/*.json")) for r in load(f)["rows"]]


def archived_curve2():
    old = optics.v3.RUN/"evaluation/E2_2k"
    saved = []
    for file in sorted((old/"frames").glob("development-*/*.json")):
        for focus,rows in load(file)["rows"].items():
            for r in rows:
                if r["focus_index"]!=1 or r["status"]!="evaluated":
                    continue
                s=r["methods"]["E_B"]
                saved.append(dict(cohort="development",scene=r["scene"],frame=r["frame"],case=r["case"],
                                  focus_mode=focus,status="evaluated",focus_index=1,
                                  far=dict(F2=s["far_off"],reference=s["far_witness"])))
    return saved


def distribution(values):
    if not values:
        return dict(median_db=None, positive=0, negative=0, tied=0, n=0)
    return optics.ref.distribution(values)


def score_map(rows, metric):
    return {key(r):r["far"][metric] for r in rows if r["status"]=="evaluated"
            and "far" in r and metric in r["far"] and r["far"][metric]["pixels"]}


def pair(current, baseline):
    intersection = sorted(current.keys() & baseline.keys())
    groups=defaultdict(list)
    for item in intersection:
        if current[item]["pixels"] != baseline[item]["pixels"]:
            raise ValueError(f"Matched methods use different score support at {item}")
        groups[item[0]].append(item)
    scene = {k:optics.base.pooled([current[v] for v in values])["psnr_db"]-
               optics.base.pooled([baseline[v] for v in values])["psnr_db"]
             for k,values in sorted(groups.items())}
    c,b=(optics.base.pooled([score[v] for v in intersection]) for score in (current,baseline))
    return dict(**distribution(list(scene.values())),by_scene_db=scene,used_cases=len(intersection),
                current_pooled=c,baseline_pooled=b,
                pooled_delta_db=c["psnr_db"]-b["psnr_db"] if intersection else None)


def summarize(rows, previous, metrics=LEARNED):
    scored=[r for r in rows if r["status"]=="evaluated"]
    values={metric:score_map(scored,metric) for metric in metrics}
    baseline={metric:score_map(previous,metric) for metric in metrics}
    pooled={metric:optics.base.pooled(list(v.values())) for metric,v in values.items()}
    against_b={metric:pair(values[metric],baseline[metric]) for metric in metrics}
    against_f2={metric:pair(values[metric],values["F2"]) for metric in metrics if metric!="F2"}
    counts=defaultdict(dict)
    for metric,maps in values.items():
        groups=defaultdict(list)
        for k,v in maps.items():
            groups[k[0]].append(v)
        for scene,items in groups.items():
            counts[scene][metric]=optics.base.pooled(items)
    zero={}
    for key0,floor in optics.base.FLOORS.items():
        support=[r[key0] for r in scored if key0 in r and r[key0]["pixels"]]
        if support:
            zero[key0]=dict(floor_db=floor,passed=sum(v["psnr"]>=floor for v in support),cases=len(support),
                            **optics.base.pooled(support))
    return dict(pooled=pooled,vs_B=against_b,vs_same_arm_F2=against_f2,by_scene=dict(counts),
                zero=zero,cases=len(scored),far_cases=sum("far" in r for r in scored),
                query_pixels=sum(r.get("query_pixels",0) for r in scored))


def scope(rows, focus, cohort):
    return [r for r in rows if (r["focus_mode"],r["cohort"])==(focus,cohort)]


def holdout_curve(rows):
    items = []
    for row in rows:
        data=row["holdout"]
        error = data["errors"]
        baseline = error["rod/OVG"]["p50"]
        rod = error["rod/decoded"]["p50"]
        items.append(dict(step=row["step"],coc_px=data["coc_balanced_px"],rod_p50=rod,
                          OVG_rod_p50=baseline,rod_ratio=rod/baseline,
                          eligible=rod<=baseline))
    return items


def read_curve_report(training, b_rows):
    old_holdout = load(optics.v3.RESULT)["holdout"]["v2"]["curve"]
    holds={"v2":holdout_curve(old_holdout), "v4":holdout_curve(training["curve"])}
    outcomes = []
    failures = []
    for version,points in optics.CURVE_STEPS.items():
        schedule={r["step"]:r for r in holds[version]}
        for step in points:
            if (version,step)==("v2",2000):
                rows=archived_curve2()
                source="E0 v3 E2_2k no-tear F2 and S2 archive"
            else:
                where=RUN/f"curve/{version}-{step:06d}"
                if not (where/"finished.json").is_file():
                    raise FileNotFoundError(f"Missing optical curve receipt for {version}@{step}")
                rows=evaluated(where)
                source="No-tear F2 archive from E0 v2 and new S2" if (version,step)==("v2",4000) else "New no-tear F2 and S2"
            failures.extend(dict(version=version,step=step,**r) for r in rows if r["status"]=="failed")
            optics_rows={f:summarize(scope(rows,f,"development"), scope(b_rows,f,"development"),("F2","reference"))
                         for f in optics.FOCI}
            outcomes.append(dict(version=version,step=step,source=source,holdout=schedule[step],optical=optics_rows))
    return holds,outcomes,failures


def success(evaluation):
    scores=evaluation["G"]["development"]["vs_B"]
    checks={}
    for metric in ("F2","reference"):
        pair0=scores[metric]
        tests=dict(paired_median_nonnegative=pair0["median_db"] is not None and pair0["median_db"]>=0,
                   pooled_not_below_B=pair0["pooled_delta_db"] is not None and pair0["pooled_delta_db"]>=0)
        checks[metric] = dict(**tests, passed=all(tests.values()),paired_median_db=pair0["median_db"],
                              pooled_delta_db=pair0["pooled_delta_db"],matched_cases=pair0["used_cases"])
    return dict(passed=all(t["passed"] for t in checks.values()),checks=checks,
                protocol="G focus; paired median and pooled PSNR delta >= 0 dB for both F2 and S2 against B.")


def chart(rows, decision):
    fig,axes=plt.subplots(2,2,figsize=(10,7.2),layout="constrained")
    variants={"v2":"#427398","v4":"#c75231"}
    lines=("--","-")
    for version,color in variants.items():
        points=[p for p in rows if p["version"]==version]
        xs=[p["step"]/1000 for p in points]
        ax,value=axes[0,0],"coc_px"
        ax.plot(xs,[p["holdout"][value] for p in points],"o-",label=version,color=color)
        axes[0,1].plot(xs,[p["holdout"]["rod_ratio"] for p in points],"o-",label=version,color=color)
        for focus,ax in zip(("G","P"),axes[1],strict=True):
            for metric,style in zip(("F2","reference"),lines,strict=True):
                name="S2@0.90" if metric=="reference" else metric
                ys=[p["optical"][focus]["pooled"][metric]["psnr_db"] for p in points]
                ax.plot(xs,ys,linestyle=style,marker="o",color=color,label=f"{version} {name}")
    axes[0,0].set_title("Holdout: balanced decoded blur error (lower is better)")
    axes[0,0].set_ylabel("Mean of four zones (px)")
    axes[0,1].set_title("Holdout rod p50 / OVG p50 (lower is better)")
    axes[0,1].axhline(1,color="#464a4f",linestyle=":",linewidth=1,label="Selection gate")
    axes[0,1].set_ylabel("Rod error ratio")
    for focus,ax in zip(("G","P"),axes[1],strict=True):
        ax.set_title(f"Development far-region PSNR · {focus} · no tear")
        ax.set_ylabel("PSNR (dB)")
    for ax in axes.ravel():
        ax.set_xlabel("EMA checkpoint (k steps)")
        ax.set_xticks([2,4,6,8])
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.suptitle(f"E0 checkpoints: depth and optics  ·  selected v4 step {decision:,}  ·  optics for reporting",
                 fontsize=12)
    path=RUN/"figures/step-depth-far-optics.png"
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=160)
    plt.close(fig)
    return str(path.relative_to(RUN))


def fmt(v,n=3):
    return "—" if v is None else f"{v:.{n}f}"


def paired(row):
    return f"{fmt(row['median_db'])}（{row['positive']}/{row['negative']}）"


def col(values):
    return " | ".join(str(v) for v in values)


def draw_report(result, previous):
    sections=["# E0 第四版","", "## 怎么读","",
        "P 由各臂按同一主体对焦，G 使用同一 GT 焦距。远焦可恢复区以相同像素掩码评分。"
        "合并 PSNR 的单位为 dB，越高越好。配对列先按场景合并该场景的有效帧，再报告差值中位数及其正／负场景数。"
        "开发集参考列是 S2@0.90，原 3 场景是 S0 同帧剥离。B 为 v1@2k 加撕裂的归档基线。"
        "解码弥散圈误差的单位为 px，越低越好。杆倍数是该点的杆 p50 除以 OVG 杆 p50。"
        "步数曲线各点使用无撕裂渲染，光学曲线用于报告。第四版正式臂的步骤由深度规则选出，完整评价使用重训的撕裂头。","",
        "## 步数–深度–光学曲线","",f"[曲线图]({result['curve_figure']})","",
        "| 版本 | 步数 | 留出解码弥散圈误差 | 杆 p50 倍数 | 焦距 | F2 | S2 | F2 对 B | S2 对 B | 数据来源 |",
        "|---|---:|---:|---:|---|---:|---:|---|---|---|"]
    for row in result["optical_curve"]:
        for f,b in row["optical"].items():
            h=row["holdout"]
            sections.append("| "+col([row["version"],row["step"],fmt(h["coc_px"]),fmt(h["rod_ratio"]),f,
                *(fmt(b["pooled"][metric]["psnr_db"],2) for metric in LEARNED[:2]),
                *(paired(b["vs_B"][metric]) for metric in LEARNED[:2]),row["source"]])+" |")
    sections.extend(["","## 正式选点与光学比较","",
        f"第四版选中 **{result['selected_step']:,} 步 EMA**。"
        f"杆门槛{'通过' if result['matched_rod_gate'] else '无通过检查点，按原规则取杆误差最低点'}。"
        f"撕裂头选中 {result['tear']['selected_step']:,} 步。"
        f"G 远焦成功标准 **{'达到' if result['success']['passed'] else '未达到'}**。",""])
    for cohort in RELEASE:
        sections.extend([f"### {'开发集' if cohort=='development' else '原 3 场景'}","",
        "| 焦距 | 臂 | F2 | 参考 | Query | 参考对本臂 F2 | Query 对本臂 F2 | F2／参考／Query 对 B |",
        "|---|---|---:|---:|---:|---|---|---|"])
        for focus in optics.FOCI:
            old=previous[focus][cohort]
            for arm in ("A","B","C","D","v4"):
                row=result["evaluation"][focus][cohort] if arm=="v4" else result["archived_pairings"][focus][cohort][arm]
                vals=row["pooled"] if arm=="v4" else old[arm]["far"]["totals"]
                totals=[fmt(vals[k]["psnr_db"],2) for k in LEARNED]
                pair_f=row["vs_same_arm_F2"] if arm=="v4" else old[arm]["paired_vs_F2"]
                cells=[f"{fmt(pair_f[k].get('median_db',pair_f[k].get('median')))}（{pair_f[k]['positive']}/{pair_f[k]['negative']}）"
                       for k in LEARNED[1:]]
                label="D（诊断）" if arm=="D" else arm
                sections.append("| "+col([focus,label,*totals,*cells,
                                          "／".join(paired(row["vs_B"][k]) for k in LEARNED)])+" |")
        sections.append("")
    sections.extend(["## 杆区与 dev003 主体","",
        "杆前分量误差带符号，单位为 1/m。后分量误差是截断前的 log 逆深度绝对误差。后分量基底在表中按 1/m 显示。","",
        "| 划分 | 场景 | 臂 | 杆前分量 p50 | 杆边缘双层后分量 p50 | 杆内部单层后分量 p50 |",
        "|---|---|---|---:|---:|---:|"])
    diagnostic=result["diagnostics"]
    for cohort,groups in diagnostic["rods_by_scene"].items():
        for scene,table in groups.items():
            label="dev"+scene.split("-")[1] if cohort=="development" else "-".join(scene.split("-")[:2])
            for arm in ("C","D","v4"):
                vals=table[arm]
                front=vals["rod_front_signed_rho"]["signed_p50"]
                edge=vals["rod_edge_dual_rear"]["abs_p50"]
                interior=vals["rod_interior_single_rear"]["abs_p50"]
                sections.append("| "+col([cohort,label,arm,fmt(front),fmt(edge),fmt(interior)])+" |")
    sections.extend(["","| 帧 | 臂 | 区域 | 像素 | 选后比例 | 前／后／GT p50 | 基底来源 ρ₂／DS／三角化 | β p50 | 1.5 tanh 残差 | 实际残差 | 饱和率 | g | a |",
                     "|---|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|"])
    for frame,groups in diagnostic["dev003_subject"].items():
        for arm in ("C","D","v4"):
            for zone in SUBJECT_ZONES:
                v=groups[arm][zone]
                sections.append("| "+col([frame,arm,"缝隙" if zone==SUBJECT_ZONES[0] else "普通",v["pixels"],
                      fmt(v["rear_selected_fraction"]), "／".join(fmt(v[k]) for k in ("front","rear","gt")),
                      "／".join(str(n) for n in v["beta_source"].values()),fmt(v["beta"]),
                      *(fmt(v[k]) for k in ("residual","applied_residual","residual_saturated_fraction","g","a"))])+" |")
    sections.extend(["","## 图像",""])
    for figure in result["panels"]["figures"]:
        label=figure["scene"]
        sections.append(f"- [{label} f{figure['frame']:03d}]({figure['path']})")
    sections.extend(["","## 结论",""])
    for metric,values in result["success"]["checks"].items():
        label="S2" if metric=="reference" else metric
        sections.append(f"- G 的 {label} 对 B 逐场景中位为 {fmt(values['paired_median_db'])} dB，合并差为 {fmt(values['pooled_delta_db'])} dB。")
    sections.append(f"- 光学跳过条目 {len(result['failures'])}，图像跳过条目 {len(result['panels']['failed'])}，测试使用数 0。")
    for failure in result["failures"]:
        sections.append(f"- {failure['scene']} {failure['case']} {failure['focus_mode']} · {failure['error']}")
    for p in result["panels"]["failed"]:
        sections.append(f"- {p['scene']} f{p['frame']:03d} · {p['reason']}")
    (RUN/"汇总.md").write_text("\n".join(sections)+"\n")


def collect():
    tag = RUN/optics.v3.MODE
    training = load(tag/"training.json")
    selected = training["selected"]["step"]
    model_path=tag/f"step-{selected:06d}.pt"
    tear_path=tag/f"tear-{selected:06d}"
    tear=load(tear_path/"training.json")
    release=load(optics.EARLIER)
    old=release["results"]
    previous=load(optics.PREVIOUS/"all-case-rows.json")
    b=previous["B"]
    holdout,curves,failures=read_curve_report(training,b)
    if not (RUN/"full/finished.json").is_file():
        raise FileNotFoundError("Full B-development and legacy evaluation has not finished")
    full=evaluated(RUN/"full")
    failures.extend(r for r in full if r["status"]=="failed")
    evaluation,comparators={},{}
    for f in optics.FOCI:
        evaluation[f],comparators[f]={},{}
        for cohort in RELEASE:
            baseline=scope(b,f,cohort)
            evaluation[f][cohort]=summarize(scope(full,f,cohort),baseline)
            comparators[f][cohort]={a:dict(vs_B=summarize(scope(previous[a],f,cohort),baseline)["vs_B"])
                                    for a in "ABCD"}
    result=dict(status="completed",recorded_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        selected_step=selected, checkpoint_sha256=sha256(model_path),matched_rod_gate=training["rod_baseline_eligible"],
        holdout_curves=holdout,optical_curve=curves,curve_figure=chart(curves,selected),
        training_resources={k:training[k] for k in ("steps","step_seconds","wall_seconds","peak_reserved_gib","gpu_wait_seconds")},
        tear=dict(selected_step=tear["selected"]["step"],weights_sha256=sha256(tear_path/"selected.pt"),
                  **{k:tear[k] for k in ("config","wall_seconds","peak_reserved_gib")}),
        evaluation=evaluation,archived_pairings=comparators,archived_source="3e135e5",
        success=success(evaluation),failures=failures,test_cases_used=0,
        panels=load(RUN/"diagnostics/panels.json"),diagnostics=load(RUN/"diagnostics/depth.json"))
    save_json(RUN/"result.json",result)
    draw_report(result,old)
    return result


def register_results():
    result=load(RUN/"result.json")
    if not result:
        raise FileNotFoundError("Run collect before updating the repository record")
    previous=load(RECORD)
    previous["status"]="completed"
    previous["result"]=result
    save_json(RECORD,previous)
    progress=RECORD.parents[1]/"当前实验进度与方法论.md"
    anchor="**隐藏层见证颜色纯度核查登记（2026-11-26）**"
    name="**E0 第四版结果**"
    note=(f"{name}。选中 {result['selected_step']:,} 步 EMA，撕裂头选中 {result['tear']['selected_step']:,} 步。"
          f"开发集 G 远焦相对 B 的 F2 与 S2 双门槛{'达到' if result['success']['passed'] else '未达到'}。"
          f"光学跳过 {len(result['failures'])} 条，图像跳过 {len(result['panels']['failed'])} 条。"
          "曲线与正式臂分别记录无撕裂和配撕裂表现，架构状态保留待定供复核。"
          f" [机器记录](records/{RECORD.name})")
    content=progress.read_text()
    if name not in content:
        if content.count(anchor)!=1:
            raise ValueError("Central progress anchor changed")
        progress.write_text(content.replace(anchor,note+"\n\n"+anchor,1))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("collect","register"))
    args=parser.parse_args()
    {"collect":collect,"register":register_results}[args.stage]()
