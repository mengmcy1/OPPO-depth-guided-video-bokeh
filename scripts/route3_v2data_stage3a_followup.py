"""Evaluate and publish the pre-registered Stage 3a milestones."""
import argparse
from collections import defaultdict
import fcntl
from pathlib import Path
import subprocess
import sys
import time

import psutil

from route3_v2data_B_batch import ROOT, sha256, write_json
from route3_v2data_B_plan import load
from route3_v2data_stage2_train import now
from route3_v2data_stage3a import CHECKPOINTS, INITIAL, OUT, checkpoint, configuration_for, crop_statistics, read_checkpoint
from route3_v2data_stage3a_prior_compare import comparison_markdown
from route3_v2data_stage3a_prior_diagnostic import diagnostic_markdown

RECORDS = ROOT/"03_实验记录/records"
PROGRESS = RECORDS.parent/"当前实验进度与方法论.md"
METRICS = ("full_image","near_in_focus","far_recoverable","boundary_2px","zero_visible","zero_combined")
AUTHOR = "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"


def git(*args, check=True):
    return subprocess.run(["git",*args],cwd=ROOT,check=check)


def synchronise():
    if git("pull","--rebase","origin","main",check=False).returncode:
        marker = subprocess.check_output(["git","rev-parse","--absolute-git-dir"],cwd=ROOT,text=True).strip()
        if any((Path(marker)/name).exists() for name in ("rebase-apply","rebase-merge")):
            git("rebase","--abort")
        return False
    return True


def publish(heading, name, payload, note):
    if not synchronise():
        print("STAGE3A_PUBLICATION_PENDING",name,flush=True)
        return None
    path = RECORDS/name
    if not path.is_file():
        write_json(path,payload)
        contents = PROGRESS.read_text().splitlines(keepends=True)
        target = max(i for i,line in enumerate(contents)
                     if line.startswith(("**阶段 3 起点外部基线**","**阶段 3a ")))
        contents[target+1:target+1] = [
            "\n",f"{heading}。{note}[阶段 3a 记录](records/{name})\n"]
        PROGRESS.write_text("".join(contents))
        git("diff","--check")
        git("add","--",str(path.relative_to(ROOT)),str(PROGRESS.relative_to(ROOT)))
        git("diff","--cached","--check")
        git("commit","-m",f"记录阶段 3a {payload['step']} 步评价","-m",AUTHOR)
    if not synchronise() or git("push","origin","HEAD:main",check=False).returncode:
        print("STAGE3A_PUBLICATION_PENDING",name,flush=True)
        return None
    return subprocess.check_output(["git","log","-1","--format=%h","--",str(path.relative_to(ROOT))],
                                   cwd=ROOT,text=True).strip()


def artifact(run,step,weights):
    return run/"evaluation"/f"step-{step:06d}"/weights/"report.json"


def evaluate_step(run,step,weights):
    path = artifact(run,step,weights)
    if not path.is_file():
        log = run/f"eval-{step:06d}-{weights}.log"
        script = ROOT/"scripts/route3_v2data_stage3a_eval.py"
        with log.open("ab") as stream:
            subprocess.run([sys.executable,"-u",str(script),"--run",str(run),"--step",str(step),
                            "--weights",weights],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
    row = load(path)
    if row.get("step")!=step or row.get("weights")!=weights:
        raise ValueError(f"Invalid stage 3a node report {path}")
    return row


def qualifies(row):
    d = row["report"]["development"]
    geometry=all(v["passed"] for v in d["geometry_limits"].values())
    zero = {k:v["scored"]>0 and v["scored"]==v["pass_cases"]
            for k,v in d["metrics"].items() if k.startswith("zero_")}
    near=d["metrics"]["near_in_focus"]["combined"]["psnr"]
    return dict(step=row["step"],geometry=geometry,zero=zero,
                near_pooled_db=near,eligible=geometry and all(zero.values()) and near is not None)


def adoption(rows):
    candidates=[qualifies(row) for row in rows]
    eligible=[c for c in candidates if c["eligible"]]
    final=dict(selected="initial",adopted_checkpoint=str(INITIAL),adopted_sha256=sha256(INITIAL),
               candidates=candidates,ranking="Best pooled near-focus score among admissible EMA checkpoints")
    if not eligible:
        return dict(final,reason="No trained EMA passed all geometry and pinhole requirements")
    best=max(eligible,key=lambda d:(d["near_pooled_db"],-d["step"]))
    row=next(r for r in rows if r["step"]==best["step"])
    paired=row["report"]["development"]["paired_vs_start"]
    near,far=(paired[k]["median_db"] for k in ("near_in_focus","far_recoverable"))
    final.update(best_step=best["step"],near_paired_median_db=near,far_paired_median_db=far)
    if near is not None and far is not None and near>=0 and far>=0:
        final.update(selected="trained",adopted_checkpoint=row["checkpoint"],adopted_sha256=row["sha256"],
                     reason="Both paired medians are nonnegative")
    else:
        final["reason"]="Best eligible near-focus checkpoint misses a nonnegative paired median"
    return final


def compact(value):
    if isinstance(value,dict):
        return {k:compact(v) for k,v in value.items() if k not in ("per_scene","scenes")}
    if isinstance(value,list):
        return [compact(v) for v in value]
    return value


def fmt(value,digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def paired_text(d):
    return f'{fmt(d["median_db"])} / {fmt(d["sd_db"])} / {d["positive"]}/{d["n"]}'


def write_summary(folder,payload):
    labels = dict(full_image="全图",near_in_focus="近焦焦内",far_recoverable="远焦可恢复",
                  boundary_2px="边界 ±2 px",zero_visible="仅可见零光圈",zero_combined="完整零光圈")
    rows = [f'# 阶段 3a · {payload["step"]:,} 步',"",
            "画质列为合并值与逐场景中位数。配对列为中位数、样本标准差与正向场景数。"
            "P2 使用 f/1.4，起点与当前均使用同一 GT 替身和 V0。",""]
    for weight,data in payload["versions"].items():
        for kind,d in data["report"].items():
            rows.extend([f"## {weight.upper()} · {kind}","",
                         "| 指标 | PSNR 合并 / 中位 | patch LPIPS 合并 / 中位 | 对起点配对 |",
                         "|---|---:|---:|---:|"])
            for metric in METRICS:
                v=d["metrics"][metric]
                p=f'{fmt(v["combined"]["psnr"])} / {fmt(v["scene_median"]["psnr"])}'
                if "pass_cases" in v:
                    p+=f'（过线 {v["pass_cases"]}/{v["scored"]}）'
                lp=f'{fmt(v["combined"]["patch_lpips"],4)} / {fmt(v["scene_median"]["patch_lpips"],4)}'
                rows.append(f'| {labels[metric]} | {p} | {lp} | {paired_text(d["paired_vs_start"][metric])} |')
            gt=d["gt_minus_f2_near"]
            rows.extend(["",f'GT 替身减 F2 的近焦增量为 {fmt(gt["pooled_delta_db"])} dB，'
                         f'配对 {paired_text(gt)}。',"",
                         "| 几何统计 | 当前 | 起点 | 允许上升 | 判定 |",
                         "|---|---:|---:|---:|---|"])
            for name,value in d["geometry"].items():
                lim=d.get("geometry_limits",{}).get(name)
                previous=fmt(lim["initial"]) if lim else "—"
                threshold=f'{100*lim["allowed_growth"]:.0f}%' if lim else "—"
                verdict=("通过" if lim["passed"] else "超限") if lim else "—"
                rows.append(f"| {name} | {fmt(value,5)} | {previous} | {threshold} | {verdict} |")
            if "groups" in d:
                rows.extend(["","| 子集 | 近焦合并 / 中位 | 远焦合并 / 中位 | 当前 CoC p50 / p95（px） | 起点 CoC p50 / p95（px） |",
                             "|---|---:|---:|---:|---:|"])
                for name,entry in d["groups"].items():
                    psnr=[f'{fmt(v["combined"]["psnr"])} / {fmt(v["scene_median"]["psnr"])}'
                          for v in entry["regions"].values()]
                    c=entry["foreground_coc_px"]
                    start=entry.get("start_foreground_coc_px") or {}
                    rows.append(f"| {name} | {psnr[0]} | {psnr[1]} | {fmt(c['p50'])} / {fmt(c['p95'])}"
                                f" | {fmt(start.get('p50'))} / {fmt(start.get('p95'))} |")
            if "paired_vs_A" in d:
                rows.append("")
                for metric,p in d["paired_vs_A"].items():
                    rows.append(f'- 相对 A {labels[metric]}配对 {paired_text(p)}，合并差 {fmt(p["pooled_delta_db"])} dB。')
            rows.extend(["",f'[图像面板]({weight}/index.html) · [观察清单]({weight}/观察清单.md)',""])
        deltas=data.get("parameter_changes")
        if deltas:
            rows.extend(["## 相对起点的参数变化","",
                         "| 模块 | L2 变化 | 相对变化 |","|---|---:|---:|"])
            for name,value in deltas["modules"].items():
                rel=value["relative_l2"]
                rows.append(f"| {name} | {value['delta_l2']:.6g} | {fmt(100*rel,5) + '%' if rel is not None else '—'} |")
            for name,value in deltas["geometry_rows"].items():
                rel=value["relative_l2"]
                rows.append(f"| geometry row {name} | {value['delta_l2']:.6g} | {fmt(100*rel,5) + '%' if rel is not None else '—'} |")
            rows.append("")
        diagnostic = data.get("thin_prior_residual")
        if diagnostic:
            rows.extend(diagnostic_markdown(diagnostic))
        comparison = data.get("prior_comparison")
        if comparison:
            rows.extend(comparison_markdown(comparison))
        rows.extend(["## 新开发集外部基线 · f/1.4","",
                     "| 方法 | 近焦合并 / 中位 | 远焦合并 / 中位 |",
                     "|---|---:|---:|"])
        for method,p in data["external_baselines"].items():
            if method in ("stage3","stage3_f2"):
                continue
            a,b=(p[k] for k in ("near_in_focus","far_recoverable"))
            rows.append(f'| {method} | {fmt(a["combined"]["psnr"])} / {fmt(a["scene_median"]["psnr"])}'
                        f' | {fmt(b["combined"]["psnr"])} / {fmt(b["scene_median"]["psnr"])} |')
        rows.extend(["","Dr.Bokeh 为 B2 Oracle 输入，含 GT 深度和覆盖率。原三场景外部基线沿用归档。",""])
    crops=payload["sampling"].get("B_crop_statistics")
    if crops:
        rows.extend(["## 原生 B 光学裁块","",
                     f"实际 {crops['crops_with_thin']}/{crops['crops']} 个裁块覆盖 GT 细结构。"
                     f"所有 B 裁块里的 GT 细结构像素平均占比为 {100*crops['mean_thin_pixel_fraction']:.3f}%。",""])
    if payload.get("adoption"):
        a=payload["adoption"]
        rows.extend(["## 采用判定","",f'采用 {a["selected"]}。{a["reason"]}。',f'检查点 {a["adopted_checkpoint"]}'])
    (folder/"汇总.md").write_text("\n".join(rows)+"\n")


def training_sample_counts(run,step):
    if not step:
        return {}
    ck=read_checkpoint(checkpoint(run,step))
    cases=ck["optical_sampling"]
    scene_counts=defaultdict(int)
    for id,n in cases.items():
        tier,rest=id.split("/",1)
        if tier=="B":
            scene=rest.split("/",1)[0]
            scene_counts[scene]+=n
    return dict(batches=ck["counts"],B_scene_counts=dict(scene_counts),B_case_counts={
        k.removeprefix("B/"):v for k,v in cases.items() if k.startswith("B/")},training_seconds=ck["seconds"],
        B_crop_statistics=crop_statistics(ck["counts"]))


def node(run,step,all_ema):
    version=configuration_for(run).get("version",1)
    ema=evaluate_step(run,step,"ema")
    if step:
        all_ema.append(ema)
    terminal=step==CHECKPOINTS[-1]
    versions={"ema":ema}
    if terminal:
        versions["model"]=evaluate_step(run,step,"model")
    folder=artifact(run,step,"ema").parents[1]
    payload=dict(step=step,run=str(run),version=version,versions=versions,
                 sampling=training_sample_counts(run,step),evaluation_seconds=sum(r["seconds"] for r in versions.values()),
                 sources={k:dict(path=str(artifact(run,step,k)),sha256=sha256(artifact(run,step,k))) for k in versions})
    if terminal:
        payload["adoption"]=adoption(all_ema)
        write_json(run/"adoption.json",payload["adoption"])
    write_summary(folder,payload)
    # Preserve the complete scored case tables locally. The milestone record
    # contains merged metrics, paired statistics and compact scene draw counts.
    for k in versions:
        versions[k]=compact(versions[k])
    payload["summary_path"]=str(folder/"汇总.md")
    d=ema["report"]["development"]
    n,f=(d["metrics"][k]["combined"]["psnr"] for k in ("near_in_focus","far_recoverable"))
    q=qualifies(ema)
    a=d["paired_vs_A"]
    note=(f"EMA 新开发集近焦／远焦 {n:.3f}／{f:.3f} dB，"
          f"相对 A 配对中位 {a['near_in_focus']['median_db']:+.3f}／{a['far_recoverable']['median_db']:+.3f} dB。"
          f"几何护栏{'通过' if q['geometry'] else '超限'}，零光圈{'两项过线' if all(q['zero'].values()) else '有未过线项'}。")
    if ema.get("thin_prior_residual"):
        note+="固定细结构帧的框内背景和前景逆深度、先验及残差分布已写入汇总与观察清单。"
    if ema.get("prior_comparison"):
        note+="细结构场景与 dev008 的 5 种先验对照已写入汇总及观察清单。"
    if terminal:
        note+=f"采用{'训练后几何' if payload['adoption']['selected']=='trained' else '起点几何'}。"
    local=folder/"milestone.json"
    write_json(local,payload)
    prefix="route3-v2data-stage3a-v2" if version==2 else "route3-v2data-stage3a"
    name=f"{prefix}-2026-11-24-step-{step:06d}.json"
    heading=f"**阶段 3a {'第二版 ' if version==2 else ''}{step:,} 步**"
    commit=publish(heading,name,payload,note)
    return dict(step=step,summary=payload["summary_path"],commit=commit,reported_at=now())


def follow(run,training_pid):
    ledger=run/"milestones.json"
    all_ema=[]
    with (run/"followup.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        reports=load(ledger)
        for step in (0,*CHECKPOINTS):
            if str(step) in reports and reports[str(step)].get("commit"):
                if step:
                    all_ema.append(load(artifact(run,step,"ema")))
                continue
            while step and not checkpoint(run,step).is_file():
                proc=psutil.Process(training_pid) if psutil.pid_exists(training_pid) else None
                if proc is None or proc.status()==psutil.STATUS_ZOMBIE:
                    raise RuntimeError(f"The trainer exited before checkpoint {step}")
                time.sleep(20)
            # Step 0 is evaluated by this watcher unless a completed reference
            # was already produced. Don't race a pending initial evaluation.
            started=run/"eval-zero.pid"
            while step==0 and not artifact(run,step,"ema").is_file() and started.is_file():
                pid=int(started.read_text())
                proc=psutil.Process(pid) if psutil.pid_exists(pid) else None
                if proc is None or proc.status()==psutil.STATUS_ZOMBIE:
                    break
                time.sleep(10)
            reports[str(step)]=node(run,step,all_ema)
            write_json(ledger,reports)
            print("STAGE3A_MILESTONE",reports[str(step)],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=Path,default=OUT)
    p.add_argument("--training-pid",type=int,required=True)
    a=p.parse_args()
    follow(a.run.resolve(),a.training_pid)
