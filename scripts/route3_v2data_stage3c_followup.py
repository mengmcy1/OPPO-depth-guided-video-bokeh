"""Continue the server run through both 3c milestones and publish the decision."""
import argparse
from pathlib import Path
import os
import subprocess
import sys
import time

import psutil

from route3_v2data_B_batch import ROOT,sha256,write_json
from route3_v2data_B_plan import load
from route3_v2data_stage2_train import now
from route3_v2data_stage3a_followup import AUTHOR,PROGRESS,RECORDS,git,synchronise
from route3_v2data_stage3b import path_for,read_checkpoint
from route3_v2data_stage3b_eval import REGIONS
from route3_v2data_stage3b_followup import fmt,pair_text
from route3_v2data_stage3c import OUT,SCHEDULE
from route3_v2data_stage3c_eval import adoption,report_path

E0=ROOT.parent/"route3-e0-20261126/outputs/route3-v2data-E0-2026-11-26-01"


def state(run,phase,**extra):
    write_json(run/"dispatch.json",dict(phase=phase,at=now(),pid=os.getpid(),**extra))


def e0_status():
    d=load(E0/"dispatch.json")
    return dict(at=now(),dispatch=d,progress={name:load(E0/name/"progress.json")
                for name in ("E-L","E-H1","E-H2")})


def e0_overlap(first,last):
    rates={}
    for mode,a in first["progress"].items():
        b=last["progress"].get(mode)
        if a and b and b["step"]>a["step"]:
            n=b["step"]-a["step"]
            rates[mode]=dict(steps=n,wall_seconds_per_step=(b["wall_seconds"]-a["wall_seconds"])/n,
                             GPU_wait_seconds=b["gpu_wait_seconds"]-a["gpu_wait_seconds"])
    return dict(before=first,after=last,interval=rates)


def process_status(pid):
    try:
        return psutil.Process(pid).status() not in (psutil.STATUS_ZOMBIE,psutil.STATUS_DEAD)
    except psutil.NoSuchProcess:
        return False


def command(run,tag,script,*args):
    path=run/f"{tag}.log"
    for attempt in range(2):
        state(run,tag,log=str(path),attempt=attempt)
        with path.open("a") as stream:
            p=subprocess.run([sys.executable,"-u",str(ROOT/"scripts"/script),*map(str,args)],
                             cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        if p.returncode==0:
            return
        problem=path.read_text()
        if attempt or not any(k in problem for k in ("CUDA out of memory","OutOfMemoryError")):
            raise RuntimeError(f"Stage 3c {tag} halted; see {path}")
        time.sleep(12)


def evaluate_pair(run,step):
    values={}
    reviewed=load(report_path(run,step,"ema").parent.parent/"visual-review.json")
    for key in ("ema","model"):
        path=report_path(run,step,key)
        if not path.is_file():
            command(run,f"eval-{step:06d}-{key}","route3_v2data_stage3c_eval.py",
                    "--run",run,"--step",step,"--weights",key)
        value=load(path)
        if (value["step"],value["weights"])!=(step,key) or value["test_cases_used"]!=0:
            raise ValueError(f"Invalid Stage 3c evaluation receipt {path}")
        for panel in value["panels"]:
            if panel["case"] in reviewed:
                panel["observation"]+=" "+reviewed[panel["case"]]
        values[key]=value
    return values


def markdown(step,versions,decision):
    names=dict(development="新开发集",legacy="原 3 场景",thin_structure="细结构 dev003/004/005",dev008="dev008")
    lines=[f"# 阶段 3c · {step:,} 步","", "配对列依次为合并差、中位数、标准差、正向场景数。P2、f/1.4、K64。",""]
    for kind,report in versions.items():
        for scope,row in report["summary"].items():
            lines.extend([f"## {kind.upper()} · {names[scope]}","",
                "| 区域 | Query 开 / F2 dB | Query−F2 | 增量相对起点 | Query 开相对起点 | GT−F2 | Query / GT 中位 |",
                "|---|---:|---:|---:|---:|---:|---:|"])
            for region,v in row["metrics"].items():
                effect=v["query_minus_f2"]
                oracle=v.get("gt_minus_f2")
                lines.append(f"| {region} | {fmt(effect['on_db'])} / {fmt(effect['off_db'])} | {pair_text(effect)} | "
                    f"{pair_text(v['query_vs_start'])} | {pair_text(v['Query_on_vs_start'])} | "
                    f"{pair_text(oracle) if oracle else '—'} | {fmt(v.get('middle_ratio'))} |")
            lines.extend(["",f"远焦隐藏有效光线 {row['far_hidden_fraction']:.2%}。"
                f"逆深度漂移 p50 / p95 为 {row['inverse_depth_drift']['p50']:.5f} / {row['inverse_depth_drift']['p95']:.5f} m⁻¹。",""])
            lines.extend(f"- {key} {fmt(v['psnr_db'])} dB，{v['passed']}/{v['scored']} 过线。"
                         for key,v in row["zero"].items())
            if "geometry_limits" in row:
                lines.extend(["","| 几何护栏 | 起点 | 当前 | 状态 |","|---|---:|---:|---|"])
                lines.extend(f"| {k} | {fmt(v['baseline'],5)} | {fmt(v['value'],5)} | {'通过' if v['passed'] else '超限'} |"
                             for k,v in row["geometry_limits"].items())
                lines.append(f"`GT_occlusion_crossing` 带符号中位 {row['geometry']['GT_occlusion_crossing_signed_median']:.5f} m⁻¹。")
            if "foreground_coc_error_px" in row:
                v=row["foreground_coc_error_px"]
                lines.append(f"GT 细结构前景 CoC 误差 p50 / p95 为 {fmt(v['p50'])} / {fmt(v['p95'])} px。")
            lines.append("")
    lines.extend(["## 外部基线","",
        "| 方法 | 新开发近焦 | 新开发远焦 | 原 3 场近焦 | 原 3 场远焦 |",
        "|---|---:|---:|---:|---:|"])
    reference=versions["ema"]["external_comparison"]
    for name,new in reference["external_baselines"].items():
        old=reference["legacy_baselines"].get(name,{})
        def psnr(values,k):
            block=values.get(k,{})
            val=block.get("combined",block.get("pooled",block))
            return fmt(val.get("psnr",val.get("psnr_db"))) if isinstance(val,dict) else "—"
        lines.append(f"| {name} | {psnr(new,REGIONS[0])} | {psnr(new,REGIONS[1])} | "
                     f"{psnr(old,REGIONS[0])} | {psnr(old,REGIONS[1])} |")
    for kind,report in versions.items():
        lines.extend(["",f"## {kind.upper()} 固定图像面板",""])
        lines.extend(f"- {p['case']}。{p['observation']} [面板]({kind}/{p['path']})" for p in report["panels"])
    if decision:
        lines.extend(["",f"采用 {decision['selected_step']:,} 步。",""])
    return "\n".join(lines)+"\n"


def record(run,step,versions,decision,first_e0):
    summary_path=report_path(run,step,"ema").parent.parent/"汇总.md"
    summary_path.write_text(markdown(step,versions,decision))
    entry=dict(step=step,decision=decision,at=now(),summary_file=str(summary_path.relative_to(run)),
               test_cases_used=0,E0=e0_overlap(first_e0,e0_status()),versions={})
    for kind,v in versions.items():
        path=report_path(run,step,kind)
        entry["versions"][kind]={k:v[k] for k in ("summary","external_comparison","panels","seconds")}
        entry["versions"][kind]["source_sha256"]=sha256(path)
        entry["versions"][kind]["GT_reference_mode"]=(
            "Recomputed with current geometry and corrected native depth fitting" if step
            else "Exact replay of FP16-corrected initial reference")
        entry["versions"][kind]["focus_by_case"]=[{k:r.get(k) for k in
            ("cohort","scene","case","frame","focus_index","status","focus_m","GT_focus_m")} for r in v["rows"]]
    if step:
        p=path_for(run,step)
        raw=read_checkpoint(p)
        entry["checkpoint"]=dict(relative_path=str(p.relative_to(run)),sha256=sha256(p),seconds=raw["seconds"],
                                 counts={k:raw["counts"][k] for k in ("B","C","thin_crops","thin_pixels")})
    name=f"route3-v2data-stage3c-2026-11-26-step-{step:06d}.json"
    destination=RECORDS/name
    if not synchronise():
        raise RuntimeError("Could not sync Stage 3c records before publication")
    git("status","--short")
    if not destination.is_file():
        write_json(destination,entry)
        notes=PROGRESS.read_text().splitlines(keepends=True)
        before=max(i for i,line in enumerate(notes) if line.startswith("**阶段 3c "))
        m=versions["ema"]["summary"]["development"]
        near,far=(m["metrics"][key]["query_minus_f2"]["pooled_db"] for key in REGIONS[:2])
        conclusion=(f" 阶段 3 采用 3c {decision['selected_step']:,} 步。" if decision else "")
        text=(f"**阶段 3c {step:,} 步**。新开发集 EMA 的 Query 相对 F2 近／远焦为 "
              f"{near:+.3f}／{far:+.3f} dB，远焦有效隐藏光线 {m['far_hidden_fraction']:.2%}。{conclusion}")
        notes[before+1:before+1]=["\n",text+f" [记录](records/{name})\n"]
        PROGRESS.write_text("".join(notes))
        git("diff","--check")
        git("add","--",str(destination.relative_to(ROOT)),str(PROGRESS.relative_to(ROOT)))
        git("diff","--cached","--check")
        git("commit","-m",f"记录阶段 3c {step:,} 步评价","-m",AUTHOR)
    if not synchronise():
        raise RuntimeError("Could not sync the Stage 3c milestone")
    git("push","origin","HEAD:main")


def follow(run):
    first=run/"E0-start.json"
    if not first.is_file():
        write_json(first,e0_status())
    initial=load(first)
    starting=evaluate_pair(run,0)
    record(run,0,starting,None,initial)
    first_boundary=SCHEDULE[1]
    pid=int((run/"train.pid").read_text())
    state(run,"training-005000",trainer_pid=pid)
    while not path_for(run,first_boundary).is_file() and process_status(pid):
        time.sleep(20)
    if not path_for(run,first_boundary).is_file():
        previous=(run/"train-005000.log").read_text()
        if "CUDA out of memory" not in previous and "OutOfMemoryError" not in previous:
            raise RuntimeError("Initial Stage 3c trainer stopped before 5k")
        command(run,"train-005000-retry","route3_v2data_stage3c.py","train","--run",run,"--through",5000)
    middle=evaluate_pair(run,5000)
    record(run,5000,middle,None,initial)
    if not path_for(run,10000).is_file():
        command(run,"train-010000","route3_v2data_stage3c.py","train","--run",run,"--through",10000)
    final=evaluate_pair(run,10000)
    decision=adoption(starting["ema"],[middle["ema"],final["ema"]])
    write_json(run/"adoption.json",decision)
    record(run,10000,final,decision,initial)
    state(run,"complete",selected_step=decision["selected_step"])


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=Path,default=OUT)
    args=p.parse_args()
    try:
        follow(args.run)
    except Exception as exc:
        state(args.run,"halted",reason=str(exc))
        raise
