"""Evaluate and publish Stage 3b milestones and the fixed terminal rule."""
import argparse
import fcntl
from pathlib import Path
import subprocess
import sys
import time

import psutil

from route3_v2data_B_batch import ROOT, sha256, write_json
from route3_v2data_B_plan import load
from route3_v2data_stage2_train import now
from route3_v2data_stage3a_followup import AUTHOR, PROGRESS, RECORDS, git, synchronise
from route3_v2data_stage3b import OUT, SCHEDULE, path_for
from route3_v2data_stage3b_eval import REGIONS, adoption, difference


def fmt(value,n=3):
    return "—" if value is None else f"{value:.{n}f}"


def pair_text(row):
    return (f'{fmt(row["pooled_db"])} / {fmt(row["median_db"])} / {fmt(row["sd_db"])} / '
            f'{row["positive"]}/{row["n"]}')


def report_path(run,step,kind):
    return run/"evaluation"/f"step-{step:06d}"/kind/"report.json"


def evaluate_version(run,step,kind):
    is_variant = kind.startswith("V")
    path = report_path(run,step,kind)
    if not path.is_file():
        exe = ROOT/"scripts/route3_v2data_stage3b_eval.py"
        args = [sys.executable,"-u",str(exe),"--run",str(run),"--step",str(step)]
        args += ["--variant",kind] if is_variant else ["--weights",kind]
        with (run/f"eval-{step:06d}-{kind}.log").open("a") as f:
            subprocess.run(args,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    return load(path)


def publish(path,payload,paragraph):
    if path.is_file():
        return
    if not synchronise():
        raise RuntimeError("Cannot sync before Stage 3b milestone publication")
    write_json(path,payload)
    lines = PROGRESS.read_text().splitlines(keepends=True)
    before = max(i for i,line in enumerate(lines) if line.startswith(("**阶段 3a ","**阶段 3b ")))
    lines[before+1:before+1] = ["\n",paragraph+f' [记录](records/{path.name})\n']
    PROGRESS.write_text("".join(lines))
    git("diff","--check")
    git("add","--",str(path.relative_to(ROOT)),str(PROGRESS.relative_to(ROOT)))
    git("diff","--cached","--check")
    git("commit","-m",f"记录阶段 3b {payload['step']} 步评价","-m",AUTHOR)
    if not synchronise():
        raise RuntimeError("Cannot sync the recorded Stage 3b milestone")
    git("push","origin","HEAD:main")


def ablation(start,final,decision,run):
    step = decision["selected_step"]
    source = final if step else start
    base = source["summary"]["development"]
    result = {}
    for method in ("V3","V3+o"):
        row = evaluate_version(run,step,method)
        d = row["summary"]["development"]
        delta = {k:difference(base["metrics"][k]["query_minus_f2"],d["metrics"][k]["query_minus_f2"]) for k in REGIONS[:2]}
        result[method] = dict(query_delta_vs_V0=delta,summary=row["summary"],report=str(report_path(run,step,method)))
    m = result["V3+o"]
    near,far = (m["query_delta_vs_V0"][k]["median_db"] for k in REGIONS[:2])
    passed = all(v["all_passed"] for v in m["summary"]["development"]["zero"].values())
    result["3c_renderer"] = "V3+o" if near is not None and far is not None and near>=.1 and far>=-.1 and passed else "V0"
    result["protocol"] = "Prediction-only links at relative inverse-depth distance below 0.03. b is zero everywhere."
    return result


def markdown(folder,data):
    rows = [f'# 阶段 3b · {data["step"]:,} 步',"",
            "每个配对格依次为合并差、中位差、样本标准差、正向场景数。单位 dB。",""]
    for version,report in data["versions"].items():
        for cohort,block in report["summary"].items():
            rows.extend([f"## {version.upper()} · {cohort}","",
                         "| 区域 | Query − F2 | 相对起点 Query | GT − F2 | 收益比中位数及场景数 |",
                         "|---|---:|---:|---:|---:|"])
            for region,v in block["metrics"].items():
                gt = v.get("gt_minus_f2")
                rows.append(f'| {region} | {pair_text(v["query_minus_f2"])} | {pair_text(v["query_vs_start"])} | '
                            f'{pair_text(gt) if gt else "—"} | {fmt(v.get("middle_ratio"))}（{v.get("ratio_count",0)}） |')
            rows.append("")
            for key,v in block["zero"].items():
                rows.append(f'- {key}。{v["passed"]}/{v["scored"]} 过线，合并 {fmt(v["psnr_db"])} dB。')
            p = block["far_hidden_fraction"]
            rows.extend([f'- 远焦可恢复区有效隐藏光线比例。{fmt(100*p if p is not None else None,4)}%'
                         f'（{block["hidden_hits"]}/{block["far_rays"]}）。',""])
        rows.extend([f'[固定面板]({version}/index.html) · [观察清单]({version}/观察清单.md)',""])
    rows.extend(["## 冻结几何下的外部参照","",
                 "| 方法 | 新开发近焦合并 | 新开发远焦合并 |","|---|---:|---:|"])
    for method,block in data["external_baselines"].items():
        a,b = [block[k]["combined"]["psnr"] for k in REGIONS[:2]]
        rows.append(f"| {method} | {fmt(a)} | {fmt(b)} |")
    rows.append("")
    if "adoption" in data:
        choice = data["adoption"]
        rows.extend(["## 终点采用判定","",f"采用 {choice['selected_step']:,} 步。判据 {choice['checks']}。",
                     f"3c 渲染表示为 {data['ablation']['3c_renderer']}。","",
                     "| 变体 | 相对 V0 的近焦 Query 增量变化 | 相对 V0 的远焦 Query 增量变化 | 仅可见 / 完整零光圈 |",
                     "|---|---:|---:|---:|"])
        for kind in ("V3","V3+o"):
            item = data["ablation"][kind]
            a,b = [item["query_delta_vs_V0"][k] for k in REGIONS[:2]]
            zero = item["summary"]["development"]["zero"]
            floors = " / ".join(f'{v["passed"]}/{v["scored"]}' for v in zero.values())
            rows.append(f"| {kind} | {pair_text(a)} | {pair_text(b)} | {floors} |")
    (folder/"汇总.md").write_text("\n".join(rows)+"\n")


def follow(run):
    ledger_path = run/"milestones.json"
    ledger = load(ledger_path) or {}
    with (run/"milestones.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for step in SCHEDULE:
            if str(step) in ledger:
                continue
            while not path_for(run,step).is_file():
                pid = int((run/"pid").read_text())
                if not psutil.pid_exists(pid) or psutil.Process(pid).status()==psutil.STATUS_ZOMBIE:
                    raise RuntimeError(f"Trainer exited before Stage 3b step {step}")
                time.sleep(30)
            zero_pid = run/"eval-zero.pid"
            if step==0 and zero_pid.is_file():
                pid = int(zero_pid.read_text())
                while not report_path(run,0,"ema").is_file() and psutil.pid_exists(pid) and psutil.Process(pid).status()!=psutil.STATUS_ZOMBIE:
                    time.sleep(10)
            ema = evaluate_version(run,step,"ema")
            raw = evaluate_version(run,step,"model") if step else ema
            d = ema["summary"]["development"]
            folder = report_path(run,step,"ema").parents[1]
            record = dict(step=step,at=now(),run=str(run),
                          checkpoint=dict(path=str(path_for(run,step)),sha256=sha256(path_for(run,step))),
                          summary=str(folder/"汇总.md"),
                          external_baselines={k:v for k,v in ema["baseline"].items() if k in ("A","R1+R2","Dr.Bokeh","oracle","Oracle")},
                          versions={n:dict(summary=data["summary"],panels=data["panels"],seconds=data["seconds"])
                                    for n,data in (("ema",ema),("model",raw))})
            if step==0:
                record["versions"].pop("model")
                record["raw_weights_at_start_identical_to_ema"] = True
            note = f"**阶段 3b {step:,} 步**。"
            hidden = 100*d["far_hidden_fraction"]
            near,far = (d["metrics"][k]["query_minus_f2"]["pooled_db"] for k in REGIONS[:2])
            note += f"新开发近／远焦 Query 相对 F2 为 {near:+.3f}／{far:+.3f} dB，远焦有效隐藏光线 {hidden:.3f}%。"
            if step==SCHEDULE[-1]:
                start = load(report_path(run,0,"ema"))
                record["adoption"] = adoption(start,ema)
                write_json(run/"adoption.json",record["adoption"])
                record["ablation"] = ablation(start,ema,record["adoption"],run)
                note += f"采用 {record['adoption']['selected_step']:,} 步，3c 渲染表示为 {record['ablation']['3c_renderer']}。"
            markdown(folder,record)
            path = RECORDS/f"route3-v2data-stage3b-2026-11-25-step-{step:06d}.json"
            publish(path,record,note)
            ledger[str(step)] = dict(path=str(path),time=now(),summary=record["summary"])
            write_json(ledger_path,ledger)
            print("STAGE3B_MILESTONE",step,ledger[str(step)],flush=True)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,default=OUT)
    follow(parser.parse_args().run)
