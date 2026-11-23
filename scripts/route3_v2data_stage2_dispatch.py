"""Continue the four Stage 2 checkpoints and publish their matched evaluations."""
from argparse import ArgumentParser
import fcntl
from pathlib import Path
import subprocess
import sys
import time

from route3_v2data_B_batch import ROOT, write_json
from route3_v2data_B_plan import load
from route3_v2data_stage2_train import RUN, fixed_inputs, now

RECORDS=Path("03_实验记录/records")
PROGRESS=Path("03_实验记录/当前实验进度与方法论.md")
TRAINER=ROOT/"scripts/route3_v2data_stage2_train.py"
EVALUATOR=ROOT/"scripts/route3_v2data_stage2_eval.py"
MILESTONES=(("2a",10000,("ema","model")),("2b",10000,("ema",)),
            ("2b",20000,("ema",)),("2b",30000,("ema","model")))


def execute(script, log, *args):
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open("ab") as stream:
        subprocess.run([sys.executable,"-u",str(script),*map(str,args)],cwd=ROOT,
                       stdout=stream,stderr=subprocess.STDOUT,check=True)


def push_record(heading, filename, payload, text):
    def git(*args):
        subprocess.run(["git",*args],cwd=ROOT,check=True)

    def sync():
        if subprocess.run(["git","pull","--rebase","origin","main"],cwd=ROOT).returncode == 0:
            return True
        git_dir=Path(subprocess.check_output(["git","rev-parse","--absolute-git-dir"],cwd=ROOT,text=True).strip())
        if (git_dir/"rebase-merge").is_dir() or (git_dir/"rebase-apply").is_dir():
            git("rebase","--abort")
        return False

    sync()
    file=RECORDS/filename
    write_json(ROOT/file,payload)
    doc=ROOT/PROGRESS
    contents=doc.read_text().splitlines(keepends=True)
    line=f"{heading}。{text}[阶段 2 记录](records/{filename})\n"
    for i,value in enumerate(contents):
        if value.startswith(heading):
            contents[i]=line
            break
    else:
        preceding=next(i for i,v in enumerate(contents) if v.startswith("**第一遍边界容差复核**"))
        index=max([preceding]+[i for i,v in enumerate(contents) if v.startswith("**新阶段 2 ")])
        contents[index+1:index+1]=["\n",line]
    doc.write_text("".join(contents))
    git("diff","--check")
    git("add","--",str(file),str(PROGRESS))
    git("diff","--cached","--check")
    git("commit","-m","记录扩充数据阶段 2 "+filename.removesuffix(".json").rsplit("-",1)[-1],
        "-m","Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>")
    pushed=sync() and subprocess.run(["git","push","origin","HEAD:main"],cwd=ROOT).returncode==0
    commit=subprocess.check_output(["git","rev-parse","--short","HEAD"],cwd=ROOT,text=True).strip()
    if not pushed:
        print("STAGE2_PUBLICATION_PENDING",commit,flush=True)
    return commit,pushed


def split_conclusion(metrics):
    pairs=metrics["split_pair"]
    oracle=pairs["oracle_minus_none"]["boundary_2px"]["pooled"]["delta_db"]
    actual=pairs["retrieved_minus_none"]["boundary_2px"]["pooled"]["delta_db"]
    actual_vs_oracle=pairs["retrieved_minus_oracle"]["boundary_2px"]["pooled"]["delta_db"]
    if actual is not None and actual>0:
        label="预测拆分有提升"
    elif oracle is not None and oracle>0 and actual_vs_oracle is not None and actual_vs_oracle<0:
        label="B_gt 有提升，换成预测后下降"
    else:
        label="B_gt 也无提升"
    return dict(conclusion=label,B_gt_minus_none_db=oracle,retrieved_minus_none_db=actual,
                retrieved_minus_B_gt_db=actual_vs_oracle,metric="combined-SSE boundary ±2 px PSNR")


def node_summary(report):
    d=report["report"]["development"]
    q=d["query_pair"]
    near=q["near_in_focus"]["pooled"]["delta_db"]
    far=q["far_recoverable"]["pooled"]["delta_db"]
    upper=d["upper_reference"]
    v,c=(d["zero_floors"][k] for k in ("zero_visible","zero_combined"))
    near_g=upper["near_in_focus"]["gt_minus_f2"]["paired"]["median_db"]
    far_g=upper["far_recoverable"]["gt_minus_f2"]["paired"]["median_db"]
    optical=report["optical"]
    def count(r):
        return f"{r['passed_cases']}/{r['scored_cases']}"
    return (f"新开发集 EMA 的第二遍 Query 近／远增量 {near:+.3f}／{far:+.3f} dB，"
            f"零光圈两项 {count(v)}、{count(c)}，GT 隐藏层增量中位数 {near_g:+.3f}／{far_g:+.3f} dB。"
            f"B 光学 {optical['admitted']} 场入列，{optical['f2_complete']} 场 F2 齐备。")


def follow(run):
    run.mkdir(parents=True,exist_ok=True)
    status=run/"dispatch.json"
    with (run/"dispatch.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            receipt=load(run/"preflight.json")
            if receipt.get("inputs")!=fixed_inputs():
                execute(TRAINER,run/"preflight.log","--run",run,"--preflight")
                receipt=load(run/"preflight.json")
            if not receipt.get("passed"):
                raise ValueError("Missing successful step zero report")
            milestones=load(run/"milestones.json")
            if "start" not in milestones:
                milestones["start"]=receipt
                write_json(run/"milestones.json",milestones)
            elapsed_start=time.monotonic()
            for phase,step,weights in MILESTONES:
                key=f"{phase}-{step//1000}k"
                if key in milestones:
                    continue
                where=run/"evaluation"/phase/f"step-{step:06d}"
                cp=run/phase/f"step-{step:06d}.pt"
                if not cp.is_file():
                    write_json(status,dict(status="training",stage=phase,target_step=step,at=now()))
                    execute(TRAINER,run/f"{phase}.log","--run",run,"--phase",phase,"--target",step)
                if phase=="2a":
                    first=run/"evaluation/2a/step-000000/ema/report.json"
                    if not first.is_file():
                        write_json(status,dict(status="evaluating_initial_leakage",stage="2a",step=0,at=now()))
                        execute(EVALUATOR,run/"initial-leakage.log","--run",run,"--phase","2a","--step",0)
                write_json(status,dict(status="evaluating",stage=phase,step=step,at=now()))
                reports={}
                for version in weights:
                    p=where/version/"report.json"
                    if not p.is_file():
                        execute(EVALUATOR,where.parent/f"{key}-{version}.log","--run",run,"--phase",phase,
                                "--step",step,"--weights",version)
                    reports[version]=load(p)
                output=dict(stage=phase,step=step,versions=reports,step_zero=receipt,
                            training_skips=load(run/"training-skips.json"))
                if phase=="2a" or step==30000:
                    start=load(run/"evaluation/2a/step-000000/ema/report.json")
                    output["initial_leakage"]=start["leakage_aggregate"]
                if step==30000:
                    output["split_conclusion"]={k:split_conclusion(v) for k,v in reports["ema"]["report"].items()}
                summary=node_summary(reports["ema"])
                if "split_conclusion" in output:
                    summary+=f"新开发集的三分支边界结论为{output['split_conclusion']['development']['conclusion']}。"
                filename=f"route3-v2data-stage2-2026-11-23-{key}.json"
                commit,pushed=push_record(f"**新阶段 2 {key}**",filename,output,summary)
                milestones[key]=dict(commit=commit,record=filename,pushed=pushed,evaluation_wall_seconds=sum(
                    r["evaluation_seconds"] for r in reports.values()),completed_at=now(),
                    optical=reports["ema"]["optical"])
                write_json(run/"milestones.json",milestones)
            write_json(status,dict(status="complete",completed_at=now(),seconds=time.monotonic()-elapsed_start))
        except Exception as error:
            write_json(status,dict(status="halted",reason=str(error),at=now()))
            raise


if __name__=="__main__":
    p=ArgumentParser(description=__doc__)
    p.add_argument("--run",type=Path,default=RUN)
    a=p.parse_args()
    follow(a.run.resolve())
