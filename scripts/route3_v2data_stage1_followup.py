"""Publish the paired Stage 1 milestones and continue into 1b when its data are ready."""
import argparse
from datetime import datetime,timedelta,timezone
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from route3_v2data_B_batch import FORMAL_DATA,sha256,write_json
from route3_v2data_B_plan import load
from route3_v2data_stage1 import BRANCHES,OUTPUT,RECORD,ROOT,TOTAL,TrainingData,checkpoint_path,complete_optical
from route3_v2data_stage1_dispatch import Runner
from route3_v2data_stage1_evaluate import QUERY_PAIR_PROTOCOL,adoption,legacy_baseline

EVALUATOR=ROOT/"scripts/route3_v2data_stage1_evaluate.py"
DRYRUN=FORMAL_DATA/"p2-review/formal-dryrun-b/optical.json"
PROGRESS=Path("03_实验记录/当前实验进度与方法论.md")
EARLY_EMA_THRESHOLD_SECONDS=30*60


def publish(checkout,values,text):
    path=checkout/RECORD.relative_to(ROOT)
    subprocess.run(["git","pull","--rebase","origin","main"],cwd=checkout,check=True)
    record=load(path)
    record.update(values)
    write_json(path,record)
    document=checkout/PROGRESS
    lines=document.read_text().splitlines(keepends=True)
    prefix="**新阶段 1 配对训练**"
    note=f"{prefix}。{text}[阶段 1 记录](records/{RECORD.name})\n"
    index=next((i for i,line in enumerate(lines) if line.startswith(prefix)),None)
    if index is None:
        index=next(i for i,line in enumerate(lines) if line.startswith("**新数据训练入口与空跑验收**"))
        lines.insert(index+1,"\n"+note)
    else:
        lines[index]=note
    document.write_text("".join(lines))
    subprocess.run(["git","diff","--check"],cwd=checkout,check=True)
    subprocess.run(["git","add","--",str(path.relative_to(checkout)),str(PROGRESS)],cwd=checkout,check=True)
    subprocess.run(["git","diff","--cached","--check"],cwd=checkout,check=True)
    if subprocess.run(["git","diff","--cached","--quiet"],cwd=checkout).returncode==1:
        subprocess.run(["git","commit","-m","记录新阶段 1 配对训练里程碑",
                        "-m","Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"],cwd=checkout,check=True)
        subprocess.run(["git","pull","--rebase","origin","main"],cwd=checkout,check=True)
        subprocess.run(["git","push","origin","HEAD:main"],cwd=checkout,check=True)


def register_milestone(run,checkout,values,text):
    record=run/"milestones.json"
    latest=load(record)
    latest.update(values)
    write_json(record,latest)
    for attempt in range(2):
        try:
            publish(checkout,latest,text)
            break
        except subprocess.CalledProcessError as error:
            print("stage1 milestone publication attempt",attempt+1,error,flush=True)
            if attempt==0:
                time.sleep(4)


def optical_counts():
    state=load(FORMAL_DATA/"manifest.json")
    return {key:state["counts"][key] for key in ("train_geometry","train_optical","development","test")}


def wait_for_1a(run):
    while True:
        status=load(run/"dispatch.json")
        checkpoints={b:checkpoint_path(run/"1a"/b,TOTAL) for b in BRANCHES}
        if all(path.is_file() for path in checkpoints.values()) and status.get("1a_checkpoints"):
            return checkpoints
        if status.get("status")=="halted":
            raise RuntimeError(status["reason"])
        time.sleep(25)


def wait_for_1b(run,runner,production_record):
    while True:
        counts=optical_counts()
        state=load(DRYRUN)
        if counts["train_optical"]>=30:
            names=complete_optical(TrainingData())
            if len(names)>=30 and state.get("passed"):
                return dict(optical_complete=counts["train_optical"],F2_complete=len(names),
                            optical_dryrun=dict(path=str(DRYRUN),sha256=sha256(DRYRUN),passed=True))
        if state and not state.get("passed"):
            raise RuntimeError(f"B formal optical dryrun failed: {state.get('failures')}")
        dep=load(production_record).get("v2",{})
        if dep.get("production_halt"):
            raise RuntimeError(f"B production stopped: {dep['production_halt']}")
        runner.status(status="waiting_for_30_optical_and_F2",optical=counts)
        time.sleep(45)


def milestone_command(run, dest, phase, branch, path, weight):
    return ["taskset","-c","23,24,25,26",sys.executable,"-u",str(EVALUATOR),
            "--phase",phase,"--branch",branch,"--weight",str(weight if branch=="same-surface" else 0.),
            "--checkpoint",str(path),"--output",str(dest),"--f2-root",str(run/"development-F2")]


def run_evaluation(args,log_path):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES="0",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",
             PYTHONPATH=f"{ROOT}/scripts:{ROOT}")
    for _ in range(2):
        with log_path.open("ab") as log:
            reply=subprocess.run(args,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        if reply.returncode==0:
            return
    raise RuntimeError(f"Stage 1 evaluation failed twice: {log_path}")


def evaluate_milestone(run, phase, step, checkpoints, weight, *, ema_only=False):
    started=time.monotonic()
    root=run/"evaluation"/phase/f"step-{step:06d}"
    root.mkdir(parents=True,exist_ok=True)
    timing_path=root/"timing.json"
    previous_timing=load(timing_path)
    versions=("ema",) if ema_only else ("ema","model")
    prepare=milestone_command(run,root,phase,"original",checkpoints["original"],weight)
    run_evaluation([*prepare,"--prepare-f2-only"],root/"development-F2.log")
    preparation_seconds=time.monotonic()-started
    result={}
    branch_seconds={}
    for branch,path in checkpoints.items():
        dest=root/branch
        dest.mkdir(parents=True,exist_ok=True)
        if any(load(dest/f"{key}.json").get("query_pair_protocol")!=QUERY_PAIR_PROTOCOL for key in versions):
            scoring_started=time.monotonic()
            args=milestone_command(run,dest,phase,branch,path,weight)
            run_evaluation([*args,"--ema-only"] if ema_only else args,dest/"evaluation.log")
            branch_seconds[branch]=time.monotonic()-scoring_started
        result[branch]={key:load(dest/f"{key}.json") for key in versions}
    elapsed=time.monotonic()-started
    timing=previous_timing if not branch_seconds and previous_timing else dict(
        wall_seconds=elapsed,development_f2_preparation_seconds=preparation_seconds,
        scoring_wall_seconds=elapsed-preparation_seconds,scoring_seconds_by_branch=branch_seconds,
        versions=list(versions),includes_query_off=True,
        early_ema_threshold_seconds=EARLY_EMA_THRESHOLD_SECONDS,
    )
    write_json(timing_path,timing)
    summary={}
    for branch,versions in result.items():
        summary[branch]={key:{field:row[field] for field in
                              ("development","legacy","same_surface_step_m_inv","data_gaps","checkpoint",
                               "checkpoint_sha256","query_pair_protocol")}
                         for key,row in versions.items()}
    legacy=legacy_baseline()
    difference=adoption(result["original"]["ema"],result["same-surface"]["ema"])
    return result,dict(by_branch=summary,legacy_1c_5k_ema=legacy,runtime=timing,
                       p2_candidate_minus_original_db=difference["p2_candidate_minus_original_db"],
                       effect_smaller_than_scene_spread=difference["effect_smaller_than_scene_spread"])


def estimate_1b(training_seconds, evaluation_seconds, full_evaluation_seconds, ema_only, now):
    evaluation_remaining=evaluation_seconds+full_evaluation_seconds if ema_only else 2*evaluation_seconds
    remaining=2*training_seconds+evaluation_remaining
    return dict(
        measured_at_utc=now.isoformat(),first_10000_training_seconds=training_seconds,
        first_10000_evaluation_seconds=evaluation_seconds,full_evaluation_seconds=full_evaluation_seconds,
        estimated_remaining_seconds=remaining,estimated_completion_utc=(now+timedelta(seconds=remaining)).isoformat(),
        method="首轮 10,000 步训练墙钟外推两段；20,000 步评价沿用首轮耗时，终点完整评价沿用 1a 已缓存评分耗时或同口径首轮。",
    )


def reference_reports(run):
    reports=[]
    for path in sorted((run/"reference-pools").glob("step-*.json")):
        snapshot=load(path)
        measured={}
        for branch in BRANCHES:
            usage=load(run/"1b"/branch/"reference-usage"/path.name)
            if usage:
                measured[branch]=usage
        reports.append(dict(snapshot,actual_B_usage_by_branch=measured))
    return reports


def follow(run,checkout,production_record):
    run.mkdir(parents=True,exist_ok=True)
    with (run/"followup.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        runner=Runner(run)
        try:
            while not (run/"benchmark.json").is_file():
                dispatch=load(run/"dispatch.json")
                if dispatch.get("status")=="halted":
                    raise RuntimeError(dispatch["reason"])
                time.sleep(10)
            bench=load(run/"benchmark.json")
            weight=load(RECORD)["same_surface_weight"]["selected"]
            if not load(run/"milestones.json").get("1a_start"):
                registry=load(ROOT.parent/"route3-cache/route3-stage1-gpu-usage.json")
                register_milestone(run,checkout,{"1a_start":dict(
                    dispatch=load(run/"dispatch.json"),benchmark=bench,optical=optical_counts(),GPU_assignment=registry,
                    decision_metric="median per-scene/direction same-surface edge absolute p95, 0–2 px",
                )},f"同表面权重为 {weight:g}，1a 两支按 {bench['adopted']} 测速选项开跑。")
            checkpoints=wait_for_1a(run)
            schedule=[torch.load(path,map_location="cpu",weights_only=True)["batches"] for path in checkpoints.values()]
            if len(schedule[0])!=TOTAL or schedule[0]!=schedule[1]:
                raise ValueError("1a checkpoint batch schedules are unequal or incomplete")
            runner.state=load(runner.summary)
            runner.state.pop("reason",None)
            runner.state.pop("halted_at_utc",None)
            runner.status(status="evaluating_1a")
            _,summary=evaluate_milestone(run,"1a",TOTAL,checkpoints,weight)
            early_ema=summary["runtime"]["wall_seconds"]>EARLY_EMA_THRESHOLD_SECONDS
            summary["early_1b_ema_only"]=early_ema
            recovery=load(run/"evaluation"/"development-F2-recovery.json")
            if recovery:
                summary["development_f2_recovery"]=recovery
            register_milestone(run,checkout,{"1a_endpoint":summary},"两支 1a 已完成 30,000 步，终点评价已登记。")
            gate=wait_for_1b(run,runner,production_record)
            register_milestone(run,checkout,{"1b_start":gate},"两支 1b 已具备光学与 F2 数据，继续配对训练。")
            mode=bench["adopted"]
            outputs={b:run/"1b"/b for b in BRANCHES}
            for step in (10_000,20_000,30_000):
                runner.status(status="1b_running",mode=mode,target_step=step)
                groups=[BRANCHES] if mode=="parallel" else [(b,) for b in BRANCHES]
                traces=[runner.group("1b",mode,outputs,group,step,checkpoints) for group in groups]
                paths={b:checkpoint_path(outputs[b],step) for b in BRANCHES}
                data=[torch.load(paths[b],weights_only=True,map_location="cpu") for b in BRANCHES]
                if data[0]["batches"]!=data[1]["batches"]:
                    raise ValueError(f"1b branch schedules differ at step {step}")
                if step==10_000:
                    elapsed=[row["measured_seconds"] for row in data]
                    training_seconds=(sum(trace["seconds"] for trace in traces) if all(traces) else
                                      max(elapsed) if mode=="parallel" else sum(elapsed))
                del data
                runner.status(status="evaluating_1b",step=step)
                results,endpoint=evaluate_milestone(run,"1b",step,paths,weight,ema_only=early_ema and step<TOTAL)
                snapshot={f"1b_step_{step}":endpoint,
                          "1b_reference_pools":reference_reports(run)}
                if step==10_000:
                    snapshot["1b_finish_estimate_after_10000"]=estimate_1b(
                        training_seconds,endpoint["runtime"]["wall_seconds"],
                        summary["runtime"]["scoring_wall_seconds"],early_ema,datetime.now(timezone.utc))
                if step==30_000:
                    snapshot["decision"]=adoption(results["original"]["ema"],results["same-surface"]["ema"])
                label=snapshot.get("decision",{}).get("selected")
                note=f"两支 1b 已完成 {step:,} 步并登记评价。"
                if label:
                    note+=f"按预定规则采用 {label} 支。"
                register_milestone(run,checkout,snapshot,note)
            runner.status(status="complete",decision=snapshot["decision"],finished_at_utc=datetime.now(timezone.utc).isoformat())
        except Exception as error:
            runner.state=load(runner.summary)
            runner.status(status="halted",reason=str(error),halted_at_utc=datetime.now(timezone.utc).isoformat())
            raise


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,default=OUTPUT)
    parser.add_argument("--publish-worktree",type=Path,required=True)
    parser.add_argument("--production-record",type=Path,required=True)
    args=parser.parse_args()
    follow(args.run.resolve(),args.publish_worktree.resolve(),args.production_record.resolve())
