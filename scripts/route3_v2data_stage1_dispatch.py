"""Supervise the paired Stage 1 runs, with a single checkpoint restart."""
import argparse
from datetime import datetime,timezone
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import torch

from route3_v2data_B_batch import FORMAL_DATA,GIB,write_json
from route3_v2data_B_plan import load
from route3_v2data_stage1 import (
    BRANCHES,OUTPUT,ROOT,TOTAL,TrainingData,checkpoint_path,complete_optical,frozen_inventory,verify_disk,
)


PROFILE_STEPS=500
DEVICE_LIMIT_MIB=30*1024
TASK_REGISTRY=ROOT.parent/"route3-cache/route3-stage1-gpu-usage.json"
WORKER=ROOT/"scripts/route3_v2data_stage1.py"


def gpu_memory_mib():
    value=subprocess.check_output(
        ["nvidia-smi","-i","0","--query-gpu=memory.used","--format=csv,noheader,nounits"],text=True,
    )
    return int(value.strip())


def stamp():
    return datetime.now(timezone.utc).isoformat()


def checkpoint_step(path):
    value=load(path/"progress.json")
    return value.get("step",0) if value else 0


class Runner:
    def __init__(self,run):
        self.run=run
        self.summary=run/"dispatch.json"
        self.state=load(self.summary) or {}
        self.affinity=[list(range(23,27)),list(range(27,31))]
        if not set(sum(self.affinity,[])).issubset(os.sched_getaffinity(0)):
            raise ValueError("individual training-worker CPU masks are unavailable")

    def status(self,**values):
        self.state.update(values)
        write_json(self.summary,self.state)

    def record_gpu(self,phase,label,jobs):
        write_json(TASK_REGISTRY,dict(
            owner_pid=os.getpid(), training_pids={branch:job.pid for branch,job in jobs.items() if job.poll() is None},
            phase=phase,mode=label,gpu_index=0, combined_device_limit_gib=30,
            uses_data_lock=False,started_at_utc=stamp(),
        ))

    def group(self,phase,label,outputs,branches,steps,sources):
        parallel=len(branches)==2
        if all(checkpoint_path(outputs[branch],steps).is_file() for branch in branches):
            return None
        start=time.monotonic()
        jobs,streams,attempts={},[],{}
        peak=0

        def start_worker(branch):
            attempts[branch]=attempts.get(branch,0)+1
            path=outputs[branch]
            path.mkdir(parents=True,exist_ok=True)
            limit=9.0 if parallel else 16.0
            cores=self.affinity[BRANCHES.index(branch)]
            args=["taskset","-c",",".join(map(str,cores)),sys.executable,"-u",str(WORKER),
                  "--run",str(self.run),"--output",str(path),"--phase",phase,"--branch",branch,
                  "--steps",str(steps),"--memory-gib",str(limit),"--threads",str(len(cores))]
            if sources.get(branch):
                args+=["--start",str(sources[branch])]
            env=dict(os.environ, CUDA_VISIBLE_DEVICES="0",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",
                     PYTHONPATH=f"{ROOT}/scripts:{ROOT}")
            stream=(path/"training.log").open("ab")
            streams.append(stream)
            jobs[branch]=subprocess.Popen(args,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
            self.record_gpu(phase,label,jobs)

        try:
            for branch in branches:
                if not checkpoint_path(outputs[branch],steps).is_file():
                    start_worker(branch)
            while jobs:
                peak=max(peak,gpu_memory_mib())
                if peak>DEVICE_LIMIT_MIB:
                    raise RuntimeError(f"combined device memory was {peak} MiB, above 30 GiB")
                for branch,job in list(jobs.items()):
                    status=job.poll()
                    if status is None:
                        continue
                    del jobs[branch]
                    if status!=0 or not checkpoint_path(outputs[branch],steps).is_file():
                        if attempts[branch]==1:
                            self.status(restarting=dict(phase=phase,branch=branch,exit_code=status,at_utc=stamp()))
                            start_worker(branch)
                        else:
                            raise RuntimeError(f"{phase} {branch} failed twice; inspect {outputs[branch]}/training.log")
                    self.record_gpu(phase,label,jobs)
                if jobs:
                    time.sleep(.5)
        finally:
            for job in jobs.values():
                if job.poll() is None:
                    job.terminate()
            for job in jobs.values():
                job.wait()
            for stream in streams:
                stream.close()
            self.record_gpu(phase,label,{})
        return dict(seconds=time.monotonic()-start,peak_combined_gpu_mib=peak,attempts=attempts)

    def benchmark(self):
        result_path=self.run/"benchmark.json"
        if result_path.is_file():
            return load(result_path)
        runs={}
        for label in ("parallel","serial"):
            paths={b:self.run/"benchmark"/label/b for b in BRANCHES}
            groups=[BRANCHES] if label=="parallel" else [(b,) for b in BRANCHES]
            self.status(status="1a_benchmark",benchmark_mode=label)
            try:
                traces=[self.group("1a",f"benchmark_{label}",paths,g,PROFILE_STEPS,{}) for g in groups]
            except RuntimeError as error:
                if label=="serial":
                    raise
                runs[label]=dict(passed=False,failure=str(error))
                continue
            checkpoints=[torch.load(checkpoint_path(paths[b],PROFILE_STEPS),weights_only=True,map_location="cpu")
                         for b in BRANCHES]
            if checkpoints[0]["batches"]!=checkpoints[1]["batches"]:
                raise ValueError("benchmark branches did not use identical batches")
            measured=[trace for trace in traces if trace]
            seconds=sum(t["seconds"] for t in measured) if measured else sum(
                ck["measured_seconds"] for ck in checkpoints)
            runs[label]=dict(passed=True,steps_per_branch=PROFILE_STEPS,seconds=seconds,
                             combined_steps_per_second=PROFILE_STEPS*2/seconds,
                             peak_combined_gpu_mib=max((t["peak_combined_gpu_mib"] for t in measured),default=0),
                             traces=traces)
        mode=max((name for name,row in runs.items() if row["passed"]),key=lambda name:runs[name]["combined_steps_per_second"])
        result=dict(benchmarks=runs,adopted=mode,recorded_at_utc=stamp(),
                    optical_concurrent=True,scope="500 steps per branch in each compared mode")
        write_json(result_path,result)
        return result

    def run_phase(self,phase,mode,sources):
        paths={branch:self.run/phase/branch for branch in BRANCHES}
        self.status(status=f"{phase}_running",mode=mode,phase_started_at_utc=stamp())
        groups=[BRANCHES] if mode=="parallel" else [(b,) for b in BRANCHES]
        for group in groups:
            self.group(phase,mode,paths,group,TOTAL,sources)
        checkpoints={branch:checkpoint_path(paths[branch],TOTAL) for branch in BRANCHES}
        receipts=[torch.load(path,map_location="cpu",weights_only=True)["batches"] for path in checkpoints.values()]
        if receipts[0]!=receipts[1]:
            raise ValueError(f"{phase} branches did not use identical complete batch sequences")
        self.status(status=f"{phase}_training_complete",**{f"{phase}_checkpoints":{b:str(p) for b,p in checkpoints.items()}})
        return checkpoints


def supervise(run):
    run.mkdir(parents=True,exist_ok=True)
    with (run/"dispatch.lock").open("a") as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        runner=Runner(run)
        try:
            verify_disk(run)
            receipt=load(FORMAL_DATA/"manifest.json")
            all_done=receipt["counts"]["train_geometry"]+len({
                r["scene_id"] for r in load(FORMAL_DATA/"units/training/failures.json")
                if r.get("counts_toward_pause",True)})==300
            if not all_done:
                raise ValueError("B's sealed geometry queue is still active")
            frozen_inventory(TrainingData(),run)
            result=runner.benchmark()
            adopted=result["adopted"]
            sources={branch:checkpoint_path(run/"benchmark"/adopted/branch,PROFILE_STEPS) for branch in BRANCHES}
            runner.run_phase("1a",adopted,sources)
        except Exception as error:
            runner.status(status="halted",reason=str(error),halted_at_utc=stamp())
            raise


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,default=OUTPUT)
    supervise(parser.parse_args().run.resolve())
