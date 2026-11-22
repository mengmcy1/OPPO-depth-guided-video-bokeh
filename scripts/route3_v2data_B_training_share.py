"""Measure training slowdown at an optical batch boundary on the shared 5090."""
from datetime import datetime,timezone
from pathlib import Path
import time

import psutil

from route3_v2data_B_batch import FORMAL_DATA, ROOT, write_json
from route3_v2data_B_plan import load


REGISTRY=ROOT.parent/"route3-cache/route3-stage1-gpu-usage.json"
RECEIPT=FORMAL_DATA/"p2-review/stage1-optical-sharing.json"
WINDOW=24
MAX_DEVICE_MIB=30*1024


def training_workers():
    registration=load(REGISTRY)
    if not registration or registration.get("phase")!="1a":
        return {}
    workers={}
    for branch,pid in registration["training_pids"].items():
        try:
            command=psutil.Process(pid).cmdline()
        except (psutil.NoSuchProcess,psutil.ZombieProcess):
            continue
        if "--output" in command and any(Path(arg).name=="route3_v2data_stage1.py" for arg in command):
            output=Path(command[command.index("--output")+1])
            progress=output/"progress.json"
            if progress.is_file():
                workers[(branch,pid)]=progress
    return workers


def optical_width():
    decision=load(RECEIPT)
    return (decision or {}).get("optical_slots",2) if training_workers() else None


class TrainingOverlap:
    def __init__(self, enabled):
        self.workers=training_workers() if enabled and not RECEIPT.exists() else {}
        self.baseline=None
        self.concurrent=None

    def sample(self):
        if training_workers()!=self.workers:
            return None
        rows={key:load(path) for key,path in self.workers.items()}
        if any(not row for row in rows.values()):
            return None
        return dict(when=time.monotonic(),steps={key:row["step"] for key,row in rows.items()})

    @staticmethod
    def throughput(first, last):
        if not first or not last:
            return None
        elapsed=last["when"]-first["when"]
        steps=sum(last["steps"][key]-value for key,value in first["steps"].items())
        if steps<=0 or elapsed<WINDOW:
            return None
        return dict(seconds=elapsed,completed_steps=steps,steps_per_second=steps/elapsed)

    def measure_without_optics(self):
        if not self.workers:
            return
        before=self.sample()
        if not before:
            return
        # The renderer already owns the data lock at this batch boundary. Neither
        # training worker acquires it or has its processes stopped.
        time.sleep(WINDOW)
        self.baseline=self.throughput(before,self.sample())

    def measure_with_optics(self, elapsed):
        if self.baseline is None or RECEIPT.exists() or elapsed<15:
            return
        if self.concurrent is None:
            self.concurrent=self.sample()
            return
        measured=self.throughput(self.concurrent,self.sample())
        if measured is None:
            return
        ratio=self.baseline["steps_per_second"]/measured["steps_per_second"]
        write_json(RECEIPT,dict(
            measured_at_utc=datetime.now(timezone.utc).isoformat(), training_workers=len(self.workers),
            training_without_optics=self.baseline, training_with_optics=measured,
            training_step_time_ratio=ratio, optical_slots=1 if ratio>1.5 else 2,
            threshold=1.5, memory_limit_gib=30,
        ))
