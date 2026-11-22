"""Give registered training workloads priority over additional optical slots."""
from datetime import datetime,timezone
from pathlib import Path
import time

import psutil

from route3_v2data_B_batch import FORMAL_DATA, ROOT, write_json
from route3_v2data_B_plan import load


REGISTRY=ROOT.parent/"route3-cache/route3-stage1-gpu-usage.json"
RECEIPT=FORMAL_DATA/"p2-review/stage1-optical-sharing.json"
F2_REVIEW=FORMAL_DATA/"p2-review/formal-dryrun-b"
WINDOW=24
MAX_DEVICE_MIB=30*1024


def register_training(phase,pids,progress_paths):
    """The registry accepts Stage 1, Stage 2 and Stage 3 progress readers."""
    write_json(REGISTRY,dict(phase=phase,training_pids=pids,progress_paths=progress_paths,
                             gpu_index=0,uses_data_lock=False,combined_device_limit_gib=30))


def training_workers():
    registration=load(REGISTRY)
    if not registration or not str(registration.get("phase","")).startswith(("1","2","3")):
        return {}
    workers={}
    for branch,pid in registration["training_pids"].items():
        try:
            command=psutil.Process(pid).cmdline()
        except (psutil.NoSuchProcess,psutil.ZombieProcess):
            continue
        explicit=registration.get("progress_paths",{}).get(branch)
        if explicit:
            progress=Path(explicit)
        elif "--output" in command and any(Path(arg).name=="route3_v2data_stage1.py" for arg in command):
            progress=Path(command[command.index("--output")+1])/"progress.json"
        else:
            continue
        if progress.is_file():
            workers[(registration["phase"],branch,pid)]=progress
    return workers


def measurement_for(phase):
    existing=load(RECEIPT)
    if not existing:
        return None
    return existing.get("measurements",{}).get(phase) or (
        existing if existing.get("phase","1a")==phase else None
    )


def optical_width():
    workers=training_workers()
    if not workers:
        return None
    phase=next(iter(workers))[0]
    return (measurement_for(phase) or {}).get("optical_slots",2)


class TrainingOverlap:
    def __init__(self,enabled):
        workers=training_workers() if enabled else {}
        self.phase=next(iter(workers))[0] if workers else None
        self.workers=workers if self.phase and not measurement_for(self.phase) else {}
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
    def throughput(first,last):
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
        # The renderer is at a lock boundary. Neither training worker acquires
        # the data lock or has its process interrupted.
        time.sleep(WINDOW)
        self.baseline=self.throughput(before,self.sample())

    def measure_with_optics(self,elapsed):
        if self.baseline is None or measurement_for(self.phase) or elapsed<15:
            return
        if self.concurrent is None:
            self.concurrent=self.sample()
            return
        measured=self.throughput(self.concurrent,self.sample())
        if measured is None:
            return
        ratio=self.baseline["steps_per_second"]/measured["steps_per_second"]
        previous=load(RECEIPT)
        history=previous.get("measurements",{}) if previous else {}
        if previous and not history:
            history[previous.get("phase","1a")]=previous
        decision=dict(
            phase=self.phase,measured_at_utc=datetime.now(timezone.utc).isoformat(),
            training_workers=len(self.workers),training_without_optics=self.baseline,
            training_with_optics=measured,training_step_time_ratio=ratio,
            optical_slots=1 if ratio>1.5 else 2,threshold=1.5,memory_limit_gib=30,
        )
        history[self.phase]=decision
        write_json(RECEIPT,dict(decision,measurements=history))


def f2_failures():
    return set(load(F2_REVIEW/"F2-failures.json"))


def wait_for_incremental_f2(names):
    if not names:
        return
    worker=load(F2_REVIEW/"F2-worker.json")
    if not worker:
        return
    root=FORMAL_DATA/"derived-f2"
    while names-f2_failures():
        pending={name for name in names-f2_failures()
                 if not all((root/name/f"f{frame:03d}"/"manifest.json").is_file() for frame in (1,30,60))}
        if not pending or (FORMAL_DATA/"stop-after-current").is_file():
            return
        try:
            process=psutil.Process(worker["pid"])
            if not any(Path(arg).name=="route3_v2data_B_dryrun_p2.py" for arg in process.cmdline()):
                return
        except (psutil.NoSuchProcess,psutil.ZombieProcess):
            return
        time.sleep(3)
