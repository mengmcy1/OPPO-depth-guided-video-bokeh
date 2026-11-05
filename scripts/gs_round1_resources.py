"""Shared GPU reservations for the existing queue and the two COLMAP workers."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import sys


def alive(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().split()[2]!='Z'
    except OSError:return False


@contextmanager
def registry():
    base=Path(os.environ['TMPDIR']);path=base/'dof-gs-round1-gpu-reservations.json'
    with (base/'dof-gs-round1-gpu-reservations.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        try:data=json.loads(path.read_text())
        except (OSError,ValueError):data={}
        data={pid:row for pid,row in data.items() if alive(pid)}
        try:yield data
        finally:
            temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)


def usage():
    output=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
    return {pid.strip():float(memory.strip()) for pid,memory in (row.split(',') for row in output.splitlines() if row.strip())}


def attach(pid,memory,kind):
    with registry() as data:data[str(pid)]={'reserve_mib':memory,'kind':kind,'registered_at':time.time()}


@contextmanager
def gpu_slot(memory,kind,exclusive=False):
    pid=str(os.getpid())
    while True:
        with registry() as data:
            observed=usage();others=(set(data)|set(observed))-{pid}
            total=sum(max(data.get(p,{}).get('reserve_mib',0),observed.get(p,0)) for p in others)
            blocked_exclusive=any(data.get(p,{}).get('exclusive') for p in others)
            queue=Path('/root/autodl-tmp/OPPO-depth-guided-video-bokeh/logs/dof-gs-repro-2026-11-05-round1/queue.json')
            try:author_pending=any(j.get('kind')=='author' and j['status'] in {'queued','rendering'} for j in json.loads(queue.read_text())['jobs'])
            except (OSError,ValueError):author_pending=False
            replay_active=any(data.get(p,{}).get('kind','').startswith('author-') for p in others)
            replay_lane=author_pending and not kind.startswith('author-') and not replay_active and len(others)>=3
            if not blocked_exclusive and not replay_lane and (not exclusive or not others) and len(others)<4 and total+memory<=40960:
                data[pid]={'reserve_mib':memory,'kind':kind,'exclusive':exclusive,'registered_at':time.time()}
                break
        time.sleep(3)
    try:yield
    finally:
        with registry() as data:data.pop(pid,None)


if __name__=='__main__':
    memory,kind,*command=sys.argv[1:]
    with gpu_slot(int(memory),kind):
        os.execv(command[0],command)
