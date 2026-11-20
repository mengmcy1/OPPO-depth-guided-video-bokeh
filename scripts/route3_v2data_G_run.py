"""Run both release lines, refresh manifests, and optionally publish the records."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from route3_v2data_G_common import DATA, DATE, RECORDS, ROOT, read, write
from route3_v2data_G_report import summarize_g, summarize_r


LOG = ROOT/'logs'/f'route3-v2data-{DATE}'
WORKERS = {
    'g': ('route3_v2data_G_acquire.py', [], 'G-acquire.log'),
    'r': ('route3_v2data_R_acquire.py', [], 'R-acquire-retry.log'),
    'gpu': ('route3_v2data_G_gpu.py', ['--watch'], 'GR-gpu.log'),
}
SOURCES = ('G_acquire', 'G_common', 'G_gpu', 'G_priors', 'G_report', 'G_run', 'R_acquire', 'R_priors')


def pid_running(pid, script):
    if pid is None:
        return False
    proc = Path('/proc')/str(pid)
    try:
        return script.encode() in (proc/'cmdline').read_bytes()
    except FileNotFoundError:
        return False


def start_worker(name):
    script, flags, logfile = WORKERS[name]
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    with (LOG/logfile).open('ab') as output:
        return subprocess.Popen([sys.executable, '-u', str(ROOT/'scripts'/script), *flags], cwd=ROOT,
                                env=env, stdout=output, stderr=subprocess.STDOUT)


def acquisition_done(tier):
    root = DATA/tier
    if (root/'acquisition-complete.json').exists() or (root/'acquisition-stop.json').exists():
        return True
    access = root/'access.json'
    return access.exists() and read(access)['status'] == 'pending credentials'


def verified_snapshot():
    summarize_g()
    summarize_r()
    counts = {tier: read(DATA/tier/'manifest.json')['achieved'] for tier in 'GR'}
    print('RELEASE_VERIFIED', counts, flush=True)
    return counts


def git_publish(worktree, message):
    def git(*args):
        return subprocess.run(['git', *args], cwd=worktree, check=True)
    git('pull', '--rebase', 'origin', 'main')
    sources = [ROOT/'scripts'/f'route3_v2data_{name}.py' for name in SOURCES]
    records = [RECORDS/f'route3-v2data-{tier}-{DATE}.json' for tier in 'GR']
    files = [str(path.relative_to(ROOT)) for path in sources + records]
    for path, relative in zip(sources + records, files):
        shutil.copyfile(path, worktree/relative)
    git('add', '--', *files)
    git('diff', '--check')
    git('diff', '--cached', '--check')
    git('status', '--short', '--branch')
    if subprocess.run(['git', 'diff', '--cached', '--quiet', '--', *files], cwd=worktree).returncode == 1:
        git('commit', '-m', message, '--', *files)
    for attempt in range(3):
        git('pull', '--rebase', 'origin', 'main')
        if subprocess.run(['git', 'push', 'origin', 'HEAD:main'], cwd=worktree).returncode == 0:
            return
        time.sleep((attempt+1)*5)
    raise RuntimeError('Could not push the G/R release commit after three attempts')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in WORKERS:
        parser.add_argument(f'--{name}-pid', type=int)
    parser.add_argument('--git-worktree', type=Path)
    parser.add_argument('--publish-snapshot', action='store_true')
    args = parser.parse_args()
    LOG.mkdir(parents=True, exist_ok=True)
    if args.publish_snapshot:
        verified_snapshot()
        if args.git_worktree:
            git_publish(args.git_worktree, '实现路线三 G/R 扩充管线并记录首批验收')
        return
    children = {}
    pids = {name: getattr(args, f'{name}_pid') for name in WORKERS}
    attempts = {name: 0 for name in WORKERS}
    previous_report = 0.
    # A coordinator plus three worker processes. Download threads are I/O bound.
    while True:
        cpu_done = {name: acquisition_done(name.upper()) for name in ('g', 'r')}
        for name, (script, _, _) in WORKERS.items():
            if name in cpu_done and cpu_done[name]:
                continue
            if pid_running(pids[name], script):
                continue
            child = children.get(name)
            status = child.poll() if child is not None else None
            if name == 'gpu' and all(cpu_done.values()) and (status == 0 or attempts[name] >= 2):
                break
            if attempts[name] >= 2:
                if name == 'gpu':
                    for tier in 'GR':
                        write(DATA/tier/'gpu-stop.json', dict(reason='GPU worker terminated twice', exit_code=status))
                    continue
                write(DATA/name.upper()/'acquisition-stop.json',
                      dict(reason='Acquisition worker terminated twice', exit_code=status))
                continue
            children[name] = start_worker(name)
            pids[name] = children[name].pid
            attempts[name] += 1
        if time.monotonic() - previous_report >= 120:
            verified_snapshot()
            previous_report = time.monotonic()
        if all(acquisition_done(tier) for tier in 'GR') and not pid_running(pids['gpu'], WORKERS['gpu'][0]):
            break
        time.sleep(20)
    verified_snapshot()
    if args.git_worktree:
        git_publish(args.git_worktree, '记录路线三 G/R 扩充数据验收与实际差额')


if __name__ == '__main__':
    main()
