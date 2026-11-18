"""Continue independent T2 work after the T4 selection, then eligible S2a."""
from concurrent.futures import ThreadPoolExecutor
import time

from route3_g3_r14 import C, OUT, PY, TRUTH, qa, read, write
from route3_g3_r14_data import BASE, run_stage
from route3_g3_r14_report import choose_geometry, get_split


def selected_precompute(name, version):
    root = OUT/'geometry'/version
    if version!='original':
        qa.run([PY,'scripts/route3_g3_r14_data.py','variant',name,'--version',version],name+'-selected-'+version)
    if not (root/name/'labels/summary.json').exists():
        qa.run([PY,'scripts/route3_g3_r14_data.py','full_labels',name,'--version',version],name+'-label512-'+version)


def continue_work():
    while True:
        qa.guard()
        split,geometry=get_split(),choose_geometry()
        if split and geometry:
            break
        time.sleep(20)
    cases=split['train']+split['heldout']
    names=[n for n in cases if (TRUTH/n/'quality.json').exists() and read(TRUTH/n/'quality.json')['passed']]
    version=geometry['default_geometry']
    failed, completed = [], []
    waiting=set(names)
    jobs={}
    with ThreadPoolExecutor(max_workers=2) as pool:
        while waiting or jobs:
            qa.guard()
            for n in names:
                if n not in waiting:
                    continue
                if (OUT/'geometry-errors'/f'{n}.json').exists():
                    failed.append(n)
                    waiting.remove(n)
                elif (BASE/n/'training-data-ready.json').exists() and (BASE/n/'visible/summary.json').exists():
                    jobs[n]=pool.submit(selected_precompute,n,version)
                    waiting.remove(n)
            for n,f in list(jobs.items()):
                if f.done():
                    try:
                        f.result()
                    except Exception as e:
                        failed.append(n)
                        write(OUT/'selected-errors'/f'{n}.json',dict(scene=n,reason=str(e)))
                    else:
                        completed.append(n)
                    del jobs[n]
            write(OUT/'selected-data-progress.json',dict(version=version,completed=completed,failed=failed))
            if waiting or jobs:
                time.sleep(15)
    write(OUT/'selected-data-ready.json',dict(scenes=completed,version=version,split=split,failed=failed))
    overfit_paths=[OUT/'overfit'/n/'training.json' for n in C['scenes'][:2]]
    if not all(p.exists() for p in overfit_paths):
        while not all(p.exists() for p in overfit_paths):
            qa.guard()
            time.sleep(10)
    if not all(read(p)['passed'] for p in overfit_paths):
        write(OUT/'S2a-blocked.json',dict(reason='single-scene direct fit stop',
              failed=[p.parent.name for p in overfit_paths if not read(p)['passed']]))
    elif failed or set(names)!=set(cases):
        write(OUT/'S2a-blocked.json',dict(reason='fixed split missing valid data', geometry_errors=failed,
                                        missing=sorted(set(cases)-set(names))))
    else:
        qa.run([PY,'scripts/route3_g3_r14_finish.py'],'S2a-controller')
    # The temporal ground truth is part of the master data, not a direct-loss
    # input to the 256/5 S2a run.
    for name in completed:
        run_stage(name,'temporal')


if __name__=='__main__':
    continue_work()
