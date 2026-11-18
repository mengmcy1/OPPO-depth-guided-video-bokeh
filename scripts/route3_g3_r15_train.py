"""Formal S2a with 256/5 supervision and controlled training ablations."""
import argparse
import fcntl
import time

import torch

from route3_g3_r15 import OUT, LOG, PY, CPU, qa, read, write, link
from route3_g3_r15_labels import CONTEXT_ROOT, VERSION
from route3_g3_r14_train import train_network


def main(mode):
    split=read(OUT/'split.json')
    variant='original' if mode=='original_geometry' else VERSION
    root=CONTEXT_ROOT/variant
    first=split['train'][0]
    warmup=CONTEXT_ROOT/VERSION/'learning-data'/first/'256-5-GTwarmup'
    rule=read(OUT/'R2/selection.json')['rule']
    qa.run([CPU,'scripts/route3_s2a_data.py','--scene',first,'--size','256','--GT-warmup',
            '--ordered-root',str(CONTEXT_ROOT/VERSION),'--ordered-rule',rule],'GT-warmup')
    while not all((OUT/p).exists() for p in ['R1-ready.json','R2/labels-ready.json','R2/placements512-ready.json']):
        qa.guard()
        time.sleep(4)
    if mode!='main':
        while not (OUT/'R3/main/training.json').exists():
            if (OUT/'R3/main/optimization-failure.json').exists():
                return
            qa.guard()
            time.sleep(4)
    if variant!=VERSION:
        link(warmup.parent,root/'learning-data'/first,[warmup.name])
    dest=OUT/'R3'/mode
    torch.set_num_threads(4)
    ablation = mode != 'main'
    lock_name = f'gpu0-fit-{mode}.lock' if ablation else 'gpu0.lock'
    with (LOG/lock_name).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        qa.guard(True)
        if ablation:
            free, _ = torch.cuda.mem_get_info()
            if free < 24 * 2**30:
                raise RuntimeError('GPU memory budget for overlapping ablation unavailable')
            torch.cuda.set_per_process_memory_fraction(.15)
        if not (dest/'training.json').exists():
            write(dest/'protocol.json',dict(split=split,geometry=variant,rule=rule,F2=True,
                    checkpoint='20000 final, no heldout selection',start='200 steps GT warmup then cached E1 depth',
                    pose_perturbation='Not applied in S2a; the 2.5 curriculum specifies pose perturbation in S2b',
                    ablation_GPU_memory_fraction=.15 if ablation else None,
                    activation='equal positive/negative means' if mode=='balanced_activation' else 'positive weight 4'))
            train_network(root,dest,split['train'],split['heldout'],
                          balance_activation=mode=='balanced_activation',require_descent=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['main','balanced_activation','original_geometry'])
    main(p.parse_args().mode)
