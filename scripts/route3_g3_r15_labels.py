"""Reapply the frozen placement rule without changing the revision 14 observations."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import time
import numpy as np

from route3_g3_r15 import DEFAULT, ORIGINAL, NEW, OUT, PREV, CPU, PY, LOG, qa, link, read, write

CONTEXT_ROOT=OUT/'supervision'
VERSION=DEFAULT.name


def scene_labels(source, target, name, rule):
    link(source,target,[name])
    # The 256 builder deterministically uses the same inputs and K16
    # correctness labels. Re-evaluate only the O1/O2 placement decision.
    qa.run([CPU,'scripts/route3_s2a_data.py','--scene',name,'--ordered-root',str(target),
            '--size','256','--ordered-rule',rule],name+'-ordered256-'+target.name)


def full_labels_for(root, name, rule):
    from route3_s2a_candidates import labels
    labels(root/name,ordered=rule)


def main():
    selection=OUT/'R2/selection.json'
    while not (selection.exists() and (OUT/'split.json').exists()):
        qa.guard()
        time.sleep(3)
    rule=read(selection)['rule']
    split=read(OUT/'split.json')
    names=split['train']+split['heldout']
    start=time.monotonic()
    with ThreadPoolExecutor(max_workers=4) as workers:
        def one(name):
            if name == NEW:
                # The large assets for this scene are made by R1.
                while not (OUT/'R1-ready.json').exists():
                    qa.guard()
                    time.sleep(4)
            scene_labels(DEFAULT,CONTEXT_ROOT/VERSION,name,rule)
            scene_labels(ORIGINAL,CONTEXT_ROOT/'original',name,rule)
            old_root=PREV/'geometry/depthsplat-official'
            if name!=NEW and rule=='O1':
                # This full-resolution placement was already verified in r14;
                # its inputs, physical radius, and rule are identical. Keep
                # the raw labels as references, audited by the evaluator.
                link(old_root/name,DEFAULT/name,['labels'])
            else:
                qa.run([PY,'scripts/route3_g3_r15_labels.py','--full',str(DEFAULT),'--scene',name,
                        '--rule',rule],name+'-ordered512')
            if name==NEW:
                qa.run([PY,'scripts/route3_g3_r15_labels.py','--full',str(ORIGINAL),'--scene',name,
                        '--rule',rule],name+'-original512')
            print('R2_LABELS_READY',name,rule,flush=True)
        for f in [workers.submit(one,n) for n in names]:
            f.result()
    write(OUT/'R2/labels-ready.json',dict(rule=rule,scenes=names,split=split,seconds=time.monotonic()-start,
          recomputed_256=True, identical_512_O1_references=True if rule=='O1' else False,
          content_labels_independent_of_order=True))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--full',type=Path)
    p.add_argument('--scene')
    p.add_argument('--rule',default='O1')
    a=p.parse_args()
    full_labels_for(a.full,a.scene,a.rule) if a.full else main()
