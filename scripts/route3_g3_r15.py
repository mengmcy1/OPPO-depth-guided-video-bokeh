"""Revision 15 run roots and the remaining fixed G3 scene."""
import argparse
import fcntl
import os
from pathlib import Path

import route3_g3_r14 as previous
import route3_g3_r14_data as data

ROOT, REC, PY, CPU = previous.ROOT, previous.REC, previous.PY, previous.CPU
read, write, qa = previous.read, previous.write, previous.qa
link = previous.reference_files
PREV = previous.OUT
PREFIX = 'route3-g3-r15-2026-11-17'
OUT = ROOT/'outputs'/f'{PREFIX}-01'
TRUTH = OUT/'truth'
LOG = ROOT/'logs'/PREFIX
GEOMETRY = OUT/'geometry'
ORIGINAL = GEOMETRY/'original'
DEFAULT = GEOMETRY/'depthsplat-official'
NEW = 'metric-scene-122116-broad'
C = dict(previous.C, output=str(TRUTH), run_output=str(OUT), record_prefix=PREFIX,
         protocol_revision='e87a75a')
for p in (TRUTH, LOG, GEOMETRY):
    p.mkdir(parents=True, exist_ok=True)
qa.OUT, qa.LOG, qa.C = TRUTH, LOG, C
previous.TRUTH, previous.OUT, previous.LOG, previous.C = TRUTH, OUT, LOG, C
data.TRUTH, data.OUT, data.LOG, data.BASE, data.NATIVE = TRUTH, OUT, LOG, ORIGINAL, GEOMETRY/'native'


def final_split(new_valid):
    split = {k:[n for n in C['split'][k] if not n.endswith('-thin') and '122113' not in n and
                (n != NEW or new_valid)] for k in ('train','heldout')}
    write(OUT/'split.json',split)
    for n in split['train']+split['heldout']:
        if n == NEW:
            continue
        link(PREV/'truth',TRUTH,[n])
        link(PREV/'geometry/original',ORIGINAL,[n])
        source=PREV/'geometry/depthsplat-official'/n
        link(source,DEFAULT/n,[p.name for p in source.iterdir() if p.name!='labels'])
        link(PREV/'geometry/depthsplat-official/learning-data',DEFAULT/'learning-data',[n])
        link(PREV/'geometry/original/learning-data',ORIGINAL/'learning-data',[n])
    return split


def prepare():
    qa.torch.set_num_threads(4)
    result=previous.truth_scene(NEW,qa.model())
    split=final_split(result['passed'])
    write(OUT/'R1-quality.json',dict(scene=NEW,quality=result,split=split,
          broad_prior_failure='122113-broad', broad_paused=not result['passed']))
    if result['passed']:
        os.environ['ROUTE3_SHARP_SAVE_ALL']='1'
        data.frontend(NEW)
        data.truth_aux(NEW)
        with (LOG/'gpu0.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            qa.guard(True)
            data.dominant(NEW,root=ORIGINAL)
        data.prepare_labels(NEW,ORIGINAL)
        data.color_anchor(NEW)
        data.geometry_version(NEW,'depthsplat-official',DEFAULT)
        data.temporal(NEW)
    write(OUT/'R1-ready.json',dict(split=split,new_scene_passed=result['passed']))


if __name__ == '__main__':
    prepare()
