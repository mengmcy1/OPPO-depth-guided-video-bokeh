"""Appendix 13 revision 14 truth preparation and category-local checks."""
import argparse
import json
import os
from pathlib import Path
import threading
import time

import route3_g3_r13 as qa

ROOT = qa.ROOT
PREFIX = 'route3-g3-r14-2026-11-17'
OUT = ROOT/'outputs'/(PREFIX+'-01')
TRUTH = OUT/'truth'
LOG = ROOT/'logs'/PREFIX
REC = qa.REC
PREVIOUS = ROOT/'outputs/route3-g3-r13-2026-11-18-01/truth'
PY = '/root/autodl-tmp/envs/route3/bin/python'
CPU = '/root/autodl-tmp/envs/dofgs/bin/python'
CONFIG = ROOT/'configs'/(PREFIX+'.json')
C = json.loads(CONFIG.read_text())
for directory in [OUT, TRUTH, LOG]:
    directory.mkdir(exist_ok=True)
qa.OUT, qa.LOG, qa.OLD, qa.C = TRUTH, LOG, OUT/'unused-historical-root', C
os.environ['ROUTE3_PILOT_CONFIG'] = str(CONFIG)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.{threading.get_ident()}.part')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def read(path):
    return json.loads(path.read_text())


def reference_files(source, destination, names):
    destination.mkdir(parents=True, exist_ok=True)
    for name in names:
        src, dest = source/name, destination/name
        if src.exists() and not dest.exists():
            dest.symlink_to(src.resolve(), target_is_directory=src.is_dir())


def reuse_scene(name):
    source, dest = PREVIOUS/name, TRUTH/name
    reference_files(source, dest, ['inputs', 'dof', 'scene.blend', 'scene-geometry.npz', 'scene.json',
                                   'truth-ready-r13.json', 'alias', 'sampling', 'sampling-checks.json'])
    write(dest/'quality.json', dict(read(source/'truth-ready-r13.json'), reused=str(source),
                                    seconds=0., own_disk_GiB=0.))
    return dest


def truth_scene(name, net):
    scene = TRUTH/name
    if (scene/'quality.json').exists():
        return read(scene/'quality.json')
    index = C['scenes'].index(name)+1
    thin = name.endswith('-thin')
    start = time.monotonic()
    scene.mkdir(exist_ok=True)
    if not (scene/'scene.blend').exists():
        qa.run([qa.BLENDER, '-b', '-t', '12', '--python', 'scripts/route3_metric_blender.py',
                '--', '--build', str(scene), '--seed', str(C['seeds'][index-1]), '--index', str(index)]
               + (['--thin-floor'] if thin else []), name+'-build')
    common = qa.prepare(scene, index)
    check = qa.surface_checks(scene)
    write(scene/'surface-checks.json', check)
    spp, samples = qa.sampling(scene, common, net, category_local=True)
    passed = spp is not None and all(check[k] for k in ['period_pass', 'alias_pass', 'zero_pass'])
    if passed:
        rows = []
        for frame in C['representative_frames']:
            for focus in C['focus_distances_m']:
                for stop in C['fstops']:
                    key = f'f{frame:03d}-N{stop:g}-F{focus:g}'
                    dest = scene/'dof'/(key+'.exr')
                    if frame == 30 and stop == 1.4:
                        dest.parent.mkdir(exist_ok=True)
                        if not dest.exists():
                            dest.symlink_to(scene/'sampling'/f'spp{spp}-seed20261112-F{focus:g}.exr')
                    rows.append(dict(**common, id=key, frame=frame, spp=spp, dof=True,
                                     focus=focus, fstop=stop, output=str(dest)))
        qa.jobs(scene, 'formal-dof', rows)
    result = dict(scene=name, seed=C['seeds'][index-1], final_spp=spp, sampling=samples,
                  surface_checks=check, passed=passed, seconds=time.monotonic()-start,
                  own_disk_GiB=sum(p.stat().st_size for p in scene.rglob('*') if p.is_file() and not p.is_symlink())/2**30,
                  texture_floors=read(scene/'scene.json').get('bar_texture_floor', []))
    write(scene/'quality.json', result)
    print('T2_QUALITY', name, spp, passed, flush=True)
    return result


def truth(category):
    qa.torch.set_num_threads(4)
    net = qa.model()
    rows = []
    for name in C['scenes']:
        if not name.endswith('-'+category):
            continue
        previous = PREVIOUS/name/'truth-ready-r13.json'
        if category != 'thin' and previous.exists() and read(previous)['passed']:
            row = read(reuse_scene(name)/'quality.json')
        else:
            row = truth_scene(name, net)
        rows.append(dict(scene=name, passed=row['passed'], final_spp=row['final_spp']))
        write(OUT/f'{category}-progress.json', rows)
        # All five corrected thin probes are explicitly requested. Once any
        # fails, thin is excluded as a class from downstream data/training.
        if category != 'thin' and not row['passed']:
            break
    write(OUT/f'{category}-done.json', dict(rows=rows, passed=all(r['passed'] for r in rows)))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('category', choices=['broad', 'multilayer', 'thin'])
    truth(p.parse_args().category)
