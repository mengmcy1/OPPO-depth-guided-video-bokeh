"""Resume fixed G3 split with per-scene 2048/4096/8192 sampling checks.

Truth is independent of the geometry frontend. Historical 1024-spp files are
read through immutable references; new renders and all records use a new root.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT/'configs/route3-g3-r13-2026-11-18.json'
os.environ['ROUTE3_PILOT_CONFIG'] = str(CONFIG)
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
import cv2
import numpy as np
import torch
from route3_s2a_proxy import read, display
from route3_upper_bound_evaluate import score
from route3_ldr_u1r import model

C = json.loads(CONFIG.read_text())
OUT = Path(C['output'])
LOG = ROOT/'logs/route3-g3-r13-2026-11-18'
REC = ROOT/'03_实验记录/records'
OLD = ROOT/'outputs/route3-g3-s2a-2026-11-17-01'
BLENDER = C['cycles']['binary']


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def guard(gpu=False):
    if (OUT.parent/'STOP.json').exists():
        raise RuntimeError('STOP: recorded condition pending; no further jobs')
    free = shutil.disk_usage(OUT).free/2**30
    if free < 60:
        write(OUT.parent/'STOP.json', dict(reason='disk below60GiB', free_GiB=free))
        raise RuntimeError('STOP disk below60GiB')
    if gpu:
        assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
        with (LOG/'gpu-checks.log').open('a') as f:
            subprocess.run(['nvidia-smi'], stdout=f, check=True)


def run(cmd, name, gpu=False):
    guard(gpu)
    start = time.monotonic()
    with (LOG/'gpu0.lock').open('a') as lock, (LOG/(name+'.log')).open('a') as f:
        if gpu:
            fcntl.flock(lock, fcntl.LOCK_EX)
            guard(True)
        subprocess.run(cmd, cwd=ROOT, env=os.environ.copy(), stdout=f,
                       stderr=subprocess.STDOUT, check=True)
    return time.monotonic()-start


def jobs(scene, tag, rows):
    path = scene/(tag+'-jobs.json')
    if path.exists():
        assert json.loads(path.read_text()) == rows
    else:
        write(path, rows)
    manifest = path.with_name(path.stem+'-results.json')
    completed = json.loads(manifest.read_text()) if manifest.exists() else []
    done = {x['id'] for x in completed}
    for row in rows:
        if row['id'] not in done and Path(row['output']).exists():
            completed.append(dict(**row, reused_existing=True,
                                  source=str(Path(row['output']).resolve())))
    write(manifest, completed)
    if not all(Path(x['output']).exists() for x in rows):
        run([BLENDER, '-b', '-t', '12', '--python',
             'scripts/route3_metric_blender.py', '--', '--jobs', str(path)],
            scene.name+'-'+tag, True)
    missing = [x['output'] for x in rows if not Path(x['output']).exists()]
    if missing:
        raise RuntimeError('Blender did not produce requested files: '+str(missing))
    return json.loads(manifest.read_text())


def prepare(scene, index):
    scene.mkdir(exist_ok=True)
    old = OLD/scene.name
    if old.exists():
        for f in C['input_frames']:
            required = [old/'inputs'/f'f{f:03d}.{ext}' for ext in ['exr', 'png']]
            required += [old/'inputs'/f'f{f:03d}-{kind}-{f:04d}.exr'
                         for kind in ['Position', 'Depth', 'IndexOB']]
            if not all(p.exists() for p in required):
                raise RuntimeError('Historical input reference incomplete; never write through its link')
        # Link only immutable source assets, not scene.json/check records (the
        # old checker wrote method-independent QA into those files).
        for name in ['scene.blend', 'scene-geometry.npz', 'authoring', 'inputs', 'alias']:
            p = scene/name
            if not p.exists():
                p.symlink_to(old/name, target_is_directory=(old/name).is_dir())
        if not (scene/'scene.json').exists():
            shutil.copy2(old/'scene.json', scene/'scene.json')
        write(scene/'historical-source.json', dict(source=str(old), files_read_only=True,
              historical_1024_outputs_retained=True))
    elif not (scene/'scene.blend').exists():
        run([BLENDER, '-b', '-t', '12', '--python', 'scripts/route3_metric_blender.py',
             '--', '--build', str(scene), '--seed', str(C['seeds'][index-1]),
             '--index', str(index)], scene.name+'-build')
    common = dict(scene=scene.name, blend=str(scene/'scene.blend'), size=512,
                  seed=20261112, omit_reserved_passes=True)
    rows = [dict(**common, id=scene.name+f'-input-{f}', frame=f, spp=256,
                 dof=False, data_passes=True, data_pass_names=['Position', 'Depth', 'IndexOB'],
                 cryptomatte=False, output=str(scene/'inputs'/f'f{f:03d}.exr'))
            for f in C['input_frames']]
    jobs(scene, 'inputs-r13', rows)
    for f in C['input_frames']:
        path = scene/'inputs'/f'f{f:03d}.png'
        if not path.exists():
            cv2.imwrite(str(path), np.uint8(np.rint(display(read(path.with_suffix('.exr')))[..., ::-1]*255)))
    if not (scene/'alias/manifest.json').exists():
        meta = json.loads((scene/'scene.json').read_text())
        p = scene/'alias-plan-r13.json'
        write(p, [dict(scene=scene.name, blend=str(scene/'scene.blend'),
                       objects=meta['objects'], output=str(scene/'alias'))])
        run([BLENDER, '-b', '-t', '12', '--python', 'scripts/route3_step2_alias_blender.py',
             '--', '--plan', str(p)], scene.name+'-alias', True)
    return common


def surface_checks(scene):
    """Same fixed period/phase checks as G2, without its old sampling branch."""
    meta = json.loads((scene/'alias/manifest.json').read_text())
    objects = {r['id']: r for r in meta['objects']}
    geom = np.load(scene/'scene-geometry.npz')
    stats = []
    for f in C['input_frames']:
        P = read(scene/'inputs'/f'f{f:03d}-Position-{f:04d}.exr')
        ids = read(scene/'inputs'/f'f{f:03d}-IndexOB-{f:04d}.exr')[..., 0].round().astype(int)
        dy, dx = np.gradient(P.astype(np.float64), axis=(0, 1))
        normal = np.cross(dx, dy)
        normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-12)
        view = geom['camera_c2w_opencv'][f-1, :3, 3]-P
        view /= np.maximum(np.linalg.norm(view, axis=-1, keepdims=True), 1e-12)
        grazing = abs((normal*view).sum(-1)) < np.cos(np.deg2rad(75))
        for oid, obj in objects.items():
            period = np.full(ids.shape, np.inf)
            for pattern in obj['patterns']:
                if 'error' in pattern:
                    raise RuntimeError('Unsupported texture coordinates '+str(pattern))
                G = np.array(pattern['world_phase_gradient_cycles_m'])
                frequency = np.sqrt((dx@G.T)**2+(dy@G.T)**2).max(-1)
                period = np.minimum(period, 1/np.maximum(frequency, 1e-12))
            interior = cv2.erode((ids == oid).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            valid = interior & ~grazing & np.isfinite(period) & (period < 1e10)
            n = int(valid.sum())
            q = float(np.quantile(period[valid], .01)) if n else None
            stats.append(dict(frame=f, object_id=oid, object_name=obj['name'],
                              valid_pixels=n, grazing_pixels=int((interior & grazing).sum()),
                              period_p01_px=q, passed=q is None or q >= 8))
    alias = {r['label']: r for r in meta['rows']}
    base, phase, noise = [read(alias[k]['path']) for k in ['center', 'shift', 'noise']]
    ids = read(alias['center']['passes']['IndexOB'])[..., 0].round().astype(int)
    shifted = cv2.warpAffine(base, np.array([[1, 0, -.5], [0, 1, 0]], np.float32),
                             (512, 512), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT101)
    rows = []
    for oid, obj in objects.items():
        valid = cv2.erode((ids == oid).astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
        if valid.sum() < 25:
            rows.append(dict(object_name=obj['name'], pixels=int(valid.sum()), passed=True,
                             insufficient_interior=True))
            continue
        e = float(np.mean((display(phase)[valid]-display(shifted)[valid])**2))
        nr = float(np.mean((display(noise)[valid]-display(base)[valid])**2))
        excess = max(e-nr, 0)**.5
        rows.append(dict(object_name=obj['name'], pixels=int(valid.sum()),
                         excess_RMSE=excess, passed=excess <= .01))
    err = np.mean((display(base).astype(np.float64)-display(read(scene/'inputs/f030.exr')))**2)
    zero = float(-10*np.log10(max(err, 1e-30)))
    return dict(periods=stats, alias=rows, zero_aperture_repeat_PSNR=zero,
                period_pass=all(r['passed'] for r in stats),
                alias_pass=all(r['passed'] for r in rows), zero_pass=zero >= 50)


def sampling(scene, common, net, *, category_local=False):
    audit = []
    for spp in C['sampling']['levels']:
        rows = []
        for focus in [1.4, 5.]:
            for n, seed in [(spp//2, 20261112), (spp, 20261112), (spp, 2026111202)]:
                path = scene/'sampling'/f'spp{n}-seed{seed}-F{focus:g}.exr'
                if not path.exists() and n == 1024 and seed == 20261112:
                    old = OLD/scene.name/'dof'/f'f030-N1.4-F{focus:g}.exr'
                    if old.exists():
                        path.parent.mkdir(exist_ok=True)
                        path.symlink_to(old)
                rows.append(dict(**{**common, 'seed': seed}, id=scene.name+path.stem,
                                 frame=30, spp=n, dof=True, focus=focus, fstop=1.4, output=str(path)))
        jobs(scene, f'sampling-{spp}', rows)
        checks = []
        for focus in [1.4, 5.]:
            current = display(read(scene/'sampling'/f'spp{spp}-seed20261112-F{focus:g}.exr'))
            for label, n, seed in [('convergence', spp//2, 20261112), ('independent', spp, 2026111202)]:
                previous = display(read(scene/'sampling'/f'spp{n}-seed{seed}-F{focus:g}.exr'))
                result = score(previous, current, dict(full_image=np.ones((512, 512), bool)), net)['full_image']
                ok = result['psnr'] >= 45 and result['patch_lpips'] <= .002
                checks.append(dict(focus_m=focus, kind=label, lower_spp=n, reference_spp=spp,
                                   passed=ok, **result))
        passed = all(x['passed'] for x in checks)
        audit.append(dict(spp=spp, checks=checks, passed=passed))
        write(scene/'sampling-checks.json', dict(levels=audit, final_spp=spp if passed else None,
                                               passed=passed, thresholds=C['sampling']))
        print('G3_SAMPLING', scene.name, spp, passed,
              [(r['kind'], r['focus_m'], r['patch_lpips']) for r in checks], flush=True)
        if passed:
            return spp, audit
    if category_local:
        return None, audit
    write(OUT.parent/'STOP.json', dict(reason='scene failed sampling at8192spp',
                                      scene=scene.name, levels=audit))
    raise RuntimeError('STOP sampling at8192spp')


def truth():
    torch.set_num_threads(4)
    net = model()
    progress = []
    for index, name in enumerate(C['scenes'], 1):
        scene = OUT/name
        ready = scene/'truth-ready-r13.json'
        if ready.exists():
            progress.append(json.loads(ready.read_text()))
            continue
        start = time.monotonic()
        common = prepare(scene, index)
        check = surface_checks(scene)
        write(scene/'surface-checks-r13.json', check)
        known_surface_failures = sum(
            not (r['surface_checks']['period_pass'] and r['surface_checks']['alias_pass']
                 and r['surface_checks']['zero_pass']) for r in progress)
        current_surface_failed = not (check['period_pass'] and check['alias_pass'] and check['zero_pass'])
        if known_surface_failures + int(current_surface_failed) >= 2:
            write(OUT.parent/'STOP.json', dict(reason='batch surface validity failure',
                  scene=name, surface_checks=check, preceding_rows=progress,
                  stop_before_starting_additional_renders=True))
            return
        spp, audit = sampling(scene, common, net)
        rows = []
        for f in [1, 30, 60]:
            for focus in [1.4, 5.]:
                for stop in [1.4, 2.]:
                    cid = f'f{f:03d}-N{stop:g}-F{focus:g}'
                    path = scene/'dof'/(cid+'.exr')
                    sampled = scene/'sampling'/f'spp{spp}-seed20261112-F{focus:g}.exr'
                    if f == 30 and stop == 1.4 and not path.exists():
                        path.parent.mkdir(exist_ok=True)
                        path.symlink_to(sampled)
                    rows.append(dict(**common, id=name+'-'+cid, frame=f, spp=spp, dof=True,
                                     focus=focus, fstop=stop, output=str(path)))
        jobs(scene, f'formal-dof-{spp}', rows)
        passed = check['period_pass'] and check['alias_pass'] and check['zero_pass']
        result = dict(scene=name, seed=C['seeds'][index-1], split='heldout' if name in C['split']['heldout'] else 'train',
                      final_spp=spp, sampling=audit, surface_checks=check,
                      seconds=time.monotonic()-start,
                      own_disk_GiB=sum(p.stat().st_size for p in scene.rglob('*') if p.is_file() and not p.is_symlink())/2**30,
                      passed=bool(passed), method_scores_used=False)
        write(ready, result)
        progress.append(result)
        write(OUT.parent/'truth-progress.json', dict(rows=progress, total=16))
        write(REC/(C['record_prefix']+'-truth-progress.json'), dict(rows=progress, total=16))
        print('G3_TRUTH_READY', name, spp, passed, result['seconds'], flush=True)
        if sum(not r['passed'] for r in progress) >= 2:
            write(OUT.parent/'STOP.json', dict(reason='batch surface validity failure', rows=progress))
            return
    write(OUT/'truth-ready.json', dict(rows=progress, all_passed=all(r['passed'] for r in progress), split=C['split']))


def truth_aux():
    """Geometry-independent K16 labels/peels. No estimated depth is consumed."""
    cpu = '/root/autodl-tmp/envs/dofgs/bin/python'
    master = []
    for name in C['scenes']:
        scene = OUT/name
        while not (scene/'truth-ready-r13.json').exists():
            if (OUT.parent/'STOP.json').exists():
                return
            time.sleep(5)
        if (scene/'truth-aux-ready.json').exists():
            continue
        if not json.loads((scene/'truth-ready-r13.json').read_text())['passed']:
            continue
        start = time.monotonic()
        for mode in ['center', 'aperture', 'training_focus']:
            run([cpu, 'scripts/route3_s2a_geometry.py', '--scene', str(scene), '--mode', mode], name+'-'+mode)
        req = json.loads((scene/'geometry/required-exclusions.json').read_text())['frames']
        names = {o['id']: o['name'] for o in json.loads((scene/'scene.json').read_text())['objects']}
        peels = []
        for frame, sets in req.items():
            for ids in sets:
                key = f'f{int(frame):03d}-exclude-'+'-'.join(map(str, ids))
                peels.append(dict(scene=name, id=name+'-'+key, blend=str(scene/'scene.blend'),
                                  frame=int(frame), spp=256, size=512, seed=20261112,
                                  omit_reserved_passes=True, dof=False, hidden_object_ids=ids,
                                  hide_names=[names[i] for i in ids], output=str(scene/'references'/(key+'.exr'))))
        master += jobs(scene, 'formal-peels-r13', peels)
        write(OUT/'peel-jobs-results.json', master)
        run([cpu, 'scripts/route3_s2a_reference_geometry.py', '--scene', str(scene)], name+'-peel-geometry')
        run([cpu, 'scripts/route3_pilot_oracle.py', '--scene', str(scene)], name+'-oracle')
        # Temporal/rack supervision is not used by S2a but remains part of the
        # G3 master data; no extra reserved AOVs outside the original pilots.
        spp = json.loads((scene/'truth-ready-r13.json').read_text())['final_spp']
        temporal = []
        for frame in C['continuous_frames']:
            for mode, focus in [('continuous', 5.), ('focus-rack', float(1/np.interp(frame, [27, 34], [1/1.4, 1/5.])) )]:
                temporal.append(dict(scene=name, id=name+f'-{mode}-{frame}', blend=str(scene/'scene.blend'),
                                     frame=frame, size=512, seed=20261112, spp=spp, omit_reserved_passes=True,
                                     dof=True, focus=focus, fstop=1.4,
                                     output=str(scene/mode/f'f{frame:03d}.exr')))
        jobs(scene, 'temporal-r13', temporal)
        write(scene/'truth-aux-ready.json', dict(scene=name, seconds=time.monotonic()-start,
              native_reserved_passes=False, K=16, training_focuses_per_target=4,
              geometry_independent=True, complete=True))
        print('G3_TRUTH_AUX_READY', name, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['truth', 'truth_aux'])
    args = parser.parse_args()
    globals()[args.mode]()
