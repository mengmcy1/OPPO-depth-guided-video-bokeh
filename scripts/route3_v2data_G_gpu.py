"""Run G and R preprocessing under the shared GPU data lock."""
import argparse
from collections import Counter, defaultdict
import resource
import time

import cv2
import torch

from route3_depthsplat_depth import network
from route3_ovg_r2_priors import MODEL
from route3_v2data_G_common import DATA, ROOT, SpaceError, locked_cuda, read, write
from route3_v2data_R_acquire import TARGET as R_TARGET


def ready_g():
    root = DATA / 'G'
    found = defaultdict(list)
    done = Counter()
    acquired = [read(path) for path in (root/'acquisition').glob('*.json')]
    selected = {row['environment']: set(row['windows']) for row in acquired if row.get('complete')}
    for path in sorted((root / 'windows').glob('*/manifest.json')):
        environment = path.parent.name.split('-Data_', 1)[0]
        if environment in selected and path.parent.name not in selected[environment]:
            continue
        if (path.parent/'prior/summary.json').exists():
            done[environment] += 1
        elif not (path.parent/'prior-error.json').exists():
            found[environment].append(path)
    order = sorted((done[environment] + i, environment, path)
                   for environment, paths in found.items() for i, path in enumerate(paths))
    return [path for _, _, path in order]


def ready_r():
    root = DATA / 'R'
    plan_path = root/'sampling-plan.json'
    old = sorted((root/'windows').glob('*/retained.json'))
    if not plan_path.exists():
        return [p.parent/'manifest.json' for p in old if not (p.parent/'readiness.json').exists()], False
    plan = read(plan_path)
    results = []
    valid = 0
    for identifier in plan['existing'] + plan['candidates']:
        dest = root/'windows'/identifier
        receipt = dest/'readiness.json'
        acquire = root/'acquisition'/f'{identifier}.json'
        if receipt.exists():
            valid += bool(read(receipt)['passed'])
            if valid == R_TARGET:
                return [], True
        elif (dest/'retained.json').exists() or (acquire.exists() and read(acquire)['passed']):
            results.append(dest/'manifest.json')
        elif not acquire.exists():
            break
    return results[:R_TARGET-valid], False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--only', choices=['G', 'R', 'both'], default='both')
    parser.add_argument('--max-windows', type=int)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    count = 0
    net = None
    geos = None
    active = set('GR' if args.only == 'both' else args.only)
    from route3_v2data_G_priors import check_sharp_reference, models, precalculate as g_prior
    from route3_v2data_R_priors import precalculate as r_prior
    while active:
        r_paths, r_done = ready_r() if 'R' in active else ([], False)
        g_paths = ready_g() if 'G' in active else []
        if r_done:
            write(DATA/'R'/'target-reached.json', dict(verified_scenes=R_TARGET))
            active.discard('R')
        for tier, paths in (('G', g_paths), ('R', r_paths)):
            if tier not in active or paths:
                continue
            root = DATA/tier
            if (root/'acquisition-complete.json').exists() or (root/'acquisition-stop.json').exists():
                active.discard(tier)
            if tier == 'R' and (root/'access.json').exists() and read(root/'access.json')['status'] != 'authorized':
                active.discard('R')
        jobs = [('R', path) for path in r_paths[:64]] + [('G', path) for path in g_paths[:8]]
        if not jobs:
            if not args.watch or not active:
                break
            time.sleep(4)
            continue
        with locked_cuda():
            if net is not None:
                net.cuda()
            if geos is not None:
                geos[0].cuda()
                geos[1].cuda()
            try:
                for tier, path in jobs:
                    if tier not in active:
                        continue
                    if tier == 'G' and (path.parent/'prior/summary.json').exists():
                        continue
                    if tier == 'R' and (path.parent/'readiness.json').exists():
                        continue
                    receipt = path.parent/('prior-error.json' if tier == 'G' else 'readiness.json')
                    try:
                        if net is None and (tier == 'G' or not (path.parent/'visibility.json').exists()):
                            net, _ = network(ROOT.parent/'depthsplat-r13-20261117', ROOT.parent/'dinov2-depthsplat-r13',
                                             ROOT/'models/DepthSplat'/MODEL)
                            net = net.cuda().eval()
                        if tier == 'G':
                            if geos is None:
                                geos = models()
                            if not (DATA/'G'/'sharp-identity-check.json').exists():
                                check_sharp_reference(geos[0], path)
                            g_prior(net, *geos, path)
                        else:
                            r_prior(net, path)
                        metrics = DATA/tier/'gpu-metrics.json'
                        old = read(metrics) if metrics.exists() else {}
                        report = dict(max_reserved_bytes=max(torch.cuda.max_memory_reserved(), old.get('max_reserved_bytes', 0)),
                                      peak_rss_kib=max(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                                                       old.get('peak_rss_kib', 0)))
                        write(metrics, report)
                        print('WINDOW', tier, path.parent.name, report, flush=True)
                    except (SpaceError, MemoryError) as error:
                        write(DATA/tier/'gpu-stop.json', dict(reason=str(error)))
                        active.discard(tier)
                    except (OSError, ValueError, RuntimeError, KeyError) as error:
                        torch.cuda.empty_cache()
                        write(receipt, dict(passed=False, reason=str(error)))
                        print('WINDOW_ERROR', tier, path.parent.name, type(error).__name__, str(error), flush=True)
                    count += 1
                    if args.max_windows and count >= args.max_windows:
                        return
            finally:
                if net is not None:
                    net.cpu()
                if geos is not None:
                    geos[0].cpu()
                    geos[1].cpu()
                torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
