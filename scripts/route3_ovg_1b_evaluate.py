"""Paired 1b development comparison, keeping scene, focus and hidden IDs fixed."""
import argparse
from collections import defaultdict
import os
import subprocess

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')

import numpy as np
import torch

from route3_ovg_1b_common import CONFIG, OUT, SOURCE, guard, read
from route3_ovg_1b_data import setup
from route3_ovg_1b_optical_data import background_samples, b_oracle, geometry_root, place
from route3_ovg_1b_train import evaluate, load_model
from route3_ovg_subpixel import save_json, save_npz
from route3_ovg_footprint import query_net
from route3_g3_r14_train import learned_hidden
from route3_ldr_u1r import model as lpips
from route3_s2a_evaluate import masks
from route3_s2a_learn import Store
from route3_upper_bound_evaluate import display, exr, pool, score
import route3_ovg_data as data
from src.aperture_renderer import aperture_disk
from src.dual_layer import depth_band
from src.footprint_order import apply_footprint_order
from src.layered_aperture import merge_layers, pixel_layer
from src.ovg_optics import raster, visible_surfels


def summarize(rows):
    stats = defaultdict(list)
    for r in rows:
        for scene in (r['scene'], 'pooled'):
            key = (scene, r['protocol'], r['nominal_focus_m'], r['fstop'],
                   r['hidden'], r['visible'], r['region'])
            stats[key].append(r)
    names = ('scene', 'protocol', 'nominal_focus_m', 'fstop', 'hidden', 'visible', 'region')
    pooled = [dict(zip(names, key), **pool(group)) for key, group in sorted(stats.items())]
    paired = []
    for r in pooled:
        if r['visible'] not in ('OVG-1b-EMA', 'OVG-1b-raw'):
            continue
        cohort = [v for v in pooled if all(v[k] == r[k] for k in names if k != 'visible')]
        for reference in ('fixed-original', 'fixed-DepthSplat', 'OVG-1a'):
            ref = next((v for v in cohort if v['visible'] == reference), None)
            if ref is None or not r['pixels'] or not ref['pixels']:
                continue
            if ref['pixels'] != r['pixels']:
                raise ValueError('Paired evaluation mask changed')
            paired.append(dict({k: r[k] for k in names}, reference=reference,
                               PSNR_delta=r['psnr']-ref['psnr'], LPIPS_delta=r['patch_lpips']-ref['patch_lpips']))
    retention = []
    for query in (r for r in paired if r['hidden'] == 'Query'):
        gt = next(r for r in paired if r['hidden'] == 'GT' and
                  all(r[k] == query[k] for k in (*names, 'reference') if k != 'hidden'))
        gain = gt['PSNR_delta']
        retention.append(dict({k: query[k] for k in names if k != 'hidden'}, reference=query['reference'],
                              delta_with_GT_db=gain, delta_with_Query_db=query['PSNR_delta'],
                              retained_fraction=query['PSNR_delta']/gain if gain > .1 else None))
    return dict(pooled=pooled, matched_deltas=paired, Query_retention=retention)


@torch.inference_mode()
def optical_compare(models, *, reference_only=False, one_case=False):
    metric = lpips()
    store = Store()
    query = query_net(SOURCE/'V1/retrained/step-020000.pt')
    roots = data.PREVIOUS
    names = read(roots/'split.json')['heldout']
    count = CONFIG['native_eval_rays']
    tensor = lambda a: torch.as_tensor(a, device='cuda')
    group = 'fixed-reference-check' if one_case else 'reference' if reference_only else 'terminal'
    work = OUT/'P3'/group
    prior = OUT/'P3/reference/results.json'
    reuse = not one_case and not reference_only and prior.exists()
    existing = read(prior) if reuse else {'rows': [], 'placements': []}
    rows, placements = existing['rows'], existing['placements']
    for name in names:
        original = roots/'geometry/original'/name
        manifest = read(roots/'evaluation-input/original'/name/'512-13/manifest.json')
        entries = read(original/'prepared.json')['rows']
        for entry in entries:
            frame = entry['frame']
            if one_case and frame != 30:
                continue
            item = ('B', name, data.TRAIN_FRAMES.index(frame))
            x, _ = data.sample(item, 'cuda')
            observation = data.load(entry['input_npz'], ('depth', 'rgb_srgb'))
            K = x['K'][0]
            arr = x['rgb'][0].permute(1,2,0)
            eye = torch.eye(4,device='cuda')
            filename = f'f{frame:03d}.npz'
            ds_root = roots/'geometry/depthsplat-official'/name
            prepared = next(e for e in read(ds_root/'prepared.json')['rows'] if e['frame'] == frame)
            depths = {'fixed-original': tensor(observation['depth']),
                      'fixed-DepthSplat': tensor(data.load(prepared['input_npz'], ('depth',))['depth'])}
            vis = {} if reuse else {label: pixel_layer(arr, depth, K, eye) for label,depth in depths.items()}
            for label,net in models.items():
                prediction = net(x)
                depths[label] = prediction['depth1'][0]
                vis[label] = visible_surfels(prediction, x)
            spec = dict(tiers='B', entry=entry)
            a_center = background_samples(spec, K, 1.4, K.new_zeros((1,2)), (512,512))
            parents, gt_depth, background = b_oracle(name, frame)
            for row in (r for r in manifest['rows'] if r['frame'] == frame):
                case = row['case']
                cid, protocol = case['id'], case['protocol']
                if one_case and (case['nominal_focus_m'], case['fstop'], protocol) != (5., 1.4, 'P2'):
                    continue
                focus = case['focus_m']
                radius = .05 / (2 * case['fstop'])
                shape = (512,512)
                offsets = aperture_disk(count, radius, device=K.device, dtype=K.dtype)
                shared_background = background_samples(spec, K, focus, offsets, shape)
                true = geometry_root(name)
                target = display(exr(true/'dof'/(cid+'.exr')))
                regions = masks(true, case)
                regions = {key: regions[key] for key in ('full_image','in_focus','band','recoverable')}
                # All rows use the same potential hidden parent identities and
                # candidates. Only their geometry-compatible placement changes.
                support = depth_band(observation['depth'], float(K[0,0]), focus, radius)[0]
                uv = np.floor(parents['uv']).astype(int).clip(0,511)
                selected = dict(parents, recoverable=parents['recoverable'] & support[uv[:,1],uv[:,0]])
                source_query = learned_hidden(query, store, row, manifest, K, support)
                zero_ref = observation['rgb_srgb']
                dest = work/'render'/name/protocol/cid
                for label,cloud in vis.items():
                    aligned, stats = place(selected, gt_depth, background, depths[label], K, 512)
                    query_placed, query_stats = apply_footprint_order(source_query, depths[label], K)
                    placements.append(dict(scene=name, id=cid, protocol=protocol, visible=label,
                                           focus_m=focus, GT=stats, Query=query_stats))
                    for hidden_name, hidden in [('GT',aligned), ('Query',query_placed)]:
                        joined = merge_layers(cloud, hidden)
                        rgb = raster(joined, K, focus, offsets, *shape, shared_background).mean(0)
                        zero = raster(joined, K, focus, K.new_zeros((1,2)), *shape, a_center)[0]
                        pred, center = display(rgb.cpu().numpy()), display(zero.cpu().numpy())
                        save_npz(dest/f'{label}-{hidden_name}.npz', prediction_srgb=pred, zero_srgb=center)
                        indices = ('full_image', 'in_focus', 'recoverable')
                        metrics = score(pred, target, regions, metric)
                        metrics.update({'zero_'+k: v for k,v in score(center, zero_ref, {
                            k:regions[k] for k in indices}, metric).items()})
                        for region,value in metrics.items():
                            rows.append(dict(scene=name, frame=frame, id=cid, protocol=protocol,
                                             nominal_focus_m=case['nominal_focus_m'], fstop=case['fstop'],
                                             focus_m=focus, visible=label, hidden=hidden_name, region=region, **value))
                if one_case:
                    return rows, placements
                guard()
                print('OVG_1B_P3', name, protocol, cid, flush=True)
                save_json(work/'partial-optical.json', dict(rows=rows, placements=placements))
    return rows, placements


def main(one_case, prior_only, reference_only):
    setup()
    subprocess.run(['nvidia-smi'], check=True)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0' or torch.cuda.mem_get_info()[0] < 8*2**30:
        raise RuntimeError('Paired evaluation GPU budget unavailable')
    torch.set_num_threads(5)
    torch.cuda.set_per_process_memory_fraction(.62)
    first = load_model()
    save_json(OUT/'P3/1a-paired-prior-reference.json', evaluate(first, 30000))
    if prior_only:
        return
    ck = OUT/'checkpoints/step-030000.pt'
    models = {'OVG-1a': first} if not (OUT/'P3/reference/results.json').exists() or reference_only or one_case else {}
    if not one_case and not reference_only:
        if not ck.exists():
            raise FileNotFoundError('P3 full paired optical comparison requires the fixed 1b terminal checkpoint')
        models['OVG-1b-EMA'] = load_model(ck,'ema')
        models['OVG-1b-raw'] = load_model(ck,'model')
    rows, placements = optical_compare(models, reference_only=reference_only, one_case=one_case)
    report = summarize(rows)
    destination = 'fixed-reference-check' if one_case else 'reference' if reference_only else 'terminal'
    save_json(OUT/'P3'/destination/'results.json',
              dict(rows=rows, placements=placements, **report, K=64,
                   common_focus='original estimated geometry P2 focus and its P1 mapped focus, common to all arms',
                   hidden_IDs_fixed=True, Query_checkpoint='1a-r2 V1 step-020000',
                   retention='same-branch delta, suppressed where GT comparison gain ≤0.1 dB'))
    print('OVG_1B_P3_DONE', len(rows), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument('--one-reference-case', action='store_true')
    scope.add_argument('--prior-reference-only', action='store_true')
    scope.add_argument('--optical-reference-only', action='store_true')
    args = parser.parse_args()
    main(args.one_reference_case, args.prior_reference_only, args.optical_reference_only)
