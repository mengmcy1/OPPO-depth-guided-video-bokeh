"""Heldout evaluation using one candidate set for S2a and the candidate Oracle."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
from pathlib import Path
import time

import numpy as np
import torch
from scipy.ndimage import maximum_filter

from route3_g3_r15 import C, DEFAULT, LOG, ORIGINAL, OUT, TRUTH, qa, read, write
from route3_g3_r15_labels import CONTEXT_ROOT
from route3_g3_r15_observations import Observations, surfaces
from route3_g3_r14_train import learned_hidden
from route3_depth_interface import subject_mask
from route3_explicit_layer import aligned_hidden
from route3_g3_geometry import fit_and_map, load, zero_parts
from route3_ldr_u1r import model as lpips
from route3_local_replacement import PROFILE
from route3_b2_oracle_3d import sparse_pixel_layer
from route3_s2a_baselines import scene
from route3_s2a_data import input_cache_512, save_npz, views_for
from route3_s2a_evaluate import masks
from route3_s2a_learn import Store, evaluate_row
from route3_s2a_proxy import read as read_exr
from route3_step2_diagnostic import fit_background
from route3_upper_bound_evaluate import display, score
from src.aperture_renderer import GaussianScene
from src.dual_layer import depth_band
from src.hidden_retrieval_rules import aggregate, collect
from src.layered_aperture import merge_layers, pixel_layer
from src.light_candidate_fusion import LightCandidateFusion
from src.ordered_layers import order_hidden, ray_samples, render_fallbacks

FRAMES = C['input_frames']


def descriptor(folder, entries):
    return dict(model=folder / 'models/SHARP', calibration=folder / 'models/calibration/SHARP',
                confidence=folder / 'models/calibration', source_root=folder, entries=entries)


def cases_for(name, folder):
    a_root = ORIGINAL / name
    fitting, output = [], []
    fg = load(folder / 'scene-geometry.npz')['foreground_object_ids']
    for entry in read(folder / 'prepared.json')['rows']:
        f = entry['frame']
        a_maps = load(a_root / 'A-depth' / f'f{f:03d}.npz')
        g = load(folder / 'geometry' / f'f{f:03d}-center.npz')
        z = load(entry['input_npz'])['depth']
        visible_bg = g['center_visible'] & ~np.isin(g['object_id'], fg)
        physical = {tag: fit_background(g['depth_axial_m'], d, visible_bg)
                    for tag, d in [('D', z), ('A', a_maps['depth'])]}
        fitting.append(dict(frame=f, **physical))
        dom = (a_maps['cluster_fraction'] >= .9) & (a_maps['contribution_alpha'] > 1e-6) & (a_maps['cluster_depth'] > 0)
        configurations = []
        for c in entry['cases']:
            nominal = c['nominal_focus_m']
            subject = subject_mask(g, nominal) & dom
            if not subject.any():
                raise ValueError(f'No A P2 subject {name} {c["id"]}')
            configurations.append(dict(c, protocol='P2', focus_a=float(np.median(a_maps['cluster_depth'][subject]))))
            if all(row['success'] for row in physical.values()):
                focus = {tag: 1 / (row['a'] / nominal + row['b']) for tag, row in physical.items()}
                configurations.append(dict(c, protocol='P1', focus_m=focus['D'], focus_a=focus['A']))
        output.append(dict(entry, cases=configurations))
    return output, fitting


def prepare(name, root):
    folder = root / name
    cache = OUT / 'evaluation-input' / root.name / name
    ready = cache / 'ready.json'
    if ready.exists():
        return read(ready)
    entries, fits = cases_for(name, folder)
    cache.mkdir(parents=True, exist_ok=True)
    inputs = cache / '512-13'
    inputs.mkdir(exist_ok=True)
    desc = descriptor(folder, entries)
    input_cache_512(name, desc, inputs)
    manifest = read(inputs / 'manifest.json')
    ctx = load(manifest['context'])
    native = surfaces(folder, FRAMES)
    rule = read(OUT / 'R2/selection.json')['rule']
    for e in entries:
        f = e['frame']
        local = inputs / f'f{f:03d}'
        inp = load(local / 'inputs.npz')
        valid = inp['candidate_valid']
        g = load(folder / 'geometry' / f'f{f:03d}-center.npz')
        # Same candidate slots and source pixels that Store.forward reads.
        obs = Observations.from_sources(native, ctx['c2w'], FRAMES.index(f), inp['target_K'],
                                       g['depth_axial_m'], inp['source_slots'][valid], inp['source_xy'][valid])
        fit = load(folder / 'placement' / f'f{f:03d}.npz')
        depth, camera, support = obs.placement(fit, inp['target_depth'], rule)
        color, appearance = obs.colors(load(folder / 'oracle' / f'f{f:03d}.npz'))
        # Audit/reuse the R2 label recalculation for cached raw candidate rows.
        revised = OUT / 'R2/placement512' / name / f'f{f:03d}' / 'candidates.npz'
        if root == DEFAULT and revised.exists():
            pool = load(e['pool'])
            ref = load(revised)
            def code(slot, xy):
                return slot.astype(np.int64) * 512**2 + xy[:, 1].astype(np.int64) * 512 + xy[:, 0]
            raw = code(pool['slot'], pool['source_xy'])
            key = code(obs.slots, obs.xy)
            sorter = np.argsort(raw)
            lookup = np.searchsorted(raw[sorter], key).clip(0, len(raw)-1)
            hit = raw[sorter[lookup]] == key
            row = sorter[lookup[hit]]
            if not np.array_equal(ref['valid'][row], support[hit]) or not np.allclose(ref['depth'][row][support[hit]], depth[hit][support[hit]], atol=2e-4):
                raise RuntimeError(f'R2 placement reproducibility mismatch: {name} {f}')
            depth[hit], camera[hit], support[hit] = ref['depth'][row], ref['camera'][row], ref['valid'][row]
        def packed(array):
            target = np.zeros((*valid.shape, *array.shape[1:]), array.dtype)
            target[valid] = array
            return target
        shared = dict(candidate_depth_target=packed(depth), candidate_camera_target=packed(camera),
                      candidate_RGB_target=packed(color), placement_support_v2=packed(support),
                      appearance_valid=packed(appearance))
        save_npz(local / 'shared-labels.npz', shared)
        for c in [r for r in e['cases'] if r['protocol'] == 'P2']:
            cid = c['id']
            path = folder / 'geometry' / (cid + '-K16.npz')
            geo = load(path)
            save_npz(local / (cid + '-labels.npz'),
                     dict(content_hit_fraction=packed(obs.hit_fraction(geo, c['nominal_focus_m'])),
                          hidden_any_at_query=geo['hidden'].any(0).ravel()[inp['query_pixel']],
                          recoverable_at_query=geo['mask_recoverable_total'].ravel()[inp['query_pixel']], **shared))
        print('R3_EVAL_INPUT', root.name, name, f, flush=True)
    result = dict(path=str(inputs / 'manifest.json'), fits=fits, geometry=root.name, scene=name)
    write(ready, result)
    return result


def oracle_hidden(inp, labels, indices, K):
    correct = (labels['content_hit_fraction'][indices] > 0) & inp['candidate_valid'][indices]
    supported = correct & labels['appearance_valid'][indices] & labels['placement_support_v2'][indices]
    best = np.where(supported, labels['content_hit_fraction'][indices], 0).argmax(-1)
    use = supported.any(-1)
    i, j = indices[use], best[use]
    audit = dict(queries=len(indices), correct_available=int(correct.any(-1).sum()),
                 correctly_labelled_activations=int(use.sum()),
                 available_but_no_supported_placement_or_color=int((correct.any(-1) & ~use).sum()))
    if not len(i):
        return None, audit
    xyz = torch.as_tensor(labels['candidate_camera_target'][i, j], device='cuda')
    rgb = torch.as_tensor(labels['candidate_RGB_target'][i, j], device='cuda')
    return one_slot_gaussians(xyz, rgb, K), audit


def one_slot_gaussians(xyz, rgb, K):
    sigma = xyz[:, 2] * .7 / K[0, 0]
    scale = torch.stack([sigma, sigma, sigma * .01], dim=-1)
    quat = torch.zeros((len(xyz), 4), device='cuda')
    quat[:, 0] = 1
    alpha = torch.full_like(sigma, .999)
    return GaussianScene(xyz, quat, scale, alpha, rgb, torch.ones_like(alpha, dtype=torch.long))


def matched_rule_hidden(inp, indices, K, depth):
    """Rule ablation with the same top eight candidates as the learned model."""
    query, candidate = np.nonzero(inp['candidate_valid'][indices])
    rows = indices[query]
    feature = inp['features'][rows, candidate]
    packed = dict(pixel=inp['query_pixel'][rows], depth=feature[:, 3], colors=feature[:, 4:7],
                  edge=feature[:, 15].astype(bool), slot=inp['source_slots'][rows, candidate])
    rule, _ = aggregate(packed, 2, depth)
    if not len(rule['uv']):
        return None
    uv = torch.as_tensor(rule['uv'], device='cuda')
    rays = torch.cat([uv, torch.ones_like(uv[:, :1])], -1) @ torch.linalg.inv(K).T
    xyz = rays * torch.as_tensor(rule['depth'], device='cuda')[:, None]
    rgb = torch.as_tensor(rule['colors_linear'], device='cuda')
    return one_slot_gaussians(xyz, rgb, K)


def zero_psnr(image, target, mask):
    error = (image[mask].astype(np.float64) - target[mask]) ** 2
    return float(-10 * np.log10(max(error.mean(), 1e-30))) if len(error) else None


def evaluate_scene(name, root, modes, metric):
    ready = prepare(name, root)
    manifest = read(Path(ready['path']))
    folder = root / name
    store = Store()
    rule = read(OUT / 'R2/selection.json')['rule']
    views = views_for(descriptor(folder, []), FRAMES, 512)
    fg = load(folder / 'scene-geometry.npz')['foreground_object_ids']
    tensor = lambda x: torch.as_tensor(x, device='cuda')
    eye = torch.eye(4, device='cuda')
    zero_known = {}
    for e in read(folder / 'prepared.json')['rows']:
        f = e['frame']
        d = load(e['input_npz'])
        depth, K = tensor(d['depth']), tensor(d['K'])
        D = pixel_layer(tensor(d['rgb_linear']), depth, K, eye)
        a_root = ORIGINAL / name
        a_entry = next(x for x in read(a_root / 'prepared.json')['rows'] if x['frame'] == f)
        A = scene(a_root / 'visible' / f'f{f:03d}.pt')
        a_maps = load(a_root / 'A-depth' / f'f{f:03d}.npz')
        fill = tensor(load(a_entry['fill'])['fill_rgb_linear'])
        bg = tensor(load(a_entry['input_npz'])['background_depth'])
        parent = load(folder / 'oracle' / f'f{f:03d}.npz')
        g = load(folder / 'geometry' / f'f{f:03d}-center.npz')
        _, _, supported, mapped, valid, _ = fit_and_map(g, d['depth'], fg, parent)
        alignment = dict(parent_depth_aligned=mapped, parent_valid=valid, local_supported=supported)
        invalid = ~load(folder / 'models/calibration/SHARP' / f'f{f:03d}.npz')['dominant_valid']
        a_r, a_alpha = zero_parts(A, K)
        a_zero = a_r + (1 - a_alpha[..., None]) * fill
        base_retrieval = None
        for row in [x for x in manifest['rows'] if x['frame'] == f]:
            c = row['case']
            cid, protocol = c['id'], c['protocol']
            path = OUT / 'R3/renders' / root.name / name / protocol / cid
            result = path / 'scores.json'
            previous = read(result) if result.exists() else None
            pending = {key: net for key, net in modes.items() if not previous or key not in previous['query']}
            if not pending:
                continue
            begin = time.monotonic()
            focus, focus_a, radius = c['focus_m'], c['focus_a'], .05 / (2 * c['fstop'])
            band = depth_band(d['depth'], d['K'][0, 0], focus, radius)[0]
            a_band = depth_band(a_maps['depth'], d['K'][0, 0], focus_a, radius)[0]
            bg_order = tensor(maximum_filter(d['depth'], 49)) if rule == 'O2' else None
            inp = store.inputs(row['input'])
            labels_path = Path(row['input']).parent / (cid + '-labels.npz')
            labels = store.inputs(str(labels_path))
            row['target'] = str(labels_path)
            active = store.active(row['input'], focus, c['fstop'])
            samples = ray_samples(A, K, focus, radius, 64, fill, bg)
            parts = []
            if previous is None:
                hidden, _ = aligned_hidden(parent, alignment, band, K)
                hidden, _ = order_hidden(hidden, depth, K, rule, bg_order)
                candidate, audit = oracle_hidden(inp, labels, active, K)
                matched_rule = matched_rule_hidden(inp, active, K, d['depth'])
                matched_rule, _ = order_hidden(matched_rule, depth, K, rule, bg_order)
                if protocol == 'P2':
                    if base_retrieval is None:
                        base_retrieval, _ = aggregate(load(e['pool']), 2, a_maps['depth'])
                    retrieval = base_retrieval
                else:
                    pool = collect(d['depth'], d['K'], d['c2w'], band | a_band, views, f)
                    retrieval, _ = aggregate(pool, 2, a_maps['depth'])
                uv = retrieval['uv'].astype(int).clip(0, 511)
                select = a_band[uv[:, 1], uv[:, 0]]
                ret = sparse_pixel_layer(retrieval, select, K, eye, PROFILE, 'cuda') if select.any() else None
                a_samples = samples if focus == focus_a else ray_samples(A, K, focus_a, radius, 64, fill, bg)
                a_image = a_samples.mean(0).cpu().numpy()
                parts = [('A', A, None, focus_a, a_samples), ('D0', D, None, focus, samples),
                         ('D1_Oracle', D, hidden, focus, samples), ('Candidate_Oracle', D, candidate, focus, samples),
                         ('R1_R2', A, ret, focus_a, a_samples),
                         ('D_R1_R2_same_candidates', D, matched_rule, focus, samples)]
                rows, zero_rows, query = [], [], {}
            else:
                rows, zero_rows, query = previous['rows'], previous['zero'], previous['query']
                audit = previous['candidate_oracle']
            for mode, net in pending.items():
                learned = learned_hidden(net, store, row, manifest, K, band)
                learned, _ = order_hidden(learned, depth, K, rule, bg_order)
                parts.append((mode, D, learned, focus, samples))
                query[mode] = evaluate_row(net, store, row, manifest,
                                           balance_activation=(mode == 'balanced_activation'))
            truth = display(read_exr(TRUTH / name / 'dof' / (cid + '.exr')))
            gt_regions = masks(TRUTH / name, dict(c, frame=f))
            regions = {key: gt_regions[key] for key in ['full_image', 'in_focus', 'band', 'recoverable']}
            regions['invalid_geometry'] = invalid
            for method, vis, hid, zf, fallback in parts:
                if method == 'A':
                    image, zero = a_image, a_zero.cpu().numpy()
                else:
                    cloud = merge_layers(vis, hid)
                    values = render_fallbacks(cloud, K, zf, radius, 64, fill, bg, fallback)
                    key = (f, method if method == 'D0' else (protocol, cid, method))
                    if key not in zero_known:
                        raster, alpha = zero_parts(cloud, K)
                        bg_zero = fill if method == 'R1_R2' else a_zero
                        zero_known[key] = (raster + (1 - alpha[..., None]) * bg_zero).cpu().numpy()
                    zero, image = zero_known[key], values['F1' if method == 'R1_R2' else 'F2'].cpu().numpy()
                    if method == 'R1_R2':
                        image = np.where(a_band[..., None], image, a_image)
                        zero = np.where(a_band[..., None], zero, a_zero.cpu().numpy())
                if not np.isfinite(image).all() or not np.isfinite(zero).all():
                    raise RuntimeError(f'Nonfinite evaluation pixels: {name} {protocol} {cid} {method}')
                pred, zero = display(image), display(zero)
                common = dict(scene=name, frame=f, id=cid, protocol=protocol, method=method,
                              focus=c['nominal_focus_m'], fstop=c['fstop'], used_focus_m=zf)
                rows.extend(dict(common, region=region, **value) for region, value in score(pred, truth, regions, metric).items())
                zero_rows.extend(dict(common, region=region, PSNR=zero_psnr(zero, d['rgb_srgb'], mask))
                                 for region, mask in regions.items() if region in ('full_image', 'in_focus', 'invalid_geometry'))
                path.mkdir(parents=True, exist_ok=True)
                save_npz(path / (method + '.npz'), dict(prediction_srgb=pred, zero_srgb=zero))
            write(result, dict(rows=rows, zero=zero_rows, query=query, candidate_oracle=audit,
                               seconds=(previous['seconds'] if previous else 0) + time.monotonic() - begin, invalid_pixels=int(invalid.sum())))
            print('R3_EVAL_RENDER', root.name, name, protocol, cid, flush=True)


def main(stage, variant):
    names = read(OUT / 'split.json')['heldout']
    root = ORIGINAL if variant == 'original' else DEFAULT
    if stage == 'prepare':
        with ThreadPoolExecutor(max_workers=3) as executor:
            list(executor.map(lambda name: prepare(name, root), names))
        return
    modes = [{'default':'main', 'balanced':'balanced_activation', 'original':'original_geometry'}[variant]]
    nets = {}
    for mode in modes:
        while not (OUT / 'R3' / mode / 'training.json').exists():
            if (OUT / 'R3' / mode / 'optimization-failure.json').exists():
                raise RuntimeError(f'{mode} training stopped under protocol')
            qa.guard()
            time.sleep(5)
    if variant == 'balanced':
        while not (OUT / 'R3/render-main-complete.json').exists():
            qa.guard()
            time.sleep(5)
    lock_name = 'gpu0-render-original.lock' if variant == 'original' else 'gpu0.lock'
    with (LOG / lock_name).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        qa.guard(True)
        if torch.cuda.mem_get_info()[0] < 20 * 2**30:
            raise RuntimeError('GPU memory budget for evaluation unavailable')
        torch.cuda.set_per_process_memory_fraction(.30 if variant == 'original' else .70)
        for mode in modes:
            net = LightCandidateFusion().cuda()
            checkpoint = OUT / 'R3' / mode / f'step-{C["light_fusion"]["S2a_steps"]:06d}.pt'
            net.load_state_dict(torch.load(checkpoint, weights_only=True, map_location='cpu')['model'])
            nets[mode] = net.eval()
        metric = lpips()
        with torch.inference_mode():
            for name in names:
                qa.guard()
                evaluate_scene(name, root, nets, metric)
    write(OUT / 'R3' / ('render-' + modes[0] + '-complete.json'), dict(variant=variant, scenes=names))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['prepare', 'render'])
    p.add_argument('variant', choices=['default', 'balanced', 'original'])
    args = p.parse_args()
    torch.set_num_threads(4)
    main(args.stage, args.variant)
