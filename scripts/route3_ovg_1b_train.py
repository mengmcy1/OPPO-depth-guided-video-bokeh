"""Continue 1a at the fixed 1b mixture with EMA and paired validation."""
import argparse
from collections import Counter, defaultdict
import os
import subprocess
import time

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')

import cv2
import numpy as np
import torch

from route3_ovg_1b_common import CONFIG, OUT, SOURCE, guard, read
from route3_ovg_1b_data import setup
from route3_ovg_1b_dl3dv import inventory as real_inventory, training_input
from route3_ovg_1b_optical_data import (background_samples, hidden_substitute, occupancy_labels,
                                      read_optical, synthetic_optical)
import route3_ovg_data as data
import route3_ovg_train as base
from route3_ovg_r2_train import groups_G, model
from route3_ovg_subpixel import save_json
from src.layered_aperture import merge_layers
from src.ovg_net import geometry_loss, masked_mean, scene_consistency
from src.ovg_optics import occlusion_loss, raster, resized_prediction, stratified_pupil, visible_surfels


def load_model(path=SOURCE/'checkpoints/step-030000.pt', key='model'):
    net = model()
    checkpoint = torch.load(path, weights_only=True, map_location='cpu')
    missing, extra = net.load_state_dict(checkpoint[key], strict=False)
    if extra or any(not p.startswith('encoder.') for p in missing):
        raise ValueError(f'Invalid stage checkpoint: {missing}, {extra}')
    return net


def state(net):
    return {k: v.detach().cpu() for k, v in net.state_dict().items() if not k.startswith('encoder.')}


def shadow(net):
    return {k: p.detach().clone() for k, p in net.named_parameters() if p.requires_grad}


@torch.no_grad()
def ema_update(net, ema):
    for name, p in net.named_parameters():
        if name in ema:
            ema[name].lerp_(p, 1. - CONFIG['ema_decay'])


def evaluate(net, step, store_predictions=False):
    """Compare fusion, refinement and best hard prior on identical pixels."""
    net.eval()
    totals = defaultdict(Counter)
    rows = []
    with torch.inference_mode():
        for item in data.inventory('dev')['B']:
            x, y = data.sample(item, 'cuda')
            pred = net(x)
            available = x['prior_conf'] > 0
            gt_inverse = y['depth1'].clamp_min(.05).reciprocal()
            errors = (x['prior_depth'].clamp_min(.05).reciprocal() - gt_inverse[:, None]).abs()
            best = torch.where(available, errors, torch.inf).argmin(1)
            oracle = x['prior_depth'].gather(1, best[:, None])[:, 0]
            edge = y['depth_boundary'] & y['valid']
            interior = y['valid_alpha'] & (y['alpha'] >= .9999) & ~edge
            for subset, mask in {
                'any_prior': available.any(1),
                'at_least_two': available.sum(1) >= 2,
                'high_conf_DepthSplat_boundary': (x['prior_conf'][:, 1] >= .6) & edge,
            }.items():
                for label, depth, alpha, back in (
                    ('selected-prior-base', pred['fused_depth'], None, None),
                    ('OVG-Net', pred['depth1'], pred['alpha'], pred['depth2']),
                    ('best-single-prior-Oracle', oracle, None, None)):
                    for focus in (1.4, 5.):
                        factor = x['K'][0, 0, 0] * .05 / (2 * 1.4) * focus / (focus-.05)
                        values = base.measure(depth, alpha, back, y, mask, factor, edge, interior)
                        for name in (item[1], 'pooled'):
                            totals[name, subset, label, focus].update(values)
            if store_predictions:
                from route3_ovg_subpixel import save_npz
                path = OUT/'dev-predictions'/item[1]/f'f{data.TRAIN_FRAMES[item[2]]:03d}.npz'
                save_npz(path, **{k: v[0].cpu().numpy() for k, v in pred.items()
                                  if k in ('depth1', 'depth2', 'alpha', 'sigma_px', 'opacity', 'normal')})
    for (name, subset, label, focus), values in sorted(totals.items()):
        rows.append(dict(scene=name, subset=subset, method=label, focus_m=focus, **base.reduce_metrics(values)))
    return dict(step=step, rows=rows, high_conf_threshold=.6,
                alpha_back_for_single_depth='undefined; null', data_split='39 development frames')


def training_objective(net, item, rng, generator):
    tier, key, index = item
    opts = CONFIG['optical_train']
    weights = CONFIG['loss_weights']
    if tier == 'R':
        x, heldout, common = training_input(key, index, 'cuda')
        truth_geometry = None
    else:
        x, truth_geometry = data.sample(item, 'cuda')
        common = torch.ones(x['rgb'].shape[-2:], device='cuda', dtype=torch.bool)
    pred = net(x)
    if tier == 'R':
        mv, n = scene_consistency(pred, heldout)
        losses = dict(mv=mv)
        counts = dict(mv=int(n))
        specs = []
    else:
        _, losses, counts = geometry_loss(pred, x, truth_geometry, weights,
             boundary_sampling=CONFIG['boundary_sampling'], generator=generator,
             alpha_split_weights=CONFIG['alpha_interior_boundary_weights'])
        specs = synthetic_optical(item)
    spec = specs[int(rng.integers(len(specs)))] if specs else None
    h0, w0 = x['rgb'].shape[-2:]
    h, w = min(h0, opts['max_image_side']), min(w0, opts['max_image_side'])
    pred, x, _ = resized_prediction(pred, x, truth_geometry, (h, w))
    K = x['K'][0]
    vis = visible_surfels(pred, x)
    counts = {k: int(v) for k, v in counts.items()}
    focus = spec['focus'] if spec else 1.4
    center = K.new_zeros((1, 2))
    reference = read_optical(spec) if spec else None
    hidden, stats = (None, {})
    if spec:
        offsets, cutoff, reveal, block = occupancy_labels(spec, h, w, truth_geometry, generator)
        losses['occ'] = occlusion_loss(vis, K, focus, offsets, cutoff, reveal, block)
        counts.update(occ_reveal=int(reveal.sum()), occ_block=int(block.sum()))
        hidden, stats = hidden_substitute(spec, pred['depth1'][0], K)
    combined = merge_layers(vis, hidden)
    zero = raster(combined, K, focus, center, h, w, background_samples(spec, K, focus, center, (h, w)))[0]
    if common.shape != (h, w):
        common = torch.nn.functional.interpolate(common[None, None].float(), size=(h, w), mode='area')[0, 0] > .999
    losses['zero'] = masked_mean((zero-x['rgb'][0].permute(1,2,0)).abs().mean(-1), common)
    counts['zero_pixels'] = int(common.sum())
    if reference is not None:
        radius = .05 / (2 * spec['fstop'])
        groups = [stratified_pupil(opts['rays_per_subset'], radius, generator, K) for _ in range(2)]
        offsets = torch.cat(groups)
        backgrounds = background_samples(spec, K, focus, offsets, (h, w))
        linear = raster(combined, K, focus, offsets, h, w, backgrounds)
        first, second = linear.chunk(2, dim=0)
        target = torch.tensor(cv2.resize(reference, (w, h), interpolation=cv2.INTER_AREA),
                              device=K.device, dtype=K.dtype)
        losses['bokeh'] = ((first.mean(0)-target)*(second.mean(0)-target)).mean()
        counts['bokeh_pixels'] = h*w
    loss = sum(weights[k] * v for k, v in losses.items())
    if not torch.isfinite(loss):
        raise FloatingPointError(f'1b nonfinite loss at {item}')
    return loss, losses, counts, stats


def check():
    """P1 optical forward and differentiated batch without a training update."""
    net = load_model()
    net.train()
    pool = data.inventory('train')
    b = next(i for i in pool['B'] if synthetic_optical(i))
    c = next(i for i in pool['C'] if any(s['truth'].exists() for s in synthetic_optical(i)))
    rng = np.random.default_rng(CONFIG['seed'])
    generator = torch.Generator(device='cuda').manual_seed(CONFIG['seed'])
    rows = []
    for item in (b,c):
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        loss, losses, counts, stats = training_objective(net, item, rng, generator)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
        if not {'occ', 'bokeh', 'zero'} <= losses.keys() or not stats.get('placed'):
            raise ValueError(f'Missing optical reference or render-only hidden aid for {item}')
        rows.append(dict(tier=item[0], sample=list(item), total=float(loss.detach()),
                         losses={k: float(v.detach()) for k,v in losses.items()}, support=counts,
                         hidden_placement=stats, optimizer_updates=0, grad_norm=float(norm),
                         seconds=time.monotonic()-start, peak_MiB=torch.cuda.max_memory_allocated()/2**20))
        print('OPTICAL_BATCH', rows[-1], flush=True)
        net.zero_grad(set_to_none=True)
    save_json(OUT/'P1/optical-batch-verified.json', dict(passed=True, rows=rows))


def train():
    probabilities = CONFIG['batch_probability']
    for path in ('P1/prefix.json', 'P1/card-audit/result.json', 'P1/optical-batch-verified.json'):
        if not read(OUT/path)['passed']:
            raise RuntimeError(f'P1 failed: {path}')
    pool = data.inventory('train')
    pool['R'] = list(real_inventory())
    if not all(pool[t] for t in probabilities):
        raise RuntimeError(f'1b required tier missing: {[t for t in probabilities if not pool[t]]}')
    torch.manual_seed(CONFIG['seed'])
    net = load_model()
    opt = torch.optim.AdamW((p for p in net.parameters() if p.requires_grad), lr=CONFIG['learning_rate'])
    rng = np.random.default_rng(CONFIG['seed'])
    g = torch.Generator(device='cuda').manual_seed(CONFIG['seed'])
    ema = shadow(net)
    counts, active, curve, joins = Counter(), Counter(), [], []
    start_at = 1
    checkpoints = OUT/'checkpoints'
    checkpoints.mkdir(exist_ok=True)
    existing = sorted(checkpoints.glob('step-*.pt'))
    if existing:
        ck = torch.load(existing[-1], weights_only=True, map_location='cpu')
        net.load_state_dict(ck['model'], strict=False)
        opt.load_state_dict(ck['optimizer'])
        for k, p in ema.items():
            p.copy_(ck['ema'][k])
        start_at = ck['step']+1
        rng.bit_generator.state = ck['sampling_rng']
        g.set_state(ck['pixel_rng'])
        history = read(OUT/'validation'/f'step-{ck["step"]:06d}-state.json')
        counts.update(history['counts'])
        active.update(history['active'])
        curve, joins = history['curve'], history['joins']
    start = time.monotonic()
    for i in range(start_at, CONFIG['steps']+1):
        if i == start_at or i % 100 == 1:
            guard()
            pool = data.inventory('train')
            pool['R'] = list(real_inventory())
            groups = groups_G(pool)
            sizes = {k: len(v) for k,v in pool.items()}
            if not joins or sizes != joins[-1]['counts']:
                joins.append(dict(step=i, counts=sizes))
        tier = str(rng.choice(list(probabilities), p=list(probabilities.values())))
        items = groups[int(rng.integers(len(groups)))] if tier == 'G' else pool[tier]
        item = items[int(rng.integers(len(items)))]
        net.train()
        opt.zero_grad(set_to_none=True)
        loss, detail, stats, hidden = training_objective(net, item, rng, g)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
        opt.step()
        ema_update(net, ema)
        counts[tier] += 1
        active.update(occ=int(stats.get('occ_reveal', 0) + stats.get('occ_block', 0) > 0),
                      bokeh=int(stats.get('bokeh_pixels', 0) > 0),
                      zero=int(stats.get('zero_pixels', 0) > 0))
        curve.append(dict(step=i, tier=tier, loss=float(loss.detach()), grad_norm=float(norm),
                          terms={k: float(v.detach()) for k,v in detail.items()}, pixels=stats,
                          hidden=hidden))
        if i == 1 or i % 100 == 0:
            report = dict(step=i, counts=counts, active=active, joins=joins, latest=curve[-1],
                          seconds=time.monotonic()-start)
            save_json(OUT/'progress.json', report)
            print('OVG_1B', i, dict(counts), dict(active), flush=True)
        if i in CONFIG['eval_steps']:
            raw = state(net)
            terminal = i == CONFIG['steps']
            val_raw = evaluate(net, i)
            net.load_state_dict(ema, strict=False)
            val_ema = evaluate(net, i, terminal)
            net.load_state_dict(raw, strict=False)
            save_json(OUT/'validation'/f'step-{i:06d}.json', dict(raw=val_raw, EMA=val_ema))
            save_json(OUT/'validation'/f'step-{i:06d}-state.json',
                      dict(step=i, curve=curve, joins=joins, counts=counts, active=active))
            path = checkpoints/f'step-{i:06d}.pt'
            tmp = path.with_suffix('.part')
            state_ema = dict(raw, **{k: p.cpu() for k,p in ema.items()})
            torch.save(dict(step=i, model=raw, ema=state_ema, optimizer=opt.state_dict(),
                            sampling_rng=rng.bit_generator.state, pixel_rng=g.get_state(), config=CONFIG), tmp)
            tmp.replace(path)
    fractions = lambda collection: {k: v / CONFIG['steps'] for k,v in collection.items()}
    save_json(OUT/'training.json', dict(steps=CONFIG['steps'], counts=counts, active=active, joins=joins,
              sampled_fractions=fractions(counts), active_fractions=fractions(active),
              curve=curve, seconds=time.monotonic()-start, peak_MiB=torch.cuda.max_memory_allocated()/2**20,
              initialization='1a round 2 fixed 30k model parameters; reset AdamW for the changed objective',
              GT_warmup=False, terminal_only=True))
    plot_training(curve)


def plot_training(curve):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11,4))
    for tier in CONFIG['batch_probability']:
        series = [(row['step'], row['loss']) for row in curve if row['tier'] == tier]
        steps, loss = np.array(series).T
        window = min(100, len(steps))
        kernel = np.full(window, 1/window)
        axes[0].plot(steps[window-1:], np.convolve(loss, kernel, mode='valid'), label=tier)
    for weights in ('EMA', 'raw'):
        points = []
        for step in CONFIG['eval_steps']:
            table = read(OUT/'validation'/f'step-{step:06d}.json')[weights]['rows']
            r = next(r for r in table if r['scene'] == 'pooled' and r['focus_m'] == 1.4 and
                     r['subset'] == 'any_prior' and r['method'] == 'OVG-Net')
            points.append((step, r['boundary_front_CoC_mae_px']))
        axes[1].plot(*zip(*points), marker='o', label=weights)
    axes[0].set(xlabel='Optimizer step', ylabel='Training loss, 100 samples per tier')
    axes[1].set(xlabel='Optimizer step', ylabel='Dev front boundary CoC MAE, px')
    for ax in axes:
        ax.legend()
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(OUT/'training-curves.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('job', choices=['check', 'train'])
    cfg = parser.parse_args()
    guard()
    setup()
    subprocess.run(['nvidia-smi'], check=True)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0' or torch.cuda.mem_get_info()[0] < 6*2**30:
        raise RuntimeError('Concurrent OVG memory budget unavailable')
    torch.set_num_threads(3)
    torch.set_float32_matmul_precision('high')
    torch.cuda.set_per_process_memory_fraction(.57)
    {'check': check, 'train': train}[cfg.job]()
