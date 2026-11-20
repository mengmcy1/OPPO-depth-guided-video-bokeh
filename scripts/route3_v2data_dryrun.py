"""Expanded 1a/1b minibatch objective and read-only training acceptance runner."""
import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager, nullcontext
from dataclasses import replace
import fcntl
import hashlib
import os
from pathlib import Path
import time

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')

import cv2
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
from torch.nn import functional as F

from route3_ovg_1b_common import CONFIG, SOURCE
from route3_ovg_1b_train import load_model
from route3_ovg_subpixel import CONFIG as ONE_A, save_json
from route3_v2data_a_build import appearance_losses
from route3_v2data_loader import FORMAL_DATA, MIX, ROOT, TrainingData, same_surface_loss, schedule, legacy
from route3_v2data_optics import hidden_substitute, linear_reference, observed_background, occupancy, ray_fields
from src.layered_aperture import merge_layers
from src.ovg_appearance import AppearanceHeads
from src.ovg_net import geometry_loss, masked_mean, scene_consistency
from src.ovg_optics import fixed_visible_surfels, occlusion_loss, raster, resized_prediction, stratified_pupil


REPORT = ROOT/'03_实验记录/records/route3-v2data-loader-dryrun-2026-11-21.json'
REVIEW = ROOT/'03_实验记录/records/route3-v2data-readiness-2026-11-20-review.json'
LOCK = Path('/root/autodl-tmp/route3-cache/gpu-data.lock')
START = SOURCE/'checkpoints/step-030000.pt'
SEED = CONFIG['seed']


def balanced_count(valid, boundary, cap):
    edge = int((valid & boundary).sum())
    inner = int((valid & ~boundary).sum())
    common = min(cap, edge, inner) if edge and inner else cap
    return min(edge, common)+min(inner, common)


def stage_objective(net, inventory, item, stage, generator, *, spec=None, audit_k16=False, audit_transmission=False):
    """Use native K16 for occlusion and an optional optical target for Bokeh."""
    tier = item[0]
    device = next(net.parameters()).device
    batch = inventory.load(item, device)
    pred = net(batch[0], protect_residual=True, render_heads=stage=='1a')
    x = batch[0]
    reference = None
    opts = CONFIG['optical_train']
    weights = (ONE_A['weights'] if stage=='1a' else CONFIG['loss_weights']).copy()
    counts = {}
    if tier == 'R':
        mv, n = scene_consistency(pred, batch[1])
        terms, counts = dict(mv=mv), dict(mv=int(n))
        common = batch[2]
    else:
        y = batch[1]
        sampling = CONFIG['boundary_sampling']
        _, terms, support = geometry_loss(
            pred, x, y, weights, boundary_sampling=sampling, generator=generator,
            alpha_split_weights=CONFIG['alpha_interior_boundary_weights'] if stage=='1b' else None)
        boundary = y['depth_boundary']
        cap = sampling[0]
        counts.update(disp=balanced_count(y['valid'],boundary,cap),
                      bimodal=balanced_count(y['valid_dual'] & y['valid_alpha'],boundary,cap),
                      alpha=(int(support['alpha']) if stage=='1b' else
                             balanced_count(y['valid_alpha'],boundary,cap)), mv=int(support['mv']))
        if 'normal' in terms:
            counts['normal'] = int(support['normal'])
        # G has an axial depth target and no alpha or bimodal target.
        if tier == 'G':
            for key in ('alpha','bimodal'):
                del terms[key], counts[key]
        common = torch.ones(x['rgb'].shape[-2:], device=device, dtype=torch.bool)
        if tier in ('B','C'):
            reference = inventory.pair_reference(item, y)
            value, pairs = same_surface_loss(pred, reference)
            terms['same_surface'] = value
            counts['same_surface'] = pairs['right']+pairs['down']
            counts.update({f'pair_{k}':v for k,v in pairs.items()})
    if stage == '1a':
        return terms, counts, weights
    shape = tuple(min(n,opts['max_image_side']) for n in x['rgb'].shape[-2:])
    pred, x, _ = resized_prediction(pred, x, None, shape)
    K = x['K'][0]
    visible = fixed_visible_surfels(pred,x)
    hidden = None
    labels = inventory.k16_specs(item)
    occupancy_spec = None
    if labels:
        if spec is None:
            choice = int(torch.randint(len(labels), (), generator=generator, device=device))
            occupancy_spec = labels[choice]
        else:
            occupancy_spec = next((row for row in labels if np.isclose(row['focus'], spec['focus'])), None)
            if occupancy_spec is None:
                raise ValueError("No K16 labels for the optical reference's focus")
    focus = occupancy_spec['focus'] if occupancy_spec else spec['focus'] if spec else 1.4
    center = K.new_zeros((1,2))
    if occupancy_spec:
        rays = ray_fields(occupancy_spec)
        offsets, cutoff, reveal, block = occupancy(rays,batch[1],shape,generator,opts['occ_ray_count'])
        terms['occ'] = occlusion_loss(visible,K,focus,offsets,cutoff,reveal,block)
        counts.update(occ_reveal=int(reveal.sum()),occ_block=int(block.sum()))
        counts['occ'] = counts['occ_reveal']+counts['occ_block']
        hidden, placed = hidden_substitute(occupancy_spec,rays,reference,pred['depth1'][0],K)
        counts.update(hidden_eligible=placed['eligible'],hidden_placed=placed['placed'])
        if audit_k16:
            audit_rng = torch.Generator(device=device).manual_seed(8)
            _, _, full_reveal, full_block = occupancy(rays,batch[1],shape,audit_rng,len(rays['depth']))
            counts.update(all_K16_occ_block=int(full_block.sum()),all_K16_occ_reveal=int(full_reveal.sum()))
    combined = merge_layers(visible,hidden)
    zero = raster(combined,K,focus,center,*shape,observed_background(spec,x,focus,center))[0]
    if common.shape != shape:
        common = F.interpolate(common[None,None].float(),size=shape,mode='area')[0,0] > .999
    terms['zero'] = masked_mean((zero-x['rgb'][0].permute(1,2,0)).abs().mean(-1),common)
    counts['zero'] = int(common.sum())
    if spec:
        radius = .05/(2*spec['fstop'])
        groups = [stratified_pupil(opts['rays_per_subset'],radius,generator,K) for _ in range(2)]
        target = torch.as_tensor(cv2.resize(linear_reference(spec),shape[::-1],interpolation=cv2.INTER_AREA),
                                 dtype=K.dtype,device=device)
        means = []
        for offsets in groups:
            background = observed_background(spec,x,focus,offsets)
            # Recompute one ray at a time in the backward pass to bound memory,
            # keeping the original two independent pupil subsets.
            def ray(means, scales, aperture, fill):
                scene = replace(combined,means=means,scales=scales)
                value = raster(scene,K,focus,aperture,*shape,fill,return_alpha=audit_transmission)
                return (value[0][0],value[1][0]) if audit_transmission else value[0]
            values = [checkpoint(ray,combined.means,combined.scales,offsets[j:j+1],
                                 None if background is None else background[j:j+1],use_reentrant=False)
                      for j in range(len(offsets))]
            if audit_transmission:
                images, alpha = zip(*values)
                a = torch.stack(alpha)
                transmission = (1-a).clamp(0,1)
                counts['uncovered_rays'] = counts.get('uncovered_rays',0)+int((a==0).sum())
                counts['rays_T_ge_0p1'] = counts.get('rays_T_ge_0p1',0)+int((transmission>=.1).sum())
                counts['residual_transmittance_sum'] = counts.get('residual_transmittance_sum',0.)+float(transmission.sum())
            else:
                images = values
            means.append(torch.stack(tuple(images)).mean(0))
        terms['bokeh'] = ((means[0]-target)*(means[1]-target)).mean()
        counts.update(bokeh=shape[0]*shape[1], bokeh_rays=shape[0]*shape[1]*sum(map(len,groups)))
    return terms, counts, weights


def a_objective(net, head, inventory, seed, device):
    item, labels, ids = inventory.a.sample(seed,stage='2a')
    rgb = torch.as_tensor(item['rgb'].transpose(2,0,1)[None],device=device)
    depth = rgb.new_zeros(1,3,*rgb.shape[-2:])
    x = dict(rgb=rgb,prior_depth=depth,prior_conf=depth,motion=depth[:,0].bool(),
             neighbors=rgb.new_zeros(1,0,3,*rgb.shape[-2:]))
    with torch.no_grad():
        geometry = net(x,protect_residual=True,render_heads=False,return_features=True)
    mem = torch.as_tensor(item['memory'].transpose(2,0,1)[None],device=device)
    conf = torch.as_tensor(item['memory_confidence'][None],device=device)
    pred = head(geometry,rgb,mem,conf)
    terms = appearance_losses(pred['foreground'][0].permute(1,2,0),labels)
    counts = dict(foreground=int(labels['color_valid'].sum()),comp=int(labels['comp_valid'].sum()))
    return terms,counts,dict(foreground=1.,comp=1.),ids


def statistics(terms, counts, weights, parameters, candidate_weight):
    terms = dict(terms)
    terms['total_base'] = sum(weights[k]*v for k,v in terms.items() if k!='same_surface')
    if 'same_surface' in terms:
        terms['total_candidate'] = terms['total_base']+candidate_weight*terms['same_surface']
    rows = {}
    for index,(key,value) in enumerate(terms.items()):
        gradients = torch.autograd.grad(value,parameters,retain_graph=index<len(terms)-1,allow_unused=True)
        norms = [torch.linalg.vector_norm(g.float()) for g in gradients if g is not None]
        norm = float(torch.linalg.vector_norm(torch.stack(norms))) if norms else 0.
        scalar = float(value.detach())
        finite = bool(np.isfinite(norm) and np.isfinite(scalar))
        rows[key] = dict(value=scalar if np.isfinite(scalar) else None,
                         grad_norm=norm if np.isfinite(norm) else None, finite=finite,
                         gradient_tensors=len(norms), support=counts.get(key))
    return rows


@contextmanager
def locked_gpu(net, head):
    with LOCK.open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':
            raise RuntimeError('Set CUDA_VISIBLE_DEVICES=0 for the assigned GPU')
        torch.cuda.set_per_process_memory_fraction(.69,0)
        torch.cuda.reset_peak_memory_stats()
        try:
            net.cuda()
            head.cuda()
            yield
            torch.cuda.synchronize()
        finally:
            net.cpu()
            head.cpu()
            torch.cuda.empty_cache()


def fingerprint(*models):
    digest = hashlib.sha256()
    for model in models:
        for name,v in model.state_dict().items():
            digest.update(name.encode())
            digest.update(v.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def summarize(batches):
    if not batches:
        return dict(batches=0,optical_batches=0,optical_fraction=None,losses={},supervision={})
    reports = defaultdict(list)
    for row in batches:
        for key,value in row['losses'].items():
            reports[key].append(value)
    def mean(vals):
        values = [v for v in vals if v is not None]
        return float(np.mean(values)) if values else None
    by_loss = {}
    for key, rows in reports.items():
        support = [x['support'] for x in rows if x['support'] is not None]
        norms = [r['grad_norm'] for r in rows if r['grad_norm'] is not None]
        by_loss[key] = dict(batches=len(rows), mean=mean([v['value'] for v in rows]),
                            grad_norm_mean=mean(norms),grad_norm_max=max(norms,default=None),
                            all_finite=all(v['finite'] for v in rows),
                            gradient_tensors_min=min(r['gradient_tensors'] for r in rows),
                            supervision=sum(support) if support else None,
                            zero_supervision_batches=support.count(0) if support else None)
    supervision = Counter()
    for row in batches:
        supervision.update(row['support'])
    optical = sum(row['optical'] is not None for row in batches)
    return dict(batches=len(batches),optical_batches=optical,optical_fraction=optical/len(batches),
                losses=by_loss, supervision=dict(supervision))


def planned(inventory):
    items = []
    for phase,stage in enumerate(MIX):
        sequence = schedule(stage,np.random.default_rng([SEED,phase]))
        for i,tier in enumerate(sequence):
            seed = SEED+10_000*phase+i
            rng = np.random.default_rng(seed)
            item = inventory.choose(tier,rng)
            specs = inventory.optical_specs(item) if stage=='1b' and item else []
            spec = specs[int(rng.integers(len(specs)))] if specs else None
            items.append(dict(stage=stage,index=i,tier=tier,seed=seed,item=item,spec=spec))
    for i in range(20):
        items.append(dict(stage='A-appearance',index=i,tier='A',seed=SEED+20000+i,item=True,spec=None))
    return items


def make_report(inventory, plan, rows, unchanged, peak, candidate_weight):
    phases = {}
    failures = []
    for stage in (*MIX,'A-appearance'):
        expected = Counter(p['tier'] for p in plan if p['stage']==stage)
        selected = [b for b in rows if b['stage']==stage]
        by_tier = {tier:summarize([b for b in selected if b['tier']==tier]) for tier in expected}
        overall = summarize(selected)
        phases[stage] = dict(planned=dict(expected),completed={k:v['batches'] for k,v in by_tier.items()},
                             optical_batches=overall['optical_batches'],optical_fraction=overall['optical_fraction'],
                             optical_fraction_of_planned=overall['optical_batches']/sum(expected.values()),
                             by_tier=by_tier)
        for tier, block in by_tier.items():
            if block['batches']!=expected[tier]:
                failures.append(f'{stage}/{tier}: completed {block["batches"]} of {expected[tier]}')
            if stage=='A-appearance':
                required = {'foreground','comp'}
            elif tier == 'R':
                required = {'mv'}
            elif tier == 'G':
                required = {'disp','mv'}
            else:
                required = {'disp','mv','bimodal','alpha','same_surface'}
            if stage=='1a':
                required.add('normal')
            elif stage=='1b':
                required.add('zero')
            if stage=='1b' and tier in ('B','C'):
                required.update(('occ','bokeh'))
            if block['batches'] and required-block['losses'].keys():
                failures.append(f'{stage}/{tier}: missing terms {sorted(required-block["losses"].keys())}')
            for key, loss in block['losses'].items():
                if not loss['all_finite'] or loss['gradient_tensors_min']==0:
                    failures.append(f'{stage}/{tier}/{key}: nonfinite or disconnected gradient')
                if loss['supervision']==0:
                    failures.append(f'{stage}/{tier}/{key}: no effective supervision')
            if stage=='1b' and tier in ('B','C') and not block['optical_batches']:
                failures.append(f'{stage}/{tier}: no optical batch')
    for row in rows:
        if row['stage']=='1b' and row['tier']=='B' and not row['support'].get('occ'):
            failures.append(f'1b/B/{row["index"]}: no effective K16 occlusion rays')
        if row['optical'] and (not {'occ','bokeh'}<=row['losses'].keys() or row['support']['occ']==0
                              or row['support']['bokeh']==0):
            failures.append(f'{row["stage"]}/{row["tier"]}/{row["index"]}: missing effective optical term')
    if not unchanged:
        failures.append('model state changed')
    b = legacy.read(FORMAL_DATA/'manifest.json')
    g = legacy.read(ROOT/'data/processed/route3-v2data-2026-11-20/G/manifest.json')
    g_done = sum(r['frames'] for r in g['counts'].values())
    return dict(passed=not failures, failures=failures, input_split='train', optimizer_steps=0,
                model_state_unchanged=unchanged, checkpoint=str(START.relative_to(ROOT)),
                seed=SEED, planned_mixes_percent=MIX, candidate_weight_for_dryrun_only=candidate_weight,
                pool_sizes={k:len(v) for k,v in inventory.items.items()},b_excluded=inventory.b_excluded,
                b_production_counts={k:b['counts'][k] for k in ('train_geometry','train_optical','development')},
                g_production_frames=g_done,g_target_frames=g['target'],
                gpu_lock=str(LOCK),gpu_peak_reserved_bytes=peak,
                batch_devices=dict(Counter(r.get('device','cuda') for r in rows)),phases=phases)


def run(args):
    if args.device=='cpu':
        if '1b' in args.phase:
            raise ValueError('Stage 1b uses the CUDA aperture renderer')
        os.environ['XFORMERS_DISABLED']='1'
    torch.set_num_threads(5 if args.device=='cpu' else 3)
    cv2.setNumThreads(1)
    torch.manual_seed(SEED)
    inventory = TrainingData()
    plan = planned(inventory)
    saved = legacy.read(args.state) if args.state.is_file() else dict(rows=[],peak=0)
    rows, peak = saved['rows'], saved['peak']
    filled = {(r['stage'],r['index']) for r in rows}
    outstanding = [p for p in plan if p['item'] and p['stage'] in args.phase
                   and (p['stage'],p['index']) not in filled]
    if args.limit:
        outstanding = outstanding[:args.limit]
    net = load_model(START,'model',device='cpu').train()
    head = AppearanceHeads(net.coverage).train()
    baseline = fingerprint(net,head)
    start = 0
    while start < len(outstanding):
        subset = outstanding[start:start+args.chunk]
        with locked_gpu(net,head) if args.device=='cuda' else nullcontext():
            deadline = time.monotonic()+15
            for p in subset:
                stage, item = p['stage'], p['item']
                gen = torch.Generator(device=args.device).manual_seed(p['seed'])
                if stage=='A-appearance':
                    terms,counts,weights,ids = a_objective(net,head,inventory,p['seed'],args.device)
                    parameters = tuple(v for v in head.parameters() if v.requires_grad)
                else:
                    terms,counts,weights = stage_objective(net,inventory,item,stage,gen,spec=p['spec'])
                    parameters = tuple(v for v in net.parameters() if v.requires_grad)
                    ids = [item[0],item[1] if item[0]=='B' else Path(item[1]).parent.name,item[2]]
                data = statistics(terms,counts,weights,parameters,args.candidate_weight)
                del terms
                optical = {k:p['spec'][k] for k in ('frame','focus','fstop')} if p['spec'] else None
                rows.append(dict(stage=stage,tier=p['tier'],index=p['index'],item=ids,optical=optical,device=args.device,
                                 losses=data,support=counts))
                start += 1
                if args.device=='cuda' and time.monotonic() >= deadline:
                    break
            if args.device=='cuda':
                peak = max(peak,torch.cuda.max_memory_reserved())
        save_json(args.state,dict(rows=rows,peak=peak))
        print('V2_DRYRUN',len(rows),outstanding[start-1]['stage'],flush=True)
    report = make_report(inventory,plan,rows,fingerprint(net,head)==baseline,peak,args.candidate_weight)
    if args.review and args.review.is_file():
        report['audit_reference'] = dict(file=args.review.name,sha256=hashlib.sha256(args.review.read_bytes()).hexdigest())
    if args.regression and args.regression.is_file():
        report['legacy_regression'] = legacy.read(args.regression)
        if not report['legacy_regression']['passed']:
            report['passed'] = False
            report['failures'].append('legacy numeric regression')
    else:
        report['passed'] = False
        report['failures'].append('legacy numeric regression pending')
    save_json(args.report,report)
    print('V2_DRYRUN_RESULT',report['passed'],report['failures'],flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state',type=Path,required=True)
    parser.add_argument('--report',type=Path,default=REPORT)
    parser.add_argument('--regression',type=Path)
    parser.add_argument('--review',type=Path,default=REVIEW)
    parser.add_argument('--phase',nargs='+',choices=(*MIX,'A-appearance'),default=[*MIX,'A-appearance'])
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--chunk',type=int,default=4)
    parser.add_argument('--limit',type=int)
    parser.add_argument('--candidate-weight',type=float,default=1.)
    run(parser.parse_args())
