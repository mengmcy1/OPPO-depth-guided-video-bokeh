"""Replay the 70 planned B batches with matched aperture draws and step timing."""
import argparse
from collections import Counter
import gc
import json
import os
from pathlib import Path
import time
from unittest.mock import patch

import cv2
import numpy as np
import torch

from route3_v2data_B_batch import sha256
from route3_v2data_B_dryrun_p2 import QualificationData, geometry_plan
from route3_v2data_B_front import write_json
from route3_v2data_dryrun import CONFIG, SEED, START, fingerprint, load_model, locked_gpu, stage_objective
from route3_v2data_optics import occupancy, ray_fields
from route3_ovg_subpixel import TRAIN_FRAMES
from src.aperture_renderer import GaussianScene
from src.ovg_appearance import AppearanceHeads
from src.prefix_transmittance import differentiable_prefix_transmittance


LABEL_COUNT = CONFIG['optical_train']['occ_ray_count']
MAX_STEP_RATIO = 1.30
MAX_RESERVED_BYTES = 28 * 2**30


class PreparedBatch:
    def __init__(self, inventory, item, focus=None):
        batch = inventory.load(item, 'cpu')
        self.reference = inventory.pair_reference(item, batch[1])
        self.batch = tuple({k:v.cuda() for k,v in fields.items()} for fields in batch)
        self.specs = inventory.k16_specs(item)
        if focus is not None:
            self.specs = [self.specs[focus]]
        self.rays = {spec['label']:ray_fields(spec) for spec in self.specs}
        self.counts = {label:self.label_counts(rays,batch[1]) for label,rays in self.rays.items()}
        self.last_focus = None
        self.last_draw = None

    def load(self, item, device):
        return self.batch

    def pair_reference(self, item, target):
        return self.reference

    def k16_specs(self, item):
        return self.specs

    def get_rays(self, spec):
        self.last_focus = spec
        return self.rays[spec['label']]

    def sample(self, rays, target, size, generator, ray_count):
        clone = generator.clone_state()
        self.last_draw = torch.randperm(len(rays['depth']), generator=clone, device=generator.device
                                      )[:ray_count].cpu().tolist()
        return occupancy(rays, target, size, generator, ray_count)

    @staticmethod
    def label_counts(rays, target):
        pure = (target['valid_alpha'][0] & ~target['depth_boundary'][0] & (target['alpha'][0] >= .9999)).numpy()
        depth = rays['depth']
        good = rays['known'] & pure & np.isfinite(depth) & (depth > 0)
        full = [good & rays['hidden'], good & ~rays['hidden'] & rays['foreground']]
        h,w = depth.shape[-2:]
        return dict(
            native=np.stack([v.sum(axis=(1,2)) for v in full],-1),
            pooled=np.stack([v.reshape(len(v),h//2,2,w//2,2).all(axis=(2,4)).sum(axis=(1,2)) for v in full],-1),
        )


def step(net, item, cached, seed, mode, *, spec=None):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    gen = torch.Generator(device='cuda').manual_seed(seed)
    params = tuple(x for x in net.parameters() if x.requires_grad)
    with patch('route3_v2data_dryrun.ray_fields', cached.get_rays), patch('route3_v2data_dryrun.occupancy', cached.sample):
        start = time.monotonic()
        terms, counts, weights = stage_objective(net,cached,item,'1b',gen,spec=spec,b_occ_mode=mode)
        loss = sum(weights.get(k,1.)*v for k,v in terms.items())
        torch.cuda.synchronize()
        forward = time.monotonic()-start
        grad = torch.autograd.grad(loss,params,allow_unused=True)
        torch.cuda.synchronize()
        elapsed = time.monotonic()-start
    peak = torch.cuda.max_memory_reserved()
    support = (counts['occ_reveal'], counts['occ_block'])
    if sum(support) != counts['occ']:
        raise AssertionError('loss support disagrees with branch counts')
    label = cached.last_focus
    sizes = cached.counts[label['label']]
    raw = sizes['pooled' if mode=='pooled' else 'native']
    expected = raw[cached.last_draw].sum(axis=0).tolist()
    if list(support)!=expected:
        raise AssertionError(f'label and loss disagree on effective rays: {support} vs {expected}')
    finite = all(bool(torch.isfinite(value)) for value in terms.values()) and all(
        bool(torch.isfinite(value).all()) for value in grad if value is not None
    )
    by_focus = {
        str(n):dict((key,{'reveal':int(v.sum(0)[0]),'block':int(v.sum(0)[1])}) for key,v in sizes.items())
        for n,sizes in enumerate(cached.counts.values())
    }
    return dict(
        mode=mode, focus_m=label['focus'], pupil_ids=cached.last_draw, reveal=support[0], block=support[1],
        forward_seconds=forward, backward_seconds=elapsed-forward, step_seconds=elapsed,
        peak_reserved_bytes=peak, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        finite=finite, gradient_tensors=sum(x is not None for x in grad),
        loss_values={k:float(v.detach()) for k,v in terms.items()},
        all_pupils=dict(reveal=int(raw.sum(0)[0]),block=int(raw.sum(0)[1])),
        expected_under_uniform_sampling=dict(reveal=float(raw.sum(0)[0]*LABEL_COUNT/len(raw)),
                                             block=float(raw.sum(0)[1]*LABEL_COUNT/len(raw))),
        original_labels_by_focus=by_focus,
    )


def measure(net, inventory, cases, modes, *, spec=None):
    pairs = []
    for index,(item,seed,selected_focus) in enumerate(cases):
        cached = PreparedBatch(inventory,item,focus=selected_focus)
        outputs = {}
        order = modes if index%2==0 else reversed(modes)
        for mode in order:
            try:
                outputs[mode] = step(net,item,cached,seed,mode,spec=spec)
            except torch.OutOfMemoryError as error:
                outputs[mode] = dict(mode=mode,oom=True,error=str(error))
                gc.collect()
                torch.cuda.empty_cache()
        if not any(row.get('oom') for row in outputs.values()):
            choices = {(row['focus_m'],tuple(row['pupil_ids'])) for row in outputs.values()}
            if len(choices)!=1:
                raise AssertionError('paired modes must use exactly the same focus and aperture offsets')
            losses = [dict((k,v) for k,v in row['loss_values'].items() if k!='occ') for row in outputs.values()]
            if not all(all(np.isclose(base,row[k],rtol=1e-6,atol=1e-8) for k,base in losses[0].items())
                       for row in losses[1:]):
                raise AssertionError('a non-occlusion loss changed under the paired replay')
        pairs.append(dict(scene_id=item[1],frame=TRAIN_FRAMES[item[2]],seed=seed,focus_index=selected_focus,
                          outcomes=outputs))
    return pairs


def timing(pairs, mode):
    rows = [row['outcomes'][mode] for row in pairs]
    if any(row.get('oom') for row in rows):
        return dict(oom=True)
    return dict(
        step_median_seconds=float(np.median([row['step_seconds'] for row in rows])),
        step_p95_seconds=float(np.quantile([row['step_seconds'] for row in rows],.95)),
        forward_median_seconds=float(np.median([row['forward_seconds'] for row in rows])),
        backward_median_seconds=float(np.median([row['backward_seconds'] for row in rows])),
        peak_reserved_bytes=max(row['peak_reserved_bytes'] for row in rows),
        peak_allocated_bytes=max(row['peak_allocated_bytes'] for row in rows),
        all_finite=all(row['finite'] and row['gradient_tensors'] for row in rows),
    )


def passes_performance(candidate, baseline):
    return (not candidate.get('oom') and candidate['all_finite'] and
            candidate['step_median_seconds'] <= baseline['step_median_seconds'] * MAX_STEP_RATIO and
            candidate['peak_reserved_bytes'] <= MAX_RESERVED_BYTES)


def full_summary(pairs, mode):
    rows = [row['outcomes'][mode] for row in pairs]
    return dict(timing=timing(pairs,mode), branch_count=Counter(
        'both' if row['block'] and row['reveal'] else 'reveal_only' if row['reveal']
        else 'block_only' if row['block'] else 'neither' for row in rows),
        reveal_nonzero_batches=sum(row['reveal']>0 for row in rows),
        block_nonzero_batches=sum(row['block']>0 for row in rows),
        reveal_possible_both_focuses_and_all_pupils=sum(
            any(value['pooled' if mode=='pooled' else 'native']['reveal']>0
                for value in row['original_labels_by_focus'].values()) for row in rows),
    )


def zero_reasons(pairs, mode):
    rows = []
    for case in pairs:
        row = case['outcomes'][mode]
        for branch in ('reveal','block'):
            if row[branch]==0:
                rows.append(dict(scene_id=case['scene_id'],frame=case['frame'],branch=branch,
                                 focus_m=row['focus_m'],pupil_ids=row['pupil_ids'],
                                 source_rays=row['all_pupils'][branch],
                                 reason=('sampled four offsets contain no eligible rays' if row['all_pupils'][branch]
                                         else 'selected focus has no eligible native rays')))
    return rows


def verify_prefix():
    device = torch.device('cuda')
    make = lambda values:torch.tensor(values,device=device,dtype=torch.float32)
    means = make([[0.,0.,1.],[-.013,.047,1.2],[.03,-.052,1.6]]).requires_grad_()
    scales = make([[.032,.017,.001],[.05,.027,.002],[.022,.032,.001]]).requires_grad_()
    rotations = make([[1.,0.,0.,0.]]*3)
    opacity = make([.66,.79,.81]).requires_grad_()
    scene = GaussianScene(means,rotations,scales,opacity,make([[0.,0.,0.]]*3),
                          torch.zeros(3,device=device,dtype=torch.long))
    K = make([[65,0,16],[0,67,16],[0,0,1]])
    view = torch.eye(4,device=device)
    view[0,3] = -.003
    limit = torch.full((64,64),1.5,device=device)
    mask = torch.zeros_like(limit,dtype=torch.bool)
    mask[26:39:2,23:42] = True
    mask[35:39,25:39:2] = True
    fine_K = K.clone()
    fine_K[:2] *= 2
    dense = differentiable_prefix_transmittance(scene,fine_K,view,limit,eps2d=.0015*4)
    cull = differentiable_prefix_transmittance(scene,fine_K,view,limit,eps2d=.0015*4,active=mask)
    subpixels = differentiable_prefix_transmittance(scene,K,view,limit,active=mask,sample_scale=2)
    params=(means,scales,opacity)
    gradient = [torch.autograd.grad(v[mask].mean(),params) for v in (dense,cull,subpixels)]
    torch.testing.assert_close(cull[mask],dense[mask],atol=2e-6,rtol=2e-5)
    torch.testing.assert_close(subpixels[mask],dense[mask],atol=2e-6,rtol=2e-5)
    for result in gradient[1:]:
        for x,y in zip(result,gradient[0]):
            torch.testing.assert_close(x,y,atol=2e-5,rtol=2e-4)
    return dict(active_rays=int(mask.sum()),max_forward_difference=float((subpixels[mask]-dense[mask]).abs().max().detach()),
                max_gradient_difference=max(float((a-b).abs().max()) for a,b in zip(gradient[2],gradient[0])))


def run(root, audit, output, optical_probe=None, f2_root=None):
    frozen = json.loads(audit.read_text())['geometry_plan_1b']
    inventory = QualificationData(*(p.parent for p in root.glob('training-*/manifest.json')))
    # The first two sealed IDs per category reproduce the formal watcher plan.
    inventory.b = dict(inventory.qualification)
    plan = [(item,SEED+i,None) for i,(stage,item,_) in enumerate(geometry_plan(inventory)) if stage=='1b']
    if [(item[1],TRAIN_FRAMES[item[2]]) for item,_,_ in plan]!=[(row['sid'],row['frame']) for row in frozen]:
        raise ValueError('Qualification replay does not match the audited official 70-batch plan')
    output.mkdir(parents=True)
    torch.manual_seed(SEED)
    net = load_model(START,'model',device='cpu').train()
    head = AppearanceHeads(net.coverage).train()
    before = fingerprint(net,head)
    probe = next(item for item,_,_ in plan if item[1].startswith('training-060-') and TRAIN_FRAMES[item[2]]==40)
    profile_plan = [(probe,SEED+105,0),(probe,SEED+106,1)]
    profile_plan += plan[0:4]
    # Warm both resolution variants on a fixed batch before paired synchronization.
    with locked_gpu(net, head):
        kernel_check = verify_prefix()
        print('OCC_PREFIX_CHECK',kernel_check,flush=True)
        measure(net,inventory,profile_plan[:1],('pooled','native','subpixel'))
        profiles = measure(net,inventory,profile_plan,('pooled','native','subpixel'))
        profiles_timing = {mode:timing(profiles,mode) for mode in ('pooled','native','subpixel')}
        chosen = 'native' if passes_performance(profiles_timing['native'],profiles_timing['pooled']) else 'subpixel'
        write_json(output/'profiles.json',dict(profiles=profiles,timing=profiles_timing,selected=chosen))
        print('OCC_PROFILES',profiles_timing,'selected',chosen,flush=True)
        paired = []
        for index,task in enumerate(plan):
            modes = ('pooled',chosen) if index%2==0 else (chosen,'pooled')
            paired.extend(measure(net,inventory,[task],modes))
            print('OCC_REPLAY',index+1,paired[-1]['scene_id'],paired[-1]['frame'],flush=True)
        native_confirmed = chosen=='native' and passes_performance(timing(paired,'native'),timing(paired,'pooled'))
        if chosen=='native' and not native_confirmed:
            chosen = 'subpixel'
            for row,task in zip(paired,plan):
                row['outcomes'].update(measure(net,inventory,[task],('subpixel',))[0]['outcomes'])
            print('OCC_REPLAY_SWITCH',chosen,flush=True)
        optical = None
        if optical_probe and f2_root:
            unit = root.parent.parent/'visibility-qualification/smoke/training-120-2db3c2fe6b5336e7'
            fixture = QualificationData(unit)
            reference = json.loads((optical_probe/'labels/optical-audit.json').read_text())['optical'][0]
            b_spec = dict(tiers='B',unit=unit,frame=30,focus=reference['focus_m'],fstop=reference['fstop'],
                          truth=optical_probe/'labels'/f"{reference['id']}-optics.npz",f2_root=f2_root)
            item = ('B',unit.name,TRAIN_FRAMES.index(30))
            optical = measure(net,fixture,[(item,SEED+300,None)],('pooled',chosen),spec=b_spec)[0]
    unchanged = fingerprint(net,head)==before
    before_stats, after_stats = (full_summary(paired,mode) for mode in ('pooled',chosen))
    saved = dict(
        selected=chosen, optimizer_steps=0, model_state_unchanged=unchanged,
        timing_boundary='preloaded inputs and K16, CUDA-synchronized objective plus one full backward; no weight updates',
        prefix_equivalence=kernel_check, baseline_source_sha256=sha256(audit), profile_timing=profiles_timing, profile_rows=profiles,
        before=before_stats, after=after_stats, rows=paired, optical_crosscheck=optical,
        zero_branches=zero_reasons(paired,chosen),
    )
    write_json(output/'occ-resolution-replay.json',saved)
    print('OCC_REPLAY_RESULT',chosen,before_stats,after_stats,flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qualification',type=Path,required=True)
    parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--optical-probe',type=Path)
    parser.add_argument('--f2-root',type=Path)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':
        parser.error('assign the RTX 5090 explicitly')
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    run(args.qualification,args.audit,args.output,args.optical_probe,args.f2_root)
