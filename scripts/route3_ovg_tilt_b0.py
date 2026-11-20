"""Render the evaluation-only b=0 surfel tilt and Query pairings."""
import argparse
from dataclasses import replace
import math
import time

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

from route3_ovg_1b_optical_data import geometry_root
from route3_ovg_gap_attribution import B_LABELS, npz, ray_attrs, write_stats
from route3_ovg_opaque_b0 import (
    FLOORS, OUT, REC, Model, appearance_states, background_samples, case_frames,
    formal_inputs, frame_models, guard, leakage_region, make_layers, optics, original_B,
    pooled, prefix_transmittance, read, save_json, save_npz, subcamera, tensor, write_csv,
)
from route3_ovg_stage2_evaluate import Q_MIN, boundary_candidate, display, ray_effect, trace_predicted
from route3_upper_bound_evaluate import region_sse
from src.layered_aperture import pixel_layer
from src.ovg_opaque_eval import opaque_view
from src.ovg_optics import raster
from src.ovg_tilt_eval import tilt_visible

DEST = OUT/'tilt-b0'
PREFIX = 'route3-ovg-stage2-2026-11-20-tilt-b0'
STAGES = ('1c', '2b')
MODES = ('original', 'tilt', 'tilt-opaque')


def surface_variant(layers, K, *, enabled=False):
    return replace(layers, visible=tilt_visible(layers.visible, layers.depth, layers.b0,
                                               layers.flags, K, enabled=enabled))


def toy():
    size = 48
    K = tensor(np.array([[70, 0, 24], [0, 70, 24], [0, 0, 1]], np.float32))
    y, x = torch.meshgrid(torch.arange(size, device=K.device)+.5,
                          torch.arange(size, device=K.device)+.5, indexing='ij')
    color = torch.stack((x/size, y/size, torch.full_like(x, .25)), -1)
    camera = torch.eye(4, device=K.device)
    region = (x >= 10) & (x < 38) & (y >= 8) & (y < 40)
    rows = []
    for angle in (0, 30, 60):
        z = (.3**-1 + (x-24)*math.tan(math.radians(angle))/(.3*70)).reciprocal()
        b0 = torch.ones_like(z, dtype=torch.bool)
        base = pixel_layer(color, z, K, camera)
        flags = torch.ones(len(base.means), dtype=torch.bool, device=K.device)
        tilted = tilt_visible(base, z, b0, flags, K, enabled=True)
        disabled = tilt_visible(base, z, b0, flags, K)
        offset = K.new_tensor([-.3*.16/max(math.tan(math.radians(angle)), .5), 0.])
        J, view = subcamera(K, .3, offset)
        Ts = [prefix_transmittance(layer, J, view, torch.full_like(z, 2.))[region]
              for layer in (base, tilted)]
        unchanged = torch.equal(raster(base,K,.3,offset[None],size,size),
                                raster(disabled,K,.3,offset[None],size,size))
        rows.append(dict(angle_deg=angle, offset_m=float(offset[0]), pixels=int(region.sum()),
                         original_gap_rays=int((Ts[0]>=.1).sum()), tilt_gap_rays=int((Ts[1]>=.1).sum()),
                         original_max_T=float(Ts[0].max()), tilt_max_T=float(Ts[1].max()),
                         disabled_bitwise=unchanged,
                         centers_unchanged=torch.equal(base.means, tilted.means),
                         opacity_unchanged=torch.equal(base.opacities, tilted.opacities)))
    return rows


def leakage(layers, K, entry, case, center, regions, target, opaque):
    labels, region = leakage_region(entry, case, center, regions)
    focus = case['nominal_focus_m']
    offsets = tensor(labels['aperture_xy'].astype(np.float32))
    backgrounds = background_samples(dict(tiers='B',entry=entry), K, focus, offsets, (512,512))
    mask = tensor(region)
    base, merged, start = layers.scenes()
    before, after, weights, depths, dominant, ts, closed = [], [], [], [], [], [], []
    for i, offset in enumerate(offsets):
        wq, zq, zv, _, trans = trace_predicted(layers.visible, base, merged, K, focus, offset, mask)
        if opaque:
            args = (K, focus, offset, layers.depth, layers.b0, layers.flags, start, backgrounds[i])
            im0, _ = opaque_view(base, *args)
            im1, meta = opaque_view(merged, *args)
            seal = meta['closed'][mask].cpu().numpy()
            wq = np.where(seal, meta['query_weight'][mask].cpu().numpy(), wq)
            zq = np.where(seal, meta['query_depth'][mask].cpu().numpy(), zq)
            active = seal & (wq >= .001)
            if active.any():
                cutoff = torch.zeros_like(layers.depth)
                cutoff[mask] = tensor(zq.astype(np.float32))
                T = prefix_transmittance(layers.visible, *subcamera(K,focus,offset), cutoff)
                trans = np.where(seal,T[mask].cpu().numpy(),trans)
        else:
            im0, im1 = [raster(scene,K,focus,offset[None],512,512,backgrounds[i:i+1])[0]
                        for scene in (base, merged)]
            seal = np.zeros_like(wq,dtype=bool)
        before.append(im0[mask].cpu().numpy())
        after.append(im1[mask].cpu().numpy())
        weights.append(wq)
        depths.append(zq)
        dominant.append(zv)
        ts.append(np.where(wq>0,trans,np.nan))
        closed.append(seal)
    _, effect, _, _ = ray_effect(np.stack(before), np.stack(after), target[region])
    return dict(mask=region, T=np.stack(ts), query_weight=np.stack(weights),
                query_depth_m=np.stack(depths), dominant_visible_depth_m=np.stack(dominant),
                per_ray_SSE_change=effect, closed=np.stack(closed))


def publish(cases, ray_cases):
    pairs, zeros, rays = [], [], []
    for stage in STAGES:
        scenes = sorted({c['scene'] for c in cases if c['stage']==stage})
        for scope in (*scenes,'pooled'):
            for mode in MODES:
                selection = [c for c in cases if c['stage']==stage and c['mode']==mode
                             and (scope=='pooled' or c['scene']==scope)]
                for focus, region in ((1.4,'in_focus'), (1.4,'recoverable'), (5.,'recoverable')):
                    current = [c for c in selection if c['focus']==focus]
                    scores = [[c['optical']['pairs'][q]['regions'][region] for c in current] for q in (0,1)]
                    q0, q1 = map(pooled, scores)
                    pairs.append(dict(stage=stage,scene=scope,mode=mode,focus_m=focus,region=region,
                        cases=len(current),pixels=sum(r['pixels'] for r in scores[0]),
                        query_off_SSE=sum(r['sse'] for r in scores[0]),
                        query_on_SSE=sum(r['sse'] for r in scores[1]),
                        query_off_PSNR=q0,query_on_PSNR=q1,delta_PSNR=q1-q0))
                for focus in (1.4,5.):
                    current = [c for c in selection if c['focus']==focus]
                    for kind, floor in FLOORS.items():
                        scores = [c['optical']['zero']['visible_only'] if kind=='visible' else
                                  c['optical']['pairs'][1]['zero_combined'] for c in current]
                        zeros.append(dict(stage=stage,scene=scope,mode=mode,focus_m=focus,kind=kind,
                            threshold_dB=floor,pooled_PSNR=pooled(scores),
                            worst_frame_PSNR=min(r['psnr'] for r in scores),
                            passed=sum(r['psnr']>=floor for r in scores),total=len(scores)))
                    sub = [r for r in ray_cases if r['stage']==stage and r['mode']==mode and
                           r['focus']==focus and (scope=='pooled' or r['scene']==scope)]
                    grouped = write_stats(stage,focus,'on' if mode=='tilt-opaque' else 'off',scope,
                                          [r['attrs'] for r in sub],sum(r['domain'] for r in sub))
                    for r in grouped:
                        r['mode'] = mode
                        r['b_source'] = 'b mask from reevaluated frozen endpoint'
                        rays.append(r)
    write_csv(REC/f'{PREFIX}-query.csv',pairs)
    write_csv(REC/f'{PREFIX}-zero.csv',zeros)
    write_csv(REC/f'{PREFIX}-leakage.csv',rays)


@torch.no_grad()
def run(smoke_only=False):
    guard(gpu=True)
    DEST.mkdir(exist_ok=True)
    rows = toy()
    checks = [dict(check='toy-plane',**r) for r in rows]
    print('TOY', rows, flush=True)
    assert all(r['disabled_bitwise'] and r['centers_unchanged'] and r['opacity_unchanged'] and
               r['tilt_gap_rays']==0 and (r['angle_deg']==0 or r['original_gap_rays']>0) for r in rows)
    model = Model()
    states = {m: s for m,s in appearance_states(model).items() if m in STAGES}
    clock = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    cases, ray_cases = [], []
    for name, frame, entry, _, _, _, manifest, items, folder in case_frames():
        guard()
        x,y,g,first,aps = frame_models(model, states, original_B(name,frame))
        K = x['K'][0]
        geom = geometry_root(name)
        fg = npz(geom/'scene-geometry.npz','foreground_object_ids')['foreground_object_ids']
        gt_boundary = npz(B_LABELS/name/'pinhole-subpixels'/f'f{frame:03d}.npz',
                          'depth_boundary')['depth_boundary']
        edge_distance = distance_transform_edt(~gt_boundary)
        for item in items:
            case = item['case']
            if (case['protocol'],case['fstop'])!=('P2',1.4):
                continue
            center,regions,target,pinhole,focus,query = formal_inputs(
                model,name,frame,folder,manifest,item,g,x,y,entry)
            label = npz(geom/'geometry'/f"{case['id']}-K16.npz",
                        'aperture_xy','object_id','depth_axial_m','reference_uv_center','K')
            for stage in STAGES:
                layers = make_layers(x,g,aps[stage],first,stage,query)
                tilted = surface_variant(layers,K,enabled=True)
                if not cases and not ray_cases and stage=='1c':
                    zero = K.new_zeros((1,2))
                    closed = surface_variant(layers,K)
                    z0,zoff,z1 = [raster(v.visible,K,focus,zero,512,512)[0] for v in (layers,closed,tilted)]
                    psnrs = [region_sse(display(im.cpu().numpy()),pinhole,regions['full_image'])['psnr']
                             for im in (z0,z1)]
                    check = dict(check='1c-dev-zero',scene=name,frame=frame,case=case['id'],
                                 disabled_bitwise=torch.equal(z0,zoff),
                                 original_PSNR=psnrs[0],tilt_PSNR=psnrs[1],delta_dB=psnrs[1]-psnrs[0])
                    save_json(DEST/'smoke.json', check)
                    write_csv(REC/f'{PREFIX}-checks.csv', [*checks,check])
                    print('DEV_ZERO',check,flush=True)
                    assert check['disabled_bitwise'] and abs(check['delta_dB'])<=.1
                    if smoke_only:
                        return
                depth = layers.depth.cpu().numpy()
                bmap = (boundary_candidate(aps[stage])&(tensor(first['q'])>=Q_MIN)).cpu().numpy().astype(np.int8)
                for mode in MODES:
                    dest = DEST/'cases'/stage/name/case['id']/mode
                    current = layers if mode=='original' else tilted
                    opaque = mode=='tilt-opaque'
                    if (dest/'case.json').is_file() and (dest/'leakage.npz').is_file():
                        result = read(dest/'case.json')
                        with np.load(dest/'leakage.npz') as f:
                            archive = {k: f[k] for k in f.files}
                    else:
                        start = time.monotonic()
                        stats,images = optics(current,K,focus,1.4,dict(tiers='B',entry=entry),
                                              pinhole,target,regions,opaque)
                        archive = leakage(current,K,entry,case,center,regions,target,opaque)
                        result = dict(stage=stage,scene=name,frame=frame,mode=mode,
                                      focus=case['nominal_focus_m'],actual_focus_m=focus,
                                      optical=stats,seconds=time.monotonic()-start)
                        save_npz(dest/'optical.npz',**images)
                        save_npz(dest/'leakage.npz',**archive)
                        save_json(dest/'case.json',result)
                    cases.append(result)
                    attrs = ray_attrs(archive,(depth,bmap),center,edge_distance,fg,label,
                                      case['nominal_focus_m'],'on' if opaque else 'off')
                    ray_cases.append(dict(stage=stage,scene=name,mode=mode,
                                          focus=case['nominal_focus_m'],attrs=attrs,
                                          domain=archive['T'].size))
                    print('TILT_B0',stage,name,case['id'],mode,round(result['seconds'],2),flush=True)
        model.query.store.inputs.cache_clear()
        model.query.store.active.cache_clear()
    assert len(cases)==108
    publish(cases,ray_cases)
    save_json(DEST/'summary.json',dict(cases=cases, wall_seconds=time.monotonic()-clock,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        scope='P2 f/1.4 K64; K16 attribution on nominal-focus label rays',
        surface='per-primitive centers and opacity unchanged; tangent covariance preserves pinhole footprint',
        closure='same b0 opacity closure as opaque-b0, at unchanged parent depth and center'))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke-only',action='store_true')
    run(parser.parse_args().smoke_only)
