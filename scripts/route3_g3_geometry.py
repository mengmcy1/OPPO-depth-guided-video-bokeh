"""Appendix13 T4: change explicit depth, keep SHARP A/fallback fixed."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
import cv2
import numpy as np
import torch
from route3_order_fallback import (SCENES, desc_for, EXPLICIT, ALIGN_ROOT, load,
                                  tensor, cd as old_cd, display, zero_parts)
from route3_local_replacement import fit_and_map, reference
from route3_depth_interface import write, save
from route3_explicit_layer import aligned_hidden
from route3_s2a_predict import visible_cloud
from route3_upper_bound_evaluate import score, pool
from route3_ldr_u1r import model
from src.layered_aperture import pixel_layer, merge_layers
from src.ordered_layers import order_hidden, ray_samples, render_fallbacks
PREFIX = 'route3-g3-r13-2026-11-18'
OUT = ROOT/'outputs'/f'{PREFIX}-01'/'T4'
REC = ROOT/'03_实验记录/records'
LOCK = ROOT/'logs'/PREFIX/'gpu0.lock'


def stats(z, gt, valid):
    valid = valid & np.isfinite(z) & (z > 0) & (gt > 0)
    if not valid.any():
        return dict(pixels=0)
    e = abs(z[valid]-gt[valid])/gt[valid]
    return dict(pixels=int(valid.sum()), AbsRel=float(e.mean()),
                relative_error_p50_p90_p99=np.quantile(e, [.5, .9, .99]).tolist())


def prepare(version):
    for name in SCENES:
        desc = desc_for(name)
        for e in desc['entries']:
            if (OUT.parent/'STOP.json').exists():
                print('T4_PAUSED_AT_T2_STOP', flush=True)
                return
            f = e['frame']
            dest = OUT/version/name/f'f{f:03d}'
            if (dest/'geometry.json').exists():
                continue
            dest.mkdir(parents=True, exist_ok=True)
            base = load(EXPLICIT/name/f'f{f:03d}'/'explicit-depth.npz')['depth']
            maps = load(EXPLICIT/name/f'f{f:03d}'/'bands.npz')
            band = np.zeros(base.shape, bool)
            max_coc = 0.
            from src.dual_layer import depth_band
            own = load(ALIGN_ROOT/name/f'f{f:03d}'/'visible-depth.npz')['depth']
            d = load(e['input_npz'])
            for c in e['cases']:
                if c['fstop'] != 1.4:
                    continue
                meta = json.loads((old_cd(name, f, c, 'P2')/'render.json').read_text())
                band |= maps['P2__'+c['id']]
                _, _, coc = depth_band(own, d['K'][0, 0], meta['focus_m'], .05/2.8)
                max_coc = max(max_coc, float(coc.max()))
            radius = int(np.ceil(max_coc))
            expanded = cv2.dilate(band.astype(np.uint8), np.ones((2*radius+1, 2*radius+1), np.uint8)) > 0
            if version == 'flow':
                refroot = (ROOT/'outputs/route3-sharp-s1c-2026-11-10-01/calibration/E1'
                           if name == 'B2-high' else desc['confidence'])
                ref = load(refroot/f'f{f:03d}-reference.npz')
                candidate = ref['depth']
                good = ref['valid'] & (ref['support'] >= 2) & np.isfinite(candidate) & (candidate > 0)
                provenance = str(refroot/f'f{f:03d}-reference.npz')
            else:
                p = OUT/'depthsplat-raw'/name/f'f{f:03d}.npz'
                raw = load(p)
                candidate = raw['depth']
                good = raw['consistent'] & np.isfinite(candidate) & (candidate > 0)
                provenance = str(p)
            replace = expanded & good
            depth = np.where(replace, candidate, base).astype(np.float32)
            g = load(desc['geometry']/f'f{f:03d}-center.npz')
            fg = np.load(desc['scene']/'scene-geometry.npz')['foreground_object_ids']
            parents = load(desc['parents'](f))
            a, b, supported, mapped, valid, audit = fit_and_map(g, depth, fg, parents)
            save(dest/'depth.npz', depth=depth, base_depth=base, band=band,
                 expanded=expanded, replaced=replace, valid_reference=good)
            save(dest/'alignment.npz', a=a, b=b, local_supported=supported,
                 parent_depth_aligned=mapped, parent_valid=valid)
            known = g['center_visible']
            write(dest/'geometry.json', dict(scene=name, frame=f, version=version,
                  reference=provenance, expansion_radius_px=radius,
                  band_pixels=int(band.sum()), expanded_pixels=int(expanded.sum()),
                  replaced_pixels=int(replace.sum()), replacement_fraction_in_band=float(replace[band].mean()),
                  depth_error=dict(before=stats(base, g['depth_axial_m'], band & known),
                                   after=stats(depth, g['depth_axial_m'], band & known)),
                  placement=audit, SHARP_A_unchanged=True,
                  P2_focus_and_A_per_ray_fallback_unchanged=True,
                  GT_depth_used_for_reference_selection=False))
            print('T4_PREPARED', version, name, f, float(replace.mean()), flush=True)


def render(version):
    from route3_g3_r13 import guard
    torch.set_num_threads(4)
    for name in SCENES:
        desc = desc_for(name)
        for e in desc['entries']:
            f = e['frame']
            dest = OUT/version/name/f'f{f:03d}'
            with LOCK.open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                guard(True)
                start = time.monotonic()
                d = load(e['input_npz'])
                K = tensor(d['K'])
                depth = load(dest/'depth.npz')['depth']
                A = visible_cloud(desc, e)
                D = pixel_layer(tensor(d['rgb_linear']), tensor(depth), K, torch.eye(4, device='cuda'))
                fill = tensor(load(EXPLICIT/'fill'/(name+f'-f{f:03d}')/'fill.npz')['fill_rgb_linear'])
                bg = tensor(load(EXPLICIT/name/f'f{f:03d}'/'fill-input.npz')['background_depth'])
                parents = load(desc['parents'](f))
                align = load(dest/'alignment.npz')
                with torch.inference_mode():
                    ra, aa = zero_parts(A, K)
                    Azero = ra+(1-aa[..., None])*fill
                for c in e['cases']:
                    if (OUT.parent/'STOP.json').exists():
                        print('T4_PAUSED_AT_T2_STOP', flush=True)
                        return
                    out = dest/c['id']
                    if (out/'render.json').exists():
                        continue
                    out.mkdir(exist_ok=True)
                    base = old_cd(name, f, c, 'P2')
                    meta = json.loads((base/'render.json').read_text())
                    updated = dest/'focus.json'
                    if updated.exists():
                        from src.dual_layer import depth_band
                        focus = json.loads(updated.read_text())[str(c['nominal_focus_m'])]['focus_m']
                        band = depth_band(depth, d['K'][0, 0], focus, .05/(2*c['fstop']))[0]
                    else:
                        focus = meta['focus_m']
                        band = load(EXPLICIT/name/f'f{f:03d}'/'P2'/c['id']/'masks.npz')['band']
                    hidden, _ = aligned_hidden(parents, align, band, K)
                    hidden, ordering = order_hidden(hidden, tensor(depth), K, 'O1')
                    with torch.inference_mode():
                        As = ray_samples(A, K, focus, .05/(2*c['fstop']), 64, fill, bg)
                    times = []
                    for label, h in [('D0', None), ('D1', hidden)]:
                        t = time.monotonic()
                        torch.cuda.reset_peak_memory_stats()
                        merged = merge_layers(D, h)
                        with torch.inference_mode():
                            p = render_fallbacks(merged, K, focus, .05/(2*c['fstop']), 64, fill, bg, As)
                            r, a = zero_parts(merged, K)
                        save(out/(label+'.npz'), rgb_linear=p['F2'].cpu().numpy(),
                             prediction_srgb=display(p['F2'].cpu().numpy()),
                             zero_srgb=display((r+(1-a[..., None])*Azero).cpu().numpy()))
                        times.append(dict(method=label, seconds=time.monotonic()-t,
                                          peak_MiB=torch.cuda.max_memory_allocated()/2**20))
                        del merged, p
                    write(out/'render.json', dict(scene=name, frame=f, id=c['id'], version=version,
                          focus_m=focus, P2_focus_fixed_to_T1=not updated.exists(), order=ordering, timings=times))
                    print('T4_RENDER', version, name, f, c['id'], flush=True)
                    del As, hidden
                del A, D
                torch.cuda.empty_cache()
                print('T4_FRAME_SECONDS', version, name, f, time.monotonic()-start, flush=True)


def evaluate(version):
    net = model()
    torch.set_num_threads(4)
    rows = []
    for name in SCENES:
        desc = desc_for(name)
        for e in desc['entries']:
            f = e['frame']
            d = load(e['input_npz'])
            for c in e['cases']:
                truth, masks = reference(name, desc, c, f)
                dest = OUT/version/name/f'f{f:03d}'/c['id']
                if not (dest/'render.json').exists():
                    continue
                methods = dict(D0=dest/'D0.npz', D1=dest/'D1.npz',
                               original_D1=old_cd(name, f, c, 'P2')/'D1_O1_F2.npz',
                               GT_D1=old_cd(name, f, c, 'P2')/'DGT1_F2.npz',
                               A=old_cd(name, f, c, 'P2')/'A.npz')
                for label, path in methods.items():
                    p = load(path)
                    result = score(p['prediction_srgb'], truth, masks, net)
                    zero = float(-10*np.log10(max(np.mean((p['zero_srgb'].astype(np.float64)-d['rgb_srgb'])**2), 1e-30)))
                    for region, s in result.items():
                        rows.append(dict(scene=name, frame=f, id=c['id'], focus=c['nominal_focus_m'],
                                         fstop=c['fstop'], method=label, region=region, zero_PSNR=zero, **s))
    groups = {}
    for r in rows:
        key = (r['scene'], r['focus'], r['fstop'], r['method'], r['region'])
        groups.setdefault(key, []).append(r)
    pooled = [dict(scene=k[0], focus=k[1], fstop=k[2], method=k[3], region=k[4], **pool(v)) for k, v in groups.items()]
    recovery = []
    for r in pooled:
        if r['method'] != 'D1':
            continue
        matching = {x['method']: x for x in pooled if all(x[k] == r[k] for k in ['scene', 'focus', 'fstop', 'region'])}
        denom = matching['GT_D1']['psnr']-matching['original_D1']['psnr']
        recovery.append(dict(scene=r['scene'], focus=r['focus'], fstop=r['fstop'], region=r['region'],
                             gain_dB=r['psnr']-matching['original_D1']['psnr'], GT_gap_dB=denom,
                             raw_formula_ratio=(r['psnr']-matching['original_D1']['psnr'])/denom if abs(denom) > 1e-8 else None,
                             ratio=(r['psnr']-matching['original_D1']['psnr'])/denom if denom > 1e-8 else None,
                             positive_GT_gap=denom > 1e-8,
                             versus_A_dB=r['psnr']-matching['A']['psnr']))
    write(REC/(PREFIX+'-T4-'+version+'.json'), dict(rows=rows, pooled=pooled, geometry_gap_recovery=recovery,
          E1_given_GT_poses=True, A_and_fallback_unchanged=True, GT_D1_uses_GT_selected_focus=True))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['prepare', 'render', 'evaluate'])
    p.add_argument('--version', choices=['flow', 'depthsplat'], required=True)
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    globals()[a.mode](a.version)
