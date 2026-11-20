"""Attribute the saved Stage 2 high-transmittance rays without model inference.

Positions are pixel centers. The off-axis views use lateral camera shifts
with no rotation; the archived K16 label, not P2, supplies focal distance.
The center depth is the frozen full-resolution cache shared by 1c and 2b.
"""
import csv
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs'
REC = ROOT/'03_实验记录/records'
STAGE = OUT/'route3-ovg-stage2-2026-11-19-01'
ALIGN = OUT/'route3-ovg-1c-2026-11-19-01'
REFOCUS = OUT/'route3-ovg-refocus-2026-11-18-01'
CONFIG = json.loads((ROOT/'configs/route3-ovg-1a-2026-11-18.json').read_text())
B_LABELS = ROOT/CONFIG['output']/'B'
TRUTH = ROOT/CONFIG['previous']/'truth'
TRAIN_FRAMES = (1, 6, 11, 16, 21, 26, 30, 35, 40, 45, 50, 55, 60)
METHODS = ('1c', '2b', 'fixed-original')
FOCI = (1.4, 5.)
CLASSES = ('crack', 'silhouette', 'other')
B_SOURCE = {
    '1c': 'cached full-resolution 1c alpha+z1/z2+q; frame positive count verified',
    '2b': 'cached frozen z1/z2+q establish zeros; remaining unknown without saved learned alpha',
    'fixed-original': 'fixed visible geometry has no first-pass split',
}
FIELDS = ('cls', 'sse', 'x_mm', 'y_mm', 'edge_px', 'delta_inv', 'b', 'reason', 'closed')


def npz(path, *keys):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in keys}


def lookups(scene, frame):
    f = f'f{frame:03d}'
    v = npz(ALIGN/'train/aligned/dev-predictions'/scene/f'{f}.npz', 'depth1', 'depth2', 'alpha')
    depth = npz(ALIGN/'C3'/scene/f/'visible-depths.npz', '1c-aligned-EMA')['1c-aligned-EMA']
    np.testing.assert_array_equal(v['depth1'], depth)
    q = npz(STAGE/'observed-cache/512/B'/scene/f'{TRAIN_FRAMES.index(frame):03d}.npz', 'q')['q']
    eligible = (q >= .5) & (v['depth2'] >= np.float32(1.03)*depth)
    b1c = (eligible & (v['alpha'] < .95)).astype(np.int8)
    old_case = STAGE/'evaluation/1c'/scene/f/'P2'/f'{f}-N1.4-F1.4'/'case.json'
    saved = next(r for r in json.loads(old_case.read_text())['rows'] if r['method'] == 'OVG'
                 and r['weights'] == 'EMA' and r['split'] == 'retrieved')
    if int(b1c.sum()) != round(saved['b']*b1c.size):
        raise ValueError(f'Cached 1c b does not reproduce the saved split count: {scene}/{f}')
    fixed = npz(REFOCUS/scene/f/'visible-depths.npz', 'fixed-original')['fixed-original']
    return {'1c': (depth, b1c), '2b': (depth, np.where(eligible, -1, 0).astype(np.int8)),
            'fixed-original': (fixed, np.zeros(fixed.shape, np.int8))}


def project(x, y, z, lens_xy, K, focus):
    return np.stack((x+.5, y+.5), axis=-1) + (
        (1/z - 1/focus)[:, None] * (lens_xy @ K[:2, :2].T)
    )


def ray_attrs(archive, maps, gt, edge, foreground_ids, label, focus, state):
    depth, bmap = maps
    mask = archive['mask']
    h, w = mask.shape
    y, x = np.nonzero(mask)
    zq, zv = archive['query_depth_m'], archive['dominant_visible_depth_m']
    T, effect, weight = archive['T'], archive['per_ray_SSE_change'], archive['query_weight']
    # Match the previous T>=0.1 bin exactly, after the front_order precedence.
    k, j = np.nonzero((weight >= .001) & (effect > 0) & np.isfinite(T) & (T >= .1)
                     & ~((zv > 0) & (zq < zv)))
    p_y, p_x, qz = y[j], x[j], zq[k, j].astype(np.float64)
    good_q = np.isfinite(qz) & (qz > 0)
    lens = label['aperture_xy'][k]
    xy = project(p_x, p_y, np.where(good_q, qz, 1.), lens, label['K'], focus)
    inside = good_q & (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
    cx = np.floor(np.where(inside, xy[:, 0], 0)).astype(np.intp)
    cy = np.floor(np.where(inside, xy[:, 1], 0)).astype(np.intp)
    zp, zpp = depth[p_y, p_x].astype(np.float64), depth[cy, cx].astype(np.float64)
    valid_pp = inside & np.isfinite(zpp) & (zpp > 0)
    valid_p = np.isfinite(zp) & (zp > 0)
    obj = gt['object_id'][p_y, p_x]
    np.testing.assert_array_equal(obj, label['object_id'][k, p_y, p_x])
    if not np.isin(obj, foreground_ids).all():
        raise ValueError('A selected K16 diagnostic ray no longer belongs to the inherited foreground mask')
    # Check the exact aperture convention against the archived GT projection,
    # while using Query depth (not GT) for the actual attribution.
    zg = label['depth_axial_m'][k, p_y, p_x].astype(np.float64)
    gt_uv = project(p_x, p_y, zg, lens, label['K'], focus)
    np.testing.assert_allclose(gt_uv, label['reference_uv_center'][k, p_y, p_x], atol=2e-4, rtol=0)
    earlier = valid_pp & (qz > 1.03*zpp)
    same = obj == gt['object_id'][cy, cx]
    crack = earlier & same & gt['center_visible'][cy, cx]
    silhouette = valid_pp & ~earlier
    cls = np.where(crack, 0, np.where(silhouette, 1, 2))
    reason = np.select(
        [~good_q, good_q & ~inside, inside & ~valid_pp, (cls == 2) & valid_pp],
        [1, 2, 3, 4], default=0,
    )
    delta = np.full(len(k), np.nan, np.float64)
    d_ok = valid_pp & valid_p
    delta[d_ok] = 1/zpp[d_ok]-1/zp[d_ok]
    return dict(cls=cls, sse=effect[k, j], x_mm=lens[:, 0]*1000, y_mm=lens[:, 1]*1000,
                edge_px=edge[p_y, p_x], delta_inv=delta, b=np.where(valid_pp, bmap[cy, cx], -1),
                reason=reason, closed=archive['closed'][k, j] if state == 'on' else np.zeros(len(k), bool))


def write_stats(stage, focus, state, scope, parts, domain):
    merged = {key: np.concatenate([p[key] for p in parts]) for key in FIELDS}
    ngap, sgap = len(merged['sse']), merged['sse'].sum(dtype=np.float64)
    aperture_mm = 1000*.05/(2*1.4)
    for code, category in enumerate(CLASSES):
        chosen = merged['cls'] == code
        m = {k: merged[k][chosen] for k in FIELDS}
        n, sse = len(m['sse']), m['sse'].sum(dtype=np.float64)
        row = dict(stage=stage, nominal_focus_m=focus, opacity_switch=state, scene=scope,
                   category=category, cases=len(parts), internal_ray_count=domain,
                   gap_ray_count=ngap, gap_positive_SSE=sgap, ray_count=n, positive_SSE=sse,
                   fraction_of_gap_rays=n/ngap if ngap else None,
                   fraction_of_gap_positive_SSE=sse/sgap if sgap else None,
                   b_source=B_SOURCE[stage])

        def bucket(tag, bins):
            for key, selects in bins:
                row[f'{tag}_{key}_rays'] = int(selects.sum())
                row[f'{tag}_{key}_positive_SSE'] = m['sse'][selects].sum(dtype=np.float64)

        def distribution(tag, data):
            data = data[np.isfinite(data)]
            quantiles = np.quantile(data, [.05, .25, .50, .75, .95]) if len(data) else [None]*5
            row[f'{tag}_valid_rays'] = len(data)
            for q, v in zip((5, 25, 50, 75, 95), quantiles):
                row[f'{tag}_p{q:02}'] = v

        r = np.hypot(m['x_mm'], m['y_mm'])
        distribution('aperture_radius_mm', r)
        row['aperture_mean_x_mm'] = m['x_mm'].mean() if n else None
        row['aperture_mean_y_mm'] = m['y_mm'].mean() if n else None
        a = r/aperture_mm
        bucket('aperture_fraction', [('0_to_025', a <= .25), ('025_to_050', (a > .25) & (a <= .5)),
                                     ('050_to_075', (a > .5) & (a <= .75)), ('above_075', a > .75)])
        d = m['edge_px']
        distribution('gt_subpixel_edge_distance_px', d)
        bucket('gt_edge_px', [('0_to_2', d <= 2), ('2_to_8', (d > 2) & (d <= 8)), ('above_8', d > 8)])
        distribution('pred_inverse_depth_pprime_minus_p_m_inv', m['delta_inv'])
        bucket('pprime_b', [('0', m['b'] == 0), ('1', m['b'] == 1), ('unknown', m['b'] == -1)])
        for j, k in enumerate(('invalid_query_depth', 'outside_image', 'invalid_cached_depth',
                               'earlier_depth_different_or_invalid_GT_object'), start=1):
            row[f'other_{k}_rays'] = int((m['reason'] == j).sum())
        row['opaque_closed_rays'] = int(m['closed'].sum()) if state == 'on' else None
        yield row


def main():
    previous_table = REC/'route3-ovg-stage2-2026-11-20-opaque-b0-leakage.csv'
    with previous_table.open(newline='') as f:
        reference = {(r['stage'], float(r['nominal_focus_m']), r['mode'], r['scene']):
                     (int(r['ray_count']), float(r['positive_SSE']), int(r['domain_rays']))
                     for r in csv.DictReader(f) if r['category'] == 'T>=0.1' and r['stage'] in METHODS}
    scenes = sorted({k[3] for k in reference if k[3] != 'pooled'})
    groups = {}
    for scene in scenes:
        geom = TRUTH/scene/'geometry'
        fg = npz(TRUTH/scene/'scene-geometry.npz', 'foreground_object_ids')['foreground_object_ids']
        for frame in (1, 30, 60):
            f = f'f{frame:03d}'
            gt = npz(geom/f'{f}-center.npz', 'object_id', 'center_visible')
            boundary = npz(B_LABELS/scene/'pinhole-subpixels'/f'{f}.npz', 'depth_boundary')['depth_boundary']
            if not boundary.any():
                raise ValueError(f'No saved GT subpixel boundary in {scene}/{f}')
            edge = distance_transform_edt(~boundary)
            maps = lookups(scene, frame)
            for focus in FOCI:
                cid = f'{f}-N1.4-F{focus:g}'
                label = npz(geom/f'{cid}-K16.npz', 'aperture_xy', 'object_id', 'depth_axial_m',
                            'reference_uv_center', 'K')
                for method in METHODS:
                    legacy_stage = '1c' if method == 'fixed-original' else method
                    old = STAGE/'diagnostics-predicted-rays'/legacy_stage/scene/f/cid/f'leakage-{method}.npz'
                    new = STAGE/'opaque-b0/cases'/method/scene/cid/'leakage-on.npz'
                    for state, source in (('off', old), ('on', new)):
                        keys = ('mask', 'query_depth_m', 'dominant_visible_depth_m', 'T',
                                'per_ray_SSE_change', 'query_weight') + (('closed',) if state == 'on' else ())
                        arc = npz(source, *keys)
                        attrs = ray_attrs(arc, maps[method], gt, edge, fg, label, focus, state)
                        key = method, focus, state, scene
                        domain = arc['T'].size
                        groups.setdefault(key, []).append((attrs, domain))
        print('GAP_ATtribution', scene, flush=True)

    rows = []
    for method in METHODS:
        for focus in FOCI:
            for state in ('off', 'on'):
                for scope in (*scenes, 'pooled'):
                    key = method, focus, state, scope
                    group = groups[key] if scope != 'pooled' else [
                        v for scene in scenes for v in groups[method, focus, state, scene]]
                    attrs, domains = zip(*group)
                    result = list(write_stats(*key, attrs, sum(domains)))
                    n, sse, domain = reference[key]
                    assert sum(r['ray_count'] for r in result) == n and sum(domains) == domain, key
                    assert np.isclose(sum(r['positive_SSE'] for r in result), sse, atol=1e-9, rtol=1e-10), key
                    rows.extend(result)

    out = REC/'route3-ovg-stage2-2026-11-20-gap-attribution.csv'
    temporary = out.with_suffix('.part')
    with temporary.open('w', newline='') as f:
        writer = csv.DictWriter(f, list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(out)
    print(f'{len(rows)} rows: {out.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
