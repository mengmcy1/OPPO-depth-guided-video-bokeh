"""Input-only candidate caches and held-apart GT coverage diagnostics."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

import numpy as np

from route3_g3_r15 import DEFAULT, ORIGINAL, OUT, PREV, qa, read, write, link
from route3_g3_r14_geometry import registered_masks, run_depthsplat
from route3_g3_r15_observations import Observations, surfaces, pixels, project
from route3_g3_geometry import SCENES, desc_for, load
from route3_pilot_models import FRAMES
from route3_s2a_candidates import load_views, prepare
from src.candidate_query_data import pack_queries
from src.dual_layer import depth_band

CASES = (1, 30, 60)
DEST = OUT / 'R4'


def pilot_cache():
    plan = []
    pilots = {}
    for name in SCENES[:2]:
        desc = desc_for(name)
        original = desc['source_root']
        views = {v['frame']: v for v in load_views(original)}
        pilots[name] = (original, views)
        raw = DEST / 'pilot-depthsplat' / name
        legacy_raw = PREV / 'T4/depthsplat-official-raw' / name
        link(legacy_raw, raw, [f'f{f:03d}.npz' for f in CASES])
        for f, v in views.items():
            closest = [f] + sorted((g for g in FRAMES if g != f),
                       key=lambda g: (np.linalg.norm(v['c2w'][:3, 3] - views[g]['c2w'][:3, 3]), g))[:7]
            plan.append(dict(scene=name, target_frame=f, target_index=0, frames=closest,
                             bounds_m=[.5, 200.], images=[str(original/'inputs'/f'f{g:03d}.png') for g in closest],
                             K=[views[g]['K'].tolist() for g in closest],
                             c2w=[views[g]['c2w'].tolist() for g in closest],
                             output=str(raw / f'f{f:03d}.npz')))
    path = DEST / 'pilot-depthsplat-plan.json'
    write(path, plan)
    if not all(Path(row['output']).exists() for row in plan):
        run_depthsplat(path, 'R4-pilot-depthsplat')
    folders = {}
    for name, (original, views) in pilots.items():
        masks = registered_masks(views)
        folders[name] = {}
        for variant in ('original', 'depthsplat-official'):
            folder = DEST / variant / name
            folders[name][variant] = folder
            if (folder / 'memory/verified-summary.json').exists():
                continue
            link(original, folder, ['scene.json', 'scene-geometry.npz', 'geometry'])
            link(original, folder, ['inputs'])
            link(original / 'models', folder / 'models', ['SHARP'])
            cal = original / 'models/calibration'
            farm = folder / 'models/calibration'
            link(cal, farm, [p.name for p in cal.iterdir() if p.name != 'SHARP'])
            (farm / 'SHARP').mkdir(parents=True, exist_ok=True)
            passes = folder / 'input-passes-v2'
            link(original / 'inputs', passes, [f'f{f:03d}-{kind}-{f:04d}.exr' for f in FRAMES
                                                for kind in ('Position', 'IndexOB')])
            for f, v in views.items():
                data = load(cal / 'SHARP' / f'f{f:03d}.npz')
                z = v['depth']
                if f in CASES:
                    z = load(PREV / 'T4' / variant / name / f'f{f:03d}/depth.npz')['depth']
                elif variant == 'depthsplat-official':
                    raw = load(DEST / 'pilot-depthsplat' / name / f'f{f:03d}.npz')
                    z = np.where(raw['consistent'] & ~masks[f], raw['depth'], z)
                dest = farm / 'SHARP' / f'f{f:03d}.npz'
                if not dest.exists():
                    np.savez_compressed(dest, **dict(data, depth=z, dominant_valid=np.isfinite(z) & (z > 0)))
            prepare(folder, dominant_p2=True)
            import route3_s2a_memory as memory
            import sys
            memo_args = sys.argv
            try:
                memory.OUT = folder.parent
                sys.argv = [memo_args[0], '--scene', name]
                memory.main()
            finally:
                sys.argv = memo_args
    return folders


def reasons(geo, needed, native, views, frame):
    """Diagnose every eligible query on the first recoverable K16 ray.

    The discrete ray is a representative witness. The uncovered/mixed labels
    are categories of that witness, and cannot be read as unique causal shares
    across all 16 rays.
    """
    y, x = np.nonzero(needed)
    if not len(y):
        return {}
    ray = geo['recoverable'][:, y, x].argmax(0)
    target = views[FRAMES.index(frame)]
    pose, K = target['c2w'], target['K']
    center_uv = geo['reference_uv_center'][ray, y, x]
    z = geo['depth_axial_m'][ray, y, x]
    if 'world_position' in geo:
        world = geo['world_position'][ray, y, x]
    else:
        camera = np.c_[center_uv, np.ones(len(z))] @ np.linalg.inv(K).T * z[:, None]
        world = camera @ pose[:3, :3].T + pose[:3, 3]
    obj = geo['object_id'][ray, y, x]
    observed, blocked, viable, consistent, color_mismatch = np.zeros((5, len(z)), bool)
    # Same physical observation is checked across all available input frames.
    for v, (surface, oid, foot) in zip(views, native):
        if v['frame'] == frame:
            continue
        p = v['c2w']
        camera = (world - p[:3, 3]) @ p[:3, :3]
        ij, inside = pixels(project(camera, K))
        sx, sy = ij.T
        same = (oid[sy, sx] == obj) & (np.linalg.norm(surface[sy, sx] - world, axis=-1) <= 1.5 * foot[sy, sx] + 1e-4)
        witness = inside & same & (camera[:, 2] > 0)
        observed |= witness
        estimated = np.c_[ij + .5, np.ones(len(z))] @ np.linalg.inv(K).T * v['depth'][sy, sx, None]
        placed = (estimated @ p[:3, :3].T + p[:3, 3] - pose[:3, 3]) @ pose[:3, :3]
        uv = project(placed, K)
        xy, inside_target = pixels(uv)
        px, py = xy.T
        visible = target['depth'][py, px]
        rejected = placed[:, 2] <= visible + np.maximum(.001, .005 * visible)
        spatial_ok = inside_target & (np.linalg.norm(uv - center_uv, axis=-1) <= 1.5)
        blocked |= witness & spatial_ok & rejected
        geometric_ok = spatial_ok & ~rejected & (abs(placed[:, 2] - z) <= np.maximum(.02, .02 * z))
        valid = witness & geometric_ok
        viable |= valid
        color_mismatch |= valid & v['edge'][sy, sx]
        consistent |= valid & ~v['edge'][sy, sx]
    # Exclusive priority with an explicit residual for packing and the
    # difference between a contributing ray and its center projection.
    classes = {
        'no_observed_witness': ~observed,
        'source_edge_mixing_risk': viable & ~consistent & color_mismatch,
        'estimated_occlusion_rejection': observed & ~viable & blocked,
        'estimated_depth_or_projection_wrong': observed & ~viable & ~blocked,
        'viable_witness_other_query_or_top8': viable & consistent,
    }
    assert all(np.sum(np.stack(list(classes.values())), axis=0) == 1)
    return {key: int(value.sum()) for key,value in classes.items()}


def one(name, folders):
    cached = DEST / name / 'coverage.json'
    if cached.exists():
        return read(cached)
    start = time.monotonic()
    native = surfaces(folders['original'], FRAMES)
    by_variant = {}
    for version, folder in folders.items():
        qa.guard()
        views = load_views(folder)
        poses = np.stack([v['c2w'] for v in views])
        entries = read(folder/'prepared.json')['rows']
        rows = []
        for entry in entries:
            f = entry['frame']
            v = views[FRAMES.index(f)]
            g = load(folder/'geometry'/f'f{f:03d}-center.npz')
            union = np.zeros((512, 512), bool)
            bands = {}
            for c in entry['cases']:
                band = depth_band(v['depth'], v['K'][0,0], c['focus_m'], .05/(2*c['fstop']))[0]
                bands[c['id']] = band
                union |= band
            # Keep the depth-sorted top 8 exactly as used by 512/13.
            raw = load(entry['pool'])
            queries, _ = pack_queries(raw, union, 8, 262144)
            p = queries['query_pixel']
            valid = queries['candidate_valid']
            packed = queries['candidate_indices'][valid]
            oid, src = raw['slot'][packed],raw['source_xy'][packed]
            real = Observations.from_sources(native, poses, FRAMES.index(f), v['K'], g['depth_axial_m'], oid, src)
            for c in entry['cases']:
                cid = c['id']
                geo = load(folder/'geometry'/(cid+'-K16.npz'))
                good = np.zeros(valid.shape,bool)
                good[valid] = real.hit_fraction(geo,c['nominal_focus_m']) > 0
                band = bands[cid]
                at = band.ravel()[p]
                selected = at & good.any(-1)
                found = np.zeros_like(band)
                found.ravel()[p[selected]] = True
                recoverable = geo['mask_recoverable_total'] & geo['recoverable'].any(0)
                target_counts = reasons(geo,recoverable & band & ~found,native,views,f)
                source_counts = np.bincount(raw['slot'][queries['candidate_indices'][good & at[:, None]]],
                                           minlength=len(FRAMES))
                rows.append(dict(id=cid,frame=f,focus=c['nominal_focus_m'],fstop=c['fstop'],
                                 band_queries=int(at.sum()),queries_with_correct=int(selected.sum()),
                                 recoverable=int(recoverable.sum()),recoverable_in_band=int((recoverable&band).sum()),
                                 recoverable_correct=int((recoverable&found).sum()),
                                 missing_causes=target_counts,
                                 correct_observations_by_source=dict(zip(map(str,FRAMES),source_counts.tolist()))))
        by_variant[version] = dict(cache=str(folder),rows=rows)
    value = dict(scene=name,geometries=by_variant,seconds=time.monotonic()-start)
    write(DEST / name / 'coverage.json',value)
    print('R4_COVERAGE',name,flush=True)
    return value


def main():
    split = read(OUT/'split.json')
    with ThreadPoolExecutor(max_workers=3) as group:
        futures = [group.submit(one,n,{'original':ORIGINAL/n,'depthsplat-official':DEFAULT/n}) for n in split['train']]
        pilots = pilot_cache()
        futures += [group.submit(one,n,folders) for n,folders in pilots.items()]
        rows = [future.result() for future in futures]
    write(DEST/'summary.json',dict(rows=rows,heldout_not_used=True, candidate_size=8,
           source_frame_distribution='all labeled correct observations in the packed query, may count repeat use',
           cause_order='one recoverable K16 witness per in-band missing pixel; reasons are exclusive proxies',
           edge_mixing_diagnostic='all feasible witnesses near their predicted source depth edge; mixing risk proxy',
           upstream_geometry='G3 two r14 provenance pools reused; G2 inputs replayed with original and official DepthSplat geometry',
           R3_default_cache='DepthSplat candidates already regenerated from this geometry and available before R3'))


if __name__ == '__main__':
    main()
