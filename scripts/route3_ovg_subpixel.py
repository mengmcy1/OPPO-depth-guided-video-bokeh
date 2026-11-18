"""Pinhole subpixel hits and visible-layer supervision, independent of K16."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT/'configs/route3-ovg-1a-2026-11-18.json').read_text())
OUT = ROOT/CONFIG['output']
PREVIOUS = ROOT/CONFIG['previous']
SPEC = CONFIG['subpixel']
TRAIN_FRAMES = [1, 6, 11, 16, 21, 26, 30, 35, 40, 45, 50, 55, 60]


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.part')
    tmp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n')
    tmp.replace(path)


def save_npz(path, **fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.part')
    with tmp.open('wb') as stream:
        np.savez_compressed(stream, **fields)
    tmp.replace(path)


def offsets(grid, shift=(0, 0)):
    x = (np.arange(grid) + .5)/grid - .5
    xx, yy = np.meshgrid(x, x)
    return np.stack([xx.ravel()+shift[0], yy.ravel()+shift[1], np.zeros(grid**2)], -1)


def two_layers(depth, normal, object_id, tolerance=SPEC['separation_relative']):
    """Cluster first hits within one pinhole pixel by relative depth.

    Mixed pixels require two supported depth modes. Empty rays and unreliable
    modes have separate supervision masks. Normals represent the front mode.
    """
    valid = (depth > 0) & np.isfinite(depth)
    count = valid.sum(-1)
    # Choose the first supported gap. A one-sample foreground remains in alpha
    # but cannot supervise the two-mode mixture.
    z = np.sort(np.where(valid, depth, np.inf), axis=-1)
    finite_pair = np.isfinite(z[..., 1:])
    step = np.zeros_like(z[..., 1:])
    np.subtract(z[..., 1:], z[..., :-1], where=finite_pair, out=step)
    gap = finite_pair & (step > tolerance*np.where(finite_pair, z[..., :-1], 1))
    first = gap.argmax(-1)
    boundary = np.take_along_axis(z, first[..., None], -1)[..., 0]
    limit = np.where(gap.any(-1), boundary, np.inf)
    front = valid & (depth <= limit[..., None])
    back = valid & ~front
    n1, n2 = front.sum(-1), back.sum(-1)
    reciprocal = np.divide(1, depth, out=np.zeros_like(depth), where=valid)
    mu1 = np.divide(n1, (reciprocal*front).sum(-1), out=np.zeros(count.shape, np.float32), where=n1>0)
    mu2 = np.divide(n2, (reciprocal*back).sum(-1), out=np.zeros(count.shape, np.float32), where=n2>0)
    mean_n = (normal * front[..., None]).sum(-2)/np.maximum(n1[..., None], 1)
    length = np.linalg.norm(mean_n, axis=-1)
    same_front = abs(depth - mu1[..., None]) <= tolerance*mu1[..., None]
    same_back = abs(depth - mu2[..., None]) <= tolerance*mu2[..., None]
    pure_clusters = ((~front | same_front) & (~back | same_back)).all(-1)
    label_valid = n1 > 0
    alpha_valid = (count == depth.shape[-1]) & pure_clusters
    dual = alpha_valid & (n1 >= SPEC['min_second_layer_hits']) & (n2 >= SPEC['min_second_layer_hits'])
    front_idx = np.where(front, depth, np.inf).argmin(-1)
    return dict(depth1=mu1, depth2=mu2, alpha=(n1/depth.shape[-1]).astype(np.float32),
                normal=(mean_n/np.maximum(length[..., None], 1e-8)).astype(np.float32),
                valid=label_valid, valid_alpha=alpha_valid, valid_dual=dual,
                valid_normal=label_valid & (length >= SPEC['stable_cluster_normal_cos']),
                first_hit_object=np.take_along_axis(object_id, front_idx[..., None], -1)[..., 0],
                valid_subrays=count.astype(np.uint16), depth_boundary=gap.any(-1))


def intersect(world, frame, xy, grid, shift=(0, 0)):
    p = np.c_[xy+.5, np.ones(len(xy))]
    C = world.cameras[frame-1]
    ray, tri_map = world.geometry(frame)
    # Plane depth of Embree's first hit with an axial (z=1) ray.
    triangles = ray.mesh.triangles
    normal = ray.mesh.face_normals
    ratio = ((triangles[:, 0]-C[:3, 3])*normal).sum(-1)
    inv = np.linalg.inv(world.K)
    samples = offsets(grid, shift)
    result = []
    for start in range(0, len(p), 4096):
        local_rays = (p[start:start+4096, None]+samples) @ inv.T
        shape = local_rays.shape[:-1]
        direction = (local_rays @ C[:3, :3].T).reshape(-1, 3)
        hit = ray.intersects_first(np.broadcast_to(C[:3, 3], direction.shape).copy(), direction)
        first = hit >= 0
        safe = np.maximum(hit, 0)
        n = normal[safe]
        den = (direction*n).sum(-1)
        ok = first & (abs(den)>1e-12)
        depth = np.divide(ratio[safe], den, out=np.zeros(len(hit)), where=ok)
        first &= depth > 0
        n_camera = (n * np.where(den[:, None] > 0, -1, 1)) @ C[:3, :3]
        result.append((np.where(first,depth,0).astype(np.float32).reshape(shape),
                       np.where(first[:, None],n_camera,0).astype(np.float32).reshape(*shape, 3),
                       np.where(first,world.tri_obj[tri_map[safe]],0).astype(np.int16).reshape(shape)))
    return tuple(np.concatenate([row[j] for row in result]) for j in range(3))


def rectangle_pixels(depth, limit):
    """Deterministic mix: 3/4 depth-edge points plus valid interior points."""
    import cv2
    d = np.where(depth>0, 1/np.maximum(depth,1e-6), 0).astype(np.float32)
    edge = (cv2.dilate(d,np.ones((5,5),np.uint8))-cv2.erode(d,np.ones((5,5),np.uint8))) > .03*d
    pixels = []
    for mask, quota in ((edge, limit*3//4),(~edge, limit//4)):
        y, x = np.nonzero(mask & (depth > 0))
        if len(x):
            pick = np.linspace(0,len(x)-1,min(quota,len(x)),dtype=int)
            pixels.append(np.c_[x[pick], y[pick]])
    return np.concatenate(pixels)


def compare(a,b):
    common = a['valid'] & b['valid']
    edge = a['depth_boundary'] | b['depth_boundary']
    eligible = a['valid_alpha'] & b['valid_alpha'] & edge
    selected = abs(a['alpha']-b['alpha'])[eligible]
    agreement = (a['valid_dual']==b['valid_dual'])[edge]
    return dict(pixels=int(common.sum()), boundary_pixels=int(edge.sum()),
                coverage_mae=float(selected.mean()) if len(selected) else 0.,
                coverage_p95=float(np.quantile(selected,.95)) if len(selected) else 0.,
                two_layer_agreement=float(agreement.mean()) if len(agreement) else 1.,
                median_front_inverse_m=float(np.median(abs(1/a['depth1'][common]-1/b['depth1'][common]))) if common.any() else 0.)


def check_pass(values):
    return all((values[k] >= bound if k=='two_layer_agreement' else values[k] <= bound)
               for k,bound in SPEC['comparison'].items())


def run_scene(name):
    from route3_blender_b2_geometry import World
    import shutil
    start = time.monotonic()
    base = PREVIOUS/'truth'/name
    dest = OUT/'B'/name
    done = dest/'manifest.json'
    if done.exists():
        return json.loads(done.read_text())
    world = World(base/'scene-geometry.npz', 512)
    checks = []
    for f in [1,30,60]:
        with np.load(base/'geometry'/f'f{f:03d}-center.npz') as center:
            xy = rectangle_pixels(center['depth_axial_m'], 8192)
        a, b = [two_layers(*intersect(world,f,xy,n)) for n in [4,8]]
        values = compare(a,b)
        checks.append(dict(frame=f,passed=check_pass(values),**values))
    grid = 4 if all(x['passed'] for x in checks) else 8
    rows = []
    yy, xx = np.mgrid[:world.size, :world.size]
    xy = np.c_[xx.ravel(), yy.ravel()]
    for f in TRAIN_FRAMES:
        if shutil.disk_usage(OUT).free/2**30 < 60:
            raise RuntimeError('Data disk below 60 GiB')
        path = dest/'pinhole-subpixels'/f'f{f:03d}.npz'
        if path.exists():
            continue
        labels = {}
        for y in range(0,world.size,16):
            sl = xy[y*world.size:(y+16)*world.size]
            result = two_layers(*intersect(world,f,sl,grid))
            for k,v in result.items():
                labels.setdefault(k,[]).append(v)
        labels = {k: np.concatenate(v).reshape(world.size,world.size,*v[0].shape[1:])
                  for k,v in labels.items()}
        with np.load(base/'geometry'/f'f{f:03d}-center.npz') as center:
            center_depth, center_valid = center['depth_axial_m'], center['center_visible']
        assert np.all(labels['depth2'][labels['valid_dual']] > labels['depth1'][labels['valid_dual']])
        assert np.isfinite(labels['normal']).all() and labels['valid'].any()
        save_npz(path, **labels, center_depth=center_depth.astype(np.float32), center_valid=center_valid,
                 subpixel_grid=np.int16(grid), K=world.K, camera_c2w=world.cameras[f-1])
        print('B_LABEL', name, f, grid, int(labels['valid_dual'].sum()), flush=True)
    for f in TRAIN_FRAMES:
        path = dest/'pinhole-subpixels'/f'f{f:03d}.npz'
        with np.load(path) as item:
            rows.append(dict(frame=f,path=str(path.relative_to(ROOT)),valid=int(item['valid'].sum()),
                             dual=int(item['valid_dual'].sum()),alpha=int(item['valid_alpha'].sum()),
                             normal=int(item['valid_normal'].sum())))
    lens = dest/'K16-aperture'
    lens.mkdir(exist_ok=True)
    source = base/'geometry'
    for p in sorted(source.glob('*-K16.npz')):
        link = lens/p.name
        if not link.exists():
            link.symlink_to(p.resolve())
    audit = dict(scene=name,grid=grid,rows=rows,compared=checks,seconds=time.monotonic()-start,
                 K16_files=len(list(lens.glob('*.npz'))),
                 label_definitions=dict(subpixel='primary pinhole jittered first hits, front mode fraction, two supported disjoint depth peaks, geometric face normals',
                                        K16='existing mesh-derived aperture first hits, separate aperture visibility/recoverability masks'))
    save_json(done,audit)
    return audit


def thin(name):
    from route3_blender_b2_geometry import World
    import cv2
    base = ROOT/'outputs/route3-g3-r14-2026-11-17-01/truth'/name
    world = World(base/'scene-geometry.npz', 512)
    recorded = json.loads((base/'quality.json').read_text())['surface_checks']
    assets = json.loads((base/'scene.json').read_text())['objects']
    foreground = {int(x['id']):x['name'] for x in assets if x['id'] in world.fg}
    rows = []
    for f in [1,30,60]:
        center = world.center(f,1)
        for oid,label in foreground.items():
            visible = center['object_id']==oid
            # Conservative candidate neighborhood via projected mesh bounds,
            # so subpixel threads with no center sample also enter the comparison.
            index = list(world.ids).index(oid)
            A=world.transforms[f-1,index]
            points=world.d[f'object_{oid}_vertices']@A[:3,:3].T+A[:3,3]
            C=world.cameras[f-1]
            cam=(points-C[:3,3])@C[:3,:3]
            if not (cam[:,2]>0).all():
                continue
            uv=cam@world.K.T
            uv=uv[:,:2]/uv[:,2:]
            x,y=np.floor(uv.min(0)-2).astype(int).clip(0,511)
            x1,y1=np.ceil(uv.max(0)+2).astype(int).clip(0,511)
            mask=np.zeros_like(visible)
            mask[y:y1+1,x:x1+1]=True
            mask &= ~cv2.erode(visible.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
            iy,ix=np.nonzero(mask)
            if not len(ix):
                continue
            pick=np.linspace(0,len(ix)-1,min(4096,len(ix)),dtype=int)
            xy=np.c_[ix[pick],iy[pick]]
            for shift in [(0,0),(.5,0)]:
                v=[(intersect(world,f,xy,g,shift)[2]==oid).mean(-1) for g in (4,8,16)]
                edge=(v[2]>0)&(v[2]<1)
                phase=next((r for r in recorded['alias'] if r['object_name']==label),{})
                rows.append(dict(scene=name,frame=f,object_id=oid,object=label,shift=shift,contour_pixels=int(edge.sum()),
                    pure_center_interior=int(cv2.erode(visible.astype(np.uint8),np.ones((3,3),np.uint8)).sum()),
                    interior_phase_RMSE=phase.get('excess_RMSE'),phase_sample_frame=30,
                    contour_4_mae=float(abs(v[0]-v[2])[edge].mean()) if edge.any() else None,
                    contour_8_mae=float(abs(v[1]-v[2])[edge].mean()) if edge.any() else None,
                    contour_8_pass=bool(abs(v[1]-v[2])[edge].mean()<=SPEC['thin_max_mean_coverage_error']) if edge.any() else None,
                    phase_check_applicable=not phase.get('insufficient_interior',True) if 'excess_RMSE' not in phase else True,
                    texture_supersample_reference='independent auxiliary comparison under B/thin-2x'))
    return rows


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--workers',type=int,default=5)
    p.add_argument('--thin',action='store_true')
    args=p.parse_args()
    split=json.loads((PREVIOUS/'split.json').read_text())
    OUT.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as workers:
        if args.thin:
            names=sorted(x.name for x in (ROOT/'outputs/route3-g3-r14-2026-11-17-01/truth').glob('*-thin'))
            records=[v for rows in workers.map(thin,names) for v in rows]
            save_json(OUT/'B/thin-check.json',dict(rows=records,diagnostic_only=True,policy=SPEC))
        else:
            groups=list(workers.map(run_scene,split['train']+split['heldout']))
            save_json(OUT/'B/manifest.json',dict(split=split,scenes=groups,policy=SPEC))


if __name__=='__main__':
    main()
