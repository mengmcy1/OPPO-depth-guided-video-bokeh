"""Build observed 256/5 Query contexts and separate K16 placement targets."""
import argparse
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
import shutil
import time

import cv2
import numpy as np
import torch

import route3_ovg_data as ovg
from route3_ovg_card import lens_offsets
from route3_v2data_B_batch import GIB, ROOT, sha256, write_json
from route3_v2data_B_inputs import library_unit_paths, observation
from route3_v2data_B_plan import load
from route3_v2data_c_build import OUT as C_ROOT, window_inventory
from route3_v2data_c_trace import trace_aperture
from route3_s2a_candidates import local_placement
from route3_s2a_data import FIVE, query_input_512, save_npz
from route3_v2data_stage1_evaluate import load_paired_checkpoint, view_from_observation
from src.dual_layer import depth_band
from src.footprint_order import footprint_scale

SIZE = 256
C_FRAMES = (0, 2, 4, 6, 7)
FSTOP = 1.4
# Keep the native 256/5 sample and the 1/512 camera convention consistent.
GRID = np.moveaxis(np.indices((SIZE, SIZE)), 0, -1)[..., ::-1].copy()


def inventory():
    from route3_v2data_loader import registered_b

    b, excluded = registered_b()
    jobs = [("B", "train", path) for path in sorted(b.values())]
    jobs += [("C", "train", path) for path in sorted({Path(p) for _, p, _ in window_inventory("train")})]
    jobs += [("B", "dev", path) for path in library_unit_paths("dev")]
    dev_c = sorted({Path(p) for _, p, _ in window_inventory("dev")})
    jobs += [("C", "dev", dev_c[i]) for i in np.linspace(0, len(dev_c)-1, 10, dtype=int)]
    return jobs, excluded


def project(camera, K):
    uvz = camera @ K.T
    return uvz[..., :2] / np.maximum(uvz[..., 2:3], 1e-6)


def camera_points(xy, depth, K):
    rays = np.concatenate((xy, np.ones((*xy.shape[:-1], 1), np.float32)), -1)
    return (rays @ np.linalg.inv(K).T) * depth[..., None]


def world_points(camera, pose):
    return camera @ pose[:3, :3].T + pose[:3, 3]


def pixels(uv, size):
    finite = np.isfinite(uv).all(-1)
    safe = np.where(finite[..., None], uv, 0)
    inside = finite & ((safe >= 0) & (safe < size)).all(-1)
    x, y = np.floor(safe).clip(0, size-1).astype(np.int32).T
    return y, x, inside


def read_fields(path, *fields):
    with np.load(path, allow_pickle=False) as values:
        return [values[k] for k in fields]


@torch.inference_mode()
def predict_context(net, tier, path):
    is_b = tier == "B"
    frames = FIVE if is_b else C_FRAMES
    native_frames = ovg.TRAIN_FRAMES if is_b else range(8)
    index = [native_frames.index(f) for f in frames]
    used = sorted(set(index).union(*(ovg.neighbors(i, len(native_frames)) for i in index)))
    if is_b:
        observed = {i: observation(path, native_frames[i], aligned=True) for i in used}
    else:
        observed = {i: ovg.gc_input(str(path), i) for i in used}
    views = []
    chunk = 1 if is_b else 3
    for start in range(0, len(index), chunk):
        selected = index[start:start+chunk]
        rows = [ovg.input_record(observed[i], [observed[j] for j in ovg.neighbors(i, len(native_frames))])
                for i in selected]
        data = {k: torch.as_tensor(np.stack([row[k] for row in rows]), device="cuda") for k in rows[0]}
        pred = net(data, protect_residual=True, render_heads=False)
        for n, i in enumerate(selected):
            raw = observed[i]
            rgb, K = raw["rgb"], raw["K"].copy()
            depth = pred["depth1"][n].cpu().numpy()
            confidence = pred["confidence"][n].cpu().numpy()
            if is_b:
                rgb = cv2.resize(rgb, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
                depth, confidence = depth[1::2, 1::2], confidence[1::2, 1::2]
                K[:2] *= .5
            view = view_from_observation(dict(rgb=rgb, K=K, pose=raw["pose"],
                                              depth=depth[None], confidence=confidence[None]), native_frames[i])
            views.append(view)
    return views


@torch.inference_mode()
def c_ray_sets(recipe, views):
    def device(value):
        return torch.as_tensor(np.array(value, np.float32), device="cuda")

    rect = device(recipe["rectangles"])
    poses = device([view["c2w"][:3, 3] for view in views])
    shift = torch.zeros(len(views), device="cuda")
    lens = device(lens_offsets())
    return [
        [trace_aperture(rect, poses, shift, lens, focus, slot, recipe["perforated"])
         for focus in recipe["focus_m"]]
        for slot in range(len(views))
    ]


def center_truth(tier, path, views, recipe=None):
    truth = []
    is_b = tier == "B"
    for view in views:
        f = view["frame"]
        if is_b:
            depth, ids, K = read_fields(path/"labels"/f"f{f:03d}-pinhole.npz", "depth_axial_m", "object_id", "K")
            peeled_depth, peeled_ids = read_fields(path/"labels"/f"f{f:03d}-peeled.npz", "depth_axial_m", "object_id")
        else:
            ids, peeled_depth, peeled_ids = read_fields(path.parent/"labels"/f"f{f:03d}.npz",
                                                        "center_surface_id", "peeled_depth", "peeled_surface_id")
            z = np.asarray(recipe["rectangles"], np.float32)[:, 0] - view["c2w"][2, 3]
            depth = np.where(ids > 0, z[np.maximum(ids, 1)-1], 0)
            K = view["K"]
        extent = depth.shape[0]
        _, step = divmod(extent, SIZE)
        if step or extent not in (SIZE, 2*SIZE):
            raise ValueError("Unsupported pinhole label resolution")
        scale = extent // SIZE
        xy = (GRID*scale + scale-.5).astype(np.float32)
        xyz = camera_points(xy, depth[scale-1::scale, scale-1::scale].astype(np.float32), K)
        truth.append(dict(depth=depth, ids=ids, K=K, c2w=view["c2w"],
                          xyz=world_points(xyz, view["c2w"]),
                          peeled_depth=peeled_depth, peeled_ids=peeled_ids))
    return truth


@lru_cache(maxsize=1)
def c_texture_paths():
    return {r["key"]: C_ROOT/r["path"] for r in load(C_ROOT/"assets.json")["assets"]}


@lru_cache(maxsize=40)
def c_texture(key):
    image = cv2.imread(str(c_texture_paths()[key]), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"Missing C label texture {key}")
    return ovg.srgb_to_linear(image[..., ::-1].astype(np.float32)/255)


def c_color(recipe, xyz, ids):
    colors = np.zeros((len(ids), 3), np.float32)
    for j, key in enumerate(recipe["assets"]):
        which = np.flatnonzero(ids == j+1)
        scale, tx, ty, gain = recipe["material_rows"][j]
        p = (xyz[which, :2]*scale+[tx, ty]) % 1 * 255
        x, y = np.minimum(np.floor(p).astype(int), 254).T
        u, v = (p-np.stack((x, y), -1)).T[..., None]
        tex = c_texture(key)
        colors[which] = gain*((tex[y, x]*(1-u)+tex[y, x+1]*u)*(1-v)
                              +(tex[y+1, x]*(1-u)+tex[y+1, x+1]*u)*v)
    return colors.clip(0, 1)


def placement(packed, current, truth, foreground, peeled_rgb, recipe):
    query, slot = np.nonzero(packed["candidate_valid"])
    src = packed["source_slots"][query, slot]
    xy = packed["source_xy"][query, slot]
    world = np.empty((len(query), 3), np.float32)
    ids = np.empty(len(query), np.int16)
    for j, source in enumerate(truth):
        select = src == j
        x, y = xy[select].T
        world[select] = source["xyz"][y, x]
        scale = source["depth"].shape[0]//SIZE
        ids[select] = source["ids"][y*scale+scale-1, x*scale+scale-1]
    g = truth[current]
    scale = g["depth"].shape[0]//SIZE
    camera = (world-g["c2w"][:3, 3]) @ g["c2w"][:3, :3]
    uv = project(camera, g["K"])
    extent = len(g["depth"])
    y, x, inside = pixels(uv, extent)
    coplanar = abs(g["peeled_depth"][y, x].astype(np.float32)-camera[:, 2])
    footprint = np.maximum(camera[:, 2]/packed["target_K"][0, 0], 1e-4)
    hidden = (inside & (ids > 0) & (g["ids"][y, x] > 0) &
              (camera[:, 2] > g["depth"][y, x].astype(np.float32)+1e-4))
    hidden &= (g["peeled_ids"][y, x] == ids) & (coplanar <= 1.5*footprint+1e-4)
    visible_depth = g["depth"][scale-1::scale, scale-1::scale].astype(np.float32)
    visible_ids = g["ids"][scale-1::scale, scale-1::scale]
    background = (visible_ids > 0) & ~np.isin(visible_ids, foreground)
    a, b, _, support = local_placement(visible_depth, packed["target_depth"], background,
                                       radius=12, return_valid=True)
    ty, tx = y//scale, x//scale
    inverse = a[ty, tx]/np.maximum(camera[:, 2], 1e-6) + b[ty, tx]
    safe = hidden & support[ty, tx] & np.isfinite(inverse) & (inverse > 1e-4)
    depth = np.where(safe, np.maximum(1/np.maximum(inverse, 1e-4), 1.03*packed["target_depth"][ty, tx]), 0)
    place = (camera*(depth/np.maximum(camera[:, 2], 1e-6))[:, None]).astype(np.float32)
    color = peeled_rgb[y, x].astype(np.float32) if peeled_rgb is not None else c_color(recipe, world, ids)
    color = np.where(hidden[:, None], color, 0)
    return dict(index=(query, slot), source_slot=src, source_xy=xy, ids=ids, world=world,
                camera=camera, support=safe, hidden=hidden, footprint=footprint,
                target_camera=place, target_depth=depth.astype(np.float32), color=color)


def recoverable_b(hidden, world, ids, current, truth):
    active = np.flatnonzero(hidden)
    seen = np.zeros(len(active), bool)
    xyz, identity = world[active], ids[active]
    for j, src in enumerate(truth):
        if j == current:
            continue
        cam = (xyz-src["c2w"][:3, 3]) @ src["c2w"][:3, :3]
        y, x, inside = pixels(project(cam, src["K"]), len(src["depth"]))
        atol = 3*np.maximum(cam[:, 2], 0)/src["K"][0, 0]+.002
        seen |= inside & (cam[:, 2] > 0) & (src["ids"][y, x] == identity) & (
            abs(src["depth"][y, x].astype(np.float32)-cam[:, 2]) <= atol)
    result = np.zeros(len(hidden), bool)
    result[active] = seen
    return result


def label_case(packed, target, current, truth, focus, rays, *, is_b):
    g = truth[current]
    extent = len(g["depth"])
    scale = extent//SIZE
    K, pose = g["K"], g["c2w"]
    active = np.flatnonzero(target["hidden"])
    xyz, identity = target["world"][active], target["ids"][active]
    angles = rays["aperture_xy"]
    bitfield = np.zeros(len(target["ids"]), np.uint16)
    hidden_maps, recover_maps = [], []
    grid = GRID.reshape(-1, 2)*scale+scale-.5
    for k, xy in enumerate(angles):
        shifted = np.r_[xy, 0.]
        intr = K.copy()
        intr[:2, 2] += np.diag(K)[:2]*xy/focus
        target_uv = project(target["camera"][active]-shifted, intr)
        y, x, inside = pixels(target_uv, extent)
        hidden_map = rays["hidden"][k]
        ids = rays["ids"][k, y, x]
        if is_b:
            known = rays["known"][k, y, x]
            hit = camera_points(np.stack((x+.5, y+.5), -1), rays["depth"][k, y, x], intr)+shifted
            hits = world_points(hit, pose)
            verified = known & inside
            h = hidden_map[scale-1::scale, scale-1::scale].ravel()
            axial = rays["depth"][k, scale-1::scale, scale-1::scale].ravel()
            points = world_points(camera_points(grid, axial, intr)+shifted, pose)
            r = recoverable_b(h, points, rays["ids"][k, scale-1::scale, scale-1::scale].ravel(),
                              current, truth).reshape(SIZE, SIZE)
        else:
            hits = rays["world"][k, y, x].astype(np.float32)
            verified = inside
            r = rays["recoverable"][k]
        same = (verified & (ids == identity) & hidden_map[y, x] &
                (np.linalg.norm(hits-xyz, axis=-1) <= 1.5*target["footprint"][active]+1e-4))
        for j, src in enumerate(truth):
            matching = np.flatnonzero(same & (target["source_slot"][active] == j))
            camera = (hits[matching]-src["c2w"][:3, 3]) @ src["c2w"][:3, :3]
            # The original source-pixel witness tolerance is measured on the 256 grid.
            same[matching] &= (abs(project(camera, src["K"])/scale
                                   -(target["source_xy"][active[matching]]+.5)) <= .75).all(-1)
        bitfield[active] |= same.astype(np.uint16) << k
        hidden_maps.append(hidden_map[scale-1::scale, scale-1::scale])
        recover_maps.append(r)
    matches = np.unpackbits(bitfield.view(np.uint8).reshape(-1, 2), axis=1).sum(1)/16
    support = target["support"].copy()
    correct = (matches > 0) & support
    camera = torch.from_numpy(target["target_camera"][correct])
    native_K = packed["target_K"]
    sigma = camera[:, 2]/native_K[0, 0]*.35
    scale_factor, kept = footprint_scale(camera, sigma[:, None].repeat(1, 2), sigma*.01,
                                         torch.from_numpy(packed["target_depth"]),
                                         torch.as_tensor(native_K, dtype=torch.float32), .15)
    support[correct] = kept.numpy()
    factors = np.ones(len(matches), np.float16)
    factors[correct] = scale_factor.numpy().astype(np.float16)
    hidden, recovered = np.stack(hidden_maps), np.stack(recover_maps)
    recover = recovered.any(0)
    def pack(array, dtype=None):
        result = np.zeros((*packed["candidate_valid"].shape, *array.shape[1:]), dtype=dtype or array.dtype)
        result[target["index"]] = array
        return result
    pick = packed["query_pixel"]
    fields = dict(
        content_hit_fraction=pack(matches, np.float16), content_hit_bits=pack(bitfield),
        candidate_depth_target=pack(target["target_depth"], np.float16),
        candidate_camera_target=pack(target["target_camera"], np.float32),
        candidate_RGB_target=pack(target["color"], np.float16),
        placement_support_v2=pack(support), appearance_valid=pack(target["hidden"]),
        footprint_tangent_factor=pack(factors), hidden_any_at_query=hidden.any(0).ravel()[pick],
        recoverable_at_query=recover.ravel()[pick], recoverable_mask=recover,
        residual_pixels=hidden.any(0)&~recover, hidden_ray_mask=hidden, recoverable_ray_mask=recovered,
    )
    stats = dict(candidates=len(matches), true_background=int((matches > 0).sum()),
                 true_background_placement_supported=int(correct.sum()),
                 shrunk_true_background=int((kept&(scale_factor<1)).sum()),
                 dropped_true_background=int((~kept).sum()))
    return fields, stats


def prepare_scene(job, dest, views, recipe, c_rays, geometry_sha):
    tier, split, path = job
    is_b = tier == "B"
    name = path.name if is_b else path.parent.name
    out = dest/"learning-data"/name/"256-5"
    out.mkdir(parents=True, exist_ok=True)
    context = out/"context.npz"
    save_npz(context, dict(images=np.stack([v["rgb_linear"] for v in views]),
                          c2w=np.stack([v["c2w"] for v in views]), K=views[0]["K"]))
    truth = center_truth(tier, path, views, recipe)
    if is_b:
        foreground, = read_fields(path/"labels/scene-geometry.npz", "foreground_object_ids")
        cases = {row["frame"]: [x["focus_m"] for x in row["K16"]]
                 for row in load(path/"labels/scene-supervision.json")["rows"]}
    else:
        foreground = np.arange(1, recipe["front_layers"]+1)
        cases = {v["frame"]: recipe["focus_m"] for v in views}
    rows, counters = [], Counter()
    for current, view in enumerate(views):
        f = view["frame"]
        focuses = cases[f]
        bands = [depth_band(view["depth"], view["K"][0, 0], d, .05/(2*FSTOP))[0] for d in focuses]
        packed, _ = query_input_512(views, view, np.logical_or.reduce(bands), 8, 32768)
        if not len(packed["query_pixel"]):
            continue
        # The original 256/5 feature contract expresses coordinates in the
        # common 512 grid, despite candidate sampling on the 256 grid.
        packed["features"][..., 12:14] *= 2
        counters["candidate_pool_entries"] += int(packed["candidate_valid"].sum())
        inputs = out/f"f{f:03d}-inputs.npz"
        save_npz(inputs, packed)
        peeled_rgb = None
        if is_b:
            peeled_rgb, = read_fields(path/"labels"/f"f{f:03d}-peeled-color.npz", "rgb_linear")
        target = placement(packed, current, truth, foreground, peeled_rgb, recipe)
        for k, focus in enumerate(focuses):
            count = int(bands[k].ravel()[packed["query_pixel"]].sum())
            if not count:
                continue
            if is_b:
                keys = ("depth_axial_m", "object_id", "known", "hidden", "aperture_xy")
                z, ids, known, hidden, xy = read_fields(path/"labels"/f"f{f:03d}-focus{k}-K16.npz", *keys)
                rays = dict(depth=z, ids=ids, known=known, hidden=hidden, aperture_xy=xy)
            else:
                raw = c_rays[current][k]
                rays = dict(raw, ids=raw["surface_id"], aperture_xy=lens_offsets()[:, :2])
            labels, stats = label_case(packed, target, current, truth, focus, rays, is_b=is_b)
            counters.update(stats)
            tag = f"f{f:03d}-focus{k}"
            label = out/(tag+"-labels.npz")
            save_npz(label, labels)
            rows.append(dict(frame=f, case=dict(id=tag, focus_m=focus, nominal_focus_m=focus,
                                                 focus_index=k, fstop=FSTOP),
                             input=str(inputs), target=str(label), active_queries=count, **stats))
    if not rows:
        raise ValueError("frozen predictions provide no active Query band")
    value = dict(scene=name, tier=tier, split=split, size=SIZE, frames=[v["frame"] for v in views],
                 context=str(context), rows=rows, stats=dict(counters), geometry_sha256=geometry_sha)
    write_json(out/"manifest.json", value)
    return dict(scene=name, tier=tier, split=split, path=str(out/"manifest.json"),
                cases=len(rows), stats=dict(counters))


def prepare(dest, checkpoint, *, limit=None):
    dest.mkdir(parents=True, exist_ok=True)
    jobs, excluded = inventory()
    if limit:
        jobs = [next(j for j in jobs if j[:2] == (t, "train")) for t in ("B", "C")]
    geometry_sha = sha256(checkpoint)
    settings = dict(geometry_sha256=geometry_sha, jobs=[(tier, split, str(path)) for tier, split, path in jobs],
                    B_registered_exclusions=excluded)
    plan = dest/"inventory.json"
    previous = load(plan)
    if previous and previous != {**settings, "jobs": [list(row) for row in settings["jobs"]]}:
        raise ValueError("Query input inventory or frozen geometry changed on restart")
    write_json(plan, settings)
    net = load_paired_checkpoint(checkpoint, "ema").eval().requires_grad_(False)
    start = time.monotonic()
    state_path = dest/"prepared.json"
    state = load(state_path) or dict(accepted={}, skipped={})
    queue = deque()
    def finish_one():
        name, tier, split, future = queue.popleft()
        try:
            state["accepted"][name] = future.result()
        except (OSError, ValueError, KeyError) as error:
            state["skipped"][name] = dict(tier=tier, split=split, reason=str(error))
        new = len(state["accepted"])+len(state["skipped"])
        if new % 10 == 0 or limit:
            write_json(state_path, state)
            print("QUERY_PREPARE",new,len(jobs),len(state["skipped"]),round(time.monotonic()-start, 1),flush=True)
    with ThreadPoolExecutor(max_workers=5) as pool:
        for job in jobs:
            tier, split, path = job
            name = path.name if tier == "B" else path.parent.name
            if name in state["accepted"] or name in state["skipped"]:
                continue
            if len(queue) >= 6:
                finish_one()
            if shutil.disk_usage(dest).free < 61*GIB:
                raise RuntimeError("Query preparation cannot preserve the 60 GiB disk reserve")
            try:
                views = predict_context(net, tier, path)
                recipe = load(ROOT/load(path)["geometry_label_recipe"]) if tier == "C" else None
                rays = c_ray_sets(recipe, views) if recipe is not None else None
            except (OSError, ValueError, KeyError) as error:
                state["skipped"][name] = dict(tier=tier, split=split, reason=str(error))
                continue
            queue.append((name, tier, split, pool.submit(prepare_scene, job, dest, views, recipe, rays, geometry_sha)))
        while queue:
            finish_one()
    state.update(seconds=state.get("seconds", 0)+time.monotonic()-start, geometry_sha256=geometry_sha,
                 peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20)
    write_json(state_path, state)
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    torch.cuda.set_per_process_memory_fraction(10*GIB/torch.cuda.get_device_properties(0).total_memory, 0)
    prepare(args.output.resolve(), args.checkpoint.resolve(), limit=args.pilot)
