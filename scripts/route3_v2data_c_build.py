"""Plan, render, precompute DepthSplat, and register route 3 tier C."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import time
from zipfile import ZIP_DEFLATED, ZipFile

import cv2
import numpy as np
import torch

from route3_depthsplat_depth import network
from route3_ovg_1b_cards import camera_rays, sample_disk
from route3_ovg_card import lens_offsets
from route3_ovg_r2_cards import cast
from route3_ovg_r2_priors import MODEL, stack_depth, support_map
from route3_ovg_subpixel import ROOT, save_json
from route3_v2data_c_assets import GIB, OUT, canonical_split
from route3_v2data_c_trace import FOCAL, trace_aperture, trace_optical, trace_pinhole

RECORD = ROOT / "03_实验记录/records/route3-v2data-C-2026-11-20.json"
LOCK = ROOT.parent / "route3-cache/gpu-data.lock"
CHECKPOINT = ROOT / "models/DepthSplat" / MODEL
GENERATOR = "route3-v2data-c/1.0"
SEED = 20261120
SIZE = 256
MAIN = {"train": 4500, "dev": 500, "test": 1}
MOTION = {"train": 90, "dev": 10}
PILOT = 300
PILOT_UPPER_BYTES = 80*2**20
BATCH = 25
K = np.array([[FOCAL, 0, 128], [0, FOCAL, 128], [0, 0, 1]], np.float32)
WIDTHS = ("thin", "medium", "broad")
WIDTH_RANGE = dict(thin=(.04,.10), medium=(.16,.31), broad=(.38,.7))
BASELINES = ("small", "medium", "large")
DEPTH_INPUT = "observed sRGB and camera K/c2w"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def window_inventory(split):
    """Return stage 1 sample keys for a single C pool."""
    if split not in MAIN:
        raise ValueError(f"unknown C pool {split}")
    units = read(OUT/"manifest.json")["units"]
    return [("C", str(OUT/unit["path"]), index) for unit in units
            if unit["split"]==split and unit["kind"]=="static" for index in range(unit["frames"])]


def optical_specs(window_manifest, frame):
    manifest = Path(window_manifest)
    return [dict(tiers="C", frame=frame, focus=r["focus_m"], fstop=r["fstop"],
                 label=ROOT/r["label"], truth=ROOT/r["reference"], manifest=str(manifest), index=frame)
            for r in read(manifest)["K16_aperture"] if r["frame"]==frame]


def scenes(assets):
    group = {split: [r["key"] for r in assets if r["split"] == split] for split in MAIN}
    if min(map(len, group.values())) < 5:
        raise ValueError(f"C needs five licensed textures in each pool, got {[(k,len(v)) for k,v in group.items()]}")
    cases = []
    profiles = {}
    for width_name, (lower, upper) in WIDTH_RANGE.items():
        width_bin = (upper-lower)/128
        for index in range(128):
            key = f"procedural-card:{width_name}-{index:03d}"
            split = canonical_split(key)
            entry = dict(key=key, source="procedural-card", asset_id=key.split(":",1)[1],
                         author="route 3 project", license="project-owned", terms_version=GENERATOR,
                         download_channel="local deterministic generator", split=split,
                         occluder_width_interval_m=[lower+index*width_bin, lower+(index+1)*width_bin])
            profiles.setdefault((width_name,split), []).append(entry)
    for kind, counts in (("static", MAIN), ("stage4-motion", MOTION)):
        for split, count in counts.items():
            for index in range(count):
                seed = SEED + len(cases)
                rng = np.random.default_rng(seed)
                front_count = index%3+1
                back_count = index//3%2+1
                width_bucket = WIDTHS[index//6%3]
                motion_bucket = BASELINES[index//18%3]
                candidates = profiles[width_bucket,split]
                profile = candidates[rng.integers(len(candidates))]
                width = rng.uniform(*profile["occluder_width_interval_m"])
                baseline = rng.uniform(*dict(small=(.025,.065), medium=(.12,.25), large=(.36,.65))[motion_bucket])
                negative = index//54%4 == 0
                holes = index//27%2 == 1
                frames = 13 if split == "test" else 8
                z = float(rng.uniform(1.15,1.65))
                x, y = float(rng.uniform(-.3,-.1)), float(rng.uniform(-.42,-.25))
                rect = [[z,x,x+width,y,y+rng.uniform(.44,.85)]]
                for j in range(front_count-1):
                    x = rng.uniform(-.14,.20) + .07*j
                    rect.append([z+.45+.52*j, x, x+rng.uniform(.24,.65), -.5+.11*j, .55])
                if back_count == 2:
                    x = rng.uniform(-1.3,-.35)
                    rect.append([rng.uniform(3.2,4.4), x, x+rng.uniform(.6,1.2), -.8, .85])
                rear = float(rng.uniform(4.8,7.6))
                rect.append([rear,-30,30,-30,30])
                t = np.linspace(-1, 1, frames)
                dir_x, dir_y = (.009/baseline, 1.) if negative else (1., .12)
                origins = np.stack((baseline*t*dir_x/2, baseline*t*dir_y/2+.018*np.sin(np.pi*t),
                                    .024*np.sin(1.5*t)), -1)
                moving = kind == "stage4-motion"
                shifts = t*rng.uniform(.08,.24) if moving else np.zeros_like(t)
                materials = rng.uniform([.6,-.55,-.55,.7], [2.1,.55,.55,1.05], (len(rect),4))
                if index%5 == 0:
                    materials[rng.integers(len(rect)),3] = rng.uniform(1.4,3.6)
                cases.append(dict(id=f"card-v2-{kind}-{split}-{index:05d}", seed=seed, split=split, kind=kind,
                                  assets=rng.choice(group[split], len(rect), replace=False).tolist(),
                                  frame_count=frames, front_layers=front_count, back_layers=back_count,
                                  parameter_profile=profile["key"],
                                  width_bucket=width_bucket, motion_bucket=motion_bucket, occluder_width_m=width,
                                  camera_span_m=float(np.linalg.norm(np.ptp(origins,axis=0))),
                                  negative_design=negative, perforated=holes, camera_origins=origins.tolist(),
                                  foreground_translation_x=shifts.tolist(), rectangles=rect,
                                  material_rows=materials.tolist(), focus_m=[z,rear]))
    return cases, [entry for group in profiles.values() for entry in group]


@contextmanager
def gpu_lock():
    with LOCK.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def archive(path, fields):
    with ZipFile(path, "w", compression=ZIP_DEFLATED, compresslevel=1) as out:
        for key, array in fields.items():
            with out.open(key+".npy", "w", force_zip64=True) as stream:
                np.lib.format.write_array(stream, np.ascontiguousarray(array), allow_pickle=False)


def rgb8(image):
    srgb = torch.where(image<=.0031308, 12.92*image, 1.055*image.pow(1/2.4)-.055)
    return (srgb*255).round().clamp(0,255).to(torch.uint8).cpu().numpy()


def from_numpy(v, dtype=np.float32):
    return torch.as_tensor(np.asarray(v, dtype=dtype), device="cuda")


def geometry_from_hits(rect, poses, ids, colors, nearest):
    n = ids.shape[-1]
    depth_grid = rect[:, 0][None, None, None, :] - poses[:, None, None, 2:3]
    depth_grid = depth_grid.expand(*ids.shape[:-1], len(rect))
    depth = depth_grid.gather(-1, ids.long())
    first_id = ids.amin(-1)
    front = (ids == first_id[..., None])
    back = ~front
    counts = front.sum(-1)
    depth1 = depth.amin(-1)
    nearest_back = torch.where(back, depth, torch.inf).amin(-1)
    farthest_back = depth.amax(-1)
    alpha_valid = (counts==n) | ((farthest_back-nearest_back) <= .03*nearest_back)
    coverage = counts / n
    depth2 = torch.where(counts<n, (n-counts)/(back/depth).sum(-1), 0.)
    dual = alpha_valid & (counts>=2) & (counts<=n-2)
    boundary = (counts < n) & (farthest_back > 1.03*depth1)
    norm = torch.zeros((*depth1.shape, 3), dtype=torch.float16, device="cuda")
    norm[..., 2] = -1
    quantity = colors.clamp(0, 1)
    light = quantity.mean(-2)
    hdr = colors.mean(-2)
    foreground = (quantity * front[..., None]).sum(-2)/counts[..., None]
    background = (quantity * back[..., None]).sum(-2)/(n-counts).clamp_min(1)[..., None]
    saturated = ((colors>=1).any(-1)&(front|back)).any(-1) & boundary
    valid_rgb = alpha_valid & boundary & ~saturated
    recomposition = coverage[...,None]*foreground + (1-coverage[...,None])*background
    error = float((recomposition-light).abs().max().item())
    if error > 3e-7 or not torch.all(depth2[dual] > depth1[dual]):
        raise ValueError(f"invalid layering or pinhole composition {error}")
    visible = nearest[..., 0].long()
    second = nearest[..., 1].long()
    hidden = second>=0
    peeled = torch.where(hidden, depth_grid.gather(-1, second.clamp_min(0)[...,None])[...,0], 0)
    shape = depth1.shape
    valid = torch.ones(shape, dtype=torch.bool, device="cuda")
    def float16(v):
        return v.half().cpu().numpy()
    fields = dict(depth1=float16(depth1), depth2=float16(depth2), alpha=float16(coverage), normal=float16(norm),
                  valid=valid.cpu().numpy(), valid_alpha=alpha_valid.cpu().numpy(), valid_dual=dual.cpu().numpy(),
                  valid_normal=valid.cpu().numpy(), depth_boundary=boundary.cpu().numpy(),
                  first_hit_object=(first_id+1).byte().cpu().numpy(),
                  foreground=float16(foreground), background=float16(background), full=float16(light),
                  saturation_invalid=saturated.cpu().numpy(), color_valid=valid_rgb.cpu().numpy(),
                  comp_valid=valid_rgb.cpu().numpy(), peeled_depth=float16(peeled), peeled_valid=hidden.cpu().numpy(),
                  peeled_surface_id=(second+1).byte().cpu().numpy(),
                  center_surface_id=(visible+1).byte().cpu().numpy())
    edges = boundary.sum().item()
    stats = dict(pinhole_boundary_pixels=int(edges), usable_color_pixels=int(valid_rgb.sum().item()),
                 saturated_boundary_pixels=int(saturated.sum().item()), max_composition_error=error,
                 depth_gap_counts=[int((dual & (depth2-depth1<.5)).sum().item()),
                                   int((dual & (depth2-depth1>=.5) & (depth2-depth1<1.5)).sum().item()),
                                   int((dual & (depth2-depth1>=1.5)).sum().item())])
    return rgb8(light), float16(hdr), fields, stats


def cpu_optical_check(rect, center, materials, images, optics, perforated, shift):
    """Compare fixed independent rays with the original CPU intersection for one pilot clip."""
    shifted = rect.copy()
    shifted[0,1:3] += shift
    holes = []
    if perforated:
        _, l, r, b, t = shifted[0]
        for i in range(3):
            for j in range(4):
                u, v = l+(i+.5)*(r-l)/3, b+(j+.5)*(t-b)/4
                dx, dy = .205*(r-l)/3, .23*(t-b)/4
                holes.append((0,u-dx,u+dx,v-dy,v+dy))
    yy, xx = np.meshgrid(np.arange(4, 256, 17), np.arange(5, 256, 17), indexing="ij")
    errors = []
    for focus, gpu_ldr, gpu_hdr, _ in optics:
        daily = np.zeros((*xx.shape,3), np.float64)
        hdr = np.zeros_like(daily)
        for dx in (.25,.75):
            for dy in (.25,.75):
                rays = camera_rays(K, 256, 256, dx, dy)[yy,xx]
                for lens in sample_disk(256,.05/2/1.4):
                    _, ids, points, _ = cast(rays*focus-lens, center+lens, shifted, holes)
                    color = np.empty_like(points)
                    for i, (scale, tx, ty, asset, gain) in enumerate(materials):
                        pick = ids==i+1
                        p = points[pick,:2]
                        if i==0:
                            p[:,0] -= shift
                        xy = ((p*scale+[tx,ty])%1)*255
                        x, y = np.minimum(np.floor(xy).astype(int),254).T
                        u, v = (xy-np.stack([x,y],-1)).T[...,None]
                        tex = images[int(asset)]
                        color[pick] = gain*((tex[y,x]*(1-u)+tex[y,x+1]*u)*(1-v)+
                                            (tex[y+1,x]*(1-u)+tex[y+1,x+1]*u)*v)
                    hdr += color
                    daily += color.clip(0,1)
        errors.append([float(np.max(np.abs(daily/1024-gpu_ldr[yy,xx]))),
                       float(np.max(np.abs(hdr/1024-gpu_hdr[yy,xx])))])
    if max(a for a,b in errors)>.0015 or max(b for a,b in errors)>.003:
        raise ValueError(f"independent CPU/GPU reference mismatch {errors}")
    return errors


def load_materials(assets):
    arrays = []
    indices = {}
    for entry in assets:
        bgr = cv2.imread(str(OUT/entry["path"]))
        if bgr is None:
            raise OSError(OUT/entry["path"])
        rgb = bgr[..., ::-1].astype(np.float32)/255
        linear = np.where(rgb<=.04045, rgb/12.92, ((rgb+.055)/1.055)**2.4)
        indices[entry["key"]] = len(arrays)
        arrays.append(linear)
    cpu = np.stack(arrays)
    return cpu, from_numpy(cpu.transpose(0,3,1,2).copy()), indices


@torch.inference_mode()
def render(spec, textures, materials_indices, pupil_set, net):
    rect_cpu = np.asarray(spec["rectangles"], np.float32)
    centers = np.asarray(spec["camera_origins"], np.float32)
    shifts = np.asarray(spec["foreground_translation_x"], np.float32)
    mats = np.asarray([[*r[:3],materials_indices[k],r[3]]
                       for r,k in zip(spec["material_rows"],spec["assets"])],np.float32)
    rect, poses, shift, material = [from_numpy(x) for x in (rect_cpu, centers, shifts, mats)]
    n = len(centers)
    middle = n//2
    hole = spec["perforated"]
    ids, colors, nearest = trace_pinhole(rect, poses, shift, textures, material, hole)
    rgb, hdr, labels, stats = geometry_from_hits(rect, poses, ids, colors, nearest)
    del ids, colors, nearest
    optics = []
    for focus in spec["focus_m"]:
        ldr, optical_hdr = trace_optical(rect, textures, material, pupil_set[0], centers[middle],
                                        shifts[middle], focus, hole)
        aperture = trace_aperture(rect, poses, shift, pupil_set[1], focus, middle, hole)
        aperture["foreground_hit"] = aperture["surface_id"] <= spec["front_layers"]
        optics.append((focus, ldr, optical_hdr, aperture))
    intr = np.broadcast_to(K, (n,3,3)).copy()
    extr = np.tile(np.eye(4,dtype=np.float32),(n,1,1))
    extr[:,:3,3] = centers
    depths = stack_depth(net, rgb, intr, extr)
    return rgb, hdr, labels, stats, optics, intr, extr, depths, mats


def write_clip(spec, render_data, comparison=None):
    rgb, hdr, labels, stats, optics, intr, extr, depths, _ = render_data
    dest = OUT/"windows"/spec["split"]/spec["id"]
    dest.mkdir(parents=True, exist_ok=True)
    image_dir, label_dir, prior_dir, opt_dir = [dest/name for name in
                                                ("input-resized","labels","prior","K16-aperture")]
    for p in (image_dir, label_dir, prior_dir, opt_dir):
        p.mkdir(exist_ok=True)
    rows, rays = [], []
    for i, frame in enumerate(rgb):
        image = image_dir/f"f{i:03d}.png"
        if not cv2.imwrite(str(image), frame[..., ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 1]):
            raise OSError(image)
        label = label_dir/f"f{i:03d}.npz"
        archive(label, dict({key:value[i] for key,value in labels.items()}, hdr_rgb=hdr[i]))
        support = support_map(depths, intr, extr, i)
        archive(prior_dir/f"f{i:03d}.npz", dict(depth=depths[i].astype(np.float16), support=support,
                                               consistent=support>=2))
        rows.append(dict(frame=i, image=str(image.relative_to(ROOT)), label=str(label.relative_to(ROOT)),
                         K=intr[i].tolist(), c2w=extr[i].tolist()))
    for focus, ldr, hdr_optical, aperture in optics:
        stem = f'f{len(rgb)//2:02d}-N1.4-F{focus:g}'
        label = opt_dir/(stem+".npz")
        target = label_dir/(stem+"-reference.npz")
        archive(label, aperture)
        archive(target, dict(optical_rgb=ldr, optical_rgb_hdr=hdr_optical))
        rays.append(dict(frame=len(rgb)//2, focus_m=focus, fstop=1.4, label=str(label.relative_to(ROOT)),
                         reference=str(target.relative_to(ROOT)), hidden=int(aperture["hidden"].sum()),
                         recoverable=int(aperture["recoverable"].sum()), never_seen=int(aperture["never_seen"].sum()),
                         rays=aperture["hidden"].size))
    # The normal sample loader consumes only observed RGB, K/c2w, and these model forecasts.
    save_json(prior_dir/"summary.json", dict(checkpoint=MODEL, dtype="float16", frames=len(rgb),
                                           inputs=DEPTH_INPUT, GT_depth_input=False))
    recipe = label_dir/"recipe.json"
    save_json(recipe, spec)
    files = {str(p.relative_to(dest)): sha(p) for p in sorted(dest.rglob("*")) if p.is_file()
             and p != dest/"manifest.json"}
    info = dict(scene=spec["id"], generator=GENERATOR, tier="C", split=spec["split"], rows=rows,
                K16_aperture=rays, geometry_label_recipe=str(recipe.relative_to(ROOT)),
                static_scene=spec["kind"]=="static",
                supervision=stats, files_sha256=files, reference_cpu_audit=comparison)
    save_json(dest/"manifest.json", info)
    path = dest/"manifest.json"
    return dict(id=spec["id"], split=spec["split"], kind=spec["kind"], frames=len(rgb), assets=spec["assets"],
                bytes=sum(p.stat().st_size for p in dest.rglob("*") if p.is_file()), sha256=sha(path),
                path=str(path.relative_to(OUT)), supervision=stats,
                parameter_profile=spec.get("parameter_profile"),
                occluder_width_m=spec["occluder_width_m"], camera_span_m=spec["camera_span_m"],
                dimensions={k: spec[k] for k in ("front_layers","back_layers","width_bucket","motion_bucket",
                                                  "negative_design","perforated")},
                hidden=sum(r["hidden"] for r in rays), aperture_rays=sum(r["rays"] for r in rays),
                recoverable=sum(r["recoverable"] for r in rays), never_seen=sum(r["never_seen"] for r in rays))


def partial_report(plan, registry, errors, first, status, resource_use):
    assets = read(OUT/"assets.json")
    units = list(registry.values())
    output = dict(generator=GENERATOR, status=status, assets=assets["assets"], units=units,
                  procedural_profiles=plan.get("procedural_profiles",[]),
                  exclusions=assets["exclusions"]+list(errors.values()), attribution=assets["attribution"],
                  pending_credentials=[], plan_sha256=sha(OUT/"plan.json"), gpu_resources=resource_use,
                  sources=sorted({r["source"] for r in assets["assets"]}),
                  use_by_split=dict(train="training", dev="per-tier train/dev gap only", test="sealed holdout"))
    save_json(OUT/"manifest.json", output)
    by_group = {}
    for group, cases in (("static", [r for r in units if r["kind"]=="static"]),
                         ("stage4-motion", [r for r in units if r["kind"]!="static"])):
        by_group[group] = dict(units={k:sum(r["split"]==k for r in cases) for k in MAIN},
                               bytes=sum(r["bytes"] for r in cases), frames=sum(r["frames"] for r in cases),
                               bytes_per_unit=minmax(cases),
                               widths_m=quantiles(cases,"occluder_width_m"),
                               camera_spans_m=quantiles(cases,"camera_span_m"),
                               dimensions={k:dict(Counter(str(r["dimensions"][k]) for r in cases))
                                           for k in cases[0]["dimensions"]} if cases else {},
                               negative_clip_fraction=sum(r["never_seen"]>0 for r in cases)/max(1,len(cases)),
                               never_seen_of_hidden=sum(r["never_seen"] for r in cases)/max(1,sum(r["hidden"] for r in cases)),
                               never_seen_of_aperture=sum(r["never_seen"] for r in cases)/max(1,sum(r["aperture_rays"] for r in cases)))
    save_json(RECORD, dict(generator=GENERATOR, status=status, target_main=5000,
                          target_by_group={"static":MAIN,"stage4-motion":MOTION},
                          target_static_train_dev_shortfall=5000-sum(by_group["static"]["units"][k] for k in ("train","dev")),
                          groups=by_group, manifest=str((OUT/"manifest.json").relative_to(ROOT)),
                          manifest_sha256=sha(OUT/"manifest.json"), disk=first, resources=resource_use,
                          data_bytes=sum(r["bytes"] for r in units),
                          texture_bytes=sum(r["bytes"] for r in assets["assets"]),
                          output_bytes=sum(r["bytes"] for r in units)+sum(r["bytes"] for r in assets["assets"])
                          +sum(p.stat().st_size for p in OUT.glob("*.json")),
                          licenses=dict(Counter(r["license"] for r in assets["assets"])),
                          source_split_assets={site:dict(Counter(r["split"] for r in assets["assets"]
                                                                if r["source"]==site))
                                               for site in output["sources"]},
                          attribution_count=len(assets["attribution"]), pending_credentials=[],
                          excluded_assets=len(assets["exclusions"]), failed_clips=len(errors),
                          planned_clips=len(plan["units"])))


def quantiles(cases, field):
    values = [r[field] for r in cases if field in r]
    return np.quantile(values,[0,.25,.5,.75,1]).tolist() if values else []


def minmax(cases):
    return [min(r["bytes"] for r in cases),max(r["bytes"] for r in cases)] if cases else []


def run_batch(batch, assets, jobs, audit_check):
    cpu_textures, textures, indices = load_materials(assets)
    pupil = [from_numpy(sample_disk(256,.05/2/1.4)), from_numpy(lens_offsets())]
    tasks, rows, errors = [], [], []
    for spec in batch:
        try:
            outputs = render(spec, textures, indices, pupil, jobs["net"])
            audit = None
            if audit_check and spec["id"]==batch[0]["id"]:
                middle = spec["frame_count"]//2
                audit = cpu_optical_check(np.array(spec["rectangles"]),
                                          np.array(spec["camera_origins"][middle]),
                                          outputs[-1], cpu_textures, outputs[4], spec["perforated"],
                                          spec["foreground_translation_x"][middle])
                print("C independent CPU optical agreement", audit, flush=True)
            tasks.append((spec, jobs["writers"].submit(write_clip,spec,outputs,audit)))
            del outputs
        except (RuntimeError, ValueError) as exc:
            if audit_check and spec["id"]==batch[0]["id"]:
                raise
            errors.append(dict(id=spec["id"], reason=str(exc)))
        if len(tasks)>=5:
            finish(tasks.pop(0), rows, errors)
    for task in tasks:
        finish(task, rows, errors)
    return rows, errors


def finish(task, rows, errors):
    spec, future = task
    try:
        rows.append(future.result())
    except (OSError, ValueError) as exc:
        errors.append(dict(id=spec["id"], reason=str(exc)))


def check_ram():
    limit = int(Path("/sys/fs/cgroup/memory.max").read_text().strip())
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    if rss > 28*GIB or limit < 28*GIB:
        raise RuntimeError(f"task memory budget exceeded: host cgroup={limit}, peak RSS={rss}")
    return rss, limit


def gpu_bytes():
    output = subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,used_memory",
                                      "--format=csv,noheader,nounits"],text=True)
    return next((int(used.strip())*2**20 for pid,used in
                 (line.split(",") for line in output.splitlines() if "," in line)
                 if pid.strip()==str(os.getpid())),0)


def run(job_limit):
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    assets = read(OUT/"assets.json")["assets"]
    if not (OUT/"plan.json").exists():
        units, profiles = scenes(assets)
        save_json(OUT/"plan.json", dict(generator=GENERATOR, seed=SEED, units=units,
                                        procedural_profiles=profiles,
                                        sizes=dict(image=SIZE, pinhole_grid=4, aperture_grid=2, aperture_samples=256,
                                                   occupancy_samples=16, normal_frames=8, test_frames=13),
                                        checkpoint_sha256=sha(CHECKPOINT), source_sha256={
                                            k.name:sha(k) for k in Path(__file__).parent.glob("route3_v2data_c_*.py")}))
    plan = read(OUT/"plan.json")
    registry_path = OUT/"build-state.json"
    state = read(registry_path) if registry_path.exists() else dict(registry={}, errors={}, resource_use={})
    registry, errors, res = (state[k] for k in ("registry","errors","resource_use"))
    first_path = OUT/"first-batch.json"
    first = read(first_path) if first_path.exists() else {}
    order = plan["units"]
    # Measure 300 ordinary training clips first, then advance every pool and the
    # Stage 4 line together. The sealed testing case is planned before any render.
    tail = sorted(order[PILOT:], key=lambda r: (
        int(r["id"].rsplit("-",1)[1])/(MAIN if r["kind"]=="static" else MOTION)[r["split"]],
        r["id"]))
    remaining = [r for r in order[:PILOT]+tail if r["id"] not in registry and r["id"] not in errors]
    start = time.monotonic()
    partial_report(plan,registry,errors,first,"waiting_for_gpu",res)
    first_error = False
    with ThreadPoolExecutor(max_workers=4) as writers:
        net = None
        for i in range(0, min(len(remaining), job_limit), BATCH):
            batch = remaining[i:min(i+BATCH, job_limit)]
            estimate = first.get("max_unit_bytes", PILOT_UPPER_BYTES)
            if shutil.disk_usage(OUT).free < 60*GIB + int(1.2*len(batch)*estimate):
                partial_report(plan,registry,errors,first,"disk_capacity",res)
                return
            batch_wait = time.monotonic()
            with gpu_lock():
                active = time.monotonic()
                if net is None:
                    torch.cuda.set_per_process_memory_fraction(.55, 0)
                    net,_ = network(ROOT.parent/"depthsplat-r13-20261117", ROOT.parent/"dinov2-depthsplat-r13",
                                    CHECKPOINT)
                    net = net.cuda().eval()
                else:
                    net.cuda()
                try:
                    rows, failures = run_batch(batch, assets, dict(net=net,writers=writers), not registry)
                    torch.cuda.synchronize()
                    rss, cgroup = check_ram()
                    res = dict(cgroup_memory_bytes=cgroup, peak_rss_bytes=max(rss,res.get("peak_rss_bytes",0)),
                               torch_peak_reserved_bytes=max(torch.cuda.max_memory_reserved(),res.get("torch_peak_reserved_bytes",0)),
                               nvidia_smi_process_bytes=max(gpu_bytes(),res.get("nvidia_smi_process_bytes",0)),
                               seconds_waiting_for_lock=round(res.get("seconds_waiting_for_lock",0)+active-batch_wait,3),
                               seconds_under_lock=round(res.get("seconds_under_lock",0)+time.monotonic()-active,3),
                               cpu_processes=1, writer_threads=4, gpu="RTX 5090", lock=str(LOCK))
                    if res["nvidia_smi_process_bytes"]>24*GIB:
                        raise RuntimeError("C exceeded 24 GiB process VRAM")
                finally:
                    net.cpu()
                    torch.cuda.empty_cache()
            registry.update((r["id"],r) for r in rows)
            errors.update((r["id"],r) for r in failures)
            state["resource_use"] = res
            save_json(registry_path,state)
            if failures:
                print("C batch exceptions", failures, flush=True)
            if not first and len(registry)>=PILOT:
                sample = list(registry.values())[:PILOT]
                mean = sum(r["bytes"] for r in sample)/PILOT
                estimate_total = int(mean*len(order))
                first = dict(units=PILOT, bytes=sum(r["bytes"] for r in sample),
                             mean_unit_bytes=mean, min_unit_bytes=min(r["bytes"] for r in sample),
                             max_unit_bytes=max(r["bytes"] for r in sample), estimated_all_bytes=estimate_total,
                             available_bytes=shutil.disk_usage(OUT).free, reserve_bytes=60*GIB)
                save_json(first_path,first)
                print("C first batch", first, flush=True)
                if first["available_bytes"] < 60*GIB + estimate_total-first["bytes"]:
                    partial_report(plan,registry,errors,first,"disk_capacity",res)
                    return
            print("C progress", len(registry), "/", len(order), "failed", len(errors),
                  "seconds", round(time.monotonic()-start,1), flush=True)
            partial_report(plan,registry,errors,first,"in_progress",res)
            if not rows:
                first_error = True
                break
    status = "render_error" if first_error else ("complete" if len(registry)==len(order) else
              "shortfall" if len(registry)+len(errors)==len(order) else "in_progress")
    partial_report(plan,registry,errors,first,status,res)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-clips", type=int, default=100_000)
    args = p.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("set CUDA_VISIBLE_DEVICES=0 for the assigned GPU")
    run(args.max_clips)


if __name__ == "__main__":
    main()
