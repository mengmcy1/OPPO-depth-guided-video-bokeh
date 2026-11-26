"""Replay far-focus GT substitutes on the specified development and legacy frames."""
from dataclasses import replace
import argparse
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch

import route3_v2data_module2_infer as ref
from route3_ovg_1b_optical_data import local_placement
from route3_v2data_E0_rerender import make_visible
from route3_v2data_E0_train import gpu_gate
from route3_v2data_E_labels import load, save_json
from route3_v2data_B_inputs import cameras, observed_rgb
from route3_v2data_B_labels import StaticWorld
from route3_v2data_optics import project, ray_fields
from route3_v2data_stage2_eval import peeled_truth

FRAME = 30
PRIMARY = {"dev010", "dev011", "dev001", "dev006", "dev008", "122104"}
F2_ONLY = {"dev004", "dev010", "dev013"}
ROOT = ref.ROOT
OUT = ROOT / "outputs/route3-v2data-gt-substitute-audit-2026-11-26-01"
ARCHIVE = ROOT / "outputs/route3-v2data-E0-rerender-2026-11-26-01/fixed-base/frames"
RECORD = ROOT / "03_实验记录/records/route3-v2data-gt-substitute-audit-2026-11-26.json"
LIMIT = 6


def scalar_stats(values):
    values = np.asarray(values, np.float64)
    if not values.size:
        return dict(count=0, p50=None, p95=None)
    return dict(count=int(values.size), p50=float(np.median(values)), p95=float(np.quantile(values, .95)))


def score(rgb, truth, mask):
    return ref.pooled([ref.region_sse(rgb, truth, mask)])


def linear_stats(rgb, reference):
    error = np.abs(rgb.astype(np.float64) - reference)
    if not len(error):
        return dict(count=0, linear_L1=None, linear_PSNR=None, biased_pixels=None)
    return dict(count=len(error), linear_L1=float(error.mean()),
                linear_PSNR=float(-10 * np.log10(max(float(np.square(error).mean()), 1e-12))),
                biased_pixels=float((error.mean(-1) > .03).mean()),
                signed_RGB_mean=(rgb-reference).astype(np.float64).mean(0).tolist())


def parent_values(unit, name, case, native, depth):
    if unit:
        path = ref.PARENTS / name / (case["id"] + "-optics.npz")
        parents = ref.load_parents(path)
        front = native["gt"]
        background = native["valid"] & ~np.isin(native["object_id"], native["foreground"])
    else:
        path = ref.geometry_root(name) / "oracle/f030.npz"
        parents, front, background = ref.b_oracle(name, FRAME)
    x, y = np.floor(parents["uv"]).astype(int).T
    own = depth.detach().float().cpu().numpy()
    a, b, _, valid = local_placement(front, own, background, radius=24, return_valid=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        inverse = a[y, x] / parents["depth"] + b[y, x]
    keep = parents["recoverable"] & valid[y, x] & np.isfinite(inverse) & (inverse > 1e-4)
    placed = np.divide(1, inverse, out=np.zeros_like(inverse), where=keep)
    return parents, path, front, background, x, y, keep, placed, a[y,x], b[y,x]


def pick_pixel_parents(x, y, mask, z_error):
    ids = np.flatnonzero(mask)
    flat = y[ids] * 512 + x[ids]
    order = np.lexsort((z_error[ids], flat))
    return ids[order][np.unique(flat[order], return_index=True)[1]]


def flat_noise(image, flat):
    grey = image.astype(np.float32).mean(-1)
    residual = grey - cv2.GaussianBlur(grey, (0, 0), 1.2)
    return dict(pixels=int(flat.sum()), highpass_RMS=float(np.sqrt(np.mean(residual[flat]**2)))) if flat.any() else dict(pixels=0, highpass_RMS=None)


def trace_witness(unit, case):
    """Audit the footprint of the nearest RGB pixel selected by the witness lookup."""
    spec = dict(tiers="B", unit=unit, frame=FRAME, focus=case["focus_m"], fstop=case["fstop"],
                label=unit/"labels"/f"f{FRAME:03d}-focus1-K16.npz")
    rays = ray_fields(spec)
    selected = rays["recoverable"]
    k, y, x = np.nonzero(selected)
    intr, poses = cameras(unit)
    inv = np.linalg.inv(intr)
    pose = poses[FRAME-1]
    z = rays["depth"][selected].astype(np.float32)
    direc = np.stack((x+.5,y+.5,np.ones_like(x)),axis=-1) @ inv.T
    offset = rays["aperture_xy"][k] * (1-z[:,None]/spec["focus"])
    points = (direc*z[:,None]+np.column_stack((offset,np.zeros(len(offset))))) @ pose[:3,:3].T + pose[:3,3]
    uv, z = project(points,pose,intr)
    witnesses = rays["first_visible_other_frame"][selected].astype(int)
    objects = rays["ids"][selected]
    keep = (z>0)&(uv>=0).all(1)&(uv<512).all(1)&(witnesses>0)&(witnesses<=len(poses))
    pixels = np.floor(uv[keep]).astype(int)
    keys = np.column_stack((pixels,objects[keep]))
    unique = np.flatnonzero(keep)[np.unique(keys,axis=0,return_index=True)[1]]
    cache = ref.PARENTS/unit.name/(case["id"]+"-optics.npz")
    parents = ref.load_parents(cache)
    if not np.array_equal(uv[unique].astype(np.float32),parents["uv"]):
        raise ValueError(f"Witness and cache coordinates differ for {unit.name}")
    colors = np.zeros_like(parents["colors_linear"])
    center = np.zeros(len(unique),bool)
    fraction = np.zeros(len(unique),np.float32)
    valid = np.zeros(len(unique),bool)
    world = StaticWorld(unit/"labels/scene-geometry.npz",512)
    for witness in np.unique(witnesses[unique]):
        ids = np.flatnonzero(witnesses[unique]==witness)
        source = unique[ids]
        other = poses[witness-1]
        u, d = project(points[source],other,intr)
        inside = (d>0)&(u>=0).all(1)&(u<512).all(1)
        ids, source = ids[inside], source[inside]
        pix = np.floor(u[inside]).astype(int)
        rgb = observed_rgb(unit,int(witness))
        colors[ids] = rgb[pix[:,1],pix[:,0]]
        valid[ids] = True
        def hit(shiftx,shifty):
            camera = np.column_stack((pix[:,0]+shiftx,pix[:,1]+shifty,np.ones(len(pix)))) @ inv.T
            _, triangle = world.cast(int(witness),other[:3,3],camera @ other[:3,:3].T)
            return (triangle>=0)&(world.tri_obj[triangle.clip(0)]==objects[source])
        center[ids] = hit(.5,.5)
        fraction[ids] = np.mean([hit(a,b) for a in (.125,.375,.625,.875)
                                 for b in (.125,.375,.625,.875)],axis=0)
    if not np.array_equal(valid,parents["recoverable"]) or not np.array_equal(colors,parents["colors_linear"]):
        raise ValueError(f"Cached witness RGB was not reproduced for {unit.name}")
    geo, rgb, available = peeled_truth(unit,unit.name,FRAME)
    u, v = np.floor(uv[unique]).astype(int).T
    z0 = geo["depth_axial_m"][v,u].astype(np.float32)
    aligned = valid & available[v,u] & (np.abs(parents["depth"]-z0)<=np.maximum(.01,.005*z0))
    return dict(cache_RGB_exact=True, witnesses=sorted(set(int(k) for k in witnesses[unique])),
                same_surface_center_fraction=float(center[aligned].mean()),
                mean_expected_surface_subpixel_coverage=float(fraction[aligned].mean()),
                pure=linear_stats(colors[aligned&(fraction==1)], rgb[v[aligned&(fraction==1)],u[aligned&(fraction==1)]]),
                mixed=linear_stats(colors[aligned&(fraction<1)], rgb[v[aligned&(fraction<1)],u[aligned&(fraction<1)]]))


def hidden_measurements(unit, name, case, native, depth, mask, K):
    p, path, front, background, x, y, keep, placed, slope, offset = parent_values(unit, name, case, native, depth)
    geo, truth, gt_valid = peeled_truth(unit, name, FRAME)
    gt_depth = geo["depth_axial_m"].astype(np.float32)
    target = gt_depth[y, x]
    delta = np.abs(p["depth"] - target)
    compared = keep & mask[y, x] & gt_valid[y, x]
    aligned = compared & (delta <= np.maximum(.01, .005 * target))
    # Nearest pre-fit depth to the reference when several surfaces project to one pixel.
    ids = pick_pixel_parents(x, y, aligned, delta)
    rgb = p["colors_linear"].astype(np.float32)
    chosen = truth[y[ids], x[ids]]
    noisy_rgb = np.zeros_like(truth)
    covered = np.zeros_like(mask)
    noisy_rgb[y[ids], x[ids]] = rgb[ids]
    covered[y[ids], x[ids]] = True
    ordinary_index = pick_pixel_parents(x, y, compared, delta)
    all_rgb = np.zeros_like(truth)
    all_match = np.zeros_like(mask)
    all_rgb[y[ordinary_index], x[ordinary_index]] = rgb[ordinary_index]
    all_match[y[ordinary_index], x[ordinary_index]] = True
    err_depth = np.zeros_like(gt_depth)
    err_depth[y[ordinary_index],x[ordinary_index]] = np.abs(placed[ordinary_index]-target[ordinary_index])
    gt_gray = truth.mean(-1)
    smooth = np.abs(gt_gray - cv2.GaussianBlur(gt_gray, (0,0), 1.2)) < .004
    contiguous = cv2.erode((covered & gt_valid & smooth).astype(np.uint8), np.ones((7,7),np.uint8)).astype(bool)
    raw_err = np.abs(placed - target)
    cloud, placement = ref.place(p, front, background, depth, K, 512, order_depth=depth)
    if placement["placed"] != int(keep.sum()):
        raise ValueError("Audit and FP16-corrected placement counts differ")
    # Controlled intervention: keep parent locations, support and compositing order.
    # Recolour only projected parents on the known centre peeled surface.
    replaceable = keep & gt_valid[y, x] & (delta <= np.maximum(.01, .005 * target))
    alternate = rgb.copy()
    alternate[replaceable] = truth[y[replaceable],x[replaceable]]
    recoloured, _ = ref.place(dict(p, colors_linear=alternate), front, background, depth, K, 512, order_depth=depth)
    samples = load(unit/"labels/radiance-audit.json") if unit else {}
    info = dict(parent_file=str(path), placement=placement,
                samples_per_pixel=dict(parent_color=samples["pinhole_samples"] if unit else 256,
                    reference_peeled=samples["pinhole_samples"] if unit else 256,
                    main_optical_subview=samples["optical_samples_per_subview"] if unit else 1024,
                    denoising=False, optical_subviews=samples.get("optical",[{}])[0].get("subviews") if unit and samples.get("optical") else None),
                support=dict(recoverable_region_pixels=int(mask.sum()), with_valid_projected_peeled_pixel=int(all_match.sum()),
                             depth_matched_pixels=int(covered.sum()), recoloured_parents=int(replaceable.sum())),
                color_depth_matched=linear_stats(rgb[ids], chosen),
                color_all_overlaps=linear_stats(rgb[ordinary_index], truth[y[ordinary_index],x[ordinary_index]]),
                depth_abs_error_m=scalar_stats(raw_err[compared]),
                depth_abs_error_depth_matched_m=scalar_stats(raw_err[aligned]),
                prefit_abs_error_m=scalar_stats(delta[compared]),
                fit=dict(abs_a_minus_1=scalar_stats(np.abs(slope[compared]-1)), abs_b=scalar_stats(np.abs(offset[compared])),
                         input="GT center axial depth with prediction at center misses"),
                noise=dict(substitute=flat_noise(noisy_rgb, contiguous), peeled=flat_noise(truth, contiguous)))
    return info, cloud, recoloured, dict(peeled=truth, parent=all_rgb, matched=all_match, depth=err_depth)


def render_trace(visible, hidden, K, focus, offsets, fill, truth, mask):
    cloud = ref.merge_layers(visible, hidden)
    marker = None
    if hidden is not None:
        values = cloud.colors.new_zeros((len(cloud.means),1))
        values[len(visible.means):] = 1
        marker = replace(cloud, colors=values)
    z = K.new_zeros((*mask.shape,3))
    off, on = z.clone(), z.clone()
    hit_counts = torch.zeros(mask.shape, dtype=torch.int16, device=K.device)
    sum_t = K.new_zeros(mask.shape)
    truth_ray = truth[None]
    pieces = {k: dict(rays=0, on=0., off=0.) for k in ("hit","miss")}
    for rays, bg in zip(offsets.split(4), fill.split(4)):
        base, alpha = ref.raster(visible, K, focus, rays, 512, 512, bg, return_alpha=True)
        altered = ref.raster(cloud, K, focus, rays, 512, 512, bg) if marker is not None else base
        hits = (ref.raster(marker, K, focus, rays, 512, 512)[...,0] > 0) if marker is not None else torch.zeros_like(alpha, dtype=torch.bool)
        off += base.clamp(0,1).sum(0)
        on += altered.clamp(0,1).sum(0)
        hit_counts += hits.sum(0).short()
        sum_t += (1-alpha).sum(0)
        if marker is None:
            continue
        before = ref.display(base.clamp(0,1))
        after = ref.display(altered.clamp(0,1))
        for key, side in (("hit", hits.cpu().numpy()), ("miss", ~hits.cpu().numpy())):
            selected = side & mask[None]
            entry = pieces[key]
            entry["rays"] += int(selected.sum())
            entry["on"] += float(np.square((after-truth_ray)[selected].astype(np.float64)).sum())
            entry["off"] += float(np.square((before-truth_ray)[selected].astype(np.float64)).sum())
    rgb = [ref.display(t/len(offsets)) for t in (off,on)]
    hit = hit_counts.cpu().numpy()>0
    split = {key: dict(on=score(rgb[1],truth,mask & support), off=score(rgb[0],truth,mask & support))
             for key,support in (("any_hidden_hit",hit),("no_hidden_hit",~hit))}
    audit = dict(ray_shared_target_sse=pieces, integrated_sse_by_pixel=split,
                 hit_fraction=float(hit_counts.cpu().numpy()[mask].mean()/len(offsets)),
                 visible_transmittance=(sum_t/len(offsets)).cpu().numpy())
    return rgb, audit


def f2_regions(unit,name,native,boundary,mask,truth,off,predicted,T,Tpred,gt,z,K,focus):
    zones = ref.subsets(unit,name,FRAME,native)
    rod, gap = zones["foreground"], zones["gap_background"]
    regions = dict(rod=rod, gap=~rod&gap, boundary=~rod&~gap&boundary, normal=~rod&~gap&~boundary)
    g, p = (d.detach().cpu().numpy() for d in (gt,z))
    coc = lambda d: np.abs(1 / d - 1 / focus) * float(K[0,0]) * (.05/2.8)
    result = {}
    for key, support in regions.items():
        keep = mask & support
        ga, pa = score(off,truth,keep), score(predicted,truth,keep)
        result[key] = dict(V0_GT=ga, V0_pred=pa, excess=float(ga["sse"]-pa["sse"]),
            mean_transmittance_GT=float(T[keep].mean()) if keep.any() else None,
            mean_transmittance_pred=float(Tpred[keep].mean()) if keep.any() else None,
            CoC_GT=scalar_stats(coc(g)[keep]), CoC_pred=scalar_stats(coc(p)[keep]))
    return result


def outline(rgb, mask):
    out = np.asarray(rgb).copy()
    border = mask & ~cv2.erode(mask.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
    out[border] = (255, 226, 110)
    return Image.fromarray(out)


def save_panel(name, truth, off, on, proxy, mask):
    gallery = ref.gallery
    crop = ref.crop_for(on, truth, mask)
    rgb = (truth,off,on, gallery.display(proxy["parent"]),gallery.display(proxy["peeled"]))
    color = np.abs(proxy["parent"]-proxy["peeled"]).mean(-1) * proxy["matched"]
    error = np.abs(on-truth).mean(-1)
    maps = [gallery.rgb_image(x) for x in rgb] + [gallery.scalar_image(x,scale,cv2.COLORMAP_INFERNO)
           for x,scale in ((color,.25),(proxy["depth"],.25),(error,.25))]
    labels = ("Cycles 真值","GT 仅 F2","GT + 替身","替身颜色","中心剥离参照","线性颜色差 0–.25",
              "深度误差 0–.25 m","合成误差 0–.25")
    w=225
    image=Image.new("RGB",(w*len(maps),498),(22,26,32))
    draw=ImageDraw.Draw(image)
    draw.text((4,5),f"{name} · f030 远焦 · G · 黄线为可恢复区，第二行为共同放大框",
              font=gallery.font(18),fill="white")
    for i,(label,base) in enumerate(zip(labels,maps)):
        x=i*w
        draw.text((x+4,34),label,font=gallery.font(14),fill="white")
        marked=outline(base,mask)
        image.paste(marked.resize((w-4,w-4)),(x+2,53))
        image.paste(marked.crop(crop).resize((w-4,w-4),Image.Resampling.NEAREST),(x+2,276))
    path=OUT/f"{name}-f030-G.jpg"
    image.save(path,quality=94)
    return path.name


@torch.inference_mode()
def run():
    started = time.monotonic()
    OUT.mkdir(parents=True,exist_ok=True)
    ref.gpu_setup(dict(training=dict(max_process_gpu_gib=LIMIT)))
    gpu_gate(LIMIT)
    network = ref.load_geometry()
    rows = {}
    for cohort,name,frame,unit,entry,cases in ref.frames():
        slug = ref.title(name)
        if frame!=FRAME or slug not in PRIMARY | F2_ONLY:
            continue
        gpu_gate(LIMIT)
        print("GT_SUB_AUDIT",slug,flush=True)
        case = next(c for c in cases if c["focus_index"]==1)
        item = ("B", name, ref.ovg.TRAIN_FRAMES.index(frame))
        x,y = ref.sample(item)
        K=x["K"][0]
        native=ref.ovg.b_edge_reference(item,y)
        gt0=K.new_tensor(native["gt"])
        valid=torch.as_tensor(native["valid"],device=K.device)&torch.isfinite(gt0)&(gt0>0)
        z=network(x,protect_residual=True,render_heads=False)["depth1"][0]
        depth=torch.where(valid,gt0,z)
        kin=K.cpu().numpy()
        subject=ref.near_subject(native["gt"],native["valid"],kin,case["focus_m"])
        focus=ref.p2_focus(depth.cpu().numpy(),subject)
        masks=ref.new_regions(unit,case,native["gt"],native["valid"],kin) if unit else ref.legacy_masks(ref.geometry_root(name),case)
        mask=masks["recoverable"]
        truth=ref.optical_target(unit,name,case)
        offsets=ref.aperture_disk(64,.05/(2*case["fstop"]),device=K.device,dtype=K.dtype)
        fill=ref.f2_for(unit,entry,frame,x,focus,offsets)
        color=x["rgb"][0].permute(1,2,0)
        visible,_=make_visible(color,depth,K,None)
        details, hidden, alternative, proxy=hidden_measurements(unit,name,case,native,depth,mask,K)
        rgb,traces=render_trace(visible,hidden,K,focus,offsets,fill,truth,mask)
        before,after=(score(v,truth,mask) for v in rgb)
        alt=score(ref.display(ref.light(visible,alternative,K,focus,offsets,fill)),truth,mask)
        archived=load(ARCHIVE/name/"f030.json")["rows"]["G"][1]["methods"]["V0_GT"]["far_diagnostic"]
        for key,result in (("off",before),("on",after)):
            expected=ref.pooled([archived[key]])
            if abs(expected["psnr_db"]-result["psnr_db"])>.01:
                raise ValueError(f"Archived render mismatch for {slug} {key}: {expected} vs {result}")
        t=traces.pop("visible_transmittance")
        row=dict(scene=name,frame=frame,focus_m=focus,nominal_focus_m=case["focus_m"],score_without=before,
                 score_with=after,score_recoloured_matched_parents=alt,gt_minus_f2=after["psnr_db"]-before["psnr_db"],
                 **details,**traces)
        luminance = truth.astype(np.float32).mean(-1)
        flat = mask & (proxy["depth"]>0) & (np.abs(cv2.Laplacian(luminance,cv2.CV_32F))<.004)
        row["noise"].update(f2_image=flat_noise(rgb[0],flat), composite_image=flat_noise(rgb[1],flat))
        if slug in F2_ONLY:
            vis,_=make_visible(color,z,K,None)
            pred,tr=render_trace(vis,None,K,focus,offsets,fill,truth,mask)
            row["f2_regions"]=f2_regions(unit,name,native,ref.bands(native,y)[0],mask,truth,
                                         rgb[0],pred[0],t,tr["visible_transmittance"],depth,z,K,focus)
            row["score_pred_only_f2"]=score(pred[0],truth,mask)
            del vis
        row["panel"] = save_panel(slug,truth,*rgb,proxy,mask)
        rows[slug]=row
        save_json(OUT/"measurements.json",dict(scenes=rows,elapsed_seconds=time.monotonic()-started))
    if set(rows)!=PRIMARY | F2_ONLY:
        raise ValueError("The audit did not visit every requested frame")
    print("GT_SUB_AUDIT_COMPLETE",len(rows),flush=True)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--witness-only",action="store_true")
    if parser.parse_args().witness_only:
        rows = {}
        for _, name, frame, unit, _, cases in ref.frames():
            if unit and frame==FRAME and ref.title(name) in PRIMARY:
                case = next(c for c in cases if c["focus_index"]==1)
                rows[ref.title(name)] = trace_witness(unit,case)
                print("GT_SUB_WITNESS",ref.title(name),rows[ref.title(name)],flush=True)
        save_json(OUT/"witness.json",rows)
    else:
        run()
