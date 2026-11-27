"""Evaluate the frozen E0 v4 EMA checkpoints using the registered Query and S2."""
import argparse
from dataclasses import replace
from time import monotonic

import numpy as np
import torch

import route3_v2data_E0_rerender as replay
import route3_v2data_E0_v3 as v3
import route3_v2data_E0_v3_color as witness
import route3_v2data_module2_infer as ref
import route3_v2data_stage3b_eval as base
from route3_v2data_B_batch import sha256
from route3_v2data_E0_evaluate import EInputs
from route3_v2data_E0_tears import features, load_model
from route3_v2data_E0_train import gpu_gate, predict
from route3_v2data_E0_v4 import CURVE_STEPS, RECORD, RUN
from route3_v2data_E_labels import load, save_json
from route3_v2data_hidden_color_purity import substitute_alternatives
from src.layered_aperture import merge_layers

FOCI = ("P", "G")
PANEL_SCENES = ("dev003", "dev005", "dev008")
GPU_GIB = 7
PREVIOUS = RUN.parent / "route3-v2data-E-query-2026-11-27-01"
EARLIER = RECORD.with_name("route3-v2data-E-query-2026-11-27.json")
V2_REPLAY = v3.v2.RUN / "evaluation/rest"
BASES = dict(v3.BASES, v4=RUN)


def setup():
    base.gpu_setup(dict(training=dict(max_process_gpu_gib=GPU_GIB)))
    gpu_gate(GPU_GIB)


def checkpoint(version, step, *, stitch):
    model, path = load_model(v3.MODE, step, run=BASES[version])
    if stitch:
        if version == "v4":
            weights = RUN / v3.MODE / f"tear-{step:06d}" / "selected.pt"
        elif (version, step) == ("v3", 4000):
            weights = PREVIOUS / "tear-D" / "selected.pt"
            if sha256(weights) != load(EARLIER)["D_tear"]["checkpoint_sha256"]:
                raise ValueError("Archived D tear head differs from the four-arm release")
        else:
            weights = v3.tear_root(version, step) / "selected.pt"
        tear = replay.load_tear(weights)
    else:
        tear = None
    return model, tear, sha256(path)


def query_net():
    registered = load(EARLIER)["query"]
    if sha256(base.QUERY) != registered["sha256"]:
        raise ValueError("The adopted 20k Query checkpoint does not match its registered hash")
    return base.query_net(base.QUERY)


def build_visible(model, tear, target, x):
    out = predict(model, target)
    depth = out["decoded"][0, 0].reciprocal()
    if not torch.isfinite(depth).all() or not (depth > 0).all():
        raise FloatingPointError("Invalid E decoded depth")
    rgb = x["rgb"][0].permute(1, 2, 0)
    field = None
    if tear is not None:
        field = (tear(features(out, x["rgb"]).half().float())[0].sigmoid() > .5).cpu().numpy().astype(np.int8)
    visible, maximum = replay.make_visible(rgb, depth, x["K"][0], field)
    return visible, depth, maximum


def reference_cloud(unit, scene, frame, case, native, K, layer, *, local_witnesses=None):
    _, depth, maximum = layer
    if unit is None:
        return ref.place_hidden(unit, scene, frame, case, native, depth, maximum, K)
    parents, variants = witness.witnesses(scene, case["id"]) if local_witnesses is None else local_witnesses
    if parents is None:
        return None
    _, result = next(substitute_alternatives(layer, unit, scene, frame, native, K, parents,
                                             {witness.S2: variants[witness.S2]}))
    return result


def render(visible, hidden, K, focus, offsets, fill, *, trace=False):
    """Reproduce the K64 RGB render and optionally recover the hidden layer's mean RGB."""
    if not trace:
        return ref.display(ref.light(visible, hidden, K, focus, offsets, fill)), None, None
    if hidden is None:
        rgb, _, _ = render(visible, None, K, focus, offsets, fill)
        return rgb, np.zeros_like(rgb), np.zeros(rgb.shape[:2], np.float32)
    scene = merge_layers(visible, hidden)
    tag = scene.colors.new_zeros((len(scene.means), 1))
    tag[len(visible.means):] = 1
    scene = replace(scene, colors=torch.cat((scene.colors, tag, scene.colors * tag), dim=1))
    rgb, mass, colour = 0, 0, 0
    for rays, background in zip(offsets.split(4), fill.split(4), strict=True):
        value, alpha = base.raster(scene, K, focus, rays, 512, 512, return_alpha=True)
        rgb += (value[..., :3] + (1 - alpha[..., None]) * background).clamp(0, 1).sum(0)
        mass += value[..., 3].sum(0)
        colour += value[..., 4:].sum(0)
    return (ref.display(rgb / len(offsets)), (colour / mass.clamp_min(1e-8)[..., None]).cpu().numpy(),
            (mass / len(offsets)).cpu().numpy())


def score(rgb, truth, support):
    row = ref.region_sse(rgb, truth, support)
    if row["pixels"] != int(support.sum()):
        raise FloatingPointError("Nonfinite RGB inside the score mask")
    return row


def hidden_error(q_rgb, q_mass, ref_rgb, ref_mass, mask):
    support = mask & (q_mass >= .001) & (ref_mass >= .001)
    error = np.abs(q_rgb - ref_rgb).mean(-1)
    return error, support


def cases_for(job, x, labels, depth, *, complete):
    _, scene, frame, unit, _, cases = job
    native = base.ovg.b_edge_reference(("B", scene, base.ovg.TRAIN_FRAMES.index(frame)), labels)
    K = x["K"][0].cpu().numpy()
    z = depth.cpu().numpy()
    work = []
    excluded = []
    for case in cases:
        idx = case["focus_index"]
        if not complete and idx != 1:
            continue
        subject = ref.near_subject(native["gt"], native["valid"], K, case["focus_m"])
        if not subject.any():
            excluded.append(case["id"])
            continue
        masks = (ref.new_regions(unit, case, native["gt"], native["valid"], K) if unit else
                 ref.legacy_masks(ref.geometry_root(scene), case))
        mask = masks["recoverable" if idx else "in_focus"]
        truth = ref.optical_target(unit, scene, case)
        foci = dict(P=ref.p2_focus(z, subject), G=ref.p2_focus(native["gt"], subject))
        for name, focus in foci.items():
            if focus is None:
                raise ValueError(f"Invalid P2 focus for {scene}/{case['id']}/{name}")
            work.append(dict(case=case, mask=mask, truth=truth, kind=name, focus=focus, gt_focus=foci["G"],
                             band=base.optical_band(z, K, focus, case["fstop"]) if complete else None))
    return native, work, excluded


def archived_v2_f2(scene, frame, case, kind, focus):
    path = V2_REPLAY / "frames" / scene / f"f{frame:03d}.json"
    row = next(r for r in load(path)["rows"][kind] if r["case"] == case["id"])["methods"]["E_B"]
    if abs(row["focus_m"] - focus) > 1e-6:
        raise ValueError("Archived v2 F2 focus differs from the current reconstruction")
    return row["far_off"]


def render_frame(job, models, inputs, geometry, net=None, coordinates=None, *, reuse_v2_f2=False):
    cohort, scene, frame, unit, entry, _ = job
    item = ("B", scene, base.ovg.TRAIN_FRAMES.index(frame))
    x, labels = base.sample(item)
    context = (inputs.native(scene, frame) if unit else inputs.observations(
        scene, frame, x, geometry(x, protect_residual=True, render_heads=False, return_features=True)))
    layer = build_visible(*models, context, x)
    visible, depth, maximum = layer
    K = x["K"][0]
    native, work, excluded = cases_for(job, x, labels, depth, complete=net is not None)
    rows, pictures = [], {}
    if net is not None and work:
        center = K.new_zeros((1, 2))
        pinhole = ref.display(x["rgb"][0].permute(1, 2, 0))
        full = np.ones((512, 512), dtype=bool)
        zero = score(ref.display(base.raster(visible, K, 1.4, center, 512, 512)[0]), pinhole, full)
        views, memory = coordinates
        packed = base.pack_frame(views, frame, depth.cpu().numpy(), [row["band"] for row in work], memory)
        store = base.FrameQueries(views, packed)
    for work_item in work:
        case, kind, focus, band, mask, truth = (work_item[k] for k in ("case","kind","focus","band","mask","truth"))
        row = dict(cohort=cohort, scene=scene, frame=frame, case=case["id"], focus_index=case["focus_index"],
                   focus_mode=kind, focus_m=focus, GT_focus_m=work_item["gt_focus"])
        try:
            gpu_gate(GPU_GIB)
            idx = case["focus_index"]
            stop = case["fstop"]
            pupil = base.aperture_disk(64, .05/(2*stop), device=K.device, dtype=K.dtype)
            fill = base.f2_for(unit, entry, frame, x, focus, pupil)
            traced = net is not None and frame == 30 and idx == 1 and ref.title(scene) in PANEL_SCENES
            if idx == 1:
                comparison = reference_cloud(unit, scene, frame, case, native, K, layer)
                if reuse_v2_f2:
                    off = archived_v2_f2(scene, frame, case, kind, focus)
                    f2_rgb = None
                else:
                    f2_rgb, _, _ = render(visible, None, K, focus, pupil, fill)
                    off = score(f2_rgb, truth, mask)
                s2_rgb, s2_color, s2_mass = render(visible, comparison, K, focus, pupil, fill, trace=traced)
                row["far"] = dict(F2=off, reference=score(s2_rgb, truth, mask))
                if off["pixels"] != int(mask.sum()):
                    raise ValueError("Archived F2 and S2 score masks differ")
                del comparison
            if net is not None:
                selected = store.active("native", focus, stop)
                cloud = base.hidden_cloud(net, store, dict(frame=frame,input="native",case=dict(case,focus_m=focus)),
                                          K, depth, band, indices=selected, placement_depth=maximum)
                query_rgb, q_color, q_mass = render(visible, cloud, K, focus, pupil, fill, trace=traced)
                row["Query"] = score(query_rgb, truth, mask)
                row["query_pixels"] = int(len(selected))
                center_fill = base.f2_for(unit, entry, frame, x, focus, center)
                composite = base.raster(merge_layers(visible, cloud), K, focus, center, 512, 512, center_fill)[0]
                row["zero_visible"] = zero
                row["zero_combined"] = score(ref.display(composite), pinhole, full)
                if idx == 1:
                    row["far"]["Query"] = row["Query"]
                if traced:
                    error, overlap = hidden_error(q_color, q_mass, s2_color, s2_mass, mask)
                    for tag, image in dict(truth=truth,F2=f2_rgb,reference=s2_rgb,Query=query_rgb,
                                           area=mask,hidden_error=error,shared_mask=overlap).items():
                        pictures[f"{kind}/{tag}"] = image if image.dtype == bool else image.astype(np.float16)
                    row["hidden_colour_comparison"] = dict(shared_pixels=int(overlap.sum()),
                        mean_linear_mae=float(error[overlap].mean()) if overlap.any() else None)
                del cloud, center_fill, composite
            del fill
            row["status"] = "evaluated"
        except (KeyError, ValueError, RuntimeError, FloatingPointError, OSError) as exc:
            row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            print("E0_V4_SKIP", scene, case["id"], kind, row["error"], flush=True)
            torch.cuda.empty_cache()
        rows.append(row)
    return dict(rows=rows, unscored=excluded), pictures


@torch.inference_mode()
def evaluate(version, step, *, full=False):
    path = RUN / ("full" if full else f"curve/{version}-{step:06d}")
    mark = path / "finished.json"
    if mark.is_file():
        return load(mark)
    setup()
    began = monotonic()
    model, tear, digest = checkpoint(version, step, stitch=full)
    e = EInputs(v3.MODE)
    geom = base.load_geometry() if full else None
    net = query_net() if full else None
    legacy = {(p["name"],p["frame"]):p for p in base.gallery.planned_frames() if p["unit"] is None} if full else {}
    jobs = [j for j in base.frames() if j[0]=="development" or full]
    if len(jobs) != (54 if full else 45):
        raise ValueError("Unexpected development or legacy frame inventory")
    view_cache = None
    for job in jobs:
        cohort, scene, frame, unit, _, cases = job
        dest, image = replay.frame_paths(path,scene,frame)
        if dest.is_file():
            continue
        try:
            if full and (view_cache is None or scene != view_cache[0]):
                views = base.gallery.scene_views(dict(unit=unit) if unit else legacy[scene,frame])
                view_cache = scene, views
            rows, pictures = render_frame(job,(model,tear),e,geom,net,view_cache[1] if full else None,
                                           reuse_v2_f2=version=="v2" and step==4000)
        except (KeyError, ValueError, RuntimeError, FloatingPointError, OSError) as exc:
            print("E0_V4_FRAME_SKIP", scene, frame, type(exc).__name__, str(exc), flush=True)
            rows = dict(unscored=[], rows=[dict(cohort=cohort,scene=scene, frame=frame,case=c["id"],
                        focus_mode=f,focus_index=c["focus_index"],status="failed",error=f"{type(exc).__name__}: {exc}")
                        for c in cases if full or c["focus_index"]==1 for f in FOCI])
            pictures = {}
            torch.cuda.empty_cache()
        dest.parent.mkdir(parents=True,exist_ok=True)
        replay.save_arrays(image,pictures)
        save_json(dest,rows)
        print("E0_V4_OPTICS",version,step,full,ref.title(scene),frame,monotonic()-began,flush=True)
    resource = dict(version=version,step=step,complete=full,checkpoint_sha256=digest,seconds=monotonic()-began,
                    frames=len(jobs),peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3)
    save_json(mark,resource)
    return resource


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("curve","full"))
    parser.add_argument("--version",choices=tuple(CURVE_STEPS))
    parser.add_argument("--step",type=int)
    args=parser.parse_args()
    if args.stage=="full":
        selected=load(RUN/v3.MODE/"training.json")["selected"]["step"]
        evaluate("v4",selected,full=True)
    elif args.version and args.step in CURVE_STEPS[args.version]:
        evaluate(args.version,args.step)
    else:
        parser.error("Use an enumerated curve version and step")
