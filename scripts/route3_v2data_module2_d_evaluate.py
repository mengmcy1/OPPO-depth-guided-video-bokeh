"""Evaluate learned tears on the same development cases as the seven-arm study."""
import argparse
from dataclasses import replace
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch

import route3_ovg_data as ovg
import route3_v2data_module2_infer as old
import route3_v2data_stage1_visualize as gallery
from route3_v2data_B_batch import GIB, sha256
from route3_v2data_B_plan import load
from route3_v2data_module2_d_labels import RUN
from route3_v2data_stage2_train import INITIAL
from src.ovg_tear_head import VARIANTS, TearHead, tear_features

BASE = old.OUT/"all-arms"
RESULTS = RUN/"evaluation"
ARMS = ("V0_pred", "d_nodepth", "d_full", "b", "c_010")
REFERENCE = dict(V0_pred="V0 预测", b="b · GT 撕裂", c_010="c · τ=.10")
LEARNED = ("d_full", "d_nodepth")


def chosen_heads(training, variants=VARIANTS):
    heads = {}
    for arm, channels in variants.items():
        selected = training["selected"][arm]
        ck = torch.load(selected["path"], map_location="cpu", weights_only=True)
        if ck["step"] != selected["step"] or ck["geometry_sha256"] != training["geometry_sha256"]:
            raise ValueError(f"Selected {arm} does not match the frozen geometry or development decision")
        heads[arm] = TearHead(channels).cuda().eval().requires_grad_(False)
        heads[arm].load_state_dict(ck["ema"])
    return heads


def predictor(heads):
    def make(x, pred):
        native = tear_features(pred["features"], x["rgb"], pred["depth1"].reciprocal()).half().float()
        maps = {name: (head(native[:, :head[0].in_channels])[0].sigmoid() > .5).to(torch.int8).cpu().numpy()
                for name, head in heads.items()}
        for tear in maps.values():
            for i, (dy, dx) in enumerate(old.DIRECTIONS):
                if dy:
                    tear[i, -1] = -2
                if dx:
                    tear[i, :, -1 if dx > 0 else 0] = -2
        return maps
    return make


def append_panels(archived):
    def draw(path, row, renders, gt, reference, masks, overlay, arms):
        saved = archived[row["case"]]
        tile, pitch = 182, 190
        inside, edge = saved["target_crop_xyxy"], saved["outline_crop_xyxy"]
        with Image.open(saved["panel"]) as previous:
            canvas = Image.new("RGB", (previous.width+pitch*len(arms), previous.height), (18, 22, 32))
            canvas.paste(previous, (0, 0))
            left = previous.width
        font = gallery.font(13)
        text = ImageDraw.Draw(canvas)
        text.text((left, 5), "学习撕裂头 · t > 0.5", font=gallery.font(17), fill="white")
        for arm in arms:
            rec = row["methods"][arm]
            image = renders[arm]
            rgb = gallery.rgb_image(image)
            error = gallery.scalar_image(np.abs(image-gt).mean(-1), .25, cv2.COLORMAP_INFERNO)
            base = reference.copy()*.48
            active = overlay[arm].any(-1)
            base[active] = overlay[arm][active]/255.
            tears = gallery.rgb_image(base).crop(edge)
            heat = rec.get("heat")
            diagnostic = tears
            if heat is not None:
                bright = gallery.scalar_image(heat, 1., cv2.COLORMAP_INFERNO)
                diagnostic = Image.new("RGB", (tile, tile))
                diagnostic.paste(bright.resize((tile, tile//2)), (0, 0))
                diagnostic.paste(tears.resize((tile, tile//2)), (0, tile//2))
            psnr = old.score_psnr(rec["optical"]["full_image"])
            text.text((left+2, 53), f"{arm.replace('_','-')} · {psnr:.1f}", font=font, fill="white")
            for j, layer in enumerate((rgb, rgb.crop(inside), rgb.crop(edge), error, diagnostic)):
                mode = Image.Resampling.NEAREST if j in (1, 2) else Image.Resampling.BOX
                canvas.paste(layer.resize((tile, tile), mode), (left, 76+j*207))
            left += pitch
        path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(path, quality=94)
        return dict(panel=str(path), target_crop_xyxy=inside, outline_crop_xyxy=edge,
                    archived_panel=saved["panel"])
    return draw


def join_cases(result, previous, reference_arms=None):
    names = reference_arms or {arm:arm for arm in REFERENCE}
    for current in result["rows"]:
        base = previous[current["case"]]
        for key in ("scene", "frame", "cohort", "status", "focus_index", "nominal_focus_m", "GT_focus_m"):
            if current[key] != base[key]:
                raise ValueError(f"Archived comparison {key} differs on {current['scene']}/{current['case']}")
        if current["status"] == "unscored":
            continue
        reference = base["methods"]["V0_pred"]
        for row in current["methods"].values():
            if not np.isclose(row["focus_m"], reference["focus_m"], rtol=0, atol=1e-7):
                raise ValueError(f"Forecast arms do not share their archived P2 focus: {current['case']}")
            for key in old.REGIONS:
                if row["optical"][key]["pixels"] != reference["optical"][key]["pixels"]:
                    raise ValueError(f"Archived {key} pixel support differs on {current['case']}")
            for key in old.ZERO:
                if row[key]["pixels"] != reference[key]["pixels"]:
                    raise ValueError(f"Archived {key} pixel support differs on {current['case']}")
            if row["rays"] and row["rays"]["rays"] != reference["rays"]["rays"]:
                raise ValueError(f"Archived K16 subject ray count differs on {current['case']}")
        current["methods"].update({arm: base["methods"][archived] for arm,archived in names.items()})


def gap_crop(mask):
    """Center a 96-pixel square on the largest concentration of labelled gap pixels."""
    integral = cv2.integral(mask.astype(np.uint8))
    size = 96
    counts = integral[size:, size:]-integral[:-size, size:]-integral[size:, :-size]+integral[:-size, :-size]
    y, x = np.unravel_index(counts.argmax(), counts.shape)
    return int(x), int(y), int(x+size), int(y+size)


def gap_ray_trace(visible, hidden, K, focus, rays, fill, mask):
    """Exact alpha-weighted hidden RGB contribution and diagnostic prefix T."""
    if hidden is None:
        pixels = int(mask.sum())
        t = np.zeros(mask.shape, np.float32)
        rgb = K.new_zeros((*mask.shape, 3))
        for r, background in zip(rays.split(4), fill.split(4)):
            values, alpha = old.raster(visible, K, focus, r, 512, 512, background, return_alpha=True)
            t += (1-alpha).sum(0).cpu().numpy()
            rgb += values.clamp(0, 1).sum(0)
        t /= len(rays)
        report = dict(gap_pixels=pixels, rays=pixels*len(rays), hidden_hit_rays=0, hidden_hit_fraction=0.,
                      mean_visible_prefix_T_given_hidden_hit=None,
                      mean_visible_prefix_T_hidden_unreached_as_zero=0.,
                      mean_visible_residual_T=float(t[mask].mean()), mean_hidden_composite_weight=0.,
                      mean_hidden_linear_RGB_contribution=[0., 0., 0.])
        return report, (old.display(rgb/len(rays)), t, np.zeros((*mask.shape, 3), np.float32))

    cloud = old.merge_layers(visible, hidden)
    membership = cloud.colors.new_zeros((len(cloud.means), 1))
    membership[len(visible.means):] = 1.
    features = torch.cat((cloud.colors, membership, cloud.colors*membership), dim=1)
    traced = replace(cloud, colors=features)
    active = torch.as_tensor(mask, device=K.device)
    rendered, color, mass, terminal = [], [], [], []
    for r, background in zip(rays.split(4), fill.split(4)):
        output, alpha = old.raster(traced, K, focus, r, 512, 512, return_alpha=True)
        rendered.append((output[..., :3]+(1-alpha[..., None])*background).clamp(0, 1).sum(0).cpu())
        mass.append(output[..., 3].sum(0).cpu().numpy())
        color.append(output[..., 4:7].sum(0).cpu().numpy())
        _, va = old.raster(visible, K, focus, r, 512, 512, return_alpha=True)
        terminal.append((1-va).sum(0).cpu().numpy())
    slots = [old.trace_predicted(visible, visible, cloud, K, focus, r, active) for r in rays]
    weights, hidden_z, _, _, prefix = map(np.stack, zip(*slots))
    reached = (weights > 0) & (hidden_z > 0) & np.isfinite(prefix)
    n = len(rays)
    out, t, c, w = (sum(a)/n for a in (rendered, terminal, color, mass))
    report = dict(gap_pixels=int(mask.sum()), rays=int(mask.sum())*n, hidden_hit_rays=int(reached.sum()),
                  hidden_hit_fraction=float(reached.mean()),
                  mean_visible_prefix_T_given_hidden_hit=float(prefix[reached].mean()) if reached.any() else None,
                  mean_visible_prefix_T_hidden_unreached_as_zero=float(np.where(reached, prefix, 0).mean()),
                  mean_visible_residual_T=float(t[mask].mean()),
                  mean_hidden_composite_weight=float(w[mask].mean()),
                  mean_hidden_linear_RGB_contribution=c[mask].mean(0).tolist())
    return report, (old.display(out), t, c)


def gap_panel(path, name, case, gt, mask, crop, pictures):
    tile, pitch = 210, 219
    xy = Image.new("RGB", (4*pitch, 61+3*(tile+14)), (18, 22, 32))
    text = ImageDraw.Draw(xy)
    text.text((6, 6), f"{name} · {case} · GT 细结构框内背景", font=gallery.font(17), fill="white")
    text.text((6, 31), "同框：远焦输出、可见层剩余透射 (0–1)、背景层线性 RGB 合成贡献 (0–0.25)。",
              font=gallery.font(13), fill="white")
    gt_mask = np.where(mask[..., None], gt, gt*.40)
    columns = {"GT（缝隙加亮）": (gallery.rgb_image(gt), gallery.rgb_image(gt_mask), None)}
    for arm, (image, t, color) in pictures.items():
        columns[arm] = (gallery.rgb_image(image),
                        gallery.scalar_image(t, 1., cv2.COLORMAP_INFERNO),
                        gallery.rgb_image(np.clip(color/.25, 0, 1)))
    for i, (label, stack) in enumerate(columns.items()):
        text.text((i*pitch+3, 48), label, font=gallery.font(14), fill="white")
        for j, layer in enumerate(stack):
            if layer is not None:
                xy.paste(layer.crop(crop).resize((tile, tile), Image.Resampling.NEAREST), (i*pitch, 69+j*(tile+14)))
    xy.save(path, quality=95)


def diagnose_dev005(net, predict, cases, unit, entry, frame, official):
    name = unit.name
    item = ("B", name, ovg.TRAIN_FRAMES.index(frame))
    x, y = old.sample(item)
    pred = net(x, protect_residual=True, render_heads=False, return_features=True)
    depth = pred["depth1"][0]
    K, color = x["K"][0], x["rgb"][0].permute(1, 2, 0)
    ref = ovg.b_edge_reference(item, y)
    gap = old.subsets(unit, name, frame, ref)["gap_background"]
    del y
    case = next(c for c in cases if c["focus_index"] == 1)
    record = next(r for r in official if r["case"] == case["id"])
    output = dict(scene=name, frame=frame, case=case["id"], gap_pixels=int(gap.sum()), methods={})
    if record["status"] != "evaluated" or not gap.any():
        return output
    focus = record["methods"]["V0_pred"]["focus_m"]
    rays = old.aperture_disk(64, .05/(2*case["fstop"]), device=K.device, dtype=K.dtype)
    fill = old.f2_for(unit, entry, frame, x, focus, rays)
    labels = dict(b=ovg.load(old.CACHE/name/f"f{frame:03d}.npz")["status"], **predict(x,pred))
    del pred
    crop = gap_crop(gap)
    output.update(focus_m=focus, GT_focus_m=record["GT_focus_m"], crop_xyxy=crop)
    pictures = {}
    for arm in ("V0_pred", "b", "d_full"):
        old.yield_gpu()
        options = {}
        maximum = depth
        if arm in labels:
            inverse, _, _ = old.subdivide(depth.reciprocal().cpu().numpy(), labels[arm])
            sub = 1/inverse
            sub[..., 2, 2] = depth.cpu().numpy()
            maximum = K.new_tensor(sub.max(axis=(2, 3)))
            options["subpixel_depth"] = K.new_tensor(sub.transpose(0, 2, 1, 3).reshape(2560, 2560))
        visible = old.pixel_layer(color, depth, K, torch.eye(4, device=K.device, dtype=K.dtype), **options)
        hidden, placement = old.place_hidden(unit, name, frame, case, ref, depth, maximum, K, return_stats=True)
        output["methods"][arm], pictures[arm] = gap_ray_trace(visible, hidden, K, focus, rays, fill, gap)
        output["methods"][arm]["placement"] = placement
        del visible, hidden
    truth = old.optical_target(unit, name, case)
    path = RESULTS/"panels"/f"dev005-f{frame:03d}-far-gap.jpg"
    gap_panel(path, name, case["id"], truth, gap, crop, pictures)
    output["panel"] = str(path)
    return output


@torch.inference_mode()
def run(one, *, run_root=RUN, archive_path=BASE/"report.json", variants=VARIANTS, references=None,
        methods=ARMS, baseline_commit="1a922ac", trace_gap=True):
    training = load(run_root/"training-report.json")
    before = load(archive_path)
    if not training or not before:
        raise FileNotFoundError("The learned checkpoint decision and archived seven-arm evaluation are required")
    if training["geometry_sha256"] != sha256(INITIAL):
        raise ValueError("The selected tear heads were trained on a different frozen geometry")
    if before["test_cases_used"] or before["summary"]["development"]["scored"] != 87:
        raise ValueError("The seven-arm archive does not match the preregistered development protocol")
    learned = tuple(variants)
    pairings = {method:"V0_pred" for method in methods if method!="V0_pred"}
    root = run_root/("evaluation" if one is None else "evaluation-smoke")
    began = time.monotonic()
    competitor_begin = old.competitor_state()
    old.gpu_setup(dict(training=dict(max_process_gpu_gib=old.GPU_GIB)))
    waited = old.yield_gpu()
    model = old.load_geometry()
    predict = predictor(chosen_heads(training, variants))
    reference = {}
    for row in before.get("case_rows",before.get("cases",())):
        reference.setdefault((row["scene"], row["frame"]), {})[row["case"]] = row
    parts, gaps = [], []
    for cohort,name,frame,unit,entry,cases in old.frames():
        if one and (old.title(name), frame) != (one, 30):
            continue
        lookup = reference.get((name, frame))
        if lookup is None:
            if cohort == "legacy":
                continue
            raise ValueError(f"Development frame missing from the frozen seven-arm archive: {name}/{frame}")
        if set(lookup) != {c["id"] for c in cases}:
            raise ValueError(f"Case inventory changed for {name}/{frame}")
        save = root/"frames"/name/f"f{frame:03d}.json"
        part = load(save)
        if not part:
            waited += old.yield_gpu()
            part = old.frame_results(model, cohort, name, frame, unit, entry, cases, root, learned,
                                     tear_predictor=predict, panel_writer=append_panels(lookup))
            join_cases(part, lookup, references)
            old.save_json(save, part)
        parts.append(part)
        print("TEAR_OPTICAL_FRAME", name, frame, len(parts), flush=True)
        if trace_gap and one is None and old.title(name) == "dev005":
            path = root/"gap"/f"f{frame:03d}.json"
            gap = load(path)
            if not gap:
                waited += old.yield_gpu()
                gap = diagnose_dev005(model, predict, cases, unit, entry, frame, part["rows"])
                old.save_json(path, gap)
            gaps.append(gap)
    summary = old.summarize(parts, methods, pairings=pairings, include_absolute_near=True)
    if not parts or (not one and any(summary[c]["scored"] != before["summary"][c]["scored"]
                                    for c in ("development", "legacy"))):
        raise ValueError("The learned arms did not cover the frozen scored case inventory")
    development = [p for p in parts if p["rows"][0]["cohort"] == "development"]
    if not one:
        # Compare the exact native threshold used for training selection and rendering.
        edge = old.acc_stats(development)["tear_accuracy"]
        for arm in learned:
            selected_step = training["selected"][arm]["step"]
            monitor = next(r["development"][arm]["regions"] for r in training["history"]
                           if r["step"] == selected_step)
            for region, counts in edge[arm].items():
                expected = monitor[region]
                if (counts["false_step_connected"], counts["GT_torn"], counts["false_step_torn"],
                    counts["false_connected"]) != (expected["false_step_edges"], expected["torn_edges"],
                                                   expected["falsely_torn"], expected["falsely_connected"]):
                    raise ValueError(f"The {arm}/{region} native edge-validation support changed")
    old.save_json(root/"report.json",
                  dict(summary=summary, diagnostics=old.acc_stats(development),
                       cases=[r for p in parts for r in p["rows"]], gap=gaps,
                       configuration=dict(geometry_sha256=training["geometry_sha256"], baseline_commit=baseline_commit,
                                          reused_arms=list(references or REFERENCE), render_K=64, subject_diagnostic_K=16),
                       seconds=time.monotonic()-began, gpu_wait_seconds=waited+sum(p["gpu_wait_seconds"] for p in parts),
                       peak_reserved_gib=torch.cuda.max_memory_reserved()/GIB,
                       peak_allocated_gib=torch.cuda.max_memory_allocated()/GIB,
                       test_cases_used=0, competitor_start=competitor_begin, competitor_end=old.competitor_state()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", metavar="SCENE_SHORT_NAME")
    parser.add_argument("--v2", action="store_true")
    arguments = parser.parse_args()
    if arguments.v2:
        from route3_v2data_module2_d_v2_train import ARMS as V2, CHANNELS, RUN as V2_RUN

        archived = dict(V0_pred="V0_pred", v1="d_full", b="b", c_010="c_010")
        run(arguments.smoke, run_root=V2_RUN, archive_path=RESULTS/"report.json",
            variants=dict.fromkeys(V2,CHANNELS), references=archived,
            methods=("V0_pred","v1",*V2,"b","c_010"), baseline_commit="32b8641", trace_gap=False)
    else:
        run(arguments.smoke)
