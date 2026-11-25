"""Cache frozen OVG features and train matched module-two tear heads."""
from collections import Counter
from copy import deepcopy
from datetime import datetime
import json
import os
import shutil
import time

import numpy as np
import torch
from torch.nn import functional as F

import route3_ovg_data as ovg
from route3_v2data_B_batch import GIB, sha256
from route3_v2data_B_inputs import library_unit_paths
from route3_v2data_module2_d_labels import CACHE, CONFIG, FRAMES, RUN, save_json
from route3_v2data_module2_infer import CACHE as DEV_LABELS
from route3_v2data_module2_infer import (GROUPS, competitor_state, subsets, yield_gpu)
from route3_v2data_stage2_train import INITIAL, sample
from route3_v2data_stage3a import gpu_setup, thin_surface_ids
from route3_v2data_stage3a_prior_diagnostic import gt_regions
from route3_v2data_stage3b import load_geometry
from src.ovg_connected_depth import DIRECTIONS, edge_slices, relative_jump
from src.ovg_tear_head import VARIANTS, TearHead, tear_features

WEIGHTS = np.array((1., 1., 1., 2.))
THRESHOLDS = np.arange(1, 10) / 10
FEATURE_CACHE = RUN/"features"


def progress(**values):
    values["at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    save_json(RUN/"training-progress.json", values)


def observations(selected):
    return [(unit, frame) for unit in selected for frame in FRAMES]


def frame_masks(status, rho, zones):
    # 1 is torn, 2/3 are the two connected groups, and 4 identifies additional
    # supervision of tears on rods and in gaps. Zero is invalid or out of frame.
    labels = np.zeros_like(status, dtype=np.uint8)
    regions = labels.copy()
    for slot, (dy, dx) in enumerate(DIRECTIONS):
        a, b = edge_slices(dy, dx)
        local = status[slot][a]
        false_step = relative_jump(rho[a], rho[b]) > .10
        rod = zones["foreground"][a] | zones["foreground"][b]
        gap = (zones["gap_background"][a] | zones["gap_background"][b]) & ~rod
        group = np.where(local == 1, np.where(rod | gap, 4, 1),
                         np.where(local == 0, np.where(false_step, 2, 3), 0))
        labels[slot][a] = group
        regions[slot][a] = np.where(rod, 0, np.where(gap, 1, 2))
    return labels, regions


def feature_paths(unit, frame):
    prefix = FEATURE_CACHE / unit.name / f"f{frame:03d}"
    return prefix.with_suffix(".npy"), prefix.with_suffix(".npz")


def write_features(model, units, labels, phase, limit=None):
    done = 0
    waited = 0.
    for unit, frame in units[:limit]:
        path, meta = feature_paths(unit, frame)
        if not meta.is_file() or not path.is_file():
            waited += yield_gpu()
            item = ("B", unit.name, ovg.TRAIN_FRAMES.index(frame))
            x, _ = sample(item)
            p = model(x, protect_residual=True, render_heads=False, return_features=True)
            inverse = p["depth1"].reciprocal()
            features = tear_features(p["features"], x["rgb"], inverse)
            status = ovg.load(labels/unit.name/f"f{frame:03d}.npz")["status"]
            reference = ovg.b_foreground_reference(unit.name, frame)
            if phase == "development":
                zones = subsets(unit, unit.name, frame, reference)
            else:
                ids = thin_surface_ids(unit)
                if not ids and json.loads((unit/"scene-spec.json").read_text())["category"] == "thin_structure":
                    ids = reference["foreground"]
                zones, _ = gt_regions(reference, ids)
            groups, regions = frame_masks(status, inverse[0].cpu().numpy(), zones)
            if not torch.isfinite(features).all():
                raise ValueError(f"Nonfinite frozen tear features in {unit.name} frame {frame}")
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix(".tmp.npy")
            np.save(temp, features[0].permute(1, 2, 0).half().contiguous().cpu().numpy())
            os.replace(temp, path)
            temp = meta.with_suffix(".tmp.npz")
            np.savez_compressed(temp, groups=groups, regions=regions)
            os.replace(temp, meta)
        done += 1
        if done % 12 == 0 or done == len(units[:limit]):
            progress(phase="native_features", partition=phase, frames=done, total=len(units[:limit]),
                     competitor=competitor_state())
    return waited


def cached(unit, frame):
    path, meta = feature_paths(unit, frame)
    with np.load(meta) as data:
        return np.load(path, mmap_mode="r"), data["groups"], data["regions"]


def batch(features, groups, side, cap, balance, rng):
    limit = groups.shape[-1] - side
    if balance and np.any((groups == 1) | (groups == 4)):
        _, yy, xx = np.where((groups == 1) | (groups == 4))
        i = rng.integers(len(yy))
        y = rng.integers(max(0, yy[i]-side+2), min(limit, yy[i]-1)+1) if 0 < yy[i] < groups.shape[-2]-1 else int(np.clip(yy[i]-side//2, 0, limit))
        x = rng.integers(max(0, xx[i]-side+2), min(limit, xx[i]-1)+1) if 0 < xx[i] < groups.shape[-1]-1 else int(np.clip(xx[i]-side//2, 0, limit))
    else:
        y, x = rng.integers(0, limit+1, size=2)
    loc = np.s_[y:y+side, x:x+side]
    selected = groups[:, loc[0], loc[1]].copy()
    for slot, (dy, dx) in enumerate(DIRECTIONS):
        if dy:
            selected[slot, -1] = 0
        if dx:
            selected[slot, :, -1 if dx > 0 else 0] = 0
    flat = selected.ravel()
    masks = [np.flatnonzero((flat == 1) | (flat == 4)),
             np.flatnonzero(flat == 2), np.flatnonzero(flat == 3), np.flatnonzero(flat == 4)]
    indices = [rng.choice(i, cap, replace=False) if len(i) > cap else i for i in masks]
    packed = np.ascontiguousarray(features[loc].transpose(2, 0, 1))
    image = torch.as_tensor(packed, device="cuda", dtype=torch.float32)[None]
    return image, [torch.as_tensor(i, device="cuda") for i in indices], (int(y), int(x))


def apply_loss(model, image, uses):
    logits = model(image[:, :model[0].in_channels]).reshape(-1)
    loss = logits.new_zeros(())
    for group, (weight, idx) in enumerate(zip(WEIGHTS, uses)):
        if idx.numel():
            picked = logits[idx]
            target = torch.full_like(picked, float(group in (0, 3)))
            loss = loss + weight*F.binary_cross_entropy_with_logits(picked, target)
    return loss


def curves(bins):
    """Summarize edge scores at all nine thresholds using strict t > threshold."""
    counts = bins.sum(1)
    values = []
    for k, threshold in enumerate(THRESHOLDS, 1):
        above = bins[:, k:].sum(1)
        torn, false_steps, connected, critical = map(int, counts)
        true_pos, false_stepped, false_other, true_critical = map(int, above)
        false_pos = false_stepped + false_other
        precision = true_pos/max(1, true_pos+false_pos)
        recall = true_pos/max(1, torn)
        fpr = false_pos/max(1, false_steps+connected)
        errors = np.array((1-recall, false_stepped/max(1, false_steps),
                           false_other/max(1, connected), 1-true_critical/max(1, critical)))
        if not np.all(counts):
            raise ValueError(f"An edge-validation group is empty: {counts}")
        values.append(dict(threshold=float(threshold), precision=precision, recall=recall,
                           balanced_precision=recall/max(1e-12, recall+fpr),
                           weighted_error=float(np.average(errors, weights=WEIGHTS)),
                           error_by_group=errors.tolist(), true_positive=true_pos, false_positive=false_pos))
    return dict(counts=counts.tolist(), at_half=values[4], curve=values)


@torch.inference_mode()
def evaluate(models, frames, partition):
    bins = torch.zeros(len(models), 4, 10, device="cuda", dtype=torch.int64)
    counts = torch.zeros(len(models), 3, 4, device="cuda", dtype=torch.int64)
    thresholds = torch.as_tensor(THRESHOLDS, device="cuda", dtype=torch.float32)
    waited = 0.
    for i, (unit, frame) in enumerate(frames):
        if i % 6 == 0:
            waited += yield_gpu()
        ft, groups, regions = cached(unit, frame)
        packed = np.ascontiguousarray(ft.transpose(2, 0, 1))
        image = torch.as_tensor(packed, device="cuda", dtype=torch.float32)[None]
        group = torch.as_tensor(groups, device="cuda").reshape(-1)
        zone = torch.as_tensor(regions, device="cuda").reshape(-1)
        torn = (group == 1) | (group == 4)
        for j, model in enumerate(models.values()):
            prob = model(image[:, :model[0].in_channels]).reshape(-1).sigmoid()
            if not torch.isfinite(prob).all():
                raise ValueError(f"Nonfinite {partition} edge probabilities")
            # bucketize(right=False) assigns exact thresholds to the negative side.
            score = torch.bucketize(prob, thresholds)
            for k, mask in enumerate((torn, group == 2, group == 3, group == 4)):
                bins[j, k] += torch.bincount(score[mask], minlength=10)
            predicted = prob > .5
            for k in range(3):
                inside = zone == k
                gt_step = (group == 2) & inside
                gt_tear = torn & inside
                counts[j, k] += torch.stack((gt_step.sum(), (gt_step & predicted).sum(),
                                            gt_tear.sum(), (gt_tear & ~predicted).sum()))
    by_model = {}
    for name, histogram, rows in zip(models, bins.cpu().numpy(), counts.cpu().tolist()):
        report = curves(histogram)
        report["regions"] = {
            name: dict(false_step_edges=ns, falsely_torn=fs, torn_edges=nt, falsely_connected=fc,
                       false_step_error=fs/ns if ns else None, false_link_error=fc/nt if nt else None)
            for name, (ns, fs, nt, fc) in zip(GROUPS, rows)}
        by_model[name] = report
    return by_model, waited


def snapshot(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def run():
    config = json.loads(CONFIG.read_text())
    cfg = config["training"]
    if not (RUN/"training-labels.json").is_file():
        raise FileNotFoundError("Generate the registered training labels before starting")
    geometry_sha = sha256(INITIAL)
    if not geometry_sha.startswith(config["frozen_geometry"]["sha256_prefix"]):
        raise ValueError(f"Frozen OVG EMA checksum differs: {geometry_sha}")
    label_report = json.loads((RUN/"training-labels.json").read_text())
    train_units = {p.name: p for p in library_unit_paths("train")}
    chosen = [train_units[row["scene"]] for row in label_report["rows"]]
    train_frames = observations(chosen)
    dev_frames = observations(library_unit_paths("dev"))
    if len(dev_frames) != 45:
        raise ValueError("The sealed development split must supply all 45 frames")
    needing = sum(not all(p.is_file() for p in feature_paths(u, f)) for u, f in train_frames+dev_frames)
    if shutil.disk_usage(RUN).free < 60*GIB + needing*(512**2*75*2 + 2*GIB//1000):
        raise OSError("Insufficient disk reserve for the frozen native feature cache")
    gpu_setup(config)
    began = time.monotonic()
    competitor_begin = competitor_state()
    waited = yield_gpu()
    torch.manual_seed(cfg["seed"])
    with torch.inference_mode():
        geometry = load_geometry()
        waited += write_features(geometry, train_frames, CACHE, "training")
        waited += write_features(geometry, dev_frames, DEV_LABELS, "development")
    del geometry
    torch.cuda.empty_cache()
    feature_seconds = time.monotonic()-began

    models = {name: TearHead(n).cuda() for name, n in VARIANTS.items()}
    with torch.no_grad():
        full, nodepth = models.values()
        nodepth[0].weight.copy_(full[0].weight[:, :VARIANTS["d_nodepth"]])
        nodepth[0].bias.copy_(full[0].bias)
        nodepth[2].load_state_dict(full[2].state_dict())
    ema = {key: deepcopy(model).eval().requires_grad_(False) for key, model in models.items()}
    parameters = [p for model in models.values() for p in model.parameters()]
    totals = {key: sum(p.numel() for p in model.parameters()) for key, model in models.items()}
    if totals != config["architecture"]["parameter_counts"]:
        raise ValueError(f"Tear architecture does not match registration: {totals}")
    optimizer = torch.optim.Adam(parameters, lr=cfg["lr_start"])
    steps = cfg["steps"]
    period = cfg["report_every"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps, eta_min=cfg["lr_end"])
    rng = np.random.default_rng(cfg["seed"])
    monitor_rng = np.random.default_rng(cfg["seed"]+1)
    monitor = [train_frames[i] for i in monitor_rng.choice(len(train_frames), cfg["train_monitor_frames"], replace=False)]
    history, best = [], {}
    scene_counts, group_draws = Counter(), np.zeros(4, dtype=np.int64)
    sums = dict.fromkeys(models, 0.)
    train_began = time.monotonic()
    for step in range(1, steps+1):
        unit, frame = train_frames[rng.integers(len(train_frames))]
        features, groups, _ = cached(unit, frame)
        image, uses, _ = batch(features, groups, cfg["crop_size"], cfg["max_edges_per_group"], step % 2 == 0, rng)
        scene_counts[unit.name] += 1
        group_draws += [use.numel() for use in uses]
        optimizer.zero_grad(set_to_none=True)
        losses = {key: apply_loss(model, image, uses) for key, model in models.items()}
        for key, loss in losses.items():
            if not loss.isfinite():
                raise ValueError(f"Nonfinite tear loss in {key} at step {step}")
            sums[key] += loss.item()
        sum(losses.values()).backward()
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            for key, model in models.items():
                for avg, param in zip(ema[key].parameters(), model.parameters()):
                    avg.lerp_(param, 1-cfg["ema"])
        if step % 50 == 0:
            waited += yield_gpu()
            progress(phase="training", step=step, total=steps, elapsed_seconds=time.monotonic()-train_began,
                     gpu_wait_seconds=waited, competitor=competitor_state())
        if step % period:
            continue
        train, wait_train = evaluate(ema, monitor, "training")
        dev, wait_dev = evaluate(ema, dev_frames, "development")
        waited += wait_train+wait_dev
        row = dict(step=step, lr=optimizer.param_groups[0]["lr"],
                   loss={k: v/period for k, v in sums.items()}, training=train, development=dev,
                   train_seconds=time.monotonic()-train_began)
        history.append(row)
        for key in models:
            half = train[key]["at_half"]
            if min(half["balanced_precision"],half["recall"]) < .1:
                save_json(RUN/"training-collapse.json", row)
                raise ValueError(f"Balanced training edge precision/recall indicates collapse: {key} step {step}")
            error = dev[key]["at_half"]["weighted_error"]
            path = RUN/key/f"step-{step:06d}.pt"
            snapshot(dict(step=step, ema=ema[key].state_dict(), raw=models[key].state_dict(),
                          development_weighted_error=error, geometry_sha256=geometry_sha), path)
            if key not in best or error < best[key]["error"]:
                best[key] = dict(step=step, error=error, path=str(path))
        sums = dict.fromkeys(models, 0.)
        save_json(RUN/"training-curve.json", dict(history=history, selected=best))
        print("TEAR_EDGE_VALIDATION", step, {k:r["at_half"]["weighted_error"] for k,r in dev.items()}, flush=True)
    save_json(RUN/"training-report.json", dict(history=history, selected=best, parameter_counts=totals,
        feature_seconds=feature_seconds, train_seconds=time.monotonic()-train_began,
        wall_seconds=time.monotonic()-began, peak_allocated_gib=torch.cuda.max_memory_allocated()/GIB,
        peak_reserved_gib=torch.cuda.max_memory_reserved()/GIB, gpu_wait_seconds=waited,
        sampled_scenes=dict(scene_counts), sampled_edges_by_group=group_draws.tolist(),
        geometry_sha256=geometry_sha, competitor_start=competitor_begin, competitor_end=competitor_state()))
    progress(phase="trained", step=steps, total=steps)


if __name__ == "__main__":
    run()
