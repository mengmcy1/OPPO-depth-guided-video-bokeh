"""Train the two registered d-full runs on one identical series of crop batches."""
from collections import Counter
from copy import deepcopy
from datetime import datetime
from functools import lru_cache
import hashlib
import shutil
import time

import numpy as np
import torch

from route3_ovg_r2_train import model as frozen_geometry_constructor
from route3_v2data_B_batch import GIB, sha256
from route3_v2data_B_inputs import library_unit_paths
from route3_v2data_B_plan import load
from route3_v2data_module2_d_labels import CONFIG as V1_CONFIG, ROOT, RUN as V1, save_json
from route3_v2data_module2_d_train import (apply_loss, cached, evaluate, feature_paths, observations,
                                         sample_crop, snapshot)
from route3_v2data_module2_infer import competitor_state, yield_gpu
from route3_v2data_stage2_train import INITIAL
from route3_v2data_stage3a import gpu_setup
from src.ovg_tear_head import VARIANTS, TearHead

RUN = V1.with_name("route3-v2data-module2-tear-d-v2-2026-11-25-02")
CONFIG = ROOT/"03_实验记录/records/route3-v2data-module2-tear-d-v2-config-2026-11-25.json"
ARMS = ("v2a", "v2b")
CHANNELS = VARIANTS["d_full"]


@lru_cache(maxsize=None)
def training_frame(unit, frame):
    features, groups, _ = cached(unit, frame)
    yy, xx = np.nonzero((groups == 1) | (groups == 4))[1:]
    return features, groups, (yy.astype(np.int16), xx.astype(np.int16))


def shared_batch(frames, rng, step, config, host, samples, groups_seen, digest):
    size = config["crop_size"]
    pixels = host.numpy()
    groups = ([], [], [], [])
    for j in range(config["crops_per_step"]):
        index = int(rng.integers(len(frames)))
        unit, frame = frames[index]
        feature, labels, torn_pixels = training_frame(unit, frame)
        bounds, edges = sample_crop(labels, size, config["max_edges_per_group"],
                                    (step+j) % 2 == 0, rng, torn_pixels=torn_pixels)
        pixels[j] = feature[bounds]
        samples[unit.name] += 1
        groups_seen += [len(v) for v in edges]
        descriptor = [index, bounds[0].start, bounds[1].start, *(len(x) for x in edges)]
        digest.update(np.asarray(descriptor,dtype="<i4").tobytes())
        for group, indices in zip(groups, edges):
            digest.update(indices.astype("<i4", copy=False).tobytes())
            group.append(indices + j*4*size*size)
    # Preserve the v1 half-precision feature cache and FP32 model arithmetic.
    image = host.to(device="cuda", dtype=torch.float16).permute(0, 3, 1, 2).float()
    packed = np.concatenate([np.concatenate(group) for group in groups])
    counts = [sum(len(crop) for crop in group) for group in groups]
    uses = torch.as_tensor(packed, device="cuda").split(counts)
    return image, uses


def run():
    config, original = load(CONFIG), load(V1_CONFIG)
    prior = load(V1/"training-report.json")
    admitted = load(V1/"training-labels.json")["rows"]
    if config["prior_config_sha256"] != sha256(V1_CONFIG):
        raise ValueError("The v1 data and sampling registration changed")
    geometry = sha256(INITIAL)
    if geometry != prior["geometry_sha256"]:
        raise ValueError("The v1 feature cache refers to different frozen geometry")
    cfg = dict(original["training"], **config["training"])
    inventory = {unit.name:unit for unit in library_unit_paths("train")}
    train = observations([inventory[row["scene"]] for row in admitted])
    dev = observations(library_unit_paths("dev"))
    if (len(train), len(dev)) != (321, 45) or any(
            not all(path.is_file() for path in feature_paths(unit, frame)) for unit,frame in train+dev):
        raise RuntimeError("The exact v1 feature inventory is required")
    RUN.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(RUN).free < 60*GIB:
        raise OSError("Insufficient disk reserve")
    if (RUN/"training-report.json").exists():
        raise FileExistsError("The two heads already completed training in this run")
    gpu_setup(dict(training=cfg))
    began = time.monotonic()
    before = competitor_state()
    waited = yield_gpu()
    # Reconstruct the original CPU RNG position before initializing the d-full head.
    # Reading cached native features skips the RNG consumed by the v1 OVG constructor.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(cfg["seed"])
        frozen_geometry_constructor("cpu")
        full = TearHead(CHANNELS)
    init_sha = hashlib.sha256(b"".join(p.detach().numpy().tobytes() for p in full.parameters())).hexdigest()
    if init_sha != config["initialization_sha256"]:
        raise ValueError(f"The d-full initialization differs from the original RNG replay: {init_sha}")
    models = {arm:deepcopy(full).cuda().to(memory_format=torch.channels_last) for arm in ARMS}
    ema = {arm:deepcopy(head).eval().requires_grad_(False) for arm,head in models.items()}
    parameters = {arm:sum(p.numel() for p in head.parameters()) for arm, head in models.items()}
    if any(n != original["architecture"]["parameter_counts"]["d_full"] for n in parameters.values()):
        raise ValueError("The tear head differs from v1 d-full")
    optimizer = torch.optim.Adam([p for head in models.values() for p in head.parameters()], lr=cfg["lr_start"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg["steps"], eta_min=cfg["lr_end"])
    rng = np.random.default_rng(cfg["seed"])
    monitor_rng = np.random.default_rng(cfg["seed"]+1)
    monitor = [train[i] for i in monitor_rng.choice(len(train), cfg["train_monitor_frames"], replace=False)]
    host = torch.empty((cfg["crops_per_step"],cfg["crop_size"],cfg["crop_size"],CHANNELS),
                       dtype=torch.float16, pin_memory=True)
    history, best = [], {}
    samples, groups_seen = Counter(), np.zeros(4, dtype=np.int64)
    losses = torch.zeros(len(ARMS), device="cuda")
    digest = hashlib.sha256()
    period = cfg["report_every"]

    def progress(step, phase):
        save_json(RUN/"training-progress.json", dict(phase=phase,step=step,total=cfg["steps"],
            at=datetime.now().astimezone().isoformat(timespec="seconds"), elapsed_seconds=time.monotonic()-began,
            gpu_wait_seconds=waited,competitor=competitor_state()))

    progress(0, "training")
    for step in range(1, cfg["steps"]+1):
        image, uses = shared_batch(train, rng, step, cfg, host, samples, groups_seen, digest)
        optimizer.zero_grad(set_to_none=True)
        values = torch.stack([apply_loss(models[arm], image, uses, config["loss_weights"][arm]) for arm in ARMS])
        values.sum().backward()
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            losses += values.detach()
            for arm in ARMS:
                for avg, param in zip(ema[arm].parameters(), models[arm].parameters()):
                    avg.lerp_(param, 1-cfg["ema"])
        del image, uses, values
        if step % 25 == 0:
            if not torch.isfinite(losses).all().item():
                raise ValueError(f"Nonfinite tear loss through step {step}")
        if step % 50 == 0:
            waited += yield_gpu()
            progress(step, "training")
        if step % period:
            continue
        validation = {}
        for arm, head in ema.items():
            validation[arm] = TearHead(CHANNELS).cuda().eval().requires_grad_(False)
            validation[arm].load_state_dict(head.state_dict())
        train_metrics, wait_train = evaluate(validation, monitor, "training")
        dev_metrics, wait_dev = evaluate(validation, dev, "development")
        del validation
        waited += wait_train + wait_dev
        row = dict(step=step, lr=optimizer.param_groups[0]["lr"],
                   loss=dict(zip(ARMS, (losses/period).cpu().tolist())), training=train_metrics,
                   development=dev_metrics, train_seconds=time.monotonic()-began)
        history.append(row)
        for arm in ARMS:
            metrics = train_metrics[arm]["at_half"]
            if min(metrics["balanced_precision"],metrics["recall"]) < .1:
                save_json(RUN/"training-collapse.json", row)
                raise ValueError(f"Balanced training edge precision or recall indicates collapse: {arm} step {step}")
            error = dev_metrics[arm]["at_half"]["weighted_error"]
            path = RUN/arm/f"step-{step:06d}.pt"

            # Save the original contiguous layout so validation and evaluation agree at strict t > 0.5.
            snapshot(dict(step=step, ema={k:v.contiguous() for k,v in ema[arm].state_dict().items()},
                          raw={k:v.contiguous() for k,v in models[arm].state_dict().items()},
                          development_weighted_error=error, geometry_sha256=geometry), path)
            if arm not in best or error < best[arm]["error"]:
                best[arm] = dict(step=step,error=error,path=str(path),
                    GT_torn_false_link=dev_metrics[arm]["at_half"]["error_by_group"][0],
                    regions=dev_metrics[arm]["regions"])
        losses.zero_()
        save_json(RUN/"training-curve.json", dict(history=history,selected=best))
        print("TEAR_V2_EDGE_VALIDATION", step, {arm:r["at_half"]["weighted_error"] for arm,r in dev_metrics.items()},flush=True)
    save_json(RUN/"training-report.json", dict(history=history,selected=best,parameter_counts=parameters,
              train_seconds=time.monotonic()-began,peak_allocated_gib=torch.cuda.max_memory_allocated()/GIB,
              peak_reserved_gib=torch.cuda.max_memory_reserved()/GIB,gpu_wait_seconds=waited,
              sampled_scenes=dict(samples),sampled_edges_by_group=groups_seen.tolist(),
              shared_crop_edge_sequence_sha256=digest.hexdigest(), initialization_sha256=init_sha, geometry_sha256=geometry,
              competitor_start=before,competitor_end=competitor_state()))
    progress(cfg["steps"], "trained")


if __name__ == "__main__":
    run()
