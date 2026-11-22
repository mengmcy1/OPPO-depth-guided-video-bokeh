"""Paired 13.15.6 Stage 1 training with a common deterministic batch schedule."""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR","1")
import cv2
import numpy as np
import torch

from route3_ovg_1b_common import CONFIG as ONE_B
from route3_ovg_r2_train import model
from route3_ovg_subpixel import CONFIG as ONE_A
from route3_v2data_B_batch import FORMAL_DATA, GIB, sha256, write_json
from route3_v2data_B_plan import load
from route3_v2data_dryrun import stage_objective
from route3_v2data_loader import MIX, ROOT, TrainingData
from route3_v2data_optics import f2_entry

SEED = ONE_B["seed"]
TOTAL = 30000
SAVE_INTERVAL = 5000
REFERENCE_INTERVAL = 5000
BRANCHES = ("original","same-surface")
RECORD = ROOT/"03_实验记录/records/route3-v2data-stage1-2026-11-22.json"
OUTPUT = ROOT/"outputs/route3-v2data-stage1-2026-11-22-01"


def verify_disk(directory):
    if shutil.disk_usage(directory).free < 60*GIB:
        raise RuntimeError("stage 1 disk reserve is below 60 GiB")


def frozen_inventory(inventory, directory):
    path = directory/"paired-inventory.json"
    with (directory/"paired-inventory.lock").open("a") as guard:
        fcntl.flock(guard,fcntl.LOCK_EX)
        if not path.is_file():
            items = {tier:rows for tier,rows in inventory.items.items() if tier!="A"}
            write_json(path,dict(items=items,g_groups=inventory.g_groups,seed=SEED))
        frozen = load(path)
    for tier,rows in frozen["items"].items():
        pairs = list(map(tuple,rows))
        if not pairs or not set(pairs).issubset(inventory.allowed[tier]):
            raise ValueError(f"missing or changed frozen training inventory for {tier}")
        inventory.items[tier] = pairs
    inventory.g_groups = [list(map(tuple,group)) for group in frozen["g_groups"]]
    return dict(path=path.name,sha256=sha256(path),sizes={k:len(v) for k,v in frozen["items"].items()})


def complete_optical(inventory):
    units = []
    manifests = sorted((FORMAL_DATA/"optical").glob("*/manifest.json"))
    for path in manifests:
        name = path.parent.name
        if name not in inventory.b:
            continue
        unit = inventory.b[name]
        record = load(path)
        if record["geometry_manifest_sha256"]!=sha256(unit/"manifest.json"):
            raise ValueError(f"stale optical geometry for {name}")
        cache = [f2_entry(unit,frame)/"manifest.json" for frame in (1,30,60)]
        if not all(path.is_file() for path in cache):
            continue
        for manifest in cache:
            entry = load(manifest)
            if entry["geometry_manifest_sha256"]!=record["geometry_manifest_sha256"]:
                raise ValueError(f"stale F2 cache for {name}")
            for filename in entry["product_sha256"]:
                if not (manifest.parent/filename).is_file():
                    raise FileNotFoundError(f"missing registered F2 product {manifest.parent/filename}")
        units.append(name)
    return units


def reference_snapshot(inventory, directory, boundary):
    pool = directory/"reference-pools"
    pool.mkdir(exist_ok=True)
    path = pool/f"step-{boundary:06d}.json"
    with (pool/"lock").open("a") as guard:
        fcntl.flock(guard,fcntl.LOCK_EX)
        if not path.is_file():
            ids = complete_optical(inventory)
            if boundary==0 and len(ids)<30:
                raise ValueError(f"1b needs 30 complete optical and F2 scenes, found {len(ids)}")
            all_frames=len(inventory.items["B"])
            reference_frames=3*len(ids)
            write_json(path,dict(step=boundary,scene_ids=ids,complete_scenes=len(ids),
                                 B_total_frames=all_frames,B_frames_with_optical_reference=reference_frames,
                                 B_reference_frame_fraction=reference_frames/all_frames))
    return load(path)


def phase_tiers(phase):
    ratios = MIX[phase]
    block_size = math.lcm(*(100//math.gcd(pct,100) for pct in ratios.values()))
    block = [tier for tier,pct in ratios.items() for _ in range(block_size*pct//100)]
    rng = np.random.default_rng([SEED, int(phase[-1]=="b")])
    return [tier for _ in range(TOTAL//block_size) for tier in rng.permutation(block)]


def batch_plan(inventory, phase, step, tier, reference):
    rng = np.random.default_rng([SEED,int(phase=="1b"),step])
    item = inventory.choose(tier,rng)
    if item is None:
        raise ValueError(f"missing training batch for {tier}")
    enabled = phase=="1b" and (tier=="C" or tier=="B" and item[1] in reference)
    specs = inventory.optical_specs(item) if enabled else []
    spec = specs[int(rng.integers(len(specs)))] if specs else None
    rng_device_seed = int(rng.integers(2**32))
    return item,spec,rng_device_seed


def update_ema(net, ema):
    with torch.no_grad():
        for name,parameter in net.named_parameters():
            if name in ema:
                ema[name].lerp_(parameter,1.0-0.999)


def checkpoint_path(output, step):
    return output/"checkpoints"/f"step-{step:06d}.pt"


def store_checkpoint(path, net, ema, opt, step, phase, branch, inventory, counts, batches, seconds,
                     reference_usage):
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    pending = path.with_name(f".{path.name}.{os.getpid()}.part")
    torch.save(dict(
        phase=phase, branch=branch, step=step, inventory=inventory, counts=dict(counts),
        batches=batches, measured_seconds=seconds,reference_usage=reference_usage,
        model={key:value.detach().cpu() for key,value in net.state_dict().items() if not key.startswith("encoder.")},
        ema={key:value.detach().cpu() for key,value in ema.items()},
        optimizer=opt.state_dict(),
    ),pending)
    pending.rename(path)


def train(args):
    base = args.run.resolve()
    base.mkdir(parents=True,exist_ok=True)
    args.output.mkdir(parents=True,exist_ok=True)
    verify_disk(base)
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="0":
        raise ValueError("assign the shared RTX 5090 explicitly")
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(1)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    total_memory = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(args.memory_gib*GIB/total_memory,0)
    torch.cuda.reset_peak_memory_stats()
    inventory = TrainingData()
    frozen = frozen_inventory(inventory,base)
    net = model("cuda").train()
    net.footprint.requires_grad_(False)
    trainable = tuple(p for p in net.parameters() if p.requires_grad)
    rate = ONE_A["learning_rate"] if args.phase=="1a" else ONE_B["learning_rate"]
    opt = torch.optim.AdamW(trainable,lr=rate)
    ema = {name:p.detach().clone() for name,p in net.named_parameters() if p.requires_grad}
    directory = args.output/"checkpoints"
    existing = sorted(directory.glob("step-*.pt"))
    source = existing[-1] if existing else args.start
    step_start, prior_seconds, counts, receipts = 1,0.0,Counter(),[]
    reference_usage = {}
    if source:
        saved = torch.load(source,weights_only=True,map_location="cpu")
        if saved["branch"]!=args.branch or saved["inventory"]!=frozen:
            raise ValueError("checkpoint does not use this branch and frozen batch inventory")
        missing, extra = net.load_state_dict(saved["model"],strict=False)
        if extra or any(not key.startswith("encoder.") for key in missing):
            raise ValueError(f"wrong checkpoint parameters: {missing}, {extra}")
        opt.load_state_dict(saved["optimizer"])
        for group in opt.param_groups:
            group["lr"]=rate
        for name in ema:
            ema[name].copy_(saved["ema"][name])
        if saved["phase"]==args.phase:
            step_start=saved["step"]+1
            prior_seconds=saved["measured_seconds"]
            counts.update(saved["counts"])
            receipts=list(saved["batches"])
            reference_usage=saved.get("reference_usage",{})
        elif args.phase!="1b" or saved["phase"]!="1a" or saved["step"]!=TOTAL:
            raise ValueError("1b requires its branch's complete 1a checkpoint")
        else:
            del saved
    weights = load(RECORD)["same_surface_weight"]
    candidate_weight = weights["selected"] if args.branch=="same-surface" else 0.0
    schedule = phase_tiers(args.phase)
    epoch = -1
    reference = set()
    began = time.monotonic()
    prior_normal = [(parameter,parameter[2:5].detach().clone()) for parameter in
                    (net.geometry.weight,net.geometry.bias)] if args.phase=="1b" else []
    if step_start > args.steps:
        return
    print(f"stage1 {args.phase} {args.branch} starting step {step_start}",flush=True)
    for step in range(step_start,args.steps+1):
        if step%100==1:
            verify_disk(base)
        if args.phase=="1b":
            boundary=min(25000,((step-1)//REFERENCE_INTERVAL)*REFERENCE_INTERVAL)
            if boundary!=epoch:
                reference=set(reference_snapshot(inventory,args.run,boundary)["scene_ids"])
                epoch=boundary
        tier=str(schedule[step-1])
        item,spec,seed=batch_plan(inventory,args.phase,step,tier,reference)
        generator=torch.Generator(device="cuda").manual_seed(seed)
        opt.zero_grad(set_to_none=True)
        terms,support,loss_weights=stage_objective(net,inventory,item,args.phase,generator,spec=spec)
        total=sum(loss_weights[name]*value for name,value in terms.items() if name!="same_surface")
        if candidate_weight and "same_surface" in terms:
            total=total+candidate_weight*terms["same_surface"]
        if not torch.isfinite(total):
            raise FloatingPointError(f"Nonfinite {args.phase} loss at step {step}: {item}")
        total.backward()
        gradient=torch.nn.utils.clip_grad_norm_(trainable,5.0,error_if_nonfinite=True)
        opt.step()
        with torch.no_grad():
            for param,normal in prior_normal:
                param[2:5].copy_(normal)
        update_ema(net,ema)
        counts[tier]+=1
        if spec:
            counts["optical_"+tier]+=1
        if args.phase=="1b" and tier=="B":
            usage=reference_usage.setdefault(str(epoch),dict(B_batches=0,B_batches_with_L_bokeh=0))
            usage["B_batches"]+=1
            usage["B_batches_with_L_bokeh"]+=int("bokeh" in terms)
        identity = [str(x) for x in item]
        if spec:
            identity += [str(spec["focus"]),str(spec["fstop"])]
        receipts.append(hashlib.sha256(json.dumps(identity).encode()).hexdigest())
        row = dict(step=step,tier=tier,total=float(total.detach()),grad_norm=float(gradient),
                   same_surface_unweighted=float(terms["same_surface"].detach()) if "same_surface" in terms else None,
                   optical=spec is not None, support_same_surface=support.get("same_surface",0))
        with (args.output/"steps.jsonl").open("a") as stream:
            stream.write(json.dumps(row)+"\n")
        if step%50==0 or step==args.steps:
            torch.cuda.synchronize()
            status=dict(step=step,phase=args.phase,branch=args.branch,counts=dict(counts),
                        seconds=prior_seconds+time.monotonic()-began,peak_cuda_reserved=torch.cuda.max_memory_reserved())
            write_json(args.output/"progress.json",status)
            print(f"stage1 {args.phase} {args.branch} step {step}",flush=True)
        if step%SAVE_INTERVAL==0 or step==args.steps:
            elapsed=prior_seconds+time.monotonic()-began
            if args.phase=="1b":
                usage=reference_usage[str(epoch)]
                directory=args.output/"reference-usage"
                directory.mkdir(exist_ok=True)
                write_json(directory/f"step-{epoch:06d}.json",dict(
                    usage,reference_step=epoch,through_step=step,branch=args.branch,
                    B_batches_L_bokeh_fraction=usage["B_batches_with_L_bokeh"]/usage["B_batches"],
                ))
            store_checkpoint(checkpoint_path(args.output,step),net,ema,opt,step,args.phase,args.branch,
                             frozen,counts,receipts,elapsed,reference_usage)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,default=OUTPUT)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--phase",choices=tuple(MIX),required=True)
    parser.add_argument("--branch",choices=BRANCHES,required=True)
    parser.add_argument("--steps",type=int,default=TOTAL)
    parser.add_argument("--start",type=Path)
    parser.add_argument("--memory-gib",type=float,required=True)
    parser.add_argument("--threads",type=int,default=4)
    args=parser.parse_args()
    if not 1<=args.steps<=TOTAL:
        parser.error(f"steps must be between 1 and {TOTAL}")
    train(args)
