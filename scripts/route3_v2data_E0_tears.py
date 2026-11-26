"""Retrain the registered small tear head on frozen features of the selected E0 arm."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from functools import lru_cache
import os
import time

import numpy as np
import torch

from route3_v2data_B_batch import GIB,sha256
from route3_v2data_E_labels import FRAMES,SEED,load,save_json,unit_paths
from route3_v2data_E0_data import RUN,SIDE,dataset
from route3_v2data_E0_train import frame_batch,gpu_gate,module,predict
from route3_v2data_module2_d_labels import CACHE as LABELS, RUN as PREVIOUS, label
from route3_v2data_module2_d_train import apply_loss,evaluate,frame_masks,sample_crop,snapshot
from route3_v2data_module2_infer import competitor_state
from route3_v2data_stage3a import gpu_setup
from src.ovg_tear_head import TearHead,e_tear_features

INPUT_CHANNELS = 55
STEPS = 20_000
BATCH = 8
MAX_PER_GROUP = 2048
LIMIT = 5.5


def missing_holdout_labels():
    index=dataset()
    folders=unit_paths(index)
    missing=[n for n in index["holdout"] if any(not all(
             (LABELS/n/f"f{f:03d}").with_suffix(s).is_file() for s in (".npz",".json")) for f in FRAMES)]
    start=time.monotonic()
    with ProcessPoolExecutor(max_workers=min(4,max(1,len(os.sched_getaffinity(0))//2))) as pool:
        rows=[future.result() for future in [pool.submit(label,folders[n],FRAMES,False) for n in missing]]
    save_json(RUN/"holdout-tear-labels.json",dict(holdout=index["holdout"],missing=missing,
                                                checks=rows,cpu_seconds=time.monotonic()-start))


def features(out,rgb):
    return e_tear_features(out["d1"].float(),rgb.float(),out["decoded"],out["means"],
                          out["log_mass"][:,1:].exp())


def pack_paths(root,unit,frame):
    prefix=root/"features"/unit.name/f"f{frame:03d}"
    return prefix.with_suffix(".npy"),prefix.with_suffix(".npz")


@torch.inference_mode()
def cache_feature_pack(root,model,frames,rows):
    by_key={(r["scene"],r["frame"]):i for i,r in enumerate(rows)}
    for i,(unit,frame) in enumerate(frames,1):
        packed,labels=pack_paths(root,unit,frame)
        if packed.is_file() and labels.is_file():
            continue
        gpu_gate(LIMIT)
        batch=frame_batch(rows,[(by_key[unit.name,frame],0,0)],model.feature_mode,512)
        out=predict(model,batch)
        x=features(out,batch["features"][:,:3])
        if x.shape[1]!=INPUT_CHANNELS or not torch.isfinite(x).all():
            raise ValueError(f"Invalid frozen E tear features in {unit.name}/{frame}")
        with np.load(LABELS/unit.name/f"f{frame:03d}.npz") as gt:
            status=gt["status"]
        zone=batch["zone"][0,0].cpu().numpy()
        groups,regions=frame_masks(status,out["decoded"][0,0].cpu().numpy(),
                                    dict(foreground=zone==0,gap_background=zone==1))
        packed.parent.mkdir(parents=True,exist_ok=True)
        tmp=packed.with_suffix(".part.npy")
        np.save(tmp,x[0].permute(1,2,0).half().contiguous().cpu().numpy())
        os.replace(tmp,packed)
        tmp=labels.with_suffix(".part.npz")
        np.savez_compressed(tmp,groups=groups,regions=regions)
        os.replace(tmp,labels)
        if i%16==0:
            print("E0_TEAR_FEATURES",i,len(frames),flush=True)


def split(index):
    check=load(PREVIOUS/"training-labels.json")
    prior=[r["scene"] for r in check["rows"]]
    if len(prior)!=107:
        raise ValueError("The 107-scene tear training inventory changed")
    folders=unit_paths(index)
    held=set(index["holdout"])
    train=[(folders[s],f) for s in prior if s not in held for f in FRAMES]
    valid=[(folders[s],f) for s in index["holdout"] for f in FRAMES]
    return train,valid,sorted(held&set(prior))


def load_model(mode,step,*,run=RUN):
    ck=run/mode/f"step-{step:06d}.pt"
    state=torch.load(ck,map_location="cpu",weights_only=True)
    network=module(mode,scheme_a=state.get("scheme_a",False)).eval().requires_grad_(False)
    network.load_state_dict(state["ema"])
    return network,ck


def train(mode,step,*,run=RUN,destination=None):
    rows=load(run/"data.json")["rows"]
    index=dataset()
    train_frames,holdout,excluded=split(index)
    root=destination if destination is not None else run/mode/f"tear-{step:06d}"
    result=root/"training.json"
    if result.is_file():
        return load(result)
    if not (RUN/"holdout-tear-labels.json").is_file():
        missing_holdout_labels()
    gpu_setup(dict(training=dict(max_process_gpu_gib=LIMIT)))
    waited=gpu_gate(LIMIT)
    began=time.monotonic()
    before=competitor_state()
    network,source=load_model(mode,step,run=run)
    root.mkdir(parents=True,exist_ok=True)
    protocol=dict(E_sha256=sha256(source),train_frames=len(train_frames),holdout_frames=len(holdout),
                  excluded_overlap=excluded,steps=STEPS,batch=BATCH,inputs=INPUT_CHANNELS,seed=SEED,
                  loss_weights=[1,1,1,2],lr_start=.001,lr_end=.00001,ema=.999,
                  selection="Minimize the mean of weighted edge error and global false-step tear rate; ties use weighted error, then earliest step")
    save_json(root/"config.json",protocol)
    cache_feature_pack(root,network,[*train_frames,*holdout],rows)
    del network
    torch.cuda.empty_cache()

    @lru_cache(maxsize=48)
    def cached(unit,frame):
        packed,labels=pack_paths(root,unit,frame)
        with np.load(labels) as gt:
            return np.load(packed,mmap_mode="r"),gt["groups"],gt["regions"]

    torch.manual_seed(SEED)
    head=TearHead(INPUT_CHANNELS).cuda().to(memory_format=torch.channels_last)
    shadow=deepcopy(head).eval().requires_grad_(False)
    params=list(head.parameters())
    ema=list(shadow.parameters())
    opt=torch.optim.Adam(params,lr=.001)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,STEPS,eta_min=.00001)
    rng=np.random.default_rng(SEED)
    host=torch.empty((BATCH,SIDE,SIDE,INPUT_CHANNELS),pin_memory=True,dtype=torch.float16)
    curve=[]
    best=None
    active=0.
    step0=0
    resume=root/"resume.pt"
    if resume.is_file():
        ck=torch.load(resume,map_location="cpu",weights_only=True)
        head.load_state_dict(ck["model"])
        shadow.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["optimizer"])
        schedule.load_state_dict(ck["scheduler"])
        rng.bit_generator.state=ck["rng"]
        step0,active,curve,best=(ck[k] for k in ("step","active","curve","best"))
    for i in range(step0+1,STEPS+1):
        if i%25==1:
            waited+=gpu_gate(LIMIT)
        started=time.monotonic()
        indices=[[] for _ in range(4)]
        for j,index in enumerate(rng.integers(len(train_frames),size=BATCH)):
            image,groups,_=cached(*train_frames[index])
            bounds,slots=sample_crop(groups,SIDE,MAX_PER_GROUP,(i+j)%2==0,rng)
            host.numpy()[j]=image[bounds]
            for target,take in zip(indices,slots):
                target.append(take + j*4*SIDE**2)
        x=host.to("cuda").permute(0,3,1,2).float()
        edges=[torch.as_tensor(np.concatenate(g),device="cuda") for g in indices]
        opt.zero_grad(set_to_none=True)
        v=apply_loss(head,x,edges)
        if not torch.isfinite(v):
            raise FloatingPointError(f"E tear loss diverged at {i}")
        v.backward()
        opt.step()
        schedule.step()
        with torch.no_grad():
            torch._foreach_lerp_(ema,params,.001)
        active+=time.monotonic()-started
        if i%100==0:
            save_json(root/"progress.json",dict(step=i,seconds=time.monotonic()-began,
                                             mean_step_s=active/i,gpu_wait_seconds=waited))
        if i%1000:
            continue
        assessment,wait=evaluate({"E":shadow},holdout,"holdout",load_frame=cached,gpu_check=gpu_gate)
        waited+=wait
        score=assessment["E"]
        at=score["at_half"]
        rank=((at["weighted_error"]+at["error_by_group"][1])/2,at["weighted_error"],i)
        summary=dict(step=i,loss=float(v),rank=list(rank),assessment=score)
        curve.append(summary)
        if best is None or rank<tuple(best["rank"]):
            best=summary
            snapshot(dict(ema=shadow.state_dict(),step=i,source_sha256=protocol["E_sha256"]),root/"selected.pt")
        save_json(root/"curve.json",curve)
        snapshot(dict(model=head.state_dict(),ema=shadow.state_dict(),optimizer=opt.state_dict(),
                      scheduler=schedule.state_dict(),rng=rng.bit_generator.state,
                      step=i,active=active,curve=curve,best=best),resume)
        print("E0_TEAR_STEP",mode,i,rank,flush=True)
    out=dict(config=protocol,selected=best,curve=curve,parameters=sum(p.numel() for p in params),
             active_step_seconds=active/STEPS,wall_seconds=time.monotonic()-began,peak_reserved_gib=torch.cuda.max_memory_reserved()/GIB,
             gpu_wait_seconds=waited,competitor_before=before,competitor_after=competitor_state())
    save_json(result,out)
    return out


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("labels","E-L","E-H1","E-H2"))
    p.add_argument("--step",type=int)
    arg=p.parse_args()
    missing_holdout_labels() if arg.mode=="labels" else train(arg.mode,arg.step)
