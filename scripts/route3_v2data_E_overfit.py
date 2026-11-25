"""Train the isolated E-L memorization check and report native depth diagnostics."""
import math
from collections import defaultdict
import time

import numpy as np
import torch
from torch.nn import functional as F

from route3_v2data_B_batch import GIB
from route3_v2data_E_cache import cache_arrays, cache_path, setup, write_arrays
from route3_v2data_E_labels import FRAMES, REGIONS, RUN, SEED, core_labels, load, save_json, stats, unit_paths
from route3_v2data_module2_d_train import snapshot
from route3_v2data_module2_infer import competitor_state, yield_gpu
from src.ovg_e import ELight, ORDER_RATIO, RESIDUAL_LIMIT, fixed_channels, log_mixture, rear_base

SIDE = 256
STEP = 200
WINDOW = 4
MIN_STEPS = 1600
MIN_IMPROVEMENT = .01
MODEL = RUN/"E-L-overfit.pt"
INPUTS = ("features","phi","mask","rho1","beta","attenuation")


def tensor(a):
    return torch.from_numpy(np.array(a,copy=True))


def case_data(name, frame, plan, margin):
    path=unit_paths(plan)[name]
    y=core_labels(path,frame,plan["metadata"][name])
    bundle=cache_arrays(cache_path("ovg",name,frame))
    return prepare(bundle,y,margin)


def prepare(bundle, y, margin):
    x={k:tensor(v)[None] for k,v in bundle.items() if k not in ("rho1","rho2","attenuation")}
    r1,r2,a=[tensor(bundle[k])[None,None] for k in ("rho1","rho2","attenuation")]
    m0,mv_source,mv_valid=rear_base(r1,r2,x["prior"],x["confidence"],margin)
    features,mask=fixed_channels(x["rgb"],r1,r2,x["prior"],x["confidence"])
    zone=tensor(y["zone"])[None,None].long()
    boundary=tensor(y["depth_boundary"])[None,None]
    loss_zone=torch.where((zone==2)&boundary,3,zone)
    b=dict(features=features,phi=x["phi_full"],mask=mask,rho1=r1,rho2=r2,attenuation=a,beta=m0,
           mv_source=mv_source,mv_valid=mv_valid,zone=zone,loss_zone=loss_zone)
    b.update({k:tensor(y[k])[None,None] for k in ("center","front","back","visible","valid_dual")})
    return b


def device(batch):
    return {k:v.cuda() for k,v in batch.items()}


def prediction(network,batch):
    values=[batch[k] if k!="phi" else batch[k].float() for k in INPUTS]
    with torch.autocast("cuda",dtype=torch.bfloat16):
        output=network(*values)
    return output


def masked_parts(values, mask):
    return (values*mask).sum(), mask.sum()


def loss_parts(output, batch):
    center=batch["center"].clamp_min(1e-8).log()
    nll=-log_mixture(center, output["means"],output["scales"],output["logit"])
    scores=[masked_parts(nll,batch["visible"]&(batch["loss_zone"]==i)) for i in range(4)]
    for i,field in enumerate(("front","back")):
        error=F.smooth_l1_loss(output["means"][:,i:i+1],batch[field].clamp_min(1e-8).log(),reduction="none",beta=.05)
        scores.append(masked_parts(error,batch["valid_dual"]))
    scores.append(masked_parts(F.softplus(output["order_gap"]),batch["visible"]))
    return torch.stack([a for a,b in scores]),torch.stack([b for a,b in scores])


def objective(numer, counts):
    averages=numer/counts.clamp_min(1)
    nll=averages[:4].sum()/(counts[:4]>0).sum().clamp_min(1)
    return nll+averages[4]+averages[5]+.01*averages[6]


def choose_crops(records, anchors, rng, iteration):
    batch=defaultdict(list)
    for region in (iteration%2,2):
        selection=anchors[region]
        chosen,yy,xx=selection[int(rng.integers(len(selection)))]
        y,x=int(yy),int(xx)
        y=int(rng.integers(max(0,y-SIDE+1),min(y,512-SIDE)+1))
        x=int(rng.integers(max(0,x-SIDE+1),min(x,512-SIDE)+1))
        for key,value in records[chosen].items():
            batch[key].append(value[...,y:y+SIDE,x:x+SIDE])
    return device({k:torch.cat(v,0) for k,v in batch.items()})


@torch.inference_mode()
def exact_loss(network, records):
    sums=None
    count=None
    for cached in records:
        tray=device(cached)
        a,b=loss_parts(prediction(network,tray),tray)
        sums=a if sums is None else sums+a
        count=b if count is None else count+b
    details=(sums/count.clamp_min(1)).cpu().tolist()
    return float(objective(sums,count)),details


def configuration():
    return dict(architecture="E-L, 32/48/64 plain conv/ReLU U-Net; 16 observed input channels plus 16 projected full-frame OVG channels",
        batch_size=2, crop_px=SIDE, crop_policy="One rod/gap anchor alternating, one multiview-valid anchor; choose a uniform crop containing the anchor.",
        depth_transform="Natural log inverse depth; Gaussian mixture NLL equally averages rod/gap/depth-boundary/remaining normal groups.",
        reductions="Depth1/depth2 SmoothL1 log inverse valid_dual, each weight 1 and beta .05. Order softplus weight .01 with hard clamp.",
        losses_disabled=["g_L1","segmentation","CoC","L_bokeh"],scale_floor_log=.02,
        mask_M=dict(rgb_sobel_luminance=.04,depth_sobel_log_inverse=.03,
            present_prior_pairwise_relative_difference=.10,radius_pixels=3,
            gate="M concatenated to the gate head; original 32-channel appearance/geometry branch"),
        adam_lr=.001,mixed_precision="BF16 conv activations, FP32 mixture, reductions and Adam",
        checks_every_steps=STEP, plateau_starts_at=MIN_STEPS, plateau_window_steps=STEP*WINDOW,
        plateau_best_improvement_lt=MIN_IMPROVEMENT, seed=SEED,
        training_split_only=True)


def run():
    inventory=load(RUN/"inventory.json")
    margin=load(RUN/"labels.json")["m"]
    if len(load(RUN/"cache-report.json")["checks"])!=3:
        raise ValueError("Native cached and online inference must be checked before fitting")
    cfg=configuration()
    save_json(RUN/"overfit-config.json",cfg)
    setup()
    started=time.monotonic()
    before=competitor_state()
    waited=yield_gpu()
    torch.manual_seed(SEED)
    model=ELight().cuda().to(memory_format=torch.channels_last)
    parameters=sum(p.numel() for p in model.parameters())
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg["adam_lr"])
    rng=np.random.default_rng(SEED)
    cases=[(name,frame) for name in inventory["overfit"] for frame in FRAMES]
    records=[case_data(name,frame,inventory,margin) for name,frame in cases]
    rng_anchors={k:[] for k in range(3)}
    for i,row in enumerate(records):
        valid=row["visible"][0,0].numpy()
        zone=row["zone"][0,0].numpy()
        areas=[valid&(zone==k) for k in (0,1)]+[valid&row["mv_valid"][0,0].numpy()]
        for k,mask in enumerate(areas):
            yy,xx=np.nonzero(mask)
            if len(xx):
                rng_anchors[k].append(np.column_stack((np.full_like(yy,i),yy,xx)).astype(np.int32))
    if any(not v for v in rng_anchors.values()):
        raise ValueError("The two native crop selectors lack rod, gap or multiview support")
    rng_anchors={k:np.concatenate(v) for k,v in rng_anchors.items()}
    curves=[]
    best=float("inf")
    best_step=0
    elapsed_train=0.
    step=0
    checked,terms=exact_loss(model,records)
    if not math.isfinite(checked):
        raise FloatingPointError("Nonfinite initial E-L objective")
    curves.append(dict(step=0,train_loss=checked,terms=terms,best=checked))
    while True:
        step+=1
        start=time.monotonic()
        batch=choose_crops(records,rng_anchors,rng,step)
        optimizer.zero_grad(set_to_none=True)
        output=prediction(model,batch)
        loss=objective(*loss_parts(output,batch))
        if not bool(loss.isfinite()):
            raise FloatingPointError(f"E-L loss diverged at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True)
        optimizer.step()
        elapsed_train+=time.monotonic()-start
        del batch,output,loss
        if step%50==0:
            waited+=yield_gpu()
            save_json(RUN/"overfit-progress.json",dict(step=step,best_step=best_step,best_loss=best if best_step else None,
                stage3b=competitor_state(),elapsed_seconds=time.monotonic()-started,gpu_wait_seconds=waited))
        if step%STEP!=0:
            continue
        value,means=exact_loss(model,records)
        if not math.isfinite(value):
            raise FloatingPointError(f"E-L full image objective diverged at step {step}")
        if value<best:
            best,best_step=value,step
            snapshot(dict(model=model.state_dict(),step=step,objective=value,margin=margin),MODEL)
        curves.append(dict(step=step,train_loss=value,terms=means,best=best))
        save_json(RUN/"overfit-curve.json",curves)
        print("E_OVERFIT",step,round(value,6),best_step,round(best,6),flush=True)
        if step>=MIN_STEPS and curves[-1-WINDOW]["best"]-best<MIN_IMPROVEMENT:
            break
    report=dict(config=cfg,parameters=parameters,steps=step,best_step=best_step,best_loss=best,
                wall_seconds=time.monotonic()-started,training_step_seconds=elapsed_train/step,curves=curves,
                frames=[dict(scene=n,frame=f) for n,f in cases],gpu_wait_seconds=waited,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/GIB,peak_allocated_gib=torch.cuda.max_memory_allocated()/GIB,
                competitor_start=before,competitor_end=competitor_state())
    save_json(RUN/"overfit-training.json",report)


def summary(parts):
    totals={}
    for scope, masks in parts.items():
        totals[scope]={}
        for part, series in masks.items():
            counts={k:int(sum(v)) for k,v in series["counts"].items()}
            row=dict(counts=counts)
            row.update({k:stats(v) for k,v in series["errors"].items()})
            gates=np.concatenate(series["gate"])
            row["g_quantiles"]=dict(zip(("p10","p50","p90"),map(float,np.quantile(gates,[.1,.5,.9])))) if len(gates) else None
            if counts["pixels"]:
                for k in ("front_saturated","back_saturated","M_hit","selected_back","selected_front",
                          "back_prior_used","order_clipped","both_unreachable"):
                    row[k+"_fraction"]=counts[k]/counts["pixels"]
            totals[scope][part]=row
    return totals


@torch.inference_mode()
def diagnose():
    inventory=load(RUN/"inventory.json")
    setup()
    before=competitor_state()
    start=time.monotonic()
    waited=yield_gpu()
    ck=torch.load(MODEL,weights_only=True,map_location="cpu")
    model=ELight().eval().cuda()
    model.load_state_dict(ck["model"])
    margin=ck["margin"]
    groups=defaultdict(lambda:defaultdict(lambda:dict(counts=defaultdict(list),errors=defaultdict(list),gate=[])))
    panel_ids=set(inventory["overfit"]) | {n for n in inventory["development"] if n.split("-")[1] in ("003","004","005","008")}
    cohorts={"overfit":inventory["overfit"], "development":inventory["development"]}
    for cohort,names in cohorts.items():
        for name in names:
            waited+=yield_gpu()
            for frame in FRAMES:
                b=device(case_data(name,frame,inventory,margin))
                p=prediction(model,b)
                f,g=(p["means"][:,i:i+1].exp() for i in (0,1))
                truth=b["center"]
                diff={n:v-truth for n,v in (("OVG_rho1",b["rho1"]),("OVG_rho2",b["rho2"]),
                      ("E_decoded",p["decoded"]),("E_front",f),("E_back",g),("beta",b["beta"].exp()))}
                front,back=diff["E_front"],diff["E_back"]
                diff["better_component"]=torch.where(front.abs()<=back.abs(),front,back)
                gap=RESIDUAL_LIMIT*b["attenuation"]
                logtruth=truth.clamp_min(1e-8).log()
                base=b["rho1"].log()
                lower_back=torch.minimum(b["beta"]-gap,base-gap+math.log(ORDER_RATIO))
                upper_back=torch.minimum(b["beta"]+gap,base+gap+math.log(ORDER_RATIO))
                unreachable=((logtruth<base-gap)|(logtruth>base+gap))&((logtruth<lower_back)|(logtruth>upper_back))
                flags={"front_saturated":p["raw_tanh"][:,0:1].abs()>.95,
                       "back_saturated":p["raw_tanh"][:,1:2].abs()>.95,"M_hit":b["mask"]>0,
                       "selected_back":p["selected_back"],"selected_front":~p["selected_back"],
                       "back_prior_used":b["mv_source"]>0,"order_clipped":p["order_gap"]>0,
                       "both_unreachable":unreachable}
                for zone,tag in enumerate(REGIONS):
                    common=(b["zone"]==zone)&b["visible"]
                    partitions={"all":common,"mv_valid":common&b["mv_valid"],"mv_invalid":common&~b["mv_valid"]}
                    for label,m in partitions.items():
                        actual=p["gate"][m].cpu().numpy()
                        errors={k:v[m].cpu().numpy() for k,v in diff.items()}
                        counts={"pixels":int(actual.size),**{k:int(v[m].sum()) for k,v in flags.items()}}
                        for scope in (cohort,name):
                            row=groups[scope][f"{tag}_{label}"]
                            row["gate"].append(actual)
                            for k,v in errors.items():
                                row["errors"][k].append(v)
                            for k,v in counts.items():
                                row["counts"][k].append(v)
                if name in panel_ids and frame==30:
                    arrays={n:v[0,0].cpu().numpy() for n,v in
                       (("center",truth),("rho1",b["rho1"]),("E_decoded",p["decoded"]),("E_front",f),
                        ("E_back",g),("w",p["logit"].sigmoid()),("selected_back",p["selected_back"]),
                        ("zone",b["zone"]),("mv_source",b["mv_source"]))}
                    write_arrays(RUN/"panels-data"/f"{name}-f{frame:03d}.npz",arrays)
            print("E_DIAGNOSTIC",cohort,name,flush=True)
    save_json(RUN/"depth-diagnostic.json",dict(summaries=summary(groups),frames_per_scene=list(FRAMES),
              elapsed_seconds=time.monotonic()-start,gpu_wait_seconds=waited,
              competitor_start=before,competitor_end=competitor_state()))


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase",choices=("train","diagnose"))
    args=p.parse_args()
    (run if args.phase=="train" else diagnose)()
