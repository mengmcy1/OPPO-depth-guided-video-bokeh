"""Qualify and train the joint Stage 3c geometry and Query run."""
import argparse
from collections import Counter
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

import route3_ovg_data as ovg
from route3_ovg_footprint import query_net
from route3_v2data_B_batch import ROOT, sha256, write_json
from route3_v2data_B_plan import load
from route3_v2data_B_training_share import register_training
from route3_v2data_dryrun import stage_objective
from route3_v2data_loader import TrainingData
from route3_v2data_stage1 import verify_disk
from route3_v2data_stage2_train import INITIAL, QUERY, now
from route3_v2data_stage3a import FIXED, OUT as A_RUN, Placement, choose_crop, group_norms, trainable_geometry
from route3_v2data_stage3b import (OUT as B_RUN, SEED, STAGE3A, BOptical, DirectBatches, atomic_checkpoint,
                                  draw_optical, frozen_inputs, gpu_setup, gradient_norms, load_geometry,
                                  optical_pool, optical_specs, path_for, read_checkpoint)

OUT=ROOT/"outputs/route3-v2data-stage3c-2026-11-26-01"
RECORD=ROOT/"03_实验记录/records/route3-v2data-stage3c-config-2026-11-26.json"
SCHEDULE=(0,5000,10000)
GEOMETRY_LR=1e-6
QUERY_LR=6e-6
QUERY_END_LR=6e-7
EMA=.999
TRAIN_GIB=15
REFRESH=500
GUARDS=dict(inverse_depth_rmse=.05,same_surface_step=.10,near_subject_coc_mae=.10,
            GT_occlusion_crossing_abs_p95=.10)
DB_FLOORS=dict(zero_visible=50.,zero_combined=60.)


def inputs():
    source=frozen_inputs()
    if load(B_RUN/"adoption.json")["selected_step"]!=0:
        raise ValueError("Stage 3b did not retain its registered starting Query")
    if not source["geometry"]["sha256"].startswith("dfb1d7e5") or not source["query"]["sha256"].startswith("1c19181f"):
        raise ValueError("Stage 3c starting checkpoints differ from the registered hashes")
    return source


def auxiliary_weights():
    return {key:value for key,value in load(STAGE3A)["loss_weights"].items() if key!="bokeh"}


def direct_weight():
    weight=load(B_RUN/"config.json")["training"]["direct_weight"]
    if not math.isclose(weight,6.8709e-6,rel_tol=1e-5):
        raise ValueError("Unexpected registered Stage 3b direct weight")
    return weight


def setup(run):
    run.mkdir(parents=True,exist_ok=True)
    verify_disk(run)
    gpu_setup(dict(training=dict(max_process_gpu_gib=TRAIN_GIB)))
    frozen=load_geometry()
    geometry=load_geometry()
    parameters,groups=trainable_geometry(geometry,2)
    if len(groups)!=14 or sum(p.numel() for p in parameters)!=648650:
        raise ValueError("Stage 3c geometry differs from the registered Stage 3a v2 depth path")
    query=query_net(QUERY).requires_grad_(True)
    optics=BOptical(run,frozen,reference_run=B_RUN)
    direct=DirectBatches(run,frozen)
    place=Placement(run/"hidden-parents",A_RUN/"hidden-parents")
    return geometry,query,parameters,groups,optics,direct,place


def auxiliary(geometry,inventory,item,spec,state,placer,rng,weights):
    pixel_rng=torch.Generator(device="cuda").manual_seed(int(rng.integers(2**32)))
    terms,counts,_=stage_objective(geometry,inventory,item,"3a",pixel_rng,spec=spec,
                                    stage3_place=placer,prepared=state,auxiliary_only=True)
    return sum(weights[k]*v for k,v in terms.items()),terms,counts


def backward(loss,step):
    if not torch.isfinite(loss):
        raise FloatingPointError(f"Nonfinite Stage 3c loss at step {step}")
    loss.backward()


def predict_batch(geometry,optics,spec):
    x,y,_,_=optics.frame(spec["unit"],spec["frame"])
    return (x,y),geometry(x,protect_residual=True,render_heads=False)


def qualify(run):
    hashes=inputs()
    geometry,query,parameters,groups,optics,direct,placer=setup(run)
    inventory=TrainingData()
    receipt=load(B_RUN/"gradient-check.json")
    name=receipt["scene"]
    spec=next(s for s in optical_specs(inventory,name) if s["truth"].stem==receipt["case"])
    crop=tuple(receipt["crop"])
    item=("B",name,ovg.TRAIN_FRAMES.index(spec["frame"]))
    manifest=direct.manifest(name,spec["unit"])
    rng=np.random.default_rng([SEED,3,2])
    all_parameters=(*parameters,*query.parameters())

    def norm(loss):
        grad=torch.autograd.grad(loss,all_parameters,allow_unused=True)
        return dict(geometry=group_norms(grad[:len(parameters)],groups),
                    Query=gradient_norms(query,grad[len(parameters):]))

    torch.cuda.reset_peak_memory_stats()
    begin=time.monotonic()
    _,pred=predict_batch(geometry,optics,spec)
    optical,support=optics.batch(query,item,spec,crop,count_rays=True,current=pred)
    b=norm(optical)
    bokeh_value=float(optical.detach())
    del optical,pred
    direct_loss,direct_terms=direct.loss(query,manifest,rng,spec)
    d=norm(direct_loss)
    a,terms,counts=auxiliary(geometry,inventory,item,spec,predict_batch(geometry,optics,spec),placer,rng,auxiliary_weights())
    g=norm(a)
    matrix=dict(L_bokeh=b,direct_unweighted=d,geometry_auxiliary_weighted=g)
    all_values=[value for loss in matrix.values() for group in loss.values() for value in group.values()]
    passed=(all(math.isfinite(v) for v in all_values)
            and all(v>0 for block in b.values() for v in block.values())
            and all(v>0 for v in d["Query"].values())
            and any(v>0 for v in g["geometry"].values())
            and not any(d["geometry"].values()) and not any(g["Query"].values()))
    result=dict(passed=passed,inputs=hashes,scene=name,case=spec["truth"].stem,crop=crop,
                support=support,auxiliary_support=counts,gradient_norms=matrix,
                losses=dict(L_bokeh=bokeh_value,direct=float(direct_loss.detach()),auxiliary=float(a.detach()),
                            direct_terms=direct_terms,geometry_terms={k:float(v.detach()) for k,v in terms.items()}),
                geometry_parameters={k:sum(p.numel() for p in v) for k,v in groups.items()},
                Query_parameter_count=sum(p.numel() for p in query.parameters()),
                seconds=time.monotonic()-begin,peak_reserved_bytes=torch.cuda.max_memory_reserved(),at=now())
    write_json(run/"gradient-check.json",result)
    print("STAGE3C_GRADIENT",json.dumps(result),flush=True)
    if not passed:
        raise RuntimeError("Joint gradient qualification failed")
    return result


def registration(run):
    hashes=inputs()
    grad=load(run/"gradient-check.json")
    zero=load(run/"zero-consistency.json")
    if not grad.get("passed") or not zero.get("passed") or grad["inputs"]!=hashes:
        raise ValueError("Register only after the passing gradient and zero checks")
    result=dict(stage="3c",created_at=now(),inputs=hashes,renderer="V0, b=0, actual Query, O1 footprint order, F2",
                training=dict(steps=SCHEDULE[-1],eval_steps=SCHEDULE,weights=["ema","model"],
                              geometry_lr=GEOMETRY_LR,query_lr=QUERY_LR,query_end_lr=QUERY_END_LR,
                              optimizer="Adam",geometry_gradient_clip=5.,query_gradient_clip=1.,ema=EMA,
                              fraction_B_optical=.5,fraction_C_direct=.5,native_crop=[256,256],
                              required_thin_fraction_of_B=.5,refresh_steps=REFRESH,seed=SEED,max_process_gpu_gib=TRAIN_GIB,
                              divergence="nonfinite loss/gradient, or three consecutive 100-B-batch medians above max(0.05, 10 × first-window optical median)"),
                losses=dict(optical=1.,Query_direct=direct_weight(),geometry_auxiliary=auxiliary_weights(),
                            auxiliary_zero="Stage 3a v2 256-pixel pinhole composition with detached GT substitute; its loss updates geometry only.",
                            batch_application="B: K64 actual-Query LDR optical + 3a auxiliary geometry + paired direct Query. C: 3b direct Query only."),
                placement="Frozen 3b training candidate inputs and annular membership; admit new B frames once using initial geometry. Reorder the predicted surfels against detached current depth; Query gradients remain enabled.",
                data=dict(training_optical_at_registration=len(optical_pool(TrainingData())["scenes"]),test_cases_used=0),
                gradient_check=grad,step0_consistency=zero,
                evaluation=dict(fixed_cases=FIXED,relative_geometry_limits=GUARDS,zero_floors_db=DB_FLOORS,
                                min_far_hidden_fraction_vs_start=.8,descriptive_equivalence_db=.1,
                                selection="Among trained EMA checkpoints at 5k/10k with all guardrails, maximize pooled near-focus Query-on score (earlier ties). Adopt only if both scene-paired median changes of the absolute Query-on near/far scores are nonnegative. Step 0 is the fallback.",
                                provenance="FP16-masked GT placement; recompute inference candidates from current target geometry and current P2 focal distance, using the same observed source views and memory protocol as Stage 3b."),
                code_sha256={name:sha256(ROOT/"scripts"/name) for name in
                    ("route3_v2data_stage3c.py","route3_v2data_stage3c_eval.py","route3_v2data_dryrun.py",
                     "route3_v2data_stage3b.py","route3_v2data_stage3b_eval.py")})
    write_json(RECORD,result)
    write_json(run/"config.json",result)


def model_states(geometry,query):
    saved={name:p for name,p in geometry.state_dict().items() if not name.startswith("encoder.")}
    return dict(geometry=saved,query=query.state_dict())


def train(run,through):
    cfg=load(run/"config.json")
    remote_contents=subprocess.check_output(["git","show",f"HEAD:{RECORD.relative_to(ROOT)}"],cwd=ROOT)
    if not cfg or cfg!=json.loads(remote_contents) or cfg["inputs"]!=inputs():
        raise ValueError("Commit the exact qualified Stage 3c configuration before training")
    for name,digest in cfg["code_sha256"].items():
        if sha256(ROOT/"scripts"/name)!=digest:
            raise ValueError(f"Registered Stage 3c code changed: {name}")
    geometry,query,parameters,_,optics,direct,placer=setup(run)
    q_params=tuple(query.parameters())
    opt=torch.optim.Adam([dict(params=parameters,lr=GEOMETRY_LR),dict(params=q_params,lr=QUERY_LR)])
    named={group:{name:p for name,p in network.named_parameters() if p.requires_grad}
           for group,network in (("geometry",geometry),("query",query))}
    ema={group:{name:p.detach().clone() for name,p in values.items()} for group,values in named.items()}
    counts=Counter()
    progress=run/"progress.json"
    digest=sha256(run/"config.json")
    begin=time.monotonic()
    initial_window=None
    trending=[]
    overflow_windows=0
    start,prior=1,0.
    archives=sorted((run/"checkpoints").glob("step-*.pt"))
    if archives:
        saved=read_checkpoint(archives[-1])
        if saved["config_sha256"]!=digest:
            raise ValueError("Cannot resume 3c with a different registered configuration")
        if saved["step"]>=through:
            return
        geometry.load_state_dict(saved["model"]["geometry"],strict=False)
        query.load_state_dict(saved["model"]["query"])
        for group,values in ema.items():
            for name,p in values.items():
                p.copy_(saved["ema"][group][name])
        opt.load_state_dict(saved["optimizer"])
        counts.update(saved["counts"])
        initial_window,trending,overflow_windows=saved["divergence_state"]
        start,prior=saved["step"]+1,saved["seconds"]
    write_json(progress,dict(step=start-1,phase="3c",at=now(),seconds=prior))
    register_training("3c",{"3c":os.getpid()},{"3c":str(progress)})
    try:
        for step in range(start,through+1):
            boundary=((step-1)//REFRESH)*REFRESH
            if step==start or step==boundary+1:
                verify_disk(run)
                inventory=TrainingData()
                target=run/"reference-pools"/f"step-{boundary:06d}.json"
                target.parent.mkdir(exist_ok=True)
                if not target.is_file():
                    write_json(target,optical_pool(inventory))
                pool=load(target)
            rng=np.random.default_rng([SEED,3,step])
            opt.param_groups[1]["lr"]=QUERY_END_LR+(QUERY_LR-QUERY_END_LR)*.5*(1+math.cos(math.pi*(step-1)/SCHEDULE[-1]))
            opt.zero_grad(set_to_none=True)
            query.train()
            if step%2:
                item,manifest,spec=draw_optical(inventory,pool,direct,rng,counts["B"]%2==0)
                crop,thin=choose_crop(spec,rng,counts["B"]%2==0)
                _,pred=predict_batch(geometry,optics,spec)
                b,support=optics.batch(query,item,spec,crop,current=pred)
                backward(b,step)
                b_value=float(b.detach())
                del b,pred
                q,q_terms=direct.loss(query,manifest,rng,spec)
                weighted_q=cfg["losses"]["Query_direct"]*q
                backward(weighted_q,step)
                a,a_terms,aux_support=auxiliary(geometry,inventory,item,spec,predict_batch(geometry,optics,spec),
                                              placer,rng,cfg["losses"]["geometry_auxiliary"])
                backward(a,step)
                total=b_value+float(weighted_q.detach())+float(a.detach())
                counts.update(B=1,thin_crops=int(thin>0),thin_pixels=thin)
                case=spec["truth"].stem
                counts[f"B/{item[1]}/{case}"]+=1
                trending.append(b_value)
                if len(trending)==100:
                    median=float(np.median(trending))
                    if initial_window is None:
                        initial_window=median
                    overflow_windows=overflow_windows+1 if median>max(.05,10*initial_window) else 0
                    trending.clear()
                    if overflow_windows>=3:
                        raise FloatingPointError("Persistent optical loss growth over three B windows")
            else:
                name=direct.c_names[int(rng.integers(len(direct.c_names)))]
                manifest=direct.manifest(name)
                q,q_terms=direct.loss(query,manifest,rng)
                weighted_q=cfg["losses"]["Query_direct"]*q
                backward(weighted_q,step)
                total=float(weighted_q.detach())
                b_value,case,crop,thin=None,None,None,0
                a_terms,support,aux_support={},{},{}
                counts.update(C=1)
            norms={name:float(torch.nn.utils.clip_grad_norm_(params,limit,error_if_nonfinite=True))
                   for name,params,limit in (("geometry",parameters,5.),("Query",q_params,1.))}
            opt.step()
            with torch.no_grad():
                for group,values in named.items():
                    for name,p in values.items():
                        ema[group][name].lerp_(p,1-EMA)
            elapsed=prior+time.monotonic()-begin
            row=dict(step=step,scene=manifest["scene"],case=case,crop=crop,thin_pixels=thin,gradient_norms=norms,
                     bokeh=b_value,total=total,
                     direct=q_terms,geometry_auxiliary={k:float(v.detach()) for k,v in a_terms.items()},
                     support=support,auxiliary_support=aux_support,elapsed=elapsed)
            with (run/"steps.jsonl").open("a") as stream:
                stream.write(json.dumps(row)+"\n")
            if step==1 or step%25==0:
                write_json(progress,dict(step=step,at=now(),seconds=elapsed,counts={k:counts[k] for k in ("B","C","thin_crops","thin_pixels")},
                                         peak_vram_reserved_bytes=torch.cuda.max_memory_reserved()))
                print("STAGE3C",step,row["total"],round(elapsed,2),flush=True)
            if step%500==0:
                original=model_states(geometry,query)
                shadow={key:dict(value,**ema[key]) for key,value in original.items()}
                atomic_checkpoint(path_for(run,step),dict(step=step,model=original,ema=shadow,optimizer=opt.state_dict(),
                                  counts=dict(counts),seconds=elapsed,config_sha256=digest,
                                  divergence_state=(initial_window,trending,overflow_windows)))
            del q,weighted_q,a_terms
            if step%2:
                del a
    finally:
        register_training("3c",{},{})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("qualify","register","train"))
    p.add_argument("--run",type=Path,default=OUT)
    p.add_argument("--through",type=int,choices=SCHEDULE[1:],default=SCHEDULE[-1])
    args=p.parse_args()
    if args.mode=="train":
        with (args.run/"train.lock").open("a") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            train(args.run,args.through)
    else:
        {"qualify":qualify,"register":registration}[args.mode](args.run)
