"""Train the three matched E0 configurations using the frozen native caches."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
import math
import subprocess
import time

import numpy as np
import psutil
import torch
from torch.nn import functional as F

from route3_v2data_B_batch import GIB,sha256
from route3_v2data_E_labels import SEED,load,save_json,stats
from route3_v2data_E0_data import (ANCHORS,BATCH,CONFIG,HOLDOUT,INPUTS,PREP,RUN,SIDE,STEPS,
                                   dataset,encoder_tokens,prepared_crop,window_tokens)
from route3_v2data_module2_d_train import snapshot
from route3_v2data_module2_infer import COMPETITOR,competitor_state
from route3_v2data_stage3a import gpu_setup
from src.ovg_e import ELight,FEATURE_MODES,ORDER_RATIO,REAR_MARGINS,mixture_density

WEIGHTS = dict(nll=1.,dual=1.,order=.1,segment=.1,coc=.1,gate_l1=.01)
GROUPS = ("rod","gap","normal","boundary")
EVERY = 1000
RAMP = 8000
EMA = .999
GPU_LIMIT = 5.5
TRAIN_BUDGET = {"E-L":3.,"E-H1":3.,"E-H2":GPU_LIMIT}


def gpu_gate(limit=GPU_LIMIT):
    def priority():
        pidfile = COMPETITOR/"followup.pid"
        if pidfile.is_file():
            try:
                tasks=psutil.Process(int(pidfile.read_text())).children(recursive=True)
                if any("route3_v2data_stage3b_eval.py" in " ".join(p.cmdline()) for p in tasks):
                    return "3b evaluation"
            except (ValueError,psutil.Error):
                pass
        earlier_replay=False
        for process in psutil.process_iter(["cmdline"]):
            command=process.info["cmdline"] or []
            if any("stage3c" in a and "eval" in a and a.endswith(".py") for a in command):
                return "3c evaluation"
            if any(a.endswith("route3_v2data_fp16_fit_module2.py") for a in command):
                earlier_replay=True
        used,total=(int(v) for v in subprocess.check_output(
            ["nvidia-smi","--query-gpu=memory.used,memory.total","--format=csv,noheader,nounits"],
            text=True).strip().split(","))
        reserved=torch.cuda.memory_reserved()/1024**2
        if used-reserved+1024*limit > min(total,30*1024):
            return f"shared memory {used} MiB"
        if earlier_replay and used-reserved+1024*limit>22.5*1024:
            return "earlier FP16 module-two rescore needs GPU headroom"
        return None
    cause=priority()
    if cause is None:
        return 0.
    print("E0_GPU_WAIT",cause,flush=True)
    start=time.monotonic()
    torch.cuda.empty_cache()
    while priority() is not None:
        time.sleep(8)
    return time.monotonic()-start


def frame_batch(rows, requests, mode, side=SIDE, *, labels=True):
    data=defaultdict(list)
    tokens=[]
    windows=[]
    offsets=[]
    coefficients=[]
    for index,x,y in requests:
        row=rows[int(index)]
        x,y=int(x),int(y)
        cropped=prepared_crop(row,x,y,side,labels=labels)
        for key,value in cropped.items():
            data[key].append(value)
        coefficients.append(row["CoC_K"])
        if mode!="E-L":
            tokens.append(encoder_tokens(row["scene"],row["frame"]))
            offsets.append((x,y))
        if mode=="E-H2":
            a,b=window_tokens(row["scene"],row["frame"],x,y,side)
            windows.append((torch.from_numpy(a).cuda(),torch.from_numpy(b).cuda()))
    batch={k:torch.from_numpy(np.stack(v)).cuda() for k,v in data.items()}
    batch["CoC_K"]=torch.tensor(coefficients,device="cuda")[:,None,None,None]
    if tokens:
        batch["context"]=(torch.from_numpy(np.stack(tokens)).cuda(),
                          torch.tensor(offsets,device="cuda"),windows)
    return batch


def predict(model,batch):
    args=[batch[k] if k!="mask" else batch[k].float() for k in INPUTS]
    options={"rear_source":batch["mv_source"]} if getattr(model,"scheme_a",False) else {}
    with torch.autocast("cuda",dtype=torch.bfloat16):
        return model(*args,context=batch.get("context"),**options)


def average(value,mask):
    return (value*mask).sum()/mask.sum().clamp_min(1)


def balanced(value, source):
    return sum(average(value,source["visible"]&(source["loss_zone"]==i)) for i in range(4))


def terms(out, target):
    center=target["center"].clamp_min(1e-8).log()
    nll=-mixture_density(center,out["means"],out["scales"],out["log_mass"])
    dual=sum(average(F.smooth_l1_loss(out["means"][:,i:i+1],
                     target[k].clamp_min(1e-8).log(),reduction="none",beta=.05),target["valid_dual"])
             for i,k in enumerate(("front","back")))
    cls=F.cross_entropy(out["segmentation"].float(),target["zone"][:,0].long(),reduction="none")[:,None]
    return dict(nll=balanced(nll,target),dual=dual,
                order=average(F.softplus(out["order_gap"]),target["visible"]),
                segment=average(cls,target["visible"]),
                coc=balanced((out["decoded"]-target["center"]).abs()*target["CoC_K"],target),
                gate_l1=average(out["gate"],target["visible"]&(target["loss_zone"]==2)))


def objective(values,phase):
    include=tuple(WEIGHTS) if phase==2 else ("nll","dual","order","segment")
    return sum(WEIGHTS[k]*values[k] for k in include)


def normal_front_bce(log_mass, loss_zone):
    # BCE(w_back_eff, 0) is -log(1 - w_back_eff).
    # The mixture head already computes this complement in stable FP32 log space.
    return average(-log_mass[:, :1], loss_zone == 2)


def single_back_target(out, target):
    return torch.minimum(target["center"].clamp_min(1e-8).log(),
                         out["means"][:,:1].detach()+math.log(ORDER_RATIO))


def single_back_loss(out, target, *, exclude_rod=False):
    active=target["visible"]&~target["valid_dual"]
    if exclude_rod:
        active=active&(target["loss_zone"]!=0)
    discrepancy=F.smooth_l1_loss(out["single_back_mean"],single_back_target(out,target),
                                 reduction="none",beta=.05)
    return balanced(discrepancy,dict(visible=active,loss_zone=target["loss_zone"]))


def module(mode, *, scheme_a=False):
    torch.manual_seed(SEED)
    return ELight(mode,scheme_a=scheme_a).cuda().to(memory_format=torch.channels_last)


def checked_prepared():
    data=load(RUN/"data.json")
    if sha256(RUN/"crop-order.npy")!=data["crop_order_sha256"]:
        raise ValueError("Shared E0 crop order changed")
    return data,np.load(RUN/"crop-order.npy",mmap_mode="r")


def register():
    data,_=checked_prepared()
    inventory=dataset()
    torch.set_num_threads(1)
    discover=subprocess.run(["/root/autodl-tmp/envs/route3/bin/python","-m","unittest","discover",
                             "-s","tests","-p","test_ovg_e.py","-v"],check=True,capture_output=True,text=True)
    geometry=load(PREP/"ovg-source.json")
    ks=np.array([r["CoC_K"] for r in data["rows"] if r["scene"].startswith("training-")
                 and r["scene"] not in set(inventory["holdout"])])
    pick=dict(rule="Among EMA candidates with holdout rod p50 no higher than frozen OVG, take lowest balanced NLL. Otherwise take lowest rod p50.",
               tie_key="lower rod p50, then earlier step, then E-L/E-H1/E-H2 in that order",
               development_used_for_selection=False)
    calibration=load(RUN/"qualification.json")
    cfg=dict(geometry=geometry,seed=SEED,precision="BF16 convolutions; FP32 heads, reductions and Adam",
             architecture={v:sum(p.numel() for p in ELight(v).parameters()) for v in FEATURE_MODES},
             source_margins=dict(DepthSplat=REAR_MARGINS[0],triangulation=REAR_MARGINS[1],priority="triangulation, DepthSplat, rho2"),
             effective_mass="log(w_back_eff)=logsigmoid(gate_logit)+logsigmoid(-front_logit); front via log1mexp; modes use both gated masses",
             unittests=discover.stderr.strip().splitlines(),frozen_cache_receipt=data["preparation_cache_sha256"],
             qualification=calibration,
             inputs=dict(native_ovg_channels=32,encoder="frozen Depth Anything V2 Small, layers 2/5/8/11 zero-based",
                         global_input_side=518,local_window_native_side=112,local_encoder_input_side=448,
                         coordinates="window in native pixel-edge coordinates; level pixel center x+0.5 and align_corners=False"),
             data=dict(train_scenes=285,train_frames=data["training_frames"],holdout_scenes=inventory["holdout"],
                       holdout_frames=data["holdout_frames"],dev_scenes=15,dev_frames=data["development_frames"],
                       independent_test_used=0,holdout_numeric_ids=HOLDOUT),
             training=dict(steps=STEPS,batch_crops=BATCH,crop_px=SIDE,region_anchors={k:2 for k in ANCHORS},
                           phase1_steps=RAMP,phase2_steps=STEPS-RAMP,coefficient={
                               "formula":"K=(0.05 m / 1.4) * fx_pixels, CoC diameter in pixels",
                               "min":float(ks.min()),"median":float(np.median(ks)),"max":float(ks.max())},
                           loss_weights=WEIGHTS,phase2_actual_term_scales="Recorded at the first phase-two batch for each arm",
                           optimizer="Adam",lr_initial=3e-4,lr_final=1e-5,ema=EMA,crop_order_sha256=data["crop_order_sha256"],
                           check_every=EVERY,gpu_process_budget_gib=TRAIN_BUDGET,h2_overlap="arithmetic mean, active only inside M",
                           gate_collapse="Report extrema; halt for exact whole-domain zero or one at two consecutive phase-two checks"),
             selection=pick,losses_disabled=["L_bokeh"],split_metrics={"rod":0,"gap":1,"normal":2,"boundary":3},
             tear_retraining=dict(steps=20000,crops_per_step=8,ema=.999,weights=[1,1,1,2],initial_lr=.001,final_lr=.00001,
                 selection="Minimize mean holdout weighted edge error and false-step tear rate; then weighted edge error and earliest step."))
    save_json(CONFIG,cfg)
    return cfg


def qualify():
    data,order=checked_prepared()
    limit=3.5
    gpu_setup(dict(training=dict(max_process_gpu_gib=limit)))
    rows={}
    for mode in FEATURE_MODES:
        gpu_gate(limit)
        torch.cuda.reset_peak_memory_stats()
        model=module(mode)
        start=time.monotonic()
        target=frame_batch(data["rows"],order[0],mode)
        prediction=predict(model,target)
        losses=terms(prediction,target)
        v=objective(losses,2)
        v.backward()
        gradients={name:float(torch.linalg.vector_norm(p.grad)) for name,p in model.named_parameters() if p.grad is not None}
        if not torch.isfinite(v) or len(gradients)!=len(list(model.parameters())) or not all(np.isfinite(list(gradients.values()))):
            raise FloatingPointError(f"{mode} has invalid qualification gradients")
        groups=defaultdict(float)
        for name,norm in gradients.items():
            groups[name.split(".")[0]]+=norm**2
        rows[mode]=dict(seconds=time.monotonic()-start,batch_crops=BATCH,
                        phase2_terms_at_initial_weights={k:float(value.detach()) for k,value in losses.items()},
                        finite_gradient_modules={k:v**.5 for k,v in groups.items()},
                        max_reserved_gib=torch.cuda.max_memory_reserved()/GIB)
        print("E0_QUALIFY",mode,rows[mode]["max_reserved_gib"],flush=True)
        del model,target,prediction,losses,v
        torch.cuda.empty_cache()
    save_json(RUN/"qualification.json",rows)


def new_diagnostics():
    return dict(error=defaultdict(list),selected=defaultdict(Counter),nll=Counter(),
                zones=Counter(),jump=Counter(),gate=Counter(),gate_by_zone=Counter(),rear_source=Counter())


def spatial_jump(weight,selected,visible):
    ambiguous=(weight>=.3)&(weight<=.7)&visible
    plus=total=0
    for axis in (0,1):
        a,b=np.swapaxes(ambiguous,axis,0)[:-1],np.swapaxes(ambiguous,axis,0)[1:]
        vis=np.swapaxes(visible,axis,0)
        support=(a|b)&vis[:-1]&vis[1:]
        picks=np.swapaxes(selected,axis,0)
        total+=int(support.sum())
        plus+=int((support&(picks[:-1]!=picks[1:])).sum())
    return plus,total


def add_diagnostics(totals,out,target):
    def cpu(t):
        return t[0,0].detach().cpu().numpy()
    visible=cpu(target["visible"])
    mv=cpu(target["mv_valid"])
    zone=cpu(target["zone"])
    loss_zone=cpu(target["loss_zone"])
    center=cpu(target["center"])
    rod_wall=cpu(target["rod_reference"])
    decoded=cpu(out["decoded"])
    base=cpu(target["rho1"])
    selected=cpu(out["selected_back"])
    weight=cpu(out["log_mass"][:,1:].exp())
    gate=cpu(out["gate"])
    rear=cpu(out["means"][:,1:].exp())
    nll=cpu(-mixture_density(target["center"].clamp_min(1e-8).log(),out["means"],out["scales"],out["log_mass"]))
    for i,word in enumerate(GROUPS):
        support=visible&(loss_zone==i)
        totals["zones"][word]+=int(support.sum())
        totals["nll"][word]+=float(np.sum(nll[support],dtype=np.float64))
        totals["gate_by_zone"][word]+=float(np.sum(gate[support],dtype=np.float64))
        choose=visible&(zone==i) if i<3 else support
        totals["error"][word+"/decoded"].append((decoded-center)[choose])
        totals["error"][word+"/OVG"].append((base-center)[choose])
        for label,valid in (("mv_valid",mv),("mv_invalid",~mv)):
            pixels=choose&valid
            branch=totals["selected"][word+"/"+label]
            branch.update(pixels=int(pixels.sum()),rear=int((pixels&selected).sum()))
    strict_normal=visible&(loss_zone==2)
    totals["error"]["normal_loss_zone/decoded"].append((decoded-center)[strict_normal])
    totals["error"]["normal_loss_zone/OVG"].append((base-center)[strict_normal])
    for label,valid in (("mv_valid",mv),("mv_invalid",~mv)):
        region=strict_normal&valid
        totals["selected"]["normal_loss_zone/"+label].update(
            pixels=int(region.sum()),rear=int((region&selected).sum()))
    gaps=visible&(zone==1)&~mv
    totals["error"]["gap_mv_invalid/rear"].append((rear-center)[gaps])
    totals["error"]["gap_mv_invalid/OVG_rho2_anchor"].append(
        (cpu(target["beta"].exp())-center)[gaps])
    rods=visible&(zone==0)&(rod_wall>0)
    totals["error"]["rod/rear_to_peeled"].append((rear-rod_wall)[rods])
    bad,count=spatial_jump(weight,selected,visible)
    totals["jump"].update(changes=bad,pairs=count)
    totals["gate"].update(zero=int((gate[visible]==0).sum()),one=int((gate[visible]==1).sum()),
                          pixels=int(visible.sum()),sum=float(np.sum(gate[visible],dtype=np.float64)))
    sources=cpu(target["mv_source"])
    for code in (0,1,2):
        totals["rear_source"][code]+=int((visible&(sources==code)).sum())


def fraction(a,b):
    return a/b if b else None


def diagnostic_summary(totals):
    return dict(errors={key:stats(value) for key,value in totals["error"].items()},
                selection={key:dict(v,selected_back=fraction(v["rear"],v["pixels"]),
                      selected_front=fraction(v["pixels"]-v["rear"],v["pixels"])) for key,v in totals["selected"].items()},
                nll=sum(totals["nll"][zone]/n for zone,n in totals["zones"].items() if n),
                nll_by_zone={k:fraction(totals["nll"][k],n) for k,n in totals["zones"].items()},
                spatial_jump=dict(totals["jump"],ratio=fraction(totals["jump"]["changes"],totals["jump"]["pairs"])),
                gate=dict(totals["gate"],mean=fraction(totals["gate"]["sum"],totals["gate"]["pixels"])),
                gate_by_zone={k:fraction(totals["gate_by_zone"][k],n) for k,n in totals["zones"].items()},
                rear_source=dict(totals["rear_source"]))


@torch.inference_mode()
def diagnose(model, rows, selected):
    totals=new_diagnostics()
    for n,index in enumerate(selected):
        if n%5==0:
            gpu_gate(TRAIN_BUDGET[model.feature_mode])
        target=frame_batch(rows,[(index,0,0)],model.feature_mode,512)
        add_diagnostics(totals,predict(model,target),target)
    return diagnostic_summary(totals)


def choose(curve):
    def rod(row):
        return row["holdout"]["errors"]["rod/decoded"]["p50"]
    available=[r for r in curve if rod(r)<=r["holdout"]["errors"]["rod/OVG"]["p50"]]
    if available:
        return min(available,key=lambda r:(r["holdout"]["nll"],rod(r),r["step"])),True
    return min(curve,key=lambda r:(rod(r),r["holdout"]["nll"],r["step"])),False


def verify_registration(data, config_path=CONFIG):
    config=load(config_path)
    if config["training"]["crop_order_sha256"]!=data["crop_order_sha256"]:
        raise ValueError("The registered E0 crop sequence disagrees with the data")
    relative=str(config_path.relative_to(config_path.parents[2]))
    committed=subprocess.check_output(["git","show",f"HEAD:{relative}"],cwd=config_path.parents[2])
    if json.loads(committed)!=config:
        raise ValueError("Commit the E0 configuration before training")
    return config


def train(mode, *, run=RUN, config_path=CONFIG, normal_front_weight=0., scheme_a=False, exclude_rod=False,
          diagnose_fn=diagnose, choose_fn=choose):
    data,order=checked_prepared()
    verify_registration(data,config_path)
    rows=data["rows"]
    index=dataset()
    holdout=[i for i,r in enumerate(rows) if r["scene"] in set(index["holdout"])]
    path=run/mode
    report=path/"training.json"
    if report.is_file():
        return load(report)
    limit=TRAIN_BUDGET[mode]
    gpu_setup(dict(training=dict(max_process_gpu_gib=limit)))
    waited=gpu_gate(limit)
    start=time.monotonic()
    competitor_before=competitor_state()
    model=module(mode,scheme_a=scheme_a)
    ema=deepcopy(model).eval().requires_grad_(False)
    params=list(model.parameters())
    shadow=list(ema.parameters())
    optimizer=torch.optim.Adam(params,lr=3e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,STEPS,eta_min=1e-5)
    step0=0
    curve=[]
    since=path/"resume.pt"
    if since.is_file():
        state=torch.load(since,map_location="cpu",weights_only=True)
        model.load_state_dict(state["model"])
        ema.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        step0=state["step"]
        curve=state["curve"]
    path.mkdir(parents=True,exist_ok=True)
    active=0.
    collapse=0
    for step in range(step0+1,STEPS+1):
        if step%25==1:
            waited+=gpu_gate(limit)
        started=time.monotonic()
        target=frame_batch(rows,order[step-1],mode)
        optimizer.zero_grad(set_to_none=True)
        out=predict(model,target)
        loss=terms(out,target)
        phase=1 if step<=RAMP else 2
        total=objective(loss,phase)
        if normal_front_weight:
            loss["normal_front_bce"]=normal_front_bce(out["log_mass"],target["loss_zone"])
            total=total+normal_front_weight*loss["normal_front_bce"]
        if scheme_a:
            loss["single_back"]=single_back_loss(out,target,exclude_rod=exclude_rod)
            total=total+loss["single_back"]
        if not torch.isfinite(total):
            raise FloatingPointError(f"Nonfinite E0 objective at {mode} step {step}")
        total.backward()
        torch.nn.utils.clip_grad_norm_(params,10.,error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            torch._foreach_lerp_(shadow,params,1.-EMA)
        active+=time.monotonic()-started
        if step==RAMP+1:
            weights=dict(WEIGHTS,normal_front_bce=normal_front_weight)
            if scheme_a:
                weights["single_back"]=1.
            save_json(path/"phase2-term-scales.json",
                      dict(step=step,unweighted={k:float(v.detach()) for k,v in loss.items()},
                           weighted={k:float(v.detach())*weights[k] for k,v in loss.items()}))
        if step%100==0 or step==1:
            row=dict(step=step,phase=phase,loss=float(total.detach()),wall_seconds=time.monotonic()-start,
                     active_seconds=active,mean_step_seconds=active/(step-step0),gpu_wait_seconds=waited,
                     peak_reserved_gib=torch.cuda.max_memory_reserved()/GIB,competitor=competitor_state())
            save_json(path/"progress.json",row)
            print("E0_STEP",mode,step,row["loss"],row["mean_step_seconds"],flush=True)
        del out,target,total,loss
        if step%EVERY:
            continue
        checkpoint=path/f"step-{step:06d}.pt"
        stats0=diagnose_fn(ema,rows,holdout)
        g=stats0["gate"]
        collapse=collapse+1 if phase==2 and (g["zero"]==g["pixels"] or g["one"]==g["pixels"]) else 0
        if collapse>=2:
            raise FloatingPointError(f"E0 gate remained collapsed at {mode} step {step}")
        snapshot(dict(model=model.state_dict(),ema=ema.state_dict(),step=step,mode=mode,scheme_a=scheme_a),checkpoint)
        curve.append(dict(step=step,phase=phase,checkpoint=checkpoint.name,holdout=stats0))
        snapshot(dict(model=model.state_dict(),ema=ema.state_dict(),optimizer=optimizer.state_dict(),
                      scheduler=scheduler.state_dict(),curve=curve,step=step),since)
        save_json(path/"holdout-curve.json",curve)
    selected,passed=choose_fn(curve)
    result=dict(mode=mode,steps=STEPS,parameters=sum(p.numel() for p in params),curve=curve,
                selected=selected,rod_baseline_eligible=passed,
                normal_front_weight=normal_front_weight,scheme_a=scheme_a,
                phase2=load(path/"phase2-term-scales.json"),wall_seconds=time.monotonic()-start,
                step_seconds=active/max(1,STEPS-step0),peak_allocated_gib=torch.cuda.max_memory_allocated()/GIB,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/GIB,
                gpu_wait_seconds=waited,competitor_before=competitor_before,competitor_after=competitor_state())
    save_json(report,result)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("register","qualify",*FEATURE_MODES))
    arg=p.parse_args()
    if arg.mode=="register":
        register()
    elif arg.mode=="qualify":
        qualify()
    else:
        train(arg.mode)
