"""Register, train and reselect the matched Scheme A E-H2 experiment."""
import argparse
import os
import time

import torch

import route3_v2data_E0_data as data
import route3_v2data_E0_tears as tears
import route3_v2data_E0_train as training
import route3_v2data_E0_v2 as v2
from route3_v2data_B_batch import GIB, sha256
from route3_v2data_E_labels import load, save_json
from route3_v2data_E0_v3_metrics import RULE, choose, diagnose, reachability
from src.ovg_e import ORDER_RATIO, RESIDUAL_LIMIT


ROOT=data.ROOT
RUN=ROOT/"outputs/route3-v2data-E0-v3-2026-11-26-01"
CONFIG=v2.RECORDS/"route3-v2data-E0-v3-config-2026-11-26.json"
RESULT=v2.RECORDS/"route3-v2data-E0-v3-2026-11-26.json"
BASES={"v1":data.RUN,"v2":v2.RUN/"v2","v3":RUN}
ORIGINAL_STEPS={"v1":2000,"v2":4000}
MODE=v2.MODE


def dispatch(phase, **details):
    save_json(RUN/"dispatch.json",dict(phase=phase,pid=os.getpid(),at=time.time(),**details))


def input_rows():
    choice=load(v2.RESULT)["geometry"]
    if choice["step"]!=0 or sha256(data.INITIAL)!=choice["sha256"]:
        raise ValueError("Frozen OVG checkpoint does not match the registered E0 v2 input")
    receipt,order=training.checked_prepared()
    index=data.dataset()
    held=set(index["holdout"])
    rows=receipt["rows"]
    holdout=[i for i,r in enumerate(rows) if r["scene"] in held]
    fitting=set(index["geometry"])-held
    fitting_rows=[r for r in rows if r["scene"] in fitting]
    if len(fitting_rows)!=receipt["training_frames"] or len(holdout)!=receipt["holdout_frames"]:
        raise ValueError("E0 training/holdout inventory disagrees with the prepared inputs")
    return choice,receipt,order,holdout,fitting_rows


def setup():
    training.gpu_setup(dict(training=dict(max_process_gpu_gib=v2.LIMIT)))
    training.gpu_gate(v2.LIMIT)


def qualify():
    choice,receipt,order,_,fitting=input_rows()
    dispatch("step-zero-and-reachability")
    setup()
    batch=training.frame_batch(receipt["rows"],order[0],MODE)
    network=training.module(MODE,scheme_a=True)
    prediction=training.predict(network,batch)
    loss=training.single_back_loss(prediction,batch)
    nll=training.terms(prediction,batch)["nll"]
    grad,gate_grad=torch.autograd.grad(loss,(prediction["mixture_head"],prediction["gate_logit"]),allow_unused=True)
    if grad is None:
        raise FloatingPointError("Scheme A has no gradient at the mixture head")
    norms=grad.detach().abs().amax(dim=(0,2,3)).cpu().tolist()
    gate_norm=0. if gate_grad is None else float(gate_grad.abs().max())
    start=dict(loss=float(loss.detach()),nll=float(nll.detach()),peak_reserved_gib=torch.cuda.max_memory_reserved()/GIB,
               max_abs_grad=dict(zip(("front_tanh","back_tanh","front_scale","back_scale","front_weight_logit"),norms)),
               gate_logit_max_abs_grad=gate_norm,
               single_pixels=int((batch["visible"]&~batch["valid_dual"]).sum()),
               direct_raw_back_matches_forward=bool(torch.equal(prediction["raw_back"],prediction["single_back_mean"])))
    start["loss_vs_abs_nll"]=start["loss"]/abs(start["nll"])
    if (not torch.isfinite(loss) or not torch.isfinite(nll) or not torch.isfinite(grad).all()
            or norms[1]<=0 or any(norms[i]!=0 for i in (0,2,3,4)) or gate_norm!=0
            or not start["direct_raw_back_matches_forward"]):
        raise FloatingPointError(f"Scheme A step zero failed: {start}")
    gate=float(prediction["gate"][0,0,0,0])
    if not (prediction["gate"]==gate).all():
        raise ValueError("Initial E-H2 gate must be spatially constant")
    del prediction,batch,network,grad,gate_grad,loss,nll
    torch.cuda.empty_cache()
    reach=reachability(fitting,gate)
    base=load(v2.CONFIG)
    cfg=dict(geometry=choice,feature_mode=MODE,training=base["training"],schedule=base["inputs"],seed=base["seed"],
             data=base["data"],tear_retraining=base["tear_retraining"],scheme_A=dict(
                 support="visible & ~valid_dual",target="min(log rho_GT_center, stopgrad(mu_f) + log(0.985))",
                 fitted_value="raw mu_b before front/back clipping; detach a and g for this term",
                 objective="Sum of the four per-zone means of Smooth L1 with beta 0.05, log inverse depth",
                 weight=1.,from_step=0,both_phases=True,rho2_rear_residual="1.5 * tanh(h1), independent of a and g",
                 multiview_rear_residual="1.5 * a * g * tanh(h1)",front_residual="unchanged from v2",
                 rear_mass="g * (1-w_f)",clip_ratio=ORDER_RATIO,residual_limit=RESIDUAL_LIMIT,
                 order_penalty="unchanged softplus(order_gap)",dual_supervision="unchanged unbalanced pixel mean",
                 normal_front_bce_weight=.1,semantics="On single-surface pixels the rear component targets the same visible surface subject to order clipping, including rod interiors. It no longer denotes a surface behind the visible one. Retrain the tear head with the changed |delta log rho_b| channel. Other seed-propagation uses are deferred.",
                 depth_offset_at_clip_percent=100*(1/ORDER_RATIO-1.),coc_offset_at_rho_half_px=25.4*.5*(1-ORDER_RATIO)),
             selection=RULE,old_selected_steps=ORIGINAL_STEPS,step_zero=start,reachability=reach,
             geometry_limits=base["geometry_limits"],evaluation=dict(development_cases=87,legacy_cases=18,test_cases=0,
                 far_primary="F2-only, substitute pending",focus=["P","G"],FP16_fix="6cff9f9"))
    save_json(CONFIG,cfg)
    print("E3_QUALIFIED",start,reach,flush=True)
    return cfg


def fit():
    dispatch("training")
    return training.train(MODE,run=RUN,config_path=CONFIG,normal_front_weight=.1,scheme_a=True,
                          diagnose_fn=diagnose,choose_fn=choose)


def reselect():
    _,receipt,_,holdout,_=input_rows()
    setup()
    result={}
    for version,old_step in ORIGINAL_STEPS.items():
        root=RUN/"retrospective"/version
        curve=[]
        missing=[]
        for step in range(1000,20001,1000):
            dest=root/f"step-{step:06d}.json"
            source=BASES[version]/MODE/f"step-{step:06d}.pt"
            if not source.is_file():
                missing.append(step)
                continue
            if dest.is_file():
                row=load(dest)
            else:
                model,_=tears.load_model(MODE,step,run=BASES[version])
                row=dict(step=step,phase=1 if step<=training.RAMP else 2,checkpoint=source.name,
                         holdout=diagnose(model,receipt["rows"],holdout))
                del model
                save_json(dest,row)
            curve.append(row)
            print("E3_RETRO",version,step,row["holdout"]["coc_balanced_px"],flush=True)
        selected,eligible=choose(curve)
        result[version]=dict(curve=curve,selected=selected,passed_rod_gate=eligible,missing_steps=missing,
                             old_step=old_step,old_point=next((r for r in curve if r["step"]==old_step),None))
    save_json(RUN/"retrospective.json",result)
    return result


def tear_root(version, step):
    known=BASES[version]/MODE/f"tear-{step:06d}"
    if (known/"selected.pt").is_file() and (known/"training.json").is_file():
        return known
    return RUN/"tear-heads"/version/f"step-{step:06d}"


def retrain(version, step):
    root=tear_root(version,step)
    if (root/"training.json").is_file():
        return load(root/"training.json")
    return tears.train(MODE,step,run=BASES[version],destination=root)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("qualify","train","reselect","tear"))
    parser.add_argument("--version",choices=BASES)
    parser.add_argument("--step",type=int)
    args=parser.parse_args()
    if args.command=="tear":
        retrain(args.version,args.step)
    else:
        {"qualify":qualify,"train":fit,"reselect":reselect}[args.command]()
