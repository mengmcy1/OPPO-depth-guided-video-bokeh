"""Audit preserved FP32 callers and replay the Stage 3a prior-only comparison."""
import argparse
from collections import defaultdict
from copy import deepcopy
import time

import numpy as np
import torch

import route3_ovg_data as ovg
from route3_ovg_1b_optical_data import place
from route3_ovg_subpixel import save_json
from route3_v2data_B_plan import load
from route3_v2data_E_cache import RUN as E_PREP
from route3_v2data_E_labels import core_labels,unit_paths
from route3_v2data_E_panels import registered_cases
from route3_v2data_fp16_fit_audit import (RUN,STAGE3A,STAGE3B,START_FIXED,archived_local_placement,
                                        background_input,compare,fits,frames,gpu_budget,summary_counts)
from route3_v2data_fp16_fit_report import matches,value_delta
from route3_v2data_query_retrain import distribution
from route3_v2data_stage2_train import INITIAL,sample
from route3_v2data_stage3a import checkpoint,checkpoint_model,gpu_setup
from route3_v2data_stage3a_eval import render
from route3_v2data_stage3a_prior_compare import (METHODS,PRIORS,REGIONS,evaluated_frames,prior_depths)
from route3_v2data_stage1_evaluate import load_paired_checkpoint,new_regions
from route3_v2data_stage2_eval import f2_for,optical_target
from route3_v2data_baseline_dev3_extended_report import pooled
from src.aperture_renderer import aperture_disk
from src.c1_bokehme import region_sse
from src.layered_aperture import merge_layers
from src.ovg_optics import fixed_visible_surfels


def e_train_parents(name,frame,case,ref,unit):
    from route3_v2data_optics import hidden_parents,ray_fields
    folder=STAGE3A/"hidden-parents"/name/(case["id"]+"-optics.npz")
    if folder.is_file():
        from route3_v2data_stage3a import load_parents
        return load_parents(folder),"archived Stage 3a GT parents"
    label=unit/"labels"/f"f{frame:03d}-focus{case['focus_index']}-K16.npz"
    spec=dict(tiers="B",unit=unit,frame=frame,focus=case["focus_m"],fstop=case["fstop"],label=label)
    return hidden_parents(spec,ray_fields(spec),ref),"exact reconstruction from archived K16 truth"


def e_check():
    plan=load(E_PREP/"inventory.json")
    roots=unit_paths(plan)
    formal=load(E_PREP/"panels.json")
    cases=registered_cases(plan)
    before_rows={(r["scene"],v["case"]):v for r in formal["rows"] for v in r["optical"]}
    scores=[]
    for name,pair in cases.items():
        unit=roots[name]
        labels=core_labels(unit,30,plan["metadata"][name])
        fg=plan["metadata"][name]["foreground"]
        ref=dict(gt=labels["depth_axial_m"].astype(np.float32),valid=labels["visible"]&labels["valid"],
                 object_id=labels["object_id"],foreground=fg)
        bg=ref["valid"]&~np.isin(ref["object_id"],fg)
        with np.load(E_PREP/"panels-data"/f"{name}-f030.npz") as raw:
            gt=np.where(ref["valid"],raw["center"],raw["rho1"])
            maps={"OVG":1/raw["rho1"],"E":1/raw["E_decoded"],"GT":1/gt}
        results={method:fits(ref["gt"],d,bg) for method,d in maps.items()}
        cohort="training" if name in plan["overfit"] else "development"
        for case in pair:
            if cohort=="training":
                parents,provenance=e_train_parents(name,30,case,ref,unit)
                if parents is None:
                    parents=dict(uv=np.empty((0,2)),depth=np.empty(0),recoverable=np.empty(0,bool))
            else:
                parents=background_input(STAGE3A,unit,name,30,case,ref)[0]
                provenance="archived Stage 3a development GT parents"
            for method,fit in results.items():
                fix=compare(parents,*fit)
                archived=before_rows[name,case["id"]]["placed"][method]
                if (archived["eligible"],archived["placed"])!=(fix["eligible"],fix["old_placed"]):
                    raise ValueError(f"E archive audit mismatch on {name}/{case['id']}/{method}")
                if fix["needs_rescore"]:
                    raise ValueError(f"Unexpected E FP32 outcome change: {name}")
                scores.append(dict(fix,case=case["id"],name=name,cohort=cohort,method=method,
                                   parent_provenance=provenance,gt_dtype="float32"))
    save_json(RUN/"E-audit.json",dict(rows=scores,by_arm={m:summary_counts([r for r in scores if r["method"]==m])
                                                                for m in ("OVG","E","GT")}))
    return scores


@torch.inference_mode()
def verify_optics():
    import route3_ovg_1b_optical_data as optical
    gpu_setup(dict(training=dict(max_process_gpu_gib=4.5)))
    network=load_paired_checkpoint(INITIAL,"ema").eval()
    reference=matches(load(STAGE3A/"evaluation/step-000000/ema/cases.json")["legacy"])
    receipts=[]
    for cohort,name,frame,unit,entry,cases in frames():
        if cohort!="legacy" or frame!=30:
            continue
        gpu_budget()
        case=next(c for c in cases if c["focus_index"]==1)
        item=("B",name,ovg.TRAIN_FRAMES.index(frame))
        inputs,labels=sample(item)
        pred=network(inputs,protect_residual=True,render_heads=False)
        ref=ovg.b_edge_reference(item,labels)
        K=inputs["K"][0]
        focus=reference[name,case["id"]]["focus_m"]
        visible=fixed_visible_surfels(pred,inputs)
        parents,gt,bg=background_input(STAGE3A,unit,name,frame,case,ref)
        clouds=[]
        for fit in (archived_local_placement(),optical.local_placement):
            current=optical.local_placement
            try:
                optical.local_placement=fit
                hidden,_=place(parents,gt,bg,pred["depth1"][0],K,512)
            finally:
                optical.local_placement=current
            clouds.append(merge_layers(visible,hidden))
        evaluated={}
        for name0,pupil in (("K64",aperture_disk(64,.05/(2*1.4),device=K.device,dtype=K.dtype)),
                            ("pinhole",K.new_zeros((1,2)))):
            fill=f2_for(unit,entry,frame,inputs,focus,pupil)
            a,b=(render(c,K,focus,pupil,fill) for c in clouds)
            if not np.array_equal(a,b):
                raise ValueError(f"Unchanged {name}/{case['id']}/{name0} differs")
            evaluated[name0]=dict(bitwise_equal=True,elements=int(a.size),max_abs=0.)
        receipts.append(dict(scene=name,case=case["id"],outputs=evaluated))
    if len(receipts)!=3:
        raise ValueError("Expected three unchanged regression cases")
    save_json(RUN/"verified-images.json",receipts)


@torch.inference_mode()
def replay_priors():
    gpu_setup(dict(training=dict(max_process_gpu_gib=4.5)))
    began=time.monotonic()
    reference=matches(load(START_FIXED)["development"])
    after_endpoint=matches(load(RUN/"stage3a/step-020000/ema/cases.json")["development"])
    prior=load(STAGE3A/"evaluation/step-020000/ema/report.json")["prior_comparison"]
    corrected=deepcopy(prior)
    rows={(r["scene"],r["case"]):r for r in corrected["cases"]}
    net=checkpoint_model(checkpoint(STAGE3A,20000),"ema").eval()
    output=RUN/"stage3a-priors"
    audits={a:[] for a in PRIORS}
    for slug,unit,frame,cases in evaluated_frames():
        gpu_budget()
        name=unit.name
        cache=output/"frames"/name/f"f{frame:03d}.json"
        if cache.is_file():
            cached=load(cache)
            for a,values in cached["audits"].items(): audits[a].extend(values)
            for row in cached["rows"]: rows[row["scene"],row["case"]].update(row)
            continue
        item=("B",name,ovg.TRAIN_FRAMES.index(frame))
        inputs,targets=sample(item)
        pred=net(inputs,protect_residual=True,render_heads=False)
        ref=ovg.b_edge_reference(item,targets)
        maps,_=prior_depths(inputs,pred["fused_depth"][0],ovg.b_foreground_reference(name,frame))
        K=inputs["K"][0]
        k=K.cpu().numpy()
        offsets=aperture_disk(64,.05/(2*1.4),device=K.device,dtype=K.dtype)
        per_frame=[]
        per_method={a:[] for a in PRIORS}
        for case in cases:
            ident=(slug,case["id"])
            official=(name,case["id"])
            if ident not in rows:
                continue
            row=rows[ident]
            idx=case["focus_index"]
            area=REGIONS[idx]
            region=new_regions(unit,case,ref["gt"],ref["valid"],k)
            mask=region["in_focus"] if idx==0 else region["recoverable"]
            truth=optical_target(unit,name,case)
            parents,gt,bg=background_input(STAGE3A,unit,name,frame,case,ref)
            for label,source in (("start",reference),("endpoint",after_endpoint)):
                r=source[official]
                score=r["scores"]["stage3"]["metrics"][area]
                row["methods"][label]=dict(score, focus_m=r["focus_m"])
            for method,depth in maps.items():
                gpu_budget()
                vals=fits(gt,depth.cpu().numpy(),bg)
                delta=compare(parents,*vals)
                per_method[method].append(dict(delta,case=case["id"],scene=name,cohort="development"))
                if not delta["needs_rescore"]:
                    continue
                hidden,stats=place(parents,gt,bg,depth,K,512)
                if stats["placed"]!=delta["new_placed"]:
                    raise ValueError("Prior pair did not match corrected fit")
                visible=fixed_visible_surfels({"depth1":depth[None]},inputs)
                focus=row["methods"][method]["focus_m"]
                rendered=render(merge_layers(visible,hidden),K,focus,offsets,f2_for(unit,None,frame,inputs,focus,offsets))
                score=region_sse(rendered,truth,mask)
                if score["pixels"]!=row["methods"][method]["pixels"]:
                    raise ValueError("Prior-only support changed")
                row["methods"][method]=dict(score, focus_m=focus)
            per_frame.append(row)
        for method,values in per_method.items(): audits[method].extend(values)
        save_json(output/"frames"/name/f"f{frame:03d}.json",dict(audits=per_method,rows=per_frame))
        print("FP16_PRIORS",name,frame,flush=True)
    deltas={}
    for group,block in corrected["groups"].items():
        active=[r for r in rows.values() if r["scene"]==group or (group=="thin_structure" and r["scene"]!="dev008")]
        deltas[group]={}
        for method in METHODS:
            before=prior["groups"][group][method]["regions"]
            updated=block[method]["regions"]
            for region in REGIONS:
                bucket=defaultdict(list)
                for row in active:
                    if row["region"]==region and row["methods"][method]["pixels"]:
                        bucket[row["scene"]].append(row["methods"][method])
                absolute={n:pooled(v) for n,v in bucket.items()}
                median=float(np.median([v["psnr"] for v in absolute.values()])) if absolute else None
                updated[region]=dict(combined=pooled([r for v in bucket.values() for r in v]),
                                     scene_median_db=median,per_scene=absolute)
            deltas[group][method]=value_delta(before,updated,lambda v:v["combined"]["psnr"],
                                           lambda v:{k:d["psnr"] for k,d in v["per_scene"].items()})
    save_json(output/"report-fp16-fit-corrected.json",dict(corrected,changes=deltas,
            validity={arm:summary_counts(v) for arm,v in audits.items()},seconds=time.monotonic()-began))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase",choices=("e","verify","priors"))
    phase=p.parse_args().phase
    {"e":e_check,"verify":verify_optics,"priors":replay_priors}[phase]()
