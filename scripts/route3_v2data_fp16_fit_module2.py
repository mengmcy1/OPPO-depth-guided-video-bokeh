"""Refresh only affected module-two optical arms on their archived cases."""
import argparse
from collections import defaultdict
from copy import deepcopy
import time

import numpy as np
import torch

import route3_ovg_data as ovg
from route3_ovg_subpixel import save_json
from route3_v2data_B_plan import load
from route3_v2data_fp16_fit_audit import (RUN, STAGE3A, background_input,compare,fits,frames,gpu_budget,summary_counts)
from route3_v2data_fp16_fit_report import matches,value_delta
from route3_v2data_stage2_train import sample
import route3_v2data_module2_infer as engine
import route3_v2data_module2_d_evaluate as learned
from route3_v2data_module2_d_labels import RUN as LEARN_V1
from route3_v2data_module2_d_v2_train import RUN as LEARN_V2,CHANNELS
from src.ovg_tear_head import VARIANTS

SEVEN=engine.OUT/"all-arms"/"report.json"
ARCHIVES=dict(v1=LEARN_V1/"evaluation/report.json",v2=LEARN_V2/"evaluation/report.json")
ARMS={"seven":engine.ARMS,"v1":learned.ARMS,"v2":("V0_pred","v1","v2a","v2b","b","c_010")}
REPLAY={"seven":engine.ARMS,"v1":learned.LEARNED,"v2":("v2a","v2b")}
PATH=RUN/"module2"
METRICS=("near_gt_minus_f2",*engine.REGIONS,*engine.ZERO)


def synthetic_reference(unit,frame):
    ref=ovg.b_foreground_reference(unit.name,frame)
    with np.load(unit/"labels"/f"f{frame:03d}-pinhole.npz") as y:
        return dict(ref, valid=ref["valid"]&y["valid"])


def frame_from_cache(scope,name,frame,unit,entry):
    archive=ROOT_E/"cache/ovg"/name/f"f{frame:03d}.npz"
    if archive.is_file():
        with np.load(archive) as a:
            pred=(np.float32(1.)/a["rho1"]).astype(np.float32)
        return pred,synthetic_reference(unit,frame)
    raise FileNotFoundError(f"Need a native legacy reference for {name}/{frame}")


ROOT_E=RUN.parents[1]/"outputs/route3-v2data-E-prep-2026-11-26-01"


@torch.inference_mode()
def scan(model,original):
    lookup=matches(original)
    proved=load(PATH/"placement-audit.json")
    if proved:
        return proved["by_case"]
    phase={}
    for scope,name,frame,unit,entry,cases in frames():
        if unit:
            pred,ref=frame_from_cache(scope,name,frame,unit,entry)
        else:
            gpu_budget()
            item=("B",name,ovg.TRAIN_FRAMES.index(frame))
            x,y=sample(item)
            pred=model(x,protect_residual=True,render_heads=False)["depth1"][0].cpu().numpy()
            ref=ovg.b_edge_reference(item,y)
        center=ref["gt"].astype(np.float32)
        known=ref["valid"]&np.isfinite(center)&(center>0)
        gt_depth=np.where(known,center,pred)
        prior=None
        for case in cases:
            key=(name,case["id"])
            previous=lookup[key]
            if previous["status"]!="evaluated":
                continue
            parents,gt,bg=background_input(STAGE3A,unit,name,frame,case,ref)
            if prior is None:
                prior={"pred":fits(gt,pred,bg),"gt":fits(gt,gt_depth,bg)}
            entries={}
            for kind,pair in prior.items():
                delta=compare(parents,*pair)
                delta.update(scene=name,frame=frame,case=case["id"],cohort=scope,dtype=str(gt.dtype))
                entries[kind]=delta
            phase[name+"/"+case["id"]]=entries
    groups={k:summary_counts([r[k] for r in phase.values()]) for k in ("gt","pred")}
    save_json(PATH/"placement-audit.json",dict(by_case=phase,summary=groups))
    return phase


def use_arm(arm,stamp):
    return stamp["gt" if arm in ("V0_GT","a") else "pred"]["needs_rescore"]


def metric_changes(original,summary):
    returned={}
    for cohort,current in summary.items():
        old=original[cohort]
        block={}
        for arm,metrics in current["methods"].items():
            per_before={n:v[arm] for n,v in old["per_scene"].items()}
            per_after={n:v[arm] for n,v in current["per_scene"].items()}
            projector=lambda pool, scenes:{k:dict(pooled=pool[k]["psnr_db"],
                            per_scene={name:row[k]["psnr_db"] for name,row in scenes.items()}) for k in METRICS}
            changes=value_delta(projector(old["methods"][arm],per_before),projector(metrics,per_after),
                                lambda r:r["pooled"],lambda r:r["per_scene"])
            changes["positive_SSE_bins"]={tag:{k:v["transmission"]["positive_sse_bins"].get(tag,0)
                                               for k,v in (("before",old["methods"][arm]),("after",metrics))}
                     for tag in sorted(set(old["methods"][arm]["transmission"]["positive_sse_bins"])|
                                       set(metrics["transmission"]["positive_sse_bins"]))}
            changes["pinhole_pass_before_after"]={k:[old["methods"][arm][k]["passed_cases"],metrics[k]["passed_cases"]]
                     for k in ("zero_visible","zero_combined")}
            block[arm]=changes
        returned[cohort]=block
    return returned


def merged_summary(prior,rows,methods):
    pairs={a:("V0_GT" if a=="a" else "V0_pred") for a in methods if a not in ("V0_GT","V0_pred")}
    if "V0_pred" not in methods:
        pairs={"a":"V0_GT"}
    return engine.summarize([{"rows":rows}],methods,pairings=pairs,include_absolute_near=True)


@torch.inference_mode()
def run(kind,scan_only=False):
    archive=load(SEVEN if kind=="seven" else ARCHIVES[kind])
    old=archive.get("case_rows",archive.get("cases"))
    baseold=load(SEVEN)["case_rows"] if kind!="seven" else old
    engine.gpu_setup(dict(training=dict(max_process_gpu_gib=engine.GPU_GIB)))
    began=time.monotonic()
    gpu_budget()
    net=engine.load_geometry()
    checks=scan(net,baseold)
    if scan_only:
        return
    source={k:deepcopy(v) for k,v in matches(old).items()}
    if kind!="seven":
        upstream=matches(load(PATH/"seven/report-fp16-fit-corrected.json")["case_rows"])
        for key,r in source.items():
            if not r["methods"]:
                continue
            for arm in ("V0_pred","b","c_010"):
                r["methods"][arm]=upstream[key]["methods"][arm]
    predict=None
    if kind in ARCHIVES:
        run=LEARN_V1 if kind=="v1" else LEARN_V2
        variants=VARIANTS if kind=="v1" else dict.fromkeys(REPLAY["v2"],CHANNELS)
        predict=learned.predictor(learned.chosen_heads(load(run/"training-report.json"),variants))
    for scope,name,frame,unit,entry,cases in frames():
        saved=PATH/kind/"frames"/name/f"f{frame:03d}.json"
        patches=load(saved)
        if not saved.is_file():
            buckets=defaultdict(list)
            for case in cases:
                tag=case["id"]
                delta=checks.get(name+"/"+tag)
                if delta:
                    methods=tuple(a for a in REPLAY[kind] if use_arm(a,delta))
                    if methods:
                        buckets[methods].append(case)
            patches=[]
            for arms,subset in buckets.items():
                gpu_budget()
                result=engine.frame_results(net,scope,name,frame,unit,entry,subset,PATH/kind,arms,
                                            tear_predictor=predict,panel_writer=lambda *a:{})
                for row in result["rows"]:
                    before=source[row["scene"],row["case"]]
                    if row["status"]!=before["status"]:
                        raise ValueError("Corrected case has different focus support")
                    for arm,m in row["methods"].items():
                        for region,s in m["optical"].items():
                            if s["pixels"]!=before["methods"][arm]["optical"][region]["pixels"]:
                                raise ValueError("Corrected case changed archived pixel support")
                    patches.append(dict(scene=row["scene"],case=row["case"],methods=row["methods"]))
            save_json(saved,patches)
        for p in patches:
            source[p["scene"],p["case"]]["methods"].update(p["methods"])
        if patches:
            print("FP16_MODULE2",kind,name,frame,len(patches),flush=True)
    rows=list(source.values())
    methods=ARMS[kind]
    summary=merged_summary(archive["summary"],rows,methods)
    audits={}
    for arm in methods:
        stamp="gt" if arm in ("V0_GT","a") else "pred"
        counts=[v[stamp] for r in rows if (v:=checks.get(r["scene"]+"/"+r["case"]))]
        audits[arm]=summary_counts(counts)
    diff=metric_changes(archive["summary"],summary)
    record=dict(case_rows=rows,summary=summary,by_arm=audits,changes=diff,seconds=time.monotonic()-began,
                previous_source=str(SEVEN if kind=="seven" else ARCHIVES[kind]),test_cases_used=0)
    save_json(PATH/kind/"report-fp16-fit-corrected.json",record)
    if kind=="v1":
        replay_gap_cases(net,predict,archive,rows,checks)
    if kind=="seven":
        gate=load(engine.OUT/"a-check/report.json")
        subset=[source[r["scene"],r["case"]] for r in gate["case_rows"]]
        gate_rows=[dict(r,methods={k:r["methods"][k] for k in ("V0_GT","a") if k in r["methods"]})
                   for r in subset]
        gate_new=merged_summary(gate["summary"],gate_rows,("V0_GT","a"))
        relevant=[checks[r["scene"]+"/"+r["case"]]["gt"] for r in gate_rows if r["methods"]]
        save_json(PATH/"a-check/report-fp16-fit-corrected.json",dict(summary=gate_new,
            changes=metric_changes(gate["summary"],gate_new),affected=summary_counts(relevant),case_rows=gate_rows))


@torch.inference_mode()
def replay_gap_cases(net,predict,archive,rows,checks):
    path=PATH/"v1/gaps"
    resolution={r["frame"]:r for r in archive["gap"]}
    unchanged=matches(rows)
    result=[]
    for _,name,frame,unit,entry,cases in frames():
        if not name.startswith("development-005-"):
            continue
        old=resolution[frame]
        flag=checks.get(name+"/"+old["case"])
        if not flag or not use_arm("d_full",flag):
            continue
        saved=path/f"f{frame:03d}.json"
        fixed=load(saved)
        if not saved.is_file():
            gpu_budget()
            archive_path=learned.RESULTS
            try:
                learned.RESULTS=path
                (path/"panels").mkdir(parents=True,exist_ok=True)
                fixed=learned.diagnose_dev005(net,predict,cases,unit,entry,frame,
                      [unchanged[name,case["id"]] for case in cases])
            finally:
                learned.RESULTS=archive_path
            save_json(saved,fixed)
        result.append(dict(frame=frame,case=old["case"],before=old["methods"],after=fixed["methods"]))
    save_json(path/"comparison.json",result)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase",choices=("seven","v1","v2"))
    p.add_argument("--scan",action="store_true")
    args=p.parse_args()
    run(args.phase,args.scan)
