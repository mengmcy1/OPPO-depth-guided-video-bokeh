"""Factor the Stage 2 boundary split into position and appearance interventions."""
import argparse
from collections import defaultdict
import fcntl
from html import escape
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

import route3_ovg_data as ovg
import route3_v2data_stage1_visualize as gallery
import route3_v2data_stage2_train as stage2
from route3_ovg_1b_common import CONFIG
from route3_ovg_1b_optical_data import geometry_root
from route3_s2a_evaluate import masks as legacy_masks
from route3_upper_bound_evaluate import display
from route3_v2data_B_batch import GIB, reserve, sha256, write_json
from route3_v2data_B_plan import load
from route3_v2data_query_retrain import distribution
from route3_v2data_stage1_evaluate import new_regions, pooled, region_errors
from route3_v2data_stage2_eval import bands, f2_for, optical_target, render_offsets
from src.aperture_renderer import aperture_disk
from src.layered_aperture import merge_layers
from src.ovg_optics import raster


METHODS=("none","C1","C2","C3","C4","C5")
COLUMNS={"input":"输入 RGB","GT":"GT · Cycles","none":"不拆分",
         "C1":"C1 · 现行拆分","C2":"C2 · 换 GT 背景","C3":"C3 · 换 GT 位置",
         "C4":"C4 · GT 位置和背景","C5":"C5 · GT 前景和背景"}
PAIRS=tuple((k,"none") for k in METHODS[1:])+(("C2","C1"),("C3","C1"),("C5","C4"))
REGIONS=("near_in_focus","near_recoverable","far_recoverable","boundary_2px","zero_combined")
PROTOCOL={
    "model": "EMA appearance and fixed Stage 1 30k same-surface geometry. The second-pass Query, P2 focal distance, 64 offsets and F2 are shared within each case.",
    "C1": "Stage 2 observed_split, Query background, predicted foreground",
    "C2": "Same split, depths, opacity and foreground as C1. Substitute valid GT complementary color. Preserve Query color at C1 pixels without a valid complementary background.",
    "C3": "Valid unsaturated GT mixed boundary intersected with q>0, Query background and predicted foreground",
    "C4": "Valid unsaturated GT mixed boundary, GT complementary color and predicted foreground. Use the original first-pass depth, initialized to frozen depth2 when q=0.",
    "C5": "Same as C4 with the GT pure foreground color. Learned opacity, depth and phase-specific footprint are fixed across factors.",
    "support": "The union of split pixels intersects each scoring region and is common to all methods. A separate full-region column retains the original Stage 2 masks.",
    "feasibility": "Linear RGB residual is the maximum over channels of the distance of I to [(1-alpha)*B, (1-alpha)*B+alpha], the set attainable by an RGB foreground in [0,1]. Threshold > .01. Both GT comparisons use identical C1 pixels with valid GT compositing labels.",
}


def item_for(plan):
    return "B",plan["name"],ovg.TRAIN_FRAMES.index(plan["frame"])


def first_observation(model, plan, geom):
    cached=stage2.RUN/"observed-first/B"/plan["name"]/f'f{plan["frame"]:03d}.npz'
    if cached.is_file():
        return ovg.load(cached)
    views,memory=model.context("B",plan["name"])
    z1,z2=(geom[k][0].cpu().numpy() for k in ("depth1","depth2"))
    return model.query.first_from_views(views,memory,plan["frame"],z1,z2)


def feasible(alpha, background, observation):
    lo=(1-alpha[...,None])*background
    hi=lo+alpha[...,None]
    return np.maximum(lo-observation,observation-hi).max(axis=-1)>.01


@torch.inference_mode()
def frame_setup(model, plan, phase):
    item=item_for(plan)
    x,y=stage2.sample(item)
    x,ref,masks,subjects=gallery.frame_reference(plan,(x,y))
    geom=model.geometry(x)
    first=first_observation(model,plan,geom)
    appearance=model.appearance(geom,x,first)
    gt=stage2.appearance_labels(item)
    alpha=y["alpha"][0].cpu().numpy()
    gt_mixed=(alpha>0)&(alpha<1)&y["valid_alpha"][0].cpu().numpy()&gt["comp_valid"]&gt["color_valid"]
    if plan["unit"]:
        path=plan["unit"]/"labels"/f'f{plan["frame"]:03d}-colors.npz'
        bg_valid=ovg.load(path,("background_valid",))["background_valid"]
    else:
        bg_valid=gt["comp_valid"]
    predicted=stage2.split_mask(appearance,first).cpu().numpy()
    available=first["q"]>0
    definitions={"none":np.zeros_like(predicted),"C1":predicted,"C2":predicted,
                 "C3":gt_mixed&available,"C4":gt_mixed,"C5":gt_mixed}
    union=predicted|gt_mixed
    frames={}
    for name,mask in definitions.items():
        split=stage2.tensor(mask)
        color=first
        if name=="C2":
            color=dict(first,color=np.where(bg_valid[...,None],gt["background"],first["color"]))
        if name in ("C4","C5"):
            color=dict(first,color=gt["background"],q=np.where(mask,1,first["q"]))
        ap=appearance
        if name=="C5":
            ap=dict(appearance,foreground=stage2.tensor(gt["foreground"]).permute(2,0,1)[None])
        visible=stage2.visible(ap,x,split,phase)
        frames[name]=merge_layers(visible,stage2.background(ap,x,split,color))
    rgb=x["rgb"][0].permute(1,2,0).cpu().numpy()
    pred_bad=feasible(appearance["alpha"][0].cpu().numpy(),first["color"],rgb)
    matched=predicted & gt_mixed
    gt_bad=feasible(alpha,gt["background"],rgb)
    facts=dict(counts={k:int(m.sum()) for k,m in definitions.items()},union=int(union.sum()),
        gt_boundary=int(gt_mixed.sum()),gt_without_query=int((gt_mixed&~available).sum()),
        C2_substituted=int((predicted&bg_valid).sum()),C2_without_gt=int((predicted&~bg_valid).sum()),
        feasibility=dict(C1_all_pixels=int(predicted.sum()),C1_all_over=int((pred_bad&predicted).sum()),
                         matched_pixels=int(matched.sum()),matched_C1_over=int((pred_bad&matched).sum()),
                         matched_GT_over=int((gt_bad&matched).sum())))
    return x,y,geom,ref,masks,subjects,frames,union,predicted,gt_mixed,rgb,facts


def measure(value,target,regions):
    return {k:pooled([region_errors(value,target,m)]) for k,m in regions.items()}


def overlay(rgb,mask,paint):
    data=rgb.copy()
    data[mask]=.40*data[mask]+.60*np.asarray(paint,np.float32)
    return data


@torch.inference_mode()
def render_case(model, plan, phase, state, index, original):
    x,y,g,ref,masks,subjects,frames,union,pred,gt,rgb,facts=state
    case,K=plan["cases"][index],x["K"][0]
    frame,unit,name=(plan[k] for k in ("frame","unit","name"))
    focus=original["actual_focus_m"]["S3"]
    from route3_v2data_stage1_evaluate import p2_focus
    own=p2_focus(g["depth1"][0].cpu().numpy(),subjects[index])
    if own is None or abs(own-focus)>1e-6:
        raise ValueError("Frozen P2 focus differs from original case")
    offsets=aperture_disk(CONFIG["native_eval_rays"],.05/(2*case["fstop"]),device=K.device,dtype=K.dtype)
    center=K.new_zeros((1,2))
    fill=f2_for(unit,plan["entry"],frame,x,focus,offsets)
    zfill=f2_for(unit,plan["entry"],frame,x,focus,center)
    truth=optical_target(unit,name,case)
    pinhole=display(rgb)
    hidden=model.second(item_for(plan),g["depth1"][0].cpu().numpy(),focus,case["fstop"])
    near=index==0
    edge=bands(ref,y)[0]
    regions={"boundary_2px":edge}
    if near:
        target=new_regions(unit,case,ref["gt"],ref["valid"],K.cpu().numpy()) if unit else legacy_masks(geometry_root(name),case)
        regions.update(near_in_focus=masks[0],near_recoverable=target["recoverable"])
    else:
        regions["far_recoverable"]=masks[1]
    common={k:m&union for k,m in regions.items()}
    summary={}
    images={"input":pinhole,"GT":truth}
    for key,scene in frames.items():
        combined=merge_layers(scene,hidden)
        value=render_offsets(combined,K,focus,offsets,fill)
        zero=raster(combined,K,focus,center,512,512,zfill)[0]
        primary=measure(value,truth,common)
        full=measure(value,truth,regions)
        primary.update(measure(zero,pinhole,{"zero_combined":union}))
        full.update(measure(zero,pinhole,{"zero_combined":np.ones_like(union)}))
        summary[key]=dict(common=primary,full=full)
        images[key]=display(value.cpu().numpy())
    primary=original["region"]
    scores={k:v["common"][primary] for k,v in summary.items()}
    scores.update(input=measure(x["rgb"][0].permute(1,2,0),truth,{"input":common[primary]})["input"],
                  GT=pooled([dict(pixels=int(common[primary].sum()),sse=0.)]))
    diagnostics={"C1 · 预测拆分":overlay(pinhole,pred,(.1,.9,.85)),
                 "GT · 有效混合边界":overlay(pinhole,gt,(1.,.35,.65))}
    row={k:original[k] for k in ("file","short","scene","frame","focus","case","why","crop_xyxy")}
    row.update(actual_focus_m=dict(input=None,GT=case["focus_m"],**dict.fromkeys(METHODS,focus)),
               region_scores=scores,metrics=summary,frame_facts=facts)
    return row,images,diagnostics,masks


def pooled_groups(rows, scope, region):
    grouped=defaultdict(list)
    for ref in rows:
        if region in ref["metrics"]["none"][scope] and ref["metrics"]["none"][scope][region]["pixels"]:
            grouped[ref["scene"]].append(ref)
    counts=sum(map(len,grouped.values()))
    systems={}
    for key in METHODS:
        values={scene:pooled([r["metrics"][key][scope][region] for r in refs]) for scene,refs in grouped.items()}
        systems[key]=dict(per_scene=values,pooled=pooled(list(values.values())))
    pairs={}
    for after,before in PAIRS:
        a,b=systems[after],systems[before]
        delta={s:a["per_scene"][s]["psnr_db"]-b["per_scene"][s]["psnr_db"] for s in grouped}
        combined=(a["pooled"]["psnr_db"]-b["pooled"]["psnr_db"]) if counts else None
        pairs[after+"-"+before]=dict(pooled_delta_db=combined,per_scene_db=delta,**distribution(list(delta.values())))
    return dict(cases=counts,systems=systems,pairs=pairs)


def summarize(rows):
    result={}
    for cohort in ("development","legacy"):
        selected=[r for r in rows if not r.get("error") and r["short"].startswith("dev")==(cohort=="development")]
        frames={(r["scene"],r["frame"]):r["frame_facts"] for r in selected}.values()
        summed={k:sum(f[k] for f in frames) for k in ("union","gt_boundary","gt_without_query","C2_substituted","C2_without_gt")}
        summed.update(counts={k:sum(f["counts"][k] for f in frames) for k in METHODS})
        summed["feasibility"]={k:sum(f["feasibility"][k] for f in frames) for k in
                               ("C1_all_pixels","C1_all_over","matched_pixels","matched_C1_over","matched_GT_over")}
        result[cohort]=dict(frame_facts=summed,supports={s:{r:pooled_groups(selected,s,r) for r in REGIONS}
                                                        for s in ("common","full")})
    return result


def number(value):
    return "—" if value is None else f"{value:+.3f}"


def share(a,b):
    return f"{a/b:.2%}（{a:,}/{b:,}）" if b else "无有效像素"


def report_files(folder, report):
    md=[f"# {report['phase']} 边界拆分五格","",
        "主表以拆分并集与评分区相交为公共支持。每格为合并增量／逐场景中位数／标准差／正、负、持平场景数，单位 dB。"
        "C2 在有 GT 背景标签的 C1 像素换色，缺标签像素保留 Query 颜色，位置和深度一致。",""]
    for cohort,part in report["summary"].items():
        md+=["",f"## {cohort}","",
             "| 区域 | 公共基线 PSNR | "+" | ".join(a+"−"+b for a,b in PAIRS)+" |",
             "|---|---|"+"---|"*len(PAIRS)]
        for region,data in part["supports"]["common"].items():
            values=[]
            for a,b in PAIRS:
                p=data["pairs"][a+"-"+b]
                values.append(f"{number(p['pooled_delta_db'])} / {number(p['median_db'])} / "
                              f"{number(p['sd_db'])} / {p['positive']},{p['negative']},{p['tied']}")
            md.append(f"| {region} | {number(data['systems']['none']['pooled']['psnr_db'])} | "+" | ".join(values)+" |")
        f=part["frame_facts"]
        feas=f["feasibility"]
        md+=["",f"- GT 边界缺 Query {share(f['gt_without_query'],f['gt_boundary'])}。",
             f"- C2 无 GT 背景标签 {share(f['C2_without_gt'],f['counts']['C1'])}。"
             f"各格拆分像素数 {f['counts']}。",
             f"- C1 全部像素的可行性残差 >0.01 为 {share(feas['C1_all_over'],feas['C1_all_pixels'])}。"
             f"同一 GT 有效子集内，C1 为 {share(feas['matched_C1_over'],feas['matched_pixels'])}，"
             f"换成 GT α 和背景为 {share(feas['matched_GT_over'],feas['matched_pixels'])}。"]
    md+=["","## 跳过",""]+[f"- {r['file']}，{r['error']}" for r in report["rows"] if r.get("error")]
    (folder/"汇总.md").write_text("\n".join(md)+"\n",encoding="utf-8")
    html=['<!doctype html><meta charset=utf-8><title>边界拆分五格</title><h1>边界拆分五格</h1>',
          '<p><a href="汇总.md">汇总</a></p><main style="display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,36rem),1fr));gap:1em">']
    for r in report["rows"]:
        if r.get("error"):
            continue
        n=filename(r)
        html.append(f'<figure style="margin:0"><a href="{n}"><img loading="lazy" width="100%" src="{n}"></a>'
                    f'<figcaption>{escape(r["file"])}</figcaption></figure>')
    (folder/"index.html").write_text("\n".join(html+["</main>"])+"\n",encoding="utf-8")


def filename(row):
    return "000_dev001_f030_near.png" if row["file"]=="021_dev001_f030_near.png" else row["file"]


def root_index(out):
    links={phase:f"{phase}-10k" for phase in ("2a","2b") if (out/f"{phase}-10k/汇总.md").is_file()}
    html=['<!doctype html><meta charset=utf-8><title>边界拆分五格</title><h1>边界拆分五格</h1>']
    md=["# 边界拆分五格",""]
    for phase,folder in links.items():
        html.append(f'<p>{phase} 10k · <a href="{folder}/index.html">图集</a> · <a href="{folder}/汇总.md">汇总</a></p>')
        md.append(f"- {phase} 10k [汇总]({folder}/汇总.md) · [图集]({folder}/index.html)")
    (out/"index.html").write_text("\n".join(html)+"\n",encoding="utf-8")
    (out/"汇总.md").write_text("\n".join(md)+"\n",encoding="utf-8")


def run(source,phase,pilot=False):
    stage2.configuration()
    torch.cuda.set_per_process_memory_fraction(6*GIB/torch.cuda.get_device_properties(0).total_memory,0)
    stage2.fixed_inputs()
    torch.backends.cuda.matmul.allow_tf32=False
    out=source/"split5"
    folder=out/f"{phase}-10k"
    folder.mkdir(parents=True,exist_ok=True)
    checkpoint=stage2.RUN/phase/"step-010000.pt"
    original=load(source/"metrics.json")
    path=folder/"metrics.json"
    digest=sha256(checkpoint)
    report=load(path) or dict(phase=phase,checkpoint=str(checkpoint),checkpoint_sha256=digest,protocol=PROTOCOL,
                             source_metrics_sha256=sha256(source/"metrics.json"),seconds=0.,
                             started_at=gallery.stamp(),rows=[r for r in original["rows"] if r.get("error")])
    if report["checkpoint_sha256"]!=digest:
        raise ValueError("Previously recorded five-grid run has different weights")
    started=time.monotonic()
    previous=report["seconds"]
    model=stage2.Model()
    model.heads.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=True)["ema"])
    model.heads.eval()
    rows={r["file"]:r for r in report["rows"]}
    source_rows={r["file"]:r for r in original["rows"] if not r.get("error")}
    plan=list(gallery.planned_frames())
    plan.sort(key=lambda t:t["files"][0]["short"]!="dev001" or t["frame"]!=30)
    for frame in plan:
        pending=[(i,source_rows[r["file"]]) for i,r in enumerate(frame["files"])
                 if r["file"] in source_rows and r["file"] not in rows and
                 (not pilot or r["file"]=="021_dev001_f030_near.png")]
        if not pending:
            continue
        reserve(folder,GIB)
        try:
            state=frame_setup(model,frame,phase)
        except Exception as error:
            for _,r in pending:
                rows[r["file"]]=dict(r,error=f"{type(error).__name__}: {error}")
                print("SPLIT5_SKIP",phase,r["file"],str(error),flush=True)
            torch.cuda.empty_cache()
        else:
            for idx,old in pending:
                try:
                    r,images,diag,masks=render_case(model,frame,phase,state,idx,old)
                    gallery.draw_panel(folder/filename(r),r,images,masks,{},columns=COLUMNS,tile=225,
                                       diagnostic_images=diag,diagnostic_subtitle="实际拆分位置")
                    rows[r["file"]]=r
                    print("SPLIT5_DONE",phase,r["file"],r["frame_facts"]["counts"],flush=True)
                except Exception as error:
                    rows[old["file"]]=dict(old,error=f"{type(error).__name__}: {error}")
                    print("SPLIT5_SKIP",phase,old["file"],str(error),flush=True)
                    torch.cuda.empty_cache()
            del state
        report["rows"]=sorted(rows.values(),key=filename)
        report["seconds"]=previous+time.monotonic()-started
        write_json(path,report)
    if not pilot:
        report["summary"]=summarize(report["rows"])
        report["completed_at"]=gallery.stamp()
        report_files(folder,report)
        root_index(out)
        report["seconds"]=previous+time.monotonic()-started
        write_json(path,report)


def follow(source):
    out=source/"split5"
    out.mkdir(exist_ok=True)
    with (out/".follow.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for phase in ("2a","2b"):
            checkpoint=stage2.RUN/phase/"step-010000.pt"
            while not checkpoint.is_file():
                write_json(out/"status.json",dict(state="waiting_for_checkpoint",phase=phase,at=gallery.stamp()))
                time.sleep(30)
            command=[sys.executable,"-u",str(Path(__file__).resolve()),"--source",str(source),"--phase",phase]
            write_json(out/"status.json",dict(state="evaluating",phase=phase,at=gallery.stamp()))
            subprocess.run(command,check=True)
        write_json(out/"status.json",dict(state="complete",at=gallery.stamp()))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,default=gallery.OUT)
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument("--phase",choices=("2a","2b"))
    g.add_argument("--follow",action="store_true")
    p.add_argument("--pilot",action="store_true")
    a=p.parse_args()
    if a.follow:
        follow(a.source.resolve())
    else:
        run(a.source.resolve(),a.phase,a.pilot)
