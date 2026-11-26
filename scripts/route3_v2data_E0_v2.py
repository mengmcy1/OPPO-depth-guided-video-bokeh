"""Run the registered E0 v2 objective after Stage 3c publishes its adoption."""
import argparse
from collections import defaultdict
from html import escape
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

import route3_v2data_E0_data as data
import route3_v2data_E0_evaluate as evaluation
import route3_v2data_E0_rerender as replay
import route3_v2data_E0_rerender_report as panels
import route3_v2data_E0_tears as tears
import route3_v2data_E0_train as training
import route3_v2data_E0_windows as windows
import route3_v2data_E_cache as feature_cache
from route3_v2data_B_batch import GIB, sha256
from route3_v2data_E_labels import load, save_json, unit_paths
from route3_v2data_module2_d_evaluate import chosen_heads, predictor
from route3_v2data_module2_d_v2_train import CHANNELS, RUN as D_RUN
from route3_v2data_stage3b import load_geometry, path_for
import route3_v2data_module2_infer as ref

ROOT=data.ROOT
OLD=data.RUN
PREP=feature_cache.RUN
RUN=ROOT/"outputs/route3-v2data-E0-v2-2026-11-26-01"
C3=replay.STAGE3C
RECORDS=ROOT/"03_实验记录/records"
CONFIG=RECORDS/"route3-v2data-E0-v2-config-2026-11-26.json"
RESULT=RECORDS/"route3-v2data-E0-v2-2026-11-26.json"
PROGRESS=ROOT/"03_实验记录/当前实验进度与方法论.md"
C3_RECORD=RECORDS/"route3-v2data-stage3c-2026-11-26-step-010000.json"
OLD_REPORT=RECORDS/"route3-v2data-E0-rerender-2026-11-26.json"
MODE="E-H2"
WEIGHT=.1
LIMIT=5.5
BASE_ARMS=("V0_pred","v2a","E_learned","V0_GT")
NEW_ARMS=("E_B","E_B_learned","E_B_gt")
ARMS=BASE_ARMS[:3]+NEW_ARMS+(BASE_ARMS[-1],)
TITLES=("V0 预测","OVG + v2a","E-H2 + 学习撕裂","E2","E2 + 学习撕裂","E2 + GT 撕裂","V0 GT")
PHASE_OPTIONS=dict(far_f2=True,far_diagnostics=False,strict_floors=False)


def git(*args, capture=False, check=True):
    return subprocess.run(["git",*args],cwd=ROOT,text=True,check=check,
                          stdout=subprocess.PIPE if capture else None,
                          stderr=subprocess.PIPE if capture else None)


def state(phase,**details):
    save_json(RUN/"dispatch.json",dict(phase=phase,at=time.time(),pid=os.getpid(),**details))


def wait_for_geometry():
    state("waiting-for-published-3c")
    label=f"origin/main:{C3_RECORD.relative_to(ROOT)}"
    while True:
        source=C3/"adoption.json"
        if source.is_file():
            local=load(source)
            fetched=git("fetch","origin","main",check=False)
            if fetched.returncode==0:
                remote=git("show",label,check=False,capture=True)
                if remote.returncode==0:
                    official=json.loads(remote.stdout)
                    if official["decision"]!=local:
                        raise ValueError("Published 3c adoption differs from adoption.json")
                    git("merge","--ff-only","origin/main")
                    step=local["selected_step"]
                    path=path_for(C3,step) if step else data.INITIAL
                    digest=sha256(path)
                    if step==0 and not digest.startswith("dfb1d7e5"):
                        raise ValueError("Original frozen geometry SHA differs from the protocol")
                    if step:
                        adopted = C3_RECORD.with_name(f"route3-v2data-stage3c-2026-11-26-step-{step:06d}.json")
                        published = json.loads(git("show",f"origin/main:{adopted.relative_to(ROOT)}",
                                                   capture=True).stdout)
                        if published["checkpoint"]["sha256"]!=digest:
                            raise ValueError("3c adopted checkpoint SHA disagrees with the published receipt")
                    result=dict(step=step,weights="EMA",sha256=digest,path=str(path),adoption=local,
                                publication=git("rev-parse","origin/main",capture=True).stdout.strip(),
                                publication_record=str(C3_RECORD.relative_to(ROOT)))
                    save_json(RUN/"geometry.json",result)
                    return result
        status=load(C3/"dispatch.json")
        if status["phase"]=="halted":
            print("E2_WAIT Stage 3c dispatch halted; waiting for a published adoption",flush=True)
        time.sleep(30)


def source():
    choice=load(RUN/"geometry.json")
    if choice["step"]:
        private=RUN/"geometry-inputs"
        data.RUN=training.RUN=windows.RUN=private
        feature_cache.RUN=RUN/"features"
    return choice,data.RUN


def geometry(choice):
    network=load_geometry()
    if choice["step"]:
        ck=torch.load(choice["path"],map_location="cpu",weights_only=True)
        missing,extra=network.load_state_dict(ck["ema"]["geometry"],strict=False)
        if extra or any(not k.startswith("encoder.") for k in missing):
            raise ValueError("Adopted 3c EMA could not restore the OVG geometry")
    return network.eval().requires_grad_(False)


def make_link(path,target):
    if path.is_symlink():
        if path.resolve()!=target.resolve():
            raise ValueError(f"Unexpected cache link {path}")
    elif path.exists():
        raise ValueError(f"Refusing to replace {path}")
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        path.symlink_to(target)


def cache_adopted():
    choice,cache=source()
    if not choice["step"]:
        return
    start=time.monotonic()
    receipt=cache/"geometry-cache.json"
    if receipt.is_file():
        if load(receipt)["geometry_sha256"]!=choice["sha256"]:
            raise ValueError("Adopted feature-cache geometry mismatch")
        return
    make_link(feature_cache.RUN/"cache/da2-small",PREP/"cache/da2-small")
    rows=load(OLD/"data.json")["rows"]
    inventory=data.dataset()
    units=unit_paths(inventory)
    stage=dict(training=dict(max_process_gpu_gib=LIMIT))
    ref.gpu_setup(stage)
    training.gpu_gate(LIMIT)
    model=geometry(choice)
    cache.mkdir(parents=True,exist_ok=True)
    for i,row in enumerate(rows,1):
        name,frame=row["scene"],row["frame"]
        location=feature_cache.cache_path("ovg",name,frame)
        if not location.is_file():
            if shutil.disk_usage(RUN).free < 61*GIB:
                raise OSError("Insufficient disk reserve for the adopted OVG cache")
            training.gpu_gate(LIMIT)
            values=feature_cache.online_ovg(model,name,frame)
            feature_cache.write_arrays(location,feature_cache.packed(values))
        data.build_aux((name,frame,units[name],inventory["metadata"][name]))
        if i%25==0 or i==len(rows):
            print("E2_ADOPTED_OVG",i,len(rows),flush=True)
    del model
    torch.cuda.empty_cache()
    old_meta=load(OLD/"data.json")
    if not (cache/"data.json").is_file():
        save_json(cache/"data.json",old_meta)
    make_link(cache/"crop-order.npy",OLD/"crop-order.npy")
    features=windows.build(rows)
    save_json(receipt,dict(geometry_sha256=choice["sha256"],frames=len(rows),windows=features,
                           order_sha256=sha256(cache/"crop-order.npy"),seconds=time.monotonic()-start))


def qualify():
    choice,cache=source()
    ref.gpu_setup(dict(training=dict(max_process_gpu_gib=LIMIT)))
    training.gpu_gate(LIMIT)
    receipt,draws=training.checked_prepared()
    x=training.frame_batch(receipt["rows"],draws[0],MODE)
    model=training.module(MODE)
    current=training.predict(model,x)
    bce=training.normal_front_bce(current["log_mass"],x["loss_zone"])
    nll=training.terms(current,x)["nll"]
    grad=torch.autograd.grad(bce,current["log_mass"],retain_graph=True)[0]
    mask=x["loss_zone"]==2
    supported=grad.abs().sum(dim=1,keepdim=True)>0
    params=list(model.parameters())
    gradients=torch.autograd.grad(bce,params,allow_unused=True)
    finite=torch.isfinite(bce) & torch.isfinite(grad).all()
    nonfinite_param=any(not torch.isfinite(g).all() for g in gradients if g is not None)
    norm=sum(float(g.square().sum()) for g in gradients if g is not None)**.5
    if not finite or nonfinite_param or not norm or not mask.any() or not supported.equal(mask):
        raise FloatingPointError("E2 step-zero normal-zone BCE has invalid support or gradients")
    value,base=float(bce.detach()),float(nll.detach())
    loss=dict(normal_pixels=int(mask.sum()),other_pixels=int((~mask).sum()),
              gradient_outside_normal_nonzero=int((supported & ~mask).sum()),gradient_l2=norm,
              normal_front_bce=value,weighted_bce=WEIGHT*value,nll=base,
              unweighted_ratio_to_abs_nll=value/abs(base),weighted_ratio_to_abs_nll=WEIGHT*value/abs(base),
              finite=True,peak_reserved_gib=torch.cuda.max_memory_reserved()/GIB)
    original=load(RECORDS/"route3-v2data-E0-config-2026-11-26.json")
    config=dict(geometry=choice,change=dict(mask="loss_zone == 2",objective="BCE(g * (1-w_f), 0)",
                  implementation="-log_mass(front), stable FP32",weight=WEIGHT,from_step=0,both_phases=True),
               step_zero=loss,training=original["training"],inputs=original["inputs"],seed=original["seed"],
               selection=original["selection"],data=original["data"],tear_retraining=original["tear_retraining"],
               source_E_H2_configuration_sha256=sha256(RECORDS/"route3-v2data-E0-config-2026-11-26.json"),
               native_cache="Rebuilt from adopted 3c EMA and original crop sequence" if choice["step"] else
                            "Reused E0 OVG and E-H2 windows without changes",
               cache_rebuild=load(cache/"geometry-cache.json") if choice["step"] else None,
               geometry_limits=load(RECORDS/"route3-v2data-stage3-config-2026-11-24.json")[
                   "evaluation"]["relative_geometry_limits"],
               optical_protocol="P/G P2, f/1.4, FP16 fix 6cff9f9; far F2 is primary, GT substitution pending",
               training_arms=["original","normal_front_bce"] if choice["step"] else ["normal_front_bce"])
    if config["training"]["crop_order_sha256"]!=receipt["crop_order_sha256"]:
        raise ValueError("E2 lost the original sampling order")
    save_json(CONFIG,config)
    print("E2_QUALIFIED",loss,flush=True)


def arm_path(arm,cache):
    dest=RUN/arm
    make_link(dest/"data.json",cache/"data.json")
    return dest


def train_arm(arm):
    choice,cache=source()
    dest=arm_path(arm,cache)
    weight=WEIGHT if arm=="v2" else 0.
    if not choice["step"] and arm=="baseline":
        raise ValueError("Historical E-H2 must be reused on the original geometry")
    training.train(MODE,run=dest,config_path=CONFIG,normal_front_weight=weight)


def retrain(arm):
    _,cache=source()
    dest=arm_path(arm,cache)
    learned=load(dest/MODE/"training.json")
    tears.train(MODE,learned["selected"]["step"],run=dest)


def selected(dest):
    return load(dest/MODE/"training.json")["selected"]["step"]


def graphics_root():
    path=RUN/"evaluation"
    (path/"figures").mkdir(parents=True,exist_ok=True)
    return path


@torch.inference_mode()
def smoke():
    root=RUN/"qualification-replay"
    (root/"figures").mkdir(parents=True,exist_ok=True)
    replay.RUN=root
    ref.gpu_setup(dict(training=dict(max_process_gpu_gib=replay.GPU_GIB)))
    training.gpu_gate(replay.GPU_GIB)
    net=load_geometry()
    inputs=evaluation.EInputs(MODE)
    bundle=arm_bundle(net,inputs,OLD,2000,baseline=True)
    p=replay.phase(root/"P-base",bundle,BASE_ARMS,("P",),smoke=True,render_options=PHASE_OPTIONS)
    g=replay.phase(root/"fixed-base",bundle,BASE_ARMS,("G",),root/"P-base",smoke=True,render_options=PHASE_OPTIONS)
    del bundle
    bundle=arm_bundle(net,inputs,OLD,3000)
    new=replay.phase(root/"rest",bundle,NEW_ARMS,("P","G"),smoke=True,render_options=PHASE_OPTIONS)
    jobs=[j for j in ref.frames() if j[2]==30 and ref.title(j[1]) in ("dev005","122104")]
    parts=join_parts(jobs,(p,g,new))
    for phase in ("P","G"):
        for entry in parts:
            for row in entry["rows"][phase]:
                if row["focus_index"]!=1 or row["status"]!="evaluated":
                    continue
                known=load(replay.ROOT/"outputs/route3-v2data-E0-rerender-2026-11-26-01"
                           /("P-base" if phase=="P" else "fixed-base")/"frames"/entry["relative"])
                ref_row=next(r for r in known["rows"][phase] if r["case"]==row["case"])
                for arm in BASE_ARMS:
                    for key in ref.REGIONS:
                        a,b=(ref.score_psnr(r["methods"][arm]["optical"][key]) for r in (row,ref_row))
                        if (a is None)!=(b is None) or (a is not None and abs(a-b)>.01):
                            raise ValueError(f"E2 existing-arm qualification did not reproduce {arm}/{key}")
        rows=[r for entry in parts for r in entry["rows"][phase]]
        far_f2(rows)
    files=write_visuals(root,parts,total=4)
    if len(files)!=4:
        raise ValueError("Optical qualification panel count mismatch")
    print("E2_OVERVIEW_CHECK",len(parts),len(files),flush=True)


def arm_bundle(net,inputs,path,step,baseline=False):
    model,_=tears.load_model(MODE,step,run=path)
    tear=replay.load_tear(path/MODE/f"tear-{step:06d}"/"selected.pt")
    head=None
    if baseline:
        archive=load(RECORDS/"route3-v2data-module2-tear-d-v2-2026-11-25.json")["training"]
        trained=load(D_RUN/"training-report.json")
        if trained["selected"]["v2a"]!=archive["selected"]["v2a"]:
            raise ValueError("Module 2 v2a checkpoint does not match its record")
        head=predictor(chosen_heads(trained,{"v2a":CHANNELS}))
    return net,head,inputs,model,tear


def check_replay(parts,choice):
    new=replay.core_summary(parts,BASE_ARMS,"P")["development"]
    if choice["step"]:
        return dict(mode="new_adopted_geometry",P=new["methods"])
    old=load(OLD_REPORT)["results"]["P"]["development"]["methods"]
    expected=("V0_pred","v2a","E_learned")
    report={}
    for arm in expected:
        report[arm]={}
        for key in ref.REGIONS:
            x=old[arm][key]["psnr_db"]
            y=new["methods"][arm][key]["psnr_db"]
            report[arm][key]=dict(old=x,new=y,delta=y-x)
            if abs(x-y)>.01:
                raise ValueError(f"E0 P self-check differs by more than .01 dB: {arm}/{key} {x} {y}")
    return dict(mode="reproduce_E0",scores=report)


def far_f2(rows, *, arms=ARMS, new_arms=NEW_ARMS):
    by_scene=defaultdict(list)
    for row in rows:
        if row["status"]=="evaluated" and row["focus_index"]==1:
            by_scene[row["scene"]].append(row)
    pooled={a:ref.pooled([r["methods"][a]["far_off"] for rr in by_scene.values() for r in rr]) for a in arms}
    scenes={key:{a:ref.pooled([r["methods"][a]["far_off"] for r in rr]) for a in arms}
            for key,rr in by_scene.items()}
    pairs={}
    relative_to_initial={}
    for arm in arms[1:]:
        def paired(base):
            diffs={n:table[arm]["psnr_db"]-table[base]["psnr_db"] for n,table in scenes.items()
                   if table[arm]["psnr_db"] is not None}
            return dict(per_scene_db=diffs,pooled_delta_db=pooled[arm]["psnr_db"]-pooled[base]["psnr_db"],
                        **ref.distribution(list(diffs.values())))
        pairs[arm]=paired("V0_pred")
        if arm in new_arms:
            relative_to_initial[arm]=paired("E_learned")
    return dict(pooled=pooled,per_scene=scenes,paired=pairs,paired_vs_E_H2=relative_to_initial)


def join_parts(jobs,parts):
    results=[]
    for job,base,fixed,new in zip(jobs,*parts,strict=True):
        _,name,frame,*_=job
        key=Path(name)/f"f{frame:03d}.json"
        indexed=dict(P=base["rows"]["P"],G=fixed["rows"]["G"])
        row=dict(relative=str(key),rows={})
        for f,values in indexed.items():
            more={r["case"]:r for r in new["rows"][f]}
            row["rows"][f]=[]
            for v in values:
                attach=more[v["case"]]
                if attach["status"]!=v["status"] or v["scene"]!=name:
                    raise ValueError("E2 methods disagree on the shared case support")
                methods=dict(v["methods"],**attach["methods"])
                row["rows"][f].append(dict(v,methods={a:methods[a] for a in ARMS} if methods else {}))
        results.append(row)
    return results


def cohorts(parts,focus):
    rows=[r for p in parts for r in p["rows"][focus]]
    table=replay.core_summary(parts,ARMS,focus)
    for cohort,data0 in table.items():
        if cohort in ("legacy","development"):
            include=lambda r:r["cohort"]==cohort
        elif cohort=="dev008":
            include=lambda r:ref.title(r["scene"])=="dev008"
        else:
            include=lambda r:ref.title(r["scene"]) in ref.THIN_SCENES
        data0["far_f2"]=far_f2([r for r in rows if include(r)])
        data0["paired_vs_E_H2"]={
            arm:ref.compare(data0["per_scene"],"E_learned",arm,data0["methods"],include_absolute_near=True)
            for arm in NEW_ARMS}
    if (table["development"]["scored"],table["legacy"]["scored"])!=(87,18):
        raise ValueError("E2 development or legacy case coverage mismatch")
    return table


def write_visuals(root,parts,*,total=38):
    panels.RUN=root
    panels.TITLES=TITLES
    figures=[]
    for item in parts:
        row=item["rows"]["P"][0]
        slug=ref.title(row["scene"])
        if row["frame"]!=30 or slug not in replay.HIGHLIGHT:
            continue
        images={}
        for phase in ("P-base","fixed-base","rest"):
            where=root/phase/"frames"/Path(item["relative"]).with_suffix(".npz")
            with np.load(where) as values:
                images.update(values)
        for focus,rows in item["rows"].items():
            for value in rows:
                if value["status"]!="evaluated":
                    continue
                ix=value["focus_index"]
                if ix==1:
                    value=dict(value,methods={a:dict(d,optical=dict(d["optical"],
                              far_recoverable=d["far_off"])) for a,d in value["methods"].items()})
                    image_data=dict(images,**{f"{focus}/{ix}/{a}":images[f"{focus}/{ix}/{a}/F2"] for a in ARMS})
                else:
                    image_data=images
                figures.append(panels.optical_panel(slug,focus,ix,value,image_data,ARMS,far_f2=ix==1))
    if len(figures)!=total:
        raise ValueError("The requested E2 panel set is incomplete")
    return figures


@torch.inference_mode()
def assess():
    choice,cache=source()
    root=graphics_root()
    replay.RUN=root
    ref.gpu_setup(dict(training=dict(max_process_gpu_gib=replay.GPU_GIB)))
    training.gpu_gate(replay.GPU_GIB)
    net=geometry(choice)
    ref.load_geometry=lambda:net
    inputs=evaluation.EInputs(MODE,cache_run=cache)
    base=RUN/"baseline" if choice["step"] else OLD
    baseline_step=selected(base) if choice["step"] else 2000
    revised_step=selected(RUN/"v2")
    bundle=arm_bundle(net,inputs,base,baseline_step,baseline=True)
    p=replay.phase(root/"P-base",bundle,BASE_ARMS,("P",),render_options=PHASE_OPTIONS)
    checks=check_replay(p,choice)
    save_json(root/"P-self-check.json",checks)
    g=replay.phase(root/"fixed-base",bundle,BASE_ARMS,("G",),root/"P-base",render_options=PHASE_OPTIONS)
    del bundle
    torch.cuda.empty_cache()
    bundle=arm_bundle(net,inputs,RUN/"v2",revised_step)
    more=replay.phase(root/"rest",bundle,NEW_ARMS,("P","G"),render_options=PHASE_OPTIONS)
    parts=join_parts(ref.frames(),(p,g,more))
    if len(parts)!=54:
        raise ValueError("Expected 54 fixed development and continuity frames")
    d1=root/"baseline-depth.json"
    if not d1.is_file():
        if choice["step"]:
            model,_=tears.load_model(MODE,baseline_step,run=base)
            old_depth=evaluation.depth_report(root/"baseline-depth",model,inputs)
            del model
        else:
            old_depth=load(base/MODE/"evaluation-002000-depth/report.json")
        save_json(d1,old_depth)
    d2=root/"v2-depth.json"
    if not d2.is_file():
        save_json(d2,evaluation.depth_report(root/"v2-depth",bundle[3],inputs))
    summaries={f:cohorts(parts,f) for f in ("P","G")}
    base_label=f"E-H2@{baseline_step//1000}k"
    subjects={base_label:sorted((x["subject"] for x in p if x["subject"]),key=lambda r:r["frame"]),
              "E2":sorted((x["subject"] for x in more if x["subject"]),key=lambda r:r["frame"])}
    if any([r["frame"] for r in values]!=[1,30,60] for values in subjects.values()):
        raise ValueError("dev003 has an incomplete P2 subject audit")
    figures=write_visuals(root,parts)
    subject_figures=panels.draw_subjects(subjects,base_key=base_label,other_key="E2")
    save_json(root/"full-summary.json",dict(results=summaries,dev003_subject=subjects,
              figures=figures,subject_figures=subject_figures,P_self_check=checks,
              baseline_step=baseline_step,E2_step=revised_step))
    print("E2_ASSESS_COMPLETE",len(figures),flush=True)


def command(name):
    log=RUN/f"{name}.log"
    with log.open("a") as stream:
        return subprocess.Popen([sys.executable,"-u",__file__,name],cwd=ROOT,
                                stdout=stream,stderr=subprocess.STDOUT)


def execute(*commands):
    processes=[(name,command(name)) for name in commands]
    failures=[name for name,child in processes if child.wait()!=0]
    if failures:
        raise RuntimeError("Failed E2 stages: "+", ".join(failures))


def append_progress(line):
    text=PROGRESS.read_text()
    if line in text:
        return
    anchor=next(s for s in text.splitlines() if s.startswith("**新开发集 GT 隐藏层替身待查"))
    if line.startswith("**E0 第二版结果"):
        anchor=next(s for s in text.splitlines() if s.startswith("**E0 第二版登记"))
    PROGRESS.write_text(text.replace(anchor,anchor+"\n\n"+line,1))


def publish(paths,title):
    git("fetch","origin","main")
    git("status","--short")
    git("diff","--check")
    git("add","--",*[str(p.relative_to(ROOT)) for p in paths])
    git("diff","--cached","--check")
    if git("diff","--cached","--quiet",check=False).returncode:
        git("commit","-m",title,"-m","Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>")
    git("rebase","origin/main")
    git("push","origin","HEAD:main")


def report_curve(curve):
    points=[]
    for row in curve:
        h=row["holdout"]
        stats=h["errors"]
        mask=h["selection"]
        rod=[mask[f"rod/{k}"] for k in ("mv_valid","mv_invalid")]
        normal=[mask[f"normal/{k}"] for k in ("mv_valid","mv_invalid")]
        rod_pixels=sum(x["pixels"] for x in rod)
        points.append(dict(step=row["step"],phase=row["phase"],nll=h["nll"],gate=h.get("gate_by_zone"),
                    gate_global=h["gate"]["mean"],
                    rod_p50=stats["rod/decoded"]["p50"],rod_p95=stats["rod/decoded"]["p95"],
                    OVG_rod_p50=stats["rod/OVG"]["p50"],gap_p50=stats["gap/decoded"]["p50"],
                    normal_p50=stats["normal/decoded"]["p50"],boundary_p50=stats["boundary/decoded"]["p50"],
                    rod_selected_front=(rod_pixels-sum(r["rear"] for r in rod))/rod_pixels,
                    normal_selected_back={k:v["selected_back"] for k,v in zip(("mv_valid","mv_invalid"),normal)},
                    loss_zone_normal_p50=stats.get("normal_loss_zone/decoded",{}).get("p50"),
                    loss_zone_normal_selected_back={k:mask.get(f"normal_loss_zone/{k}",{}).get("selected_back")
                                                    for k in ("mv_valid","mv_invalid")}))
    if [r["step"] for r in points]!=list(range(1000,20001,1000)):
        raise ValueError("Holdout curve does not contain the registered checkpoints")
    return points


def text_summary(report):
    scope_zh=dict(development="新开发集",legacy="原 3 场景",thin_structure="细结构",dev008="dev008")
    keys=ref.REGIONS
    fmt=lambda x:"—" if x is None else f"{x:.3f}"
    percent=lambda x:"—" if x is None else f"{100*x:.2f}%"
    lines=["# E0 第二版","",
           f"E2 为第 {report['E2_step']:,} 步 EMA。比较基线为 E-H2 {report['baseline_step']:,} 步 EMA。",
           report["selection_reason"],"",
           "## 留出曲线","",
           "正常区的历次对照沿用原曲线口径。E2 另列 loss_zone == 2 严格区，原存档无严格区和分区 g。",
           "",
           "| 组 | 步数 | 期 | 杆 p50 / p95 | 缝 p50 | 正常 p50 旧／严格 | 边界 p50 | NLL | 杆选前 | 正常选后有效／无效 | 严格正常选后有效／无效 | 杆／缝／正常／边界 g |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for step,base in zip(report["curve"],report["baseline_curve"],strict=True):
        for name,row in (("E2",step),("原 E-H2",base)):
            gate=row["gate"]
            gate_text=" / ".join(fmt(gate[k]) for k in training.GROUPS) if gate else "—"
            rear=" / ".join(percent(row["normal_selected_back"][k]) for k in ("mv_valid","mv_invalid"))
            strict_rear=" / ".join(percent(row["loss_zone_normal_selected_back"][k])
                                   for k in ("mv_valid","mv_invalid"))
            lines.append(f"| {name} | {row['step']} | {row['phase']} | {fmt(row['rod_p50'])} / {fmt(row['rod_p95'])} | "
                         f"{fmt(row['gap_p50'])} | {fmt(row['normal_p50'])} / {fmt(row['loss_zone_normal_p50'])} | "
                         f"{fmt(row['boundary_p50'])} | {fmt(row['nll'])} | "
                         f"{percent(row['rod_selected_front'])} | {rear} | {strict_rear} | {gate_text} |")
    lines.extend(["","## 新开发集深度","",
                  "| 区域 | OVG p50 / p95 | E-H2 p50 / p95 | E2 p50 / p95 |","|---|---:|---:|---:|"])
    base=report["depth"]["baseline"]["cohorts"]["development"]
    new=report["depth"]["E2"]["cohorts"]["development"]
    for word in training.GROUPS:
        v0,v1,v2=(d["errors"][f"{word}/{layer}"] for d,layer in ((base,"OVG"),(base,"decoded"),(new,"decoded")))
        lines.append("| "+word+" | "+" | ".join(f"{fmt(v['p50'])} / {fmt(v['p95'])}" for v in (v0,v1,v2))+" |")
    lines.extend(["","正常区选后分量比例与解码跳变率见指标记录。","",
                  "| 护栏 | OVG | E-H2 | E2 | E2 相对 OVG | 上限 | 通过 |","|---|---:|---:|---:|---:|---:|---|"])
    base_guard=report["depth"]["baseline"]["guards"]["development"]
    new_guard=report["depth"]["E2"]["guards"]["development"]
    for k in evaluation.GUARDS:
        base=base_guard["OVG"][k]
        new=new_guard["E"][k]
        limit=report["geometry_limits"][k]
        lines.append(f"| {k} | {fmt(base)} | {fmt(base_guard['E'][k])} | {fmt(new)} | {new/base-1:+.2%} | "
                     f"{limit:+.0%} | {'是' if new<=base*(1+limit) else '否'} |")
    lines.extend(["","## 光学对照","","近焦以 P 为准。新开发集远焦以 F2 为准。含替身列保留待查标记。"])
    for focus,table in report["optical"].items():
        for scope,group in table.items():
            lines.extend(["",f"### {focus} · {scope_zh[scope]}","",
                          "| 档 | 近焦 | 远焦含替身 | 远焦 F2 | 边界 | 全图 | 近焦／远焦 F2 配对中位 |",
                          "|---|---:|---:|---:|---:|---:|---:|"])
            for i,(arm,caption) in enumerate(zip(ARMS,TITLES),1):
                scores=group["methods"][arm]
                f2=group["far_f2"]["pooled"][arm]["psnr_db"]
                near_median=group["paired"].get(arm,{}).get(keys[0],{}).get("median_db")
                far_median=group["far_f2"]["paired"].get(arm,{}).get("median_db")
                lines.append(f"| {i} {caption} | {fmt(scores[keys[0]]['psnr_db'])} | {fmt(scores[keys[1]]['psnr_db'])} | {fmt(f2)} | "
                             f"{fmt(scores[keys[2]]['psnr_db'])} | {fmt(scores[keys[3]]['psnr_db'])} | "
                             f"{fmt(near_median)} / {fmt(far_median)} |")
    lines.extend(["","## dev003 远焦 P2 主体","",
                  "| 检查点 | 帧 | 杆 / 缝 / 正常 / 边界 | 前 / 后分量 | ρ₂ / DS / 三角化 | OVG ρ₁ / E ρc | GT 中心深度 m | 实际 P2 m |",
                  "|---|---:|---|---|---|---|---:|---:|"])
    for name,probe in report["dev003_subject"].items():
        for row in probe:
            region=" / ".join(str(v) for v in row["regions"].values())
            source=" / ".join(str(v) for v in row["beta_source"].values())
            m=row["medians"]
            lines.append(f"| {name} | {row['frame']} | {region} | {row['selected_front']} / {row['selected_back']} | "
                         f"{source} | {fmt(m['OVG rho1 (m^-1)'])} / {fmt(m['E rho_c (m^-1)'])} | "
                         f"{fmt(m['GT central axial depth (m)'])} | {fmt(row['actual_P2_focus_m'])} |")
    return "\n".join(lines)+"\n"


def collect():
    root=graphics_root()
    choice=load(RUN/"geometry.json")
    metrics=load(root/"full-summary.json")
    v2=load(RUN/"v2"/MODE/"training.json")
    old=RUN/"baseline" if choice["step"] else OLD
    baseline=load(old/MODE/"training.json")
    step=v2["selected"]["step"]
    base_step=metrics["baseline_step"]
    tear=load(RUN/"v2"/MODE/f"tear-{step:06d}"/"training.json")
    base_tear=load(old/MODE/f"tear-{base_step:06d}"/"training.json")
    curve=report_curve(v2["curve"])
    eligible=[r["step"] for r in curve if r["rod_p50"]<=r["OVG_rod_p50"]]
    reason=("在杆 p50 过线的留出检查点中选 NLL 最低者。"
            if eligible else "留出曲线上没有杆 p50 低于或等于 OVG 的检查点，回退到杆 p50 最低者。")
    records={}
    for focus,cohorts0 in metrics["results"].items():
        records[focus]={name:{k:v for k,v in scope.items() if k!="per_scene"} for name,scope in cohorts0.items()}
    baselines=report_curve(baseline["curve"])
    v2_path=RUN/"v2"/MODE/f"step-{step:06d}.pt"
    report=dict(geometry=choice,geometry_limits=load(CONFIG)["geometry_limits"],
                training={k:v for k,v in v2.items() if k not in ("curve","selected")},
                curve=curve,baseline_curve=baselines,E2_step=step,baseline_step=base_step,
                E2_sha256=sha256(v2_path),selection_reason=reason,rod_eligible_steps=eligible,
                tear={k:dict(selected=v["selected"],config=v["config"],wall_seconds=v["wall_seconds"])
                      for k,v in (("baseline",base_tear),("E2",tear))},
                depth={"baseline":load(root/"baseline-depth.json"),"E2":load(root/"v2-depth.json")},
                optical=records,self_check=metrics["P_self_check"],figures=metrics["figures"],
                subject_figures=metrics["subject_figures"],dev003_subject=metrics["dev003_subject"],
                test_cases_used=0,far_primary="F2 only",substitute="GT hidden substitute pending audit")
    save_json(RESULT,report)
    (root/"汇总.md").write_text(text_summary(report))
    links=[*report["subject_figures"],*report["figures"]]
    (root/"index.html").write_text('<!doctype html><meta charset="utf-8"><title>E0 第二版</title>'
        '<h1>E0 第二版</h1><p>远焦面板为仅 F2。dev010 无有效近焦主体。</p><p><a href="汇总.md">汇总</a></p>'
        +"\n".join(f'<figure><a href="{escape(f)}"><img width="870" loading="lazy" src="{escape(f)}"></a>'
                    f'<figcaption>{escape(Path(f).stem)}</figcaption></figure>' for f in links))
    d=records["P"]["development"]
    near=d["methods"]["E_B_learned"]["near_in_focus"]["psnr_db"]
    far=[records[f]["development"]["far_f2"]["pooled"]["E_B_learned"]["psnr_db"] for f in ("P","G")]
    append_progress(f"**E0 第二版结果（2026-11-26）**。E2 选中 {step:,} 步 EMA，重训撕裂头选中 {tear['selected']['step']:,} 步，"
                    f"开发集近焦 P 为 {near:.2f} dB，远焦仅 F2 的 P／G 为 {far[0]:.2f}／{far[1]:.2f} dB。"
                    "采用结论待定。 [曲线、深度与光学记录](records/route3-v2data-E0-v2-2026-11-26.json)")
    publish((RESULT,PROGRESS),"记录 E0 第二版训练、撕裂头与 P/G 光学对照")


def follow():
    RUN.mkdir(parents=True,exist_ok=True)
    choice=wait_for_geometry()
    if choice["step"]:
        state("rebuilding-adopted-cache",step=choice["step"])
        execute("cache")
    state("step-zero-check")
    execute("qualify")
    cfg=load(CONFIG)
    region=cfg["step_zero"]
    initial="阶段 3c 采用的 EMA" if choice["step"] else "同表面支 30k EMA，复用原缓存"
    append_progress(f"**E0 第二版登记（2026-11-26）**。3c 的 10k 判定已发布。冻结{initial}（{choice['sha256'][:8]}）。"
                    f"仅在 GT 正常区从第 0 步加入 0.1 倍朝前分量 BCE；首批加权损失／|NLL| 为 {region['weighted_ratio_to_abs_nll']:.2%}，"
                    "区外梯度为零。远焦以仅 F2 为准。 [配置与零步检查](records/route3-v2data-E0-v2-config-2026-11-26.json)")
    publish((CONFIG,PROGRESS),"登记 E0 第二版的正常区选前约束及零步检查")
    arms=("train-baseline","train-v2") if choice["step"] else ("train-v2",)
    state("training",arms=arms)
    execute(*arms)
    state("tear-retraining")
    execute(*(("tear-baseline","tear-v2") if choice["step"] else ("tear-v2",)))
    state("depth-and-optics")
    execute("assess")
    state("collecting")
    collect()
    state("complete",commit=git("rev-parse","HEAD",capture=True).stdout.strip())


COMMANDS={"follow":follow,"cache":cache_adopted,"qualify":qualify,"assess":assess,"collect":collect,"smoke":smoke,
          "train-v2":lambda:train_arm("v2"),"train-baseline":lambda:train_arm("baseline"),
          "tear-v2":lambda:retrain("v2"),"tear-baseline":lambda:retrain("baseline")}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=tuple(COMMANDS))
    args=parser.parse_args()
    try:
        COMMANDS[args.command]()
    except Exception as error:
        if args.command=="follow":
            state("halted",reason=str(error))
        raise
