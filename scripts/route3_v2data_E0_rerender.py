"""Replay the E0 geometry and tear arms with paired P, G, and V focus distances."""
import argparse
from dataclasses import replace
import os
import time

import numpy as np
import psutil
import torch

import route3_ovg_data as ovg
import route3_v2data_module2_infer as ref
from route3_v2data_B_batch import sha256
from route3_v2data_E0_data import ROOT,RUN as E0
from route3_v2data_E0_evaluate import EInputs
from route3_v2data_E0_reselect import RECORD as SELECTION
from route3_v2data_E0_tears import INPUT_CHANNELS,features,load_model
from route3_v2data_E0_train import gpu_gate,predict
from route3_v2data_E_labels import load,save_json
from route3_v2data_module2_d_evaluate import chosen_heads,predictor
from route3_v2data_module2_d_v2_train import CHANNELS,RUN as D_RUN
from route3_v2data_stage3b import load_geometry
from src.ovg_tear_head import TearHead

RUN=ROOT/"outputs/route3-v2data-E0-rerender-2026-11-26-01"
STAGE3C=ROOT.parent/"route3-stage3c-20261126/outputs/route3-v2data-stage3c-2026-11-26-01"
GPU_GIB=7
METHODS=("V0_pred","v2a","b","E","E_learned","E_rule","E_gt","V0_GT","a","E_B","E_B_learned","E_B_gt")
TITLES=("V0 预测","OVG + (d) v2a","OVG + GT 撕裂","E-H2 2k","E 2k + 学习撕裂","E 2k + 规则 .10",
        "E 2k + GT 撕裂","V0 GT","V0 GT + GT 撕裂","E B 3k","E B 3k + 学习撕裂","E B 3k + GT 撕裂")
GEOMETRY=dict.fromkeys(METHODS[:3],"V") | dict.fromkeys(METHODS[3:7],"E") | dict.fromkeys(METHODS[7:9],"G") | dict.fromkeys(METHODS[9:],"B")
HIGHLIGHT=("dev000","dev001","dev003","dev004","dev005","dev007","dev008","dev009","dev010","dev011")
DIAGNOSTIC_ARMS=("V0_pred","E_learned","V0_GT","a")
EXPECTED=((35.135,24.125),(35.899,23.995),(36.170,23.887),(34.964,28.254),
          (35.855,28.333),(35.268,28.205),(36.136,28.291),(39.429,30.365))


def state(phase,**fields):
    save_json(RUN/"dispatch.json",dict(phase=phase,at=time.time(),pid=os.getpid(),**fields))


def stage3c_snapshot():
    return dict(wall_time=time.time(),progress=load(STAGE3C/"progress.json"),
                dispatch=load(STAGE3C/"dispatch.json"))


def case_focus(depths,subject):
    return {name:ref.p2_focus(depth,subject) for name,depth in depths.items()}


def load_tear(checkpoint):
    state=torch.load(checkpoint,map_location="cpu",weights_only=True)
    tear=TearHead(INPUT_CHANNELS).cuda().eval().requires_grad_(False)
    tear.load_state_dict(state["ema"])
    return tear


def models(mode,step,base):
    ovg_net=load_geometry()
    if base:
        mode,step="E-H2",2000
        source=ROOT/"03_实验记录/records/route3-v2data-module2-tear-d-v2-2026-11-25.json"
        archived=load(source)["training"]
        training=load(D_RUN/"training-report.json")
        if archived["selected"]["v2a"]!=training["selected"]["v2a"]:
            raise ValueError("v2a checkpoint differs from the registered Module 2 arm")
        ovg_tear=predictor(chosen_heads(training,{"v2a":CHANNELS}))
    else:
        ovg_tear=None
    e,_=load_model(mode,step)
    tear=load_tear(E0/mode/f"tear-{step:06d}/selected.pt")
    return ovg_net,ovg_tear,EInputs(mode),e,tear


def make_visible(color,depth,K,tear):
    if tear is None:
        options={}
        maximum=depth
    else:
        inverse,_,_=ref.subdivide(depth.reciprocal().cpu().numpy(),tear)
        if not np.array_equal(inverse[...,2,2],depth.reciprocal().cpu().numpy()):
            raise ValueError("Connected interpolation changed the P2 centers")
        sub=1/inverse
        sub[...,2,2]=depth.cpu().numpy()
        maximum=K.new_tensor(sub.max(axis=(2,3)))
        options=dict(subpixel_depth=K.new_tensor(sub.transpose(0,2,1,3).reshape(2560,2560)))
    visible=ref.pixel_layer(color,depth,K,torch.eye(4,device=K.device,dtype=K.dtype),**options)
    return visible,maximum


def near_trace(unit,entry,name,frame,case,x,y,native,K,truth,visible,hidden):
    labelled=ref.k16(unit,name,frame,case)
    z=labelled["depth"]
    good=((labelled["ids"]==native["object_id"][None])&labelled["known"]&~labelled["hidden"]&np.isfinite(z)&(z>0))
    interior=(y["alpha"][0].cpu().numpy()>=.9999)&y["valid_alpha"][0].cpu().numpy()&~native["boundary"]
    support=interior&native["valid"]&np.isin(native["object_id"],native["foreground"])&good.all(0)
    offsets=K.new_tensor(labelled["aperture_xy"])
    nominal=case["focus_m"]
    fill=ref.f2_for(unit,entry,frame,x,nominal,offsets)
    return ref.ray_diagnostics(visible,hidden,K,nominal,offsets,fill,support,truth)[0]


def far_trace(visible,hidden,K,focus,offsets,fill,mask,truth,score):
    support=torch.as_tensor(mask,device=K.device)
    total=len(offsets)*int(mask.sum())
    t_sum=hits=0
    if not total:
        return dict(rays=0,residual_sum=0.,hidden_hits=0,hidden_significant=0,off=score,on=score)
    combined=None
    if hidden is not None:
        cloud=ref.merge_layers(visible,hidden)
        indicator=cloud.colors.new_zeros((len(cloud.means),1))
        indicator[len(visible.means):]=1.
        combined=replace(cloud,colors=indicator)
    off=K.new_zeros((512,512,3))
    meaningful=0
    for rays,bg in zip(offsets.split(4),fill.split(4)):
        value,alpha=ref.raster(visible,K,focus,rays,512,512,bg,return_alpha=True)
        off+=value.clamp(0,1).sum(0)
        t_sum+=float((1-alpha[:,support]).double().sum())
        if combined is not None:
            weight=ref.raster(combined,K,focus,rays,512,512)[...,0][:,support]
            hits+=int((weight>0).sum())
            meaningful+=int((weight>=.001).sum())
    return dict(rays=total,residual_sum=t_sum,hidden_hits=hits,hidden_significant=meaningful,
                off=ref.region_sse(ref.display(off/len(offsets)),truth,mask),on=score)


def zero_scores(visible,hidden,K,focus,center,fill,pinhole,full,common,zv):
    composed=ref.display(ref.raster(ref.merge_layers(visible,hidden),K,focus,center,512,512,fill)[0])
    return {k:ref.region_sse(image,pinhole,mask) for k,image,mask in
            (("zero_visible",zv,full),("zero_combined",composed,full),
             ("zero_visible_common",zv,common),("zero_combined_common",composed,common))}


def check_zero(old,new,scene,case,arm,*,strict_floors=True):
    for name,floor in ref.FLOORS.items():
        a,b=(values[name] for values in (old,new))
        if strict_floors and b["pixels"] and min(ref.score_psnr(a),ref.score_psnr(b))<floor:
            raise ValueError(f"Fixed-focus {name} did not pass for {scene}/{case}/{arm}: {a}, {b}")
        if a["pixels"]!=b["pixels"] or not np.isclose(a["sse"],b["sse"],rtol=1e-5,atol=1e-9):
            raise ValueError(f"Pinhole changed with focus for {scene}/{case}/{arm}/{name}")


def fixed_panel(name,frame):
    return frame==30 and ref.title(name) in HIGHLIGHT


def frame_paths(folder,name,frame):
    path=folder/"frames"/name/f"f{frame:03d}.json"
    return path,path.with_suffix(".npz")


def save_arrays(path,values):
    if not values:
        return
    temporary=path.with_suffix(".part")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream,**values)
    temporary.replace(path)


def subject_probe(folder,scene,frame,context,decoded,subject,depths,native):
    fields={
        "OVG rho1 (m^-1)":context["rho1"][0,0].cpu().numpy(),
        "E rho_c (m^-1)":decoded["decoded"][0,0].cpu().numpy(),
        "GT central axial depth (m)":native["gt"].astype(np.float32),
        "OVG axial depth (m)":depths["V"],
        "E axial depth (m)":depths["E_current"],
    }
    rear=decoded["selected_back"][0,0].cpu().numpy()
    source=context["mv_source"][0,0].cpu().numpy()
    n=int(subject.sum())
    path=folder/"subjects"/f"{scene}-f{frame:03d}.npz"
    path.parent.mkdir(parents=True,exist_ok=True)
    save_arrays(path,dict(subject=subject,**fields))
    row=dict(scene=scene,frame=frame,pixels=n,selected_front=int((subject&~rear).sum()),
                selected_back=int((subject&rear).sum()),
                beta_source={name:int((subject&(source==i)).sum()) for i,name in enumerate(("rho2","DepthSplat","triangulation"))},
                medians={name:float(np.median(value[subject])) if n else None for name,value in fields.items()},
                actual_P2_focus_m=ref.p2_focus(depths["E_current"],subject),
                panel_data=str(path.relative_to(RUN)))
    if "loss_zone" in context:
        zones=context["loss_zone"][0,0].cpu().numpy()
        row["regions"]={name:int((subject&(zones==i)).sum())
                        for i,name in enumerate(("rod","gap","normal","boundary"))}
    return row


@torch.inference_mode()
def render_frame(folder,bundle,job,arms,conventions,previous=None,*,far_f2=False,
                 far_diagnostics=True,strict_floors=True,extra_far=None):
    cohort,name,frame,unit,entry,cases=job
    ovg_net,ovg_tear,inputs,e,tear=bundle
    item=("B",name,ovg.TRAIN_FRAMES.index(frame))
    x,y=ref.sample(item)
    pred=ovg_net(x,protect_residual=True,render_heads=False,return_features=True)
    K=x["K"][0]
    k=K.cpu().numpy()
    native=ovg.b_edge_reference(item,y)
    sensor=torch.as_tensor(native["gt"],device=K.device,dtype=K.dtype)
    known=torch.as_tensor(native["valid"],device=K.device)&torch.isfinite(sensor)&(sensor>0)
    front=pred["depth1"][0]
    features0=inputs.observations(name,frame,x,pred)
    decoded=predict(e,features0)
    rho=decoded["decoded"][0,0]
    if not torch.isfinite(rho).all() or not (rho>0).all():
        raise FloatingPointError(f"Invalid E geometry for {name}/{frame}")
    current="E" if ovg_tear is not None else "B"
    depths={"V":front,"G":torch.where(known,sensor,front),current:rho.reciprocal()}
    arrays={name0:d.cpu().numpy() for name0,d in depths.items()}
    label=ovg.load(ref.CACHE/name/f"f{frame:03d}.npz")["status"]
    learn=(tear(features(decoded,x["rgb"]).half().float()).sigmoid()[0]>.5).cpu().numpy().astype(np.int8)
    tears=({"b":label,"a":label,"E_gt":label,"E_rule":ref.rule_tears(rho.cpu().numpy(),.10),
            "E_learned":learn,**ovg_tear(x,pred)} if ovg_tear is not None else {"E_B_learned":learn,"E_B_gt":label})
    save_panel=fixed_panel(name,frame)
    overlays={}
    if save_panel:
        zones=ref.subsets(unit,name,frame,native)
        for arm,t in tears.items():
            surface=1/arrays[GEOMETRY[arm]]
            _,_,map0=ref.tear_audit(label,surface,dict(ref.labels_for_arm(label,surface),**{arm:t}),zones,(arm,))
            drawn=np.zeros((*surface.shape,3),np.uint8)
            for i,(dy,dx) in enumerate(ref.DIRECTIONS):
                a,_=ref.edge_slices(dy,dx)
                ref.mark_overlay(drawn,t[i][a]==1,a,dx,dy,(35,190,145))
            active=map0[arm].any(-1)
            drawn[active]=map0[arm][active]
            overlays[arm]=drawn
    color=x["rgb"][0].permute(1,2,0)
    pinhole=ref.display(color)
    full=np.ones_like(native["valid"],bool)
    edge=ref.bands(native,y)[0]
    common=ref.common_split(y,native,item)
    center=K.new_zeros((1,2))
    previous_rows={} if previous is None else {r["case"]:r for r in previous["rows"]["P"]}
    rows={kind:[] for kind in conventions}
    pending=[]
    pictures={"rod_gap":zones["foreground"]|zones["gap_background"]} if save_panel else {}
    probe=None
    for case in cases:
        tag=case["id"]
        idx=case["focus_index"]
        subject=ref.near_subject(native["gt"],native["valid"],k,case["focus_m"])
        focus=case_focus(arrays,subject)
        if "P" in conventions and idx==1 and ref.title(name)=="dev003":
            probe=subject_probe(folder,name,frame,features0,decoded,subject,dict(arrays,E_current=arrays[current]),native)
        work={kind:dict(scene=name,cohort=cohort,frame=frame,case=tag,focus_index=idx,GT_focus_m=focus["G"],
                        status="unscored" if not subject.any() else "evaluated",methods={}) for kind in conventions}
        for kind,row in work.items(): rows[kind].append(row)
        if not subject.any():
            if save_panel:
                pictures[f"{idx}/unscored"]=np.array(True)
            continue
        if any(f is None for f in focus.values()):
            raise ValueError(f"P2 subject exists but its focus could not be computed: {name}/{tag}")
        truth=ref.optical_target(unit,name,case)
        regions=ref.new_regions(unit,case,native["gt"],native["valid"],k) if unit else ref.legacy_masks(ref.geometry_root(name),case)
        empty=np.zeros_like(edge)
        masks=dict(boundary_2px=edge,full_image=full,near_in_focus=regions["in_focus"] if idx==0 else empty,
                   far_recoverable=regions["recoverable"] if idx==1 else empty)
        if save_panel:
            pictures.update({f"{idx}/truth":truth.astype(np.float16),f"{idx}/edge":edge,
                             f"{idx}/target":masks["near_in_focus" if idx==0 else "far_recoverable"]})
        offsets=ref.aperture_disk(64,.05/(2*case["fstop"]),device=K.device,dtype=K.dtype)
        pending.append((case,focus,truth,masks,work,offsets))
    del pred,features0,decoded
    old_arrays=None
    if save_panel and previous is not None:
        path,_=frame_paths(folder,name,frame)
        old_file=path.relative_to(folder)
        with np.load(folder.parent/("P-base" if ovg_tear else "rest")/old_file.with_suffix(".npz")) as cached:
            old_arrays=dict(cached)
    for arm in arms:
        gpu_gate(GPU_GIB)
        source=GEOMETRY[arm]
        z=depths[source]
        visible,maximum=make_visible(color,z,K,tears.get(arm))
        zv=ref.display(ref.raster(visible,K,1.4,center,512,512)[0])
        if save_panel:
            pictures[f"depth/{arm}"]=(1/arrays[source]).astype(np.float16)
            if arm in overlays:
                pictures[f"tear/{arm}"]=overlays[arm]
        for case,focus,truth,masks,work,offsets in pending:
            idx=case["focus_index"]
            tag=case["id"]
            method0=previous_rows.get(tag,{}).get("methods",{}).get(arm)
            seen={} if method0 is None else {method0["focus_m"]:(method0,None,None,{})}
            hidden=ref.place_hidden(unit,name,frame,case,native,z,maximum,K)
            rays=None if idx else method0["rays"] if method0 else near_trace(
                unit,entry,name,frame,case,x,y,native,K,truth,visible,hidden)
            for kind,row in work.items():
                target=focus[source if kind=="P" else kind]
                if target in seen:
                    method,on,off,extras=seen[target]
                    if on is None and save_panel and kind!="V":
                        on=old_arrays[f"P/{idx}/{arm}"]
                        if far_f2 and idx==1:
                            off=old_arrays[f"P/{idx}/{arm}/F2"]
                else:
                    extras={}
                    fill=ref.f2_for(unit,entry,frame,x,target,offsets)
                    on=ref.display(ref.light(visible,hidden,K,target,offsets,fill))
                    off=None
                    if idx==0 or far_f2:
                        off=ref.display(ref.light(visible,None,K,target,offsets,fill)) if hidden is not None else on
                    zeros=zero_scores(visible,hidden,K,target,center,ref.f2_for(unit,entry,frame,x,target,center),
                                      pinhole,full,common,zv)
                    method=dict(focus_m=target,GT_focus_m=row["GT_focus_m"],
                        optical={k0:ref.region_sse(on,truth,mask) for k0,mask in masks.items()},
                        near_off=ref.region_sse(off,truth,masks["near_in_focus"]) if idx==0 else None,
                        rays=rays,**zeros)
                    if far_f2 and idx==1:
                        method["far_off"]=ref.region_sse(off,truth,masks["far_recoverable"])
                        if extra_far is not None:
                            extras=extra_far((visible,z,maximum),unit,name,frame,native,case,K,target,
                                             offsets,fill,on,off)
                            mask=masks["far_recoverable"]
                            method.update({k:ref.region_sse(image,truth,mask) for k,image in extras.items()})
                    if any(method["optical"][k0]["pixels"]!=int(mask.sum()) for k0,mask in masks.items()):
                        raise ValueError(f"Nonfinite optical support: {name}/{tag}/{arm}/{kind}")
                    seen[target]=(method,on,off,extras)
                if kind=="P":
                    method0=method
                else:
                    check_zero(method0,method,name,tag,arm,strict_floors=strict_floors)
                row["methods"][arm]=dict(method)
                if far_diagnostics and idx==1 and kind=="G" and arm in DIAGNOSTIC_ARMS:
                    fill=ref.f2_for(unit,entry,frame,x,target,offsets)
                    row["methods"][arm]["far_diagnostic"]=far_trace(visible,hidden,K,target,offsets,fill,
                            masks["far_recoverable"],truth,method["optical"]["far_recoverable"])
                if save_panel and kind!="V":
                    pictures[f"{kind}/{idx}/{arm}"]=on.astype(np.float16)
                    if far_f2 and idx==1:
                        pictures[f"{kind}/{idx}/{arm}/F2"]=off.astype(np.float16)
                        for key,image in extras.items():
                            pictures[f"{kind}/{idx}/{arm}/{key}"]=image.astype(np.float16)
        del visible,maximum,hidden
    return dict(rows=rows,subject=probe),pictures


def phase(folder,bundle,arms,conventions,earlier=None,smoke=False,*,render_options=None):
    rows=[]
    for job in ref.frames():
        cohort,name,frame,unit,entry,cases=job
        if smoke and (ref.title(name) not in ("dev005","122104") or frame!=30):
            continue
        dest,picture=frame_paths(folder,name,frame)
        if not dest.is_file():
            previous=load(frame_paths(earlier,name,frame)[0]) if earlier else None
            if earlier and not previous:
                raise FileNotFoundError(f"Missing earlier P score for {name}/{frame}")
            entry,pictures=render_frame(folder,bundle,job,arms,conventions,previous,**(render_options or {}))
            dest.parent.mkdir(parents=True,exist_ok=True)
            save_arrays(picture,pictures)
            save_json(dest,entry)
        rows.append(load(dest))
        print("E0_RERENDER",folder.name,len(rows),ref.title(name),frame,flush=True)
    return rows


def core_summary(parts,arms,convention):
    return ref.summarize([{"rows":p["rows"][convention]} for p in parts],arms,
                         pairings={a:"V0_pred" for a in arms if a!="V0_pred"},include_absolute_near=True)


def guard_self_check(parts):
    metrics=core_summary(parts,METHODS[:9],"P")["development"]["methods"]
    result={}
    for arm,expected in zip(METHODS,EXPECTED):
        observed=[metrics[arm][k]["psnr_db"] for k in ref.REGIONS[:2]]
        result[arm]=dict(expected=expected,actual=observed,passed=all(abs(x-y)<=.01 for x,y in zip(observed,expected)))
    save_json(RUN/"P-self-check.json",result)
    if not all(r["passed"] for r in result.values()):
        raise ValueError(f"Archived P matching failed: {result}")


def wait_tear(path):
    if path.is_file():
        return
    state("waiting-E_B-tear")
    torch.cuda.empty_cache()
    pid=int((RUN/"tear.pid").read_text())
    while not path.is_file():
        try:
            if not psutil.Process(pid).is_running() or psutil.Process(pid).status()==psutil.STATUS_ZOMBIE:
                raise RuntimeError("E_B tear trainer stopped; see tear.log")
        except psutil.NoSuchProcess as error:
            raise RuntimeError("E_B tear trainer stopped; see tear.log") from error
        time.sleep(15)


def run(smoke):
    selected=load(SELECTION)["selected"]
    mode,step=selected["mode"],selected["step"]
    include_new=(mode,step)!=("E-H2",2000)
    root=RUN/"smoke" if smoke else RUN
    root.mkdir(parents=True,exist_ok=True)
    ref.gpu_setup(dict(training=dict(max_process_gpu_gib=GPU_GIB)))
    gpu_gate(GPU_GIB)
    baseline=root/"P-base"
    fixed=root/"fixed-base"
    if not smoke:
        save_json(RUN/"3c-P-start.json",stage3c_snapshot())
        state("P-base",total_frames=54)
    bundle=models("E-H2",2000,True)
    first=phase(baseline,bundle,METHODS[:9],("P",),smoke=smoke)
    if not smoke:
        guard_self_check(first)
        save_json(RUN/"3c-fixed-start.json",stage3c_snapshot())
        state("fixed-base",total_frames=54)
    phase(fixed,bundle,METHODS[:9],("G","V"),baseline,smoke=smoke)
    del bundle
    torch.cuda.empty_cache()
    if smoke:
        return
    if include_new:
        wait_tear(E0/mode/f"tear-{step:06d}/training.json")
        save_json(RUN/"3c-rest-start.json",stage3c_snapshot())
        state("rest",total_frames=54)
        bundle=models(mode,step,False)
        phase(root/"rest",bundle,METHODS[9:],("P","G","V"))
    save_json(RUN/"3c-render-end.json",stage3c_snapshot())
    state("ready-to-summarize",include_new=include_new)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke",action="store_true")
    a=p.parse_args()
    try:
        run(a.smoke)
    except Exception as error:
        if not a.smoke:
            state("halted",reason=str(error))
        raise
