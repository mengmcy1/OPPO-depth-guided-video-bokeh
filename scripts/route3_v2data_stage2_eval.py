"""P2 paired appearance evaluation, with GT hidden-layer and ray diagnostics."""
import argparse
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
import time

import cv2
import numpy as np
import torch

import route3_ovg_data as ovg
import route3_v2data_stage2_train as training
from route3_ldr_u1r import model as lpips_model
from route3_ovg_1b_optical_data import background_samples, geometry_root, place
from route3_ovg_1c_evaluate import case_frames
from route3_ovg_opaque_b0 import RAY_CLASSES, bins
from route3_ovg_phase2_labels import image as rgba
from route3_ovg_stage2_evaluate import ray_effect, trace_predicted
from route3_s2a_evaluate import masks as legacy_masks
from route3_upper_bound_evaluate import display, exr, score
from route3_v2data_B_batch import FORMAL_DATA, GIB, reserve, write_json
from route3_v2data_B_inputs import library_unit_paths
from route3_v2data_B_plan import load
from route3_v2data_loader import b_k16_cases
from route3_v2data_optics import observed_background, ray_fields
from route3_v2data_query_retrain import distribution
from route3_v2data_stage1_evaluate import (
    FRAMES, development_cases, near_subject, new_regions, p2_focus, pooled, region_errors,
)
from src.aperture_renderer import aperture_disk
from src.dual_layer import depth_band
from src.layered_aperture import merge_layers
from src.ovg_opaque_eval import opaque_view, subpixel_flags
from src.ovg_optics import raster

LEVELS = ("0", "1_to_2", "3_to_5", "over_5")
MODES = ("none", "oracle", "retrieved")
SCORES = ("near_in_focus", "near_recoverable", "far_recoverable", "boundary_2px")
GUARDS = {"zero_visible":50., "zero_combined":60.}


def frames():
    for unit in library_unit_paths("dev"):
        cases = [c for c in development_cases(unit) if c["fstop"]==1.4]
        for frame in FRAMES:
            yield "development",unit.name,frame,unit,None,[c for c in cases if c["frame"]==frame]
    for name,frame,entry,_,_,_,_,original,_ in case_frames():
        if frame in FRAMES:
            cases = [dict(c["case"],focus_m=c["case"]["nominal_focus_m"],
                          focus_index=int(c["case"]["nominal_focus_m"]!=1.4))
                     for c in original if c["case"]["protocol"]=="P2" and c["case"]["fstop"]==1.4]
            yield "legacy",name,frame,None,entry,cases


def bands(reference, y):
    center = reference["boundary"] & reference["valid"]
    d2 = cv2.dilate(center.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
    d5 = cv2.dilate(center.astype(np.uint8),np.ones((11,11),np.uint8)).astype(bool)
    masks = dict(zip(LEVELS,(center,d2&~center,d5&~d2,~d5)))
    ref_valid = reference["valid"] & y["valid_alpha"][0].cpu().numpy()
    interior = ref_valid & (y["alpha"][0].cpu().numpy()>=.9999) & ~d2
    return d2,masks,interior


def optical_target(unit, name, case):
    if unit:
        return display(ovg.load(unit/"labels"/(case["id"]+"-optics.npz"),("rgb_linear",))["rgb_linear"].astype(np.float32))
    return display(exr(geometry_root(name)/"dof"/(case["id"]+".exr")))


def f2_for(unit, entry, frame, inputs, focus, offsets):
    if unit:
        spec=dict(tiers="B",unit=unit,frame=frame,f2_root=training.STAGE1/"development-F2")
        return observed_background(spec,inputs,focus,offsets)
    return background_samples(dict(tiers="B",entry=entry),inputs["K"][0],focus,offsets,(512,512))


def peeled_truth(unit, name, frame):
    if unit:
        folder=unit/"labels"
        geo=ovg.load(folder/f"f{frame:03d}-peeled.npz",("depth_axial_m","center_visible","object_id"))
        colors=ovg.load(folder/f"f{frame:03d}-peeled-color.npz",("rgb_linear",))["rgb_linear"].astype(np.float32)
        return geo,colors,geo["center_visible"]
    from route3_ovg_stage2_train import B_LABELS
    geo=ovg.load(geometry_root(name)/"geometry"/f"f{frame:03d}-background.npz",
                 ("depth_axial_m","center_visible","object_id"))
    color=rgba(B_LABELS/name/f"f{frame:03d}-visible-background.exr")
    keep=color[...,3]>0
    rgb=np.divide(color[...,:3],color[...,3,None],out=np.zeros_like(color[...,:3]),where=keep[...,None])
    return geo,rgb,keep&geo["center_visible"]


def oracle_hidden(peeled, reference, depth, K, focus, stop):
    g,rgb,known=peeled
    ring=depth_band(depth.cpu().numpy(),float(K[0,0]),focus,.05/(2*stop))[0]
    keep=(known & ring & reference["valid"] & (g["object_id"]!=reference["object_id"])
          & (g["depth_axial_m"].astype(np.float32)>reference["gt"].astype(np.float32)+1e-4))
    y,x=np.nonzero(keep)
    parents=dict(uv=np.stack((x+.5,y+.5),-1).astype(np.float32),
                 depth=g["depth_axial_m"][keep].astype(np.float32),
                 recoverable=np.ones(len(y),bool),colors_linear=rgb[keep])
    foreground=np.isin(reference["object_id"],reference["foreground"])
    return place(parents,reference["gt"].astype(np.float32),reference["valid"]&~foreground,depth,K,512)


def k16(unit,name,frame,case):
    if unit:
        spec=b_k16_cases(unit,frame)[case["focus_index"]]
        return ray_fields(spec)
    d=ovg.load(geometry_root(name)/"geometry"/(case["id"]+"-K16.npz"),
               ("depth_axial_m","known","hidden","object_id","aperture_xy"))
    return dict(d,ids=d["object_id"],depth=d["depth_axial_m"])


def leakage(model,item,case,unit,entry,x,g,vis,front,ref,interior,target,destination):
    name,frame=item[1],case["frame"]
    labels=k16(unit,name,frame,case)
    same=(labels["ids"]==ref["object_id"][None]) & labels["known"] & ~labels["hidden"]
    region=interior&ref["valid"]&np.isin(ref["object_id"],ref["foreground"])&same.all(0)
    region &= (np.isfinite(labels["depth"])&(labels["depth"]>0)).all(0)
    if not region.any():
        return dict(scene=name,case=case["id"],domain_rays=0,categories=[])
    focus=case["focus_m"]
    depth=g["depth1"][0]
    K=x["K"][0]
    rays=torch.as_tensor(labels["aperture_xy"],device=K.device,dtype=K.dtype)
    hidden=model.second(item,depth.cpu().numpy(),focus,case["fstop"])
    base=merge_layers(vis,front)
    full=merge_layers(base,hidden)
    fill=f2_for(unit,entry,frame,x,focus,rays)
    mask=training.tensor(region)
    before,after,wqs,tvals,first,last=[],[],[],[],[],[]
    for i,offset in enumerate(rays):
        before.append(raster(base,K,focus,rays[i:i+1],512,512,fill[i:i+1])[0][mask].cpu().numpy())
        after.append(raster(full,K,focus,rays[i:i+1],512,512,fill[i:i+1])[0][mask].cpu().numpy())
        if hidden is not None:
            w,zq,zv,wv,T=trace_predicted(vis,base,full,K,focus,offset,mask)
        else:
            w,zq,zv,T=(np.zeros(int(region.sum()),np.float32) for _ in range(4))
        wqs.append(w)
        tvals.append(T)
        first.append(zq)
        last.append(zv)
    wq,T,zq,zv=map(np.stack,(wqs,tvals,first,last))
    delta,effect,sse_off,sse_on=ray_effect(np.stack(before),np.stack(after),target[region])
    categories=bins(wq,zq,zv,T,effect)
    dest=destination/f"{name}-{case['id']}.npz"
    np.savez_compressed(dest,mask=region,T=np.where(wq>0,T,np.nan),query_weight=wq,query_depth_m=zq,
                        dominant_visible_depth_m=zv,effect=effect,pixel_SSE_change=delta)
    return dict(scene=name,case=case["id"],domain_rays=wq.size,nominal_focus_m=focus,categories=categories,
                net_sse_delta=float(delta.sum()),sse_off=sse_off,sse_on=sse_on)


def render_offsets(cloud,K,focus,offsets,filler,opaque=None):
    chunks=[]
    for i in range(0,len(offsets),8):
        off, fondo=offsets[i:i+8],filler[i:i+8]
        if opaque is None:
            chunk=raster(cloud,K,focus,off,512,512,fondo)
        else:
            chunk=torch.stack([opaque_view(cloud,K,focus,v,*opaque,b)[0] for v,b in zip(off,fondo)])
        chunks.append(chunk.sum(0))
    return sum(chunks)/len(offsets)


def split_pixel_groups(b):
    neighbors=cv2.dilate(b.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
    return {"b1":b,"b0_adjacent":neighbors&~b,"b0_other":~neighbors}


def split_zero_scores(image, target, groups):
    error=np.square(image.astype(np.float64)-target.astype(np.float64)).sum(-1)
    return {key:pooled([dict(sse=float(error[area].sum()),pixels=int(area.sum()))])
            for key,area in groups.items()}


@torch.inference_mode()
def audit_zero_split(run):
    training.configuration()
    training.fixed_inputs()
    checkpoint=run/"2a/step-010000.pt"
    destination=run/"evaluation/2a/step-010000/ema/zero-split-audit.json"
    if destination.exists():
        raise FileExistsError(destination)
    model=training.Model(run)
    model.heads.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=True)["ema"])
    rows=[]
    for split,name,frame,unit,entry,cases in frames():
        if split!="development" or not name.startswith(("development-005-","development-006-")):
            continue
        item=("B",name,ovg.TRAIN_FRAMES.index(frame))
        x,y=training.sample(item)
        g=model.geometry(x)
        first=model.first(item,g)
        ap=model.appearance(g,x,first)
        b=training.split_mask(ap,first)
        groups=split_pixel_groups(b.cpu().numpy())
        K=x["K"][0]
        ref=ovg.b_edge_reference(item,y)
        case=next(c for c in cases if c["focus_index"]==0)
        focus=p2_focus(g["depth1"][0].cpu().numpy(),near_subject(ref["gt"],ref["valid"],K.cpu().numpy(),case["focus_m"]))
        if focus is None:
            raise ValueError(f"Missing near-focus subject for {name} f{frame:03d}")
        center=K.new_zeros((1,2))
        fill=f2_for(unit,entry,frame,x,focus,center)
        vis=training.visible(ap,x,b,"2a")
        front=training.background(ap,x,b,first)
        fixed=training.fixed_visible_surfels(g,x)
        rendered=raster(merge_layers(vis,front),K,focus,center,512,512,fill)[0].cpu().numpy()
        initial=raster(fixed,K,focus,center,512,512,fill)[0].cpu().numpy()
        rgb=x["rgb"][0].permute(1,2,0).cpu().numpy()
        image,baseline,target=display(rendered),display(initial),display(rgb)
        other=groups["b0_other"]
        designated=groups["b1"]
        row=dict(scene=name,frame=frame,case=case["id"],actual_focus_m=focus,b1_fraction=float(designated.mean()),
                 groups=split_zero_scores(image,target,groups),
                 step_zero_groups=split_zero_scores(baseline,target,groups),
                 b0_other_max_abs_display=float(np.abs(image[other]-baseline[other]).max()),
                 b0_other_max_abs_linear=float(np.abs(rendered[other]-initial[other]).max()),
                 b1_foreground_input_mae=float(np.abs(ap["foreground"][0].permute(1,2,0).cpu().numpy()[designated]
                                                     -rgb[designated]).mean()) if designated.any() else None,
                 b1_opacity_p10_p50_p90=np.quantile(ap["opacity"][0][b].cpu().numpy(),[.1,.5,.9]).tolist()
                 if designated.any() else [None,None,None])
        rows.append(row)
        print("STAGE2_ZERO_SPLIT",name,frame,row["b0_other_max_abs_display"],row["b0_other_max_abs_linear"],flush=True)
    if len(rows)!=6:
        raise ValueError(f"Expected six audit frames, found {len(rows)}")
    aggregate={key:pooled([row["groups"][key] for row in rows]) for key in rows[0]["groups"]}
    maximum_display=max(row["b0_other_max_abs_display"] for row in rows)
    maximum_linear=max(row["b0_other_max_abs_linear"] for row in rows)
    write_json(destination,dict(scope="2a 10k EMA; center ray, first-pass background and fixed F2; second-pass Query disabled",
                                rows=rows,aggregate=aggregate,limit=1e-4,max_abs_display=maximum_display,
                                max_abs_linear=maximum_linear,passed=max(maximum_display,maximum_linear)<1e-4,
                                completed_at=training.now()))


@torch.inference_mode()
def evaluate_frames(model, phase, raw, out, *, endpoint=False, only_leaks=False, probe=False, audit_split=False):
    records, spatial, leaks, unscored=[],[],[],[]
    lpips=None if only_leaks else lpips_model()
    ray_dir=out/"leakage"
    ray_dir.mkdir(parents=True,exist_ok=True)
    seen=set()
    for split,name,frame,unit,entry,cases in frames():
        if probe and (split in seen or frame!=30):
            continue
        seen.add(split)
        reserve(out,GIB)
        item=("B",name,ovg.TRAIN_FRAMES.index(frame))
        x,y=training.sample(item)
        g=model.geometry(x)
        first=model.first(item,g)
        ap=model.appearance(g,x,first)
        depth,K=g["depth1"][0],x["K"][0]
        mask=training.split_mask(ap,first)
        ref=ovg.b_edge_reference(item,y)
        edge,distance_masks,interior=bands(ref,y)
        valid=np.ones_like(edge)
        pinhole=display(x["rgb"][0].permute(1,2,0).cpu().numpy())
        target=training.appearance_labels(item)
        oracle_mask=training.candidate_mask(ap)&training.tensor(target["comp_valid"])
        front_oracle=dict(color=target["background"],depth=g["depth2"][0].cpu().numpy(),
                          q=np.ones(mask.shape,np.float32))
        layers={}
        for mode,b,background in (("none",torch.zeros_like(mask),first),
                                    ("oracle",oracle_mask,front_oracle),("retrieved",mask,first)):
            if only_leaks and mode!="retrieved":
                continue
            layers[mode]=(training.visible(ap,x,b,phase),training.background(ap,x,b,background))
        center=K.new_zeros((1,2))
        raster_zero={}
        for mode,(vis,_) in layers.items():
            z,a=raster(vis,K,1.4,center,512,512,return_alpha=True)
            raster_zero[mode]=z[0],a[0]
        if not only_leaks:
            peeled=peeled_truth(unit,name,frame)
            b=mask.cpu().numpy()
            zero_groups=split_pixel_groups(b) if audit_split else None
            low_quality=b & (first["q"]>0) & (first["q"]<.1)
            T=1-raster_zero["retrieved"][1].cpu().numpy()
            alpha=ap["alpha"][0].cpu().numpy()
            bins_and_full=dict(distance_masks,all=valid)
            categories={}
            for k,area in bins_and_full.items():
                inside=interior & area
                categories[k]=dict(pixels=int(area.sum()),b1=int((b&area).sum()),transmittance=T[b&area],
                                   gt_inside=int(inside.sum()),
                                   false_semitransparent=int((inside&(alpha<1-training.EPSILON)).sum()))
            spatial.append(dict(split=split,scene=name,frame=frame,bins=categories,
                                alpha_hist=np.histogram(alpha[interior],bins=np.linspace(0,1,21))[0],
                                alpha_middle=int((interior&(alpha>=.45)&(alpha<=.65)).sum()),
                                alpha_count=int(interior.sum())))
        for case in cases:
            target_rgb=optical_target(unit,name,case)
            emphasis=case["focus_index"]==0
            if emphasis and not raw and (endpoint or only_leaks):
                vis,front=layers["retrieved"]
                leaks.append(dict(split=split,**leakage(model,item,case,unit,entry,x,g,vis,front,ref,
                                                        interior,target_rgb,ray_dir)))
            if only_leaks:
                continue
            focus=p2_focus(depth.cpu().numpy(),near_subject(ref["gt"],ref["valid"],K.cpu().numpy(),case["focus_m"]))
            common=dict(split=split,scene=name,frame=frame,case=case["id"],focus_index=case["focus_index"])
            if focus is None:
                unscored.append(dict(common,reason="no_valid_nominal_subject"))
                continue
            spec_masks=new_regions(unit,case,ref["gt"],ref["valid"],K.cpu().numpy()) if unit else (
                legacy_masks(geometry_root(name),case))
            regions={"boundary_2px":edge}
            if emphasis:
                regions.update(near_in_focus=spec_masks["in_focus"],near_recoverable=spec_masks["recoverable"])
            else:
                regions["far_recoverable"]=spec_masks["recoverable"]
            offsets=aperture_disk(64,.05/(2*case["fstop"]),device=K.device,dtype=K.dtype)
            fill=f2_for(unit,entry,frame,x,focus,offsets)
            zfill=f2_for(unit,entry,frame,x,focus,center)
            hidden=model.second(item,depth.cpu().numpy(),focus,case["fstop"])
            gt, gt_audit=oracle_hidden(peeled,ref,depth,K,focus,case["fstop"])
            for mode,(vis,front) in layers.items():
                base=merge_layers(vis,front)
                rendered=merge_layers(base,hidden)
                on=render_offsets(rendered,K,focus,offsets,fill)
                zv=raster_zero[mode][0]
                z=raster(rendered,K,focus,center,512,512,zfill)[0]
                score_sets=dict(all=valid,interior=interior,boundary_2px=edge)
                row=dict(common,mode=mode,actual_focus_m=focus,
                         optical=score(display(on.cpu().numpy()),target_rgb,regions,lpips),
                         zero_visible={k:region_errors(zv,pinhole,v) for k,v in score_sets.items()},
                         zero_combined={k:region_errors(z,pinhole,v) for k,v in score_sets.items()})
                if mode=="retrieved":
                    if zero_groups is not None:
                        first_pass_zero=raster(base,K,focus,center,512,512,zfill)[0].cpu().numpy()
                        row["zero_split_groups"]=split_zero_scores(display(first_pass_zero),pinhole,zero_groups)
                    off=render_offsets(base,K,focus,offsets,fill)
                    row["query_off"]={k:region_errors(off,target_rgb,v) for k,v in regions.items()}
                    oracle=render_offsets(merge_layers(base,gt),K,focus,offsets,fill)
                    row["gt_hidden"]={k:region_errors(oracle,target_rgb,v) for k,v in regions.items()}
                    row["gt_placement"]=gt_audit
                    row["distance_zero"]={k:dict(visible=region_errors(zv,pinhole,v),
                                               combined=region_errors(z,pinhole,v)) for k,v in bins_and_full.items()}
                    row["low_q_zero"]={k:region_errors(image,pinhole,low_quality)
                                       for k,image in (("zero_visible",zv),("zero_combined",z))}
                    if endpoint and not raw:
                        b0=(~mask)&torch.isfinite(depth)&(depth>0)&torch.isfinite(x["rgb"][0]).all(0)
                        flags=subpixel_flags(torch.isfinite(depth)&(depth>0)&torch.isfinite(x["rgb"][0]).all(0),b0,5)
                        args=(depth,b0,flags,len(base.means))
                        row["opaque_b0"]=dict(zero_visible=region_errors(
                            opaque_view(vis,K,focus,center[0],*args)[0],pinhole,valid),
                            zero_combined=region_errors(opaque_view(rendered,K,focus,center[0],*args,zfill[0])[0],
                                                        pinhole,valid))
                        if emphasis:
                            for variant,layer in (("off",base),("on",rendered)):
                                image=render_offsets(layer,K,focus,offsets,fill,args)
                                row["opaque_b0"][variant]=region_errors(image,target_rgb,regions["near_in_focus"])
                records.append(row)
        print("STAGE2_FRAME",phase,"raw" if raw else "EMA",split,name,frame,flush=True)
    return records,spatial,unscored,leaks


def aggregate(values):
    score=pooled(values)
    available=[r for r in values if r["pixels"] and r.get("patch_lpips") is not None]
    score["patch_lpips"]=sum(r["pixels"]*r["patch_lpips"] for r in available)/sum(
        r["pixels"] for r in available) if available else None
    return score


def domains(rows):
    grouped=defaultdict(list)
    for row in rows:
        grouped[row["scene"]].append(row)
    return dict(all=rows,**grouped)


def subtract(before,after):
    a,b=before["psnr_db"],after["psnr_db"]
    if before["pixels"]!=after["pixels"]:
        raise ValueError("Evaluation changed the pixel support of a pair")
    return b-a if a is not None and b is not None else None


def pair(left,right):
    a,b=domains(left),domains(right)
    if a.keys()!=b.keys():
        raise ValueError("Evaluation changed the scenes of a pair")
    grouped={}
    for label in a:
        lo,hi=aggregate(a[label]),aggregate(b[label])
        grouped[label]=dict(before=lo,after=hi,delta_db=subtract(lo,hi),
                            delta_patch_lpips=hi["patch_lpips"]-lo["patch_lpips"]
                            if None not in (hi["patch_lpips"],lo["patch_lpips"]) else None)
    rows=[v["delta_db"] for k,v in grouped.items() if k!="all" and v["delta_db"] is not None]
    return dict(pooled=grouped.pop("all"),per_scene=grouped,paired=distribution(rows),
                absolute_scene_sd={p:distribution([v[p]["psnr_db"] for v in grouped.values()
                                                    if v[p]["psnr_db"] is not None])["sd_db"] for p in ("before","after")})


def data_rows(rows, section, metric, sub=None):
    result=[]
    for row in rows:
        values=row.get(section,{})
        if metric in values:
            score=values[metric] if sub is None else values[metric][sub]
            result.append(dict(scene=row["scene"],**score))
    return result


def guards(rows,metric):
    values=[r[metric]["all"] for r in rows if r[metric]["all"]["pixels"]]
    floor=GUARDS[metric]
    passed=sum(aggregate([r])["psnr_db"]>=floor for r in values)
    return dict(floor_db=floor,passed_cases=passed,scored_cases=len(values),all_passed=bool(values) and passed==len(values),
                **{k:aggregate([r[metric][k] for r in rows]) for k in ("all","interior","boundary_2px")})


def upper_bound(retrieved, metric):
    off=data_rows(retrieved,"query_off",metric)
    gt=pair(off,data_rows(retrieved,"gt_hidden",metric))
    query=pair(off,data_rows(retrieved,"optical",metric))
    names=gt["per_scene"]
    ratios={}
    for name in names:
        value=gt["per_scene"][name]["delta_db"]
        observed=query["per_scene"][name]["delta_db"]
        ratios[name]=observed/value if value is not None and value>.1 else None
    valid=[v for v in ratios.values() if v is not None]
    all_gt,all_q=(r["pooled"]["delta_db"] for r in (gt,query))
    return dict(gt_minus_f2=gt,query_minus_f2=query,per_scene_ratio=ratios,
                ratio_median=float(np.median(valid)) if valid else None,ratio_scenes=len(valid),
                pooled_ratio=all_q/all_gt if all_gt is not None and all_gt>.1 else None)


def spatial_summary(rows,frames):
    values={}
    all_b1=sum(r["bins"]["all"]["b1"] for r in frames)
    for name in (*LEVELS,"all"):
        bins=[r["bins"][name] for r in frames]
        sums=Counter()
        for b in bins:
            sums.update({k:v for k,v in b.items() if k!="transmittance"})
        samples=np.concatenate([b["transmittance"] for b in bins])
        quantiles=np.quantile(samples,[.5,.95]).tolist() if len(samples) else [None,None]
        values[name]=dict(**sums,b1_fraction=sums["b1"]/max(1,sums["pixels"]),
                          b1_share=sums["b1"]/max(1,all_b1),transmittance_p50_p95=quantiles,
                          gt_inside_false_semitransparent_rate=sums["false_semitransparent"]/max(1,sums["gt_inside"]),
                          visible=aggregate([r["distance_zero"][name]["visible"] for r in rows]),
                          combined=aggregate([r["distance_zero"][name]["combined"] for r in rows]))
    n=sum(r["alpha_count"] for r in frames)
    inside=sum(r["alpha_middle"] for r in frames)
    hist=np.sum([r["alpha_hist"] for r in frames],axis=0).tolist()
    return dict(bands=values,gt_interior_alpha=dict(histogram=hist,edges=np.linspace(0,1,21).tolist(),
                                                    pixels=n,fraction_045_065=inside/max(1,n)))


def describe(rows,spatial,unscored,leaks):
    result={}
    for split in ("development","legacy"):
        chosen=[r for r in rows if r["split"]==split]
        retrieved=[r for r in chosen if r["mode"]=="retrieved"]
        modes={m:[r for r in chosen if r["mode"]==m] for m in MODES}
        pair_one={k:pair(data_rows(retrieved,"query_off",k),data_rows(retrieved,"optical",k))
                  for k in SCORES[:3]}
        pair_two={}
        for p,q in combinations(MODES,2):
            data={k:pair(data_rows(modes[p],"optical",k),data_rows(modes[q],"optical",k)) for k in SCORES}
            for metric in GUARDS:
                for region in ("all","boundary_2px","interior"):
                    data[f"{metric}_{region}"]=pair(data_rows(modes[p],metric,region),data_rows(modes[q],metric,region))
            pair_two[q+"_minus_"+p]=data
        zero={k:guards(retrieved,k) for k in GUARDS}
        result[split]=dict(query_pair=pair_one,split_pair=pair_two,zero_floors=zero,
                           b1_diagnostics=spatial_summary(retrieved,[r for r in spatial if r["split"]==split]),
                           unscored=[r for r in unscored if r["split"]==split],
                           upper_reference={k:upper_bound(retrieved,k) for k in ("near_in_focus","far_recoverable")},
                           leakage=[r for r in leaks if r["split"]==split])
        result[split]["b1_diagnostics"]["low_q_zero"]={
            key:{name:aggregate(rows) for name,rows in domains(data_rows(retrieved,"low_q_zero",key)).items()}
            for key in GUARDS}
        if retrieved and "zero_split_groups" in retrieved[0]:
            result[split]["zero_split_groups"]={
                key:{name:pooled(values) for name,values in domains(data_rows(retrieved,"zero_split_groups",key)).items()}
                for key in retrieved[0]["zero_split_groups"]}
        opaque=[r for r in retrieved if "opaque_b0" in r]
        if opaque:
            result[split]["opaque_b0"]=dict(
                near_query_pair=pair(data_rows(opaque,"opaque_b0","off"),data_rows(opaque,"opaque_b0","on")),
                zeros={metric:aggregate([r["opaque_b0"][metric] for r in opaque]) for metric in GUARDS})
    return result


def leakage_summary(folder, rows):
    result={}
    for split in ("development","legacy"):
        data=[r for r in rows if r["split"]==split]
        strata={k:dict(rays=0,positive_sse=0.,T=[],q=[]) for k in RAY_CLASSES}
        for row in data:
            if not row["domain_rays"]:
                continue
            with np.load(folder/f"{row['scene']}-{row['case']}.npz") as f:
                w,zq,zv,T,effect=(f[k] for k in ("query_weight","query_depth_m","dominant_visible_depth_m","T","effect"))
            harmful=(w>=.001)&(effect>0)
            order=harmful&(zv>0)&(zq<zv)
            ordinary=harmful&~order&np.isfinite(T)
            masks=(order,ordinary&(T<.02),ordinary&(T>=.02)&(T<.1),ordinary&(T>=.1),harmful&~order&~ordinary)
            for label,m in zip(RAY_CLASSES,masks):
                cell=strata[label]
                cell["rays"]+=int(m.sum())
                cell["positive_sse"]+=float(effect[m].sum(dtype=np.float64))
                cell["T"].append(T[m&np.isfinite(T)])
                cell["q"].append(w[m])
        for cell in strata.values():
            for field in ("T","q"):
                arrays=cell.pop(field)
                x=np.concatenate(arrays) if arrays else []
                cell[field+"_p50_p95"]=np.quantile(x,[.5,.95]).tolist() if len(x) else [None,None]
        result[split]=dict(domain_rays=sum(r["domain_rays"] for r in data),categories=strata,
                           net_sse_delta=sum(r.get("net_sse_delta",0.) for r in data),
                           sse_off=sum(r.get("sse_off",0.) for r in data),sse_on=sum(r.get("sse_on",0.) for r in data))
    return result


@torch.no_grad()
def fixed_losses(model,phase):
    data=training.Dataset()
    source={"B":{"train":data.train["B"],"dev":[("B",unit.name,i)
             for unit in library_unit_paths("dev") for i in range(len(ovg.TRAIN_FRAMES))]},
            "C":data.c,"A":{p:list(range(len(data.a[p].foreground))) for p in ("train","dev")}}
    result={}
    for tier in ("A","B","C"):
        result[tier]={}
        for role in ("train","dev"):
            rng=np.random.default_rng([training.SEED,ord(tier)])
            gen=torch.Generator(device="cuda").manual_seed(training.SEED+ord(tier))
            items=source[tier][role]
            picked=rng.choice(len(items),min(len(items),16),replace=False)
            scores=[]
            for i in picked:
                if tier=="A":
                    terms,support=training.a_objective(model,data.a[role],training.SEED+int(i),phase)
                else:
                    terms,support=training.labeled_objective(model,items[i],rng,gen,phase,common_audit=True)
                scores.append(dict(scene=support["scene"],total=float(training.weighted(terms)),
                                   terms={k:float(v) for k,v in terms.items()}))
            result[tier][role]=dict(samples=scores,mean=float(np.mean([r["total"] for r in scores])))
        result[tier]["dev_minus_train"]=result[tier]["dev"]["mean"]-result[tier]["train"]["mean"]
    return result


def trained_report(ck, report, losses):
    return dict(report=report,fixed_losses=losses,
                scope="P2 f/1.4 (the Phase 1 continuity protocol), 15 development scenes and 3 legacy scenes separately",
                historical_legacy_2b=dict(near_query_delta_db=-2.01,zero_visible="0/72",zero_combined="0/72",
                                           zero_scope="historical full P1/P2, two-aperture protocol"),
                training_seconds=float(ck["seconds"]),batch_counts={k:int(v) for k,v in ck["counts"].items()},
                loss_batches={k:float(v) for k,v in ck["uses"].items()},
                loss_batches_scope="Sum of unweighted batch losses per term",
                loss_scope="16 seeded units per split and tier; loss terms common to training and development (without optional optical references or K16)")


def main(run,phase,step,weights,probe=False):
    training.configuration()
    training.fixed_inputs()
    run.mkdir(parents=True,exist_ok=True)
    out=run/("evaluation-probe" if probe else "evaluation")/phase/f"step-{step:06d}"/weights
    out.mkdir(parents=True,exist_ok=True)
    model=training.Model(run)
    started=time.monotonic()
    raw=weights=="model"
    if step:
        path=run/phase/f"step-{step:06d}.pt"
        ck=torch.load(path,map_location="cpu",weights_only=True)
        model.heads.load_state_dict(ck[weights])
    rows,spatial,unscored,leaks=evaluate_frames(model,phase,raw,out,endpoint=phase=="2b" and step==30000,
                                                 only_leaks=not step and not probe,probe=probe,
                                                 audit_split=phase=="2b" and step==10000)
    if step:
        result=trained_report(ck,describe(rows,spatial,unscored,leaks),fixed_losses(model,phase))
    elif probe:
        result=dict(report=describe(rows,spatial,unscored,leaks),scope="one frame per evaluation split")
    else:
        result=dict(leakage=leaks,scope="step-zero frozen Stage 1 geometry; K16 nominal focus, GT interior")
    if leaks:
        result["leakage_aggregate"]=leakage_summary(out/"leakage",leaks)
    manifest=load(FORMAL_DATA/"manifest.json")
    optics=dict(admitted=manifest["counts"]["train_optical"],f2_complete=sum(
        all((FORMAL_DATA/"derived-f2"/row["scene_id"]/f"f{i:03d}"/"manifest.json").is_file() for i in FRAMES)
        for row in manifest["optical_units"]))
    result.update(evaluation_seconds=time.monotonic()-started,stage=phase,step=step,weights=weights,
                  optical=optics,completed_at=training.now())
    if rows:
        write_json(out/"cases.json",dict(rows=rows,unscored=unscored))
    write_json(out/"report.json",result)
    print("STAGE2_EVAL_DONE",out,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=Path,default=training.RUN)
    p.add_argument("--phase",choices=("2a","2b"),required=True)
    p.add_argument("--step",type=int,required=True)
    p.add_argument("--weights",choices=("ema","model"),default="ema")
    p.add_argument("--probe",action="store_true")
    p.add_argument("--audit-zero-split",action="store_true")
    a=p.parse_args()
    if a.audit_zero_split:
        if (a.phase,a.step,a.weights)!=("2a",10000,"ema") or a.probe:
            p.error("The zero-split audit uses 2a 10k EMA")
        audit_zero_split(a.run.resolve())
    else:
        main(a.run.resolve(),a.phase,a.step,a.weights,a.probe)
