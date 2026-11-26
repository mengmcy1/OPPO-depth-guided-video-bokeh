"""Isolated hidden-colour inference with registered scenes and full sensor scoring."""
import argparse
from dataclasses import replace
import os
import time

import cv2
import numpy as np
import torch

import route3_v2data_module2_infer as ref
from route3_ldr_u1r import model as load_lpips
from route3_ovg_footprint import query_net
from route3_v2data_E0_evaluate import EInputs
from route3_v2data_E0_rerender import make_visible, load_tear
from route3_v2data_E0_tears import features, load_model
from route3_v2data_E0_train import gpu_gate, predict
from route3_v2data_E_labels import save_json
from route3_v2data_hidden_color_query import geometry_cloud, color_variants, proxy_generator
from route3_v2data_hidden_color_sources import OUT, CUTS, SceneEvidence, bin_counts, read, read_witnesses
from route3_v2data_hidden_color_zero import FAILURE_CASE, ZeroAperture, verify_case
from route3_v2data_stage3b import QUERY, FrameQueries, optical_band, pack_frame

ROOT=ref.ROOT
ARCHIVE=ROOT/"outputs/route3-v2data-E0-rerender-2026-11-26-01"
OLD_ARM={"V0_pred":"V0_pred","V0_GT":"V0_GT","E2k":"E_learned"}
MAIN_CASES={"dev001","dev003","dev004","dev005","dev006","dev008","dev010","dev011","dev013","122104"}
E0=ROOT/"outputs/route3-v2data-E0-2026-11-26-01"
PLOT_LIMIT=.99
LIMIT=8


def state(phase,**items):
    save_json(OUT/"dispatch.json",dict(phase=phase,at=time.time(),pid=os.getpid(),**items))


def render_chunked(visible,hidden,K,focus,offsets,fill):
    """Exact K64 arithmetic from the E0 optical replay."""
    return ref.display(ref.light(visible,hidden,K,focus,offsets,fill))


def patch_layout(mask):
    return [(y,x,int(mask[y:y+64,x:x+64].sum())) for y in range(0,512,64)
            for x in range(0,512,64) if mask[y:y+64,x:x+64].any()]


def lpips_between(model, a, b, layout):
    if not layout:
        return None
    values=[]
    for n in range(0,len(layout),16):
        def patches(image):
            arrays=[image[y:y+64,x:x+64] for y,x,_ in layout[n:n+16]]
            return torch.from_numpy(np.stack(arrays)).permute(0,3,1,2).cuda()
        values.extend(model(2*patches(a)-1,2*patches(b)-1).reshape(-1).cpu().tolist())
    w=np.array([k for _,_,k in layout],np.float64)
    return float(np.dot(values,w)/w.sum())


def render_bins(visible, cloud, purity, K, focus, offsets):
    """Keep zero-coverage separate so the positive-support cut is exact."""
    if cloud is None:
        return np.zeros((512,512,6),np.float32)
    positive=2+np.searchsorted(CUTS,purity,side="right")
    groups=np.where(np.isfinite(purity),np.where(purity>0,positive,1),0)
    src=torch.as_tensor(groups,device=K.device,dtype=torch.long)
    tags=torch.nn.functional.one_hot(src,6).float()
    combined=ref.merge_layers(replace(visible,colors=visible.colors.new_zeros((len(visible.means),6))),
                              replace(cloud,colors=tags))
    image=sum(ref.raster(combined,K,focus,part,512,512).sum(0) for part in offsets.split(4))
    return (image/len(offsets)).cpu().numpy()


def grades(weights):
    merged=np.concatenate((weights[...,:1],weights[...,1:3].sum(-1,keepdims=True),weights[...,3:]),-1)
    return np.where(merged.sum(-1)>1e-6,merged.argmax(-1)-1,-1).astype(np.int8)


def stratify(rgb,truth,area,weights):
    group=grades(weights)
    splits={f"bin{i}":ref.region_sse(rgb,truth,area&(group==i)) for i in range(4)}
    splits["unknown/no_hit"]=ref.region_sse(rgb,truth,area&(group==-1))
    gate={}
    for index,threshold in enumerate((0.,*CUTS.tolist())):
        keep=weights[...,index+2:].sum(-1)>1e-6
        gate[f"{threshold:.2f}"]=dict(
            recoverable=ref.region_sse(rgb,truth,area&keep),
            unrecoverable=ref.region_sse(rgb,truth,area&~keep))
    return dict(bins=splits,thresholds=gate)


def placed(parents,unit,name,frame,native,depth,maximum,K,*,color=None,support=None):
    if unit:
        front=native["gt"]
        bg=native["valid"]&~np.isin(native["object_id"],native["foreground"])
    else:
        _,front,bg=ref.b_oracle(name,frame)
    values=parents if color is None else dict(parents,colors_linear=color,recoverable=support)
    return ref.place(values,front,bg,depth,K,512,order_depth=maximum)[0]


def parent_labels(geometry,parents,unit,name,frame,native,K):
    _,depth,maximum=geometry
    n=len(parents["depth"])
    storage=np.broadcast_to((np.arange(1,n+1,dtype=np.float32)/(n+1))[:,None],(n,3)).copy()
    h=placed(parents,unit,name,frame,native,depth,maximum,K,color=storage,support=parents["recoverable"])
    if h is None:
        return None,np.empty(0,int)
    origin=(h.colors[:,0].cpu().numpy()*(n+1)).round().astype(int)-1
    if ((origin<0)|(origin>=n)).any():
        raise ValueError("Parent identity did not survive the optical footprint expansion")
    return h,origin


def stat_record(rgb,truth,area,weights,lpips,layout,baseline):
    r=dict(score=ref.region_sse(rgb,truth,area))
    if layout:
        r.update(patch_lpips=lpips_between(lpips,rgb,truth,layout),
                 direct_lpips_to_baseline=lpips_between(lpips,rgb,baseline,layout))
    if weights is not None:
        r["stratified"]=stratify(rgb,truth,area,weights)
    return r


def target_metadata(name,frame,case):
    path=ARCHIVE/"fixed-base"/"frames"/name/f"f{frame:03d}.json"
    return next(r for r in read(path)["rows"]["G"] if r["case"]==case["id"])


def expected_score(raw,arm,region):
    return ref.score_psnr(raw["methods"][OLD_ARM[arm]]["optical"][region])


def color_storage(v):
    return v.astype(np.float16)


def substitute_alternatives(geometry,unit,name,frame,native,K,parents,variants):
    _,depth,maximum=geometry
    for kind,(color,keep) in variants.items():
        if kind=="S0" or (kind.startswith("S1") and kind!="S1@0.00"):
            continue
        cloud=placed(parents,unit,name,frame,native,depth,maximum,K,color=color,support=keep) if keep.any() else None
        yield kind,cloud


def render_substitutes(geometry,arm,tag,focus,unit,name,frame,entry,case,native,x,K,
                       offsets,center,pinhole,truth,area,region,lpips,layout,parents,variants,best,pictures,raw):
    visible,depth,maximum=geometry
    fill=ref.f2_for(unit,entry,frame,x,focus,offsets)
    zero=ZeroAperture(visible,K,focus,center,ref.f2_for(unit,entry,frame,x,focus,center),pinhole,
                      (name,case["id"],tag,arm))
    f2=render_chunked(visible,None,K,focus,offsets,fill)
    cloud=placed(parents,unit,name,frame,native,depth,maximum,K)
    s0=render_chunked(visible,cloud,K,focus,offsets,fill)
    observed=ref.region_sse(s0,truth,area)["psnr"]
    if tag=="G":
        expected=expected_score(raw,arm,region)
        if (expected is None)!=(observed is None) or (expected is not None and abs(expected-observed)>.01):
            raise ValueError(f"S0 G differs for {name}/{case['id']}/{arm}: {observed} vs {expected}")
    weights=None
    if case["focus_index"]==1:
        marker,parents_map=parent_labels(geometry,parents,unit,name,frame,native,K)
        weights=render_bins(visible,marker,best[parents_map],K,focus,offsets)
    score={"F2":dict(stat_record(f2,truth,area,weights,lpips,layout,s0),**zero.arm("F2")),
           "S0":dict(stat_record(s0,truth,area,weights,lpips,layout,s0),**zero.arm("S0",cloud))}
    if pictures is not None:
        pictures[f"{arm}/S0"]=color_storage(s0)
        pictures[f"{arm}/F2"]=color_storage(f2)
        pictures[f"{arm}/purity"]=grades(weights)
        pictures[f"{arm}/unrecoverable"]=weights[...,-1]<=1e-6
    for kind,altered in substitute_alternatives(geometry,unit,name,frame,native,K,parents,variants):
        rgb=f2 if altered is None else render_chunked(visible,altered,K,focus,offsets,fill)
        score[kind]=dict(stat_record(rgb,truth,area,weights,lpips,layout,s0),**zero.arm(kind,altered))
        if pictures is not None and kind in ("S1@0.00","S2@0.99","S3"):
            pictures[f"{arm}/{kind}"]=color_storage(rgb)
    return score


def render_query(visible,cloud,details,args,case,tag,focus,offsets,pictures):
    unit,name,frame,entry,x,K,truth,area,lpips,layout,evidence,views,proxy,center,pinhole=args
    fill=ref.f2_for(unit,entry,frame,x,focus,offsets)
    key=(name,case["id"],tag)
    zero=ZeroAperture(visible,K,focus,center,ref.f2_for(unit,entry,frame,x,focus,center),pinhole,key)
    trace_at="QGT@0.50" if key==FAILURE_CASE else None
    off=render_chunked(visible,None,K,focus,offsets,fill)
    change=None
    if cloud is None:
        variants={"Query":None}
        for typ in ("QGT","QPred"):
            variants.update({f"{typ}@{cut:.2f}":None for cut in (0.,*CUTS.tolist())})
        src,best=dict(retained_heads=0),{"QGT":np.empty(0),"QPred":np.empty(0)}
    else:
        variants,src,best,change=color_variants(cloud,details,evidence,views,frame,proxy,trace_at=trace_at)
    on=render_chunked(visible,cloud,K,focus,offsets,fill)
    weights=render_bins(visible,cloud,best["QGT"],K,focus,offsets) if case["focus_index"]==1 else None
    metrics={"F2":dict(stat_record(off,truth,area,weights,lpips,layout,on),**zero.arm("F2"))}
    for arm,splats in variants.items():
        image=on if arm=="Query" else (off if splats is None else render_chunked(visible,splats,K,focus,offsets,fill))
        metrics[arm]=dict(stat_record(image,truth,area,weights,lpips,layout,on),**zero.arm(arm,splats))
        if pictures is not None and (arm=="Query" or arm.endswith("@0.99")):
            pictures[arm]=color_storage(image)
    if pictures is not None:
        for arm,weight in (("Query",weights),("QGT@0.99",weights),
                ("QPred@0.99",render_bins(visible,cloud,best["QPred"],K,focus,offsets))):
            pictures[f"{arm}/purity"]=grades(weight)
            pictures[f"{arm}/unrecoverable"]=weight[...,-1]<=1e-6
    result=dict(methods=metrics,visible_zero=metrics["Query"]["visible_zero"],source=src)
    if change is not None:
        removed,recolored=(change.pop(k) for k in ("removed","recolored"))
        original=metrics["Query"]["zero"]["psnr"]
        variant=metrics[trace_at]["zero"]["psnr"]
        result["zero_change"]=dict(change,original_query_db=original,variant_db=variant,
            variant_minus_original_db=variant-original,original_hidden_points=len(removed),
            removed_hidden_points=int(removed.sum()),recolored_hidden_points=int(recolored.sum()),
            pinhole_rays=zero.changed_rays(cloud,removed,recolored))
    return result


def selected_view_context(unit,name,views,memory,frame,depth,K,cases,focus_tags):
    stores={}
    for tag,focuses in focus_tags.items():
        bands=[optical_band(depth,K,f,q["fstop"]) for q,f in zip(cases,focuses) if f is not None]
        if bands:
            packed=pack_frame(views,frame,depth,bands,memory)
            stores[tag]=FrameQueries(views,packed)
    return stores


def frame_file(name,frame,smoke):
    dest=(OUT/"smoke" if smoke else OUT)/"frames"/name/f"f{frame:03d}.json"
    return dest


def zero_file(name,frame):
    return OUT/"zeros-only"/name/f"f{frame:03d}.json"


def supplement_zero(previous, surface, unit, name, frame, entry, native, x, K, center, pinhole, cases):
    """Backfill the new fields, keeping earlier K64 scores and Query pinhole scores."""
    rows=[]
    for old in previous["rows"]:
        if old["status"]!="evaluated":
            continue
        case=next(c for c in cases if c["id"]==old["case"])
        parents,variants,_,_=read_witnesses(name,case)
        scored={}
        cached_visible=[]
        for tag in ("G","P"):
            scored[tag]={}
            for arm,geometry in surface.items():
                focus=old["focus_error"][arm]["GT_focus_m" if tag=="G" else "focus_m"]
                visible,depth,maximum=geometry
                zfill=ref.f2_for(unit,entry,frame,x,focus,center)
                zero=ZeroAperture(visible,K,focus,center,zfill,pinhole,(name,case["id"],tag,arm))
                s0=placed(parents,unit,name,frame,native,depth,maximum,K)
                candidate={"F2":zero.arm("F2"),"S0":zero.arm("S0",s0)}
                for method,hidden in substitute_alternatives(geometry,unit,name,frame,native,K,parents,variants):
                    candidate[method]=zero.arm(method,hidden)
                scored[tag][arm]=candidate
                if arm=="V0_pred":
                    original=old["render"][tag]["Query"]
                    cached_visible.append(original["visible_zero"]["psnr"])
                    scored[tag]["Query"]=dict(methods={method:zero.arm(method,saved_combined=m["zero"])
                                                        for method,m in original["methods"].items()})
        checks=verify_case(dict(old,render=scored),cached_visible=cached_visible)
        rows.append(dict(case=case["id"],render=scored,zero_checks=checks))
    return dict(rows=rows)


def comparison_map(subject, focus, target, predicted):
    shift=abs(1/focus-1/target)
    valid=subject&np.isfinite(predicted)&(predicted>0)
    values=np.abs(1/predicted[valid].astype(np.float64)-1/target)
    return dict(focus_m=focus,GT_focus_m=target,subject_pixels=int(valid.sum()),delta_rho=shift,
                CoC_at_K25_4=25.4*shift,
                subject_mean_abs_rho=float(values.mean()) if len(values) else None,
                subject_p50_abs_rho=float(np.median(values)) if len(values) else None)


def raw_s3_source_coverage(best,old,parents,variants):
    known=parents["recoverable"]
    return dict(parents=int(known.sum()),best=bin_counts(best,known),S0=bin_counts(old,known),
                any_at_threshold={f"{cut:.2f}":int((known&np.isfinite(best)&(best>0)&(best>=cut)).sum())
                                  for cut in (0.,*CUTS.tolist())},
                S3_qualified=int(variants["S3"][1].sum()) if "S3" in variants else None)


@torch.inference_mode()
def evaluate(smoke=False):
    ref.gpu_setup(dict(training=dict(max_process_gpu_gib=LIMIT)))
    gpu_gate(LIMIT)
    geometry=ref.load_geometry()
    qnet=query_net(QUERY).eval()
    e_net,_=load_model("E-H2",2000)
    inputs=EInputs("E-H2")
    tear=load_tear(E0/"E-H2/tear-002000/selected.pt")
    os.environ["TORCH_HOME"]=str(OUT/"lpips-weights")
    lpips=load_lpips().cuda().eval()
    from route3_v2data_hidden_color_panels import draw_panel
    legacy={(p["name"],p["frame"]):p for p in ref.gallery.planned_frames() if p["unit"] is None}
    last=None
    totals=0
    for cohort,name,frame,unit,entry,cases in ref.frames():
        slug=ref.title(name)
        if smoke and (frame!=30 or slug not in ("dev010","dev005","122104")):
            continue
        dest=frame_file(name,frame,smoke)
        saved=dest if dest.is_file() else frame_file(name,frame,True)
        previous=read(saved) if saved.is_file() else None
        sidecar=zero_file(name,frame)
        needs_zero=previous is not None and any(r["status"]=="evaluated" and "zero_checks" not in r
                                               for r in previous["rows"])
        if previous is not None and (not needs_zero or sidecar.is_file()):
            totals+=1
            continue
        gpu_gate(LIMIT)
        sample=("B",name,ref.ovg.TRAIN_FRAMES.index(frame))
        x,y=ref.sample(sample)
        predicted=geometry(x,protect_residual=True,render_heads=False,return_features=True)
        K=x["K"][0]
        native=ref.ovg.b_edge_reference(sample,y)
        center=K.new_zeros((1,2))
        known=torch.as_tensor(native["valid"],device=K.device)
        gt0=K.new_tensor(native["gt"])
        z=predicted["depth1"][0]
        gt=torch.where(known&torch.isfinite(gt0)&(gt0>0),gt0,z)
        decoded=predict(e_net,inputs.observations(name,frame,x,predicted))
        e=decoded["decoded"][0,0].reciprocal()
        t=(tear(features(decoded,x["rgb"]).half().float()).sigmoid()[0]>.5).cpu().numpy().astype(np.int8)
        color=x["rgb"][0].permute(1,2,0)
        pinhole=ref.display(color)
        depths=dict(V0_pred=z,V0_GT=gt,E2k=e)
        arrays={a:d.cpu().numpy() for a,d in depths.items()}
        surface={arm:(*make_visible(color,d,K,t if arm=="E2k" else None),d) for arm,d in depths.items()}
        surface={arm:(vis,d,maximum) for arm,(vis,maximum,d) in surface.items()}
        if previous is not None:
            save_json(sidecar,supplement_zero(previous,surface,unit,name,frame,entry,native,x,K,center,pinhole,cases))
            totals+=1
            print("HIDDEN_COLOR_ZERO_ONLY",totals,slug,frame,flush=True)
            continue
        if last!=name:
            views,memory=ref.gallery.scene_views(dict(unit=unit) if unit else legacy[name,frame])
            evidence=SceneEvidence(unit,name,views)
            last=name
        subj=[ref.near_subject(native["gt"],native["valid"],K.cpu().numpy(),c["focus_m"]) for c in cases]
        selected={a:[ref.p2_focus(d,s) for s in subj] for a,d in arrays.items()}
        stores=selected_view_context(unit,name,views,memory,frame,arrays["V0_pred"],K,cases,
                                     {"P":selected["V0_pred"],"G":selected["V0_GT"]})
        proxy=proxy_generator(geometry,e_net,inputs,evidence,ref.sample,ref.ovg.TRAIN_FRAMES)
        rows=[]
        for i,case in enumerate(cases):
            focus_gt=selected["V0_GT"][i]
            row=dict(cohort=cohort,scene=name,case=case["id"],frame=frame,focus_index=case["focus_index"])
            if focus_gt is None:
                rows.append(dict(row,status="no_valid_nominal_subject"))
                continue
            truth=ref.optical_target(unit,name,case)
            ref_masks=(ref.new_regions(unit,case,native["gt"],native["valid"],K.cpu().numpy()) if unit else
                       ref.legacy_masks(ref.geometry_root(name),case))
            far=case["focus_index"]==1
            region="far_recoverable" if far else "near_in_focus"
            area=ref_masks["recoverable" if far else "in_focus"]
            layout=patch_layout(area) if far else []
            offsets=ref.aperture_disk(64,.05/(2*case["fstop"]),device=K.device,dtype=K.dtype)
            raw=target_metadata(name,frame,case)
            parents,variants,best,witness=read_witnesses(name,case)
            row.update(status="evaluated",area_pixels=int(area.sum()),focus_error={
                arm:comparison_map(subj[i],selected[arm][i],focus_gt,arrays[arm]) for arm in arrays},
                substitute_source=raw_s3_source_coverage(best,witness,parents,variants),render={})
            pictures={} if frame==30 and far and slug in MAIN_CASES else None
            reference_args=(unit,name,frame,entry,case,native,x,K,offsets,center,pinhole,truth,area,
                            region,lpips,layout,parents,variants,best)
            query_args=(unit,name,frame,entry,x,K,truth,area,lpips,layout,evidence,views,proxy,center,pinhole)
            for tag in ("G","P"):
                scored={}
                for arm,info in surface.items():
                    f=focus_gt if tag=="G" else selected[arm][i]
                    if tag=="P" and f==focus_gt:
                        scored[arm]=row["render"]["G"][arm]
                    else:
                        scored[arm]=render_substitutes(info,arm,tag,f,*reference_args,
                                      pictures if tag=="G" else None,raw)
                if tag=="P" and selected["V0_pred"][i]==focus_gt:
                    scored["Query"]=row["render"]["G"]["Query"]
                else:
                    f=focus_gt if tag=="G" else selected["V0_pred"][i]
                    hidden,trace=geometry_cloud(qnet,stores[tag],frame,dict(case,focus_m=f),z,K)
                    scored["Query"]=render_query(surface["V0_pred"][0],hidden,trace,query_args,case,tag,f,
                                                  offsets,pictures if tag=="G" else None)
                row["render"][tag]=scored
            row["zero_checks"]=verify_case(row)
            if pictures is not None:
                picture_root=OUT/"smoke" if smoke else OUT
                files=draw_panel(picture_root,row,truth,area,ref.bands(native,y)[0],pictures)
                row["panel"]=[str(f.relative_to(OUT)) for f in files]
            rows.append(row)
        dest.parent.mkdir(parents=True,exist_ok=True)
        save_json(dest,dict(rows=rows))
        totals+=1
        print("HIDDEN_COLOR",totals,slug,frame,flush=True)
    if not smoke and totals!=54:
        raise ValueError(f"Expected 54 distinct sealed dev and legacy frames, got {totals}")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args()
    try:
        if not args.smoke:
            state("inference")
        evaluate(args.smoke)
        if not args.smoke:
            state("ready-to-collect")
    except BaseException as error:
        if not args.smoke:
            state("halted",reason=str(error))
        raise


if __name__=="__main__":
    main()
