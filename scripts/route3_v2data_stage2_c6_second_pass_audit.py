"""Evaluate whether second-pass Query intercepts the first-pass C6 background."""
import argparse
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
import torch

import route3_v2data_stage1_visualize as gallery
import route3_v2data_stage2_train as stage2
import route3_v2data_stage2_visualize_split5 as split5
from route3_ovg_1b_common import CONFIG
from route3_ovg_1b_optical_data import geometry_root
from route3_s2a_evaluate import masks as legacy_masks
from route3_upper_bound_evaluate import display
from route3_v2data_B_batch import GIB, sha256, write_json
from route3_v2data_B_plan import load
from route3_v2data_stage1_evaluate import new_regions, p2_focus, pooled, region_errors
from route3_v2data_stage2_eval import f2_for, optical_target, render_offsets
from route3_v2data_query_retrain import distribution
from src.aperture_renderer import aperture_disk
from src.footprint_order import push_behind_split_background
from src.layered_aperture import merge_layers
from src.ovg_leakage import query_contribution
from src.ovg_optics import fixed_visible_surfels, raster, subcamera


ZOOM_CASES = (
    "021_dev001_f030_near.png", "051_dev005_f030_near.png",
    "027_dev006_f030_near.png", "003_122104_f030_near.png",
)
FAR_CASES = {"022_dev001_f030_far.png", "052_dev005_f030_far.png", "028_dev006_f030_far.png"}


def quantile(values, probabilities):
    return np.quantile(values, probabilities).tolist() if len(values) else [None]*len(probabilities)


def channel_error(image, ideal):
    return np.abs(image.astype(np.float64)-ideal).max(-1)


def display_pixel_sse(image, input_display):
    difference = display(image).astype(np.float64)-input_display
    return np.square(difference).sum(-1)


def pinhole_partition(image, pinhole, split):
    errors=display_pixel_sse(image,pinhole)
    total=float(errors.sum())
    split_sse=float(errors[split].sum())
    return dict(**pooled([dict(sse=total,pixels=errors.size)]),
                split_sse=split_sse,non_split_sse=float(errors[~split].sum()),
                split_sse_fraction=split_sse/total if total else None,
                non_split_sse_fraction=float(errors[~split].sum())/total if total else None)


def actual_weights(scene, K, focus, center):
    # The RGB channels independently encode source membership. A normal
    # gsplat render then returns the weights after depth sorting.
    ids=scene.source_frames
    indicators=torch.stack((ids==0,ids==2,ids==1),-1).to(scene.colors.dtype).contiguous()
    color,alpha=raster(replace(scene,colors=indicators),K,focus,center,512,512,return_alpha=True)
    weights=torch.cat((color[0],(1-alpha[0])[...,None]),-1)
    return weights.cpu().numpy()


def front_query_weights(scene, K, focus, center, first_depth, predicted_depth, hidden):
    if hidden is None:
        return np.zeros((*first_depth.shape,2),np.float32),np.zeros(first_depth.shape,np.float32)
    intr,view=subcamera(K,focus,center[0])
    total,before=query_contribution(scene,intr,view,K.new_tensor(first_depth))
    if np.array_equal(first_depth,predicted_depth):
        before_prediction=before
    else:
        _,before_prediction=query_contribution(scene,intr,view,K.new_tensor(predicted_depth))
    return torch.stack((before,before_prediction),-1).cpu().numpy(),total.cpu().numpy()


def weighted_summary(facts, restrict=None):
    alpha=np.concatenate([f["alpha"] for f in facts])
    old_error=np.concatenate([f["on_sse"] for f in facts])
    off_error=np.concatenate([f["off_linear_error"] for f in facts])
    on_error=np.concatenate([f["on_linear_error"] for f in facts])
    earlier=np.concatenate([f["query_before"] for f in facts])
    off=np.concatenate([f["off_weights"] for f in facts])
    on=np.concatenate([f["on_weights"] for f in facts])
    mask=np.ones(len(alpha),bool) if restrict is None else restrict(alpha)
    alpha,old_error,off_error,on_error,earlier,off,on=(v[mask] for v in (alpha,old_error,off_error,on_error,earlier,off,on))
    weights={}
    expected=(alpha,1-alpha,np.zeros_like(alpha),np.zeros_like(alpha))
    for mode,values in (("off",off),("on",on)):
        weights[mode]={}
        for i,name in enumerate(("front","first_background","second_query","F2")):
            actual=values[:,i]
            med,p95=quantile(actual,[.5,.95])
            delta=actual-expected[i]
            residual=quantile(np.abs(delta),[.5,.95])
            weights[mode][name]=dict(median=med,p95=p95,mean=float(actual.mean()) if len(actual) else None,
                                     ideal_mean=float(expected[i].mean()) if len(actual) else None,
                                     abs_error_p50=residual[0],abs_error_p95=residual[1])
    positions={}
    total=float(old_error.sum())
    for j,key in enumerate(("rendered_background","predicted_depth2")):
        for threshold,tag in ((0.,"positive"),(.001,"at_least_0_001")):
            contributing=on[:,2]>0 if threshold==0 else on[:,2]>=threshold
            member=(earlier[:,j]>0 if threshold==0 else earlier[:,j]>=threshold)&contributing
            count=int(member.sum())
            fraction=float(old_error[member].sum()/total) if total else 0.
            positions[key+"_"+tag]=dict(pixels=count,pixel_fraction=count/len(alpha) if len(alpha) else 0.,
                                       on_sse_fraction=fraction)
    error={}
    for mode,values in (("off",off_error),("on",on_error)):
        a,b,c=quantile(values,[.5,.95,1.])
        error[mode]=dict(p50=a,p95=b,max=c)
    return dict(pixels=len(alpha),original_split_sse=total,position=positions,weights=weights,linear_error=error)


def write_crop(output, row, truth, off, on, mask, linear_off, linear_on, details):
    # One shared 96×96 window centered on the strongest discrepancy in either
    # condition among the mixed pixels with GT alpha at most one half.
    u=np.maximum(linear_off,linear_on)
    if not mask.any():
        raise ValueError("The designated C6 crop has no low-alpha split pixel")
    ideal,off_rgb,on_rgb=map(display,(truth,off,on))
    box=gallery.largest_error_crop(np.repeat(u[...,None],3,axis=-1),
                                   np.zeros_like(ideal),mask)
    height,width=274,248
    pic=Image.new("RGB",(5*(width+10)+22,height+113),(18,23,31))
    draw=ImageDraw.Draw(pic)
    draw.text((14,8),f'{row["short"]} · f{row["frame"]:03d} · C6 零光圈',
              font=gallery.font(28),fill="white")
    panels=(
        ("GT α 混合",gallery.rgb_image(ideal)),
        ("关第二遍",gallery.rgb_image(off_rgb)),
        ("开第二遍",gallery.rgb_image(on_rgb)),
        ("关第二遍误差",gallery.scalar_image(linear_off*mask,.25,gallery.ERROR_SCALE)),
        ("开第二遍误差",gallery.scalar_image(linear_on*mask,.25,gallery.ERROR_SCALE)),
    )
    for i,(label,image) in enumerate(panels):
        x=14+i*(width+10)
        draw.text((x,55),label,font=gallery.font(18),fill="white")
        tile=image.crop(box).resize((width,width),Image.Resampling.NEAREST)
        pic.paste(tile,(x,84))
    low=details["alpha_at_most_half"]
    wo=low["weights"]["on"]["second_query"]
    footer=(f'α ≤ 0.5 · {low["pixels"]:,} 像素 · 误差色标 0–0.25 · '
            f'p95 关 {low["linear_error"]["off"]["p95"]:.5f} / 开 {low["linear_error"]["on"]["p95"]:.5f} · '
            f'Query 权重 p50 {wo["median"]:.5f} / p95 {wo["p95"]:.5f}')
    draw.text((14,350),footer,font=gallery.font(16),fill="white")
    name=f'crop_{row["file"]}'
    pic.save(output/name)
    return dict(file=name,crop_xyxy=box)


def write_far_crop(output, row, target, images, box, scores):
    width,gap=228,12
    image=Image.new("RGB",(5*width+6*gap,width+115),(18,23,31))
    draw=ImageDraw.Draw(image)
    draw.text((gap,8),f'{row["short"]} · f{row["frame"]:03d} · 远焦可恢复区',
              font=gallery.font(24),fill="white")
    columns=[("GT · Cycles",target)]
    labels=dict(before="推后前",after="推后后")
    for method,views in images.items():
        columns.extend((f'{method} · {labels[mode]}',value) for mode,value in views.items())
    for i,(label,value) in enumerate(columns):
        x=gap+i*(width+gap)
        draw.text((x,51),label,font=gallery.font(18),fill="white")
        image.paste(gallery.rgb_image(value).crop(box).resize((width,width),Image.Resampling.NEAREST),(x,80))
        if i:
            method,mode=label[:2],("before" if i%2 else "after")
            score=scores[method][mode]["psnr_db"]
            if score is not None:
                draw.text((x,width+86),f"{score:.3f} dB",font=gallery.font(16),fill="white")
    image.save(output/("recoverable_"+row["file"]))


def recoverable_scores(rows):
    result={}
    for cohort in ("development","legacy"):
        result[cohort]={}
        for region in ("near","far"):
            grouped=defaultdict(list)
            for row in rows:
                if row["cohort"]==cohort and row["focus"]==region and row["scores"]["C1"]["before"]["pixels"]:
                    grouped[row["scene"]].append(row["scores"])
            result[cohort][region]={}
            for method in ("C1","C6"):
                totals,scenes={},{}
                for mode in ("before","after"):
                    totals[mode]=pooled([r[method][mode] for group in grouped.values() for r in group])
                for name,group in grouped.items():
                    a,b=(pooled([r[method][mode] for r in group]) for mode in ("before","after"))
                    scenes[name]=dict(before=a,after=b,delta_db=b["psnr_db"]-a["psnr_db"])
                a,b=(totals[mode]["psnr_db"] for mode in ("before","after"))
                stats=distribution([r["delta_db"] for r in scenes.values()])
                result[cohort][region][method]=dict(pooled=totals,pooled_delta_db=b-a if scenes else None,
                    scene_values=scenes,scene_deltas=stats,passed=bool(scenes) and stats["median_db"]>=-.1)
    return result


@torch.inference_mode()
def run_recoverable(output,source,validated):
    stage2.configuration()
    torch.cuda.set_per_process_memory_fraction(6*GIB/torch.cuda.get_device_properties(0).total_memory,0)
    stage2.fixed_inputs()
    checkpoint=stage2.RUN/"2a/step-010000.pt"
    validation=load(validated)
    if not validation["pushback"] or not all(validation["composition_gate"].values()):
        raise ValueError("The scaled pushback must pass the complete C6 pinhole gate first")
    checkpoint_hash=sha256(checkpoint)
    if checkpoint_hash!=validation["checkpoint_sha256"]:
        raise ValueError("The pinhole and recovery tests use different checkpoints")
    valid_cases={r["file"] for r in validation["case_rows"]}
    originals={r["file"]:r for r in load(source/"metrics.json")["rows"] if r["file"] in valid_cases}
    model=stage2.Model(stage2.RUN)
    model.heads.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=True)["ema"])
    model.heads.eval()
    rows,skipped,figures=[],[],[]
    started=time.monotonic()
    for plan in gallery.planned_frames():
        active=[(i,originals[r["file"]]) for i,r in enumerate(plan["files"]) if r["file"] in originals]
        if not active:
            continue
        try:
            x,_,g,ref,_,subjects,frames,_,predicted,gt,*_=split5.frame_setup(
                model,plan,"2a",methods=("C1","C6"))
            first=split5.first_observation(model,plan,g)
            K=x["K"][0]
            depths=K.new_tensor(first["depth"])
            masks=dict(C1=torch.as_tensor(predicted,device=K.device),C6=torch.as_tensor(gt,device=K.device))
        except Exception as error:
            skipped.extend(dict(file=r["file"],error=str(error)) for _,r in active)
            continue
        for i,r in active:
            try:
                case=plan["cases"][i]
                unit,name,frame=(plan[key] for key in ("unit","name","frame"))
                depth=g["depth1"][0].cpu().numpy()
                focus=p2_focus(depth,subjects[i])
                if focus is None or abs(focus-r["actual_focus_m"]["S3"])>1e-6:
                    raise ValueError("The pinned focus changed")
                region=(new_regions(unit,case,ref["gt"],ref["valid"],K.cpu().numpy()) if unit
                        else legacy_masks(geometry_root(name),case))["recoverable"]
                target=optical_target(unit,name,case)
                offsets=aperture_disk(CONFIG["native_eval_rays"],.05/(2*case["fstop"]),device=K.device,dtype=K.dtype)
                filler=f2_for(unit,plan["entry"],frame,x,focus,offsets)
                hidden=model.second(split5.item_for(plan),depth,focus,case["fstop"])
                scores,images={},{}
                for method,base in frames.items():
                    moved=push_behind_split_background(hidden,masks[method],depths,K)
                    values={mode:render_offsets(merge_layers(base,layer),K,focus,offsets,filler)
                            for mode,layer in (("before",hidden),("after",moved))}
                    scores[method]={mode:pooled([region_errors(value,target,region)]) for mode,value in values.items()}
                    if r["file"] in FAR_CASES:
                        images[method]={mode:display(value.cpu().numpy()) for mode,value in values.items()}
                if images:
                    write_far_crop(output,r,target,images,r["crop_xyxy"],scores)
                    figures.append("recoverable_"+r["file"])
                rows.append(dict(file=r["file"],scene=name,focus=r["focus"],
                                 cohort="development" if unit else "legacy",scores=scores))
                print("C6_RECOVERABLE",r["file"],{k:[v[m]["psnr_db"] for m in ("before","after")]
                      for k,v in scores.items()},flush=True)
            except Exception as error:
                skipped.append(dict(file=r["file"],error=str(error)))
                print("C6_RECOVERABLE_SKIP",r["file"],str(error),flush=True)
    totals=recoverable_scores(rows)
    complete={r["file"] for r in rows}==valid_cases
    passed=complete and all(v["passed"] for r in totals.values() for m in r.values() for v in m.values())
    write_json(output/"report.json",dict(checkpoint_sha256=checkpoint_hash,seconds=time.monotonic()-started,
        totals=totals,cases=rows,skipped=skipped,figures=figures,complete=complete,passed=passed,
        mask="Full P2 recoverable region, paired on the same pixels at f/1.4",
        priority="Each cohort, method and region must have a scene-median paired delta of at least -0.1 dB."))


@torch.inference_mode()
def run(output, source, pushback=False):
    stage2.configuration()
    torch.cuda.set_per_process_memory_fraction(6*GIB/torch.cuda.get_device_properties(0).total_memory,0)
    torch.backends.cuda.matmul.allow_tf32=False
    stage2.fixed_inputs()
    checkpoint=stage2.RUN/"2a/step-010000.pt"
    reference=load(source/"depth-order-audit/report.json")
    assert sha256(checkpoint)==reference["checkpoint_sha256"]
    original=load(source/"metrics.json")
    originals={r["file"]:r for r in original["rows"] if not r.get("error")}
    pinned={r["file"]:r for r in reference["cases"]}
    model=stage2.Model(stage2.RUN)
    model.heads.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=True)["ema"])
    model.heads.eval()
    rows,facts,crops=[],[],{}
    moves,dev005=[],{}
    skipped=[dict(file=r["file"],error=r["error"]) for r in original["rows"] if r.get("error")]
    largest_discrepancy=0.
    b0_pixels=b0_changed=0
    b0_max_difference=0.
    started=time.monotonic()
    plans=list(gallery.planned_frames())
    plans.sort(key=lambda plan:next((ZOOM_CASES.index(v["file"]) for v in plan["files"]
                                     if v["file"] in ZOOM_CASES),len(ZOOM_CASES)))
    for plan in plans:
        active=[(i,originals[r["file"]]) for i,r in enumerate(plan["files"]) if r["file"] in originals]
        if not active:
            continue
        try:
            x,y,g,_,_,subjects,frames,_,_,split,rgb,_,mixture=split5.frame_setup(model,plan,"2a",only_c6=True)
            K=x["K"][0]
            alpha=y["alpha"][0].cpu().numpy()
            foreground_depth=g["depth1"][0].cpu().numpy()
            first_depth=split5.first_observation(model,plan,g)["depth"]
            predicted_depth=g["depth2"][0].cpu().numpy()
            pinhole=display(rgb).astype(np.float64)
            base=frames["C6"]
            assert torch.unique(base.source_frames).cpu().tolist() in ([0,2],[0])
        except Exception as error:
            skips=[dict(file=r["file"],error=str(error)) for _,r in active]
            skipped.extend(skips)
            print("C6_QUERY_SKIP",skips,flush=True)
            continue
        for i,r in active:
            try:
                case=plan["cases"][i]
                focus=r["actual_focus_m"]["S3"]
                computed=p2_focus(foreground_depth,subjects[i])
                if computed is None or abs(focus-computed)>1e-6:
                    raise ValueError("Actual P2 focus differs from the calibrated baseline")
                center=K.new_zeros((1,2))
                f2=f2_for(plan["unit"],plan["entry"],plan["frame"],x,focus,center)
                hidden=model.second(split5.item_for(plan),foreground_depth,focus,case["fstop"])
                if hidden is not None:
                    assert (hidden.source_frames==1).all()
                combined=merge_layers(base,hidden)
                off=raster(base,K,focus,center,512,512,f2)[0].cpu().numpy()
                ordinary=raster(combined,K,focus,center,512,512,f2)[0].cpu().numpy()
                if abs(display_pixel_sse(ordinary,pinhole).sum()-pinned[r["file"]]["sse"])>1e-7:
                    raise ValueError("Second-pass-on result differs from the calibrated depth audit")
                on=ordinary
                if pushback:
                    original_hidden=hidden
                    move={}
                    hidden=push_behind_split_background(hidden,torch.as_tensor(split,device=K.device),
                                                        K.new_tensor(first_depth),K,audit=move)
                    combined=merge_layers(base,hidden)
                    on=raster(combined,K,focus,center,512,512,f2)[0].cpu().numpy()
                    difference=channel_error(on,ordinary)[~split]
                    b0_pixels+=difference.size
                    b0_changed+=int((difference>1e-6).sum())
                    b0_max_difference=max(b0_max_difference,float(difference.max()))
                    if r["short"]=="dev005" and r["frame"]==30:
                        none=merge_layers(fixed_visible_surfels(g,x),original_hidden)
                        none=raster(none,K,focus,center,512,512,f2)[0].cpu().numpy()
                        dev005[r["file"]]={key:pinhole_partition(value,pinhole,split)
                                           for key,value in (("none",none),("before",ordinary),("after",on))}
                off_sse,on_sse=(display_pixel_sse(v,pinhole) for v in (off,on))
                off_error,on_error=(channel_error(v,mixture) for v in (off,on))
                off_weights=actual_weights(base,K,focus,center)
                on_weights=actual_weights(combined,K,focus,center)
                before,trace_total=front_query_weights(combined,K,focus,center,first_depth,predicted_depth,hidden)
                if split.any():
                    largest_discrepancy=max(largest_discrepancy,float(np.max(np.abs(trace_total[split]-on_weights[split,2]))))
                fact=dict(alpha=alpha[split],on_sse=on_sse[split],off_linear_error=off_error[split],
                          on_linear_error=on_error[split],query_before=before[split],
                          off_weights=off_weights[split],on_weights=on_weights[split])
                if r["file"] in ZOOM_CASES:
                    facts_case=dict(all_split=weighted_summary([fact]),
                                    alpha_at_most_half=weighted_summary([fact],lambda a:a<=.5))
                    crops[r["file"]]=dict(facts_case,**write_crop(output,r,mixture,off,on,split&(alpha<=.5),
                                                                 off_error,on_error,facts_case))
                    highlight=facts_case["alpha_at_most_half"]
                    print("C6_QUERY_CROP",r["file"],highlight["linear_error"],
                          highlight["weights"]["on"]["second_query"],flush=True)
                    if pushback and r["file"] in split5.PINHOLE_CASES:
                        crops[r["file"]]["repaired_pinhole"]=split5.pinhole_audit(output,r,on,mixture,split)
                if pushback:
                    moves.append((r["file"],move))
                facts.append((plan["unit"] is not None,fact))
                rows.append(dict(file=r["file"],scene=r["scene"],split_pixels=int(split.sum()),pixels=off_sse.size,
                                 off_sse=float(off_sse.sum()),on_sse=float(on_sse.sum())))
                value=pooled([dict(sse=off_sse.sum().item(),pixels=off_sse.size)])["psnr_db"]
                print("C6_QUERY_OFF",r["file"],round(value,4),flush=True)
            except Exception as error:
                skipped.append(dict(file=r["file"],error=str(error)))
                print("C6_QUERY_SKIP",r["file"],str(error),flush=True)
    result={}
    for group in ("all","development","legacy"):
        chosen=rows if group=="all" else [r for r in rows if r["scene"].startswith("development-")==(group=="development")]
        seen=[f for v,f in facts if group=="all" or v==(group=="development")]
        result[group]={"split":weighted_summary(seen),"split_alpha_at_most_half":weighted_summary(seen,lambda a:a<=.5)}
        for mode in ("off","on"):
            scores=[dict(sse=r[mode+"_sse"],pixels=r["pixels"]) for r in chosen]
            result[group][mode]=dict(**pooled(scores),cases=len(scores),
                                     pass_60=sum(pooled([s])["psnr_db"]>=60 for s in scores))
    dev=result["development"]
    before=dev["split"]["position"]["rendered_background_positive"]
    pass_60=dev["off"]["psnr_db"]>=60
    p95={key:summary["all_split"]["linear_error"]["on"]["p95"] for key,summary in crops.items()
         if key in split5.PINHOLE_CASES}
    movement=None
    if pushback:
        total=sum(v["total"] for _,v in moves)
        moved=sum(v["moved"] for _,v in moves)
        movement=dict(total=total,moved=moved,fraction=moved/total if total else 0.,
                      per_case={k:dict(total=v["total"],moved=v["moved"]) for k,v in moves},
                      quantiles={k:dict(zip(("min","p10","p50","p95","max"),
                          quantile(np.concatenate([v[k] for _,v in moves]),[0,.1,.5,.95,1.])))
                          for k in ("depth_ratio","center_depth_ratio","scale_ratio")})
    write_json(output/"report.json",dict(checkpoint_sha256=sha256(checkpoint),seconds=time.monotonic()-started,
               totals=result,frames=crops,case_rows=rows,skipped=skipped,query_trace_max_difference=largest_discrepancy,
               pushback=pushback,movement=movement,dev005_f030=dev005,
               b0_change=dict(pixels=b0_pixels,changed_above_1e_6=b0_changed,max_abs_linear=b0_max_difference),
               composition_gate=dict(cases_complete=len(rows)==len(pinned),
                                     full_psnr_at_least_60=result["all"]["on"]["psnr_db"]>=60,
                                     three_p95_below_0_01=set(p95)==split5.PINHOLE_CASES and all(v<.01 for v in p95.values())),
               gate=dict(dev_off_at_least_60=pass_60,front_query_sse_at_least_half=before["on_sse_fraction"]>=.5,
                         apply_pushback=pass_60,reason="The supplemental gate requires real Query-off C6 >=60 dB."),
               definitions=dict(positive_before="An admitted second-pass contribution with nonzero actual compositing weight and depth below the limit.",
                                rendered_background="Selected first-pass depth actually sent to the C6 renderer.",
                                depth2="Frozen geometry head depth2, measured separately.",
                                error="Maximum absolute linear RGB channel error versus the GT mixture.",
                                weights="RGB-tagged render gives actual front, first-pass, second-pass weights. Residual alpha goes to F2.",
                                sse="Display RGB squared error against input pinhole, original second-pass-on rendering.")))
    print("C6_QUERY_RESULT",dev["off"]["psnr_db"],before,result["legacy"]["off"]["psnr_db"],flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=gallery.OUT)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--pushback",action="store_true")
    parser.add_argument("--recoverable-after",type=Path,help="Successful scaled C6 pinhole report")
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    if args.recoverable_after:
        run_recoverable(args.output.resolve(),args.source.resolve(),args.recoverable_after.resolve())
    else:
        run(args.output.resolve(),args.source.resolve(),args.pushback)
