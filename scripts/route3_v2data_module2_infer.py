"""Paired V0 and connected-depth evaluation on the sealed development split."""
import argparse
from collections import defaultdict
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw
import psutil
import torch

import route3_ovg_data as ovg
import route3_v2data_stage1_visualize as gallery
from route3_ovg_1b_optical_data import b_oracle, place
from route3_ovg_opaque_b0 import RAY_CLASSES, bins
from route3_ovg_stage2_evaluate import ray_effect, trace_predicted
from route3_ovg_subpixel import save_json
from route3_v2data_B_batch import GIB, ROOT
from route3_v2data_B_plan import load
from route3_v2data_module2_labels import CACHE, FOUR, OUT
from route3_v2data_optics import hidden_parents, ray_fields
from route3_v2data_query_retrain import distribution
from route3_v2data_stage1_evaluate import legacy_masks, near_subject, new_regions, p2_focus, pooled
from route3_v2data_stage2_eval import bands, f2_for, frames, geometry_root, k16, optical_target
from route3_v2data_stage2_train import appearance_labels, sample
from route3_v2data_stage3a import gpu_setup, load_parents
from route3_v2data_stage3a_prior_diagnostic import TARGETS, THIN_SCENES, thin_regions
from route3_v2data_stage3b import load_geometry
from src.aperture_renderer import aperture_disk
from src.c1_bokehme import region_sse
from src.layered_aperture import merge_layers, pixel_layer
from src.ovg_connected_depth import DIRECTIONS, edge_slices, relative_jump, rule_tears, subdivide
from src.ovg_optics import raster

ARMS = ("V0_GT","a","V0_pred","b","b_union","c_003","c_010")
PAIR = dict(a="V0_GT",b="V0_pred",b_union="V0_pred",c_003="V0_pred",c_010="V0_pred")
ARM_NAMES = dict(zip(ARMS,("V0 GT","a · GT","V0 预测","b · GT 撕裂","b∪ · 并集","c · τ=.03","c · τ=.10")))
REGIONS = ("near_in_focus","far_recoverable","boundary_2px","full_image")
ZERO = ("zero_visible","zero_combined","zero_visible_common","zero_combined_common")
FLOORS = dict(zero_visible=50.,zero_combined=60.)
GROUPS = ("foreground","gap_background","other")
PARENTS = ROOT/"outputs/route3-v2data-stage3a-2026-11-24-02/dev-hidden-parents"
COMPETITOR = ROOT.parent/"route3-stage3b-20261125/outputs/route3-v2data-stage3b-2026-11-25-01"
GPU_GIB = 4.5


def title(name):
    return "dev"+name.split("-")[1] if name.startswith("development-") else name.split("-")[2]


def competitor_state():
    progress = load(COMPETITOR/"progress.json") or {}
    return {k:progress.get(k) for k in ("step","seconds","at")}


def priority():
    file = COMPETITOR/"followup.pid"
    if file.is_file():
        try:
            pid = int(file.read_text())
            if any("route3_v2data_stage3b_eval.py" in " ".join(child.cmdline())
                   for child in psutil.Process(pid).children(recursive=True)):
                return "3b 正式评价"
        except (psutil.Error,ValueError):
            pass
    used,total = (int(n.strip()) for n in subprocess.check_output(
        ["nvidia-smi","--query-gpu=memory.used,memory.total","--format=csv,noheader,nounits"],text=True).strip().split(","))
    if min(total,30*1024)-used < (GPU_GIB+3)*1024:
        return f"共享 GPU 显存已使用 {used} MiB"
    return None


def yield_gpu():
    reason = priority()
    if reason is None:
        return 0.
    began = time.monotonic()
    torch.cuda.empty_cache()
    print("MODULE2_GPU_YIELD",reason,flush=True)
    while priority() is not None:
        time.sleep(10)
    return time.monotonic()-began


def subsets(unit,name,frame,ref):
    slug = title(name)
    if slug in TARGETS:
        native = ovg.b_foreground_reference(name,frame)
        out = thin_regions(slug,unit,native)[0]
    else:
        out = dict(foreground=np.zeros_like(ref["valid"]),gap_background=np.zeros_like(ref["valid"]))
    return dict(out,other=~(out["foreground"]|out["gap_background"]))


def labels_for_arm(status,pred_inv):
    loose = rule_tears(pred_inv,.10)
    strict = rule_tears(pred_inv,.03)
    merged = np.where(status==-2,-2,((status!=0)|(loose==1)).astype(np.int8))
    return dict(a=status,b=status,b_union=merged,c_003=strict,c_010=loose)


def mark_overlay(overlay,mask,a,dx,dy,paint):
    yy,xx=np.nonzero(mask)
    # Draw each half-edge at both ends so it stays visible after scaling.
    yy+=a[0].start or 0
    xx+=a[1].start or 0
    overlay[yy,xx]=paint
    overlay[yy+dy,xx+dx]=paint


def tear_audit(gt,pred_inv,labels,zones,learned=()):
    errors = {key:{g:defaultdict(int) for g in GROUPS} for key in ("b_union","c_003","c_010",*learned)}
    mismatch = {k:dict(edges=0,GT_connected=0) for k in GROUPS}
    maps = {k:np.zeros((*pred_inv.shape,3),np.uint8) for k in ("b","b_union","c_003","c_010",*learned)}
    for k,(dy,dx) in enumerate(DIRECTIONS):
        a,b = edge_slices(dy,dx)
        t = gt[k][a]
        connected,disconnected = t==0,t==1
        rod = zones["foreground"][a]|zones["foreground"][b]
        gap = (zones["gap_background"][a]|zones["gap_background"][b])&~rod
        regions = dict(foreground=rod,gap_background=gap,other=~(rod|gap))
        misaligned = connected&(relative_jump(pred_inv[a],pred_inv[b])>.10)
        mark_overlay(maps["b"],misaligned,a,dx,dy,(255,203,32))
        for name,m in regions.items():
            mismatch[name]["edges"] += int((misaligned&m).sum())
            mismatch[name]["GT_connected"] += int((connected&m).sum())
        for arm in errors:
            predicted = labels[arm][k][a]!=0
            false_tear = predicted&connected
            false_link = ~predicted&disconnected
            mark_overlay(maps[arm],false_link,a,dx,dy,(255,30,30))
            mark_overlay(maps[arm],false_tear,a,dx,dy,(30,155,255))
            if arm in learned:
                mark_overlay(maps[arm],false_tear&misaligned,a,dx,dy,(255,203,32))
            for name,m in regions.items():
                c = errors[arm][name]
                c["GT_connected"] += int((connected&m).sum())
                c["GT_torn"] += int((disconnected&m).sum())
                c["false_torn"] += int((false_tear&m).sum())
                c["false_connected"] += int((false_link&m).sum())
                if arm in learned:
                    c["false_step_connected"] += int((misaligned&m).sum())
                    c["false_step_torn"] += int((false_tear&misaligned&m).sum())
    return errors,mismatch,maps


def interp_counts(mode,zones):
    return {zone:{name:int((mode[m]==i).sum()) for i,name in
                  enumerate(("constant","linear","triangle","fixed_center"))}
            for zone,m in zones.items()}


def display(value):
    return gallery.display(value.cpu().numpy())


def light(visible,hidden,K,focus,rays,filled):
    cloud = merge_layers(visible,hidden)
    parts = (raster(cloud,K,focus,r,512,512,f).clamp(0,1).sum(0)
             for r,f in zip(rays.split(4),filled.split(4)))
    return sum(parts)/len(rays)


def place_hidden(unit,name,frame,case,ref,depth,maximum,K):
    if unit is None:
        parents,front,background = b_oracle(name,frame)
    else:
        path = PARENTS/name/(case["id"]+"-optics.npz")
        if path.is_file():
            parents = load_parents(path)
        else:
            spec = dict(tiers="B",unit=unit,frame=frame,focus=case["focus_m"],fstop=case["fstop"],
                        label=unit/"labels"/f'f{frame:03d}-focus{case["focus_index"]}-K16.npz')
            parents = hidden_parents(spec,ray_fields(spec),ref)
        front = ref["gt"]
        background = ref["valid"]&~np.isin(ref["object_id"],ref["foreground"])
    if parents is None:
        return None
    return place(parents,front,background,depth,K,512,order_depth=maximum)[0]


def ray_diagnostics(visible,hidden,K,nominal,offsets,fill,mask,truth):
    base_result,alphas = [],[]
    for r,f in zip(offsets.split(4),fill.split(4)):
        rgb,alpha = raster(visible,K,nominal,r,512,512,f,return_alpha=True)
        base_result.append(rgb)
        alphas.append(alpha)
    coverage = torch.cat(alphas)
    high = (1-coverage)>=.1
    support = torch.as_tensor(mask,device=K.device)
    result = dict(rays=int(mask.sum())*len(offsets),high=int(high[:,support].sum()),positive_sse_bins=[])
    heat = high.float().mean(0).cpu().numpy()
    if hidden is None or not mask.any():
        return result,heat
    combined = merge_layers(visible,hidden)
    before = torch.cat(base_result)[:,support].cpu().numpy()
    after = np.concatenate([raster(combined,K,nominal,r,512,512,f)[:,support].cpu().numpy()
                            for r,f in zip(offsets.split(4),fill.split(4))])
    _,effect,_,_ = ray_effect(before,after,truth[mask])
    traces = [trace_predicted(visible,visible,combined,K,nominal,d,support) for d in offsets]
    weights,hidden_z,visible_z,_,prefix = map(np.stack,zip(*traces))
    result["positive_sse_bins"] = bins(weights,hidden_z,visible_z,prefix,effect)
    return result,heat


def common_split(y,ref,item):
    gt = appearance_labels(item)
    first,second,alpha = [y[k][0].cpu().numpy() for k in ("depth1","depth2","alpha")]
    return (y["valid_alpha"][0].cpu().numpy()&y["valid_dual"][0].cpu().numpy()&ref["valid"]
            &np.isfinite(first)&np.isfinite(second)&(first>0)&(second>first)
            &(alpha>0)&(alpha<1)&gt["comp_valid"]&gt["color_valid"])


def crop_for(value,reference,mask):
    return gallery.largest_error_crop(value,reference,mask if mask.any() else np.ones(mask.shape,bool))


def panel(path,row,snap,truth,ref,masks,overlays,arms):
    tile,gap=182,8
    columns=("GT",*arms)
    pictures = {"GT":truth,**snap}
    focus = row["focus_index"]
    base = snap["V0_pred"] if "V0_pred" in snap else snap["V0_GT"]
    region = masks["near_in_focus"] if focus==0 else masks["far_recoverable"]
    edge = masks["boundary_2px"]
    inside = crop_for(base,truth,region&~edge if focus==0 else region&edge)
    outline = crop_for(base,truth,edge)
    figure=Image.new("RGB",(len(columns)*(tile+gap),52+5*(tile+25)+55),(18,22,32))
    label=ImageDraw.Draw(figure)
    label.text((7,5),f'{row["scene"]} · {row["case"]}',font=gallery.font(17),fill="white")
    label.text((7,28),"上起全图、目标区放大、边界放大、光学误差、透射≥0.1 / 故障边",font=gallery.font(13),fill="white")
    for x,method in enumerate(columns):
        left=x*(tile+gap)
        value = pictures[method]
        if method=="GT":
            cname="Cycles 真值"
            error=Image.new("RGB",(512,512))
            diagnostic=None
        else:
            rec=row["methods"][method]
            pname=score_psnr(rec["optical"]["full_image"])
            cname=f"{ARM_NAMES[method]} · {pname:.1f}"
            error=gallery.scalar_image(np.abs(value-truth).mean(-1),.25,cv2.COLORMAP_INFERNO)
            diagnostic=None
            heat=rec.get("heat")
            if heat is not None:
                diagnostic=gallery.scalar_image(heat,1.,cv2.COLORMAP_INFERNO)
            if method in overlays:
                plain=ref.copy()*0.48
                marked=overlays[method]>0
                plain[marked.any(-1)]=overlays[method][marked.any(-1)]/255.
                tear=gallery.rgb_image(plain).crop(outline)
                if diagnostic is None:
                    diagnostic=tear
                else:
                    combined=Image.new("RGB",(tile,tile))
                    combined.paste(diagnostic.resize((tile,tile//2)),(0,0))
                    combined.paste(tear.resize((tile,tile//2)),(0,tile//2))
                    diagnostic=combined
        label.text((left+2,53),cname,font=gallery.font(13),fill="white")
        rgb=gallery.rgb_image(value)
        whole=rgb.copy()
        if method=="GT":
            d=ImageDraw.Draw(whole)
            d.rectangle(tuple(inside),outline=(255,192,70),width=2)
            d.rectangle(tuple(outline),outline=(70,205,235),width=2)
        for j,layer in enumerate((whole,rgb.crop(inside),rgb.crop(outline),error,diagnostic)):
            if layer is not None:
                y=76+j*(tile+25)
                figure.paste(layer.resize((tile,tile),Image.Resampling.NEAREST if j in (1,2) else Image.Resampling.BOX),(left,y))
    label.text((7,76+5*(tile+25)-8),"固定误差色标 0–0.25 · 红色误连 · 蓝色误断 · 黄色 GT 连通但预测深度台阶 > 0.10",
               font=gallery.font(14),fill="white")
    path.parent.mkdir(parents=True,exist_ok=True)
    figure.save(path,quality=91)
    return dict(target_crop_xyxy=inside,outline_crop_xyxy=outline,panel=str(path))


def score_psnr(score):
    return pooled([score])["psnr_db"]


def frame_results(net,cohort,name,frame,unit,entry,cases,root,arms,*,
                  tear_predictor=None,panel_writer=panel):
    item = ("B",name,ovg.TRAIN_FRAMES.index(frame))
    x,y = sample(item)
    prediction = net(x,protect_residual=True,render_heads=False,return_features=tear_predictor is not None)
    pred = prediction["depth1"][0]
    K = x["K"][0]
    k = K.cpu().numpy()
    ref = ovg.b_edge_reference(item,y)
    sensor = torch.as_tensor(ref["gt"],device=K.device,dtype=K.dtype)
    known = torch.as_tensor(ref["valid"],device=K.device)&torch.isfinite(sensor)&(sensor>0)
    gt_depth = torch.where(known,sensor,pred)
    color = x["rgb"][0].permute(1,2,0)
    pinhole = display(color)
    full = np.ones_like(ref["valid"],bool)
    edge = bands(ref,y)[0]
    occupied = common_split(y,ref,item)
    center = K.new_zeros((1,2))
    zones = subsets(unit,name,frame,ref)
    label = ovg.load(CACHE/name/f"f{frame:03d}.npz")["status"]
    rho_pred = pred.reciprocal().cpu().numpy()
    tears = labels_for_arm(label,rho_pred)
    learned = tear_predictor(x,prediction) if tear_predictor is not None else {}
    tears.update(learned)
    audit,misaligned,overlay = tear_audit(label,rho_pred,tears,zones,learned)
    del prediction
    pending = []
    for case in cases:
        nominal,idx = case["focus_m"],case["focus_index"]
        subject = near_subject(ref["gt"],ref["valid"],k,nominal)
        focus_gt,focus_pred = (p2_focus(d.cpu().numpy(),subject) for d in (gt_depth,pred))
        row = dict(scene=name,cohort=cohort,frame=frame,case=case["id"],focus_index=idx,
                   nominal_focus_m=nominal,GT_focus_m=focus_gt,methods={},
                   status="unscored" if focus_gt is None or focus_pred is None else "evaluated")
        if row["status"]=="unscored":
            pending.append((case,row,None,None,None))
            continue
        gt = optical_target(unit,name,case)
        rs = new_regions(unit,case,ref["gt"],ref["valid"],k) if unit else legacy_masks(geometry_root(name),case)
        masks = dict(boundary_2px=edge,full_image=full)
        if idx==0:
            masks["near_in_focus"]=rs["in_focus"]
            masks["far_recoverable"]=np.zeros_like(edge)
        else:
            masks["far_recoverable"]=rs["recoverable"]
            masks["near_in_focus"]=np.zeros_like(edge)
        pending.append((case,row,gt,masks,dict(gt=focus_gt,pred=focus_pred)))
    images = {c["id"]:{} for c,*_ in pending}
    stats = {}
    stopped = 0.
    for arm in arms:
        stopped += yield_gpu()
        which = "gt" if arm in ("V0_GT","a") else "pred"
        depth = gt_depth if which=="gt" else pred
        maximum = depth
        options = {}
        if arm in tears:
            native_inv = depth.reciprocal().cpu().numpy()
            inverse,mode,diagonals = subdivide(native_inv,tears[arm])
            if not np.array_equal(inverse[...,2,2],native_inv):
                raise ValueError("The interpolation changed the native P2 center depth")
            sub = 1/inverse
            sub[...,2,2] = depth.cpu().numpy()
            maximum = K.new_tensor(sub.max(axis=(2,3)))
            options["subpixel_depth"] = K.new_tensor(sub.transpose(0,2,1,3).reshape(2560,2560))
            stats[arm] = dict(interpolation=interp_counts(mode,zones),diagonals=diagonals)
        visible = pixel_layer(color,depth,K,torch.eye(4,device=K.device,dtype=K.dtype),**options)
        zv = display(raster(visible,K,1.4,center,512,512)[0])
        for case,row,gt,masks,focuses in pending:
            if focuses is None:
                continue
            focus = focuses[which]
            rays = aperture_disk(64,.05/(2*case["fstop"]),device=K.device,dtype=K.dtype)
            fill = f2_for(unit,entry,frame,x,focus,rays)
            zfill = f2_for(unit,entry,frame,x,focus,center)
            hidden = place_hidden(unit,name,frame,case,ref,depth,maximum,K)
            on = display(light(visible,hidden,K,focus,rays,fill))
            if case["focus_index"]==0:
                off = display(light(visible,None,K,focus,rays,fill)) if hidden is not None else on
                off_score = region_sse(off,gt,masks["near_in_focus"])
                rays_label = k16(unit,name,frame,case)
                z=rays_label["depth"]
                good = ((rays_label["ids"]==ref["object_id"][None])&rays_label["known"]&~rays_label["hidden"]
                        &np.isfinite(z)&(z>0))
                interior = (y["alpha"][0].cpu().numpy()>=.9999)&y["valid_alpha"][0].cpu().numpy()&~ref["boundary"]
                support = interior&ref["valid"]&np.isin(ref["object_id"],ref["foreground"])&good.all(0)
                nominal = float(case["focus_m"])
                replays = K.new_tensor(rays_label["aperture_xy"])
                kfill = f2_for(unit,entry,frame,x,nominal,replays)
                diagnostics,heat = ray_diagnostics(visible,hidden,K,nominal,replays,kfill,support,gt)
            else:
                off_score,diagnostics,heat = None,None,None
            combined = display(raster(merge_layers(visible,hidden),K,focus,center,512,512,zfill)[0])
            row["methods"][arm] = dict(focus_m=focus,GT_focus_m=row["GT_focus_m"],
                optical={m:region_sse(on,gt,s) for m,s in masks.items()},
                near_off=off_score,rays=diagnostics,heat=heat,
                zero_visible=region_sse(zv,pinhole,full),zero_combined=region_sse(combined,pinhole,full),
                zero_visible_common=region_sse(zv,pinhole,occupied),zero_combined_common=region_sse(combined,pinhole,occupied))
            if any(row["methods"][arm]["optical"][key]["pixels"]!=int(mask.sum()) for key,mask in masks.items()):
                raise ValueError(f"Nonfinite rendered support for {name}/{case['id']}/{arm}")
            images[case["id"]][arm]=on
        del visible,options
    for case,row,gt,masks,focuses in pending:
        if focuses is not None:
            file = root/"panels"/f'{title(name)}-{case["id"]}.jpg'
            row.update(panel_writer(file,row,images[case["id"]],gt,pinhole,masks,overlay,arms))
            for rec in row["methods"].values():
                del rec["heat"]
    return dict(rows=[r[1] for r in pending],tear_accuracy=audit,alignment=misaligned,
                interpolation=stats,shared_dual_pixels=int(occupied.sum()),gpu_wait_seconds=stopped)


def lab_stats(path):
    data = load(path)
    totals = {}
    keys = ("edges","prescreen","rays","connected","torn","invalid","different_id_connected",
            "center_invalid","midpoint_no_hit","third_surface","midpoint_uncertain")
    for group in ("development","thin_structure","dev008","legacy"):
        rows = [r for scene in data["rows"] for r in scene.get("frames",[])
                if group==r["cohort"] or group==r["subset"]]
        counts = {key:sum(c[key] for row in rows for c in row["directions"]) for key in keys}
        counts["fractions"] = {key:counts[key]/counts["edges"] if counts["edges"] else None
                               for key in ("connected","torn","invalid")}
        counts["alignment"] = {k:sum(r["alignment"][k] for r in rows) for k in ("rays","matched")}
        totals[group]=counts
    return dict(protocol=data["protocol"],totals=totals,
                skipped_legacy=[r for r in data["rows"] if r.get("skipped")],
                seconds=data["seconds"],workers=data["workers"])


def pooled_arms(rows,arm):
    near = [r["methods"][arm] for r in rows if r["focus_index"]==0]
    active = [r["methods"][arm] for r in rows]
    on = pooled([r["optical"]["near_in_focus"] for r in near])
    off = pooled([r["near_off"] for r in near])
    if on["pixels"]!=off["pixels"]:
        raise ValueError("Unequal support in the GT substitute–F2 comparison")
    optics = dict(near_gt_minus_f2=dict(on=on,off=off,psnr_db=on["psnr_db"]-off["psnr_db"] if on["pixels"] else None))
    optics.update({k:pooled([r["optical"][k] for r in active]) for k in REGIONS})
    for key in ZERO:
        uses = [r[key] for r in active if r[key]["pixels"]]
        floor = FLOORS.get(key)
        optics[key] = dict(**pooled(uses),scored_cases=len(uses),
                           passed_cases=sum(score_psnr(v)>=floor for v in uses) if floor else None)
    rays,high = (sum(r["rays"][key] for r in near) for key in ("rays","high"))
    sbins = defaultdict(float)
    for r in near:
        for b in r["rays"]["positive_sse_bins"]:
            sbins[b["category"]] += b["positive_SSE"]
    optics["transmission"] = dict(rays=rays,high_rays=high,high_fraction=high/rays if rays else None,
                                  positive_sse_bins=dict(sbins),
                                  high_positive_sse_share=sbins[RAY_CLASSES[3]]/sum(sbins.values()) if sum(sbins.values()) else None)
    return optics


def compare(group,base,new,result,*,include_absolute_near=False):
    regions = REGIONS if include_absolute_near else REGIONS[1:]
    keys = ("near_gt_minus_f2",*regions,*ZERO)
    values = {}
    for k in keys:
        ds = {}
        for name,rows in group.items():
            before,after = (r[k]["psnr_db"] for r in (rows[base],rows[new]))
            if before is not None and after is not None:
                ds[name]=after-before
        before,after = (result[r][k]["psnr_db"] for r in (base,new))
        values[k]=dict(pooled_delta_db=after-before if before is not None and after is not None else None,
                       per_scene_db=ds,**distribution(list(ds.values())))
    return values


def summarize(parts,arms,*,pairings=PAIR,include_absolute_near=False):
    raw = [r for part in parts for r in part["rows"]]
    result = {}
    for cohort in ("development","legacy","thin_structure","dev008"):
        if cohort in ("development","legacy"):
            own = [r for r in raw if r["cohort"]==cohort]
        else:
            ids = THIN_SCENES if cohort=="thin_structure" else ("dev008",)
            own = [r for r in raw if title(r["scene"]) in ids]
        rows = [r for r in own if r["status"]=="evaluated"]
        if not rows:
            continue
        by_scene = defaultdict(list)
        for r in rows:
            by_scene[r["scene"]].append(r)
        grouped = {scene:{arm:pooled_arms(rr,arm) for arm in arms} for scene,rr in by_scene.items()}
        all_methods = {arm:pooled_arms(rows,arm) for arm in arms}
        comparisons = {arm:compare(grouped,pairings[arm],arm,all_methods,include_absolute_near=include_absolute_near)
                       for arm in arms if arm in pairings}
        result[cohort] = dict(scored=len(rows),unscored=[r for r in own if r["status"]!="evaluated"],
                              methods=all_methods,per_scene=grouped,paired=comparisons)
    return result


def acc_stats(parts):
    accuracy,misaligned,interp={},defaultdict(lambda:defaultdict(int)),{}
    for p in parts:
        for arm,labels in p["tear_accuracy"].items():
            out = accuracy.setdefault(arm,{zone:defaultdict(int) for zone in GROUPS})
            for k,row in labels.items():
                for attr,n in row.items():
                    out[k][attr]+=n
        for zone,row in p["alignment"].items():
            for k,v in row.items():
                misaligned[zone][k]+=v
        for arm,row in p["interpolation"].items():
            out = interp.setdefault(arm,dict(zones={g:defaultdict(int) for g in GROUPS},diagonals=defaultdict(int)))
            for zone,counts in row["interpolation"].items():
                for k,v in counts.items():
                    out["zones"][zone][k]+=v
            for key,n in row["diagonals"].items():
                out["diagonals"][key]+=n
    for arm in accuracy.values():
        for v in arm.values():
            if "false_step_connected" in v:
                v["false_step_torn_fraction"] = (v["false_step_torn"]/v["false_step_connected"]
                                                 if v["false_step_connected"] else None)
            v["false_connected_fraction"] = v["false_connected"]/v["GT_torn"] if v["GT_torn"] else None
            v["false_torn_fraction"] = v["false_torn"]/v["GT_connected"] if v["GT_connected"] else None
    for row in interp.values():
        for v in row["zones"].values():
            all_samples = sum(v.values())
            v["fractions"]={k:n/all_samples if all_samples else None for k,n in v.items()}
    return dict(tear_accuracy=accuracy,alignment=misaligned,interpolation=interp)


@torch.inference_mode()
def run(only_gate):
    label_metadata=load(OUT/"labels.json")
    if not label_metadata:
        raise FileNotFoundError("Complete the CPU labels before rendering")
    root=OUT/("a-check" if only_gate else "all-arms")
    root.mkdir(parents=True,exist_ok=True)
    gate = load(OUT/"a-visual-review.json")
    if not only_gate and (not gate or not gate.get("passed")):
        raise ValueError("Read the GT geometry A panels before starting the comparison")
    arms=ARMS[:2] if only_gate else ARMS
    skip={r["scene"] for r in label_metadata["rows"] if r.get("skipped")}
    start=time.monotonic()
    competitor_start=competitor_state()
    gpu_setup(dict(training=dict(max_process_gpu_gib=GPU_GIB)))
    waited=yield_gpu()
    torch.cuda.reset_peak_memory_stats()
    geometry=load_geometry()
    parts=[]
    work=(root/"frames")
    for cohort,name,frame,unit,entry,cases in frames():
        if name in skip or (only_gate and (cohort!="development" or name.split("-")[1] not in FOUR or frame!=30)):
            continue
        cached=work/name/f"f{frame:03d}.json"
        if cached.is_file():
            parts.append(load(cached))
            continue
        waited += yield_gpu()
        row=frame_results(geometry,cohort,name,frame,unit,entry,cases,root,arms)
        parts.append(row)
        save_json(cached,row)
        print("MODULE2_FRAME",cohort,name,frame,len(parts),flush=True)
    cases=[r for part in parts for r in part["rows"]]
    meta=dict(arms=list(arms),summary=summarize(parts,arms),case_rows=cases,diagnostics=acc_stats(parts),
              labels=lab_stats(OUT/"labels.json"),seconds=time.monotonic()-start,
              peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),test_cases_used=0,
              competitor_start=competitor_start,competitor_end=competitor_state(),
              gpu_wait_seconds=waited+sum(p["gpu_wait_seconds"] for p in parts),review=gate)
    save_json(root/"report.json",meta)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate",action="store_true")
    run(parser.parse_args().gate)
