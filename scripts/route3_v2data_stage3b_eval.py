"""Evaluate Stage 3b Query checkpoints with the fixed geometry and F2."""
import argparse
from collections import defaultdict
from html import escape
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch

import route3_ovg_data as ovg
import route3_v2data_stage1_visualize as gallery
from route3_ovg_footprint import query_net
from route3_v2data_B_batch import ROOT, write_json
from route3_v2data_B_plan import load
from route3_v2data_query_retrain import distribution
from route3_v2data_stage1_evaluate import FrameQueries, legacy_masks, new_regions, near_subject, p2_focus, pooled, region_errors
from route3_v2data_stage2_eval import bands, f2_for, frames, geometry_root, optical_target
from route3_v2data_stage2_train import QUERY, sample
from route3_v2data_stage3a import FIXED, gpu_setup
from route3_v2data_stage3a_eval import BASELINE, render
from route3_v2data_stage3b import OUT, hidden_cloud, load_geometry, optical_band, pack_frame, path_for, read_checkpoint, traced_render
from src.aperture_renderer import aperture_disk
from src.c1_bokehme import region_sse
from src.layered_aperture import merge_layers
from src.ovg_opaque_eval import subpixel_flags
from src.ovg_optics import fixed_visible_surfels, raster
from src.ovg_representation_eval import geometry_basis, representation
from route3_v2data_stage1_visualize_repr import layered_rays

START_GT = ROOT/"outputs/route3-v2data-stage3a-2026-11-24-02/evaluation/step-000000/ema/cases.json"
COHORTS = ("development","legacy","thin_structure","dev008")
REGIONS = ("near_in_focus","far_recoverable","boundary_2px")
FLOORS = dict(zero_visible=50.,zero_combined=60.)


def gt_reference_path(original=START_GT):
    corrected = original.with_name("cases-fp16-fit-corrected.json")
    pending = original.with_name("fp16-fit-correction.pending")
    if pending.is_file() and not corrected.is_file():
        print("STAGE3B_WAIT_GT_REFERENCE", corrected, flush=True)
        while pending.is_file() and not corrected.is_file():
            time.sleep(5)
    return corrected if corrected.is_file() else original


def slug(name):
    return "dev"+name.split("-")[1] if name.startswith("development-") else name.split("-")[2]


def display(tensor):
    return gallery.display(tensor.cpu().numpy())


def score(image,truth,mask):
    return region_sse(display(image),truth,mask)


def comparison(a,b):
    if a["pixels"] != b["pixels"]:
        raise ValueError("The two methods have different evaluated support")
    if not a["pixels"]:
        return None
    return b["psnr_db"]-a["psnr_db"]


def scores_by_scene(rows,region,key):
    per_scene = defaultdict(list)
    for row in rows:
        if region in row[key]:
            per_scene[row["scene"]].append(row[key][region])
    return {name:pooled(block) for name,block in per_scene.items()}


def deltas(off,on):
    if off.keys()!=on.keys():
        raise ValueError("Query evaluation changed the set of scenes")
    all_off,all_on = (pooled(table.values()) for table in (off,on))
    scenes = {scene:comparison(off[scene],on[scene]) for scene in off}
    return dict(pooled_db=comparison(all_off,all_on),
                off_db=all_off["psnr_db"],on_db=all_on["psnr_db"],
                per_scene=scenes,**distribution([v for v in scenes.values() if v is not None]))


def difference(base,new):
    values = [(name,new["per_scene"][name]-a) for name,a in base["per_scene"].items()
              if a is not None and new["per_scene"][name] is not None]
    aa,bb = base["pooled_db"],new["pooled_db"]
    return dict(pooled_db=bb-aa if aa is not None and bb is not None else None,
                per_scene=dict(values),**distribution([v for _,v in values]))


def summary(rows,start=None):
    result = {}
    initial = {(r["scene"],r["case"]):r for r in start} if start else {}
    for name in COHORTS:
        if name in ("development","legacy"):
            block = [r for r in rows if r["cohort"]==name and r["status"]=="evaluated"]
        else:
            ids = ("dev003","dev004","dev005") if name=="thin_structure" else ("dev008",)
            block = [r for r in rows if slug(r["scene"]) in ids and r["status"]=="evaluated"]
        metrics = {}
        for region in REGIONS:
            off,on,gt = (scores_by_scene(block,region,key) for key in ("off","on","gt"))
            effect = deltas(off,on)
            if initial:
                base = [initial[r["scene"],r["case"]] for r in block]
                old = deltas(scores_by_scene(base,region,"off"),scores_by_scene(base,region,"on"))
            else:
                old = effect
            ratios = {}
            if region!="boundary_2px" and off:
                oracle = deltas(off,gt)
                ratios = {s:effect["per_scene"][s]/a for s,a in oracle["per_scene"].items()
                          if a is not None and a>.1}
                pooled_ratio = (effect["pooled_db"]/oracle["pooled_db"]
                                if oracle["pooled_db"] is not None and oracle["pooled_db"]>.1 else None)
                metrics[region] = dict(gt_minus_f2=oracle,middle_ratio=float(np.median(list(ratios.values()))) if ratios else None,
                                       pooled_ratio=pooled_ratio,ratio_count=len(ratios),ratios=ratios)
            metrics.setdefault(region,{})
            metrics[region].update(query_minus_f2=effect,query_vs_start=difference(old,effect))
        zero = {}
        for key,floor in FLOORS.items():
            active = [pooled([r[key]]) for r in block if r[key]["pixels"]]
            passed = sum(m["psnr_db"]>=floor for m in active)
            zero[key] = dict(passed=passed,scored=len(active),floor=floor,
                             all_passed=bool(active) and passed==len(active),**pooled(active))
        far = [r["rays"] for r in block if r["focus_index"]==1]
        rays,hits = (sum(r[k] for r in far) for k in ("total_region_rays","effective_hidden_rays"))
        result[name] = dict(metrics=metrics,zero=zero,hidden_hits=hits,far_rays=rays,
                            far_hidden_fraction=hits/rays if rays else None,scored_cases=len(block))
    return result


def adoption(start,final):
    a,b = (r["summary"]["development"] for r in (start,final))
    near = b["metrics"]["near_in_focus"]["query_vs_start"]["pooled_db"]
    far = b["metrics"]["far_recoverable"]["query_vs_start"]["median_db"]
    base,now = a["far_hidden_fraction"],b["far_hidden_fraction"]
    checks = dict(far=far is not None and far>=0,near=near is not None and near>=-.1,
                  zero_visible=b["zero"]["zero_visible"]["all_passed"],
                  zero_combined=b["zero"]["zero_combined"]["all_passed"],
                  far_hidden=base is not None and now is not None and now>=.8*base)
    return dict(selected_step=final["step"] if all(checks.values()) else 0,checks=checks,
                near_pooled_change_db=near,far_paired_median_db=far,far_hidden_start=base,far_hidden_now=now,
                interpretation="基本持平" if far is not None and abs(far)<.1 else None)


def create_panel(output, tag, inputs, truth, rendered, baseline, maps, mask, row):
    whole = dict(input=inputs,GT=truth,start=baseline,updated=rendered)
    box = gallery.largest_error_crop(rendered,truth,mask)
    tile,gap = 224,10
    fig = Image.new("RGB",(4*(tile+gap),2*(tile+gap)+82),(18,23,31))
    draw = ImageDraw.Draw(fig)
    draw.text((10,7),tag,font=gallery.font(20),fill="white")
    draw.text((10,33),f'P2 {row["focus_m"]:.3f} m · {row["phase"]}',font=gallery.font(15),fill="white")
    labels = ("输入","Cycles 参考","起点 Query 放大","当前 Query 放大")
    for i,((key,img),label) in enumerate(zip(whole.items(),labels)):
        x = i*(tile+gap)
        view = gallery.rgb_image(img)
        if i>=2:
            view = view.crop(box)
        else:
            ImageDraw.Draw(view).rectangle(tuple(box),outline=(250,151,99),width=2)
        fig.paste(view.resize((tile,tile)),(x,73))
        draw.text((x+3,52),label,font=gallery.font(14),fill="white")
    mask_img = gallery.scalar_image(mask.astype(np.float32),1.,cv2.COLORMAP_BONE)
    heat = (("目标区域",mask_img),("隐藏层实际权重 · 0–1",gallery.scalar_image(maps["hidden"],1.,cv2.COLORMAP_INFERNO)),
            ("F2 兜底权重 · 0–1",gallery.scalar_image(maps["f2"],1.,cv2.COLORMAP_INFERNO)),
            ("当前光学误差 · 0–0.25",gallery.scalar_image(np.abs(rendered-truth).mean(-1),.25,cv2.COLORMAP_INFERNO)))
    y = tile+gap+75
    for i,(label,image) in enumerate(heat):
        x=i*(tile+gap)
        draw.text((x+3,y-8),label,font=gallery.font(14),fill="white")
        fig.paste(image.resize((tile,tile)),(x,y+13))
    path = output/"panels"/f"{tag}.jpg"
    path.parent.mkdir(exist_ok=True)
    fig.save(path,quality=93)
    return str(path.relative_to(output))


def panel_observation(row,base):
    region = REGIONS[row["focus_index"]]
    result = pooled([row["on"][region]])
    old = pooled([base["on"][region]])
    delta = comparison(old,result)
    rays = row["rays"]
    share = rays["effective_hidden_rays"]/max(rays["total_region_rays"],1)
    if delta is None:
        return "该帧的目标区域无有效评分像素，面板保留光线分布。"
    return f"目标区域相对起点 {delta:+.3f} dB，{share:.2%} 光线中第二遍实际权重达到 0.001。"


def variant_state(visible,depth,color,K,variant):
    if variant=="V0":
        return visible,None
    valid = torch.isfinite(depth)&(depth>0)
    inv = depth.reciprocal()
    links = []
    for a,b in ((np.s_[:,:-1],np.s_[:,1:]),(np.s_[:-1,:],np.s_[1:,:])):
        links.append(valid[a]&valid[b]&((inv[a]-inv[b]).abs()<.03*torch.minimum(inv[a],inv[b])))
    basis = geometry_basis(depth,K,links)
    layer = representation(visible,depth,color,K,valid,links,"V3",basis)
    if variant=="V3":
        return layer,None
    base,fill = layer
    closure = ((depth,valid,subpixel_flags(valid,valid,5),len(base.means)),
               (depth*1.005,basis[3]&valid,torch.ones(len(fill.means),device=K.device,dtype=torch.bool),len(fill.means)))
    return layer,closure


def variant_image(layer,closure,hidden,K,focus,offsets,fill):
    chunks = []
    for a,b in zip(offsets.split(4),fill.split(4) if fill is not None else [None]*((len(offsets)+3)//4)):
        chunks.append(layered_rays(layer,hidden,K,focus,a,b,closure=closure).clamp(0,1).sum(0))
    return sum(chunks)/len(offsets)


@torch.inference_mode()
def evaluate(run,step,weights,*,variant="V0",smoke=False,joint=None):
    gt_path = gt_reference_path()
    gpu_setup(dict(training=dict(max_process_gpu_gib=7)))
    began = time.monotonic()
    directory = "evaluation-smoke" if smoke else "evaluation"
    output = run/directory/f"step-{step:06d}"/(weights if variant=="V0" else variant)
    output.mkdir(parents=True,exist_ok=True)
    path = output/("joint-report.json" if joint is not None else "report.json")
    if path.is_file():
        return load(path)
    geometry = load_geometry()
    query = query_net(QUERY)
    if step:
        saved=read_checkpoint(path_for(run,step))[weights]
        if joint is None:
            query.load_state_dict(saved)
        else:
            joint.restore(geometry,query,saved)
    initial_path = run/"evaluation/step-000000/ema"
    starting = load(initial_path/"report.json")
    if step and not starting and not smoke:
        raise ValueError("Register the stage 3b starting evaluation first")
    prior = {(r["scene"],r["case"]):r for r in starting["rows"]} if starting else {}
    gt = load(gt_path)
    archived = {(r["scene"],r["case"]):r for v in gt.values() for r in v}
    legacy_plans = {(p["name"],p["frame"]):p for p in gallery.planned_frames() if p["unit"] is None}
    rows,panels = [],[]
    previous_name = None
    for cohort,name,frame,unit,entry,cases in frames():
        ident = slug(name)
        if smoke and (ident!="dev005" or frame!=30):
            continue
        if name != previous_name:
            plan = dict(unit=unit) if unit else legacy_plans[name,frame]
            views,memory = gallery.scene_views(plan)
            previous_name = name
        item = ("B",name,ovg.TRAIN_FRAMES.index(frame))
        x,y = sample(item)
        pred = geometry(x,protect_residual=True,render_heads=False)
        depth = pred["depth1"][0]
        z = depth.cpu().numpy()
        K = x["K"][0]
        k = K.cpu().numpy()
        ref = ovg.b_edge_reference(item,y)
        if joint is not None:
            joint.frame(cohort,name,frame,z,x,y,ref)
        focuses = [p2_focus(z,near_subject(ref["gt"],ref["valid"],k,c["focus_m"])) for c in cases]
        active = [optical_band(z,K,f,c["fstop"]) if f is not None else None for c,f in zip(cases,focuses)]
        if any(m is not None for m in active):
            packed = pack_frame(views,frame,z,[m for m in active if m is not None],memory)
            store = FrameQueries(views,packed)
        visible = fixed_visible_surfels(pred,x)
        layer,closure = variant_state(visible,depth,x["rgb"][0].permute(1,2,0),K,variant)
        full = np.ones_like(z,bool)
        pinhole = display(x["rgb"][0].permute(1,2,0))
        edge = bands(ref,y)[0]
        center = K.new_zeros((1,2))
        if variant=="V0":
            zero_visible = score(raster(visible,K,1.4,center,512,512)[0],pinhole,full)
        else:
            zero_visible = score(variant_image(layer,closure,None,K,1.4,center,None),pinhole,full)
        for case,focus,band in zip(cases,focuses,active):
            idx,stop,tag = case["focus_index"],case["fstop"],case["id"]
            row = dict(cohort=cohort,scene=name,frame=frame,case=tag,focus_index=idx,phase=f"{step} {weights} {variant}")
            if focus is None:
                rows.append(dict(row,status="no_valid_nominal_subject"))
                continue
            cached = archived[name,tag]
            if joint is None and abs(focus-cached["focus_m"])>1e-6:
                raise ValueError(f"3a reference has a different P2 focus for {name}/{tag}")
            regions = (new_regions(unit,case,ref["gt"],ref["valid"],k) if unit else
                       legacy_masks(geometry_root(name),case))
            area = regions["in_focus"] if idx==0 else regions["recoverable"]
            metric = REGIONS[idx]
            truth = optical_target(unit,name,case)
            raw = dict(case,focus_m=focus)
            hidden = hidden_cloud(query,store,dict(frame=frame,input="native",case=raw),K,depth,band)
            offsets = aperture_disk(64,.05/(2*stop),device=K.device,dtype=K.dtype)
            fill = f2_for(unit,entry,frame,x,focus,offsets)
            zfill = f2_for(unit,entry,frame,x,focus,center)
            if variant=="V0":
                on,maps,rays = traced_render(visible,hidden,K,focus,offsets,fill,(512,512),area)
                off = render(visible,K,focus,offsets,fill)
                zero = raster(merge_layers(visible,hidden),K,focus,center,512,512,zfill)[0]
            else:
                on = variant_image(layer,closure,hidden,K,focus,offsets,fill)
                off = display(variant_image(layer,closure,None,K,focus,offsets,fill))
                zero = variant_image(layer,closure,hidden,K,focus,center,zfill)
                rays = dict(total_region_rays=0,effective_hidden_rays=0)
            if joint is None:
                gt_case = cached["scores"]["stage3"]["metrics"][metric]
            else:
                gt_case = joint.oracle(unit,entry,case,ref,depth,K,focus,visible,offsets,fill,truth,area)
            row.update(status="evaluated",focus_m=focus,on={metric:score(on,truth,area),
                          "boundary_2px":score(on,truth,edge)},
                       off={metric:region_sse(off,truth,area),"boundary_2px":region_sse(off,truth,edge)},
                       gt={metric:gt_case},rays=rays,zero_visible=zero_visible,zero_combined=score(zero,pinhole,full))
            if joint is not None:
                joint.case(row,unit,case,area)
            rows.append(row)
            if (ident,frame,idx) in FIXED and variant=="V0":
                focal_tag = f"{ident}-{tag}"
                current = display(on)
                cache = output/"panel-images"
                cache.mkdir(exist_ok=True)
                np.save(cache/f"{focal_tag}.npy",current.astype(np.float16))
                old = np.load(initial_path/"panel-images"/f"{focal_tag}.npy").astype(np.float32) if starting else current
                file = create_panel(output,focal_tag,pinhole,truth,current,old,
                                    {n:v.cpu().numpy() for n,v in maps.items()},area,row)
                panels.append(dict(case=focal_tag,path=file,observation=panel_observation(row,prior.get((name,tag),row))))
                if joint is not None:
                    joint.panel(output,panels[-1],row,case,area,current,truth)
        print("STAGE3B_EVAL",step,weights,variant,name,frame,flush=True)
    payload = dict(step=step,weights=weights,variant=variant,rows=rows,summary=summary(rows,starting["rows"] if starting else None),
                   baseline=load(BASELINE)["summaries"]["f1.4"],panels=panels,seconds=time.monotonic()-began,
                   hidden_weight_threshold=.001,GT_reference=str(gt_path),test_cases_used=0)
    if joint is not None:
        joint.summarize(payload)
    if panels:
        (output/"观察清单.md").write_text("# 固定面板观察清单\n\n"+"\n".join(
            f'- {p["case"]}。{p["observation"]} [面板]({p["path"]})' for p in panels)+"\n")
        (output/"index.html").write_text('<!doctype html><meta charset="utf-8"><title>3b</title>\n'+"\n".join(
            f'<figure><img loading="lazy" width="900" src="{escape(p["path"])}"><figcaption>{escape(p["case"])}</figcaption></figure>' for p in panels))
    write_json(path,payload)
    return payload


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,default=OUT)
    parser.add_argument("--step",type=int,required=True)
    parser.add_argument("--weights",choices=("ema","model"),default="ema")
    parser.add_argument("--variant",choices=("V0","V3","V3+o"),default="V0")
    parser.add_argument("--smoke",action="store_true")
    args = parser.parse_args()
    evaluate(args.run,args.step,args.weights,variant=args.variant,smoke=args.smoke)
