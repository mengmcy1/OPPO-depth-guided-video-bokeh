"""Matched P2 development evaluation for Stage 3a geometry checkpoints."""
import argparse
from pathlib import Path
import time

import numpy as np
import torch

import route3_ovg_data as ovg
import route3_v2data_stage1_visualize as gallery
from route3_ldr_u1r import model as lpips_model
from route3_ovg_1b_optical_data import b_oracle, place
from route3_upper_bound_evaluate import score
from route3_v2data_baseline_dev3_extended_report import (
    REGIONS, group_regions, pooled, percentile,
)
from route3_v2data_B_batch import ROOT, sha256, write_json
from route3_v2data_B_plan import load
from route3_v2data_query_retrain import distribution
from route3_v2data_stage1_evaluate import (
    load_paired_checkpoint, near_subject, new_regions, p2_focus,
)
from route3_v2data_stage2_eval import (
    bands, f2_for, frames, legacy_masks, geometry_root, optical_target,
)
from route3_v2data_stage2_train import INITIAL, sample
from route3_v2data_stage3a import (
    OUT, Placement, checkpoint, checkpoint_model, configuration_for, gpu_setup, parameter_changes,
)
from route3_ovg_subpixel import save_npz
from src.aperture_renderer import aperture_disk
from src.c1_bokehme import region_sse
from src.layered_aperture import merge_layers
from src.ovg_edge_statistics import EdgeStatistics
from src.ovg_optics import fixed_visible_surfels, raster

BASELINE = ROOT/"03_实验记录/records/route3-v2data-stage3-2026-11-24-baselines.json"
METRICS = (*REGIONS,"zero_visible","zero_combined")
GIB = 2**30


def render(cloud,K,focus,offsets,filler=None):
    total = None
    for first in range(0,len(offsets),8):
        subfill = None if filler is None else filler[first:first+8]
        image = raster(cloud,K,focus,offsets[first:first+8],512,512,subfill).clamp(0,1).sum(0)
        total = image if total is None else total+image
    return gallery.display((total/len(offsets)).cpu().numpy())


def hidden_layer(placer,unit,name,frame,case,ref,pred,K):
    if unit is None:
        parents, gt, background = b_oracle(name,frame)
        return place(parents,gt,background,pred.detach(),K,512)
    key = f'f{frame:03d}-focus{case["focus_index"]}'
    spec = dict(tiers="B",unit=unit,frame=frame,focus=case["focus_m"],fstop=case["fstop"],
                truth=unit/"labels"/(case["id"]+"-optics.npz"),label=unit/"labels"/(key+"-K16.npz"))
    return placer(spec,ref,pred.detach(),K)


def evaluate_frame(net,placer,unit,name,frame,entry,cases,loss_model,edges,output):
    item = ("B",name,ovg.TRAIN_FRAMES.index(frame))
    x,y = sample(item)
    pred = net(x,protect_residual=True,render_heads=False)
    ref = ovg.b_edge_reference(item,y)
    depth = pred["depth1"][0].cpu().numpy()
    gt = y["depth1"][0].cpu().numpy().astype(np.float32)
    valid = ref["valid"] & np.isfinite(gt) & (gt>0)
    if not (np.isfinite(depth).all() and (depth>0).all()):
        raise FloatingPointError(f"Invalid depth from 3a for {name} f{frame:03d}")
    errors = np.zeros_like(depth,dtype=np.float32)
    errors[valid] = np.abs(1/depth[valid].astype(np.float64)-1/gt[valid].astype(np.float64))
    inverse = dict(pixels=int(valid.sum()),sse=float(np.square(errors[valid].astype(np.float64)).sum()))
    edges.add(name,"3a",depth,ref,valid)
    K = x["K"][0]
    k = K.cpu().numpy()
    edge = bands(ref,y)[0]
    forehead = valid & np.isin(ref["object_id"],ref["foreground"])
    visible = fixed_visible_surfels(pred,x)
    center = K.new_zeros((1,2))
    pinhole = gallery.display(x["rgb"][0].permute(1,2,0).cpu().numpy())
    zero_visible = render(visible,K,1.4,center)
    full = np.ones(depth.shape,bool)
    zero = region_sse(zero_visible,pinhole,full)
    root = output/"cases"/name
    root.mkdir(parents=True,exist_ok=True)
    optical = aperture_disk(64,.05/(2*1.4),device=K.device,dtype=K.dtype)
    rows = []
    for case in cases:
        ident = case["id"]
        idx = case["focus_index"]
        subject = near_subject(ref["gt"],ref["valid"],k,case["focus_m"])
        focus = p2_focus(depth,subject)
        row = dict(scene=name,frame=frame,case=ident,focus_index=idx,fstop=case["fstop"],scores={},
                   status="no_valid_nominal_subject" if focus is None else "evaluated")
        if focus is None:
            rows.append(row)
            continue
        truth = optical_target(unit,name,case)
        m = new_regions(unit,case,ref["gt"],ref["valid"],k) if unit else legacy_masks(geometry_root(name),case)
        area = "near_in_focus" if idx==0 else "far_recoverable"
        masks = dict(full_image=full,boundary_2px=edge)
        masks[area] = m["in_focus"] if idx==0 else m["recoverable"]
        hidden,audit = hidden_layer(placer,unit,name,frame,case,ref,pred["depth1"][0],K)
        cloud = merge_layers(visible,hidden)
        fill = f2_for(unit,entry,frame,x,focus,optical)
        on = render(cloud,K,focus,optical,fill)
        off = render(visible,K,focus,optical,fill)
        zfill = f2_for(unit,entry,frame,x,focus,center)
        combined = render(cloud,K,focus,center,zfill)
        metrics = score(on,truth,masks,loss_model)
        metrics.update(zero_visible=zero,zero_combined=region_sse(combined,pinhole,full))
        row["scores"] = {"stage3":{"metrics":metrics}}
        # Separate fixed nominal-focus geometry guard from the visual map
        # made using this method's own P2 focus.
        nominal = case["focus_m"]
        scale = float(k[0,0])*.05/(2*case["fstop"])*nominal/(nominal-.05)
        coc = errors*scale
        guard = forehead & masks[area] if idx==0 else np.zeros_like(full)
        row.update(focus_m=focus,nominal_focus_m=nominal,gt_hidden=audit,
                   f2=region_sse(off,truth,masks[area]),
                   coc_guard=dict(pixels=int(guard.sum()),absolute_error_sum=float(coc[guard].astype(np.float64).sum())))
        path = root/(ident+".npz")
        save_npz(path,input=pinhole.astype(np.float16),truth=truth.astype(np.float16),
            current=on.astype(np.float16),f2=off.astype(np.float16),depth=depth.astype(np.float16),
            gt_depth=gt.astype(np.float16),fg=forehead,coc=coc.astype(np.float16),boundary=edge,
            near_or_far=masks[area],zero_error=np.abs(combined-pinhole).mean(-1).astype(np.float16),
            coc_foreground=coc[forehead])
        row["artifact"] = str(path.relative_to(output))
        rows.append(row)
    return rows,dict(scene=name,frame=frame,**inverse)


def match_metric(rows, base, metric):
    grouped = {}
    for r in rows:
        k=(r["scene"],r["case"])
        if not r["scores"] or metric not in r["scores"]["stage3"]["metrics"]:
            continue
        current=r["scores"]["stage3"]["metrics"][metric]
        before=base.get(k)
        if not before or not before["scores"]:
            continue
        original=before["scores"]["stage3"]["metrics"][metric]
        if not current["pixels"]:
            continue
        if original["pixels"]!=current["pixels"]:
            raise ValueError(f"Unmatched comparison support for {k}, {metric}")
        grouped.setdefault(r["scene"],[]).append((original,current))
    return pair_from_grouped(grouped)


def pair_from_grouped(grouped):
    scenes = {}
    for name,pairs in grouped.items():
        a,b=(pooled([r[i] for r in pairs]) for i in (0,1))
        scenes[name]=dict(before=a,after=b,delta_db=b["psnr"]-a["psnr"])
    pair = list(grouped.values())
    a,b=(pooled([r[i] for group in pair for r in group]) for i in (0,1))
    stats = distribution([v["delta_db"] for v in scenes.values()])
    return dict(scenes=scenes,before=a,after=b,pooled_delta_db=b["psnr"]-a["psnr"] if pair else None,
                **stats)


def pair_vs_A(current, external, metric):
    before = external["A"][metric]["per_scene"]
    now = current[metric]["per_scene"]
    pairs = {}
    for scene in sorted(set(now)&set(before)):
        a,b = before[scene],now[scene]
        if a["pixels"]!=b["pixels"]:
            raise ValueError(f"Archived A support differs for {scene} {metric}")
        pairs[scene] = [(a,b)]
    return pair_from_grouped(pairs)


def guard_report(frames,rows,edges):
    n=sum(r["pixels"] for r in frames)
    dn=sum(r["coc_guard"]["pixels"] for r in rows if "coc_guard" in r)
    near=[v["absolute_p95_m_inv"] for v in edges.rows()
          if v["relation"]=="same_foreground_surface" and v["distance_band"]=="0_to_2_px"
          and v["absolute_p95_m_inv"] is not None]
    crossing=np.concatenate([a for k,b in edges.samples.items() if k[2]=="GT_occlusion_crossing" for a in b])
    return dict(inverse_depth_rmse=float(np.sqrt(sum(r["sse"] for r in frames)/n)),
                inverse_depth_pixels=n,same_surface_step=float(np.median(near)),
                near_subject_coc_mae=sum(r["coc_guard"]["absolute_error_sum"] for r in rows if "coc_guard" in r)/dn,
                near_coc_pixels=dn,GT_occlusion_crossing_abs_p95=float(np.quantile(np.abs(crossing),.95)),
                GT_occlusion_crossing_signed_median=float(np.median(crossing)), crossing_pairs=len(crossing))


def subgroup(output,rows):
    regions={k:group_regions(rows,"stage3",k) for k in ("near_in_focus","far_recoverable")}
    values=[]
    for r in rows:
        if r.get("artifact"):
            with np.load(output/r["artifact"]) as arr:
                values.append(arr["coc_foreground"])
    data=np.concatenate(values)
    return dict(regions=regions,foreground_coc_px=dict(pixels=int(data.size),**percentile(data)))


@torch.inference_mode()
def evaluate(run,step,weights,*,smoke=False):
    configuration=configuration_for(run)
    gpu_setup(configuration)
    torch.cuda.set_per_process_memory_fraction(8*GIB/torch.cuda.get_device_properties(0).total_memory,0)
    began=time.monotonic()
    output=run/("evaluation-smoke" if smoke else "evaluation")/f"step-{step:06d}"/weights
    output.mkdir(parents=True,exist_ok=True)
    if (output/"report.json").is_file():
        return load(output/"report.json")
    src=checkpoint(run,step) if step else INITIAL
    net=(checkpoint_model(src,weights) if step else load_paired_checkpoint(src,"ema")).eval()
    place=Placement(run/"dev-hidden-parents")
    model=lpips_model()
    torch.set_num_threads(3)
    rows={"development":[],"legacy":[]}
    geometry={"development":[],"legacy":[]}
    edge={key:EdgeStatistics() for key in rows}
    for kind,name,frame,unit,entry,cases in frames():
        if smoke and (name.split("-")[1:2]!=["001"] or frame!=30) and ("122104" not in name or frame!=30):
            continue
        r,g=evaluate_frame(net,place,unit,name,frame,entry,cases,model,edge[kind],output)
        rows[kind].extend(r)
        geometry[kind].append(g)
        print("STAGE3A_EVAL",step,weights,kind,name,frame,flush=True)
    baseline=load(BASELINE)
    baseline_path=Path(baseline["artifacts"]["summary"])
    if sha256(baseline_path)!=baseline["artifacts"]["summary_sha256"]:
        raise ValueError("External baseline archive changed after registration")
    external=load(baseline_path)
    start_dir=output.parents[1]/"step-000000/ema"
    start=load(start_dir/"cases.json") if step else rows
    if not start:
        raise FileNotFoundError("Evaluate step 0 with the same GT hidden substitute first")
    report={}
    for kind in rows:
        active=rows[kind]
        paired_against={(r["scene"],r["case"]):r for r in start[kind]}
        metrics={k:group_regions(active,"stage3",k) for k in METRICS}
        near={}
        for r in active:
            if r["scores"] and r["focus_index"]==0 and r["f2"]["pixels"]:
                uses=r["scores"]["stage3"]["metrics"]["near_in_focus"]
                near.setdefault(r["scene"],[]).append((r["f2"],uses))
        report[kind]=dict(metrics=metrics,geometry=guard_report(geometry[kind],active,edge[kind]),
            paired_vs_start={k:match_metric(active,paired_against,k) for k in METRICS},
            gt_minus_f2_near=pair_from_grouped(near),
            unscored=[dict(scene=r["scene"],case=r["case"],reason=r["status"]) for r in active if not r["scores"]])
    development=report["development"]
    development["groups"]={tag:subgroup(output,[r for r in rows["development"] if r["scene"].split("-")[1] in ids])
                            for tag,ids in (("thin_structure",{"003","004","005"}),("development-008",{"008"}))
                            if any(r["scene"].split("-")[1] in ids for r in rows["development"])}
    geometry_now=development["geometry"]
    guards=configuration["evaluation"]["relative_geometry_limits"]
    start_report = load(start_dir/"report.json")["report"]["development"] if step else development
    previous=start_report["geometry"]
    development["geometry_limits"]={k:dict(value=geometry_now[k],initial=previous[k],allowed_growth=v,
                                         passed=geometry_now[k]<=previous[k]*(1+v)) for k,v in guards.items()}
    for name,group in development["groups"].items():
        group["start_foreground_coc_px"] = start_report.get("groups",{}).get(name,{}).get("foreground_coc_px")
    if not smoke:
        development["paired_vs_A"]={k:pair_vs_A(development["metrics"],external["summaries"]["f1.4"],k)
                                    for k in ("near_in_focus","far_recoverable")}
    payload=dict(step=step,weights=weights,checkpoint=str(src),sha256=sha256(src),
        report=report,external_baselines=baseline["summaries"]["f1.4"],legacy_baselines=baseline["legacy"],
        optical_protocol="P2 f/1.4, GT hidden substitute on and the identical F2 branch off, 64 rays clipped in linear RGB before averaging",
        test_cases_used=0,seconds=time.monotonic()-began)
    if step:
        payload["parameter_changes"] = parameter_changes(src,weights,configuration)
    write_json(output/"cases.json",rows)
    if not smoke:
        from route3_v2data_stage3a_panels import panels
        payload["panels"]=panels(output,start_dir if step else output,rows,development,
                                 configuration["evaluation"]["fixed_cases"])
    write_json(output/"report.json",payload)
    return payload


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=Path,default=OUT)
    p.add_argument("--step",type=int,required=True)
    p.add_argument("--weights",choices=("ema","model"),default="ema")
    p.add_argument("--smoke",action="store_true")
    args=p.parse_args()
    evaluate(args.run.resolve(),args.step,args.weights,smoke=args.smoke)
