"""Paired Stage 1 evaluation on 15 formal development scenes and the legacy three."""
import argparse
from collections import defaultdict
import csv
from pathlib import Path
import os

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR","1")
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
import torch

import route3_ovg_data as legacy
from route3_g3_r14_train import learned_hidden
from route3_ovg_1b_common import CONFIG,SOURCE
from route3_ovg_1b_optical_data import background_samples,geometry_root
from route3_ovg_1b_train import load_model
from route3_ovg_1c import OUT as ONE_C,aligned_sample
from route3_ovg_1c_evaluate import case_frames
from route3_ovg_footprint import query_net
from route3_s2a_data import query_input_512,query_memory
from route3_ovg_refocus_query import original_views
from route3_s2a_evaluate import masks as legacy_masks
from route3_s2a_learn import Store
from route3_upper_bound_evaluate import display,exr
from route3_v2data_B_batch import FORMAL_DATA,GIB,ROOT,sha256,write_json
from route3_v2data_B_f2 import build_frame
from route3_v2data_B_inputs import library_unit_paths,observation
from route3_v2data_B_plan import load
from route3_v2data_dryrun import stage_objective
from route3_v2data_loader import TrainingData,window_inventory
from route3_v2data_G_priors import inventory as g_inventory
from route3_v2data_R_priors import inventory as r_inventory
from route3_v2data_optics import f2_entry,observed_background,ray_fields
from src.aperture_renderer import aperture_disk
from src.dual_layer import depth_band
from src.footprint_order import apply_footprint_order
from src.layered_aperture import merge_layers
from src.ovg_edge_statistics import EdgeStatistics
from src.ovg_optics import fixed_visible_surfels,raster


FRAMES=(1,30,60)
LEGACY_CHECKPOINT=ONE_C/"train/aligned/checkpoints/step-005000.pt"
QUERY=SOURCE/"V1/retrained/step-020000.pt"
METRICS=("near_in_focus","far_recoverable")
LEGACY_QUERY_PAIRS=Path("03_实验记录/records/route3-ovg-stage2-2026-11-19-S3-query-pairs.csv")
QUERY_PAIR_PROTOCOL=(
    "P2 f/1.4。每个 case 固定权重、可见层、实际焦点、F2 背景、光圈偏移和评分掩码，"
    "关 Query 只移除第二遍环带隐藏层。阶段 1 使用固定可见层，未启用第一遍边界拆分，"
    "与 §13.16 的 1c 起点关法相同。§13.16 的 2a/2b 配对还保留第一遍边界背景。"
    "原三场景沿用旧区域掩码和 F2，按当前检查点重新定焦并构建环带 Query；新开发集使用 v2 LDR 参考。"
    "按开、关各自的 RGB SSE 和像素数合并求 PSNR，再相减；仅作诊断。"
)


def development_cases(unit):
    audit=load(unit/"labels/radiance-audit.json")
    cases=[]
    for row in audit["optical"]:
        frame,focus,_=row["id"].split("-")
        cases.append(dict(id=row["id"],frame=int(frame[1:]),focus_index=int(focus[-1]),
                          focus_m=row["focus_m"],fstop=row["fstop"]))
    if len(cases)!=12:
        raise ValueError(f"development optical grid is incomplete for {unit.name}")
    return cases


def dev_cache(unit, cases, f2_root, frames=FRAMES):
    if torch.is_inference_mode_enabled() or not torch.is_grad_enabled():
        raise RuntimeError("Development F2 preparation needs autograd for the A-color fit")
    for frame in frames:
        # The derived F2 cache consumes only the observation-derived inputs.
        if not (f2_entry(unit,frame,f2_root)/"manifest.json").is_file():
            stops=[(case["focus_m"],case["fstop"]) for case in cases if case["frame"]==frame]
            build_frame(unit,frame,stops,directory=f2_root)
    return f2_root


def view_from_observation(raw,frame):
    z=np.where((raw["confidence"][0]>0)&np.isfinite(raw["depth"][0])&(raw["depth"][0]>0),
               raw["depth"][0],1e4).astype(np.float32)
    rho=1/np.maximum(z,1e-6)
    kernel=np.ones((3,3),np.uint8)
    edge=cv2.dilate(((cv2.dilate(rho,kernel)-cv2.erode(rho,kernel))>.03*cv2.dilate(rho,kernel)).astype(np.uint8),
                    np.ones((7,7),np.uint8))>0
    # v2 stores confidence but does not retain RAFT evidence rasters. The existing
    # Query feature encoder handles missing evidence through its explicit NaN flags.
    missing=np.full(z.shape,np.nan,np.float32)
    return dict(frame=frame,depth=z,K=raw["K"],c2w=raw["pose"],
                rgb_linear=raw["rgb"],rgb_srgb=display(raw["rgb"]),confidence=raw["confidence"][0],
                edge=edge,edge_distance=distance_transform_edt(~edge).astype(np.float32),
                reprojection=missing,FB=missing,angle_radians=missing)


class FrameQueries(Store):
    """Use the existing frozen Query forward path with in-memory dev observations."""
    def __init__(self,views,conditioned):
        self.features=conditioned
        self.observed=dict(
            images=torch.as_tensor(np.stack([row["rgb_linear"] for row in views]),device="cuda").permute(0,3,1,2).contiguous(),
            K=torch.as_tensor(views[0]["K"],device="cuda"),
            c2w=torch.as_tensor(np.stack([row["c2w"] for row in views]),device="cuda"),
        )

    def inputs(self,path):
        return self.features

    def context(self,manifest):
        return self.observed

    def active(self,input_path,focus,fstop):
        d=self.features
        band=depth_band(d["target_depth"],d["target_K"][0,0],focus,.05/(2*fstop))[0]
        return np.flatnonzero(band.ravel()[d["query_pixel"]])


def region_errors(image,target,mask):
    rgb=display(image.detach().cpu().numpy())
    error=rgb[mask].astype(np.float64)-target[mask].astype(np.float64)
    return dict(sse=float(np.square(error).sum()),pixels=int(mask.sum()))


def pooled(values):
    pixels=sum(v["pixels"] for v in values)
    sse=sum(v["sse"] for v in values)
    return dict(sse=sse,pixels=pixels,psnr_db=-10*np.log10(max(sse/(3*pixels),1e-12)) if pixels else None)


def paired_psnr(on, off):
    query_on, query_off = pooled(on), pooled(off)
    if query_on["pixels"] != query_off["pixels"]:
        raise ValueError("Query pair uses different pixel support")
    delta = query_on["psnr_db"] - query_off["psnr_db"] if query_on["pixels"] else None
    return dict(on=query_on,off=query_off,delta_db=delta)


def near_subject(gt,valid,K,nominal):
    return valid&(gt>0)&(K[0,0]*.05/2.8*np.abs(1/np.maximum(gt,1e-6)-1/nominal)<=1)


def p2_focus(depth,subject):
    target=subject&np.isfinite(depth)&(depth>0)
    return float(np.median(depth[target])) if target.any() else None


def new_regions(unit,case,gt,valid,K):
    focus,stop=case["focus_m"],case["fstop"]
    band,_,coc=depth_band(np.where(valid,gt,1e4),K[0,0],focus,.05/(2*stop))
    key=unit/"labels"/f'f{case["frame"]:03d}-focus{case["focus_index"]}-K16.npz'
    traced=ray_fields(dict(tiers="B",unit=unit,frame=case["frame"],focus=focus,fstop=stop,
                           label=key if stop==1.4 else None))
    return dict(in_focus=valid&(coc<=1),recoverable=band&np.any(traced["recoverable"],axis=0))


def frame_scores(net,query,views,item,cases,scene,model_label,edge,target_for,background_for,regions_for,memory=None):
    inputs,y=aligned_sample(item,"cuda")
    pred=net(inputs,protect_residual=True,render_heads=False)
    reference=legacy.b_edge_reference(item,y)
    valid=reference["valid"]
    gt=reference["gt"]
    depth=pred["depth1"][0]
    depth_np=depth.cpu().numpy()
    edge.add(scene,model_label,depth_np,reference,valid)
    visible=fixed_visible_surfels(pred,inputs)
    K=inputs["K"][0]
    intrinsics=K.cpu().numpy()
    center=K.new_zeros((1,2))
    image=display(inputs["rgb"][0].permute(1,2,0).cpu().numpy())
    all_pixels=np.ones_like(valid)
    zero_visible=region_errors(raster(visible,K,1.4,center,512,512)[0],image,all_pixels)
    frame=legacy.TRAIN_FRAMES[item[2]]
    source=next(view for view in views if view["frame"]==frame)
    bands={}
    focuses={}
    for case in cases:
        focus=p2_focus(depth_np,near_subject(gt,valid,intrinsics,case["focus_m"]))
        if focus is None:
            continue
        focuses[case["id"]]=focus
        bands[case["id"]]=depth_band(depth_np,intrinsics[0,0],focus,.05/(2*case["fstop"]))[0]
    if not bands:
        return [dict(scene=scene,case=case["id"],frame=frame,status="no_valid_nominal_subject") for case in cases]
    union=np.logical_or.reduce(list(bands.values()))
    packed,_=query_input_512(views,dict(source,depth=depth_np),union,8,262144,memory)
    store=FrameQueries(views,packed)
    rows=[]
    for case in cases:
        ident=case["id"]
        row=dict(scene=scene,case=ident,frame=frame,focus_index=case["focus_index"],fstop=case["fstop"])
        if ident not in focuses:
            rows.append(dict(row,status="no_valid_nominal_subject"))
            continue
        focus=focuses[ident]
        hidden=learned_hidden(query,store,dict(frame=frame,input="current-frame",case=dict(case,focus_m=focus)),
                              dict(size=512),K,bands[ident])
        hidden,_=apply_footprint_order(hidden,depth,K)
        combined=merge_layers(visible,hidden)
        bg_center=background_for(inputs,case,focus,center)
        zero=region_errors(raster(combined,K,focus,center,512,512,bg_center)[0],image,all_pixels)
        offsets=aperture_disk(CONFIG["native_eval_rays"],.05/(2*case["fstop"]),device=K.device,dtype=K.dtype)
        target=display(target_for(case))
        regions=regions_for(case,gt,valid,intrinsics)
        rays_on,rays_off=[],[]
        for part in offsets.split(8):
            bg=background_for(inputs,case,focus,part)
            rays_on.append(raster(combined,K,focus,part,512,512,bg).sum(0))
            rays_off.append(raster(visible,K,focus,part,512,512,bg).sum(0))
        rgb_on=sum(rays_on)/len(offsets)
        rgb_off=sum(rays_off)/len(offsets)
        metric,mask = (("near_in_focus",regions["in_focus"]) if case["focus_index"]==0 else
                       ("far_recoverable",regions["recoverable"]))
        on=region_errors(rgb_on,target,mask)
        off=region_errors(rgb_off,target,mask)
        rows.append(dict(row,status="evaluated",actual_focus_m=focus,zero_visible=zero_visible,zero_combined=zero,
                         near_in_focus=on if metric=="near_in_focus" else None,
                         far_recoverable=on if metric=="far_recoverable" else None,
                         query_pair={metric:paired_psnr([on],[off])}))
    return rows


def dev_rows(net,query,units,label,edge,f2_root):
    result=[]
    for unit in units:
        cases=development_cases(unit)
        views=[view_from_observation(observation(unit,frame,aligned=True),frame) for frame in legacy.TRAIN_FRAMES]
        for frame in FRAMES:
            item=("B",unit.name,legacy.TRAIN_FRAMES.index(frame))
            subset=[c for c in cases if c["frame"]==frame and c["fstop"]==1.4]
            def target(case):
                return legacy.load(unit/"labels"/(case["id"]+"-optics.npz"),("rgb_linear",))["rgb_linear"].astype(np.float32)
            def bg(inputs,case,focus,offsets):
                return observed_background(dict(tiers="B",unit=unit,frame=case["frame"],f2_root=f2_root),inputs,focus,offsets)
            result.extend(frame_scores(net,query,views,item,subset,unit.name,label,edge,target,bg,
                                       lambda c,g,v,k:new_regions(unit,c,g,v,k)))
        print("stage1 dev scoring",label,unit.name,flush=True)
    return result


def old_rows(net,query,label,edge):
    result=[]
    sources={}
    memories={}
    for name,frame,entry,_,source,_,manifest,original,_ in case_frames():
        if frame not in FRAMES:
            continue
        if name not in sources:
            sources[name]=original_views(source,manifest)
            memories[name]=query_memory(source,manifest["frames"])
        cases=[dict(entry["case"],focus_m=entry["case"]["nominal_focus_m"],
                    focus_index=0 if entry["case"]["nominal_focus_m"]==1.4 else 1)
               for entry in original if entry["case"]["protocol"]=="P2" and entry["case"]["fstop"]==1.4]
        def target(case):
            return exr(geometry_root(name)/"dof"/(case["id"]+".exr"))
        def bg(inputs,case,focus,offsets):
            return background_samples(dict(tiers="B",entry=entry),inputs["K"][0],focus,offsets,(512,512))
        result.extend(frame_scores(net,query,sources[name],("B",name,legacy.TRAIN_FRAMES.index(frame)),
                                   cases,name,label,edge,target,bg,
                                   lambda case,*_:legacy_masks(geometry_root(name),case),memories[name]))
    return result


def summary(rows):
    active=[row for row in rows if row["status"]=="evaluated"]
    out={}
    for metric in (*METRICS,"zero_visible","zero_combined"):
        values=[r[metric] for r in active if r[metric] and r[metric]["pixels"]]
        measure=pooled(values)
        measure["scored_cases"]=len(values)
        if metric.startswith("zero_"):
            floor=50.0 if metric=="zero_visible" else 60.0
            measure.update(floor_db=floor,passed_cases=sum(pooled([v])["psnr_db"]>=floor for v in values),
                           all_passed=bool(values and len(active)==len(rows) and
                                           all(pooled([v])["psnr_db"]>=floor for v in values)))
        else:
            by_scene=defaultdict(list)
            for r in active:
                if r[metric] and r[metric]["pixels"]:
                    by_scene[r["scene"]].append(r[metric])
            psnr={scene:pooled(block)["psnr_db"] for scene,block in by_scene.items()}
            measure.update(per_scene=psnr,scene_standard_deviation_db=float(np.std(list(psnr.values()),ddof=1)) if len(psnr)>1 else None)
        out[metric]=measure
    pairs={}
    for metric in METRICS:
        by_scene=defaultdict(list)
        for row in active:
            pair=row.get("query_pair",{}).get(metric)
            if pair and pair["on"]["pixels"]:
                by_scene[row["scene"]].append(pair)
        if by_scene:
            def merge(block):
                return dict(paired_psnr([p["on"] for p in block],[p["off"] for p in block]),
                            scored_cases=len(block))
            pairs[metric]=dict(merge([p for values in by_scene.values() for p in values]),
                               per_scene={scene:merge(block) for scene,block in by_scene.items()})
    if pairs:
        out["query_pair"]=pairs
    return out


class DevelopmentData(TrainingData):
    def __init__(self,units):
        super().__init__()
        self.b=dict(self.b,**{unit.name:unit for unit in units})
        self.dev={
            "B":[("B",unit.name,idx) for unit in units for idx in range(len(legacy.TRAIN_FRAMES))],
            "C":list(window_inventory("dev")),"G":list(g_inventory("development")),"R":list(r_inventory("development")),
        }
        for tier, rows in self.dev.items():
            self.allowed[tier].update(rows)
        self.dev_c={item[1] for item in self.dev["C"]}

    def load(self,item,device):
        if item[0]=="C" and item[1] in self.dev_c:
            x,y=legacy.sample(item,device)
            return x,{k:v.float() if v.is_floating_point() else v for k,v in y.items()}
        return super().load(item,device)


def geometry_loss_gaps(net, inventory, phase, weight):
    result={}
    for index,tier in enumerate(inventory.dev):
        if tier=="R" and phase=="1a":
            continue
        result[tier]={}
        for role,pool in (("train",inventory.items[tier]),("dev",inventory.dev[tier])):
            if not pool:
                result[tier][role]=dict(count=0)
                continue
            rng=np.random.default_rng([CONFIG["seed"],index])
            pick=rng.choice(len(pool),min(16,len(pool)),replace=False)
            rows=[]
            for n,i in enumerate(pick):
                gen=torch.Generator(device="cuda").manual_seed(CONFIG["seed"]+n)
                terms,_,weights=stage_objective(net,inventory,pool[i],phase,gen)
                total=sum(weights[key]*value for key,value in terms.items() if key!="same_surface")
                surface=terms.get("same_surface")
                value=float(surface) if surface is not None else None
                if value is not None:
                    total=total+weight*surface
                rows.append(dict(total=float(total),same_surface=value))
            result[tier][role]=dict(count=len(rows),mean_loss=float(np.mean([r["total"] for r in rows])),
                                    mean_same_surface=float(np.mean([r["same_surface"] for r in rows])) if rows[0]["same_surface"] is not None else None)
        a,b=(result[tier][role].get("mean_loss") for role in ("train","dev"))
        result[tier]["dev_minus_train"]=b-a if a is not None and b is not None else None
    return result


def load_paired_checkpoint(checkpoint,key,device="cuda"):
    net=load_model(checkpoint,"model",device=device)
    if key=="ema":
        weights=torch.load(checkpoint,weights_only=True,map_location="cpu")["ema"]
        expected={name for name,p in net.named_parameters() if p.requires_grad and not name.startswith("footprint.")}
        if set(weights)!=expected:
            raise ValueError("the stage 1 EMA parameters differ from the paired training state")
        net.load_state_dict(weights,strict=False)
    return net


def development_units(smoke=False):
    units=library_unit_paths("dev")
    if len(units)!=15:
        raise ValueError("the paired Stage 1 evaluation needs the 15 registered development scenes")
    return units[:1] if smoke else units


def prepare_development(units,f2_root):
    f2_root.mkdir(parents=True,exist_ok=True)
    for unit in units:
        dev_cache(unit,development_cases(unit),f2_root)


def evaluate(checkpoint,output,phase,branch,weight,*,f2_root,smoke=False,versions=("ema","model")):
    output.mkdir(parents=True,exist_ok=True)
    regular=development_units(smoke)
    prepare_development(regular,f2_root)
    return score_checkpoint(checkpoint,output,phase,branch,weight,regular,f2_root,smoke,versions)


@torch.inference_mode()
def score_checkpoint(checkpoint,output,phase,branch,weight,regular,f2_root,smoke,versions):
    query=query_net(QUERY)
    results={}
    for key in versions:
        path=output/(key+".json")
        previous=load(path)
        if previous.get("query_pair_protocol")==QUERY_PAIR_PROTOCOL:
            results[key]=previous
            continue
        net=load_paired_checkpoint(checkpoint,key).eval()
        edge=EdgeStatistics()
        dev=dev_rows(net,query,regular,key,edge,f2_root)
        old=[] if smoke else old_rows(net,query,key,edge)
        all_edges=list(edge.rows())
        near=[row["absolute_p95_m_inv"] for row in all_edges
              if row["scene"] in {q.name for q in regular} and row["relation"]=="same_foreground_surface"
              and row["distance_band"]=="0_to_2_px" and row["absolute_p95_m_inv"] is not None]
        gaps={} if smoke else geometry_loss_gaps(net,DevelopmentData(regular),phase,weight)
        results[key]=dict(branch=branch,phase=phase,checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint),
                          data_gaps=gaps,development=summary(dev),legacy=summary(old),
                          development_rows=dev,legacy_rows=old,edge_statistics=all_edges,
                          same_surface_step_m_inv=float(np.median(near)) if near else None,
                          same_surface_metric="median of per-scene/direction same-surface |inverse-depth excess| p95, 0–2 px",
                          evaluation="P2 actual frozen Query, f/1.4, LDR v2 reference; legacy rows separate",
                          query_pair_protocol=QUERY_PAIR_PROTOCOL,
                          legacy_note="B K16 L_occ now uses native 512 instead of 2x2 agreement pooling")
        write_json(path,results[key])
        del net
        torch.cuda.empty_cache()
    return results


def legacy_baseline():
    source=ONE_C/"C3-optical.json"
    fields={}
    for row in load(source)["rows"]:
        if row["method"]!="1c-aligned-EMA" or row["protocol"]!="P2" or row["hidden"]!="Query" or row["aperture"]!=1.4:
            continue
        focus=row["名义对焦距离_m"]
        key=(row["scene"],row["id"])
        row_group=fields.setdefault(key,dict(scene=key[0],case=key[1],status="evaluated",
                                             near_in_focus=None,far_recoverable=None))
        target={("in_focus",1.4):"near_in_focus",("recoverable",5.0):"far_recoverable"}.get(
            (row["region"],focus),{"zero_full_image":"zero_combined",
                                   "zero_visible_full_image":"zero_visible"}.get(row["region"]))
        if target:
            row_group[target]=dict(pixels=row["pixels"],sse=row["sse"])
    pair={}
    with (ROOT/LEGACY_QUERY_PAIRS).open(newline="") as stream:
        for row in csv.DictReader(stream):
            if (row["stage"],row["scene"],row["protocol"],row["aperture"],row["method"],row["weights"],
                row["split"],row["second"])!=("1c","pooled","P2","1.4","OVG","EMA","retrieved","True"):
                continue
            metric={("1.4","in_focus"):"near_in_focus",("5.0","recoverable"):"far_recoverable"}.get(
                (row["nominal_focus_m"],row["region"]))
            if metric:
                pair[metric]=dict(on_db=float(row["psnr"]),off_db=float(row["reference_psnr"]),
                                  delta_db=float(row["PSNR_delta"]),pixels=int(row["pixels"]),cases=int(row["cases"]))
    return dict(checkpoint=str(LEGACY_CHECKPOINT),source=str(source),summary=summary(list(fields.values())),
                query_pair=pair,query_pair_source=str(LEGACY_QUERY_PAIRS),
                note="以前的旧 1c 计算记录。新 B L_occ 使用原生 512。")


def adoption(original, candidate):
    baseline=original["development"]
    proposed=candidate["development"]
    old_step,new_step=original["same_surface_step_m_inv"],candidate["same_surface_step_m_inv"]
    differences={}
    unreliable={}
    for metric in METRICS:
        a,b=baseline[metric],proposed[metric]
        delta=b["psnr_db"]-a["psnr_db"] if None not in (a["psnr_db"],b["psnr_db"]) else None
        differences[metric]=delta
        deviations=[v for v in (a["scene_standard_deviation_db"],b["scene_standard_deviation_db"]) if v is not None]
        unreliable[metric]=delta is not None and bool(deviations) and abs(delta)<max(deviations)
    checks=dict(same_surface_step_improved=old_step is not None and new_step is not None and new_step<old_step,
                both_zero_floors=all(proposed[key]["all_passed"] for key in ("zero_visible","zero_combined")),
                both_p2_noninferior=all(delta is not None and delta>=-.1 for delta in differences.values()))
    return dict(checks=checks,selected="same-surface" if all(checks.values()) else "original",
                p2_candidate_minus_original_db=differences,effect_smaller_than_scene_spread=unreliable,
                same_surface_p95=dict(original=old_step,candidate=new_step),
                scope="paired EMA endpoints on the 15-scene development set")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--f2-root",type=Path,required=True)
    p.add_argument("--phase",choices=("1a","1b"),required=True)
    p.add_argument("--branch",choices=("original","same-surface"),required=True)
    p.add_argument("--weight",type=float,required=True)
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--ema-only",action="store_true")
    p.add_argument("--prepare-f2-only",action="store_true")
    args=p.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="0":
        p.error("assign the shared CUDA device before evaluation")
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    if args.prepare_f2_only:
        prepare_development(development_units(args.smoke),args.f2_root)
    else:
        torch.cuda.set_per_process_memory_fraction(12*GIB/torch.cuda.get_device_properties(0).total_memory,0)
        versions=("ema",) if args.ema_only else ("ema","model")
        evaluate(args.checkpoint,args.output,args.phase,args.branch,args.weight,
                 f2_root=args.f2_root,smoke=args.smoke,versions=versions)
