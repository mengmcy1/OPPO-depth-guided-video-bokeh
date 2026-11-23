"""Compare frozen visible-layer representations and their continuity decisions."""
import argparse
from collections import defaultdict
from html import escape
from pathlib import Path
import time

import numpy as np
import torch

import route3_ovg_data as ovg
import route3_v2data_stage1_visualize as gallery
import route3_v2data_stage2_train as stage2
from route3_ovg_1b_common import CONFIG
from route3_ovg_opaque_b0 import RAY_CLASSES, bins
from route3_ovg_stage2_evaluate import ray_effect, trace_predicted
from route3_ovg_stage2_query import FrozenQuery
from route3_upper_bound_evaluate import display
from route3_v2data_B_batch import GIB, reserve, sha256, write_json
from route3_v2data_B_plan import load
from route3_v2data_query_retrain import distribution
from route3_v2data_stage1_evaluate import load_paired_checkpoint, pooled, region_errors
from route3_v2data_stage1_visualize_opaque import first_pass
from route3_v2data_stage2_eval import f2_for, k16, optical_target, oracle_hidden, peeled_truth, render_offsets
from src.aperture_renderer import aperture_disk
from src.layered_aperture import merge_layers
from src.ovg_edge_statistics import adjacent_surface_pairs
from src.ovg_opaque_eval import opaque_view, subpixel_flags
from src.ovg_optics import raster, subcamera
from src.prefix_transmittance import prefix_transmittance
from src.ovg_representation_eval import THRESHOLDS, VARIANTS, geometry_basis, representation


MISS_ALPHA = 1/255
BINS = 4096
V2 = ("跳过。当前环境缺 nvdiffrast、PyTorch3D。其他环境的 pyrender EGL 实测缺 libGLU，"
      "Open3D 离屏初始化实测缺 Vulkan 库。")
PROTOCOL = {
    "coverage": "Full visible alpha < 1/255, matching the renderer fragment admission cutoff",
    "historical": "13.16.6 uses T >= .1, hidden weight >= .001 and positive per-ray SSE, excluding front order from the T bin. The 118k count was this high-transmittance bin after closure.",
    "ray_domain": "GT opaque foreground interior (exclude native depth boundaries), all 16 nominal P2 K16 samples known, positive-depth, same foreground identity and not hidden. Main scores use the original actual P2 focus.",
    "transmittance": "1 minus accumulated visible alpha at the end of a ray. K16 positive SSE bins use prefix transmittance before the weighted hidden depth, as in the original leakage audit.",
    "bins": ["front_order", "T<0.02", "0.02<=T<0.1", "T>=0.1", "color_vs_F2"],
    "GT_links": "Native right/down continuous/crossing masks from src.ovg_edge_statistics.adjacent_surface_pairs. b1 is never linked. At unlabeled or unlinked points V0 is retained.",
    "predicted_links": "Absolute adjacent inverse-depth difference divided by the smaller positive inverse depth. Both endpoints must be valid b0. Threshold is strict < tau.",
    "thresholds": list(THRESHOLDS),
    "b1": "Frozen Stage 1 alpha/depth and frozen new Query first-pass q, decided by Stage 2 split_mask.",
    "variants": {
        "V0": "Frozen camera-facing 5x5 pixel_layer",
        "V1a": "5x5 centers, opacity and world sigmas unchanged. Link-masked Sobel tangents align each covariance.",
        "V1b": "Same centers and orientation as V1a. Per-axis tangent sigma=k times linked native 3D neighbor spacing (k=.5,.75,1).",
        "V3": "V0 plus a single linked tangent Gaussian per eligible b0 pixel at 1.005 times axial depth, native RGB, sigma=1 times linked native neighbor spacing, opacity .999. Resolve the fill only when V0 has no admitted fragment on that ray, to preserve V0 colors on hit rays even when a neighbor's fill has smaller axial depth.",
        "V2": V2,
    },
    "selection": "Rank by new-development near PSNR gain over V0. Require positive gain plus no reduction in either pooled pinhole PSNR on the new development set, using common cases. Keep at most two.",
    "zero_floors_db": {"zero_visible": 50., "zero_combined": 60.},
    "quantile_bin_width": 1/BINS,
}


def group(row):
    return "development" if row["short"].startswith("dev") else "legacy"


def neighbors(depth, ref, valid, split):
    z = depth.detach().cpu().numpy()
    good = valid.cpu().numpy() & ref["valid"]
    b = split.cpu().numpy()
    rho = np.divide(1., z, out=np.zeros_like(z), where=np.isfinite(z)&(z>0))
    gt_links, relative, samples = [], [], defaultdict(list)
    pred_valid = valid.cpu().numpy()
    eligible = []
    for _, p, q, same, crossing in adjacent_surface_pairs(ref["gt"],ref["object_id"],ref["foreground"],
                                                           good,ref["boundary"]):
        possible = pred_valid[p] & pred_valid[q] & ~b[p] & ~b[q]
        distance = np.abs(rho[p]-rho[q])/np.maximum(np.minimum(rho[p],rho[q]),1e-12)
        for label, support in (("same",same),("crossing",crossing)):
            samples[label].append((distance[support].astype(np.float32), possible[support]))
        gt_links.append(same & possible)
        relative.append(distance)
        eligible.append(possible)
    links={"GT":gt_links}
    for tau in THRESHOLDS:
        links[f"{tau:.2f}"]=[m&(v<tau) for m,v in zip(eligible,relative)]
    links={k:[torch.as_tensor(v,device=depth.device) for v in edge] for k,edge in links.items()}
    samples={k:tuple(np.concatenate(v) for v in zip(*parts)) for k,parts in samples.items()}
    return links,samples


def sample_stats(samples):
    result = {}
    for relation, values in samples.items():
        a,b = (np.concatenate([v[i] for v in values]) for i in (0,1))
        result[relation] = dict(pairs=len(a),quantiles=dict(zip(("p10","p50","p90","p95","p99"),
            np.quantile(a,[.1,.5,.9,.95,.99]).tolist())) if len(a) else {},
            threshold={f"{tau:.2f}":dict(within=int((a<tau).sum()),effective_joined=int(((a<tau)&b).sum()))
                       for tau in THRESHOLDS})
    return result


@torch.inference_mode()
def frame_setup(plan, net, query, views, memory):
    item=("B",plan["name"],ovg.TRAIN_FRAMES.index(plan["frame"]))
    x,y=stage2.sample(item)
    x,ref,masks,subjects=gallery.frame_reference(plan,(x,y))
    if query is None:
        pred=net(x,protect_residual=True,render_heads=False)
        depth=pred["depth1"][0]
        z=depth.cpu().numpy()
        from route3_v2data_stage1_evaluate import p2_focus
        from src.ovg_optics import fixed_visible_surfels
        new=dict(prediction=pred,depth=depth,visible=fixed_visible_surfels(pred,x),
                 focuses=[p2_focus(z,subject) for subject in subjects])
    else:
        new=gallery.prepare_geometry(net,x,views,memory,plan["frame"],plan["cases"],subjects)
        depth=new["depth"]
    cached=stage2.RUN/"observed-first/B"/plan["name"]/f'f{plan["frame"]:03d}.npz'
    if cached.is_file():
        first=ovg.load(cached,("q",))
    else:
        if query is None:
            query=FrozenQuery(stage2.QUERY)
            views,memory=gallery.scene_views(plan)
        first=first_pass(plan,new["prediction"],query,views,memory)[0]
    split=stage2.split_mask(new["prediction"],first)
    valid=torch.isfinite(depth)&(depth>0)&torch.isfinite(x["rgb"][0]).all(0)
    new["b0"]=valid&~split
    links,samples=neighbors(depth,ref,valid,split)
    interior=(y["alpha"][0].cpu().numpy()>=.9999)&y["valid_alpha"][0].cpu().numpy()&~ref["boundary"]
    return x,ref,masks,new,valid,links,samples,interior


def prep_case(plan, state, query, index, original):
    x,ref,masks,new,_,_,_,interior=state
    unit,name,frame=(plan[k] for k in ("unit","name","frame"))
    case,K=plan["cases"][index],x["K"][0]
    focus=original["actual_focus_m"]["S3"]
    if new["focuses"][index] is None or abs(new["focuses"][index]-focus)>1e-6:
        raise ValueError("Frozen P2 focus differs from source gallery")
    offsets=aperture_disk(CONFIG["native_eval_rays"],.05/(2*case["fstop"]),device=K.device,dtype=K.dtype)
    center=K.new_zeros((1,2))
    fill=f2_for(unit,plan["entry"],frame,x,focus,offsets)
    zfill=f2_for(unit,plan["entry"],frame,x,focus,center)
    truth=optical_target(unit,name,case)
    pinhole=display(x["rgb"][0].permute(1,2,0).cpu().numpy())
    gt,_=oracle_hidden(peeled_truth(unit,name,frame),ref,new["depth"],K,focus,case["fstop"])
    hidden={"S5":gt}
    if query is not None:
        hidden.update(S3=gallery.query_layer(query.network,new,K,frame,case,index),S4=None)
    k=k16(unit,name,frame,case)
    z=k["depth"]
    good=(k["ids"]==ref["object_id"][None])&k["known"]&~k["hidden"]&np.isfinite(z)&(z>0)
    ray_mask=interior&ref["valid"]&np.isin(ref["object_id"],ref["foreground"])&good.all(0)
    replays=K.new_tensor(k["aperture_xy"])
    nominal=float(case["focus_m"])
    kfill=f2_for(unit,plan["entry"],frame,x,nominal,replays)
    return dict(K=K,focus=focus,offsets=offsets,fill=fill,center=center,zfill=zfill,truth=truth,pinhole=pinhole,
                hidden=hidden,ray_mask=ray_mask,replays=replays,nominal=nominal,kfill=kfill,mask=masks[index],
                is_near=index==0)


def close_layer(visible, hidden, K, focus, offsets, bg, params, alpha):
    merged=merge_layers(visible,hidden)
    result,closed=[],[]
    for i,offset in enumerate(offsets):
        value,info=opaque_view(merged,K,focus,offset,*params,None if bg is None else bg[i])
        result.append(value)
        closed.append(info["closed"])
    images=torch.stack(result)
    if not alpha:
        return images
    _,raw=raster(visible,K,focus,offsets,512,512,return_alpha=True)
    return images,raw.masked_fill(torch.stack(closed),1)


def layered_rays(visible, hidden, K, focus, offsets, bg=None, *, alpha=False, closure=None):
    if not isinstance(visible,tuple):
        return raster(merge_layers(visible,hidden),K,focus,offsets,512,512,bg,return_alpha=alpha)
    base,fill=visible
    front,coverage=raster(base,K,focus,offsets,512,512,bg,return_alpha=True)
    miss=coverage<MISS_ALPHA
    if closure is not None:
        front=close_layer(base,hidden,K,focus,offsets,bg,closure[0],alpha)
        if alpha:
            front,coverage=front
        if not len(fill.means):
            return (front,coverage) if alpha else front
        farther=close_layer(fill,hidden,K,focus,offsets,bg,closure[1],alpha)
        if alpha:
            farther,other=farther
    else:
        if hidden is not None:
            front=raster(merge_layers(base,hidden),K,focus,offsets,512,512,bg)
        farther,other=raster(merge_layers(fill,hidden),K,focus,offsets,512,512,bg,return_alpha=True)
    result=torch.where(miss[...,None],farther,front)
    return (result,torch.where(miss,other,coverage)) if alpha else result


def rays_for(layer, case, visible_result, hidden=None, closure=None):
    K,focus,offsets,bg=(case[k] for k in ("K","nominal","replays","kfill"))
    images,alphas=[],[]
    for a,b in zip(offsets.split(4),bg.split(4)):
        outputs=layered_rays(layer,hidden,K,focus,a,b,alpha=visible_result,closure=closure)
        if visible_result:
            rgb,alpha=outputs
            images.append(rgb)
            alphas.append(alpha)
        else:
            images.append(outputs)
    return torch.cat(images),torch.cat(alphas) if alphas else None


def opaque_trace(visible, merged, K, focus, offset, support, params):
    w,z,zv,wv,T=trace_predicted(visible,visible,merged,K,focus,offset,support)
    if params is not None:
        _,diag=opaque_view(merged,K,focus,offset,*params)
        closed=diag["closed"][support].cpu().numpy()
        w[closed]=diag["query_weight"][support].cpu().numpy()[closed]
        z[closed]=diag["query_depth"][support].cpu().numpy()[closed]
        T[closed]=np.nan
        survived=closed&(w>=.001)
        if survived.any():
            depths=torch.zeros_like(params[0]); depths[support]=K.new_tensor(z)
            intr,view=subcamera(K,focus,offset)
            prefix=prefix_transmittance(visible,intr,view,depths)[support].cpu().numpy()
            T[survived]=prefix[survived]
    return w,z,zv,wv,T


def ray_metrics(visible, hidden, case, base, closure=None):
    images,alpha=base
    mask=case["ray_mask"]
    map_miss=(alpha<MISS_ALPHA).float().mean(0).cpu().numpy()
    T=(1-alpha[:,mask]).clamp(0,1).cpu().numpy()
    hist=np.histogram(T,bins=BINS,range=(0.,1.))[0]
    result=dict(rays=T.size,missed=int((alpha[:,mask]<MISS_ALPHA).sum()),T_hist=hist.tolist(),positive_sse_bins=[])
    if case["is_near"] and T.size:
        rgb=rays_for(visible,case,False,hidden,closure)[0]
        _,effect,_,_=ray_effect(images[:,mask].cpu().numpy(),rgb[:,mask].cpu().numpy(),case["truth"][mask])
        sel=torch.from_numpy(mask).to(case["K"].device)
        traces=[]
        for offset in case["replays"]:
            if isinstance(visible,tuple):
                a,b=visible
                hit=raster(a,case["K"],case["nominal"],offset[None],512,512,return_alpha=True)[1][0]>=MISS_ALPHA
                params=closure if closure is not None else (None,None)
                components=[(a,sel&hit,params[0]),(b,sel&~hit,params[1])]
            else:
                components=[(visible,sel,None)]
            row=np.zeros((5,int(mask.sum())),np.float32)
            for part,support,params in components:
                if support.any():
                    row[:,support[sel].cpu().numpy()]=opaque_trace(part,merge_layers(part,hidden),
                        case["K"],case["nominal"],offset,support,params)
            traces.append(row)
        w,zh,zv,_,prefix=map(np.stack,zip(*traces))
        result["positive_sse_bins"]=bins(w,zh,zv,prefix,effect)
    return result,map_miss


def score(value,target,mask):
    return pooled([region_errors(value,target,mask)])


def evaluate_variant(layer, case, name, system, measure_rays, visible_k16, closure=None):
    K,focus,center,zfill=(case[k] for k in ("K","focus","center","zfill"))
    mask,truth,pinhole=(case[k] for k in ("mask","truth","pinhole"))
    hidden=case["hidden"][system]
    if isinstance(layer,tuple):
        chunks=[layered_rays(layer,hidden,K,focus,a,b,closure=closure).sum(0)
                for a,b in zip(case["offsets"].split(8),case["fill"].split(8))]
        rgb=sum(chunks)/len(case["offsets"])
    else:
        rgb=render_offsets(merge_layers(layer,hidden),K,focus,case["offsets"],case["fill"])
    valid=np.ones(mask.shape,bool)
    row=dict(variant=name, system=system, optical=score(rgb,truth,mask),
             zero_visible=score(layered_rays(layer,None,K,focus,center,closure=closure)[0],pinhole,valid),
             zero_combined=score(layered_rays(layer,hidden,K,focus,center,zfill,closure=closure)[0],pinhole,valid))
    if measure_rays:
        row["rays"],miss=ray_metrics(layer,hidden,case,visible_k16,closure)
    else:
        miss=None
    return row,display(rgb.cpu().numpy()),miss


def stats(rows, phase):
    output={}
    for partition in ("development","legacy"):
        result=output[partition]={}
        source=[r for r in rows if group(r)==partition and not r.get("error")]
        variants={(c["variant"],c["system"]) for r in source for c in r["variants"]}
        for variant,system in sorted(variants):
            pairs=[]
            for r in source:
                chosen=next((v for v in r["variants"] if (v["variant"],v["system"])==(variant,system)),None)
                comparison=("V0","S5") if phase=="step1" else (variant.split("@")[0]+"@GT",system)
                base=next((v for v in r["variants"] if (v["variant"],v["system"])==comparison),None)
                if chosen is not None:
                    pairs.append((r,chosen,base))
            item=result[variant+"/"+system]={}
            for focus in ("near","far"):
                sub=[(r,c,b) for r,c,b in pairs if r["focus"]==focus]
                scenes=defaultdict(list)
                for r,c,b in sub:
                    scenes[r["scene"]].append((c,b))
                per_scene={}
                for name,values in scenes.items():
                    p=pooled([a["optical"] for a,_ in values])["psnr_db"]
                    b=[q["optical"] for _,q in values if q is not None]
                    per_scene[name]=dict(psnr_db=p,delta_db=p-pooled(b)["psnr_db"] if len(b)==len(values) else None)
                item[focus]=dict(cases=len(sub),pooled=pooled([c["optical"] for _,c,_ in sub]),per_scene=per_scene,
                    median_db=distribution([v["psnr_db"] for v in per_scene.values()])["median_db"],
                    paired=distribution([v["delta_db"] for v in per_scene.values() if v["delta_db"] is not None]))
            for kind,floor in PROTOCOL["zero_floors_db"].items():
                item[kind]=dict(pooled=pooled([c[kind] for _,c,_ in pairs]),
                    cases=len(pairs),passed=sum(c[kind]["psnr_db"]>=floor for _,c,_ in pairs))
            ray_rows=[c["rays"] for _,c,_ in pairs if "rays" in c]
            if ray_rows:
                count=sum(r["rays"] for r in ray_rows)
                hist=np.sum([r["T_hist"] for r in ray_rows],axis=0).cumsum()
                quantiles=[(int(np.searchsorted(hist,f*count))+.5)/BINS if count else None for f in (.5,.95)]
                sse=defaultdict(float)
                for r in ray_rows:
                    for b in r["positive_sse_bins"]:
                        sse[b["category"]]+=b["positive_SSE"]
                pos=sum(sse.values())
                item["rays"]=dict(n=count,miss=sum(r["missed"] for r in ray_rows)/count if count else None,
                  T_p50_p95=quantiles,positive_sse=dict(sse),high_sse_fraction=sse.get(RAY_CLASSES[3],0.)/pos if pos else None)
    return output


def select(stats):
    data=stats["development"]
    ref=data["V0/S5"]
    ordered=[]
    for name in VARIANTS[1:]:
        label=name+"/S5"
        if label not in data:
            continue
        row=data[label]
        gain=row["near"]["pooled"]["psnr_db"]-ref["near"]["pooled"]["psnr_db"]
        all_zero=all(row[k]["cases"]==ref[k]["cases"] and
                     row[k]["pooled"]["psnr_db"]>=ref[k]["pooled"]["psnr_db"] for k in PROTOCOL["zero_floors_db"])
        if gain>0 and all_zero:
            ordered.append((gain,name))
    return [label for _,label in sorted(ordered,key=lambda x:(-x[0],x[1]))[:2]]


def fig_key(row, phase, repr_name=None):
    name=row["file"]
    if name=="021_dev001_f030_near.png":
        name="000_dev001_f030_near.png"
    return f"{phase}_{repr_name+'_' if repr_name else ''}{name}"


def plot_results(out,reports):
    plots=[]
    cache=out/".work"
    selected=reports["selected"]
    if selected:
        lead=selected[0]
    else:
        report=reports["step1"]["summary"]["development"]
        options=[v for v in VARIANTS[1:] if v+"/S5" in report]
        lead=max(options,key=lambda v:report[v+"/S5"]["near"]["pooled"]["psnr_db"]) if options else "V0"
    for phase in ("step1","v3opaque","step2"):
        if phase not in reports:
            continue
        for row in sorted(reports[phase]["rows"],key=lambda r:fig_key(r,phase)):
            if row.get("error"):
                continue
            for variant in (selected if phase=="step2" else (None,)):
                name=fig_key(row,phase,variant)
                p=out/name
                if p.exists():
                    plots.append((name,row))
                    continue
                suffix=f"{phase}_{row['file']}.npz"
                with np.load(cache/suffix,allow_pickle=False) as d:
                    images={"input":d["input"].astype(np.float32),"GT":d["GT"].astype(np.float32)}
                    scores={"input":row["input"],"GT":row["GT"]}
                    if phase=="step1":
                        cells=[(v["variant"],v["variant"],v) for v in row["variants"]]
                        map_keys=("V0",lead) if lead!="V0" else ("V0",)
                    elif phase=="v3opaque":
                        cells=[(v["variant"],v["variant"],v) for v in row["variants"]]
                        baseline=cache/f"step1_{row['file']}.npz"
                        if baseline.is_file():
                            with np.load(baseline,allow_pickle=False) as original:
                                for key in ("V0","V3"):
                                    images[key]=original[key].astype(np.float32)
                        map_keys=("V3+o",)
                    else:
                        kept=[v for v in row["variants"] if v["variant"].startswith(variant+"@")]
                        cells=[(v["variant"]+"/"+v["system"],v["variant"].split("@")[1]+" "+v["system"],v)
                               for v in kept if v["system"]=="S5" or v["variant"].endswith("@0.10")]
                        map_keys=()
                    columns={"input":"输入 RGB","GT":"GT · Cycles"}
                    for key,title,v in cells:
                        if key not in images:
                            images[key]=d[key].astype(np.float32)
                        scores[key]=v["optical"]
                        columns[key]=title
                    maps={v+" · 未命中率":d["miss_"+v].astype(np.float32) for v in map_keys
                          if "miss_"+v in d}
                    focus={k:row["actual_focus_m"] for k in scores}
                    tile=min(252,(2400-40-10*(len(columns)+len(maps)-1))//(len(columns)+len(maps)))
                    gallery.draw_panel(p,dict(row,region_scores=scores,actual_focus_m=focus),images,
                                       (d["near"],d["far"]),maps,columns=columns,tile=tile)
                plots.append((name,row))
    return plots


@torch.inference_mode()
def run_phase(out, originals, phase, selected, pilot=False, retry=False):
    path=out/(phase+".json")
    report=load(path) or dict(rows=[],seconds=0.)
    existing={r["file"]:r for r in report["rows"]}
    retry_names={name for name,r in existing.items() if r.get("error","").startswith("OutOfMemoryError:")} if retry else set()
    for name in retry_names:
        failed=existing.pop(name)
        report.setdefault("retry_history",[]).append(dict(file=name,original_error=failed["error"],at=gallery.stamp()))
    previous=report["seconds"]
    start=time.monotonic()
    net=load_paired_checkpoint(stage2.INITIAL,"ema").eval()
    query=FrozenQuery(stage2.QUERY) if phase=="step2" else None
    plan=list(gallery.planned_frames())
    plan.sort(key=lambda t:t["files"][0]["short"]!="dev001" or t["frame"]!=30)
    saved=originals["rows"]
    source={r["file"]:r for r in saved if "error" not in r}
    scene=None
    views=memory=None
    samples=defaultdict(list)

    for task in plan:
        pending=[(i,source[f["file"]]) for i,f in enumerate(task["files"])
                 if f["file"] in source and f["file"] not in existing]
        if retry:
            pending=[p for p in pending if p[1]["file"] in retry_names]
        if pilot:
            pending=[p for p in pending if p[1]["file"]=="021_dev001_f030_near.png"]
        if not pending:
            continue
        reserve(out,GIB)
        try:
            if query is not None and scene!=task["name"]:
                views,memory=gallery.scene_views(task)
                scene=task["name"]
            state=frame_setup(task,net,query,views,memory)
            x,ref,masks,new,valid,links,pairs,_=state
            if phase=="step1":
                for label,values in pairs.items():
                    samples[(group(pending[0][1]),label)].append(values)
            cases=[]
            for i, r in pending:
                try:
                    cases.append((i,r,prep_case(task,state,query,i,r)))
                except Exception as error:
                    existing[r["file"]]=dict(r,error=f"{type(error).__name__}: {error}")
                    print("REPR_SKIP",phase,r["file"],str(error),flush=True)
                    torch.cuda.empty_cache()
            if phase=="step1":
                grids=[(v,"GT",v) for v in VARIANTS]
            elif phase=="v3opaque":
                grids=[("V3","GT","V3+o")]
            else:
                grids=[(v,t,v+"@"+t) for v in selected for t in ("GT","0.03","0.10","0.25")]
            arrays={r["file"]:dict(input=c["pinhole"].astype(np.float16),GT=c["truth"].astype(np.float16),
                                    near=masks[0],far=masks[1]) for _,r,c in cases}
            values=defaultdict(list)
            color=x["rgb"][0].permute(1,2,0)
            basis_by_link={}
            for variant,method,label in grids:
                if variant!="V0" and method not in basis_by_link:
                    basis_by_link[method]=geometry_basis(new["depth"],x["K"][0],links[method])
                layer=representation(new["visible"],new["depth"],color,x["K"][0],valid,links[method],variant,
                                     basis_by_link.get(method))
                closure=None
                if phase=="v3opaque":
                    depth=new["depth"]
                    b0=new["b0"]
                    fill_mask=basis_by_link[method][3]&valid
                    base,fill=layer
                    closure=((depth,b0,subpixel_flags(valid,b0,5),len(base.means)),
                             (depth*1.005,fill_mask,torch.ones(len(fill.means),dtype=torch.bool,device=depth.device),len(fill.means)))
                for _,ref_row,c in cases:
                    name=ref_row["file"]
                    if name in existing:
                        continue
                    try:
                        measure_rays=phase!="step2"
                        base=rays_for(layer,c,True,closure=closure) if measure_rays else None
                        for system in ("S5",) if phase!="step2" else ("S3","S4","S5"):
                            row,rgb,miss=evaluate_variant(layer,c,label,system,measure_rays,base,closure)
                            key=label if phase!="step2" else label+"/"+system
                            if variant=="V0":
                                ref_score=ref_row["region_scores"]["S5"]
                                if row["optical"]["pixels"]!=ref_score["pixels"] or abs(row["optical"]["psnr_db"]-ref_score["psnr_db"])>.001:
                                    raise ValueError("V0 S5 differs from the original gallery")
                            arrays[name][key]=rgb.astype(np.float16)
                            if miss is not None:
                                arrays[name]["miss_"+label]=miss.astype(np.float16)
                            values[name].append(row)
                    except Exception as error:
                        existing[name]=dict(ref_row,error=f"{type(error).__name__}: {error}")
                        print("REPR_SKIP",phase,name,str(error),flush=True)
                        torch.cuda.empty_cache()
                del layer
            for _,old,c in cases:
                name=old["file"]
                if name in existing:
                    continue
                images=arrays[name]
                np.savez_compressed(out/".work"/f"{phase}_{name}.npz",**images)
                count=int(c["mask"].sum())
                result=dict(file=name,scene=old["scene"],short=old["short"],frame=old["frame"],focus=old["focus"],
                            crop_xyxy=old["crop_xyxy"],actual_focus_m=c["focus"],variants=values[name],
                            input=old["region_scores"]["input"],GT=pooled([dict(pixels=count,sse=0.)]))
                existing[name]=result
                print("REPR_DONE",phase,name,[(r["variant"],r["system"],round(r["optical"]["psnr_db"],3))
                                              for r in values[name] if r["system"]=="S5"],flush=True)
        except Exception as error:
            for _,r in pending:
                existing.setdefault(r["file"],dict(r,error=f"{type(error).__name__}: {error}"))
                print("REPR_SKIP",phase,r["file"],str(error),flush=True)
            torch.cuda.empty_cache()
        report["rows"]=list(existing.values())
        report["seconds"]=previous+time.monotonic()-start
        write_json(path,report)
    if not pilot:
        report["summary"]=stats(report["rows"],"step1" if phase=="v3opaque" else phase)
        if phase=="step1" and not retry:
            report["adjacent"]={g:sample_stats({r:samples[(g,r)] for r in ("same","crossing")})
                                for g in ("development","legacy") if samples[(g,"same")]}
        report["completed_at"]=gallery.stamp()
        report["seconds"]=previous+time.monotonic()-start
        write_json(path,report)
    return report


def compare_opaque(ordinary, opaque):
    by_case={r["file"]:r for r in ordinary["rows"] if not r.get("error")}
    for row in opaque["rows"]:
        if row["file"] in by_case and not row.get("error"):
            baseline=[x for x in by_case[row["file"]]["variants"] if x["variant"] in ("V0","V3")]
            row["variants"]=baseline+[x for x in row["variants"] if x["variant"]=="V3+o"]
    common=[row for row in opaque["rows"] if not row.get("error") and len(row["variants"])==3]
    opaque["summary"]=stats(common,"step1")


def number(v, digits=3):
    return "—" if v is None else f"{v:.{digits}f}"


def publish(out,reports,plots):
    lines=["# 可见层表示与连续性判定","",V2,"",
           "近焦和远焦列为合并 PSNR／场景中位数／配对中位数／正、负、平场景数。"
           "第一步以 V0 为参照，第二步以同表示、同隐藏层的 GT 连接为参照。"
           "零光圈列为合并 PSNR 与过线数。未命中以累计 α < 1/255 定义，高透射 SSE 按 §13.16.6 统计。",""]
    lines.append("选中表示为 "+("、".join(reports["selected"]) if reports["selected"] else "无，通过两个零光圈约束的正收益候选为空")+"。")
    for phase in ("step1","v3opaque","step2"):
        if phase not in reports:
            continue
        phase_report=reports[phase]
        lines += ["",f"## {phase} · {phase_report['seconds']:.1f} 秒",""]
        for name,part in phase_report["summary"].items():
            lines += [f"### {name}","",
                "| 表示 | 近焦 | 远焦 | 零光圈可见层 | 零光圈含隐藏层 | 未命中率 | T p50 / p95 | T≥0.1 正向 SSE 份额 |",
                "|---|---|---|---|---|---|---|---|"]
            for key,row in part.items():
                f=lambda s:f"{number(s['pooled']['psnr_db'])} / {s['passed']}/{s['cases']}"
                def optic(s):
                    p=s["paired"]
                    return (f"{number(s['pooled']['psnr_db'])} / {number(s['median_db'])} / {number(p['median_db'])} / "
                            f"{p['positive']},{p['negative']},{p['tied']}")
                r=row.get("rays",{})
                t=" / ".join(number(x,5) for x in r.get("T_p50_p95",()))
                percent=lambda x:number(x*100,3)+"%" if x is not None else "—"
                lines.append(f"| {key} | {optic(row['near'])} | {optic(row['far'])} | {f(row['zero_visible'])} | "
                             f"{f(row['zero_combined'])} | {percent(r.get('miss'))} | {t or '—'} | {percent(r.get('high_sse_fraction'))} |")
            lines.append("")
    adjacent=reports["step1"].get("adjacent",{})
    lines+=["## 相邻逆深度统计",""]
    for partition,by_relation in adjacent.items():
        for label,b in by_relation.items():
            lines.append(f"- {partition} {label}，{b['pairs']:,} 对。分位数 {b['quantiles']}。")
            for tau,r in b["threshold"].items():
                n=r["effective_joined"] if label=="crossing" else b["pairs"]-r["effective_joined"]
                lines.append(f"  - τ={tau}，{'跨断层误缝合' if label=='crossing' else '同表面未缝合'} {n/max(1,b['pairs']):.2%}。")
    lines+=["","## 跳过",""]+[f"- {r['file']}，{r['error']}" for r in reports["skipped"]]
    for retry in reports["step1"].get("retry_history",[]):
        name=retry["file"]
        current=next(r for r in reports["step1"]["rows"] if r["file"]==name)
        result="已补跑成功" if "error" not in current else "补跑后仍失败"
        lines.append(f"- {name}，首次显存不足，{result}。")
    (out/"汇总.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    html=['<!doctype html><meta charset=utf-8><title>可见层表示对照</title><h1>可见层表示对照</h1>',
          '<p><a href="汇总.md">汇总</a></p><div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,34em),1fr));gap:1em">']
    html.extend(f'<figure style="margin:0"><a href="{n}"><img src="{n}" width="100%" loading="lazy"></a>'
                f'<figcaption>{escape(row["short"])} f{row["frame"]:03d} · {row["focus"]} · {n}</figcaption></figure>'
                for n,row in plots)
    (out/"index.html").write_text("\n".join(html+["</div>"])+"\n",encoding="utf-8")


def main(source,pilot,retry=False):
    stage2.configuration()
    torch.cuda.set_per_process_memory_fraction((9 if retry else 7)*GIB/torch.cuda.get_device_properties(0).total_memory,0)
    torch.backends.cuda.matmul.allow_tf32=False
    stage2.fixed_inputs()
    out=source/"repr"
    out.mkdir(exist_ok=True)
    (out/".work").mkdir(exist_ok=True)
    if not (out/"protocol.json").exists():
        write_json(out/"protocol.json",dict(PROTOCOL,source_sha256=sha256(source/"metrics.json"),registered_at=gallery.stamp()))
    originals=load(source/"metrics.json")
    report=load(out/"step1.json")
    if retry:
        run_phase(out,originals,"step1",[],retry=True)
        return
    if not report.get("completed_at"):
        report=run_phase(out,originals,"step1",[],pilot)
    if pilot:
        return
    selected=select(report["summary"])
    result=dict(step1=report,selected=selected)
    note=out/"v3opaque-protocol.json"
    if not note.is_file():
        write_json(note,dict(variant="V3+o",connection="GT",system="S5",registered_at=gallery.stamp(),
            closure="First resolve the V3 branch from ordinary V0 admission, then apply the existing b=0 opaque operation to the selected layer. Both the V0 and fill branch retain their own eligible pixel centers."))
    closed=load(out/"v3opaque.json")
    if not closed.get("completed_at"):
        closed=run_phase(out,originals,"v3opaque",[])
    compare_opaque(report,closed)
    result["v3opaque"]=closed
    if selected:
        second=load(out/"step2.json")
        result["step2"]=second if second.get("completed_at") else run_phase(out,originals,"step2",selected)
    result["skipped"]=[r for r in originals["rows"] if r.get("error")]
    for name in ("step1","v3opaque","step2"):
        result["skipped"].extend(r for r in result.get(name,{}).get("rows",[]) if r.get("error"))
    plots=plot_results(out,result)
    write_json(out/"metrics.json",result)
    publish(out,result,plots)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,default=gallery.OUT)
    p.add_argument("--pilot",action="store_true")
    p.add_argument("--retry-oom",action="store_true")
    a=p.parse_args()
    main(a.source.resolve(),a.pilot,a.retry_oom)
