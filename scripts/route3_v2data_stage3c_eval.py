"""Evaluate joint Stage 3c checkpoints against the independently replayed start."""
import argparse
from collections import defaultdict
from html import escape
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch

import route3_ovg_data as ovg
import route3_v2data_stage1_visualize as gallery
from route3_v2data_B_batch import sha256,write_json
from route3_v2data_B_plan import load
from route3_v2data_stage1_evaluate import near_subject,new_regions,p2_focus
from route3_v2data_stage2_eval import frames,geometry_root,legacy_masks
from route3_v2data_stage2_train import sample
from route3_v2data_stage3a import FIXED,OUT as A_RUN,Placement,gpu_setup
from route3_v2data_stage3a_eval import guard_report,hidden_layer,render
from route3_v2data_stage3a_prior_diagnostic import thin_regions
from route3_v2data_stage3b import OUT as B_RUN,load_geometry
from route3_v2data_stage3b_eval import COHORTS,REGIONS,deltas,evaluate,gt_reference_path,scores_by_scene,slug,summary
from route3_v2data_stage3b_followup import frozen_references
from route3_v2data_stage3c import DB_FLOORS,GUARDS,OUT
from src.c1_bokehme import region_sse
from src.layered_aperture import merge_layers
from src.ovg_edge_statistics import EdgeStatistics


def report_path(run,step,weights):
    return run/"evaluation"/f"step-{step:06d}"/weights/"joint-report.json"


def zero_compare(run):
    old=B_RUN/"evaluation/step-000000/ema/report-fp16-fit-corrected.json"
    replay=report_path(run,0,"ema").with_name("report.json")
    a,b=load(old),load(replay)
    if not a or not b:
        raise FileNotFoundError(f"Missing corrected 3b reference or independent 3c replay: {old}, {replay}")
    identical=a["rows"]==b["rows"]
    regrouped=summary(b["rows"])
    mismatches=[(r["scene"],r["case"],[k for k in r if r[k]!=s.get(k)])
                for r,s in zip(a["rows"],b["rows"]) if r!=s]
    corrected=gt_reference_path()
    result=dict(passed=identical and a["summary"]==regrouped and str(b["GT_reference"]).endswith(corrected.name),
                scored_development=regrouped["development"]["scored_cases"],scored_legacy=regrouped["legacy"]["scored_cases"],
                checked_rows=len(b["rows"]),row_mismatches=mismatches,all_summaries_equal=a["summary"]==regrouped,
                baseline_sha256=sha256(old),replay_sha256=sha256(replay),corrected_GT_sha256=sha256(corrected),
                comparison="Exact equality of all case fields and every regrouped 3b score. Zero EMA and raw contain the same initialization.")
    write_json(run/"zero-consistency.json",result)
    if not result["passed"]:
        raise ValueError(f"3c zero does not match FP16-corrected 3b zero: {mismatches[:4]}")
    return b


def percentiles(samples):
    if not samples:
        return dict(pixels=0,p50=None,p95=None)
    data=np.concatenate(samples)
    if not data.size:
        return dict(pixels=0,p50=None,p95=None)
    p50,p95=np.quantile(data,[.5,.95])
    return dict(pixels=int(data.size),p50=float(p50),p95=float(p95))


class Joint:
    def __init__(self,run,step):
        self.run=run
        self.step=step
        self.edges=defaultdict(EdgeStatistics)
        self.inverse_stats=defaultdict(list)
        self.coc_guards=defaultdict(list)
        self.drift=defaultdict(list)
        self.thin_coc=defaultdict(list)
        self.place=Placement(run/"dev-hidden-parents",A_RUN/"dev-hidden-parents")
        self.start=load(report_path(run,0,"ema").with_name("report.json"))["rows"]
        self.by_start={(r["scene"],r["case"]):r for r in self.start}

    def restore(self,geometry,query,saved):
        missing,unexpected=geometry.load_state_dict(saved["geometry"],strict=False)
        if unexpected or any(not name.startswith("encoder.") for name in missing):
            raise ValueError("Unexpected joint geometry checkpoint fields")
        query.load_state_dict(saved["query"])

    def frame(self,cohort,name,frame,depth,x,y,ref):
        if not (np.isfinite(depth).all() and (depth>0).all()):
            raise FloatingPointError(f"Invalid depth for {name}/f{frame:03d}")
        target=y["depth1"][0].cpu().numpy().astype(np.float32)
        valid=ref["valid"] & np.isfinite(target)&(target>0)
        disparity=np.zeros_like(depth,dtype=np.float32)
        disparity[valid]=(1/depth[valid].astype(np.float64)-1/target[valid].astype(np.float64))
        receipt=dict(pixels=int(valid.sum()),sse=float(np.square(disparity[valid].astype(np.float64)).sum()))
        self.inverse_stats[cohort].append(receipt)
        self.edges[cohort].add(name,"3c",depth,ref,valid)
        baseline=self.run/"evaluation/zero-depth"/name/f"f{frame:03d}.npy"
        if not self.step:
            baseline.parent.mkdir(parents=True,exist_ok=True)
            np.save(baseline,depth)
        old=np.load(baseline)
        delta=np.abs(1/depth-1/old)
        self.drift[cohort].append(delta.ravel())
        if slug(name) in ("dev003","dev004","dev005","dev008"):
            self.drift["dev008" if slug(name)=="dev008" else "thin_structure"].append(delta.ravel())
        self.current=dict(cohort=cohort,name=name,frame=frame,depth=depth,baseline=old,ref=ref,valid=valid,
                          disparity=disparity,K=x["K"][0].cpu().numpy())

    def oracle(self,unit,entry,case,ref,depth,K,focus,visible,offsets,fill,truth,area):
        state=self.current
        hidden,_=hidden_layer(self.place,unit,state["name"],state["frame"],case,ref,depth,K)
        return region_sse(render(merge_layers(visible,hidden),K,focus,offsets,fill),truth,area)

    def case(self,row,unit,case,area):
        frame=self.current
        k,ref=frame["K"],frame["ref"]
        focus=case["focus_m"]
        subject=near_subject(ref["gt"],ref["valid"],k,focus)
        row["GT_focus_m"]=p2_focus(ref["gt"].astype(np.float32),subject)
        foreground=frame["valid"]&np.isin(ref["object_id"],ref["foreground"])
        coefficient=float(k[0,0])*.05/(2*case["fstop"])*focus/(focus-.05)
        error=np.abs(frame["disparity"])*coefficient
        if row["focus_index"]==0:
            use=foreground&area
            self.coc_guards[frame["cohort"]].append(dict(coc_guard=dict(
                pixels=int(use.sum()),absolute_error_sum=float(error[use].astype(np.float64).sum()))))
        ident=slug(row["scene"])
        if unit and ident in ("dev003","dev004","dev005","dev008"):
            native, _=thin_regions(ident,unit,ovg.b_foreground_reference(row["scene"],row["frame"]))
            group="dev008" if ident=="dev008" else "thin_structure"
            self.thin_coc[group].append(error[native["foreground"]&frame["valid"]])

    def panel(self,output,panel,row,case,area,current=None,truth=None):
        frame=self.current
        off,curr=(frame[key] for key in ("baseline","depth"))
        ref=frame["ref"]
        use=frame["valid"]&np.isin(ref["object_id"],ref["foreground"])
        base=self.by_start[row["scene"],row["case"]]
        a,b=base["focus_m"],row["focus_m"]
        def coc(depth,focus):
            coefficient=float(frame["K"][0,0])*.05/(2*case["fstop"])*focus/(focus-.05)
            return np.where(use,np.abs(1/depth-1/focus)*coefficient,0.)
        start_coc,now_coc=coc(off,a),coc(curr,b)
        box=gallery.largest_error_crop(current,truth,area) if current is not None else (0,0,512,512)
        stop=max(.01,float(np.quantile(np.r_[1/off.ravel(),1/curr.ravel()],.99)))
        panels=(("起点逆深度",1/off,stop),("当前逆深度",1/curr,stop),
                ("起点前景 CoC",start_coc,6.),("当前前景 CoC",now_coc,6.))
        file=output/panel["path"]
        with Image.open(file) as opened:
            rendered=opened.convert("RGB")
        combined=Image.new("RGB",(rendered.width,rendered.height+255),(18,23,31))
        combined.paste(rendered,(0,0))
        draw=ImageDraw.Draw(combined)
        for i,(label,values,limit) in enumerate(panels):
            x=i*(rendered.width//4)
            draw.text((x+2,rendered.height+5),f"{label}，0–{limit:.2f}",font=gallery.font(14),fill="white")
            picture=gallery.scalar_image(values,limit,cv2.COLORMAP_VIRIDIS).crop(box)
            combined.paste(picture.resize((220,220)),(x,rendered.height+29))
        path=file.with_name(file.stem+"-geometry.jpg")
        combined.save(path,quality=93)
        # Verify the generated optical and geometry image can be read before recording the observation.
        with Image.open(path) as exported:
            exported.verify()
        focus_mask=use&area
        if not focus_mask.any():
            focus_mask=use
        drift=np.abs(1/curr-1/off)
        p95=float(np.quantile(drift[focus_mask],.95)) if focus_mask.any() else 0.
        if focus_mask.any():
            start_median,current_median=(float(np.median(x[focus_mask])) for x in (start_coc,now_coc))
            foreground_note=f"主体 CoC 中位 {start_median:.3f} → {current_median:.3f} px"
        else:
            foreground_note="图内主体无有效 GT 前景像素"
        panel.update(path=str(path.relative_to(output)),
                     observation=f'{panel["observation"]} {foreground_note}，逆深度漂移 p95 {p95:.4f} m⁻¹。')

    def summarize(self,payload):
        baseline=load(report_path(self.run,0,"ema")) if self.step else None
        rows=payload["rows"]
        for scope in ("development","legacy"):
            stats=guard_report(self.inverse_stats[scope],self.coc_guards[scope],self.edges[scope])
            target=payload["summary"][scope]
            anchor=baseline["summary"][scope]["geometry"] if baseline else stats
            target["geometry"]=stats
            target["geometry_limits"]={k:dict(value=stats[k],baseline=anchor[k],max_relative_increase=v,
                    passed=stats[k]<=anchor[k]*(1+v)) for k,v in GUARDS.items()}
        for scope in COHORTS:
            report=payload["summary"][scope]
            report["inverse_depth_drift"]=percentiles(self.drift[scope])
            if scope in ("thin_structure","dev008"):
                report["foreground_coc_error_px"]=percentiles(self.thin_coc[scope])
            def selected(r):
                if scope in ("development","legacy"):
                    return r["cohort"]==scope
                return slug(r["scene"]) in (("dev003","dev004","dev005") if scope=="thin_structure" else ("dev008",))
            block=[r for r in rows if r["status"]=="evaluated" and selected(r)]
            report["unscored_cases"]=[dict(scene=r["scene"],case=r["case"],reason=r["status"])
                                      for r in rows if r["status"]!="evaluated" and selected(r)]
            initial=[self.by_start[r["scene"],r["case"]] for r in block]
            for region in REGIONS:
                report["metrics"][region]["Query_on_vs_start"]=deltas(
                    scores_by_scene(initial,region,"on"),scores_by_scene(block,region,"on"))
        payload["external_comparison"]=frozen_references()


@torch.inference_mode()
def zero(run):
    if report_path(run,0,"ema").is_file():
        start=load(report_path(run,0,"ema"))
        finalize_zero(run,start)
        return start
    start=zero_compare(run)
    joint=Joint(run,0)
    gpu_setup(dict(training=dict(max_process_gpu_gib=5)))
    geometry=load_geometry()
    lookup={(r["scene"],r["case"]):r for r in start["rows"]}
    panels={p["case"]:p for p in start["panels"]}
    output=report_path(run,0,"ema").parent
    for cohort,name,frame,unit,entry,cases in frames():
        item=("B",name,ovg.TRAIN_FRAMES.index(frame))
        x,y=sample(item)
        depth=geometry(x,protect_residual=True,render_heads=False)["depth1"][0].cpu().numpy()
        ref=ovg.b_edge_reference(item,y)
        joint.frame(cohort,name,frame,depth,x,y,ref)
        for case in cases:
            row=lookup[name,case["id"]]
            if row["status"]!="evaluated":
                continue
            k=joint.current["K"]
            regions=(new_regions(unit,case,ref["gt"],ref["valid"],k) if unit else legacy_masks(geometry_root(name),case))
            area=regions["in_focus" if row["focus_index"]==0 else "recoverable"]
            joint.case(row,unit,case,area)
            tag=f'{slug(name)}-{case["id"]}'
            if (slug(name),frame,row["focus_index"]) in FIXED:
                joint.panel(output,panels[tag],row,case,area)
    joint.summarize(start)
    write_json(report_path(run,0,"ema"),start)
    finalize_zero(run,start)
    return start


def finalize_zero(run,start):
    path=report_path(run,0,"model")
    path.parent.mkdir(parents=True,exist_ok=True)
    raw=dict(start,weights="model",zero_identical_initialization=True,
             panels=[dict(p,path="../ema/"+p["path"]) for p in start["panels"]])
    write_json(path,raw)
    observe(report_path(run,0,"ema").parent,start["panels"])
    observe(path.parent,raw["panels"])


def observe(output,panels):
    (output/"观察清单.md").write_text("# 固定面板观察清单\n\n"+"\n".join(
        f'- {p["case"]}。{p["observation"]} [面板]({p["path"]})' for p in panels)+"\n")
    (output/"index.html").write_text('<!doctype html><meta charset="utf-8"><title>阶段 3c</title>\n'+"\n".join(
        f'<figure><img width="900" loading="lazy" src="{escape(p["path"])}"><figcaption>{escape(p["case"])}</figcaption></figure>'
        for p in panels))


def adoption(start,trained):
    base=start["summary"]["development"]["far_hidden_fraction"]
    options=[]
    for value in trained:
        d=value["summary"]["development"]
        near,far=(d["metrics"][k]["Query_on_vs_start"] for k in REGIONS[:2])
        criteria=dict(geometry=all(v["passed"] for v in d["geometry_limits"].values()),
                      **{k:d["zero"][k]["all_passed"] for k in DB_FLOORS},
                      far_hidden=d["far_hidden_fraction"]>=.8*base)
        options.append(dict(step=value["step"],checks=criteria,eligible=all(criteria.values()),
                            near_pooled=near["on_db"],near_paired_median_db=near["median_db"],
                            far_paired_median_db=far["median_db"],far_hidden_fraction=d["far_hidden_fraction"]))
    candidates=[p for p in options if p["eligible"]]
    result=dict(selected_step=0,candidates=options,lowest_far_hidden_fraction=base*.8)
    if not candidates:
        return dict(result,reason="No trained EMA checkpoint passed all guards")
    best=max(candidates,key=lambda p:(p["near_pooled"],-p["step"]))
    passes=min(best["near_paired_median_db"],best["far_paired_median_db"])>=0
    return dict(result,selected_step=best["step"] if passes else 0,best_eligible_step=best["step"],
                paired_medians_pass=passes,
                interpretation={k:"基本持平" if abs(best[k])<.1 else "改善" if best[k]>0 else "下降"
                                for k in ("near_paired_median_db","far_paired_median_db")})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=Path,default=OUT)
    p.add_argument("--step",type=int,required=True)
    p.add_argument("--weights",choices=("ema","model"),default="ema")
    p.add_argument("--compare-only",action="store_true")
    args=p.parse_args()
    if args.compare_only:
        zero_compare(args.run)
    elif not args.step:
        zero(args.run)
    else:
        evaluate(args.run,args.step,args.weights,joint=Joint(args.run,args.step))
