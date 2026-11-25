"""Read-only comparison of center, front-mode and dominant-mode depth labels."""
import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch

import route3_ovg_data as ovg
import route3_v2data_stage1_visualize as gallery
from route3_v2data_B_batch import write_json
from route3_v2data_stage2_eval import frames, peeled_truth
from route3_v2data_stage2_train import sample
from route3_v2data_stage3a import gpu_setup
from route3_v2data_stage3a_prior_diagnostic import TARGETS, THIN_SCENES, thin_regions
from route3_v2data_stage3b import OUT, load_geometry, observed_views
from route3_s2a_data import query_input_512

OBSERVED_PRIORS = ("SHARP","DepthSplat","triangulation")
AREAS = ("gap_background","foreground","full")
CLASSES = ("M_bg","M_fg","opaque_reference")
STRATA = ("all_alpha","alpha_lt_half","alpha_ge_half")


def inverse(depth):
    depth = depth.astype(np.float32)
    return np.divide(1,depth,out=np.full_like(depth,np.nan),where=np.isfinite(depth)&(depth>0))


def stats(arrays):
    value = np.concatenate(arrays) if arrays else np.empty(0,np.float32)
    value = value[np.isfinite(value)]
    if not value.size:
        return dict(n=0,p50=None,p95=None,mean=None)
    a,b = np.percentile(value,(50,95))
    return dict(n=len(value),p50=float(a),p95=float(b),mean=float(value.mean(dtype=np.float64)))


class Samples:
    def __init__(self):
        self.frames = defaultdict(lambda:defaultdict(list))
        self.counts = defaultdict(lambda:defaultdict(int))

    def add(self,scene,path,arr):
        self.frames[scene][path].append(arr.astype(np.float32,copy=True))

    def count(self,scene,path,n):
        self.counts[scene][path] += int(n)

    @staticmethod
    def rows(sample,counts,names):
        keys = sorted(set().union(*(sample[n] for n in names)))
        values = {k:stats([v for n in names for v in sample[n].get(k,[])]) for k in keys}
        keys = sorted(set().union(*(counts[n] for n in names)))
        return dict(values=values,counts={k:sum(counts[n].get(k,0) for n in names) for k in keys})

    def report(self):
        names = sorted(self.frames)
        per_scene = {k:self.rows(self.frames,self.counts,[k]) for k in names}
        groups = {"development":[n for n in names if n.startswith("development-")],
                  "legacy":[n for n in names if n.startswith("metric-scene-")],
                  "thin_structure":[n for n in names if "dev"+n.split("-")[1] in THIN_SCENES],
                  "dev008":[n for n in names if n.startswith("development-008-")]}
        clips = {}
        for label,scene_names in groups.items():
            if not scene_names:
                continue
            combined = self.rows(self.frames,self.counts,scene_names)
            middle = {}
            for key in combined["values"]:
                fields = {field:[per_scene[s]["values"][key][field] for s in scene_names
                                  if key in per_scene[s]["values"] and per_scene[s]["values"][key][field] is not None]
                          for field in ("p50","p95","mean")}
                middle[key] = {f:float(np.median(v)) if v else None for f,v in fields.items()}
            clips[label] = dict(combined,scene_medians=middle,scenes=len(scene_names))
        return dict(cohorts=clips,per_scene=per_scene)


def visual(output,slug,center,errors,mbg,mfg,favor,rgb):
    support = mbg|mfg
    box = gallery.largest_error_crop(np.abs(errors)[...,None].repeat(3,axis=-1),np.zeros_like(rgb),support)
    signed = np.clip(np.nan_to_num(errors)/.3,-1,1)
    panel = np.stack((1-np.maximum(-signed,0),1-np.abs(signed),1-np.maximum(signed,0)),-1)
    masks = np.stack((mbg,mfg,np.zeros_like(mbg)),-1).astype(np.float32)
    chosen = np.where((support&favor)[...,None],np.array([1.,.55,.1]),rgb*.45)
    image = Image.new("RGB",(960,355),(18,23,31))
    draw = ImageDraw.Draw(image)
    for i,(name,arr) in enumerate(zip(("中心深度带符号逆深度差 ±0.3 m⁻¹","红 M_bg · 绿 M_fg","橙色更接近前模式"),(panel,masks,chosen))):
        tile = gallery.rgb_image(arr.astype(np.float32)).crop(box).resize((308,308),Image.Resampling.NEAREST)
        draw.text((i*320+6,5),name,font=gallery.font(15),fill="white")
        image.paste(tile,(i*320+6,34))
    path = output/f"{slug}-f030-depth-convention.jpg"
    image.save(path,quality=94)
    return dict(path=str(path),crop=box,center_visible=int(center.sum()))


def memory_evidence(views,frame,depth,gaps):
    target = next(v for v in views if v["frame"]==frame)
    packed,_ = query_input_512(views,dict(target,depth=depth),gaps,8,depth.size)
    pixels = packed["query_pixel"]
    confident = ((packed["features"][...,11]>0)&packed["candidate_valid"]).any(-1)
    return dict(gap_pixels=int(gaps.sum()),candidate_pixels=int(confident.sum()),
                candidate_fraction=int(confident.sum())/int(gaps.sum()) if gaps.any() else None,
                packed_pixels=len(pixels),memory_verified_support="Formal B uses input-only observed candidates with source confidence; its native path has no separate verified-memory index.")


@torch.inference_mode()
def run(output,smoke):
    output.mkdir(parents=True,exist_ok=True)
    gpu_setup(dict(training=dict(max_process_gpu_gib=3)))
    geom = load_geometry()
    acc,seed_rows,figures,frame_rows = Samples(),[],[],[]
    for cohort,name,frame,unit,_,_ in frames():
        key = name.split("-")[1]
        slug = "dev"+key if cohort=="development" else name.split("-")[2]
        if smoke and (slug,frame) not in (("dev005",30),("122104",30)):
            continue
        item = ("B",name,ovg.TRAIN_FRAMES.index(frame))
        x,y = sample(item)
        pred = geom(x,protect_residual=True,render_heads=False)
        ref = ovg.b_foreground_reference(name,frame)
        peel,_,known = peeled_truth(unit,name,frame)
        z = ref["gt"]
        labels = {k:y[k][0].cpu().numpy() for k in ("depth1","depth2","alpha","valid","valid_alpha","valid_dual","depth_boundary")}
        if any(v.shape != z.shape for v in (*labels.values(),ref["valid"],peel["depth_axial_m"],known)):
            raise ValueError(f"Center, peeled and subpixel labels are misaligned for {name} f{frame:03d}")
        if unit:
            center_file = unit/"labels"/f"f{frame:03d}-pinhole.npz"
            data = ovg.load(center_file,("K","depth_axial_m","center_visible"))
            if not np.allclose(data["K"],x["K"][0].cpu().numpy(),atol=1e-3,rtol=1e-5):
                raise ValueError(f"Center and subpixel camera mismatch for {name} f{frame:03d}")
        rho,front,back,peeled = map(inverse,(z,labels["depth1"],labels["depth2"],peel["depth_axial_m"]))
        center = ref["valid"]&np.isfinite(rho)
        a = labels["alpha"]
        mixed = labels["valid_alpha"]&center&(a>0)&(a<1)
        if not mixed.any() or not (np.isfinite(front[mixed])&np.isfinite(back[mixed])).all():
            raise ValueError(f"Absent mixed pixels or invalid subpixel modes for {name} f{frame:03d}")
        mbg = mixed&(np.abs(rho-back)<np.abs(rho-front))
        mfg = mixed&~mbg
        opaque = labels["valid_alpha"]&center&(a==1)
        other = np.where(a>=.5,front,back)
        prediction = pred["inverse1"][0].cpu().numpy()
        switched = torch.where(pred["alpha"]>=.5,pred["inverse1"],pred["inverse2"])[0].cpu().numpy()
        displacement = prediction-rho
        favor = np.abs(prediction-front)<np.abs(displacement)
        masks = {"full":center}
        if slug in TARGETS:
            masks.update(thin_regions(slug,unit,ref)[0])
        strata = (np.ones_like(center),a<.5,a>=.5)
        frame_rows.append(dict(scene=name,frame=frame,mixed_pixels=int(mixed.sum()),center_pixels=int(center.sum()),
                               center_selected_background=int(mbg.sum()),valid_dual_in_mixed=int((mixed&labels["valid_dual"]).sum())))
        for subset,roi in masks.items():
            acc.count(name,f"{subset}/valid_center_pixels",roi.sum())
            for label,pixels in zip(CLASSES,(mbg,mfg,opaque)):
                for stratum,filter_alpha in zip(STRATA,strata):
                    mask = roi&pixels&filter_alpha
                    prefix = f"{subset}/{label}/{stratum}"
                    acc.count(name,prefix+"/pixels",mask.sum())
                    acc.count(name,prefix+"/dual_pixels",(mask&labels["valid_dual"]).sum())
                    estimates = {"center":rho,"front_mode":front,"dominant_mode":other}
                    for target,depth in estimates.items():
                        acc.add(name,prefix+"/inverse1_abs_to_"+target,np.abs(prediction[mask]-depth[mask]))
                        both = mask&labels["valid_dual"]
                        acc.add(name,prefix+"/alpha_dual_abs_to_"+target,np.abs(switched[both]-depth[both]))
                    acc.add(name,prefix+"/signed_to_center",displacement[mask])
                    acc.add(name,prefix+"/closer_to_front",favor[mask])
        prior_z = x["prior_depth"][0].cpu().numpy()
        confidence = x["prior_conf"][0].cpu().numpy()
        weights = pred["prior_weights"][0].cpu().numpy()
        fused = pred["fused_depth"][0].reciprocal().cpu().numpy()
        deeper = known&center&np.isfinite(peeled)&(peeled<rho)
        for subset,roi in masks.items():
            for idx,prior in enumerate(OBSERVED_PRIORS):
                estimate = inverse(prior_z[idx])
                m = roi&(confidence[idx]>0)&np.isfinite(estimate)
                n = m&deeper
                prefix = f"{subset}/prior/{prior}"
                visible,hidden = np.abs(estimate[n]-rho[n]),np.abs(estimate[n]-peeled[n])
                acc.add(name,prefix+"/visible_error",np.abs(estimate[m]-rho[m]))
                acc.add(name,prefix+"/fused_visible_error",np.abs(fused[m]-rho[m]))
                acc.add(name,prefix+"/paired_visible_error",visible)
                acc.add(name,prefix+"/paired_peeled_error",hidden)
                acc.add(name,prefix+"/closer_to_peeled",hidden<visible)
                for j,source in enumerate(OBSERVED_PRIORS):
                    acc.add(name,prefix+f"/source_{source}_confidence",confidence[j,m])
                    acc.add(name,prefix+f"/source_{source}_weight",weights[j,m])
                    acc.add(name,prefix+f"/paired_source_{source}_weight",weights[j,n])
        if "gap_background" in masks:
            gaps = masks["gap_background"]
            count,components = cv2.connectedComponents(gaps.astype(np.uint8),connectivity=8)
            valid_ds = (confidence[1]>0)&np.isfinite(prior_z[1])&(prior_z[1]>0)
            valid_tri = (confidence[2]>0)&np.isfinite(prior_z[2])&(prior_z[2]>0)
            seeds = gaps&(valid_ds| (valid_tri&(confidence[2]>=1)))
            good = len(np.unique(components[seeds]))
            seed_rows.append(dict(scene=name,frame=frame,component_count=count-1,
                                  components_with_seed=good,components_without_seed=count-1-good,
                                  seeds=int(seeds.sum()),DepthSplat_not_admitted=int((gaps&~valid_ds).sum()),
                                  triangulation_no_accepted_match=int((gaps&~valid_tri).sum()),
                                  both_unavailable=int((gaps&~valid_ds&~valid_tri).sum()),
                                  memory=memory_evidence(observed_views(unit),frame,pred["depth1"][0].cpu().numpy(),gaps)))
            if frame==30:
                row = visual(output,slug,center,displacement,mbg,mfg,favor,
                             gallery.display(x["rgb"][0].permute(1,2,0).cpu().numpy()))
                figures.append(dict(scene=name,**row))
        print("DEPTH_CONVENTION",name,frame,int(mixed.sum()),flush=True)
    payload = acc.report()
    for v in [*payload["per_scene"].values(),*(r for r in payload["cohorts"].values())]:
        counts = v["counts"]
        v["group_fractions"] = {k:count/counts[k.split("/")[0]+"/valid_center_pixels"]
                                for k,count in counts.items() if k.endswith("/pixels")
                                and counts[k.split("/")[0]+"/valid_center_pixels"]>0}
    payload.update(frame_rows=frame_rows,seeds=seed_rows,figures=figures,test_cases_used=0,
                   protocol="Native center GT vs front and dominant subpixel modes. M_bg uses strictly closer inverse depth to depth2. Positive signed error means closer to camera. Paired peeled errors restrict to a valid deeper peeled hit. Priors are relative to the same axial-meters camera. Optical mixture alpha is read from labels only.",
                   unavailable_reason_detail="B released priors do not retain per-pixel raw support counts or rejected match causes. DS admitted confidence encodes support/7 but zero conflates support<2 and invalid/out-of-range depth. Triangulation zero denotes no accepted triangulated result, with no split by rejection cause.",
                   frozen_ovg=True)
    write_json(output/"report.json",payload)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=OUT/"depth-convention")
    parser.add_argument("--smoke",action="store_true")
    args = parser.parse_args()
    run(args.output,args.smoke)
