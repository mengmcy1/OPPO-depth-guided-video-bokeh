"""Read-only two-arm C7 split comparison on the archived 2a 10k release."""
import argparse
from collections import defaultdict
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch

import route3_ovg_data as ovg
import route3_v2data_stage1_visualize as gallery
import route3_v2data_stage2_train as stage2
import route3_v2data_stage2_visualize_split5 as split5
from route3_v2data_B_batch import sha256, write_json
from route3_v2data_B_plan import load
from route3_v2data_query_retrain import distribution
from route3_v2data_stage1_evaluate import FrameQueries, p2_focus, pooled
from route3_v2data_stage2_eval import bands,f2_for,optical_target,render_offsets
from route3_v2data_stage3a import gpu_setup
from route3_v2data_stage3b import OUT, hidden_cloud, optical_band, pack_frame
from src.aperture_renderer import aperture_disk
from src.c1_bokehme import region_sse
from src.footprint_order import push_behind_split_background
from src.layered_aperture import merge_layers
from src.ovg_optics import fixed_visible_surfels, raster

SOURCE = Path("/root/autodl-tmp/route3-cache/visual-20261123/second-pass-pushback-baseline/split5/2a-10k/metrics.json")
VISUALS = {("dev001",30,"near"),("dev005",30,"far"),("122104",30,"far")}
ARMS = (("pred_none","C6"),("gt_none","C7"))


def score(image,target,masks):
    rgb = gallery.display(image.cpu().numpy())
    return {k:region_sse(rgb,target,v) for k,v in masks.items()},rgb


def panel(path,slug,focus,rgb,truth,box):
    error = gallery.scalar_image(np.abs(rgb[1]-truth).mean(-1),.25,cv2.COLORMAP_INFERNO)
    pictures = [gallery.rgb_image(v) for v in (*rgb,truth)]+[error]
    image = Image.new("RGB",(1040,310),(18,23,31))
    d = ImageDraw.Draw(image)
    d.text((8,5),f"{slug} · P2 {focus:.4f} m",font=gallery.font(20),fill="white")
    for i,(name,picture) in enumerate(zip(("不拆分","拆分","Cycles GT","拆分绝对误差 · 0–0.25"),pictures)):
        x = i*260+5
        d.text((x,34),name,font=gallery.font(14),fill="white")
        image.paste(picture.crop(box).resize((250,250),Image.Resampling.NEAREST),(x,55))
    image.save(path,quality=94)
    return dict(path=str(path),crop_xyxy=box)


def gt_arm(x,y,geom,ref,gt_mixed,labels):
    center = stage2.tensor(ref["gt"])
    center_valid = stage2.tensor(ref["valid"])&torch.isfinite(center)&(center>0)
    base = torch.where(center_valid,center,geom["depth1"][0])
    a,b = y["depth1"][0],y["depth2"][0]
    valid_dual = y["valid_dual"][0]&center_valid&torch.isfinite(a)&torch.isfinite(b)&(a>0)&(b>a)
    shared = valid_dual&torch.as_tensor(gt_mixed,device=center.device)
    front = torch.where(shared,a,base)
    # Invalid depth2 never enters the background geometry or O1 ordering.
    rear = torch.where(shared,b,base)
    fg = stage2.tensor(labels["foreground"]).permute(2,0,1)[None]
    foreground = dict(geom,depth1=front[None],opacity=y["alpha"],foreground=fg)
    first = dict(color=labels["background"],depth=rear.cpu().numpy(),q=shared.cpu().numpy().astype(np.float32))
    geometry = fixed_visible_surfels({"depth1":base[None]},x)
    c7 = merge_layers(stage2.visible(foreground,x,shared,"2a"),stage2.background(foreground,x,shared,first))
    return base,front,rear,valid_dual,shared,geometry,c7


def row_metrics(records,method,scope,metric):
    grouped = defaultdict(list)
    for row in records:
        grouped[row["scene"]].append(row["methods"][method][scope][metric])
    return {name:pooled(scores) for name,scores in grouped.items()}


def differences(rows,a,b,scope,metric):
    before,after = (row_metrics(rows,m,scope,metric) for m in (a,b))
    base,end = (pooled(v.values()) for v in (before,after))
    paired = {}
    for name in before:
        p,q = before[name],after[name]
        if p["pixels"]!=q["pixels"]:
            raise ValueError(f"C7 support changed for {name}/{scope}/{metric}")
        if p["pixels"]:
            paired[name] = q["psnr_db"]-p["psnr_db"]
    return dict(before=base,after=end,pooled_delta_db=end["psnr_db"]-base["psnr_db"] if base["pixels"] else None,
                per_scene_db=paired,**distribution(list(paired.values())))


def summarize(rows):
    result = {}
    for cohort in ("development","legacy"):
        subset = [r for r in rows if r["cohort"]==cohort]
        if not subset:
            continue
        scores = {}
        for a,b in (*ARMS,("C6","C7")):
            scopes = ("common_valid_dual",) if (a,b)==("C6","C7") else ("full","common_valid_dual")
            scores[b+"-"+a] = {scope:{metric:differences(subset,a,b,scope,metric)
                                      for metric in ("boundary_2px","full_image","zero_combined")}
                                for scope in scopes}
        result[cohort] = scores
    return result


@torch.inference_mode()
def run(output,smoke):
    output.mkdir(parents=True,exist_ok=True)
    gpu_setup(dict(training=dict(max_process_gpu_gib=7)))
    start = time.monotonic()
    expected = load(SOURCE)
    checkpoint = Path(expected["checkpoint"])
    if sha256(checkpoint)!=expected["checkpoint_sha256"]:
        raise ValueError("Archived C6 appearance checkpoint differs")
    model = stage2.Model(checkpoint.parents[1])
    model.heads.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=True)["ema"])
    model.heads.eval()
    archived = {r["file"]:r for r in expected["rows"] if not r.get("error")}
    rows,figures,skips = [],[],[]
    for plan in gallery.planned_frames():
        accepted = [(i,archived[r["file"]]) for i,r in enumerate(plan["files"]) if r["file"] in archived]
        unit,name,frame,entry = (plan[k] for k in ("unit","name","frame","entry"))
        slug = plan["files"][0]["short"]
        if smoke and (slug,frame)!=("dev005",30):
            continue
        if not accepted:
            continue
        state = split5.frame_setup(model,plan,"2a",methods=("none","C6"))
        x,y,g,ref,_,subjects,pred_clouds,_,_,gt_mixed,rgb,_,_=state
        item = split5.item_for(plan)
        labels = stage2.appearance_labels(item)
        base,front,rear,dual,common,gt_none,c7 = gt_arm(x,y,g,ref,gt_mixed,labels)
        gt_shared = common.cpu().numpy()
        center_np = base.cpu().numpy()
        pred_z = g["depth1"][0].cpu().numpy()
        K = x["K"][0]
        common_masks = dict(full_image=gt_shared,boundary_2px=gt_shared & bands(ref,y)[0],zero_combined=gt_shared)
        full = np.ones_like(gt_shared)
        full_masks = dict(full_image=full,boundary_2px=bands(ref,y)[0],zero_combined=full)
        pinhole = gallery.display(rgb)
        first = split5.first_observation(model,plan,g)
        for idx,old in accepted:
            case = plan["cases"][idx]
            focus_p = old["actual_focus_m"]["C6"]
            focus_g = p2_focus(center_np,subjects[idx])
            expected_focus = p2_focus(pred_z,subjects[idx])
            if expected_focus is None or abs(focus_p-expected_focus)>1e-6:
                raise ValueError(f"Cannot reproduce archived P2 focus for {old['file']}")
            if focus_g is None:
                skips.append(dict(case=old["file"],reason="no GT center focus support"))
                continue
            offsets = aperture_disk(64,.05/(2*case["fstop"]),device=K.device,dtype=K.dtype)
            center = K.new_zeros((1,2))
            truth = optical_target(unit,name,case)
            row = dict(scene=name,cohort="development" if unit else "legacy",case=case["id"],frame=frame,
                       valid_dual=int(dual.sum()),common_split=int(common.sum()),original_split=int(gt_mixed.sum()),
                       sensor_pixels=full.size,methods={})
            for arm,focus,z,surfaces in (
                ("prediction",focus_p,pred_z,(("pred_none",pred_clouds["none"]),("C6",pred_clouds["C6"]))),
                ("GT",focus_g,center_np,(("gt_none",gt_none),("C7",c7))),
            ):
                if arm=="prediction":
                    hidden = model.second(item,z,focus,case["fstop"])
                else:
                    views,memory = model.context("B",name)
                    gt_band = optical_band(center_np,K,focus,case["fstop"])
                    store = FrameQueries(views,pack_frame(views,frame,center_np,[gt_band],memory))
                    query_row = dict(input="c7",frame=frame,case=dict(case,focus_m=focus))
                    hidden = hidden_cloud(model.query.network,store,query_row,K,base,gt_band)
                bg = f2_for(unit,entry,frame,x,focus,offsets)
                bg0 = f2_for(unit,entry,frame,x,focus,center)
                photos = []
                for method,appearance in surfaces:
                    h = hidden
                    if method=="C6":
                        h = push_behind_split_background(h,stage2.tensor(gt_mixed),stage2.tensor(first["depth"]),K)
                    if method=="C7":
                        # Reuse this arm's observation candidates, recompute their
                        # order against the actual C7 front and GT rear layers.
                        h = hidden_cloud(model.query.network,store,query_row,K,base,gt_band,
                                         placement_depth=front)
                        h = push_behind_split_background(h,common,rear,K)
                    cloud = merge_layers(appearance,h)
                    optical = render_offsets(cloud,K,focus,offsets,bg)
                    zero = raster(cloud,K,focus,center,512,512,bg0)[0]
                    scored = dict(focus_m=focus)
                    for scope,masks in (("full",full_masks),("common_valid_dual",common_masks)):
                        scored[scope],photo = score(optical,truth,{k:v for k,v in masks.items() if k!="zero_combined"})
                        z,_ = score(zero,pinhole,{"zero_combined":masks["zero_combined"]})
                        scored[scope].update(z)
                    row["methods"][method] = scored
                    photos.append(photo)
                if (slug,frame,old["focus"]) in VISUALS:
                    image = output/f"{slug}-f{frame:03d}-{old['focus']}-{arm}.jpg"
                    figures.append(dict(arm=arm,case=old["file"],**panel(image,image.stem,focus,photos,truth,old["crop_xyxy"])))
            for old_key,new_key in (("none","pred_none"),("C6","C6")):
                prior = old["metrics"][old_key]["full"]
                now = row["methods"][new_key]["full"]
                row.setdefault("archived_replay",{})[new_key] = {
                    k:dict(pixel_delta=now[k]["pixels"]-prior[k]["pixels"],sse_delta=now[k]["sse"]-prior[k]["sse"])
                    for k in ("boundary_2px","zero_combined")}
            rows.append(row)
            print("C7",old["file"],row["common_split"],flush=True)
    protocol = ("Two P2 arms with matching focus and observed F2 within each arm. "
                "GT arm reconstructs second-pass candidates in center geometry, then reapplies order to C7 front and rear. "
                "C7 adds GT depth1/depth2 only on valid-dual GT mixed color support. Outside that support both GT methods share "
                "pinhole center depth, with frozen predicted depth as a fallback only on invalid center hits. "
                "Cross-arm C7/C6 is reported only on common valid-dual split pixels.")
    result = dict(protocol=protocol,checkpoint=str(checkpoint),checkpoint_sha256=expected["checkpoint_sha256"],
                  archived_metrics=str(SOURCE),rows=rows,summary=summarize(rows),figures=figures,skips=skips,
                  test_cases_used=0,seconds=time.monotonic()-start)
    write_json(output/"report.json",result)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=OUT/"c7")
    parser.add_argument("--smoke",action="store_true")
    a=parser.parse_args()
    run(a.output,a.smoke)
