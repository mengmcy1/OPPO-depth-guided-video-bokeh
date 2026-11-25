"""Render the fixed E-L depth and K64 V0 comparison panels."""
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch

from route3_v2data_E_cache import cache_arrays,cache_path,setup
from route3_v2data_E_labels import (DEV_PANEL,RUN,core_labels,load,save_json,unit_paths)
from route3_v2data_loader import b_optical_cases
from route3_v2data_module2_infer import competitor_state,light,place_hidden,yield_gpu
import route3_v2data_stage1_visualize as gallery
from route3_v2data_optics import observed_background
from route3_v2data_stage1_evaluate import near_subject,new_regions,p2_focus,pooled
from route3_v2data_stage2_eval import f2_for,frames as dev_frames
from src.aperture_renderer import aperture_disk
from src.c1_bokehme import region_sse
from src.layered_aperture import pixel_layer

FRAME=30
OUT=RUN/"panels"
VIEWS=("OVG","E","GT")


def crop(mask,side=96):
    if not mask.any():
        raise ValueError("A fixed comparison has no rod or gap in its native target")
    integral=cv2.integral(mask.astype(np.uint8))
    total=integral[side:,side:]-integral[:-side,side:]-integral[side:,:-side]+integral[:-side,:-side]
    r,c=np.unravel_index(total.argmax(),total.shape)
    return int(c),int(r),int(c+side),int(r+side)


def rgb(value):
    return gallery.rgb_image(value)


def scalar(value,hi):
    return gallery.scalar_image(value,hi,cv2.COLORMAP_VIRIDIS)


def palette(labels,colors):
    return Image.fromarray(np.asarray(colors,dtype=np.uint8)[labels.astype(np.int32)])


def save_depth(name,bundle,raw,crops,hi,*,output=OUT):
    photos=[
        ("RGB",rgb(gallery.display(bundle["rgb"].transpose(1,2,0)))),
        ("GT 中心",scalar(raw["center"],hi)),
        ("OVG ρ₁",scalar(raw["rho1"],hi)),
        ("E ρc",scalar(raw["E_decoded"],hi)),
        ("E 前分量",scalar(raw["E_front"],hi)),
        ("E 后分量",scalar(raw["E_back"],hi)),
        ("混合权重 w",scalar(raw["w"],1)),
        ("众数 前／后",palette(raw["selected_back"],[(238,143,61),(65,196,204)])),
        ("杆／缝／正常",palette(raw["zone"],[(233,111,79),(87,203,255),(64,69,79)])),
        ("βb ρ₂／DS／三角",palette(raw["mv_source"],[(64,69,79),(107,222,144),(192,133,238)])),
    ]
    if "gate" in raw:
        photos.insert(7,("门控 g",scalar(raw["gate"],1)))
    tile=184
    pitch=tile+7
    fig=Image.new("RGB",(len(photos)*pitch,77+3*(tile+26)),(16,21,29))
    text=ImageDraw.Draw(fig)
    text.text((6,3),f"{name} · f030 · 逆深度 0–{hi:.3f} m⁻¹（全轮共用色标）",font=gallery.font(18),fill="white")
    text.text((6,26),"上：整图 · 中：杆放大 · 下：缝放大",font=gallery.font(15),fill="white")
    for k,(title,photo) in enumerate(photos):
        left=k*pitch
        text.text((left+2,49),title,font=gallery.font(13),fill="white")
        full=photo.copy()
        if k==0:
            mark=ImageDraw.Draw(full)
            for box,color in zip(crops,((252,170,83),(87,203,255))):
                mark.rectangle(box,outline=color,width=2)
        for i,img in enumerate((full, *(photo.crop(box) for box in crops))):
            filt=Image.Resampling.BOX if i==0 else Image.Resampling.NEAREST
            fig.paste(img.resize((tile,tile),filt),(left,72+i*(tile+26)))
    path=output/f"{name}-f030-depth.jpg"
    fig.save(path,quality=95)
    return path


def save_optical(name,case,truth,renders,crops,scores):
    photos=[(key,rgb(renders[key]),gallery.scalar_image(np.abs(renders[key]-truth).mean(-1),.25,cv2.COLORMAP_INFERNO))
            for key in VIEWS]
    photos.append(("Cycles",rgb(truth),Image.new("RGB",(512,512))))
    tile,pitch=212,221
    fig=Image.new("RGB",(len(photos)*pitch,75+4*(tile+17)),(16,21,29))
    text=ImageDraw.Draw(fig)
    text.text((6,3),f"{name} · {case['id']} · P2 f/1.4 K64",font=gallery.font(17),fill="white")
    text.text((6,28),"上起：整图、杆、缝、显示 RGB 误差 0–0.25；角标为全图 PSNR",font=gallery.font(13),fill="white")
    for i,(key,photo,error) in enumerate(photos):
        label=key if key=="Cycles" else f"V0 {key} · {pooled([scores[key]['full']])['psnr_db']:.2f} dB"
        text.text((i*pitch+2,49),label,font=gallery.font(14),fill="white")
        for j,view in enumerate((photo,*(photo.crop(box) for box in crops),error)):
            fig.paste(view.resize((tile,tile),Image.Resampling.NEAREST if j in (1,2) else Image.Resampling.BOX),
                      (i*pitch,72+j*(tile+17)))
    path=OUT/f"{name}-{case['id']}-optics.jpg"
    fig.save(path,quality=95)
    return path


def registered_cases(inventory):
    paths=unit_paths(inventory)
    evaluation={}
    for name in inventory["overfit"]:
        specs=[row for row in b_optical_cases(paths[name]) if row["frame"]==FRAME and row["fstop"]==1.4]
        output=[]
        for spec in specs:
            case=spec["truth"].name.removesuffix("-optics.npz")
            focus_index=int(case.split("-")[1].removeprefix("focus"))
            output.append(dict(id=case,frame=FRAME,focus_index=focus_index,focus_m=spec["focus"],
                               fstop=spec["fstop"],truth=str(spec["truth"])))
        if len(output)!=2:
            raise ValueError(f"Missing two f/1.4 reference cases in {name}")
        evaluation[name]=output
    for _,name,f,unit,entry,cases in dev_frames():
        if f==FRAME and name.startswith("development-") and name.split("-")[1] in DEV_PANEL:
            evaluation[name]=[dict(case,truth=str(unit/"labels"/(case["id"]+"-optics.npz"))) for case in cases]
    if len(evaluation)!=9:
        raise ValueError("The five training and four development frame panels are required")
    return evaluation


@torch.inference_mode()
def render():
    inventory=load(RUN/"inventory.json")
    paths=unit_paths(inventory)
    cases=registered_cases(inventory)
    frame_maps={n:cache_arrays(RUN/"panels-data"/f"{n}-f030.npz") for n in cases}
    hi=float(np.quantile(np.concatenate([m["center"][m["center"]>0] for m in frame_maps.values()]),.99))
    setup()
    before=competitor_state()
    started=time.monotonic()
    waited=yield_gpu()
    OUT.mkdir(parents=True,exist_ok=True)
    result=[]
    for name,pair in cases.items():
        waited+=yield_gpu()
        unit=paths[name]
        training=name in inventory["overfit"]
        raw=frame_maps[name]
        bundle=cache_arrays(cache_path("ovg",name,FRAME))
        y=core_labels(unit,FRAME,inventory["metadata"][name])
        ref=dict(gt=y["depth_axial_m"].astype(np.float32), valid=y["visible"]&y["valid"],
                 object_id=y["object_id"],foreground=inventory["metadata"][name]["foreground"])
        k=np.asarray(y["K"],dtype=np.float32)
        K=torch.as_tensor(k,device="cuda")
        color=torch.as_tensor(bundle["rgb"].transpose(1,2,0),device="cuda")
        inputs={"K":K[None],"rgb":color.permute(2,0,1)[None]}
        pos=[crop(raw["zone"]==i) for i in (0,1)]
        path=save_depth(name,bundle,raw,pos,hi)
        centers=np.where(ref["valid"],raw["center"],raw["rho1"])
        native={"OVG":1/raw["rho1"],"E":1/raw["E_decoded"],"GT":1/centers}
        row=dict(scene=name,frame=FRAME,cohort="training" if training else "development",
                 depth_panel=path.name,rod_crop=pos[0],gap_crop=pos[1],optical=[])
        for case in pair:
            nominal=case["focus_m"]
            subject=near_subject(ref["gt"],ref["valid"],k,nominal)
            focus={key:p2_focus(z,subject) for key,z in native.items()}
            if any(v is None for v in focus.values()):
                raise ValueError(f"Fixed P2 focus has no nominal subject in {name}/{case['id']}")
            truth=gallery.display(cache_arrays(Path(case["truth"]))["rgb_linear"].astype(np.float32))
            regions=new_regions(unit,case,ref["gt"],ref["valid"],k)
            primary="in_focus" if case["focus_index"]==0 else "recoverable"
            support={"full":np.ones((512,512),bool), "focus_region":regions[primary],
                     "rod":raw["zone"]==0,"gap":raw["zone"]==1}
            rays=aperture_disk(64,.05/(2*case["fstop"]),device=K.device,dtype=K.dtype)
            scores,renders,counts={},{},{}
            for key,value in native.items():
                waited+=yield_gpu()
                z=K.new_tensor(value)
                visible=pixel_layer(color,z,K,torch.eye(4,device=K.device,dtype=K.dtype))
                hidden,counts[key]=place_hidden(unit,name,FRAME,case,ref,z,z,K,return_stats=True)
                if training:
                    fill=observed_background(dict(tiers="B",unit=unit,frame=FRAME),inputs,focus[key],rays)
                else:
                    fill=f2_for(unit,None,FRAME,inputs,focus[key],rays)
                rendered=gallery.display(light(visible,hidden,K,focus[key],rays,fill).cpu().numpy())
                renders[key]=rendered
                scores[key]={k:region_sse(rendered,truth,m) for k,m in support.items()}
                del visible,hidden,fill
            image=save_optical(name,case,truth,renders,pos,scores)
            row["optical"].append(dict(case=case["id"],focus_index=case["focus_index"],
                focus_m=focus,panel=image.name,scores=scores,placed=counts))
        result.append(row)
        print("E_PANELS",name,flush=True)
    save_json(RUN/"panels.json",dict(rows=result,shared_inverse_color_range=[0.,hi],
              elapsed_seconds=time.monotonic()-started,gpu_wait_seconds=waited,
              competitor_start=before,competitor_end=competitor_state()))


if __name__=="__main__":
    render()
