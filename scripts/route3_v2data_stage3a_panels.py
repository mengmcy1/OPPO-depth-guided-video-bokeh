"""Fixed and paired-extreme Stage 3a optical, depth and CoC panels."""
from html import escape
import json

import cv2
import numpy as np
from PIL import Image, ImageDraw

import route3_v2data_stage1_visualize as gallery
from route3_v2data_baseline_dev3_extended_report import box_for
from route3_v2data_B_batch import write_json
from route3_v2data_stage3a_prior_diagnostic import diagnostic_markdown


def short(name):
    return ("dev" if name.startswith("development-") else "")+name.split("-")[1 if name.startswith("development-") else 2]


def review_rows(rows, development, fixed):
    cases = [r for block in rows.values() for r in block if r.get("artifact")]
    chosen = {}
    unavailable = []
    for scene,frame,focus in fixed:
        row = next((r for r in cases if (short(r["scene"]),r["frame"],r["focus_index"])==(scene,frame,focus)),None)
        if row is not None:
            chosen[(row["scene"],row["case"])] = (row,"固定")
        else:
            unavailable.append(dict(scene=scene,frame=frame,focus_index=focus))
    ranked = development["paired_vs_start"]["near_in_focus"]["scenes"]
    ordered = sorted(ranked,key=lambda k:(ranked[k]["delta_db"],k))
    for group, scenes in (("近焦配对较低",ordered[:2]),("近焦配对较高",ordered[-2:])):
        for scene in scenes:
            methods=sorted((r for r in cases if r["scene"]==scene and r["focus_index"]==0),
                           key=lambda r:(r["frame"]!=30,r["frame"]))
            if methods:
                row=methods[0]
                key=(row["scene"],row["case"])
                detail=f'{group} {ranked[scene]["delta_db"]:+.3f} dB'
                label=chosen[key][1]+" · "+detail if key in chosen else detail
                chosen[key]=(row,label)
    return list(chosen.values()),unavailable


def diagnostics(new,old,focus,previous):
    mask = new["fg"] & new["near_or_far"]
    valid = mask & (new["depth"]>0) & (old["depth"]>0)
    z0,z1 = (1/d["depth"][valid].astype(np.float64) for d in (old,new))
    edge = new["boundary"]
    def variation(img):
        image=img.astype(np.float32).mean(-1)
        gx=cv2.Sobel(image,cv2.CV_32F,1,0,ksize=3)
        gy=cv2.Sobel(image,cv2.CV_32F,0,1,ksize=3)
        return float(np.hypot(gx,gy)[edge].mean()) if edge.any() else None
    return dict(
        subject_pixels=int(valid.sum()),
        depth_inverse_spread=[float(z.std()) if z.size else None for z in (z0,z1)],
        inverse_distance_to_focus=[float(np.abs(z-1/f).mean()) if z.size else None
                                   for z,f in ((z0,previous),(z1,focus))],
        foreground_coc_mae_px=[float(d["coc"][valid].mean(dtype=np.float64)) if valid.any() else None
                               for d in (old,new)],
        mean_edge_gradient={k:variation(v) for k,v in
                            (("reference",new["truth"]),("start",old["current"]),("current",new["current"]))},
        interpretation="Lower inverse-depth spread can indicate flattening; lower distance to focus with greater GT CoC error flags focus drift. Edge gradient is sharpness evidence, not a width measurement.",
    )


def draw(path,row,previous,new,old,old_focus,diagnostic=None):
    tile,gap,left=235,10,12
    cells=[("输入 RGB",new["input"]),("LDR 参考",new["truth"]),("当前 F2",new["f2"]),
           ("当前 GT 替身",new["current"]),("起点 GT 替身",old["current"])]
    truth=new["truth"]
    boxes=(box_for(new["current"],truth,new["boundary"]),
           box_for(new["current"],truth,new["near_or_far"] & ~new["boundary"]))
    labels=["全图","边界放大","分区放大","显示 RGB 绝对误差","零光圈绝对误差","前层深度","前景 CoC 误差"]
    total = len(cells) + (2 if diagnostic is not None else 0)
    pic=Image.new("RGB",(left*2+total*(tile+gap)-gap,80+len(labels)*(tile+29)),(18,23,31))
    d=ImageDraw.Draw(pic)
    d.text((left,6),f'{row["scene"]} · {row["case"]}',font=gallery.font(22),fill="white")
    d.text((left,34),f'P2 对焦距离 · 当前 {row["focus_m"]:.3f} m · 起点 {old_focus:.3f} m',
           font=gallery.font(16),fill="white")
    valid = new["gt_depth"]>0
    top=float(np.percentile(new["gt_depth"][valid],99)) if valid.any() else 1.
    metric=next(r for r in ("near_in_focus","far_recoverable")
                if r in row["scores"]["stage3"]["metrics"])
    scores=[None,None,row["f2"]["psnr"],row["scores"]["stage3"]["metrics"][metric]["psnr"],
            previous["scores"]["stage3"]["metrics"][metric]["psnr"]]
    for j,label in enumerate(labels):
        y=69+j*(tile+29)
        d.text((left,y),label,font=gallery.font(14),fill=(196,205,220))
    for i,(name,data) in enumerate(cells):
        x=left+i*(tile+gap)
        psnr="" if scores[i] is None else f" · {scores[i]:.2f} dB"
        d.text((x,52),name+psnr,font=gallery.font(14),fill="white")
        source=gallery.rgb_image(data.astype(np.float32))
        full=source.copy()
        if i==1:
            p=ImageDraw.Draw(full)
            for box,color in zip(boxes,((255,148,70),(70,216,219))):
                p.rectangle(box,outline=color,width=2)
            if diagnostic is not None:
                for box in diagnostic["bounding_boxes"]:
                    y0,x0,y1,x1 = box["yxyx_exclusive"]
                    p.rectangle((x0,y0,x1-1,y1-1),outline=(236,98,238),width=2)
        abs_error=np.abs(data.astype(np.float32)-truth).mean(-1)
        pictures=[full, *(source.crop(box) for box in boxes),
                  gallery.scalar_image(abs_error,.25,cv2.COLORMAP_INFERNO)]
        selected=old if i==4 else new
        pictures.append(gallery.scalar_image(selected["zero_error"],.05,cv2.COLORMAP_INFERNO) if i>=3 else None)
        depth=selected["depth"] if i>=3 else new["gt_depth"] if i==1 else None
        pictures.append(gallery.scalar_image(depth,top,cv2.COLORMAP_VIRIDIS) if depth is not None else None)
        pictures.append(gallery.scalar_image(np.where(selected["fg"],selected["coc"],0),2.,cv2.COLORMAP_INFERNO)
                        if i>=3 else None)
        for j,layer in enumerate(pictures):
            if layer is not None:
                y=88+j*(tile+29)
                pic.paste(layer.resize((tile,tile),Image.Resampling.NEAREST if j in (1,2) else Image.Resampling.BOX),(x,y))
    if diagnostic is not None:
        maps = diagnostic["maps"]
        with np.load(maps) as arr:
            extra = [
                (f"先验融合深度 · 0–{top:.1f} m",gallery.scalar_image(arr["fused_depth"],top,cv2.COLORMAP_VIRIDIS)),
                ("残差衰减系数 · 0–1",gallery.scalar_image(arr["attenuation"],1.,cv2.COLORMAP_VIRIDIS)),
            ]
        for i,(name,layer) in enumerate(extra,len(cells)):
            x=left+i*(tile+gap)
            d.text((x,52),name,font=gallery.font(14),fill="white")
            for j,part in enumerate((layer,*(layer.crop(box) for box in boxes))):
                pic.paste(part.resize((tile,tile),Image.Resampling.NEAREST),(x,88+j*(tile+29)))
    pic.save(path,quality=94,subsampling=0)


def panels(output,start_dir,rows,development,fixed,*,diagnostic=None):
    start = json.loads((start_dir/"cases.json").read_text())
    start_rows = {(r["scene"],r["case"]):r for block in start.values() for r in block if r.get("artifact")}
    selection,unavailable=review_rows(rows,development,fixed)
    folder=output/"panels"
    folder.mkdir(exist_ok=True)
    results=[]
    html=['<!doctype html><meta charset="utf-8"><title>阶段 3a 对照图</title>',
          '<h1>阶段 3a 对照图</h1><p>光学误差色标 0–0.25，零光圈 0–0.05，CoC 0–2 px。</p>']
    markdown=["# 阶段 3a 图像观察清单","",
              "深度摊平用逆深度标准差描述。向焦平面移动用平均逆深度到实际焦点的距离描述。"
              "同时列真值 CoC 误差和光学边缘梯度以交叉查看，边缘梯度不换算为虚化宽度。"
              "所有数字按起点／当前排列。",""]
    for row,label in selection:
        key=(row["scene"],row["case"])
        previous=start_rows.get(key)
        if previous is None:
            unavailable.append(dict(scene=key[0],case=key[1],reason="no matched start image"))
            continue
        name=f"{short(row['scene'])}-{row['case']}.jpg"
        view = None
        if diagnostic is not None and (short(row["scene"]),row["frame"],row["focus_index"]) in map(tuple,fixed):
            data = diagnostic["frames"].get(f'{short(row["scene"])}-f{row["frame"]:03d}')
            if data is not None:
                view = dict(bounding_boxes=data["bounding_boxes"],maps=output/data["artifact"])
        with np.load(output/row["artifact"]) as new, np.load(start_dir/previous["artifact"]) as old:
            stats=diagnostics(new,old,row["focus_m"],previous["focus_m"])
            draw(folder/name,row,previous,new,old,previous["focus_m"],view)
        results.append(dict(case=row["case"],scene=row["scene"],file=f"panels/{name}",reason=label,observations=stats))
        def values(key):
            return " / ".join("—" if v is None else f"{v:.4f}" for v in stats[key])
        markdown.extend([f"## {name}","",
                         f"- {label}。[对照图](panels/{name})",
                         f"- 逆深度标准差 {values('depth_inverse_spread')} m⁻¹，"
                         f"离焦平面逆深度距离 {values('inverse_distance_to_focus')} m⁻¹。",
                         f"- GT 前景 CoC MAE {values('foreground_coc_mae_px')} px，"
                         "全图与同框放大用于核对光晕、裂缝和过锐边缘。",""])
        html.append(f'<figure><a href="panels/{escape(name)}"><img src="panels/{escape(name)}" '
                    f'loading="lazy" width="100%"></a><figcaption>{escape(name)} · {escape(label)}</figcaption></figure>')
    if diagnostic is not None:
        markdown.extend(diagnostic_markdown(diagnostic))
        html.insert(2,'<p>固定细结构帧的 GT 洋红框用于选取框内背景，新增融合深度与衰减系数图。</p>')
    (output/"index.html").write_text("\n".join(html)+"\n")
    (output/"观察清单.md").write_text("\n".join(markdown)+"\n")
    summary=dict(rows=results,unavailable=unavailable,index=str(output/"index.html"),observations=str(output/"观察清单.md"))
    write_json(output/"panel-index.json",summary)
    return summary
