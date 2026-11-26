"""Put colour handling, F2 and witness support on the same sensor crops."""
import cv2
import numpy as np
from PIL import Image, ImageDraw

from route3_v2data_module2_infer import gallery, crop_for, title

PALETTE=np.array([[25,26,34],[136,48,86],[225,118,68],[232,193,83],[85,201,166]],np.uint8)
COLUMNS=(
    ("Cycles","Cycles 真值",None),
    ("V0_pred/F2","预测 V0 · F2","V0_pred"),
    ("V0_pred/S0","预测 V0 · S0","V0_pred"),
    ("V0_pred/S1@0.00","预测 V0 · S1","V0_pred"),
    ("V0_pred/S2@0.99","预测 V0 · S2 ≥.99","V0_pred"),
    ("V0_pred/S3","预测 V0 · S3","V0_pred"),
    ("V0_GT/S0","GT V0 · S0","V0_GT"),
    ("V0_GT/S2@0.99","GT V0 · S2 ≥.99","V0_GT"),
    ("V0_GT/S3","GT V0 · S3","V0_GT"),
    ("Query","实际 Query","Query"),
    ("QGT@0.99","Query GT ≥.99","QGT@0.99"),
    ("QPred@0.99","Query 预测 ≥.99","QPred@0.99"),
)


def outline(image,area):
    rgb=np.asarray(image).copy()
    mark=cv2.dilate(area.astype(np.uint8), np.ones((3,3),np.uint8)).astype(bool)&~area
    rgb[mark]=(252,230,135)
    return Image.fromarray(rgb)


def draw_panel(root,row,truth,area,edge,pictures):
    where=root/"figures"
    where.mkdir(parents=True,exist_ok=True)
    shape=(512,512)
    boundary=area&edge
    crop=crop_for(pictures["V0_GT/S0"].astype(np.float32),truth,boundary if boundary.any() else area)
    w,h=196,196
    board=Image.new("RGB",(w*len(COLUMNS),h*5+116),(22,26,32))
    draw=ImageDraw.Draw(board)
    scene=title(row["scene"])
    draw.text((5,6),f"{scene} f030 远焦 G · 金线为可恢复评价区 · 区域 PSNR",
              font=gallery.font(19), fill="white")
    for j,(key,caption) in enumerate(zip(("未知","<.50",".50–.90",".90–.99","≥.99"),PALETTE)):
        x=j*120+6
        draw.rectangle((x,34,x+18,49),fill=tuple(caption))
        draw.text((x+22,33),key,font=gallery.font(14),fill="white")
    draw.text((630,33),"纯度按贡献最大的档着色 · 红色为无 ≥.99 有效隐藏贡献",
              font=gallery.font(14),fill="white")
    for i,(key,caption,tag) in enumerate(COLUMNS):
        x=i*w
        if tag is None:
            image=truth
            psnr=""
        else:
            image=pictures[key].astype(np.float32)
            if "/" in key:
                family,mode=key.split("/",1)
                metrics=row["render"]["G"][family][mode]
            else:
                metrics=row["render"]["G"]["Query"]["methods"][key]
            q=metrics["score"]["psnr"]
            psnr=f" · {q:.2f}" if q is not None else ""
        draw.text((x+3,67),caption+psnr,font=gallery.font(14),fill="white")
        original=outline(gallery.rgb_image(image),area)
        draw_box=ImageDraw.Draw(original)
        draw_box.rectangle(crop,outline=(80,211,250),width=2)
        error=np.abs(image-truth).mean(-1)
        heat=outline(gallery.scalar_image(error,.25,cv2.COLORMAP_INFERNO),area)
        purity=np.full(shape,-1,np.int8) if tag is None else pictures[f"{tag}/purity"]
        unrecoverable=np.zeros(shape,bool) if tag is None else pictures[f"{tag}/unrecoverable"]&area
        missing=np.zeros((*shape,3),np.uint8)
        missing[area]=(36,57,64)
        missing[unrecoverable]=(218,78,84)
        purity_rgb=PALETTE[np.where(area,purity+1,0)]
        tiles=(original,outline(gallery.rgb_image(image),area).crop(crop),heat,
               outline(Image.fromarray(purity_rgb),area),Image.fromarray(missing))
        for j,tile in enumerate(tiles):
            offset=87+j*h
            if i==0:
                draw.text((5,offset),("整图","边缘放大","误差 0–0.25","见证纯度","不可恢复")[j],
                          font=gallery.font(12),fill="white")
            board.paste(tile.resize((w-5,h-16),Image.Resampling.NEAREST if j else Image.Resampling.BOX),
                        (x+2,offset+16))
    path=where/f"{scene}-f030-G.jpg"
    board.save(path,quality=94)
    with Image.open(path) as verified:
        verified.verify()
    return [path]
