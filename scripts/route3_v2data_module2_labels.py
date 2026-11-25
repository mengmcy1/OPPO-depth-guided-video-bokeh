"""Cast midpoint tear labels from sealed development meshes on CPU."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from route3_ovg_subpixel import PREVIOUS, ROOT, intersect, save_json, save_npz
from route3_blender_b2_geometry import World
from route3_v2data_B_inputs import FORMAL_DATA, library_unit_paths, observed_rgb, to_srgb8
from route3_v2data_B_labels import StaticWorld
from src.ovg_connected_depth import DIRECTIONS, edge_slices

CACHE = FORMAL_DATA/"derived-module2-tears-2026-11-25-v2"
OUT = ROOT/"outputs/route3-v2data-module2-infer-2026-11-25-01"
FRAMES = (1,30,60)
LEGACY = ("metric-scene-122104-broad","metric-scene-122105-multilayer","metric-scene-122107-broad")
THIN = {"003","004","005"}
FOUR = THIN|{"008"}
PROTOCOL = dict(version="v2",prescreen_same_id_relative_lt=.005,middle_fraction=.25,
                middle_relative_tolerance=.002,tear_min_relative=.03,
                near_side_fraction=.1,invalid_edge_is_disconnected=True,
                relative_denominator="max of endpoint inverse depths",
                different_id_step="A >=3% jump with midpoint between the endpoint inverse depths can aid a tear only if the midpoint continuity test fails.")


def inputs():
    dev = [("development",p.name,p/"labels/scene-geometry.npz",p) for p in library_unit_paths("dev")]
    old = [("legacy",n,PREVIOUS/"truth"/n/"scene-geometry.npz",None) for n in LEGACY]
    return dev+old


def archive(c, name, unit, frame):
    if c in ("development","training"):
        return unit/"labels"/f"f{frame:03d}-pinhole.npz"
    return PREVIOUS/"truth"/name/"geometry"/f"f{frame:03d}-center.npz"


def align(world,frame,gt):
    h,w=gt["depth_axial_m"].shape
    if (h,w)!=(512,512) or not np.allclose(gt["K"],world.K,atol=1e-4,rtol=1e-6) or not np.allclose(
            gt["camera_c2w"],world.cameras[frame-1],atol=1e-5,rtol=1e-6):
        raise ValueError(f"Saved mesh and center-label camera mismatch at f{frame:03d}")
    # Cross-check the same center rays across the image before using adjacent labels.
    yy,xx=np.mgrid[1:h:7,1:w:7]
    xy=np.c_[xx.ravel(),yy.ravel()]
    depth,_,obj = intersect(world,frame,xy,1)
    saved = gt["depth_axial_m"][yy,xx].ravel().astype(np.float32)
    known = gt["center_visible"][yy,xx].ravel()
    old_id = gt["object_id"][yy,xx].ravel()
    depth,obj = depth[:,0],obj[:,0]
    stable = (known==(depth>0)) & (~known|((old_id==obj)&np.isclose(saved,depth,rtol=.001,atol=.001)))
    if stable.mean()<.999:
        raise ValueError(f"Saved center labels disagree with mesh first hits at f{frame:03d}: {stable.mean():.4%}")
    return dict(rays=len(xy),matched=int(stable.sum()))


def classify(world,frame,gt):
    z = gt["depth_axial_m"].astype(np.float32)
    obj = gt["object_id"]
    center = gt["center_visible"]&np.isfinite(z)&(z>0)
    inv = np.divide(1.,z,out=np.zeros_like(z),where=center)
    status = np.full((4,*z.shape),-2,np.int8)
    rows = []
    for slot,(dy,dx) in enumerate(DIRECTIONS):
        a,b = edge_slices(dy,dx)
        r,s = inv[a],inv[b]
        scale = np.maximum(r,s)
        gap = np.abs(r-s)
        relative = np.divide(gap,scale,out=np.zeros_like(gap),where=scale>0)
        different = obj[a]!=obj[b]
        valid_center = center[a]&center[b]
        pre = valid_center&~different&(relative<PROTOCOL["prescreen_same_id_relative_lt"])
        label = np.full(r.shape,-1,np.int8)
        label[pre] = 0
        picked = valid_center&~pre
        yy,xx = np.nonzero(picked)
        xy = np.c_[xx+(a[1].start or 0),yy+(a[0].start or 0)]
        middle_hit,no_hit,third,uncertain = 0,0,0,0
        if len(xy):
            depth,_,identity = intersect(world,frame,xy,1,(dx*.5,dy*.5))
            depth,identity = depth[:,0],identity[:,0]
            hit = np.isfinite(depth)&(depth>0)
            mid = np.divide(1.,depth,out=np.zeros_like(depth),where=hit)
            rp,sp,g,mx = r[picked],s[picked],gap[picked],scale[picked]
            known_side = (identity==obj[a][picked])|(identity==obj[b][picked])
            close = np.minimum(abs(mid-rp),abs(mid-sp))<=PROTOCOL["near_side_fraction"]*g
            middle = abs(mid-(rp+sp)/2) <= PROTOCOL["middle_fraction"]*g+PROTOCOL["middle_relative_tolerance"]*mx
            within = (mid>=np.minimum(rp,sp)-.1*g)&(mid<=mx+.1*g)
            usable = hit&known_side
            stepped = (relative[picked]>=PROTOCOL["tear_min_relative"]) & (close | (different[picked]&within&~middle))
            connected = usable&middle&~stepped
            torn = usable&stepped&~middle
            label[picked] = np.where(connected,0,np.where(torn,1,-1))
            middle_hit = int(hit.sum())
            no_hit = int((~hit).sum())
            third = int((hit&~known_side).sum())
            uncertain = int((usable&~(connected|torn)).sum())
        status[slot][a] = label
        rows.append(dict(edges=label.size,prescreen=int(pre.sum()),rays=len(xy),midpoint_hits=middle_hit,
                         connected=int((label==0).sum()),torn=int((label==1).sum()),invalid=int((label==-1).sum()),
                         center_invalid=int((~valid_center).sum()),midpoint_no_hit=no_hit,third_surface=third,
                         midpoint_uncertain=uncertain,different_id_connected=int((different&(label==0)).sum())))
    return status,rows


def font(n):
    p=Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    return ImageFont.truetype(str(p),n) if p.is_file() else ImageFont.load_default()


def overlay(path,unit,frame,gt,status):
    rgb = to_srgb8(observed_rgb(unit,frame))
    z = gt["depth_axial_m"].astype(np.float32)
    valid = gt["center_visible"]&(z>0)
    inv = np.divide(1.,z,out=np.zeros_like(z),where=valid)
    limit = float(np.quantile(inv[valid],.99))
    depth = cv2.applyColorMap(np.uint8(np.clip(inv/limit,0,1)*255),cv2.COLORMAP_VIRIDIS)[...,::-1]
    n=512
    canvas=Image.new("RGB",(n*2+18,n+46),(16,20,26))
    text=ImageDraw.Draw(canvas)
    text.text((3,3),unit.name+" · f030 · red tear / orange invalid",font=font(16),fill="white")
    for i,(title,img) in enumerate((("RGB",rgb),("GT inverse depth + midpoint edges",depth))):
        x0=i*(n+12)
        zoom=np.array(Image.fromarray(img).resize((n*2,n*2),Image.Resampling.NEAREST)).copy()
        for k,(dy,dx) in enumerate(DIRECTIONS):
            for status_value,color in ((1,(255,20,25)),(-1,(255,166,30))):
                yy,xx=np.nonzero(status[k]==status_value)
                zoom[yy*2+1+dy,xx*2+1+dx]=color
        # Same-object continuous interiors remain in the underlying photograph.
        shape=Image.fromarray(zoom).resize((n,n),Image.Resampling.BOX)
        canvas.paste(shape,(x0,39))
        text.text((x0+3,20),title,font=font(14),fill="white")
    path.parent.mkdir(parents=True,exist_ok=True)
    canvas.save(path,quality=96)


def scene(job,frames,pilot):
    cv2.setNumThreads(1)
    cohort,name,mesh,unit = job
    paths = [archive(cohort,name,unit,f) for f in frames]
    if not mesh.is_file() or not all(p.is_file() for p in paths):
        if cohort=="legacy":
            return dict(scene=name,cohort=cohort,skipped="mesh or center labels absent")
        raise FileNotFoundError(f"Missing required mesh or center labels for {name}")
    world = (StaticWorld if unit else World)(mesh,512)
    rows = []
    for frame,path in zip(frames,paths):
        dest = CACHE/name/f"f{frame:03d}.npz"
        met = dest.with_suffix(".json")
        if dest.is_file() and met.is_file():
            row = json.loads(met.read_text())
            if row["protocol"]!=PROTOCOL:
                raise ValueError(f"Existing midpoint cache uses a different protocol: {dest}")
            rows.append(row)
            if pilot:
                with np.load(path) as data,np.load(dest) as arrays:
                    overlay(OUT/"label-audit-v2"/f"{name}-f{frame:03d}.jpg",unit,frame,data,arrays["status"])
            continue
        with np.load(path) as gt:
            alignment = align(world,frame,gt)
            status, counts = classify(world,frame,gt)
            if pilot:
                overlay(OUT/"label-audit-v2"/f"{name}-f{frame:03d}.jpg",unit,frame,gt,status)
        save_npz(dest,status=status)
        row=dict(scene=name,cohort=cohort,frame=frame,protocol=PROTOCOL,alignment=alignment,
                 directions=counts,cache=str(dest),subset="thin_structure" if cohort=="development" and name.split("-")[1] in THIN else
                 "dev008" if name.startswith("development-008-") else cohort)
        save_json(met,row)
        rows.append(row)
        print("MODULE2_LABEL",cohort,name,frame,sum(r["rays"] for r in counts),flush=True)
    return dict(scene=name,cohort=cohort,frames=rows)


def run(pilot,workers):
    import os
    available=len(os.sched_getaffinity(0))
    if not 1<=workers<=available//2:
        raise ValueError("Midpoint audit may use at most half of available CPUs")
    jobs = inputs()
    if pilot:
        jobs = [r for r in jobs if r[0]=="development" and r[1].split("-")[1] in FOUR]
    start=time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        tasks=[pool.submit(scene,job,(30,) if pilot else FRAMES,pilot) for job in jobs]
        rows=[f.result() for f in tasks]
    save_json(OUT/("label-pilot-v2.json" if pilot else "labels.json"),
              dict(protocol=PROTOCOL,rows=rows,workers=workers,seconds=time.monotonic()-start,test_cases_used=0))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pilot",action="store_true")
    p.add_argument("--workers",type=int,default=4)
    a=p.parse_args()
    run(a.pilot,a.workers)
