"""Read Stage 3a crop traces and compile exact geometric hit-mask statistics."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import json

import numpy as np

from route3_v2data_B_labels import StaticWorld
from route3_v2data_optics import ray_fields, project
from route3_v2data_B_inputs import cameras, observed_rgb
from route3_v2data_stage2_eval import frames, peeled_truth
from route3_v2data_B_batch import ROOT, FORMAL_DATA
from route3_v2data_module2_infer import PARENTS, geometry_root, gallery, load_parents
from route3_v2data_E_labels import save_json
from route3_blender_b2_full_geometry import FullWorld
from src.dual_layer import depth_band

OUT = ROOT/"outputs/route3-v2data-hidden-color-purity-2026-11-26-01"
STAGE3A = ROOT/"outputs/route3-v2data-stage3a-2026-11-24-02"
E0_FRAMES = ROOT/"outputs/route3-v2data-E0-rerender-2026-11-26-01/fixed-base/frames"
CUTS = np.array([.5,.9,.99], np.float32)
SAMPLE_GRID = (.125,.375,.625,.875)


def read(path):
    return json.loads(path.read_text())


def read_arrays(path,*keys):
    with np.load(path,allow_pickle=False) as f:
        return {k:f[k] for k in keys}


def hit_masks(unit, frame, index, stop, focus, world=None):
    labels=unit/"labels"
    center=read_arrays(labels/f"f{frame:03d}-pinhole.npz","depth_axial_m","center_visible","K")
    gt=np.where(center["center_visible"],center["depth_axial_m"],1e4).astype(np.float32)
    band=depth_band(gt,center["K"][0,0],focus,.05/(2*stop))[0]
    if stop==1.4:
        raw=read_arrays(labels/f"f{frame:03d}-focus{index}-K16.npz","hidden","recoverable")
    else:
        if world is None:
            raise ValueError("An uncached pupil needs the saved scene mesh")
        traced,_=world.aperture(frame,16,stop,focus,track=True)
        raw={k:traced[k] for k in ("recoverable","hidden")}
    return raw["hidden"].any(0),band&raw["recoverable"].any(0)


def summarize_crop_scene(job):
    name,logged=job
    destination=OUT/"crop-scenes"/f"{name}.json"
    if destination.exists():
        return read(destination)
    unit=FORMAL_DATA/"units"/"training"/name
    by_key=defaultdict(list)
    meta={r["id"]:r for r in read(FORMAL_DATA/"optical"/name/"labels/optical-audit.json")["optical"]}
    for row in logged:
        by_key[row["case"]].append(row)
    world=None
    sums=defaultdict(Counter)
    for case,draws in sorted(by_key.items()):
        tag=case.removesuffix("-optics")
        f,j,_=tag.split("-")
        f=int(f[1:])
        j=int(j.removeprefix("focus"))
        config=meta[tag]
        stop=float(config["fstop"])
        if stop!=1.4 and world is None:
            world=StaticWorld(unit/"labels/scene-geometry.npz",512)
        hide,rec=hit_masks(unit,f,j,stop,float(config["focus_m"]),world)
        for row in draws:
            y,x,h,w=row["crop"]["sensor_yxhw"]
            region=np.s_[y:y+h,x:x+w]
            counts=Counter(crops=1,pixels=h*w,hidden_geometry_pixels=int(hide[region].sum()),
                           recoverable_geometry_pixels=int(rec[region].sum()),
                           placed_parents=int(row["support"]["hidden_placed"]),
                           eligible_parents=int(row["support"]["hidden_eligible"]))
            for k in ("all",f"f/{stop:g}","far" if j else "near",f"{'far' if j else 'near'} f/{stop:g}"):
                sums[k].update(counts)
    save_json(destination,dict(sums={k:dict(v) for k,v in sums.items()},scene=name))
    return read(destination)


def crop_audit():
    steps=STAGE3A/"steps.jsonl"
    job=defaultdict(list)
    for line in steps.open():
        row=json.loads(line)
        if row.get("crop"):
            job[row["item"][1]].append(row)
    if sum(map(len,job.values()))!=10_000:
        raise ValueError("Expected exactly the 10,000 archived Stage 3a B optical crops")
    dest=OUT/"crop-scenes"
    dest.mkdir(parents=True,exist_ok=True)
    merged=defaultdict(Counter)
    with ProcessPoolExecutor(max_workers=4) as executor:
        for i,scene in enumerate(executor.map(summarize_crop_scene,sorted(job.items())),1):
            for k,values in scene["sums"].items():
                merged[k].update(values)
            if i%10==0 or i==len(job):
                print("CROP_AUDIT",i,len(job),flush=True)
    total=Counter()
    for cohort,_,frame,unit,_,cases in frames():
        if cohort!="development":
            continue
        c=next(q for q in cases if q["focus_index"]==1)
        h,r=hit_masks(unit,frame,1,1.4,c["focus_m"])
        total.update(frames=1,pixels=h.size,hidden_geometry_pixels=int(h.sum()),
                     recoverable_geometry_pixels=int(r.sum()))
    def rates(row):
        return dict(row,**{k.replace("_pixels","_fraction"):row[k]/row["pixels"]
                           for k in ("hidden_geometry_pixels","recoverable_geometry_pixels")})
    save_json(OUT/"source-crop-summary.json",dict(
        source=str(steps),scenes=len(job),time_rows=sum(map(len,job.values())),
        definitions={
            "recoverable":"nominal-focus GT depth band intersected with any of 16 recoverable off-axis first intersections",
            "hidden":"any of the 16 off-axis rays hits a surface hidden in the centre view",
            "actual_compositing":"Historical steps retain parent counts and crop coordinates, not per-pixel placed Gaussian hit maps or the randomly drawn aperture offsets. The pixel figures are geometric hits, and parent counts retain their separate units.",
            "fstop":"f/1.4 uses the archived K16. f/2 uses the same deterministic pupil and saved mesh, with no training or rendered model inference."},
        training={k:rates(v) for k,v in merged.items()},development_far=rates(total)))


def bin_counts(x, valid=None):
    good=np.isfinite(x) if valid is None else np.isfinite(x)&valid
    counts=np.bincount(np.searchsorted(CUTS,x[good],side="right"),minlength=4)
    return dict(bins=counts.tolist(),unknown=int(x.size-good.sum()))


def ray_geometry(unit, frame, case):
    """Recover the exact identity and point associated with each stored parent."""
    if not unit:
        raise ValueError("Use the legacy world_position and object_id already in its oracle file")
    path=PARENTS/unit.name/(case["id"]+"-optics.npz")
    p=load_parents(path)
    k,poses=cameras(unit)
    pose=poses[frame-1]
    spec=dict(tiers="B",unit=unit,frame=frame,focus=case["focus_m"],fstop=case["fstop"],
              label=unit/"labels"/f"f{frame:03d}-focus{case['focus_index']}-K16.npz")
    raw=ray_fields(spec)
    use=raw["recoverable"]
    j,y,x=np.nonzero(use)
    depths=raw["depth"][use].astype(np.float32)
    pixels=np.stack((x+.5,y+.5,np.ones_like(x)),-1)
    grid=pixels @ np.linalg.inv(k).T
    xy=raw["aperture_xy"][j]*(1-depths[:,None]/spec["focus"])
    points=(grid*depths[:,None]+np.column_stack((xy,np.zeros(len(xy))))) @ pose[:3,:3].T+pose[:3,3]
    witness=raw["first_visible_other_frame"][use].astype(int)
    ids=raw["ids"][use]
    uv,z=project(points,pose,k)
    good=(z>0)&((uv>=0)&(uv<512)).all(-1)&(witness>0)&(witness<=len(poses))
    keys=np.column_stack((np.floor(uv[good]).astype(int),ids[good]))
    pick=np.flatnonzero(good)[np.unique(keys,axis=0,return_index=True)[1]]
    if not (np.array_equal(p["uv"],uv[pick].astype(np.float32))
            and np.array_equal(p["depth"],z[pick].astype(np.float32))):
        raise ValueError(f"Stored parent mapping differs in {unit.name}/{case['id']}")
    return p,points[pick],ids[pick],witness[pick],pixels[pick],raw["aperture_xy"][j[pick]]


class SceneEvidence:
    """Measure a real source pixel's 16 ray surface coverage on the saved geometry."""
    def __init__(self, unit, name, views=None):
        self.unit=unit
        self.name=name
        if unit:
            self.world=StaticWorld(unit/"labels/scene-geometry.npz",512)
            self.K,self.poses=cameras(unit)
            self.witness_frames=self.world.d["input_frames"].tolist()
        else:
            self.world=FullWorld(geometry_root(name)/"scene-geometry.npz",512)
            self.K,self.poses=self.world.K,self.world.cameras
            self.witness_frames=self.world.d["input_frames"].tolist()
        self.Kinv=np.linalg.inv(self.K)
        self.views={v["frame"]:v for v in views or []}
        # Store only pixel footprints requested by this inference run.
        self.pixels={}
        self.colors={}

    def rgb(self, frame):
        if self.unit:
            if frame not in self.colors:
                self.colors[frame]=observed_rgb(self.unit,frame)
            return self.colors[frame]
        return self.views[frame]["rgb_linear"]

    def coverage(self,frame,pixels,identity):
        if not len(pixels):
            return np.empty(0,np.float32)
        if (pixels<0).any() or (pixels>=512).any():
            raise ValueError("Pixel coverage coordinates lie outside the camera")
        packed=pixels[:,1]*512+pixels[:,0]
        unique,back=np.unique(packed,return_inverse=True)
        if frame not in self.pixels:
            self.pixels[frame]=(np.zeros((512*512,16),np.int16),np.zeros(512*512,bool))
        cache,seen=self.pixels[frame]
        fresh=unique[~seen[unique]]
        pose=self.poses[frame-1]
        for keys in np.array_split(fresh,max(1,(len(fresh)+8191)//8192)):
            if not len(keys):
                continue
            y,x=np.divmod(keys,512)
            offsets=np.array([(a,b) for a in SAMPLE_GRID for b in SAMPLE_GRID])
            xy=np.stack((x,y),axis=-1)[:,None]+offsets[None]
            dirs=np.concatenate((xy,np.ones((*xy.shape[:2],1))),axis=-1) @ self.Kinv.T @ pose[:3,:3].T
            _, hit=self.world.cast(frame,pose[:3,3],dirs.reshape(-1,3))
            objects=np.where(hit>=0,self.world.tri_obj[hit.clip(min=0)],0).reshape(-1,16)
            cache[keys]=objects
            seen[keys]=True
        surface=cache[unique][back]
        return ((surface==identity[:,None])&(identity[:,None]>0)).mean(1).astype(np.float32)

    def visible(self, frame, points, ids):
        if self.unit:
            return self.world.visible(frame,points,ids,tolerance=1e-4)[0]
        return self.world.visible_at(frame,frame,points,ids,tolerance=1e-4)[0]

    def peeled(self, frame, points, ids):
        valid=np.zeros(len(ids),bool)
        rgb=np.zeros((len(ids),3),np.float32)
        if frame!=30:
            return valid,rgb
        field,color,known=peeled_truth(self.unit,self.name,frame)
        uv,z=project(points,self.poses[frame-1],self.K)
        valid=(z>0)&((uv>=0)&(uv<512)).all(-1)
        x,y=np.floor(uv).clip(0,511).astype(int).T
        depth=field["depth_axial_m"][y,x].astype(np.float32)
        valid &= known[y,x]&(ids==field["object_id"][y,x])&(np.abs(z-depth)<=np.maximum(.01,.005*depth))
        rgb[valid]=color[y[valid],x[valid]]
        return valid,rgb


def verify_witness(audit, parents, points, ids, frames):
    rgb=np.zeros_like(parents["colors_linear"])
    usable=np.zeros(len(points),bool)
    purity=np.full(len(points),np.nan,np.float32)
    for f in np.unique(frames):
        idx=np.flatnonzero(frames==f)
        uv,z=project(points[idx],audit.poses[f-1],audit.K)
        ok=(z>0)&((uv>=0)&(uv<512)).all(-1)
        xy=np.floor(uv[ok]).astype(int)
        idx=idx[ok]
        rgb[idx]=audit.rgb(int(f))[xy[:,1],xy[:,0]]
        usable[idx]=True
        purity[idx]=audit.coverage(int(f),xy,ids[idx])
    if not (np.array_equal(usable,parents["recoverable"])
            and np.array_equal(rgb,parents["colors_linear"])):
        raise ValueError("S0 witness frame or pixel RGB does not reproduce the frozen cache")
    return purity


def substitute_variants(evidence, unit, name, frame, case):
    if unit:
        parents,points,ids,first,pixels,pupil=ray_geometry(unit,frame,case)
        s0=verify_witness(evidence,parents,points,ids,first)
        # Recast the exact K16 ray before testing alternate viewpoints. The
        # camera-space depth stored in the label is half precision.
        pose=evidence.world.cameras[frame-1]
        exact_k=evidence.world.K
        grid=pixels@np.linalg.inv(exact_k).T
        shift=np.column_stack((pupil,np.zeros(len(pupil))))
        starts=pose[:3,3]+shift@pose[:3,:3].T
        directions=(grid*case["focus_m"]-shift)@pose[:3,:3].T
        physical,hit=evidence.world.cast(frame,starts,directions)
        same=(hit>=0)&(evidence.world.tri_obj[hit.clip(min=0)]==ids)
        if not same.all():
            raise ValueError(f"K16 recast surface identity differs in {name}/{case['id']}")
        points=physical
    else:
        parents=read_arrays(geometry_root(name)/"oracle"/f"f{frame:03d}.npz",
                            "uv","depth","recoverable","colors_linear","world_position","object_id")
        points,ids=parents["world_position"],parents["object_id"]
        s0=np.full(len(ids),np.nan,np.float32)  # S0 uses same-frame peeled RGB on the legacy path.
    options=[f for f in evidence.witness_frames if f!=frame and (unit or f in evidence.views)]
    purity=np.full((len(points),len(options)),np.nan,np.float32)
    color=np.full((*purity.shape,3),np.nan,np.float32)
    support=parents["recoverable"]
    for column,other in enumerate(options):
        uv,z=project(points,evidence.poses[other-1],evidence.K)
        possible=np.flatnonzero(support&(z>0)&((uv>=0)&(uv<512)).all(-1))
        found=evidence.visible(other,points[possible],ids[possible])
        pick=possible[found]
        xy=np.floor(uv[pick]).astype(int)
        purity[pick,column]=evidence.coverage(other,xy,ids[pick])
        color[pick,column]=evidence.rgb(other)[xy[:,1],xy[:,0]]
    valid=np.isfinite(purity)
    best_idx=np.argmax(np.where(valid,purity,-1),axis=1)
    best=purity[np.arange(len(points)),best_idx]
    values=dict(S0=(parents["colors_linear"],support))
    for threshold in (0,*CUTS.tolist()):
        codec=f"{threshold:.2f}"
        keep=valid&(purity>0)&(purity>=threshold)
        admitted=keep.any(1)
        values[f"S1@{codec}"]=(np.nan_to_num(color[np.arange(len(points)),best_idx]),admitted)
        med=np.zeros((len(points),3),np.float32)
        med[admitted]=np.nanmedian(np.where(keep[admitted,...,None],color[admitted],np.nan),axis=1)
        values[f"S2@{codec}"]=(med,admitted)
    matched,peeled=evidence.peeled(frame,points,ids)
    if frame==30:
        values["S3"]=(peeled,matched&support)
    return parents,values,best,s0,points,ids


def witness_path(name, case):
    return OUT/"source-witnesses"/name/(case["id"]+".npz")


def archive_witnesses(item):
    name,jobs=item
    unit=jobs[0][1]
    views=[]
    if not unit:
        plan=next(p for p in gallery.planned_frames() if p["unit"] is None and p["name"]==name)
        views,_=gallery.scene_views(plan)
    evidence=SceneEvidence(unit,name,views)
    files=[]
    for frame,unit,cases in jobs:
        evaluated={r["case"] for r in read(E0_FRAMES/name/f"f{frame:03d}.json")["rows"]["G"]
                   if r["status"]=="evaluated"}
        for case in cases:
            if case["id"] not in evaluated:
                continue
            path=witness_path(name,case)
            if not path.exists():
                p,options,best,actual,_,_=substitute_variants(evidence,unit,name,frame,case)
                values=dict(uv=p["uv"],depth=p["depth"],recoverable=p["recoverable"],colors_linear=p["colors_linear"],
                            best=best,actual=actual)
                if "object_id" in p:
                    values["object_id"]=p["object_id"]
                for method,(rgb,keep) in options.items():
                    values[f"{method}/rgb"]=rgb
                    values[f"{method}/keep"]=keep
                path.parent.mkdir(parents=True,exist_ok=True)
                tmp=path.with_suffix(".part")
                with tmp.open("wb") as fh:
                    np.savez_compressed(fh,**values)
                tmp.replace(path)
            files.append(str(path.relative_to(OUT)))
    return name, files


def witness_audit():
    jobs=defaultdict(list)
    for _,name,frame,unit,_,cases in frames():
        jobs[name].append((frame,unit,cases))
    results={}
    with ProcessPoolExecutor(max_workers=3) as workers:
        for name,files in workers.map(archive_witnesses,sorted(jobs.items()),chunksize=1):
            results[name]=files
            print("SOURCE_WITNESSES",len(results),len(jobs),name,len(files),flush=True)
    if sum(map(len,results.values()))!=105:
        raise ValueError("Expected witness coverage for 87 development and 18 legacy evaluated cases")
    save_json(OUT/"source-witnesses.json",dict(files=results))


def read_witnesses(name,case):
    path=witness_path(name,case)
    with np.load(path,allow_pickle=False) as f:
        parent={k:f[k] for k in ("uv","depth","recoverable","colors_linear")}
        variants={k.split("/")[0]:(f[k],f[k.replace("/rgb","/keep")]) for k in f.files if k.endswith("/rgb")}
        return parent,variants,f["best"],f["actual"]


if __name__=="__main__":
    args=argparse.ArgumentParser(description=__doc__)
    args.add_argument("--witnesses",action="store_true")
    witness_audit() if args.parse_args().witnesses else crop_audit()
