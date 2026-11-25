"""Freeze B inventory and measure native E supervision and multiview margin."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
import os
import time

import numpy as np

from route3_ovg_subpixel import ROOT, intersect, save_json
from route3_v2data_B_inputs import FORMAL_DATA, library_unit_paths, observation
from route3_v2data_B_labels import StaticWorld
from route3_v2data_loader import b_optical_cases, registered_b
from route3_v2data_stage3a import thin_surface_ids
from route3_v2data_stage3a_prior_diagnostic import gt_regions

RUN = ROOT/"outputs/route3-v2data-E-prep-2026-11-26-01"
FRAMES = (1,30,60)
REGIONS = ("rod","gap","normal")
PRIORS = ("SHARP","DepthSplat","triangulation")
NATIVE = (512,512)
ABS_TOLERANCE = .002
REL_TOLERANCE = .01
SEED = 20261126
V1_PLAN = ROOT/"03_实验记录/records/route3-v2data-module2-tear-d-config-2026-11-25.json"
PINHOLE = ("depth_axial_m","depth1","depth2","alpha","valid_dual","center_visible",
           "valid","valid_alpha","object_id","first_hit_object","depth_boundary","K","camera_c2w")
DEV_PANEL = {"003","004","005","008"}
WORKERS = min(6, len(os.sched_getaffinity(0))//2)


def load(path):
    return json.loads(path.read_text())


def pixels(path, names):
    with np.load(path) as arrays:
        return {k:arrays[k] for k in names}


def inverse(z):
    f = z.astype(np.float32,copy=False)
    return np.divide(1., f, out=np.zeros_like(f), where=np.isfinite(f)&(f>0))


def stats(parts):
    joined = np.concatenate(parts)
    if not joined.size:
        return dict(n=0,p50=None,p95=None,signed_p50=None)
    absolute = np.abs(joined)
    q50,q95 = np.quantile(absolute,[.5,.95],overwrite_input=True)
    return dict(n=int(joined.size),p50=float(q50),p95=float(q95),
                signed_p50=float(np.median(joined,overwrite_input=True)))


def core_labels(unit, frame, meta):
    labels = unit/"labels"
    key = f"f{frame:03d}"
    x = pixels(labels/f"{key}-pinhole.npz",PINHOLE)
    for field in PINHOLE[:-2]:
        if x[field].shape != NATIVE:
            raise ValueError(f"{unit.name}/{key} non-native {field}: {x[field].shape}")
    gt = x["depth_axial_m"].astype(np.float32)
    front,back = (inverse(x[k]) for k in ("depth1","depth2"))
    center = inverse(gt)
    if (x["center_visible"] & (center<=0)).any() or (x["valid_dual"] & ((back<=0)|(front<=back))).any():
        raise ValueError(f"{unit.name}/{key} invalid center hit or ordered dual depth")
    visible = x["center_visible"] & (center>0)
    reference = dict(gt=gt,valid=visible,object_id=x["object_id"],foreground=np.array(meta["foreground"]))
    ids = meta["thin"]
    masks, boxes = gt_regions(reference,ids)
    rod,gap = masks["foreground"],masks["gap_background"]
    if (rod&gap).any() or rod.shape!=NATIVE or gap.shape!=NATIVE:
        raise ValueError(f"{unit.name}/{key} invalid native rod/gap partition")
    zone = np.where(rod,0,np.where(gap,1,2)).astype(np.uint8)
    peeled = labels/f"{key}-peeled.npz"
    if peeled.is_file():
        q = pixels(peeled,("depth_axial_m","center_visible","K","camera_c2w"))
        if (q["depth_axial_m"].shape!=NATIVE or q["center_visible"].shape!=NATIVE
                or not np.array_equal(q["K"],x["K"]) or not np.array_equal(q["camera_c2w"],x["camera_c2w"])):
            raise ValueError(f"{unit.name}/{key} misaligned peeled label")
        rear = inverse(q["depth_axial_m"])
        rod_reference = np.where(q["center_visible"],rear,0.)
        ref_source = "peeled"
    else:
        rod_reference = np.where(x["valid_dual"],back,0.)
        ref_source = "depth2_valid_dual"
    x.update(center=center,front=front,back=back,visible=visible,zone=zone,boxes=boxes,
             rod_reference=rod_reference,rod_reference_source=ref_source)
    return x


def inventory():
    training,excluded = registered_b()
    archive=load(FORMAL_DATA/"manifest.json")
    optical = sorted({r["scene_id"] for r in archive["optical_units"]})
    if set(optical)-set(training):
        raise ValueError("The admitted optical snapshot contains units outside registered training geometry")
    development = {p.name:p for p in library_unit_paths("dev")}
    if len(development)!=15:
        raise ValueError("E preparation needs the 15 registered development scenes")
    metadata={}
    for role,units in (("train",training),("dev",development)):
        for name,unit in units.items():
            category=load(unit/"scene-spec.json")["category"]
            foreground=pixels(unit/"labels/scene-geometry.npz",("foreground_object_ids",))["foreground_object_ids"].tolist()
            ids=[int(a) for a in thin_surface_ids(unit)]
            if (not ids and category=="thin_structure") or (role=="dev" and name.split("-")[1]=="008"):
                ids=foreground
            metadata[name]=dict(role=role,category=category,thin=ids,foreground=foreground)
    for name in optical:
        if len(b_optical_cases(training[name]))!=12:
            raise ValueError(f"{name} has an incomplete registered optical release")
    rng=np.random.default_rng(SEED)
    groups=("thin_structure","multiple_layers")
    holdout=sorted(n for cat in groups for n in rng.choice(
        sorted(n for n in training if metadata[n]["category"]==cat),5,replace=False).tolist())
    pool=[n for n,cat in load(V1_PLAN)["data"]["training_scenes"].items() if cat=="thin_structure"]
    if len(pool)!=30:
        raise ValueError("The previous thin-structure inventory has changed")
    optical_set=set(optical)
    candidates=sorted((set(pool)&optical_set)-set(holdout))
    # Keep the same three optical frames available for all five overfit scenes.
    candidates=[n for n in candidates if all(
        (FORMAL_DATA/"derived-f2"/n/f"f{f:03d}"/"manifest.json").is_file() for f in FRAMES)]
    ordered=rng.permutation(candidates).tolist()
    overfit=[]
    for name in ordered:
        frame_rows=[core_labels(training[name],f,metadata[name]) for f in FRAMES]
        if all((r["zone"]==0).any() and (r["zone"]==1).any() for r in frame_rows):
            overfit.append(name)
        if len(overfit)==5:
            break
    if len(overfit)!=5:
        raise ValueError("Five eligible thin training scenes with rods, gaps and optical targets are required")
    excluded.update({n:v for n,v in archive.get("generation_failures",{}).get("training",{}).items()
                    if n not in training} if isinstance(archive.get("generation_failures",{}).get("training"),dict) else {})
    return dict(disjoint_optical_geometry=False,geometry=sorted(training),optical=optical,
                development=sorted(development),holdout=holdout,overfit=overfit,metadata=metadata,
                excluded=excluded,frames=list(FRAMES),optical_snapshot_counts=archive["counts"],
                overfit_eligible_scene_count=len(candidates),independent_test_cases_used=0)


def unit_paths(plan):
    return {n:FORMAL_DATA/"units"/("training" if info["role"]=="train" else "development")/n
            for n,info in plan["metadata"].items()}


def phase_shift(world, frame, labels, mask):
    """Compare the modal class of the center to four offset pinhole hits."""
    yy,xx=np.nonzero(mask)
    if not yy.size:
        return (0,0,0)
    xy=np.c_[xx,yy]
    u=labels["center"][yy,xx]
    f,g=(labels[k][yy,xx] for k in ("front","back"))
    initial=np.abs(u-f)<np.abs(u-g)
    supported=np.zeros_like(initial)
    changed=np.zeros_like(initial)
    for shift in ((-.25,0),(.25,0),(0,-.25),(0,.25)):
        z,_,_=intersect(world,frame,xy,1,shift)
        r=inverse(z[:,0])
        a,b=(np.abs(r-q) for q in (f,g))
        known=(r>0)&((a<=ABS_TOLERANCE+REL_TOLERANCE*f)|(b<=ABS_TOLERANCE+REL_TOLERANCE*g))
        supported|=known
        changed|=known&((a<b)!=initial)
    return int(mask.sum()),int(supported.sum()),int(changed.sum())


def scan(unit, meta):
    rows=[]
    errors=[[],[]]
    world=StaticWorld(unit/"labels/scene-geometry.npz",512) if meta["thin"] else None
    for frame in FRAMES:
        y=core_labels(unit,frame,meta)
        inp=observation(unit,frame,aligned=True)
        if (inp["depth"].shape!=(3,*NATIVE) or inp["confidence"].shape!=(3,*NATIVE)
                or inp["rgb"].shape!=(*NATIVE,3) or not np.allclose(inp["K"],y["K"],atol=1e-4)
                or not np.allclose(inp["pose"],y["camera_c2w"],atol=1e-5)
                or not np.isfinite(inp["rgb"]).all()):
            raise ValueError(f"{unit.name}/{frame} native observations do not match supervision")
        prior=inverse(inp["depth"])
        single=y["visible"]&y["valid_alpha"]&(y["alpha"]==1)&(y["zone"]==2)
        for i in (1,2):
            m=single&(prior[i]>0)&(inp["confidence"][i]>0)
            errors[i-1].append(np.log(prior[i,m])-np.log(y["center"][m]))
        dual=y["visible"]&y["valid_dual"]
        front,back=y["front"],y["back"]
        d1,d2=(np.abs(y["center"]-d) for d in (front,back))
        off=(d1>ABS_TOLERANCE+REL_TOLERANCE*front)&(d2>ABS_TOLERANCE+REL_TOLERANCE*back)
        thin_dual=dual&np.isin(y["first_hit_object"],meta["thin"])
        groups={}
        for al,mask in (("alpha_lt_half",y["alpha"]<.5),("alpha_ge_half",y["alpha"]>=.5)):
            s=dual&mask
            thin=thin_dual&mask
            n,valid,jumps=phase_shift(world,frame,y,thin&~off) if world else (0,0,0)
            groups[al]=dict(dual=int(s.sum()),conflict=int((s&off).sum()),thin_dual=int(thin.sum()),
                 thin_center_front=int((thin&~off&(d1<d2)).sum()),thin_center_back=int((thin&~off&(d2<d1)).sum()),
                 phase_candidates=n,phase_scored=valid,phase_changed=jumps)
        rows.append(dict(scene=unit.name,frame=frame,region_counts=np.bincount(y["zone"].ravel(),minlength=3).tolist(),
                         rod_reference=y["rod_reference_source"],groups=groups))
    return rows,errors


def summarize_groups(rows):
    result={}
    for group in ("alpha_lt_half","alpha_ge_half"):
        count=Counter()
        for row in rows:
            count.update(row["groups"][group])
        result[group]=dict(count,conflict_fraction=count["conflict"]/max(1,count["dual"]),
                           phase_change_fraction=count["phase_changed"]/max(1,count["phase_scored"]))
    return result


def main():
    began=time.monotonic()
    plan=inventory()
    paths=unit_paths(plan)
    save_json(RUN/"inventory.json",plan)
    regions={}
    by_scope=dict(geometry=set(plan["geometry"]),optical=set(plan["optical"]),development=set(plan["development"]))
    by_scope["margin_train"]=by_scope["geometry"]-set(plan["holdout"])
    totals={k:defaultdict(list) for k in by_scope}
    frames=[]
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for name,(rows,values) in zip(paths,pool.map(scan,paths.values(),plan["metadata"].values())):
            frames.extend(rows)
            for scope,group in by_scope.items():
                if name in group:
                    for prior,errors in zip(PRIORS[1:],values):
                        totals[scope][prior].extend(errors)
            if len(frames)%90==0:
                print("E_LABEL",len(frames),name,flush=True)
    for scope,group in by_scope.items():
        own=[row for row in frames if row["scene"] in group]
        regions[scope]=dict(scenes=len(group),frames=len(own),
            categories=dict(Counter(plan["metadata"][n]["category"] for n in group)),
            frames_with_rods=sum(row["region_counts"][0]>0 for row in own),
            frames_with_gaps=sum(row["region_counts"][1]>0 for row in own),
            conflicts=summarize_groups(own),
            single_surface_log_inverse_error={k:stats(v) for k,v in totals[scope].items()})
    margin=max(v["p95"] for v in regions["margin_train"]["single_surface_log_inverse_error"].values())
    save_json(RUN/"labels.json",dict(frames=frames,scopes=regions,m=margin,workers=WORKERS,
        comparison=dict(absolute_inv_m=ABS_TOLERANCE,relative_inverse=REL_TOLERANCE,
            phase_shifts_center_pixels=[[-.25,0],[.25,0],[0,-.25],[0,.25]],
            phase_denominator="valid dual first hit on designated thin object, nonconflicting center, at least one valid offset ray",
            m_rule="maximum of the two per-source absolute log-inverse p95 on training excluding the ten holdouts",
            priority="DepthSplat if condition passes, otherwise qualifying triangulation"),
        elapsed_seconds=time.monotonic()-began))
    print("E_LABEL_DONE",len(frames),margin,flush=True)


if __name__=="__main__":
    main()
