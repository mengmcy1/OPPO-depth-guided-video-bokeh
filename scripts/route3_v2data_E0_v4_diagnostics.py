"""Compare component diagnostics and produce the four prescribed E0 v4 panels."""
import argparse
from collections import defaultdict

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

import route3_v2data_E0_v4_optics as optics
from route3_v2data_B_inputs import FORMAL_DATA, cameras, formal_unit
from route3_v2data_E0_data import dataset
from route3_v2data_E0_train import single_back_target
from route3_v2data_E_labels import load, save_json
from route3_v2data_hidden_color_sources import SceneEvidence, substitute_variants
from route3_v2data_optics import f2_entry, hidden_parents, observed_background, ray_fields

ROOT = optics.RUN
GALLERY = optics.base.gallery
ARMS = ("B","C","D","v4")
TITLES = {"B":"B · v1@2k", "C":"C · v2@4k", "D":"D · v3@4k（诊断）", "v4":"第四版 · 深度选点"}
MODEL_SPECS = {"B":("v1",2000), "C":("v2",4000), "D":("v3",4000)}
LENSES = (("P","各臂对焦"),("G","GT 对焦"))
GROUPS = ("development","holdout")


def plane(t):
    return t[0,0].float().cpu().numpy()


def median(value):
    return float(np.median(value)) if value.size else None


def error_stats(value):
    return dict(pixels=int(value.size), signed_p50=median(value), abs_p50=median(np.abs(value)),
                abs_p95=float(np.percentile(np.abs(value),95)) if value.size else None)


def fields(network, target):
    out = optics.predict(network, target)
    front_log = plane(out["means"][:,:1])
    rear_log = plane(out["raw_back"])
    bias = plane(target["beta"])
    dual = plane(target["valid_dual"]).astype(bool)
    center = plane(target["center"])
    rear_target = np.where(dual,np.log(np.maximum(plane(target["back"]),1e-8)),
                           plane(single_back_target(out,target)))
    valid = plane(target["visible"]).astype(bool)
    zones = plane(target["loss_zone"]).astype(np.uint8)
    return dict(front=np.exp(front_log), rear=np.exp(plane(out["means"][:,1:2])),
                rear_raw=np.exp(rear_log), gt=center, dual=dual,
                front_error=np.exp(front_log)-center, rear_error=rear_log-rear_target,
                visible=valid, rod=valid&(zones==0), gap=valid&(zones==1), normal=valid&(zones==2),
                selected=plane(out["selected_back"]).astype(bool), beta=np.exp(bias),
                beta_source=plane(target["mv_source"]).astype(np.uint8), residual=1.5*plane(out["raw_tanh"][:,1:2]),
                applied_residual=rear_log-bias, saturated=np.abs(plane(out["raw_tanh"][:,1:2]))>.95,
                g=plane(out["gate"]), a=plane(target["attenuation"]))


def rod_samples(m):
    return dict(rod_front_signed_rho=m["front_error"][m["rod"]],
                rod_edge_dual_rear=m["rear_error"][m["rod"]&m["dual"]],
                rod_interior_single_rear=m["rear_error"][m["rod"]&~m["dual"]])


def subject_metrics(m, mask):
    return dict(pixels=int(mask.sum()), rear_selected_fraction=float(m["selected"][mask].mean()) if mask.any() else None,
                beta_source={label:int((mask&(m["beta_source"]==i)).sum()) for i,label in
                             enumerate(("rho2","DepthSplat","triangulation"))},
                residual_saturated_fraction=float(m["saturated"][mask].mean()) if mask.any() else None,
                **{k:median(m[k][mask]) for k in
                   ("front","rear","rear_raw","gt","beta","residual","applied_residual","g","a")})


def focus_subject(unit, reference, case):
    return optics.ref.near_subject(reference["gt"],reference["valid"],cameras(unit)[0],case["focus_m"])


def nominal_case(scene, frame):
    unit = formal_unit(scene)
    if not unit:
        raise FileNotFoundError(f"No formal geometry for {scene}")
    folder = FORMAL_DATA/"optical"/scene
    audit = unit/"labels/radiance-audit.json" if scene.startswith("development-") else folder/"labels/optical-audit.json"
    identifier = f"f{frame:03d}-focus1-N1.4"
    row = next((r for r in load(audit)["optical"] if r["id"]==identifier),None)
    if row is None:
        raise FileNotFoundError(f"No registered far f/1.4 optical case for {scene}/{frame}")
    case = dict(id=identifier, frame=frame, focus_index=1, focus_m=row["focus_m"], fstop=1.4)
    return unit, folder if scene.startswith("training-") else unit, case


def truth_reference(scene, frame):
    item = ("B",scene,optics.base.ovg.TRAIN_FRAMES.index(frame))
    x,y = optics.base.sample(item)
    return x,optics.base.ovg.b_edge_reference(item,y)


def maps_for_panels(data):
    return {k:(v if v.dtype==bool else v.astype(np.float16)) for k,v in data.items()
            if k in ("front","rear","gt","rod","dual","front_error","rear_error")}


@torch.inference_mode()
def depth_scan(step):
    done = ROOT/"diagnostics/depth.json"
    if done.is_file():
        return load(done)
    optics.setup()
    model,_,_ = optics.checkpoint("v4",step,stitch=False)
    e = optics.EInputs(optics.v3.MODE)
    inv = dataset()
    prior = load(optics.EARLIER)["diagnostics"]["depth"]
    rods, subject = {}, {}
    for group in GROUPS:
        rods[group] = {}
        for scene in sorted(inv[group]):
            if not scene.startswith("development-" if group=="development" else "training-"):
                raise ValueError("Unexpected component diagnosis scene")
            groups = defaultdict(list)
            per_frame = {}
            for frame in (1,30,60):
                optics.gpu_gate(optics.GPU_GIB)
                data = fields(model,e.native(scene,frame))
                samples = rod_samples(data)
                for key,values in samples.items():
                    groups[key].append(values)
                per_frame[f"f{frame:03d}"] = {k:error_stats(v) for k,v in samples.items()}
                if group=="development" and optics.ref.title(scene)=="dev003":
                    unit,_,case = nominal_case(scene,frame)
                    _,native = truth_reference(scene,frame)
                    mask = focus_subject(unit,native,case)
                    subject[f"f{frame:03d}"] = dict(
                        **prior["dev003"][f"f{frame:03d}"],
                        v4=dict(gap_in_subject=subject_metrics(data,mask&data["gap"]),
                                normal_in_subject=subject_metrics(data,mask&data["normal"]),
                                gap_full_frame=subject_metrics(data,data["gap"]),subject_pixels=int(mask.sum())))
            scores = {k:error_stats(np.concatenate(v)) for k,v in groups.items()}
            if scores["rod_front_signed_rho"]["pixels"]:
                previous = prior["rods_by_scene"][group][scene]
                rods[group][scene] = dict(C=previous["C"],D=previous["D"],v4=scores,frames=per_frame)
            print("E0_V4_DEPTH",group,scene,flush=True)
    key = "rod_edge_dual_rear"
    rank = []
    for scene,row in rods["holdout"].items():
        present,old = row["v4"][key]["abs_p50"],row["C"][key]["abs_p50"]
        if present is not None and old is not None:
            rank.append(dict(scene=scene,delta_vs_C=present-old,v4=present,C=old))
    rank.sort(key=lambda v:(-v["delta_vs_C"],v["scene"]))
    result = dict(step=step, rods_by_scene=rods, dev003_subject=subject, holdout_rod_rank=rank,
        definitions=dict(front="signed forward component inverse depth minus GT center, in 1/m",
                         rear="absolute raw pre-order log inverse-depth error to GT depth2 (dual) or the v3 clipped single target",
                         rank="v4 minus C in pooled holdout scene dual-rod rear abs-p50",
                         zones="visible loss_zone 0 rod, 1 gap, 2 normal, 3 boundary"),
        scenes={k:len(inv[k]) for k in GROUPS})
    save_json(done,result)
    return result


def zoom_map(data, support, limit, box):
    shown = np.minimum(np.abs(data), limit).astype(np.float32)
    value = np.asarray(GALLERY.scalar_image(shown, limit, cv2.COLORMAP_INFERNO)).copy()
    value[~support]=(70,73,85)
    return Image.fromarray(value).crop(box)


def comparison_panel(path, name, images, depths, *, holdout=False):
    first=next(iter(images.values()))
    truth=first["truth"]
    shared=np.maximum.reduce([d["shared_mask"] for (f,_),d in images.items() if f=="G"])
    if shared.any():
        deviation=np.maximum.reduce([np.where(d["shared_mask"],d["hidden_error"],0).astype(np.float32)
                                     for (f,_),d in images.items() if f=="G"])
        picture=np.repeat(deviation[...,None],3,axis=-1)
        box=GALLERY.largest_error_crop(picture,np.zeros_like(truth),shared)
    else:
        support=np.maximum.reduce([d["rod"] for d in depths.values()])
        box=optics.ref.crop_for(first["reference"],truth,support)
    tile=155
    show_depth=any(d["rod"].any() for d in depths.values())
    rod_box = None
    if show_depth:
        rod = depths["v4"]["rod"]
        change = np.abs(np.abs(depths["v4"]["rear_error"].astype(np.float32))
                        - np.abs(depths["C"]["rear_error"].astype(np.float32)))
        rod_box = GALLERY.largest_error_crop(np.repeat(change[...,None],3,axis=-1),
                                             np.zeros_like(truth),rod)
    head=89
    scene_rows = []
    for focus,description in LENSES:
        scene_rows.append((f"{focus} {description} F2", {arm:GALLERY.rgb_image(d["F2"]).crop(box)
                           for (f,arm),d in images.items() if f==focus}))
        scene_rows.append((f"{focus} {description} S2", {arm:GALLERY.rgb_image(d["reference"]).crop(box)
                           for (f,arm),d in images.items() if f==focus}))
        scene_rows.append((f"{focus} 隐藏 RGB 差", {arm:zoom_map(d["hidden_error"],d["shared_mask"],.25,box)
                           for (f,arm),d in images.items() if f==focus}))
    if show_depth:
        hi=max(float(np.percentile(d["gt"][d["gt"]>0],99)) for d in depths.values())
        for name_key,key,limit in (("前分量逆深度","front",hi),("后分量逆深度","rear",hi),
                                  ("前分量误差","front_error",.5),("后分量 log 误差","rear_error",.5)):
            scene_rows.append((name_key,{arm:zoom_map(d[key],d["rod"],limit,rod_box) for arm,d in depths.items()}))
    picture=Image.new("RGB",(5*tile,head+(tile+18)*(len(scene_rows)+1)),(22,26,36))
    draw=ImageDraw.Draw(picture)
    draw.text((8,5),f"{name} 远焦对照"+(" · 留出诊断" if holdout else ""),font=GALLERY.font(20),fill="white")
    draw.text((8,35),"F2／S2 相同放大框。隐藏颜色差为 Query 与 S2 的共同命中区域，固定色标 0–0.25。",
              font=GALLERY.font(13),fill="#c5d6ee")
    draw.text((8,60),"细杆图用绿色框，灰色为掩码外。第四版的步骤按深度规则选定。",
              font=GALLERY.font(12),fill="#c5d6ee")
    whole=GALLERY.rgb_image(truth)
    borders=ImageDraw.Draw(whole)
    borders.rectangle(tuple(box),outline="#ffe76d",width=4)
    if rod_box:
        borders.rectangle(tuple(rod_box),outline="#54da91",width=3)
    picture.paste(whole.resize((tile,tile)),(0,head+16))
    draw.text((4,head),"Cycles 与同框位置",font=GALLERY.font(12),fill="white")
    for j,arm in enumerate(ARMS,1):
        main=GALLERY.rgb_image(images["G",arm]["reference"]) if ("G",arm) in images else None
        if main is not None:
            picture.paste(main.resize((tile,tile)),(j*tile,head+16))
        draw.text((j*tile+4,head),TITLES[arm],font=GALLERY.font(12),fill="white")
    for i,(row,items) in enumerate(scene_rows,1):
        y=head+i*(tile+18)
        draw.text((5,y+5),row,font=GALLERY.font(13),fill="white")
        if row.endswith((" F2"," S2")):
            picture.paste(GALLERY.rgb_image(truth).crop(box).resize((tile,tile)),(0,y+18))
        for j,arm in enumerate(ARMS,1):
            if arm in items:
                picture.paste(items[arm].resize((tile,tile),Image.Resampling.NEAREST),(j*tile,y+18))
    path.parent.mkdir(parents=True,exist_ok=True)
    picture.save(path)
    with Image.open(path) as check:
        check.verify()
    return dict(path=str(path.relative_to(ROOT)),crop=list(box),rod_crop=rod_box,includes_rod=show_depth)


def load_views(path):
    with np.load(path) as arrays:
        return {f:{k:arrays[f"{f}/{k}"] for k in ("truth","F2","reference","hidden_error","shared_mask")}
                for f,_ in LENSES}


def holdout_case(scene, frame, models, target, *, purpose):
    unit, optical_dir, case = nominal_case(scene,frame)
    x,native = truth_reference(scene,frame)
    subject = focus_subject(unit,native,case)
    if not subject.any() or not (f2_entry(unit,frame)/"manifest.json").is_file():
        raise ValueError("Selected holdout frame lacks its P2 subject or pinned F2 cache")
    truth = optics.ref.optical_target(optical_dir,scene,case)
    mask = optics.ref.new_regions(unit,case,native["gt"],native["valid"],cameras(unit)[0])["recoverable"]
    spec = dict(tiers="B",unit=unit,frame=frame,focus=case["focus_m"],fstop=1.4,
                label=unit/"labels"/f"f{frame:03d}-focus1-K16.npz")
    parents = hidden_parents(spec,ray_fields(spec),native)
    options = {}
    if parents is not None and len(parents["depth"]):
        point_evidence=SceneEvidence(unit,scene,views=[])
        _,options,*_ = substitute_variants(point_evidence,unit,scene,frame,case,parents=parents)
    else:
        parents=None
    net = optics.query_net()
    views,memory = optics.base.gallery.scene_views(dict(unit=unit))
    K = x["K"][0]
    rays = optics.base.aperture_disk(64,.05/(2*case["fstop"]),device=K.device,dtype=K.dtype)
    photos, components, counts = {}, {}, {}
    for arm,(network,tear,_) in models.items():
        optics.gpu_gate(optics.GPU_GIB)
        layer = optics.build_visible(network,tear,target,x)
        visible, depth, maximum = layer
        z=depth.cpu().numpy()
        foci=dict(P=optics.ref.p2_focus(z,subject),G=optics.ref.p2_focus(native["gt"],subject))
        bands={f:optics.base.optical_band(z,K,v,1.4) for f,v in foci.items()}
        store=optics.base.FrameQueries(views,optics.base.pack_frame(views,frame,z,list(bands.values()),memory))
        s2=optics.reference_cloud(unit,scene,frame,case,native,K,layer,local_witnesses=(parents,options))
        for focus,dist in foci.items():
            candidate=optics.base.hidden_cloud(net,store,dict(frame=frame,input="native",case=dict(case,focus_m=dist)),
                                               K,depth,bands[focus],placement_depth=maximum)
            fill=observed_background(spec,x,dist,rays)
            off,_,_=optics.render(visible,None,K,dist,rays,fill)
            reference,ref_rgb,ref_mass=optics.render(visible,s2,K,dist,rays,fill,trace=True)
            _,q_rgb,q_mass=optics.render(visible,candidate,K,dist,rays,fill,trace=True)
            error,valid=optics.hidden_error(q_rgb,q_mass,ref_rgb,ref_mass,mask)
            photos[focus,arm]=dict(F2=off,reference=reference,truth=truth,hidden_error=error,shared_mask=valid)
            counts[f"{focus}/{arm}"]=dict(F2=optics.score(off,truth,mask),reference=optics.score(reference,truth,mask))
            del candidate,fill
        components[arm]=maps_for_panels(fields(network,target))
        del visible,s2
    dest=ROOT/"figures"/f"{scene.split('-')[1]}-holdout-f{frame:03d}.png"
    return dict(scene=scene,frame=frame,qualification=purpose,diagnostic_scores=counts,
                **comparison_panel(dest,f"{scene.split('-')[1]} f{frame:03d}",photos,components,holdout=True))


@torch.inference_mode()
def panels(step):
    end=ROOT/"diagnostics/panels.json"
    if end.is_file():
        return load(end)
    depth=depth_scan(step)
    optics.setup()
    e=optics.EInputs(optics.v3.MODE)
    models={arm:optics.checkpoint(ver,s,stitch=True) for arm,(ver,s) in dict(MODEL_SPECS,v4=("v4",step)).items()}
    plots, skipped = [], []
    dev=dataset()["development"]
    for slug in optics.PANEL_SCENES:
        scene=next(n for n in dev if optics.ref.title(n)==slug)
        try:
            images={}
            for arm in ARMS:
                phase=ROOT/"full" if arm=="v4" else optics.PREVIOUS/arm
                path=phase/"frames"/scene/"f030.npz"
                for focus,arrays in load_views(path).items():
                    images[focus,arm] = arrays
            context=e.native(scene,30)
            depths={arm:maps_for_panels(fields(net,context)) for arm,(net,_,_) in models.items()}
            dest=ROOT/"figures"/f"{slug}-f030.png"
            plots.append(dict(scene=scene,frame=30,**comparison_panel(dest,slug+" f030",images,depths)))
        except (OSError,RuntimeError,KeyError,ValueError) as exc:
            skipped.append(dict(scene=scene,frame=30,reason=f"{type(exc).__name__}: {exc}"))
    selected=depth["holdout_rod_rank"][0]
    scene=selected["scene"]
    frame_scores=depth["rods_by_scene"]["holdout"][scene]["frames"]
    deltas={}
    key="rod_edge_dual_rear"
    for frame in (1,30,60):
        value=frame_scores[f"f{frame:03d}"][key]["abs_p50"]
        if value is not None:
            c=error_stats(rod_samples(fields(models["C"][0],e.native(scene,frame)))[key])["abs_p50"]
            deltas[frame]=value-c
    frame=max(deltas,key=deltas.get)
    try:
        plots.append(holdout_case(scene,frame,models,e.native(scene,frame),
                     purpose=dict(scene_rank=selected,ranked_frame_delta=deltas[frame],frame_delta=deltas)))
    except (OSError, RuntimeError, KeyError, ValueError, StopIteration) as exc:
        skipped.append(dict(scene=scene,frame=frame,reason=f"{type(exc).__name__}: {exc}"))
        torch.cuda.empty_cache()
    results=dict(figures=plots,failed=skipped,required=4)
    save_json(end,results)
    return results


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("depth","panels"))
    args=parser.parse_args()
    step=load(ROOT/optics.v3.MODE/"training.json")["selected"]["step"]
    {"depth":depth_scan,"panels":panels}[args.stage](step)
