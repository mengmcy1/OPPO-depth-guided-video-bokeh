"""Frozen-geometry Stage 3b training and the optical-gradient qualification."""
import argparse
from collections import Counter
from dataclasses import replace
from functools import lru_cache
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint as rematerialize

import route3_ovg_data as ovg
from route3_g3_r14_train import SETTINGS
from route3_ovg_footprint import query_net
from route3_ovg_subpixel import save_npz
from route3_v2data_B_batch import GIB, ROOT, reserve, sha256, write_json
from route3_v2data_B_inputs import observation
from route3_v2data_B_plan import load
from route3_v2data_B_training_share import register_training
from route3_v2data_loader import TrainingData, b_optical_cases
from route3_v2data_optics import linear_reference, observed_background
from route3_v2data_query_prepare import predict_context, prepare_scene
from route3_v2data_query_retrain import TrainingStore
from route3_v2data_stage1_evaluate import FrameQueries, load_paired_checkpoint, new_regions, view_from_observation
from route3_v2data_stage2_train import INITIAL, QUERY, now, sample
from route3_v2data_stage3a import choose_crop, gpu_setup, optical_pool, optical_specs
from route3_s2a_data import query_input_512
from src.aperture_renderer import GaussianScene, aperture_disk
from src.dual_layer import depth_band
from src.footprint_order import footprint_scale
from src.layered_aperture import merge_layers
from src.light_candidate_supervision import objectives
from src.ordered_layers import order_hidden
from src.ovg_optics import crop_camera, fixed_visible_surfels, raster

OUT = ROOT/"outputs/route3-v2data-stage3b-2026-11-25-01"
RECORD = ROOT/"03_实验记录/records/route3-v2data-stage3b-config-2026-11-25.json"
STAGE3A = RECORD.with_name("route3-v2data-stage3-config-2026-11-24.json")
ADOPTION = RECORD.with_name("route3-v2data-stage3a-v2-2026-11-24-step-020000.json")
SEED = SETTINGS["seeds"]["S2a"]
LR = .3*SETTINGS["learning_rate"]
FINAL_LR = .3*SETTINGS["final_learning_rate"]
SCHEDULE = (0,5000,10000,15000,20000)
GPU_GIB = 12
DIRECT = QUERY.parent.parent/"data"
FRACTION_B = .5
RENDER_RAYS = 64
RENDER_WEIGHTS = .001
FEATURES = ("query_pixel","query_xy","candidate_valid","features","source_slots","source_xy",
            "source_depth","target_K","target_c2w","target_depth")


def frozen_inputs():
    config = load(STAGE3A)
    decision = load(ADOPTION)["adoption"]
    geometry,query = sha256(INITIAL),sha256(QUERY)
    if decision["selected"] != "initial" or decision["adopted_sha256"] != geometry:
        raise ValueError("3a did not select the registered initial geometry")
    if (geometry,query) != (config["geometry"]["sha256"],config["query"]["sha256"]):
        raise ValueError("3b input hashes differ from the Stage 3 registration")
    return dict(geometry=dict(path=str(INITIAL),sha256=geometry,weights="EMA"),
                query=dict(path=str(QUERY),sha256=query))


def path_for(run, step):
    return run/"checkpoints"/f"step-{step:06d}.pt"


def atomic_checkpoint(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix(".part")
    torch.save(value,temporary)
    temporary.replace(path)


def load_geometry():
    return load_paired_checkpoint(INITIAL,"ema").eval().requires_grad_(False)


def hidden_cloud(network, store, row, K, depth, band, indices=None):
    """Preserve tensor outputs while keeping the existing discrete O1 decisions."""
    c = row["case"]
    if indices is None:
        indices = store.active(row["input"],c["focus_m"],c["fstop"])
    if not len(indices):
        return None
    parts = [store.forward(network,row,ix,{"size":depth.shape[-1]})[0]
             for ix in np.array_split(indices,(len(indices)+2047)//2048)]
    xyz = torch.cat([p["means_camera"].reshape(-1,3) for p in parts])
    colors = torch.cat([p["colors"].reshape(-1,3) for p in parts])
    opacity = torch.cat([p["opacities"].flatten() for p in parts])
    projected = xyz@K.T
    uv = projected[:,:2]/projected[:,2:].clamp_min(1e-8)
    height,width = depth.shape
    ix = uv[:,0].floor().long().clamp(0,width-1)
    iy = uv[:,1].floor().long().clamp(0,height-1)
    inside = ((xyz[:,2]>0)&(opacity>1e-4)&(uv>=0).all(-1)
              &(uv[:,0]<width)&(uv[:,1]<height)&torch.as_tensor(band,device=K.device)[iy,ix])
    if not inside.any():
        return None
    xyz,colors,opacity = xyz[inside],colors[inside],opacity[inside]
    sigma = xyz[:,2]/K[0,0]*.7
    scales = torch.stack((sigma,sigma,sigma*.01),-1)
    quats = xyz.new_zeros((len(xyz),4))
    quats[:,0] = 1
    cloud = GaussianScene(xyz,quats,scales,opacity,colors,torch.ones_like(opacity,dtype=torch.long))
    ordered,_ = order_hidden(cloud,depth,K,"O1")
    with torch.no_grad():
        factor,keep = footprint_scale(ordered.means,ordered.scales[:,:2],ordered.scales[:,2],depth,K)
    scales = torch.cat((ordered.scales[keep,:2]*factor[keep,None],ordered.scales[keep,2:]),1)
    return GaussianScene(ordered.means[keep],ordered.quats[keep],scales,ordered.opacities[keep],
                         ordered.colors[keep],ordered.source_frames[keep]) if keep.any() else None


@lru_cache(maxsize=2)
def observed_views(unit):
    return [view_from_observation(observation(unit,f,aligned=True),f) for f in ovg.TRAIN_FRAMES]


def pack_frame(views, frame, z, bands, memory=None):
    anchor = next(v for v in views if v["frame"] == frame)
    packed,_ = query_input_512(views,dict(anchor,depth=z),np.logical_or.reduce(bands),8,z.size,memory)
    return {k:packed[k] for k in FEATURES}


def optical_band(z, K, focus, fstop):
    return depth_band(z,float(K[0,0]),focus,.05/(2*fstop))[0]


class BOptical:
    def __init__(self,run,backbone):
        self.run = run
        self.backbone = backbone

    @lru_cache(maxsize=8)
    @torch.no_grad()
    def frame(self,unit,frame):
        item = ("B",unit.name,ovg.TRAIN_FRAMES.index(frame))
        x,y = sample(item)
        K = x["K"][0]
        path = self.run/"native-query"/unit.name/f"f{frame:03d}.npz"
        if path.is_file():
            packed = ovg.load(path)
            z = packed["target_depth"]
        else:
            z = self.backbone(x,protect_residual=True,render_heads=False)["depth1"][0].cpu().numpy()
            cases = b_optical_cases(unit)
            bands = [optical_band(z,K,s["focus"],s["fstop"]) for s in cases if s["frame"]==frame]
            if not bands:
                raise ValueError(f"Missing admitted optical cases for {unit.name} f{frame:03d}")
            packed = pack_frame(observed_views(unit),frame,z,bands)
            path.parent.mkdir(parents=True,exist_ok=True)
            save_npz(path,**packed)
        return x,y,K.new_tensor(z),packed

    def batch(self, network, item, spec, crop, *, count_rays=False):
        unit,frame,focus,fstop = (spec[k] for k in ("unit","frame","focus","fstop"))
        x,y,depth,packed = self.frame(unit,frame)
        K = x["K"][0]
        band = optical_band(depth.cpu().numpy(),K,focus,fstop)
        indices = np.flatnonzero(band.ravel()[packed["query_pixel"]] & packed["candidate_valid"].any(-1))
        # Keep candidates selected from the native annulus. Cropping happens in
        # the camera, so off-crop surfels still contribute via the pupil shift.
        store = FrameQueries(observed_views(unit),packed)
        row = dict(frame=frame,input="native",case=dict(focus_m=focus,fstop=fstop))
        hidden = hidden_cloud(network,store,row,K,depth,band,indices)
        visible = fixed_visible_surfels({"depth1":depth[None]},x)
        offsets = aperture_disk(RENDER_RAYS,.05/(2*fstop),device=K.device,dtype=K.dtype)
        fill = observed_background(spec,x,focus,offsets,crop=crop)
        top,left,h,w = crop
        kc = crop_camera(K,crop)
        target = K.new_tensor(linear_reference(spec)[top:top+h,left:left+w])
        if hidden is None:
            with torch.no_grad():
                rgb = raster(visible,kc,focus,offsets,h,w,fill).clamp(0,1).mean(0)
            signal = sum(p.sum()*0 for p in network.parameters())
        else:
            combined = merge_layers(visible,hidden)

            def ray(means,scales,opacity,colors,offset,background):
                cloud = replace(combined,means=means,scales=scales,opacities=opacity,colors=colors)
                return raster(cloud,kc,focus,offset,h,w,background)[0].clamp(0,1)

            parts = [rematerialize(ray,combined.means,combined.scales,combined.opacities,combined.colors,
                                   offsets[j:j+1],fill[j:j+1],use_reentrant=False)
                     for j in range(len(offsets))]
            rgb = torch.stack(parts).mean(0)
            signal = 0.
        support = dict(queries=len(indices),hidden_surfels=len(hidden.means) if hidden is not None else 0)
        if count_rays:
            with torch.no_grad():
                reference = ovg.b_edge_reference(item,y)
                case = dict(frame=frame,focus_index=int(spec["label"].stem.split("-focus")[1][0]),
                            focus_m=focus,fstop=fstop)
                mask = new_regions(unit,case,reference["gt"],reference["valid"],K.cpu().numpy())["recoverable"]
                mask = mask[top:top+h,left:left+w]
                _,_,hit = traced_render(visible,hidden,kc,focus,offsets,fill,(h,w),mask)
                support.update(hit)
        return (rgb-target).square().mean()+signal,support


def traced_render(visible,hidden,K,focus,offsets,fill,shape,mask):
    """Composite a hidden-membership feature with exactly the same alpha weights."""
    cloud = merge_layers(visible,hidden)
    indicator = cloud.colors.new_zeros((len(cloud.means),1))
    if hidden is not None:
        indicator[len(visible.means):] = 1.
    cloud = replace(cloud,colors=torch.cat((cloud.colors,indicator),1))
    h,w = shape
    hidden_sum = K.new_zeros((h,w))
    f2_sum = K.new_zeros((h,w))
    total = K.new_zeros((h,w,3))
    active = torch.as_tensor(mask,device=K.device)
    count = 0
    for i in range(0,len(offsets),4):
        output,alpha = raster(cloud,K,focus,offsets[i:i+4],h,w,return_alpha=True)
        residual = 1-alpha
        weight = output[...,3]
        total += (output[...,:3]+residual[...,None]*fill[i:i+4]).clamp(0,1).sum(0)
        hidden_sum += weight.sum(0)
        f2_sum += residual.sum(0)
        count += int(((weight>=RENDER_WEIGHTS)&active).sum())
    n = len(offsets)
    return total/n,dict(hidden=hidden_sum/n,f2=f2_sum/n),dict(
        effective_hidden_rays=count,total_region_rays=int(active.sum())*n)


class DirectBatches:
    def __init__(self,run,geometry):
        self.run = run
        self.geometry = geometry
        self.geometry_sha = sha256(INITIAL)
        self.store = TrainingStore()
        prepared = load(DIRECT/"prepared.json")
        self.paths = {name:Path(row["path"]) for name,row in prepared["accepted"].items() if row["split"]=="train"}
        self.c_names = sorted(name for name,row in prepared["accepted"].items()
                              if row["tier"]=="C" and row["split"]=="train")
        self.skipped = load(run/"skipped-caches.json") or {}

    def manifest(self,name,unit=None):
        if name not in self.paths:
            if unit is None or name in self.skipped:
                return None
            try:
                reserve(self.run,GIB)
                views = predict_context(self.geometry,"B",unit)
                result = prepare_scene(("B","train",unit),self.run/"direct-cache",views,None,None,self.geometry_sha)
            except (ValueError,KeyError,OSError) as error:
                self.skipped[name] = str(error)
                write_json(self.run/"skipped-caches.json",self.skipped)
                return None
            self.paths[name] = Path(result["path"])
        value = load(self.paths[name])
        if value["geometry_sha256"] != self.geometry_sha or value["split"]!="train":
            raise ValueError(f"Direct label provenance mismatch for {name}")
        return value

    def loss(self,network,manifest,rng,spec=None):
        choices = manifest["rows"]
        if spec:
            idx = int(spec["label"].stem.split("-focus")[1][0])
            choices = [r for r in choices if r["frame"]==spec["frame"] and r["case"]["focus_index"]==idx]
            if not choices:
                choices = [r for r in manifest["rows"] if r["case"]["focus_index"]==idx] or manifest["rows"]
        row = choices[int(rng.integers(len(choices)))]
        active = self.store.active(row["input"],row["case"]["focus_m"],row["case"]["fstop"])
        if len(active)!=row["active_queries"]:
            raise ValueError(f"Stored direct Query support changed for {manifest['scene']}/{row['case']['id']}")
        ids = rng.choice(active,size=SETTINGS["batch_queries"],replace=len(active)<SETTINGS["batch_queries"])
        pred,valid,_ = self.store.forward(network,row,ids,manifest)
        value,parts = objectives(pred,self.store.target_batch(row,ids),valid,balance_activation=True)
        return value,{k:float(v.detach()) for k,v in parts.items()}


def gradient_norms(network, gradients):
    def group(name,tensor):
        if name.startswith("geometry."):
            return (("selection",tensor[:2]),("position",tensor[2:4]),("quality",tensor[4:]))
        if name.startswith("appearance.2."):
            return (("color",tensor[:3]),("opacity",tensor[3:]))
        return ((name.split(".")[0],tensor),)
    parts = {}
    for (name,parameter),gradient in zip(network.named_parameters(),gradients):
        for key,value in group(name,gradient if gradient is not None else torch.zeros_like(parameter)):
            parts.setdefault(key,[]).append(value.float().norm())
    return {key:float(torch.linalg.vector_norm(torch.stack(values))) for key,values in parts.items()}


def qualify(run,geometry,network,optics,direct,inventory):
    pool = optical_pool(inventory)
    initial = [(k,v.detach().cpu().clone()) for k,v in geometry.state_dict().items()]
    params = tuple(network.parameters())
    names = pool["hard"] + [n for n in pool["scenes"] if n not in pool["hard"]]
    for name in names:
        if name not in pool["thin"]:
            continue
        for spec in optical_specs(inventory,name):
            if (spec["fstop"]!=1.4 or "focus1" not in spec["truth"].stem
                    or spec["frame"] not in pool["thin"][name]):
                continue
            item = ("B",name,ovg.TRAIN_FRAMES.index(spec["frame"]))
            rng = np.random.default_rng([SEED,3,2])
            manifest = direct.manifest(name,spec["unit"])
            if manifest is None:
                continue
            crop,thin = choose_crop(spec,rng,True)
            torch.cuda.reset_peak_memory_stats()
            t = time.monotonic()
            optical,support = optics.batch(network,item,spec,crop,count_rays=True)
            if not support["effective_hidden_rays"]:
                del optical
                continue
            q,terms = direct.loss(network,manifest,rng,spec)
            bgrad = torch.autograd.grad(optical,params,allow_unused=True)
            qgrad = torch.autograd.grad(q,params,allow_unused=True)
            b = gradient_norms(network,bgrad)
            d = gradient_norms(network,qgrad)
            bn,dn = (float(np.linalg.norm(list(groups.values()))) for groups in (b,d))
            if np.isfinite((bn,dn)).all() and bn > 0 and dn > 0:
                probe_opt = torch.optim.Adam(params,lr=LR)
                for p,gb,gq in zip(params,bgrad,qgrad):
                    p.grad = (torch.zeros_like(p) if gb is None else gb)+(0. if gq is None else bn/dn*gq)
                probe_opt.step()
            unchanged = all(torch.equal(v,geometry.state_dict()[k].cpu()) for k,v in initial)
            no_grad = all(not p.requires_grad and p.grad is None for p in geometry.parameters())
            passed = (unchanged and no_grad and np.isfinite(list(b.values())+list(d.values())).all()
                      and min(b["color"],b["opacity"],b["position"],dn)>0)
            torch.cuda.synchronize()
            report = dict(passed=bool(passed),scene=name,case=spec["truth"].stem,crop=list(crop),thin_pixels=thin,
                          geometry_unchanged=unchanged,geometry_no_grad=no_grad,L_bokeh=float(optical.detach()),
                          direct_loss=float(q.detach()),direct_terms=terms,gradient_norms=dict(L_bokeh=b,direct=d),
                          unweighted_gradient_norms=dict(L_bokeh=bn,direct=dn),support=support,
                          seconds=time.monotonic()-t,peak_vram_reserved_bytes=torch.cuda.max_memory_reserved())
            write_json(run/"gradient-check.json",report)
            print("STAGE3B_GRADIENT",report,flush=True)
            if not passed:
                raise RuntimeError("The isolated optical loss must reach color, opacity and position")
            return report
    raise RuntimeError("No far-focus B crop put actual hidden weight on a recoverable ray")


def setup(run):
    run.mkdir(parents=True,exist_ok=True)
    gpu_setup(dict(training=dict(max_process_gpu_gib=GPU_GIB)))
    hashes = frozen_inputs()
    geometry = load_geometry()
    network = query_net(QUERY)
    return hashes,geometry,network,BOptical(run,geometry),DirectBatches(run,geometry),TrainingData()


def register(run,hashes,gradient):
    if not gradient.get("passed"):
        raise ValueError("Missing passing optical gradient qualification")
    norms = gradient["unweighted_gradient_norms"]
    weight = norms["L_bokeh"]/norms["direct"]
    path = run/"config.json"
    if path.is_file():
        return load(path)
    cfg = dict(stage="3b",created_at=now(),inputs=hashes,
               renderer="V0, b≡0, O1 footprint order, F2, K64. Crop at native resolution. P2 evaluation.",
               training=dict(optimizer="Adam",learning_rate=LR,final_learning_rate=FINAL_LR,decay="cosine",
                             optical_weight=1.,direct_weight=weight,weighted_direct_to_optical_gradient_ratio=1.,
                             steps=SCHEDULE[-1],eval_steps=list(SCHEDULE),EMA=.999,scenes_per_batch=1,
                             direct_queries=SETTINGS["batch_queries"],seed=SEED,refresh_steps=500,
                             fraction_B_optical=FRACTION_B,fraction_C_direct=1-FRACTION_B,
                             max_process_gpu_gib=GPU_GIB,reason="3a B optical share; remaining draws use the prior Query C direct-label pool."),
               code_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
               source_sha256=sha256(Path(__file__)),gradient_check=gradient,
               adoption=dict(far_paired_median_min_db=0.,near_pooled_drop_max_db=.1,
                             zero_visible_floor_db=50.,zero_combined_floor_db=60.,
                             minimum_far_hidden_ray_ratio_vs_start=.8))
    write_json(RECORD,cfg)
    write_json(path,cfg)
    return cfg


def draw_optical(inventory,pool,direct,rng,require_thin):
    options = [n for n in (pool["thin"] if require_thin else pool["scenes"]) if n not in direct.skipped]
    while options:
        weights = np.where(np.isin(options,pool["hard"]),2.,1.)
        name = str(rng.choice(options,p=weights/weights.sum()))
        manifest = direct.manifest(name,inventory.b[name])
        if manifest is None:
            options.remove(name)
            continue
        specs = optical_specs(inventory,name)
        if require_thin:
            specs = [r for r in specs if r["frame"] in pool["thin"][name]]
        spec = specs[int(rng.integers(len(specs)))]
        return ("B",name,ovg.TRAIN_FRAMES.index(spec["frame"])),manifest,spec
    raise RuntimeError("No B optical scene has matching frozen Query labels")


def train(run,components):
    hashes,geometry,network,optics,direct,inventory = components
    cfg = load(run/"config.json")
    if not cfg or (cfg["inputs"]!=hashes) or not load(run/"gradient-check.json").get("passed"):
        raise ValueError("Register the qualified Stage 3b configuration before training")
    config_sha = sha256(run/"config.json")
    opt = torch.optim.Adam(network.parameters(),lr=LR)
    ema = {key:p.detach().clone() for key,p in network.named_parameters()}
    snapshots = sorted((run/"checkpoints").glob("step-*.pt"))
    start,prior,counts = 1,0.,Counter()
    if snapshots:
        saved = torch.load(snapshots[-1],map_location="cpu",weights_only=True)
        if saved["config_sha256"]!=config_sha:
            raise ValueError("Stage 3b config changed since the last checkpoint")
        network.load_state_dict(saved["model"])
        opt.load_state_dict(saved["optimizer"])
        for key in ema:
            ema[key].copy_(saved["ema"][key])
        start,prior = saved["step"]+1,saved["seconds"]
        counts.update(saved["counts"])
    else:
        atomic_checkpoint(path_for(run,0),dict(step=0,model=network.state_dict(),ema=network.state_dict(),
                          optimizer=opt.state_dict(),seconds=0.,counts={},config_sha256=config_sha))
    training = cfg["training"]
    frozen = {name:p._version for name,p in geometry.named_parameters()}
    begin = time.monotonic()
    progress = run/"progress.json"
    register_training("3b",{"3b":os.getpid()},{"3b":str(progress)})
    try:
        for step in range(start,SCHEDULE[-1]+1):
            base = ((step-1)//training["refresh_steps"])*training["refresh_steps"]
            if step==start or step==base+1:
                directory = run/"reference-pools"
                directory.mkdir(exist_ok=True)
                path = directory/f"step-{base:06d}.json"
                if not path.is_file():
                    inventory = TrainingData()
                    write_json(path,optical_pool(inventory))
                pool = load(path)
            rng = np.random.default_rng([SEED,3,step])
            is_b = step%2==1
            opt.param_groups[0]["lr"] = FINAL_LR+(LR-FINAL_LR)*.5*(1+np.cos(np.pi*(step-1)/SCHEDULE[-1]))
            opt.zero_grad(set_to_none=True)
            network.train()
            if is_b:
                required = counts["B"]%2==0
                item,manifest,spec = draw_optical(inventory,pool,direct,rng,required)
                crop,thin = choose_crop(spec,rng,required)
                optical,support = optics.batch(network,item,spec,crop)
                q,direct_terms = direct.loss(network,manifest,rng,spec)
                loss = optical+training["direct_weight"]*q
                counts.update(B=1,thin_crops=int(thin>0),thin_pixels=thin)
                case = spec["truth"].stem
                counts[f"B/{item[1]}/{case}"] += 1
            else:
                name = direct.c_names[int(rng.integers(len(direct.c_names)))]
                manifest = direct.manifest(name)
                q,direct_terms = direct.loss(network,manifest,rng)
                loss = training["direct_weight"]*q
                optical,support,case,crop,thin = None,{},None,None,0
                counts.update(C=1)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite Stage 3b loss at step {step}")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(network.parameters(),1.,error_if_nonfinite=True)
            opt.step()
            with torch.no_grad():
                for key,p in network.named_parameters():
                    ema[key].lerp_(p,1-training["EMA"])
            elapsed = prior+time.monotonic()-begin
            row = dict(step=step,scene=manifest["scene"],case=case,crop=crop,thin_pixels=thin,
                       L_bokeh=None if optical is None else float(optical.detach()),direct=direct_terms,
                       total=float(loss.detach()),gradient_norm=float(norm),support=support)
            with (run/"steps.jsonl").open("a") as f:
                f.write(json.dumps(row)+"\n")
            if step==1 or step%25==0:
                write_json(progress,dict(step=step,seconds=elapsed,at=now(),counts={k:counts[k] for k in ("B","C","thin_crops","thin_pixels")},
                                         peak_vram_reserved_bytes=torch.cuda.max_memory_reserved()))
                print("STAGE3B",step,row["total"],round(elapsed,2),flush=True)
            if step%500==0:
                if any(p.requires_grad or p.grad is not None or p._version != frozen[key]
                       for key,p in geometry.named_parameters()):
                    raise RuntimeError("Frozen geometry unexpectedly changed or took gradients")
                atomic_checkpoint(path_for(run,step),dict(step=step,model=network.state_dict(),
                    ema=dict(network.state_dict(),**ema),optimizer=opt.state_dict(),seconds=elapsed,
                    counts=dict(counts),config_sha256=config_sha))
    finally:
        register_training("3b",{},{})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,default=OUT)
    parser.add_argument("--mode",choices=("qualify","register","train"),required=True)
    args = parser.parse_args()
    if args.mode=="register":
        register(args.run,frozen_inputs(),load(args.run/"gradient-check.json"))
    else:
        components = setup(args.run)
        if args.mode=="qualify":
            qualify(args.run,*components[1:])
        else:
            train(args.run,components)
