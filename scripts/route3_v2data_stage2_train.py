"""Stage 2 appearance training on the expanded A, B and C data."""
import argparse
from collections import Counter
from dataclasses import replace
from datetime import datetime
from functools import lru_cache
import json
import os
from pathlib import Path
import time
from zipfile import BadZipFile
from zoneinfo import ZoneInfo

import cv2
import numpy as np
import torch
from torch.nn import functional as F

import route3_ovg_data as ovg
from route3_ovg_1b_common import CONFIG, ROOT
from route3_ovg_1c import aligned_sample
from route3_ovg_stage2_query import FrozenQuery, views_for
from route3_ovg_stage2_train import SEED, SIZE, WEIGHTS, output_state, tensor
from route3_ovg_subpixel import save_npz
from route3_v2data_a_build import OnlineA, appearance_losses
from route3_v2data_B_batch import FORMAL_DATA, GIB, reserve, sha256, write_json
from route3_v2data_B_inputs import observation
from route3_v2data_B_plan import load
from route3_v2data_B_training_share import register_training
from route3_v2data_loader import b_k16_cases, b_optical_cases, c_optical_specs, registered_b, window_inventory
from route3_v2data_optics import f2_entry, linear_reference, observed_background, occupancy, ray_fields
from route3_v2data_stage1_evaluate import load_paired_checkpoint, view_from_observation
from src.layered_aperture import merge_layers
from src.ovg_appearance import AppearanceHeads, boundary_layer, cutoff
from src.ovg_first_pass import boundary_candidate, observed_split
from src.ovg_net import geometry_loss, masked_mean
from src.ovg_optics import fixed_visible_surfels, occlusion_loss, raster, stratified_pupil, subcamera
from src.prefix_transmittance import differentiable_prefix_transmittance

RUN = ROOT/"outputs/route3-v2data-stage2-2026-11-23-02"
STAGE1 = ROOT.parent/"route3-stage1-pair-20261122/outputs/route3-v2data-stage1-2026-11-22-01"
INITIAL = STAGE1/"1b/same-surface/checkpoints/step-030000.pt"
QUERY = ROOT.parent/"route3-query-v2-20261123/outputs/route3-query-v2-20261123/train/step-020000.pt"
QUERY_RECORD = ROOT/"03_实验记录/records/route3-v2data-query-2026-11-23.json"
ZONE = ZoneInfo("Asia/Shanghai")
EPSILON = .40
PHASE_PLAN = {"2a": (10_000, {"A": 7, "C": 3}), "2b": (30_000, {"C": 4, "B": 4, "A": 2})}
CHECKPOINT_EVERY = 2000
LABEL_KEYS = ("foreground", "background", "full", "color_valid", "comp_valid")
B_REFERENCE_POOL = None


def now():
    return datetime.now(ZONE).isoformat(timespec="seconds")


def frame_number(item):
    return ovg.TRAIN_FRAMES[item[2]] if item[0] == "B" else item[2]


def fixed_inputs():
    r = load(QUERY_RECORD)
    chosen = r["first_pass_final"]["q_positive_rule"]
    if chosen["epsilon"]!=EPSILON or not chosen["observed_background_required"]:
        raise ValueError("The frozen first pass requires epsilon=.40 and a background candidate")
    geometry_sha, query_sha = sha256(INITIAL), sha256(QUERY)
    if (geometry_sha, query_sha) != (r["stage1_geometry_sha256"], r["new_query_sha256"]):
        raise ValueError("Frozen stage 1 or Query checkpoint differs from the registered selection")
    return dict(geometry_sha256=geometry_sha,query_sha256=query_sha,epsilon=EPSILON,
                observed_background_required=True,rule="candidate_mask and q > 0")


def configuration():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("Select the shared RTX 5090 explicitly")
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    os.environ.setdefault("TORCH_HOME",str(ROOT.parent/".cache/torch"))
    torch.cuda.set_per_process_memory_fraction(15*GIB/torch.cuda.get_device_properties(0).total_memory, 0)


def occurrence_slots(d, a, stage):
    return {"B": [("B", name, i) for name in d for i in range(len(ovg.TRAIN_FRAMES))],
            "C": a[stage]}


class Dataset:
    def __init__(self):
        self.a = {p: OnlineA(p) for p in ("train", "dev")}
        self.b, exclusions = registered_b()
        self.exclusions = {name: dict(tier="B", reason=reason) for name, reason in exclusions.items()}
        self.b = {name: path for name, path in self.b.items() if self.check_b(name, path)}
        self.c = {}
        for role in ("train", "dev"):
            self.c[role] = []
            for item in window_inventory(role):
                fname = Path(item[1])
                label = ROOT/c_metadata(fname)["rows"][item[2]]["label"]
                if label.is_file():
                    self.c[role].append(item)
                else:
                    self.exclusions[fname.parent.name] = dict(tier="C", reason=f"Missing input label {label.name}")
            self.c[role] = [item for item in self.c[role] if Path(item[1]).parent.name not in self.exclusions]
        self.train = occurrence_slots(self.b,self.c,"train")

    def check_b(self, scene, path):
        for f in ovg.TRAIN_FRAMES:
            for suffix in ("colors", "pinhole"):
                if not (path/"labels"/f"f{f:03d}-{suffix}.npz").is_file():
                    self.exclusions[scene] = dict(tier="B", reason=f"Missing f{f:03d}-{suffix}.npz")
                    return False
        return True

    def report(self):
        return dict(training_scenes={"A":len(self.a["train"].foreground), "B":len(self.b),
                                     "C":len({x[1] for x in self.c["train"]})},
                    development_scenes={"A":len(self.a["dev"].foreground), "B":15,
                                        "C":len({x[1] for x in self.c["dev"]})},
                    skipped=self.exclusions)


@lru_cache(maxsize=32)
def c_metadata(path):
    return load(Path(path))


@lru_cache(maxsize=8)
def rebuilt_c(path):
    """Repair missing appearance fields from the released analytical cards recipe."""
    from route3_ovg_card import lens_offsets
    from route3_v2data_c_build import OUT, from_numpy, geometry_from_hits, load_materials
    from route3_v2data_c_trace import trace_pinhole

    r = c_metadata(path)
    spec = load(ROOT/r["geometry_label_recipe"])
    entries = {row["key"]:row for row in load(OUT/"assets.json")["assets"]}
    _, textures, slots = load_materials([entries[k] for k in dict.fromkeys(spec["assets"])])
    mats = [[*row[:3], slots[k], row[3]] for row,k in zip(spec["material_rows"],spec["assets"])]
    rect, poses, motion, materials = map(from_numpy, (spec["rectangles"], spec["camera_origins"],
                                                      spec["foreground_translation_x"], mats))
    ids, colors, peeled = trace_pinhole(rect, poses, motion, textures, materials, spec["perforated"])
    _, _, labels, _ = geometry_from_hits(rect, poses, ids, colors, peeled)
    return {key:labels[key] for key in LABEL_KEYS}


@lru_cache(maxsize=12)
def appearance_labels(item):
    tier, key, index = item
    f = frame_number(item)
    if tier == "B" and ovg.formal_unit(key):
        folder = ovg.formal_unit(key)/"labels"
        d = ovg.load(folder/f"f{f:03d}-colors.npz")
        return dict(foreground=d["foreground_rgb"].astype(np.float32),
                    background=d["complementary_background_rgb"].astype(np.float32),
                    color_valid=d["foreground_valid"], comp_valid=d["compositing_valid"])
    if tier == "B":
        from route3_ovg_stage2_train import label_appearance
        return label_appearance(tier, key, index)
    path = Path(key)
    label = ROOT/c_metadata(path)["rows"][index]["label"]
    with np.load(label, allow_pickle=False) as d:
        if set(LABEL_KEYS).issubset(d.files):
            return {k:d[k].astype(np.float32) if d[k].dtype.kind == "f" else d[k] for k in LABEL_KEYS}
    return {k:v[index] for k,v in rebuilt_c(path).items()}


def sample(item):
    x,y = aligned_sample(item, "cuda")
    return x,{k:v.float() if v.is_floating_point() else v for k,v in y.items()}


def candidate_mask(appearance):
    return boundary_candidate(appearance["alpha"][0],
                              appearance["depth2"][0]>=1.03*appearance["depth1"][0],EPSILON)


def split_mask(appearance, first):
    return observed_split(appearance["alpha"][0],appearance["depth2"][0]>=1.03*appearance["depth1"][0],
                          tensor(first["q"]),EPSILON)


def subpixels(x):
    return x.repeat_interleave(5,0).repeat_interleave(5,1)


def visible(appearance, inputs, split, stage):
    base = fixed_visible_surfels(appearance, inputs)
    rgb = inputs["rgb"][0].permute(1,2,0)
    color = torch.where(split[...,None], appearance["foreground"][0].permute(1,2,0), rgb)
    opacity = torch.where(split, appearance["opacity"][0], .999)
    if stage == "2a":
        scales = base.scales
    else:
        radius = appearance["radius_multiplier"][0].permute(1,2,0)
        multiplier = torch.where(split[...,None], radius, 1.)
        scales = torch.cat((base.scales[:,:2]*subpixels(multiplier).reshape(-1,2),base.scales[:,2:]),-1)
    return replace(base, colors=subpixels(color).reshape(-1,3), opacities=subpixels(opacity).reshape(-1), scales=scales)


def background(appearance, inputs, split, first):
    if not split.any():
        return None
    if not (tensor(first["q"])[split] > 0).all():
        raise ValueError("A first-pass split lacks its background candidate")
    return boundary_layer(appearance, inputs, split, tensor(first["color"]), tensor(first["depth"]))


def low_resolution(x, appearance, first):
    if x["rgb"].shape[-1] == SIZE:
        return x, appearance, first
    h = x["rgb"].shape[-1]
    K = x["K"].clone()
    K[:,:2] *= SIZE/h
    small = dict(x, rgb=F.interpolate(x["rgb"], (SIZE,SIZE), mode="area"), K=K)
    projected = {}
    for key in ("depth1", "depth2", "foreground", "alpha", "opacity", "radius_multiplier"):
        value = appearance[key]
        mono = value.ndim == 3
        resized = F.interpolate(value[:,None] if mono else value,(SIZE,SIZE),mode="nearest-exact")
        projected[key] = resized[:,0] if mono else resized
    first = {key:cv2.resize(v,(SIZE,SIZE),interpolation=cv2.INTER_NEAREST_EXACT)
             for key,v in first.items() if key in ("q","color","depth")}
    return small, projected, first


class Model:
    def __init__(self, run=RUN):
        self.run = run
        self.backbone = load_paired_checkpoint(INITIAL,"ema").eval().requires_grad_(False)
        self.heads = AppearanceHeads(self.backbone.coverage).cuda()
        self.query = FrozenQuery(QUERY)

    @torch.no_grad()
    def geometry(self, x):
        return self.backbone(x, protect_residual=True, render_heads=False, return_features=True)

    @lru_cache(maxsize=2)
    def context(self, tier, key):
        unit = ovg.formal_unit(key) if tier == "B" else None
        if unit:
            return [view_from_observation(observation(unit, f, aligned=True), f) for f in ovg.TRAIN_FRAMES],None
        return views_for(tier,key,512 if tier == "B" else SIZE)

    @torch.no_grad()
    def first(self, item, g):
        tier, key, index = item
        name = key if tier == "B" else Path(key).parent.name
        path = self.run/"observed-first"/tier/name/f"f{frame_number(item):03d}.npz"
        if path.is_file():
            return ovg.load(path)
        context, memory = self.context(tier,key)
        z1,z2 = (g[k][0].cpu().numpy() for k in ("depth1","depth2"))
        data = self.query.first_from_views(context,memory,frame_number(item),z1,z2)
        path.parent.mkdir(parents=True,exist_ok=True)
        save_npz(path, **data)
        return data

    def appearance(self, g, x, first):
        memory = tensor(first["memory"]).permute(2,0,1)[None]
        q = tensor(first["memory_q"])[None]
        return self.heads(g,x["rgb"],memory,q)

    @torch.no_grad()
    def second(self, item, depth, focus, fstop):
        from route3_ovg_stage2_query import smaller_views
        tier, key, _ = item
        context,memory = self.context(tier,key)
        if context[0]["depth"].shape != depth.shape:
            context,memory = smaller_views(context,len(depth)),None
        return self.query.second_from_views(context,memory,frame_number(item),depth,focus,fstop)


def a_objective(model, source, seed, phase):
    inputs, labels, names = source.sample(seed, stage=phase)
    rgb = tensor(inputs["rgb"].transpose(2,0,1)[None])
    h,w = rgb.shape[-2:]
    depth = rgb.new_zeros((1,3,h,w))
    x = dict(rgb=rgb, prior_depth=depth, prior_conf=depth.clone(),
             motion=torch.zeros((1,h,w),device=rgb.device,dtype=torch.bool),
             neighbors=rgb.new_zeros((1,0,3,h,w)))
    output = model.appearance(model.geometry(x),x,dict(memory=inputs["memory"],memory_q=inputs["memory_confidence"]))
    terms = appearance_losses(output["foreground"][0].permute(1,2,0),labels)
    return terms, dict(scene=names[0],actual_memory=False,simulated_memory=phase=="2a")


@lru_cache(maxsize=32)
def released_b_optics(unit):
    return b_optical_cases(unit)


def optical_specs(item):
    tier,key,index = item
    if tier == "C":
        rows = c_optical_specs(key,index)
        return rows, rows
    unit = ovg.formal_unit(key)
    f = frame_number(item)
    occ = b_k16_cases(unit, f)
    ready = (f2_entry(unit,f)/"manifest.json").is_file() if B_REFERENCE_POOL is None else (key,f) in B_REFERENCE_POOL
    refs = [row for row in released_b_optics(unit) if row["frame"]==f] if ready else []
    return occ,refs


def labeled_objective(model, item, rng, generator, phase, *, common_audit=False):
    tier, key, _ = item
    x, y = sample(item)
    g = model.geometry(x)
    first = model.first(item,g)
    ap = model.appearance(g,x,first)
    native_ap = ap
    native_split = split_mask(ap,first)
    small, ap, first_small = low_resolution(x,ap,first)
    split = split_mask(ap,first_small)
    vis = visible(ap,small,split,phase)
    front_bg = background(ap,small,split,first_small)
    base = merge_layers(vis,front_bg)
    K = small["K"][0]
    center = K.new_zeros((1,2))
    cut = cutoff(ap["depth1"][0],ap["depth2"][0])
    T = differentiable_prefix_transmittance(vis,*subcamera(K,1.4,center[0]),cut)
    size = cut.shape

    def reduced(field):
        a = tensor(field)
        if a.shape[:2] == size:
            return a
        selected = a[1::2,1::2]
        return selected

    target = appearance_labels(item)
    f = ap["foreground"][0].permute(1,2,0)
    alpha = ap["alpha"][0][...,None]
    alpha_true = reduced(y["valid_alpha"][0])
    rgb = small["rgb"][0].permute(1,2,0)
    stage1_target = reduced(x["rgb"][0].permute(1,2,0)) if tier=="B" else reduced(target["full"])
    terms = dict(
        coverage=masked_mean((1-T-ap["alpha"][0]).abs(),alpha_true),
        foreground=masked_mean((f-reduced(target["foreground"])).abs().mean(-1),reduced(target["color_valid"])),
        comp=masked_mean((alpha*f+(1-alpha)*reduced(target["background"])-stage1_target).abs().mean(-1),
                         reduced(target["comp_valid"])),
    )
    _, geometry, _ = geometry_loss(native_ap,x,y,WEIGHTS,boundary_sampling=CONFIG["boundary_sampling"],
                                  generator=generator, alpha_split_weights=CONFIG["alpha_interior_boundary_weights"])
    terms.update({k:geometry[k] for k in ("alpha","bimodal")})
    occ, refs = ([],[]) if common_audit else optical_specs(item)
    spec = refs[int(rng.integers(len(refs)))] if refs and not common_audit else None
    if occ:
        matching = [r for r in occ if spec is not None and np.isclose(r["focus"],spec["focus"])]
        selected = matching[0] if matching else occ[int(rng.integers(len(occ)))]
        native_shape = native_ap["depth1"].shape[-2:]
        offsets,cuts,reveal,block = occupancy(ray_fields(selected),y,native_shape,generator,
                                             CONFIG["optical_train"]["occ_ray_count"])
        native_vis = visible(native_ap,x,native_split,phase) if tier=="B" else vis
        terms["occ"] = occlusion_loss(native_vis,x["K"][0],selected["focus"],offsets,cuts,reveal,block,
                                      sparse=tier=="B")
    bg_spec = spec if tier=="B" else None
    if split.any():
        zero_filler = observed_background(bg_spec,small,1.4,center)
        zero = raster(base,K,1.4,center,*size,zero_filler)[0]
        terms["zero"] = masked_mean((zero-rgb).abs().mean(-1), split)
    else:
        terms["zero"] = (f*0.).sum()
    if spec is not None:
        focus, fstop = spec["focus"],spec["fstop"]
        count = CONFIG["optical_train"]["rays_per_subset"]
        offsets = torch.cat([stratified_pupil(count,.05/(2*fstop),generator,K) for _ in range(2)])
        hidden = model.second(item,ap["depth1"][0].detach().cpu().numpy(),focus,fstop)
        cloud = merge_layers(base,hidden)
        filler = observed_background(bg_spec,small,focus,offsets)
        colors = raster(cloud,K,focus,offsets,*size,filler)
        a,b = colors.chunk(2)
        ref = linear_reference(spec)
        ref = tensor(cv2.resize(ref,(size[1],size[0]),interpolation=cv2.INTER_AREA))
        terms["bokeh"] = ((a.mean(0)-ref)*(b.mean(0)-ref)).mean()
    return terms,dict(scene=key if tier=="B" else Path(key).parent.name,
                      actual_memory=True,b1=float(split.float().mean()),
                      occ_reveal=int(reveal.sum()) if occ else 0,occ_block=int(block.sum()) if occ else 0)


def weighted(terms):
    return sum(WEIGHTS[k]*value for k,value in terms.items())


def preflight(run):
    from route3_ovg_1b_optical_data import background_samples
    from route3_ovg_1c_evaluate import case_frames
    from route3_upper_bound_evaluate import display
    from route3_v2data_stage1_evaluate import region_errors

    torch.manual_seed(SEED)
    inputs = fixed_inputs()
    data = Dataset()
    model = Model(run)
    baseline = load(QUERY.parent.parent/"new-query/ema.json")
    observations = []
    with torch.no_grad():
        for split in ("development","legacy"):
            row = next(r for r in baseline[split+"_rows"]
                       if r["status"]=="evaluated" and r["frame"]==30 and r["focus_index"]==0)
            name, f, focus = row["scene"],row["frame"],row["actual_focus_m"]
            item = ("B",name,ovg.TRAIN_FRAMES.index(f))
            x,y = sample(item)
            g = model.geometry(x)
            first = model.first(item,g)
            ap = model.appearance(g,x,first)
            b = split_mask(ap,first)
            vis = visible(ap,x,b,"2b")
            fixed = fixed_visible_surfels(g,x)
            for key in ("means","scales","opacities","quats","colors"):
                torch.testing.assert_close(getattr(fixed,key),getattr(vis,key),rtol=0,atol=0)
            first_bg=background(ap,x,b,first)
            if first_bg is not None and torch.count_nonzero(first_bg.opacities):
                raise ValueError("First-pass background is active at step zero")
            K = x["K"][0]; center = K.new_zeros((1,2))
            truth = display(x["rgb"][0].permute(1,2,0).cpu().numpy())
            support = np.ones(truth.shape[:2],bool)
            if split=="development":
                spec=dict(tiers="B",unit=ovg.formal_unit(name),frame=f,f2_root=STAGE1/"development-F2")
                filler=observed_background(spec,x,focus,center)
            else:
                entry=next(r[2] for r in case_frames() if r[0]==name and r[1]==f)
                filler=background_samples(dict(tiers="B",entry=entry),K,focus,center,(512,512))
            hidden=model.second(item,g["depth1"][0].cpu().numpy(),focus,row["fstop"])
            z0=region_errors(raster(vis,K,focus,center,512,512)[0],truth,support)
            z1=region_errors(raster(merge_layers(merge_layers(vis,first_bg),hidden),K,focus,center,512,512,filler)[0],
                             truth,support)
            for key,value in (("zero_visible",z0),("zero_combined",z1)):
                np.testing.assert_allclose(value["sse"],row[key]["sse"],rtol=1e-5,atol=1e-7)
                if value["pixels"] != row[key]["pixels"]:
                    raise ValueError("Step zero changed the Stage 1 scoring support")
            observations.append(dict(split=split,scene=name,case=row["case"],b1=float(b.float().mean()),
                                     stage1={k:row[k] for k in ("zero_visible","zero_combined")},
                                     zero_visible=z0,zero_combined=z1))
    torch.cuda.empty_cache()
    timing = {}
    rng = np.random.default_rng(SEED)
    gen = torch.Generator(device="cuda").manual_seed(SEED)
    for tier,phase in (("A","2a"),("C","2a"),("B","2b")):
        steps = []
        for _ in range(2):
            model.heads.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            start = time.monotonic()
            if tier=="A":
                terms,_=a_objective(model,data.a["train"],SEED,phase)
            else:
                item = next((item for item in data.train["B"] if optical_specs(item)[1]), data.train["B"][6]) if tier=="B" else data.train["C"][4]
                terms,_=labeled_objective(model,item,rng,gen,phase)
            loss=weighted(terms)
            if not torch.isfinite(loss):
                raise ValueError(f"{tier} step-zero loss is nonfinite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.heads.parameters(),5.,error_if_nonfinite=True)
            torch.cuda.synchronize()
            steps.append(time.monotonic()-start)
            if tier=="A" and model.heads.coverage_delta.weight.grad is not None:
                raise ValueError("A appearance loss backpropagated into the coverage head")
            if phase=="2a" and model.heads.surface.weight.grad is not None and model.heads.surface.weight.grad[1:].any():
                raise ValueError("2a wrote a footprint or tilt gradient")
        timing[tier]=dict(first_seconds=steps[0],repeat_seconds=steps[1],terms=list(terms))
    write_json(run/"preflight.json",dict(passed=True,inputs=inputs,partition=data.report(),zero_identity=observations,
                                        direct_path_seconds=timing,completed_at=now()))
    print("STAGE2_PREFLIGHT_PASSED", timing, flush=True)


def schedule(phase):
    steps, mix = PHASE_PLAN[phase]
    rng = np.random.default_rng([SEED,int(phase=="2b")])
    block = [tier for tier,n in mix.items() for _ in range(n)]
    return [str(tier) for _ in range(steps//10) for tier in rng.permutation(block)]


def save_checkpoint(path, model, ema, opt, rng, gen, step, counts, uses, seconds, peak):
    temp = path.with_suffix(".part")
    torch.save(dict(step=step,model=output_state(model.heads),ema={k:v.cpu() for k,v in ema.items()},
                    optimizer=opt.state_dict(),rng=rng.bit_generator.state,pixel_rng=gen.get_state(),
                    counts=dict(counts),uses={k:float(v) for k,v in uses.items()},
                    seconds=seconds,peak_allocated_MiB=peak),temp)
    temp.replace(path)


def train(phase, target, run):
    global B_REFERENCE_POOL
    torch.manual_seed(SEED)
    reserve(run, GIB)
    data = Dataset()
    model = Model(run)
    dest = run/phase
    dest.mkdir(exist_ok=True)
    if phase=="2b":
        path=dest/"B-optical-pool.json"
        if not path.is_file():
            eligible=[(name,f) for name,unit in data.b.items() for f in (1,30,60)
                      if (f2_entry(unit,f)/"manifest.json").is_file()]
            write_json(path,dict(scene_frames=eligible,registered_at=now(),policy="fixed at 2b start"))
        B_REFERENCE_POOL=set(map(tuple,load(path)["scene_frames"]))
    opt = torch.optim.AdamW((p for p in model.heads.parameters() if p.requires_grad),lr=CONFIG["learning_rate"])
    rng = np.random.default_rng([SEED,int(phase=="2b")])
    gen = torch.Generator(device="cuda").manual_seed(SEED+int(phase=="2b"))
    ema = {k:v.detach().clone() for k,v in model.heads.state_dict().items()}
    available = sorted(p for p in dest.glob("step-*.pt") if int(p.stem[5:])<=target)
    previous = available[-1] if available else run/"2a/step-010000.pt" if phase=="2b" else None
    counts, uses = Counter(), Counter()
    start, prior_seconds, peak = 1, 0., 0.
    if previous:
        ck = torch.load(previous,map_location="cpu",weights_only=True)
        model.heads.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        for k,v in ema.items():
            v.copy_(ck["ema"][k])
        if available:
            rng.bit_generator.state = ck["rng"]
            gen.set_state(ck["pixel_rng"])
            counts.update(ck["counts"])
            uses.update({k:float(v) for k,v in ck["uses"].items()})
            start,prior_seconds,peak=ck["step"]+1,ck["seconds"],ck["peak_allocated_MiB"]
    terms_path = dest/"losses.jsonl"
    failures_path=run/"training-skips.json"
    failures=load(failures_path)
    def unit_id(tier,item):
        return item[1] if tier=="B" else Path(item[1]).parent.name
    for tier,items in data.train.items():
        data.train[tier] = [item for item in items if unit_id(tier,item) not in failures]
    schedule_rows = schedule(phase)
    progress = dest/"progress.json"
    register_training(phase,{phase:os.getpid()},{phase:str(progress)})
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    print("STAGE2_START",phase,start,target,now(),flush=True)
    with terms_path.open("a") as stream:
        for step in range(start,target+1):
            tier = schedule_rows[step-1]
            opt.zero_grad(set_to_none=True)
            if tier == "A":
                terms, support = a_objective(model,data.a["train"],int(rng.integers(2**32)),phase)
            else:
                while data.train[tier]:
                    items=data.train[tier]
                    item=items[int(rng.integers(len(items)))]
                    try:
                        terms,support=labeled_objective(model,item,rng,gen,phase)
                        break
                    except (OSError,EOFError,BadZipFile) as error:
                        key=unit_id(tier,item)
                        failures[key]=dict(tier=tier,reason=str(error),stage=phase,step=step)
                        write_json(failures_path,failures)
                        data.train[tier]=[row for row in items if unit_id(tier,row)!=key]
                else:
                    raise ValueError(f"No {tier} units remain with readable labels")
            loss = weighted(terms)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{phase} {step} {tier} nonfinite training loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.heads.parameters(),5,error_if_nonfinite=True)
            opt.step()
            with torch.no_grad():
                for k,p in model.heads.state_dict().items():
                    ema[k].lerp_(p,1-CONFIG["ema_decay"])
            counts[tier]+=1
            uses.update({k:v.detach() for k,v in terms.items()})
            if step%20==0 or step==1:
                out=dict(step=step,phase=phase,tier=tier,counts=dict(counts),loss=float(loss.detach()),
                         grad_norm=float(norm),terms={k:float(v.detach()) for k,v in terms.items()},
                         seconds=prior_seconds+time.monotonic()-started,
                         peak_allocated_MiB=max(peak,torch.cuda.max_memory_allocated()/2**20),**support)
                write_json(progress,out)
                stream.write(json.dumps(out)+"\n")
                stream.flush()
                if step%100==0 or step==1:
                    print("STAGE2",phase,step,out["loss"],out["seconds"],flush=True)
            if step%CHECKPOINT_EVERY==0 or step==target:
                reserve(run,GIB)
                elapsed=prior_seconds+time.monotonic()-started
                peak=max(peak,torch.cuda.max_memory_allocated()/2**20)
                save_checkpoint(dest/f"step-{step:06d}.pt",model,ema,opt,rng,gen,step,counts,uses,elapsed,peak)
    register_training(phase,{}, {})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=Path,default=RUN)
    p.add_argument("--phase",choices=tuple(PHASE_PLAN))
    p.add_argument("--target",type=int)
    p.add_argument("--preflight",action="store_true")
    args = p.parse_args()
    if not args.preflight and (args.phase is None or args.target is None or not 0 < args.target <= PHASE_PLAN[args.phase][0]
                               or args.target%CHECKPOINT_EVERY):
        p.error("Target must be on the stage's checkpoint schedule")
    args.run.mkdir(parents=True,exist_ok=True)
    configuration()
    fixed_inputs()
    if args.preflight:
        preflight(args.run.resolve())
    else:
        train(args.phase,args.target,args.run.resolve())
