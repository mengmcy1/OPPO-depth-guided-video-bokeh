"""Stage 3a training, with frozen Query and detached B hidden-surface placement."""
import argparse
from collections import Counter
from functools import lru_cache
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
from numpy.core.multiarray import scalar as numpy_scalar
import torch

from route3_ovg_1b_common import CONFIG as STAGE1
from route3_ovg_1b_train import load_model, state
from route3_ovg_subpixel import TRAIN_FRAMES, save_npz
from route3_v2data_B_inputs import supervision
from route3_v2data_B_batch import FORMAL_DATA, GIB, ROOT, sha256, write_json
from route3_v2data_B_plan import load
from route3_v2data_B_training_share import register_training
from route3_v2data_dryrun import stage_objective
from route3_v2data_loader import TrainingData
from route3_v2data_optics import hidden_parents, hidden_substitute, ray_fields
from route3_v2data_stage1 import complete_optical, verify_disk
from route3_v2data_stage1_evaluate import load_paired_checkpoint
from route3_v2data_stage2_train import INITIAL, QUERY, now

RECORD = ROOT/"03_实验记录/records/route3-v2data-stage3-config-2026-11-24.json"
V1_OUT = ROOT/"outputs/route3-v2data-stage3a-2026-11-24-01"
OUT = ROOT/"outputs/route3-v2data-stage3a-2026-11-24-02"
PARAMETERS = {"front_inverse_depth_residual":0, "multiview_residual_gate":5}
DEPTH_MODULES = (
    "rgb_half", "rgb_quarter", "depth_quarter", "semantic", "dpt", "quarter",
    "shared_view", "match", "refine", "half_decoder", "full", "prior_head", "prior_pixel", "geometry",
)
CROP_SIZE = 256
CHECKPOINTS = (5000,10000,15000,20000)
MIX = {"B":50,"C":30,"R":10,"G":10}
SEED = STAGE1["seed"]
FIXED = [
    ("dev001",30,0), ("dev006",30,0), ("dev003",30,0), ("dev004",30,0),
    ("dev005",30,1), ("dev008",30,0), ("dev008",30,1), ("dev000",30,1), ("122104",30,1),
]


def configuration_record(inventory, pool):
    initial = sha256(INITIAL)
    if not initial.startswith("dfb1d7e5"):
        raise ValueError("Unexpected Stage 1 geometry checkpoint")
    return dict(
        created_at=now(), scheme="13.17.3 and 13.17.4.1", stage="3a",
        run=str(V1_OUT), geometry=dict(checkpoint=str(INITIAL),sha256=initial,weights="ema",
                  split="b ≡ 0",renderer="V0, fixed camera-facing 5×5 per-pixel surfels, opacity .999"),
        query=dict(checkpoint=str(QUERY),sha256=sha256(QUERY),frozen=True,loaded_in_3a=False),
        hidden=dict(tiers=["B optical"],implementation="hidden_substitute → place",
                    policy="Use GT recoverable surface points and input witness RGB. Cache only fixed parent samples. Recompute local disparity fit and footprint order every step under no_grad with detached current front depth, without passing labels to the network.",
                    other_tiers="Visible only; C can retain its existing analytic LDR optical batches and K16 supervision.",
                    evaluation_legacy="Archived GT recoverable parents and the same detached place()."),
        parameter_groups={
            group:dict(tensors=["geometry.weight","geometry.bias"],active_output_row=row,
                       elements=65)
            for group,row in PARAMETERS.items()
        },
        frozen="All other rows of the geometry head (rear offset and normal), encoder, all feature blocks, learned prior selector, coverage and footprint modules. The input-derived total residual attenuation remains fixed.",
        loss_weights=dict(bokeh=1.,disp=.00025,bimodal=.0000625,alpha=.000125,mv=.0000125,
                          occ=.00025,same_surface=.005,zero=.025),
        weights_reason="Raise optical emphasis relative to Stage 1. The supervised geometry and same-surface weights share a 1/4000 multiplier; the old C dry run measured mean raw disp and bokeh gradient norms 13.20 and 0.0060. Keep larger relative occ/zero regularizers. Unchanged coverage is still logged.",
        optical=dict(reference="13.15.5 per-ray clipped linear RGB LDR", training_resolution_limit=STAGE1["optical_train"]["max_image_side"],
                     estimator="Independent stratified pupil-subset cross product, 8+8 rays. Clip each predicted ray before averaging.",
                     occ_B_resolution=512,occ_rays=STAGE1["optical_train"]["occ_ray_count"],eval_rays=64),
        data=dict(batch_percent=MIX,batch_scenes=1,seed=SEED,
                  B_optical_start=len(pool["scenes"]),B_hard_at_start=len(pool["hard"]),
                  hard_selection="Weight 2 for completed hard milestone scenes, weight 1 for other complete scenes. Suppress the foreground-focus priority for imported foregrounds below 1% as in 13.15.8.",
                  refresh_steps=500, B_usage_keys=["scene_id","frame","focus_index","fstop"],
                  final_stage_full_window_target_draws_per_scene=20, test_cases_used=0,
                  inventory_slots={k:len(inventory.items[k]) for k in MIX}),
        training=dict(steps=20000,learning_rate=1.e-5,optimizer="AdamW with weight decay 0 for row freezing",
                      gradient_clip=5.,ema_decay=.999,checkpoint_steps=list(CHECKPOINTS),
                      eval_steps=[0,*CHECKPOINTS],eval_aperture=1.4,
                      evaluate_ema_every_node=True,raw_at_terminal=True,
                      max_process_gpu_gib=14,
                      rationale="One-tenth of Stage 1 LR for a two-row geometry fine-tune; evaluate P2 f/1.4 on the benchmark's main masks."),
        evaluation=dict(fixed_cases=[list(x) for x in FIXED],extra_scene_selection="Top and bottom two paired near-focus scene deltas; f030 near with evaluated fallback frame",
                        relative_geometry_limits=dict(inverse_depth_rmse=.05,same_surface_step=.10,near_subject_coc_mae=.10,
                                                      GT_occlusion_crossing_abs_p95=.10),
                        zero_floors_db=dict(zero_visible=50.,zero_combined=60.),
                        selection="Among EMA checkpoints satisfying all geometry and zero floors, maximize the new development near-focus pooled PSNR (earlier step wins ties). Adopt only if that checkpoint also has nonnegative paired near and far medians against step 0 with the same GT substitute and V0.",
                        pair_tie_threshold_db=0.),
    )


def configuration_for(run):
    config = load(RECORD)
    replacement = config.get("stage3a_v2")
    if replacement and run.resolve() == Path(replacement["run"]).resolve():
        return {**config, **replacement}
    if run.resolve() != Path(config["run"]).resolve():
        raise ValueError(f"No registered Stage 3a configuration for {run}")
    return config


def register_v2(stopped):
    config = load(RECORD)
    if "stage3a_v2" in config:
        raise FileExistsError("Stage 3a v2 is already registered")
    inventory = TrainingData()
    pool = optical_pool(inventory)
    net = load_paired_checkpoint(INITIAL,"ema",device="cpu")
    modules = {name:sum(p.numel() for p in net.get_submodule(name).parameters()) for name in DEPTH_MODULES}
    original = V1_OUT/"evaluation/step-000000/ema/report.json"
    if not original.is_file():
        raise FileNotFoundError(original)
    optical = dict(config["optical"],B_source_resolution=[512,512],B_crop_size=[CROP_SIZE,CROP_SIZE],
                    B_crops_per_batch=1,B_min_thin_crop_fraction=.5,
                    B_crop_sampling="One native crop. Alternate B draws: first draw and every second thereafter require at least one GT thin foreground pixel. Choose a random eligible configuration and uniform legal crop origin. Ordinary draws keep the full pool and the existing difficulty weights.",
                    B_thin_mask="Native first-hit object IDs for foreground_structure from the B generation label or thin-structure difficulty entry. Other foreground objects are excluded.",
                    B_screen_coordinates="Render all native V0 and detached GT geometry with only the principal point shifted; crop the native target and retain full-sensor F2 lookup.",
                    C="Retain the original resolution limit and optical sampling.")
    config["stage3a_v1_stop"] = dict(load(stopped),configuration_sha256=sha256(RECORD))
    config["stage3a_v2"] = dict(version=2,created_at=now(),run=str(OUT),
        parameter_groups={name:dict(elements=n) for name,n in modules.items()},
        trainable_parameters=sum(modules.values()),
        frozen="ViT encoder, coverage and footprint heads. Prior selector participates because its weights directly change fused front depth. All geometry rows are trainable. Rear depth keeps its existing geometry supervision; normal rows have no direct V0 or auxiliary loss in this configuration. Parameter deltas are recorded for each row.",
        optical=optical,
        data=dict(config["data"],B_optical_start=len(pool["scenes"]),B_hard_at_start=len(pool["hard"]),
                  B_thin_at_start=len(pool["thin"])),
        training=dict(config["training"],optimizer="AdamW, weight decay 0",eval_steps=list(CHECKPOINTS),
                      rationale="Keep the registered 3a learning rate and step count. Expand the existing depth decoder and use one native B crop in the per-process 14 GiB limit."),
        step_zero=dict(report=str(original),sha256=sha256(original),commit="0a3ea35"),
        reuse_hidden_parents=str(V1_OUT/"hidden-parents"))
    write_json(RECORD,config)


@lru_cache(maxsize=256)
def thin_surface_ids(unit):
    presence = load(unit/"labels/foreground-presence.json")
    return tuple(i for name,ids in presence.get("foreground_surface_ids",{}).items()
                 if name.startswith("foreground_structure") for i in ids)


@lru_cache(maxsize=192)
def thin_frames(unit):
    ids = thin_surface_ids(unit)
    if not ids:
        return ()
    presence = load(unit/"labels/foreground-presence.json")
    return tuple(row["frame"] for row in presence["frames"] if row["frame"] in (1,30,60)
                 and any(name.startswith("foreground_structure") and item["center_hits"]>0
                         for name,item in row["foreground"].items()))


def optical_pool(inventory):
    hard = load(FORMAL_DATA/"optical/hard-milestone.json")
    if not hard:
        raise FileNotFoundError("Missing completed optical hard-scene manifest")
    primary = {r["scene_id"]:r["primary_difficulty"] for r in load(FORMAL_DATA/"optical/selection.json")}
    optics = {r["scene_id"]:r for r in load(FORMAL_DATA/"manifest.json")["optical_units"]}
    primary.update({name:r["primary_difficulty"] for name,r in optics.items()})
    available = complete_optical(inventory)
    hard_ids = set(hard["scene_ids"])
    priority = []
    for scene in available:
        if scene not in hard_ids:
            continue
        if primary.get(scene)=="foreground_focus":
            presence = load(inventory.b[scene]/"labels/foreground-presence.json")
            area = max((item["surface_pixels"] for r in presence.get("frames",[]) if r["frame"] in (1,30,60)
                        for name,item in r["foreground"].items() if name.startswith("foreground_import_")),default=0)
            if area < .01*512**2:
                continue
        priority.append(scene)
    if not available:
        raise FileNotFoundError("No completed and geometry-matched training optical scene")
    thin = {}
    for name in available:
        unit = inventory.b[name]
        category = load(unit/"manifest.json")["category"]
        difficulty = optics.get(name,{}).get("difficulties",())
        if category == "thin_structure" or "thin_and_complex_edge_transmission" in difficulty:
            frames = thin_frames(unit)
            if frames:
                thin[name] = frames
    return dict(scenes=available,hard=priority,thin=thin)


def reference(run, inventory, step, interval):
    start = ((step-1)//interval)*interval
    directory = run/"reference-pools"
    directory.mkdir(exist_ok=True)
    path = directory/f"step-{start:06d}.json"
    if not path.is_file():
        value = optical_pool(inventory)
        write_json(path,dict(**value,step=start,recorded_at=now()))
    return load(path)


@lru_cache(maxsize=64)
def optical_specs(inventory, scene):
    return [s for f in (1,30,60) for s in inventory.optical_specs(("B",scene,TRAIN_FRAMES.index(f)))]


class Placement:
    """Reuse immutable ray parents, never a fit or projected Gaussian."""
    def __init__(self, directory, fallback=None):
        self.directory = directory
        self.fallback = Path(fallback) if fallback else None

    def __call__(self, spec, ref, depth, K):
        name = spec["unit"].name
        case = spec["truth"].stem
        key = Path(name)/f"{case}.npz"
        path = self.directory/key
        if not path.is_file() and self.fallback is not None and (self.fallback/key).is_file():
            path = self.fallback/key
        if path.is_file():
            parents = load_parents(path)
        else:
            parents = hidden_parents(spec,ray_fields(spec),ref)
            if parents is None:
                parents = dict(uv=np.empty((0,2),np.float32),depth=np.empty(0,np.float32),
                               recoverable=np.empty(0,bool),colors_linear=np.empty((0,3),np.float32))
            path.parent.mkdir(parents=True,exist_ok=True)
            save_npz(path,**parents)
        return hidden_substitute(spec,None,ref,depth.detach(),K,parents=parents)


@lru_cache(maxsize=48)
def load_parents(path):
    with np.load(path) as d:
        return {k:d[k] for k in d.files}


def gpu_setup(config):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise ValueError("Assign the shared RTX 5090 explicitly with CUDA_VISIBLE_DEVICES=0")
    subprocess.run(["nvidia-smi","--query-gpu=memory.used,memory.total,utilization.gpu","--format=csv"],check=True)
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(SEED)
    torch.cuda.set_per_process_memory_fraction(config["training"]["max_process_gpu_gib"]*GIB/
                                               torch.cuda.get_device_properties(0).total_memory,0)


def trainable_geometry(net, version):
    net.eval().requires_grad_(False)
    if version==2:
        groups = {name:tuple(net.get_submodule(name).requires_grad_(True).parameters()) for name in DEPTH_MODULES}
        return tuple(p for values in groups.values() for p in values), groups
    tensors = (net.geometry.weight,net.geometry.bias)
    for tensor in tensors:
        active = torch.zeros_like(tensor)
        for row in PARAMETERS.values():
            active[row] = 1
        tensor.requires_grad_(True)
        tensor.register_hook(lambda grad,mask=active: grad*mask)
    return tensors, None


def group_norms(gradients, groups):
    def norm(values):
        measured = [tensor.float().norm() for tensor in values if tensor is not None]
        return float(torch.linalg.vector_norm(torch.stack(measured))) if measured else 0.
    if groups is None:
        return {name:norm([v[row] for v in gradients]) for name,row in PARAMETERS.items()}
    by_parameter = {p:g for p,g in zip((p for ps in groups.values() for p in ps),gradients)}
    return {name:norm([by_parameter[p] for p in ps]) for name,ps in groups.items()}


def choose_crop(spec, rng, require_thin):
    if spec is None or spec["tiers"]!="B":
        return None, 0
    unit = spec["unit"]
    ids = thin_surface_ids(unit)
    mask = None
    if ids:
        label = supervision(unit,spec["frame"],("object_id","center_visible"))
        mask = np.isin(label["object_id"],ids) & label["center_visible"]
        if mask.shape != (512,512):
            raise ValueError(f"Invalid native thin mask for {unit.name}")
    if require_thin:
        if mask is None or not mask.any():
            raise ValueError(f"A required native crop has no labeled thin foreground in {unit.name}")
        integral = cv2.integral(mask.astype(np.uint8),sdepth=cv2.CV_32S)
        n=CROP_SIZE
        window = integral[n:,n:]-integral[n:,:-n]-integral[:-n,n:]+integral[:-n,:-n]
        origins = np.argwhere(window>0)
        top,left = map(int,origins[int(rng.integers(len(origins)))])
    else:
        top,left = map(int,rng.integers(513-CROP_SIZE,size=2))
    support = int(mask[top:top+CROP_SIZE,left:left+CROP_SIZE].sum()) if mask is not None else 0
    return (top,left,CROP_SIZE,CROP_SIZE),support


def smoke(net, inventory, tensors, groups, pool, placer, run):
    # Pick one eligible near-focus B batch with live view matching before
    # computing any optical loss; avoid an inactive residual gate by design.
    spec = None
    ordered = pool["hard"]+sorted(set(pool["scenes"])-set(pool["hard"]))
    for scene in ordered:
        if groups is not None and scene not in pool["thin"]:
            continue
        row=next(s for s in optical_specs(inventory,scene)
                 if s["fstop"]==1.4 and (s["frame"] in pool["thin"][scene] if groups is not None else s["frame"]==30))
        item = ("B",scene,TRAIN_FRAMES.index(row["frame"]))
        with torch.no_grad():
            x,_ = inventory.load(item,"cuda")
            pred = net(x,protect_residual=True,render_heads=False)
            active = int(pred["mv_mask"].sum())
        if active:
            spec = row
            break
    if spec is None:
        raise ValueError("No eligible optical sample exercises view matching")
    rng = torch.Generator(device="cuda").manual_seed(SEED)
    crop,thin = choose_crop(spec,np.random.default_rng([SEED,2]),True) if groups is not None else (None,0)
    torch.cuda.reset_peak_memory_stats()
    begin = time.monotonic()
    losses, supports, _ = stage_objective(net,inventory,item,"3a",rng,spec=spec,stage3_place=placer,stage3_crop=crop)
    grads = torch.autograd.grad(losses["bokeh"],tensors,allow_unused=True)
    group = group_norms(grads,groups)
    result = dict(at=now(),scene=item[1],case=spec["truth"].stem,gate_pixels=active,
                  L_bokeh=float(losses["bokeh"].detach()),gradient_norms=group,
                  hidden_has_grad=supports["hidden_has_grad"],hidden_placed=supports["hidden_placed"],
                  auxiliary={k:float(v.detach()) for k,v in losses.items() if k!="bokeh"},
                  native_crop=crop,thin_pixels=thin,thin_fraction=thin/(CROP_SIZE**2) if crop else None,
                  seconds=time.monotonic()-begin,peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
    result["passed"]=all(np.isfinite(v) and v>0 for v in group.values()) and not result["hidden_has_grad"]
    write_json(run/"gradient-check.json",result)
    print("STAGE3A_GRADIENT",result,flush=True)
    if not result["passed"]:
        raise RuntimeError("The optical composition graph does not reach every selected parameter group")


def choose_item(inventory, tier, pool, rng, require_thin=False):
    if tier=="B":
        options = list(pool["thin"]) if require_thin else pool["scenes"]
        if not options:
            raise ValueError("No admitted B optical case has GT thin foreground")
        weights = np.where(np.isin(options,pool["hard"]),2.,1.)
        scene = str(rng.choice(options,p=weights/weights.sum()))
        all_specs = optical_specs(inventory,scene)
        if require_thin:
            all_specs = [s for s in all_specs if s["frame"] in pool["thin"][scene]]
        spec = all_specs[int(rng.integers(len(all_specs)))]
        return ("B",scene,TRAIN_FRAMES.index(spec["frame"])),spec
    item = inventory.choose(tier,rng)
    if item is None:
        raise ValueError(f"No admissible {tier} batch")
    specs = inventory.optical_specs(item) if tier=="C" else []
    return item,specs[int(rng.integers(len(specs)))] if specs else None


def checkpoint(run, step):
    return run/"checkpoints"/f"step-{step:06d}.pt"


def checkpoint_types():
    # The live run was launched with NumPy strings in its draw counters.
    # Permit just these scalar types for its own Stage 3a checkpoints.
    return torch.serialization.safe_globals([numpy_scalar,np.dtype,np.dtypes.StrDType,np.str_])


def read_checkpoint(path):
    with checkpoint_types():
        return torch.load(path,weights_only=True,map_location="cpu")


def checkpoint_model(path,key):
    with checkpoint_types():
        return load_model(path,key=key)


def crop_statistics(counts):
    crops = counts.get("B_crops",0)
    if not crops:
        return {}
    return dict(crops=crops,crops_with_thin=counts["B_thin_crops"],
                quota_draws=counts["B_required_thin_crops"],
                thin_crop_fraction=counts["B_thin_crops"]/crops,
                mean_thin_pixel_fraction=counts["B_crop_thin_pixels"]/(crops*CROP_SIZE**2))


def parameter_changes(path, key, config):
    if config.get("version") != 2:
        return {}
    initial = read_checkpoint(INITIAL)
    baseline = dict(initial["model"],**initial["ema"])
    current = read_checkpoint(path)[key]
    def compare(tensors):
        before = sum(float(a.double().square().sum()) for a,_ in tensors)**.5
        delta = sum(float((b.double()-a.double()).square().sum()) for a,b in tensors)**.5
        return dict(initial_l2=before,delta_l2=delta,relative_l2=delta/before if before else None)
    modules = {name:compare([(baseline[k],v) for k,v in current.items() if k.startswith(name+".")])
               for name in config["parameter_groups"]}
    rows = {str(row):compare([(baseline[k][row],current[k][row])
                              for k in ("geometry.weight","geometry.bias")]) for row in range(6)}
    return dict(modules=modules,geometry_rows=rows)


def save_checkpoint(path, net, ema, opt, counts, sampled, seconds, config_hash):
    if path.is_file():
        raise FileExistsError(path)
    path.parent.mkdir(exist_ok=True)
    temp = path.with_name(path.name+".part")
    model = state(net)
    shadow = dict(model,**{k:v.detach().cpu() for k,v in ema.items()})
    torch.save(dict(phase="3a",step=int(path.stem[5:]),model=model,ema=shadow,optimizer=opt.state_dict(),
                    counts={str(k):v for k,v in counts.items()},optical_sampling=dict(sampled),
                    seconds=seconds,config_sha256=config_hash),temp)
    temp.replace(path)


def train(run):
    run.mkdir(parents=True,exist_ok=True)
    if not RECORD.is_file():
        raise FileNotFoundError("Write the stage 3 configuration first")
    cfg = configuration_for(run)
    version = cfg.get("version",1)
    digest = cfg.get("stage3a_v1_stop",{}).get("configuration_sha256") if version==1 and "stage3a_v2" in cfg else sha256(RECORD)
    if sha256(INITIAL)!=cfg["geometry"]["sha256"]:
        raise ValueError("Stage 3a initialization hash changed")
    verify_disk(run)
    with (run/"train.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        gpu_setup(cfg)
        inventory = TrainingData()
        placer = Placement(run/"hidden-parents",cfg.get("reuse_hidden_parents"))
        existing = sorted((run/"checkpoints").glob("step-*.pt"))
        saved = read_checkpoint(existing[-1]) if existing else None
        if saved and saved["config_sha256"]!=digest:
            raise ValueError("Checkpoint configuration changed")
        net = checkpoint_model(existing[-1],"model") if saved else load_paired_checkpoint(INITIAL,"ema")
        tensors, groups = trainable_geometry(net,version)
        opt = torch.optim.AdamW(tensors,lr=cfg["training"]["learning_rate"],weight_decay=0.)
        ema = {name:p.detach().clone() for name,p in net.named_parameters() if p.requires_grad}
        start, prior, counts, sampled = 1,0.,Counter(),Counter()
        if saved:
            start,prior = saved["step"]+1,saved["seconds"]
            opt.load_state_dict(saved["optimizer"])
            for name,value in ema.items():
                value.copy_(saved["ema"][name])
            counts.update(saved["counts"])
            sampled.update(saved["optical_sampling"])
        interval = cfg["data"]["refresh_steps"]
        pool = reference(run,inventory,start,interval)
        if not load(run/"gradient-check.json").get("passed"):
            smoke(net,inventory,tensors,groups,pool,placer,run)
        begin = time.monotonic()
        progress = run/"progress.json"
        write_json(progress,dict(step=start-1,phase="3a",counts=counts,seconds=prior,at=now()))
        register_training("3a",{"3a":os.getpid()},{"3a":str(progress)})
        segment = [k for k,v in MIX.items() for _ in range(v//10)]
        schedule = np.random.default_rng([SEED,3]).permuted(
            np.tile(segment,(cfg["training"]["steps"]//10,1)),axis=1).ravel()
        try:
            for step in range(start,cfg["training"]["steps"]+1):
                if step%interval==1:
                    verify_disk(run)
                    pool=reference(run,inventory,step,interval)
                tier = str(schedule[step-1])
                rng = np.random.default_rng([SEED,3,step])
                require_thin = version==2 and tier=="B" and counts["B"]%2==0
                item,spec=choose_item(inventory,tier,pool,rng,require_thin)
                crop,thin = choose_crop(spec,rng,require_thin) if version==2 else (None,0)
                pixel_rng=torch.Generator(device="cuda").manual_seed(int(rng.integers(2**32)))
                opt.zero_grad(set_to_none=True)
                terms,support,_=stage_objective(net,inventory,item,"3a",pixel_rng,spec=spec,
                                                stage3_place=placer,stage3_crop=crop)
                total=sum(cfg["loss_weights"][k]*v for k,v in terms.items())
                if not torch.isfinite(total):
                    raise FloatingPointError(f"Nonfinite 3a loss at step {step}, {tier}")
                total.backward()
                norm=torch.nn.utils.clip_grad_norm_(tensors,cfg["training"]["gradient_clip"],error_if_nonfinite=True)
                opt.step()
                with torch.no_grad():
                    for name,p in net.named_parameters():
                        if name in ema:
                            ema[name].lerp_(p,1-cfg["training"]["ema_decay"])
                counts[tier]+=1
                case=spec["truth"].stem if spec else None
                if spec and "bokeh" in terms:
                    sampled[f"{tier}/{item[1]}/{case}"]+=1
                    counts[f"bokeh_{tier}"]+=1
                row=dict(step=step,tier=tier,item=item,case=case,
                         total=float(total.detach()),grad_norm=float(norm),
                         losses={k:float(v.detach()) for k,v in terms.items()},support=support)
                if crop:
                    counts.update(B_crops=1,B_thin_crops=int(thin>0), B_crop_thin_pixels=thin,
                                  B_required_thin_crops=int(require_thin))
                    row["crop"]=dict(sensor_yxhw=crop,thin_pixels=thin,thin_fraction=thin/(CROP_SIZE**2),
                                     required=require_thin)
                with (run/"steps.jsonl").open("a") as stream:
                    stream.write(json.dumps(row)+"\n")
                if step==start or step%25==0 or step in CHECKPOINTS:
                    elapsed=prior+time.monotonic()-begin
                    write_json(progress,dict(step=step,phase="3a",counts=counts,seconds=elapsed,at=now(),
                                             B_crop_statistics=crop_statistics(counts),
                                             peak_reserved_bytes=torch.cuda.max_memory_reserved()))
                    print("STAGE3A",step,float(total.detach()),elapsed,flush=True)
                if step%500==0:
                    elapsed=prior+time.monotonic()-begin
                    save_checkpoint(checkpoint(run,step),net,ema,opt,counts,sampled,elapsed,digest)
                    write_json(run/"optical-sampling.json",dict(step=step,counts=sampled,
                                                               B_crop_statistics=crop_statistics(counts)))
        finally:
            register_training("3a",{},{})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=Path,default=OUT)
    p.add_argument("--write-config",action="store_true")
    p.add_argument("--register-v2",type=Path,metavar="STOP_RECEIPT")
    a=p.parse_args()
    if a.register_v2:
        register_v2(a.register_v2)
    elif a.write_config:
        if RECORD.exists():
            raise FileExistsError(RECORD)
        data=TrainingData()
        write_json(RECORD,configuration_record(data,optical_pool(data)))
    else:
        train(a.run.resolve())


if __name__=="__main__":
    main()
