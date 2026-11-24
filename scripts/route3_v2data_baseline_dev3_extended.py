"""Evaluate the frozen Stage 3 start and external baselines on the 15 B development scenes."""
import argparse
from dataclasses import fields
import json
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import time

import cv2
import numpy as np
import torch

from route3_v2data_baseline_dev3 import FRAMES, ROOT, display, guard, read, write
from route3_v2data_B_inputs import library_unit_paths, observation, supervision
from route3_v2data_stage1_evaluate import (
    development_cases, load_paired_checkpoint, near_subject, p2_focus, view_from_observation,
)
import route3_v2data_stage1_visualize as gallery
from route3_v2data_stage2_train import INITIAL, QUERY, STAGE1, fixed_inputs, sample
from route3_v2data_stage2_eval import bands, optical_target, oracle_hidden, peeled_truth
from route3_v2data_optics import f2_entry, read_f2, ray_fields
from route3_ovg_1b_common import CONFIG
from route3_ovg_1b_optical_data import render_f2
from route3_ovg_footprint import query_net
from route3_ovg_subpixel import TRAIN_FRAMES
from route3_b2_oracle_3d import sparse_pixel_layer
from route3_sharp_s1 import raster as sharp_raster
from route3_ovg_data import b_edge_reference
from src.aperture_renderer import GaussianScene, aperture_disk
from src.dual_layer import depth_band
from src.hidden_retrieval_rules import aggregate, collect
from src.layered_aperture import merge_layers
from src.ovg_optics import fixed_visible_surfels, raster
from route3_v2data_baseline_dev3_extended_report import finish, save_case, METHODS


STOP_VALUES = (1.4, 2.)
F2_ROOT = STAGE1/"development-F2"
PROFILE = ROOT/"03_实验记录/records/route3-dual-layer-criteria-L2-2026-11-11.json"
FIXED_REVIEW = (
    ("000", 30, 0), ("001", 30, 0), ("003", 30, 0), ("004", 30, 1),
    ("005", 30, 1), ("006", 30, 0), ("008", 30, 1), ("011", 30, 0),
)
GIB = 2**30


class DrBokehWorker:
    """Keep the B2 adapter and its original dependency stack in a child."""
    def __init__(self, output):
        self.output = output
        python = ROOT.parent/"envs/conda-magicbokeh/bin/python"
        script = Path(__file__).with_name("route3_v2data_baseline_dev3_extended_drbokeh.py")
        with (output/"drbokeh.log").open("ab") as log:
            self.process = subprocess.Popen([python,"-u",script],stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,stderr=log,text=True,cwd=ROOT)
        try:
            self.response()
        except Exception:
            self.close()
            raise

    def response(self):
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("Dr.Bokeh worker exited; see drbokeh.log")
        result = json.loads(line)
        if "error" in result:
            raise RuntimeError(result["error"])
        return result

    def render(self, rgb, depth, alpha, focus, fx, radius):
        with TemporaryDirectory(prefix=".drbokeh-",dir=self.output) as folder:
            source = Path(folder)/"source.npz"
            dest = Path(folder)/"result.npz"
            np.savez(source,rgb=rgb,depth=depth,alpha=alpha)
            request = dict(input=str(source),output=str(dest),focus=focus,fx=float(fx),radius=radius)
            self.process.stdin.write(json.dumps(request)+"\n")
            self.process.stdin.flush()
            response = self.response()
            with np.load(dest) as data:
                return data["rgb"],data["zero"],response["audit"]

    def close(self):
        self.process.stdin.close()
        self.process.wait()
        self.process.stdout.close()


def readiness():
    result = subprocess.check_output([
        "nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits",
    ], text=True)
    return int(result.splitlines()[0]) >= 10*1024


def display_tensor(tensor):
    return display(tensor.cpu().numpy())


def foreground_coverage(unit, frame):
    alpha = supervision(unit,frame,("foreground_alpha",))["foreground_alpha"]
    if not np.isfinite(alpha).all() or alpha.min() < 0 or alpha.max() > 1:
        raise ValueError("B2 oracle input requires finite foreground coverage in [0,1]")
    return alpha


def copy_scene(scene, include):
    return GaussianScene(**{key.name: getattr(scene, key.name)[include] for key in fields(scene)})


def build_frame(unit, frame, cases, network, views):
    item = ("B", unit.name, TRAIN_FRAMES.index(frame))
    x, y = sample(item)
    reference = b_edge_reference(item, y)
    K = x["K"][0]
    intr = K.cpu().numpy()
    raw = observation(unit, frame, aligned=True)
    rgb = x["rgb"][0].permute(1,2,0)
    image = display_tensor(rgb)
    gt = reference["gt"].astype(np.float32)
    known = reference["valid"] & np.isfinite(gt) & (gt > 0)
    subject = [near_subject(gt, known, intr, c["focus_m"]) for c in cases]
    rig = gallery.prepare_geometry(network, x, views, None, frame, cases, subject)
    oracle_z = np.where(known, gt, 1e4).astype(np.float32)
    oracle_visible = fixed_visible_surfels(dict(depth1=K.new_tensor(oracle_z)[None]), x)
    gt_hidden = peeled_truth(unit, unit.name, frame)
    profile = read(PROFILE)["profile"]

    lookup, arrays = read_f2(f2_entry(unit, frame, F2_ROOT), unit)
    a_color = GaussianScene(**{k: t.to(K.device) for k,t in arrays.items()})
    a_visible = copy_scene(a_color, a_color.source_frames == 0)
    # Match S2a: initialize the band and focus from aligned SHARP, with
    # A-color expected depth used for the stricter R1 placement rejection.
    sharp_z = raw["depth"][0]
    sharp_valid = (raw["confidence"][0] > 0) & np.isfinite(sharp_z) & (sharp_z > 0)
    sharp_z = np.where(sharp_valid, sharp_z, 1e4).astype(np.float32)
    sharp_focus_z = np.where(sharp_valid, sharp_z, 0.)
    _, expected, opacity = sharp_raster(a_color, K)
    expected, opacity = expected.cpu().numpy(), opacity.cpu().numpy()
    target = next(v for v in views if v["frame"] == frame)
    own = np.where((opacity >= .1) & (expected > 0), expected, sharp_z)
    sharp_focuses = [p2_focus(sharp_focus_z, selector) for selector in subject]
    a_bands = [None if f is None else depth_band(sharp_z, intr[0,0], f, .05/(2*c["fstop"]))[0]
               for f,c in zip(sharp_focuses,cases)]
    active = [band for band in a_bands if band is not None]
    r2, retrieval = None, None
    if active:
        try:
            pool = collect(sharp_z,intr,target["c2w"],np.logical_or.reduce(active),views,frame)
            r2, retrieval = aggregate(pool,2,own)
        except (ValueError, OSError) as error:
            retrieval = {"error": f"{type(error).__name__}: {error}"}
    main_gt = y["depth1"][0].cpu().numpy().astype(np.float32)
    foreground = np.isin(reference["object_id"], reference["foreground"])
    fg_mask = known & foreground & np.isfinite(main_gt) & (main_gt > 0)
    return dict(cases=cases, K=K, intr=intr, ref=reference, gt=gt, main_gt=main_gt, known=known,
                rgb=image, subject=subject, rig=rig, oracle_z=oracle_z, oracle_visible=oracle_visible,
                gt_hidden=gt_hidden, profile=profile, a_color=a_color, a_visible=a_visible,
                sharp=sharp_focus_z, sharp_focuses=sharp_focuses, a_bands=a_bands, r2=r2,
                retrieval=retrieval, lookup=lookup, foreground=fg_mask, alpha=foreground_coverage(unit,frame),
                edge=bands(reference,y)[0])


def recoverable_mask(unit, case, state):
    if case["focus_index"] == 0:
        return None
    key = unit/"labels"/f'f{case["frame"]:03d}-focus{case["focus_index"]}-K16.npz'
    truth = ray_fields(dict(tiers="B", unit=unit, frame=case["frame"], focus=case["focus_m"],
                            fstop=case["fstop"], label=key if case["fstop"]==1.4 else None))
    band = depth_band(state["oracle_z"],state["intr"][0,0],case["focus_m"],.05/(2*case["fstop"]))[0]
    return band & np.any(truth["recoverable"],axis=0)


def score_masks(unit, case, state):
    region = {"full_image": np.ones(state["gt"].shape,bool), "boundary_2px": state["edge"]}
    if case["focus_index"] == 0:
        rho = 1/state["oracle_z"]
        coc = state["intr"][0,0]*.05/(2*case["fstop"])*np.abs(rho - 1/case["focus_m"])
        region["near_in_focus"] = state["known"] & (coc <= 1)
    else:
        region["far_recoverable"] = recoverable_mask(unit,case,state)
    return region


def method_result(rgb, depth, focus, zero_visible=None, zero_combined=None, **notes):
    if focus is None or not np.isfinite(focus) or focus <= .05:
        raise ValueError("No valid depth for the shared P2 subject")
    rgb = np.asarray(rgb)
    if rgb.shape != (512,512,3) or not np.isfinite(rgb).all():
        raise ValueError("Invalid display RGB")
    return dict(rgb=rgb, depth=depth, focus=focus, zero_visible=zero_visible, zero_combined=zero_combined,
                audit=notes)


def case_output(unit, frame, case, index, state, query, dr):
    K, intr = state["K"], state["intr"]
    radius = .05/(2*case["fstop"])
    offsets = aperture_disk(CONFIG["native_eval_rays"],radius,device=K.device,dtype=K.dtype)
    center = K.new_zeros((1,2))
    lookup, A = state["lookup"], state["a_color"]

    def f2(focus, rays):
        return render_f2(A,lookup["background_depth"],lookup["fill_rgb_linear"],K,focus,rays,(512,512))

    def render(cloud, focus, rays, fill=None):
        return raster(cloud,K,focus,rays,512,512,fill)

    out, skip = {}, {}
    def record(name, callback):
        try:
            out[name] = callback()
        except (ValueError, RuntimeError, OSError, AssertionError) as error:
            skip[name] = f"{type(error).__name__}: {error}"
            if isinstance(error, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()

    start_focus = state["rig"]["focuses"][index]
    sharp_focus = state["sharp_focuses"][index]
    gt_focus = p2_focus(state["oracle_z"],state["subject"][index])

    def guard_focus(focus):
        if focus is None or not np.isfinite(focus) or focus <= .05:
            raise ValueError("No valid depth for the shared P2 subject")

    def own_layers():
        guard_focus(start_focus)
        visible = state["rig"]["visible"]
        fallback = f2(start_focus, offsets)
        zfill = f2(start_focus, center)
        pin = display_tensor(render(visible,start_focus,center)[0])
        return visible, fallback, zfill, pin

    start = None
    try:
        start = own_layers()
    except (ValueError, RuntimeError, OSError) as error:
        skip.update({"stage3":str(error),"stage3_f2":str(error)})
    if start:
        vis, fill, zfill, z0 = start

        def evaluate_ours(open_query):
            hidden = gallery.query_layer(query,state["rig"],K,frame,case,index) if open_query else None
            cloud = merge_layers(vis,hidden)
            return method_result(display_tensor(render(cloud,start_focus,offsets,fill).mean(0)),
                state["rig"]["depth"].cpu().numpy(),start_focus,z0,
                display_tensor(render(cloud,start_focus,center,zfill)[0]),
                hidden_gaussians=0 if hidden is None else len(hidden.means))

        record("stage3_f2",lambda: evaluate_ours(False))
        record("stage3",lambda: evaluate_ours(True))

    def run_a():
        guard_focus(sharp_focus)
        return method_result(display_tensor(f2(sharp_focus,offsets).mean(0)), state["sharp"], sharp_focus,
            display_tensor(render(state["a_visible"],sharp_focus,center)[0]),
            display_tensor(f2(sharp_focus,center)[0]))

    record("A",run_a)

    def run_r2():
        if "A" not in out:
            raise ValueError("A-color prerequisite missing")
        if state["r2"] is None:
            raise ValueError(f"R1+R2 candidate projection failed: {state['retrieval']}")
        a = out["A"]
        candidates = state["r2"]
        band = state["a_bands"][index]
        px,py = candidates["uv"].astype(int).T
        keep = band[py,px]
        if not keep.any():
            return method_result(a["rgb"],a["depth"],sharp_focus,a["zero_visible"],a["zero_combined"],
                                 **state["retrieval"],case_parents=0)
        hidden = sparse_pixel_layer(candidates,keep,K,torch.eye(4,device=K.device),state["profile"],K.device)
        cloud = merge_layers(A,hidden)
        # The registered R1+R2 method splices its render inside its own ring.
        # Use the same observed LaMa residual as A and project it per pupil ray.
        tail = GaussianScene(*(getattr(A,k)[:0] for k in A.__dataclass_fields__))
        def residual(rays):
            return render_f2(tail,lookup["background_depth"],lookup["fill_rgb_linear"],K,sharp_focus,rays,(512,512))
        rgb = display_tensor(render(cloud,sharp_focus,offsets,residual(offsets)).mean(0))
        zero = display_tensor(render(cloud,sharp_focus,center,residual(center))[0])
        return method_result(np.where(band[...,None],rgb,a["rgb"]),a["depth"],sharp_focus,a["zero_visible"],
                             np.where(band[...,None],zero,a["zero_combined"]),
                             **state["retrieval"],case_parents=int(keep.sum()))

    record("R1_R2",run_r2)

    def run_dr():
        guard_focus(gt_focus)
        if isinstance(dr,str):
            raise ValueError(dr)
        rgb,zero,audit = dr.render(state["rgb"],state["oracle_z"],state["alpha"],gt_focus,intr[0,0],radius)
        return method_result(rgb,state["oracle_z"],gt_focus,None,zero,**audit)

    record("DrBokeh",run_dr)

    def run_oracle():
        guard_focus(gt_focus)
        vis = state["oracle_visible"]
        depth = K.new_tensor(state["oracle_z"])
        hidden, audit = oracle_hidden(state["gt_hidden"],state["ref"],depth,K,gt_focus,case["fstop"])
        cloud = merge_layers(vis,hidden)
        rgb = render(cloud,gt_focus,offsets,f2(gt_focus,offsets)).mean(0)
        z0 = render(vis,gt_focus,center)[0]
        z1 = render(cloud,gt_focus,center,f2(gt_focus,center))[0]
        return method_result(display_tensor(rgb),state["oracle_z"],gt_focus,display_tensor(z0),
                             display_tensor(z1),hidden=audit)

    record("GT_oracle",run_oracle)
    return out, skip


def planned_units():
    units = library_unit_paths("dev")
    if len(units) != 15 or any(unit.parent.name != "development" or read(unit/"manifest.json")["partition"] != "development"
                               for unit in units):
        raise ValueError("Expected exactly the 15 registered B development scenes")
    return units


@torch.inference_mode()
def run(output, smoke):
    commenced = time.monotonic()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise ValueError("Set CUDA_VISIBLE_DEVICES=0 explicitly")
    subprocess.run(["nvidia-smi","--query-gpu=memory.used,memory.total,utilization.gpu","--format=csv"],check=True)
    torch.set_num_threads(3)
    cv2.setNumThreads(1)
    torch.cuda.set_per_process_memory_fraction(10*GIB/torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cuda.matmul.allow_tf32=False
    units = planned_units()
    output.mkdir(parents=True,exist_ok=True)
    fixed = fixed_inputs()
    protocol = dict(fixed, scenes=[u.name for u in units], frames=list(FRAMES), stops=list(STOP_VALUES),
                    fixed_review=list(map(list,FIXED_REVIEW)), scoring="P2, Cycles LDR, 64 pupil rays",
                    crop_and_colors="Same ROI and scale across all methods within each panel.",
                    geometry="1b same-surface 30k EMA. V0, b=0 everywhere. New 20k Query.",
                    drbokeh="B2 oracle-input GT axial depth and foreground coverage. Lens 71. No native depth or alpha inference.",
                    gt_oracle="GT center depth + GT peeled hidden layers with existing placement; input RGB, V0 and F2.",
                    A="Aligned SHARP + previously fitted A-color + observed LaMa F2. No fitting in this run.",
                    R1_R2="Same A-color and calibrated SHARP depth, observed-view R1+R2 deterministic retrieval and its own aperture ring.",
                    focus="Method depth median on same nominal GT subject; no-method scoring masks are shared.",
                    smoke=smoke)
    protocol_path = output/"protocol.json"
    if protocol_path.exists() and read(protocol_path) != protocol:
        raise ValueError("Output was initialized with a different protocol")
    write(protocol_path,protocol)
    model = load_paired_checkpoint(INITIAL,"ema").eval()
    query = query_net(QUERY)
    try:
        dr = DrBokehWorker(output)
    except (ValueError, RuntimeError, OSError, ImportError, AttributeError, AssertionError) as error:
        dr = f"{type(error).__name__}: {error}"
        print("DR_SETUP_FAILED",dr,flush=True)
    from route3_ldr_u1r import model as lpips_model

    lpips = lpips_model()
    torch.set_num_threads(3)
    for unit in units:
        if smoke and not unit.name.startswith("development-005-"):
            continue
        category = read(unit/"manifest.json")["category"]
        all_cases = sorted(development_cases(unit),key=lambda c:(c["frame"], c["fstop"],c["focus_index"]))
        views = [view_from_observation(observation(unit,f,aligned=True),f) for f in TRAIN_FRAMES]
        for frame in ((30,) if smoke else FRAMES):
            cases = [c for c in all_cases if c["frame"] == frame and c["fstop"] in STOP_VALUES]
            if smoke:
                cases = [c for c in cases if c["focus_index"] == 1]
            pending = [(i,c) for i,c in enumerate(cases)
                       if not (output/"cases"/unit.name/(c["id"]+".json")).is_file()]
            if not pending:
                continue
            guard(40*2**20)
            while not readiness():
                time.sleep(15)
            try:
                state = build_frame(unit,frame,cases,model,views)
                frame_error = None
            except (ValueError,RuntimeError,OSError) as error:
                frame_error = f"{type(error).__name__}: {error}"
                torch.cuda.empty_cache()
            for index,case in pending:
                tic = time.monotonic()
                path = output/"cases"/unit.name/(case["id"]+".json")
                path.parent.mkdir(parents=True,exist_ok=True)
                row = dict(case, scene=unit.name, category=category)
                try:
                    if frame_error:
                        raise ValueError(frame_error)
                    if not state["subject"][index].any():
                        raise ValueError("no_valid_nominal_subject")
                    masks = score_masks(unit,case,state)
                    truth = optical_target(unit,unit.name,case)
                    predictions,skips = case_output(unit,frame,case,index,state,query,dr)
                    save_case(output,row,state,truth,masks,predictions,skips,lpips)
                except (ValueError,RuntimeError,OSError) as error:
                    row.update(scores={},skips={name:f"{type(error).__name__}: {error}" for name in METHODS})
                    torch.cuda.empty_cache()
                row["seconds"] = time.monotonic()-tic
                write(path,row)
                print("DEV15",unit.name,case["id"],len(row["scores"]),row["skips"],round(row["seconds"],1),flush=True)
            if frame_error is None:
                del state
        del views
    if isinstance(dr,DrBokehWorker):
        dr.close()
    finish(output, time.monotonic()-commenced)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--smoke",action="store_true")
    args = p.parse_args()
    run(args.output.resolve(),args.smoke)
