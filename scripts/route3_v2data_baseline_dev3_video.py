"""Run the frozen Any-to-Bokeh video adapter using RGB-only VDA depth."""
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time

import cv2
import numpy as np
import torch

from route3_v2data_baseline_dev3 import (
    FOCUSES, FRAMES, LOCK, OUT, ROOT, SCENES, SHAPE, STOPS, case_id, central_square,
    coc_slope, display, frame_truth, guard, model_canvas, p1_control, prediction_path,
    read, read_exr, scene_root, write,
)
from route3_v2data_baseline_dev3_methods import q8, save, unit_range

REFERENCE = ROOT / "configs/davis-baselines.json"
FROZEN = read(REFERENCE)["any2bokeh"]
REPO = ROOT.parent / "any-to-bokeh"


def configure():
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(23.5 * 2**30 / torch.cuda.get_device_properties(0).total_memory)


def prepare_video_inputs():
    root = OUT / "video-input"
    blender = ROOT.parent / "tools/blender-4.5.13-linux-x64/blender"
    for scene in SCENES:
        original = scene_root(scene)
        folder = root / scene
        rows = []
        for frame in range(1, 61):
            key = f"f{frame:03d}"
            if (original / "inputs" / (key + ".exr")).exists():
                continue
            rows.append(dict(scene=scene, id=key, frame=frame, blend=str(original / "scene.blend"),
                             size=512, spp=256, seed=20261112, dof=False, omit_reserved_passes=True,
                             output=str(folder / (key + ".exr"))))
        if all(os.path.isfile(r["output"]) for r in rows):
            continue
        guard(len(rows) * 512 * 512 * 16)
        plan = root / f"{scene}-jobs.json"
        if plan.exists() and read(plan) != rows:
            raise ValueError(f"Video input plan changed for {scene}")
        write(plan, rows)
        log = OUT / "logs" / f"video-{scene}-jobs.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a") as stream:
            subprocess.run(["flock", str(LOCK), "nice", "-n", "10", str(blender), "-b", "-t", "4",
                            "--python-exit-code", "1", "--python", str(ROOT / "scripts/route3_s2a_blender.py"),
                            "--", "--jobs", str(plan)], check=True, stdout=stream, stderr=subprocess.STDOUT,
                           env=dict(os.environ, CUDA_VISIBLE_DEVICES="0"))


def rgb_sequence(scene, length):
    folder = OUT / "video-input" / scene / "rgb"
    folder.mkdir(parents=True, exist_ok=True)
    frames = []
    original = scene_root(scene) / "inputs"
    for index in range(1, length + 1):
        key = f"f{index:03d}.exr"
        path = original / key
        if not path.exists():
            path = folder.parent / key
        image = q8(display(read_exr(path)))
        frames.append(image)
        dest = folder / f"{index:05d}.png"
        if not dest.exists() and not cv2.imwrite(str(dest), q8(model_canvas(image / 255.))[..., ::-1]):
            raise OSError(dest)
    return np.stack(frames)


def vda(smoke):
    config = read(ROOT / "configs/davis-eval-v2.json")["vda"]
    repo = (ROOT / config["repository"]).resolve()
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if commit != config["commit"]:
        raise ValueError("Unexpected Video Depth Anything revision")
    sys.path.insert(0, str(repo))
    from video_depth_anything.video_depth import VideoDepthAnything
    from xformers.ops.fmha import _set_use_fa3

    configure()
    _set_use_fa3(False)
    torch.manual_seed(0)
    model = VideoDepthAnything(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024])
    model.load_state_dict(torch.load(ROOT / config["checkpoint"], map_location="cpu", weights_only=True), strict=True)
    model.cuda().eval()
    for scene in (SCENES[:1] if smoke else SCENES):
        dest = OUT / "video-input" / scene / ("vda-smoke.npz" if smoke else "vda.npz")
        if dest.exists():
            continue
        guard(320 * 2**20)
        frames = rgb_sequence(scene, 12 if smoke else 60)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.monotonic()
        value, _ = model.infer_video_depth(frames, 1, input_size=518, device="cuda", fp32=False)
        torch.cuda.synchronize()
        value = np.uint8(unit_range(value) * 255)
        manifest = dict(depth_seconds=time.monotonic() - start, frames=len(frames),
                        peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                        peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                        commit=commit, checkpoint=config["checkpoint"], normalize="whole sequence PNG8 truncation")
        np.savez_compressed(dest, disparity=value, audit=json.dumps(manifest))
        print("VDA", scene, manifest["frames"], manifest["depth_seconds"], flush=True)


def get_pipeline():
    os.environ.setdefault("HF_HOME", str(ROOT / "models/huggingface"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    sys.path.insert(0, str(REPO))
    from models.unet import UNetSpatioTemporalConditionModel
    from models.vae import AutoencoderKLTemporalDecoder
    from pipelines.any2bokeh_pipe import StableVideoDiffusionPipeline

    checkpoint = ROOT / FROZEN["checkpoint"]
    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        str(checkpoint), subfolder="unet", torch_dtype=torch.float16,
        low_cpu_mem_usage=False, local_files_only=True)
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        str(checkpoint / "vae"), torch_dtype=torch.float16, local_files_only=True)
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt", unet=unet, vae=vae, torch_dtype=torch.float16,
        variant="fp16", low_cpu_mem_usage=False, local_files_only=True).to("cuda:0")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def run_clip(pipe, folder, rgbs, disparity, focal, K, window, decode, keep):
    from utils.dataset import BokehCocMPIDatasetWholeVid, collate_fn_val_coc_mpi_whole_video

    depth_root, rgb_root = folder / "depth", folder / "rgb"
    depth_root.mkdir(parents=True, exist_ok=True)
    rgb_root.mkdir(exist_ok=True)
    for index, (d, f, k, source) in enumerate(zip(disparity, focal, K, rgbs), 1):
        target = rgb_root / f"{index:05d}.png"
        if not target.exists():
            target.symlink_to(source.resolve())
        depth = depth_root / f"{index:05d}_zf_{f:.9f}_k_{k:.9f}.png"
        if not depth.exists() and not cv2.imwrite(str(depth), d):
            raise OSError(depth)
    control_file = folder / "input.csv"
    with control_file.open("w") as stream:
        csv.writer(stream).writerows([["aif_folder", "disp_folder", "k"], [rgb_root, depth_root, "change"]])
    dataset = BokehCocMPIDatasetWholeVid(str(control_file), sample_size=SHAPE,
                                       group_frames=window, overlap_frames=window // 2)
    batch = collate_fn_val_coc_mpi_whole_video([dataset[0]])
    frames = {}
    cache = None
    output_count = 0
    seconds = 0.
    heartbeats = []
    length = batch["coc"].shape[0]
    torch.cuda.reset_peak_memory_stats()
    for i in range(0, length, 2):
        tensors = {k: batch[k][i:i + 2].to("cuda", dtype=torch.float32 if k in ("pixel_values", "coc")
                   else torch.float16) for k in ("pixel_values", "coc", "k", "mpi_mask", "weight")}
        torch.cuda.synchronize()
        tic = time.monotonic()
        with torch.inference_mode():
            result, cache = pipe(tensors["coc"], tensors["pixel_values"], input_k=tensors["k"],
                mpi_mask=tensors["mpi_mask"], weight=tensors["weight"], latent_cache=cache,
                is_last_batch=i + 2 >= length, overlap_frame=window // 2, height=SHAPE[0], width=SHAPE[1],
                num_frames=window, decode_chunk_size=decode, motion_bucket_id=127, fps=7,
                noise_aug_strength=0., num_inference_steps=1,
                generator=torch.Generator(device="cpu").manual_seed(0),
                batch_size=tensors["weight"].shape[0], output_type="pt")
        torch.cuda.synchronize()
        elapsed = time.monotonic() - tic
        seconds += elapsed
        heartbeats.append(elapsed)
        if not torch.isfinite(result[0]).all():
            raise ValueError("Nonfinite Any-to-Bokeh result")
        for j, image in enumerate(result[0], output_count + 1):
            if j in keep:
                frames[j] = q8(image.permute(1, 2, 0).cpu().numpy())
        output_count += len(result[0])
        print("ATB_CHUNK", folder.name, output_count, elapsed, flush=True)
        del result, tensors
    if output_count != len(disparity) or frames.keys() != set(keep):
        raise ValueError(f"Decoded {output_count} frames for {len(disparity)} ordered inputs")
    return frames, dict(clip_render_s=seconds, clip_frames=output_count, render_s=seconds / output_count,
                        chunk_s=heartbeats, peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                        peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)


def calibrate(pipe, window, decode, tag):
    from calibrate_deeplens_control import edge_response
    folder = OUT / "calibration" / "any2bokeh" / tag
    record = folder / "control.json"
    if record.exists():
        return read(record)["radius_per_native_strength"]
    length = 2 * window
    source = folder / "plane.png"
    source.parent.mkdir(parents=True, exist_ok=True)
    img = np.full((*SHAPE, 3), 64, np.uint8)
    img[:, SHAPE[1] // 2:] = 191
    cv2.imwrite(str(source), img)
    d = np.full((length, *SHAPE), 64, np.uint8)
    d[:, :64, :120] = 191
    rows = []
    for k in (4., 8., 16., 32.):
        images, _ = run_clip(pipe, folder / f"K{k:g}", [source] * length, d,
                            [191 / 255] * length, [k] * length, window, decode, (length // 2,))
        result = edge_response(images[length // 2].astype(np.float64) / 255, "vertical")
        rows.append(dict(K=k, radius_px=result["edge_10_90_width_px"] / 1.374097652,
                         ringing=result["negative_lsf_mass_ratio"]))
    x = np.array([row["K"] * (191 - 64) / 255 for row in rows])
    y = np.array([row["radius_px"] for row in rows])
    scale = float(x @ y / (x @ x))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Toy-plane Any-to-Bokeh calibration has no positive response")
    write(record, dict(scene="independent two-plane edge", rows=rows, radius_per_native_strength=scale,
                       monotonic=bool(np.all(np.diff(y) > 0)),
                       rmse_px=float(np.sqrt(np.mean((y - scale * x)**2)))))
    return scale


def evaluate(smoke, window, decode, encoder_scale_one):
    configure()
    torch.manual_seed(0)
    revision = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    if revision != FROZEN["author_commit"]:
        raise ValueError("Any-to-Bokeh author revision differs from frozen config")
    tag = f"w{window}-d{decode}" + ("-scale-one" if encoder_scale_one else "")
    stage = "smoke" if smoke else "full"
    inventory = dict(method="any2bokeh", arm="native", author_commit=revision, stage=stage,
        config=str(REFERENCE.relative_to(ROOT)), config_sha256=hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
        group_frames=window, overlap_frames=window // 2, decode_chunk_size=decode, seed=0, device="RTX 5090",
        deviations=["Existing 13 RGB keyframes plus missing contiguous AIF frames regenerated from the same frozen scene",
                    "Square scaled to 576px and mirror-padded to frozen 1024x576",
                    "P1 controls interpolated between the three evaluated frames for temporal context"])
    if (window, decode) != (8, 8):
        inventory["deviations"].append("Reduced frozen temporal configuration for RTX 5090 memory budget")
    if encoder_scale_one:
        inventory["deviations"].append("Skip the exact division by 1 only inside the VAE encoder residual blocks")
    record = OUT / "inventory" / f"any2bokeh-{stage}-{tag}.json"
    try:
        pipe = get_pipeline()
        if encoder_scale_one:
            from baselines.any2bokeh_scale1 import EncoderScale1
            EncoderScale1(pipe, enabled=True)
        response = 1. if smoke else calibrate(pipe, window, decode, tag)
        if not smoke:
            inventory["calibration"] = read(OUT / "calibration" / "any2bokeh" / tag / "control.json")
            torch.manual_seed(0)
        completed = 0
        pending = []
        for scene in (SCENES[:1] if smoke else SCENES):
            video = OUT / "video-input" / scene
            with np.load(video / ("vda-smoke.npz" if smoke else "vda.npz")) as archive:
                disparity, depth_audit = archive["disparity"], json.loads(archive["audit"].item())
            keep = FRAMES[:1] if smoke else FRAMES
            length = len(disparity)
            if smoke:
                length = 2 * window
                disparity = disparity[:length]
            widths = list(range(1, length + 1))
            d_canvas = np.stack([model_canvas(frame / 255.) for frame in disparity])
            depth_png = q8(d_canvas)
            for stop in (STOPS[:1] if smoke else STOPS):
                for nominal in (FOCUSES[:1] if smoke else FOCUSES):
                    focals, strengths, fits = [], [], []
                    for index in keep:
                        truth, K, _, masks = frame_truth(scene, index)
                        pred = depth_png[index - 1] / 255.
                        focus, fit = p1_control(pred, truth, masks["common_visible_background"], nominal)
                        focals.append(focus)
                        strengths.append(coc_slope(K, stop, nominal) * SHAPE[0] / 512 / abs(fit["a"]) / response)
                        fits.append(fit)
                    focal = np.interp(widths, keep, focals)
                    strengths_full = np.interp(widths, keep, strengths)
                    sources = [video / "rgb" / f"{i:05d}.png" for i in widths]
                    work = OUT / "video-controls" / stage / tag / scene / f"N{stop:g}-F{nominal:g}"
                    outputs, timing = run_clip(pipe, work, sources, depth_png, focal, strengths_full,
                                               window, decode, keep)
                    for frame, focus, strength, fit in zip(keep, focals, strengths, fits):
                        cid = case_id(frame, stop, nominal)
                        dest = (OUT / "smoke" / f"any2bokeh-{tag}.npz" if smoke else
                                OUT / "video-candidates" / tag / scene / f"{cid}.npz")
                        if not smoke:
                            pending.append((dest, prediction_path("any2bokeh", "native", "P1", scene, cid)))
                        peak = max(timing["peak_allocated_gib"], depth_audit["peak_allocated_gib"])
                        save(dest, central_square(outputs[frame].astype(np.float32) / 255),
                             dict(timing, depth_s=depth_audit["depth_seconds"] / depth_audit["frames"],
                                  focus_distance_m=nominal, focus_disparity=focus, p1_fit_a=fit["a"],
                                  p1_fit_b=fit["b"], p1_holdout_rmse=fit["heldout_rmse"], K=strength, seed=0,
                                  renderer_and_depth_peak_gib=peak, window=window, decode_chunk_size=decode))
                        completed += 1
        if pending and len(pending) != len(SCENES) * len(FRAMES) * len(STOPS) * len(FOCUSES):
            raise ValueError("Any-to-Bokeh prediction sweep is incomplete")
        for candidate, destination in pending:
            destination.parent.mkdir(parents=True, exist_ok=True)
            candidate.replace(destination)
        inventory.update(status="completed", completed=completed, depth=depth_audit)
    except Exception as error:
        inventory.update(status="failed", error_type=type(error).__name__, reason=str(error))
        raise
    finally:
        write(record, inventory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("inputs", "vda", "evaluate"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--window", type=int, default=8, choices=(4, 6, 8))
    parser.add_argument("--decode", type=int, default=8, choices=(2, 3, 4, 6, 8))
    parser.add_argument("--encoder-scale-one", action="store_true")
    args = parser.parse_args()
    if args.operation == "inputs":
        prepare_video_inputs()
    elif args.operation == "vda":
        vda(args.smoke)
    else:
        evaluate(args.smoke, args.window, args.decode, args.encoder_scale_one)
