"""One posed monocular CamVFX window, one MoVieS forward pass, no training.

With fixed cameras absolute scale cannot be inferred from poses. Estimate one
multiplicative scale using independently exported static-background mesh depth;
save normalized and metric geometry in the first camera's canonical frame.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image
import torch
from safetensors.torch import load_file

PROJECT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repository", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--scene", type=Path, required=True)
    p.add_argument("--geometry", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--view", type=int, default=0)
    p.add_argument("--frames", type=int, default=13)
    p.add_argument("--size", type=int, default=518)
    a = p.parse_args()
    if not 1 <= a.frames <= 13 or a.size > 518 or a.size % 14:
        raise ValueError("Use <=13 frames and a multiple-of-14 long edge <=518")
    a.repository, a.weights, a.scene, a.geometry, a.output = [v.resolve() for v in
        [a.repository, a.weights, a.scene, a.geometry, a.output]]
    a.output.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location("camvfx_input", PROJECT / "src/camvfx4d.py")
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    frames, metadata = helper.load_camera_frames(a.scene / "pinhole/transforms_train.json")
    chosen = np.rint(np.linspace(1, 50, a.frames)).astype(int).tolist()
    cameras = np.stack([frames[(a.view, i)]["c2w"] for i in chosen])
    original_camera = cameras[0].copy()
    cameras = np.linalg.inv(cameras[:1]) @ cameras
    K = helper.intrinsics_from_fov(metadata, a.size, a.size).astype(np.float32)
    normalized_K = np.array([K[0, 0] / a.size, K[1, 1] / a.size, .5, .5], np.float32)
    images = np.stack([np.asarray(Image.open(a.scene / f"pinhole/{a.view}/{i:06d}.png").convert("RGB").resize(
        (a.size, a.size), Image.Resampling.BICUBIC), np.float32).transpose(2, 0, 1) / 255 for i in chosen])
    times = np.array([frames[(a.view, i)]["time"] for i in chosen], np.float32)
    times = (times - times.min()) / max(float(times.max() - times.min()), 1e-8)
    data = dict(images=torch.from_numpy(images)[None], C2W=torch.from_numpy(cameras).float()[None],
                intrinsics=torch.from_numpy(np.broadcast_to(normalized_K, (a.frames, 4)).copy())[None],
                times=torch.from_numpy(times)[None], query=torch.zeros(1, 1))
    os.chdir(a.repository)
    (a.repository / "resources").mkdir(exist_ok=True)
    sys.path[:0] = [str(a.repository), str(a.repository / "extensions/vggt")]
    from src.options import opt_dict
    from src.models import SplatRecon
    from extensions.vggt.vggt.models.vggt import VGGT
    # Every parameter is overwritten by the strict MoVieS checkpoint below.
    # Avoid downloading an unnecessary separate VGGT checkpoint for initialization.
    VGGT.from_pretrained = classmethod(lambda cls, *args, **kwargs: cls())
    opt = opt_dict["movies"]
    opt.memory_efficient_attention = False  # official torch SDPA implementation
    started = time.monotonic()
    model = SplatRecon(opt, load_lpips=False)
    state = load_file(str(a.weights))
    loading = model.load_state_dict(state, strict=True)
    del state
    model.eval().to("cuda:0")
    load_seconds = time.monotonic() - started
    data = {k: v.to(device="cuda:0", dtype=torch.bfloat16) for k, v in data.items()}
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); started = time.monotonic()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        outputs, motion, motion_gaussians = model.backbone(data["images"], data["C2W"], data["intrinsics"],
                                                         data["times"], data["query"], frames_chunk_size=4)
    torch.cuda.synchronize(); forward_seconds = time.monotonic() - started
    forward_peak = torch.cuda.max_memory_allocated() / 2**20
    saved = {k: v.float().cpu() for k, v in outputs.items()}
    saved["query_motion"] = None if motion is None else motion.float().cpu()
    saved["query_motion_gaussians"] = None if motion_gaussians is None else [{k: v.float().cpu() for k, v in d.items()} for d in motion_gaussians]
    saved.update(input_frame_ids=chosen, input_frame_indices=[i - 1 for i in chosen],
                 query_time=0., original_camera=torch.from_numpy(original_camera),
                 normalized_C2W=torch.from_numpy(cameras), intrinsics_512=helper.intrinsics_from_fov(metadata).tolist())
    torch.save(saved, a.output / "same_forward_outputs.pt")
    prediction = outputs["depth"][0, 0, 0].float().cpu().numpy()
    gt = np.load(a.geometry)["reference_depth"]
    gt = np.asarray(Image.fromarray(gt.astype(np.float32)).resize((a.size, a.size), Image.Resampling.NEAREST))
    valid = (gt > 0) & np.isfinite(prediction) & (prediction > 0)
    # Prefer far static background over moving foreground in this preparation probe.
    background = valid & (gt >= np.quantile(gt[valid], .5))
    scale = float(np.median(gt[background] / prediction[background]))
    relative = np.abs(prediction * scale - gt) / np.maximum(gt, 1e-6)
    render_inputs = dict(outputs)
    if motion is not None:
        render_inputs["offset"] = motion[:, 0, :, :3]
    if motion_gaussians is not None:
        render_inputs.update(motion_gaussians[0])
    torch.cuda.synchronize(); started = time.monotonic()
    with torch.inference_mode():
        rendered = model.gs_renderer.render(render_inputs, data["C2W"], data["intrinsics"],
                                             data["C2W"][:, :1], data["intrinsics"][:, :1], return_pc=True)
    torch.cuda.synchronize(); render_seconds = time.monotonic() - started
    pc = rendered["pc"][0]
    opacity = render_inputs.get("motion_opacity", render_inputs["opacity"])
    opacity = model.gs_renderer.opacity_activation(opacity)[0].permute(0, 2, 3, 1).reshape(-1)
    keep = opacity > opt.opacity_threshold if opt.prune_ratio > 0 else torch.ones_like(opacity, dtype=torch.bool)
    source_ids = (torch.tensor(chosen, device="cuda:0") - 1).repeat_interleave(a.size * a.size)[keep]
    assert len(source_ids) == len(pc.xyz)
    np.savez(a.output / "metric_gaussians.npz", means=pc.xyz.float().cpu().numpy() * scale,
             scales=pc.scale.float().cpu().numpy() * scale, quats=pc.rotation.float().cpu().numpy(),
             opacities=pc.opacity.float().cpu().numpy().reshape(-1), colors_srgb=pc.color.float().cpu().numpy(),
             source_frames=source_ids.cpu().numpy(), canonical_to_dataset_world=original_camera, scale_to_meters=scale,
             K=helper.intrinsics_from_fov(metadata).astype(np.float32), world_to_camera=np.eye(4, dtype=np.float32),
             source_frame_indexing=np.array("zero-based global frame indices; filename ID minus one"))
    rgb = rendered["image"][0, 0].float().permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    Image.fromarray(np.uint8(rgb * 255)).save(a.output / "native_pinhole.png")
    np.savez(a.output / "depth_scale_alignment.npz", normalized_depth=prediction, metric_depth=prediction * scale,
             geometry_depth=gt, alignment_background_mask=background)
    record = {"frames": chosen, "view": a.view, "input_size": [a.size, a.size], "query_times": [0.],
              "current_frame_id": chosen[0],
              "current_frame_index": chosen[0] - 1,
              "dataset_poses_used": True, "camera_baseline_m": float(np.linalg.norm(cameras[:, :3, 3], axis=-1).max()),
              "checkpoint_strict_load": str(loading), "initialization": "VGGT architecture only, then complete strict MoVieS checkpoint",
              "attention": "official torch SDPA path", "model_load_seconds": load_seconds,
              "forward_seconds": forward_seconds, "forward_peak_allocated_mib": forward_peak,
              "native_render_seconds": render_seconds, "gaussians_after_native_opacity_filter": len(pc.xyz),
              "scale_to_dataset_meters": scale, "alignment_pixels": int(background.sum()),
              "static_background_relative_depth_error_p50": float(np.median(relative[background])),
              "static_background_relative_depth_error_p90": float(np.quantile(relative[background], .9)),
              "scale_method": "one median multiplicative ratio to independent mesh depth, no depth shift",
              "scale_limit": "fixed input poses have zero baseline; poses alone cannot supply absolute scale; mesh time is unspecified",
              "metric_transform": "multiply depth, Gaussian means/scales and motion offsets by the same scale; canonical world uses first input camera",
              "same_forward_outputs_saved": True, "training": False}
    (a.output / "inference.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
