"""Measure the native tiled MagicBokeh generator on Table 3's four fixed RGBs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def focus_diagnostic(out, inputs_audit, common):
    """Check native automatic focus against the requested plane, without inference."""
    import cv2
    import numpy as np
    from PIL import Image
    rows = []
    for item in inputs_audit:
        sid, K = item["scene"], item["K"]
        source = ROOT / common["input"] / sid
        paths = list((source / "disparity").glob("00000_zf_*.png"))
        if len(paths) != 1:
            raise ValueError(f"Expected one benchmark control map: {sid}")
        focus = float(paths[0].stem.split("_zf_")[1])
        geometry = np.asarray(Image.open(paths[0])).astype(float) / 255
        difference = np.abs(geometry - focus)
        nearest = float(difference.min())
        target = np.uint8(difference == nearest)
        _, labels, stats, _ = cv2.connectedComponentsWithStats(target, connectivity=8)
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        region = np.uint8(labels == largest)
        # Image boundaries count as region boundaries when selecting an interior point.
        distance = cv2.distanceTransform(np.pad(region, 1), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
        y, x = np.unravel_index(np.argmax(distance), distance.shape)
        q = (np.load(out / "depth" / (sid + ".npy")) * 255).astype(np.uint8)
        mask = np.asarray(Image.open(out / "mask" / (sid + ".png")))
        values = q[region > 0] / 255
        point_depth, native_focus = float(q[y, x] / 255), item["focus"]
        condition = np.abs(values - native_focus) * (K / 32)
        rows.append(dict(scene=sid, K=K, benchmark_focus=focus,
            nearest_shared_geometry_error=nearest,
            target_plane_visible_within_one_png8_step=nearest <= 1 / 255,
            target_point_xy=[int(x), int(y)],
            point_rule="Farthest interior pixel of largest component at shared-depth value nearest requested focus; diagnostic only",
            native_DA_depth_at_point=point_depth, native_LDF_median_focus=native_focus,
            focus_difference_abs=abs(point_depth - native_focus), target_component_area=int(region.sum()),
            target_DA_depth_quantiles=np.quantile(values, [.1, .5, .9]).tolist(),
            target_component_native_focused_fraction=float((condition <= .1).mean()),
            LDF_nonzero_pixels_on_target_fraction=float(((mask > 0) & (region > 0)).sum() / max(1, (mask > 0).sum()))))
    return dict(status="NATIVE_AUTOMATIC_FOCUS_CHECK_ON_FOUR_FIXED_TIMING_INPUTS", rows=rows,
        scope="Shared geometry labels the diagnostic target and point; it is not generator conditioning. No paired quality metric or adapter freeze.",
        next="Verify official point-focus path in a target-aware Synthetic adapter preflight")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--focus-control", choices=["native-auto", "frozen-b"], default="native-auto")
    args = parser.parse_args()
    from measure_efficiency import assert_idle_gpu, gpu_processes, ParticipatingParameters
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0"
    assert_idle_gpu()
    protocol = json.loads((ROOT / "configs/magicbokeh-native.json").read_text())
    frozen = None
    if args.focus_control == "frozen-b":
        frozen = json.loads((ROOT / "configs/magicbokeh-track-a-frozen.json").read_text())
        assert frozen["status"] == "FROZEN_TARGET_GUIDED_DEPTH_MEDOID_POINT_FOCUS"
        protocol.update(name="MagicBokeh frozen medoid B Table3 generator cost", focus_control="frozen-b",
            native_input=frozen["model_input_permission"], prompt=frozen["prompt"],
            synthetic_status=frozen["status"], focus="Native PNG8 predicted depth at target-guided medoid click",
            RNG="Separate timing process seed0 and continuous stream over warmups/repeats; does not change quality outputs")
    common = json.loads((ROOT / "configs/efficiency-track-a.json").read_text())
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    inputs, depth, mask = [out / x for x in ["inputs", "depth", "mask"]]
    for p in [inputs, depth, mask]:
        p.mkdir()
    for sid in common["scenes"]:
        (inputs / (sid + ".png")).symlink_to((ROOT / common["input"] / sid / "rgb/00000.png").resolve())
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if frozen is None:
        subprocess.run([
            sys.executable, "prepare_data.py", "--input_dir", str(inputs),
            "--depth_dir", str(depth), "--mask_dir", str(mask),
            "--resnet_path", str(ROOT / "models/DrBokeh/resnet50-19c8e357.pth"),
            "--model_path", str(ROOT / "models/DrBokeh/model-40"),
            "--depth_ckpt", str(ROOT / "models/MagicBokeh/DAdepth.pth"),
        ], cwd=ROOT.parent / "MagicBokeh", check=True)
    else:
        for sid in common["scenes"]:
            (depth / (sid + ".npy")).symlink_to((ROOT / frozen["preflight"] / "depth" / (sid + "-00000.npy")).resolve())
    assert_idle_gpu()

    import numpy as np
    import torch
    from PIL import Image
    from baselines.magicbokeh import OfficialMagicBokeh
    torch.set_num_threads(4)
    torch.manual_seed(protocol["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    adapter = OfficialMagicBokeh(ROOT.parent / "MagicBokeh", ROOT / "models/MagicBokeh", ROOT / "models/sd21-base")
    components = {name: getattr(adapter.model, name) for name in ["unet", "controlnet", "vae", "text_encoder"]}
    tracker = ParticipatingParameters(components)
    inputs_audit, rows, warmups, samples, errors = [], [], [], [], []
    stop = threading.Event()

    def memory_sample():
        return int(subprocess.check_output([
            "nvidia-smi", "--id=0", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True).strip())

    def monitor():
        while not stop.is_set():
            try:
                samples.append(memory_sample())
            except Exception as error:
                errors.append(str(error))
                return
            stop.wait(protocol["memory_sample_seconds"])

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    try:
        with torch.no_grad():
            for sid, K in zip(common["scenes"], common["K"]):
                image = Image.open(inputs / (sid + ".png")).convert("RGB")
                d = np.load(depth / (sid + ".npy"))
                if image.size != (1024, 576) or d.shape != (576, 1024) or not np.isfinite(d).all():
                    raise ValueError(f"Invalid native timing input: {sid}")
                q = Image.fromarray((d * 255).astype(np.uint8), mode="L")
                point_info = {}
                if frozen is None:
                    m = Image.open(mask / (sid + ".png")).convert("L")
                    focus = adapter.module.adjust_depth_with_mask(q, m) / 255
                    point_info["mask_nonzero_fraction"] = float((np.asarray(m) > 0).mean())
                else:
                    from baselines.magicbokeh_focus import visible_target_alpha, safe_target_region, medoid_point
                    folder = ROOT / common["input"] / sid
                    metadata = json.loads((folder / "metadata.json").read_text())
                    alpha, identity = visible_target_alpha(folder, metadata, metadata["frames"][0])
                    region, distance, target = safe_target_region(alpha)
                    if region is None:
                        raise ValueError("Frozen target point unavailable")
                    point, medoid = medoid_point(region, distance, np.asarray(q))
                    focus = q.getpixel(tuple(point)) / 255
                    point_info.update(point_B_xy=point, target_source_id=identity, target=target, medoid=medoid)
                prepared = adapter.prepare(image, q, focus, K)
                if len(prepared["positions"]) != 6:
                    raise ValueError("Expected six512px native tiles")
                inputs_audit.append(dict(scene=sid, K=K, focus=focus, native_tiles=6,
                    positions=prepared["positions"], depth_range=[float(d.min()), float(d.max())],
                    **point_info,
                    input_permission="RGB plus native automatic focus; no external geometry, GT alpha or hidden layers" if frozen is None else frozen["model_input_permission"]))
                for repeat in range(-protocol["warmups_per_scene"], protocol["repeats_per_scene"]):
                    if not set(gpu_processes()) <= {os.getpid()}:
                        raise RuntimeError("Foreign GPU process appeared; discard timing attempt")
                    start_sample = len(samples)
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    start = time.perf_counter()
                    value = adapter.forward_cuda(prepared, prompt=protocol.get("prompt", ""))
                    torch.cuda.synchronize()
                    seconds = time.perf_counter() - start
                    allocated = torch.cuda.max_memory_allocated() / 2**30
                    reserved = torch.cuda.max_memory_reserved() / 2**30
                    # Validation and CPU postprocessing happen after the synchronized timer.
                    if value.shape != (1, 3, 576, 1024) or not torch.isfinite(value).all():
                        raise ValueError("Invalid native generator output")
                    observed = samples[start_sample:] or [memory_sample()]
                    row = dict(scene=sid, K=K, repeat=repeat, output_frames=1, native_tile_calls=6,
                        core_seconds=seconds, peak_torch_allocated_GiB=allocated,
                        peak_torch_reserved_GiB=reserved, peak_device_memory_MiB_sampled=max(observed))
                    (rows if repeat >= 0 else warmups).append(row)
                    if repeat == 0:
                        adapter.to_pil(value, image).save(out / (sid + "-native.png"))
                    print("MAGICBOKEH_TIMING", sid, K, repeat, round(seconds, 5), flush=True)
                    del value
                del prepared
            if errors:
                raise RuntimeError(errors)
            if not set(gpu_processes()) <= {os.getpid()}:
                raise RuntimeError("Foreign GPU process appeared")
    finally:
        stop.set()
        monitor_thread.join()
    parameters = tracker.summary()
    for handle in tracker.handles:
        handle.remove()
    diagnostic = focus_diagnostic(out, inputs_audit, common) if frozen is None else dict(status="FROZEN_B_POINT_CONTROL_USED", inputs=inputs_audit)
    (out / "focus-diagnostic.json").write_text(json.dumps(diagnostic, indent=2) + "\n")
    summary = dict(status="MAGICBOKEH_NATIVE_TABLE3_COMPLETE" if frozen is None else "MAGICBOKEH_FROZEN_B_TABLE3_COMPLETE", protocol=protocol,
        common_scene_selection=common["scene_selection"], inputs=inputs_audit, rows=rows, warmups=warmups,
        runtime=dict(torch=torch.__version__, cuda=torch.version.cuda,
                     gpu=subprocess.check_output(["nvidia-smi", "--id=0", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], text=True).strip(),
                     author_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT.parent / "MagicBokeh", text=True).strip()),
        parameters=parameters,
        loaded_parameter_components_M={k:sum(p.numel() for p in v.parameters()) / 1e6 for k,v in components.items()},
        Table3=dict(Time_per_output_frame_seconds=float(np.mean([r["core_seconds"] for r in rows])),
                    Params_M=parameters["active_parameters"] / 1e6, GFLOPs_per_output_frame=None,
                    GFLOPs_status="N/A: full operator coverage not established; no partial count reported",
                    Peak_device_memory_GiB_sampled=max(r["peak_device_memory_MiB_sampled"] for r in rows) / 1024,
                    Peak_torch_allocated_GiB=max(r["peak_torch_allocated_GiB"] for r in rows),
                    Peak_torch_reserved_GiB=max(r["peak_torch_reserved_GiB"] for r in rows)),
        by_K={str(K):dict(mean=float(np.mean([r["core_seconds"] for r in rows if r["K"] == K])),
                         std=float(np.std([r["core_seconds"] for r in rows if r["K"] == K], ddof=1))) for K in common["K"]})
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["Table3"], indent=2))


if __name__ == "__main__":
    main()
