"""Measure actual25-frame frozen MagicBokeh B, including its native depth path."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open("x") as file:
        json.dump(value, file, indent=2, allow_nan=False)
        file.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    common = read(ROOT / "configs/efficiency-track-a.json")
    protocol = read(ROOT / "configs/magicbokeh-video-e2e.json")
    frozen = read(ROOT / protocol["frozen_adapter"])
    assert frozen["status"] == "FROZEN_TARGET_GUIDED_DEPTH_MEDOID_POINT_FOCUS"
    assert common["frames"] == 25 and common["resolution"] == [1024, 576]
    assert common["warmup_clips_per_scene"] == 1 and common["repeats_per_scene"] == 5
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0"
    out = args.output_dir.resolve()
    if (out / "summary.json").exists():
        previous = read(out / "summary.json")
        assert previous["status"] == "MAGICBOKEH_B_VIDEO_E2E_COMPLETE"
        assert read(out / "protocol.json") == dict(common=common, magic=protocol, frozen=frozen)
        print("MAGICBOKEH_VIDEO_E2E_ALREADY_COMPLETE", flush=True)
        return
    from measure_efficiency import assert_idle_gpu, gpu_processes, CoreMeter
    assert_idle_gpu()
    out.mkdir(parents=True, exist_ok=False)
    write(out / "protocol.json", dict(common=common, magic=protocol, frozen=frozen))

    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from baselines.magicbokeh_focus import visible_target_alpha, safe_target_region, medoid_point
    from run_magicbokeh_track_a import load_models
    cv2.setNumThreads(2)
    depth_model, adapter = load_models()
    meter = CoreMeter()
    meter.wrap(depth_model.model, "forward", "DAdepth_network_forward")
    manifest = read(ROOT / common["input"] / "config.json")
    indices = {scene["id"]: i for i, scene in enumerate(manifest["scenes"])}
    runtime = dict(torch=torch.__version__, CUDA=torch.version.cuda,
        gpu_hardware=subprocess.check_output(["nvidia-smi", "--id=0", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], text=True).strip(),
        cpu_model=next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")),
        cpu_quota=Path("/sys/fs/cgroup/cpu.max").read_text().strip(),
        memory_quota_bytes=Path("/sys/fs/cgroup/memory.max").read_text().strip(),
        torch_threads=torch.get_num_threads(), cv2_threads=cv2.getNumThreads(),
        torch_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        native_loader_and_precision_unchanged=True,
        project_commit=subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        author_commit=subprocess.check_output(["git", "-C", str(ROOT.parent / "MagicBokeh"), "rev-parse", "HEAD"], text=True).strip())
    write(out / "runtime.json", runtime)
    samples, errors, rows, warmups = [], [], [], []
    stop = threading.Event()
    page_bytes = os.sysconf("SC_PAGE_SIZE")

    def memory_sample():
        return dict(gpu_MiB=int(subprocess.check_output(["nvidia-smi", "--id=0", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True).strip()),
            rss_MiB=int(Path("/proc/self/statm").read_text().split()[1]) * page_bytes / 2**20)

    def monitor():
        while not stop.is_set():
            try:
                samples.append(memory_sample())
            except Exception as error:
                errors.append(str(error))
                return
            stop.wait(protocol["memory_sample_seconds"])

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        for sid, K in zip(common["scenes"], common["K"]):
            folder = ROOT / common["input"] / sid
            for repeat in range(-1, 5):
                assert set(gpu_processes()) <= {os.getpid()}, "Foreign GPU process appeared"
                meter.reset()
                memory_start = len(samples)
                meter.sync()
                start = time.perf_counter()
                metadata = read(folder / "metadata.json")
                assert metadata["source"]["K"] == K and len(metadata["frames"]) == 25
                pixels, frame_checks = [], []
                with torch.no_grad():
                    for frame in metadata["frames"]:
                        t = frame["index"]
                        meter.begin_unit()
                        image = Image.open(folder / "rgb" / f"{t:05d}.png").convert("RGB")
                        d = depth_model.infer(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR))
                        assert d.shape == (576, 1024) and np.isfinite(d).all() and d.min() >= 0 and d.max() <= 1
                        q = (d * 255).astype(np.uint8)
                        quantized = Image.fromarray(q, mode="L")
                        alpha, identity = visible_target_alpha(folder, metadata, frame)
                        region, distance, target = safe_target_region(alpha)
                        assert region is not None, "Frozen B target unavailable; no new fallback"
                        point, _ = medoid_point(region, distance, q)
                        focus = float(q[point[1], point[0]] / 255)
                        prepared = adapter.prepare(image, quantized, focus, K)
                        assert len(prepared["positions"]) == 6
                        seed = 42 + 25 * indices[sid] + t
                        torch.manual_seed(seed)
                        value = meter.call("MagicBokeh_six_tiles_and_GPU_reconstruction", adapter.forward_cuda, prepared, prompt=frozen["prompt"])
                        assert value.shape == (1, 3, 576, 1024) and torch.isfinite(value).all()
                        output = adapter.to_pil(value, image)
                        pixels.append(np.asarray(output, dtype=np.uint8))
                        meter.end_unit(1)
                        frame_checks.append(dict(frame=t, seed=seed, point_xy=point, focus=focus,
                            target_source_id=identity, target_alpha_threshold=target["alpha_threshold"], native_tiles=6,
                            native_depth_finite=True, generator_finite=True))
                        del value, prepared, image, d, quantized, alpha, region, distance, output
                meter.sync()
                seconds = time.perf_counter() - start
                assert len(pixels) == 25 and all(p.shape == (576, 1024, 3) and p.dtype == np.uint8 and np.isfinite(p).all() and p.max() > p.min() for p in pixels)
                assert len(meter.units) == 25 and 0 < meter.seconds <= seconds
                assert meter.calls["DAdepth_network_forward"] == 25
                assert meter.calls["MagicBokeh_six_tiles_and_GPU_reconstruction"] == 25
                assert not errors, errors
                observed = samples[memory_start:] or [memory_sample()]
                row = dict(scene=sid, K=K, repeat=repeat, output_frames=25,
                    core_seconds=meter.seconds, E2E_given_inputs_seconds=seconds,
                    core_regions_seconds=dict(meter.regions), core_region_calls=dict(meter.calls), native_units=meter.units,
                    peak_torch_allocated_GiB=max(x["torch_peak_allocated_GiB"] for x in meter.units),
                    peak_torch_reserved_GiB=max(x["torch_peak_reserved_GiB"] for x in meter.units),
                    peak_device_memory_MiB_sampled=max(x["gpu_MiB"] for x in observed),
                    peak_process_RSS_MiB_sampled=max(x["rss_MiB"] for x in observed),
                    output_validation="25 CPU uint8 RGB; original shape and finite guards passed", frame_checks=frame_checks)
                if repeat < 0:
                    warmups.append(row)
                    print("FULL_CLIP_WARMUP_CHECK_PASSED magicbokeh", sid, flush=True)
                else:
                    rows.append(row)
                    write(out / f"{sid}-repeat{repeat}.json", row)
                print("MAGICBOKEH_VIDEO_E2E", sid, repeat, round(meter.seconds, 4), round(seconds, 4), flush=True)
                del pixels
        assert set(gpu_processes()) <= {os.getpid()}
    finally:
        stop.set()
        thread.join()
    write(out / "warmups.json", warmups)
    summary = dict(method="magicbokeh", status="MAGICBOKEH_B_VIDEO_E2E_COMPLETE",
        completed_UTC=datetime.now(timezone.utc).isoformat(), protocol=protocol, runtime=runtime, rows=rows,
        Core_seconds_mean=float(np.mean([r["core_seconds"] for r in rows])),
        Core_seconds_std=float(np.std([r["core_seconds"] for r in rows], ddof=1)),
        E2E_given_inputs_seconds_mean=float(np.mean([r["E2E_given_inputs_seconds"] for r in rows])),
        E2E_given_inputs_seconds_std=float(np.std([r["E2E_given_inputs_seconds"] for r in rows], ddof=1)),
        FPS_E2E=25 / float(np.mean([r["E2E_given_inputs_seconds"] for r in rows])),
        peak_device_memory_MiB_sampled=max(r["peak_device_memory_MiB_sampled"] for r in rows),
        peak_process_RSS_MiB_sampled=max(r["peak_process_RSS_MiB_sampled"] for r in rows),
        memory_sampling_seconds=protocol["memory_sample_seconds"], frames_per_execution=25,
        by_K={str(K):{key:dict(mean=float(np.mean([r[key] for r in rows if r["K"] == K])),
            std=float(np.std([r[key] for r in rows if r["K"] == K], ddof=1)))
            for key in ["core_seconds", "E2E_given_inputs_seconds"]} for K in common["K"]})
    write(out / "summary.json", summary)
    print("MAGICBOKEH_B_VIDEO_E2E_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
