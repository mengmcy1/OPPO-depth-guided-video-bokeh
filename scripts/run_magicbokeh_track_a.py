"""Run the frozen native-depth, target-guided medoid adapter on200x25 frames."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as file:
        json.dump(value, file, indent=2, allow_nan=False)
        file.write("\n")


def load_models():
    import torch
    torch.set_num_threads(4)
    torch.manual_seed(42)
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    repository = ROOT.parent / "MagicBokeh"
    sys.path.insert(0, str(repository))
    from prepare_data import DADepthInference
    from baselines.magicbokeh import OfficialMagicBokeh
    depth = DADepthInference(ckpt_path=str(ROOT / "models/MagicBokeh/DAdepth.pth"))
    generator = OfficialMagicBokeh(repository, ROOT / "models/MagicBokeh", ROOT / "models/sd21-base")
    return depth, generator


def render_frame(folder, metadata, frame, image, d, index, adapter, config):
    import numpy as np
    import torch
    from PIL import Image
    from baselines.magicbokeh_focus import visible_target_alpha, safe_target_region, medoid_point, control_diagnostics
    if d.shape != (576, 1024) or not np.isfinite(d).all() or d.min() < 0 or d.max() > 1:
        raise ValueError("Invalid native predicted depth")
    q = (d * 255).astype(np.uint8)
    depth = Image.fromarray(q, mode="L")
    alpha, identity = visible_target_alpha(folder, metadata, frame)
    region, distance, target = safe_target_region(alpha)
    if region is None:
        raise ValueError("focus_point_unavailable; no new fallback is authorized")
    point, medoid = medoid_point(region, distance, q)
    x, y = point
    focus = float(q[y, x] / 255)
    K, t = metadata["source"]["K"], frame["index"]
    prepared = adapter.prepare(image, depth, focus, K)
    native_depth = adapter.module.tensor_transforms(depth).unsqueeze(0).cuda()
    focused = 1 - ((native_depth - focus).abs() * (K / 32) > .1).float()
    diagnostics = control_diagnostics(alpha, q, focus, K, native_focused=focused[0, 0].bool().cpu().numpy())
    torch.manual_seed(42 + 25 * index + t)
    with torch.no_grad():
        result = adapter.forward_cuda(prepared, prompt=config["prompt"])
    if result.shape != (1, 3, 576, 1024) or not torch.isfinite(result).all():
        raise ValueError("Invalid native frozen-adapter output")
    output = adapter.to_pil(result, image)
    audit = dict(frame=t, K=K, target_source_id=identity, target=target, point_B_xy=point, click_xy=point,
        inside_visible_target=bool(region[y, x]), boundary_margin_pixels=float(distance[y, x]),
        focus=focus, magic_depth_at_click_png8=int(q[y, x]), medoid=medoid, **diagnostics,
        seed=42 + 25 * index + t, native_tiles=len(prepared["positions"]), finite=True,
        output_size=list(output.size), depth_range=[float(d.min()), float(d.max())])
    return output, audit


def verify(config, output):
    import cv2
    import numpy as np
    from PIL import Image
    source = ROOT / config["input"]
    preflight = ROOT / config["preflight"]
    manifest = json.loads((source / "config.json").read_text())
    index = next(i for i, s in enumerate(manifest["scenes"]) if s["id"] == "repro_004")
    folder = source / "repro_004"
    metadata = json.loads((folder / "metadata.json").read_text())
    depth_model, adapter = load_models()
    image = Image.open(folder / "rgb/00000.png").convert("RGB")
    previous = np.load(preflight / "depth/repro_004-00000.npy")
    native = depth_model.infer(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR))
    result, audit = render_frame(folder, metadata, metadata["frames"][0], image, previous, index, adapter, config)
    reference = np.asarray(Image.open(preflight / "medoid_B/repro_004-00000.png").convert("RGB"))
    error = np.abs(np.asarray(result).astype(int) - reference.astype(int))
    if error.any():
        raise ValueError(f"Frozen runner replay differs from preflight: {error.max()}")
    report = dict(status="FROZEN_RUNNER_B_EQUALS_PREFLIGHT_RGB8", maximum_RGB8_difference=int(error.max()),
        native_depth_max_difference=float(np.max(np.abs(native - previous))),
        native_PNG8_different_pixels=int(((native * 255).astype(np.uint8) != (previous * 255).astype(np.uint8)).sum()),
        scope="Fresh depth+generator process; cached preflight conditioning replays exact B output; native depth inference also compared",
        frame=audit)
    output.mkdir(parents=True, exist_ok=False)
    result.save(output / "replayed-B.png")
    write(output / "result.json", report)
    print(json.dumps(report, indent=2), flush=True)


def worker(config, output, shard, workers):
    import cv2
    import numpy as np
    from PIL import Image
    cv2.setNumThreads(2)
    source = ROOT / config["input"]
    preflight = ROOT / config["preflight"]
    manifest = json.loads((source / "config.json").read_text())
    indices = {s["id"]: i for i, s in enumerate(manifest["scenes"])}
    chosen = manifest["scenes"][shard::workers]
    depth_model, adapter = load_models()
    start = time.monotonic()
    for scene in chosen:
        sid = scene["id"]
        folder = source / sid
        metadata = json.loads((folder / "metadata.json").read_text())
        renders = output / "clips" / sid / "pred_magicbokeh"
        renders.mkdir(parents=True, exist_ok=True)
        native_dir = output / "geometry" / sid
        native_dir.mkdir(parents=True, exist_ok=True)
        audits = []
        scene_start = time.monotonic()
        for frame in metadata["frames"]:
            t = frame["index"]
            record = output / "timing" / sid / f"{t:05d}.json"
            destination = renders / f"{t:05d}.png"
            native_file = native_dir / f"{t:05d}.npy"
            if record.is_file():
                if not destination.is_file() or not native_file.is_file():
                    raise ValueError("Incomplete saved frame; preserve and inspect")
                audits.append(json.loads(record.read_text()))
                continue
            name = f"{sid}-{t:05d}"
            cached = preflight / "medoid_B" / (name + ".png")
            if cached.is_file():
                if destination.exists() or native_file.exists():
                    raise FileExistsError("Incomplete cached-frame write; preserve and inspect")
                destination.symlink_to(cached)
                native_file.symlink_to(preflight / "depth" / (name + ".npy"))
                old = json.loads((preflight / "visuals" / (name + ".json")).read_text())
                row = dict(frame=t, K=scene["K"], target_source_id=old["target_source_id"], target=old["target"],
                    **old["candidates"]["medoid_B"], seed=old["seed"], reused_preflight_B=True)
            else:
                image = Image.open(folder / "rgb" / f"{t:05d}.png").convert("RGB")
                d = depth_model.infer(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR))
                result, row = render_frame(folder, metadata, frame, image, d, indices[sid], adapter, config)
                if destination.exists() or native_file.exists():
                    raise FileExistsError("Incomplete generated-frame write; preserve and inspect")
                np.save(native_file, d)
                result.save(destination)
                row["reused_preflight_B"] = False
            write(record, row)
            audits.append(row)
        if len(list(renders.glob("*.png"))) != 25 or len(audits) != 25:
            raise ValueError("Incomplete scene")
        marker = output / "timing" / sid / "complete.json"
        if not marker.exists():
            write(marker, dict(status="COMPLETE", scene=sid, K=scene["K"], frames=25,
                reused_frames=sum(r["reused_preflight_B"] for r in audits), seconds=time.monotonic()-scene_start))
        print("MAGICBOKEH_SCENE_COMPLETE", shard, sid, round(time.monotonic()-scene_start, 2), flush=True)
    print("MAGICBOKEH_SHARD_COMPLETE", shard, len(chosen), round(time.monotonic()-start, 2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--worker", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify-preflight", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    assert 1 <= args.workers <= 4, "This server protocol supports at most four workers"
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0"
    config = json.loads((ROOT / "configs/magicbokeh-track-a-frozen.json").read_text())
    assert config["status"] == "FROZEN_TARGET_GUIDED_DEPTH_MEDOID_POINT_FOCUS"
    out = (args.output_dir or ROOT / config["output"]).resolve()
    if args.verify_preflight:
        verify(config, out)
        return
    if args.worker is not None:
        worker(config, out, args.worker, args.workers)
        return
    from measure_efficiency import assert_idle_gpu
    assert_idle_gpu()
    out.mkdir(parents=True, exist_ok=args.resume)
    protocol = out / "protocol.json"
    if protocol.exists():
        assert json.loads(protocol.read_text()) == config
    else:
        write(protocol, config)
    logs = ROOT / "logs/magicbokeh-point-focus-20260930"
    logs.mkdir(parents=True, exist_ok=True)
    children, streams, jobs = [], [], []
    for shard in range(args.workers):
        log = logs / f"full-worker{shard}-{int(time.time())}.log"
        stream = log.open("x")
        child = subprocess.Popen([sys.executable, "-u", __file__, "--worker", str(shard),
            "--workers", str(args.workers), "--output-dir", str(out)], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        children.append(child)
        streams.append(stream)
        jobs.append(dict(shard=shard, pid=child.pid, log=str(log.relative_to(ROOT))))
    write(out / f"jobs-{int(time.time())}.json", dict(workers=args.workers, jobs=jobs))
    try:
        while any(child.poll() is None for child in children):
            failed = [child for child in children if child.poll() not in (None, 0)]
            if failed:
                raise RuntimeError("A worker failed; inspect logs and preserve outputs")
            time.sleep(5)
        if any(child.returncode != 0 for child in children):
            raise RuntimeError("Worker failure")
        complete = list((out / "timing").glob("*/complete.json"))
        if len(complete) != 200:
            raise ValueError("Not all200 scenes completed")
        write(out / "inference-complete.json", dict(status="ALL_200x25_INFERENCE_COMPLETE", scenes=200,
            frames=5000, workers=args.workers, reused_preflight_frames=40, protocol=config))
        print("ALL_MAGICBOKEH_200x25_COMPLETE", flush=True)
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            child.wait()
        for stream in streams:
            stream.close()


if __name__ == "__main__":
    main()
