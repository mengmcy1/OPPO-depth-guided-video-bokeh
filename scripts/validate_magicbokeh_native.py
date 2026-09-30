"""Validate full author images through the unchanged MagicBokeh CLI."""
import argparse
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent / "MagicBokeh"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--conditions", choices=["supplied", "predicted"], required=True)
    parser.add_argument("--names", nargs="+", default=["EBB_LR_1321", "EBB_LR_1365", "IMG_0256"])
    parser.add_argument("--K", type=int, default=32)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    inputs, depth, mask, renders = [out / name for name in ["inputs", "depth", "mask", "renders"]]
    for directory in [inputs, depth, mask, renders]:
        directory.mkdir()
    for name in args.names:
        matches = list((REPO / "test_data/inputs").glob(name + ".*"))
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one author input for {name}")
        (inputs / matches[0].name).symlink_to(matches[0])
        if args.conditions == "supplied":
            for target, source, ext in [(depth, "depth", "npy"), (mask, "mask", "png")]:
                path = REPO / f"test_data/{source}/{name}.{ext}"
                if not path.is_file():
                    raise FileNotFoundError(path)
                (target / path.name).symlink_to(path)

    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if args.conditions == "predicted":
        subprocess.run([
            sys.executable, "prepare_data.py", "--input_dir", str(inputs),
            "--depth_dir", str(depth), "--mask_dir", str(mask),
            "--resnet_path", str(ROOT / "models/DrBokeh/resnet50-19c8e357.pth"),
            "--model_path", str(ROOT / "models/DrBokeh/model-40"),
            "--depth_ckpt", str(ROOT / "models/MagicBokeh/DAdepth.pth"),
        ], cwd=REPO, check=True)

    import numpy as np
    import torch
    from PIL import Image
    from run_magicbokeh import configure_attention
    sys.path.insert(0, str(REPO))
    os.chdir(REPO)
    from magicbokeh import model_test
    from inference import adjust_depth_with_mask, extract_patches

    torch.set_num_threads(4)
    # The author CLI parses --seed but does not apply it; set it in this wrapper.
    torch.manual_seed(42)
    configure_attention()
    calls = []
    original_forward = model_test.forward

    def checked_forward(self, *inputs, **kwargs):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        result = original_forward(self, *inputs, **kwargs)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        if result.shape != inputs[0].shape or not torch.isfinite(result).all():
            raise ValueError("Invalid native tile output")
        calls.append(dict(shape=list(result.shape), seconds=seconds,
                          minimum=result.min().item(), maximum=result.max().item(),
                          peak_allocated_GiB=torch.cuda.max_memory_allocated() / 2**30,
                          peak_reserved_GiB=torch.cuda.max_memory_reserved() / 2**30))
        return result

    model_test.forward = checked_forward
    sys.argv = ["inference.py", "--input_dir", str(inputs), "--depth_dir", str(depth),
                "--mask_dir", str(mask), "--output_dir", str(renders), "--K", str(args.K),
                "--pretrained_model_name_or_path", str(ROOT / "models/sd21-base"),
                "--model_path", str(ROOT / "models/MagicBokeh/lora.pkl")]
    try:
        runpy.run_path(str(REPO / "inference.py"), run_name="__main__")
    finally:
        model_test.forward = original_forward

    images = []
    for path in sorted(inputs.iterdir()):
        image = Image.open(path).convert("RGB")
        result = Image.open(renders / (path.stem + ".png")).convert("RGB")
        if image.size != result.size:
            raise ValueError(f"Output size mismatch: {path.name}")
        d = np.load(depth / (path.stem + ".npy"))
        if not np.isfinite(d).all() or d.shape != (image.height, image.width):
            raise ValueError(f"Invalid depth: {path.name}")
        quantized = Image.fromarray((d * 255).astype(np.uint8), mode="L")
        m = Image.open(mask / (path.stem + ".png")).convert("L")
        focus = adjust_depth_with_mask(quantized, m) / 255
        _, positions = extract_patches(torch.zeros(1, 1, image.height, image.width))
        coverage = np.zeros((image.height, image.width), dtype=np.uint16)
        for y, x in positions:
            coverage[y:y+512, x:x+512] += 1
        if not (coverage > 0).all():
            raise ValueError(f"Uncovered image pixels: {path.name}")
        rgb = np.asarray(result)
        images.append(dict(name=path.stem, width=image.width, height=image.height,
                           native_tiles=len(positions), uncovered_pixels=int((coverage == 0).sum()),
                           focus=focus, depth_range=[float(d.min()), float(d.max())],
                           mask_nonzero_fraction=float((np.asarray(m) > 0).mean()),
                           rgb8_range=[int(rgb.min()), int(rgb.max())], rgb8_std=float(rgb.std())))
    if sum(x["native_tiles"] for x in images) != len(calls):
        raise ValueError("Native tile count mismatch")
    report = dict(status="FULL_NATIVE_IMAGES_VALIDATED", conditions=args.conditions, K=args.K,
                  seed=42, torch=torch.__version__, device=torch.cuda.get_device_name(0),
                  author_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                  images=images, native_calls=calls, all_tiles_finite=True,
                  timing_scope="Unwarmed native tile validation; not Table3 latency",
                  native_depth_source="Author-provided conditions" if args.conditions == "supplied" else "DAdepth.pth and LDF from RGB via author prepare_data.py",
                  color_postprocess="Unchanged author PIL conversion and adain_color_fix",
                  boundary="No paired bokeh GT; no quality metric or temporal claim")
    (out / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k:v for k,v in report.items() if k != "native_calls"}, indent=2))


if __name__ == "__main__":
    main()
