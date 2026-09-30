"""Compare three fixed focus controls on40 frames; do not compute bokeh GT scores."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0"
    config = json.loads((ROOT / "configs/magicbokeh-point-focus.json").read_text())
    source = ROOT / config["input"]
    manifest = json.loads((source / "config.json").read_text())
    indices = {s["id"]:i for i,s in enumerate(manifest["scenes"])}
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    for name in ["inputs", "depth", "mask", "visuals", *config["candidates"]]:
        (out / name).mkdir()
    for sid in config["preflight_scenes"]:
        for t in config["preflight_frames"]:
            (out / "inputs" / f"{sid}-{t:05d}.png").symlink_to((source / sid / "rgb" / f"{t:05d}.png").resolve())
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    subprocess.run([sys.executable, "prepare_data.py", "--input_dir", str(out / "inputs"),
        "--depth_dir", str(out / "depth"), "--mask_dir", str(out / "mask"),
        "--resnet_path", str(ROOT / "models/DrBokeh/resnet50-19c8e357.pth"),
        "--model_path", str(ROOT / "models/DrBokeh/model-40"),
        "--depth_ckpt", str(ROOT / "models/MagicBokeh/DAdepth.pth")], cwd=ROOT.parent / "MagicBokeh", check=True)

    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    from baselines.magicbokeh import OfficialMagicBokeh
    from baselines.magicbokeh_focus import visible_target_alpha, safe_target_region, medoid_point, control_diagnostics
    torch.set_num_threads(4)
    torch.manual_seed(42)
    adapter = OfficialMagicBokeh(ROOT.parent / "MagicBokeh", ROOT / "models/MagicBokeh", ROOT / "models/sd21-base")
    rows = []

    def mark(image, points):
        image = image.convert("RGB").copy()
        draw = ImageDraw.Draw(image)
        for label, (x, y) in points.items():
            color = "red" if label == "A" else "lime"
            draw.ellipse((x-7, y-7, x+7, y+7), outline=color, width=3)
            draw.text((x+9, y-9), label, fill=color)
        return image

    def panel(canvas, image, column, row, label):
        image = image.convert("RGB").resize((320, 180))
        canvas.paste(image, (column * 320, row * 204 + 24))
        ImageDraw.Draw(canvas).text((column * 320 + 5, row * 204 + 6), label, fill="black")

    with torch.no_grad():
        for sid in config["preflight_scenes"]:
            metadata = json.loads((source / sid / "metadata.json").read_text())
            K = metadata["source"]["K"]
            contact = Image.new("RGB", (9 * 320, len(config["preflight_frames"]) * 204), "white")
            for display_row, t in enumerate(config["preflight_frames"]):
                name = f"{sid}-{t:05d}"
                image = Image.open(out / "inputs" / (name + ".png")).convert("RGB")
                d = np.load(out / "depth" / (name + ".npy"))
                if image.size != (1024, 576) or d.shape != (576, 1024) or not np.isfinite(d).all():
                    raise ValueError("Invalid native preflight depth/RGB")
                q = (d * 255).astype(np.uint8)
                depth = Image.fromarray(q, mode="L")
                alpha, identity = visible_target_alpha(source / sid, metadata, metadata["frames"][t])
                region, distance, target_info = safe_target_region(alpha)
                points, controls = {}, {}
                ldf = Image.open(out / "mask" / (name + ".png")).convert("L")
                controls["auto_ldf"] = dict(focus=adapter.module.adjust_depth_with_mask(depth, ldf) / 255,
                    click_xy=None, inside_visible_target=None, boundary_margin_pixels=None)
                if region is not None:
                    points["A"] = target_info["point_A_xy"]
                    points["B"], medoid = medoid_point(region, distance, q)
                    for variant, label in [("interior_A", "A"), ("medoid_B", "B")]:
                        x, y = points[label]
                        controls[variant] = dict(focus=float(q[y, x] / 255), click_xy=[x, y],
                            inside_visible_target=bool(alpha[y, x] > target_info["alpha_threshold"]),
                            boundary_margin_pixels=float(distance[y, x]), magic_depth_at_click_png8=int(q[y, x]))
                    controls["medoid_B"].update(medoid)
                row = dict(scene=sid, frame=t, K=K, target_source_id=identity, target=target_info,
                    seed=42 + 25 * indices[sid] + t, candidates={})
                panel(contact, mark(image, points), 0, display_row, f"t{t} RGB A:red B:green")
                panel(contact, Image.fromarray(np.uint8(np.round(alpha * 255))), 1, display_row, "Visible target alpha")
                panel(contact, mark(depth, points), 2, display_row, "Native DA depth and clicks")
                for column, variant in enumerate(config["candidates"]):
                    if variant not in controls:
                        panel(contact, Image.new("RGB", image.size, "white"), 3+column, display_row, "focus_point_unavailable")
                        continue
                    control = controls[variant]
                    prepared = adapter.prepare(image, depth, control["focus"], K)
                    full_depth = adapter.module.tensor_transforms(depth).unsqueeze(0).cuda()
                    focused = 1 - ((full_depth - control["focus"]).abs() * (K / 32) > .1).float()
                    native_focused = focused[0, 0].bool().cpu().numpy()
                    for tile, (y, x) in zip(prepared["focused"], prepared["positions"]):
                        if not np.array_equal((tile[0, 0].cpu().numpy() > 0), native_focused[y:y+512, x:x+512]):
                            raise ValueError("Diagnostic/native focused-mask mismatch")
                    torch.manual_seed(row["seed"])
                    result = adapter.forward_cuda(prepared, prompt=config["prompt"])
                    if result.shape != (1, 3, 576, 1024) or not torch.isfinite(result).all():
                        raise ValueError("Invalid native point-focus output")
                    output = adapter.to_pil(result, image)
                    output.save(out / variant / (name + ".png"))
                    pixels = np.asarray(output)
                    diagnostics = control_diagnostics(alpha, q, control["focus"], K, native_focused=native_focused)
                    row["candidates"][variant] = dict(**control, **diagnostics, finite=True,
                        RGB8_range=[int(pixels.min()), int(pixels.max())], RGB8_std=float(pixels.std()),
                        native_tiles=len(prepared["positions"]), output_size=list(output.size))
                    panel(contact, Image.fromarray(native_focused.astype(np.uint8) * 255), 3+column, display_row,
                        f"{variant} focus={control['focus']:.4f}")
                    panel(contact, output, 6+column, display_row, variant + " output")
                    del prepared, result, full_depth, focused
                rows.append(row)
                (out / "visuals" / (name + ".json")).write_text(json.dumps(row, indent=2) + "\n")
                print("MAGIC_FOCUS_PREFLIGHT", sid, t, {k:round(v["target_focus_coverage"], 4) if v["target_focus_coverage"] is not None else None for k,v in row["candidates"].items()}, flush=True)
            contact.save(out / "visuals" / (sid + "-contact.jpg"), quality=95)
    by_K = {}
    for K in [4, 8, 16, 32]:
        by_K[str(K)] = {}
        for variant in config["candidates"]:
            selected = [r["candidates"][variant] for r in rows if r["K"] == K and variant in r["candidates"]]
            by_K[str(K)][variant] = {}
            for key in ["target_focus_coverage", "focus_leakage", "boundary_margin_pixels"]:
                values = [v[key] for v in selected if v[key] is not None]
                by_K[str(K)][variant][key] = None if not values else dict(mean=float(np.mean(values)), min=float(np.min(values)), max=float(np.max(values)))
    summary = dict(status="THREE_CANDIDATE_PREFLIGHT_COMPLETE_AWAITING_VISUAL_REVIEW", protocol=config,
        frames=len(rows), generated_outputs=sum(len(r["candidates"]) for r in rows), rows=rows, by_K=by_K,
        unavailable=[dict(scene=r["scene"], frame=r["frame"]) for r in rows if r["target"]["status"] != "AVAILABLE"],
        metrics_selection="No PSNR, SSIM, LPIPS, FVD, FD or RM computed; no bokeh GT read",
        scope="GT visible target alpha only chooses one control point; generator RGB and native predicted geometry retained; no temporal smoothing",
        torch=torch.__version__, device=torch.cuda.get_device_name(0))
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(by_K, indent=2), flush=True)


if __name__ == "__main__":
    main()
