"""Minimal gsplat checks, then one static CamVFX frame at several K values."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.aperture_renderer import GaussianScene, render_aperture, srgb_to_linear, linear_to_srgb
from src.aperture_renderer import observability_statistics, disocclusion_mask
from src.aperture_metrics import regional_metrics
from src.camvfx4d import effect_parameters, read_rgb


def synthetic_checks():
    device = "cuda:0"
    K = torch.tensor([[80., 0, 32], [0, 80., 32], [0, 0, 1.]], device=device)
    eye = torch.eye(4, device=device)
    scene = GaussianScene(torch.tensor([[.1, 0, 4.], [-.3, .1, 2.]], device=device, requires_grad=True),
                          torch.tensor([[1., 0, 0, 0]] * 2, device=device, requires_grad=True),
                          torch.tensor([[.03, .03, .0001]] * 2, device=device, requires_grad=True),
                          torch.tensor([.9, .9], device=device, requires_grad=True),
                          torch.tensor([[.8, .3, .2], [.2, .6, .9]], device=device, requires_grad=True),
                          torch.tensor([0, 1], device=device))
    focus = torch.tensor(4., device=device, requires_grad=True)
    radius = torch.tensor(.08, device=device, requires_grad=True)
    output = render_aperture(scene, K, eye, 64, 64, focus, radius, 8)
    output["rgb_linear"].square().sum().backward()
    gradients = {k: v.grad is not None and bool(torch.isfinite(v.grad).all()) and float(v.grad.abs().max()) > 0 for k, v in
                 {"means": scene.means, "scales": scene.scales, "opacity": scene.opacities,
                  "rotation": scene.quats, "colors": scene.colors, "focus": focus, "radius": radius}.items()}
    assert all(gradients.values()), gradients
    focus_field = torch.full((64, 64), 4., device=device, requires_grad=True)
    field_output = render_aperture(scene, K, eye, 64, 64, focus_field, .08, 8)
    field_output["rgb_linear"].square().sum().backward()
    gradients["focus_field"] = bool(torch.isfinite(focus_field.grad).all()) and float(focus_field.grad.abs().max()) > 0
    assert gradients["focus_field"]
    with torch.no_grad():
        one = scene.select([0])
        a = render_aperture(one, K, eye, 64, 64, 4., .08, 1)
        b = render_aperture(one, K, eye, 64, 64, 4., .08, 8)
        focus_error = float((a["rgb_linear"] - b["rgb_linear"]).abs().max())
        assert focus_error < 1e-3, focus_error
        field = render_aperture(scene, K, eye, 64, 64, torch.full((64, 64), 4., device=device), .08, 8)
        scalar = render_aperture(scene, K, eye, 64, 64, 4., .08, 8)
        difference = float((field["rgb_linear"] - scalar["rgb_linear"]).abs().mean())
        assert difference < .005, difference
        assert bool(a["other_frame_weight"].eq(0).all())
        provenance_error = float((scalar["t_ref_weights"].sum(-1) - scalar["alpha"]).abs().max())
        assert provenance_error < 1e-4, provenance_error
        scaled = GaussianScene(scene.means * 10, scene.quats, scene.scales * 10,
                               scene.opacities, scene.colors, scene.source_frames)
        unit_scaled = render_aperture(scaled, K, eye, 64, 64, 40., .8, 8)
        unit_error = float((unit_scaled["rgb_linear"] - scalar["rgb_linear"]).abs().max())
        assert unit_error < 1e-4, unit_error
    return {"finite_nonzero_gradients": gradients, "focus_plane_invariance_max_error": focus_error,
            "scalar_constant_field_rgb_mae": difference, "provenance_alpha_max_error": provenance_error,
            "source_filter_passed": True, "world_unit_rescale_max_error": unit_error}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("geometry", type=Path)
    p.add_argument("scene", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--samples", nargs="+", type=int, default=[1, 4, 16, 64])
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    record = {"synthetic": synthetic_checks(), "static_camvfx": []}
    data = np.load(a.geometry)
    tensor = {k: torch.tensor(data[k], device="cuda:0") for k in
              ["means", "colors_srgb", "scales", "quats", "source_frames", "K", "world_to_camera", "reference_depth"]}
    scene = GaussianScene(tensor["means"], tensor["quats"], tensor["scales"],
                          torch.full((len(tensor["means"]),), .999, device="cuda:0"),
                          srgb_to_linear(tensor["colors_srgb"]), tensor["source_frames"])
    target = torch.tensor(read_rgb(a.scene / "dof/0/000001.png"), device="cuda:0")
    pinhole = torch.tensor(read_rgb(a.scene / "pinhole/0/000001.png"), device="cuda:0")
    params = effect_parameters(a.scene)[0]
    for count in a.samples:
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); start = time.monotonic()
        with torch.no_grad():
            rendered = render_aperture(scene, tensor["K"], tensor["world_to_camera"], 512, 512,
                                       params["focus_distance_m"], params["aperture_radius_m"], count,
                                       current_frame=0, return_samples=True)
            rgb = linear_to_srgb(rendered["rgb_linear"]).clamp(0, 1)
        torch.cuda.synchronize(); seconds = time.monotonic() - start
        row = {"K": count, "elapsed_seconds": seconds, "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
               "full": regional_metrics(rgb, target, torch.ones((512, 512), device="cuda:0", dtype=torch.bool)),
               "covered": regional_metrics(rgb, target, rendered["alpha"] > .95),
               "mean_alpha": float(rendered["alpha"].mean())}
        if count == 1:
            row["pinhole_initialization_reference"] = regional_metrics(rgb, pinhole, torch.ones((512, 512), device="cuda:0", dtype=torch.bool))
        Image.fromarray(np.uint8(rgb.cpu().numpy() * 255)).save(a.output / f"K{count}.png")
        record["static_camvfx"].append(row)
        print(json.dumps(row), flush=True)
        (a.output / "validation.json").write_text(json.dumps(record, indent=2) + "\n")
    record["scale"] = {"world_units": "dataset meters", "focus_distance_m": params["focus_distance_m"],
                       "aperture_radius_m": params["aperture_radius_m"],
                       "geometry": "provided mesh vertices with explicit Y reflection; no learned scale"}
    (a.output / "validation.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
