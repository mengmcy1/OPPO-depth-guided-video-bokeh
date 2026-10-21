"""Check weight-free FlashReFocus interfaces on CPU; never construct or run a model.

Selected function definitions are compiled from the pinned official source. This
deliberately avoids package import, which eagerly imports PiSA-SR / Diffusers.
These checks do not establish full dependency compatibility or model correctness.
"""
import argparse
import ast
import datetime
import json
import os
from pathlib import Path
import platform
import subprocess

# Keep this check independent of GPU availability and of the main ATB environment.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
from PIL import Image
import PIL
import torch
import torch.nn.functional as F


def functions_from_source(root, relative, names):
    path = root / relative
    tree = ast.parse(path.read_text(), filename=str(path))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes} != set(names):
        raise ValueError(f"Missing function definitions in {relative}")
    namespace = {"np": np, "Image": Image, "torch": torch, "F": F}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace, {n.name: f"{relative}:{n.lineno}" for n in nodes}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--official-repo", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="new JSON, never overwritten")
    a = ap.parse_args()
    root = a.official_repo.resolve()
    if a.out.exists():
        raise FileExistsError(a.out)
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    compiled = sorted((root / "flashrefocus").rglob("*.py")) + [root / "scripts/refocus.py", root / "scripts/benchmark_cost.py"]
    for path in compiled:
        compile(path.read_text(), str(path), "exec")
    io, io_lines = functions_from_source(root, "flashrefocus/image_io.py", [
        "image_to_tensor", "tensor_to_uint8", "to_signed", "to_unit_range", "resize_to_multiple"])
    dep, dep_lines = functions_from_source(root, "flashrefocus/rendering/depth.py", ["disparity_to_depth"])
    render, render_lines = functions_from_source(root, "flashrefocus/rendering/renderer.py", [
        "constant_map", "resize_bilinear", "condition_maps"])
    edits, edits_lines = functions_from_source(root, "flashrefocus/rendering/depth_edits.py", ["depth_at"])
    arr = (np.arange(37 * 53 * 3) % 256).astype(np.uint8).reshape(37, 53, 3)
    x = io["image_to_tensor"](Image.fromarray(arr))
    assert x.shape == (1, 3, 37, 53) and x.dtype == torch.float32
    assert np.array_equal(io["tensor_to_uint8"](x), arr)
    assert io["image_to_tensor"](arr, multiple=8).shape == (1, 3, 32, 48)
    fixed = io["resize_to_multiple"](Image.new("RGB", (1024, 576)), 1024)
    assert fixed.size == (1024, 576)
    assert io["resize_to_multiple"](Image.new("RGB", (1537, 1001)), 1024).size == (1024, 656)
    d = torch.tensor([[[0.2, 0.5, 0.8]]], dtype=torch.float32)
    native = dep["disparity_to_depth"](d)
    assert torch.allclose(native, torch.tensor([[[1.0, 0.5, 0.0]]]))
    constant = dep["disparity_to_depth"](torch.ones(1, 2, 3))
    assert torch.equal(constant, torch.ones_like(constant))
    assert torch.isfinite(dep["disparity_to_depth"](torch.tensor([[[float("nan"), float("inf"), 1.0]]]))).all()
    # Proposed shared-coordinate adapter algebra only; no learned renderer runs.
    external_D, external_F = 1 - d, 1 - torch.tensor(0.5)
    assert torch.allclose(external_D - external_F, -(d - 0.5))
    assert torch.allclose((external_D - external_F).abs(), (d - 0.5).abs())
    grid = torch.linspace(0, 1, 48).reshape(1, 1, 6, 8)
    signed, cond = render["condition_maps"](grid, torch.tensor([[0.5]]), torch.tensor([[0.375]]), (6, 8))
    assert cond.shape == (1, 3, 6, 8)
    assert torch.equal(signed, grid - 0.5)
    assert torch.equal(cond[:, 1:2], F.max_pool2d((signed < 1e-3).float(), 5, 1, 2))
    assert torch.equal(cond[:, 2:3], torch.full_like(grid, 0.375))
    assert edits["depth_at"](torch.arange(12).reshape(1, 3, 4), 1.6, 0.4) == 2
    assert edits["depth_at"](torch.arange(12).reshape(1, 3, 4), -5, 10) == 8
    result = dict(
        checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), official_commit=commit,
        status="CPU_SOURCE_FUNCTION_CHECKS_PASSED_NOT_MODEL_INFERENCE",
        environment=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__, pillow=PIL.__version__,
                         torch_cuda_build=torch.version.cuda, cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"]),
        compiled_first_party_files=len(compiled),
        function_locations={**io_lines, **dep_lines, **render_lines, **edits_lines},
        checks=dict(RGB_uint8_roundtrip=True, RGB_float32_shape=True, crop_multiple=True, fixed_1024x576_unchanged=True,
                    resize_LANCZOS_multiple16=True, native_depth_direction=True, native_constant_depth_is_one=True,
                    native_nonfinite_sanitization=True, proposed_shared_depth_focus_algebra=True,
                    native_condition_signed_mask_dilation_strength=True, focal_pixel_round_and_clamp=True),
        models_constructed=0, neural_network_calls=0, weight_loads=0,
        limits=["Selected original functions only; no full flashrefocus package import or GPU dependency verification.",
                "No official example, deblurring=False inference, performance or visual-quality validation.",
                "Shared-coordinate conversion remains an adapter proposal; strength mapping is uncalibrated."])
    with a.out.open("x") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
