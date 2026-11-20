"""Native depth plus frozen renderer adapters for the still-image baseline rows."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import cv2
import numpy as np

from route3_v2data_baseline_dev3 import (
    AUXILIARY, FOCUSES, FRAMES, ONE_C, OUT, ROOT, SCENES, SHAPE, STOPS, case_id, center_rgb,
    central_square, coc_slope, frame_truth, guard, model_canvas, p1_control, prediction_path,
    read, write,
)

sys.path.insert(0, str(ROOT / "src"))
CONFIGS = {
    "bokehme": "bokehme-track-a-frozen.json", "mpib": "mpib-track-a-frozen.json",
    "drbokeh": "drbokeh-track-a-frozen.json", "deeplens": "deeplens-track-a-frozen.json",
    "magicbokeh": "magicbokeh-track-a-frozen.json", "bokehdiff": "bokehdiff-adapter-recovery.json",
}
REPOSITORIES = dict(bokehme="BokehMe", mpib="MPIB", drbokeh="DrBokeh", deeplens="DeepLens",
                    magicbokeh="MagicBokeh", bokehdiff="BokehDiff")
DEPTHS = dict(bokehme="MiDaS v2.1 384", mpib="DPT-Large MiDaS checkpoint",
              drbokeh="Author DPT-Large and LDF alpha", deeplens="Official DeepLens DepthNet",
              magicbokeh="Official DAdepth ViT-L", bokehdiff="Official Depth-Anything-V2-Base-hf")


def unit_range(disparity):
    disparity = disparity.astype(np.float64)
    low, high = float(disparity.min()), float(disparity.max())
    if not np.isfinite(disparity).all() or high <= low:
        raise ValueError("Invalid native relative disparity")
    return (disparity - low) / (high - low)


def q8(value):
    return np.uint8(np.rint(np.clip(value, 0, 1) * 255))


def save(path, rgb, audit):
    if rgb.shape != (512, 512, 3) or not np.isfinite(rgb).all():
        raise ValueError("Image shape or finiteness check failed")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".part")
    with temp.open("wb") as stream:
        np.savez_compressed(stream, prediction_srgb=rgb.astype(np.float16),
                            audit=json.dumps(audit, allow_nan=False))
    temp.replace(path)


def load_dpt():
    from contextlib import contextmanager

    @contextmanager
    def inside(directory):
        cwd = os.getcwd()
        os.chdir(directory)
        try:
            yield
        finally:
            os.chdir(cwd)

    path = ROOT.parent / "DrBokeh"
    sys.path.insert(0, str(path / "app/Render"))
    sys.path.insert(0, str(path / "app/Render/Depth/DPT"))
    with inside(path):
        from depth_predict import Depth_Inference
        model = Depth_Inference()
    return model.inference


def load_midas():
    import torch
    import torch.nn.functional as F
    source = ROOT.parent / "route3-cache/midas-v2_1"
    checkpoint = ROOT / "models/MiDaS/midas_v21_384.pt"
    if not (source / "hubconf.py").is_file() or not checkpoint.is_file():
        raise FileNotFoundError("Official MiDaS v2.1 code or weights unavailable")
    from unittest.mock import patch
    from torchvision.models import resnext101_32x8d
    sys.path.insert(0, str(source))
    from midas import blocks
    from midas.midas_net import MidasNet
    # The complete published MiDaS checkpoint overwrites every initialized tensor.
    # Construct the same WSL ResNeXt architecture locally to avoid a redundant pretrained download.
    with patch.object(blocks, "_make_pretrained_resnext101_wsl",
                      lambda unused: blocks._make_resnet_backbone(resnext101_32x8d(weights=None))):
        model = MidasNet()
    model.load_state_dict(torch.load(str(checkpoint), weights_only=True, map_location="cpu"), strict=True)
    model = model.cuda().eval()
    transform = torch.hub.load(str(source), "transforms", source="local").default_transform

    def infer(image):
        value = transform(q8(image)).cuda()
        with torch.inference_mode():
            result = F.interpolate(model(value)[:, None], size=image.shape[:2],
                                   mode="bicubic", align_corners=False)
        return result[0, 0].cpu().numpy()

    return infer


def renderer(method, repo, config, arm):
    if method == "deeplens":
        from baselines.deeplens_native import DeepLensNative
        model = DeepLensNative(repo, ROOT / config["checkpoint"], threads=4)

        def depth(rgb):
            high = np.ascontiguousarray(rgb[..., ::-1])
            low = cv2.resize(high, (512, 288), interpolation=cv2.INTER_AREA)
            value = model.predict_depth(low)
            return cv2.resize(value, SHAPE[::-1]), (low, high, value)

        def render(rgb, d, focus, slope, state, subject, seed):
            low, high, small = state
            region = cv2.resize(model_canvas(subject.astype(np.uint8)), small.shape[::-1],
                                interpolation=cv2.INTER_NEAREST) > 0
            region[:, :112] = region[:, -112:] = False
            cost = np.where(region, np.abs(small - focus), np.inf)
            if not np.isfinite(cost).any():
                raise ValueError("No native DeepLens focus point on the target")
            y, x = np.unravel_index(cost.argmin(), cost.shape)
            strength = min(slope / config["response_scale"], 1.)
            rgb_bgr, _ = model.predict(low, high, int(x), int(y), strength, small)
            return q8(rgb_bgr[..., ::-1]), dict(aperture_parameter=strength,
                            saturation=bool(strength == 1), focus_used=float(small[y, x]), focus_xy=[int(x), int(y)])

        return depth, render

    import torch

    if method == "bokehme":
        from baselines.bokehme import setup, defocus_input, predict
        pipeline, args, nets, _ = setup(repo, ROOT / "models/BokehMe", config)

        def render(rgb, d, focus, slope, state, subject, seed):
            K = slope * np.ptp(d)
            control, _ = defocus_input(d.astype(np.float32), focus, K, "normalized", args.defocus_scale)
            value, _ = predict(pipeline, args, nets, rgb, control.astype(np.float32))
            return q8(value), dict(K=float(K), focus_used=focus)

        infer = load_midas() if arm == "native" else None
    elif method == "mpib":
        from run_mpib_recovery import setup, predict_layers
        nets, _ = setup()

        def render(rgb, d, focus, slope, state, subject, seed):
            with torch.inference_mode():
                layers, alpha = predict_layers(rgb, d.astype(np.float32), slope, nets)
                value = nets["renderer"](layers, alpha, slope, focus, config["gamma_candidates"][0],
                                         norm=config["norm"], increase_focus_range=config["increase_focus_range"])
            return q8(value[0].permute(1, 2, 0).cpu().numpy()), dict(K=slope, focus_used=focus)

        infer = load_dpt() if arm == "native" else None
    elif method == "drbokeh":
        from baselines.drbokeh_cuda import load_grid2d
        # The author module imports this legacy name before the frozen adapter replaces it.
        sys.modules["scatter_cuda"] = load_grid2d()
        from baselines.drbokeh import OfficialDrBokeh, map_geometry
        model = OfficialDrBokeh(repo, config)

        def render(rgb, d, focus, slope, state, subject, seed):
            disp, f, k, _ = map_geometry(d, focus, slope)
            value, audit, _ = model.predict(rgb, disp, f, k)
            if audit["calls"]["LDF"] != 1 or audit["calls"]["MiDaS"] != 0:
                raise ValueError("Dr.Bokeh alpha/input audit failed")
            tail = max(v["radial_tail_mass_at_max"] for v in audit["actual_layers"].values())
            return q8(value), dict(K=float(k), focus_used=focus, LDF_alpha=True, max_omitted_radial_mass=tail)

        infer = model.prep.depth_predict if arm == "native" else None
    elif method == "magicbokeh":
        from PIL import Image
        from run_magicbokeh_track_a import load_models
        from baselines.magicbokeh_focus import safe_target_region
        estimator, model = load_models()

        def infer(rgb):
            return estimator.infer(cv2.cvtColor(q8(rgb), cv2.COLOR_RGB2BGR))

        def render(rgb, d, focus, slope, state, subject, seed):
            from route3_v2data_baseline_dev3 import PAD
            source = Image.fromarray(q8(rgb))
            q = np.uint8(np.clip(d, 0, 1) * 255)
            selected = model_canvas(subject.astype(np.uint8)).astype(bool)
            selected[:, :PAD] = selected[:, -PAD:] = False
            region, distance, _ = safe_target_region(selected.astype(np.float32))
            if region is None:
                raise ValueError("No target support for MagicBokeh point focus")
            yy, xx = np.nonzero(region)
            index = np.lexsort((xx, yy, -distance[yy, xx], np.abs(q[yy, xx] / 255. - focus)))[0]
            y, x = int(yy[index]), int(xx[index])
            focal = float(q[y, x] / 255)
            with torch.inference_mode():
                torch.manual_seed(seed)
                prepared = model.prepare(source, Image.fromarray(q), focal, slope)
                result = model.forward_cuda(prepared, prompt=config["prompt"])
            if not torch.isfinite(result).all():
                raise ValueError("Nonfinite MagicBokeh output")
            return np.asarray(model.to_pil(result, source)), dict(K=slope, focus_xy=[x, y], focus_used=focal, seed=seed)
    else:
        from accelerate.utils import set_seed
        from transformers import pipeline
        from PIL import Image
        from baselines.bokehdiff import OfficialBokehDiff, map_controls
        set_seed(42)
        model = OfficialBokehDiff(repo, ROOT / "models/BokehDiff")
        checkpoint = ROOT / "models/BokehDiff/Depth-Anything-V2-Base-hf"
        if not (checkpoint / "model.safetensors").exists():
            raise FileNotFoundError("Official BokehDiff Depth-Anything-V2-Base-hf weights")
        estimator = pipeline("depth-estimation", model=str(checkpoint), device=0)

        def infer(rgb):
            # Raw inverse depth from the published estimator in the frozen synthetic near-is-large convention.
            return estimator(Image.fromarray(q8(rgb)))["predicted_depth"].squeeze().cpu().numpy()

        def render(rgb, d, focus, slope, state, subject, seed):
            K = slope * np.ptp(d)
            disp, f, _ = map_controls(d, focus, K, "normalized")
            result, audit = model.predict(rgb, disp, f, K, 42, reset_global_seed=False, quantization="round_png8")
            return np.asarray(result), dict(K=float(K), focus_used=focus, seed=audit["seed"])

    def depth(rgb):
        value = infer(rgb)
        if value.shape != SHAPE:
            value = cv2.resize(value, SHAPE[::-1], interpolation=cv2.INTER_CUBIC)
        if method != "magicbokeh":
            value = q8(unit_range(value)).astype(np.float64) / 255
        else:
            value = np.uint8(value * 255).astype(np.float64) / 255
        return value, None

    return depth, render


def toy_calibration(method, render):
    from calibrate_deeplens_control import edge_response
    folder = OUT / "calibration" / method
    record = folder / "control.json"
    if record.exists():
        return read(record)["radius_per_native_strength"]
    h, w = SHAPE
    image = np.full((h, w, 3), .25, np.float32)
    image[:, w // 2:] = .75
    disparity = np.full((h, w), .25, np.float64)
    disparity[:64, :64] = .75
    disparity[-64:, :64] = 0
    disparity[-64:, -64:] = 1
    subject = np.zeros((512, 512), bool)
    # A patch in the central square defines the in-focus plane outside the measured edge.
    subject[0:48, :48] = True
    disparity[:64, 224:300] = .75
    rows = []
    for control in (4., 8., 16., 32.):
        value, audit = render(image, disparity, .75, control, None, subject, 42)
        measured = edge_response(value.astype(np.float64) / 255, "vertical")
        rows.append(dict(strength=control, radius_px=float(measured["edge_10_90_width_px"] / 1.374097652),
                         ringing=measured["negative_lsf_mass_ratio"], **audit))
    x = np.array([r["strength"] * .5 for r in rows])
    y = np.array([r["radius_px"] for r in rows])
    scale = float(x @ y / (x @ x))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Independent planar edge calibration has no positive response")
    write(record, dict(scene="independent two-plane edge", rows=rows, radius_per_native_strength=scale,
                       monotonic=bool(np.all(np.diff(y) > 0)),
                       rmse_px=float(np.sqrt(np.mean((y - scale * x)**2)))))
    return scale


def run(method, arm, smoke):
    config_path = ROOT / "configs" / CONFIGS[method]
    config = read(config_path)
    repo = ROOT.parent / REPOSITORIES[method]
    revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if not revision.startswith(config["author_commit"]):
        raise ValueError(f"{method} author revision differs from frozen config")
    record = OUT / "inventory" / f"{method}-{arm}.json"
    inventory = dict(method=method, arm=arm, author_commit=revision,
                     config=str(config_path.relative_to(ROOT)), config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
                     depth=DEPTHS[method] if arm == "native" else "1c aligned 5k EMA visible inverse depth",
                     deviations=["Scale the square field to 576px, reflect-pad left/right to frozen 1024x576 and crop for scoring"])
    if method == "drbokeh":
        inventory["deviations"] += ["Official RGB-only LDF alpha in place of the historical Synthetic GT-alpha branch",
                                    "CUDA 12.8 sm_120 build of the frozen grid2d extension"]
    if method == "deeplens":
        inventory["deviations"].append("Official DepthNet replaces the historical external disparity, with single SR and frozen saturation")
    if method in ("magicbokeh", "deeplens"):
        inventory["deviations"].append("Native click is the target pixel closest to the fitted P1 disparity")
    if method == "bokehdiff":
        inventory["deviations"].append("Use official DAv2 inverse disparity in the frozen Synthetic near-is-large direction")
    if method in ("bokehme", "mpib"):
        inventory["deviations"].append("Officially documented estimator family with an available pinned checkpoint variant")
    inventory["device"] = "CPU (TF1)" if method == "deeplens" else "RTX 5090"
    gpu = method != "deeplens"
    if gpu:
        import torch
        torch.set_num_threads(4)
        torch.manual_seed(0)
        torch.cuda.set_per_process_memory_fraction(24 * 2**30 / torch.cuda.get_device_properties(0).total_memory)
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = torch.backends.cudnn.benchmark = False
    cv2.setNumThreads(2)
    failures = []
    completed = 0
    try:
        infer, render = renderer(method, repo, config, arm)
        response = toy_calibration(method, render) if method in ("magicbokeh", "bokehdiff") else 1.
        if method in ("magicbokeh", "bokehdiff"):
            inventory["calibration"] = read(OUT / "calibration" / method / "control.json")
            if method == "bokehdiff":
                import torch
                # Calibration runs independently of the ordered evaluation RNG stream.
                from accelerate.utils import set_seed
                set_seed(42)
        for scene_index, scene in enumerate(SCENES[:1] if smoke else SCENES):
            for frame in (FRAMES[:1] if smoke else FRAMES):
                guard(24 * 2**20)
                rgb = q8(model_canvas(center_rgb(scene, frame))).astype(np.float32) / 255
                truth, intrinsic, _, masks = frame_truth(scene, frame)
                try:
                    if gpu:
                        torch.cuda.reset_peak_memory_stats()
                        torch.cuda.synchronize()
                    tic = time.monotonic()
                    if arm == "native":
                        disparity, state = infer(rgb)
                    else:
                        with np.load(ONE_C / "C3" / scene / f"f{frame:03d}/visible-depths.npz") as archive:
                            depth = archive["1c-aligned-EMA"]
                        inverse = np.divide(1., depth, out=np.zeros_like(depth), where=depth > 0)
                        disparity, state = model_canvas(unit_range(inverse)), None
                    if gpu:
                        torch.cuda.synchronize()
                    depth_time = time.monotonic() - tic
                    depth_peak = torch.cuda.max_memory_allocated() / 2**30 if gpu else None
                    depth_reserved = torch.cuda.max_memory_reserved() / 2**30 if gpu else None
                except Exception as error:
                    failures.append(dict(scene=scene, frame=frame, reason=str(error)))
                    print("DEPTH_FAILED", method, scene, frame, repr(error), flush=True)
                    continue
                for stop in (STOPS[:1] if smoke else STOPS):
                    for nominal in (FOCUSES[:1] if smoke else FOCUSES):
                        cid = case_id(frame, stop, nominal)
                        destination = (OUT / "smoke" / f"{method}-{arm}.npz" if smoke else
                                       prediction_path(method, arm, "P1", scene, cid))
                        existing = destination.exists()
                        if existing and method != "bokehdiff":
                            completed += 1
                            continue
                        try:
                            focus, fit = p1_control(disparity, truth, masks["common_visible_background"], nominal)
                            strength = coc_slope(intrinsic, stop, nominal) * SHAPE[0] / 512 / abs(fit["a"]) / response
                            subject = masks["subject_near" if nominal == 1.4 else "subject_far"]
                            if gpu:
                                torch.cuda.reset_peak_memory_stats()
                                torch.cuda.synchronize()
                            tic = time.monotonic()
                            seed = 42 + 60 * scene_index + frame - 1
                            result, audit = render(rgb, disparity, focus, strength, state, subject, seed)
                            if gpu:
                                torch.cuda.synchronize()
                            used = audit["focus_used"]
                            inverse_focus = (used - fit["b"]) / fit["a"]
                            audit.update(render_s=time.monotonic() - tic,
                                depth_s=depth_time if arm == "native" else None,
                                peak_allocated_gib=max(depth_peak, torch.cuda.max_memory_allocated() / 2**30) if gpu else None,
                                peak_reserved_gib=max(depth_reserved, torch.cuda.max_memory_reserved() / 2**30) if gpu else None,
                                p1_fit_a=fit["a"], p1_fit_b=fit["b"], p1_holdout_rmse=fit["heldout_rmse"],
                                expected_focus_disparity=focus, focus_disparity=used,
                                focus_distance_m=1. / inverse_focus if inverse_focus > 0 else None)
                            output = central_square(result.astype(np.float32) / 255)
                            if existing:
                                with np.load(destination) as previous:
                                    if not np.array_equal(output.astype(np.float16), previous["prediction_srgb"]):
                                        raise ValueError("Ordered BokehDiff RNG replay changed a saved result")
                            else:
                                save(destination, output, audit)
                            completed += 1
                            print("BASELINE", method, arm, scene, cid, audit["render_s"], flush=True)
                        except Exception as error:
                            if existing and method == "bokehdiff":
                                raise
                            failures.append(dict(scene=scene, case=cid, reason=str(error)))
                            print("BASELINE_FAILED", method, arm, scene, cid, repr(error), flush=True)
    except Exception as error:
        failures.append(dict(reason=str(error)))
        raise
    finally:
        inventory.update(stage="smoke" if smoke else "full", completed=completed, failures=failures)
        write(record, inventory)
    if not completed:
        raise RuntimeError(f"{method} {arm} completed zero cases")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=CONFIGS)
    parser.add_argument("--arm", choices=("native", "1c-depth"), default="native")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.arm != "native" and args.method not in AUXILIARY:
        parser.error("This method has no 1c-depth auxiliary row")
    run(args.method, args.arm, args.smoke)
