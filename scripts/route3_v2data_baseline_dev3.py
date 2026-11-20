"""Run and score the frozen Route 3 baselines on the original nine development frames."""
import argparse
from collections import defaultdict
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "route3-baseline-dev3-2026-11-20"
OUT = ROOT / "outputs" / (PREFIX + "-01")
RECORDS = ROOT / "03_实验记录/records"
PREVIOUS = ROOT / "outputs/route3-g3-r15-2026-11-17-01"
ONE_C = ROOT / "outputs/route3-ovg-1c-2026-11-19-01"
REFOCUS = ROOT / "outputs/route3-ovg-refocus-2026-11-18-01"
LOCK = ROOT.parent / "route3-cache/gpu-data.lock"
SCENES = ("metric-scene-122104-broad", "metric-scene-122105-multilayer",
          "metric-scene-122107-broad")
FRAMES = (1, 30, 60)
STOPS = (1.4, 5.)
FOCUSES = (1.4, 5.)
METHODS = ("1c-query-f2", "1c-visible-f2", "A-color", "bokehme", "mpib", "drbokeh",
           "any2bokeh", "magicbokeh", "bokehdiff", "deeplens")
AUXILIARY = ("bokehme", "mpib", "drbokeh")
REGIONS = ("full_image", "near_in_focus", "near_recoverable", "far_recoverable", "boundary_2px")
SHAPE = (576, 1024)
PAD = (SHAPE[1] - SHAPE[0]) // 2


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".part")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    temporary.replace(path)


def guard(batch_bytes):
    if shutil.disk_usage(OUT).free < 60 * 2**30 + batch_bytes:
        raise OSError("Insufficient space for this batch plus 60 GiB")


def case_id(frame, stop, focus):
    return f"f{frame:03d}-N{stop:g}-F{focus:g}"


def scene_root(scene):
    return PREVIOUS / "truth" / scene


def read_exr(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    return image[..., :3][..., ::-1].copy()


def display(linear):
    return np.where(linear <= .0031308, 12.92 * linear,
                    1.055 * np.maximum(linear, 0) ** (1 / 2.4) - .055).clip(0, 1).astype(np.float32)


def center_rgb(scene, frame):
    return display(read_exr(scene_root(scene) / "inputs" / f"f{frame:03d}.exr"))


def model_canvas(value):
    side = SHAPE[0]
    return cv2.copyMakeBorder(cv2.resize(value, (side, side), interpolation=cv2.INTER_LINEAR),
                              0, 0, PAD, PAD, cv2.BORDER_REFLECT_101)


def central_square(value):
    return cv2.resize(value[:, PAD:PAD + SHAPE[0]], (512, 512), interpolation=cv2.INTER_AREA)


def frame_truth(scene, frame):
    with np.load(scene_root(scene) / "geometry" / f"f{frame:03d}-center.npz") as label:
        depth, intrinsic, valid = (label[k] for k in ("depth_axial_m", "K", "center_visible"))
    with np.load(REFOCUS / scene / f"f{frame:03d}/R1-masks.npz") as masks:
        values = {k: masks[k] for k in ("common_visible_background", "subject_near", "subject_far")}
    return depth, intrinsic, valid, values


def p1_control(disparity, truth, bg, focus):
    d = central_square(disparity).astype(np.float64)
    good = bg & np.isfinite(d) & np.isfinite(truth) & (truth > 0)
    y, x = np.indices(d.shape)
    held = good & ((x // 8 + y // 8) % 5 == 0)
    train = good & ~held
    if train.sum() < 1000 or held.sum() < 100:
        raise ValueError("Insufficient common-background samples for P1")
    inverse_truth = 1. / truth[train].astype(np.float64)
    if np.ptp(np.quantile(inverse_truth, [.05, .95])) < .005:
        raise ValueError("P1 reference inverse-depth span below .005")
    X = np.column_stack((inverse_truth, np.ones_like(inverse_truth)))
    values = d[train]
    ab = np.linalg.lstsq(X, values, rcond=None)[0]
    for _ in range(12):
        residual = X @ ab - values
        delta = max(.002, 1.345 * 1.4826 * np.median(abs(residual - np.median(residual))))
        weights = np.sqrt(np.minimum(1., delta / np.maximum(abs(residual), 1e-12)))
        ab = np.linalg.lstsq(X * weights[:, None], values * weights, rcond=None)[0]
    slope, intercept = (float(v) for v in ab)
    if not np.isfinite(ab).all() or abs(slope) <= 1e-6:
        raise ValueError(f"P1 degenerate disparity slope: {slope}")
    residual = slope / truth[held] + intercept - d[held]
    return slope / focus + intercept, dict(a=slope, b=intercept, train=int(train.sum()),
            heldout=int(held.sum()), heldout_rmse=float(np.sqrt(np.mean(residual**2))))


def coc_slope(intrinsic, stop, focus):
    lens = .05
    # Radius in 512px image coordinates per unit inverse meter.
    return float(intrinsic[0, 0]) * lens / (2 * stop) * focus / (focus - lens)


def prediction_path(method, arm, protocol, scene, cid):
    return OUT / "predictions" / method / arm / protocol / scene / f"{cid}.npz"


def reference_plan(scene, frame):
    from route3_pilot_plan import disk
    folder = OUT / "references" / scene
    blend = scene_root(scene) / "scene.blend"
    rows = []
    for stop in STOPS:
        for focus in FOCUSES:
            cid = case_id(frame, stop, focus)
            for index, xy in enumerate(disk(64, .05 / (2 * stop))):
                key = f"{cid}-k{index:02}"
                rows.append(dict(scene=scene, id=key, blend=str(blend), size=512,
                                 frame=frame, spp=64, seed=20261112, dof=False, focus=focus,
                                 fstop=stop, aperture_xy=xy.tolist(), omit_reserved_passes=True,
                                 output=str(folder / "subviews" / (key + ".exr"))))
    return rows


def references():
    from route3_ovg_subpixel import save_npz

    blender = ROOT.parent / "tools/blender-4.5.13-linux-x64/blender"
    keep_cycles_data = """import bpy
from bpy.app.handlers import persistent
@persistent
def keep_cycles_data(scene):
    scene.render.use_persistent_data = True
bpy.app.handlers.render_pre.append(keep_cycles_data)
"""
    for scene in SCENES:
        for frame in FRAMES:
            folder = OUT / "references" / scene
            paths = [folder / (case_id(frame, stop, focus) + ".npz")
                     for stop in STOPS for focus in FOCUSES]
            if all(path.is_file() for path in paths):
                continue
            jobs = reference_plan(scene, frame)
            plan = OUT / "references" / f"{scene}-f{frame:03d}.json"
            if plan.exists():
                if read(plan) != jobs:
                    raise ValueError(f"Reference plan changed for {scene} {frame}")
            else:
                write(plan, jobs)
            sample = next((p for p in (OUT / "references").glob("*/subviews/*.exr")), None)
            guard(256 * (sample.stat().st_size if sample else 512 * 512 * 16))
            log = OUT / "logs" / f"reference-{scene[13:19]}-f{frame:03d}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("a") as stream:
                result = subprocess.run(["flock", str(LOCK), "nice", "-n", "10", str(blender), "-b",
                                         "-t", "4", "--python-exit-code", "1",
                                         "--python-expr", keep_cycles_data, "--python",
                                         str(ROOT / "scripts/route3_s2a_blender.py"), "--", "--jobs", str(plan)],
                                        stdout=stream, stderr=subprocess.STDOUT,
                                        env=dict(os.environ, CUDA_VISIBLE_DEVICES="0"), check=False)
            if result.returncode:
                print("REFERENCE_FAILED", scene, frame, result.returncode, flush=True)
                continue
            for path in paths:
                if path.exists():
                    continue
                group = [Path(job["output"]) for job in jobs if job["id"].startswith(path.stem + "-")]
                if len(group) != 64 or not all(p.is_file() for p in group):
                    raise ValueError(f"Incomplete subviews for {path}")
                ldr = np.zeros((512, 512, 3), dtype=np.float64)
                hdr = np.zeros_like(ldr)
                for subview in group:
                    rgb = read_exr(subview)
                    if not np.isfinite(rgb).all():
                        raise ValueError(f"Nonfinite Cycles subview {subview}")
                    ldr += rgb.clip(0, 1)
                    hdr += rgb
                save_npz(path, reference_srgb=display(ldr / 64), linear_ldr=(ldr / 64).astype(np.float32),
                         linear_hdr=(hdr / 64).astype(np.float32))
                print("REFERENCE", scene, path.stem, flush=True)


def clean_subviews():
    """Release the reproducible EXRs once all four integrated references exist."""
    for scene in SCENES:
        folder = OUT / "references" / scene
        subviews = folder / "subviews"
        for frame in FRAMES:
            for stop in STOPS:
                for focus in FOCUSES:
                    cid = case_id(frame, stop, focus)
                    if not (folder / (cid + ".npz")).exists():
                        continue
                    for index in range(64):
                        (subviews / f"{cid}-k{index:02d}.exr").unlink(missing_ok=True)


def aperture_labels():
    from route3_blender_b2_full_geometry import FullWorld
    from route3_ovg_subpixel import save_npz
    for scene in SCENES:
        world = FullWorld(scene_root(scene) / "scene-geometry.npz", 512)
        for frame in FRAMES:
            for focus in FOCUSES:
                cid = case_id(frame, 5., focus)
                path = OUT / "labels" / scene / (cid + "-K16.npz")
                if path.exists():
                    continue
                values, _ = world.aperture_full(frame, 16, 5., focus)
                save_npz(path, mask_recoverable_total=values["mask_recoverable_total"])
                print("K16_LABELS", scene, cid, flush=True)


def evaluation_masks(scene, frame, stop, focus, truth, intrinsic, valid):
    from src.dual_layer import depth_band
    cid = case_id(frame, stop, focus)
    folder = scene_root(scene) / "geometry" if stop == 1.4 else OUT / "labels" / scene
    with np.load(folder / (cid + "-K16.npz")) as archive:
        recoverable = archive["mask_recoverable_total"]
    band, _, coc = depth_band(np.where(valid, truth, 1e4), intrinsic[0, 0], focus, .05 / (2 * stop))
    label = ROOT / "outputs/route3-ovg-1b-2026-11-18-01/B" / scene
    with np.load(label / "pinhole-subpixels" / f"f{frame:03d}.npz") as archive:
        edge = cv2.dilate(archive["depth_boundary"].astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    masks = {"full_image": np.ones(truth.shape, bool), "boundary_2px": edge}
    masks["near_recoverable" if focus == 1.4 else "far_recoverable"] = band & recoverable
    if focus == 1.4:
        masks["near_in_focus"] = valid & (coc <= 1)
    return masks


def csv_rows(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def report():
    import torch
    from route3_ldr_u1r import model
    from route3_upper_bound_evaluate import score, pool

    torch.set_num_threads(4)
    os.environ.setdefault("TORCH_HOME", str(ROOT.parent / ".cache/torch"))
    metric = model()
    rows, controls, incomplete = [], [], []
    reference_count = 0
    for scene in SCENES:
        for frame in FRAMES:
            truth, intrinsic, valid, _ = frame_truth(scene, frame)
            for stop in STOPS:
                for focus in FOCUSES:
                    cid = case_id(frame, stop, focus)
                    ref = OUT / "references" / scene / (cid + ".npz")
                    masks = None
                    if ref.exists():
                        reference_count += 1
                        with np.load(ref) as data:
                            target = data["reference_srgb"]
                        try:
                            masks = evaluation_masks(scene, frame, stop, focus, truth, intrinsic, valid)
                        except FileNotFoundError:
                            pass
                    for method in METHODS:
                        arms = ("native", "1c-depth") if method in AUXILIARY else ("native",)
                        protocols = ("P1", "P2") if method in METHODS[:3] else ("P1",)
                        for arm in arms:
                            for protocol in protocols:
                                path = prediction_path(method, arm, protocol, scene, cid)
                                base = dict(scene=scene, frame=frame, aperture=stop, nominal_focus_m=focus,
                                            method=method, arm=arm, protocol=protocol)
                                if not path.is_file():
                                    incomplete.append(base)
                                    continue
                                with np.load(path) as data:
                                    prediction = data["prediction_srgb"].astype(np.float32)
                                    audit = json.loads(data["audit"].item())
                                if prediction.shape != (512, 512, 3) or not np.isfinite(prediction).all():
                                    raise ValueError(f"Invalid prediction {path}")
                                controls.append(dict(base, **audit))
                                if masks is not None:
                                    values = score(prediction, target, masks, metric)
                                    rows.extend(dict(base, region=region, **value)
                                                for region, value in values.items())
            print("SCORED", scene, frame, len(rows), flush=True)
    pooled = defaultdict(list)
    for row in rows:
        for scene in (row["scene"], "pooled"):
            key = (scene, row["method"], row["arm"], row["protocol"], row["aperture"], row["region"])
            pooled[key].append(row)
    table = []
    for (scene, method, arm, protocol, aperture, region), cells in sorted(pooled.items()):
        expected = (9 if scene == "pooled" else 3) * (2 if region in ("full_image", "boundary_2px") else 1)
        table.append(dict(scene=scene, method=method, arm=arm, protocol=protocol, aperture=aperture,
                          region=region, cases=len(cells), expected_cases=expected,
                          complete=len(cells) == expected, **pool(cells)))
    resource_file = OUT / "logs/resource-peaks.json"
    resources = read(resource_file) if resource_file.exists() else {}
    runtime = []
    for method in METHODS:
        for arm in (("native", "1c-depth") if method in AUXILIARY else ("native",)):
            records = [r for r in controls if r["method"] == method and r["arm"] == arm]
            render = [r["render_s"] for r in records if r.get("render_s") is not None]
            depths = {(r["scene"], r["frame"]): r.get("depth_s") for r in records
                      if r.get("depth_s") is not None}
            peak = [r.get("renderer_and_depth_peak_gib", r.get("peak_allocated_gib")) for r in records]
            reserved = [r.get("peak_reserved_gib") for r in records]
            sample_prefix = "ours-" if method in METHODS[:3] else (f"{method}-{arm}-" if method != "any2bokeh" else "any2bokeh-")
            samples = [r for name, r in resources.items() if name.startswith(sample_prefix) and "full" in name]
            runtime.append(dict(method=method, arm=arm, device="TF1 CPU on RTX 5090 host" if method == "deeplens" else "RTX 5090",
                                timed_render_cases=len(render), render_mean_s=float(np.mean(render)) if render else None,
                                timed_depth_frames=len(depths),
                                depth_mean_s=float(np.mean(list(depths.values()))) if depths else None,
                                peak_allocated_gib=max((v for v in peak if v is not None), default=None),
                                peak_reserved_gib=max((v for v in reserved if v is not None), default=None),
                                sampled_vram_mib=max((r["sampled_gpu_MiB"] for r in samples), default=None),
                                sampled_peak_RSS_GiB=max((r["process_peak_RSS_KiB"] / 2**20 for r in samples), default=None)))
    indexed = {(r["method"], r["aperture"], r["region"]): r for r in table
               if r["scene"] == "pooled" and r["protocol"] == "P1" and r["arm"] == "native"}
    main_table = []
    for method in METHODS:
        for aperture in STOPS:
            row = dict(method=method, aperture=aperture)
            for region in REGIONS:
                entry = indexed.get((method, aperture, region), {})
                row.update({f"{region}_{key}": entry.get(key) for key in ("psnr", "patch_lpips", "sse", "pixels", "cases")})
            main_table.append(row)
    auxiliary = [r for r in table if r["protocol"] == "P2" or r["arm"] != "native"]
    for tag, values in (("main", main_table), ("by-scene", [r for r in table if r["scene"] != "pooled"]),
                        ("auxiliary", auxiliary), ("per-frame", rows), ("focus", controls), ("runtime", runtime)):
        csv_rows(RECORDS / f"{PREFIX}-{tag}.csv", values)
    inventory = [read(p) for p in sorted((OUT / "inventory").glob("*.json"))]
    case_coverage = []
    for row in runtime:
        method, arm = row["method"], row["arm"]
        actual = [r for r in controls if r["method"] == method and r["arm"] == arm]
        expected = 72 if method in METHODS[:3] else 36
        case_coverage.append(dict(method=method, arm=arm, completed=len(actual), expected=expected,
                                  missing=expected - len(actual),
                                  negative_p1_fit_cases=sum(r.get("p1_fit_a", 0) < 0 for r in actual if r["protocol"] == "P1"),
                                  saturated_aperture_cases=sum(r.get("saturation", False) for r in actual),
                                  maximum_drbokeh_omitted_radial_mass=max(
                                      (r["max_omitted_radial_mass"] for r in actual if "max_omitted_radial_mass" in r),
                                      default=None)))
    by_scene_disk = {scene: 0 for scene in SCENES}
    for p in OUT.rglob("*"):
        if p.is_file() and not p.is_symlink():
            for scene in SCENES:
                if scene in str(p.relative_to(OUT)):
                    by_scene_disk[scene] += p.stat().st_size
                    break
    summary = dict(scope="Original three development scenes, positioning only", training_run=False, scenes=SCENES, frames=FRAMES,
                   fstop=STOPS, focus_targets_m=FOCUSES, reference_count=reference_count, reference_target=36,
                   reference_blender="4.5.13, OptiX on RTX 5090",
                   reference_cache_check="Persistent Cycles geometry validated on 8 paired original subviews. Maximum linear RGB delta 4.18e-7.",
                   historical_apertures=[1.4, 2.0],
                   reference_deviation="Render both stops again under 13.15.5, extending the historical sweep with f/5",
                   split_counts={"train": 0, "development": 3, "test": 0},
                   case_coverage=case_coverage,
                   scene_output_bytes=by_scene_disk,
                   committed_data_assets=0, credentials_required=[],
                   additional_public_weights=[
                       dict(id="MiDaS v2.1 midas_v21_384.pt", authors="René Ranftl et al.",
                            source="https://github.com/isl-org/MiDaS/releases/tag/v2_1",
                            checkout="94afff70fe0873c3af70355e5388e4d482bfc807", source_license="MIT",
                            sha256="f6b980704cfd7259c7cc2b058c2f160159c55e84b3b6c08331b4156a84629f70"),
                       dict(id="Depth-Anything-V2-Base-hf", authors="Lihe Yang et al.",
                            source="https://huggingface.co/depth-anything/Depth-Anything-V2-Base-hf",
                            checkout="b1958afc87fb45a9e3746cb387596094de553ed8", model_license="CC BY-NC 4.0",
                            sha256="9996d15c334d264cfcc29f4318d7d3789a006ecc444782c2fb82f7eaa1acb8bf")],
                   gpu_lock="route3-cache/gpu-data.lock", gpu_limit_gib=24,
                   timing_scope="Render call with frozen inputs. Mean depth separately. Any-to-Bokeh clip render normalized by "
                                "all emitted frames. DeepLens native CPU graph repeats DepthNet inside render.",
                   memory_scope="PyTorch allocated and reserved peaks plus sampled driver VRAM where available",
                   formal_followup={"development": 15, "independent_test": 30},
                   ground_truth_role="Evaluator-only P1 background fit, target selection, metrics, and fixed masks",
                   p2_scope="Metric native depth only. Relative baseline and DeepLens DepthNet have no P2 row.",
                   radius_mapping="K_x * 0.05/(2N) * focus/(focus-0.05) / abs(P1 slope); scale to adapter canvas",
                   patch_lpips="Historical AlexNet LPIPS, 64px grid, pixel-support-weighted pooling",
                   aggregation="Sum RGB SSE and support before PSNR. Full and boundary pool both focus targets.",
                   precision="Float16 saved predictions, cast to float32 for scoring, float32 LDR/HDR references",
                   output_bytes=sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file() and not p.is_symlink()),
                   negative_p1_fit_cases=sum(1 for r in controls if r.get("p1_fit_a",0) < 0),
                   reference="Cycles 64x64 spp, clip each linear RGB subview to [0,1] before averaging, then sRGB",
                   configurations=inventory, runtime=runtime, job_resource_samples=resources, missing_predictions=incomplete,
                   completed_prediction_cases=len(controls), scored_region_rows=len(rows),
                   exclusions={"DoF-GS": "Defocused multiview input and per-scene optimization",
                               "DoF-Gaussian": "Defocused multiview input and per-scene optimization", "2b": "Out of scope"})
    write(RECORDS / f"{PREFIX}-summary.json", summary)
    write(OUT / "summary.json", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("references", "clean_subviews", "aperture_labels", "report"))
    globals()[parser.parse_args().operation]()
