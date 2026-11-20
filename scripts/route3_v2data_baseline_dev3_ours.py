"""Replay the frozen 1c aligned EMA, its visible-only pair, and A-color."""
import argparse
import json
import time

import numpy as np
import torch

import route3_ovg_data as data
from route3_g3_r14_train import learned_hidden
from route3_ovg_1b_common import CONFIG, SOURCE
from route3_ovg_1b_optical_data import background_samples
from route3_ovg_1b_prime_evaluate import A as ANCHOR
from route3_ovg_1b_train import load_model
from route3_ovg_1c import aligned_input
from route3_ovg_1c_evaluate import case_frames
from route3_ovg_footprint import query_net
from route3_ovg_refocus_query import cache_path, method_band
from route3_ovg_refocus_render import check_cache
from route3_ovg_subpixel import save_npz
from route3_s2a_learn import Store
from route3_v2data_baseline_dev3 import ONE_C, OUT, STOPS, case_id, display, guard, prediction_path, read, write
from src.aperture_renderer import aperture_disk
from src.footprint_order import apply_footprint_order
from src.layered_aperture import merge_layers
from src.ovg_optics import fixed_visible_surfels, raster

SELECTED = "1c-aligned-EMA"
CHECKPOINT = ONE_C / "train/aligned/checkpoints/step-005000.pt"


@torch.inference_mode()
def run(smoke):
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(24 * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    net = load_model(CHECKPOINT, "ema").eval()
    query = query_net(SOURCE / "V1/retrained/step-020000.pt")
    store = Store()
    replay_errors, historical_errors = [], []
    for scene, frame, entry, _, _, _, manifest, original, folder in case_frames():
        guard(32 * 2**20)
        frozen = {(r["method"], r["protocol"], r["id"]): r for r in read(folder / "focus.json")["rows"]}
        source_frame = data.TRAIN_FRAMES.index(frame)
        current = aligned_input(scene, frame)
        neighbors = [aligned_input(scene, data.TRAIN_FRAMES[k])
                     for k in data.neighbors(source_frame, len(data.TRAIN_FRAMES))]
        inputs = {k: torch.as_tensor(v[None].copy(), device="cuda")
                  for k, v in data.input_record(current, neighbors).items()}
        torch.cuda.synchronize()
        start = time.monotonic()
        depth = net(inputs, protect_residual=True, render_heads=False)["depth1"][0]
        torch.cuda.synchronize()
        depth_time = time.monotonic() - start
        with np.load(folder / "visible-depths.npz") as old:
            replay_difference = float(np.max(np.abs(depth.cpu().numpy() - old[SELECTED])))
            # Use the same cached geometry that defines the frozen Query observation pack.
            depth = torch.as_tensor(old[SELECTED], device="cuda")
        replay_errors.append(replay_difference)
        K = inputs["K"][0]
        cloud = fixed_visible_surfels(dict(depth1=depth[None]), inputs)
        source = dict(tiers="B", entry=entry)
        for old_item in original:
            old = old_item["case"]
            if old["fstop"] != 1.4 or (smoke and (old["protocol"] != "P1" or old["nominal_focus_m"] != 1.4)):
                continue
            protocol, nominal = old["protocol"], old["nominal_focus_m"]
            for stop in ((1.4,) if smoke else STOPS):
                cid = case_id(frame, stop, nominal)
                destination = OUT / "smoke" if smoke else OUT / "predictions"
                offsets = aperture_disk(CONFIG["native_eval_rays"], .05 / (2 * stop), device=K.device, dtype=K.dtype)
                for method in ("A-color", "1c-visible-f2", "1c-query-f2"):
                    dest = (destination / f"{method}.npz" if smoke else
                            prediction_path(method, "native", protocol, scene, cid))
                    if dest.exists():
                        continue
                    key = ANCHOR if method == "A-color" else SELECTED
                    plan = frozen[key, protocol, old["id"]]
                    if plan["失败"]:
                        continue
                    focus = plan["实际对焦距离_m"]
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                    start = time.monotonic()
                    fallback = background_samples(source, K, focus, offsets, (512, 512))
                    if method == "A-color":
                        rays = fallback
                    else:
                        visible = cloud
                        if method == "1c-query-f2":
                            band = method_band(depth.cpu().numpy(), K.cpu().numpy(),
                                               dict(plan, aperture=stop))
                            path = str(cache_path(folder, SELECTED))
                            check_cache(store, path, focus, stop, band, depth.cpu().numpy())
                            item = dict(frame=frame, input=path,
                                        case=dict(old, id=cid, focus_m=focus, fstop=stop))
                            hidden = learned_hidden(query, store, item, manifest, K, band)
                            hidden, _ = apply_footprint_order(hidden, depth, K)
                            visible = merge_layers(visible, hidden)
                        rays = raster(visible, K, focus, offsets, 512, 512, fallback)
                    rgb = display(rays.mean(0).cpu().numpy())
                    torch.cuda.synchronize()
                    elapsed = time.monotonic() - start
                    audit = dict(focus_disparity=1 / focus, focus_distance_m=focus, render_s=elapsed,
                                 depth_s=depth_time if method != "A-color" else None,
                                 peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                                 peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                                 replay_max_depth_difference_m=replay_difference,
                                 p1_fit_a=plan["P1拟合斜率"], p1_fit_b=plan["P1拟合截距_逆米"])
                    historical = folder / "render" / protocol / old["id"] / f"{SELECTED}-Query.npz"
                    if stop == 1.4 and method == "1c-query-f2" and historical.exists():
                        with np.load(historical) as archive:
                            error = float(np.max(np.abs(rgb - archive["prediction_srgb"])))
                        audit["original_render_max_abs_srgb"] = error
                        historical_errors.append(error)
                    save_npz(dest, prediction_srgb=rgb.astype(np.float16), audit=json.dumps(audit))
                    print("OURS", method, scene, protocol, cid, elapsed, flush=True)
                    del rays, fallback
        store.inputs.cache_clear()
        store.active.cache_clear()
        if smoke:
            break
    for method in ("A-color", "1c-visible-f2", "1c-query-f2"):
        write(OUT / "inventory" / f"{method}.json", dict(
            method=method, configuration="Route 3 §13.13, 1c C3 / §13.12 R2", smoke=smoke,
            checkpoint=str(CHECKPOINT.relative_to(OUT.parents[1])) if method != "A-color" else "existing SHARP and A-color",
            checkpoint_key="ema" if method != "A-color" else None,
            query_checkpoint="V1/retrained/step-020000.pt" if method == "1c-query-f2" else None,
            max_5090_replay_depth_difference_m=max(replay_errors),
            max_5090_vs_frozen_render_srgb=max(historical_errors, default=None) if method == "1c-query-f2" else None,
            deviations=["Extended existing 1c optical aperture sweep with f/5",
                        "Render timing starts at frozen input cache, excludes SHARP and other cached preprocessing"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args().smoke)
