"""Check and summarize the completed frozen MagicBokeh200-video reproduction."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path("03_实验记录/records/magicbokeh-track-a-200-20260930.json"))
    args = parser.parse_args()
    frozen = json.loads((ROOT / "configs/magicbokeh-track-a-frozen.json").read_text())
    out = ROOT / frozen["output"]
    assert (out / "inference-complete.json").is_file() and (out / "metrics-complete.json").is_file()
    source = ROOT / frozen["input"]
    scenes = json.loads((source / "config.json").read_text())["scenes"]
    ids = [s["id"] for s in scenes]
    scene_map = {s["id"]:s for s in scenes}
    batches = sorted((out / "metrics").glob("batch_*.json"))
    assert [p.name for p in batches] == [f"batch_{b:02d}.json" for b in range(20)]
    rows = [r for p in batches for r in json.loads(p.read_text())["rows"]]
    assert [r["scene"] for r in rows] == ids and len(rows) == 200
    keys = ["PSNR_TorchMetrics", "SSIM_TorchMetrics", "LPIPS", "FD", "RM", "PSNR", "PSNR_clip_mse", "SSIM", "RM_author_public_2transitions"]
    for row in rows:
        row["K"] = scene_map[row["scene"]]["K"]
        assert all(np.isfinite(row[k]) for k in keys)
    mean = {k:float(np.mean([r[k] for r in rows])) for k in keys}
    by_K = {str(K):dict(n=sum(r["K"] == K for r in rows),
        **{k:float(np.mean([r[k] for r in rows if r["K"] == K])) for k in keys}) for K in [4, 8, 16, 32]}
    assert all(abs(sum(v[k]*v["n"] for v in by_K.values()) / 200 - mean[k]) < 1e-12 for k in keys)
    distributions = {}
    for phase, filename, shape in [("fvd", "fvd-videomae-features.npz", (200, 1408)), ("vfid", "vfid-i-features.npz", (400, 1024))]:
        report = json.loads((out / "author-public" / (phase + ".json")).read_text())
        assert report["video_count"] == 200 and report["protocol"]["scene_ids"] == ids
        features = np.load(out / "author-public" / filename)
        reference = np.load(ROOT / "outputs/author-public-metrics-20260928" / filename)
        assert list(features["scene_ids"]) == ids and features["pred"].shape == features["gt"].shape == shape
        assert np.array_equal(features["gt"], reference["gt"]) and np.isfinite(features["pred"]).all()
        assert all(np.isfinite(x) for x in report["values"].values())
        distributions[phase] = {k:v for k,v in report.items() if k not in ["protocol", "checkpoint"]}
    audits = []
    for index, sid in enumerate(ids):
        renders = sorted((out / "clips" / sid / "pred_magicbokeh").glob("*.png"))
        assert [p.name for p in renders] == [f"{t:05d}.png" for t in range(25)]
        assert json.loads((out / "timing" / sid / "complete.json").read_text())["frames"] == 25
        for t, path in enumerate(renders):
            with Image.open(path) as image:
                assert image.mode == "RGB" and image.size == (1024, 576)
                image.verify()
            row = json.loads((out / "timing" / sid / f"{t:05d}.json").read_text())
            assert row["finite"] and row["inside_visible_target"] and row["native_tiles"] == 6
            assert row["seed"] == 42 + 25*index + t and row["K"] == scene_map[sid]["K"]
            assert row["target"]["alpha_threshold"] == .9
            native = np.load(out / "geometry" / sid / f"{t:05d}.npy")
            assert native.shape == (576, 1024) and np.isfinite(native).all()
            x, y = row["click_xy"]
            pixel = int((native * 255).astype(np.uint8)[y, x])
            assert abs(pixel / 255 - row["focus"]) < 1e-12
            audits.append(row)
        print("VERIFIED_MAGICBOKEH_CLIP", sid, flush=True)
    assert len(audits) == 5000 and sum(r["reused_preflight_B"] for r in audits) == 40
    control_by_K = {str(K):dict(frames=sum(r["K"] == K for r in audits),
        **{key:dict(mean=float(np.mean([r[key] for r in audits if r["K"] == K])),
                    min=float(np.min([r[key] for r in audits if r["K"] == K])),
                    max=float(np.max([r[key] for r in audits if r["K"] == K])))
           for key in ["target_focus_coverage", "focus_leakage", "boundary_margin_pixels"]}) for K in [4, 8, 16, 32]}
    result = dict(status="MAGICBOKEH_FROZEN_B_200x25_COMPLETE_AND_VERIFIED", completed_UTC=datetime.now(timezone.utc).isoformat(),
        protocol=frozen, frames=5000, scenes=200, mean=mean, by_K=by_K,
        Table1=dict(PSNR=mean["PSNR_TorchMetrics"], SSIM=mean["SSIM_TorchMetrics"], LPIPS=mean["LPIPS"],
                    FVD=distributions["fvd"]["values"]["FVD_VideoMAE"], FD=mean["FD"], RM=mean["RM"]),
        supplementary_VFID_I=distributions["vfid"]["values"]["VFID_I"], distributions=distributions,
        control_by_K=control_by_K, rows=rows,
        verification=dict(all5000PNG_counts_sizes_modes_integrity=True, all_native_outputs_finite=True,
            all_target_clicks_valid=True, all_focus_scalars_read_from_own_depth_at_click=True,
            per_frame_RNG_matches_frozen_protocol=True, no_alpha_fallback=True,
            reused_preflight_B_frames=40, all_metrics_finite=True, identical_GT_feature_cache=True, weighted_means_match=True),
        limits=["Public-proxy reproduction development, not held-out or official author test set.",
                "GT visible target alpha and own predicted depth assist the selected click; input permissions differ from external-geometry methods.",
                "K is nominal manifest K with nativeK/32; no physical CoC equivalence claimed.",
                "Native depth variation and partial-target defocus are retained; no threshold or strength compensation.",
                "Frame-wise model, no temporal smoothing or propagation; separated preflight frames do not prove no flicker."],
        artifacts=dict(output=frozen["output"], logs="logs/magicbokeh-point-focus-20260930",
                       preflight_record="03_实验记录/records/magicbokeh-point-focus-preflight-20260930.json"))
    target = ROOT / args.report
    with target.open("x") as file:
        json.dump(result, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\n")
    print(json.dumps(result["Table1"], indent=2), flush=True)


if __name__ == "__main__":
    main()
