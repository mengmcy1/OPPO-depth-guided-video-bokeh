"""Check full-video timing records and retain measured current/old hardware tables."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
METHODS = ["deeplens", "drbokeh", "bokehme", "mpib", "bokehdiff", "magicbokeh", "any2bokeh"]


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    args = parser.parse_args()
    base = ROOT / args.output_dir
    assert read(base / "execution-complete.json")["status"] == "SEVEN_VIDEO_E2E_METHODS_COMPLETE"
    common = read(ROOT / "configs/efficiency-track-a.json")
    executed = read(base / "any2bokeh/protocol.json")
    for key in ["input", "scenes", "K", "resolution", "frames", "batch", "warmup_clips_per_scene", "repeats_per_scene", "core", "e2e", "rng", "core_regions"]:
        assert executed[key] == common[key], key
    source = ROOT / "03_实验记录/records/efficiency-track-a-3ab-20260930.json"
    old = read(source)
    table, methods, comparison = [], {}, []
    expected = {(sid, K, i) for sid, K in zip(common["scenes"], common["K"]) for i in range(5)}
    warm_expected = {(sid, K, -1) for sid, K in zip(common["scenes"], common["K"])}
    for method in METHODS:
        folder = base / method
        summary = read(folder / "summary.json")
        assert summary["status"] == ("MAGICBOKEH_B_VIDEO_E2E_COMPLETE" if method == "magicbokeh" else "STANDARDIZED_TIMING_COMPLETE")
        rows, warmups = summary["rows"], read(folder / "warmups.json")
        assert len(rows) == 20 and len(warmups) == 4
        assert {(r["scene"], r["K"], r["repeat"]) for r in rows} == expected
        assert {(r["scene"], r["K"], r["repeat"]) for r in warmups} == warm_expected
        for row in rows + warmups:
            assert row["output_frames"] == 25
            assert 0 < row["core_seconds"] <= row["E2E_given_inputs_seconds"]
            assert math.isfinite(row["core_seconds"]) and math.isfinite(row["E2E_given_inputs_seconds"])
            units = row["native_units"]
            assert [u["output_frames"] for u in units] == ([8, 8, 9] if method == "any2bokeh" else [1] * 25)
            assert math.isclose(sum(u["core_seconds"] for u in units), row["core_seconds"], abs_tol=1e-9)
            if method == "magicbokeh":
                assert row["core_region_calls"] == {"DAdepth_network_forward": 25, "MagicBokeh_six_tiles_and_GPU_reconstruction": 25}
                assert len(row["frame_checks"]) == 25 and all(f["native_depth_finite"] and f["generator_finite"] and f["native_tiles"] == 6 for f in row["frame_checks"])
        core = statistics.mean(r["core_seconds"] for r in rows)
        e2e = statistics.mean(r["E2E_given_inputs_seconds"] for r in rows)
        assert math.isclose(core, summary["Core_seconds_mean"], rel_tol=1e-12)
        assert math.isclose(e2e, summary["E2E_given_inputs_seconds_mean"], rel_tol=1e-12)
        assert math.isclose(25 / e2e, summary["FPS_E2E"], rel_tol=1e-12)
        cpu = method == "deeplens"
        if cpu:
            assert all(r["peak_device_memory_MiB_sampled"] is None and r["peak_torch_allocated_GiB"] is None for r in rows)
            assert summary["runtime"]["TF_intra_threads"] == 8 and summary["runtime"]["TF_inter_threads"] == 1
        else:
            assert "NVIDIA GeForce RTX 4090 D" in summary["runtime"]["gpu_hardware"]
        entry = dict(method=method, device="CPU / Intel Xeon Platinum 8481C / TF1.15 / intra8 inter1" if cpu else "NVIDIA GeForce RTX 4090 D",
            Core_seconds_mean=core, Core_seconds_std=statistics.stdev(r["core_seconds"] for r in rows),
            E2E_given_inputs_seconds_mean=e2e, E2E_given_inputs_seconds_std=statistics.stdev(r["E2E_given_inputs_seconds"] for r in rows),
            FPS_E2E=25 / e2e,
            Peak_device_memory_GiB_sampled=None if cpu else max(r["peak_device_memory_MiB_sampled"] for r in rows) / 1024,
            Peak_torch_allocated_GiB=None if cpu else max(r["peak_torch_allocated_GiB"] for r in rows),
            Peak_torch_reserved_GiB=None if cpu else max(r["peak_torch_reserved_GiB"] for r in rows),
            CPU_RSS_GiB_sampled=max(r["peak_process_RSS_MiB_sampled"] for r in rows) / 1024 if cpu else None,
            formal_samples=20, warmups=4, frames_per_execution=25)
        table.append(entry)
        methods[method] = dict(runtime=summary["runtime"], by_K=summary["by_K"],
            raw_summary=str(args.output_dir / method / "summary.json"),
            raw_warmups=str(args.output_dir / method / "warmups.json"),
            rows=[{k: v for k, v in row.items() if k not in ["native_units", "frame_checks"]} for row in rows])
        prior = old["methods"].get(method)
        old_entry = None if prior is None else dict(
            device="CPU / Intel Xeon Platinum 8470Q / TF1.15 / 8threads" if cpu else "RTX PRO 6000 Blackwell",
            Core_seconds_mean=prior["Core_seconds_mean"], E2E_given_inputs_seconds_mean=prior["E2E_given_inputs_seconds_mean"],
            FPS_E2E=prior["FPS_E2E"], Peak_device_memory_GiB_sampled=None if cpu else prior["peak_device_memory_MiB_sampled"] / 1024,
            CPU_RSS_GiB_sampled=prior["peak_process_RSS_MiB_sampled"] / 1024 if cpu else None)
        comparison.append(dict(method=method, old_hardware=old_entry, current_hardware=entry))
    result = dict(status="SEVEN_METHOD_VIDEO_E2E_COMPLETE", completed_UTC=datetime.now(timezone.utc).isoformat(),
        protocol=executed, MagicBokeh_boundary=read(ROOT / "configs/magicbokeh-video-e2e.json"), table=table,
        methods=methods, old_current_comparison=comparison,
        old_hardware_source=str(source.relative_to(ROOT)),
        validation=dict(methods=7, formal_samples=140, full_clip_warmups=28,
            all_actual25_frames=True, native_units_and_timing_bounds=True,
            output_shape_and_finite_guards_passed=True, core_and_E2E_means_FPS_verified=True,
            CPU_RSS_separate_from_GPU_VRAM=True, old_MagicBokeh_E2E_missing_not_estimated=True),
        statistics="Equal-weight4 fixed clips and5 repeats; reported std is over20 formal samples (includes clip differences); within-K repeat stats retained separately.",
        memory="Device0 memory sampled every0.2s with existing endpoint fallback; PyTorch allocated/reserved and CPU RSS are separate scopes; model loading excluded.",
        limits=["Hardware/CPU/environment differ across machines; no cross-device difference claimed as model optimization.",
            "MagicBokeh Core includes own depth network plus generator; model-cost table remains generator-only.",
            "Old hardware has no MagicBokeh video-E2E measurement; leave empty.",
            "No model-cost, quality or DAVIS reruns; no model, precision, adapter, batch or cache optimization."],
        artifacts=dict(output=str(args.output_dir), logs="logs/video-e2e-4090d-20261001"))
    with (ROOT / args.record).open("x") as file:
        json.dump(result, file, indent=2, ensure_ascii=False, allow_nan=False)
        file.write("\n")
    print(json.dumps(table, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
