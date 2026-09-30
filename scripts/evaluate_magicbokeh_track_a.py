"""Evaluate frozen MagicBokeh clips with the existing six-method metric kernels."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    os.chdir(ROOT)
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0"
    config = json.loads((ROOT / "configs/magicbokeh-track-a-frozen.json").read_text())
    out = (args.output_dir or ROOT / config["output"]).resolve()
    source = ROOT / config["input"]
    ids = [s["id"] for s in json.loads((source / "config.json").read_text())["scenes"]]
    logs = ROOT / "logs/magicbokeh-point-focus-20260930"
    (out / "metrics").mkdir(parents=True, exist_ok=True)

    def run(command, log):
        with log.open("x") as stream:
            print("START_MAGICBOKEH_METRIC", log.name, flush=True)
            subprocess.run([sys.executable, *command], stdout=stream, stderr=subprocess.STDOUT, check=True)
        print("DONE_MAGICBOKEH_METRIC", log.name, flush=True)

    for batch in range(20):
        selected = ids[batch*10:(batch+1)*10]
        report = out / "metrics" / f"batch_{batch:02d}.json"
        if report.exists():
            saved = json.loads(report.read_text())
            assert [r["scene"] for r in saved["rows"]] == selected
            continue
        start = time.monotonic()
        while not all((out / "timing" / sid / "complete.json").is_file() for sid in selected):
            job_files = sorted(out.glob("jobs-*.json"))
            if job_files:
                jobs = json.loads(job_files[-1].read_text())
                for sid in selected:
                    if (out / "timing" / sid / "complete.json").exists():
                        continue
                    owner = ids.index(sid) % jobs["workers"]
                    pid = next(x["pid"] for x in jobs["jobs"] if x["shard"] == owner)
                    stat = Path(f"/proc/{pid}/stat")
                    if not stat.exists() or stat.read_text().split()[2] == "Z":
                        raise RuntimeError(f"Inference worker exited before {sid}; inspect logs")
            if time.monotonic() - start > 14400:
                raise TimeoutError(f"Inference batch{batch} did not complete in4hours")
            time.sleep(5)
        run(["scripts/evaluate_renderer_pilot.py", "--input", config["input"], "--only", ",".join(selected),
            "--reference-dir", "gt_mpib_public_srgb", "--reference-label", "MPIB-public proxy",
            "--prediction-root", str(out / "clips"), "--prediction-dir", "pred_magicbokeh",
            "--author-rm", "--torchmetrics", "--scope", "Frozen target-guided medoid B; native DA depth; GT target alpha only assists point selection; no GT geometry fed to generator",
            "--report", str(report)], logs / f"metrics-batch{batch:02d}-{int(time.time())}.log")
    if not (out / "inference-complete.json").is_file():
        start = time.monotonic()
        while not (out / "inference-complete.json").is_file():
            if time.monotonic() - start > 60:
                raise RuntimeError("All clips exist but coordinator did not finish")
            time.sleep(2)
    for phase in ["vfid", "fvd"]:
        run(["scripts/evaluate_author_public_metrics.py", "--clips", config["input"],
            "--prediction-root", str(out / "clips"), "--prediction-dir", "pred_magicbokeh",
            "--reference-features", "outputs/author-public-metrics-20260928", "--output", str(out / "author-public"),
            "--phase", phase], logs / f"{phase}-{int(time.time())}.log")
    with (out / "metrics-complete.json").open("x") as file:
        json.dump(dict(status="ALL_MAGICBOKEH_200x25_METRICS_COMPLETE", scenes=200,
            primary=["PSNR_TorchMetrics", "SSIM_TorchMetrics", "LPIPS", "FVD_VideoMAE", "FD", "RM"],
            supplementary="VFID-I consistent with existing Table1"), file, indent=2)
    print("ALL_MAGICBOKEH_METRICS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
