"""Run the frozen video efficiency entries serially, resuming completed methods."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--logs-dir", type=Path, required=True)
    args = parser.parse_args()
    output, logs = ROOT / args.output_dir, ROOT / args.logs_dir
    output.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    # A partial method cannot simply skip RNG-consuming repeats. Preserve it and
    # restart that method; completed summaries retain the original resume path.
    for method in ["any2bokeh", "mpib", "bokehme", "drbokeh", "bokehdiff", "deeplens", "magicbokeh"]:
        folder = output / method
        if folder.exists() and not (folder / "summary.json").exists():
            preserved = output / f"{method}-incomplete-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}"
            folder.rename(preserved)
            print("PRESERVED_INCOMPLETE_METHOD", method, preserved.name, flush=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1",
        PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
    six_record = output / "six-methods.json"
    if not six_record.exists():
        command = [sys.executable, str(ROOT / "scripts/measure_efficiency.py"), "--all", "--table", "3b",
            "--skip-flop-diagnostic", "--output-dir", str(args.output_dir), "--logs-dir", str(args.logs_dir),
            "--record", str(args.output_dir / "six-methods.json")]
        subprocess.run(command, cwd=ROOT, env=env, check=True)
    else:
        record = json.loads(six_record.read_text())
        assert record["status"] == "TABLE3B_SIX_METHODS_COMPLETE_REVIEW_PENDING"
        assert (ROOT / record["protocol"]["output"]).resolve() == output.resolve()
        print("SIX_VIDEO_METHODS_ALREADY_COMPLETE", flush=True)
    command = [sys.executable, str(ROOT / "scripts/measure_magicbokeh_e2e.py"),
        "--output-dir", str(args.output_dir / "magicbokeh")]
    with (logs / "magicbokeh.log").open("a") as file:
        subprocess.run(command, cwd=ROOT, env=env, stdout=file, stderr=subprocess.STDOUT, check=True)
    print("SEVEN_VIDEO_E2E_METHODS_COMPLETE", flush=True)
    marker = output / "execution-complete.json"
    if not marker.exists():
        marker.write_text(json.dumps(dict(status="SEVEN_VIDEO_E2E_METHODS_COMPLETE",
            completed_UTC=datetime.now(timezone.utc).isoformat()), indent=2) + "\n")


if __name__ == "__main__":
    main()
