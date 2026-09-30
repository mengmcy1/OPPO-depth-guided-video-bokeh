"""Count target visibility and fixed interior clicks without running a model."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from baselines.magicbokeh_focus import visible_target_alpha, safe_target_region


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    inputs = ROOT / "outputs/any2bokeh-mpib-public-proxy-200-20260928/clips"
    manifest = json.loads((inputs / "config.json").read_text())
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    rows, unavailable, tiers, semantic_changes = [], [], Counter(), []
    start = time.monotonic()
    for scene in manifest["scenes"]:
        folder = inputs / scene["id"]
        metadata = json.loads((folder / "metadata.json").read_text())
        identities = set()
        for frame in metadata["frames"]:
            alpha, identity = visible_target_alpha(folder, metadata, frame)
            region, distance, info = safe_target_region(alpha)
            row = dict(scene=scene["id"], frame=frame["index"], K=scene["K"], target_source_id=identity, **info)
            rows.append(row)
            identities.add(identity)
            if info["status"] != "AVAILABLE":
                unavailable.append(dict(scene=scene["id"], frame=frame["index"], target_source_id=identity))
            else:
                tiers[str(info["alpha_threshold"])] += 1
        if len(identities) != 1:
            semantic_changes.append(scene["id"])
        print("TARGET_INVENTORY", scene["id"], len(rows), flush=True)
    summary = dict(status="TARGET_VISIBILITY_INVENTORY_COMPLETE", scenes=len(manifest["scenes"]),
        frames=len(rows), focus_point_unavailable=len(unavailable), unavailable=unavailable,
        alpha_fallback_counts=dict(tiers), semantic_target_changes=semantic_changes,
        visible_alpha="Target layer alpha times product(1-alpha) of all nearer layers; numeric depth only verifies target identity",
        click_rule="Largest8-connected component, exact Euclidean distance with zero-padded image boundary, maximal margin; ties row-major",
        no_models_or_metrics_run=True, seconds=time.monotonic()-start)
    (out / "frames.json").write_text(json.dumps(rows, indent=2) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
