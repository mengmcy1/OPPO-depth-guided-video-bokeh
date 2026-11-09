"""One three-frame real-video pose/depth check with licensed DA3-BASE weights."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from depth_anything_3.api import DepthAnything3


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--video-frames", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    images = sorted(a.video_frames.glob("*.jpg"))
    chosen = [str(images[i]) for i in [0, 5, 10]]
    model = DepthAnything3.from_pretrained(str(a.weights.resolve())).to("cuda:0").eval()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); start = time.monotonic()
    result = model.inference(chosen, process_res=336, export_dir=None)
    torch.cuda.synchronize(); seconds = time.monotonic() - start
    assert np.isfinite(result.depth).all() and np.isfinite(result.extrinsics).all()
    assert np.isfinite(result.intrinsics).all()
    R = result.extrinsics[..., :3, :3]
    rotation_error = float(np.abs(np.swapaxes(R, -1, -2) @ R - np.eye(3)).max())
    assert rotation_error < .01 and bool((np.linalg.det(R) > 0).all())
    np.savez(a.output / "prediction.npz", depth=result.depth, extrinsics=result.extrinsics,
             intrinsics=result.intrinsics, confidence=result.conf, processed_images=result.processed_images)
    record = {"model": "DA3-BASE", "weight_license": "Apache-2.0", "input_frames": chosen,
              "depth_shape": list(result.depth.shape), "extrinsics_shape": list(result.extrinsics.shape),
              "intrinsics_shape": list(result.intrinsics.shape), "elapsed_seconds": seconds,
              "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
              "max_rotation_orthogonality_error": rotation_error,
              "pose_convention": "OpenCV world-to-camera", "scale": "relative, not metric",
              "scope": "inference/finite pose check only; no pose accuracy evaluation"}
    (a.output / "inference.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
