"""Evaluate one NPZ pair/mask, optionally a temporal pair; no full-data run."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.aperture_metrics import regional_metrics, temporal_warp_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="NPZ: prediction,target,disocclusion_mask; optional temporal fields")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = np.load(args.input)
    tensor = {k: torch.as_tensor(data[k]) for k in data.files}
    pred, target = tensor["prediction"].float(), tensor["target"].float()
    masks = {"full": torch.ones(pred.shape[:2], dtype=torch.bool)}
    masks.update({k.removesuffix("_mask"): v.bool() for k, v in tensor.items() if k.endswith("_mask") and k != "flow_valid_mask"})
    result = {"regions": {k: regional_metrics(pred, target, v) for k, v in masks.items()}}
    if "backward_flow" in tensor:
        result["temporal"] = temporal_warp_error(pred, tensor["previous_prediction"].float(),
                                                 tensor["backward_flow"].float(), tensor["flow_valid_mask"],
                                                 target, tensor["previous_target"].float())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
