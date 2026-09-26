"""Run the author BokehMe/MPIB demo with current CuPy and local weights."""
import argparse
import os
from pathlib import Path
import runpy
import sys

parser = argparse.ArgumentParser()
parser.add_argument("method", choices=["BokehMe", "MPIB"])
args, remaining = parser.parse_known_args()
root = Path(__file__).resolve().parents[1]
repo = root.parent / args.method
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
# Author checkpoints include legacy Python metadata; only use trusted weights.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
import torch
import cupy
import numpy as np

torch.set_num_threads(4)
# These legacy CuPy names were removed. Preserve the author's CUDA kernels.
if not hasattr(cupy, "int"):
    cupy.int = np.int32
if not hasattr(cupy.cuda, "compile_with_cache"):
    cupy.cuda.compile_with_cache = lambda source: cupy.RawModule(code=source)
os.chdir(repo)
sys.path.insert(0, str(repo))
weights = root / "models" / args.method
defaults = []
if args.method == "MPIB":
    # The full MPIB checkpoint replaces the encoder; skip redundant ImageNet download.
    defaults.extend(["--config_path", str(root / "configs/mpib-inference.yaml")])
for component in (["arnet", "iunet"] if args.method == "BokehMe" else ["mpinet", "aupnet", "inpnet"]):
    defaults.extend([f"--{component}_checkpoint_path", str(weights / f"{component}.pth")])
sys.argv = [str(repo / "demo.py"), *defaults, *remaining]
runpy.run_path(sys.argv[0], run_name="__main__")
