"""One 512x512 author-example forward; uses the unmodified sibling repository."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
out = args.output_dir.resolve()
out.mkdir(parents=True, exist_ok=False)
repo = root.parent / "MagicBokeh"
os.chdir(repo)
sys.path.insert(0, str(repo))

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image
from inference import adjust_depth_with_mask
from magicbokeh import model_test
from run_magicbokeh import configure_attention

torch.manual_seed(42)
torch.set_num_threads(4)
configure_attention()
model_args = argparse.Namespace(
    pretrained_model_name_or_path=str(root / "models/sd21-base"),
    model_path=str(root / "models/MagicBokeh/lora.pkl"),
    mixed_precision="fp16",
)
model = model_test(model_args).eval()
name = "EBB_LR_1321"
image = Image.open(repo / f"test_data/inputs/{name}.png").convert("RGB")
depth_array = np.load(repo / f"test_data/depth/{name}.npy")
depth = Image.fromarray((depth_array * 255).astype(np.uint8), mode="L")
mask = Image.open(repo / f"test_data/mask/{name}.png").convert("L")
focus = adjust_depth_with_mask(depth, mask) / 255.0
# Use the same top-left tile and conditioning as the author's tiled CLI.
image = image.crop((0, 0, 512, 512))
depth = depth.crop((0, 0, 512, 512))
rgb = to_tensor(image).unsqueeze(0).cuda() * 2 - 1
coc = (to_tensor(depth).unsqueeze(0).cuda() - focus).abs()
focus_mask = 1 - (coc > 0.1).float()
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
start = time.monotonic()
with torch.inference_mode():
    result = model(rgb, coc * 2 - 1, focus_mask * 2 - 1, "")
torch.cuda.synchronize()
assert result.shape == rgb.shape, (result.shape, rgb.shape)
assert torch.isfinite(result).all(), "Non-finite model output"
elapsed = time.monotonic() - start
to_pil_image((result[0].float().cpu() + 1) / 2).save(out / "output.png")
image.save(out / "input.png")
report = dict(model="MagicBokeh", input_shape=list(rgb.shape),
              attention_compatibility="xformers FA3 disabled on sm_120",
              output_shape=list(result.shape), finite=True,
              output_min=result.min().item(), output_max=result.max().item(),
              output_std=result.float().std().item(),
              forward_seconds=elapsed,
              peak_allocated_GiB=torch.cuda.max_memory_allocated() / 2**30,
              peak_reserved_GiB=torch.cuda.max_memory_reserved() / 2**30,
              torch=torch.__version__, device=torch.cuda.get_device_name(0))
(out / "result.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
