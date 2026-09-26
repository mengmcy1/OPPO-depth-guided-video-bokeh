"""One streaming forward on 12 example frames (two overlapping 8-frame groups)."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", type=Path, required=True)
# Author resolution, verified on the current 96 GB GPU.
parser.add_argument("--height", type=int, default=576)
parser.add_argument("--width", type=int, default=1024)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
out = args.output_dir.resolve()
out.mkdir(parents=True, exist_ok=False)
repo = root.parent / "any-to-bokeh"
os.chdir(repo)
sys.path.insert(0, str(repo))
os.environ.setdefault("HF_HOME", str(root / "models/huggingface"))

import imageio.v2 as imageio
import torch
from models.unet import UNetSpatioTemporalConditionModel
from models.vae import AutoencoderKLTemporalDecoder
from pipelines.any2bokeh_pipe import StableVideoDiffusionPipeline
from utils.dataset import BokehCocMPIDatasetWholeVid, collate_fn_val_coc_mpi_whole_video

torch.set_num_threads(4)
torch.manual_seed(0)
for source, target in [("videos", "frames"), ("disp", "disp")]:
    dest = out / target
    dest.mkdir()
    files = sorted((repo / "demo_dataset" / source / "breakdance").iterdir(),
                   key=lambda p: int(p.name.split("_")[0].split(".")[0]))[:12]
    assert len(files) == 12
    for f in files:
        (dest / f.name).symlink_to(f)
with (out / "input.csv").open("w") as f:
    writer = csv.writer(f)
    writer.writerow(["aif_folder", "disp_folder", "k"])
    writer.writerow([out / "frames", out / "disp", 16])
dataset = BokehCocMPIDatasetWholeVid(str(out / "input.csv"),
                                   sample_size=(args.height, args.width))
batch = collate_fn_val_coc_mpi_whole_video([dataset[0]])
checkpoint = root / "models/any-to-bokeh/checkpoints"
unet = UNetSpatioTemporalConditionModel.from_pretrained(
    str(checkpoint), subfolder="unet", torch_dtype=torch.float16,
    low_cpu_mem_usage=False, local_files_only=True)
vae = AutoencoderKLTemporalDecoder.from_pretrained(
    str(checkpoint / "vae"), torch_dtype=torch.float16, local_files_only=True)
pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt", unet=unet, vae=vae,
    torch_dtype=torch.float16, variant="fp16", low_cpu_mem_usage=False,
    local_files_only=True).to("cuda:0")
pipe.set_progress_bar_config(disable=True)
# Match the author CLI: the VAE upcasts to float32 for RGB/CoC encoding.
inputs = {k: batch[k].to(device="cuda", dtype=(torch.float32 if k in
          ("pixel_values", "coc") else torch.float16))
          for k in ["pixel_values", "coc", "k", "mpi_mask", "weight"]}
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
start = time.monotonic()
with torch.inference_mode():
    frames, cache = pipe(
        inputs["coc"], inputs["pixel_values"], input_k=inputs["k"],
        mpi_mask=inputs["mpi_mask"], weight=inputs["weight"],
        latent_cache=None, is_last_batch=False, overlap_frame=4,
        height=args.height, width=args.width, num_frames=8, decode_chunk_size=8,
        motion_bucket_id=127, fps=7, noise_aug_strength=0.0,
        num_inference_steps=1, generator=torch.Generator(device="cpu").manual_seed(0),
        batch_size=2, output_type="pt")
torch.cuda.synchronize()
elapsed = time.monotonic() - start
result = frames[0]
assert tuple(result.shape) == (8, 3, args.height, args.width), result.shape
assert torch.isfinite(result).all(), "Non-finite model output"
assert torch.isfinite(cache).all(), "Non-finite streaming cache"
video = (result.detach().float().cpu().permute(0, 2, 3, 1).clamp(0, 1) * 255).byte().numpy()
imageio.mimwrite(out / "output.mp4", video, fps=20, codec="libx264")
imageio.imwrite(out / "first_frame.png", video[0])
report = dict(model="Any-to-Bokeh", input_shape=list(inputs["pixel_values"].shape),
              output_shape=list(result.shape), cache_shape=list(cache.shape),
              note="One streaming call: 8 output frames; 4 trailing frames remain in cache.",
              finite=True, forward_seconds=elapsed,
              output_min=result.min().item(), output_max=result.max().item(),
              output_std=result.float().std().item(),
              peak_allocated_GiB=torch.cuda.max_memory_allocated() / 2**30,
              peak_reserved_GiB=torch.cuda.max_memory_reserved() / 2**30,
              torch=torch.__version__, device=torch.cuda.get_device_name(0))
(out / "result.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
