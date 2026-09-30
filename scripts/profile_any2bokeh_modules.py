"""Attribute the frozen25-frame ATB core to native stages without changing inference."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


class NativeStages:
    def __init__(self, pipe):
        import torch
        self.torch, self.pipe = torch, pipe
        self.enabled = False
        self.originals = []
        self.reset()

    def reset(self):
        self.rows, self.encoder_calls, self.decoder_calls = [], [], []
        self.observed_peak = 0

    def cache(self):
        tensors = getattr(self.pipe.vae.encoder, "residual_down_blocks", [])
        unique = {}
        rows = []
        for value in tensors:
            storage = value.untyped_storage()
            unique[(value.device.type, storage.data_ptr())] = storage.nbytes()
            rows.append(dict(shape=list(value.shape), dtype=str(value.dtype), bytes=value.numel()*value.element_size()))
        return dict(tensors=rows, unique_storage_bytes=sum(unique.values()), GiB=sum(unique.values())/2**30)

    def wrap(self, obj, attribute, label):
        original = getattr(obj, attribute)
        self.originals.append((obj, attribute, original))
        def measured(*args, **kwargs):
            if not self.enabled:
                return original(*args, **kwargs)
            actual = label(kwargs) if callable(label) else label
            tensor = next((x for x in args if isinstance(x, self.torch.Tensor)), None)
            self.torch.cuda.synchronize()
            self.observed_peak = max(self.observed_peak, self.torch.cuda.max_memory_allocated())
            before = self.torch.cuda.memory_allocated()
            cache_before = self.cache()
            self.torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            result = original(*args, **kwargs)
            self.torch.cuda.synchronize()
            seconds = time.perf_counter()-start
            peak = self.torch.cuda.max_memory_allocated()
            self.observed_peak = max(self.observed_peak, peak)
            self.rows.append(dict(stage=actual, seconds=seconds, allocated_before_GiB=before/2**30,
                allocated_after_GiB=self.torch.cuda.memory_allocated()/2**30, peak_allocated_GiB=peak/2**30,
                incremental_peak_above_stage_entry_GiB=max(0, peak-before)/2**30,
                input_shape=None if tensor is None else list(tensor.shape),
                input_dtype=None if tensor is None else str(tensor.dtype),
                RGB_encoder_cache_before=cache_before, RGB_encoder_cache_after=self.cache()))
            return result
        setattr(obj, attribute, measured)

    def count_calls(self, obj, attribute, destination):
        original = getattr(obj, attribute)
        self.originals.append((obj, attribute, original))
        def counted(*args, **kwargs):
            if self.enabled:
                destination().append(dict(shape=list(args[0].shape), dtype=str(args[0].dtype),
                    cache_flag=kwargs.get("cache_flag"), num_frames=kwargs.get("num_frames"),
                    start_frame=kwargs.get("start_frame")))
            return original(*args, **kwargs)
        setattr(obj, attribute, counted)

    def attach(self):
        self.wrap(self.pipe, "_encode_image", "CLIP_image_embedding")
        self.wrap(self.pipe, "_tensor_to_vae_latent", lambda kw: "VAE_RGB_encode_sample_cache" if kw.get("cache_flag") else "VAE_CoC_encode_sample")
        self.wrap(self.pipe.unet, "forward", "UNet_spatiotemporal_condition")
        self.wrap(self.pipe, "decode_latents", "Overlap_and_temporal_VAE_decode")
        self.count_calls(self.pipe.vae, "encode", lambda:self.encoder_calls)
        self.count_calls(self.pipe.vae, "decode", lambda:self.decoder_calls)

    def detach(self):
        for obj, name, original in reversed(self.originals):
            setattr(obj, name, original)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0"
    from measure_efficiency import assert_idle_gpu, gpu_processes, load, prepare, infer
    assert_idle_gpu()
    import torch
    import numpy as np
    config = json.loads((ROOT / "configs/efficiency-track-a.json").read_text())
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    state, runtime = load("any2bokeh", config)
    profiler = NativeStages(state["pipe"])
    profiler.attach()
    runs = []
    try:
        for sid, K in zip(config["scenes"], config["K"]):
            csvpath = out / (sid + ".csv")
            folder = ROOT / config["input"] / sid
            import csv
            with csvpath.open("x") as file:
                csv.writer(file).writerows([["aif_folder", "disp_folder", "k"], [str(folder / "rgb"), str(folder / "disparity"), K]])
            frames = prepare("any2bokeh", state, folder, K, csvpath, 25)
            profiler.enabled = False
            state["meter"].reset()
            warmup = infer("any2bokeh", state, frames)
            assert len(warmup) == 25
            del warmup
            for repeat in range(args.repeats):
                assert set(gpu_processes()) <= {os.getpid()}, "Foreign GPU process appeared"
                profiler.reset()
                profiler.enabled = True
                state["meter"].reset()
                outputs = infer("any2bokeh", state, frames)
                assert len(outputs) == 25 and all(v.shape == (576, 1024, 3) and np.isfinite(v).all() for v in outputs)
                profiler.observed_peak = max(profiler.observed_peak, torch.cuda.max_memory_allocated())
                by_stage = defaultdict(float)
                for row in profiler.rows:
                    by_stage[row["stage"]] += row["seconds"]
                core = state["meter"].seconds
                assert sum(by_stage.values()) <= core
                row = dict(scene=sid, K=K, repeat=repeat, output_frames=25, core_seconds_with_phase_sync=core,
                    stage_seconds=dict(by_stage), unattributed_core_seconds=core-sum(by_stage.values()),
                    stages=profiler.rows, encoder_calls=profiler.encoder_calls, decoder_calls=profiler.decoder_calls,
                    observed_peak_allocated_GiB=profiler.observed_peak/2**30)
                runs.append(row)
                (out / f"{sid}-repeat{repeat}.json").write_text(json.dumps(row, indent=2)+"\n")
                print("ATB_MODULE_PROFILE", sid, repeat, dict(by_stage), row["observed_peak_allocated_GiB"], flush=True)
                del outputs
            del frames
    finally:
        profiler.detach()
    names = sorted({name for run in runs for name in run["stage_seconds"]})
    summary = dict(status="ATB_NATIVE_MODULE_PROFILE_COMPLETE", runtime=runtime, protocol_source="configs/efficiency-track-a.json",
        scope="Same frozen25-frame native core and weights; extra per-stage synchronization measures attribution, not replacement Table3 latency",
        memory="Stage peaks are absolute allocations including resident weights and retained RGB feature caches; do not add stage peaks together. Unique cache storage bytes recorded separately.",
        native_units="Three two-window calls output8/8/9 frames. VAE encode/decode input slots counted, including overlap and padding.",
        repeats_per_scene=args.repeats, warmups_per_scene=1,
        mean_stage_seconds_per25frame_clip={name:float(np.mean([r["stage_seconds"].get(name, 0) for r in runs])) for name in names},
        peak_allocated_GiB=max(r["observed_peak_allocated_GiB"] for r in runs), runs=runs,
        model_or_precision_changes=False, GFLOPs_not_measured=True,
        conclusion_boundary="Native execution attribution supports optimization hypotheses; no improvement or quality equivalence established")
    (out / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps({k:v for k,v in summary.items() if k not in ["runtime", "runs"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
