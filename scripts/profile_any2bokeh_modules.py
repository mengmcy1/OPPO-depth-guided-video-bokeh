"""Attribute the frozen25-frame ATB core to native stages without changing inference."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys
import time
import subprocess
from datetime import datetime, timezone

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
        self.skip_calls, self.reset_events, self.active = [], [], []
        self.observed_peak = self.observed_reserved_peak = 0

    def capture_peaks(self):
        allocated = self.torch.cuda.max_memory_allocated()
        reserved = self.torch.cuda.max_memory_reserved()
        self.observed_peak = max(self.observed_peak, allocated)
        self.observed_reserved_peak = max(self.observed_reserved_peak, reserved)
        for stage in self.active:
            stage['allocated'] = max(stage['allocated'], allocated)
            stage['reserved'] = max(stage['reserved'], reserved)
        return allocated, reserved

    def guard_resets(self):
        cuda = self.torch.cuda
        original = cuda.reset_peak_memory_stats
        self.originals.append((cuda, 'reset_peak_memory_stats', original))
        def guarded(*args, **kwargs):
            if self.enabled:
                allocated, reserved = self.capture_peaks()
                self.reset_events.append(dict(active_stages=[s['name'] for s in self.active],
                    allocated_peak_before_reset_GiB=allocated/2**30,
                    reserved_peak_before_reset_GiB=reserved/2**30))
            return original(*args, **kwargs)
        cuda.reset_peak_memory_stats = guarded

    @staticmethod
    def tensor_storage(value):
        storage = value.untyped_storage()
        return dict(shape=list(value.shape), dtype=str(value.dtype), device=str(value.device),
            bytes=value.numel()*value.element_size(), storage_bytes=storage.nbytes(),
            storage_pointer=storage.data_ptr(), storage_offset_elements=value.storage_offset())

    def cache(self):
        tensors = getattr(self.pipe.vae.encoder, "residual_down_blocks", [])
        unique = {}
        rows = []
        for value in tensors:
            storage = value.untyped_storage()
            unique[(value.device.type, value.device.index, storage.data_ptr())] = storage.nbytes()
            rows.append(self.tensor_storage(value))
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
            self.capture_peaks()
            before = self.torch.cuda.memory_allocated()
            reserved_before = self.torch.cuda.memory_reserved()
            cache_before = self.cache()
            # Native decode_latents passes these arguments positionally.
            latent_cache_before = kwargs.get('latent_cache', args[5] if actual == 'Overlap_and_temporal_VAE_decode' and len(args) > 5 else None)
            latent_cache_before = None if latent_cache_before is None else self.tensor_storage(latent_cache_before)
            is_last_batch = kwargs.get('is_last_batch', args[6] if actual == 'Overlap_and_temporal_VAE_decode' and len(args) > 6 else None)
            self.torch.cuda.reset_peak_memory_stats()
            active = dict(name=actual, allocated=before, reserved=reserved_before)
            self.active.append(active)
            start = time.perf_counter()
            try:
                result = original(*args, **kwargs)
                self.torch.cuda.synchronize()
                seconds = time.perf_counter()-start
                self.capture_peaks()
            finally:
                self.active.pop()
            peak, reserved_peak = active['allocated'], active['reserved']
            self.rows.append(dict(stage=actual, seconds=seconds, allocated_before_GiB=before/2**30,
                allocated_after_GiB=self.torch.cuda.memory_allocated()/2**30, peak_allocated_GiB=peak/2**30,
                reserved_before_GiB=reserved_before/2**30,
                reserved_after_GiB=self.torch.cuda.memory_reserved()/2**30, peak_reserved_GiB=reserved_peak/2**30,
                incremental_peak_above_stage_entry_GiB=max(0, peak-before)/2**30,
                incremental_reserved_peak_above_stage_entry_GiB=max(0, reserved_peak-reserved_before)/2**30,
                input_shape=None if tensor is None else list(tensor.shape),
                input_dtype=None if tensor is None else str(tensor.dtype),
                latent_cache_before=latent_cache_before,
                latent_cache_after=self.tensor_storage(result[1]) if actual == 'Overlap_and_temporal_VAE_decode' else None,
                is_last_batch=is_last_batch,
                RGB_encoder_cache_before=cache_before, RGB_encoder_cache_after=self.cache()))
            return result
        setattr(obj, attribute, measured)

    def count_calls(self, obj, attribute, destination):
        original = getattr(obj, attribute)
        self.originals.append((obj, attribute, original))
        def counted(*args, **kwargs):
            if self.enabled:
                row = dict(shape=list(args[0].shape), dtype=str(args[0].dtype),
                    cache_flag=kwargs.get("cache_flag"), num_frames=kwargs.get("num_frames"),
                    start_frame=kwargs.get("start_frame"))
                if attribute == 'decode':
                    row['RGB_encoder_cache'] = self.cache()
                destination().append(row)
            return original(*args, **kwargs)
        setattr(obj, attribute, counted)

    def observe_skip(self, obj, level):
        original = obj.forward
        self.originals.append((obj, 'forward', original))
        def observed(value, *args, **kwargs):
            if self.enabled:
                self.skip_calls.append(dict(decoder_call=len(self.decoder_calls)-1,
                    RGB_cache_level=level, input_storage=self.tensor_storage(value)))
            return original(value, *args, **kwargs)
        obj.forward = observed

    def attach(self):
        self.guard_resets()
        self.wrap(self.pipe, "_encode_image", "CLIP_image_embedding")
        self.wrap(self.pipe, "_tensor_to_vae_latent", lambda kw: "VAE_RGB_encode_sample_cache" if kw.get("cache_flag") else "VAE_CoC_encode_sample")
        self.wrap(self.pipe.unet, "forward", "UNet_spatiotemporal_condition")
        self.wrap(self.pipe, "decode_latents", "Overlap_and_temporal_VAE_decode")
        self.count_calls(self.pipe.vae, "encode", lambda:self.encoder_calls)
        self.count_calls(self.pipe.vae, "decode", lambda:self.decoder_calls)
        for index in range(4):
            self.observe_skip(getattr(self.pipe.vae.decoder, f'skip_conv_{index+1}'), 3-index)

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
    runtime.update(gpu=torch.cuda.get_device_name(0), gpu_total_GiB=torch.cuda.get_device_properties(0).total_memory/2**30,
        driver=subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], text=True).strip(),
        CPU=next(line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),
        container_cpu_max=Path('/sys/fs/cgroup/cpu.max').read_text().strip(),
        container_memory_limit_bytes=int(Path('/sys/fs/cgroup/memory.max').read_text()),
        python=sys.version, torch_threads=torch.get_num_threads(),
        project_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        native_code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT.parent/'any-to-bokeh', text=True).strip())
    runs, warmups = [], []
    try:
        for sid, K in zip(config["scenes"], config["K"]):
            csvpath = out / (sid + ".csv")
            folder = ROOT / config["input"] / sid
            import csv
            with csvpath.open("x") as file:
                csv.writer(file).writerows([["aif_folder", "disp_folder", "k"], [str(folder / "rgb"), str(folder / "disparity"), K]])
            frames = prepare("any2bokeh", state, folder, K, csvpath, 25)
            windows = dict(input_shapes={key:list(value.shape) for key,value in frames.items() if isinstance(value, torch.Tensor)},
                weights=frames['weight'].tolist(),
                source_frame_indices=[list(range(start, min(start+8, 25))) + [None]*max(0, start+8-25) for start in range(0, 24, 4)],
                native_window_pairs=[[0,1], [2,3], [4,5]], overlap_frames=4, padding_slots=3)
            profiler.enabled = False
            state["meter"].reset()
            warmup = infer("any2bokeh", state, frames)
            assert len(warmup) == 25 and all(v.shape == (576, 1024, 3) and np.isfinite(v).all() for v in warmup)
            warmups.append(dict(scene=sid, K=K, output_frames=25, output_shape=[576,1024,3], finite=True,
                native_output_units=[unit['output_frames'] for unit in state['meter'].units]))
            (out / f'{sid}-warmup.json').write_text(json.dumps(warmups[-1], indent=2)+'\n')
            del warmup
            for repeat in range(args.repeats):
                assert set(gpu_processes()) <= {os.getpid()}, "Foreign GPU process appeared"
                profiler.reset()
                profiler.enabled = True
                state["meter"].reset()
                outputs = infer("any2bokeh", state, frames)
                assert len(outputs) == 25 and all(v.shape == (576, 1024, 3) and np.isfinite(v).all() for v in outputs)
                profiler.capture_peaks()
                by_stage = defaultdict(float)
                for row in profiler.rows:
                    by_stage[row["stage"]] += row["seconds"]
                core = state["meter"].seconds
                assert sum(by_stage.values()) <= core
                row = dict(scene=sid, K=K, repeat=repeat, output_frames=25, core_seconds_with_phase_sync=core,
                    output_shape=[576,1024,3], finite=True, input_windows=windows,
                    native_output_units=[unit['output_frames'] for unit in state['meter'].units],
                    stage_seconds=dict(by_stage), unattributed_core_seconds=core-sum(by_stage.values()),
                    stages=profiler.rows, encoder_calls=profiler.encoder_calls, decoder_calls=profiler.decoder_calls,
                    decoder_skip_calls=profiler.skip_calls, peak_reset_events=profiler.reset_events,
                    observed_peak_allocated_GiB=profiler.observed_peak/2**30,
                    observed_peak_reserved_GiB=profiler.observed_reserved_peak/2**30)
                assert len(profiler.rows) == 15 and len(profiler.encoder_calls) == 6 and len(profiler.decoder_calls) == 4
                assert len(profiler.skip_calls) == 16 and row['native_output_units'] == [8,8,9]
                runs.append(row)
                (out / f"{sid}-repeat{repeat}.json").write_text(json.dumps(row, indent=2)+"\n")
                print("ATB_MODULE_PROFILE", sid, repeat, dict(by_stage), row["observed_peak_allocated_GiB"], flush=True)
                del outputs
            del frames
    finally:
        profiler.detach()
    names = sorted({name for run in runs for name in run["stage_seconds"]})
    summary = dict(status="ATB_NATIVE_MODULE_PROFILE_COMPLETE", completed_UTC=datetime.now(timezone.utc).isoformat(), runtime=runtime, protocol_source="configs/efficiency-track-a.json",
        scope="Same frozen25-frame native core and weights; extra per-stage synchronization measures attribution, not replacement Table3 latency",
        memory="Absolute allocated/reserved stage maxima, preserved before every nested peak reset. Includes resident weights, retained caches and intermediates; never add peaks or name entry-relative increments pure activations. Reserved is allocator-held capacity. No empty_cache or native cache clearing.",
        native_units="Three two-window calls output8/8/9 frames. VAE encode/decode input slots counted, including overlap and padding.",
        repeats_per_scene=args.repeats, warmups_per_scene=1,
        mean_stage_seconds_per25frame_clip={name:float(np.mean([r["stage_seconds"].get(name, 0) for r in runs])) for name in names},
        peak_allocated_GiB=max(r["observed_peak_allocated_GiB"] for r in runs),
        peak_reserved_GiB=max(r['observed_peak_reserved_GiB'] for r in runs), runs=runs, warmups=warmups,
        model_or_precision_changes=False, GFLOPs_not_measured=True,
        conclusion_boundary="Native execution attribution supports optimization hypotheses; no improvement or quality equivalence established")
    (out / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps({k:v for k,v in summary.items() if k not in ["runtime", "runs"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
