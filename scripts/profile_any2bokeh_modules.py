"""Attribute the frozen25-frame ATB core to native stages without changing inference."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys
import time
import subprocess
import pickle
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def analyze_memory_history(output):
    """Replay requested-byte lifetimes; allocator block peaks stay separately measured."""
    output = output.resolve()
    summary = json.loads((output/'summary.json').read_text())
    with (output/'memory-initial-snapshot.pickle').open('rb') as file:
        initial = pickle.load(file)
    with (output/'memory-snapshot.pickle').open('rb') as file:
        final = pickle.load(file)
    trace = final['device_traces'][0]
    assert len(trace) < summary['allocation_history']['max_entries'] and trace[0]['action'] == 'snapshot'
    live = {block['address']: dict(address=block['address'], bytes=block['requested_size'],
        allocated_event=None, free_requested_event=None, frames=[])
        for segment in initial['segments'] for block in segment['blocks'] if block['state'] == 'active_allocated'}
    total = sum(value['bytes'] for value in live.values())
    points = summary['allocator']['points']
    boundaries = {point['history_event_index_exclusive']: point for point in points if 'history_event_index_exclusive' in point}
    targets = []
    for entry in points:
        if entry['edge'] == 'entry' and entry['stage'] in ['VAE_RGB_encode_sample_cache', 'VAE_CoC_encode_sample']:
            exit_point = next(point for point in points if point['stage'] == entry['stage'] and point['native_unit'] == entry['native_unit'] and point['edge'] == 'exit')
            targets.append(dict(entry=entry, exit=exit_point, peak=-1))
    for index, event in enumerate(trace):
        if index in boundaries:
            assert total == boundaries[index]['allocator_current']['requested_bytes.all.current']['bytes']
        if event['action'] == 'alloc':
            assert event['addr'] not in live
            live[event['addr']] = dict(address=event['addr'], bytes=event['size'], allocated_event=index,
                free_requested_event=None, frames=event.get('frames', []))
            total += event['size']
        elif event['action'] == 'free_requested':
            value = live.pop(event['addr'])
            value['free_requested_event'] = index
            total -= value['bytes']
        for target in targets:
            if target['entry']['history_event_index_exclusive'] <= index < target['exit']['history_event_index_exclusive'] and total > target['peak']:
                target.update(peak=total, peak_event=index, live=dict(live))
    assert total == sum(block['requested_size'] for segment in final['segments'] for block in segment['blocks'] if block['state'] == 'active_allocated')
    rows = []
    for target in targets:
        entry, exit_point = target['entry'], target['exit']
        cached = entry['RGB_cache']['tensors']
        exit_cached = exit_point['RGB_cache']['tensors']
        allocations = []
        for address, value in target['live'].items():
            if value['allocated_event'] is None or value['allocated_event'] < entry['history_event_index_exclusive'] or value['bytes'] < 2**30:
                continue
            locations = []
            for frame in value['frames']:
                filename = frame['filename']
                if '/any-to-bokeh/' in filename or '/site-packages/diffusers/models/' in filename:
                    filename = filename.split('/any-to-bokeh/', 1)[-1] if '/any-to-bokeh/' in filename else filename.split('/site-packages/', 1)[-1]
                    locations.append(dict(file=filename, line=frame['line'], function=frame['name']))
            allocations.append(dict(requested_GiB=value['bytes']/2**30,
                allocated_event=value['allocated_event'], free_requested_event=value['free_requested_event'],
                code_locations=locations[:5],
                exit_RGB_cache_level=next((i for i,v in enumerate(exit_cached) if v['storage_pointer']==address), None)))
        rows.append(dict(stage=entry['stage'], native_unit=entry['native_unit'],
            stage_start_event=entry['history_event_index_exclusive'], stage_end_event=exit_point['history_event_index_exclusive'],
            requested_live_peak_event=target['peak_event'], requested_live_peak_GiB=target['peak']/2**30,
            incremental_requested_live_peak_above_entry_GiB=(target['peak']-entry['allocator_current']['requested_bytes.all.current']['bytes'])/2**30,
            measured_stage_allocated_peak_GiB=exit_point['stage_peak_allocated_GiB'],
            incremental_allocated_peak_above_stage_entry_GiB=exit_point['incremental_allocated_peak_above_stage_entry_GiB'],
            entry_RGB_cache_GiB=entry['RGB_cache']['GiB'],
            entry_RGB_cache_all_live_at_requested_peak=all(v['storage_pointer'] in target['live'] for v in cached),
            entry_RGB_cache_lifetimes=[dict(level=i, requested_GiB=target['live'][v['storage_pointer']]['bytes']/2**30,
                allocated_event=target['live'][v['storage_pointer']]['allocated_event'],
                free_requested_event=target['live'][v['storage_pointer']]['free_requested_event']) for i,v in enumerate(cached)],
            exit_RGB_cache_levels_live_at_requested_peak=[i for i,v in enumerate(exit_cached) if v['storage_pointer'] in target['live']],
            large_phase_allocations_at_requested_peak=allocations))
    evidence = dict(events=len(trace), max_entries=summary['allocation_history']['max_entries'],
        first_initial_snapshot_marker_preserved=True, target_interval_not_overwritten=True,
        synchronized_requested_byte_boundaries_verified=len(boundaries), final_requested_byte_state_verified=True,
        boundary='Trace allocation sizes are requested bytes, not necessarily actual allocator block sizes. Replay matches all synchronized requested-current observations; allocator absolute peaks come from phase measurement. Python allocation stack at resnet.py:258 identifies the add/divide expression, not a separate operator dispatch trace. Pre-history allocations and non-PyTorch CUDA memory are not fully attributed.',
        peak_evidence=rows)
    with (output/'memory-history-evidence.json').open('x') as file:
        json.dump(evidence, file, indent=2)
        file.write('\n')
    print('History replay verified', len(boundaries), 'boundaries and', len(rows), 'RGB/CoC stage calls')
    return evidence


class AllocatorObservation:
    FIELDS = ['allocated_bytes.all.current', 'reserved_bytes.all.current',
              'active_bytes.all.current', 'inactive_split_bytes.all.current',
              'requested_bytes.all.current']

    def __init__(self, profiler):
        self.profiler, self.points = profiler, []
        self.history_enabled = False
        self.backend = profiler.torch.cuda.memory.get_allocator_backend()
        names = {'PYTORCH_ALLOC_CONF', 'PYTORCH_CUDA_ALLOC_CONF', 'PYTORCH_NO_CUDA_MEMORY_CACHING'}
        names.update(k for k in os.environ if k.startswith('PYTORCH_') and 'ALLOC' in k)
        self.environment = {key: os.environ.get(key) for key in sorted(names)}

    def begin_history(self):
        cuda = self.profiler.torch.cuda
        cuda.synchronize()
        self.history_limit = 200000
        cuda.memory._record_memory_history(enabled='all', context='alloc', stacks='python', max_entries=self.history_limit)
        self.history_enabled = True
        self.initial_snapshot = cuda.memory._snapshot()

    def end_history(self, output):
        cuda = self.profiler.torch.cuda
        cuda.synchronize()
        snapshot = cuda.memory._snapshot()
        trace = snapshot['device_traces'][0]
        initial_trace = self.initial_snapshot['device_traces'][0]
        complete = len(trace) < self.history_limit and trace[:len(initial_trace)] == initial_trace and trace[0]['action'] == 'snapshot'
        for name, value in [('memory-initial-snapshot.pickle', self.initial_snapshot), ('memory-snapshot.pickle', snapshot)]:
            with (output/name).open('xb') as file:
                pickle.dump(value, file, protocol=pickle.HIGHEST_PROTOCOL)
        cuda.memory._record_memory_history(enabled=None)
        self.history_enabled = False
        assert complete, 'History event limit reached or beginning overwritten; do not infer missing lifetimes'
        return dict(max_entries=self.history_limit, events=len(trace), initial_events=len(initial_trace),
            initial_prefix_preserved=True, target_interval_not_overwritten=True,
            first_initial_snapshot_marker_preserved=True,
            allocator_settings=snapshot.get('allocator_settings'),
            initial_snapshot='memory-initial-snapshot.pickle', final_snapshot='memory-snapshot.pickle',
            scope='History only for one post-warmup native25-frame supplement. PyTorch allocator only; non-PyTorch CUDA allocations remain outside this evidence.')

    def module_storage(self):
        roots = {name: getattr(self.profiler.pipe, name) for name in ['unet', 'vae', 'image_encoder']}
        roots.update(vae_encoder=self.profiler.pipe.vae.encoder, vae_decoder=self.profiler.pipe.vae.decoder)
        modules, global_unique = {}, {}
        for name, module in roots.items():
            unique, kinds = {}, {}
            for kind, values in [('parameters', module.parameters()), ('buffers', module.buffers())]:
                storages, dtypes = {}, defaultdict(lambda: dict(tensors=0, logical_bytes=0, unique_storage_bytes=0))
                for value in values:
                    storage = value.untyped_storage()
                    key = (str(value.device), storage.data_ptr())
                    dtype = str(value.dtype)
                    dtypes[dtype]['tensors'] += 1
                    dtypes[dtype]['logical_bytes'] += value.numel()*value.element_size()
                    if key not in storages:
                        dtypes[dtype]['unique_storage_bytes'] += storage.nbytes()
                    storages[key] = storage.nbytes()
                    unique[key] = storage.nbytes()
                    if name in ['unet', 'vae', 'image_encoder']:
                        global_unique[key] = storage.nbytes()
                kinds[kind] = dict(by_dtype=dict(dtypes), unique_storage_bytes=sum(storages.values()))
            modules[name] = dict(**kinds, combined_unique_storage_bytes=sum(unique.values()),
                combined_unique_storage_GiB=sum(unique.values())/2**30,
                submodule_of='vae' if name.startswith('vae_') else None)
        return dict(modules=modules, three_root_union_unique_storage_bytes=sum(global_unique.values()),
            boundary='Parameters/buffers only, deduplicated by device/storage; VAE submodule totals overlap its parent. Not a full resident-allocation decomposition.')

    def capture(self, label, edge='checkpoint', unit=None, peaks=None):
        cuda = self.profiler.torch.cuda
        cuda.synchronize()
        stats = cuda.memory_stats()  # All five current fields use this one synchronized read.
        fields = {}
        for key in self.FIELDS:
            meaningful = key in stats and (self.backend == 'native' or key.startswith(('allocated_bytes', 'reserved_bytes')))
            fields[key] = dict(bytes=stats[key] if meaningful else None,
                GiB=stats[key]/2**30 if meaningful else None,
                applicability='supported' if meaningful else 'not_applicable_or_not_exposed')
        point = dict(index=len(self.points), scene='repro_004', repeat=0, native_unit=unit,
            stage=label, edge=edge, allocator_current=fields,
            allocator_allocations_total=stats.get('allocated.all.allocated'),
            allocator_frees_total=stats.get('allocated.all.freed'),
            RGB_cache=self.profiler.cache(), module_storage=self.module_storage())
        if peaks is not None:
            point.update(peaks)
        if self.history_enabled:
            snapshot = cuda.memory._snapshot()
            trace = snapshot['device_traces'][0]
            point.update(history_event_index_exclusive=len(trace), history_last_action=trace[-1]['action'] if trace else None)
        self.points.append(point)
        return point['index']


class NativeStages:
    def __init__(self, pipe):
        import torch
        self.torch, self.pipe = torch, pipe
        self.enabled = False
        self.allocator_observer = None
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
            allocator_entry = None
            if self.allocator_observer is not None:
                allocator_entry = self.allocator_observer.capture(actual, 'entry', len(self.rows)//5)
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
            allocator_exit = None
            if self.allocator_observer is not None:
                allocator_exit = self.allocator_observer.capture(actual, 'exit', len(self.rows)//5, dict(
                    stage_peak_allocated_GiB=peak/2**30, stage_peak_reserved_GiB=reserved_peak/2**30,
                    incremental_allocated_peak_above_stage_entry_GiB=max(0, peak-before)/2**30))
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
                allocator_entry_index=allocator_entry, allocator_exit_index=allocator_exit,
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
    parser.add_argument('--allocator-supplement', action='store_true',
        help='Only repro_004: one original warmup and one supplement observation, excluded from original timing statistics')
    parser.add_argument('--memory-history', action='store_true',
        help='Separate allocator supplement only; record the post-warmup clip with bounded allocation history')
    parser.add_argument('--analyze-history', action='store_true', help='Offline CPU replay of existing snapshots in output-dir; no inference')
    args = parser.parse_args()
    if args.analyze_history:
        analyze_memory_history(args.output_dir)
        return
    assert not args.memory_history or args.allocator_supplement, 'History is supplement-only'
    if args.allocator_supplement:
        args.repeats = 1
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
    if args.allocator_supplement:
        profiler.allocator_observer = AllocatorObservation(profiler)
        profiler.allocator_observer.capture('model_load_complete')
    runtime.update(gpu=torch.cuda.get_device_name(0), gpu_total_GiB=torch.cuda.get_device_properties(0).total_memory/2**30,
        driver=subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], text=True).strip(),
        CPU=next(line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),
        container_cpu_max=Path('/sys/fs/cgroup/cpu.max').read_text().strip(),
        container_memory_limit_bytes=int(Path('/sys/fs/cgroup/memory.max').read_text()),
        python=sys.version, torch_threads=torch.get_num_threads(),
        project_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        native_code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT.parent/'any-to-bokeh', text=True).strip())
    runs, warmups, history = [], [], None
    try:
        selected = list(zip(config['scenes'], config['K']))
        if args.allocator_supplement:
            selected = selected[:1]
        for sid, K in selected:
            csvpath = out / (sid + ".csv")
            folder = ROOT / config["input"] / sid
            import csv
            with csvpath.open("x") as file:
                csv.writer(file).writerows([["aif_folder", "disp_folder", "k"], [str(folder / "rgb"), str(folder / "disparity"), K]])
            frames = prepare("any2bokeh", state, folder, K, csvpath, 25)
            if args.allocator_supplement:
                profiler.allocator_observer.capture('input_prepare_complete')
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
            if args.allocator_supplement:
                profiler.allocator_observer.capture('warmup_complete')
            if args.memory_history:
                profiler.allocator_observer.begin_history()
            for repeat in range(args.repeats):
                assert set(gpu_processes()) <= {os.getpid()}, "Foreign GPU process appeared"
                profiler.reset()
                profiler.enabled = True
                state["meter"].reset()
                outputs = infer("any2bokeh", state, frames)
                if args.allocator_supplement:
                    profiler.allocator_observer.capture('clip_complete')
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
        if args.memory_history:
            history = profiler.allocator_observer.end_history(out)
    finally:
        if args.memory_history and profiler.allocator_observer.history_enabled:
            torch.cuda.memory._record_memory_history(enabled=None)
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
    if args.allocator_supplement:
        observer = profiler.allocator_observer
        summary.update(status='ATB_ALLOCATOR_SUPPLEMENT_COMPLETE',
            source_commit='30382b0', source_record='03_实验记录/records/any2bokeh-native-module-profile-4090d-20261001.json',
            included_in_original_eight_run_timing_statistics=False,
            allocator=dict(backend=observer.backend,environment=observer.environment,points=observer.points,
                boundary='Synchronized entry/exit current fields from one memory_stats call per point; not allocator composition at peak. No allocator setting changes.'))
        if history is not None:
            summary['allocation_history'] = history
    (out / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps({k:v for k,v in summary.items() if k not in ["runtime", "runs", "allocator"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
