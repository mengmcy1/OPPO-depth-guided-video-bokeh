"""Measure the frozen Track A adapters only after DAVIS closure, on an exclusive device."""
import argparse
from collections import Counter, defaultdict
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
METHODS = ['any2bokeh', 'mpib', 'bokehme', 'drbokeh', 'bokehdiff', 'deeplens']


def read(path): return json.loads(path.read_text())


def gpu_processes():
    return [int(x.strip()) for x in subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).splitlines() if x.strip()]


def assert_idle_gpu():
    # A container may see no compute PID even while the device reports activity.
    # Do not label that situation as an exclusive idle timing environment.
    # NVML's utilization sample can still cover the preceding method's final
    # kernels after its process exits. Require three idle samples within30s.
    consecutive = 0
    for i in range(30):
        assert not gpu_processes(), 'Wait for an idle GPU; do not stop other processes'
        utilization = int(subprocess.check_output(['nvidia-smi', '--id=0', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip())
        consecutive = consecutive + 1 if utilization <= 5 else 0
        if consecutive == 3: return
        time.sleep(1)
    raise AssertionError(f'GPU utilization {utilization}% before timing; idle state not established within30s')


class CoreMeter:
    """Nonoverlapping named model regions; transport remains in outer E2E."""
    def __init__(self, cpu=False):
        self.cpu = cpu
        self.enabled = True
        if not cpu:
            import torch
            self.torch = torch
        self.reset()

    def sync(self):
        if not self.cpu: self.torch.cuda.synchronize()

    def reset(self):
        self.seconds = 0.; self.regions = defaultdict(float); self.calls = Counter(); self.units = []

    def call(self, label, function, *args, **kwargs):
        if not self.enabled: return function(*args, **kwargs)
        self.sync(); start = time.perf_counter()
        result = function(*args, **kwargs)
        self.sync(); elapsed = time.perf_counter() - start
        self.seconds += elapsed; self.regions[label] += elapsed; self.calls[label] += 1
        return result

    def wrap(self, obj, attribute, label):
        original = getattr(obj, attribute)
        def measured(*args, **kwargs): return self.call(label, original, *args, **kwargs)
        setattr(obj, attribute, measured)

    def begin_unit(self):
        self.unit_start = self.seconds
        if self.enabled and not self.cpu: self.torch.cuda.reset_peak_memory_stats()

    def end_unit(self, frames):
        if not self.enabled: return
        self.units.append(dict(output_frames=frames, core_seconds=self.seconds - self.unit_start,
            torch_peak_allocated_GiB=None if self.cpu else self.torch.cuda.max_memory_allocated()/2**30,
            torch_peak_reserved_GiB=None if self.cpu else self.torch.cuda.max_memory_reserved()/2**30))


class ParticipatingParameters:
    """Learned parameters owned by neural modules observed in actual forwards."""
    def __init__(self, roots):
        self.parameters = {}; self.module_names = set(); self.handles = []; self.seen = set()
        for label, root in roots.items():
            for name, module in root.named_modules():
                direct = tuple(module.parameters(recurse=False))
                if not direct: continue
                def track(mod, args, label=label, name=name, direct=direct):
                    if id(mod) in self.seen: return
                    self.seen.add(id(mod)); self.module_names.add(label + '.' + name)
                    for parameter in direct: self.parameters[id(parameter)] = parameter.numel()
                self.handles.append(module.register_forward_pre_hook(track))

    def summary(self):
        return dict(active_parameters=sum(self.parameters.values()), modules=sorted(self.module_names),
            convention='Parameters of invoked neural modules, unique tensors; buffers and fixed renderer kernels excluded. Not restricted to requires_grad=True.')


def load(method, c):
    """Reuse the existing adapters; no checkpoint/control changes for timing."""
    if method == 'deeplens':
        from run_deeplens_track_a import native_model
        cfg = read(ROOT / 'configs/deeplens-track-a-frozen.json')
        model = native_model(cfg)
        from baselines.deeplens import tf
        with model.graph.as_default(): count = sum(int(np.prod(v.shape.as_list())) for v in tf.trainable_variables())
        meter = CoreMeter(cpu=True); meter.wrap(model.session, 'run', 'TF1_LensBlur_singleSR')
        return dict(model=model, cfg=cfg, meter=meter), dict(active_parameters=count, device='CPU', runtime=model.audit)
    import torch
    torch.set_num_threads(c['torch_threads']); torch.manual_seed(42 if method == 'bokehdiff' else 0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cudnn.benchmark = False
    if method == 'mpib':
        from run_mpib_recovery import setup
        nets, counts = setup(); state = dict(nets=nets); active = counts
        roots = {k: v for k, v in nets.items() if k != 'renderer'}
    elif method == 'bokehme':
        from baselines.bokehme import setup
        cfg = read(ROOT / 'configs/bokehme-track-a-frozen.json')
        pipe, args, nets, active = setup(ROOT.parent / 'BokehMe', ROOT / 'models/BokehMe', cfg)
        state = dict(pipe=pipe, args=args, nets=nets)
        roots = {k: v for k, v in nets.items() if k != 'classical'}
    elif method == 'drbokeh':
        from baselines.drbokeh import OfficialDrBokeh
        cfg = read(ROOT / 'configs/drbokeh-track-a-frozen.json')
        model = OfficialDrBokeh(ROOT.parent / 'DrBokeh', cfg, instrument=False)
        state = dict(model=model, cfg=cfg)
        active = dict(LaMa=sum(p.numel() for p in model.prep.rgb_inpainting_inferencer.model.parameters()))
        roots = dict(LaMa=model.prep.rgb_inpainting_inferencer.model)
    elif method == 'bokehdiff':
        from accelerate.utils import set_seed
        from baselines.bokehdiff import OfficialBokehDiff
        set_seed(42); model = OfficialBokehDiff(ROOT.parent / 'BokehDiff', ROOT / 'models/BokehDiff', instrument=False)
        state = dict(model=model)
        active = {k: sum(p.numel() for p in getattr(model.pipeline, k).parameters()) for k in ['unet', 'vae', 'text_encoder', 'text_encoder_2']}
        roots = {k: getattr(model.pipeline, k) for k in active}
    else:
        os.environ.setdefault('HF_HOME', str(ROOT / 'models/huggingface'))
        sys.path.insert(0, str(ROOT.parent / 'any-to-bokeh'))
        from models.unet import UNetSpatioTemporalConditionModel
        from models.vae import AutoencoderKLTemporalDecoder
        from pipelines.any2bokeh_pipe import StableVideoDiffusionPipeline
        cp = ROOT / 'models/any-to-bokeh/checkpoints'
        unet = UNetSpatioTemporalConditionModel.from_pretrained(str(cp), subfolder='unet', torch_dtype=torch.float16, low_cpu_mem_usage=False, local_files_only=True)
        vae = AutoencoderKLTemporalDecoder.from_pretrained(str(cp / 'vae'), torch_dtype=torch.float16, local_files_only=True)
        pipe = StableVideoDiffusionPipeline.from_pretrained('stabilityai/stable-video-diffusion-img2vid-xt', unet=unet, vae=vae, torch_dtype=torch.float16, variant='fp16', low_cpu_mem_usage=False, local_files_only=True).to('cuda:0')
        pipe.set_progress_bar_config(disable=True); state = dict(pipe=pipe)
        active = {k: sum(p.numel() for p in getattr(pipe, k).parameters()) for k in ['unet', 'vae', 'image_encoder']}
        roots = {k: getattr(pipe, k) for k in active}
    meter = CoreMeter()
    if method == 'mpib':
        for key, net in state['nets'].items(): meter.wrap(net, 'forward', key)
    elif method == 'drbokeh':
        meter.wrap(model.prep.rgb_inpainting_inferencer.model, 'forward', 'LaMa')
        meter.wrap(model.model.render_model, 'forward', 'Multi_Layer_Renderer')
    state.update(meter=meter, parameter_tracker=ParticipatingParameters(roots))
    return state, dict(loaded_learned_parameters=sum(active.values()), loaded_parameter_components=active, device='GPU0', torch=torch.__version__, CUDA=torch.version.cuda)


def prepare(method, state, folder, K, csvpath, frame_limit=25):
    if method == 'any2bokeh':
        from utils.dataset import BokehCocMPIDatasetWholeVid, collate_fn_val_coc_mpi_whole_video
        ds = BokehCocMPIDatasetWholeVid(str(csvpath), sample_size=(576, 1024))
        return collate_fn_val_coc_mpi_whole_video([ds[0]])
    frames = []
    for dp in sorted((folder / 'disparity').glob('*.png'))[:frame_limit]:
        t = dp.stem.split('_')[0]; focus = float(dp.stem.split('_zf_')[1])
        rgb = cv2.imread(str(folder / 'rgb' / (t + '.png')))
        if method != 'deeplens': rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = rgb.astype(np.float32) / 255
        d = cv2.imread(str(dp), 0).astype(np.float64) / 255
        if method == 'mpib': item = (rgb, d.astype(np.float32), focus, K)
        elif method == 'bokehme':
            from baselines.bokehme import defocus_input
            control, _ = defocus_input(d.astype(np.float32), focus, K, 'normalized', state['args'].defocus_scale)
            item = (rgb, control)
        elif method == 'drbokeh':
            from baselines.drbokeh import map_geometry
            dn, fn, kn, _ = map_geometry(d, focus, K)
            alpha = cv2.imread(str(folder / 'alpha' / (t + '.png')), 0).astype(np.float32)[..., None] / 255
            item = (rgb, dn, fn, kn, alpha)
        elif method == 'bokehdiff':
            from baselines.bokehdiff import map_controls
            dn, fn, _ = map_controls(d, focus, K, 'normalized'); item = (rgb, dn, fn, K)
        else:
            from baselines.deeplens import map_geometry
            dn, fn, ap, _ = map_geometry(d, focus, K, state['cfg']['response_scale'])
            item = (cv2.resize(rgb, (512, 288), interpolation=cv2.INTER_AREA), rgb, dn, fn, ap)
        frames.append(item)
    assert len(frames) == frame_limit
    return frames


def infer(method, state, frames):
    outputs = []; meter = state['meter']
    if method == 'any2bokeh':
        import torch
        batch = frames; cache = None
        for i in range(0, 6, 2):
            meter.begin_unit()
            x = {k: batch[k][i:i+2].to('cuda', dtype=torch.float32 if k in ['pixel_values', 'coc'] else torch.float16) for k in ['pixel_values', 'coc', 'k', 'mpi_mask', 'weight']}
            with torch.inference_mode():
                result, cache = meter.call('ATB_video_pipeline', state['pipe'], x['coc'], x['pixel_values'], input_k=x['k'], mpi_mask=x['mpi_mask'], weight=x['weight'], latent_cache=cache, is_last_batch=i == 4, overlap_frame=4, height=576, width=1024, num_frames=8, decode_chunk_size=8, motion_bucket_id=127, fps=7, noise_aug_strength=0., num_inference_steps=1, generator=torch.Generator(device='cpu').manual_seed(0), batch_size=2, output_type='pt')
            assert result[0].device.type == 'cuda'
            n = len(result[0]); outputs.extend(v.permute(1, 2, 0).detach().cpu().numpy() for v in result[0])
            meter.end_unit(n)
            del x, result
        assert len(outputs) == 25
        return outputs
    for item in frames:
        meter.begin_unit()
        if method == 'mpib':
            import torch
            from run_mpib_recovery import predict_layers
            rgb, d, focus, K = item
            with torch.inference_mode():
                layers, alpha = predict_layers(rgb, d, K, state['nets'])
                result = state['nets']['renderer'](layers, alpha, K, focus, 2.2, norm=True, increase_focus_range=True)
            value = result[0].permute(1, 2, 0).detach().cpu().numpy()
            del result, layers, alpha
        elif method == 'bokehme':
            import torch
            rgb, defocus = item
            image = torch.from_numpy(rgb).permute(2, 0, 1)[None].cuda()
            control = torch.from_numpy(defocus)[None, None].cuda()
            with torch.inference_mode():
                values = meter.call('BokehMe_demo_pipeline', state['pipe'], state['nets']['classical'], state['nets']['arnet'], state['nets']['iunet'], image, control, state['args'].gamma, state['args'])
            value = values[0][0].permute(1, 2, 0).detach().cpu().numpy()
            del image, control, values
        elif method == 'drbokeh':
            import torch
            rgb, dn, fn, kn, alpha = item; cfg = state['cfg']
            params = dict(cfg['preprocessing'], focal=fn, lens_effect=kn, gamma=cfg['gamma'], offset=cfg['offset'])
            with torch.inference_mode():
                value = state['model'].model.inference(rgb, dn[..., None], alpha, params)['bokeh']
        elif method == 'bokehdiff':
            import torch
            rgb, dn, fn, K = item; pipe = state['model'].pipeline
            image = torch.from_numpy(rgb).permute(2, 0, 1)[None].cuda() * 2 - 1
            d = torch.from_numpy(dn)[None, None].cuda()
            control = torch.cat([d, (d - fn).abs() * (K/10)], dim=1)
            def public_tensor_path():
                latents = pipe.vae.encode(image).latent_dist.mode().detach() * pipe.vae.config.scaling_factor
                return pipe('an excellent photo with a large aperture', height=576, width=1024,
                    generator=torch.Generator(device='cuda:0').manual_seed(42), guidance_scale=1,
                    return_dict=False, timesteps=[499], cross_attention_kwargs={'disp_coc': control},
                    output_type='pt', latents=latents, TILE_SIZE=64)[0]
            with torch.inference_mode(): result = meter.call('BokehDiff_VAE_TiledSDXL', public_tensor_path)
            assert result.device.type == 'cuda'
            value = result[0].permute(1, 2, 0).detach().cpu().numpy()
            del image, d, control, result
        else:
            value, _ = state['model'].predict(*item); value = value[..., ::-1]
        outputs.append(value); meter.end_unit(1)
    return outputs


def flop_diagnostic(method, state, c, out):
    # Runs after all latency samples, with instrumentation excluded from timing.
    if method == 'deeplens': return dict(status='CPU_NOT_IN_TABLE3A')
    import torch
    from torch.utils.flop_counter import FlopCounterMode
    from torch.utils._python_dispatch import TorchDispatchMode
    rows = []; state['meter'].enabled = False
    class Operators(TorchDispatchMode):
        def __init__(self): super().__init__(); self.calls = Counter()
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            self.calls[str(func)] += 1
            return func(*args, **(kwargs or {}))
    for sid, K in zip(c['scenes'], c['K']):
        frames = prepare(method, state, ROOT/c['input']/sid, K, out/(sid+'.csv'), 25 if method=='any2bokeh' else 1)
        try:
            counter = FlopCounterMode(display=False); operators = Operators()
            with counter, operators: values = infer(method, state, frames)
            torch.cuda.synchronize()
            counts = counter.get_flop_counts().get('Global', {})
            rows.append(dict(scene=sid, K=K, output_frames=len(values),
                counted_GFLOPs=float(counter.get_total_flops()/1e9),
                counted_GFLOPs_per_output_frame=float(counter.get_total_flops()/1e9/len(values)),
                counted_operators={str(k):int(v) for k,v in counts.items()}, observed_operators=dict(operators.calls)))
            del values
        except Exception as error:
            rows.append(dict(scene=sid, K=K, profiler_error=type(error).__name__+': '+str(error)))
        del frames
    state['meter'].enabled = True
    return dict(status='PARTIAL_OPERATOR_COUNT_NOT_FULL_PIPELINE_GFLOPS', reported_GFLOPs=None,
        reason=('Custom CUDA/CuPy scatter and dynamic renderer operations are outside the dispatcher counter.' if method in ['drbokeh','mpib','bokehme'] else
                'Dispatcher counter covers convolution/matrix products/attention formulas, not every executed floating-point normalization, elementwise, resampling and stochastic/control operation. Partial count is diagnostic only.'),
        counter='torch.utils.flop_counter.FlopCounterMode; fused multiply-add counts2; measured on the actual frozen path',rows=rows)


def worker(method, c):
    assert_idle_gpu()
    cpu = method == 'deeplens'
    expected_frames = 1 if c['table']=='3a' and method!='any2bokeh' else 25
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ('' if cpu else '0')
    cv2.setNumThreads(2)
    out = ROOT/c['output']/method; out.mkdir(parents=True, exist_ok=False)
    state, runtime = load(method, c); meter = state['meter']
    runtime['project_commit'] = subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'], text=True).strip()
    runtime['gpu_hardware'] = subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'], text=True).strip()
    runtime['cpu_model'] = next(line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name'))
    runtime['cpu_quota'] = Path('/sys/fs/cgroup/cpu.max').read_text().strip()
    runtime['memory_quota_bytes'] = Path('/sys/fs/cgroup/memory.max').read_text().strip()
    runtime['core_region_definition'] = c['core_regions'][method]
    (out/'protocol.json').write_text(json.dumps(c,indent=2)+'\n')
    (out/'runtime.json').write_text(json.dumps(runtime,indent=2)+'\n')
    samples = []; monitor_errors = []; stop = threading.Event(); page_bytes = os.sysconf('SC_PAGE_SIZE')
    def poll():
        while not stop.is_set():
            try:
                rss = int(Path('/proc/self/statm').read_text().split()[1])*page_bytes/2**20
                gpu = None if cpu else int(subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.used','--format=csv,noheader,nounits'], text=True).strip())
                samples.append(dict(rss_MiB=rss, gpu_MiB=gpu))
            except Exception as error: monitor_errors.append(str(error)); return
            stop.wait(.2)
    monitor = threading.Thread(target=poll,daemon=True); monitor.start()
    rows = []; warmups = []
    try:
        for sid,K in zip(c['scenes'],c['K']):
            folder=ROOT/c['input']/sid; csvpath=out/(sid+'.csv')
            with csvpath.open('w') as f: csv.writer(f).writerows([['aif_folder','disp_folder','k'],[str(folder/'rgb'),str(folder/'disparity'),K]])
            for repeat in range(-c['warmup_clips_per_scene'],c['repeats_per_scene']):
                assert set(gpu_processes()) <= ({os.getpid()} if not cpu else set()), 'Foreign GPU process appeared; discard timing attempt'
                meter.reset(); memory_start=len(samples)
                meter.sync(); start=time.perf_counter()
                prepared=prepare(method,state,folder,K,csvpath,expected_frames)
                values=infer(method,state,prepared)
                pixels=[np.uint8(np.round(np.clip(v,0,1)*255)) for v in values]
                meter.sync(); seconds=time.perf_counter()-start
                assert len(values)==expected_frames and all(v.shape==(576,1024,3) and np.isfinite(v).all() for v in values)
                assert all(v.max()>v.min() for v in pixels)
                assert sum(x['output_frames'] for x in meter.units)==expected_frames and 0<meter.seconds<=seconds
                assert not monitor_errors, monitor_errors
                observed=samples[memory_start:]
                # Very fast single-frame runs may fit between monitor samples.
                # Add an endpoint sample outside all timers; keep the limitation explicit.
                if not observed:
                    rss=int(Path('/proc/self/statm').read_text().split()[1])*page_bytes/2**20
                    gpu=None if cpu else int(subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
                    observed=[dict(rss_MiB=rss,gpu_MiB=gpu)]
                row=dict(scene=sid,K=K,repeat=repeat,output_frames=expected_frames,core_seconds=meter.seconds,E2E_given_inputs_seconds=seconds,
                    core_regions_seconds=dict(meter.regions), core_region_calls=dict(meter.calls), native_units=meter.units,
                    peak_torch_allocated_GiB=None if cpu else max(x['torch_peak_allocated_GiB'] for x in meter.units),
                    peak_torch_reserved_GiB=None if cpu else max(x['torch_peak_reserved_GiB'] for x in meter.units),
                    peak_device_memory_MiB_sampled=None if cpu else max(x['gpu_MiB'] for x in observed),
                    peak_process_RSS_MiB_sampled=max(x['rss_MiB'] for x in observed))
                if repeat>=0:
                    rows.append(row); (out/f'{sid}-repeat{repeat}.json').write_text(json.dumps(row,indent=2)+'\n')
                else: warmups.append(row)
                print('TIMING',method,sid,repeat,round(meter.seconds,3),round(seconds,3),flush=True)
                del prepared,values,pixels
        assert set(gpu_processes()) <= ({os.getpid()} if not cpu else set())
    finally:
        stop.set(); monitor.join()
    (out/'warmups.json').write_text(json.dumps(warmups,indent=2)+'\n')
    # Repeated hardware timing can reuse the frozen complexity evidence.
    previous = None
    if c.get('reuse_complexity_record'):
        previous = read(ROOT / c['reuse_complexity_record'])['methods'][method]
        flops = dict(status=previous['Table3A']['GFLOPs_status'],
            reported_GFLOPs=previous['Table3A']['GFLOPs_per_output_frame'],
            source=c['reuse_complexity_record'],
            reason='Frozen model and coverage convention reused; no new FLOP diagnostic pass.')
    else:
        # Profiling is a separate pass, so its overhead never enters measured rows.
        flops=flop_diagnostic(method,state,c,out)
    (out/'flop-coverage.json').write_text(json.dumps(flops,indent=2)+'\n')
    parameters=dict(active_parameters=runtime['active_parameters']) if cpu else state['parameter_tracker'].summary()
    if previous is not None:
        assert parameters['active_parameters'] == previous['parameters']['active_parameters'], 'Frozen invoked model parameters changed'
        parameters = dict(previous['parameters'], reused_from=c['reuse_complexity_record'])
    result=dict(method=method,status='STANDARDIZED_TIMING_COMPLETE',runtime=runtime,parameters=parameters,rows=rows,
        Core_seconds_mean=float(np.mean([r['core_seconds'] for r in rows])),
        Core_seconds_std=float(np.std([r['core_seconds'] for r in rows],ddof=1)),
        E2E_given_inputs_seconds_mean=float(np.mean([r['E2E_given_inputs_seconds'] for r in rows])),
        E2E_given_inputs_seconds_std=float(np.std([r['E2E_given_inputs_seconds'] for r in rows],ddof=1)),
        peak_device_memory_MiB_sampled=None if cpu else max(r['peak_device_memory_MiB_sampled'] for r in rows),
        peak_process_RSS_MiB_sampled=max(r['peak_process_RSS_MiB_sampled'] for r in rows),memory_sampling_seconds=.2,
        by_K={str(K):{key:dict(mean=float(np.mean([r[key] for r in rows if r['K']==K])),std=float(np.std([r[key] for r in rows if r['K']==K],ddof=1))) for key in ['core_seconds','E2E_given_inputs_seconds']} for K in c['K']},
        Table3A=None if cpu else dict(Time_per_output_frame_seconds=float(np.mean([r['core_seconds']/r['output_frames'] for r in rows])),
            Params_M=parameters['active_parameters']/1e6,GFLOPs_per_output_frame=None,GFLOPs_status=flops['status'],
            Peak_device_memory_GiB_sampled=max(r['peak_device_memory_MiB_sampled'] for r in rows)/1024,
            Peak_torch_allocated_GiB=max(r['peak_torch_allocated_GiB'] for r in rows),
            scope='Same measured native model units, output-frame weighted; excludes transfer/adapter portions included in Table3B E2E. See per-method Core definition.'))
    result['measured_table']=c['table']; result['frames_per_execution']=expected_frames
    result['FPS_E2E']=None if c['table']=='3a' else 25/result['E2E_given_inputs_seconds_mean']
    if c['table']=='3a': result['E2E_scope']='Unit-pipeline diagnostic only; not a25-frame Table3B result'
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print('METHOD_TIMING_COMPLETE',method,flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--method', choices=METHODS); parser.add_argument('--all', action='store_true'); parser.add_argument('--prepare-only', action='store_true'); parser.add_argument('--table',choices=['3a','3b'],default='3b')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--logs-dir', type=Path)
    parser.add_argument('--record', type=Path)
    parser.add_argument('--reuse-complexity-record', type=Path)
    args = parser.parse_args(); c = read(ROOT / 'configs/efficiency-track-a.json')
    c['table']=args.table
    if args.table=='3a':
        c.update(output='outputs/efficiency-3a-20260930',logs='logs/efficiency-3a-20260930',record='03_实验记录/records/efficiency-3a-20260930.json',
            selection='First frame of each of the fixed4 clips for frame methods; full25-frame native path for Any-to-Bokeh. Selected before timing; no independent one-frame ATB entry.')
    for argument, key in [('output_dir', 'output'), ('logs_dir', 'logs'), ('record', 'record'), ('reuse_complexity_record', 'reuse_complexity_record')]:
        value = getattr(args, argument)
        if value is not None: c[key] = str(value)
    if args.all and args.output_dir is not None:
        assert args.logs_dir is not None and args.record is not None, 'Provide isolated log and record destinations with --all'
    methods=['drbokeh','bokehme','mpib','bokehdiff','any2bokeh'] if args.table=='3a' else METHODS
    scenes = {s['id']: s for s in read(ROOT / c['input'] / 'config.json')['scenes']}
    for sid, K in zip(c['scenes'], c['K']):
        assert scenes[sid]['K'] == K
        for field in ['rgb', 'disparity', 'alpha']:
            assert len(list((ROOT / c['input'] / sid / field).glob('*.png'))) == 25
    if args.prepare_only:
        print('TIMING_INPUTS_AND_PROTOCOL_READY; no inference or timing performed'); return
    final = read(ROOT / '03_实验记录/records/davis-final-review-20260929.json')
    assert final['status'] == c['gate']
    if args.all:
        assert not args.method
        python = dict(any2bokeh='/root/miniconda3/envs/any2bokeh/bin/python', bokehdiff='/root/autodl-tmp/envs/bokehdiff/bin/python', deeplens='/root/autodl-tmp/envs/deeplens-tf1-cpu/bin/python')
        logs = ROOT / c['logs']; logs.mkdir(parents=True, exist_ok=True)
        for method in methods:
            summary = ROOT / c['output'] / method / 'summary.json'
            if summary.exists():
                assert read(summary)['status'] == 'STANDARDIZED_TIMING_COMPLETE'
                assert read(summary.parent / 'protocol.json') == c
                continue
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='' if method == 'deeplens' else '0', OMP_NUM_THREADS='8' if method == 'deeplens' else '4', OPENBLAS_NUM_THREADS='1', TF_ENABLE_ONEDNN_OPTS='0')
            command = [python.get(method, '/root/miniconda3/envs/magicbokeh/bin/python'), str(Path(__file__).resolve()), '--method', method, '--table', args.table]
            for flag, value in [('--output-dir', args.output_dir), ('--logs-dir', args.logs_dir), ('--record', args.record), ('--reuse-complexity-record', args.reuse_complexity_record)]:
                if value is not None: command.extend([flag, str(value)])
            with (logs / f'{method}.log').open('a') as log:
                subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        result = dict(status='TABLE3A_FIVE_GPU_METHODS_COMPLETE_REVIEW_PENDING' if args.table=='3a' else 'TABLE3B_SIX_METHODS_COMPLETE_REVIEW_PENDING', protocol=c,
            methods={m: read(ROOT / c['output'] / m / 'summary.json') for m in methods})
        from prepare_davis_v2 import write_new
        write_new(ROOT / c['record'], result)
        print('TABLE_TIMING_COMPLETE',args.table,flush=True); return
    assert args.method in methods
    worker(args.method, c)


if __name__ == '__main__': main()
