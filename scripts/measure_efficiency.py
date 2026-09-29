"""Measure the frozen Track A adapters only after DAVIS closure, on an exclusive device."""
import argparse
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


def load(method, c):
    """Reuse the existing adapters; no checkpoint/control changes for timing."""
    if method == 'deeplens':
        from run_deeplens_track_a import native_model
        cfg = read(ROOT / 'configs/deeplens-track-a-frozen.json')
        model = native_model(cfg)
        from baselines.deeplens import tf
        with model.graph.as_default(): count = sum(int(np.prod(v.shape.as_list())) for v in tf.trainable_variables())
        return dict(model=model, cfg=cfg), dict(active_parameters=count, device='CPU', runtime=model.audit)
    import torch
    torch.set_num_threads(c['torch_threads']); torch.manual_seed(42 if method == 'bokehdiff' else 0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cudnn.benchmark = False
    if method == 'mpib':
        from run_mpib_recovery import setup
        nets, counts = setup(); state = dict(nets=nets); active = counts
    elif method == 'bokehme':
        from baselines.bokehme import setup
        cfg = read(ROOT / 'configs/bokehme-track-a-frozen.json')
        pipe, args, nets, active = setup(ROOT.parent / 'BokehMe', ROOT / 'models/BokehMe', cfg)
        state = dict(pipe=pipe, args=args, nets=nets)
    elif method == 'drbokeh':
        from baselines.drbokeh import OfficialDrBokeh
        cfg = read(ROOT / 'configs/drbokeh-track-a-frozen.json')
        model = OfficialDrBokeh(ROOT.parent / 'DrBokeh', cfg)
        state = dict(model=model, cfg=cfg)
        active = dict(LaMa=sum(p.numel() for p in model.prep.rgb_inpainting_inferencer.model.parameters()))
    elif method == 'bokehdiff':
        from accelerate.utils import set_seed
        from baselines.bokehdiff import OfficialBokehDiff
        set_seed(42); model = OfficialBokehDiff(ROOT.parent / 'BokehDiff', ROOT / 'models/BokehDiff')
        state = dict(model=model)
        active = {k: sum(p.numel() for p in getattr(model.pipeline, k).parameters()) for k in ['unet', 'vae', 'text_encoder', 'text_encoder_2']}
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
    return state, dict(active_parameters=sum(active.values()), parameter_components=active, device='GPU0', torch=torch.__version__, CUDA=torch.version.cuda)


def prepare(method, state, folder, K, csvpath):
    if method == 'any2bokeh':
        from utils.dataset import BokehCocMPIDatasetWholeVid, collate_fn_val_coc_mpi_whole_video
        ds = BokehCocMPIDatasetWholeVid(str(csvpath), sample_size=(576, 1024))
        return collate_fn_val_coc_mpi_whole_video([ds[0]])
    frames = []
    for dp in sorted((folder / 'disparity').glob('*.png')):
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
    assert len(frames) == 25
    return frames


def infer(method, state, frames):
    outputs = []
    if method == 'any2bokeh':
        import torch
        batch = frames; cache = None
        for i in range(0, 6, 2):
            x = {k: batch[k][i:i+2].to('cuda', dtype=torch.float32 if k in ['pixel_values', 'coc'] else torch.float16) for k in ['pixel_values', 'coc', 'k', 'mpi_mask', 'weight']}
            with torch.inference_mode():
                result, cache = state['pipe'](x['coc'], x['pixel_values'], input_k=x['k'], mpi_mask=x['mpi_mask'], weight=x['weight'], latent_cache=cache, is_last_batch=i == 4, overlap_frame=4, height=576, width=1024, num_frames=8, decode_chunk_size=8, motion_bucket_id=127, fps=7, noise_aug_strength=0., num_inference_steps=1, generator=torch.Generator(device='cpu').manual_seed(0), batch_size=2, output_type='pt')
            outputs.extend(v.permute(1, 2, 0).detach().cpu().numpy() for v in result[0])
        assert len(outputs) == 25
        return outputs
    for item in frames:
        if method == 'mpib':
            import torch
            from run_mpib_recovery import predict_layers
            rgb, d, focus, K = item
            with torch.inference_mode():
                layers, alpha = predict_layers(rgb, d, K, state['nets'])
                result = state['nets']['renderer'](layers, alpha, K, focus, 2.2, norm=True, increase_focus_range=True)
            value = result[0].permute(1, 2, 0).detach().cpu().numpy()
        elif method == 'bokehme':
            from baselines.bokehme import predict
            value, _ = predict(state['pipe'], state['args'], state['nets'], *item)
        elif method == 'drbokeh': value, _, _ = state['model'].predict(*item[:4], alpha_override=item[4])
        elif method == 'bokehdiff':
            state['model'].predict(*item, 42, reset_global_seed=False, quantization='round_png8')
            value = state['model'].output_float
        else:
            value, _ = state['model'].predict(*item); value = value[..., ::-1]
        outputs.append(value)
    return outputs


def worker(method, c):
    assert not gpu_processes(), 'Wait for an idle GPU; do not stop other processes'
    cpu = method == 'deeplens'
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ('' if cpu else '0')
    if not cpu: import torch
    sync = (lambda: None) if cpu else torch.cuda.synchronize
    cv2.setNumThreads(2)
    out = ROOT / c['output'] / method; out.mkdir(parents=True, exist_ok=False)
    state, runtime = load(method, c)
    (out / 'protocol.json').write_text(json.dumps(c, indent=2) + '\n')
    (out / 'runtime.json').write_text(json.dumps(runtime, indent=2) + '\n')
    counts = []; stop = threading.Event()
    def poll():
        while not stop.is_set():
            counts.append(int(subprocess.check_output(['nvidia-smi', '--id=0', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip()))
            stop.wait(.2)
    if not cpu: monitor = threading.Thread(target=poll, daemon=True); monitor.start()
    rows = []
    try:
        for sid, K in zip(c['scenes'], c['K']):
            folder = ROOT / c['input'] / sid; csvpath = out / (sid + '.csv')
            with csvpath.open('w') as f: csv.writer(f).writerows([['aif_folder', 'disp_folder', 'k'], [str(folder / 'rgb'), str(folder / 'disparity'), K]])
            for repeat in range(-c['warmup_clips_per_scene'], c['repeats_per_scene']):
                assert set(gpu_processes()) <= ({os.getpid()} if not cpu else set()), 'Foreign GPU process appeared; discard this timing run'
                if not cpu: torch.cuda.reset_peak_memory_stats()
                memory_start = len(counts)
                sync(); start = time.perf_counter()
                prepared = prepare(method, state, folder, K, csvpath)
                sync(); core_start = time.perf_counter()
                values = infer(method, state, prepared)
                sync(); core_seconds = time.perf_counter() - core_start
                pixels = [np.uint8(np.round(np.clip(v, 0, 1) * 255)) for v in values]
                sync(); seconds = time.perf_counter() - start
                assert len(values) == 25 and all(v.shape == (576, 1024, 3) and np.isfinite(v).all() for v in values)
                assert all(v.max() > v.min() for v in pixels)
                if repeat >= 0:
                    row = dict(scene=sid, K=K, repeat=repeat, core_seconds=core_seconds, E2E_given_inputs_seconds=seconds,
                        peak_torch_allocated_GiB=None if cpu else torch.cuda.max_memory_allocated() / 2**30,
                        peak_torch_reserved_GiB=None if cpu else torch.cuda.max_memory_reserved() / 2**30,
                        peak_device_memory_MiB_sampled=None if cpu else max(counts[memory_start:]))
                    rows.append(row); (out / f'{sid}-repeat{repeat}.json').write_text(json.dumps(row, indent=2) + '\n')
                print('TIMING', method, sid, repeat, round(core_seconds, 3), round(seconds, 3), flush=True)
                del prepared, values, pixels
        assert set(gpu_processes()) <= ({os.getpid()} if not cpu else set())
    finally:
        stop.set()
        if not cpu: monitor.join()
    result = dict(method=method, status='STANDARDIZED_TIMING_COMPLETE', runtime=runtime, rows=rows,
        Core_seconds_mean=float(np.mean([r['core_seconds'] for r in rows])),
        E2E_given_inputs_seconds_mean=float(np.mean([r['E2E_given_inputs_seconds'] for r in rows])),
        peak_device_memory_MiB_sampled=None if cpu else max(r['peak_device_memory_MiB_sampled'] for r in rows), memory_sampling_seconds=.2,
        by_K={str(K): {k: dict(mean=float(np.mean([r[k] for r in rows if r['K'] == K])), std=float(np.std([r[k] for r in rows if r['K'] == K], ddof=1))) for k in ['core_seconds', 'E2E_given_inputs_seconds']} for K in c['K']})
    result['FPS_E2E'] = 25 / result['E2E_given_inputs_seconds_mean']
    (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--method', choices=METHODS); parser.add_argument('--all', action='store_true'); parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args(); c = read(ROOT / 'configs/efficiency-track-a.json')
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
        logs = ROOT / 'logs/efficiency-track-a-20260929'; logs.mkdir(parents=True, exist_ok=True)
        for method in METHODS:
            summary = ROOT / c['output'] / method / 'summary.json'
            if summary.exists():
                assert read(summary)['status'] == 'STANDARDIZED_TIMING_COMPLETE'
                assert read(summary.parent / 'protocol.json') == c
                continue
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='' if method == 'deeplens' else '0', OMP_NUM_THREADS='8' if method == 'deeplens' else '4', OPENBLAS_NUM_THREADS='1', TF_ENABLE_ONEDNN_OPTS='0')
            with (logs / f'{method}.log').open('a') as log:
                subprocess.run([python.get(method, '/root/miniconda3/envs/magicbokeh/bin/python'), str(Path(__file__).resolve()), '--method', method], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        result = dict(status='SIX_METHOD_TIMING_COMPLETE_REVIEW_PENDING', protocol=c,
            methods={m: read(ROOT / c['output'] / m / 'summary.json') for m in METHODS})
        from prepare_davis_v2 import write_new
        write_new(ROOT / c['record'], result)
        print('SIX_METHOD_TIMING_COMPLETE', flush=True); return
    assert args.method
    worker(args.method, c)


if __name__ == '__main__': main()
