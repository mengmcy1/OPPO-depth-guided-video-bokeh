"""B36 paired MPI-only invariance and fixed CoC positive control, original ATB.

Run with the existing any2bokeh environment. Large tensors and logs stay in outputs.
All forward hooks return None; the posterior observer calls the original sampler
once with unchanged arguments. No processor, model forward, RNG policy or cache
lifetime is replaced. Independent workers establish fresh native cache state.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / 'configs/any2bokeh-B36-MPI-invariance-20261021.json'
KEYS = ['pixel_values', 'coc', 'k', 'mpi_mask', 'weight']


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def digest(t):
    return hashlib.sha256(t.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def summary(t):
    return dict(shape=list(t.shape), dtype=str(t.dtype), finite=bool(torch.isfinite(t).all()),
                minimum=float(t.min()), maximum=float(t.max()),
                nonzero=int(torch.count_nonzero(t)), elements=t.numel(),
                nonzero_fraction=float(torch.count_nonzero(t)) / t.numel())


def rng_state():
    return dict(cpu=digest(torch.get_rng_state()), cuda=digest(torch.cuda.get_rng_state(0)))


def tensor_record(t):
    return dict(**summary(t), sha256=digest(t))


def author_imports():
    p = read(ROOT / 'configs/davis-baselines.json')['any2bokeh']
    repo = ROOT.parent / 'any-to-bokeh'
    assert subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip() == p['author_commit']
    assert not subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True), 'Preserve and investigate modified author checkout'
    os.environ.setdefault('HF_HOME', str(ROOT / 'models/huggingface'))
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    sys.path.insert(0, str(repo))
    from models.unet import UNetSpatioTemporalConditionModel
    from models.vae import AutoencoderKLTemporalDecoder
    from pipelines.any2bokeh_pipe import StableVideoDiffusionPipeline
    from utils.dataset import BokehCocMPIDatasetWholeVid, collate_fn_val_coc_mpi_whole_video
    return p, UNetSpatioTemporalConditionModel, AutoencoderKLTemporalDecoder, StableVideoDiffusionPipeline, BokehCocMPIDatasetWholeVid, collate_fn_val_coc_mpi_whole_video


def prepare(c, out):
    assert not (out / 'prepared.json').exists(), 'Reuse prepared inputs rather than overwrite'
    p, _, _, Pipeline, Dataset, collate = author_imports()
    prior = read(ROOT / c['prior_config'])
    old = ROOT / prior['output']
    info = read(old / 'prepared.json')
    assert info['config'] == prior and info['target_q'] == 147
    source = ROOT / 'data/processed/davis2016-v2/eval' / c['scene']
    geometry = read(ROOT / 'data/processed/davis2016-v2/sequences' / c['scene'] / 'geometry.json')
    base_ds = Dataset(str(old / 'inputs/native/input.csv'), sample_size=(576, 1024))
    base = collate([base_ds[0]])
    # Existing author-input and identity-decoder indexing checks, no model inference.
    from run_davis_baseline import audit_atb_batch
    audit = audit_atb_batch(base_ds, base, source, geometry, Pipeline)
    changed = collate([Dataset(str(old / 'inputs/head_to_neck/input.csv'), sample_size=(576, 1024))[0]])
    original_q = np.asarray(Image.open(source / 'disparity' / geometry['frames'][0]['file']))
    changed_q = np.asarray(Image.open(old / 'inputs/head_to_neck/disparity' / geometry['frames'][0]['file']))
    mask = np.asarray(Image.open(old / 'head-intervention-native.png')) != 0
    assert int(mask.sum()) == info['head_pixels'] == 14607
    assert np.array_equal(changed_q[~mask], original_q[~mask]) and np.all(changed_q[mask] == 147)
    valid = torch.zeros((6, 8), dtype=torch.bool)
    slots = []
    for g, group in enumerate(audit['input_groups']):
        for j, frame in enumerate(group):
            valid[g, j] = True
            slots.append(dict(group=g, slot=j, source_frame=frame, weight=float(base['weight'][g, j])))
    assert int(valid.sum()) == 45 and int((~valid).sum()) == 3
    for k in ['pixel_values', 'k', 'weight']:
        assert digest(base[k]) == digest(changed[k])
    changed_slots = {}
    for k in ['coc', 'mpi_mask']:
        changed_slots[k] = [[g, j] for g in range(6) for j in range(8) if not torch.equal(base[k][g, j], changed[k][g, j])]
        assert changed_slots[k] == [[0, 0]]
    out.mkdir(parents=True, exist_ok=True)
    torch.save(dict(base={k:base[k] for k in KEYS}, coc_source0=changed['coc'][0, 0].clone(), valid=valid), out / 'prepared-inputs.pt')
    save(out / 'prepared.json', dict(config=c, prior_target_q=147, prior_head_pixels=int(mask.sum()),
        prior_record=c['prior_record'], original_focus_values=[f['focus'] for f in geometry['frames'][:25]],
        audit=audit, slots=slots, real_zero_weight_slots=[s for s in slots if s['weight'] == 0],
        original={k:tensor_record(base[k]) for k in KEYS}, coc_source0=tensor_record(changed['coc'][0, 0]),
        prior_head_to_neck_changed_slots=changed_slots, rois=info['rois'],
        note='Native batch cached on CPU once. D transplants only source0 CoC; original MPI retained. No focus recomputation.'))
    print('PREPARED', out, flush=True)


def load_model():
    p, UNet, VAE, Pipeline, _, _ = author_imports()
    torch.manual_seed(0)
    cp = ROOT / p['checkpoint']
    unet = UNet.from_pretrained(str(cp), subfolder='unet', torch_dtype=torch.float16, low_cpu_mem_usage=False, local_files_only=True)
    vae = VAE.from_pretrained(str(cp / 'vae'), torch_dtype=torch.float16, local_files_only=True)
    pipe = Pipeline.from_pretrained('stabilityai/stable-video-diffusion-img2vid-xt', unet=unet, vae=vae,
        torch_dtype=torch.float16, variant='fp16', low_cpu_mem_usage=False, local_files_only=True).to('cuda:0')
    pipe.set_progress_bar_config(disable=True)
    return pipe, p


def worker(c, out, condition):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
    dest = out / condition
    assert not dest.exists(), 'Preserve partial outputs; use an explicit new run directory after investigation'
    dest.mkdir(parents=True)
    pipe, p = load_model()
    prepared = read(out / 'prepared.json')
    assert prepared['config'] == c
    loaded = torch.load(out / 'prepared-inputs.pt', weights_only=True, map_location='cpu')
    base, valid = loaded['base'], loaded['valid']
    assert {k:digest(v) for k,v in base.items()} == {k:v['sha256'] for k,v in prepared['original'].items()}
    batch = dict(base)
    if condition in ['mpi_zero', 'mpi_three']:
        batch['mpi_mask'] = base['mpi_mask'].clone()
        batch['mpi_mask'][valid] = 0 if condition == 'mpi_zero' else 3
    elif condition == 'coc_only':
        batch['coc'] = base['coc'].clone()
        for s in prepared['slots']:
            if s['source_frame'] == c['source_frame']:
                batch['coc'][s['group'], s['slot']] = loaded['coc_source0']
    input_check = {k:dict(**tensor_record(batch[k]), bitwise_equal_to_A=digest(batch[k]) == digest(base[k])) for k in KEYS}
    assert all(input_check[k]['finite'] for k in KEYS)
    assert all(input_check[k]['bitwise_equal_to_A'] for k in ['pixel_values', 'k', 'weight'])
    changed_slots = {k:[[g,j] for g in range(6) for j in range(8) if not torch.equal(batch[k][g,j], base[k][g,j])] for k in ['coc', 'mpi_mask']}
    assert all(torch.equal(batch[k][~valid], base[k][~valid]) for k in KEYS)
    if condition in ['mpi_zero', 'mpi_three']:
        assert input_check['coc']['bitwise_equal_to_A'] and changed_slots['mpi_mask']
        assert bool((batch['mpi_mask'][valid] == (0 if condition == 'mpi_zero' else 3)).all())
        assert all(valid[g,j] for g,j in changed_slots['mpi_mask'])
    else:
        assert input_check['mpi_mask']['bitwise_equal_to_A']
        assert changed_slots['coc'] == ([[0,0]] if condition == 'coc_only' else [])
    save(dest / 'input-checks.json', dict(tensors=input_check, changed_slots=changed_slots,
        padding_bitwise_unchanged=True, effective_slots=45, padding_slots=3,
        real_zero_weight_slots_included=prepared['real_zero_weight_slots']))
    legacy = ROOT / 'outputs/davis-baselines-20260929/any2bokeh'
    order = read(legacy / 'protocol.json')['scene_order']
    index = order.index(c['scene'])
    rng_path = legacy / 'audit' / order[index-1] / 'rng-after.pt'
    state = torch.load(rng_path, weights_only=True)
    assert state['next_scene_index'] == index == 33
    assert not hasattr(pipe.vae.encoder, 'residual_down_blocks'), 'Fresh encoder cache required'
    assert not hasattr(pipe.vae, 'encoder_residual'), 'Fresh decoder cache required'
    torch.set_rng_state(state['cpu'])
    torch.cuda.set_rng_state(state['cuda'], device=0)
    entry_rng = rng_state()
    events = dict(unet=[], fuser=[], key_mask=[], posterior=[], units=[])
    current = dict(unit=None, n_visual={})
    handles = []
    from models.attention import MPIAttention
    def unet_hook(module, args, kwargs):
        events['unet'].append(dict(unit=current['unit'], **summary(kwargs['mpi_mask']), sha256=digest(kwargs['mpi_mask'])))
    handles.append(pipe.unet.register_forward_pre_hook(unet_hook, with_kwargs=True))
    fusers = {}
    for name, module in pipe.unet.named_modules():
        if not isinstance(module, MPIAttention):
            continue
        fusers[name] = dict(enabled=module.enabled, processor=type(module.mpi_attn.processor).__module__+'.'+type(module.mpi_attn.processor).__name__,
            processor_source=inspect.getfile(type(module.mpi_attn.processor)), forward_source=inspect.getfile(type(module)))
        def fuser_hook(mod, args, kwargs, name=name):
            mask = args[2] if len(args) > 2 else kwargs['mpi_mask']
            x = args[0] if args else kwargs['x']
            current['n_visual'][name] = x.shape[1]
            events['fuser'].append(dict(unit=current['unit'], module=name, **summary(mask)))
        def key_hook(mod, args, kwargs, name=name):
            mask = kwargs['attention_mask']
            n = current['n_visual'][name]
            assert mask.dtype == torch.bool and mask.shape[-1] == 2*n
            events['key_mask'].append(dict(unit=current['unit'], module=name, shape=list(mask.shape), dtype=str(mask.dtype),
                main_tokens=n, appended_RGB_tokens=n, main_allowed_fraction=float(mask[...,:n].float().mean()),
                appended_RGB_allowed_fraction=float(mask[...,n:].float().mean())))
        handles.append(module.register_forward_pre_hook(fuser_hook, with_kwargs=True))
        handles.append(module.mpi_attn.register_forward_pre_hook(key_hook, with_kwargs=True))
    assert fusers
    from models.vae import DiagonalGaussianDistribution
    original_encode = pipe.vae.encode
    original_sample = DiagonalGaussianDistribution.sample
    def observe_encode(*args, **kwargs):
        current['branch'] = 'RGB' if kwargs.get('cache_flag', False) else 'CoC'
        return original_encode(*args, **kwargs)
    def observe_sample(posterior, *sargs, **skwargs):
        generator = skwargs.get('generator', sargs[0] if sargs else None)
        before = rng_state()
        generator_before = None if generator is None else digest(generator.get_state())
        value = original_sample(posterior, *sargs, **skwargs)
        events['posterior'].append(dict(unit=current['unit'], branch=current['branch'], shape=list(value.shape), dtype=str(value.dtype),
            generator='global/default' if generator is None else str(generator.device),
            generator_before=generator_before, generator_after=None if generator is None else digest(generator.get_state()),
            rng_before=before, rng_after=rng_state(), sample_sha256=digest(value)))
        return value
    pipe.vae.encode = observe_encode
    # A process-local observer, restored below. No per-posterior closures or GPU
    # tensor references are retained; the original sampler is called exactly once.
    DiagonalGaussianDistribution.sample = observe_sample
    import diffusers
    runtime = dict(python=sys.version, executable=sys.executable, torch=torch.__version__, cuda=torch.version.cuda,
        diffusers=diffusers.__version__, device=torch.cuda.get_device_name(0), visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
        author_commit=p['author_commit'], project_parent_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        imported_sources={name:inspect.getfile(type(obj)) for name,obj in [('pipeline',pipe),('unet',pipe.unet),('vae',pipe.vae)]},
        fusers=fusers, RNG_source=str(rng_path.relative_to(ROOT)), scene_index=index, entry_RNG=entry_rng,
        fresh_encoder_cache_absent=True, initial_latent_cache=None,
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_tf32=torch.backends.cudnn.allow_tf32,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        allocator_backend=torch.cuda.memory.get_allocator_backend(),
        allocator_environment={k:v for k,v in os.environ.items() if k.startswith(('PYTORCH_ALLOC','PYTORCH_CUDA_ALLOC'))})
    save(dest / 'runtime.json', runtime)
    floats = np.lib.format.open_memmap(dest / 'float.npy', mode='w+', dtype=np.float32, shape=(25,576,1024,3))
    (dest / 'png').mkdir()
    cache = None
    offset = 0
    start = time.monotonic()
    for unit, i in enumerate([0, 2, 4]):
        current['unit'] = unit
        tensors = {k:batch[k][i:i+2].to('cuda', dtype=torch.float32 if k in ['pixel_values','coc'] else torch.float16) for k in KEYS}
        generator = torch.Generator(device='cpu').manual_seed(0)
        unit_info = dict(unit=unit, groups=[i,i+1], source_groups=prepared['audit']['input_groups'][i:i+2],
            rng_before=rng_state(), generator_seed=generator.initial_seed(), generator_before=digest(generator.get_state()),
            latent_cache_in=None if cache is None else tensor_record(cache),
            RGB_cache_in_shapes=[list(t.shape) for t in getattr(pipe.vae.encoder,'residual_down_blocks',[])],
            GPU_inputs={k:tensor_record(v) for k,v in tensors.items()})
        with torch.inference_mode():
            result, cache = pipe(tensors['coc'], tensors['pixel_values'], input_k=tensors['k'], mpi_mask=tensors['mpi_mask'],
                weight=tensors['weight'], latent_cache=cache, is_last_batch=i==4, overlap_frame=4, height=576, width=1024,
                num_frames=8, decode_chunk_size=8, motion_bucket_id=127, fps=7, noise_aug_strength=0., num_inference_steps=1,
                generator=generator, batch_size=tensors['weight'].shape[0], output_type='pt')
        assert result[0].shape == ((9 if i==4 else 8),3,576,1024) and bool(torch.isfinite(result[0]).all())
        arr = result[0].detach().cpu().permute(0,2,3,1).numpy()
        floats[offset:offset+len(arr)] = arr
        for j, frame in enumerate(arr):
            Image.fromarray(np.uint8(np.round(np.clip(frame,0,1)*255))).save(dest / 'png' / f'{offset+j:05d}.png')
        unit_info.update(rng_after=rng_state(), generator_after=digest(generator.get_state()),
            latent_cache_out=tensor_record(cache), output_frames=len(arr))
        events['units'].append(unit_info)
        offset += len(arr)
        print('CHUNK', condition, unit, offset, flush=True)
        del result, tensors, arr
    assert offset == 25
    floats.flush()
    for handle in handles:
        handle.remove()
    pipe.vae.encode = original_encode
    DiagonalGaussianDistribution.sample = original_sample
    save(dest / 'events.json', events)
    save(dest / 'complete.json', dict(condition=condition, shape=list(floats.shape), dtype=str(floats.dtype), finite=bool(np.isfinite(floats).all()),
        frame_count=25, png_count=len(list((dest/'png').glob('*.png'))), end_RNG=rng_state(),
        seconds_including_observation_and_save=time.monotonic()-start, efficiency_measurement=False))
    print('DONE', condition, flush=True)


def compare(out, condition):
    a = np.load(out / 'native/float.npy', mmap_mode='r')
    b = np.load(out / condition / 'float.npy', mmap_mode='r')
    assert a.shape == b.shape == (25,576,1024,3) and a.dtype == b.dtype == np.float32
    rows = []
    for t in range(25):
        delta = np.abs(a[t].astype(np.float64)-b[t].astype(np.float64))
        pa = np.asarray(Image.open(out/'native/png'/f'{t:05d}.png'))
        pb = np.asarray(Image.open(out/condition/'png'/f'{t:05d}.png'))
        assert pa.shape == pb.shape == (576,1024,3) and np.isfinite(b[t]).all()
        assert np.array_equal(pb, np.uint8(np.round(np.clip(b[t],0,1)*255)))
        rows.append(dict(frame=t, float_max_abs_diff=float(delta.max()), float_mean_abs_diff=float(delta.mean()),
            different_float_elements=int(np.count_nonzero(a[t] != b[t])),
            different_float_bit_patterns=int(np.count_nonzero(a[t].view(np.uint32) != b[t].view(np.uint32))),
            RGB8_different_bytes=int(np.count_nonzero(pa != pb))))
    return dict(comparison=condition+' vs native', float_max_abs_diff=max(r['float_max_abs_diff'] for r in rows),
        float_mean_abs_diff=float(np.mean([r['float_mean_abs_diff'] for r in rows])),
        different_float_elements=sum(r['different_float_elements'] for r in rows),
        different_float_bit_patterns=sum(r['different_float_bit_patterns'] for r in rows),
        RGB8_different_bytes=sum(r['RGB8_different_bytes'] for r in rows),
        affected_float_frames=[r['frame'] for r in rows if r['different_float_elements']],
        affected_RGB8_frames=[r['frame'] for r in rows if r['RGB8_different_bytes']], per_frame=rows)


def analyze(c, out):
    prepared = read(out/'prepared.json')
    runs = {v:dict(complete=read(out/v/'complete.json'),runtime=read(out/v/'runtime.json'),
        input_checks=read(out/v/'input-checks.json'),events=read(out/v/'events.json')) for v in c['conditions']}
    comparisons = [compare(out,v) for v in c['conditions'][1:]]
    a = runs['native']
    paths = []
    for v, r in runs.items():
        assert r['runtime']['entry_RNG'] == a['runtime']['entry_RNG']
        assert r['complete']['end_RNG'] == a['complete']['end_RNG']
        assert [x['output_frames'] for x in r['events']['units']] == [8,8,9]
        assert len(r['events']['posterior']) == 6 and len(r['events']['unet']) == 3
        for actual, ref in zip(r['events']['posterior'], a['events']['posterior']):
            for field in ['unit','branch','shape','dtype','generator','rng_before','rng_after']:
                assert actual[field] == ref[field], (v,field)
            if v != 'coc_only' or actual['branch'] == 'RGB' or actual['unit'] != 0:
                assert actual['sample_sha256'] == ref['sample_sha256']
        for actual, ref in zip(r['events']['units'], a['events']['units']):
            for field in ['groups','source_groups','rng_before','rng_after','generator_seed','generator_before','generator_after','RGB_cache_in_shapes']:
                assert actual[field] == ref[field], (v,field)
            assert actual['generator_before'] == actual['generator_after']  # current forward does not consume this generator
            if v != 'coc_only':
                assert actual['latent_cache_in'] == ref['latent_cache_in']
                assert actual['latent_cache_out'] == ref['latent_cache_out']
            else:
                assert actual['unit'] != 0 or actual['latent_cache_in'] is None
            for key in KEYS:
                av, rv = actual['GPU_inputs'][key], ref['GPU_inputs'][key]
                assert av['shape'] == rv['shape'] and av['dtype'] == rv['dtype'] and av['finite']
                may_change = (v in ['mpi_zero','mpi_three'] and key=='mpi_mask') or (v=='coc_only' and key=='coc' and actual['unit']==0)
                if not may_change:
                    assert av['sha256'] == rv['sha256'], (v,actual['unit'],key)
        fusers = r['events']['fuser']; keys = r['events']['key_mask']
        assert len(fusers) == len(keys) == 3*len(r['runtime']['fusers'])
        paths.append(dict(condition=v, UNet_entry_MPI=r['events']['unet'], fuser_calls=len(fusers),
            fuser_calls_by_stage={stage:sum(x['module'].startswith(stage) for x in fusers) for stage in ['down_blocks','mid_block','up_blocks']},
            all_fuser_masks_zero=all(x['nonzero']==0 for x in fusers),
            main_token_allowed_fractions=sorted(set(x['main_allowed_fraction'] for x in keys)),
            appended_RGB_token_allowed_fractions=sorted(set(x['appended_RGB_allowed_fraction'] for x in keys)),
            fixed_input_checks_passed=True, GPU_cast_input_checks_passed=True, RNG_and_posterior_sequence_match_A=True, initial_cache_matches_A=True,
            latent_cache_sequence_matches_A=all(x['latent_cache_in']==y['latent_cache_in'] and x['latent_cache_out']==y['latent_cache_out']
                for x,y in zip(r['events']['units'],a['events']['units']))))
    assert comparisons[0]['different_float_bit_patterns'] == 0, 'Repeat unstable; investigate before interpreting contrasts'
    result = dict(config=c, prepared=prepared, execution_paths=paths, comparisons=comparisons, runs=runs,
        limits=['One exploration video and one fixed source-frame CoC intervention; no claim about all ATB versions or private implementation.',
            'No new visual quality judgments; output differences measure condition response, not quality or physical correctness.',
            'No performance result; hooks synchronize/copy metadata and latent samples to CPU.',
            'Existing historical CoC+MPI intervention is preserved; these locally paired outputs alone determine the present contrasts.'])
    split = read(ROOT/'03_实验记录/records/any2bokeh-failure-analysis-20261009-source-split.json')
    source_rows = [x for x in split['rows'] if x['dataset']=='DAVIS' and x['scene_id']==c['scene']]
    assert len(source_rows)==1 and source_rows[0]['split']=='exploration'
    result['source_split'] = source_rows[0]
    launch = read(out/'launch.json')
    result['reproducibility'] = dict(launch=launch,
        executed_script_copy=str((out/'executed-check_atb_mpi_invariance.py').relative_to(ROOT)),
        analysis_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        environment='Existing any2bokeh environment; CUDA_VISIBLE_DEVICES=0; no dependency/allocator/precision changes.',
        commands=[
            '/root/miniconda3/envs/any2bokeh/bin/python scripts/check_atb_mpi_invariance.py prepare',
            'CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 /root/miniconda3/envs/any2bokeh/bin/python scripts/check_atb_mpi_invariance.py run',
            '/root/miniconda3/envs/any2bokeh/bin/python scripts/check_atb_mpi_invariance.py analyze'],
        resume='run reuses completed conditions. Incomplete worker directories are preserved and cause a stop; investigate and use an explicit new run path rather than overwrite.',
        sampling='Six original posterior sample calls: RGB then CoC in each unit, shape16x4x72x128. No explicit sampler generator; global CUDA states agree before/after every matched sample. The original per-call CPU generator0 is preserved and its state is unchanged by this forward.',
        cache='Five fresh interpreter/model processes; encoder cache absent and latent_cache=None at first call. Subsequent native cache progression recorded without clearing or extra tensor references.',
        instrumentation='Read-only pre-hooks for UNet/fuser/attention. Process-local observers call original encode/sample exactly once and return original values; restored after run. Metadata/CPU hashes only, no retained GPU tensors.',
        raw_logs=[str((out/f'{v}.log').relative_to(ROOT)) for v in c['conditions']],
        parent_log='logs/B36-MPI-invariance-20261021.log')
    assert hashlib.sha256((out/'executed-check_atb_mpi_invariance.py').read_bytes()).hexdigest()==launch['script_sha256']
    result['anomalies_and_recovery'] = dict(failed_GPU_conditions=[],recovery_runs=[],
        warnings='Existing transformers/diffusers deprecated pytree warnings retained in raw logs; no dependency change.',
        unresolved_execution_anomalies=[],
        post_run_changes='Only CPU analysis checks, provenance and page metadata were supplemented after inference. No worker rerun or inference logic change.')
    regions = []
    aa = np.load(out/'native/float.npy',mmap_mode='r'); dd = np.load(out/'coc_only/float.npy',mmap_mode='r')
    for t in range(25):
        for name,(x,y,w,h) in prepared['rois'].items():
            delta=np.abs(aa[t,y:y+h,x:x+w].astype(np.float64)-dd[t,y:y+h,x:x+w].astype(np.float64))
            qa=np.asarray(Image.open(out/'native/png'/f'{t:05d}.png'))[y:y+h,x:x+w]
            qd=np.asarray(Image.open(out/'coc_only/png'/f'{t:05d}.png'))[y:y+h,x:x+w]
            regions.append(dict(frame=t,region=name,display_xywh=[x,y,w,h],float_max_abs_diff=float(delta.max()),
                float_mean_abs_diff=float(delta.mean()),RGB8_mean_abs_diff=float(np.abs(qa.astype(float)-qd.astype(float)).mean()),
                RGB8_different_bytes=int(np.count_nonzero(qa!=qd)),
                location_limit='Fixed historical frame0 ROI; later-frame values are fixed-image-coordinate response, not tracked anatomy.'))
    result['D_fixed_ROI_response'] = regions
    clean = all(x['different_float_bit_patterns']==0 for x in comparisons[:3]) and all(x['all_fuser_masks_zero'] and x['main_token_allowed_fractions']==[1.] and x['appended_RGB_token_allowed_fractions']==[0.] for x in paths)
    positive = comparisons[3]['different_float_elements'] > 0
    result['decision'] = dict(MPI_invariant=clean,CoC_response=positive,ready_for_next_full25_local_correction_plan=clean and positive,
        observation='A-repeat/B/C bitwise identical to A; all actual fuser masks zero, appended RGB keys forbidden; D responds.' if clean and positive else 'Unexpected result; investigate execution before causal interpretation.',
        explanation='In this pinned public version and tested input, passed MPI partition labels do not affect output. MPIAttention still executes main-token attention, K embedding and feed-forward; CoC path is active.' if clean and positive else 'Execution evidence needs investigation; do not claim an active/inactive causal path yet.',
        next_step='B36 full25-frame local-correction validation can be planned with fixed per-frame support/reference and visual review; not started by this run.' if clean and positive else 'Investigate versions, processor, RNG/cache or intervention placement; no expanded intervention.')
    verification_path=out/'verification.json'
    if verification_path.exists():
        result['verification']=read(verification_path)
    save(ROOT/c['record'],result)
    # Faithful CPU crops; no enhancement, visual acceptance remains with the user.
    rgbdir=ROOT/'outputs/any2bokeh-failure-analysis-20261009/B1-batch2-20261014/B36/originals/Input_RGB'
    html=['<!doctype html><meta charset="utf-8"><title>B36 MPI-only / CoC-only</title><style>body{font-family:system-ui;margin:24px;background:#f6f9ff}section{display:flex;flex-wrap:wrap;gap:16px}figure{margin:0}img{image-rendering:pixelated}</style><h1>B36：A原版与D仅改CoC</h1><p>仅源帧0头部使用既定颈部参考值，MPI/focus/K保持原版。下面是原PNG与最近邻放大；数值变化不是质量评分，观感待用户复核。24→0是循环回到开头，不是连续transition。</p>']
    for t in [0,1,7,8,24]:
        html.append(f'<h2>帧 {t}</h2><section>')
        for name,path in [('Input',rgbdir/f'{t:05d}.png'),('A',out/'native/png'/f'{t:05d}.png'),('D',out/'coc_only/png'/f'{t:05d}.png')]:
            target=out/f'{name}-{t:05d}-head.png'
            with Image.open(path) as im:im.crop((632,128,760,288)).save(target)
            full=os.path.relpath(path,out)
            html.append(f'<figure><figcaption>{name} / 固定历史裁剪 [632,128,128,160]</figcaption><a href="{full}">完整尺寸PNG</a><br><a href="{target.name}"><img src="{target.name}" width="256" height="320"></a></figure>')
        html.append('</section>')
    for v in c['conditions']:
        html.append(f'<details><summary>{v}：完整25帧原PNG</summary>'+ ' '.join(f'<a href="{v}/png/{t:05d}.png">{t}</a>' for t in range(25))+'</details>')
    (out/'index.html').write_text(''.join(html))
    print('RESULT', ROOT/c['record'], result['decision'], flush=True)


def run(c, out):
    if not (out/'prepared.json').exists():
        prepare(c,out)
    assert read(out/'prepared.json')['config']==c
    for v in c['conditions']:
        if (out/v/'complete.json').exists():
            print('REUSE_COMPLETE',v,flush=True)
        else:
            apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            assert not apps, 'Another GPU process is present; do not stop it'
            log=out/f'{v}.log'
            with log.open('xb') as f:
                subprocess.run([sys.executable,str(Path(__file__).resolve()),'worker','--condition',v],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
        if v=='native_repeat':
            assert compare(out,v)['different_float_bit_patterns']==0, 'Stop: repeat differs from A'
        if v in ['mpi_zero','mpi_three']:
            assert compare(out,v)['different_float_bit_patterns']==0, 'Stop: MPI contrast differs from A; investigate first'
    analyze(c,out)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','worker','analyze','run'])
    parser.add_argument('--condition',choices=['native','native_repeat','mpi_zero','mpi_three','coc_only'])
    args=parser.parse_args()
    torch.set_num_threads(4)
    c=read(CFG);out=ROOT/c['output']
    if args.stage=='worker':
        assert args.condition is not None
        worker(c,out,args.condition)
    else:
        globals()[args.stage](c,out)
